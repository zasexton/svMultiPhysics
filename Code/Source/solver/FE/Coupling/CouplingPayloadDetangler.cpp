/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingPayloadDetangler.h"

#include "Analysis/FormExprScanner.h"
#include "Core/FEException.h"

#include <algorithm>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

namespace forms = svmp::FE::forms;

struct PayloadAnalysis {
    CouplingPayloadKind selected_kind{CouplingPayloadKind::CoefficientExpression};
    CouplingPayloadExtractionReason reason{CouplingPayloadExtractionReason::Exact};
    bool exact{true};
    forms::FormExpr payload_expression;
};

std::string defaultProducerPortName(std::string_view exchange_name)
{
    return std::string(exchange_name) + ".producer";
}

std::string defaultConsumerPortName(std::string_view exchange_name)
{
    return std::string(exchange_name) + ".consumer";
}

std::string providerNameFor(const CouplingPayloadExtractionRequest& request)
{
    return request.exchange_name + ".forms_payload";
}

const CouplingFormContribution* findContribution(
    std::span<const CouplingFormContribution> contributions,
    std::string_view contribution_name)
{
    const auto it = std::find_if(
        contributions.begin(),
        contributions.end(),
        [contribution_name](const CouplingFormContribution& contribution) {
            return contribution.contribution_name == contribution_name;
        });
    return it == contributions.end() ? nullptr : &*it;
}

std::optional<FieldId> fieldIdFor(const CouplingContext& ctx,
                                  std::string_view participant_name,
                                  std::string_view field_name)
{
    if (!ctx.hasField(participant_name, field_name)) {
        return std::nullopt;
    }
    return ctx.field(participant_name, field_name).field_id;
}

std::optional<std::string_view> participantForFieldId(
    const CouplingContext& ctx,
    FieldId field_id)
{
    const auto& fields = ctx.fields();
    const auto it = std::find_if(
        fields.begin(),
        fields.end(),
        [field_id](const CouplingFieldRef& field) {
            return field.field_id == field_id;
        });
    if (it == fields.end()) {
        return std::nullopt;
    }
    return std::string_view(it->participant_name);
}

bool nodeHasParticipantField(const CouplingContext& ctx,
                             const forms::FormExprNode& node,
                             std::string_view participant_name)
{
    using FT = forms::FormExprType;
    if (node.type() == FT::StateField ||
        node.type() == FT::DiscreteField ||
        node.type() == FT::PreviousSolutionRef ||
        node.type() == FT::TimeDerivative) {
        if (const auto field_id = node.fieldId()) {
            const auto participant = participantForFieldId(ctx, *field_id);
            if (participant.has_value() && *participant == participant_name) {
                return true;
            }
        }
    }
    for (const auto& child : node.childrenShared()) {
        if (child && nodeHasParticipantField(ctx, *child, participant_name)) {
            return true;
        }
    }
    return false;
}

bool isTargetTest(const forms::FormExprNode& node, FieldId target_field)
{
    if (node.type() != forms::FormExprType::TestFunction) {
        return false;
    }
    const auto field_id = node.fieldId();
    return field_id.has_value() && *field_id == target_field;
}

bool isTargetTestFactor(const forms::FormExprNode& node, FieldId target_field)
{
    if (isTargetTest(node, target_field)) {
        return true;
    }
    if (node.type() == forms::FormExprType::RestrictMinus ||
        node.type() == forms::FormExprType::RestrictPlus) {
        const auto kids = node.childrenShared();
        return kids.size() == 1u && kids[0] &&
               isTargetTest(*kids[0], target_field);
    }
    return false;
}

bool containsTargetTest(const forms::FormExprNode& node, FieldId target_field)
{
    if (isTargetTest(node, target_field)) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsTargetTest(*child, target_field)) {
            return true;
        }
    }
    return false;
}

bool containsAnyTest(const forms::FormExprNode& node)
{
    if (node.type() == forms::FormExprType::TestFunction) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsAnyTest(*child)) {
            return true;
        }
    }
    return false;
}

int countTargetTests(const forms::FormExprNode& node, FieldId target_field)
{
    int count = isTargetTest(node, target_field) ? 1 : 0;
    for (const auto& child : node.childrenShared()) {
        if (child) {
            count += countTargetTests(*child, target_field);
        }
    }
    return count;
}

int countAllTests(const forms::FormExprNode& node)
{
    int count = node.type() == forms::FormExprType::TestFunction ? 1 : 0;
    for (const auto& child : node.childrenShared()) {
        if (child) {
            count += countAllTests(*child);
        }
    }
    return count;
}

bool hasTargetTestUnderTraceOrGradient(const forms::FormExprNode& node,
                                       FieldId target_field,
                                       bool trace_or_gradient_context = false)
{
    using FT = forms::FormExprType;
    const bool next_context =
        trace_or_gradient_context ||
        node.type() == FT::Gradient ||
        node.type() == FT::Divergence ||
        node.type() == FT::Curl ||
        node.type() == FT::Hessian ||
        node.type() == FT::Jump ||
        node.type() == FT::Average;
    if (next_context && isTargetTest(node, target_field)) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child &&
            hasTargetTestUnderTraceOrGradient(*child,
                                              target_field,
                                              next_context)) {
            return true;
        }
    }
    return false;
}

bool hasProductWithTestOnBothSides(const forms::FormExprNode& node,
                                   FieldId target_field)
{
    using FT = forms::FormExprType;
    const auto kids = node.childrenShared();
    if ((node.type() == FT::Multiply ||
         node.type() == FT::Divide ||
         node.type() == FT::InnerProduct ||
         node.type() == FT::DoubleContraction ||
         node.type() == FT::OuterProduct) &&
        kids.size() >= 2u &&
        kids[0] &&
        kids[1]) {
        if (containsTargetTest(*kids[0], target_field) &&
            containsTargetTest(*kids[1], target_field)) {
            return true;
        }
    }
    for (const auto& child : kids) {
        if (child && hasProductWithTestOnBothSides(*child, target_field)) {
            return true;
        }
    }
    return false;
}

bool hasOtherTest(const forms::FormExprNode& node, FieldId target_field)
{
    if (node.type() == forms::FormExprType::TestFunction) {
        const auto field_id = node.fieldId();
        if (!field_id.has_value() || *field_id != target_field) {
            return true;
        }
    }
    for (const auto& child : node.childrenShared()) {
        if (child && hasOtherTest(*child, target_field)) {
            return true;
        }
    }
    return false;
}

std::optional<forms::FormExpr> extractCoefficient(
    const forms::FormExpr& expr,
    FieldId target_field);

std::optional<forms::FormExpr> extractFromNodeChildren(
    const forms::FormExprNode& node,
    FieldId target_field,
    bool subtract)
{
    const auto kids = node.childrenShared();
    if (kids.size() != 2u || !kids[0] || !kids[1]) {
        return std::nullopt;
    }
    auto lhs = extractCoefficient(forms::FormExpr(kids[0]), target_field);
    auto rhs = extractCoefficient(forms::FormExpr(kids[1]), target_field);
    if (!lhs.has_value() || !rhs.has_value()) {
        return std::nullopt;
    }
    if (subtract) {
        return *lhs - *rhs;
    }
    return *lhs + *rhs;
}

std::optional<forms::FormExpr> extractCoefficient(
    const forms::FormExpr& expr,
    FieldId target_field)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return std::nullopt;
    }
    using FT = forms::FormExprType;
    const auto& node = *expr.node();
    const auto kids = node.childrenShared();
    switch (node.type()) {
    case FT::CellIntegral:
    case FT::BoundaryIntegral:
    case FT::InteriorFaceIntegral:
    case FT::InterfaceIntegral:
        if (kids.size() == 1u && kids[0]) {
            return extractCoefficient(forms::FormExpr(kids[0]), target_field);
        }
        return std::nullopt;
    case FT::Negate:
        if (kids.size() == 1u && kids[0]) {
            if (auto inner = extractCoefficient(forms::FormExpr(kids[0]),
                                                target_field)) {
                return -*inner;
            }
        }
        return std::nullopt;
    case FT::Add:
        return extractFromNodeChildren(node, target_field, false);
    case FT::Subtract:
        return extractFromNodeChildren(node, target_field, true);
    case FT::InnerProduct:
    case FT::Multiply:
        if (kids.size() == 2u && kids[0] && kids[1]) {
            const bool lhs_is_test = isTargetTestFactor(*kids[0], target_field);
            const bool rhs_is_test = isTargetTestFactor(*kids[1], target_field);
            if (lhs_is_test && !containsAnyTest(*kids[1])) {
                return forms::FormExpr(kids[1]);
            }
            if (rhs_is_test && !containsAnyTest(*kids[0])) {
                return forms::FormExpr(kids[0]);
            }
        }
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

CouplingPayloadKind fallbackKindFor(CouplingPayloadExtractionReason reason,
                                    CouplingPayloadFallbackPolicy policy)
{
    if (reason == CouplingPayloadExtractionReason::ConstraintResidualNotLoad) {
        return CouplingPayloadKind::ConstraintResidual;
    }
    switch (policy) {
    case CouplingPayloadFallbackPolicy::Error:
        return CouplingPayloadKind::DriverOwned;
    case CouplingPayloadFallbackPolicy::WarnAndUseDualResidual:
        return CouplingPayloadKind::DualResidualVector;
    case CouplingPayloadFallbackPolicy::WarnAndUseResidualRecipe:
        return CouplingPayloadKind::ResidualRecipe;
    case CouplingPayloadFallbackPolicy::WarnAndSplitSymmetric:
        return CouplingPayloadKind::ResidualRecipe;
    case CouplingPayloadFallbackPolicy::WarnAndUseConstraintResidual:
        return CouplingPayloadKind::ConstraintResidual;
    case CouplingPayloadFallbackPolicy::WarnAndUseDriverOwned:
        return CouplingPayloadKind::DriverOwned;
    }
    return CouplingPayloadKind::ResidualRecipe;
}

std::string diagnosticMessage(const CouplingPayloadExtractionRequest& request,
                              CouplingPayloadKind selected_kind,
                              CouplingPayloadExtractionReason reason)
{
    if (reason == CouplingPayloadExtractionReason::Exact) {
        return {};
    }
    std::ostringstream out;
    out << "Coupling payload extraction for '" << request.exchange_name
        << "' from contribution '" << request.contribution_name
        << "' could not produce unique '"
        << toString(request.preferred_kind)
        << "': " << toString(reason)
        << ". Falling back to '" << toString(selected_kind)
        << "' using policy '" << toString(request.fallback_policy) << "'.";
    if (selected_kind == CouplingPayloadKind::ResidualRecipe ||
        selected_kind == CouplingPayloadKind::DualResidualVector) {
        out << " Partitioned equivalence requires fixed-point convergence of the same interface residual.";
    } else if (selected_kind == CouplingPayloadKind::ConstraintResidual) {
        out << " The payload is a constraint residual, not a transferable physical load.";
    } else if (selected_kind == CouplingPayloadKind::DriverOwned) {
        out << " Runtime data ownership is delegated to the coupling driver.";
    }
    return out.str();
}

PayloadAnalysis analyzePayload(const CouplingContext& ctx,
                               const CouplingFormContribution& contribution,
                               const CouplingPayloadExtractionRequest& request)
{
    PayloadAnalysis analysis;
    const auto consumer_field =
        fieldIdFor(ctx, request.consumer_participant_name,
                   request.consumer_field_name);
    if (!consumer_field.has_value() ||
        !contribution.residual.isValid() ||
        contribution.residual.node() == nullptr) {
        analysis.reason = CouplingPayloadExtractionReason::NoConsumerTest;
        analysis.exact = false;
        analysis.selected_kind =
            fallbackKindFor(analysis.reason, request.fallback_policy);
        return analysis;
    }

    const auto& root = *contribution.residual.node();
    const auto scan = analysis::scanFormExpr(root);
    const bool has_target_test = containsTargetTest(root, *consumer_field);
    const bool has_other_test = hasOtherTest(root, *consumer_field);
    const bool producer_deps =
        nodeHasParticipantField(ctx, root, request.producer_participant_name);
    const bool consumer_deps =
        nodeHasParticipantField(ctx, root, request.consumer_participant_name);
    const bool stabilized_trace =
        scan.has_jump || scan.has_average ||
        scan.has_interior_face_integral ||
        hasTargetTestUnderTraceOrGradient(root, *consumer_field);

    if (!has_target_test) {
        analysis.reason = CouplingPayloadExtractionReason::NoConsumerTest;
    } else if (hasProductWithTestOnBothSides(root, *consumer_field)) {
        analysis.reason = CouplingPayloadExtractionReason::NonlinearInConsumerTest;
    } else if (has_other_test) {
        analysis.reason = CouplingPayloadExtractionReason::SymmetricWeakEnforcement;
    } else if (stabilized_trace) {
        analysis.reason = CouplingPayloadExtractionReason::StabilizedTraceOperator;
    } else if (producer_deps && consumer_deps) {
        analysis.reason = CouplingPayloadExtractionReason::BothSideStateDependency;
    } else if (countAllTests(root) > countTargetTests(root, *consumer_field)) {
        analysis.reason = CouplingPayloadExtractionReason::MultipleConsumerTestsInTerm;
    } else if (!producer_deps && consumer_deps) {
        analysis.reason = CouplingPayloadExtractionReason::ConstraintResidualNotLoad;
    } else if (auto coefficient =
                   extractCoefficient(contribution.residual, *consumer_field)) {
        analysis.reason = CouplingPayloadExtractionReason::Exact;
        analysis.selected_kind = request.preferred_kind;
        analysis.exact = true;
        analysis.payload_expression = std::move(*coefficient);
        return analysis;
    } else {
        analysis.reason = CouplingPayloadExtractionReason::UnsupportedRuntimeProvider;
    }

    analysis.exact = false;
    analysis.selected_kind = fallbackKindFor(analysis.reason,
                                            request.fallback_policy);
    analysis.payload_expression = contribution.residual;
    return analysis;
}

CouplingEndpointRef producerEndpointFor(
    const CouplingPayloadExtractionRequest& request,
    const CouplingPartitionedPayloadMetadata& metadata)
{
    if (metadata.payload_kind == CouplingPayloadKind::DriverOwned) {
        return CouplingEndpointRef{
            .kind = CouplingEndpointKind::ExternalBuffer,
            .participant_name = request.producer_participant_name,
            .endpoint_name = metadata.provider_name,
            .temporal = request.producer_temporal,
        };
    }
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::RegionData,
        .participant_name = request.producer_participant_name,
        .endpoint_name = metadata.provider_name,
        .temporal = request.producer_temporal,
    };
}

CouplingExchangeDeclaration makeExchange(
    const std::string& contract_name,
    const CouplingPayloadExtractionRequest& request,
    CouplingPartitionedPayloadMetadata metadata)
{
    CouplingTransferDeclaration transfer = request.transfer;
    if (metadata.payload_kind == CouplingPayloadKind::DriverOwned &&
        transfer.kind == CouplingTransferKind::Unspecified) {
        transfer.kind = CouplingTransferKind::DriverOwned;
        transfer.driver_owned_name = metadata.provider_name;
    }

    CouplingExchangeDeclaration exchange;
    exchange.producer_port = CouplingPortId{
        .contract_instance_name = contract_name,
        .port_name = request.producer_port_name.empty()
                         ? defaultProducerPortName(request.exchange_name)
                         : request.producer_port_name,
    };
    exchange.consumer_port = CouplingPortId{
        .contract_instance_name = contract_name,
        .port_name = request.consumer_port_name.empty()
                         ? defaultConsumerPortName(request.exchange_name)
                         : request.consumer_port_name,
    };
    exchange.value = request.value;
    exchange.producer = producerEndpointFor(request, metadata);
    exchange.consumer = CouplingEndpointRef{
        .kind = CouplingEndpointKind::Field,
        .participant_name = request.consumer_participant_name,
        .endpoint_name = request.consumer_field_name,
        .temporal = request.consumer_temporal,
    };
    if (!request.shared_region_name.empty()) {
        exchange.shared_region_name = request.shared_region_name;
    }
    exchange.transfer = std::move(transfer);
    exchange.strategy = request.strategy;
    exchange.extracted_payload = std::move(metadata);
    return exchange;
}

CouplingPayloadExtractionDiagnostic makeDiagnostic(
    const CouplingPayloadExtractionRequest& request,
    CouplingPayloadKind selected_kind,
    CouplingPayloadExtractionReason reason,
    CouplingDiagnosticSeverity severity)
{
    return CouplingPayloadExtractionDiagnostic{
        .severity = severity,
        .exchange_name = request.exchange_name,
        .contribution_name = request.contribution_name,
        .preferred_kind = request.preferred_kind,
        .selected_kind = selected_kind,
        .fallback_policy = request.fallback_policy,
        .reason = reason,
        .message = diagnosticMessage(request, selected_kind, reason),
    };
}

} // namespace

CouplingPayloadExtractionResult CouplingPayloadDetangler::extract(
    const CouplingContext& ctx,
    std::span<const CouplingFormContribution> contributions,
    std::span<const CouplingPayloadExtractionRequest> requests,
    const std::string& contract_name) const
{
    CouplingPayloadExtractionResult result;
    for (const auto& request : requests) {
        const auto* contribution = findContribution(contributions,
                                                    request.contribution_name);
        if (contribution == nullptr) {
            const auto selected =
                fallbackKindFor(CouplingPayloadExtractionReason::ContributionNotFound,
                                request.fallback_policy);
            result.diagnostics.push_back(makeDiagnostic(
                request,
                selected,
                CouplingPayloadExtractionReason::ContributionNotFound,
                CouplingDiagnosticSeverity::Error));
            continue;
        }

        auto analysis = analyzePayload(ctx, *contribution, request);
        const bool fallback_is_error =
            request.fallback_policy == CouplingPayloadFallbackPolicy::Error &&
            analysis.reason != CouplingPayloadExtractionReason::Exact;
        if (fallback_is_error) {
            result.diagnostics.push_back(makeDiagnostic(
                request,
                analysis.selected_kind,
                analysis.reason,
                CouplingDiagnosticSeverity::Error));
            continue;
        }

        CouplingPartitionedPayloadMetadata metadata;
        metadata.preferred_kind = request.preferred_kind;
        metadata.payload_kind = analysis.selected_kind;
        metadata.fallback_policy = request.fallback_policy;
        metadata.reason = analysis.reason;
        metadata.exact = analysis.exact;
        metadata.source_contribution_name = contribution->contribution_name;
        metadata.provider_name = providerNameFor(request);
        metadata.diagnostic_message =
            diagnosticMessage(request, analysis.selected_kind, analysis.reason);
        metadata.payload_expression = std::move(analysis.payload_expression);

        if (analysis.reason != CouplingPayloadExtractionReason::Exact) {
            result.diagnostics.push_back(makeDiagnostic(
                request,
                analysis.selected_kind,
                analysis.reason,
                CouplingDiagnosticSeverity::Warning));
        }

        result.exchanges.push_back(makeExchange(contract_name,
                                                request,
                                                std::move(metadata)));
    }
    return result;
}

CouplingDiagnostic diagnosticFromPayloadExtraction(
    const CouplingPayloadExtractionDiagnostic& diagnostic)
{
    return CouplingDiagnostic{
        .severity = diagnostic.severity,
        .category = CouplingDiagnosticCategory::General,
        .endpoint_name = diagnostic.exchange_name,
        .message = diagnostic.message,
    };
}

} // namespace coupling
} // namespace FE
} // namespace svmp
