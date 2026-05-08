/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/AnalysisSummaryProducer.h"

#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/FortinOperatorAutogeneration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

template <typename T>
void appendVector(std::vector<T>& target, const std::vector<T>& source)
{
    target.insert(target.end(), source.begin(), source.end());
}

template <typename T>
void appendUnique(std::vector<T>& target, const T& value)
{
    if (std::find(target.begin(), target.end(), value) == target.end()) {
        target.push_back(value);
    }
}

[[nodiscard]] std::vector<VariableKey>
requestVariablesOrBlockVariables(const AnalysisSummaryRequest& request)
{
    if (!request.variables.empty()) {
        return request.variables;
    }
    return variablesForBlock(requestTargetBlock(request));
}

[[nodiscard]] std::string variableId(const VariableKey& variable)
{
    std::ostringstream os;
    os << static_cast<int>(variable.kind) << ":"
       << variable.field_id << ":"
       << variable.component << ":"
       << variable.name;
    return os.str();
}

[[nodiscard]] const FieldDescriptor*
fieldForVariable(const ProblemAnalysisContext& context,
                 const VariableKey& variable)
{
    if (variable.kind != VariableKind::FieldComponent ||
        variable.field_id == INVALID_FIELD_ID) {
        return nullptr;
    }
    return context.fieldDescriptor(variable.field_id);
}

[[nodiscard]] std::string requestScopeDescription(
    const AnalysisSummaryRequest& request)
{
    std::ostringstream os;
    os << "request_id='" << request.request_id << "'"
       << ", kind=" << toString(request.summary_kind)
       << ", domain=" << toString(request.domain);
    if (!request.block_id.empty()) {
        os << ", block='" << request.block_id << "'";
    }
    if (!request.contribution_id.empty()) {
        os << ", contribution='" << request.contribution_id << "'";
    }
    if (!request.scope_id.empty()) {
        os << ", scope='" << request.scope_id << "'";
    }
    if (!request.variables.empty()) {
        os << ", variables=";
        bool first = true;
        for (const auto& variable : request.variables) {
            if (!first) {
                os << "+";
            }
            first = false;
            os << variableId(variable);
        }
    }
    return os.str();
}

[[nodiscard]] std::string productNormId(
    const AnalysisSummaryRequest& request,
    const std::vector<VariableKey>& variables)
{
    std::ostringstream os;
    os << "norm:product";
    if (!request.block_id.empty()) {
        os << ":" << request.block_id;
    }
    if (!request.contribution_id.empty()) {
        os << ":contribution:" << request.contribution_id;
    }
    if (!request.scope_id.empty()) {
        os << ":scope:" << request.scope_id;
    }
    os << ":vars:";
    bool first = true;
    for (const auto& variable : variables) {
        if (!first) {
            os << "+";
        }
        first = false;
        os << variableId(variable);
    }
    return os.str();
}

void markGenerated(SummaryEvidenceMetadata& evidence,
                   EvidenceProvenance provenance)
{
    appendUnique(evidence.provenance, provenance);
    switch (provenance) {
        case EvidenceProvenance::GeneratedFromAssembledMatrix:
        case EvidenceProvenance::GeneratedFromReducedMatrix:
        case EvidenceProvenance::GeneratedFromLocalProjection:
        case EvidenceProvenance::GeneratedFromRefinementExperiment:
            evidence.generated_numeric_evidence = true;
            break;
        case EvidenceProvenance::MatchedToTheoremRegistry:
            evidence.theorem_matched_evidence = true;
            evidence.theorem_family_evidence = true;
            break;
        case EvidenceProvenance::UserDeclared:
            evidence.user_declared_evidence = true;
            break;
        case EvidenceProvenance::Unknown:
            break;
        default:
            evidence.inferred_evidence = true;
            break;
    }
}

SummaryProductionResult makeResult(const AnalysisSummaryRequest& request,
                                   const AnalysisSummaryProducer& producer,
                                   SummaryProductionStatus status,
                                   std::string message)
{
    SummaryProductionResult result;
    result.request = request;
    result.status = status;
    result.producer_id = producer.producerId();
    result.producer_version = producer.producerVersion();
    result.message = std::move(message);
    return result;
}

void attachCommonEvidence(SummaryEvidenceMetadata& evidence,
                          const AnalysisSummaryRequest& request,
                          AnalysisSummaryKind kind,
                          const ProblemAnalysisContext& context,
                          const MeshAccess& mesh,
                          const AnalysisSummaryProducer& producer)
{
    evidence = makeSummaryEvidenceMetadata(request,
                                           kind,
                                           context,
                                           mesh,
                                           producer.producerId(),
                                           producer.producerVersion());
}

[[nodiscard]] bool requestMentionsPair(const AnalysisSummaryRequest& request,
                                       const VariableKey& a,
                                       const VariableKey& b)
{
    const auto variables = requestVariablesOrBlockVariables(request);
    return std::find(variables.begin(), variables.end(), a) != variables.end() &&
           std::find(variables.begin(), variables.end(), b) != variables.end();
}

[[nodiscard]] bool requestMatchesCandidate(
    const AnalysisSummaryRequest& request,
    const FortinCandidate& candidate)
{
    if (!requestMentionsPair(request,
                             candidate.coupling.primal_variable,
                             candidate.coupling.multiplier_variable)) {
        return false;
    }
    if (request.domain != candidate.coupling.domain) {
        return false;
    }
    if (!request.block_id.empty() &&
        request.block_id != candidate.coupling.operator_tag) {
        return false;
    }
    if (!request.contribution_id.empty() &&
        std::find(candidate.coupling.contribution_ids.begin(),
                  candidate.coupling.contribution_ids.end(),
                  request.contribution_id) ==
            candidate.coupling.contribution_ids.end()) {
        return false;
    }
    return true;
}

void appendMissingHook(SummaryProductionResult& result, std::string hook)
{
    result.missing_backend_hooks.push_back(std::move(hook));
    result.request.missing_backend_hooks = result.missing_backend_hooks;
}

[[nodiscard]] SummaryProductionResult makePending(
    const AnalysisSummaryRequest& request,
    const AnalysisSummaryProducer& producer,
    std::string message,
    const char* hook = nullptr)
{
    auto result = makeResult(request,
                             producer,
                             SummaryProductionStatus::Pending,
                             std::move(message));
    if (hook != nullptr) {
        appendMissingHook(result, hook);
    }
    return result;
}

[[nodiscard]] SummaryProductionResult forwardUnavailable(
    const AnalysisSummaryRequest& request,
    const AnalysisSummaryProducer& producer,
    const char* hook)
{
    auto result = makeResult(request,
                             producer,
                             SummaryProductionStatus::Unavailable,
                             std::string("backend did not provide ") + hook);
    appendMissingHook(result, hook);
    return result;
}

[[nodiscard]] SummaryProductionResult forwardPendingOrUnavailable(
    const AnalysisSummaryRequest& request,
    const AnalysisSummaryProducer& producer,
    const char* hook,
    bool pending)
{
    if (pending) {
        return makePending(request,
                           producer,
                           std::string("backend evidence pending for ") + hook,
                           hook);
    }
    return forwardUnavailable(request, producer, hook);
}

[[nodiscard]] NullspaceHandlingClass conservativeNullspaceHandling(
    const std::vector<NormMetadataSummary>& summaries)
{
    if (summaries.empty()) {
        return NullspaceHandlingClass::Unknown;
    }
    bool saw_unknown = false;
    bool saw_uncontrolled = false;
    bool saw_retained = false;
    bool saw_projected = false;
    bool saw_anchored = false;
    bool saw_not_applicable = false;
    for (const auto& summary : summaries) {
        switch (summary.nullspace_handling) {
            case NullspaceHandlingClass::Unknown:
                saw_unknown = true;
                break;
            case NullspaceHandlingClass::Uncontrolled:
                saw_uncontrolled = true;
                break;
            case NullspaceHandlingClass::Retained:
                saw_retained = true;
                break;
            case NullspaceHandlingClass::ProjectedOut:
                saw_projected = true;
                break;
            case NullspaceHandlingClass::AnchoredByConstraints:
                saw_anchored = true;
                break;
            case NullspaceHandlingClass::NotApplicable:
                saw_not_applicable = true;
                break;
        }
    }
    if (saw_unknown) return NullspaceHandlingClass::Unknown;
    if (saw_uncontrolled) return NullspaceHandlingClass::Uncontrolled;
    if (saw_retained) return NullspaceHandlingClass::Retained;
    if (saw_projected) return NullspaceHandlingClass::ProjectedOut;
    if (saw_anchored) return NullspaceHandlingClass::AnchoredByConstraints;
    if (saw_not_applicable) return NullspaceHandlingClass::NotApplicable;
    return NullspaceHandlingClass::Unknown;
}

[[nodiscard]] NormMetadataSummary makeAggregateProductNorm(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const MeshAccess& mesh,
    const AnalysisSummaryProducer& producer,
    const std::vector<NormMetadataSummary>& components)
{
    NormMetadataSummary summary;
    summary.block = requestTargetBlock(request);
    summary.variables = requestVariablesOrBlockVariables(request);
    summary.contribution_ids = requestContributionIds(request);
    summary.scope_id = request.scope_id;
    summary.norm_id = productNormId(request, summary.variables);
    summary.norm_family = "Product";
    summary.norm_metadata_present = !components.empty();
    summary.evidence.strict_scope_complete = true;
    attachCommonEvidence(summary.evidence,
                         request,
                         AnalysisSummaryKind::NormMetadata,
                         context,
                         mesh,
                         producer);
    markGenerated(summary.evidence,
                  EvidenceProvenance::InferredFromFormAndSpaces);

    bool all_components_have_metadata = !components.empty();
    bool any_nullspace_metadata = false;
    for (const auto& component : components) {
        all_components_have_metadata =
            all_components_have_metadata && component.norm_metadata_present;
        summary.norm_matrix_available =
            summary.norm_matrix_available || component.norm_matrix_available;
        summary.mass_matrix_component_present =
            summary.mass_matrix_component_present ||
            component.mass_matrix_component_present;
        summary.gradient_matrix_component_present =
            summary.gradient_matrix_component_present ||
            component.gradient_matrix_component_present;
        summary.divergence_matrix_component_present =
            summary.divergence_matrix_component_present ||
            component.divergence_matrix_component_present;
        summary.curl_matrix_component_present =
            summary.curl_matrix_component_present ||
            component.curl_matrix_component_present;
        summary.jump_penalty_component_present =
            summary.jump_penalty_component_present ||
            component.jump_penalty_component_present;
        summary.trace_mass_component_present =
            summary.trace_mass_component_present ||
            component.trace_mass_component_present;
        summary.energy_or_entropy_weight_present =
            summary.energy_or_entropy_weight_present ||
            component.energy_or_entropy_weight_present;
        summary.equivalence_bounds_present =
            summary.equivalence_bounds_present ||
            component.equivalence_bounds_present;
        any_nullspace_metadata =
            any_nullspace_metadata ||
            component.gauge_or_nullspace_metadata_present;
    }
    summary.norm_metadata_present = all_components_have_metadata;
    summary.gauge_or_nullspace_metadata_present = any_nullspace_metadata;
    summary.nullspace_handling = conservativeNullspaceHandling(components);
    summary.nullspace_scope = request.scope_id;
    summary.evidence.norm_id = summary.norm_id;
    return summary;
}

[[nodiscard]] std::vector<VariableKey> contributionVariables(
    const ContributionDescriptor& contribution)
{
    std::vector<VariableKey> variables;
    for (const auto& variable : contribution.test_variables) {
        appendUnique(variables, variable);
    }
    for (const auto& variable : contribution.trial_variables) {
        appendUnique(variables, variable);
    }
    for (const auto& variable : contribution.related_variables) {
        appendUnique(variables, variable);
    }
    return variables;
}

[[nodiscard]] OperatorBlockId contributionBlock(
    const ContributionDescriptor& contribution)
{
    OperatorBlockId block;
    block.test_variables = contribution.test_variables;
    block.trial_variables = contribution.trial_variables;
    block.domain = contribution.domain;
    block.role = contribution.role;
    block.contribution_id = contribution.contribution_id;
    block.operator_tag = contribution.operator_tag;
    if (contribution.domain == DomainKind::Boundary) {
        block.marker = contribution.boundary_marker;
    } else if (contribution.domain == DomainKind::InterfaceFace) {
        block.marker = contribution.interface_marker;
    }
    return block;
}

[[nodiscard]] bool contributionCoversRequest(
    const ContributionDescriptor& contribution,
    const AnalysisSummaryRequest& request)
{
    return summaryBlockCoversRequest(contributionBlock(contribution),
                                     contributionVariables(contribution),
                                     contribution.contribution_id,
                                     request);
}

[[nodiscard]] std::string contributionIdentity(
    const ContributionDescriptor& contribution)
{
    if (!contribution.contribution_id.empty()) {
        return contribution.contribution_id;
    }
    if (!contribution.operator_tag.empty()) {
        return contribution.operator_tag;
    }
    return "contribution";
}

[[nodiscard]] NormMetadataSummary inferNormForField(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const MeshAccess& mesh,
    const AnalysisSummaryProducer& producer,
    const VariableKey& variable,
    const FieldDescriptor& field)
{
    NormMetadataSummary summary;
    summary.block = requestTargetBlock(request);
    summary.block.test_variables = {variable};
    summary.block.trial_variables = {variable};
    summary.variables = {variable};
    summary.contribution_ids = requestContributionIds(request);
    if (requestVariablesOrBlockVariables(request).size() == 1u) {
        summary.scope_id = request.scope_id;
    }
    summary.norm_metadata_present = true;
    summary.evidence.strict_scope_complete = true;
    attachCommonEvidence(summary.evidence,
                         request,
                         AnalysisSummaryKind::NormMetadata,
                         context,
                         mesh,
                         producer);
    markGenerated(summary.evidence,
                  EvidenceProvenance::InferredFromFormAndSpaces);

    const std::string base = "field:" + variableId(variable);
    switch (field.space_family) {
        case SpaceFamily::H1:
            summary.norm_family =
                field.value_dimension > 1 ? "H1-vector" : "H1";
            summary.norm_id = "norm:h1:" + base;
            summary.mass_matrix_component_present = true;
            summary.gradient_matrix_component_present = true;
            break;
        case SpaceFamily::HDiv:
            summary.norm_family = "HDiv";
            summary.norm_id = "norm:hdiv:" + base;
            summary.mass_matrix_component_present = true;
            summary.divergence_matrix_component_present = true;
            break;
        case SpaceFamily::HCurl:
            summary.norm_family = "HCurl";
            summary.norm_id = "norm:hcurl:" + base;
            summary.mass_matrix_component_present = true;
            summary.curl_matrix_component_present = true;
            break;
        case SpaceFamily::L2:
            summary.norm_family = "L2";
            summary.norm_id = "norm:l2:" + base;
            summary.mass_matrix_component_present = true;
            break;
        case SpaceFamily::DG:
            summary.norm_family = "DG-broken-energy";
            summary.norm_id = "norm:dg-broken-energy:" + base;
            summary.mass_matrix_component_present = true;
            summary.gradient_matrix_component_present = true;
            summary.jump_penalty_component_present = true;
            break;
        case SpaceFamily::Trace:
            summary.norm_family = "Trace";
            summary.norm_id = "norm:trace:" + base;
            summary.trace_mass_component_present = true;
            break;
        case SpaceFamily::Custom:
        case SpaceFamily::Unknown:
            summary.norm_family = "Unknown";
            summary.norm_id = "norm:unknown:" + base;
            summary.norm_metadata_present = false;
            break;
    }

    if (field.gauge_fixing_metadata_present) {
        summary.nullspace_handling = NullspaceHandlingClass::AnchoredByConstraints;
        summary.gauge_or_nullspace_metadata_present = true;
        summary.nullspace_scope = field.declared_nullspace_scope;
    } else if (field.mean_zero_constraint_present) {
        summary.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
        summary.gauge_or_nullspace_metadata_present = true;
    } else if (field.declared_nullspace_metadata_present) {
        summary.nullspace_handling = NullspaceHandlingClass::Retained;
        summary.gauge_or_nullspace_metadata_present = true;
        summary.nullspace_scope = field.declared_nullspace_scope;
    } else {
        summary.nullspace_handling = NullspaceHandlingClass::Unknown;
    }

    summary.evidence.norm_id = summary.norm_id;
    return summary;
}

[[nodiscard]] TheoremFamily theoremFamilyForRequest(
    const AnalysisSummaryRequest& request) noexcept
{
    for (const auto kind : request.source_claim_kinds) {
        switch (kind) {
            case PropertyKind::InfSupCondition:
            case PropertyKind::MixedSaddlePoint:
            case PropertyKind::SpaceCompatibility:
                return TheoremFamily::InfSup;
            case PropertyKind::MMatrixStructure:
                return TheoremFamily::MMatrix;
            case PropertyKind::DiscreteMaximumPrinciple:
            case PropertyKind::ZMatrixStructure:
            case PropertyKind::MatrixMonotonicityRisk:
                return TheoremFamily::ScalarDMP;
            case PropertyKind::TemporalStability:
                return TheoremFamily::TemporalCFL;
            case PropertyKind::EnergyStability:
            case PropertyKind::EntropyStability:
                return TheoremFamily::EnergyEntropy;
            case PropertyKind::IndefiniteOperatorResolution:
                return TheoremFamily::Schur;
            default:
                break;
        }
    }
    return TheoremFamily::ScalarDMP;
}

template <typename Summary>
void stampForwardedEvidence(Summary& summary,
                            const AnalysisSummaryRequest& request,
                            AnalysisSummaryKind kind,
                            const ProblemAnalysisContext& context,
                            const MeshAccess& mesh,
                            const AnalysisSummaryProducer& producer,
                            EvidenceProvenance provenance)
{
    attachCommonEvidence(summary.evidence, request, kind, context, mesh, producer);
    summary.evidence.strict_scope_complete = true;
    markGenerated(summary.evidence, provenance);
}

} // namespace

std::optional<NormMetadataSummary>
AssemblyAccess::normMetadata(const AnalysisSummaryRequest&) const { return std::nullopt; }

bool AssemblyAccess::evidencePending() const noexcept { return false; }

std::optional<DiscreteMatrixSummary>
AssemblyAccess::discreteMatrixSummary(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<ReducedMatrixSummary>
AssemblyAccess::reducedMatrixSummary(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<CoefficientPropertySummary>
AssemblyAccess::coefficientProperties(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<BoundarySymbolSummary>
AssemblyAccess::boundarySymbol(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<InfSupEstimateSummary>
AssemblyAccess::infSupEstimate(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<DAEStructureEvidenceSummary>
AssemblyAccess::daeStructureEvidence(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<QuadratureAdequacySummary>
AssemblyAccess::quadratureAdequacy(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<NullspaceDegeneracySummary>
AssemblyAccess::nullspaceDegeneracy(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<ParameterScaleSummary>
AssemblyAccess::parameterScale(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<StabilizationAdequacySummary>
AssemblyAccess::stabilizationAdequacy(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<InitialCompatibilitySummary>
AssemblyAccess::initialCompatibility(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<LocalStencilSummary>
AssemblyAccess::localStencil(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<NumericalErrorBudgetSummary>
AssemblyAccess::numericalErrorBudget(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::string MeshAccess::meshRevision() const { return {}; }

bool MeshAccess::evidencePending() const noexcept { return false; }

std::optional<MeshGeometryQualitySummary>
MeshAccess::meshGeometryQuality(const AnalysisSummaryRequest&) const { return std::nullopt; }

std::optional<RobustnessTrendSummary>
MeshAccess::refinementExperiment(const AnalysisSummaryRequest&,
                                 const AssemblyAccess&) const { return std::nullopt; }

std::optional<TemporalStabilitySummary>
SolverAccess::temporalStability(const AnalysisSummaryRequest&) const { return std::nullopt; }

bool SolverAccess::evidencePending() const noexcept { return false; }

std::optional<SchurComplementSummary>
SolverAccess::schurComplement(const AnalysisSummaryRequest&) const { return std::nullopt; }

const char* toString(SummaryProductionStatus status) noexcept
{
    switch (status) {
        case SummaryProductionStatus::Produced: return "Produced";
        case SummaryProductionStatus::AlreadyAvailable: return "AlreadyAvailable";
        case SummaryProductionStatus::Pending: return "Pending";
        case SummaryProductionStatus::Unavailable: return "Unavailable";
        case SummaryProductionStatus::Unsupported: return "Unsupported";
        case SummaryProductionStatus::Failed: return "Failed";
        case SummaryProductionStatus::FailedStrictScopeCoverage:
            return "FailedStrictScopeCoverage";
        case SummaryProductionStatus::Skipped: return "Skipped";
    }
    return "Unavailable";
}

void mergeAnalysisSummarySets(AnalysisSummarySet& target,
                              const AnalysisSummarySet& source)
{
    appendVector(target.norm_metadata, source.norm_metadata);
    appendVector(target.coefficient_properties, source.coefficient_properties);
    appendVector(target.discrete_matrices, source.discrete_matrices);
    appendVector(target.reduced_matrices, source.reduced_matrices);
    appendVector(target.schur_complements, source.schur_complements);
    appendVector(target.nullspace_degeneracies, source.nullspace_degeneracies);
    appendVector(target.robustness_trends, source.robustness_trends);
    appendVector(target.applicability, source.applicability);
    appendVector(target.numerical_error_budgets, source.numerical_error_budgets);
    appendVector(target.local_stencils, source.local_stencils);
    appendVector(target.mesh_geometry_quality, source.mesh_geometry_quality);
    appendVector(target.flux_balances, source.flux_balances);
    appendVector(target.temporal_stability, source.temporal_stability);
    appendVector(target.boundary_symbols, source.boundary_symbols);
    appendVector(target.inf_sup_estimates, source.inf_sup_estimates);
    appendVector(target.inf_sup_pair_certifications,
                 source.inf_sup_pair_certifications);
    appendVector(target.energy_entropy, source.energy_entropy);
    appendVector(target.invariant_domains, source.invariant_domains);
    appendVector(target.equilibrium_preservation, source.equilibrium_preservation);
    appendVector(target.moving_domain, source.moving_domain);
    appendVector(target.transfer_operators, source.transfer_operators);
    appendVector(target.adjoint_consistency, source.adjoint_consistency);
    appendVector(target.parameter_scales, source.parameter_scales);
    appendVector(target.stabilization_adequacy, source.stabilization_adequacy);
    appendVector(target.initial_compatibility, source.initial_compatibility);
    appendVector(target.dae_structure_evidence, source.dae_structure_evidence);
    appendVector(target.compatible_complexes, source.compatible_complexes);
    appendVector(target.nonlinear_tangents, source.nonlinear_tangents);
    appendVector(target.spectral_structures, source.spectral_structures);
    appendVector(target.error_estimators, source.error_estimators);
    appendVector(target.quadrature_adequacy, source.quadrature_adequacy);
    appendVector(target.coupled_system_stability,
                 source.coupled_system_stability);
    appendVector(target.minimum_residual_stability,
                 source.minimum_residual_stability);
}

SummaryEvidenceMetadata makeSummaryEvidenceMetadata(
    const AnalysisSummaryRequest& request,
    AnalysisSummaryKind kind,
    const ProblemAnalysisContext& context,
    const MeshAccess& mesh,
    const std::string& producer_id,
    const std::string& producer_version)
{
    SummaryEvidenceMetadata evidence;
    evidence.summary_kind = kind;
    evidence.summary_id = request.request_id;
    evidence.request_id = request.request_id;
    evidence.producer_id = producer_id;
    evidence.producer_version = producer_version;
    evidence.mesh_revision = mesh.meshRevision();
    evidence.formulation_revision = std::to_string(context.inputsVersion());
    evidence.operator_tag = request.block_id;
    evidence.contribution_ids = requestContributionIds(request);
    evidence.variables = requestVariablesOrBlockVariables(request);
    evidence.domain = request.domain;
    evidence.norm_id = request.scope_id;
    evidence.theorem_id = request.scope_id;
    return evidence;
}

void AnalysisSummaryProducerRegistry::registerProducer(
    std::unique_ptr<AnalysisSummaryProducer> producer)
{
    if (!producer) {
        return;
    }
    const auto kinds = producer->producedKinds();
    for (const auto kind : kinds) {
        producers_[kind].push_back(std::move(producer));
        break;
    }
}

std::vector<const AnalysisSummaryProducer*>
AnalysisSummaryProducerRegistry::producersFor(AnalysisSummaryKind kind) const
{
    std::vector<const AnalysisSummaryProducer*> result;
    const auto it = producers_.find(kind);
    if (it == producers_.end()) {
        return result;
    }
    result.reserve(it->second.size());
    for (const auto& producer : it->second) {
        result.push_back(producer.get());
    }
    return result;
}

SummaryProductionBatch AnalysisSummaryProducerRegistry::fulfillRequestPlan(
    const AnalysisRequestPlan& request_plan,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess& solver,
    AnalysisSummarySet existing_summaries) const
{
    SummaryProductionBatch batch;
    batch.fulfilled_plan = request_plan;

    for (auto& request : batch.fulfilled_plan.summary_requests) {
        if (analysisSummarySetCoversRequest(existing_summaries, request)) {
            request.already_available = true;
            request.production_status = toString(SummaryProductionStatus::AlreadyAvailable);
            ++batch.already_available_count;
            continue;
        }

        request.production_attempted = true;
        const auto candidates = producersFor(request.summary_kind);
        if (candidates.empty()) {
            request.production_unavailable = true;
            request.production_status = toString(SummaryProductionStatus::Unavailable);
            request.unavailable_reason = "no registered producer";
            ++batch.unavailable_count;
            continue;
        }

        bool produced = false;
        SummaryProductionResult last_result;
        for (const auto* producer : candidates) {
            if (!producer || !producer->canProduce(request, context)) {
                continue;
            }
            auto result = producer->produce(request, context, assembly, mesh, solver);

            AnalyzerRunLogSummary log;
            log.analyzer = result.producer_id;
            log.pass_name = "AnalysisSummaryProducerRegistry";
            log.pass_version = result.producer_version;
            log.summary_id = request.request_id;
            log.attempted_count = 1;

            if (result.status == SummaryProductionStatus::Produced) {
                if (analysisSummarySetCoversRequest(result.produced_summaries,
                                                    request)) {
                    mergeAnalysisSummarySets(existing_summaries,
                                             result.produced_summaries);
                    mergeAnalysisSummarySets(batch.produced_summaries,
                                             result.produced_summaries);
                    request.already_available = true;
                    request.production_succeeded = true;
                    request.production_status = toString(result.status);
                    batch.produced_count +=
                        static_cast<std::uint64_t>(result.producedSummaryCount());
                    log.certified_count =
                        static_cast<std::uint64_t>(result.producedSummaryCount());
                    log.status = toString(result.status);
                    log.message = result.message;
                    log.diagnostics = result.diagnostics;
                    produced = true;
                    batch.results.push_back(result);
                    batch.run_logs.push_back(std::move(log));
                    break;
                }

                result.status =
                    SummaryProductionStatus::FailedStrictScopeCoverage;
                result.message =
                    "strict scope coverage failed for " +
                    requestScopeDescription(request);
                result.missing_scope_components.push_back(
                    "request strict block/variable/contribution/scope coverage");
                result.diagnostics.push_back(
                    "produced_summary_count=" +
                    std::to_string(result.producedSummaryCount()));
                result.diagnostics.push_back(
                    "producer_id='" + result.producer_id + "'");
                AnalysisIssue issue;
                issue.severity = IssueSeverity::Warning;
                issue.message =
                    result.producer_id +
                    " produced summary evidence that failed strict scope coverage for request '" +
                    request.request_id + "'";
                batch.issues.push_back(std::move(issue));
            }

            log.status = toString(result.status);
            log.message = result.message;
            log.diagnostics = result.diagnostics;

            last_result = result;
            if (result.status == SummaryProductionStatus::Failed ||
                result.status ==
                    SummaryProductionStatus::FailedStrictScopeCoverage) {
                ++batch.failed_count;
            } else if (result.status == SummaryProductionStatus::Pending) {
                ++batch.pending_count;
            }
            for (const auto& issue : result.issues) {
                batch.issues.push_back(issue);
            }
            batch.results.push_back(result);
            batch.run_logs.push_back(std::move(log));
        }

        if (!produced) {
            const bool pending =
                !last_result.producer_id.empty() &&
                last_result.status == SummaryProductionStatus::Pending;
            request.production_pending = pending;
            request.production_unavailable = !pending;
            request.production_status = pending
                ? toString(SummaryProductionStatus::Pending)
                : (last_result.producer_id.empty()
                       ? toString(SummaryProductionStatus::Unavailable)
                       : toString(last_result.status));
            request.unavailable_reason = last_result.message.empty()
                ? "no producer could satisfy strict scoped request"
                : last_result.message;
            request.missing_backend_hooks = last_result.missing_backend_hooks;
            if (!pending) {
                ++batch.unavailable_count;
            }
        }
    }

    return batch;
}

AnalysisSummaryProducerRegistry
AnalysisSummaryProducerRegistry::createDefault()
{
    AnalysisSummaryProducerRegistry registry;
    registry.registerProducer(std::make_unique<NormMetadataProducer>());
    registry.registerProducer(std::make_unique<TheoremApplicabilityProducer>());
    registry.registerProducer(std::make_unique<FortinStablePairProducer>());
    registry.registerProducer(std::make_unique<InfSupNumericalTestProducer>());
    registry.registerProducer(std::make_unique<MeshFamilyProducer>());
    registry.registerProducer(std::make_unique<DiscreteMatrixSummaryProducer>());
    registry.registerProducer(std::make_unique<ReducedMatrixSummaryProducer>());
    registry.registerProducer(std::make_unique<CoefficientBoundsProducer>());
    registry.registerProducer(std::make_unique<BoundarySymbolProducer>());
    registry.registerProducer(std::make_unique<TemporalStabilityProducer>());
    registry.registerProducer(std::make_unique<DAEStructureEvidenceProducer>());
    registry.registerProducer(std::make_unique<SchurComplementProducer>());
    registry.registerProducer(std::make_unique<RefinementExperimentRunner>());
    registry.registerProducer(std::make_unique<QuadratureAdequacyProducer>());
    registry.registerProducer(std::make_unique<InvariantDomainProducer>());
    registry.registerProducer(std::make_unique<NullspaceDegeneracyProducer>());
    registry.registerProducer(std::make_unique<ParameterScaleProducer>());
    registry.registerProducer(std::make_unique<StabilizationAdequacyProducer>());
    registry.registerProducer(std::make_unique<InitialCompatibilityProducer>());
    registry.registerProducer(std::make_unique<LocalStencilProducer>());
    registry.registerProducer(std::make_unique<NumericalErrorBudgetProducer>());
    return registry;
}

std::string NormMetadataProducer::producerId() const { return "NormMetadataProducer"; }

std::vector<AnalysisSummaryKind> NormMetadataProducer::producedKinds() const
{
    return {AnalysisSummaryKind::NormMetadata};
}

bool NormMetadataProducer::canProduce(const AnalysisSummaryRequest& request,
                                      const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::NormMetadata;
}

SummaryProductionResult NormMetadataProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto from_backend = assembly.normMetadata(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "norm metadata provided by assembly backend");
        stampForwardedEvidence(*from_backend,
                               request,
                               AnalysisSummaryKind::NormMetadata,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.norm_metadata.push_back(std::move(*from_backend));
        return result;
    }

    auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                             "norm metadata inferred from field and space descriptors");
    const auto requested_variables = requestVariablesOrBlockVariables(request);
    std::vector<NormMetadataSummary> components;
    for (const auto& variable : requested_variables) {
        const auto* field = fieldForVariable(context, variable);
        if (!field) {
            result.diagnostics.push_back(
                "no FieldDescriptor for variable " + variableId(variable));
            continue;
        }
        auto summary = inferNormForField(request, context, mesh, *this, variable, *field);
        components.push_back(std::move(summary));
    }
    for (const auto& summary : components) {
        result.produced_summaries.norm_metadata.push_back(summary);
    }
    if (requested_variables.size() > 1u &&
        components.size() == requested_variables.size()) {
        result.produced_summaries.norm_metadata.push_back(
            makeAggregateProductNorm(request, context, mesh, *this, components));
    }
    if (result.produced_summaries.norm_metadata.empty()) {
        if (assembly.evidencePending()) {
            result.status = SummaryProductionStatus::Pending;
            result.message =
                "backend evidence pending for norm metadata and no descriptor-backed variables are available";
            appendMissingHook(result, "AssemblyAccess::normMetadata");
        } else {
            result.status = SummaryProductionStatus::Unavailable;
            result.message = "no descriptor-backed variables available for norm inference";
        }
    }
    return result;
}

std::string TheoremApplicabilityProducer::producerId() const
{
    return "TheoremApplicabilityProducer";
}

std::vector<AnalysisSummaryKind> TheoremApplicabilityProducer::producedKinds() const
{
    return {AnalysisSummaryKind::Applicability};
}

bool TheoremApplicabilityProducer::canProduce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::Applicability;
}

SummaryProductionResult TheoremApplicabilityProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess&,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                             "theorem applicability inferred from mathematical metadata");
    ApplicabilitySummary summary;
    summary.theorem_family = theoremFamilyForRequest(request);
    summary.block = requestTargetBlock(request);
    summary.variables = requestVariablesOrBlockVariables(request);
    summary.applicability = ApplicabilityClass::InsufficientMetadata;
    summary.reason =
        "curated theorem registry did not find a complete assumption set";
    summary.missing_assumptions_present = true;
    summary.missing_assumptions.push_back(
        "complete theorem, norm, mesh-family, coefficient, and boundary/nullspace scope");
    summary.required_followup_summary_kinds = {
        AnalysisSummaryKind::NormMetadata,
        AnalysisSummaryKind::MeshGeometryQuality,
    };
    summary.inferred_from_field_descriptors = !context.fieldDescriptors().empty();
    summary.inferred_from_contribution_traits = !context.contributions().empty();
    summary.inferred_from_block_structure = !request.block_id.empty() ||
                                            !request.variables.empty();
    attachCommonEvidence(summary.evidence,
                         request,
                         AnalysisSummaryKind::Applicability,
                         context,
                         mesh,
                         *this);
    markGenerated(summary.evidence,
                  EvidenceProvenance::MatchedToTheoremRegistry);
    summary.evidence.strict_scope_complete = true;
    result.produced_summaries.applicability.push_back(std::move(summary));
    return result;
}

std::string FortinStablePairProducer::producerId() const
{
    return "FortinStablePairProducer";
}

std::vector<AnalysisSummaryKind> FortinStablePairProducer::producedKinds() const
{
    return {AnalysisSummaryKind::InfSupPairCertification};
}

bool FortinStablePairProducer::canProduce(const AnalysisSummaryRequest& request,
                                          const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::InfSupPairCertification;
}

SummaryProductionResult FortinStablePairProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess&,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    auto result = makeResult(request, *this, SummaryProductionStatus::Unavailable,
                             "no complete metadata-driven stable-pair theorem match");
    FortinCandidateBuilder builder;
    const auto build = builder.build(context);
    for (const auto& diagnostic : build.diagnostics) {
        result.diagnostics.push_back(diagnostic);
    }
    for (const auto& candidate : build.candidates) {
        if (!requestMatchesCandidate(request, candidate)) {
            continue;
        }
        for (const auto& diagnostic : candidate.diagnostics) {
            result.diagnostics.push_back(diagnostic);
        }
        if (candidate.status != FortinCandidateStatus::Complete) {
            continue;
        }
        auto summary = makeInfSupPairCertificationSummary(candidate, context);
        if (!summary) {
            continue;
        }
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::InfSupPairCertification,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::MatchedToTheoremRegistry);
        summary->evidence.theorem_id = summary->inf_sup_theorem_id;
        summary->evidence.norm_id = summary->projection_target_norm;
        result.produced_summaries.inf_sup_pair_certifications.push_back(
            std::move(*summary));
    }
    if (!result.produced_summaries.inf_sup_pair_certifications.empty()) {
        result.status = SummaryProductionStatus::Produced;
        result.message =
            "stable-pair/Fortin metadata matched from spaces, coupling, mesh, boundary, and nullspace descriptors";
    }
    return result;
}

#define SVMP_FORWARD_PRODUCER(CLASS_NAME, ID_TEXT, KIND_VALUE, ACCESS_EXPR, HOOK_TEXT, PROVENANCE_VALUE, VECTOR_NAME) \
std::string CLASS_NAME::producerId() const { return ID_TEXT; } \
std::vector<AnalysisSummaryKind> CLASS_NAME::producedKinds() const { return {KIND_VALUE}; } \
bool CLASS_NAME::canProduce(const AnalysisSummaryRequest& request, const ProblemAnalysisContext&) const { \
    return request.summary_kind == KIND_VALUE; \
} \
SummaryProductionResult CLASS_NAME::produce( \
    const AnalysisSummaryRequest& request, \
    const ProblemAnalysisContext& context, \
    const AssemblyAccess& assembly, \
    const MeshAccess& mesh, \
    const SolverAccess& solver) const \
{ \
    (void)assembly; \
    (void)solver; \
    auto summary = ACCESS_EXPR; \
    if (!summary) { \
        return forwardPendingOrUnavailable(request, *this, HOOK_TEXT, \
                                           assembly.evidencePending() || \
                                               mesh.evidencePending() || \
                                               solver.evidencePending()); \
    } \
    auto result = makeResult(request, *this, SummaryProductionStatus::Produced, \
                             std::string(HOOK_TEXT) + " provided scoped summary"); \
    stampForwardedEvidence(*summary, request, KIND_VALUE, context, mesh, *this, PROVENANCE_VALUE); \
    result.produced_summaries.VECTOR_NAME.push_back(std::move(*summary)); \
    return result; \
}

SVMP_FORWARD_PRODUCER(InfSupNumericalTestProducer,
                      "InfSupNumericalTestProducer",
                      AnalysisSummaryKind::InfSupEstimate,
                      assembly.infSupEstimate(request),
                      "AssemblyAccess::infSupEstimate",
                      EvidenceProvenance::GeneratedFromAssembledMatrix,
                      inf_sup_estimates)

SVMP_FORWARD_PRODUCER(DiscreteMatrixSummaryProducer,
                      "DiscreteMatrixSummaryProducer",
                      AnalysisSummaryKind::DiscreteMatrix,
                      assembly.discreteMatrixSummary(request),
                      "AssemblyAccess::discreteMatrixSummary",
                      EvidenceProvenance::GeneratedFromAssembledMatrix,
                      discrete_matrices)

SVMP_FORWARD_PRODUCER(ReducedMatrixSummaryProducer,
                      "ReducedMatrixSummaryProducer",
                      AnalysisSummaryKind::ReducedMatrix,
                      assembly.reducedMatrixSummary(request),
                      "AssemblyAccess::reducedMatrixSummary",
                      EvidenceProvenance::GeneratedFromReducedMatrix,
                      reduced_matrices)

SVMP_FORWARD_PRODUCER(CoefficientBoundsProducer,
                      "CoefficientBoundsProducer",
                      AnalysisSummaryKind::CoefficientProperties,
                      assembly.coefficientProperties(request),
                      "AssemblyAccess::coefficientProperties",
                      EvidenceProvenance::GeneratedFromAssembledMatrix,
                      coefficient_properties)

SVMP_FORWARD_PRODUCER(TemporalStabilityProducer,
                      "TemporalStabilityProducer",
                      AnalysisSummaryKind::TemporalStability,
                      solver.temporalStability(request),
                      "SolverAccess::temporalStability",
                      EvidenceProvenance::InferredFromSolverSettings,
                      temporal_stability)

SVMP_FORWARD_PRODUCER(DAEStructureEvidenceProducer,
                      "DAEStructureEvidenceProducer",
                      AnalysisSummaryKind::DAEStructureEvidence,
                      assembly.daeStructureEvidence(request),
                      "AssemblyAccess::daeStructureEvidence",
                      EvidenceProvenance::GeneratedFromAssembledMatrix,
                      dae_structure_evidence)

SVMP_FORWARD_PRODUCER(SchurComplementProducer,
                      "SchurComplementProducer",
                      AnalysisSummaryKind::SchurComplement,
                      solver.schurComplement(request),
                      "SolverAccess::schurComplement",
                      EvidenceProvenance::GeneratedFromReducedMatrix,
                      schur_complements)

SVMP_FORWARD_PRODUCER(RefinementExperimentRunner,
                      "RefinementExperimentRunner",
                      AnalysisSummaryKind::RobustnessTrend,
                      mesh.refinementExperiment(request, assembly),
                      "MeshAccess::refinementExperiment",
                      EvidenceProvenance::GeneratedFromRefinementExperiment,
                      robustness_trends)

SVMP_FORWARD_PRODUCER(QuadratureAdequacyProducer,
                      "QuadratureAdequacyProducer",
                      AnalysisSummaryKind::QuadratureAdequacy,
                      assembly.quadratureAdequacy(request),
                      "AssemblyAccess::quadratureAdequacy",
                      EvidenceProvenance::InferredFromFormAndSpaces,
                      quadrature_adequacy)

#undef SVMP_FORWARD_PRODUCER

std::string NullspaceDegeneracyProducer::producerId() const
{
    return "NullspaceDegeneracyProducer";
}

std::vector<AnalysisSummaryKind> NullspaceDegeneracyProducer::producedKinds() const
{
    return {AnalysisSummaryKind::NullspaceDegeneracy};
}

bool NullspaceDegeneracyProducer::canProduce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::NullspaceDegeneracy;
}

SummaryProductionResult NullspaceDegeneracyProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = assembly.nullspaceDegeneracy(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "assembly backend provided nullspace-degeneracy summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::NullspaceDegeneracy,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.nullspace_degeneracies.push_back(
            std::move(*summary));
        return result;
    }

    std::vector<VariableKey> affected_variables = requestVariablesOrBlockVariables(request);
    bool saw_nullspace_metadata = false;
    NullspaceHandlingClass handling = NullspaceHandlingClass::Unknown;
    for (const auto& variable : affected_variables) {
        const auto* field = fieldForVariable(context, variable);
        if (field == nullptr) {
            continue;
        }
        if (field->gauge_fixing_metadata_present ||
            field->mean_zero_constraint_present ||
            field->declared_nullspace_metadata_present) {
            saw_nullspace_metadata = true;
            if (field->gauge_fixing_metadata_present) {
                handling = NullspaceHandlingClass::AnchoredByConstraints;
            } else if (field->mean_zero_constraint_present) {
                handling = NullspaceHandlingClass::ProjectedOut;
            } else if (handling == NullspaceHandlingClass::Unknown) {
                handling = NullspaceHandlingClass::Retained;
            }
        }
    }

    if (!saw_nullspace_metadata) {
        return forwardPendingOrUnavailable(request,
                                           *this,
                                           "AssemblyAccess::nullspaceDegeneracy",
                                           assembly.evidencePending());
    }

    NullspaceDegeneracySummary summary;
    summary.degeneracy_id = request.scope_id.empty()
        ? request.request_id + ":nullspace"
        : request.scope_id;
    summary.block = requestTargetBlock(request);
    summary.affected_variables = std::move(affected_variables);
    summary.nullspace_handling = handling;
    summary.kernel_claim_evidence_present = true;
    summary.reason =
        "field-level nullspace metadata is present; rank/nullity remain unavailable without assembled matrix evidence";
    stampForwardedEvidence(summary,
                           request,
                           AnalysisSummaryKind::NullspaceDegeneracy,
                           context,
                           mesh,
                           *this,
                           EvidenceProvenance::InferredFromFormAndSpaces);
    auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                             "nullspace-degeneracy metadata inferred from field descriptors");
    result.produced_summaries.nullspace_degeneracies.push_back(std::move(summary));
    return result;
}

std::string ParameterScaleProducer::producerId() const
{
    return "ParameterScaleProducer";
}

std::vector<AnalysisSummaryKind> ParameterScaleProducer::producedKinds() const
{
    return {AnalysisSummaryKind::ParameterScale};
}

bool ParameterScaleProducer::canProduce(const AnalysisSummaryRequest& request,
                                        const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::ParameterScale;
}

SummaryProductionResult ParameterScaleProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = assembly.parameterScale(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "assembly backend provided parameter-scale summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::ParameterScale,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.parameter_scales.push_back(std::move(*summary));
        return result;
    }

    auto result = makeResult(request, *this, SummaryProductionStatus::Unavailable,
                             "no scoped contribution scale metadata covered request");
    int index = 0;
    for (const auto& contribution : context.contributions()) {
        if (contribution.scale_usages.empty() && !contribution.scaling.has_value()) {
            continue;
        }
        if (!contributionCoversRequest(contribution, request)) {
            continue;
        }
        ParameterScaleSummary summary;
        summary.nondimensional_parameter_id =
            contributionIdentity(contribution) + ":scale:" + std::to_string(index++);
        summary.role = ParameterScaleRole::Generic;
        if (contribution.role == ContributionRole::BoundaryConstraint) {
            summary.role = ParameterScaleRole::WeakBoundaryPenalty;
        }
        summary.block = requestTargetBlock(request);
        summary.variables = requestVariablesOrBlockVariables(request);
        summary.contribution_id = request.contribution_id;
        summary.min_scale_value = static_cast<Real>(1);
        summary.max_scale_value = static_cast<Real>(1);
        summary.evidence_scope_id = request.scope_id;
        if (contribution.scaling.has_value()) {
            summary.scale_theorem_id = "ContributionDescriptor::ScalingDescriptor";
            summary.trace_inverse_metadata_present =
                summary.role == ParameterScaleRole::WeakBoundaryPenalty &&
                contribution.scaling->h_power < 0;
            if (summary.trace_inverse_metadata_present) {
                summary.trace_inverse_constant = static_cast<Real>(1);
            }
            summary.coefficient_contrast_factor =
                contribution.scaling->coefficient_scaled ? static_cast<Real>(1)
                                                         : static_cast<Real>(0);
        } else {
            summary.scale_theorem_id = "ContributionDescriptor::FormScaleUsage";
        }
        stampForwardedEvidence(summary,
                               request,
                               AnalysisSummaryKind::ParameterScale,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::InferredFromFormAndSpaces);
        result.produced_summaries.parameter_scales.push_back(std::move(summary));
    }

    if (!result.produced_summaries.parameter_scales.empty()) {
        result.status = SummaryProductionStatus::Produced;
        result.message =
            "parameter-scale summary inferred from scoped contribution descriptors";
        return result;
    }
    return forwardPendingOrUnavailable(request,
                                       *this,
                                       "AssemblyAccess::parameterScale",
                                       assembly.evidencePending());
}

std::string StabilizationAdequacyProducer::producerId() const
{
    return "StabilizationAdequacyProducer";
}

std::vector<AnalysisSummaryKind> StabilizationAdequacyProducer::producedKinds() const
{
    return {AnalysisSummaryKind::StabilizationAdequacy};
}

bool StabilizationAdequacyProducer::canProduce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::StabilizationAdequacy;
}

SummaryProductionResult StabilizationAdequacyProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = assembly.stabilizationAdequacy(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "assembly backend provided stabilization-adequacy summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::StabilizationAdequacy,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.stabilization_adequacy.push_back(
            std::move(*summary));
        return result;
    }

    auto result = makeResult(request, *this, SummaryProductionStatus::Unavailable,
                             "no scoped stabilization contribution metadata covered request");
    for (const auto& contribution : context.contributions()) {
        if (contribution.role != ContributionRole::StabilizationBlock ||
            !contributionCoversRequest(contribution, request)) {
            continue;
        }
        StabilizationAdequacySummary summary;
        summary.stabilization_id = contributionIdentity(contribution);
        summary.method_family = "contribution-descriptor-stabilization";
        summary.family = StabilizationFamily::Unknown;
        summary.block = requestTargetBlock(request);
        summary.variables = requestVariablesOrBlockVariables(request);
        summary.requirement_metadata_present = true;
        summary.method_scope_metadata_present = true;
        summary.residual_consistency_evidence_present =
            contribution.consistency_kind.has_value() &&
            *contribution.consistency_kind != ConsistencyKind::Unknown;
        summary.parameter_formula_metadata_present =
            contribution.scaling.has_value() || !contribution.scale_usages.empty();
        summary.scaling_law_metadata_present = summary.parameter_formula_metadata_present;
        if (summary.parameter_formula_metadata_present) {
            summary.stabilization_theorem_id =
                "ContributionDescriptor::ScalingDescriptor";
        }
        summary.regime_metadata_present = contribution.transport_character.has_value();
        summary.violation_count = 0u;
        stampForwardedEvidence(summary,
                               request,
                               AnalysisSummaryKind::StabilizationAdequacy,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::InferredFromFormAndSpaces);
        result.produced_summaries.stabilization_adequacy.push_back(
            std::move(summary));
    }
    if (!result.produced_summaries.stabilization_adequacy.empty()) {
        result.status = SummaryProductionStatus::Produced;
        result.message =
            "stabilization-adequacy summary inferred from scoped contribution descriptors";
        return result;
    }
    return forwardPendingOrUnavailable(request,
                                       *this,
                                       "AssemblyAccess::stabilizationAdequacy",
                                       assembly.evidencePending());
}

std::string InitialCompatibilityProducer::producerId() const
{
    return "InitialCompatibilityProducer";
}

std::vector<AnalysisSummaryKind> InitialCompatibilityProducer::producedKinds() const
{
    return {AnalysisSummaryKind::InitialCompatibility};
}

bool InitialCompatibilityProducer::canProduce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::InitialCompatibility;
}

SummaryProductionResult InitialCompatibilityProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = assembly.initialCompatibility(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "assembly backend provided initial-compatibility summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::InitialCompatibility,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.initial_compatibility.push_back(
            std::move(*summary));
        return result;
    }

    const auto* constraints = context.constraintSummary();
    const bool has_constraint_scope =
        constraints != nullptr && !constraints->constrained_sets.empty();
    const bool has_boundary_scope = !context.bcDescriptors().empty();
    if (!has_constraint_scope && !has_boundary_scope) {
        return forwardPendingOrUnavailable(request,
                                           *this,
                                           "AssemblyAccess::initialCompatibility",
                                           assembly.evidencePending());
    }

    InitialCompatibilitySummary summary;
    summary.compatibility_scope = request.scope_id.empty()
        ? "FE constraint and boundary metadata"
        : request.scope_id;
    summary.algebraic_constraint_metadata_present = has_constraint_scope;
    summary.boundary_constraint_metadata_present = has_boundary_scope;
    summary.checked_constraint_family_count = has_constraint_scope
        ? static_cast<std::uint64_t>(constraints->constrained_sets.size())
        : 0u;
    summary.checked_boundary_condition_count =
        static_cast<std::uint64_t>(context.bcDescriptors().size());
    summary.invariant_domain_variables = request.variables;
    if (summary.invariant_domain_variables.empty()) {
        for (const auto& field : context.fieldDescriptors()) {
            appendUnique(summary.invariant_domain_variables,
                         VariableKey::field(field.field_id));
        }
    }
    stampForwardedEvidence(summary,
                           request,
                           AnalysisSummaryKind::InitialCompatibility,
                           context,
                           mesh,
                           *this,
                           EvidenceProvenance::InferredFromBoundaryConditions);
    auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                             "initial-compatibility metadata inferred from constraints and boundary descriptors");
    result.produced_summaries.initial_compatibility.push_back(std::move(summary));
    return result;
}

std::string LocalStencilProducer::producerId() const
{
    return "LocalStencilProducer";
}

std::vector<AnalysisSummaryKind> LocalStencilProducer::producedKinds() const
{
    return {AnalysisSummaryKind::LocalStencil};
}

bool LocalStencilProducer::canProduce(const AnalysisSummaryRequest& request,
                                      const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::LocalStencil;
}

SummaryProductionResult LocalStencilProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = assembly.localStencil(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "assembly backend provided local-stencil summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::LocalStencil,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.local_stencils.push_back(std::move(*summary));
        return result;
    }
    return forwardPendingOrUnavailable(request,
                                       *this,
                                       "AssemblyAccess::localStencil",
                                       assembly.evidencePending());
}

std::string NumericalErrorBudgetProducer::producerId() const
{
    return "NumericalErrorBudgetProducer";
}

std::vector<AnalysisSummaryKind> NumericalErrorBudgetProducer::producedKinds() const
{
    return {AnalysisSummaryKind::NumericalErrorBudget};
}

bool NumericalErrorBudgetProducer::canProduce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::NumericalErrorBudget;
}

SummaryProductionResult NumericalErrorBudgetProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = assembly.numericalErrorBudget(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "assembly backend provided numerical-error-budget summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::NumericalErrorBudget,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.numerical_error_budgets.push_back(
            std::move(*summary));
        return result;
    }

    const auto* options = context.solverOptions();
    if (options == nullptr) {
        return forwardPendingOrUnavailable(request,
                                           *this,
                                           "AssemblyAccess::numericalErrorBudget",
                                           assembly.evidencePending());
    }

    NumericalErrorBudgetSummary budget;
    budget.budget_id = request.scope_id.empty()
        ? request.request_id + ":error-budget"
        : request.scope_id;
    budget.block = requestTargetBlock(request);
    budget.variables = requestVariablesOrBlockVariables(request);
    const Real eps = std::numeric_limits<Real>::epsilon();
    budget.matrix_norm_estimate = static_cast<Real>(1);
    budget.matrix_norm_present = false;
    budget.condition_estimate = static_cast<Real>(1);
    budget.condition_estimate_present = false;
    budget.linear_tolerance = options->rel_tol;
    budget.linear_tolerance_present = true;
    budget.verification_tolerance = options->abs_tol;
    budget.verification_tolerance_present = options->abs_tol > Real{};
    budget.machine_epsilon_amplification = eps;
    budget.expected_absolute_floor = static_cast<Real>(100) * eps;
    budget.expected_relative_floor = static_cast<Real>(100) * eps;
    budget.recommended_verification_tolerance =
        std::max({budget.expected_absolute_floor,
                  budget.linear_tolerance,
                  Real{1.0e-14}});
    budget.recommended_tolerance_present = true;
    budget.adequacy_class = budget.verification_tolerance_present
        ? ToleranceAdequacyClass::Reasonable
        : ToleranceAdequacyClass::Inconclusive;
    budget.reason =
        budget.verification_tolerance_present
            ? "solver tolerance metadata is present; assembled matrix scale was not available"
            : "solver tolerance metadata is present but no verification tolerance was supplied";
    stampForwardedEvidence(budget,
                           request,
                           AnalysisSummaryKind::NumericalErrorBudget,
                           context,
                           mesh,
                           *this,
                           EvidenceProvenance::InferredFromSolverSettings);
    auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                             "numerical-error-budget metadata inferred from solver options");
    result.produced_summaries.numerical_error_budgets.push_back(std::move(budget));
    return result;
}

std::string MeshFamilyProducer::producerId() const { return "MeshFamilyProducer"; }

std::vector<AnalysisSummaryKind> MeshFamilyProducer::producedKinds() const
{
    return {AnalysisSummaryKind::MeshGeometryQuality};
}

bool MeshFamilyProducer::canProduce(const AnalysisSummaryRequest& request,
                                    const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::MeshGeometryQuality;
}

SummaryProductionResult MeshFamilyProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess&,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = mesh.meshGeometryQuality(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "mesh backend provided geometry quality summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::MeshGeometryQuality,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::InferredFromMeshMetadata);
        result.produced_summaries.mesh_geometry_quality.push_back(std::move(*summary));
        return result;
    }

    MeshGeometryQualitySummary summary;
    summary.domain = request.domain;
    attachCommonEvidence(summary.evidence,
                         request,
                         AnalysisSummaryKind::MeshGeometryQuality,
                         context,
                         mesh,
                         *this);
    markGenerated(summary.evidence,
                  EvidenceProvenance::InferredFromMeshMetadata);
    for (const auto& field : context.fieldDescriptors()) {
        if (field.domain != request.domain && request.domain != DomainKind::Cell) {
            continue;
        }
        if (field.shape_regular_mesh_assumed ||
            !field.mesh_family_scope.empty()) {
            summary.shape_regular_evidence_present =
                field.shape_regular_mesh_assumed;
            summary.mesh_family_scope = field.mesh_family_scope;
            summary.mesh_family_scope_present =
                !field.mesh_family_scope.empty();
            summary.evidence.mesh_family_scope = field.mesh_family_scope;
            break;
        }
    }
    if (!summary.shape_regular_evidence_present &&
        !summary.mesh_family_scope_present) {
        return forwardPendingOrUnavailable(request,
                                           *this,
                                           "MeshAccess::meshGeometryQuality",
                                           mesh.evidencePending());
    }
    auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                             "mesh-family metadata inferred from space descriptors");
    result.produced_summaries.mesh_geometry_quality.push_back(std::move(summary));
    return result;
}

std::string BoundarySymbolProducer::producerId() const
{
    return "BoundarySymbolProducer";
}

std::vector<AnalysisSummaryKind> BoundarySymbolProducer::producedKinds() const
{
    return {AnalysisSummaryKind::BoundarySymbol};
}

bool BoundarySymbolProducer::canProduce(const AnalysisSummaryRequest& request,
                                        const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::BoundarySymbol;
}

SummaryProductionResult BoundarySymbolProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    if (auto summary = assembly.boundarySymbol(request)) {
        auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                                 "assembly backend provided boundary-symbol summary");
        stampForwardedEvidence(*summary,
                               request,
                               AnalysisSummaryKind::BoundarySymbol,
                               context,
                               mesh,
                               *this,
                               EvidenceProvenance::GeneratedFromAssembledMatrix);
        result.produced_summaries.boundary_symbols.push_back(std::move(*summary));
        return result;
    }

    BoundarySymbolSummary summary;
    summary.block = requestTargetBlock(request);
    attachCommonEvidence(summary.evidence,
                         request,
                         AnalysisSummaryKind::BoundarySymbol,
                         context,
                         mesh,
                         *this);
    markGenerated(summary.evidence,
                  EvidenceProvenance::InferredFromBoundaryConditions);
    for (const auto& bc : context.bcDescriptors()) {
        if (bc.domain != request.domain) {
            continue;
        }
        if (!request.variables.empty() &&
            std::find(request.variables.begin(),
                      request.variables.end(),
                      bc.primary_variable) == request.variables.end()) {
            continue;
        }
        ++summary.boundary_condition_count;
        summary.block.domain = bc.domain;
        summary.block.marker = bc.boundary_marker;
        summary.block.test_variables.push_back(bc.primary_variable);
        summary.block.trial_variables.push_back(bc.primary_variable);
        if (bc.enforcement_kind == EnforcementKind::WeakNitsche ||
            bc.enforcement_kind == EnforcementKind::WeakPenalty) {
            summary.weak_boundary_route = bc.weak_boundary_route;
        }
        if (const auto* field = fieldForVariable(context, bc.primary_variable)) {
            summary.trace_coverage =
                summary.trace_coverage | field->trace_capabilities;
        }
    }
    if (summary.boundary_condition_count == 0u) {
        return forwardPendingOrUnavailable(request,
                                           *this,
                                           "AssemblyAccess::boundarySymbol",
                                           assembly.evidencePending());
    }
    auto result = makeResult(request, *this, SummaryProductionStatus::Produced,
                             "boundary-symbol scope inferred from BC descriptors and trace capabilities");
    result.produced_summaries.boundary_symbols.push_back(std::move(summary));
    return result;
}

std::string InvariantDomainProducer::producerId() const
{
    return "InvariantDomainProducer";
}

std::vector<AnalysisSummaryKind> InvariantDomainProducer::producedKinds() const
{
    return {AnalysisSummaryKind::InvariantDomain};
}

bool InvariantDomainProducer::canProduce(const AnalysisSummaryRequest& request,
                                         const ProblemAnalysisContext&) const
{
    return request.summary_kind == AnalysisSummaryKind::InvariantDomain;
}

SummaryProductionResult InvariantDomainProducer::produce(
    const AnalysisSummaryRequest& request,
    const ProblemAnalysisContext& context,
    const AssemblyAccess&,
    const MeshAccess& mesh,
    const SolverAccess&) const
{
    auto result = makeResult(request, *this, SummaryProductionStatus::Unavailable,
                             "no invariant-domain descriptor covered request");
    for (const auto& record : context.formulationRecords()) {
        for (const auto& descriptor : record.invariant_domain_descriptors) {
            if (!request.scope_id.empty() &&
                descriptor.invariant_set_id != request.scope_id) {
                continue;
            }
            if (!request.variables.empty() &&
                !variableSetCoversAll(descriptor.variables, request.variables)) {
                continue;
            }
            InvariantDomainSummary summary;
            summary.invariant_set_id = descriptor.invariant_set_id;
            summary.variables = descriptor.variables;
            if (descriptor.lower_bound) {
                summary.lower_bound = *descriptor.lower_bound;
                summary.lower_bound_active = true;
            }
            if (descriptor.upper_bound) {
                summary.upper_bound = *descriptor.upper_bound;
                summary.upper_bound_active = true;
            }
            if (descriptor.excluded_value) {
                summary.excluded_value = *descriptor.excluded_value;
                summary.excluded_value_active = true;
            }
            if (descriptor.cfl_estimate) {
                summary.cfl_estimate = *descriptor.cfl_estimate;
                summary.cfl_estimate_present = true;
            }
            if (descriptor.accepted_cfl_bound) {
                summary.accepted_cfl_bound = *descriptor.accepted_cfl_bound;
                summary.accepted_cfl_bound_present = true;
            }
            if (descriptor.wave_speed_bound) {
                summary.wave_speed_bound = *descriptor.wave_speed_bound;
                summary.wave_speed_bound_present = true;
            }
            summary.time_step_scope = descriptor.time_step_scope;
            summary.mesh_size_scope = descriptor.mesh_size_scope;
            summary.limiter_evidence_present = descriptor.limiter_evidence_present;
            summary.cfl_condition_satisfied = descriptor.cfl_condition_satisfied;
            summary.ssp_time_discretization_evidence_present =
                descriptor.ssp_time_discretization_evidence_present;
            summary.source_admissibility_evidence_present =
                descriptor.source_admissibility_evidence_present;
            summary.low_order_invariant_domain_evidence_present =
                descriptor.low_order_invariant_domain_evidence_present;
            summary.convex_limiting_evidence_present =
                descriptor.convex_limiting_evidence_present;
            summary.spatial_monotonicity_evidence_present =
                descriptor.spatial_monotonicity_evidence_present;
            summary.mass_positivity_evidence_present =
                descriptor.mass_positivity_evidence_present;
            summary.invariant_domain_theorem_id = descriptor.theorem_id;
            attachCommonEvidence(summary.evidence,
                                 request,
                                 AnalysisSummaryKind::InvariantDomain,
                                 context,
                                 mesh,
                                 *this);
            markGenerated(summary.evidence,
                          EvidenceProvenance::InferredFromFormAndSpaces);
            result.produced_summaries.invariant_domains.push_back(std::move(summary));
        }
    }
    if (!result.produced_summaries.invariant_domains.empty()) {
        result.status = SummaryProductionStatus::Produced;
        result.message =
            "invariant-domain summary inferred from formulation descriptors";
    }
    return result;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
