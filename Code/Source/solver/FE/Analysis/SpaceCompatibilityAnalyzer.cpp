/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/SpaceCompatibilityAnalyzer.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/BoundaryConditionDescriptor.h"

#include <algorithm>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

std::string SpaceCompatibilityAnalyzer::name() const {
    return "SpaceCompatibilityAnalyzer";
}

/// Map a TraceKind to the TraceCapabilityFlags it requires
static TraceCapabilityFlags requiredTraceCapability(TraceKind tk) noexcept {
    switch (tk) {
    case TraceKind::Value:
        return TraceCapabilityFlags::Value;
    case TraceKind::NormalComponent:
        return TraceCapabilityFlags::NormalComponent;
    case TraceKind::TangentialComponent:
        return TraceCapabilityFlags::TangentialComponent;
    case TraceKind::NormalFlux:
        return TraceCapabilityFlags::NormalFlux;
    case TraceKind::Flux:
        return TraceCapabilityFlags::NormalFlux;  // flux requires normal flux trace
    case TraceKind::Mixed:
        // Robin requires both value and flux trace
        return TraceCapabilityFlags::Value | TraceCapabilityFlags::NormalFlux;
    case TraceKind::AlgebraicRelation:
        return TraceCapabilityFlags::None;  // no trace capability needed
    }
    return TraceCapabilityFlags::None;
}

static void appendUnique(std::vector<VariableKey>& values, const VariableKey& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

static void emitCompatibleComplexClaim(ProblemAnalysisReport& report,
                                       std::vector<VariableKey> variables,
                                       bool exact_sequence,
                                       std::optional<bool> commuting_projection,
                                       std::uint64_t missing_space_count,
                                       std::string label,
                                       std::string evidence)
{
    PropertyClaim claim;
    claim.kind = PropertyKind::CompatibleComplexStructure;
    claim.variables = std::move(variables);
    claim.exact_sequence_compatible = exact_sequence;
    if (commuting_projection.has_value()) {
        claim.commuting_projection_available = *commuting_projection;
    }
    claim.claim_origin = "SpaceCompatibilityAnalyzer";
    claim.estimate_scope = std::move(label);

    if (!exact_sequence || missing_space_count > 0u) {
        claim.status = PropertyStatus::Violated;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = CertificationClass::Violated;
        claim.space_compatibility_class = SpaceCompatibilityClass::Incompatible;
        claim.description =
            "Compatible-complex metadata violates exact-sequence requirements";
    } else if (commuting_projection.value_or(true)) {
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = CertificationClass::Certified;
        claim.space_compatibility_class = SpaceCompatibilityClass::Compatible;
        claim.description =
            "Compatible-complex metadata preserves exact-sequence requirements";
    } else {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::NotCertified;
        claim.space_compatibility_class = SpaceCompatibilityClass::WeaklyCompatible;
        claim.description =
            "Exact sequence is present but commuting-projection evidence is missing";
    }

    claim.addEvidence("SpaceCompatibilityAnalyzer", std::move(evidence),
                      claim.confidence);
    report.claims.push_back(std::move(claim));
}

void SpaceCompatibilityAnalyzer::run(const ProblemAnalysisContext& context,
                                     ProblemAnalysisReport& report) const
{
    const auto& bcs = context.bcDescriptors();
    const auto& field_descs = context.fieldDescriptors();

    // =====================================================================
    // Check compatible-complex metadata from explicit summaries and field
    // descriptors. This stays physics-agnostic: it only consumes sequence,
    // trace, and commuting-projection metadata.
    // =====================================================================

    if (const auto* summaries = context.analysisSummaries()) {
        for (const auto& summary : summaries->compatible_complexes) {
            emitCompatibleComplexClaim(
                report,
                summary.variables,
                summary.exact_sequence_compatible &&
                    summary.trace_sequence_compatible,
                summary.commuting_projection_available,
                summary.missing_space_count,
                summary.complex_id,
                "CompatibleComplexSummary exact_sequence=" +
                    std::string(summary.exact_sequence_compatible ? "true" : "false") +
                    ", trace_sequence=" +
                    std::string(summary.trace_sequence_compatible ? "true" : "false") +
                    ", commuting_projection=" +
                    std::string(summary.commuting_projection_available ? "true" : "false") +
                    ", missing_spaces=" +
                    std::to_string(summary.missing_space_count));
        }
    }

    std::vector<VariableKey> exact_sequence_fields;
    for (const auto& fd : field_descs) {
        if (!fd.has_exact_sequence_structure) continue;
        appendUnique(exact_sequence_fields, VariableKey::field(fd.field_id));
    }
    if (!exact_sequence_fields.empty()) {
        emitCompatibleComplexClaim(
            report,
            std::move(exact_sequence_fields),
            true,
            std::nullopt,
            0u,
            "field-descriptor-exact-sequence",
            "FieldDescriptor metadata reports exact-sequence structure");
    }

    // =====================================================================
    // Check BC trace_kind vs field trace_capabilities
    // =====================================================================

    for (const auto& bc : bcs) {
        if (bc.primary_variable.kind != VariableKind::FieldComponent) continue;

        const auto* fd = context.fieldDescriptor(bc.primary_variable.field_id);
        if (!fd) continue;

        // Skip check when trace_capabilities is None (unknown/not populated)
        if (fd->trace_capabilities == TraceCapabilityFlags::None) continue;

        TraceCapabilityFlags required = requiredTraceCapability(bc.trace_kind);
        if (required == TraceCapabilityFlags::None) continue;

        // Check if the field's trace capabilities support the BC's trace kind
        bool supported = (fd->trace_capabilities & required) == required;

        if (!supported) {
            PropertyClaim claim;
            claim.kind = PropertyKind::SpaceCompatibility;
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.space_compatibility_class = SpaceCompatibilityClass::Incompatible;
            claim.field = fd->field_id;
            claim.variables.push_back(bc.primary_variable);
            claim.description =
                "BC trace kind '" + std::string(toString(bc.trace_kind)) +
                "' is incompatible with field '" + fd->name +
                "' (space family: " + std::string(toString(fd->space_family)) +
                "): field does not support required trace capability";
            claim.claim_origin = "SpaceCompatibilityAnalyzer";
            claim.addEvidence("SpaceCompatibilityAnalyzer",
                "Field trace_capabilities=0x" +
                std::to_string(static_cast<std::uint32_t>(fd->trace_capabilities)) +
                " does not include required=0x" +
                std::to_string(static_cast<std::uint32_t>(required)) +
                " for BC on " + bc.source);
            report.claims.push_back(std::move(claim));
        }
    }

    // =====================================================================
    // Check mixed saddle-point space compatibility
    // =====================================================================

    auto saddle_claims = report.claimsOfKind(PropertyKind::MixedSaddlePoint);
    for (const auto* sc : saddle_claims) {
        // Identify momentum and constraint variables from the saddle-point claim
        // The MixedOperatorAnalyzer puts all variables in the claim;
        // we need to identify which has coercive diagonal and which doesn't.
        // Use field descriptors to distinguish.

        const FieldDescriptor* momentum_fd = nullptr;
        const FieldDescriptor* constraint_fd = nullptr;

        for (const auto& vk : sc->variables) {
            if (vk.kind != VariableKind::FieldComponent) continue;
            const auto* fd = context.fieldDescriptor(vk.field_id);
            if (!fd) continue;

            // Heuristic: higher value_dimension is momentum, lower is constraint
            if (!momentum_fd || fd->value_dimension > momentum_fd->value_dimension) {
                constraint_fd = momentum_fd;
                momentum_fd = fd;
            } else if (!constraint_fd) {
                constraint_fd = fd;
            }
        }

        if (!momentum_fd || !constraint_fd) continue;

        // Skip if space families are unknown
        if (momentum_fd->space_family == SpaceFamily::Unknown &&
            constraint_fd->space_family == SpaceFamily::Unknown) {
            continue;
        }

        // Both H1 is fine
        if (momentum_fd->space_family == SpaceFamily::H1 &&
            constraint_fd->space_family == SpaceFamily::H1) {
            PropertyClaim claim;
            claim.kind = PropertyKind::SpaceCompatibility;
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.space_compatibility_class = SpaceCompatibilityClass::Compatible;
            claim.variables = sc->variables;
            claim.description =
                "Mixed system space pair is compatible: H1/H1 ("
                + momentum_fd->name + " order "
                + std::to_string(momentum_fd->polynomial_order) + " / "
                + constraint_fd->name + " order "
                + std::to_string(constraint_fd->polynomial_order) + ")";
            claim.claim_origin = "SpaceCompatibilityAnalyzer";
            claim.addEvidence("SpaceCompatibilityAnalyzer",
                "Both fields use H1 space family");
            report.claims.push_back(std::move(claim));
            continue;
        }

        // H1 pressure with HDiv velocity needs specific handling
        if (momentum_fd->space_family == SpaceFamily::HDiv &&
            constraint_fd->space_family == SpaceFamily::H1) {
            PropertyClaim claim;
            claim.kind = PropertyKind::SpaceCompatibility;
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.space_compatibility_class = SpaceCompatibilityClass::WeaklyCompatible;
            claim.variables = sc->variables;
            claim.description =
                "Mixed system has HDiv/" + std::string(toString(constraint_fd->space_family)) +
                " space pair (" + momentum_fd->name + "/" + constraint_fd->name +
                "): compatible but requires specific handling (e.g., Raviart-Thomas)";
            claim.claim_origin = "SpaceCompatibilityAnalyzer";
            claim.addEvidence("SpaceCompatibilityAnalyzer",
                "HDiv velocity + H1 pressure needs compatible interpolation",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
            continue;
        }

        // L2 constraint with non-L2 momentum -- weakly compatible (DG/mixed methods)
        if (constraint_fd->space_family == SpaceFamily::L2) {
            PropertyClaim claim;
            claim.kind = PropertyKind::SpaceCompatibility;
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.space_compatibility_class = SpaceCompatibilityClass::WeaklyCompatible;
            claim.variables = sc->variables;
            claim.description =
                "Mixed system has " + std::string(toString(momentum_fd->space_family)) +
                "/L2 space pair (" + momentum_fd->name + "/" + constraint_fd->name +
                "): weakly compatible (DG or mixed method)";
            claim.claim_origin = "SpaceCompatibilityAnalyzer";
            claim.addEvidence("SpaceCompatibilityAnalyzer",
                "L2 constraint field in saddle-point system",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
            continue;
        }

        // Other combinations: unknown compatibility
        PropertyClaim claim;
        claim.kind = PropertyKind::SpaceCompatibility;
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Low;
        claim.space_compatibility_class = SpaceCompatibilityClass::Unknown;
        claim.variables = sc->variables;
        claim.description =
            "Mixed system space pair compatibility unknown: " +
            std::string(toString(momentum_fd->space_family)) + "/" +
            std::string(toString(constraint_fd->space_family)) +
            " (" + momentum_fd->name + "/" + constraint_fd->name + ")";
        claim.claim_origin = "SpaceCompatibilityAnalyzer";
        claim.addEvidence("SpaceCompatibilityAnalyzer",
            "Unrecognized space family combination",
            AnalysisConfidence::Low);
        report.claims.push_back(std::move(claim));
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
