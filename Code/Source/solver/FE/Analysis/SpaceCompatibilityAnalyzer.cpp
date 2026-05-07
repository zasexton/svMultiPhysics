/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/SpaceCompatibilityAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/BoundaryConditionDescriptor.h"

#include <algorithm>
#include <cmath>
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

static bool containsVariable(const std::vector<VariableKey>& values,
                             const VariableKey& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

static bool claimContainsAllVariables(const PropertyClaim& claim,
                                      const std::vector<VariableKey>& variables)
{
    return std::all_of(variables.begin(), variables.end(),
                       [&](const VariableKey& variable) {
                           return containsVariable(claim.variables, variable);
                       });
}

static bool hasStabilizedInfSupEvidence(const ProblemAnalysisReport& report,
                                        const std::vector<VariableKey>& variables)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.inf_sup_class &&
            *claim.inf_sup_class == InfSupClass::StabilizedSurrogate &&
            claimContainsAllVariables(claim, variables)) {
            return true;
        }
    }
    return false;
}

static bool hasCertifiedInfSupEvidence(const ProblemAnalysisReport& report,
                                       const std::vector<VariableKey>& variables)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified &&
            claimContainsAllVariables(claim, variables)) {
            return true;
        }
    }
    return false;
}

static bool hasCertifiedCompatibleComplexEvidence(
    const ProblemAnalysisReport& report,
    const std::vector<VariableKey>& variables)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::CompatibleComplexStructure &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified &&
            claimContainsAllVariables(claim, variables)) {
            return true;
        }
    }
    return false;
}

static void emitCompatibleComplexClaim(ProblemAnalysisReport& report,
                                       std::vector<VariableKey> variables,
                                       bool exact_sequence,
                                       std::optional<bool> commuting_projection,
                                       bool bounded_projection_certificate,
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
    } else if (commuting_projection.has_value() && *commuting_projection &&
               bounded_projection_certificate) {
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = CertificationClass::Certified;
        claim.space_compatibility_class = SpaceCompatibilityClass::Compatible;
        claim.description =
            "Compatible-complex metadata preserves exact-sequence requirements with bounded cochain-projection evidence";
    } else {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::NotCertified;
        claim.space_compatibility_class = SpaceCompatibilityClass::WeaklyCompatible;
        claim.description =
            "Exact sequence is present but bounded commuting-projection, mesh-family, or theorem evidence is missing";
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
            const bool projection_bound_valid =
                summary.projection_bound_present &&
                numeric::finitePositive(summary.projection_bound);
            const bool bounded_projection_certificate =
                summary.bounded_cochain_projection_evidence_present &&
                projection_bound_valid &&
                summary.projection_stability_metadata_present &&
                summary.mesh_family_scope_present &&
                summary.shape_regular_mesh_evidence_present &&
                !summary.compatible_complex_theorem_id.empty();
            emitCompatibleComplexClaim(
                report,
                summary.variables,
                summary.exact_sequence_compatible &&
                    summary.trace_sequence_compatible,
                summary.commuting_projection_available,
                bounded_projection_certificate,
                summary.missing_space_count,
                summary.complex_id,
                "CompatibleComplexSummary exact_sequence=" +
                    std::string(summary.exact_sequence_compatible ? "true" : "false") +
                    ", trace_sequence=" +
                    std::string(summary.trace_sequence_compatible ? "true" : "false") +
                    ", commuting_projection=" +
                    std::string(summary.commuting_projection_available ? "true" : "false") +
                    ", bounded_projection=" +
                    std::string(summary.bounded_cochain_projection_evidence_present ? "true" : "false") +
                    ", projection_bound=" +
                    std::to_string(summary.projection_bound) +
                    ", projection_stability=" +
                    std::string(summary.projection_stability_metadata_present ? "true" : "false") +
                    ", mesh_family_scope=" +
                    std::string(summary.mesh_family_scope_present ? "true" : "false") +
                    ", shape_regular_mesh=" +
                    std::string(summary.shape_regular_mesh_evidence_present ? "true" : "false") +
                    ", theorem='" + summary.compatible_complex_theorem_id + "'" +
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
            false,
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
        if (sc->variables.size() >= 2u &&
            sc->variables[0].kind == VariableKind::FieldComponent &&
            sc->variables[1].kind == VariableKind::FieldComponent) {
            momentum_fd = context.fieldDescriptor(sc->variables[0].field_id);
            constraint_fd = context.fieldDescriptor(sc->variables[1].field_id);
        }

        if (!momentum_fd || !constraint_fd) {
            for (const auto& vk : sc->variables) {
                if (vk.kind != VariableKind::FieldComponent) continue;
                const auto* fd = context.fieldDescriptor(vk.field_id);
                if (!fd) continue;

                // Fallback only for legacy claims that do not encode pair order.
                if (!momentum_fd ||
                    fd->value_dimension > momentum_fd->value_dimension) {
                    constraint_fd = momentum_fd;
                    momentum_fd = fd;
                } else if (!constraint_fd) {
                    constraint_fd = fd;
                }
            }
        }

        if (!momentum_fd || !constraint_fd) continue;

        // Skip if space families are unknown
        if (momentum_fd->space_family == SpaceFamily::Unknown &&
            constraint_fd->space_family == SpaceFamily::Unknown) {
            continue;
        }

        if (momentum_fd->space_family == SpaceFamily::H1 &&
            constraint_fd->space_family == SpaceFamily::H1) {
            PropertyClaim claim;
            claim.kind = PropertyKind::SpaceCompatibility;
            claim.variables = sc->variables;
            claim.claim_origin = "SpaceCompatibilityAnalyzer";
            claim.domain = sc->domain;
            claim.tested_block_id = sc->tested_block_id;
            claim.estimate_scope = sc->estimate_scope;
            const bool taylor_hood_like =
                momentum_fd->polynomial_order > constraint_fd->polynomial_order;
            const bool certified_infsup =
                hasCertifiedInfSupEvidence(report, sc->variables);
            const bool certified_complex =
                hasCertifiedCompatibleComplexEvidence(report, sc->variables);
            const bool stabilized =
                hasStabilizedInfSupEvidence(report, sc->variables);

            if (certified_infsup || certified_complex) {
                claim.status = PropertyStatus::Preserved;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Certified;
                claim.space_compatibility_class = SpaceCompatibilityClass::Compatible;
                if (certified_infsup) {
                    claim.description =
                        "Mixed H1/H1 space pair is backed by certified inf-sup evidence";
                    claim.addEvidence("SpaceCompatibilityAnalyzer",
                        "Certified InfSupCondition claim covers this mixed pair");
                } else {
                    claim.description =
                        "Mixed H1/H1 space pair is backed by certified compatible-complex evidence";
                    claim.addEvidence("SpaceCompatibilityAnalyzer",
                        "Certified CompatibleComplexStructure claim covers this mixed pair");
                }
            } else if (taylor_hood_like) {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.space_compatibility_class = SpaceCompatibilityClass::WeaklyCompatible;
                claim.description =
                    "Mixed H1/H1 space pair has higher-order primary field evidence but lacks certified inf-sup or compatible-complex evidence";
                claim.addEvidence("SpaceCompatibilityAnalyzer",
                    "H1 primary order " +
                    std::to_string(momentum_fd->polynomial_order) +
                    " exceeds H1 constraint order " +
                    std::to_string(constraint_fd->polynomial_order),
                    AnalysisConfidence::Medium);
            } else if (stabilized) {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.space_compatibility_class = SpaceCompatibilityClass::WeaklyCompatible;
                claim.description =
                    "Equal-order H1/H1 mixed pair relies on stabilization surrogate evidence";
                claim.addEvidence("SpaceCompatibilityAnalyzer",
                    "InfSupAnalyzer reported a stabilization surrogate for this mixed pair",
                    AnalysisConfidence::Medium);
            } else {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.space_compatibility_class = SpaceCompatibilityClass::Incompatible;
                claim.description =
                    "Equal-order or reversed-order H1/H1 mixed pair lacks inf-sup/stabilization evidence";
                claim.addEvidence("SpaceCompatibilityAnalyzer",
                    "H1/H1 mixed compatibility requires a known stable order pair, stabilization, or numeric inf-sup evidence",
                    AnalysisConfidence::Medium);
            }
            report.claims.push_back(std::move(claim));
            continue;
        }

        // H(div) velocity with H1 pressure is not a de Rham/mixed-method
        // compatibility certificate by itself.  It needs explicit pair evidence.
        if (momentum_fd->space_family == SpaceFamily::HDiv &&
            constraint_fd->space_family == SpaceFamily::H1) {
            PropertyClaim claim;
            claim.kind = PropertyKind::SpaceCompatibility;
            claim.variables = sc->variables;
            claim.claim_origin = "SpaceCompatibilityAnalyzer";
            claim.domain = sc->domain;
            claim.tested_block_id = sc->tested_block_id;
            claim.estimate_scope = sc->estimate_scope;
            const bool certified_infsup =
                hasCertifiedInfSupEvidence(report, sc->variables);
            const bool certified_complex =
                hasCertifiedCompatibleComplexEvidence(report, sc->variables);
            if (certified_infsup || certified_complex) {
                claim.status = PropertyStatus::Preserved;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Certified;
                claim.space_compatibility_class = SpaceCompatibilityClass::Compatible;
                claim.description =
                    "Mixed HDiv/H1 space pair is backed by certified compatibility evidence";
                claim.addEvidence("SpaceCompatibilityAnalyzer",
                    certified_infsup
                        ? "Certified InfSupCondition claim covers this HDiv/H1 pair"
                        : "Certified CompatibleComplexStructure claim covers this HDiv/H1 pair");
            } else {
                claim.status = PropertyStatus::Unknown;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.space_compatibility_class = SpaceCompatibilityClass::Unknown;
                claim.description =
                    "Mixed system has HDiv/H1 space pair (" + momentum_fd->name +
                    "/" + constraint_fd->name +
                    ") without explicit inf-sup or compatible-complex evidence";
                claim.addEvidence("SpaceCompatibilityAnalyzer",
                    "Standard divergence-compatible mixed pairs require explicit HDiv/L2-style or theorem-backed compatibility evidence",
                    AnalysisConfidence::Medium);
            }
            report.claims.push_back(std::move(claim));
            continue;
        }

        // L2 multipliers are common in divergence-conforming and DG/mixed
        // methods, but stability is a property of the particular pair and mesh
        // assumptions.  Require a certified inf-sup or compatible-complex
        // claim before marking the pair compatible.
        if (constraint_fd->space_family == SpaceFamily::L2) {
            PropertyClaim claim;
            claim.kind = PropertyKind::SpaceCompatibility;
            claim.variables = sc->variables;
            claim.claim_origin = "SpaceCompatibilityAnalyzer";
            claim.domain = sc->domain;
            claim.tested_block_id = sc->tested_block_id;
            claim.estimate_scope = sc->estimate_scope;
            const bool certified_infsup =
                hasCertifiedInfSupEvidence(report, sc->variables);
            const bool certified_complex =
                hasCertifiedCompatibleComplexEvidence(report, sc->variables);
            if (certified_infsup || certified_complex) {
                claim.status = PropertyStatus::Preserved;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Certified;
                claim.space_compatibility_class = SpaceCompatibilityClass::Compatible;
                claim.description =
                    "Mixed " +
                    std::string(toString(momentum_fd->space_family)) +
                    "/L2 space pair is backed by certified compatibility evidence";
                claim.addEvidence("SpaceCompatibilityAnalyzer",
                    certified_infsup
                        ? "Certified InfSupCondition claim covers this L2 multiplier pair"
                        : "Certified CompatibleComplexStructure claim covers this L2 multiplier pair");
            } else {
                claim.status = PropertyStatus::Unknown;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.space_compatibility_class = SpaceCompatibilityClass::Unknown;
                claim.description =
                    "Mixed system has " +
                    std::string(toString(momentum_fd->space_family)) +
                    "/L2 space pair (" + momentum_fd->name + "/" +
                    constraint_fd->name +
                    ") without explicit inf-sup or compatible-complex evidence";
                claim.addEvidence("SpaceCompatibilityAnalyzer",
                    "L2 multiplier compatibility requires theorem-backed stable-pair, Fortin, or compatible-complex evidence",
                    AnalysisConfidence::Medium);
            }
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
        claim.domain = sc->domain;
        claim.tested_block_id = sc->tested_block_id;
        claim.estimate_scope = sc->estimate_scope;
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
