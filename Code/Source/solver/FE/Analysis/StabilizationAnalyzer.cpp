/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/StabilizationAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormStructureAnalyzer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

struct EffectiveStabilizationRequirements {
    bool requires_peclet_evidence{false};
    bool requires_cfl_evidence{false};
    bool requires_trace_penalty_bound{false};
    bool requires_inf_sup_surrogate{false};
    bool requires_residual_consistency{true};
    bool requires_adjoint_consistency{false};
    bool requires_mass_lumping{false};
    bool requires_limiter_bounds{false};
    bool requirement_metadata_present{false};
};

bool containsCaseInsensitive(const std::string& text, const char* token)
{
    auto lower_text = text;
    auto lower_token = std::string(token);
    for (auto& ch : lower_text) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    for (auto& ch : lower_token) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return lower_text.find(lower_token) != std::string::npos;
}

EffectiveStabilizationRequirements effectiveRequirements(
    const StabilizationAdequacySummary& summary)
{
    EffectiveStabilizationRequirements req;
    req.requires_peclet_evidence = summary.requires_peclet_evidence;
    req.requires_cfl_evidence = summary.requires_cfl_evidence;
    req.requires_trace_penalty_bound = summary.requires_trace_penalty_bound;
    req.requires_inf_sup_surrogate = summary.requires_inf_sup_surrogate;
    req.requires_residual_consistency = summary.requires_residual_consistency;
    req.requires_adjoint_consistency = summary.requires_adjoint_consistency;
    req.requires_mass_lumping = summary.requires_mass_lumping;
    req.requires_limiter_bounds = summary.requires_limiter_bounds;
    req.requirement_metadata_present =
        summary.requirement_metadata_present ||
        summary.requires_peclet_evidence ||
        summary.requires_cfl_evidence ||
        summary.requires_trace_penalty_bound ||
        summary.requires_inf_sup_surrogate ||
        summary.requires_adjoint_consistency ||
        summary.requires_mass_lumping ||
        summary.requires_limiter_bounds;

    switch (summary.family) {
        case StabilizationFamily::StreamlineUpwind:
        case StabilizationFamily::GalerkinLeastSquares:
        case StabilizationFamily::UpwindFlux:
            req.requires_peclet_evidence = true;
            req.requires_cfl_evidence = true;
            req.requirement_metadata_present = true;
            break;
        case StabilizationFamily::Penalty:
        case StabilizationFamily::Nitsche:
        case StabilizationFamily::ContinuousInteriorPenalty:
        case StabilizationFamily::Trace:
        case StabilizationFamily::GhostPenalty:
            req.requires_trace_penalty_bound = true;
            req.requirement_metadata_present = true;
            break;
        case StabilizationFamily::Constraint:
            req.requires_inf_sup_surrogate = true;
            req.requirement_metadata_present = true;
            break;
        case StabilizationFamily::MassLumping:
            req.requires_mass_lumping = true;
            req.requirement_metadata_present = true;
            break;
        case StabilizationFamily::Limiter:
            req.requires_limiter_bounds = true;
            req.requirement_metadata_present = true;
            break;
        case StabilizationFamily::Unknown:
            if (containsCaseInsensitive(summary.method_family, "supg") ||
                containsCaseInsensitive(summary.method_family, "gls") ||
                containsCaseInsensitive(summary.method_family, "upwind")) {
                req.requires_peclet_evidence = true;
                req.requires_cfl_evidence = true;
                req.requirement_metadata_present = true;
            } else if (containsCaseInsensitive(summary.method_family, "nitsche") ||
                       containsCaseInsensitive(summary.method_family, "penalty") ||
                       containsCaseInsensitive(summary.method_family, "cip") ||
                       containsCaseInsensitive(summary.method_family, "ghost") ||
                       containsCaseInsensitive(summary.method_family, "trace")) {
                req.requires_trace_penalty_bound = true;
                req.requirement_metadata_present = true;
            }
            break;
        case StabilizationFamily::LeastSquares:
        case StabilizationFamily::Other:
            req.requirement_metadata_present = true;
            break;
    }

    return req;
}

void emitAdequacyClaims(const ProblemAnalysisContext& context,
                        ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->stabilization_adequacy) {
        const auto requirements = effectiveRequirements(summary);
        const bool theorem_scoped = !summary.stabilization_theorem_id.empty();
        const bool norm_scoped =
            summary.stability_norm_metadata_present &&
            !summary.stability_norm_id.empty();
        const bool parameter_bounds_valid =
            summary.stabilization_parameter_bounds_present &&
            numeric::finitePositiveOrdered(
                summary.minimum_stabilization_parameter,
                summary.maximum_stabilization_parameter);
        const bool consistency_order_valid =
            summary.consistency_order_metadata_present &&
            summary.consistency_order >= 0;
        const bool peclet_regime_valid =
            summary.peclet_condition_satisfied &&
            summary.peclet_estimate_present &&
            summary.peclet_regime_bounds_present &&
            numeric::finiteNonnegative(summary.peclet_estimate) &&
            numeric::finiteNonnegative(summary.peclet_regime_lower_bound) &&
            numeric::finite(summary.peclet_regime_upper_bound) &&
            summary.peclet_regime_lower_bound <=
                summary.peclet_regime_upper_bound &&
            summary.peclet_estimate + Real{1.0e-14} >=
                summary.peclet_regime_lower_bound &&
            summary.peclet_estimate <=
                summary.peclet_regime_upper_bound + Real{1.0e-14} &&
            !summary.peclet_scope.empty();
        const bool cfl_bound_valid =
            summary.cfl_condition_satisfied &&
            summary.cfl_estimate_present &&
            summary.accepted_cfl_bound_present &&
            numeric::finiteNonnegative(summary.cfl_estimate) &&
            numeric::finitePositive(summary.accepted_cfl_bound) &&
            summary.cfl_estimate <=
                summary.accepted_cfl_bound + Real{1.0e-14} &&
            !summary.cfl_scope.empty();
        const bool invalid_numeric_evidence =
            (summary.peclet_estimate_present &&
             !numeric::finiteNonnegative(summary.peclet_estimate)) ||
            (summary.peclet_regime_bounds_present &&
             (!numeric::finiteNonnegative(summary.peclet_regime_lower_bound) ||
              !numeric::finite(summary.peclet_regime_upper_bound) ||
              summary.peclet_regime_lower_bound >
                  summary.peclet_regime_upper_bound)) ||
            (summary.peclet_estimate_present &&
             summary.peclet_regime_bounds_present &&
             numeric::finiteNonnegative(summary.peclet_estimate) &&
             numeric::finiteNonnegative(summary.peclet_regime_lower_bound) &&
             numeric::finite(summary.peclet_regime_upper_bound) &&
             requirements.requires_peclet_evidence &&
             (summary.peclet_estimate + Real{1.0e-14} <
                  summary.peclet_regime_lower_bound ||
              summary.peclet_estimate >
                  summary.peclet_regime_upper_bound + Real{1.0e-14})) ||
            (summary.cfl_estimate_present &&
             !numeric::finiteNonnegative(summary.cfl_estimate)) ||
            (summary.accepted_cfl_bound_present &&
             !numeric::finitePositive(summary.accepted_cfl_bound)) ||
            (summary.cfl_estimate_present &&
             summary.accepted_cfl_bound_present &&
             numeric::finiteNonnegative(summary.cfl_estimate) &&
             numeric::finitePositive(summary.accepted_cfl_bound) &&
             requirements.requires_cfl_evidence &&
             summary.cfl_estimate >
                 summary.accepted_cfl_bound + Real{1.0e-14});
        const bool metadata_complete =
            summary.parameter_formula_metadata_present &&
            (!requirements.requires_residual_consistency ||
             summary.residual_consistency_evidence_present) &&
            summary.regime_metadata_present &&
            summary.method_scope_metadata_present &&
            requirements.requirement_metadata_present &&
            theorem_scoped &&
            norm_scoped &&
            parameter_bounds_valid &&
            summary.scaling_law_metadata_present &&
            consistency_order_valid &&
            summary.boundary_treatment_metadata_present &&
            (!requirements.requires_peclet_evidence || peclet_regime_valid) &&
            (!requirements.requires_cfl_evidence || cfl_bound_valid) &&
            (!requirements.requires_trace_penalty_bound ||
             (summary.trace_penalty_bound_present &&
              summary.trace_penalty_bound_valid)) &&
            (!requirements.requires_inf_sup_surrogate ||
             summary.inf_sup_surrogate_evidence_present) &&
            (!requirements.requires_adjoint_consistency ||
             summary.adjoint_consistency_evidence_present) &&
            (!requirements.requires_mass_lumping ||
             summary.mass_lumping_evidence_present) &&
            (!requirements.requires_limiter_bounds ||
             summary.limiter_bounds_evidence_present);

        PropertyClaim claim;
        claim.kind = PropertyKind::Stabilization;
        claim.domain = summary.block.domain;
        claim.variables = !summary.variables.empty()
            ? summary.variables
            : summary.block.test_variables;
        claim.tested_block_id = summary.block.operator_tag.empty()
            ? summary.stabilization_id
            : summary.block.operator_tag;
        claim.estimate_scope = summary.method_family;
        claim.claim_origin = "StabilizationAnalyzer";

        if (summary.violation_count > 0u || invalid_numeric_evidence) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Stabilization adequacy summary reports violated parameter, consistency, or theorem-required regime checks";
        } else if (metadata_complete) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.evidence_level = EvidenceLevel::CertifiedNumericTheorem;
            claim.description =
                "Stabilization adequacy is certified by theorem-scoped parameter scaling, consistency, norm, and family-specific requirement metadata";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.evidence_level = EvidenceLevel::StructuralMetadata;
            claim.description =
                "Stabilization adequacy is unknown because theorem, parameter-scaling, consistency, norm, boundary, regime, or family-specific requirement metadata is incomplete";
        }

        claim.addEvidence("StabilizationAnalyzer",
            "StabilizationAdequacySummary id='" + summary.stabilization_id +
            "', method='" + summary.method_family +
            "', requirement_metadata=" +
            std::string(requirements.requirement_metadata_present ? "true" : "false") +
            ", requires_peclet=" +
            std::string(requirements.requires_peclet_evidence ? "true" : "false") +
            ", requires_cfl=" +
            std::string(requirements.requires_cfl_evidence ? "true" : "false") +
            ", requires_trace_penalty=" +
            std::string(requirements.requires_trace_penalty_bound ? "true" : "false") +
            ", requires_inf_sup_surrogate=" +
            std::string(requirements.requires_inf_sup_surrogate ? "true" : "false") +
            ", requires_adjoint_consistency=" +
            std::string(requirements.requires_adjoint_consistency ? "true" : "false") +
            "', parameter_formula=" +
            std::string(summary.parameter_formula_metadata_present ? "true" : "false") +
            ", residual_consistency=" +
            std::string(summary.residual_consistency_evidence_present ? "true" : "false") +
            ", regime_metadata=" +
            std::string(summary.regime_metadata_present ? "true" : "false") +
            ", method_scope=" +
            std::string(summary.method_scope_metadata_present ? "true" : "false") +
            ", theorem='" + summary.stabilization_theorem_id + "'" +
            ", stability_norm_metadata=" +
            std::string(summary.stability_norm_metadata_present ? "true" : "false") +
            ", parameter_bounds=" +
            std::string(summary.stabilization_parameter_bounds_present ? "true" : "false") +
            ", parameter_min=" +
            std::to_string(summary.minimum_stabilization_parameter) +
            ", parameter_max=" +
            std::to_string(summary.maximum_stabilization_parameter) +
            ", scaling_law=" +
            std::string(summary.scaling_law_metadata_present ? "true" : "false") +
            ", consistency_order=" +
            std::to_string(summary.consistency_order) +
            ", boundary_treatment=" +
            std::string(summary.boundary_treatment_metadata_present ? "true" : "false") +
            ", peclet_ok=" +
            std::string(summary.peclet_condition_satisfied ? "true" : "false") +
            ", peclet_estimate_present=" +
            std::string(summary.peclet_estimate_present ? "true" : "false") +
            ", peclet_estimate=" +
            std::to_string(summary.peclet_estimate) +
            ", peclet_regime_bounds_present=" +
            std::string(summary.peclet_regime_bounds_present ? "true" : "false") +
            ", peclet_regime=[" +
            std::to_string(summary.peclet_regime_lower_bound) + "," +
            std::to_string(summary.peclet_regime_upper_bound) + "]" +
            ", peclet_scope='" + summary.peclet_scope + "'" +
            ", cfl_ok=" +
            std::string(summary.cfl_condition_satisfied ? "true" : "false") +
            ", cfl_estimate_present=" +
            std::string(summary.cfl_estimate_present ? "true" : "false") +
            ", cfl_estimate=" +
            std::to_string(summary.cfl_estimate) +
            ", accepted_cfl_bound_present=" +
            std::string(summary.accepted_cfl_bound_present ? "true" : "false") +
            ", accepted_cfl_bound=" +
            std::to_string(summary.accepted_cfl_bound) +
            ", cfl_scope='" + summary.cfl_scope + "'" +
            ", trace_penalty_bound=" +
            std::string(summary.trace_penalty_bound_present ? "true" : "false") +
            ", trace_penalty_valid=" +
            std::string(summary.trace_penalty_bound_valid ? "true" : "false"),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }
}

} // namespace

std::string StabilizationAnalyzer::name() const {
    return "StabilizationAnalyzer";
}

void StabilizationAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    // =====================================================================
    // PRIMARY PATH: Consume ContributionDescriptors
    // =====================================================================
    const auto& contributions = context.contributions();
    std::vector<std::string> covered_operator_tags;
    if (!contributions.empty()) {
        for (const auto& contrib : contributions) {
            if (contrib.role != ContributionRole::StabilizationBlock) continue;

            PropertyClaim claim;
            claim.kind = PropertyKind::Stabilization;
            const bool mesh_scale_hint_only =
                hasFlag(contrib.traits, OperatorTraitFlags::MeshScaleDependentHint) &&
                !hasFlag(contrib.traits, OperatorTraitFlags::StabilizationLike);
            claim.status = mesh_scale_hint_only
                ? PropertyStatus::Likely
                : PropertyStatus::Exact;
            claim.confidence = mesh_scale_hint_only
                ? AnalysisConfidence::Low
                : contrib.confidence;
            claim.certification_class = CertificationClass::NotCertified;
            claim.evidence_level = mesh_scale_hint_only
                ? EvidenceLevel::DescriptorHint
                : EvidenceLevel::StructuralMetadata;
            claim.domain = contrib.domain;
            claim.claim_origin = "StabilizationAnalyzer";
            if (!contrib.operator_tag.empty()) {
                covered_operator_tags.push_back(contrib.operator_tag);
            }

            if (!contrib.test_variables.empty()) {
                for (const auto& tv : contrib.test_variables) {
                    claim.variables.push_back(tv);
                }
                claim.description =
                    "Stabilization detected in contribution '" +
                    contrib.operator_tag + "' for " +
                    std::to_string(contrib.test_variables.size()) +
                    " variable(s); adequacy is reported by separate stabilization summaries";
            } else {
                claim.description =
                    "Stabilization detected in contribution '" +
                    contrib.operator_tag +
                    "'; adequacy is reported by separate stabilization summaries";
            }

            claim.addEvidence("StabilizationAnalyzer",
                "ContributionDescriptor role=StabilizationBlock from " +
                contrib.origin);

            report.claims.push_back(std::move(claim));
        }
    }

    // =====================================================================
    // FALLBACK PATH: FormulationRecords
    // =====================================================================
    const auto& records = context.formulationRecords();
    if (records.empty()) {
        emitAdequacyClaims(context, report);
        return;
    }

    FormStructureAnalyzer fsa;

    for (const auto& rec : records) {
        if (std::find(covered_operator_tags.begin(),
                      covered_operator_tags.end(),
                      rec.operator_tag) != covered_operator_tags.end()) {
            continue;
        }
        // Check the record-level flag first
        bool record_has_stab = rec.has_stabilization_terms;

        // If we have a residual expression, also check per-field
        std::vector<FieldId> stabilized_fields;

        if (rec.residual_expr) {
            for (FieldId fid : rec.active_fields) {
                auto fs = fsa.analyzeField(*rec.residual_expr, fid);
                if (fs.has_stabilization) {
                    stabilized_fields.push_back(fid);
                }
            }
        }

        bool has_stabilization = record_has_stab || !stabilized_fields.empty();
        if (!has_stabilization) continue;

        PropertyClaim claim;
        claim.kind = PropertyKind::Stabilization;
        claim.status = PropertyStatus::Likely;
        claim.confidence = record_has_stab
            ? AnalysisConfidence::Medium
            : AnalysisConfidence::Low;
        claim.certification_class = CertificationClass::NotCertified;
        claim.evidence_level = record_has_stab
            ? EvidenceLevel::DescriptorHint
            : EvidenceLevel::SyntaxPattern;
        claim.domain = DomainKind::Cell;
        claim.claim_origin = "StabilizationAnalyzer";

        // If specific stabilized fields were identified, list them
        if (!stabilized_fields.empty()) {
            for (FieldId fid : stabilized_fields) {
                claim.variables.push_back(VariableKey::field(fid));
            }
            claim.description =
                "Stabilization detected in formulation '" +
                rec.operator_tag + "' for " +
                std::to_string(stabilized_fields.size()) +
                " field(s); adequacy is reported by separate stabilization summaries";
            claim.addEvidence("StabilizationAnalyzer",
                "FormStructureAnalyzer: per-field has_stabilization=true");
        } else {
            // Only record-level flag
            for (FieldId fid : rec.active_fields) {
                claim.variables.push_back(VariableKey::field(fid));
            }
            claim.description =
                "Stabilization detected in formulation '" +
                rec.operator_tag +
                "'; adequacy is reported by separate stabilization summaries";
            claim.addEvidence("StabilizationAnalyzer",
                "FormulationRecord::has_stabilization_terms=true");
        }

        report.claims.push_back(std::move(claim));
    }

    emitAdequacyClaims(context, report);

}

} // namespace analysis
} // namespace FE
} // namespace svmp
