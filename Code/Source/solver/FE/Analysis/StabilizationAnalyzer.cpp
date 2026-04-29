/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/StabilizationAnalyzer.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormStructureAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void emitAdequacyClaims(const ProblemAnalysisContext& context,
                        ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->stabilization_adequacy) {
        const bool metadata_complete =
            summary.parameter_formula_metadata_present &&
            summary.residual_consistency_evidence_present &&
            summary.regime_metadata_present &&
            summary.peclet_condition_satisfied &&
            summary.cfl_condition_satisfied;

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

        if (summary.violation_count > 0u) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Stabilization adequacy summary reports violated parameter, consistency, or regime checks";
        } else if (metadata_complete) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Stabilization adequacy is certified by parameter, consistency, Peclet, and CFL metadata";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Stabilization adequacy is unknown because parameter, consistency, or regime metadata is incomplete";
        }

        claim.addEvidence("StabilizationAnalyzer",
            "StabilizationAdequacySummary id='" + summary.stabilization_id +
            "', method='" + summary.method_family +
            "', parameter_formula=" +
            std::string(summary.parameter_formula_metadata_present ? "true" : "false") +
            ", residual_consistency=" +
            std::string(summary.residual_consistency_evidence_present ? "true" : "false") +
            ", regime_metadata=" +
            std::string(summary.regime_metadata_present ? "true" : "false") +
            ", peclet_ok=" +
            std::string(summary.peclet_condition_satisfied ? "true" : "false") +
            ", cfl_ok=" +
            std::string(summary.cfl_condition_satisfied ? "true" : "false"),
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
    if (!contributions.empty()) {
        for (const auto& contrib : contributions) {
            if (contrib.role != ContributionRole::StabilizationBlock) continue;

            PropertyClaim claim;
            claim.kind = PropertyKind::Stabilization;
            claim.status = PropertyStatus::Exact;
            claim.confidence = contrib.confidence;
            claim.certification_class = CertificationClass::NotCertified;
            claim.domain = contrib.domain;
            claim.claim_origin = "StabilizationAnalyzer";

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

        emitAdequacyClaims(context, report);
        return;
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
        claim.status = PropertyStatus::Exact;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = CertificationClass::NotCertified;
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
