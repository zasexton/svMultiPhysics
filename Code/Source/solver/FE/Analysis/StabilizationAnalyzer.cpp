/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/StabilizationAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormStructureAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

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
            claim.status = PropertyStatus::Preserved;
            claim.confidence = contrib.confidence;
            claim.domain = contrib.domain;

            if (!contrib.test_variables.empty()) {
                for (const auto& tv : contrib.test_variables) {
                    claim.variables.push_back(tv);
                }
                claim.description =
                    "Stabilization detected in contribution '" +
                    contrib.operator_tag + "' for " +
                    std::to_string(contrib.test_variables.size()) + " variable(s)";
            } else {
                claim.description =
                    "Stabilization detected in contribution '" +
                    contrib.operator_tag + "'";
            }

            claim.addEvidence("StabilizationAnalyzer",
                "ContributionDescriptor role=StabilizationBlock from " +
                contrib.origin);

            report.claims.push_back(std::move(claim));
        }

        return;
    }

    // =====================================================================
    // FALLBACK PATH: FormulationRecords
    // =====================================================================
    const auto& records = context.formulationRecords();
    if (records.empty()) return;

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
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.domain = DomainKind::Cell;

        // If specific stabilized fields were identified, list them
        if (!stabilized_fields.empty()) {
            for (FieldId fid : stabilized_fields) {
                claim.variables.push_back(VariableKey::field(fid));
            }
            claim.description =
                "Stabilization detected in formulation '" +
                rec.operator_tag + "' for " +
                std::to_string(stabilized_fields.size()) + " field(s)";
            claim.addEvidence("StabilizationAnalyzer",
                "FormStructureAnalyzer: per-field has_stabilization=true");
        } else {
            // Only record-level flag
            for (FieldId fid : rec.active_fields) {
                claim.variables.push_back(VariableKey::field(fid));
            }
            claim.description =
                "Stabilization detected in formulation '" +
                rec.operator_tag + "'";
            claim.addEvidence("StabilizationAnalyzer",
                "FormulationRecord::has_stabilization_terms=true");
        }

        report.claims.push_back(std::move(claim));
    }

}

} // namespace analysis
} // namespace FE
} // namespace svmp
