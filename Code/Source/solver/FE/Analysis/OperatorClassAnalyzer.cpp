/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/OperatorClassAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormStructureAnalyzer.h"

#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

std::string OperatorClassAnalyzer::name() const {
    return "OperatorClassAnalyzer";
}

void OperatorClassAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    // =====================================================================
    // PRIMARY PATH: Consume ContributionDescriptors
    // =====================================================================
    const auto& contributions = context.contributions();
    if (!contributions.empty()) {
        // Per-variable, collect all DiagonalBlock contributions and check traits.
        struct DiagInfo {
            bool all_symmetric = true;
            bool all_psd = true;
            bool any_first_order = false;
            bool any_stabilization = false;
            int count = 0;
            DomainKind domain = DomainKind::Cell;
        };
        std::unordered_map<VariableKey, DiagInfo, VariableKeyHash> per_var;

        for (const auto& contrib : contributions) {
            if (contrib.role != ContributionRole::DiagonalBlock) continue;

            for (const auto& tv : contrib.test_variables) {
                auto& info = per_var[tv];
                info.count++;
                info.domain = contrib.domain;

                if (!hasFlag(contrib.traits, OperatorTraitFlags::SymmetricLike)) {
                    info.all_symmetric = false;
                }
                if (!hasFlag(contrib.traits, OperatorTraitFlags::PositiveSemiDefiniteLike)) {
                    info.all_psd = false;
                }
                if (hasFlag(contrib.traits, OperatorTraitFlags::HasFirstOrder)) {
                    info.any_first_order = true;
                }

                // Check for stabilization via the StabilizationBlock role
                // on sibling contributions for the same variable — but here
                // we are iterating DiagonalBlocks only. Use traits instead.
            }
        }

        // Check for stabilization contributions per variable
        for (const auto& contrib : contributions) {
            if (contrib.role == ContributionRole::StabilizationBlock) {
                for (const auto& tv : contrib.test_variables) {
                    auto it = per_var.find(tv);
                    if (it != per_var.end()) {
                        it->second.any_stabilization = true;
                    }
                }
            }
        }

        for (const auto& [vk, info] : per_var) {
            if (info.count == 0) continue;

            // Symmetry: all diagonal blocks for this variable have SymmetricLike
            if (info.all_symmetric) {
                PropertyClaim claim;
                claim.kind = PropertyKind::OperatorSymmetry;
                claim.status = PropertyStatus::Likely;
                claim.confidence = info.any_stabilization
                    ? AnalysisConfidence::Low
                    : AnalysisConfidence::Medium;
                if (vk.kind == VariableKind::FieldComponent) {
                    claim.field = vk.field_id;
                }
                claim.domain = info.domain;
                claim.variables.push_back(vk);

                claim.description =
                    "Bilinear form uses only gradient with no lower-order "
                    "terms (Laplacian-like, self-adjoint)";
                if (info.any_stabilization) {
                    claim.description +=
                        " (stabilization may break exact symmetry)";
                }
                claim.symmetry_class = OperatorTraitFlags::SymmetricLike;
                claim.claim_origin = "OperatorClassAnalyzer";
                claim.addEvidence("OperatorClassAnalyzer",
                    "All DiagonalBlock contributions have SymmetricLike trait",
                    claim.confidence);
                report.claims.push_back(std::move(claim));
            }

            // Definiteness: all PSD and no first-order terms
            if (info.all_psd && !info.any_first_order) {
                PropertyClaim claim;
                claim.kind = PropertyKind::OperatorDefiniteness;
                claim.status = PropertyStatus::Likely;
                claim.confidence = info.any_stabilization
                    ? AnalysisConfidence::Low
                    : AnalysisConfidence::Medium;
                if (vk.kind == VariableKind::FieldComponent) {
                    claim.field = vk.field_id;
                }
                claim.domain = info.domain;
                claim.variables.push_back(vk);

                claim.description =
                    "Bilinear form is positive semi-definite "
                    "(gradient-based, no lower-order terms)";
                if (info.any_stabilization) {
                    claim.description +=
                        " (stabilization may affect definiteness)";
                }
                claim.definiteness_class = OperatorTraitFlags::PositiveSemiDefiniteLike;
                claim.claim_origin = "OperatorClassAnalyzer";
                claim.addEvidence("OperatorClassAnalyzer",
                    "All DiagonalBlock contributions have "
                    "PositiveSemiDefiniteLike and no HasFirstOrder",
                    claim.confidence);
                report.claims.push_back(std::move(claim));
            }
        }

        // Also emit symmetry claims from kernel contribution records with
        // explicit is_symmetric_like flags.
        for (const auto& krec : context.kernelContributionRecords()) {
            if (krec.is_symmetric_like && !krec.test_variables.empty()) {
                PropertyClaim claim;
                claim.kind = PropertyKind::OperatorSymmetry;
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.domain = krec.domain;
                for (const auto& v : krec.test_variables) claim.variables.push_back(v);
                claim.description =
                    "Hand-written kernel '" + krec.operator_tag +
                    "' declares symmetric-like structure";
                claim.addEvidence("OperatorClassAnalyzer",
                    "KernelContributionRecord from " + krec.source_name,
                    AnalysisConfidence::Medium);
                report.claims.push_back(std::move(claim));
            }
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
        if (!rec.residual_expr) continue;

        for (FieldId fid : rec.active_fields) {
            auto fs = fsa.analyzeField(*rec.residual_expr, fid);
            if (fs.occurrence_count == 0) continue;

            // --- Symmetry classification ---
            bool is_symmetric = false;
            std::string symmetry_reason;

            if (fs.only_through_annihilating_ops && !fs.has_absolute_value) {
                if (fs.has_gradient && !fs.has_sym_grad && !fs.has_divergence &&
                    !fs.has_curl) {
                    is_symmetric = true;
                    symmetry_reason =
                        "Bilinear form uses only gradient with no lower-order "
                        "terms (Laplacian-like, self-adjoint)";
                } else if (fs.only_through_sym_grad && !fs.has_plain_grad) {
                    is_symmetric = true;
                    symmetry_reason =
                        "Bilinear form uses only sym(grad) with no lower-order "
                        "terms (elasticity-like, self-adjoint)";
                }
            }

            if (is_symmetric) {
                PropertyClaim claim;
                claim.kind = PropertyKind::OperatorSymmetry;
                claim.status = PropertyStatus::Likely;
                claim.confidence = fs.has_stabilization
                    ? AnalysisConfidence::Low
                    : AnalysisConfidence::Medium;
                claim.field = fid;
                claim.domain = DomainKind::Cell;
                claim.variables.push_back(VariableKey::field(fid));
                claim.description = symmetry_reason;
                if (fs.has_stabilization) {
                    claim.description +=
                        " (stabilization may break exact symmetry)";
                }
                claim.addEvidence("OperatorClassAnalyzer", symmetry_reason,
                    claim.confidence);
                report.claims.push_back(std::move(claim));
            }

            // --- Definiteness classification ---
            if (fs.only_through_annihilating_ops &&
                !fs.has_absolute_value &&
                (fs.has_gradient || fs.has_sym_grad) &&
                !fs.has_time_derivative) {

                PropertyClaim claim;
                claim.kind = PropertyKind::OperatorDefiniteness;
                claim.status = PropertyStatus::Likely;
                claim.confidence = fs.has_stabilization
                    ? AnalysisConfidence::Low
                    : AnalysisConfidence::Medium;
                claim.field = fid;
                claim.domain = DomainKind::Cell;
                claim.variables.push_back(VariableKey::field(fid));
                claim.description =
                    "Bilinear form is positive semi-definite "
                    "(gradient-based, no lower-order terms)";
                if (fs.has_stabilization) {
                    claim.description +=
                        " (stabilization may affect definiteness)";
                }
                claim.addEvidence("OperatorClassAnalyzer",
                    "only_through_annihilating_ops=true, has_absolute_value=false",
                    claim.confidence);
                report.claims.push_back(std::move(claim));
            }
        }
    }

    // Also emit symmetry claims from kernel contribution records with
    // explicit is_symmetric_like flags.
    for (const auto& krec : context.kernelContributionRecords()) {
        if (krec.is_symmetric_like && !krec.test_variables.empty()) {
            PropertyClaim claim;
            claim.kind = PropertyKind::OperatorSymmetry;
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.domain = krec.domain;
            for (const auto& v : krec.test_variables) claim.variables.push_back(v);
            claim.description =
                "Hand-written kernel '" + krec.operator_tag +
                "' declares symmetric-like structure";
            claim.addEvidence("OperatorClassAnalyzer",
                "KernelContributionRecord from " + krec.source_name,
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
