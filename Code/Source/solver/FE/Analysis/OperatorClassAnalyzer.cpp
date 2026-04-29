/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/OperatorClassAnalyzer.h"
#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormStructureAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

Real definitenessTolerance(const DiscreteMatrixSummary& matrix) noexcept
{
    return std::max({matrix.sign_tolerance,
                     matrix.symmetry_tolerance,
                     Real{1.0e-12}});
}

enum class DefinitenessEvidence {
    PositiveDefinite,
    Semidefinite,
    Indefinite,
    Unknown,
};

bool coefficientCoverageComplete(const CoefficientPropertySummary& summary) noexcept
{
    const bool state_scope_ok =
        (!summary.state_dependent && !summary.time_dependent) ||
        summary.state_sample_coverage_complete;
    return summary.coefficient_region_coverage_complete &&
           summary.quadrature_point_coverage_complete &&
           summary.lower_bound_valid_for_all_samples &&
           summary.tolerance_metadata_present &&
           coefficientLowerBoundMatchesPositivity(summary) &&
           state_scope_ok;
}

DefinitenessEvidence classifyDefinitenessEvidence(
    const DiscreteMatrixSummary& matrix)
{
    const Real tol = definitenessTolerance(matrix);
    if (matrix.coercivity_lower_bound) {
        if (!numeric::finite(*matrix.coercivity_lower_bound)) {
            return DefinitenessEvidence::Unknown;
        }
        if (*matrix.coercivity_lower_bound > tol) {
            return DefinitenessEvidence::PositiveDefinite;
        }
        if (*matrix.coercivity_lower_bound < -tol) {
            return DefinitenessEvidence::Indefinite;
        }
        return DefinitenessEvidence::Semidefinite;
    }
    if (matrix.min_eigenvalue_estimate) {
        if (!numeric::finite(*matrix.min_eigenvalue_estimate)) {
            return DefinitenessEvidence::Unknown;
        }
        if (*matrix.min_eigenvalue_estimate > tol) {
            return DefinitenessEvidence::PositiveDefinite;
        }
        if (*matrix.min_eigenvalue_estimate < -tol) {
            return DefinitenessEvidence::Indefinite;
        }
        return DefinitenessEvidence::Semidefinite;
    }
    if (matrix.cholesky_factorization_succeeded) {
        return DefinitenessEvidence::PositiveDefinite;
    }
    if (matrix.ldlt_factorization_nonnegative) {
        return DefinitenessEvidence::Semidefinite;
    }
    return DefinitenessEvidence::Unknown;
}

void emitCoefficientClaim(ProblemAnalysisReport& report,
                          const CoefficientPropertySummary& summary)
{
    PropertyClaim claim;
    claim.kind = PropertyKind::CoefficientPositivity;
    claim.domain = summary.domain;
    claim.confidence = AnalysisConfidence::High;
    claim.variables = !summary.variables.empty()
        ? summary.variables
        : variablesForBlock(summary.block);
    claim.coefficient_id = summary.coefficient;
    claim.claim_origin = "OperatorClassAnalyzer";

    switch (summary.positivity) {
        case PositivityClass::Positive:
        case PositivityClass::Nonnegative:
            if (coefficientDeclaredBoundContradictsPositivity(summary)) {
                claim.status = PropertyStatus::Violated;
                claim.certification_class = CertificationClass::Violated;
                claim.description =
                    "Coefficient '" + summary.coefficient +
                    "' declared nonnegative/positive sign structure but finite lower-bound evidence contradicts it";
            } else if (coefficientCoverageComplete(summary)) {
                claim.status = PropertyStatus::Preserved;
                claim.certification_class = CertificationClass::Certified;
                claim.description =
                    "Coefficient '" + summary.coefficient +
                    "' has certified nonnegative/positive sign structure with scoped coverage metadata";
            } else {
                claim.status = PropertyStatus::Likely;
                claim.certification_class = CertificationClass::NotCertified;
                claim.description =
                    "Coefficient '" + summary.coefficient +
                    "' reports nonnegative/positive sign structure but lacks full coverage metadata";
            }
            break;
        case PositivityClass::Negative:
        case PositivityClass::Nonpositive:
        case PositivityClass::Indefinite:
            claim.status = PropertyStatus::Violated;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Coefficient '" + summary.coefficient +
                "' violates elliptic positivity assumptions";
            break;
        case PositivityClass::Unknown:
            claim.status = PropertyStatus::Unknown;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Coefficient '" + summary.coefficient +
                "' has unknown positivity";
            break;
    }

    claim.addEvidence("OperatorClassAnalyzer",
        "CoefficientPropertySummary: min_eigenvalue=" +
        std::to_string(summary.min_eigenvalue) +
        ", max_eigenvalue=" + std::to_string(summary.max_eigenvalue) +
        ", anisotropy_ratio=" +
        std::to_string(summary.anisotropy_ratio) +
        ", region_coverage=" +
        std::string(summary.coefficient_region_coverage_complete ? "true" : "false") +
        ", quadrature_coverage=" +
        std::string(summary.quadrature_point_coverage_complete ? "true" : "false") +
        ", lower_bound_all_samples=" +
        std::string(summary.lower_bound_valid_for_all_samples ? "true" : "false") +
        ", tolerance_metadata=" +
        std::string(summary.tolerance_metadata_present ? "true" : "false"));
    report.claims.push_back(std::move(claim));
}

void emitReducedDefinitenessClaim(ProblemAnalysisReport& report,
                                  const ReducedMatrixSummary& summary)
{
    const auto& matrix = summary.free_free_matrix;
    if (matrix.rows == 0 || matrix.cols == 0) {
        return;
    }

    const bool symmetric_evidence =
        matrix.square &&
        matrix.symmetry_evidence_complete &&
        matrix.structurally_symmetric &&
        matrix.numerically_symmetric;
    const auto definiteness = classifyDefinitenessEvidence(matrix);
    const bool exact_reduction = summary.reduction_exact_for_analysis;

    PropertyClaim claim;
    claim.kind = PropertyKind::OperatorDefiniteness;
    claim.domain = matrix.block.domain;
    claim.variables = variablesForBlock(matrix.block);
    claim.tested_block_id = matrix.block.operator_tag;
    claim.claim_origin = "OperatorClassAnalyzer";

    if (!exact_reduction || !symmetric_evidence) {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.coercivity_class = CoercivityClass::Unknown;
        claim.reduced_definiteness_class = CertificationClass::Unknown;
        claim.description =
            "Reduced free-free operator definiteness is unknown because exact reduction or symmetry evidence is missing";
    } else if (definiteness == DefinitenessEvidence::PositiveDefinite) {
        claim.status = PropertyStatus::Exact;
        claim.confidence = AnalysisConfidence::High;
        claim.coercivity_class = CoercivityClass::Coercive;
        claim.reduced_definiteness_class = CertificationClass::Certified;
        claim.description =
            "Reduced free-free operator has scoped SPD/coercivity evidence";
    } else if (definiteness == DefinitenessEvidence::Semidefinite) {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.coercivity_class = CoercivityClass::Semicoercive;
        claim.reduced_definiteness_class = CertificationClass::NotCertified;
        claim.description =
            "Reduced free-free operator has semidefinite evidence but no positive coercivity lower bound";
    } else if (definiteness == DefinitenessEvidence::Indefinite) {
        claim.status = PropertyStatus::Violated;
        claim.confidence = AnalysisConfidence::High;
        claim.coercivity_class = CoercivityClass::Indefinite;
        claim.reduced_definiteness_class = CertificationClass::Violated;
        claim.description =
            "Reduced free-free operator violates positive-definiteness evidence";
    } else {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.coercivity_class = CoercivityClass::Unknown;
        claim.reduced_definiteness_class = CertificationClass::Unknown;
        claim.description =
            "Reduced free-free operator definiteness is unknown without eigenvalue, factorization, or coercivity evidence";
    }

    claim.addEvidence("OperatorClassAnalyzer",
        "ReducedMatrixSummary reduction=" +
        std::to_string(static_cast<int>(summary.reduction_kind)) +
        ", symmetry_complete=" +
        std::string(matrix.symmetry_evidence_complete ? "true" : "false") +
        ", exact_reduction=" +
        std::string(summary.reduction_exact_for_analysis ? "true" : "false") +
        ", min_eigenvalue=" +
        (matrix.min_eigenvalue_estimate
             ? std::to_string(*matrix.min_eigenvalue_estimate)
             : std::string("unset")) +
        ", coercivity_lower_bound=" +
        (matrix.coercivity_lower_bound
             ? std::to_string(*matrix.coercivity_lower_bound)
             : std::string("unset")) +
        ", cholesky_success=" +
        std::string(matrix.cholesky_factorization_succeeded ? "true" : "false"),
        claim.confidence);
    report.claims.push_back(std::move(claim));
}

void emitSummaryBackedOperatorClaims(const ProblemAnalysisContext& context,
                                     ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return;
    }

    for (const auto& summary : summaries->coefficient_properties) {
        emitCoefficientClaim(report, summary);
    }

    for (const auto& summary : summaries->reduced_matrices) {
        emitReducedDefinitenessClaim(report, summary);
    }
}

} // namespace

std::string OperatorClassAnalyzer::name() const {
    return "OperatorClassAnalyzer";
}

void OperatorClassAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    emitSummaryBackedOperatorClaims(context, report);

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
                claim.operator_symmetry_class = OperatorSymmetryClass::Symmetric;
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
                claim.coercivity_class = CoercivityClass::Semicoercive;
                claim.claim_origin = "OperatorClassAnalyzer";
                claim.addEvidence("OperatorClassAnalyzer",
                    "All DiagonalBlock contributions have "
                    "PositiveSemiDefiniteLike and no HasFirstOrder",
                    claim.confidence);
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
                claim.operator_symmetry_class = OperatorSymmetryClass::Symmetric;
                claim.claim_origin = "OperatorClassAnalyzer";
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
                claim.coercivity_class = CoercivityClass::Semicoercive;
                claim.claim_origin = "OperatorClassAnalyzer";
                claim.addEvidence("OperatorClassAnalyzer",
                    "only_through_annihilating_ops=true, has_absolute_value=false",
                    claim.confidence);
                report.claims.push_back(std::move(claim));
            }
        }
    }

}

} // namespace analysis
} // namespace FE
} // namespace svmp
