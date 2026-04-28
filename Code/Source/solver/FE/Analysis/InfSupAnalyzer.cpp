/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/InfSupAnalyzer.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void emitNumericInfSupClaim(ProblemAnalysisReport& report,
                            const InfSupEstimateSummary& summary)
{
    const bool positive_estimate = summary.estimate_value > 0.0;

    PropertyClaim claim;
    claim.kind = PropertyKind::InfSupCondition;
    claim.status = positive_estimate ? PropertyStatus::Exact
                                     : PropertyStatus::Violated;
    claim.confidence = AnalysisConfidence::High;
    claim.domain = summary.block.domain;
    claim.variables.push_back(summary.primal_variable);
    claim.variables.push_back(summary.multiplier_variable);
    claim.inf_sup_class = positive_estimate ? InfSupClass::StructurallySupported
                                            : InfSupClass::LikelyViolated;
    claim.certification_class = positive_estimate ? CertificationClass::Certified
                                                  : CertificationClass::Violated;
    claim.inf_sup_estimate = summary.estimate_value;
    claim.nullspace_handling_class = summary.nullspace_handling;
    claim.tested_block_id = summary.block.operator_tag;
    claim.estimate_scope = summary.estimate_scope;
    claim.description = positive_estimate
        ? "Numeric inf-sup estimate is positive for the mixed pair"
        : "Numeric inf-sup estimate is nonpositive for the mixed pair";
    claim.claim_origin = "InfSupAnalyzer";
    claim.addEvidence("InfSupAnalyzer",
        "InfSupEstimateSummary estimate=" +
        std::to_string(summary.estimate_value) +
        ", rows=" + std::to_string(summary.test_rows) +
        ", cols=" + std::to_string(summary.test_cols));
    report.claims.push_back(std::move(claim));
}

void emitNumericSummaryHooks(const ProblemAnalysisContext& context,
                             ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return;
    }

    for (const auto& summary : summaries->inf_sup_estimates) {
        emitNumericInfSupClaim(report, summary);
    }
}

} // namespace

std::string InfSupAnalyzer::name() const {
    return "InfSupAnalyzer";
}

void InfSupAnalyzer::run(const ProblemAnalysisContext& context,
                         ProblemAnalysisReport& report) const
{
    const auto& contributions = context.contributions();

    // =====================================================================
    // Collect pairing descriptors from contributions
    // =====================================================================

    struct PairingInfo {
        VariableKey row_var;
        VariableKey col_var;
        PairingKind kind{PairingKind::Unknown};
        std::string pairing_group;
        bool has_stabilizing_surrogate{false};
    };

    std::vector<PairingInfo> pairings;

    for (const auto& contrib : contributions) {
        for (const auto& pd : contrib.pairings) {
            PairingInfo info;
            info.row_var = pd.row_var;
            info.col_var = pd.col_var;
            info.kind = pd.kind;
            info.pairing_group = pd.pairing_group;
            info.has_stabilizing_surrogate = pd.has_stabilizing_surrogate;
            pairings.push_back(std::move(info));
        }
    }

    // =====================================================================
    // Check for saddle-point pairings.
    // Track covered variable PAIRS (not individuals) so that a pairing for
    // one subsystem can't suppress the fallback for a different subsystem
    // that merely shares a momentum variable (Issue 3).
    // =====================================================================

    // Commutative pair: {A,B} == {B,A}
    struct VarPair {
        VariableKey a, b;
        bool operator==(const VarPair& o) const {
            return (a == o.a && b == o.b) || (a == o.b && b == o.a);
        }
    };
    struct VarPairHash {
        size_t operator()(const VarPair& p) const {
            // Commutative hash: h(a)^h(b) is order-independent
            return VariableKeyHash{}(p.a) ^ VariableKeyHash{}(p.b);
        }
    };
    std::unordered_set<VarPair, VarPairHash> covered_pairs;

    for (const auto& pi : pairings) {
        if (pi.kind != PairingKind::ConstraintPair &&
            pi.kind != PairingKind::FormalAdjointPair) {
            continue;
        }

        covered_pairs.insert(VarPair{pi.row_var, pi.col_var});

        // Check if any contribution has StabilizedConstraintPair for the
        // same pairing group -- that means inf-sup is replaced by stabilization.
        bool has_stabilized_surrogate = false;
        for (const auto& other : pairings) {
            if (other.kind == PairingKind::StabilizedConstraintPair &&
                other.pairing_group == pi.pairing_group) {
                has_stabilized_surrogate = true;
                break;
            }
        }

        // Also check the has_stabilizing_surrogate flag directly
        if (pi.has_stabilizing_surrogate) {
            has_stabilized_surrogate = true;
        }

        if (has_stabilized_surrogate) {
            // Inf-sup replaced by stabilization
            PropertyClaim claim;
            claim.kind = PropertyKind::InfSupCondition;
            claim.status = PropertyStatus::Exact;
            claim.confidence = AnalysisConfidence::High;
            claim.inf_sup_class = InfSupClass::StabilizedSurrogate;
            claim.variables.push_back(pi.row_var);
            claim.variables.push_back(pi.col_var);
            claim.description =
                "Inf-sup condition replaced by stabilization surrogate"
                " for pairing group '" + pi.pairing_group + "'";
            claim.claim_origin = "InfSupAnalyzer";
            claim.addEvidence("InfSupAnalyzer",
                "StabilizedConstraintPair found for pairing group");
            report.claims.push_back(std::move(claim));
            continue;
        }

        // Check if field descriptors show compatible space families
        // (different polynomial orders => structurally supported, e.g. Taylor-Hood)
        bool structurally_supported = false;

        if (pi.row_var.kind == VariableKind::FieldComponent &&
            pi.col_var.kind == VariableKind::FieldComponent) {
            const auto* row_fd = context.fieldDescriptor(pi.row_var.field_id);
            const auto* col_fd = context.fieldDescriptor(pi.col_var.field_id);

            if (row_fd && col_fd) {
                bool both_h1 = (row_fd->space_family == SpaceFamily::H1 &&
                                col_fd->space_family == SpaceFamily::H1);
                bool different_orders =
                    (row_fd->polynomial_order != col_fd->polynomial_order);

                if (both_h1 && different_orders) {
                    structurally_supported = true;
                }
            }
        }

        if (structurally_supported) {
            PropertyClaim claim;
            claim.kind = PropertyKind::InfSupCondition;
            claim.status = PropertyStatus::Exact;
            claim.confidence = AnalysisConfidence::High;
            claim.inf_sup_class = InfSupClass::StructurallySupported;
            claim.variables.push_back(pi.row_var);
            claim.variables.push_back(pi.col_var);
            claim.description =
                "Inf-sup structurally supported by compatible space pair"
                " (different polynomial orders) for pairing group '" +
                pi.pairing_group + "'";
            claim.claim_origin = "InfSupAnalyzer";
            claim.addEvidence("InfSupAnalyzer",
                "H1 space pair with different polynomial orders");
            report.claims.push_back(std::move(claim));
        } else {
            PropertyClaim claim;
            claim.kind = PropertyKind::InfSupCondition;
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.inf_sup_class = InfSupClass::LikelyViolated;
            claim.variables.push_back(pi.row_var);
            claim.variables.push_back(pi.col_var);
            claim.description =
                "Inf-sup condition likely violated: same-order or unknown"
                " space pair without stabilization for pairing group '" +
                pi.pairing_group + "'";
            claim.claim_origin = "InfSupAnalyzer";
            claim.addEvidence("InfSupAnalyzer",
                "No stabilization surrogate and space pair does not have"
                " different polynomial orders",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
        }
    }

    // =====================================================================
    // Fallback: check prior MixedSaddlePoint claims whose variable PAIRS
    // are NOT already covered by explicit pairing-based analysis. A claim
    // is covered only if a covered pair has BOTH members in the claim's
    // variable list — sharing a single momentum variable across different
    // subsystems does not suppress the fallback for the uncovered one.
    // =====================================================================

    auto saddle_claims = report.claimsOfKind(PropertyKind::MixedSaddlePoint);
    for (const auto* sc : saddle_claims) {
        // Check if any pair of variables in this claim appears in covered_pairs
        bool claim_covered = false;
        for (std::size_t i = 0; i < sc->variables.size() && !claim_covered; ++i) {
            for (std::size_t j = i + 1; j < sc->variables.size(); ++j) {
                if (covered_pairs.count(
                        VarPair{sc->variables[i], sc->variables[j]})) {
                    claim_covered = true;
                    break;
                }
            }
        }
        if (claim_covered) continue;

        PropertyClaim claim;
        claim.kind = PropertyKind::InfSupCondition;
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.inf_sup_class = InfSupClass::Required;
        claim.variables = sc->variables;
        claim.description =
            "Inf-sup condition required for mixed saddle-point system"
            " (inferred from MixedSaddlePoint claim, no pairing metadata)";
        claim.claim_origin = "InfSupAnalyzer";
        claim.addEvidence("InfSupAnalyzer",
            "MixedSaddlePoint claim present but no PairingDescriptor"
            " metadata available for detailed classification",
            AnalysisConfidence::Medium);
        report.claims.push_back(std::move(claim));
    }

    emitNumericSummaryHooks(context, report);
}

} // namespace analysis
} // namespace FE
} // namespace svmp
