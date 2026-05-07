/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/InfSupAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

Real effectiveTolerance(const InfSupEstimateSummary& summary) noexcept
{
    return numeric::finiteDeclaredTolerance(summary.estimate_tolerance)
        ? summary.estimate_tolerance
        : Real{1.0e-12};
}

bool nullspaceHandlingAcceptable(NullspaceHandlingClass handling) noexcept
{
    return handling == NullspaceHandlingClass::NotApplicable ||
           handling == NullspaceHandlingClass::AnchoredByConstraints ||
           handling == NullspaceHandlingClass::ProjectedOut;
}

void appendUnique(std::vector<VariableKey>& values, const VariableKey& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::vector<VariableKey> blockVariables(const OperatorBlockId& block)
{
    std::vector<VariableKey> variables;
    for (const auto& variable : block.test_variables) {
        appendUnique(variables, variable);
    }
    for (const auto& variable : block.trial_variables) {
        appendUnique(variables, variable);
    }
    return variables;
}

bool containsVariable(const std::vector<VariableKey>& values,
                      const VariableKey& variable)
{
    return std::find(values.begin(), values.end(), variable) != values.end();
}

bool numericScopeMatchesPair(const InfSupEstimateSummary& summary)
{
    const auto variables = blockVariables(summary.block);
    return containsVariable(variables, summary.primal_variable) &&
           containsVariable(variables, summary.multiplier_variable);
}

bool variableSetsIntersect(const std::vector<VariableKey>& a,
                           const std::vector<VariableKey>& b)
{
    for (const auto& variable : a) {
        if (containsVariable(b, variable)) {
            return true;
        }
    }
    return false;
}

bool degeneracyClassIsOutOfScope(DegeneracyClass c) noexcept
{
    return c == DegeneracyClass::DegenerateDiagnostic ||
           c == DegeneracyClass::GaugeLikeNullspace ||
           c == DegeneracyClass::UnanchoredKernel ||
           c == DegeneracyClass::ProjectedKernel;
}

const NullspaceDegeneracySummary* matchingDegeneracy(
    const AnalysisSummarySet* summaries,
    const InfSupEstimateSummary& estimate)
{
    if (!summaries) {
        return nullptr;
    }
    std::vector<VariableKey> estimate_variables{
        estimate.primal_variable,
        estimate.multiplier_variable,
    };
    const auto block_variables = blockVariables(estimate.block);
    for (const auto& summary : summaries->nullspace_degeneracies) {
        if (!degeneracyClassIsOutOfScope(summary.degeneracy_class)) {
            continue;
        }
        const auto degeneracy_variables = summary.affected_variables.empty()
            ? blockVariables(summary.block)
            : summary.affected_variables;
        if (variableSetsIntersect(degeneracy_variables, estimate_variables) ||
            variableSetsIntersect(degeneracy_variables, block_variables)) {
            return &summary;
        }
    }
    return nullptr;
}

bool numericMetadataComplete(const InfSupEstimateSummary& summary)
{
    const bool uniform_lower_bound_valid =
        summary.uniform_lower_bound_value_present &&
        numeric::finitePositive(summary.uniform_lower_bound) &&
        summary.uniform_lower_bound > effectiveTolerance(summary);
    return summary.estimator_metadata_present &&
           summary.norm_metadata_present &&
           summary.mesh_refinement_evidence_present &&
           summary.mesh_refinement_sample_count >= 2u &&
           summary.uniform_lower_bound_evidence_present &&
           uniform_lower_bound_valid &&
           summary.mesh_family_scope_present &&
           summary.boundary_condition_scope_present &&
           !summary.inf_sup_theorem_id.empty() &&
           summary.test_rows > 0 &&
           summary.test_cols > 0 &&
           !summary.estimate_scope.empty() &&
           numericScopeMatchesPair(summary) &&
           nullspaceHandlingAcceptable(summary.nullspace_handling);
}

bool numericFailureScopeValid(const InfSupEstimateSummary& summary)
{
    return numeric::finite(summary.estimate_value) &&
           numeric::finiteDeclaredTolerance(summary.estimate_tolerance) &&
           summary.estimator_metadata_present &&
           summary.norm_metadata_present &&
           summary.mesh_family_scope_present &&
           summary.boundary_condition_scope_present &&
           summary.test_rows > 0 &&
           summary.test_cols > 0 &&
           !summary.estimate_scope.empty() &&
           numericScopeMatchesPair(summary) &&
           nullspaceHandlingAcceptable(summary.nullspace_handling);
}

bool sameUnorderedPair(const VariableKey& a,
                       const VariableKey& b,
                       const VariableKey& c,
                       const VariableKey& d)
{
    return (a == c && b == d) || (a == d && b == c);
}

bool fieldMetadataMatchesSummary(const ProblemAnalysisContext& context,
                                 const InfSupPairCertificationSummary& summary)
{
    if (summary.primal_variable.kind != VariableKind::FieldComponent ||
        summary.multiplier_variable.kind != VariableKind::FieldComponent) {
        return false;
    }

    const auto* primal =
        context.fieldDescriptor(summary.primal_variable.field_id);
    const auto* multiplier =
        context.fieldDescriptor(summary.multiplier_variable.field_id);
    if (!primal || !multiplier) {
        return false;
    }

    if (summary.primal_polynomial_order >= 0 &&
        primal->polynomial_order != summary.primal_polynomial_order) {
        return false;
    }
    if (summary.multiplier_polynomial_order >= 0 &&
        multiplier->polynomial_order != summary.multiplier_polynomial_order) {
        return false;
    }
    if (summary.primal_space_family != SpaceFamily::Unknown &&
        primal->space_family != summary.primal_space_family) {
        return false;
    }
    if (summary.multiplier_space_family != SpaceFamily::Unknown &&
        multiplier->space_family != summary.multiplier_space_family) {
        return false;
    }
    return true;
}

bool certifiedPairMetadataComplete(const ProblemAnalysisContext& context,
                                   const InfSupPairCertificationSummary& summary)
{
    const auto variables = blockVariables(summary.block);
    const bool scope_matches =
        containsVariable(variables, summary.primal_variable) &&
        containsVariable(variables, summary.multiplier_variable);
    const bool stable_pair_or_fortin =
        summary.known_stable_pair ||
        summary.fortin_operator_evidence_present;
    const bool beta_bound_valid =
        (summary.beta_lower_bound_present &&
         numeric::finitePositive(summary.beta_lower_bound)) ||
        summary.beta_lower_bound_symbolic_present;
    const bool fortin_norm_ok =
        !summary.fortin_operator_evidence_present ||
        (summary.fortin_operator_norm_bound_present &&
         numeric::finitePositive(summary.fortin_operator_norm_bound)) ||
        summary.fortin_operator_norm_bound_symbolic_present;
    return scope_matches &&
           stable_pair_or_fortin &&
           !summary.pair_family.empty() &&
           !summary.inf_sup_theorem_id.empty() &&
           summary.mesh_assumption_evidence_present &&
           summary.domain_assumption_evidence_present &&
           summary.boundary_condition_scope_present &&
           beta_bound_valid &&
           fortin_norm_ok &&
           fieldMetadataMatchesSummary(context, summary);
}

bool hasCertifiedInfSupForPair(const ProblemAnalysisReport& report,
                               const VariableKey& a,
                               const VariableKey& b)
{
    for (const auto& claim : report.claims) {
        if (claim.kind != PropertyKind::InfSupCondition ||
            !claim.certification_class ||
            *claim.certification_class != CertificationClass::Certified ||
            claim.variables.size() < 2u) {
            continue;
        }
        if (sameUnorderedPair(a, b, claim.variables[0], claim.variables[1])) {
            return true;
        }
    }
    return false;
}

void emitNumericInfSupClaim(ProblemAnalysisReport& report,
                            const InfSupEstimateSummary& summary,
                            const AnalysisSummarySet* summaries)
{
    const Real tol = effectiveTolerance(summary);
    const bool tolerance_declared =
        numeric::finiteDeclaredTolerance(summary.estimate_tolerance);
    const bool estimate_finite = numeric::finite(summary.estimate_value);
    const bool positive_above_tolerance =
        tolerance_declared && estimate_finite && summary.estimate_value > tol;
    const bool nonpositive_estimate =
        estimate_finite && summary.estimate_value <= Real{};
    const bool estimate_at_or_below_tolerance =
        tolerance_declared && estimate_finite && summary.estimate_value <= tol;
    const bool metadata_complete = numericMetadataComplete(summary);
    const bool failure_scope_valid = numericFailureScopeValid(summary);
    const auto* degeneracy = matchingDegeneracy(summaries, summary);

    PropertyClaim claim;
    claim.kind = PropertyKind::InfSupCondition;
    claim.domain = summary.block.domain;
    claim.variables.push_back(summary.primal_variable);
    claim.variables.push_back(summary.multiplier_variable);
    claim.inf_sup_estimate = summary.estimate_value;
    claim.nullspace_handling_class = summary.nullspace_handling;
    claim.tested_block_id = summary.block.operator_tag;
    claim.estimate_scope = summary.estimate_scope;
    claim.claim_origin = "InfSupAnalyzer";

    if (nonpositive_estimate && degeneracy != nullptr) {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.inf_sup_class = InfSupClass::Unknown;
        claim.applicability_class = ApplicabilityClass::NotApplicable;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            "Numeric inf-sup estimate is nonpositive, but the reduced operator was classified as a degenerate diagnostic scope";
    } else if (estimate_at_or_below_tolerance && failure_scope_valid) {
        claim.status = PropertyStatus::Violated;
        claim.confidence = AnalysisConfidence::High;
        claim.inf_sup_class = InfSupClass::LikelyViolated;
        claim.certification_class = CertificationClass::Violated;
        claim.description =
            "Scoped numeric inf-sup estimate fails the declared tolerance for the mixed pair";
    } else if (estimate_at_or_below_tolerance || nonpositive_estimate) {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.inf_sup_class = InfSupClass::Unknown;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            "Numeric inf-sup estimate is at or below the tolerance, but estimator, norm, pair, mesh, boundary, or kernel metadata is incomplete";
    } else if (positive_above_tolerance && metadata_complete) {
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.inf_sup_class = InfSupClass::NumericallySupported;
        claim.certification_class = CertificationClass::Certified;
        claim.description =
            "Scoped numeric inf-sup estimate exceeds tolerance with required metadata";
    } else if (positive_above_tolerance) {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.inf_sup_class = InfSupClass::NumericallySupported;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            "Numeric inf-sup estimate is positive but lacks complete uniform-family certification metadata";
    } else {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.inf_sup_class = InfSupClass::Unknown;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            tolerance_declared && estimate_finite
                ? "Numeric inf-sup estimate is positive but not separated from tolerance"
                : "Numeric inf-sup estimate lacks finite estimate or finite declared tolerance";
    }

    claim.addEvidence("InfSupAnalyzer",
        "InfSupEstimateSummary estimate=" +
        std::to_string(summary.estimate_value) +
        ", tolerance=" + std::to_string(tol) +
        ", tolerance_declared=" +
        std::string(tolerance_declared ? "true" : "false") +
        ", finite_estimate=" +
        std::string(estimate_finite ? "true" : "false") +
        ", rows=" + std::to_string(summary.test_rows) +
        ", cols=" + std::to_string(summary.test_cols) +
        ", norm_metadata=" +
        std::string(summary.norm_metadata_present ? "true" : "false") +
        ", refinement_samples=" +
        std::to_string(summary.mesh_refinement_sample_count) +
        ", uniform_lower_bound=" +
        std::string(summary.uniform_lower_bound_evidence_present ? "true" : "false") +
        ", uniform_lower_bound_value=" +
        std::to_string(summary.uniform_lower_bound) +
        ", theorem='" + summary.inf_sup_theorem_id + "'" +
        ", mesh_family_scope=" +
        std::string(summary.mesh_family_scope_present ? "true" : "false") +
        ", boundary_scope=" +
        std::string(summary.boundary_condition_scope_present ? "true" : "false") +
        ", metadata_complete=" +
        std::string(metadata_complete ? "true" : "false") +
        ", failure_scope_valid=" +
        std::string(failure_scope_valid ? "true" : "false") +
        ", degeneracy_scope=" +
        std::string(degeneracy != nullptr ? "true" : "false") +
        (degeneracy != nullptr
             ? ", degeneracy_reason='" + degeneracy->reason + "'"
             : ""),
        claim.confidence);
    report.claims.push_back(std::move(claim));
}

void emitCertifiedPairInfSupClaim(const ProblemAnalysisContext& context,
                                  ProblemAnalysisReport& report,
                                  const InfSupPairCertificationSummary& summary)
{
    const bool metadata_complete = certifiedPairMetadataComplete(context, summary);

    PropertyClaim claim;
    claim.kind = PropertyKind::InfSupCondition;
    claim.domain = summary.block.domain;
    claim.variables.push_back(summary.primal_variable);
    claim.variables.push_back(summary.multiplier_variable);
    claim.tested_block_id = summary.block.operator_tag;
    claim.estimate_scope = summary.pair_family;
    claim.claim_origin = "InfSupAnalyzer";
    claim.inf_sup_class = metadata_complete
        ? InfSupClass::StructurallySupported
        : InfSupClass::Unknown;

    if (metadata_complete) {
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = CertificationClass::Certified;
        claim.description =
            "Inf-sup condition is certified by known stable-pair/Fortin metadata";
    } else {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            "Inf-sup pair metadata is present but lacks stable-pair, mesh, domain, scope, or matching field/order evidence";
    }

    claim.addEvidence("InfSupAnalyzer",
        "InfSupPairCertificationSummary family='" + summary.pair_family +
        "', known_stable_pair=" +
        std::string(summary.known_stable_pair ? "true" : "false") +
        ", fortin=" +
        std::string(summary.fortin_operator_evidence_present ? "true" : "false") +
        ", mesh_assumptions=" +
        std::string(summary.mesh_assumption_evidence_present ? "true" : "false") +
        ", domain_assumptions=" +
        std::string(summary.domain_assumption_evidence_present ? "true" : "false") +
        ", boundary_scope=" +
        std::string(summary.boundary_condition_scope_present ? "true" : "false") +
        ", beta_lower_bound=" +
        std::to_string(summary.beta_lower_bound) +
        ", beta_symbolic=" +
        std::string(summary.beta_lower_bound_symbolic_present ? "true" : "false") +
        ", theorem='" + summary.inf_sup_theorem_id + "'" +
        ", fortin_norm_bound=" +
        std::to_string(summary.fortin_operator_norm_bound) +
        ", fortin_norm_symbolic=" +
        std::string(summary.fortin_operator_norm_bound_symbolic_present ? "true" : "false") +
        ", matching_field_metadata=" +
        std::string(fieldMetadataMatchesSummary(context, summary) ? "true" : "false"),
        claim.confidence);
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
        emitNumericInfSupClaim(report, summary, summaries);
    }
    for (const auto& summary : summaries->inf_sup_pair_certifications) {
        emitCertifiedPairInfSupClaim(context, report, summary);
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
            // Stabilization may replace the inf-sup requirement, but adequacy
            // must be certified by stabilization-specific metadata.
            PropertyClaim claim;
            claim.kind = PropertyKind::InfSupCondition;
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.inf_sup_class = InfSupClass::StabilizedSurrogate;
            claim.certification_class = CertificationClass::NotCertified;
            claim.variables.push_back(pi.row_var);
            claim.variables.push_back(pi.col_var);
            claim.description =
                "Inf-sup condition has a stabilization surrogate but adequacy is not certified from presence alone"
                " for pairing group '" + pi.pairing_group + "'";
            claim.claim_origin = "InfSupAnalyzer";
            claim.addEvidence("InfSupAnalyzer",
                "StabilizedConstraintPair or stabilization-surrogate flag found for pairing group",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
            continue;
        }

        if (hasCertifiedInfSupForPair(report, pi.row_var, pi.col_var)) {
            continue;
        }

        // A higher/lower-order H1 pair is only heuristic evidence here.  Exact
        // certification requires InfSupPairCertificationSummary or a scoped
        // numerical inf-sup estimate.
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
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.inf_sup_class = InfSupClass::StructurallySupported;
            claim.certification_class = CertificationClass::NotCertified;
            claim.variables.push_back(pi.row_var);
            claim.variables.push_back(pi.col_var);
            claim.description =
                "Inf-sup may be structurally supported by a higher/lower-order H1 pair,"
                " but known stable-pair or Fortin metadata is required for certification"
                " for pairing group '" +
                pi.pairing_group + "'";
            claim.claim_origin = "InfSupAnalyzer";
            claim.addEvidence("InfSupAnalyzer",
                "H1 space pair with different polynomial orders only",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
        } else {
            PropertyClaim claim;
            claim.kind = PropertyKind::InfSupCondition;
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.inf_sup_class = InfSupClass::Required;
            claim.certification_class = CertificationClass::NotCertified;
            claim.variables.push_back(pi.row_var);
            claim.variables.push_back(pi.col_var);
            claim.description =
                "Inf-sup evidence is required: same-order or unknown"
                " space pair without stabilization metadata is not a certification-grade failure for pairing group '" +
                pi.pairing_group + "'";
            claim.claim_origin = "InfSupAnalyzer";
            claim.addEvidence("InfSupAnalyzer",
                "No stabilization surrogate and space pair does not have"
                " different polynomial orders; no theorem or scoped numeric failure evidence was supplied",
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
        claim.domain = sc->domain;
        claim.tested_block_id = sc->tested_block_id;
        claim.estimate_scope = sc->estimate_scope;
        claim.certification_class = CertificationClass::NotCertified;
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
