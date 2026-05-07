/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/AdvancedStabilityAnalyzers.h"

#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ConstraintAnalysisSummary.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

inline Real effectiveTolerance(Real tolerance, Real fallback = 1.0e-10) noexcept
{
    return numeric::finiteDeclaredTolerance(tolerance) ? tolerance : fallback;
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
    for (const auto& v : block.test_variables) {
        appendUnique(variables, v);
    }
    for (const auto& v : block.trial_variables) {
        appendUnique(variables, v);
    }
    return variables;
}

void addWarning(ProblemAnalysisReport& report, std::string message)
{
    AnalysisIssue issue;
    issue.severity = IssueSeverity::Warning;
    issue.message = std::move(message);
    report.issues.push_back(std::move(issue));
}

bool sameVariableSetIsNontrivial(const std::vector<VariableKey>& variables)
{
    if (variables.size() < 2u) {
        return false;
    }
    return std::any_of(variables.begin() + 1, variables.end(),
                       [&](const VariableKey& variable) {
                           return variable != variables.front();
                       });
}

bool containsVariable(const std::vector<VariableKey>& variables,
                      const VariableKey& target)
{
    return std::find(variables.begin(), variables.end(), target) !=
           variables.end();
}

bool variableSetContainsAll(const std::vector<VariableKey>& superset,
                            const std::vector<VariableKey>& subset)
{
    return std::all_of(subset.begin(), subset.end(),
                       [&](const VariableKey& variable) {
                           return containsVariable(superset, variable);
                       });
}

bool localVariableSetsIntersect(const std::vector<VariableKey>& a,
                                const std::vector<VariableKey>& b)
{
    for (const auto& variable : a) {
        if (containsVariable(b, variable)) {
            return true;
        }
    }
    return false;
}

bool coveredByAnyVariableSet(
    const std::vector<VariableKey>& variables,
    const std::vector<std::vector<VariableKey>>& covered_sets)
{
    if (variables.empty()) {
        return false;
    }
    for (const auto& covered : covered_sets) {
        if (variableSetContainsAll(covered, variables)) {
            return true;
        }
    }
    return false;
}

bool productionSignSatisfied(BalanceSignClass sign, Real value, Real tol) noexcept
{
    switch (sign) {
        case BalanceSignClass::Nonpositive: return value <= tol;
        case BalanceSignClass::Nonnegative: return value >= -tol;
        case BalanceSignClass::Zero: return std::abs(value) <= tol;
        case BalanceSignClass::Unconstrained: return true;
        case BalanceSignClass::Unknown: return false;
    }
    return false;
}

using numeric::finiteNonnegative;
using numeric::finitePositive;

bool theoremScoped(const std::string& theorem_id) noexcept
{
    return !theorem_id.empty();
}

bool energyMetadataComplete(const EnergyEntropySummary& summary) noexcept
{
    const bool norm_bounds_valid =
        summary.energy_norm_equivalence_bounds_present &&
        numeric::finitePositiveOrdered(
            summary.energy_norm_equivalence_lower_bound,
            summary.energy_norm_equivalence_upper_bound);
    const bool dissipation_bound_valid =
        summary.energy_dissipation_residual_bound_present &&
        summary.energy_dissipation_tolerance_present &&
        finiteNonnegative(summary.energy_dissipation_residual_bound) &&
        finiteNonnegative(summary.energy_dissipation_tolerance) &&
        summary.energy_dissipation_residual_bound <=
            summary.energy_dissipation_tolerance;
    return summary.energy_functional_metadata_present &&
           summary.energy_norm_metadata_present &&
           summary.energy_positivity_evidence_present &&
           summary.energy_coercivity_evidence_present &&
           summary.discrete_dissipation_identity_evidence_present &&
           summary.boundary_source_energy_accounting_present &&
           summary.energy_coercivity_lower_bound_present &&
           finitePositive(summary.energy_coercivity_lower_bound) &&
           norm_bounds_valid &&
           dissipation_bound_valid &&
           theoremScoped(summary.energy_entropy_theorem_id) &&
           !summary.energy_functional_id.empty() &&
           !summary.energy_norm_id.empty();
}

bool entropyMetadataComplete(const EnergyEntropySummary& summary) noexcept
{
    const bool flux_inequality_valid =
        summary.entropy_flux_inequality_residual_present &&
        summary.entropy_flux_inequality_tolerance_present &&
        numeric::finite(summary.entropy_flux_inequality_residual) &&
        finiteNonnegative(summary.entropy_flux_inequality_tolerance) &&
        summary.entropy_flux_inequality_residual <=
            summary.entropy_flux_inequality_tolerance;
    return summary.convex_entropy_metadata_present &&
           summary.entropy_variables_metadata_present &&
           summary.entropy_flux_metadata_present &&
           summary.entropy_dissipation_metadata_present &&
           summary.boundary_source_entropy_metadata_present &&
           summary.entropy_convexity_lower_bound_present &&
           finitePositive(summary.entropy_convexity_lower_bound) &&
           flux_inequality_valid &&
           summary.entropy_dissipation_bound_present &&
           finiteNonnegative(summary.entropy_dissipation_bound) &&
           theoremScoped(summary.energy_entropy_theorem_id);
}

bool energyEntropyMetadataComplete(const EnergyEntropySummary& summary) noexcept
{
    return summary.law_kind == EnergyEntropyLawKind::Entropy
        ? entropyMetadataComplete(summary)
        : energyMetadataComplete(summary);
}

PropertyKind lawPropertyKind(EnergyEntropyLawKind kind) noexcept
{
    return kind == EnergyEntropyLawKind::Entropy
        ? PropertyKind::EntropyStability
        : PropertyKind::EnergyStability;
}

const char* tangentConsistencyName(TangentConsistencyClass consistency) noexcept
{
    switch (consistency) {
        case TangentConsistencyClass::Exact: return "Exact";
        case TangentConsistencyClass::Approximate: return "Approximate";
        case TangentConsistencyClass::Frozen: return "Frozen";
        case TangentConsistencyClass::Inconsistent: return "Inconsistent";
        case TangentConsistencyClass::Unknown: return "Unknown";
    }
    return "Unknown";
}

const char* symmetryName(SymmetryClass symmetry) noexcept
{
    switch (symmetry) {
        case SymmetryClass::Symmetric: return "Symmetric";
        case SymmetryClass::Skew: return "Skew";
        case SymmetryClass::Nonsymmetric: return "Nonsymmetric";
        case SymmetryClass::NotApplicable: return "NotApplicable";
        case SymmetryClass::Unknown: return "Unknown";
    }
    return "Unknown";
}

const char* positivityName(PositivityClass positivity) noexcept
{
    switch (positivity) {
        case PositivityClass::Positive: return "Positive";
        case PositivityClass::Nonnegative: return "Nonnegative";
        case PositivityClass::Negative: return "Negative";
        case PositivityClass::Nonpositive: return "Nonpositive";
        case PositivityClass::Indefinite: return "Indefinite";
        case PositivityClass::Unknown: return "Unknown";
    }
    return "Unknown";
}

std::optional<OperatorSymmetryClass> operatorSymmetryFrom(
    SymmetryClass symmetry)
{
    switch (symmetry) {
        case SymmetryClass::Symmetric:
            return OperatorSymmetryClass::Symmetric;
        case SymmetryClass::Skew:
            return OperatorSymmetryClass::Skew;
        case SymmetryClass::Nonsymmetric:
            return OperatorSymmetryClass::Nonsymmetric;
        case SymmetryClass::NotApplicable:
        case SymmetryClass::Unknown:
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<CoercivityClass> coercivityFrom(PositivityClass positivity)
{
    switch (positivity) {
        case PositivityClass::Positive:
            return CoercivityClass::Coercive;
        case PositivityClass::Nonnegative:
            return CoercivityClass::Semicoercive;
        case PositivityClass::Negative:
        case PositivityClass::Nonpositive:
            return CoercivityClass::NotCoercive;
        case PositivityClass::Indefinite:
            return CoercivityClass::Indefinite;
        case PositivityClass::Unknown:
            return std::nullopt;
    }
    return std::nullopt;
}

bool tangentPositivityViolation(PositivityClass positivity) noexcept
{
    return positivity == PositivityClass::Negative ||
           positivity == PositivityClass::Nonpositive;
}

PropertyStatus coefficientStatus(PositivityClass positivity) noexcept
{
    switch (positivity) {
        case PositivityClass::Positive:
        case PositivityClass::Nonnegative:
            return PropertyStatus::Preserved;
        case PositivityClass::Negative:
        case PositivityClass::Nonpositive:
        case PositivityClass::Indefinite:
            return PropertyStatus::Violated;
        case PositivityClass::Unknown:
            return PropertyStatus::Unknown;
    }
    return PropertyStatus::Unknown;
}

PositivityClass positivityFromRange(Real min_value,
                                    Real max_value,
                                    Real tolerance) noexcept
{
    if (!numeric::finiteOrdered(min_value, max_value)) {
        return PositivityClass::Unknown;
    }
    const Real tol = tolerance > Real{} ? tolerance : Real{1.0e-12};
    if (min_value > tol) {
        return PositivityClass::Positive;
    }
    if (min_value >= -tol && max_value >= -tol) {
        return PositivityClass::Nonnegative;
    }
    if (max_value < -tol) {
        return PositivityClass::Negative;
    }
    if (max_value <= tol && min_value <= tol) {
        return PositivityClass::Nonpositive;
    }
    return PositivityClass::Indefinite;
}

bool localSpectrumComplete(const CoefficientPropertySummary& summary) noexcept
{
    return summary.local_spectrum_coverage_present &&
           summary.local_spectrum_coverage_complete &&
           summary.local_spectrum_quadrature_coverage_complete &&
           summary.local_spectrum_nonfinite_count == 0u &&
           numeric::finiteOrdered(summary.local_symmetric_part_min_eigenvalue,
                                  summary.local_symmetric_part_max_eigenvalue);
}

bool localSpectrumComplete(const NonlinearTangentSummary& summary) noexcept
{
    return summary.local_spectrum_coverage_present &&
           summary.local_spectrum_coverage_complete &&
           summary.local_spectrum_quadrature_coverage_complete &&
           summary.local_spectrum_nonfinite_count == 0u &&
           numeric::finiteOrdered(summary.local_symmetric_part_min_eigenvalue,
                                  summary.local_symmetric_part_max_eigenvalue);
}

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

CertificationClass certificationFromStatus(PropertyStatus status) noexcept
{
    switch (status) {
        case PropertyStatus::Exact:
        case PropertyStatus::Preserved:
            return CertificationClass::Certified;
        case PropertyStatus::Violated:
            return CertificationClass::Violated;
        case PropertyStatus::Likely:
            return CertificationClass::NotCertified;
        case PropertyStatus::Unknown:
            return CertificationClass::Unknown;
    }
    return CertificationClass::Unknown;
}

} // namespace

std::string TemporalStabilityAnalyzer::name() const
{
    return "TemporalStabilityAnalyzer";
}

void TemporalStabilityAnalyzer::run(const ProblemAnalysisContext& context,
                                    ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->temporal_stability) {
        const Real tol = effectiveTolerance(1.0e-12);
        const bool amplification_numeric_valid =
            summary.amplification_radius_present &&
            finiteNonnegative(summary.amplification_radius);
        const bool amplification_bounded =
            amplification_numeric_valid &&
            summary.amplification_radius <= Real{1} + tol;
        const bool cfl_numeric_valid =
            !summary.cfl_estimate_present ||
            finiteNonnegative(summary.cfl_estimate);
        const bool accepted_cfl_bound_valid =
            summary.accepted_cfl_bound_present &&
            finitePositive(summary.accepted_cfl_bound);
        const bool cfl_bounded =
            summary.cfl_estimate_present &&
            cfl_numeric_valid &&
            accepted_cfl_bound_valid &&
            summary.cfl_estimate <= summary.accepted_cfl_bound + tol;
        const bool conditional =
            summary.stability_class == TemporalStabilityClass::ConditionallyStable;
        const bool ssp =
            summary.stability_class == TemporalStabilityClass::SSP;
        const bool theorem_scoped = theoremScoped(summary.stability_theorem_id);
        const bool norm_scope_complete =
            summary.stability_norm_metadata_present &&
            !summary.stability_norm_id.empty();
        const bool operator_scope_complete =
            summary.operator_scope_metadata_present &&
            !summary.operator_scope_id.empty();
        const bool time_horizon_complete =
            summary.time_horizon_metadata_present &&
            finiteNonnegative(summary.time_horizon);
        const bool nonnormal_growth_bound_valid =
            summary.nonnormal_growth_bound_present &&
            summary.nonnormal_growth_bound_finite &&
            finiteNonnegative(summary.nonnormal_growth_bound);
        const bool logarithmic_norm_bound_valid =
            summary.logarithmic_norm_bound_present &&
            summary.logarithmic_norm_bound_finite &&
            numeric::finite(summary.logarithmic_norm_bound) &&
            (!summary.accepted_logarithmic_norm_bound_present ||
             (numeric::finite(summary.accepted_logarithmic_norm_bound) &&
              summary.logarithmic_norm_bound <=
                  summary.accepted_logarithmic_norm_bound + tol));
        const bool accepted_growth_bound_valid =
            summary.accepted_nonnormal_growth_bound_present &&
            finiteNonnegative(summary.accepted_nonnormal_growth_bound) &&
            nonnormal_growth_bound_valid &&
            summary.nonnormal_growth_bound <=
                summary.accepted_nonnormal_growth_bound + tol;
        const bool cfl_margin_valid =
            !summary.cfl_margin_present ||
            finiteNonnegative(summary.cfl_margin);
        const bool cfl_certificate =
            !conditional ||
            (summary.cfl_derivation_metadata_present &&
             accepted_cfl_bound_valid &&
             !summary.cfl_bound_scope.empty() &&
             cfl_margin_valid &&
             (summary.operator_spectrum_coverage_present ||
              summary.numerical_range_coverage_present ||
              theorem_scoped));
        const bool operator_stability_region_scope =
            (summary.operator_spectrum_coverage_present &&
             summary.operator_normality_evidence_present) ||
            summary.numerical_range_coverage_present;
        const bool stability_region_certificate =
            summary.stability_region_evidence_present &&
            theorem_scoped &&
            (operator_stability_region_scope ||
            (summary.energy_norm_contractivity_evidence_present &&
             summary.contractivity_norm_metadata_present &&
             norm_scope_complete) ||
            (summary.logarithmic_norm_bound_present &&
             logarithmic_norm_bound_valid &&
             norm_scope_complete));
        const bool norm_certificate =
            (summary.energy_norm_contractivity_evidence_present &&
             summary.contractivity_norm_metadata_present &&
             norm_scope_complete) ||
            (summary.logarithmic_norm_bound_present &&
             logarithmic_norm_bound_valid &&
             norm_scope_complete &&
             (summary.numerical_range_coverage_present || theorem_scoped)) ||
            (summary.invariant_domain_evidence_present &&
             theorem_scoped) ||
            (summary.nonlinear_stability_evidence_present &&
             theorem_scoped);
        const bool nonnormal_certificate =
            theorem_scoped &&
            norm_scope_complete &&
            operator_scope_complete &&
            time_horizon_complete &&
            accepted_growth_bound_valid &&
            summary.nonnormal_operator_evidence_present &&
            (summary.pseudospectral_bound_present ||
             summary.nonnormal_growth_bound_present);
        const bool ssp_certificate =
            !ssp ||
            (summary.ssp_or_tvd_evidence_present && theorem_scoped);
        const bool certification_evidence_complete =
            cfl_certificate &&
            ssp_certificate &&
            (stability_region_certificate ||
             norm_certificate ||
             nonnormal_certificate);
        const bool nonmodal_certificate =
            norm_certificate || nonnormal_certificate ||
            (ssp && ssp_certificate);
        const bool requires_amplification = !nonmodal_certificate;

        PropertyClaim claim;
        claim.kind = PropertyKind::TemporalStability;
        claim.domain = summary.block.domain;
        claim.variables = !summary.variables.empty()
            ? summary.variables
            : blockVariables(summary.block);
        claim.tested_block_id = summary.block.operator_tag;
        if (!summary.contribution_id.empty()) {
            claim.estimate_scope = summary.contribution_id;
        }
        claim.temporal_stability_class = summary.stability_class;
        if (summary.cfl_estimate_present) {
            claim.cfl_number = summary.cfl_estimate;
        }
        claim.claim_origin = "TemporalStabilityAnalyzer";

        if (summary.stability_class == TemporalStabilityClass::Unknown) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description = "Time-integration stability class is unknown";
        } else if (!summary.stability_metadata_present) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Time-integration stability class lacks certification metadata";
        } else if (requires_amplification &&
                   !summary.amplification_radius_present) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Time-integration stability lacks amplification-radius or non-modal theorem/norm evidence";
        } else if (summary.amplification_radius_present &&
                   !amplification_numeric_valid) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Time-integration amplification radius is non-finite or negative";
        } else if (summary.cfl_estimate_present && !cfl_numeric_valid) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Time-integration CFL estimate is non-finite or negative";
        } else if (conditional &&
                   summary.accepted_cfl_bound_present &&
                   !accepted_cfl_bound_valid) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Conditional time-integration accepted CFL bound is non-finite or non-positive";
        } else if (conditional && !summary.cfl_estimate_present) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Conditional time-integration stability lacks CFL evidence";
        } else if (conditional && !summary.accepted_cfl_bound_present) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Conditional time-integration stability lacks a method-specific accepted CFL bound";
        } else if (ssp && !summary.ssp_or_tvd_evidence_present) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "SSP time-integration stability lacks SSP/TVD contractivity evidence";
        } else if (conditional &&
                   !cfl_bounded) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Conditional time-integration stability violated by CFL estimate";
        } else if (requires_amplification && !amplification_bounded) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Time-integration amplification radius exceeds the stable bound";
        } else if (!certification_evidence_complete ||
                   summary.scalar_modal_bound_only) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Time-integration scalar/modal bounds pass, but theorem-scoped stability-region, CFL, norm, nonnormal, SSP, or invariant-domain certification evidence is incomplete";
        } else {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Time-integration stability summary satisfies scalar bounds and theorem-scoped problem/norm certification metadata";
        }

        claim.addEvidence("TemporalStabilityAnalyzer",
            "TemporalStabilitySummary scheme='" + summary.time_scheme +
            "', cfl=" + std::to_string(summary.cfl_estimate) +
            ", cfl_present=" +
            std::string(summary.cfl_estimate_present ? "true" : "false") +
            ", accepted_cfl_bound=" +
            std::to_string(summary.accepted_cfl_bound) +
            ", accepted_cfl_bound_present=" +
            std::string(summary.accepted_cfl_bound_present ? "true" : "false") +
            ", cfl_bound_scope='" + summary.cfl_bound_scope + "'" +
            ", eigenvalue_scale=" +
            std::to_string(summary.eigenvalue_scale_estimate) +
            ", amplification_radius=" +
            std::to_string(summary.amplification_radius) +
            ", amplification_present=" +
            std::string(summary.amplification_radius_present ? "true" : "false") +
            ", stability_metadata=" +
            std::string(summary.stability_metadata_present ? "true" : "false") +
            ", theorem='" + summary.stability_theorem_id + "'" +
            ", stability_region=" +
            std::string(summary.stability_region_evidence_present ? "true" : "false") +
            ", spectrum_coverage=" +
            std::string(summary.operator_spectrum_coverage_present ? "true" : "false") +
            ", numerical_range_coverage=" +
            std::string(summary.numerical_range_coverage_present ? "true" : "false") +
            ", cfl_derivation=" +
            std::string(summary.cfl_derivation_metadata_present ? "true" : "false") +
            ", normality=" +
            std::string(summary.operator_normality_evidence_present ? "true" : "false") +
            ", energy_contractivity=" +
            std::string(summary.energy_norm_contractivity_evidence_present ? "true" : "false") +
            ", logarithmic_norm=" +
            std::string(summary.logarithmic_norm_bound_present ? "true" : "false") +
            ", logarithmic_norm_value=" +
            std::to_string(summary.logarithmic_norm_bound) +
            ", accepted_logarithmic_norm_present=" +
            std::string(summary.accepted_logarithmic_norm_bound_present ? "true" : "false") +
            ", accepted_logarithmic_norm_value=" +
            std::to_string(summary.accepted_logarithmic_norm_bound) +
            ", logarithmic_norm_valid=" +
            std::string(logarithmic_norm_bound_valid ? "true" : "false") +
            ", contractivity_norm=" +
            std::string(summary.contractivity_norm_metadata_present ? "true" : "false") +
            ", stability_norm_metadata=" +
            std::string(summary.stability_norm_metadata_present ? "true" : "false") +
            ", time_horizon_metadata=" +
            std::string(summary.time_horizon_metadata_present ? "true" : "false") +
            ", operator_scope_metadata=" +
            std::string(summary.operator_scope_metadata_present ? "true" : "false") +
            ", accepted_nonnormal_growth=" +
            std::string(summary.accepted_nonnormal_growth_bound_present ? "true" : "false") +
            ", pseudospectral=" +
            std::string(summary.pseudospectral_bound_present ? "true" : "false") +
            ", nonnormal_growth=" +
            std::string(summary.nonnormal_growth_bound_present ? "true" : "false") +
            ", nonnormal_growth_value=" +
            std::to_string(summary.nonnormal_growth_bound) +
            ", accepted_nonnormal_growth_value=" +
            std::to_string(summary.accepted_nonnormal_growth_bound) +
            ", nonnormal_growth_valid=" +
            std::string(nonnormal_growth_bound_valid ? "true" : "false") +
            ", ssp_tvd=" +
            std::string(summary.ssp_or_tvd_evidence_present ? "true" : "false"),
            claim.confidence);
        if (claim.status == PropertyStatus::Violated) {
            addWarning(report,
                "Temporal stability violation from summary '" +
                summary.time_scheme + "'");
        }
        report.claims.push_back(std::move(claim));
    }
}

std::string EnergyEntropyLawAnalyzer::name() const
{
    return "EnergyEntropyLawAnalyzer";
}

void EnergyEntropyLawAnalyzer::run(const ProblemAnalysisContext& context,
                                   ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->energy_entropy) {
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(summary.balance_tolerance);
        const Real tol = effectiveTolerance(summary.balance_tolerance);
        const bool balance_value_valid =
            numeric::finite(summary.observed_discrete_balance);
        const bool production_value_valid =
            summary.expected_production_sign == BalanceSignClass::Unconstrained ||
            numeric::finite(summary.observed_production);
        const bool numeric_evidence_valid =
            tolerance_declared && balance_value_valid && production_value_valid;
        const bool sign_ok =
            numeric_evidence_valid &&
            productionSignSatisfied(summary.expected_production_sign,
                                    summary.observed_production,
                                    tol);
        const bool balance_ok =
            numeric_evidence_valid &&
            std::abs(summary.observed_discrete_balance) <= tol;
        const bool violated =
            summary.violation_count > 0u ||
            (numeric_evidence_valid && (!sign_ok || !balance_ok));
        const bool metadata_complete = energyEntropyMetadataComplete(summary);

        PropertyClaim claim;
        claim.kind = lawPropertyKind(summary.law_kind);
        claim.status = violated
            ? PropertyStatus::Violated
            : (summary.expected_production_sign == BalanceSignClass::Unknown ||
               !numeric_evidence_valid
                ? PropertyStatus::Unknown
                : (metadata_complete ? PropertyStatus::Preserved
                                     : PropertyStatus::Unknown));
        claim.confidence = claim.status == PropertyStatus::Unknown
            ? AnalysisConfidence::Medium
            : AnalysisConfidence::High;
        claim.certification_class =
            claim.status == PropertyStatus::Unknown
                ? CertificationClass::NotCertified
                : certificationFromStatus(claim.status);
        claim.estimate_scope = summary.energy_entropy_id;
        claim.description = claim.status == PropertyStatus::Violated
            ? "Discrete energy/entropy balance violates the declared production sign or tolerance"
            : (claim.status == PropertyStatus::Preserved
                ? "Discrete energy/entropy balance satisfies the declared production sign and tolerance"
                : (summary.expected_production_sign == BalanceSignClass::Unknown
                    ? "Discrete energy/entropy balance has unknown expected production sign or invalid finite numeric evidence"
                    : "Discrete energy/entropy balance passes numerically but lacks theorem-scoped energy functional/norm or entropy-variable/flux/dissipation metadata"));
        claim.claim_origin = "EnergyEntropyLawAnalyzer";
        claim.addEvidence("EnergyEntropyLawAnalyzer",
            "EnergyEntropySummary id='" + summary.energy_entropy_id +
            "', observed_balance=" +
            std::to_string(summary.observed_discrete_balance) +
            ", observed_production=" +
            std::to_string(summary.observed_production) +
            ", violations=" + std::to_string(summary.violation_count) +
            ", theorem='" + summary.energy_entropy_theorem_id + "'" +
            ", energy_functional='" + summary.energy_functional_id + "'" +
            ", energy_norm='" + summary.energy_norm_id + "'" +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", finite_balance=" +
            std::string(balance_value_valid ? "true" : "false") +
            ", finite_production=" +
            std::string(production_value_valid ? "true" : "false") +
            ", metadata_complete=" +
            std::string(metadata_complete ? "true" : "false"),
            claim.confidence);
        if (claim.status == PropertyStatus::Violated) {
            addWarning(report,
                "Energy/entropy balance violation for summary '" +
                summary.energy_entropy_id + "'");
        } else if (!numeric_evidence_valid) {
            addWarning(report,
                "Energy/entropy summary '" + summary.energy_entropy_id +
                "' has invalid tolerance, balance, or production evidence");
        }
        report.claims.push_back(std::move(claim));
    }
}

std::string CoefficientConstitutiveAnalyzer::name() const
{
    return "CoefficientConstitutiveAnalyzer";
}

void CoefficientConstitutiveAnalyzer::run(const ProblemAnalysisContext& context,
                                          ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->coefficient_properties) {
        const bool local_spectrum_certifies_current_state =
            localSpectrumComplete(summary);
        const auto effective_positivity =
            local_spectrum_certifies_current_state
                ? positivityFromRange(
                      summary.local_symmetric_part_min_eigenvalue,
                      summary.local_symmetric_part_max_eigenvalue,
                      summary.positivity_tolerance)
                : summary.positivity;
        const auto status = coefficientStatus(effective_positivity);
        const bool positive_or_nonnegative =
            effective_positivity == PositivityClass::Positive ||
            effective_positivity == PositivityClass::Nonnegative;
        const bool coverage_complete =
            coefficientCoverageComplete(summary) ||
            local_spectrum_certifies_current_state;
        const bool contradicted_bound =
            positive_or_nonnegative &&
            coefficientDeclaredBoundContradictsPositivity(summary);

        PropertyClaim positivity;
        positivity.kind = PropertyKind::CoefficientPositivity;
        positivity.status =
            contradicted_bound
                ? PropertyStatus::Violated
                : (positive_or_nonnegative && !coverage_complete
                ? PropertyStatus::Likely
                : status);
        positivity.confidence = status == PropertyStatus::Unknown
            ? AnalysisConfidence::Medium
            : (contradicted_bound
                   ? AnalysisConfidence::High
                   : (positive_or_nonnegative && !coverage_complete
                   ? AnalysisConfidence::Medium
                   : AnalysisConfidence::High));
        positivity.domain = summary.domain;
        positivity.certification_class =
            contradicted_bound
                ? CertificationClass::Violated
                : (positive_or_nonnegative && !coverage_complete
                ? CertificationClass::NotCertified
                : certificationFromStatus(status));
        positivity.coefficient_id = summary.coefficient;
        positivity.claim_origin = "CoefficientConstitutiveAnalyzer";
        positivity.description = status == PropertyStatus::Violated
            ? "Coefficient/constitutive summary violates positivity requirements"
            : (contradicted_bound
                ? "Coefficient/constitutive lower-bound evidence contradicts the declared positivity class"
                : (positivity.status == PropertyStatus::Preserved
                ? "Coefficient/constitutive summary preserves positivity requirements with scoped coverage metadata"
                : (positivity.status == PropertyStatus::Likely
                    ? "Coefficient/constitutive summary reports positive sign but lacks full coverage metadata"
                    : "Coefficient/constitutive positivity is unknown")));
        positivity.addEvidence("CoefficientConstitutiveAnalyzer",
            "CoefficientPropertySummary coefficient='" + summary.coefficient +
            "', min_eigenvalue=" +
            std::to_string(summary.min_eigenvalue) +
            ", max_eigenvalue=" +
            std::to_string(summary.max_eigenvalue) +
            ", positivity=" +
            std::to_string(static_cast<int>(effective_positivity)) +
            ", coverage_complete=" +
            std::string(coverage_complete ? "true" : "false") +
            ", local_spectrum=" +
            std::string(local_spectrum_certifies_current_state ? "true" : "false") +
            ", local_min=" +
            std::to_string(summary.local_symmetric_part_min_eigenvalue) +
            ", local_max=" +
            std::to_string(summary.local_symmetric_part_max_eigenvalue) +
            ", local_samples=" +
            std::to_string(summary.local_spectrum_sample_count),
            positivity.confidence);
        report.claims.push_back(std::move(positivity));

        const Real anisotropy = summary.anisotropy_ratio;
        const Real contrast = summary.contrast_ratio;
        const Real worst_scale = std::max(anisotropy, contrast);
        if (worst_scale <= Real{} &&
            !summary.state_dependent &&
            !summary.time_dependent) {
            continue;
        }

        PropertyClaim robustness;
        robustness.kind = PropertyKind::ParameterRobustness;
        robustness.domain = summary.domain;
        robustness.coefficient_id = summary.coefficient;
        robustness.claim_origin = "CoefficientConstitutiveAnalyzer";
        robustness.confidence = AnalysisConfidence::Medium;
        const bool robustness_metadata_complete =
            summary.robustness_certificate_present &&
            theoremScoped(summary.robustness_theorem_id) &&
            !summary.robustness_certificate_scope.empty() &&
            summary.robustness_norm_metadata_present &&
            !summary.robustness_norm_id.empty() &&
            summary.robustness_parameter_range_metadata_present &&
            !summary.robustness_parameter_range_scope.empty() &&
            summary.robustness_mesh_family_metadata_present &&
            !summary.robustness_mesh_family_scope.empty() &&
            summary.robustness_uniform_constant_present &&
            finitePositive(summary.robustness_uniform_constant);
        if (robustness_metadata_complete) {
            robustness.status = PropertyStatus::Preserved;
            robustness.certification_class = CertificationClass::Certified;
            robustness.description =
                "Coefficient contrast and anisotropy are covered by reported robustness evidence";
        } else if (summary.robustness_certificate_present) {
            robustness.status = PropertyStatus::Likely;
            robustness.certification_class = CertificationClass::NotCertified;
            robustness.description =
                "Coefficient robustness certificate is present but lacks theorem, norm, range, mesh-family, or uniform-constant metadata";
        } else if (worst_scale > Real{1.0e3} ||
                   summary.state_dependent ||
                   summary.time_dependent) {
            robustness.status = PropertyStatus::Likely;
            robustness.certification_class = CertificationClass::NotCertified;
            robustness.description =
                "Coefficient contrast, anisotropy, or state dependence requires robustness evidence";
        } else {
            robustness.status = PropertyStatus::Likely;
            robustness.certification_class = CertificationClass::NotCertified;
            robustness.description =
                "Coefficient contrast and anisotropy are modest, but robustness is not certified without solver/norm evidence";
        }
        robustness.addEvidence("CoefficientConstitutiveAnalyzer",
            "anisotropy_ratio=" + std::to_string(anisotropy) +
            ", contrast_ratio=" + std::to_string(contrast) +
            ", state_dependent=" +
            std::string(summary.state_dependent ? "true" : "false") +
            ", time_dependent=" +
            std::string(summary.time_dependent ? "true" : "false") +
            ", robustness_certificate=" +
            std::string(summary.robustness_certificate_present ? "true" : "false") +
            ", robustness_scope='" + summary.robustness_certificate_scope + "'" +
            ", theorem='" + summary.robustness_theorem_id + "'" +
            ", norm_metadata=" +
            std::string(summary.robustness_norm_metadata_present ? "true" : "false") +
            ", parameter_range_metadata=" +
            std::string(summary.robustness_parameter_range_metadata_present ? "true" : "false") +
            ", mesh_family_metadata=" +
            std::string(summary.robustness_mesh_family_metadata_present ? "true" : "false") +
            ", uniform_constant_present=" +
            std::string(summary.robustness_uniform_constant_present ? "true" : "false"),
            robustness.confidence);
        report.claims.push_back(std::move(robustness));
    }
}

std::string NonlinearTangentAnalyzer::name() const
{
    return "NonlinearTangentAnalyzer";
}

void NonlinearTangentAnalyzer::run(const ProblemAnalysisContext& context,
                                   ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->nonlinear_tangents) {
        const bool local_spectrum_certifies_current_state =
            localSpectrumComplete(summary);
        const auto effective_tangent_positivity =
            local_spectrum_certifies_current_state
                ? positivityFromRange(
                      summary.local_symmetric_part_min_eigenvalue,
                      summary.local_symmetric_part_max_eigenvalue,
                      summary.finite_difference_tolerance)
                : summary.tangent_positivity;
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(
                summary.finite_difference_tolerance);
        const bool action_error_valid =
            numeric::finiteNonnegative(
                summary.finite_difference_action_error);
        const Real tol = effectiveTolerance(summary.finite_difference_tolerance);
        const bool action_checked = summary.jacobian_action_available;
        const bool action_numeric_valid =
            action_checked && tolerance_declared && action_error_valid;
        const bool action_numeric_invalid =
            action_checked && !action_numeric_valid;
        const bool action_ok =
            action_numeric_valid &&
            summary.finite_difference_action_error <= tol;
        const bool action_violation =
            action_numeric_valid &&
            summary.finite_difference_action_error > tol;
        const bool indefinite_tangent =
            effective_tangent_positivity == PositivityClass::Indefinite;
        const bool inconsistent =
            summary.tangent_consistency == TangentConsistencyClass::Frozen ||
            summary.tangent_consistency == TangentConsistencyClass::Inconsistent ||
            action_violation ||
            summary.newton_stagnation_count > 0u ||
            tangentPositivityViolation(effective_tangent_positivity);

        PropertyClaim claim;
        claim.kind = PropertyKind::NonlinearTangentStructure;
        claim.variables = blockVariables(summary.block);
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag.empty()
            ? summary.residual_id
            : summary.block.operator_tag;
        claim.operator_symmetry_class =
            operatorSymmetryFrom(summary.tangent_symmetry);
        claim.coercivity_class = coercivityFrom(effective_tangent_positivity);
        claim.claim_origin = "NonlinearTangentAnalyzer";

        if (tangentPositivityViolation(effective_tangent_positivity)) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Nonlinear tangent positivity/definiteness metadata violates solver stability requirements";
        } else if (summary.tangent_consistency == TangentConsistencyClass::Unknown &&
                   !action_checked &&
                   summary.tangent_symmetry == SymmetryClass::Unknown &&
                   summary.tangent_positivity == PositivityClass::Unknown) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Nonlinear residual/tangent consistency is unknown";
        } else if (inconsistent) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Nonlinear residual/tangent consistency is violated by tangent metadata";
        } else if (summary.tangent_consistency == TangentConsistencyClass::Exact) {
            if (action_ok) {
                claim.status = PropertyStatus::Preserved;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Certified;
                claim.description =
                    "Nonlinear residual/tangent consistency is certified by exact tangent metadata and Jacobian-action evidence";
            } else if (action_numeric_invalid) {
                claim.status = PropertyStatus::Unknown;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.description =
                    "Nonlinear tangent is marked exact, but finite-difference action evidence has invalid finite tolerance or residual";
            } else {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.description =
                    "Nonlinear tangent is marked exact, but certification requires Jacobian-action or finite-difference consistency evidence";
            }
        } else {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Nonlinear tangent is approximate but no violation was reported";
        }

        claim.addEvidence("NonlinearTangentAnalyzer",
            "NonlinearTangentSummary residual='" + summary.residual_id +
            "', consistency=" +
            tangentConsistencyName(summary.tangent_consistency) +
            ", fd_action_error=" +
            std::to_string(summary.finite_difference_action_error) +
            ", tolerance=" + std::to_string(tol) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", finite_action_error=" +
            std::string(action_error_valid ? "true" : "false") +
            ", stagnation_count=" +
            std::to_string(summary.newton_stagnation_count) +
            ", tangent_symmetry=" +
            symmetryName(summary.tangent_symmetry) +
            ", tangent_positivity=" +
            positivityName(effective_tangent_positivity) +
            ", local_spectrum=" +
            std::string(local_spectrum_certifies_current_state ? "true" : "false") +
            ", local_min=" +
            std::to_string(summary.local_symmetric_part_min_eigenvalue) +
            ", local_max=" +
            std::to_string(summary.local_symmetric_part_max_eigenvalue) +
            ", local_samples=" +
            std::to_string(summary.local_spectrum_sample_count) +
            ", jacobian_nonsingularity=" +
            std::string(summary.jacobian_nonsingularity_evidence_present ? "true" : "false") +
            ", residual_decrease=" +
            std::string(summary.residual_decrease_evidence_present ? "true" : "false") +
            ", globalization=" +
            std::string(summary.line_search_or_trust_region_evidence_present ? "true" : "false"),
            claim.confidence);
        if (claim.status == PropertyStatus::Violated) {
            addWarning(report,
                "Nonlinear tangent consistency violation for summary '" +
                summary.residual_id + "'");
        } else if (action_numeric_invalid) {
            addWarning(report,
                "Nonlinear tangent summary '" + summary.residual_id +
                "' has invalid finite-difference action tolerance or residual");
        }
        report.claims.push_back(std::move(claim));

        if (indefinite_tangent) {
            PropertyClaim resolution_claim;
            resolution_claim.kind = PropertyKind::IndefiniteOperatorResolution;
            resolution_claim.status = PropertyStatus::Unknown;
            resolution_claim.confidence = AnalysisConfidence::Medium;
            resolution_claim.certification_class = CertificationClass::NotCertified;
            resolution_claim.variables = blockVariables(summary.block);
            resolution_claim.domain = summary.block.domain;
            resolution_claim.tested_block_id = summary.block.operator_tag.empty()
                ? summary.residual_id
                : summary.block.operator_tag;
            resolution_claim.estimate_scope = summary.residual_id;
            resolution_claim.coercivity_class = CoercivityClass::Indefinite;
            resolution_claim.claim_origin = "NonlinearTangentAnalyzer";
            resolution_claim.description =
                "Indefinite nonlinear tangent requires an indefinite-system resolution route rather than a positivity violation";
            resolution_claim.addEvidence("NonlinearTangentAnalyzer",
                "Tangent symmetric-part metadata is indefinite; certification is delegated to Schur, minimum-residual, Krylov, or equivalent indefinite solver evidence",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(resolution_claim));
        }
    }
}

std::string NullspaceDegeneracyAnalyzer::name() const
{
    return "NullspaceDegeneracyAnalyzer";
}

void NullspaceDegeneracyAnalyzer::run(const ProblemAnalysisContext& context,
                                      ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    AnalyzerRunLogSummary log;
    log.analyzer = "NullspaceDegeneracyAnalyzer";
    log.summary_id = "assembled-nullspace-degeneracy";
    log.status = summaries->nullspace_degeneracies.empty()
        ? "unavailable"
        : "evaluated";
    log.attempted_count =
        static_cast<std::uint64_t>(summaries->nullspace_degeneracies.size());

    for (const auto& summary : summaries->nullspace_degeneracies) {
        const bool degenerate =
            summary.degeneracy_class == DegeneracyClass::DegenerateDiagnostic ||
            summary.degeneracy_class == DegeneracyClass::GaugeLikeNullspace ||
            summary.degeneracy_class == DegeneracyClass::UnanchoredKernel ||
            summary.degeneracy_class == DegeneracyClass::ProjectedKernel;
        if (degenerate) {
            ++log.incomplete_count;
        }
        log.detail_lines.push_back(
            "id='" + summary.degeneracy_id +
            "' nullity=" + std::to_string(summary.nullity) +
            " near_zero_pivots=" +
            std::to_string(summary.near_zero_pivot_count) +
            " constrained_dofs=" +
            std::to_string(summary.constrained_dof_count) +
            " class=" +
            std::to_string(static_cast<int>(summary.degeneracy_class)) +
            " reason='" + summary.reason + "'");

        if (!degenerate) {
            continue;
        }

        PropertyClaim claim;
        claim.kind = PropertyKind::InfSupCondition;
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.domain = summary.block.domain;
        claim.variables = summary.affected_variables.empty()
            ? blockVariables(summary.block)
            : summary.affected_variables;
        claim.inf_sup_class = InfSupClass::Unknown;
        claim.applicability_class = ApplicabilityClass::NotApplicable;
        claim.certification_class = CertificationClass::NotCertified;
        claim.nullspace_handling_class = summary.nullspace_handling;
        claim.tested_block_id = summary.block.operator_tag;
        claim.estimate_scope = summary.degeneracy_id;
        claim.claim_origin = "NullspaceDegeneracyAnalyzer";
        claim.description =
            "Inf-sup and solver-stability diagnostics are out of scope for this degenerate reduced operator: " +
            summary.reason;
        claim.addEvidence("NullspaceDegeneracyAnalyzer",
            "NullspaceDegeneracySummary rank=" +
            std::to_string(summary.estimated_rank) +
            ", nullity=" + std::to_string(summary.nullity) +
            ", near_zero_rows=" +
            std::to_string(summary.near_zero_row_count) +
            ", near_zero_pivots=" +
            std::to_string(summary.near_zero_pivot_count) +
            ", free_dofs=" + std::to_string(summary.free_dof_count) +
            ", constrained_dofs=" +
            std::to_string(summary.constrained_dof_count) +
            ", tolerance=" + std::to_string(summary.rank_tolerance) +
            ", class=" +
            std::to_string(static_cast<int>(summary.degeneracy_class)),
            AnalysisConfidence::Medium);
        report.claims.push_back(std::move(claim));
    }

    if (log.attempted_count > 0u) {
        report.run_logs.push_back(std::move(log));
    }
}

std::string OperatorApplicabilityAnalyzer::name() const
{
    return "OperatorApplicabilityAnalyzer";
}

void OperatorApplicabilityAnalyzer::run(const ProblemAnalysisContext& context,
                                        ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->applicability) {
        PropertyKind kind = PropertyKind::MatrixMonotonicityRisk;
        if (summary.theorem_family == TheoremFamily::ScalarDMP) {
            kind = PropertyKind::DiscreteMaximumPrinciple;
        } else if (summary.theorem_family == TheoremFamily::MMatrix) {
            kind = PropertyKind::MMatrixStructure;
        } else if (summary.theorem_family == TheoremFamily::InvariantDomain) {
            kind = PropertyKind::InvariantDomainPreservation;
        } else if (summary.theorem_family == TheoremFamily::InfSup ||
                   summary.theorem_family == TheoremFamily::Fortin) {
            kind = PropertyKind::InfSupCondition;
        } else if (summary.theorem_family == TheoremFamily::Schur) {
            kind = PropertyKind::IndefiniteOperatorResolution;
        } else if (summary.theorem_family == TheoremFamily::TemporalCFL) {
            kind = PropertyKind::TemporalStability;
        } else if (summary.theorem_family == TheoremFamily::EnergyEntropy) {
            kind = PropertyKind::EnergyStability;
        }

        PropertyClaim claim;
        claim.kind = kind;
        claim.status = summary.applicability == ApplicabilityClass::NotApplicable
            ? PropertyStatus::Unknown
            : (summary.applicability == ApplicabilityClass::Applicable
                ? PropertyStatus::Likely
                : PropertyStatus::Unknown);
        claim.confidence = AnalysisConfidence::Medium;
        claim.domain = summary.block.domain;
        claim.variables = summary.variables.empty()
            ? blockVariables(summary.block)
            : summary.variables;
        claim.applicability_class = summary.applicability;
        claim.certification_class =
            summary.applicability == ApplicabilityClass::NotApplicable
                ? CertificationClass::Unknown
                : CertificationClass::NotCertified;
        claim.tested_block_id = summary.block.operator_tag;
        claim.claim_origin = "OperatorApplicabilityAnalyzer";
        claim.description =
            summary.applicability == ApplicabilityClass::NotApplicable
                ? "Theorem family is structurally not applicable to this operator"
                : (summary.applicability == ApplicabilityClass::Applicable
                    ? "Theorem family is structurally applicable, pending numeric evidence"
                    : "Theorem-family applicability is unknown");
        claim.addEvidence("OperatorApplicabilityAnalyzer",
            "ApplicabilitySummary family=" +
            std::to_string(static_cast<int>(summary.theorem_family)) +
            ", applicability=" +
            std::to_string(static_cast<int>(summary.applicability)) +
            ", field_descriptors=" +
            std::string(summary.inferred_from_field_descriptors ? "true" : "false") +
            ", contribution_traits=" +
            std::string(summary.inferred_from_contribution_traits ? "true" : "false") +
            ", block_structure=" +
            std::string(summary.inferred_from_block_structure ? "true" : "false") +
            ", reason='" + summary.reason + "'",
            AnalysisConfidence::Medium);
        report.claims.push_back(std::move(claim));
    }
}

std::string robustnessTrendGroupKey(const RobustnessTrendSummary& summary)
{
    return summary.metric_name + "|" +
           (summary.operator_tag.empty() ? summary.block.operator_tag
                                         : summary.operator_tag) +
           "|" + summary.case_name;
}

bool robustnessMetricDeterioratesWhenIncreasing(
    const std::string& metric_name)
{
    return metric_name.find("condition") != std::string::npos ||
           metric_name.find("norm") != std::string::npos;
}

RobustnessTrendClass classifyAggregatedTrend(const std::string& metric_name,
                                             Real first_value,
                                             Real last_value)
{
    if (!numeric::finite(first_value) || !numeric::finite(last_value)) {
        return RobustnessTrendClass::Unknown;
    }
    const Real delta = last_value - first_value;
    const Real scale = std::max({Real{1}, std::abs(first_value),
                                 std::abs(last_value)});
    if (std::abs(delta) <= Real{1.0e-12} * scale) {
        return RobustnessTrendClass::Stable;
    }
    const bool increasing_is_bad =
        robustnessMetricDeterioratesWhenIncreasing(metric_name);
    if (increasing_is_bad) {
        return delta > Real{} ? RobustnessTrendClass::Deteriorating
                              : RobustnessTrendClass::Improving;
    }
    return delta > Real{} ? RobustnessTrendClass::Improving
                          : RobustnessTrendClass::Deteriorating;
}

std::vector<RobustnessTrendSummary> aggregateRobustnessTrendRecords(
    const std::vector<RobustnessTrendSummary>& input)
{
    std::vector<RobustnessTrendSummary> output;
    std::vector<bool> used(input.size(), false);

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (used[i]) {
            continue;
        }
        if (input[i].sample_count > 1u) {
            output.push_back(input[i]);
            used[i] = true;
            continue;
        }

        const auto key = robustnessTrendGroupKey(input[i]);
        std::vector<std::size_t> group;
        for (std::size_t j = i; j < input.size(); ++j) {
            if (!used[j] &&
                input[j].sample_count <= 1u &&
                robustnessTrendGroupKey(input[j]) == key) {
                group.push_back(j);
                used[j] = true;
            }
        }
        if (group.size() <= 1u) {
            output.push_back(input[i]);
            continue;
        }
        std::sort(group.begin(),
                  group.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      const auto& a = input[lhs];
                      const auto& b = input[rhs];
                      if (!a.dof_counts.empty() && !b.dof_counts.empty() &&
                          a.dof_counts.front() != b.dof_counts.front()) {
                          return a.dof_counts.front() < b.dof_counts.front();
                      }
                      if (!a.mesh_sizes.empty() && !b.mesh_sizes.empty() &&
                          a.mesh_sizes.front() != b.mesh_sizes.front()) {
                          return a.mesh_sizes.front() > b.mesh_sizes.front();
                      }
                      if (!a.run_ids.empty() && !b.run_ids.empty() &&
                          a.run_ids.front() != b.run_ids.front()) {
                          return a.run_ids.front() < b.run_ids.front();
                      }
                      return lhs < rhs;
                  });

        RobustnessTrendSummary aggregate = input[group.front()];
        aggregate.sample_count = 0u;
        aggregate.run_ids.clear();
        aggregate.dof_counts.clear();
        aggregate.mesh_sizes.clear();
        aggregate.min_value = std::numeric_limits<Real>::infinity();
        aggregate.max_value = -std::numeric_limits<Real>::infinity();
        aggregate.explicit_uniform_lower_bound_present = true;
        aggregate.explicit_uniform_lower_bound =
            std::numeric_limits<Real>::infinity();
        aggregate.explicit_uniform_upper_bound_present = true;
        aggregate.explicit_uniform_upper_bound =
            -std::numeric_limits<Real>::infinity();
        aggregate.pass_threshold_present = false;
        aggregate.comparable_parameter_scope = true;

        const Real first_value = input[group.front()].min_value;
        const Real last_value = input[group.back()].max_value;
        const auto& first_hash = input[group.front()].parameter_hash;
        for (const auto idx : group) {
            const auto& sample = input[idx];
            aggregate.sample_count +=
                std::max<std::uint64_t>(sample.sample_count, 1u);
            aggregate.run_ids.insert(aggregate.run_ids.end(),
                                     sample.run_ids.begin(),
                                     sample.run_ids.end());
            aggregate.dof_counts.insert(aggregate.dof_counts.end(),
                                        sample.dof_counts.begin(),
                                        sample.dof_counts.end());
            aggregate.mesh_sizes.insert(aggregate.mesh_sizes.end(),
                                        sample.mesh_sizes.begin(),
                                        sample.mesh_sizes.end());
            aggregate.min_value = std::min(aggregate.min_value,
                                           sample.min_value);
            aggregate.max_value = std::max(aggregate.max_value,
                                           sample.max_value);
            if (sample.parameter_hash != first_hash) {
                aggregate.comparable_parameter_scope = false;
            }
            if (sample.pass_threshold_present) {
                aggregate.pass_threshold_present = true;
                aggregate.pass_threshold =
                    std::max(aggregate.pass_threshold,
                             sample.pass_threshold);
            }
            if (sample.explicit_uniform_lower_bound_present &&
                numeric::finite(sample.explicit_uniform_lower_bound)) {
                aggregate.explicit_uniform_lower_bound =
                    std::min(aggregate.explicit_uniform_lower_bound,
                             sample.explicit_uniform_lower_bound);
            } else {
                aggregate.explicit_uniform_lower_bound_present = false;
            }
            if (sample.explicit_uniform_upper_bound_present &&
                numeric::finite(sample.explicit_uniform_upper_bound)) {
                aggregate.explicit_uniform_upper_bound =
                    std::max(aggregate.explicit_uniform_upper_bound,
                             sample.explicit_uniform_upper_bound);
            } else {
                aggregate.explicit_uniform_upper_bound_present = false;
            }
        }
        if (!aggregate.explicit_uniform_lower_bound_present) {
            aggregate.explicit_uniform_lower_bound = Real{};
        }
        if (!aggregate.explicit_uniform_upper_bound_present) {
            aggregate.explicit_uniform_upper_bound = Real{};
        }

        Real denominator = static_cast<Real>(group.size() - 1u);
        if (!aggregate.dof_counts.empty() &&
            aggregate.dof_counts.front() != aggregate.dof_counts.back()) {
            denominator = static_cast<Real>(
                aggregate.dof_counts.back()) -
                static_cast<Real>(aggregate.dof_counts.front());
        }
        aggregate.slope_estimate =
            denominator != Real{} ? (last_value - first_value) / denominator
                                  : Real{};
        aggregate.trend_class =
            aggregate.comparable_parameter_scope
                ? classifyAggregatedTrend(aggregate.metric_name,
                                          first_value,
                                          last_value)
                : RobustnessTrendClass::Incomparable;
        output.push_back(std::move(aggregate));
    }

    return output;
}

std::string RobustnessTrendAnalyzer::name() const
{
    return "RobustnessTrendAnalyzer";
}

void RobustnessTrendAnalyzer::run(const ProblemAnalysisContext& context,
                                  ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;
    const auto trends =
        aggregateRobustnessTrendRecords(summaries->robustness_trends);

    AnalyzerRunLogSummary log;
    log.analyzer = "RobustnessTrendAnalyzer";
    log.summary_id = "robustness-trends";
    log.status = trends.empty() ? "unavailable" : "evaluated";
    log.attempted_count =
        static_cast<std::uint64_t>(trends.size());

    for (const auto& summary : trends) {
        const bool enough_samples = summary.sample_count >= 2u;
        const bool increasing_is_bad =
            robustnessMetricDeterioratesWhenIncreasing(summary.metric_name);
        const bool lower_bound_certified =
            !increasing_is_bad &&
            summary.explicit_uniform_lower_bound_present &&
            finitePositive(summary.explicit_uniform_lower_bound) &&
            (!summary.pass_threshold_present ||
             summary.explicit_uniform_lower_bound >= summary.pass_threshold);
        const bool upper_bound_certified =
            increasing_is_bad &&
            summary.explicit_uniform_upper_bound_present &&
            finitePositive(summary.explicit_uniform_upper_bound) &&
            (!summary.pass_threshold_present ||
             summary.explicit_uniform_upper_bound <= summary.pass_threshold);
        const bool certified =
            enough_samples &&
            summary.comparable_parameter_scope &&
            (lower_bound_certified || upper_bound_certified);
        const bool deteriorating =
            summary.trend_class == RobustnessTrendClass::Deteriorating;
        const bool incomparable =
            !summary.comparable_parameter_scope ||
            summary.trend_class == RobustnessTrendClass::Incomparable;

        if (certified) {
            ++log.certified_count;
        } else if (!enough_samples || incomparable) {
            ++log.incomplete_count;
        }
        log.detail_lines.push_back(
            "metric='" + summary.metric_name +
            "' samples=" + std::to_string(summary.sample_count) +
            " trend=" + std::to_string(static_cast<int>(summary.trend_class)) +
            " min=" + std::to_string(summary.min_value) +
            " max=" + std::to_string(summary.max_value));

        PropertyClaim claim;
        claim.kind = summary.metric_name.find("inf_sup") != std::string::npos
            ? PropertyKind::InfSupCondition
            : PropertyKind::ParameterRobustness;
        claim.variables = summary.variables.empty()
            ? blockVariables(summary.block)
            : summary.variables;
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag;
        claim.estimate_scope = summary.metric_name;
        claim.claim_origin = "RobustnessTrendAnalyzer";

        if (certified) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                increasing_is_bad
                    ? "Cross-run trend has comparable samples and an explicit uniform upper-bound certificate"
                    : "Cross-run trend has comparable samples and an explicit uniform lower-bound certificate";
        } else if (deteriorating && enough_samples) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Cross-run trend deteriorates; this is robustness evidence, not a single-run theorem failure";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                enough_samples
                    ? "Cross-run trend evidence is available but lacks comparable scope or an explicit uniform lower bound"
                    : "Robustness is not certified from a single run";
        }

        claim.addEvidence("RobustnessTrendAnalyzer",
            "RobustnessTrendSummary metric='" + summary.metric_name +
            "', samples=" + std::to_string(summary.sample_count) +
            ", run_ids=" + std::to_string(summary.run_ids.size()) +
            ", min=" + std::to_string(summary.min_value) +
            ", max=" + std::to_string(summary.max_value) +
            ", slope=" + std::to_string(summary.slope_estimate) +
            ", comparable_scope=" +
            std::string(summary.comparable_parameter_scope ? "true" : "false") +
            ", uniform_lower_bound=" +
            std::string(summary.explicit_uniform_lower_bound_present ? "true" : "false") +
            ", uniform_upper_bound=" +
            std::string(summary.explicit_uniform_upper_bound_present ? "true" : "false") +
            ", upper_bound=" +
            std::to_string(summary.explicit_uniform_upper_bound),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    if (log.attempted_count > 0u) {
        report.run_logs.push_back(std::move(log));
    }
}

std::string SchurQualityAnalyzer::name() const
{
    return "SchurQualityAnalyzer";
}

namespace {

bool schurNullspaceHandlingAcceptable(NullspaceHandlingClass handling) noexcept
{
    return handling == NullspaceHandlingClass::NotApplicable ||
           handling == NullspaceHandlingClass::AnchoredByConstraints ||
           handling == NullspaceHandlingClass::ProjectedOut;
}

bool schurQualityCertificationComplete(
    const SchurComplementSummary& summary) noexcept
{
    const bool spectral_bounds_valid =
        summary.spectral_equivalence_bounds_present &&
        numeric::finitePositiveOrdered(
            summary.spectral_equivalence_lower_bound,
            summary.spectral_equivalence_upper_bound);
    const bool preconditioner_bounds_valid =
        summary.preconditioner_equivalence_bounds_present &&
        numeric::finitePositiveOrdered(
            summary.preconditioner_equivalence_lower_bound,
            summary.preconditioner_equivalence_upper_bound);
    return summary.schur_available &&
           summary.reduction_exact_for_analysis &&
           summary.inf_sup_evidence_present &&
           summary.nullspace_handling_evidence_present &&
           schurNullspaceHandlingAcceptable(summary.nullspace_handling) &&
           spectral_bounds_valid &&
           preconditioner_bounds_valid;
}

} // namespace

void SchurQualityAnalyzer::run(const ProblemAnalysisContext& context,
                               ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->schur_complements) {
        const bool robust_certified =
            summary.condition_risk_class ==
                SchurQualityClass::PreconditionedRobust &&
            schurQualityCertificationComplete(summary);
        PropertyClaim claim;
        claim.kind = PropertyKind::IndefiniteOperatorResolution;
        claim.variables = summary.variables.empty()
            ? blockVariables(summary.block)
            : summary.variables;
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag.empty()
            ? summary.schur_id
            : summary.block.operator_tag;
        claim.estimate_scope = summary.schur_id;
        claim.claim_origin = "SchurQualityAnalyzer";
        claim.nullspace_handling_class = summary.nullspace_handling;

        if (!summary.schur_available) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.reduced_definiteness_class = CertificationClass::Unknown;
            claim.description = "Schur complement evidence is unavailable";
        } else if (robust_certified) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.reduced_definiteness_class = CertificationClass::Certified;
            claim.description =
                "Configured Schur/preconditioner evidence has finite equivalence bounds, exact reduction, inf-sup scope, and acceptable nullspace handling";
        } else if (summary.condition_risk_class ==
                   SchurQualityClass::PreconditionedRobust) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.reduced_definiteness_class = CertificationClass::NotCertified;
            claim.description =
                "Schur quality is classified as preconditioned robust, but certification requires finite spectral/preconditioner equivalence bounds, exact reduction, inf-sup evidence, and nullspace scope";
        } else if (summary.condition_risk_class ==
                   SchurQualityClass::PreconditionedRisk) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.reduced_definiteness_class = CertificationClass::NotCertified;
            claim.description =
                "Configured Schur/preconditioner evidence indicates solver-quality risk";
        } else if (summary.condition_risk_class ==
                   SchurQualityClass::RawConditionOnly) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.reduced_definiteness_class = CertificationClass::NotCertified;
            claim.description =
                "Only raw Schur conditioning evidence is available; solver-quality risk depends on preconditioned evidence";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.reduced_definiteness_class = CertificationClass::NotCertified;
            claim.description =
                "Schur complement exists but preconditioned quality evidence is insufficient";
        }

        claim.addEvidence("SchurQualityAnalyzer",
            "SchurComplementSummary id='" + summary.schur_id +
            "', available=" +
            std::string(summary.schur_available ? "true" : "false") +
            ", raw_condition_present=" +
            std::string(summary.condition_estimate_present ? "true" : "false") +
            ", raw_condition=" + std::to_string(summary.condition_estimate) +
            ", preconditioner='" + summary.preconditioner_type + "'" +
            ", probe_available=" +
            std::string(summary.preconditioned_probe_available ? "true" : "false") +
            ", contraction_present=" +
            std::string(summary.preconditioned_residual_contraction_present ? "true" : "false") +
            ", contraction=" +
            std::to_string(summary.preconditioned_residual_contraction) +
            ", block_tol=" + std::to_string(summary.block_solve_tolerance) +
            ", risk_class=" +
            std::to_string(static_cast<int>(summary.condition_risk_class)),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }
}

std::string ToleranceAdequacyAnalyzer::name() const
{
    return "ToleranceAdequacyAnalyzer";
}

void ToleranceAdequacyAnalyzer::run(const ProblemAnalysisContext& context,
                                    ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->numerical_error_budgets) {
        PropertyClaim claim;
        claim.kind = PropertyKind::SolverCompatibility;
        claim.variables = summary.variables.empty()
            ? blockVariables(summary.block)
            : summary.variables;
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag;
        claim.estimate_scope = summary.budget_id;
        claim.claim_origin = "ToleranceAdequacyAnalyzer";

        if (summary.adequacy_class ==
            ToleranceAdequacyClass::TooStrictForConditioning) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Fixed verification tolerance appears too strict for the observed conditioning-derived numerical floor";
        } else if (summary.adequacy_class ==
                   ToleranceAdequacyClass::Reasonable) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Fixed verification tolerance is consistent with the observed conditioning-derived numerical floor";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Numerical error budget is inconclusive";
        }

        claim.addEvidence("ToleranceAdequacyAnalyzer",
            "NumericalErrorBudgetSummary id='" + summary.budget_id +
            "', residual_present=" +
            std::string(summary.residual_norm_present ? "true" : "false") +
            ", residual=" + std::to_string(summary.residual_norm) +
            ", matrix_norm=" +
            std::to_string(summary.matrix_norm_estimate) +
            ", condition=" + std::to_string(summary.condition_estimate) +
            ", linear_tol=" + std::to_string(summary.linear_tolerance) +
            ", verification_tol=" +
            std::to_string(summary.verification_tolerance) +
            ", expected_abs_floor=" +
            std::to_string(summary.expected_absolute_floor) +
            ", expected_rel_floor=" +
            std::to_string(summary.expected_relative_floor) +
            ", recommended_tol=" +
            std::to_string(summary.recommended_verification_tolerance) +
            ", adequacy=" +
            std::to_string(static_cast<int>(summary.adequacy_class)) +
            ", reason='" + summary.reason + "'",
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }
}

std::string LockingRiskAnalyzer::name() const
{
    return "LockingRiskAnalyzer";
}

void LockingRiskAnalyzer::run(const ProblemAnalysisContext& context,
                              ProblemAnalysisReport& report) const
{
    bool saw_evidence = false;
    bool hard_violation = false;
    bool likely_risk = false;
    std::vector<VariableKey> variables;
    std::string evidence;

    if (const auto* constraints = context.constraintSummary()) {
        for (const auto& set : constraints->constrained_sets) {
            if (set.field != INVALID_FIELD_ID) {
                appendUnique(variables, VariableKey::field(set.field, set.component));
            }
            if (set.num_total_dofs > 0) {
                saw_evidence = true;
                if (set.constrained_fraction >= 0.99) {
                    hard_violation = true;
                } else if (set.constrained_fraction >= 0.85) {
                    likely_risk = true;
                }
            }
        }
        if (constraints->hasConflicts()) {
            saw_evidence = true;
            hard_violation = true;
        }
        evidence += "constraint_summary_sets=" +
                    std::to_string(constraints->constrained_sets.size()) + "; ";
    }

    const auto* summaries = context.analysisSummaries();
    if (summaries) {
        for (const auto& reduced : summaries->reduced_matrices) {
            if (reduced.free_free_matrix.condition_estimate &&
                *reduced.free_free_matrix.condition_estimate > Real{1.0e12}) {
                saw_evidence = true;
                likely_risk = true;
                auto block_vars = blockVariables(reduced.free_free_matrix.block);
                for (const auto& var : block_vars) appendUnique(variables, var);
            }
        }
        for (const auto& scale : summaries->parameter_scales) {
            if ((scale.layer_resolution_metric > Real{} &&
                 scale.layer_resolution_metric < Real{1}) ||
                scale.frequency_resolution_metric > Real{1}) {
                saw_evidence = true;
                likely_risk = true;
            }
        }
    }

    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition) {
            if (claim.status == PropertyStatus::Violated) {
                saw_evidence = true;
                hard_violation = true;
                for (const auto& var : claim.variables) appendUnique(variables, var);
            } else if (claim.inf_sup_class == InfSupClass::LikelyViolated) {
                saw_evidence = true;
                likely_risk = true;
                for (const auto& var : claim.variables) appendUnique(variables, var);
            } else if (claim.status == PropertyStatus::Exact ||
                       claim.status == PropertyStatus::Preserved) {
                saw_evidence = true;
            }
        }
        if (claim.kind == PropertyKind::SpaceCompatibility) {
            if (claim.space_compatibility_class == SpaceCompatibilityClass::Incompatible) {
                saw_evidence = true;
                hard_violation = true;
                for (const auto& var : claim.variables) appendUnique(variables, var);
            } else if (claim.status == PropertyStatus::Preserved) {
                saw_evidence = true;
            }
        }
    }

    if (!saw_evidence) return;

    PropertyClaim claim;
    claim.kind = PropertyKind::LockingRisk;
    claim.variables = variables;
    claim.claim_origin = "LockingRiskAnalyzer";
    if (hard_violation) {
        claim.status = PropertyStatus::Violated;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = CertificationClass::Violated;
        claim.description =
            "Constraint/space evidence indicates overstiffness or locking risk";
    } else if (likely_risk) {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            "Constraint/space evidence indicates possible overstiffness or locking risk";
    } else {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            "Available constraint/space evidence does not indicate locking risk, but locking-free behavior is not certified without uniform inf-sup, patch/hourglass, or parameter-robust conditioning evidence";
    }
    claim.addEvidence("LockingRiskAnalyzer",
        evidence.empty()
            ? "Inf-sup and space-compatibility claims were available"
            : evidence,
        claim.confidence);
    if (claim.status == PropertyStatus::Violated) {
        addWarning(report, "Locking/overstiffness risk detected");
    }
    report.claims.push_back(std::move(claim));
}

std::string SpectralSpuriousModeAnalyzer::name() const
{
    return "SpectralSpuriousModeAnalyzer";
}

void SpectralSpuriousModeAnalyzer::run(const ProblemAnalysisContext& context,
                                       ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->spectral_structures) {
        const bool theorem_scoped =
            theoremScoped(summary.spectral_convergence_theorem_id);
        const bool operator_or_gap_convergence_evidence =
            summary.operator_convergence_evidence ||
            summary.gap_convergence_evidence;
        const bool projection_bound_valid =
            summary.projection_bound_present &&
            finitePositive(summary.projection_bound);
        const bool spectral_complex_provenance_complete =
            theorem_scoped &&
            summary.compatible_complex_evidence &&
            summary.compatible_complex_spectral_theorem_evidence &&
            summary.bounded_projection_evidence_present &&
            projection_bound_valid &&
            summary.mesh_family_scope_present &&
            !summary.mesh_family_scope.empty() &&
            summary.shape_regular_mesh_evidence_present;
        const bool discrete_compactness_route =
            summary.discrete_compactness_evidence &&
            summary.discrete_compactness_provenance_present &&
            spectral_complex_provenance_complete;
        const bool compatible_complex_route =
            spectral_complex_provenance_complete;
        const bool convergence_evidence =
            operator_or_gap_convergence_evidence ||
            discrete_compactness_route ||
            compatible_complex_route;
        const bool diagnostic_scope_ok =
            !operator_or_gap_convergence_evidence ||
            theorem_scoped ||
            (summary.refinement_scope_metadata_present &&
             summary.refinement_sample_count >= 2u);
        const bool numeric_diagnostic_scope =
            operator_or_gap_convergence_evidence ||
            summary.refinement_scope_metadata_present;
        const bool spectral_tolerance_ok =
            !numeric_diagnostic_scope ||
            finitePositive(summary.spectral_tolerance);
        const bool rayleigh_bound_ok =
            summary.rayleigh_quotient_lower_bound == Real{} ||
            numeric::finite(summary.rayleigh_quotient_lower_bound);
        const bool nullspace_handling_ok =
            summary.rayleigh_quotient_lower_bound > Real{} ||
            summary.nullspace_handling != NullspaceHandlingClass::Unknown;
        const bool certified =
            summary.eigenproblem_declared &&
            summary.self_adjoint_evidence &&
            summary.compactness_evidence &&
            convergence_evidence &&
            diagnostic_scope_ok &&
            spectral_tolerance_ok &&
            rayleigh_bound_ok &&
            nullspace_handling_ok &&
            summary.spurious_mode_count == 0u;

        PropertyClaim claim;
        claim.kind = PropertyKind::SpectralCorrectness;
        claim.variables = blockVariables(summary.block);
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag;
        claim.nullspace_handling_class = summary.nullspace_handling;
        claim.claim_origin = "SpectralSpuriousModeAnalyzer";

        if (summary.spurious_mode_count > 0u) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Spectral summary reports spurious modes";
        } else if (certified) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Spectral summary has compact/self-adjoint evidence plus scoped operator/gap convergence, discrete-compactness, or compatible-complex spectral provenance";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Spectral correctness is unknown from available summary evidence";
        }

        claim.addEvidence("SpectralSpuriousModeAnalyzer",
            "SpectralStructureSummary block='" +
            summary.block.operator_tag +
            "', eigenproblem_declared=" +
            std::string(summary.eigenproblem_declared ? "true" : "false") +
            ", self_adjoint=" +
            std::string(summary.self_adjoint_evidence ? "true" : "false") +
            ", compactness=" +
            std::string(summary.compactness_evidence ? "true" : "false") +
            ", operator_convergence=" +
            std::string(summary.operator_convergence_evidence ? "true" : "false") +
            ", discrete_compactness=" +
            std::string(summary.discrete_compactness_evidence ? "true" : "false") +
            ", compatible_complex=" +
            std::string(summary.compatible_complex_evidence ? "true" : "false") +
            ", compatible_complex_theorem=" +
            std::string(summary.compatible_complex_spectral_theorem_evidence ? "true" : "false") +
            ", discrete_compactness_provenance=" +
            std::string(summary.discrete_compactness_provenance_present ? "true" : "false") +
            ", bounded_projection=" +
            std::string(summary.bounded_projection_evidence_present ? "true" : "false") +
            ", projection_bound_present=" +
            std::string(summary.projection_bound_present ? "true" : "false") +
            ", projection_bound=" +
            std::to_string(summary.projection_bound) +
            ", mesh_family_scope_present=" +
            std::string(summary.mesh_family_scope_present ? "true" : "false") +
            ", mesh_family_scope='" + summary.mesh_family_scope + "'" +
            ", shape_regular_mesh=" +
            std::string(summary.shape_regular_mesh_evidence_present ? "true" : "false") +
            ", gap_convergence=" +
            std::string(summary.gap_convergence_evidence ? "true" : "false") +
            ", theorem='" + summary.spectral_convergence_theorem_id + "'" +
            ", refinement_scope=" +
            std::string(summary.refinement_scope_metadata_present ? "true" : "false") +
            ", refinement_samples=" +
            std::to_string(summary.refinement_sample_count) +
            ", spurious_modes=" +
            std::to_string(summary.spurious_mode_count) +
            ", spectral_tolerance=" +
            std::to_string(summary.spectral_tolerance) +
            ", rayleigh_lower_bound=" +
            std::to_string(summary.rayleigh_quotient_lower_bound) +
            ", nullspace_handling=" +
            std::to_string(static_cast<int>(summary.nullspace_handling)),
            claim.confidence);
        if (claim.status == PropertyStatus::Violated) {
            addWarning(report,
                "Spurious-mode spectral violation in block '" +
                summary.block.operator_tag + "'");
        }
        report.claims.push_back(std::move(claim));
    }
}

std::string ErrorEstimatorAnalyzer::name() const
{
    return "ErrorEstimatorAnalyzer";
}

void ErrorEstimatorAnalyzer::run(const ProblemAnalysisContext& context,
                                 ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->error_estimators) {
        const bool has_residual_channel = summary.residual_metadata_present;
        const bool has_localization_channel =
            summary.jump_metadata_present || summary.flux_reconstruction_present;
        const bool missing =
            summary.missing_required_metadata_count > 0u ||
            !has_residual_channel;
        const bool reliability_constant_valid =
            summary.reliability_constant_metadata_present &&
            finitePositive(summary.reliability_constant);
        const bool efficiency_constant_valid =
            summary.efficiency_constant_metadata_present &&
            finitePositive(summary.efficiency_constant);
        const bool effectivity_bounds_valid =
            summary.effectivity_bounds_present &&
            finitePositive(summary.effectivity_lower_bound) &&
            finitePositive(summary.effectivity_upper_bound) &&
            summary.effectivity_lower_bound <= summary.effectivity_upper_bound &&
            summary.effectivity_sample_count >= 2u;
        const bool shape_regular_mesh_metadata_complete =
            summary.shape_regular_mesh_evidence_present &&
            summary.mesh_family_scope_present &&
            !summary.mesh_family_scope.empty() &&
            summary.shape_regular_constant_present &&
            finitePositive(summary.shape_regular_constant);
        const bool invalid_quantitative_metadata =
            (summary.reliability_constant_metadata_present &&
             !finitePositive(summary.reliability_constant)) ||
            (summary.efficiency_constant_metadata_present &&
             !finitePositive(summary.efficiency_constant)) ||
            (summary.effectivity_bounds_present &&
             !effectivity_bounds_valid) ||
            (summary.shape_regular_constant_present &&
             !finitePositive(summary.shape_regular_constant));
        const bool reliability_metadata_complete =
            has_residual_channel &&
            has_localization_channel &&
            summary.norm_metadata_present &&
            summary.estimator_norm_scope_metadata_present &&
            !summary.estimator_norm_id.empty() &&
            summary.pde_operator_class_metadata_present &&
            summary.boundary_residual_metadata_present &&
            summary.data_oscillation_metadata_present &&
            summary.coefficient_source_regularity_metadata_present &&
            shape_regular_mesh_metadata_complete &&
            reliability_constant_valid &&
            efficiency_constant_valid &&
            effectivity_bounds_valid &&
            summary.refinement_evidence_present &&
            theoremScoped(summary.estimator_theorem_id);
        const bool goal_metadata_ok =
            !summary.adjoint_weighting_available ||
            (summary.goal_functional_metadata_present &&
             summary.adjoint_residual_metadata_present);

        PropertyClaim claim;
        claim.kind = PropertyKind::ErrorEstimatorEligibility;
        claim.variables = blockVariables(summary.block);
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag.empty()
            ? summary.estimator_id
            : summary.block.operator_tag;
        claim.estimate_scope = summary.estimator_id;
        claim.claim_origin = "ErrorEstimatorAnalyzer";

        if (missing) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "A posteriori estimator metadata is missing required residual evidence";
        } else if (invalid_quantitative_metadata) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "A posteriori estimator metadata reports invalid reliability, efficiency, or effectivity constants";
        } else if (reliability_metadata_complete && goal_metadata_ok) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "A posteriori estimator metadata has residual, localization, norm, regularity, boundary, oscillation, scoped shape-regular mesh, effectivity, and refinement evidence";
        } else if (has_localization_channel) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "A posteriori estimator is eligible, but reliability/efficiency certification metadata is incomplete";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "A posteriori estimator eligibility lacks localization evidence";
        }

        claim.addEvidence("ErrorEstimatorAnalyzer",
            "ErrorEstimatorSummary id='" + summary.estimator_id +
            "', residual=" +
            std::string(summary.residual_metadata_present ? "true" : "false") +
            ", jump=" +
            std::string(summary.jump_metadata_present ? "true" : "false") +
            ", flux_reconstruction=" +
            std::string(summary.flux_reconstruction_present ? "true" : "false") +
            ", missing_required=" +
            std::to_string(summary.missing_required_metadata_count) +
            ", norm_id='" + summary.estimator_norm_id + "'" +
            ", theorem='" + summary.estimator_theorem_id + "'" +
            ", reliability_constant=" +
            std::to_string(summary.reliability_constant) +
            ", efficiency_constant=" +
            std::to_string(summary.efficiency_constant) +
            ", effectivity_lower=" +
            std::to_string(summary.effectivity_lower_bound) +
            ", effectivity_upper=" +
            std::to_string(summary.effectivity_upper_bound) +
            ", effectivity_samples=" +
            std::to_string(summary.effectivity_sample_count) +
            ", shape_regular_mesh=" +
            std::string(summary.shape_regular_mesh_evidence_present ? "true" : "false") +
            ", mesh_family_scope_present=" +
            std::string(summary.mesh_family_scope_present ? "true" : "false") +
            ", mesh_family_scope='" + summary.mesh_family_scope + "'" +
            ", shape_regular_constant_present=" +
            std::string(summary.shape_regular_constant_present ? "true" : "false") +
            ", shape_regular_constant=" +
            std::to_string(summary.shape_regular_constant) +
            ", reliability_metadata_complete=" +
            std::string(reliability_metadata_complete ? "true" : "false") +
            ", goal_metadata_ok=" +
            std::string(goal_metadata_ok ? "true" : "false"),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }
}

std::string QuadratureAdequacyAnalyzer::name() const
{
    return "QuadratureAdequacyAnalyzer";
}

void QuadratureAdequacyAnalyzer::run(const ProblemAnalysisContext& context,
                                     ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->quadrature_adequacy) {
        const bool aliasing_metric_declared =
            summary.aliasing_indicator_present ||
            summary.aliasing_tolerance_present;
        const bool aliasing_indicator_valid =
            !summary.aliasing_indicator_present ||
            numeric::finiteNonnegative(summary.aliasing_indicator);
        const bool aliasing_tolerance_valid =
            !summary.aliasing_tolerance_present ||
            numeric::finiteDeclaredTolerance(summary.aliasing_tolerance);
        const bool aliasing_numeric_valid =
            aliasing_indicator_valid && aliasing_tolerance_valid;
        const Real tol = summary.aliasing_tolerance_present
            ? effectiveTolerance(summary.aliasing_tolerance)
            : effectiveTolerance(Real{});
        const bool degree_known =
            summary.integrand_polynomial_degree >= 0 &&
            summary.quadrature_exact_degree >= 0;
        const bool degree_exact =
            degree_known &&
            summary.quadrature_exact_degree >= summary.integrand_polynomial_degree;
        const bool polynomial_exactness_scope_complete =
            summary.affine_mapping_evidence_present &&
            summary.polynomial_integrand_metadata_complete &&
            summary.coefficient_degree_metadata_present &&
            summary.mapped_integrand_metadata_present &&
            summary.basis_degree_metadata_present &&
            summary.geometry_jacobian_degree_metadata_present &&
            summary.tensor_contraction_metadata_present &&
            summary.component_coverage_metadata_present &&
            theoremScoped(summary.quadrature_theorem_id) &&
            (!summary.curved_or_nonlinear_mapping ||
             summary.overintegration_metadata_present ||
             summary.nonlinear_aliasing_control_present);
        const bool aliasing_required =
            summary.curved_or_nonlinear_mapping ||
            summary.reduced_integration_declared ||
            summary.nonlinear_aliasing_control_required;
        const bool aliasing_invalid =
            aliasing_metric_declared && !aliasing_numeric_valid;
        const bool aliasing_failed =
            aliasing_metric_declared &&
            aliasing_numeric_valid &&
            summary.aliasing_indicator_present &&
            summary.aliasing_tolerance_present &&
            summary.aliasing_indicator > tol;
        const bool aliasing_missing =
            aliasing_required && !aliasing_metric_declared;
        const bool aliasing_incomplete =
            aliasing_required &&
            aliasing_metric_declared &&
            (!summary.aliasing_indicator_present ||
             !summary.aliasing_tolerance_present);
        const bool aliasing_violation =
            aliasing_invalid ||
            aliasing_failed ||
            summary.underintegrated_entry_count > 0u ||
            summary.zero_energy_mode_count > 0u;
        const bool reduced_with_control =
            summary.reduced_integration_declared &&
            summary.hourglass_control_present &&
            summary.zero_energy_mode_count == 0u;

        PropertyClaim claim;
        claim.kind = PropertyKind::QuadratureAdequacy;
        claim.variables = blockVariables(summary.block);
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag;
        claim.claim_origin = "QuadratureAdequacyAnalyzer";

        if (aliasing_violation ||
            (degree_known && !degree_exact && !reduced_with_control)) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Quadrature summary violates exactness or aliasing requirements";
        } else if (aliasing_missing || aliasing_incomplete) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Quadrature exactness or aliasing control requires scoped aliasing metric evidence";
        } else if ((degree_exact && polynomial_exactness_scope_complete) ||
                   (summary.nonlinear_aliasing_control_present &&
                    !aliasing_violation) ||
                   reduced_with_control) {
            claim.status = reduced_with_control && !degree_exact
                ? PropertyStatus::Likely
                : PropertyStatus::Preserved;
            claim.confidence = degree_exact && polynomial_exactness_scope_complete
                ? AnalysisConfidence::High
                : AnalysisConfidence::Medium;
            claim.certification_class = degree_exact && polynomial_exactness_scope_complete
                ? CertificationClass::Certified
                : CertificationClass::NotCertified;
            claim.description =
                "Quadrature summary has adequate exactness or aliasing-control evidence";
        } else if (degree_exact) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Quadrature degree is sufficient, but affine mapping, polynomial-integrand, or coefficient/mapping metadata is incomplete";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Quadrature adequacy is unknown from available summary evidence";
        }

        claim.addEvidence("QuadratureAdequacyAnalyzer",
            "QuadratureAdequacySummary block='" +
            summary.block.operator_tag +
            "', integrand_degree=" +
            std::to_string(summary.integrand_polynomial_degree) +
            ", exact_degree=" +
            std::to_string(summary.quadrature_exact_degree) +
            ", affine_mapping=" +
            std::string(summary.affine_mapping_evidence_present ? "true" : "false") +
            ", polynomial_metadata=" +
            std::string(summary.polynomial_integrand_metadata_complete ? "true" : "false") +
            ", coefficient_degree_metadata=" +
            std::string(summary.coefficient_degree_metadata_present ? "true" : "false") +
            ", mapped_integrand_metadata=" +
            std::string(summary.mapped_integrand_metadata_present ? "true" : "false") +
            ", basis_degree_metadata=" +
            std::string(summary.basis_degree_metadata_present ? "true" : "false") +
            ", geometry_jacobian_metadata=" +
            std::string(summary.geometry_jacobian_degree_metadata_present ? "true" : "false") +
            ", tensor_contraction_metadata=" +
            std::string(summary.tensor_contraction_metadata_present ? "true" : "false") +
            ", component_coverage_metadata=" +
            std::string(summary.component_coverage_metadata_present ? "true" : "false") +
            ", theorem='" + summary.quadrature_theorem_id + "'" +
            ", curved_or_nonlinear_mapping=" +
            std::string(summary.curved_or_nonlinear_mapping ? "true" : "false") +
            ", underintegrated_entries=" +
            std::to_string(summary.underintegrated_entry_count) +
            ", zero_energy_modes=" +
            std::to_string(summary.zero_energy_mode_count) +
            ", aliasing_required=" +
            std::string(aliasing_required ? "true" : "false") +
            ", aliasing_indicator_present=" +
            std::string(summary.aliasing_indicator_present ? "true" : "false") +
            ", aliasing_tolerance_present=" +
            std::string(summary.aliasing_tolerance_present ? "true" : "false") +
            ", aliasing_indicator=" +
            std::to_string(summary.aliasing_indicator),
            claim.confidence);
        if (claim.status == PropertyStatus::Violated) {
            addWarning(report,
                "Quadrature/aliasing violation in block '" +
                summary.block.operator_tag + "'");
        }
        report.claims.push_back(std::move(claim));
    }
}

std::string CoupledSystemStabilityAnalyzer::name() const
{
    return "CoupledSystemStabilityAnalyzer";
}

std::string MinimumResidualStabilityAnalyzer::name() const
{
    return "MinimumResidualStabilityAnalyzer";
}

std::string PreservationStructureAnalyzer::name() const
{
    return "PreservationStructureAnalyzer";
}

void MinimumResidualStabilityAnalyzer::run(
    const ProblemAnalysisContext& context,
    ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->minimum_residual_stability) {
        const bool known_method =
            summary.method_class != MinimumResidualMethodClass::Unknown;
        const bool distinct_spaces_ok =
            summary.method_class == MinimumResidualMethodClass::LeastSquares ||
            summary.distinct_test_trial_spaces;
        const bool residual_control_valid =
            summary.residual_control_constant_present &&
            finitePositive(summary.residual_control_constant);
        const bool local_conditioning_valid =
            summary.local_trial_to_test_conditioning_present &&
            finitePositive(summary.local_trial_to_test_condition_estimate);
        const bool normal_conditioning_valid =
            summary.normal_equation_conditioning_present &&
            finitePositive(summary.normal_equation_condition_estimate);
        const bool accepted_local_condition_bound_valid =
            summary.accepted_local_trial_to_test_condition_bound_present &&
            finitePositive(summary.accepted_local_trial_to_test_condition_bound) &&
            local_conditioning_valid &&
            summary.local_trial_to_test_condition_estimate <=
                summary.accepted_local_trial_to_test_condition_bound + Real{1.0e-14};
        const bool accepted_normal_condition_bound_valid =
            summary.accepted_normal_equation_condition_bound_present &&
            finitePositive(summary.accepted_normal_equation_condition_bound) &&
            normal_conditioning_valid &&
            summary.normal_equation_condition_estimate <=
                summary.accepted_normal_equation_condition_bound + Real{1.0e-14};
        const bool fortin_norm_valid =
            summary.fortin_operator_norm_bound_present &&
            finitePositive(summary.fortin_operator_norm_bound);
        const bool accepted_fortin_bound_valid =
            !summary.accepted_fortin_operator_norm_bound_present ||
            (finitePositive(summary.accepted_fortin_operator_norm_bound) &&
             fortin_norm_valid &&
             summary.fortin_operator_norm_bound <=
                 summary.accepted_fortin_operator_norm_bound + Real{1.0e-14});
        const bool fortin_route_complete =
            summary.fortin_operator_evidence_present &&
            fortin_norm_valid &&
            accepted_fortin_bound_valid;
        const bool optimal_test_route_complete =
            summary.optimal_test_metadata_present;
        const bool core_metadata_present =
            known_method &&
            distinct_spaces_ok &&
            summary.trial_space_metadata_present &&
            summary.test_space_metadata_present &&
            summary.residual_norm_metadata_present &&
            summary.test_norm_metadata_present &&
            summary.method_scope_metadata_present &&
            !summary.residual_norm_id.empty() &&
            !summary.test_norm_id.empty() &&
            theoremScoped(summary.minimum_residual_theorem_id);
        const bool dpg_or_pg_stability_metadata_present =
            summary.riesz_map_metadata_present &&
            (fortin_route_complete ||
             optimal_test_route_complete) &&
            summary.enrichment_sufficiency_evidence_present &&
            residual_control_valid;
        const bool conditioning_metadata_present =
            local_conditioning_valid &&
            normal_conditioning_valid &&
            accepted_local_condition_bound_valid &&
            accepted_normal_condition_bound_valid &&
            summary.condition_bound_scope_metadata_present &&
            !summary.condition_bound_scope.empty();
        const bool certified =
            core_metadata_present &&
            dpg_or_pg_stability_metadata_present &&
            conditioning_metadata_present &&
            summary.missing_required_metadata_count == 0u &&
            summary.violation_count == 0u;
        const bool invalid_numeric_metadata =
            (summary.residual_control_constant_present &&
             !finitePositive(summary.residual_control_constant)) ||
            (summary.local_trial_to_test_conditioning_present &&
             !finitePositive(summary.local_trial_to_test_condition_estimate)) ||
            (summary.normal_equation_conditioning_present &&
             !finitePositive(summary.normal_equation_condition_estimate)) ||
            (summary.accepted_local_trial_to_test_condition_bound_present &&
             !finitePositive(summary.accepted_local_trial_to_test_condition_bound)) ||
            (summary.accepted_normal_equation_condition_bound_present &&
             !finitePositive(summary.accepted_normal_equation_condition_bound)) ||
            (summary.accepted_local_trial_to_test_condition_bound_present &&
             local_conditioning_valid &&
             summary.local_trial_to_test_condition_estimate >
                 summary.accepted_local_trial_to_test_condition_bound + Real{1.0e-14}) ||
            (summary.accepted_normal_equation_condition_bound_present &&
             normal_conditioning_valid &&
             summary.normal_equation_condition_estimate >
                 summary.accepted_normal_equation_condition_bound + Real{1.0e-14}) ||
            (summary.fortin_operator_norm_bound_present &&
             !finitePositive(summary.fortin_operator_norm_bound)) ||
            (summary.accepted_fortin_operator_norm_bound_present &&
             !finitePositive(summary.accepted_fortin_operator_norm_bound)) ||
            (summary.accepted_fortin_operator_norm_bound_present &&
             fortin_norm_valid &&
             summary.fortin_operator_norm_bound >
                 summary.accepted_fortin_operator_norm_bound + Real{1.0e-14}) ||
            (summary.discrete_inf_sup_lower_bound_present &&
             !finitePositive(summary.discrete_inf_sup_lower_bound));
        const bool violated =
            summary.violation_count > 0u ||
            invalid_numeric_metadata ||
            (known_method && !distinct_spaces_ok);

        PropertyClaim claim;
        claim.kind = PropertyKind::MinimumResidualStability;
        claim.variables = !summary.variables.empty()
            ? summary.variables
            : blockVariables(summary.block);
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag.empty()
            ? summary.method_id
            : summary.block.operator_tag;
        claim.estimate_scope = summary.method_id;
        claim.claim_origin = "MinimumResidualStabilityAnalyzer";

        const auto gate = certificationGate(
            violated,
            certified,
            core_metadata_present);
        claim.status = gate.status;
        claim.certification_class = gate.certification;
        claim.confidence = gate.confidence;

        if (violated) {
            claim.description =
                "Minimum-residual/Petrov-Galerkin summary reports violated residual, Fortin, Riesz, enrichment, distinct-space, or accepted conditioning checks";
        } else if (certified) {
            claim.description =
                "Minimum-residual/Petrov-Galerkin stability is certified by scoped trial/test, residual norm, test norm, Riesz map, quantified Fortin or optimal-test evidence, enrichment, residual-control, theorem, and accepted conditioning-bound evidence";
        } else if (core_metadata_present) {
            claim.description =
                "Minimum-residual/Petrov-Galerkin method is structurally eligible but lacks complete Fortin/optimal-test, enrichment, residual-control, or accepted conditioning-bound evidence";
        } else {
            claim.description =
                "Minimum-residual/Petrov-Galerkin stability is unknown because trial/test space or residual/test norm metadata is incomplete";
        }

        claim.addEvidence("MinimumResidualStabilityAnalyzer",
            "MinimumResidualStabilitySummary method='" + summary.method_id +
            "', method_class=" +
            std::to_string(static_cast<int>(summary.method_class)) +
            ", theorem='" + summary.minimum_residual_theorem_id + "'" +
            ", residual_norm='" + summary.residual_norm_id + "'" +
            ", test_norm='" + summary.test_norm_id + "'" +
            ", trial_space=" +
            std::string(summary.trial_space_metadata_present ? "true" : "false") +
            ", test_space=" +
            std::string(summary.test_space_metadata_present ? "true" : "false") +
            ", distinct_spaces=" +
            std::string(summary.distinct_test_trial_spaces ? "true" : "false") +
            ", method_scope=" +
            std::string(summary.method_scope_metadata_present ? "true" : "false") +
            ", residual_norm=" +
            std::string(summary.residual_norm_metadata_present ? "true" : "false") +
            ", test_norm=" +
            std::string(summary.test_norm_metadata_present ? "true" : "false") +
            ", riesz_map=" +
            std::string(summary.riesz_map_metadata_present ? "true" : "false") +
            ", fortin=" +
            std::string(summary.fortin_operator_evidence_present ? "true" : "false") +
            ", fortin_norm_present=" +
            std::string(summary.fortin_operator_norm_bound_present ? "true" : "false") +
            ", fortin_norm=" +
            std::to_string(summary.fortin_operator_norm_bound) +
            ", accepted_fortin_norm_present=" +
            std::string(summary.accepted_fortin_operator_norm_bound_present ? "true" : "false") +
            ", accepted_fortin_norm=" +
            std::to_string(summary.accepted_fortin_operator_norm_bound) +
            ", discrete_inf_sup_present=" +
            std::string(summary.discrete_inf_sup_lower_bound_present ? "true" : "false") +
            ", discrete_inf_sup=" +
            std::to_string(summary.discrete_inf_sup_lower_bound) +
            ", optimal_test=" +
            std::string(summary.optimal_test_metadata_present ? "true" : "false") +
            ", enrichment=" +
            std::string(summary.enrichment_sufficiency_evidence_present ? "true" : "false") +
            ", residual_control=" +
            std::string(summary.residual_control_constant_present ? "true" : "false") +
            ", residual_control_value=" +
            std::to_string(summary.residual_control_constant) +
            ", conditioning=" +
            std::string(conditioning_metadata_present ? "true" : "false") +
            ", local_condition=" +
            std::to_string(summary.local_trial_to_test_condition_estimate) +
            ", accepted_local_condition_present=" +
            std::string(summary.accepted_local_trial_to_test_condition_bound_present ? "true" : "false") +
            ", accepted_local_condition=" +
            std::to_string(summary.accepted_local_trial_to_test_condition_bound) +
            ", normal_condition=" +
            std::to_string(summary.normal_equation_condition_estimate) +
            ", accepted_normal_condition_present=" +
            std::string(summary.accepted_normal_equation_condition_bound_present ? "true" : "false") +
            ", accepted_normal_condition=" +
            std::to_string(summary.accepted_normal_equation_condition_bound) +
            ", condition_scope='" + summary.condition_bound_scope + "'" +
            ", violations=" +
            std::to_string(summary.violation_count),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }
}

void PreservationStructureAnalyzer::run(const ProblemAnalysisContext& context,
                                        ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->invariant_domains) {
        const auto applicability_it = std::find_if(
            summaries->applicability.begin(),
            summaries->applicability.end(),
            [&](const ApplicabilitySummary& applicability) {
                return applicability.theorem_family ==
                           TheoremFamily::InvariantDomain &&
                       applicability.applicability ==
                           ApplicabilityClass::NotApplicable &&
                       (localVariableSetsIntersect(applicability.variables,
                                                   summary.variables) ||
                        summary.variables.empty());
            });
        if (applicability_it != summaries->applicability.end()) {
            PropertyClaim claim;
            claim.kind = PropertyKind::InvariantDomainPreservation;
            claim.variables = summary.variables;
            claim.invariant_set_id = summary.invariant_set_id;
            claim.claim_origin = "PreservationStructureAnalyzer";
            claim.applicability_class = ApplicabilityClass::NotApplicable;
            claim.certification_class = CertificationClass::Unknown;
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.description =
                "Invariant-domain theorem is not structurally applicable to this operator scope";
            claim.addEvidence("PreservationStructureAnalyzer",
                "ApplicabilitySummary reason='" +
                applicability_it->reason + "'",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
            continue;
        }

        const bool bounds_declared =
            summary.lower_bound_active || summary.upper_bound_active;
        const bool lower_bound_valid =
            !summary.lower_bound_active ||
            numeric::finite(summary.lower_bound);
        const bool upper_bound_valid =
            !summary.upper_bound_active ||
            numeric::finite(summary.upper_bound);
        const bool bounds_order_valid =
            !(summary.lower_bound_active && summary.upper_bound_active) ||
            summary.lower_bound <= summary.upper_bound;
        const bool bounds_valid =
            bounds_declared &&
            lower_bound_valid &&
            upper_bound_valid &&
            bounds_order_valid;
        const bool cfl_bound_valid =
            summary.cfl_estimate_present &&
            summary.accepted_cfl_bound_present &&
            finiteNonnegative(summary.cfl_estimate) &&
            finitePositive(summary.accepted_cfl_bound) &&
            summary.cfl_estimate <= summary.accepted_cfl_bound + Real{1.0e-14};
        const bool cfl_scope_complete =
            !summary.time_step_scope.empty() &&
            !summary.mesh_size_scope.empty() &&
            summary.wave_speed_bound_present &&
            finiteNonnegative(summary.wave_speed_bound);
        const bool active_bounds_invalid =
            bounds_declared &&
            (!lower_bound_valid ||
             !upper_bound_valid ||
             !bounds_order_valid);
        const bool cfl_numeric_invalid =
            (summary.cfl_estimate_present &&
             !finiteNonnegative(summary.cfl_estimate)) ||
            (summary.accepted_cfl_bound_present &&
             !finitePositive(summary.accepted_cfl_bound)) ||
            (summary.wave_speed_bound_present &&
             !finiteNonnegative(summary.wave_speed_bound)) ||
            (summary.cfl_estimate_present &&
             summary.accepted_cfl_bound_present &&
             finiteNonnegative(summary.cfl_estimate) &&
             finitePositive(summary.accepted_cfl_bound) &&
             summary.cfl_estimate > summary.accepted_cfl_bound + Real{1.0e-14});
        const bool metadata_complete =
            bounds_valid &&
            summary.limiter_evidence_present &&
            summary.cfl_condition_satisfied &&
            cfl_bound_valid &&
            cfl_scope_complete &&
            summary.ssp_time_discretization_evidence_present &&
            summary.source_admissibility_evidence_present &&
            summary.low_order_invariant_domain_evidence_present &&
            summary.convex_limiting_evidence_present &&
            summary.spatial_monotonicity_evidence_present &&
            summary.mass_positivity_evidence_present &&
            !summary.invariant_domain_theorem_id.empty();
        PropertyClaim claim;
        claim.kind = PropertyKind::InvariantDomainPreservation;
        claim.variables = summary.variables;
        claim.invariant_set_id = summary.invariant_set_id;
        claim.invariant_domain_metadata_present = summary.limiter_evidence_present;
        claim.claim_origin = "PreservationStructureAnalyzer";
        claim.applicability_class = ApplicabilityClass::Applicable;
        if (active_bounds_invalid) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Invariant-domain summary has non-finite or unordered active bounds";
        } else if (cfl_numeric_invalid) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Invariant-domain CFL, accepted bound, or wave-speed metadata is invalid or exceeds the accepted bound";
        } else if (summary.post_step_violation_count > 0u) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Invariant-domain summary reports post-step bound violations";
        } else if (metadata_complete) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Invariant-domain summary preserves the declared bounds";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Invariant-domain preservation lacks limiter, bound, quantitative CFL/SSP, wave-speed, source-admissibility, monotone low-order, mass-positivity, or theorem metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "InvariantDomainSummary id='" + summary.invariant_set_id +
            "', lower_active=" +
            std::string(summary.lower_bound_active ? "true" : "false") +
            ", lower_bound=" +
            std::to_string(summary.lower_bound) +
            ", upper_active=" +
            std::string(summary.upper_bound_active ? "true" : "false") +
            ", upper_bound=" +
            std::to_string(summary.upper_bound) +
            ", excluded_active=" +
            std::string(summary.excluded_value_active ? "true" : "false") +
            ", excluded_value=" +
            std::to_string(summary.excluded_value) +
            ", bounds_valid=" +
            std::string(bounds_valid ? "true" : "false") +
            ", limiter=" +
            std::string(summary.limiter_evidence_present ? "true" : "false") +
            ", cfl_condition=" +
            std::string(summary.cfl_condition_satisfied ? "true" : "false") +
            ", cfl_estimate_present=" +
            std::string(summary.cfl_estimate_present ? "true" : "false") +
            ", cfl_estimate=" +
            std::to_string(summary.cfl_estimate) +
            ", accepted_cfl_bound_present=" +
            std::string(summary.accepted_cfl_bound_present ? "true" : "false") +
            ", accepted_cfl_bound=" +
            std::to_string(summary.accepted_cfl_bound) +
            ", wave_speed_bound_present=" +
            std::string(summary.wave_speed_bound_present ? "true" : "false") +
            ", wave_speed_bound=" +
            std::to_string(summary.wave_speed_bound) +
            ", time_step_scope='" + summary.time_step_scope + "'" +
            ", mesh_size_scope='" + summary.mesh_size_scope + "'" +
            ", ssp_time_discretization=" +
            std::string(summary.ssp_time_discretization_evidence_present ? "true" : "false") +
            ", source_admissibility=" +
            std::string(summary.source_admissibility_evidence_present ? "true" : "false") +
            ", low_order_invariant_domain=" +
            std::string(summary.low_order_invariant_domain_evidence_present ? "true" : "false") +
            ", convex_limiting=" +
            std::string(summary.convex_limiting_evidence_present ? "true" : "false") +
            ", spatial_monotonicity=" +
            std::string(summary.spatial_monotonicity_evidence_present ? "true" : "false") +
            ", mass_positivity=" +
            std::string(summary.mass_positivity_evidence_present ? "true" : "false") +
            ", theorem='" + summary.invariant_domain_theorem_id + "'" +
            ", post_step_violations=" +
            std::to_string(summary.post_step_violation_count),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->equilibrium_preservation) {
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(summary.residual_tolerance);
        const bool residual_finite =
            numeric::finite(summary.flux_source_residual);
        const bool numeric_evidence_valid =
            tolerance_declared && residual_finite;
        const Real tol = effectiveTolerance(summary.residual_tolerance);
        const bool local_balance_metadata_present =
            summary.source_quadrature_metadata_present &&
            summary.reconstruction_metadata_present &&
            summary.boundary_compatibility_metadata_present;
        const bool equilibrium_scope_complete =
            !summary.equilibrium_id.empty() &&
            !summary.equilibrium_family_id.empty() &&
            theoremScoped(summary.equilibrium_preservation_theorem_id) &&
            summary.equilibrium_scope_metadata_present &&
            summary.source_model_scope_metadata_present &&
            summary.reconstruction_scope_metadata_present;
        const bool metadata_present =
            local_balance_metadata_present &&
            equilibrium_scope_complete;
        const bool residual_ok =
            numeric_evidence_valid &&
            std::abs(summary.flux_source_residual) <= tol;
        const bool residual_violation =
            numeric_evidence_valid &&
            std::abs(summary.flux_source_residual) > tol;

        PropertyClaim claim;
        claim.kind = PropertyKind::EquilibriumPreservation;
        claim.equilibrium_id = summary.equilibrium_id;
        claim.well_balanced_metadata_present = metadata_present;
        claim.flux_balance_residual = summary.flux_source_residual;
        claim.claim_origin = "PreservationStructureAnalyzer";
        if (residual_violation) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Equilibrium-preservation flux/source residual exceeds tolerance";
        } else if (residual_ok && metadata_present) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Equilibrium-preservation summary has zero flux/source residual within tolerance and scoped equilibrium-family evidence";
        } else if (residual_ok && local_balance_metadata_present) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Equilibrium residual is small with local balance metadata, but theorem, equilibrium-family, source-model, or reconstruction scope is incomplete";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                numeric_evidence_valid
                    ? "Equilibrium residual is small but metadata is incomplete"
                    : "Equilibrium-preservation summary lacks finite declared tolerance or finite residual evidence";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "EquilibriumPreservationSummary id='" +
            summary.equilibrium_id +
            "', family='" + summary.equilibrium_family_id +
            "', theorem='" + summary.equilibrium_preservation_theorem_id +
            "', flux_source_residual=" +
            std::to_string(summary.flux_source_residual) +
            ", tolerance=" + std::to_string(tol) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", finite_residual=" +
            std::string(residual_finite ? "true" : "false") +
            ", source_quadrature=" +
            std::string(summary.source_quadrature_metadata_present ? "true" : "false") +
            ", reconstruction=" +
            std::string(summary.reconstruction_metadata_present ? "true" : "false") +
            ", boundary_compatibility=" +
            std::string(summary.boundary_compatibility_metadata_present ? "true" : "false") +
            ", equilibrium_scope=" +
            std::string(summary.equilibrium_scope_metadata_present ? "true" : "false") +
            ", source_model_scope=" +
            std::string(summary.source_model_scope_metadata_present ? "true" : "false") +
            ", reconstruction_scope=" +
            std::string(summary.reconstruction_scope_metadata_present ? "true" : "false") +
            ", full_metadata=" +
            std::string(metadata_present ? "true" : "false"),
            claim.confidence);
        if (!residual_violation && !numeric_evidence_valid) {
            addWarning(report,
                "Equilibrium-preservation summary '" +
                summary.equilibrium_id +
                "' has invalid residual tolerance or residual evidence");
        }
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->moving_domain) {
        const Real tol = effectiveTolerance(summary.geometric_conservation_tolerance);
        const bool tolerance_declared =
            summary.geometric_conservation_tolerance_declared &&
            numeric::finiteDeclaredTolerance(
                summary.geometric_conservation_tolerance);
        const bool jacobian_positive =
            numeric::finitePositiveOrdered(summary.min_geometric_jacobian,
                                           summary.max_geometric_jacobian);
        const bool residual_ok =
            tolerance_declared &&
            numeric::finite(summary.geometric_conservation_residual) &&
            std::abs(summary.geometric_conservation_residual) <= tol;
        const bool free_stream_residual_ok =
            summary.free_stream_preservation_residual_present &&
            numeric::finite(summary.free_stream_preservation_residual) &&
            std::abs(summary.free_stream_preservation_residual) <= tol;
        const bool metadata_present =
            summary.mesh_velocity_metadata_present &&
            summary.time_integration_metadata_present &&
            summary.remap_metadata_present &&
            summary.metric_identity_evidence_present &&
            free_stream_residual_ok &&
            theoremScoped(summary.gcl_theorem_id) &&
            !summary.constant_state_scope.empty() &&
            !summary.mesh_update_time_scheme.empty();

        PropertyClaim claim;
        claim.kind = PropertyKind::GeometricConservation;
        claim.flux_balance_residual = summary.geometric_conservation_residual;
        claim.claim_origin = "PreservationStructureAnalyzer";
        claim.estimate_scope =
            "mesh_revision=" + std::to_string(summary.mesh_revision);
        if (!jacobian_positive || !residual_ok ||
            (summary.free_stream_preservation_residual_present &&
             !free_stream_residual_ok)) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Moving-domain summary violates positive mapping or geometric conservation";
        } else if (metadata_present) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Moving-domain summary preserves positive mapping, discrete geometric conservation, and free-stream constant states";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Moving-domain geometric conservation lacks declared tolerance, metric identity, theorem, free-stream, mesh-update, or remap metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "MovingDomainSummary min_jacobian=" +
            std::to_string(summary.min_geometric_jacobian) +
            ", max_jacobian=" +
            std::to_string(summary.max_geometric_jacobian) +
            ", gcl_residual=" +
            std::to_string(summary.geometric_conservation_residual) +
            ", tolerance=" + std::to_string(tol) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", theorem='" + summary.gcl_theorem_id + "'" +
            ", metric_identity=" +
            std::string(summary.metric_identity_evidence_present ? "true" : "false") +
            ", free_stream_residual_present=" +
            std::string(summary.free_stream_preservation_residual_present ? "true" : "false") +
            ", free_stream_residual=" +
            std::to_string(summary.free_stream_preservation_residual) +
            ", constant_state_scope='" + summary.constant_state_scope + "'" +
            ", mesh_update_time_scheme='" + summary.mesh_update_time_scheme + "'" +
            ", mesh_velocity_metadata=" +
            std::string(summary.mesh_velocity_metadata_present ? "true" : "false") +
            ", time_integration_metadata=" +
            std::string(summary.time_integration_metadata_present ? "true" : "false") +
            ", remap_metadata=" +
            std::string(summary.remap_metadata_present ? "true" : "false"),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->transfer_operators) {
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(summary.residual_tolerance);
        const bool conservation_residual_finite =
            numeric::finite(summary.conservation_residual);
        const bool constant_residual_finite =
            numeric::finite(summary.constant_preservation_residual);
        const bool numeric_evidence_valid =
            tolerance_declared &&
            conservation_residual_finite &&
            constant_residual_finite;
        const Real tol = effectiveTolerance(summary.residual_tolerance);
        const bool residual_ok =
            numeric_evidence_valid &&
            std::abs(summary.conservation_residual) <= tol &&
            std::abs(summary.constant_preservation_residual) <= tol;
        const bool residual_violation =
            numeric_evidence_valid &&
            (std::abs(summary.conservation_residual) > tol ||
             std::abs(summary.constant_preservation_residual) > tol);
        const bool rank_defect_valid =
            summary.rank_defect_present &&
            numeric::finiteNonnegative(summary.rank_defect) &&
            summary.rank_defect <= tol;
        const bool projection_norm_valid =
            summary.projection_operator_norm_present &&
            finitePositive(summary.projection_operator_norm);
        const bool accepted_projection_norm_valid =
            summary.accepted_projection_operator_norm_present &&
            finitePositive(summary.accepted_projection_operator_norm) &&
            projection_norm_valid &&
            summary.projection_operator_norm <=
                summary.accepted_projection_operator_norm + Real{1.0e-14};
        const bool mortar_inf_sup_bound_valid =
            summary.mortar_inf_sup_lower_bound_present &&
            finitePositive(summary.mortar_inf_sup_lower_bound);
        const bool mass_condition_bound_valid =
            summary.interface_mass_condition_number_present &&
            summary.accepted_interface_mass_condition_bound_present &&
            finitePositive(summary.interface_mass_condition_number) &&
            finitePositive(summary.accepted_interface_mass_condition_bound) &&
            summary.interface_mass_condition_number <=
                summary.accepted_interface_mass_condition_bound + Real{1.0e-14};
        const bool invalid_quantitative_transfer_evidence =
            (summary.rank_defect_present &&
             (!numeric::finiteNonnegative(summary.rank_defect) ||
              summary.rank_defect > tol)) ||
            (summary.projection_operator_norm_present &&
             !finitePositive(summary.projection_operator_norm)) ||
            (summary.accepted_projection_operator_norm_present &&
             !finitePositive(summary.accepted_projection_operator_norm)) ||
            (summary.accepted_projection_operator_norm_present &&
             projection_norm_valid &&
             summary.projection_operator_norm >
                 summary.accepted_projection_operator_norm + Real{1.0e-14}) ||
            (summary.mortar_inf_sup_lower_bound_present &&
             !finitePositive(summary.mortar_inf_sup_lower_bound)) ||
            (summary.interface_mass_condition_number_present &&
             !finitePositive(summary.interface_mass_condition_number)) ||
            (summary.accepted_interface_mass_condition_bound_present &&
             !finitePositive(summary.accepted_interface_mass_condition_bound)) ||
            (summary.interface_mass_condition_number_present &&
             summary.accepted_interface_mass_condition_bound_present &&
             finitePositive(summary.interface_mass_condition_number) &&
             finitePositive(summary.accepted_interface_mass_condition_bound) &&
             summary.interface_mass_condition_number >
                 summary.accepted_interface_mass_condition_bound + Real{1.0e-14});
        const bool metadata_complete =
            summary.rank_metadata_present &&
            rank_defect_valid &&
            summary.interface_scope_metadata_present &&
            summary.projection_consistency_metadata_present &&
            accepted_projection_norm_valid &&
            summary.mortar_inf_sup_or_dual_consistency_metadata_present &&
            mortar_inf_sup_bound_valid &&
            summary.interface_mass_conditioning_metadata_present &&
            mass_condition_bound_valid &&
            summary.action_reaction_flux_metadata_present &&
            !summary.interface_pair_id.empty() &&
            !summary.projection_space_id.empty() &&
            !summary.transfer_theorem_id.empty() &&
            !summary.interface_quadrature_scope.empty();

        PropertyClaim claim;
        claim.kind = PropertyKind::TransferOperatorCompatibility;
        claim.flux_balance_residual = numeric_evidence_valid
            ? std::max(std::abs(summary.conservation_residual),
                       std::abs(summary.constant_preservation_residual))
            : Real{};
        claim.estimate_scope = summary.interface_pair_id;
        claim.tested_block_id = summary.projection_space_id;
        claim.claim_origin = "PreservationStructureAnalyzer";
        if (residual_violation || invalid_quantitative_transfer_evidence) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                residual_violation
                    ? "Transfer summary violates conservation or constant preservation residual tolerance"
                    : "Transfer summary has invalid rank, mortar inf-sup, projection norm, or interface mass conditioning evidence";
        } else if (residual_ok && metadata_complete) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Transfer summary preserves constants and conservation with complete projection, mortar/dual-consistency, conditioning, action-reaction, scope, and rank metadata";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                numeric_evidence_valid
                    ? "Transfer summary residuals pass but projection, mortar/dual-consistency, conditioning, action-reaction, scope, or rank metadata is incomplete"
                    : "Transfer summary lacks finite declared tolerance or finite conservation/constant residual evidence";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "TransferOperatorSummary interface='" +
            summary.interface_pair_id +
            "', projection='" + summary.projection_space_id +
            "', conservation_residual=" +
            std::to_string(summary.conservation_residual) +
            ", constant_residual=" +
            std::to_string(summary.constant_preservation_residual) +
            ", tolerance=" + std::to_string(tol) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", conservation_residual_finite=" +
            std::string(conservation_residual_finite ? "true" : "false") +
            ", constant_residual_finite=" +
            std::string(constant_residual_finite ? "true" : "false") +
            ", theorem='" + summary.transfer_theorem_id + "'" +
            ", quadrature_scope='" + summary.interface_quadrature_scope + "'" +
            ", rank_defect_present=" +
            std::string(summary.rank_defect_present ? "true" : "false") +
            ", rank_defect=" + std::to_string(summary.rank_defect) +
            ", projection_norm_present=" +
            std::string(summary.projection_operator_norm_present ? "true" : "false") +
            ", projection_norm=" +
            std::to_string(summary.projection_operator_norm) +
            ", accepted_projection_norm_present=" +
            std::string(summary.accepted_projection_operator_norm_present ? "true" : "false") +
            ", accepted_projection_norm=" +
            std::to_string(summary.accepted_projection_operator_norm) +
            ", mortar_inf_sup_present=" +
            std::string(summary.mortar_inf_sup_lower_bound_present ? "true" : "false") +
            ", mortar_inf_sup=" +
            std::to_string(summary.mortar_inf_sup_lower_bound) +
            ", mass_condition_present=" +
            std::string(summary.interface_mass_condition_number_present ? "true" : "false") +
            ", mass_condition=" +
            std::to_string(summary.interface_mass_condition_number) +
            ", accepted_mass_condition_present=" +
            std::string(summary.accepted_interface_mass_condition_bound_present ? "true" : "false") +
            ", accepted_mass_condition=" +
            std::to_string(summary.accepted_interface_mass_condition_bound) +
            ", rank_metadata=" +
            std::string(summary.rank_metadata_present ? "true" : "false") +
            ", interface_scope_metadata=" +
            std::string(summary.interface_scope_metadata_present ? "true" : "false") +
            ", projection_consistency=" +
            std::string(summary.projection_consistency_metadata_present ? "true" : "false") +
            ", mortar_or_dual_consistency=" +
            std::string(summary.mortar_inf_sup_or_dual_consistency_metadata_present ? "true" : "false") +
            ", mass_conditioning=" +
            std::string(summary.interface_mass_conditioning_metadata_present ? "true" : "false") +
            ", action_reaction_flux=" +
            std::string(summary.action_reaction_flux_metadata_present ? "true" : "false"),
            claim.confidence);
        if (!residual_violation && !numeric_evidence_valid) {
            addWarning(report,
                "Transfer-operator summary for interface '" +
                summary.interface_pair_id +
                "' has invalid residual tolerance or residual evidence");
        }
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->adjoint_consistency) {
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(
                summary.discrete_adjoint_tolerance);
        const bool residual_finite =
            !summary.discrete_adjoint_residual_present ||
            numeric::finite(summary.discrete_adjoint_residual);
        const Real adjoint_tol =
            effectiveTolerance(summary.discrete_adjoint_tolerance);
        const bool adjoint_residual_ok =
            summary.discrete_adjoint_residual_present &&
            tolerance_declared &&
            residual_finite &&
            std::abs(summary.discrete_adjoint_residual) <= adjoint_tol;
        const bool adjoint_residual_violation =
            summary.discrete_adjoint_residual_present &&
            tolerance_declared &&
            residual_finite &&
            std::abs(summary.discrete_adjoint_residual) > adjoint_tol;
        const bool metadata_complete =
            summary.boundary_adjoint_metadata_present &&
            summary.stabilization_adjoint_metadata_present &&
            summary.goal_linearization_metadata_present &&
            adjoint_residual_ok &&
            !summary.goal_functional_id.empty();
        PropertyClaim claim;
        claim.kind = PropertyKind::AdjointConsistency;
        claim.tested_block_id = summary.contribution_id;
        claim.estimate_scope = summary.goal_functional_id;
        claim.claim_origin = "PreservationStructureAnalyzer";
        if (summary.adjoint_consistency == AdjointConsistencyKind::Yes &&
            summary.transpose_backend_support &&
            metadata_complete) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Adjoint-consistency summary is certified for the goal functional with discrete adjoint residual evidence";
        } else if (summary.adjoint_consistency == AdjointConsistencyKind::No ||
                   !summary.transpose_backend_support ||
                   adjoint_residual_violation) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Adjoint-consistency summary reports a goal-functional risk";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Adjoint-consistency summary lacks boundary, stabilization, goal-linearization, or discrete adjoint residual metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "AdjointConsistencySummary contribution='" +
            summary.contribution_id +
            "', consistency=" +
            std::string(toString(summary.adjoint_consistency)) +
            ", transpose_backend=" +
            std::string(summary.transpose_backend_support ? "true" : "false") +
            ", boundary_metadata=" +
            std::string(summary.boundary_adjoint_metadata_present ? "true" : "false") +
            ", stabilization_metadata=" +
            std::string(summary.stabilization_adjoint_metadata_present ? "true" : "false") +
            ", goal_linearization=" +
            std::string(summary.goal_linearization_metadata_present ? "true" : "false") +
            ", discrete_adjoint_residual_present=" +
            std::string(summary.discrete_adjoint_residual_present ? "true" : "false") +
            ", discrete_adjoint_residual=" +
            std::to_string(summary.discrete_adjoint_residual) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", finite_residual=" +
            std::string(residual_finite ? "true" : "false") +
            ", goal='" + summary.goal_functional_id + "'",
            claim.confidence);
        if (!adjoint_residual_violation &&
            summary.discrete_adjoint_residual_present &&
            (!tolerance_declared || !residual_finite)) {
            addWarning(report,
                "Adjoint-consistency summary for contribution '" +
                summary.contribution_id +
                "' has invalid adjoint residual tolerance or residual");
        }
        report.claims.push_back(std::move(claim));
    }
}

void CoupledSystemStabilityAnalyzer::run(const ProblemAnalysisContext& context,
                                         ProblemAnalysisReport& report) const
{
    std::vector<std::vector<VariableKey>> summary_covered_variable_sets;
    const auto* summaries = context.analysisSummaries();
    if (summaries) {
        for (const auto& summary : summaries->coupled_system_stability) {
            if (!summary.variables.empty()) {
                summary_covered_variable_sets.push_back(summary.variables);
            }
            const Real tol = effectiveTolerance(summary.coupling_tolerance);
            const bool tolerance_declared =
                summary.coupling_tolerance_present &&
                numeric::finiteDeclaredTolerance(summary.coupling_tolerance);
            const bool exchange_residual_finite =
                summary.exchange_residual_present &&
                numeric::finite(summary.exchange_residual);
            const bool constraint_drift_finite =
                summary.constraint_drift_present &&
                numeric::finite(summary.constraint_drift_norm);
            const bool residual_evidence_present =
                summary.exchange_residual_present &&
                summary.constraint_drift_present &&
                summary.coupling_tolerance_present;
            const bool residual_numeric_evidence_present =
                tolerance_declared &&
                exchange_residual_finite &&
                constraint_drift_finite;
            const bool partition_spectral_finite =
                !summary.partitioned_coupling ||
                (summary.partition_iteration_spectral_radius_present &&
                 numeric::finiteNonnegative(
                     summary.partition_iteration_spectral_radius));
            const bool spectral_evidence_present =
                !summary.partitioned_coupling ||
                summary.partition_iteration_spectral_radius_present;
            const bool nonnormal_coupling_bound_valid =
                !summary.nonnormal_coupling_bound_present ||
                (summary.accepted_nonnormal_coupling_growth_bound_present &&
                 numeric::finiteNonnegative(
                     summary.nonnormal_coupling_growth_bound) &&
                 numeric::finiteNonnegative(
                     summary.accepted_nonnormal_coupling_growth_bound) &&
                 summary.nonnormal_coupling_growth_bound <=
                     summary.accepted_nonnormal_coupling_growth_bound + tol);
            const bool quantitative_nonnormal_coupling_evidence =
                summary.nonnormal_coupling_bound_present &&
                nonnormal_coupling_bound_valid;
            const bool theorem_scope_complete =
                !summary.coupled_stability_theorem_id.empty();
            const bool norm_scope_complete =
                summary.coupling_norm_metadata_present &&
                !summary.coupling_norm_id.empty();
            const bool operator_scope_complete =
                summary.coupling_operator_scope_metadata_present &&
                !summary.coupling_operator_scope_id.empty();
            const bool time_horizon_complete =
                summary.coupling_time_horizon_present &&
                numeric::finiteNonnegative(summary.coupling_time_horizon) &&
                !summary.coupling_time_horizon_scope.empty();
            const bool common_stability_scope_complete =
                theorem_scope_complete &&
                norm_scope_complete &&
                operator_scope_complete &&
                time_horizon_complete;
            const bool contraction_factor_bound_valid =
                summary.contraction_factor_bound_present &&
                summary.accepted_contraction_factor_bound_present &&
                numeric::finiteNonnegative(summary.contraction_factor_bound) &&
                numeric::finiteNonnegative(
                    summary.accepted_contraction_factor_bound) &&
                summary.accepted_contraction_factor_bound < Real{1} &&
                summary.contraction_factor_bound <=
                    summary.accepted_contraction_factor_bound + tol;
            const bool energy_norm_bounds_valid =
                summary.coupled_energy_norm_equivalence_bounds_present &&
                numeric::finitePositiveOrdered(
                    summary.coupled_energy_norm_equivalence_lower_bound,
                    summary.coupled_energy_norm_equivalence_upper_bound);
            const bool energy_coercivity_bound_valid =
                summary.coupled_energy_coercivity_lower_bound_present &&
                numeric::finitePositive(
                    summary.coupled_energy_coercivity_lower_bound);
            const bool invalid_quantitative_coupling_evidence =
                (summary.coupling_time_horizon_present &&
                 !numeric::finiteNonnegative(summary.coupling_time_horizon)) ||
                (summary.contraction_factor_bound_present &&
                 !numeric::finiteNonnegative(summary.contraction_factor_bound)) ||
                (summary.accepted_contraction_factor_bound_present &&
                 (!numeric::finiteNonnegative(
                      summary.accepted_contraction_factor_bound) ||
                  summary.accepted_contraction_factor_bound >= Real{1})) ||
                (summary.contraction_factor_bound_present &&
                 summary.accepted_contraction_factor_bound_present &&
                 numeric::finiteNonnegative(summary.contraction_factor_bound) &&
                 numeric::finiteNonnegative(
                     summary.accepted_contraction_factor_bound) &&
                 summary.contraction_factor_bound >
                     summary.accepted_contraction_factor_bound + tol) ||
                (summary.coupled_energy_coercivity_lower_bound_present &&
                 !numeric::finitePositive(
                     summary.coupled_energy_coercivity_lower_bound)) ||
                (summary.coupled_energy_norm_equivalence_bounds_present &&
                 !numeric::finitePositiveOrdered(
                     summary.coupled_energy_norm_equivalence_lower_bound,
                     summary.coupled_energy_norm_equivalence_upper_bound));
            const bool numeric_evidence_present =
                residual_evidence_present &&
                residual_numeric_evidence_present &&
                spectral_evidence_present &&
                partition_spectral_finite &&
                nonnormal_coupling_bound_valid &&
                !invalid_quantitative_coupling_evidence;
            const bool spectral_ok =
                !summary.partitioned_coupling ||
                (summary.partition_iteration_spectral_radius_present &&
                 partition_spectral_finite &&
                 summary.partition_iteration_spectral_radius < Real{1});
            const bool residual_ok =
                residual_numeric_evidence_present &&
                std::abs(summary.exchange_residual) <= tol &&
                std::abs(summary.constraint_drift_norm) <= tol &&
                summary.unstable_exchange_count == 0u;
            const bool residual_violation =
                summary.unstable_exchange_count > 0u ||
                (residual_numeric_evidence_present && !residual_ok);
            const bool spectral_violation =
                summary.partitioned_coupling &&
                summary.partition_iteration_spectral_radius_present &&
                partition_spectral_finite &&
                !spectral_ok;
            const bool nonnormal_coupling_bound_violation =
                summary.nonnormal_coupling_bound_present &&
                !nonnormal_coupling_bound_valid;
            const bool contractive_or_bounded_operator_evidence =
                summary.coupled_operator_stability_evidence_present &&
                common_stability_scope_complete &&
                ((summary.contraction_norm_evidence_present &&
                  contraction_factor_bound_valid) ||
                 quantitative_nonnormal_coupling_evidence ||
                 (summary.interface_energy_balance_evidence_present &&
                  summary.coupled_norm_coercivity_evidence_present &&
                  energy_coercivity_bound_valid &&
                  energy_norm_bounds_valid));
            const bool partition_metadata_complete =
                !summary.partitioned_coupling ||
                (summary.linear_stationary_iteration_evidence_present &&
                 summary.relaxation_metadata_present &&
                 summary.added_mass_risk_assessed);

            PropertyClaim claim;
            claim.kind = PropertyKind::CoupledSystemStructure;
            claim.variables = summary.variables;
            claim.constraint_drift_norm = summary.constraint_drift_norm;
            claim.claim_origin = "CoupledSystemStabilityAnalyzer";
            claim.estimate_scope = summary.coupling_group;
            if (residual_violation || spectral_violation ||
                invalid_quantitative_coupling_evidence ||
                nonnormal_coupling_bound_violation) {
                claim.status = PropertyStatus::Violated;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Violated;
                claim.description =
                    "Coupled-system stability summary violates residual, partitioned-iteration, or nonnormal coupling bounds";
            } else if (!numeric_evidence_present) {
                claim.status = PropertyStatus::Unknown;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.description =
                    "Coupled-system stability summary lacks residual, tolerance, drift, or partitioned spectral-radius evidence";
            } else if (summary.monolithic_coupling || summary.partitioned_coupling) {
                if (contractive_or_bounded_operator_evidence &&
                    partition_metadata_complete) {
                    claim.status = PropertyStatus::Preserved;
                    claim.confidence = AnalysisConfidence::High;
                    claim.certification_class = CertificationClass::Certified;
                    claim.description =
                        "Coupled-system stability summary satisfies residual bounds with coercive/contractive coupled-operator metadata";
                } else {
                    claim.status = PropertyStatus::Likely;
                    claim.confidence = AnalysisConfidence::Medium;
                    claim.certification_class = CertificationClass::NotCertified;
                    claim.description =
                        "Coupled-system residual and spectral-radius bounds pass, but coercive norm, coupled-operator, contraction/nonnormal, or partitioned-coupling metadata is incomplete";
                }
            } else {
                claim.status = PropertyStatus::Unknown;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::Unknown;
                claim.description =
                    "Coupled-system stability summary lacks coupling-mode metadata";
            }
            claim.addEvidence("CoupledSystemStabilityAnalyzer",
                "CoupledSystemStabilitySummary group='" +
                summary.coupling_group +
                "', exchange_residual=" +
                std::to_string(summary.exchange_residual) +
                ", exchange_residual_present=" +
                std::string(summary.exchange_residual_present ? "true" : "false") +
                ", spectral_radius=" +
                std::to_string(summary.partition_iteration_spectral_radius) +
                ", spectral_radius_present=" +
                std::string(summary.partition_iteration_spectral_radius_present ? "true" : "false") +
                ", constraint_drift=" +
                std::to_string(summary.constraint_drift_norm) +
                ", constraint_drift_present=" +
                std::string(summary.constraint_drift_present ? "true" : "false") +
                ", coupling_tolerance_present=" +
                std::string(summary.coupling_tolerance_present ? "true" : "false") +
                ", finite_tolerance=" +
                std::string(tolerance_declared ? "true" : "false") +
                ", finite_exchange_residual=" +
                std::string(exchange_residual_finite ? "true" : "false") +
                ", finite_constraint_drift=" +
                std::string(constraint_drift_finite ? "true" : "false") +
                ", finite_partition_spectral_radius=" +
                std::string(partition_spectral_finite ? "true" : "false") +
                ", unstable_exchanges=" +
                std::to_string(summary.unstable_exchange_count) +
                ", contraction_norm=" +
                std::string(summary.contraction_norm_evidence_present ? "true" : "false") +
                ", theorem='" + summary.coupled_stability_theorem_id + "'" +
                ", norm='" + summary.coupling_norm_id + "'" +
                ", norm_metadata=" +
                std::string(summary.coupling_norm_metadata_present ? "true" : "false") +
                ", operator_scope='" + summary.coupling_operator_scope_id + "'" +
                ", operator_scope_metadata=" +
                std::string(summary.coupling_operator_scope_metadata_present ? "true" : "false") +
                ", time_horizon_scope='" + summary.coupling_time_horizon_scope + "'" +
                ", time_horizon_present=" +
                std::string(summary.coupling_time_horizon_present ? "true" : "false") +
                ", time_horizon=" +
                std::to_string(summary.coupling_time_horizon) +
                ", contraction_factor_present=" +
                std::string(summary.contraction_factor_bound_present ? "true" : "false") +
                ", contraction_factor=" +
                std::to_string(summary.contraction_factor_bound) +
                ", accepted_contraction_factor_present=" +
                std::string(summary.accepted_contraction_factor_bound_present ? "true" : "false") +
                ", accepted_contraction_factor=" +
                std::to_string(summary.accepted_contraction_factor_bound) +
                ", nonnormal_coupling_bound_present=" +
                std::string(summary.nonnormal_coupling_bound_present ? "true" : "false") +
                ", nonnormal_coupling_growth_bound=" +
                std::to_string(summary.nonnormal_coupling_growth_bound) +
                ", accepted_nonnormal_coupling_growth_bound_present=" +
                std::string(summary.accepted_nonnormal_coupling_growth_bound_present ? "true" : "false") +
                ", accepted_nonnormal_coupling_growth_bound=" +
                std::to_string(summary.accepted_nonnormal_coupling_growth_bound) +
                ", nonnormal_coupling_bound_valid=" +
                std::string(nonnormal_coupling_bound_valid ? "true" : "false") +
                ", interface_energy=" +
                std::string(summary.interface_energy_balance_evidence_present ? "true" : "false") +
                ", coupled_norm_coercivity=" +
                std::string(summary.coupled_norm_coercivity_evidence_present ? "true" : "false") +
                ", energy_coercivity_lower_present=" +
                std::string(summary.coupled_energy_coercivity_lower_bound_present ? "true" : "false") +
                ", energy_coercivity_lower=" +
                std::to_string(summary.coupled_energy_coercivity_lower_bound) +
                ", energy_norm_equivalence_present=" +
                std::string(summary.coupled_energy_norm_equivalence_bounds_present ? "true" : "false") +
                ", energy_norm_equivalence_lower=" +
                std::to_string(summary.coupled_energy_norm_equivalence_lower_bound) +
                ", energy_norm_equivalence_upper=" +
                std::to_string(summary.coupled_energy_norm_equivalence_upper_bound) +
                ", coupled_operator_stability=" +
                std::string(summary.coupled_operator_stability_evidence_present ? "true" : "false") +
                ", partition_metadata=" +
                std::string(partition_metadata_complete ? "true" : "false"),
                claim.confidence);
            if (!residual_violation && !spectral_violation &&
                residual_evidence_present && spectral_evidence_present &&
                !numeric_evidence_present) {
                addWarning(report,
                    "Coupled-system stability summary '" +
                    summary.coupling_group +
                    "' has invalid residual, tolerance, drift, or spectral-radius evidence");
            }
            report.claims.push_back(std::move(claim));
        }
    }

    bool has_nontrivial_coupling = false;
    bool has_preserved_stability = false;
    bool has_dae_violation = false;
    std::vector<VariableKey> variables;

    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::CoupledSystemStructure &&
            claim.claim_origin != "CoupledSystemStabilityAnalyzer" &&
            !coveredByAnyVariableSet(claim.variables,
                                     summary_covered_variable_sets) &&
            sameVariableSetIsNontrivial(claim.variables)) {
            has_nontrivial_coupling = true;
            for (const auto& var : claim.variables) appendUnique(variables, var);
        }
        if (claim.kind == PropertyKind::DifferentialAlgebraicStructure &&
            claim.dae_class == DAEClass::HigherIndexRisk) {
            has_dae_violation = true;
        }
    }

    for (const auto& claim : report.claims) {
        if ((claim.kind == PropertyKind::TemporalStability ||
             claim.kind == PropertyKind::EnergyStability ||
             claim.kind == PropertyKind::ConservationStructure) &&
            (claim.status == PropertyStatus::Preserved ||
             claim.status == PropertyStatus::Exact) &&
            (localVariableSetsIntersect(claim.variables, variables) ||
             (claim.variables.empty() && claim.domain == DomainKind::Global))) {
            has_preserved_stability = true;
        }
    }

    if (!has_nontrivial_coupling) {
        return;
    }

    PropertyClaim claim;
    claim.kind = PropertyKind::CoupledSystemStructure;
    claim.variables = variables;
    claim.claim_origin = "CoupledSystemStabilityAnalyzer";
    if (has_dae_violation) {
        claim.status = PropertyStatus::Violated;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = CertificationClass::Violated;
        claim.description =
            "Coupled-system structure inherits a higher-index DAE stability risk";
    } else if (has_preserved_stability) {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::NotCertified;
        claim.description =
            "Coupled-system structure has supporting temporal, energy, or balance evidence but lacks coupling contraction metadata";
    } else {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::Unknown;
        claim.description =
            "Coupled-system stability requires numeric residual or temporal evidence";
    }
    claim.addEvidence("CoupledSystemStabilityAnalyzer",
        "Coupling claims and DAE/stability claims were inspected without physics-module names",
        claim.confidence);
    report.claims.push_back(std::move(claim));
}

} // namespace analysis
} // namespace FE
} // namespace svmp
