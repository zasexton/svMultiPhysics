/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/AdvancedStabilityAnalyzers.h"

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
    return tolerance > Real{} ? tolerance : fallback;
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

bool finiteReal(Real value) noexcept
{
    return std::isfinite(static_cast<double>(value));
}

bool finitePositive(Real value) noexcept
{
    return finiteReal(value) && value > Real{};
}

bool finiteNonnegative(Real value) noexcept
{
    return finiteReal(value) && value >= Real{};
}

bool theoremScoped(const std::string& theorem_id) noexcept
{
    return !theorem_id.empty();
}

bool energyMetadataComplete(const EnergyEntropySummary& summary) noexcept
{
    return summary.energy_functional_metadata_present &&
           summary.energy_norm_metadata_present &&
           summary.energy_positivity_evidence_present &&
           summary.energy_coercivity_evidence_present &&
           summary.discrete_dissipation_identity_evidence_present &&
           summary.boundary_source_energy_accounting_present &&
           theoremScoped(summary.energy_entropy_theorem_id) &&
           !summary.energy_functional_id.empty() &&
           !summary.energy_norm_id.empty();
}

bool entropyMetadataComplete(const EnergyEntropySummary& summary) noexcept
{
    return summary.convex_entropy_metadata_present &&
           summary.entropy_variables_metadata_present &&
           summary.entropy_flux_metadata_present &&
           summary.entropy_dissipation_metadata_present &&
           summary.boundary_source_entropy_metadata_present &&
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
           positivity == PositivityClass::Nonpositive ||
           positivity == PositivityClass::Indefinite;
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

bool coefficientCoverageComplete(const CoefficientPropertySummary& summary) noexcept
{
    const bool state_scope_ok =
        (!summary.state_dependent && !summary.time_dependent) ||
        summary.state_sample_coverage_complete;
    return summary.coefficient_region_coverage_complete &&
           summary.quadrature_point_coverage_complete &&
           summary.lower_bound_valid_for_all_samples &&
           summary.tolerance_metadata_present &&
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
        const bool amplification_bounded =
            summary.amplification_radius_present &&
            summary.amplification_radius <= Real{1} + tol;
        const bool cfl_bounded =
            summary.cfl_estimate_present &&
            summary.cfl_estimate <= Real{1} + tol;
        const bool conditional =
            summary.stability_class == TemporalStabilityClass::ConditionallyStable;
        const bool ssp =
            summary.stability_class == TemporalStabilityClass::SSP;
        const bool theorem_scoped = theoremScoped(summary.stability_theorem_id);
        const bool cfl_margin_valid =
            !summary.cfl_margin_present ||
            finiteNonnegative(summary.cfl_margin);
        const bool cfl_certificate =
            !conditional ||
            (summary.cfl_derivation_metadata_present &&
             cfl_margin_valid &&
             (summary.operator_spectrum_coverage_present ||
              summary.numerical_range_coverage_present ||
              theorem_scoped));
        const bool stability_region_certificate =
            summary.stability_region_evidence_present &&
            theorem_scoped &&
            (summary.operator_spectrum_coverage_present ||
             summary.numerical_range_coverage_present ||
             summary.energy_norm_contractivity_evidence_present ||
             summary.logarithmic_norm_bound_present);
        const bool norm_certificate =
            (summary.energy_norm_contractivity_evidence_present &&
             summary.contractivity_norm_metadata_present) ||
            (summary.logarithmic_norm_bound_present &&
             (summary.numerical_range_coverage_present || theorem_scoped)) ||
            (summary.invariant_domain_evidence_present &&
             theorem_scoped) ||
            (summary.nonlinear_stability_evidence_present &&
             theorem_scoped);
        const bool nonnormal_certificate =
            (summary.pseudospectral_bound_present &&
             theorem_scoped) ||
            (summary.nonnormal_growth_bound_present &&
             summary.nonnormal_growth_bound_finite &&
             finiteNonnegative(summary.nonnormal_growth_bound));
        const bool ssp_certificate =
            !ssp ||
            (summary.ssp_or_tvd_evidence_present && theorem_scoped);
        const bool certification_evidence_complete =
            cfl_certificate &&
            ssp_certificate &&
            (stability_region_certificate ||
             norm_certificate ||
             nonnormal_certificate);

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
        } else if (!summary.amplification_radius_present) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Time-integration stability lacks amplification-radius evidence";
        } else if (conditional && !summary.cfl_estimate_present) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Conditional time-integration stability lacks CFL evidence";
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
        } else if (!amplification_bounded) {
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
            ", contractivity_norm=" +
            std::string(summary.contractivity_norm_metadata_present ? "true" : "false") +
            ", pseudospectral=" +
            std::string(summary.pseudospectral_bound_present ? "true" : "false") +
            ", nonnormal_growth=" +
            std::string(summary.nonnormal_growth_bound_present ? "true" : "false") +
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
        const Real tol = effectiveTolerance(summary.balance_tolerance);
        const bool sign_ok =
            productionSignSatisfied(summary.expected_production_sign,
                                    summary.observed_production,
                                    tol);
        const bool balance_ok = std::abs(summary.observed_discrete_balance) <= tol;
        const bool violated = summary.violation_count > 0u || !sign_ok || !balance_ok;
        const bool metadata_complete = energyEntropyMetadataComplete(summary);

        PropertyClaim claim;
        claim.kind = lawPropertyKind(summary.law_kind);
        claim.status = summary.expected_production_sign == BalanceSignClass::Unknown
            ? PropertyStatus::Unknown
            : (violated
                ? PropertyStatus::Violated
                : (metadata_complete ? PropertyStatus::Preserved
                                     : PropertyStatus::Unknown));
        claim.confidence = claim.status == PropertyStatus::Unknown
            ? AnalysisConfidence::Medium
            : AnalysisConfidence::High;
        claim.certification_class =
            claim.status == PropertyStatus::Unknown && !metadata_complete
                ? CertificationClass::NotCertified
                : certificationFromStatus(claim.status);
        claim.estimate_scope = summary.energy_entropy_id;
        claim.description = claim.status == PropertyStatus::Violated
            ? "Discrete energy/entropy balance violates the declared production sign or tolerance"
            : (claim.status == PropertyStatus::Preserved
                ? "Discrete energy/entropy balance satisfies the declared production sign and tolerance"
                : (summary.expected_production_sign == BalanceSignClass::Unknown
                    ? "Discrete energy/entropy balance has unknown expected production sign"
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
            ", metadata_complete=" +
            std::string(metadata_complete ? "true" : "false"),
            claim.confidence);
        if (claim.status == PropertyStatus::Violated) {
            addWarning(report,
                "Energy/entropy balance violation for summary '" +
                summary.energy_entropy_id + "'");
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
        const auto status = coefficientStatus(summary.positivity);
        const bool positive_or_nonnegative =
            summary.positivity == PositivityClass::Positive ||
            summary.positivity == PositivityClass::Nonnegative;
        const bool coverage_complete = coefficientCoverageComplete(summary);

        PropertyClaim positivity;
        positivity.kind = PropertyKind::CoefficientPositivity;
        positivity.status =
            positive_or_nonnegative && !coverage_complete
                ? PropertyStatus::Likely
                : status;
        positivity.confidence = status == PropertyStatus::Unknown
            ? AnalysisConfidence::Medium
            : (positive_or_nonnegative && !coverage_complete
                   ? AnalysisConfidence::Medium
                   : AnalysisConfidence::High);
        positivity.domain = summary.domain;
        positivity.certification_class =
            positive_or_nonnegative && !coverage_complete
                ? CertificationClass::NotCertified
                : certificationFromStatus(status);
        positivity.coefficient_id = summary.coefficient;
        positivity.claim_origin = "CoefficientConstitutiveAnalyzer";
        positivity.description = status == PropertyStatus::Violated
            ? "Coefficient/constitutive summary violates positivity requirements"
            : (positivity.status == PropertyStatus::Preserved
                ? "Coefficient/constitutive summary preserves positivity requirements with scoped coverage metadata"
                : (positivity.status == PropertyStatus::Likely
                    ? "Coefficient/constitutive summary reports positive sign but lacks full coverage metadata"
                    : "Coefficient/constitutive positivity is unknown"));
        positivity.addEvidence("CoefficientConstitutiveAnalyzer",
            "CoefficientPropertySummary coefficient='" + summary.coefficient +
            "', min_eigenvalue=" +
            std::to_string(summary.min_eigenvalue) +
            ", max_eigenvalue=" +
            std::to_string(summary.max_eigenvalue) +
            ", positivity=" +
            std::to_string(static_cast<int>(summary.positivity)) +
            ", coverage_complete=" +
            std::string(coverage_complete ? "true" : "false"),
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
        if (summary.robustness_certificate_present) {
            robustness.status = PropertyStatus::Preserved;
            robustness.certification_class = CertificationClass::Certified;
            robustness.description =
                "Coefficient contrast and anisotropy are covered by reported robustness evidence";
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
            ", robustness_scope='" + summary.robustness_certificate_scope + "'",
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
        const Real tol = effectiveTolerance(summary.finite_difference_tolerance);
        const bool action_checked = summary.jacobian_action_available;
        const bool action_ok =
            action_checked &&
            summary.finite_difference_action_error <= tol;
        const bool action_violation =
            action_checked &&
            summary.finite_difference_action_error > tol;
        const bool inconsistent =
            summary.tangent_consistency == TangentConsistencyClass::Frozen ||
            summary.tangent_consistency == TangentConsistencyClass::Inconsistent ||
            action_violation ||
            summary.newton_stagnation_count > 0u ||
            tangentPositivityViolation(summary.tangent_positivity);

        PropertyClaim claim;
        claim.kind = PropertyKind::NonlinearTangentStructure;
        claim.variables = blockVariables(summary.block);
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag.empty()
            ? summary.residual_id
            : summary.block.operator_tag;
        claim.operator_symmetry_class =
            operatorSymmetryFrom(summary.tangent_symmetry);
        claim.coercivity_class = coercivityFrom(summary.tangent_positivity);
        claim.claim_origin = "NonlinearTangentAnalyzer";

        if (tangentPositivityViolation(summary.tangent_positivity)) {
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
            ", stagnation_count=" +
            std::to_string(summary.newton_stagnation_count) +
            ", tangent_symmetry=" +
            symmetryName(summary.tangent_symmetry) +
            ", tangent_positivity=" +
            positivityName(summary.tangent_positivity) +
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
        }
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
        const bool convergence_evidence =
            summary.operator_convergence_evidence ||
            summary.discrete_compactness_evidence ||
            summary.gap_convergence_evidence ||
            (summary.compatible_complex_evidence &&
             summary.compatible_complex_spectral_theorem_evidence &&
             theorem_scoped);
        const bool diagnostic_scope_ok =
            theorem_scoped ||
            (summary.refinement_scope_metadata_present &&
             summary.refinement_sample_count >= 2u);
        const bool certified =
            summary.eigenproblem_declared &&
            summary.self_adjoint_evidence &&
            summary.compactness_evidence &&
            convergence_evidence &&
            diagnostic_scope_ok &&
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
                "Spectral summary has compact/self-adjoint evidence plus theorem-scoped convergence, discrete-compactness, or gap metadata";
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
            ", gap_convergence=" +
            std::string(summary.gap_convergence_evidence ? "true" : "false") +
            ", theorem='" + summary.spectral_convergence_theorem_id + "'" +
            ", refinement_scope=" +
            std::string(summary.refinement_scope_metadata_present ? "true" : "false") +
            ", refinement_samples=" +
            std::to_string(summary.refinement_sample_count) +
            ", spurious_modes=" +
            std::to_string(summary.spurious_mode_count),
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
        const bool invalid_quantitative_metadata =
            (summary.reliability_constant_metadata_present &&
             !finitePositive(summary.reliability_constant)) ||
            (summary.efficiency_constant_metadata_present &&
             !finitePositive(summary.efficiency_constant)) ||
            (summary.effectivity_bounds_present &&
             !effectivity_bounds_valid);
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
            summary.shape_regular_mesh_evidence_present &&
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
                "A posteriori estimator metadata has residual, localization, norm, regularity, boundary, oscillation, effectivity, and refinement evidence";
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
        const Real tol = effectiveTolerance(summary.aliasing_tolerance);
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
            (!summary.curved_or_nonlinear_mapping ||
             summary.overintegration_metadata_present ||
             summary.nonlinear_aliasing_control_present);
        const bool aliasing_violation =
            summary.aliasing_indicator > tol ||
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
            ", curved_or_nonlinear_mapping=" +
            std::string(summary.curved_or_nonlinear_mapping ? "true" : "false") +
            ", underintegrated_entries=" +
            std::to_string(summary.underintegrated_entry_count) +
            ", zero_energy_modes=" +
            std::to_string(summary.zero_energy_mode_count) +
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
            (summary.fortin_operator_evidence_present ||
             summary.optimal_test_metadata_present) &&
            summary.enrichment_sufficiency_evidence_present &&
            residual_control_valid;
        const bool conditioning_metadata_present =
            local_conditioning_valid &&
            normal_conditioning_valid;
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
             !finitePositive(summary.normal_equation_condition_estimate));
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
                "Minimum-residual/Petrov-Galerkin summary reports violated residual, Fortin, Riesz, enrichment, distinct-space, or conditioning checks";
        } else if (certified) {
            claim.description =
                "Minimum-residual/Petrov-Galerkin stability is certified by scoped trial/test, residual norm, test norm, Riesz map, Fortin/optimal-test, enrichment, residual-control, theorem, and conditioning evidence";
        } else if (core_metadata_present) {
            claim.description =
                "Minimum-residual/Petrov-Galerkin method is structurally eligible but lacks complete Fortin/optimal-test, enrichment, residual-control, or conditioning evidence";
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
            ", normal_condition=" +
            std::to_string(summary.normal_equation_condition_estimate) +
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
        const bool bounds_declared =
            summary.lower_bound_active || summary.upper_bound_active;
        const bool metadata_complete =
            bounds_declared &&
            summary.limiter_evidence_present &&
            summary.cfl_condition_satisfied &&
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
        if (summary.post_step_violation_count > 0u) {
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
                "Invariant-domain preservation lacks limiter, bound, CFL/SSP, source-admissibility, monotone low-order, mass-positivity, or theorem metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "InvariantDomainSummary id='" + summary.invariant_set_id +
            "', lower_active=" +
            std::string(summary.lower_bound_active ? "true" : "false") +
            ", upper_active=" +
            std::string(summary.upper_bound_active ? "true" : "false") +
            ", limiter=" +
            std::string(summary.limiter_evidence_present ? "true" : "false") +
            ", cfl_condition=" +
            std::string(summary.cfl_condition_satisfied ? "true" : "false") +
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
        const Real tol = effectiveTolerance(summary.residual_tolerance);
        const bool metadata_present =
            summary.source_quadrature_metadata_present &&
            summary.reconstruction_metadata_present &&
            summary.boundary_compatibility_metadata_present;
        const bool residual_ok = std::abs(summary.flux_source_residual) <= tol;

        PropertyClaim claim;
        claim.kind = PropertyKind::EquilibriumPreservation;
        claim.equilibrium_id = summary.equilibrium_id;
        claim.well_balanced_metadata_present = metadata_present;
        claim.flux_balance_residual = summary.flux_source_residual;
        claim.claim_origin = "PreservationStructureAnalyzer";
        if (!residual_ok) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Equilibrium-preservation flux/source residual exceeds tolerance";
        } else if (metadata_present) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Equilibrium-preservation summary has zero flux/source residual within tolerance";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Equilibrium residual is small but metadata is incomplete";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "EquilibriumPreservationSummary id='" +
            summary.equilibrium_id +
            "', flux_source_residual=" +
            std::to_string(summary.flux_source_residual) +
            ", tolerance=" + std::to_string(tol) +
            ", source_quadrature=" +
            std::string(summary.source_quadrature_metadata_present ? "true" : "false") +
            ", reconstruction=" +
            std::string(summary.reconstruction_metadata_present ? "true" : "false") +
            ", boundary_compatibility=" +
            std::string(summary.boundary_compatibility_metadata_present ? "true" : "false"),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->moving_domain) {
        const Real tol = effectiveTolerance(summary.geometric_conservation_tolerance);
        const bool jacobian_positive = summary.min_geometric_jacobian > Real{};
        const bool residual_ok =
            std::abs(summary.geometric_conservation_residual) <= tol;
        const bool metadata_present =
            summary.mesh_velocity_metadata_present &&
            summary.time_integration_metadata_present &&
            summary.remap_metadata_present;

        PropertyClaim claim;
        claim.kind = PropertyKind::GeometricConservation;
        claim.flux_balance_residual = summary.geometric_conservation_residual;
        claim.claim_origin = "PreservationStructureAnalyzer";
        claim.estimate_scope =
            "mesh_revision=" + std::to_string(summary.mesh_revision);
        if (!jacobian_positive || !residual_ok) {
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
                "Moving-domain summary preserves positive mapping and geometric conservation";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Moving-domain geometric conservation lacks mesh-velocity, time-integration, or remap metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "MovingDomainSummary min_jacobian=" +
            std::to_string(summary.min_geometric_jacobian) +
            ", max_jacobian=" +
            std::to_string(summary.max_geometric_jacobian) +
            ", gcl_residual=" +
            std::to_string(summary.geometric_conservation_residual) +
            ", tolerance=" + std::to_string(tol) +
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
        const Real tol = effectiveTolerance(summary.residual_tolerance);
        const bool residual_ok =
            std::abs(summary.conservation_residual) <= tol &&
            std::abs(summary.constant_preservation_residual) <= tol;
        const bool metadata_complete =
            summary.rank_metadata_present &&
            summary.interface_scope_metadata_present &&
            summary.projection_consistency_metadata_present &&
            summary.mortar_inf_sup_or_dual_consistency_metadata_present &&
            summary.interface_mass_conditioning_metadata_present &&
            summary.action_reaction_flux_metadata_present &&
            !summary.interface_pair_id.empty() &&
            !summary.projection_space_id.empty();

        PropertyClaim claim;
        claim.kind = PropertyKind::TransferOperatorCompatibility;
        claim.flux_balance_residual = std::max(
            std::abs(summary.conservation_residual),
            std::abs(summary.constant_preservation_residual));
        claim.estimate_scope = summary.interface_pair_id;
        claim.tested_block_id = summary.projection_space_id;
        claim.claim_origin = "PreservationStructureAnalyzer";
        if (!residual_ok) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Transfer summary violates conservation or constant preservation residual tolerance";
        } else if (metadata_complete) {
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
                "Transfer summary residuals pass but projection, mortar/dual-consistency, conditioning, action-reaction, scope, or rank metadata is incomplete";
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
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->adjoint_consistency) {
        const Real adjoint_tol =
            effectiveTolerance(summary.discrete_adjoint_tolerance);
        const bool adjoint_residual_ok =
            summary.discrete_adjoint_residual_present &&
            std::abs(summary.discrete_adjoint_residual) <= adjoint_tol;
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
                   (summary.discrete_adjoint_residual_present &&
                    !adjoint_residual_ok)) {
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
            ", goal='" + summary.goal_functional_id + "'",
            claim.confidence);
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
            const bool residual_evidence_present =
                summary.exchange_residual_present &&
                summary.constraint_drift_present &&
                summary.coupling_tolerance_present;
            const bool spectral_evidence_present =
                !summary.partitioned_coupling ||
                summary.partition_iteration_spectral_radius_present;
            const bool numeric_evidence_present =
                residual_evidence_present && spectral_evidence_present;
            const bool spectral_ok =
                !summary.partitioned_coupling ||
                (summary.partition_iteration_spectral_radius_present &&
                 summary.partition_iteration_spectral_radius < Real{1});
            const bool residual_ok =
                residual_evidence_present &&
                std::abs(summary.exchange_residual) <= tol &&
                std::abs(summary.constraint_drift_norm) <= tol &&
                summary.unstable_exchange_count == 0u;
            const bool residual_violation =
                residual_evidence_present && !residual_ok;
            const bool spectral_violation =
                summary.partitioned_coupling &&
                summary.partition_iteration_spectral_radius_present &&
                !spectral_ok;
            const bool contractive_or_bounded_operator_evidence =
                summary.coupled_operator_stability_evidence_present &&
                (summary.contraction_norm_evidence_present ||
                 summary.nonnormal_coupling_bound_present ||
                 (summary.interface_energy_balance_evidence_present &&
                  summary.coupled_norm_coercivity_evidence_present));
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
            if (residual_violation || spectral_violation) {
                claim.status = PropertyStatus::Violated;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Violated;
                claim.description =
                    "Coupled-system stability summary violates residual or partitioned-iteration bounds";
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
                ", unstable_exchanges=" +
                std::to_string(summary.unstable_exchange_count) +
                ", contraction_norm=" +
                std::string(summary.contraction_norm_evidence_present ? "true" : "false") +
                ", interface_energy=" +
                std::string(summary.interface_energy_balance_evidence_present ? "true" : "false") +
                ", coupled_norm_coercivity=" +
                std::string(summary.coupled_norm_coercivity_evidence_present ? "true" : "false") +
                ", coupled_operator_stability=" +
                std::string(summary.coupled_operator_stability_evidence_present ? "true" : "false") +
                ", partition_metadata=" +
                std::string(partition_metadata_complete ? "true" : "false"),
                claim.confidence);
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
