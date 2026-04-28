/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/AdvancedStabilityAnalyzers.h"

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ConstraintAnalysisSummary.h"

#include <algorithm>
#include <cmath>
#include <limits>
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
            summary.amplification_radius == Real{} ||
            summary.amplification_radius <= Real{1} + tol;
        const bool cfl_bounded =
            summary.cfl_estimate == Real{} ||
            summary.cfl_estimate <= Real{1} + tol;

        PropertyClaim claim;
        claim.kind = PropertyKind::TemporalStability;
        claim.temporal_stability_class = summary.stability_class;
        claim.cfl_number = summary.cfl_estimate;
        claim.claim_origin = "TemporalStabilityAnalyzer";

        if (summary.stability_class == TemporalStabilityClass::Unknown) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description = "Time-integration stability class is unknown";
        } else if (summary.stability_class == TemporalStabilityClass::ConditionallyStable &&
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
        } else {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Time-integration stability summary satisfies the available scalar bounds";
        }

        claim.addEvidence("TemporalStabilityAnalyzer",
            "TemporalStabilitySummary scheme='" + summary.time_scheme +
            "', cfl=" + std::to_string(summary.cfl_estimate) +
            ", eigenvalue_scale=" +
            std::to_string(summary.eigenvalue_scale_estimate) +
            ", amplification_radius=" +
            std::to_string(summary.amplification_radius),
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

        PropertyClaim claim;
        claim.kind = lawPropertyKind(summary.law_kind);
        claim.status = summary.expected_production_sign == BalanceSignClass::Unknown
            ? PropertyStatus::Unknown
            : (violated ? PropertyStatus::Violated : PropertyStatus::Preserved);
        claim.confidence = claim.status == PropertyStatus::Unknown
            ? AnalysisConfidence::Medium
            : AnalysisConfidence::High;
        claim.certification_class = certificationFromStatus(claim.status);
        claim.estimate_scope = summary.energy_entropy_id;
        claim.description = claim.status == PropertyStatus::Violated
            ? "Discrete energy/entropy balance violates the declared production sign or tolerance"
            : (claim.status == PropertyStatus::Preserved
                ? "Discrete energy/entropy balance satisfies the declared production sign and tolerance"
                : "Discrete energy/entropy balance has unknown expected production sign");
        claim.claim_origin = "EnergyEntropyLawAnalyzer";
        claim.addEvidence("EnergyEntropyLawAnalyzer",
            "EnergyEntropySummary id='" + summary.energy_entropy_id +
            "', observed_balance=" +
            std::to_string(summary.observed_discrete_balance) +
            ", observed_production=" +
            std::to_string(summary.observed_production) +
            ", violations=" + std::to_string(summary.violation_count),
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

        PropertyClaim positivity;
        positivity.kind = PropertyKind::CoefficientPositivity;
        positivity.status = status;
        positivity.confidence = status == PropertyStatus::Unknown
            ? AnalysisConfidence::Medium
            : AnalysisConfidence::High;
        positivity.domain = summary.domain;
        positivity.certification_class = certificationFromStatus(status);
        positivity.coefficient_id = summary.coefficient;
        positivity.claim_origin = "CoefficientConstitutiveAnalyzer";
        positivity.description = status == PropertyStatus::Violated
            ? "Coefficient/constitutive summary violates positivity requirements"
            : (status == PropertyStatus::Preserved
                ? "Coefficient/constitutive summary preserves positivity requirements"
                : "Coefficient/constitutive positivity is unknown");
        positivity.addEvidence("CoefficientConstitutiveAnalyzer",
            "CoefficientPropertySummary coefficient='" + summary.coefficient +
            "', min_eigenvalue=" +
            std::to_string(summary.min_eigenvalue) +
            ", max_eigenvalue=" +
            std::to_string(summary.max_eigenvalue) +
            ", positivity=" +
            std::to_string(static_cast<int>(summary.positivity)),
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
        if (worst_scale > Real{1.0e6}) {
            robustness.status = PropertyStatus::Violated;
            robustness.certification_class = CertificationClass::Violated;
            robustness.description =
                "Coefficient contrast or anisotropy is outside the robust range";
        } else if (worst_scale > Real{1.0e3} ||
                   summary.state_dependent ||
                   summary.time_dependent) {
            robustness.status = PropertyStatus::Likely;
            robustness.certification_class = CertificationClass::NotCertified;
            robustness.description =
                "Coefficient contrast, anisotropy, or state dependence requires robustness evidence";
        } else {
            robustness.status = PropertyStatus::Preserved;
            robustness.certification_class = CertificationClass::Certified;
            robustness.description =
                "Coefficient contrast and anisotropy are within the generic robust range";
        }
        robustness.addEvidence("CoefficientConstitutiveAnalyzer",
            "anisotropy_ratio=" + std::to_string(anisotropy) +
            ", contrast_ratio=" + std::to_string(contrast) +
            ", state_dependent=" +
            std::string(summary.state_dependent ? "true" : "false") +
            ", time_dependent=" +
            std::string(summary.time_dependent ? "true" : "false"),
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
            !action_checked ||
            summary.finite_difference_action_error <= tol;
        const bool inconsistent =
            summary.tangent_consistency == TangentConsistencyClass::Frozen ||
            summary.tangent_consistency == TangentConsistencyClass::Inconsistent ||
            !action_ok ||
            summary.newton_stagnation_count > 0u;

        PropertyClaim claim;
        claim.kind = PropertyKind::NonlinearTangentStructure;
        claim.variables = blockVariables(summary.block);
        claim.domain = summary.block.domain;
        claim.tested_block_id = summary.block.operator_tag.empty()
            ? summary.residual_id
            : summary.block.operator_tag;
        claim.claim_origin = "NonlinearTangentAnalyzer";

        if (summary.tangent_consistency == TangentConsistencyClass::Unknown &&
            !action_checked) {
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
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Nonlinear residual/tangent consistency is certified by exact tangent metadata";
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
            std::to_string(summary.newton_stagnation_count),
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
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::Certified;
        claim.description =
            "Constraint/space evidence does not indicate a locking risk";
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
        const bool certified =
            summary.eigenproblem_declared &&
            summary.self_adjoint_evidence &&
            summary.compactness_evidence &&
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
                "Spectral summary has self-adjoint, compact, spurious-mode-free evidence";
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
        } else if (has_localization_channel) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "A posteriori estimator metadata has residual and localization evidence";
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
            std::to_string(summary.missing_required_metadata_count),
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
        } else if (degree_exact ||
                   (summary.nonlinear_aliasing_control_present &&
                    !aliasing_violation) ||
                   reduced_with_control) {
            claim.status = reduced_with_control && !degree_exact
                ? PropertyStatus::Likely
                : PropertyStatus::Preserved;
            claim.confidence = degree_exact
                ? AnalysisConfidence::High
                : AnalysisConfidence::Medium;
            claim.certification_class = degree_exact
                ? CertificationClass::Certified
                : CertificationClass::NotCertified;
            claim.description =
                "Quadrature summary has adequate exactness or aliasing-control evidence";
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

std::string PreservationStructureAnalyzer::name() const
{
    return "PreservationStructureAnalyzer";
}

void PreservationStructureAnalyzer::run(const ProblemAnalysisContext& context,
                                        ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->invariant_domains) {
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
        } else if (summary.limiter_evidence_present ||
                   summary.lower_bound_active ||
                   summary.upper_bound_active) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Invariant-domain summary preserves the declared bounds";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Invariant-domain preservation lacks limiter or active-bound metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "InvariantDomainSummary id='" + summary.invariant_set_id +
            "', lower_active=" +
            std::string(summary.lower_bound_active ? "true" : "false") +
            ", upper_active=" +
            std::string(summary.upper_bound_active ? "true" : "false") +
            ", limiter=" +
            std::string(summary.limiter_evidence_present ? "true" : "false") +
            ", post_step_violations=" +
            std::to_string(summary.post_step_violation_count),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->equilibrium_preservation) {
        const Real tol = effectiveTolerance(summary.residual_tolerance);
        const bool metadata_present =
            summary.source_quadrature_metadata_present &&
            summary.reconstruction_metadata_present;
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
            std::string(summary.reconstruction_metadata_present ? "true" : "false"),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->moving_domain) {
        const Real tol = effectiveTolerance(summary.geometric_conservation_tolerance);
        const bool jacobian_positive = summary.min_geometric_jacobian > Real{};
        const bool residual_ok =
            std::abs(summary.geometric_conservation_residual) <= tol;

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
        } else if (summary.mesh_velocity_metadata_present) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Moving-domain summary preserves positive mapping and geometric conservation";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Moving-domain geometric conservation lacks mesh-velocity metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "MovingDomainSummary min_jacobian=" +
            std::to_string(summary.min_geometric_jacobian) +
            ", max_jacobian=" +
            std::to_string(summary.max_geometric_jacobian) +
            ", gcl_residual=" +
            std::to_string(summary.geometric_conservation_residual) +
            ", tolerance=" + std::to_string(tol),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->transfer_operators) {
        constexpr Real tol = Real{1.0e-10};
        const bool residual_ok =
            std::abs(summary.conservation_residual) <= tol &&
            std::abs(summary.constant_preservation_residual) <= tol;
        const bool certified = residual_ok && summary.rank_metadata_present;

        PropertyClaim claim;
        claim.kind = PropertyKind::TransferOperatorCompatibility;
        claim.flux_balance_residual = std::max(
            std::abs(summary.conservation_residual),
            std::abs(summary.constant_preservation_residual));
        claim.estimate_scope = summary.interface_pair_id;
        claim.tested_block_id = summary.projection_space_id;
        claim.claim_origin = "PreservationStructureAnalyzer";
        if (!residual_ok || !summary.rank_metadata_present) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Transfer summary violates conservation, constant preservation, or rank metadata requirements";
        } else if (certified) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Transfer summary preserves constants, conservation, and rank metadata";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "TransferOperatorSummary interface='" +
            summary.interface_pair_id +
            "', projection='" + summary.projection_space_id +
            "', conservation_residual=" +
            std::to_string(summary.conservation_residual) +
            ", constant_residual=" +
            std::to_string(summary.constant_preservation_residual) +
            ", rank_metadata=" +
            std::string(summary.rank_metadata_present ? "true" : "false"),
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }

    for (const auto& summary : summaries->adjoint_consistency) {
        PropertyClaim claim;
        claim.kind = PropertyKind::AdjointConsistency;
        claim.tested_block_id = summary.contribution_id;
        claim.estimate_scope = summary.goal_functional_id;
        claim.claim_origin = "PreservationStructureAnalyzer";
        if (summary.adjoint_consistency == AdjointConsistencyKind::Yes &&
            summary.transpose_backend_support) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.description =
                "Adjoint-consistency summary is certified for the goal functional";
        } else if (summary.adjoint_consistency == AdjointConsistencyKind::No ||
                   !summary.transpose_backend_support) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.description =
                "Adjoint-consistency summary reports a goal-functional risk";
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::Unknown;
            claim.description =
                "Adjoint-consistency summary is unknown";
        }
        claim.addEvidence("PreservationStructureAnalyzer",
            "AdjointConsistencySummary contribution='" +
            summary.contribution_id +
            "', consistency=" +
            std::string(toString(summary.adjoint_consistency)) +
            ", transpose_backend=" +
            std::string(summary.transpose_backend_support ? "true" : "false") +
            ", goal='" + summary.goal_functional_id + "'",
            claim.confidence);
        report.claims.push_back(std::move(claim));
    }
}

void CoupledSystemStabilityAnalyzer::run(const ProblemAnalysisContext& context,
                                         ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (summaries) {
        for (const auto& summary : summaries->coupled_system_stability) {
            const Real tol = effectiveTolerance(summary.coupling_tolerance);
            const bool spectral_ok =
                summary.partition_iteration_spectral_radius == Real{} ||
                summary.partition_iteration_spectral_radius < Real{1};
            const bool residual_ok =
                std::abs(summary.exchange_residual) <= tol &&
                std::abs(summary.constraint_drift_norm) <= tol &&
                summary.unstable_exchange_count == 0u;

            PropertyClaim claim;
            claim.kind = PropertyKind::CoupledSystemStructure;
            claim.variables = summary.variables;
            claim.constraint_drift_norm = summary.constraint_drift_norm;
            claim.claim_origin = "CoupledSystemStabilityAnalyzer";
            claim.estimate_scope = summary.coupling_group;
            if (!residual_ok || !spectral_ok) {
                claim.status = PropertyStatus::Violated;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Violated;
                claim.description =
                    "Coupled-system stability summary violates residual or partitioned-iteration bounds";
            } else if (summary.monolithic_coupling || summary.partitioned_coupling) {
                claim.status = PropertyStatus::Preserved;
                claim.confidence = AnalysisConfidence::High;
                claim.certification_class = CertificationClass::Certified;
                claim.description =
                    "Coupled-system stability summary satisfies residual and iteration bounds";
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
                ", spectral_radius=" +
                std::to_string(summary.partition_iteration_spectral_radius) +
                ", constraint_drift=" +
                std::to_string(summary.constraint_drift_norm) +
                ", unstable_exchanges=" +
                std::to_string(summary.unstable_exchange_count),
                claim.confidence);
            report.claims.push_back(std::move(claim));
        }
        if (!summaries->coupled_system_stability.empty()) {
            return;
        }
    }

    bool has_nontrivial_coupling = false;
    bool has_preserved_stability = false;
    bool has_dae_violation = false;
    std::vector<VariableKey> variables;

    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::CoupledSystemStructure &&
            sameVariableSetIsNontrivial(claim.variables)) {
            has_nontrivial_coupling = true;
            for (const auto& var : claim.variables) appendUnique(variables, var);
        }
        if (claim.kind == PropertyKind::DifferentialAlgebraicStructure &&
            claim.dae_class == DAEClass::HigherIndexRisk) {
            has_dae_violation = true;
        }
        if ((claim.kind == PropertyKind::TemporalStability ||
             claim.kind == PropertyKind::EnergyStability ||
             claim.kind == PropertyKind::ConservationStructure) &&
            (claim.status == PropertyStatus::Preserved ||
             claim.status == PropertyStatus::Exact)) {
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
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::Medium;
        claim.certification_class = CertificationClass::Certified;
        claim.description =
            "Coupled-system structure has supporting temporal, energy, or balance evidence";
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
