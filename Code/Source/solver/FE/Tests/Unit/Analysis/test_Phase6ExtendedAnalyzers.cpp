/**
 * @file test_Phase6ExtendedAnalyzers.cpp
 * @brief Phase 6 tests for extended, physics-agnostic stability analyzers.
 */

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ConstraintAnalysisSummary.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

namespace {

const PropertyClaim* firstClaimFrom(const ProblemAnalysisReport& report,
                                    PropertyKind kind,
                                    const std::string& origin)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == kind && claim.claim_origin == origin) {
            return &claim;
        }
    }
    return nullptr;
}

std::vector<const PropertyClaim*> claimsFrom(const ProblemAnalysisReport& report,
                                             PropertyKind kind,
                                             const std::string& origin)
{
    std::vector<const PropertyClaim*> claims;
    for (const auto& claim : report.claims) {
        if (claim.kind == kind && claim.claim_origin == origin) {
            claims.push_back(&claim);
        }
    }
    return claims;
}

bool hasClaimFrom(const ProblemAnalysisReport& report,
                  PropertyKind kind,
                  const std::string& origin)
{
    return firstClaimFrom(report, kind, origin) != nullptr;
}

OperatorBlockId scalarBlock(std::string tag = "generic-block",
                            DomainKind domain = DomainKind::Cell)
{
    OperatorBlockId block;
    block.test_variables = {VariableKey::field(0)};
    block.trial_variables = {VariableKey::field(0)};
    block.operator_tag = std::move(tag);
    block.domain = domain;
    return block;
}

ProblemAnalysisReport analyzeWithSummaries(AnalysisSummarySet summaries)
{
    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    return ProblemAnalyzer::createDefault().analyze(ctx);
}

} // namespace

TEST(Phase6ExtendedAnalyzers, SpaceCompatibilityCompatibleComplexCertifiedViolatedAndMissing)
{
    AnalysisSummarySet good_summaries;
    CompatibleComplexSummary good;
    good.complex_id = "sequence-ok";
    good.variables = {VariableKey::field(0), VariableKey::field(1)};
    good.exact_sequence_compatible = true;
    good.trace_sequence_compatible = true;
    good.commuting_projection_available = true;
    good_summaries.compatible_complexes.push_back(good);

    auto good_report = analyzeWithSummaries(std::move(good_summaries));
    const auto* good_claim = firstClaimFrom(
        good_report, PropertyKind::CompatibleComplexStructure,
        "SpaceCompatibilityAnalyzer");
    ASSERT_NE(good_claim, nullptr);
    EXPECT_EQ(good_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(good_claim->exact_sequence_compatible.has_value());
    EXPECT_TRUE(*good_claim->exact_sequence_compatible);
    ASSERT_TRUE(good_claim->commuting_projection_available.has_value());
    EXPECT_TRUE(*good_claim->commuting_projection_available);

    AnalysisSummarySet bad_summaries;
    CompatibleComplexSummary bad;
    bad.complex_id = "sequence-bad";
    bad.variables = {VariableKey::field(0), VariableKey::field(1)};
    bad.exact_sequence_compatible = false;
    bad.trace_sequence_compatible = true;
    bad.commuting_projection_available = false;
    bad.missing_space_count = 1;
    bad_summaries.compatible_complexes.push_back(bad);

    auto bad_report = analyzeWithSummaries(std::move(bad_summaries));
    const auto* bad_claim = firstClaimFrom(
        bad_report, PropertyKind::CompatibleComplexStructure,
        "SpaceCompatibilityAnalyzer");
    ASSERT_NE(bad_claim, nullptr);
    EXPECT_EQ(bad_claim->status, PropertyStatus::Violated);

    auto missing_report = ProblemAnalyzer::createDefault().analyze(
        ProblemAnalysisContext{});
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::CompatibleComplexStructure,
                              "SpaceCompatibilityAnalyzer"));
}

TEST(Phase6ExtendedAnalyzers, TransportCharacterConsumesPecletCflNonnormalSummaries)
{
    ProblemAnalysisContext ctx;
    ctx.addContribution(ContributionDescriptor::transportLike(
        VariableKey::field(0), "transport-block", "test"));

    AnalysisSummarySet summaries;
    ParameterScaleSummary scale;
    scale.max_scale_value = 12.0;
    summaries.parameter_scales.push_back(scale);

    TemporalStabilitySummary temporal;
    temporal.stability_class = TemporalStabilityClass::ConditionallyStable;
    temporal.cfl_estimate = 1.25;
    summaries.temporal_stability.push_back(temporal);

    DiscreteMatrixSummary matrix;
    matrix.block = scalarBlock("transport-block");
    matrix.max_abs_entry = 4.0;
    matrix.max_symmetry_error = 1.0;
    summaries.discrete_matrices.push_back(matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);
    const auto* transport = firstClaimFrom(
        report, PropertyKind::OperatorTransportCharacter,
        "TransportCharacterAnalyzer");
    ASSERT_NE(transport, nullptr);
    ASSERT_TRUE(transport->peclet_number.has_value());
    EXPECT_DOUBLE_EQ(*transport->peclet_number, 12.0);
    ASSERT_TRUE(transport->cfl_number.has_value());
    EXPECT_DOUBLE_EQ(*transport->cfl_number, 1.25);
    ASSERT_TRUE(transport->nonnormality_indicator.has_value());
    EXPECT_DOUBLE_EQ(*transport->nonnormality_indicator, 0.25);
    EXPECT_TRUE(transport->transport_character_class.has_value());
    EXPECT_EQ(*transport->transport_character_class,
              TransportCharacterClass::NonNormalRisk);

    ProblemAnalysisContext missing_ctx;
    missing_ctx.addContribution(ContributionDescriptor::transportLike(
        VariableKey::field(0), "transport-block", "test"));
    auto missing_report = ProblemAnalyzer::createDefault().analyze(missing_ctx);
    const auto* missing_transport = firstClaimFrom(
        missing_report, PropertyKind::OperatorTransportCharacter,
        "TransportCharacterAnalyzer");
    ASSERT_NE(missing_transport, nullptr);
    EXPECT_FALSE(missing_transport->peclet_number.has_value());
    EXPECT_FALSE(missing_transport->cfl_number.has_value());
    EXPECT_FALSE(missing_transport->nonnormality_indicator.has_value());
}

TEST(Phase6ExtendedAnalyzers, ConservationConsumesFluxBalanceCertifiedViolatedAndMissing)
{
    AnalysisSummarySet good_summaries;
    FluxBalanceSummary good;
    good.block = scalarBlock("balance-ok");
    good.balance_tolerance = 1.0e-6;
    good.local_residual_norm = 1.0e-8;
    good.global_residual_norm = 2.0e-8;
    good_summaries.flux_balances.push_back(good);

    auto good_report = analyzeWithSummaries(std::move(good_summaries));
    const auto* good_claim = firstClaimFrom(
        good_report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer");
    ASSERT_NE(good_claim, nullptr);
    EXPECT_EQ(good_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(good_claim->flux_balance_residual.has_value());
    EXPECT_LT(*good_claim->flux_balance_residual, 1.0e-6);

    AnalysisSummarySet bad_summaries;
    FluxBalanceSummary bad;
    bad.block = scalarBlock("balance-bad");
    bad.balance_tolerance = 1.0e-6;
    bad.local_residual_norm = 1.0e-4;
    bad.local_violation_count = 1;
    bad_summaries.flux_balances.push_back(bad);

    auto bad_report = analyzeWithSummaries(std::move(bad_summaries));
    const auto* bad_claim = firstClaimFrom(
        bad_report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer");
    ASSERT_NE(bad_claim, nullptr);
    EXPECT_EQ(bad_claim->status, PropertyStatus::Violated);

    auto missing_report = ProblemAnalyzer::createDefault().analyze(
        ProblemAnalysisContext{});
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::ConservationStructure,
                              "ConservationAnalyzer"));
}

TEST(Phase6ExtendedAnalyzers, InterfaceValidationConsumesBoundaryPenaltyAndFluxEvidence)
{
    AnalysisSummarySet summaries;
    BoundarySymbolSummary boundary;
    boundary.block = scalarBlock("weak-boundary", DomainKind::Boundary);
    boundary.principal_operator_order = 2;
    boundary.boundary_operator_order = 1;
    boundary.complementing_condition_satisfied = true;
    summaries.boundary_symbols.push_back(boundary);
    ParameterScaleSummary penalty;
    penalty.max_scale_value = 2.0;
    summaries.parameter_scales.push_back(penalty);
    FluxBalanceSummary flux;
    flux.block = scalarBlock("weak-boundary", DomainKind::Boundary);
    flux.balance_tolerance = 1.0e-6;
    flux.interface_pair_residual_norm = 1.0e-8;
    summaries.flux_balances.push_back(flux);

    auto report = analyzeWithSummaries(std::move(summaries));
    const auto* complementing = firstClaimFrom(
        report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(complementing, nullptr);
    EXPECT_EQ(complementing->status, PropertyStatus::Preserved);
    const auto* coercivity = firstClaimFrom(
        report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(coercivity, nullptr);
    EXPECT_EQ(coercivity->status, PropertyStatus::Preserved);
    ASSERT_TRUE(coercivity->penalty_scale.has_value());
    EXPECT_DOUBLE_EQ(*coercivity->penalty_scale, 2.0);
    const auto* interface_flux = firstClaimFrom(
        report, PropertyKind::InterfaceCondition,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(interface_flux, nullptr);
    EXPECT_EQ(interface_flux->status, PropertyStatus::Preserved);

    AnalysisSummarySet bad_summaries;
    BoundarySymbolSummary bad_boundary;
    bad_boundary.block = scalarBlock("weak-boundary", DomainKind::Boundary);
    bad_boundary.complementing_condition_satisfied = false;
    bad_summaries.boundary_symbols.push_back(bad_boundary);
    ParameterScaleSummary bad_penalty;
    bad_penalty.max_scale_value = 0.25;
    bad_summaries.parameter_scales.push_back(bad_penalty);

    auto bad_report = analyzeWithSummaries(std::move(bad_summaries));
    const auto* bad_complementing = firstClaimFrom(
        bad_report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(bad_complementing, nullptr);
    EXPECT_EQ(bad_complementing->status, PropertyStatus::Violated);
    const auto* bad_coercivity = firstClaimFrom(
        bad_report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(bad_coercivity, nullptr);
    EXPECT_EQ(bad_coercivity->status, PropertyStatus::Violated);
}

TEST(Phase6ExtendedAnalyzers, TemporalStabilityCertifiedViolatedAndMissing)
{
    AnalysisSummarySet good_summaries;
    TemporalStabilitySummary good;
    good.time_scheme = "stable-scheme";
    good.stability_class = TemporalStabilityClass::AStable;
    good.cfl_estimate = 2.0;
    good.amplification_radius = 1.0;
    good_summaries.temporal_stability.push_back(good);

    auto good_report = analyzeWithSummaries(std::move(good_summaries));
    const auto* good_claim = firstClaimFrom(
        good_report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_NE(good_claim, nullptr);
    EXPECT_EQ(good_claim->status, PropertyStatus::Preserved);

    AnalysisSummarySet bad_summaries;
    TemporalStabilitySummary bad;
    bad.time_scheme = "conditional-scheme";
    bad.stability_class = TemporalStabilityClass::ConditionallyStable;
    bad.cfl_estimate = 1.5;
    bad_summaries.temporal_stability.push_back(bad);

    auto bad_report = analyzeWithSummaries(std::move(bad_summaries));
    const auto* bad_claim = firstClaimFrom(
        bad_report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_NE(bad_claim, nullptr);
    EXPECT_EQ(bad_claim->status, PropertyStatus::Violated);

    auto missing_report = ProblemAnalyzer::createDefault().analyze(
        ProblemAnalysisContext{});
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::TemporalStability,
                              "TemporalStabilityAnalyzer"));
}

TEST(Phase6ExtendedAnalyzers, EnergyEntropyLawCertifiedViolatedAndMissing)
{
    AnalysisSummarySet summaries;
    EnergyEntropySummary energy;
    energy.energy_entropy_id = "energy-law";
    energy.law_kind = EnergyEntropyLawKind::Energy;
    energy.expected_production_sign = BalanceSignClass::Nonpositive;
    energy.balance_tolerance = 1.0e-6;
    energy.observed_discrete_balance = 0.0;
    energy.observed_production = -1.0e-8;
    summaries.energy_entropy.push_back(energy);
    EnergyEntropySummary entropy;
    entropy.energy_entropy_id = "entropy-law";
    entropy.law_kind = EnergyEntropyLawKind::Entropy;
    entropy.expected_production_sign = BalanceSignClass::Nonnegative;
    entropy.balance_tolerance = 1.0e-6;
    entropy.observed_discrete_balance = 0.0;
    entropy.observed_production = -1.0e-3;
    summaries.energy_entropy.push_back(entropy);

    auto report = analyzeWithSummaries(std::move(summaries));
    const auto* energy_claim = firstClaimFrom(
        report, PropertyKind::EnergyStability,
        "EnergyEntropyLawAnalyzer");
    ASSERT_NE(energy_claim, nullptr);
    EXPECT_EQ(energy_claim->status, PropertyStatus::Preserved);
    const auto* entropy_claim = firstClaimFrom(
        report, PropertyKind::EntropyStability,
        "EnergyEntropyLawAnalyzer");
    ASSERT_NE(entropy_claim, nullptr);
    EXPECT_EQ(entropy_claim->status, PropertyStatus::Violated);

    auto missing_report = ProblemAnalyzer::createDefault().analyze(
        ProblemAnalysisContext{});
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::EnergyStability,
                              "EnergyEntropyLawAnalyzer"));
}

TEST(Phase6ExtendedAnalyzers, CoefficientConstitutiveCertifiedViolatedAndMissing)
{
    AnalysisSummarySet summaries;
    CoefficientPropertySummary positive;
    positive.coefficient = "positive-coefficient";
    positive.positivity = PositivityClass::Positive;
    positive.anisotropy_ratio = 2.0;
    positive.contrast_ratio = 3.0;
    summaries.coefficient_properties.push_back(positive);
    CoefficientPropertySummary indefinite;
    indefinite.coefficient = "indefinite-coefficient";
    indefinite.positivity = PositivityClass::Indefinite;
    indefinite.contrast_ratio = 1.0e7;
    summaries.coefficient_properties.push_back(indefinite);

    auto report = analyzeWithSummaries(std::move(summaries));
    const auto claims = claimsFrom(report, PropertyKind::CoefficientPositivity,
                                   "CoefficientConstitutiveAnalyzer");
    ASSERT_GE(claims.size(), 2u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Preserved);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Violated);
    const auto* robustness = firstClaimFrom(
        report, PropertyKind::ParameterRobustness,
        "CoefficientConstitutiveAnalyzer");
    ASSERT_NE(robustness, nullptr);

    auto missing_report = ProblemAnalyzer::createDefault().analyze(
        ProblemAnalysisContext{});
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::CoefficientPositivity,
                              "CoefficientConstitutiveAnalyzer"));
}

TEST(Phase6ExtendedAnalyzers, NonlinearTangentCertifiedViolatedAndUnknown)
{
    AnalysisSummarySet summaries;
    NonlinearTangentSummary exact;
    exact.residual_id = "exact";
    exact.block = scalarBlock("exact");
    exact.tangent_consistency = TangentConsistencyClass::Exact;
    exact.jacobian_action_available = true;
    exact.finite_difference_action_error = 1.0e-12;
    exact.finite_difference_tolerance = 1.0e-8;
    summaries.nonlinear_tangents.push_back(exact);
    NonlinearTangentSummary bad;
    bad.residual_id = "bad";
    bad.block = scalarBlock("bad");
    bad.tangent_consistency = TangentConsistencyClass::Inconsistent;
    bad.jacobian_action_available = true;
    bad.finite_difference_action_error = 1.0e-3;
    bad.finite_difference_tolerance = 1.0e-8;
    summaries.nonlinear_tangents.push_back(bad);
    NonlinearTangentSummary unknown;
    unknown.residual_id = "unknown";
    unknown.tangent_consistency = TangentConsistencyClass::Unknown;
    summaries.nonlinear_tangents.push_back(unknown);

    auto report = analyzeWithSummaries(std::move(summaries));
    const auto claims = claimsFrom(report, PropertyKind::NonlinearTangentStructure,
                                   "NonlinearTangentAnalyzer");
    ASSERT_EQ(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Preserved);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Violated);
    EXPECT_EQ(claims[2]->status, PropertyStatus::Unknown);
}

TEST(Phase6ExtendedAnalyzers, LockingRiskUsesConstraintInfSupAndSpaceEvidence)
{
    AnalysisSummarySet good_summaries;
    InfSupEstimateSummary infsup;
    infsup.primal_variable = VariableKey::field(0);
    infsup.multiplier_variable = VariableKey::field(1);
    infsup.estimate_value = 0.2;
    good_summaries.inf_sup_estimates.push_back(infsup);
    auto good_report = analyzeWithSummaries(std::move(good_summaries));
    const auto* good_claim = firstClaimFrom(
        good_report, PropertyKind::LockingRisk,
        "LockingRiskAnalyzer");
    ASSERT_NE(good_claim, nullptr);
    EXPECT_EQ(good_claim->status, PropertyStatus::Preserved);

    ProblemAnalysisContext bad_ctx;
    ConstraintAnalysisSummary constraints;
    ConstrainedDofSet constrained;
    constrained.field = 0;
    constrained.num_constrained_dofs = 10;
    constrained.num_total_dofs = 10;
    constrained.constrained_fraction = 1.0;
    constraints.constrained_sets.push_back(constrained);
    bad_ctx.setConstraintSummary(std::move(constraints));
    auto bad_report = ProblemAnalyzer::createDefault().analyze(bad_ctx);
    const auto* bad_claim = firstClaimFrom(
        bad_report, PropertyKind::LockingRisk,
        "LockingRiskAnalyzer");
    ASSERT_NE(bad_claim, nullptr);
    EXPECT_EQ(bad_claim->status, PropertyStatus::Violated);

    auto missing_report = ProblemAnalyzer::createDefault().analyze(
        ProblemAnalysisContext{});
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::LockingRisk,
                              "LockingRiskAnalyzer"));
}

TEST(Phase6ExtendedAnalyzers, SpectralErrorEstimatorQuadratureAndCoupledSummaries)
{
    AnalysisSummarySet summaries;
    SpectralStructureSummary spectral_ok;
    spectral_ok.block = scalarBlock("spectral-ok");
    spectral_ok.eigenproblem_declared = true;
    spectral_ok.self_adjoint_evidence = true;
    spectral_ok.compactness_evidence = true;
    summaries.spectral_structures.push_back(spectral_ok);
    SpectralStructureSummary spectral_bad;
    spectral_bad.block = scalarBlock("spectral-bad");
    spectral_bad.eigenproblem_declared = true;
    spectral_bad.spurious_mode_count = 2;
    summaries.spectral_structures.push_back(spectral_bad);

    ErrorEstimatorSummary estimator_ok;
    estimator_ok.estimator_id = "estimator-ok";
    estimator_ok.block = scalarBlock("estimator-ok");
    estimator_ok.residual_metadata_present = true;
    estimator_ok.jump_metadata_present = true;
    summaries.error_estimators.push_back(estimator_ok);
    ErrorEstimatorSummary estimator_bad;
    estimator_bad.estimator_id = "estimator-bad";
    estimator_bad.block = scalarBlock("estimator-bad");
    estimator_bad.missing_required_metadata_count = 1;
    summaries.error_estimators.push_back(estimator_bad);

    QuadratureAdequacySummary quadrature_ok;
    quadrature_ok.block = scalarBlock("quadrature-ok");
    quadrature_ok.integrand_polynomial_degree = 2;
    quadrature_ok.quadrature_exact_degree = 3;
    summaries.quadrature_adequacy.push_back(quadrature_ok);
    QuadratureAdequacySummary quadrature_bad;
    quadrature_bad.block = scalarBlock("quadrature-bad");
    quadrature_bad.integrand_polynomial_degree = 4;
    quadrature_bad.quadrature_exact_degree = 1;
    quadrature_bad.zero_energy_mode_count = 1;
    summaries.quadrature_adequacy.push_back(quadrature_bad);

    CoupledSystemStabilitySummary coupled_ok;
    coupled_ok.coupling_group = "coupled-ok";
    coupled_ok.variables = {VariableKey::field(0), VariableKey::field(1)};
    coupled_ok.monolithic_coupling = true;
    coupled_ok.coupling_tolerance = 1.0e-6;
    summaries.coupled_system_stability.push_back(coupled_ok);
    CoupledSystemStabilitySummary coupled_bad;
    coupled_bad.coupling_group = "coupled-bad";
    coupled_bad.variables = {VariableKey::field(0), VariableKey::field(1)};
    coupled_bad.partitioned_coupling = true;
    coupled_bad.coupling_tolerance = 1.0e-6;
    coupled_bad.partition_iteration_spectral_radius = 1.2;
    coupled_bad.exchange_residual = 1.0e-4;
    summaries.coupled_system_stability.push_back(coupled_bad);

    auto report = analyzeWithSummaries(std::move(summaries));
    const auto spectral_claims = claimsFrom(
        report, PropertyKind::SpectralCorrectness,
        "SpectralSpuriousModeAnalyzer");
    ASSERT_EQ(spectral_claims.size(), 2u);
    EXPECT_EQ(spectral_claims[0]->status, PropertyStatus::Preserved);
    EXPECT_EQ(spectral_claims[1]->status, PropertyStatus::Violated);

    const auto estimator_claims = claimsFrom(
        report, PropertyKind::ErrorEstimatorEligibility,
        "ErrorEstimatorAnalyzer");
    ASSERT_EQ(estimator_claims.size(), 2u);
    EXPECT_EQ(estimator_claims[0]->status, PropertyStatus::Preserved);
    EXPECT_EQ(estimator_claims[1]->status, PropertyStatus::Violated);

    const auto quadrature_claims = claimsFrom(
        report, PropertyKind::QuadratureAdequacy,
        "QuadratureAdequacyAnalyzer");
    ASSERT_EQ(quadrature_claims.size(), 2u);
    EXPECT_EQ(quadrature_claims[0]->status, PropertyStatus::Preserved);
    EXPECT_EQ(quadrature_claims[1]->status, PropertyStatus::Violated);

    const auto coupled_claims = claimsFrom(
        report, PropertyKind::CoupledSystemStructure,
        "CoupledSystemStabilityAnalyzer");
    ASSERT_EQ(coupled_claims.size(), 2u);
    EXPECT_EQ(coupled_claims[0]->status, PropertyStatus::Preserved);
    EXPECT_EQ(coupled_claims[1]->status, PropertyStatus::Violated);

    auto missing_report = ProblemAnalyzer::createDefault().analyze(
        ProblemAnalysisContext{});
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::SpectralCorrectness,
                              "SpectralSpuriousModeAnalyzer"));
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::ErrorEstimatorEligibility,
                              "ErrorEstimatorAnalyzer"));
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::QuadratureAdequacy,
                              "QuadratureAdequacyAnalyzer"));
    EXPECT_FALSE(hasClaimFrom(missing_report,
                              PropertyKind::CoupledSystemStructure,
                              "CoupledSystemStabilityAnalyzer"));
}
