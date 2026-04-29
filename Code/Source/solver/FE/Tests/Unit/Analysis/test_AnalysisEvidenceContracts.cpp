/**
 * @file test_AnalysisEvidenceContracts.cpp
 * @brief Tests that certification claims require theorem-specific evidence.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "Analysis/AdvancedStabilityAnalyzers.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ConstraintRankAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/CouplingGraphAnalyzer.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Backends/Utils/BackendOptions.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

namespace {

FieldDescriptor h1Field(FieldId id,
                        int order,
                        FieldType type,
                        int value_dimension,
                        std::string name)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = type;
    fd.value_dimension = value_dimension;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::H1;
    fd.trace_capabilities = TraceCapabilityFlags::Value |
                            TraceCapabilityFlags::NormalFlux;
    return fd;
}

FieldDescriptor hDivField(FieldId id, int order, std::string name)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 3;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::HDiv;
    fd.trace_capabilities = TraceCapabilityFlags::NormalComponent |
                            TraceCapabilityFlags::NormalFlux;
    fd.component_extractable = false;
    fd.has_exact_sequence_structure = true;
    fd.supports_local_balance_closure = true;
    return fd;
}

OperatorBlockId scalarBlock(std::string tag,
                            DomainKind domain = DomainKind::Cell)
{
    OperatorBlockId block;
    block.test_variables = {VariableKey::field(0)};
    block.trial_variables = {VariableKey::field(0)};
    block.operator_tag = std::move(tag);
    block.domain = domain;
    return block;
}

ProblemAnalysisReport analyze(ProblemAnalysisContext ctx)
{
    return ProblemAnalyzer::createDefault().analyze(ctx);
}

const PropertyClaim* firstFrom(const ProblemAnalysisReport& report,
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

const PropertyClaim* firstForBlock(const ProblemAnalysisReport& report,
                                   PropertyKind kind,
                                   const std::string& origin,
                                   const std::string& block_id)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == kind &&
            claim.claim_origin == origin &&
            claim.tested_block_id &&
            *claim.tested_block_id == block_id) {
            return &claim;
        }
    }
    return nullptr;
}

const PropertyClaim* firstForCoefficient(const ProblemAnalysisReport& report,
                                         const std::string& origin,
                                         const std::string& coefficient_id)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::CoefficientPositivity &&
            claim.claim_origin == origin &&
            claim.coefficient_id &&
            *claim.coefficient_id == coefficient_id) {
            return &claim;
        }
    }
    return nullptr;
}

bool hasCertifiedClaim(const ProblemAnalysisReport& report, PropertyKind kind)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == kind &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(AnalysisEvidenceContracts, GenericH1OrderDifferenceDoesNotCertifyInfSup)
{
    const auto velocity = VariableKey::field(1);
    const auto pressure = VariableKey::field(0);

    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(h1Field(1, 2, FieldType::Vector, 3, "velocity"));
    ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "pressure"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        velocity, "velocity-block", "evidence-contract"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        velocity, pressure, "stokes-pair", "divergence-pair",
        "evidence-contract"));

    const auto report = analyze(std::move(ctx));
    EXPECT_FALSE(hasCertifiedClaim(report, PropertyKind::InfSupCondition));

    bool saw_structural_heuristic = false;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.inf_sup_class == InfSupClass::StructurallySupported) {
            saw_structural_heuristic = true;
            EXPECT_EQ(claim.status, PropertyStatus::Likely);
            ASSERT_TRUE(claim.certification_class.has_value());
            EXPECT_EQ(*claim.certification_class,
                      CertificationClass::NotCertified);
        }
    }
    EXPECT_TRUE(saw_structural_heuristic);
}

TEST(AnalysisEvidenceContracts, StablePairMetadataCertifiesInfSup)
{
    const auto velocity = VariableKey::field(1);
    const auto pressure = VariableKey::field(0);

    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(h1Field(1, 2, FieldType::Vector, 3, "velocity"));
    ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "pressure"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        velocity, "velocity-block", "evidence-contract"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        velocity, pressure, "stokes-pair", "divergence-pair",
        "evidence-contract"));

    AnalysisSummarySet summaries;
    InfSupPairCertificationSummary stable;
    stable.block.operator_tag = "divergence-pair";
    stable.block.test_variables = {velocity, pressure};
    stable.block.trial_variables = {velocity, pressure};
    stable.primal_variable = velocity;
    stable.multiplier_variable = pressure;
    stable.pair_family = "Taylor-Hood P2/P1";
    stable.primal_polynomial_order = 2;
    stable.multiplier_polynomial_order = 1;
    stable.known_stable_pair = true;
    stable.mesh_assumption_evidence_present = true;
    stable.domain_assumption_evidence_present = true;
    stable.boundary_condition_scope_present = true;
    stable.beta_lower_bound_present = true;
    stable.beta_lower_bound = 0.25;
    stable.inf_sup_theorem_id = "Brezzi-Fortin Taylor-Hood stable-pair theorem";
    summaries.inf_sup_pair_certifications.push_back(stable);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    bool saw_certified_pair = false;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.inf_sup_class == InfSupClass::StructurallySupported &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified) {
            saw_certified_pair = true;
            EXPECT_EQ(claim.status, PropertyStatus::Preserved);
        }
    }
    EXPECT_TRUE(saw_certified_pair);
}

TEST(AnalysisEvidenceContracts, WeakBoundaryCoercivityRequiresLowerBound)
{
    AnalysisSummarySet missing_bound_summaries;
    BoundarySymbolSummary boundary;
    boundary.block = scalarBlock("nitsche-face", DomainKind::Boundary);
    boundary.complementing_condition_satisfied = true;
    missing_bound_summaries.boundary_symbols.push_back(boundary);

    ParameterScaleSummary penalty;
    penalty.role = ParameterScaleRole::WeakBoundaryPenalty;
    penalty.block = boundary.block;
    penalty.max_scale_value = 10.0;
    missing_bound_summaries.parameter_scales.push_back(penalty);

    ProblemAnalysisContext missing_ctx;
    missing_ctx.setAnalysisSummaries(std::move(missing_bound_summaries));
    const auto missing_report = analyze(std::move(missing_ctx));
    const auto* missing = firstFrom(
        missing_report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(missing, nullptr);
    EXPECT_EQ(missing->status, PropertyStatus::Unknown);
    ASSERT_TRUE(missing->certification_class.has_value());
    EXPECT_EQ(*missing->certification_class, CertificationClass::NotCertified);

    AnalysisSummarySet missing_trace_summaries;
    missing_trace_summaries.boundary_symbols.push_back(boundary);
    penalty.required_lower_bound_present = true;
    penalty.required_lower_bound = 4.0;
    missing_trace_summaries.parameter_scales.push_back(penalty);

    ProblemAnalysisContext missing_trace_ctx;
    missing_trace_ctx.setAnalysisSummaries(std::move(missing_trace_summaries));
    const auto missing_trace_report = analyze(std::move(missing_trace_ctx));
    const auto* missing_trace = firstFrom(
        missing_trace_report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(missing_trace, nullptr);
    EXPECT_EQ(missing_trace->status, PropertyStatus::Unknown);
    ASSERT_TRUE(missing_trace->certification_class.has_value());
    EXPECT_EQ(*missing_trace->certification_class,
              CertificationClass::NotCertified);

    AnalysisSummarySet certified_summaries;
    certified_summaries.boundary_symbols.push_back(boundary);
    penalty.trace_inverse_metadata_present = true;
    certified_summaries.parameter_scales.push_back(penalty);

    ProblemAnalysisContext certified_ctx;
    certified_ctx.setAnalysisSummaries(std::move(certified_summaries));
    const auto certified_report = analyze(std::move(certified_ctx));
    const auto* certified = firstFrom(
        certified_report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(certified, nullptr);
    EXPECT_EQ(certified->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified->certification_class.has_value());
    EXPECT_EQ(*certified->certification_class, CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, TransportCflRequiresExplicitEstimatePresence)
{
    const auto u = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addContribution(ContributionDescriptor::transportLike(
        u, "transport", "evidence-contract"));

    AnalysisSummarySet summaries;
    TemporalStabilitySummary temporal;
    temporal.block = scalarBlock("transport");
    temporal.cfl_estimate = 8.0;
    temporal.cfl_estimate_present = false;
    summaries.temporal_stability.push_back(temporal);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    const auto* transport = firstFrom(
        report, PropertyKind::OperatorTransportCharacter,
        "TransportCharacterAnalyzer");
    ASSERT_NE(transport, nullptr);
    EXPECT_FALSE(transport->cfl_number.has_value());
}

TEST(AnalysisEvidenceContracts, StabilizationPresenceIsSeparateFromAdequacy)
{
    const auto field = VariableKey::field(0);
    ProblemAnalysisContext presence_ctx;
    presence_ctx.addContribution(ContributionDescriptor::stabilization(
        field, "supg-term", "evidence-contract"));

    const auto presence_report = analyze(std::move(presence_ctx));
    const auto* presence = firstFrom(
        presence_report, PropertyKind::Stabilization,
        "StabilizationAnalyzer");
    ASSERT_NE(presence, nullptr);
    EXPECT_EQ(presence->status, PropertyStatus::Exact);

    AnalysisSummarySet summaries;
    StabilizationAdequacySummary adequacy;
    adequacy.stabilization_id = "supg-term";
    adequacy.method_family = "SUPG";
    adequacy.block = scalarBlock("supg-term");
    adequacy.variables = {field};
    adequacy.parameter_formula_metadata_present = true;
    adequacy.residual_consistency_evidence_present = true;
    adequacy.regime_metadata_present = true;
    adequacy.peclet_condition_satisfied = true;
    adequacy.cfl_condition_satisfied = true;
    summaries.stabilization_adequacy.push_back(adequacy);

    ProblemAnalysisContext adequate_ctx;
    adequate_ctx.addContribution(ContributionDescriptor::stabilization(
        field, "supg-term", "evidence-contract"));
    adequate_ctx.setAnalysisSummaries(std::move(summaries));
    const auto adequate_report = analyze(std::move(adequate_ctx));
    bool saw_certified_adequacy = false;
    for (const auto& claim : adequate_report.claims) {
        if (claim.kind == PropertyKind::Stabilization &&
            claim.status == PropertyStatus::Preserved &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified) {
            saw_certified_adequacy = true;
        }
    }
    EXPECT_TRUE(saw_certified_adequacy);
}

TEST(AnalysisEvidenceContracts, TemporalStabilityNeedsNormEvidence)
{
    AnalysisSummarySet modal_only_summaries;
    TemporalStabilitySummary modal;
    modal.time_scheme = "modal-bound";
    modal.stability_class = TemporalStabilityClass::AStable;
    modal.amplification_radius = 0.9;
    modal.amplification_radius_present = true;
    modal.stability_metadata_present = true;
    modal.scalar_modal_bound_only = true;
    modal_only_summaries.temporal_stability.push_back(modal);

    ProblemAnalysisContext modal_ctx;
    modal_ctx.setAnalysisSummaries(std::move(modal_only_summaries));
    const auto modal_report = analyze(std::move(modal_ctx));
    const auto* modal_claim = firstFrom(
        modal_report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_NE(modal_claim, nullptr);
    EXPECT_EQ(modal_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(modal_claim->certification_class.has_value());
    EXPECT_EQ(*modal_claim->certification_class,
              CertificationClass::NotCertified);

    AnalysisSummarySet normal_summaries;
    modal.scalar_modal_bound_only = false;
    modal.operator_normality_evidence_present = true;
    modal.stability_theorem_id = "Dahlquist A-stability region";
    modal.stability_region_evidence_present = true;
    modal.operator_spectrum_coverage_present = true;
    normal_summaries.temporal_stability.push_back(modal);

    ProblemAnalysisContext normal_ctx;
    normal_ctx.setAnalysisSummaries(std::move(normal_summaries));
    const auto normal_report = analyze(std::move(normal_ctx));
    const auto* normal_claim = firstFrom(
        normal_report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_NE(normal_claim, nullptr);
    EXPECT_EQ(normal_claim->status, PropertyStatus::Preserved);
}

TEST(AnalysisEvidenceContracts, EntropyStabilityRequiresEntropyVariableMetadata)
{
    AnalysisSummarySet summaries;
    EnergyEntropySummary missing;
    missing.energy_entropy_id = "entropy-missing-metadata";
    missing.law_kind = EnergyEntropyLawKind::Entropy;
    missing.expected_production_sign = BalanceSignClass::Nonnegative;
    missing.balance_tolerance = 1.0e-8;
    missing.observed_discrete_balance = 0.0;
    missing.observed_production = 0.0;
    summaries.energy_entropy.push_back(missing);

    EnergyEntropySummary certified = missing;
    certified.energy_entropy_id = "entropy-certified";
    certified.convex_entropy_metadata_present = true;
    certified.entropy_variables_metadata_present = true;
    certified.entropy_flux_metadata_present = true;
    certified.entropy_dissipation_metadata_present = true;
    certified.boundary_source_entropy_metadata_present = true;
    certified.energy_entropy_theorem_id = "Tadmor entropy-stability theorem";
    summaries.energy_entropy.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto entropy = claimsFrom(
        report, PropertyKind::EntropyStability,
        "EnergyEntropyLawAnalyzer");
    ASSERT_EQ(entropy.size(), 2u);
    EXPECT_EQ(entropy[0]->status, PropertyStatus::Unknown);
    ASSERT_TRUE(entropy[0]->certification_class.has_value());
    EXPECT_EQ(*entropy[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(entropy[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(entropy[1]->certification_class.has_value());
    EXPECT_EQ(*entropy[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, DAERankEvidenceCertifiesIndexOne)
{
    const auto u = VariableKey::field(0);
    const auto lambda = VariableKey::field(1);

    ProblemAnalysisContext ctx;
    VariableDescriptor u_desc;
    u_desc.key = u;
    u_desc.temporal_state_kind = TemporalStateKind::Dynamic;
    u_desc.max_time_derivative_order = 1;
    ctx.addVariableDescriptor(u_desc);
    VariableDescriptor lambda_desc;
    lambda_desc.key = lambda;
    lambda_desc.temporal_state_kind = TemporalStateKind::Algebraic;
    lambda_desc.participates_in_constraint_blocks = true;
    ctx.addVariableDescriptor(lambda_desc);

    AnalysisSummarySet summaries;
    DAEStructureEvidenceSummary evidence;
    evidence.system_id = "semi-explicit-index-one";
    evidence.variables = {u, lambda};
    evidence.dae_form_class = DAEFormClass::SemiExplicit;
    evidence.mass_matrix_rank_metadata_present = true;
    evidence.algebraic_jacobian_rank_metadata_present = true;
    evidence.algebraic_jacobian_full_rank = true;
    evidence.hidden_constraint_metadata_present = true;
    evidence.consistent_initial_condition_evidence_present = true;
    evidence.initial_constraint_residual = 1.0e-12;
    evidence.residual_tolerance = 1.0e-10;
    summaries.dae_structure_evidence.push_back(evidence);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    const auto* dae = firstFrom(
        report, PropertyKind::DifferentialAlgebraicStructure,
        "DAEStructureAnalyzer");
    ASSERT_NE(dae, nullptr);
    EXPECT_EQ(dae->status, PropertyStatus::Preserved);
    ASSERT_TRUE(dae->certification_class.has_value());
    EXPECT_EQ(*dae->certification_class, CertificationClass::Certified);
    ASSERT_TRUE(dae->dae_class.has_value());
    EXPECT_EQ(*dae->dae_class, DAEClass::Index1DAELike);
}

TEST(AnalysisEvidenceContracts, SpectralAndQuadratureNeedScopeEvidence)
{
    AnalysisSummarySet summaries;
    SpectralStructureSummary spectral_missing;
    spectral_missing.block = scalarBlock("eigenproblem");
    spectral_missing.eigenproblem_declared = true;
    spectral_missing.self_adjoint_evidence = true;
    spectral_missing.compactness_evidence = true;
    summaries.spectral_structures.push_back(spectral_missing);

    SpectralStructureSummary spectral_certified = spectral_missing;
    spectral_certified.block.operator_tag = "eigenproblem-certified";
    spectral_certified.operator_convergence_evidence = true;
    spectral_certified.spectral_convergence_theorem_id =
        "Boffi spectral approximation theorem";
    summaries.spectral_structures.push_back(spectral_certified);

    QuadratureAdequacySummary quadrature_missing;
    quadrature_missing.block = scalarBlock("quad-missing");
    quadrature_missing.integrand_polynomial_degree = 2;
    quadrature_missing.quadrature_exact_degree = 3;
    summaries.quadrature_adequacy.push_back(quadrature_missing);

    QuadratureAdequacySummary quadrature_certified = quadrature_missing;
    quadrature_certified.block.operator_tag = "quad-certified";
    quadrature_certified.affine_mapping_evidence_present = true;
    quadrature_certified.polynomial_integrand_metadata_complete = true;
    quadrature_certified.coefficient_degree_metadata_present = true;
    summaries.quadrature_adequacy.push_back(quadrature_certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto spectral = claimsFrom(
        report, PropertyKind::SpectralCorrectness,
        "SpectralSpuriousModeAnalyzer");
    ASSERT_EQ(spectral.size(), 2u);
    EXPECT_EQ(spectral[0]->status, PropertyStatus::Unknown);
    EXPECT_EQ(spectral[1]->status, PropertyStatus::Preserved);

    const auto quadrature = claimsFrom(
        report, PropertyKind::QuadratureAdequacy,
        "QuadratureAdequacyAnalyzer");
    ASSERT_EQ(quadrature.size(), 2u);
    EXPECT_EQ(quadrature[0]->status, PropertyStatus::Likely);
    EXPECT_EQ(quadrature[1]->status, PropertyStatus::Preserved);
}

TEST(AnalysisEvidenceContracts, CoupledStabilityNeedsContractionEvidence)
{
    const auto a = VariableKey::field(0);
    const auto b = VariableKey::field(1);

    AnalysisSummarySet summaries;
    CoupledSystemStabilitySummary spectral_only;
    spectral_only.coupling_group = "spectral-only";
    spectral_only.variables = {a, b};
    spectral_only.partitioned_coupling = true;
    spectral_only.coupling_tolerance = 1.0e-8;
    spectral_only.partition_iteration_spectral_radius = 0.5;
    spectral_only.partition_iteration_spectral_radius_present = true;
    summaries.coupled_system_stability.push_back(spectral_only);

    CoupledSystemStabilitySummary certified = spectral_only;
    certified.coupling_group = "contractive";
    certified.coupling_tolerance_present = true;
    certified.exchange_residual_present = true;
    certified.constraint_drift_present = true;
    certified.linear_stationary_iteration_evidence_present = true;
    certified.contraction_norm_evidence_present = true;
    certified.coupled_operator_stability_evidence_present = true;
    certified.relaxation_metadata_present = true;
    certified.added_mass_risk_assessed = true;
    summaries.coupled_system_stability.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto coupled = claimsFrom(
        report, PropertyKind::CoupledSystemStructure,
        "CoupledSystemStabilityAnalyzer");
    ASSERT_EQ(coupled.size(), 2u);
    EXPECT_EQ(coupled[0]->status, PropertyStatus::Unknown);
    ASSERT_TRUE(coupled[0]->certification_class.has_value());
    EXPECT_EQ(*coupled[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(coupled[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(coupled[1]->certification_class.has_value());
    EXPECT_EQ(*coupled[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, CoupledEnergyBalanceNeedsCoerciveOperatorEvidence)
{
    const auto a = VariableKey::field(0);
    const auto b = VariableKey::field(1);

    AnalysisSummarySet summaries;
    CoupledSystemStabilitySummary energy_only;
    energy_only.coupling_group = "energy-only";
    energy_only.variables = {a, b};
    energy_only.monolithic_coupling = true;
    energy_only.coupling_tolerance = 1.0e-8;
    energy_only.coupling_tolerance_present = true;
    energy_only.exchange_residual_present = true;
    energy_only.constraint_drift_present = true;
    energy_only.interface_energy_balance_evidence_present = true;
    summaries.coupled_system_stability.push_back(energy_only);

    CoupledSystemStabilitySummary certified = energy_only;
    certified.coupling_group = "energy-coercive";
    certified.coupled_norm_coercivity_evidence_present = true;
    certified.coupled_operator_stability_evidence_present = true;
    summaries.coupled_system_stability.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto coupled = claimsFrom(
        report, PropertyKind::CoupledSystemStructure,
        "CoupledSystemStabilityAnalyzer");
    ASSERT_EQ(coupled.size(), 2u);
    EXPECT_EQ(coupled[0]->status, PropertyStatus::Likely);
    ASSERT_TRUE(coupled[0]->certification_class.has_value());
    EXPECT_EQ(*coupled[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(coupled[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(coupled[1]->certification_class.has_value());
    EXPECT_EQ(*coupled[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, DiscreteMatrixCertificationRequiresCompleteSignAndRowEvidence)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext incomplete_ctx;
    incomplete_ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    incomplete_ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "diffusion", "evidence-contract"));

    auto matrix_block = scalarBlock("diffusion");
    AnalysisSummarySet incomplete_summaries;
    CoefficientPropertySummary coeff;
    coeff.block = matrix_block;
    coeff.positivity = PositivityClass::Positive;
    coeff.coefficient_region_coverage_complete = true;
    coeff.quadrature_point_coverage_complete = true;
    coeff.lower_bound_valid_for_all_samples = true;
    coeff.tolerance_metadata_present = true;
    incomplete_summaries.coefficient_properties.push_back(coeff);

    DiscreteMatrixSummary matrix;
    matrix.block = matrix_block;
    matrix.rows = 2;
    matrix.cols = 2;
    matrix.square = true;
    matrix.sign_tolerance = 1.0e-12;
    matrix.row_sum_tolerance = 1.0e-12;
    matrix.diagonal_count = 2;
    matrix.offdiag_count = 2;
    matrix.negative_offdiag_count = 2;
    matrix.min_row_sum = 0.0;
    matrix.max_row_sum = 1.0;
    matrix.m_matrix_certification_evidence = true;
    matrix.stieltjes_matrix_evidence = true;
    matrix.dmp_applicability_evidence = true;
    matrix.dmp_rhs_sign_evidence = true;
    incomplete_summaries.discrete_matrices.push_back(matrix);
    incomplete_ctx.setAnalysisSummaries(std::move(incomplete_summaries));

    const auto incomplete_report = analyze(std::move(incomplete_ctx));
    const auto z_claims = claimsFrom(
        incomplete_report, PropertyKind::ZMatrixStructure,
        "DiscreteMonotonicityAnalyzer");
    ASSERT_FALSE(z_claims.empty());
    EXPECT_EQ(z_claims.front()->status, PropertyStatus::Unknown);
    EXPECT_FALSE(hasCertifiedClaim(incomplete_report,
                                   PropertyKind::DiscreteMaximumPrinciple));

    ProblemAnalysisContext certified_ctx;
    certified_ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    certified_ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "diffusion", "evidence-contract"));
    AnalysisSummarySet certified_summaries;
    certified_summaries.coefficient_properties.push_back(coeff);
    matrix.sign_evidence_complete = true;
    matrix.row_sum_evidence_complete = true;
    matrix.scanned_row_count = 2;
    matrix.expected_row_count = 2;
    matrix.scanned_entry_count = 4;
    certified_summaries.discrete_matrices.push_back(matrix);
    certified_ctx.setAnalysisSummaries(std::move(certified_summaries));

    const auto certified_report = analyze(std::move(certified_ctx));
    EXPECT_TRUE(hasCertifiedClaim(certified_report,
                                  PropertyKind::ZMatrixStructure));
    EXPECT_TRUE(hasCertifiedClaim(certified_report,
                                  PropertyKind::DiscreteMaximumPrinciple));
}

TEST(AnalysisEvidenceContracts, GenericMMatrixBooleanNeedsTheoremSpecificEvidence)
{
    const auto scalar = VariableKey::field(0);
    DiscreteMatrixSummary matrix;
    matrix.block = scalarBlock("diffusion");
    matrix.rows = 2;
    matrix.cols = 2;
    matrix.square = true;
    matrix.sign_evidence_complete = true;
    matrix.row_sum_evidence_complete = true;
    matrix.sign_tolerance = 1.0e-12;
    matrix.row_sum_tolerance = 1.0e-12;
    matrix.diagonal_count = 2;
    matrix.offdiag_count = 2;
    matrix.negative_offdiag_count = 2;
    matrix.min_row_sum = 0.0;
    matrix.max_row_sum = 1.0;
    matrix.scanned_row_count = 2;
    matrix.expected_row_count = 2;
    matrix.scanned_entry_count = 4;
    matrix.m_matrix_certification_evidence = true;

    ProblemAnalysisContext generic_ctx;
    generic_ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    generic_ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "diffusion", "evidence-contract"));
    AnalysisSummarySet generic_summaries;
    generic_summaries.discrete_matrices.push_back(matrix);
    generic_ctx.setAnalysisSummaries(std::move(generic_summaries));

    const auto generic_report = analyze(std::move(generic_ctx));
    const auto generic_mmatrix = claimsFrom(
        generic_report, PropertyKind::MMatrixStructure,
        "DiscreteMonotonicityAnalyzer");
    ASSERT_FALSE(generic_mmatrix.empty());
    EXPECT_EQ(generic_mmatrix.front()->status, PropertyStatus::Unknown);
    EXPECT_FALSE(hasCertifiedClaim(generic_report,
                                   PropertyKind::MMatrixStructure));

    ProblemAnalysisContext theorem_ctx;
    theorem_ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    theorem_ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "diffusion", "evidence-contract"));
    AnalysisSummarySet theorem_summaries;
    matrix.stieltjes_matrix_evidence = true;
    theorem_summaries.discrete_matrices.push_back(matrix);
    theorem_ctx.setAnalysisSummaries(std::move(theorem_summaries));

    const auto theorem_report = analyze(std::move(theorem_ctx));
    EXPECT_TRUE(hasCertifiedClaim(theorem_report,
                                  PropertyKind::MMatrixStructure));
}

TEST(AnalysisEvidenceContracts, NumericInfSupRequiresUniformFamilyEvidence)
{
    const auto velocity = VariableKey::field(1);
    const auto pressure = VariableKey::field(0);

    AnalysisSummarySet summaries;
    InfSupEstimateSummary single_mesh;
    single_mesh.block.operator_tag = "mixed-pair";
    single_mesh.block.test_variables = {velocity, pressure};
    single_mesh.block.trial_variables = {velocity, pressure};
    single_mesh.primal_variable = velocity;
    single_mesh.multiplier_variable = pressure;
    single_mesh.estimate_value = 0.2;
    single_mesh.estimate_tolerance = 1.0e-8;
    single_mesh.test_rows = 12;
    single_mesh.test_cols = 4;
    single_mesh.estimate_scope = "free-free";
    single_mesh.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    single_mesh.estimator_metadata_present = true;
    single_mesh.norm_metadata_present = true;
    single_mesh.mesh_refinement_evidence_present = true;
    single_mesh.mesh_refinement_sample_count = 1;
    summaries.inf_sup_estimates.push_back(single_mesh);

    InfSupEstimateSummary uniform = single_mesh;
    uniform.block.operator_tag = "mixed-pair-uniform";
    uniform.mesh_refinement_sample_count = 3;
    uniform.uniform_lower_bound_evidence_present = true;
    uniform.uniform_lower_bound_value_present = true;
    uniform.uniform_lower_bound = 0.15;
    uniform.mesh_family_scope_present = true;
    uniform.boundary_condition_scope_present = true;
    uniform.inf_sup_theorem_id = "uniform discrete LBB lower-bound study";
    summaries.inf_sup_estimates.push_back(uniform);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto infsup = claimsFrom(
        report, PropertyKind::InfSupCondition, "InfSupAnalyzer");
    ASSERT_EQ(infsup.size(), 2u);
    EXPECT_EQ(infsup[0]->status, PropertyStatus::Likely);
    ASSERT_TRUE(infsup[0]->certification_class.has_value());
    EXPECT_EQ(*infsup[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(infsup[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(infsup[1]->certification_class.has_value());
    EXPECT_EQ(*infsup[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, HDivH1MixedPairNeedsExplicitCertifiedEvidence)
{
    const auto velocity = VariableKey::field(1);
    const auto pressure = VariableKey::field(0);

    auto add_mixed_pair = [&](ProblemAnalysisContext& ctx) {
        ctx.addFieldDescriptor(hDivField(1, 1, "hdiv-velocity"));
        ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1,
                                       "h1-pressure"));
        ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
            velocity, "velocity-block", "evidence-contract"));
        ctx.addContribution(ContributionDescriptor::constraintPairDesc(
            velocity, pressure, "mixed-pair", "divergence-pair",
            "evidence-contract"));
    };

    ProblemAnalysisContext heuristic_ctx;
    add_mixed_pair(heuristic_ctx);
    const auto heuristic_report = analyze(std::move(heuristic_ctx));
    const auto* heuristic = firstFrom(
        heuristic_report, PropertyKind::SpaceCompatibility,
        "SpaceCompatibilityAnalyzer");
    ASSERT_NE(heuristic, nullptr);
    EXPECT_EQ(heuristic->status, PropertyStatus::Unknown);
    ASSERT_TRUE(heuristic->certification_class.has_value());
    EXPECT_EQ(*heuristic->certification_class,
              CertificationClass::NotCertified);

    ProblemAnalysisContext certified_ctx;
    add_mixed_pair(certified_ctx);
    AnalysisSummarySet summaries;
    InfSupPairCertificationSummary certified_pair;
    certified_pair.block.operator_tag = "divergence-pair";
    certified_pair.block.test_variables = {velocity, pressure};
    certified_pair.block.trial_variables = {velocity, pressure};
    certified_pair.primal_variable = velocity;
    certified_pair.multiplier_variable = pressure;
    certified_pair.pair_family = "documented HDiv/H1 saddle pair";
    certified_pair.primal_polynomial_order = 1;
    certified_pair.multiplier_polynomial_order = 1;
    certified_pair.primal_space_family = SpaceFamily::HDiv;
    certified_pair.multiplier_space_family = SpaceFamily::H1;
    certified_pair.known_stable_pair = true;
    certified_pair.mesh_assumption_evidence_present = true;
    certified_pair.domain_assumption_evidence_present = true;
    certified_pair.boundary_condition_scope_present = true;
    certified_pair.beta_lower_bound_present = true;
    certified_pair.beta_lower_bound = 0.12;
    certified_pair.inf_sup_theorem_id = "documented HDiv/H1 Fortin theorem";
    summaries.inf_sup_pair_certifications.push_back(certified_pair);
    certified_ctx.setAnalysisSummaries(std::move(summaries));

    const auto certified_report = analyze(std::move(certified_ctx));
    bool saw_certified_hdiv_h1 = false;
    for (const auto& claim : certified_report.claims) {
        if (claim.kind == PropertyKind::SpaceCompatibility &&
            claim.claim_origin == "SpaceCompatibilityAnalyzer" &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified) {
            saw_certified_hdiv_h1 = true;
            EXPECT_EQ(claim.status, PropertyStatus::Preserved);
            ASSERT_TRUE(claim.space_compatibility_class.has_value());
            EXPECT_EQ(*claim.space_compatibility_class,
                      SpaceCompatibilityClass::Compatible);
        }
    }
    EXPECT_TRUE(saw_certified_hdiv_h1);
}

TEST(AnalysisEvidenceContracts, AdjointCertificationRequiresDiscreteResidualEvidence)
{
    AnalysisSummarySet summaries;
    AdjointConsistencySummary missing_residual;
    missing_residual.contribution_id = "residual";
    missing_residual.goal_functional_id = "goal";
    missing_residual.adjoint_consistency = AdjointConsistencyKind::Yes;
    missing_residual.transpose_backend_support = true;
    missing_residual.boundary_adjoint_metadata_present = true;
    missing_residual.stabilization_adjoint_metadata_present = true;
    missing_residual.goal_linearization_metadata_present = true;
    summaries.adjoint_consistency.push_back(missing_residual);

    AdjointConsistencySummary certified = missing_residual;
    certified.contribution_id = "residual-certified";
    certified.discrete_adjoint_residual_present = true;
    certified.discrete_adjoint_residual = 1.0e-12;
    certified.discrete_adjoint_tolerance = 1.0e-8;
    summaries.adjoint_consistency.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto adjoint = claimsFrom(
        report, PropertyKind::AdjointConsistency,
        "PreservationStructureAnalyzer");
    ASSERT_EQ(adjoint.size(), 2u);
    EXPECT_EQ(adjoint[0]->status, PropertyStatus::Unknown);
    ASSERT_TRUE(adjoint[0]->certification_class.has_value());
    EXPECT_EQ(*adjoint[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(adjoint[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(adjoint[1]->certification_class.has_value());
    EXPECT_EQ(*adjoint[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, CompatibleComplexExactSequenceNeedsCommutingProjectionEvidence)
{
    ProblemAnalysisContext ctx;
    auto field = h1Field(0, 1, FieldType::Scalar, 1, "exact-sequence-field");
    field.has_exact_sequence_structure = true;
    ctx.addFieldDescriptor(field);

    const auto report = analyze(std::move(ctx));
    const auto* claim = firstFrom(
        report, PropertyKind::CompatibleComplexStructure,
        "SpaceCompatibilityAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class, CertificationClass::NotCertified);
}

TEST(AnalysisEvidenceContracts, StablePairMetadataMustMatchFieldOrders)
{
    const auto velocity = VariableKey::field(1);
    const auto pressure = VariableKey::field(0);

    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(h1Field(1, 1, FieldType::Vector, 3, "velocity"));
    ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "pressure"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        velocity, pressure, "stokes-pair", "divergence-pair",
        "evidence-contract"));

    AnalysisSummarySet summaries;
    InfSupPairCertificationSummary stale;
    stale.block.operator_tag = "divergence-pair";
    stale.block.test_variables = {velocity, pressure};
    stale.block.trial_variables = {velocity, pressure};
    stale.primal_variable = velocity;
    stale.multiplier_variable = pressure;
    stale.pair_family = "Taylor-Hood P2/P1";
    stale.primal_polynomial_order = 2;
    stale.multiplier_polynomial_order = 1;
    stale.primal_space_family = SpaceFamily::H1;
    stale.multiplier_space_family = SpaceFamily::H1;
    stale.known_stable_pair = true;
    stale.mesh_assumption_evidence_present = true;
    stale.domain_assumption_evidence_present = true;
    summaries.inf_sup_pair_certifications.push_back(stale);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    EXPECT_FALSE(hasCertifiedClaim(report, PropertyKind::InfSupCondition));

    bool saw_stale_summary = false;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.estimate_scope == "Taylor-Hood P2/P1") {
            saw_stale_summary = true;
            EXPECT_EQ(claim.status, PropertyStatus::Unknown);
            ASSERT_TRUE(claim.certification_class.has_value());
            EXPECT_EQ(*claim.certification_class,
                      CertificationClass::NotCertified);
        }
    }
    EXPECT_TRUE(saw_stale_summary);
}

TEST(AnalysisEvidenceContracts, MixedTemporalStateContributesDynamicAndAlgebraicParts)
{
    const auto u = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    VariableDescriptor desc;
    desc.key = u;
    desc.temporal_state_kind = TemporalStateKind::Mixed;
    desc.max_time_derivative_order = 1;
    desc.participates_in_constraint_blocks = true;
    ctx.addVariableDescriptor(desc);

    AnalysisSummarySet summaries;
    DAEStructureEvidenceSummary evidence;
    evidence.system_id = "mixed-variable-index-one";
    evidence.variables = {u};
    evidence.dae_form_class = DAEFormClass::SemiExplicit;
    evidence.mass_matrix_rank_metadata_present = true;
    evidence.algebraic_jacobian_rank_metadata_present = true;
    evidence.algebraic_jacobian_full_rank = true;
    evidence.hidden_constraint_metadata_present = true;
    evidence.consistent_initial_condition_evidence_present = true;
    evidence.initial_constraint_residual = 0.0;
    evidence.residual_tolerance = 1.0e-10;
    summaries.dae_structure_evidence.push_back(evidence);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    const auto* claim = firstFrom(
        report, PropertyKind::DifferentialAlgebraicStructure,
        "DAEStructureAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claim->dae_class.has_value());
    EXPECT_EQ(*claim->dae_class, DAEClass::Index1DAELike);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class, CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, InvariantDomainCertificationRequiresTypedTheoremEvidence)
{
    AnalysisSummarySet summaries;
    InvariantDomainSummary missing_typed;
    missing_typed.invariant_set_id = "legacy-bounds";
    missing_typed.variables = {VariableKey::field(0)};
    missing_typed.lower_bound_active = true;
    missing_typed.limiter_evidence_present = true;
    missing_typed.cfl_condition_satisfied = true;
    missing_typed.ssp_time_discretization_evidence_present = true;
    missing_typed.source_admissibility_evidence_present = true;
    summaries.invariant_domains.push_back(missing_typed);

    InvariantDomainSummary certified = missing_typed;
    certified.invariant_set_id = "typed-bounds";
    certified.low_order_invariant_domain_evidence_present = true;
    certified.convex_limiting_evidence_present = true;
    certified.spatial_monotonicity_evidence_present = true;
    certified.mass_positivity_evidence_present = true;
    certified.invariant_domain_theorem_id =
        "Guermond-Popov convex limiting invariant domain theorem";
    summaries.invariant_domains.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto invariant = claimsFrom(
        report, PropertyKind::InvariantDomainPreservation,
        "PreservationStructureAnalyzer");
    ASSERT_EQ(invariant.size(), 2u);
    EXPECT_EQ(invariant[0]->status, PropertyStatus::Unknown);
    EXPECT_EQ(invariant[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(invariant[1]->certification_class.has_value());
    EXPECT_EQ(*invariant[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, CouplingGraphDoesNotEmitSelfEdgesForDiagonalBlocks)
{
    const auto u = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        u, "diagonal-block", "evidence-contract"));

    ProblemAnalysisReport report;
    CouplingGraphAnalyzer analyzer;
    analyzer.run(ctx, report);

    EXPECT_TRUE(report.claimsOfKind(PropertyKind::CoupledSystemStructure).empty());
}

TEST(AnalysisEvidenceContracts, ConstraintRankUsesStructuredRigidBodyNullspaceFamily)
{
    const auto u = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    BoundaryConditionDescriptor bc;
    bc.primary_variable = u;
    bc.enforcement_kind = EnforcementKind::Strong;
    bc.anchors_constant_mode = true;
    bc.anchors_rigid_body_translation = true;
    bc.anchors_rigid_body_rotation = false;
    ctx.addBCDescriptor(bc);

    ProblemAnalysisReport report;
    PropertyClaim nullspace;
    nullspace.kind = PropertyKind::Nullspace;
    nullspace.status = PropertyStatus::Exact;
    nullspace.confidence = AnalysisConfidence::High;
    nullspace.field = 0;
    nullspace.variables = {u};
    nullspace.nullspace_family = NullspaceFamily::KernelOfSymGrad;
    nullspace.description = "structured vector kernel";
    report.claims.push_back(std::move(nullspace));

    ConstraintRankAnalyzer analyzer;
    analyzer.run(ctx, report);

    const auto under = report.claimsOfKind(PropertyKind::UnderConstraint);
    ASSERT_EQ(under.size(), 1u);
    EXPECT_EQ(under.front()->status, PropertyStatus::Violated);
}

TEST(AnalysisEvidenceContracts, CoefficientPositivityRequiresCoverageMetadata)
{
    AnalysisSummarySet summaries;
    CoefficientPropertySummary partial;
    partial.coefficient = "partial-positive";
    partial.block = scalarBlock("partial-positive");
    partial.positivity = PositivityClass::Positive;
    summaries.coefficient_properties.push_back(partial);

    CoefficientPropertySummary certified = partial;
    certified.coefficient = "certified-positive";
    certified.block = scalarBlock("certified-positive");
    certified.coefficient_region_coverage_complete = true;
    certified.quadrature_point_coverage_complete = true;
    certified.lower_bound_valid_for_all_samples = true;
    certified.tolerance_metadata_present = true;
    summaries.coefficient_properties.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* partial_operator = firstForCoefficient(
        report, "OperatorClassAnalyzer", "partial-positive");
    ASSERT_NE(partial_operator, nullptr);
    EXPECT_EQ(partial_operator->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_operator->certification_class.has_value());
    EXPECT_EQ(*partial_operator->certification_class,
              CertificationClass::NotCertified);

    const auto* certified_operator = firstForCoefficient(
        report, "OperatorClassAnalyzer", "certified-positive");
    ASSERT_NE(certified_operator, nullptr);
    EXPECT_EQ(certified_operator->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_operator->certification_class.has_value());
    EXPECT_EQ(*certified_operator->certification_class,
              CertificationClass::Certified);

    const auto* partial_constitutive = firstForCoefficient(
        report, "CoefficientConstitutiveAnalyzer", "partial-positive");
    ASSERT_NE(partial_constitutive, nullptr);
    EXPECT_EQ(partial_constitutive->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_constitutive->certification_class.has_value());
    EXPECT_EQ(*partial_constitutive->certification_class,
              CertificationClass::NotCertified);
}

TEST(AnalysisEvidenceContracts, SchurComplementCertificationControlsBlockSchurSolver)
{
    auto make_ctx = [](SchurComplementSummary schur) {
        ProblemAnalysisContext ctx;
        AnalysisSummarySet summaries;
        summaries.schur_complements.push_back(std::move(schur));
        ctx.setAnalysisSummaries(std::move(summaries));
        backends::SolverOptions options;
        options.method = backends::SolverMethod::BlockSchur;
        options.preconditioner = backends::PreconditionerType::FieldSplit;
        ctx.setSolverOptions(options);
        return ctx;
    };

    SchurComplementSummary partial;
    partial.schur_id = "partial-schur";
    partial.block = scalarBlock("partial-schur");
    partial.schur_available = true;
    partial.reduction_exact_for_analysis = true;

    const auto partial_report = analyze(make_ctx(partial));
    const auto* partial_resolution = firstForBlock(
        partial_report, PropertyKind::IndefiniteOperatorResolution,
        "MixedOperatorAnalyzer", "partial-schur");
    ASSERT_NE(partial_resolution, nullptr);
    EXPECT_EQ(partial_resolution->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_resolution->reduced_definiteness_class.has_value());
    EXPECT_EQ(*partial_resolution->reduced_definiteness_class,
              CertificationClass::NotCertified);
    const auto* partial_solver = firstFrom(
        partial_report, PropertyKind::SolverCompatibility,
        "SolverCompatibilityAnalyzer");
    ASSERT_NE(partial_solver, nullptr);
    EXPECT_EQ(partial_solver->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_solver->certification_class.has_value());
    EXPECT_EQ(*partial_solver->certification_class,
              CertificationClass::NotCertified);

    SchurComplementSummary certified = partial;
    certified.schur_id = "certified-schur";
    certified.block = scalarBlock("certified-schur");
    certified.primal_block_invertible_evidence_present = true;
    certified.inf_sup_evidence_present = true;
    certified.nullspace_handling_evidence_present = true;
    certified.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    certified.schur_definiteness_evidence_present = true;
    certified.schur_positivity = PositivityClass::Positive;
    certified.spectral_equivalence_bounds_present = true;
    certified.preconditioner_equivalence_bounds_present = true;

    const auto certified_report = analyze(make_ctx(certified));
    const auto* certified_resolution = firstForBlock(
        certified_report, PropertyKind::IndefiniteOperatorResolution,
        "MixedOperatorAnalyzer", "certified-schur");
    ASSERT_NE(certified_resolution, nullptr);
    EXPECT_EQ(certified_resolution->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_resolution->reduced_definiteness_class.has_value());
    EXPECT_EQ(*certified_resolution->reduced_definiteness_class,
              CertificationClass::Certified);
    const auto* certified_solver = firstFrom(
        certified_report, PropertyKind::SolverCompatibility,
        "SolverCompatibilityAnalyzer");
    ASSERT_NE(certified_solver, nullptr);
    EXPECT_EQ(certified_solver->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_solver->certification_class.has_value());
    EXPECT_EQ(*certified_solver->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, ConservationCertificationRequiresFluxClosureMetadata)
{
    AnalysisSummarySet summaries;
    FluxBalanceSummary partial;
    partial.block = scalarBlock("partial-balance");
    partial.balance_group = "mass";
    partial.symbolic_balance_group = "mass";
    partial.balance_tolerance = 1.0e-8;
    partial.local_residual_norm = 1.0e-12;
    partial.symbolic_balance_evidence_present = true;
    summaries.flux_balances.push_back(partial);

    FluxBalanceSummary certified = partial;
    certified.block = scalarBlock("certified-balance");
    certified.flux_variable_metadata_present = true;
    certified.element_residual_evidence_present = true;
    certified.source_quadrature_consistency_present = true;
    certified.orientation_consistency_present = true;
    certified.boundary_flux_accounted_for = true;
    summaries.flux_balances.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* partial_claim = firstForBlock(
        report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer", "partial-balance");
    ASSERT_NE(partial_claim, nullptr);
    EXPECT_EQ(partial_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_claim->certification_class.has_value());
    EXPECT_EQ(*partial_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* certified_claim = firstForBlock(
        report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer", "certified-balance");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, BoundaryComplementingRequiresRankCountAndCoverageEvidence)
{
    AnalysisSummarySet summaries;
    BoundarySymbolSummary partial;
    partial.block = scalarBlock("partial-boundary", DomainKind::Boundary);
    partial.principal_operator_order = 2;
    partial.boundary_operator_order = 1;
    partial.complementing_condition_satisfied = true;
    summaries.boundary_symbols.push_back(partial);

    BoundarySymbolSummary certified = partial;
    certified.block = scalarBlock("certified-boundary", DomainKind::Boundary);
    certified.boundary_condition_count = 2;
    certified.required_boundary_condition_count = 2;
    certified.principal_symbol_rank_evidence_present = true;
    certified.boundary_symbol_rank_evidence_present = true;
    certified.component_coverage_complete = true;
    certified.dof_coverage_complete = true;
    certified.tangential_frequency_coverage_present = true;
    certified.decaying_root_count_evidence_present = true;
    certified.stable_subspace_dimension_evidence_present = true;
    certified.parameter_ellipticity_evidence_present = true;
    certified.complementing_margin_present = true;
    certified.complementing_margin = 0.2;
    certified.complementing_theorem_id =
        "Agmon-Douglis-Nirenberg complementing condition";
    summaries.boundary_symbols.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* partial_claim = firstForBlock(
        report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer", "partial-boundary");
    ASSERT_NE(partial_claim, nullptr);
    EXPECT_EQ(partial_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_claim->certification_class.has_value());
    EXPECT_EQ(*partial_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* certified_claim = firstForBlock(
        report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer", "certified-boundary");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, DAECertificationRequiresSemiExplicitFormMetadata)
{
    const auto u = VariableKey::field(0);
    const auto lambda = VariableKey::field(1);

    ProblemAnalysisContext ctx;
    VariableDescriptor u_desc;
    u_desc.key = u;
    u_desc.temporal_state_kind = TemporalStateKind::Dynamic;
    u_desc.max_time_derivative_order = 1;
    ctx.addVariableDescriptor(u_desc);
    VariableDescriptor lambda_desc;
    lambda_desc.key = lambda;
    lambda_desc.temporal_state_kind = TemporalStateKind::Algebraic;
    lambda_desc.participates_in_constraint_blocks = true;
    ctx.addVariableDescriptor(lambda_desc);

    AnalysisSummarySet summaries;
    DAEStructureEvidenceSummary evidence;
    evidence.system_id = "rank-only-dae";
    evidence.variables = {u, lambda};
    evidence.mass_matrix_rank_metadata_present = true;
    evidence.algebraic_jacobian_rank_metadata_present = true;
    evidence.algebraic_jacobian_full_rank = true;
    evidence.hidden_constraint_metadata_present = true;
    evidence.consistent_initial_condition_evidence_present = true;
    evidence.initial_constraint_residual = 0.0;
    evidence.residual_tolerance = 1.0e-10;
    summaries.dae_structure_evidence.push_back(evidence);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    const auto* claim = firstFrom(
        report, PropertyKind::DifferentialAlgebraicStructure,
        "DAEStructureAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class, CertificationClass::NotCertified);
}

TEST(AnalysisEvidenceContracts, NonlinearExactTangentRequiresActionEvidence)
{
    AnalysisSummarySet summaries;
    NonlinearTangentSummary partial;
    partial.residual_id = "exact-without-action";
    partial.block = scalarBlock("exact-without-action");
    partial.tangent_consistency = TangentConsistencyClass::Exact;
    partial.tangent_symmetry = SymmetryClass::Symmetric;
    partial.tangent_positivity = PositivityClass::Positive;
    summaries.nonlinear_tangents.push_back(partial);

    NonlinearTangentSummary certified = partial;
    certified.residual_id = "exact-with-action";
    certified.block = scalarBlock("exact-with-action");
    certified.jacobian_action_available = true;
    certified.finite_difference_action_error = 1.0e-12;
    certified.finite_difference_tolerance = 1.0e-8;
    summaries.nonlinear_tangents.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* partial_claim = firstForBlock(
        report, PropertyKind::NonlinearTangentStructure,
        "NonlinearTangentAnalyzer", "exact-without-action");
    ASSERT_NE(partial_claim, nullptr);
    EXPECT_EQ(partial_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_claim->certification_class.has_value());
    EXPECT_EQ(*partial_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* certified_claim = firstForBlock(
        report, PropertyKind::NonlinearTangentStructure,
        "NonlinearTangentAnalyzer", "exact-with-action");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, ErrorEstimatorCertificationRequiresReliabilityMetadata)
{
    AnalysisSummarySet summaries;
    ErrorEstimatorSummary partial;
    partial.estimator_id = "residual-jump-only";
    partial.block = scalarBlock("residual-jump-only");
    partial.residual_metadata_present = true;
    partial.jump_metadata_present = true;
    summaries.error_estimators.push_back(partial);

    ErrorEstimatorSummary certified = partial;
    certified.estimator_id = "certified-estimator";
    certified.block = scalarBlock("certified-estimator");
    certified.norm_metadata_present = true;
    certified.estimator_norm_scope_metadata_present = true;
    certified.estimator_norm_id = "energy-norm";
    certified.pde_operator_class_metadata_present = true;
    certified.boundary_residual_metadata_present = true;
    certified.data_oscillation_metadata_present = true;
    certified.coefficient_source_regularity_metadata_present = true;
    certified.shape_regular_mesh_evidence_present = true;
    certified.reliability_constant_metadata_present = true;
    certified.reliability_constant = 3.0;
    certified.efficiency_constant_metadata_present = true;
    certified.efficiency_constant = 0.25;
    certified.effectivity_bounds_present = true;
    certified.effectivity_lower_bound = 0.7;
    certified.effectivity_upper_bound = 1.4;
    certified.effectivity_sample_count = 4;
    certified.refinement_evidence_present = true;
    certified.estimator_theorem_id = "Verfurth residual estimator theorem";
    summaries.error_estimators.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* partial_claim = firstForBlock(
        report, PropertyKind::ErrorEstimatorEligibility,
        "ErrorEstimatorAnalyzer", "residual-jump-only");
    ASSERT_NE(partial_claim, nullptr);
    EXPECT_EQ(partial_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_claim->certification_class.has_value());
    EXPECT_EQ(*partial_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* certified_claim = firstForBlock(
        report, PropertyKind::ErrorEstimatorEligibility,
        "ErrorEstimatorAnalyzer", "certified-estimator");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, MinimumResidualStabilityRequiresFortinRieszAndConditioningEvidence)
{
    AnalysisSummarySet summaries;
    MinimumResidualStabilitySummary core_only;
    core_only.method_id = "core-only-dpg";
    core_only.block = scalarBlock("core-only-dpg");
    core_only.method_class = MinimumResidualMethodClass::DPG;
    core_only.trial_space_metadata_present = true;
    core_only.test_space_metadata_present = true;
    core_only.distinct_test_trial_spaces = true;
    core_only.residual_norm_metadata_present = true;
    core_only.test_norm_metadata_present = true;
    core_only.residual_norm_id = "graph-residual";
    core_only.test_norm_id = "optimal-test-norm";
    core_only.minimum_residual_theorem_id =
        "Demkowicz-Gopalakrishnan DPG stability theorem";
    core_only.method_scope_metadata_present = true;
    summaries.minimum_residual_stability.push_back(core_only);

    MinimumResidualStabilitySummary certified = core_only;
    certified.method_id = "certified-dpg";
    certified.block = scalarBlock("certified-dpg");
    certified.residual_norm_id = "graph-residual";
    certified.test_norm_id = "optimal-test-norm";
    certified.minimum_residual_theorem_id =
        "Demkowicz-Gopalakrishnan DPG stability theorem";
    certified.method_scope_metadata_present = true;
    certified.riesz_map_metadata_present = true;
    certified.optimal_test_metadata_present = true;
    certified.enrichment_sufficiency_evidence_present = true;
    certified.residual_control_constant_present = true;
    certified.residual_control_constant = 2.0;
    certified.local_trial_to_test_conditioning_present = true;
    certified.local_trial_to_test_condition_estimate = 10.0;
    certified.normal_equation_conditioning_present = true;
    certified.normal_equation_condition_estimate = 100.0;
    summaries.minimum_residual_stability.push_back(certified);

    MinimumResidualStabilitySummary violated = certified;
    violated.method_id = "violated-dpg";
    violated.block = scalarBlock("violated-dpg");
    violated.violation_count = 1;
    summaries.minimum_residual_stability.push_back(violated);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::MinimumResidualStability,
        "MinimumResidualStabilityAnalyzer");
    ASSERT_EQ(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Likely);
    ASSERT_TRUE(claims[0]->certification_class.has_value());
    EXPECT_EQ(*claims[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claims[1]->certification_class.has_value());
    EXPECT_EQ(*claims[1]->certification_class,
              CertificationClass::Certified);
    EXPECT_EQ(claims[2]->status, PropertyStatus::Violated);
    ASSERT_TRUE(claims[2]->certification_class.has_value());
    EXPECT_EQ(*claims[2]->certification_class,
              CertificationClass::Violated);
}

TEST(AnalysisEvidenceContracts, EnergyStabilityRequiresEnergyFunctionalMetadata)
{
    AnalysisSummarySet summaries;
    EnergyEntropySummary missing;
    missing.energy_entropy_id = "energy-missing-functional";
    missing.law_kind = EnergyEntropyLawKind::Energy;
    missing.expected_production_sign = BalanceSignClass::Nonpositive;
    missing.balance_tolerance = 1.0e-8;
    missing.observed_discrete_balance = 0.0;
    missing.observed_production = -1.0e-9;
    summaries.energy_entropy.push_back(missing);

    EnergyEntropySummary certified = missing;
    certified.energy_entropy_id = "energy-certified";
    certified.energy_functional_id = "quadratic-energy";
    certified.energy_norm_id = "mass-inner-product";
    certified.energy_entropy_theorem_id = "discrete energy identity";
    certified.energy_functional_metadata_present = true;
    certified.energy_norm_metadata_present = true;
    certified.energy_positivity_evidence_present = true;
    certified.energy_coercivity_evidence_present = true;
    certified.discrete_dissipation_identity_evidence_present = true;
    certified.boundary_source_energy_accounting_present = true;
    summaries.energy_entropy.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto energy = claimsFrom(
        report, PropertyKind::EnergyStability,
        "EnergyEntropyLawAnalyzer");
    ASSERT_EQ(energy.size(), 2u);
    EXPECT_EQ(energy[0]->status, PropertyStatus::Unknown);
    ASSERT_TRUE(energy[0]->certification_class.has_value());
    EXPECT_EQ(*energy[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(energy[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(energy[1]->certification_class.has_value());
    EXPECT_EQ(*energy[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, BoundaryComplementingRejectsRootSubspaceMismatch)
{
    AnalysisSummarySet summaries;
    BoundarySymbolSummary boundary;
    boundary.block = scalarBlock("mismatched-boundary", DomainKind::Boundary);
    boundary.complementing_condition_satisfied = true;
    boundary.boundary_condition_count = 2;
    boundary.required_boundary_condition_count = 2;
    boundary.principal_symbol_rank_evidence_present = true;
    boundary.boundary_symbol_rank_evidence_present = true;
    boundary.component_coverage_complete = true;
    boundary.dof_coverage_complete = true;
    boundary.tangential_frequency_coverage_present = true;
    boundary.decaying_root_count_evidence_present = true;
    boundary.stable_subspace_dimension_evidence_present = true;
    boundary.parameter_ellipticity_evidence_present = true;
    boundary.complementing_margin_present = true;
    boundary.complementing_margin = 0.1;
    boundary.complementing_theorem_id =
        "Agmon-Douglis-Nirenberg complementing condition";
    boundary.root_subspace_mismatch_count = 1;
    summaries.boundary_symbols.push_back(boundary);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto* claim = firstForBlock(
        report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer", "mismatched-boundary");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class, CertificationClass::Violated);
}

TEST(AnalysisEvidenceContracts, CompatibleComplexAloneDoesNotCertifySpectralCorrectness)
{
    AnalysisSummarySet summaries;
    SpectralStructureSummary compatible_only;
    compatible_only.block = scalarBlock("compatible-only-spectrum");
    compatible_only.eigenproblem_declared = true;
    compatible_only.self_adjoint_evidence = true;
    compatible_only.compactness_evidence = true;
    compatible_only.compatible_complex_evidence = true;
    summaries.spectral_structures.push_back(compatible_only);

    SpectralStructureSummary certified = compatible_only;
    certified.block = scalarBlock("compatible-theorem-spectrum");
    certified.compatible_complex_spectral_theorem_evidence = true;
    certified.spectral_convergence_theorem_id =
        "FEEC spectral correctness theorem";
    summaries.spectral_structures.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto* compatible = firstForBlock(
        report, PropertyKind::SpectralCorrectness,
        "SpectralSpuriousModeAnalyzer", "compatible-only-spectrum");
    ASSERT_NE(compatible, nullptr);
    EXPECT_EQ(compatible->status, PropertyStatus::Unknown);
    const auto* theorem = firstForBlock(
        report, PropertyKind::SpectralCorrectness,
        "SpectralSpuriousModeAnalyzer", "compatible-theorem-spectrum");
    ASSERT_NE(theorem, nullptr);
    EXPECT_EQ(theorem->status, PropertyStatus::Preserved);
    ASSERT_TRUE(theorem->certification_class.has_value());
    EXPECT_EQ(*theorem->certification_class, CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, ErrorEstimatorRejectsInvalidQuantitativeBounds)
{
    AnalysisSummarySet summaries;
    ErrorEstimatorSummary invalid;
    invalid.estimator_id = "invalid-effectivity";
    invalid.block = scalarBlock("invalid-effectivity");
    invalid.residual_metadata_present = true;
    invalid.jump_metadata_present = true;
    invalid.norm_metadata_present = true;
    invalid.estimator_norm_scope_metadata_present = true;
    invalid.estimator_norm_id = "energy-norm";
    invalid.pde_operator_class_metadata_present = true;
    invalid.boundary_residual_metadata_present = true;
    invalid.data_oscillation_metadata_present = true;
    invalid.coefficient_source_regularity_metadata_present = true;
    invalid.shape_regular_mesh_evidence_present = true;
    invalid.reliability_constant_metadata_present = true;
    invalid.reliability_constant = 2.0;
    invalid.efficiency_constant_metadata_present = true;
    invalid.efficiency_constant = 0.5;
    invalid.effectivity_bounds_present = true;
    invalid.effectivity_lower_bound = 2.0;
    invalid.effectivity_upper_bound = 1.0;
    invalid.effectivity_sample_count = 4;
    invalid.refinement_evidence_present = true;
    invalid.estimator_theorem_id = "Verfurth residual estimator theorem";
    summaries.error_estimators.push_back(invalid);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto* claim = firstForBlock(
        report, PropertyKind::ErrorEstimatorEligibility,
        "ErrorEstimatorAnalyzer", "invalid-effectivity");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class, CertificationClass::Violated);
}

TEST(AnalysisEvidenceContracts, MinimumResidualRejectsUnknownMethodOrInvalidConstants)
{
    AnalysisSummarySet summaries;
    MinimumResidualStabilitySummary unknown;
    unknown.method_id = "unknown-method";
    unknown.block = scalarBlock("unknown-method");
    unknown.trial_space_metadata_present = true;
    unknown.test_space_metadata_present = true;
    unknown.distinct_test_trial_spaces = true;
    unknown.residual_norm_metadata_present = true;
    unknown.test_norm_metadata_present = true;
    unknown.method_scope_metadata_present = true;
    unknown.residual_norm_id = "graph-residual";
    unknown.test_norm_id = "optimal-test-norm";
    unknown.minimum_residual_theorem_id = "method theorem";
    summaries.minimum_residual_stability.push_back(unknown);

    MinimumResidualStabilitySummary invalid = unknown;
    invalid.method_id = "invalid-residual-control";
    invalid.block = scalarBlock("invalid-residual-control");
    invalid.method_class = MinimumResidualMethodClass::DPG;
    invalid.riesz_map_metadata_present = true;
    invalid.optimal_test_metadata_present = true;
    invalid.enrichment_sufficiency_evidence_present = true;
    invalid.residual_control_constant_present = true;
    invalid.residual_control_constant = 0.0;
    invalid.local_trial_to_test_conditioning_present = true;
    invalid.local_trial_to_test_condition_estimate = 10.0;
    invalid.normal_equation_conditioning_present = true;
    invalid.normal_equation_condition_estimate = 100.0;
    summaries.minimum_residual_stability.push_back(invalid);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::MinimumResidualStability,
        "MinimumResidualStabilityAnalyzer");
    ASSERT_EQ(claims.size(), 2u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Unknown);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Violated);
}

TEST(AnalysisEvidenceContracts, DescriptorPencilEvidenceCertifiesIndexOneDAE)
{
    const auto u = VariableKey::field(0);
    const auto lambda = VariableKey::field(1);

    ProblemAnalysisContext ctx;
    VariableDescriptor u_desc;
    u_desc.key = u;
    u_desc.temporal_state_kind = TemporalStateKind::Dynamic;
    u_desc.max_time_derivative_order = 1;
    ctx.addVariableDescriptor(u_desc);
    VariableDescriptor lambda_desc;
    lambda_desc.key = lambda;
    lambda_desc.temporal_state_kind = TemporalStateKind::Algebraic;
    lambda_desc.participates_in_constraint_blocks = true;
    ctx.addVariableDescriptor(lambda_desc);

    AnalysisSummarySet summaries;
    DAEStructureEvidenceSummary evidence;
    evidence.system_id = "descriptor-index-one";
    evidence.variables = {u, lambda};
    evidence.dae_form_class = DAEFormClass::DescriptorPencil;
    evidence.descriptor_pencil_metadata_present = true;
    evidence.regular_descriptor_pencil_evidence_present = true;
    evidence.strangeness_index_metadata_present = true;
    evidence.strangeness_index = 1;
    evidence.projector_index_metadata_present = true;
    evidence.projector_consistency_evidence_present = true;
    evidence.hidden_constraint_metadata_present = true;
    evidence.consistent_initial_condition_evidence_present = true;
    evidence.initial_constraint_residual = 0.0;
    evidence.residual_tolerance = 1.0e-10;
    evidence.dae_index_theorem_id = "Kunkel-Mehrmann regular pencil theorem";
    summaries.dae_structure_evidence.push_back(evidence);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    const auto* claim = firstFrom(
        report, PropertyKind::DifferentialAlgebraicStructure,
        "DAEStructureAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class, CertificationClass::Certified);
    ASSERT_TRUE(claim->dae_class.has_value());
    EXPECT_EQ(*claim->dae_class, DAEClass::Index1DAELike);
}
