/**
 * @file test_AnalysisEvidenceContracts.cpp
 * @brief Tests that certification claims require theorem-specific evidence.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
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

FieldDescriptor l2Field(FieldId id, int order, std::string name)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = FieldType::Scalar;
    fd.value_dimension = 1;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::L2;
    fd.trace_capabilities = TraceCapabilityFlags::None;
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

void addSemiExplicitDAECertificationScope(DAEStructureEvidenceSummary& summary)
{
    summary.dae_index_theorem_id =
        "Hairer-Wanner semi-explicit index-1 DAE theorem";
    summary.dae_index_scope = "local smooth semi-explicit residual map";
    summary.local_validity_scope_present = true;
    summary.smoothness_or_regular_operator_evidence_present = true;
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

const PropertyClaim* firstRobustnessForCoefficient(
    const ProblemAnalysisReport& report,
    const std::string& coefficient_id)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::ParameterRobustness &&
            claim.claim_origin == "CoefficientConstitutiveAnalyzer" &&
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

void addEnergyQuantitativeEvidence(EnergyEntropySummary& summary)
{
    summary.energy_coercivity_lower_bound_present = true;
    summary.energy_coercivity_lower_bound = 0.25;
    summary.energy_norm_equivalence_bounds_present = true;
    summary.energy_norm_equivalence_lower_bound = 0.25;
    summary.energy_norm_equivalence_upper_bound = 4.0;
    summary.energy_dissipation_residual_bound_present = true;
    summary.energy_dissipation_residual_bound = 1.0e-12;
    summary.energy_dissipation_tolerance_present = true;
    summary.energy_dissipation_tolerance = 1.0e-8;
}

void addEntropyQuantitativeEvidence(EnergyEntropySummary& summary)
{
    summary.entropy_convexity_lower_bound_present = true;
    summary.entropy_convexity_lower_bound = 0.1;
    summary.entropy_flux_inequality_residual_present = true;
    summary.entropy_flux_inequality_residual = 1.0e-12;
    summary.entropy_flux_inequality_tolerance_present = true;
    summary.entropy_flux_inequality_tolerance = 1.0e-8;
    summary.entropy_dissipation_bound_present = true;
    summary.entropy_dissipation_bound = 1.0e-12;
}

void addSpectralQuantitativeEvidence(SpectralStructureSummary& summary)
{
    summary.spectral_tolerance = 1.0e-8;
    summary.rayleigh_quotient_lower_bound = 0.1;
    summary.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
}

void addSpectralComplexProvenance(SpectralStructureSummary& summary)
{
    summary.compatible_complex_evidence = true;
    summary.compatible_complex_spectral_theorem_evidence = true;
    summary.bounded_projection_evidence_present = true;
    summary.projection_bound_present = true;
    summary.projection_bound = 2.0;
    summary.mesh_family_scope_present = true;
    summary.mesh_family_scope = "shape-regular tetrahedral mesh family";
    summary.shape_regular_mesh_evidence_present = true;
}

void addEstimatorShapeRegularityEvidence(ErrorEstimatorSummary& summary)
{
    summary.shape_regular_mesh_evidence_present = true;
    summary.mesh_family_scope_present = true;
    summary.mesh_family_scope = "shape-regular adaptive mesh family";
    summary.shape_regular_constant_present = true;
    summary.shape_regular_constant = 4.0;
}

void addEquilibriumScopeEvidence(EquilibriumPreservationSummary& summary)
{
    summary.equilibrium_family_id = "lake-at-rest";
    summary.equilibrium_preservation_theorem_id =
        "hydrostatic reconstruction well-balanced theorem";
    summary.equilibrium_scope_metadata_present = true;
    summary.source_model_scope_metadata_present = true;
    summary.reconstruction_scope_metadata_present = true;
}

void addInvariantDomainCflEvidence(InvariantDomainSummary& summary)
{
    summary.cfl_estimate_present = true;
    summary.cfl_estimate = 0.4;
    summary.accepted_cfl_bound_present = true;
    summary.accepted_cfl_bound = 0.5;
    summary.wave_speed_bound_present = true;
    summary.wave_speed_bound = 2.0;
    summary.time_step_scope = "forward Euler step";
    summary.mesh_size_scope = "active cell family";
}

void addMovingDomainGclEvidence(MovingDomainSummary& summary)
{
    summary.geometric_conservation_tolerance_declared = true;
    summary.metric_identity_evidence_present = true;
    summary.free_stream_preservation_residual_present = true;
    summary.free_stream_preservation_residual = 0.0;
    summary.gcl_theorem_id = "ALE discrete geometric conservation law";
    summary.constant_state_scope = "free-stream constant state";
    summary.mesh_update_time_scheme = "matching ALE mesh update";
}

void addTemporalCflCertificate(TemporalStabilitySummary& summary,
                               Real cfl,
                               Real accepted_bound)
{
    summary.cfl_estimate_present = true;
    summary.cfl_estimate = cfl;
    summary.accepted_cfl_bound_present = true;
    summary.accepted_cfl_bound = accepted_bound;
    summary.cfl_derivation_metadata_present = true;
    summary.cfl_bound_scope = "scheme-specific spatial operator CFL bound";
    summary.cfl_margin_present = true;
    summary.cfl_margin = accepted_bound >= cfl ? accepted_bound - cfl : 0.0;
}

void addTransferQuantitativeCertificate(TransferOperatorSummary& summary)
{
    summary.transfer_theorem_id =
        "mortar projection stability and conservation theorem";
    summary.interface_quadrature_scope = "nonmatching interface quadrature";
    summary.rank_defect_present = true;
    summary.rank_defect = 0.0;
    summary.projection_operator_norm_present = true;
    summary.projection_operator_norm = 1.25;
    summary.accepted_projection_operator_norm_present = true;
    summary.accepted_projection_operator_norm = 2.0;
    summary.mortar_inf_sup_lower_bound_present = true;
    summary.mortar_inf_sup_lower_bound = 0.2;
    summary.interface_mass_condition_number_present = true;
    summary.interface_mass_condition_number = 8.0;
    summary.accepted_interface_mass_condition_bound_present = true;
    summary.accepted_interface_mass_condition_bound = 20.0;
}

void addCoupledCommonScope(CoupledSystemStabilitySummary& summary)
{
    summary.coupled_stability_theorem_id =
        "coupled fixed-point stability theorem";
    summary.coupling_norm_id = "interface energy norm";
    summary.coupling_norm_metadata_present = true;
    summary.coupling_operator_scope_id = "coupled interface iteration";
    summary.coupling_operator_scope_metadata_present = true;
    summary.coupling_time_horizon_scope = "one coupled time step";
    summary.coupling_time_horizon_present = true;
    summary.coupling_time_horizon = 1.0;
}

void addCoupledContractionCertificate(CoupledSystemStabilitySummary& summary)
{
    addCoupledCommonScope(summary);
    summary.contraction_factor_bound_present = true;
    summary.contraction_factor_bound = 0.5;
    summary.accepted_contraction_factor_bound_present = true;
    summary.accepted_contraction_factor_bound = 0.9;
}

void addCoupledEnergyCertificate(CoupledSystemStabilitySummary& summary)
{
    addCoupledCommonScope(summary);
    summary.coupled_energy_coercivity_lower_bound_present = true;
    summary.coupled_energy_coercivity_lower_bound = 0.25;
    summary.coupled_energy_norm_equivalence_bounds_present = true;
    summary.coupled_energy_norm_equivalence_lower_bound = 0.25;
    summary.coupled_energy_norm_equivalence_upper_bound = 4.0;
}

void addMinimumResidualConditioningBounds(
    MinimumResidualStabilitySummary& summary)
{
    summary.accepted_local_trial_to_test_condition_bound_present = true;
    summary.accepted_local_trial_to_test_condition_bound = 20.0;
    summary.accepted_normal_equation_condition_bound_present = true;
    summary.accepted_normal_equation_condition_bound = 200.0;
    summary.condition_bound_scope_metadata_present = true;
    summary.condition_bound_scope = "active enriched test-search space";
}

DiscreteMatrixSummary certifiedSignPatternMatrix(std::string tag)
{
    DiscreteMatrixSummary matrix;
    matrix.block = scalarBlock(std::move(tag));
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
    matrix.max_abs_row_sum = 1.0;
    matrix.scanned_row_count = 2;
    matrix.expected_row_count = 2;
    matrix.scanned_entry_count = 4;
    matrix.m_matrix_certification_evidence = true;
    return matrix;
}

FluxBalanceSummary certifiedFluxBalance(std::string tag,
                                         DomainKind domain = DomainKind::Cell)
{
    FluxBalanceSummary flux;
    flux.block = scalarBlock(std::move(tag), domain);
    flux.balance_group = "closed";
    flux.symbolic_balance_group = "closed";
    flux.balance_tolerance = 1.0e-8;
    flux.local_residual_norm = 1.0e-12;
    flux.global_residual_norm = 1.0e-12;
    flux.interface_pair_residual_norm = 0.0;
    flux.symbolic_balance_evidence_present = true;
    flux.flux_variable_metadata_present = true;
    flux.element_residual_evidence_present = true;
    flux.source_quadrature_consistency_present = true;
    flux.orientation_consistency_present = true;
    flux.boundary_flux_accounted_for = true;
    flux.steady_balance_scope = true;
    if (domain == DomainKind::InterfaceFace) {
        flux.interface_pair_count = 1;
        flux.face_pair_residual_evidence_present = true;
    }
    return flux;
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
    penalty.min_scale_value = 10.0;
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
    penalty.trace_inverse_constant = 4.0;
    penalty.scale_theorem_id = "Nitsche trace-inverse coercivity bound";
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

    AnalysisSummarySet invalid_summaries;
    invalid_summaries.boundary_symbols.push_back(boundary);
    ParameterScaleSummary invalid_penalty = penalty;
    invalid_penalty.max_scale_value =
        std::numeric_limits<Real>::infinity();
    invalid_summaries.parameter_scales.push_back(invalid_penalty);
    ProblemAnalysisContext invalid_ctx;
    invalid_ctx.setAnalysisSummaries(std::move(invalid_summaries));
    const auto invalid_report = analyze(std::move(invalid_ctx));
    const auto* invalid = firstFrom(
        invalid_report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(invalid, nullptr);
    EXPECT_EQ(invalid->status, PropertyStatus::Violated);
}

TEST(AnalysisEvidenceContracts, WeakBoundaryCoercivityDoesNotAggregateSplitCertificates)
{
    AnalysisSummarySet summaries;
    BoundarySymbolSummary boundary;
    boundary.block = scalarBlock("split-nitsche-face", DomainKind::Boundary);
    boundary.complementing_condition_satisfied = true;
    summaries.boundary_symbols.push_back(boundary);

    ParameterScaleSummary scale_only;
    scale_only.role = ParameterScaleRole::WeakBoundaryPenalty;
    scale_only.block = boundary.block;
    scale_only.min_scale_value = 10.0;
    scale_only.max_scale_value = 10.0;
    summaries.parameter_scales.push_back(scale_only);

    ParameterScaleSummary lower_bound_only = scale_only;
    lower_bound_only.required_lower_bound_present = true;
    lower_bound_only.required_lower_bound = 4.0;
    summaries.parameter_scales.push_back(lower_bound_only);

    ParameterScaleSummary theorem_only = scale_only;
    theorem_only.trace_inverse_metadata_present = true;
    theorem_only.trace_inverse_constant = 4.0;
    theorem_only.scale_theorem_id =
        "Nitsche trace-inverse coercivity bound";
    summaries.parameter_scales.push_back(theorem_only);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto* claim = firstFrom(
        report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class,
              CertificationClass::NotCertified);
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
    adequacy.method_scope_metadata_present = true;
    adequacy.stabilization_theorem_id =
        "Brooks-Hughes SUPG residual-consistent stability estimate";
    adequacy.stability_norm_id = "streamline-diffusion norm";
    adequacy.stability_norm_metadata_present = true;
    adequacy.stabilization_parameter_bounds_present = true;
    adequacy.minimum_stabilization_parameter = 0.05;
    adequacy.maximum_stabilization_parameter = 0.5;
    adequacy.scaling_law_metadata_present = true;
    adequacy.consistency_order_metadata_present = true;
    adequacy.consistency_order = 1;
    adequacy.boundary_treatment_metadata_present = true;
    adequacy.peclet_condition_satisfied = true;
    adequacy.peclet_estimate_present = true;
    adequacy.peclet_estimate = 3.0;
    adequacy.peclet_regime_bounds_present = true;
    adequacy.peclet_regime_lower_bound = 1.0;
    adequacy.peclet_regime_upper_bound = 10.0;
    adequacy.peclet_scope = "streamline cell Peclet regime";
    adequacy.cfl_condition_satisfied = true;
    adequacy.cfl_estimate_present = true;
    adequacy.cfl_estimate = 0.25;
    adequacy.accepted_cfl_bound_present = true;
    adequacy.accepted_cfl_bound = 0.5;
    adequacy.cfl_scope = "explicit stabilized transport step";
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

    AnalysisSummarySet invalid_summaries;
    StabilizationAdequacySummary invalid = adequacy;
    invalid.stabilization_id = "supg-invalid";
    invalid.minimum_stabilization_parameter =
        std::numeric_limits<Real>::quiet_NaN();
    invalid.maximum_stabilization_parameter = 0.5;
    invalid_summaries.stabilization_adequacy.push_back(invalid);
    ProblemAnalysisContext invalid_ctx;
    invalid_ctx.setAnalysisSummaries(std::move(invalid_summaries));
    const auto invalid_report = analyze(std::move(invalid_ctx));
    const auto* invalid_claim = firstFrom(
        invalid_report, PropertyKind::Stabilization,
        "StabilizationAnalyzer");
    ASSERT_NE(invalid_claim, nullptr);
    EXPECT_EQ(invalid_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(invalid_claim->certification_class.has_value());
    EXPECT_EQ(*invalid_claim->certification_class,
              CertificationClass::NotCertified);

    AnalysisSummarySet cfl_summaries;
    StabilizationAdequacySummary cfl_excess = adequacy;
    cfl_excess.stabilization_id = "supg-cfl-excess";
    cfl_excess.block = scalarBlock("supg-cfl-excess");
    cfl_excess.cfl_estimate = 0.75;
    cfl_excess.accepted_cfl_bound = 0.5;
    cfl_summaries.stabilization_adequacy.push_back(cfl_excess);
    ProblemAnalysisContext cfl_ctx;
    cfl_ctx.setAnalysisSummaries(std::move(cfl_summaries));
    const auto cfl_report = analyze(std::move(cfl_ctx));
    const auto* cfl_claim = firstFrom(
        cfl_report, PropertyKind::Stabilization,
        "StabilizationAnalyzer");
    ASSERT_NE(cfl_claim, nullptr);
    EXPECT_EQ(cfl_claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(cfl_claim->certification_class.has_value());
    EXPECT_EQ(*cfl_claim->certification_class,
              CertificationClass::Violated);
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

TEST(AnalysisEvidenceContracts, NonnormalTemporalStabilityNeedsNormHorizonAndAcceptedBound)
{
    TemporalStabilitySummary base;
    base.time_scheme = "nonnormal-growth";
    base.stability_class = TemporalStabilityClass::AStable;
    base.amplification_radius = 0.9;
    base.amplification_radius_present = true;
    base.stability_metadata_present = true;
    base.nonnormal_growth_bound = 2.0;
    base.nonnormal_growth_bound_present = true;
    base.nonnormal_growth_bound_finite = true;

    AnalysisSummarySet partial_summaries;
    partial_summaries.temporal_stability.push_back(base);
    ProblemAnalysisContext partial_ctx;
    partial_ctx.setAnalysisSummaries(std::move(partial_summaries));
    const auto partial_report = analyze(std::move(partial_ctx));
    const auto* partial_claim = firstFrom(
        partial_report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_NE(partial_claim, nullptr);
    EXPECT_EQ(partial_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_claim->certification_class.has_value());
    EXPECT_EQ(*partial_claim->certification_class,
              CertificationClass::NotCertified);

    TemporalStabilitySummary pseudospectral_only = base;
    pseudospectral_only.stability_theorem_id =
        "Trefethen-Embree pseudospectral transient-growth bound";
    pseudospectral_only.stability_norm_id = "discrete l2 operator norm";
    pseudospectral_only.stability_norm_metadata_present = true;
    pseudospectral_only.operator_scope_id = "assembled-step-operator";
    pseudospectral_only.operator_scope_metadata_present = true;
    pseudospectral_only.time_horizon = 0.25;
    pseudospectral_only.time_horizon_metadata_present = true;
    pseudospectral_only.accepted_nonnormal_growth_bound = 3.0;
    pseudospectral_only.accepted_nonnormal_growth_bound_present = true;
    pseudospectral_only.nonnormal_operator_evidence_present = true;
    pseudospectral_only.pseudospectral_bound_present = true;
    pseudospectral_only.nonnormal_growth_bound_present = false;
    pseudospectral_only.nonnormal_growth_bound_finite = false;
    AnalysisSummarySet pseudospectral_summaries;
    pseudospectral_summaries.temporal_stability.push_back(
        pseudospectral_only);
    ProblemAnalysisContext pseudospectral_ctx;
    pseudospectral_ctx.setAnalysisSummaries(
        std::move(pseudospectral_summaries));
    const auto pseudospectral_report = analyze(
        std::move(pseudospectral_ctx));
    const auto* pseudospectral_claim = firstFrom(
        pseudospectral_report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_NE(pseudospectral_claim, nullptr);
    EXPECT_EQ(pseudospectral_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(pseudospectral_claim->certification_class.has_value());
    EXPECT_EQ(*pseudospectral_claim->certification_class,
              CertificationClass::NotCertified);

    TemporalStabilitySummary certified = base;
    certified.stability_theorem_id =
        "Trefethen-Embree pseudospectral transient-growth bound";
    certified.stability_norm_id = "discrete l2 operator norm";
    certified.stability_norm_metadata_present = true;
    certified.operator_scope_id = "assembled-step-operator";
    certified.operator_scope_metadata_present = true;
    certified.time_horizon = 0.25;
    certified.time_horizon_metadata_present = true;
    certified.accepted_nonnormal_growth_bound = 3.0;
    certified.accepted_nonnormal_growth_bound_present = true;
    certified.nonnormal_operator_evidence_present = true;
    AnalysisSummarySet certified_summaries;
    certified_summaries.temporal_stability.push_back(certified);
    ProblemAnalysisContext certified_ctx;
    certified_ctx.setAnalysisSummaries(std::move(certified_summaries));
    const auto certified_report = analyze(std::move(certified_ctx));
    const auto* certified_claim = firstFrom(
        certified_report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, TemporalStabilityRejectsInvalidScalarBounds)
{
    auto base = TemporalStabilitySummary{};
    base.time_scheme = "invalid-scalar-bound";
    base.stability_class = TemporalStabilityClass::AStable;
    base.stability_metadata_present = true;
    base.amplification_radius_present = true;
    base.amplification_radius = 0.9;
    base.stability_theorem_id = "Dahlquist A-stability region";
    base.stability_region_evidence_present = true;
    base.operator_spectrum_coverage_present = true;
    base.operator_normality_evidence_present = true;

    AnalysisSummarySet summaries;
    auto negative_amplification = base;
    negative_amplification.time_scheme = "negative-amplification";
    negative_amplification.amplification_radius = -1.0;
    summaries.temporal_stability.push_back(negative_amplification);

    auto nonfinite_amplification = base;
    nonfinite_amplification.time_scheme = "nonfinite-amplification";
    nonfinite_amplification.amplification_radius =
        -std::numeric_limits<Real>::infinity();
    summaries.temporal_stability.push_back(nonfinite_amplification);

    auto negative_cfl = base;
    negative_cfl.time_scheme = "negative-cfl";
    negative_cfl.stability_class = TemporalStabilityClass::ConditionallyStable;
    negative_cfl.cfl_estimate_present = true;
    negative_cfl.cfl_estimate = -0.25;
    summaries.temporal_stability.push_back(negative_cfl);

    auto nan_cfl = negative_cfl;
    nan_cfl.time_scheme = "nan-cfl";
    nan_cfl.cfl_estimate = std::numeric_limits<Real>::quiet_NaN();
    summaries.temporal_stability.push_back(nan_cfl);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_EQ(claims.size(), 4u);
    for (const auto* claim : claims) {
        EXPECT_EQ(claim->status, PropertyStatus::Violated);
        ASSERT_TRUE(claim->certification_class.has_value());
        EXPECT_EQ(*claim->certification_class,
                  CertificationClass::Violated);
    }
}

TEST(AnalysisEvidenceContracts, ConditionalTemporalStabilityUsesAcceptedCflBound)
{
    TemporalStabilitySummary missing_bound;
    missing_bound.time_scheme = "missing-accepted-cfl";
    missing_bound.stability_class = TemporalStabilityClass::ConditionallyStable;
    missing_bound.stability_metadata_present = true;
    missing_bound.amplification_radius_present = true;
    missing_bound.amplification_radius = 0.9;
    missing_bound.stability_theorem_id =
        "scheme-specific absolute stability region";
    missing_bound.stability_region_evidence_present = true;
    missing_bound.operator_spectrum_coverage_present = true;
    missing_bound.operator_normality_evidence_present = true;
    missing_bound.cfl_estimate_present = true;
    missing_bound.cfl_estimate = 0.5;
    missing_bound.cfl_derivation_metadata_present = true;
    missing_bound.cfl_bound_scope = "cell spectral-radius CFL";

    TemporalStabilitySummary certified = missing_bound;
    certified.time_scheme = "accepted-cfl-greater-than-one";
    addTemporalCflCertificate(certified, 1.5, 2.0);

    TemporalStabilitySummary violated = missing_bound;
    violated.time_scheme = "accepted-cfl-exceeded";
    addTemporalCflCertificate(violated, 0.75, 0.5);

    AnalysisSummarySet summaries;
    summaries.temporal_stability.push_back(missing_bound);
    summaries.temporal_stability.push_back(certified);
    summaries.temporal_stability.push_back(violated);
    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer");
    ASSERT_EQ(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Unknown);
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
    addEntropyQuantitativeEvidence(certified);
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

TEST(AnalysisEvidenceContracts, EnergyEntropyCertificationRequiresQuantitativeBounds)
{
    AnalysisSummarySet summaries;
    EnergyEntropySummary energy;
    energy.energy_entropy_id = "energy-boolean-only";
    energy.law_kind = EnergyEntropyLawKind::Energy;
    energy.expected_production_sign = BalanceSignClass::Nonpositive;
    energy.balance_tolerance = 1.0e-8;
    energy.observed_discrete_balance = 0.0;
    energy.observed_production = -1.0e-9;
    energy.energy_functional_id = "quadratic-energy";
    energy.energy_norm_id = "mass-inner-product";
    energy.energy_entropy_theorem_id = "discrete energy identity";
    energy.energy_functional_metadata_present = true;
    energy.energy_norm_metadata_present = true;
    energy.energy_positivity_evidence_present = true;
    energy.energy_coercivity_evidence_present = true;
    energy.discrete_dissipation_identity_evidence_present = true;
    energy.boundary_source_energy_accounting_present = true;
    summaries.energy_entropy.push_back(energy);

    EnergyEntropySummary entropy;
    entropy.energy_entropy_id = "entropy-boolean-only";
    entropy.law_kind = EnergyEntropyLawKind::Entropy;
    entropy.expected_production_sign = BalanceSignClass::Nonnegative;
    entropy.balance_tolerance = 1.0e-8;
    entropy.observed_discrete_balance = 0.0;
    entropy.observed_production = 0.0;
    entropy.energy_entropy_theorem_id = "Tadmor entropy-stability theorem";
    entropy.convex_entropy_metadata_present = true;
    entropy.entropy_variables_metadata_present = true;
    entropy.entropy_flux_metadata_present = true;
    entropy.entropy_dissipation_metadata_present = true;
    entropy.boundary_source_entropy_metadata_present = true;
    summaries.energy_entropy.push_back(entropy);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto energy_claims = claimsFrom(
        report, PropertyKind::EnergyStability,
        "EnergyEntropyLawAnalyzer");
    ASSERT_EQ(energy_claims.size(), 1u);
    EXPECT_EQ(energy_claims.front()->status, PropertyStatus::Unknown);
    ASSERT_TRUE(energy_claims.front()->certification_class.has_value());
    EXPECT_EQ(*energy_claims.front()->certification_class,
              CertificationClass::NotCertified);

    const auto entropy_claims = claimsFrom(
        report, PropertyKind::EntropyStability,
        "EnergyEntropyLawAnalyzer");
    ASSERT_EQ(entropy_claims.size(), 1u);
    EXPECT_EQ(entropy_claims.front()->status, PropertyStatus::Unknown);
    ASSERT_TRUE(entropy_claims.front()->certification_class.has_value());
    EXPECT_EQ(*entropy_claims.front()->certification_class,
              CertificationClass::NotCertified);
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
    addSemiExplicitDAECertificationScope(evidence);
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

TEST(AnalysisEvidenceContracts, SemiExplicitDAECertificationRequiresScopeMetadata)
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
    evidence.system_id = "semi-explicit-no-scope";
    evidence.variables = {u, lambda};
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
    const auto* dae = firstFrom(
        report, PropertyKind::DifferentialAlgebraicStructure,
        "DAEStructureAnalyzer");
    ASSERT_NE(dae, nullptr);
    EXPECT_EQ(dae->status, PropertyStatus::Likely);
    ASSERT_TRUE(dae->certification_class.has_value());
    EXPECT_EQ(*dae->certification_class, CertificationClass::NotCertified);
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
    addSpectralQuantitativeEvidence(spectral_certified);
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
    quadrature_certified.mapped_integrand_metadata_present = true;
    quadrature_certified.basis_degree_metadata_present = true;
    quadrature_certified.geometry_jacobian_degree_metadata_present = true;
    quadrature_certified.tensor_contraction_metadata_present = true;
    quadrature_certified.component_coverage_metadata_present = true;
    quadrature_certified.quadrature_theorem_id =
        "Strang-Fix exact polynomial quadrature condition";
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

TEST(AnalysisEvidenceContracts, SpectralCertificationRequiresNumericDiagnosticScope)
{
    AnalysisSummarySet summaries;
    SpectralStructureSummary boolean_only;
    boolean_only.block = scalarBlock("spectral-boolean-only");
    boolean_only.eigenproblem_declared = true;
    boolean_only.self_adjoint_evidence = true;
    boolean_only.compactness_evidence = true;
    boolean_only.operator_convergence_evidence = true;
    boolean_only.spectral_convergence_theorem_id =
        "Boffi spectral approximation theorem";
    summaries.spectral_structures.push_back(boolean_only);

    SpectralStructureSummary certified = boolean_only;
    certified.block = scalarBlock("spectral-quantitative");
    addSpectralQuantitativeEvidence(certified);
    summaries.spectral_structures.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::SpectralCorrectness,
        "SpectralSpuriousModeAnalyzer");
    ASSERT_EQ(claims.size(), 2u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Unknown);
    ASSERT_TRUE(claims[0]->certification_class.has_value());
    EXPECT_EQ(*claims[0]->certification_class,
              CertificationClass::Unknown);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claims[1]->certification_class.has_value());
    EXPECT_EQ(*claims[1]->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, GeometryQuadratureAndGclRejectInvalidScalars)
{
    AnalysisSummarySet summaries;

    MeshGeometryQualitySummary mesh;
    mesh.min_jacobian = std::numeric_limits<Real>::quiet_NaN();
    mesh.max_jacobian = 1.0;
    summaries.mesh_geometry_quality.push_back(mesh);

    QuadratureAdequacySummary quadrature;
    quadrature.block = scalarBlock("invalid-aliasing");
    quadrature.integrand_polynomial_degree = 2;
    quadrature.quadrature_exact_degree = 3;
    quadrature.affine_mapping_evidence_present = true;
    quadrature.polynomial_integrand_metadata_complete = true;
    quadrature.coefficient_degree_metadata_present = true;
    quadrature.mapped_integrand_metadata_present = true;
    quadrature.basis_degree_metadata_present = true;
    quadrature.geometry_jacobian_degree_metadata_present = true;
    quadrature.tensor_contraction_metadata_present = true;
    quadrature.component_coverage_metadata_present = true;
    quadrature.quadrature_theorem_id =
        "Strang-Fix exact polynomial quadrature condition";
    quadrature.aliasing_indicator =
        std::numeric_limits<Real>::quiet_NaN();
    quadrature.aliasing_tolerance = 1.0e-8;
    summaries.quadrature_adequacy.push_back(quadrature);

    MovingDomainSummary moving;
    moving.min_geometric_jacobian = 1.0;
    moving.max_geometric_jacobian = 0.5;
    moving.geometric_conservation_residual = 0.0;
    moving.geometric_conservation_tolerance = 1.0e-8;
    moving.mesh_velocity_metadata_present = true;
    moving.time_integration_metadata_present = true;
    moving.remap_metadata_present = true;
    summaries.moving_domain.push_back(moving);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* mesh_claim = firstFrom(
        report, PropertyKind::MeshGeometryValidity,
        "MeshGeometryAnalyzer");
    ASSERT_NE(mesh_claim, nullptr);
    EXPECT_EQ(mesh_claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(mesh_claim->certification_class.has_value());
    EXPECT_EQ(*mesh_claim->certification_class,
              CertificationClass::Violated);

    const auto* quadrature_claim = firstForBlock(
        report, PropertyKind::QuadratureAdequacy,
        "QuadratureAdequacyAnalyzer", "invalid-aliasing");
    ASSERT_NE(quadrature_claim, nullptr);
    EXPECT_EQ(quadrature_claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(quadrature_claim->certification_class.has_value());
    EXPECT_EQ(*quadrature_claim->certification_class,
              CertificationClass::Violated);

    const auto* moving_claim = firstFrom(
        report, PropertyKind::GeometricConservation,
        "PreservationStructureAnalyzer");
    ASSERT_NE(moving_claim, nullptr);
    EXPECT_EQ(moving_claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(moving_claim->certification_class.has_value());
    EXPECT_EQ(*moving_claim->certification_class,
              CertificationClass::Violated);
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
    addCoupledContractionCertificate(certified);
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

TEST(AnalysisEvidenceContracts, CoupledNonnormalRouteNeedsQuantitativeBound)
{
    const auto a = VariableKey::field(0);
    const auto b = VariableKey::field(1);

    AnalysisSummarySet summaries;
    CoupledSystemStabilitySummary missing_bound;
    missing_bound.coupling_group = "nonnormal-missing-bound";
    missing_bound.variables = {a, b};
    missing_bound.monolithic_coupling = true;
    missing_bound.coupling_tolerance = 1.0e-8;
    missing_bound.coupling_tolerance_present = true;
    missing_bound.exchange_residual = 0.0;
    missing_bound.exchange_residual_present = true;
    missing_bound.constraint_drift_norm = 0.0;
    missing_bound.constraint_drift_present = true;
    missing_bound.coupled_operator_stability_evidence_present = true;
    missing_bound.nonnormal_coupling_bound_present = true;
    summaries.coupled_system_stability.push_back(missing_bound);

    CoupledSystemStabilitySummary certified = missing_bound;
    certified.coupling_group = "nonnormal-bounded";
    certified.nonnormal_coupling_growth_bound = 1.5;
    certified.accepted_nonnormal_coupling_growth_bound = 2.0;
    certified.accepted_nonnormal_coupling_growth_bound_present = true;
    addCoupledCommonScope(certified);
    summaries.coupled_system_stability.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto coupled = claimsFrom(
        report, PropertyKind::CoupledSystemStructure,
        "CoupledSystemStabilityAnalyzer");
    ASSERT_EQ(coupled.size(), 2u);
    EXPECT_EQ(coupled[0]->status, PropertyStatus::Violated);
    ASSERT_TRUE(coupled[0]->certification_class.has_value());
    EXPECT_EQ(*coupled[0]->certification_class,
              CertificationClass::Violated);
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
    addCoupledEnergyCertificate(certified);
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
    coeff.min_eigenvalue = 0.5;
    coeff.max_eigenvalue = 2.0;
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
    matrix.max_abs_row_sum = 1.0;
    matrix.structurally_symmetric = true;
    matrix.numerically_symmetric = true;
    matrix.symmetry_evidence_complete = true;
    matrix.cholesky_factorization_succeeded = true;
    matrix.m_matrix_certification_evidence = true;
    matrix.stieltjes_matrix_evidence = true;
    matrix.m_matrix_theorem_id = "stieltjes-spd-z";
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

TEST(AnalysisEvidenceContracts, DmpRequiresCoefficientCoverageForAllMatrixVariables)
{
    const auto u = VariableKey::field(0);
    const auto w = VariableKey::field(1);
    OperatorBlockId coupled_block;
    coupled_block.test_variables = {u, w};
    coupled_block.trial_variables = {u, w};
    coupled_block.operator_tag = "coupled-diffusion";
    coupled_block.domain = DomainKind::Cell;

    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    ctx.addFieldDescriptor(h1Field(1, 1, FieldType::Scalar, 1, "w"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        u, "coupled-diffusion", "evidence-contract"));

    CoefficientPropertySummary partial_coefficient;
    partial_coefficient.block = scalarBlock("coupled-diffusion");
    partial_coefficient.positivity = PositivityClass::Positive;
    partial_coefficient.min_eigenvalue = 0.5;
    partial_coefficient.max_eigenvalue = 2.0;
    partial_coefficient.coefficient_region_coverage_complete = true;
    partial_coefficient.quadrature_point_coverage_complete = true;
    partial_coefficient.lower_bound_valid_for_all_samples = true;
    partial_coefficient.tolerance_metadata_present = true;

    DiscreteMatrixSummary matrix;
    matrix.block = coupled_block;
    matrix.rows = 4;
    matrix.cols = 4;
    matrix.square = true;
    matrix.sign_evidence_complete = true;
    matrix.row_sum_evidence_complete = true;
    matrix.sign_tolerance = 1.0e-12;
    matrix.row_sum_tolerance = 1.0e-12;
    matrix.diagonal_count = 4;
    matrix.offdiag_count = 12;
    matrix.negative_offdiag_count = 12;
    matrix.min_row_sum = 0.0;
    matrix.max_row_sum = 1.0;
    matrix.max_abs_row_sum = 1.0;
    matrix.scanned_row_count = 4;
    matrix.expected_row_count = 4;
    matrix.scanned_entry_count = 16;
    matrix.structurally_symmetric = true;
    matrix.numerically_symmetric = true;
    matrix.symmetry_evidence_complete = true;
    matrix.cholesky_factorization_succeeded = true;
    matrix.m_matrix_certification_evidence = true;
    matrix.stieltjes_matrix_evidence = true;
    matrix.m_matrix_theorem_id = "stieltjes-spd-z";
    matrix.dmp_applicability_evidence = true;
    matrix.dmp_rhs_sign_evidence = true;

    AnalysisSummarySet summaries;
    summaries.coefficient_properties.push_back(partial_coefficient);
    summaries.discrete_matrices.push_back(matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    const auto* m_matrix = firstForBlock(
        report, PropertyKind::MMatrixStructure,
        "DiscreteMonotonicityAnalyzer", "coupled-diffusion");
    ASSERT_NE(m_matrix, nullptr);
    EXPECT_EQ(m_matrix->status, PropertyStatus::Exact);
    ASSERT_TRUE(m_matrix->certification_class.has_value());
    EXPECT_EQ(*m_matrix->certification_class,
              CertificationClass::Certified);

    const auto* dmp = firstForBlock(
        report, PropertyKind::DiscreteMaximumPrinciple,
        "DiscreteMonotonicityAnalyzer", "coupled-diffusion");
    ASSERT_NE(dmp, nullptr);
    EXPECT_EQ(dmp->status, PropertyStatus::Unknown);
    ASSERT_TRUE(dmp->certification_class.has_value());
    EXPECT_EQ(*dmp->certification_class,
              CertificationClass::Unknown);
    ASSERT_FALSE(dmp->evidence.empty());
    EXPECT_NE(dmp->evidence.front().description.find(
                  "scoped positive coefficient evidence"),
              std::string::npos);
}

TEST(AnalysisEvidenceContracts, DiscreteMatrixNonfiniteEvidenceCannotCertifyDmp)
{
    const auto scalar = VariableKey::field(0);
    auto matrix_block = scalarBlock("nonfinite-diffusion");

    CoefficientPropertySummary coeff;
    coeff.block = matrix_block;
    coeff.positivity = PositivityClass::Positive;
    coeff.min_eigenvalue = 0.5;
    coeff.max_eigenvalue = 2.0;
    coeff.coefficient_region_coverage_complete = true;
    coeff.quadrature_point_coverage_complete = true;
    coeff.lower_bound_valid_for_all_samples = true;
    coeff.tolerance_metadata_present = true;

    DiscreteMatrixSummary matrix;
    matrix.block = matrix_block;
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
    matrix.max_abs_row_sum = 1.0;
    matrix.scanned_row_count = 2;
    matrix.expected_row_count = 2;
    matrix.scanned_entry_count = 4;
    matrix.nonfinite_entry_count = 1;
    matrix.nonfinite_row_sum_count = 1;
    matrix.structurally_symmetric = true;
    matrix.numerically_symmetric = true;
    matrix.symmetry_evidence_complete = true;
    matrix.cholesky_factorization_succeeded = true;
    matrix.m_matrix_certification_evidence = true;
    matrix.stieltjes_matrix_evidence = true;
    matrix.m_matrix_theorem_id = "stieltjes-spd-z";
    matrix.dmp_applicability_evidence = true;
    matrix.dmp_rhs_sign_evidence = true;

    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "nonfinite-diffusion", "evidence-contract"));
    AnalysisSummarySet summaries;
    summaries.coefficient_properties.push_back(coeff);
    summaries.discrete_matrices.push_back(matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    const auto report = analyze(std::move(ctx));
    const auto* z_claim = firstForBlock(
        report, PropertyKind::ZMatrixStructure,
        "DiscreteMonotonicityAnalyzer", "nonfinite-diffusion");
    ASSERT_NE(z_claim, nullptr);
    EXPECT_EQ(z_claim->status, PropertyStatus::Unknown);
    EXPECT_FALSE(hasCertifiedClaim(report,
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
    EXPECT_FALSE(hasCertifiedClaim(theorem_report,
                                   PropertyKind::MMatrixStructure));

    ProblemAnalysisContext stieltjes_ctx;
    stieltjes_ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    stieltjes_ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "diffusion", "evidence-contract"));
    AnalysisSummarySet stieltjes_summaries;
    matrix.symmetry_evidence_complete = true;
    matrix.structurally_symmetric = true;
    matrix.numerically_symmetric = true;
    matrix.cholesky_factorization_succeeded = true;
    matrix.m_matrix_theorem_id = "stieltjes-spd-z";
    stieltjes_summaries.discrete_matrices.push_back(matrix);
    stieltjes_ctx.setAnalysisSummaries(std::move(stieltjes_summaries));

    const auto stieltjes_report = analyze(std::move(stieltjes_ctx));
    EXPECT_TRUE(hasCertifiedClaim(stieltjes_report,
                                  PropertyKind::MMatrixStructure));
}

TEST(AnalysisEvidenceContracts, MMatrixRoutesRequireRouteSpecificMetadata)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    for (const auto* tag : {"inverse-missing",
                            "inverse-certified",
                            "dd-missing",
                            "dd-certified"}) {
        ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
            scalar, tag, "evidence-contract"));
    }

    AnalysisSummarySet summaries;
    auto inverse_missing = certifiedSignPatternMatrix("inverse-missing");
    inverse_missing.m_matrix_theorem_id = "inverse-positive M-matrix theorem";
    inverse_missing.inverse_positivity_evidence = true;
    summaries.discrete_matrices.push_back(inverse_missing);

    auto inverse_certified = inverse_missing;
    inverse_certified.block = scalarBlock("inverse-certified");
    inverse_certified.inverse_positivity_metadata_present = true;
    summaries.discrete_matrices.push_back(inverse_certified);

    auto dd_missing = certifiedSignPatternMatrix("dd-missing");
    dd_missing.m_matrix_theorem_id =
        "irreducible weak diagonal-dominance M-matrix theorem";
    dd_missing.irreducible_diagonal_dominance_evidence = true;
    summaries.discrete_matrices.push_back(dd_missing);

    auto dd_certified = dd_missing;
    dd_certified.block = scalarBlock("dd-certified");
    dd_certified.diagonal_dominance_evidence_complete = true;
    dd_certified.irreducibility_evidence_present = true;
    summaries.discrete_matrices.push_back(dd_certified);

    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* inverse_missing_claim = firstForBlock(
        report, PropertyKind::MMatrixStructure,
        "DiscreteMonotonicityAnalyzer", "inverse-missing");
    ASSERT_NE(inverse_missing_claim, nullptr);
    ASSERT_TRUE(inverse_missing_claim->certification_class.has_value());
    EXPECT_NE(*inverse_missing_claim->certification_class,
              CertificationClass::Certified);

    const auto* inverse_certified_claim = firstForBlock(
        report, PropertyKind::MMatrixStructure,
        "DiscreteMonotonicityAnalyzer", "inverse-certified");
    ASSERT_NE(inverse_certified_claim, nullptr);
    EXPECT_EQ(inverse_certified_claim->status, PropertyStatus::Exact);
    ASSERT_TRUE(inverse_certified_claim->certification_class.has_value());
    EXPECT_EQ(*inverse_certified_claim->certification_class,
              CertificationClass::Certified);

    const auto* dd_missing_claim = firstForBlock(
        report, PropertyKind::MMatrixStructure,
        "DiscreteMonotonicityAnalyzer", "dd-missing");
    ASSERT_NE(dd_missing_claim, nullptr);
    ASSERT_TRUE(dd_missing_claim->certification_class.has_value());
    EXPECT_NE(*dd_missing_claim->certification_class,
              CertificationClass::Certified);

    const auto* dd_certified_claim = firstForBlock(
        report, PropertyKind::MMatrixStructure,
        "DiscreteMonotonicityAnalyzer", "dd-certified");
    ASSERT_NE(dd_certified_claim, nullptr);
    EXPECT_EQ(dd_certified_claim->status, PropertyStatus::Exact);
    ASSERT_TRUE(dd_certified_claim->certification_class.has_value());
    EXPECT_EQ(*dd_certified_claim->certification_class,
              CertificationClass::Certified);
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

TEST(AnalysisEvidenceContracts, HDivL2MixedPairNeedsExplicitCertifiedEvidence)
{
    const auto flux = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);

    auto add_mixed_pair = [&](ProblemAnalysisContext& ctx) {
        ctx.addFieldDescriptor(hDivField(1, 1, "hdiv-flux"));
        ctx.addFieldDescriptor(l2Field(0, 0, "l2-multiplier"));
        ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
            flux, "flux-block", "evidence-contract"));
        ctx.addContribution(ContributionDescriptor::constraintPairDesc(
            flux, multiplier, "mixed-pair", "divergence-l2-pair",
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
    ASSERT_TRUE(heuristic->space_compatibility_class.has_value());
    EXPECT_EQ(*heuristic->space_compatibility_class,
              SpaceCompatibilityClass::Unknown);

    ProblemAnalysisContext certified_ctx;
    add_mixed_pair(certified_ctx);
    AnalysisSummarySet summaries;
    InfSupPairCertificationSummary certified_pair;
    certified_pair.block.operator_tag = "divergence-l2-pair";
    certified_pair.block.test_variables = {flux, multiplier};
    certified_pair.block.trial_variables = {flux, multiplier};
    certified_pair.primal_variable = flux;
    certified_pair.multiplier_variable = multiplier;
    certified_pair.pair_family = "documented HDiv/L2 saddle pair";
    certified_pair.primal_polynomial_order = 1;
    certified_pair.multiplier_polynomial_order = 0;
    certified_pair.primal_space_family = SpaceFamily::HDiv;
    certified_pair.multiplier_space_family = SpaceFamily::L2;
    certified_pair.known_stable_pair = true;
    certified_pair.mesh_assumption_evidence_present = true;
    certified_pair.domain_assumption_evidence_present = true;
    certified_pair.boundary_condition_scope_present = true;
    certified_pair.beta_lower_bound_present = true;
    certified_pair.beta_lower_bound = 0.1;
    certified_pair.inf_sup_theorem_id = "documented HDiv/L2 Fortin theorem";
    summaries.inf_sup_pair_certifications.push_back(certified_pair);
    certified_ctx.setAnalysisSummaries(std::move(summaries));

    const auto certified_report = analyze(std::move(certified_ctx));
    bool saw_certified_hdiv_l2 = false;
    for (const auto& claim : certified_report.claims) {
        if (claim.kind == PropertyKind::SpaceCompatibility &&
            claim.claim_origin == "SpaceCompatibilityAnalyzer" &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified) {
            saw_certified_hdiv_l2 = true;
            EXPECT_EQ(claim.status, PropertyStatus::Preserved);
            ASSERT_TRUE(claim.space_compatibility_class.has_value());
            EXPECT_EQ(*claim.space_compatibility_class,
                      SpaceCompatibilityClass::Compatible);
        }
    }
    EXPECT_TRUE(saw_certified_hdiv_l2);
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

TEST(AnalysisEvidenceContracts, CompatibleComplexCertificationNeedsBoundedCochainProjection)
{
    AnalysisSummarySet summaries;
    CompatibleComplexSummary commuting_only;
    commuting_only.complex_id = "commuting-only";
    commuting_only.variables = {VariableKey::field(0), VariableKey::field(1)};
    commuting_only.exact_sequence_compatible = true;
    commuting_only.trace_sequence_compatible = true;
    commuting_only.commuting_projection_available = true;
    summaries.compatible_complexes.push_back(commuting_only);

    CompatibleComplexSummary invalid_bound = commuting_only;
    invalid_bound.complex_id = "nonfinite-cochain";
    invalid_bound.compatible_complex_theorem_id =
        "Arnold-Falk-Winther bounded cochain projection";
    invalid_bound.bounded_cochain_projection_evidence_present = true;
    invalid_bound.projection_bound_present = true;
    invalid_bound.projection_bound =
        std::numeric_limits<Real>::infinity();
    invalid_bound.projection_stability_metadata_present = true;
    invalid_bound.mesh_family_scope_present = true;
    invalid_bound.shape_regular_mesh_evidence_present = true;
    summaries.compatible_complexes.push_back(invalid_bound);

    CompatibleComplexSummary certified = commuting_only;
    certified.complex_id = "bounded-cochain";
    certified.compatible_complex_theorem_id =
        "Arnold-Falk-Winther bounded cochain projection";
    certified.bounded_cochain_projection_evidence_present = true;
    certified.projection_bound_present = true;
    certified.projection_bound = 2.0;
    certified.projection_stability_metadata_present = true;
    certified.mesh_family_scope_present = true;
    certified.shape_regular_mesh_evidence_present = true;
    summaries.compatible_complexes.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::CompatibleComplexStructure,
        "SpaceCompatibilityAnalyzer");
    ASSERT_GE(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Likely);
    ASSERT_TRUE(claims[0]->certification_class.has_value());
    EXPECT_EQ(*claims[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Likely);
    ASSERT_TRUE(claims[1]->certification_class.has_value());
    EXPECT_EQ(*claims[1]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(claims[2]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claims[2]->certification_class.has_value());
    EXPECT_EQ(*claims[2]->certification_class,
              CertificationClass::Certified);
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

TEST(AnalysisEvidenceContracts, InitialCompatibilityRequiresDeclaredToleranceAndCheckedScope)
{
    AnalysisSummarySet missing_summaries;
    InitialCompatibilitySummary missing;
    missing_summaries.initial_compatibility.push_back(missing);
    ProblemAnalysisContext missing_ctx;
    missing_ctx.setAnalysisSummaries(std::move(missing_summaries));
    const auto missing_report = analyze(std::move(missing_ctx));
    const auto* missing_claim = firstFrom(
        missing_report, PropertyKind::InitialDataCompatibility,
        "CompatibilityAnalyzer");
    ASSERT_NE(missing_claim, nullptr);
    EXPECT_EQ(missing_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(missing_claim->certification_class.has_value());
    EXPECT_EQ(*missing_claim->certification_class,
              CertificationClass::NotCertified);

    AnalysisSummarySet invalid_numeric_summaries;
    InitialCompatibilitySummary invalid_numeric;
    invalid_numeric.initial_constraint_residual =
        std::numeric_limits<Real>::quiet_NaN();
    invalid_numeric.initial_boundary_residual = 0.0;
    invalid_numeric.residual_tolerance = 1.0e-8;
    invalid_numeric.residual_tolerance_declared = true;
    invalid_numeric.compatibility_scope = "dae-algebraic";
    invalid_numeric.algebraic_constraint_metadata_present = true;
    invalid_numeric.checked_constraint_family_count = 1;
    invalid_numeric_summaries.initial_compatibility.push_back(
        invalid_numeric);
    ProblemAnalysisContext invalid_numeric_ctx;
    invalid_numeric_ctx.setAnalysisSummaries(
        std::move(invalid_numeric_summaries));
    const auto invalid_numeric_report = analyze(
        std::move(invalid_numeric_ctx));
    const auto* invalid_numeric_claim = firstFrom(
        invalid_numeric_report, PropertyKind::InitialDataCompatibility,
        "CompatibilityAnalyzer");
    ASSERT_NE(invalid_numeric_claim, nullptr);
    EXPECT_EQ(invalid_numeric_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(invalid_numeric_claim->certification_class.has_value());
    EXPECT_EQ(*invalid_numeric_claim->certification_class,
              CertificationClass::NotCertified);

    AnalysisSummarySet certified_summaries;
    InitialCompatibilitySummary certified;
    certified.initial_constraint_residual = 1.0e-12;
    certified.initial_boundary_residual = 2.0e-12;
    certified.residual_tolerance = 1.0e-8;
    certified.residual_tolerance_declared = true;
    certified.compatibility_scope = "dae-algebraic-and-boundary";
    certified.algebraic_constraint_metadata_present = true;
    certified.boundary_constraint_metadata_present = true;
    certified.checked_constraint_family_count = 1;
    certified.checked_boundary_condition_count = 1;
    certified_summaries.initial_compatibility.push_back(certified);
    ProblemAnalysisContext certified_ctx;
    certified_ctx.setAnalysisSummaries(std::move(certified_summaries));
    const auto certified_report = analyze(std::move(certified_ctx));
    const auto* certified_claim = firstFrom(
        certified_report, PropertyKind::InitialDataCompatibility,
        "CompatibilityAnalyzer");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, InitialInvariantCompatibilityRequiresCoverageEvidence)
{
    const auto u = VariableKey::field(0);
    AnalysisSummarySet summaries;

    InitialCompatibilitySummary metadata_only;
    metadata_only.residual_tolerance = 1.0e-8;
    metadata_only.residual_tolerance_declared = true;
    metadata_only.compatibility_scope = "invariant-domain-initial-state";
    metadata_only.invariant_domain_metadata_present = true;
    summaries.initial_compatibility.push_back(metadata_only);

    InitialCompatibilitySummary certified = metadata_only;
    certified.invariant_set_id = "positive-density-pressure";
    certified.invariant_domain_variables = {u};
    certified.checked_invariant_state_count = 16;
    certified.invariant_domain_admissibility_residual_present = true;
    certified.invariant_domain_admissibility_residual = 0.0;
    summaries.initial_compatibility.push_back(certified);

    InitialCompatibilitySummary violated = certified;
    violated.invariant_set_id = "violated-positive-density-pressure";
    violated.invariant_domain_initial_violation_count = 1;
    summaries.initial_compatibility.push_back(violated);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::InitialDataCompatibility,
        "CompatibilityAnalyzer");
    ASSERT_EQ(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Unknown);
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
    addSemiExplicitDAECertificationScope(evidence);
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
    addInvariantDomainCflEvidence(certified);
    certified.invariant_domain_theorem_id =
        "Guermond-Popov convex limiting invariant domain theorem";
    summaries.invariant_domains.push_back(certified);

    InvariantDomainSummary invalid_bounds = certified;
    invalid_bounds.invariant_set_id = "reversed-bounds";
    invalid_bounds.upper_bound_active = true;
    invalid_bounds.lower_bound = 1.0;
    invalid_bounds.upper_bound = 0.0;
    summaries.invariant_domains.push_back(invalid_bounds);

    InvariantDomainSummary cfl_exceeds_bound = certified;
    cfl_exceeds_bound.invariant_set_id = "cfl-exceeds-bound";
    cfl_exceeds_bound.cfl_estimate = 0.75;
    cfl_exceeds_bound.accepted_cfl_bound = 0.5;
    summaries.invariant_domains.push_back(cfl_exceeds_bound);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto invariant = claimsFrom(
        report, PropertyKind::InvariantDomainPreservation,
        "PreservationStructureAnalyzer");
    ASSERT_EQ(invariant.size(), 4u);
    EXPECT_EQ(invariant[0]->status, PropertyStatus::Unknown);
    EXPECT_EQ(invariant[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(invariant[1]->certification_class.has_value());
    EXPECT_EQ(*invariant[1]->certification_class,
              CertificationClass::Certified);
    EXPECT_EQ(invariant[2]->status, PropertyStatus::Violated);
    ASSERT_TRUE(invariant[2]->certification_class.has_value());
    EXPECT_EQ(*invariant[2]->certification_class,
              CertificationClass::Violated);
    EXPECT_EQ(invariant[3]->status, PropertyStatus::Violated);
    ASSERT_TRUE(invariant[3]->certification_class.has_value());
    EXPECT_EQ(*invariant[3]->certification_class,
              CertificationClass::Violated);
}

TEST(AnalysisEvidenceContracts, TransferCompatibilityRequiresQuantitativeMortarEvidence)
{
    TransferOperatorSummary base;
    base.interface_pair_id = "nonmatching-interface";
    base.projection_space_id = "mortar-projection";
    base.residual_tolerance = 1.0e-10;
    base.conservation_residual = 0.0;
    base.constant_preservation_residual = 0.0;
    base.rank_metadata_present = true;
    base.interface_scope_metadata_present = true;
    base.projection_consistency_metadata_present = true;
    base.mortar_inf_sup_or_dual_consistency_metadata_present = true;
    base.interface_mass_conditioning_metadata_present = true;
    base.action_reaction_flux_metadata_present = true;

    AnalysisSummarySet summaries;
    summaries.transfer_operators.push_back(base);

    auto certified = base;
    certified.interface_pair_id = "certified-nonmatching-interface";
    addTransferQuantitativeCertificate(certified);
    summaries.transfer_operators.push_back(certified);

    auto bad_mass_condition = certified;
    bad_mass_condition.interface_pair_id = "bad-interface-mass";
    bad_mass_condition.interface_mass_condition_number = 50.0;
    bad_mass_condition.accepted_interface_mass_condition_bound = 20.0;
    summaries.transfer_operators.push_back(bad_mass_condition);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::TransferOperatorCompatibility,
        "PreservationStructureAnalyzer");
    ASSERT_EQ(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Unknown);
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

TEST(AnalysisEvidenceContracts, MovingDomainCertificationRequiresDgclScope)
{
    AnalysisSummarySet summaries;

    MovingDomainSummary residual_only;
    residual_only.mesh_revision = 1;
    residual_only.min_geometric_jacobian = 0.75;
    residual_only.max_geometric_jacobian = 1.25;
    residual_only.geometric_conservation_tolerance_declared = true;
    residual_only.geometric_conservation_residual = 0.0;
    residual_only.geometric_conservation_tolerance = 1.0e-8;
    residual_only.mesh_velocity_metadata_present = true;
    residual_only.time_integration_metadata_present = true;
    residual_only.remap_metadata_present = true;
    summaries.moving_domain.push_back(residual_only);

    MovingDomainSummary certified = residual_only;
    certified.mesh_revision = 2;
    addMovingDomainGclEvidence(certified);
    summaries.moving_domain.push_back(certified);

    MovingDomainSummary free_stream_violation = certified;
    free_stream_violation.mesh_revision = 3;
    free_stream_violation.free_stream_preservation_residual = 1.0e-4;
    summaries.moving_domain.push_back(free_stream_violation);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::GeometricConservation,
        "PreservationStructureAnalyzer");
    ASSERT_EQ(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Unknown);
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
    certified.min_eigenvalue = 0.5;
    certified.max_eigenvalue = 2.0;
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

TEST(AnalysisEvidenceContracts, CoefficientPositivityRejectsInvalidEigenvalueBounds)
{
    AnalysisSummarySet summaries;

    CoefficientPropertySummary nonfinite;
    nonfinite.coefficient = "nonfinite-positive";
    nonfinite.block = scalarBlock("nonfinite-positive");
    nonfinite.positivity = PositivityClass::Positive;
    nonfinite.min_eigenvalue =
        std::numeric_limits<Real>::quiet_NaN();
    nonfinite.max_eigenvalue = 2.0;
    nonfinite.coefficient_region_coverage_complete = true;
    nonfinite.quadrature_point_coverage_complete = true;
    nonfinite.lower_bound_valid_for_all_samples = true;
    nonfinite.tolerance_metadata_present = true;
    summaries.coefficient_properties.push_back(nonfinite);

    CoefficientPropertySummary contradicted = nonfinite;
    contradicted.coefficient = "negative-positive";
    contradicted.block = scalarBlock("negative-positive");
    contradicted.min_eigenvalue = -0.25;
    contradicted.max_eigenvalue = 2.0;
    summaries.coefficient_properties.push_back(contradicted);

    CoefficientPropertySummary certified = contradicted;
    certified.coefficient = "valid-positive";
    certified.block = scalarBlock("valid-positive");
    certified.min_eigenvalue = 0.5;
    certified.max_eigenvalue = 2.0;
    summaries.coefficient_properties.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* nonfinite_operator = firstForCoefficient(
        report, "OperatorClassAnalyzer", "nonfinite-positive");
    ASSERT_NE(nonfinite_operator, nullptr);
    EXPECT_EQ(nonfinite_operator->status, PropertyStatus::Likely);
    ASSERT_TRUE(nonfinite_operator->certification_class.has_value());
    EXPECT_EQ(*nonfinite_operator->certification_class,
              CertificationClass::NotCertified);

    const auto* contradicted_operator = firstForCoefficient(
        report, "OperatorClassAnalyzer", "negative-positive");
    ASSERT_NE(contradicted_operator, nullptr);
    EXPECT_EQ(contradicted_operator->status, PropertyStatus::Violated);
    ASSERT_TRUE(contradicted_operator->certification_class.has_value());
    EXPECT_EQ(*contradicted_operator->certification_class,
              CertificationClass::Violated);

    const auto* contradicted_constitutive = firstForCoefficient(
        report, "CoefficientConstitutiveAnalyzer", "negative-positive");
    ASSERT_NE(contradicted_constitutive, nullptr);
    EXPECT_EQ(contradicted_constitutive->status, PropertyStatus::Violated);
    ASSERT_TRUE(contradicted_constitutive->certification_class.has_value());
    EXPECT_EQ(*contradicted_constitutive->certification_class,
              CertificationClass::Violated);

    const auto* certified_operator = firstForCoefficient(
        report, "OperatorClassAnalyzer", "valid-positive");
    ASSERT_NE(certified_operator, nullptr);
    EXPECT_EQ(certified_operator->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_operator->certification_class.has_value());
    EXPECT_EQ(*certified_operator->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, ParameterRobustnessRequiresTheoremNormRangesAndUniformConstant)
{
    AnalysisSummarySet summaries;
    CoefficientPropertySummary partial;
    partial.coefficient = "partial-robust";
    partial.positivity = PositivityClass::Positive;
    partial.anisotropy_ratio = 1.0e4;
    partial.contrast_ratio = 1.0e5;
    partial.robustness_certificate_present = true;
    partial.robustness_certificate_scope = "high-contrast diffusion";
    summaries.coefficient_properties.push_back(partial);

    CoefficientPropertySummary certified = partial;
    certified.coefficient = "certified-robust";
    certified.robustness_theorem_id =
        "parameter-uniform preconditioner equivalence theorem";
    certified.robustness_norm_id = "energy norm";
    certified.robustness_norm_metadata_present = true;
    certified.robustness_parameter_range_scope = "contrast in [1,1e6]";
    certified.robustness_parameter_range_metadata_present = true;
    certified.robustness_mesh_family_scope = "shape-regular simplicial meshes";
    certified.robustness_mesh_family_metadata_present = true;
    certified.robustness_uniform_constant_present = true;
    certified.robustness_uniform_constant = 8.0;
    summaries.coefficient_properties.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* partial_claim =
        firstRobustnessForCoefficient(report, "partial-robust");
    ASSERT_NE(partial_claim, nullptr);
    EXPECT_EQ(partial_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(partial_claim->certification_class.has_value());
    EXPECT_EQ(*partial_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* certified_claim =
        firstRobustnessForCoefficient(report, "certified-robust");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
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
    certified.spectral_equivalence_lower_bound = 0.25;
    certified.spectral_equivalence_upper_bound = 4.0;
    certified.preconditioner_equivalence_bounds_present = true;
    certified.preconditioner_equivalence_lower_bound = 0.2;
    certified.preconditioner_equivalence_upper_bound = 5.0;

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

    SchurComplementSummary unordered = certified;
    unordered.schur_id = "unordered-schur";
    unordered.block = scalarBlock("unordered-schur");
    unordered.spectral_equivalence_lower_bound = 4.0;
    unordered.spectral_equivalence_upper_bound = 0.25;
    const auto unordered_report = analyze(make_ctx(unordered));
    const auto* unordered_resolution = firstForBlock(
        unordered_report, PropertyKind::IndefiniteOperatorResolution,
        "MixedOperatorAnalyzer", "unordered-schur");
    ASSERT_NE(unordered_resolution, nullptr);
    EXPECT_EQ(unordered_resolution->status, PropertyStatus::Likely);
    ASSERT_TRUE(unordered_resolution->reduced_definiteness_class.has_value());
    EXPECT_EQ(*unordered_resolution->reduced_definiteness_class,
              CertificationClass::NotCertified);

    SchurComplementSummary nonfinite = certified;
    nonfinite.schur_id = "nonfinite-schur";
    nonfinite.block = scalarBlock("nonfinite-schur");
    nonfinite.spectral_equivalence_lower_bound =
        std::numeric_limits<Real>::infinity();
    const auto nonfinite_report = analyze(make_ctx(nonfinite));
    const auto* nonfinite_resolution = firstForBlock(
        nonfinite_report, PropertyKind::IndefiniteOperatorResolution,
        "MixedOperatorAnalyzer", "nonfinite-schur");
    ASSERT_NE(nonfinite_resolution, nullptr);
    EXPECT_EQ(nonfinite_resolution->status, PropertyStatus::Likely);
    ASSERT_TRUE(nonfinite_resolution->reduced_definiteness_class.has_value());
    EXPECT_EQ(*nonfinite_resolution->reduced_definiteness_class,
              CertificationClass::NotCertified);
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
    certified.steady_balance_scope = true;
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

TEST(AnalysisEvidenceContracts, InterfaceFluxResidualOnlyIsNotCertified)
{
    AnalysisSummarySet summaries;
    FluxBalanceSummary residual_only;
    residual_only.block = scalarBlock("interface-residual-only",
                                      DomainKind::InterfaceFace);
    residual_only.balance_group = "mass";
    residual_only.balance_tolerance = 1.0e-8;
    residual_only.local_residual_norm = 1.0e-12;
    summaries.flux_balances.push_back(residual_only);

    FluxBalanceSummary certified = residual_only;
    certified.block = scalarBlock("interface-certified",
                                  DomainKind::InterfaceFace);
    certified.symbolic_balance_evidence_present = true;
    certified.symbolic_balance_group = "mass";
    certified.flux_variable_metadata_present = true;
    certified.element_residual_evidence_present = true;
    certified.source_quadrature_consistency_present = true;
    certified.orientation_consistency_present = true;
    certified.boundary_flux_accounted_for = true;
    certified.steady_balance_scope = true;
    summaries.flux_balances.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* residual_claim = firstForBlock(
        report, PropertyKind::InterfaceCondition,
        "InterfaceValidationAnalyzer", "interface-residual-only");
    ASSERT_NE(residual_claim, nullptr);
    EXPECT_EQ(residual_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(residual_claim->certification_class.has_value());
    EXPECT_EQ(*residual_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* certified_claim = firstForBlock(
        report, PropertyKind::InterfaceCondition,
        "InterfaceValidationAnalyzer", "interface-certified");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, TransientConservationRequiresTimeUpdateBalance)
{
    FluxBalanceSummary transient;
    transient.block = scalarBlock("transient-balance");
    transient.balance_group = "mass";
    transient.symbolic_balance_group = "mass";
    transient.balance_tolerance = 1.0e-8;
    transient.local_residual_norm = 1.0e-12;
    transient.symbolic_balance_evidence_present = true;
    transient.flux_variable_metadata_present = true;
    transient.element_residual_evidence_present = true;
    transient.source_quadrature_consistency_present = true;
    transient.orientation_consistency_present = true;
    transient.boundary_flux_accounted_for = true;
    transient.transient_balance_scope = true;

    AnalysisSummarySet missing_summaries;
    missing_summaries.flux_balances.push_back(transient);
    ProblemAnalysisContext missing_ctx;
    missing_ctx.setAnalysisSummaries(std::move(missing_summaries));
    const auto missing_report = analyze(std::move(missing_ctx));
    const auto* missing_claim = firstForBlock(
        missing_report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer", "transient-balance");
    ASSERT_NE(missing_claim, nullptr);
    EXPECT_EQ(missing_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(missing_claim->certification_class.has_value());
    EXPECT_EQ(*missing_claim->certification_class,
              CertificationClass::NotCertified);

    transient.time_update_balance_present = true;
    AnalysisSummarySet certified_summaries;
    certified_summaries.flux_balances.push_back(transient);
    ProblemAnalysisContext certified_ctx;
    certified_ctx.setAnalysisSummaries(std::move(certified_summaries));
    const auto certified_report = analyze(std::move(certified_ctx));
    const auto* certified_claim = firstForBlock(
        certified_report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer", "transient-balance");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
}

TEST(AnalysisEvidenceContracts, InvalidFluxNumericsCannotCertifyConservationOrInterfaceBalance)
{
    AnalysisSummarySet summaries;
    auto invalid_tolerance = certifiedFluxBalance("invalid-conservation");
    invalid_tolerance.balance_tolerance =
        std::numeric_limits<Real>::infinity();
    summaries.flux_balances.push_back(invalid_tolerance);

    auto invalid_residual = certifiedFluxBalance(
        "invalid-interface", DomainKind::InterfaceFace);
    invalid_residual.interface_pair_residual_norm =
        std::numeric_limits<Real>::quiet_NaN();
    summaries.flux_balances.push_back(invalid_residual);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* conservation = firstForBlock(
        report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer", "invalid-conservation");
    ASSERT_NE(conservation, nullptr);
    EXPECT_EQ(conservation->status, PropertyStatus::Unknown);
    ASSERT_TRUE(conservation->certification_class.has_value());
    EXPECT_EQ(*conservation->certification_class,
              CertificationClass::NotCertified);

    const auto* interface = firstForBlock(
        report, PropertyKind::InterfaceCondition,
        "InterfaceValidationAnalyzer", "invalid-interface");
    ASSERT_NE(interface, nullptr);
    EXPECT_EQ(interface->status, PropertyStatus::Unknown);
    ASSERT_TRUE(interface->certification_class.has_value());
    EXPECT_EQ(*interface->certification_class,
              CertificationClass::NotCertified);
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

    BoundarySymbolSummary mixed_corner_missing = certified;
    mixed_corner_missing.block =
        scalarBlock("mixed-corner-missing", DomainKind::Boundary);
    mixed_corner_missing.mixed_boundary_or_corner_scope_present = true;
    mixed_corner_missing.mixed_corner_edge_coverage_present = false;
    summaries.boundary_symbols.push_back(mixed_corner_missing);

    BoundarySymbolSummary mixed_corner_certified = mixed_corner_missing;
    mixed_corner_certified.block =
        scalarBlock("mixed-corner-certified", DomainKind::Boundary);
    mixed_corner_certified.mixed_corner_edge_coverage_present = true;
    summaries.boundary_symbols.push_back(mixed_corner_certified);

    BoundarySymbolSummary invalid_margin = certified;
    invalid_margin.block = scalarBlock("invalid-boundary", DomainKind::Boundary);
    invalid_margin.complementing_margin =
        std::numeric_limits<Real>::quiet_NaN();
    summaries.boundary_symbols.push_back(invalid_margin);

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

    const auto* mixed_missing_claim = firstForBlock(
        report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer", "mixed-corner-missing");
    ASSERT_NE(mixed_missing_claim, nullptr);
    EXPECT_EQ(mixed_missing_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(mixed_missing_claim->certification_class.has_value());
    EXPECT_EQ(*mixed_missing_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* mixed_certified_claim = firstForBlock(
        report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer", "mixed-corner-certified");
    ASSERT_NE(mixed_certified_claim, nullptr);
    EXPECT_EQ(mixed_certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(mixed_certified_claim->certification_class.has_value());
    EXPECT_EQ(*mixed_certified_claim->certification_class,
              CertificationClass::Certified);

    const auto* invalid_claim = firstForBlock(
        report, PropertyKind::BoundaryComplementingCondition,
        "InterfaceValidationAnalyzer", "invalid-boundary");
    ASSERT_NE(invalid_claim, nullptr);
    EXPECT_EQ(invalid_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(invalid_claim->certification_class.has_value());
    EXPECT_EQ(*invalid_claim->certification_class,
              CertificationClass::NotCertified);
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
    ErrorEstimatorSummary boolean_shape = certified;
    boolean_shape.estimator_id = "shape-boolean-only";
    boolean_shape.block = scalarBlock("shape-boolean-only");
    boolean_shape.shape_regular_mesh_evidence_present = true;
    summaries.error_estimators.push_back(boolean_shape);
    addEstimatorShapeRegularityEvidence(certified);
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

    const auto* boolean_shape_claim = firstForBlock(
        report, PropertyKind::ErrorEstimatorEligibility,
        "ErrorEstimatorAnalyzer", "shape-boolean-only");
    ASSERT_NE(boolean_shape_claim, nullptr);
    EXPECT_EQ(boolean_shape_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(boolean_shape_claim->certification_class.has_value());
    EXPECT_EQ(*boolean_shape_claim->certification_class,
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
    addMinimumResidualConditioningBounds(certified);
    summaries.minimum_residual_stability.push_back(certified);

    MinimumResidualStabilitySummary violated = certified;
    violated.method_id = "violated-dpg";
    violated.block = scalarBlock("violated-dpg");
    violated.violation_count = 1;
    summaries.minimum_residual_stability.push_back(violated);

    MinimumResidualStabilitySummary conditioning_exceeds_bound = certified;
    conditioning_exceeds_bound.method_id = "conditioning-exceeds-bound";
    conditioning_exceeds_bound.block =
        scalarBlock("conditioning-exceeds-bound");
    conditioning_exceeds_bound.local_trial_to_test_condition_estimate = 30.0;
    summaries.minimum_residual_stability.push_back(conditioning_exceeds_bound);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::MinimumResidualStability,
        "MinimumResidualStabilityAnalyzer");
    ASSERT_EQ(claims.size(), 4u);
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
    EXPECT_EQ(claims[3]->status, PropertyStatus::Violated);
    ASSERT_TRUE(claims[3]->certification_class.has_value());
    EXPECT_EQ(*claims[3]->certification_class,
              CertificationClass::Violated);
}

TEST(AnalysisEvidenceContracts, MinimumResidualFortinRouteRequiresQuantifiedNormBound)
{
    AnalysisSummarySet summaries;
    MinimumResidualStabilitySummary missing_norm;
    missing_norm.method_id = "fortin-missing-norm";
    missing_norm.block = scalarBlock("fortin-missing-norm");
    missing_norm.method_class = MinimumResidualMethodClass::DPG;
    missing_norm.trial_space_metadata_present = true;
    missing_norm.test_space_metadata_present = true;
    missing_norm.distinct_test_trial_spaces = true;
    missing_norm.residual_norm_metadata_present = true;
    missing_norm.test_norm_metadata_present = true;
    missing_norm.residual_norm_id = "graph-residual";
    missing_norm.test_norm_id = "enriched-test-norm";
    missing_norm.minimum_residual_theorem_id =
        "Gopalakrishnan-Qiu practical DPG Fortin theorem";
    missing_norm.method_scope_metadata_present = true;
    missing_norm.riesz_map_metadata_present = true;
    missing_norm.fortin_operator_evidence_present = true;
    missing_norm.enrichment_sufficiency_evidence_present = true;
    missing_norm.residual_control_constant_present = true;
    missing_norm.residual_control_constant = 2.0;
    missing_norm.local_trial_to_test_conditioning_present = true;
    missing_norm.local_trial_to_test_condition_estimate = 10.0;
    missing_norm.normal_equation_conditioning_present = true;
    missing_norm.normal_equation_condition_estimate = 100.0;
    addMinimumResidualConditioningBounds(missing_norm);
    summaries.minimum_residual_stability.push_back(missing_norm);

    MinimumResidualStabilitySummary quantified = missing_norm;
    quantified.method_id = "fortin-quantified";
    quantified.block = scalarBlock("fortin-quantified");
    quantified.fortin_operator_norm_bound_present = true;
    quantified.fortin_operator_norm_bound = 3.0;
    quantified.accepted_fortin_operator_norm_bound_present = true;
    quantified.accepted_fortin_operator_norm_bound = 4.0;
    quantified.discrete_inf_sup_lower_bound_present = true;
    quantified.discrete_inf_sup_lower_bound = 0.2;
    summaries.minimum_residual_stability.push_back(quantified);

    MinimumResidualStabilitySummary excessive = quantified;
    excessive.method_id = "fortin-excessive";
    excessive.block = scalarBlock("fortin-excessive");
    excessive.accepted_fortin_operator_norm_bound = 2.0;
    summaries.minimum_residual_stability.push_back(excessive);

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

TEST(AnalysisEvidenceContracts, EquilibriumPreservationRequiresFamilyAndTheoremScope)
{
    AnalysisSummarySet summaries;
    EquilibriumPreservationSummary residual_only;
    residual_only.equilibrium_id = "equilibrium-residual-only";
    residual_only.flux_source_residual = 0.0;
    residual_only.residual_tolerance = 1.0e-10;
    residual_only.source_quadrature_metadata_present = true;
    residual_only.reconstruction_metadata_present = true;
    residual_only.boundary_compatibility_metadata_present = true;
    summaries.equilibrium_preservation.push_back(residual_only);

    EquilibriumPreservationSummary certified = residual_only;
    certified.equilibrium_id = "equilibrium-certified";
    addEquilibriumScopeEvidence(certified);
    summaries.equilibrium_preservation.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::EquilibriumPreservation,
        "PreservationStructureAnalyzer");
    ASSERT_EQ(claims.size(), 2u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Likely);
    ASSERT_TRUE(claims[0]->certification_class.has_value());
    EXPECT_EQ(*claims[0]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claims[1]->certification_class.has_value());
    EXPECT_EQ(*claims[1]->certification_class,
              CertificationClass::Certified);
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
    addEnergyQuantitativeEvidence(certified);
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
    certified.spectral_convergence_theorem_id =
        "FEEC spectral correctness theorem";
    addSpectralComplexProvenance(certified);
    addSpectralQuantitativeEvidence(certified);
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

TEST(AnalysisEvidenceContracts, SpectralDiscreteCompactnessRequiresScopedProvenance)
{
    AnalysisSummarySet summaries;
    SpectralStructureSummary boolean_only;
    boolean_only.block = scalarBlock("discrete-compactness-boolean");
    boolean_only.eigenproblem_declared = true;
    boolean_only.self_adjoint_evidence = true;
    boolean_only.compactness_evidence = true;
    boolean_only.discrete_compactness_evidence = true;
    boolean_only.spectral_convergence_theorem_id =
        "Boffi discrete compactness theorem";
    addSpectralQuantitativeEvidence(boolean_only);
    summaries.spectral_structures.push_back(boolean_only);

    SpectralStructureSummary certified = boolean_only;
    certified.block = scalarBlock("discrete-compactness-provenance");
    certified.discrete_compactness_provenance_present = true;
    addSpectralComplexProvenance(certified);
    summaries.spectral_structures.push_back(certified);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto* boolean_claim = firstForBlock(
        report, PropertyKind::SpectralCorrectness,
        "SpectralSpuriousModeAnalyzer", "discrete-compactness-boolean");
    ASSERT_NE(boolean_claim, nullptr);
    EXPECT_EQ(boolean_claim->status, PropertyStatus::Unknown);
    const auto* certified_claim = firstForBlock(
        report, PropertyKind::SpectralCorrectness,
        "SpectralSpuriousModeAnalyzer", "discrete-compactness-provenance");
    ASSERT_NE(certified_claim, nullptr);
    EXPECT_EQ(certified_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified_claim->certification_class.has_value());
    EXPECT_EQ(*certified_claim->certification_class,
              CertificationClass::Certified);
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

TEST(AnalysisEvidenceContracts, ErrorEstimatorRejectsInvalidShapeRegularityConstant)
{
    AnalysisSummarySet summaries;
    ErrorEstimatorSummary invalid_shape;
    invalid_shape.estimator_id = "invalid-shape-regularity";
    invalid_shape.block = scalarBlock("invalid-shape-regularity");
    invalid_shape.residual_metadata_present = true;
    invalid_shape.jump_metadata_present = true;
    invalid_shape.norm_metadata_present = true;
    invalid_shape.estimator_norm_scope_metadata_present = true;
    invalid_shape.estimator_norm_id = "energy-norm";
    invalid_shape.pde_operator_class_metadata_present = true;
    invalid_shape.boundary_residual_metadata_present = true;
    invalid_shape.data_oscillation_metadata_present = true;
    invalid_shape.coefficient_source_regularity_metadata_present = true;
    invalid_shape.shape_regular_mesh_evidence_present = true;
    invalid_shape.mesh_family_scope_present = true;
    invalid_shape.mesh_family_scope = "shape-regular adaptive mesh family";
    invalid_shape.shape_regular_constant_present = true;
    invalid_shape.shape_regular_constant = -1.0;
    invalid_shape.reliability_constant_metadata_present = true;
    invalid_shape.reliability_constant = 2.0;
    invalid_shape.efficiency_constant_metadata_present = true;
    invalid_shape.efficiency_constant = 0.5;
    invalid_shape.effectivity_bounds_present = true;
    invalid_shape.effectivity_lower_bound = 0.7;
    invalid_shape.effectivity_upper_bound = 1.4;
    invalid_shape.effectivity_sample_count = 4;
    invalid_shape.refinement_evidence_present = true;
    invalid_shape.estimator_theorem_id = "Verfurth residual estimator theorem";
    summaries.error_estimators.push_back(invalid_shape);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto* claim = firstForBlock(
        report, PropertyKind::ErrorEstimatorEligibility,
        "ErrorEstimatorAnalyzer", "invalid-shape-regularity");
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

TEST(AnalysisEvidenceContracts, InvalidNumericEvidencePreventsAdvancedCertification)
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
    EnergyEntropySummary energy;
    energy.energy_entropy_id = "invalid-energy-numeric";
    energy.law_kind = EnergyEntropyLawKind::Energy;
    energy.expected_production_sign = BalanceSignClass::Nonpositive;
    energy.balance_tolerance = std::numeric_limits<Real>::quiet_NaN();
    energy.observed_discrete_balance = 0.0;
    energy.observed_production = -1.0e-9;
    energy.energy_functional_id = "E";
    energy.energy_norm_id = "energy-norm";
    energy.energy_entropy_theorem_id = "discrete energy identity";
    energy.energy_functional_metadata_present = true;
    energy.energy_norm_metadata_present = true;
    energy.energy_positivity_evidence_present = true;
    energy.energy_coercivity_evidence_present = true;
    energy.discrete_dissipation_identity_evidence_present = true;
    energy.boundary_source_energy_accounting_present = true;
    addEnergyQuantitativeEvidence(energy);
    summaries.energy_entropy.push_back(energy);

    EquilibriumPreservationSummary equilibrium;
    equilibrium.equilibrium_id = "invalid-equilibrium-numeric";
    equilibrium.flux_source_residual = 0.0;
    equilibrium.residual_tolerance =
        std::numeric_limits<Real>::infinity();
    equilibrium.source_quadrature_metadata_present = true;
    equilibrium.reconstruction_metadata_present = true;
    equilibrium.boundary_compatibility_metadata_present = true;
    summaries.equilibrium_preservation.push_back(equilibrium);

    TransferOperatorSummary transfer;
    transfer.interface_pair_id = "invalid-transfer-interface";
    transfer.projection_space_id = "mortar-space";
    transfer.residual_tolerance = 1.0e-8;
    transfer.conservation_residual =
        std::numeric_limits<Real>::quiet_NaN();
    transfer.constant_preservation_residual = 0.0;
    transfer.rank_metadata_present = true;
    transfer.interface_scope_metadata_present = true;
    transfer.projection_consistency_metadata_present = true;
    transfer.mortar_inf_sup_or_dual_consistency_metadata_present = true;
    transfer.interface_mass_conditioning_metadata_present = true;
    transfer.action_reaction_flux_metadata_present = true;
    summaries.transfer_operators.push_back(transfer);

    AdjointConsistencySummary adjoint;
    adjoint.contribution_id = "invalid-adjoint-numeric";
    adjoint.goal_functional_id = "J";
    adjoint.adjoint_consistency = AdjointConsistencyKind::Yes;
    adjoint.transpose_backend_support = true;
    adjoint.boundary_adjoint_metadata_present = true;
    adjoint.stabilization_adjoint_metadata_present = true;
    adjoint.goal_linearization_metadata_present = true;
    adjoint.discrete_adjoint_residual_present = true;
    adjoint.discrete_adjoint_residual = 0.0;
    adjoint.discrete_adjoint_tolerance =
        std::numeric_limits<Real>::infinity();
    summaries.adjoint_consistency.push_back(adjoint);

    NonlinearTangentSummary tangent;
    tangent.residual_id = "invalid-tangent-numeric";
    tangent.block = scalarBlock("invalid-tangent-numeric");
    tangent.tangent_consistency = TangentConsistencyClass::Exact;
    tangent.jacobian_action_available = true;
    tangent.finite_difference_action_error = 0.0;
    tangent.finite_difference_tolerance =
        std::numeric_limits<Real>::infinity();
    summaries.nonlinear_tangents.push_back(tangent);

    CoupledSystemStabilitySummary coupled;
    coupled.coupling_group = "invalid-coupling-numeric";
    coupled.variables = {u, lambda};
    coupled.monolithic_coupling = true;
    coupled.exchange_residual_present = true;
    coupled.exchange_residual = 0.0;
    coupled.constraint_drift_present = true;
    coupled.constraint_drift_norm = 0.0;
    coupled.coupling_tolerance_present = true;
    coupled.coupling_tolerance =
        std::numeric_limits<Real>::infinity();
    coupled.coupled_operator_stability_evidence_present = true;
    coupled.contraction_norm_evidence_present = true;
    summaries.coupled_system_stability.push_back(coupled);

    DAEStructureEvidenceSummary dae;
    dae.system_id = "invalid-dae-numeric";
    dae.variables = {u, lambda};
    dae.dae_form_class = DAEFormClass::SemiExplicit;
    dae.mass_matrix_rank_metadata_present = true;
    dae.algebraic_jacobian_rank_metadata_present = true;
    dae.algebraic_jacobian_full_rank = true;
    dae.hidden_constraint_metadata_present = true;
    dae.consistent_initial_condition_evidence_present = true;
    dae.initial_constraint_residual = 0.0;
    dae.residual_tolerance =
        std::numeric_limits<Real>::infinity();
    summaries.dae_structure_evidence.push_back(dae);

    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));

    const auto* energy_claim = firstFrom(
        report, PropertyKind::EnergyStability,
        "EnergyEntropyLawAnalyzer");
    ASSERT_NE(energy_claim, nullptr);
    EXPECT_EQ(energy_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(energy_claim->certification_class.has_value());
    EXPECT_EQ(*energy_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* equilibrium_claim = firstFrom(
        report, PropertyKind::EquilibriumPreservation,
        "PreservationStructureAnalyzer");
    ASSERT_NE(equilibrium_claim, nullptr);
    EXPECT_EQ(equilibrium_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(equilibrium_claim->certification_class.has_value());
    EXPECT_EQ(*equilibrium_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* transfer_claim = firstFrom(
        report, PropertyKind::TransferOperatorCompatibility,
        "PreservationStructureAnalyzer");
    ASSERT_NE(transfer_claim, nullptr);
    EXPECT_EQ(transfer_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(transfer_claim->certification_class.has_value());
    EXPECT_EQ(*transfer_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* adjoint_claim = firstFrom(
        report, PropertyKind::AdjointConsistency,
        "PreservationStructureAnalyzer");
    ASSERT_NE(adjoint_claim, nullptr);
    EXPECT_EQ(adjoint_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(adjoint_claim->certification_class.has_value());
    EXPECT_EQ(*adjoint_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* tangent_claim = firstForBlock(
        report, PropertyKind::NonlinearTangentStructure,
        "NonlinearTangentAnalyzer", "invalid-tangent-numeric");
    ASSERT_NE(tangent_claim, nullptr);
    EXPECT_EQ(tangent_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(tangent_claim->certification_class.has_value());
    EXPECT_EQ(*tangent_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* coupled_claim = firstFrom(
        report, PropertyKind::CoupledSystemStructure,
        "CoupledSystemStabilityAnalyzer");
    ASSERT_NE(coupled_claim, nullptr);
    EXPECT_EQ(coupled_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(coupled_claim->certification_class.has_value());
    EXPECT_EQ(*coupled_claim->certification_class,
              CertificationClass::NotCertified);

    const auto* dae_claim = firstFrom(
        report, PropertyKind::DifferentialAlgebraicStructure,
        "DAEStructureAnalyzer");
    ASSERT_NE(dae_claim, nullptr);
    EXPECT_EQ(dae_claim->status, PropertyStatus::Likely);
    ASSERT_TRUE(dae_claim->certification_class.has_value());
    EXPECT_EQ(*dae_claim->certification_class,
              CertificationClass::NotCertified);
}

TEST(AnalysisEvidenceContracts, TransportNumericEnrichmentIgnoresInvalidScalars)
{
    const auto u = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(h1Field(0, 1, FieldType::Scalar, 1, "u"));
    ctx.addContribution(ContributionDescriptor::transportLike(
        u, "invalid-transport-numerics", "evidence-contract"));

    AnalysisSummarySet summaries;
    ParameterScaleSummary peclet;
    peclet.role = ParameterScaleRole::PecletLike;
    peclet.block = scalarBlock("invalid-transport-numerics");
    peclet.max_scale_value = std::numeric_limits<Real>::infinity();
    summaries.parameter_scales.push_back(peclet);

    TemporalStabilitySummary temporal;
    temporal.block = scalarBlock("invalid-transport-numerics");
    temporal.cfl_estimate_present = true;
    temporal.cfl_estimate = -0.25;
    summaries.temporal_stability.push_back(temporal);

    DiscreteMatrixSummary matrix;
    matrix.block = scalarBlock("invalid-transport-numerics");
    matrix.nonnormality_indicator =
        std::numeric_limits<Real>::quiet_NaN();
    summaries.discrete_matrices.push_back(matrix);

    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto* transport = firstFrom(
        report, PropertyKind::OperatorTransportCharacter,
        "TransportCharacterAnalyzer");
    ASSERT_NE(transport, nullptr);
    EXPECT_FALSE(transport->peclet_number.has_value())
        << "peclet=" << transport->peclet_number.value_or(-999.0);
    EXPECT_FALSE(transport->cfl_number.has_value())
        << "cfl=" << transport->cfl_number.value_or(-999.0);
    EXPECT_FALSE(transport->nonnormality_indicator.has_value())
        << "nonnormality="
        << transport->nonnormality_indicator.value_or(-999.0);
}

TEST(AnalysisEvidenceContracts, MeshMappingValidityDoesNotImplyShapeRegularity)
{
    AnalysisSummarySet summaries;
    MeshGeometryQualitySummary mapping_only;
    mapping_only.min_jacobian = 0.25;
    mapping_only.max_jacobian = 2.0;
    summaries.mesh_geometry_quality.push_back(mapping_only);

    MeshGeometryQualitySummary invalid_shape = mapping_only;
    invalid_shape.shape_regular_evidence_present = true;
    invalid_shape.shape_regular_constant = -1.0;
    summaries.mesh_geometry_quality.push_back(invalid_shape);

    MeshGeometryQualitySummary valid_shape = mapping_only;
    valid_shape.shape_regular_evidence_present = true;
    valid_shape.shape_regular_constant = 4.0;
    valid_shape.mesh_family_scope_present = true;
    valid_shape.mesh_family_scope = "uniform shape-regular mesh family";
    summaries.mesh_geometry_quality.push_back(valid_shape);

    ProblemAnalysisContext ctx;
    ctx.setAnalysisSummaries(std::move(summaries));
    const auto report = analyze(std::move(ctx));
    const auto claims = claimsFrom(
        report, PropertyKind::MeshGeometryValidity,
        "MeshGeometryAnalyzer");
    ASSERT_EQ(claims.size(), 3u);
    EXPECT_EQ(claims[0]->status, PropertyStatus::Preserved);
    EXPECT_NE(claims[0]->description.find("out of scope"),
              std::string::npos);
    EXPECT_EQ(claims[1]->status, PropertyStatus::Preserved);
    ASSERT_FALSE(claims[1]->evidence.empty());
    EXPECT_NE(claims[1]->evidence.front().description.find(
                  "not certified"),
              std::string::npos);
    EXPECT_EQ(claims[2]->status, PropertyStatus::Preserved);
    EXPECT_NE(claims[2]->description.find("shape-regularity evidence"),
              std::string::npos);
    ASSERT_FALSE(claims[2]->evidence.empty());
    EXPECT_NE(claims[2]->evidence.front().description.find(
                  "shape_regular_constant"),
              std::string::npos);
}
