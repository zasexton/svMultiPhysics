/**
 * @file test_Phase7MathematicalIntegration.cpp
 * @brief Phase 7 generic mathematical integration tests for FE analysis.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/SparseMatrixSummaryScanner.h"
#include "Backends/Utils/BackendOptions.h"

#if FE_HAS_MPI
#include <mpi.h>
#endif

using namespace svmp::FE;
using namespace svmp::FE::analysis;

namespace {

FieldDescriptor scalarH1(FieldId id, int order = 1, std::string name = "scalar")
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = FieldType::Scalar;
    fd.value_dimension = 1;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::H1;
    fd.trace_capabilities = TraceCapabilityFlags::Value |
                            TraceCapabilityFlags::NormalFlux;
    return fd;
}

FieldDescriptor vectorH1(FieldId id, int order = 2, std::string name = "primary")
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 3;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::H1;
    fd.trace_capabilities = TraceCapabilityFlags::Value |
                            TraceCapabilityFlags::NormalFlux;
    return fd;
}

OperatorBlockId blockFor(VariableKey variable,
                         std::string tag,
                         DomainKind domain = DomainKind::Cell)
{
    OperatorBlockId block;
    block.test_variables = {variable};
    block.trial_variables = {variable};
    block.operator_tag = std::move(tag);
    block.domain = domain;
    block.role = ContributionRole::DiagonalBlock;
    return block;
}

ContributionDescriptor scalarGradientOperator(VariableKey variable,
                                              std::string tag = "scalar-gradient")
{
    auto contribution = ContributionDescriptor::diagonalSymmetric(
        variable, std::move(tag), "phase7-toy");
    NullspaceHint hint;
    hint.field = variable.field_id;
    hint.family = NullspaceFamily::ScalarConstant;
    hint.confidence = AnalysisConfidence::High;
    hint.reason =
        "generic scalar gradient operator has a constant-shift nullspace";
    contribution.nullspace_hints.push_back(std::move(hint));
    return contribution;
}

BoundaryConditionDescriptor strongValueBoundary(VariableKey variable)
{
    BoundaryConditionDescriptor bc;
    bc.primary_variable = variable;
    bc.trace_kind = TraceKind::Value;
    bc.enforcement_kind = EnforcementKind::Strong;
    bc.boundary_marker = 1;
    bc.source = "phase7-strong-boundary";
    return bc;
}

ReducedMatrixSummary reducedMatrix(VariableKey variable,
                                   std::string tag,
                                   bool z_matrix = true,
                                   bool exact_reduction = true)
{
    ReducedMatrixSummary reduced;
    auto& matrix = reduced.free_free_matrix;
    matrix.block = blockFor(variable, std::move(tag));
    matrix.scope = NumericSummaryScope::ReducedFreeFree;
    matrix.rows = 3;
    matrix.cols = 3;
    matrix.square = true;
    matrix.structurally_symmetric = true;
    matrix.numerically_symmetric = true;
    matrix.symmetry_evidence_complete = true;
    matrix.sign_tolerance = 1.0e-12;
    matrix.row_sum_tolerance = 1.0e-12;
    matrix.diagonal_count = 3;
    matrix.offdiag_count = 4;
    matrix.nonpositive_diagonal_count = 0;
    matrix.negative_diagonal_count = 0;
    matrix.near_zero_diagonal_count = 0;
    matrix.min_row_sum = 0.0;
    matrix.max_row_sum = 1.0;

    if (z_matrix) {
        matrix.negative_offdiag_count = 4;
        matrix.positive_offdiag_count = 0;
        matrix.max_positive_offdiag = 0.0;
    } else {
        matrix.negative_offdiag_count = 3;
        matrix.positive_offdiag_count = 1;
        matrix.max_positive_offdiag = 0.25;
        matrix.addWorstEntry(MatrixEntrySample{1, 2, 0.25, 0, 0,
                                               "positive offdiagonal"});
    }

    reduced.reduction_kind = ConstraintReductionKind::StrongDirichletElimination;
    reduced.free_dof_count = 3;
    reduced.constrained_dof_count = 1;
    reduced.reduction_exact_for_analysis = exact_reduction;
    return reduced;
}

CoefficientPropertySummary positiveCoefficient(std::string id = "positive-coefficient")
{
    CoefficientPropertySummary coeff;
    coeff.coefficient = std::move(id);
    coeff.tensor_rank = TensorRank::Scalar;
    coeff.symmetry = SymmetryClass::Symmetric;
    coeff.positivity = PositivityClass::Positive;
    coeff.min_eigenvalue = 1.0;
    coeff.max_eigenvalue = 2.0;
    return coeff;
}

MeshGeometryQualitySummary positiveGeometry()
{
    MeshGeometryQualitySummary mesh;
    mesh.min_jacobian = 0.25;
    mesh.max_jacobian = 1.5;
    return mesh;
}

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

bool hasClaimWithStatus(const ProblemAnalysisReport& report,
                        PropertyKind kind,
                        PropertyStatus status)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == kind && claim.status == status) return true;
    }
    return false;
}

bool hasClaimFromWithStatus(const ProblemAnalysisReport& report,
                            PropertyKind kind,
                            const std::string& origin,
                            PropertyStatus status)
{
    for (const auto& claim : report.claims) {
        if (claim.kind == kind &&
            claim.claim_origin == origin &&
            claim.status == status) {
            return true;
        }
    }
    return false;
}

ProblemAnalysisReport analyze(ProblemAnalysisContext ctx)
{
    return ProblemAnalyzer::createDefault().analyze(ctx);
}

std::vector<std::vector<SparseMatrixRowEntry>> tridiagonalRows(GlobalIndex rows,
                                                               GlobalIndex begin,
                                                               GlobalIndex end)
{
    std::vector<std::vector<SparseMatrixRowEntry>> out;
    out.reserve(static_cast<std::size_t>(end - begin));
    for (GlobalIndex row = begin; row < end; ++row) {
        std::vector<SparseMatrixRowEntry> entries;
        if (row > 0) entries.push_back({row - 1, -1.0});
        entries.push_back({row, 2.0});
        if (row + 1 < rows) entries.push_back({row + 1, -1.0});
        out.push_back(std::move(entries));
    }
    return out;
}

void expectSameMatrixClassification(const DiscreteMatrixSummary& a,
                                    const DiscreteMatrixSummary& b)
{
    EXPECT_EQ(a.rows, b.rows);
    EXPECT_EQ(a.cols, b.cols);
    EXPECT_EQ(a.square, b.square);
    EXPECT_EQ(a.positive_offdiag_count, b.positive_offdiag_count);
    EXPECT_EQ(a.negative_offdiag_count, b.negative_offdiag_count);
    EXPECT_EQ(a.row_sum_violation_count, b.row_sum_violation_count);
    EXPECT_DOUBLE_EQ(a.max_positive_offdiag, b.max_positive_offdiag);
    EXPECT_DOUBLE_EQ(a.min_row_sum, b.min_row_sum);
    EXPECT_DOUBLE_EQ(a.max_row_sum, b.max_row_sum);
}

} // namespace

TEST(Phase7Integration, ScalarDiffusion_AcuteSimplex_ZMatrixCertified)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(scalarGradientOperator(scalar, "acute-scalar"));
    ctx.addBCDescriptor(strongValueBoundary(scalar));

    AnalysisSummarySet summaries;
    summaries.coefficient_properties.push_back(positiveCoefficient());
    summaries.reduced_matrices.push_back(
        reducedMatrix(scalar, "acute-scalar", true));
    summaries.mesh_geometry_quality.push_back(positiveGeometry());
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));

    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::Nullspace,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::OperatorDefiniteness,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::CoefficientPositivity,
                                   PropertyStatus::Preserved));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::ZMatrixStructure,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MMatrixStructure,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::DiscreteMaximumPrinciple,
                                   PropertyStatus::Preserved));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MeshGeometryValidity,
                                   PropertyStatus::Preserved));
}

TEST(Phase7Integration, ScalarDiffusion_ObtuseSimplex_ZMatrixViolated)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarH1(0));

    AnalysisSummarySet summaries;
    summaries.coefficient_properties.push_back(positiveCoefficient());
    auto reduced = reducedMatrix(scalar, "obtuse-scalar", false);
    ASSERT_EQ(reduced.free_free_matrix.worst_entries.size(), 1u);
    summaries.reduced_matrices.push_back(reduced);
    MeshGeometryQualitySummary mesh;
    mesh.min_jacobian = 0.1;
    mesh.max_jacobian = 1.0;
    mesh.poor_quality_element_count = 1;
    mesh.max_aspect_ratio = 20.0;
    mesh.worst_elements = {7};
    summaries.mesh_geometry_quality.push_back(mesh);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));

    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::ZMatrixStructure,
                                   PropertyStatus::Violated));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::DiscreteMaximumPrinciple,
                                   PropertyStatus::Violated));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MeshGeometryValidity,
                                   PropertyStatus::Likely));
}

TEST(Phase7Integration, ScalarDiffusion_ConstrainedPositiveOffdiag_NotReducedViolation)
{
    const auto scalar = VariableKey::field(0);
    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 2.0}, {1, 0.5}},
        {{0, 0.5}, {1, 2.0}, {2, -1.0}},
        {{1, -1.0}, {2, 2.0}, {3, -1.0}},
        {{2, -1.0}, {3, 2.0}},
    };
    const auto source = CsrSparseRowScanSource::fromRows(4, 4, rows);
    const auto mask = ConstraintReductionMask::fromConstrainedDofs(
        4, {0}, ConstraintReductionKind::StrongDirichletElimination);

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;

    const auto full = scanSparseMatrixSummary(source, blockFor(scalar, "full"),
                                              options);
    const auto reduced = scanReducedFreeFreeSummary(
        source, mask, blockFor(scalar, "reduced"), options);
    ASSERT_GT(full.summary.positive_offdiag_count, 0u);
    ASSERT_EQ(reduced.free_free_matrix.positive_offdiag_count, 0u);

    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarH1(0));
    AnalysisSummarySet summaries;
    summaries.reduced_matrices.push_back(reduced);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::ZMatrixStructure,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MMatrixStructure,
                                   PropertyStatus::Exact));
}

TEST(Phase7Integration, ScalarDiffusion_AffineConstraint_NoReductionSummary_Unknown)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarH1(0));

    AnalysisSummarySet summaries;
    auto reduced = reducedMatrix(scalar, "affine-unreduced", true, false);
    reduced.reduction_kind = ConstraintReductionKind::AffineTransform;
    reduced.affine_terms_accounted_for = false;
    reduced.reduction_exact_for_analysis = false;
    summaries.reduced_matrices.push_back(reduced);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MMatrixStructure,
                                   PropertyStatus::Unknown));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::InitialDataCompatibility,
                                       "ConstraintRankAnalyzer",
                                       PropertyStatus::Unknown));
}

TEST(Phase7Integration, MixedPair_StablePair_InfSupLikelySatisfied)
{
    ProblemAnalysisContext ctx;
    const auto primary = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);
    ctx.addFieldDescriptor(vectorH1(1, 2, "primary"));
    ctx.addFieldDescriptor(scalarH1(0, 1, "multiplier"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        primary, "mixed-primary", "phase7-toy"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        primary, multiplier, "generic-pair", "mixed-pair", "phase7-toy"));

    AnalysisSummarySet summaries;
    InfSupEstimateSummary estimate;
    estimate.primal_variable = primary;
    estimate.multiplier_variable = multiplier;
    estimate.estimate_value = 0.25;
    estimate.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    estimate.estimate_scope = "free-free";
    summaries.inf_sup_estimates.push_back(estimate);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MixedSaddlePoint,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::InfSupCondition,
                                   PropertyStatus::Exact));
    const PropertyClaim* numeric = nullptr;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.claim_origin == "InfSupAnalyzer" &&
            claim.inf_sup_estimate.has_value()) {
            numeric = &claim;
            break;
        }
    }
    ASSERT_NE(numeric, nullptr);
    EXPECT_TRUE(numeric->inf_sup_class.has_value());
    EXPECT_EQ(*numeric->inf_sup_class, InfSupClass::StructurallySupported);
    EXPECT_TRUE(numeric->nullspace_handling_class.has_value());
}

TEST(Phase7Integration, MixedPair_UnstablePair_InfSupRisk)
{
    ProblemAnalysisContext ctx;
    const auto primary = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);
    ctx.addFieldDescriptor(vectorH1(1, 1, "primary"));
    ctx.addFieldDescriptor(scalarH1(0, 1, "multiplier"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        primary, "mixed-primary", "phase7-toy"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        primary, multiplier, "generic-pair", "mixed-pair", "phase7-toy"));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::InfSupCondition,
                                   PropertyStatus::Likely));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::LockingRisk,
                                   PropertyStatus::Likely));
}

TEST(Phase7Integration, Transport_HighPeclet_NoStabilization_Risk)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(ContributionDescriptor::transportLike(
        scalar, "first-order-operator", "phase7-toy"));

    AnalysisSummarySet summaries;
    ParameterScaleSummary peclet;
    peclet.nondimensional_parameter_id = "dimensionless-transport-scale";
    peclet.max_scale_value = 50.0;
    summaries.parameter_scales.push_back(peclet);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* transport = firstClaimFrom(
        report, PropertyKind::OperatorTransportCharacter,
        "TransportCharacterAnalyzer");
    ASSERT_NE(transport, nullptr);
    ASSERT_TRUE(transport->peclet_number.has_value());
    EXPECT_GT(*transport->peclet_number, 1.0);
    EXPECT_TRUE(transport->transport_character_class.has_value());
    EXPECT_EQ(*transport->transport_character_class,
              TransportCharacterClass::TransportDominatedRisk);
    EXPECT_EQ(report.countByKind(PropertyKind::Stabilization), 0u);

    ProblemAnalysisContext stabilized_ctx;
    stabilized_ctx.addFieldDescriptor(scalarH1(0));
    stabilized_ctx.addContribution(ContributionDescriptor::transportLike(
        scalar, "first-order-operator", "phase7-toy"));
    stabilized_ctx.addContribution(ContributionDescriptor::stabilization(
        scalar, "consistent-stabilization", "phase7-toy"));
    auto stabilized_report = analyze(std::move(stabilized_ctx));
    EXPECT_GT(stabilized_report.countByKind(PropertyKind::Stabilization), 0u);
}

TEST(Phase7Integration, DG_SIPG_PenaltyTooSmall_CoercivityWarning)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    BoundarySymbolSummary symbol;
    symbol.block = blockFor(VariableKey::field(0), "face-operator",
                            DomainKind::InteriorFace);
    symbol.complementing_condition_satisfied = true;
    summaries.boundary_symbols.push_back(symbol);
    ParameterScaleSummary penalty;
    penalty.nondimensional_parameter_id = "weak-boundary-penalty";
    penalty.max_scale_value = 0.25;
    summaries.parameter_scales.push_back(penalty);
    FluxBalanceSummary flux;
    flux.block = symbol.block;
    flux.balance_tolerance = 1.0e-8;
    flux.interface_pair_residual_norm = 1.0e-4;
    summaries.flux_balances.push_back(flux);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::WeakBoundaryCoercivity,
                                       "InterfaceValidationAnalyzer",
                                       PropertyStatus::Violated));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::InterfaceCondition,
                                       "InterfaceValidationAnalyzer",
                                       PropertyStatus::Violated));
}

TEST(Phase7Integration, CompatibleComplex_ExactSequenceSupported)
{
    ProblemAnalysisContext ctx;
    FieldDescriptor hcurl;
    hcurl.field_id = 0;
    hcurl.name = "curl-conforming";
    hcurl.space_family = SpaceFamily::HCurl;
    hcurl.has_exact_sequence_structure = true;
    ctx.addFieldDescriptor(hcurl);
    FieldDescriptor hdiv;
    hdiv.field_id = 1;
    hdiv.name = "div-conforming";
    hdiv.space_family = SpaceFamily::HDiv;
    hdiv.has_exact_sequence_structure = true;
    ctx.addFieldDescriptor(hdiv);

    AnalysisSummarySet summaries;
    CompatibleComplexSummary complex;
    complex.complex_id = "generic-exact-sequence";
    complex.variables = {VariableKey::field(0), VariableKey::field(1)};
    complex.exact_sequence_compatible = true;
    complex.trace_sequence_compatible = true;
    complex.commuting_projection_available = true;
    summaries.compatible_complexes.push_back(complex);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report,
                                   PropertyKind::CompatibleComplexStructure,
                                   PropertyStatus::Preserved));
}

TEST(Phase7Integration, CompatibleComplex_IncompatibleSpacesFlagged)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    CompatibleComplexSummary complex;
    complex.complex_id = "generic-broken-sequence";
    complex.variables = {VariableKey::field(0), VariableKey::field(1)};
    complex.exact_sequence_compatible = false;
    complex.trace_sequence_compatible = true;
    complex.commuting_projection_available = false;
    complex.missing_space_count = 1;
    summaries.compatible_complexes.push_back(complex);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report,
                                   PropertyKind::CompatibleComplexStructure,
                                   PropertyStatus::Violated));
}

TEST(Phase7Integration, TransientParabolic_TemporalClassAndEnergyDecay)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(ContributionDescriptor::massLike(
        scalar, "time-accumulation", "phase7-toy"));
    VariableDescriptor vd;
    vd.key = scalar;
    vd.temporal_state_kind = TemporalStateKind::Dynamic;
    vd.max_time_derivative_order = 1;
    ctx.addVariableDescriptor(vd);

    AnalysisSummarySet summaries;
    TemporalStabilitySummary temporal;
    temporal.time_scheme = "generic-a-stable-step";
    temporal.stability_class = TemporalStabilityClass::AStable;
    temporal.amplification_radius = 0.95;
    summaries.temporal_stability.push_back(temporal);
    EnergyEntropySummary energy;
    energy.energy_entropy_id = "quadratic-energy";
    energy.law_kind = EnergyEntropyLawKind::Energy;
    energy.expected_production_sign = BalanceSignClass::Nonpositive;
    energy.balance_tolerance = 1.0e-8;
    energy.observed_discrete_balance = 0.0;
    energy.observed_production = -1.0e-5;
    summaries.energy_entropy.push_back(energy);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report,
                                   PropertyKind::DifferentialAlgebraicStructure,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::TemporalStability,
                                       "TemporalStabilityAnalyzer",
                                       PropertyStatus::Preserved));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::EnergyStability,
                                       "EnergyEntropyLawAnalyzer",
                                       PropertyStatus::Preserved));
}

TEST(Phase7Integration, NonlinearTangent_ExactVsInconsistent)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    NonlinearTangentSummary exact;
    exact.residual_id = "exact-residual";
    exact.block = blockFor(VariableKey::field(0), "exact-tangent");
    exact.tangent_consistency = TangentConsistencyClass::Exact;
    exact.jacobian_action_available = true;
    exact.finite_difference_tolerance = 1.0e-8;
    exact.finite_difference_action_error = 1.0e-12;
    summaries.nonlinear_tangents.push_back(exact);
    NonlinearTangentSummary bad;
    bad.residual_id = "inconsistent-residual";
    bad.block = blockFor(VariableKey::field(0), "inconsistent-tangent");
    bad.tangent_consistency = TangentConsistencyClass::Inconsistent;
    bad.jacobian_action_available = true;
    bad.finite_difference_tolerance = 1.0e-8;
    bad.finite_difference_action_error = 1.0e-3;
    summaries.nonlinear_tangents.push_back(bad);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::NonlinearTangentStructure,
                                       "NonlinearTangentAnalyzer",
                                       PropertyStatus::Preserved));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::NonlinearTangentStructure,
                                       "NonlinearTangentAnalyzer",
                                       PropertyStatus::Violated));
}

TEST(Phase7Integration, NoncoerciveWave_ResolutionWarningAndCgRejected)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    DiscreteMatrixSummary matrix;
    matrix.block = blockFor(scalar, "oscillatory-operator");
    matrix.rows = 2;
    matrix.cols = 2;
    matrix.square = true;
    matrix.symmetry_evidence_complete = true;
    matrix.structurally_symmetric = false;
    matrix.numerically_symmetric = false;
    matrix.diagonal_count = 2;
    matrix.offdiag_count = 2;
    matrix.max_symmetry_error = 0.2;
    matrix.max_abs_entry = 1.0;

    AnalysisSummarySet summaries;
    summaries.discrete_matrices.push_back(matrix);
    ParameterScaleSummary frequency;
    frequency.nondimensional_parameter_id = "frequency-resolution";
    frequency.frequency_resolution_metric = 2.5;
    summaries.parameter_scales.push_back(frequency);
    ctx.setAnalysisSummaries(std::move(summaries));

    backends::SolverOptions options;
    options.method = backends::SolverMethod::CG;
    options.preconditioner = backends::PreconditionerType::Diagonal;
    ctx.setSolverOptions(options);

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::LockingRisk,
                                   PropertyStatus::Likely));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::SolverCompatibility,
                                       "SolverCompatibilityAnalyzer",
                                       PropertyStatus::Violated));
}

TEST(Phase7Integration, MovingDomain_GCLResidualNonzero_Warning)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    MovingDomainSummary moving;
    moving.mesh_revision = 42;
    moving.mesh_velocity_metadata_present = true;
    moving.min_geometric_jacobian = 0.75;
    moving.max_geometric_jacobian = 1.25;
    moving.geometric_conservation_residual = 1.0e-4;
    moving.geometric_conservation_tolerance = 1.0e-8;
    summaries.moving_domain.push_back(moving);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* claim = firstClaimFrom(report, PropertyKind::GeometricConservation,
                                       "PreservationStructureAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(claim->flux_balance_residual.has_value());
    EXPECT_GT(*claim->flux_balance_residual, 1.0e-8);
    EXPECT_NE(claim->evidence.front().description.find("min_jacobian=0.750000"),
              std::string::npos);
}

TEST(Phase7Integration, Transfer_NonmatchingProjection_NotConstantPreserving)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    TransferOperatorSummary good;
    good.interface_pair_id = "pair-ok";
    good.projection_space_id = "projection-ok";
    good.conservation_residual = 0.0;
    good.constant_preservation_residual = 0.0;
    good.rank_metadata_present = true;
    summaries.transfer_operators.push_back(good);
    TransferOperatorSummary bad;
    bad.interface_pair_id = "pair-bad";
    bad.projection_space_id = "projection-bad";
    bad.conservation_residual = 1.0e-4;
    bad.constant_preservation_residual = 1.0e-3;
    bad.rank_metadata_present = false;
    summaries.transfer_operators.push_back(bad);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::TransferOperatorCompatibility,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Preserved));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::TransferOperatorCompatibility,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Violated));
}

TEST(Phase7Integration, BoundPreservingUpdate_PreservedOrViolationReported)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    InvariantDomainSummary ok;
    ok.invariant_set_id = "bounds-ok";
    ok.variables = {VariableKey::field(0)};
    ok.lower_bound_active = true;
    ok.lower_bound = 0.0;
    ok.limiter_evidence_present = true;
    ok.post_step_violation_count = 0;
    summaries.invariant_domains.push_back(ok);
    InvariantDomainSummary bad = ok;
    bad.invariant_set_id = "bounds-bad";
    bad.post_step_violation_count = 2;
    summaries.invariant_domains.push_back(bad);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::InvariantDomainPreservation,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Preserved));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::InvariantDomainPreservation,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Violated));
}

TEST(Phase7Integration, EquilibriumPreservingBalance_FluxSourceResidualZero)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    EquilibriumPreservationSummary equilibrium;
    equilibrium.equilibrium_id = "declared-equilibrium";
    equilibrium.flux_source_residual = 0.0;
    equilibrium.residual_tolerance = 1.0e-10;
    equilibrium.source_quadrature_metadata_present = true;
    equilibrium.reconstruction_metadata_present = true;
    summaries.equilibrium_preservation.push_back(equilibrium);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* claim = firstClaimFrom(report,
                                       PropertyKind::EquilibriumPreservation,
                                       "PreservationStructureAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claim->well_balanced_metadata_present.has_value());
    EXPECT_TRUE(*claim->well_balanced_metadata_present);
}

TEST(Phase7Integration, DAE_InconsistentInitialState_Warning)
{
    ProblemAnalysisContext ctx;
    const auto dynamic = VariableKey::field(0);
    const auto algebraic = VariableKey::field(1);
    ctx.addContribution(ContributionDescriptor::massLike(
        dynamic, "dynamic-block", "phase7-toy"));
    ContributionDescriptor constraint;
    constraint.operator_tag = "algebraic-block";
    constraint.origin = "phase7-toy";
    constraint.role = ContributionRole::ConstraintBlock;
    constraint.test_variables = {algebraic};
    constraint.trial_variables = {dynamic};
    constraint.temporal = TemporalDescriptor{0, TemporalContributionKind::PureConstraint};
    ctx.addContribution(std::move(constraint));
    VariableDescriptor vd0;
    vd0.key = dynamic;
    vd0.temporal_state_kind = TemporalStateKind::Dynamic;
    ctx.addVariableDescriptor(vd0);
    VariableDescriptor vd1;
    vd1.key = algebraic;
    vd1.temporal_state_kind = TemporalStateKind::Algebraic;
    vd1.participates_in_constraint_blocks = true;
    ctx.addVariableDescriptor(vd1);

    AnalysisSummarySet summaries;
    InitialCompatibilitySummary initial;
    initial.initial_constraint_residual = 1.0e-3;
    initial.initial_boundary_residual = 0.0;
    initial.residual_tolerance = 1.0e-8;
    summaries.initial_compatibility.push_back(initial);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report,
                                   PropertyKind::DifferentialAlgebraicStructure,
                                   PropertyStatus::Likely));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::InitialDataCompatibility,
                                       "CompatibilityAnalyzer",
                                       PropertyStatus::Violated));
}

TEST(Phase7Integration, InitialDataCompatibility_SatisfiedAtInitialTime)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    InitialCompatibilitySummary initial;
    initial.initial_constraint_residual = 1.0e-12;
    initial.initial_boundary_residual = 2.0e-12;
    initial.residual_tolerance = 1.0e-8;
    summaries.initial_compatibility.push_back(initial);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* claim = firstClaimFrom(report,
                                       PropertyKind::InitialDataCompatibility,
                                       "CompatibilityAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(claim->initial_data_compatible.has_value());
    EXPECT_TRUE(*claim->initial_data_compatible);
}

TEST(Phase7Integration, Adjoint_NonAdjointConsistentBoundary_GoalRisk)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    AdjointConsistencySummary adjoint;
    adjoint.contribution_id = "boundary-goal-block";
    adjoint.goal_functional_id = "generic-goal";
    adjoint.adjoint_consistency = AdjointConsistencyKind::No;
    adjoint.transpose_backend_support = true;
    summaries.adjoint_consistency.push_back(adjoint);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::AdjointConsistency,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Violated));
}

TEST(Phase7Integration, Conservation_InterfaceFluxMismatch_Violation)
{
    ProblemAnalysisContext ctx;
    auto c0 = ContributionDescriptor::exchangeCoupling(
        VariableKey::field(0), VariableKey::field(1),
        "generic-exchange", "interface-exchange", "phase7-toy");
    c0.domain = DomainKind::InterfaceFace;
    c0.balance->sign = 1;
    ctx.addContribution(c0);
    auto c1 = ContributionDescriptor::exchangeCoupling(
        VariableKey::field(1), VariableKey::field(0),
        "generic-exchange", "interface-exchange", "phase7-toy");
    c1.domain = DomainKind::InterfaceFace;
    c1.balance->sign = -1;
    ctx.addContribution(c1);

    AnalysisSummarySet summaries;
    FluxBalanceSummary flux;
    flux.block = blockFor(VariableKey::field(0), "interface-exchange",
                          DomainKind::InterfaceFace);
    flux.balance_tolerance = 1.0e-8;
    flux.interface_pair_residual_norm = 1.0e-3;
    flux.local_violation_count = 1;
    flux.interface_pair_count = 1;
    summaries.flux_balances.push_back(flux);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::ConservationStructure,
                                       "ConservationAnalyzer",
                                       PropertyStatus::Violated));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::InterfaceCondition,
                                       "InterfaceValidationAnalyzer",
                                       PropertyStatus::Violated));
}

#if FE_HAS_MPI
TEST(Phase7Integration, ScalarDiffusion_MPI_ReducedScanMatchesSerial)
{
    if (std::getenv("SVMP_FE_RUN_MPI_TESTS") == nullptr) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_MPI_TESTS=1 and run under mpiexec.";
    }

    struct MpiSession {
        bool owns_initialization{false};
        ~MpiSession()
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (owns_initialization && finalized == 0) {
                MPI_Finalize();
            }
        }
    } session;

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        int argc = 0;
        char** argv = nullptr;
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
        session.owns_initialization = true;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This MPI integration test expects exactly two ranks.";
    }

    constexpr GlobalIndex rows = 4;
    const GlobalIndex begin = rank == 0 ? 0 : 2;
    const GlobalIndex end = rank == 0 ? 2 : 4;
    const auto local_source = CsrSparseRowScanSource::fromRows(
        rows, rows, tridiagonalRows(rows, begin, end),
        backends::BackendKind::FSILS, begin, false, rank);
    const auto serial_source = CsrSparseRowScanSource::fromRows(
        rows, rows, tridiagonalRows(rows, 0, rows),
        backends::BackendKind::FSILS);
    const auto reduction = ConstraintReductionMask::fromConstrainedDofs(
        rows, {0}, ConstraintReductionKind::StrongDirichletElimination);

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.compute_symmetry = false;

    auto serial = scanReducedFreeFreeSummary(serial_source, reduction, {},
                                             options);
    options.mpi_comm = MPI_COMM_WORLD;
    auto mpi = scanReducedFreeFreeSummary(local_source, reduction, {},
                                          options);

    expectSameMatrixClassification(serial.free_free_matrix,
                                   mpi.free_free_matrix);
    EXPECT_EQ(mpi.free_dof_count, serial.free_dof_count);
    EXPECT_EQ(mpi.constrained_dof_count, serial.constrained_dof_count);
}
#endif
