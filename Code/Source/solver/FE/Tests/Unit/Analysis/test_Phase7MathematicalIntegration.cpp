/**
 * @file test_Phase7MathematicalIntegration.cpp
 * @brief Phase 7 generic mathematical integration tests for FE analysis.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AdvancedStabilityAnalyzers.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/SolverCompatibilityAnalyzer.h"
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
    matrix.sign_evidence_complete = true;
    matrix.row_sum_evidence_complete = true;
    matrix.sign_tolerance = 1.0e-12;
    matrix.row_sum_tolerance = 1.0e-12;
    matrix.diagonal_count = 3;
    matrix.offdiag_count = 4;
    matrix.scanned_row_count = 3;
    matrix.expected_row_count = 3;
    matrix.scanned_entry_count = 7;
    matrix.nonpositive_diagonal_count = 0;
    matrix.negative_diagonal_count = 0;
    matrix.near_zero_diagonal_count = 0;
    matrix.min_row_sum = 0.0;
    matrix.max_row_sum = 1.0;
    matrix.min_eigenvalue_estimate = 1.0;
    matrix.coercivity_lower_bound = 1.0;
    matrix.cholesky_factorization_succeeded = true;
    matrix.m_matrix_certification_evidence = z_matrix;
    matrix.stieltjes_matrix_evidence = z_matrix;
    if (z_matrix) {
        matrix.m_matrix_theorem_id = "stieltjes-spd-z";
    }
    matrix.dmp_applicability_evidence = z_matrix;
    matrix.dmp_rhs_sign_evidence = z_matrix;

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
    coeff.coefficient_region_coverage_complete = true;
    coeff.quadrature_point_coverage_complete = true;
    coeff.lower_bound_valid_for_all_samples = true;
    coeff.tolerance_metadata_present = true;
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

const PropertyClaim* firstClaimForBlock(const ProblemAnalysisReport& report,
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
    auto coeff = positiveCoefficient();
    coeff.block = blockFor(scalar, "acute-scalar");
    summaries.coefficient_properties.push_back(std::move(coeff));
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
    auto coeff = positiveCoefficient();
    coeff.block = blockFor(scalar, "obtuse-scalar");
    summaries.coefficient_properties.push_back(std::move(coeff));
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
    auto reduced = scanReducedFreeFreeSummary(
        source, mask, blockFor(scalar, "reduced"), options);
    reduced.free_free_matrix.m_matrix_certification_evidence = true;
    reduced.free_free_matrix.stieltjes_matrix_evidence = true;
    reduced.free_free_matrix.m_matrix_theorem_id = "stieltjes-spd-z";
    reduced.free_free_matrix.cholesky_factorization_succeeded = true;
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
    estimate.estimate_tolerance = 1.0e-10;
    estimate.test_rows = 12;
    estimate.test_cols = 4;
    estimate.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    estimate.estimate_scope = "free-free";
    estimate.estimator_metadata_present = true;
    estimate.norm_metadata_present = true;
    estimate.mesh_refinement_evidence_present = true;
    estimate.mesh_refinement_sample_count = 3;
    estimate.uniform_lower_bound_evidence_present = true;
    estimate.uniform_lower_bound_value_present = true;
    estimate.uniform_lower_bound = 0.2;
    estimate.mesh_family_scope_present = true;
    estimate.boundary_condition_scope_present = true;
    estimate.inf_sup_theorem_id = "uniform discrete LBB lower-bound study";
    estimate.block.operator_tag = "mixed-pair";
    estimate.block.test_variables = {primary, multiplier};
    estimate.block.trial_variables = {primary, multiplier};
    summaries.inf_sup_estimates.push_back(estimate);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MixedSaddlePoint,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::InfSupCondition,
                                   PropertyStatus::Preserved));
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
    EXPECT_EQ(*numeric->inf_sup_class, InfSupClass::NumericallySupported);
    EXPECT_EQ(numeric->status, PropertyStatus::Preserved);
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
                                   PropertyStatus::Violated));
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
    peclet.role = ParameterScaleRole::PecletLike;
    peclet.block = blockFor(scalar, "first-order-operator");
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
    penalty.role = ParameterScaleRole::WeakBoundaryPenalty;
    penalty.block = symbol.block;
    penalty.max_scale_value = 0.25;
    penalty.required_lower_bound_present = true;
    penalty.required_lower_bound = 1.0;
    penalty.trace_inverse_metadata_present = true;
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
    complex.compatible_complex_theorem_id =
        "Arnold-Falk-Winther bounded cochain projection";
    complex.bounded_cochain_projection_evidence_present = true;
    complex.projection_bound_present = true;
    complex.projection_bound = 2.0;
    complex.projection_stability_metadata_present = true;
    complex.mesh_family_scope_present = true;
    complex.shape_regular_mesh_evidence_present = true;
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
    temporal.amplification_radius_present = true;
    temporal.stability_metadata_present = true;
    temporal.operator_normality_evidence_present = true;
    temporal.stability_theorem_id = "Dahlquist A-stability theorem";
    temporal.stability_region_evidence_present = true;
    temporal.operator_spectrum_coverage_present = true;
    summaries.temporal_stability.push_back(temporal);
    EnergyEntropySummary energy;
    energy.energy_entropy_id = "quadratic-energy";
    energy.law_kind = EnergyEntropyLawKind::Energy;
    energy.expected_production_sign = BalanceSignClass::Nonpositive;
    energy.balance_tolerance = 1.0e-8;
    energy.observed_discrete_balance = 0.0;
    energy.observed_production = -1.0e-5;
    energy.energy_functional_id = "quadratic-energy";
    energy.energy_norm_id = "mass-inner-product";
    energy.energy_entropy_theorem_id = "discrete energy identity";
    energy.energy_functional_metadata_present = true;
    energy.energy_norm_metadata_present = true;
    energy.energy_positivity_evidence_present = true;
    energy.energy_coercivity_evidence_present = true;
    energy.discrete_dissipation_identity_evidence_present = true;
    energy.boundary_source_energy_accounting_present = true;
    energy.energy_coercivity_lower_bound_present = true;
    energy.energy_coercivity_lower_bound = 0.25;
    energy.energy_norm_equivalence_bounds_present = true;
    energy.energy_norm_equivalence_lower_bound = 0.25;
    energy.energy_norm_equivalence_upper_bound = 4.0;
    energy.energy_dissipation_residual_bound_present = true;
    energy.energy_dissipation_residual_bound = 1.0e-12;
    energy.energy_dissipation_tolerance_present = true;
    energy.energy_dissipation_tolerance = 1.0e-8;
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
    exact.tangent_symmetry = SymmetryClass::Symmetric;
    exact.tangent_positivity = PositivityClass::Positive;
    exact.jacobian_action_available = true;
    exact.finite_difference_tolerance = 1.0e-8;
    exact.finite_difference_action_error = 1.0e-12;
    summaries.nonlinear_tangents.push_back(exact);
    NonlinearTangentSummary bad;
    bad.residual_id = "inconsistent-residual";
    bad.block = blockFor(VariableKey::field(0), "inconsistent-tangent");
    bad.tangent_consistency = TangentConsistencyClass::Inconsistent;
    bad.tangent_symmetry = SymmetryClass::Nonsymmetric;
    bad.tangent_positivity = PositivityClass::Indefinite;
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
    moving.time_integration_metadata_present = true;
    moving.remap_metadata_present = true;
    moving.geometric_conservation_tolerance_declared = true;
    moving.metric_identity_evidence_present = true;
    moving.free_stream_preservation_residual_present = true;
    moving.free_stream_preservation_residual = 0.0;
    moving.gcl_theorem_id = "ALE discrete geometric conservation theorem";
    moving.constant_state_scope = "constant free-stream states";
    moving.mesh_update_time_scheme = "matching ALE update";
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
    good.residual_tolerance = 1.0e-10;
    good.rank_metadata_present = true;
    good.interface_scope_metadata_present = true;
    good.projection_consistency_metadata_present = true;
    good.mortar_inf_sup_or_dual_consistency_metadata_present = true;
    good.interface_mass_conditioning_metadata_present = true;
    good.action_reaction_flux_metadata_present = true;
    summaries.transfer_operators.push_back(good);
    TransferOperatorSummary bad;
    bad.interface_pair_id = "pair-bad";
    bad.projection_space_id = "projection-bad";
    bad.conservation_residual = 1.0e-4;
    bad.constant_preservation_residual = 1.0e-3;
    bad.residual_tolerance = 1.0e-10;
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
    ok.cfl_condition_satisfied = true;
    ok.cfl_estimate_present = true;
    ok.cfl_estimate = 0.4;
    ok.accepted_cfl_bound_present = true;
    ok.accepted_cfl_bound = 0.5;
    ok.wave_speed_bound_present = true;
    ok.wave_speed_bound = 2.0;
    ok.time_step_scope = "SSP forward-Euler stage";
    ok.mesh_size_scope = "active cell family";
    ok.ssp_time_discretization_evidence_present = true;
    ok.source_admissibility_evidence_present = true;
    ok.low_order_invariant_domain_evidence_present = true;
    ok.convex_limiting_evidence_present = true;
    ok.spatial_monotonicity_evidence_present = true;
    ok.mass_positivity_evidence_present = true;
    ok.invariant_domain_theorem_id = "Guermond-Popov invariant-domain theorem";
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
    equilibrium.equilibrium_family_id = "lake-at-rest";
    equilibrium.equilibrium_preservation_theorem_id =
        "hydrostatic reconstruction well-balanced theorem";
    equilibrium.flux_source_residual = 0.0;
    equilibrium.residual_tolerance = 1.0e-10;
    equilibrium.source_quadrature_metadata_present = true;
    equilibrium.reconstruction_metadata_present = true;
    equilibrium.boundary_compatibility_metadata_present = true;
    equilibrium.equilibrium_scope_metadata_present = true;
    equilibrium.source_model_scope_metadata_present = true;
    equilibrium.reconstruction_scope_metadata_present = true;
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
    initial.residual_tolerance_declared = true;
    initial.compatibility_scope = "algebraic-constraint";
    initial.algebraic_constraint_metadata_present = true;
    initial.checked_constraint_family_count = 1;
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
    initial.residual_tolerance_declared = true;
    initial.compatibility_scope = "algebraic-and-boundary";
    initial.algebraic_constraint_metadata_present = true;
    initial.boundary_constraint_metadata_present = true;
    initial.checked_constraint_family_count = 1;
    initial.checked_boundary_condition_count = 1;
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

TEST(Phase7Integration, ReducedDefinitenessRequiresSpectralOrFactorizationEvidence)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);

    AnalysisSummarySet summaries;
    auto positive_diagonal_only =
        reducedMatrix(scalar, "positive-diagonal-only", true);
    positive_diagonal_only.free_free_matrix.min_eigenvalue_estimate.reset();
    positive_diagonal_only.free_free_matrix.coercivity_lower_bound.reset();
    positive_diagonal_only.free_free_matrix.cholesky_factorization_succeeded = false;
    summaries.reduced_matrices.push_back(positive_diagonal_only);

    auto indefinite = reducedMatrix(scalar, "symmetric-indefinite", true);
    indefinite.free_free_matrix.min_eigenvalue_estimate = -1.0;
    indefinite.free_free_matrix.coercivity_lower_bound = -1.0;
    indefinite.free_free_matrix.cholesky_factorization_succeeded = false;
    summaries.reduced_matrices.push_back(indefinite);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* unknown = firstClaimForBlock(
        report, PropertyKind::OperatorDefiniteness,
        "OperatorClassAnalyzer", "positive-diagonal-only");
    ASSERT_NE(unknown, nullptr);
    EXPECT_EQ(unknown->status, PropertyStatus::Unknown);
    EXPECT_EQ(*unknown->coercivity_class, CoercivityClass::Unknown);

    const auto* violated = firstClaimForBlock(
        report, PropertyKind::OperatorDefiniteness,
        "OperatorClassAnalyzer", "symmetric-indefinite");
    ASSERT_NE(violated, nullptr);
    EXPECT_EQ(violated->status, PropertyStatus::Violated);
    EXPECT_EQ(*violated->coercivity_class, CoercivityClass::Indefinite);
}

TEST(Phase7Integration, TransportSummaryScopesRejectUnrelatedPecletCflAndNonnormalEvidence)
{
    ProblemAnalysisContext ctx;
    const auto target = VariableKey::field(0);
    const auto other = VariableKey::field(1);
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addFieldDescriptor(scalarH1(1));
    ctx.addContribution(ContributionDescriptor::transportLike(
        target, "target-transport", "phase7-scope"));

    AnalysisSummarySet summaries;
    ParameterScaleSummary unrelated_peclet;
    unrelated_peclet.role = ParameterScaleRole::PecletLike;
    unrelated_peclet.block = blockFor(other, "other-transport");
    unrelated_peclet.max_scale_value = 100.0;
    summaries.parameter_scales.push_back(unrelated_peclet);

    ParameterScaleSummary target_peclet;
    target_peclet.role = ParameterScaleRole::PecletLike;
    target_peclet.block = blockFor(target, "target-transport");
    target_peclet.max_scale_value = 0.25;
    summaries.parameter_scales.push_back(target_peclet);

    TemporalStabilitySummary unrelated_temporal;
    unrelated_temporal.block = blockFor(other, "other-transport");
    unrelated_temporal.cfl_estimate = 8.0;
    unrelated_temporal.cfl_estimate_present = true;
    summaries.temporal_stability.push_back(unrelated_temporal);

    TemporalStabilitySummary target_temporal;
    target_temporal.block = blockFor(target, "target-transport");
    target_temporal.cfl_estimate = 0.5;
    target_temporal.cfl_estimate_present = true;
    summaries.temporal_stability.push_back(target_temporal);

    DiscreteMatrixSummary unrelated_matrix;
    unrelated_matrix.block = blockFor(other, "other-transport");
    unrelated_matrix.nonnormality_indicator = 5.0;
    summaries.discrete_matrices.push_back(unrelated_matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* transport = firstClaimFrom(
        report, PropertyKind::OperatorTransportCharacter,
        "TransportCharacterAnalyzer");
    ASSERT_NE(transport, nullptr);
    ASSERT_TRUE(transport->peclet_number.has_value());
    EXPECT_DOUBLE_EQ(*transport->peclet_number, 0.25);
    ASSERT_TRUE(transport->cfl_number.has_value());
    EXPECT_DOUBLE_EQ(*transport->cfl_number, 0.5);
    EXPECT_FALSE(transport->nonnormality_indicator.has_value());
}

TEST(Phase7Integration, TransportNonsymmetricNormalMatrixDoesNotCreateNonnormalClaim)
{
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(ContributionDescriptor::transportLike(
        scalar, "normal-rotation", "phase7-scope"));

    AnalysisSummarySet summaries;
    DiscreteMatrixSummary normal_rotation;
    normal_rotation.block = blockFor(scalar, "normal-rotation");
    normal_rotation.rows = 2;
    normal_rotation.cols = 2;
    normal_rotation.square = true;
    normal_rotation.max_abs_entry = 1.0;
    normal_rotation.max_symmetry_error = 2.0; // [0 -1; 1 0] is nonsymmetric but normal.
    normal_rotation.nonsymmetry_indicator = 2.0;
    summaries.discrete_matrices.push_back(normal_rotation);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* transport = firstClaimFrom(
        report, PropertyKind::OperatorTransportCharacter,
        "TransportCharacterAnalyzer");
    ASSERT_NE(transport, nullptr);
    EXPECT_FALSE(transport->nonnormality_indicator.has_value());
    ASSERT_TRUE(transport->transport_character_class.has_value());
    EXPECT_NE(*transport->transport_character_class,
              TransportCharacterClass::NonNormalRisk);
}

TEST(Phase7Integration, InterfacePenaltyScopingPreventsCrossBoundaryCertification)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;

    BoundarySymbolSummary weak_boundary;
    weak_boundary.block = blockFor(VariableKey::field(0), "weak-face",
                                   DomainKind::InteriorFace);
    weak_boundary.complementing_condition_satisfied = true;
    summaries.boundary_symbols.push_back(weak_boundary);

    BoundarySymbolSummary strong_boundary;
    strong_boundary.block = blockFor(VariableKey::field(0), "strong-face",
                                     DomainKind::InteriorFace);
    strong_boundary.complementing_condition_satisfied = true;
    summaries.boundary_symbols.push_back(strong_boundary);

    ParameterScaleSummary weak_penalty;
    weak_penalty.role = ParameterScaleRole::WeakBoundaryPenalty;
    weak_penalty.block = weak_boundary.block;
    weak_penalty.max_scale_value = 0.25;
    weak_penalty.required_lower_bound_present = true;
    weak_penalty.required_lower_bound = 1.0;
    weak_penalty.trace_inverse_metadata_present = true;
    summaries.parameter_scales.push_back(weak_penalty);

    ParameterScaleSummary unrelated_high_penalty;
    unrelated_high_penalty.role = ParameterScaleRole::WeakBoundaryPenalty;
    unrelated_high_penalty.block = strong_boundary.block;
    unrelated_high_penalty.max_scale_value = 10.0;
    unrelated_high_penalty.required_lower_bound_present = true;
    unrelated_high_penalty.required_lower_bound = 1.0;
    unrelated_high_penalty.trace_inverse_metadata_present = true;
    unrelated_high_penalty.scale_theorem_id =
        "Nitsche trace-inverse coercivity bound";
    summaries.parameter_scales.push_back(unrelated_high_penalty);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* weak = firstClaimForBlock(
        report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer", "weak-face");
    ASSERT_NE(weak, nullptr);
    EXPECT_EQ(weak->status, PropertyStatus::Violated);
    ASSERT_TRUE(weak->penalty_scale.has_value());
    EXPECT_DOUBLE_EQ(*weak->penalty_scale, 0.25);

    const auto* strong = firstClaimForBlock(
        report, PropertyKind::WeakBoundaryCoercivity,
        "InterfaceValidationAnalyzer", "strong-face");
    ASSERT_NE(strong, nullptr);
    EXPECT_EQ(strong->status, PropertyStatus::Preserved);
    ASSERT_TRUE(strong->penalty_scale.has_value());
    EXPECT_DOUBLE_EQ(*strong->penalty_scale, 10.0);
}

TEST(Phase7Integration, ScalarP1Diffusion_UniformMesh_ProvenStieltjesDMP)
{
    // The 3x3 tridiagonal stiffness matrix below is the free-free operator for
    // 1D P1 diffusion on a uniform mesh after strong Dirichlet elimination. It
    // is symmetric positive definite with eigenvalues 2 - sqrt(2), 2, 2 + sqrt(2),
    // has nonpositive off-diagonals, and is a Stieltjes M-matrix.
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(scalarGradientOperator(scalar, "p1-diffusion"));

    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 2.0}, {1, -1.0}},
        {{0, -1.0}, {1, 2.0}, {2, -1.0}},
        {{1, -1.0}, {2, 2.0}},
    };
    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;
    auto matrix = scanSparseMatrixSummary(
        CsrSparseRowScanSource::fromRows(3, 3, rows),
        blockFor(scalar, "p1-diffusion"), options).summary;
    matrix.min_eigenvalue_estimate = 0.5857864376269049;
    matrix.coercivity_lower_bound = 0.5857864376269049;
    matrix.cholesky_factorization_succeeded = true;
    matrix.m_matrix_certification_evidence = true;
    matrix.stieltjes_matrix_evidence = true;
    matrix.m_matrix_theorem_id = "stieltjes-spd-z";
    matrix.dmp_applicability_evidence = true;
    matrix.dmp_rhs_sign_evidence = true;

    AnalysisSummarySet summaries;
    auto coeff = positiveCoefficient("p1-diffusion-coeff");
    coeff.block = matrix.block;
    summaries.coefficient_properties.push_back(coeff);
    summaries.discrete_matrices.push_back(matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::ZMatrixStructure,
                                       "DiscreteMonotonicityAnalyzer",
                                       PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimFromWithStatus(report, PropertyKind::MMatrixStructure,
                                       "DiscreteMonotonicityAnalyzer",
                                       PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::DiscreteMaximumPrinciple,
                                       "DiscreteMonotonicityAnalyzer",
                                       PropertyStatus::Preserved));
}

TEST(Phase7Integration, DiscreteMaximumPrincipleRequiresScopedApplicabilityGates)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(scalarGradientOperator(scalar, "gated-diffusion"));

    AnalysisSummarySet summaries;
    auto unrelated_coeff = positiveCoefficient("unrelated-coeff");
    unrelated_coeff.block = blockFor(VariableKey::field(1), "other-diffusion");
    summaries.coefficient_properties.push_back(unrelated_coeff);
    summaries.reduced_matrices.push_back(
        reducedMatrix(scalar, "gated-diffusion", true));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* dmp = firstClaimForBlock(
        report, PropertyKind::DiscreteMaximumPrinciple,
        "DiscreteMonotonicityAnalyzer", "gated-diffusion");
    ASSERT_NE(dmp, nullptr);
    EXPECT_EQ(dmp->status, PropertyStatus::Unknown);
}

TEST(Phase7Integration, DiscreteMaximumPrincipleRejectsTransportContamination)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(scalarGradientOperator(scalar, "contaminated"));
    ctx.addContribution(ContributionDescriptor::transportLike(
        scalar, "contaminated", "phase7-contamination"));

    AnalysisSummarySet summaries;
    auto coeff = positiveCoefficient("scoped-coeff");
    coeff.block = blockFor(scalar, "contaminated");
    summaries.coefficient_properties.push_back(coeff);
    summaries.reduced_matrices.push_back(
        reducedMatrix(scalar, "contaminated", true));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* dmp = firstClaimForBlock(
        report, PropertyKind::DiscreteMaximumPrinciple,
        "DiscreteMonotonicityAnalyzer", "contaminated");
    ASSERT_NE(dmp, nullptr);
    EXPECT_EQ(dmp->status, PropertyStatus::Unknown);
}

TEST(Phase7Integration, MeshAspectRatioRequiresThresholdBeforeRisk)
{
    ProblemAnalysisContext ok_ctx;
    AnalysisSummarySet ok_summaries;
    MeshGeometryQualitySummary ok_mesh;
    ok_mesh.min_jacobian = 0.5;
    ok_mesh.max_jacobian = 1.5;
    ok_mesh.max_aspect_ratio = 2.0;
    ok_mesh.aspect_ratio_warning_threshold = 10.0;
    ok_summaries.mesh_geometry_quality.push_back(ok_mesh);
    ok_ctx.setAnalysisSummaries(std::move(ok_summaries));
    auto ok_report = analyze(std::move(ok_ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(ok_report,
                                       PropertyKind::MeshGeometryValidity,
                                       "MeshGeometryAnalyzer",
                                       PropertyStatus::Preserved));

    ProblemAnalysisContext bad_ctx;
    AnalysisSummarySet bad_summaries;
    MeshGeometryQualitySummary bad_mesh;
    bad_mesh.min_jacobian = 0.5;
    bad_mesh.max_jacobian = 1.5;
    bad_mesh.max_aspect_ratio = 25.0;
    bad_mesh.aspect_ratio_warning_threshold = 10.0;
    bad_summaries.mesh_geometry_quality.push_back(bad_mesh);
    bad_ctx.setAnalysisSummaries(std::move(bad_summaries));
    auto bad_report = analyze(std::move(bad_ctx));
    EXPECT_TRUE(hasClaimFromWithStatus(bad_report,
                                       PropertyKind::MeshGeometryValidity,
                                       "MeshGeometryAnalyzer",
                                       PropertyStatus::Likely));
}

TEST(Phase7Integration, NonlinearTangentConsumesSymmetryAndPositivityMetadata)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;

    NonlinearTangentSummary good;
    good.residual_id = "good";
    good.block = blockFor(VariableKey::field(0), "good-tangent");
    good.tangent_consistency = TangentConsistencyClass::Exact;
    good.tangent_symmetry = SymmetryClass::Symmetric;
    good.tangent_positivity = PositivityClass::Positive;
    good.jacobian_action_available = true;
    good.finite_difference_tolerance = 1.0e-8;
    good.finite_difference_action_error = 1.0e-12;
    summaries.nonlinear_tangents.push_back(good);

    NonlinearTangentSummary bad;
    bad.residual_id = "bad";
    bad.block = blockFor(VariableKey::field(0), "bad-tangent");
    bad.tangent_consistency = TangentConsistencyClass::Exact;
    bad.tangent_symmetry = SymmetryClass::Nonsymmetric;
    bad.tangent_positivity = PositivityClass::Indefinite;
    bad.jacobian_action_available = true;
    bad.finite_difference_tolerance = 1.0e-8;
    bad.finite_difference_action_error = 1.0e-12;
    summaries.nonlinear_tangents.push_back(bad);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* good_claim = firstClaimForBlock(
        report, PropertyKind::NonlinearTangentStructure,
        "NonlinearTangentAnalyzer", "good-tangent");
    ASSERT_NE(good_claim, nullptr);
    EXPECT_EQ(good_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(good_claim->operator_symmetry_class.has_value());
    EXPECT_EQ(*good_claim->operator_symmetry_class,
              OperatorSymmetryClass::Symmetric);
    ASSERT_TRUE(good_claim->coercivity_class.has_value());
    EXPECT_EQ(*good_claim->coercivity_class, CoercivityClass::Coercive);

    const auto* bad_claim = firstClaimForBlock(
        report, PropertyKind::NonlinearTangentStructure,
        "NonlinearTangentAnalyzer", "bad-tangent");
    ASSERT_NE(bad_claim, nullptr);
    EXPECT_EQ(bad_claim->status, PropertyStatus::Violated);
    ASSERT_TRUE(bad_claim->operator_symmetry_class.has_value());
    EXPECT_EQ(*bad_claim->operator_symmetry_class,
              OperatorSymmetryClass::Nonsymmetric);
    ASSERT_TRUE(bad_claim->coercivity_class.has_value());
    EXPECT_EQ(*bad_claim->coercivity_class, CoercivityClass::Indefinite);
}

TEST(Phase7Integration, ScopedEvidenceDoesNotTreatContributionIdAsOperatorTag)
{
    const auto scalar = VariableKey::field(0);
    auto target = blockFor(scalar, "assembled-operator");

    CoefficientPropertySummary by_id_only;
    by_id_only.contribution_id = "assembled-operator";
    by_id_only.positivity = PositivityClass::Positive;
    EXPECT_FALSE(coefficientSummaryMatches(by_id_only, target));

    target.contribution_id = "forms:block:diffusion";
    by_id_only.contribution_id = "forms:block:diffusion";
    EXPECT_TRUE(coefficientSummaryMatches(by_id_only, target));

    by_id_only.contribution_id = "forms:block:transport";
    EXPECT_FALSE(coefficientSummaryMatches(by_id_only, target));

    DiscreteMatrixSummary aggregate;
    aggregate.block = blockFor(scalar, "assembled-operator");
    aggregate.contribution_ids = {"forms:block:diffusion"};
    by_id_only.contribution_id = "forms:block:diffusion";
    EXPECT_TRUE(coefficientSummaryMatches(by_id_only, aggregate));

    aggregate.contribution_ids.clear();
    EXPECT_FALSE(coefficientSummaryMatches(by_id_only, aggregate));
}

TEST(Phase7Integration, DmpRejectsSameVariableTransportWithDifferentTagWhenMatrixUnprovenanced)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarH1(0));
    ctx.addContribution(scalarGradientOperator(scalar, "diffusion-part"));
    ctx.addContribution(ContributionDescriptor::transportLike(
        scalar, "transport-part", "phase7-aggregate"));

    AnalysisSummarySet summaries;
    auto coeff = positiveCoefficient("aggregate-coeff");
    coeff.block = blockFor(scalar, "aggregate-matrix");
    summaries.coefficient_properties.push_back(coeff);
    summaries.reduced_matrices.push_back(
        reducedMatrix(scalar, "aggregate-matrix", true));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* dmp = firstClaimForBlock(
        report, PropertyKind::DiscreteMaximumPrinciple,
        "DiscreteMonotonicityAnalyzer", "aggregate-matrix");
    ASSERT_NE(dmp, nullptr);
    EXPECT_EQ(dmp->status, PropertyStatus::Unknown);
    EXPECT_NE(dmp->evidence.front().description.find(
                  "absence of transport contamination"),
              std::string::npos);
}

TEST(Phase7Integration, DmpUsesCompleteContributionProvenanceToExcludeUnrelatedTransport)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarH1(0));
    auto diffusion = scalarGradientOperator(scalar, "diffusion-part");
    const auto diffusion_id = diffusion.contribution_id;
    ASSERT_FALSE(diffusion_id.empty());
    ctx.addContribution(std::move(diffusion));
    ctx.addContribution(ContributionDescriptor::transportLike(
        scalar, "transport-part", "phase7-aggregate"));

    AnalysisSummarySet summaries;
    auto coeff = positiveCoefficient("aggregate-coeff");
    coeff.block = blockFor(scalar, "aggregate-matrix");
    summaries.coefficient_properties.push_back(coeff);
    auto reduced = reducedMatrix(scalar, "aggregate-matrix", true);
    reduced.free_free_matrix.contribution_ids = {diffusion_id};
    reduced.free_free_matrix.contribution_tags = {"diffusion-part"};
    reduced.free_free_matrix.contribution_provenance_complete = true;
    summaries.reduced_matrices.push_back(reduced);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* dmp = firstClaimForBlock(
        report, PropertyKind::DiscreteMaximumPrinciple,
        "DiscreteMonotonicityAnalyzer", "aggregate-matrix");
    ASSERT_NE(dmp, nullptr);
    EXPECT_EQ(dmp->status, PropertyStatus::Preserved);
}

TEST(Phase7Integration, SolverCgRequiresPositiveDefiniteEvidence)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;

    AnalysisSummarySet summaries;
    DiscreteMatrixSummary matrix;
    matrix.block = blockFor(scalar, "positive-diagonal-only");
    matrix.rows = 2;
    matrix.cols = 2;
    matrix.square = true;
    matrix.symmetry_evidence_complete = true;
    matrix.structurally_symmetric = true;
    matrix.numerically_symmetric = true;
    matrix.diagonal_count = 2;
    matrix.nonpositive_diagonal_count = 0;
    matrix.negative_diagonal_count = 0;
    summaries.discrete_matrices.push_back(matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    backends::SolverOptions options;
    options.method = backends::SolverMethod::CG;
    options.preconditioner = backends::PreconditionerType::Diagonal;
    ctx.setSolverOptions(options);

    auto report = analyze(std::move(ctx));
    const auto* solver = firstClaimFrom(
        report, PropertyKind::SolverCompatibility,
        "SolverCompatibilityAnalyzer");
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->status, PropertyStatus::Unknown);
    ASSERT_TRUE(solver->certification_class.has_value());
    EXPECT_EQ(*solver->certification_class, CertificationClass::NotCertified);
}

TEST(Phase7Integration, SolverCgRequiresSpdEvidenceToCoverActiveSystem)
{
    const auto u = VariableKey::field(0);
    const auto v = VariableKey::field(1);

    auto make_spd_matrix = [](VariableKey variable, std::string tag) {
        DiscreteMatrixSummary matrix;
        matrix.block = blockFor(variable, std::move(tag));
        matrix.rows = 2;
        matrix.cols = 2;
        matrix.square = true;
        matrix.scope = NumericSummaryScope::ReducedFreeFree;
        matrix.symmetry_evidence_complete = true;
        matrix.structurally_symmetric = true;
        matrix.numerically_symmetric = true;
        matrix.diagonal_count = 2;
        matrix.nonpositive_diagonal_count = 0;
        matrix.negative_diagonal_count = 0;
        matrix.min_eigenvalue_estimate = 1.0;
        matrix.coercivity_lower_bound = 1.0;
        matrix.cholesky_factorization_succeeded = true;
        return matrix;
    };

    ProblemAnalysisContext local_ctx;
    local_ctx.addFieldDescriptor(scalarH1(0, 1, "u"));
    local_ctx.addFieldDescriptor(scalarH1(1, 1, "v"));
    AnalysisSummarySet local_summaries;
    local_summaries.discrete_matrices.push_back(make_spd_matrix(u, "u-block"));
    local_ctx.setAnalysisSummaries(std::move(local_summaries));
    backends::SolverOptions options;
    options.method = backends::SolverMethod::CG;
    options.preconditioner = backends::PreconditionerType::Diagonal;
    local_ctx.setSolverOptions(options);

    auto local_report = analyze(std::move(local_ctx));
    const auto* local_solver = firstClaimFrom(
        local_report, PropertyKind::SolverCompatibility,
        "SolverCompatibilityAnalyzer");
    ASSERT_NE(local_solver, nullptr);
    EXPECT_EQ(local_solver->status, PropertyStatus::Unknown);
    ASSERT_TRUE(local_solver->certification_class.has_value());
    EXPECT_EQ(*local_solver->certification_class,
              CertificationClass::NotCertified);

    ProblemAnalysisContext full_ctx;
    full_ctx.addFieldDescriptor(scalarH1(0, 1, "u"));
    full_ctx.addFieldDescriptor(scalarH1(1, 1, "v"));
    AnalysisSummarySet full_summaries;
    DiscreteMatrixSummary full_matrix = make_spd_matrix(v, "unused-block");
    full_matrix.block = OperatorBlockId{};
    full_matrix.scope = NumericSummaryScope::FullMatrix;
    full_summaries.discrete_matrices.push_back(full_matrix);
    full_ctx.setAnalysisSummaries(std::move(full_summaries));
    full_ctx.setSolverOptions(options);

    auto full_report = analyze(std::move(full_ctx));
    const auto* full_solver = firstClaimFrom(
        full_report, PropertyKind::SolverCompatibility,
        "SolverCompatibilityAnalyzer");
    ASSERT_NE(full_solver, nullptr);
    EXPECT_EQ(full_solver->status, PropertyStatus::Preserved);
    ASSERT_TRUE(full_solver->certification_class.has_value());
    EXPECT_EQ(*full_solver->certification_class, CertificationClass::Certified);
}

TEST(Phase7Integration, NumericInfSupRequiresToleranceScopeAndMetadata)
{
    const auto primary = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);
    ProblemAnalysisContext ctx;

    AnalysisSummarySet summaries;
    InfSupEstimateSummary tiny;
    tiny.primal_variable = primary;
    tiny.multiplier_variable = multiplier;
    tiny.estimate_value = 1.0e-12;
    tiny.estimate_tolerance = 1.0e-8;
    tiny.test_rows = 8;
    tiny.test_cols = 4;
    tiny.estimate_scope = "free-free";
    tiny.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    tiny.estimator_metadata_present = true;
    tiny.norm_metadata_present = true;
    tiny.mesh_refinement_evidence_present = true;
    tiny.mesh_refinement_sample_count = 3;
    tiny.uniform_lower_bound_evidence_present = true;
    summaries.inf_sup_estimates.push_back(tiny);

    InfSupEstimateSummary missing_metadata = tiny;
    missing_metadata.estimate_value = 0.2;
    missing_metadata.estimate_tolerance = 1.0e-8;
    missing_metadata.estimator_metadata_present = false;
    summaries.inf_sup_estimates.push_back(missing_metadata);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto infsup = report.claimsOfKind(PropertyKind::InfSupCondition);
    ASSERT_GE(infsup.size(), 2u);
    EXPECT_EQ(infsup[infsup.size() - 2]->status, PropertyStatus::Unknown);
    ASSERT_TRUE(infsup[infsup.size() - 2]->certification_class.has_value());
    EXPECT_EQ(*infsup[infsup.size() - 2]->certification_class,
              CertificationClass::NotCertified);
    EXPECT_EQ(infsup.back()->status, PropertyStatus::Likely);
    ASSERT_TRUE(infsup.back()->certification_class.has_value());
    EXPECT_EQ(*infsup.back()->certification_class,
              CertificationClass::NotCertified);
}

TEST(Phase7Integration, NumericInfSupEstimateMustScopeBothMixedVariables)
{
    const auto primary = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);
    ProblemAnalysisContext ctx;

    AnalysisSummarySet summaries;
    InfSupEstimateSummary estimate;
    estimate.primal_variable = primary;
    estimate.multiplier_variable = multiplier;
    estimate.estimate_value = 0.25;
    estimate.estimate_tolerance = 1.0e-8;
    estimate.test_rows = 8;
    estimate.test_cols = 4;
    estimate.estimate_scope = "free-free";
    estimate.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    estimate.estimator_metadata_present = true;
    estimate.norm_metadata_present = true;
    estimate.mesh_refinement_evidence_present = true;
    estimate.mesh_refinement_sample_count = 3;
    estimate.uniform_lower_bound_evidence_present = true;
    estimate.block = blockFor(primary, "wrong-scope");
    summaries.inf_sup_estimates.push_back(estimate);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* numeric = firstClaimForBlock(
        report, PropertyKind::InfSupCondition,
        "InfSupAnalyzer", "wrong-scope");
    ASSERT_NE(numeric, nullptr);
    EXPECT_EQ(numeric->status, PropertyStatus::Likely);
    ASSERT_TRUE(numeric->certification_class.has_value());
    EXPECT_EQ(*numeric->certification_class, CertificationClass::NotCertified);
}

TEST(Phase7Integration, EqualOrderH1MixedPairIsNotAutomaticallyCompatible)
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
    const auto* compat = firstClaimFrom(
        report, PropertyKind::SpaceCompatibility,
        "SpaceCompatibilityAnalyzer");
    ASSERT_NE(compat, nullptr);
    EXPECT_EQ(compat->status, PropertyStatus::Likely);
    ASSERT_TRUE(compat->space_compatibility_class.has_value());
    EXPECT_EQ(*compat->space_compatibility_class,
              SpaceCompatibilityClass::Incompatible);
}

TEST(Phase7Integration, HigherOrderH1MixedPairRequiresCertifiedEvidenceForCompatibility)
{
    const auto primary = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);

    auto add_mixed_pair = [&](ProblemAnalysisContext& ctx) {
        ctx.addFieldDescriptor(vectorH1(1, 2, "primary"));
        ctx.addFieldDescriptor(scalarH1(0, 1, "multiplier"));
        ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
            primary, "mixed-primary", "phase7-toy"));
        ctx.addContribution(ContributionDescriptor::constraintPairDesc(
            primary, multiplier, "generic-pair", "mixed-pair",
            "phase7-toy"));
    };

    ProblemAnalysisContext heuristic_ctx;
    add_mixed_pair(heuristic_ctx);
    auto heuristic_report = analyze(std::move(heuristic_ctx));
    const auto* heuristic = firstClaimFrom(
        heuristic_report, PropertyKind::SpaceCompatibility,
        "SpaceCompatibilityAnalyzer");
    ASSERT_NE(heuristic, nullptr);
    EXPECT_EQ(heuristic->status, PropertyStatus::Likely);
    ASSERT_TRUE(heuristic->certification_class.has_value());
    EXPECT_EQ(*heuristic->certification_class,
              CertificationClass::NotCertified);
    ASSERT_TRUE(heuristic->space_compatibility_class.has_value());
    EXPECT_EQ(*heuristic->space_compatibility_class,
              SpaceCompatibilityClass::WeaklyCompatible);

    ProblemAnalysisContext certified_ctx;
    add_mixed_pair(certified_ctx);
    AnalysisSummarySet summaries;
    InfSupEstimateSummary estimate;
    estimate.primal_variable = primary;
    estimate.multiplier_variable = multiplier;
    estimate.estimate_value = 0.25;
    estimate.estimate_tolerance = 1.0e-8;
    estimate.test_rows = 12;
    estimate.test_cols = 4;
    estimate.estimate_scope = "free-free";
    estimate.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    estimate.estimator_metadata_present = true;
    estimate.norm_metadata_present = true;
    estimate.mesh_refinement_evidence_present = true;
    estimate.mesh_refinement_sample_count = 3;
    estimate.uniform_lower_bound_evidence_present = true;
    estimate.uniform_lower_bound_value_present = true;
    estimate.uniform_lower_bound = 0.2;
    estimate.mesh_family_scope_present = true;
    estimate.boundary_condition_scope_present = true;
    estimate.inf_sup_theorem_id = "uniform discrete LBB lower-bound study";
    estimate.block.operator_tag = "mixed-pair";
    estimate.block.test_variables = {primary, multiplier};
    estimate.block.trial_variables = {primary, multiplier};
    summaries.inf_sup_estimates.push_back(estimate);
    certified_ctx.setAnalysisSummaries(std::move(summaries));

    auto certified_report = analyze(std::move(certified_ctx));
    const auto* certified = firstClaimFrom(
        certified_report, PropertyKind::SpaceCompatibility,
        "SpaceCompatibilityAnalyzer");
    ASSERT_NE(certified, nullptr);
    EXPECT_EQ(certified->status, PropertyStatus::Preserved);
    ASSERT_TRUE(certified->certification_class.has_value());
    EXPECT_EQ(*certified->certification_class, CertificationClass::Certified);
    ASSERT_TRUE(certified->space_compatibility_class.has_value());
    EXPECT_EQ(*certified->space_compatibility_class,
              SpaceCompatibilityClass::Compatible);
}

TEST(Phase7Integration, TemporalClaimScopesBlockAndDoesNotCertifyMissingBounds)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    TemporalStabilitySummary temporal;
    temporal.time_scheme = "metadata-only";
    temporal.block = blockFor(scalar, "time-block");
    temporal.stability_class = TemporalStabilityClass::AStable;
    temporal.stability_metadata_present = true;
    summaries.temporal_stability.push_back(temporal);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* claim = firstClaimForBlock(
        report, PropertyKind::TemporalStability,
        "TemporalStabilityAnalyzer", "time-block");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Unknown);
    EXPECT_FALSE(claim->variables.empty());
    EXPECT_EQ(claim->variables.front(), scalar);
}

TEST(Phase7Integration, CoupledSystemSummariesDoNotSuppressUncoveredFallback)
{
    const auto a = VariableKey::field(0);
    const auto b = VariableKey::field(1);
    const auto c = VariableKey::field(2);
    const auto d = VariableKey::field(3);

    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    CoupledSystemStabilitySummary summary;
    summary.coupling_group = "covered";
    summary.variables = {a, b};
    summary.monolithic_coupling = true;
    summary.coupling_tolerance = 1.0e-8;
    summary.coupling_tolerance_present = true;
    summary.exchange_residual_present = true;
    summary.constraint_drift_present = true;
    summary.interface_energy_balance_evidence_present = true;
    summary.coupled_norm_coercivity_evidence_present = true;
    summary.coupled_operator_stability_evidence_present = true;
    summaries.coupled_system_stability.push_back(summary);
    ctx.setAnalysisSummaries(std::move(summaries));

    ProblemAnalysisReport report;
    PropertyClaim existing;
    existing.kind = PropertyKind::CoupledSystemStructure;
    existing.status = PropertyStatus::Likely;
    existing.confidence = AnalysisConfidence::Medium;
    existing.variables = {c, d};
    existing.claim_origin = "CouplingGraphAnalyzer";
    report.claims.push_back(existing);
    PropertyClaim unrelated_temporal;
    unrelated_temporal.kind = PropertyKind::TemporalStability;
    unrelated_temporal.status = PropertyStatus::Preserved;
    unrelated_temporal.confidence = AnalysisConfidence::High;
    unrelated_temporal.certification_class = CertificationClass::Certified;
    unrelated_temporal.variables = {a, b};
    unrelated_temporal.claim_origin = "TemporalStabilityAnalyzer";
    report.claims.push_back(unrelated_temporal);

    CoupledSystemStabilityAnalyzer analyzer;
    analyzer.run(ctx, report);

    const auto coupled = report.claimsOfKind(PropertyKind::CoupledSystemStructure);
    ASSERT_EQ(coupled.size(), 3u);
    EXPECT_EQ(coupled[1]->claim_origin, "CoupledSystemStabilityAnalyzer");
    EXPECT_EQ(coupled[1]->status, PropertyStatus::Preserved);
    EXPECT_EQ(coupled[2]->claim_origin, "CoupledSystemStabilityAnalyzer");
    EXPECT_EQ(coupled[2]->status, PropertyStatus::Unknown);
    EXPECT_NE(std::find(coupled[2]->variables.begin(),
                        coupled[2]->variables.end(), c),
              coupled[2]->variables.end());
    EXPECT_NE(std::find(coupled[2]->variables.begin(),
                        coupled[2]->variables.end(), d),
              coupled[2]->variables.end());
}

TEST(Phase7Integration, CoupledSystemFallbackAcceptsOnlyOverlappingPreservedStability)
{
    const auto c = VariableKey::field(2);
    const auto d = VariableKey::field(3);

    ProblemAnalysisContext ctx;
    ProblemAnalysisReport report;
    PropertyClaim coupling;
    coupling.kind = PropertyKind::CoupledSystemStructure;
    coupling.status = PropertyStatus::Likely;
    coupling.confidence = AnalysisConfidence::Medium;
    coupling.variables = {c, d};
    coupling.claim_origin = "CouplingGraphAnalyzer";
    report.claims.push_back(coupling);

    PropertyClaim temporal;
    temporal.kind = PropertyKind::TemporalStability;
    temporal.status = PropertyStatus::Preserved;
    temporal.confidence = AnalysisConfidence::High;
    temporal.certification_class = CertificationClass::Certified;
    temporal.variables = {c};
    temporal.claim_origin = "TemporalStabilityAnalyzer";
    report.claims.push_back(temporal);

    CoupledSystemStabilityAnalyzer analyzer;
    analyzer.run(ctx, report);

    const auto coupled = report.claimsOfKind(PropertyKind::CoupledSystemStructure);
    ASSERT_EQ(coupled.size(), 2u);
    EXPECT_EQ(coupled.back()->claim_origin, "CoupledSystemStabilityAnalyzer");
    EXPECT_EQ(coupled.back()->status, PropertyStatus::Likely);
    ASSERT_TRUE(coupled.back()->certification_class.has_value());
    EXPECT_EQ(*coupled.back()->certification_class,
              CertificationClass::NotCertified);
}

TEST(Phase7Integration, ConservationSummaryRequiresSymbolicBalanceToCertify)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    FluxBalanceSummary flux;
    flux.block = blockFor(scalar, "numeric-balance");
    flux.balance_tolerance = 1.0e-8;
    flux.local_residual_norm = 1.0e-12;
    summaries.flux_balances.push_back(flux);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto* claim = firstClaimFrom(
        report, PropertyKind::ConservationStructure,
        "ConservationAnalyzer");
    ASSERT_NE(claim, nullptr);
    EXPECT_EQ(claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(claim->certification_class.has_value());
    EXPECT_EQ(*claim->certification_class, CertificationClass::NotCertified);
}

TEST(Phase7Integration, ConservationSymbolicBalanceEvidenceMustBeScoped)
{
    const auto scalar = VariableKey::field(0);
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;

    FluxBalanceSummary unscoped;
    unscoped.balance_tolerance = 1.0e-8;
    unscoped.local_residual_norm = 1.0e-12;
    unscoped.symbolic_balance_evidence_present = true;
    summaries.flux_balances.push_back(unscoped);

    FluxBalanceSummary scoped;
    scoped.block = blockFor(scalar, "scoped-balance");
    scoped.balance_group = "mass";
    scoped.balance_tolerance = 1.0e-8;
    scoped.local_residual_norm = 1.0e-12;
    scoped.symbolic_balance_evidence_present = true;
    scoped.symbolic_balance_group = "mass";
    scoped.flux_variable_metadata_present = true;
    scoped.element_residual_evidence_present = true;
    scoped.source_quadrature_consistency_present = true;
    scoped.orientation_consistency_present = true;
    scoped.boundary_flux_accounted_for = true;
    scoped.steady_balance_scope = true;
    summaries.flux_balances.push_back(scoped);
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = analyze(std::move(ctx));
    const auto claims =
        report.claimsOfKind(PropertyKind::ConservationStructure);
    ASSERT_GE(claims.size(), 2u);

    const PropertyClaim* unscoped_claim = nullptr;
    const PropertyClaim* scoped_claim = nullptr;
    for (const auto* claim : claims) {
        if (!claim->tested_block_id.has_value() ||
            claim->tested_block_id->empty()) {
            unscoped_claim = claim;
        } else if (*claim->tested_block_id == "scoped-balance") {
            scoped_claim = claim;
        }
    }
    ASSERT_NE(unscoped_claim, nullptr);
    EXPECT_EQ(unscoped_claim->status, PropertyStatus::Unknown);
    ASSERT_TRUE(unscoped_claim->certification_class.has_value());
    EXPECT_EQ(*unscoped_claim->certification_class,
              CertificationClass::NotCertified);

    ASSERT_NE(scoped_claim, nullptr);
    EXPECT_EQ(scoped_claim->status, PropertyStatus::Preserved);
    ASSERT_TRUE(scoped_claim->certification_class.has_value());
    EXPECT_EQ(*scoped_claim->certification_class,
              CertificationClass::Certified);
}

TEST(Phase7Integration, PreservationSummariesRequireCompleteMetadata)
{
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;

    InvariantDomainSummary invariant;
    invariant.invariant_set_id = "bounds-without-cfl";
    invariant.variables = {VariableKey::field(0)};
    invariant.lower_bound_active = true;
    invariant.limiter_evidence_present = true;
    summaries.invariant_domains.push_back(invariant);

    TransferOperatorSummary transfer;
    transfer.interface_pair_id = "pair";
    transfer.projection_space_id = "projection";
    transfer.residual_tolerance = 1.0e-8;
    transfer.conservation_residual = 0.0;
    transfer.constant_preservation_residual = 0.0;
    transfer.rank_metadata_present = true;
    summaries.transfer_operators.push_back(transfer);

    AdjointConsistencySummary adjoint;
    adjoint.contribution_id = "boundary-goal";
    adjoint.goal_functional_id = "goal";
    adjoint.adjoint_consistency = AdjointConsistencyKind::Yes;
    adjoint.transpose_backend_support = true;
    summaries.adjoint_consistency.push_back(adjoint);

    ctx.setAnalysisSummaries(std::move(summaries));
    auto report = analyze(std::move(ctx));

    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::InvariantDomainPreservation,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Unknown));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::TransferOperatorCompatibility,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Unknown));
    EXPECT_TRUE(hasClaimFromWithStatus(report,
                                       PropertyKind::AdjointConsistency,
                                       "PreservationStructureAnalyzer",
                                       PropertyStatus::Unknown));
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
