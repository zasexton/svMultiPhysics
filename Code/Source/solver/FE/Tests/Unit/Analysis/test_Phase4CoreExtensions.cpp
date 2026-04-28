/**
 * @file test_Phase4CoreExtensions.cpp
 * @brief Phase 4 tests for first numeric-summary-consuming analysis passes.
 */

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Backends/Utils/BackendOptions.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

namespace {

FieldDescriptor scalarField(FieldId id, std::string name = "scalar") {
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.value_dimension = 1;
    fd.field_type = FieldType::Scalar;
    fd.space_family = SpaceFamily::H1;
    fd.polynomial_order = 1;
    return fd;
}

FieldDescriptor vectorField(FieldId id, std::string name = "vector") {
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.value_dimension = 3;
    fd.field_type = FieldType::Vector;
    fd.space_family = SpaceFamily::H1;
    fd.polynomial_order = 2;
    return fd;
}

DiscreteMatrixSummary certifiedScalarMatrix(VariableKey variable,
                                            std::string tag = "generic_scalar_diffusion") {
    DiscreteMatrixSummary matrix;
    matrix.block.test_variables = {variable};
    matrix.block.trial_variables = {variable};
    matrix.block.operator_tag = std::move(tag);
    matrix.block.domain = DomainKind::Cell;
    matrix.block.role = ContributionRole::DiagonalBlock;
    matrix.rows = 3;
    matrix.cols = 3;
    matrix.square = true;
    matrix.structurally_symmetric = true;
    matrix.numerically_symmetric = true;
    matrix.symmetry_evidence_complete = true;
    matrix.sign_tolerance = 1.0e-14;
    matrix.row_sum_tolerance = 1.0e-14;
    matrix.diagonal_count = 3;
    matrix.offdiag_count = 4;
    matrix.negative_offdiag_count = 4;
    matrix.positive_offdiag_count = 0;
    matrix.nonpositive_diagonal_count = 0;
    matrix.negative_diagonal_count = 0;
    matrix.near_zero_diagonal_count = 0;
    matrix.min_row_sum = 0.0;
    matrix.max_row_sum = 2.0;
    return matrix;
}

ReducedMatrixSummary certifiedReducedScalarMatrix(VariableKey variable,
                                                  std::string tag = "generic_scalar_diffusion") {
    ReducedMatrixSummary reduced;
    reduced.free_free_matrix = certifiedScalarMatrix(variable, std::move(tag));
    reduced.reduction_kind = ConstraintReductionKind::StrongDirichletElimination;
    reduced.free_dof_count = 3;
    reduced.constrained_dof_count = 1;
    reduced.reduction_exact_for_analysis = true;
    return reduced;
}

const PropertyClaim* firstClaim(const ProblemAnalysisReport& report,
                                PropertyKind kind) {
    for (const auto& claim : report.claims) {
        if (claim.kind == kind) {
            return &claim;
        }
    }
    return nullptr;
}

const PropertyClaim* firstClaimFrom(const ProblemAnalysisReport& report,
                                    PropertyKind kind,
                                    const std::string& origin) {
    for (const auto& claim : report.claims) {
        if (claim.kind == kind && claim.claim_origin == origin) {
            return &claim;
        }
    }
    return nullptr;
}

bool hasClaimWithStatus(const ProblemAnalysisReport& report,
                        PropertyKind kind,
                        PropertyStatus status) {
    for (const auto& claim : report.claims) {
        if (claim.kind == kind && claim.status == status) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(Phase4CoreExtensions, ScalarDiffusionReceivesCoercivityDMPAndGeometryReports) {
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarField(0));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "generic_scalar_diffusion", "test"));

    AnalysisSummarySet summaries;
    CoefficientPropertySummary coeff;
    coeff.coefficient = "diffusion_tensor";
    coeff.tensor_rank = TensorRank::Scalar;
    coeff.symmetry = SymmetryClass::Symmetric;
    coeff.positivity = PositivityClass::Positive;
    coeff.min_eigenvalue = 1.0;
    coeff.max_eigenvalue = 2.0;
    summaries.coefficient_properties.push_back(std::move(coeff));
    summaries.reduced_matrices.push_back(
        certifiedReducedScalarMatrix(scalar, "generic_scalar_diffusion"));

    MeshGeometryQualitySummary mesh;
    mesh.min_jacobian = 0.2;
    mesh.max_jacobian = 1.5;
    summaries.mesh_geometry_quality.push_back(std::move(mesh));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);

    ASSERT_TRUE(hasClaimWithStatus(report, PropertyKind::CoefficientPositivity,
                                   PropertyStatus::Preserved));
    ASSERT_TRUE(hasClaimWithStatus(report, PropertyKind::ZMatrixStructure,
                                   PropertyStatus::Exact));
    ASSERT_TRUE(hasClaimWithStatus(report, PropertyKind::MMatrixStructure,
                                   PropertyStatus::Exact));
    ASSERT_TRUE(hasClaimWithStatus(report, PropertyKind::DiscreteMaximumPrinciple,
                                   PropertyStatus::Preserved));
    ASSERT_TRUE(hasClaimWithStatus(report, PropertyKind::MeshGeometryValidity,
                                   PropertyStatus::Preserved));

    const auto* dmp = firstClaim(report, PropertyKind::DiscreteMaximumPrinciple);
    ASSERT_NE(dmp, nullptr);
    ASSERT_TRUE(dmp->matrix_sign_structure_class.has_value());
    EXPECT_EQ(*dmp->matrix_sign_structure_class,
              MatrixSignStructureClass::MMatrixCertified);
    EXPECT_EQ(report.countBySeverity(IssueSeverity::Error), 0u);
}

TEST(Phase4CoreExtensions, DarcyLikeScalarDiffusionUsesGenericMonotonicityEvidence) {
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarField(0, "head_like_unknown"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        scalar, "generic_symmetric_diffusion", "test"));

    AnalysisSummarySet summaries;
    summaries.reduced_matrices.push_back(
        certifiedReducedScalarMatrix(scalar, "generic_symmetric_diffusion"));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);

    const auto* dmp = firstClaim(report, PropertyKind::DiscreteMaximumPrinciple);
    ASSERT_NE(dmp, nullptr);
    EXPECT_EQ(dmp->status, PropertyStatus::Preserved);
    EXPECT_EQ(dmp->description.find("Darcy"), std::string::npos);
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::ZMatrixStructure,
                                   PropertyStatus::Exact));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::MMatrixStructure,
                                   PropertyStatus::Exact));
}

TEST(Phase4CoreExtensions, PositiveOffDiagonalViolatesZMatrixAndDMP) {
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarField(0));

    AnalysisSummarySet summaries;
    auto matrix = certifiedScalarMatrix(scalar, "positive_offdiag_case");
    matrix.positive_offdiag_count = 1;
    matrix.negative_offdiag_count = 3;
    matrix.max_positive_offdiag = 0.25;
    summaries.discrete_matrices.push_back(std::move(matrix));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);

    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::ZMatrixStructure,
                                   PropertyStatus::Violated));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::DiscreteMaximumPrinciple,
                                   PropertyStatus::Violated));
    EXPECT_GT(report.countBySeverity(IssueSeverity::Warning), 0u);
}

TEST(Phase4CoreExtensions, MixedSystemReceivesNumericInfSupSchurAndSolverReports) {
    ProblemAnalysisContext ctx;
    const auto primal = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);

    ctx.addFieldDescriptor(vectorField(1, "primary"));
    ctx.addFieldDescriptor(scalarField(0, "multiplier"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        primal, "mixed_system", "test"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        primal, multiplier, "generic_constraint", "mixed_system", "test"));

    AnalysisSummarySet summaries;
    InfSupEstimateSummary infsup;
    infsup.primal_variable = primal;
    infsup.multiplier_variable = multiplier;
    infsup.block.operator_tag = "generic_constraint";
    infsup.block.test_variables = {primal};
    infsup.block.trial_variables = {multiplier};
    infsup.estimate_value = 0.13;
    infsup.test_rows = 9;
    infsup.test_cols = 3;
    infsup.estimate_scope = "free-free";
    infsup.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    summaries.inf_sup_estimates.push_back(std::move(infsup));

    ReducedMatrixSummary schur;
    schur.free_free_matrix = certifiedScalarMatrix(multiplier, "generic_schur");
    schur.free_free_matrix.block.role = ContributionRole::ConstraintBlock;
    schur.reduction_kind = ConstraintReductionKind::StrongDirichletElimination;
    schur.reduction_exact_for_analysis = true;
    summaries.reduced_matrices.push_back(std::move(schur));
    ctx.setAnalysisSummaries(std::move(summaries));

    backends::SolverOptions options;
    options.method = backends::SolverMethod::BlockSchur;
    options.preconditioner = backends::PreconditionerType::FieldSplit;
    ctx.setSolverOptions(options);

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);

    bool found_numeric_infsup = false;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.inf_sup_estimate.has_value()) {
            found_numeric_infsup = true;
            EXPECT_EQ(*claim.inf_sup_estimate, 0.13);
            EXPECT_EQ(*claim.certification_class, CertificationClass::Certified);
        }
    }
    EXPECT_TRUE(found_numeric_infsup);
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::IndefiniteOperatorResolution,
                                   PropertyStatus::Preserved));
    EXPECT_TRUE(hasClaimWithStatus(report, PropertyKind::SolverCompatibility,
                                   PropertyStatus::Preserved));
}

TEST(Phase4CoreExtensions, SolverCompatibilityRejectsCGForMixedSystem) {
    ProblemAnalysisContext ctx;
    const auto primal = VariableKey::field(1);
    const auto multiplier = VariableKey::field(0);
    ctx.addFieldDescriptor(vectorField(1, "primary"));
    ctx.addFieldDescriptor(scalarField(0, "multiplier"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        primal, "mixed_system", "test"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        primal, multiplier, "generic_constraint", "mixed_system", "test"));

    backends::SolverOptions options;
    options.method = backends::SolverMethod::CG;
    options.preconditioner = backends::PreconditionerType::Diagonal;
    ctx.setSolverOptions(options);

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);

    const auto* solver_claim =
        firstClaimFrom(report, PropertyKind::SolverCompatibility,
                       "SolverCompatibilityAnalyzer");
    ASSERT_NE(solver_claim, nullptr);
    EXPECT_EQ(solver_claim->status, PropertyStatus::Violated);
    EXPECT_EQ(report.countBySeverity(IssueSeverity::Error), 1u);
}

TEST(Phase4CoreExtensions, MeshGeometryWorstSamplesAreTopologyScoped) {
    ProblemAnalysisContext ctx;
    TopologyAnalysisContext topo;
    ConnectedComponent region0;
    region0.region_id = 0;
    region0.cell_indices = {0, 1};
    ConnectedComponent region1;
    region1.region_id = 1;
    region1.cell_indices = {2, 3};
    topo.components = {region0, region1};
    ctx.setTopologyContext(std::move(topo));

    AnalysisSummarySet summaries;
    MeshGeometryQualitySummary mesh;
    mesh.poor_quality_element_count = 1;
    mesh.worst_elements = {2};
    summaries.mesh_geometry_quality.push_back(std::move(mesh));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);

    bool found_region_claim = false;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::MeshGeometryValidity &&
            claim.region == 1) {
            found_region_claim = true;
            EXPECT_EQ(claim.status, PropertyStatus::Likely);
        }
    }
    EXPECT_TRUE(found_region_claim);
}

TEST(Phase4CoreExtensions, ConstraintAndInitialCompatibilitySummariesAreConsumed) {
    ProblemAnalysisContext ctx;
    const auto scalar = VariableKey::field(0);
    ctx.addFieldDescriptor(scalarField(0));

    AnalysisSummarySet summaries;
    ReducedMatrixSummary reduced;
    reduced.free_free_matrix = certifiedScalarMatrix(scalar, "affine_reduced");
    reduced.reduction_kind = ConstraintReductionKind::AffineTransform;
    reduced.affine_terms_accounted_for = false;
    reduced.reduction_exact_for_analysis = false;
    summaries.reduced_matrices.push_back(std::move(reduced));

    InitialCompatibilitySummary initial;
    initial.initial_constraint_residual = 2.0e-3;
    initial.initial_boundary_residual = 0.0;
    initial.residual_tolerance = 1.0e-6;
    summaries.initial_compatibility.push_back(std::move(initial));
    ctx.setAnalysisSummaries(std::move(summaries));

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);

    EXPECT_NE(firstClaimFrom(report, PropertyKind::InitialDataCompatibility,
                             "ConstraintRankAnalyzer"),
              nullptr);
    const auto* compat =
        firstClaimFrom(report, PropertyKind::InitialDataCompatibility,
                       "CompatibilityAnalyzer");
    ASSERT_NE(compat, nullptr);
    EXPECT_EQ(compat->status, PropertyStatus::Violated);
    ASSERT_TRUE(compat->initial_data_compatible.has_value());
    EXPECT_FALSE(*compat->initial_data_compatible);
}
