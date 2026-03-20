/**
 * @file test_FormContributionLowerer.cpp
 * @brief Unit tests for FormContributionLowerer — FormExpr → ContributionDescriptor
 */

#include <gtest/gtest.h>

#include "Analysis/FormContributionLowerer.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

auto scalarH1() { return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1); }
auto vectorH1(int dim = 3) {
    return std::make_shared<spaces::ProductSpace>(
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1), dim);
}

} // namespace

// ============================================================================
// Scalar Poisson → DiagonalBlock with SymmetricLike + HasSecondOrder + PSD
// ============================================================================

TEST(FormContributionLowerer, ScalarPoisson) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    const auto& d = contributions[0];

    EXPECT_EQ(d.role, ContributionRole::DiagonalBlock);
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::HasSecondOrder));
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::SymmetricLike));
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::PositiveSemiDefiniteLike));
    EXPECT_FALSE(hasFlag(d.traits, OperatorTraitFlags::HasFirstOrder));
    EXPECT_FALSE(hasFlag(d.traits, OperatorTraitFlags::HasMass));
    EXPECT_EQ(d.confidence, AnalysisConfidence::High);

    // Nullspace hint
    ASSERT_EQ(d.nullspace_hints.size(), 1u);
    EXPECT_EQ(d.nullspace_hints[0].family, NullspaceFamily::ScalarConstant);
    EXPECT_EQ(d.nullspace_hints[0].field, FieldId{0});
}

// ============================================================================
// Linear Elasticity → DiagonalBlock with SymmetricLike + KernelOfSymGrad hint
// ============================================================================

TEST(FormContributionLowerer, LinearElasticity) {
    auto space = vectorH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions[0].role, ContributionRole::DiagonalBlock);
    EXPECT_TRUE(hasFlag(contributions[0].traits, OperatorTraitFlags::SymmetricLike));

    ASSERT_EQ(contributions[0].nullspace_hints.size(), 1u);
    EXPECT_EQ(contributions[0].nullspace_hints[0].family, NullspaceFamily::KernelOfSymGrad);
}

// ============================================================================
// Stokes blocks
// ============================================================================

TEST(FormContributionLowerer, Stokes_VVBlock) {
    auto vector_space = vectorH1();
    auto u = FormExpr::stateField(1, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");
    auto vv_residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0, 1};
    rec.residual_expr = vv_residual.nodeShared();
    rec.block_residual_exprs.push_back({{1, 1}, vv_residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions[0].role, ContributionRole::DiagonalBlock);
    EXPECT_TRUE(hasFlag(contributions[0].traits, OperatorTraitFlags::HasSecondOrder));
}

// ============================================================================
// Stabilized term → StabilizationBlock
// ============================================================================

TEST(FormContributionLowerer, StabilizedPressure) {
    auto space = scalarH1();
    auto p = FormExpr::stateField(0, *space, "p");
    auto q = FormExpr::testFunction(*space, "q");
    auto h = FormExpr::cellDiameter();
    auto residual = (h * inner(grad(p), grad(q))).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.has_stabilization_terms = true;
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions[0].role, ContributionRole::StabilizationBlock);
}

// ============================================================================
// Boundary marker carried through
// ============================================================================

TEST(FormContributionLowerer, BoundaryMarkerCarried) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto alpha = FormExpr::constant(1.0);
    auto residual = (alpha * u * v).ds(5);

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions[0].boundary_marker, 5);
    EXPECT_EQ(contributions[0].domain, DomainKind::Boundary);
}

// ============================================================================
// No residual expression → empty contributions
// ============================================================================

TEST(FormContributionLowerer, NoResidualExpr) {
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    // residual_expr is null

    auto contributions = lowerFormulation(rec);
    EXPECT_TRUE(contributions.empty());
}

// ============================================================================
// Fallback path — no block_residual_exprs, just active_fields
// ============================================================================

TEST(FormContributionLowerer, FallbackPath) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    // No block_residual_exprs

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions[0].role, ContributionRole::DiagonalBlock);
}

// ============================================================================
// Self-adjoint pattern detection
// ============================================================================

TEST(FormContributionLowerer, SelfAdjointPattern) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    // Self-adjoint pattern → SymmetricLike
    EXPECT_TRUE(hasFlag(contributions[0].traits, OperatorTraitFlags::SymmetricLike));
}
