/**
 * @file test_FormContributionLowerer.cpp
 * @brief Unit tests for FormContributionLowerer — FormExpr → ContributionDescriptor
 */

#include <gtest/gtest.h>

#include "Analysis/FormContributionLowerer.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <set>

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

TEST(FormContributionLowerer, RobinBoundaryTermPreservesInteriorNullspaceHint) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual =
        inner(grad(u), grad(v)).dx() + (FormExpr::constant(2.0) * u * v).ds(5);

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    ASSERT_EQ(contributions[0].nullspace_hints.size(), 1u);
    EXPECT_EQ(contributions[0].nullspace_hints[0].field, FieldId{0});
    EXPECT_EQ(contributions[0].nullspace_hints[0].family,
              NullspaceFamily::ScalarConstant);
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

// ============================================================================
// Phase 4: Mixed-form provenance threading tests
// ============================================================================

TEST(FormContributionLowerer, SourceBlockKeyPopulated) {
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
    ASSERT_TRUE(contributions[0].source_block_key.has_value());
    EXPECT_EQ(contributions[0].source_block_key->first, FieldId{0});
    EXPECT_EQ(contributions[0].source_block_key->second, FieldId{0});
}

TEST(FormContributionLowerer, SourceExpressionRetained) {
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
    EXPECT_NE(contributions[0].source_expression, nullptr);
    // Source expression should be the same shared_ptr as the block node
    EXPECT_EQ(contributions[0].source_expression.get(), residual.node());
}

TEST(FormContributionLowerer, FieldNamesInOrigin) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});
    rec.field_names.emplace_back(FieldId{0}, "velocity");

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    // Origin should mention field name
    EXPECT_NE(contributions[0].origin.find("velocity"), std::string::npos);
    // block_context should also mention field name
    EXPECT_NE(contributions[0].block_context.find("velocity"), std::string::npos);
}

TEST(FormContributionLowerer, MixedBlockProvenanceMultiField) {
    auto vel_space = vectorH1();
    auto pres_space = scalarH1();

    // Simulate multi-field: momentum test block contains both velocity and pressure state
    auto u = FormExpr::stateField(1, *vel_space, "u");
    auto p = FormExpr::stateField(2, *pres_space, "p");
    auto v = FormExpr::testFunction(*vel_space, "v");

    // Momentum: grad(u):grad(v) - p div(v)
    auto momentum = (inner(grad(u), grad(v)) + FormExpr::constant(-1.0) * p * div(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {1, 2};
    rec.residual_expr = momentum.nodeShared();
    rec.is_mixed = true;

    // Pseudo-block: (test=velocity, test=velocity) containing ALL trial fields
    rec.block_residual_exprs.push_back({{1, 1}, momentum.nodeShared()});

    rec.field_names.emplace_back(FieldId{1}, "velocity");
    rec.field_names.emplace_back(FieldId{2}, "pressure");

    auto contributions = lowerFormulation(rec);

    // Should produce 2 contributions: VV (diagonal) and VP (off-diagonal)
    ASSERT_EQ(contributions.size(), 2u);

    // Both should have source_block_key pointing to the momentum pseudo-block
    std::set<std::string> contribution_ids;
    for (const auto& c : contributions) {
        ASSERT_TRUE(c.source_block_key.has_value());
        EXPECT_EQ(c.source_block_key->first, FieldId{1});  // test = velocity
        EXPECT_NE(c.source_expression, nullptr);
        EXPECT_FALSE(c.block_context.empty());
        EXPECT_FALSE(c.contribution_id.empty());
        contribution_ids.insert(c.contribution_id);
    }
    EXPECT_EQ(contribution_ids.size(), contributions.size());

    // VV contribution should mention velocity in origin
    bool found_vv = false, found_vp = false;
    for (const auto& c : contributions) {
        bool is_velocity_test = false;
        bool is_velocity_trial = false;
        bool is_pressure_trial = false;
        for (const auto& tv : c.test_variables) {
            if (tv.field_id == FieldId{1}) is_velocity_test = true;
        }
        for (const auto& tv : c.trial_variables) {
            if (tv.field_id == FieldId{1}) is_velocity_trial = true;
            if (tv.field_id == FieldId{2}) is_pressure_trial = true;
        }
        if (is_velocity_test && is_velocity_trial) found_vv = true;
        if (is_velocity_test && is_pressure_trial) found_vp = true;
    }
    EXPECT_TRUE(found_vv) << "Expected VV contribution";
    EXPECT_TRUE(found_vp) << "Expected VP contribution";
}

TEST(FormContributionLowerer, ContextAddContributionNormalizesMissingStableId) {
    ProblemAnalysisContext ctx;
    ContributionDescriptor contribution;
    contribution.operator_tag = "manual-block";
    contribution.origin = "unit-test";
    contribution.role = ContributionRole::DiagonalBlock;
    contribution.test_variables = {VariableKey::field(0)};
    contribution.trial_variables = {VariableKey::field(0)};

    ASSERT_TRUE(contribution.contribution_id.empty());
    ctx.addContribution(std::move(contribution));

    ASSERT_EQ(ctx.contributions().size(), 1u);
    EXPECT_FALSE(ctx.contributions().front().contribution_id.empty());
    EXPECT_NE(ctx.contributions().front().contribution_id.find("manual-block"),
              std::string::npos);
}

TEST(FormContributionLowerer, FallbackPathHasProvenance) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.field_names.emplace_back(FieldId{0}, "temperature");
    // No block_residual_exprs — fallback path

    auto contributions = lowerFormulation(rec);

    ASSERT_EQ(contributions.size(), 1u);
    // Fallback path should still populate provenance
    ASSERT_TRUE(contributions[0].source_block_key.has_value());
    EXPECT_EQ(contributions[0].source_block_key->first, FieldId{0});
    EXPECT_EQ(contributions[0].source_block_key->second, FieldId{0});
    EXPECT_NE(contributions[0].source_expression, nullptr);
    EXPECT_NE(contributions[0].origin.find("temperature"), std::string::npos);
}

// ============================================================================
// Pure-source row emits a contribution
// ============================================================================

TEST(FormContributionLowerer, SingleFieldTrialFunction_NotSource) {
    auto space = scalarH1();
    auto u = FormExpr::trialFunction(*space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});

    auto contributions = lowerFormulation(rec);

    // Should emit a real (non-source) contribution with trial variables
    ASSERT_GE(contributions.size(), 1u);
    EXPECT_FALSE(contributions[0].trial_variables.empty())
        << "TrialFunction residual should have trial variables in contribution";
    // Origin should NOT contain "source"
    EXPECT_EQ(contributions[0].origin.find("source"), std::string::npos)
        << "TrialFunction residual should not be classified as source-only";
}

TEST(FormContributionLowerer, SingleFieldSourceOnlyEmitsSourceContribution) {
    auto space = scalarH1();
    auto v = FormExpr::testFunction(*space, "v");
    auto f = FormExpr::constant(1.0);
    auto residual = (f * v).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});
    rec.field_names.emplace_back(FieldId{0}, "temperature");

    auto contributions = lowerFormulation(rec);

    // Should emit a source-like contribution (empty trial_variables)
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_TRUE(contributions[0].trial_variables.empty())
        << "Source-only contribution should have no trial variables";
    EXPECT_NE(contributions[0].origin.find("source"), std::string::npos)
        << "Should be marked as source in origin";
}

TEST(FormContributionLowerer, PureSourceRowEmitsContribution) {
    auto vel_space = vectorH1();
    auto pres_space = scalarH1();

    // Momentum: grad(u):grad(v) (state-dependent)
    auto u = FormExpr::stateField(1, *vel_space, "u");
    auto v = FormExpr::testFunction(*vel_space, "v");
    auto momentum = inner(grad(u), grad(v)).dx();

    // Continuity: pure source g*q (no state dependency)
    auto q = FormExpr::testFunction(*pres_space, "q");
    auto g = FormExpr::constant(3.0);
    auto continuity = (g * q).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {1, 2};
    rec.residual_expr = (momentum + continuity).nodeShared();
    rec.is_mixed = true;

    // Per-test pseudo-blocks
    rec.block_residual_exprs.push_back({{1, 1}, momentum.nodeShared()});
    rec.block_residual_exprs.push_back({{2, 2}, continuity.nodeShared()});

    rec.field_names.emplace_back(FieldId{1}, "velocity");
    rec.field_names.emplace_back(FieldId{2}, "pressure");

    auto contributions = lowerFormulation(rec);

    // Should have at least 2 contributions: momentum blocks + source
    ASSERT_GE(contributions.size(), 2u);

    // Find the source contribution for the pressure test field
    bool found_source = false;
    for (const auto& c : contributions) {
        bool is_pressure_test = false;
        for (const auto& tv : c.test_variables) {
            if (tv.field_id == FieldId{2}) is_pressure_test = true;
        }
        if (is_pressure_test && c.trial_variables.empty()) {
            found_source = true;
            EXPECT_NE(c.origin.find("source"), std::string::npos)
                << "Pure-source contribution should have 'source' in origin";
            EXPECT_TRUE(c.source_block_key.has_value());
            EXPECT_NE(c.source_expression, nullptr);
        }
    }
    EXPECT_TRUE(found_source) << "Expected a source contribution for the pressure row";
}
