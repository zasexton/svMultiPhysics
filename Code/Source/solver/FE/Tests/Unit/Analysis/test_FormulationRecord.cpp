/**
 * @file test_FormulationRecord.cpp
 * @brief Unit tests for FormulationRecord population and FormExprScanner
 *
 * Tests verify:
 *   - FormExprScanner DAG scanning utility
 *   - FormulationRecord structural correctness
 *   - Residual expr handle lifetime
 *   - Stabilization / time derivative / DG flags
 */

#include <gtest/gtest.h>

#include "Analysis/FormulationRecord.h"
#include "Analysis/FormExprScanner.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include "Forms/FormExpr.h"
#include "Forms/AffineAnalysis.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <algorithm>
#include <memory>

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

std::shared_ptr<spaces::FunctionSpace> scalarH1() {
    return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
}

std::shared_ptr<spaces::FunctionSpace> vectorH1(int dim = 3) {
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    return std::make_shared<spaces::ProductSpace>(base, dim);
}

} // namespace

// ============================================================================
// FormExprScanner — isolated DAG scanning tests
// ============================================================================

TEST(FormExprScanner, GradGrad_NoSpecialFlags) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto expr = inner(grad(u), grad(v));

    auto result = scanFormExpr(*expr.node());

    EXPECT_FALSE(result.has_time_derivative);
    EXPECT_FALSE(result.has_cell_diameter);
    EXPECT_FALSE(result.has_jump);
    EXPECT_FALSE(result.has_average);
    EXPECT_FALSE(result.has_stabilization());
    EXPECT_FALSE(result.has_interior_face_terms());
    EXPECT_TRUE(result.boundary_functional_names.empty());
    EXPECT_TRUE(result.auxiliary_state_names.empty());
}

TEST(FormExprScanner, CellDiameter_Alone) {
    // CellDiameter node appears in expression
    auto space = scalarH1();
    auto v = FormExpr::testFunction(*space, "v");
    auto h = FormExpr::cellDiameter();
    auto expr = h * v;

    auto result = scanFormExpr(*expr.node());

    EXPECT_TRUE(result.has_cell_diameter);
    EXPECT_TRUE(result.has_stabilization());
    EXPECT_FALSE(result.has_time_derivative);
}

TEST(FormExprScanner, CellDiameter_Detected) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto h = FormExpr::cellDiameter();
    auto expr = inner(h * grad(u), grad(v));

    auto result = scanFormExpr(*expr.node());

    EXPECT_TRUE(result.has_cell_diameter);
    EXPECT_TRUE(result.has_stabilization());
}

TEST(FormExprScanner, ActiveDomains_DefaultIsCell) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto expr = inner(u, v);

    auto result = scanFormExpr(*expr.node());
    auto domains = result.activeDomains();

    ASSERT_FALSE(domains.empty());
    EXPECT_EQ(domains[0], DomainKind::Cell);
}

TEST(FormExprScanner, ScanResultConvenience) {
    // Verify convenience methods on an empty result
    FormExprScanResult empty;
    EXPECT_FALSE(empty.has_stabilization());
    EXPECT_FALSE(empty.has_interior_face_terms());

    auto domains = empty.activeDomains();
    ASSERT_EQ(domains.size(), 1u);
    EXPECT_EQ(domains[0], DomainKind::Cell);  // Default when no integral nodes found
}

// ============================================================================
// FormulationRecord — structural tests
// ============================================================================

TEST(FormulationRecord, DefaultConstruction) {
    FormulationRecord rec;
    EXPECT_TRUE(rec.operator_tag.empty());
    EXPECT_TRUE(rec.active_fields.empty());
    EXPECT_TRUE(rec.active_variables.empty());
    EXPECT_EQ(rec.residual_expr, nullptr);
    EXPECT_FALSE(rec.affine_split_succeeded);
    EXPECT_FALSE(rec.is_mixed);
    EXPECT_FALSE(rec.has_interior_face_terms);
    EXPECT_FALSE(rec.has_time_derivative);
    EXPECT_FALSE(rec.has_stabilization_terms);
    EXPECT_TRUE(rec.active_domains.empty());
    EXPECT_TRUE(rec.block_couplings.empty());
    EXPECT_TRUE(rec.variable_couplings.empty());
    EXPECT_TRUE(rec.boundary_functional_dependencies.empty());
    EXPECT_TRUE(rec.auxiliary_state_dependencies.empty());
}

TEST(FormulationRecord, PopulateManually_Poisson) {
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.is_mixed = false;
    rec.has_time_derivative = false;
    rec.has_stabilization_terms = false;
    rec.has_interior_face_terms = false;
    rec.active_domains = {DomainKind::Cell};
    rec.block_couplings = {{0, 0}};
    rec.active_variables.push_back(VariableKey::field(0));

    EXPECT_EQ(rec.operator_tag, "equations");
    ASSERT_EQ(rec.active_fields.size(), 1u);
    EXPECT_EQ(rec.active_fields[0], 0);
    EXPECT_FALSE(rec.is_mixed);
    ASSERT_EQ(rec.block_couplings.size(), 1u);
    EXPECT_EQ(rec.block_couplings[0], std::make_pair(FieldId{0}, FieldId{0}));
}

TEST(FormulationRecord, PopulateManually_Stokes) {
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0, 1};
    rec.is_mixed = true;
    rec.block_couplings = {{0,0}, {0,1}, {1,0}, {1,1}};
    rec.active_variables.push_back(VariableKey::field(0));
    rec.active_variables.push_back(VariableKey::field(1));

    EXPECT_TRUE(rec.is_mixed);
    ASSERT_EQ(rec.block_couplings.size(), 4u);
    ASSERT_EQ(rec.active_variables.size(), 2u);
}

TEST(FormulationRecord, PopulateManually_CoupledBoundary) {
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.active_variables.push_back(VariableKey::field(0));

    auto bf = VariableKey::named(VariableKind::BoundaryFunctional, "Q_out");
    auto aux = VariableKey::named(VariableKind::AuxiliaryState, "P_d");
    rec.active_variables.push_back(bf);
    rec.active_variables.push_back(aux);
    rec.boundary_functional_dependencies.push_back(bf);
    rec.auxiliary_state_dependencies.push_back(aux);
    rec.variable_couplings.emplace_back(VariableKey::field(0), bf);
    rec.variable_couplings.emplace_back(VariableKey::field(0), aux);

    ASSERT_EQ(rec.active_variables.size(), 3u);
    ASSERT_EQ(rec.boundary_functional_dependencies.size(), 1u);
    ASSERT_EQ(rec.auxiliary_state_dependencies.size(), 1u);
    ASSERT_EQ(rec.variable_couplings.size(), 2u);
    EXPECT_EQ(rec.boundary_functional_dependencies[0].name, "Q_out");
    EXPECT_EQ(rec.auxiliary_state_dependencies[0].name, "P_d");
}

TEST(FormulationRecord, ResidualExprHandle_KeepsNodeAlive) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v));

    FormulationRecord rec;
    rec.residual_expr = residual.nodeShared();

    ASSERT_NE(rec.residual_expr, nullptr);
    EXPECT_EQ(rec.residual_expr->type(), FormExprType::InnerProduct);
}

// ============================================================================
// affine_split_succeeded — via AffineAnalysis
// ============================================================================

TEST(FormulationRecord, AffineSplit_LinearPoisson) {
    // Linear Poisson: ∫ grad(u)·grad(v) dx is affine in TrialFunction
    auto space = scalarH1();
    auto u = FormExpr::trialFunction(*space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    forms::AffineResidualOptions opts;
    auto split = forms::trySplitAffineResidual(residual, opts);
    EXPECT_TRUE(split.has_value());
}

TEST(FormulationRecord, AffineSplit_ReactionDiffusion) {
    // Reaction-diffusion: ∫ grad(u)·grad(v) + u*v is also affine
    auto space = scalarH1();
    auto u = FormExpr::trialFunction(*space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx() + (u * v).dx();

    auto split = forms::trySplitAffineResidual(residual);
    EXPECT_TRUE(split.has_value());
}

// ============================================================================
// block_residual_exprs
// ============================================================================

TEST(FormulationRecord, BlockResidualExprs_Default) {
    FormulationRecord rec;
    EXPECT_TRUE(rec.block_residual_exprs.empty());
}

TEST(FormulationRecord, BlockResidualExprs_SingleField) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v));

    FormulationRecord rec;
    rec.active_fields = {0};
    rec.block_residual_exprs.push_back(
        {{0, 0}, residual.nodeShared()});

    ASSERT_EQ(rec.block_residual_exprs.size(), 1u);
    EXPECT_EQ(rec.block_residual_exprs[0].first.first, FieldId{0});
    EXPECT_EQ(rec.block_residual_exprs[0].first.second, FieldId{0});
    ASSERT_NE(rec.block_residual_exprs[0].second, nullptr);
}
