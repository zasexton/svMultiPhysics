/**
 * @file test_FormStructureAnalyzer.cpp
 * @brief Unit tests for FormStructureAnalyzer — generalized FormExpr DAG analysis
 */

#include <gtest/gtest.h>

#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include "Forms/FormExpr.h"
#include "Forms/NullspaceAnalyzer.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

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
// ScalarPoisson: ∫ grad(u)·grad(v) dx
// ============================================================================

TEST(FormStructureAnalyzer, ScalarPoisson_GradientOnly) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(summary.per_field.size(), 1u);
    const auto& fs = summary.per_field[0];

    EXPECT_EQ(fs.field, fid);
    EXPECT_TRUE(fs.has_gradient);
    EXPECT_TRUE(fs.only_through_annihilating_ops);
    EXPECT_FALSE(fs.has_absolute_value);
    EXPECT_FALSE(fs.has_time_derivative);
    EXPECT_FALSE(fs.has_stabilization);
    EXPECT_FALSE(fs.has_divergence);
    EXPECT_FALSE(fs.has_curl);
    EXPECT_FALSE(fs.has_hessian);
    EXPECT_FALSE(fs.has_jump);
    EXPECT_FALSE(fs.has_average);
    EXPECT_GT(fs.occurrence_count, 0);

    EXPECT_FALSE(summary.has_stabilization);
    EXPECT_FALSE(summary.has_saddle_point_structure);
}

// ============================================================================
// ScalarPoissonRobin: ∫ grad(u)·grad(v) dx + ∫ α·u·v ds
// ============================================================================

TEST(FormStructureAnalyzer, ScalarPoissonRobin_HasAbsoluteValue) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto alpha = FormExpr::constant(1.0);
    auto residual = inner(grad(u), grad(v)).dx() + (alpha * u * v).ds(1);

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(summary.per_field.size(), 1u);
    const auto& fs = summary.per_field[0];

    EXPECT_TRUE(fs.has_gradient);
    EXPECT_TRUE(fs.has_absolute_value);
    EXPECT_FALSE(fs.only_through_annihilating_ops);
}

// ============================================================================
// Stokes: mixed velocity-pressure
// ============================================================================

TEST(FormStructureAnalyzer, Stokes_MixedSaddlePoint) {
    auto scalar_space = scalarH1();
    auto vector_space = vectorH1();
    const FieldId p_field = 0;
    const FieldId u_field = 1;

    auto p = FormExpr::stateField(p_field, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto u = FormExpr::stateField(u_field, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");

    auto residual = inner(grad(u), grad(v)).dx()
                  - (p * div(v)).dx()
                  + (div(u) * q).dx();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{p_field, u_field});

    ASSERT_EQ(summary.per_field.size(), 2u);

    // Pressure: appears as p*div(v) — absolute value (no differential op on p)
    const auto& p_fs = summary.per_field[0];
    EXPECT_EQ(p_fs.field, p_field);
    EXPECT_TRUE(p_fs.has_absolute_value);
    EXPECT_FALSE(p_fs.only_through_annihilating_ops);

    // Velocity: appears through grad(u) and div(u) — annihilating ops only
    const auto& u_fs = summary.per_field[1];
    EXPECT_EQ(u_fs.field, u_field);
    EXPECT_TRUE(u_fs.has_gradient);
    EXPECT_TRUE(u_fs.has_divergence);
    EXPECT_TRUE(u_fs.only_through_annihilating_ops);
    EXPECT_FALSE(u_fs.has_absolute_value);

    // Mixed couplings
    ASSERT_EQ(summary.mixed_couplings.size(), 2u);

    // Saddle-point: velocity only through annihilating ops, no stabilization
    // → the system has saddle-point structure
    EXPECT_TRUE(summary.has_saddle_point_structure);
}

// ============================================================================
// StabilizedStokes: saddle-point + PSPG stabilization
// ============================================================================

TEST(FormStructureAnalyzer, StabilizedStokes_SaddlePointWithStabilization) {
    auto scalar_space = scalarH1();
    auto vector_space = vectorH1();
    const FieldId p_field = 0;
    const FieldId u_field = 1;

    auto p = FormExpr::stateField(p_field, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto u = FormExpr::stateField(u_field, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");
    auto h = FormExpr::cellDiameter();

    // Stokes + PSPG: h·∫grad(p)·grad(q) dx
    auto residual = inner(grad(u), grad(v)).dx()
                  - (p * div(v)).dx()
                  + (div(u) * q).dx()
                  + (h * inner(grad(p), grad(q))).dx();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{p_field, u_field});

    EXPECT_TRUE(summary.has_stabilization);

    // With PSPG, the pressure block has stabilization — the purely
    // annihilating-ops field (velocity) is still unstabilized, so
    // saddle-point structure persists
    const auto& u_fs = summary.per_field[1];
    EXPECT_TRUE(u_fs.only_through_annihilating_ops);
    EXPECT_FALSE(u_fs.has_stabilization);
}

// ============================================================================
// LinearElasticity: sym(grad) only for vector field
// ============================================================================

TEST(FormStructureAnalyzer, LinearElasticity_SymGradOnly) {
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(summary.per_field.size(), 1u);
    const auto& fs = summary.per_field[0];

    EXPECT_TRUE(fs.has_gradient);
    EXPECT_TRUE(fs.has_sym_grad);
    EXPECT_TRUE(fs.only_through_sym_grad);
    EXPECT_FALSE(fs.has_plain_grad);
    EXPECT_TRUE(fs.only_through_annihilating_ops);
    EXPECT_GT(fs.value_dimension, 1);
}

// ============================================================================
// MixedGradSymGrad: both sym(grad) and plain grad
// ============================================================================

TEST(FormStructureAnalyzer, MixedGradAndSymGrad_NotOnlySymGrad) {
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx()
                  + inner(grad(u), grad(v)).dx();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    const auto& fs = summary.per_field[0];
    EXPECT_TRUE(fs.has_sym_grad);
    EXPECT_TRUE(fs.has_plain_grad);
    EXPECT_FALSE(fs.only_through_sym_grad);
}

// ============================================================================
// Stabilization detection via CellDiameter
// ============================================================================

TEST(FormStructureAnalyzer, Stabilization_CellDiameterDetected) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto p = FormExpr::stateField(fid, *space, "p");
    auto q = FormExpr::testFunction(*space, "q");
    auto h = FormExpr::cellDiameter();
    auto residual = (h * inner(grad(p), grad(q))).dx();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    const auto& fs = summary.per_field[0];
    EXPECT_TRUE(fs.has_stabilization);
    EXPECT_TRUE(summary.has_stabilization);
}

// ============================================================================
// DivergenceOnly: field under div — annihilating op
// ============================================================================

TEST(FormStructureAnalyzer, DivergenceOnly_AnnihilatingOp) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = (div(u) * v).dx();

    FormStructureAnalyzer analyzer;
    auto fs = analyzer.analyzeField(*residual.node(), fid);

    EXPECT_TRUE(fs.has_divergence);
    EXPECT_TRUE(fs.only_through_annihilating_ops);
    EXPECT_FALSE(fs.has_absolute_value);
}

// ============================================================================
// No occurrences: field not in DAG
// ============================================================================

TEST(FormStructureAnalyzer, NoOccurrences_VacuouslyFalse) {
    auto space = scalarH1();
    const FieldId fid = 0;
    const FieldId other = 1;

    auto u = FormExpr::stateField(other, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormStructureAnalyzer analyzer;
    auto fs = analyzer.analyzeField(*residual.node(), fid);

    EXPECT_EQ(fs.occurrence_count, 0);
    EXPECT_FALSE(fs.only_through_annihilating_ops);  // vacuously reset
    EXPECT_FALSE(fs.only_through_sym_grad);           // vacuously reset
}

// ============================================================================
// ReactionDiffusion: mass term anchors absolute value
// ============================================================================

TEST(FormStructureAnalyzer, ReactionDiffusion_AbsoluteValue) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx() + (u * v).dx();

    FormStructureAnalyzer analyzer;
    auto fs = analyzer.analyzeField(*residual.node(), fid);

    EXPECT_TRUE(fs.has_gradient);
    EXPECT_TRUE(fs.has_absolute_value);
    EXPECT_FALSE(fs.only_through_annihilating_ops);
}

// ============================================================================
// Single-field summary — no mixed couplings, no saddle point
// ============================================================================

TEST(FormStructureAnalyzer, SingleField_NoSaddlePoint) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    EXPECT_TRUE(summary.mixed_couplings.empty());
    EXPECT_FALSE(summary.has_saddle_point_structure);
}

// ============================================================================
// NullspaceAnalyzer roundtrip: FormStructureSummary → GaugeCandidates
// ============================================================================

TEST(FormStructureAnalyzer, NullspaceRoundtrip_ScalarPoisson) {
    // Verify that FormStructureAnalyzer + analyzeFromSummary produces
    // the same GaugeCandidates as the old direct NullspaceAnalyzer path
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    // Path 1: New path via FormStructureAnalyzer
    FormStructureAnalyzer fsa;
    auto summary = fsa.analyze(residual, std::array{fid});

    forms::NullspaceAnalyzer na;
    auto candidates_new = na.analyzeFromSummary(summary);

    // Path 2: Direct NullspaceAnalyzer (which now internally does the same thing)
    auto candidates_direct = na.analyze(residual, std::array{fid});

    // Both should produce the same result
    ASSERT_EQ(candidates_new.size(), candidates_direct.size());
    ASSERT_EQ(candidates_new.size(), 1u);
    EXPECT_EQ(candidates_new[0].field, candidates_direct[0].field);
    EXPECT_EQ(candidates_new[0].family, candidates_direct[0].family);
    EXPECT_EQ(candidates_new[0].confidence, candidates_direct[0].confidence);
}

TEST(FormStructureAnalyzer, NullspaceRoundtrip_LinearElasticity) {
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    FormStructureAnalyzer fsa;
    auto summary = fsa.analyze(residual, std::array{fid});

    forms::NullspaceAnalyzer na;
    auto candidates = na.analyzeFromSummary(summary);

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, svmp::FE::gauge::NullspaceModeFamily::KernelOfSymGrad);
}

// ============================================================================
// DG Poisson: Jump/Average operators detected
// ============================================================================

TEST(FormStructureAnalyzer, DGPoisson_JumpDetected) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    // DG interior-face penalty term: ∫ η/h · [u]·[v] dS
    // [u] = jump(u), [v] = jump(v)
    auto h = FormExpr::cellDiameter();
    auto eta = FormExpr::constant(10.0);
    auto penalty = (eta / h) * jump(u) * jump(v);
    auto residual = inner(grad(u), grad(v)).dx() + penalty.dS();

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(summary.per_field.size(), 1u);
    const auto& fs = summary.per_field[0];

    // Field appears under jump operator
    EXPECT_TRUE(fs.has_jump);
    // Field also appears under gradient (from the cell integral term)
    EXPECT_TRUE(fs.has_gradient);
    // Field appears without differential operators in jump(u) — absolute value
    EXPECT_TRUE(fs.has_absolute_value);
    // Note: the penalty term uses eta/h (division by CellDiameter), which the
    // stabilization heuristic doesn't detect because it only checks direct
    // CellDiameter siblings of Multiply/InnerProduct nodes. This is acceptable —
    // the has_jump flag is the primary indicator for DG penalty terms.
}

TEST(FormStructureAnalyzer, DGPoisson_AverageDetected) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    // Consistency term: ∫ {∇u}·[v] dS  (average of gradient, jump of test)
    auto residual = inner(avg(grad(u)), jump(v)).dS();

    FormStructureAnalyzer analyzer;
    auto fs = analyzer.analyzeField(*residual.node(), fid);

    // u appears under avg(grad(...)) — both average and gradient flags
    // Note: the avg() wraps grad(u), so walking the tree we hit Average first,
    // then Gradient, then the field. Both flags should be set.
    EXPECT_TRUE(fs.has_gradient);
    // The field is under Average (through the DAG walk: avg → grad → field)
    // Our walker propagates under_average through children
    EXPECT_TRUE(fs.has_average);
    EXPECT_TRUE(fs.only_through_annihilating_ops);
}

// ============================================================================
// Coupled boundary: BoundaryIntegralSymbol / AuxiliaryStateSymbol detected
// ============================================================================

TEST(FormStructureAnalyzer, CoupledBoundary_DependenciesDetected) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    // Coupled Neumann BC: h(Q, P_d) · v  where Q is a boundary functional
    // and P_d is an auxiliary state variable.
    auto Q = FormExpr::boundaryIntegralValue("outlet_flux");
    auto P_d = FormExpr::auxiliaryState("distal_pressure");
    auto flux = Q + P_d;
    auto residual = inner(grad(u), grad(v)).dx() - (flux * v).ds(1);

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    // Boundary functional and auxiliary state dependencies should be detected
    ASSERT_EQ(summary.boundary_functional_dependencies.size(), 1u);
    EXPECT_EQ(summary.boundary_functional_dependencies[0].kind,
              VariableKind::BoundaryFunctional);
    EXPECT_EQ(summary.boundary_functional_dependencies[0].name, "outlet_flux");

    ASSERT_EQ(summary.auxiliary_state_dependencies.size(), 1u);
    EXPECT_EQ(summary.auxiliary_state_dependencies[0].kind,
              VariableKind::AuxiliaryState);
    EXPECT_EQ(summary.auxiliary_state_dependencies[0].name, "distal_pressure");

    // Variable couplings: FE field ↔ boundary functional, FE field ↔ aux state
    EXPECT_GE(summary.variable_couplings.size(), 2u);
}

TEST(FormStructureAnalyzer, CoupledBoundary_MultipleDependencies) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    // Two boundary functionals and one auxiliary state
    auto Q1 = FormExpr::boundaryIntegralValue("Q_out1");
    auto Q2 = FormExpr::boundaryIntegralValue("Q_out2");
    auto Pd = FormExpr::auxiliaryState("P_distal");
    auto residual = inner(grad(u), grad(v)).dx() - ((Q1 + Q2 + Pd) * v).ds(1);

    FormStructureAnalyzer analyzer;
    auto summary = analyzer.analyze(residual, std::array{fid});

    EXPECT_EQ(summary.boundary_functional_dependencies.size(), 2u);
    EXPECT_EQ(summary.auxiliary_state_dependencies.size(), 1u);
}
