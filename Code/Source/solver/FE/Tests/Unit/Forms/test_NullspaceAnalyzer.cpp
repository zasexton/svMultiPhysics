/**
 * @file test_NullspaceAnalyzer.cpp
 * @brief Unit tests for NullspaceAnalyzer — FormExpr DAG nullspace inference
 */

#include <gtest/gtest.h>

#include "Forms/NullspaceAnalyzer.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::forms;
using namespace svmp::FE::gauge;

namespace {

// Helper: create a scalar H1 space (Tet4, P1)
std::shared_ptr<spaces::FunctionSpace> scalarH1()
{
    return std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
}

// Helper: create a vector H1 space (Tet4, P1, 3 components)
std::shared_ptr<spaces::FunctionSpace> vectorH1(int dim = 3)
{
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
    return std::make_shared<spaces::ProductSpace>(base, dim);
}

} // namespace

// ============================================================================
// Scalar Poisson: ∫ grad(u)·grad(v) dx
// ============================================================================

TEST(NullspaceAnalyzer, ScalarPoisson_GradOnly)
{
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].field, fid);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(candidates[0].confidence, Confidence::High);
}

// ============================================================================
// Scalar Poisson with Robin: ∫ grad(u)·grad(v) dx + ∫ alpha*u*v ds
// ============================================================================

TEST(NullspaceAnalyzer, ScalarPoissonRobin_Anchored)
{
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto alpha = FormExpr::constant(1.0);
    auto residual = inner(grad(u), grad(v)).dx() + (alpha * u * v).ds(1);

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    // Robin term provides absolute value of u → no nullspace candidate
    EXPECT_TRUE(candidates.empty());
}

// ============================================================================
// Scalar reaction-diffusion: ∫ grad(u)·grad(v) dx + ∫ u*v dx
// ============================================================================

TEST(NullspaceAnalyzer, ScalarReactionDiffusion_Anchored)
{
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx() + (u * v).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    // Mass term u*v anchors the constant mode
    EXPECT_TRUE(candidates.empty());
}

// ============================================================================
// Incompressible Stokes: pressure appears only through grad(p)
// ============================================================================

TEST(NullspaceAnalyzer, StokesPressure_AppearsInCouplingTerm)
{
    auto scalar_space = scalarH1();
    auto vector_space = vectorH1();
    const FieldId p_field = 0;
    const FieldId u_field = 1;

    auto p = FormExpr::stateField(p_field, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto u = FormExpr::stateField(u_field, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");

    // Simplified Stokes: ∫ grad(u):grad(v) dx - ∫ p*div(v) dx + ∫ div(u)*q dx
    auto residual = inner(grad(u), grad(v)).dx()
                  - (p * div(v)).dx()
                  + (div(u) * q).dx();

    NullspaceAnalyzer analyzer;

    // Pressure appears as p*div(v) — the pressure field itself is NOT under
    // any differential operator (div is applied to test function v, not p).
    // The analyzer correctly classifies this as an absolute-value term.
    // The pressure nullspace in Stokes is a saddle-point property of the
    // full system, not something detectable from symbolic form analysis alone.
    auto p_candidates = analyzer.analyze(residual, std::array{p_field});
    EXPECT_TRUE(p_candidates.empty());

    // Verify the classification details: p has absolute terms
    auto p_class = analyzer.classifyField(*residual.node(), p_field);
    EXPECT_TRUE(p_class.has_absolute_terms);
    EXPECT_FALSE(p_class.only_through_annihilating_ops);

    // Velocity field — appears through grad(u) and div(u), both annihilating
    auto u_candidates = analyzer.analyze(residual, std::array{u_field});
    ASSERT_EQ(u_candidates.size(), 1u);
    EXPECT_EQ(u_candidates[0].family, NullspaceModeFamily::ComponentwiseConstant);
}

// ============================================================================
// Stabilization detection: h-scaled penalty terms
// ============================================================================

TEST(NullspaceAnalyzer, StabilizedPressure_MediumConfidence)
{
    auto space = scalarH1();
    const FieldId fid = 0;

    auto p = FormExpr::stateField(fid, *space, "p");
    auto q = FormExpr::testFunction(*space, "q");
    auto h = FormExpr::cellDiameter();

    // PSPG-like term: h * ∫ grad(p)·grad(q) dx
    auto residual = (h * inner(grad(p), grad(q))).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(candidates[0].confidence, Confidence::Medium);
}

// ============================================================================
// classifyField — low-level tests
// ============================================================================

TEST(NullspaceAnalyzer, ClassifyField_NoOccurrences)
{
    auto space = scalarH1();
    const FieldId fid = 0;
    const FieldId other = 1;

    auto u = FormExpr::stateField(other, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    NullspaceAnalyzer analyzer;
    auto c = analyzer.classifyField(*residual.node(), fid);

    EXPECT_EQ(c.occurrence_count, 0);
    EXPECT_FALSE(c.only_through_annihilating_ops);  // vacuously reset
}

TEST(NullspaceAnalyzer, ClassifyField_AbsoluteTerms)
{
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    NullspaceAnalyzer analyzer;
    auto c = analyzer.classifyField(*residual.node(), fid);

    EXPECT_GT(c.occurrence_count, 0);
    EXPECT_FALSE(c.only_through_annihilating_ops);
    EXPECT_TRUE(c.has_absolute_terms);
}

TEST(NullspaceAnalyzer, ClassifyField_GradientOnly)
{
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    NullspaceAnalyzer analyzer;
    auto c = analyzer.classifyField(*residual.node(), fid);

    EXPECT_GT(c.occurrence_count, 0);
    EXPECT_TRUE(c.only_through_annihilating_ops);
    EXPECT_FALSE(c.has_absolute_terms);
}

// ============================================================================
// Phase 2: Vector field — componentwise constant
// ============================================================================

TEST(NullspaceAnalyzer, VectorField_GradOnly_ComponentwiseConstant)
{
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::ComponentwiseConstant);
    EXPECT_EQ(candidates[0].confidence, Confidence::High);
}

// ============================================================================
// Phase 2: Vector field — rigid-body modes via sym(grad)
// ============================================================================

TEST(NullspaceAnalyzer, VectorField_SymGradOnly_KernelOfSymGrad)
{
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    // Linear elasticity: ∫ sym(grad(u)) : sym(grad(v)) dx
    auto eps_u = sym(grad(u));
    auto eps_v = sym(grad(v));
    auto residual = inner(eps_u, eps_v).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::KernelOfSymGrad);
    EXPECT_EQ(candidates[0].confidence, Confidence::High);
}

TEST(NullspaceAnalyzer, VectorField_MixedGradAndSymGrad_ComponentwiseConstant)
{
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    // Mix of sym(grad) and plain grad — not pure sym(grad),
    // so only componentwise constant, not rigid body
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx()
                  + inner(grad(u), grad(v)).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::ComponentwiseConstant);
}

TEST(NullspaceAnalyzer, VectorField_SymGradPlusMass_NoNullspace)
{
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    // sym(grad) + mass term → mode is anchored
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx()
                  + inner(u, v).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    // Mass term anchors the mode
    EXPECT_TRUE(candidates.empty());
}

// ============================================================================
// Phase 2: 2D vector field rigid-body modes
// ============================================================================

TEST(NullspaceAnalyzer, VectorField_2D_SymGrad_KernelOfSymGrad)
{
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto space = std::make_shared<spaces::ProductSpace>(base, 2);
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    NullspaceAnalyzer analyzer;
    auto candidates = analyzer.analyze(residual, std::array{fid});

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::KernelOfSymGrad);
}

// ============================================================================
// Multi-field: analyze specific fields in a coupled system
// ============================================================================

TEST(NullspaceAnalyzer, MultiField_OnlyAnalyzesRequestedFields)
{
    auto scalar_space = scalarH1();
    auto vector_space = vectorH1();
    const FieldId p_field = 0;
    const FieldId u_field = 1;

    auto p = FormExpr::stateField(p_field, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto u = FormExpr::stateField(u_field, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");

    // Use a form where both fields appear through annihilating ops:
    // ∫ grad(u):grad(v) dx + ∫ grad(p)·grad(q) dx
    auto residual = inner(grad(u), grad(v)).dx() + inner(grad(p), grad(q)).dx();

    NullspaceAnalyzer analyzer;

    // Only analyze p_field
    auto p_only = analyzer.analyze(residual, std::array{p_field});
    ASSERT_EQ(p_only.size(), 1u);
    EXPECT_EQ(p_only[0].field, p_field);
    EXPECT_EQ(p_only[0].family, NullspaceModeFamily::ScalarConstant);

    // Only analyze u_field
    auto u_only = analyzer.analyze(residual, std::array{u_field});
    ASSERT_EQ(u_only.size(), 1u);
    EXPECT_EQ(u_only[0].field, u_field);
    EXPECT_EQ(u_only[0].family, NullspaceModeFamily::ComponentwiseConstant);
}

// ============================================================================
// Divergence operator
// ============================================================================

TEST(NullspaceAnalyzer, DivergenceOnly_ScalarConstant)
{
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    // div(u) term — divergence annihilates constants
    auto residual = (div(u) * v).dx();

    NullspaceAnalyzer analyzer;
    auto c = analyzer.classifyField(*residual.node(), fid);

    EXPECT_TRUE(c.only_through_annihilating_ops);
    EXPECT_FALSE(c.has_absolute_terms);
}
