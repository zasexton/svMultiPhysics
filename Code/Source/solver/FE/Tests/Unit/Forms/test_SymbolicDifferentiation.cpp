/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SymbolicDifferentiation.cpp
 * @brief Unit tests for symbolic differentiation + tangent decomposition (SymbolicNonlinearFormKernel)
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/JIT/InlinableConstitutiveModel.h"
#include "Forms/Dual.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <ctime>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

void expectDenseNear(const assembly::DenseMatrixView& A,
                     const assembly::DenseMatrixView& B,
                     Real tol)
{
    ASSERT_EQ(A.numRows(), B.numRows());
    ASSERT_EQ(A.numCols(), B.numCols());
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            EXPECT_NEAR(A.getMatrixEntry(i, j), B.getMatrixEntry(i, j), tol);
        }
    }
}

void expectDenseNear(const assembly::DenseVectorView& a,
                     const assembly::DenseVectorView& b,
                     Real tol)
{
    ASSERT_EQ(a.numRows(), b.numRows());
    for (GlobalIndex i = 0; i < a.numRows(); ++i) {
        EXPECT_NEAR(a.getVectorEntry(i), b.getVectorEntry(i), tol);
    }
}

struct CompareOptions {
    Real vec_tol{1e-12};
    Real mat_tol{1e-12};

    const assembly::TimeIntegrationContext* time_ctx{nullptr};
    Real time_step{0.0};
    std::optional<std::vector<Real>> prev_solution{};
};

void compareADAndSymbolic(const assembly::IMeshAccess& mesh,
                          const dofs::DofMap& dof_map,
                          const spaces::FunctionSpace& test_space,
                          const spaces::FunctionSpace& trial_space,
                          const FormExpr& residual,
                          const std::vector<Real>& U,
                          const CompareOptions& options = {})
{
    FormCompiler compiler;
    auto ir_ad = compiler.compileResidual(residual);
    auto ir_sym = compiler.compileResidual(residual);

    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

    if (options.prev_solution.has_value()) {
        assembler.setPreviousSolution(*options.prev_solution);
    }
    assembler.setTimeStep(options.time_step);
    assembler.setTimeIntegrationContext(options.time_ctx);

    const GlobalIndex n = dof_map.getNumDofs();

    assembly::DenseMatrixView J_ad(n);
    assembly::DenseVectorView R_ad(n);
    J_ad.zero();
    R_ad.zero();
    (void)assembler.assembleBoth(mesh, test_space, trial_space, ad_kernel, J_ad, R_ad);

    assembly::DenseMatrixView J_sym(n);
    assembly::DenseVectorView R_sym(n);
    J_sym.zero();
    R_sym.zero();
    (void)assembler.assembleBoth(mesh, test_space, trial_space, sym_kernel, J_sym, R_sym);

    expectDenseNear(R_sym, R_ad, options.vec_tol);
    expectDenseNear(J_sym, J_ad, options.mat_tol);
}

[[nodiscard]] dofs::DofMap makeSingleCellDofMap(GlobalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, static_cast<LocalIndex>(n_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

} // namespace

TEST(SymbolicDifferentiationMultiFieldTest, DifferentiateWrtFieldIdMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr FieldId u_field = 11;
    constexpr FieldId p_field = 12;

    const auto u = FormExpr::stateField(u_field, space, "u");
    const auto p_trial = FormExpr::trialFunction(space, "p");
    const auto v = FormExpr::testFunction(space, "v");

    // AD reference: differentiate residual w.r.t the active TrialFunction p.
    const auto residual_ad = (u * p_trial * v).dx();

    FormCompiler compiler;
    auto ir_ad = compiler.compileResidual(residual_ad);
    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);

    // Symbolic: differentiate w.r.t a FieldId-backed state variable p_field.
    const auto p_state = FormExpr::stateField(p_field, space, "p");
    const auto residual_sym = (u * p_state * v).dx();
    const auto tangent_by_field = differentiateResidual(residual_sym, p_field);
    const auto tangent_by_expr = differentiateResidual(residual_sym, p_state);
    EXPECT_EQ(tangent_by_field.toString(), tangent_by_expr.toString());

    auto ir_sym = compiler.compileBilinear(tangent_by_field);
    FormKernel sym_kernel(std::move(ir_sym));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n = dof_map.getNumDofs();
    const std::array<assembly::FieldSolutionAccess, 1> field_access = {
        assembly::FieldSolutionAccess{u_field, &space, &dof_map, n},
    };
    assembler.setFieldSolutionAccess(field_access);

    // Global state vector packs [p, u] with offsets [0..n) and [n..2n).
    std::vector<Real> state(static_cast<std::size_t>(2 * n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] = 0.1 * static_cast<Real>(i + 1);             // p
        state[static_cast<std::size_t>(n + i)] = -0.05 * static_cast<Real>(i + 1);        // u
    }
    assembler.setCurrentSolution(state);

    assembly::DenseMatrixView J_ad(n);
    assembly::DenseMatrixView J_sym(n);
    J_ad.zero();
    J_sym.zero();

    (void)assembler.assembleMatrix(mesh, space, space, ad_kernel, J_ad);
    (void)assembler.assembleMatrix(mesh, space, space, sym_kernel, J_sym);

    expectDenseNear(J_sym, J_ad, 1e-12);
}

TEST(SymbolicDifferentiationMultiFieldTest, DifferentiateWrtFieldIdRewritesTrialPrimalToProvidedField)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr FieldId u_field = 11;
    constexpr FieldId p_field = 12;

    // AD reference: u is an external field, p is the active TrialFunction.
    FormCompiler compiler;
    const auto u_state = FormExpr::stateField(u_field, space, "u");
    const auto p_trial = FormExpr::trialFunction(space, "p");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual_ad = (u_state * p_trial * v).dx();

    auto ir_ad = compiler.compileResidual(residual_ad);
    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward, NonlinearKernelOutput::MatrixOnly);

    // Symbolic: residual written with a TrialFunction u, but we request that its primal be rewritten to StateField(u_field)
    // so the resulting tangent can be assembled in the p block.
    const auto u_trial = FormExpr::trialFunction(space, "u");
    const auto p_state = FormExpr::stateField(p_field, space, "p");
    const auto residual_sym = (u_trial * p_state * v).dx();
    const auto tangent = differentiateResidual(residual_sym, p_field, /*trial_state_field=*/u_field);

    auto ir_sym = compiler.compileBilinear(tangent);
    FormKernel sym_kernel(std::move(ir_sym));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n = dof_map.getNumDofs();
    const std::array<assembly::FieldSolutionAccess, 1> field_access = {
        assembly::FieldSolutionAccess{u_field, &space, &dof_map, n},
    };
    assembler.setFieldSolutionAccess(field_access);

    // Global state vector packs [p, u] with offsets [0..n) and [n..2n).
    std::vector<Real> state(static_cast<std::size_t>(2 * n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] = 0.1 * static_cast<Real>(i + 1);             // p
        state[static_cast<std::size_t>(n + i)] = -0.05 * static_cast<Real>(i + 1);        // u
    }
    assembler.setCurrentSolution(state);

    assembly::DenseMatrixView J_ad(n);
    assembly::DenseMatrixView J_sym(n);
    J_ad.zero();
    J_sym.zero();

    (void)assembler.assembleMatrix(mesh, space, space, ad_kernel, J_ad);
    (void)assembler.assembleMatrix(mesh, space, space, sym_kernel, J_sym);

    expectDenseNear(J_sym, J_ad, 1e-12);
}

TEST(SymbolicNonlinearFormKernelTest, PolynomialJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (u * u * v).dx();

    auto ir_ad = compiler.compileResidual(residual);
    auto ir_sym = compiler.compileResidual(residual);

    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J_ad(4);
    assembly::DenseVectorView R_ad(4);
    J_ad.zero();
    R_ad.zero();
    (void)assembler.assembleBoth(mesh, space, space, ad_kernel, J_ad, R_ad);

    assembly::DenseMatrixView J_sym(4);
    assembly::DenseVectorView R_sym(4);
    J_sym.zero();
    R_sym.zero();
    (void)assembler.assembleBoth(mesh, space, space, sym_kernel, J_sym, R_sym);

    expectDenseNear(R_sym, R_ad, 1e-12);
    expectDenseNear(J_sym, J_ad, 1e-12);
}

TEST(SymbolicNonlinearFormKernelTest, DivideQuotientRuleMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // R = (u / (1 + u^2)) * v
    const auto denom = FormExpr::constant(1.0) + u * u;
    const auto residual = ((u / denom) * v).dx();

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, NegateAndSubtractRulesMatchAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // R = (-(u) - u^2) * v
    const auto residual = ((-u - (u * u)) * v).dx();

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, PoissonJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, HessianJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // R = inner(H(u), H(v))
    const auto residual = inner(u.hessian(), v.hessian()).dx();

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, NonlinearDiffusionJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // R(u;v) = inner(k(u) * grad(u), grad(v)), k(u) = 1 + u^2
    const auto k = FormExpr::constant(1.0) + (u * u);
    const auto residual = inner(k * grad(u), grad(v)).dx();

    auto ir_ad = compiler.compileResidual(residual);
    auto ir_sym = compiler.compileResidual(residual);

    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J_ad(4);
    assembly::DenseVectorView R_ad(4);
    J_ad.zero();
    R_ad.zero();
    (void)assembler.assembleBoth(mesh, space, space, ad_kernel, J_ad, R_ad);

    assembly::DenseMatrixView J_sym(4);
    assembly::DenseVectorView R_sym(4);
    J_sym.zero();
    R_sym.zero();
    (void)assembler.assembleBoth(mesh, space, space, sym_kernel, J_sym, R_sym);

    expectDenseNear(R_sym, R_ad, 1e-12);
    expectDenseNear(J_sym, J_ad, 1e-12);
}

TEST(SymbolicNonlinearFormKernelTest, ComparisonPredicatesMatchAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto c = FormExpr::constant(0.123);
    const auto preds =
        lt(u, c) + le(u, c) + gt(u, c) + ge(u, c) + eq(u, c) + ne(u, c);

    const auto residual = (preds * v).dx();

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, ScalarFunctionRulesMatchAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto expr =
        sqrt(FormExpr::constant(1.0) + u * u) +
        exp(u) +
        log(FormExpr::constant(2.0) + u * u) +
        abs(u) +
        pow(u, FormExpr::constant(3.0));

    const auto residual = (expr * v).dx();

    const std::vector<Real> U = {0.6, -0.4, 0.2, -0.3};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, PowerExponentZeroMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (pow(u, FormExpr::constant(0.0)) * v).dx();

    const std::vector<Real> U = {0.0, 0.0, 0.0, 0.0};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, CurlJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto residual = inner(curl(u), curl(v)).dx();

    const std::vector<Real> U = {
        0.1, -0.05, 0.07, -0.02,
        -0.03, 0.04, -0.06, 0.08,
        0.02, 0.01, -0.04, 0.05
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, OuterAndCrossProductRulesMatchAD)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto c = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});

    // R = inner(outer(u,u), grad(v)) + inner(cross(u,c), v)
    const auto residual = (inner(outer(u, u), grad(v)) + inner(cross(u, c), v)).dx();

    const std::vector<Real> U = {
        0.1, -0.05, 0.07, -0.02,
        -0.03, 0.04, -0.06, 0.08,
        0.02, 0.01, -0.04, 0.05
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, DeterminantAtZeroDoesNotThrowAndMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = makeSingleCellDofMap(12);

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = grad(u);            // 3x3
    const auto s = det(A);             // scalar
    const auto e0 = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(0.0), FormExpr::constant(0.0)});
    const auto residual = inner(s * e0, v).dx();

    const std::vector<Real> U(12, 0.0);
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, HyperelasticLikeJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = makeSingleCellDofMap(12);

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto F = FormExpr::identity(3) + grad(u);
    const auto J = det(F);
    const auto FinvT = inv(transpose(F));

    // Simple neo-Hookean-like P = (F - F^{-T}) + alpha*log(J)*F^{-T}
    const auto alpha = FormExpr::constant(0.1);
    const auto P = (F - FinvT) + (alpha * log(J) * FinvT);
    const auto residual = P.doubleContraction(grad(v)).dx();

    const std::vector<Real> U = {
        0.01, -0.02, 0.03, -0.01, // ux dofs
        0.02, 0.01, -0.01, 0.03,  // uy dofs
        -0.01, 0.02, 0.01, -0.02  // uz dofs
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, NavierStokesConvectionJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = makeSingleCellDofMap(12);

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto gu = grad(u); // 3x3

    std::vector<FormExpr> conv;
    conv.reserve(3);
    for (int i = 0; i < 3; ++i) {
        FormExpr sum = FormExpr::constant(0.0);
        for (int j = 0; j < 3; ++j) {
            sum = sum + u.component(j) * gu.component(i, j);
        }
        conv.push_back(sum);
    }
    const auto convection = FormExpr::asVector(std::move(conv));

    const auto residual = inner(convection, v).dx();

    const std::vector<Real> U = {
        0.1, -0.05, 0.07, -0.02,
        -0.03, 0.04, -0.06, 0.08,
        0.02, 0.01, -0.04, 0.05
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, MatrixTensorOpsJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = makeSingleCellDofMap(12);

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto F = FormExpr::identity(3) + grad(u);
    const auto K = cofactor(F) + dev(sym(F)) + skew(F);

    const auto term1 = K.doubleContraction(grad(v));
    const auto term2 = trace(F) * v.component(0);
    const auto term3 = norm(F) * v.component(1);

    const auto residual = (term1 + term2 + term3).dx();

    const std::vector<Real> U = {
        0.02, -0.01, 0.03, -0.02,
        -0.02, 0.01, -0.03, 0.02,
        0.01, 0.02, -0.01, -0.02
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, CofactorIsolatedJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = makeSingleCellDofMap(12);

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto F = FormExpr::identity(3) + grad(u);
    const auto residual = inner(cofactor(F), grad(v)).dx();

    const std::vector<Real> U = {
        0.01, -0.02, 0.03, -0.01,
        0.02, 0.01, -0.01, 0.03,
        -0.01, 0.02, 0.01, -0.02
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, DGInteriorPenaltyJacobianMatchesAD)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::L2Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto penalty = FormExpr::constant(10.0);
    const auto sipg =
        (-inner(avg(grad(u)), jump(v)) +
         penalty * inner(jump(u), jump(v))).dS();

    const std::vector<Real> U = {0.1, -0.2, 0.3, -0.1, 0.05, -0.07, 0.08, -0.09};
    compareADAndSymbolic(mesh, dof_map, space, space, sipg, U);
}

TEST(SymbolicNonlinearFormKernelTest, RestrictMinusPlusJacobianMatchesAD)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::L2Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto residual = (u.minus() * v.minus() + u.plus() * v.plus()).dS();

    const std::vector<Real> U = {0.1, -0.2, 0.3, -0.1, 0.05, -0.07, 0.08, -0.09};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, ConditionalAndMinMaxRulesMatchAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto c = FormExpr::constant(0.15);
    const auto expr =
        ge(u, FormExpr::constant(0.0)).conditional(u * u, -u) +
        min(u, c) +
        max(u, -c) +
        sign(u);

    const auto residual = (expr * v).dx();

    const std::vector<Real> U = {0.2, -0.1, 0.3, -0.25};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, ZeroVectorNormAndNormalizeMatchAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto residual =
        (norm(grad(u)) * v +
         inner(normalize(grad(u)), grad(v))).dx();

    // Constant u => grad(u) = 0 at all quadrature points.
    const std::vector<Real> U = {0.2, 0.2, 0.2, 0.2};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, TimeDerivativeJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto residual = (u.dt(1) * v + inner(grad(u), grad(v))).dx();

    assembly::TimeIntegrationContext tctx;
    tctx.integrator_name = "backward_euler";
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    const Real dt = 0.25;
    dt1.a = {1.0 / dt, -1.0 / dt};
    tctx.dt1 = dt1;

    CompareOptions opts;
    opts.time_ctx = &tctx;
    opts.time_step = dt;
    opts.prev_solution = std::vector<Real>{0.0, 0.1, -0.05, 0.2};

    const std::vector<Real> U = {0.2, -0.1, 0.05, -0.15};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U, opts);
}

class InlinableSquareModel final : public ConstitutiveModel, public InlinableConstitutiveModel {
public:
    [[nodiscard]] const InlinableConstitutiveModel* inlinable() const noexcept override { return this; }
    [[nodiscard]] std::uint64_t kindId() const noexcept override { return fnv1a64("test.InlinableSquareModel.v1"); }
    [[nodiscard]] MaterialStateAccess stateAccess() const noexcept override { return MaterialStateAccess::None; }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind() const override { return ValueKind::Scalar; }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, int /*dim*/) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("InlinableSquareModel: expected scalar input");
        }
        Value<Real> out;
        out.kind = Value<Real>::Kind::Scalar;
        out.s = input.s * input.s;
        return out;
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       int /*dim*/,
                                       DualWorkspace& workspace) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("InlinableSquareModel: expected scalar input (dual)");
        }
        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, input.s, makeDualConstant(0.0, workspace.alloc()));
        return out;
    }

    [[nodiscard]] InlinedConstitutiveExpansion
    inlineExpand(std::span<const FormExpr> inputs, const InlinableConstitutiveContext& /*ctx*/) const override
    {
        if (inputs.size() != 1u || !inputs[0].isValid()) {
            throw std::invalid_argument("InlinableSquareModel::inlineExpand: expected 1 valid input");
        }
        InlinedConstitutiveExpansion out;
        out.outputs.push_back(inputs[0] * inputs[0]);
        return out;
    }

    [[nodiscard]] OutputSpec outputSpec(std::size_t output_index) const override
    {
        if (output_index != 0u) {
            throw std::invalid_argument("InlinableSquareModel::outputSpec: output_index out of range");
        }
        OutputSpec spec;
        spec.kind = ValueKind::Scalar;
        return spec;
    }
};

TEST(SymbolicNonlinearFormKernelTest, InlinableConstitutiveJacobianMatchesAD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    auto model = std::make_shared<InlinableSquareModel>();
    const auto sigma = FormExpr::constitutive(std::move(model), u);
    const auto residual = (sigma * v).dx();

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(SymbolicNonlinearFormKernelTest, FiniteDifferenceMatchesSymbolicTangent)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto k = FormExpr::constant(1.0) + (u * u);
    const auto residual = inner(k * grad(u), grad(v)).dx();

    FormCompiler compiler;
    auto ir_both = compiler.compileResidual(residual);
    auto ir_vec = compiler.compileResidual(residual);

    SymbolicNonlinearFormKernel k_both(std::move(ir_both), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel k_vec(std::move(ir_vec), NonlinearKernelOutput::VectorOnly);
    k_both.resolveInlinableConstitutives();
    k_vec.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n = dof_map.getNumDofs();
    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    const std::vector<Real> dU = {0.2, -0.1, 0.05, 0.03};

    assembly::DenseMatrixView J(n);
    assembly::DenseVectorView R(n);
    J.zero();
    R.zero();
    assembler.setCurrentSolution(U);
    (void)assembler.assembleBoth(mesh, space, space, k_both, J, R);

    const Real eps = 1e-7;
    std::vector<Real> U_eps = U;
    for (std::size_t i = 0; i < U_eps.size(); ++i) {
        U_eps[i] += eps * dU[i];
    }

    assembly::DenseMatrixView dummy(n);
    assembly::DenseVectorView R_eps(n);
    dummy.zero();
    R_eps.zero();
    assembler.setCurrentSolution(U_eps);
    (void)assembler.assembleBoth(mesh, space, space, k_vec, dummy, R_eps);

    // FD approximation: (R(u+eps*dU) - R(u)) / eps
    std::vector<Real> fd(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        fd[static_cast<std::size_t>(i)] =
            (R_eps.getVectorEntry(i) - R.getVectorEntry(i)) / eps;
    }

    // Predicted directional derivative: J(u) * dU
    std::vector<Real> JdU(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        Real sum = 0.0;
        for (GlobalIndex j = 0; j < n; ++j) {
            sum += J.getMatrixEntry(i, j) * dU[static_cast<std::size_t>(j)];
        }
        JdU[static_cast<std::size_t>(i)] = sum;
    }

    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_NEAR(fd[static_cast<std::size_t>(i)], JdU[static_cast<std::size_t>(i)], 1e-6);
    }
}

TEST(SymbolicNonlinearFormKernelBenchmark, DISABLED_ADvsSymbolic_NonlinearDiffusion_CellMatrix)
{
    SingleTetraMeshAccess mesh;

    auto benchOrder = [&](int order) {
        spaces::H1Space space(ElementType::Tetra4, order);
        const auto n = static_cast<GlobalIndex>(space.dofs_per_element());
        auto dof_map = makeSingleCellDofMap(n);

        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");

        const auto k = FormExpr::constant(1.0) + (u * u);
        const auto residual = inner(k * grad(u), grad(v)).dx();

        FormCompiler compiler;
        auto ir_ad = compiler.compileResidual(residual);
        auto ir_sym = compiler.compileResidual(residual);

        NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward, NonlinearKernelOutput::MatrixOnly);
        SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::MatrixOnly);
        sym_kernel.resolveInlinableConstitutives();

        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);

        assembly::DenseMatrixView J(n);
        std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
        for (GlobalIndex i = 0; i < n; ++i) {
            U[static_cast<std::size_t>(i)] = 0.01 * static_cast<Real>(i + 1);
        }
        assembler.setCurrentSolution(U);

        // Warm-up
        J.zero();
        (void)assembler.assembleMatrix(mesh, space, space, ad_kernel, J);
        J.zero();
        (void)assembler.assembleMatrix(mesh, space, space, sym_kernel, J);

        const int iters = (n <= 4) ? 300 : (n <= 10) ? 120 : 40;

        auto bench = [&](assembly::AssemblyKernel& kernel) {
            const double t0 = static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
            for (int it = 0; it < iters; ++it) {
                J.zero();
                (void)assembler.assembleMatrix(mesh, space, space, kernel, J);
            }
            const double t1 = static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
            const double dt = t1 - t0;
            return (dt / static_cast<double>(iters)) * 1e6;
        };

        const double us_ad = bench(ad_kernel);
        const double us_sym = bench(sym_kernel);

        std::cout << "[BENCH] order=" << order << " dofs=" << n
                  << " AD=" << us_ad << " us"
                  << " symbolic=" << us_sym << " us";
        if (us_sym > 0.0) {
            std::cout << " speedup=" << (us_ad / us_sym) << "x";
        }
        std::cout << " (iters=" << iters << ")\n";
    };

    benchOrder(1);
    benchOrder(2);
    benchOrder(3);
}

#if SVMP_FE_ENABLE_LLVM_JIT
TEST(SymbolicNonlinearFormKernelTest, JITMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (u * u * v).dx();

    auto ir_interp = compiler.compileResidual(residual);
    auto ir_jit = compiler.compileResidual(residual);

    SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel.resolveInlinableConstitutives();

    auto jit_fallback = std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;

    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J_interp(4);
    assembly::DenseVectorView R_interp(4);
    J_interp.zero();
    R_interp.zero();
    (void)assembler.assembleBoth(mesh, space, space, interp_kernel, J_interp, R_interp);

    assembly::DenseMatrixView J_jit(4);
    assembly::DenseVectorView R_jit(4);
    J_jit.zero();
    R_jit.zero();
    (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, 1e-12);
    expectDenseNear(J_jit, J_interp, 1e-12);
}

TEST(SymbolicNonlinearFormKernelTest, JITMatchesInterpreter_Triangle3_GradAndGeometryOps)
{
    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // Exercise 2D geometry terminals (Jinv) and differential ops (grad) together.
    const auto Jinv = FormExpr::jacobianInverse();
    const auto K = transpose(Jinv) * Jinv;
    const auto residual = (inner(grad(u), grad(u)) * v + trace(K) * u * u * v).dx();

    auto ir_interp = compiler.compileResidual(residual);
    auto ir_jit = compiler.compileResidual(residual);

    SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel.resolveInlinableConstitutives();

    auto jit_fallback = std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;

    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J_interp(3);
    assembly::DenseVectorView R_interp(3);
    J_interp.zero();
    R_interp.zero();
    (void)assembler.assembleBoth(mesh, space, space, interp_kernel, J_interp, R_interp);

    assembly::DenseMatrixView J_jit(3);
    assembly::DenseVectorView R_jit(3);
    J_jit.zero();
    R_jit.zero();
    (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, 1e-12);
    expectDenseNear(J_jit, J_interp, 1e-12);
}

TEST(SymbolicNonlinearFormKernelTest, JITMatchesInterpreter_Triangle3_VectorConvection)
{
    SingleTriangleMeshAccess mesh;
    auto dof_map = makeSingleCellDofMap(6);

    auto base = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    spaces::ProductSpace space(base, 2);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // Exercise matrix-vector multiplication with non-scalar operands:
    //   convection = grad(u) * u
    const auto convection = grad(u) * u;
    const auto residual = inner(convection, v).dx();

    auto ir_interp = compiler.compileResidual(residual);
    auto ir_jit = compiler.compileResidual(residual);

    SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel.resolveInlinableConstitutives();

    auto jit_fallback = std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;

    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    // Nonzero current solution needed to activate convection and its tangent terms.
    std::vector<Real> U = {0.1, -0.2, 0.3, -0.05, 0.07, -0.01};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J_interp(6);
    assembly::DenseVectorView R_interp(6);
    J_interp.zero();
    R_interp.zero();
    (void)assembler.assembleBoth(mesh, space, space, interp_kernel, J_interp, R_interp);

    assembly::DenseMatrixView J_jit(6);
    assembly::DenseVectorView R_jit(6);
    J_jit.zero();
    R_jit.zero();
    (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, 1e-12);
    expectDenseNear(J_jit, J_interp, 1e-12);
}
#endif

TEST(SymbolicDifferentiationNewPhysicsTest, MatrixFunctionDerivativesUseDirectionalDerivativeNodes)
{
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // SPD-ish 2x2 matrix built from a scalar field (keeps log/sqrt well-defined in practice).
    const auto A = FormExpr::identity(2) * (u * u + FormExpr::constant(Real(2.0)));

    const auto t_exp = differentiateResidual((A.matrixExp().trace() * v).dx());
    EXPECT_NE(t_exp.toString().find("expm_dd("), std::string::npos);

    const auto t_log = differentiateResidual((A.matrixLog().trace() * v).dx());
    EXPECT_NE(t_log.toString().find("logm_dd("), std::string::npos);

    const auto t_sqrt = differentiateResidual((A.matrixSqrt().trace() * v).dx());
    EXPECT_NE(t_sqrt.toString().find("sqrtm_dd("), std::string::npos);

    const auto t_pow = differentiateResidual((A.matrixPow(FormExpr::constant(Real(2.0))).trace() * v).dx());
    EXPECT_NE(t_pow.toString().find("powm_dd("), std::string::npos);
}

TEST(SymbolicDifferentiationNewPhysicsTest, EigenAndHistoryDerivativesUseDirectionalDerivativeNodes)
{
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto A = FormExpr::identity(2) * (u * u + FormExpr::constant(Real(1.0)));

    const auto t_eigvec =
        differentiateResidual((inner(A.symmetricEigenvector(0), A.symmetricEigenvector(0)) * v).dx());
    EXPECT_NE(t_eigvec.toString().find("eigvec_sym_dd("), std::string::npos);

    const auto t_spec = differentiateResidual((A.spectralDecomposition().trace() * v).dx());
    EXPECT_NE(t_spec.toString().find("spectral_decomp_dd("), std::string::npos);

    const auto hist = FormExpr::historyWeightedSum({u, FormExpr::constant(Real(0.0))});
    const auto t_hist = differentiateResidual((hist * v).dx());
    EXPECT_NE(t_hist.toString().find("u_prev(1)"), std::string::npos);
}

TEST(SymbolicDifferentiationNewPhysicsTest, SmoothOpsDifferentiateWithoutThrow)
{
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto eps = FormExpr::constant(Real(1e-3));

    const auto t = differentiateResidual((u.smoothAbs(eps) * v).dx());
    EXPECT_NE(t.toString().find("smooth_abs("), std::string::npos);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
