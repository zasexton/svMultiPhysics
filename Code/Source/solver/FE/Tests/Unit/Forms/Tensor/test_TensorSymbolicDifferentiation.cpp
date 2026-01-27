/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_TensorSymbolicDifferentiation.cpp
 * @brief Verification tests for tensor/index-notation symbolic differentiation
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/Dual.h"
#include "Forms/Einsum.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/Tensor/TensorContraction.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <vector>

namespace svmp::FE::forms::tensor {

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

void compareADAndSymbolic(const assembly::IMeshAccess& mesh,
                          const dofs::DofMap& dof_map,
                          const spaces::FunctionSpace& test_space,
                          const spaces::FunctionSpace& trial_space,
                          const FormExpr& residual,
                          const std::vector<Real>& U,
                          Real tol = 1e-12)
{
    FormCompiler compiler;
    auto opts = compiler.options();
    opts.jit.enable = true;
    compiler.setOptions(std::move(opts));

    auto ir_ad = compiler.compileResidual(residual);
    auto ir_sym = compiler.compileResidual(residual);

    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

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

    expectDenseNear(R_sym, R_ad, tol);
    expectDenseNear(J_sym, J_ad, tol);
}

} // namespace

TEST(TensorSymbolicDifferentiation, NonlinearDiffusionIndexNotationMatchesAD)
{
    forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    forms::Index i("i");
    const auto k = FormExpr::constant(1.0) + (u * u);
    const auto integrand = (k * grad(u)(i) * grad(v)(i));
    const auto residual = integrand.dx();

    // Smoke-check: symbolic tangent remains scalar after lowering IndexedAccess.
    const auto dI = forms::differentiateResidual(integrand);
    const auto a = analyzeContractions(dI);
    EXPECT_TRUE(a.ok);
    EXPECT_TRUE(a.free_indices.empty());
    const auto expanded = forms::einsum(dI);
    ASSERT_NE(expanded.node(), nullptr);
    EXPECT_NE(expanded.node()->type(), FormExprType::AsVector);
    EXPECT_NE(expanded.node()->type(), FormExprType::AsTensor);

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(TensorSymbolicDifferentiation, HyperelasticLikeDetFDotMatchesAD)
{
    forms::test::SingleTetraMeshAccess mesh;

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto F = FormExpr::identity(3) + grad(u);
    const auto J = det(F);

    forms::Index i("i");
    const auto residual = (J * u(i) * v(i)).dx();

    const std::vector<Real> U = {
        0.01, -0.02, 0.03, -0.01,
        0.02, 0.01, -0.01, 0.03,
        -0.01, 0.02, 0.01, -0.02
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(TensorSymbolicDifferentiation, SymmetricEigenvalueMatchesAD)
{
    forms::test::SingleTetraMeshAccess mesh;

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 2);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto u0 = u.component(0);
    const auto u1 = u.component(1);
    const auto v0 = v.component(0);

    std::vector<std::vector<FormExpr>> rows(2);
    rows[0] = {FormExpr::constant(1.0) + u0 * u0, u0 * u1};
    rows[1] = {u0 * u1, FormExpr::constant(2.0) + u1 * u1};
    const auto A = FormExpr::asTensor(std::move(rows));

    const auto lam0 = A.symmetricEigenvalue(0);
    const auto residual = (lam0 * u0 * v0).dx();

    // Pick values away from eigenvalue degeneracy.
    const std::vector<Real> U = {0.11, -0.07, 0.05, -0.13,
                                 0.09, 0.02, -0.04, 0.08};
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(TensorSymbolicDifferentiation, ConvectionIndexNotationMatchesAD)
{
    forms::test::SingleTetraMeshAccess mesh;

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto gu = grad(u); // 3x3
    forms::Index i("i");
    forms::Index j("j");

    // (u · ∇)u · v = u_j * (∂u_i/∂x_j) * v_i
    const auto residual = (u(j) * gu(i, j) * v(i)).dx();

    const std::vector<Real> U = {
        0.1, -0.05, 0.07, -0.02,
        -0.03, 0.04, -0.06, 0.08,
        0.02, 0.01, -0.04, 0.05
    };
    compareADAndSymbolic(mesh, dof_map, space, space, residual, U);
}

TEST(TensorSymbolicDifferentiation, FiniteDifferenceMatchesSymbolicTangentIndexNotation)
{
    forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    forms::Index i("i");
    const auto k = FormExpr::constant(1.0) + (u * u);
    const auto residual = (k * grad(u)(i) * grad(v)(i)).dx();

    FormCompiler compiler;
    auto opts = compiler.options();
    opts.jit.enable = true;
    compiler.setOptions(std::move(opts));

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
    for (std::size_t idx = 0; idx < U_eps.size(); ++idx) {
        U_eps[idx] += eps * dU[idx];
    }

    assembly::DenseMatrixView dummy(n);
    assembly::DenseVectorView R_eps(n);
    dummy.zero();
    R_eps.zero();
    assembler.setCurrentSolution(U_eps);
    (void)assembler.assembleBoth(mesh, space, space, k_vec, dummy, R_eps);

    // FD approximation: (R(u+eps*dU) - R(u)) / eps
    std::vector<Real> fd(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex row = 0; row < n; ++row) {
        fd[static_cast<std::size_t>(row)] =
            (R_eps.getVectorEntry(row) - R.getVectorEntry(row)) / eps;
    }

    // Predicted directional derivative: J(u) * dU
    std::vector<Real> JdU(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex row = 0; row < n; ++row) {
        Real sum = 0.0;
        for (GlobalIndex col = 0; col < n; ++col) {
            sum += J.getMatrixEntry(row, col) * dU[static_cast<std::size_t>(col)];
        }
        JdU[static_cast<std::size_t>(row)] = sum;
    }

    for (GlobalIndex row = 0; row < n; ++row) {
        EXPECT_NEAR(fd[static_cast<std::size_t>(row)], JdU[static_cast<std::size_t>(row)], 1e-6);
    }
}

} // namespace svmp::FE::forms::tensor
