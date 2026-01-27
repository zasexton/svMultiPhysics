/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <vector>

namespace svmp::FE::forms::test {

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

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

} // namespace

TEST(HessianVectorProduct, MatchesFiniteDifferenceOfJacobian)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr FieldId w_field = 101;

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // Residual: R(u;v) = ∫ u^3 v dx
    const auto residual = (u * u * u * v).dx();

    const auto tangent = forms::differentiateResidual(residual);
    const auto w = FormExpr::stateField(w_field, space, "w");
    const auto hvp = forms::differentiateResidualHessianVector(residual, w);

    FormCompiler compiler;
    auto ir_J = compiler.compileBilinear(tangent);
    auto ir_H = compiler.compileBilinear(hvp);

    FormKernel J_kernel(std::move(ir_J));
    FormKernel H_kernel(std::move(ir_H));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n = dof_map.getNumDofs();
    ASSERT_EQ(n, 4);

    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    const std::vector<Real> W = {0.2, -0.1, 0.05, 0.03};

    const std::array<assembly::FieldSolutionAccess, 1> field_access = {
        assembly::FieldSolutionAccess{w_field, &space, &dof_map, n},
    };
    assembler.setFieldSolutionAccess(field_access);

    std::vector<Real> state(static_cast<std::size_t>(2 * n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] = U[static_cast<std::size_t>(i)];
        state[static_cast<std::size_t>(n + i)] = W[static_cast<std::size_t>(i)];
    }

    assembly::DenseMatrixView J0(n);
    J0.zero();
    assembler.setCurrentSolution(state);
    (void)assembler.assembleMatrix(mesh, space, space, J_kernel, J0);

    const Real eps = 1e-7;
    std::vector<Real> state_eps = state;
    for (GlobalIndex i = 0; i < n; ++i) {
        state_eps[static_cast<std::size_t>(i)] += eps * W[static_cast<std::size_t>(i)];
    }

    assembly::DenseMatrixView J1(n);
    J1.zero();
    assembler.setCurrentSolution(state_eps);
    (void)assembler.assembleMatrix(mesh, space, space, J_kernel, J1);

    assembly::DenseMatrixView Jfd(n);
    Jfd.zero();
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            const Real fd_value = (J1.getMatrixEntry(i, j) - J0.getMatrixEntry(i, j)) / eps;
            Jfd.addMatrixEntry(i, j, fd_value, assembly::AddMode::Insert);
        }
    }

    assembly::DenseMatrixView H(n);
    H.zero();
    assembler.setCurrentSolution(state);
    (void)assembler.assembleMatrix(mesh, space, space, H_kernel, H);

    expectDenseNear(H, Jfd, 1e-6);
}

#if SVMP_FE_ENABLE_LLVM_JIT
TEST(HessianVectorProductJIT, MatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr FieldId w_field = 101;

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto residual = (u * u * u * v).dx();
    const auto w = FormExpr::stateField(w_field, space, "w");
    const auto hvp = forms::differentiateResidualHessianVector(residual, w);

    FormCompiler compiler;
    auto ir_interp = compiler.compileBilinear(hvp);
    auto ir_jit = compiler.compileBilinear(hvp);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n = dof_map.getNumDofs();
    const std::vector<Real> U = {0.3, -0.2, 0.1, -0.4};
    const std::vector<Real> W = {0.2, -0.1, 0.05, 0.03};

    const std::array<assembly::FieldSolutionAccess, 1> field_access = {
        assembly::FieldSolutionAccess{w_field, &space, &dof_map, n},
    };
    assembler.setFieldSolutionAccess(field_access);

    std::vector<Real> state(static_cast<std::size_t>(2 * n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] = U[static_cast<std::size_t>(i)];
        state[static_cast<std::size_t>(n + i)] = W[static_cast<std::size_t>(i)];
    }

    assembler.setCurrentSolution(state);

    assembly::DenseMatrixView H_interp(n);
    H_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, H_interp);

    assembly::DenseMatrixView H_jit(n);
    H_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, H_jit);

    expectDenseNear(H_jit, H_interp, 1e-12);
}
#endif

} // namespace svmp::FE::forms::test
