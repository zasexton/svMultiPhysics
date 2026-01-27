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
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

namespace svmp::FE::forms::test {

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
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

TEST(SpectralEigenJIT, EigenNodesMatchInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // 2x2 symmetric matrices (row-major literals).
    const auto A2 = FormExpr::asTensor({{FormExpr::constant(1.2), FormExpr::constant(0.3)},
                                        {FormExpr::constant(0.3), FormExpr::constant(0.7)}});
    const auto dA2 = FormExpr::asTensor({{FormExpr::constant(0.1), FormExpr::constant(0.05)},
                                         {FormExpr::constant(0.05), FormExpr::constant(-0.2)}});

    const auto lam2 = A2.symmetricEigenvalue(0);
    const auto dd2 = FormExpr::symmetricEigenvalueDirectionalDerivative(A2, dA2, 0);
    const auto ddA2 = FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(A2, dA2, dA2, 0);

    // 3x3 symmetric matrices.
    const auto A3 = FormExpr::asTensor({{FormExpr::constant(2.0), FormExpr::constant(0.1), FormExpr::constant(0.2)},
                                        {FormExpr::constant(0.1), FormExpr::constant(1.5), FormExpr::constant(0.05)},
                                        {FormExpr::constant(0.2), FormExpr::constant(0.05), FormExpr::constant(1.0)}});
    const auto dA3 = FormExpr::asTensor({{FormExpr::constant(-0.3), FormExpr::constant(0.02), FormExpr::constant(0.04)},
                                         {FormExpr::constant(0.02), FormExpr::constant(0.15), FormExpr::constant(0.01)},
                                         {FormExpr::constant(0.04), FormExpr::constant(0.01), FormExpr::constant(0.05)}});

    const auto lam3 = A3.symmetricEigenvalue(1);
    const auto dd3 = FormExpr::symmetricEigenvalueDirectionalDerivative(A3, dA3, 1);
    const auto ddA3 = FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(A3, dA3, dA3, 1);

    const auto form = ((lam2 + dd2 + ddA2 + lam3 + dd3 + ddA3) * u * v).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(dof_map.getNumDofs());
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(dof_map.getNumDofs());
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);
}

#endif // SVMP_FE_ENABLE_LLVM_JIT

} // namespace svmp::FE::forms::test

