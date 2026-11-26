/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Tensor/SpectralEigen.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

namespace svmp::FE::forms::test {

TEST(SpectralEigen, SimpleRootDerivativesAcrossUnitsAndShifts)
{
    for (const auto [scale, shift] : {std::pair{1.e-18, 0.}, {1.e-9, 0.}, {1., 1.e15}, {1.e110, 0.}}) {
        SCOPED_TRACE(::testing::Message() << "scale=" << scale << ", shift=" << shift);
        const std::array<double, 4> a2{shift + scale, 0., 0., shift + 2*scale};
        const std::array<double, 4> cross2{0., 1., 1., 0.}, xy{0., scale, scale, 0.};
        const std::array<double, 9> a3{shift + scale, 0., 0., 0., shift + 2*scale, 0., 0., 0., shift + 3*scale};
        const std::array<double, 9> zz{0., 0., 0., 0., 0., 0., 0., 0., 1.};
        const std::array<double, 9> cross3{0., 0., 0., 0., 0., 1., 0., 1., 0.};
        const std::array<double, 9> yz{0., 0., 0., 0., 0., scale, 0., scale, 0.};
        EXPECT_NEAR(svmp_fe_sym_eigenvalue_dd_3x3_v1(a3.data(), zz.data(), 0), 1., 1e-13);
        EXPECT_NEAR(svmp_fe_sym_eigenvalue_ddA_2x2_v1(a2.data(), cross2.data(), xy.data(), 0), 2., 1e-12);
        const double hessian = svmp_fe_sym_eigenvalue_ddA_3x3_v1(a3.data(), cross3.data(), yz.data(), 0);
        EXPECT_NEAR(hessian, 2., 1e-12);
        auto plus = a3, minus = a3;
        constexpr double h = 1e-5;
        for (std::size_t k = 0; k < yz.size(); ++k) {
            plus[k] += h * yz[k];
            minus[k] -= h * yz[k];
        }
        const double fd = (svmp_fe_sym_eigenvalue_dd_3x3_v1(plus.data(), cross3.data(), 0) -
                           svmp_fe_sym_eigenvalue_dd_3x3_v1(minus.data(), cross3.data(), 0)) / (2*h);
        EXPECT_NEAR(hessian, fd, 1e-8);
    }
    const std::array<double, 4> a{0., 0., 0., 1.e-210};
    const std::array<double, 4> b{0., 1.e-240, 1.e-240, 0.}, d{0., 1.e240, 1.e240, 0.};
    EXPECT_NEAR(svmp_fe_sym_eigenvalue_ddA_2x2_v1(a.data(), b.data(), d.data(), 0) / 1.e210, 2., 1e-13);
    const std::array<double, 4> wide{-1.e308, 0., 0., 1.e308}, cross{0., 1.e154, 1.e154, 0.};
    EXPECT_NEAR(svmp_fe_sym_eigenvalue_ddA_2x2_v1(wide.data(), cross.data(), cross.data(), 0), 1., 1e-13);
    const double limit = std::numeric_limits<double>::max();
    std::array<double, 9> shifted{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            shifted[3*i+j] = (i == j ? .30 : -.63) * limit;
    const std::array<double, 9> xx{std::sqrt(limit)};
    EXPECT_NEAR(svmp_fe_sym_eigenvalue_ddA_3x3_v1(shifted.data(), xx.data(), xx.data(), 2),
                -4. / (9. * 1.89), 1.e-14);
}

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
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());

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

TEST(SpectralEigenJIT, EigenNodesHandleSIMDCellVaryingInputs)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto volume = FormExpr::cellVolume();
    const auto A = FormExpr::asTensor({{FormExpr::constant(1.0) + volume, FormExpr::constant(0.08)},
                                       {FormExpr::constant(0.08), FormExpr::constant(1.7) - volume}});
    const auto dA = FormExpr::asTensor({{FormExpr::constant(0.3) * volume, FormExpr::constant(0.02)},
                                        {FormExpr::constant(0.02), FormExpr::constant(0.11) + volume}});

    const auto lam = A.symmetricEigenvalue(0);
    const auto dd = FormExpr::symmetricEigenvalueDirectionalDerivative(A, dA, 0);
    const auto ddA = FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(A, dA, dA, 0);
    const auto form = ((lam + dd + ddA) * u * v).dx();

    auto interp_kernel = std::make_shared<FormKernel>(compiler.compileBilinear(form));
    auto jit_fallback = std::make_shared<FormKernel>(compiler.compileBilinear(form));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());

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
