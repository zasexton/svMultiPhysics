/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace forms {
namespace test {
namespace {

struct CellEnv {
    SingleTetraMeshAccess mesh{};
    dofs::DofMap dof_map{createSingleTetraDofMap()};
    spaces::H1Space space{ElementType::Tetra4, 1};
    assembly::FunctionalAssembler assembler{};
    Real volume{0.0};

    CellEnv()
    {
        assembler.setMesh(mesh);
        assembler.setDofMap(dof_map);
        assembler.setSpace(space);

        const auto one = FormExpr::constant(Real(1.0));
        auto one_kernel = makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell);
        volume = assembler.assembleScalar(*one_kernel);
    }
};

Real assembleJIT(const FormExpr& integrand, CellEnv& env)
{
    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());
    return env.assembler.assembleScalar(jit_kernel);
}

Real assembleInterp(const FormExpr& integrand, CellEnv& env)
{
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);
    return env.assembler.assembleScalar(*interp);
}

} // namespace

TEST(LLVMGenTensorOps, NormVector_ComputesL2Norm)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto v = FormExpr::asVector({FormExpr::constant(Real(3.0)),
                                       FormExpr::constant(Real(4.0)),
                                       FormExpr::constant(Real(0.0))});
    const auto integrand = v.norm();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(5.0), 1e-12);
}

TEST(LLVMGenTensorOps, NormMatrix_ComputesFrobeniusNorm)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
    });
    const auto integrand = A.norm();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(5.0), 1e-12);
}

TEST(LLVMGenTensorOps, NormalizeVector_ProducesUnitVector)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto v = FormExpr::asVector({FormExpr::constant(Real(3.0)),
                                       FormExpr::constant(Real(4.0)),
                                       FormExpr::constant(Real(0.0))});
    const auto u = v.normalize();

    // Encode two components in one scalar to keep this test to one compilation.
    const auto integrand = u.component(0) + u.component(1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(8.6), 1e-12);
}

TEST(LLVMGenTensorOps, Transpose_SwapsRowsAndCols)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto At = A.transpose();
    const auto integrand = At.component(0, 1) + At.component(1, 0) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    // At(0,1)=A(1,0)=3 and At(1,0)=A(0,1)=2 => 3 + 10*2 = 23
    EXPECT_NEAR(jit / env.volume, Real(23.0), 1e-12);
}

TEST(LLVMGenTensorOps, Trace_SumsDiagonal)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto integrand = A.trace();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(5.0), 1e-12);
}

TEST(LLVMGenTensorOps, Determinant2x2_ComputesCorrectly)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto integrand = A.det();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(-2.0), 1e-12);
}

TEST(LLVMGenTensorOps, Inverse2x2_ComputesCorrectly)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto invA = A.inv();
    const auto integrand = invA.component(0, 0) + invA.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    // inv(A) = 1/(-2) * [[4,-2],[-3,1]] => inv(0,0)=-2, inv(1,1)=-0.5
    EXPECT_NEAR(jit / env.volume, Real(-2.0) + Real(10.0) * Real(-0.5), 1e-12);
}

TEST(LLVMGenTensorOps, Cofactor_ComputesCorrectly)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto C = A.cofactor();
    const auto integrand = C.component(0, 0) + C.component(0, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    // Cofactor(A) = [[4, -3], [-2, 1]]
    EXPECT_NEAR(jit / env.volume, Real(4.0) + Real(10.0) * Real(-3.0), 1e-12);
}

TEST(LLVMGenTensorOps, SymmetricPart_Symmetrizes)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto S = A.sym();
    const auto integrand = S.component(0, 1);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(2.5), 1e-12);
}

TEST(LLVMGenTensorOps, SkewPart_SkewSymmetrizes)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto W = A.skew();
    const auto integrand = W.component(0, 1);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(-0.5), 1e-12);
}

TEST(LLVMGenTensorOps, Deviator_SubtractsHydrostatic)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
    });
    const auto devA = A.dev();

    // trace(dev(A)) = 0 and dev(A)_{11} = -1 for this input (trace=3 => subtract 1*I).
    const auto integrand = devA.trace() + devA.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    EXPECT_NEAR(jit / env.volume, Real(0.0) + Real(10.0) * Real(-1.0), 1e-12);
}

TEST(LLVMGenTensorOps, OuterProduct_ProducesCorrectMatrix)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto a = FormExpr::asVector({FormExpr::constant(Real(1.0)),
                                       FormExpr::constant(Real(2.0)),
                                       FormExpr::constant(Real(3.0))});
    const auto b = FormExpr::asVector({FormExpr::constant(Real(4.0)),
                                       FormExpr::constant(Real(5.0)),
                                       FormExpr::constant(Real(6.0))});
    const auto M = a.outer(b);
    const auto integrand = M.component(1, 2);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(12.0), 1e-12);
}

TEST(LLVMGenTensorOps, CrossProduct_ProducesCorrectVector)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto a = FormExpr::asVector({FormExpr::constant(Real(1.0)),
                                       FormExpr::constant(Real(0.0)),
                                       FormExpr::constant(Real(0.0))});
    const auto b = FormExpr::asVector({FormExpr::constant(Real(0.0)),
                                       FormExpr::constant(Real(1.0)),
                                       FormExpr::constant(Real(0.0))});
    const auto c = a.cross(b);
    const auto integrand = c.component(2);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(1.0), 1e-12);
}

TEST(LLVMGenTensorOps, InnerProduct_VectorVector_ProducesScalar)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto a = FormExpr::asVector({FormExpr::constant(Real(1.0)),
                                       FormExpr::constant(Real(2.0)),
                                       FormExpr::constant(Real(3.0))});
    const auto b = FormExpr::asVector({FormExpr::constant(Real(4.0)),
                                       FormExpr::constant(Real(5.0)),
                                       FormExpr::constant(Real(6.0))});
    const auto integrand = a.inner(b);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(32.0), 1e-12);
}

TEST(LLVMGenTensorOps, DoubleContraction_MatrixMatrix_ProducesScalar)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))},
        {FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0))},
    });
    const auto B = FormExpr::asTensor({
        {FormExpr::constant(Real(5.0)), FormExpr::constant(Real(6.0))},
        {FormExpr::constant(Real(7.0)), FormExpr::constant(Real(8.0))},
    });
    const auto integrand = A.doubleContraction(B);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(70.0), 1e-12);
}

TEST(LLVMGenTensorOps, Determinant3x3_ComputesCorrectly)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(2.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(3.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(4.0))},
    });
    const auto integrand = A.det();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(24.0), 1e-12);
}

TEST(LLVMGenTensorOps, Inverse3x3_ComputesCorrectly)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(2.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(3.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(4.0))},
    });
    const auto invA = A.inv();
    const auto integrand = invA.component(0, 0) + invA.component(1, 1) * Real(10.0) + invA.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(0.5) + Real(10.0) * (Real(1.0) / Real(3.0)) + Real(100.0) * Real(0.25);
    EXPECT_NEAR(jit / env.volume, expected, 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

