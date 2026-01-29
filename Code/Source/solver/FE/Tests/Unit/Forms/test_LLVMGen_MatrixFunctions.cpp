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

TEST(LLVMGenMatrixFunctions, MatrixExp2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 0.0;
    const Real b = 1.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto E = A.matrixExp();
    const auto integrand = E.component(0, 0) + E.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(std::exp(a)) + Real(10.0) * Real(std::exp(b));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctions, MatrixExp3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 0.0;
    const Real b = 1.0;
    const Real c = 2.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto E = A.matrixExp();
    const auto integrand = E.component(0, 0) + E.component(1, 1) * Real(10.0) + E.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected =
        Real(std::exp(a)) + Real(10.0) * Real(std::exp(b)) + Real(100.0) * Real(std::exp(c));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctions, MatrixLog2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto L = A.matrixLog();
    const auto integrand = L.component(0, 0) + L.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(std::log(a)) + Real(10.0) * Real(std::log(b));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctions, MatrixLog3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real c = 4.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto L = A.matrixLog();
    const auto integrand = L.component(0, 0) + L.component(1, 1) * Real(10.0) + L.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected =
        Real(std::log(a)) + Real(10.0) * Real(std::log(b)) + Real(100.0) * Real(std::log(c));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctions, MatrixSqrt2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto S = A.matrixSqrt();
    const auto integrand = S.component(0, 0) + S.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(std::sqrt(a)) + Real(10.0) * Real(std::sqrt(b));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctions, MatrixSqrt3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real c = 4.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto S = A.matrixSqrt();
    const auto integrand = S.component(0, 0) + S.component(1, 1) * Real(10.0) + S.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected =
        Real(std::sqrt(a)) + Real(10.0) * Real(std::sqrt(b)) + Real(100.0) * Real(std::sqrt(c));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctions, MatrixPow2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto P = A.matrixPow(FormExpr::constant(Real(2.0)));
    const auto integrand = P.component(0, 0) + P.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(a * a) + Real(10.0) * Real(b * b);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctions, MatrixPow3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real c = 4.0;
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto P = A.matrixPow(FormExpr::constant(Real(2.0)));
    const auto integrand = P.component(0, 0) + P.component(1, 1) * Real(10.0) + P.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected =
        Real(a * a) + Real(10.0) * Real(b * b) + Real(100.0) * Real(c * c);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

