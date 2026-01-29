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
        volume = assembler.assembleScalar(*makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell));
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

TEST(LLVMGenMatrixFunctionDD, MatrixExpDD2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 0.2;
    const Real b = 1.1;
    const Real da = -0.4;
    const Real db = 0.3;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db)},
    });

    const auto dE = FormExpr::matrixExpDirectionalDerivative(A, dA);
    const auto integrand = dE.component(0, 0) + dE.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(std::exp(a) * da) + Real(10.0) * Real(std::exp(b) * db);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctionDD, MatrixExpDD3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 0.2;
    const Real b = 1.1;
    const Real c = 2.2;
    const Real da = -0.4;
    const Real db = 0.3;
    const Real dc = 0.1;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(dc)},
    });

    const auto dE = FormExpr::matrixExpDirectionalDerivative(A, dA);
    const auto integrand =
        dE.component(0, 0) + dE.component(1, 1) * Real(10.0) + dE.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected =
        Real(std::exp(a) * da) + Real(10.0) * Real(std::exp(b) * db) + Real(100.0) * Real(std::exp(c) * dc);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctionDD, MatrixLogDD2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real da = 0.4;
    const Real db = -0.3;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db)},
    });

    const auto dL = FormExpr::matrixLogDirectionalDerivative(A, dA);
    const auto integrand = dL.component(0, 0) + dL.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(da / a) + Real(10.0) * Real(db / b);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctionDD, MatrixLogDD3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real c = 4.0;
    const Real da = 0.4;
    const Real db = -0.3;
    const Real dc = 0.1;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(dc)},
    });

    const auto dL = FormExpr::matrixLogDirectionalDerivative(A, dA);
    const auto integrand =
        dL.component(0, 0) + dL.component(1, 1) * Real(10.0) + dL.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(da / a) + Real(10.0) * Real(db / b) + Real(100.0) * Real(dc / c);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctionDD, MatrixSqrtDD2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real da = 0.4;
    const Real db = -0.3;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db)},
    });

    const auto dS = FormExpr::matrixSqrtDirectionalDerivative(A, dA);
    const auto integrand = dS.component(0, 0) + dS.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(da / (2.0 * std::sqrt(a))) + Real(10.0) * Real(db / (2.0 * std::sqrt(b)));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctionDD, MatrixSqrtDD3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real c = 4.0;
    const Real da = 0.4;
    const Real db = -0.3;
    const Real dc = 0.1;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(dc)},
    });

    const auto dS = FormExpr::matrixSqrtDirectionalDerivative(A, dA);
    const auto integrand =
        dS.component(0, 0) + dS.component(1, 1) * Real(10.0) + dS.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected =
        Real(da / (2.0 * std::sqrt(a))) + Real(10.0) * Real(db / (2.0 * std::sqrt(b))) +
        Real(100.0) * Real(dc / (2.0 * std::sqrt(c)));
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctionDD, MatrixPowDD2x2_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real da = 0.4;
    const Real db = -0.3;
    const Real p = 2.0;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db)},
    });

    const auto dP = FormExpr::matrixPowDirectionalDerivative(A, dA, FormExpr::constant(p));
    const auto integrand = dP.component(0, 0) + dP.component(1, 1) * Real(10.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected = Real(p * std::pow(a, p - 1.0) * da) + Real(10.0) * Real(p * std::pow(b, p - 1.0) * db);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

TEST(LLVMGenMatrixFunctionDD, MatrixPowDD3x3_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const Real a = 2.0;
    const Real b = 3.0;
    const Real c = 4.0;
    const Real da = 0.4;
    const Real db = -0.3;
    const Real dc = 0.1;
    const Real p = 2.0;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(a), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(b), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(c)},
    });
    const auto dA = FormExpr::asTensor({
        {FormExpr::constant(da), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(db), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(dc)},
    });

    const auto dP = FormExpr::matrixPowDirectionalDerivative(A, dA, FormExpr::constant(p));
    const auto integrand =
        dP.component(0, 0) + dP.component(1, 1) * Real(10.0) + dP.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real expected =
        Real(p * std::pow(a, p - 1.0) * da) + Real(10.0) * Real(p * std::pow(b, p - 1.0) * db) +
        Real(100.0) * Real(p * std::pow(c, p - 1.0) * dc);
    EXPECT_NEAR(jit / env.volume, expected, 1e-10);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

