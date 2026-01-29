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

TEST(LLVMGenScalarOps, PowerComputesExponentiation)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(2.0)).pow(FormExpr::constant(Real(3.0)));

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(8.0), 1e-12);
}

TEST(LLVMGenScalarOps, MinimumSelectsSmaller)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(-1.0)).min(FormExpr::constant(Real(2.0)));

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(-1.0), 1e-12);
}

TEST(LLVMGenScalarOps, MaximumSelectsLarger)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(-1.0)).max(FormExpr::constant(Real(2.0)));

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(2.0), 1e-12);
}

TEST(LLVMGenScalarOps, ConditionalSelectsCorrectBranch)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto cond = FormExpr::constant(Real(1.0)).lt(FormExpr::constant(Real(2.0)));
    const auto integrand = cond.conditional(FormExpr::constant(Real(3.0)), FormExpr::constant(Real(4.0)));

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(3.0), 1e-12);
}

TEST(LLVMGenScalarOps, ComparisonOperatorsReturnCorrectValues)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto lt = FormExpr::constant(Real(1.0)).lt(FormExpr::constant(Real(2.0)));
    const auto le = FormExpr::constant(Real(2.0)).le(FormExpr::constant(Real(2.0)));
    const auto gt = FormExpr::constant(Real(2.0)).gt(FormExpr::constant(Real(1.0)));
    const auto ge = FormExpr::constant(Real(2.0)).ge(FormExpr::constant(Real(2.0)));
    const auto eq = FormExpr::constant(Real(3.0)).eq(FormExpr::constant(Real(3.0)));
    const auto ne = FormExpr::constant(Real(3.0)).ne(FormExpr::constant(Real(4.0)));

    const auto false_lt = FormExpr::constant(Real(2.0)).lt(FormExpr::constant(Real(1.0)));
    const auto false_eq = FormExpr::constant(Real(1.0)).eq(FormExpr::constant(Real(2.0)));

    const auto integrand =
        lt + le + gt + ge + eq + ne +
        false_lt + false_eq;

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    // All true comparisons should contribute 1 and false ones 0.
    EXPECT_NEAR(jit / env.volume, Real(6.0), 1e-12);
}

TEST(LLVMGenScalarOps, AbsoluteValueMatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(-2.0)).abs();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(2.0), 1e-12);
}

TEST(LLVMGenScalarOps, SignMatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(-2.0)).sign();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(-1.0), 1e-12);
}

TEST(LLVMGenScalarOps, SqrtMatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(4.0)).sqrt();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(2.0), 1e-12);
}

TEST(LLVMGenScalarOps, ExpMatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(0.0)).exp();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(1.0), 1e-12);
}

TEST(LLVMGenScalarOps, LogMatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::constant(Real(1.0)).log();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(0.0), 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
