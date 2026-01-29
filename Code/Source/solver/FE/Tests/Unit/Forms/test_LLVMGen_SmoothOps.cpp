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

TEST(LLVMGenSmoothOps, SmoothAbsoluteValueApproximatesAbs)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto x = FormExpr::constant(Real(-2.0));
    const auto eps = FormExpr::constant(Real(1e-2));
    const auto integrand = x.smoothAbs(eps);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real v = jit / env.volume;
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GE(v, Real(2.0));
}

TEST(LLVMGenSmoothOps, SmoothSignApproximatesSign)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto eps = FormExpr::constant(Real(1e-2));

    const auto pos = FormExpr::constant(Real(2.0)).smoothSign(eps);
    const auto neg = FormExpr::constant(Real(-2.0)).smoothSign(eps);

    const Real pos_ref = assembleInterp(pos, env);
    const Real pos_jit = assembleJIT(pos, env);
    EXPECT_NEAR(pos_jit, pos_ref, 1e-12);

    const Real neg_ref = assembleInterp(neg, env);
    const Real neg_jit = assembleJIT(neg, env);
    EXPECT_NEAR(neg_jit, neg_ref, 1e-12);

    EXPECT_GT(pos_jit / env.volume, Real(0.5));
    EXPECT_LT(neg_jit / env.volume, Real(-0.5));
}

TEST(LLVMGenSmoothOps, SmoothHeavisideApproximatesStep)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto eps = FormExpr::constant(Real(1e-2));

    const auto pos = FormExpr::constant(Real(2.0)).smoothHeaviside(eps);
    const auto neg = FormExpr::constant(Real(-2.0)).smoothHeaviside(eps);

    const Real pos_ref = assembleInterp(pos, env);
    const Real pos_jit = assembleJIT(pos, env);
    EXPECT_NEAR(pos_jit, pos_ref, 1e-12);

    const Real neg_ref = assembleInterp(neg, env);
    const Real neg_jit = assembleJIT(neg, env);
    EXPECT_NEAR(neg_jit, neg_ref, 1e-12);

    EXPECT_GT(pos_jit / env.volume, Real(0.5));
    EXPECT_LT(neg_jit / env.volume, Real(0.5));
}

TEST(LLVMGenSmoothOps, SmoothMinApproximatesMinWithEpsilon)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto a = FormExpr::constant(Real(1.0));
    const auto b = FormExpr::constant(Real(2.0));
    const auto eps = FormExpr::constant(Real(1e-2));
    const auto integrand = a.smoothMin(b, eps);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real v = jit / env.volume;
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_NEAR(v, Real(1.0), 1e-2);
}

TEST(LLVMGenSmoothOps, SmoothMaxApproximatesMaxWithEpsilon)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto a = FormExpr::constant(Real(1.0));
    const auto b = FormExpr::constant(Real(2.0));
    const auto eps = FormExpr::constant(Real(1e-2));
    const auto integrand = a.smoothMax(b, eps);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    const Real v = jit / env.volume;
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_NEAR(v, Real(2.0), 1e-2);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

