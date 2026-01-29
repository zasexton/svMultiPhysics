/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {
namespace {

struct TimeEnv {
    SingleTetraMeshAccess mesh{};
    dofs::DofMap dof_map{createSingleTetraDofMap()};
    spaces::H1Space space{ElementType::Tetra4, 1};
    assembly::TimeIntegrationContext ti{};
    std::vector<Real> u{};
    std::vector<Real> u_prev{};
    std::vector<Real> u_prev2{};
    std::vector<Real> history_weights{};

    assembly::FunctionalAssembler assembler{};
    Real volume{0.0};

    TimeEnv()
    {
        u.assign(4, Real(1.0));
        u_prev.assign(4, Real(2.0));
        u_prev2.assign(4, Real(3.0));

        assembler.setMesh(mesh);
        assembler.setDofMap(dof_map);
        assembler.setSpace(space);
        assembler.setPrimaryField(0);
        assembler.setSolution(u);
        assembler.setPreviousSolution(u_prev);
        assembler.setPreviousSolution2(u_prev2);
        assembler.setTimeStep(Real(0.2));
        assembler.setTimeIntegrationContext(&ti);

        const auto one = FormExpr::constant(Real(1.0));
        volume = assembler.assembleScalar(*makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell));
    }
};

Real assembleJIT(const FormExpr& integrand, TimeEnv& env)
{
    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());
    return env.assembler.assembleScalar(jit_kernel);
}

Real assembleInterp(const FormExpr& integrand, TimeEnv& env)
{
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);
    return env.assembler.assembleScalar(*interp);
}

} // namespace

TEST(LLVMGenTimeOps, EffectiveTimeStep_LoadsCorrectly)
{
    requireLLVMJITOrSkip();
    TimeEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {Real(10.0), Real(-10.0)}; // a0 = 1/dt => dt_eff = 0.1
    env.ti.dt1 = dt1;

    // env.assembler.dt is set to 0.2; EffectiveTimeStep should use dt1.a0 instead.
    const auto integrand = FormExpr::effectiveTimeStep();

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(0.1), 1e-12);
}

TEST(LLVMGenTimeOps, TimeDerivativeOrder2_UsesCorrectStencil)
{
    requireLLVMJITOrSkip();
    TimeEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    assembly::TimeDerivativeStencil dt2;
    dt2.order = 2;
    dt2.a = {Real(10.0), Real(20.0), Real(30.0)};
    env.ti.dt2 = dt2;

    const auto integrand = FormExpr::stateField(INVALID_FIELD_ID, env.space, "u").dt(2);

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(140.0), 1e-12); // 10*1 + 20*2 + 30*3
}

TEST(LLVMGenTimeOps, HistoryWeightedSum_AppliesWeightsCorrectly)
{
    requireLLVMJITOrSkip();
    TimeEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    env.history_weights = {Real(0.25), Real(0.75)};
    env.assembler.setHistoryWeights(env.history_weights);

    const auto integrand = FormExpr::historyWeightedSum(std::vector<FormExpr>{});

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(2.75), 1e-12); // 0.25*2 + 0.75*3
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

