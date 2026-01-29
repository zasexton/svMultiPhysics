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

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {
namespace {

struct ScalarFieldEnv {
    static constexpr FieldId field = 0;

    SingleTetraMeshAccess mesh{};
    dofs::DofMap dof_map{};
    spaces::H1Space space{ElementType::Tetra4, 1};
    std::vector<Real> u{};
    std::vector<Real> u_prev{};

    assembly::FunctionalAssembler assembler{};
    Real volume{0.0};

    ScalarFieldEnv()
        : dof_map(/*n_cells=*/1, /*n_dofs_total=*/10, /*dofs_per_cell=*/4)
    {
        const std::array<GlobalIndex, 4> cell_dofs = {5, 6, 7, 8};
        dof_map.setCellDofs(0, cell_dofs);
        dof_map.setNumDofs(10);
        dof_map.setNumLocalDofs(10);
        dof_map.finalize();

        u.assign(10, Real(123.0));
        u_prev.assign(10, Real(456.0));

        // Current solution u_h is constant 1 on the element.
        for (GlobalIndex dof : cell_dofs) {
            u[static_cast<std::size_t>(dof)] = Real(1.0);
            u_prev[static_cast<std::size_t>(dof)] = Real(2.0);
        }

        assembler.setMesh(mesh);
        assembler.setDofMap(dof_map);
        assembler.setSpace(space);
        assembler.setPrimaryField(field);
        assembler.setSolution(u);
        assembler.setPreviousSolution(u_prev);

        const auto one = FormExpr::constant(Real(1.0));
        auto one_kernel = makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell);
        volume = assembler.assembleScalar(*one_kernel);
    }
};

Real assembleJIT(const FormExpr& integrand, ScalarFieldEnv& env)
{
    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());
    return env.assembler.assembleScalar(jit_kernel);
}

Real assembleInterp(const FormExpr& integrand, ScalarFieldEnv& env)
{
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);
    return env.assembler.assembleScalar(*interp);
}

} // namespace

TEST(LLVMGenFieldOps, DiscreteField_ComputesSolutionAtQuadPoint)
{
    requireLLVMJITOrSkip();
    ScalarFieldEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::discreteField(ScalarFieldEnv::field, env.space, "u");

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(1.0), 1e-12);
}

TEST(LLVMGenFieldOps, StateField_LoadsFromFieldSolutionTable)
{
    requireLLVMJITOrSkip();
    ScalarFieldEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::stateField(ScalarFieldEnv::field, env.space, "u_state");

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(1.0), 1e-12);
}

TEST(LLVMGenFieldOps, PreviousSolutionRef_LoadsHistorySlot)
{
    requireLLVMJITOrSkip();
    ScalarFieldEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    // PreviousSolutionRef requires a DiscreteField/StateField operand to infer scalar vs vector shape.
    const auto anchor = FormExpr::discreteField(ScalarFieldEnv::field, env.space, "u") * Real(0.0);
    const auto integrand = FormExpr::previousSolution(/*steps_back=*/1) + anchor;

    const Real ref = assembleInterp(integrand, env);
    const Real jit = assembleJIT(integrand, env);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(2.0), 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

