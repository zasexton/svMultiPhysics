/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <functional>
#include <optional>

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

} // namespace

TEST(JITCompilerErrorHandling, InvalidFormExprType_ReturnsErrorNotCrash)
{
    requireLLVMJITOrSkip();

    auto compiler = jit::JITCompiler::getOrCreate(makeUnitTestJITOptions());
    ASSERT_TRUE(compiler != nullptr);

    jit::ValidationOptions vopt;
    vopt.strictness = jit::Strictness::AllowExternalCalls;

    // Measure nodes (dx/ds/...) are not valid as integrands for kernel lowering.
    const auto bad = FormExpr::constant(Real(1.0)).dx();

    const auto r = compiler->compileFunctional(bad, IntegralDomain::Cell, vopt);
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.message.empty());
}

TEST(JITFunctionalKernelWrapper, CompilationFailure_FallsBackToInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    std::function<std::optional<Real>(std::string_view)> get_real_param =
        [](std::string_view key) -> std::optional<Real> {
        if (key == "a") return Real(2.5);
        return std::nullopt;
    };
    env.assembler.setRealParameterGetter(&get_real_param);

    const auto integrand = FormExpr::parameter("a");

    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);
    const Real ref = env.assembler.assembleScalar(*interp);

    jit::JITFunctionalKernelWrapper jit_kernel(
        interp, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());
    const Real jit = env.assembler.assembleScalar(jit_kernel);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / env.volume, Real(2.5), 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
