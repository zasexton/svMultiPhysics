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

struct BoundaryEnv {
    static constexpr int marker = 2;

    SingleTetraOneBoundaryFaceMeshAccess mesh{marker};
    dofs::DofMap dof_map{createSingleTetraDofMap()};
    spaces::H1Space space{ElementType::Tetra4, 1};
    assembly::FunctionalAssembler assembler{};
    Real area{0.0};

    BoundaryEnv()
    {
        assembler.setMesh(mesh);
        assembler.setDofMap(dof_map);
        assembler.setSpace(space);

        const auto one = FormExpr::constant(Real(1.0));
        area = assembler.assembleBoundaryScalar(*makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::BoundaryFace), marker);
    }
};

Real assembleJITCell(const FormExpr& integrand, CellEnv& env)
{
    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());
    return env.assembler.assembleScalar(jit_kernel);
}

Real assembleInterpCell(const FormExpr& integrand, CellEnv& env)
{
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);
    return env.assembler.assembleScalar(*interp);
}

Real assembleJITBoundary(const FormExpr& integrand, BoundaryEnv& env)
{
    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/false, /*has_boundary=*/true);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::BoundaryFace, makeUnitTestJITOptions());
    return env.assembler.assembleBoundaryScalar(jit_kernel, BoundaryEnv::marker);
}

Real assembleInterpBoundary(const FormExpr& integrand, BoundaryEnv& env)
{
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::BoundaryFace);
    return env.assembler.assembleBoundaryScalar(*interp, BoundaryEnv::marker);
}

} // namespace

TEST(LLVMGenGeometryOps, JacobianDeterminant_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto integrand = FormExpr::jacobianDeterminant();

    const Real ref = assembleInterpCell(integrand, env);
    const Real jit = assembleJITCell(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    // For the unit tetra, |det(J)| = 1 everywhere.
    EXPECT_NEAR(jit / env.volume, Real(1.0), 1e-12);
}

TEST(LLVMGenGeometryOps, Identity_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto I = FormExpr::identity();
    const auto integrand = I.component(0, 0) + I.component(1, 1) * Real(10.0) + I.component(2, 2) * Real(100.0);

    const Real ref = assembleInterpCell(integrand, env);
    const Real jit = assembleJITCell(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    EXPECT_NEAR(jit / env.volume, Real(111.0), 1e-12);
}

TEST(LLVMGenGeometryOps, Jacobian_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto J = FormExpr::jacobian();
    const auto integrand = J.component(0, 0) + J.component(1, 1) * Real(10.0) + J.component(2, 2) * Real(100.0);

    const Real ref = assembleInterpCell(integrand, env);
    const Real jit = assembleJITCell(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    EXPECT_NEAR(jit / env.volume, Real(111.0), 1e-12);
}

TEST(LLVMGenGeometryOps, JacobianInverse_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    CellEnv env;
    ASSERT_GT(env.volume, Real(0.0));

    const auto Jinv = FormExpr::jacobianInverse();
    const auto integrand =
        Jinv.component(0, 0) + Jinv.component(1, 1) * Real(10.0) + Jinv.component(2, 2) * Real(100.0);

    const Real ref = assembleInterpCell(integrand, env);
    const Real jit = assembleJITCell(integrand, env);
    EXPECT_NEAR(jit, ref, 1e-12);

    EXPECT_NEAR(jit / env.volume, Real(111.0), 1e-12);
}

TEST(LLVMGenGeometryOps, Normal_MatchesInterpreter)
{
    requireLLVMJITOrSkip();
    BoundaryEnv env;
    ASSERT_GT(env.area, Real(0.0));

    const auto n = FormExpr::normal();
    const auto nx = n.component(0);
    const auto ny = n.component(1);
    const auto nz = n.component(2);

    const Real ref_x = assembleInterpBoundary(nx, env);
    const Real jit_x = assembleJITBoundary(nx, env);
    EXPECT_NEAR(jit_x, ref_x, 1e-12);

    const Real ref_y = assembleInterpBoundary(ny, env);
    const Real jit_y = assembleJITBoundary(ny, env);
    EXPECT_NEAR(jit_y, ref_y, 1e-12);

    const Real ref_z = assembleInterpBoundary(nz, env);
    const Real jit_z = assembleJITBoundary(nz, env);
    EXPECT_NEAR(jit_z, ref_z, 1e-12);

    const Real nx_avg = jit_x / env.area;
    const Real ny_avg = jit_y / env.area;
    const Real nz_avg = jit_z / env.area;

    // The {0,1,2} face lies in z=0, so the normal is aligned with ±z.
    EXPECT_NEAR(nx_avg, Real(0.0), 1e-12);
    EXPECT_NEAR(ny_avg, Real(0.0), 1e-12);
    EXPECT_NEAR(std::abs(nz_avg), Real(1.0), 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
