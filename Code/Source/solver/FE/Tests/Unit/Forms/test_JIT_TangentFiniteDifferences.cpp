/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_JIT_TangentFiniteDifferences.cpp
 * @brief JIT tangent verification against central finite differences of JIT residual assembly.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <array>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

void expectCellJitJacobianMatchesCentralFD(const assembly::IMeshAccess& mesh,
                                           const dofs::DofMap& dof_map,
                                           const spaces::FunctionSpace& space,
                                           const FormExpr& residual,
                                           const std::vector<Real>& U,
                                           Real eps,
                                           Real tol)
{
    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);

    auto fallback = std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(fallback, makeUnitTestJITOptions());
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n_dofs = dof_map.getNumDofs();
    assembly::DenseMatrixView J(n_dofs);
    assembly::DenseVectorView R(n_dofs);
    J.zero();
    R.zero();

    assembler.setCurrentSolution(U);
    (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J, R);

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, jit_kernel, Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(n_dofs);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, jit_kernel, Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

void expectBoundaryJitJacobianMatchesCentralFD(const assembly::IMeshAccess& mesh,
                                               int boundary_marker,
                                               const dofs::DofMap& dof_map,
                                               const spaces::FunctionSpace& space,
                                               const FormExpr& residual,
                                               const std::vector<Real>& U,
                                               Real eps,
                                               Real tol)
{
    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);

    auto fallback = std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(fallback, makeUnitTestJITOptions());
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n_dofs = dof_map.getNumDofs();
    assembly::DenseMatrixView J(n_dofs);
    assembly::DenseVectorView R(n_dofs);
    J.zero();
    R.zero();

    assembler.setCurrentSolution(U);
    (void)assembler.assembleBoundaryFaces(mesh, boundary_marker, space, jit_kernel, &J, &R);

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)assembler.assembleBoundaryFaces(mesh, boundary_marker, space, jit_kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(n_dofs);
        Rm.zero();
        (void)assembler.assembleBoundaryFaces(mesh, boundary_marker, space, jit_kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

void expectInteriorFaceJitJacobianMatchesCentralFD(const assembly::IMeshAccess& mesh,
                                                   const dofs::DofMap& dof_map,
                                                   const spaces::FunctionSpace& space,
                                                   const FormExpr& residual,
                                                   const std::vector<Real>& U,
                                                   Real eps,
                                                   Real tol)
{
    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);

    auto fallback = std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(fallback, makeUnitTestJITOptions());
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n_dofs = dof_map.getNumDofs();
    assembly::DenseMatrixView J(n_dofs);
    assembly::DenseVectorView R(n_dofs);
    J.zero();
    R.zero();

    assembler.setCurrentSolution(U);
    (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, J, &R);

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseMatrixView M_dummy_p(n_dofs);
        assembly::DenseVectorView Rp(n_dofs);
        M_dummy_p.zero();
        Rp.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, M_dummy_p, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseMatrixView M_dummy_m(n_dofs);
        assembly::DenseVectorView Rm(n_dofs);
        M_dummy_m.zero();
        Rm.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, M_dummy_m, &Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

FormExpr spd2x2FromGradU(const FormExpr& u)
{
    const auto I = FormExpr::identity(2);
    return (2.0 * I) + outer(grad(u), grad(u));
}

} // namespace

TEST(JITTangentFiniteDifferences, PoissonCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    const std::vector<Real> U = {0.13, -0.20, 0.07, 0.18};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(JITTangentFiniteDifferences, NonlinearDiffusionCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = inner((FormExpr::constant(Real(1.0)) + u * u) * grad(u), grad(v)).dx();

    const std::vector<Real> U = {0.11, -0.21, 0.17, -0.08};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(JITTangentFiniteDifferences, NonlinearReactionCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = (u * u * u * v).dx();

    const std::vector<Real> U = {0.09, -0.14, 0.19, -0.05};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-8);
}

TEST(JITTangentFiniteDifferences, NitscheBoundaryTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto n = FormExpr::normal();
    const auto gamma = FormExpr::constant(Real(25.0));
    const auto residual =
        (-inner(grad(u), n) * v - u * inner(grad(v), n) + (gamma / h()) * u * v).ds(2);

    const std::vector<Real> U = {0.10, -0.20, 0.30, -0.10};
    expectBoundaryJitJacobianMatchesCentralFD(mesh, /*boundary_marker=*/2, dof_map, space, residual, U,
                                              /*eps=*/1e-6, /*tol=*/1e-8);
}

TEST(JITTangentFiniteDifferences, DGPenaltyInteriorFaceTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto eta = FormExpr::constant(Real(10.0));
    const auto residual = (eta * inner(jump(u), jump(v))).dS();

    const std::vector<Real> U = {0.1, -0.2, 0.3, -0.1, -0.05, 0.07, -0.11, 0.13};
    expectInteriorFaceJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-8);
}

TEST(JITTangentFiniteDifferences, MatrixExpCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (A.matrixExp().trace() * v).dx();

    const std::vector<Real> U = {0.12, -0.08, 0.15};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(JITTangentFiniteDifferences, MatrixLogCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (A.matrixLog().trace() * v).dx();

    const std::vector<Real> U = {0.10, -0.12, 0.18};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/2e-5);
}

TEST(JITTangentFiniteDifferences, MatrixSqrtCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (A.matrixSqrt().trace() * v).dx();

    const std::vector<Real> U = {0.08, -0.05, 0.14};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/2e-5);
}

TEST(JITTangentFiniteDifferences, MatrixDetCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (det(A) * v).dx();

    const std::vector<Real> U = {0.09, -0.07, 0.16};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(JITTangentFiniteDifferences, MatrixInvCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (inv(A).trace() * v).dx();

    const std::vector<Real> U = {0.10, -0.09, 0.13};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/2e-6);
}

TEST(JITTangentFiniteDifferences, MatrixCofactorCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (cofactor(A).trace() * v).dx();

    const std::vector<Real> U = {0.07, -0.11, 0.15};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(JITTangentFiniteDifferences, SymmetricEigenvalueCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (A.symmetricEigenvalue(/*which=*/0) * v).dx();

    const std::vector<Real> U = {0.12, -0.04, 0.11};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/2e-5);
}

TEST(JITTangentFiniteDifferences, SmoothHeavisideCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto eps = FormExpr::constant(Real(0.1));
    const auto residual = (smoothHeaviside(u, eps) * v).dx();

    const std::vector<Real> U = {0.12, -0.08, 0.10, -0.05};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(JITTangentFiniteDifferences, SmoothMinMaxCellTangentMatchesCentralDifferences)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto c = FormExpr::constant(Real(0.05));
    const auto eps = FormExpr::constant(Real(0.1));

    // Exercise both smoothMin and smoothMax operators.
    const auto residual = ((smoothMin(u, c, eps) + smoothMax(u, c, eps)) * v).dx();

    const std::vector<Real> U = {0.04, 0.06, -0.02, 0.08};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(JITTangentFiniteDifferences, ConditionalCellTangentMatchesCentralDifferences_PositiveBranch)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = (conditional(gt(u, FormExpr::constant(Real(0.0))), u * u, -u) * v).dx();

    // Keep u > 0 everywhere so the branch is stable under +/- eps perturbations.
    const std::vector<Real> U = {0.40, 0.35, 0.28, 0.50};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(JITTangentFiniteDifferences, ConditionalCellTangentMatchesCentralDifferences_NegativeBranch)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = (conditional(gt(u, FormExpr::constant(Real(0.0))), u * u, -u) * v).dx();

    // Keep u < 0 everywhere so the branch is stable under +/- eps perturbations.
    const std::vector<Real> U = {-0.40, -0.35, -0.28, -0.50};
    expectCellJitJacobianMatchesCentralFD(mesh, dof_map, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
