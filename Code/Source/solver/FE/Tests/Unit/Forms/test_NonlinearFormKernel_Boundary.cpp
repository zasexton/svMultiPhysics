/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearFormKernel_Boundary.cpp
 * @brief Unit tests for FE/Forms nonlinear boundary (ds) residual + AD Jacobian assembly
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(NonlinearFormKernelBoundaryTest, JacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // Nonlinear boundary residual: ∫ u^3 v ds
    const auto residual = (u * u * u * v).ds(2);
    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    const auto result = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    std::array<Real, 4> R0{};
    for (GlobalIndex i = 0; i < 4; ++i) {
        R0[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    // Central-difference check: J(:,j) ~= (R(U+eps e_j) - R(U-eps e_j)) / (2 eps)
    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 1e-7);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, NitscheWeakDirichletJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto n = FormExpr::normal();
    const auto gamma = FormExpr::constant(Real(25.0));

    // Symmetric Nitsche-style boundary terms (physics-agnostic verification of normal() + h()).
    const auto residual =
        (-inner(grad(u), n) * v - u * inner(grad(v), n) + (gamma / h()) * u * v).ds(2);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    const auto result = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 1e-9);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, NonlinearRobinJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto n = FormExpr::normal();
    const auto beta = FormExpr::constant(Real(0.2));
    const auto alpha = (FormExpr::constant(Real(1.0)) + u * u);

    // Nonlinear Robin: ∫ (alpha(u)*u + beta*∂u/∂n) v ds
    const auto residual = ((alpha * u + beta * inner(grad(u), n)) * v).ds(2);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    const auto result = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-7);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, SurfaceGradientJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto n = FormExpr::normal();

    // Surface diffusion-like term on the boundary:
    //   ∫ inner(∇_s(u^2), ∇_s(v)) ds
    const auto residual = inner(surfaceGradient(u * u, n), surfaceGradient(v, n)).ds(2);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    const auto result = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-7);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, SurfaceLaplacianJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto n = FormExpr::normal();

    // Boundary surface-Laplacian term:
    //   ∫ (Δ_s(u^2)) v ds
    const auto residual = (surfaceLaplacian(u * u, n) * v).ds(2);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    const auto result = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, CellPlusBoundaryJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto gamma = FormExpr::constant(Real(25.0));

    const auto residual_cell = inner(grad(u), grad(v)).dx();
    const auto residual_boundary = ((gamma / h()) * u * v).ds(2);

    auto ir_cell = compiler.compileResidual(residual_cell);
    auto ir_boundary = compiler.compileResidual(residual_boundary);
    NonlinearFormKernel kernel_cell(std::move(ir_cell), ADMode::Forward);
    NonlinearFormKernel kernel_boundary(std::move(ir_boundary), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel_cell, J, R);
    const auto bnd = assembler.assembleBoundaryFaces(mesh, 2, space, kernel_boundary, &J, &R);
    EXPECT_EQ(bnd.boundary_faces_assembled, 1);

    // Central-difference check for combined (dx + ds) residual:
    // J(:,j) ~= (R(U+eps e_j) - R(U-eps e_j)) / (2 eps)
    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel_cell, Rp);
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel_boundary, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, kernel_cell, Rm);
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel_boundary, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 1e-9);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, CellPlusBoundarySingleKernelJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto gamma = FormExpr::constant(Real(25.0));

    const auto residual = inner(grad(u), grad(v)).dx() + ((gamma / h()) * u * v).ds(2);
    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);
    const auto bnd = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(bnd.boundary_faces_assembled, 1);

    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rm);
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 1e-9);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, CellPlusBoundaryNonlinearSingleKernelJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto alpha = FormExpr::constant(Real(1.0)) + u * u;

    const auto residual =
        inner(alpha * grad(u), grad(v)).dx() +
        ((alpha * u) * v).ds(2);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);
    const auto bnd = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(bnd.boundary_faces_assembled, 1);

    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(4);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rm);
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &Rm);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-7);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
