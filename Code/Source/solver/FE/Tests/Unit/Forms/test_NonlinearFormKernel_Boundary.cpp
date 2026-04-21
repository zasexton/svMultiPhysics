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
#include "Forms/BoundaryConditions.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] std::array<Real, 4> solveLinear4x4(const assembly::DenseMatrixView& A,
                                                 const assembly::DenseVectorView& b)
{
    std::array<std::array<Real, 5>, 4> aug{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            aug[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                A.getMatrixEntry(static_cast<GlobalIndex>(i), static_cast<GlobalIndex>(j));
        }
        aug[static_cast<std::size_t>(i)][4] = b.getVectorEntry(static_cast<GlobalIndex>(i));
    }

    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        Real pivot_abs = std::abs(aug[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(col)]);
        for (int row = col + 1; row < 4; ++row) {
            const Real cand_abs = std::abs(aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
            if (cand_abs > pivot_abs) {
                pivot = row;
                pivot_abs = cand_abs;
            }
        }
        EXPECT_GT(pivot_abs, 1e-14);
        if (pivot != col) {
            std::swap(aug[static_cast<std::size_t>(pivot)], aug[static_cast<std::size_t>(col)]);
        }

        const Real diag = aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
        for (int j = col; j < 5; ++j) {
            aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] /= diag;
        }
        for (int row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const Real factor = aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (factor == Real(0)) {
                continue;
            }
            for (int j = col; j < 5; ++j) {
                aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                    factor * aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
            }
        }
    }

    std::array<Real, 4> x{};
    for (int i = 0; i < 4; ++i) {
        x[static_cast<std::size_t>(i)] = aug[static_cast<std::size_t>(i)][4];
    }
    return x;
}

[[nodiscard]] Real vectorNorm4(const assembly::DenseVectorView& v)
{
    Real sum = 0.0;
    for (GlobalIndex i = 0; i < 4; ++i) {
        const Real value = v.getVectorEntry(i);
        sum += value * value;
    }
    return std::sqrt(sum);
}

void assembleCellAndBoundary(const SingleTetraOneBoundaryFaceMeshAccess& mesh,
                             spaces::H1Space& space,
                             assembly::StandardAssembler& assembler,
                             assembly::AssemblyKernel& kernel,
                             assembly::DenseMatrixView* J,
                             assembly::DenseVectorView* R)
{
    if (J) {
        J->zero();
    }
    if (R) {
        R->zero();
    }
    if (J && R) {
        (void)assembler.assembleBoth(mesh, space, kernel, *J, *R);
    } else if (J) {
        (void)assembler.assembleMatrix(mesh, space, kernel, *J);
    } else if (R) {
        (void)assembler.assembleVector(mesh, space, kernel, *R);
    }
    (void)assembler.assembleBoundaryFaces(mesh, /*boundary_marker=*/2, space, kernel, J, R);
}

} // namespace

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

TEST(NonlinearFormKernelBoundaryTest, TraceNitscheBoundaryJacobianMatchesCentralDifferences)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto n = FormExpr::normal();

    auto residual = (u * v).dx();
    residual = bc::applyTraceNitsche(std::move(residual),
                                     u,
                                     v,
                                     /*boundary_marker=*/2,
                                     FormExpr::constant(0.0),
                                     inner(grad(u), n),
                                     inner(grad(v), n),
                                     FormExpr::constant(1.0) / h(),
                                     bc::ScalarTraceOperator::Identity);

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

TEST(NonlinearFormKernelBoundaryTest, TraceInequalityBoundaryJacobianMatchesCentralDifferencesWhenActive)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    auto residual = (FormExpr::constant(0.0) * u * v).dx();
    bc::TraceInequalityOptions opts;
    opts.trace_operator = bc::ScalarTraceOperator::Identity;
    opts.sense = bc::TraceInequalitySense::LessEqual;
    residual = bc::applyTraceInequality(std::move(residual),
                                        u,
                                        v,
                                        /*boundary_marker=*/2,
                                        FormExpr::constant(0.0),
                                        FormExpr::constant(4.0),
                                        opts);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.2, 0.3, 0.4, 0.25};
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
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 1e-8);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, TraceInequalityBoundaryResidualAndJacobianVanishWhenInactive)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    auto residual = (FormExpr::constant(0.0) * u * v).dx();
    bc::TraceInequalityOptions opts;
    opts.trace_operator = bc::ScalarTraceOperator::Identity;
    opts.sense = bc::TraceInequalitySense::LessEqual;
    residual = bc::applyTraceInequality(std::move(residual),
                                        u,
                                        v,
                                        /*boundary_marker=*/2,
                                        FormExpr::constant(0.0),
                                        FormExpr::constant(4.0),
                                        opts);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {-0.2, -0.3, -0.4, -0.25};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    const auto result = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, &J, &R);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(R.getVectorEntry(i), 0.0, 1e-12);
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(J.getMatrixEntry(i, j), 0.0, 1e-12);
        }
    }
}

TEST(NonlinearFormKernelBoundaryTest, TraceInequalityBoundarySemiSmoothNewtonSwitchesActiveSet)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    auto residual = ((u - FormExpr::constant(-0.1)) * v).dx();
    bc::TraceInequalityOptions opts;
    opts.trace_operator = bc::ScalarTraceOperator::Identity;
    opts.sense = bc::TraceInequalitySense::LessEqual;
    residual = bc::applyTraceInequality(std::move(residual),
                                        u,
                                        v,
                                        /*boundary_marker=*/2,
                                        FormExpr::constant(0.0),
                                        FormExpr::constant(40.0),
                                        opts);

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U0 = {0.3, 0.3, 0.3, 0.3};
    assembler.setCurrentSolution(U0);

    assembly::DenseMatrixView J0(4);
    assembly::DenseVectorView R0(4);
    assembleCellAndBoundary(mesh, space, assembler, kernel, &J0, &R0);

    const Real r0_norm = vectorNorm4(R0);
    EXPECT_GT(r0_norm, 1e-6);

    const auto du0 = solveLinear4x4(J0, R0);
    std::vector<Real> U1 = U0;
    for (std::size_t i = 0; i < U1.size(); ++i) {
        U1[i] -= du0[i];
    }

    for (const Real value : U1) {
        EXPECT_LT(value, 0.0);
    }

    assembler.setCurrentSolution(U1);
    assembly::DenseMatrixView J1(4);
    assembly::DenseVectorView R1(4);
    assembleCellAndBoundary(mesh, space, assembler, kernel, &J1, &R1);

    const Real r1_norm = vectorNorm4(R1);
    EXPECT_GT(r1_norm, 1e-10);
    EXPECT_LT(r1_norm, r0_norm);

    const auto du1 = solveLinear4x4(J1, R1);
    std::vector<Real> U2 = U1;
    for (std::size_t i = 0; i < U2.size(); ++i) {
        U2[i] -= du1[i];
    }

    assembler.setCurrentSolution(U2);
    assembly::DenseMatrixView J2(4);
    assembly::DenseVectorView R2(4);
    assembleCellAndBoundary(mesh, space, assembler, kernel, &J2, &R2);

    const Real r2_norm = vectorNorm4(R2);
    EXPECT_LT(r2_norm, 1e-12);
    for (const Real value : U2) {
        EXPECT_NEAR(value, -0.1, 1e-12);
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
