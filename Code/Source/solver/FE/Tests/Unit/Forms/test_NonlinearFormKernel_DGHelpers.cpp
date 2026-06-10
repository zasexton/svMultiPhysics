/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearFormKernel_DGHelpers.cpp
 * @brief Central-difference Jacobian verification for DG vocabulary helper operators.
 */

#include <gtest/gtest.h>

#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/CutCellForms.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

void expectInteriorFaceJacobianMatchesCentralFD(const assembly::IMeshAccess& mesh,
                                               const dofs::DofMap& dof_map,
                                               const spaces::FunctionSpace& space,
                                               assembly::AssemblyKernel& kernel_both,
                                               assembly::AssemblyKernel& kernel_vec,
                                               const std::vector<Real>& U,
                                               Real eps,
                                               Real tol,
                                               int interior_facet_marker = -1,
                                               const assembly::CutIntegrationContext* cut_context = nullptr,
                                               const assembly::TimeIntegrationContext* time_context = nullptr,
                                               const std::vector<Real>* U_prev = nullptr,
                                               const std::vector<Real>* U_prev2 = nullptr)
{
    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);
    if (time_context != nullptr) {
        assembler.setTimeIntegrationContext(time_context);
    }
    if (U_prev != nullptr) {
        assembler.setPreviousSolution(*U_prev);
    }
    if (U_prev2 != nullptr) {
        assembler.setPreviousSolution2(*U_prev2);
    }
    if (cut_context != nullptr) {
        assembler.setCutIntegrationContext(cut_context);
    }

    const auto n = dof_map.getNumDofs();

    assembly::DenseMatrixView J(n);
    assembly::DenseVectorView R(n);
    J.zero();
    R.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel_both, J, &R,
                                                        interior_facet_marker);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    for (GlobalIndex j = 0; j < n; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseMatrixView M_dummy_p(n);
        assembly::DenseVectorView Rp(n);
        M_dummy_p.zero();
        Rp.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, kernel_vec, M_dummy_p, &Rp,
                                              interior_facet_marker);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseMatrixView M_dummy_m(n);
        assembly::DenseVectorView Rm(n);
        M_dummy_m.zero();
        Rm.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, kernel_vec, M_dummy_m, &Rm,
                                              interior_facet_marker);

        for (GlobalIndex i = 0; i < n; ++i) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

} // namespace

TEST(NonlinearFormKernelDGHelpersTest, UpwindValueJacobianMatchesCentralDifferences)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto beta = as_vector({FormExpr::constant(Real(1.0)),
                                 FormExpr::constant(Real(1.0)),
                                 FormExpr::constant(Real(1.0))});

    // Minimal upwind DG residual: ∫ u_upwind(u, beta) * [[v]] dS
    const auto residual = (upwindValue(u, beta) * jump(v)).dS();

    auto ir = compiler.compileResidual(residual);
    auto ir_vec = compiler.compileResidual(residual);
    NonlinearFormKernel kernel_both(std::move(ir), ADMode::Forward, NonlinearKernelOutput::Both);
    NonlinearFormKernel kernel_vec(std::move(ir_vec), ADMode::Forward, NonlinearKernelOutput::VectorOnly);

    std::vector<Real> U = {0.12, -0.05, 0.08, 0.02, -0.07, 0.2, 0.05, -0.15};
    expectInteriorFaceJacobianMatchesCentralFD(mesh, dof_map, space,
                                              kernel_both, kernel_vec,
                                              U, /*eps=*/1e-6, /*tol=*/5e-7);
}

TEST(NonlinearFormKernelDGHelpersTest, HarmonicAverageJacobianMatchesCentralDifferences)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    // Face coefficient from harmonic average of a positive nonlinear diffusion k(u)=1+u^2.
    const auto k = FormExpr::constant(Real(1.0)) + u * u;
    const auto residual = (harmonicAverage(k) * jump(v)).dS();

    auto ir = compiler.compileResidual(residual);
    auto ir_vec = compiler.compileResidual(residual);
    NonlinearFormKernel kernel_both(std::move(ir), ADMode::Forward, NonlinearKernelOutput::Both);
    NonlinearFormKernel kernel_vec(std::move(ir_vec), ADMode::Forward, NonlinearKernelOutput::VectorOnly);

    std::vector<Real> U = {0.12, -0.05, 0.08, 0.02, -0.07, 0.2, 0.05, -0.15};
    expectInteriorFaceJacobianMatchesCentralFD(mesh, dof_map, space,
                                              kernel_both, kernel_vec,
                                              U, /*eps=*/1e-6, /*tol=*/5e-7);
}

TEST(NonlinearFormKernelDGHelpersTest, CutAdjacentGhostPenaltyJacobianMatchesCentralDifferences)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto residual = cutAdjacentFacetIntegral(
        FormExpr::constant(Real{2.5}) *
        cutAdjacentFacetNormalGradientJump(u) *
            cutAdjacentFacetNormalGradientJump(v),
        /*facet_set_marker=*/12);

    auto ir = compiler.compileResidual(residual);
    auto ir_vec = compiler.compileResidual(residual);
    NonlinearFormKernel kernel_both(std::move(ir), ADMode::Forward, NonlinearKernelOutput::Both);
    NonlinearFormKernel kernel_vec(std::move(ir_vec), ADMode::Forward, NonlinearKernelOutput::VectorOnly);

    assembly::CutIntegrationContext cut_context;
    assembly::CutFacetSetHandle handle;
    handle.marker = 12;
    handle.name = "test-cut-adjacent-facets";
    handle.facets = {0};
    cut_context.addFacetSetHandle(std::move(handle));

    std::vector<Real> U = {0.12, -0.05, 0.08, 0.02, -0.07, 0.2, 0.05, -0.15};
    expectInteriorFaceJacobianMatchesCentralFD(mesh, dof_map, space,
                                              kernel_both, kernel_vec,
                                              U, /*eps=*/1e-6, /*tol=*/5e-7,
                                              /*interior_facet_marker=*/12,
                                              &cut_context);
}

TEST(NonlinearFormKernelDGHelpersTest, CutAdjacentEffectiveDtGradientJumpMatchesExplicitHistory)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    constexpr Real dt_step = 0.2;
    const Real a0 = Real(1.5) / dt_step;
    const Real a1 = Real(-2.0) / dt_step;
    const Real a2 = Real(0.5) / dt_step;
    const Real dt_eff = Real(1.0) / std::abs(a0);

    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "unit_bdf2_cut_adjacent";
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {a0, a1, a2};
    ti.dt1 = dt1;

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    constexpr int marker = 12;

    const auto incremental = FormExpr::effectiveTimeStep() * dt(u, 1);
    const auto explicit_increment =
        FormExpr::constant(dt_eff) *
        (FormExpr::constant(a0) * u +
         FormExpr::constant(a1) * FormExpr::previousSolution(1) +
         FormExpr::constant(a2) * FormExpr::previousSolution(2));
    const auto residual_dt = cutAdjacentFacetIntegral(
        inner(cutAdjacentFacetGradientJump(incremental),
              cutAdjacentFacetGradientJump(v)),
        marker);
    const auto residual_explicit = cutAdjacentFacetIntegral(
        inner(cutAdjacentFacetGradientJump(explicit_increment),
              cutAdjacentFacetGradientJump(v)),
        marker);

    auto ir_dt_both = compiler.compileResidual(residual_dt);
    auto ir_dt_vec = compiler.compileResidual(residual_dt);
    auto ir_ex = compiler.compileResidual(residual_explicit);
    NonlinearFormKernel kernel_dt_both(std::move(ir_dt_both), ADMode::Forward, NonlinearKernelOutput::Both);
    NonlinearFormKernel kernel_dt_vec(std::move(ir_dt_vec), ADMode::Forward, NonlinearKernelOutput::VectorOnly);
    NonlinearFormKernel kernel_ex(std::move(ir_ex), ADMode::Forward, NonlinearKernelOutput::Both);

    assembly::CutIntegrationContext cut_context;
    assembly::CutFacetSetHandle handle;
    handle.marker = marker;
    handle.name = "test-cut-adjacent-dt-facets";
    handle.facets = {0};
    cut_context.addFacetSetHandle(std::move(handle));

    std::vector<Real> U = {0.12, -0.05, 0.08, 0.02, -0.07, 0.2, 0.05, -0.15};
    std::vector<Real> U_prev = {0.09, -0.02, 0.05, -0.04, -0.01, 0.12, 0.02, -0.08};
    std::vector<Real> U_prev2 = {0.06, 0.01, -0.03, 0.07, 0.04, -0.1, 0.11, 0.03};

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);
    assembler.setPreviousSolution(U_prev);
    assembler.setPreviousSolution2(U_prev2);
    assembler.setTimeIntegrationContext(&ti);
    assembler.setCutIntegrationContext(&cut_context);

    const auto n = dof_map.getNumDofs();
    assembly::DenseMatrixView J_dt(n);
    assembly::DenseVectorView R_dt(n);
    assembly::DenseMatrixView J_ex(n);
    assembly::DenseVectorView R_ex(n);
    J_dt.zero();
    R_dt.zero();
    J_ex.zero();
    R_ex.zero();

    const auto result_dt =
        assembler.assembleInteriorFaces(mesh, space, space, kernel_dt_both, J_dt, &R_dt, marker);
    const auto result_ex =
        assembler.assembleInteriorFaces(mesh, space, space, kernel_ex, J_ex, &R_ex, marker);
    EXPECT_EQ(result_dt.interior_faces_assembled, 1);
    EXPECT_EQ(result_ex.interior_faces_assembled, 1);

    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_NEAR(R_dt.getVectorEntry(i), R_ex.getVectorEntry(i), 1.0e-12);
        for (GlobalIndex j = 0; j < n; ++j) {
            EXPECT_NEAR(J_dt.getMatrixEntry(i, j), J_ex.getMatrixEntry(i, j), 1.0e-12);
        }
    }

    expectInteriorFaceJacobianMatchesCentralFD(mesh, dof_map, space,
                                              kernel_dt_both, kernel_dt_vec,
                                              U, /*eps=*/1e-6, /*tol=*/5e-7,
                                              marker,
                                              &cut_context,
                                              &ti,
                                              &U_prev,
                                              &U_prev2);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
