/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearFormKernel_TimeAndHistory.cpp
 * @brief Numerical verification of transient and history operators: dt(·,2), BDF2 dt(·), historyConvolution().
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

void expectJacobianMatchesCentralFD(const assembly::IMeshAccess& mesh,
                                   const spaces::FunctionSpace& space,
                                   assembly::StandardAssembler& assembler,
                                   assembly::AssemblyKernel& kernel,
                                   std::span<const Real> U,
                                   Real eps,
                                   Real tol)
{
    const auto n_dofs = static_cast<GlobalIndex>(U.size());

    assembly::DenseMatrixView J(n_dofs);
    assembly::DenseVectorView R0(n_dofs);
    J.zero();
    R0.zero();

    assembler.setCurrentSolution(U);
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R0);

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = std::vector<Real>(U.begin(), U.end());
        auto U_minus = std::vector<Real>(U.begin(), U.end());
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(n_dofs);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

void expectResidualAndJacobianMatch(const assembly::IMeshAccess& mesh,
                                   const spaces::FunctionSpace& space,
                                   assembly::StandardAssembler& assembler,
                                   assembly::AssemblyKernel& kernel_a,
                                   assembly::AssemblyKernel& kernel_b,
                                   GlobalIndex n_dofs,
                                   Real tol)
{
    assembly::DenseMatrixView Ja(n_dofs);
    assembly::DenseVectorView Ra(n_dofs);
    assembly::DenseMatrixView Jb(n_dofs);
    assembly::DenseVectorView Rb(n_dofs);
    Ja.zero();
    Ra.zero();
    Jb.zero();
    Rb.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel_a, Ja, Ra);
    (void)assembler.assembleBoth(mesh, space, space, kernel_b, Jb, Rb);

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        EXPECT_NEAR(Ra.getVectorEntry(i), Rb.getVectorEntry(i), tol);
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            EXPECT_NEAR(Ja.getMatrixEntry(i, j), Jb.getMatrixEntry(i, j), tol);
        }
    }
}

} // namespace

TEST(TimeDerivativeOpsTest, Dt2MatchesExplicitHistoryCombinationAndJacobianMatchesCentralFD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    constexpr Real dt_step = 0.25;
    const Real a0 = Real(1.0) / (dt_step * dt_step);
    const Real a1 = Real(-2.0) / (dt_step * dt_step);
    const Real a2 = Real(1.0) / (dt_step * dt_step);

    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "unit_dt2";
    assembly::TimeDerivativeStencil dt2;
    dt2.order = 2;
    dt2.a = {a0, a1, a2};
    ti.dt2 = dt2;

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto dt_form = (dt(u, 2) * v).dx();
    const auto explicit_form =
        ((FormExpr::constant(a0) * u +
          FormExpr::constant(a1) * FormExpr::previousSolution(1) +
          FormExpr::constant(a2) * FormExpr::previousSolution(2)) *
         v)
            .dx();

    auto ir_dt = compiler.compileResidual(dt_form);
    auto ir_ex = compiler.compileResidual(explicit_form);

    NonlinearFormKernel kernel_dt(std::move(ir_dt), ADMode::Forward);
    NonlinearFormKernel kernel_ex(std::move(ir_ex), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setTimeIntegrationContext(&ti);

    std::vector<Real> U = {0.12, -0.05, 0.08, -0.02};
    std::vector<Real> U_prev = {0.07, -0.01, 0.02, 0.05};
    std::vector<Real> U_prev2 = {0.09, 0.03, -0.04, 0.01};
    assembler.setCurrentSolution(U);
    assembler.setPreviousSolution(U_prev);
    assembler.setPreviousSolution2(U_prev2);

    expectResidualAndJacobianMatch(mesh, space, assembler, kernel_dt, kernel_ex, /*n_dofs=*/4, /*tol=*/1e-12);

    expectJacobianMatchesCentralFD(mesh, space, assembler, kernel_dt, U, /*eps=*/1e-7, /*tol=*/1e-10);
}

TEST(TimeDerivativeOpsTest, BDF2Dt1MatchesExplicitHistoryCombinationAndJacobianMatchesCentralFD)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    constexpr Real dt_step = 0.2;
    // BDF2: (3 u^n - 4 u^{n-1} + u^{n-2}) / (2 dt)
    const Real a0 = Real(1.5) / dt_step;
    const Real a1 = Real(-2.0) / dt_step;
    const Real a2 = Real(0.5) / dt_step;

    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "unit_bdf2";
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {a0, a1, a2};
    ti.dt1 = dt1;

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto dt_form = (dt(u, 1) * v).dx();
    const auto explicit_form =
        ((FormExpr::constant(a0) * u +
          FormExpr::constant(a1) * FormExpr::previousSolution(1) +
          FormExpr::constant(a2) * FormExpr::previousSolution(2)) *
         v)
            .dx();

    auto ir_dt = compiler.compileResidual(dt_form);
    auto ir_ex = compiler.compileResidual(explicit_form);

    NonlinearFormKernel kernel_dt(std::move(ir_dt), ADMode::Forward);
    NonlinearFormKernel kernel_ex(std::move(ir_ex), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setTimeIntegrationContext(&ti);

    std::vector<Real> U = {0.03, 0.11, -0.07, 0.05};
    std::vector<Real> U_prev = {-0.02, 0.06, 0.09, -0.01};
    std::vector<Real> U_prev2 = {0.04, -0.05, 0.01, 0.08};
    assembler.setCurrentSolution(U);
    assembler.setPreviousSolution(U_prev);
    assembler.setPreviousSolution2(U_prev2);

    expectResidualAndJacobianMatch(mesh, space, assembler, kernel_dt, kernel_ex, /*n_dofs=*/4, /*tol=*/1e-12);

    expectJacobianMatchesCentralFD(mesh, space, assembler, kernel_dt, U, /*eps=*/1e-7, /*tol=*/1e-10);
}

TEST(HistoryOpsTest, HistoryConvolutionMatchesExplicitSumAndHasNoJacobianContribution)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    constexpr Real w1 = 0.25;
    constexpr Real w2 = -1.75;

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto hist = FormExpr::historyConvolution({FormExpr::constant(w1), FormExpr::constant(w2)});
    const auto hist_ex = FormExpr::constant(w1) * FormExpr::previousSolution(1) +
                         FormExpr::constant(w2) * FormExpr::previousSolution(2);

    // Add a TrialFunction term so FormCompiler treats this as a proper residual form.
    // The history term should contribute *no* Jacobian; we verify this by comparing against the
    // mass-only form.
    const auto residual = ((u + hist) * v).dx();
    const auto residual_ex = ((u + hist_ex) * v).dx();
    const auto residual_mass = (u * v).dx();

    auto ir = compiler.compileResidual(residual);
    auto ir_ex = compiler.compileResidual(residual_ex);
    auto ir_mass = compiler.compileResidual(residual_mass);

    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel kernel_ex(std::move(ir_ex), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel kernel_mass(std::move(ir_mass), NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();
    kernel_ex.resolveInlinableConstitutives();
    kernel_mass.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    // StandardAssembler currently populates previous-solution quadrature data only when a
    // transient (time-integration) context is present. Provide a dummy stencil requiring
    // 2 history states so previousSolution(k) and historyConvolution() can be evaluated.
    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "unit_history";
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {Real(1.0), Real(-1.0), Real(1.0)};
    ti.dt1 = dt1;
    assembler.setTimeIntegrationContext(&ti);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    std::vector<Real> U_prev = {0.07, -0.01, 0.02, 0.05};
    std::vector<Real> U_prev2 = {0.09, 0.03, -0.04, 0.01};
    assembler.setCurrentSolution(U);
    assembler.setPreviousSolution(U_prev);
    assembler.setPreviousSolution2(U_prev2);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    assembly::DenseVectorView R_ex(4);
    R_ex.zero();
    (void)assembler.assembleVector(mesh, space, kernel_ex, R_ex);

    assembly::DenseMatrixView J_mass(4);
    J_mass.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel_mass, J_mass);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(R.getVectorEntry(i), R_ex.getVectorEntry(i), 1e-12);
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(J.getMatrixEntry(i, j), J_mass.getMatrixEntry(i, j), 1e-12);
        }
    }

    // Ensure the explicit form also produces the same Jacobian as the full form.
    assembly::DenseMatrixView J_ex(4);
    assembly::DenseVectorView R_tmp(4);
    J_ex.zero();
    R_tmp.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel_ex, J_ex, R_tmp);
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(J_ex.getMatrixEntry(i, j), J_mass.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(HistoryOpsTest, HistoryWeightedSumMatchesExplicitSumAndHasNoJacobianContribution)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    constexpr Real w1 = 1.0;
    constexpr Real w2 = -0.5;

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto hist = FormExpr::historyWeightedSum({FormExpr::constant(w1), FormExpr::constant(w2)});
    const auto hist_ex = FormExpr::constant(w1) * FormExpr::previousSolution(1) +
                         FormExpr::constant(w2) * FormExpr::previousSolution(2);

    // Add a TrialFunction term so FormCompiler treats this as a proper residual form.
    const auto residual = ((u + hist) * v).dx();
    const auto residual_ex = ((u + hist_ex) * v).dx();
    const auto residual_mass = (u * v).dx();

    auto ir = compiler.compileResidual(residual);
    auto ir_ex = compiler.compileResidual(residual_ex);
    auto ir_mass = compiler.compileResidual(residual_mass);

    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel kernel_ex(std::move(ir_ex), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel kernel_mass(std::move(ir_mass), NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();
    kernel_ex.resolveInlinableConstitutives();
    kernel_mass.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    // Provide a dummy transient context so previousSolution(k) values are available at quadrature.
    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "unit_history_weighted_sum";
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {Real(1.0), Real(-1.0), Real(1.0)};
    ti.dt1 = dt1;
    assembler.setTimeIntegrationContext(&ti);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    std::vector<Real> U_prev = {0.07, -0.01, 0.02, 0.05};
    std::vector<Real> U_prev2 = {0.09, 0.03, -0.04, 0.01};
    assembler.setCurrentSolution(U);
    assembler.setPreviousSolution(U_prev);
    assembler.setPreviousSolution2(U_prev2);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    assembly::DenseVectorView R_ex(4);
    R_ex.zero();
    (void)assembler.assembleVector(mesh, space, kernel_ex, R_ex);

    assembly::DenseMatrixView J_mass(4);
    J_mass.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel_mass, J_mass);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(R.getVectorEntry(i), R_ex.getVectorEntry(i), 1e-12);
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(J.getMatrixEntry(i, j), J_mass.getMatrixEntry(i, j), 1e-12);
        }
    }

    // Ensure the explicit form also produces the same Jacobian as the full form.
    assembly::DenseMatrixView J_ex(4);
    assembly::DenseVectorView R_tmp(4);
    J_ex.zero();
    R_tmp.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel_ex, J_ex, R_tmp);
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(J_ex.getMatrixEntry(i, j), J_mass.getMatrixEntry(i, j), 1e-12);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
