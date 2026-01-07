/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearFormKernel.cpp
 * @brief Unit tests for FE/Forms nonlinear residual + AD Jacobian assembly
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(NonlinearFormKernelTest, JacobianMatchesFiniteDifferences)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (u * u * v).dx();

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

    std::array<Real, 4> R0{};
    for (GlobalIndex i = 0; i < 4; ++i) {
        R0[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        assembler.setCurrentSolution(U_plus);

        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(NonlinearFormKernelTest, DivergenceAndCurlOfCoefficientsSupportedInResidual)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto beta = FormExpr::coefficient("beta", [](Real x, Real y, Real z) {
        return std::array<Real, 3>{x * x, y + 1.0, z - 2.0};
    });

    // Include TrialFunction so this is a valid residual, and include div/curl
    // so Dual evaluation must support these coefficient operators.
    const auto residual = (u * v + div(beta) * v + component(curl(beta), 2) * v).dx();

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

    std::array<Real, 4> R0{};
    for (GlobalIndex i = 0; i < 4; ++i) {
        R0[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        assembler.setCurrentSolution(U_plus);

        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(NonlinearFormKernelTest, HessianJacobianMatchesFiniteDifferences_P2)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 2);

    dofs::DofMap dof_map(1, 10, 10);
    std::vector<GlobalIndex> cell_dofs(10);
    for (GlobalIndex i = 0; i < 10; ++i) cell_dofs[static_cast<std::size_t>(i)] = i;
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(10);
    dof_map.setNumLocalDofs(10);
    dof_map.finalize();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(u.hessian(), v.hessian()).dx();

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U(10, 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.05 * static_cast<Real>(i + 1);
    }
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(10);
    assembly::DenseVectorView R(10);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    std::vector<Real> R0(10, 0.0);
    for (GlobalIndex i = 0; i < 10; ++i) {
        R0[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    for (GlobalIndex j = 0; j < 10; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        assembler.setCurrentSolution(U_plus);

        assembly::DenseVectorView Rp(10);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        for (GlobalIndex i = 0; i < 10; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(NonlinearFormKernelTest, HessianJacobianMatchesFiniteDifferences_VectorP2)
{
    SingleTetraMeshAccess mesh;
    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 2);
    spaces::ProductSpace vec_space(scalar_space, /*components=*/2);

    const auto n_dofs = static_cast<GlobalIndex>(vec_space.dofs_per_element());
    dofs::DofMap dof_map(1, n_dofs, static_cast<LocalIndex>(n_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(vec_space, "u");
    const auto v = FormExpr::testFunction(vec_space, "v");
    const auto residual = inner(component(u, 0).hessian(), component(v, 0).hessian()).dx();

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.05 * static_cast<Real>(i + 1);
    }
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(n_dofs);
    assembly::DenseVectorView R(n_dofs);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, vec_space, vec_space, kernel, J, R);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        assembler.setCurrentSolution(U_plus);

        assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)assembler.assembleVector(mesh, vec_space, kernel, Rp);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(NonlinearFormKernelTest, DiscreteFieldHessianAffectsResidualAndJacobian_P2)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 2);

    const GlobalIndex n_dofs = static_cast<GlobalIndex>(space.dofs_per_element());
    ASSERT_EQ(n_dofs, 10);

    dofs::DofMap dof_map(1, n_dofs, static_cast<LocalIndex>(n_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();

    constexpr FieldId a_field = 42;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto a = FormExpr::discreteField(a_field, space, "a");
    const auto residual = (inner(u.hessian(), a.hessian()) * v).dx();

    auto ir = compiler.compileResidual(residual);
    ASSERT_EQ(ir.fieldRequirements().size(), 1u);
    EXPECT_EQ(ir.fieldRequirements()[0].field, a_field);
    EXPECT_TRUE(assembly::hasFlag(ir.fieldRequirements()[0].required, assembly::RequiredData::SolutionValues));
    EXPECT_TRUE(assembly::hasFlag(ir.fieldRequirements()[0].required, assembly::RequiredData::SolutionHessians));

    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const std::array<assembly::FieldSolutionAccess, 1> field_access = {
        assembly::FieldSolutionAccess{a_field, &space, &dof_map, n_dofs}};
    assembler.setFieldSolutionAccess(field_access);

    std::vector<Real> U(static_cast<std::size_t>(2 * n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = 0.05 * static_cast<Real>(i + 1);
    }

    auto assemble_and_measure = [&](const std::vector<Real>& state) {
        assembler.setCurrentSolution(state);

        assembly::DenseMatrixView J(n_dofs);
        assembly::DenseVectorView R(n_dofs);
        J.zero();
        R.zero();

        (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

        Real max_R = 0.0;
        Real max_J = 0.0;
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            max_R = std::max(max_R, std::abs(R.getVectorEntry(i)));
            for (GlobalIndex j = 0; j < n_dofs; ++j) {
                max_J = std::max(max_J, std::abs(J.getMatrixEntry(i, j)));
            }
        }
        return std::pair<Real, Real>{max_R, max_J};
    };

    // Zero auxiliary field -> zero residual/Jacobian for this term.
    {
        auto U0 = U;
        const auto [max_R, max_J] = assemble_and_measure(U0);
        EXPECT_NEAR(max_R, 0.0, 1e-12);
        EXPECT_NEAR(max_J, 0.0, 1e-12);
    }

    // Activate auxiliary field DOFs -> residual/Jacobian should become nonzero.
    {
        auto U1 = U;
        U1[static_cast<std::size_t>(n_dofs + 0)] = 1.0;
        const auto [max_R, max_J] = assemble_and_measure(U1);
        EXPECT_GT(max_R, 1e-12);
        EXPECT_GT(max_J, 1e-12);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
