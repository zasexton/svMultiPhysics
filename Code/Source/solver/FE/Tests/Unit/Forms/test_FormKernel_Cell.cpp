/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormKernel_Cell.cpp
 * @brief Unit tests for FE/Forms cell (dx) assembly
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] dofs::DofMap createSingleTetraDofMap(LocalIndex n_dofs)
{
    dofs::DofMap dof_map(1, static_cast<GlobalIndex>(n_dofs), n_dofs);
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (LocalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = static_cast<GlobalIndex>(i);
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(static_cast<GlobalIndex>(n_dofs));
    dof_map.setNumLocalDofs(static_cast<GlobalIndex>(n_dofs));
    dof_map.finalize();
    return dof_map;
}

[[nodiscard]] Real matrixInner(const basis::VectorJacobian& A,
                               const basis::VectorJacobian& B,
                               int value_dim,
                               int dim)
{
    Real sum = Real(0);
    for (int r = 0; r < value_dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            sum += A(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) *
                   B(static_cast<std::size_t>(r), static_cast<std::size_t>(c));
        }
    }
    return sum;
}

[[nodiscard]] basis::VectorJacobian sym3(const basis::VectorJacobian& A)
{
    basis::VectorJacobian out{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            out(r, c) = Real(0.5) * (A(r, c) + A(c, r));
        }
    }
    return out;
}

[[nodiscard]] assembly::DenseMatrixView assembleVectorGradientMatrix(
    const spaces::FunctionSpace& space,
    const FormExpr& form)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

    FormCompiler compiler;
    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(space.dofs_per_element()));
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    return mat;
}

} // namespace

TEST(FormKernelCellTest, LinearDxIntegratesBasisFunctions)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = v.dx();

    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    (void)assembler.assembleVector(mesh, space, kernel, vec);

    const Real V = 1.0 / 6.0;
    const Real expected = V / 4.0;

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormKernelCellTest, LinearDxWithConstantScaling)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(2.0) * v).dx();

    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    (void)assembler.assembleVector(mesh, space, kernel, vec);

    const Real V = 1.0 / 6.0;
    const Real expected = 2.0 * (V / 4.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormKernelCellTest, HDivVectorBasisGradInnerProductUsesAnalyticJacobians)
{
    spaces::HDivSpace space(ElementType::Tetra4, 0, BasisType::RaviartThomas);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mat = assembleVectorGradientMatrix(space, inner(grad(u), grad(v)).dx());

    const auto& basis = space.element().basis();
    const auto quad = space.element().quadrature();
    ASSERT_NE(quad.get(), nullptr);

    std::vector<basis::VectorJacobian> jacobians;
    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            Real expected = Real(0);
            for (std::size_t q = 0; q < quad->num_points(); ++q) {
                basis.evaluate_vector_jacobians(quad->point(q), jacobians);
                ASSERT_EQ(jacobians.size(), space.dofs_per_element());
                expected += quad->weight(q) *
                    matrixInner(jacobians[static_cast<std::size_t>(j)],
                                jacobians[static_cast<std::size_t>(i)],
                                space.value_dimension(),
                                space.topological_dimension());
            }
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12)
                << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FormKernelCellTest, HCurlVectorBasisSymGradInnerProductUsesAnalyticJacobians)
{
    spaces::HCurlSpace space(ElementType::Tetra4, 0, BasisType::Nedelec);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mat = assembleVectorGradientMatrix(space, inner(sym(grad(u)), sym(grad(v))).dx());

    const auto& basis = space.element().basis();
    const auto quad = space.element().quadrature();
    ASSERT_NE(quad.get(), nullptr);

    std::vector<basis::VectorJacobian> jacobians;
    basis.evaluate_vector_jacobians(quad->point(0), jacobians);
    ASSERT_EQ(jacobians.size(), space.dofs_per_element());

    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            Real expected = Real(0);
            for (std::size_t q = 0; q < quad->num_points(); ++q) {
                basis.evaluate_vector_jacobians(quad->point(q), jacobians);
                expected += quad->weight(q) *
                    matrixInner(sym3(jacobians[static_cast<std::size_t>(j)]),
                                sym3(jacobians[static_cast<std::size_t>(i)]),
                                space.value_dimension(),
                                space.topological_dimension());
            }
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12)
                << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FormKernelCellTest, DtBilinearRequiresTransientContextAndSignalsTemporalOrder)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (dt(u, 2) * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));
    EXPECT_EQ(kernel.maxTemporalDerivativeOrder(), 2);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();

    try {
        (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
        FAIL() << "Expected assembly to fail without a transient time-integration context";
    } catch (const svmp::FE::FEException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("dt(...) operator requires a transient time-integration context"), std::string::npos);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
