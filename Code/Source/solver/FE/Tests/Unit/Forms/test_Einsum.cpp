/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Einsum.cpp
 * @brief Unit tests for FE/Forms Einstein index notation lowering (einsum)
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/Einsum.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] dofs::DofMap makeSingleCellDofMap(GlobalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, static_cast<LocalIndex>(n_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

assembly::DenseMatrixView assembleCellBilinear(const FormExpr& bilinear_form,
                                               dofs::DofMap& dof_map,
                                               const assembly::IMeshAccess& mesh,
                                               const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    auto ir = compiler.compileBilinear(bilinear_form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    return mat;
}

assembly::DenseVectorView assembleCellLinear(const FormExpr& linear_form,
                                             dofs::DofMap& dof_map,
                                             const assembly::IMeshAccess& mesh,
                                             const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    auto ir = compiler.compileLinear(linear_form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec;
}

} // namespace

TEST(EinsumTest, LowersGradInnerProductToExplicitComponentSum)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto a_ref = inner(grad(u), grad(v)).dx();

    const Index i("i");
    const auto a_idx = einsum(grad(u)(i) * grad(v)(i)).dx();

    const auto A_ref = assembleCellBilinear(a_ref, dof_map, mesh, space);
    const auto A_idx = assembleCellBilinear(a_idx, dof_map, mesh, space);

    for (GlobalIndex r = 0; r < 4; ++r) {
        for (GlobalIndex c = 0; c < 4; ++c) {
            EXPECT_NEAR(A_ref.getMatrixEntry(r, c), A_idx.getMatrixEntry(r, c), 1e-12);
        }
    }
}

TEST(EinsumTest, EinsumSupportsVectorOutputForOneFreeIndex)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto v = FormExpr::testFunction(space, "v");

    const auto A = FormExpr::coefficient("A", [](Real, Real, Real) {
        return std::array<std::array<Real, 3>, 3>{
            std::array<Real, 3>{1.0, 2.0, 3.0},
            std::array<Real, 3>{4.0, 5.0, 6.0},
            std::array<Real, 3>{7.0, 8.0, 9.0},
        };
    });
    const auto B = FormExpr::coefficient("B", [](Real, Real, Real) {
        return std::array<Real, 3>{0.25, -1.5, 2.0};
    });

    const Index i("i");
    const Index j("j");
    const auto vec_idx = einsum(A(i, j) * B(j));

    const auto L_ref = inner(A * B, v).dx();
    const auto L_idx = inner(vec_idx, v).dx();

    const auto R_ref = assembleCellLinear(L_ref, dof_map, mesh, space);
    const auto R_idx = assembleCellLinear(L_idx, dof_map, mesh, space);

    ASSERT_EQ(R_ref.numRows(), R_idx.numRows());
    for (GlobalIndex d = 0; d < R_ref.numRows(); ++d) {
        EXPECT_NEAR(R_ref.getVectorEntry(d), R_idx.getVectorEntry(d), 1e-12);
    }
}

TEST(EinsumTest, EinsumSupportsMatrixOutputForTwoFreeIndices)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = FormExpr::coefficient("A", [](Real, Real, Real) {
        return std::array<std::array<Real, 3>, 3>{
            std::array<Real, 3>{1.0, 0.5, -2.0},
            std::array<Real, 3>{0.0, -3.0, 4.0},
            std::array<Real, 3>{2.5, 1.25, -0.75},
        };
    });
    const auto B = FormExpr::coefficient("B", [](Real, Real, Real) {
        return std::array<std::array<Real, 3>, 3>{
            std::array<Real, 3>{-1.0, 2.0, 0.0},
            std::array<Real, 3>{0.25, -0.5, 3.0},
            std::array<Real, 3>{1.5, 0.0, -2.0},
        };
    });

    const Index i("i");
    const Index j("j");
    const Index k("k");
    const auto mat_idx = einsum(A(i, k) * B(k, j));

    const auto a_ref = inner((A * B) * u, v).dx();
    const auto a_idx = inner(mat_idx * u, v).dx();

    const auto K_ref = assembleCellBilinear(a_ref, dof_map, mesh, space);
    const auto K_idx = assembleCellBilinear(a_idx, dof_map, mesh, space);

    for (GlobalIndex r = 0; r < static_cast<GlobalIndex>(space.dofs_per_element()); ++r) {
        for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(space.dofs_per_element()); ++c) {
            EXPECT_NEAR(K_ref.getMatrixEntry(r, c), K_idx.getMatrixEntry(r, c), 1e-12);
        }
    }
}

TEST(EinsumTest, CompileRejectsUnloweredIndexedAccess)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");

    FormCompiler compiler;

    try {
        (void)compiler.compileBilinear((grad(u)(i) * grad(v)(i)).dx());
        FAIL() << "Expected FormCompiler to reject unlowered indexed access";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("forms::einsum"), std::string::npos);
    }
}

TEST(EinsumTest, EinsumRejectsMoreThanTwoFreeIndices)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const Index i("i");
    const Index j("j");
    const Index k("k");

    try {
        // i,j,k are all free.
        (void)einsum(grad(u)(i) * grad(u)(j) * grad(u)(k));
        FAIL() << "Expected einsum to reject 3 free indices";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("free indices"), std::string::npos);
    }
}

TEST(EinsumTest, EinsumRejectsTripleIndexUse)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");

    // i appears 3 times.
    const auto expr = (grad(u)(i) * grad(v)(i)) * grad(v)(i);

    try {
        (void)einsum(expr);
        FAIL() << "Expected einsum to reject an index used more than twice";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("exactly twice"), std::string::npos);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
