/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormKernel_Boundary.cpp
 * @brief Unit tests for FE/Forms boundary (ds) assembly
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(FormKernelBoundaryTest, DsSingleFaceLinear)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = v.ds(2);
    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    const auto result = assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &vec);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    const Real area = 0.5;       // face {0,1,2} on reference tetra
    const Real expected = area / 3.0;

    EXPECT_NEAR(vec.getVectorEntry(0), expected, 1e-12);
    EXPECT_NEAR(vec.getVectorEntry(1), expected, 1e-12);
    EXPECT_NEAR(vec.getVectorEntry(2), expected, 1e-12);
    EXPECT_NEAR(vec.getVectorEntry(3), 0.0, 1e-12);
}

TEST(FormKernelBoundaryTest, DsMarkerFiltering)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = v.ds(1) + (FormExpr::constant(2.0) * v).ds(2);
    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &vec);

    const Real area = 0.5;
    const Real expected = 2.0 * (area / 3.0);

    EXPECT_NEAR(vec.getVectorEntry(0), expected, 1e-12);
    EXPECT_NEAR(vec.getVectorEntry(1), expected, 1e-12);
    EXPECT_NEAR(vec.getVectorEntry(2), expected, 1e-12);
    EXPECT_NEAR(vec.getVectorEntry(3), 0.0, 1e-12);
}

TEST(FormKernelBoundaryTest, DsVectorBoundaryMass_IsBlockDiagonal)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    auto scalar_dof_map = createSingleTetraDofMap();
    dofs::DofMap vec_dof_map(1, /*n_dofs_total=*/12, /*dofs_per_cell=*/12);
    std::vector<GlobalIndex> cell_dofs(12);
    for (GlobalIndex i = 0; i < 12; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    vec_dof_map.setCellDofs(0, cell_dofs);
    vec_dof_map.setNumDofs(12);
    vec_dof_map.setNumLocalDofs(12);
    vec_dof_map.finalize();

    spaces::H1Space scalar_space(ElementType::Tetra4, 1);
    auto vec_space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Tetra4, /*order=*/1, /*components=*/3);

    FormCompiler compiler;

    // Reference scalar boundary mass: (u, v)_ds
    {
        const auto u = FormExpr::trialFunction(scalar_space, "u");
        const auto v = FormExpr::testFunction(scalar_space, "v");
        auto ir = compiler.compileBilinear((u * v).ds(2));
        FormKernel kernel(std::move(ir));

        assembly::StandardAssembler assembler;
        assembler.setDofMap(scalar_dof_map);

        assembly::DenseMatrixView Ms(4);
        Ms.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, scalar_space, kernel, &Ms, nullptr);

        // Vector boundary mass: inner(u, v)_ds should assemble as 3 identical scalar blocks.
        const auto U = FormExpr::trialFunction(*vec_space, "U");
        const auto V = FormExpr::testFunction(*vec_space, "V");
        auto ir_vec = compiler.compileBilinear(inner(U, V).ds(2));
        FormKernel kernel_vec(std::move(ir_vec));

        assembly::StandardAssembler assembler_vec;
        assembler_vec.setDofMap(vec_dof_map);

        assembly::DenseMatrixView Mv(12);
        Mv.zero();
        (void)assembler_vec.assembleBoundaryFaces(mesh, 2, *vec_space, kernel_vec, &Mv, nullptr);

        const GlobalIndex dofs_per_comp = 4;
        for (GlobalIndex ci = 0; ci < 3; ++ci) {
            for (GlobalIndex cj = 0; cj < 3; ++cj) {
                for (GlobalIndex i = 0; i < dofs_per_comp; ++i) {
                    for (GlobalIndex j = 0; j < dofs_per_comp; ++j) {
                        const GlobalIndex I = ci * dofs_per_comp + i;
                        const GlobalIndex J = cj * dofs_per_comp + j;
                        const Real expected = (ci == cj) ? Ms.getMatrixEntry(i, j) : 0.0;
                        EXPECT_NEAR(Mv.getMatrixEntry(I, J), expected, 1e-12);
                    }
                }
            }
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
