/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormKernel_DG.cpp
 * @brief Unit tests for FE/Forms interior-facet (dS) assembly
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(FormKernelDGTest, PenaltyJumpJumpProducesExpectedBlocks)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real eta = 2.5;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(eta) * inner(jump(u), jump(v))).dS();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area = std::sqrt(3.0) / 2.0;
    const Real mdiag = eta * (area / 6.0);
    const Real moff = eta * (area / 12.0);

    auto expected_entry = [&](GlobalIndex i, GlobalIndex j) -> Real {
        // Map global DOFs to (side, local index on shared face)
        // Minus cell DOFs: 0,1,2,3 (face DOFs are 1,2,3 -> map to 0,1,2)
        // Plus cell DOFs: 4,5,6,7 (face DOFs are 4,5,6 -> map to 0,1,2)
        auto faceIndex = [](GlobalIndex dof) -> int {
            if (dof == 1) return 0;
            if (dof == 2) return 1;
            if (dof == 3) return 2;
            if (dof == 4) return 0;
            if (dof == 5) return 1;
            if (dof == 6) return 2;
            return -1;
        };
        auto isMinus = [](GlobalIndex dof) -> bool { return dof < 4; };

        const int fi = faceIndex(i);
        const int fj = faceIndex(j);
        if (fi < 0 || fj < 0) return 0.0;

        const bool ii_minus = isMinus(i);
        const bool jj_minus = isMinus(j);
        const bool same_side = (ii_minus == jj_minus);

        const Real base = (fi == fj) ? mdiag : moff;
        return same_side ? base : -base;
    };

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected_entry(i, j), 5e-11);
        }
    }
}

TEST(FormKernelDGTest, PenaltyJumpJumpHandlesPermutedPlusFaceOrdering)
{
    TwoTetraSharedFacePermutedPlusMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real eta = 2.5;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(eta) * inner(jump(u), jump(v))).dS();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area = std::sqrt(3.0) / 2.0;
    const Real mdiag = eta * (area / 6.0);
    const Real moff = eta * (area / 12.0);

    // Minus face ordering (cell 0 face 2): global nodes {1,2,3} in that order.
    // Plus face ordering  (cell 1 face 0): global nodes {2,3,1} in that order.
    // perm_plus_to_minus[j] gives which minus vertex index matches plus vertex j.
    const std::array<int, 3> perm_plus_to_minus = {1, 2, 0};

    auto minusFaceIndex = [](GlobalIndex dof) -> int {
        if (dof == 1) return 0;
        if (dof == 2) return 1;
        if (dof == 3) return 2;
        return -1;
    };
    auto plusFaceIndex = [](GlobalIndex dof) -> int {
        if (dof == 4) return 0;
        if (dof == 5) return 1;
        if (dof == 6) return 2;
        return -1;
    };
    auto isMinus = [](GlobalIndex dof) -> bool { return dof < 4; };

    auto expected_entry = [&](GlobalIndex i, GlobalIndex j) -> Real {
        const bool i_minus = isMinus(i);
        const bool j_minus = isMinus(j);

        const int fi = i_minus ? minusFaceIndex(i) : plusFaceIndex(i);
        const int fj = j_minus ? minusFaceIndex(j) : plusFaceIndex(j);
        if (fi < 0 || fj < 0) return 0.0;

        const bool same_side = (i_minus == j_minus);
        if (same_side) {
            const Real base = (fi == fj) ? mdiag : moff;
            return base;
        }

        // Cross terms: diag/off-diag determined by matching physical vertices.
        const bool i_is_minus_j_is_plus = i_minus && !j_minus;
        bool match = false;
        if (i_is_minus_j_is_plus) {
            match = (fi == perm_plus_to_minus[static_cast<std::size_t>(fj)]);
        } else {
            // i is plus, j is minus
            match = (perm_plus_to_minus[static_cast<std::size_t>(fi)] == fj);
        }

        const Real base = match ? mdiag : moff;
        return -base;
    };

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected_entry(i, j), 5e-11);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
