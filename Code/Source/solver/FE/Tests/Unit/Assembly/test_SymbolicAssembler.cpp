/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SymbolicAssembler.cpp
 * @brief Unit tests for SymbolicAssembler integration with FE/Forms
 */

#include <gtest/gtest.h>

#include "Assembly/SymbolicAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"

#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

class SingleTetraMeshAccess final : public IMeshAccess {
public:
    SingleTetraMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 4> cell_{};
};

inline dofs::DofMap createSingleTetraDofMap()
{
    dofs::DofMap dof_map(1, 4, 4);
    std::vector<GlobalIndex> cell_dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();
    return dof_map;
}

TEST(SymbolicAssemblerTest, SymbolicOptionsRoundTrip)
{
    forms::SymbolicOptions options;
    options.ad_mode = forms::ADMode::Forward;
    options.verbose = true;

    SymbolicAssembler assembler(options);
    EXPECT_EQ(assembler.getSymbolicOptions().ad_mode, forms::ADMode::Forward);
    EXPECT_TRUE(assembler.getSymbolicOptions().verbose);

    forms::SymbolicOptions options2;
    options2.cache_expressions = false;
    assembler.setSymbolicOptions(options2);
    EXPECT_FALSE(assembler.getSymbolicOptions().cache_expressions);
}

TEST(SymbolicAssemblerTest, PrecompileBilinear)
{
    SymbolicAssembler assembler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = forms::FormExpr::trialFunction(space, "u");
    const auto v = forms::FormExpr::testFunction(space, "v");
    const auto form = (u * v).dx();

    auto kernel = assembler.precompileBilinear(form);
    EXPECT_NE(kernel, nullptr);
}

TEST(SymbolicAssemblerTest, AssembleMassMatrix_SingleTetra)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    SymbolicAssembler assembler;
    assembler.setDofMap(dof_map);

    const auto u = forms::FormExpr::trialFunction(space, "u");
    const auto v = forms::FormExpr::testFunction(space, "v");
    const auto mass = (u * v).dx();

    DenseMatrixView matrix_view(4);
    matrix_view.zero();

    const auto result = assembler.assembleForm(mass, mesh, space, space, matrix_view);
    EXPECT_EQ(result.elements_assembled, 1);

    const Real V = 1.0 / 6.0;
    const Real diag = V / 10.0;
    const Real off = V / 20.0;

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(matrix_view.getMatrixEntry(i, j), expected, 1e-12);
        }
    }
}

TEST(SymbolicAssemblerTest, AssembleStiffnessMatrix_SingleTetra)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    SymbolicAssembler assembler;
    assembler.setDofMap(dof_map);

    const auto u = forms::FormExpr::trialFunction(space, "u");
    const auto v = forms::FormExpr::testFunction(space, "v");
    const auto stiff = forms::inner(forms::grad(u), forms::grad(v)).dx();

    DenseMatrixView matrix_view(4);
    matrix_view.zero();

    const auto result = assembler.assembleForm(stiff, mesh, space, space, matrix_view);
    EXPECT_EQ(result.elements_assembled, 1);

    const Real V = 1.0 / 6.0;
    const Real e00 = 3.0 * V;
    const Real em = -1.0 * V;
    const Real e11 = 1.0 * V;

    const Real expected[4][4] = {
        {e00, em,  em,  em},
        {em,  e11, 0.0, 0.0},
        {em,  0.0, e11, 0.0},
        {em,  0.0, 0.0, e11}
    };

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(matrix_view.getMatrixEntry(i, j), expected[i][j], 1e-12);
        }
    }
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
