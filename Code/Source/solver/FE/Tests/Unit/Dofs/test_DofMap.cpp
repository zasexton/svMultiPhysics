/**
 * @file test_DofMap.cpp
 * @brief Unit tests for DofMap (cell->DOF CSR mapping)
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofMap.h"
#include "FE/Core/FEException.h"

#include <numeric>
#include <vector>

using svmp::FE::FEException;
using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::dofs::DofMap;

namespace {

DofMap makeTwoCellMapBuilding() {
    DofMap map;
    map.reserve(2, /*dofs_per_cell=*/2);
    {
        const std::vector<GlobalIndex> dofs0 = {0, 1};
        map.setCellDofs(0, dofs0);
    }
    {
        const std::vector<GlobalIndex> dofs1 = {1, 2};
        map.setCellDofs(1, dofs1);
    }
    map.setNumDofs(3);
    map.setNumLocalDofs(3);
    return map;
}

} // namespace

TEST(DofMap, DefaultConstruction) {
    DofMap map;
    EXPECT_FALSE(map.isFinalized());
    EXPECT_EQ(map.getNumCells(), 0);
    EXPECT_EQ(map.getNumDofs(), 0);
}

TEST(DofMap, FinalizeAndQuery) {
    auto map = makeTwoCellMapBuilding();
    EXPECT_NO_THROW(map.finalize());
    EXPECT_TRUE(map.isFinalized());
    EXPECT_EQ(map.getNumCells(), 2);
    EXPECT_EQ(map.getNumDofs(), 3);

    auto c0 = map.getCellDofs(0);
    ASSERT_EQ(c0.size(), 2u);
    EXPECT_EQ(c0[0], 0);
    EXPECT_EQ(c0[1], 1);

    auto c1 = map.getCellDofs(1);
    ASSERT_EQ(c1.size(), 2u);
    EXPECT_EQ(c1[0], 1);
    EXPECT_EQ(c1[1], 2);

    EXPECT_EQ(map.localToGlobal(1, static_cast<LocalIndex>(0)), 1);
    EXPECT_EQ(map.localToGlobal(1, static_cast<LocalIndex>(1)), 2);
    EXPECT_EQ(map.getNumCellDofs(0), static_cast<LocalIndex>(2));
}

TEST(DofMap, FinalizeValidatesDofRange) {
    DofMap map;
    map.reserve(1, 2);
    map.setCellDofs(0, std::vector<GlobalIndex>{0, 10});
    map.setNumDofs(2);
    map.setNumLocalDofs(2);

    EXPECT_THROW(map.finalize(), FEException);
}

TEST(DofMap, SetCellDofsOutOfOrderCanInvalidateCSR) {
    DofMap map;
    map.reserve(2, 2);
    map.setCellDofs(1, std::vector<GlobalIndex>{0, 1});
    map.setCellDofs(0, std::vector<GlobalIndex>{1, 2});
    map.setNumDofs(3);
    map.setNumLocalDofs(3);

    EXPECT_THROW(map.finalize(), FEException);
}

TEST(DofMap, BatchConstruction) {
    DofMap map;
    const std::vector<GlobalIndex> cell_ids = {0, 1};
    const std::vector<GlobalIndex> offsets = {0, 2, 4};
    const std::vector<GlobalIndex> dofs = {0, 1, 1, 2};

    map.setCellDofsBatch(cell_ids, offsets, dofs);
    map.setNumDofs(3);
    map.setNumLocalDofs(3);
    map.finalize();

    EXPECT_EQ(map.getNumCells(), 2);
    EXPECT_EQ(map.getNumDofs(), 3);

    auto c0 = map.getCellDofs(0);
    ASSERT_EQ(c0.size(), 2u);
    EXPECT_EQ(c0[0], 0);
    EXPECT_EQ(c0[1], 1);
}

TEST(DofMap, DeviceView) {
    auto map = makeTwoCellMapBuilding();
    map.finalize();

    auto view = map.getDeviceView();
    EXPECT_NE(view.cell_dof_offsets, nullptr);
    EXPECT_NE(view.cell_dofs, nullptr);
    EXPECT_EQ(view.n_cells, 2);
    EXPECT_EQ(view.n_dofs_total, 3);
    EXPECT_EQ(view.n_dofs_local, 3);
}

TEST(DofMap, MutationsAfterFinalizeThrow) {
    auto map = makeTwoCellMapBuilding();
    map.finalize();

    EXPECT_THROW(map.setNumDofs(3), FEException);
    EXPECT_THROW(map.reserve(2, 2), FEException);
    EXPECT_THROW(map.setCellDofs(0, std::vector<GlobalIndex>{0, 1}), FEException);
}

