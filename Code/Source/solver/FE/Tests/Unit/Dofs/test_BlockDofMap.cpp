/**
 * @file test_BlockDofMap.cpp
 * @brief Unit tests for BlockDofMap
 */

#include <gtest/gtest.h>

#include "FE/Dofs/BlockDofMap.h"
#include "FE/Core/FEException.h"

#include <vector>

using svmp::FE::FEException;
using svmp::FE::GlobalIndex;
using svmp::FE::dofs::BlockCoupling;
using svmp::FE::dofs::BlockDofMap;

TEST(BlockDofMap, BasicBlocksAndRanges) {
    BlockDofMap map;
    map.addBlock("velocity", 12);
    map.addBlock("pressure", 4);
    map.finalize();

    EXPECT_TRUE(map.isFinalized());
    EXPECT_EQ(map.numBlocks(), 2u);
    EXPECT_EQ(map.totalDofs(), 16);

    auto vel = map.getBlockRange(0);
    EXPECT_EQ(vel.first, 0);
    EXPECT_EQ(vel.second, 12);

    auto p = map.getBlockRange(1);
    EXPECT_EQ(p.first, 12);
    EXPECT_EQ(p.second, 16);
}

TEST(BlockDofMap, CouplingAndSaddlePointDetection) {
    BlockDofMap map;
    map.addBlock("A", 10);
    map.addBlock("B", 5);

    map.setCoupling(0, 0, BlockCoupling::Full);
    map.setCoupling(1, 1, BlockCoupling::None);
    map.setCoupling(0, 1, BlockCoupling::TwoWay);
    map.setCoupling(1, 0, BlockCoupling::TwoWay);

    map.finalize();

    EXPECT_EQ(map.getCoupling(1, 1), BlockCoupling::None);
    EXPECT_TRUE(map.hasSaddlePointStructure());
}

TEST(BlockDofMap, BlockView) {
    BlockDofMap map;
    map.addBlock("A", 3);
    map.addBlock("B", 2);
    map.finalize();

    auto view = map.getBlockView("B");
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->getLocalSize(), 2);
    EXPECT_TRUE(view->contains(3));
    EXPECT_TRUE(view->contains(4));
    EXPECT_FALSE(view->contains(2));
}

TEST(BlockDofMap, ThrowsOnInvalidBlock) {
    BlockDofMap map;
    map.addBlock("A", 1);
    map.finalize();

    EXPECT_THROW(map.getBlockName(5), FEException);
}

