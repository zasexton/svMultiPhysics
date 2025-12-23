/**
 * @file test_DofGraph.cpp
 * @brief Unit tests for DofGraph
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofGraph.h"
#include "FE/Dofs/DofMap.h"

#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::dofs::DofGraph;
using svmp::FE::dofs::DofGraphOptions;
using svmp::FE::dofs::DofMap;

namespace {

DofMap make2x2QuadMap() {
    DofMap map;
    map.reserve(4, 4);
    map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 4, 3});
    map.setCellDofs(1, std::vector<GlobalIndex>{1, 2, 5, 4});
    map.setCellDofs(2, std::vector<GlobalIndex>{3, 4, 7, 6});
    map.setCellDofs(3, std::vector<GlobalIndex>{4, 5, 8, 7});
    map.setNumDofs(9);
    map.setNumLocalDofs(9);
    map.finalize();
    return map;
}

} // namespace

TEST(DofGraph, BuildCellOnly) {
    auto map = make2x2QuadMap();

    DofGraph graph;
    graph.build(map, DofGraphOptions{});

    EXPECT_TRUE(graph.isValid());
    EXPECT_EQ(graph.numDofs(), 9);

    // Center DOF connects to all (including self).
    auto n4 = graph.getNeighbors(4);
    EXPECT_EQ(n4.size(), 9u);

    // Corner DOF 0 connects to {0,1,3,4}.
    auto n0 = graph.getNeighbors(0);
    EXPECT_EQ(n0.size(), 4u);
}

TEST(DofGraph, Statistics) {
    auto map = make2x2QuadMap();
    DofGraph graph;
    graph.build(map);

    EXPECT_GE(graph.getBandwidth(), 4);
    EXPECT_EQ(graph.getMaxRowNnz(), 9);
    EXPECT_GT(graph.getAvgRowNnz(), 0.0);

    auto stats = graph.getStatistics();
    EXPECT_EQ(stats.n_dofs, 9);
    EXPECT_TRUE(stats.symmetric);
}

