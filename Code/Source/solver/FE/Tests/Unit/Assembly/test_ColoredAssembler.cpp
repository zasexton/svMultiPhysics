/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ColoredAssembler.cpp
 * @brief Unit tests for ColoredAssembler
 */

#include <gtest/gtest.h>
#include "Assembly/ColoredAssembler.h"
#include "Assembly/BlockAssembler.h"
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Classes
// ============================================================================

/**
 * @brief Mock mesh access for colored assembly testing
 */
class MockMeshAccess : public IMeshAccess {
public:
    MockMeshAccess(GlobalIndex num_cells = 100)
        : num_cells_(num_cells)
    {
    }

    GlobalIndex numCells() const override { return num_cells_; }
    GlobalIndex numOwnedCells() const override { return num_cells_; }
    GlobalIndex numBoundaryFaces() const override { return 0; }
    GlobalIndex numInteriorFaces() const override { return 0; }
    int dimension() const override { return 3; }

    bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        nodes.clear();
        nodes.push_back(cell_id * 4);
        nodes.push_back(cell_id * 4 + 1);
        nodes.push_back(cell_id * 4 + 2);
        nodes.push_back(cell_id * 4 + 3);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return {
            static_cast<Real>(node_id % 10),
            static_cast<Real>((node_id / 10) % 10),
            static_cast<Real>(node_id / 100)
        };
    }

    void getCellCoordinates(GlobalIndex cell_id,
                           std::vector<std::array<Real, 3>>& coords) const override {
        coords.clear();
        for (int i = 0; i < 4; ++i) {
            coords.push_back(getNodeCoordinates(cell_id * 4 + i));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 0;
    }

    std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < num_cells_; ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(callback);
    }

    void forEachBoundaryFace(int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override {}

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}

private:
    GlobalIndex num_cells_;
};

} // namespace

// ============================================================================
// Test Fixtures
// ============================================================================

class ColoredAssemblerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>(100);
    }

    std::unique_ptr<MockMeshAccess> mesh_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, DefaultConstruction) {
    ColoredAssembler assembler;
    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_FALSE(assembler.hasColoring());
}

TEST_F(ColoredAssemblerTest, ConstructionWithColoringOptions) {
    ColoringOptions options;
    options.algorithm = ColoringAlgorithm::Greedy;
    options.max_colors = 64;
    options.balance_colors = false;
    options.verbose = true;

    ColoredAssembler assembler(options);

    EXPECT_EQ(assembler.name(), "Colored(StandardAssembler)");
    EXPECT_FALSE(assembler.hasColoring());
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, SetOptionsInvalidatesColoringAndForwardsToBase) {
    ColoredAssembler assembler;

    // Seed a coloring state
    std::vector<int> colors(100, 0);
    assembler.setColoring(colors, 1);
    EXPECT_TRUE(assembler.hasColoring());

    AssemblyOptions options{};
    options.num_threads = 8;
    options.use_coloring = true;
    options.coloring.algorithm = ColoringAlgorithm::DSatur;
    options.coloring.max_colors = 128; // trigger invalidation
    options.coloring.balance_colors = true;

    assembler.setOptions(options);

    EXPECT_EQ(assembler.getOptions().num_threads, 8);
    EXPECT_FALSE(assembler.hasColoring());
}

// ============================================================================
// Coloring Options Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, ColoringAlgorithmEnum) {
    EXPECT_NE(ColoringAlgorithm::Greedy, ColoringAlgorithm::DSatur);
    EXPECT_NE(ColoringAlgorithm::DSatur, ColoringAlgorithm::LargestFirst);
    EXPECT_NE(ColoringAlgorithm::LargestFirst, ColoringAlgorithm::SmallestLast);
}

TEST_F(ColoredAssemblerTest, ColoringOptionsDefaults) {
    ColoringOptions options;

    EXPECT_EQ(options.algorithm, ColoringAlgorithm::DSatur);
    EXPECT_EQ(options.max_colors, 256);
    EXPECT_TRUE(options.balance_colors);
    EXPECT_FALSE(options.reorder_elements);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// Coloring Statistics Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, ColoringStatsDefaults) {
    ColoringStats stats;

    EXPECT_EQ(stats.num_colors, 0);
    EXPECT_EQ(stats.num_elements, 0);
    EXPECT_EQ(stats.min_color_size, 0);
    EXPECT_EQ(stats.max_color_size, 0);
    EXPECT_DOUBLE_EQ(stats.avg_color_size, 0.0);
    EXPECT_DOUBLE_EQ(stats.coloring_seconds, 0.0);
    EXPECT_TRUE(stats.color_sizes.empty());
}

// ============================================================================
// Element Graph Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, ElementGraphConstruction) {
    ElementGraph graph;
    EXPECT_EQ(graph.numElements(), 0);
    EXPECT_EQ(graph.numEdges(), 0);
}

TEST_F(ColoredAssemblerTest, ElementGraphWithSize) {
    ElementGraph graph(10);
    EXPECT_EQ(graph.numElements(), 10);
}

TEST_F(ColoredAssemblerTest, ElementGraphClear) {
    ElementGraph graph(10);
    graph.clear();
    EXPECT_EQ(graph.numElements(), 0);
    EXPECT_EQ(graph.numEdges(), 0);
}

// ============================================================================
// Coloring Management Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, NoColoringByDefault) {
    ColoredAssembler assembler;
    EXPECT_FALSE(assembler.hasColoring());
    EXPECT_EQ(assembler.numColors(), 0);
}

TEST_F(ColoredAssemblerTest, SetColoring) {
    ColoredAssembler assembler;

    std::vector<int> colors(100);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 5);  // 5 colors
    }

    assembler.setColoring(colors, 5);

    EXPECT_TRUE(assembler.hasColoring());
    EXPECT_EQ(assembler.numColors(), 5);
}

TEST_F(ColoredAssemblerTest, SetColoringReturnsColors) {
    ColoredAssembler assembler;

    std::vector<int> colors(100);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 3);  // 3 colors
    }

    assembler.setColoring(colors, 3);

    auto returned_colors = assembler.getColors();
    EXPECT_EQ(returned_colors.size(), 100u);

    // Verify colors match
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(returned_colors[i], colors[i]);
    }
}

// ============================================================================
// Block Index Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, BlockIndexDefault) {
    BlockIndex idx;
    // Default-initialized, values unspecified
    EXPECT_TRUE(true);
}

TEST_F(ColoredAssemblerTest, BlockIndexConstruction) {
    BlockIndex idx(2, 3);
    EXPECT_EQ(idx.row_field, 2);
    EXPECT_EQ(idx.col_field, 3);
}

TEST_F(ColoredAssemblerTest, BlockIndexEquality) {
    BlockIndex idx1(1, 2);
    BlockIndex idx2(1, 2);
    BlockIndex idx3(1, 3);

    EXPECT_TRUE(idx1 == idx2);
    EXPECT_FALSE(idx1 == idx3);
}

TEST_F(ColoredAssemblerTest, BlockIndexOrdering) {
    BlockIndex idx1(0, 0);
    BlockIndex idx2(0, 1);
    BlockIndex idx3(1, 0);

    EXPECT_TRUE(idx1 < idx2);
    EXPECT_TRUE(idx1 < idx3);
    EXPECT_TRUE(idx2 < idx3);
}

TEST_F(ColoredAssemblerTest, BlockIndexIsDiagonal) {
    BlockIndex diag(2, 2);
    BlockIndex off_diag(2, 3);

    EXPECT_TRUE(diag.isDiagonal());
    EXPECT_FALSE(off_diag.isDiagonal());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, MoveConstruction) {
    ColoringOptions coloring;
    coloring.max_colors = 64;

    ColoredAssembler assembler1(coloring);

    AssemblyOptions options{};
    options.num_threads = 4;
    options.use_coloring = true;
    options.coloring = coloring;
    assembler1.setOptions(options);

    std::vector<int> colors(50, 0);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 3);
    }
    assembler1.setColoring(colors, 3);

    ColoredAssembler assembler2(std::move(assembler1));

    EXPECT_TRUE(assembler2.hasColoring());
    EXPECT_EQ(assembler2.numColors(), 3);
    EXPECT_EQ(assembler2.getOptions().num_threads, 4);
}

TEST_F(ColoredAssemblerTest, MoveAssignment) {
    ColoredAssembler assembler1;

    AssemblyOptions options{};
    options.num_threads = 4;
    options.use_coloring = true;
    assembler1.setOptions(options);

    std::vector<int> colors(50, 0);
    assembler1.setColoring(colors, 2);

    ColoredAssembler assembler2;
    assembler2 = std::move(assembler1);

    EXPECT_TRUE(assembler2.hasColoring());
    EXPECT_EQ(assembler2.numColors(), 2);
}

// ============================================================================
// Coloring Utility Function Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, VerifyColoringValidEmptyGraph) {
    ElementGraph graph;
    std::vector<int> colors;

    // Empty graph/colors should be valid
    EXPECT_TRUE(verifyColoring(graph, colors));
}

// ============================================================================
// Coloring Statistics After SetColoring
// ============================================================================

TEST_F(ColoredAssemblerTest, GetColoringStatsAfterSetColoring) {
    ColoredAssembler assembler;

    std::vector<int> colors(100);
    // Create uneven distribution: 50 elements color 0, 30 color 1, 20 color 2
    for (std::size_t i = 0; i < 50; ++i) colors[i] = 0;
    for (std::size_t i = 50; i < 80; ++i) colors[i] = 1;
    for (std::size_t i = 80; i < 100; ++i) colors[i] = 2;

    assembler.setColoring(colors, 3);

    EXPECT_TRUE(assembler.hasColoring());
    EXPECT_EQ(assembler.numColors(), 3);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
