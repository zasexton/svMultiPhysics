/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyLoop.cpp
 * @brief Unit tests for AssemblyLoop
 */

#include <gtest/gtest.h>
#include "Assembly/AssemblyLoop.h"
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Classes
// ============================================================================

/**
 * @brief Simple mock mesh access for testing
 */
class MockMeshAccess : public IMeshAccess {
public:
    MockMeshAccess(GlobalIndex num_cells = 100)
        : num_cells_(num_cells)
    {
        // Generate mock node coordinates
        for (GlobalIndex i = 0; i < num_cells_ * 4; ++i) {
            node_coords_.push_back({
                static_cast<Real>(i % 10),
                static_cast<Real>((i / 10) % 10),
                static_cast<Real>(i / 100)
            });
        }
    }

    GlobalIndex numCells() const override { return num_cells_; }
    GlobalIndex numOwnedCells() const override { return num_cells_ * 3 / 4; }
    GlobalIndex numBoundaryFaces() const override { return 20; }
    GlobalIndex numInteriorFaces() const override { return num_cells_ * 3 / 2; }
    int dimension() const override { return 3; }

    bool isOwnedCell(GlobalIndex cell_id) const override {
        return cell_id < num_cells_ * 3 / 4;  // 75% owned
    }

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
        if (static_cast<std::size_t>(node_id) < node_coords_.size()) {
            return node_coords_[static_cast<std::size_t>(node_id)];
        }
        return {0.0, 0.0, 0.0};
    }

    void getCellCoordinates(GlobalIndex cell_id,
                           std::vector<std::array<Real, 3>>& coords) const override {
        coords.clear();
        coords.push_back(getNodeCoordinates(cell_id * 4));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 1));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 2));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 3));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;  // Simplified
    }

    int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 1;  // Default marker
    }

    std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override {
        return {face_id % num_cells_, (face_id + 1) % num_cells_};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < num_cells_; ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < num_cells_ * 3 / 4; ++i) {
            callback(i);
        }
    }

    void forEachBoundaryFace(int marker,
        std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker == 1) {
            for (GlobalIndex i = 0; i < 10; ++i) {
                callback(i, i);  // face i belongs to cell i
            }
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        for (GlobalIndex i = 0; i < numInteriorFaces(); ++i) {
            callback(i, i % num_cells_, (i + 1) % num_cells_);
        }
    }

private:
    GlobalIndex num_cells_;
    std::vector<std::array<Real, 3>> node_coords_;
};

} // namespace

// ============================================================================
// Test Fixtures
// ============================================================================

class AssemblyLoopTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>(100);
    }

    std::unique_ptr<MockMeshAccess> mesh_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(AssemblyLoopTest, DefaultConstruction) {
    AssemblyLoop loop;
    EXPECT_FALSE(loop.isConfigured());
}

TEST_F(AssemblyLoopTest, ConstructionWithOptions) {
    LoopOptions options;
    options.mode = LoopMode::OpenMP;
    options.num_threads = 4;
    options.deterministic = true;

    AssemblyLoop loop(options);
    EXPECT_EQ(loop.getOptions().mode, LoopMode::OpenMP);
    EXPECT_EQ(loop.getOptions().num_threads, 4);
    EXPECT_TRUE(loop.getOptions().deterministic);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(AssemblyLoopTest, SetMesh) {
    AssemblyLoop loop;
    loop.setMesh(*mesh_);
    // Would need DofMap to be fully configured
    EXPECT_FALSE(loop.isConfigured());  // Still needs DofMap
}

TEST_F(AssemblyLoopTest, SetOptions) {
    AssemblyLoop loop;

    LoopOptions options;
    options.mode = LoopMode::Colored;
    options.num_threads = 8;
    options.skip_ghost_cells = true;

    loop.setOptions(options);

    EXPECT_EQ(loop.getOptions().mode, LoopMode::Colored);
    EXPECT_EQ(loop.getOptions().num_threads, 8);
    EXPECT_TRUE(loop.getOptions().skip_ghost_cells);
}

// ============================================================================
// Loop Options Tests
// ============================================================================

TEST_F(AssemblyLoopTest, LoopModeEnum) {
    EXPECT_NE(LoopMode::Sequential, LoopMode::OpenMP);
    EXPECT_NE(LoopMode::OpenMP, LoopMode::Colored);
    EXPECT_NE(LoopMode::Colored, LoopMode::WorkStream);
}

TEST_F(AssemblyLoopTest, LoopOptionsDefaults) {
    LoopOptions options;

    EXPECT_EQ(options.mode, LoopMode::Sequential);
    EXPECT_EQ(options.num_threads, 1);
    EXPECT_TRUE(options.deterministic);
    EXPECT_FALSE(options.skip_ghost_cells);
    EXPECT_FALSE(options.prefetch_next);
    EXPECT_EQ(options.batch_size, 1);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// Work Item Tests
// ============================================================================

TEST_F(AssemblyLoopTest, CellWorkItem) {
    CellWorkItem item(42, ElementType::Hex8, true);

    EXPECT_EQ(item.cell_id, 42);
    EXPECT_EQ(item.cell_type, ElementType::Hex8);
    EXPECT_TRUE(item.is_owned);
}

TEST_F(AssemblyLoopTest, BoundaryFaceWorkItem) {
    BoundaryFaceWorkItem item(10, 5, 2, 1, ElementType::Tetra4);

    EXPECT_EQ(item.face_id, 10);
    EXPECT_EQ(item.cell_id, 5);
    EXPECT_EQ(item.local_face_id, 2u);
    EXPECT_EQ(item.boundary_marker, 1);
    EXPECT_EQ(item.cell_type, ElementType::Tetra4);
}

TEST_F(AssemblyLoopTest, InteriorFaceWorkItem) {
    InteriorFaceWorkItem item(10, 5, 6, 2, 3, ElementType::Tetra4, ElementType::Tetra4);

    EXPECT_EQ(item.face_id, 10);
    EXPECT_EQ(item.minus_cell_id, 5);
    EXPECT_EQ(item.plus_cell_id, 6);
    EXPECT_EQ(item.minus_local_face_id, 2u);
    EXPECT_EQ(item.plus_local_face_id, 3u);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(AssemblyLoopTest, LoopStatisticsDefaults) {
    LoopStatistics stats;

    EXPECT_EQ(stats.total_iterations, 0);
    EXPECT_EQ(stats.skipped_iterations, 0);
    EXPECT_DOUBLE_EQ(stats.elapsed_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.kernel_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.insert_seconds, 0.0);
    EXPECT_EQ(stats.num_threads_used, 1);
}

// ============================================================================
// Coloring Tests
// ============================================================================

TEST_F(AssemblyLoopTest, SetColoring) {
    AssemblyLoop loop;

    std::vector<int> colors(100);
    for (int i = 0; i < 100; ++i) {
        colors[static_cast<std::size_t>(i)] = i % 5;  // 5 colors
    }

    loop.setColoring(colors, 5);

    EXPECT_TRUE(loop.hasColoring());
}

TEST_F(AssemblyLoopTest, NoColoringByDefault) {
    AssemblyLoop loop;
    EXPECT_FALSE(loop.hasColoring());
}

// ============================================================================
// Free Function Tests
// ============================================================================

TEST_F(AssemblyLoopTest, ForEachCellFunction) {
    GlobalIndex count = 0;

    forEachCell(*mesh_, [&count](GlobalIndex /*cell_id*/) {
        ++count;
    }, false);  // all cells

    EXPECT_EQ(count, mesh_->numCells());
}

TEST_F(AssemblyLoopTest, ForEachOwnedCellFunction) {
    GlobalIndex count = 0;

    forEachCell(*mesh_, [&count](GlobalIndex /*cell_id*/) {
        ++count;
    }, true);  // owned only

    // Should be 75% of cells
    EXPECT_EQ(count, mesh_->numCells() * 3 / 4);
}

TEST_F(AssemblyLoopTest, ForEachBoundaryFaceFunction) {
    GlobalIndex count = 0;

    forEachBoundaryFace(*mesh_, 1, [&count](GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) {
        ++count;
    });

    EXPECT_EQ(count, 10);  // Mock returns 10 faces for marker 1
}

TEST_F(AssemblyLoopTest, ForEachInteriorFaceFunction) {
    GlobalIndex count = 0;

    forEachInteriorFace(*mesh_, [&count](GlobalIndex /*face_id*/,
                                         GlobalIndex /*minus*/,
                                         GlobalIndex /*plus*/) {
        ++count;
    });

    EXPECT_EQ(count, mesh_->numInteriorFaces());
}

// ============================================================================
// Element Graph Coloring Tests
// ============================================================================

TEST_F(AssemblyLoopTest, ComputeElementColoring) {
    // Note: Full test requires DofMap implementation
    // This is a placeholder test
    EXPECT_TRUE(true);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
