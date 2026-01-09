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
#include "Assembly/GlobalSystemView.h"
#include "Core/Types.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"

#include <vector>
#include <memory>
#include <cmath>
#include <array>
#include <algorithm>
#include <unordered_map>

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

static dofs::DofMap buildSharedChainDofMap(GlobalIndex num_cells)
{
    // Chain: each cell shares one DOF with the next.
    // cell i: [3*i, 3*i+1, 3*i+2, 3*i+3]
    // Shared DOF between i and i+1 is 3*i+3.
    const GlobalIndex n_dofs = num_cells * 3 + 1;
    dofs::DofMap dof_map(num_cells, n_dofs, 4);

    for (GlobalIndex cell = 0; cell < num_cells; ++cell) {
        const std::array<GlobalIndex, 4> dofs = {
            3 * cell + 0,
            3 * cell + 1,
            3 * cell + 2,
            3 * cell + 3,
        };
        dof_map.setCellDofs(cell, dofs);
    }

    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

static std::vector<Real> runCellLoopAccumulateOnDofs(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    const spaces::FunctionSpace& space,
    LoopMode mode,
    bool deterministic,
    bool prefetch_next,
    std::span<const int> colors,
    int num_colors,
    LoopStatistics* out_stats = nullptr)
{
    AssemblyLoop loop;
    loop.setMesh(mesh);
    loop.setDofMap(dof_map);

    LoopOptions opts;
    opts.mode = mode;
    opts.num_threads = 4;
    opts.deterministic = deterministic;
    opts.prefetch_next = prefetch_next;
    loop.setOptions(opts);

    if (mode == LoopMode::Colored) {
        loop.setColoring(colors, num_colors);
    }

    const GlobalIndex n = dof_map.getNumDofs();
    std::vector<Real> y(static_cast<std::size_t>(n), 0.0);

    const auto stats = loop.cellLoop(
        space, space, RequiredData::None,
        [](const CellWorkItem& /*cell*/, AssemblyContext& ctx, KernelOutput& out) {
            const LocalIndex n_dofs = ctx.numTestDofs();
            out.reserve(n_dofs, n_dofs, /*want_matrix=*/false, /*want_vector=*/true);
            out.clear();
            for (LocalIndex i = 0; i < n_dofs; ++i) {
                out.vectorEntry(i) = 1.0;
            }
        },
        [&y](const CellWorkItem& /*cell*/,
             const KernelOutput& out,
             std::span<const GlobalIndex> row_dofs,
             std::span<const GlobalIndex> /*col_dofs*/) {
            for (std::size_t i = 0; i < row_dofs.size(); ++i) {
                y[static_cast<std::size_t>(row_dofs[i])] += out.local_vector[i];
            }
        });

    if (out_stats != nullptr) {
        *out_stats = stats;
    }

    return y;
}

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
    EXPECT_EQ(stats.prefetch_hints, 0u);
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

TEST_F(AssemblyLoopTest, ComputeElementColoringProducesValidColoring) {
    const GlobalIndex n_cells = 16;
    MockMeshAccess mesh(n_cells);
    auto dof_map = buildSharedChainDofMap(n_cells);

    std::vector<int> colors;
    const int num_colors = computeElementColoring(mesh, dof_map, colors);

    ASSERT_EQ(colors.size(), static_cast<std::size_t>(n_cells));
    EXPECT_GT(num_colors, 0);
    EXPECT_LT(num_colors, 20);  // "reasonable" for small meshes

    // Verify: for every DOF shared by multiple cells, all incident cells have different colors.
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> dof_to_cells;
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        for (GlobalIndex dof : dof_map.getCellDofs(cell)) {
            dof_to_cells[dof].push_back(cell);
        }
    }

    for (const auto& [dof, cells] : dof_to_cells) {
        (void)dof;
        if (cells.size() < 2) {
            continue;
        }
        for (std::size_t i = 0; i < cells.size(); ++i) {
            for (std::size_t j = i + 1; j < cells.size(); ++j) {
                EXPECT_NE(colors[static_cast<std::size_t>(cells[i])],
                          colors[static_cast<std::size_t>(cells[j])]);
            }
        }
    }

    // Coloring quality metric (max/min elements per color).
    std::vector<int> counts(static_cast<std::size_t>(num_colors), 0);
    for (int c : colors) {
        ASSERT_GE(c, 0);
        ASSERT_LT(c, num_colors);
        counts[static_cast<std::size_t>(c)]++;
    }
    const auto [min_it, max_it] = std::minmax_element(counts.begin(), counts.end());
    ASSERT_NE(*min_it, 0);
    const double imbalance = static_cast<double>(*max_it) / static_cast<double>(*min_it);
    EXPECT_LE(imbalance, 2.0);
}

TEST_F(AssemblyLoopTest, ComputeOptimizedColoringDoesNotIncreaseColors) {
    const GlobalIndex n_cells = 25;
    MockMeshAccess mesh(n_cells);
    auto dof_map = buildSharedChainDofMap(n_cells);

    std::vector<int> greedy_colors;
    const int greedy = computeElementColoring(mesh, dof_map, greedy_colors);

    std::vector<int> opt_colors;
    const int opt = computeOptimizedColoring(mesh, dof_map, opt_colors, /*max_iterations=*/20);

    EXPECT_LE(opt, greedy);
    EXPECT_EQ(opt_colors.size(), static_cast<std::size_t>(n_cells));
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
// Parallel Correctness Tests
// ============================================================================

TEST_F(AssemblyLoopTest, CellLoopOpenMPProducesSameResultAsSequential) {
#ifdef _OPENMP
    const GlobalIndex n_cells = 64;
    MockMeshAccess mesh(n_cells);
    auto dof_map = buildSharedChainDofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto y_seq = runCellLoopAccumulateOnDofs(
        mesh, dof_map, space, LoopMode::Sequential,
        /*deterministic=*/true, /*prefetch_next=*/false,
        std::span<const int>{}, 0);

    const auto y_omp = runCellLoopAccumulateOnDofs(
        mesh, dof_map, space, LoopMode::OpenMP,
        /*deterministic=*/true, /*prefetch_next=*/false,
        std::span<const int>{}, 0);

    EXPECT_EQ(y_omp, y_seq);
#else
    GTEST_SKIP() << "OpenMP not enabled";
#endif
}

TEST_F(AssemblyLoopTest, CellLoopColoredProducesSameResultAsSequential) {
#ifdef _OPENMP
    const GlobalIndex n_cells = 64;
    MockMeshAccess mesh(n_cells);
    auto dof_map = buildSharedChainDofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    std::vector<int> colors;
    const int num_colors = computeElementColoring(mesh, dof_map, colors);

    const auto y_seq = runCellLoopAccumulateOnDofs(
        mesh, dof_map, space, LoopMode::Sequential,
        /*deterministic=*/true, /*prefetch_next=*/false,
        std::span<const int>{}, 0);

    const auto y_colored = runCellLoopAccumulateOnDofs(
        mesh, dof_map, space, LoopMode::Colored,
        /*deterministic=*/true, /*prefetch_next=*/false,
        std::span<const int>(colors.data(), colors.size()), num_colors);

    EXPECT_EQ(y_colored, y_seq);
#else
    GTEST_SKIP() << "OpenMP not enabled";
#endif
}

TEST_F(AssemblyLoopTest, DeterministicParallelAssemblyIsReproducible) {
#ifdef _OPENMP
    const GlobalIndex n_cells = 64;
    MockMeshAccess mesh(n_cells);
    auto dof_map = buildSharedChainDofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto y1 = runCellLoopAccumulateOnDofs(
        mesh, dof_map, space, LoopMode::OpenMP,
        /*deterministic=*/true, /*prefetch_next=*/false,
        std::span<const int>{}, 0);
    const auto y2 = runCellLoopAccumulateOnDofs(
        mesh, dof_map, space, LoopMode::OpenMP,
        /*deterministic=*/true, /*prefetch_next=*/false,
        std::span<const int>{}, 0);

    EXPECT_EQ(y1, y2);
#else
    GTEST_SKIP() << "OpenMP not enabled";
#endif
}

// ============================================================================
// Unified Loop Tests
// ============================================================================

namespace {

class UnifiedKernel final : public AssemblyKernel {
public:
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return true; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return true; }
    [[nodiscard]] RequiredData getRequiredData() const noexcept override { return RequiredData::None; }

    void computeCell(const AssemblyContext& ctx, KernelOutput& out) override {
        const LocalIndex n = ctx.numTestDofs();
        out.reserve(n, n, /*want_matrix=*/true, /*want_vector=*/false);
        out.clear();
        for (LocalIndex i = 0; i < n; ++i) {
            out.matrixEntry(i, i) = 1.0;
        }
    }

    void computeBoundaryFace(const AssemblyContext& ctx, int /*marker*/, KernelOutput& out) override {
        const LocalIndex n = ctx.numTestDofs();
        out.reserve(n, n, /*want_matrix=*/true, /*want_vector=*/false);
        out.clear();
        for (LocalIndex i = 0; i < n; ++i) {
            out.matrixEntry(i, i) = 10.0;
        }
    }

    void computeInteriorFace(const AssemblyContext& ctx_minus,
                             const AssemblyContext& ctx_plus,
                             KernelOutput& out_minus,
                             KernelOutput& out_plus,
                             KernelOutput& coupling_mp,
                             KernelOutput& coupling_pm) override {
        const LocalIndex nm = ctx_minus.numTestDofs();
        const LocalIndex np = ctx_plus.numTestDofs();

        out_minus.reserve(nm, nm, /*want_matrix=*/true, /*want_vector=*/false);
        out_plus.reserve(np, np, /*want_matrix=*/true, /*want_vector=*/false);
        coupling_mp.reserve(nm, np, /*want_matrix=*/true, /*want_vector=*/false);
        coupling_pm.reserve(np, nm, /*want_matrix=*/true, /*want_vector=*/false);

        out_minus.clear();
        out_plus.clear();
        coupling_mp.clear();
        coupling_pm.clear();

        for (LocalIndex i = 0; i < nm; ++i) {
            out_minus.matrixEntry(i, i) = 100.0;
        }
        for (LocalIndex i = 0; i < np; ++i) {
            out_plus.matrixEntry(i, i) = 100.0;
        }

        if (nm > 0 && np > 0) {
            coupling_mp.matrixEntry(0, 0) = 0.5;
            coupling_pm.matrixEntry(0, 0) = 0.5;
        }
    }
};

} // namespace

TEST_F(AssemblyLoopTest, UnifiedLoopCellAndBoundaryAssemblesContributions) {
    const GlobalIndex n_cells = 100;
    MockMeshAccess mesh(n_cells);
    auto dof_map = buildSharedChainDofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    DenseMatrixView A(dof_map.getNumDofs());
    UnifiedKernel kernel;
    const std::array<int, 1> boundary_markers = {1};

    AssemblyLoop loop;
    loop.setMesh(mesh);
    loop.setDofMap(dof_map);

    LoopOptions opts;
    opts.mode = LoopMode::Sequential;
    loop.setOptions(opts);

    loop.unifiedLoop(space, space, kernel, /*boundary_markers=*/boundary_markers,
                     /*include_interior_faces=*/false, &A, nullptr);

    // Cell 0 has a boundary face in this mock => diagonal includes 1 + 10.
    EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 0), 11.0);

    // Interior coupling should not be present.
    EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 3), 0.0);
}

TEST_F(AssemblyLoopTest, UnifiedLoopCellBoundaryAndInteriorAssemblesDGContributions) {
    const GlobalIndex n_cells = 100;
    MockMeshAccess mesh(n_cells);

    // Use disjoint DOFs for predictable global indexing in the DG coupling checks.
    dofs::DofMap dof_map(n_cells, n_cells * 4, 4);
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        const std::array<GlobalIndex, 4> dofs = {
            4 * cell + 0,
            4 * cell + 1,
            4 * cell + 2,
            4 * cell + 3,
        };
        dof_map.setCellDofs(cell, dofs);
    }
    dof_map.setNumDofs(n_cells * 4);
    dof_map.setNumLocalDofs(n_cells * 4);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
    DenseMatrixView A(dof_map.getNumDofs());
    UnifiedKernel kernel;
    const std::array<int, 1> boundary_markers = {1};

    AssemblyLoop loop;
    loop.setMesh(mesh);
    loop.setDofMap(dof_map);
    LoopOptions opts;
    opts.mode = LoopMode::Sequential;
    loop.setOptions(opts);

    loop.unifiedLoop(space, space, kernel, /*boundary_markers=*/boundary_markers,
                     /*include_interior_faces=*/true, &A, nullptr);

    // Cell 0 appears in 3 interior faces in this mock (minus twice, plus once):
    // diag = cell(1) + boundary(10) + 3 * 100
    EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 0), 311.0);

    // Cell 50 has the same number of interior contributions but no boundary face:
    // diag = 1 + 3 * 100
    EXPECT_DOUBLE_EQ(A.getMatrixEntry(200, 200), 301.0);

    // Coupling between cell 0 and cell 1 happens twice (faces 0 and 100).
    EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 4), 1.0);
    EXPECT_DOUBLE_EQ(A.getMatrixEntry(4, 0), 1.0);
}

// ============================================================================
// Prefetch Tests
// ============================================================================

TEST_F(AssemblyLoopTest, PrefetchHintGenerationInSequentialLoop) {
    const GlobalIndex n_cells = 32;
    MockMeshAccess mesh(n_cells);
    auto dof_map = buildSharedChainDofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    LoopStatistics stats{};
    (void)runCellLoopAccumulateOnDofs(
        mesh, dof_map, space, LoopMode::Sequential,
        /*deterministic=*/true, /*prefetch_next=*/true,
        std::span<const int>{}, 0, &stats);

    EXPECT_GT(stats.prefetch_hints, 0u);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
