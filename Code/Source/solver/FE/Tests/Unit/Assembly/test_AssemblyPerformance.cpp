/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyPerformance.cpp
 * @brief Optional performance smoke tests for FE/Assembly
 *
 * These tests are skipped by default. Enable by setting:
 *   SVMP_FE_RUN_PERF_TESTS=1
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyLoop.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string_view>
#include <limits>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

bool perfTestsEnabled()
{
    const char* v = std::getenv("SVMP_FE_RUN_PERF_TESTS");
    return v != nullptr && std::string_view(v) == "1";
}

#if defined(__unix__) || defined(__APPLE__)
std::size_t peakRssKilobytes()
{
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0u;
    }
    // On Linux, ru_maxrss is kilobytes. On macOS it is bytes. We report raw.
    return static_cast<std::size_t>(usage.ru_maxrss);
}
#endif

class PerfMeshAccess final : public IMeshAccess {
public:
    explicit PerfMeshAccess(GlobalIndex num_cells)
        : num_cells_(num_cells)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        nodes.clear();
        nodes.push_back(cell_id * 4 + 0);
        nodes.push_back(cell_id * 4 + 1);
        nodes.push_back(cell_id * 4 + 2);
        nodes.push_back(cell_id * 4 + 3);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        const GlobalIndex cell_id = node_id / 4;
        const int local = static_cast<int>(node_id % 4);
        const Real x0 = static_cast<Real>(cell_id) * 2.0;

        switch (local) {
            case 0: return {x0, 0.0, 0.0};
            case 1: return {x0 + 1.0, 0.0, 0.0};
            case 2: return {x0, 1.0, 0.0};
            case 3: return {x0, 0.0, 1.0};
            default: return {x0, 0.0, 0.0};
        }
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override {
        coords.clear();
        coords.push_back(getNodeCoordinates(cell_id * 4 + 0));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 1));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 2));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 3));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < num_cells_; ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    GlobalIndex num_cells_{0};
};

static dofs::DofMap buildDisjointTetra4DofMap(GlobalIndex num_cells)
{
    dofs::DofMap dof_map(num_cells, /*n_dofs_total=*/num_cells * 4, /*dofs_per_cell=*/4);
    for (GlobalIndex cell = 0; cell < num_cells; ++cell) {
        const std::array<GlobalIndex, 4> dofs = {
            cell * 4 + 0,
            cell * 4 + 1,
            cell * 4 + 2,
            cell * 4 + 3,
        };
        dof_map.setCellDofs(cell, dofs);
    }
    dof_map.setNumDofs(num_cells * 4);
    dof_map.setNumLocalDofs(num_cells * 4);
    dof_map.finalize();
    return dof_map;
}

static void fillSmallKernel(const AssemblyContext& ctx, KernelOutput& output)
{
    const LocalIndex n = ctx.numTestDofs();
    output.reserve(n, n, true, false);
    output.clear();
    for (LocalIndex i = 0; i < n; ++i) {
        output.matrixEntry(i, i) = 1.0;
    }
}

static void fillFlopsKernel(const AssemblyContext& ctx, KernelOutput& output)
{
    const LocalIndex n = ctx.numTestDofs();
    output.reserve(n, n, true, false);
    output.clear();
    for (LocalIndex i = 0; i < n; ++i) {
        for (LocalIndex j = 0; j < n; ++j) {
            const Real a = static_cast<Real>(i + 1);
            const Real b = static_cast<Real>(j + 1);
            output.matrixEntry(i, j) = a * b + a - b;
        }
    }
}

} // namespace

TEST(AssemblyPerformanceTest, ElementsPerSecondSequentialCellLoop) {
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const GlobalIndex num_cells = 5000;
    PerfMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(mesh.numCells());
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    AssemblyLoop loop;
    loop.setMesh(mesh);
    loop.setDofMap(dof_map);
    LoopOptions opts;
    opts.mode = LoopMode::Sequential;
    opts.num_threads = 1;
    loop.setOptions(opts);

    const auto stats = loop.cellLoop(
        space, space, RequiredData::None,
        [](const CellWorkItem& /*cell*/, AssemblyContext& ctx, KernelOutput& out) {
            fillSmallKernel(ctx, out);
        },
        [](const CellWorkItem& /*cell*/, const KernelOutput& /*out*/,
           std::span<const GlobalIndex> /*r*/, std::span<const GlobalIndex> /*c*/) {});

    ASSERT_GT(stats.elapsed_seconds, 0.0);
    const double elements_per_second =
        static_cast<double>(stats.total_iterations) / stats.elapsed_seconds;
    EXPECT_GT(elements_per_second, 0.0);
    RecordProperty("elements_per_second", elements_per_second);
    RecordProperty("dofs_per_second",
                   static_cast<double>(dof_map.getNumDofs()) / stats.elapsed_seconds);
}

TEST(AssemblyPerformanceTest, FlopsAndBandwidthEstimatesSequentialCellLoop) {
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const GlobalIndex num_cells = 5000;
    PerfMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(mesh.numCells());
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    AssemblyLoop loop;
    loop.setMesh(mesh);
    loop.setDofMap(dof_map);
    LoopOptions opts;
    opts.mode = LoopMode::Sequential;
    opts.num_threads = 1;
    loop.setOptions(opts);

    const auto stats = loop.cellLoop(
        space, space, RequiredData::None,
        [](const CellWorkItem& /*cell*/, AssemblyContext& ctx, KernelOutput& out) {
            fillFlopsKernel(ctx, out);
        },
        [](const CellWorkItem& /*cell*/, const KernelOutput& /*out*/,
           std::span<const GlobalIndex> /*r*/, std::span<const GlobalIndex> /*c*/) {});

    ASSERT_GT(stats.elapsed_seconds, 0.0);

    // For Tetra4/H1(order=1): 4 test dofs, kernel executes 3 FLOPs per entry.
    constexpr std::size_t dofs_per_cell = 4;
    constexpr std::size_t flops_per_entry = 3;
    const double flops_per_cell =
        static_cast<double>(dofs_per_cell * dofs_per_cell * flops_per_entry);
    const double flops_per_second =
        flops_per_cell * static_cast<double>(stats.total_iterations) / stats.elapsed_seconds;
    RecordProperty("flops_per_second_estimate", flops_per_second);

    const double bytes_per_cell =
        static_cast<double>(dofs_per_cell * dofs_per_cell * sizeof(Real));
    const double bytes_per_second =
        bytes_per_cell * static_cast<double>(stats.total_iterations) / stats.elapsed_seconds;
    RecordProperty("bytes_per_second_estimate", bytes_per_second);

    const double bytes_per_flop =
        (flops_per_second > 0.0) ? (bytes_per_second / flops_per_second) : 0.0;
    RecordProperty("bytes_per_flop_estimate", bytes_per_flop);

    EXPECT_GT(flops_per_second, 0.0);
    EXPECT_GT(bytes_per_second, 0.0);
    EXPECT_GE(bytes_per_flop, 0.0);
}

#ifdef _OPENMP
TEST(AssemblyPerformanceTest, StrongScalingOpenMPCellLoop) {
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const GlobalIndex num_cells = 5000;
    PerfMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(mesh.numCells());
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    auto run = [&](LoopMode mode, int threads) {
        AssemblyLoop loop;
        loop.setMesh(mesh);
        loop.setDofMap(dof_map);
        LoopOptions opts;
        opts.mode = mode;
        opts.num_threads = threads;
        loop.setOptions(opts);
        return loop.cellLoop(
            space, space, RequiredData::None,
            [](const CellWorkItem& /*cell*/, AssemblyContext& ctx, KernelOutput& out) {
                fillSmallKernel(ctx, out);
            },
            [](const CellWorkItem& /*cell*/, const KernelOutput& /*out*/,
               std::span<const GlobalIndex> /*r*/, std::span<const GlobalIndex> /*c*/) {});
    };

    const auto seq = run(LoopMode::Sequential, 1);
    const auto omp = run(LoopMode::OpenMP, 0);

    ASSERT_GT(seq.elapsed_seconds, 0.0);
    ASSERT_GT(omp.elapsed_seconds, 0.0);

    const double speedup = seq.elapsed_seconds / omp.elapsed_seconds;
    EXPECT_GT(speedup, 0.0);
    RecordProperty("strong_scaling_speedup", speedup);
}

TEST(AssemblyPerformanceTest, WeakScalingOpenMPCellLoop) {
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int max_threads = omp_get_max_threads();
    if (max_threads < 2) {
        GTEST_SKIP() << "OpenMP max threads < 2";
    }

    const GlobalIndex base_cells = 2000;
    PerfMeshAccess mesh_seq(base_cells);
    auto dof_map_seq = buildDisjointTetra4DofMap(mesh_seq.numCells());
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    auto run = [&](const IMeshAccess& mesh, const dofs::DofMap& dof_map, LoopMode mode, int threads) {
        AssemblyLoop loop;
        loop.setMesh(mesh);
        loop.setDofMap(dof_map);
        LoopOptions opts;
        opts.mode = mode;
        opts.num_threads = threads;
        loop.setOptions(opts);
        return loop.cellLoop(
            space, space, RequiredData::None,
            [](const CellWorkItem& /*cell*/, AssemblyContext& ctx, KernelOutput& out) {
                fillSmallKernel(ctx, out);
            },
            [](const CellWorkItem& /*cell*/, const KernelOutput& /*out*/,
               std::span<const GlobalIndex> /*r*/, std::span<const GlobalIndex> /*c*/) {});
    };

    const auto seq = run(mesh_seq, dof_map_seq, LoopMode::Sequential, 1);
    ASSERT_GT(seq.elapsed_seconds, 0.0);
    const double seq_eps = static_cast<double>(seq.total_iterations) / seq.elapsed_seconds;

    const GlobalIndex omp_cells = base_cells * static_cast<GlobalIndex>(max_threads);
    PerfMeshAccess mesh_omp(omp_cells);
    auto dof_map_omp = buildDisjointTetra4DofMap(mesh_omp.numCells());
    const auto omp_stats = run(mesh_omp, dof_map_omp, LoopMode::OpenMP, max_threads);
    ASSERT_GT(omp_stats.elapsed_seconds, 0.0);
    const double omp_eps =
        static_cast<double>(omp_stats.total_iterations) / omp_stats.elapsed_seconds;

    const double weak_efficiency = omp_eps / (seq_eps * static_cast<double>(max_threads));
    RecordProperty("weak_scaling_efficiency", weak_efficiency);
    EXPECT_GT(weak_efficiency, 0.0);
}
#else
TEST(AssemblyPerformanceTest, StrongScalingOpenMPCellLoop) {
    GTEST_SKIP() << "OpenMP not enabled";
}
#endif

#if defined(__unix__) || defined(__APPLE__)
TEST(AssemblyPerformanceTest, PeakMemoryUsageSmoke) {
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const GlobalIndex num_cells = 10000;
    PerfMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(mesh.numCells());
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    AssemblyLoop loop;
    loop.setMesh(mesh);
    loop.setDofMap(dof_map);
    LoopOptions opts;
    opts.mode = LoopMode::Sequential;
    opts.num_threads = 1;
    loop.setOptions(opts);

    (void)loop.cellLoop(
        space, space, RequiredData::None,
        [](const CellWorkItem& /*cell*/, AssemblyContext& ctx, KernelOutput& out) {
            fillSmallKernel(ctx, out);
        },
        [](const CellWorkItem& /*cell*/, const KernelOutput& /*out*/,
           std::span<const GlobalIndex> /*r*/, std::span<const GlobalIndex> /*c*/) {});

    const std::size_t rss = peakRssKilobytes();
    RecordProperty("peak_rss_raw", rss);
    EXPECT_GE(rss, 0u);
}
#endif

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
