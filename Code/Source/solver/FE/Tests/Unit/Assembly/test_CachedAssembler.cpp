/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_CachedAssembler.cpp
 * @brief Unit tests for CachedAssembler and element matrix caching
 */

#include <gtest/gtest.h>

#include "Assembly/CachedAssembler.h"
#include "Assembly/AssemblyContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"

#include <cmath>
#include <array>
#include <thread>
#include <atomic>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

namespace {

class SimpleMeshAccess final : public IMeshAccess {
public:
    explicit SimpleMeshAccess(GlobalIndex num_cells) : num_cells_(num_cells) {}

    [[nodiscard]] GlobalIndex numCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        nodes.resize(4);
        const GlobalIndex base = static_cast<GlobalIndex>(cell_id * 4);
        nodes[0] = base + 0;
        nodes[1] = base + 1;
        nodes[2] = base + 2;
        nodes[3] = base + 3;
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        const GlobalIndex cell_id = node_id / 4;
        const GlobalIndex local_id = node_id % 4;
        const Real offset = static_cast<Real>(cell_id) * 2.0;
        switch (local_id) {
        case 0:
            return {offset, 0.0, 0.0};
        case 1:
            return {offset + 1.0, 0.0, 0.0};
        case 2:
            return {offset, 1.0, 0.0};
        default:
            return {offset, 0.0, 1.0};
        }
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override {
        coords.resize(4);
        const Real offset = static_cast<Real>(cell_id) * 2.0;
        coords[0] = {offset, 0.0, 0.0};
        coords[1] = {offset + 1.0, 0.0, 0.0};
        coords[2] = {offset, 1.0, 0.0};
        coords[3] = {offset, 0.0, 1.0};
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex c = 0; c < num_cells_; ++c) {
            callback(c);
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

class CountingIdentityKernel final : public AssemblyKernel {
public:
    void computeCell(const AssemblyContext& ctx, KernelOutput& out) override
    {
        ++compute_calls_;
        const LocalIndex n = ctx.numTestDofs();
        out.reserve(n, n, /*want_matrix=*/true, /*want_vector=*/false);
        out.clear();
        for (LocalIndex i = 0; i < n; ++i) {
            out.matrixEntry(i, i) = 1.0;
        }
    }

    [[nodiscard]] RequiredData getRequiredData() const noexcept override { return RequiredData::None; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return true; }

    int computeCalls() const noexcept { return compute_calls_.load(); }

private:
    std::atomic<int> compute_calls_{0};
};

} // namespace

// ============================================================================
// CacheStrategy Tests
// ============================================================================

TEST(CacheStrategyTest, EnumValues) {
    EXPECT_NE(CacheStrategy::FullMatrix, CacheStrategy::ReferenceElement);
    EXPECT_NE(CacheStrategy::ReferenceElement, CacheStrategy::GeometricFactors);
    EXPECT_NE(CacheStrategy::GeometricFactors, CacheStrategy::None);
}

// ============================================================================
// EvictionPolicy Tests
// ============================================================================

TEST(EvictionPolicyTest, EnumValues) {
    EXPECT_NE(EvictionPolicy::LRU, EvictionPolicy::FIFO);
    EXPECT_NE(EvictionPolicy::FIFO, EvictionPolicy::None);
}

// ============================================================================
// CacheOptions Tests
// ============================================================================

TEST(CacheOptionsTest, Defaults) {
    CacheOptions options;

    EXPECT_EQ(options.strategy, CacheStrategy::FullMatrix);
    EXPECT_EQ(options.eviction, EvictionPolicy::LRU);
    EXPECT_EQ(options.max_memory_bytes, 0u);  // Unlimited
    EXPECT_EQ(options.max_elements, 0u);      // Unlimited
    EXPECT_TRUE(options.auto_invalidate_on_mesh_change);
    EXPECT_TRUE(options.parallel_population);
    EXPECT_EQ(options.num_threads, 4);
    EXPECT_FALSE(options.verify_cache);
    EXPECT_NEAR(options.verify_tolerance, 1e-12, 1e-15);
}

TEST(CacheOptionsTest, CustomValues) {
    CacheOptions options;
    options.strategy = CacheStrategy::ReferenceElement;
    options.eviction = EvictionPolicy::FIFO;
    options.max_memory_bytes = 1024 * 1024;  // 1 MB
    options.max_elements = 1000;
    options.num_threads = 8;

    EXPECT_EQ(options.strategy, CacheStrategy::ReferenceElement);
    EXPECT_EQ(options.eviction, EvictionPolicy::FIFO);
    EXPECT_EQ(options.max_memory_bytes, 1024u * 1024u);
    EXPECT_EQ(options.max_elements, 1000u);
    EXPECT_EQ(options.num_threads, 8);
}

// ============================================================================
// CacheStats Tests
// ============================================================================

TEST(CacheStatsTest, DefaultValues) {
    CacheStats stats;

    EXPECT_EQ(stats.total_elements, 0u);
    EXPECT_EQ(stats.memory_bytes, 0u);
    EXPECT_EQ(stats.cache_hits, 0u);
    EXPECT_EQ(stats.cache_misses, 0u);
    EXPECT_EQ(stats.evictions, 0u);
    EXPECT_EQ(stats.invalidations, 0u);
    EXPECT_DOUBLE_EQ(stats.population_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.hit_rate, 0.0);
}

TEST(CacheStatsTest, UpdateHitRate) {
    CacheStats stats;
    stats.cache_hits = 80;
    stats.cache_misses = 20;

    stats.updateHitRate();

    EXPECT_DOUBLE_EQ(stats.hit_rate, 0.8);
}

TEST(CacheStatsTest, UpdateHitRateZero) {
    CacheStats stats;
    stats.cache_hits = 0;
    stats.cache_misses = 0;

    stats.updateHitRate();

    EXPECT_DOUBLE_EQ(stats.hit_rate, 0.0);
}

TEST(CacheStatsTest, UpdateHitRatePerfect) {
    CacheStats stats;
    stats.cache_hits = 100;
    stats.cache_misses = 0;

    stats.updateHitRate();

    EXPECT_DOUBLE_EQ(stats.hit_rate, 1.0);
}

// ============================================================================
// CachedElementData Tests
// ============================================================================

TEST(CachedElementDataTest, DefaultConstruction) {
    CachedElementData data;

    EXPECT_TRUE(data.local_matrix.empty());
    EXPECT_TRUE(data.local_vector.empty());
    EXPECT_TRUE(data.row_dofs.empty());
    EXPECT_TRUE(data.col_dofs.empty());
    EXPECT_EQ(data.element_type, ElementType::Unknown);
    EXPECT_FALSE(data.has_matrix);
    EXPECT_FALSE(data.has_vector);
    EXPECT_EQ(data.last_access, 0u);
}

TEST(CachedElementDataTest, MemoryBytes) {
    CachedElementData data;

    // Empty data
    std::size_t base_size = data.memoryBytes();
    EXPECT_GT(base_size, 0u);  // At least sizeof(CachedElementData)

    // Add matrix data
    data.local_matrix.resize(16, 1.0);
    std::size_t with_matrix = data.memoryBytes();
    EXPECT_GT(with_matrix, base_size);

    // Add vector data
    data.local_vector.resize(4, 0.0);
    std::size_t with_vector = data.memoryBytes();
    EXPECT_GT(with_vector, with_matrix);
}

TEST(CachedElementDataTest, PopulatedData) {
    CachedElementData data;

    // Populate with 4x4 element matrix
    data.local_matrix.resize(16, 0.0);
    for (int i = 0; i < 4; ++i) {
        data.local_matrix[static_cast<std::size_t>(i * 4 + i)] = 2.0;  // Diagonal = 2
    }
    data.has_matrix = true;

    data.row_dofs = {10, 11, 12, 13};
    data.col_dofs = {10, 11, 12, 13};
    data.element_type = ElementType::Tetra4;
    data.last_access = 42;

    EXPECT_TRUE(data.has_matrix);
    EXPECT_EQ(data.local_matrix.size(), 16u);
    EXPECT_DOUBLE_EQ(data.local_matrix[0], 2.0);
    EXPECT_EQ(data.row_dofs.size(), 4u);
    EXPECT_EQ(data.element_type, ElementType::Tetra4);
    EXPECT_EQ(data.last_access, 42u);
}

// ============================================================================
// ElementMatrixCache Tests
// ============================================================================

TEST(ElementMatrixCacheTest, DefaultConstruction) {
    ElementMatrixCache cache;

    EXPECT_EQ(cache.memoryUsage(), 0u);
}

TEST(ElementMatrixCacheTest, ConstructionWithOptions) {
    CacheOptions options;
    options.max_elements = 100;

    ElementMatrixCache cache(options);

    EXPECT_EQ(cache.getOptions().max_elements, 100u);
}

TEST(ElementMatrixCacheTest, SetOptions) {
    ElementMatrixCache cache;

    CacheOptions options;
    options.strategy = CacheStrategy::GeometricFactors;

    cache.setOptions(options);

    EXPECT_EQ(cache.getOptions().strategy, CacheStrategy::GeometricFactors);
}

TEST(ElementMatrixCacheTest, InsertAndContains) {
    ElementMatrixCache cache;

    CachedElementData data;
    data.local_matrix.resize(16, 1.0);
    data.has_matrix = true;

    bool inserted = cache.insert(0, data);
    EXPECT_TRUE(inserted);
    EXPECT_TRUE(cache.contains(0));
    EXPECT_FALSE(cache.contains(1));
}

TEST(ElementMatrixCacheTest, Get) {
    ElementMatrixCache cache;

    CachedElementData data;
    data.local_matrix.resize(9, 0.0);
    data.local_matrix[4] = 5.0;  // Center of 3x3
    data.has_matrix = true;

    cache.insert(42, data);

    const CachedElementData* retrieved = cache.get(42);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_DOUBLE_EQ(retrieved->local_matrix[4], 5.0);

    // Non-existent
    EXPECT_EQ(cache.get(999), nullptr);
}

TEST(ElementMatrixCacheTest, Invalidate) {
    ElementMatrixCache cache;

    CachedElementData data;
    data.local_matrix.resize(4, 1.0);
    data.has_matrix = true;

    cache.insert(0, data);
    cache.insert(1, data);

    EXPECT_TRUE(cache.contains(0));
    EXPECT_TRUE(cache.contains(1));

    cache.invalidate(0);

    EXPECT_FALSE(cache.contains(0));
    EXPECT_TRUE(cache.contains(1));
}

TEST(ElementMatrixCacheTest, InvalidateAll) {
    ElementMatrixCache cache;

    CachedElementData data;
    data.local_matrix.resize(4, 1.0);
    data.has_matrix = true;

    cache.insert(0, data);
    cache.insert(1, data);
    cache.insert(2, data);

    cache.invalidateAll();

    EXPECT_FALSE(cache.contains(0));
    EXPECT_FALSE(cache.contains(1));
    EXPECT_FALSE(cache.contains(2));
}

TEST(ElementMatrixCacheTest, Clear) {
    ElementMatrixCache cache;

    CachedElementData data;
    data.local_matrix.resize(4, 1.0);
    data.has_matrix = true;

    cache.insert(0, data);
    cache.insert(1, data);

    cache.clear();

    EXPECT_EQ(cache.memoryUsage(), 0u);
}

TEST(ElementMatrixCacheTest, Statistics) {
    ElementMatrixCache cache;

    CachedElementData data;
    data.local_matrix.resize(16, 1.0);
    data.has_matrix = true;

    cache.insert(0, data);
    cache.insert(1, data);

    CacheStats stats = cache.getStats();
    EXPECT_GE(stats.total_elements, 2u);
}

TEST(ElementMatrixCacheTest, Reserve) {
    ElementMatrixCache cache;

    cache.reserve(1000);  // Reserve for 1000 elements

    // Should not throw
    SUCCEED();
}

TEST(ElementMatrixCacheTest, ResetStats) {
    ElementMatrixCache cache;

    CachedElementData data;
    data.local_matrix.resize(4, 1.0);
    data.has_matrix = true;

    cache.insert(0, data);
    cache.get(0);  // Hit
    cache.get(1);  // Miss

    cache.resetStats();
    CacheStats stats = cache.getStats();

    EXPECT_EQ(stats.cache_hits, 0u);
    EXPECT_EQ(stats.cache_misses, 0u);
}

TEST(ElementMatrixCacheTest, LRUEvictionEvictsLeastRecentlyUsed) {
    CacheOptions options;
    options.max_elements = 2;
    options.eviction = EvictionPolicy::LRU;

    ElementMatrixCache cache(options);

    CachedElementData data;
    data.local_matrix.resize(16, 1.0);
    data.has_matrix = true;

    ASSERT_TRUE(cache.insert(0, data));
    ASSERT_TRUE(cache.insert(1, data));

    // Touch element 0 so element 1 becomes least recently used.
    ASSERT_NE(cache.get(0), nullptr);

    ASSERT_TRUE(cache.insert(2, data));

    EXPECT_TRUE(cache.contains(0));
    EXPECT_FALSE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

TEST(ElementMatrixCacheTest, FIFOEvictionEvictsFirstInserted) {
    CacheOptions options;
    options.max_elements = 2;
    options.eviction = EvictionPolicy::FIFO;

    ElementMatrixCache cache(options);

    CachedElementData data;
    data.local_matrix.resize(16, 1.0);
    data.has_matrix = true;

    ASSERT_TRUE(cache.insert(0, data));
    ASSERT_TRUE(cache.insert(1, data));

    // Touches should not affect FIFO.
    ASSERT_NE(cache.get(0), nullptr);

    ASSERT_TRUE(cache.insert(2, data));

    EXPECT_FALSE(cache.contains(0));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

TEST(ElementMatrixCacheTest, MemoryLimitEnforcementRejectsTooLargeElement) {
    CachedElementData data;
    data.local_matrix.resize(64 * 64, 1.0);
    data.has_matrix = true;
    data.local_matrix.shrink_to_fit();

    CacheOptions options;
    options.max_memory_bytes = data.memoryBytes() - 1u;
    options.eviction = EvictionPolicy::LRU;

    ElementMatrixCache cache(options);

    EXPECT_FALSE(cache.insert(0, data));
    EXPECT_EQ(cache.memoryUsage(), 0u);
}

TEST(ElementMatrixCacheTest, MemoryLimitEnforcementEvictsToStayWithinLimit) {
    CachedElementData data;
    data.local_matrix.resize(64, 1.0);
    data.has_matrix = true;
    data.local_matrix.shrink_to_fit();

    const std::size_t element_bytes = data.memoryBytes();

    CacheOptions options;
    options.max_memory_bytes = element_bytes * 2u;
    options.eviction = EvictionPolicy::LRU;

    ElementMatrixCache cache(options);

    ASSERT_TRUE(cache.insert(0, data));
    ASSERT_TRUE(cache.insert(1, data));
    EXPECT_LE(cache.memoryUsage(), options.max_memory_bytes);

    auto before = cache.getStats();
    ASSERT_TRUE(cache.insert(2, data));
    auto after = cache.getStats();

    EXPECT_LE(cache.memoryUsage(), options.max_memory_bytes);
    EXPECT_GE(after.evictions, before.evictions + 1u);
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

// ============================================================================
// CachedAssembler Tests
// ============================================================================

TEST(CachedAssemblerTest, DefaultConstruction) {
    CachedAssembler assembler;

    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_FALSE(assembler.isCachePopulated());
}

TEST(CachedAssemblerTest, ConstructionWithOptions) {
    CacheOptions options;
    options.strategy = CacheStrategy::ReferenceElement;

    CachedAssembler assembler(options);

    EXPECT_EQ(assembler.getCacheOptions().strategy, CacheStrategy::ReferenceElement);
}

TEST(CachedAssemblerTest, SetCacheOptions) {
    CachedAssembler assembler;

    CacheOptions options;
    options.max_memory_bytes = 10 * 1024 * 1024;  // 10 MB

    assembler.setCacheOptions(options);

    EXPECT_EQ(assembler.getCacheOptions().max_memory_bytes, 10u * 1024u * 1024u);
}

TEST(CachedAssemblerTest, SetAssemblyOptions) {
    CachedAssembler assembler;

    AssemblyOptions options;
    options.verbose = true;

    assembler.setOptions(options);

    EXPECT_TRUE(assembler.getOptions().verbose);
}

TEST(CachedAssemblerTest, GetCacheStats) {
    CachedAssembler assembler;

    CacheStats stats = assembler.getCacheStats();

    // Initial stats should be zeroed
    EXPECT_EQ(stats.total_elements, 0u);
    EXPECT_EQ(stats.cache_hits, 0u);
}

TEST(CachedAssemblerTest, ResetCacheStats) {
    CachedAssembler assembler;

    assembler.resetCacheStats();

    CacheStats stats = assembler.getCacheStats();
    EXPECT_EQ(stats.cache_hits, 0u);
}

TEST(CachedAssemblerCachingTest, CacheHitAvoidsKernelRecompute) {
    const GlobalIndex n_cells = 3;
    SimpleMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    CacheOptions cache_opts;
    cache_opts.strategy = CacheStrategy::FullMatrix;
    CachedAssembler assembler(cache_opts);
    assembler.setDofMap(dof_map);

    CountingIdentityKernel kernel;

    DenseMatrixView A1(dof_map.getNumDofs());
    assembler.resetCacheStats();
    assembler.assembleMatrix(mesh, space, space, kernel, A1);

    EXPECT_TRUE(assembler.isCachePopulated());
    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells));

    auto stats1 = assembler.getCacheStats();
    EXPECT_EQ(stats1.cache_hits, 0u);
    EXPECT_EQ(stats1.cache_misses, static_cast<std::size_t>(n_cells));
    EXPECT_DOUBLE_EQ(stats1.hit_rate, 0.0);

    DenseMatrixView A2(dof_map.getNumDofs());
    assembler.resetCacheStats();
    assembler.assembleMatrix(mesh, space, space, kernel, A2);

    // Second assembly should use cached element matrices only.
    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells));

    auto stats2 = assembler.getCacheStats();
    EXPECT_EQ(stats2.cache_hits, static_cast<std::size_t>(n_cells));
    EXPECT_EQ(stats2.cache_misses, 0u);
    EXPECT_DOUBLE_EQ(stats2.hit_rate, 1.0);

    EXPECT_DOUBLE_EQ(A2.getMatrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A2.getMatrixEntry(5, 5), 1.0);
}

TEST(CachedAssemblerCachingTest, CacheEvictedWhenMaxElementsReduced) {
    const GlobalIndex n_cells = 5;
    SimpleMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    CacheOptions cache_opts;
    cache_opts.strategy = CacheStrategy::FullMatrix;
    cache_opts.max_elements = 0u;
    cache_opts.eviction = EvictionPolicy::LRU;

    CachedAssembler assembler(cache_opts);
    assembler.setDofMap(dof_map);

    CountingIdentityKernel kernel;
    DenseMatrixView A(dof_map.getNumDofs());
    assembler.assembleMatrix(mesh, space, space, kernel, A);

    EXPECT_TRUE(assembler.isCachePopulated());
    EXPECT_EQ(assembler.numCachedElements(), static_cast<std::size_t>(n_cells));

    CacheOptions reduced = cache_opts;
    reduced.max_elements = 2u;
    assembler.setCacheOptions(reduced);

    EXPECT_LE(assembler.numCachedElements(), 2u);
}

TEST(CachedAssemblerCachingTest, CacheEvictedWhenMaxMemoryBytesReduced) {
    const GlobalIndex n_cells = 5;
    SimpleMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    CacheOptions cache_opts;
    cache_opts.strategy = CacheStrategy::FullMatrix;
    cache_opts.max_memory_bytes = 0u;
    cache_opts.eviction = EvictionPolicy::LRU;

    CachedAssembler assembler(cache_opts);
    assembler.setDofMap(dof_map);

    CountingIdentityKernel kernel;
    DenseMatrixView A(dof_map.getNumDofs());
    assembler.assembleMatrix(mesh, space, space, kernel, A);

    ASSERT_TRUE(assembler.isCachePopulated());
    const std::size_t before_bytes = assembler.cacheMemoryUsage();
    ASSERT_GT(before_bytes, 0u);

    CacheOptions reduced = cache_opts;
    reduced.max_memory_bytes = before_bytes / 2u;
    assembler.setCacheOptions(reduced);

    EXPECT_LE(assembler.cacheMemoryUsage(), reduced.max_memory_bytes);
    EXPECT_LT(assembler.numCachedElements(), static_cast<std::size_t>(n_cells));
}

TEST(CachedAssemblerCachingTest, StrategyReferenceElementCaches) {
    const GlobalIndex n_cells = 3;
    SimpleMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    CacheOptions cache_opts;
    cache_opts.strategy = CacheStrategy::ReferenceElement;
    CachedAssembler assembler(cache_opts);
    assembler.setDofMap(dof_map);

    CountingIdentityKernel kernel;
    DenseMatrixView A1(dof_map.getNumDofs());
    assembler.resetCacheStats();
    assembler.assembleMatrix(mesh, space, space, kernel, A1);
    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells));

    DenseMatrixView A2(dof_map.getNumDofs());
    assembler.resetCacheStats();
    assembler.assembleMatrix(mesh, space, space, kernel, A2);

    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells));
    auto stats = assembler.getCacheStats();
    EXPECT_EQ(stats.cache_hits, static_cast<std::size_t>(n_cells));
}

TEST(CachedAssemblerCachingTest, StrategyGeometricFactorsCaches) {
    const GlobalIndex n_cells = 3;
    SimpleMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    CacheOptions cache_opts;
    cache_opts.strategy = CacheStrategy::GeometricFactors;
    CachedAssembler assembler(cache_opts);
    assembler.setDofMap(dof_map);

    CountingIdentityKernel kernel;
    DenseMatrixView A1(dof_map.getNumDofs());
    assembler.resetCacheStats();
    assembler.assembleMatrix(mesh, space, space, kernel, A1);
    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells));

    DenseMatrixView A2(dof_map.getNumDofs());
    assembler.resetCacheStats();
    assembler.assembleMatrix(mesh, space, space, kernel, A2);

    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells));
    auto stats = assembler.getCacheStats();
    EXPECT_EQ(stats.cache_hits, static_cast<std::size_t>(n_cells));
}

TEST(CachedAssemblerCachingTest, SelectiveInvalidationRecomputesOnlyInvalidatedCells) {
    const GlobalIndex n_cells = 4;
    SimpleMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    CacheOptions cache_opts;
    cache_opts.strategy = CacheStrategy::FullMatrix;
    CachedAssembler assembler(cache_opts);
    assembler.setDofMap(dof_map);

    CountingIdentityKernel kernel;
    DenseMatrixView A1(dof_map.getNumDofs());
    assembler.assembleMatrix(mesh, space, space, kernel, A1);
    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells));

    // Invalidate one element and reassemble.
    const std::array<GlobalIndex, 1> ids = {2};
    assembler.invalidateElements(ids);

    DenseMatrixView A2(dof_map.getNumDofs());
    assembler.resetCacheStats();
    assembler.assembleMatrix(mesh, space, space, kernel, A2);

    // Only invalidated element should trigger a recompute.
    EXPECT_EQ(kernel.computeCalls(), static_cast<int>(n_cells + 1));

    auto stats = assembler.getCacheStats();
    EXPECT_EQ(stats.cache_hits, static_cast<std::size_t>(n_cells - 1));
    EXPECT_EQ(stats.cache_misses, 1u);
}

TEST(CachedAssemblerCachingTest, StrategyNoneDisablesCaching) {
    const GlobalIndex n_cells = 2;
    SimpleMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    CacheOptions cache_opts;
    cache_opts.strategy = CacheStrategy::None;
    CachedAssembler assembler(cache_opts);
    assembler.setDofMap(dof_map);

    CountingIdentityKernel kernel;
    DenseMatrixView A(dof_map.getNumDofs());
    assembler.assembleMatrix(mesh, space, space, kernel, A);

    EXPECT_FALSE(assembler.isCachePopulated());
}

TEST(CachedAssemblerTest, InvalidateCache) {
    CachedAssembler assembler;

    // Should not throw even with empty cache
    assembler.invalidateCache();

    EXPECT_FALSE(assembler.isCachePopulated());
}

TEST(CachedAssemblerTest, NumCachedElements) {
    CachedAssembler assembler;

    EXPECT_EQ(assembler.numCachedElements(), 0u);
}

TEST(CachedAssemblerTest, CacheMemoryUsage) {
    CachedAssembler assembler;

    EXPECT_EQ(assembler.cacheMemoryUsage(), 0u);
}

TEST(CachedAssemblerTest, Initialize) {
    CachedAssembler assembler;

    EXPECT_THROW(assembler.initialize(), std::runtime_error);
}

TEST(CachedAssemblerTest, Reset) {
    CachedAssembler assembler;
    EXPECT_NO_THROW(assembler.reset());
    EXPECT_FALSE(assembler.isCachePopulated());
}

TEST(CachedAssemblerTest, MoveConstruction) {
    CacheOptions options;
    options.max_elements = 500;

    CachedAssembler assembler1(options);
    CachedAssembler assembler2(std::move(assembler1));

    EXPECT_EQ(assembler2.getCacheOptions().max_elements, 500u);
}

TEST(CachedAssemblerTest, MoveAssignment) {
    CacheOptions options;
    options.strategy = CacheStrategy::GeometricFactors;

    CachedAssembler assembler1(options);
    CachedAssembler assembler2;

    assembler2 = std::move(assembler1);

    EXPECT_EQ(assembler2.getCacheOptions().strategy, CacheStrategy::GeometricFactors);
}

TEST(CachedAssemblerTest, InvalidationCallback) {
    CachedAssembler assembler;

    GlobalIndex invalidated_id = -1;
    assembler.setInvalidationCallback([&invalidated_id](GlobalIndex id) {
        invalidated_id = id;
    });

    // Callback stored but not yet invoked
    SUCCEED();
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(CachedAssemblerFactoryTest, CreateDefault) {
    auto assembler = createCachedAssembler();

    EXPECT_NE(assembler, nullptr);
}

TEST(CachedAssemblerFactoryTest, CreateWithOptions) {
    CacheOptions options;
    options.eviction = EvictionPolicy::FIFO;

    auto assembler = createCachedAssembler(options);

    EXPECT_NE(assembler, nullptr);

    auto* cached = dynamic_cast<CachedAssembler*>(assembler.get());
    if (cached) {
        EXPECT_EQ(cached->getCacheOptions().eviction, EvictionPolicy::FIFO);
    }
}

// ============================================================================
// Thread Safety Tests (Basic)
// ============================================================================

TEST(ElementMatrixCacheTest, ConcurrentReads) {
    ElementMatrixCache cache;

    // Pre-populate cache
    for (GlobalIndex i = 0; i < 100; ++i) {
        CachedElementData data;
        data.local_matrix.resize(16, static_cast<Real>(i));
        data.has_matrix = true;
        cache.insert(i, data);
    }

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&cache, &success_count]() {
            for (int i = 0; i < 25; ++i) {
                const CachedElementData* data = cache.get(static_cast<GlobalIndex>(i));
                if (data != nullptr) {
                    success_count++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(success_count.load(), 100);  // 4 threads * 25 reads each
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
