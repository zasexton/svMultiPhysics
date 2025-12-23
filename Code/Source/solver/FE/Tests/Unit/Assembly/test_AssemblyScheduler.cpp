/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyScheduler.cpp
 * @brief Unit tests for AssemblyScheduler
 */

#include <gtest/gtest.h>
#include "Assembly/AssemblyScheduler.h"
#include "Assembly/Assembler.h"
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Classes
// ============================================================================

/**
 * @brief Mock mesh access for scheduler testing
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

    std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(
        GlobalIndex /*face_id*/) const override {
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

class AssemblySchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>(100);
    }

    std::unique_ptr<MockMeshAccess> mesh_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, DefaultConstruction) {
    AssemblyScheduler scheduler;
    EXPECT_FALSE(scheduler.isConfigured());
}

TEST_F(AssemblySchedulerTest, ConstructionWithOptions) {
    SchedulerOptions options;
    options.ordering = OrderingStrategy::Hilbert;
    options.num_threads = 4;
    options.cache_block_size = 128;

    AssemblyScheduler scheduler(options);

    EXPECT_EQ(scheduler.getOptions().ordering, OrderingStrategy::Hilbert);
    EXPECT_EQ(scheduler.getOptions().num_threads, 4);
    EXPECT_EQ(scheduler.getOptions().cache_block_size, 128u);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, SetMesh) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);
    EXPECT_TRUE(scheduler.isConfigured());
}

TEST_F(AssemblySchedulerTest, SetOptions) {
    AssemblyScheduler scheduler;

    SchedulerOptions options;
    options.ordering = OrderingStrategy::Morton;
    options.numa = NUMAStrategy::Partitioned;
    options.load_balance = LoadBalanceMode::Dynamic;

    scheduler.setOptions(options);

    EXPECT_EQ(scheduler.getOptions().ordering, OrderingStrategy::Morton);
    EXPECT_EQ(scheduler.getOptions().numa, NUMAStrategy::Partitioned);
    EXPECT_EQ(scheduler.getOptions().load_balance, LoadBalanceMode::Dynamic);
}

// ============================================================================
// Ordering Strategy Enum Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, OrderingStrategyEnum) {
    EXPECT_NE(OrderingStrategy::Natural, OrderingStrategy::Hilbert);
    EXPECT_NE(OrderingStrategy::Hilbert, OrderingStrategy::Morton);
    EXPECT_NE(OrderingStrategy::Morton, OrderingStrategy::RCM);
    EXPECT_NE(OrderingStrategy::RCM, OrderingStrategy::ComplexityBased);
    EXPECT_NE(OrderingStrategy::ComplexityBased, OrderingStrategy::CacheBlocked);
}

TEST_F(AssemblySchedulerTest, NUMAStrategyEnum) {
    EXPECT_NE(NUMAStrategy::None, NUMAStrategy::Interleaved);
    EXPECT_NE(NUMAStrategy::Interleaved, NUMAStrategy::FirstTouch);
    EXPECT_NE(NUMAStrategy::FirstTouch, NUMAStrategy::Partitioned);
}

TEST_F(AssemblySchedulerTest, LoadBalanceModeEnum) {
    EXPECT_NE(LoadBalanceMode::Static, LoadBalanceMode::Dynamic);
    EXPECT_NE(LoadBalanceMode::Dynamic, LoadBalanceMode::Guided);
    EXPECT_NE(LoadBalanceMode::Guided, LoadBalanceMode::Adaptive);
}

// ============================================================================
// Scheduler Options Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, SchedulerOptionsDefaults) {
    SchedulerOptions options;

    EXPECT_EQ(options.ordering, OrderingStrategy::Natural);
    EXPECT_EQ(options.numa, NUMAStrategy::None);
    EXPECT_EQ(options.load_balance, LoadBalanceMode::Static);
    EXPECT_EQ(options.num_threads, 0);
    EXPECT_EQ(options.cache_block_size, 64u);
    EXPECT_TRUE(options.enable_prefetch);
    EXPECT_FALSE(options.auto_reorder);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// Natural Scheduling Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, NaturalSchedule) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    auto result = scheduler.computeNaturalSchedule();

    EXPECT_EQ(result.ordering.size(), 100u);

    // Natural ordering should be 0, 1, 2, ..., n-1
    for (std::size_t i = 0; i < result.ordering.size(); ++i) {
        EXPECT_EQ(result.ordering[i], static_cast<GlobalIndex>(i));
    }
}

TEST_F(AssemblySchedulerTest, NaturalScheduleEmpty) {
    MockMeshAccess empty_mesh(0);
    AssemblyScheduler scheduler;
    scheduler.setMesh(empty_mesh);

    auto result = scheduler.computeNaturalSchedule();

    EXPECT_TRUE(result.ordering.empty());
}

// ============================================================================
// Hilbert Schedule Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, HilbertScheduleProducesPermutation) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    // Set up centroids
    std::vector<std::array<Real, 3>> centroids(100);
    for (std::size_t i = 0; i < 100; ++i) {
        centroids[i][0] = static_cast<Real>(i % 10);
        centroids[i][1] = static_cast<Real>((i / 10) % 10);
        centroids[i][2] = static_cast<Real>(i / 100);
    }
    scheduler.setCentroids(centroids);

    auto result = scheduler.computeHilbertSchedule();

    EXPECT_EQ(result.ordering.size(), 100u);

    // Verify it's a permutation (all elements present exactly once)
    std::unordered_set<GlobalIndex> seen;
    for (GlobalIndex idx : result.ordering) {
        EXPECT_TRUE(idx >= 0 && idx < 100);
        EXPECT_TRUE(seen.find(idx) == seen.end());
        seen.insert(idx);
    }
    EXPECT_EQ(seen.size(), 100u);
}

TEST_F(AssemblySchedulerTest, HilbertScheduleRecordsTiming) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    auto result = scheduler.computeHilbertSchedule();

    EXPECT_GE(result.scheduling_seconds, 0.0);
}

// ============================================================================
// Morton Schedule Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, MortonScheduleProducesPermutation) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    auto result = scheduler.computeMortonSchedule();

    EXPECT_EQ(result.ordering.size(), 100u);

    // Verify it's a permutation
    std::unordered_set<GlobalIndex> seen;
    for (GlobalIndex idx : result.ordering) {
        EXPECT_TRUE(idx >= 0 && idx < 100);
        seen.insert(idx);
    }
    EXPECT_EQ(seen.size(), 100u);
}

// ============================================================================
// RCM Schedule Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, RCMScheduleProducesPermutation) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    auto result = scheduler.computeRCMSchedule();

    EXPECT_EQ(result.ordering.size(), 100u);

    // Verify it's a permutation
    std::unordered_set<GlobalIndex> seen;
    for (GlobalIndex idx : result.ordering) {
        EXPECT_TRUE(idx >= 0 && idx < 100);
        seen.insert(idx);
    }
    EXPECT_EQ(seen.size(), 100u);
}

// ============================================================================
// Complexity-Based Schedule Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, ComplexityScheduleProducesPermutation) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    auto result = scheduler.computeComplexitySchedule();

    EXPECT_EQ(result.ordering.size(), 100u);

    // Verify it's a permutation
    std::unordered_set<GlobalIndex> seen;
    for (GlobalIndex idx : result.ordering) {
        EXPECT_TRUE(idx >= 0 && idx < 100);
        seen.insert(idx);
    }
    EXPECT_EQ(seen.size(), 100u);
}

// ============================================================================
// Cache-Blocked Schedule Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, CacheBlockedScheduleProducesPermutation) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    SchedulerOptions options;
    options.cache_block_size = 16;
    scheduler.setOptions(options);

    auto result = scheduler.computeCacheBlockedSchedule();

    EXPECT_EQ(result.ordering.size(), 100u);

    // Verify it's a permutation
    std::unordered_set<GlobalIndex> seen;
    for (GlobalIndex idx : result.ordering) {
        EXPECT_TRUE(idx >= 0 && idx < 100);
        seen.insert(idx);
    }
    EXPECT_EQ(seen.size(), 100u);
}

// ============================================================================
// Thread Distribution Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, ComputeStaticAssignment) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    std::vector<GlobalIndex> ordering(100);
    std::iota(ordering.begin(), ordering.end(), 0);

    auto assignment = scheduler.computeStaticAssignment(ordering, 4);

    EXPECT_EQ(assignment.size(), 100u);

    // All thread IDs should be in range [0, 3]
    for (int tid : assignment) {
        EXPECT_GE(tid, 0);
        EXPECT_LT(tid, 4);
    }
}

TEST_F(AssemblySchedulerTest, ComputeThreadRanges) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    std::vector<GlobalIndex> ordering(100);
    std::iota(ordering.begin(), ordering.end(), 0);

    auto ranges = scheduler.computeThreadRanges(ordering, 4);

    EXPECT_EQ(ranges.size(), 5u);  // num_threads + 1
    EXPECT_EQ(ranges[0], 0u);
    EXPECT_EQ(ranges[4], 100u);

    // Ranges should be increasing
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        EXPECT_GE(ranges[i], ranges[i-1]);
    }
}

TEST_F(AssemblySchedulerTest, ComputeScheduleWithThreads) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    SchedulerOptions options;
    options.num_threads = 4;
    scheduler.setOptions(options);

    auto result = scheduler.computeSchedule();

    EXPECT_EQ(result.thread_assignment.size(), 100u);
    EXPECT_EQ(result.thread_ranges.size(), 5u);
    EXPECT_EQ(result.thread_work.size(), 4u);
}

// ============================================================================
// Work Stealing Chunks Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, GetWorkStealingChunks) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    std::vector<GlobalIndex> ordering(100);
    std::iota(ordering.begin(), ordering.end(), 0);

    auto chunks = scheduler.getWorkStealingChunks(ordering, 4, 10);

    // Should have (100 / 10) + 1 = 11 entries
    EXPECT_EQ(chunks.size(), 11u);
    EXPECT_EQ(chunks.front(), 0u);
    EXPECT_EQ(chunks.back(), 100u);
}

// ============================================================================
// Space Filling Curve Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, Morton2D) {
    // Test known Morton codes
    EXPECT_EQ(SpaceFillingCurve::morton2D(0, 0), 0u);
    EXPECT_EQ(SpaceFillingCurve::morton2D(1, 0), 1u);
    EXPECT_EQ(SpaceFillingCurve::morton2D(0, 1), 2u);
    EXPECT_EQ(SpaceFillingCurve::morton2D(1, 1), 3u);
}

TEST_F(AssemblySchedulerTest, Morton3D) {
    // Test known Morton codes
    EXPECT_EQ(SpaceFillingCurve::morton3D(0, 0, 0), 0u);
    EXPECT_EQ(SpaceFillingCurve::morton3D(1, 0, 0), 1u);
    EXPECT_EQ(SpaceFillingCurve::morton3D(0, 1, 0), 2u);
    EXPECT_EQ(SpaceFillingCurve::morton3D(0, 0, 1), 4u);
}

TEST_F(AssemblySchedulerTest, Hilbert2D) {
    // Basic Hilbert curve test - first 4 cells
    EXPECT_EQ(SpaceFillingCurve::hilbert2D(0, 0, 2), 0u);
    // Other values depend on Hilbert curve rotation
}

TEST_F(AssemblySchedulerTest, Discretize) {
    EXPECT_EQ(SpaceFillingCurve::discretize(0.0, 0.0, 1.0, 10), 0u);
    EXPECT_EQ(SpaceFillingCurve::discretize(0.5, 0.0, 1.0, 10), 5u);
    EXPECT_EQ(SpaceFillingCurve::discretize(0.99, 0.0, 1.0, 10), 9u);

    // Clamp to range
    EXPECT_EQ(SpaceFillingCurve::discretize(-1.0, 0.0, 1.0, 10), 0u);
}

// ============================================================================
// Complexity Estimator Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, ComplexityEstimatorEstimate) {
    ComplexityEstimator estimator;

    auto result = estimator.estimate(ElementType::Tetra4, 4, 4, 1);

    EXPECT_EQ(result.num_dofs, 4u);
    EXPECT_EQ(result.num_qpts, 4u);
    EXPECT_EQ(result.polynomial_order, 1);
    EXPECT_GT(result.estimated_flops, 0.0);
    EXPECT_GT(result.estimated_memory, 0.0);
    EXPECT_GT(result.cost(), 0.0);
    EXPECT_GT(result.arithmeticIntensity(), 0.0);
}

TEST_F(AssemblySchedulerTest, ComplexityEstimatorEstimateAll) {
    ComplexityEstimator estimator;

    // Need DofMap - skip full test for now
    // auto results = estimator.estimateAll(*mesh_, dof_map);
}

TEST_F(AssemblySchedulerTest, ComplexityEstimatorTotalCost) {
    ComplexityEstimator estimator;

    std::vector<ElementComplexity> complexities;
    for (int i = 0; i < 10; ++i) {
        auto c = estimator.estimate(ElementType::Tetra4, 4, 4, 1);
        complexities.push_back(c);
    }

    double total = estimator.totalCost(complexities);
    EXPECT_GT(total, 0.0);
}

TEST_F(AssemblySchedulerTest, ComplexityEstimatorLoadImbalance) {
    ComplexityEstimator estimator;

    std::vector<ElementComplexity> complexities;
    for (int i = 0; i < 100; ++i) {
        auto c = estimator.estimate(ElementType::Tetra4, 4, 4, 1);
        complexities.push_back(c);
    }

    double imbalance = estimator.loadImbalance(complexities, 4);
    EXPECT_GE(imbalance, 1.0);  // Perfect balance would be 1.0
}

// ============================================================================
// NUMA Topology Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, NUMATopologyBasic) {
    NUMATopology topo;

    EXPECT_GE(topo.numNodes(), 1);
    EXPECT_GE(topo.cpusPerNode(), 1);
}

TEST_F(AssemblySchedulerTest, NUMATopologyComputeThreadMapping) {
    NUMATopology topo;

    auto mapping = topo.computeThreadMapping(4);

    EXPECT_EQ(mapping.size(), 4u);
}

TEST_F(AssemblySchedulerTest, NUMATopologyGetCPUsInNode) {
    NUMATopology topo;

    auto cpus = topo.getCPUsInNode(0);
    EXPECT_FALSE(cpus.empty());
}

// ============================================================================
// Color Optimization Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, OptimizeColorOrder) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    // Create simple coloring: 3 colors
    std::vector<int> colors(100);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 3);
    }

    auto optimized = scheduler.optimizeColorOrder(colors, 3);

    EXPECT_EQ(optimized.size(), 100u);

    // Verify it's a permutation
    std::unordered_set<GlobalIndex> seen;
    for (GlobalIndex idx : optimized) {
        EXPECT_TRUE(idx >= 0 && idx < 100);
        seen.insert(idx);
    }
    EXPECT_EQ(seen.size(), 100u);
}

TEST_F(AssemblySchedulerTest, OptimizeColorSequence) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    std::vector<int> colors(100);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 5);
    }

    auto sequence = scheduler.optimizeColorSequence(colors, 5);

    EXPECT_EQ(sequence.size(), 5u);

    // All colors should be present
    std::unordered_set<int> seen(sequence.begin(), sequence.end());
    EXPECT_EQ(seen.size(), 5u);
}

// ============================================================================
// Custom Comparator Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, CustomComparator) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    SchedulerOptions options;
    options.ordering = OrderingStrategy::Custom;
    scheduler.setOptions(options);

    // Reverse ordering
    scheduler.setCustomComparator(
        [](GlobalIndex a, GlobalIndex b) { return a > b; });

    auto result = scheduler.computeSchedule();

    EXPECT_EQ(result.ordering.size(), 100u);

    // Should be in reverse order
    for (std::size_t i = 1; i < result.ordering.size(); ++i) {
        EXPECT_GE(result.ordering[i-1], result.ordering[i]);
    }
}

// ============================================================================
// Scheduling Result Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, SchedulingResultDefaults) {
    SchedulingResult result;

    EXPECT_TRUE(result.ordering.empty());
    EXPECT_TRUE(result.thread_assignment.empty());
    EXPECT_TRUE(result.thread_ranges.empty());
    EXPECT_TRUE(result.thread_work.empty());
    EXPECT_EQ(result.scheduling_seconds, 0.0);
    EXPECT_EQ(result.estimated_imbalance, 1.0);
}

TEST_F(AssemblySchedulerTest, SchedulingResultPopulated) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    SchedulerOptions options;
    options.num_threads = 4;
    options.ordering = OrderingStrategy::Hilbert;
    scheduler.setOptions(options);

    auto result = scheduler.computeSchedule();

    EXPECT_FALSE(result.ordering.empty());
    EXPECT_FALSE(result.thread_assignment.empty());
    EXPECT_FALSE(result.thread_ranges.empty());
    EXPECT_FALSE(result.thread_work.empty());
    EXPECT_GE(result.scheduling_seconds, 0.0);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, MoveConstruction) {
    AssemblyScheduler scheduler1;
    scheduler1.setMesh(*mesh_);

    SchedulerOptions options;
    options.num_threads = 4;
    scheduler1.setOptions(options);

    AssemblyScheduler scheduler2(std::move(scheduler1));

    EXPECT_EQ(scheduler2.getOptions().num_threads, 4);
}

TEST_F(AssemblySchedulerTest, MoveAssignment) {
    AssemblyScheduler scheduler1;
    scheduler1.setMesh(*mesh_);

    SchedulerOptions options;
    options.num_threads = 8;
    scheduler1.setOptions(options);

    AssemblyScheduler scheduler2;
    scheduler2 = std::move(scheduler1);

    EXPECT_EQ(scheduler2.getOptions().num_threads, 8);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, CreateAssemblySchedulerDefault) {
    auto scheduler = createAssemblyScheduler();
    ASSERT_NE(scheduler, nullptr);
    EXPECT_EQ(scheduler->getOptions().ordering, OrderingStrategy::Natural);
}

TEST_F(AssemblySchedulerTest, CreateAssemblySchedulerWithOptions) {
    SchedulerOptions options;
    options.ordering = OrderingStrategy::Morton;
    options.num_threads = 8;

    auto scheduler = createAssemblyScheduler(options);
    ASSERT_NE(scheduler, nullptr);
    EXPECT_EQ(scheduler->getOptions().ordering, OrderingStrategy::Morton);
    EXPECT_EQ(scheduler->getOptions().num_threads, 8);
}

TEST_F(AssemblySchedulerTest, CreateAssemblySchedulerWithOrdering) {
    auto scheduler = createAssemblyScheduler(OrderingStrategy::Hilbert);
    ASSERT_NE(scheduler, nullptr);
    EXPECT_EQ(scheduler->getOptions().ordering, OrderingStrategy::Hilbert);
}

// ============================================================================
// Get Last Result Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, GetLastResult) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    auto result1 = scheduler.computeSchedule();
    const auto& last = scheduler.getLastResult();

    EXPECT_EQ(result1.ordering.size(), last.ordering.size());
}

// ============================================================================
// Complexity Estimator Access Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, GetComplexityEstimator) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    ComplexityEstimator& estimator = scheduler.getComplexityEstimator();

    auto complexity = estimator.estimate(ElementType::Hex8, 8, 8, 1);
    EXPECT_EQ(complexity.num_dofs, 8u);
}

// ============================================================================
// Schedule Subset Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, ComputeScheduleSubset) {
    AssemblyScheduler scheduler;
    scheduler.setMesh(*mesh_);

    // Schedule only elements 10-50
    std::vector<GlobalIndex> subset;
    for (GlobalIndex i = 10; i < 50; ++i) {
        subset.push_back(i);
    }

    auto result = scheduler.computeSchedule(subset);

    EXPECT_EQ(result.ordering.size(), 40u);

    // All elements should be from the subset
    std::unordered_set<GlobalIndex> subset_set(subset.begin(), subset.end());
    for (GlobalIndex idx : result.ordering) {
        EXPECT_TRUE(subset_set.count(idx) > 0);
    }
}

// ============================================================================
// Element Complexity Tests
// ============================================================================

TEST_F(AssemblySchedulerTest, ElementComplexityCost) {
    ElementComplexity c;
    c.num_dofs = 10;
    c.num_qpts = 8;

    double cost = c.cost();
    EXPECT_EQ(cost, 10.0 * 10.0 * 8.0);
}

TEST_F(AssemblySchedulerTest, ElementComplexityArithmeticIntensity) {
    ElementComplexity c;
    c.estimated_flops = 1000.0;
    c.estimated_memory = 100.0;

    EXPECT_DOUBLE_EQ(c.arithmeticIntensity(), 10.0);
}

TEST_F(AssemblySchedulerTest, ElementComplexityArithmeticIntensityZeroMemory) {
    ElementComplexity c;
    c.estimated_flops = 1000.0;
    c.estimated_memory = 0.0;

    EXPECT_DOUBLE_EQ(c.arithmeticIntensity(), 0.0);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
