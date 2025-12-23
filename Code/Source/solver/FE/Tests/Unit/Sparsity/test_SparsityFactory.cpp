/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include "Sparsity/SparsityFactory.h"
#include "Sparsity/SparsityPattern.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofConstraints.h"
#include <vector>
#include <algorithm>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;
using namespace svmp::FE::dofs;

// ============================================================================
// Helper functions to create test DofMaps
// ============================================================================

namespace {

// Helper to create a simple mesh DofMap
DofMap createSimpleDofMap(GlobalIndex n_dofs, GlobalIndex n_cells,
                           LocalIndex dofs_per_cell) {
    DofMap dof_map(n_cells, n_dofs, dofs_per_cell);

    for (GlobalIndex e = 0; e < n_cells; ++e) {
        std::vector<GlobalIndex> dofs;
        for (LocalIndex d = 0; d < dofs_per_cell; ++d) {
            dofs.push_back(static_cast<GlobalIndex>((e * dofs_per_cell + d) % n_dofs));
        }
        dof_map.setCellDofs(e, dofs);
    }
    dof_map.setNumDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

// Helper to create overlapping element DofMap
DofMap createOverlappingDofMap(GlobalIndex n_dofs, GlobalIndex n_cells) {
    DofMap dof_map(n_cells, n_dofs, 3);

    for (GlobalIndex e = 0; e < n_cells; ++e) {
        std::vector<GlobalIndex> dofs;
        for (GlobalIndex d = 0; d < 3; ++d) {
            dofs.push_back((e + d) % n_dofs);
        }
        dof_map.setCellDofs(e, dofs);
    }
    dof_map.setNumDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

} // anonymous namespace

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(SparsityFactoryTest, DefaultConstruction) {
    SparsityFactory factory;
    auto options = factory.getOptions();

    EXPECT_EQ(options.type, PatternType::Standard);
    EXPECT_EQ(options.backend, TargetBackend::Generic);
    EXPECT_EQ(options.strategy, ConstructionStrategy::Auto);
}

TEST(SparsityFactoryTest, ConstructWithOptions) {
    FactoryOptions opts;
    opts.type = PatternType::DG;
    opts.backend = TargetBackend::PETSc;

    SparsityFactory factory(opts);
    auto stored_opts = factory.getOptions();

    EXPECT_EQ(stored_opts.type, PatternType::DG);
    EXPECT_EQ(stored_opts.backend, TargetBackend::PETSc);
}

TEST(SparsityFactoryTest, SetOptions) {
    SparsityFactory factory;

    FactoryOptions opts;
    opts.ensure_diagonal = false;
    factory.setOptions(opts);

    EXPECT_FALSE(factory.getOptions().ensure_diagonal);
}

// ============================================================================
// Standard Pattern Creation Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateLaplacianPattern1D) {
    SparsityFactory factory;
    std::vector<GlobalIndex> dims = {10};

    auto pattern = factory.createLaplacianPattern(dims);

    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.numCols(), 10);

    // Check tridiagonal structure
    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_TRUE(pattern.hasEntry(i, i));
        if (i > 0) EXPECT_TRUE(pattern.hasEntry(i, i - 1));
        if (i < 9) EXPECT_TRUE(pattern.hasEntry(i, i + 1));
    }
}

TEST(SparsityFactoryTest, CreateLaplacianPattern2D) {
    SparsityFactory factory;
    std::vector<GlobalIndex> dims = {4, 4};

    auto pattern = factory.createLaplacianPattern(dims);

    EXPECT_EQ(pattern.numRows(), 16);
    EXPECT_EQ(pattern.numCols(), 16);

    // Each interior point has 5 neighbors (including self)
    // Boundary points have fewer
    GlobalIndex interior = 1 * 4 + 1;  // (1, 1)
    EXPECT_EQ(pattern.getRowNnz(interior), 5);
}

TEST(SparsityFactoryTest, CreateLaplacianPattern3D) {
    SparsityFactory factory;
    std::vector<GlobalIndex> dims = {3, 3, 3};

    auto pattern = factory.createLaplacianPattern(dims);

    EXPECT_EQ(pattern.numRows(), 27);

    // Center point (1,1,1) has 7 neighbors (including self)
    GlobalIndex center = 1 * 9 + 1 * 3 + 1;
    EXPECT_EQ(pattern.getRowNnz(center), 7);
}

TEST(SparsityFactoryTest, CreateLaplacianPatternPeriodic) {
    SparsityFactory factory;
    std::vector<GlobalIndex> dims = {5};

    auto pattern = factory.createLaplacianPattern(dims, true);

    EXPECT_EQ(pattern.numRows(), 5);

    // All points should have 3 neighbors (left, self, right)
    // with periodic wrap
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_EQ(pattern.getRowNnz(i), 3);
    }

    // Check wrap-around
    EXPECT_TRUE(pattern.hasEntry(0, 4));
    EXPECT_TRUE(pattern.hasEntry(4, 0));
}

TEST(SparsityFactoryTest, CreateBandPattern) {
    SparsityFactory factory;

    auto pattern = factory.createBandPattern(10, 2, 3);

    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.computeBandwidth(), 3);

    // Check middle row
    EXPECT_TRUE(pattern.hasEntry(5, 3));  // -2
    EXPECT_TRUE(pattern.hasEntry(5, 4));  // -1
    EXPECT_TRUE(pattern.hasEntry(5, 5));  // 0
    EXPECT_TRUE(pattern.hasEntry(5, 6));  // +1
    EXPECT_TRUE(pattern.hasEntry(5, 7));  // +2
    EXPECT_TRUE(pattern.hasEntry(5, 8));  // +3
}

TEST(SparsityFactoryTest, CreateDiagonalPattern) {
    SparsityFactory factory;

    auto pattern = factory.createDiagonalPattern(5);

    EXPECT_EQ(pattern.numRows(), 5);
    EXPECT_EQ(pattern.getNnz(), 5);

    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(pattern.hasEntry(i, i));
        EXPECT_EQ(pattern.getRowNnz(i), 1);
    }
}

TEST(SparsityFactoryTest, CreateDensePattern) {
    SparsityFactory factory;

    auto pattern = factory.createDensePattern(3, 4);

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 4);
    EXPECT_EQ(pattern.getNnz(), 12);

    for (GlobalIndex i = 0; i < 3; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_TRUE(pattern.hasEntry(i, j));
        }
    }
}

// ============================================================================
// DOF Map Based Creation Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateFromDofMap) {
    auto dof_map = createSimpleDofMap(10, 4, 3);

    SparsityFactory factory;
    auto pattern = factory.createFromDofMap(dof_map);

    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.numCols(), 10);
}

TEST(SparsityFactoryTest, CreateFromDofMapOverlapping) {
    auto dof_map = createOverlappingDofMap(10, 8);

    SparsityFactory factory;
    auto pattern = factory.createFromDofMap(dof_map);

    EXPECT_TRUE(pattern.isFinalized());

    // Should have couplings between adjacent elements
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(1, 2));
}

TEST(SparsityFactoryTest, CreateRejectsDGWithoutFaceAdjacency) {
    auto dof_map = createSimpleDofMap(12, 4, 3);

    SparsityFactory factory;
    FactoryOptions opts;
    opts.type = PatternType::DG;

    EXPECT_THROW((void)factory.create(dof_map, opts), InvalidArgumentException);
}

TEST(SparsityFactoryTest, CreateRejectsRectangularWithoutColMap) {
    auto dof_map = createSimpleDofMap(12, 4, 3);

    SparsityFactory factory;
    FactoryOptions opts;
    opts.type = PatternType::Rectangular;

    EXPECT_THROW((void)factory.create(dof_map, opts), InvalidArgumentException);
}

TEST(SparsityFactoryTest, CreateRejectsBlockStructuredWithoutFieldMaps) {
    auto dof_map = createSimpleDofMap(12, 4, 3);

    SparsityFactory factory;
    FactoryOptions opts;
    opts.type = PatternType::BlockStructured;

    EXPECT_THROW((void)factory.create(dof_map, opts), InvalidArgumentException);
}

TEST(SparsityFactoryTest, CreateRejectsCustomWithoutCallback) {
    auto dof_map = createSimpleDofMap(12, 4, 3);

    SparsityFactory factory;
    FactoryOptions opts;
    opts.type = PatternType::Custom;

    EXPECT_THROW((void)factory.create(dof_map, opts), InvalidArgumentException);
}

TEST(SparsityFactoryTest, CreateWithOptions) {
    auto dof_map = createSimpleDofMap(10, 4, 3);

    FactoryOptions opts;
    opts.ensure_diagonal = true;

    SparsityFactory factory;
    auto result = factory.create(dof_map, opts);

    EXPECT_TRUE(result.pattern != nullptr);

    // Check diagonal is ensured
    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_TRUE(result.pattern->hasEntry(i, i));
    }
}

TEST(SparsityFactoryTest, CreateWithOptimization) {
    auto dof_map = createOverlappingDofMap(20, 15);

    FactoryOptions opts;
    opts.optimize = true;
    opts.optimization_goal = OptimizationGoal::MinimizeBandwidth;

    SparsityFactory factory;
    auto result = factory.create(dof_map, opts);

    EXPECT_TRUE(result.pattern != nullptr);
    EXPECT_TRUE(result.optimization_result.has_value());
}

// ============================================================================
// Distributed Pattern Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateDistributedRespectsGhostCouplingOption) {
    DofMap dof_map(/*n_cells=*/1, /*n_dofs_total=*/6, /*dofs_per_cell=*/3);
    std::vector<GlobalIndex> dofs = {0, 1, 3};  // Row 0/1 owned, col 3 ghost
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(6);
    dof_map.finalize();

    SparsityFactory factory;
    FactoryOptions opts;

    opts.include_ghost_rows = true;
    auto full = factory.createDistributed(dof_map, {0, 3}, 6, opts);
    ASSERT_TRUE(full.distributed_pattern != nullptr);
    EXPECT_EQ(full.distributed_pattern->numGhostCols(), 1);
    auto ghost_cols = full.distributed_pattern->getGhostColMap();
    ASSERT_EQ(ghost_cols.size(), 1);
    EXPECT_EQ(ghost_cols[0], 3);

    opts.include_ghost_rows = false;
    auto diag_only = factory.createDistributed(dof_map, {0, 3}, 6, opts);
    ASSERT_TRUE(diag_only.distributed_pattern != nullptr);
    EXPECT_EQ(diag_only.distributed_pattern->numGhostCols(), 0);
    EXPECT_FALSE(diag_only.distributed_pattern->hasEntry(0, 3));
    EXPECT_TRUE(diag_only.distributed_pattern->hasEntry(0, 0));  // diagonal still allowed
}

// ============================================================================
// Rectangular Pattern Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateRectangular) {
    auto row_map = createSimpleDofMap(8, 4, 2);
    auto col_map = createSimpleDofMap(12, 4, 3);

    SparsityFactory factory;
    auto result = factory.createRectangular(row_map, col_map);

    EXPECT_TRUE(result.pattern != nullptr);
    EXPECT_EQ(result.pattern->numRows(), 8);
    EXPECT_EQ(result.pattern->numCols(), 12);
}

// ============================================================================
// Block Pattern Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateBlockPatternPopulatesBlocks) {
    auto field0 = createSimpleDofMap(4, 2, 2);
    auto field1 = createSimpleDofMap(3, 2, 2);

    std::vector<const DofMap*> field_maps = {&field0, &field1};

    SparsityFactory factory;
    FactoryOptions opts;
    opts.ensure_diagonal = true;

    auto result = factory.createBlockPattern(field_maps, opts);

    ASSERT_TRUE(result.pattern != nullptr);
    ASSERT_TRUE(result.block_pattern != nullptr);
    EXPECT_TRUE(result.pattern->isFinalized());
    EXPECT_TRUE(result.block_pattern->isFinalized());

    // Off-diagonal block (field0, field1) should contain couplings from cell 0.
    EXPECT_TRUE(result.block_pattern->getBlock(0, 1).hasEntry(0, 0));
    EXPECT_TRUE(result.block_pattern->hasBlock(0, 1));

    // Block -> monolithic conversion matches the monolithic pattern returned by the factory.
    auto mono_from_blocks = result.block_pattern->toMonolithic();
    EXPECT_EQ(mono_from_blocks.getNnz(), result.pattern->getNnz());
    EXPECT_TRUE(std::equal(mono_from_blocks.getRowPtr().begin(),
                           mono_from_blocks.getRowPtr().end(),
                           result.pattern->getRowPtr().begin()));
    EXPECT_TRUE(std::equal(mono_from_blocks.getColIndices().begin(),
                           mono_from_blocks.getColIndices().end(),
                           result.pattern->getColIndices().begin()));
}

TEST(SparsityFactoryTest, CreateBlockPatternRespectsCouplingMatrix) {
    auto field0 = createSimpleDofMap(4, 2, 2);
    auto field1 = createSimpleDofMap(3, 2, 2);

    std::vector<const DofMap*> field_maps = {&field0, &field1};

    std::vector<std::vector<bool>> coupling = {
        {true,  false},
        {false, true}
    };

    SparsityFactory factory;
    FactoryOptions opts;
    opts.ensure_diagonal = true;

    auto result = factory.createBlockPattern(field_maps, coupling, opts);

    ASSERT_TRUE(result.pattern != nullptr);
    ASSERT_TRUE(result.block_pattern != nullptr);

    // No off-diagonal coupling allowed.
    EXPECT_FALSE(result.block_pattern->hasBlock(0, 1));
    EXPECT_FALSE(result.pattern->hasEntry(0, 4));  // field1 offset is 4 in monolithic numbering
}

// ============================================================================
// DG Pattern Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateDGPattern) {
    auto dof_map = createSimpleDofMap(12, 4, 3);

    // Face adjacency: element pairs that share a face
    std::vector<std::pair<GlobalIndex, GlobalIndex>> face_adjacency = {
        {0, 1}, {1, 2}, {2, 3}
    };

    SparsityFactory factory;
    FactoryOptions opts;
    opts.type = PatternType::DG;

    auto result = factory.createDGPattern(dof_map, face_adjacency, opts);

    ASSERT_TRUE(result.pattern != nullptr);
    EXPECT_EQ(result.pattern->numRows(), 12);
    EXPECT_TRUE(result.pattern->isFinalized());

    // Should have coupling between elements 0 and 1
    // Elements have DOFs {0,1,2} and {3,4,5}
    EXPECT_TRUE(result.pattern->hasEntry(0, 3));
    EXPECT_TRUE(result.pattern->hasEntry(3, 0));
    EXPECT_TRUE(result.pattern->hasEntry(2, 5));

    // No direct coupling between non-adjacent elements (0 and 2)
    EXPECT_FALSE(result.pattern->hasEntry(0, 6));
}

// ============================================================================
// Constraint Application Tests
// ============================================================================

TEST(SparsityFactoryTest, ApplyConstraintsFinalizesAndAddsFill) {
    SparsityFactory factory;

    SparsityPattern base(6, 6);
    base.addEntry(0, 5);
    base.finalize();

    std::vector<SparsityConstraint> constraints = {{5, {1}}};

    auto augmented = factory.applyConstraints(base, constraints, AugmentationMode::EliminationFill);
    EXPECT_TRUE(augmented.isFinalized());
    EXPECT_TRUE(augmented.hasEntry(0, 1));
}

// ============================================================================
// From Arrays Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateFromArrays) {
    GlobalIndex n_rows = 10;
    GlobalIndex n_cols = 10;
    GlobalIndex n_elements = 3;

    std::vector<GlobalIndex> elem_offsets = {0, 3, 6, 9};
    std::vector<GlobalIndex> elem_dofs = {0, 1, 2, 2, 3, 4, 4, 5, 6};

    SparsityFactory factory;
    FactoryOptions opts;
    opts.ensure_diagonal = true;

    auto result = factory.createFromArrays(n_rows, n_cols, n_elements,
                                            elem_offsets, elem_dofs, opts);

    EXPECT_TRUE(result.pattern != nullptr);
    EXPECT_EQ(result.pattern->numRows(), 10);

    // Check element couplings
    EXPECT_TRUE(result.pattern->hasEntry(0, 1));
    EXPECT_TRUE(result.pattern->hasEntry(2, 3));
}

// ============================================================================
// From Callback Tests
// ============================================================================

TEST(SparsityFactoryTest, CreateFromCallback) {
    GlobalIndex n = 5;

    // Create tridiagonal pattern via callback
    auto row_entries = [n](GlobalIndex row) -> std::vector<GlobalIndex> {
        std::vector<GlobalIndex> cols;
        if (row > 0) cols.push_back(row - 1);
        cols.push_back(row);
        if (row < n - 1) cols.push_back(row + 1);
        return cols;
    };

    SparsityFactory factory;
    FactoryOptions opts;

    auto result = factory.createFromCallback(n, n, row_entries, opts);

    EXPECT_TRUE(result.pattern != nullptr);
    EXPECT_EQ(result.pattern->numRows(), 5);
    EXPECT_EQ(result.pattern->computeBandwidth(), 1);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST(SparsityFactoryTest, CachingDisabled) {
    auto dof_map = createSimpleDofMap(10, 4, 3);

    FactoryOptions opts;
    opts.enable_caching = false;

    SparsityFactory factory;
    auto result1 = factory.create(dof_map, opts);
    auto result2 = factory.create(dof_map, opts);

    EXPECT_FALSE(result1.from_cache);
    EXPECT_FALSE(result2.from_cache);
}

TEST(SparsityFactoryTest, CachingEnabled) {
    auto dof_map = createSimpleDofMap(10, 4, 3);

    FactoryOptions opts;
    opts.enable_caching = true;

    SparsityFactory factory;
    auto result1 = factory.create(dof_map, opts);
    auto result2 = factory.create(dof_map, opts);

    EXPECT_FALSE(result1.from_cache);
    EXPECT_TRUE(result2.from_cache);
    ASSERT_TRUE(result2.pattern != nullptr);
    EXPECT_TRUE(result2.pattern->isFinalized());
}

TEST(SparsityFactoryTest, DistributedCachingEnabled) {
    auto dof_map = createOverlappingDofMap(10, 8);

    FactoryOptions opts;
    opts.enable_caching = true;

    SparsityFactory factory;
    auto r1 = factory.createDistributed(dof_map, {0, 5}, dof_map.getNumDofs(), opts);
    auto r2 = factory.createDistributed(dof_map, {0, 5}, dof_map.getNumDofs(), opts);

    EXPECT_FALSE(r1.from_cache);
    EXPECT_TRUE(r2.from_cache);
    ASSERT_TRUE(r2.distributed_pattern != nullptr);
    EXPECT_TRUE(r2.distributed_pattern->isFinalized());
}

TEST(SparsityFactoryTest, BlockCachingEnabled) {
    auto field0 = createSimpleDofMap(4, 3, 2);
    auto field1 = createSimpleDofMap(5, 3, 2);

    std::vector<const DofMap*> field_maps = {&field0, &field1};
    std::vector<std::vector<bool>> coupling = {
        {true,  false},
        {false, true}
    };

    FactoryOptions opts;
    opts.enable_caching = true;

    SparsityFactory factory;
    auto r1 = factory.createBlockPattern(field_maps, coupling, opts);
    auto r2 = factory.createBlockPattern(field_maps, coupling, opts);

    EXPECT_FALSE(r1.from_cache);
    EXPECT_TRUE(r2.from_cache);
    ASSERT_TRUE(r2.block_pattern != nullptr);
    EXPECT_TRUE(r2.block_pattern->isFinalized());
    ASSERT_TRUE(r2.pattern != nullptr);
    EXPECT_TRUE(r2.pattern->isFinalized());
}

TEST(SparsityFactoryTest, CacheKeyIncludesDofMapConnectivity) {
    auto dof_map1 = createSimpleDofMap(10, 4, 3);
    auto dof_map2 = createOverlappingDofMap(10, 4);  // same dims, different connectivity

    FactoryOptions opts;
    opts.enable_caching = true;

    SparsityFactory factory;
    auto r1 = factory.create(dof_map1, opts);
    auto r2 = factory.create(dof_map2, opts);
    auto r3 = factory.create(dof_map1, opts);

    EXPECT_FALSE(r1.from_cache);
    EXPECT_FALSE(r2.from_cache);
    EXPECT_TRUE(r3.from_cache);
}

TEST(SparsityFactoryTest, CreateWithApplyConstraintsRequiresConfiguredConstraints) {
    auto dof_map = createSimpleDofMap(10, 4, 3);

    FactoryOptions opts;
    opts.apply_constraints = true;
    opts.constraint_mode = AugmentationMode::EliminationFill;

    SparsityFactory factory;
    EXPECT_THROW((void)factory.create(dof_map, opts), svmp::FE::FEException);
}

TEST(SparsityFactoryTest, CreateWithApplyConstraintsAppliesConfiguredConstraints) {
    // Build a minimal pattern where row 0 couples to constrained DOF 5.
    DofMap dof_map(1, 6, 2);
    std::vector<GlobalIndex> dofs = {0, 5};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(6);
    dof_map.finalize();

    dofs::DofConstraints constraints;
    constraints.addPeriodicBC(1, 5);  // slave 5 depends on master 1

    SparsityFactory factory;
    factory.setConstraints(constraints);

    FactoryOptions opts;
    opts.apply_constraints = true;
    opts.constraint_mode = AugmentationMode::EliminationFill;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    auto result = factory.create(dof_map, opts);
    ASSERT_TRUE(result.pattern != nullptr);
    EXPECT_TRUE(result.pattern->isFinalized());

    // Base coupling exists.
    EXPECT_TRUE(result.pattern->hasEntry(0, 5));
    // Induced coupling via constraint fill.
    EXPECT_TRUE(result.pattern->hasEntry(0, 1));
}

TEST(SparsityFactoryTest, ClearCache) {
    auto dof_map = createSimpleDofMap(10, 4, 3);

    FactoryOptions opts;
    opts.enable_caching = true;

    SparsityFactory factory;
    factory.create(dof_map, opts);

    auto [hits, misses] = factory.getCacheStats();
    EXPECT_EQ(misses, 1);

    factory.clearCache();

    auto [hits2, misses2] = factory.getCacheStats();
    EXPECT_EQ(hits2, 0);
    EXPECT_EQ(misses2, 0);
}

// ============================================================================
// Strategy Selection Tests
// ============================================================================

TEST(SparsityFactoryTest, SuggestStrategy) {
    SparsityFactory factory;

    // Small problem
    auto strategy1 = factory.suggestStrategy(100, 50, 4);
    EXPECT_EQ(strategy1, ConstructionStrategy::Standard);

    // Large problem
    auto strategy2 = factory.suggestStrategy(1000000, 500000, 10);
    EXPECT_EQ(strategy2, ConstructionStrategy::TwoPass);
}

TEST(SparsityFactoryTest, EstimateNnz) {
    SparsityFactory factory;

    GlobalIndex estimate = factory.estimateNnz(100, 8);
    EXPECT_EQ(estimate, 100 * 64);  // n_elements * avg^2
}

TEST(SparsityFactoryTest, EstimateMemoryBytes) {
    SparsityFactory factory;

    auto bytes_standard = factory.estimateMemoryBytes(1000, 10000, ConstructionStrategy::Standard);
    auto bytes_twopass = factory.estimateMemoryBytes(1000, 10000, ConstructionStrategy::TwoPass);

    EXPECT_GT(bytes_standard, 0);
    EXPECT_GT(bytes_twopass, 0);
}

TEST(SparsityFactoryTest, AutoStrategySelection) {
    auto dof_map = createSimpleDofMap(100, 50, 4);

    FactoryOptions opts;
    opts.strategy = ConstructionStrategy::Auto;

    SparsityFactory factory;
    auto result = factory.create(dof_map, opts);

    EXPECT_TRUE(result.pattern != nullptr);
    // Strategy should be selected automatically
    EXPECT_NE(result.strategy_used, ConstructionStrategy::Auto);
}

TEST(SparsityFactoryTest, ForceTwoPassStrategy) {
    auto dof_map = createSimpleDofMap(50, 20, 4);

    FactoryOptions opts;
    opts.strategy = ConstructionStrategy::TwoPass;

    SparsityFactory factory;
    auto result = factory.create(dof_map, opts);

    EXPECT_TRUE(result.pattern != nullptr);
    EXPECT_EQ(result.strategy_used, ConstructionStrategy::TwoPass);
}

// ============================================================================
// Result Metadata Tests
// ============================================================================

TEST(SparsityFactoryTest, ConstructionTimeRecorded) {
    auto dof_map = createSimpleDofMap(100, 50, 4);

    SparsityFactory factory;
    auto result = factory.create(dof_map);

    EXPECT_GE(result.construction_time_sec, 0.0);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(SparsityFactoryTest, CreatePatternFunction) {
    auto dof_map = createSimpleDofMap(10, 4, 3);

    auto pattern = createPattern(dof_map);

    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_EQ(pattern.numRows(), 10);
}

TEST(SparsityFactoryTest, CreateOptimizedPatternFunction) {
    auto dof_map = createOverlappingDofMap(20, 15);

    auto pattern = createOptimizedPattern(dof_map);

    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_EQ(pattern.numRows(), 20);
}

TEST(SparsityFactoryTest, CreateDGPatternFunction) {
    auto dof_map = createSimpleDofMap(12, 4, 3);
    std::vector<std::pair<GlobalIndex, GlobalIndex>> adjacency = {{0, 1}, {1, 2}};

    auto pattern = createDGPatternFromMap(dof_map, adjacency);

    EXPECT_TRUE(pattern.isFinalized());
}

TEST(SparsityFactoryTest, RecommendOptionsFunction) {
    auto opts = recommendOptions(10000, 5000, false, false);

    EXPECT_EQ(opts.backend, TargetBackend::Generic);
    EXPECT_FALSE(opts.apply_constraints);

    auto dist_opts = recommendOptions(10000, 5000, true, true);

    EXPECT_EQ(dist_opts.backend, TargetBackend::PETSc);
    EXPECT_TRUE(dist_opts.apply_constraints);
    EXPECT_TRUE(dist_opts.include_ghost_rows);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST(SparsityFactoryTest, EmptyDofMap) {
    DofMap empty_map(0, 0, 0);
    empty_map.setNumDofs(0);
    empty_map.finalize();

    SparsityFactory factory;
    auto result = factory.create(empty_map);

    EXPECT_TRUE(result.pattern != nullptr);
    EXPECT_EQ(result.pattern->numRows(), 0);
}

TEST(SparsityFactoryTest, SingleElementDofMap) {
    DofMap single_map(1, 3, 3);
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    single_map.setCellDofs(0, dofs);
    single_map.setNumDofs(3);
    single_map.finalize();

    SparsityFactory factory;
    auto pattern = factory.createFromDofMap(single_map);

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.getNnz(), 9);  // 3x3 dense
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST(SparsityFactoryTest, DeterministicCreation) {
    auto create_pattern = []() {
        auto dof_map = createOverlappingDofMap(20, 15);
        SparsityFactory factory;
        return factory.createFromDofMap(dof_map);
    };

    auto p1 = create_pattern();
    auto p2 = create_pattern();

    EXPECT_EQ(p1.getNnz(), p2.getNnz());

    auto ci1 = p1.getColIndices();
    auto ci2 = p2.getColIndices();
    EXPECT_TRUE(std::equal(ci1.begin(), ci1.end(), ci2.begin()));
}
