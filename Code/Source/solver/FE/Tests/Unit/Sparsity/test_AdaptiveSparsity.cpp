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
#include "Sparsity/AdaptiveSparsity.h"
#include <set>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

SparsityPattern createQuadElementPattern(GlobalIndex n_elements_1d) {
    // Create pattern for n_elements_1d x n_elements_1d quad mesh
    GlobalIndex nx = n_elements_1d;
    GlobalIndex ny = n_elements_1d;
    GlobalIndex nodes_per_row = nx + 1;
    GlobalIndex n_nodes = (nx + 1) * (ny + 1);

    SparsityPattern pattern(n_nodes, n_nodes);

    for (GlobalIndex ey = 0; ey < ny; ++ey) {
        for (GlobalIndex ex = 0; ex < nx; ++ex) {
            GlobalIndex n0 = ey * nodes_per_row + ex;
            GlobalIndex n1 = n0 + 1;
            GlobalIndex n2 = n0 + nodes_per_row + 1;
            GlobalIndex n3 = n0 + nodes_per_row;

            std::vector<GlobalIndex> dofs = {n0, n1, n2, n3};
            for (GlobalIndex row : dofs) {
                for (GlobalIndex col : dofs) {
                    pattern.addEntry(row, col);
                }
            }
        }
    }

    pattern.finalize();
    return pattern;
}

SparsityPattern createTridiagonalPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(i, i);
        if (i > 0) pattern.addEntry(i, i - 1);
        if (i < n - 1) pattern.addEntry(i, i + 1);
    }
    pattern.finalize();
    return pattern;
}

} // anonymous namespace

// ============================================================================
// PatternDelta Tests
// ============================================================================

TEST(PatternDeltaTest, DefaultConstruction) {
    PatternDelta delta;
    EXPECT_FALSE(delta.hasChanges());
    EXPECT_EQ(delta.numAddedEntries(), 0);
    EXPECT_EQ(delta.numRemovedEntries(), 0);
}

TEST(PatternDeltaTest, AddEntry) {
    PatternDelta delta;
    delta.addEntry(0, 1);
    delta.addEntry(0, 2);
    delta.addEntry(1, 0);

    EXPECT_TRUE(delta.hasChanges());
    EXPECT_EQ(delta.numAddedEntries(), 3);
}

TEST(PatternDeltaTest, RemoveEntry) {
    PatternDelta delta;
    delta.removeEntry(0, 1);
    delta.removeEntry(0, 2);

    EXPECT_TRUE(delta.hasChanges());
    EXPECT_EQ(delta.numRemovedEntries(), 2);
}

TEST(PatternDeltaTest, AddThenRemoveCancels) {
    PatternDelta delta;
    delta.addEntry(0, 1);
    delta.removeEntry(0, 1);

    EXPECT_EQ(delta.numAddedEntries(), 0);
    EXPECT_EQ(delta.numRemovedEntries(), 0);
}

TEST(PatternDeltaTest, RemoveThenAddCancels) {
    PatternDelta delta;
    delta.removeEntry(0, 1);
    delta.addEntry(0, 1);

    EXPECT_EQ(delta.numAddedEntries(), 0);
    EXPECT_EQ(delta.numRemovedEntries(), 0);
}

TEST(PatternDeltaTest, AddRow) {
    PatternDelta delta;
    delta.addRow(5);
    delta.addRow(10);

    EXPECT_EQ(delta.getAddedRows().size(), 2);
    EXPECT_TRUE(delta.getAddedRows().count(5));
    EXPECT_TRUE(delta.getAddedRows().count(10));
}

TEST(PatternDeltaTest, RemoveRow) {
    PatternDelta delta;
    delta.removeRow(5);

    EXPECT_EQ(delta.getRemovedRows().size(), 1);
    EXPECT_TRUE(delta.getRemovedRows().count(5));
}

TEST(PatternDeltaTest, AddRemoveRowCancels) {
    PatternDelta delta;
    delta.addRow(5);
    delta.removeRow(5);

    EXPECT_EQ(delta.getAddedRows().size(), 0);
    EXPECT_EQ(delta.getRemovedRows().size(), 0);
}

TEST(PatternDeltaTest, Clear) {
    PatternDelta delta;
    delta.addEntry(0, 1);
    delta.removeEntry(1, 0);
    delta.addRow(5);

    delta.clear();

    EXPECT_FALSE(delta.hasChanges());
    EXPECT_EQ(delta.numAddedEntries(), 0);
    EXPECT_EQ(delta.numRemovedEntries(), 0);
}

TEST(PatternDeltaTest, Merge) {
    PatternDelta delta1;
    delta1.addEntry(0, 1);
    delta1.addEntry(0, 2);

    PatternDelta delta2;
    delta2.addEntry(1, 0);
    delta2.removeEntry(0, 1);  // Cancels delta1's add

    delta1.merge(delta2);

    EXPECT_EQ(delta1.numAddedEntries(), 2);  // (0,2) and (1,0)
}

// ============================================================================
// AdaptiveSparsity Construction Tests
// ============================================================================

TEST(AdaptiveSparsityTest, DefaultConstruction) {
    AdaptiveSparsity adaptive;
    EXPECT_EQ(adaptive.numRows(), 0);
    EXPECT_EQ(adaptive.numCols(), 0);
    EXPECT_EQ(adaptive.getNnz(), 0);
}

TEST(AdaptiveSparsityTest, ConstructWithDimensions) {
    AdaptiveSparsity adaptive(10, 10);
    EXPECT_EQ(adaptive.numRows(), 10);
    EXPECT_EQ(adaptive.numCols(), 10);
}

TEST(AdaptiveSparsityTest, ConstructWithPattern) {
    auto pattern = createTridiagonalPattern(10);
    GlobalIndex original_nnz = pattern.getNnz();

    AdaptiveSparsity adaptive(std::move(pattern));

    EXPECT_EQ(adaptive.numRows(), 10);
    EXPECT_EQ(adaptive.getNnz(), original_nnz);
}

TEST(AdaptiveSparsityTest, ReleasePattern) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    auto released = adaptive.releasePattern();

    EXPECT_EQ(released.numRows(), 10);
    EXPECT_EQ(adaptive.numRows(), 0);  // Moved from
}

// ============================================================================
// Refinement Tests
// ============================================================================

TEST(AdaptiveSparsityTest, ApplySimpleRefinement) {
    // Start with 2x2 quad mesh (9 nodes)
    auto pattern = createQuadElementPattern(2);
    GlobalIndex original_nnz = pattern.getNnz();

    AdaptiveSparsity adaptive(std::move(pattern));

    // Simulate refining one element by adding new interior DOFs
    RefinementEvent event;
    event.parent_element = 0;
    event.child_elements = {4, 5, 6, 7};  // 4 children
    event.old_dofs = {0, 1, 3, 4};  // Original element DOFs
    event.new_dofs = {0, 1, 3, 4, 9, 10, 11, 12};  // All DOFs after refinement
    event.added_dofs = {9, 10, 11, 12};  // New interior DOFs

    adaptive.applyRefinement(event);
    adaptive.finalize();

    EXPECT_GT(adaptive.getNnz(), original_nnz);
    EXPECT_GT(adaptive.numRows(), 9);
}

TEST(AdaptiveSparsityTest, ApplyMultipleRefinements) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    std::vector<RefinementEvent> events(3);
    for (int i = 0; i < 3; ++i) {
        events[static_cast<std::size_t>(i)].parent_element = i;
        events[static_cast<std::size_t>(i)].child_elements = {i * 2, i * 2 + 1};
        events[static_cast<std::size_t>(i)].old_dofs = {static_cast<GlobalIndex>(i)};
        events[static_cast<std::size_t>(i)].new_dofs = {static_cast<GlobalIndex>(10 + i * 2), static_cast<GlobalIndex>(10 + i * 2 + 1)};
        events[static_cast<std::size_t>(i)].added_dofs = {static_cast<GlobalIndex>(10 + i * 2), static_cast<GlobalIndex>(10 + i * 2 + 1)};
    }

    adaptive.applyRefinements(events);
    adaptive.finalize();

    auto stats = adaptive.getStats();
    EXPECT_EQ(stats.total_refinements, 3);
}

TEST(AdaptiveSparsityTest, AddDofCouplings) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    std::vector<GlobalIndex> new_dofs = {10, 11};
    std::vector<GlobalIndex> coupled_dofs = {4, 5, 6};

    adaptive.addDofCouplings(new_dofs, coupled_dofs);
    adaptive.finalize();

    const auto& updated = adaptive.getPattern();

    // New DOFs should couple with all specified DOFs
    EXPECT_TRUE(updated.hasEntry(10, 4));
    EXPECT_TRUE(updated.hasEntry(10, 5));
    EXPECT_TRUE(updated.hasEntry(10, 6));
    EXPECT_TRUE(updated.hasEntry(11, 4));

    // And vice versa
    EXPECT_TRUE(updated.hasEntry(4, 10));
    EXPECT_TRUE(updated.hasEntry(5, 10));

    // New DOFs should couple with each other
    EXPECT_TRUE(updated.hasEntry(10, 11));
    EXPECT_TRUE(updated.hasEntry(11, 10));
}

TEST(AdaptiveSparsityTest, AddElementCouplings) {
    AdaptiveSparsity adaptive(10, 10);

    std::vector<GlobalIndex> element_dofs = {0, 1, 2, 3};
    adaptive.addElementCouplings(element_dofs);
    adaptive.finalize();

    const auto& pattern = adaptive.getPattern();

    // All pairs should be coupled
    for (GlobalIndex i : element_dofs) {
        for (GlobalIndex j : element_dofs) {
            EXPECT_TRUE(pattern.hasEntry(i, j));
        }
    }

    // DOFs outside element should not be coupled
    EXPECT_FALSE(pattern.hasEntry(4, 5));
}

// ============================================================================
// Coarsening Tests
// ============================================================================

TEST(AdaptiveSparsityTest, ApplySimpleCoarsening) {
    // Start with a pattern where DOFs 5,6,7,8 will be removed
    SparsityPattern pattern(10, 10);
    for (GlobalIndex i = 0; i < 10; ++i) {
        pattern.addEntry(i, i);
        if (i < 9) pattern.addEntry(i, i + 1);
        if (i > 0) pattern.addEntry(i, i - 1);
    }
    pattern.finalize();

    AdaptiveSparsity adaptive(std::move(pattern));

    CoarseningEvent event;
    event.child_elements = {1, 2, 3, 4};
    event.parent_element = 0;
    event.old_dofs = {5, 6, 7, 8};
    event.new_dofs = {4, 9};  // Keep boundary DOFs
    event.removed_dofs = {5, 6, 7, 8};

    adaptive.applyCoarsening(event);
    adaptive.finalize();

    const auto& updated = adaptive.getPattern();

    // Removed DOFs should have no entries
    for (GlobalIndex removed : event.removed_dofs) {
        EXPECT_EQ(updated.getRowNnz(removed), 0);
    }
}

TEST(AdaptiveSparsityTest, RemoveDofs) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    std::vector<GlobalIndex> dofs_to_remove = {3, 5, 7};
    adaptive.removeDofs(dofs_to_remove);
    adaptive.finalize();

    const auto& updated = adaptive.getPattern();

    // Removed DOFs should have no entries
    for (GlobalIndex dof : dofs_to_remove) {
        EXPECT_EQ(updated.getRowNnz(dof), 0);
    }
}

TEST(AdaptiveSparsityTest, EliminateDofs) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    std::vector<GlobalIndex> dofs_to_eliminate = {3, 5, 7};
    adaptive.eliminateDofs(dofs_to_eliminate);
    adaptive.finalize();

    const auto& updated = adaptive.getPattern();

    // Eliminated DOFs should have only diagonal entry
    for (GlobalIndex dof : dofs_to_eliminate) {
        EXPECT_EQ(updated.getRowNnz(dof), 1);
        EXPECT_TRUE(updated.hasEntry(dof, dof));
    }
}

// ============================================================================
// Renumbering Tests
// ============================================================================

TEST(AdaptiveSparsityTest, ApplyRenumbering) {
    auto pattern = createTridiagonalPattern(5);
    AdaptiveSparsity adaptive(std::move(pattern));

    // Reverse numbering: 0->4, 1->3, 2->2, 3->1, 4->0
    DofRenumbering renumbering;
    renumbering.old_to_new = {4, 3, 2, 1, 0};
    renumbering.new_to_old = {4, 3, 2, 1, 0};
    renumbering.new_n_dofs = 5;

    adaptive.applyRenumbering(renumbering);

    const auto& updated = adaptive.getPattern();

    // Check that pattern structure is preserved (just with reversed indices)
    EXPECT_EQ(updated.numRows(), 5);
    EXPECT_TRUE(updated.hasEntry(4, 4));  // Was (0,0)
    EXPECT_TRUE(updated.hasEntry(4, 3));  // Was (0,1)
    EXPECT_TRUE(updated.hasEntry(0, 0));  // Was (4,4)
    EXPECT_TRUE(updated.hasEntry(0, 1));  // Was (4,3)
}

TEST(AdaptiveSparsityTest, IdentityRenumbering) {
    auto pattern = createTridiagonalPattern(5);
    GlobalIndex original_nnz = pattern.getNnz();
    AdaptiveSparsity adaptive(std::move(pattern));

    // Identity renumbering
    DofRenumbering renumbering;
    renumbering.old_to_new = {0, 1, 2, 3, 4};
    renumbering.new_to_old = {0, 1, 2, 3, 4};
    renumbering.new_n_dofs = 5;

    adaptive.applyRenumbering(renumbering);

    EXPECT_EQ(adaptive.getNnz(), original_nnz);
}

TEST(AdaptiveSparsityTest, Compact) {
    SparsityPattern pattern(10, 10);
    // Only add entries for rows 0, 2, 4, 6, 8 (skip odd rows)
    for (GlobalIndex i = 0; i < 10; i += 2) {
        pattern.addEntry(i, i);
        if (i + 2 < 10) pattern.addEntry(i, i + 2);
    }
    pattern.finalize();

    AdaptiveSparsity adaptive(std::move(pattern));

    std::vector<GlobalIndex> old_to_new;
    adaptive.compact(&old_to_new);

    EXPECT_EQ(adaptive.numRows(), 5);  // Only 5 non-empty rows
    EXPECT_EQ(old_to_new.size(), 10);
    EXPECT_EQ(old_to_new[0], 0);
    EXPECT_EQ(old_to_new[1], -1);  // Empty row removed
    EXPECT_EQ(old_to_new[2], 1);
}

// ============================================================================
// Pattern Merging Tests
// ============================================================================

TEST(AdaptiveSparsityTest, MergePattern) {
    auto pattern1 = createTridiagonalPattern(5);
    SparsityPattern pattern2(5, 5);
    pattern2.addEntry(0, 4);
    pattern2.addEntry(4, 0);
    pattern2.finalize();

    AdaptiveSparsity adaptive(std::move(pattern1));
    adaptive.merge(pattern2);
    adaptive.finalize();

    const auto& merged = adaptive.getPattern();

    // Should have both tridiagonal and new entries
    EXPECT_TRUE(merged.hasEntry(0, 0));
    EXPECT_TRUE(merged.hasEntry(0, 1));
    EXPECT_TRUE(merged.hasEntry(0, 4));  // New
    EXPECT_TRUE(merged.hasEntry(4, 0));  // New
}

TEST(AdaptiveSparsityTest, MergeWithLargerDimensions) {
    auto pattern1 = createTridiagonalPattern(5);
    SparsityPattern pattern2(10, 10);
    pattern2.addEntry(7, 8);
    pattern2.finalize();

    AdaptiveSparsity adaptive(std::move(pattern1));
    adaptive.merge(pattern2);
    adaptive.finalize();

    const auto& merged = adaptive.getPattern();

    EXPECT_GE(merged.numRows(), 10);
    EXPECT_TRUE(merged.hasEntry(7, 8));
}

// ============================================================================
// Finalization Tests
// ============================================================================

TEST(AdaptiveSparsityTest, FinalizeIncrementalUpdate) {
    AdaptiveSparsityConfig config;
    config.rebuild_threshold = 0.9;  // High threshold, prefer incremental

    auto pattern = createTridiagonalPattern(100);
    AdaptiveSparsity adaptive(std::move(pattern), config);

    // Small change within existing DOF range (no resize)
    // Add a coupling between existing DOFs that don't currently couple
    std::vector<GlobalIndex> new_dofs = {50};
    std::vector<GlobalIndex> coupled = {0};  // DOF 0 and 50 don't couple in tridiagonal
    adaptive.addDofCouplings(new_dofs, coupled);
    adaptive.finalize();

    auto stats = adaptive.getStats();
    EXPECT_EQ(stats.total_incremental_updates, 1);
}

TEST(AdaptiveSparsityTest, FinalizeFullRebuild) {
    AdaptiveSparsityConfig config;
    config.rebuild_threshold = 0.01;  // Low threshold, trigger rebuild

    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern), config);

    // Add many entries to trigger rebuild
    for (GlobalIndex i = 0; i < 10; ++i) {
        std::vector<GlobalIndex> elem_dofs = {static_cast<GlobalIndex>(10 + i), static_cast<GlobalIndex>(11 + i)};
        adaptive.addElementCouplings(elem_dofs);
    }
    adaptive.finalize();

    auto stats = adaptive.getStats();
    EXPECT_EQ(stats.total_rebuilds, 1);
}

TEST(AdaptiveSparsityTest, ClearDelta) {
    // Create pattern with room for DOF 9
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    // Add couplings within existing DOF range (no resize needed)
    std::vector<GlobalIndex> new_dofs = {5};  // Existing DOF
    std::vector<GlobalIndex> existing_dofs = {0, 1};
    adaptive.addDofCouplings(new_dofs, existing_dofs);
    EXPECT_TRUE(adaptive.hasPendingChanges());

    adaptive.clearDelta();
    // After clearing delta, only needs_rebuild_ flag remains if set
    // Since we didn't trigger resize, needs_rebuild_ should be false
    EXPECT_FALSE(adaptive.hasPendingChanges());
}

TEST(AdaptiveSparsityTest, RequestRebuild) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    EXPECT_FALSE(adaptive.needsRebuild());
    adaptive.requestRebuild();
    EXPECT_TRUE(adaptive.needsRebuild());

    adaptive.finalize();
    EXPECT_FALSE(adaptive.needsRebuild());
}

// ============================================================================
// Hierarchical Support Tests
// ============================================================================

TEST(AdaptiveSparsityTest, CreateCoarsePattern) {
    auto pattern = createTridiagonalPattern(8);
    AdaptiveSparsity adaptive(std::move(pattern));

    // Map every 2 fine DOFs to 1 coarse DOF
    std::vector<GlobalIndex> fine_to_coarse = {0, 0, 1, 1, 2, 2, 3, 3};

    auto coarse = adaptive.createCoarsePattern(fine_to_coarse);

    EXPECT_EQ(coarse.numRows(), 4);
    EXPECT_EQ(coarse.numCols(), 4);

    // Coarse DOFs should couple if fine DOFs coupled
    EXPECT_TRUE(coarse.hasEntry(0, 0));
    EXPECT_TRUE(coarse.hasEntry(0, 1));  // 0,1 -> 0; 2,3 -> 1; they couple in fine
}

TEST(AdaptiveSparsityTest, CreateRestrictionPattern) {
    auto pattern = createTridiagonalPattern(8);
    AdaptiveSparsity adaptive(std::move(pattern));

    std::vector<GlobalIndex> fine_to_coarse = {0, 0, 1, 1, 2, 2, 3, 3};

    auto restriction = adaptive.createRestrictionPattern(fine_to_coarse);

    EXPECT_EQ(restriction.numRows(), 4);   // Coarse
    EXPECT_EQ(restriction.numCols(), 8);   // Fine

    // Restriction: coarse DOF receives from corresponding fine DOFs
    EXPECT_TRUE(restriction.hasEntry(0, 0));
    EXPECT_TRUE(restriction.hasEntry(0, 1));
    EXPECT_TRUE(restriction.hasEntry(1, 2));
    EXPECT_TRUE(restriction.hasEntry(1, 3));
}

TEST(AdaptiveSparsityTest, CreateProlongationPattern) {
    auto pattern = createTridiagonalPattern(8);
    AdaptiveSparsity adaptive(std::move(pattern));

    std::vector<GlobalIndex> fine_to_coarse = {0, 0, 1, 1, 2, 2, 3, 3};

    auto prolongation = adaptive.createProlongationPattern(fine_to_coarse);

    EXPECT_EQ(prolongation.numRows(), 8);  // Fine
    EXPECT_EQ(prolongation.numCols(), 4);  // Coarse

    // Prolongation: fine DOF receives from corresponding coarse DOF
    EXPECT_TRUE(prolongation.hasEntry(0, 0));
    EXPECT_TRUE(prolongation.hasEntry(1, 0));
    EXPECT_TRUE(prolongation.hasEntry(2, 1));
    EXPECT_TRUE(prolongation.hasEntry(3, 1));
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(AdaptiveSparsityTest, Validate) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    EXPECT_TRUE(adaptive.validate());
}

TEST(AdaptiveSparsityTest, ValidationError) {
    AdaptiveSparsity adaptive(10, 10);
    auto error = adaptive.validationError();
    // Empty pattern should be valid
}

TEST(AdaptiveSparsityTest, GetStats) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    RefinementEvent event;
    event.parent_element = 0;
    event.child_elements = {1};
    event.old_dofs = {0};
    event.new_dofs = {0, 10};
    event.added_dofs = {10};
    adaptive.applyRefinement(event);

    CoarseningEvent coarsen;
    coarsen.parent_element = 0;
    coarsen.child_elements = {1};
    coarsen.old_dofs = {5, 6};
    coarsen.new_dofs = {5};
    coarsen.removed_dofs = {6};
    adaptive.applyCoarsening(coarsen);

    adaptive.finalize();

    auto stats = adaptive.getStats();
    EXPECT_EQ(stats.total_refinements, 1);
    EXPECT_EQ(stats.total_coarsenings, 1);
    EXPECT_GT(stats.total_dof_additions, 0);
    EXPECT_GT(stats.total_dof_removals, 0);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(AdaptiveSparsityTest, MakeAdaptive) {
    auto pattern = createTridiagonalPattern(10);
    auto adaptive = makeAdaptive(std::move(pattern));

    EXPECT_EQ(adaptive.numRows(), 10);
}

TEST(AdaptiveSparsityTest, ApplyRefinementConvenience) {
    auto pattern = createTridiagonalPattern(10);

    RefinementEvent event;
    event.parent_element = 0;
    event.child_elements = {1};
    event.old_dofs = {0, 1};
    event.new_dofs = {0, 1, 10};
    event.added_dofs = {10};

    applyRefinement(pattern, event);

    EXPECT_GE(pattern.numRows(), 11);
}

TEST(AdaptiveSparsityTest, ApplyCoarseningConvenience) {
    auto pattern = createTridiagonalPattern(10);

    CoarseningEvent event;
    event.parent_element = 0;
    event.child_elements = {1, 2};
    event.old_dofs = {5, 6, 7};
    event.new_dofs = {5};
    event.removed_dofs = {6, 7};

    applyCoarsening(pattern, event);

    EXPECT_EQ(pattern.getRowNnz(6), 0);
    EXPECT_EQ(pattern.getRowNnz(7), 0);
}

TEST(AdaptiveSparsityTest, CompactPatternConvenience) {
    SparsityPattern pattern(10, 10);
    pattern.addEntry(0, 0);
    pattern.addEntry(2, 2);
    pattern.addEntry(4, 4);
    pattern.addEntry(0, 2);
    pattern.addEntry(2, 0);
    pattern.addEntry(2, 4);
    pattern.addEntry(4, 2);
    pattern.finalize();

    std::vector<GlobalIndex> old_to_new;
    auto compact = compactPattern(pattern, &old_to_new);

    EXPECT_EQ(compact.numRows(), 3);  // Only 3 non-empty rows
    EXPECT_TRUE(compact.hasEntry(0, 0));  // Was (0,0)
    EXPECT_TRUE(compact.hasEntry(0, 1));  // Was (0,2)
    EXPECT_TRUE(compact.hasEntry(1, 2));  // Was (2,4)
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(AdaptiveSparsityTest, EmptyPattern) {
    AdaptiveSparsity adaptive(0, 0);
    EXPECT_EQ(adaptive.numRows(), 0);
    EXPECT_FALSE(adaptive.hasPendingChanges());
}

TEST(AdaptiveSparsityTest, InvalidRefinementEvent) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    RefinementEvent invalid_event;  // Default, invalid
    adaptive.applyRefinement(invalid_event);  // Should be no-op

    auto stats = adaptive.getStats();
    EXPECT_EQ(stats.total_refinements, 0);
}

TEST(AdaptiveSparsityTest, InvalidCoarseningEvent) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    CoarseningEvent invalid_event;  // Default, invalid
    adaptive.applyCoarsening(invalid_event);  // Should be no-op

    auto stats = adaptive.getStats();
    EXPECT_EQ(stats.total_coarsenings, 0);
}

TEST(AdaptiveSparsityTest, InvalidRenumbering) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern));

    DofRenumbering invalid;  // Empty, invalid
    adaptive.applyRenumbering(invalid);  // Should be no-op

    EXPECT_EQ(adaptive.numRows(), 10);  // Unchanged
}

TEST(AdaptiveSparsityTest, MoveConstruction) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive1(std::move(pattern));
    std::vector<GlobalIndex> new_dofs = {10};
    std::vector<GlobalIndex> existing_dofs = {0};
    adaptive1.addDofCouplings(new_dofs, existing_dofs);

    AdaptiveSparsity adaptive2(std::move(adaptive1));

    EXPECT_EQ(adaptive2.numRows(), 10);
    EXPECT_TRUE(adaptive2.hasPendingChanges());
}

TEST(AdaptiveSparsityTest, MoveAssignment) {
    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive1(std::move(pattern));
    AdaptiveSparsity adaptive2;

    adaptive2 = std::move(adaptive1);

    EXPECT_EQ(adaptive2.numRows(), 10);
}

TEST(AdaptiveSparsityTest, VerifyAfterUpdate) {
    AdaptiveSparsityConfig config;
    config.verify_after_update = true;

    auto pattern = createTridiagonalPattern(10);
    AdaptiveSparsity adaptive(std::move(pattern), config);

    std::vector<GlobalIndex> new_dofs = {10};
    std::vector<GlobalIndex> existing_dofs = {0, 1};
    adaptive.addDofCouplings(new_dofs, existing_dofs);
    adaptive.finalize();  // Should verify

    EXPECT_TRUE(adaptive.validate());
}
