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
#include "Sparsity/ConstraintSparsityAugmenter.h"
#include "Dofs/DofConstraints.h"
#include <vector>
#include <algorithm>
#include <memory>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;
using namespace svmp::FE::dofs;

// ============================================================================
// SimpleConstraintSet Tests
// ============================================================================

TEST(SimpleConstraintSetTest, EmptySet) {
    SimpleConstraintSet cs;
    EXPECT_EQ(cs.numConstraints(), 0);
    EXPECT_FALSE(cs.isConstrained(0));
    EXPECT_TRUE(cs.getAllConstrainedDofs().empty());
}

TEST(SimpleConstraintSetTest, AddDirichlet) {
    SimpleConstraintSet cs;
    cs.addDirichlet(5);

    EXPECT_EQ(cs.numConstraints(), 1);
    EXPECT_TRUE(cs.isConstrained(5));
    EXPECT_FALSE(cs.isConstrained(0));

    auto masters = cs.getMasterDofs(5);
    EXPECT_TRUE(masters.empty());  // Dirichlet has no masters
}

TEST(SimpleConstraintSetTest, AddSingleMasterConstraint) {
    SimpleConstraintSet cs;
    cs.addConstraint(5, 3);  // DOF 5 = DOF 3

    EXPECT_TRUE(cs.isConstrained(5));
    EXPECT_FALSE(cs.isConstrained(3));  // Master not constrained

    auto masters = cs.getMasterDofs(5);
    ASSERT_EQ(masters.size(), 1);
    EXPECT_EQ(masters[0], 3);
}

TEST(SimpleConstraintSetTest, AddMultipleMasterConstraint) {
    SimpleConstraintSet cs;
    std::vector<GlobalIndex> masters = {1, 3, 5};
    cs.addConstraint(10, masters);  // DOF 10 depends on DOFs 1, 3, 5

    EXPECT_TRUE(cs.isConstrained(10));

    auto retrieved = cs.getMasterDofs(10);
    ASSERT_EQ(retrieved.size(), 3);
    EXPECT_EQ(retrieved[0], 1);
    EXPECT_EQ(retrieved[1], 3);
    EXPECT_EQ(retrieved[2], 5);
}

TEST(SimpleConstraintSetTest, GetAllConstrainedDofs) {
    SimpleConstraintSet cs;
    cs.addDirichlet(5);
    cs.addConstraint(3, 1);
    cs.addConstraint(8, 2);

    auto constrained = cs.getAllConstrainedDofs();
    ASSERT_EQ(constrained.size(), 3);

    // Should be sorted
    EXPECT_EQ(constrained[0], 3);
    EXPECT_EQ(constrained[1], 5);
    EXPECT_EQ(constrained[2], 8);
}

TEST(SimpleConstraintSetTest, Clear) {
    SimpleConstraintSet cs;
    cs.addDirichlet(5);
    cs.addConstraint(3, 1);
    cs.clear();

    EXPECT_EQ(cs.numConstraints(), 0);
    EXPECT_FALSE(cs.isConstrained(5));
}

// ============================================================================
// ConstraintSparsityAugmenter Construction Tests
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, DefaultConstruction) {
    ConstraintSparsityAugmenter augmenter;
    EXPECT_FALSE(augmenter.hasConstraints());
}

TEST(ConstraintSparsityAugmenterTest, ConstructWithConstraintQuery) {
    auto cs = std::make_shared<SimpleConstraintSet>();
    cs->addDirichlet(5);

    ConstraintSparsityAugmenter augmenter(cs);
    EXPECT_TRUE(augmenter.hasConstraints());
}

TEST(ConstraintSparsityAugmenterTest, ConstructWithSparsityConstraints) {
    std::vector<SparsityConstraint> constraints;
    constraints.push_back({5, {}});  // Dirichlet
    constraints.push_back({3, {1, 2}});  // Multi-master

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    EXPECT_TRUE(augmenter.hasConstraints());
}

TEST(ConstraintSparsityAugmenterTest, DofConstraintsAdapterEliminationFill) {
    DofConstraints constraints;
    constraints.addPeriodicBC(1, 5);      // slave 5 depends on master 1
    constraints.addDirichletBC(2, 0.0);   // Dirichlet has no masters
    constraints.close();

    auto adapter = std::make_shared<DofConstraintsAdapter>(constraints);
    ConstraintSparsityAugmenter augmenter(adapter);

    SparsityPattern pattern(6, 6);
    pattern.addEntry(0, 5);  // Coupling to constrained DOF

    augmenter.augment(pattern, AugmentationMode::EliminationFill);

    // (0,5) induces (0,1) via periodic constraint 5 -> 1
    EXPECT_TRUE(pattern.hasEntry(0, 1));

    // ensure_diagonal=true by default; Dirichlet DOF should have diagonal present
    EXPECT_TRUE(pattern.hasEntry(2, 2));
}

// ============================================================================
// EliminationFill Mode Tests
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, EliminationFillSimple) {
    // Create pattern with coupling to constrained DOF
    SparsityPattern pattern(6, 6);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 2);  // Row 1 couples to column 2
    pattern.addEntry(2, 3);  // Column 2 (constrained) to column 3

    // DOF 2 is constrained to DOF 4
    std::vector<SparsityConstraint> constraints = {{2, {4}}};

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    auto stats = augmenter.augment(pattern, AugmentationMode::EliminationFill);

    // Original pattern + induced couplings
    EXPECT_GT(pattern.getNnz(), 3);

    // Row 1 should now couple to DOF 4 (master of constrained DOF 2)
    // Also diagonal of constrained DOF
    EXPECT_TRUE(pattern.hasEntry(2, 4));  // Constrained row to master
}

TEST(ConstraintSparsityAugmenterTest, EliminationFillMultipleMasters) {
    SparsityPattern pattern(10, 10);
    pattern.addEntry(0, 5);  // Row 0 couples to constrained DOF 5

    // DOF 5 is constrained to DOFs {1, 2, 3}
    std::vector<SparsityConstraint> constraints = {{5, {1, 2, 3}}};

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    augmenter.augment(pattern, AugmentationMode::EliminationFill);

    // Row 0 should couple to all masters: 1, 2, 3
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(0, 2));
    EXPECT_TRUE(pattern.hasEntry(0, 3));
}

TEST(ConstraintSparsityAugmenterTest, EliminationFillDirichlet) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 2);

    // DOF 3 is Dirichlet (no masters)
    std::vector<SparsityConstraint> constraints = {{3, {}}};

    AugmentationOptions opts;
    opts.ensure_diagonal = true;

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    augmenter.setOptions(opts);
    augmenter.augment(pattern, AugmentationMode::EliminationFill);

    // Diagonal of constrained DOF should be present
    EXPECT_TRUE(pattern.hasEntry(3, 3));
}

TEST(ConstraintSparsityAugmenterTest, DistributedEliminationFillAddsMastersInOwnedRows) {
    // Rank-local view: own rows/cols [0,2) of a 4x4 system (ghost cols allowed).
    DistributedSparsityPattern pattern(IndexRange{0, 2}, IndexRange{0, 2}, 4, 4);

    // Row 0 couples to constrained column 1.
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 1);

    // Row 1 is constrained (ensure_diagonal=true by default).
    pattern.addEntry(1, 0);
    pattern.addEntry(1, 1);

    // Constrain DOF 1 -> master DOF 2 (ghost for this rank).
    std::vector<SparsityConstraint> constraints = {{1, {2}}};
    ConstraintSparsityAugmenter augmenter(std::move(constraints));

    EXPECT_THROW(pattern.getOwnedRowGlobalCols(0), FEException);  // Not finalized yet
    augmenter.augment(pattern, AugmentationMode::EliminationFill);
    pattern.finalize();

    const auto cols0 = pattern.getOwnedRowGlobalCols(0);
    EXPECT_NE(std::find(cols0.begin(), cols0.end(), 2), cols0.end());
}

// ============================================================================
// KeepRowsSetDiag Mode Tests
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, KeepRowsSetDiag) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 1);
    // Row 3 is empty (constrained DOF)

    std::vector<SparsityConstraint> constraints = {{3, {1, 2}}};

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    augmenter.augment(pattern, AugmentationMode::KeepRowsSetDiag);

    // Diagonal should be set for constrained row
    EXPECT_TRUE(pattern.hasEntry(3, 3));

    // Should NOT expand to masters in this mode
    // (no structural fill to columns 1, 2)
}

TEST(ConstraintSparsityAugmenterTest, KeepRowsSetDiagMultiple) {
    SparsityPattern pattern(10, 10);

    // Multiple constrained DOFs
    std::vector<SparsityConstraint> constraints = {
        {2, {}},      // Dirichlet
        {5, {1}},     // Periodic
        {8, {3, 4}}   // MPC
    };

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    auto stats = augmenter.augment(pattern, AugmentationMode::KeepRowsSetDiag);

    EXPECT_TRUE(pattern.hasEntry(2, 2));
    EXPECT_TRUE(pattern.hasEntry(5, 5));
    EXPECT_TRUE(pattern.hasEntry(8, 8));

    EXPECT_EQ(stats.n_diagonal_added, 3);
}

// ============================================================================
// ReducedSystem Mode Tests
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, BuildReducedPattern) {
    // 5x5 pattern, eliminate DOF 2
    SparsityPattern original(5, 5);
    original.addEntry(0, 0);
    original.addEntry(0, 1);
    original.addEntry(0, 2);  // Couples to constrained
    original.addEntry(1, 1);
    original.addEntry(2, 0);  // Constrained row
    original.addEntry(2, 2);
    original.addEntry(3, 3);
    original.addEntry(4, 4);
    original.finalize();

    // DOF 2 constrained to DOF 1
    std::vector<SparsityConstraint> constraints = {{2, {1}}};

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    auto reduced = augmenter.buildReducedPattern(original);

    // Reduced system: 4x4 (DOFs 0, 1, 3, 4)
    EXPECT_EQ(reduced.numRows(), 4);
    EXPECT_EQ(reduced.numCols(), 4);

    // Check some entries exist
    EXPECT_TRUE(reduced.hasEntry(0, 0));  // (0,0) -> (0,0)
    EXPECT_TRUE(reduced.hasEntry(0, 1));  // (0,1) -> (0,1) or from (0,2)->(0,1)
}

TEST(ConstraintSparsityAugmenterTest, ReducedMappingCorrect) {
    std::vector<SparsityConstraint> constraints = {
        {2, {0}},
        {5, {1}}
    };

    ConstraintSparsityAugmenter augmenter(std::move(constraints));

    auto full_to_reduced = augmenter.getReducedMapping(8);

    // DOFs 0, 1, 3, 4, 6, 7 are unconstrained
    // DOFs 2, 5 are constrained
    EXPECT_EQ(full_to_reduced[0], 0);
    EXPECT_EQ(full_to_reduced[1], 1);
    EXPECT_EQ(full_to_reduced[2], -1);  // Constrained
    EXPECT_EQ(full_to_reduced[3], 2);
    EXPECT_EQ(full_to_reduced[4], 3);
    EXPECT_EQ(full_to_reduced[5], -1);  // Constrained
    EXPECT_EQ(full_to_reduced[6], 4);
    EXPECT_EQ(full_to_reduced[7], 5);
}

TEST(ConstraintSparsityAugmenterTest, NumUnconstrainedDofs) {
    std::vector<SparsityConstraint> constraints = {{2, {}}, {5, {}}, {8, {}}};

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    EXPECT_EQ(augmenter.numUnconstrainedDofs(10), 7);
}

// ============================================================================
// Transitive Closure Tests
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, TransitiveClosure) {
    // DOF 5 depends on DOF 3
    // DOF 3 depends on DOFs {1, 2}
    // Transitive: DOF 5 should effectively depend on {1, 2}
    SparsityPattern pattern(10, 10);
    pattern.addEntry(0, 5);  // Row 0 couples to DOF 5

    std::vector<SparsityConstraint> constraints = {
        {5, {3}},     // 5 -> 3
        {3, {1, 2}}   // 3 -> 1, 2
    };

    AugmentationOptions opts;
    opts.compute_transitive_closure = true;

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    augmenter.setOptions(opts);
    augmenter.augment(pattern, AugmentationMode::EliminationFill);

    // Row 0 should couple to DOFs 1 and 2 (transitive masters)
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(0, 2));
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, StatsReported) {
    SparsityPattern pattern(10, 10);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 1);

    std::vector<SparsityConstraint> constraints = {
        {2, {}},        // Dirichlet
        {3, {0}},       // Periodic
        {4, {5, 6, 7}}  // MPC
    };

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    auto stats = augmenter.augment(pattern, AugmentationMode::EliminationFill);

    EXPECT_EQ(stats.n_constraints, 3);
    EXPECT_EQ(stats.n_dirichlet, 1);
    EXPECT_EQ(stats.n_periodic, 1);
    EXPECT_EQ(stats.n_multipoint, 1);
    EXPECT_EQ(stats.original_nnz, 2);
    EXPECT_GT(stats.augmented_nnz, 2);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, ConvenienceAugment) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 2);

    std::vector<SparsityConstraint> constraints = {{2, {1}}};

    auto stats = augmentWithConstraints(pattern, constraints);
    EXPECT_TRUE(pattern.hasEntry(0, 1));
}

TEST(ConstraintSparsityAugmenterTest, ConvenienceBuildReduced) {
    SparsityPattern original(5, 5);
    original.addEntry(0, 0);
    original.addEntry(1, 1);
    original.addEntry(2, 2);
    original.addEntry(3, 3);
    original.addEntry(4, 4);
    original.finalize();

    std::vector<SparsityConstraint> constraints = {{2, {1}}};

    auto reduced = buildReducedSparsityPattern(original, constraints);
    EXPECT_EQ(reduced.numRows(), 4);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ConstraintSparsityAugmenterTest, EmptyPattern) {
    SparsityPattern pattern(5, 5);

    std::vector<SparsityConstraint> constraints = {{2, {1}}};

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    auto stats = augmenter.augment(pattern, AugmentationMode::EliminationFill);

    // Should still add diagonal for constrained DOF
    EXPECT_TRUE(pattern.hasEntry(2, 2));
}

TEST(ConstraintSparsityAugmenterTest, NoConstraints) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 1);

    std::vector<SparsityConstraint> constraints;  // Empty

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    EXPECT_FALSE(augmenter.hasConstraints());
}

TEST(ConstraintSparsityAugmenterTest, AllDirichlet) {
    SparsityPattern pattern(5, 5);

    std::vector<SparsityConstraint> constraints = {
        {0, {}}, {1, {}}, {2, {}}, {3, {}}, {4, {}}
    };

    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    augmenter.augment(pattern, AugmentationMode::KeepRowsSetDiag);

    // All diagonals should be present
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(pattern.hasEntry(i, i));
    }
}

TEST(ConstraintSparsityAugmenterTest, SymmetryPreservedForUnconstrainedNeighbors) {
    // 0 -- 1 (constrained -> 2)
    // Row 0 couples to 1. 1 depends on 2.
    // Result should couple 0 to 2.
    // If symmetric, should also couple 2 to 0.
    
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 0); // Symmetric input
    
    std::vector<SparsityConstraint> constraints = {{1, {2}}}; // 1 -> 2
    
    AugmentationOptions opts;
    opts.symmetric_fill = true;
    
    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    augmenter.setOptions(opts);
    augmenter.augment(pattern, AugmentationMode::EliminationFill);
    
    EXPECT_TRUE(pattern.hasEntry(0, 2)); // Induced: 0->1->2
    EXPECT_TRUE(pattern.hasEntry(2, 0)); // Symmetric induced: 2<-1<-0
}

TEST(ConstraintSparsityAugmenterTest, EliminationFillPropagatesCouplings) {
    // Test that (u_s, j) -> (u_m, j)
    // u_s = 1, u_m = 2
    // Matrix has entry (1, 0) -> u_s couples to 0
    // Result should have (2, 0) -> u_m couples to 0
    
    SparsityPattern pattern(3, 3);
    pattern.addEntry(1, 0); // Constrained row 1 couples to column 0
    
    std::vector<SparsityConstraint> constraints = {{1, {2}}}; // 1 -> 2
    
    ConstraintSparsityAugmenter augmenter(std::move(constraints));
    augmenter.augment(pattern, AugmentationMode::EliminationFill);
    
    // Original entry
    EXPECT_TRUE(pattern.hasEntry(1, 0));
    
    // Propagated entry: master 2 should inherit coupling to 0
    EXPECT_TRUE(pattern.hasEntry(2, 0));
}
