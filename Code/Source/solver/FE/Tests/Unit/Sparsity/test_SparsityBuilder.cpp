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
#include "Sparsity/SparsityBuilder.h"
#include "Dofs/BlockDofMap.h"
#include <vector>
#include <memory>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;
using namespace svmp::FE::dofs;

// ============================================================================
// Mock DOF Map Query for testing
// ============================================================================

class MockDofMapQuery : public IDofMapQuery {
public:
    MockDofMapQuery(GlobalIndex n_dofs, GlobalIndex n_cells,
                    std::vector<std::vector<GlobalIndex>> cell_dofs)
        : n_dofs_(n_dofs), n_cells_(n_cells), cell_dofs_(std::move(cell_dofs)) {}

    [[nodiscard]] GlobalIndex getNumDofs() const override { return n_dofs_; }
    [[nodiscard]] GlobalIndex getNumLocalDofs() const override { return n_dofs_; }
    [[nodiscard]] GlobalIndex getNumCells() const override { return n_cells_; }

    [[nodiscard]] std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const override {
        return cell_dofs_[static_cast<std::size_t>(cell_id)];
    }

    [[nodiscard]] bool isOwnedDof(GlobalIndex /*dof*/) const override { return true; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getOwnedRange() const override {
        return {0, n_dofs_};
    }

private:
    GlobalIndex n_dofs_;
    GlobalIndex n_cells_;
    std::vector<std::vector<GlobalIndex>> cell_dofs_;
};

// ============================================================================
// Helper to create simple mesh connectivity
// ============================================================================

std::shared_ptr<MockDofMapQuery> createLineMesh(GlobalIndex n_elements) {
    // 1D mesh with n_elements linear elements
    // Element i has nodes i and i+1
    GlobalIndex n_nodes = n_elements + 1;
    std::vector<std::vector<GlobalIndex>> cell_dofs(static_cast<std::size_t>(n_elements));

    for (GlobalIndex e = 0; e < n_elements; ++e) {
        cell_dofs[static_cast<std::size_t>(e)] = {e, e + 1};
    }

    return std::make_shared<MockDofMapQuery>(n_nodes, n_elements, cell_dofs);
}

std::shared_ptr<MockDofMapQuery> createQuadMesh(GlobalIndex nx, GlobalIndex ny) {
    // 2D mesh of nx x ny quadrilateral elements
    // Node numbering: row-major, nodes_per_row = nx + 1
    GlobalIndex nodes_per_row = nx + 1;
    GlobalIndex n_nodes = (nx + 1) * (ny + 1);
    GlobalIndex n_elements = nx * ny;

    std::vector<std::vector<GlobalIndex>> cell_dofs(static_cast<std::size_t>(n_elements));

    for (GlobalIndex ey = 0; ey < ny; ++ey) {
        for (GlobalIndex ex = 0; ex < nx; ++ex) {
            GlobalIndex e = ey * nx + ex;
            GlobalIndex n0 = ey * nodes_per_row + ex;
            GlobalIndex n1 = n0 + 1;
            GlobalIndex n2 = n0 + nodes_per_row + 1;
            GlobalIndex n3 = n0 + nodes_per_row;

            cell_dofs[static_cast<std::size_t>(e)] = {n0, n1, n2, n3};
        }
    }

    return std::make_shared<MockDofMapQuery>(n_nodes, n_elements, cell_dofs);
}

// ============================================================================
// SparsityBuilder Tests
// ============================================================================

TEST(SparsityBuilderTest, BuildFromSingleElement) {
    // Single element with 3 DOFs
    auto dof_map = std::make_shared<MockDofMapQuery>(
        3, 1, std::vector<std::vector<GlobalIndex>>{{0, 1, 2}});

    SparsityBuilder builder(dof_map);
    auto pattern = builder.build();

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 3);
    EXPECT_EQ(pattern.getNnz(), 9);  // Full 3x3 block

    // All pairs should exist
    for (GlobalIndex i = 0; i < 3; ++i) {
        for (GlobalIndex j = 0; j < 3; ++j) {
            EXPECT_TRUE(pattern.hasEntry(i, j)) << "Missing entry (" << i << "," << j << ")";
        }
    }
}

TEST(SparsityBuilderTest, BuildFromLineMesh) {
    // 1D mesh with 5 elements: 0-1-2-3-4-5 (6 nodes)
    auto dof_map = createLineMesh(5);

    SparsityBuilder builder(dof_map);
    auto pattern = builder.build();

    EXPECT_EQ(pattern.numRows(), 6);
    EXPECT_EQ(pattern.numCols(), 6);
    EXPECT_TRUE(pattern.hasAllDiagonals());

    // Check tridiagonal structure
    for (GlobalIndex i = 0; i < 6; ++i) {
        EXPECT_TRUE(pattern.hasEntry(i, i));  // Diagonal
        if (i > 0) EXPECT_TRUE(pattern.hasEntry(i, i - 1));  // Sub-diagonal
        if (i < 5) EXPECT_TRUE(pattern.hasEntry(i, i + 1));  // Super-diagonal
    }

    // Interior nodes should have exactly 3 entries
    EXPECT_EQ(pattern.getRowNnz(2), 3);
    EXPECT_EQ(pattern.getRowNnz(3), 3);

    // Boundary nodes should have exactly 2 entries
    EXPECT_EQ(pattern.getRowNnz(0), 2);
    EXPECT_EQ(pattern.getRowNnz(5), 2);
}

TEST(SparsityBuilderTest, BuildFromQuadMesh) {
    // 2x2 quad mesh: 9 nodes
    auto dof_map = createQuadMesh(2, 2);

    SparsityBuilder builder(dof_map);
    auto pattern = builder.build();

    EXPECT_EQ(pattern.numRows(), 9);
    EXPECT_EQ(pattern.numCols(), 9);
    EXPECT_TRUE(pattern.hasAllDiagonals());

    // Corner node (0,0) -> node 0: should couple with nodes 0,1,3,4
    auto row0 = pattern.getRowIndices(0);
    std::vector<GlobalIndex> expected0 = {0, 1, 3, 4};
    std::vector<GlobalIndex> actual0(row0.begin(), row0.end());
    EXPECT_EQ(actual0, expected0);

    // Center node -> node 4: should couple with all 9 nodes
    EXPECT_EQ(pattern.getRowNnz(4), 9);
}

TEST(SparsityBuilderTest, BuildWithOptions) {
    auto dof_map = createLineMesh(3);

    SparsityBuildOptions options;
    options.ensure_diagonal = true;
    options.ensure_non_empty_rows = true;
    options.symmetric_pattern = true;

    SparsityBuilder builder(dof_map);
    builder.setOptions(options);
    auto pattern = builder.build();

    EXPECT_TRUE(pattern.hasAllDiagonals());
    EXPECT_TRUE(pattern.isSymmetric());
}

TEST(SparsityBuilderTest, BuildRectangular) {
    // Row DOFs: 2 elements, 3 DOFs each
    auto row_map = std::make_shared<MockDofMapQuery>(
        6, 2, std::vector<std::vector<GlobalIndex>>{{0, 1, 2}, {3, 4, 5}});

    // Col DOFs: 2 elements, 2 DOFs each
    auto col_map = std::make_shared<MockDofMapQuery>(
        4, 2, std::vector<std::vector<GlobalIndex>>{{0, 1}, {2, 3}});

    SparsityBuilder builder;
    builder.setRowDofMap(row_map);
    builder.setColDofMap(col_map);

    auto pattern = builder.build();

    EXPECT_EQ(pattern.numRows(), 6);
    EXPECT_EQ(pattern.numCols(), 4);
    EXPECT_FALSE(pattern.isSquare());

    // Element 0: rows 0,1,2 couple with cols 0,1
    for (GlobalIndex r = 0; r < 3; ++r) {
        EXPECT_TRUE(pattern.hasEntry(r, 0));
        EXPECT_TRUE(pattern.hasEntry(r, 1));
    }

    // Element 1: rows 3,4,5 couple with cols 2,3
    for (GlobalIndex r = 3; r < 6; ++r) {
        EXPECT_TRUE(pattern.hasEntry(r, 2));
        EXPECT_TRUE(pattern.hasEntry(r, 3));
    }
}

TEST(SparsityBuilderTest, BuildRectangularWithMismatchedCellCountsThrows) {
    auto row_map = std::make_shared<MockDofMapQuery>(
        6, 2, std::vector<std::vector<GlobalIndex>>{{0, 1, 2}, {3, 4, 5}});

    // Mismatched number of cells (1 instead of 2).
    auto col_map = std::make_shared<MockDofMapQuery>(
        4, 1, std::vector<std::vector<GlobalIndex>>{{0, 1}});

    SparsityBuilder builder;
    builder.setRowDofMap(row_map);
    builder.setColDofMap(col_map);

    EXPECT_THROW((void)builder.build(), svmp::FE::FEException);
}

TEST(SparsityBuilderTest, BuildSubset) {
    // 5 element line mesh
    auto dof_map = createLineMesh(5);

    SparsityBuilder builder(dof_map);

    // Only build for elements 1 and 3
    std::vector<GlobalIndex> cells = {1, 3};
    auto pattern = builder.build(cells);

    // Should have 6 nodes but only couplings from elements 1 (nodes 1,2) and 3 (nodes 3,4)
    EXPECT_EQ(pattern.numRows(), 6);

    // Node 0 should have 1 entry (just diagonal from ensure_diagonal)
    // Node 1 should have 2 entries (1,2) from element 1
    // Node 2 should have 2 entries (1,2) from element 1
    // Node 5 should have 1 entry (just diagonal)
    EXPECT_EQ(pattern.getRowNnz(0), 1);
    EXPECT_EQ(pattern.getRowNnz(1), 2);
    EXPECT_EQ(pattern.getRowNnz(2), 2);
    EXPECT_EQ(pattern.getRowNnz(3), 2);
    EXPECT_EQ(pattern.getRowNnz(4), 2);
    EXPECT_EQ(pattern.getRowNnz(5), 1);

    EXPECT_TRUE(pattern.hasEntry(1, 2));
    EXPECT_FALSE(pattern.hasEntry(1, 0));
    EXPECT_TRUE(pattern.hasEntry(3, 4));
    EXPECT_FALSE(pattern.hasEntry(3, 2));
}

TEST(SparsityBuilderTest, BuildWithCustomDofGetter) {
    GlobalIndex n_rows = 4;
    GlobalIndex n_cols = 4;
    GlobalIndex n_elements = 2;

    std::vector<std::vector<GlobalIndex>> row_dofs = {{0, 1}, {2, 3}};
    std::vector<std::vector<GlobalIndex>> col_dofs = {{0, 1}, {2, 3}};

    SparsityBuilder builder;
    auto pattern = builder.build(
        n_rows, n_cols, n_elements,
        [&](GlobalIndex e) { return std::span<const GlobalIndex>(row_dofs[static_cast<std::size_t>(e)]); },
        [&](GlobalIndex e) { return std::span<const GlobalIndex>(col_dofs[static_cast<std::size_t>(e)]); }
    );

    EXPECT_EQ(pattern.numRows(), 4);
    EXPECT_EQ(pattern.numCols(), 4);

    // Block diagonal pattern
    EXPECT_TRUE(pattern.hasEntry(0, 0));
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(1, 0));
    EXPECT_TRUE(pattern.hasEntry(1, 1));
    EXPECT_TRUE(pattern.hasEntry(2, 2));
    EXPECT_TRUE(pattern.hasEntry(2, 3));
    EXPECT_TRUE(pattern.hasEntry(3, 2));
    EXPECT_TRUE(pattern.hasEntry(3, 3));

    EXPECT_FALSE(pattern.hasEntry(0, 2));
    EXPECT_FALSE(pattern.hasEntry(2, 0));
}

// ============================================================================
// Multi-field coupling mode tests
// ============================================================================

TEST(SparsityBuilderTest, DiagonalCouplingModeBuildsBlockDiagonal) {
    // Two-field monolithic system: field0 DOFs [0..2], field1 DOFs [3..4]
    auto dof_map = std::make_shared<MockDofMapQuery>(
        5, 1, std::vector<std::vector<GlobalIndex>>{{0, 1, 3, 4}});

    BlockDofMap blocks;
    blocks.addBlock("field0", 3);
    blocks.addBlock("field1", 2);
    blocks.finalize();

    SparsityBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;
    opts.symmetric_pattern = false;

    SparsityBuilder builder(dof_map);
    builder.setOptions(opts);
    builder.setRowFieldMap(blocks);
    builder.setCouplingMode(CouplingMode::Diagonal);

    auto pattern = builder.build();

    // Within-field couplings should exist.
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(3, 4));

    // Cross-field couplings should not exist.
    EXPECT_FALSE(pattern.hasEntry(0, 3));
    EXPECT_FALSE(pattern.hasEntry(3, 0));
}

TEST(SparsityBuilderTest, CustomCouplingModeRespectsBlockPairs) {
    auto dof_map = std::make_shared<MockDofMapQuery>(
        5, 1, std::vector<std::vector<GlobalIndex>>{{0, 1, 3, 4}});

    BlockDofMap blocks;
    blocks.addBlock("field0", 3);
    blocks.addBlock("field1", 2);
    blocks.finalize();

    SparsityBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;
    opts.symmetric_pattern = false;

    SparsityBuilder builder(dof_map);
    builder.setOptions(opts);
    builder.setRowFieldMap(blocks);

    // Only allow field0 -> field1 (no within-field coupling, no reverse).
    builder.addCoupling(0, 1, false);

    auto pattern = builder.build();

    EXPECT_TRUE(pattern.hasEntry(0, 3));
    EXPECT_TRUE(pattern.hasEntry(1, 4));

    EXPECT_FALSE(pattern.hasEntry(3, 0));
    EXPECT_FALSE(pattern.hasEntry(0, 1));
    EXPECT_FALSE(pattern.hasEntry(3, 4));
}

TEST(SparsityBuilderTest, SymmetricPatternAddsTransposeRegardlessOfCouplingDirection) {
    auto dof_map = std::make_shared<MockDofMapQuery>(
        5, 1, std::vector<std::vector<GlobalIndex>>{{0, 1, 3, 4}});

    BlockDofMap blocks;
    blocks.addBlock("field0", 3);
    blocks.addBlock("field1", 2);
    blocks.finalize();

    SparsityBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;
    opts.symmetric_pattern = true;

    SparsityBuilder builder(dof_map);
    builder.setOptions(opts);
    builder.setRowFieldMap(blocks);

    // One-way coupling requested.
    builder.addCoupling(0, 1, false);

    auto pattern = builder.build();

    // Base coupling plus symmetric closure.
    EXPECT_TRUE(pattern.hasEntry(0, 3));
    EXPECT_TRUE(pattern.hasEntry(3, 0));
}

TEST(SparsityBuilderTest, CouplingModeWithoutFieldMapThrows) {
    auto dof_map = std::make_shared<MockDofMapQuery>(
        5, 1, std::vector<std::vector<GlobalIndex>>{{0, 1, 3, 4}});

    SparsityBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityBuilder builder(dof_map);
    builder.setOptions(opts);
    builder.setCouplingMode(CouplingMode::Diagonal);

    EXPECT_THROW((void)builder.build(), svmp::FE::FEException);
}

// ============================================================================
// Static Build Methods Tests
// ============================================================================

TEST(SparsityBuilderTest, BuildFromArrays) {
    GlobalIndex n_dofs = 5;
    GlobalIndex n_elements = 2;

    // Element 0: DOFs 0, 1, 2
    // Element 1: DOFs 2, 3, 4
    std::vector<GlobalIndex> elem_offsets = {0, 3, 6};
    std::vector<GlobalIndex> elem_dofs = {0, 1, 2, 2, 3, 4};

    auto pattern = SparsityBuilder::buildFromArrays(
        n_dofs, n_elements, elem_offsets, elem_dofs);

    EXPECT_EQ(pattern.numRows(), 5);
    EXPECT_EQ(pattern.numCols(), 5);
    EXPECT_TRUE(pattern.hasAllDiagonals());

    // Node 2 is shared: should couple with 0,1,2,3,4
    EXPECT_EQ(pattern.getRowNnz(2), 5);
}

TEST(SparsityBuilderTest, BuildFromArraysRectangular) {
    GlobalIndex n_rows = 4;
    GlobalIndex n_cols = 3;
    GlobalIndex n_elements = 2;

    std::vector<GlobalIndex> row_offsets = {0, 2, 4};
    std::vector<GlobalIndex> row_dofs = {0, 1, 2, 3};
    std::vector<GlobalIndex> col_offsets = {0, 2, 4};
    std::vector<GlobalIndex> col_dofs = {0, 1, 1, 2};

    SparsityBuildOptions opts;
    opts.ensure_diagonal = false;

    auto pattern = SparsityBuilder::buildFromArrays(
        n_rows, n_cols, n_elements,
        row_offsets, row_dofs, col_offsets, col_dofs, opts);

    EXPECT_EQ(pattern.numRows(), 4);
    EXPECT_EQ(pattern.numCols(), 3);
    EXPECT_FALSE(pattern.isSquare());
}

// ============================================================================
// DistributedSparsityBuilder Tests
// ============================================================================

TEST(DistributedSparsityBuilderTest, BasicBuild) {
    // 10 global DOFs, this "rank" owns DOFs 0-4
    auto dof_map = createLineMesh(9);  // 10 nodes

    DistributedSparsityBuilder builder;
    builder.setRowOwnership(0, 5, 10);
    builder.setColOwnership(0, 5, 10);
    builder.setRowDofMap(dof_map);

    auto pattern = builder.build();

    EXPECT_EQ(pattern.globalRows(), 10);
    EXPECT_EQ(pattern.globalCols(), 10);
    EXPECT_EQ(pattern.numOwnedRows(), 5);
    EXPECT_EQ(pattern.numOwnedCols(), 5);

    // Row 4 (last owned) couples with node 5 (ghost)
    EXPECT_GT(pattern.numGhostCols(), 0);
}

TEST(DistributedSparsityBuilderTest, DiagOffdiagSplit) {
    // Simple case: 6 DOFs, own 0-2, elements couple across boundary
    auto dof_map = std::make_shared<MockDofMapQuery>(
        6, 2,
        std::vector<std::vector<GlobalIndex>>{
            {0, 1, 2, 3},  // Element crosses into ghost region
            {2, 3, 4, 5}   // Also crosses
        });

    DistributedSparsityBuilder builder;
    builder.setRowOwnership(0, 3, 6);  // Own rows 0,1,2
    builder.setColOwnership(0, 3, 6);  // Own cols 0,1,2
    builder.setRowDofMap(dof_map);

    auto pattern = builder.build();

    EXPECT_EQ(pattern.numOwnedRows(), 3);
    EXPECT_EQ(pattern.numOwnedCols(), 3);

    // Ghost columns should be 3, 4, 5
    EXPECT_EQ(pattern.numGhostCols(), 3);

    // Diagonal entries (couplings within owned cols)
    EXPECT_GT(pattern.getDiagNnz(), 0);

    // Off-diagonal entries (couplings to ghost cols)
    EXPECT_GT(pattern.getOffdiagNnz(), 0);
}

TEST(DistributedSparsityBuilderTest, BuildSubset) {
    auto dof_map = createLineMesh(5);

    DistributedSparsityBuilder builder;
    builder.setRowOwnership(0, 3, 6);
    builder.setColOwnership(0, 3, 6);
    builder.setRowDofMap(dof_map);

    // Only build from element 1 (nodes 1,2)
    std::vector<GlobalIndex> cells = {1};
    auto pattern = builder.build(cells);

    EXPECT_EQ(pattern.numOwnedRows(), 3);

    // Row 1 and 2 should have entries, row 0 should be minimal (diagonal only)
    // Pattern is already finalized by build(), no need to call finalize() again
    EXPECT_EQ(pattern.numGhostCols(), 0);

    EXPECT_EQ(pattern.getRowNnz(0), 1);
    EXPECT_EQ(pattern.getRowDiagNnz(0), 1);
    EXPECT_EQ(pattern.getRowOffdiagNnz(0), 0);

    EXPECT_EQ(pattern.getRowNnz(1), 2);
    EXPECT_EQ(pattern.getRowOffdiagNnz(1), 0);

    EXPECT_EQ(pattern.getRowNnz(2), 2);
    EXPECT_EQ(pattern.getRowOffdiagNnz(2), 0);
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST(SparsityBuilderTest, DeterministicBuild) {
    // Build same pattern twice, verify identical results
    auto build_pattern = []() {
        auto dof_map = createQuadMesh(3, 3);
        SparsityBuilder builder(dof_map);
        return builder.build();
    };

    auto p1 = build_pattern();
    auto p2 = build_pattern();

    EXPECT_EQ(p1.getNnz(), p2.getNnz());

    auto rp1 = p1.getRowPtr();
    auto rp2 = p2.getRowPtr();
    EXPECT_TRUE(std::equal(rp1.begin(), rp1.end(), rp2.begin()));

    auto ci1 = p1.getColIndices();
    auto ci2 = p2.getColIndices();
    EXPECT_TRUE(std::equal(ci1.begin(), ci1.end(), ci2.begin()));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SparsityBuilderTest, EmptyMesh) {
    auto dof_map = std::make_shared<MockDofMapQuery>(
        5, 0, std::vector<std::vector<GlobalIndex>>{});

    SparsityBuilder builder(dof_map);
    auto pattern = builder.build();

    EXPECT_EQ(pattern.numRows(), 5);
    // With ensure_diagonal=true, should have 5 entries
    EXPECT_EQ(pattern.getNnz(), 5);
}

TEST(SparsityBuilderTest, SingleNodeElements) {
    // Elements with only 1 DOF each
    auto dof_map = std::make_shared<MockDofMapQuery>(
        3, 3,
        std::vector<std::vector<GlobalIndex>>{{0}, {1}, {2}});

    SparsityBuilder builder(dof_map);
    auto pattern = builder.build();

    // Diagonal only pattern
    EXPECT_EQ(pattern.getNnz(), 3);
    for (GlobalIndex i = 0; i < 3; ++i) {
        EXPECT_EQ(pattern.getRowNnz(i), 1);
    }
}

TEST(SparsityBuilderTest, LargeElement) {
    // Single element with many DOFs
    std::vector<GlobalIndex> dofs(20);
    for (GlobalIndex i = 0; i < 20; ++i) dofs[i] = i;

    auto dof_map = std::make_shared<MockDofMapQuery>(
        20, 1, std::vector<std::vector<GlobalIndex>>{dofs});

    SparsityBuilder builder(dof_map);
    auto pattern = builder.build();

    EXPECT_EQ(pattern.getNnz(), 400);  // 20x20 dense block
}
