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
#include "Sparsity/DGSparsityBuilder.h"
#include <vector>
#include <memory>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Test fixtures and mocks
// ============================================================================

/**
 * Simple DG DOF map for testing - each cell has its own DOFs
 */
class SimpleDGDofMap : public IDGDofMapQuery {
public:
    SimpleDGDofMap(GlobalIndex n_cells, GlobalIndex dofs_per_cell)
        : n_cells_(n_cells), dofs_per_cell_(dofs_per_cell)
    {
        // Build cell DOF arrays
        cell_dofs_.resize(static_cast<std::size_t>(n_cells));
        GlobalIndex dof = 0;
        for (GlobalIndex c = 0; c < n_cells; ++c) {
            cell_dofs_[static_cast<std::size_t>(c)].resize(
                static_cast<std::size_t>(dofs_per_cell));
            for (GlobalIndex i = 0; i < dofs_per_cell; ++i) {
                cell_dofs_[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] = dof++;
            }
        }
        n_dofs_ = dof;
    }

    [[nodiscard]] GlobalIndex getNumDofs() const override { return n_dofs_; }
    [[nodiscard]] GlobalIndex getNumLocalDofs() const override { return n_dofs_; }

    [[nodiscard]] std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const override {
        return cell_dofs_[static_cast<std::size_t>(cell_id)];
    }

    [[nodiscard]] GlobalIndex getNumCells() const override { return n_cells_; }

    [[nodiscard]] bool isOwnedDof(GlobalIndex /*dof*/) const override { return true; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getOwnedRange() const override {
        return {0, n_dofs_};
    }

    [[nodiscard]] std::span<const GlobalIndex>
    getFaceDofs(GlobalIndex cell_id, LocalIndex /*local_face*/) const override {
        return getCellDofs(cell_id);
    }

    [[nodiscard]] LocalIndex getNumFacesPerCell(GlobalIndex /*cell_id*/) const override {
        return 4;
    }

    [[nodiscard]] bool isDG() const override { return true; }

private:
    GlobalIndex n_cells_;
    GlobalIndex dofs_per_cell_;
    GlobalIndex n_dofs_;
    std::vector<std::vector<GlobalIndex>> cell_dofs_;
};

// ============================================================================
// SimpleFaceConnectivity Tests
// ============================================================================

TEST(SimpleFaceConnectivityTest, EmptyConnectivity) {
    SimpleFaceConnectivity fc;
    EXPECT_EQ(fc.getNumInteriorFaces(), 0);
    EXPECT_EQ(fc.getNumBoundaryFaces(), 0);
}

TEST(SimpleFaceConnectivityTest, AddInteriorFace) {
    SimpleFaceConnectivity fc;
    GlobalIndex face_id = fc.addInteriorFace(0, 1);

    EXPECT_EQ(face_id, 0);
    EXPECT_EQ(fc.getNumInteriorFaces(), 1);

    auto [cell_plus, cell_minus] = fc.getInteriorFaceCells(0);
    EXPECT_EQ(cell_plus, 0);
    EXPECT_EQ(cell_minus, 1);
}

TEST(SimpleFaceConnectivityTest, AddBoundaryFace) {
    SimpleFaceConnectivity fc;
    GlobalIndex face_id = fc.addBoundaryFace(2, 100);  // cell 2, tag 100

    EXPECT_EQ(face_id, 0);
    EXPECT_EQ(fc.getNumBoundaryFaces(), 1);
    EXPECT_EQ(fc.getBoundaryFaceCell(0), 2);
    EXPECT_EQ(fc.getBoundaryTag(0), 100);
}

TEST(SimpleFaceConnectivityTest, MultipleInteriorFaces) {
    SimpleFaceConnectivity fc;
    fc.addInteriorFace(0, 1);
    fc.addInteriorFace(1, 2);
    fc.addInteriorFace(0, 3);

    EXPECT_EQ(fc.getNumInteriorFaces(), 3);

    auto [c0p, c0m] = fc.getInteriorFaceCells(0);
    EXPECT_EQ(c0p, 0);
    EXPECT_EQ(c0m, 1);

    auto [c2p, c2m] = fc.getInteriorFaceCells(2);
    EXPECT_EQ(c2p, 0);
    EXPECT_EQ(c2m, 3);
}

TEST(SimpleFaceConnectivityTest, Clear) {
    SimpleFaceConnectivity fc;
    fc.addInteriorFace(0, 1);
    fc.addBoundaryFace(0, 0);
    fc.clear();

    EXPECT_EQ(fc.getNumInteriorFaces(), 0);
    EXPECT_EQ(fc.getNumBoundaryFaces(), 0);
}

// ============================================================================
// DGSparsityBuilder Construction Tests
// ============================================================================

TEST(DGSparsityBuilderTest, DefaultConstruction) {
    DGSparsityBuilder builder;
    // Should not throw
}

TEST(DGSparsityBuilderTest, ConstructWithComponents) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(4, 3);  // 4 cells, 3 DOFs each
    auto fc = std::make_shared<SimpleFaceConnectivity>();

    DGSparsityBuilder builder(dof_map, fc);
    // Should not throw
}

// ============================================================================
// Cell Coupling Tests
// ============================================================================

TEST(DGSparsityBuilderTest, CellCouplingsOnly) {
    // 2 cells, 3 DOFs each -> 6 total DOFs
    auto dof_map = std::make_shared<SimpleDGDofMap>(2, 3);
    auto fc = std::make_shared<SimpleFaceConnectivity>();

    DGSparsityBuilder builder;
    builder.setDofMap(std::static_pointer_cast<IDGDofMapQuery>(dof_map));
    builder.setFaceConnectivity(fc);

    DGSparsityOptions opts;
    opts.include_cell_couplings = true;
    opts.include_face_couplings = false;
    opts.include_boundary_couplings = false;
    builder.setOptions(opts);

    SparsityPattern pattern = builder.build();

    // Each cell should have 3x3 = 9 couplings
    // Cell 0: DOFs 0, 1, 2
    // Cell 1: DOFs 3, 4, 5
    // Total: 18 entries (no cross-cell coupling)

    EXPECT_EQ(pattern.numRows(), 6);
    EXPECT_EQ(pattern.numCols(), 6);
    EXPECT_EQ(pattern.getNnz(), 18);

    // Check cell 0 couplings
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_TRUE(pattern.hasEntry(i, j));
        }
    }

    // Check cell 1 couplings
    for (int i = 3; i < 6; ++i) {
        for (int j = 3; j < 6; ++j) {
            EXPECT_TRUE(pattern.hasEntry(i, j));
        }
    }

    // No cross-cell coupling
    EXPECT_FALSE(pattern.hasEntry(0, 3));
    EXPECT_FALSE(pattern.hasEntry(3, 0));
}

// ============================================================================
// Face Coupling Tests
// ============================================================================

TEST(DGSparsityBuilderTest, FaceCouplingsAdded) {
    // 2 cells with one interior face
    auto dof_map = std::make_shared<SimpleDGDofMap>(2, 3);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addInteriorFace(0, 1);

    DGSparsityBuilder builder;
    builder.setDofMap(std::static_pointer_cast<IDGDofMapQuery>(dof_map));
    builder.setFaceConnectivity(fc);

    DGSparsityOptions opts;
    opts.include_cell_couplings = false;  // Only face couplings
    opts.include_face_couplings = true;
    opts.include_boundary_couplings = false;
    builder.setOptions(opts);

    SparsityPattern pattern = builder.build();

    // Face coupling creates:
    // - Cell 0 DOFs couple to Cell 0 DOFs (9)
    // - Cell 0 DOFs couple to Cell 1 DOFs (9)
    // - Cell 1 DOFs couple to Cell 0 DOFs (9)
    // - Cell 1 DOFs couple to Cell 1 DOFs (9)
    // Total: 36 entries (full coupling of both cells)

    // Cross-cell couplings should exist
    EXPECT_TRUE(pattern.hasEntry(0, 3));
    EXPECT_TRUE(pattern.hasEntry(0, 4));
    EXPECT_TRUE(pattern.hasEntry(3, 0));
    EXPECT_TRUE(pattern.hasEntry(5, 2));
}

TEST(DGSparsityBuilderTest, MultipleFaces) {
    // 3 cells in a line: 0 - 1 - 2
    auto dof_map = std::make_shared<SimpleDGDofMap>(3, 2);  // 6 total DOFs
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addInteriorFace(0, 1);  // Between cells 0 and 1
    fc->addInteriorFace(1, 2);  // Between cells 1 and 2

    DGSparsityBuilder builder(dof_map, fc);
    SparsityPattern pattern = builder.build();

    // Cell 0 couples to cell 1, cell 1 couples to cells 0 and 2
    // DOFs: 0-1 (cell 0), 2-3 (cell 1), 4-5 (cell 2)

    // 0-1 coupling exists (via face 0)
    EXPECT_TRUE(pattern.hasEntry(0, 2));
    EXPECT_TRUE(pattern.hasEntry(2, 0));

    // 1-2 coupling exists (via face 1)
    EXPECT_TRUE(pattern.hasEntry(2, 4));
    EXPECT_TRUE(pattern.hasEntry(4, 2));

    // 0-2 coupling should NOT exist (no direct face)
    EXPECT_FALSE(pattern.hasEntry(0, 4));
    EXPECT_FALSE(pattern.hasEntry(4, 0));
}

// ============================================================================
// Boundary Face Tests
// ============================================================================

TEST(DGSparsityBuilderTest, BoundaryFaceCouplings) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(2, 3);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addBoundaryFace(0, 1);  // Cell 0 has boundary face with tag 1
    fc->addBoundaryFace(1, 2);  // Cell 1 has boundary face with tag 2

    DGSparsityBuilder builder;
    builder.setDofMap(std::static_pointer_cast<IDGDofMapQuery>(dof_map));
    builder.setFaceConnectivity(fc);

    DGSparsityOptions opts;
    opts.include_cell_couplings = false;
    opts.include_face_couplings = false;
    opts.include_boundary_couplings = true;
    builder.setOptions(opts);

    SparsityPattern pattern = builder.build();

    // Boundary couplings are cell-local
    // Cell 0: DOFs 0,1,2 should couple
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(1, 2));

    // Cell 1: DOFs 3,4,5 should couple
    EXPECT_TRUE(pattern.hasEntry(3, 4));
    EXPECT_TRUE(pattern.hasEntry(4, 5));
}

TEST(DGSparsityBuilderTest, BoundaryFacesByTag) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(3, 2);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addBoundaryFace(0, 1);
    fc->addBoundaryFace(1, 2);
    fc->addBoundaryFace(2, 1);

    DGSparsityBuilder builder;
    builder.setDofMap(std::static_pointer_cast<IDGDofMapQuery>(dof_map));
    builder.setFaceConnectivity(fc);

    SparsityPattern pattern(6, 6);

    // Only add boundary couplings for tag 1
    builder.buildBoundaryFaceCouplings(pattern, 1);
    pattern.finalize();

    // Cells 0 and 2 have tag 1
    EXPECT_TRUE(pattern.hasEntry(0, 1));  // Cell 0
    EXPECT_TRUE(pattern.hasEntry(4, 5));  // Cell 2

    // Cell 1 has tag 2, should not be included
    EXPECT_FALSE(pattern.hasEntry(2, 3));
}

// ============================================================================
// Full Build Tests
// ============================================================================

TEST(DGSparsityBuilderTest, FullBuild) {
    // 4 cells in 2x2 arrangement
    auto dof_map = std::make_shared<SimpleDGDofMap>(4, 4);  // 16 DOFs
    auto fc = std::make_shared<SimpleFaceConnectivity>();

    // Interior faces
    fc->addInteriorFace(0, 1);  // Horizontal
    fc->addInteriorFace(2, 3);  // Horizontal
    fc->addInteriorFace(0, 2);  // Vertical
    fc->addInteriorFace(1, 3);  // Vertical

    // Boundary faces
    fc->addBoundaryFace(0, 0);
    fc->addBoundaryFace(1, 0);
    fc->addBoundaryFace(2, 0);
    fc->addBoundaryFace(3, 0);

    DGSparsityBuilder builder(dof_map, fc);
    SparsityPattern pattern = builder.build();

    EXPECT_EQ(pattern.numRows(), 16);
    EXPECT_TRUE(pattern.isFinalized());

    auto stats = builder.getLastStats();
    EXPECT_EQ(stats.n_cells, 4);
    EXPECT_EQ(stats.n_interior_faces, 4);
    EXPECT_EQ(stats.n_boundary_faces, 4);
}

TEST(DGSparsityBuilderTest, BuildByTermType) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(2, 3);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addInteriorFace(0, 1);

    DGSparsityBuilder builder(dof_map, fc);

    // Build only volume integrals
    SparsityPattern volume = builder.build(DGTermType::VolumeIntegral);
    EXPECT_EQ(volume.getNnz(), 18);  // 2 cells * 3x3

    // Build only face integrals
    SparsityPattern face = builder.build(DGTermType::InteriorFace);
    EXPECT_GT(face.getNnz(), 0);

    // Build all
    SparsityPattern all = builder.build(DGTermType::All);
    EXPECT_GE(all.getNnz(), volume.getNnz());
}

// ============================================================================
// Options Tests
// ============================================================================

TEST(DGSparsityBuilderTest, EnsureDiagonal) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(2, 3);
    auto fc = std::make_shared<SimpleFaceConnectivity>();

    DGSparsityBuilder builder(dof_map, fc);

    DGSparsityOptions opts;
    opts.ensure_diagonal = true;
    opts.include_cell_couplings = false;
    opts.include_face_couplings = false;
    opts.include_boundary_couplings = false;
    builder.setOptions(opts);

    SparsityPattern pattern = builder.build();

    // All diagonals should be present
    for (GlobalIndex i = 0; i < 6; ++i) {
        EXPECT_TRUE(pattern.hasEntry(i, i));
    }
}

TEST(DGSparsityBuilderTest, SymmetricPattern) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(2, 2);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addInteriorFace(0, 1);

    DGSparsityBuilder builder(dof_map, fc);

    DGSparsityOptions opts;
    opts.symmetric_pattern = true;
    builder.setOptions(opts);

    SparsityPattern pattern = builder.build();
    EXPECT_TRUE(pattern.isSymmetric());
}

// ============================================================================
// Incremental Build Tests
// ============================================================================

TEST(DGSparsityBuilderTest, IncrementalBuild) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(3, 2);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addInteriorFace(0, 1);
    fc->addInteriorFace(1, 2);

    DGSparsityBuilder builder(dof_map, fc);

    SparsityPattern pattern(6, 6);

    // Build incrementally
    builder.buildCellCouplings(pattern);
    GlobalIndex after_cells = pattern.getNnz();
    EXPECT_GT(after_cells, 0);

    builder.buildFaceCouplings(pattern);
    GlobalIndex after_faces = pattern.getNnz();
    EXPECT_GT(after_faces, after_cells);

    pattern.finalize();
    EXPECT_TRUE(pattern.validate());
}

TEST(DGSparsityBuilderTest, SelectedFaces) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(4, 2);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addInteriorFace(0, 1);  // Face 0
    fc->addInteriorFace(1, 2);  // Face 1
    fc->addInteriorFace(2, 3);  // Face 2

    DGSparsityBuilder builder(dof_map, fc);

    SparsityPattern pattern(8, 8);

    // Only process faces 0 and 2
    std::vector<GlobalIndex> selected_faces = {0, 2};
    builder.buildFaceCouplings(pattern, selected_faces);
    pattern.finalize();

    // 0-1 coupling (face 0)
    EXPECT_TRUE(pattern.hasEntry(0, 2));

    // 2-3 coupling (face 2)
    EXPECT_TRUE(pattern.hasEntry(4, 6));

    // 1-2 coupling (face 1) should NOT exist
    EXPECT_FALSE(pattern.hasEntry(2, 4));
}

// ============================================================================
// NNZ Estimation Tests
// ============================================================================

TEST(DGSparsityBuilderTest, EstimateNnz) {
    GlobalIndex n_cells = 100;
    GlobalIndex dofs_per_cell = 10;
    double avg_neighbors = 4.0;

    GlobalIndex estimate = DGSparsityBuilder::estimateNnz(
        n_cells, dofs_per_cell, avg_neighbors);

    // Should be reasonable estimate
    // Cell-local: 100 * 100 = 10000
    // Face: ~200 faces * 2 * 100 = 40000
    EXPECT_GT(estimate, 0);
    EXPECT_LT(estimate, n_cells * dofs_per_cell * dofs_per_cell * 10);
}

// ============================================================================
// DGDofMapAdapter Tests
// ============================================================================

TEST(DGDofMapAdapterTest, WrapsStandardDofMap) {
    class MockDofMap : public IDofMapQuery {
    public:
        GlobalIndex getNumDofs() const override { return 10; }
        GlobalIndex getNumLocalDofs() const override { return 10; }
        std::span<const GlobalIndex> getCellDofs(GlobalIndex) const override {
            return dofs_;
        }
        GlobalIndex getNumCells() const override { return 2; }
        bool isOwnedDof(GlobalIndex) const override { return true; }
        std::pair<GlobalIndex, GlobalIndex> getOwnedRange() const override {
            return {0, 10};
        }
        std::vector<GlobalIndex> dofs_ = {0, 1, 2, 3, 4};
    };

    auto mock = std::make_shared<MockDofMap>();
    DGDofMapAdapter adapter(mock);

    EXPECT_EQ(adapter.getNumDofs(), 10);
    EXPECT_EQ(adapter.getNumCells(), 2);
    EXPECT_TRUE(adapter.isDG());

    auto cell_dofs = adapter.getCellDofs(0);
    EXPECT_EQ(cell_dofs.size(), 5);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(DGSparsityBuilderTest, BuildDGPatternConvenience) {
    auto dof_map = std::make_shared<SimpleDGDofMap>(2, 3);
    auto fc = std::make_shared<SimpleFaceConnectivity>();
    fc->addInteriorFace(0, 1);

    SparsityPattern pattern = buildDGPattern(dof_map, fc);

    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_GT(pattern.getNnz(), 0);
}

TEST(DGSparsityBuilderTest, CombineHybridPattern) {
    SparsityPattern cg(5, 5);
    cg.addEntry(0, 0);
    cg.addEntry(0, 1);
    cg.finalize();

    SparsityPattern dg(5, 5);
    dg.addEntry(2, 2);
    dg.addEntry(2, 3);
    dg.finalize();

    SparsityPattern hybrid = combineHybridPattern(cg, dg);

    EXPECT_TRUE(hybrid.hasEntry(0, 0));
    EXPECT_TRUE(hybrid.hasEntry(0, 1));
    EXPECT_TRUE(hybrid.hasEntry(2, 2));
    EXPECT_TRUE(hybrid.hasEntry(2, 3));
}
