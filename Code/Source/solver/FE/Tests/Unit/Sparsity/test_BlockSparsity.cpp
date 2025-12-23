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
#include "Sparsity/BlockSparsity.h"
#include <vector>
#include <algorithm>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(BlockSparsityTest, DefaultConstruction) {
    BlockSparsity bs;
    EXPECT_EQ(bs.numBlockRows(), 0);
    EXPECT_EQ(bs.numBlockCols(), 0);
    EXPECT_EQ(bs.totalRows(), 0);
    EXPECT_EQ(bs.totalCols(), 0);
}

TEST(BlockSparsityTest, ConstructWithBlockSizesSquare) {
    std::vector<GlobalIndex> sizes = {10, 5, 3};  // 3x3 block structure
    BlockSparsity bs(sizes);

    EXPECT_EQ(bs.numBlockRows(), 3);
    EXPECT_EQ(bs.numBlockCols(), 3);
    EXPECT_EQ(bs.totalRows(), 18);  // 10 + 5 + 3
    EXPECT_EQ(bs.totalCols(), 18);
    EXPECT_TRUE(bs.isSquare());
}

TEST(BlockSparsityTest, ConstructWithSeparateRowColSizes) {
    std::vector<GlobalIndex> row_sizes = {10, 5};
    std::vector<GlobalIndex> col_sizes = {8, 4, 2};

    BlockSparsity bs(row_sizes, col_sizes);

    EXPECT_EQ(bs.numBlockRows(), 2);
    EXPECT_EQ(bs.numBlockCols(), 3);
    EXPECT_EQ(bs.totalRows(), 15);  // 10 + 5
    EXPECT_EQ(bs.totalCols(), 14);  // 8 + 4 + 2
    EXPECT_FALSE(bs.isSquare());
}

TEST(BlockSparsityTest, ConstructWithBlockInfo) {
    std::vector<BlockInfo> row_blocks = {
        {0, 10, 0, "velocity"},
        {1, 5, 10, "pressure"}
    };
    std::vector<BlockInfo> col_blocks = {
        {0, 10, 0, "velocity"},
        {1, 5, 10, "pressure"}
    };

    BlockSparsity bs(row_blocks, col_blocks);

    EXPECT_EQ(bs.numBlockRows(), 2);
    EXPECT_EQ(bs.numBlockCols(), 2);
    EXPECT_EQ(bs.getRowBlockInfo(0).name, "velocity");
    EXPECT_EQ(bs.getRowBlockInfo(1).name, "pressure");
}

TEST(BlockSparsityTest, CopyConstruction) {
    BlockSparsity original({10, 5});
    original.createBlock(0, 0);
    original.getBlock(0, 0).addEntry(0, 0);
    original.getBlock(0, 0).addEntry(0, 1);

    BlockSparsity copy(original);

    EXPECT_EQ(copy.numBlockRows(), original.numBlockRows());
    EXPECT_EQ(copy.totalRows(), original.totalRows());
    EXPECT_TRUE(copy.getBlock(0, 0).hasEntry(0, 0));
    EXPECT_TRUE(copy.getBlock(0, 0).hasEntry(0, 1));
}

// ============================================================================
// Block Size and Offset Tests
// ============================================================================

TEST(BlockSparsityTest, GetRowBlockSize) {
    BlockSparsity bs({10, 5, 3});

    EXPECT_EQ(bs.getRowBlockSize(0), 10);
    EXPECT_EQ(bs.getRowBlockSize(1), 5);
    EXPECT_EQ(bs.getRowBlockSize(2), 3);
}

TEST(BlockSparsityTest, GetColBlockSize) {
    BlockSparsity bs({10, 5}, {8, 4, 2});

    EXPECT_EQ(bs.getColBlockSize(0), 8);
    EXPECT_EQ(bs.getColBlockSize(1), 4);
    EXPECT_EQ(bs.getColBlockSize(2), 2);
}

TEST(BlockSparsityTest, GetRowBlockOffset) {
    BlockSparsity bs({10, 5, 3});

    EXPECT_EQ(bs.getRowBlockOffset(0), 0);
    EXPECT_EQ(bs.getRowBlockOffset(1), 10);
    EXPECT_EQ(bs.getRowBlockOffset(2), 15);
}

TEST(BlockSparsityTest, GetColBlockOffset) {
    BlockSparsity bs({10, 5}, {8, 4, 2});

    EXPECT_EQ(bs.getColBlockOffset(0), 0);
    EXPECT_EQ(bs.getColBlockOffset(1), 8);
    EXPECT_EQ(bs.getColBlockOffset(2), 12);
}

TEST(BlockSparsityTest, GetBlockSizesSpan) {
    BlockSparsity bs({10, 5, 3});

    auto row_sizes = bs.getRowBlockSizes();
    ASSERT_EQ(row_sizes.size(), 3);
    EXPECT_EQ(row_sizes[0], 10);
    EXPECT_EQ(row_sizes[1], 5);
    EXPECT_EQ(row_sizes[2], 3);
}

// ============================================================================
// Block Access Tests
// ============================================================================

TEST(BlockSparsityTest, CreateBlock) {
    BlockSparsity bs({10, 5});

    auto& block = bs.createBlock(0, 0);
    EXPECT_EQ(block.numRows(), 10);
    EXPECT_EQ(block.numCols(), 10);

    // Note: hasBlock() returns true only if the block has entries (NNZ > 0)
    // A newly created empty block will return false from hasBlock()
    // Add an entry to verify hasBlock works correctly
    block.addEntry(0, 0);
    EXPECT_TRUE(bs.hasBlock(0, 0));
}

TEST(BlockSparsityTest, SetAndGetBlock) {
    BlockSparsity bs({10, 5});

    SparsityPattern pattern(10, 10);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 1);
    pattern.finalize();

    bs.setBlock(0, 0, pattern);

    EXPECT_TRUE(bs.hasBlock(0, 0));
    EXPECT_EQ(bs.getBlock(0, 0).getNnz(), 2);
}

TEST(BlockSparsityTest, SetBlockMove) {
    BlockSparsity bs({10, 5});

    SparsityPattern pattern(10, 10);
    pattern.addEntry(0, 0);
    pattern.finalize();

    bs.setBlock(0, 0, std::move(pattern));

    EXPECT_TRUE(bs.hasBlock(0, 0));
    EXPECT_EQ(bs.getBlock(0, 0).getNnz(), 1);
}

TEST(BlockSparsityTest, ClearBlock) {
    BlockSparsity bs({10, 5});
    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);

    EXPECT_TRUE(bs.hasBlock(0, 0));

    bs.clearBlock(0, 0);
    // After clear, block should be empty (getNnz == 0)
    EXPECT_EQ(bs.getBlock(0, 0).getNnz(), 0);
}

TEST(BlockSparsityTest, GetMutableBlock) {
    BlockSparsity bs({10, 5});
    bs.createBlock(0, 0);

    SparsityPattern& block = bs.getBlock(0, 0);
    block.addEntry(0, 1);
    block.addEntry(1, 0);

    EXPECT_EQ(bs.getBlock(0, 0).getNnz(), 2);
}

TEST(BlockSparsityTest, IsBlockFinalized) {
    BlockSparsity bs({10, 5});
    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);

    EXPECT_FALSE(bs.isBlockFinalized(0, 0));

    bs.finalizeBlock(0, 0);
    EXPECT_TRUE(bs.isBlockFinalized(0, 0));
}

// ============================================================================
// Block Construction Helper Tests
// ============================================================================

TEST(BlockSparsityTest, AddEntryScalar) {
    // 2x2 block structure: block 0 = [0..9], block 1 = [10..14]
    BlockSparsity bs({10, 5});

    // Entry in block (0, 0): row 3, col 7
    bs.addEntry(3, 7);
    // Entry in block (0, 1): row 3, col 12 (global) -> col 2 (in block 1)
    bs.addEntry(3, 12);
    // Entry in block (1, 0): row 11 (global) -> row 1 (in block 1), col 5
    bs.addEntry(11, 5);
    // Entry in block (1, 1): row 12, col 13
    bs.addEntry(12, 13);

    EXPECT_TRUE(bs.hasBlock(0, 0));
    EXPECT_TRUE(bs.hasBlock(0, 1));
    EXPECT_TRUE(bs.hasBlock(1, 0));
    EXPECT_TRUE(bs.hasBlock(1, 1));
}

TEST(BlockSparsityTest, AddElementCouplingsScalar) {
    // 2x2 block structure: block 0 = [0..9], block 1 = [10..14]
    BlockSparsity bs({10, 5});

    // DOFs spanning both blocks
    std::vector<GlobalIndex> dofs = {0, 5, 10, 12};
    bs.addElementCouplings(dofs);

    // All 4 blocks should now have entries
    EXPECT_TRUE(bs.hasBlock(0, 0));  // (0,0), (0,5), (5,0), (5,5)
    EXPECT_TRUE(bs.hasBlock(0, 1));  // (0,10-10), (5,10-10), etc.
    EXPECT_TRUE(bs.hasBlock(1, 0));  // (10-10,0), (12-10,5), etc.
    EXPECT_TRUE(bs.hasBlock(1, 1));  // (10-10,10-10), etc.
}

TEST(BlockSparsityTest, AddFieldCoupling) {
    BlockSparsity bs({10, 5});

    std::vector<GlobalIndex> row_dofs = {0, 1, 2};  // In block 0
    std::vector<GlobalIndex> col_dofs = {0, 1, 2};  // Also in block 0

    bs.addFieldCoupling(row_dofs, col_dofs, 0, 0);

    EXPECT_TRUE(bs.hasBlock(0, 0));
    // Should have 3x3 = 9 entries (with duplicates merged)
    EXPECT_GE(bs.getBlock(0, 0).getNnz(), 9);
}

// ============================================================================
// Finalization Tests
// ============================================================================

TEST(BlockSparsityTest, FinalizeAll) {
    BlockSparsity bs({10, 5});
    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);
    bs.createBlock(0, 1);
    bs.getBlock(0, 1).addEntry(0, 0);

    EXPECT_FALSE(bs.isFinalized());

    bs.finalize();

    EXPECT_TRUE(bs.isFinalized());
    EXPECT_TRUE(bs.isBlockFinalized(0, 0));
    EXPECT_TRUE(bs.isBlockFinalized(0, 1));
}

TEST(BlockSparsityTest, EnsureDiagonals) {
    BlockSparsity bs({10, 5});
    bs.createBlock(0, 0);  // Empty block
    bs.createBlock(1, 1);  // Empty block

    bs.ensureDiagonals();
    bs.finalize();

    // Diagonal blocks should have diagonal entries
    const auto& block00 = bs.getBlock(0, 0);
    const auto& block11 = bs.getBlock(1, 1);

    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_TRUE(block00.hasEntry(i, i));
    }
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(block11.hasEntry(i, i));
    }
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST(BlockSparsityTest, ToMonolithic) {
    BlockSparsity bs({10, 5});

    // Create diagonal entries in both blocks
    bs.createBlock(0, 0);
    for (GlobalIndex i = 0; i < 10; ++i) {
        bs.getBlock(0, 0).addEntry(i, i);
    }
    bs.createBlock(1, 1);
    for (GlobalIndex i = 0; i < 5; ++i) {
        bs.getBlock(1, 1).addEntry(i, i);
    }
    // Add off-diagonal coupling
    bs.createBlock(0, 1);
    bs.getBlock(0, 1).addEntry(0, 0);

    SparsityPattern mono = bs.toMonolithic();

    EXPECT_EQ(mono.numRows(), 15);
    EXPECT_EQ(mono.numCols(), 15);

    // Check diagonal entries
    for (GlobalIndex i = 0; i < 15; ++i) {
        EXPECT_TRUE(mono.hasEntry(i, i));
    }
    // Check off-diagonal: row 0 (block 0), col 0 (block 1) -> col 10 in monolithic
    EXPECT_TRUE(mono.hasEntry(0, 10));
}

TEST(BlockSparsityTest, FromMonolithic) {
    // Create monolithic pattern
    SparsityPattern mono(15, 15);
    // Diagonal entries
    for (GlobalIndex i = 0; i < 15; ++i) {
        mono.addEntry(i, i);
    }
    // Off-diagonal: (0, 10)
    mono.addEntry(0, 10);
    mono.finalize();

    std::vector<GlobalIndex> row_sizes = {10, 5};
    std::vector<GlobalIndex> col_sizes = {10, 5};

    BlockSparsity bs = BlockSparsity::fromMonolithic(mono, row_sizes, col_sizes);

    EXPECT_EQ(bs.numBlockRows(), 2);
    EXPECT_EQ(bs.numBlockCols(), 2);

    // Block (0,0) should have 10 diagonal entries
    EXPECT_EQ(bs.getBlock(0, 0).getNnz(), 10);
    // Block (1,1) should have 5 diagonal entries
    EXPECT_EQ(bs.getBlock(1, 1).getNnz(), 5);
    // Block (0,1) should have 1 entry
    EXPECT_EQ(bs.getBlock(0, 1).getNnz(), 1);
}

TEST(BlockSparsityTest, ExtractDiagonalBlocks) {
    BlockSparsity bs({10, 5, 3});

    // Create diagonal blocks
    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);
    bs.createBlock(1, 1);
    bs.getBlock(1, 1).addEntry(0, 1);
    bs.createBlock(2, 2);
    bs.getBlock(2, 2).addEntry(1, 2);

    // Create off-diagonal block (should not be in result)
    bs.createBlock(0, 1);
    bs.getBlock(0, 1).addEntry(0, 0);

    bs.finalize();

    auto diag_blocks = bs.extractDiagonalBlocks();

    ASSERT_EQ(diag_blocks.size(), 3);
    EXPECT_EQ(diag_blocks[0].numRows(), 10);
    EXPECT_EQ(diag_blocks[1].numRows(), 5);
    EXPECT_EQ(diag_blocks[2].numRows(), 3);
}

TEST(BlockSparsityTest, ExtractSchurComplement) {
    // 2x2 saddle-point structure
    // [A  B]
    // [C  D]
    // Schur complement = D - C * A^{-1} * B
    BlockSparsity bs({10, 5});

    // Block A (10x10) - full coupling
    bs.createBlock(0, 0);
    for (GlobalIndex i = 0; i < 10; ++i) {
        for (GlobalIndex j = 0; j < 10; ++j) {
            bs.getBlock(0, 0).addEntry(i, j);
        }
    }

    // Block B (10x5)
    bs.createBlock(0, 1);
    for (GlobalIndex i = 0; i < 10; ++i) {
        bs.getBlock(0, 1).addEntry(i, 0);
    }

    // Block C (5x10)
    bs.createBlock(1, 0);
    for (GlobalIndex j = 0; j < 10; ++j) {
        bs.getBlock(1, 0).addEntry(0, j);
    }

    // Block D (5x5) - diagonal
    bs.createBlock(1, 1);
    for (GlobalIndex i = 0; i < 5; ++i) {
        bs.getBlock(1, 1).addEntry(i, i);
    }

    bs.finalize();

    SparsityPattern schur = bs.extractSchurComplement();

    // Schur complement is 5x5
    EXPECT_EQ(schur.numRows(), 5);
    EXPECT_EQ(schur.numCols(), 5);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(BlockSparsityTest, ComputeStats) {
    BlockSparsity bs({10, 5});

    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);
    bs.getBlock(0, 0).addEntry(1, 1);

    bs.createBlock(0, 1);
    bs.getBlock(0, 1).addEntry(0, 0);

    bs.finalize();

    auto stats = bs.computeStats();

    EXPECT_EQ(stats.n_block_rows, 2);
    EXPECT_EQ(stats.n_block_cols, 2);
    EXPECT_EQ(stats.total_rows, 15);
    EXPECT_EQ(stats.total_cols, 15);
    EXPECT_EQ(stats.total_nnz, 3);
    EXPECT_EQ(stats.n_nonzero_blocks, 2);  // (0,0) and (0,1)
}

TEST(BlockSparsityTest, GetTotalNnz) {
    BlockSparsity bs({10, 5});

    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);
    bs.getBlock(0, 0).addEntry(1, 1);

    bs.createBlock(1, 1);
    bs.getBlock(1, 1).addEntry(0, 0);

    bs.finalize();

    EXPECT_EQ(bs.getTotalNnz(), 3);
}

TEST(BlockSparsityTest, GetBlockNnz) {
    BlockSparsity bs({10, 5});

    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);
    bs.getBlock(0, 0).addEntry(1, 1);
    bs.getBlock(0, 0).addEntry(2, 2);

    bs.finalize();

    EXPECT_EQ(bs.getBlockNnz(0, 0), 3);
    EXPECT_EQ(bs.getBlockNnz(0, 1), 0);  // Empty block
}

TEST(BlockSparsityTest, MemoryUsage) {
    BlockSparsity bs({100, 50});

    bs.createBlock(0, 0);
    for (GlobalIndex i = 0; i < 100; ++i) {
        bs.getBlock(0, 0).addEntry(i, i);
    }

    bs.finalize();

    std::size_t mem = bs.memoryUsageBytes();
    EXPECT_GT(mem, 0);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(BlockSparsityTest, ValidateCorrect) {
    BlockSparsity bs({10, 5});

    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);
    bs.finalize();

    std::string error = bs.validate();
    EXPECT_TRUE(error.empty()) << "Validation error: " << error;
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(BlockSparsityTest, CreateSaddlePointStructure) {
    auto bs = createSaddlePointStructure(100, 20);

    EXPECT_EQ(bs.numBlockRows(), 2);
    EXPECT_EQ(bs.numBlockCols(), 2);
    EXPECT_EQ(bs.totalRows(), 120);
    EXPECT_EQ(bs.totalCols(), 120);
    EXPECT_EQ(bs.getRowBlockSize(0), 100);
    EXPECT_EQ(bs.getRowBlockSize(1), 20);
}

TEST(BlockSparsityTest, CreateDiagonalBlockStructure) {
    std::vector<GlobalIndex> sizes = {10, 20, 30};
    auto bs = createDiagonalBlockStructure(sizes);

    EXPECT_EQ(bs.numBlockRows(), 3);
    EXPECT_EQ(bs.totalRows(), 60);
}

TEST(BlockSparsityTest, BlockDiagonalFromPatterns) {
    SparsityPattern p1(10, 10);
    p1.addEntry(0, 0);
    p1.finalize();

    SparsityPattern p2(5, 5);
    p2.addEntry(0, 0);
    p2.finalize();

    std::vector<SparsityPattern> patterns = {p1, p2};
    auto bs = blockDiagonal(patterns);

    EXPECT_EQ(bs.numBlockRows(), 2);
    EXPECT_EQ(bs.totalRows(), 15);
    EXPECT_EQ(bs.getTotalNnz(), 2);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(BlockSparsityTest, SingleBlock) {
    BlockSparsity bs({10});

    EXPECT_EQ(bs.numBlockRows(), 1);
    EXPECT_EQ(bs.numBlockCols(), 1);

    bs.createBlock(0, 0);
    bs.getBlock(0, 0).addEntry(0, 0);
    bs.finalize();

    SparsityPattern mono = bs.toMonolithic();
    EXPECT_EQ(mono.numRows(), 10);
    EXPECT_EQ(mono.getNnz(), 1);
}

TEST(BlockSparsityTest, EmptyBlocks) {
    BlockSparsity bs({10, 5});

    // Don't add any entries
    bs.finalize();

    SparsityPattern mono = bs.toMonolithic();
    EXPECT_EQ(mono.numRows(), 15);
    EXPECT_EQ(mono.getNnz(), 0);
}

TEST(BlockSparsityTest, LargeBlockStructure) {
    // 10x10 block structure
    std::vector<GlobalIndex> sizes(10, 100);  // 10 blocks of size 100
    BlockSparsity bs(sizes);

    EXPECT_EQ(bs.numBlockRows(), 10);
    EXPECT_EQ(bs.totalRows(), 1000);

    // Add diagonal blocks
    for (GlobalIndex i = 0; i < 10; ++i) {
        bs.createBlock(i, i);
        bs.getBlock(i, i).addEntry(0, 0);
    }

    bs.finalize();
    EXPECT_EQ(bs.getTotalNnz(), 10);
}

TEST(BlockSparsityTest, RectangularBlockStructure) {
    // 2 row blocks x 3 col blocks
    BlockSparsity bs({10, 5}, {8, 4, 2});

    EXPECT_EQ(bs.numBlockRows(), 2);
    EXPECT_EQ(bs.numBlockCols(), 3);
    EXPECT_FALSE(bs.isSquare());

    bs.createBlock(0, 2);  // (10x2) block
    EXPECT_EQ(bs.getBlock(0, 2).numRows(), 10);
    EXPECT_EQ(bs.getBlock(0, 2).numCols(), 2);
}

// ============================================================================
// Stokes-like Pattern Tests
// ============================================================================

TEST(BlockSparsityTest, StokesPatternExample) {
    // Typical Stokes: velocity (3D) + pressure
    // n_vel_nodes = 10, n_pres_nodes = 4
    // Velocity DOFs = 3 * 10 = 30, Pressure DOFs = 4
    GlobalIndex n_vel_dofs = 30;
    GlobalIndex n_pres_dofs = 4;

    BlockSparsity bs({n_vel_dofs, n_pres_dofs});

    // K_uu block: velocity-velocity coupling (symmetric, sparse)
    bs.createBlock(0, 0);
    for (GlobalIndex i = 0; i < n_vel_dofs; ++i) {
        bs.getBlock(0, 0).addEntry(i, i);  // Diagonal
        if (i > 0) bs.getBlock(0, 0).addEntry(i, i-1);
        if (i < n_vel_dofs - 1) bs.getBlock(0, 0).addEntry(i, i+1);
    }

    // K_up block: velocity-pressure coupling
    bs.createBlock(0, 1);
    for (GlobalIndex i = 0; i < n_vel_dofs; i += 3) {
        bs.getBlock(0, 1).addEntry(i, i / 10);  // Simplified coupling
    }

    // K_pu block: pressure-velocity coupling (transpose of K_up structurally)
    bs.createBlock(1, 0);
    for (GlobalIndex j = 0; j < n_pres_dofs; ++j) {
        bs.getBlock(1, 0).addEntry(j, j * 3);
    }

    // K_pp block: usually zero or stabilization
    // (empty in unstabilized Stokes)

    bs.finalize();

    auto stats = bs.computeStats();
    EXPECT_EQ(stats.n_block_rows, 2);
    EXPECT_EQ(stats.n_block_cols, 2);
    EXPECT_EQ(stats.total_rows, 34);
    EXPECT_GT(stats.total_nnz, 0);
}

