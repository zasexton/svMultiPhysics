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

#ifndef SVMP_FE_SPARSITY_BLOCK_SPARSITY_H
#define SVMP_FE_SPARSITY_BLOCK_SPARSITY_H

/**
 * @file BlockSparsity.h
 * @brief Block-structured sparsity patterns for multi-field problems
 *
 * This header provides the BlockSparsity class for representing and
 * constructing block-structured sparsity patterns that arise in multi-field
 * finite element problems. For example, a Stokes problem might have:
 *
 * [ K_uu   K_up ] [ u ]   [ f_u ]
 * [ K_pu   K_pp ] [ p ] = [ f_p ]
 *
 * where each K_ij block has its own sparsity structure.
 *
 * Key features:
 * - Nested block structure (2D array of SparsityPattern)
 * - Support for variable block sizes
 * - Efficient block-wise construction
 * - Conversion to/from monolithic patterns
 * - Compatible with block preconditioners
 *
 * Design notes:
 * - Blocks may have different sizes (rectangular allowed)
 * - Empty blocks are allowed (zero coupling)
 * - Block indices are (field_row, field_col)
 * - DOF ordering within blocks follows field DOF maps
 *
 * Complexity:
 * - Block access: O(1)
 * - Monolithic conversion: O(total_nnz)
 * - Memory: O(n_blocks) + sum of block patterns
 *
 * @see SparsityPattern for individual block representation
 * @see BlockDofMap for multi-field DOF numbering
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <string>

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Block structure type
 */
enum class BlockStructure : std::uint8_t {
    General,        ///< General block structure
    Symmetric,      ///< Block-symmetric (A_ij = A_ji^T structurally)
    LowerTriangular,///< Lower triangular block structure
    UpperTriangular,///< Upper triangular block structure
    Diagonal,       ///< Block-diagonal (only A_ii blocks)
    Saddle          ///< Saddle-point structure (common in CFD)
};

/**
 * @brief Block information descriptor
 */
struct BlockInfo {
    FieldId field_id{INVALID_FIELD_ID};  ///< Field identifier
    GlobalIndex size{0};                  ///< Number of DOFs in this block
    GlobalIndex offset{0};                ///< Offset in monolithic numbering
    std::string name;                     ///< Optional block name
};

/**
 * @brief Statistics for block sparsity pattern
 */
struct BlockSparsityStats {
    GlobalIndex n_block_rows{0};      ///< Number of block rows
    GlobalIndex n_block_cols{0};      ///< Number of block columns
    GlobalIndex total_rows{0};        ///< Total scalar rows
    GlobalIndex total_cols{0};        ///< Total scalar columns
    GlobalIndex total_nnz{0};         ///< Total NNZ
    GlobalIndex n_nonzero_blocks{0};  ///< Number of non-empty blocks
    GlobalIndex n_zero_blocks{0};     ///< Number of empty blocks
    GlobalIndex max_block_nnz{0};     ///< Maximum NNZ in any block
    double avg_block_density{0.0};    ///< Average block fill ratio
};

/**
 * @brief Block-structured sparsity pattern
 *
 * BlockSparsity represents the sparsity structure of a block matrix where
 * each block corresponds to coupling between two fields. This is useful for:
 *
 * 1. Multi-field problems (velocity-pressure, displacement-pressure, etc.)
 * 2. Block preconditioners (Schur complement, block Jacobi)
 * 3. Segregated solvers (separate field solves)
 * 4. Physics-based partitioning
 *
 * Usage:
 * @code
 * // Create 2x2 block pattern for Stokes (velocity 3D, pressure scalar)
 * std::vector<GlobalIndex> block_sizes = {3 * n_vel_nodes, n_pres_nodes};
 * BlockSparsity block_pattern(block_sizes);
 *
 * // Build individual blocks
 * block_pattern.setBlock(0, 0, K_uu_pattern);  // velocity-velocity
 * block_pattern.setBlock(0, 1, K_up_pattern);  // velocity-pressure
 * block_pattern.setBlock(1, 0, K_pu_pattern);  // pressure-velocity
 * block_pattern.setBlock(1, 1, K_pp_pattern);  // pressure-pressure (may be empty)
 *
 * // Get monolithic pattern for global assembly
 * SparsityPattern monolithic = block_pattern.toMonolithic();
 *
 * // Get individual blocks for block preconditioner
 * const auto& Kuu = block_pattern.getBlock(0, 0);
 * @endcode
 */
class BlockSparsity {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor - empty block pattern
     */
    BlockSparsity() = default;

    /**
     * @brief Construct with block sizes (square structure)
     *
     * @param block_sizes Sizes of each block (both rows and columns)
     */
    explicit BlockSparsity(std::vector<GlobalIndex> block_sizes);

    /**
     * @brief Construct with separate row and column block sizes
     *
     * @param row_block_sizes Row block sizes
     * @param col_block_sizes Column block sizes
     */
    BlockSparsity(std::vector<GlobalIndex> row_block_sizes,
                  std::vector<GlobalIndex> col_block_sizes);

    /**
     * @brief Construct with BlockInfo descriptors
     *
     * @param row_blocks Row block information
     * @param col_blocks Column block information
     */
    BlockSparsity(std::vector<BlockInfo> row_blocks,
                  std::vector<BlockInfo> col_blocks);

    /// Copy constructor
    BlockSparsity(const BlockSparsity& other);

    /// Copy assignment
    BlockSparsity& operator=(const BlockSparsity& other);

    /// Move constructor
    BlockSparsity(BlockSparsity&&) noexcept = default;

    /// Move assignment
    BlockSparsity& operator=(BlockSparsity&&) noexcept = default;

    /// Destructor
    ~BlockSparsity() = default;

    // =========================================================================
    // Block structure queries
    // =========================================================================

    /**
     * @brief Get number of block rows
     */
    [[nodiscard]] GlobalIndex numBlockRows() const noexcept { return n_block_rows_; }

    /**
     * @brief Get number of block columns
     */
    [[nodiscard]] GlobalIndex numBlockCols() const noexcept { return n_block_cols_; }

    /**
     * @brief Get total scalar rows
     */
    [[nodiscard]] GlobalIndex totalRows() const noexcept { return total_rows_; }

    /**
     * @brief Get total scalar columns
     */
    [[nodiscard]] GlobalIndex totalCols() const noexcept { return total_cols_; }

    /**
     * @brief Check if block structure is square
     */
    [[nodiscard]] bool isSquare() const noexcept {
        return n_block_rows_ == n_block_cols_ && total_rows_ == total_cols_;
    }

    /**
     * @brief Get row block size
     */
    [[nodiscard]] GlobalIndex getRowBlockSize(GlobalIndex block_row) const;

    /**
     * @brief Get column block size
     */
    [[nodiscard]] GlobalIndex getColBlockSize(GlobalIndex block_col) const;

    /**
     * @brief Get row block offset in monolithic numbering
     */
    [[nodiscard]] GlobalIndex getRowBlockOffset(GlobalIndex block_row) const;

    /**
     * @brief Get column block offset in monolithic numbering
     */
    [[nodiscard]] GlobalIndex getColBlockOffset(GlobalIndex block_col) const;

    /**
     * @brief Get row block info
     */
    [[nodiscard]] const BlockInfo& getRowBlockInfo(GlobalIndex block_row) const;

    /**
     * @brief Get column block info
     */
    [[nodiscard]] const BlockInfo& getColBlockInfo(GlobalIndex block_col) const;

    /**
     * @brief Get all row block sizes
     */
    [[nodiscard]] std::span<const GlobalIndex> getRowBlockSizes() const noexcept {
        return std::span<const GlobalIndex>(row_block_sizes_);
    }

    /**
     * @brief Get all column block sizes
     */
    [[nodiscard]] std::span<const GlobalIndex> getColBlockSizes() const noexcept {
        return std::span<const GlobalIndex>(col_block_sizes_);
    }

    // =========================================================================
    // Block access
    // =========================================================================

    /**
     * @brief Get block at (i, j)
     *
     * @param block_row Block row index
     * @param block_col Block column index
     * @return Reference to block pattern (may be empty)
     */
    [[nodiscard]] const SparsityPattern& getBlock(GlobalIndex block_row,
                                                   GlobalIndex block_col) const;

    /**
     * @brief Get mutable block at (i, j)
     *
     * @param block_row Block row index
     * @param block_col Block column index
     * @return Mutable reference to block pattern
     */
    [[nodiscard]] SparsityPattern& getBlock(GlobalIndex block_row,
                                            GlobalIndex block_col);

    /**
     * @brief Set block at (i, j)
     *
     * @param block_row Block row index
     * @param block_col Block column index
     * @param pattern Pattern to set (copied)
     */
    void setBlock(GlobalIndex block_row, GlobalIndex block_col,
                  const SparsityPattern& pattern);

    /**
     * @brief Set block at (i, j) (move)
     *
     * @param block_row Block row index
     * @param block_col Block column index
     * @param pattern Pattern to set (moved)
     */
    void setBlock(GlobalIndex block_row, GlobalIndex block_col,
                  SparsityPattern&& pattern);

    /**
     * @brief Clear a block (set to empty)
     *
     * @param block_row Block row index
     * @param block_col Block column index
     */
    void clearBlock(GlobalIndex block_row, GlobalIndex block_col);

    /**
     * @brief Check if a block is non-empty
     *
     * @param block_row Block row index
     * @param block_col Block column index
     * @return true if block has any entries
     */
    [[nodiscard]] bool hasBlock(GlobalIndex block_row, GlobalIndex block_col) const;

    /**
     * @brief Check if block is finalized
     */
    [[nodiscard]] bool isBlockFinalized(GlobalIndex block_row, GlobalIndex block_col) const;

    // =========================================================================
    // Block construction helpers
    // =========================================================================

    /**
     * @brief Create empty block at (i, j) with correct dimensions
     *
     * @param block_row Block row index
     * @param block_col Block column index
     * @return Reference to newly created block
     */
    SparsityPattern& createBlock(GlobalIndex block_row, GlobalIndex block_col);

    /**
     * @brief Add entry at scalar indices, automatically finding block
     *
     * @param row Scalar row index
     * @param col Scalar column index
     */
    void addEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Add element coupling at scalar indices
     *
     * @param dofs Scalar DOF indices
     */
    void addElementCouplings(std::span<const GlobalIndex> dofs);

    /**
     * @brief Add coupling between two fields
     *
     * @param row_dofs DOFs from row field
     * @param col_dofs DOFs from column field
     * @param row_block Row block index
     * @param col_block Column block index
     */
    void addFieldCoupling(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          GlobalIndex row_block, GlobalIndex col_block);

    // =========================================================================
    // Finalization
    // =========================================================================

    /**
     * @brief Finalize all blocks
     */
    void finalize();

    /**
     * @brief Finalize specific block
     */
    void finalizeBlock(GlobalIndex block_row, GlobalIndex block_col);

    /**
     * @brief Check if all blocks are finalized
     */
    [[nodiscard]] bool isFinalized() const noexcept { return is_finalized_; }

    /**
     * @brief Clone preserving finalized state
     *
     * BlockSparsity copy construction resets to Building state; this helper
     * returns a deep copy that remains finalized, enabling safe caching.
     */
    [[nodiscard]] BlockSparsity cloneFinalized() const;

    /**
     * @brief Ensure diagonal blocks have diagonal entries
     */
    void ensureDiagonals();

    // =========================================================================
    // Conversion
    // =========================================================================

    /**
     * @brief Convert to monolithic sparsity pattern
     *
     * Creates a single SparsityPattern containing all blocks.
     *
     * @return Monolithic sparsity pattern
     */
    [[nodiscard]] SparsityPattern toMonolithic() const;

    /**
     * @brief Create BlockSparsity from monolithic pattern
     *
     * Extracts blocks from a monolithic pattern given block structure.
     *
     * @param pattern Monolithic pattern
     * @param row_block_sizes Row block sizes
     * @param col_block_sizes Column block sizes
     * @return Block sparsity pattern
     */
    [[nodiscard]] static BlockSparsity fromMonolithic(
        const SparsityPattern& pattern,
        std::vector<GlobalIndex> row_block_sizes,
        std::vector<GlobalIndex> col_block_sizes);

    /**
     * @brief Extract diagonal blocks only
     *
     * @return Vector of diagonal block patterns
     */
    [[nodiscard]] std::vector<SparsityPattern> extractDiagonalBlocks() const;

    /**
     * @brief Extract Schur complement sparsity (for saddle-point systems)
     *
     * For system [A B; C D], computes sparsity of S = D - C A^{-1} B.
     * Uses structural analysis (assumes A^{-1} is dense).
     *
     * @return Schur complement sparsity pattern
     */
    [[nodiscard]] SparsityPattern extractSchurComplement() const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Compute block sparsity statistics
     */
    [[nodiscard]] BlockSparsityStats computeStats() const;

    /**
     * @brief Get total NNZ across all blocks
     */
    [[nodiscard]] GlobalIndex getTotalNnz() const;

    /**
     * @brief Get NNZ for specific block
     */
    [[nodiscard]] GlobalIndex getBlockNnz(GlobalIndex block_row, GlobalIndex block_col) const;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate block dimensions and structure
     *
     * @return Empty string if valid, error message otherwise
     */
    [[nodiscard]] std::string validate() const;

    /**
     * @brief Memory usage in bytes
     */
    [[nodiscard]] std::size_t memoryUsageBytes() const noexcept;

private:
    // Helpers
    void computeOffsets();
    void validateBlockIndices(GlobalIndex block_row, GlobalIndex block_col) const;
    std::pair<GlobalIndex, GlobalIndex> scalarToBlock(GlobalIndex row, GlobalIndex col) const;
    std::size_t blockIndex(GlobalIndex block_row, GlobalIndex block_col) const;

    // Structure
    GlobalIndex n_block_rows_{0};
    GlobalIndex n_block_cols_{0};
    GlobalIndex total_rows_{0};
    GlobalIndex total_cols_{0};

    // Block sizes and offsets
    std::vector<GlobalIndex> row_block_sizes_;
    std::vector<GlobalIndex> col_block_sizes_;
    std::vector<GlobalIndex> row_block_offsets_;
    std::vector<GlobalIndex> col_block_offsets_;

    // Block info
    std::vector<BlockInfo> row_block_info_;
    std::vector<BlockInfo> col_block_info_;

    // Blocks stored in row-major order
    std::vector<SparsityPattern> blocks_;

    // State
    bool is_finalized_{false};
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Create 2x2 saddle-point block structure
 *
 * Common structure for mixed FEM (Stokes, Biot, etc.)
 *
 * @param n_primal Number of primal DOFs (e.g., velocity)
 * @param n_dual Number of dual DOFs (e.g., pressure)
 * @return 2x2 BlockSparsity
 */
[[nodiscard]] inline BlockSparsity createSaddlePointStructure(
    GlobalIndex n_primal, GlobalIndex n_dual)
{
    return BlockSparsity({n_primal, n_dual});
}

/**
 * @brief Create diagonal block structure
 *
 * @param block_sizes Sizes of diagonal blocks
 * @return Block-diagonal BlockSparsity
 */
[[nodiscard]] BlockSparsity createDiagonalBlockStructure(
    const std::vector<GlobalIndex>& block_sizes);

/**
 * @brief Combine patterns into block-diagonal structure
 *
 * @param patterns Individual patterns to place on diagonal
 * @return Block-diagonal BlockSparsity
 */
[[nodiscard]] BlockSparsity blockDiagonal(
    const std::vector<SparsityPattern>& patterns);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_BLOCK_SPARSITY_H
