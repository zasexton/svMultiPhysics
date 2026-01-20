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

#include "BlockSparsity.h"
#include <algorithm>
#include <numeric>
#include <sstream>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Construction
// ============================================================================

BlockSparsity::BlockSparsity(std::vector<GlobalIndex> block_sizes)
    : BlockSparsity(block_sizes, block_sizes)
{
}

BlockSparsity::BlockSparsity(std::vector<GlobalIndex> row_block_sizes,
                             std::vector<GlobalIndex> col_block_sizes)
    : n_block_rows_(static_cast<GlobalIndex>(row_block_sizes.size())),
      n_block_cols_(static_cast<GlobalIndex>(col_block_sizes.size())),
      row_block_sizes_(std::move(row_block_sizes)),
      col_block_sizes_(std::move(col_block_sizes))
{
    // Validate block sizes
    for (GlobalIndex size : row_block_sizes_) {
        FE_CHECK_ARG(size >= 0, "Block size must be non-negative");
    }
    for (GlobalIndex size : col_block_sizes_) {
        FE_CHECK_ARG(size >= 0, "Block size must be non-negative");
    }

    computeOffsets();

    // Initialize block patterns (empty but correctly sized)
    blocks_.resize(static_cast<std::size_t>(n_block_rows_ * n_block_cols_));
    for (GlobalIndex i = 0; i < n_block_rows_; ++i) {
        for (GlobalIndex j = 0; j < n_block_cols_; ++j) {
            blocks_[blockIndex(i, j)] = SparsityPattern(
                row_block_sizes_[static_cast<std::size_t>(i)],
                col_block_sizes_[static_cast<std::size_t>(j)]);
        }
    }

    // Create default block info
    row_block_info_.resize(static_cast<std::size_t>(n_block_rows_));
    col_block_info_.resize(static_cast<std::size_t>(n_block_cols_));
    for (GlobalIndex i = 0; i < n_block_rows_; ++i) {
        row_block_info_[static_cast<std::size_t>(i)].field_id = static_cast<FieldId>(i);
        row_block_info_[static_cast<std::size_t>(i)].size = row_block_sizes_[static_cast<std::size_t>(i)];
        row_block_info_[static_cast<std::size_t>(i)].offset = row_block_offsets_[static_cast<std::size_t>(i)];
    }
    for (GlobalIndex j = 0; j < n_block_cols_; ++j) {
        col_block_info_[static_cast<std::size_t>(j)].field_id = static_cast<FieldId>(j);
        col_block_info_[static_cast<std::size_t>(j)].size = col_block_sizes_[static_cast<std::size_t>(j)];
        col_block_info_[static_cast<std::size_t>(j)].offset = col_block_offsets_[static_cast<std::size_t>(j)];
    }
}

BlockSparsity::BlockSparsity(std::vector<BlockInfo> row_blocks,
                             std::vector<BlockInfo> col_blocks)
    : n_block_rows_(static_cast<GlobalIndex>(row_blocks.size())),
      n_block_cols_(static_cast<GlobalIndex>(col_blocks.size())),
      row_block_info_(std::move(row_blocks)),
      col_block_info_(std::move(col_blocks))
{
    // Extract sizes
    row_block_sizes_.resize(row_block_info_.size());
    col_block_sizes_.resize(col_block_info_.size());
    for (std::size_t i = 0; i < row_block_info_.size(); ++i) {
        row_block_sizes_[i] = row_block_info_[i].size;
    }
    for (std::size_t j = 0; j < col_block_info_.size(); ++j) {
        col_block_sizes_[j] = col_block_info_[j].size;
    }

    computeOffsets();

    // Update offsets in block info
    for (std::size_t i = 0; i < row_block_info_.size(); ++i) {
        row_block_info_[i].offset = row_block_offsets_[i];
    }
    for (std::size_t j = 0; j < col_block_info_.size(); ++j) {
        col_block_info_[j].offset = col_block_offsets_[j];
    }

    // Initialize blocks
    blocks_.resize(static_cast<std::size_t>(n_block_rows_ * n_block_cols_));
    for (GlobalIndex i = 0; i < n_block_rows_; ++i) {
        for (GlobalIndex j = 0; j < n_block_cols_; ++j) {
            blocks_[blockIndex(i, j)] = SparsityPattern(
                row_block_sizes_[static_cast<std::size_t>(i)],
                col_block_sizes_[static_cast<std::size_t>(j)]);
        }
    }
}

BlockSparsity::BlockSparsity(const BlockSparsity& other)
    : n_block_rows_(other.n_block_rows_),
      n_block_cols_(other.n_block_cols_),
      total_rows_(other.total_rows_),
      total_cols_(other.total_cols_),
      row_block_sizes_(other.row_block_sizes_),
      col_block_sizes_(other.col_block_sizes_),
      row_block_offsets_(other.row_block_offsets_),
      col_block_offsets_(other.col_block_offsets_),
      row_block_info_(other.row_block_info_),
      col_block_info_(other.col_block_info_),
      blocks_(other.blocks_),
      is_finalized_(false)  // Copy goes to non-finalized state
{
}

BlockSparsity& BlockSparsity::operator=(const BlockSparsity& other) {
    if (this != &other) {
        n_block_rows_ = other.n_block_rows_;
        n_block_cols_ = other.n_block_cols_;
        total_rows_ = other.total_rows_;
        total_cols_ = other.total_cols_;
        row_block_sizes_ = other.row_block_sizes_;
        col_block_sizes_ = other.col_block_sizes_;
        row_block_offsets_ = other.row_block_offsets_;
        col_block_offsets_ = other.col_block_offsets_;
        row_block_info_ = other.row_block_info_;
        col_block_info_ = other.col_block_info_;
        blocks_ = other.blocks_;
        is_finalized_ = false;
    }
    return *this;
}

BlockSparsity BlockSparsity::cloneFinalized() const {
    FE_THROW_IF(!is_finalized_, InvalidArgumentException,
                "Cannot clone non-finalized BlockSparsity");

    BlockSparsity copy;
    copy.n_block_rows_ = n_block_rows_;
    copy.n_block_cols_ = n_block_cols_;
    copy.total_rows_ = total_rows_;
    copy.total_cols_ = total_cols_;
    copy.row_block_sizes_ = row_block_sizes_;
    copy.col_block_sizes_ = col_block_sizes_;
    copy.row_block_offsets_ = row_block_offsets_;
    copy.col_block_offsets_ = col_block_offsets_;
    copy.row_block_info_ = row_block_info_;
    copy.col_block_info_ = col_block_info_;

    copy.blocks_.resize(blocks_.size());
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        copy.blocks_[i] = blocks_[i].cloneFinalized();
    }

    copy.is_finalized_ = true;
    return copy;
}

// ============================================================================
// Block structure queries
// ============================================================================

GlobalIndex BlockSparsity::getRowBlockSize(GlobalIndex block_row) const {
    validateBlockIndices(block_row, 0);
    return row_block_sizes_[static_cast<std::size_t>(block_row)];
}

GlobalIndex BlockSparsity::getColBlockSize(GlobalIndex block_col) const {
    validateBlockIndices(0, block_col);
    return col_block_sizes_[static_cast<std::size_t>(block_col)];
}

GlobalIndex BlockSparsity::getRowBlockOffset(GlobalIndex block_row) const {
    validateBlockIndices(block_row, 0);
    return row_block_offsets_[static_cast<std::size_t>(block_row)];
}

GlobalIndex BlockSparsity::getColBlockOffset(GlobalIndex block_col) const {
    validateBlockIndices(0, block_col);
    return col_block_offsets_[static_cast<std::size_t>(block_col)];
}

const BlockInfo& BlockSparsity::getRowBlockInfo(GlobalIndex block_row) const {
    validateBlockIndices(block_row, 0);
    return row_block_info_[static_cast<std::size_t>(block_row)];
}

const BlockInfo& BlockSparsity::getColBlockInfo(GlobalIndex block_col) const {
    validateBlockIndices(0, block_col);
    return col_block_info_[static_cast<std::size_t>(block_col)];
}

// ============================================================================
// Block access
// ============================================================================

const SparsityPattern& BlockSparsity::getBlock(GlobalIndex block_row,
                                                GlobalIndex block_col) const {
    validateBlockIndices(block_row, block_col);
    return blocks_[blockIndex(block_row, block_col)];
}

SparsityPattern& BlockSparsity::getBlock(GlobalIndex block_row,
                                          GlobalIndex block_col) {
    validateBlockIndices(block_row, block_col);
    is_finalized_ = false;  // Modification invalidates finalization
    return blocks_[blockIndex(block_row, block_col)];
}

void BlockSparsity::setBlock(GlobalIndex block_row, GlobalIndex block_col,
                              const SparsityPattern& pattern) {
    validateBlockIndices(block_row, block_col);

    GlobalIndex expected_rows = row_block_sizes_[static_cast<std::size_t>(block_row)];
    GlobalIndex expected_cols = col_block_sizes_[static_cast<std::size_t>(block_col)];
    FE_CHECK_ARG(pattern.numRows() == expected_rows && pattern.numCols() == expected_cols,
                 "Block dimensions mismatch");

    blocks_[blockIndex(block_row, block_col)] = pattern;
    is_finalized_ = false;
}

void BlockSparsity::setBlock(GlobalIndex block_row, GlobalIndex block_col,
                              SparsityPattern&& pattern) {
    validateBlockIndices(block_row, block_col);

    GlobalIndex expected_rows = row_block_sizes_[static_cast<std::size_t>(block_row)];
    GlobalIndex expected_cols = col_block_sizes_[static_cast<std::size_t>(block_col)];
    FE_CHECK_ARG(pattern.numRows() == expected_rows && pattern.numCols() == expected_cols,
                 "Block dimensions mismatch");

    blocks_[blockIndex(block_row, block_col)] = std::move(pattern);
    is_finalized_ = false;
}

void BlockSparsity::clearBlock(GlobalIndex block_row, GlobalIndex block_col) {
    validateBlockIndices(block_row, block_col);
    blocks_[blockIndex(block_row, block_col)].clear();
    is_finalized_ = false;
}

bool BlockSparsity::hasBlock(GlobalIndex block_row, GlobalIndex block_col) const {
    validateBlockIndices(block_row, block_col);
    return blocks_[blockIndex(block_row, block_col)].getNnz() > 0;
}

bool BlockSparsity::isBlockFinalized(GlobalIndex block_row, GlobalIndex block_col) const {
    validateBlockIndices(block_row, block_col);
    return blocks_[blockIndex(block_row, block_col)].isFinalized();
}

// ============================================================================
// Block construction helpers
// ============================================================================

SparsityPattern& BlockSparsity::createBlock(GlobalIndex block_row, GlobalIndex block_col) {
    validateBlockIndices(block_row, block_col);

    GlobalIndex n_rows = row_block_sizes_[static_cast<std::size_t>(block_row)];
    GlobalIndex n_cols = col_block_sizes_[static_cast<std::size_t>(block_col)];

    auto& block = blocks_[blockIndex(block_row, block_col)];
    block = SparsityPattern(n_rows, n_cols);
    is_finalized_ = false;
    return block;
}

void BlockSparsity::addEntry(GlobalIndex row, GlobalIndex col) {
    FE_CHECK_ARG(row >= 0 && row < total_rows_, "Row index out of range");
    FE_CHECK_ARG(col >= 0 && col < total_cols_, "Column index out of range");

    auto [block_row, block_col] = scalarToBlock(row, col);

    GlobalIndex local_row = row - row_block_offsets_[static_cast<std::size_t>(block_row)];
    GlobalIndex local_col = col - col_block_offsets_[static_cast<std::size_t>(block_col)];

    auto& block = blocks_[blockIndex(block_row, block_col)];
    FE_THROW_IF(block.isFinalized(), InvalidArgumentException,
                "Cannot add entry to finalized block");
    block.addEntry(local_row, local_col);
    is_finalized_ = false;
}

void BlockSparsity::addElementCouplings(std::span<const GlobalIndex> dofs) {
    for (GlobalIndex row_dof : dofs) {
        for (GlobalIndex col_dof : dofs) {
            if (row_dof >= 0 && row_dof < total_rows_ &&
                col_dof >= 0 && col_dof < total_cols_) {
                addEntry(row_dof, col_dof);
            }
        }
    }
}

void BlockSparsity::addFieldCoupling(std::span<const GlobalIndex> row_dofs,
                                      std::span<const GlobalIndex> col_dofs,
                                      GlobalIndex row_block, GlobalIndex col_block) {
    validateBlockIndices(row_block, col_block);

    auto& block = blocks_[blockIndex(row_block, col_block)];
    FE_THROW_IF(block.isFinalized(), InvalidArgumentException,
                "Cannot add coupling to finalized block");

    // row_dofs and col_dofs are assumed to be local to their respective blocks
    block.addElementCouplings(row_dofs, col_dofs);
    is_finalized_ = false;
}

// ============================================================================
// Finalization
// ============================================================================

void BlockSparsity::finalize() {
    for (auto& block : blocks_) {
        if (!block.isFinalized()) {
            block.finalize();
        }
    }
    is_finalized_ = true;
}

void BlockSparsity::finalizeBlock(GlobalIndex block_row, GlobalIndex block_col) {
    validateBlockIndices(block_row, block_col);
    auto& block = blocks_[blockIndex(block_row, block_col)];
    if (!block.isFinalized()) {
        block.finalize();
    }
}

void BlockSparsity::ensureDiagonals() {
    GlobalIndex n_diag = std::min(n_block_rows_, n_block_cols_);
    for (GlobalIndex i = 0; i < n_diag; ++i) {
        auto& block = blocks_[blockIndex(i, i)];
        if (!block.isFinalized() && block.isSquare()) {
            block.ensureDiagonal();
        }
    }
}

// ============================================================================
// Conversion
// ============================================================================

SparsityPattern BlockSparsity::toMonolithic() const {
    SparsityPattern mono(total_rows_, total_cols_);

    for (GlobalIndex bi = 0; bi < n_block_rows_; ++bi) {
        GlobalIndex row_offset = row_block_offsets_[static_cast<std::size_t>(bi)];

        for (GlobalIndex bj = 0; bj < n_block_cols_; ++bj) {
            GlobalIndex col_offset = col_block_offsets_[static_cast<std::size_t>(bj)];
            const auto& block = blocks_[blockIndex(bi, bj)];

            if (block.getNnz() == 0) continue;

            if (block.isFinalized()) {
                for (GlobalIndex local_row = 0; local_row < block.numRows(); ++local_row) {
                    auto cols = block.getRowSpan(local_row);
                    for (GlobalIndex local_col : cols) {
                        mono.addEntry(row_offset + local_row,
                                     col_offset + local_col);
                    }
                }
            } else {
                // Block not finalized - iterate differently
                for (GlobalIndex local_row = 0; local_row < block.numRows(); ++local_row) {
                    for (GlobalIndex local_col = 0; local_col < block.numCols(); ++local_col) {
                        if (block.hasEntry(local_row, local_col)) {
                            mono.addEntry(row_offset + local_row,
                                         col_offset + local_col);
                        }
                    }
                }
            }
        }
    }

    mono.finalize();
    return mono;
}

BlockSparsity BlockSparsity::fromMonolithic(
    const SparsityPattern& pattern,
    std::vector<GlobalIndex> row_block_sizes,
    std::vector<GlobalIndex> col_block_sizes)
{
    BlockSparsity result(std::move(row_block_sizes), std::move(col_block_sizes));

    FE_CHECK_ARG(pattern.numRows() == result.total_rows_ &&
                 pattern.numCols() == result.total_cols_,
                 "Pattern dimensions don't match block sizes");

    // Extract entries into blocks
    if (pattern.isFinalized()) {
        for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
            auto cols = pattern.getRowSpan(row);
            for (GlobalIndex col : cols) {
                result.addEntry(row, col);
            }
        }
    } else {
        for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

std::vector<SparsityPattern> BlockSparsity::extractDiagonalBlocks() const {
    GlobalIndex n_diag = std::min(n_block_rows_, n_block_cols_);
    std::vector<SparsityPattern> diags;
    diags.reserve(static_cast<std::size_t>(n_diag));

    for (GlobalIndex i = 0; i < n_diag; ++i) {
        diags.push_back(blocks_[blockIndex(i, i)]);
    }

    return diags;
}

SparsityPattern BlockSparsity::extractSchurComplement() const {
    FE_CHECK_ARG(n_block_rows_ == 2 && n_block_cols_ == 2,
                 "Schur complement requires 2x2 block structure");

    // For [A B; C D], Schur complement S = D - C A^{-1} B
    // Structurally: S has entries wherever D has entries, plus
    // entries from C * fill(A) * B

    [[maybe_unused]] const auto& A = blocks_[blockIndex(0, 0)];
    const auto& B = blocks_[blockIndex(0, 1)];
    const auto& C = blocks_[blockIndex(1, 0)];
    const auto& D = blocks_[blockIndex(1, 1)];

    GlobalIndex n_schur = row_block_sizes_[1];
    SparsityPattern S(n_schur, n_schur);

    // Copy D structure
    if (D.isFinalized()) {
        for (GlobalIndex row = 0; row < D.numRows(); ++row) {
            auto cols = D.getRowSpan(row);
            S.addEntries(row, cols);
        }
    }

    // Add C * B structure (assuming A^{-1} fills all of A)
    // For each (i, k) in C and (k', j) in B where k, k' couple through A,
    // add (i, j) to S. With fill assumption, any k couples to any k'.
    // So S gets entries wherever C has a row and B has a column.

    std::vector<GlobalIndex> C_nonempty_rows;
    std::vector<GlobalIndex> B_nonempty_cols;

    for (GlobalIndex i = 0; i < C.numRows(); ++i) {
        if (C.getRowNnz(i) > 0) {
            C_nonempty_rows.push_back(i);
        }
    }

    // Find B columns - need transpose-like query
    // For now, use conservative estimate: all columns
    for (GlobalIndex j = 0; j < B.numCols(); ++j) {
        B_nonempty_cols.push_back(j);
    }

    // Add fill from C * A^{-1} * B
    for (GlobalIndex i : C_nonempty_rows) {
        S.addEntries(i, B_nonempty_cols);
    }

    S.finalize();
    return S;
}

// ============================================================================
// Statistics
// ============================================================================

BlockSparsityStats BlockSparsity::computeStats() const {
    BlockSparsityStats stats;
    stats.n_block_rows = n_block_rows_;
    stats.n_block_cols = n_block_cols_;
    stats.total_rows = total_rows_;
    stats.total_cols = total_cols_;

    GlobalIndex total_block_capacity = 0;

    for (GlobalIndex bi = 0; bi < n_block_rows_; ++bi) {
        for (GlobalIndex bj = 0; bj < n_block_cols_; ++bj) {
            const auto& block = blocks_[blockIndex(bi, bj)];
            GlobalIndex block_nnz = block.getNnz();

            stats.total_nnz += block_nnz;
            if (block_nnz > 0) {
                stats.n_nonzero_blocks++;
                stats.max_block_nnz = std::max(stats.max_block_nnz, block_nnz);
            } else {
                stats.n_zero_blocks++;
            }

            total_block_capacity += block.numRows() * block.numCols();
        }
    }

    if (total_block_capacity > 0) {
        stats.avg_block_density = static_cast<double>(stats.total_nnz) /
                                  static_cast<double>(total_block_capacity);
    }

    return stats;
}

GlobalIndex BlockSparsity::getTotalNnz() const {
    GlobalIndex total = 0;
    for (const auto& block : blocks_) {
        total += block.getNnz();
    }
    return total;
}

GlobalIndex BlockSparsity::getBlockNnz(GlobalIndex block_row, GlobalIndex block_col) const {
    validateBlockIndices(block_row, block_col);
    return blocks_[blockIndex(block_row, block_col)].getNnz();
}

// ============================================================================
// Validation
// ============================================================================

std::string BlockSparsity::validate() const {
    std::ostringstream oss;

    if (n_block_rows_ < 0 || n_block_cols_ < 0) {
        oss << "Invalid block dimensions: " << n_block_rows_ << " x " << n_block_cols_;
        return oss.str();
    }

    if (static_cast<GlobalIndex>(row_block_sizes_.size()) != n_block_rows_) {
        oss << "Row block sizes count mismatch";
        return oss.str();
    }

    if (static_cast<GlobalIndex>(col_block_sizes_.size()) != n_block_cols_) {
        oss << "Column block sizes count mismatch";
        return oss.str();
    }

    if (static_cast<GlobalIndex>(blocks_.size()) != n_block_rows_ * n_block_cols_) {
        oss << "Blocks array size mismatch";
        return oss.str();
    }

    // Validate each block
    for (GlobalIndex bi = 0; bi < n_block_rows_; ++bi) {
        for (GlobalIndex bj = 0; bj < n_block_cols_; ++bj) {
            const auto& block = blocks_[blockIndex(bi, bj)];

            GlobalIndex expected_rows = row_block_sizes_[static_cast<std::size_t>(bi)];
            GlobalIndex expected_cols = col_block_sizes_[static_cast<std::size_t>(bj)];

            if (block.numRows() != expected_rows || block.numCols() != expected_cols) {
                oss << "Block (" << bi << "," << bj << ") dimension mismatch: "
                    << block.numRows() << "x" << block.numCols()
                    << " vs expected " << expected_rows << "x" << expected_cols;
                return oss.str();
            }

            if (!block.validate()) {
                oss << "Block (" << bi << "," << bj << ") failed validation: "
                    << block.validationError();
                return oss.str();
            }
        }
    }

    return "";  // Valid
}

std::size_t BlockSparsity::memoryUsageBytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    bytes += row_block_sizes_.capacity() * sizeof(GlobalIndex);
    bytes += col_block_sizes_.capacity() * sizeof(GlobalIndex);
    bytes += row_block_offsets_.capacity() * sizeof(GlobalIndex);
    bytes += col_block_offsets_.capacity() * sizeof(GlobalIndex);
    bytes += row_block_info_.capacity() * sizeof(BlockInfo);
    bytes += col_block_info_.capacity() * sizeof(BlockInfo);
    bytes += blocks_.capacity() * sizeof(SparsityPattern);

    for (const auto& block : blocks_) {
        bytes += block.memoryUsageBytes();
    }

    return bytes;
}

// ============================================================================
// Private helpers
// ============================================================================

void BlockSparsity::computeOffsets() {
    row_block_offsets_.resize(row_block_sizes_.size() + 1);
    col_block_offsets_.resize(col_block_sizes_.size() + 1);

    row_block_offsets_[0] = 0;
    for (std::size_t i = 0; i < row_block_sizes_.size(); ++i) {
        row_block_offsets_[i + 1] = row_block_offsets_[i] + row_block_sizes_[i];
    }

    col_block_offsets_[0] = 0;
    for (std::size_t j = 0; j < col_block_sizes_.size(); ++j) {
        col_block_offsets_[j + 1] = col_block_offsets_[j] + col_block_sizes_[j];
    }

    total_rows_ = row_block_offsets_.back();
    total_cols_ = col_block_offsets_.back();
}

void BlockSparsity::validateBlockIndices(GlobalIndex block_row, GlobalIndex block_col) const {
    FE_CHECK_ARG(block_row >= 0 && block_row < n_block_rows_,
                 "Block row index out of range");
    FE_CHECK_ARG(block_col >= 0 && block_col < n_block_cols_,
                 "Block column index out of range");
}

std::pair<GlobalIndex, GlobalIndex> BlockSparsity::scalarToBlock(
    GlobalIndex row, GlobalIndex col) const
{
    // Binary search for block row
    auto row_it = std::upper_bound(row_block_offsets_.begin(),
                                   row_block_offsets_.end(), row);
    GlobalIndex block_row = static_cast<GlobalIndex>(
        std::distance(row_block_offsets_.begin(), row_it)) - 1;

    // Binary search for block col
    auto col_it = std::upper_bound(col_block_offsets_.begin(),
                                   col_block_offsets_.end(), col);
    GlobalIndex block_col = static_cast<GlobalIndex>(
        std::distance(col_block_offsets_.begin(), col_it)) - 1;

    return {block_row, block_col};
}

std::size_t BlockSparsity::blockIndex(GlobalIndex block_row, GlobalIndex block_col) const {
    return static_cast<std::size_t>(block_row * n_block_cols_ + block_col);
}

// ============================================================================
// Free functions
// ============================================================================

BlockSparsity createDiagonalBlockStructure(const std::vector<GlobalIndex>& block_sizes) {
    BlockSparsity result(block_sizes);
    // All off-diagonal blocks remain empty
    return result;
}

BlockSparsity blockDiagonal(const std::vector<SparsityPattern>& patterns) {
    std::vector<GlobalIndex> block_sizes;
    block_sizes.reserve(patterns.size());

    for (const auto& p : patterns) {
        FE_CHECK_ARG(p.isSquare(), "Block diagonal requires square patterns");
        block_sizes.push_back(p.numRows());
    }

    BlockSparsity result(block_sizes);

    for (std::size_t i = 0; i < patterns.size(); ++i) {
        result.setBlock(static_cast<GlobalIndex>(i),
                       static_cast<GlobalIndex>(i),
                       patterns[i]);
    }

    return result;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
