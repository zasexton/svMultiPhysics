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

#include "SparsityFormat.h"
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// COOData implementation
// ============================================================================

void COOData::sort() {
    if (rows.empty()) return;

    // Create index array for stable sort
    std::vector<std::size_t> indices(rows.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by (row, col)
    std::sort(indices.begin(), indices.end(),
              [this](std::size_t i, std::size_t j) {
                  if (rows[i] != rows[j]) return rows[i] < rows[j];
                  return cols[i] < cols[j];
              });

    // Apply permutation
    std::vector<GlobalIndex> new_rows(rows.size());
    std::vector<GlobalIndex> new_cols(cols.size());

    for (std::size_t i = 0; i < indices.size(); ++i) {
        new_rows[i] = rows[indices[i]];
        new_cols[i] = cols[indices[i]];
    }

    rows = std::move(new_rows);
    cols = std::move(new_cols);
}

void COOData::deduplicate() {
    if (rows.empty()) return;

    // Sort first
    sort();

    // Remove duplicates
    std::vector<GlobalIndex> new_rows;
    std::vector<GlobalIndex> new_cols;
    new_rows.reserve(rows.size());
    new_cols.reserve(cols.size());

    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (i == 0 || rows[i] != new_rows.back() || cols[i] != new_cols.back()) {
            new_rows.push_back(rows[i]);
            new_cols.push_back(cols[i]);
        }
    }

    rows = std::move(new_rows);
    cols = std::move(new_cols);
}

// ============================================================================
// CSCData implementation
// ============================================================================

std::span<const GlobalIndex> CSCData::getColumn(GlobalIndex col) const {
    FE_CHECK_ARG(col >= 0 && col < n_cols, "Column index out of range");
    GlobalIndex start = col_ptr[static_cast<std::size_t>(col)];
    GlobalIndex end = col_ptr[static_cast<std::size_t>(col) + 1];
    return std::span<const GlobalIndex>(
        row_idx.data() + start,
        static_cast<std::size_t>(end - start));
}

GlobalIndex CSCData::getColNnz(GlobalIndex col) const {
    FE_CHECK_ARG(col >= 0 && col < n_cols, "Column index out of range");
    return col_ptr[static_cast<std::size_t>(col) + 1] -
           col_ptr[static_cast<std::size_t>(col)];
}

bool CSCData::isValid() const noexcept {
    if (n_rows < 0 || n_cols < 0) return false;
    if (static_cast<GlobalIndex>(col_ptr.size()) != n_cols + 1) return false;
    if (col_ptr.empty()) return true;
    if (col_ptr[0] != 0) return false;

    GlobalIndex nnz = col_ptr.back();
    if (static_cast<GlobalIndex>(row_idx.size()) != nnz) return false;

    // Check monotonicity and row indices
    for (GlobalIndex col = 0; col < n_cols; ++col) {
        GlobalIndex start = col_ptr[static_cast<std::size_t>(col)];
        GlobalIndex end = col_ptr[static_cast<std::size_t>(col) + 1];
        if (end < start) return false;

        GlobalIndex prev = -1;
        for (GlobalIndex i = start; i < end; ++i) {
            GlobalIndex row = row_idx[static_cast<std::size_t>(i)];
            if (row < 0 || row >= n_rows) return false;
            if (row <= prev) return false;  // Not sorted or duplicate
            prev = row;
        }
    }

    return true;
}

// ============================================================================
// BSRData implementation
// ============================================================================

bool BSRData::isValid() const noexcept {
    if (n_block_rows < 0 || n_block_cols < 0) return false;
    if (block_size_row <= 0 || block_size_col <= 0) return false;
    if (static_cast<GlobalIndex>(row_ptr.size()) != n_block_rows + 1) return false;
    if (row_ptr.empty()) return true;
    if (row_ptr[0] != 0) return false;

    GlobalIndex nb = row_ptr.back();
    if (static_cast<GlobalIndex>(col_idx.size()) != nb) return false;

    // Check monotonicity and column indices
    for (GlobalIndex br = 0; br < n_block_rows; ++br) {
        GlobalIndex start = row_ptr[static_cast<std::size_t>(br)];
        GlobalIndex end = row_ptr[static_cast<std::size_t>(br) + 1];
        if (end < start) return false;

        GlobalIndex prev = -1;
        for (GlobalIndex i = start; i < end; ++i) {
            GlobalIndex bc = col_idx[static_cast<std::size_t>(i)];
            if (bc < 0 || bc >= n_block_cols) return false;
            if (bc <= prev) return false;
            prev = bc;
        }
    }

    return true;
}

// ============================================================================
// ELLPACKData implementation
// ============================================================================

GlobalIndex ELLPACKData::getEntry(GlobalIndex row, GlobalIndex k) const {
    FE_CHECK_ARG(row >= 0 && row < n_rows, "Row index out of range");
    FE_CHECK_ARG(k >= 0 && k < max_nnz_per_row, "Entry index out of range");
    return col_idx[static_cast<std::size_t>(row * max_nnz_per_row + k)];
}

GlobalIndex ELLPACKData::nnz() const {
    GlobalIndex count = 0;
    for (GlobalIndex val : col_idx) {
        if (val != padding_value) {
            count++;
        }
    }
    return count;
}

bool ELLPACKData::isValid() const noexcept {
    if (n_rows < 0 || n_cols < 0 || max_nnz_per_row < 0) return false;
    if (static_cast<GlobalIndex>(col_idx.size()) != n_rows * max_nnz_per_row) {
        return false;
    }

    // Check column indices are valid or padding
    for (std::size_t i = 0; i < col_idx.size(); ++i) {
        GlobalIndex val = col_idx[i];
        if (val != padding_value && (val < 0 || val >= n_cols)) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Format conversion functions
// ============================================================================

COOData csrToCoo(const SparsityPattern& pattern) {
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized for conversion");

    COOData coo;
    coo.n_rows = pattern.numRows();
    coo.n_cols = pattern.numCols();

    GlobalIndex nnz = pattern.getNnz();
    coo.rows.reserve(static_cast<std::size_t>(nnz));
    coo.cols.reserve(static_cast<std::size_t>(nnz));

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        auto cols = pattern.getRowSpan(row);
        for (GlobalIndex col : cols) {
            coo.rows.push_back(row);
            coo.cols.push_back(col);
        }
    }

    return coo;
}

SparsityPattern cooToCsr(const COOData& coo) {
    FE_THROW_IF(!coo.isValid(), InvalidArgumentException,
                "Invalid COO data");

    SparsityPattern pattern(coo.n_rows, coo.n_cols);

    for (std::size_t i = 0; i < coo.rows.size(); ++i) {
        pattern.addEntry(coo.rows[i], coo.cols[i]);
    }

    pattern.finalize();
    return pattern;
}

CSCData csrToCsc(const SparsityPattern& pattern) {
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized for conversion");

    CSCData csc;
    csc.n_rows = pattern.numRows();
    csc.n_cols = pattern.numCols();

    GlobalIndex nnz = pattern.getNnz();
    csc.col_ptr.resize(static_cast<std::size_t>(csc.n_cols) + 1);
    csc.row_idx.resize(static_cast<std::size_t>(nnz));

    // Count entries per column
    std::fill(csc.col_ptr.begin(), csc.col_ptr.end(), 0);
    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        auto cols = pattern.getRowSpan(row);
        for (GlobalIndex col : cols) {
            csc.col_ptr[static_cast<std::size_t>(col) + 1]++;
        }
    }

    // Compute column pointers (cumulative sum)
    for (GlobalIndex col = 0; col < csc.n_cols; ++col) {
        csc.col_ptr[static_cast<std::size_t>(col) + 1] +=
            csc.col_ptr[static_cast<std::size_t>(col)];
    }

    // Fill row indices (need working copy of col_ptr)
    std::vector<GlobalIndex> col_count(csc.col_ptr.begin(), csc.col_ptr.end() - 1);
    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        auto cols = pattern.getRowSpan(row);
        for (GlobalIndex col : cols) {
            GlobalIndex idx = col_count[static_cast<std::size_t>(col)]++;
            csc.row_idx[static_cast<std::size_t>(idx)] = row;
        }
    }

    // Sort row indices within each column (should already be sorted due to CSR order)
    for (GlobalIndex col = 0; col < csc.n_cols; ++col) {
        GlobalIndex start = csc.col_ptr[static_cast<std::size_t>(col)];
        GlobalIndex end = csc.col_ptr[static_cast<std::size_t>(col) + 1];
        std::sort(csc.row_idx.begin() + start, csc.row_idx.begin() + end);
    }

    return csc;
}

SparsityPattern cscToCsr(const CSCData& csc) {
    FE_THROW_IF(!csc.isValid(), InvalidArgumentException,
                "Invalid CSC data");

    SparsityPattern pattern(csc.n_rows, csc.n_cols);

    for (GlobalIndex col = 0; col < csc.n_cols; ++col) {
        auto rows = csc.getColumn(col);
        for (GlobalIndex row : rows) {
            pattern.addEntry(row, col);
        }
    }

    pattern.finalize();
    return pattern;
}

BSRData csrToBsr(const SparsityPattern& pattern, GlobalIndex block_size) {
    return csrToBsr(pattern, block_size, block_size);
}

BSRData csrToBsr(const SparsityPattern& pattern,
                  GlobalIndex block_rows, GlobalIndex block_cols) {
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized for conversion");
    FE_CHECK_ARG(block_rows > 0 && block_cols > 0,
                 "Block sizes must be positive");
    FE_CHECK_ARG(pattern.numRows() % block_rows == 0,
                 "Number of rows must be divisible by block row size");
    FE_CHECK_ARG(pattern.numCols() % block_cols == 0,
                 "Number of columns must be divisible by block column size");

    BSRData bsr;
    bsr.n_block_rows = pattern.numRows() / block_rows;
    bsr.n_block_cols = pattern.numCols() / block_cols;
    bsr.block_size_row = block_rows;
    bsr.block_size_col = block_cols;

    // Find which blocks are present
    std::vector<std::unordered_set<GlobalIndex>> block_cols_per_row(
        static_cast<std::size_t>(bsr.n_block_rows));

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        GlobalIndex br = row / block_rows;
        auto cols = pattern.getRowSpan(row);
        for (GlobalIndex col : cols) {
            GlobalIndex bc = col / block_cols;
            block_cols_per_row[static_cast<std::size_t>(br)].insert(bc);
        }
    }

    // Build BSR structure
    bsr.row_ptr.resize(static_cast<std::size_t>(bsr.n_block_rows) + 1);
    bsr.row_ptr[0] = 0;

    for (GlobalIndex br = 0; br < bsr.n_block_rows; ++br) {
        const auto& block_set = block_cols_per_row[static_cast<std::size_t>(br)];
        bsr.row_ptr[static_cast<std::size_t>(br) + 1] =
            bsr.row_ptr[static_cast<std::size_t>(br)] +
            static_cast<GlobalIndex>(block_set.size());
    }

    bsr.col_idx.resize(static_cast<std::size_t>(bsr.row_ptr.back()));

    // Fill block column indices (sorted)
    GlobalIndex idx = 0;
    for (GlobalIndex br = 0; br < bsr.n_block_rows; ++br) {
        auto& block_set = block_cols_per_row[static_cast<std::size_t>(br)];
        std::vector<GlobalIndex> sorted_cols(block_set.begin(), block_set.end());
        std::sort(sorted_cols.begin(), sorted_cols.end());
        for (GlobalIndex bc : sorted_cols) {
            bsr.col_idx[static_cast<std::size_t>(idx++)] = bc;
        }
    }

    return bsr;
}

SparsityPattern bsrToCsr(const BSRData& bsr) {
    FE_THROW_IF(!bsr.isValid(), InvalidArgumentException,
                "Invalid BSR data");

    GlobalIndex n_rows = bsr.numRows();
    GlobalIndex n_cols = bsr.numCols();
    SparsityPattern pattern(n_rows, n_cols);

    for (GlobalIndex br = 0; br < bsr.n_block_rows; ++br) {
        GlobalIndex start = bsr.row_ptr[static_cast<std::size_t>(br)];
        GlobalIndex end = bsr.row_ptr[static_cast<std::size_t>(br) + 1];

        for (GlobalIndex i = start; i < end; ++i) {
            GlobalIndex bc = bsr.col_idx[static_cast<std::size_t>(i)];

            // Add all entries in block (br, bc)
            for (GlobalIndex lr = 0; lr < bsr.block_size_row; ++lr) {
                GlobalIndex row = br * bsr.block_size_row + lr;
                for (GlobalIndex lc = 0; lc < bsr.block_size_col; ++lc) {
                    GlobalIndex col = bc * bsr.block_size_col + lc;
                    pattern.addEntry(row, col);
                }
            }
        }
    }

    pattern.finalize();
    return pattern;
}

ELLPACKData csrToEllpack(const SparsityPattern& pattern, GlobalIndex padding_value) {
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized for conversion");

    ELLPACKData ell;
    ell.n_rows = pattern.numRows();
    ell.n_cols = pattern.numCols();
    ell.padding_value = padding_value;

    // Find max entries per row
    ell.max_nnz_per_row = 0;
    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        ell.max_nnz_per_row = std::max(ell.max_nnz_per_row, pattern.getRowNnz(row));
    }

    // Allocate and fill
    ell.col_idx.resize(static_cast<std::size_t>(ell.n_rows * ell.max_nnz_per_row),
                       padding_value);

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        auto cols = pattern.getRowSpan(row);
        GlobalIndex k = 0;
        for (GlobalIndex col : cols) {
            ell.col_idx[static_cast<std::size_t>(row * ell.max_nnz_per_row + k)] = col;
            k++;
        }
    }

    return ell;
}

SparsityPattern ellpackToCsr(const ELLPACKData& ellpack) {
    FE_THROW_IF(!ellpack.isValid(), InvalidArgumentException,
                "Invalid ELLPACK data");

    SparsityPattern pattern(ellpack.n_rows, ellpack.n_cols);

    for (GlobalIndex row = 0; row < ellpack.n_rows; ++row) {
        for (GlobalIndex k = 0; k < ellpack.max_nnz_per_row; ++k) {
            GlobalIndex col = ellpack.col_idx[static_cast<std::size_t>(
                row * ellpack.max_nnz_per_row + k)];
            if (col != ellpack.padding_value) {
                pattern.addEntry(row, col);
            }
        }
    }

    pattern.finalize();
    return pattern;
}

// ============================================================================
// Index conversion utilities
// ============================================================================

void convertIndexBase(std::span<GlobalIndex> indices,
                      IndexBase from, IndexBase to) {
    if (from == to) return;

    GlobalIndex offset = (to == IndexBase::OneBased) ? 1 : -1;
    for (GlobalIndex& idx : indices) {
        idx += offset;
    }
}

std::tuple<std::vector<GlobalIndex>, std::vector<GlobalIndex>>
getCSRArrays(const SparsityPattern& pattern, IndexBase base) {
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized");

    auto row_ptr_span = pattern.getRowPtr();
    auto col_idx_span = pattern.getColIndices();

    std::vector<GlobalIndex> row_ptr(row_ptr_span.begin(), row_ptr_span.end());
    std::vector<GlobalIndex> col_idx(col_idx_span.begin(), col_idx_span.end());

    if (base == IndexBase::OneBased) {
        convertIndexBase(row_ptr, IndexBase::ZeroBased, IndexBase::OneBased);
        convertIndexBase(col_idx, IndexBase::ZeroBased, IndexBase::OneBased);
    }

    return {std::move(row_ptr), std::move(col_idx)};
}

// ============================================================================
// Format analysis utilities
// ============================================================================

bool isBSRCompatible(const SparsityPattern& pattern, GlobalIndex block_size) {
    if (!pattern.isFinalized()) return false;
    if (block_size <= 0) return false;
    if (pattern.numRows() % block_size != 0) return false;
    if (pattern.numCols() % block_size != 0) return false;

    // Check that entries form complete blocks
    // For each block that has any entry, check all entries exist
    GlobalIndex n_block_rows = pattern.numRows() / block_size;
    GlobalIndex n_block_cols = pattern.numCols() / block_size;

    for (GlobalIndex br = 0; br < n_block_rows; ++br) {
        for (GlobalIndex bc = 0; bc < n_block_cols; ++bc) {
            // Check if any entry exists in this block
            bool has_any = false;
            for (GlobalIndex lr = 0; lr < block_size && !has_any; ++lr) {
                for (GlobalIndex lc = 0; lc < block_size && !has_any; ++lc) {
                    if (pattern.hasEntry(br * block_size + lr,
                                        bc * block_size + lc)) {
                        has_any = true;
                    }
                }
            }

            if (has_any) {
                // Check all entries exist
                for (GlobalIndex lr = 0; lr < block_size; ++lr) {
                    for (GlobalIndex lc = 0; lc < block_size; ++lc) {
                        if (!pattern.hasEntry(br * block_size + lr,
                                             bc * block_size + lc)) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}

GlobalIndex detectBlockSize(const SparsityPattern& pattern,
                             GlobalIndex max_block_size) {
    if (!pattern.isFinalized() || !pattern.isSquare()) {
        return 1;
    }

    // Try block sizes from largest to smallest
    for (GlobalIndex bs = max_block_size; bs > 1; --bs) {
        if (pattern.numRows() % bs == 0 && isBSRCompatible(pattern, bs)) {
            return bs;
        }
    }

    return 1;
}

double computeEllpackEfficiency(const SparsityPattern& pattern) {
    if (!pattern.isFinalized() || pattern.numRows() == 0) {
        return 0.0;
    }

    GlobalIndex actual_nnz = pattern.getNnz();
    GlobalIndex max_row_nnz = 0;
    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        max_row_nnz = std::max(max_row_nnz, pattern.getRowNnz(row));
    }

    GlobalIndex ellpack_storage = pattern.numRows() * max_row_nnz;
    if (ellpack_storage == 0) return 1.0;

    return static_cast<double>(actual_nnz) / static_cast<double>(ellpack_storage);
}

SparseFormat recommendFormat(const SparsityPattern& pattern) {
    if (!pattern.isFinalized()) {
        return SparseFormat::CSR;
    }

    // Check for block structure
    GlobalIndex block_size = detectBlockSize(pattern, 8);
    if (block_size > 1) {
        return SparseFormat::BSR;
    }

    // Check ELLPACK efficiency
    double ell_eff = computeEllpackEfficiency(pattern);
    if (ell_eff > 0.7) {
        return SparseFormat::ELLPACK;
    }

    // Default to CSR
    return SparseFormat::CSR;
}

// ============================================================================
// SparsityFormatConverter implementation
// ============================================================================

SparsityFormatConverter::SparsityFormatConverter(const SparsityPattern& pattern)
    : pattern_(pattern)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized");
}

const COOData& SparsityFormatConverter::asCOO() {
    if (!coo_cache_) {
        coo_cache_ = csrToCoo(pattern_);
    }
    return *coo_cache_;
}

const CSCData& SparsityFormatConverter::asCSC() {
    if (!csc_cache_) {
        csc_cache_ = csrToCsc(pattern_);
    }
    return *csc_cache_;
}

const BSRData& SparsityFormatConverter::asBSR(GlobalIndex block_size) {
    if (!bsr_cache_ || bsr_block_size_ != block_size) {
        bsr_cache_ = csrToBsr(pattern_, block_size);
        bsr_block_size_ = block_size;
    }
    return *bsr_cache_;
}

const ELLPACKData& SparsityFormatConverter::asELLPACK() {
    if (!ellpack_cache_) {
        ellpack_cache_ = csrToEllpack(pattern_);
    }
    return *ellpack_cache_;
}

void SparsityFormatConverter::clearCache() {
    coo_cache_.reset();
    csc_cache_.reset();
    bsr_cache_.reset();
    ellpack_cache_.reset();
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
