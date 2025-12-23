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

#ifndef SVMP_FE_SPARSITY_SPARSITY_FORMAT_H
#define SVMP_FE_SPARSITY_SPARSITY_FORMAT_H

/**
 * @file SparsityFormat.h
 * @brief Convert between sparse storage formats (CSR/CSC/COO/BSR)
 *
 * This header provides utilities for converting sparsity patterns between
 * different sparse matrix storage formats. These are structure-only
 * operations (no values involved).
 *
 * Supported formats:
 * - CSR (Compressed Sparse Row) - our native format
 * - CSC (Compressed Sparse Column)
 * - COO (Coordinate/triplet format)
 * - BSR (Block Sparse Row)
 * - ELLPACK (for GPU-oriented storage)
 *
 * Format descriptions:
 * - CSR: row_ptr[n+1], col_idx[nnz] - efficient row access
 * - CSC: col_ptr[m+1], row_idx[nnz] - efficient column access
 * - COO: rows[nnz], cols[nnz] - easy construction, inefficient ops
 * - BSR: row_ptr[nb+1], col_idx[nnz_blocks] with fixed block_size
 * - ELLPACK: col_idx[n * max_nnz_per_row] padded to uniform width
 *
 * Design notes:
 * - All conversions are structure-only (no matrix values)
 * - Conversions return new data structures (no in-place modification)
 * - Index types are configurable via template parameters
 * - Zero-based indexing throughout
 *
 * @see SparsityPattern for the primary CSR representation
 */

#include "SparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <tuple>
#include <optional>
#include <cstdint>

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Sparse storage format identifier
 */
enum class SparseFormat : std::uint8_t {
    CSR,        ///< Compressed Sparse Row
    CSC,        ///< Compressed Sparse Column
    COO,        ///< Coordinate (triplet)
    BSR,        ///< Block Sparse Row
    ELLPACK,    ///< ELLPACK (padded row format)
    DIA,        ///< Diagonal format (for banded matrices)
    Custom
};

/**
 * @brief Index base (0 or 1)
 */
enum class IndexBase : std::uint8_t {
    ZeroBased = 0,
    OneBased = 1
};

/**
 * @brief COO (coordinate) format data structure
 *
 * Stores entries as (row, col) pairs. Commonly used for:
 * - Matrix Market format I/O
 * - Simple construction
 * - Format conversion intermediate
 */
struct COOData {
    std::vector<GlobalIndex> rows;    ///< Row indices
    std::vector<GlobalIndex> cols;    ///< Column indices
    GlobalIndex n_rows{0};            ///< Number of rows
    GlobalIndex n_cols{0};            ///< Number of columns

    /**
     * @brief Get number of entries
     */
    [[nodiscard]] GlobalIndex nnz() const noexcept {
        return static_cast<GlobalIndex>(rows.size());
    }

    /**
     * @brief Check if data is valid
     */
    [[nodiscard]] bool isValid() const noexcept {
        return rows.size() == cols.size();
    }

    /**
     * @brief Clear data
     */
    void clear() {
        rows.clear();
        cols.clear();
    }

    /**
     * @brief Sort entries by (row, col) for determinism
     */
    void sort();

    /**
     * @brief Remove duplicate entries
     */
    void deduplicate();
};

/**
 * @brief CSC (Compressed Sparse Column) format data structure
 *
 * Column-oriented analog of CSR. Efficient for:
 * - Column slicing
 * - Transpose operations
 * - Some direct solvers (CHOLMOD, etc.)
 */
struct CSCData {
    std::vector<GlobalIndex> col_ptr;   ///< Column pointers [n_cols + 1]
    std::vector<GlobalIndex> row_idx;   ///< Row indices [nnz]
    GlobalIndex n_rows{0};              ///< Number of rows
    GlobalIndex n_cols{0};              ///< Number of columns

    /**
     * @brief Get number of entries
     */
    [[nodiscard]] GlobalIndex nnz() const noexcept {
        return col_ptr.empty() ? 0 : col_ptr.back();
    }

    /**
     * @brief Get entries in column
     */
    [[nodiscard]] std::span<const GlobalIndex> getColumn(GlobalIndex col) const;

    /**
     * @brief Get column NNZ
     */
    [[nodiscard]] GlobalIndex getColNnz(GlobalIndex col) const;

    /**
     * @brief Check if data is valid
     */
    [[nodiscard]] bool isValid() const noexcept;

    /**
     * @brief Clear data
     */
    void clear() {
        col_ptr.clear();
        row_idx.clear();
        n_rows = 0;
        n_cols = 0;
    }
};

/**
 * @brief BSR (Block Sparse Row) format data structure
 *
 * Stores dense blocks instead of individual entries.
 * Efficient when matrix has natural block structure.
 */
struct BSRData {
    std::vector<GlobalIndex> row_ptr;     ///< Block row pointers [n_block_rows + 1]
    std::vector<GlobalIndex> col_idx;     ///< Block column indices [nnz_blocks]
    GlobalIndex n_block_rows{0};          ///< Number of block rows
    GlobalIndex n_block_cols{0};          ///< Number of block columns
    GlobalIndex block_size_row{1};        ///< Block size (rows)
    GlobalIndex block_size_col{1};        ///< Block size (columns)

    /**
     * @brief Get number of blocks
     */
    [[nodiscard]] GlobalIndex nnzBlocks() const noexcept {
        return row_ptr.empty() ? 0 : row_ptr.back();
    }

    /**
     * @brief Get scalar dimensions
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept {
        return n_block_rows * block_size_row;
    }
    [[nodiscard]] GlobalIndex numCols() const noexcept {
        return n_block_cols * block_size_col;
    }

    /**
     * @brief Get scalar NNZ
     */
    [[nodiscard]] GlobalIndex nnz() const noexcept {
        return nnzBlocks() * block_size_row * block_size_col;
    }

    /**
     * @brief Check if data is valid
     */
    [[nodiscard]] bool isValid() const noexcept;

    /**
     * @brief Clear data
     */
    void clear() {
        row_ptr.clear();
        col_idx.clear();
        n_block_rows = 0;
        n_block_cols = 0;
    }
};

/**
 * @brief ELLPACK format data structure
 *
 * Pads rows to uniform length for GPU efficiency.
 * Good for matrices with similar row lengths.
 */
struct ELLPACKData {
    std::vector<GlobalIndex> col_idx;   ///< Column indices [n_rows * max_nnz_per_row]
    GlobalIndex n_rows{0};              ///< Number of rows
    GlobalIndex n_cols{0};              ///< Number of columns
    GlobalIndex max_nnz_per_row{0};     ///< Maximum entries per row (padding width)
    GlobalIndex padding_value{-1};      ///< Value used for padding

    /**
     * @brief Get column index at (row, k)
     *
     * @param row Row index
     * @param k Entry index within row
     * @return Column index or padding_value
     */
    [[nodiscard]] GlobalIndex getEntry(GlobalIndex row, GlobalIndex k) const;

    /**
     * @brief Get actual NNZ (excluding padding)
     */
    [[nodiscard]] GlobalIndex nnz() const;

    /**
     * @brief Check if data is valid
     */
    [[nodiscard]] bool isValid() const noexcept;

    /**
     * @brief Clear data
     */
    void clear() {
        col_idx.clear();
        n_rows = 0;
        n_cols = 0;
        max_nnz_per_row = 0;
    }
};

// ============================================================================
// Format conversion functions
// ============================================================================

/**
 * @brief Convert SparsityPattern (CSR) to COO format
 *
 * @param pattern Source pattern (must be finalized)
 * @return COO data
 */
[[nodiscard]] COOData csrToCoo(const SparsityPattern& pattern);

/**
 * @brief Convert COO to SparsityPattern (CSR)
 *
 * @param coo Source COO data
 * @return Finalized sparsity pattern
 */
[[nodiscard]] SparsityPattern cooToCsr(const COOData& coo);

/**
 * @brief Convert SparsityPattern (CSR) to CSC format
 *
 * @param pattern Source pattern (must be finalized)
 * @return CSC data
 */
[[nodiscard]] CSCData csrToCsc(const SparsityPattern& pattern);

/**
 * @brief Convert CSC to SparsityPattern (CSR)
 *
 * @param csc Source CSC data
 * @return Finalized sparsity pattern
 */
[[nodiscard]] SparsityPattern cscToCsr(const CSCData& csc);

/**
 * @brief Convert SparsityPattern (CSR) to BSR format
 *
 * Requires pattern dimensions to be divisible by block size.
 *
 * @param pattern Source pattern (must be finalized)
 * @param block_size Block size (square blocks)
 * @return BSR data
 * @throws FEException if dimensions not divisible by block_size
 */
[[nodiscard]] BSRData csrToBsr(const SparsityPattern& pattern,
                                GlobalIndex block_size);

/**
 * @brief Convert SparsityPattern to BSR with rectangular blocks
 *
 * @param pattern Source pattern
 * @param block_rows Block row size
 * @param block_cols Block column size
 * @return BSR data
 */
[[nodiscard]] BSRData csrToBsr(const SparsityPattern& pattern,
                                GlobalIndex block_rows,
                                GlobalIndex block_cols);

/**
 * @brief Convert BSR to SparsityPattern (CSR)
 *
 * @param bsr Source BSR data
 * @return Finalized sparsity pattern
 */
[[nodiscard]] SparsityPattern bsrToCsr(const BSRData& bsr);

/**
 * @brief Convert SparsityPattern to ELLPACK format
 *
 * @param pattern Source pattern (must be finalized)
 * @param padding_value Value for padding (-1 by default)
 * @return ELLPACK data
 */
[[nodiscard]] ELLPACKData csrToEllpack(const SparsityPattern& pattern,
                                        GlobalIndex padding_value = -1);

/**
 * @brief Convert ELLPACK to SparsityPattern (CSR)
 *
 * @param ellpack Source ELLPACK data
 * @return Finalized sparsity pattern
 */
[[nodiscard]] SparsityPattern ellpackToCsr(const ELLPACKData& ellpack);

// ============================================================================
// Index conversion utilities
// ============================================================================

/**
 * @brief Convert indices to different base (0->1 or 1->0)
 *
 * @param indices Index array to convert (modified in-place)
 * @param from Current index base
 * @param to Target index base
 */
void convertIndexBase(std::span<GlobalIndex> indices,
                      IndexBase from, IndexBase to);

/**
 * @brief Get CSR arrays with specified index base
 *
 * @param pattern Source pattern
 * @param base Target index base
 * @return Tuple of (row_ptr, col_idx) with requested base
 */
[[nodiscard]] std::tuple<std::vector<GlobalIndex>, std::vector<GlobalIndex>>
    getCSRArrays(const SparsityPattern& pattern, IndexBase base);

/**
 * @brief Get CSR arrays with specific integer type
 *
 * @tparam IndexType Target integer type
 * @param pattern Source pattern
 * @param base Target index base
 * @return Tuple of (row_ptr, col_idx) as IndexType
 */
template<typename IndexType>
[[nodiscard]] std::tuple<std::vector<IndexType>, std::vector<IndexType>>
getCSRArraysAs(const SparsityPattern& pattern, IndexBase base = IndexBase::ZeroBased) {
    auto [row_ptr, col_idx] = getCSRArrays(pattern, base);

    std::vector<IndexType> row_ptr_typed(row_ptr.size());
    std::vector<IndexType> col_idx_typed(col_idx.size());

    for (std::size_t i = 0; i < row_ptr.size(); ++i) {
        row_ptr_typed[i] = static_cast<IndexType>(row_ptr[i]);
    }
    for (std::size_t i = 0; i < col_idx.size(); ++i) {
        col_idx_typed[i] = static_cast<IndexType>(col_idx[i]);
    }

    return {std::move(row_ptr_typed), std::move(col_idx_typed)};
}

// ============================================================================
// Format analysis utilities
// ============================================================================

/**
 * @brief Check if pattern is suitable for BSR with given block size
 *
 * @param pattern Pattern to check
 * @param block_size Block size
 * @return true if pattern is block-structured
 */
[[nodiscard]] bool isBSRCompatible(const SparsityPattern& pattern,
                                    GlobalIndex block_size);

/**
 * @brief Detect natural block size from pattern
 *
 * Analyzes pattern structure to find repeating block patterns.
 *
 * @param pattern Pattern to analyze
 * @param max_block_size Maximum block size to consider
 * @return Detected block size, or 1 if no block structure found
 */
[[nodiscard]] GlobalIndex detectBlockSize(const SparsityPattern& pattern,
                                           GlobalIndex max_block_size = 16);

/**
 * @brief Compute ELLPACK efficiency
 *
 * @param pattern Pattern to analyze
 * @return Ratio of actual NNZ to ELLPACK storage (higher is better)
 */
[[nodiscard]] double computeEllpackEfficiency(const SparsityPattern& pattern);

/**
 * @brief Recommend best format for pattern
 *
 * Analyzes pattern characteristics and recommends storage format.
 *
 * @param pattern Pattern to analyze
 * @return Recommended format
 */
[[nodiscard]] SparseFormat recommendFormat(const SparsityPattern& pattern);

// ============================================================================
// SparsityFormatConverter class
// ============================================================================

/**
 * @brief Utility class for format conversions with caching
 *
 * Caches converted formats to avoid repeated conversions.
 */
class SparsityFormatConverter {
public:
    /**
     * @brief Construct with source pattern
     *
     * @param pattern Source CSR pattern (must be finalized)
     */
    explicit SparsityFormatConverter(const SparsityPattern& pattern);

    /**
     * @brief Get as COO format
     */
    [[nodiscard]] const COOData& asCOO();

    /**
     * @brief Get as CSC format
     */
    [[nodiscard]] const CSCData& asCSC();

    /**
     * @brief Get as BSR format
     *
     * @param block_size Block size
     */
    [[nodiscard]] const BSRData& asBSR(GlobalIndex block_size);

    /**
     * @brief Get as ELLPACK format
     */
    [[nodiscard]] const ELLPACKData& asELLPACK();

    /**
     * @brief Clear all cached conversions
     */
    void clearCache();

    /**
     * @brief Get source pattern
     */
    [[nodiscard]] const SparsityPattern& source() const noexcept {
        return pattern_;
    }

private:
    const SparsityPattern& pattern_;

    // Cached conversions
    std::optional<COOData> coo_cache_;
    std::optional<CSCData> csc_cache_;
    std::optional<BSRData> bsr_cache_;
    std::optional<ELLPACKData> ellpack_cache_;
    GlobalIndex bsr_block_size_{0};
};

// ============================================================================
// Format name utilities
// ============================================================================

/**
 * @brief Get string name for format
 */
[[nodiscard]] inline const char* formatName(SparseFormat format) noexcept {
    switch (format) {
        case SparseFormat::CSR: return "CSR";
        case SparseFormat::CSC: return "CSC";
        case SparseFormat::COO: return "COO";
        case SparseFormat::BSR: return "BSR";
        case SparseFormat::ELLPACK: return "ELLPACK";
        case SparseFormat::DIA: return "DIA";
        case SparseFormat::Custom: return "Custom";
        default: return "Unknown";
    }
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_SPARSITY_FORMAT_H
