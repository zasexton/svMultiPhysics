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

#ifndef SVMP_FE_SPARSITY_SPARSITY_PREALLOCATION_H
#define SVMP_FE_SPARSITY_SPARSITY_PREALLOCATION_H

/**
 * @file SparsityPreallocation.h
 * @brief Backend-agnostic preallocation descriptors for sparse matrices
 *
 * This header provides the SparsityPreallocation class for describing
 * matrix structure in a format suitable for backend preallocation. Different
 * sparse matrix libraries have different preallocation APIs:
 *
 * - PETSc: MatSeqAIJSetPreallocation(mat, 0, nnz_per_row)
 *          MatMPIAIJSetPreallocation(mat, 0, d_nnz, 0, o_nnz)
 * - Trilinos: Tpetra::CrsGraph(rowMap, nnzPerRow)
 * - Eigen: mat.reserve(nnz_per_row) or mat.reserve(VectorXi)
 *
 * SparsityPreallocation provides the data needed for all these formats
 * in a backend-agnostic way.
 *
 * Key features:
 * - Per-row NNZ arrays (diagNnzPerRow, offdiagNnzPerRow)
 * - Scalar preallocation info (maxRowNnz, totalNnz)
 * - Block preallocation support (for block matrices)
 * - Computed from SparsityPattern or DistributedSparsityPattern
 * - Memory-efficient storage
 *
 * @see SparsityPattern for the pattern representation
 * @see DistributedSparsityPattern for distributed patterns
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <optional>
#include <cstdint>

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Preallocation strategy for unknown patterns
 */
enum class PreallocationStrategy : std::uint8_t {
    Exact,           ///< Use exact NNZ from pattern (best performance)
    Overestimate,    ///< Multiply by safety factor (avoid reallocation)
    Uniform,         ///< Use uniform NNZ per row (simplest)
    Adaptive         ///< Use statistics-based heuristics
};

/**
 * @brief Block preallocation information
 *
 * For block matrices (BSR format), preallocation is specified
 * per block rather than per entry.
 */
struct BlockPreallocationInfo {
    GlobalIndex block_rows{1};         ///< Block row size
    GlobalIndex block_cols{1};         ///< Block column size
    std::vector<GlobalIndex> blocks_per_row; ///< Number of blocks per row
    GlobalIndex max_blocks_per_row{0}; ///< Maximum blocks in any row
    GlobalIndex total_blocks{0};       ///< Total number of blocks
};

/**
 * @brief Backend-agnostic preallocation descriptor
 *
 * SparsityPreallocation encapsulates all the preallocation information
 * needed to efficiently create sparse matrices in various backends.
 * It can be computed from a SparsityPattern or DistributedSparsityPattern.
 *
 * Usage:
 * @code
 * // From pattern
 * SparsityPattern pattern = builder.build();
 * SparsityPreallocation prealloc(pattern);
 *
 * // For PETSc sequential
 * MatSeqAIJSetPreallocation(mat, 0, prealloc.getNnzPerRowData());
 *
 * // For PETSc MPI
 * DistributedSparsityPreallocation dist_prealloc(dist_pattern);
 * MatMPIAIJSetPreallocation(mat, 0, dist_prealloc.getDiagNnzData(),
 *                           0, dist_prealloc.getOffdiagNnzData());
 *
 * // For Trilinos
 * auto nnz = prealloc.getNnzPerRowVector<size_t>();
 * Tpetra::CrsGraph graph(rowMap, Teuchos::arcp(nnz.data(), 0, nnz.size()));
 *
 * // For Eigen
 * Eigen::SparseMatrix<double> mat(n, n);
 * mat.reserve(prealloc.getNnzPerRowVector<int>());
 * @endcode
 */
class SparsityPreallocation {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor - empty preallocation
     */
    SparsityPreallocation() = default;

    /**
     * @brief Construct from SparsityPattern
     *
     * @param pattern Finalized sparsity pattern
     */
    explicit SparsityPreallocation(const SparsityPattern& pattern);

    /**
     * @brief Construct with uniform NNZ per row
     *
     * @param n_rows Number of rows
     * @param nnz_per_row Uniform NNZ for all rows
     */
    SparsityPreallocation(GlobalIndex n_rows, GlobalIndex nnz_per_row);

    /**
     * @brief Construct with per-row NNZ array
     *
     * @param nnz_per_row NNZ for each row
     */
    explicit SparsityPreallocation(std::vector<GlobalIndex> nnz_per_row);

    /// Copy constructor
    SparsityPreallocation(const SparsityPreallocation&) = default;

    /// Copy assignment
    SparsityPreallocation& operator=(const SparsityPreallocation&) = default;

    /// Move constructor
    SparsityPreallocation(SparsityPreallocation&&) noexcept = default;

    /// Move assignment
    SparsityPreallocation& operator=(SparsityPreallocation&&) noexcept = default;

    /// Destructor
    ~SparsityPreallocation() = default;

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept { return n_rows_; }

    /**
     * @brief Get number of columns (if known)
     */
    [[nodiscard]] GlobalIndex numCols() const noexcept { return n_cols_; }

    /**
     * @brief Get total NNZ
     */
    [[nodiscard]] GlobalIndex totalNnz() const noexcept { return total_nnz_; }

    /**
     * @brief Get maximum NNZ in any row
     */
    [[nodiscard]] GlobalIndex maxRowNnz() const noexcept { return max_row_nnz_; }

    /**
     * @brief Get minimum NNZ in any row
     */
    [[nodiscard]] GlobalIndex minRowNnz() const noexcept { return min_row_nnz_; }

    /**
     * @brief Get average NNZ per row
     */
    [[nodiscard]] double avgRowNnz() const noexcept {
        return n_rows_ > 0 ? static_cast<double>(total_nnz_) / static_cast<double>(n_rows_) : 0.0;
    }

    /**
     * @brief Get NNZ for a specific row
     *
     * @param row Row index
     * @return NNZ for that row
     */
    [[nodiscard]] GlobalIndex getRowNnz(GlobalIndex row) const;

    /**
     * @brief Get NNZ per row as span
     *
     * @return Span of NNZ values indexed by row
     */
    [[nodiscard]] std::span<const GlobalIndex> getNnzPerRow() const noexcept {
        return std::span<const GlobalIndex>(nnz_per_row_.data(), nnz_per_row_.size());
    }

    /**
     * @brief Get raw pointer to NNZ per row array
     *
     * Suitable for C-style APIs like PETSc.
     */
    [[nodiscard]] const GlobalIndex* getNnzPerRowData() const noexcept {
        return nnz_per_row_.data();
    }

    /**
     * @brief Get NNZ per row as vector of specific type
     *
     * Useful for backends that require specific integer types.
     *
     * @tparam T Target integer type
     * @return Vector of NNZ values
     */
    template<typename T>
    [[nodiscard]] std::vector<T> getNnzPerRowVector() const {
        std::vector<T> result(nnz_per_row_.size());
        for (std::size_t i = 0; i < nnz_per_row_.size(); ++i) {
            result[i] = static_cast<T>(nnz_per_row_[i]);
        }
        return result;
    }

    /**
     * @brief Check if preallocation info is uniform
     *
     * @return true if all rows have the same NNZ
     */
    [[nodiscard]] bool isUniform() const noexcept {
        return min_row_nnz_ == max_row_nnz_;
    }

    /**
     * @brief Check if preallocation data is empty
     */
    [[nodiscard]] bool empty() const noexcept {
        return n_rows_ == 0;
    }

    // =========================================================================
    // Modification methods
    // =========================================================================

    /**
     * @brief Apply safety factor to preallocation
     *
     * Multiplies all NNZ values by factor, useful for avoiding
     * reallocation during assembly.
     *
     * @param factor Safety factor (>= 1.0)
     * @return Reference to this for chaining
     */
    SparsityPreallocation& applySafetyFactor(double factor);

    /**
     * @brief Add extra entries to each row
     *
     * @param extra_per_row Additional entries per row
     * @return Reference to this for chaining
     */
    SparsityPreallocation& addExtraPerRow(GlobalIndex extra_per_row);

    /**
     * @brief Clamp NNZ to maximum value
     *
     * @param max_nnz Maximum allowed NNZ per row
     * @return Reference to this for chaining
     */
    SparsityPreallocation& clampToMax(GlobalIndex max_nnz);

    /**
     * @brief Ensure minimum NNZ per row
     *
     * @param min_nnz Minimum NNZ per row
     * @return Reference to this for chaining
     */
    SparsityPreallocation& ensureMinimum(GlobalIndex min_nnz);

    // =========================================================================
    // Combination methods
    // =========================================================================

    /**
     * @brief Combine with another preallocation (element-wise max)
     *
     * @param other Other preallocation
     * @return Combined preallocation
     */
    [[nodiscard]] SparsityPreallocation combine(const SparsityPreallocation& other) const;

    /**
     * @brief Add another preallocation (element-wise sum)
     *
     * @param other Other preallocation
     * @return Combined preallocation
     */
    [[nodiscard]] SparsityPreallocation add(const SparsityPreallocation& other) const;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate preallocation data
     *
     * @return true if valid
     */
    [[nodiscard]] bool validate() const noexcept;

    /**
     * @brief Memory usage in bytes
     */
    [[nodiscard]] std::size_t memoryUsageBytes() const noexcept;

private:
    void computeStatistics();

    GlobalIndex n_rows_{0};
    GlobalIndex n_cols_{0};
    GlobalIndex total_nnz_{0};
    GlobalIndex max_row_nnz_{0};
    GlobalIndex min_row_nnz_{0};
    std::vector<GlobalIndex> nnz_per_row_;
};

/**
 * @brief Distributed preallocation descriptor with diag/offdiag split
 *
 * For MPI-parallel matrices, preallocation is split into diagonal
 * (local) and off-diagonal (remote) parts.
 */
class DistributedSparsityPreallocation {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    DistributedSparsityPreallocation() = default;

    /**
     * @brief Construct from DistributedSparsityPattern
     *
     * @param pattern Finalized distributed sparsity pattern
     */
    explicit DistributedSparsityPreallocation(const DistributedSparsityPattern& pattern);

    /**
     * @brief Construct with uniform diag/offdiag NNZ per row
     *
     * @param n_owned_rows Number of owned rows
     * @param diag_nnz_per_row Uniform diagonal NNZ
     * @param offdiag_nnz_per_row Uniform off-diagonal NNZ
     */
    DistributedSparsityPreallocation(GlobalIndex n_owned_rows,
                                      GlobalIndex diag_nnz_per_row,
                                      GlobalIndex offdiag_nnz_per_row);

    /**
     * @brief Construct with per-row arrays
     *
     * @param diag_nnz_per_row Diagonal NNZ per row
     * @param offdiag_nnz_per_row Off-diagonal NNZ per row
     */
    DistributedSparsityPreallocation(std::vector<GlobalIndex> diag_nnz_per_row,
                                      std::vector<GlobalIndex> offdiag_nnz_per_row);

    /// Copy constructor
    DistributedSparsityPreallocation(const DistributedSparsityPreallocation&) = default;

    /// Copy assignment
    DistributedSparsityPreallocation& operator=(const DistributedSparsityPreallocation&) = default;

    /// Move constructor
    DistributedSparsityPreallocation(DistributedSparsityPreallocation&&) noexcept = default;

    /// Move assignment
    DistributedSparsityPreallocation& operator=(DistributedSparsityPreallocation&&) noexcept = default;

    /// Destructor
    ~DistributedSparsityPreallocation() = default;

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Get number of owned rows
     */
    [[nodiscard]] GlobalIndex numOwnedRows() const noexcept { return n_owned_rows_; }

    /**
     * @brief Get number of owned columns
     */
    [[nodiscard]] GlobalIndex numOwnedCols() const noexcept { return n_owned_cols_; }

    /**
     * @brief Get number of ghost columns
     */
    [[nodiscard]] GlobalIndex numGhostCols() const noexcept { return n_ghost_cols_; }

    // --- Diagonal block queries ---

    /**
     * @brief Get total diagonal NNZ
     */
    [[nodiscard]] GlobalIndex totalDiagNnz() const noexcept { return total_diag_nnz_; }

    /**
     * @brief Get maximum diagonal NNZ in any row
     */
    [[nodiscard]] GlobalIndex maxDiagRowNnz() const noexcept { return max_diag_row_nnz_; }

    /**
     * @brief Get diagonal NNZ for a row
     */
    [[nodiscard]] GlobalIndex getDiagRowNnz(GlobalIndex local_row) const;

    /**
     * @brief Get diagonal NNZ per row as span
     */
    [[nodiscard]] std::span<const GlobalIndex> getDiagNnzPerRow() const noexcept {
        return std::span<const GlobalIndex>(diag_nnz_per_row_.data(), diag_nnz_per_row_.size());
    }

    /**
     * @brief Get raw pointer to diagonal NNZ array (for PETSc d_nnz)
     */
    [[nodiscard]] const GlobalIndex* getDiagNnzData() const noexcept {
        return diag_nnz_per_row_.data();
    }

    // --- Off-diagonal block queries ---

    /**
     * @brief Get total off-diagonal NNZ
     */
    [[nodiscard]] GlobalIndex totalOffdiagNnz() const noexcept { return total_offdiag_nnz_; }

    /**
     * @brief Get maximum off-diagonal NNZ in any row
     */
    [[nodiscard]] GlobalIndex maxOffdiagRowNnz() const noexcept { return max_offdiag_row_nnz_; }

    /**
     * @brief Get off-diagonal NNZ for a row
     */
    [[nodiscard]] GlobalIndex getOffdiagRowNnz(GlobalIndex local_row) const;

    /**
     * @brief Get off-diagonal NNZ per row as span
     */
    [[nodiscard]] std::span<const GlobalIndex> getOffdiagNnzPerRow() const noexcept {
        return std::span<const GlobalIndex>(offdiag_nnz_per_row_.data(), offdiag_nnz_per_row_.size());
    }

    /**
     * @brief Get raw pointer to off-diagonal NNZ array (for PETSc o_nnz)
     */
    [[nodiscard]] const GlobalIndex* getOffdiagNnzData() const noexcept {
        return offdiag_nnz_per_row_.data();
    }

    // --- Combined queries ---

    /**
     * @brief Get total local NNZ (diag + offdiag)
     */
    [[nodiscard]] GlobalIndex totalLocalNnz() const noexcept {
        return total_diag_nnz_ + total_offdiag_nnz_;
    }

    /**
     * @brief Get total NNZ for a row (diag + offdiag)
     */
    [[nodiscard]] GlobalIndex getRowNnz(GlobalIndex local_row) const;

    /**
     * @brief Get combined preallocation (diag + offdiag per row)
     */
    [[nodiscard]] SparsityPreallocation getCombinedPreallocation() const;

    /**
     * @brief Get diagonal-only preallocation
     */
    [[nodiscard]] SparsityPreallocation getDiagPreallocation() const;

    /**
     * @brief Get off-diagonal-only preallocation
     */
    [[nodiscard]] SparsityPreallocation getOffdiagPreallocation() const;

    // =========================================================================
    // Template accessors for backend-specific types
    // =========================================================================

    /**
     * @brief Get diagonal NNZ as vector of specific type
     */
    template<typename T>
    [[nodiscard]] std::vector<T> getDiagNnzVector() const {
        std::vector<T> result(diag_nnz_per_row_.size());
        for (std::size_t i = 0; i < diag_nnz_per_row_.size(); ++i) {
            result[i] = static_cast<T>(diag_nnz_per_row_[i]);
        }
        return result;
    }

    /**
     * @brief Get off-diagonal NNZ as vector of specific type
     */
    template<typename T>
    [[nodiscard]] std::vector<T> getOffdiagNnzVector() const {
        std::vector<T> result(offdiag_nnz_per_row_.size());
        for (std::size_t i = 0; i < offdiag_nnz_per_row_.size(); ++i) {
            result[i] = static_cast<T>(offdiag_nnz_per_row_[i]);
        }
        return result;
    }

    // =========================================================================
    // Modification methods
    // =========================================================================

    /**
     * @brief Apply safety factor to both diag and offdiag
     *
     * @param factor Safety factor
     * @return Reference to this
     */
    DistributedSparsityPreallocation& applySafetyFactor(double factor);

    /**
     * @brief Apply separate safety factors
     *
     * @param diag_factor Safety factor for diagonal
     * @param offdiag_factor Safety factor for off-diagonal
     * @return Reference to this
     */
    DistributedSparsityPreallocation& applySafetyFactors(double diag_factor,
                                                          double offdiag_factor);

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate preallocation data
     */
    [[nodiscard]] bool validate() const noexcept;

    /**
     * @brief Check if empty
     */
    [[nodiscard]] bool empty() const noexcept {
        return n_owned_rows_ == 0;
    }

    /**
     * @brief Memory usage in bytes
     */
    [[nodiscard]] std::size_t memoryUsageBytes() const noexcept;

private:
    void computeStatistics();

    GlobalIndex n_owned_rows_{0};
    GlobalIndex n_owned_cols_{0};
    GlobalIndex n_ghost_cols_{0};

    // Diagonal block
    std::vector<GlobalIndex> diag_nnz_per_row_;
    GlobalIndex total_diag_nnz_{0};
    GlobalIndex max_diag_row_nnz_{0};

    // Off-diagonal block
    std::vector<GlobalIndex> offdiag_nnz_per_row_;
    GlobalIndex total_offdiag_nnz_{0};
    GlobalIndex max_offdiag_row_nnz_{0};
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Create preallocation from pattern (convenience)
 *
 * @param pattern Finalized sparsity pattern
 * @return Preallocation descriptor
 */
[[nodiscard]] inline SparsityPreallocation createPreallocation(
    const SparsityPattern& pattern)
{
    return SparsityPreallocation(pattern);
}

/**
 * @brief Create distributed preallocation from pattern (convenience)
 *
 * @param pattern Finalized distributed sparsity pattern
 * @return Distributed preallocation descriptor
 */
[[nodiscard]] inline DistributedSparsityPreallocation createPreallocation(
    const DistributedSparsityPattern& pattern)
{
    return DistributedSparsityPreallocation(pattern);
}

/**
 * @brief Create uniform preallocation (convenience)
 *
 * @param n_rows Number of rows
 * @param nnz_per_row Uniform NNZ per row
 * @return Preallocation descriptor
 */
[[nodiscard]] inline SparsityPreallocation uniformPreallocation(
    GlobalIndex n_rows, GlobalIndex nnz_per_row)
{
    return SparsityPreallocation(n_rows, nnz_per_row);
}

/**
 * @brief Estimate preallocation from element data
 *
 * @param n_dofs Number of DOFs
 * @param n_elements Number of elements
 * @param dofs_per_element DOFs per element
 * @param avg_elements_per_dof Average elements sharing a DOF
 * @return Estimated preallocation
 */
[[nodiscard]] SparsityPreallocation estimatePreallocation(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    GlobalIndex dofs_per_element,
    double avg_elements_per_dof = 4.0);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_SPARSITY_PREALLOCATION_H
