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

#ifndef SVMP_FE_ASSEMBLY_GLOBAL_SYSTEM_VIEW_H
#define SVMP_FE_ASSEMBLY_GLOBAL_SYSTEM_VIEW_H

/**
 * @file GlobalSystemView.h
 * @brief Backend-neutral local-to-global insertion adapter
 *
 * GlobalSystemView provides an abstract interface for inserting local element
 * contributions into global matrices and vectors. This abstraction allows the
 * assembly code to be independent of the specific backend (PETSc, Trilinos,
 * Eigen, etc.).
 *
 * Key features:
 * - ADD vs INSERT modes for different assembly scenarios
 * - Support for both matrix and vector insertion
 * - Rectangular matrices (independent row and col DOF lists)
 * - Two-phase assembly (begin/end for collective operations)
 * - Backend-specific implementations via inheritance
 *
 * Design inspired by:
 * - PETSc MatSetValues with ADD_VALUES/INSERT_VALUES
 * - Trilinos Tpetra::CrsMatrix insertGlobalValues/sumIntoGlobalValues
 * - deal.II ConstraintMatrix distribute_local_to_global
 *
 * Module boundary:
 * - This module OWNS: insertion mechanics, add/insert modes
 * - This module does NOT OWN: actual matrix storage (that's Backends)
 *
 * @see Assembler for the orchestration interface
 * @see constraints::ConstraintDistributor for constraint-aware distribution
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include <span>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include <functional>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// Assembly Add Mode
// ============================================================================

/**
 * @brief Mode for adding values to the global system
 *
 * Controls whether values are added to existing entries or replace them.
 * Inspired by PETSc InsertMode (ADD_VALUES vs INSERT_VALUES).
 */
enum class AddMode : std::uint8_t {
    Add,       ///< Add to existing value (default for assembly)
    Insert,    ///< Replace existing value
    Max,       ///< Take maximum of existing and new value
    Min        ///< Take minimum of existing and new value
};

/**
 * @brief Convert AddMode to string for debugging
 */
inline const char* addModeToString(AddMode mode) noexcept {
    switch (mode) {
        case AddMode::Add:    return "Add";
        case AddMode::Insert: return "Insert";
        case AddMode::Max:    return "Max";
        case AddMode::Min:    return "Min";
        default:              return "Unknown";
    }
}

// ============================================================================
// Assembly Phase
// ============================================================================

/**
 * @brief Phase of the assembly process
 *
 * Some backends (especially PETSc) require explicit phase transitions.
 */
enum class AssemblyPhase : std::uint8_t {
    NotStarted,   ///< Assembly has not begun
    Building,     ///< In the middle of adding values
    Flushing,     ///< Intermediate communication (if needed)
    Finalized     ///< Assembly complete, ready for use
};

// ============================================================================
// Global System View Interface
// ============================================================================

/**
 * @brief Abstract interface for global matrix/vector insertion
 *
 * GlobalSystemView abstracts the backend-specific details of inserting
 * local element contributions into global sparse matrices and vectors.
 * Implementations wrap specific backend types (PETSc Mat/Vec, Tpetra objects,
 * Eigen sparse matrices, etc.).
 *
 * Usage pattern:
 * @code
 *   GlobalSystemView& view = ...;
 *
 *   view.beginAssemblyPhase();
 *
 *   for (auto cell : cells) {
 *       auto row_dofs = getRowDofs(cell);
 *       auto col_dofs = getColDofs(cell);
 *       auto local_matrix = computeLocalMatrix(cell);
 *
 *       view.addMatrixEntries(row_dofs, col_dofs, local_matrix);
 *   }
 *
 *   view.endAssemblyPhase();  // Triggers collective operations
 *   view.finalizeAssembly(); // Final flush
 * @endcode
 *
 * Thread safety:
 * - Implementations must document their thread-safety guarantees
 * - DenseSystemView is thread-safe for concurrent addMatrixEntries calls
 * - Backend-specific views may have different guarantees
 */
class GlobalSystemView {
public:
    virtual ~GlobalSystemView() = default;

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    /**
     * @brief Add entries to a matrix (square coupling)
     *
     * Inserts a dense local matrix into the global sparse matrix.
     * The local matrix is stored in row-major order.
     *
     * @param dofs Global DOF indices (both rows and columns)
     * @param local_matrix Dense local matrix [n_dofs x n_dofs], row-major
     * @param mode Add or insert mode
     *
     * Complexity: Depends on backend (typically O(n^2) for n DOFs)
     */
    virtual void addMatrixEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) = 0;

    /**
     * @brief Add entries to a matrix (rectangular coupling)
     *
     * Inserts a dense local matrix with independent row and column DOFs.
     * The local matrix is stored in row-major order: local_matrix[i * n_cols + j]
     * corresponds to (row_dofs[i], col_dofs[j]).
     *
     * @param row_dofs Global row DOF indices
     * @param col_dofs Global column DOF indices
     * @param local_matrix Dense local matrix [n_rows x n_cols], row-major
     * @param mode Add or insert mode
     *
     * This is the primary method for rectangular assembly and mixed operators.
     */
    virtual void addMatrixEntries(
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) = 0;

    /**
     * @brief Add a single matrix entry
     *
     * @param row Global row index
     * @param col Global column index
     * @param value Value to add/insert
     * @param mode Add or insert mode
     */
    virtual void addMatrixEntry(
        GlobalIndex row,
        GlobalIndex col,
        Real value,
        AddMode mode = AddMode::Add) = 0;

    /**
     * @brief Set diagonal entries
     *
     * @param dofs DOF indices
     * @param values Diagonal values
     */
    virtual void setDiagonal(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) = 0;

    /**
     * @brief Set a single diagonal entry
     *
     * @param dof DOF index (row = col = dof)
     * @param value Diagonal value
     */
    virtual void setDiagonal(GlobalIndex dof, Real value) = 0;

    /**
     * @brief Zero out matrix rows
     *
     * Sets all entries in the specified rows to zero.
     * Used for applying Dirichlet boundary conditions.
     *
     * @param rows Row indices to zero
     * @param set_diagonal If true, set diagonal to 1.0
     */
    virtual void zeroRows(
        std::span<const GlobalIndex> rows,
        bool set_diagonal = true) = 0;

    // =========================================================================
    // Vector Operations
    // =========================================================================

    /**
     * @brief Add entries to a vector
     *
     * @param dofs Global DOF indices
     * @param local_vector Local vector values
     * @param mode Add or insert mode
     */
    virtual void addVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_vector,
        AddMode mode = AddMode::Add) = 0;

    /**
     * @brief Add a single vector entry
     *
     * @param dof Global DOF index
     * @param value Value to add/insert
     * @param mode Add or insert mode
     */
    virtual void addVectorEntry(
        GlobalIndex dof,
        Real value,
        AddMode mode = AddMode::Add) = 0;

    /**
     * @brief Set vector entries (always insert mode)
     *
     * @param dofs Global DOF indices
     * @param values Values to set
     */
    virtual void setVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) = 0;

    /**
     * @brief Zero vector entries
     *
     * @param dofs DOF indices to zero
     */
    virtual void zeroVectorEntries(std::span<const GlobalIndex> dofs) = 0;

    /**
     * @brief Get vector value (for backends that support it)
     *
     * @param dof DOF index
     * @return Value at the DOF, or 0.0 if not supported
     */
    [[nodiscard]] virtual Real getVectorEntry(GlobalIndex dof) const {
        (void)dof;
        return 0.0;
    }

    // =========================================================================
    // Assembly Lifecycle
    // =========================================================================

    /**
     * @brief Begin an assembly phase
     *
     * Must be called before adding entries. For some backends (PETSc),
     * this sets up internal state for efficient insertion.
     */
    virtual void beginAssemblyPhase() = 0;

    /**
     * @brief End an assembly phase
     *
     * Called after adding entries for this phase. May trigger
     * intermediate communication but does not finalize.
     */
    virtual void endAssemblyPhase() = 0;

    /**
     * @brief Finalize assembly (collective operation)
     *
     * Completes all assembly operations. For distributed backends,
     * this triggers communication to exchange ghost contributions.
     *
     * After this call:
     * - Matrix/vector are ready for use with solvers
     * - Further insertions may require re-entering assembly phase
     *
     * Inspired by PETSc MatAssemblyEnd with MAT_FINAL_ASSEMBLY.
     */
    virtual void finalizeAssembly() = 0;

    /**
     * @brief Get current assembly phase
     */
    [[nodiscard]] virtual AssemblyPhase getPhase() const noexcept = 0;

    // =========================================================================
    // Properties
    // =========================================================================

    /**
     * @brief Check if this view wraps a matrix
     */
    [[nodiscard]] virtual bool hasMatrix() const noexcept = 0;

    /**
     * @brief Check if this view wraps a vector
     */
    [[nodiscard]] virtual bool hasVector() const noexcept = 0;

    /**
     * @brief Get number of rows (matrix) or size (vector)
     */
    [[nodiscard]] virtual GlobalIndex numRows() const noexcept = 0;

    /**
     * @brief Get number of columns (for matrices)
     */
    [[nodiscard]] virtual GlobalIndex numCols() const noexcept = 0;

    /**
     * @brief Get number of locally owned rows
     */
    [[nodiscard]] virtual GlobalIndex numLocalRows() const noexcept {
        return numRows();  // Default: serial (all rows local)
    }

    /**
     * @brief Check if the view is for a distributed system
     */
    [[nodiscard]] virtual bool isDistributed() const noexcept { return false; }

    /**
     * @brief Get backend name for debugging
     */
    [[nodiscard]] virtual std::string backendName() const = 0;

    /**
     * @brief Zero all entries in the matrix and/or vector
     *
     * Sets all stored values to zero. This is typically called before
     * beginning a new assembly phase.
     */
    virtual void zero() = 0;

    /**
     * @brief Get a matrix entry (if supported by the backend)
     *
     * @param row Global row index
     * @param col Global column index
     * @return Matrix value at (row, col), or 0.0 if not supported
     *
     * Note: Not all backends support efficient random access to matrix entries.
     * This method is primarily for testing and verification purposes.
     */
    [[nodiscard]] virtual Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const {
        (void)row;
        (void)col;
        return 0.0;
    }
};

// ============================================================================
// Dense Matrix/Vector View (for testing and small problems)
// ============================================================================

/**
 * @brief Dense matrix storage with GlobalSystemView interface
 *
 * This implementation stores assembled values in a dense row-major array.
 * Useful for testing, verification, and small problems.
 *
 * Thread safety: Thread-safe for concurrent addMatrixEntries calls using
 * atomic operations or mutex protection (configurable).
 */
class DenseMatrixView : public GlobalSystemView {
public:
    /**
     * @brief Construct for a square matrix
     *
     * @param n_dofs Number of DOFs (rows = cols)
     */
    explicit DenseMatrixView(GlobalIndex n_dofs);

    /**
     * @brief Construct for a rectangular matrix
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     */
    DenseMatrixView(GlobalIndex n_rows, GlobalIndex n_cols);

    ~DenseMatrixView() override = default;

    // Matrix operations
    void addMatrixEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override;

    void addMatrixEntries(
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override;

    void addMatrixEntry(
        GlobalIndex row,
        GlobalIndex col,
        Real value,
        AddMode mode = AddMode::Add) override;

    void setDiagonal(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) override;

    void setDiagonal(GlobalIndex dof, Real value) override;

    void zeroRows(
        std::span<const GlobalIndex> rows,
        bool set_diagonal = true) override;

    // Vector operations (no-op for matrix-only view)
    void addVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_vector,
        AddMode mode = AddMode::Add) override;

    void addVectorEntry(
        GlobalIndex dof,
        Real value,
        AddMode mode = AddMode::Add) override;

    void setVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) override;

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override;

    // Assembly lifecycle
    void beginAssemblyPhase() override;
    void endAssemblyPhase() override;
    void finalizeAssembly() override;
    [[nodiscard]] AssemblyPhase getPhase() const noexcept override;

    // Properties
    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return false; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return n_rows_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return n_cols_; }
    [[nodiscard]] std::string backendName() const override { return "DenseMatrix"; }

    // Zero and access
    void zero() override;
    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override;

    // Dense-specific accessors
    /**
     * @brief Get matrix entry (read-only)
     */
    [[nodiscard]] Real operator()(GlobalIndex row, GlobalIndex col) const;

    /**
     * @brief Get raw data (row-major)
     */
    [[nodiscard]] std::span<const Real> data() const noexcept { return data_; }

    /**
     * @brief Get mutable data
     */
    [[nodiscard]] std::span<Real> dataMutable() noexcept { return data_; }

    /**
     * @brief Clear matrix to zero
     */
    void clear();

    /**
     * @brief Check if matrix is symmetric (within tolerance)
     */
    [[nodiscard]] bool isSymmetric(Real tol = 1e-12) const;

private:
    GlobalIndex n_rows_;
    GlobalIndex n_cols_;
    std::vector<Real> data_;
    AssemblyPhase phase_{AssemblyPhase::NotStarted};
};

/**
 * @brief Dense vector storage with GlobalSystemView interface
 */
class DenseVectorView : public GlobalSystemView {
public:
    /**
     * @brief Construct with size
     *
     * @param size Number of DOFs
     */
    explicit DenseVectorView(GlobalIndex size);

    ~DenseVectorView() override = default;

    // Matrix operations (no-op for vector-only view)
    void addMatrixEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override;

    void addMatrixEntries(
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override;

    void addMatrixEntry(
        GlobalIndex row,
        GlobalIndex col,
        Real value,
        AddMode mode = AddMode::Add) override;

    void setDiagonal(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) override;

    void setDiagonal(GlobalIndex dof, Real value) override;

    void zeroRows(
        std::span<const GlobalIndex> rows,
        bool set_diagonal = true) override;

    // Vector operations
    void addVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_vector,
        AddMode mode = AddMode::Add) override;

    void addVectorEntry(
        GlobalIndex dof,
        Real value,
        AddMode mode = AddMode::Add) override;

    void setVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) override;

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override;

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override;

    // Assembly lifecycle
    void beginAssemblyPhase() override;
    void endAssemblyPhase() override;
    void finalizeAssembly() override;
    [[nodiscard]] AssemblyPhase getPhase() const noexcept override;

    // Properties
    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return size_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] std::string backendName() const override { return "DenseVector"; }

    // Zero
    void zero() override;

    // Dense-specific accessors
    /**
     * @brief Get vector entry
     */
    [[nodiscard]] Real operator[](GlobalIndex dof) const;

    /**
     * @brief Get raw data
     */
    [[nodiscard]] std::span<const Real> data() const noexcept { return data_; }

    /**
     * @brief Get mutable data
     */
    [[nodiscard]] std::span<Real> dataMutable() noexcept { return data_; }

    /**
     * @brief Clear vector to zero
     */
    void clear();

    /**
     * @brief Compute L2 norm
     */
    [[nodiscard]] Real norm() const;

private:
    GlobalIndex size_;
    std::vector<Real> data_;
    AssemblyPhase phase_{AssemblyPhase::NotStarted};
};

/**
 * @brief Combined matrix and vector view (for coupled assembly)
 */
class DenseSystemView : public GlobalSystemView {
public:
    /**
     * @brief Construct for a square system
     *
     * @param n_dofs Number of DOFs
     */
    explicit DenseSystemView(GlobalIndex n_dofs);

    /**
     * @brief Construct for a rectangular system
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     */
    DenseSystemView(GlobalIndex n_rows, GlobalIndex n_cols);

    ~DenseSystemView() override = default;

    // Matrix operations
    void addMatrixEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override;

    void addMatrixEntries(
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override;

    void addMatrixEntry(
        GlobalIndex row,
        GlobalIndex col,
        Real value,
        AddMode mode = AddMode::Add) override;

    void setDiagonal(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) override;

    void setDiagonal(GlobalIndex dof, Real value) override;

    void zeroRows(
        std::span<const GlobalIndex> rows,
        bool set_diagonal = true) override;

    // Vector operations
    void addVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_vector,
        AddMode mode = AddMode::Add) override;

    void addVectorEntry(
        GlobalIndex dof,
        Real value,
        AddMode mode = AddMode::Add) override;

    void setVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> values) override;

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override;

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override;

    // Assembly lifecycle
    void beginAssemblyPhase() override;
    void endAssemblyPhase() override;
    void finalizeAssembly() override;
    [[nodiscard]] AssemblyPhase getPhase() const noexcept override;

    // Properties
    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return n_rows_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return n_cols_; }
    [[nodiscard]] std::string backendName() const override { return "DenseSystem"; }

    // Zero and access
    void zero() override;
    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override;

    // Accessors
    [[nodiscard]] Real matrixEntry(GlobalIndex row, GlobalIndex col) const;
    [[nodiscard]] Real vectorEntry(GlobalIndex dof) const;
    [[nodiscard]] std::span<const Real> matrixData() const noexcept { return matrix_data_; }
    [[nodiscard]] std::span<const Real> vectorData() const noexcept { return vector_data_; }
    [[nodiscard]] std::span<Real> matrixDataMutable() noexcept { return matrix_data_; }
    [[nodiscard]] std::span<Real> vectorDataMutable() noexcept { return vector_data_; }

    /**
     * @brief Clear both matrix and vector
     */
    void clear();

private:
    GlobalIndex n_rows_;
    GlobalIndex n_cols_;
    std::vector<Real> matrix_data_;
    std::vector<Real> vector_data_;
    AssemblyPhase phase_{AssemblyPhase::NotStarted};
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create a dense matrix view
 */
std::unique_ptr<GlobalSystemView> createDenseMatrixView(GlobalIndex n_rows, GlobalIndex n_cols = -1);

/**
 * @brief Create a dense vector view
 */
std::unique_ptr<GlobalSystemView> createDenseVectorView(GlobalIndex size);

/**
 * @brief Create a dense system view (matrix + vector)
 */
std::unique_ptr<GlobalSystemView> createDenseSystemView(GlobalIndex n_rows, GlobalIndex n_cols = -1);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_GLOBAL_SYSTEM_VIEW_H
