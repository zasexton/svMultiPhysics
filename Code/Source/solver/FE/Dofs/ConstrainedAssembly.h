/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_CONSTRAINEDASSEMBLY_H
#define SVMP_FE_DOFS_CONSTRAINEDASSEMBLY_H

/**
 * @file ConstrainedAssembly.h
 * @brief Constraint-aware local-to-global distribution during assembly
 *
 * ConstrainedAssembly provides methods for distributing element contributions
 * to global matrices and vectors while honoring DOF constraints. This is
 * more efficient than assembling first and applying constraints later.
 *
 * The class uses a BackendAdapter interface to remain backend-agnostic
 * while still allowing efficient matrix/vector operations.
 */

#include "DofConstraints.h"
#include "DofMap.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Backend adapter for matrix/vector operations
 *
 * Backends (PETSc, Trilinos, etc.) implement this interface to allow
 * constraint-aware assembly without backend-specific types.
 */
class BackendAdapter {
public:
    virtual ~BackendAdapter() = default;

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    /**
     * @brief Add values to matrix (row-major dense block)
     *
     * @param rows Row indices
     * @param cols Column indices
     * @param values Dense block in row-major order [n_rows x n_cols]
     */
    virtual void addMatrixValues(std::span<const GlobalIndex> rows,
                                  std::span<const GlobalIndex> cols,
                                  std::span<const double> values) = 0;

    /**
     * @brief Add values to matrix (sparse)
     *
     * @param row_indices Row index for each value
     * @param col_indices Column index for each value
     * @param values Values to add
     */
    virtual void addMatrixValuesSparse(std::span<const GlobalIndex> row_indices,
                                        std::span<const GlobalIndex> col_indices,
                                        std::span<const double> values) = 0;

    // =========================================================================
    // Vector Operations
    // =========================================================================

    /**
     * @brief Add values to vector
     *
     * @param indices DOF indices
     * @param values Values to add
     */
    virtual void addVectorValues(std::span<const GlobalIndex> indices,
                                  std::span<const double> values) = 0;

    /**
     * @brief Set values in vector
     */
    virtual void setVectorValues(std::span<const GlobalIndex> indices,
                                  std::span<const double> values) = 0;

    // =========================================================================
    // State Management
    // =========================================================================

    /**
     * @brief Begin assembly phase
     *
     * Called before distributing element contributions.
     * May be used to set up caching or batching.
     */
    virtual void beginAssembly() {}

    /**
     * @brief End assembly phase
     *
     * Called after all element contributions are distributed.
     * May flush cached operations to backend.
     */
    virtual void endAssembly() {}
};

/**
 * @brief Options for constrained assembly
 */
struct ConstrainedAssemblyOptions {
    bool use_symmetric_elimination{true};   ///< Use symmetric elimination for constraints
    bool cache_constraint_info{true};       ///< Cache constraint lookups
    bool zero_constrained_rows{true};       ///< Zero rows for constrained DOFs
    double diagonal_value{1.0};             ///< Value for diagonal of constrained rows
};

/**
 * @brief Constraint-aware assembly manager
 *
 * Distributes element contributions to global system while respecting
 * DOF constraints, avoiding the need for post-assembly constraint application.
 */
class ConstrainedAssembly {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    ConstrainedAssembly();
    ~ConstrainedAssembly();

    // Move semantics
    ConstrainedAssembly(ConstrainedAssembly&&) noexcept;
    ConstrainedAssembly& operator=(ConstrainedAssembly&&) noexcept;

    // No copy
    ConstrainedAssembly(const ConstrainedAssembly&) = delete;
    ConstrainedAssembly& operator=(const ConstrainedAssembly&) = delete;

    /**
     * @brief Initialize with constraints and options
     *
     * @param constraints Constraint manager
     * @param options Assembly options
     */
    void initialize(const DofConstraints& constraints,
                    const ConstrainedAssemblyOptions& options = {});

    // =========================================================================
    // Single Element Distribution
    // =========================================================================

    /**
     * @brief Distribute element matrix and RHS to global system
     *
     * @param cell_matrix Element stiffness matrix (row-major, n_dofs x n_dofs)
     * @param cell_rhs Element right-hand side (n_dofs)
     * @param cell_dof_ids Global DOF indices for element DOFs
     * @param adapter Backend adapter for matrix/vector operations
     */
    void distributeLocalToGlobal(
        std::span<const double> cell_matrix,
        std::span<const double> cell_rhs,
        std::span<const GlobalIndex> cell_dof_ids,
        BackendAdapter& adapter) const;

    /**
     * @brief Distribute element matrix only
     */
    void distributeMatrixToGlobal(
        std::span<const double> cell_matrix,
        std::span<const GlobalIndex> cell_dof_ids,
        BackendAdapter& adapter) const;

    /**
     * @brief Distribute element RHS only
     */
    void distributeRhsToGlobal(
        std::span<const double> cell_rhs,
        std::span<const GlobalIndex> cell_dof_ids,
        BackendAdapter& adapter) const;

    // =========================================================================
    // Batch Distribution
    // =========================================================================

    /**
     * @brief Distribute multiple elements (batch version)
     *
     * More efficient than repeated single-element calls due to:
     * - Better cache utilization
     * - Batched backend operations
     *
     * @param n_cells Number of cells
     * @param cell_matrices Element matrices (flat, row-major per element)
     * @param cell_rhs_vectors Element RHS vectors (flat)
     * @param cell_dof_offsets CSR offsets into cell_dof_ids
     * @param cell_dof_ids All cell DOF indices (flat)
     * @param adapter Backend adapter
     */
    void distributeLocalToGlobalBatch(
        GlobalIndex n_cells,
        std::span<const double> cell_matrices,
        std::span<const double> cell_rhs_vectors,
        std::span<const GlobalIndex> cell_dof_offsets,
        std::span<const GlobalIndex> cell_dof_ids,
        BackendAdapter& adapter) const;

    /**
     * @brief Batch distribution with cell DOF counts
     *
     * Alternative interface when all cells have same DOF count.
     *
     * @param n_cells Number of cells
     * @param dofs_per_cell DOFs per cell
     * @param cell_matrices Element matrices [n_cells * dofs_per_cell^2]
     * @param cell_rhs_vectors Element RHS vectors [n_cells * dofs_per_cell]
     * @param cell_dof_ids DOF indices [n_cells * dofs_per_cell]
     * @param adapter Backend adapter
     */
    void distributeLocalToGlobalUniform(
        GlobalIndex n_cells,
        LocalIndex dofs_per_cell,
        std::span<const double> cell_matrices,
        std::span<const double> cell_rhs_vectors,
        std::span<const GlobalIndex> cell_dof_ids,
        BackendAdapter& adapter) const;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set assembly options
     */
    void setOptions(const ConstrainedAssemblyOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const ConstrainedAssemblyOptions& options() const noexcept {
        return options_;
    }

    /**
     * @brief Check if initialized
     */
    [[nodiscard]] bool isInitialized() const noexcept {
        return constraints_ != nullptr;
    }

private:
    const DofConstraints* constraints_{nullptr};
    ConstrainedAssemblyOptions options_;

    // Workspace for constraint handling
    mutable std::vector<GlobalIndex> work_rows_;
    mutable std::vector<GlobalIndex> work_cols_;
    mutable std::vector<double> work_values_;
    mutable std::vector<double> work_rhs_;

    // Apply constraints to element matrix/RHS before distribution
    void applyElementConstraints(
        std::span<const double> cell_matrix,
        std::span<const double> cell_rhs,
        std::span<const GlobalIndex> cell_dof_ids,
        std::vector<GlobalIndex>& out_rows,
        std::vector<GlobalIndex>& out_cols,
        std::vector<double>& out_values,
        std::vector<GlobalIndex>& out_rhs_indices,
        std::vector<double>& out_rhs_values) const;
};

// =============================================================================
// Simple Backend Adapters
// =============================================================================

/**
 * @brief Dense matrix adapter for testing
 *
 * Stores assembled matrix as dense array.
 */
class DenseMatrixAdapter : public BackendAdapter {
public:
    /**
     * @brief Construct dense adapter
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     */
    DenseMatrixAdapter(GlobalIndex n_rows, GlobalIndex n_cols);

    void addMatrixValues(std::span<const GlobalIndex> rows,
                          std::span<const GlobalIndex> cols,
                          std::span<const double> values) override;

    void addMatrixValuesSparse(std::span<const GlobalIndex> row_indices,
                                std::span<const GlobalIndex> col_indices,
                                std::span<const double> values) override;

    void addVectorValues(std::span<const GlobalIndex> indices,
                          std::span<const double> values) override;

    void setVectorValues(std::span<const GlobalIndex> indices,
                          std::span<const double> values) override;

    /**
     * @brief Get matrix entry
     */
    [[nodiscard]] double getMatrixEntry(GlobalIndex row, GlobalIndex col) const;

    /**
     * @brief Get vector entry
     */
    [[nodiscard]] double getVectorEntry(GlobalIndex index) const;

    /**
     * @brief Get matrix data (row-major)
     */
    [[nodiscard]] std::span<const double> getMatrixData() const noexcept {
        return matrix_;
    }

    /**
     * @brief Get vector data
     */
    [[nodiscard]] std::span<const double> getVectorData() const noexcept {
        return vector_;
    }

    [[nodiscard]] GlobalIndex numRows() const noexcept { return n_rows_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept { return n_cols_; }

private:
    GlobalIndex n_rows_;
    GlobalIndex n_cols_;
    std::vector<double> matrix_;  // Row-major
    std::vector<double> vector_;
};

/**
 * @brief CSR matrix adapter
 *
 * Assembles into CSR sparse format.
 */
class CSRMatrixAdapter : public BackendAdapter {
public:
    /**
     * @brief Construct with known sparsity pattern
     *
     * @param row_offsets CSR row offsets
     * @param col_indices CSR column indices
     */
    CSRMatrixAdapter(std::span<const GlobalIndex> row_offsets,
                      std::span<const GlobalIndex> col_indices);

    void addMatrixValues(std::span<const GlobalIndex> rows,
                          std::span<const GlobalIndex> cols,
                          std::span<const double> values) override;

    void addMatrixValuesSparse(std::span<const GlobalIndex> row_indices,
                                std::span<const GlobalIndex> col_indices,
                                std::span<const double> values) override;

    void addVectorValues(std::span<const GlobalIndex> indices,
                          std::span<const double> values) override;

    void setVectorValues(std::span<const GlobalIndex> indices,
                          std::span<const double> values) override;

    /**
     * @brief Get CSR data
     */
    [[nodiscard]] std::span<const GlobalIndex> getRowOffsets() const noexcept {
        return row_offsets_;
    }

    [[nodiscard]] std::span<const GlobalIndex> getColIndices() const noexcept {
        return col_indices_;
    }

    [[nodiscard]] std::span<const double> getValues() const noexcept {
        return values_;
    }

    [[nodiscard]] std::span<const double> getVectorData() const noexcept {
        return vector_;
    }

private:
    std::vector<GlobalIndex> row_offsets_;
    std::vector<GlobalIndex> col_indices_;
    std::vector<double> values_;
    std::vector<double> vector_;

    // Column index lookup within row
    GlobalIndex findColIndex(GlobalIndex row, GlobalIndex col) const;
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_CONSTRAINEDASSEMBLY_H
