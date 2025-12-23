/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONSTRAINTDISTRIBUTOR_H
#define SVMP_FE_CONSTRAINTS_CONSTRAINTDISTRIBUTOR_H

/**
 * @file ConstraintDistributor.h
 * @brief Constraint-aware assembly integration (local-to-global distribution)
 *
 * ConstraintDistributor provides the core functionality for assembling element
 * contributions into global matrices and vectors while respecting DOF constraints.
 * Instead of assembling first and applying constraints later (which can be
 * inefficient and may lose information), this class applies constraints during
 * distribution.
 *
 * For a constrained DOF u_s = sum(a_i * u_m_i) + b, the element matrix/RHS
 * contributions involving u_s are distributed to the master DOFs u_m_i with
 * appropriate scaling.
 *
 * Key features:
 * - Single element distribution (distributeLocalToGlobal)
 * - Batch element distribution (for vectorized assembly)
 * - Inhomogeneous constraint handling
 * - Symmetric and non-symmetric elimination modes
 * - Backend-agnostic via abstract matrix/vector interfaces
 *
 * Module boundary:
 * - This module OWNS constraint-aware distribution logic
 * - This module does NOT OWN matrix/vector storage (uses abstract interfaces)
 * - This module does NOT OWN assembly loops (caller provides element data)
 *
 * @see AffineConstraints for constraint storage
 * @see Dofs/ConstrainedAssembly for the Dofs-level counterpart
 */

#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief Abstract interface for matrix operations during constraint distribution
 *
 * Backends (PETSc, Trilinos, Eigen, dense arrays) implement this interface
 * to allow ConstraintDistributor to work without backend-specific code.
 */
class IMatrixOperations {
public:
    virtual ~IMatrixOperations() = default;

    /**
     * @brief Add values to matrix (dense block)
     *
     * @param rows Row indices
     * @param cols Column indices
     * @param values Dense block in row-major order [n_rows x n_cols]
     */
    virtual void addValues(std::span<const GlobalIndex> rows,
                           std::span<const GlobalIndex> cols,
                           std::span<const double> values) = 0;

    /**
     * @brief Add a single value to the matrix
     *
     * @param row Row index
     * @param col Column index
     * @param value Value to add
     */
    virtual void addValue(GlobalIndex row, GlobalIndex col, double value) = 0;

    /**
     * @brief Set diagonal entry to a value
     *
     * @param row Row index (row = col)
     * @param value Diagonal value
     */
    virtual void setDiagonal(GlobalIndex row, double value) = 0;

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] virtual GlobalIndex numRows() const = 0;

    /**
     * @brief Get number of columns
     */
    [[nodiscard]] virtual GlobalIndex numCols() const = 0;
};

/**
 * @brief Abstract interface for vector operations during constraint distribution
 */
class IVectorOperations {
public:
    virtual ~IVectorOperations() = default;

    /**
     * @brief Add values to vector at indices
     *
     * @param indices DOF indices
     * @param values Values to add
     */
    virtual void addValues(std::span<const GlobalIndex> indices,
                           std::span<const double> values) = 0;

    /**
     * @brief Add a single value
     *
     * @param index DOF index
     * @param value Value to add
     */
    virtual void addValue(GlobalIndex index, double value) = 0;

    /**
     * @brief Set a single value
     *
     * @param index DOF index
     * @param value Value to set
     */
    virtual void setValue(GlobalIndex index, double value) = 0;

    /**
     * @brief Get value at index
     */
    [[nodiscard]] virtual double getValue(GlobalIndex index) const = 0;

    /**
     * @brief Get size of vector
     */
    [[nodiscard]] virtual GlobalIndex size() const = 0;
};

/**
 * @brief Options for constraint distribution
 */
struct DistributorOptions {
    /**
     * @brief Use symmetric elimination (apply to both rows and columns)
     *
     * Symmetric elimination modifies both the constrained row and column,
     * preserving matrix symmetry but requiring additional work.
     */
    bool symmetric{true};

    /**
     * @brief Set diagonal of constrained rows to this value
     *
     * After distribution, constrained rows have their diagonal set to ensure
     * the matrix is non-singular. Use 1.0 for identity-like rows.
     */
    double constrained_diagonal{1.0};

    /**
     * @brief Zero tolerance for skipping near-zero contributions
     */
    double zero_tolerance{1e-15};

    /**
     * @brief Whether to apply inhomogeneity contributions to RHS
     *
     * Inhomogeneous constraints (u_s = sum + b with b != 0) contribute
     * terms to the RHS. Set false to skip this (e.g., for homogeneous problems).
     */
    bool apply_inhomogeneities{true};

    /**
     * @brief Cache constraint lookups for repeated distributions
     */
    bool use_cache{true};
};

/**
 * @brief Constraint-aware local-to-global distribution
 *
 * ConstraintDistributor handles the distribution of element matrices and
 * RHS vectors to global storage while applying DOF constraints. This is
 * more efficient and accurate than post-assembly constraint application.
 *
 * Usage:
 * @code
 *   // Setup
 *   AffineConstraints constraints;
 *   // ... add constraints ...
 *   constraints.close();
 *
 *   ConstraintDistributor distributor(constraints);
 *
 *   // During assembly loop
 *   for (auto cell : cells) {
 *       std::vector<GlobalIndex> dofs = cell.getDofs();
 *       std::vector<double> cell_matrix = computeElementMatrix(cell);
 *       std::vector<double> cell_rhs = computeElementRhs(cell);
 *
 *       distributor.distributeLocalToGlobal(
 *           cell_matrix, cell_rhs, dofs, matrix_ops, vector_ops);
 *   }
 * @endcode
 *
 * Thread safety: Multiple threads can distribute to thread-safe matrix/vector
 * backends. The distributor itself maintains thread-local scratch space.
 */
class ConstraintDistributor {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (requires setConstraints before use)
     */
    ConstraintDistributor();

    /**
     * @brief Construct with constraints
     *
     * @param constraints The closed AffineConstraints object
     */
    explicit ConstraintDistributor(const AffineConstraints& constraints);

    /**
     * @brief Construct with constraints and options
     */
    ConstraintDistributor(const AffineConstraints& constraints,
                          const DistributorOptions& options);

    /**
     * @brief Destructor
     */
    ~ConstraintDistributor();

    /**
     * @brief Move constructor
     */
    ConstraintDistributor(ConstraintDistributor&& other) noexcept;

    /**
     * @brief Move assignment
     */
    ConstraintDistributor& operator=(ConstraintDistributor&& other) noexcept;

    // Non-copyable (holds reference to constraints)
    ConstraintDistributor(const ConstraintDistributor&) = delete;
    ConstraintDistributor& operator=(const ConstraintDistributor&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the constraint object
     *
     * @param constraints The closed AffineConstraints object
     */
    void setConstraints(const AffineConstraints& constraints);

    /**
     * @brief Set distribution options
     */
    void setOptions(const DistributorOptions& options) { options_ = options; }

    /**
     * @brief Get current options
     */
    [[nodiscard]] const DistributorOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Check if constraints are set
     */
    [[nodiscard]] bool hasConstraints() const noexcept {
        return constraints_ != nullptr;
    }

    // =========================================================================
    // Single element distribution
    // =========================================================================

    /**
     * @brief Distribute element matrix and RHS to global system
     *
     * @param cell_matrix Element stiffness matrix (row-major, n_dofs x n_dofs)
     * @param cell_rhs Element right-hand side (n_dofs)
     * @param cell_dofs Global DOF indices for element DOFs
     * @param matrix Global matrix operations interface
     * @param rhs Global RHS vector operations interface
     */
    void distributeLocalToGlobal(
        std::span<const double> cell_matrix,
        std::span<const double> cell_rhs,
        std::span<const GlobalIndex> cell_dofs,
        IMatrixOperations& matrix,
        IVectorOperations& rhs) const;

    /**
     * @brief Distribute element matrix only
     *
     * @param cell_matrix Element stiffness matrix
     * @param cell_dofs Global DOF indices
     * @param matrix Global matrix operations interface
     */
    void distributeMatrixToGlobal(
        std::span<const double> cell_matrix,
        std::span<const GlobalIndex> cell_dofs,
        IMatrixOperations& matrix) const;

    /**
     * @brief Distribute element RHS only
     *
     * @param cell_rhs Element right-hand side
     * @param cell_dofs Global DOF indices
     * @param rhs Global RHS vector operations interface
     */
    void distributeRhsToGlobal(
        std::span<const double> cell_rhs,
        std::span<const GlobalIndex> cell_dofs,
        IVectorOperations& rhs) const;

    // =========================================================================
    // Batch distribution
    // =========================================================================

    /**
     * @brief Distribute multiple elements (batch version)
     *
     * @param n_cells Number of cells
     * @param cell_matrices Element matrices (flat, row-major per element)
     * @param cell_rhs_vectors Element RHS vectors (flat)
     * @param cell_dof_offsets CSR offsets into cell_dof_ids
     * @param cell_dof_ids All cell DOF indices (flat)
     * @param matrix Global matrix operations
     * @param rhs Global RHS operations
     */
    void distributeLocalToGlobalBatch(
        GlobalIndex n_cells,
        std::span<const double> cell_matrices,
        std::span<const double> cell_rhs_vectors,
        std::span<const GlobalIndex> cell_dof_offsets,
        std::span<const GlobalIndex> cell_dof_ids,
        IMatrixOperations& matrix,
        IVectorOperations& rhs) const;

    /**
     * @brief Batch distribution with uniform DOF count per cell
     *
     * @param n_cells Number of cells
     * @param dofs_per_cell DOFs per cell (all cells same)
     * @param cell_matrices Element matrices [n_cells * dofs_per_cell^2]
     * @param cell_rhs_vectors Element RHS [n_cells * dofs_per_cell]
     * @param cell_dof_ids DOF indices [n_cells * dofs_per_cell]
     * @param matrix Global matrix operations
     * @param rhs Global RHS operations
     */
    void distributeLocalToGlobalUniform(
        GlobalIndex n_cells,
        LocalIndex dofs_per_cell,
        std::span<const double> cell_matrices,
        std::span<const double> cell_rhs_vectors,
        std::span<const GlobalIndex> cell_dof_ids,
        IMatrixOperations& matrix,
        IVectorOperations& rhs) const;

    // =========================================================================
    // Direct vector operations
    // =========================================================================

    /**
     * @brief Distribute constraints to a solution vector
     *
     * Same as AffineConstraints::distribute(), provided for convenience.
     *
     * @param solution Solution vector to modify
     */
    void distributeSolution(IVectorOperations& solution) const;

    /**
     * @brief Set constrained entries in a vector
     *
     * Sets constrained DOFs to their inhomogeneity values.
     *
     * @param vec Vector to modify
     */
    void setConstrainedEntries(IVectorOperations& vec) const;

    /**
     * @brief Zero constrained entries in a vector
     *
     * Sets constrained DOFs to zero.
     *
     * @param vec Vector to modify
     */
    void zeroConstrainedEntries(IVectorOperations& vec) const;

    // =========================================================================
    // Condensation (for matrix-free or reduced systems)
    // =========================================================================

    /**
     * @brief Condense element matrix in-place
     *
     * Modifies the element matrix to account for constraints, returning
     * a reduced matrix that can be assembled without constraint handling.
     *
     * @param cell_matrix Element matrix (modified in-place)
     * @param cell_rhs Element RHS (modified in-place)
     * @param cell_dofs DOF indices
     */
    void condenseLocal(
        std::vector<double>& cell_matrix,
        std::vector<double>& cell_rhs,
        std::span<const GlobalIndex> cell_dofs) const;

private:
    // =========================================================================
    // Internal implementation
    // =========================================================================

    /**
     * @brief Core distribution logic for a single element
     */
    void distributeElementCore(
        std::span<const double> cell_matrix,
        std::span<const double> cell_rhs,
        std::span<const GlobalIndex> cell_dofs,
        IMatrixOperations* matrix,
        IVectorOperations* rhs) const;

    /**
     * @brief Check if any DOF in the set is constrained
     */
    [[nodiscard]] bool hasConstrainedDof(std::span<const GlobalIndex> dofs) const;

    /**
     * @brief Apply inhomogeneity contributions to RHS
     */
    void applyInhomogeneityToRhs(
        std::span<const double> cell_matrix,
        std::span<const GlobalIndex> cell_dofs,
        IVectorOperations& rhs) const;

    // Data members
    const AffineConstraints* constraints_{nullptr};
    DistributorOptions options_;

    // Scratch space for distribution (mutable for const methods)
    mutable std::vector<GlobalIndex> work_rows_;
    mutable std::vector<GlobalIndex> work_cols_;
    mutable std::vector<double> work_values_;
    mutable std::vector<GlobalIndex> work_rhs_indices_;
    mutable std::vector<double> work_rhs_values_;
};

// ============================================================================
// Simple implementations of matrix/vector interfaces
// ============================================================================

/**
 * @brief Dense matrix implementation of IMatrixOperations
 *
 * Stores assembled matrix as a dense row-major array.
 * Useful for testing and small problems.
 */
class DenseMatrixOps : public IMatrixOperations {
public:
    /**
     * @brief Construct with size
     */
    DenseMatrixOps(GlobalIndex n_rows, GlobalIndex n_cols);

    void addValues(std::span<const GlobalIndex> rows,
                   std::span<const GlobalIndex> cols,
                   std::span<const double> values) override;

    void addValue(GlobalIndex row, GlobalIndex col, double value) override;

    void setDiagonal(GlobalIndex row, double value) override;

    [[nodiscard]] GlobalIndex numRows() const override { return n_rows_; }
    [[nodiscard]] GlobalIndex numCols() const override { return n_cols_; }

    /**
     * @brief Get matrix entry
     */
    [[nodiscard]] double operator()(GlobalIndex row, GlobalIndex col) const {
        return data_[static_cast<std::size_t>(row * n_cols_ + col)];
    }

    /**
     * @brief Get raw data
     */
    [[nodiscard]] std::span<const double> data() const { return data_; }

    /**
     * @brief Get mutable raw data
     */
    [[nodiscard]] std::span<double> dataMutable() { return data_; }

    /**
     * @brief Clear matrix to zero
     */
    void clear();

private:
    GlobalIndex n_rows_;
    GlobalIndex n_cols_;
    std::vector<double> data_;
};

/**
 * @brief Dense vector implementation of IVectorOperations
 */
class DenseVectorOps : public IVectorOperations {
public:
    /**
     * @brief Construct with size
     */
    explicit DenseVectorOps(GlobalIndex size);

    void addValues(std::span<const GlobalIndex> indices,
                   std::span<const double> values) override;

    void addValue(GlobalIndex index, double value) override;

    void setValue(GlobalIndex index, double value) override;

    [[nodiscard]] double getValue(GlobalIndex index) const override;

    [[nodiscard]] GlobalIndex size() const override {
        return static_cast<GlobalIndex>(data_.size());
    }

    /**
     * @brief Get entry by index
     */
    [[nodiscard]] double operator[](GlobalIndex index) const {
        return data_[static_cast<std::size_t>(index)];
    }

    /**
     * @brief Get raw data
     */
    [[nodiscard]] std::span<const double> data() const { return data_; }

    /**
     * @brief Get mutable raw data
     */
    [[nodiscard]] std::span<double> dataMutable() { return data_; }

    /**
     * @brief Clear vector to zero
     */
    void clear();

private:
    std::vector<double> data_;
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONSTRAINTDISTRIBUTOR_H
