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

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLY_CONSTRAINT_DISTRIBUTOR_H
#define SVMP_FE_ASSEMBLY_ASSEMBLY_CONSTRAINT_DISTRIBUTOR_H

/**
 * @file AssemblyConstraintDistributor.h
 * @brief Assembly-time constraint handling wrapper
 *
 * AssemblyConstraintDistributor is a thin adapter that bridges the Assembly
 * module with the Constraints module. It provides assembly-friendly methods
 * for distributing element contributions while respecting DOF constraints.
 *
 * Module boundary (CRITICAL):
 * - This wrapper is OWNED by Assembly
 * - All constraint algebra is DELEGATED to FE/Constraints/ConstraintDistributor
 * - Assembly only provides the orchestration interface
 *
 * Key responsibilities:
 * - Adapting GlobalSystemView to IMatrixOperations/IVectorOperations interfaces
 * - Providing convenient assembly-time distribution methods
 * - Managing "lifting" of inhomogeneous Dirichlet BCs into RHS
 * - Zeroing constrained rows/columns with diagonal treatment
 *
 * Design rationale:
 * - Assembly needs constraint-aware distribution during element loops
 * - Constraints module owns all constraint algebra and definitions
 * - This wrapper avoids Assembly depending on Constraints internals
 *
 * @see FE/Constraints/ConstraintDistributor for the actual implementation
 * @see FE/Constraints/AffineConstraints for constraint storage
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "GlobalSystemView.h"
#include "Constraints/ConstraintDistributor.h"  // For IMatrixOperations and IVectorOperations

#include <vector>
#include <span>
#include <memory>

namespace svmp {
namespace FE {

// Forward declarations
namespace constraints {
    class AffineConstraints;
}

namespace assembly {

// ============================================================================
// GlobalSystemView Adapters for Constraints Module
// ============================================================================

/**
 * @brief Adapter from GlobalSystemView to IMatrixOperations
 *
 * Allows the Constraints module's ConstraintDistributor to operate on
 * GlobalSystemView objects without direct dependency.
 */
class GlobalSystemMatrixAdapter : public constraints::IMatrixOperations {
public:
    /**
     * @brief Construct adapter wrapping a GlobalSystemView
     */
    explicit GlobalSystemMatrixAdapter(GlobalSystemView& view);

    ~GlobalSystemMatrixAdapter() override = default;

    void addValues(std::span<const GlobalIndex> rows,
                   std::span<const GlobalIndex> cols,
                   std::span<const double> values) override;

    void addValue(GlobalIndex row, GlobalIndex col, double value) override;

    void setDiagonal(GlobalIndex row, double value) override;

    [[nodiscard]] GlobalIndex numRows() const override;
    [[nodiscard]] GlobalIndex numCols() const override;

private:
    GlobalSystemView& view_;
};

/**
 * @brief Adapter from GlobalSystemView to IVectorOperations
 */
class GlobalSystemVectorAdapter : public constraints::IVectorOperations {
public:
    /**
     * @brief Construct adapter wrapping a GlobalSystemView
     */
    explicit GlobalSystemVectorAdapter(GlobalSystemView& view);

    ~GlobalSystemVectorAdapter() override = default;

    void addValues(std::span<const GlobalIndex> indices,
                   std::span<const double> values) override;

    void addValue(GlobalIndex index, double value) override;

    void setValue(GlobalIndex index, double value) override;

    [[nodiscard]] double getValue(GlobalIndex index) const override;

    [[nodiscard]] GlobalIndex size() const override;

private:
    GlobalSystemView& view_;
};

// ============================================================================
// Assembly Constraint Distributor Options
// ============================================================================

/**
 * @brief Options for assembly-time constraint distribution
 */
struct AssemblyConstraintOptions {
    /**
     * @brief Apply constraints during assembly (default: true)
     *
     * If false, constraints are not applied and element matrices/vectors
     * are inserted directly. Constraints must be applied post-assembly.
     */
    bool apply_constraints{true};

    /**
     * @brief Use symmetric elimination (default: true)
     *
     * Symmetric elimination modifies both rows and columns for constrained DOFs,
     * preserving matrix symmetry for symmetric problems.
     */
    bool symmetric_elimination{true};

    /**
     * @brief Value to set on diagonal for constrained rows (default: 1.0)
     *
     * After distributing constrained rows, the diagonal entry is set to this
     * value to ensure the matrix remains non-singular.
     */
    double constrained_diagonal{1.0};

    /**
     * @brief Apply inhomogeneity "lifting" to RHS (default: true)
     *
     * For constraints u_s = sum(a_i * u_m_i) + b, the inhomogeneity b
     * contributes to the RHS. This option enables that contribution.
     */
    bool apply_inhomogeneities{true};

    /**
     * @brief Skip distribution for unconstrained DOFs (optimization)
     *
     * If true, checks if element has any constrained DOFs and skips the
     * full distribution logic if not. Improves performance for meshes
     * with few constraints.
     */
    bool skip_unconstrained{true};

    /**
     * @brief Zero tolerance for skipping small values
     */
    double zero_tolerance{1e-15};
};

// ============================================================================
// Assembly Constraint Distributor
// ============================================================================

/**
 * @brief Assembly-time constraint distribution wrapper
 *
 * AssemblyConstraintDistributor provides the assembly module's interface to
 * constraint-aware distribution. It wraps the Constraints module's
 * ConstraintDistributor and adapts it for use with GlobalSystemView.
 *
 * Usage:
 * @code
 *   // Setup
 *   AffineConstraints constraints;
 *   // ... add constraints, then close() ...
 *
 *   AssemblyConstraintDistributor distributor(constraints);
 *
 *   // During assembly
 *   DenseSystemView system(n_dofs);
 *   system.beginAssemblyPhase();
 *
 *   for (auto cell : cells) {
 *       auto dofs = getDofs(cell);
 *       auto local_mat = computeLocalMatrix(cell);
 *       auto local_rhs = computeLocalRhs(cell);
 *
 *       distributor.distributeLocalToGlobal(
 *           local_mat, local_rhs, dofs, system, system);
 *   }
 *
 *   // Finalize constrained rows
 *   distributor.finalizeConstrainedRows(system);
 *   system.endAssemblyPhase();
 * @endcode
 *
 * Thread safety:
 * - Multiple threads can call distributeLocalToGlobal concurrently
 *   if the GlobalSystemView is thread-safe
 * - The distributor maintains thread-local scratch buffers internally
 */
class AssemblyConstraintDistributor {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (requires setConstraints)
     */
    AssemblyConstraintDistributor();

    /**
     * @brief Construct with constraints
     *
     * @param constraints Closed AffineConstraints object
     */
    explicit AssemblyConstraintDistributor(const constraints::AffineConstraints& constraints);

    /**
     * @brief Construct with constraints and options
     */
    AssemblyConstraintDistributor(const constraints::AffineConstraints& constraints,
                                  const AssemblyConstraintOptions& options);

    /**
     * @brief Destructor
     */
    ~AssemblyConstraintDistributor();

    /**
     * @brief Move constructor
     */
    AssemblyConstraintDistributor(AssemblyConstraintDistributor&& other) noexcept;

    /**
     * @brief Move assignment
     */
    AssemblyConstraintDistributor& operator=(AssemblyConstraintDistributor&& other) noexcept;

    // Non-copyable
    AssemblyConstraintDistributor(const AssemblyConstraintDistributor&) = delete;
    AssemblyConstraintDistributor& operator=(const AssemblyConstraintDistributor&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the constraints object
     *
     * @param constraints Closed AffineConstraints
     * @throws FEException if constraints not closed
     */
    void setConstraints(const constraints::AffineConstraints& constraints);

    /**
     * @brief Set distribution options
     */
    void setOptions(const AssemblyConstraintOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const AssemblyConstraintOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Check if constraints are set
     */
    [[nodiscard]] bool hasConstraints() const noexcept {
        return constraints_ != nullptr;
    }

    // =========================================================================
    // Element Distribution
    // =========================================================================

    /**
     * @brief Distribute element matrix and vector to global system
     *
     * This is the primary assembly-time distribution method. It handles:
     * - Expanding constrained DOF contributions to master DOFs
     * - Applying inhomogeneity contributions to RHS
     * - Optionally using symmetric elimination
     *
     * @param cell_matrix Local element matrix (row-major, n_dofs x n_dofs)
     * @param cell_rhs Local element RHS vector (n_dofs)
     * @param cell_dofs Global DOF indices for the element
     * @param matrix_view Global matrix to insert into
     * @param rhs_view Global RHS vector to insert into
     */
    void distributeLocalToGlobal(
        std::span<const Real> cell_matrix,
        std::span<const Real> cell_rhs,
        std::span<const GlobalIndex> cell_dofs,
        GlobalSystemView& matrix_view,
        GlobalSystemView& rhs_view);

    /**
     * @brief Distribute element matrix only
     *
     * @param cell_matrix Local element matrix
     * @param cell_dofs Global DOF indices
     * @param matrix_view Global matrix
     */
    void distributeMatrixToGlobal(
        std::span<const Real> cell_matrix,
        std::span<const GlobalIndex> cell_dofs,
        GlobalSystemView& matrix_view);

    /**
     * @brief Distribute element vector only
     *
     * @param cell_rhs Local element RHS
     * @param cell_dofs Global DOF indices
     * @param rhs_view Global RHS vector
     */
    void distributeVectorToGlobal(
        std::span<const Real> cell_rhs,
        std::span<const GlobalIndex> cell_dofs,
        GlobalSystemView& rhs_view);

    /**
     * @brief Distribute with rectangular assembly (different test/trial spaces)
     *
     * @param cell_matrix Local element matrix (n_test x n_trial)
     * @param cell_rhs Local element RHS (n_test)
     * @param row_dofs Test space DOF indices
     * @param col_dofs Trial space DOF indices
     * @param matrix_view Global matrix
     * @param rhs_view Global RHS vector
     */
    void distributeLocalToGlobal(
        std::span<const Real> cell_matrix,
        std::span<const Real> cell_rhs,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView& matrix_view,
        GlobalSystemView& rhs_view);

    // =========================================================================
    // Finalization
    // =========================================================================

    /**
     * @brief Finalize constrained rows in matrix
     *
     * Sets diagonal entries for constrained rows to ensure non-singularity.
     * Call this after all element distributions are complete.
     *
     * @param matrix_view Global matrix
     */
    void finalizeConstrainedRows(GlobalSystemView& matrix_view);

    /**
     * @brief Set constrained DOF values in RHS to inhomogeneities
     *
     * Sets RHS[s] = inhomogeneity for each constrained DOF s.
     *
     * @param rhs_view Global RHS vector
     */
    void setConstrainedRhsValues(GlobalSystemView& rhs_view);

    /**
     * @brief Zero constrained entries in a vector
     *
     * @param vec_view Vector to modify
     */
    void zeroConstrainedEntries(GlobalSystemView& vec_view);

    // =========================================================================
    // Solution Post-Processing
    // =========================================================================

    /**
     * @brief Distribute solution to enforce constraints
     *
     * After solving the system, call this to set constrained DOF values
     * based on their master DOFs and inhomogeneities.
     *
     * @param solution_view Solution vector (modified in-place)
     */
    void distributeSolution(GlobalSystemView& solution_view);

    /**
     * @brief Distribute solution to a raw array
     *
     * @param solution Solution array
     * @param size Size of solution array
     */
    void distributeSolution(Real* solution, GlobalIndex size);

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Check if any DOF in a set is constrained
     *
     * Useful for optimization - skip constraint logic if no DOFs are constrained.
     *
     * @param dofs DOF indices to check
     * @return true if at least one DOF is constrained
     */
    [[nodiscard]] bool hasConstrainedDofs(std::span<const GlobalIndex> dofs) const;

    /**
     * @brief Check if a single DOF is constrained
     */
    [[nodiscard]] bool isConstrained(GlobalIndex dof) const;

    /**
     * @brief Get all constrained DOF indices
     */
    [[nodiscard]] std::vector<GlobalIndex> getConstrainedDofs() const;

    /**
     * @brief Get number of constraints
     */
    [[nodiscard]] std::size_t numConstraints() const;

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Core distribution with adapters
     */
    void distributeCore(
        std::span<const Real> cell_matrix,
        std::span<const Real> cell_rhs,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemMatrixAdapter* matrix_adapter,
        GlobalSystemVectorAdapter* rhs_adapter);

    /**
     * @brief Direct distribution without constraint handling
     */
    void distributeDirectly(
        std::span<const Real> cell_matrix,
        std::span<const Real> cell_rhs,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView& matrix_view,
        GlobalSystemView& rhs_view);

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    const constraints::AffineConstraints* constraints_{nullptr};
    AssemblyConstraintOptions options_;

    // Underlying constraint distributor from Constraints module
    std::unique_ptr<constraints::ConstraintDistributor> distributor_;

    // Thread-local scratch space (mutable for const methods)
    mutable std::vector<GlobalIndex> scratch_rows_;
    mutable std::vector<GlobalIndex> scratch_cols_;
    mutable std::vector<Real> scratch_matrix_;
    mutable std::vector<Real> scratch_rhs_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Create an assembly constraint distributor
 *
 * @param constraints Closed AffineConstraints
 * @return Unique pointer to distributor
 */
std::unique_ptr<AssemblyConstraintDistributor> createAssemblyConstraintDistributor(
    const constraints::AffineConstraints& constraints);

/**
 * @brief Create with options
 */
std::unique_ptr<AssemblyConstraintDistributor> createAssemblyConstraintDistributor(
    const constraints::AffineConstraints& constraints,
    const AssemblyConstraintOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLY_CONSTRAINT_DISTRIBUTOR_H
