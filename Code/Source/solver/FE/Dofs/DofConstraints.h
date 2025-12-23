/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_DOFCONSTRAINTS_H
#define SVMP_FE_DOFS_DOFCONSTRAINTS_H

/**
 * @file DofConstraints.h
 * @brief DOF constraint management (Dirichlet, periodic, linear)
 *
 * The DofConstraints class manages various types of constraints on DOFs:
 *  - Dirichlet BCs: Fix DOF to a prescribed value
 *  - Periodic BCs: Relate master/slave DOF pairs
 *  - Linear constraints: General constraint u_i = sum(c_j * u_j) + inhom
 *  - Hanging node constraints: From mesh adaptivity
 *
 * Design principles:
 *  - Backend-agnostic: Uses abstract interfaces for matrix/vector operations
 *  - Constraint-aware assembly: Apply during element matrix distribution
 *  - Condensation support: Build transformation matrices for static condensation
 */

#include "DofMap.h"
#include "DofIndexSet.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <functional>
#include <memory>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Type of DOF constraint
 */
enum class ConstraintType : std::uint8_t {
    Dirichlet,      ///< Fixed value: u_i = value
    Periodic,       ///< Periodic: u_slave = u_master
    Linear,         ///< Linear: u_i = sum(c_j * u_j) + inhom
    HangingNode     ///< Hanging node from refinement
};

/**
 * @brief Single constraint entry (for linear constraints)
 */
struct ConstraintEntry {
    GlobalIndex dof;    ///< DOF index
    double coefficient; ///< Coefficient
};

/**
 * @brief A single DOF constraint (constraint line)
 *
 * Represents: constrained_dof = sum(entries[i].coeff * entries[i].dof) + inhomogeneity
 */
struct ConstraintLine {
    GlobalIndex constrained_dof;        ///< The DOF being constrained
    std::vector<ConstraintEntry> entries; ///< Dependencies (master DOFs)
    double inhomogeneity{0.0};          ///< Constant term
    ConstraintType type{ConstraintType::Linear};

    /**
     * @brief Check if this is a homogeneous constraint
     */
    [[nodiscard]] bool isHomogeneous() const noexcept {
        return std::abs(inhomogeneity) < 1e-15;
    }

    /**
     * @brief Check if this is a simple Dirichlet (fixed value) constraint
     */
    [[nodiscard]] bool isDirichlet() const noexcept {
        return type == ConstraintType::Dirichlet || entries.empty();
    }
};

/**
 * @brief Abstract interface for backend matrix operations
 *
 * Backends (PETSc, Trilinos, etc.) implement this to allow
 * constraint application without backend-specific types in Dofs.
 */
class AbstractMatrix {
public:
    virtual ~AbstractMatrix() = default;

    /**
     * @brief Add values to matrix entries
     *
     * @param rows Row indices
     * @param cols Column indices
     * @param values Values in row-major order
     */
    virtual void addValues(std::span<const GlobalIndex> rows,
                           std::span<const GlobalIndex> cols,
                           std::span<const double> values) = 0;

    /**
     * @brief Set a row to identity (for Dirichlet)
     *
     * @param row Row index
     */
    virtual void setRowToIdentity(GlobalIndex row) = 0;

    /**
     * @brief Zero a row
     */
    virtual void zeroRow(GlobalIndex row) = 0;

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] virtual GlobalIndex numRows() const = 0;
};

/**
 * @brief Abstract interface for backend vector operations
 */
class AbstractVector {
public:
    virtual ~AbstractVector() = default;

    /**
     * @brief Set values at indices
     */
    virtual void setValues(std::span<const GlobalIndex> indices,
                           std::span<const double> values) = 0;

    /**
     * @brief Add values at indices
     */
    virtual void addValues(std::span<const GlobalIndex> indices,
                           std::span<const double> values) = 0;

    /**
     * @brief Get value at index
     */
    [[nodiscard]] virtual double getValue(GlobalIndex index) const = 0;

    /**
     * @brief Get size
     */
    [[nodiscard]] virtual GlobalIndex size() const = 0;
};

/**
 * @brief DOF constraint manager
 *
 * Manages constraints on DOFs and provides methods for applying them
 * during assembly or as post-processing.
 */
class DofConstraints {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    DofConstraints();
    ~DofConstraints();

    // Move semantics
    DofConstraints(DofConstraints&&) noexcept;
    DofConstraints& operator=(DofConstraints&&) noexcept;

    // No copy
    DofConstraints(const DofConstraints&) = delete;
    DofConstraints& operator=(const DofConstraints&) = delete;

    // =========================================================================
    // Adding Constraints
    // =========================================================================

    /**
     * @brief Add Dirichlet (fixed value) constraint
     *
     * @param dof DOF index
     * @param value Prescribed value
     */
    void addDirichletBC(GlobalIndex dof, double value);

    /**
     * @brief Add Dirichlet constraints for multiple DOFs
     */
    void addDirichletBC(std::span<const GlobalIndex> dofs, double value);

    /**
     * @brief Add Dirichlet constraints with different values
     */
    void addDirichletBC(std::span<const GlobalIndex> dofs,
                        std::span<const double> values);

    /**
     * @brief Add periodic boundary condition
     *
     * u_slave = u_master
     *
     * @param master Master DOF
     * @param slave Slave DOF (will be eliminated)
     */
    void addPeriodicBC(GlobalIndex master, GlobalIndex slave);

    /**
     * @brief Add multiple periodic pairs
     */
    void addPeriodicBC(std::span<const GlobalIndex> masters,
                       std::span<const GlobalIndex> slaves);

    /**
     * @brief Add general linear constraint
     *
     * constrained_dof = sum(coefficients[i] * dofs[i]) + inhomogeneity
     *
     * @param constrained_dof DOF being constrained
     * @param dofs Master DOF indices
     * @param coefficients Coefficients for each master DOF
     * @param inhomogeneity Constant term
     */
    void addLinearConstraint(GlobalIndex constrained_dof,
                             std::span<const GlobalIndex> dofs,
                             std::span<const double> coefficients,
                             double inhomogeneity = 0.0);

    /**
     * @brief Add hanging node constraint
     *
     * Hanging node DOF is constrained by parent DOFs.
     */
    void addHangingNodeConstraint(GlobalIndex hanging_dof,
                                   std::span<const GlobalIndex> parent_dofs,
                                   std::span<const double> weights);

    /**
     * @brief Clear all constraints
     */
    void clear();

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * @brief Check if a DOF is constrained
     */
    [[nodiscard]] bool isConstrained(GlobalIndex dof) const noexcept;

    /**
     * @brief Get constraint line for a DOF
     *
     * @param dof DOF index
     * @return Constraint line, or nullopt if not constrained
     */
    [[nodiscard]] std::optional<ConstraintLine> getConstraintLine(GlobalIndex dof) const;

    /**
     * @brief Get all constrained DOFs
     */
    [[nodiscard]] const IndexSet& getConstrainedDofs() const noexcept {
        return constrained_dofs_;
    }

    /**
     * @brief Get number of constraints
     */
    [[nodiscard]] std::size_t numConstraints() const noexcept {
        return constraints_.size();
    }

    /**
     * @brief Check if there are any constraints
     */
    [[nodiscard]] bool empty() const noexcept {
        return constraints_.empty();
    }

    /**
     * @brief Get Dirichlet value for a DOF
     *
     * @param dof DOF index
     * @return Dirichlet value, or nullopt if not a Dirichlet constraint
     */
    [[nodiscard]] std::optional<double> getDirichletValue(GlobalIndex dof) const;

    // =========================================================================
    // Constraint Application
    // =========================================================================

    /**
     * @brief Apply constraints to assembled matrix and RHS
     *
     * Uses symmetric elimination: modifies both matrix and RHS.
     *
     * @param matrix System matrix (modified)
     * @param rhs Right-hand side vector (modified)
     */
    void applyConstraints(AbstractMatrix& matrix, AbstractVector& rhs) const;

    /**
     * @brief Apply constraints to matrix only
     *
     * Sets constrained rows to identity.
     */
    void applyToMatrix(AbstractMatrix& matrix) const;

    /**
     * @brief Apply constraints to RHS only
     *
     * Sets constrained entries to prescribed values.
     */
    void applyToRhs(AbstractVector& rhs) const;

    /**
     * @brief Apply constraints after solution
     *
     * Recovers constrained DOF values from master DOFs.
     *
     * @param solution Solution vector (modified)
     */
    void applySolutionConstraints(AbstractVector& solution) const;

    // =========================================================================
    // Condensation
    // =========================================================================

    /**
     * @brief Build constraint transformation matrix
     *
     * Creates a sparse matrix C such that u_full = C * u_reduced + u_inhom
     * where u_reduced contains only unconstrained DOFs.
     *
     * @param n_total_dofs Total number of DOFs before condensation
     * @param row_offsets CSR row offsets (output)
     * @param col_indices CSR column indices (output)
     * @param values CSR values (output)
     */
    void buildConstraintMatrix(
        GlobalIndex n_total_dofs,
        std::vector<GlobalIndex>& row_offsets,
        std::vector<GlobalIndex>& col_indices,
        std::vector<double>& values) const;

    /**
     * @brief Get unconstrained DOF mapping
     *
     * @param n_total_dofs Total number of DOFs
     * @return Mapping from full DOF index to reduced index (-1 if constrained)
     */
    [[nodiscard]] std::vector<GlobalIndex> getReducedMapping(GlobalIndex n_total_dofs) const;

    /**
     * @brief Get number of unconstrained DOFs
     */
    [[nodiscard]] GlobalIndex numUnconstrainedDofs(GlobalIndex n_total_dofs) const;

    // =========================================================================
    // Closure and Validation
    // =========================================================================

    /**
     * @brief Close constraint set (compute transitive closure)
     *
     * Ensures constraints are consistent and resolves chains.
     */
    void close();

    /**
     * @brief Check if constraints are closed
     */
    [[nodiscard]] bool isClosed() const noexcept { return is_closed_; }

    /**
     * @brief Validate constraint consistency
     *
     * Checks for:
     * - Circular dependencies
     * - Invalid DOF references
     * - Conflicting constraints
     *
     * @return Error message, or empty string if valid
     */
    [[nodiscard]] std::string validate() const;

private:
    // Internal constraint storage
    std::unordered_map<GlobalIndex, ConstraintLine> constraints_;
    IndexSet constrained_dofs_;

    // State
    bool is_closed_{false};

    // Close a single constraint line (resolve dependencies transitively)
    void closeConstraint(GlobalIndex dof,
                         std::unordered_set<GlobalIndex>& visiting,
                         std::unordered_set<GlobalIndex>& closed);
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_DOFCONSTRAINTS_H
