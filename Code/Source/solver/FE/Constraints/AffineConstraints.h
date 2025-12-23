/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_AFFINECONSTRAINTS_H
#define SVMP_FE_CONSTRAINTS_AFFINECONSTRAINTS_H

/**
 * @file AffineConstraints.h
 * @brief Canonical algebraic model for linear DOF constraints
 *
 * AffineConstraints is the backbone of the Constraints module. It represents
 * algebraic relationships between degrees of freedom in the canonical form:
 *
 *   u_s = sum_{i} a_i * u_{m_i} + b
 *
 * where u_s is the "slave" (constrained) DOF, u_{m_i} are "master" DOFs with
 * weights a_i, and b is the inhomogeneity constant.
 *
 * Design principles (inspired by deal.II AffineConstraints):
 * - Two-phase construction: building phase (addLine, addEntry) then close()
 * - Transitive closure computed at close() time
 * - Cycle detection with clear error diagnostics
 * - Deterministic ordering for reproducible results
 * - Optimized CSR-like storage for fast read access during assembly
 * - Efficient update path for time-dependent inhomogeneities
 *
 * Module boundaries:
 * - This module OWNS: constraint storage, closure, validation, read-optimized form
 * - This module does NOT OWN: DOF numbering (Dofs/), mesh (Mesh/), solvers (Backends/)
 *
 * @see ConstraintDistributor for assembly integration
 * @see Dofs/DofConstraints for the simpler DOF-level constraint interface
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <functional>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace constraints {

// Forward declarations
class ConstraintDistributor;

/**
 * @brief Single entry in a constraint line (master DOF with weight)
 */
struct ConstraintEntry {
    GlobalIndex master_dof;   ///< Master DOF index
    double weight{1.0};       ///< Coefficient/weight for this master

    bool operator==(const ConstraintEntry& other) const noexcept {
        return master_dof == other.master_dof &&
               std::abs(weight - other.weight) < 1e-15;
    }

    bool operator<(const ConstraintEntry& other) const noexcept {
        return master_dof < other.master_dof;
    }
};

/**
 * @brief A single constraint line representing: slave = sum(weight_i * master_i) + inhomogeneity
 *
 * This is the "open" form used during construction. After close(), constraints
 * are stored in the more efficient CSR format within AffineConstraints.
 */
struct ConstraintLine {
    GlobalIndex slave_dof{-1};              ///< The constrained (slave) DOF
    std::vector<ConstraintEntry> entries;   ///< Master DOFs with weights
    double inhomogeneity{0.0};              ///< Constant term

    /**
     * @brief Check if this is a homogeneous constraint (b = 0)
     */
    [[nodiscard]] bool isHomogeneous() const noexcept {
        return std::abs(inhomogeneity) < 1e-15;
    }

    /**
     * @brief Check if this is a Dirichlet-style constraint (no masters)
     */
    [[nodiscard]] bool isDirichlet() const noexcept {
        return entries.empty();
    }

    /**
     * @brief Check if this is a simple periodic constraint (single master with weight 1)
     */
    [[nodiscard]] bool isSimplePeriodic() const noexcept {
        return entries.size() == 1 &&
               std::abs(entries[0].weight - 1.0) < 1e-15 &&
               isHomogeneous();
    }

    /**
     * @brief Sort entries by master DOF index (for determinism)
     */
    void sortEntries() {
        std::sort(entries.begin(), entries.end());
    }

    /**
     * @brief Merge duplicate masters (sum weights)
     */
    void mergeEntries();
};

/**
 * @brief Validation result from constraint checking
 */
struct ValidationResult {
    bool valid{true};
    std::string error_message;
    std::vector<GlobalIndex> problematic_dofs;

    operator bool() const noexcept { return valid; }
};

/**
 * @brief Statistics about the constraint set
 */
struct ConstraintStatistics {
    GlobalIndex n_constraints{0};           ///< Total number of constrained DOFs
    GlobalIndex n_dirichlet{0};             ///< Number with no masters (Dirichlet)
    GlobalIndex n_simple_periodic{0};       ///< Number with single master, weight 1
    GlobalIndex n_multipoint{0};            ///< Number with multiple masters
    GlobalIndex n_inhomogeneous{0};         ///< Number with nonzero inhomogeneity
    GlobalIndex total_entries{0};           ///< Total number of master entries
    double avg_masters_per_constraint{0.0}; ///< Average number of masters
};

/**
 * @brief Options for AffineConstraints operations
 */
struct AffineConstraintsOptions {
    double zero_tolerance{1e-15};           ///< Threshold for treating values as zero
    bool allow_overwrite{false};            ///< Allow overwriting existing constraints
    bool detect_cycles{true};               ///< Check for constraint cycles during close()
    bool deterministic_order{true};         ///< Sort for deterministic results
    bool merge_duplicates{true};            ///< Merge duplicate master entries
};

/**
 * @brief Canonical algebraic constraint storage with transitive closure
 *
 * AffineConstraints manages linear relationships between DOFs. The typical
 * workflow is:
 *
 * 1. Construction: Create empty or with options
 * 2. Building: Call addLine(), addEntry(), setInhomogeneity() to add constraints
 * 3. Closing: Call close() to compute transitive closure and finalize
 * 4. Usage: Query isConstrained(), getConstraint(), or use with ConstraintDistributor
 *
 * Example:
 * @code
 *   AffineConstraints constraints;
 *
 *   // Add Dirichlet BC: u_5 = 1.0
 *   constraints.addLine(5);
 *   constraints.setInhomogeneity(5, 1.0);
 *
 *   // Add periodic BC: u_10 = u_20
 *   constraints.addLine(10);
 *   constraints.addEntry(10, 20, 1.0);
 *
 *   // Add MPC: u_15 = 0.5*u_1 + 0.5*u_2
 *   constraints.addLine(15);
 *   constraints.addEntry(15, 1, 0.5);
 *   constraints.addEntry(15, 2, 0.5);
 *
 *   // Finalize
 *   constraints.close();
 *
 *   // Use
 *   if (constraints.isConstrained(5)) {
 *       auto line = constraints.getConstraint(5);
 *       // ...
 *   }
 * @endcode
 *
 * Thread safety:
 * - Building phase: NOT thread-safe (single writer)
 * - After close(): Read operations are thread-safe
 */
class AffineConstraints {
public:
    // =========================================================================
    // Types
    // =========================================================================

    /**
     * @brief Read-only view of a constraint (after close)
     */
    struct ConstraintView {
        GlobalIndex slave_dof{-1};
        std::span<const ConstraintEntry> entries;
        double inhomogeneity{0.0};

        [[nodiscard]] bool isDirichlet() const noexcept {
            return entries.empty();
        }

        [[nodiscard]] bool isHomogeneous() const noexcept {
            return std::abs(inhomogeneity) < 1e-15;
        }
    };

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    AffineConstraints();

    /**
     * @brief Construct with options
     */
    explicit AffineConstraints(const AffineConstraintsOptions& options);

    /**
     * @brief Destructor
     */
    ~AffineConstraints();

    /**
     * @brief Move constructor
     */
    AffineConstraints(AffineConstraints&& other) noexcept;

    /**
     * @brief Move assignment
     */
    AffineConstraints& operator=(AffineConstraints&& other) noexcept;

    /**
     * @brief Copy constructor
     */
    AffineConstraints(const AffineConstraints& other);

    /**
     * @brief Copy assignment
     */
    AffineConstraints& operator=(const AffineConstraints& other);

    // =========================================================================
    // Building phase (before close)
    // =========================================================================

    /**
     * @brief Add a new constraint line for a DOF
     *
     * This declares that `slave_dof` is constrained. You must then use
     * addEntry() to specify the master DOFs and weights, and optionally
     * setInhomogeneity() for the constant term.
     *
     * @param slave_dof The DOF being constrained
     * @throws FEException if already closed or DOF already constrained (unless allow_overwrite)
     */
    void addLine(GlobalIndex slave_dof);

    /**
     * @brief Add multiple constraint lines
     */
    void addLines(std::span<const GlobalIndex> slave_dofs);

    /**
     * @brief Add a master DOF entry to an existing constraint line
     *
     * @param slave_dof The constrained DOF (must have been added via addLine)
     * @param master_dof The master DOF
     * @param weight The coefficient for this master
     * @throws FEException if slave not found or already closed
     */
    void addEntry(GlobalIndex slave_dof, GlobalIndex master_dof, double weight);

    /**
     * @brief Add multiple entries to a constraint line
     */
    void addEntries(GlobalIndex slave_dof,
                    std::span<const GlobalIndex> master_dofs,
                    std::span<const double> weights);

    /**
     * @brief Set the inhomogeneity (constant term) for a constraint
     *
     * @param slave_dof The constrained DOF
     * @param value The inhomogeneity value
     * @throws FEException if slave not found or already closed
     */
    void setInhomogeneity(GlobalIndex slave_dof, double value);

    /**
     * @brief Add to the inhomogeneity (constant term) for a constraint
     *
     * @param slave_dof The constrained DOF
     * @param value The value to add
     * @throws FEException if slave not found or already closed
     */
    void addInhomogeneity(GlobalIndex slave_dof, double value);

    /**
     * @brief Add a complete constraint line
     *
     * Convenience method to add a constraint in one call.
     *
     * @param line The constraint line to add
     * @throws FEException if already closed
     */
    void addConstraintLine(const ConstraintLine& line);

    /**
     * @brief Add a Dirichlet constraint: slave = value
     *
     * Convenience method for the common case of fixing a DOF to a value.
     */
    void addDirichlet(GlobalIndex dof, double value);

    /**
     * @brief Add multiple Dirichlet constraints with same value
     */
    void addDirichlet(std::span<const GlobalIndex> dofs, double value);

    /**
     * @brief Add multiple Dirichlet constraints with different values
     */
    void addDirichlet(std::span<const GlobalIndex> dofs,
                      std::span<const double> values);

    /**
     * @brief Add a periodic constraint: slave = master
     *
     * Convenience method for simple periodicity.
     */
    void addPeriodic(GlobalIndex slave_dof, GlobalIndex master_dof);

    /**
     * @brief Merge constraints from another AffineConstraints object
     *
     * @param other The constraints to merge in
     * @param overwrite If true, other's constraints override conflicts
     * @throws FEException if either object is closed
     */
    void merge(const AffineConstraints& other, bool overwrite = false);

    /**
     * @brief Clear all constraints and reset to building state
     */
    void clear();

    // =========================================================================
    // Closing
    // =========================================================================

    /**
     * @brief Finalize constraints and compute transitive closure
     *
     * After calling close():
     * - No more modifications are allowed
     * - Constraints are stored in optimized CSR format
     * - Transitive closure is computed (chains resolved)
     * - Read operations become thread-safe
     *
     * @throws FEException if cycles detected or invalid constraints found
     */
    void close();

    /**
     * @brief Check if constraints are closed (finalized)
     */
    [[nodiscard]] bool isClosed() const noexcept { return is_closed_; }

    // =========================================================================
    // Query (after close)
    // =========================================================================

    /**
     * @brief Check if a DOF is constrained
     */
    [[nodiscard]] bool isConstrained(GlobalIndex dof) const noexcept;

    /**
     * @brief Check if any DOF in a set is constrained
     */
    [[nodiscard]] bool hasConstrainedDofs(std::span<const GlobalIndex> dofs) const noexcept;

    /**
     * @brief Get constraint for a DOF
     *
     * @param dof The DOF to query
     * @return ConstraintView if constrained, nullopt otherwise
     *
     * @note Requires close() to have been called
     */
    [[nodiscard]] std::optional<ConstraintView> getConstraint(GlobalIndex dof) const;

    /**
     * @brief Get inhomogeneity value for a DOF
     *
     * @param dof The DOF to query
     * @return Inhomogeneity value, or 0.0 if not constrained
     */
    [[nodiscard]] double getInhomogeneity(GlobalIndex dof) const noexcept;

    /**
     * @brief Get all constrained DOF indices
     */
    [[nodiscard]] std::vector<GlobalIndex> getConstrainedDofs() const;

    /**
     * @brief Get number of constraints
     */
    [[nodiscard]] std::size_t numConstraints() const noexcept {
        return slave_to_index_.size();
    }

    /**
     * @brief Check if there are any constraints
     */
    [[nodiscard]] bool empty() const noexcept {
        return slave_to_index_.empty();
    }

    /**
     * @brief Get statistics about the constraint set
     */
    [[nodiscard]] ConstraintStatistics getStatistics() const;

    // =========================================================================
    // Constraint application (distribute)
    // =========================================================================

    /**
     * @brief Distribute constraints to a solution vector
     *
     * Enforces constraints on the vector: for each constrained DOF,
     * set vec[slave] = sum(weight_i * vec[master_i]) + inhomogeneity
     *
     * @param vec Vector to modify (indexed by DOF)
     * @param vec_size Size of vector
     */
    void distribute(double* vec, GlobalIndex vec_size) const;

    /**
     * @brief Distribute constraints to a solution vector (std::vector)
     */
    void distribute(std::vector<double>& vec) const {
        distribute(vec.data(), static_cast<GlobalIndex>(vec.size()));
    }

    /**
     * @brief Set constrained DOF values in vector (just inhomogeneities)
     *
     * Sets vec[slave] = inhomogeneity for each constrained DOF.
     * Used for setting initial values or post-solve cleanup.
     *
     * @param vec Vector to modify
     * @param vec_size Size of vector
     */
    void setConstrainedValues(double* vec, GlobalIndex vec_size) const;

    /**
     * @brief Set constrained DOF values in vector (std::vector)
     */
    void setConstrainedValues(std::vector<double>& vec) const {
        setConstrainedValues(vec.data(), static_cast<GlobalIndex>(vec.size()));
    }

    // =========================================================================
    // Inhomogeneity updates (for time-dependent BCs)
    // =========================================================================

    /**
     * @brief Update inhomogeneity for a DOF (does not require re-closing)
     *
     * This allows efficient updates of Dirichlet BC values without
     * rebuilding the entire constraint structure.
     *
     * @param dof The constrained DOF
     * @param value New inhomogeneity value
     * @throws FEException if DOF is not constrained
     */
    void updateInhomogeneity(GlobalIndex dof, double value);

    /**
     * @brief Update multiple inhomogeneities
     */
    void updateInhomogeneities(std::span<const GlobalIndex> dofs,
                                std::span<const double> values);

    /**
     * @brief Set all inhomogeneities to zero
     */
    void clearInhomogeneities();

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate constraints (can be called before or after close)
     *
     * Checks for:
     * - Cycles in constraint dependencies
     * - Masters that are themselves constrained (resolved by closure)
     * - Invalid DOF indices (negative)
     * - Empty constraint lines
     *
     * @return Validation result with error details if invalid
     */
    [[nodiscard]] ValidationResult validate() const;

    // =========================================================================
    // Iteration support
    // =========================================================================

    /**
     * @brief Iterate over all constraints
     *
     * @param callback Function called for each constraint
     */
    void forEach(std::function<void(const ConstraintView&)> callback) const;

    // =========================================================================
    // Options
    // =========================================================================

    /**
     * @brief Get current options
     */
    [[nodiscard]] const AffineConstraintsOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Set options (only before close)
     */
    void setOptions(const AffineConstraintsOptions& options);

private:
    // =========================================================================
    // Internal helpers
    // =========================================================================

    /**
     * @brief Ensure not closed (throws if closed)
     */
    void checkNotClosed() const;

    /**
     * @brief Ensure closed (throws if not closed)
     */
    void checkClosed() const;

    /**
     * @brief Find or create constraint line for a DOF
     */
    ConstraintLine& getOrCreateLine(GlobalIndex slave_dof);

    /**
     * @brief Find constraint line for a DOF
     */
    ConstraintLine* findLine(GlobalIndex slave_dof);
    const ConstraintLine* findLine(GlobalIndex slave_dof) const;

    /**
     * @brief Compute transitive closure
     *
     * Resolves chains where a master DOF is itself constrained.
     */
    void computeTransitiveClosure();

    /**
     * @brief Close a single constraint line (recursive with cycle detection)
     */
    void closeLine(GlobalIndex slave_dof,
                   std::unordered_set<GlobalIndex>& visiting,
                   std::unordered_set<GlobalIndex>& closed);

    /**
     * @brief Build CSR storage from constraint lines
     */
    void buildCSRStorage();

    /**
     * @brief Sort constraints for deterministic iteration
     */
    void sortConstraints();

    /**
     * @brief Merge duplicate entries in constraint lines
     */
    void mergeAllDuplicates();

    // =========================================================================
    // Data members
    // =========================================================================

    // Options
    AffineConstraintsOptions options_;

    // State
    bool is_closed_{false};

    // Building phase storage (sparse map)
    std::unordered_map<GlobalIndex, ConstraintLine> building_lines_;

    // Closed phase storage (CSR-like for fast read)
    // slave_dofs_[i] is the i-th constrained DOF
    std::vector<GlobalIndex> slave_dofs_;

    // Map from slave DOF to index in slave_dofs_
    std::unordered_map<GlobalIndex, std::size_t> slave_to_index_;

    // CSR storage for entries
    std::vector<GlobalIndex> entry_offsets_;  // Size: n_constraints + 1
    std::vector<ConstraintEntry> entries_;    // All entries, concatenated

    // Inhomogeneities (parallel to slave_dofs_)
    std::vector<double> inhomogeneities_;
};

// ============================================================================
// Exception types
// ============================================================================

/**
 * @brief Exception for constraint-related errors
 */
class ConstraintException : public FEException {
public:
    ConstraintException(const std::string& message,
                        GlobalIndex dof = INVALID_GLOBAL_INDEX,
                        const char* file = "",
                        int line = 0)
        : FEException(buildMessage(message, dof), file, line),
          dof_(dof) {}

    [[nodiscard]] GlobalIndex dof() const noexcept { return dof_; }

private:
    GlobalIndex dof_;

    static std::string buildMessage(const std::string& msg, GlobalIndex dof) {
        if (dof != INVALID_GLOBAL_INDEX) {
            return msg + " (DOF: " + std::to_string(dof) + ")";
        }
        return msg;
    }
};

/**
 * @brief Exception for constraint cycles
 */
class ConstraintCycleException : public ConstraintException {
public:
    ConstraintCycleException(const std::vector<GlobalIndex>& cycle,
                             const char* file = "",
                             int line = 0)
        : ConstraintException(buildCycleMessage(cycle), cycle.empty() ? -1 : cycle[0], file, line),
          cycle_(cycle) {}

    [[nodiscard]] const std::vector<GlobalIndex>& getCycle() const noexcept {
        return cycle_;
    }

private:
    std::vector<GlobalIndex> cycle_;

    static std::string buildCycleMessage(const std::vector<GlobalIndex>& cycle) {
        std::string msg = "Constraint cycle detected: ";
        for (std::size_t i = 0; i < cycle.size(); ++i) {
            if (i > 0) msg += " -> ";
            msg += std::to_string(cycle[i]);
        }
        return msg;
    }
};

// ============================================================================
// Macros for constraint-related exceptions
// ============================================================================

#define CONSTRAINT_THROW(message) \
    throw ConstraintException(message, INVALID_GLOBAL_INDEX, __FILE__, __LINE__)

#define CONSTRAINT_THROW_DOF(message, dof) \
    throw ConstraintException(message, dof, __FILE__, __LINE__)

#define CONSTRAINT_THROW_IF(condition, message) \
    do { \
        if (condition) { \
            CONSTRAINT_THROW(message); \
        } \
    } while(0)

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_AFFINECONSTRAINTS_H
