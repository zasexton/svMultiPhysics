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

#ifndef SVMP_FE_SPARSITY_CONSTRAINT_SPARSITY_AUGMENTER_H
#define SVMP_FE_SPARSITY_CONSTRAINT_SPARSITY_AUGMENTER_H

/**
 * @file ConstraintSparsityAugmenter.h
 * @brief Structural fill-in for constraints (MPC, periodic, hanging nodes)
 *
 * This header provides the ConstraintSparsityAugmenter class for adding
 * structural couplings induced by DOF constraints. When a slave DOF u_s
 * is constrained to depend on master DOFs u_m via:
 *
 *   u_s = sum(alpha_i * u_{m_i}) + c
 *
 * Any row/column coupling involving u_s must also couple with the masters.
 * This class computes and inserts such induced couplings.
 *
 * Key features:
 * - Multiple augmentation modes (EliminationFill, KeepRowsSetDiag, ReducedSystem)
 * - Transitive closure handling (constraints depending on other constraints)
 * - Works with both SparsityPattern and DistributedSparsityPattern
 * - Deterministic fill-in for reproducibility
 * - Thread-safe for concurrent pattern access after augmentation
 *
 * Module boundary:
 * - This module OWNS structural augmentation (adding induced couplings)
 * - This module does NOT OWN constraint definitions (comes from FE/Dofs)
 * - This module does NOT OWN constraint algebra (enforcement done by Assembly)
 *
 * Complexity notes:
 * - augment(): O(n_constrained * avg_row_nnz * n_masters) for induced fill
 * - Memory: Additional O(n_fill) entries added to pattern
 *
 * @see SparsityPattern for the pattern representation
 * @see DofConstraints for constraint definitions (in FE/Dofs)
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <optional>

namespace svmp {
namespace FE {

namespace dofs {
class DofConstraints;
}

namespace sparsity {

/**
 * @brief Augmentation mode for constraint handling
 */
enum class AugmentationMode : std::uint8_t {
    /**
     * @brief Elimination fill-in mode
     *
     * Expands couplings induced by elimination/condensation. For each
     * constrained DOF u_s with masters {u_m}, adds couplings:
     * - For each (i, u_s) coupling: add (i, u_m) for all masters
     * - For each (u_s, j) coupling: add (u_m, j) for all masters
     *
     * This is the standard approach for static condensation.
     */
    EliminationFill,

    /**
     * @brief Keep rows, set diagonal mode
     *
     * Keeps constrained rows structurally nonempty with diagonal present.
     * Used when constraint enforcement modifies the matrix in-place
     * (e.g., setting row to identity for Dirichlet).
     *
     * Adds:
     * - Diagonal entry (u_s, u_s) for each constrained DOF
     * - Does not expand couplings to masters
     */
    KeepRowsSetDiag,

    /**
     * @brief Reduced system mode
     *
     * Build pattern for the reduced system where constrained DOFs
     * are eliminated. Pattern contains only unconstrained DOFs with
     * induced fill from elimination.
     *
     * Result has size (n_unconstrained x n_unconstrained).
     */
    ReducedSystem
};

/**
 * @brief Single constraint entry (lightweight view)
 *
 * Represents a single term in a linear constraint:
 *   constrained_dof = sum(coefficient * master_dof) + inhomogeneity
 */
struct ConstraintTerm {
    GlobalIndex master_dof;   ///< Master DOF index
    double coefficient{1.0};  ///< Coefficient (for structural purposes, only presence matters)
};

/**
 * @brief A constraint line (lightweight struct for sparsity purposes)
 *
 * For structural fill-in, we only need to know which DOFs are involved,
 * not the actual coefficients. This struct captures the minimum information
 * needed from a constraint definition.
 */
struct SparsityConstraint {
    GlobalIndex constrained_dof;             ///< The slave DOF being constrained
    std::vector<GlobalIndex> master_dofs;    ///< The master DOFs it depends on

    /**
     * @brief Check if this is a Dirichlet (no masters) constraint
     */
    [[nodiscard]] bool isDirichlet() const noexcept {
        return master_dofs.empty();
    }
};

/**
 * @brief Abstract interface for constraint query
 *
 * This interface allows ConstraintSparsityAugmenter to work with different
 * constraint implementations without a hard dependency on DofConstraints.
 */
class IConstraintQuery {
public:
    virtual ~IConstraintQuery() = default;

    /**
     * @brief Check if a DOF is constrained
     */
    [[nodiscard]] virtual bool isConstrained(GlobalIndex dof) const = 0;

    /**
     * @brief Get the master DOFs for a constrained DOF
     *
     * @param constrained_dof The slave DOF
     * @return Vector of master DOF indices, or empty if not constrained or Dirichlet
     */
    [[nodiscard]] virtual std::vector<GlobalIndex> getMasterDofs(GlobalIndex constrained_dof) const = 0;

    /**
     * @brief Get all constrained DOFs
     */
    [[nodiscard]] virtual std::vector<GlobalIndex> getAllConstrainedDofs() const = 0;

    /**
     * @brief Get number of constraints
     */
    [[nodiscard]] virtual std::size_t numConstraints() const = 0;
};

/**
 * @brief Adapter to wrap dofs::DofConstraints as IConstraintQuery
 *
 * Provides a lightweight bridge so Sparsity can leverage the existing
 * constraints implementation in FE/Dofs without requiring callers to
 * translate into SparsityConstraint manually.
 *
 * Note: The referenced DofConstraints object must outlive this adapter.
 */
class DofConstraintsAdapter : public IConstraintQuery {
public:
    explicit DofConstraintsAdapter(const dofs::DofConstraints& constraints);

    [[nodiscard]] bool isConstrained(GlobalIndex dof) const override;
    [[nodiscard]] std::vector<GlobalIndex> getMasterDofs(GlobalIndex constrained_dof) const override;
    [[nodiscard]] std::vector<GlobalIndex> getAllConstrainedDofs() const override;
    [[nodiscard]] std::size_t numConstraints() const override;

private:
    const dofs::DofConstraints* constraints_{nullptr};
};

/**
 * @brief Simple constraint set implementation
 *
 * A concrete implementation of IConstraintQuery using internal storage.
 * Useful for building constraints programmatically.
 */
class SimpleConstraintSet : public IConstraintQuery {
public:
    SimpleConstraintSet() = default;

    /**
     * @brief Add a Dirichlet constraint (no masters)
     */
    void addDirichlet(GlobalIndex dof);

    /**
     * @brief Add a constraint with single master
     */
    void addConstraint(GlobalIndex slave, GlobalIndex master);

    /**
     * @brief Add a constraint with multiple masters
     */
    void addConstraint(GlobalIndex slave, std::span<const GlobalIndex> masters);

    /**
     * @brief Add constraint from SparsityConstraint struct
     */
    void addConstraint(const SparsityConstraint& constraint);

    /**
     * @brief Clear all constraints
     */
    void clear();

    // IConstraintQuery interface
    [[nodiscard]] bool isConstrained(GlobalIndex dof) const override;
    [[nodiscard]] std::vector<GlobalIndex> getMasterDofs(GlobalIndex dof) const override;
    [[nodiscard]] std::vector<GlobalIndex> getAllConstrainedDofs() const override;
    [[nodiscard]] std::size_t numConstraints() const override;

private:
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> constraints_;
};

/**
 * @brief Statistics about constraint augmentation
 */
struct AugmentationStats {
    GlobalIndex n_constraints{0};         ///< Number of constraints processed
    GlobalIndex n_dirichlet{0};           ///< Number of Dirichlet (no-master) constraints
    GlobalIndex n_periodic{0};            ///< Number of single-master constraints
    GlobalIndex n_multipoint{0};          ///< Number of multi-master constraints
    GlobalIndex n_fill_entries{0};        ///< Number of structural fill entries added
    GlobalIndex n_diagonal_added{0};      ///< Number of diagonal entries added
    GlobalIndex original_nnz{0};          ///< NNZ before augmentation
    GlobalIndex augmented_nnz{0};         ///< NNZ after augmentation
};

/**
 * @brief Result of building a reduced distributed sparsity pattern
 *
 * The returned pattern lives in the reduced (unconstrained-only) index space.
 * Mapping vectors describe how locally owned full-system DOFs map into that
 * reduced index space.
 */
struct ReducedDistributedPatternResult {
    DistributedSparsityPattern pattern;              ///< Reduced distributed pattern (finalized)
    IndexRange owned_reduced_range;                  ///< Owned reduced rows on this rank
    GlobalIndex global_reduced_size{0};              ///< Total unconstrained DOFs globally
    std::vector<GlobalIndex> full_to_reduced_owned;  ///< Per-owned-full-row mapping (size = n_owned_full_rows)
    std::vector<GlobalIndex> reduced_to_full_owned;  ///< Owned reduced -> full global DOF (size = n_owned_reduced_rows)
};

/**
 * @brief Options for constraint sparsity augmentation
 */
struct AugmentationOptions {
    AugmentationMode mode{AugmentationMode::EliminationFill};  ///< Augmentation mode
    bool compute_transitive_closure{true};   ///< Resolve constraint chains
    bool symmetric_fill{true};               ///< Add symmetric fill (if A[i,j] induced, also A[j,i])
    bool ensure_diagonal{true};              ///< Ensure diagonal present for constrained rows
    bool preserve_original_entries{true};    ///< Keep original pattern entries
    bool include_ghost_columns{true};        ///< Allow adding non-owned columns in distributed patterns
};

/**
 * @brief Augments sparsity patterns with constraint-induced structural fill
 *
 * ConstraintSparsityAugmenter handles the structural consequences of DOF
 * constraints on sparsity patterns. When assembling matrices with constraints,
 * the pattern must include induced couplings from constraint elimination.
 *
 * Example: If DOF 5 is constrained to depend on DOFs {2, 3}:
 * - Any row that coupled with column 5 must also couple with columns 2, 3
 * - Any column that coupled with row 5 must also couple with rows 2, 3
 *
 * Usage:
 * @code
 * // Build base sparsity pattern
 * SparsityBuilder builder;
 * builder.setRowDofMap(dof_map);
 * SparsityPattern pattern = builder.build();
 *
 * // Create constraint query from DofConstraints
 * ConstraintSparsityAugmenter augmenter;
 * augmenter.setConstraints(constraint_query);
 *
 * // Augment pattern with constraint fill
 * augmenter.augment(pattern, AugmentationMode::EliminationFill);
 * @endcode
 *
 * @note The input pattern must be in Building state (not finalized).
 *       After augmentation, call pattern.finalize() before use.
 */
class ConstraintSparsityAugmenter {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    ConstraintSparsityAugmenter() = default;

    /**
     * @brief Construct with constraint query interface
     *
     * @param constraint_query Interface to query constraint information
     */
    explicit ConstraintSparsityAugmenter(std::shared_ptr<IConstraintQuery> constraint_query);

    /**
     * @brief Construct with simple constraint set
     *
     * @param constraints Vector of sparsity constraints
     */
    explicit ConstraintSparsityAugmenter(std::vector<SparsityConstraint> constraints);

    /// Destructor
    ~ConstraintSparsityAugmenter() = default;

    // Non-copyable (constraint query may be shared_ptr)
    ConstraintSparsityAugmenter(const ConstraintSparsityAugmenter&) = delete;
    ConstraintSparsityAugmenter& operator=(const ConstraintSparsityAugmenter&) = delete;

    // Movable
    ConstraintSparsityAugmenter(ConstraintSparsityAugmenter&&) = default;
    ConstraintSparsityAugmenter& operator=(ConstraintSparsityAugmenter&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set constraint query interface
     *
     * @param constraint_query Interface to query constraint information
     */
    void setConstraints(std::shared_ptr<IConstraintQuery> constraint_query);

    /**
     * @brief Set constraints from vector of SparsityConstraint
     *
     * @param constraints Vector of constraints
     */
    void setConstraints(std::vector<SparsityConstraint> constraints);

    /**
     * @brief Set augmentation options
     */
    void setOptions(const AugmentationOptions& options) {
        options_ = options;
    }

    /**
     * @brief Get current options
     */
    [[nodiscard]] const AugmentationOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Check if constraints are configured
     */
    [[nodiscard]] bool hasConstraints() const noexcept {
        return constraint_query_ != nullptr && constraint_query_->numConstraints() > 0;
    }

    // =========================================================================
    // Augmentation methods
    // =========================================================================

    /**
     * @brief Augment sparsity pattern with constraint-induced fill
     *
     * @param pattern Pattern to augment (must be in Building state)
     * @param mode Augmentation mode (overrides options_.mode if specified)
     * @return Augmentation statistics
     * @throws FEException if pattern is finalized or constraints not set
     *
     * The pattern is modified in-place. After augmentation, the pattern
     * will contain all original entries plus induced fill from constraints.
     *
     * Complexity: O(n_constrained * (avg_row_nnz + n_masters))
     */
    AugmentationStats augment(SparsityPattern& pattern,
                              std::optional<AugmentationMode> mode = std::nullopt);

    /**
     * @brief Augment distributed sparsity pattern
     *
     * @param pattern Distributed pattern to augment (must be in Building state)
     * @param mode Augmentation mode
     * @return Augmentation statistics
     *
     * Handles ghost column relationships appropriately.
     */
    AugmentationStats augment(DistributedSparsityPattern& pattern,
                              std::optional<AugmentationMode> mode = std::nullopt);

    /**
     * @brief Build reduced system pattern (ReducedSystem mode)
     *
     * @param original Original pattern (can be finalized)
     * @return New pattern containing only unconstrained DOFs with fill
     *
     * Creates a new pattern of size (n_unconstrained x n_unconstrained)
     * with constraint-induced fill. Original pattern is not modified.
     */
    [[nodiscard]] SparsityPattern buildReducedPattern(const SparsityPattern& original);

#if FE_HAS_MPI
    /**
     * @brief Build a reduced distributed pattern (ReducedSystem mode)
     *
     * @param original Finalized distributed pattern in the full index space
     * @param comm MPI communicator defining rank layout for the reduced system
     * @return Reduced distributed pattern + local mapping vectors
     *
     * The reduced distribution is computed by compacting locally owned
     * unconstrained DOFs and performing a prefix sum across ranks.
     */
    [[nodiscard]] ReducedDistributedPatternResult buildReducedDistributedPattern(
        const DistributedSparsityPattern& original,
        MPI_Comm comm) const;
#endif

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Get the mapping from full to reduced DOF indices
     *
     * @param n_total Total number of DOFs
     * @return Vector where result[full_dof] = reduced_dof, or -1 if constrained
     */
    [[nodiscard]] std::vector<GlobalIndex> getReducedMapping(GlobalIndex n_total) const;

    /**
     * @brief Get the mapping from reduced to full DOF indices
     *
     * @param n_total Total number of DOFs
     * @return Vector where result[reduced_dof] = full_dof
     */
    [[nodiscard]] std::vector<GlobalIndex> getFullMapping(GlobalIndex n_total) const;

    /**
     * @brief Get number of unconstrained DOFs
     */
    [[nodiscard]] GlobalIndex numUnconstrainedDofs(GlobalIndex n_total) const;

    /**
     * @brief Get statistics from last augmentation
     */
    [[nodiscard]] const AugmentationStats& getLastStats() const noexcept {
        return last_stats_;
    }

private:
    // Internal implementation methods
    void augmentEliminationFill(SparsityPattern& pattern);
    void augmentKeepRowsSetDiag(SparsityPattern& pattern);
    SparsityPattern buildReducedSystemPattern(const SparsityPattern& original);

    // Compute transitive closure of constraints
    std::vector<GlobalIndex> getTransitiveMasters(GlobalIndex constrained_dof) const;

    // Validate inputs
    void validateConstraints() const;

    // Constraint query interface
    std::shared_ptr<IConstraintQuery> constraint_query_;

    // Options
    AugmentationOptions options_;

    // Statistics from last operation
    mutable AugmentationStats last_stats_;
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Augment pattern with constraints (convenience function)
 *
 * @param pattern Pattern to augment
 * @param constraints Constraint definitions
 * @param mode Augmentation mode
 * @return Augmentation statistics
 */
inline AugmentationStats augmentWithConstraints(
    SparsityPattern& pattern,
    const std::vector<SparsityConstraint>& constraints,
    AugmentationMode mode = AugmentationMode::EliminationFill)
{
    std::vector<SparsityConstraint> constraints_copy(constraints);
    ConstraintSparsityAugmenter augmenter{std::move(constraints_copy)};
    return augmenter.augment(pattern, mode);
}

/**
 * @brief Build reduced sparsity pattern (convenience function)
 *
 * @param original Original pattern
 * @param constraints Constraint definitions
 * @return Reduced pattern with constraint fill
 */
inline SparsityPattern buildReducedSparsityPattern(
    const SparsityPattern& original,
    const std::vector<SparsityConstraint>& constraints)
{
    std::vector<SparsityConstraint> constraints_copy(constraints);
    ConstraintSparsityAugmenter augmenter{std::move(constraints_copy)};
    return augmenter.buildReducedPattern(original);
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_CONSTRAINT_SPARSITY_AUGMENTER_H
