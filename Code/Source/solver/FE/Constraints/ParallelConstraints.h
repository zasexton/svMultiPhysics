/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_PARALLELCONSTRAINTS_H
#define SVMP_FE_CONSTRAINTS_PARALLELCONSTRAINTS_H

/**
 * @file ParallelConstraints.h
 * @brief MPI constraint consistency and exchange
 *
 * ParallelConstraints handles the parallel aspects of constraint management:
 *
 * 1. **Consistency**: Ensures constraints for shared DOFs are identical across ranks
 * 2. **Exchange**: Transfers constraint lines for ghost DOFs from owning ranks
 * 3. **Conflict resolution**: Deterministic handling of constraint conflicts
 *
 * In a distributed setting:
 * - Each rank knows constraints for its locally owned DOFs
 * - Each rank needs constraints for locally relevant DOFs (owned + ghost)
 * - Constraints must be consistent across ranks for correctness
 *
 * The workflow is:
 * 1. Each rank builds local constraints (for owned DOFs)
 * 2. Call makeConsistent() to resolve conflicts and exchange
 * 3. Call close() on the resulting AffineConstraints
 *
 * Conflict resolution strategy:
 * - For DOFs with constraints from multiple ranks, the owning rank wins
 * - If ownership is ambiguous, smallest rank wins (deterministic)
 *
 * Module boundary:
 * - This module OWNS parallel constraint consistency logic
 * - This module does NOT OWN constraint storage (uses AffineConstraints)
 * - This module does NOT OWN DOF ownership (queries DofPartition)
 */

#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"
#include "Dofs/DofIndexSet.h"

#include <vector>
#include <span>
#include <optional>
#include <memory>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief Options for parallel constraint handling
 */
struct ParallelConstraintOptions {
    /**
     * @brief Conflict resolution strategy
     */
    enum class ConflictResolution {
        OwnerWins,      ///< Owning rank's constraint wins
        SmallestRank,   ///< Smallest rank's constraint wins
        Error           ///< Throw exception on conflict
    };

    ConflictResolution conflict_resolution{ConflictResolution::OwnerWins};

    /**
     * @brief Whether to validate constraints across ranks (expensive)
     */
    bool validate_consistency{false};

    /**
     * @brief Tolerance for comparing constraint values
     */
    double tolerance{1e-12};
};

/**
 * @brief Statistics from parallel constraint operations
 */
struct ParallelConstraintStats {
    GlobalIndex n_local_constraints{0};     ///< Constraints on locally owned DOFs
    GlobalIndex n_ghost_constraints{0};     ///< Constraints received for ghost DOFs
    GlobalIndex n_conflicts_resolved{0};    ///< Number of conflicts resolved
    GlobalIndex n_messages_sent{0};         ///< MPI messages sent
    GlobalIndex n_messages_received{0};     ///< MPI messages received
};

/**
 * @brief Parallel constraint manager
 *
 * ParallelConstraints coordinates constraint handling across MPI ranks.
 * It ensures that:
 * - All ranks have consistent constraints for shared DOFs
 * - Ghost DOF constraints are imported from owning ranks
 * - Conflicts are resolved deterministically
 *
 * Usage:
 * @code
 *   // Each rank builds local constraints
 *   AffineConstraints local_constraints;
 *   // ... add constraints for locally owned DOFs ...
 *
 *   // Create parallel handler
 *   ParallelConstraints parallel(MPI_COMM_WORLD, partition);
 *
 *   // Make constraints consistent across ranks
 *   parallel.makeConsistent(local_constraints);
 *
 *   // Import constraints for ghost DOFs
 *   parallel.importGhostConstraints(local_constraints);
 *
 *   // Now safe to close
 *   local_constraints.close();
 * @endcode
 */
class ParallelConstraints {
public:
    // =========================================================================
    // Construction
    // =========================================================================

#if FE_HAS_MPI
    /**
     * @brief Construct with MPI communicator and DOF partition
     *
     * @param comm MPI communicator
     * @param partition DOF ownership information
     */
    ParallelConstraints(MPI_Comm comm, const dofs::DofPartition& partition);
#endif

    /**
     * @brief Construct for serial execution (no-op operations)
     */
    ParallelConstraints();

    /**
     * @brief Construct with partition only (serial mode)
     */
    explicit ParallelConstraints(const dofs::DofPartition& partition);

    /**
     * @brief Destructor
     */
    ~ParallelConstraints();

    /**
     * @brief Move constructor
     */
    ParallelConstraints(ParallelConstraints&& other) noexcept;

    /**
     * @brief Move assignment
     */
    ParallelConstraints& operator=(ParallelConstraints&& other) noexcept;

    // Non-copyable
    ParallelConstraints(const ParallelConstraints&) = delete;
    ParallelConstraints& operator=(const ParallelConstraints&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set options for parallel operations
     */
    void setOptions(const ParallelConstraintOptions& options) {
        options_ = options;
    }

    /**
     * @brief Get current options
     */
    [[nodiscard]] const ParallelConstraintOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Check if running in parallel mode
     */
    [[nodiscard]] bool isParallel() const noexcept {
        return world_size_ > 1;
    }

    /**
     * @brief Get MPI rank
     */
    [[nodiscard]] int getRank() const noexcept { return my_rank_; }

    /**
     * @brief Get MPI world size
     */
    [[nodiscard]] int getWorldSize() const noexcept { return world_size_; }

    // =========================================================================
    // Main operations
    // =========================================================================

    /**
     * @brief Make constraints consistent across all ranks
     *
     * This operation:
     * 1. Gathers constraints for shared DOFs
     * 2. Resolves conflicts using the configured strategy
     * 3. Updates local constraints with resolved values
     *
     * @param constraints The local constraints (modified in place)
     * @return Statistics about the operation
     */
    ParallelConstraintStats makeConsistent(AffineConstraints& constraints);

    /**
     * @brief Import constraints for ghost DOFs from their owners
     *
     * After this call, constraints will contain lines for all locally
     * relevant DOFs (owned + ghost).
     *
     * @param constraints The local constraints (modified in place)
     * @return Statistics about the operation
     */
    ParallelConstraintStats importGhostConstraints(AffineConstraints& constraints);

    /**
     * @brief Combined operation: make consistent and import ghosts
     *
     * Convenience method that calls makeConsistent() then importGhostConstraints().
     *
     * @param constraints The local constraints (modified in place)
     * @return Combined statistics
     */
    ParallelConstraintStats synchronize(AffineConstraints& constraints);

    /**
     * @brief Export constraint lines for DOFs needed by other ranks
     *
     * Called implicitly by importGhostConstraints() on the sending side.
     * Can be called explicitly for custom communication patterns.
     *
     * @param constraints The local constraints (read-only)
     * @param requested_dofs DOFs requested by this rank
     * @return Constraint lines for the requested DOFs
     */
    std::vector<ConstraintLine> exportConstraints(
        const AffineConstraints& constraints,
        std::span<const GlobalIndex> requested_dofs) const;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate that constraints are consistent across ranks
     *
     * Expensive operation that compares constraints on all ranks.
     * Only call for debugging purposes.
     *
     * @param constraints The constraints to validate
     * @return true if constraints are consistent
     */
    [[nodiscard]] bool validateConsistency(const AffineConstraints& constraints) const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get statistics from last operation
     */
    [[nodiscard]] const ParallelConstraintStats& getLastStats() const noexcept {
        return last_stats_;
    }

private:
    // =========================================================================
    // Internal implementation
    // =========================================================================

#if FE_HAS_MPI
    // MPI helpers are implemented in ParallelConstraints.cpp
#endif

    /**
     * @brief Resolve conflict between two constraints for same DOF
     */
    [[nodiscard]] ConstraintLine resolveConflict(
        const ConstraintLine& local,
        const ConstraintLine& remote,
        int remote_rank) const;

    /**
     * @brief Determine which ranks need constraints for given DOFs
     */
    [[nodiscard]] std::vector<int> findNeighborRanks() const;

    // Data members
#if FE_HAS_MPI
    MPI_Comm comm_{MPI_COMM_NULL};
#endif
    const dofs::DofPartition* partition_{nullptr};
    ParallelConstraintOptions options_;
    ParallelConstraintStats last_stats_;

    int my_rank_{0};
    int world_size_{1};
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Make constraints consistent and import ghosts (convenience function)
 *
 * @param constraints The local constraints
 * @param partition DOF ownership information
 * @return Statistics about the operation
 */
#if FE_HAS_MPI
inline ParallelConstraintStats synchronizeConstraints(
    AffineConstraints& constraints,
    MPI_Comm comm,
    const dofs::DofPartition& partition)
{
    ParallelConstraints parallel(comm, partition);
    return parallel.synchronize(constraints);
}
#endif

/**
 * @brief Serial version (no-op, returns empty stats)
 */
inline ParallelConstraintStats synchronizeConstraintsSerial(
    [[maybe_unused]] AffineConstraints& constraints)
{
    return {};
}

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_PARALLELCONSTRAINTS_H
