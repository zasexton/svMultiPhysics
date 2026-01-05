/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_GHOSTDOFMANAGER_H
#define SVMP_FE_DOFS_GHOSTDOFMANAGER_H

/**
 * @file GhostDofManager.h
 * @brief Manage ghost and shared DOFs in parallel
 *
 * The GhostDofManager handles the distinction between:
 * - **Shared DOFs**: DOFs on partition interfaces that exist on multiple ranks.
 *   For assembly, each shared DOF has a designated owner.
 * - **Ghost DOFs**: DOFs owned by other ranks but needed locally for computation
 *   (e.g., evaluating residuals, applying stencils).
 *
 * Key responsibilities:
 * - Identify ghost DOFs based on ghost cells from mesh
 * - Track shared DOFs by neighbor rank for communication
 * - Build send/receive schedules for ghost exchange
 * - Resolve ownership for shared DOFs deterministically
 *
 * The send/receive schedules are suitable for MPI communication patterns
 * but are expressed without MPI dependency.
 */

#include "DofMap.h"
#include "DofIndexSet.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>

// Forward declarations
namespace svmp {
    class DistributedMesh;
    // Phase 5 (UNIFY_MESH): prefer the unified runtime mesh type name.
    // In the Mesh library, `Mesh` is currently an alias of `DistributedMesh`.
    using Mesh = DistributedMesh;
}

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Communication schedule for ghost exchange
 *
 * Describes what DOFs to send to/receive from each neighbor rank.
 */
struct CommSchedule {
    std::vector<int> neighbor_ranks;                    ///< Sorted neighbor ranks
    std::vector<std::vector<GlobalIndex>> send_lists;   ///< DOFs to send per neighbor
    std::vector<std::vector<GlobalIndex>> recv_lists;   ///< DOFs to receive per neighbor

    /**
     * @brief Get total number of DOFs to send
     */
    [[nodiscard]] std::size_t totalSendCount() const noexcept {
        std::size_t count = 0;
        for (const auto& list : send_lists) count += list.size();
        return count;
    }

    /**
     * @brief Get total number of DOFs to receive
     */
    [[nodiscard]] std::size_t totalRecvCount() const noexcept {
        std::size_t count = 0;
        for (const auto& list : recv_lists) count += list.size();
        return count;
    }

    /**
     * @brief Check if schedule is empty (no communication needed)
     */
    [[nodiscard]] bool empty() const noexcept {
        return neighbor_ranks.empty();
    }
};

/**
 * @brief Ownership resolution strategies for shared DOFs
 */
enum class OwnerResolutionStrategy : std::uint8_t {
    LowestRank,     ///< Lowest rank touching the DOF owns it
    HighestRank,    ///< Highest rank touching the DOF owns it
    MinVertexGID,   ///< Owner based on minimum vertex GID (deterministic)
    Custom          ///< Custom function
};

/**
 * @brief Manager for ghost and shared DOFs in parallel
 *
 * This class identifies and manages DOFs that need to be communicated
 * between MPI ranks. It distinguishes between:
 * - **Shared DOFs**: On partition boundaries, appear on multiple ranks
 * - **Ghost DOFs**: Owned by other ranks, needed locally
 *
 * The manager builds communication schedules that can be used with
 * MPI (or any other communication layer) to synchronize DOF values.
 */
class GhostDofManager {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    GhostDofManager();
    ~GhostDofManager();

    // Move semantics
    GhostDofManager(GhostDofManager&&) noexcept;
    GhostDofManager& operator=(GhostDofManager&&) noexcept;

    // No copy
    GhostDofManager(const GhostDofManager&) = delete;
    GhostDofManager& operator=(const GhostDofManager&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * @brief Set local MPI rank for ownership queries (mesh-independent path)
     *
     * Required when using setGhostDofs()/addSharedDofsWithNeighbor() without
     * identifyGhostDofs()/identifySharedDofs() (which otherwise populate rank info).
     */
    void setMyRank(int my_rank) noexcept { my_rank_ = my_rank; }

    /**
     * @brief Provide locally-owned DOF ordering for packed-owned vector layouts
     *
     * Builds a stable map global_dof -> local_owned_index in the given ordering.
     * If provided, syncGhostValues() will use it to pack send buffers.
     */
    void setOwnedDofs(std::span<const GlobalIndex> owned_dofs);

    /**
     * @brief Map an owned global DOF to its packed-owned index (if available)
     */
    [[nodiscard]] std::optional<std::size_t> ownedIndex(GlobalIndex dof) const noexcept;

    /**
     * @brief Map a ghost global DOF to its ghost-buffer index (after buildGhostExchange)
     */
    [[nodiscard]] std::optional<std::size_t> ghostIndex(GlobalIndex dof) const noexcept;

    /**
     * @brief Identify ghost DOFs from distributed mesh
     *
     * @param mesh The distributed mesh
     * @param dof_map The DOF map
     * @throws FEException if mesh/dof_map is inconsistent
     */
    void identifyGhostDofs(const Mesh& mesh, const DofMap& dof_map);

    /**
     * @brief Identify shared DOFs on partition boundaries
     *
     * @param mesh The distributed mesh
     * @param dof_map The DOF map
     * @throws FEException if mesh/dof_map is inconsistent
     */
    void identifySharedDofs(const Mesh& mesh, const DofMap& dof_map);

    /**
     * @brief Manually set ghost DOFs (for testing/custom setups)
     *
     * @param ghost_dofs Global indices of ghost DOFs
     * @param owners Owning ranks for each ghost DOF
     */
    void setGhostDofs(std::span<const GlobalIndex> ghost_dofs,
                      std::span<const int> owners);

    /**
     * @brief Manually set shared DOFs by neighbor
     *
     * @param neighbor_rank The neighbor rank
     * @param shared_dofs DOFs shared with this neighbor
     */
    void addSharedDofsWithNeighbor(int neighbor_rank,
                                   std::span<const GlobalIndex> shared_dofs);

    /**
     * @brief Resolve ownership of shared DOFs
     *
     * @param strategy Resolution strategy
     * @param custom_func Custom function for Custom strategy
     */
    void resolveSharedOwnership(
        OwnerResolutionStrategy strategy = OwnerResolutionStrategy::LowestRank,
        std::function<int(GlobalIndex, const std::vector<int>&)> custom_func = nullptr);

    // =========================================================================
    // Communication Schedule Building
    // =========================================================================

    /**
     * @brief Build ghost exchange communication schedule
     *
     * Creates send/receive lists for synchronizing ghost DOF values.
     * Call after identifyGhostDofs() and resolveSharedOwnership().
     */
    void buildGhostExchange();

    /**
     * @brief Get the communication schedule
     */
    [[nodiscard]] const CommSchedule& getCommSchedule() const noexcept {
        return comm_schedule_;
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * @brief Get all ghost DOFs
     */
    [[nodiscard]] const IndexSet& getGhostDofs() const noexcept {
        return ghost_dofs_;
    }

    /**
     * @brief Get all shared DOFs (union across all neighbors)
     */
    [[nodiscard]] const IndexSet& getSharedDofs() const noexcept {
        return shared_dofs_;
    }

    /**
     * @brief Get shared DOFs with a specific neighbor
     *
     * @param rank Neighbor rank
     * @return DOFs shared with this rank, empty if not a neighbor
     */
    [[nodiscard]] std::span<const GlobalIndex> getSharedDofsByNeighbor(int rank) const;

    /**
     * @brief Get owner rank for a DOF
     *
     * @param dof Global DOF index
     * @return Owning rank, or -1 if unknown
     */
    [[nodiscard]] int getDofOwner(GlobalIndex dof) const;

    /**
     * @brief Check if a DOF is a ghost (not owned locally)
     */
    [[nodiscard]] bool isGhost(GlobalIndex dof) const noexcept {
        return ghost_dofs_.contains(dof);
    }

    /**
     * @brief Check if a DOF is shared with other ranks
     */
    [[nodiscard]] bool isShared(GlobalIndex dof) const noexcept {
        return shared_dofs_.contains(dof);
    }

    /**
     * @brief Get set of neighbor ranks
     */
    [[nodiscard]] const std::unordered_set<int>& getNeighborRanks() const noexcept {
        return neighbor_ranks_;
    }

    /**
     * @brief Get send schedule for ghost exchange
     *
     * Returns (neighbor_ranks, dof_lists_per_neighbor).
     */
    [[nodiscard]] std::pair<std::span<const int>,
                            std::span<const std::vector<GlobalIndex>>>
    getSendSchedule() const noexcept;

    /**
     * @brief Get receive schedule for ghost exchange
     */
    [[nodiscard]] std::pair<std::span<const int>,
                            std::span<const std::vector<GlobalIndex>>>
    getRecvSchedule() const noexcept;

    // =========================================================================
    // Synchronization (high-level interface)
    // =========================================================================

    /**
     * @brief Callback type for performing communication
     *
     * @param send_rank Rank to send to
     * @param send_data Data to send (DOF values)
     * @param recv_rank Rank to receive from
     * @param recv_data Buffer for received data
     */
    using CommCallback = std::function<void(
        int send_rank, std::span<const double> send_data,
        int recv_rank, std::span<double> recv_data)>;

    /**
     * @brief Synchronize ghost values using provided communication callback
     *
     * @param local_values Local DOF values. If setOwnedDofs() was called, this
     *        is interpreted as a packed-owned vector in that ordering. Otherwise
     *        it is treated as globally indexed (debug-only, not backend-friendly).
     * @param ghost_values Ghost DOF values to fill (indexed by ghost order)
     * @param comm_func Communication callback
     *
     * This method orchestrates the ghost exchange but delegates actual
     * communication to the provided callback (which can use MPI, etc.).
     */
    void syncGhostValues(std::span<const double> local_values,
                         std::span<double> ghost_values,
                         CommCallback comm_func) const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get ghost/shared DOF statistics
     */
    struct Statistics {
        GlobalIndex n_ghost_dofs{0};
        GlobalIndex n_shared_dofs{0};
        std::size_t n_neighbors{0};
        std::size_t total_send_count{0};
        std::size_t total_recv_count{0};
    };

    [[nodiscard]] Statistics getStatistics() const;

private:
    // Ghost DOF storage
    IndexSet ghost_dofs_;
    std::unordered_map<GlobalIndex, int> ghost_owners_;  // DOF -> owning rank
    std::unordered_map<GlobalIndex, std::size_t> ghost_dof_to_index_;  // DOF -> index in ghost_dofs_ ordering
    std::unordered_map<GlobalIndex, std::size_t> owned_dof_to_index_;  // DOF -> index in packed-owned ordering

    // Shared DOF storage
    IndexSet shared_dofs_;
    std::unordered_map<int, std::vector<GlobalIndex>> shared_by_neighbor_;  // rank -> DOFs

    // Neighbor information
    std::unordered_set<int> neighbor_ranks_;

    // Communication schedule
    CommSchedule comm_schedule_;

    // Local rank info
    int my_rank_{0};
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_GHOSTDOFMANAGER_H
