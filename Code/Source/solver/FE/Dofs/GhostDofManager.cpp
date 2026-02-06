/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "GhostDofManager.h"

// This module can be built without the Mesh library. Use an explicit compile
// definition when available; __has_include is not sufficient when headers are
// present but the Mesh library is not linked (e.g., FE standalone builds).
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#  define GHOSTMANAGER_HAS_MESH 1
#else
#  define GHOSTMANAGER_HAS_MESH 0
#endif

#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

GhostDofManager::GhostDofManager() = default;
GhostDofManager::~GhostDofManager() = default;

GhostDofManager::GhostDofManager(GhostDofManager&&) noexcept = default;
GhostDofManager& GhostDofManager::operator=(GhostDofManager&&) noexcept = default;

// =============================================================================
// Initialization
// =============================================================================

#if GHOSTMANAGER_HAS_MESH

void GhostDofManager::identifyGhostDofs(const Mesh& mesh,
                                         const DofMap& dof_map) {
    my_rank_ = mesh.rank();

    std::vector<GlobalIndex> ghost_dof_list;
    std::unordered_set<GlobalIndex> seen_dofs;

    // Iterate over ghost cells and collect their DOFs
    auto n_cells = static_cast<GlobalIndex>(mesh.local_mesh().n_cells());

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        if (mesh.is_ghost_cell(static_cast<index_t>(c))) {
            // This is a ghost cell - all its DOFs are potentially ghost DOFs
            auto cell_dofs = dof_map.getCellDofs(c);
            int owner_rank = mesh.owner_rank_cell(static_cast<index_t>(c));

            for (auto dof : cell_dofs) {
                if (seen_dofs.insert(dof).second) {
                    ghost_dof_list.push_back(dof);
                    ghost_owners_[dof] = owner_rank;
                    neighbor_ranks_.insert(owner_rank);
                }
            }
        }
    }

    // Build index set
    ghost_dofs_ = IndexSet(std::move(ghost_dof_list));
}

void GhostDofManager::identifySharedDofs(const Mesh& mesh,
                                          const DofMap& dof_map) {
    my_rank_ = mesh.rank();

    // Shared DOFs are on cells that touch partition boundaries
    // A DOF is shared if it appears on both owned and ghost cells

    std::unordered_set<GlobalIndex> owned_cell_dofs;
    std::unordered_set<GlobalIndex> ghost_cell_dofs;

    auto n_cells = static_cast<GlobalIndex>(mesh.local_mesh().n_cells());

    // Collect DOFs from owned vs ghost cells
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto cell_dofs = dof_map.getCellDofs(c);

        if (mesh.is_owned_cell(static_cast<index_t>(c))) {
            for (auto dof : cell_dofs) {
                owned_cell_dofs.insert(dof);
            }
        } else if (mesh.is_ghost_cell(static_cast<index_t>(c))) {
            int owner_rank = mesh.owner_rank_cell(static_cast<index_t>(c));
            for (auto dof : cell_dofs) {
                ghost_cell_dofs.insert(dof);
                // Track which neighbor shares this DOF
                shared_by_neighbor_[owner_rank].push_back(dof);
            }
            neighbor_ranks_.insert(owner_rank);
        }
    }

    // Shared DOFs are in the intersection
    std::vector<GlobalIndex> shared_list;
    for (auto dof : owned_cell_dofs) {
        if (ghost_cell_dofs.count(dof) > 0) {
            shared_list.push_back(dof);
        }
    }

    shared_dofs_ = IndexSet(std::move(shared_list));

    // Deduplicate shared_by_neighbor lists
    for (auto& [rank, dofs] : shared_by_neighbor_) {
        std::sort(dofs.begin(), dofs.end());
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    }
}

#else

void GhostDofManager::identifyGhostDofs(const Mesh& /*mesh*/,
                                         const DofMap& /*dof_map*/) {
    throw FEException("GhostDofManager: Mesh library not available");
}

void GhostDofManager::identifySharedDofs(const Mesh& /*mesh*/,
                                          const DofMap& /*dof_map*/) {
    throw FEException("GhostDofManager: Mesh library not available");
}

#endif // GHOSTMANAGER_HAS_MESH

void GhostDofManager::setGhostDofs(std::span<const GlobalIndex> ghost_dofs,
                                    std::span<const int> owners) {
    if (ghost_dofs.size() != owners.size()) {
        throw FEException("GhostDofManager::setGhostDofs: size mismatch");
    }

    ghost_owners_.clear();
    ghost_dof_to_index_.clear();
    owned_dof_to_index_.clear();
    shared_by_neighbor_.clear();
    shared_dofs_ = IndexSet{};
    neighbor_ranks_.clear();
    comm_schedule_ = CommSchedule{};

    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        ghost_owners_[ghost_dofs[i]] = owners[i];
        neighbor_ranks_.insert(owners[i]);
    }

    ghost_dofs_ = IndexSet(std::vector<GlobalIndex>(ghost_dofs.begin(), ghost_dofs.end()));
}

void GhostDofManager::setOwnedDofs(std::span<const GlobalIndex> owned_dofs) {
    owned_dof_to_index_.clear();
    if (owned_dofs.empty()) {
        return;
    }

    owned_dof_to_index_.reserve(owned_dofs.size());
    for (std::size_t i = 0; i < owned_dofs.size(); ++i) {
        owned_dof_to_index_.emplace(owned_dofs[i], i);
    }
}

std::optional<std::size_t> GhostDofManager::ownedIndex(GlobalIndex dof) const noexcept {
    const auto it = owned_dof_to_index_.find(dof);
    if (it == owned_dof_to_index_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<std::size_t> GhostDofManager::ghostIndex(GlobalIndex dof) const noexcept {
    const auto it = ghost_dof_to_index_.find(dof);
    if (it == ghost_dof_to_index_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void GhostDofManager::addSharedDofsWithNeighbor(int neighbor_rank,
                                                 std::span<const GlobalIndex> shared_dofs) {
    auto& list = shared_by_neighbor_[neighbor_rank];
    list.insert(list.end(), shared_dofs.begin(), shared_dofs.end());

    // Deduplicate
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());

    neighbor_ranks_.insert(neighbor_rank);

    // Rebuild shared_dofs_ index set
    std::vector<GlobalIndex> all_shared;
    for (const auto& [rank, dofs] : shared_by_neighbor_) {
        all_shared.insert(all_shared.end(), dofs.begin(), dofs.end());
    }
    shared_dofs_ = IndexSet(std::move(all_shared));

    // Invalidate any previously-built schedules; caller must rebuild.
    comm_schedule_ = CommSchedule{};
}

void GhostDofManager::resolveSharedOwnership(
    OwnerResolutionStrategy strategy,
    std::function<int(GlobalIndex, const std::vector<int>&)> custom_func) {

    // Ownership changes invalidate any previously-built schedules.
    comm_schedule_ = CommSchedule{};
    ghost_dof_to_index_.clear();

    // For shared DOFs, determine the owner
    // This affects which rank is responsible for the DOF value

    for (auto dof : shared_dofs_) {
        // Collect all ranks that have this DOF
        std::vector<int> touching_ranks;
        touching_ranks.push_back(my_rank_);  // We have it

        for (const auto& [rank, dofs] : shared_by_neighbor_) {
            if (std::binary_search(dofs.begin(), dofs.end(), dof)) {
                touching_ranks.push_back(rank);
            }
        }

        std::sort(touching_ranks.begin(), touching_ranks.end());

        int owner = my_rank_;

        switch (strategy) {
            case OwnerResolutionStrategy::LowestRank:
                owner = touching_ranks.front();
                break;

            case OwnerResolutionStrategy::HighestRank:
                owner = touching_ranks.back();
                break;

            case OwnerResolutionStrategy::MinVertexGID:
                // For vertex-based DOFs, owner is rank with minimum vertex GID
                // This requires vertex GID info which we don't have here
                // Fall back to lowest rank
                owner = touching_ranks.front();
                break;

            case OwnerResolutionStrategy::Custom:
                if (custom_func) {
                    owner = custom_func(dof, touching_ranks);
                } else {
                    owner = touching_ranks.front();
                }
                break;
        }

        // ghost_owners_ is the authoritative map for all DOFs not owned by this rank.
        // If we own a shared DOF, ensure it is not marked as a ghost.
        if (owner == my_rank_) {
            ghost_owners_.erase(dof);
        } else {
            ghost_owners_[dof] = owner;
        }
    }

    // Rebuild ghost_dofs_ to match the resolved ownership map.
    if (!ghost_owners_.empty()) {
        std::vector<GlobalIndex> ghost_list;
        ghost_list.reserve(ghost_owners_.size());
        for (const auto& [dof, owner] : ghost_owners_) {
            if (owner != my_rank_) {
                ghost_list.push_back(dof);
            }
        }
        ghost_dofs_ = IndexSet(std::move(ghost_list));
    } else {
        ghost_dofs_ = IndexSet{};
    }
}

// =============================================================================
// Communication Schedule Building
// =============================================================================

void GhostDofManager::buildGhostExchange() {
    comm_schedule_.neighbor_ranks.clear();
    comm_schedule_.send_lists.clear();
    comm_schedule_.recv_lists.clear();
    ghost_dof_to_index_.clear();

    // Sort neighbor ranks for deterministic ordering
    std::vector<int> sorted_neighbors(neighbor_ranks_.begin(), neighbor_ranks_.end());
    std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

    comm_schedule_.neighbor_ranks = sorted_neighbors;
    comm_schedule_.send_lists.resize(sorted_neighbors.size());
    comm_schedule_.recv_lists.resize(sorted_neighbors.size());

    // Build send lists: DOFs we own that neighbors need
    // Build recv lists: DOFs neighbors own that we need (ghost DOFs)

    for (std::size_t i = 0; i < sorted_neighbors.size(); ++i) {
        int neighbor = sorted_neighbors[i];

        // Recv list: ghost DOFs owned by this neighbor
        for (const auto& [dof, owner] : ghost_owners_) {
            if (owner == neighbor) {
                comm_schedule_.recv_lists[i].push_back(dof);
            }
        }

        // Send list: shared DOFs with this neighbor that we own
        auto it = shared_by_neighbor_.find(neighbor);
        if (it != shared_by_neighbor_.end()) {
            for (auto dof : it->second) {
                // Check if we own this DOF
                auto owner_it = ghost_owners_.find(dof);
                if (owner_it == ghost_owners_.end()) {
                    // Not in ghost_owners_ means we own it
                    comm_schedule_.send_lists[i].push_back(dof);
                }
            }
        }

        // Sort lists for deterministic communication
        std::sort(comm_schedule_.send_lists[i].begin(), comm_schedule_.send_lists[i].end());
        std::sort(comm_schedule_.recv_lists[i].begin(), comm_schedule_.recv_lists[i].end());
    }

    // Build a stable DOF -> ghost-buffer index map. Ghost buffers are expected
    // to be ordered according to ghost_dofs_ iteration order.
    if (!ghost_dofs_.empty()) {
        const auto n_ghost = static_cast<std::size_t>(ghost_dofs_.size());
        ghost_dof_to_index_.reserve(n_ghost);
        std::size_t idx = 0;
        for (auto dof : ghost_dofs_) {
            ghost_dof_to_index_.emplace(dof, idx++);
        }
        if (idx != n_ghost) {
            throw FEException("GhostDofManager::buildGhostExchange: failed to build ghost index map");
        }
    }
}

// =============================================================================
// Query Methods
// =============================================================================

std::span<const GlobalIndex> GhostDofManager::getSharedDofsByNeighbor(int rank) const {
    auto it = shared_by_neighbor_.find(rank);
    if (it != shared_by_neighbor_.end()) {
        return it->second;
    }
    return {};
}

int GhostDofManager::getDofOwner(GlobalIndex dof) const {
    auto it = ghost_owners_.find(dof);
    if (it != ghost_owners_.end()) {
        return it->second;
    }
    return my_rank_;  // We own it
}

std::pair<std::span<const int>, std::span<const std::vector<GlobalIndex>>>
GhostDofManager::getSendSchedule() const noexcept {
    return {comm_schedule_.neighbor_ranks, comm_schedule_.send_lists};
}

std::pair<std::span<const int>, std::span<const std::vector<GlobalIndex>>>
GhostDofManager::getRecvSchedule() const noexcept {
    return {comm_schedule_.neighbor_ranks, comm_schedule_.recv_lists};
}

// =============================================================================
// Synchronization
// =============================================================================

void GhostDofManager::syncGhostValues(std::span<const double> local_values,
                                       std::span<double> ghost_values,
                                       CommCallback comm_func) const {
    if (comm_schedule_.empty()) return;
    if (!comm_func) {
        throw FEException("GhostDofManager::syncGhostValues: comm_func is empty");
    }

    if (ghost_values.size() != static_cast<std::size_t>(ghost_dofs_.size())) {
        throw FEException("GhostDofManager::syncGhostValues: ghost_values size does not match ghost DOF count");
    }
    if (!ghost_dofs_.empty() && ghost_dof_to_index_.size() != static_cast<std::size_t>(ghost_dofs_.size())) {
        throw FEException("GhostDofManager::syncGhostValues: ghost index map not built; call buildGhostExchange() after setting ghost DOFs");
    }

    // This is a high-level interface - actual MPI calls are in comm_func
    // We just orchestrate the data packing/unpacking

    thread_local std::vector<std::vector<double>> send_buffers;
    thread_local std::vector<std::vector<double>> recv_buffers;
    if (send_buffers.size() != comm_schedule_.neighbor_ranks.size()) {
        send_buffers.resize(comm_schedule_.neighbor_ranks.size());
        recv_buffers.resize(comm_schedule_.neighbor_ranks.size());
    }

    for (std::size_t i = 0; i < comm_schedule_.neighbor_ranks.size(); ++i) {
        int neighbor = comm_schedule_.neighbor_ranks[i];
        const auto& send_list = comm_schedule_.send_lists[i];
        const auto& recv_list = comm_schedule_.recv_lists[i];

        // Pack send data
        auto& send_data = send_buffers[i];
        send_data.resize(send_list.size());
        for (std::size_t j = 0; j < send_list.size(); ++j) {
            const auto dof = send_list[j];
            if (!owned_dof_to_index_.empty()) {
                const auto it = owned_dof_to_index_.find(dof);
                if (it == owned_dof_to_index_.end()) {
                    throw FEException("GhostDofManager::syncGhostValues: send DOF not present in owned ordering map");
                }
                const auto idx = it->second;
                if (idx >= local_values.size()) {
                    throw FEException("GhostDofManager::syncGhostValues: local_values too small for owned ordering map");
                }
                send_data[j] = local_values[idx];
            } else {
                // Fallback: interpret local_values as globally indexed.
                if (dof < 0 || dof >= static_cast<GlobalIndex>(local_values.size())) {
                    throw FEException("GhostDofManager::syncGhostValues: local_values does not provide global-indexed access for send DOF");
                }
                send_data[j] = local_values[static_cast<std::size_t>(dof)];
            }
        }

        // Reuse recv buffer for this neighbor.
        auto& recv_data = recv_buffers[i];
        recv_data.resize(recv_list.size());

        // Call user-provided communication
        comm_func(neighbor, send_data, neighbor, recv_data);

        // Unpack recv data into ghost_values
        // Note: ghost_values is indexed by ghost ordering in ghost_dofs_.
        for (std::size_t j = 0; j < recv_list.size(); ++j) {
            const auto dof = recv_list[j];
            const auto it = ghost_dof_to_index_.find(dof);
            if (it == ghost_dof_to_index_.end()) {
                throw FEException("GhostDofManager::syncGhostValues: received DOF not present in ghost set");
            }
            const auto idx = it->second;
            if (idx >= ghost_values.size()) {
                throw FEException("GhostDofManager::syncGhostValues: ghost index out of range");
            }
            ghost_values[idx] = recv_data[j];
        }
    }
}

// =============================================================================
// Statistics
// =============================================================================

GhostDofManager::Statistics GhostDofManager::getStatistics() const {
    Statistics stats;
    stats.n_ghost_dofs = ghost_dofs_.size();
    stats.n_shared_dofs = shared_dofs_.size();
    stats.n_neighbors = neighbor_ranks_.size();
    stats.total_send_count = comm_schedule_.totalSendCount();
    stats.total_recv_count = comm_schedule_.totalRecvCount();
    return stats;
}

} // namespace dofs
} // namespace FE
} // namespace svmp
