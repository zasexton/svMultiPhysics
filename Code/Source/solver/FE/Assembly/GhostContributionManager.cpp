/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "GhostContributionManager.h"
#include "Dofs/DofMap.h"
#include "Dofs/GhostDofManager.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstring>

namespace svmp {
namespace FE {
namespace assembly {

#if FE_HAS_MPI
namespace {
void set_mpi_rank_and_size(MPI_Comm comm, int& rank, int& size)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    } else {
        rank = 0;
        size = 1;
    }
}
} // namespace
#endif

// ============================================================================
// Construction
// ============================================================================

GhostContributionManager::GhostContributionManager() = default;

GhostContributionManager::GhostContributionManager(const dofs::DofMap& dof_map)
    : dof_map_(&dof_map)
{
}

#if FE_HAS_MPI
GhostContributionManager::GhostContributionManager(const dofs::DofMap& dof_map, MPI_Comm comm)
    : dof_map_(&dof_map)
    , comm_(comm)
{
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
}
#endif

GhostContributionManager::~GhostContributionManager() = default;

GhostContributionManager::GhostContributionManager(GhostContributionManager&& other) noexcept = default;

GhostContributionManager& GhostContributionManager::operator=(GhostContributionManager&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void GhostContributionManager::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
    initialized_ = false;
}

void GhostContributionManager::setGhostDofManager(const dofs::GhostDofManager& ghost_manager)
{
    ghost_dof_manager_ = &ghost_manager;
    initialized_ = false;
}

#if FE_HAS_MPI
void GhostContributionManager::setComm(MPI_Comm comm)
{
    comm_ = comm;
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
    initialized_ = false;
}
#endif

// ============================================================================
// Initialization
// ============================================================================

void GhostContributionManager::initialize()
{
    if (!dof_map_) {
        throw std::runtime_error("GhostContributionManager: DOF map not set");
    }

#if FE_HAS_MPI
    // Ensure rank/size are consistent even when constructed without explicit setComm().
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
#endif

    // Build communication graph (determine neighbors)
    buildCommunicationGraph();

    // Allocate send buffers
    send_buffers_.resize(neighbor_ranks_.size());
    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        send_buffers_[i].dest_rank = neighbor_ranks_[i];
    }

#if FE_HAS_MPI
    recv_buffers_raw_.resize(neighbor_ranks_.size());
    send_requests_.resize(neighbor_ranks_.size(), MPI_REQUEST_NULL);
    recv_requests_.resize(neighbor_ranks_.size(), MPI_REQUEST_NULL);
#endif

    initialized_ = true;
}

void GhostContributionManager::buildCommunicationGraph()
{
    neighbor_ranks_.clear();
    rank_to_neighbor_idx_.clear();
    ghost_owner_rank_.clear();

    // Determine ghost DOFs and their owners.
    //
    // Prefer the GhostDofManager when it matches the currently configured DOF map.
    // In multi-field/block workflows, callers may keep a system-level GhostDofManager
    // while switching the row DofMap; in that case the GhostDofManager's indices may
    // be out of range for the row DofMap and we must fall back to the DofMap itself.
    bool can_use_ghost_manager = (ghost_dof_manager_ != nullptr);
    if (can_use_ghost_manager) {
        FE_CHECK_NOT_NULL(dof_map_, "GhostContributionManager::buildCommunicationGraph: dof_map");
        const auto n = dof_map_->getNumDofs();
        const auto& ghost_dofs = ghost_dof_manager_->getGhostDofs();
        for (const auto g : ghost_dofs) {
            if (g < 0 || g >= n) {
                can_use_ghost_manager = false;
                break;
            }
        }
    }

    if (can_use_ghost_manager) {
        const auto& ghost_dofs = ghost_dof_manager_->getGhostDofs();

        for (GlobalIndex ghost : ghost_dofs) {
            int owner = ghost_dof_manager_->getDofOwner(ghost);
            ghost_owner_rank_[ghost] = owner;

            // Track unique neighbor ranks
            if (rank_to_neighbor_idx_.find(owner) == rank_to_neighbor_idx_.end()) {
                rank_to_neighbor_idx_[owner] = static_cast<int>(neighbor_ranks_.size());
                neighbor_ranks_.push_back(owner);
            }
        }
    } else {
        // Fallback: use DofMap's ownership info
        // Iterate over all cells and check DOF ownership
        GlobalIndex n_cells = dof_map_->getNumCells();

        for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
            auto cell_dofs = dof_map_->getCellDofs(cell);
            for (GlobalIndex dof : cell_dofs) {
                if (!dof_map_->isOwnedDof(dof)) {
                    // It's a ghost - determine owner
                    int owner = dof_map_->getDofOwner(dof);
                    if (owner < 0) {
                        // Fallback heuristic if owner unknown
                        owner = static_cast<int>(dof % world_size_);
                    }
                    ghost_owner_rank_[dof] = owner;

                    if (rank_to_neighbor_idx_.find(owner) == rank_to_neighbor_idx_.end()) {
                        rank_to_neighbor_idx_[owner] = static_cast<int>(neighbor_ranks_.size());
                        neighbor_ranks_.push_back(owner);
                    }
                }
            }
        }
    }

#if FE_HAS_MPI
    // The point-to-point exchange in exchangeContributions() requires a symmetric neighbor list.
    // Building neighbors from "owners of my ghost DOFs" is not symmetric in general (e.g. when this
    // rank owns all shared interface DOFs, it may have no ghost DOFs owned by the neighbor, yet the
    // neighbor must still send contributions to this rank). Symmetrize the communication graph via
    // a lightweight all-to-all handshake on the computed send targets.
    if (world_size_ > 1) {
        std::vector<int> send_flags(static_cast<std::size_t>(world_size_), 0);
        for (const int r : neighbor_ranks_) {
            if (r >= 0 && r < world_size_ && r != my_rank_) {
                send_flags[static_cast<std::size_t>(r)] = 1;
            }
        }

        std::vector<int> recv_flags(static_cast<std::size_t>(world_size_), 0);
        MPI_Alltoall(send_flags.data(), 1, MPI_INT,
                     recv_flags.data(), 1, MPI_INT,
                     comm_);

        std::vector<int> symmetric;
        symmetric.reserve(static_cast<std::size_t>(world_size_));
        for (int r = 0; r < world_size_; ++r) {
            if (r == my_rank_) {
                continue;
            }
            if (send_flags[static_cast<std::size_t>(r)] || recv_flags[static_cast<std::size_t>(r)]) {
                symmetric.push_back(r);
            }
        }
        neighbor_ranks_ = std::move(symmetric);
    }
#endif

    // Sort neighbors for deterministic communication order
    std::sort(neighbor_ranks_.begin(), neighbor_ranks_.end());

    // Rebuild rank to index mapping after sort
    rank_to_neighbor_idx_.clear();
    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        rank_to_neighbor_idx_[neighbor_ranks_[i]] = static_cast<int>(i);
    }
}

// ============================================================================
// Contribution Accumulation
// ============================================================================

bool GhostContributionManager::addMatrixContribution(
    GlobalIndex global_row,
    GlobalIndex global_col,
    Real value)
{
    if (isOwned(global_row)) {
        return true;  // Locally owned, caller inserts directly
    }

    // Ghost row - buffer for later exchange
    if (policy_ == GhostPolicy::OwnedRowsOnly) {
        return false;  // Discard ghost contributions
    }

    // Find owner and add to buffer
    int owner = getOwnerRank(global_row);
    int neighbor_idx = findNeighborIndex(owner);

    if (neighbor_idx >= 0) {
        send_buffers_[static_cast<std::size_t>(neighbor_idx)].entries.push_back(
            {global_row, global_col, value});
    }

    return false;
}

bool GhostContributionManager::addVectorContribution(
    GlobalIndex global_row,
    Real value)
{
    if (isOwned(global_row)) {
        return true;  // Locally owned, caller inserts directly
    }

    if (policy_ == GhostPolicy::OwnedRowsOnly) {
        return false;
    }

    int owner = getOwnerRank(global_row);
    int neighbor_idx = findNeighborIndex(owner);

    if (neighbor_idx >= 0) {
        auto& buffer = send_buffers_[static_cast<std::size_t>(neighbor_idx)];
        buffer.vector_entries.push_back({global_row, value});
    }

    return false;
}

void GhostContributionManager::addMatrixContributions(
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    std::span<const Real> values,
    std::vector<GhostContribution>& owned_contributions)
{
    const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
    const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

    owned_contributions.clear();

    for (GlobalIndex i = 0; i < n_rows; ++i) {
        GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
        bool is_owned = isOwned(row);

        for (GlobalIndex j = 0; j < n_cols; ++j) {
            GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
            Real val = values[static_cast<std::size_t>(i * n_cols + j)];

            if (is_owned) {
                owned_contributions.push_back({row, col, val});
            } else if (policy_ == GhostPolicy::ReverseScatter) {
                int owner = getOwnerRank(row);
                int neighbor_idx = findNeighborIndex(owner);
                if (neighbor_idx >= 0) {
                    send_buffers_[static_cast<std::size_t>(neighbor_idx)].entries.push_back(
                        {row, col, val});
                }
            }
        }
    }
}

bool GhostContributionManager::isOwned(GlobalIndex global_dof) const
{
    FE_CHECK_NOT_NULL(dof_map_, "GhostContributionManager::isOwned: dof_map");
    FE_THROW_IF(global_dof < ownership_offset_, FEException,
                "GhostContributionManager::isOwned: global DOF index is below ownershipOffset()");
    const auto local = global_dof - ownership_offset_;
    return dof_map_->isOwnedDof(local);
}

int GhostContributionManager::getOwnerRank(GlobalIndex global_dof) const
{
    FE_CHECK_NOT_NULL(dof_map_, "GhostContributionManager::getOwnerRank: dof_map");
    FE_THROW_IF(global_dof < ownership_offset_, FEException,
                "GhostContributionManager::getOwnerRank: global DOF index is below ownershipOffset()");
    const auto local = global_dof - ownership_offset_;

    auto it = ghost_owner_rank_.find(local);
    if (it != ghost_owner_rank_.end()) {
        return it->second;
    }

    // Fall back to the DOF map ownership function if available.
    const int owner = dof_map_->getDofOwner(local);
    return owner >= 0 ? owner : my_rank_;
}

int GhostContributionManager::findNeighborIndex(int rank) const
{
    auto it = rank_to_neighbor_idx_.find(rank);
    if (it != rank_to_neighbor_idx_.end()) {
        return it->second;
    }
    return -1;
}

// ============================================================================
// Communication
// ============================================================================

void GhostContributionManager::exchangeContributions()
{
    if (world_size_ == 1 || neighbor_ranks_.empty()) {
        return;  // Nothing to exchange in serial
    }

    auto start_time = std::chrono::steady_clock::now();

    if (deterministic_) {
        sortBuffersForDeterminism();
    }

#if FE_HAS_MPI
    // Pack send buffers into byte arrays
    std::vector<std::vector<char>> send_data(neighbor_ranks_.size());

    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        const auto& buffer = send_buffers_[i];

        // Calculate packed size
        std::size_t matrix_size = buffer.entries.size() * sizeof(GhostContribution);
        std::size_t vector_size = buffer.vector_entries.size() * sizeof(GhostVectorContribution);
        std::size_t total_size =
            sizeof(std::size_t) + matrix_size +
            sizeof(std::size_t) + vector_size;

        send_data[i].resize(total_size);

        // Pack: [num_matrix][matrix...][num_vector][vector...]
        char* ptr = send_data[i].data();
        std::size_t num_entries = buffer.entries.size();
        std::memcpy(ptr, &num_entries, sizeof(std::size_t));
        ptr += sizeof(std::size_t);

        if (!buffer.entries.empty()) {
            std::memcpy(ptr, buffer.entries.data(), matrix_size);
        }
        ptr += matrix_size;

        std::size_t num_vector_entries = buffer.vector_entries.size();
        std::memcpy(ptr, &num_vector_entries, sizeof(std::size_t));
        ptr += sizeof(std::size_t);
        if (!buffer.vector_entries.empty()) {
            std::memcpy(ptr, buffer.vector_entries.data(), vector_size);
        }

        last_stats_.bytes_sent += send_data[i].size();
        last_stats_.matrix_entries_sent += buffer.entries.size();
        last_stats_.vector_entries_sent += buffer.vector_entries.size();
    }

    // Exchange sizes first
    std::vector<int> send_sizes(neighbor_ranks_.size());
    std::vector<int> recv_sizes(neighbor_ranks_.size());

    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        send_sizes[i] = static_cast<int>(send_data[i].size());
    }

    std::vector<MPI_Request> size_requests(2 * neighbor_ranks_.size());

    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        MPI_Isend(&send_sizes[i], 1, MPI_INT, neighbor_ranks_[i], 0, comm_,
                  &size_requests[i]);
        MPI_Irecv(&recv_sizes[i], 1, MPI_INT, neighbor_ranks_[i], 0, comm_,
                  &size_requests[neighbor_ranks_.size() + i]);
    }

    MPI_Waitall(static_cast<int>(size_requests.size()), size_requests.data(),
                MPI_STATUSES_IGNORE);

    // Allocate receive buffers
    recv_buffers_raw_.resize(neighbor_ranks_.size());
    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        recv_buffers_raw_[i].resize(static_cast<std::size_t>(recv_sizes[i]));
        last_stats_.bytes_received += static_cast<std::size_t>(recv_sizes[i]);
    }

    // Exchange data
    std::vector<MPI_Request> data_requests(2 * neighbor_ranks_.size());

    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        MPI_Isend(send_data[i].data(), static_cast<int>(send_data[i].size()),
                  MPI_CHAR, neighbor_ranks_[i], 1, comm_, &data_requests[i]);
        MPI_Irecv(recv_buffers_raw_[i].data(), recv_sizes[i], MPI_CHAR,
                  neighbor_ranks_[i], 1, comm_,
                  &data_requests[neighbor_ranks_.size() + i]);
    }

    MPI_Waitall(static_cast<int>(data_requests.size()), data_requests.data(),
                MPI_STATUSES_IGNORE);

    // Unpack received data
    received_matrix_.clear();
    received_vector_.clear();

    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        if (recv_buffers_raw_[i].empty()) continue;

        const char* ptr = recv_buffers_raw_[i].data();

        std::size_t num_matrix_entries = 0;
        std::memcpy(&num_matrix_entries, ptr, sizeof(std::size_t));
        ptr += sizeof(std::size_t);

        last_stats_.matrix_entries_received += num_matrix_entries;

        for (std::size_t j = 0; j < num_matrix_entries; ++j) {
            GhostContribution entry;
            std::memcpy(&entry, ptr, sizeof(GhostContribution));
            ptr += sizeof(GhostContribution);
            received_matrix_.push_back(entry);
        }

        std::size_t num_vector_entries = 0;
        std::memcpy(&num_vector_entries, ptr, sizeof(std::size_t));
        ptr += sizeof(std::size_t);
        last_stats_.vector_entries_received += num_vector_entries;

        for (std::size_t j = 0; j < num_vector_entries; ++j) {
            GhostVectorContribution entry;
            std::memcpy(&entry, ptr, sizeof(GhostVectorContribution));
            ptr += sizeof(GhostVectorContribution);
            received_vector_.push_back(entry);
        }
    }

    // Sort received for deterministic accumulation
    if (deterministic_) {
        std::sort(received_matrix_.begin(), received_matrix_.end());
        std::sort(received_vector_.begin(), received_vector_.end());
    }

#endif // FE_HAS_MPI

    auto end_time = std::chrono::steady_clock::now();
    last_stats_.exchange_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
}

void GhostContributionManager::startExchange()
{
    // Non-blocking version - similar to exchangeContributions but split
    exchange_in_progress_ = true;
    // Implementation would post MPI_Isend/Irecv without waiting
}

void GhostContributionManager::waitExchange()
{
    // Complete non-blocking exchange
    exchange_in_progress_ = false;
}

void GhostContributionManager::sortBuffersForDeterminism()
{
    for (auto& buffer : send_buffers_) {
        std::sort(buffer.entries.begin(), buffer.entries.end());
        std::sort(buffer.vector_entries.begin(), buffer.vector_entries.end());
    }
}

// ============================================================================
// Received Contribution Access
// ============================================================================

std::span<const GhostContribution> GhostContributionManager::getReceivedMatrixContributions() const
{
    return received_matrix_;
}

std::span<const GhostVectorContribution> GhostContributionManager::getReceivedVectorContributions() const
{
    return received_vector_;
}

void GhostContributionManager::clearReceivedContributions()
{
    received_matrix_.clear();
    received_vector_.clear();
}

// ============================================================================
// Buffer Management
// ============================================================================

void GhostContributionManager::clearSendBuffers()
{
    for (auto& buffer : send_buffers_) {
        buffer.clear();
    }

    last_stats_ = ExchangeStats{};
}

void GhostContributionManager::reserveBuffers(std::size_t entries_per_rank)
{
    for (auto& buffer : send_buffers_) {
        buffer.reserve(entries_per_rank, entries_per_rank / 10);
    }
}

std::size_t GhostContributionManager::numBufferedMatrixContributions() const
{
    std::size_t total = 0;
    for (const auto& buffer : send_buffers_) {
        total += buffer.entries.size();
    }
    return total;
}

std::size_t GhostContributionManager::numBufferedVectorContributions() const
{
    std::size_t total = 0;
    for (const auto& buffer : send_buffers_) {
        total += buffer.vector_entries.size();
    }
    return total;
}

} // namespace assembly
} // namespace FE
} // namespace svmp
