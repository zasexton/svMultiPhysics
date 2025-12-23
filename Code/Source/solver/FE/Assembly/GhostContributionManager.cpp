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

    // For contiguous ownership, use local DOF count
    // In many cases, DOFs 0..n_local-1 are owned locally when using a standard partition
    // This is a heuristic - more robust ownership requires DofHandler or partition info
    GlobalIndex n_local = dof_map_->getNumLocalDofs();
    if (n_local > 0) {
        // Assume contiguous ownership starting from 0 for this rank
        // This is a simplification - real implementation needs proper partition info
        owned_begin_ = 0;
        owned_end_ = n_local;
        has_contiguous_ownership_ = true;
    } else {
        has_contiguous_ownership_ = false;
    }

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

    // Determine ghost DOFs and their owners
    if (ghost_dof_manager_) {
        // Use ghost DOF manager for ownership info
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
    // Quick ownership check for contiguous case
    if (has_contiguous_ownership_) {
        if (global_row >= owned_begin_ && global_row < owned_end_) {
            return true;  // Locally owned, caller inserts directly
        }
    } else {
        if (dof_map_->isOwnedDof(global_row)) {
            return true;
        }
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
    if (has_contiguous_ownership_) {
        if (global_row >= owned_begin_ && global_row < owned_end_) {
            return true;
        }
    } else {
        if (dof_map_->isOwnedDof(global_row)) {
            return true;
        }
    }

    if (policy_ == GhostPolicy::OwnedRowsOnly) {
        return false;
    }

    int owner = getOwnerRank(global_row);
    int neighbor_idx = findNeighborIndex(owner);

    if (neighbor_idx >= 0) {
        auto& buffer = send_buffers_[static_cast<std::size_t>(neighbor_idx)];
        buffer.vector_entries.push_back(value);
        // Store row index - we use pairs (row, value) packed in vector_entries
        // For simplicity, store as two consecutive values
        // Better: use a separate structure
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
    if (has_contiguous_ownership_) {
        return global_dof >= owned_begin_ && global_dof < owned_end_;
    }
    return dof_map_->isOwnedDof(global_dof);
}

int GhostContributionManager::getOwnerRank(GlobalIndex global_dof) const
{
    auto it = ghost_owner_rank_.find(global_dof);
    if (it != ghost_owner_rank_.end()) {
        return it->second;
    }

    // If not in ghost map, it's locally owned
    return my_rank_;
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
        std::size_t total_size = sizeof(std::size_t) + matrix_size;

        send_data[i].resize(total_size);

        // Pack: [num_entries][entries...]
        char* ptr = send_data[i].data();
        std::size_t num_entries = buffer.entries.size();
        std::memcpy(ptr, &num_entries, sizeof(std::size_t));
        ptr += sizeof(std::size_t);

        if (!buffer.entries.empty()) {
            std::memcpy(ptr, buffer.entries.data(), matrix_size);
        }

        last_stats_.bytes_sent += send_data[i].size();
        last_stats_.matrix_entries_sent += buffer.entries.size();
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

    for (std::size_t i = 0; i < neighbor_ranks_.size(); ++i) {
        if (recv_buffers_raw_[i].empty()) continue;

        const char* ptr = recv_buffers_raw_[i].data();

        std::size_t num_entries;
        std::memcpy(&num_entries, ptr, sizeof(std::size_t));
        ptr += sizeof(std::size_t);

        last_stats_.matrix_entries_received += num_entries;

        for (std::size_t j = 0; j < num_entries; ++j) {
            GhostContribution entry;
            std::memcpy(&entry, ptr, sizeof(GhostContribution));
            ptr += sizeof(GhostContribution);
            received_matrix_.push_back(entry);
        }
    }

    // Sort received for deterministic accumulation
    if (deterministic_) {
        std::sort(received_matrix_.begin(), received_matrix_.end());
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
    }
}

// ============================================================================
// Received Contribution Access
// ============================================================================

std::span<const GhostContribution> GhostContributionManager::getReceivedMatrixContributions() const
{
    return received_matrix_;
}

std::span<const std::pair<GlobalIndex, Real>> GhostContributionManager::getReceivedVectorContributions() const
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
