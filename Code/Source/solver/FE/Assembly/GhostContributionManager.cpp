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
#include <array>
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

namespace {

constexpr std::size_t kRadixSortThreshold = 2048u;

[[nodiscard]] inline std::uint64_t sortableKey(GlobalIndex v) noexcept
{
    using Signed = std::int64_t;
    using Unsigned = std::uint64_t;
    return static_cast<Unsigned>(static_cast<Signed>(v)) ^ (Unsigned(1) << 63);
}

template <typename T, typename KeyFn>
void radixSortByUint64(std::vector<T>& data, std::vector<T>& scratch, KeyFn key_fn)
{
    const std::size_t n = data.size();
    if (n < 2u) {
        return;
    }

    constexpr std::size_t kRadix = 256u;
    scratch.resize(n);

    std::vector<T>* src = &data;
    std::vector<T>* dst = &scratch;

    for (std::size_t pass = 0u; pass < 8u; ++pass) {
        std::array<std::size_t, kRadix> counts{};
        for (const auto& v : *src) {
            const auto key = key_fn(v);
            const auto bucket = static_cast<std::size_t>((key >> (pass * 8u)) & 0xFFu);
            ++counts[bucket];
        }

        std::array<std::size_t, kRadix> pos{};
        std::size_t sum = 0u;
        for (std::size_t i = 0u; i < kRadix; ++i) {
            pos[i] = sum;
            sum += counts[i];
        }

        for (auto& v : *src) {
            const auto key = key_fn(v);
            const auto bucket = static_cast<std::size_t>((key >> (pass * 8u)) & 0xFFu);
            (*dst)[pos[bucket]++] = std::move(v);
        }

        std::swap(src, dst);
    }

    if (src != &data) {
        data.swap(*src);
    }
}

void sortGhostMatrixEntries(std::vector<GhostContribution>& entries,
                            std::vector<GhostContribution>& scratch)
{
    if (entries.size() < kRadixSortThreshold) {
        std::sort(entries.begin(), entries.end());
        return;
    }

    // Stable radix by secondary key first, then primary.
    radixSortByUint64(entries, scratch, [](const GhostContribution& e) { return sortableKey(e.global_col); });
    radixSortByUint64(entries, scratch, [](const GhostContribution& e) { return sortableKey(e.global_row); });
}

void sortGhostVectorEntries(std::vector<GhostVectorContribution>& entries,
                            std::vector<GhostVectorContribution>& scratch)
{
    if (entries.size() < kRadixSortThreshold) {
        std::sort(entries.begin(), entries.end());
        return;
    }

    radixSortByUint64(entries, scratch, [](const GhostVectorContribution& e) { return sortableKey(e.global_row); });
}

} // namespace

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
    startExchange();
    waitExchange();
}

void GhostContributionManager::startExchange()
{
    FE_THROW_IF(exchange_in_progress_, FEException,
                "GhostContributionManager::startExchange: exchange already in progress");

    exchange_start_time_ = std::chrono::steady_clock::now();
    last_stats_ = ExchangeStats{};

    if (deterministic_) {
        sortBuffersForDeterminism();
    }

    received_matrix_.clear();
    received_vector_.clear();

    // Mark as in progress even for single-rank / non-MPI builds so callers can
    // treat startExchange()/waitExchange() as a consistent state machine.
    exchange_in_progress_ = true;

#if FE_HAS_MPI
    if (world_size_ == 1 || neighbor_ranks_.empty()) {
        return;  // Trivial/loopback exchange; waitExchange() will complete it.
    }

    // Pack send buffers into byte arrays that remain alive until waitExchange().
    const std::size_t n_neighbors = neighbor_ranks_.size();
    send_buffers_raw_.resize(n_neighbors);
    send_sizes_.assign(n_neighbors, 0);
    recv_sizes_.assign(n_neighbors, 0);
    recv_buffers_raw_.resize(n_neighbors);

    for (std::size_t i = 0; i < n_neighbors; ++i) {
        const auto& buffer = send_buffers_[i];

        const std::size_t matrix_size = buffer.entries.size() * sizeof(GhostContribution);
        const std::size_t vector_size = buffer.vector_entries.size() * sizeof(GhostVectorContribution);
        const std::size_t total_size =
            sizeof(std::size_t) + matrix_size +
            sizeof(std::size_t) + vector_size;

        send_buffers_raw_[i].resize(total_size);

        // Pack: [num_matrix][matrix...][num_vector][vector...]
        char* ptr = send_buffers_raw_[i].data();
        const std::size_t num_entries = buffer.entries.size();
        std::memcpy(ptr, &num_entries, sizeof(std::size_t));
        ptr += sizeof(std::size_t);
        if (matrix_size > 0u) {
            std::memcpy(ptr, buffer.entries.data(), matrix_size);
            ptr += matrix_size;
        }

        const std::size_t num_vector_entries = buffer.vector_entries.size();
        std::memcpy(ptr, &num_vector_entries, sizeof(std::size_t));
        ptr += sizeof(std::size_t);
        if (vector_size > 0u) {
            std::memcpy(ptr, buffer.vector_entries.data(), vector_size);
        }

        send_sizes_[i] = static_cast<int>(send_buffers_raw_[i].size());
        last_stats_.bytes_sent += send_buffers_raw_[i].size();
        last_stats_.matrix_entries_sent += buffer.entries.size();
        last_stats_.vector_entries_sent += buffer.vector_entries.size();
    }

    received_matrix_.clear();
    received_vector_.clear();

    size_requests_.resize(2 * n_neighbors);
    data_send_requests_.resize(n_neighbors);
    for (std::size_t i = 0; i < n_neighbors; ++i) {
        MPI_Isend(&send_sizes_[i], 1, MPI_INT, neighbor_ranks_[i], 0, comm_,
                  &size_requests_[i]);
        MPI_Irecv(&recv_sizes_[i], 1, MPI_INT, neighbor_ranks_[i], 0, comm_,
                  &size_requests_[n_neighbors + i]);

        const int count = send_sizes_[i];
        const char* buf = send_buffers_raw_[i].empty() ? nullptr : send_buffers_raw_[i].data();
        MPI_Isend(const_cast<char*>(buf), count, MPI_CHAR, neighbor_ranks_[i], 1, comm_,
                  &data_send_requests_[i]);
    }

    // After packing, clear the user-facing send buffers so subsequent assembly
    // can continue buffering while this exchange is in-flight.
    for (auto& buffer : send_buffers_) {
        buffer.clear();
    }

    exchange_in_progress_ = true;
#endif // FE_HAS_MPI
}

void GhostContributionManager::waitExchange()
{
    if (!exchange_in_progress_) {
        return;
    }

#if FE_HAS_MPI
    if (world_size_ == 1 || neighbor_ranks_.empty()) {
        // Loopback: treat buffered contributions as locally received.
        for (auto& buffer : send_buffers_) {
            last_stats_.matrix_entries_sent += buffer.entries.size();
            last_stats_.vector_entries_sent += buffer.vector_entries.size();

            last_stats_.matrix_entries_received += buffer.entries.size();
            last_stats_.vector_entries_received += buffer.vector_entries.size();

            received_matrix_.insert(received_matrix_.end(),
                                    buffer.entries.begin(), buffer.entries.end());
            received_vector_.insert(received_vector_.end(),
                                    buffer.vector_entries.begin(), buffer.vector_entries.end());
            buffer.clear();
        }

        if (deterministic_) {
            std::vector<GhostContribution> matrix_scratch;
            std::vector<GhostVectorContribution> vector_scratch;
            sortGhostMatrixEntries(received_matrix_, matrix_scratch);
            sortGhostVectorEntries(received_vector_, vector_scratch);
        }

        auto end_time = std::chrono::steady_clock::now();
        last_stats_.exchange_time_seconds =
            std::chrono::duration<double>(end_time - exchange_start_time_).count();
        exchange_in_progress_ = false;
        return;
    }

    // Wait for size exchange to learn receive sizes.
    if (!size_requests_.empty()) {
        MPI_Waitall(static_cast<int>(size_requests_.size()), size_requests_.data(),
                    MPI_STATUSES_IGNORE);
    }

    const std::size_t n_neighbors = neighbor_ranks_.size();
    data_recv_requests_.resize(n_neighbors);
    for (std::size_t i = 0; i < n_neighbors; ++i) {
        recv_buffers_raw_[i].resize(static_cast<std::size_t>(recv_sizes_[i]));
        last_stats_.bytes_received += recv_buffers_raw_[i].size();

        const int count = recv_sizes_[i];
        char* buf = recv_buffers_raw_[i].empty() ? nullptr : recv_buffers_raw_[i].data();
        MPI_Irecv(buf, count, MPI_CHAR, neighbor_ranks_[i], 1, comm_,
                  &data_recv_requests_[i]);
    }

    if (!data_send_requests_.empty()) {
        MPI_Waitall(static_cast<int>(data_send_requests_.size()), data_send_requests_.data(),
                    MPI_STATUSES_IGNORE);
    }
    if (!data_recv_requests_.empty()) {
        MPI_Waitall(static_cast<int>(data_recv_requests_.size()), data_recv_requests_.data(),
                    MPI_STATUSES_IGNORE);
    }

    // Unpack received data
    received_matrix_.clear();
    received_vector_.clear();

    for (std::size_t i = 0; i < n_neighbors; ++i) {
        if (recv_buffers_raw_[i].empty()) continue;

        const char* ptr = recv_buffers_raw_[i].data();

        std::size_t num_matrix_entries = 0;
        std::memcpy(&num_matrix_entries, ptr, sizeof(std::size_t));
        ptr += sizeof(std::size_t);

        last_stats_.matrix_entries_received += num_matrix_entries;
        const auto old_m = received_matrix_.size();
        received_matrix_.resize(old_m + num_matrix_entries);
        if (num_matrix_entries > 0u) {
            std::memcpy(received_matrix_.data() + old_m, ptr,
                        num_matrix_entries * sizeof(GhostContribution));
            ptr += num_matrix_entries * sizeof(GhostContribution);
        }

        std::size_t num_vector_entries = 0;
        std::memcpy(&num_vector_entries, ptr, sizeof(std::size_t));
        ptr += sizeof(std::size_t);

        last_stats_.vector_entries_received += num_vector_entries;
        const auto old_v = received_vector_.size();
        received_vector_.resize(old_v + num_vector_entries);
        if (num_vector_entries > 0u) {
            std::memcpy(received_vector_.data() + old_v, ptr,
                        num_vector_entries * sizeof(GhostVectorContribution));
        }
    }

    // Sort received for deterministic accumulation
    if (deterministic_) {
        std::vector<GhostContribution> matrix_scratch;
        std::vector<GhostVectorContribution> vector_scratch;
        sortGhostMatrixEntries(received_matrix_, matrix_scratch);
        sortGhostVectorEntries(received_vector_, vector_scratch);
    }

    auto end_time = std::chrono::steady_clock::now();
    last_stats_.exchange_time_seconds =
        std::chrono::duration<double>(end_time - exchange_start_time_).count();

    exchange_in_progress_ = false;
#endif // FE_HAS_MPI

#if !FE_HAS_MPI
    // Non-MPI build: treat buffered contributions as locally received.
    for (auto& buffer : send_buffers_) {
        last_stats_.matrix_entries_sent += buffer.entries.size();
        last_stats_.vector_entries_sent += buffer.vector_entries.size();

        last_stats_.matrix_entries_received += buffer.entries.size();
        last_stats_.vector_entries_received += buffer.vector_entries.size();

        received_matrix_.insert(received_matrix_.end(),
                                buffer.entries.begin(), buffer.entries.end());
        received_vector_.insert(received_vector_.end(),
                                buffer.vector_entries.begin(), buffer.vector_entries.end());
        buffer.clear();
    }

    if (deterministic_) {
        std::vector<GhostContribution> matrix_scratch;
        std::vector<GhostVectorContribution> vector_scratch;
        sortGhostMatrixEntries(received_matrix_, matrix_scratch);
        sortGhostVectorEntries(received_vector_, vector_scratch);
    }

    auto end_time = std::chrono::steady_clock::now();
    last_stats_.exchange_time_seconds =
        std::chrono::duration<double>(end_time - exchange_start_time_).count();

    exchange_in_progress_ = false;
#endif // !FE_HAS_MPI
}

void GhostContributionManager::sortBuffersForDeterminism()
{
    std::vector<GhostContribution> matrix_scratch;
    std::vector<GhostVectorContribution> vector_scratch;
    for (auto& buffer : send_buffers_) {
        sortGhostMatrixEntries(buffer.entries, matrix_scratch);
        sortGhostVectorEntries(buffer.vector_entries, vector_scratch);
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

std::vector<GhostContribution> GhostContributionManager::takeReceivedMatrixContributions()
{
    return std::move(received_matrix_);
}

std::vector<GhostVectorContribution> GhostContributionManager::takeReceivedVectorContributions()
{
    return std::move(received_vector_);
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
