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

#include "ParallelSparsity.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// DofOwnership
// ============================================================================

DofOwnership::DofOwnership(GlobalIndex n_global_dofs, int n_ranks, int my_rank)
    : n_global_dofs_(n_global_dofs),
      n_ranks_(n_ranks),
      my_rank_(my_rank)
{
    FE_CHECK_ARG(n_global_dofs >= 0, "Global DOF count must be non-negative");
    FE_CHECK_ARG(n_ranks > 0, "Number of ranks must be positive");
    FE_CHECK_ARG(my_rank >= 0 && my_rank < n_ranks, "Invalid rank");

    // Compute balanced block distribution
    rank_offsets_.resize(static_cast<std::size_t>(n_ranks) + 1);

    GlobalIndex base_count = n_global_dofs / n_ranks;
    GlobalIndex remainder = n_global_dofs % n_ranks;

    rank_offsets_[0] = 0;
    for (int r = 0; r < n_ranks; ++r) {
        GlobalIndex count = base_count + (r < remainder ? 1 : 0);
        rank_offsets_[static_cast<std::size_t>(r) + 1] =
            rank_offsets_[static_cast<std::size_t>(r)] + count;
    }

    owned_range_.first = rank_offsets_[static_cast<std::size_t>(my_rank)];
    owned_range_.last = rank_offsets_[static_cast<std::size_t>(my_rank) + 1];
}

DofOwnership::DofOwnership(std::span<const GlobalIndex> rank_offsets, int my_rank)
    : my_rank_(my_rank)
{
    FE_CHECK_ARG(rank_offsets.size() >= 2, "Rank offsets must have at least 2 elements");
    FE_CHECK_ARG(rank_offsets[0] == 0, "Rank offsets must start at 0");

    n_ranks_ = static_cast<int>(rank_offsets.size()) - 1;
    n_global_dofs_ = rank_offsets.back();

    FE_CHECK_ARG(my_rank >= 0 && my_rank < n_ranks_, "Invalid rank");

    rank_offsets_.assign(rank_offsets.begin(), rank_offsets.end());

    owned_range_.first = rank_offsets_[static_cast<std::size_t>(my_rank)];
    owned_range_.last = rank_offsets_[static_cast<std::size_t>(my_rank) + 1];
}

DofOwnership::DofOwnership(GlobalIndex n_global_dofs,
                           std::function<int(GlobalIndex)> owner_func,
                           int my_rank)
    : n_global_dofs_(n_global_dofs),
      n_ranks_(1),
      my_rank_(my_rank),
      owner_func_(std::move(owner_func)),
      using_custom_func_(true)
{
    FE_CHECK_ARG(owner_func_ != nullptr, "Owner function must not be null");
    FE_CHECK_ARG(n_global_dofs_ >= 0, "Global DOF count must be non-negative");
    FE_CHECK_ARG(my_rank_ >= 0, "Invalid rank");

    // For custom functions, infer (1) number of ranks and (2) owned range.
    // The owned range is required to be contiguous, matching common MPI sparse
    // matrix ownership models (PETSc/Trilinos).
    GlobalIndex first_owned = -1;
    GlobalIndex last_owned = -1;
    bool exited_owned_region = false;
    int max_rank_id = -1;

    for (GlobalIndex dof = 0; dof < n_global_dofs; ++dof) {
        int owner = owner_func_(dof);
        FE_CHECK_ARG(owner >= 0, "Owner function returned negative rank for DOF " +
                                   std::to_string(dof));
        max_rank_id = std::max(max_rank_id, owner);

        if (owner == my_rank_) {
            FE_CHECK_ARG(!exited_owned_region,
                         "Custom ownership describes a non-contiguous owned range for rank " +
                             std::to_string(my_rank_));
            if (first_owned < 0) first_owned = dof;
            last_owned = dof + 1;
        } else if (first_owned >= 0) {
            exited_owned_region = true;
        }
    }

    // Allow ranks with zero owned DOFs.
    n_ranks_ = std::max(max_rank_id + 1, my_rank_ + 1);

    if (first_owned < 0) {
        owned_range_ = {0, 0};  // Empty range
    } else {
        owned_range_ = {first_owned, last_owned};
    }
}

int DofOwnership::getOwner(GlobalIndex dof) const {
    FE_CHECK_ARG(dof >= 0 && dof < n_global_dofs_, "DOF index out of range");
    if (using_custom_func_) {
        return owner_func_(dof);
    }

    // Binary search in rank_offsets
    auto it = std::upper_bound(rank_offsets_.begin(), rank_offsets_.end(), dof);
    if (it == rank_offsets_.begin()) {
        return 0;
    }
    return static_cast<int>(std::distance(rank_offsets_.begin(), it)) - 1;
}

IndexRange DofOwnership::getRankRange(int rank) const {
    FE_CHECK_ARG(!using_custom_func_,
                 "getRankRange not supported for custom ownership functions");
    FE_CHECK_ARG(rank >= 0 && rank < n_ranks_, "Invalid rank");

    return {
        rank_offsets_[static_cast<std::size_t>(rank)],
        rank_offsets_[static_cast<std::size_t>(rank) + 1]
    };
}

void DofOwnership::addGhostDof(GlobalIndex dof) {
    if (!isOwned(dof) && ghost_set_.find(dof) == ghost_set_.end()) {
        ghost_dofs_.push_back(dof);
        ghost_set_.insert(dof);
    }
    ghosts_finalized_ = false;
}

void DofOwnership::finalizeGhosts() {
    if (ghosts_finalized_) return;

    // Sort and remove duplicates (should already be unique due to set check)
    std::sort(ghost_dofs_.begin(), ghost_dofs_.end());
    auto it = std::unique(ghost_dofs_.begin(), ghost_dofs_.end());
    ghost_dofs_.erase(it, ghost_dofs_.end());

    ghosts_finalized_ = true;
}

// ============================================================================
// ParallelSparsityManager
// ============================================================================

ParallelSparsityManager::ParallelSparsityManager()
    : my_rank_(0), n_ranks_(1)
{
}

#if FE_HAS_MPI
ParallelSparsityManager::ParallelSparsityManager(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm, &my_rank_);
    MPI_Comm_size(comm, &n_ranks_);
}
#endif

ParallelSparsityManager::ParallelSparsityManager(int n_ranks, int my_rank)
    : my_rank_(my_rank), n_ranks_(n_ranks)
{
    FE_CHECK_ARG(n_ranks > 0, "Number of ranks must be positive");
    FE_CHECK_ARG(my_rank >= 0 && my_rank < n_ranks, "Invalid rank");
}

void ParallelSparsityManager::setOwnership(DofOwnership ownership) {
    row_ownership_ = std::move(ownership);
}

void ParallelSparsityManager::setBlockOwnership(GlobalIndex n_global_dofs) {
    row_ownership_ = DofOwnership(n_global_dofs, n_ranks_, my_rank_);
}

void ParallelSparsityManager::setRowDofMap(const dofs::DofMap& dof_map) {
    row_dof_map_ = std::make_shared<DofMapAdapter>(dof_map);
}

void ParallelSparsityManager::setRowDofMap(std::shared_ptr<IDofMapQuery> dof_map_query) {
    row_dof_map_ = std::move(dof_map_query);
}

void ParallelSparsityManager::setColDofMap(const dofs::DofMap& dof_map) {
    col_dof_map_ = std::make_shared<DofMapAdapter>(dof_map);
}

void ParallelSparsityManager::setColDofMap(std::shared_ptr<IDofMapQuery> dof_map_query) {
    col_dof_map_ = std::move(dof_map_query);
}

void ParallelSparsityManager::setColOwnership(DofOwnership ownership) {
    col_ownership_ = std::move(ownership);
    col_ownership_set_ = true;
}

DistributedSparsityPattern ParallelSparsityManager::build() {
    FE_CHECK_ARG(row_dof_map_ != nullptr, "Row DOF map not set");
    FE_CHECK_ARG(row_ownership_.globalNumDofs() > 0, "Row ownership not set");

    // Use row settings for columns if not explicitly set
    IDofMapQuery* col_map = col_dof_map_ ? col_dof_map_.get() : row_dof_map_.get();
    const DofOwnership& col_own = col_ownership_set_ ? col_ownership_ : row_ownership_;
    FE_CHECK_ARG(row_ownership_.myRank() == my_rank_, "Row ownership rank does not match manager");
    FE_CHECK_ARG(row_ownership_.numRanks() == n_ranks_,
                 "Row ownership rank count does not match manager");
    FE_CHECK_ARG(col_own.myRank() == my_rank_, "Column ownership rank does not match manager");
    FE_CHECK_ARG(col_own.numRanks() == n_ranks_,
                 "Column ownership rank count does not match manager");
    FE_CHECK_ARG(row_dof_map_->getNumDofs() == row_ownership_.globalNumDofs(),
                 "Row DOF map size does not match ownership global DOF count");
    FE_CHECK_ARG(col_map->getNumDofs() == col_own.globalNumDofs(),
                 "Column DOF map size does not match ownership global DOF count");
    FE_CHECK_ARG(row_dof_map_->getNumCells() == col_map->getNumCells(),
                 "Row and column DOF maps must have the same number of cells");

    // Create distributed pattern
    DistributedSparsityPattern pattern(
        row_ownership_.ownedRange(),
        col_own.ownedRange(),
        row_ownership_.globalNumDofs(),
        col_own.globalNumDofs()
    );

    const IndexRange& owned_rows = row_ownership_.ownedRange();
    const IndexRange& owned_cols = col_own.ownedRange();

    const bool symmetrize = options_.symmetric_pattern &&
                            row_ownership_.globalNumDofs() == col_own.globalNumDofs();

    const auto insert_pair = [&](GlobalIndex global_row, GlobalIndex global_col) {
        if (!options_.include_ghost_rows && !owned_cols.contains(global_col)) {
            return;
        }
        pattern.addEntry(global_row, global_col);
    };

    struct RowColPair {
        GlobalIndex row;
        GlobalIndex col;
        bool operator<(const RowColPair& other) const noexcept {
            return (row < other.row) || (row == other.row && col < other.col);
        }
        bool operator==(const RowColPair& other) const noexcept {
            return row == other.row && col == other.col;
        }
    };

    bool use_mpi_exchange = false;
#if FE_HAS_MPI
    use_mpi_exchange = (comm_ != MPI_COMM_SELF && n_ranks_ > 1);
#endif

    std::vector<std::vector<RowColPair>> send_pairs_by_rank;
    if (use_mpi_exchange) {
        send_pairs_by_rank.resize(static_cast<std::size_t>(n_ranks_));
    }

    const auto queue_remote_row = [&](int owner_rank,
                                      GlobalIndex global_row,
                                      std::span<const GlobalIndex> col_dofs) {
        auto& pairs = send_pairs_by_rank[static_cast<std::size_t>(owner_rank)];
        for (GlobalIndex global_col : col_dofs) {
            if (global_col < 0 || global_col >= col_own.globalNumDofs()) continue;
            pairs.push_back({global_row, global_col});
        }
    };

    const auto process_couplings = [&](std::span<const GlobalIndex> row_dofs,
                                       std::span<const GlobalIndex> col_dofs) {
        for (GlobalIndex global_row : row_dofs) {
            if (global_row < 0 || global_row >= row_ownership_.globalNumDofs()) continue;

            const int owner_rank = row_ownership_.getOwner(global_row);
            if (owner_rank == my_rank_) {
                if (!owned_rows.contains(global_row)) {
                    continue;
                }
                for (GlobalIndex global_col : col_dofs) {
                    if (global_col < 0 || global_col >= col_own.globalNumDofs()) continue;
                    insert_pair(global_row, global_col);
                }
            } else if (use_mpi_exchange) {
                queue_remote_row(owner_rank, global_row, col_dofs);
            }
        }
    };

    // Build local sparsity by iterating over local elements
    const GlobalIndex n_cells = row_dof_map_->getNumCells();
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        process_couplings(row_dof_map_->getCellDofs(cell), col_map->getCellDofs(cell));
    }

    // Symmetrize if requested (square only)
    if (symmetrize) {
        for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
            process_couplings(col_map->getCellDofs(cell), row_dof_map_->getCellDofs(cell));
        }
    }

#if FE_HAS_MPI
    if (use_mpi_exchange) {
        std::vector<int> send_counts(n_ranks_, 0);
        for (int r = 0; r < n_ranks_; ++r) {
            auto& pairs = send_pairs_by_rank[static_cast<std::size_t>(r)];
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

            const std::size_t n_vals = pairs.size() * 2;
            FE_CHECK_ARG(n_vals <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                         "Too many remote row coupling entries to communicate");
            send_counts[r] = static_cast<int>(n_vals);
        }

        std::vector<int> recv_counts(n_ranks_, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, comm_);

        std::vector<int> send_displs(n_ranks_, 0);
        std::vector<int> recv_displs(n_ranks_, 0);
        for (int r = 1; r < n_ranks_; ++r) {
            send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }

        const int total_send = send_displs.back() + send_counts.back();
        const int total_recv = recv_displs.back() + recv_counts.back();

        std::vector<GlobalIndex> send_buf(static_cast<std::size_t>(total_send));
        for (int r = 0; r < n_ranks_; ++r) {
            int pos = send_displs[r];
            const auto& pairs = send_pairs_by_rank[static_cast<std::size_t>(r)];
            for (const auto& p : pairs) {
                send_buf[static_cast<std::size_t>(pos++)] = p.row;
                send_buf[static_cast<std::size_t>(pos++)] = p.col;
            }
        }

        std::vector<GlobalIndex> recv_buf(static_cast<std::size_t>(total_recv));
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                      recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                      comm_);

        FE_CHECK_ARG(total_recv % 2 == 0, "Remote row coupling receive buffer malformed");
        for (int i = 0; i < total_recv; i += 2) {
            const GlobalIndex global_row = recv_buf[static_cast<std::size_t>(i)];
            const GlobalIndex global_col = recv_buf[static_cast<std::size_t>(i + 1)];
            if (!owned_rows.contains(global_row)) {
                continue;
            }
            if (global_col < 0 || global_col >= col_own.globalNumDofs()) continue;
            insert_pair(global_row, global_col);
        }
    }
#endif

    if (options_.ensure_diagonal) {
        pattern.ensureDiagonal();
    }

    if (options_.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    detectGhostDofs(pattern, col_own);

    if (exchange_ghost_rows_ && !isSerial()) {
        exchangeGhostRowSparsity(pattern);
    }

    return pattern;
}

DistributedSparsityPattern ParallelSparsityManager::build(std::span<const GlobalIndex> cell_ids) {
    FE_CHECK_ARG(row_dof_map_ != nullptr, "Row DOF map not set");
    FE_CHECK_ARG(row_ownership_.globalNumDofs() > 0, "Row ownership not set");

    IDofMapQuery* col_map = col_dof_map_ ? col_dof_map_.get() : row_dof_map_.get();
    const DofOwnership& col_own = col_ownership_set_ ? col_ownership_ : row_ownership_;
    FE_CHECK_ARG(row_ownership_.myRank() == my_rank_, "Row ownership rank does not match manager");
    FE_CHECK_ARG(row_ownership_.numRanks() == n_ranks_,
                 "Row ownership rank count does not match manager");
    FE_CHECK_ARG(col_own.myRank() == my_rank_, "Column ownership rank does not match manager");
    FE_CHECK_ARG(col_own.numRanks() == n_ranks_,
                 "Column ownership rank count does not match manager");
    FE_CHECK_ARG(row_dof_map_->getNumDofs() == row_ownership_.globalNumDofs(),
                 "Row DOF map size does not match ownership global DOF count");
    FE_CHECK_ARG(col_map->getNumDofs() == col_own.globalNumDofs(),
                 "Column DOF map size does not match ownership global DOF count");
    FE_CHECK_ARG(row_dof_map_->getNumCells() == col_map->getNumCells(),
                 "Row and column DOF maps must have the same number of cells");

    DistributedSparsityPattern pattern(
        row_ownership_.ownedRange(),
        col_own.ownedRange(),
        row_ownership_.globalNumDofs(),
        col_own.globalNumDofs()
    );

    const IndexRange& owned_rows = row_ownership_.ownedRange();
    const IndexRange& owned_cols = col_own.ownedRange();

    const bool symmetrize = options_.symmetric_pattern &&
                            row_ownership_.globalNumDofs() == col_own.globalNumDofs();

    const auto insert_pair = [&](GlobalIndex global_row, GlobalIndex global_col) {
        if (!options_.include_ghost_rows && !owned_cols.contains(global_col)) {
            return;
        }
        pattern.addEntry(global_row, global_col);
    };

    struct RowColPair {
        GlobalIndex row;
        GlobalIndex col;
        bool operator<(const RowColPair& other) const noexcept {
            return (row < other.row) || (row == other.row && col < other.col);
        }
        bool operator==(const RowColPair& other) const noexcept {
            return row == other.row && col == other.col;
        }
    };

    bool use_mpi_exchange = false;
#if FE_HAS_MPI
    use_mpi_exchange = (comm_ != MPI_COMM_SELF && n_ranks_ > 1);
#endif

    std::vector<std::vector<RowColPair>> send_pairs_by_rank;
    if (use_mpi_exchange) {
        send_pairs_by_rank.resize(static_cast<std::size_t>(n_ranks_));
    }

    const auto queue_remote_row = [&](int owner_rank,
                                      GlobalIndex global_row,
                                      std::span<const GlobalIndex> col_dofs) {
        auto& pairs = send_pairs_by_rank[static_cast<std::size_t>(owner_rank)];
        for (GlobalIndex global_col : col_dofs) {
            if (global_col < 0 || global_col >= col_own.globalNumDofs()) continue;
            pairs.push_back({global_row, global_col});
        }
    };

    const auto process_couplings = [&](std::span<const GlobalIndex> row_dofs,
                                       std::span<const GlobalIndex> col_dofs) {
        for (GlobalIndex global_row : row_dofs) {
            if (global_row < 0 || global_row >= row_ownership_.globalNumDofs()) continue;

            const int owner_rank = row_ownership_.getOwner(global_row);
            if (owner_rank == my_rank_) {
                if (!owned_rows.contains(global_row)) {
                    continue;
                }
                for (GlobalIndex global_col : col_dofs) {
                    if (global_col < 0 || global_col >= col_own.globalNumDofs()) continue;
                    insert_pair(global_row, global_col);
                }
            } else if (use_mpi_exchange) {
                queue_remote_row(owner_rank, global_row, col_dofs);
            }
        }
    };

    const GlobalIndex n_cells = row_dof_map_->getNumCells();
    for (GlobalIndex cell : cell_ids) {
        if (cell < 0 || cell >= n_cells) continue;
        process_couplings(row_dof_map_->getCellDofs(cell), col_map->getCellDofs(cell));
    }

    if (symmetrize) {
        for (GlobalIndex cell : cell_ids) {
            if (cell < 0 || cell >= n_cells) continue;
            process_couplings(col_map->getCellDofs(cell), row_dof_map_->getCellDofs(cell));
        }
    }

#if FE_HAS_MPI
    if (use_mpi_exchange) {
        std::vector<int> send_counts(n_ranks_, 0);
        for (int r = 0; r < n_ranks_; ++r) {
            auto& pairs = send_pairs_by_rank[static_cast<std::size_t>(r)];
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

            const std::size_t n_vals = pairs.size() * 2;
            FE_CHECK_ARG(n_vals <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                         "Too many remote row coupling entries to communicate");
            send_counts[r] = static_cast<int>(n_vals);
        }

        std::vector<int> recv_counts(n_ranks_, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, comm_);

        std::vector<int> send_displs(n_ranks_, 0);
        std::vector<int> recv_displs(n_ranks_, 0);
        for (int r = 1; r < n_ranks_; ++r) {
            send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }

        const int total_send = send_displs.back() + send_counts.back();
        const int total_recv = recv_displs.back() + recv_counts.back();

        std::vector<GlobalIndex> send_buf(static_cast<std::size_t>(total_send));
        for (int r = 0; r < n_ranks_; ++r) {
            int pos = send_displs[r];
            const auto& pairs = send_pairs_by_rank[static_cast<std::size_t>(r)];
            for (const auto& p : pairs) {
                send_buf[static_cast<std::size_t>(pos++)] = p.row;
                send_buf[static_cast<std::size_t>(pos++)] = p.col;
            }
        }

        std::vector<GlobalIndex> recv_buf(static_cast<std::size_t>(total_recv));
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                      recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                      comm_);

        FE_CHECK_ARG(total_recv % 2 == 0, "Remote row coupling receive buffer malformed");
        for (int i = 0; i < total_recv; i += 2) {
            const GlobalIndex global_row = recv_buf[static_cast<std::size_t>(i)];
            const GlobalIndex global_col = recv_buf[static_cast<std::size_t>(i + 1)];
            if (!owned_rows.contains(global_row)) {
                continue;
            }
            if (global_col < 0 || global_col >= col_own.globalNumDofs()) continue;
            insert_pair(global_row, global_col);
        }
    }
#endif

    if (options_.ensure_diagonal) {
        pattern.ensureDiagonal();
    }

    if (options_.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    detectGhostDofs(pattern, col_own);

    if (exchange_ghost_rows_ && !isSerial()) {
        exchangeGhostRowSparsity(pattern);
    }

    return pattern;
}

std::span<const GlobalIndex> ParallelSparsityManager::ghostColsOwnedBy(int owner_rank) const {
    FE_CHECK_ARG(owner_rank >= 0 && owner_rank < n_ranks_, "Invalid owner rank");
    if (ghost_cols_by_owner_.empty()) {
        return {};
    }
    const auto& cols = ghost_cols_by_owner_[static_cast<std::size_t>(owner_rank)];
    return std::span<const GlobalIndex>(cols.data(), cols.size());
}

void ParallelSparsityManager::exchangeGhostRowSparsity(
    [[maybe_unused]] DistributedSparsityPattern& pattern) {
    if (isSerial()) {
        return;
    }

#if FE_HAS_MPI
    FE_CHECK_ARG(pattern.isFinalized(), "Ghost row exchange requires a finalized pattern");
    FE_THROW_IF(comm_ == MPI_COMM_SELF, NotImplementedException,
                "Ghost row exchange requires an MPI communicator (not MPI_COMM_SELF)");

    const IndexRange& owned_rows = row_ownership_.ownedRange();

    // Default policy: request ghost rows corresponding to the ghost columns
    // present in owned rows (square matrices / overlapping Schwarz use case).
    std::vector<GlobalIndex> ghost_rows_requested;
    ghost_rows_requested.reserve(ghost_cols_.size());
    for (GlobalIndex col : ghost_cols_) {
        if (col < 0 || col >= pattern.globalRows()) continue;
        if (owned_rows.contains(col)) continue;
        ghost_rows_requested.push_back(col);
    }
    if (ghost_rows_requested.empty()) {
        pattern.clearGhostRows();
        return;
    }

    // Group requested rows by their owning rank.
    std::vector<std::vector<GlobalIndex>> rows_by_owner(static_cast<std::size_t>(n_ranks_));
    for (GlobalIndex row : ghost_rows_requested) {
        const int owner = row_ownership_.getOwner(row);
        if (owner == my_rank_) continue;
        rows_by_owner[static_cast<std::size_t>(owner)].push_back(row);
    }
    for (auto& v : rows_by_owner) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // ---------------------------------------------------------------------
    // Phase 1: exchange row requests
    // ---------------------------------------------------------------------
    std::vector<int> send_counts_rows(n_ranks_, 0);
    for (int r = 0; r < n_ranks_; ++r) {
        const auto& rows = rows_by_owner[static_cast<std::size_t>(r)];
        FE_CHECK_ARG(rows.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                     "Too many ghost row requests for MPI_Alltoallv");
        send_counts_rows[r] = static_cast<int>(rows.size());
    }

    std::vector<int> recv_counts_rows(n_ranks_, 0);
    MPI_Alltoall(send_counts_rows.data(), 1, MPI_INT,
                 recv_counts_rows.data(), 1, MPI_INT, comm_);

    std::vector<int> send_displs_rows(n_ranks_, 0);
    std::vector<int> recv_displs_rows(n_ranks_, 0);
    for (int r = 1; r < n_ranks_; ++r) {
        send_displs_rows[r] = send_displs_rows[r - 1] + send_counts_rows[r - 1];
        recv_displs_rows[r] = recv_displs_rows[r - 1] + recv_counts_rows[r - 1];
    }

    const int total_send_rows = send_displs_rows.back() + send_counts_rows.back();
    const int total_recv_rows = recv_displs_rows.back() + recv_counts_rows.back();

    std::vector<GlobalIndex> send_buf_rows(static_cast<std::size_t>(total_send_rows));
    for (int r = 0; r < n_ranks_; ++r) {
        int pos = send_displs_rows[r];
        for (GlobalIndex row : rows_by_owner[static_cast<std::size_t>(r)]) {
            send_buf_rows[static_cast<std::size_t>(pos++)] = row;
        }
    }

    std::vector<GlobalIndex> recv_buf_rows(static_cast<std::size_t>(total_recv_rows));
    MPI_Alltoallv(send_buf_rows.data(), send_counts_rows.data(), send_displs_rows.data(), MPI_INT64_T,
                  recv_buf_rows.data(), recv_counts_rows.data(), recv_displs_rows.data(), MPI_INT64_T,
                  comm_);

    // ---------------------------------------------------------------------
    // Phase 2: each owner rank prepares row sizes + column lists for requests
    // ---------------------------------------------------------------------
    std::vector<std::vector<GlobalIndex>> send_row_sizes(static_cast<std::size_t>(n_ranks_));
    std::vector<std::vector<GlobalIndex>> send_row_cols(static_cast<std::size_t>(n_ranks_));

    for (int r = 0; r < n_ranks_; ++r) {
        const int n_req = recv_counts_rows[r];
        if (n_req <= 0) continue;

        auto& sizes = send_row_sizes[static_cast<std::size_t>(r)];
        sizes.reserve(static_cast<std::size_t>(n_req));

        for (int i = 0; i < n_req; ++i) {
            const GlobalIndex row = recv_buf_rows[static_cast<std::size_t>(recv_displs_rows[r] + i)];
            if (row < 0 || row >= row_ownership_.globalNumDofs() || !owned_rows.contains(row)) {
                sizes.push_back(0);
                continue;
            }

            auto cols = pattern.getOwnedRowGlobalCols(row);
            sizes.push_back(static_cast<GlobalIndex>(cols.size()));
            auto& flat = send_row_cols[static_cast<std::size_t>(r)];
            flat.insert(flat.end(), cols.begin(), cols.end());
        }
    }

    // Exchange row sizes (1 value per requested row).
    std::vector<int> send_counts_sizes(n_ranks_, 0);
    for (int r = 0; r < n_ranks_; ++r) {
        const auto& sizes = send_row_sizes[static_cast<std::size_t>(r)];
        FE_CHECK_ARG(sizes.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                     "Too many ghost row size entries for MPI_Alltoallv");
        send_counts_sizes[r] = static_cast<int>(sizes.size());
    }

    std::vector<int> recv_counts_sizes(n_ranks_, 0);
    MPI_Alltoall(send_counts_sizes.data(), 1, MPI_INT,
                 recv_counts_sizes.data(), 1, MPI_INT, comm_);

    std::vector<int> send_displs_sizes(n_ranks_, 0);
    std::vector<int> recv_displs_sizes(n_ranks_, 0);
    for (int r = 1; r < n_ranks_; ++r) {
        send_displs_sizes[r] = send_displs_sizes[r - 1] + send_counts_sizes[r - 1];
        recv_displs_sizes[r] = recv_displs_sizes[r - 1] + recv_counts_sizes[r - 1];
    }

    const int total_send_sizes = send_displs_sizes.back() + send_counts_sizes.back();
    const int total_recv_sizes = recv_displs_sizes.back() + recv_counts_sizes.back();

    std::vector<GlobalIndex> send_buf_sizes(static_cast<std::size_t>(total_send_sizes));
    for (int r = 0; r < n_ranks_; ++r) {
        int pos = send_displs_sizes[r];
        for (GlobalIndex sz : send_row_sizes[static_cast<std::size_t>(r)]) {
            send_buf_sizes[static_cast<std::size_t>(pos++)] = sz;
        }
    }

    std::vector<GlobalIndex> recv_buf_sizes(static_cast<std::size_t>(total_recv_sizes));
    MPI_Alltoallv(send_buf_sizes.data(), send_counts_sizes.data(), send_displs_sizes.data(), MPI_INT64_T,
                  recv_buf_sizes.data(), recv_counts_sizes.data(), recv_displs_sizes.data(), MPI_INT64_T,
                  comm_);

    // Exchange flattened column lists.
    std::vector<int> send_counts_cols(n_ranks_, 0);
    for (int r = 0; r < n_ranks_; ++r) {
        const auto& cols = send_row_cols[static_cast<std::size_t>(r)];
        FE_CHECK_ARG(cols.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                     "Too many ghost row column entries for MPI_Alltoallv");
        send_counts_cols[r] = static_cast<int>(cols.size());
    }

    std::vector<int> recv_counts_cols(n_ranks_, 0);
    MPI_Alltoall(send_counts_cols.data(), 1, MPI_INT,
                 recv_counts_cols.data(), 1, MPI_INT, comm_);

    std::vector<int> send_displs_cols(n_ranks_, 0);
    std::vector<int> recv_displs_cols(n_ranks_, 0);
    for (int r = 1; r < n_ranks_; ++r) {
        send_displs_cols[r] = send_displs_cols[r - 1] + send_counts_cols[r - 1];
        recv_displs_cols[r] = recv_displs_cols[r - 1] + recv_counts_cols[r - 1];
    }

    const int total_send_cols = send_displs_cols.back() + send_counts_cols.back();
    const int total_recv_cols = recv_displs_cols.back() + recv_counts_cols.back();

    std::vector<GlobalIndex> send_buf_cols(static_cast<std::size_t>(total_send_cols));
    for (int r = 0; r < n_ranks_; ++r) {
        int pos = send_displs_cols[r];
        for (GlobalIndex col : send_row_cols[static_cast<std::size_t>(r)]) {
            send_buf_cols[static_cast<std::size_t>(pos++)] = col;
        }
    }

    std::vector<GlobalIndex> recv_buf_cols(static_cast<std::size_t>(total_recv_cols));
    MPI_Alltoallv(send_buf_cols.data(), send_counts_cols.data(), send_displs_cols.data(), MPI_INT64_T,
                  recv_buf_cols.data(), recv_counts_cols.data(), recv_displs_cols.data(), MPI_INT64_T,
                  comm_);

    // ---------------------------------------------------------------------
    // Phase 3: reconstruct ghost row CSR in deterministic order.
    // ---------------------------------------------------------------------
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> ghost_row_to_cols;
    ghost_row_to_cols.reserve(ghost_rows_requested.size());

    for (int owner = 0; owner < n_ranks_; ++owner) {
        const auto& rows = rows_by_owner[static_cast<std::size_t>(owner)];
        if (rows.empty()) continue;

        const int n_sizes = recv_counts_sizes[owner];
        FE_CHECK_ARG(n_sizes == static_cast<int>(rows.size()),
                     "Ghost row exchange size mismatch for owner rank " + std::to_string(owner));

        std::size_t sizes_pos = static_cast<std::size_t>(recv_displs_sizes[owner]);
        std::size_t cols_pos = static_cast<std::size_t>(recv_displs_cols[owner]);

        for (std::size_t i = 0; i < rows.size(); ++i) {
            const GlobalIndex row = rows[i];
            const GlobalIndex ncols_g = recv_buf_sizes[sizes_pos + i];
            FE_CHECK_ARG(ncols_g >= 0, "Ghost row exchange returned negative column count");
            const std::size_t ncols = static_cast<std::size_t>(ncols_g);
            FE_CHECK_ARG(cols_pos + ncols <= recv_buf_cols.size(),
                         "Ghost row exchange column payload truncated");

            std::vector<GlobalIndex> cols;
            cols.reserve(ncols);
            for (std::size_t j = 0; j < ncols; ++j) {
                cols.push_back(recv_buf_cols[cols_pos + j]);
            }
            cols_pos += ncols;
            ghost_row_to_cols.emplace(row, std::move(cols));
        }
    }

    std::vector<GlobalIndex> ghost_row_map;
    ghost_row_map.reserve(ghost_row_to_cols.size());
    for (const auto& kv : ghost_row_to_cols) {
        ghost_row_map.push_back(kv.first);
    }
    std::sort(ghost_row_map.begin(), ghost_row_map.end());
    ghost_row_map.erase(std::unique(ghost_row_map.begin(), ghost_row_map.end()), ghost_row_map.end());

    std::vector<GlobalIndex> ghost_row_ptr;
    std::vector<GlobalIndex> ghost_row_cols;
    ghost_row_ptr.reserve(ghost_row_map.size() + 1);
    ghost_row_ptr.push_back(0);

    for (GlobalIndex row : ghost_row_map) {
        auto it = ghost_row_to_cols.find(row);
        FE_CHECK_ARG(it != ghost_row_to_cols.end(), "Missing ghost row payload for row " + std::to_string(row));
        const auto& cols = it->second;
        ghost_row_cols.insert(ghost_row_cols.end(), cols.begin(), cols.end());
        ghost_row_ptr.push_back(static_cast<GlobalIndex>(ghost_row_cols.size()));
    }

    pattern.setGhostRows(std::move(ghost_row_map),
                         std::move(ghost_row_ptr),
                         std::move(ghost_row_cols));
#else
    FE_NOT_IMPLEMENTED("ParallelSparsityManager ghost row sparsity exchange (requires MPI)");
#endif
}

void ParallelSparsityManager::detectGhostDofs(const DistributedSparsityPattern& pattern,
                                              const DofOwnership& col_own) {
    ghost_cols_.clear();
    ghost_cols_by_owner_.clear();
    ghost_cols_by_owner_.resize(static_cast<std::size_t>(n_ranks_));

    if (!pattern.isFinalized()) {
        return;
    }

    auto ghost_map = pattern.getGhostColMap();
    ghost_cols_.assign(ghost_map.begin(), ghost_map.end());

    for (GlobalIndex col : ghost_cols_) {
        const int owner = col_own.getOwner(col);
        FE_CHECK_ARG(owner >= 0 && owner < n_ranks_, "Column owner rank out of range");
        ghost_cols_by_owner_[static_cast<std::size_t>(owner)].push_back(col);
    }
}

bool ParallelSparsityManager::validateParallel(
    [[maybe_unused]] const DistributedSparsityPattern& pattern) const {

#if FE_HAS_MPI
    if (isSerial()) {
        return pattern.validate();
    }

    // Validate locally first
    bool local_valid = pattern.validate();

    // Reduce to check if all ranks are valid
    int local_flag = local_valid ? 1 : 0;
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MIN, comm_);

    return global_flag == 1;
#else
    return pattern.validate();
#endif
}

GlobalIndex ParallelSparsityManager::computeGlobalNnz(
    const DistributedSparsityPattern& pattern) const {

    GlobalIndex local_nnz = pattern.getLocalNnz();

#if FE_HAS_MPI
    if (!isSerial()) {
        GlobalIndex global_nnz = 0;
        MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, comm_);
        return global_nnz;
    }
#endif

    return local_nnz;
}

DistributedSparsityStats ParallelSparsityManager::computeGlobalStats(
    const DistributedSparsityPattern& pattern) const {

    DistributedSparsityStats stats = pattern.computeStats();

#if FE_HAS_MPI
    if (!isSerial()) {
        // Reduce various statistics
        MPI_Allreduce(MPI_IN_PLACE, &stats.global_nnz, 1, MPI_INT64_T, MPI_SUM, comm_);

        // Could add more reductions for min/max/avg statistics across ranks
    }
#endif

    return stats;
}

// ============================================================================
// Convenience functions
// ============================================================================

#if FE_HAS_MPI
DistributedSparsityPattern buildDistributedPattern(
    const dofs::DofMap& dof_map,
    MPI_Comm comm,
    const SparsityBuildOptions& options) {

    int my_rank, n_ranks;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    GlobalIndex n_global_dofs = dof_map.getNumDofs();

    // Compute balanced distribution
    GlobalIndex base_count = n_global_dofs / n_ranks;
    GlobalIndex remainder = n_global_dofs % n_ranks;
    GlobalIndex first_owned = my_rank * base_count + std::min(static_cast<GlobalIndex>(my_rank), remainder);
    GlobalIndex n_owned = base_count + (my_rank < remainder ? 1 : 0);

    DistributedSparsityBuilder builder(dof_map, first_owned, n_owned, n_global_dofs);
    builder.setOptions(options);
    return builder.build();
}
#endif

} // namespace sparsity
} // namespace FE
} // namespace svmp
