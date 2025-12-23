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

#include "ConstraintSparsityAugmenter.h"
#include "Dofs/DofConstraints.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <queue>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// DofConstraintsAdapter implementation
// ============================================================================

DofConstraintsAdapter::DofConstraintsAdapter(const dofs::DofConstraints& constraints)
    : constraints_(&constraints)
{
}

bool DofConstraintsAdapter::isConstrained(GlobalIndex dof) const {
    FE_THROW_IF(!constraints_, InvalidArgumentException, "DofConstraintsAdapter not initialized");
    return constraints_->isConstrained(dof);
}

std::vector<GlobalIndex> DofConstraintsAdapter::getMasterDofs(GlobalIndex constrained_dof) const {
    FE_THROW_IF(!constraints_, InvalidArgumentException, "DofConstraintsAdapter not initialized");

    auto line_opt = constraints_->getConstraintLine(constrained_dof);
    if (!line_opt.has_value()) {
        return {};
    }

    const auto& line = *line_opt;
    std::vector<GlobalIndex> masters;
    masters.reserve(line.entries.size());
    for (const auto& entry : line.entries) {
        masters.push_back(entry.dof);
    }

    // Deterministic ordering and deduplication (defensive)
    std::sort(masters.begin(), masters.end());
    masters.erase(std::unique(masters.begin(), masters.end()), masters.end());
    return masters;
}

std::vector<GlobalIndex> DofConstraintsAdapter::getAllConstrainedDofs() const {
    FE_THROW_IF(!constraints_, InvalidArgumentException, "DofConstraintsAdapter not initialized");

    const auto& set = constraints_->getConstrainedDofs();
    std::vector<GlobalIndex> dofs;
    dofs.reserve(static_cast<std::size_t>(set.size()));
    for (GlobalIndex dof : set) {
        dofs.push_back(dof);
    }
    return dofs;
}

std::size_t DofConstraintsAdapter::numConstraints() const {
    FE_THROW_IF(!constraints_, InvalidArgumentException, "DofConstraintsAdapter not initialized");
    return constraints_->numConstraints();
}

// ============================================================================
// SimpleConstraintSet implementation
// ============================================================================

void SimpleConstraintSet::addDirichlet(GlobalIndex dof) {
    constraints_[dof] = {};  // Empty master list = Dirichlet
}

void SimpleConstraintSet::addConstraint(GlobalIndex slave, GlobalIndex master) {
    constraints_[slave] = {master};
}

void SimpleConstraintSet::addConstraint(GlobalIndex slave, std::span<const GlobalIndex> masters) {
    constraints_[slave] = std::vector<GlobalIndex>(masters.begin(), masters.end());
}

void SimpleConstraintSet::addConstraint(const SparsityConstraint& constraint) {
    constraints_[constraint.constrained_dof] = constraint.master_dofs;
}

void SimpleConstraintSet::clear() {
    constraints_.clear();
}

bool SimpleConstraintSet::isConstrained(GlobalIndex dof) const {
    return constraints_.find(dof) != constraints_.end();
}

std::vector<GlobalIndex> SimpleConstraintSet::getMasterDofs(GlobalIndex dof) const {
    auto it = constraints_.find(dof);
    if (it == constraints_.end()) {
        return {};
    }
    return it->second;
}

std::vector<GlobalIndex> SimpleConstraintSet::getAllConstrainedDofs() const {
    std::vector<GlobalIndex> result;
    result.reserve(constraints_.size());
    for (const auto& [dof, masters] : constraints_) {
        result.push_back(dof);
    }
    // Sort for determinism
    std::sort(result.begin(), result.end());
    return result;
}

std::size_t SimpleConstraintSet::numConstraints() const {
    return constraints_.size();
}

// ============================================================================
// ConstraintSparsityAugmenter construction
// ============================================================================

ConstraintSparsityAugmenter::ConstraintSparsityAugmenter(
    std::shared_ptr<IConstraintQuery> constraint_query)
    : constraint_query_(std::move(constraint_query))
{
}

ConstraintSparsityAugmenter::ConstraintSparsityAugmenter(
    std::vector<SparsityConstraint> constraints)
{
    auto simple_set = std::make_shared<SimpleConstraintSet>();
    for (const auto& c : constraints) {
        simple_set->addConstraint(c);
    }
    constraint_query_ = std::move(simple_set);
}

// ============================================================================
// Configuration
// ============================================================================

void ConstraintSparsityAugmenter::setConstraints(
    std::shared_ptr<IConstraintQuery> constraint_query)
{
    constraint_query_ = std::move(constraint_query);
}

void ConstraintSparsityAugmenter::setConstraints(
    std::vector<SparsityConstraint> constraints)
{
    auto simple_set = std::make_shared<SimpleConstraintSet>();
    for (const auto& c : constraints) {
        simple_set->addConstraint(c);
    }
    constraint_query_ = std::move(simple_set);
}

// ============================================================================
// Augmentation methods
// ============================================================================

AugmentationStats ConstraintSparsityAugmenter::augment(
    SparsityPattern& pattern,
    std::optional<AugmentationMode> mode)
{
    FE_THROW_IF(pattern.isFinalized(), InvalidArgumentException,
                "Cannot augment finalized pattern - must be in Building state");
    FE_THROW_IF(!constraint_query_, InvalidArgumentException,
                "No constraints configured");

    // Reset statistics
    last_stats_ = AugmentationStats{};
    last_stats_.original_nnz = pattern.getNnz();

    // Count constraint types
    auto constrained_dofs = constraint_query_->getAllConstrainedDofs();
    last_stats_.n_constraints = static_cast<GlobalIndex>(constrained_dofs.size());

    for (GlobalIndex dof : constrained_dofs) {
        auto masters = constraint_query_->getMasterDofs(dof);
        if (masters.empty()) {
            last_stats_.n_dirichlet++;
        } else if (masters.size() == 1) {
            last_stats_.n_periodic++;
        } else {
            last_stats_.n_multipoint++;
        }
    }

    // Select mode
    AugmentationMode actual_mode = mode.value_or(options_.mode);

    // Perform augmentation based on mode
    switch (actual_mode) {
        case AugmentationMode::EliminationFill:
            augmentEliminationFill(pattern);
            break;

        case AugmentationMode::KeepRowsSetDiag:
            augmentKeepRowsSetDiag(pattern);
            break;

        case AugmentationMode::ReducedSystem:
            FE_THROW(InvalidArgumentException,
                     "ReducedSystem mode not supported for in-place augmentation. "
                     "Use buildReducedPattern() instead.");
            break;
    }

    last_stats_.augmented_nnz = pattern.getNnz();
    last_stats_.n_fill_entries = last_stats_.augmented_nnz - last_stats_.original_nnz;

    return last_stats_;
}

AugmentationStats ConstraintSparsityAugmenter::augment(
    DistributedSparsityPattern& pattern,
    std::optional<AugmentationMode> mode)
{
    FE_THROW_IF(pattern.isFinalized(), InvalidArgumentException,
                "Cannot augment finalized pattern");
    FE_THROW_IF(!constraint_query_, InvalidArgumentException,
                "No constraints configured");

    // Reset statistics
    last_stats_ = AugmentationStats{};
    last_stats_.original_nnz = pattern.getLocalNnz();

    const auto constrained_dofs = constraint_query_->getAllConstrainedDofs();
    last_stats_.n_constraints = static_cast<GlobalIndex>(constrained_dofs.size());

    for (GlobalIndex dof : constrained_dofs) {
        auto masters = constraint_query_->getMasterDofs(dof);
        if (masters.empty()) {
            last_stats_.n_dirichlet++;
        } else if (masters.size() == 1) {
            last_stats_.n_periodic++;
        } else {
            last_stats_.n_multipoint++;
        }
    }

    const AugmentationMode actual_mode = mode.value_or(options_.mode);

    if (actual_mode == AugmentationMode::ReducedSystem) {
        FE_THROW(InvalidArgumentException,
                 "ReducedSystem mode is not supported for in-place distributed augmentation. "
                 "Use buildReducedDistributedPattern() instead.");
    }

    const auto& owned_rows = pattern.ownedRows();
    const auto& owned_cols = pattern.ownedCols();
    const bool square = pattern.isSquare();

    // Precompute master lists for constrained DOFs.
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> master_map;
    master_map.reserve(constrained_dofs.size());

    for (GlobalIndex cdof : constrained_dofs) {
        auto masters = options_.compute_transitive_closure
            ? getTransitiveMasters(cdof)
            : constraint_query_->getMasterDofs(cdof);

        std::sort(masters.begin(), masters.end());
        masters.erase(std::unique(masters.begin(), masters.end()), masters.end());
        master_map.emplace(cdof, std::move(masters));
    }

    if (actual_mode == AugmentationMode::KeepRowsSetDiag) {
        for (GlobalIndex cdof : constrained_dofs) {
            if (!owned_rows.contains(cdof)) continue;

            const auto local_row = static_cast<std::size_t>(cdof - owned_rows.first);
            if (pattern.building_rows_[local_row].insert(cdof).second) {
                last_stats_.n_diagonal_added++;
            }
        }

        last_stats_.augmented_nnz = pattern.getLocalNnz();
        last_stats_.n_fill_entries = last_stats_.augmented_nnz - last_stats_.original_nnz;
        return last_stats_;
    }

    // ---------------------------------------------------------------------
    // EliminationFill: add fill in owned rows that couple to constrained columns,
    // and propagate constrained-row couplings to owned master rows.
    // ---------------------------------------------------------------------

    // Pass 1: for every owned row, if it contains constrained columns, add the
    // associated master DOFs to that same row (O(nnz) with hash lookups).
    for (GlobalIndex local_row = 0; local_row < pattern.numOwnedRows(); ++local_row) {
        const GlobalIndex global_row = owned_rows.first + local_row;
        auto& row_set = pattern.building_rows_[static_cast<std::size_t>(local_row)];

        // Ensure diagonal for constrained rows if requested.
        if (options_.ensure_diagonal && master_map.find(global_row) != master_map.end()) {
            if (row_set.insert(global_row).second) {
                last_stats_.n_diagonal_added++;
            }
        }

        // Snapshot of columns to avoid extra passes over newly inserted entries.
        std::vector<GlobalIndex> cols_snapshot;
        cols_snapshot.reserve(row_set.size());
        for (GlobalIndex col : row_set) {
            cols_snapshot.push_back(col);
        }

        for (GlobalIndex col : cols_snapshot) {
            const auto it = master_map.find(col);
            if (it == master_map.end()) continue;

            for (GlobalIndex master : it->second) {
                if (master < 0 || master >= pattern.globalCols()) continue;
                if (!options_.include_ghost_columns && !owned_cols.contains(master)) continue;
                const bool inserted = row_set.insert(master).second;

                if (inserted && options_.symmetric_fill && square && owned_rows.contains(master)) {
                    auto& master_row_set = pattern.building_rows_[
                        static_cast<std::size_t>(master - owned_rows.first)];
                    if (options_.include_ghost_columns || owned_cols.contains(global_row)) {
                        master_row_set.insert(global_row);
                    }
                }
            }
        }
    }

    // Pass 2: for each owned constrained row, add its couplings to owned master rows.
    for (GlobalIndex cdof : constrained_dofs) {
        if (!owned_rows.contains(cdof)) continue;

        const auto it = master_map.find(cdof);
        if (it == master_map.end() || it->second.empty()) continue;

        const auto& slave_row_set = pattern.building_rows_[
            static_cast<std::size_t>(cdof - owned_rows.first)];

        for (GlobalIndex master : it->second) {
            if (!owned_rows.contains(master)) continue;

            auto& master_row_set = pattern.building_rows_[
                static_cast<std::size_t>(master - owned_rows.first)];

            for (GlobalIndex col : slave_row_set) {
                if (!options_.include_ghost_columns && !owned_cols.contains(col)) continue;
                const bool inserted = master_row_set.insert(col).second;
                if (inserted && options_.symmetric_fill && square && owned_rows.contains(col)) {
                    auto& col_row_set = pattern.building_rows_[
                        static_cast<std::size_t>(col - owned_rows.first)];
                    if (options_.include_ghost_columns || owned_cols.contains(master)) {
                        col_row_set.insert(master);
                    }
                }
            }
        }
    }

    last_stats_.augmented_nnz = pattern.getLocalNnz();
    last_stats_.n_fill_entries = last_stats_.augmented_nnz - last_stats_.original_nnz;

    return last_stats_;
}

SparsityPattern ConstraintSparsityAugmenter::buildReducedPattern(
    const SparsityPattern& original)
{
    FE_THROW_IF(!constraint_query_, InvalidArgumentException,
                "No constraints configured");

    return buildReducedSystemPattern(original);
}

#if FE_HAS_MPI
ReducedDistributedPatternResult ConstraintSparsityAugmenter::buildReducedDistributedPattern(
    const DistributedSparsityPattern& original,
    MPI_Comm comm) const
{
    FE_THROW_IF(!constraint_query_, InvalidArgumentException,
                "No constraints configured");
    FE_THROW_IF(!original.isFinalized(), InvalidArgumentException,
                "Reduced distributed pattern requires a finalized original pattern");
    FE_CHECK_ARG(original.isSquare(), "ReducedSystem mode requires a square pattern");

    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    const IndexRange owned_full = original.ownedRows();
    const GlobalIndex n_full = original.globalRows();

    // Gather full ownership ranges to compute a global owner() for full DOFs.
    std::vector<GlobalIndex> full_ranges(static_cast<std::size_t>(2 * n_ranks));
    const GlobalIndex send_range[2] = {owned_full.first, owned_full.last};
    MPI_Allgather(send_range, 2, MPI_INT64_T,
                  full_ranges.data(), 2, MPI_INT64_T, comm);

    std::vector<GlobalIndex> full_offsets(static_cast<std::size_t>(n_ranks) + 1, 0);
    FE_CHECK_ARG(full_ranges[0] == 0, "Full ownership must start at 0 on rank 0");
    for (int r = 0; r < n_ranks; ++r) {
        const GlobalIndex first = full_ranges[static_cast<std::size_t>(2 * r)];
        const GlobalIndex last = full_ranges[static_cast<std::size_t>(2 * r + 1)];
        FE_CHECK_ARG(first >= 0 && last >= first, "Invalid owned range gathered for rank " + std::to_string(r));
        if (r == 0) {
            full_offsets[0] = first;
        } else {
            FE_CHECK_ARG(first == full_offsets[static_cast<std::size_t>(r)],
                         "Full ownership ranges must be contiguous across ranks");
        }
        full_offsets[static_cast<std::size_t>(r) + 1] = last;
    }
    FE_CHECK_ARG(full_offsets.front() == 0, "Full ownership offsets must start at 0");
    FE_CHECK_ARG(full_offsets.back() == n_full, "Full ownership offsets do not match global size");

    const auto owner_of_full = [&](GlobalIndex dof) -> int {
        FE_CHECK_ARG(dof >= 0 && dof < n_full, "Full DOF out of range");
        auto it = std::upper_bound(full_offsets.begin(), full_offsets.end(), dof);
        FE_CHECK_ARG(it != full_offsets.begin(), "Full ownership offsets malformed");
        const int owner = static_cast<int>(std::distance(full_offsets.begin(), it)) - 1;
        FE_CHECK_ARG(owner >= 0 && owner < n_ranks, "Computed owner out of range");
        return owner;
    };

    // Build local reduced ownership via compacting unconstrained owned DOFs.
    GlobalIndex n_unconstrained_local = 0;
    for (GlobalIndex dof = owned_full.first; dof < owned_full.last; ++dof) {
        if (!constraint_query_->isConstrained(dof)) {
            ++n_unconstrained_local;
        }
    }

    std::vector<GlobalIndex> unconstrained_counts(static_cast<std::size_t>(n_ranks), 0);
    MPI_Allgather(&n_unconstrained_local, 1, MPI_INT64_T,
                  unconstrained_counts.data(), 1, MPI_INT64_T, comm);

    std::vector<GlobalIndex> reduced_offsets(static_cast<std::size_t>(n_ranks) + 1, 0);
    for (int r = 0; r < n_ranks; ++r) {
        reduced_offsets[static_cast<std::size_t>(r) + 1] =
            reduced_offsets[static_cast<std::size_t>(r)] + unconstrained_counts[static_cast<std::size_t>(r)];
    }

    const GlobalIndex n_reduced_global = reduced_offsets.back();
    IndexRange owned_reduced{
        reduced_offsets[static_cast<std::size_t>(my_rank)],
        reduced_offsets[static_cast<std::size_t>(my_rank) + 1]
    };

    // Local mapping for owned full DOFs.
    std::unordered_map<GlobalIndex, GlobalIndex> owned_full_to_reduced;
    owned_full_to_reduced.reserve(static_cast<std::size_t>(n_unconstrained_local) * 2);

    std::vector<GlobalIndex> full_to_reduced_owned(
        static_cast<std::size_t>(owned_full.size()), GlobalIndex{-1});
    std::vector<GlobalIndex> reduced_to_full_owned;
    reduced_to_full_owned.reserve(static_cast<std::size_t>(n_unconstrained_local));

    GlobalIndex next_reduced = owned_reduced.first;
    for (GlobalIndex dof = owned_full.first; dof < owned_full.last; ++dof) {
        const std::size_t local = static_cast<std::size_t>(dof - owned_full.first);
        if (constraint_query_->isConstrained(dof)) {
            full_to_reduced_owned[local] = -1;
            continue;
        }
        full_to_reduced_owned[local] = next_reduced;
        reduced_to_full_owned.push_back(dof);
        owned_full_to_reduced.emplace(dof, next_reduced);
        ++next_reduced;
    }
    FE_CHECK_ARG(next_reduced == owned_reduced.last, "Reduced ownership range mismatch after compaction");

    // Master map for constraint elimination in reduced system.
    const auto constrained_dofs = constraint_query_->getAllConstrainedDofs();
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> master_map;
    master_map.reserve(constrained_dofs.size());
    for (GlobalIndex cdof : constrained_dofs) {
        master_map[cdof] = options_.compute_transitive_closure
            ? getTransitiveMasters(cdof)
            : constraint_query_->getMasterDofs(cdof);
    }

    // ---------------------------------------------------------------------
    // Build reduced-index mapping for non-owned unconstrained DOFs referenced locally.
    // ---------------------------------------------------------------------
    std::unordered_set<GlobalIndex> query_set;
    query_set.reserve(1024);

    const auto ensure_query = [&](GlobalIndex dof) {
        if (dof < 0 || dof >= n_full) return;
        if (constraint_query_->isConstrained(dof)) return;
        if (owned_full.contains(dof)) return;
        query_set.insert(dof);
    };

    for (GlobalIndex full_row = owned_full.first; full_row < owned_full.last; ++full_row) {
        auto row_cols = original.getOwnedRowGlobalCols(full_row);
        for (GlobalIndex full_col : row_cols) {
            if (constraint_query_->isConstrained(full_col)) {
                const auto& masters_col = master_map[full_col];
                for (GlobalIndex mc : masters_col) {
                    ensure_query(mc);
                }
            } else {
                ensure_query(full_col);
            }
        }

        if (constraint_query_->isConstrained(full_row)) {
            const auto& masters_row = master_map[full_row];
            for (GlobalIndex mr : masters_row) {
                ensure_query(mr);
            }
        }
    }

    std::vector<std::vector<GlobalIndex>> req_by_owner(static_cast<std::size_t>(n_ranks));
    for (GlobalIndex dof : query_set) {
        const int owner = owner_of_full(dof);
        if (owner == my_rank) continue;
        req_by_owner[static_cast<std::size_t>(owner)].push_back(dof);
    }
    for (auto& v : req_by_owner) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // Exchange requests.
    std::vector<int> send_counts_req(n_ranks, 0);
    for (int r = 0; r < n_ranks; ++r) {
        FE_CHECK_ARG(req_by_owner[static_cast<std::size_t>(r)].size() <=
                         static_cast<std::size_t>(std::numeric_limits<int>::max()),
                     "Too many reduced-index mapping requests for MPI_Alltoallv");
        send_counts_req[r] = static_cast<int>(req_by_owner[static_cast<std::size_t>(r)].size());
    }

    std::vector<int> recv_counts_req(n_ranks, 0);
    MPI_Alltoall(send_counts_req.data(), 1, MPI_INT,
                 recv_counts_req.data(), 1, MPI_INT, comm);

    std::vector<int> send_displs_req(n_ranks, 0);
    std::vector<int> recv_displs_req(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) {
        send_displs_req[r] = send_displs_req[r - 1] + send_counts_req[r - 1];
        recv_displs_req[r] = recv_displs_req[r - 1] + recv_counts_req[r - 1];
    }

    const int total_send_req = send_displs_req.back() + send_counts_req.back();
    const int total_recv_req = recv_displs_req.back() + recv_counts_req.back();

    std::vector<GlobalIndex> send_buf_req(static_cast<std::size_t>(total_send_req));
    for (int r = 0; r < n_ranks; ++r) {
        int pos = send_displs_req[r];
        for (GlobalIndex dof : req_by_owner[static_cast<std::size_t>(r)]) {
            send_buf_req[static_cast<std::size_t>(pos++)] = dof;
        }
    }

    std::vector<GlobalIndex> recv_buf_req(static_cast<std::size_t>(total_recv_req));
    MPI_Alltoallv(send_buf_req.data(), send_counts_req.data(), send_displs_req.data(), MPI_INT64_T,
                  recv_buf_req.data(), recv_counts_req.data(), recv_displs_req.data(), MPI_INT64_T,
                  comm);

    // Build response: reduced indices for each requested DOF.
    std::vector<std::vector<GlobalIndex>> resp_by_rank(static_cast<std::size_t>(n_ranks));
    for (int r = 0; r < n_ranks; ++r) {
        const int n_req = recv_counts_req[r];
        if (n_req <= 0) continue;

        auto& resp = resp_by_rank[static_cast<std::size_t>(r)];
        resp.resize(static_cast<std::size_t>(n_req), GlobalIndex{-1});
        for (int i = 0; i < n_req; ++i) {
            const GlobalIndex dof = recv_buf_req[static_cast<std::size_t>(recv_displs_req[r] + i)];
            auto it = owned_full_to_reduced.find(dof);
            resp[static_cast<std::size_t>(i)] = (it != owned_full_to_reduced.end()) ? it->second : GlobalIndex{-1};
        }
    }

    // Exchange responses.
    std::vector<int> send_counts_resp(n_ranks, 0);
    for (int r = 0; r < n_ranks; ++r) {
        send_counts_resp[r] = static_cast<int>(resp_by_rank[static_cast<std::size_t>(r)].size());
    }

    std::vector<int> recv_counts_resp(n_ranks, 0);
    MPI_Alltoall(send_counts_resp.data(), 1, MPI_INT,
                 recv_counts_resp.data(), 1, MPI_INT, comm);

    std::vector<int> send_displs_resp(n_ranks, 0);
    std::vector<int> recv_displs_resp(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) {
        send_displs_resp[r] = send_displs_resp[r - 1] + send_counts_resp[r - 1];
        recv_displs_resp[r] = recv_displs_resp[r - 1] + recv_counts_resp[r - 1];
    }

    const int total_send_resp = send_displs_resp.back() + send_counts_resp.back();
    const int total_recv_resp = recv_displs_resp.back() + recv_counts_resp.back();

    std::vector<GlobalIndex> send_buf_resp(static_cast<std::size_t>(total_send_resp));
    for (int r = 0; r < n_ranks; ++r) {
        int pos = send_displs_resp[r];
        for (GlobalIndex idx : resp_by_rank[static_cast<std::size_t>(r)]) {
            send_buf_resp[static_cast<std::size_t>(pos++)] = idx;
        }
    }

    std::vector<GlobalIndex> recv_buf_resp(static_cast<std::size_t>(total_recv_resp));
    MPI_Alltoallv(send_buf_resp.data(), send_counts_resp.data(), send_displs_resp.data(), MPI_INT64_T,
                  recv_buf_resp.data(), recv_counts_resp.data(), recv_displs_resp.data(), MPI_INT64_T,
                  comm);

    std::unordered_map<GlobalIndex, GlobalIndex> remote_full_to_reduced;
    remote_full_to_reduced.reserve(query_set.size() * 2);
    for (int owner = 0; owner < n_ranks; ++owner) {
        const auto& dofs = req_by_owner[static_cast<std::size_t>(owner)];
        if (dofs.empty()) continue;

        const int n_resp = recv_counts_resp[owner];
        FE_CHECK_ARG(n_resp == static_cast<int>(dofs.size()),
                     "Reduced-index mapping response size mismatch for rank " + std::to_string(owner));

        const std::size_t base = static_cast<std::size_t>(recv_displs_resp[owner]);
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const GlobalIndex reduced_idx = recv_buf_resp[base + i];
            FE_CHECK_ARG(reduced_idx >= 0,
                         "Owner did not provide a reduced index for full DOF " + std::to_string(dofs[i]));
            remote_full_to_reduced.emplace(dofs[i], reduced_idx);
        }
    }

    const auto full_to_reduced = [&](GlobalIndex dof) -> GlobalIndex {
        if (constraint_query_->isConstrained(dof)) {
            return -1;
        }
        if (owned_full.contains(dof)) {
            auto it = owned_full_to_reduced.find(dof);
            FE_CHECK_ARG(it != owned_full_to_reduced.end(), "Missing local reduced index");
            return it->second;
        }
        auto it = remote_full_to_reduced.find(dof);
        FE_CHECK_ARG(it != remote_full_to_reduced.end(),
                     "Missing reduced index for non-owned DOF " + std::to_string(dof));
        return it->second;
    };

    const auto owner_of_reduced = [&](GlobalIndex idx) -> int {
        FE_CHECK_ARG(idx >= 0 && idx < n_reduced_global, "Reduced index out of range");
        auto it = std::upper_bound(reduced_offsets.begin(), reduced_offsets.end(), idx);
        FE_CHECK_ARG(it != reduced_offsets.begin(), "Reduced ownership offsets malformed");
        const int owner = static_cast<int>(std::distance(reduced_offsets.begin(), it)) - 1;
        FE_CHECK_ARG(owner >= 0 && owner < n_ranks, "Computed reduced owner out of range");
        return owner;
    };

    // ---------------------------------------------------------------------
    // Build reduced distributed pattern with MPI exchange for remote owned rows.
    // ---------------------------------------------------------------------
    DistributedSparsityPattern reduced_pattern(owned_reduced, owned_reduced, n_reduced_global, n_reduced_global);

    struct Pair {
        GlobalIndex row;
        GlobalIndex col;
        bool operator<(const Pair& other) const noexcept {
            return (row < other.row) || (row == other.row && col < other.col);
        }
        bool operator==(const Pair& other) const noexcept {
            return row == other.row && col == other.col;
        }
    };

    std::vector<std::vector<Pair>> send_pairs(static_cast<std::size_t>(n_ranks));

    const auto insert_entry = [&](GlobalIndex rrow, GlobalIndex rcol) {
        if (rrow < 0 || rcol < 0) return;
        const int owner = owner_of_reduced(rrow);
        if (owner == my_rank) {
            if (!owned_reduced.contains(rrow)) {
                return;
            }
            reduced_pattern.addEntry(rrow, rcol);
        } else {
            send_pairs[static_cast<std::size_t>(owner)].push_back({rrow, rcol});
        }
    };

    for (GlobalIndex full_row = owned_full.first; full_row < owned_full.last; ++full_row) {
        const bool row_constrained = constraint_query_->isConstrained(full_row);
        const auto row_cols = original.getOwnedRowGlobalCols(full_row);

        if (!row_constrained) {
            const GlobalIndex rrow = full_to_reduced(full_row);
            if (rrow < 0) continue;

            for (GlobalIndex full_col : row_cols) {
                const GlobalIndex rcol = full_to_reduced(full_col);
                if (rcol >= 0) {
                    insert_entry(rrow, rcol);
                } else {
                    const auto& masters_col = master_map[full_col];
                    for (GlobalIndex mc : masters_col) {
                        const GlobalIndex rmc = full_to_reduced(mc);
                        if (rmc >= 0) {
                            insert_entry(rrow, rmc);
                        }
                    }
                }
            }
        } else {
            const auto& masters_row = master_map[full_row];
            for (GlobalIndex mr : masters_row) {
                const GlobalIndex rmr = full_to_reduced(mr);
                if (rmr < 0) continue;

                for (GlobalIndex full_col : row_cols) {
                    const GlobalIndex rcol = full_to_reduced(full_col);
                    if (rcol >= 0) {
                        insert_entry(rmr, rcol);
                    } else {
                        const auto& masters_col = master_map[full_col];
                        for (GlobalIndex mc : masters_col) {
                            const GlobalIndex rmc = full_to_reduced(mc);
                            if (rmc >= 0) {
                                insert_entry(rmr, rmc);
                            }
                        }
                    }
                }
            }
        }
    }

    if (n_ranks > 1) {
        std::vector<int> send_counts(n_ranks, 0);
        for (int r = 0; r < n_ranks; ++r) {
            auto& pairs = send_pairs[static_cast<std::size_t>(r)];
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

            const std::size_t n_vals = pairs.size() * 2;
            FE_CHECK_ARG(n_vals <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                         "Too many reduced pattern entries to communicate");
            send_counts[r] = static_cast<int>(n_vals);
        }

        std::vector<int> recv_counts(n_ranks, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, comm);

        std::vector<int> send_displs(n_ranks, 0);
        std::vector<int> recv_displs(n_ranks, 0);
        for (int r = 1; r < n_ranks; ++r) {
            send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }

        const int total_send = send_displs.back() + send_counts.back();
        const int total_recv = recv_displs.back() + recv_counts.back();

        std::vector<GlobalIndex> send_buf(static_cast<std::size_t>(total_send));
        for (int r = 0; r < n_ranks; ++r) {
            int pos = send_displs[r];
            const auto& pairs = send_pairs[static_cast<std::size_t>(r)];
            for (const auto& p : pairs) {
                send_buf[static_cast<std::size_t>(pos++)] = p.row;
                send_buf[static_cast<std::size_t>(pos++)] = p.col;
            }
        }

        std::vector<GlobalIndex> recv_buf(static_cast<std::size_t>(total_recv));
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                      recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                      comm);

        FE_CHECK_ARG(total_recv % 2 == 0, "Reduced pattern receive buffer malformed");
        for (int i = 0; i < total_recv; i += 2) {
            const GlobalIndex rrow = recv_buf[static_cast<std::size_t>(i)];
            const GlobalIndex rcol = recv_buf[static_cast<std::size_t>(i + 1)];
            if (!owned_reduced.contains(rrow)) continue;
            reduced_pattern.addEntry(rrow, rcol);
        }
    }

    if (options_.ensure_diagonal) {
        reduced_pattern.ensureDiagonal();
    }

    reduced_pattern.finalize();

    ReducedDistributedPatternResult result;
    result.pattern = std::move(reduced_pattern);
    result.owned_reduced_range = owned_reduced;
    result.global_reduced_size = n_reduced_global;
    result.full_to_reduced_owned = std::move(full_to_reduced_owned);
    result.reduced_to_full_owned = std::move(reduced_to_full_owned);
    return result;
}
#endif

// ============================================================================
// Internal implementation
// ============================================================================

void ConstraintSparsityAugmenter::augmentEliminationFill(SparsityPattern& pattern) {
    const GlobalIndex n_rows = pattern.numRows();
    const GlobalIndex n_cols = pattern.numCols();

    auto constrained_dofs = constraint_query_->getAllConstrainedDofs();

    // Precompute master lists
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> master_map;
    for (GlobalIndex cdof : constrained_dofs) {
        master_map[cdof] = options_.compute_transitive_closure
            ? getTransitiveMasters(cdof)
            : constraint_query_->getMasterDofs(cdof);
    }

    // 1. Initial setup: Ensure diagonals and add explicit constraint couplings (u_s, u_m)
    for (GlobalIndex cdof : constrained_dofs) {
        if (cdof < 0 || cdof >= n_rows) continue;
        
        auto& row_set = pattern.row_sets_[static_cast<std::size_t>(cdof)];
        
        // Ensure diagonal if requested
        if (options_.ensure_diagonal && cdof < n_cols) {
            if (row_set.insert(cdof).second) {
                last_stats_.n_diagonal_added++;
            }
        }

        // Add explicit couplings to masters: (u_s, u_m)
        const auto& masters = master_map[cdof];
        for (GlobalIndex m : masters) {
            if (m >= 0 && m < n_cols) {
                if (row_set.insert(m).second) {
                    last_stats_.n_fill_entries++;
                }
                
                // If symmetric fill, also add (u_m, u_s)
                if (options_.symmetric_fill && m < n_rows && cdof < n_cols) {
                    if (pattern.row_sets_[static_cast<std::size_t>(m)].insert(cdof).second) {
                        last_stats_.n_fill_entries++;
                    }
                }
            }
        }
    }

    // 2. Rule 1: For each row i having col u_s, add (i, u_m)
    //    If symmetric_fill: also add (u_m, i)
    for (GlobalIndex row = 0; row < n_rows; ++row) {
        auto& row_set = pattern.row_sets_[static_cast<std::size_t>(row)];
        
        // Snapshot to safely iterate while potentially modifying other rows
        std::vector<GlobalIndex> current_cols(row_set.begin(), row_set.end());
        
        for (GlobalIndex col : current_cols) {
            // Check if col is a constrained DOF (u_s)
            auto it = master_map.find(col);
            if (it != master_map.end()) {
                const auto& masters = it->second;
                for (GlobalIndex m : masters) {
                    if (m >= 0 && m < n_cols) {
                        // Add (row, m) i.e. (i, u_m)
                        if (row_set.insert(m).second) {
                            last_stats_.n_fill_entries++;
                        }
                        
                        // Symmetric: add (m, row) i.e. (u_m, i)
                        if (options_.symmetric_fill && m < n_rows && row < n_cols) {
                            if (pattern.row_sets_[static_cast<std::size_t>(m)].insert(row).second) {
                                last_stats_.n_fill_entries++;
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. Rule 2: For each constrained row u_s having col j, add (u_m, j)
    //    If symmetric_fill: also add (j, u_m)
    for (GlobalIndex cdof : constrained_dofs) {
        if (cdof < 0 || cdof >= n_rows) continue;
        
        auto it = master_map.find(cdof);
        if (it == master_map.end()) continue;
        const auto& masters = it->second;
        
        // Snapshot of columns in the constrained row u_s
        const auto& slave_row_set = pattern.row_sets_[static_cast<std::size_t>(cdof)];
        std::vector<GlobalIndex> slave_cols(slave_row_set.begin(), slave_row_set.end());
        
        for (GlobalIndex m : masters) {
            if (m < 0 || m >= n_rows) continue;
            auto& master_row_set = pattern.row_sets_[static_cast<std::size_t>(m)];
            
            for (GlobalIndex j : slave_cols) {
                if (j >= 0 && j < n_cols) {
                    // Add (m, j) i.e. (u_m, j)
                    if (master_row_set.insert(j).second) {
                        last_stats_.n_fill_entries++;
                    }
                    
                    // Symmetric: add (j, m) i.e. (j, u_m)
                    if (options_.symmetric_fill && j < n_rows && m < n_cols) {
                        if (pattern.row_sets_[static_cast<std::size_t>(j)].insert(m).second) {
                            last_stats_.n_fill_entries++;
                        }
                    }
                }
            }
        }
    }
}

void ConstraintSparsityAugmenter::augmentKeepRowsSetDiag(SparsityPattern& pattern) {
    auto constrained_dofs = constraint_query_->getAllConstrainedDofs();
    const GlobalIndex n_rows = pattern.numRows();
    const GlobalIndex n_cols = pattern.numCols();

    for (GlobalIndex cdof : constrained_dofs) {
        // Only add diagonal for valid indices
        if (cdof >= 0 && cdof < n_rows && cdof < n_cols) {
            pattern.addEntry(cdof, cdof);
            last_stats_.n_diagonal_added++;
        }
    }
}

SparsityPattern ConstraintSparsityAugmenter::buildReducedSystemPattern(
    const SparsityPattern& original)
{
    const GlobalIndex n_total = original.numRows();
    FE_CHECK_ARG(original.isSquare(),
                 "ReducedSystem mode requires square pattern");

    // Build mappings
    auto full_to_reduced = getReducedMapping(n_total);
    auto reduced_to_full = getFullMapping(n_total);
    const GlobalIndex n_reduced = static_cast<GlobalIndex>(reduced_to_full.size());

    if (n_reduced == 0) {
        return SparsityPattern(0, 0);
    }

    // Create reduced pattern
    SparsityPattern reduced(n_reduced, n_reduced);

    // Build master map for fast lookup
    auto constrained_dofs = constraint_query_->getAllConstrainedDofs();
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> master_map;
    for (GlobalIndex cdof : constrained_dofs) {
        master_map[cdof] = options_.compute_transitive_closure
            ? getTransitiveMasters(cdof)
            : constraint_query_->getMasterDofs(cdof);
    }

    std::unordered_set<GlobalIndex> constrained_set(
        constrained_dofs.begin(), constrained_dofs.end());

    // Process original pattern
    // We need to handle both finalized and non-finalized patterns
    if (original.isFinalized()) {
        for (GlobalIndex full_row = 0; full_row < n_total; ++full_row) {
            GlobalIndex reduced_row = full_to_reduced[static_cast<std::size_t>(full_row)];

            if (reduced_row < 0) {
                // This row is constrained - distribute to masters
                auto masters_row = master_map[full_row];

                auto row_span = original.getRowSpan(full_row);
                for (GlobalIndex full_col : row_span) {
                    GlobalIndex reduced_col = full_to_reduced[static_cast<std::size_t>(full_col)];

                    if (reduced_col >= 0) {
                        // Column is unconstrained - add from all masters
                        for (GlobalIndex mr : masters_row) {
                            GlobalIndex mr_reduced = full_to_reduced[static_cast<std::size_t>(mr)];
                            if (mr_reduced >= 0) {
                                reduced.addEntry(mr_reduced, reduced_col);
                            }
                        }
                    } else {
                        // Column is also constrained - add from all master combinations
                        auto masters_col = master_map[full_col];
                        for (GlobalIndex mr : masters_row) {
                            GlobalIndex mr_reduced = full_to_reduced[static_cast<std::size_t>(mr)];
                            if (mr_reduced < 0) continue;

                            for (GlobalIndex mc : masters_col) {
                                GlobalIndex mc_reduced = full_to_reduced[static_cast<std::size_t>(mc)];
                                if (mc_reduced >= 0) {
                                    reduced.addEntry(mr_reduced, mc_reduced);
                                }
                            }
                        }
                    }
                }
            } else {
                // Row is unconstrained
                auto row_span = original.getRowSpan(full_row);
                for (GlobalIndex full_col : row_span) {
                    GlobalIndex reduced_col = full_to_reduced[static_cast<std::size_t>(full_col)];

                    if (reduced_col >= 0) {
                        // Direct coupling
                        reduced.addEntry(reduced_row, reduced_col);
                    } else {
                        // Column is constrained - add coupling to masters
                        auto masters_col = master_map[full_col];
                        for (GlobalIndex mc : masters_col) {
                            GlobalIndex mc_reduced = full_to_reduced[static_cast<std::size_t>(mc)];
                            if (mc_reduced >= 0) {
                                reduced.addEntry(reduced_row, mc_reduced);
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Pattern not finalized - iterate differently
        for (GlobalIndex full_row = 0; full_row < n_total; ++full_row) {
            GlobalIndex reduced_row = full_to_reduced[static_cast<std::size_t>(full_row)];

            for (GlobalIndex full_col = 0; full_col < n_total; ++full_col) {
                if (!original.hasEntry(full_row, full_col)) {
                    continue;
                }

                GlobalIndex reduced_col = full_to_reduced[static_cast<std::size_t>(full_col)];

                if (reduced_row >= 0 && reduced_col >= 0) {
                    // Both unconstrained - direct copy
                    reduced.addEntry(reduced_row, reduced_col);
                } else if (reduced_row >= 0 && reduced_col < 0) {
                    // Row unconstrained, column constrained
                    auto masters_col = master_map[full_col];
                    for (GlobalIndex mc : masters_col) {
                        GlobalIndex mc_reduced = full_to_reduced[static_cast<std::size_t>(mc)];
                        if (mc_reduced >= 0) {
                            reduced.addEntry(reduced_row, mc_reduced);
                        }
                    }
                } else if (reduced_row < 0 && reduced_col >= 0) {
                    // Row constrained, column unconstrained
                    auto masters_row = master_map[full_row];
                    for (GlobalIndex mr : masters_row) {
                        GlobalIndex mr_reduced = full_to_reduced[static_cast<std::size_t>(mr)];
                        if (mr_reduced >= 0) {
                            reduced.addEntry(mr_reduced, reduced_col);
                        }
                    }
                } else {
                    // Both constrained
                    auto masters_row = master_map[full_row];
                    auto masters_col = master_map[full_col];
                    for (GlobalIndex mr : masters_row) {
                        GlobalIndex mr_reduced = full_to_reduced[static_cast<std::size_t>(mr)];
                        if (mr_reduced < 0) continue;

                        for (GlobalIndex mc : masters_col) {
                            GlobalIndex mc_reduced = full_to_reduced[static_cast<std::size_t>(mc)];
                            if (mc_reduced >= 0) {
                                reduced.addEntry(mr_reduced, mc_reduced);
                            }
                        }
                    }
                }
            }
        }
    }

    // Ensure diagonal if requested
    if (options_.ensure_diagonal) {
        reduced.ensureDiagonal();
    }

    reduced.finalize();
    return reduced;
}

std::vector<GlobalIndex> ConstraintSparsityAugmenter::getTransitiveMasters(
    GlobalIndex constrained_dof) const
{
    if (!constraint_query_->isConstrained(constrained_dof)) {
        return {};
    }

    std::vector<GlobalIndex> result;
    std::unordered_set<GlobalIndex> visited;
    std::queue<GlobalIndex> to_process;

    // Start with direct masters
    auto direct_masters = constraint_query_->getMasterDofs(constrained_dof);
    for (GlobalIndex m : direct_masters) {
        to_process.push(m);
    }
    visited.insert(constrained_dof);  // Don't revisit the original DOF

    while (!to_process.empty()) {
        GlobalIndex current = to_process.front();
        to_process.pop();

        if (visited.count(current) > 0) {
            continue;
        }
        visited.insert(current);

        if (constraint_query_->isConstrained(current)) {
            // This master is also constrained - add its masters
            auto masters = constraint_query_->getMasterDofs(current);
            for (GlobalIndex m : masters) {
                if (visited.count(m) == 0) {
                    to_process.push(m);
                }
            }
        } else {
            // This is a true master (unconstrained)
            result.push_back(current);
        }
    }

    // Sort for determinism
    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================================
// Query methods
// ============================================================================

std::vector<GlobalIndex> ConstraintSparsityAugmenter::getReducedMapping(
    GlobalIndex n_total) const
{
    std::vector<GlobalIndex> mapping(static_cast<std::size_t>(n_total));

    auto constrained = constraint_query_->getAllConstrainedDofs();
    std::unordered_set<GlobalIndex> constrained_set(
        constrained.begin(), constrained.end());

    GlobalIndex reduced_idx = 0;
    for (GlobalIndex i = 0; i < n_total; ++i) {
        if (constrained_set.count(i) > 0) {
            mapping[static_cast<std::size_t>(i)] = -1;  // Constrained
        } else {
            mapping[static_cast<std::size_t>(i)] = reduced_idx++;
        }
    }

    return mapping;
}

std::vector<GlobalIndex> ConstraintSparsityAugmenter::getFullMapping(
    GlobalIndex n_total) const
{
    auto constrained = constraint_query_->getAllConstrainedDofs();
    std::unordered_set<GlobalIndex> constrained_set(
        constrained.begin(), constrained.end());

    std::vector<GlobalIndex> mapping;
    mapping.reserve(static_cast<std::size_t>(n_total - static_cast<GlobalIndex>(constrained.size())));

    for (GlobalIndex i = 0; i < n_total; ++i) {
        if (constrained_set.count(i) == 0) {
            mapping.push_back(i);
        }
    }

    return mapping;
}

GlobalIndex ConstraintSparsityAugmenter::numUnconstrainedDofs(
    GlobalIndex n_total) const
{
    if (!constraint_query_) {
        return n_total;
    }
    return n_total - static_cast<GlobalIndex>(constraint_query_->numConstraints());
}

void ConstraintSparsityAugmenter::validateConstraints() const {
    FE_THROW_IF(!constraint_query_, InvalidArgumentException,
                "No constraint query configured");
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
