/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsMatrix.h"

#include "Backends/FSILS/FsilsVector.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "Sparsity/SparsityPattern.h"

#include "Backends/FSILS/liner_solver/commu.h"
#include "Backends/FSILS/liner_solver/fsils_api.hpp"
#include "Backends/FSILS/liner_solver/lhs.h"
#include "Backends/FSILS/liner_solver/spar_mul.h"

#include "Array.h"
#include "Vector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

std::atomic<std::uint64_t> FsilsMatrix::dropped_entry_count_{0};
std::atomic<std::uint64_t> FsilsMatrix::off_owner_write_count_{0};

std::uint64_t FsilsMatrix::droppedEntryCount() noexcept
{
    return dropped_entry_count_.load(std::memory_order_relaxed);
}

void FsilsMatrix::resetDroppedEntryCount() noexcept
{
    dropped_entry_count_.store(0, std::memory_order_relaxed);
}

std::uint64_t FsilsMatrix::offOwnerWriteCount() noexcept
{
    return off_owner_write_count_.load(std::memory_order_relaxed);
}

void FsilsMatrix::resetOffOwnerWriteCount() noexcept
{
    off_owner_write_count_.store(0, std::memory_order_relaxed);
}

namespace {

[[nodiscard]] fe_fsi_linear_solver::FSILS_commuType make_fsils_commu(MPI_Comm comm)
{
    fe_fsi_linear_solver::FSILS_commuType commu{};
    commu.foC = true;
    commu.comm = comm;
    commu.nTasks = 1;
    commu.task = 0;
    commu.master = 0;
    commu.masF = true;
    commu.tF = 0;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        fe_fsi_linear_solver::fsils_commu_create(commu, comm);
    }
    return commu;
}

[[nodiscard]] int as_int(GlobalIndex v, const char* what)
{
    FE_THROW_IF(v < 0, InvalidArgumentException, std::string("FSILS: negative ") + what);
    FE_THROW_IF(v > static_cast<GlobalIndex>(std::numeric_limits<int>::max()),
                InvalidArgumentException,
                std::string("FSILS: ") + what + " exceeds int range");
    return static_cast<int>(v);
}

[[nodiscard]] std::size_t block_entry_index(int dof, int row_comp, int col_comp)
{
    return static_cast<std::size_t>(row_comp * dof + col_comp);
}

[[nodiscard]] int next_power_of_two(int value) noexcept
{
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

[[nodiscard]] bool env_flag_enabled(const char* name) noexcept
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool matrix_diag_trace_enabled() noexcept
{
    return env_flag_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING") ||
           env_flag_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS");
}

[[nodiscard]] GlobalIndex fsils_trace_add_row_dof() noexcept
{
    static const GlobalIndex traced = []() noexcept {
        const char* env = std::getenv("SVMP_FSILS_TRACE_ADD_ROW_DOF");
        if (env == nullptr || *env == '\0') {
            return INVALID_GLOBAL_INDEX;
        }
        char* end = nullptr;
        const auto value = std::strtoll(env, &end, 10);
        if (end == env) {
            return INVALID_GLOBAL_INDEX;
        }
        return static_cast<GlobalIndex>(value);
    }();
    return traced;
}

[[nodiscard]] bool fsils_trace_add_row(GlobalIndex fe_row) noexcept
{
    return fe_row == fsils_trace_add_row_dof();
}

enum class LocalRowOwnership : std::uint8_t {
    Absent,
    Owned,
    NonOwnedLocal
};

[[nodiscard]] LocalRowOwnership classify_fe_matrix_row(const FsilsShared& shared,
                                                       GlobalIndex fe_row,
                                                       GlobalIndex global_rows) noexcept
{
    if (fe_row < 0 || fe_row >= global_rows || shared.dof <= 0) {
        return LocalRowOwnership::Absent;
    }

    GlobalIndex backend_row = fe_row;
    if (const auto perm = shared.dof_permutation; perm && !perm->empty()) {
        if (static_cast<std::size_t>(fe_row) >= perm->forward.size()) {
            return LocalRowOwnership::Absent;
        }
        backend_row = perm->forward[static_cast<std::size_t>(fe_row)];
    }
    if (backend_row < 0 || backend_row >= global_rows) {
        return LocalRowOwnership::Absent;
    }

    const int global_node = static_cast<int>(backend_row / shared.dof);
    const int old = shared.globalNodeToOld(global_node);
    if (old < 0 || old >= shared.lhs.nNo) {
        return LocalRowOwnership::Absent;
    }
    return (old < shared.owned_node_count) ? LocalRowOwnership::Owned
                                           : LocalRowOwnership::NonOwnedLocal;
}

[[nodiscard]] bool is_valid_fe_matrix_col(const FsilsShared& shared,
                                          GlobalIndex fe_col,
                                          GlobalIndex global_cols) noexcept
{
    if (fe_col < 0 || fe_col >= global_cols || shared.dof <= 0) {
        return false;
    }

    GlobalIndex backend_col = fe_col;
    if (const auto perm = shared.dof_permutation; perm && !perm->empty()) {
        if (static_cast<std::size_t>(fe_col) >= perm->forward.size()) {
            return false;
        }
        backend_col = perm->forward[static_cast<std::size_t>(fe_col)];
    }
    return backend_col >= 0 && backend_col < global_cols;
}

[[nodiscard]] bool ghost_rows_look_nodal_interleaved(std::span<const GlobalIndex> ghost_rows, int dof)
{
    if (dof <= 0) {
        return false;
    }
    if (ghost_rows.empty()) {
        return false;
    }

    const std::size_t per_node = static_cast<std::size_t>(dof);
    if (ghost_rows.size() % per_node != 0u) {
        return false;
    }

    for (std::size_t i = 0; i < ghost_rows.size(); i += per_node) {
        const GlobalIndex base = ghost_rows[i];
        if (base < 0 || (base % dof) != 0) {
            return false;
        }
        for (int c = 1; c < dof; ++c) {
            const std::size_t idx = i + static_cast<std::size_t>(c);
            if (idx >= ghost_rows.size() || ghost_rows[idx] != base + c) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] std::shared_ptr<const DofPermutation> normalize_dof_permutation(std::shared_ptr<const DofPermutation> perm,
                                                                              GlobalIndex global_size,
                                                                              std::string_view context,
                                                                              bool allow_partial)
{
    if (!perm || perm->empty()) {
        return perm;
    }

    FE_THROW_IF(perm->forward.size() != static_cast<std::size_t>(global_size) ||
                    perm->inverse.size() != static_cast<std::size_t>(global_size),
                InvalidArgumentException,
                std::string(context) + ": dof permutation size mismatch with global system size");

    if (!allow_partial) {
        std::vector<GlobalIndex> inverse_from_forward(static_cast<std::size_t>(global_size), INVALID_GLOBAL_INDEX);
        for (GlobalIndex fe = 0; fe < global_size; ++fe) {
            const auto fe_idx = static_cast<std::size_t>(fe);
            const GlobalIndex be = perm->forward[fe_idx];
            FE_THROW_IF(be < 0 || be >= global_size,
                        InvalidArgumentException,
                        std::string(context) + ": dof permutation mapped FE DOF to out-of-range backend DOF");

            const auto be_idx = static_cast<std::size_t>(be);
            FE_THROW_IF(inverse_from_forward[be_idx] != INVALID_GLOBAL_INDEX,
                        InvalidArgumentException,
                        std::string(context) + ": dof permutation is not one-to-one");
            inverse_from_forward[be_idx] = fe;
        }

        for (GlobalIndex be = 0; be < global_size; ++be) {
            if (inverse_from_forward[static_cast<std::size_t>(be)] == INVALID_GLOBAL_INDEX) {
                FE_THROW(InvalidArgumentException,
                         std::string(context) + ": dof permutation is not onto");
            }
        }

        if (inverse_from_forward == perm->inverse) {
            return perm;
        }

        auto normalized = std::make_shared<DofPermutation>();
        normalized->forward = perm->forward;
        normalized->inverse = std::move(inverse_from_forward);
        normalized->owner_rank = perm->owner_rank;
        return normalized;
    }

    // Partial permutations are allowed for distributed overlap runs: entries not present on this
    // rank may be left INVALID. Rebuild inverse from forward for mapped entries and normalize it.
    const auto& fwd = perm->forward;
    std::vector<GlobalIndex> inverse_from_forward(static_cast<std::size_t>(global_size), INVALID_GLOBAL_INDEX);
    for (GlobalIndex fe = 0; fe < global_size; ++fe) {
        const GlobalIndex be = fwd[static_cast<std::size_t>(fe)];
        if (be == INVALID_GLOBAL_INDEX) {
            continue;
        }
        FE_THROW_IF(be < 0 || be >= global_size,
                    InvalidArgumentException,
                    std::string(context) + ": dof permutation mapped FE DOF to out-of-range backend DOF");
        auto& slot = inverse_from_forward[static_cast<std::size_t>(be)];
        FE_THROW_IF(slot != INVALID_GLOBAL_INDEX && slot != fe,
                    InvalidArgumentException,
                    std::string(context) + ": dof permutation is not one-to-one on mapped entries");
        slot = fe;
    }

    if (inverse_from_forward == perm->inverse) {
        return perm;
    }

    auto normalized = std::make_shared<DofPermutation>();
    normalized->forward = perm->forward;
    normalized->inverse = std::move(inverse_from_forward);
    normalized->owner_rank = perm->owner_rank;
    return normalized;
}

void sort_row_columns_and_values(FsilsShared& shared, std::vector<Real>& values)
{
    const int dof = shared.dof;
    FE_THROW_IF(dof <= 0, FEException, "FsilsMatrix: invalid dof");

    auto& lhs = shared.lhs;
    const int nNo = lhs.nNo;
    const int nnz = lhs.nnz;
    FE_THROW_IF(nNo < 0 || nnz < 0, FEException, "FsilsMatrix: invalid lhs sizes");

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    FE_THROW_IF(values.size() != static_cast<std::size_t>(nnz) * block_size,
                FEException, "FsilsMatrix: values size mismatch");

    // Sort within each internal row for efficient binary search insertion.
    auto* cols = lhs.colPtr.data();
    Real* vals = values.data();

    std::vector<std::pair<fe_fsi_linear_solver::fsils_int, int>> key_idx;
    std::vector<fe_fsi_linear_solver::fsils_int> cols_sorted;
    std::vector<Real> vals_sorted;

    for (int row = 0; row < nNo; ++row) {
        const int start = lhs.rowPtr(0, row);
        const int end = lhs.rowPtr(1, row);
        FE_THROW_IF(start < 0 || end < start || end >= nnz, FEException,
                    "FsilsMatrix: invalid FSILS rowPtr range");

        const int len = end - start + 1;
        if (len <= 1) {
            lhs.diagPtr(row) = start;
            continue;
        }

        key_idx.resize(static_cast<std::size_t>(len));
        for (int k = 0; k < len; ++k) {
            const int idx = start + k;
            key_idx[static_cast<std::size_t>(k)] = {cols[idx], idx};
        }
        std::sort(key_idx.begin(), key_idx.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        cols_sorted.resize(static_cast<std::size_t>(len));
        vals_sorted.resize(static_cast<std::size_t>(len) * block_size);
        for (int k = 0; k < len; ++k) {
            const int old_idx = key_idx[static_cast<std::size_t>(k)].second;
            cols_sorted[static_cast<std::size_t>(k)] = cols[old_idx];
            const std::size_t src = static_cast<std::size_t>(old_idx) * block_size;
            const std::size_t dst = static_cast<std::size_t>(k) * block_size;
            std::copy(vals + src, vals + src + block_size, vals_sorted.data() + dst);
        }

        for (int k = 0; k < len; ++k) {
            const int idx = start + k;
            cols[idx] = cols_sorted[static_cast<std::size_t>(k)];
            const std::size_t dst = static_cast<std::size_t>(idx) * block_size;
            const std::size_t src = static_cast<std::size_t>(k) * block_size;
            std::copy(vals_sorted.data() + src, vals_sorted.data() + src + block_size, vals + dst);
        }

        // Recompute diagPtr after sorting.
        const auto* begin = cols + start;
        const auto* finish = cols + end + 1;
        const auto it = std::lower_bound(begin, finish, row);
        FE_THROW_IF(it == finish || *it != row, FEException,
                    "FsilsMatrix: missing diagonal block in FSILS pattern");
        lhs.diagPtr(row) = static_cast<int>(it - cols);
    }
}

void maybe_log_matrix_locality(const FsilsShared& shared)
{
    if (!env_flag_enabled("SVMP_FSILS_MATRIX_LOCALITY_PROFILE")) {
        return;
    }

    const auto& lhs = shared.lhs;
    const int nNo = lhs.nNo;
    if (nNo <= 0 || lhs.nnz <= 0) {
        return;
    }

    long long row_nnz_sum = 0;
    long long abs_delta_sum = 0;
    long long diag_position_sum = 0;
    int max_row_nnz = 0;
    int max_abs_delta = 0;
    long long near_cacheline_edges = 0;

    for (int row = 0; row < nNo; ++row) {
        const int start = lhs.rowPtr(0, row);
        const int end = lhs.rowPtr(1, row);
        if (start < 0 || end < start) {
            continue;
        }
        const int row_nnz = end - start + 1;
        row_nnz_sum += row_nnz;
        max_row_nnz = std::max(max_row_nnz, row_nnz);

        const int diag = lhs.diagPtr(row);
        if (diag >= start && diag <= end) {
            diag_position_sum += diag - start;
        }

        for (int idx = start; idx <= end; ++idx) {
            const int col = lhs.colPtr(idx);
            const int delta = std::abs(col - row);
            abs_delta_sum += delta;
            max_abs_delta = std::max(max_abs_delta, delta);
            if (delta <= 8) {
                ++near_cacheline_edges;
            }
        }
    }

    const double nnz = static_cast<double>(lhs.nnz > 0 ? lhs.nnz : 1);
    const double rows = static_cast<double>(std::max(1, nNo));
    std::fprintf(stderr,
                 "[FSILS_MATRIX_LOCALITY] rank=%d nNo=%d mynNo=%d nnz=%d avg_row_nnz=%.3f max_row_nnz=%d "
                 "avg_abs_col_delta=%.3f max_abs_col_delta=%d near8=%.2f%% avg_diag_pos=%.3f dof=%d\n",
                 lhs.commu.task,
                 nNo,
                 lhs.mynNo,
                 lhs.nnz,
                 static_cast<double>(row_nnz_sum) / rows,
                 max_row_nnz,
                 static_cast<double>(abs_delta_sum) / nnz,
                 max_abs_delta,
                 100.0 * static_cast<double>(near_cacheline_edges) / nnz,
                 static_cast<double>(diag_position_sum) / rows,
                 shared.dof);
}

void restore_owned_row_operator_ghost_identity(FsilsShared& shared, std::vector<Real>& values)
{
    auto& lhs = shared.lhs;
    if (!lhs.owned_row_operator || lhs.commu.nTasks <= 1 || lhs.mynNo >= lhs.nNo) {
        return;
    }

    const int dof = shared.dof;
    FE_THROW_IF(dof <= 0, FEException, "FsilsMatrix: invalid dof");
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    FE_THROW_IF(values.size() != static_cast<std::size_t>(lhs.nnz) * block_size,
                FEException,
                "FsilsMatrix: values size mismatch while restoring ghost identity");

    for (int row = lhs.mynNo; row < lhs.nNo; ++row) {
        const int diag = lhs.diagPtr(row);
        FE_THROW_IF(diag < 0 || diag >= lhs.nnz, FEException,
                    "FsilsMatrix: invalid ghost diagonal pointer");

        Real* block = values.data() + static_cast<std::size_t>(diag) * block_size;
        std::fill(block, block + block_size, Real(0));
        for (int c = 0; c < dof; ++c) {
            block[block_entry_index(dof, c, c)] = Real(1);
        }
    }
}

[[nodiscard]] int owner_rank_for_backend_node(const FsilsShared& shared, int backend_node) noexcept
{
    if (backend_node < 0 || shared.dof <= 0 || !shared.dof_permutation ||
        shared.dof_permutation->owner_rank.empty()) {
        return -1;
    }
    const GlobalIndex first_backend_dof =
        static_cast<GlobalIndex>(backend_node) * static_cast<GlobalIndex>(shared.dof);
    if (first_backend_dof < 0) {
        return -1;
    }
    for (int c = 0; c < shared.dof; ++c) {
        const GlobalIndex backend_dof = first_backend_dof + static_cast<GlobalIndex>(c);
        if (backend_dof < 0 ||
            static_cast<std::size_t>(backend_dof) >= shared.dof_permutation->owner_rank.size()) {
            continue;
        }
        const int owner = shared.dof_permutation->owner_rank[static_cast<std::size_t>(backend_dof)];
        if (owner >= 0) {
            return owner;
        }
    }
    return -1;
}

void clear_owned_row_halo_plan(FsilsShared& shared)
{
    shared.lhs.owned_halo_neighbor_ranks.clear();
    shared.lhs.owned_halo_send_nodes.clear();
    shared.lhs.owned_halo_recv_nodes.clear();
    shared.lhs.owned_halo_send_buffer.clear();
    shared.lhs.owned_halo_recv_buffer.clear();
}

void build_owned_row_halo_plan(FsilsShared& shared)
{
    clear_owned_row_halo_plan(shared);

#if FE_HAS_MPI
    auto& lhs = shared.lhs;
    if (!lhs.owned_row_operator || lhs.commu.nTasks <= 1) {
        return;
    }
    const int rank = lhs.commu.task;
    const int size = lhs.commu.nTasks;
    std::vector<std::vector<int>> ghost_requests_by_owner(static_cast<std::size_t>(size));
    int local_plan_ok = 1;
    const bool have_owner_metadata =
        shared.dof_permutation != nullptr && !shared.dof_permutation->owner_rank.empty();

    if (have_owner_metadata) {
        for (int old = shared.owned_node_count; old < lhs.nNo; ++old) {
            const int node = shared.oldToGlobalNode(old);
            const int owner = owner_rank_for_backend_node(shared, node);
            if (owner < 0 || owner >= size || owner == rank) {
                local_plan_ok = 0;
                continue;
            }
            ghost_requests_by_owner[static_cast<std::size_t>(owner)].push_back(node);
        }
    } else {
        local_plan_ok = 0;
    }

    int global_plan_ok = 0;
    MPI_Allreduce(&local_plan_ok,
                  &global_plan_ok,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  lhs.commu.comm);
    if (!have_owner_metadata || global_plan_ok == 0) {
        // Some distributed setup paths do not carry explicit owner metadata for
        // every backend node. Derive ownership by exchanging owned nodes once
        // during setup; solve-time communication must use the owned halo plan.
        std::vector<int> local_owned_nodes;
        local_owned_nodes.reserve(static_cast<std::size_t>(std::max(shared.owned_node_count, 0)));
        for (int old = 0; old < shared.owned_node_count; ++old) {
            const int node = shared.oldToGlobalNode(old);
            if (node >= 0) {
                local_owned_nodes.push_back(node);
            }
        }

        std::vector<int> owned_counts(static_cast<std::size_t>(size), 0);
        const int local_owned_count = static_cast<int>(local_owned_nodes.size());
        MPI_Allgather(&local_owned_count,
                      1,
                      MPI_INT,
                      owned_counts.data(),
                      1,
                      MPI_INT,
                      lhs.commu.comm);

        std::vector<int> owned_displs(static_cast<std::size_t>(size + 1), 0);
        for (int r = 0; r < size; ++r) {
            owned_displs[static_cast<std::size_t>(r + 1)] =
                owned_displs[static_cast<std::size_t>(r)] + owned_counts[static_cast<std::size_t>(r)];
        }
        std::vector<int> all_owned_nodes(static_cast<std::size_t>(owned_displs.back()), 0);
        MPI_Allgatherv(local_owned_nodes.data(),
                       local_owned_count,
                       MPI_INT,
                       all_owned_nodes.data(),
                       owned_counts.data(),
                       owned_displs.data(),
                       MPI_INT,
                       lhs.commu.comm);

        std::unordered_map<int, int> owner_by_node;
        owner_by_node.reserve(all_owned_nodes.size());
        for (int r = 0; r < size; ++r) {
            const int begin = owned_displs[static_cast<std::size_t>(r)];
            const int end = owned_displs[static_cast<std::size_t>(r + 1)];
            for (int i = begin; i < end; ++i) {
                owner_by_node.emplace(all_owned_nodes[static_cast<std::size_t>(i)], r);
            }
        }

        for (auto& requests : ghost_requests_by_owner) {
            requests.clear();
        }
        local_plan_ok = 1;
        for (int old = shared.owned_node_count; old < lhs.nNo; ++old) {
            const int node = shared.oldToGlobalNode(old);
            const auto owner_it = owner_by_node.find(node);
            if (owner_it == owner_by_node.end() || owner_it->second == rank) {
                local_plan_ok = 0;
                continue;
            }
            ghost_requests_by_owner[static_cast<std::size_t>(owner_it->second)].push_back(node);
        }
        MPI_Allreduce(&local_plan_ok,
                      &global_plan_ok,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      lhs.commu.comm);
        if (global_plan_ok == 0) {
            if (std::getenv("SVMP_FSILS_HALO_TRACE") != nullptr) {
                std::fprintf(stderr,
                             "[FSILS_HALO] rank=%d explicit halo plan disabled during owner validation local_ok=%d owned=%lld local=%lld\n",
                             rank,
                             local_plan_ok,
                             static_cast<long long>(lhs.mynNo),
                             static_cast<long long>(lhs.nNo));
            }
            clear_owned_row_halo_plan(shared);
            FE_THROW(InvalidArgumentException,
                     "FsilsMatrix: failed to build explicit owned-row halo plan from owned-node exchange");
        }
    }

    std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
    std::vector<int> recv_counts(static_cast<std::size_t>(size), 0);
    for (int r = 0; r < size; ++r) {
        send_counts[static_cast<std::size_t>(r)] =
            static_cast<int>(ghost_requests_by_owner[static_cast<std::size_t>(r)].size());
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 lhs.commu.comm);

    std::vector<int> send_displs(static_cast<std::size_t>(size + 1), 0);
    std::vector<int> recv_displs(static_cast<std::size_t>(size + 1), 0);
    for (int r = 0; r < size; ++r) {
        send_displs[static_cast<std::size_t>(r + 1)] =
            send_displs[static_cast<std::size_t>(r)] + send_counts[static_cast<std::size_t>(r)];
        recv_displs[static_cast<std::size_t>(r + 1)] =
            recv_displs[static_cast<std::size_t>(r)] + recv_counts[static_cast<std::size_t>(r)];
    }

    std::vector<int> send_nodes(static_cast<std::size_t>(send_displs.back()), 0);
    for (int r = 0; r < size; ++r) {
        auto& nodes = ghost_requests_by_owner[static_cast<std::size_t>(r)];
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        send_counts[static_cast<std::size_t>(r)] = static_cast<int>(nodes.size());
    }

    // Counts may shrink after unique(); exchange the final request counts.
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 lhs.commu.comm);
    send_displs.assign(static_cast<std::size_t>(size + 1), 0);
    recv_displs.assign(static_cast<std::size_t>(size + 1), 0);
    for (int r = 0; r < size; ++r) {
        send_displs[static_cast<std::size_t>(r + 1)] =
            send_displs[static_cast<std::size_t>(r)] + send_counts[static_cast<std::size_t>(r)];
        recv_displs[static_cast<std::size_t>(r + 1)] =
            recv_displs[static_cast<std::size_t>(r)] + recv_counts[static_cast<std::size_t>(r)];
    }
    send_nodes.assign(static_cast<std::size_t>(send_displs.back()), 0);
    for (int r = 0; r < size; ++r) {
        const auto& nodes = ghost_requests_by_owner[static_cast<std::size_t>(r)];
        std::copy(nodes.begin(),
                  nodes.end(),
                  send_nodes.begin() + send_displs[static_cast<std::size_t>(r)]);
    }
    std::vector<int> recv_nodes(static_cast<std::size_t>(recv_displs.back()), 0);

    MPI_Alltoallv(send_nodes.data(),
                  send_counts.data(),
                  send_displs.data(),
                  MPI_INT,
                  recv_nodes.data(),
                  recv_counts.data(),
                  recv_displs.data(),
                  MPI_INT,
                  lhs.commu.comm);

    local_plan_ok = 1;
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            continue;
        }
        const int n_send = recv_counts[static_cast<std::size_t>(r)];
        const int n_recv = send_counts[static_cast<std::size_t>(r)];
        if (n_send == 0 && n_recv == 0) {
            continue;
        }

        lhs.owned_halo_neighbor_ranks.push_back(r);
        auto& send_internal = lhs.owned_halo_send_nodes.emplace_back();
        auto& recv_internal = lhs.owned_halo_recv_nodes.emplace_back();
        send_internal.reserve(static_cast<std::size_t>(std::max(n_send, 0)));
        recv_internal.reserve(static_cast<std::size_t>(std::max(n_recv, 0)));

        const int recv_begin = recv_displs[static_cast<std::size_t>(r)];
        for (int i = 0; i < n_send; ++i) {
            const int old = shared.globalNodeToOld(recv_nodes[static_cast<std::size_t>(recv_begin + i)]);
            if (old < 0 || old >= shared.owned_node_count) {
                local_plan_ok = 0;
                continue;
            }
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                local_plan_ok = 0;
                continue;
            }
            send_internal.push_back(static_cast<fe_fsi_linear_solver::fsils_int>(internal));
        }

        const auto& requested_nodes = ghost_requests_by_owner[static_cast<std::size_t>(r)];
        for (const int node : requested_nodes) {
            const int old = shared.globalNodeToOld(node);
            if (old < shared.owned_node_count || old >= lhs.nNo) {
                local_plan_ok = 0;
                continue;
            }
            const int internal = lhs.map(old);
            if (internal < lhs.mynNo || internal >= lhs.nNo) {
                local_plan_ok = 0;
                continue;
            }
            recv_internal.push_back(static_cast<fe_fsi_linear_solver::fsils_int>(internal));
        }
    }

    MPI_Allreduce(&local_plan_ok,
                  &global_plan_ok,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  lhs.commu.comm);
    if (global_plan_ok == 0) {
        if (std::getenv("SVMP_FSILS_HALO_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[FSILS_HALO] rank=%d explicit halo plan disabled during node mapping local_ok=%d owned=%lld local=%lld\n",
                         rank,
                         local_plan_ok,
                         static_cast<long long>(lhs.mynNo),
                         static_cast<long long>(lhs.nNo));
        }
        clear_owned_row_halo_plan(shared);
        FE_THROW(InvalidArgumentException,
                 "FsilsMatrix: failed to map explicit owned-row halo plan to local FSILS nodes");
    }

    if (std::getenv("SVMP_FSILS_HALO_TRACE") != nullptr) {
        std::size_t send_nodes_total = 0;
        std::size_t recv_nodes_total = 0;
        for (const auto& nodes : lhs.owned_halo_send_nodes) {
            send_nodes_total += nodes.size();
        }
        for (const auto& nodes : lhs.owned_halo_recv_nodes) {
            recv_nodes_total += nodes.size();
        }
        std::fprintf(stderr,
                     "[FSILS_HALO] rank=%d neighbors=%zu send_nodes=%zu recv_nodes=%zu owned=%lld local=%lld\n",
                     rank,
                     lhs.owned_halo_neighbor_ranks.size(),
                     send_nodes_total,
                     recv_nodes_total,
                     static_cast<long long>(lhs.mynNo),
                     static_cast<long long>(lhs.nNo));
    }
#else
    (void)shared;
#endif
}

void validate_owned_row_halo_plan(const FsilsShared& shared)
{
#if FE_HAS_MPI
    const auto& lhs = shared.lhs;
    if (!lhs.owned_row_operator || lhs.commu.nTasks <= 1) {
        return;
    }

    FE_THROW_IF(lhs.owned_halo_send_nodes.size() != lhs.owned_halo_neighbor_ranks.size() ||
                    lhs.owned_halo_recv_nodes.size() != lhs.owned_halo_neighbor_ranks.size(),
                InvalidArgumentException,
                "FsilsMatrix: distributed owned-row FSILS matrix has an invalid explicit owned halo plan");

    if (lhs.nNo > lhs.mynNo) {
        FE_THROW_IF(lhs.owned_halo_neighbor_ranks.empty(),
                    InvalidArgumentException,
                    "FsilsMatrix: distributed owned-row FSILS matrix with ghost nodes is missing an explicit owned halo plan");
    }
#else
    (void)shared;
#endif
}

/// Build per-row block-base lookup tables.
/// Must be called AFTER sort_row_columns_and_values() so that the stored bases
/// correspond to the final positions in colPtr/values_.
void buildBlockLookupTables(FsilsShared& shared)
{
    if (shared.lhs.nNo <= 0) return;

    const int nNo_int = shared.lhs.nNo;
    shared.block_lookup_row_ptr_.assign(static_cast<std::size_t>(nNo_int + 1), 0);
    shared.block_lookup_row_mask_.assign(static_cast<std::size_t>(nNo_int), -1);
    const std::size_t block_size =
        static_cast<std::size_t>(shared.dof) * static_cast<std::size_t>(shared.dof);

    for (int row = 0; row < nNo_int; ++row) {
        const int start = shared.lhs.rowPtr(0, row);
        const int end = shared.lhs.rowPtr(1, row);
        if (start < 0 || end < start) {
            shared.block_lookup_row_ptr_[static_cast<std::size_t>(row + 1)] =
                shared.block_lookup_row_ptr_[static_cast<std::size_t>(row)];
            continue;
        }
        const int n_cols = end - start + 1;
        const int table_size = next_power_of_two(std::max(2, n_cols * 2));
        shared.block_lookup_row_mask_[static_cast<std::size_t>(row)] = table_size - 1;
        shared.block_lookup_row_ptr_[static_cast<std::size_t>(row + 1)] =
            shared.block_lookup_row_ptr_[static_cast<std::size_t>(row)] + table_size;
    }

    const int table_total = shared.block_lookup_row_ptr_[static_cast<std::size_t>(nNo_int)];
    shared.block_lookup_cols_.assign(static_cast<std::size_t>(table_total), -1);
    shared.block_lookup_base_.assign(static_cast<std::size_t>(table_total), INVALID_GLOBAL_INDEX);

    for (int row = 0; row < nNo_int; ++row) {
        const int start = shared.lhs.rowPtr(0, row);
        const int end = shared.lhs.rowPtr(1, row);
        if (start < 0 || end < start) continue;

        const int row_begin = shared.block_lookup_row_ptr_[static_cast<std::size_t>(row)];
        const int mask = shared.block_lookup_row_mask_[static_cast<std::size_t>(row)];
        FE_THROW_IF(mask < 0, FEException, "FsilsMatrix: invalid block lookup row mask");

        for (int idx = start; idx <= end; ++idx) {
            const int col = shared.lhs.colPtr[idx];
            int slot = col & mask;
            while (shared.block_lookup_cols_[static_cast<std::size_t>(row_begin + slot)] != -1) {
                slot = (slot + 1) & mask;
            }
            shared.block_lookup_cols_[static_cast<std::size_t>(row_begin + slot)] = col;
            shared.block_lookup_base_[static_cast<std::size_t>(row_begin + slot)] =
                static_cast<GlobalIndex>(static_cast<std::size_t>(idx) * block_size);
        }
    }
}

void validateBlockLookupTables(const FsilsShared& shared)
{
    const auto& lhs = shared.lhs;
    const std::size_t block_size =
        static_cast<std::size_t>(shared.dof) * static_cast<std::size_t>(shared.dof);

    for (int old = 0; old < lhs.nNo; ++old) {
        const int global_node = shared.oldToGlobalNode(old);
        FE_THROW_IF(global_node < 0,
                    FEException,
                    "FsilsMatrix: validation found invalid old-to-global node mapping");
        const int expected_internal = lhs.map(old);
        const int actual_internal = shared.globalNodeToInternal(global_node);
        FE_THROW_IF(actual_internal != expected_internal,
                    FEException,
                    "FsilsMatrix: global-to-internal validation failed for global_node=" +
                        std::to_string(global_node) + " expected=" +
                        std::to_string(expected_internal) + " actual=" +
                        std::to_string(actual_internal));
    }

    for (int row = 0; row < lhs.nNo; ++row) {
        const int start = lhs.rowPtr(0, row);
        const int end = lhs.rowPtr(1, row);
        if (start < 0 || end < start) {
            continue;
        }

        const auto* begin = lhs.colPtr.data() + start;
        const auto* finish = lhs.colPtr.data() + end + 1;
        for (int idx = start; idx <= end; ++idx) {
            const int col = lhs.colPtr[idx];
            const auto* first = std::lower_bound(begin, finish, col);
            FE_THROW_IF(first == finish || *first != col,
                        FEException,
                        "FsilsMatrix: validation could not locate row column");
            const GlobalIndex expected = static_cast<GlobalIndex>(
                static_cast<std::size_t>(first - lhs.colPtr.data()) * block_size);
            const GlobalIndex actual = shared.blockBase(row, col);
            FE_THROW_IF(actual != expected,
                        FEException,
                        "FsilsMatrix: block lookup table validation failed for row=" +
                            std::to_string(row) + " col=" + std::to_string(col) +
                            " expected=" + std::to_string(expected) +
                            " actual=" + std::to_string(actual) +
                            " row_begin=" +
                            std::to_string(shared.block_lookup_row_ptr_[static_cast<std::size_t>(row)]) +
                            " row_end=" +
                            std::to_string(shared.block_lookup_row_ptr_[static_cast<std::size_t>(row + 1)]) +
                            " mask=" +
                            std::to_string(shared.block_lookup_row_mask_[static_cast<std::size_t>(row)]));
        }
    }
}

void build_old_of_internal(FsilsShared& shared)
{
    auto& lhs = shared.lhs;
    const int nNo = lhs.nNo;
    shared.old_of_internal.assign(static_cast<std::size_t>(nNo), -1);
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        FE_THROW_IF(internal < 0 || internal >= nNo, FEException, "FsilsMatrix: invalid lhs.map permutation");
        shared.old_of_internal[static_cast<std::size_t>(internal)] = old;
    }
    for (int internal = 0; internal < nNo; ++internal) {
        FE_THROW_IF(shared.old_of_internal[static_cast<std::size_t>(internal)] < 0,
                    FEException, "FsilsMatrix: invalid inverse permutation");
    }
}

void resolveFsilsMatrixEntrySlotsUncached(const FsilsMatrix& matrix,
                                          std::span<const GlobalIndex> row_dofs,
                                          std::span<const GlobalIndex> col_dofs,
                                          std::span<GlobalIndex> resolved)
{
    const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
    const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());
    FE_THROW_IF(resolved.size() != static_cast<std::size_t>(n_rows * n_cols),
                InvalidArgumentException,
                "resolveFsilsMatrixEntrySlotsUncached: resolved size mismatch");

    std::fill(resolved.begin(), resolved.end(), INVALID_GLOBAL_INDEX);

    const auto shared = matrix.shared();
    if (!shared) {
        return;
    }

    const int dof = shared->dof;
    const auto perm = shared->dof_permutation;
    const bool have_perm = perm && !perm->empty();
    const auto* perm_data = have_perm ? perm->forward.data() : nullptr;
    const auto perm_size = have_perm ? perm->forward.size() : std::size_t{0};
    const GlobalIndex num_rows_global = matrix.numRows();
    const GlobalIndex num_cols_global = matrix.numCols();

    struct DofInfo {
        int internal_node;
        int component;
    };

    thread_local std::vector<DofInfo> row_info;
    thread_local std::vector<DofInfo> col_info;
    row_info.resize(static_cast<std::size_t>(n_rows));
    col_info.resize(static_cast<std::size_t>(n_cols));

    int cached_global_node = -1;
    int cached_internal_node = -1;
    auto resolve_dof = [&](GlobalIndex fe_dof, GlobalIndex limit) -> DofInfo {
        if (fe_dof < 0 || fe_dof >= limit) {
            return {-1, -1};
        }
        GlobalIndex fs_dof = fe_dof;
        if (have_perm) {
            if (static_cast<std::size_t>(fe_dof) >= perm_size) {
                return {-1, -1};
            }
            fs_dof = perm_data[static_cast<std::size_t>(fe_dof)];
        }
        if (fs_dof < 0 || fs_dof >= limit) {
            return {-1, -1};
        }

        const int global_node = static_cast<int>(fs_dof / dof);
        const int comp = static_cast<int>(fs_dof % dof);
        if (global_node == cached_global_node) {
            return {cached_internal_node, comp};
        }

        const int internal = shared->globalNodeToInternal(global_node);
        if (internal < 0) {
            return {-1, -1};
        }

        cached_global_node = global_node;
        cached_internal_node = internal;
        return {internal, comp};
    };

    for (GlobalIndex i = 0; i < n_rows; ++i) {
        row_info[static_cast<std::size_t>(i)] =
            resolve_dof(row_dofs[static_cast<std::size_t>(i)], num_rows_global);
    }
    for (GlobalIndex j = 0; j < n_cols; ++j) {
        col_info[static_cast<std::size_t>(j)] =
            resolve_dof(col_dofs[static_cast<std::size_t>(j)], num_cols_global);
    }

    const std::size_t block_sz = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const auto values_size = static_cast<std::size_t>(matrix.fsilsNnz()) * block_sz;

    for (GlobalIndex i0 = 0; i0 < n_rows; ) {
        const auto& ri0 = row_info[static_cast<std::size_t>(i0)];
        if (ri0.internal_node < 0) {
            ++i0;
            continue;
        }

        GlobalIndex i1 = i0 + 1;
        while (i1 < n_rows &&
               row_info[static_cast<std::size_t>(i1)].internal_node == ri0.internal_node) {
            ++i1;
        }

        for (GlobalIndex j0 = 0; j0 < n_cols; ) {
            const auto& ci0 = col_info[static_cast<std::size_t>(j0)];
            if (ci0.internal_node < 0) {
                ++j0;
                continue;
            }

            GlobalIndex j1 = j0 + 1;
            while (j1 < n_cols &&
                   col_info[static_cast<std::size_t>(j1)].internal_node == ci0.internal_node) {
                ++j1;
            }

            const GlobalIndex base = shared->blockBase(ri0.internal_node, ci0.internal_node);
            if (base != INVALID_GLOBAL_INDEX &&
                static_cast<std::size_t>(base) + block_sz <= values_size) {
                for (GlobalIndex di = i0; di < i1; ++di) {
                    const int r = row_info[static_cast<std::size_t>(di)].component;
                    for (GlobalIndex dj = j0; dj < j1; ++dj) {
                        const int c = col_info[static_cast<std::size_t>(dj)].component;
                        const auto local_idx = static_cast<std::size_t>(di * n_cols + dj);
                        resolved[local_idx] = static_cast<GlobalIndex>(
                            static_cast<std::size_t>(base) + block_entry_index(dof, r, c));
                    }
                }
            }

            j0 = j1;
        }

        i0 = i1;
    }
}

class FsilsMatrixView final : public assembly::GlobalSystemView {
public:
    explicit FsilsMatrixView(FsilsMatrix& matrix) : matrix_(&matrix) {}

    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

        if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
            FE_THROW(InvalidArgumentException, "FsilsMatrixView::addMatrixEntries: local_matrix size mismatch");
        }

        const auto shared = matrix_->shared();
        if (!shared) {
            for (GlobalIndex i = 0; i < n_rows; ++i) {
                const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
                if (row < 0 || row >= matrix_->numRows()) continue;
                for (GlobalIndex j = 0; j < n_cols; ++j) {
                    const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                    if (col < 0 || col >= matrix_->numCols()) continue;
                    const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                    ++matrix_requested_;
                    ++matrix_valid_;
                    accumulateMatrixValueStats(local_matrix[local_idx]);
                    matrix_->addValue(row, col, local_matrix[local_idx], mode);
                }
            }
            return;
        }

        thread_local std::vector<GlobalIndex> resolved;
        resolved.resize(local_matrix.size());
        matrix_->resolveMatrixEntrySlotsCached(row_dofs, col_dofs, resolved);

        matrix_requested_ += resolved.size();
        const std::size_t values_size =
            static_cast<std::size_t>(matrix_->fsilsNnz()) *
            static_cast<std::size_t>(matrix_->fsilsDof()) *
            static_cast<std::size_t>(matrix_->fsilsDof());
        for (std::size_t idx = 0; idx < resolved.size(); ++idx) {
            const auto slot = resolved[idx];
            if (slot >= 0 && static_cast<std::size_t>(slot) < values_size) {
                ++matrix_valid_;
                accumulateMatrixValueStats(local_matrix[idx]);
            }
        }
        matrix_->addResolvedMatrixEntries(row_dofs, col_dofs, resolved, local_matrix, mode);
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        ++matrix_requested_;
        if (row >= 0 && row < matrix_->numRows() && col >= 0 && col < matrix_->numCols()) {
            ++matrix_valid_;
            accumulateMatrixValueStats(value);
        }
        matrix_->addValue(row, col, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "FsilsMatrixView::setDiagonal: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            setDiagonal(dofs[i], values[i]);
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        addMatrixEntry(dof, dof, value, assembly::AddMode::Insert);
    }

	    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
	    {
	        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");

	        const auto shared = matrix_->shared();
	        if (!shared) return;

        auto& lhs = *static_cast<fe_fsi_linear_solver::FSILS_lhsType*>(matrix_->fsilsLhsPtr());
        const int dof = matrix_->fsilsDof();
        const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
	        Real* values = matrix_->fsilsValuesPtr();

		        for (const GlobalIndex row_dof : rows) {
		            if (row_dof < 0 || row_dof >= matrix_->numRows()) continue;

		            GlobalIndex row_fs = row_dof;
		            if (const auto perm = shared->dof_permutation; perm && !perm->empty()) {
		                if (static_cast<std::size_t>(row_dof) >= perm->forward.size()) {
		                    continue;
		                }
		                row_fs = perm->forward[static_cast<std::size_t>(row_dof)];
		                if (row_fs == INVALID_GLOBAL_INDEX) {
		                    // Partial permutations are permitted for distributed overlap runs; unmapped DOFs are not
		                    // locally present on this rank and must be skipped.
		                    continue;
		                }
		            }
		            if (row_fs < 0 || row_fs >= matrix_->numRows()) {
		                continue;
		            }

            const int global_node = static_cast<int>(row_fs / dof);
            const int row_comp = static_cast<int>(row_fs % dof);

            const int internal = shared->globalNodeToInternal(global_node);
            const int old = shared->globalNodeToOld(global_node);
            if (internal < 0 || old < 0) continue;

            const int start = lhs.rowPtr(0, internal);
            const int end = lhs.rowPtr(1, internal);

            for (int j = start; j <= end; ++j) {
                const std::size_t base = static_cast<std::size_t>(j) * block_size;
                for (int c = 0; c < dof; ++c) {
                    values[base + block_entry_index(dof, row_comp, c)] = 0.0;
                }
            }

            // Distributed constrained rows are already represented on ghost ranks.
            // Zero the ghost copies too, but only restore the identity diagonal on
            // the owning row so overlap-summed matvecs do not overcount it.
            const bool row_is_owned = old < shared->owned_node_count;
            if (set_diagonal && row_is_owned && row_dof < matrix_->numCols()) {
                matrix_->addValue(row_dof, row_dof, 1.0, assembly::AddMode::Insert);
            }
        }
    }

    // Vector operations (no-op)
    void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void zeroVectorEntries(std::span<const GlobalIndex>) override {}

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override {
        phase_ = assembly::AssemblyPhase::Finalized;
        if (matrix_ != nullptr && matrix_diag_trace_enabled()) {
            if (const auto shared = matrix_->shared()) {
                static int diag_trace_budget = 0;
                static bool diag_trace_init = false;
                if (!diag_trace_init) {
                    diag_trace_init = true;
                    diag_trace_budget = 8;
                }
                const bool trace_all_ranks =
                    env_flag_enabled("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS");
                const bool trace_this_rank = trace_all_ranks || shared->lhs.commu.task == 0;
                if (trace_this_rank && diag_trace_budget > 0) {
                    --diag_trace_budget;
                    const auto* perm = shared->dof_permutation.get();
                    const bool has_perm = (perm != nullptr && !perm->forward.empty());
                    constexpr GlobalIndex kTraceDofs = 12;
                    std::ostringstream oss;
                    oss << "[FSILS_MATRIX_DIAG] rank=" << shared->lhs.commu.task;
                    for (GlobalIndex fe_dof = 0; fe_dof < std::min(matrix_->numRows(), kTraceDofs); ++fe_dof) {
                        GlobalIndex fs_dof = fe_dof;
                        if (has_perm) {
                            fs_dof = perm->forward[static_cast<std::size_t>(fe_dof)];
                        }
                        int global_node = -1;
                        int comp = -1;
                        if (fs_dof >= 0 && fs_dof < matrix_->numRows()) {
                            global_node = static_cast<int>(fs_dof / shared->dof);
                            comp = static_cast<int>(fs_dof % shared->dof);
                        }
                        oss << " fe(" << fe_dof << ")="
                            << matrix_->getEntry(fe_dof, fe_dof)
                            << "{fs=" << fs_dof
                            << ",gn=" << global_node
                            << ",c=" << comp
                            << "}";
                    }
                    std::fprintf(stderr, "%s\n", oss.str().c_str());
                }
            }
        }
        if (std::getenv("SVMP_FSILS_MATRIX_VIEW_TRACE") != nullptr && matrix_requested_ > 0) {
            int rank = 0;
            if (matrix_ != nullptr) {
                if (const auto shared = matrix_->shared()) {
                    rank = shared->lhs.commu.task;
                }
            }
            std::fprintf(stderr,
                         "[FsilsMatrixView] rank=%d requested=%zu valid=%zu invalid=%zu nonzero=%zu value_l1=%.17g value_l2=%.17g value_max_abs=%.17g\n",
                         rank,
                         matrix_requested_,
                         matrix_valid_,
                         matrix_requested_ - matrix_valid_,
                         matrix_nonzero_,
                         static_cast<double>(matrix_value_l1_),
                         static_cast<double>(std::sqrt(matrix_value_l2_sq_)),
                         static_cast<double>(matrix_value_max_abs_));
        }
    }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return false; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return matrix_ ? matrix_->numRows() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return matrix_ ? matrix_->numCols() : 0; }
    [[nodiscard]] bool isDistributed() const noexcept override { return true; }
    [[nodiscard]] std::string backendName() const override { return "FSILSMatrix"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        matrix_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        return matrix_->getEntry(row, col);
    }

    [[nodiscard]] const void* matrixLayoutHandle() const noexcept override
    {
        if (matrix_ == nullptr) {
            return nullptr;
        }
        if (const auto shared = matrix_->shared()) {
            return shared.get();
        }
        return matrix_;
    }

    [[nodiscard]] assembly::InsertionCapabilities insertionCapabilities() const noexcept override
    {
        return assembly::InsertionCapabilities{
            .resolved_matrix_entries = matrixLayoutHandle() != nullptr,
            .resolved_vector_entries = false,
            .contiguous_combined_matrix_insert = true,
            .exact_rank_one_updates = false,
        };
    }

    void resolveMatrixEntries(std::span<const GlobalIndex> row_dofs,
                              std::span<const GlobalIndex> col_dofs,
                              std::span<GlobalIndex> resolved) const override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        matrix_->resolveMatrixEntrySlotsCached(row_dofs, col_dofs, resolved);
    }

    void addMatrixEntriesResolved(std::span<const GlobalIndex> row_dofs,
                                  std::span<const GlobalIndex> col_dofs,
                                  std::span<const GlobalIndex> resolved,
                                  std::span<const Real> local_matrix,
                                  assembly::AddMode mode = assembly::AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        matrix_requested_ += resolved.size();
        const std::size_t values_size =
            (matrix_->shared() != nullptr)
                ? static_cast<std::size_t>(matrix_->fsilsNnz()) *
                      static_cast<std::size_t>(matrix_->fsilsDof()) *
                      static_cast<std::size_t>(matrix_->fsilsDof())
                : static_cast<std::size_t>(matrix_->numRows()) * static_cast<std::size_t>(matrix_->numCols());
        for (std::size_t i = 0; i < resolved.size(); ++i) {
            const auto slot = resolved[i];
            if (slot >= 0 && static_cast<std::size_t>(slot) < values_size) {
                ++matrix_valid_;
                accumulateMatrixValueStats(local_matrix[i]);
            }
        }
        matrix_->addResolvedMatrixEntries(row_dofs, col_dofs, resolved, local_matrix, mode);
    }

private:
    void accumulateMatrixValueStats(Real value) noexcept
    {
        const Real abs_value = std::abs(value);
        matrix_value_l1_ += abs_value;
        matrix_value_l2_sq_ += value * value;
        matrix_value_max_abs_ = std::max(matrix_value_max_abs_, abs_value);
        if (abs_value > 0.0) {
            ++matrix_nonzero_;
        }
    }

    FsilsMatrix* matrix_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
    std::size_t matrix_requested_{0};
    std::size_t matrix_valid_{0};
    std::size_t matrix_nonzero_{0};
    Real matrix_value_l1_{0.0};
    Real matrix_value_l2_sq_{0.0};
    Real matrix_value_max_abs_{0.0};
};

} // namespace

FsilsMatrix::FsilsMatrix(const sparsity::SparsityPattern& sparsity)
    : FsilsMatrix(sparsity,
                  /*dof_per_node=*/1,
                  /*dof_permutation=*/{}
#if defined(FE_HAS_MPI) && FE_HAS_MPI
                  ,
                  MPI_COMM_WORLD
#endif
      )
{
}

FsilsMatrix::FsilsMatrix(const sparsity::SparsityPattern& pattern,
                         int dof_per_node,
                         std::shared_ptr<const DofPermutation> dof_permutation
#if defined(FE_HAS_MPI) && FE_HAS_MPI
                         ,
                         MPI_Comm comm
#endif
                         )
#if defined(FE_HAS_MPI) && FE_HAS_MPI
    : comm_(comm)
#endif
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException, "FsilsMatrix: sparsity pattern must be finalized");
    FE_THROW_IF(dof_per_node <= 0, InvalidArgumentException, "FsilsMatrix: dof_per_node must be > 0");

    const MPI_Comm backend_comm =
#if defined(FE_HAS_MPI) && FE_HAS_MPI
        comm_;
#else
        MPI_COMM_WORLD;
#endif

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        int comm_size = 1;
        MPI_Comm_size(backend_comm, &comm_size);
        FE_THROW_IF(comm_size != 1, NotImplementedException,
                    "FsilsMatrix: sequential SparsityPattern is only supported in serial; "
                    "use DistributedSparsityPattern for MPI runs");
    }

    global_rows_ = pattern.numRows();
    global_cols_ = pattern.numCols();
    FE_THROW_IF(global_rows_ != global_cols_, NotImplementedException,
                "FsilsMatrix: rectangular systems not supported");

    const int dof = dof_per_node;
    FE_THROW_IF(global_rows_ % dof != 0, InvalidArgumentException,
                "FsilsMatrix: global size must be divisible by dof_per_node");

    dof_permutation = normalize_dof_permutation(std::move(dof_permutation), global_rows_, "FsilsMatrix", /*allow_partial=*/false);
    const bool have_perm = dof_permutation && !dof_permutation->empty();

    const GlobalIndex gnNo_g = global_rows_ / dof;
    const int gnNo = as_int(gnNo_g, "global node count");
    const int nNo = gnNo;

    const auto row_ptr = pattern.getRowPtr();
    const auto col_idx = pattern.getColIndices();

    std::vector<int> node_row_ptr(static_cast<std::size_t>(nNo + 1), 0);
    std::vector<int> node_col_ptr;
    node_col_ptr.reserve(static_cast<std::size_t>(row_ptr.back()));

    for (int node = 0; node < nNo; ++node) {
        std::vector<int> cols;
        cols.reserve(32);
        for (int r = 0; r < dof; ++r) {
            const GlobalIndex row_fs = static_cast<GlobalIndex>(node) * dof + r;
            const GlobalIndex row_dof = have_perm
                                            ? dof_permutation->inverse[static_cast<std::size_t>(row_fs)]
                                            : row_fs;
            const auto start = row_ptr[static_cast<std::size_t>(row_dof)];
            const auto end = row_ptr[static_cast<std::size_t>(row_dof + 1)];
            for (GlobalIndex k = start; k < end; ++k) {
                const GlobalIndex col_dof = col_idx[static_cast<std::size_t>(k)];
                if (col_dof < 0 || col_dof >= global_cols_) continue;
                const GlobalIndex col_fs = have_perm
                                               ? dof_permutation->forward[static_cast<std::size_t>(col_dof)]
                                               : col_dof;
                cols.push_back(static_cast<int>(col_fs / dof));
            }
        }
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        if (cols.empty() || cols.front() != node) {
            cols.insert(std::lower_bound(cols.begin(), cols.end(), node), node);
        }

        node_row_ptr[static_cast<std::size_t>(node + 1)] =
            node_row_ptr[static_cast<std::size_t>(node)] + static_cast<int>(cols.size());
        node_col_ptr.insert(node_col_ptr.end(), cols.begin(), cols.end());
    }

    const int nnz = as_int(static_cast<GlobalIndex>(node_col_ptr.size()), "FSILS nnz blocks");
    nnz_ = nnz;

    auto shared = std::make_shared<FsilsShared>();
    shared->global_dofs = global_rows_;
    shared->dof = dof;
    shared->gnNo = gnNo;
    shared->owned_node_start = 0;
    shared->owned_node_count = nNo;
    shared->dof_permutation = std::move(dof_permutation);

    Vector<int> gNodes(nNo);
    for (int i = 0; i < nNo; ++i) {
        gNodes(i) = i;
    }

    Vector<int> rowPtr(nNo + 1);
    for (int i = 0; i < nNo + 1; ++i) {
        rowPtr(i) = node_row_ptr[static_cast<std::size_t>(i)];
    }

    Vector<int> colPtr(nnz);
    for (int i = 0; i < nnz; ++i) {
        colPtr(i) = node_col_ptr[static_cast<std::size_t>(i)];
    }

    auto commu = make_fsils_commu(backend_comm);
    fe_fsi_linear_solver::fsils_lhs_create_with_explicit_owned_nodes(
        shared->lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, /*nFaces=*/0, /*owned_nNo=*/nNo);
    FE_THROW_IF(!shared->lhs.owned_row_operator, InvalidArgumentException,
                "FsilsMatrix: FE FSILS matrices must use explicit owned-row layout");

    build_old_of_internal(*shared);

    shared->buildGlobalToOldTable();
    shared->buildGlobalToInternalTable();
    build_owned_row_halo_plan(*shared);
    validate_owned_row_halo_plan(*shared);

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    values_.assign(static_cast<std::size_t>(nnz) * block_size, 0.0);
    sort_row_columns_and_values(*shared, values_);
    maybe_log_matrix_locality(*shared);
    restore_owned_row_operator_ghost_identity(*shared, values_);

    // Build direct block-base lookup tables after sorting so the stored
    // offsets match the final colPtr/values_ layout.
    buildBlockLookupTables(*shared);
    if (env_flag_enabled("SVMP_FSILS_VALIDATE_BLOCK_LOOKUP")) {
        validateBlockLookupTables(*shared);
    }

    shared_ = std::move(shared);
}

FsilsMatrix::FsilsMatrix(const sparsity::DistributedSparsityPattern& pattern,
                         int dof_per_node,
                         std::shared_ptr<const DofPermutation> dof_permutation
#if defined(FE_HAS_MPI) && FE_HAS_MPI
                         ,
                         MPI_Comm comm
#endif
                         )
#if defined(FE_HAS_MPI) && FE_HAS_MPI
    : comm_(comm)
#endif
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException, "FsilsMatrix: distributed sparsity must be finalized");
    FE_THROW_IF(dof_per_node <= 0, InvalidArgumentException, "FsilsMatrix: dof_per_node must be > 0");
    FE_THROW_IF(!pattern.isSquare(), NotImplementedException, "FsilsMatrix: rectangular systems not supported");

    const MPI_Comm backend_comm =
#if defined(FE_HAS_MPI) && FE_HAS_MPI
        comm_;
#else
        MPI_COMM_WORLD;
#endif

    global_rows_ = pattern.globalRows();
    global_cols_ = pattern.globalCols();

    const int dof = dof_per_node;
    FE_THROW_IF(global_rows_ % dof != 0, InvalidArgumentException,
                "FsilsMatrix: global size must be divisible by dof_per_node");

    dof_permutation = normalize_dof_permutation(std::move(dof_permutation), global_rows_, "FsilsMatrix", /*allow_partial=*/true);
    const bool have_perm = dof_permutation && !dof_permutation->empty();

    const GlobalIndex gnNo_g = global_rows_ / dof;
    const int gnNo = as_int(gnNo_g, "global node count");

    const auto& owned_rows = pattern.ownedRows();
    const auto& owned_cols = pattern.ownedCols();
    FE_THROW_IF(owned_rows.size() % dof != 0,
                InvalidArgumentException,
                "FsilsMatrix: owned row count must be divisible by dof_per_node");
    FE_THROW_IF(owned_cols.first != owned_rows.first || owned_cols.last != owned_rows.last,
                InvalidArgumentException,
                "FsilsMatrix: FSILS backend requires identical row/column ownership ranges");

    const bool pattern_indices_are_backend =
        (pattern.dofIndexing() == sparsity::DistributedSparsityPattern::DofIndexing::NodalInterleaved);

    if (pattern_indices_are_backend && pattern.numGhostRows() > 0) {
        FE_THROW_IF(!ghost_rows_look_nodal_interleaved(pattern.getGhostRowMap(), dof),
                    InvalidArgumentException,
                    "FsilsMatrix: nodal-interleaved distributed sparsity must store ghost rows in node-block ordering");
    }

    auto map_pattern_to_backend = [&](GlobalIndex dof_pat) -> GlobalIndex {
        if (!have_perm || pattern_indices_are_backend) {
            return dof_pat;
        }
        if (dof_pat < 0 || dof_pat >= global_rows_) {
            return INVALID_GLOBAL_INDEX;
        }
        return dof_permutation->forward[static_cast<std::size_t>(dof_pat)];
    };

    auto map_backend_to_pattern = [&](GlobalIndex dof_fs) -> GlobalIndex {
        if (!have_perm || pattern_indices_are_backend) {
            return dof_fs;
        }
        if (dof_fs < 0 || dof_fs >= global_rows_) {
            return INVALID_GLOBAL_INDEX;
        }
        return dof_permutation->inverse[static_cast<std::size_t>(dof_fs)];
    };

    int owned_node_start = 0;
    int owned_node_count = 0;
    std::vector<int> owned_nodes;

    if (pattern_indices_are_backend) {
        FE_THROW_IF(owned_rows.first % dof != 0, InvalidArgumentException,
                    "FsilsMatrix: owned row range must align with dof_per_node blocks in nodal-interleaved indexing");
        owned_node_start = as_int(owned_rows.first / dof, "owned node start");
        owned_node_count = as_int(owned_rows.size() / dof, "owned node count");
    } else {
        std::unordered_map<int, int> count_per_node;
        count_per_node.reserve(static_cast<std::size_t>(owned_rows.size() / dof) + 1u);

        for (GlobalIndex row_pat = owned_rows.first; row_pat < owned_rows.last; ++row_pat) {
            const GlobalIndex row_fs = map_pattern_to_backend(row_pat);
            FE_THROW_IF(row_fs < 0 || row_fs >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation produced out-of-range backend row");
            const int node = static_cast<int>(row_fs / dof);
            FE_THROW_IF(node < 0 || node >= gnNo, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation mapped owned row to out-of-range node");
            ++count_per_node[node];
        }

        owned_nodes.reserve(count_per_node.size());
        for (const auto& kv : count_per_node) {
            const int node = kv.first;
            const int count = kv.second;
            FE_THROW_IF(count != dof, InvalidArgumentException,
                        "FsilsMatrix: owned DOFs do not form complete node blocks after applying DOF permutation");
            owned_nodes.push_back(node);
        }

        std::sort(owned_nodes.begin(), owned_nodes.end());
        owned_node_count = static_cast<int>(owned_nodes.size());
        FE_THROW_IF(owned_node_count <= 0, InvalidArgumentException, "FsilsMatrix: no owned nodes");

        owned_node_start = owned_nodes.front();
        const bool contiguous = (owned_nodes.back() - owned_nodes.front() + 1) == owned_node_count;
        if (contiguous) {
            owned_nodes.clear();
        }
    }

    auto is_owned_node = [&](int node) -> bool {
        if (!owned_nodes.empty()) {
            return std::binary_search(owned_nodes.begin(), owned_nodes.end(), node);
        }
        return node >= owned_node_start && node < owned_node_start + owned_node_count;
    };

    // Derive ghost nodes from stored ghost rows (overlap model).
    std::vector<int> ghost_nodes;
    if (pattern.numGhostRows() > 0) {
        auto ghost_row_map = pattern.getGhostRowMap();

        std::unordered_map<int, int> count_per_node;
        count_per_node.reserve(static_cast<std::size_t>(ghost_row_map.size() / dof) + 1u);

        for (const GlobalIndex row_pat : ghost_row_map) {
            const GlobalIndex row_fs = map_pattern_to_backend(row_pat);
            FE_THROW_IF(row_fs < 0 || row_fs >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: invalid ghost row index after applying DOF permutation");
            const int node = static_cast<int>(row_fs / dof);
            FE_THROW_IF(node < 0 || node >= gnNo, InvalidArgumentException,
                        "FsilsMatrix: ghost row out of range");
            ++count_per_node[node];
        }

        ghost_nodes.reserve(count_per_node.size());
        for (const auto& kv : count_per_node) {
            const int node = kv.first;
            const int count = kv.second;
            FE_THROW_IF(is_owned_node(node), InvalidArgumentException,
                        "FsilsMatrix: ghost rows must not overlap owned rows");
            FE_THROW_IF(count != dof, InvalidArgumentException,
                        "FsilsMatrix: ghost rows must include all dof components for each ghost node");
            ghost_nodes.push_back(node);
        }

        std::sort(ghost_nodes.begin(), ghost_nodes.end());
        ghost_nodes.erase(std::unique(ghost_nodes.begin(), ghost_nodes.end()), ghost_nodes.end());
    }

    // For distributed overlap runs we allow partial DOF permutations, but the locally present
    // (owned + ghost) backend DOFs must still be mapped consistently.
    if (have_perm) {
        const auto& fwd = dof_permutation->forward;
        const auto& inv = dof_permutation->inverse;

        auto validate_backend_dof = [&](GlobalIndex dof_fs) {
            FE_THROW_IF(dof_fs < 0 || dof_fs >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: local backend DOF out of range");
            const GlobalIndex fe = inv[static_cast<std::size_t>(dof_fs)];
            FE_THROW_IF(fe == INVALID_GLOBAL_INDEX, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation missing mapping for locally present backend DOF");
            FE_THROW_IF(fe < 0 || fe >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation inverse mapped to out-of-range FE DOF");
            const GlobalIndex dof_fs_back = fwd[static_cast<std::size_t>(fe)];
            FE_THROW_IF(dof_fs_back != dof_fs, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation forward/inverse mismatch for locally present backend DOF");
        };

        auto node_has_complete_mapping = [&](int global_node) {
            if (global_node < 0 || global_node >= gnNo) {
                return false;
            }
            for (int r = 0; r < dof; ++r) {
                const auto dof_fs = static_cast<GlobalIndex>(global_node) * dof + r;
                if (dof_fs < 0 || static_cast<std::size_t>(dof_fs) >= inv.size()) {
                    return false;
                }
                const auto fe = inv[static_cast<std::size_t>(dof_fs)];
                if (fe == INVALID_GLOBAL_INDEX) {
                    return false;
                }
            }
            return true;
        };

        auto validate_node = [&](int global_node) {
            FE_THROW_IF(global_node < 0 || global_node >= gnNo, InvalidArgumentException,
                        "FsilsMatrix: local node out of range");
            for (int r = 0; r < dof; ++r) {
                validate_backend_dof(static_cast<GlobalIndex>(global_node) * dof + r);
            }
        };

        if (!owned_nodes.empty()) {
            for (const int node : owned_nodes) {
                validate_node(node);
            }
        } else {
            for (int node = owned_node_start; node < owned_node_start + owned_node_count; ++node) {
                validate_node(node);
            }
        }
        for (const int node : ghost_nodes) {
            if (!pattern_indices_are_backend || node_has_complete_mapping(node)) {
                validate_node(node);
            }
        }
    }

    const int nNo = owned_node_count + static_cast<int>(ghost_nodes.size());
    const int nnz_max = std::numeric_limits<int>::max();

    // Build node-level CSR (old local ordering: owned nodes then ghosts).
    std::vector<int> node_row_ptr(static_cast<std::size_t>(nNo + 1), 0);
    std::vector<int> node_col_ptr;
    node_col_ptr.reserve(static_cast<std::size_t>(pattern.getLocalNnz()));

    auto shared = std::make_shared<FsilsShared>();
    shared->global_dofs = global_rows_;
    shared->dof = dof;
    shared->gnNo = gnNo;
    shared->owned_node_start = owned_node_start;
    shared->owned_node_count = owned_node_count;
    shared->owned_nodes = std::move(owned_nodes);
    shared->ghost_nodes = ghost_nodes;
    shared->dof_permutation = dof_permutation;

    auto gather_row_nodes = [&](GlobalIndex row_fs, std::vector<int>& out_nodes) {
        out_nodes.clear();

        const GlobalIndex row_pat = map_backend_to_pattern(row_fs);
        if (row_pat < 0 || row_pat >= global_rows_) {
            return;
        }

        auto push_col_node = [&](GlobalIndex col_pat) {
            if (col_pat < 0 || col_pat >= global_cols_) {
                return;
            }
            const GlobalIndex col_fs = map_pattern_to_backend(col_pat);
            if (col_fs < 0 || col_fs >= global_cols_) {
                return;
            }
            out_nodes.push_back(static_cast<int>(col_fs / dof));
        };

        if (owned_rows.contains(row_pat)) {
            const GlobalIndex local_row = row_pat - owned_rows.first;
            const auto diag_cols = pattern.getRowDiagCols(local_row);
            const auto offdiag_cols = pattern.getRowOffdiagCols(local_row);
            out_nodes.reserve(static_cast<std::size_t>(diag_cols.size() + offdiag_cols.size()));

            for (const GlobalIndex local_col : diag_cols) {
                const GlobalIndex col_pat = local_col + owned_cols.first;
                push_col_node(col_pat);
            }
            for (const GlobalIndex ghost_idx : offdiag_cols) {
                const GlobalIndex col_pat = pattern.ghostColToGlobal(ghost_idx);
                push_col_node(col_pat);
            }
            return;
        }

        const GlobalIndex ghost_row = pattern.globalToGhostRow(row_pat);
        FE_THROW_IF(ghost_row < 0, InvalidArgumentException,
                    "FsilsMatrix: missing ghost row sparsity for row " + std::to_string(row_pat));
        const auto cols = pattern.getGhostRowCols(ghost_row);
        out_nodes.reserve(cols.size());
        for (const GlobalIndex col_pat : cols) {
            push_col_node(col_pat);
        }
    };

    std::vector<int> dof_row_nodes;
    std::vector<int> node_cols;

    for (int old = 0; old < nNo; ++old) {
        const int global_node = shared->oldToGlobalNode(old);
        FE_THROW_IF(global_node < 0, InvalidArgumentException,
                    "FsilsMatrix: invalid old->global node mapping");

        node_cols.clear();
        if (old >= owned_node_count) {
            // PETSc-style distributed matrices store only owned rows. Ghost
            // nodes remain in the local layout as columns/vector halo slots;
            // keep a diagonal placeholder so FSILS preconditioner bookkeeping
            // has a valid row and diagonal pointer for every local node.
            node_cols.push_back(global_node);
        } else {
            for (int r = 0; r < dof; ++r) {
                const GlobalIndex row_dof = static_cast<GlobalIndex>(global_node) * dof + r;
                gather_row_nodes(row_dof, dof_row_nodes);
                node_cols.insert(node_cols.end(), dof_row_nodes.begin(), dof_row_nodes.end());
            }
        }

        std::sort(node_cols.begin(), node_cols.end());
        node_cols.erase(std::unique(node_cols.begin(), node_cols.end()), node_cols.end());

        if (node_cols.empty() || node_cols.front() != global_node) {
            node_cols.insert(std::lower_bound(node_cols.begin(), node_cols.end(), global_node), global_node);
        }

        for (const int col_global_node : node_cols) {
            const int col_old = shared->globalNodeToOld(col_global_node);
            FE_THROW_IF(col_old < 0, InvalidArgumentException,
                        "FsilsMatrix: column node " + std::to_string(col_global_node) +
                            " is not present locally (ghost row closure required)");
            node_col_ptr.push_back(col_old);
        }

        FE_THROW_IF(node_col_ptr.size() > static_cast<std::size_t>(nnz_max), InvalidArgumentException,
                    "FsilsMatrix: local nnz exceeds FSILS int index range");
        node_row_ptr[static_cast<std::size_t>(old + 1)] = static_cast<int>(node_col_ptr.size());
    }

    const int nnz = static_cast<int>(node_col_ptr.size());
    nnz_ = nnz;

    Vector<int> gNodes(nNo);
    for (int old = 0; old < nNo; ++old) {
        const int global_node = shared->oldToGlobalNode(old);
        FE_THROW_IF(global_node < 0, InvalidArgumentException,
                    "FsilsMatrix: invalid old->global node mapping");
        gNodes(old) = global_node;
    }

    Vector<int> rowPtr(nNo + 1);
    for (int i = 0; i < nNo + 1; ++i) {
        rowPtr(i) = node_row_ptr[static_cast<std::size_t>(i)];
    }

    Vector<int> colPtr(nnz);
    for (int i = 0; i < nnz; ++i) {
        colPtr(i) = node_col_ptr[static_cast<std::size_t>(i)];
    }

    auto commu = make_fsils_commu(backend_comm);
    fe_fsi_linear_solver::fsils_lhs_create_with_explicit_owned_nodes(
        shared->lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, /*nFaces=*/0, owned_node_count);
    FE_THROW_IF(!shared->lhs.owned_row_operator, InvalidArgumentException,
                "FsilsMatrix: FE FSILS matrices must use explicit owned-row layout");

    build_old_of_internal(*shared);

    shared->buildGlobalToOldTable();
    shared->buildGlobalToInternalTable();
    build_owned_row_halo_plan(*shared);
    validate_owned_row_halo_plan(*shared);

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    values_.assign(static_cast<std::size_t>(nnz) * block_size, 0.0);
    sort_row_columns_and_values(*shared, values_);
    maybe_log_matrix_locality(*shared);
    restore_owned_row_operator_ghost_identity(*shared, values_);

    // Build direct block-base lookup tables after sorting so the stored
    // offsets match the final colPtr/values_ layout.
    buildBlockLookupTables(*shared);
    if (env_flag_enabled("SVMP_FSILS_VALIDATE_BLOCK_LOOKUP")) {
        validateBlockLookupTables(*shared);
    }

    shared_ = std::move(shared);
}

FsilsMatrix::~FsilsMatrix() = default;

FsilsMatrix::FsilsMatrix(FsilsMatrix&&) noexcept = default;
FsilsMatrix& FsilsMatrix::operator=(FsilsMatrix&&) noexcept = default;

GlobalIndex FsilsMatrix::numRows() const noexcept
{
    return global_rows_;
}

GlobalIndex FsilsMatrix::numCols() const noexcept
{
    return global_cols_;
}

void FsilsMatrix::resolveMatrixEntrySlotsCached(std::span<const GlobalIndex> row_dofs,
                                                std::span<const GlobalIndex> col_dofs,
                                                std::span<GlobalIndex> resolved) const
{
    FE_THROW_IF(resolved.size() != row_dofs.size() * col_dofs.size(),
                InvalidArgumentException,
                "FsilsMatrix::resolveMatrixEntrySlotsCached: resolved size mismatch");

    if (!shared_) {
        std::fill(resolved.begin(), resolved.end(), INVALID_GLOBAL_INDEX);
        return;
    }
    resolveFsilsMatrixEntrySlotsUncached(*this, row_dofs, col_dofs, resolved);
}

void FsilsMatrix::addResolvedMatrixEntries(std::span<const GlobalIndex> row_dofs,
                                           std::span<const GlobalIndex> col_dofs,
                                           std::span<const GlobalIndex> resolved,
                                           std::span<const Real> local_matrix,
                                           assembly::AddMode mode)
{
    FE_THROW_IF(resolved.size() != local_matrix.size() ||
                    resolved.size() != row_dofs.size() * col_dofs.size(),
                InvalidArgumentException,
                "FsilsMatrix::addResolvedMatrixEntries: size mismatch");

    if (!shared_) {
        const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());
        for (GlobalIndex i = 0; i < n_rows; ++i) {
            const auto row = row_dofs[static_cast<std::size_t>(i)];
            if (row < 0 || row >= numRows()) {
                continue;
            }
            for (GlobalIndex j = 0; j < n_cols; ++j) {
                const auto col = col_dofs[static_cast<std::size_t>(j)];
                if (col < 0 || col >= numCols()) {
                    continue;
                }
                const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                addValue(row, col, local_matrix[local_idx], mode);
            }
        }
        return;
    }

    auto* values = fsilsValuesPtr();
    FE_CHECK_NOT_NULL(values, "FsilsMatrix::addResolvedMatrixEntries: values");
    const std::size_t values_size =
        static_cast<std::size_t>(fsilsNnz()) *
        static_cast<std::size_t>(shared_->dof) *
        static_cast<std::size_t>(shared_->dof);

    const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
    const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());
    std::vector<unsigned char> owned_row;
    std::vector<unsigned char> off_owner_row;
    std::vector<unsigned char> valid_col;
    bool all_rows_owned = true;
    owned_row.resize(row_dofs.size(), 0);
    off_owner_row.resize(row_dofs.size(), 0);
    valid_col.resize(col_dofs.size(), 0);
    for (GlobalIndex i = 0; i < n_rows; ++i) {
        const auto status = classify_fe_matrix_row(
            *shared_, row_dofs[static_cast<std::size_t>(i)], numRows());
        if (status == LocalRowOwnership::Owned) {
            owned_row[static_cast<std::size_t>(i)] = 1;
        } else {
            all_rows_owned = false;
            if (status == LocalRowOwnership::NonOwnedLocal) {
                off_owner_row[static_cast<std::size_t>(i)] = 1;
            }
        }
    }
    for (GlobalIndex j = 0; j < n_cols; ++j) {
        if (is_valid_fe_matrix_col(
                *shared_, col_dofs[static_cast<std::size_t>(j)], numCols())) {
            valid_col[static_cast<std::size_t>(j)] = 1;
        }
    }

    std::uint64_t dropped_entries = 0;
    for (GlobalIndex i = 0; i < n_rows; ++i) {
        const auto row_idx = static_cast<std::size_t>(i);
        if (owned_row[row_idx] == 0) {
            continue;
        }
        for (GlobalIndex j = 0; j < n_cols; ++j) {
            const auto col_idx = static_cast<std::size_t>(j);
            if (valid_col[col_idx] == 0) {
                continue;
            }
            const auto idx = static_cast<std::size_t>(i * n_cols + j);
            const auto slot = resolved[idx];
            if (slot < 0 || static_cast<std::size_t>(slot) >= values_size) {
                ++dropped_entries;
            }
        }
    }
    if (dropped_entries > 0) {
        dropped_entry_count_.fetch_add(dropped_entries, std::memory_order_relaxed);
    }

    const auto* slots = resolved.data();
    const auto* local = local_matrix.data();
    const std::size_t n = resolved.size();
    constexpr std::size_t kMinContiguousRun = 2u;

    auto apply_scalar = [&](GlobalIndex slot, Real value) {
        if (slot < 0 || static_cast<std::size_t>(slot) >= values_size) {
            return;
        }
        Real& dst = values[static_cast<std::size_t>(slot)];
        switch (mode) {
            case assembly::AddMode::Add:
                dst += value;
                break;
            case assembly::AddMode::Insert:
                dst = value;
                break;
            case assembly::AddMode::Max:
                dst = std::max(dst, value);
                break;
            case assembly::AddMode::Min:
                dst = std::min(dst, value);
                break;
        }
    };

    if (!all_rows_owned) {
        for (GlobalIndex i = 0; i < n_rows; ++i) {
            const auto row_idx = static_cast<std::size_t>(i);
            if (off_owner_row[row_idx] != 0) {
                off_owner_write_count_.fetch_add(static_cast<std::uint64_t>(n_cols),
                                                 std::memory_order_relaxed);
                continue;
            }
            if (owned_row[row_idx] == 0) {
                continue;
            }
            for (GlobalIndex j = 0; j < n_cols; ++j) {
                const auto idx = static_cast<std::size_t>(i * n_cols + j);
                apply_scalar(resolved[idx], local_matrix[idx]);
            }
        }
        return;
    }

    const auto applyContiguousRuns =
        [&](auto&& op_scalar, auto&& op_run) {
            std::size_t idx = 0;
            while (idx < n) {
                const auto slot = slots[idx];
                if (slot < 0 || static_cast<std::size_t>(slot) >= values_size) {
                    ++idx;
                    continue;
                }

                std::size_t run = 1u;
                while (idx + run < n) {
                    const auto next_slot = slots[idx + run];
                    if (next_slot < 0 ||
                        static_cast<std::size_t>(next_slot) >= values_size ||
                        next_slot != slot + static_cast<GlobalIndex>(run)) {
                        break;
                    }
                    ++run;
                }

                if (run >= kMinContiguousRun) {
                    op_run(values + static_cast<std::size_t>(slot), local + idx, run);
                    idx += run;
                } else {
                    op_scalar(values[static_cast<std::size_t>(slot)], local[idx]);
                    ++idx;
                }
            }
        };

    switch (mode) {
        case assembly::AddMode::Add:
            applyContiguousRuns(
                [](Real& dst, Real src) { dst += src; },
                [](Real* dst, const Real* src, std::size_t count) {
                    for (std::size_t k = 0; k < count; ++k) {
                        dst[k] += src[k];
                    }
                });
            break;
        case assembly::AddMode::Insert:
            applyContiguousRuns(
                [](Real& dst, Real src) { dst = src; },
                [](Real* dst, const Real* src, std::size_t count) {
                    std::copy_n(src, count, dst);
                });
            break;
        case assembly::AddMode::Max:
            applyContiguousRuns(
                [](Real& dst, Real src) { dst = std::max(dst, src); },
                [](Real* dst, const Real* src, std::size_t count) {
                    for (std::size_t k = 0; k < count; ++k) {
                        dst[k] = std::max(dst[k], src[k]);
                    }
                });
            break;
        case assembly::AddMode::Min:
            applyContiguousRuns(
                [](Real& dst, Real src) { dst = std::min(dst, src); },
                [](Real* dst, const Real* src, std::size_t count) {
                    for (std::size_t k = 0; k < count; ++k) {
                        dst[k] = std::min(dst[k], src[k]);
                    }
                });
            break;
    }
}

void FsilsMatrix::addMatrixEntriesCached(std::span<const GlobalIndex> row_dofs,
                                         std::span<const GlobalIndex> col_dofs,
                                         std::span<const Real> local_matrix,
                                         assembly::AddMode mode)
{
    const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
    const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());
    FE_THROW_IF(local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols),
                InvalidArgumentException,
                "FsilsMatrix::addMatrixEntriesCached: local_matrix size mismatch");

    if (!shared_) {
        for (GlobalIndex i = 0; i < n_rows; ++i) {
            const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
            if (row < 0 || row >= numRows()) {
                continue;
            }
            for (GlobalIndex j = 0; j < n_cols; ++j) {
                const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                if (col < 0 || col >= numCols()) {
                    continue;
                }
                const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                addValue(row, col, local_matrix[local_idx], mode);
            }
        }
        return;
    }

    const int dof = shared_->dof;
    const std::size_t block_sz = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const auto perm = shared_->dof_permutation;
    const bool have_perm = perm && !perm->empty();
    const auto* perm_data = have_perm ? perm->forward.data() : nullptr;
    const auto perm_size = have_perm ? perm->forward.size() : std::size_t{0};
    const GlobalIndex num_rows_global = numRows();
    const GlobalIndex num_cols_global = numCols();

    struct DofInfo {
        int internal_node{-1};
        int component{-1};
        int old_node{-1};
        bool valid_global{false};
    };

    thread_local std::vector<DofInfo> row_info;
    thread_local std::vector<DofInfo> col_info;
    row_info.resize(static_cast<std::size_t>(n_rows));
    col_info.resize(static_cast<std::size_t>(n_cols));

    int cached_global_node = -1;
    int cached_internal_node = -1;
    auto resolve_dof = [&](GlobalIndex fe_dof, GlobalIndex limit) -> DofInfo {
        if (fe_dof < 0 || fe_dof >= limit) {
            return {};
        }

        GlobalIndex fs_dof = fe_dof;
        if (have_perm) {
            if (static_cast<std::size_t>(fe_dof) >= perm_size) {
                return {};
            }
            fs_dof = perm_data[static_cast<std::size_t>(fe_dof)];
        }
        if (fs_dof < 0 || fs_dof >= limit) {
            return {};
        }

        const int global_node = static_cast<int>(fs_dof / dof);
        const int comp = static_cast<int>(fs_dof % dof);
        if (global_node == cached_global_node) {
            const int old = shared_->globalNodeToOld(global_node);
            return {cached_internal_node, comp, old, true};
        }

        const int old = shared_->globalNodeToOld(global_node);
        if (old < 0 || old >= shared_->lhs.nNo) {
            return {-1, comp, old, true};
        }
        const int internal = shared_->lhs.map(old);

        cached_global_node = global_node;
        cached_internal_node = internal;
        return {internal, comp, old, true};
    };

    for (GlobalIndex i = 0; i < n_rows; ++i) {
        row_info[static_cast<std::size_t>(i)] =
            resolve_dof(row_dofs[static_cast<std::size_t>(i)], num_rows_global);
    }
    for (GlobalIndex j = 0; j < n_cols; ++j) {
        col_info[static_cast<std::size_t>(j)] =
            resolve_dof(col_dofs[static_cast<std::size_t>(j)], num_cols_global);
    }

    std::vector<std::size_t> validate_slots;
    std::vector<Real> validate_expected;
    if (mode == assembly::AddMode::Add &&
        env_flag_enabled("SVMP_FSILS_VALIDATE_ADD_MATRIX")) {
        validate_slots.reserve(static_cast<std::size_t>(n_rows * n_cols));
        validate_expected.reserve(static_cast<std::size_t>(n_rows * n_cols));
        auto accumulate_expected = [&](std::size_t slot, Real delta) {
            for (std::size_t idx = 0; idx < validate_slots.size(); ++idx) {
                if (validate_slots[idx] == slot) {
                    validate_expected[idx] += delta;
                    return;
                }
            }
            validate_slots.push_back(slot);
            validate_expected.push_back(values_[slot] + delta);
        };

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            const auto& ri = row_info[static_cast<std::size_t>(i)];
            if (ri.internal_node < 0) {
                continue;
            }
            if (ri.old_node >= shared_->owned_node_count) {
                continue;
            }
            for (GlobalIndex j = 0; j < n_cols; ++j) {
                const auto& ci = col_info[static_cast<std::size_t>(j)];
                if (ci.internal_node < 0) {
                    continue;
                }
                const GlobalIndex block_base = shared_->blockBase(ri.internal_node, ci.internal_node);
                if (block_base == INVALID_GLOBAL_INDEX) {
                    continue;
                }
                const std::size_t slot = static_cast<std::size_t>(block_base) +
                    block_entry_index(dof, ri.component, ci.component);
                if (slot >= values_.size()) {
                    continue;
                }
                const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                accumulate_expected(slot, local_matrix[local_idx]);
            }
        }
    }

    thread_local std::vector<Real> block_buf;
    block_buf.resize(block_sz);

    for (GlobalIndex i0 = 0; i0 < n_rows; ) {
        const auto& ri0 = row_info[static_cast<std::size_t>(i0)];
        if (ri0.internal_node < 0) {
            ++i0;
            continue;
        }

        GlobalIndex i1 = i0 + 1;
        while (i1 < n_rows &&
               row_info[static_cast<std::size_t>(i1)].internal_node == ri0.internal_node) {
            ++i1;
        }

        if (ri0.old_node >= shared_->owned_node_count) {
            off_owner_write_count_.fetch_add(
                static_cast<std::uint64_t>(i1 - i0) * static_cast<std::uint64_t>(n_cols),
                std::memory_order_relaxed);
            i0 = i1;
            continue;
        }

        for (GlobalIndex j0 = 0; j0 < n_cols; ) {
            const auto& ci0 = col_info[static_cast<std::size_t>(j0)];
            if (ci0.internal_node < 0) {
                if (ci0.valid_global) {
                    dropped_entry_count_.fetch_add(
                        static_cast<std::uint64_t>(i1 - i0),
                        std::memory_order_relaxed);
                }
                ++j0;
                continue;
            }

            GlobalIndex j1 = j0 + 1;
            while (j1 < n_cols &&
                   col_info[static_cast<std::size_t>(j1)].internal_node == ci0.internal_node) {
                ++j1;
            }

            const auto row_run = i1 - i0;
            const auto col_run = j1 - j0;
            const GlobalIndex block_base = shared_->blockBase(ri0.internal_node, ci0.internal_node);
            if (block_base == INVALID_GLOBAL_INDEX) {
                dropped_entry_count_.fetch_add(
                    static_cast<std::uint64_t>(row_run) * static_cast<std::uint64_t>(col_run),
                    std::memory_order_relaxed);
                j0 = j1;
                continue;
            }

            const auto base = static_cast<std::size_t>(block_base);
            if (base + block_sz > values_.size()) {
                dropped_entry_count_.fetch_add(
                    static_cast<std::uint64_t>(row_run) *
                        static_cast<std::uint64_t>(col_run),
                    std::memory_order_relaxed);
                j0 = j1;
                continue;
            }

            Real* dst = values_.data() + base;
            if (mode == assembly::AddMode::Add) {
                for (GlobalIndex di = i0; di < i1; ++di) {
                    const int r = row_info[static_cast<std::size_t>(di)].component;
                    for (GlobalIndex dj = j0; dj < j1; ++dj) {
                        const int c = col_info[static_cast<std::size_t>(dj)].component;
                        const auto local_idx = static_cast<std::size_t>(di * n_cols + dj);
                        dst[block_entry_index(dof, r, c)] += local_matrix[local_idx];
                    }
                }
                j0 = j1;
                continue;
            }

            if (row_run == dof && col_run == dof) {
                std::fill(block_buf.begin(), block_buf.end(), Real(0));
                for (GlobalIndex di = i0; di < i1; ++di) {
                    const int r = row_info[static_cast<std::size_t>(di)].component;
                    for (GlobalIndex dj = j0; dj < j1; ++dj) {
                        const int c = col_info[static_cast<std::size_t>(dj)].component;
                        const auto local_idx = static_cast<std::size_t>(di * n_cols + dj);
                        block_buf[block_entry_index(dof, r, c)] = local_matrix[local_idx];
                    }
                }
                switch (mode) {
                    case assembly::AddMode::Insert:
                        for (std::size_t k = 0; k < block_sz; ++k) dst[k] = block_buf[k];
                        break;
                    case assembly::AddMode::Max:
                        for (std::size_t k = 0; k < block_sz; ++k) dst[k] = std::max(dst[k], block_buf[k]);
                        break;
                    case assembly::AddMode::Min:
                        for (std::size_t k = 0; k < block_sz; ++k) dst[k] = std::min(dst[k], block_buf[k]);
                        break;
                    default:
                        break;
                }
                j0 = j1;
                continue;
            }

            for (GlobalIndex di = i0; di < i1; ++di) {
                const int r = row_info[static_cast<std::size_t>(di)].component;
                for (GlobalIndex dj = j0; dj < j1; ++dj) {
                    const int c = col_info[static_cast<std::size_t>(dj)].component;
                    const auto local_idx = static_cast<std::size_t>(di * n_cols + dj);
                    auto& dst_scalar = dst[block_entry_index(dof, r, c)];
                    switch (mode) {
                        case assembly::AddMode::Insert:
                            dst_scalar = local_matrix[local_idx];
                            break;
                        case assembly::AddMode::Max:
                            dst_scalar = std::max(dst_scalar, local_matrix[local_idx]);
                            break;
                        case assembly::AddMode::Min:
                            dst_scalar = std::min(dst_scalar, local_matrix[local_idx]);
                            break;
                        default:
                            break;
                    }
                }
            }

            j0 = j1;
        }

        i0 = i1;
    }

    if (!validate_slots.empty()) {
        constexpr Real abs_tol = 1e-12;
        constexpr Real rel_tol = 1e-10;
        for (std::size_t idx = 0; idx < validate_slots.size(); ++idx) {
            const auto slot = validate_slots[idx];
            const Real actual = values_[slot];
            const Real expected = validate_expected[idx];
            const Real tol = abs_tol + rel_tol * std::max(std::abs(expected), std::abs(actual));
            FE_THROW_IF(std::abs(actual - expected) > tol,
                        FEException,
                        "FsilsMatrix: Add-mode insertion validation failed at slot=" +
                            std::to_string(slot) + " actual=" + std::to_string(actual) +
                            " expected=" + std::to_string(expected));
        }
    }
}

void FsilsMatrix::zero()
{
    std::fill(values_.begin(), values_.end(), 0.0);
    if (shared_) {
        restore_owned_row_operator_ghost_identity(*shared_, values_);
    }
    resetDroppedEntryCount();
    resetOffOwnerWriteCount();
}

void FsilsMatrix::finalizeAssembly()
{
    if (shared_) {
        restore_owned_row_operator_ghost_identity(*shared_, values_);
    }
}

void FsilsMatrix::mult(const GenericVector& x_in, GenericVector& y_in) const
{
    const auto* x = dynamic_cast<const FsilsVector*>(&x_in);
    auto* y = dynamic_cast<FsilsVector*>(&y_in);
    FE_THROW_IF(!x || !y, InvalidArgumentException, "FsilsMatrix::mult: backend mismatch");

    FE_THROW_IF(x->shared() != shared_.get() || y->shared() != shared_.get(),
                InvalidArgumentException, "FsilsMatrix::mult: vector layout mismatch");

    auto& lhs = shared_->lhs;
    const int dof = shared_->dof;
    const int nNo = lhs.nNo;
    const int nnz = lhs.nnz;
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);

    FE_THROW_IF(static_cast<int>(x->data().size()) != dof * nNo ||
                    static_cast<int>(y->data().size()) != dof * nNo,
                InvalidArgumentException, "FsilsMatrix::mult: local size mismatch");
    FE_THROW_IF(values_.size() != static_cast<std::size_t>(nnz) * block_size, FEException,
                "FsilsMatrix::mult: invalid FSILS value storage");

    // Map input from old local ordering -> FSILS internal ordering.
    std::vector<double> u_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo), 0.0);
    const auto& x_old = x->data();
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            u_internal[static_cast<std::size_t>(c) + static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)] =
                x_old[static_cast<std::size_t>(c) + static_cast<std::size_t>(old) * static_cast<std::size_t>(dof)];
        }
    }

    std::vector<double> ku_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo), 0.0);

    Array<double> K(dof * dof, nnz, const_cast<double*>(values_.data()));
    Array<double> U(dof, nNo, u_internal.data());
    Array<double> KU(dof, nNo, ku_internal.data());

    fe_fsi_linear_solver::fsils_syncv_owned_to_ghost(lhs, dof, U);
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, K, U, KU);

    // Map output back to old local ordering.
    auto& y_old = y->data();
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            y_old[static_cast<std::size_t>(c) + static_cast<std::size_t>(old) * static_cast<std::size_t>(dof)] =
                ku_internal[static_cast<std::size_t>(c) + static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)];
        }
    }
}

void FsilsMatrix::multAdd(const GenericVector& x_in, GenericVector& y_in) const
{
    auto* y = dynamic_cast<FsilsVector*>(&y_in);
    FE_THROW_IF(!y, InvalidArgumentException, "FsilsMatrix::multAdd: backend mismatch");

    FsilsVector tmp(shared());
    tmp.zero();
    mult(x_in, tmp);

    auto yspan = y->localSpan();
    auto tspan = tmp.localSpan();
    FE_THROW_IF(yspan.size() != tspan.size(), FEException, "FsilsMatrix::multAdd: size mismatch");

    for (std::size_t i = 0; i < yspan.size(); ++i) {
        yspan[i] += tspan[i];
    }
}

std::unique_ptr<assembly::GlobalSystemView> FsilsMatrix::createAssemblyView()
{
    return std::make_unique<FsilsMatrixView>(*this);
}

bool FsilsMatrix::usesOwnedRowOperator() const noexcept
{
    return shared_ != nullptr && shared_->lhs.owned_row_operator;
}

bool FsilsMatrix::ownsFeDofRow(GlobalIndex fe_dof) const noexcept
{
    if (shared_ == nullptr || fe_dof < 0 || fe_dof >= global_rows_ || shared_->dof <= 0) {
        return false;
    }

    GlobalIndex backend_dof = fe_dof;
    if (const auto perm = shared_->dof_permutation; perm && !perm->empty()) {
        if (static_cast<std::size_t>(fe_dof) >= perm->forward.size()) {
            return false;
        }
        backend_dof = perm->forward[static_cast<std::size_t>(fe_dof)];
    }
    if (backend_dof < 0 || backend_dof >= global_rows_) {
        return false;
    }

    const int global_node = static_cast<int>(backend_dof / shared_->dof);
    const int old = shared_->globalNodeToOld(global_node);
    return old >= 0 && old < shared_->owned_node_count;
}

	Real FsilsMatrix::getEntry(GlobalIndex row, GlobalIndex col) const
	{
	    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
	        return 0.0;
	    }
	    FE_THROW_IF(!shared_, FEException, "FsilsMatrix::getEntry: missing FSILS layout");

		    if (const auto perm = shared_->dof_permutation; perm && !perm->empty()) {
		        const auto& fwd = perm->forward;
		        if (static_cast<std::size_t>(row) >= fwd.size() || static_cast<std::size_t>(col) >= fwd.size()) {
		            return 0.0;
		        }
		        row = fwd[static_cast<std::size_t>(row)];
		        col = fwd[static_cast<std::size_t>(col)];
		    }
		    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
		        return 0.0;
		    }

		    const int dof = shared_->dof;
		    const int global_row_node = static_cast<int>(row / dof);
		    const int global_col_node = static_cast<int>(col / dof);
		    const int row_comp = static_cast<int>(row % dof);
	    const int col_comp = static_cast<int>(col % dof);

	    const int row_internal = shared_->globalNodeToInternal(global_row_node);
	    const int col_internal = shared_->globalNodeToInternal(global_col_node);
	    if (row_internal < 0 || col_internal < 0) {
	        return 0.0;
	    }

	    const auto& lhs = shared_->lhs;
	    const int start = lhs.rowPtr(0, row_internal);
	    const int end = lhs.rowPtr(1, row_internal);
	    if (start < 0 || end < start) {
	        return 0.0;
	    }
	    const auto* begin = lhs.colPtr.data() + start;
	    const auto* finish = lhs.colPtr.data() + end + 1;
	    const auto col_it = std::lower_bound(begin, finish,
	                                         static_cast<fe_fsi_linear_solver::fsils_int>(col_internal));
	    if (col_it == finish || *col_it != col_internal) {
	        return 0.0;
	    }

	    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
	    const std::size_t base = static_cast<std::size_t>(col_it - lhs.colPtr.data()) * block_size;
	    const std::size_t off = block_entry_index(dof, row_comp, col_comp);
	    if (base + off >= values_.size()) {
	        return 0.0;
	    }
	    return values_[base + off];
}

	void FsilsMatrix::addValue(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode)
	{
        const GlobalIndex fe_row_in = row;
        const GlobalIndex fe_col_in = col;
        const bool trace_row = fsils_trace_add_row(fe_row_in);
	    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
            if (trace_row) {
                std::fprintf(stderr,
                             "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld value=%.17g mode=%d path=outside_global\n",
                             shared_ ? shared_->lhs.commu.task : -1,
                             static_cast<long long>(fe_row_in),
                             static_cast<long long>(fe_col_in),
                             static_cast<double>(value),
                             static_cast<int>(mode));
            }
	        return;
	    }
	    FE_THROW_IF(!shared_, FEException, "FsilsMatrix::addValue: missing FSILS layout");

		    if (const auto perm = shared_->dof_permutation; perm && !perm->empty()) {
		        const auto& fwd = perm->forward;
		        if (static_cast<std::size_t>(row) >= fwd.size() || static_cast<std::size_t>(col) >= fwd.size()) {
                    if (trace_row) {
                        std::fprintf(stderr,
                                     "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld value=%.17g mode=%d path=perm_index_outside fwd_size=%zu\n",
                                     shared_->lhs.commu.task,
                                     static_cast<long long>(fe_row_in),
                                     static_cast<long long>(fe_col_in),
                                     static_cast<double>(value),
                                     static_cast<int>(mode),
                                     fwd.size());
                    }
		            return;
		        }
		        row = fwd[static_cast<std::size_t>(row)];
		        col = fwd[static_cast<std::size_t>(col)];
		    }
		    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
                if (trace_row) {
                    std::fprintf(stderr,
                                 "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld backend_row=%lld backend_col=%lld value=%.17g mode=%d path=backend_invalid\n",
                                 shared_->lhs.commu.task,
                                 static_cast<long long>(fe_row_in),
                                 static_cast<long long>(fe_col_in),
                                 static_cast<long long>(row),
                                 static_cast<long long>(col),
                                 static_cast<double>(value),
                                 static_cast<int>(mode));
                }
		        return;
		    }

		    const int dof = shared_->dof;
		    const int global_row_node = static_cast<int>(row / dof);
		    const int global_col_node = static_cast<int>(col / dof);
		    const int row_comp = static_cast<int>(row % dof);
	    const int col_comp = static_cast<int>(col % dof);

        const int row_old = shared_->globalNodeToOld(global_row_node);
        if (row_old >= shared_->owned_node_count && row_old < shared_->lhs.nNo) {
            off_owner_write_count_.fetch_add(1, std::memory_order_relaxed);
            if (trace_row) {
                std::fprintf(stderr,
                             "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld backend_row=%lld backend_col=%lld row_node=%d row_old=%d owned_nodes=%d value=%.17g mode=%d path=off_owner_row\n",
                             shared_->lhs.commu.task,
                             static_cast<long long>(fe_row_in),
                             static_cast<long long>(fe_col_in),
                             static_cast<long long>(row),
                             static_cast<long long>(col),
                             global_row_node,
                             row_old,
                             shared_->owned_node_count,
                             static_cast<double>(value),
                             static_cast<int>(mode));
            }
            return;
        }

	    const int row_internal = (row_old >= 0 && row_old < shared_->lhs.nNo)
            ? shared_->lhs.map(row_old)
            : -1;
	    const int col_internal = shared_->globalNodeToInternal(global_col_node);
	    if (row_internal < 0 || col_internal < 0) {
            if (row_old >= 0 && row_old < shared_->owned_node_count) {
                dropped_entry_count_.fetch_add(1, std::memory_order_relaxed);
            }
            if (trace_row) {
                std::fprintf(stderr,
                             "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld backend_row=%lld backend_col=%lld row_node=%d col_node=%d row_int=%d col_int=%d value=%.17g mode=%d path=internal_missing\n",
                             shared_->lhs.commu.task,
                             static_cast<long long>(fe_row_in),
                             static_cast<long long>(fe_col_in),
                             static_cast<long long>(row),
                             static_cast<long long>(col),
                             global_row_node,
                             global_col_node,
                             row_internal,
                             col_internal,
                             static_cast<double>(value),
                             static_cast<int>(mode));
            }
	        return;
	    }

	    const auto& lhs = shared_->lhs;
	    const int start = lhs.rowPtr(0, row_internal);
	    const int end = lhs.rowPtr(1, row_internal);
	    if (start < 0 || end < start) {
	        dropped_entry_count_.fetch_add(1, std::memory_order_relaxed);
            if (trace_row) {
                std::fprintf(stderr,
                             "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld backend_row=%lld backend_col=%lld row_node=%d col_node=%d row_int=%d col_int=%d value=%.17g mode=%d path=row_empty\n",
                             shared_->lhs.commu.task,
                             static_cast<long long>(fe_row_in),
                             static_cast<long long>(fe_col_in),
                             static_cast<long long>(row),
                             static_cast<long long>(col),
                             global_row_node,
                             global_col_node,
                             row_internal,
                             col_internal,
                             static_cast<double>(value),
                             static_cast<int>(mode));
            }
	        return;
	    }
	    const auto* begin = lhs.colPtr.data() + start;
	    const auto* finish = lhs.colPtr.data() + end + 1;
	    const auto col_it = std::lower_bound(begin, finish,
	                                         static_cast<fe_fsi_linear_solver::fsils_int>(col_internal));
	    if (col_it == finish || *col_it != col_internal) {
	        dropped_entry_count_.fetch_add(1, std::memory_order_relaxed);
            if (trace_row) {
                std::fprintf(stderr,
                             "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld backend_row=%lld backend_col=%lld row_node=%d col_node=%d row_int=%d col_int=%d row_range=[%d,%d] value=%.17g mode=%d path=missing_column\n",
                             shared_->lhs.commu.task,
                             static_cast<long long>(fe_row_in),
                             static_cast<long long>(fe_col_in),
                             static_cast<long long>(row),
                             static_cast<long long>(col),
                             global_row_node,
                             global_col_node,
                             row_internal,
                             col_internal,
                             start,
                             end,
                             static_cast<double>(value),
                             static_cast<int>(mode));
            }
	        return;
	    }

	    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
	    const std::size_t base = static_cast<std::size_t>(col_it - lhs.colPtr.data()) * block_size;
	    const std::size_t off = block_entry_index(dof, row_comp, col_comp);
	    if (base + off >= values_.size()) {
	        dropped_entry_count_.fetch_add(1, std::memory_order_relaxed);
	        return;
	    }
	    Real& dst = values_[base + off];
        if (trace_row) {
            std::fprintf(stderr,
                         "[FSILS_ADD] rank=%d fe_row=%lld fe_col=%lld backend_row=%lld backend_col=%lld row_node=%d col_node=%d row_int=%d col_int=%d slot=%zu old=%.17g delta=%.17g mode=%d path=apply\n",
                         shared_->lhs.commu.task,
                         static_cast<long long>(fe_row_in),
                         static_cast<long long>(fe_col_in),
                         static_cast<long long>(row),
                         static_cast<long long>(col),
                         global_row_node,
                         global_col_node,
                         row_internal,
                         col_internal,
                         base + off,
                         static_cast<double>(dst),
                         static_cast<double>(value),
                         static_cast<int>(mode));
        }

    switch (mode) {
        case assembly::AddMode::Add:
            dst += value;
            break;
        case assembly::AddMode::Insert:
            dst = value;
            break;
        case assembly::AddMode::Max:
            dst = std::max(dst, value);
            break;
        case assembly::AddMode::Min:
            dst = std::min(dst, value);
            break;
    }
}

void FsilsMatrix::addBlock(int row_internal, int col_internal, const Real* block_data,
                           int dof, assembly::AddMode mode)
{
    if (row_internal < 0 || row_internal >= shared_->lhs.nNo ||
        col_internal < 0 || col_internal >= shared_->lhs.nNo) {
        return;
    }

    if (static_cast<std::size_t>(row_internal) < shared_->old_of_internal.size()) {
        const int row_old = shared_->old_of_internal[static_cast<std::size_t>(row_internal)];
        if (row_old >= shared_->owned_node_count && row_old < shared_->lhs.nNo) {
            off_owner_write_count_.fetch_add(
                static_cast<std::uint64_t>(dof) * static_cast<std::uint64_t>(dof),
                std::memory_order_relaxed);
            return;
        }
    }

    const auto& lhs = shared_->lhs;
    const int start = lhs.rowPtr(0, row_internal);
    const int end = lhs.rowPtr(1, row_internal);
    if (start < 0 || end < start) {
        dropped_entry_count_.fetch_add(static_cast<std::uint64_t>(dof) * static_cast<std::uint64_t>(dof), std::memory_order_relaxed);
        return;
    }
    const auto* begin = lhs.colPtr.data() + start;
    const auto* finish = lhs.colPtr.data() + end + 1;
    const auto col_it = std::lower_bound(begin, finish,
                                         static_cast<fe_fsi_linear_solver::fsils_int>(col_internal));
    if (col_it == finish || *col_it != col_internal) {
        dropped_entry_count_.fetch_add(static_cast<std::uint64_t>(dof) * static_cast<std::uint64_t>(dof), std::memory_order_relaxed);
        return;
    }

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t base = static_cast<std::size_t>(col_it - lhs.colPtr.data()) * block_size;
    if (base + block_size > values_.size()) {
        dropped_entry_count_.fetch_add(
            static_cast<std::uint64_t>(dof) * static_cast<std::uint64_t>(dof),
            std::memory_order_relaxed);
        return;
    }
    Real* dst = values_.data() + base;

    switch (mode) {
        case assembly::AddMode::Add:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] += block_data[k];
            }
            break;
        case assembly::AddMode::Insert:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] = block_data[k];
            }
            break;
        case assembly::AddMode::Max:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] = std::max(dst[k], block_data[k]);
            }
            break;
        case assembly::AddMode::Min:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] = std::min(dst[k], block_data[k]);
            }
            break;
    }
}

int FsilsMatrix::fsilsDof() const noexcept
{
    return shared_ ? shared_->dof : 0;
}

void* FsilsMatrix::fsilsLhsPtr() noexcept
{
    return shared_ ? static_cast<void*>(&shared_->lhs) : nullptr;
}

const void* FsilsMatrix::fsilsLhsPtr() const noexcept
{
    return shared_ ? static_cast<const void*>(&shared_->lhs) : nullptr;
}

Real* FsilsMatrix::fsilsValuesPtr() noexcept
{
    return values_.data();
}

const Real* FsilsMatrix::fsilsValuesPtr() const noexcept
{
    return values_.data();
}

GlobalIndex FsilsMatrix::fsilsNnz() const noexcept
{
    return nnz_;
}

} // namespace backends
} // namespace FE
} // namespace svmp
