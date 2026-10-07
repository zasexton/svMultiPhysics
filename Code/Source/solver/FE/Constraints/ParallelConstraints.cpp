/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ParallelConstraints.h"

#include "Core/Logger.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>

#if FE_HAS_MPI
#include <cstdint>
#endif

namespace svmp {
namespace FE {
namespace constraints {

#if FE_HAS_MPI
namespace {

struct RankedConstraintLine {
    ConstraintLine line;
    int source_rank{-1};
    bool source_claims_ownership{false};
};

using CanonicalConstraintMap =
    std::unordered_map<GlobalIndex, RankedConstraintLine>;

void coordinateDistributedPhaseFailure(
    MPI_Comm comm,
    const std::exception_ptr& local_exception,
    const char* phase)
{
    const int local_ok = local_exception == nullptr ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (all_ok != 0) {
        return;
    }
    if (local_exception != nullptr) {
        std::rethrow_exception(local_exception);
    }
    CONSTRAINT_THROW(
        std::string(
            "ParallelConstraints: diagnostic="
            "distributed_parallel_constraint_phase_failure phase='") +
        phase +
        "' another communicator rank failed its local constraint phase");
}

void requireDistributedPartition(
    MPI_Comm comm,
    const dofs::DofPartition* partition)
{
    std::exception_ptr local_exception;
    try {
        if (partition == nullptr) {
            CONSTRAINT_THROW(
                "ParallelConstraints requires a DofPartition");
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm, local_exception, "partition_precondition");
}

ConstraintLine toConstraintLine(const AffineConstraints::ConstraintView& view) {
    ConstraintLine line;
    line.slave_dof = view.slave_dof;
    line.inhomogeneity = view.inhomogeneity;
    line.entries.assign(view.entries.begin(), view.entries.end());
    // Canonicalize for deterministic comparisons across ranks
    line.mergeEntries();
    return line;
}

bool equivalentConstraintLines(const ConstraintLine& a,
                               const ConstraintLine& b,
                               double tol) {
    if (a.slave_dof != b.slave_dof) return false;
    if (std::abs(a.inhomogeneity - b.inhomogeneity) > tol) return false;
    if (a.entries.size() != b.entries.size()) return false;
    for (std::size_t i = 0; i < a.entries.size(); ++i) {
        if (a.entries[i].master_dof != b.entries[i].master_dof) return false;
        if (std::abs(a.entries[i].weight - b.entries[i].weight) > tol) return false;
    }
    return true;
}

RankedConstraintLine chooseWinner(const RankedConstraintLine& a,
                                 const RankedConstraintLine& b,
                                 ParallelConstraintOptions::ConflictResolution strategy,
                                 double tol,
                                 bool& had_real_conflict) {
    had_real_conflict = !equivalentConstraintLines(a.line, b.line, tol);

    switch (strategy) {
        case ParallelConstraintOptions::ConflictResolution::OwnerWins: {
            if (a.source_claims_ownership != b.source_claims_ownership) {
                return a.source_claims_ownership ? a : b;
            }
            return (a.source_rank <= b.source_rank) ? a : b;
        }
        case ParallelConstraintOptions::ConflictResolution::SmallestRank:
            return (a.source_rank <= b.source_rank) ? a : b;
        case ParallelConstraintOptions::ConflictResolution::Error:
            if (!equivalentConstraintLines(a.line, b.line, tol)) {
                CONSTRAINT_THROW_DOF("Conflicting constraints from different ranks", a.line.slave_dof);
            }
            // Identical constraints: prefer owner if available, otherwise keep lowest rank
            if (a.source_claims_ownership != b.source_claims_ownership) {
                return a.source_claims_ownership ? a : b;
            }
            return (a.source_rank <= b.source_rank) ? a : b;
        default:
            return a;
    }
}

std::vector<char> packLocalConstraints(const AffineConstraints& constraints,
                                      const dofs::DofPartition& partition,
                                      MPI_Comm comm) {
    const auto constrained_dofs = constraints.getConstrainedDofs();
    if (constrained_dofs.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        CONSTRAINT_THROW(
            "ParallelConstraints: local constraint count exceeds the "
            "distributed wire range");
    }
    const std::int64_t n_lines = static_cast<std::int64_t>(constrained_dofs.size());

    int sz_i64 = 0;
    int sz_int = 0;
    int sz_double = 0;
    MPI_Pack_size(1, MPI_INT64_T, comm, &sz_i64);
    MPI_Pack_size(1, MPI_INT, comm, &sz_int);
    MPI_Pack_size(1, MPI_DOUBLE, comm, &sz_double);

    // Compute an upper bound for packed buffer size.
    const auto max_packed_size =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    std::size_t total = 0;
    const auto add_packed_bytes =
        [&](std::size_t count, int bytes_per_value) {
            if (bytes_per_value < 0) {
                CONSTRAINT_THROW(
                    "ParallelConstraints: MPI returned a negative packed "
                    "value size");
            }
            const auto bytes =
                static_cast<std::size_t>(bytes_per_value);
            if (bytes != 0u &&
                count > (max_packed_size - total) / bytes) {
                CONSTRAINT_THROW(
                    "ParallelConstraints: local constraint payload exceeds "
                    "the MPI count range");
            }
            total += count * bytes;
        };
    add_packed_bytes(1u, sz_i64); // n_lines
    for (GlobalIndex dof : constrained_dofs) {
        const auto view = constraints.getConstraint(dof);
        if (!view) continue;
        add_packed_bytes(2u, sz_i64);    // slave_dof, n_entries
        add_packed_bytes(1u, sz_int);    // owned flag
        add_packed_bytes(1u, sz_double); // inhomogeneity
        add_packed_bytes(view->entries.size(), sz_i64);
        add_packed_bytes(view->entries.size(), sz_double);
    }

    std::vector<char> buffer(total);
    int position = 0;

    MPI_Pack(&n_lines, 1, MPI_INT64_T,
             buffer.data(), static_cast<int>(buffer.size()), &position, comm);

    for (GlobalIndex dof : constrained_dofs) {
        const auto view = constraints.getConstraint(dof);
        if (!view) continue;

        ConstraintLine line = toConstraintLine(*view);
        if (line.entries.size() >
            static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            CONSTRAINT_THROW(
                "ParallelConstraints: local constraint entry count exceeds "
                "the distributed wire range");
        }
        const std::int64_t slave = static_cast<std::int64_t>(line.slave_dof);
        const int owned_flag = partition.isOwned(line.slave_dof) ? 1 : 0;
        const double inhom = line.inhomogeneity;
        const std::int64_t n_entries = static_cast<std::int64_t>(line.entries.size());

        MPI_Pack(&slave, 1, MPI_INT64_T,
                 buffer.data(), static_cast<int>(buffer.size()), &position, comm);
        MPI_Pack(&owned_flag, 1, MPI_INT,
                 buffer.data(), static_cast<int>(buffer.size()), &position, comm);
        MPI_Pack(&inhom, 1, MPI_DOUBLE,
                 buffer.data(), static_cast<int>(buffer.size()), &position, comm);
        MPI_Pack(&n_entries, 1, MPI_INT64_T,
                 buffer.data(), static_cast<int>(buffer.size()), &position, comm);

        for (const auto& entry : line.entries) {
            const std::int64_t master = static_cast<std::int64_t>(entry.master_dof);
            const double weight = entry.weight;
            MPI_Pack(&master, 1, MPI_INT64_T,
                     buffer.data(), static_cast<int>(buffer.size()), &position, comm);
            MPI_Pack(&weight, 1, MPI_DOUBLE,
                     buffer.data(), static_cast<int>(buffer.size()), &position, comm);
        }
    }

    buffer.resize(static_cast<std::size_t>(position));
    return buffer;
}

std::vector<RankedConstraintLine> unpackConstraintsForRank(std::span<const char> buffer,
                                                           int source_rank,
                                                           MPI_Comm comm) {
    int position = 0;
    std::int64_t n_lines = 0;
    MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position,
               &n_lines, 1, MPI_INT64_T, comm);

    std::vector<RankedConstraintLine> lines;
    lines.reserve(static_cast<std::size_t>(std::max<std::int64_t>(n_lines, 0)));

    for (std::int64_t i = 0; i < n_lines; ++i) {
        std::int64_t slave = -1;
        int owned_flag = 0;
        double inhom = 0.0;
        std::int64_t n_entries = 0;

        MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position,
                   &slave, 1, MPI_INT64_T, comm);
        MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position,
                   &owned_flag, 1, MPI_INT, comm);
        MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position,
                   &inhom, 1, MPI_DOUBLE, comm);
        MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position,
                   &n_entries, 1, MPI_INT64_T, comm);

        ConstraintLine line;
        line.slave_dof = static_cast<GlobalIndex>(slave);
        line.inhomogeneity = inhom;
        line.entries.reserve(static_cast<std::size_t>(std::max<std::int64_t>(n_entries, 0)));

        for (std::int64_t e = 0; e < n_entries; ++e) {
            std::int64_t master = -1;
            double weight = 0.0;
            MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position,
                       &master, 1, MPI_INT64_T, comm);
            MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position,
                       &weight, 1, MPI_DOUBLE, comm);
            line.entries.push_back({static_cast<GlobalIndex>(master), weight});
        }

        line.mergeEntries();
        lines.push_back({std::move(line), source_rank, owned_flag != 0});
    }

    return lines;
}

CanonicalConstraintMap
gatherAndResolveConstraints(MPI_Comm comm,
                            int world_size,
                            const dofs::DofPartition& partition,
                            const ParallelConstraintOptions& options,
                            const AffineConstraints& local_constraints,
                            ParallelConstraintStats& stats) {
    std::vector<char> send_buffer;
    int send_size = 0;
    std::vector<int> recv_sizes;
    std::exception_ptr local_pack_exception;
    try {
        send_buffer =
            packLocalConstraints(local_constraints, partition, comm);
        if (send_buffer.size() >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            CONSTRAINT_THROW(
                "ParallelConstraints: local constraint payload exceeds the "
                "MPI count range");
        }
        send_size = static_cast<int>(send_buffer.size());
        recv_sizes.assign(static_cast<std::size_t>(world_size), 0);
    } catch (...) {
        local_pack_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm, local_pack_exception, "pack_and_count_allocation");
    MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm);

    std::vector<int> displs;
    std::vector<char> recv_buffer;
    std::exception_ptr local_receive_exception;
    try {
        displs.assign(static_cast<std::size_t>(world_size), 0);
        int total = 0;
        for (int r = 0; r < world_size; ++r) {
            const int rank_size =
                recv_sizes[static_cast<std::size_t>(r)];
            if (rank_size < 0 ||
                rank_size > std::numeric_limits<int>::max() - total) {
                CONSTRAINT_THROW(
                    "ParallelConstraints: gathered constraint payload "
                    "exceeds the MPI displacement range");
            }
            displs[static_cast<std::size_t>(r)] = total;
            total += rank_size;
        }
        recv_buffer.resize(static_cast<std::size_t>(total));
    } catch (...) {
        local_receive_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm, local_receive_exception, "receive_layout_allocation");
    MPI_Allgatherv(send_buffer.data(), send_size, MPI_BYTE,
                   recv_buffer.data(), recv_sizes.data(), displs.data(), MPI_BYTE,
                   comm);

    CanonicalConstraintMap canonical;
    std::exception_ptr local_decode_exception;
    try {
        stats.n_messages_sent +=
            static_cast<GlobalIndex>(
                world_size > 0 ? world_size - 1 : 0);
        stats.n_messages_received +=
            static_cast<GlobalIndex>(
                world_size > 0 ? world_size - 1 : 0);

        canonical.reserve(static_cast<std::size_t>(
            local_constraints.getConstrainedDofs().size()));

        for (int r = 0; r < world_size; ++r) {
            const int sz = recv_sizes[static_cast<std::size_t>(r)];
            const int disp = displs[static_cast<std::size_t>(r)];
            if (sz <= 0) continue;

            const auto span =
                std::span<const char>(
                    recv_buffer.data() + disp,
                    static_cast<std::size_t>(sz));
            auto lines = unpackConstraintsForRank(span, r, comm);
            for (auto& ranked : lines) {
                const GlobalIndex dof = ranked.line.slave_dof;
                auto it = canonical.find(dof);
                if (it == canonical.end()) {
                    canonical.emplace(dof, std::move(ranked));
                    continue;
                }

                bool had_real_conflict = false;
                const auto winner =
                    chooseWinner(
                        it->second,
                        ranked,
                        options.conflict_resolution,
                        options.tolerance,
                        had_real_conflict);

                if (had_real_conflict &&
                    options.conflict_resolution !=
                        ParallelConstraintOptions::
                            ConflictResolution::Error) {
                    ++stats.n_conflicts_resolved;
                }

                it->second = winner;
            }
        }
    } catch (...) {
        local_decode_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm, local_decode_exception, "decode_and_resolve");

    return canonical;
}

bool envFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void appendLineWords(std::vector<std::int64_t>& words,
                     const ConstraintLine& line,
                     int source_rank,
                     bool claims_ownership)
{
    words.push_back(static_cast<std::int64_t>(line.slave_dof));
    words.push_back(static_cast<std::int64_t>(source_rank));
    words.push_back(claims_ownership ? 1 : 0);
    words.push_back(std::bit_cast<std::int64_t>(static_cast<double>(line.inhomogeneity)));
    words.push_back(static_cast<std::int64_t>(line.entries.size()));
    for (const auto& entry : line.entries) {
        words.push_back(static_cast<std::int64_t>(entry.master_dof));
        words.push_back(std::bit_cast<std::int64_t>(static_cast<double>(entry.weight)));
    }
}

RankedConstraintLine readLineWords(const std::vector<std::int64_t>& words,
                                   std::size_t& position,
                                   std::size_t end)
{
    if (end - position < 5u) {
        CONSTRAINT_THROW("ParallelConstraints: malformed routed constraint line header");
    }
    RankedConstraintLine ranked;
    ranked.line.slave_dof = static_cast<GlobalIndex>(words[position++]);
    ranked.source_rank = static_cast<int>(words[position++]);
    ranked.source_claims_ownership = words[position++] != 0;
    ranked.line.inhomogeneity = std::bit_cast<double>(words[position++]);
    const auto n_entries = words[position++];
    if (n_entries < 0 ||
        static_cast<std::uint64_t>(n_entries) * 2u >
            static_cast<std::uint64_t>(end - position)) {
        CONSTRAINT_THROW("ParallelConstraints: malformed routed constraint line entries");
    }
    ranked.line.entries.reserve(static_cast<std::size_t>(n_entries));
    for (std::int64_t e = 0; e < n_entries; ++e) {
        const auto master = static_cast<GlobalIndex>(words[position++]);
        const double weight = std::bit_cast<double>(words[position++]);
        ranked.line.entries.push_back({master, weight});
    }
    // As unpackConstraintsForRank: lines are merged on both ends of the wire.
    ranked.line.mergeEntries();
    return ranked;
}

// Sparse all-to-all of int64 words: counts by MPI_Alltoall, payload by
// MPI_Alltoallv.  Returns the received words and per-source displacements.
void alltoallWords(MPI_Comm comm,
                   int world_size,
                   const std::vector<std::vector<std::int64_t>>& send,
                   std::vector<std::int64_t>& recv,
                   std::vector<int>& recv_counts,
                   std::vector<int>& recv_displs)
{
    std::vector<int> send_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> send_displs(static_cast<std::size_t>(world_size), 0);
    std::size_t total_send = 0;
    for (int r = 0; r < world_size; ++r) {
        const auto n = send[static_cast<std::size_t>(r)].size();
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            total_send > static_cast<std::size_t>(std::numeric_limits<int>::max()) - n) {
            CONSTRAINT_THROW("ParallelConstraints: routed constraint payload exceeds the MPI count range");
        }
        send_displs[static_cast<std::size_t>(r)] = static_cast<int>(total_send);
        send_counts[static_cast<std::size_t>(r)] = static_cast<int>(n);
        total_send += n;
    }
    std::vector<std::int64_t> send_buffer;
    send_buffer.reserve(total_send);
    for (int r = 0; r < world_size; ++r) {
        const auto& part = send[static_cast<std::size_t>(r)];
        send_buffer.insert(send_buffer.end(), part.begin(), part.end());
    }
    recv_counts.assign(static_cast<std::size_t>(world_size), 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    recv_displs.assign(static_cast<std::size_t>(world_size), 0);
    std::size_t total_recv = 0;
    for (int r = 0; r < world_size; ++r) {
        const auto n = static_cast<std::size_t>(recv_counts[static_cast<std::size_t>(r)]);
        if (total_recv > static_cast<std::size_t>(std::numeric_limits<int>::max()) - n) {
            CONSTRAINT_THROW("ParallelConstraints: routed constraint payload exceeds the MPI displacement range");
        }
        recv_displs[static_cast<std::size_t>(r)] = static_cast<int>(total_recv);
        total_recv += n;
    }
    recv.assign(total_recv, 0);
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  recv.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T, comm);
}

// Owner-routed resolution of the canonical constraint lines (see
// ParallelConstraints::setDofOwnerFunction).  Returns the canonical lines of
// every locally relevant constrained slave (owned and ghost), or nullopt on
// every rank when a precondition fails on any rank.  Collective.
std::optional<CanonicalConstraintMap>
resolveConstraintsByOwner(MPI_Comm comm,
                          int world_size,
                          int my_rank,
                          const dofs::DofPartition& partition,
                          const std::function<int(GlobalIndex)>& owner_of,
                          const ParallelConstraintOptions& options,
                          const AffineConstraints& local_constraints,
                          ParallelConstraintStats& stats)
{
    // Preconditions: every local line has a locally relevant slave, and
    // every ghost DOF has a valid remote owner.
    int local_ok = 1;
    std::vector<std::vector<std::int64_t>> requests(static_cast<std::size_t>(world_size));
    std::vector<std::vector<std::int64_t>> routed_lines(static_cast<std::size_t>(world_size));
    // Own lines for owned slaves, folded at position my_rank.
    std::vector<RankedConstraintLine> own_owned_lines;
    std::exception_ptr local_exception;
    try {
        for (const GlobalIndex dof : partition.ghost()) {
            const int owner = owner_of ? owner_of(dof) : -1;
            if (owner < 0 || owner >= world_size || owner == my_rank) {
                local_ok = 0;
                break;
            }
            requests[static_cast<std::size_t>(owner)].push_back(static_cast<std::int64_t>(dof));
        }
        if (local_ok != 0) {
            for (const GlobalIndex dof : local_constraints.getConstrainedDofs()) {
                const auto view = local_constraints.getConstraint(dof);
                if (!view) {
                    continue;
                }
                if (!partition.isRelevant(dof)) {
                    local_ok = 0;
                    break;
                }
                ConstraintLine line = toConstraintLine(*view);
                if (partition.isOwned(dof)) {
                    // The all-gather merges a line when packing and again
                    // when unpacking; do the same for the owner's own line.
                    line.mergeEntries();
                    own_owned_lines.push_back({std::move(line), my_rank, true});
                    continue;
                }
                const int owner = owner_of(dof);
                if (owner < 0 || owner >= world_size || owner == my_rank) {
                    local_ok = 0;
                    break;
                }
                appendLineWords(routed_lines[static_cast<std::size_t>(owner)], line, my_rank, false);
            }
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(comm, local_exception, "routed_precondition");
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (all_ok == 0) {
        return std::nullopt;
    }

    // Phase A: ghost-DOF requests and lines go to the owners.
    std::vector<std::int64_t> recv_a;
    std::vector<int> counts_a;
    std::vector<int> displs_a;
    std::exception_ptr phase_a_exception;
    std::vector<std::vector<std::int64_t>> send_a(static_cast<std::size_t>(world_size));
    try {
        for (int r = 0; r < world_size; ++r) {
            auto& out = send_a[static_cast<std::size_t>(r)];
            const auto& req = requests[static_cast<std::size_t>(r)];
            const auto& lines = routed_lines[static_cast<std::size_t>(r)];
            if (req.empty() && lines.empty()) {
                continue;
            }
            out.reserve(1u + req.size() + lines.size());
            out.push_back(static_cast<std::int64_t>(req.size()));
            out.insert(out.end(), req.begin(), req.end());
            out.insert(out.end(), lines.begin(), lines.end());
        }
    } catch (...) {
        phase_a_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(comm, phase_a_exception, "routed_send_lines");
    alltoallWords(comm, world_size, send_a, recv_a, counts_a, displs_a);

    // Owner fold in ascending source rank, the order of the all-gather fold.
    // A request or line that reaches a rank not owning its DOF means the
    // owner function disagrees with the partition: fall back collectively.
    CanonicalConstraintMap canonical;
    std::vector<std::vector<GlobalIndex>> requested_by(static_cast<std::size_t>(world_size));
    int local_route_ok = 1;
    std::exception_ptr fold_exception;
    try {
        std::unordered_map<GlobalIndex, std::vector<RankedConstraintLine>> lines_by_slave;
        lines_by_slave.reserve(own_owned_lines.size());
        for (auto& ranked : own_owned_lines) {
            const auto slave = ranked.line.slave_dof;
            lines_by_slave[slave].push_back(std::move(ranked));
        }
        for (int r = 0; r < world_size; ++r) {
            std::size_t position = static_cast<std::size_t>(displs_a[static_cast<std::size_t>(r)]);
            const std::size_t end = position + static_cast<std::size_t>(counts_a[static_cast<std::size_t>(r)]);
            if (position == end) {
                continue;
            }
            const auto n_requests = recv_a[position++];
            if (n_requests < 0 ||
                static_cast<std::uint64_t>(n_requests) > static_cast<std::uint64_t>(end - position)) {
                CONSTRAINT_THROW("ParallelConstraints: malformed routed ghost request");
            }
            auto& req = requested_by[static_cast<std::size_t>(r)];
            req.reserve(static_cast<std::size_t>(n_requests));
            for (std::int64_t i = 0; i < n_requests; ++i) {
                const auto dof = static_cast<GlobalIndex>(recv_a[position++]);
                if (!partition.isOwned(dof)) {
                    local_route_ok = 0;
                }
                req.push_back(dof);
            }
            while (position < end) {
                auto ranked = readLineWords(recv_a, position, end);
                if (ranked.source_rank != r) {
                    CONSTRAINT_THROW_DOF("ParallelConstraints: routed constraint line has a wrong source rank",
                                         ranked.line.slave_dof);
                }
                if (!partition.isOwned(ranked.line.slave_dof)) {
                    local_route_ok = 0;
                }
                const auto slave = ranked.line.slave_dof;
                lines_by_slave[slave].push_back(std::move(ranked));
            }
        }
        canonical.reserve(lines_by_slave.size());
        for (auto& [slave, lines] : lines_by_slave) {
            std::stable_sort(lines.begin(), lines.end(),
                             [](const RankedConstraintLine& a, const RankedConstraintLine& b) {
                                 return a.source_rank < b.source_rank;
                             });
            RankedConstraintLine current = std::move(lines.front());
            for (std::size_t k = 1; k < lines.size(); ++k) {
                bool had_real_conflict = false;
                current = chooseWinner(current,
                                       lines[k],
                                       options.conflict_resolution,
                                       options.tolerance,
                                       had_real_conflict);
                if (had_real_conflict &&
                    options.conflict_resolution !=
                        ParallelConstraintOptions::ConflictResolution::Error) {
                    ++stats.n_conflicts_resolved;
                }
            }
            canonical.emplace(slave, std::move(current));
        }
    } catch (...) {
        fold_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(comm, fold_exception, "routed_owner_fold");
    int all_route_ok = 0;
    MPI_Allreduce(&local_route_ok, &all_route_ok, 1, MPI_INT, MPI_MIN, comm);
    if (all_route_ok == 0) {
        return std::nullopt;
    }

    // Phase B: owners return the canonical lines of the requested ghosts.
    std::vector<std::vector<std::int64_t>> send_b(static_cast<std::size_t>(world_size));
    std::exception_ptr phase_b_exception;
    try {
        for (int r = 0; r < world_size; ++r) {
            for (const auto dof : requested_by[static_cast<std::size_t>(r)]) {
                const auto it = canonical.find(dof);
                if (it == canonical.end()) {
                    continue;
                }
                appendLineWords(send_b[static_cast<std::size_t>(r)],
                                it->second.line,
                                it->second.source_rank,
                                it->second.source_claims_ownership);
            }
        }
    } catch (...) {
        phase_b_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(comm, phase_b_exception, "routed_reply_lines");
    std::vector<std::int64_t> recv_b;
    std::vector<int> counts_b;
    std::vector<int> displs_b;
    alltoallWords(comm, world_size, send_b, recv_b, counts_b, displs_b);

    std::exception_ptr decode_exception;
    try {
        for (int r = 0; r < world_size; ++r) {
            std::size_t position = static_cast<std::size_t>(displs_b[static_cast<std::size_t>(r)]);
            const std::size_t end = position + static_cast<std::size_t>(counts_b[static_cast<std::size_t>(r)]);
            while (position < end) {
                auto ranked = readLineWords(recv_b, position, end);
                const auto slave = ranked.line.slave_dof;
                if (!partition.isGhost(slave) ||
                    !canonical.emplace(slave, std::move(ranked)).second) {
                    CONSTRAINT_THROW_DOF("ParallelConstraints: unexpected routed canonical line", slave);
                }
            }
        }
        stats.n_messages_sent += static_cast<GlobalIndex>(
            std::count_if(send_a.begin(), send_a.end(), [](const auto& v) { return !v.empty(); }) +
            std::count_if(send_b.begin(), send_b.end(), [](const auto& v) { return !v.empty(); }));
        stats.n_messages_received += static_cast<GlobalIndex>(
            std::count_if(counts_a.begin(), counts_a.end(), [](int n) { return n > 0; }) +
            std::count_if(counts_b.begin(), counts_b.end(), [](int n) { return n > 0; }));
    } catch (...) {
        decode_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(comm, decode_exception, "routed_decode");
    return canonical;
}

bool sameRankedLineBits(const RankedConstraintLine& a, const RankedConstraintLine& b)
{
    if (a.source_rank != b.source_rank ||
        a.source_claims_ownership != b.source_claims_ownership ||
        a.line.slave_dof != b.line.slave_dof ||
        std::bit_cast<std::uint64_t>(static_cast<double>(a.line.inhomogeneity)) !=
            std::bit_cast<std::uint64_t>(static_cast<double>(b.line.inhomogeneity)) ||
        a.line.entries.size() != b.line.entries.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.line.entries.size(); ++i) {
        if (a.line.entries[i].master_dof != b.line.entries[i].master_dof ||
            std::bit_cast<std::uint64_t>(static_cast<double>(a.line.entries[i].weight)) !=
                std::bit_cast<std::uint64_t>(static_cast<double>(b.line.entries[i].weight))) {
            return false;
        }
    }
    return true;
}

// Canonical lines for the locally relevant slaves: owner-routed when an
// owner function is available (see setDofOwnerFunction), otherwise the
// all-gather.  Collective.
CanonicalConstraintMap
resolveCanonicalConstraints(MPI_Comm comm,
                            int world_size,
                            int my_rank,
                            const dofs::DofPartition& partition,
                            const std::function<int(GlobalIndex)>& owner_of,
                            const ParallelConstraintOptions& options,
                            const AffineConstraints& local_constraints,
                            ParallelConstraintStats& stats)
{
    static const bool force_allgather = envFlagEnabled("SVMP_PARALLEL_CONSTRAINTS_ALLGATHER");
    static const bool check = envFlagEnabled("SVMP_PARALLEL_CONSTRAINTS_CHECK");
    if (!owner_of || force_allgather) {
        return gatherAndResolveConstraints(comm, world_size, partition, options,
                                           local_constraints, stats);
    }
    auto routed = resolveConstraintsByOwner(comm, world_size, my_rank, partition, owner_of,
                                            options, local_constraints, stats);
    if (!routed) {
        // Collective decision: every rank takes this branch together.
        static bool reported = false;
        if (!reported) {
            reported = true;
            FE_LOG_INFO(
                "ParallelConstraints: diagnostic=owner_routed_constraints_unavailable "
                "a line has a non-relevant slave or the owner function disagrees with the "
                "partition; using the all-gather");
        }
        return gatherAndResolveConstraints(comm, world_size, partition, options,
                                           local_constraints, stats);
    }
    if (check) {
        ParallelConstraintStats reference_stats;
        const auto reference = gatherAndResolveConstraints(comm, world_size, partition, options,
                                                           local_constraints, reference_stats);
        std::exception_ptr local_exception;
        try {
            std::size_t relevant_reference = 0;
            for (const auto& [dof, ranked] : reference) {
                if (!partition.isRelevant(dof)) {
                    continue;
                }
                ++relevant_reference;
                const auto it = routed->find(dof);
                if (it == routed->end() || !sameRankedLineBits(it->second, ranked)) {
                    CONSTRAINT_THROW_DOF(
                        "ParallelConstraints: owner-routed canonical line differs from the all-gather",
                        dof);
                }
            }
            if (relevant_reference != routed->size()) {
                CONSTRAINT_THROW(
                    "ParallelConstraints: owner-routed canonical lines cover a different slave set "
                    "than the all-gather");
            }
        } catch (...) {
            local_exception = std::current_exception();
        }
        coordinateDistributedPhaseFailure(comm, local_exception, "routed_self_check");
    }
    return std::move(*routed);
}

/// A ghost copy of a constraint line is only representable on a rank that
/// also carries every master; lines whose masters lie outside the local halo
/// stay with the ranks that assemble with them (see
/// SmallCutAggregationConstraint, which proves that).
[[nodiscard]] bool mastersRelevant(const ConstraintLine& line,
                                   const dofs::DofPartition& partition)
{
    return std::all_of(line.entries.begin(), line.entries.end(),
                       [&](const auto& entry) {
                           return partition.isRelevant(entry.master_dof);
                       });
}

enum class LocalConstraintSelection {
    Owned,
    Relevant
};

AffineConstraints rebuildLocalConstraints(
    const CanonicalConstraintMap& canonical,
    const dofs::DofPartition& partition,
    const AffineConstraintsOptions& options,
    LocalConstraintSelection selection,
    ParallelConstraintStats& stats)
{
    AffineConstraints updated(options);
    stats.n_local_constraints = 0;
    stats.n_ghost_constraints = 0;
    for (const auto& [dof, ranked] : canonical) {
        const bool keep =
            selection == LocalConstraintSelection::Owned
                ? partition.isOwned(dof)
                : partition.isRelevant(dof) &&
                      (partition.isOwned(dof) ||
                       mastersRelevant(ranked.line, partition));
        if (!keep) {
            continue;
        }
        updated.addConstraintLine(ranked.line);
        if (partition.isOwned(dof)) {
            ++stats.n_local_constraints;
        } else if (partition.isGhost(dof)) {
            ++stats.n_ghost_constraints;
        }
    }
    return updated;
}

} // namespace
#endif // FE_HAS_MPI

// ============================================================================
// Construction
// ============================================================================

#if FE_HAS_MPI
ParallelConstraints::ParallelConstraints(MPI_Comm comm,
                                          const dofs::DofPartition& partition)
    : comm_(comm), partition_(&partition) {
    MPI_Comm_rank(comm, &my_rank_);
    MPI_Comm_size(comm, &world_size_);
}
#endif

ParallelConstraints::ParallelConstraints()
    : partition_(nullptr), my_rank_(0), world_size_(1) {}

ParallelConstraints::ParallelConstraints(const dofs::DofPartition& partition)
    : partition_(&partition), my_rank_(0), world_size_(1) {}

ParallelConstraints::~ParallelConstraints() = default;

ParallelConstraints::ParallelConstraints(ParallelConstraints&& other) noexcept = default;

ParallelConstraints& ParallelConstraints::operator=(ParallelConstraints&& other) noexcept = default;

// ============================================================================
// Main operations
// ============================================================================

ParallelConstraintStats ParallelConstraints::makeConsistent(
    AffineConstraints& constraints)
{
    ParallelConstraintStats stats;

    if (world_size_ == 1) {
        // Serial mode - nothing to do
        stats.n_local_constraints = static_cast<GlobalIndex>(constraints.getConstrainedDofs().size());
        last_stats_ = stats;
        return stats;
    }

#if FE_HAS_MPI
    // In parallel:
    // 1. Each rank identifies shared DOFs that have constraints
    // 2. Exchange constraints for shared DOFs
    // 3. Resolve conflicts using configured strategy

    requireDistributedPartition(comm_, partition_);

    auto canonical = resolveCanonicalConstraints(comm_, world_size_, my_rank_, *partition_, dof_owner_, options_, constraints, stats);

    std::optional<AffineConstraints> updated;
    std::exception_ptr local_rebuild_exception;
    try {
        updated.emplace(rebuildLocalConstraints(
            canonical,
            *partition_,
            constraints.getOptions(),
            LocalConstraintSelection::Owned,
            stats));
    } catch (...) {
        local_rebuild_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm_, local_rebuild_exception, "make_consistent_local_rebuild");
    constraints = std::move(*updated);
    last_stats_ = stats;
#endif

    return stats;
}

ParallelConstraintStats ParallelConstraints::importGhostConstraints(
    AffineConstraints& constraints)
{
    ParallelConstraintStats stats;
    static_cast<void>(constraints);

    if (world_size_ == 1) {
        // Serial mode - nothing to do
        last_stats_ = stats;
        return stats;
    }

#if FE_HAS_MPI
    requireDistributedPartition(comm_, partition_);

    auto canonical = resolveCanonicalConstraints(comm_, world_size_, my_rank_, *partition_, dof_owner_, options_, constraints, stats);

    std::optional<AffineConstraints> updated;
    std::exception_ptr local_rebuild_exception;
    try {
        updated.emplace(rebuildLocalConstraints(
            canonical,
            *partition_,
            constraints.getOptions(),
            LocalConstraintSelection::Relevant,
            stats));
    } catch (...) {
        local_rebuild_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm_, local_rebuild_exception, "import_ghost_local_rebuild");
    constraints = std::move(*updated);
    last_stats_ = stats;
#endif

    return stats;
}

ParallelConstraintStats ParallelConstraints::synchronize(AffineConstraints& constraints) {
    ParallelConstraintStats stats;

    if (world_size_ == 1) {
        stats.n_local_constraints = static_cast<GlobalIndex>(constraints.getConstrainedDofs().size());
        last_stats_ = stats;
        return stats;
    }

#if FE_HAS_MPI
    requireDistributedPartition(comm_, partition_);

    auto canonical = resolveCanonicalConstraints(comm_, world_size_, my_rank_, *partition_, dof_owner_, options_, constraints, stats);

    std::optional<AffineConstraints> updated;
    std::exception_ptr local_rebuild_exception;
    try {
        updated.emplace(rebuildLocalConstraints(
            canonical,
            *partition_,
            constraints.getOptions(),
            LocalConstraintSelection::Relevant,
            stats));
    } catch (...) {
        local_rebuild_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm_, local_rebuild_exception, "synchronize_local_rebuild");
    constraints = std::move(*updated);
    last_stats_ = stats;
#endif

    return stats;
}

std::vector<ConstraintLine> ParallelConstraints::exportConstraints(
    const AffineConstraints& constraints,
    std::span<const GlobalIndex> requested_dofs) const
{
    std::vector<ConstraintLine> result;
    result.reserve(requested_dofs.size());

    for (GlobalIndex dof : requested_dofs) {
        auto constraint = constraints.getConstraint(dof);
        if (constraint) {
            ConstraintLine line;
            line.slave_dof = constraint->slave_dof;
            line.inhomogeneity = constraint->inhomogeneity;
            for (const auto& entry : constraint->entries) {
                line.entries.push_back(entry);
            }
            result.push_back(std::move(line));
        }
    }

    return result;
}

// ============================================================================
// Validation
// ============================================================================

bool ParallelConstraints::validateConsistency(
    const AffineConstraints& constraints) const
{
    static_cast<void>(constraints);
    if (world_size_ == 1) {
        return true;  // Always consistent in serial
    }

#if FE_HAS_MPI
    requireDistributedPartition(comm_, partition_);

    ParallelConstraintStats stats;
    auto canonical = resolveCanonicalConstraints(comm_, world_size_, my_rank_, *partition_, dof_owner_, options_, constraints, stats);

    bool local_valid = true;
    std::exception_ptr local_validation_exception;
    try {
        // Check that all locally relevant constrained DOFs match the
        // canonical constraint.
        for (const auto& [dof, ranked] : canonical) {
            if (!partition_->isRelevant(dof)) {
                continue;
            }
            if (!partition_->isOwned(dof) &&
                !mastersRelevant(ranked.line, *partition_)) {
                continue;
            }

            const auto local = constraints.getConstraint(dof);
            if (!local) {
                local_valid = false;
                break;
            }

            ConstraintLine local_line = toConstraintLine(*local);
            if (!equivalentConstraintLines(
                    local_line, ranked.line, options_.tolerance)) {
                local_valid = false;
                break;
            }
        }
    } catch (...) {
        local_validation_exception = std::current_exception();
    }
    coordinateDistributedPhaseFailure(
        comm_, local_validation_exception, "validate_local_comparison");

    const int local_valid_int = local_valid ? 1 : 0;
    int all_valid = 0;
    MPI_Allreduce(
        &local_valid_int, &all_valid, 1, MPI_INT, MPI_MIN, comm_);
    return all_valid != 0;
#else
    return true;
#endif
}

// ============================================================================
// Internal implementation
// ============================================================================

ConstraintLine ParallelConstraints::resolveConflict(
    const ConstraintLine& local,
    const ConstraintLine& remote,
    int remote_rank) const
{
    switch (options_.conflict_resolution) {
        case ParallelConstraintOptions::ConflictResolution::OwnerWins:
            // If we own the DOF, keep local; otherwise use remote
            if (partition_ && partition_->isOwned(local.slave_dof)) {
                return local;
            }
            return remote;

        case ParallelConstraintOptions::ConflictResolution::SmallestRank:
            // Deterministic: smallest rank wins
            if (my_rank_ <= remote_rank) {
                return local;
            }
            return remote;

        case ParallelConstraintOptions::ConflictResolution::Error:
            // Check if constraints are equivalent
            if (local.entries.size() != remote.entries.size() ||
                std::abs(local.inhomogeneity - remote.inhomogeneity) > options_.tolerance) {
                CONSTRAINT_THROW_DOF("Conflicting constraints from different ranks",
                                     local.slave_dof);
            }
            // Check entries match
            for (std::size_t i = 0; i < local.entries.size(); ++i) {
                if (local.entries[i].master_dof != remote.entries[i].master_dof ||
                    std::abs(local.entries[i].weight - remote.entries[i].weight) > options_.tolerance) {
                    CONSTRAINT_THROW_DOF("Conflicting constraints from different ranks",
                                         local.slave_dof);
                }
            }
            return local;  // They match

        default:
            return local;
    }
}

std::vector<int> ParallelConstraints::findNeighborRanks() const {
    // In a full implementation, this would determine which ranks
    // share DOFs with this rank (based on ghost DOF ownership)
    return {};
}

} // namespace constraints
} // namespace FE
} // namespace svmp
