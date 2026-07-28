/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ParallelConstraints.h"

#include <algorithm>
#include <cmath>
#include <exception>
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
                : partition.isRelevant(dof);
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

    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

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

    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

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

    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

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
    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

    bool local_valid = true;
    std::exception_ptr local_validation_exception;
    try {
        // Check that all locally relevant constrained DOFs match the
        // canonical constraint.
        for (const auto& [dof, ranked] : canonical) {
            if (!partition_->isRelevant(dof)) {
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
