/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ParallelConstraints.h"

#include <algorithm>
#include <cmath>
#include <numeric>
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
    const std::int64_t n_lines = static_cast<std::int64_t>(constrained_dofs.size());

    int sz_i64 = 0;
    int sz_int = 0;
    int sz_double = 0;
    MPI_Pack_size(1, MPI_INT64_T, comm, &sz_i64);
    MPI_Pack_size(1, MPI_INT, comm, &sz_int);
    MPI_Pack_size(1, MPI_DOUBLE, comm, &sz_double);

    // Compute an upper bound for packed buffer size.
    int total = sz_i64; // n_lines
    for (GlobalIndex dof : constrained_dofs) {
        const auto view = constraints.getConstraint(dof);
        if (!view) continue;
        const std::int64_t n_entries = static_cast<std::int64_t>(view->entries.size());
        total += sz_i64;     // slave_dof
        total += sz_int;     // owned flag
        total += sz_double;  // inhomogeneity
        total += sz_i64;     // n_entries
        total += static_cast<int>(n_entries) * (sz_i64 + sz_double);
    }

    std::vector<char> buffer(static_cast<std::size_t>(total));
    int position = 0;

    MPI_Pack(&n_lines, 1, MPI_INT64_T,
             buffer.data(), static_cast<int>(buffer.size()), &position, comm);

    for (GlobalIndex dof : constrained_dofs) {
        const auto view = constraints.getConstraint(dof);
        if (!view) continue;

        ConstraintLine line = toConstraintLine(*view);
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

std::unordered_map<GlobalIndex, RankedConstraintLine>
gatherAndResolveConstraints(MPI_Comm comm,
                            int world_size,
                            const dofs::DofPartition& partition,
                            const ParallelConstraintOptions& options,
                            const AffineConstraints& local_constraints,
                            ParallelConstraintStats& stats) {
    const auto send_buffer = packLocalConstraints(local_constraints, partition, comm);
    const int send_size = static_cast<int>(send_buffer.size());

    std::vector<int> recv_sizes(static_cast<std::size_t>(world_size), 0);
    MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<std::size_t>(world_size), 0);
    int total = 0;
    for (int r = 0; r < world_size; ++r) {
        displs[static_cast<std::size_t>(r)] = total;
        total += recv_sizes[static_cast<std::size_t>(r)];
    }

    std::vector<char> recv_buffer(static_cast<std::size_t>(total));
    MPI_Allgatherv(send_buffer.data(), send_size, MPI_BYTE,
                   recv_buffer.data(), recv_sizes.data(), displs.data(), MPI_BYTE,
                   comm);

    stats.n_messages_sent += static_cast<GlobalIndex>(world_size > 0 ? world_size - 1 : 0);
    stats.n_messages_received += static_cast<GlobalIndex>(world_size > 0 ? world_size - 1 : 0);

    std::unordered_map<GlobalIndex, RankedConstraintLine> canonical;
    canonical.reserve(static_cast<std::size_t>(local_constraints.getConstrainedDofs().size()));

    for (int r = 0; r < world_size; ++r) {
        const int sz = recv_sizes[static_cast<std::size_t>(r)];
        const int disp = displs[static_cast<std::size_t>(r)];
        if (sz <= 0) continue;

        const auto span = std::span<const char>(recv_buffer.data() + disp,
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
            const auto winner = chooseWinner(it->second, ranked,
                                             options.conflict_resolution,
                                             options.tolerance,
                                             had_real_conflict);

            if (had_real_conflict &&
                options.conflict_resolution != ParallelConstraintOptions::ConflictResolution::Error) {
                ++stats.n_conflicts_resolved;
            }

            it->second = winner;
        }
    }

    return canonical;
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

    if (!partition_) {
        CONSTRAINT_THROW("ParallelConstraints requires a DofPartition");
    }

    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

    // Keep only locally owned constraints after consistency resolution
    AffineConstraints updated(constraints.getOptions());
    for (auto& [dof, ranked] : canonical) {
        if (partition_->isOwned(dof)) {
            updated.addConstraintLine(ranked.line);
        }
    }
    constraints = std::move(updated);

    stats.n_local_constraints = static_cast<GlobalIndex>(constraints.getConstrainedDofs().size());
    last_stats_ = stats;
#endif

    return stats;
}

ParallelConstraintStats ParallelConstraints::importGhostConstraints(
    AffineConstraints& constraints)
{
    ParallelConstraintStats stats;

    if (world_size_ == 1) {
        // Serial mode - nothing to do
        last_stats_ = stats;
        return stats;
    }

#if FE_HAS_MPI
    if (!partition_) {
        CONSTRAINT_THROW("ParallelConstraints requires a DofPartition");
    }

    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

    // Rebuild constraints for all locally relevant DOFs (owned + ghosts)
    AffineConstraints updated(constraints.getOptions());
    for (auto& [dof, ranked] : canonical) {
        if (partition_->isRelevant(dof)) {
            updated.addConstraintLine(ranked.line);
        }
    }
    constraints = std::move(updated);

    for (GlobalIndex dof : constraints.getConstrainedDofs()) {
        if (partition_->isOwned(dof)) {
            ++stats.n_local_constraints;
        } else if (partition_->isGhost(dof)) {
            ++stats.n_ghost_constraints;
        }
    }

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
    if (!partition_) {
        CONSTRAINT_THROW("ParallelConstraints requires a DofPartition");
    }

    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

    // Rebuild constraints for all locally relevant DOFs (owned + ghosts)
    AffineConstraints updated(constraints.getOptions());
    for (auto& [dof, ranked] : canonical) {
        if (partition_->isRelevant(dof)) {
            updated.addConstraintLine(ranked.line);
        }
    }
    constraints = std::move(updated);

    for (GlobalIndex dof : constraints.getConstrainedDofs()) {
        if (partition_->isOwned(dof)) {
            ++stats.n_local_constraints;
        } else if (partition_->isGhost(dof)) {
            ++stats.n_ghost_constraints;
        }
    }

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
    if (world_size_ == 1) {
        return true;  // Always consistent in serial
    }

#if FE_HAS_MPI
    if (!partition_) {
        CONSTRAINT_THROW("ParallelConstraints requires a DofPartition");
    }

    ParallelConstraintStats stats;
    auto canonical = gatherAndResolveConstraints(comm_, world_size_, *partition_, options_, constraints, stats);

    // Check that all locally relevant constrained DOFs match the canonical constraint
    for (const auto& [dof, ranked] : canonical) {
        if (!partition_->isRelevant(dof)) {
            continue;
        }

        const auto local = constraints.getConstraint(dof);
        if (!local) {
            return false;
        }

        ConstraintLine local_line = toConstraintLine(*local);
        if (!equivalentConstraintLines(local_line, ranked.line, options_.tolerance)) {
            return false;
        }
    }

    return true;
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
