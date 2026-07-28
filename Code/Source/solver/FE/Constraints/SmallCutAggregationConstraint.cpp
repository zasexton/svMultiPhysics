/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/SmallCutAggregationConstraint.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Constraints/AffineConstraints.h"
#include "Core/Logger.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/FrameGeometry.h"
#include "Geometry/GeometryFrameUtils.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

using math::Vector;

struct CellClass {
    bool full_active{false};
    bool cut{false};
};

enum class AggregationDeclarationState : std::int64_t {
    NonCandidate = 0,
    Rooted = 1,
    NoRootIsland = 2,
    SupportedWithoutForeignRoot = 3,
};

struct AggregationDeclaration {
    AggregationDeclarationState state{AggregationDeclarationState::NonCandidate};
    std::size_t root_distance{std::numeric_limits<std::size_t>::max()};
    GlobalIndex root_cell_gid{INVALID_GLOBAL_INDEX};
    std::vector<GlobalIndex> component_dofs{};
    std::vector<GlobalIndex> root_master_dofs{};
    std::vector<ConstraintLine> lines{};
};

[[nodiscard]] detail::SmallCutAggregationPhysicalRootKey physicalRootKey(
    const AggregationDeclaration& declaration) noexcept
{
    return detail::SmallCutAggregationPhysicalRootKey{
        declaration.root_distance,
        declaration.root_cell_gid,
    };
}

using LocalAggregationDeclarationMap =
    std::unordered_map<GlobalIndex, std::vector<AggregationDeclaration>>;

struct LocalCandidateSupport {
    GlobalIndex vertex{-1};
    GlobalIndex declaration_dof{-1};
    std::vector<GlobalIndex> component_dofs{};
    std::array<Real, 3> coordinates{};
    bool touches_cut{false};
    bool touches_full_active{false};
    bool excluded{false};
};

struct GlobalCandidateSupport {
    std::vector<GlobalIndex> component_dofs{};
    std::array<Real, 3> coordinates{};
    bool has_full_support{false};
    bool globally_rooted{false};
};

using CellKey = std::vector<GlobalIndex>;

struct GlobalRootCandidate {
    CellKey key{};
    std::size_t distance{std::numeric_limits<std::size_t>::max()};
};

struct RootedLineValidation {
    bool valid{false};
    const char* reason{"unknown"};
};

struct AggregationLineGuardValidation {
    bool valid{false};
    const char* reason{"unknown"};
    Real maximum_absolute_coefficient{0.0};
    Real row_l1_norm{0.0};
};

// Aggregation is a polynomial extension, so it must reproduce constants.
// Validate this after duplicate masters have been merged: validating before
// merge misses cancellation to an empty line and can silently turn a rooted
// aggregate into a homogeneous pin.  The relative term permits extrapolation
// lines with large positive/negative weights while retaining a tight absolute
// check near the usual convex case.
[[nodiscard]] RootedLineValidation normalizeAndValidateRootedLine(
    ConstraintLine& line)
{
    line.mergeEntries();
    if (line.slave_dof < 0 || !std::isfinite(line.inhomogeneity)) {
        return {false, "invalid_slave_or_inhomogeneity"};
    }
    if (line.entries.empty()) {
        return {false, "empty_after_merge"};
    }

    long double weight_sum = 0.0L;
    long double weight_l1 = 0.0L;
    GlobalIndex previous_master = -1;
    for (const auto& entry : line.entries) {
        if (entry.master_dof < 0 || entry.master_dof == line.slave_dof) {
            return {false, "invalid_or_self_master"};
        }
        if (!std::isfinite(entry.weight)) {
            return {false, "non_finite_weight"};
        }
        if (previous_master >= entry.master_dof) {
            return {false, "duplicate_or_unsorted_master"};
        }
        previous_master = entry.master_dof;
        weight_sum += static_cast<long double>(entry.weight);
        weight_l1 += std::abs(static_cast<long double>(entry.weight));
    }
    const long double scale = std::max(1.0L, weight_l1);
    const long double tolerance =
        1.0e-10L * scale +
        64.0L * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(weight_sum - 1.0L) > tolerance) {
        return {false, "partition_of_unity"};
    }
    return {true, "valid"};
}

[[nodiscard]] AggregationLineGuardValidation validateAggregationLineGuards(
    const ConstraintLine& line,
    const SmallCutAggregationGuardOptions& guards)
{
    AggregationLineGuardValidation result;
    result.valid = true;
    result.reason = "valid";
    for (const auto& entry : line.entries) {
        const auto magnitude = std::abs(static_cast<Real>(entry.weight));
        result.maximum_absolute_coefficient =
            std::max(result.maximum_absolute_coefficient, magnitude);
        result.row_l1_norm += magnitude;
    }
    const auto tolerance =
        Real{128.0} * std::numeric_limits<Real>::epsilon() *
        std::max({Real{1.0},
                  guards.maximum_absolute_coefficient,
                  guards.maximum_row_l1_norm});
    if (result.maximum_absolute_coefficient >
        guards.maximum_absolute_coefficient + tolerance) {
        result.valid = false;
        result.reason = "maximum_absolute_coefficient";
    } else if (result.row_l1_norm >
               guards.maximum_row_l1_norm + tolerance) {
        result.valid = false;
        result.reason = "maximum_row_l1_norm";
    }
    return result;
}

// Classify a public/mesh-local reference node against the affine hull of a
// linear face's corners.  A rank-revealing Gram-Schmidt basis handles general
// planes (including x+y=1 wedge faces and sloping pyramid faces), unlike the
// former constant-coordinate/coordinate-sum heuristic.
[[nodiscard]] bool referenceNodeOnFaceAffineHull(
    ElementType cell_type,
    std::span<const LocalIndex> face_corners,
    std::size_t local_node,
    int cell_dimension)
{
    if (face_corners.empty() || cell_dimension < 1 || cell_dimension > 3) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: invalid reference face topology");
    }
    const auto origin = basis::ReferenceNodeLayout::get_node_coords(
        cell_type, static_cast<std::size_t>(face_corners.front()));
    std::array<Vector<Real, 3>, 2> tangents{};
    const std::size_t expected_rank =
        static_cast<std::size_t>(cell_dimension - 1);
    std::size_t rank = 0u;
    Real face_scale = Real{1};
    constexpr Real rank_tolerance = Real{1e-12};
    for (std::size_t corner = 1;
         corner < face_corners.size() && rank < expected_rank;
         ++corner) {
        const auto point = basis::ReferenceNodeLayout::get_node_coords(
            cell_type, static_cast<std::size_t>(face_corners[corner]));
        Vector<Real, 3> residual{};
        Real original_norm_sq = Real{0};
        for (int d = 0; d < cell_dimension; ++d) {
            residual[d] = point[d] - origin[d];
            original_norm_sq += residual[d] * residual[d];
        }
        face_scale = std::max(face_scale, std::sqrt(original_norm_sq));
        for (std::size_t basis_index = 0; basis_index < rank; ++basis_index) {
            Real projection = Real{0};
            for (int d = 0; d < cell_dimension; ++d) {
                projection += residual[d] * tangents[basis_index][d];
            }
            for (int d = 0; d < cell_dimension; ++d) {
                residual[d] -= projection * tangents[basis_index][d];
            }
        }
        Real residual_norm_sq = Real{0};
        for (int d = 0; d < cell_dimension; ++d) {
            residual_norm_sq += residual[d] * residual[d];
        }
        const Real residual_norm = std::sqrt(residual_norm_sq);
        if (residual_norm <= rank_tolerance * face_scale) {
            continue;
        }
        for (int d = 0; d < cell_dimension; ++d) {
            tangents[rank][d] = residual[d] / residual_norm;
        }
        ++rank;
    }
    if (rank != expected_rank) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: degenerate reference face affine hull");
    }

    const auto point =
        basis::ReferenceNodeLayout::get_node_coords(cell_type, local_node);
    Vector<Real, 3> residual{};
    Real point_norm_sq = Real{0};
    for (int d = 0; d < cell_dimension; ++d) {
        residual[d] = point[d] - origin[d];
        point_norm_sq += residual[d] * residual[d];
    }
    for (std::size_t basis_index = 0; basis_index < rank; ++basis_index) {
        Real projection = Real{0};
        for (int d = 0; d < cell_dimension; ++d) {
            projection += residual[d] * tangents[basis_index][d];
        }
        for (int d = 0; d < cell_dimension; ++d) {
            residual[d] -= projection * tangents[basis_index][d];
        }
    }
    Real residual_norm_sq = Real{0};
    for (int d = 0; d < cell_dimension; ++d) {
        residual_norm_sq += residual[d] * residual[d];
    }
    const Real classification_scale =
        std::max({Real{1}, face_scale, std::sqrt(point_norm_sq)});
    return std::sqrt(residual_norm_sq) <= Real{1e-10} * classification_scale;
}

struct AggregationRuntimeOptions {
    bool slave_all_cut{false};
    bool linear_extension{false};
    bool allow_unaggregated{false};
    std::size_t max_lines{std::numeric_limits<std::size_t>::max()};
    bool flags_valid{true};
    bool max_lines_valid{true};
};

[[nodiscard]] bool strictEnvironmentFlag(const char* name, bool& valid)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0' || std::string_view(value) == "0") {
        return false;
    }
    if (std::string_view(value) == "1") {
        return true;
    }
    valid = false;
    return false;
}

[[nodiscard]] AggregationRuntimeOptions readAggregationRuntimeOptions()
{
    AggregationRuntimeOptions options;
    options.slave_all_cut =
        strictEnvironmentFlag("SVMP_AGGREGATION_SLAVE_ALL_CUT",
                              options.flags_valid);
    options.linear_extension =
        strictEnvironmentFlag("SVMP_AGGREGATION_LINEAR_EXTENSION",
                              options.flags_valid);
    options.allow_unaggregated =
        strictEnvironmentFlag("SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
                              options.flags_valid);

    const char* max_lines = std::getenv("SVMP_AGGREGATION_MAX_LINES");
    if (max_lines == nullptr || max_lines[0] == '\0') {
        return options;
    }
    std::uint64_t parsed = 0u;
    const auto* begin = max_lines;
    const auto* end = max_lines + std::char_traits<char>::length(max_lines);
    const auto parse = std::from_chars(begin, end, parsed, 10);
    if (parse.ec != std::errc{} || parse.ptr != end ||
        parsed > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
        options.max_lines_valid = false;
        options.max_lines = 0u;
        return options;
    }
    options.max_lines = static_cast<std::size_t>(parsed);
    return options;
}

void coordinateDistributedLocalFailure(
    const systems::FESystem& system,
    const std::exception_ptr& local_exception,
    std::string_view phase)
{
    bool any_failed = local_exception != nullptr;
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        int world_size = 1;
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            const int local_ok = local_exception == nullptr ? 1 : 0;
            int all_ok = 0;
            MPI_Allreduce(&local_ok,
                          &all_ok,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          comm);
            any_failed = all_ok == 0;
        }
    }
#endif
    if (!any_failed) {
        return;
    }
    if (local_exception != nullptr) {
        std::rethrow_exception(local_exception);
    }
    throw std::runtime_error(
        "SmallCutAggregationConstraint: diagnostic="
        "distributed_local_phase_failure phase='" +
        std::string(phase) +
        "' another communicator rank rejected its local aggregation data");
}

enum class DistributedAggregationValidation {
    NotParallel,
    Passed,
    DebugBypass,
};

struct DistributedAggregationResult {
    DistributedAggregationValidation validation{
        DistributedAggregationValidation::NotParallel};
    std::vector<ConstraintLine> relevant_lines{};
    std::vector<GlobalIndex> canonical_slaves{};
    std::size_t canonical_candidate_vertices{0u};
    std::size_t canonical_rooted_candidate_vertices{0u};
    std::size_t canonical_rootless_candidate_vertices{0u};
    std::size_t canonical_owned_aggregate_dofs{0u};
    std::size_t canonical_owned_pinned_dofs{0u};
    std::size_t canonical_strong_suppressed_dofs{0u};
};

struct GatheredInt64Words {
    std::vector<std::int64_t> words{};
    std::vector<int> counts{};
    std::vector<int> displacements{};
};

#if FE_HAS_MPI

[[nodiscard]] GatheredInt64Words allGatherInt64Words(
    MPI_Comm comm,
    std::span<const std::int64_t> local_words)
{
    const bool local_count_overflows =
        local_words.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    int local_overflow = local_count_overflows ? 1 : 0;
    int global_overflow = 0;
    MPI_Allreduce(&local_overflow, &global_overflow, 1, MPI_INT, MPI_MAX, comm);
    if (global_overflow != 0) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: distributed aggregation declaration "
            "exceeds the MPI count range");
    }

    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    const int local_count = static_cast<int>(local_words.size());
    GatheredInt64Words gathered;
    auto coordinate_allocation = [&](const std::exception_ptr& exception,
                                     const char* phase) {
        const int local_ok = exception == nullptr ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
        if (all_ok != 0) {
            return;
        }
        if (exception != nullptr) {
            std::rethrow_exception(exception);
        }
        throw std::runtime_error(
            std::string("SmallCutAggregationConstraint: distributed gather ") +
            phase + " failed on another communicator rank");
    };
    std::exception_ptr counts_allocation_exception;
    try {
        gathered.counts.assign(static_cast<std::size_t>(world_size), 0);
    } catch (...) {
        counts_allocation_exception = std::current_exception();
    }
    coordinate_allocation(counts_allocation_exception, "count allocation");
    MPI_Allgather(&local_count,
                  1,
                  MPI_INT,
                  gathered.counts.data(),
                  1,
                  MPI_INT,
                  comm);

    std::int64_t total = 0;
    std::exception_ptr receive_allocation_exception;
    try {
        gathered.displacements.assign(static_cast<std::size_t>(world_size), 0);
        for (int rank = 0; rank < world_size; ++rank) {
            if (total > std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: gathered distributed "
                    "aggregation declarations exceed the MPI displacement "
                    "range");
            }
            gathered.displacements[static_cast<std::size_t>(rank)] =
                static_cast<int>(total);
            total += gathered.counts[static_cast<std::size_t>(rank)];
        }
        if (total > std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: gathered distributed aggregation "
                "declarations exceed the MPI displacement range");
        }
        gathered.words.resize(static_cast<std::size_t>(total));
    } catch (...) {
        receive_allocation_exception = std::current_exception();
    }
    coordinate_allocation(receive_allocation_exception,
                          "receive-layout allocation");
    MPI_Allgatherv(local_words.data(),
                   local_count,
                   MPI_INT64_T,
                   gathered.words.data(),
                   gathered.counts.data(),
                   gathered.displacements.data(),
                   MPI_INT64_T,
                   comm);
    return gathered;
}

using GlobalCandidateMap =
    std::map<GlobalIndex, GlobalCandidateSupport>;

[[nodiscard]] GlobalCandidateMap resolveGlobalCandidateSupport(
    const systems::FESystem& system,
    std::string_view field_name,
    std::span<const LocalCandidateSupport> local_support,
    bool slave_all_cut)
{
    struct CombinedFacts {
        std::vector<GlobalIndex> component_dofs{};
        std::array<Real, 3> coordinates{};
        bool touches_cut{false};
        bool touches_full_active{false};
        bool excluded{false};
    };
    std::map<GlobalIndex, CombinedFacts> facts;
    auto merge_facts = [&](const LocalCandidateSupport& support,
                           int source_rank) {
        if (support.declaration_dof < 0 || support.component_dofs.empty() ||
            support.component_dofs.front() != support.declaration_dof ||
            !std::all_of(support.coordinates.begin(),
                         support.coordinates.end(),
                         [](Real value) { return std::isfinite(value); })) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: malformed local candidate "
                "support for field '" + std::string(field_name) + "'");
        }
        auto [it, inserted] = facts.try_emplace(
            support.declaration_dof,
            CombinedFacts{.component_dofs = support.component_dofs,
                          .coordinates = support.coordinates,
                          .touches_cut = support.touches_cut,
                          .touches_full_active = support.touches_full_active,
                          .excluded = support.excluded});
        if (!inserted) {
            if (it->second.component_dofs != support.component_dofs) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: communicator ranks "
                    "disagree on component DOFs for candidate " +
                    std::to_string(support.declaration_dof) + " in field '" +
                    std::string(field_name) + "' (source rank " +
                    std::to_string(source_rank) + ")");
            }
            for (std::size_t d = 0; d < support.coordinates.size(); ++d) {
                const auto scale = std::max(
                    {Real{1}, std::abs(it->second.coordinates[d]),
                     std::abs(support.coordinates[d])});
                if (std::abs(it->second.coordinates[d] -
                             support.coordinates[d]) > Real{1e-12} * scale) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: communicator ranks "
                        "disagree on candidate coordinates for DOF " +
                        std::to_string(support.declaration_dof) +
                        " in field '" + std::string(field_name) + "'");
                }
            }
            it->second.touches_cut =
                it->second.touches_cut || support.touches_cut;
            it->second.touches_full_active =
                it->second.touches_full_active || support.touches_full_active;
            it->second.excluded = it->second.excluded || support.excluded;
        }
    };

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        for (const auto& support : local_support) {
            merge_facts(support, 0);
        }
    } else {
        const auto comm = system.dofHandler().mpiComm();
        int world_size = 1;
        MPI_Comm_size(comm, &world_size);
        if (world_size <= 1) {
            for (const auto& support : local_support) {
                merge_facts(support, 0);
            }
        } else {
            std::vector<const LocalCandidateSupport*> ordered;
            // [base_dof, flags, component_count, xyz_bits[3],
            //  component_dofs...], where
            // flags = cut | (full << 1) | (excluded << 2).
            std::vector<std::int64_t> local_words;
            std::exception_ptr local_serialization_exception;
            try {
            ordered.reserve(local_support.size());
            for (const auto& support : local_support) {
                ordered.push_back(&support);
            }
            std::sort(ordered.begin(), ordered.end(), [](const auto* a,
                                                         const auto* b) {
                return a->declaration_dof < b->declaration_dof;
            });
            for (const auto* support : ordered) {
                const std::int64_t flags =
                    (support->touches_cut ? 1 : 0) |
                    (support->touches_full_active ? 2 : 0) |
                    (support->excluded ? 4 : 0);
                local_words.push_back(
                    static_cast<std::int64_t>(support->declaration_dof));
                local_words.push_back(flags);
                local_words.push_back(static_cast<std::int64_t>(
                    support->component_dofs.size()));
                for (const auto coordinate : support->coordinates) {
                    local_words.push_back(std::bit_cast<std::int64_t>(
                        static_cast<double>(coordinate)));
                }
                for (const auto dof : support->component_dofs) {
                    local_words.push_back(static_cast<std::int64_t>(dof));
                }
            }
            } catch (...) {
                local_serialization_exception = std::current_exception();
            }
            coordinateDistributedLocalFailure(
                system,
                local_serialization_exception,
                "candidate_support_serialization");
            const auto gathered = allGatherInt64Words(
                comm, std::span<const std::int64_t>(local_words));
            std::exception_ptr local_decode_exception;
            try {
            for (int rank = 0; rank < world_size; ++rank) {
                std::size_t position = static_cast<std::size_t>(
                    gathered.displacements[static_cast<std::size_t>(rank)]);
                const auto end = position + static_cast<std::size_t>(
                    gathered.counts[static_cast<std::size_t>(rank)]);
                while (position < end) {
                    if (end - position < 6u) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: malformed "
                            "distributed candidate-support declaration");
                    }
                    LocalCandidateSupport support;
                    support.declaration_dof = static_cast<GlobalIndex>(
                        gathered.words[position++]);
                    const auto flags = gathered.words[position++];
                    const auto component_count = gathered.words[position++];
                    if (flags < 0 || flags > 7 || component_count <= 0 ||
                        static_cast<std::uint64_t>(component_count) >
                            static_cast<std::uint64_t>(end - position - 3u)) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: malformed "
                            "distributed candidate-support payload");
                    }
                    support.touches_cut = (flags & 1) != 0;
                    support.touches_full_active = (flags & 2) != 0;
                    support.excluded = (flags & 4) != 0;
                    for (auto& coordinate : support.coordinates) {
                        coordinate = static_cast<Real>(std::bit_cast<double>(
                            gathered.words[position++]));
                    }
                    support.component_dofs.reserve(
                        static_cast<std::size_t>(component_count));
                    for (std::int64_t i = 0; i < component_count; ++i) {
                        support.component_dofs.push_back(
                            static_cast<GlobalIndex>(
                                gathered.words[position++]));
                    }
                    merge_facts(support, rank);
                }
            }
            } catch (...) {
                local_decode_exception = std::current_exception();
            }
            coordinateDistributedLocalFailure(
                system,
                local_decode_exception,
                "candidate_support_decode");
        }
    }

    GlobalCandidateMap combined;
    std::exception_ptr local_combination_exception;
    try {
    for (auto& [dof, global] : facts) {
        const bool candidate =
            global.touches_cut && !global.excluded &&
            (slave_all_cut || !global.touches_full_active);
        if (!candidate) {
            continue;
        }
        combined.emplace(
            dof,
            GlobalCandidateSupport{
                .component_dofs = std::move(global.component_dofs),
                .coordinates = global.coordinates,
                .has_full_support = global.touches_full_active});
    }
    } catch (...) {
        local_combination_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_combination_exception,
                                      "candidate_support_combination");
    return combined;
}

[[nodiscard]] bool equivalentConstraintLines(const ConstraintLine& a,
                                             const ConstraintLine& b,
                                             double tolerance)
{
    if (a.slave_dof != b.slave_dof ||
        std::abs(a.inhomogeneity - b.inhomogeneity) > tolerance ||
        a.entries.size() != b.entries.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.entries.size(); ++i) {
        if (a.entries[i].master_dof != b.entries[i].master_dof ||
            std::abs(a.entries[i].weight - b.entries[i].weight) > tolerance) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] DistributedAggregationResult
resolveDistributedAggregationDeclarations(
    const systems::FESystem& system,
    const AffineConstraints& existing_constraints,
    std::string_view field_name,
    const GlobalCandidateMap& global_candidates,
    const LocalAggregationDeclarationMap& local_candidates,
    bool slave_all_cut,
    bool allow_unaggregated)
{
    DistributedAggregationResult result;
    int initialized = 0;
    MPI_Initialized(&initialized);
    const auto comm = system.dofHandler().mpiComm();
    int world_size = 1;
    if (initialized != 0) {
        MPI_Comm_size(comm, &world_size);
    }
    result.validation =
        initialized != 0 && world_size > 1
            ? DistributedAggregationValidation::Passed
            : DistributedAggregationValidation::NotParallel;
    result.canonical_candidate_vertices = global_candidates.size();
    result.canonical_rooted_candidate_vertices =
        static_cast<std::size_t>(std::count_if(
            global_candidates.begin(), global_candidates.end(),
            [](const auto& entry) { return entry.second.globally_rooted; }));
    result.canonical_rootless_candidate_vertices =
        global_candidates.size() - result.canonical_rooted_candidate_vertices;

    std::size_t local_declaration_count = 0u;
    std::vector<AggregationDeclaration> normalized_local_declarations;
    std::vector<std::pair<GlobalIndex, const AggregationDeclaration*>> ordered;
    std::vector<std::int64_t> local_candidate_words;
    bool local_candidate_payload_valid = true;
    std::exception_ptr local_candidate_serialization_exception;
    try {
    for (const auto& [dof, declarations] : local_candidates) {
        if (global_candidates.count(dof) != 0u) {
            local_declaration_count += declarations.size();
        }
    }
    normalized_local_declarations.reserve(local_declaration_count);
    ordered.reserve(local_declaration_count);
    for (const auto& [dof, declarations] : local_candidates) {
        if (global_candidates.count(dof) == 0u) {
            continue;
        }
        for (const auto& declaration : declarations) {
            auto normalized = declaration;
            bool valid = true;
            if (normalized.state == AggregationDeclarationState::Rooted) {
                std::set<GlobalIndex> line_slaves;
                valid = normalized.root_cell_gid >= 0 &&
                        !normalized.component_dofs.empty() &&
                        !normalized.root_master_dofs.empty() &&
                        normalized.lines.size() ==
                            normalized.component_dofs.size();
                for (auto& line : normalized.lines) {
                    const auto line_validation =
                        normalizeAndValidateRootedLine(line);
                    valid = valid && line_validation.valid &&
                            std::find(normalized.component_dofs.begin(),
                                      normalized.component_dofs.end(),
                                      line.slave_dof) !=
                                normalized.component_dofs.end() &&
                            line_slaves.insert(line.slave_dof).second;
                }
                valid = valid &&
                        line_slaves.size() == normalized.component_dofs.size();
            }
            // A bad root proposal is not a communicator failure if another
            // root/provider can furnish a valid extension for the candidate.
            if (!valid) {
                continue;
            }
            normalized_local_declarations.push_back(std::move(normalized));
            ordered.emplace_back(dof, &normalized_local_declarations.back());
        }
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return detail::smallCutAggregationPhysicalRootLess(
                      physicalRootKey(*a.second),
                      /*lhs_provider_rank=*/0,
                      physicalRootKey(*b.second),
                      /*rhs_provider_rank=*/0);
              });

    // Proposal payload:
    // [base, state, distance, root_cell_gid, n_components, n_roots, n_lines,
    //  components..., roots..., {slave,n_entries,{master,weight_bits}...}...]
    for (const auto& [dof, declaration] : ordered) {
        local_candidate_words.push_back(static_cast<std::int64_t>(dof));
        local_candidate_words.push_back(
            static_cast<std::int64_t>(declaration->state));
        local_candidate_words.push_back(
            declaration->root_distance ==
                    std::numeric_limits<std::size_t>::max()
                ? std::int64_t{-1}
                : static_cast<std::int64_t>(declaration->root_distance));
        local_candidate_words.push_back(
            static_cast<std::int64_t>(declaration->root_cell_gid));
        local_candidate_words.push_back(static_cast<std::int64_t>(
            declaration->component_dofs.size()));
        local_candidate_words.push_back(static_cast<std::int64_t>(
            declaration->root_master_dofs.size()));
        local_candidate_words.push_back(static_cast<std::int64_t>(
            declaration->lines.size()));
        for (const auto component : declaration->component_dofs) {
            local_candidate_words.push_back(
                static_cast<std::int64_t>(component));
        }
        for (const auto master : declaration->root_master_dofs) {
            local_candidate_words.push_back(static_cast<std::int64_t>(master));
        }
        for (auto line : declaration->lines) {
            const auto line_validation =
                normalizeAndValidateRootedLine(line);
            if (!line_validation.valid) {
                local_candidate_payload_valid = false;
            }
            local_candidate_words.push_back(
                static_cast<std::int64_t>(line.slave_dof));
            local_candidate_words.push_back(static_cast<std::int64_t>(
                line.entries.size()));
            for (const auto& entry : line.entries) {
                if (!std::isfinite(entry.weight)) {
                    local_candidate_payload_valid = false;
                }
                local_candidate_words.push_back(
                    static_cast<std::int64_t>(entry.master_dof));
                local_candidate_words.push_back(
                    std::bit_cast<std::int64_t>(entry.weight));
            }
        }
    }
    } catch (...) {
        local_candidate_payload_valid = false;
        local_candidate_serialization_exception = std::current_exception();
    }
    GatheredInt64Words gathered_candidates;
    if (initialized != 0 && world_size > 1) {
        const int local_valid = local_candidate_payload_valid ? 1 : 0;
        int all_valid = 0;
        MPI_Allreduce(&local_valid,
                      &all_valid,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      comm);
        if (all_valid == 0) {
            if (local_candidate_serialization_exception != nullptr) {
                std::rethrow_exception(
                    local_candidate_serialization_exception);
            }
            throw std::runtime_error(
                "SmallCutAggregationConstraint: non-finite local "
                "aggregation weight on at least one communicator rank");
        }
        gathered_candidates = allGatherInt64Words(
            comm, std::span<const std::int64_t>(local_candidate_words));
    } else {
        if (!local_candidate_payload_valid) {
            if (local_candidate_serialization_exception != nullptr) {
                std::rethrow_exception(
                    local_candidate_serialization_exception);
            }
            throw std::runtime_error(
                "SmallCutAggregationConstraint: non-finite local "
                "aggregation weight");
        }
        gathered_candidates.words = local_candidate_words;
        gathered_candidates.counts = {
            static_cast<int>(local_candidate_words.size())};
        gathered_candidates.displacements = {0};
        world_size = 1;
    }

    struct RankedDeclaration {
        int rank{-1};
        AggregationDeclaration declaration{};
    };
    std::map<GlobalIndex, std::vector<RankedDeclaration>> declarations_by_dof;
    std::exception_ptr local_declaration_decode_exception;
    try {
    for (int rank = 0; rank < world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered_candidates.displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered_candidates.counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            if (end - position < 7u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "aggregation candidate declaration");
            }
            const auto dof = static_cast<GlobalIndex>(
                gathered_candidates.words[position++]);
            const auto raw_state = gathered_candidates.words[position++];
            const auto raw_distance = gathered_candidates.words[position++];
            const auto raw_root_cell_gid =
                gathered_candidates.words[position++];
            const auto component_count = gathered_candidates.words[position++];
            const auto master_count = gathered_candidates.words[position++];
            const auto line_count = gathered_candidates.words[position++];
            if (raw_state < static_cast<std::int64_t>(
                                AggregationDeclarationState::Rooted) ||
                raw_state > static_cast<std::int64_t>(
                                AggregationDeclarationState::SupportedWithoutForeignRoot) ||
                raw_distance < -1 || component_count <= 0 ||
                master_count < 0 || line_count < 0 ||
                (raw_state == static_cast<std::int64_t>(
                                  AggregationDeclarationState::Rooted) &&
                 raw_root_cell_gid < 0)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "aggregation candidate payload");
            }
            const auto remaining_header_words =
                static_cast<std::uint64_t>(end - position);
            const auto component_words =
                static_cast<std::uint64_t>(component_count);
            const auto master_words = static_cast<std::uint64_t>(master_count);
            const auto line_headers = static_cast<std::uint64_t>(line_count);
            if (component_words > remaining_header_words ||
                master_words > remaining_header_words - component_words ||
                line_headers >
                    (remaining_header_words - component_words - master_words) /
                        2u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "aggregation candidate counts");
            }
            RankedDeclaration ranked;
            ranked.rank = rank;
            ranked.declaration.state =
                static_cast<AggregationDeclarationState>(raw_state);
            ranked.declaration.root_distance =
                raw_distance < 0
                    ? std::numeric_limits<std::size_t>::max()
                    : static_cast<std::size_t>(raw_distance);
            ranked.declaration.root_cell_gid =
                static_cast<GlobalIndex>(raw_root_cell_gid);
            ranked.declaration.component_dofs.reserve(
                static_cast<std::size_t>(component_count));
            for (std::int64_t i = 0; i < component_count; ++i) {
                ranked.declaration.component_dofs.push_back(
                    static_cast<GlobalIndex>(
                        gathered_candidates.words[position++]));
            }
            ranked.declaration.root_master_dofs.reserve(
                static_cast<std::size_t>(master_count));
            for (std::int64_t i = 0; i < master_count; ++i) {
                ranked.declaration.root_master_dofs.push_back(
                    static_cast<GlobalIndex>(
                        gathered_candidates.words[position++]));
            }
            ranked.declaration.lines.reserve(
                static_cast<std::size_t>(line_count));
            bool ranked_valid = true;
            std::set<GlobalIndex> ranked_line_slaves;
            for (std::int64_t i = 0; i < line_count; ++i) {
                if (end - position < 2u) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: malformed distributed "
                        "aggregation line header");
                }
                ConstraintLine line;
                line.slave_dof = static_cast<GlobalIndex>(
                    gathered_candidates.words[position++]);
                const auto entry_count =
                    gathered_candidates.words[position++];
                if (entry_count < 0 ||
                    static_cast<std::uint64_t>(entry_count) * 2u >
                        static_cast<std::uint64_t>(end - position)) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: malformed distributed "
                        "aggregation line payload");
                }
                line.entries.reserve(static_cast<std::size_t>(entry_count));
                for (std::int64_t entry = 0; entry < entry_count; ++entry) {
                    const auto master = static_cast<GlobalIndex>(
                        gathered_candidates.words[position++]);
                    const auto weight = std::bit_cast<double>(
                        gathered_candidates.words[position++]);
                    line.entries.push_back({master, weight});
                }
                const auto line_validation =
                    normalizeAndValidateRootedLine(line);
                ranked_valid = ranked_valid && line_validation.valid &&
                    std::find(ranked.declaration.component_dofs.begin(),
                              ranked.declaration.component_dofs.end(),
                              line.slave_dof) !=
                        ranked.declaration.component_dofs.end() &&
                    ranked_line_slaves.insert(line.slave_dof).second;
                ranked.declaration.lines.push_back(std::move(line));
            }
            if (ranked.declaration.state ==
                    AggregationDeclarationState::Rooted) {
                ranked_valid = ranked_valid &&
                    ranked.declaration.root_cell_gid >= 0 &&
                    !ranked.declaration.root_master_dofs.empty() &&
                    ranked.declaration.lines.size() ==
                        ranked.declaration.component_dofs.size() &&
                    ranked_line_slaves.size() ==
                        ranked.declaration.component_dofs.size();
            }
            if (ranked_valid) {
                declarations_by_dof[dof].push_back(std::move(ranked));
            }
        }
    }
    } catch (...) {
        local_declaration_decode_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_declaration_decode_exception,
                                      "root_proposal_decode");

    // Root selection has to account for the overlap on every rank that will
    // assemble a slave.  Choosing the lexicographically first root and only
    // then checking its masters can reject an otherwise usable aggregate: a
    // second rank may have proposed a slightly more distant root whose basis
    // is visible on all slave-relevant ranks.  Gather relevance for every DOF
    // mentioned by any proposal so unavailable roots can be removed before
    // applying the deterministic (distance, physical root cell GID, rank)
    // ordering. Algebraic master DOFs are deliberately excluded because
    // owner-contiguous numbering changes when the physical mesh is
    // repartitioned.
    const auto& partition = system.dofHandler().getPartition();
    std::set<GlobalIndex> proposal_dofs;
    std::vector<std::int64_t> local_proposal_visibility_words;
    std::exception_ptr local_proposal_visibility_exception;
    try {
    for (const auto& [dof, support] : global_candidates) {
        (void)dof;
        proposal_dofs.insert(support.component_dofs.begin(),
                             support.component_dofs.end());
    }
    for (const auto& [dof, declarations] : declarations_by_dof) {
        (void)dof;
        for (const auto& ranked : declarations) {
            for (const auto& line : ranked.declaration.lines) {
                proposal_dofs.insert(line.slave_dof);
                for (const auto& entry : line.entries) {
                    proposal_dofs.insert(entry.master_dof);
                }
            }
        }
    }

    // [dof, owned, preconstrained] for locally relevant proposal DOFs.
    for (const auto dof : proposal_dofs) {
        if (!partition.isRelevant(dof)) {
            continue;
        }
        local_proposal_visibility_words.push_back(
            static_cast<std::int64_t>(dof));
        local_proposal_visibility_words.push_back(
            partition.isOwned(dof) ? 1 : 0);
        local_proposal_visibility_words.push_back(
            existing_constraints.isConstrained(dof) ? 1 : 0);
    }
    } catch (...) {
        local_proposal_visibility_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_proposal_visibility_exception,
                                      "proposal_visibility_serialization");
    GatheredInt64Words gathered_proposal_visibility;
    if (initialized != 0 && world_size > 1) {
        gathered_proposal_visibility = allGatherInt64Words(
            comm,
            std::span<const std::int64_t>(local_proposal_visibility_words));
    } else {
        gathered_proposal_visibility.words =
            local_proposal_visibility_words;
        gathered_proposal_visibility.counts = {
            static_cast<int>(local_proposal_visibility_words.size())};
        gathered_proposal_visibility.displacements = {0};
    }

    struct ProposalDofVisibility {
        int rank{-1};
        bool owned{false};
        bool preconstrained{false};
    };
    std::map<GlobalIndex, std::vector<ProposalDofVisibility>>
        proposal_visibility_by_dof;
    std::exception_ptr local_proposal_visibility_decode_exception;
    try {
    for (int rank = 0; rank < world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered_proposal_visibility
                .displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered_proposal_visibility
                .counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            if (end - position < 3u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed proposal-DOF "
                    "visibility declaration");
            }
            const auto dof = static_cast<GlobalIndex>(
                gathered_proposal_visibility.words[position++]);
            ProposalDofVisibility visibility;
            visibility.rank = rank;
            visibility.owned =
                gathered_proposal_visibility.words[position++] != 0;
            visibility.preconstrained =
                gathered_proposal_visibility.words[position++] != 0;
            proposal_visibility_by_dof[dof].push_back(visibility);
        }
    }
    } catch (...) {
        local_proposal_visibility_decode_exception =
            std::current_exception();
    }
    coordinateDistributedLocalFailure(
        system,
        local_proposal_visibility_decode_exception,
        "proposal_visibility_decode");

    auto globallyPreconstrained = [&](GlobalIndex dof) {
        const auto found = proposal_visibility_by_dof.find(dof);
        return found != proposal_visibility_by_dof.end() &&
               std::any_of(
                   found->second.begin(), found->second.end(),
                   [](const ProposalDofVisibility& visibility) {
                       return visibility.preconstrained;
                   });
    };
    auto relevantOnRank = [&](GlobalIndex dof, int rank) {
        const auto found = proposal_visibility_by_dof.find(dof);
        return found != proposal_visibility_by_dof.end() &&
               std::any_of(
                   found->second.begin(), found->second.end(),
                   [&](const ProposalDofVisibility& visibility) {
                       return visibility.rank == rank;
                   });
    };
    auto ownerCount = [&](GlobalIndex dof) {
        const auto found = proposal_visibility_by_dof.find(dof);
        return found == proposal_visibility_by_dof.end()
                   ? std::size_t{0}
                   : static_cast<std::size_t>(std::count_if(
                         found->second.begin(), found->second.end(),
                         [](const ProposalDofVisibility& visibility) {
                             return visibility.owned;
                         }));
    };

    std::map<GlobalIndex, ConstraintLine> canonical_lines;
    std::set<GlobalIndex> expected_slaves;
    std::ostringstream failures;
    std::size_t failure_count = 0u;
    std::vector<std::int64_t> local_visibility_words;
    std::exception_ptr local_canonical_selection_exception;
    constexpr double line_tolerance = 1.0e-12;
    try {
    for (const auto& [dof, support] : global_candidates) {
        const bool all_components_preconstrained = std::all_of(
            support.component_dofs.begin(), support.component_dofs.end(),
            [&](GlobalIndex slave) {
                return globallyPreconstrained(slave);
            });
        if (all_components_preconstrained) {
            expected_slaves.insert(support.component_dofs.begin(),
                                   support.component_dofs.end());
            continue;
        }
        const auto found = declarations_by_dof.find(dof);
        std::vector<const RankedDeclaration*> rooted;
        if (found != declarations_by_dof.end()) {
            for (const auto& declaration : found->second) {
                if (declaration.declaration.component_dofs !=
                    support.component_dofs) {
                    ++failure_count;
                    if (failure_count <= 4u) {
                        failures << " dof=" << dof
                                 << " reason=component_dof_mismatch;";
                    }
                    continue;
                }
                if (declaration.declaration.state ==
                    AggregationDeclarationState::Rooted) {
                    rooted.push_back(&declaration);
                }
            }
        }

        if (rooted.empty()) {
            if (support.globally_rooted) {
                if (allow_unaggregated) {
                    continue;
                }
                ++failure_count;
                if (failure_count <= 4u) {
                    failures << " dof=" << dof
                             << " reason=no_valid_root_proposal;";
                }
                continue;
            }
            if (slave_all_cut && support.has_full_support) {
                // Globally supported slave-all candidate with no foreign root:
                // it remains free by the documented experimental policy.
                continue;
            }
            if (!allow_unaggregated) {
                for (const auto slave : support.component_dofs) {
                    expected_slaves.insert(slave);
                    canonical_lines.emplace(
                        slave, ConstraintLine{.slave_dof = slave});
                }
            }
            continue;
        }

        std::vector<const RankedDeclaration*> available_rooted;
        std::string first_unavailable_reason;
        for (const auto* declaration : rooted) {
            bool available = true;
            for (const auto slave : support.component_dofs) {
                if (globallyPreconstrained(slave)) {
                    continue;
                }
                const auto line_it = std::find_if(
                    declaration->declaration.lines.begin(),
                    declaration->declaration.lines.end(),
                    [&](const ConstraintLine& line) {
                        return line.slave_dof == slave;
                    });
                if (line_it == declaration->declaration.lines.end()) {
                    available = false;
                    if (first_unavailable_reason.empty()) {
                        first_unavailable_reason =
                            " reason=no_communicator_global_root_line";
                    }
                    break;
                }
                for (const auto& entry : line_it->entries) {
                    const auto master_owner_count =
                        ownerCount(entry.master_dof);
                    if (master_owner_count == 1u) {
                        continue;
                    }
                    available = false;
                    if (first_unavailable_reason.empty()) {
                        first_unavailable_reason =
                            " reason=canonical_master_owner_count:" +
                            std::to_string(master_owner_count) +
                            " master=" +
                            std::to_string(entry.master_dof);
                    }
                    break;
                }
                if (!available) {
                    break;
                }
                const auto slave_visibility =
                    proposal_visibility_by_dof.find(slave);
                if (slave_visibility == proposal_visibility_by_dof.end()) {
                    available = false;
                    if (first_unavailable_reason.empty()) {
                        first_unavailable_reason =
                            " reason=canonical_slave_not_relevant";
                    }
                    break;
                }
                for (const auto& visibility : slave_visibility->second) {
                    std::vector<GlobalIndex> missing_masters;
                    for (const auto& entry : line_it->entries) {
                        if (!relevantOnRank(entry.master_dof,
                                            visibility.rank)) {
                            missing_masters.push_back(entry.master_dof);
                        }
                    }
                    if (missing_masters.empty()) {
                        continue;
                    }
                    available = false;
                    if (first_unavailable_reason.empty()) {
                        std::ostringstream reason;
                        reason << " reason=canonical_master_not_relevant"
                               << " rank=" << visibility.rank
                               << " masters=";
                        for (const auto master : missing_masters) {
                            reason << master << ",";
                        }
                        first_unavailable_reason = reason.str();
                    }
                    break;
                }
                if (!available) {
                    break;
                }
            }
            if (available) {
                available_rooted.push_back(declaration);
            }
        }
        if (available_rooted.empty()) {
            if (allow_unaggregated) {
                continue;
            }
            ++failure_count;
            if (failure_count <= 4u) {
                failures << " dof=" << dof
                         << (first_unavailable_reason.empty()
                                 ? " reason=no_globally_available_root"
                                 : first_unavailable_reason)
                         << ";";
            }
            continue;
        }

        std::sort(available_rooted.begin(), available_rooted.end(),
                  [](const auto* a, const auto* b) {
            return detail::smallCutAggregationPhysicalRootLess(
                physicalRootKey(a->declaration),
                a->rank,
                physicalRootKey(b->declaration),
                b->rank);
        });
        const auto& chosen = available_rooted.front()->declaration;
        std::vector<const RankedDeclaration*> providers;
        for (const auto* declaration : available_rooted) {
            if (detail::smallCutAggregationSamePhysicalRoot(
                    physicalRootKey(declaration->declaration),
                    physicalRootKey(chosen))) {
                providers.push_back(declaration);
            }
        }

        for (const auto slave : support.component_dofs) {
            const ConstraintLine* selected_line = nullptr;
            for (const auto* provider : providers) {
                const auto line_it = std::find_if(
                    provider->declaration.lines.begin(),
                    provider->declaration.lines.end(),
                    [&](const ConstraintLine& line) {
                        return line.slave_dof == slave;
                    });
                if (line_it == provider->declaration.lines.end()) {
                    continue;
                }
                if (selected_line == nullptr) {
                    selected_line = &*line_it;
                } else if (!equivalentConstraintLines(
                               *selected_line, *line_it, line_tolerance)) {
                    ++failure_count;
                    if (failure_count <= 4u) {
                        failures << " dof=" << slave
                                 << " reason=canonical_root_weight_mismatch;";
                    }
                }
            }
            if (selected_line != nullptr) {
                expected_slaves.insert(slave);
                canonical_lines.emplace(slave, *selected_line);
            } else if (!allow_unaggregated) {
                // Keep the slave in the availability phase: a globally
                // pre-existing strong constraint legitimately explains the
                // missing aggregation line; otherwise this is a fail-closed
                // machinery/overlap error.
                expected_slaves.insert(slave);
            }
        }
    }

    if (failure_count > 0u) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "incomplete_distributed_aggregation_halo field='" +
            std::string(field_name) + "' inconsistent_candidate_dofs=" +
            std::to_string(failure_count) + failures.str() +
            " Communicator-global candidate/root construction produced "
            "inconsistent component/weight data or no proposed root whose "
            "masters are relevant on every slave-relevant rank; refusing "
            "owner-wins constraint resolution.");
    }

    // Every rank now knows the same canonical lines.  Before installing them,
    // prove that each slave has exactly one owner and that every rank on which
    // the slave is locally relevant also carries every nonzero master.  A
    // strong pre-existing constraint on any relevant copy suppresses the
    // aggregation line globally, preserving strong-BC precedence independent
    // of which rank owns the boundary entity.
    for (const auto slave : expected_slaves) {
        if (!partition.isRelevant(slave)) {
            continue;
        }
        const auto line_it = canonical_lines.find(slave);
        std::vector<GlobalIndex> missing_masters;
        if (line_it != canonical_lines.end()) {
            for (const auto& entry : line_it->second.entries) {
                if (!partition.isRelevant(entry.master_dof)) {
                    missing_masters.push_back(entry.master_dof);
                }
            }
        }
        local_visibility_words.push_back(static_cast<std::int64_t>(slave));
        local_visibility_words.push_back(partition.isOwned(slave) ? 1 : 0);
        local_visibility_words.push_back(
            existing_constraints.isConstrained(slave) ? 1 : 0);
        local_visibility_words.push_back(static_cast<std::int64_t>(
            missing_masters.size()));
        for (const auto master : missing_masters) {
            local_visibility_words.push_back(static_cast<std::int64_t>(master));
        }
    }
    } catch (...) {
        local_canonical_selection_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_canonical_selection_exception,
                                      "canonical_root_selection_and_visibility");
    GatheredInt64Words gathered_visibility;
    if (initialized != 0 && world_size > 1) {
        gathered_visibility = allGatherInt64Words(
            comm, std::span<const std::int64_t>(local_visibility_words));
    } else {
        gathered_visibility.words = local_visibility_words;
        gathered_visibility.counts = {
            static_cast<int>(local_visibility_words.size())};
        gathered_visibility.displacements = {0};
    }

    std::exception_ptr local_final_validation_exception;
    try {
    struct Visibility {
        int rank{-1};
        bool owned{false};
        bool preconstrained{false};
        std::vector<GlobalIndex> missing_masters{};
    };
    std::map<GlobalIndex, std::vector<Visibility>> visibility_by_slave;
    for (int rank = 0; rank < world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered_visibility.displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered_visibility.counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            if (end - position < 4u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed canonical-line "
                    "visibility declaration");
            }
            const auto slave = static_cast<GlobalIndex>(
                gathered_visibility.words[position++]);
            Visibility visibility;
            visibility.rank = rank;
            visibility.owned = gathered_visibility.words[position++] != 0;
            visibility.preconstrained =
                gathered_visibility.words[position++] != 0;
            const auto missing_count = gathered_visibility.words[position++];
            if (missing_count < 0 ||
                static_cast<std::uint64_t>(missing_count) >
                    static_cast<std::uint64_t>(end - position)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed canonical-line "
                    "master-availability payload");
            }
            visibility.missing_masters.reserve(
                static_cast<std::size_t>(missing_count));
            for (std::int64_t i = 0; i < missing_count; ++i) {
                visibility.missing_masters.push_back(
                    static_cast<GlobalIndex>(
                        gathered_visibility.words[position++]));
            }
            visibility_by_slave[slave].push_back(std::move(visibility));
        }
    }

    std::set<GlobalIndex> globally_preconstrained;
    failures.str({});
    failures.clear();
    failure_count = 0u;
    for (const auto slave : expected_slaves) {
        const auto found = visibility_by_slave.find(slave);
        const std::size_t owner_count =
            found == visibility_by_slave.end()
                ? 0u
                : static_cast<std::size_t>(std::count_if(
                      found->second.begin(), found->second.end(),
                      [](const Visibility& visibility) {
                          return visibility.owned;
                      }));
        const bool any_preconstrained =
            found != visibility_by_slave.end() &&
            std::any_of(found->second.begin(), found->second.end(),
                        [](const Visibility& visibility) {
                            return visibility.preconstrained;
                        });
        if (any_preconstrained) {
            globally_preconstrained.insert(slave);
            continue;
        }
        if (owner_count != 1u) {
            ++failure_count;
            if (failure_count <= 4u) {
                failures << " dof=" << slave
                         << " reason=canonical_slave_owner_count:"
                         << owner_count << ";";
            }
            continue;
        }
        const auto line_it = canonical_lines.find(slave);
        if (line_it == canonical_lines.end()) {
            ++failure_count;
            if (failure_count <= 4u) {
                failures << " dof=" << slave
                         << " reason=no_communicator_global_root_line;";
            }
            continue;
        }
        bool master_owner_failure = false;
        for (const auto& entry : line_it->second.entries) {
            const auto master_owner_count = ownerCount(entry.master_dof);
            if (master_owner_count == 1u) {
                continue;
            }
            ++failure_count;
            master_owner_failure = true;
            if (failure_count <= 4u) {
                failures << " dof=" << slave
                         << " reason=canonical_master_owner_count:"
                         << master_owner_count
                         << " master=" << entry.master_dof << ";";
            }
            break;
        }
        if (master_owner_failure) {
            continue;
        }
        for (const auto& visibility : found->second) {
            if (!visibility.missing_masters.empty()) {
                ++failure_count;
                if (failure_count <= 4u) {
                    failures << " dof=" << slave
                             << " reason=canonical_master_not_relevant rank="
                             << visibility.rank << " masters=";
                    for (const auto master : visibility.missing_masters) {
                        failures << master << ",";
                    }
                    failures << ";";
                }
                break;
            }
        }
    }
    if (failure_count > 0u) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "incomplete_distributed_aggregation_halo field='" +
            std::string(field_name) + "' inconsistent_candidate_dofs=" +
            std::to_string(failure_count) + failures.str() +
            " Increase mesh/DOF overlap until every rank that assembles a "
            "canonical slave also carries all of its nonzero aggregate "
            "masters; refusing owner-wins constraint resolution.");
    }

    result.canonical_strong_suppressed_dofs =
        globally_preconstrained.size();
    for (const auto& [slave, line] : canonical_lines) {
        if (globally_preconstrained.count(slave) != 0u) {
            continue;
        }
        result.canonical_slaves.push_back(slave);
        if (line.entries.empty()) {
            ++result.canonical_owned_pinned_dofs;
        } else {
            ++result.canonical_owned_aggregate_dofs;
        }
        if (partition.isRelevant(slave)) {
            result.relevant_lines.push_back(line);
        }
    }
    } catch (...) {
        local_final_validation_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_final_validation_exception,
                                      "canonical_line_final_validation");
    return result;
}

#else

using GlobalCandidateMap =
    std::map<GlobalIndex, GlobalCandidateSupport>;

[[nodiscard]] GlobalCandidateMap resolveGlobalCandidateSupport(
    const systems::FESystem&,
    std::string_view,
    std::span<const LocalCandidateSupport> local_support,
    bool slave_all_cut)
{
    GlobalCandidateMap out;
    for (const auto& support : local_support) {
        if (support.touches_cut && !support.excluded &&
            (slave_all_cut || !support.touches_full_active)) {
            out.emplace(
                support.declaration_dof,
                GlobalCandidateSupport{
                    .component_dofs = support.component_dofs,
                    .coordinates = support.coordinates,
                    .has_full_support = support.touches_full_active});
        }
    }
    return out;
}

[[nodiscard]] DistributedAggregationResult
resolveDistributedAggregationDeclarations(
    const systems::FESystem& system,
    const AffineConstraints& existing_constraints,
    std::string_view,
    const GlobalCandidateMap& global_candidates,
    const LocalAggregationDeclarationMap& local_candidates,
    bool slave_all_cut,
    bool allow_unaggregated)
{
    DistributedAggregationResult result;
    const auto& partition = system.dofHandler().getPartition();
    result.canonical_candidate_vertices = global_candidates.size();
    result.canonical_rooted_candidate_vertices =
        static_cast<std::size_t>(std::count_if(
            global_candidates.begin(), global_candidates.end(),
            [](const auto& entry) { return entry.second.globally_rooted; }));
    result.canonical_rootless_candidate_vertices =
        global_candidates.size() - result.canonical_rooted_candidate_vertices;
    for (const auto& [dof, support] : global_candidates) {
        const bool all_components_preconstrained = std::all_of(
            support.component_dofs.begin(), support.component_dofs.end(),
            [&](GlobalIndex slave) {
                return existing_constraints.isConstrained(slave);
            });
        if (all_components_preconstrained) {
            result.canonical_strong_suppressed_dofs +=
                support.component_dofs.size();
            continue;
        }
        const auto found = local_candidates.find(dof);
        std::vector<AggregationDeclaration> valid_declarations;
        if (found != local_candidates.end()) {
            for (const auto& declaration : found->second) {
                if (declaration.state !=
                    AggregationDeclarationState::Rooted) {
                    continue;
                }
                auto normalized = declaration;
                std::set<GlobalIndex> line_slaves;
                bool valid = normalized.component_dofs ==
                                 support.component_dofs &&
                             normalized.root_cell_gid >= 0 &&
                             !normalized.root_master_dofs.empty() &&
                             normalized.lines.size() ==
                                 support.component_dofs.size();
                for (auto& line : normalized.lines) {
                    const auto line_validation =
                        normalizeAndValidateRootedLine(line);
                    valid = valid && line_validation.valid &&
                        std::find(support.component_dofs.begin(),
                                  support.component_dofs.end(),
                                  line.slave_dof) !=
                            support.component_dofs.end() &&
                        line_slaves.insert(line.slave_dof).second;
                }
                valid = valid &&
                        line_slaves.size() == support.component_dofs.size();
                if (valid) {
                    valid_declarations.push_back(std::move(normalized));
                }
            }
        }
        std::sort(valid_declarations.begin(), valid_declarations.end(),
                  [](const auto& a, const auto& b) {
                      return detail::smallCutAggregationPhysicalRootLess(
                          physicalRootKey(a),
                          /*lhs_provider_rank=*/0,
                          physicalRootKey(b),
                          /*rhs_provider_rank=*/0);
                  });
        if (!valid_declarations.empty()) {
            for (const auto& line : valid_declarations.front().lines) {
                if (existing_constraints.isConstrained(line.slave_dof)) {
                    ++result.canonical_strong_suppressed_dofs;
                    continue;
                }
                result.canonical_slaves.push_back(line.slave_dof);
                ++result.canonical_owned_aggregate_dofs;
                if (partition.isRelevant(line.slave_dof)) {
                    result.relevant_lines.push_back(line);
                }
            }
        } else if (support.globally_rooted) {
            if (allow_unaggregated) {
                continue;
            }
            throw std::runtime_error(
                "SmallCutAggregationConstraint: globally rooted candidate " +
                std::to_string(dof) +
                " has no valid serial root proposal");
        } else if (!(slave_all_cut && support.has_full_support) &&
                   !allow_unaggregated) {
            for (const auto slave : support.component_dofs) {
                if (existing_constraints.isConstrained(slave)) {
                    ++result.canonical_strong_suppressed_dofs;
                    continue;
                }
                result.canonical_slaves.push_back(slave);
                ++result.canonical_owned_pinned_dofs;
                if (partition.isRelevant(slave)) {
                    result.relevant_lines.push_back(
                        ConstraintLine{.slave_dof = slave});
                }
            }
        }
    }
    std::sort(result.canonical_slaves.begin(),
              result.canonical_slaves.end());
    return result;
}

#endif

/// Corner (linear) topology of an element type. Identity for linear types.
[[nodiscard]] ElementType linearElementType(ElementType type) noexcept
{
    switch (type) {
        case ElementType::Line3: return ElementType::Line2;
        case ElementType::Triangle6: return ElementType::Triangle3;
        case ElementType::Quad8:
        case ElementType::Quad9: return ElementType::Quad4;
        case ElementType::Tetra10: return ElementType::Tetra4;
        case ElementType::Hex20:
        case ElementType::Hex27: return ElementType::Hex8;
        case ElementType::Wedge15:
        case ElementType::Wedge18: return ElementType::Wedge6;
        case ElementType::Pyramid13:
        case ElementType::Pyramid14: return ElementType::Pyramid5;
        default: return type;
    }
}

[[nodiscard]] std::shared_ptr<const geometry::GeometryMapping> makeMapping(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell)
{
    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(cell, coords);
    if (coords.empty()) {
        return nullptr;
    }
    std::vector<Vector<Real, 3>> nodes;
    nodes.reserve(coords.size());
    for (const auto& coord : coords) {
        Vector<Real, 3> node{};
        node[0] = coord[0];
        node[1] = coord[1];
        node[2] = coord[2];
        nodes.push_back(node);
    }
    geometry::MappingRequest request;
    request.element_type = mesh.getCellType(cell);
    request.geometry_order = mesh.getCellGeometryOrder(cell);
    request.use_affine = request.geometry_order <= 1;
    return geometry::MappingFactory::create(request, nodes);
}

[[nodiscard]] Real physicalRetainedVolumeRuleMeasure(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell,
    const geometry::CutQuadratureRule& rule)
{
    if (rule.kind != geometry::CutQuadratureKind::Volume) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: retained-volume telemetry "
            "requires a volume quadrature rule");
    }
    Real measure = 0.0;
    if (!rule.points.empty()) {
        auto mapped_rule = rule;
        mapped_rule.provenance.parent_entity =
            static_cast<MeshIndex>(cell);
        measure =
            geometry::physicalCutQuadratureMeasure(mesh, mapped_rule);
    } else if (rule.frame == geometry::CutGeometryFrame::Current) {
        // Hand-built constraint fixtures historically provide only a
        // current-frame measure. Production generated rules carry points and
        // use the common pointwise mapping path below.
        if (!std::isfinite(rule.measure) || rule.measure < Real{0.0}) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume current-frame measure must "
                "be finite and nonnegative");
        }
        measure = rule.measure;
    } else if (rule.full_cell_equivalent) {
        // Retain compatibility with an empty synthetic reference-frame
        // full-cell rule. Production rules always take the authoritative
        // pointwise path above.
        measure =
            geometry::physicalCellMeasureFromMapping(mesh, cell);
    } else {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "invalid_active_feature_volume partial retained rule has "
            "no quadrature points");
    }
    if (!std::isfinite(measure) || measure < Real{0.0}) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "invalid_active_feature_volume accumulated physical measure "
            "is invalid");
    }
    return measure;
}

/// Newton inversion of the cell geometry mapping. Extrapolation (reference
/// coordinates outside the unit cell) is expected and valid for aggregation.
[[nodiscard]] bool invertMapping(const geometry::GeometryMapping& mapping,
                                 const std::array<Real, 3>& physical,
                                 int dimension,
                                 Vector<Real, 3>& xi)
{
    const auto& nodes = mapping.nodes();
    if (nodes.empty()) {
        return false;
    }
    auto minimum = nodes.front();
    auto maximum = nodes.front();
    Real coordinate_scale = geometry::detail::stable_vector_norm(nodes.front());
    for (const auto& node : nodes) {
        const Real node_norm = geometry::detail::stable_vector_norm(node);
        if (!std::isfinite(node_norm)) {
            return false;
        }
        coordinate_scale = std::max(coordinate_scale, node_norm);
        for (std::size_t d = 0; d < 3u; ++d) {
            minimum[d] = std::min(minimum[d], node[d]);
            maximum[d] = std::max(maximum[d], node[d]);
        }
    }
    const Real span = geometry::detail::stable_vector_norm(maximum - minimum);
    const Vector<Real, 3> physical_vector{
        physical[0], physical[1], physical[2]};
    coordinate_scale = std::max(
        {coordinate_scale,
         geometry::detail::stable_vector_norm(physical_vector),
         span});
    if (!std::isfinite(span) || span <= Real(0) ||
        !std::isfinite(coordinate_scale)) {
        return false;
    }

    constexpr Real relative_physical_tolerance = Real(1e-12);
    constexpr Real machine_floor_factor = Real(64);
    const Real relative_tolerance = relative_physical_tolerance * span;
    const Real machine_floor =
        machine_floor_factor * std::numeric_limits<Real>::epsilon() *
        coordinate_scale;
    const Real physical_tolerance =
        std::max(relative_tolerance, machine_floor);

    xi = Vector<Real, 3>{};
    for (int d = 0; d < dimension; ++d) {
        xi[d] = Real(0.25);
    }
    bool have_update = false;
    Real last_update_norm = std::numeric_limits<Real>::infinity();
    for (int iteration = 0; iteration < 25; ++iteration) {
        const auto mapped = mapping.map_to_physical(xi);
        Vector<Real, 3> residual{};
        for (int d = 0; d < 3; ++d) {
            residual[d] = physical[d] - mapped[d];
        }
        const Real residual_norm =
            geometry::detail::stable_vector_norm(residual);
        if (!std::isfinite(residual_norm)) {
            return false;
        }
        if (residual_norm <= physical_tolerance) {
            const Real xi_scale = std::max(
                Real(1), geometry::detail::stable_vector_norm(xi));
            const Real reference_update_tolerance =
                machine_floor_factor * std::numeric_limits<Real>::epsilon() *
                xi_scale;
            const bool tolerance_is_machine_limited =
                machine_floor > relative_tolerance;
            if (!tolerance_is_machine_limited || residual_norm == Real(0) ||
                (have_update &&
                 last_update_norm <= reference_update_tolerance)) {
                return true;
            }
        }
        const auto jacobian = mapping.jacobian(xi);
        auto j_inv = jacobian;
        if (dimension < 3) {
            // Regularize the unused row/column so the 3x3 inverse exists.
            for (int d = dimension; d < 3; ++d) {
                for (int k = 0; k < 3; ++k) {
                    j_inv(d, k) = Real(0);
                    j_inv(k, d) = Real(0);
                }
                j_inv(d, d) = Real(1);
            }
        }
        const auto inverse =
            geometry::scaleConditionedJacobianInverse(j_inv);
        Vector<Real, 3> reference_update{};
        for (int r = 0; r < 3; ++r) {
            Real update = Real(0);
            for (int c = 0; c < 3; ++c) {
                update += inverse(r, c) * residual[c];
            }
            reference_update[r] = update;
            xi[r] += reference_update[r];
        }
        last_update_norm =
            geometry::detail::stable_vector_norm(reference_update);
        have_update = true;
        for (int d = dimension; d < 3; ++d) {
            xi[d] = Real(0);
        }
        if (!std::isfinite(last_update_norm) ||
            !std::isfinite(xi[0]) || !std::isfinite(xi[1]) ||
            !std::isfinite(xi[2])) {
            return false;
        }
    }
    return false;
}

[[nodiscard]] Real unitSimplexExtrapolationDistance(
    const Vector<Real, 3>& xi,
    int dimension)
{
    std::array<Real, 3> projected{{0.0, 0.0, 0.0}};
    Real positive_sum = 0.0;
    for (int d = 0; d < dimension; ++d) {
        projected[static_cast<std::size_t>(d)] =
            std::max(xi[static_cast<std::size_t>(d)], Real{0.0});
        positive_sum += projected[static_cast<std::size_t>(d)];
    }
    if (positive_sum > Real{1.0}) {
        std::array<Real, 3> sorted{{xi[0], xi[1], xi[2]}};
        std::sort(sorted.begin(), sorted.begin() + dimension,
                  std::greater<Real>{});
        Real prefix = 0.0;
        Real theta = 0.0;
        for (int i = 0; i < dimension; ++i) {
            prefix += sorted[static_cast<std::size_t>(i)];
            const Real candidate =
                (prefix - Real{1.0}) / static_cast<Real>(i + 1);
            if (i + 1 == dimension ||
                sorted[static_cast<std::size_t>(i + 1)] <= candidate) {
                theta = candidate;
                break;
            }
        }
        for (int d = 0; d < dimension; ++d) {
            projected[static_cast<std::size_t>(d)] =
                std::max(xi[static_cast<std::size_t>(d)] - theta, Real{0.0});
        }
    }

    Real squared_distance = 0.0;
    for (int d = 0; d < dimension; ++d) {
        const Real delta =
            xi[static_cast<std::size_t>(d)] -
            projected[static_cast<std::size_t>(d)];
        squared_distance += delta * delta;
    }
    return std::sqrt(squared_distance);
}

[[nodiscard]] Real normalizedReferenceExtrapolationDistance(
    ElementType element_type,
    const Vector<Real, 3>& xi)
{
    const auto type = linearElementType(element_type);
    auto tensor_distance = [&](int dimension) {
        Real squared_distance = 0.0;
        for (int d = 0; d < dimension; ++d) {
            const Real projected = std::clamp(
                xi[static_cast<std::size_t>(d)], Real{-1.0}, Real{1.0});
            const Real delta =
                xi[static_cast<std::size_t>(d)] - projected;
            squared_distance += delta * delta;
        }
        return std::sqrt(squared_distance);
    };

    switch (type) {
    case ElementType::Line2:
        return tensor_distance(1);
    case ElementType::Quad4:
        return tensor_distance(2);
    case ElementType::Hex8:
        return tensor_distance(3);
    case ElementType::Triangle3:
        return unitSimplexExtrapolationDistance(xi, 2);
    case ElementType::Tetra4:
        return unitSimplexExtrapolationDistance(xi, 3);
    case ElementType::Wedge6: {
        const Real in_plane = unitSimplexExtrapolationDistance(xi, 2);
        const Real projected_z =
            std::clamp(xi[2], Real{-1.0}, Real{1.0});
        const Real dz = xi[2] - projected_z;
        return std::sqrt(in_plane * in_plane + dz * dz);
    }
    case ElementType::Pyramid5: {
        const Real abs_x = std::abs(xi[0]);
        const Real abs_y = std::abs(xi[1]);
        const auto squared_distance_at_z = [&](Real z) {
            const Real half_width = Real{1.0} - z;
            const Real dx = std::max(abs_x - half_width, Real{0.0});
            const Real dy = std::max(abs_y - half_width, Real{0.0});
            const Real dz = xi[2] - z;
            return dx * dx + dy * dy + dz * dz;
        };
        Real lower = 0.0;
        Real upper = 1.0;
        for (int iteration = 0; iteration < 80; ++iteration) {
            const Real left = (Real{2.0} * lower + upper) / Real{3.0};
            const Real right = (lower + Real{2.0} * upper) / Real{3.0};
            if (squared_distance_at_z(left) <=
                squared_distance_at_z(right)) {
                upper = right;
            } else {
                lower = left;
            }
        }
        const Real z = Real{0.5} * (lower + upper);
        return std::sqrt(squared_distance_at_z(z));
    }
    default:
        return std::numeric_limits<Real>::infinity();
    }
}

} // namespace

SmallCutAggregationConstraint::SmallCutAggregationConstraint(
    FieldId field,
    geometry::CutIntegrationSide active_side,
    int interface_marker,
    std::vector<int> excluded_boundary_markers,
    std::vector<GlobalIndex> excluded_vertices,
    SmallCutAggregationGuardOptions guards)
    : field_(field),
      active_side_(active_side),
      interface_marker_(interface_marker),
      excluded_boundary_markers_(std::move(excluded_boundary_markers)),
      excluded_vertices_(std::move(excluded_vertices)),
      guards_(guards)
{
    if (interface_marker_ < 0) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: interface marker must be "
            "nonnegative");
    }
    if (active_side_ != geometry::CutIntegrationSide::Negative &&
        active_side_ != geometry::CutIntegrationSide::Positive) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: active side must be Negative or "
            "Positive");
    }
    if (guards_.maximum_root_path_length == 0u ||
        !(guards_.maximum_reference_extrapolation_distance >= Real{0.0}) ||
        !std::isfinite(guards_.maximum_reference_extrapolation_distance) ||
        !(guards_.maximum_absolute_coefficient >= Real{1.0}) ||
        !std::isfinite(guards_.maximum_absolute_coefficient) ||
        !(guards_.maximum_row_l1_norm >= Real{1.0}) ||
        !std::isfinite(guards_.maximum_row_l1_norm) ||
        guards_.maximum_row_l1_norm <
            guards_.maximum_absolute_coefficient) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: guard limits require a positive "
            "root path, finite nonnegative extrapolation distance, finite "
            "coefficient and row L1 limits at least one, and a row L1 limit "
            "no smaller than the coefficient limit");
    }
}

void SmallCutAggregationConstraint::apply(const systems::FESystem& system,
                                          AffineConstraints& constraints)
{
    completed_refresh_report_.reset();
    const auto* cut_context = system.cutIntegrationContext();
    bool distributed_aggregation = false;
#if FE_HAS_MPI
    // A rank-local early return would let another rank enter the declaration
    // collectives below alone.  More importantly, a cut context installed on
    // only part of the communicator is itself an incomplete distributed
    // aggregation stencil, so diagnose it collectively before doing any work.
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        int world_size = 1;
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            distributed_aggregation = true;
            const int local_has_context = cut_context != nullptr ? 1 : 0;
            int minimum_has_context = 0;
            int maximum_has_context = 0;
            MPI_Allreduce(&local_has_context,
                          &minimum_has_context,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          comm);
            MPI_Allreduce(&local_has_context,
                          &maximum_has_context,
                          1,
                          MPI_INT,
                          MPI_MAX,
                          comm);
            if (minimum_has_context != maximum_has_context) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "incomplete_distributed_aggregation_halo cut integration "
                    "context is not installed consistently across the field "
                    "communicator");
            }
        }
    }
#endif
    if (cut_context == nullptr) {
        // setup() applies registered constraints before generated level-set
        // quadrature exists. Permit only that initial lifecycle pass. Once
        // setup is complete, an explicit rebuild/refresh without its required
        // cut context would silently disable aggregation and must fail closed.
        if (!system.isSetup()) {
            return;
        }
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "missing_cut_integration_context aggregation requires a generated "
            "cut context before post-setup constraint rebuild");
    }

#if FE_HAS_MPI
    if (distributed_aggregation) {
        const auto comm = system.dofHandler().mpiComm();
        const int local_has_global_entity_ids =
            system.meshAccess().globalEntityIdsAvailable() ? 1 : 0;
        int all_have_global_entity_ids = 0;
        MPI_Allreduce(&local_has_global_entity_ids,
                      &all_have_global_entity_ids,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      comm);
        if (all_have_global_entity_ids == 0) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "missing_distributed_global_cell_ids physical root selection "
                "requires globally unique cell IDs on every field-communicator "
                "rank");
        }
    }
#else
    static_cast<void>(distributed_aggregation);
#endif

    cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        system.meshAccess());

    // Contexts may be ownership-filtered, so an empty-work rank need not
    // retain this marker locally. At least one rank in the field communicator
    // must, however, prove that the installed aggregation constraint refers
    // to a generated volume marker; otherwise a spelling/configuration error
    // would silently disable the stabilization.
    int communicator_has_volume_marker =
        cut_context->hasGeneratedVolumeMarker(interface_marker_) ? 1 : 0;
#if FE_HAS_MPI
    if (mpi_initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        int global_has_volume_marker = 0;
        MPI_Allreduce(&communicator_has_volume_marker,
                      &global_has_volume_marker,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      comm);
        communicator_has_volume_marker = global_has_volume_marker;
    }
#endif
    if (communicator_has_volume_marker == 0) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "missing_marker_cell_classification interface_marker=" +
            std::to_string(interface_marker_) +
            " is absent from the generated cut-volume context");
    }

    auto coordinateLocalPhaseFailure =
        [&](const std::exception_ptr& local_exception,
            std::string_view phase) {
            coordinateDistributedLocalFailure(system, local_exception, phase);
        };

    auto runtime_options = readAggregationRuntimeOptions();
#if FE_HAS_MPI
    if (mpi_initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        int world_size = 1;
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            const std::array<int, 3> local_flags{
                runtime_options.slave_all_cut ? 1 : 0,
                runtime_options.linear_extension ? 1 : 0,
                runtime_options.allow_unaggregated ? 1 : 0};
            std::array<int, 3> minimum_flags{};
            std::array<int, 3> maximum_flags{};
            MPI_Allreduce(local_flags.data(),
                          minimum_flags.data(),
                          static_cast<int>(local_flags.size()),
                          MPI_INT,
                          MPI_MIN,
                          comm);
            MPI_Allreduce(local_flags.data(),
                          maximum_flags.data(),
                          static_cast<int>(local_flags.size()),
                          MPI_INT,
                          MPI_MAX,
                          comm);
            const int local_valid =
                runtime_options.flags_valid &&
                        runtime_options.max_lines_valid
                    ? 1
                    : 0;
            int all_valid = 0;
            MPI_Allreduce(&local_valid,
                          &all_valid,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          comm);
            const auto local_max_lines = static_cast<std::uint64_t>(
                runtime_options.max_lines);
            std::uint64_t minimum_max_lines = 0u;
            std::uint64_t maximum_max_lines = 0u;
            MPI_Allreduce(&local_max_lines,
                          &minimum_max_lines,
                          1,
                          MPI_UINT64_T,
                          MPI_MIN,
                          comm);
            MPI_Allreduce(&local_max_lines,
                          &maximum_max_lines,
                          1,
                          MPI_UINT64_T,
                          MPI_MAX,
                          comm);
            const auto local_root_path = static_cast<std::uint64_t>(
                guards_.maximum_root_path_length);
            std::uint64_t minimum_root_path = 0u;
            std::uint64_t maximum_root_path = 0u;
            MPI_Allreduce(&local_root_path,
                          &minimum_root_path,
                          1,
                          MPI_UINT64_T,
                          MPI_MIN,
                          comm);
            MPI_Allreduce(&local_root_path,
                          &maximum_root_path,
                          1,
                          MPI_UINT64_T,
                          MPI_MAX,
                          comm);
            const std::array<double, 3> local_guard_values{{
                static_cast<double>(
                    guards_.maximum_reference_extrapolation_distance),
                static_cast<double>(guards_.maximum_absolute_coefficient),
                static_cast<double>(guards_.maximum_row_l1_norm),
            }};
            std::array<double, 3> minimum_guard_values{};
            std::array<double, 3> maximum_guard_values{};
            MPI_Allreduce(local_guard_values.data(),
                          minimum_guard_values.data(),
                          static_cast<int>(local_guard_values.size()),
                          MPI_DOUBLE,
                          MPI_MIN,
                          comm);
            MPI_Allreduce(local_guard_values.data(),
                          maximum_guard_values.data(),
                          static_cast<int>(local_guard_values.size()),
                          MPI_DOUBLE,
                          MPI_MAX,
                          comm);
            if (all_valid == 0) {
                throw std::invalid_argument(
                    "SmallCutAggregationConstraint: aggregation boolean "
                    "environment knobs must be exactly 0 or 1 and "
                    "SVMP_AGGREGATION_MAX_LINES must be an unsigned decimal "
                    "integer in the local size_t range on every rank");
            }
            if (minimum_flags != maximum_flags ||
                minimum_max_lines != maximum_max_lines ||
                minimum_root_path != maximum_root_path ||
                minimum_guard_values != maximum_guard_values) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "inconsistent_distributed_runtime_options; aggregation "
                    "environment knobs and guard limits must have identical "
                    "values on every field-communicator rank");
            }
        }
    }
#endif
    if (!runtime_options.flags_valid || !runtime_options.max_lines_valid) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: aggregation boolean environment "
            "knobs must be exactly 0 or 1 and "
            "SVMP_AGGREGATION_MAX_LINES must be an unsigned decimal integer "
            "in the local size_t range");
    }

    std::exception_ptr local_preflight_exception;
    try {
        const auto& candidate_record = system.fieldRecord(field_);
        if (!candidate_record.space) {
            throw std::invalid_argument(
                "SmallCutAggregationConstraint: field has no function space");
        }
        const bool candidate_scalar_or_product_h1 =
            candidate_record.space->space_type() == spaces::SpaceType::H1 ||
            candidate_record.space->space_type() == spaces::SpaceType::Product;
        if (!candidate_scalar_or_product_h1 ||
            candidate_record.space->continuity() != Continuity::C0 ||
            candidate_record.space->value_dimension() !=
                candidate_record.components ||
            candidate_record.space->element().basis().is_vector_valued()) {
            throw std::invalid_argument(
                "SmallCutAggregationConstraint requires an H1/C0 scalar "
                "field or Product H1 vector field");
        }
        const auto& candidate_dh = system.fieldDofHandler(field_);
        if (candidate_dh.getEntityDofMap() == nullptr) {
            throw std::invalid_argument(
                "SmallCutAggregationConstraint: field DofHandler has no "
                "EntityDofMap");
        }
        static_cast<void>(system.meshAccess());
        static_cast<void>(system.fieldDofOffset(field_));
    } catch (...) {
        local_preflight_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_preflight_exception,
                                "field_and_dof_preflight");

    const auto& rec = system.fieldRecord(field_);
    if (!rec.space) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: field has no function space");
    }
    const bool scalar_or_product_h1 =
        rec.space->space_type() == spaces::SpaceType::H1 ||
        rec.space->space_type() == spaces::SpaceType::Product;
    if (!scalar_or_product_h1 ||
        rec.space->continuity() != Continuity::C0 ||
        rec.space->value_dimension() != rec.components ||
        rec.space->element().basis().is_vector_valued()) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint requires an H1/C0 scalar field or Product H1 vector field");
    }

    const auto& dh = system.fieldDofHandler(field_);
    const auto* entity_map = dh.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument(
            "SmallCutAggregationConstraint: field DofHandler has no EntityDofMap");
    }
    const auto n_vertices = entity_map->numVertices();

    const auto& mesh = system.meshAccess();
    const auto offset = system.fieldDofOffset(field_);
    const auto components = static_cast<std::size_t>(std::max(1, rec.components));

    // DOF resolution is cell-local: getCellDofs(cell) pairs with the cell's
    // FIELD nodes in mesh-node order (field node i holds components
    // [i*components + c]), where the field nodes are the first
    // n_field_basis entries of the cell's mesh nodes (corners-first
    // convention covers iso- and sub-parametric nodal Lagrange layouts).
    // EntityDofMap::getVertexDofs covers corner vertices only — on
    // iso-parametric Q2 meshes the midside nodes are mesh nodes whose dofs
    // live on edge entities, so the vertex lookup silently returns empty.
    // Relying on it skipped midside slaves AND dropped the midside master
    // entries of every emitted line, leaving inconsistent partial-polynomial
    // extensions.
    const auto& field_basis = rec.space->element().basis();
    const auto n_field_basis = static_cast<std::size_t>(field_basis.size());

    // The per-cell dof layout (node-major [i*components+c] vs component-major
    // [c*n_basis+i]) is a DofMap construction detail; assuming the wrong one
    // slaves the dofs of unrelated nodes. Detect it empirically from a corner
    // vertex whose dofs are independently known via EntityDofMap, and
    // cross-validate on every cell-local lookup that has entity-map coverage.
    enum class CellDofLayout { Unknown, NodeMajor, ComponentMajor };
    CellDofLayout layout = CellDofLayout::Unknown;
    auto detect_layout = [&](GlobalIndex cell,
                             std::span<const GlobalIndex> nodes) {
        const auto cell_dofs = dh.getCellDofs(cell);
        for (std::size_t i = 0; i < nodes.size() && i < n_field_basis; ++i) {
            const auto vdofs = entity_map->getVertexDofs(nodes[i]);
            if (vdofs.size() < components) {
                continue;
            }
            const auto node_major_pos = i * components;
            const auto comp_major_pos = i;
            const bool node_major_match =
                node_major_pos + components <= cell_dofs.size() &&
                cell_dofs[node_major_pos] == vdofs[0] &&
                (components < 2 || cell_dofs[node_major_pos + 1] == vdofs[1]);
            const bool comp_major_match =
                comp_major_pos + (components - 1) * n_field_basis < cell_dofs.size() &&
                cell_dofs[comp_major_pos] == vdofs[0] &&
                (components < 2 ||
                 cell_dofs[comp_major_pos + n_field_basis] == vdofs[1]);
            if (node_major_match == comp_major_match) {
                continue;  // ambiguous for this node (components==1 identical)
            }
            layout = node_major_match ? CellDofLayout::NodeMajor
                                      : CellDofLayout::ComponentMajor;
            return;
        }
    };
    std::vector<GlobalIndex> node_dof_storage(components);
    auto cell_node_dofs = [&](GlobalIndex cell,
                              std::size_t local_node) -> std::span<const GlobalIndex> {
        const auto cell_dofs = dh.getCellDofs(cell);
        if (layout == CellDofLayout::ComponentMajor) {
            if (local_node + (components - 1) * n_field_basis >=
                cell_dofs.size()) {
                return {};
            }
            for (std::size_t c = 0; c < components; ++c) {
                node_dof_storage[c] = cell_dofs[c * n_field_basis + local_node];
            }
            return std::span<const GlobalIndex>(node_dof_storage.data(), components);
        }
        const auto begin = local_node * components;
        if (begin + components > cell_dofs.size()) {
            return {};
        }
        return cell_dofs.subspan(begin, components);
    };
    auto check_cell_pairable = [&](GlobalIndex cell) {
        const auto cell_dofs = dh.getCellDofs(cell);
        if (cell_dofs.size() != n_field_basis * components) {
            throw std::invalid_argument(
                "SmallCutAggregationConstraint: field '" + rec.name +
                "' does not pair nodally with its basis (cell dofs " +
                std::to_string(cell_dofs.size()) + " != basis " +
                std::to_string(n_field_basis) + " x components " +
                std::to_string(components) + ")");
        }
    };

    // 1. Classify cells from this marker's retained volume rules.
    std::unordered_map<GlobalIndex, CellClass> cell_class;
    struct LocalActiveCellMeasure {
        std::size_t rule_count{0u};
        Real physical_volume{0.0};
        std::set<std::uint64_t> stable_rule_ids{};
    };
    std::unordered_map<GlobalIndex, LocalActiveCellMeasure>
        local_active_cell_measures;
    std::size_t active_side_volume_rules = 0u;
    // Mesh-local cell/vertex IDs are not stable communicator identifiers.
    // A C0 cell's sorted system-global field-DOF support is stable.
    auto cell_key = [&](GlobalIndex cell) {
        const auto dofs = dh.getCellDofs(cell);
        CellKey key;
        key.reserve(dofs.size());
        for (const auto dof : dofs) {
            key.push_back(offset + dof);
        }
        std::sort(key.begin(), key.end());
        return key;
    };
    std::map<CellKey, GlobalIndex> local_cell_by_key;
    std::vector<std::pair<CellKey, CellClass>> ordered_cell_classes;
    std::vector<std::int64_t> local_cell_class_words;
    std::exception_ptr local_cell_declaration_exception;
    try {
    {
        const auto& metadata = cut_context->metadata();
        const auto& rules = cut_context->volumeRules();
        const auto inactive_side =
            active_side_ == geometry::CutIntegrationSide::Negative
                ? geometry::CutIntegrationSide::Positive
                : geometry::CutIntegrationSide::Negative;
        const auto classify = [&](geometry::CutIntegrationSide side) {
            for (const auto index :
                 cut_context->generatedVolumeRuleIndicesForMarkerAndSide(
                     interface_marker_, side)) {
                if (index >= rules.size() || index >= metadata.size()) {
                    continue;
                }
                if (side == active_side_) {
                    ++active_side_volume_rules;
                }
                const auto& meta = metadata[index];
                const auto& rule = rules[index];
                const auto fraction = rule.volume_fraction;
                const auto metadata_fraction = meta.volume_fraction;
                if (!std::isfinite(fraction) || fraction < Real(0) ||
                    fraction > Real(1) ||
                    !std::isfinite(metadata_fraction) ||
                    metadata_fraction < Real(0) ||
                    metadata_fraction > Real(1)) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: diagnostic="
                        "invalid_retained_volume_fraction fractions must be "
                        "finite and lie in [0,1]");
                }
                const auto fraction_scale =
                    std::max({Real(1), std::abs(fraction),
                              std::abs(metadata_fraction)});
                if (std::abs(fraction - metadata_fraction) >
                    Real(64) * std::numeric_limits<Real>::epsilon() *
                        fraction_scale) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: diagnostic="
                        "inconsistent_retained_volume_fraction metadata and "
                        "quadrature rule fractions disagree");
                }
                const auto cell = meta.cell >= 0 ? meta.cell : meta.parent_entity;
                if (cell < 0) {
                    continue;
                }
                auto& entry = cell_class[static_cast<GlobalIndex>(cell)];
                if (side == active_side_ && rule.full_cell_equivalent) {
                    entry.full_active = true;
                }
                // Aggregation stabilizes the selected active domain. A
                // retained inactive-side complement proves that the cell was
                // classified, but it must not resurrect a pruned/absent
                // active sliver as a traversable cut cell.
                if (side == active_side_ && !rule.full_cell_equivalent &&
                    fraction > Real(0) && fraction < Real(1)) {
                    entry.cut = true;
                }
                if (side == active_side_ &&
                    (rule.full_cell_equivalent ||
                     (fraction > Real(0) && fraction < Real(1)))) {
                    const auto physical_volume =
                        physicalRetainedVolumeRuleMeasure(
                            mesh,
                            static_cast<GlobalIndex>(cell),
                            rule);
                    if (!(physical_volume > Real{0.0}) ||
                        !std::isfinite(physical_volume)) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: diagnostic="
                            "invalid_active_feature_volume retained "
                            "active-side rule has no positive physical "
                            "measure");
                    }
                    auto& measure =
                        local_active_cell_measures[
                            static_cast<GlobalIndex>(cell)];
                    const auto stable_rule_id =
                        rule.provenance.source_stable_id != 0u
                            ? rule.provenance.source_stable_id
                            : rule.provenance.cut_topology_revision;
                    if (!measure.stable_rule_ids.insert(
                            stable_rule_id).second) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: diagnostic="
                            "duplicate_active_feature_volume_rule retained "
                            "rules for one cell repeat a stable identity");
                    }
                    ++measure.rule_count;
                    measure.physical_volume += physical_volume;
                    if (!std::isfinite(measure.physical_volume)) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: diagnostic="
                            "invalid_active_feature_volume per-cell "
                            "physical measure overflow");
                    }
                }
            }
        };
        classify(active_side_);
        classify(inactive_side);
    }

    mesh.forEachCell([&](GlobalIndex cell) {
        auto key = cell_key(cell);
        if (key.empty()) {
            return;
        }
        const auto [it, inserted] =
            local_cell_by_key.emplace(std::move(key), cell);
        if (!inserted && it->second != cell) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: duplicate local cells have "
                "the same global field-DOF support");
        }
    });

    ordered_cell_classes.reserve(cell_class.size());
    for (const auto& [cell, klass] : cell_class) {
        auto key = cell_key(cell);
        if (!key.empty()) {
            ordered_cell_classes.emplace_back(std::move(key), klass);
        }
    }
    std::sort(ordered_cell_classes.begin(), ordered_cell_classes.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (const auto& [key, klass] : ordered_cell_classes) {
        const std::int64_t flags =
            (klass.full_active ? 1 : 0) | (klass.cut ? 2 : 0);
        local_cell_class_words.push_back(flags);
        local_cell_class_words.push_back(
            static_cast<std::int64_t>(key.size()));
        for (const auto dof : key) {
            local_cell_class_words.push_back(static_cast<std::int64_t>(dof));
        }
        const auto local_cell = local_cell_by_key.find(key);
        if (local_cell == local_cell_by_key.end()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: classified cell has no "
                "local cell-key resolution");
        }
        const auto cell = local_cell->second;
        const auto measure = local_active_cell_measures.find(cell);
        const bool active = klass.full_active || klass.cut;
        if (active &&
            (measure == local_active_cell_measures.end() ||
             measure->second.rule_count == 0u ||
             !(measure->second.physical_volume > Real{0.0}) ||
             !std::isfinite(measure->second.physical_volume))) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume active cell has no complete "
                "physical measure declaration");
        }
        if (!active && measure != local_active_cell_measures.end()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "inconsistent_retained_volume_fraction inactive cell has an "
                "active physical measure declaration");
        }
        const auto physical_cell_gid =
            mesh.globalEntityIdsAvailable()
                ? mesh.getCellGlobalId(cell)
                : cell;
        if (physical_cell_gid < 0) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume physical cell ID is "
                "unavailable");
        }
        const auto rule_count =
            measure == local_active_cell_measures.end()
                ? std::size_t{0u}
                : measure->second.rule_count;
        if (rule_count >
            static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume rule count exceeds the "
                "distributed wire range");
        }
        const auto physical_volume =
            measure == local_active_cell_measures.end()
                ? Real{0.0}
                : measure->second.physical_volume;
        local_cell_class_words.push_back(
            static_cast<std::int64_t>(physical_cell_gid));
        local_cell_class_words.push_back(
            static_cast<std::int64_t>(rule_count));
        local_cell_class_words.push_back(
            std::bit_cast<std::int64_t>(
                static_cast<double>(physical_volume)));
    }
    } catch (...) {
        local_cell_declaration_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_cell_declaration_exception,
                                "cell_class_and_key_serialization");

    GatheredInt64Words gathered_cell_classes;
    int aggregation_world_size = 1;
#if FE_HAS_MPI
    if (mpi_initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        MPI_Comm_size(comm, &aggregation_world_size);
        if (aggregation_world_size > 1) {
            gathered_cell_classes = allGatherInt64Words(
                comm,
                std::span<const std::int64_t>(local_cell_class_words));
        }
    }
#endif
    if (aggregation_world_size == 1) {
        gathered_cell_classes.words = local_cell_class_words;
        gathered_cell_classes.counts = {
            static_cast<int>(local_cell_class_words.size())};
        gathered_cell_classes.displacements = {0};
    }

    std::map<CellKey, CellClass> global_cell_classes;
    struct ActiveCellMeasureDeclaration {
        GlobalIndex physical_cell_gid{INVALID_GLOBAL_INDEX};
        std::size_t rule_count{0u};
        Real physical_volume{0.0};
        int provider_rank{-1};
    };
    std::map<CellKey, std::vector<ActiveCellMeasureDeclaration>>
        active_cell_measure_declarations;
    std::map<CellKey, GlobalIndex> global_physical_cell_gids;
    std::map<CellKey, Real> global_active_cell_physical_volumes;
    // A declaration is a positive classification fact even when flags==0
    // (inactive-full). Every rank retaining the same cell must therefore
    // report the exact same fact; OR-combining flags would silently accept,
    // for example, inactive-full on one rank and cut on another.
    std::map<CellKey, std::int64_t> global_cell_class_flags;
    std::uint64_t local_owned_mesh_cells = 0u;
    std::exception_ptr local_class_decode_exception;
    try {
    for (int rank = 0; rank < aggregation_world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered_cell_classes.displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered_cell_classes.counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            if (end - position < 2u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "cell-class declaration");
            }
            const auto flags = gathered_cell_classes.words[position++];
            const auto key_count = gathered_cell_classes.words[position++];
            if (flags < 0 || flags > 3 || key_count <= 0 ||
                static_cast<std::uint64_t>(key_count) >
                    static_cast<std::uint64_t>(end - position)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "cell-class payload");
            }
            CellKey key;
            key.reserve(static_cast<std::size_t>(key_count));
            for (std::int64_t i = 0; i < key_count; ++i) {
                key.push_back(static_cast<GlobalIndex>(
                    gathered_cell_classes.words[position++]));
            }
            if (end - position < 3u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "cell-measure declaration");
            }
            const auto physical_cell_gid = static_cast<GlobalIndex>(
                gathered_cell_classes.words[position++]);
            const auto rule_count_word =
                gathered_cell_classes.words[position++];
            static_assert(sizeof(double) == sizeof(std::int64_t));
            const auto physical_volume = static_cast<Real>(
                std::bit_cast<double>(
                    gathered_cell_classes.words[position++]));
            if (physical_cell_gid < 0 || rule_count_word < 0 ||
                static_cast<std::uint64_t>(rule_count_word) >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max()) ||
                !std::isfinite(physical_volume) ||
                physical_volume < Real{0.0}) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "cell-measure payload");
            }
            const bool active = (flags & 3) != 0;
            if ((active &&
                 (rule_count_word == 0 ||
                  !(physical_volume > Real{0.0}))) ||
                (!active &&
                 (rule_count_word != 0 ||
                  physical_volume != Real{0.0}))) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume cell class and physical "
                    "measure declaration disagree");
            }
            const auto [flags_it, inserted] =
                global_cell_class_flags.emplace(key, flags);
            if (!inserted && flags_it->second != flags) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "inconsistent_distributed_cell_classification field='" +
                    rec.name + "' communicator ranks reported different "
                    "class flags for the same cell");
            }
            auto& global = global_cell_classes[key];
            global.full_active = global.full_active || (flags & 1) != 0;
            global.cut = global.cut || (flags & 2) != 0;
            if (active) {
                active_cell_measure_declarations[key].push_back(
                    ActiveCellMeasureDeclaration{
                        .physical_cell_gid = physical_cell_gid,
                        .rule_count =
                            static_cast<std::size_t>(rule_count_word),
                        .physical_volume = physical_volume,
                        .provider_rank = rank,
                    });
            }
        }
    }
    const auto inconsistent_cell = std::find_if(
        global_cell_classes.begin(), global_cell_classes.end(),
        [](const auto& entry) {
            return entry.second.full_active && entry.second.cut;
        });
    if (inconsistent_cell != global_cell_classes.end()) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "inconsistent_distributed_cell_classification field='" +
            rec.name + "' communicator ranks classified the same cell as "
            "both full-active and cut");
    }

    for (const auto& [key, klass] : global_cell_classes) {
        if (!klass.full_active && !klass.cut) {
            continue;
        }
        auto declarations = active_cell_measure_declarations.find(key);
        if (declarations == active_cell_measure_declarations.end() ||
            declarations->second.empty()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume active cell has no "
                "communicator-visible measure provider");
        }
        auto& providers = declarations->second;
        std::sort(
            providers.begin(),
            providers.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.provider_rank < rhs.provider_rank;
        });
        const auto& canonical = providers.front();
        for (const auto& provider : providers) {
            if (provider.physical_cell_gid !=
                    canonical.physical_cell_gid ||
                provider.rule_count != canonical.rule_count ||
                provider.physical_volume !=
                    canonical.physical_volume) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "inconsistent_distributed_active_feature_volume "
                    "providers disagree for the same physical cell");
            }
        }
        global_physical_cell_gids.emplace(
            key, canonical.physical_cell_gid);
        global_active_cell_physical_volumes.emplace(
            key, canonical.physical_volume);
    }

    for (const auto& [key, cell] : local_cell_by_key) {
        static_cast<void>(key);
        if (mesh.isOwnedCell(cell)) {
            ++local_owned_mesh_cells;
        }
    }
    } catch (...) {
        local_class_decode_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_class_decode_exception,
                                "cell_class_decode_and_owned_count");
    if (global_cell_classes.empty()) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "missing_marker_cell_classification generated marker has no "
            "valid retained cell classifications");
    }
    std::uint64_t communicator_owned_mesh_cells = local_owned_mesh_cells;
#if FE_HAS_MPI
    if (mpi_initialized != 0 && aggregation_world_size > 1) {
        MPI_Allreduce(&local_owned_mesh_cells,
                      &communicator_owned_mesh_cells,
                      1,
                      MPI_UINT64_T,
                      MPI_SUM,
                      system.dofHandler().mpiComm());
    }
#endif
    const bool communicator_cell_classification_complete =
        communicator_owned_mesh_cells ==
        static_cast<std::uint64_t>(global_cell_classes.size());
    if (!communicator_cell_classification_complete &&
        runtime_options.max_lines ==
            std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "incomplete_distributed_aggregation_context reason="
            "global_root_traversal_requires_all_owned_cell_classification "
            "classified_cells=" +
            std::to_string(global_cell_classes.size()) +
            " communicator_owned_cells=" +
            std::to_string(communicator_owned_mesh_cells));
    }

    // Prove that every classified global cell has exactly one owner. This is
    // necessary for accepting owner-sourced topology. A separate all-owned
    // cell-count equality above proves that flags=0 inactive-full declarations
    // from the opposite-side traversal exhaust the mesh before any rootless
    // candidate may be pinned.
    std::vector<std::int64_t> local_cell_visibility_words;
    std::exception_ptr local_cell_visibility_exception;
    try {
    for (const auto& [key, cell] : local_cell_by_key) {
        if (global_cell_classes.count(key) == 0u) {
            continue;
        }
        local_cell_visibility_words.push_back(mesh.isOwnedCell(cell) ? 1 : 0);
        local_cell_visibility_words.push_back(
            static_cast<std::int64_t>(key.size()));
        for (const auto dof : key) {
            local_cell_visibility_words.push_back(
                static_cast<std::int64_t>(dof));
        }
    }
    } catch (...) {
        local_cell_visibility_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_cell_visibility_exception,
                                "cell_owner_serialization");
    GatheredInt64Words gathered_cell_visibility;
#if FE_HAS_MPI
    if (mpi_initialized != 0 && aggregation_world_size > 1) {
        gathered_cell_visibility = allGatherInt64Words(
            system.dofHandler().mpiComm(),
            std::span<const std::int64_t>(local_cell_visibility_words));
    }
#endif
    if (aggregation_world_size == 1) {
        gathered_cell_visibility.words = local_cell_visibility_words;
        gathered_cell_visibility.counts = {
            static_cast<int>(local_cell_visibility_words.size())};
        gathered_cell_visibility.displacements = {0};
    }
    std::map<CellKey, std::size_t> global_cell_owner_counts;
    const bool slave_all_cut = runtime_options.slave_all_cut;
    using FaceKey = std::vector<GlobalIndex>;
    struct OwnedCellFaces {
        CellKey cell{};
        std::vector<FaceKey> faces{};
    };
    std::vector<OwnedCellFaces> local_owned_cell_faces;
    std::vector<GlobalIndex> graph_cell_nodes;
    std::vector<std::int64_t> local_face_words;
    std::exception_ptr local_owner_face_exception;
    try {
    for (int rank = 0; rank < aggregation_world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered_cell_visibility
                .displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered_cell_visibility.counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            if (end - position < 2u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed cell-owner "
                    "declaration");
            }
            const bool owned = gathered_cell_visibility.words[position++] != 0;
            const auto key_count = gathered_cell_visibility.words[position++];
            if (key_count <= 0 ||
                static_cast<std::uint64_t>(key_count) >
                    static_cast<std::uint64_t>(end - position)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed cell-owner "
                    "payload");
            }
            CellKey key;
            key.reserve(static_cast<std::size_t>(key_count));
            for (std::int64_t i = 0; i < key_count; ++i) {
                key.push_back(static_cast<GlobalIndex>(
                    gathered_cell_visibility.words[position++]));
            }
            global_cell_owner_counts[key] += owned ? 1u : 0u;
        }
    }
    for (const auto& [key, klass] : global_cell_classes) {
        static_cast<void>(klass);
        if (global_cell_owner_counts[key] != 1u) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "incomplete_distributed_aggregation_halo reason="
                "global_cell_owner_count:" +
                std::to_string(global_cell_owner_counts[key]));
        }
    }

    // Copy communicator-global classifications onto every local geometry
    // holder, including ghosts whose cut rules were owner-filtered.
    for (const auto& [key, cell] : local_cell_by_key) {
        const auto found = global_cell_classes.find(key);
        if (found != global_cell_classes.end()) {
            cell_class[cell] = found->second;
        }
    }

    // 2. Reconstruct the classified-band face graph from owner-sourced face
    // signatures.  Relying on forEachInteriorFace() requires an undocumented
    // mutual-cell halo: if neither owner stores its neighbour, both can omit
    // their shared face and a truncated cut component can be mistaken for a
    // genuinely rootless island.  Every classified cell has exactly one owner
    // (proved above); that owner can independently emit each face's sorted
    // corner-DOF signature. Matching signatures therefore provide a complete,
    // communicator-verifiable graph induced by the classified cells without
    // any inter-owner halo assumption.
    for (const auto& [key, cell] : local_cell_by_key) {
        if (global_cell_classes.count(key) == 0u ||
            !mesh.isOwnedCell(cell)) {
            continue;
        }
        mesh.getCellNodes(cell, graph_cell_nodes);
        if (layout == CellDofLayout::Unknown && components > 1) {
            detect_layout(cell, graph_cell_nodes);
        }
        if (layout == CellDofLayout::Unknown && components > 1) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: cannot determine product-field "
                "cell DOF ordering for owner face signatures");
        }
        check_cell_pairable(cell);
        const auto ref =
            elements::ReferenceElement::create(mesh.getCellType(cell));
        OwnedCellFaces record{.cell = key};
        record.faces.reserve(ref.num_faces());
        for (std::size_t face = 0; face < ref.num_faces(); ++face) {
            FaceKey face_key;
            const auto& face_nodes = ref.face_nodes(face);
            face_key.reserve(face_nodes.size() * components);
            for (const auto local_node : face_nodes) {
                if (local_node < 0 ||
                    static_cast<std::size_t>(local_node) >=
                        graph_cell_nodes.size() ||
                    static_cast<std::size_t>(local_node) >= n_field_basis) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: classified owner "
                        "cannot form a complete face-DOF signature");
                }
                const auto node_dofs = cell_node_dofs(
                    cell, static_cast<std::size_t>(local_node));
                if (node_dofs.size() != components) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: incomplete nodal DOFs "
                        "while forming a face signature");
                }
                for (const auto dof : node_dofs) {
                    face_key.push_back(offset + dof);
                }
            }
            std::sort(face_key.begin(), face_key.end());
            if (face_key.empty() ||
                std::adjacent_find(face_key.begin(), face_key.end()) !=
                    face_key.end()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: invalid classified-cell "
                    "face-DOF signature");
            }
            record.faces.push_back(std::move(face_key));
        }
        std::sort(record.faces.begin(), record.faces.end());
        if (std::adjacent_find(record.faces.begin(), record.faces.end()) !=
            record.faces.end()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: classified cell has duplicate "
                "face-DOF signatures");
        }
        local_owned_cell_faces.push_back(std::move(record));
    }
    std::sort(local_owned_cell_faces.begin(), local_owned_cell_faces.end(),
              [](const auto& a, const auto& b) { return a.cell < b.cell; });

    // [cell_key_count, cell_key..., face_count,
    //  {face_key_count, face_key...}...]
    for (const auto& record : local_owned_cell_faces) {
        local_face_words.push_back(
            static_cast<std::int64_t>(record.cell.size()));
        for (const auto dof : record.cell) {
            local_face_words.push_back(static_cast<std::int64_t>(dof));
        }
        local_face_words.push_back(
            static_cast<std::int64_t>(record.faces.size()));
        for (const auto& face : record.faces) {
            local_face_words.push_back(
                static_cast<std::int64_t>(face.size()));
            for (const auto dof : face) {
                local_face_words.push_back(static_cast<std::int64_t>(dof));
            }
        }
    }
    } catch (...) {
        local_owner_face_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_owner_face_exception,
                                "cell_owner_decode_and_face_serialization");
    GatheredInt64Words gathered_faces;
#if FE_HAS_MPI
    if (mpi_initialized != 0 && aggregation_world_size > 1) {
        gathered_faces = allGatherInt64Words(
            system.dofHandler().mpiComm(),
            std::span<const std::int64_t>(local_face_words));
    }
#endif
    if (aggregation_world_size == 1) {
        gathered_faces.words = local_face_words;
        gathered_faces.counts = {static_cast<int>(local_face_words.size())};
        gathered_faces.displacements = {0};
    }

    std::map<CellKey, std::set<CellKey>> global_neighbors;
    std::exception_ptr local_face_decode_exception;
    try {
    std::map<CellKey, std::vector<FaceKey>> global_owned_faces;
    std::map<FaceKey, std::vector<CellKey>> cells_by_face;
    auto read_serialized_key = [](const GatheredInt64Words& gathered,
                                  std::size_t& position,
                                  std::size_t end,
                                  std::string_view kind) {
        if (position >= end) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: malformed distributed " +
                std::string(kind) + " declaration");
        }
        const auto count = gathered.words[position++];
        if (count <= 0 ||
            static_cast<std::uint64_t>(count) >
                static_cast<std::uint64_t>(end - position)) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: malformed distributed " +
                std::string(kind) + " payload");
        }
        std::vector<GlobalIndex> key;
        key.reserve(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            key.push_back(
                static_cast<GlobalIndex>(gathered.words[position++]));
        }
        if (!std::is_sorted(key.begin(), key.end()) ||
            std::adjacent_find(key.begin(), key.end()) != key.end()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: non-canonical distributed " +
                std::string(kind) + " key");
        }
        return key;
    };
    for (int rank = 0; rank < aggregation_world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered_faces.displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered_faces.counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            auto cell = read_serialized_key(
                gathered_faces, position, end, "owner-face cell");
            if (position >= end) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed owner-face "
                    "count payload");
            }
            const auto face_count = gathered_faces.words[position++];
            if (face_count <= 0 ||
                static_cast<std::uint64_t>(face_count) >
                    static_cast<std::uint64_t>(end - position)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed owner-face "
                    "record");
            }
            std::vector<FaceKey> faces;
            faces.reserve(static_cast<std::size_t>(face_count));
            for (std::int64_t face = 0; face < face_count; ++face) {
                faces.push_back(read_serialized_key(
                    gathered_faces, position, end, "owner-face signature"));
            }
            const auto [record_it, inserted] =
                global_owned_faces.emplace(cell, faces);
            if (!inserted) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: classified cell has "
                    "multiple owner-face declarations");
            }
            for (const auto& face : record_it->second) {
                cells_by_face[face].push_back(record_it->first);
            }
        }
    }
    for (const auto& [cell, klass] : global_cell_classes) {
        static_cast<void>(klass);
        if (global_owned_faces.count(cell) != 1u) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "incomplete_distributed_aggregation_halo reason="
                "missing_owned_face_signature");
        }
    }

    for (auto& [face, cells] : cells_by_face) {
        static_cast<void>(face);
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        if (cells.size() > 2u) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: non-manifold classified "
                "band face has more than two incident cells");
        }
        if (cells.size() == 2u) {
            global_neighbors[cells[0]].insert(cells[1]);
            global_neighbors[cells[1]].insert(cells[0]);
        }
    }
    } catch (...) {
        local_face_decode_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_face_decode_exception,
                                "face_ledger_decode_and_graph_construction");

    // 2b. Canonical retained active features. The graph and per-cell measure
    // declarations are communicator-global on every rank, so component
    // construction and sorted-GID accumulation require no replicated MPI
    // reduction.
    std::vector<SmallCutAggregationActiveFeatureReport>
        canonical_active_features;
    std::size_t canonical_rooted_active_feature_count = 0u;
    std::size_t canonical_rootless_active_feature_count = 0u;
    Real canonical_rootless_active_physical_volume = 0.0;
    std::exception_ptr local_feature_exception;
    try {
    std::set<GlobalIndex> active_physical_cell_gids;
    for (const auto& [cell, klass] : global_cell_classes) {
        if (!klass.full_active && !klass.cut) {
            continue;
        }
        const auto gid = global_physical_cell_gids.find(cell);
        const auto volume =
            global_active_cell_physical_volumes.find(cell);
        if (gid == global_physical_cell_gids.end() ||
            volume == global_active_cell_physical_volumes.end() ||
            gid->second < 0 ||
            !active_physical_cell_gids.insert(gid->second).second) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume communicator active cells "
                "have missing or duplicate physical cell IDs");
        }
    }
    std::set<CellKey> visited_active_cells;
    for (const auto& [seed, seed_class] : global_cell_classes) {
        if ((!seed_class.full_active && !seed_class.cut) ||
            visited_active_cells.count(seed) > 0u) {
            continue;
        }

        std::deque<CellKey> queue;
        std::vector<CellKey> component_cells;
        queue.push_back(seed);
        visited_active_cells.insert(seed);
        while (!queue.empty()) {
            auto cell = std::move(queue.front());
            queue.pop_front();
            component_cells.push_back(cell);
            const auto neighbors = global_neighbors.find(cell);
            if (neighbors == global_neighbors.end()) {
                continue;
            }
            for (const auto& neighbor : neighbors->second) {
                const auto neighbor_class =
                    global_cell_classes.find(neighbor);
                if (neighbor_class == global_cell_classes.end() ||
                    (!neighbor_class->second.full_active &&
                     !neighbor_class->second.cut)) {
                    continue;
                }
                if (visited_active_cells.insert(neighbor).second) {
                    queue.push_back(neighbor);
                }
            }
        }

        std::sort(
            component_cells.begin(),
            component_cells.end(),
            [&](const CellKey& lhs, const CellKey& rhs) {
                return std::tie(global_physical_cell_gids.at(lhs), lhs) <
                       std::tie(global_physical_cell_gids.at(rhs), rhs);
            });
        if (component_cells.empty()) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: active feature traversal "
                "produced an empty component");
        }

        SmallCutAggregationActiveFeatureReport feature;
        feature.stable_feature_id =
            global_physical_cell_gids.at(component_cells.front());
        feature.canonical_cell_count = component_cells.size();
        feature.canonical_cell_gid_digest = 14695981039346656037ull;
        long double physical_volume = 0.0L;
        GlobalIndex previous_gid = INVALID_GLOBAL_INDEX;
        for (const auto& cell : component_cells) {
            const auto physical_gid =
                global_physical_cell_gids.at(cell);
            if (physical_gid < 0 || physical_gid == previous_gid) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume active component has "
                    "missing or duplicate physical cell IDs");
            }
            previous_gid = physical_gid;
            feature.canonical_cell_gid_digest ^=
                static_cast<std::uint64_t>(physical_gid);
            feature.canonical_cell_gid_digest *= 1099511628211ull;
            const auto& klass = global_cell_classes.at(cell);
            feature.canonical_full_active_cell_count +=
                klass.full_active ? 1u : 0u;
            feature.canonical_cut_cell_count += klass.cut ? 1u : 0u;
            physical_volume += static_cast<long double>(
                global_active_cell_physical_volumes.at(cell));
        }
        if (feature.canonical_cell_count !=
            feature.canonical_full_active_cell_count +
                feature.canonical_cut_cell_count) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: active feature class "
                "accounting is inconsistent");
        }
        feature.canonical_retained_physical_volume =
            static_cast<Real>(physical_volume);
        if (!(feature.canonical_retained_physical_volume > Real{0.0}) ||
            !std::isfinite(
                feature.canonical_retained_physical_volume)) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume active component has no "
                "finite positive retained physical volume");
        }
        if (feature.canonical_full_active_cell_count > 0u) {
            feature.disposition =
                SmallCutAggregationActiveFeatureDisposition::Rooted;
            ++canonical_rooted_active_feature_count;
        } else {
            feature.disposition =
                SmallCutAggregationActiveFeatureDisposition::Rootless;
            ++canonical_rootless_active_feature_count;
        }
        canonical_active_features.push_back(std::move(feature));
    }
    std::sort(
        canonical_active_features.begin(),
        canonical_active_features.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.stable_feature_id < rhs.stable_feature_id;
        });
    if (std::adjacent_find(
            canonical_active_features.begin(),
            canonical_active_features.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.stable_feature_id == rhs.stable_feature_id;
            }) != canonical_active_features.end()) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "invalid_active_feature_volume active features have duplicate "
            "stable IDs");
    }
    long double rootless_physical_volume = 0.0L;
    for (const auto& feature : canonical_active_features) {
        if (feature.disposition ==
            SmallCutAggregationActiveFeatureDisposition::Rootless) {
            rootless_physical_volume += static_cast<long double>(
                feature.canonical_retained_physical_volume);
        }
    }
    canonical_rootless_active_physical_volume =
        static_cast<Real>(rootless_physical_volume);
    if (canonical_active_features.size() !=
            canonical_rooted_active_feature_count +
                canonical_rootless_active_feature_count ||
        !std::isfinite(
            canonical_rootless_active_physical_volume) ||
        canonical_rootless_active_physical_volume < Real{0.0}) {
        throw std::logic_error(
            "SmallCutAggregationConstraint: active feature totals are "
            "inconsistent");
    }
    } catch (...) {
        local_feature_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_feature_exception,
                                "active_feature_ledger_construction");

    // 3. Vertices of cut cells + their incident band cells.
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> vertex_cells;
    std::vector<GlobalIndex> cell_nodes;
    std::unordered_set<GlobalIndex> excluded_vertices;
    std::vector<LocalCandidateSupport> local_candidate_support;
    std::exception_ptr local_support_exception;
    try {
    for (const auto& [cell, klass] : cell_class) {
        if (!klass.cut && !klass.full_active) {
            continue;
        }
        mesh.getCellNodes(cell, cell_nodes);
        for (const auto node : cell_nodes) {
            if (node >= 0 && node < n_vertices) {
                vertex_cells[node].push_back(cell);
            }
        }
    }

    // 3b. Vertices carrying strong Dirichlet data keep their BC: collect the
    //     face vertices of every excluded boundary marker. Explicitly pinned
    //     vertices (e.g. pressure gauge nodes) are excluded for the same
    //     reason — their pin must win over aggregation.
    excluded_vertices.insert(excluded_vertices_.begin(),
                             excluded_vertices_.end());
    if (!excluded_boundary_markers_.empty()) {
        std::vector<GlobalIndex> face_cell_nodes;
        for (const auto marker : excluded_boundary_markers_) {
            mesh.forEachBoundaryFace(
                marker,
                [&](GlobalIndex face, GlobalIndex cell) {
                    const auto local_face = mesh.getLocalFaceIndex(face, cell);
                    if (local_face < 0) {
                        return;
                    }
                    const auto cell_type = mesh.getCellType(cell);
                    const auto ref = elements::ReferenceElement::create(cell_type);
                    const auto& face_nodes =
                        ref.face_nodes(static_cast<std::size_t>(local_face));
                    mesh.getCellNodes(cell, face_cell_nodes);
                    for (const auto local_node : face_nodes) {
                        if (local_node >= 0 &&
                            static_cast<std::size_t>(local_node) <
                                face_cell_nodes.size()) {
                            excluded_vertices.insert(
                                face_cell_nodes[static_cast<std::size_t>(local_node)]);
                        }
                    }
                    // ReferenceElement canonicalizes to the linear topology,
                    // so face_nodes lists corners only. Higher-order cells
                    // carry midside/face-interior nodes whose strong BC must
                    // equally win over aggregation. Classify every public
                    // mesh slot against the generic affine hull of the face
                    // corners; this covers non-axis-aligned wedge and pyramid
                    // faces as well as tensor/simplex faces.
                    if (face_cell_nodes.size() > face_nodes.size()) {
                        for (std::size_t local = 0;
                             local < face_cell_nodes.size(); ++local) {
                            if (referenceNodeOnFaceAffineHull(
                                    cell_type,
                                    std::span<const LocalIndex>(face_nodes),
                                    local,
                                    mesh.dimension())) {
                                excluded_vertices.insert(face_cell_nodes[local]);
                            }
                        }
                    }
                });
        }
    }

    // 4. Constrained vertices: vertices of cut cells with no full-active
    //    incident cell. SVMP_AGGREGATION_SLAVE_ALL_CUT=1 is a mitigation
    //    experiment knob that ALSO slaves the supported band vertices
    //    (every cut-cell vertex), constraining them to the extension of the
    //    nearest full-active cell carrying NO candidate vertex — the
    //    strong-AgFEM analogue of the band smoothing the deleted velocity
    //    ghost penalty used to provide. Roots whose own nodes are candidates
    //    would make slaves of each other's masters (constraint cycles), so
    //    the experiment requires retained layer-2 classification
    //    (Generated_interface_affected_cell_neighborhood_layers >= 1).
    // Resolve each locally classified mesh node to communicator-global field
    // DOFs before deciding whether it is a candidate.  Candidate status is an
    // OR-reduction of cut/full support and boundary exclusion facts keyed by
    // that global DOF, not a rank-local interpretation of owned cut rules.
      local_candidate_support.reserve(vertex_cells.size());
      for (const auto& [vertex, cells] : vertex_cells) {
        bool touches_cut = false;
        bool touches_full_active = false;
        for (const auto cell : cells) {
            const auto it = cell_class.find(cell);
            if (it == cell_class.end()) {
                continue;
            }
            touches_cut = touches_cut || it->second.cut;
            touches_full_active = touches_full_active || it->second.full_active;
        }

        std::vector<GlobalIndex> component_dofs;
        for (const auto cell : cells) {
            mesh.getCellNodes(cell, cell_nodes);
            if (layout == CellDofLayout::Unknown && components > 1) {
                detect_layout(cell, cell_nodes);
            }
            const auto limit = std::min(cell_nodes.size(), n_field_basis);
            for (std::size_t i = 0; i < limit; ++i) {
                if (cell_nodes[i] != vertex) {
                    continue;
                }
                check_cell_pairable(cell);
                const auto found = cell_node_dofs(cell, i);
                component_dofs.reserve(found.size());
                for (const auto dof : found) {
                    component_dofs.push_back(offset + dof);
                }
                break;
            }
            if (!component_dofs.empty()) {
                break;
            }
        }
        if (component_dofs.empty()) {
            continue;
        }
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        if (vertex_dofs.size() >= components) {
            for (std::size_t c = 0; c < components; ++c) {
                if (offset + vertex_dofs[c] != component_dofs[c]) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: cell-local dof "
                        "resolution disagrees with EntityDofMap for field '" +
                        rec.name + "' (vertex " +
                        std::to_string(vertex) + ")");
                }
            }
        }
          local_candidate_support.push_back(LocalCandidateSupport{
              .vertex = vertex,
              .declaration_dof = component_dofs.front(),
              .component_dofs = std::move(component_dofs),
              .coordinates = mesh.getNodeCoordinates(vertex),
              .touches_cut = touches_cut,
              .touches_full_active = touches_full_active,
              .excluded = excluded_vertices.count(vertex) > 0u,
          });
      }
    } catch (...) {
        local_support_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_support_exception,
                                "candidate_support_construction");

    auto global_candidates = resolveGlobalCandidateSupport(
        system,
        rec.name,
        std::span<const LocalCandidateSupport>(local_candidate_support),
        slave_all_cut);

    std::set<GlobalIndex> all_candidate_component_dofs;
    std::map<GlobalIndex, std::vector<GlobalRootCandidate>>
        global_roots_by_candidate;
    std::size_t maximum_observed_root_path = 0u;
    std::size_t root_path_guard_rejections = 0u;
    std::exception_ptr local_global_graph_exception;
    try {
    for (const auto& [dof, support] : global_candidates) {
        static_cast<void>(dof);
        all_candidate_component_dofs.insert(support.component_dofs.begin(),
                                            support.component_dofs.end());
    }
    for (auto& [dof, support] : global_candidates) {
        std::deque<std::pair<CellKey, std::size_t>> queue;
        std::set<CellKey> visited;
        for (const auto& [key, klass] : global_cell_classes) {
            if (klass.cut &&
                std::binary_search(key.begin(), key.end(), dof)) {
                queue.emplace_back(key, 0u);
                visited.insert(key);
            }
        }
        if (queue.empty()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "incomplete_distributed_aggregation_halo reason="
                "global_candidate_has_no_cut_seed dof=" +
                std::to_string(dof));
        }

        auto& roots = global_roots_by_candidate[dof];
        while (!queue.empty()) {
            auto [key, distance] = std::move(queue.front());
            queue.pop_front();
            const auto klass_it = global_cell_classes.find(key);
            if (klass_it == global_cell_classes.end()) {
                continue;
            }
            const auto& klass = klass_it->second;
            bool acceptable_root = klass.full_active;
            if (acceptable_root && slave_all_cut) {
                acceptable_root = std::none_of(
                    key.begin(), key.end(), [&](GlobalIndex root_dof) {
                        return all_candidate_component_dofs.count(root_dof) >
                               0u;
                    });
            }
            if (acceptable_root) {
                roots.push_back(GlobalRootCandidate{key, distance});
                continue;
            }
            if (!klass.cut && !(slave_all_cut && klass.full_active)) {
                continue;
            }
            const auto neighbors_it = global_neighbors.find(key);
            if (neighbors_it == global_neighbors.end()) {
                continue;
            }
            for (const auto& next : neighbors_it->second) {
                if (visited.insert(next).second) {
                    queue.emplace_back(next, distance + 1u);
                }
            }
        }
        std::sort(roots.begin(), roots.end(),
                  [](const GlobalRootCandidate& a,
                     const GlobalRootCandidate& b) {
                      return std::tie(a.distance, a.key) <
                             std::tie(b.distance, b.key);
                  });
        roots.erase(std::unique(
                        roots.begin(), roots.end(),
                        [](const GlobalRootCandidate& a,
                           const GlobalRootCandidate& b) {
                            return a.key == b.key;
                        }),
                    roots.end());
        const bool had_unguarded_root = !roots.empty();
        for (const auto& root : roots) {
            maximum_observed_root_path =
                std::max(maximum_observed_root_path, root.distance);
        }
        const auto first_rejected = std::remove_if(
            roots.begin(), roots.end(), [&](const auto& root) {
                return root.distance > guards_.maximum_root_path_length;
            });
        root_path_guard_rejections += static_cast<std::size_t>(
            std::distance(first_rejected, roots.end()));
        roots.erase(first_rejected, roots.end());
        if (had_unguarded_root && roots.empty()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "root_path_guard_rejection candidate_dof=" +
                std::to_string(dof) + " maximum_observed_path=" +
                std::to_string(maximum_observed_root_path) +
                " maximum_allowed_path=" +
                std::to_string(guards_.maximum_root_path_length));
        }
        support.globally_rooted = !roots.empty();
    }

    } catch (...) {
        local_global_graph_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_global_graph_exception,
                                "global_candidate_root_traversal");

    std::size_t constrained_vertex_count = 0u;
    std::size_t aggregated = 0u;
    std::size_t no_root = 0u;
    std::size_t island_pinned_dofs = 0u;
    std::size_t inversion_failed = 0u;
    std::size_t non_field_nodes = 0u;
    std::size_t empty_lines = 0u;
    std::size_t extrapolation_guard_rejections = 0u;
    std::size_t line_guard_rejections = 0u;
    Real maximum_observed_reference_extrapolation = 0.0;
    Real maximum_observed_absolute_coefficient = 0.0;
    Real maximum_observed_row_l1_norm = 0.0;
    std::vector<Real> values;
    std::vector<std::pair<GlobalIndex, double>> pending_entries;
    std::vector<GlobalIndex> current_slaves;
    std::vector<GlobalIndex> root_check_nodes;
    LocalAggregationDeclarationMap local_aggregation_declarations;
    const bool allow_unaggregated = runtime_options.allow_unaggregated;
    const std::size_t max_lines = runtime_options.max_lines;
    std::exception_ptr local_proposal_exception;
    try {

    for (const auto& support : local_candidate_support) {
        const auto global_it = global_candidates.find(support.declaration_dof);
        if (global_it != global_candidates.end() && support.touches_cut) {
            ++constrained_vertex_count;
        }
    }

    // The root provider needs the field's complete nodal support. Reject a
    // sub-parametric cell before proposal generation instead of emitting a
    // partial polynomial extension.
    for (const auto& [cell, klass] : cell_class) {
        if (!klass.cut && !klass.full_active) {
            continue;
        }
        mesh.getCellNodes(cell, cell_nodes);
        if (cell_nodes.size() < n_field_basis) {
            throw std::invalid_argument(
                "SmallCutAggregationConstraint: field '" + rec.name +
                "' has sub-vertex dofs that are not mesh nodes (cell " +
                std::to_string(cell) + " carries " +
                std::to_string(cell_nodes.size()) + " mesh nodes < " +
                std::to_string(n_field_basis) +
                " field basis functions); sub-parametric fields are not "
                "supported — use an iso-parametric mesh for higher-order "
                "aggregation");
        }
    }

    std::unordered_map<GlobalIndex,
                       std::shared_ptr<const geometry::GeometryMapping>> mappings;
    // Read per apply() call (not a cached static): constraint refreshes are
    // infrequent, and in-process A/B tests must be able to flip the knob.
    const bool use_linear_extension = runtime_options.linear_extension;
    auto extension_element_type = rec.space->element_type();
    std::unique_ptr<basis::LagrangeBasis> linear_extension_basis;
    if (use_linear_extension) {
        // element_type() reports the cell TOPOLOGY type, which is already the
        // corner type for sub-parametric layouts (a Q2 field on Quad4
        // topology), so the A/B gate must compare basis sizes, not topology
        // names: engage whenever a strictly smaller corner sub-basis exists.
        const auto corner_type = linearElementType(extension_element_type);
        auto corner_basis = std::make_unique<basis::LagrangeBasis>(corner_type, 1);
        if (static_cast<std::size_t>(corner_basis->size()) < n_field_basis) {
            extension_element_type = corner_type;
            linear_extension_basis = std::move(corner_basis);
        }
    }
    const basis::BasisFunction& extension_basis =
        linear_extension_basis ? static_cast<const basis::BasisFunction&>(*linear_extension_basis)
                               : field_basis;
    const auto n_extension = static_cast<std::size_t>(extension_basis.size());
    const auto& field_mesh_to_basis =
        basis::ReferenceNodeLayout::mesh_to_basis_ordering(extension_element_type);

    no_root = static_cast<std::size_t>(std::count_if(
        global_candidates.begin(), global_candidates.end(),
        [&](const auto& entry) {
            return !entry.second.globally_rooted &&
                   !(slave_all_cut && entry.second.has_full_support);
        }));

    // Every root in the communicator-global cut component is considered.
    // Only ranks that actually hold a root cell generate its mapping and
    // interpolation weights. Mapping/line failure invalidates that proposal,
    // not the candidate; the global resolver fails closed only when no valid,
    // globally available proposal remains for a globally rooted candidate.
    for (const auto& [declaration_dof, candidate] : global_candidates) {
        if (aggregated >= max_lines) {
            break;
        }
        if (!candidate.globally_rooted) {
            continue;
        }
        bool candidate_emitted = false;
        const auto roots_it = global_roots_by_candidate.find(declaration_dof);
        if (roots_it == global_roots_by_candidate.end()) {
            continue;
        }
        for (const auto& root_candidate : roots_it->second) {
            const auto local_root = local_cell_by_key.find(root_candidate.key);
            if (local_root == local_cell_by_key.end()) {
                continue;
            }
            const auto root = local_root->second;
            mesh.getCellNodes(root, root_check_nodes);
            check_cell_pairable(root);
            const auto n_declared_root_nodes = std::min(
                {root_check_nodes.size(), n_field_basis, n_extension});
            std::vector<GlobalIndex> declared_root_masters;
            declared_root_masters.reserve(n_declared_root_nodes);
            for (std::size_t node = 0; node < n_declared_root_nodes; ++node) {
                const auto master_dofs = cell_node_dofs(root, node);
                if (!master_dofs.empty()) {
                    declared_root_masters.push_back(offset + master_dofs.front());
                }
            }
            std::sort(declared_root_masters.begin(),
                      declared_root_masters.end());
            declared_root_masters.erase(
                std::unique(declared_root_masters.begin(),
                            declared_root_masters.end()),
                declared_root_masters.end());

            auto mapping_it = mappings.find(root);
            if (mapping_it == mappings.end()) {
                mapping_it =
                    mappings.emplace(root, makeMapping(mesh, root)).first;
            }
            const auto& mapping = mapping_it->second;
            Vector<Real, 3> xi{};
            if (mapping == nullptr ||
                !invertMapping(*mapping, candidate.coordinates,
                               mesh.dimension(), xi)) {
                ++inversion_failed;
                continue;
            }
            const Real reference_extrapolation =
                normalizedReferenceExtrapolationDistance(
                    mesh.getCellType(root), xi);
            maximum_observed_reference_extrapolation = std::max(
                maximum_observed_reference_extrapolation,
                reference_extrapolation);
            if (!std::isfinite(reference_extrapolation) ||
                reference_extrapolation >
                    guards_.maximum_reference_extrapolation_distance +
                        Real{128.0} * std::numeric_limits<Real>::epsilon() *
                            std::max(
                                Real{1.0},
                                guards_
                                    .maximum_reference_extrapolation_distance)) {
                ++extrapolation_guard_rejections;
                continue;
            }
            extension_basis.evaluate_values(xi, values);
            mesh.getCellNodes(root, cell_nodes);
            const auto n_root_field_nodes = std::min(
                {cell_nodes.size(), n_field_basis, n_extension});
            if (values.size() < n_root_field_nodes ||
                (!field_mesh_to_basis.empty() &&
                 field_mesh_to_basis.size() < n_root_field_nodes)) {
                ++inversion_failed;
                continue;
            }

            AggregationDeclaration declaration{
                .state = AggregationDeclarationState::Rooted,
                .root_distance = root_candidate.distance,
                .root_cell_gid = mesh.globalEntityIdsAvailable()
                                     ? mesh.getCellGlobalId(root)
                                     : root,
                .component_dofs = candidate.component_dofs,
                .root_master_dofs = std::move(declared_root_masters)};
            bool proposal_valid = true;
            for (std::size_t component = 0;
                 component < candidate.component_dofs.size(); ++component) {
                const auto slave = candidate.component_dofs[component];
                ConstraintLine line;
                line.slave_dof = slave;
                for (std::size_t m = 0; m < n_root_field_nodes; ++m) {
                    const auto master_dofs = cell_node_dofs(root, m);
                    const auto basis_index = field_mesh_to_basis.empty()
                                                 ? m
                                                 : field_mesh_to_basis[m];
                    if (component >= master_dofs.size() ||
                        basis_index >= values.size()) {
                        continue;
                    }
                    const auto master = offset + master_dofs[component];
                    const auto weight =
                        static_cast<double>(values[basis_index]);
                    if (master != slave && std::isfinite(weight) &&
                        std::abs(weight) >= 1e-12) {
                        line.entries.push_back({master, weight});
                    } else if (!std::isfinite(weight)) {
                        proposal_valid = false;
                    }
                }
                const auto line_validation =
                    normalizeAndValidateRootedLine(line);
                if (!line_validation.valid) {
                    proposal_valid = false;
                    break;
                }
                const auto line_guard =
                    validateAggregationLineGuards(line, guards_);
                maximum_observed_absolute_coefficient = std::max(
                    maximum_observed_absolute_coefficient,
                    line_guard.maximum_absolute_coefficient);
                maximum_observed_row_l1_norm = std::max(
                    maximum_observed_row_l1_norm,
                    line_guard.row_l1_norm);
                if (!line_guard.valid) {
                    ++line_guard_rejections;
                    proposal_valid = false;
                    break;
                }
                declaration.lines.push_back(std::move(line));
            }
            if (!proposal_valid ||
                declaration.lines.size() != candidate.component_dofs.size()) {
                ++empty_lines;
                continue;
            }
            local_aggregation_declarations[declaration_dof].push_back(
                std::move(declaration));
            candidate_emitted = true;
        }
        if (candidate_emitted) {
            ++aggregated;
        }
    }
    } catch (...) {
        local_proposal_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_proposal_exception,
                                "root_proposal_construction");

    std::size_t communicator_root_proposal_failures =
        inversion_failed + empty_lines + extrapolation_guard_rejections +
        line_guard_rejections;
    std::size_t communicator_extrapolation_guard_rejections =
        extrapolation_guard_rejections;
    std::size_t communicator_line_guard_rejections = line_guard_rejections;
#if FE_HAS_MPI
    if (mpi_initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        int world_size = 1;
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            const auto local_failures = static_cast<std::uint64_t>(
                communicator_root_proposal_failures);
            std::uint64_t summed_failures = 0u;
            MPI_Allreduce(&local_failures,
                          &summed_failures,
                          1,
                          MPI_UINT64_T,
                          MPI_SUM,
                          comm);
            communicator_root_proposal_failures =
                static_cast<std::size_t>(summed_failures);
            const std::array<std::uint64_t, 2> local_guard_rejections{{
                static_cast<std::uint64_t>(
                    extrapolation_guard_rejections),
                static_cast<std::uint64_t>(line_guard_rejections),
            }};
            std::array<std::uint64_t, 2> summed_guard_rejections{};
            MPI_Allreduce(local_guard_rejections.data(),
                          summed_guard_rejections.data(),
                          static_cast<int>(summed_guard_rejections.size()),
                          MPI_UINT64_T,
                          MPI_SUM,
                          comm);
            communicator_extrapolation_guard_rejections =
                static_cast<std::size_t>(summed_guard_rejections[0]);
            communicator_line_guard_rejections =
                static_cast<std::size_t>(summed_guard_rejections[1]);
            const std::array<double, 3> local_guard_maxima{{
                static_cast<double>(
                    maximum_observed_reference_extrapolation),
                static_cast<double>(maximum_observed_absolute_coefficient),
                static_cast<double>(maximum_observed_row_l1_norm),
            }};
            std::array<double, 3> global_guard_maxima{};
            MPI_Allreduce(local_guard_maxima.data(),
                          global_guard_maxima.data(),
                          static_cast<int>(global_guard_maxima.size()),
                          MPI_DOUBLE,
                          MPI_MAX,
                          comm);
            maximum_observed_reference_extrapolation =
                static_cast<Real>(global_guard_maxima[0]);
            maximum_observed_absolute_coefficient =
                static_cast<Real>(global_guard_maxima[1]);
            maximum_observed_row_l1_norm =
                static_cast<Real>(global_guard_maxima[2]);
        }
    }
#endif

    DistributedAggregationResult distributed_result;
    if (max_lines == std::numeric_limits<std::size_t>::max()) {
        distributed_result = resolveDistributedAggregationDeclarations(
            system,
            constraints,
            rec.name,
            global_candidates,
            local_aggregation_declarations,
            slave_all_cut,
            allow_unaggregated);
    } else {
        distributed_result.validation =
            DistributedAggregationValidation::DebugBypass;
        // Preserve the debug line-cap contract: no global completion is
        // attempted while intentional partial coverage is requested.
        for (const auto& [dof, declarations] : local_aggregation_declarations) {
            static_cast<void>(dof);
            const auto declaration = std::find_if(
                declarations.begin(), declarations.end(),
                [](const AggregationDeclaration& candidate) {
                    return candidate.state ==
                           AggregationDeclarationState::Rooted;
                });
            if (declaration == declarations.end()) {
                continue;
            }
            for (const auto& line : declaration->lines) {
                if (!constraints.isConstrained(line.slave_dof)) {
                    distributed_result.relevant_lines.push_back(line);
                }
            }
        }
    }
    for (const auto& line : distributed_result.relevant_lines) {
        if (constraints.isConstrained(line.slave_dof)) {
            continue;
        }
        if (!line.entries.empty()) {
            const auto guard =
                validateAggregationLineGuards(line, guards_);
            if (!guard.valid) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "canonical_line_guard_rejection slave_dof=" +
                    std::to_string(line.slave_dof) + " reason=" +
                    guard.reason);
            }
        }
        constraints.addConstraintLine(line);
        current_slaves.push_back(line.slave_dof);
        if (line.entries.empty()) {
            ++island_pinned_dofs;
        }
    }
    const auto distributed_validation = distributed_result.validation;

    {
        static const bool dump_probe = [] {
            const char* env = std::getenv("SVMP_AGGREGATION_DUMP");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }();
        if (dump_probe) {
            for (const GlobalIndex probe : {GlobalIndex(1657), GlobalIndex(1342),
                                            GlobalIndex(363), GlobalIndex(365),
                                            GlobalIndex(474), GlobalIndex(476)}) {
                if (probe >= 0 && probe < n_vertices) {
                    const auto xyz = mesh.getNodeCoordinates(probe);
                    std::ostringstream probe_line;
                    probe_line << "SmallCutAggregationConstraint: probe vertex="
                               << probe << " xyz=(" << xyz[0] << "," << xyz[1]
                               << "," << xyz[2] << ")";
                    FE_LOG_INFO(probe_line.str());
                }
            }
        }
    }

    std::ostringstream oss;
    oss << "SmallCutAggregationConstraint: diagnostic=small_cut_aggregation"
        << " field='" << rec.name << "'"
        << " marker=" << interface_marker_
        << " active_side="
        << (active_side_ == geometry::CutIntegrationSide::Negative ? "Negative"
                                                                   : "Positive")
        << " band_cells=" << cell_class.size()
        << " active_side_volume_rules=" << active_side_volume_rules
        << " candidate_vertices=" << constrained_vertex_count
        << " local_candidate_vertices=" << constrained_vertex_count
        // Preserve the original diagnostic keys for log consumers while the
        // explicitly scoped keys below make their rank-local/global meaning
        // unambiguous.
        << " aggregated_vertices=" << aggregated
        << " vertices_without_root=" << no_root
        << " island_pinned_dofs=" << island_pinned_dofs
        << " inversion_failures=" << inversion_failed
        << " empty_line_failures=" << empty_lines
        << " local_root_proposal_candidates=" << aggregated
        << " communicator_rootless_candidates=" << no_root
        << " local_island_pinned_dofs=" << island_pinned_dofs
        << " local_root_proposal_inversion_failures=" << inversion_failed
        << " non_field_nodes=" << non_field_nodes
        << " local_root_proposal_line_failures=" << empty_lines
        << " communicator_root_proposal_failures="
        << communicator_root_proposal_failures
        << " maximum_root_path_length="
        << guards_.maximum_root_path_length
        << " maximum_observed_root_path="
        << maximum_observed_root_path
        << " root_path_guard_rejections="
        << root_path_guard_rejections
        << " maximum_reference_extrapolation_distance="
        << guards_.maximum_reference_extrapolation_distance
        << " maximum_observed_reference_extrapolation="
        << maximum_observed_reference_extrapolation
        << " communicator_extrapolation_guard_rejections="
        << communicator_extrapolation_guard_rejections
        << " maximum_absolute_coefficient="
        << guards_.maximum_absolute_coefficient
        << " maximum_observed_absolute_coefficient="
        << maximum_observed_absolute_coefficient
        << " maximum_row_l1_norm="
        << guards_.maximum_row_l1_norm
        << " maximum_observed_row_l1_norm="
        << maximum_observed_row_l1_norm
        << " communicator_line_guard_rejections="
        << communicator_line_guard_rejections
        << " canonical_candidate_vertices="
        << distributed_result.canonical_candidate_vertices
        << " canonical_rooted_candidate_vertices="
        << distributed_result.canonical_rooted_candidate_vertices
        << " canonical_rootless_candidate_vertices="
        << distributed_result.canonical_rootless_candidate_vertices
        << " canonical_owned_aggregate_dofs="
        << distributed_result.canonical_owned_aggregate_dofs
        << " canonical_owned_pinned_dofs="
        << distributed_result.canonical_owned_pinned_dofs
        << " canonical_strong_suppressed_dofs="
        << distributed_result.canonical_strong_suppressed_dofs
        << " canonical_active_feature_count="
        << canonical_active_features.size()
        << " canonical_rooted_active_feature_count="
        << canonical_rooted_active_feature_count
        << " canonical_rootless_active_feature_count="
        << canonical_rootless_active_feature_count
        << " canonical_rootless_active_physical_volume="
        << canonical_rootless_active_physical_volume
        << " local_relevant_lines_installed=" << current_slaves.size()
        << " distributed_halo_validation="
        << (distributed_validation == DistributedAggregationValidation::Passed
                ? "passed"
                : distributed_validation ==
                          DistributedAggregationValidation::DebugBypass
                      ? "debug_bypass"
                      : "not_parallel")
        << " excluded_dirichlet_vertices=" << excluded_vertices.size()
        << " pruned_volume_rules=" << cut_context->generatedPrunedVolumeRuleCount()
        << " pruned_volume_measure=" << cut_context->generatedPrunedVolumeMeasure();
    FE_LOG_INFO(oss.str());

    {
        // Churn is computed from the replicated canonical slave set, not the
        // overlap-dependent locally relevant set. State belongs to this
        // constraint instance, whose lifetime is tied to the FESystem; this
        // avoids stale process-global state when a system address is reused.
        const auto& canonical_slaves = distributed_result.canonical_slaves;
        const auto& previous = previous_canonical_slaves_;
        std::size_t entered = 0u;
        std::size_t left = 0u;
        std::size_t i = 0u;
        std::size_t j = 0u;
        while (i < canonical_slaves.size() || j < previous.size()) {
            if (j >= previous.size() ||
                (i < canonical_slaves.size() &&
                 canonical_slaves[i] < previous[j])) {
                ++entered;
                ++i;
            } else if (i >= canonical_slaves.size() ||
                       previous[j] < canonical_slaves[i]) {
                ++left;
                ++j;
            } else {
                ++i;
                ++j;
            }
        }
        std::ostringstream churn;
        churn << "SmallCutAggregationConstraint: diagnostic=small_cut_aggregation_churn"
              << " field='" << rec.name << "'"
              << " canonical_slaves=" << canonical_slaves.size()
              << " slaves=" << canonical_slaves.size()
              << " entered=" << entered
              << " left=" << left;
        FE_LOG_INFO(churn.str());
        previous_canonical_slaves_ = canonical_slaves;
    }

    // Root-proposal failures are diagnostic at proposal granularity: another
    // root/rank may still provide a valid canonical line. The resolver above
    // fails closed only when a globally rooted candidate has no usable
    // proposal. The debug allow-unaggregated knob restores fail-open behavior.
    if (max_lines == std::numeric_limits<std::size_t>::max()) {
        if (allow_unaggregated) {
            if (no_root > 0 || communicator_root_proposal_failures > 0) {
                FE_LOG_WARNING(
                    "SmallCutAggregationConstraint: communicator-rootless " +
                    std::to_string(no_root) +
                    ", invalid root proposals " +
                    std::to_string(communicator_root_proposal_failures) +
                    "; continuing "
                    "fail-open because SVMP_AGGREGATION_ALLOW_UNAGGREGATED "
                    "is set (" + oss.str() + ")");
            }
        }
    }

    if (distributed_validation !=
        DistributedAggregationValidation::DebugBypass) {
        completed_refresh_report_ = SmallCutAggregationRefreshReport{
            .field = field_,
            .active_side = active_side_,
            .interface_marker = interface_marker_,
            .maximum_root_path_length =
                guards_.maximum_root_path_length,
            .maximum_observed_root_path =
                maximum_observed_root_path,
            .root_path_guard_rejections =
                root_path_guard_rejections,
            .maximum_reference_extrapolation_distance =
                guards_.maximum_reference_extrapolation_distance,
            .maximum_observed_reference_extrapolation =
                maximum_observed_reference_extrapolation,
            .extrapolation_guard_rejections =
                communicator_extrapolation_guard_rejections,
            .maximum_absolute_coefficient =
                guards_.maximum_absolute_coefficient,
            .maximum_observed_absolute_coefficient =
                maximum_observed_absolute_coefficient,
            .maximum_row_l1_norm =
                guards_.maximum_row_l1_norm,
            .maximum_observed_row_l1_norm =
                maximum_observed_row_l1_norm,
            .line_guard_rejections =
                communicator_line_guard_rejections,
            .canonical_candidate_vertices =
                distributed_result.canonical_candidate_vertices,
            .canonical_rooted_candidate_vertices =
                distributed_result.canonical_rooted_candidate_vertices,
            .canonical_rootless_candidate_vertices =
                distributed_result.canonical_rootless_candidate_vertices,
            .canonical_owned_aggregate_dofs =
                distributed_result.canonical_owned_aggregate_dofs,
            .canonical_owned_pinned_dofs =
                distributed_result.canonical_owned_pinned_dofs,
            .canonical_strong_suppressed_dofs =
                distributed_result.canonical_strong_suppressed_dofs,
            .canonical_active_feature_count =
                canonical_active_features.size(),
            .canonical_rooted_active_feature_count =
                canonical_rooted_active_feature_count,
            .canonical_rootless_active_feature_count =
                canonical_rootless_active_feature_count,
            .canonical_rootless_active_physical_volume =
                canonical_rootless_active_physical_volume,
            .canonical_active_features =
                canonical_active_features,
        };
    }
}

bool SmallCutAggregationConstraint::updateValues(const systems::FESystem& /*system*/,
                                                 AffineConstraints& /*constraints*/,
                                                 double /*time*/,
                                                 double /*dt*/)
{
    return false;
}

ConstraintDependencyDeclaration SmallCutAggregationConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = ISystemConstraint::dependencyDeclaration();
    out.structural.fe_constraint_layout = true;
    out.structural.mesh_field_layout = true;
    out.structural.mesh_field_values = true;
    out.structural.active_configuration = true;
    return out;
}

systems::SetupStorageRequirements
SmallCutAggregationConstraint::storageRequirements() const noexcept
{
    systems::SetupStorageRequirements req;
    req.entity_dof_map = true;
    req.vertex_topology = true;
    req.cell_topology = true;
    req.interior_face_topology = true;
    return req;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
