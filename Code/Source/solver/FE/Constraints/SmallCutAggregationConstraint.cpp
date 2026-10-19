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
#include "Constraints/SmallCutAggregationCellIndex.h"
#include "Core/AggregationGuardDiagnostics.h"
#include "Core/FEException.h"
#include "Core/HaloDiagnostics.h"
#include "Core/Logger.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/FrameGeometry.h"
#include "Geometry/GeometryFrameUtils.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

namespace detail {

struct SmallCutAggregationPendingProlongation {
    FieldId field{INVALID_FIELD_ID};
    geometry::CutIntegrationSide active_side{
        geometry::CutIntegrationSide::Negative};
    int interface_marker{-1};
    bool slave_all_cut{false};
    bool linear_extension{false};
    bool allow_unaggregated{false};
    bool trace_bound_eligible{false};
    SmallCutAggregationGuardOptions guards{};
    std::uint64_t cut_context_content_revision{0u};
    bool has_free_surface_snapshot_revision{false};
    std::uint64_t free_surface_snapshot_revision{0u};
    bool has_source_value_revision{false};
    std::uint64_t source_value_revision{0u};
    std::vector<SmallCutAggregationProlongationRow> rows{};
    std::vector<SmallCutAggregationProlongationCell> active_cells{};
    std::vector<SmallCutAggregationProlongationPatch> patches{};

    // Sorted (DOF, active cell GID) incidences and (cell GID, feature) pairs
    // of `active_cells`, built by the first finalizeProlongationReport() of
    // a refresh.  They depend only on the cell GIDs, field DOFs and feature
    // IDs, which a reuse copies unchanged (it refreshes volumes and rule
    // identities only), so reused pendings share the index.
    struct ActiveCellIndex {
        std::size_t cell_count{0u};
        std::vector<std::pair<GlobalIndex, GlobalIndex>> cell_gids_by_dof{};
        std::vector<std::pair<GlobalIndex, GlobalIndex>> feature_by_cell_gid{};
    };
    mutable std::shared_ptr<const ActiveCellIndex> active_cell_index{};
};

/**
 * Rank-local key of everything apply() reads, except the retained physical
 * volumes and retained-rule identities, which a reuse recomputes:
 *  - `topology`: for both sides of the marker, the ordered generated volume
 *    rules as the classification reads them (cell, retained/full/cut flags,
 *    availability of the stable rule identity);
 *  - `inputs`: the field and its DOF layout content, the system DOF
 *    partition and assembly ghost policy, the mesh revision tuple, the set
 *    of DOFs already constrained when apply() runs, the excluded markers and
 *    vertices, the guards and the runtime options.
 * Each word is mixed into two independent 64-bit digests.
 */
struct SmallCutAggregationReuseKey {
    std::array<std::uint64_t, 2> topology{};
    std::array<std::uint64_t, 2> inputs{};

    [[nodiscard]] bool operator==(
        const SmallCutAggregationReuseKey&) const noexcept = default;
};

// Communicator-canonical per-cell data that a reuse refreshes, aligned with
// the recorded canonical active cells (sorted by physical cell ID).
struct SmallCutAggregationRetainedMeasures {
    std::vector<Real> volumes{};
    std::vector<std::vector<std::uint64_t>> stable_rule_ids{};
};

struct SmallCutAggregationReuseInputs {
    SmallCutAggregationReuseKey key{};
    // Field-independent digests that identify the per-cell retained volumes
    // and rule identities (shared by the velocity and pressure constraints
    // of one marker): the classification sequence, and the retained active
    // rules' identities, points, weights, cell types, cell node coordinates
    // and physical cell IDs.
    std::array<std::uint64_t, 2> classification{};
    std::array<std::uint64_t, 2> geometry_digest{};
};

/**
 * Outputs of one successful full refresh, keyed by its inputs. The lines,
 * canonical ledgers, rows, cells and patches are pure functions of the key;
 * retained volumes and rule identities, geometry identity, lineage, topology
 * transition, churn and the publication ordinal are recomputed by
 * publishReusedRefresh().
 */
struct SmallCutAggregationReuseRecord {
    SmallCutAggregationReuseKey key{};
    std::string field_name{};
    std::vector<ConstraintLine> added_lines{};
    // The refresh diagnostic split around its two input-dependent parts.
    std::string diagnostic_head{};
    std::string diagnostic_middle{};
    SmallCutAggregationRefreshReport report{};
    std::shared_ptr<const SmallCutAggregationPendingProlongation> pending{};
    std::vector<GlobalIndex> canonical_slaves{};
};

namespace {
std::atomic<bool> owner_sourced_refresh_enabled{true};
std::atomic<std::uint64_t> owner_sourced_refreshes_completed{0u};
std::atomic<std::uint64_t> owner_sourced_refreshes_returned{0u};
} // namespace

void setSmallCutAggregationOwnerSourcedRefresh(bool enabled) noexcept
{
    owner_sourced_refresh_enabled.store(enabled, std::memory_order_relaxed);
}

bool smallCutAggregationOwnerSourcedRefreshEnabled() noexcept
{
    return owner_sourced_refresh_enabled.load(std::memory_order_relaxed);
}

SmallCutAggregationOwnerSourcedCounters
smallCutAggregationOwnerSourcedCounters() noexcept
{
    return SmallCutAggregationOwnerSourcedCounters{
        .completed =
            owner_sourced_refreshes_completed.load(std::memory_order_relaxed),
        .returned_to_global =
            owner_sourced_refreshes_returned.load(std::memory_order_relaxed),
    };
}

namespace {
void noteSmallCutAggregationOwnerSourcedRefresh(bool completed) noexcept
{
    (completed ? owner_sourced_refreshes_completed
               : owner_sourced_refreshes_returned)
        .fetch_add(1u, std::memory_order_relaxed);
}
} // namespace

} // namespace detail

static_assert(std::is_nothrow_move_assignable_v<
              std::optional<SmallCutAggregationRefreshReport>>);
static_assert(std::is_nothrow_move_constructible_v<
              std::optional<SmallCutAggregationRefreshReport>>);
static_assert(std::is_nothrow_move_assignable_v<std::shared_ptr<
              const detail::SmallCutAggregationPendingProlongation>>);
static_assert(noexcept(std::declval<std::vector<GlobalIndex>&>().swap(
    std::declval<std::vector<GlobalIndex>&>())));
static_assert(noexcept(
    std::declval<AffineConstraints&>() =
        std::declval<AffineConstraints&&>()));

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
    // Cells of the canonical active feature of the candidate's cut seed
    // (diagnostics of the guard-rootless policy).
    std::size_t feature_cells{0u};
};

// (reason, candidate DOF, candidate, detail) of a guard-rootless candidate.
using GuardRootlessLogger = std::function<void(std::string_view,
                                               GlobalIndex,
                                               const GlobalCandidateSupport&,
                                               const std::string&)>;

using CellKey = std::vector<GlobalIndex>;

// Root candidate of the global cut-band search: `cell` indexes the sorted
// communicator-global classified cells (see ClassifiedCellIndex), so
// (distance, cell) orders exactly like (distance, key).
struct GlobalRootCandidate {
    std::size_t cell{0u};
    std::size_t distance{std::numeric_limits<std::size_t>::max()};
};

// Sorted cell-key tables (see SmallCutAggregationCellIndex.h).
using LocalCellKeyTable = detail::SmallCutAggregationLocalCellTable;
using ClassifiedCellIndex = detail::SmallCutAggregationCellIndex;

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
    const std::size_t dimension =
        static_cast<std::size_t>(cell_dimension);
    const std::size_t expected_rank =
        dimension - 1u;
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
        for (std::size_t d = 0; d < dimension; ++d) {
            residual[d] = point[d] - origin[d];
            original_norm_sq += residual[d] * residual[d];
        }
        face_scale = std::max(face_scale, std::sqrt(original_norm_sq));
        for (std::size_t basis_index = 0; basis_index < rank; ++basis_index) {
            Real projection = Real{0};
            for (std::size_t d = 0; d < dimension; ++d) {
                projection += residual[d] * tangents[basis_index][d];
            }
            for (std::size_t d = 0; d < dimension; ++d) {
                residual[d] -= projection * tangents[basis_index][d];
            }
        }
        Real residual_norm_sq = Real{0};
        for (std::size_t d = 0; d < dimension; ++d) {
            residual_norm_sq += residual[d] * residual[d];
        }
        const Real residual_norm = std::sqrt(residual_norm_sq);
        if (residual_norm <= rank_tolerance * face_scale) {
            continue;
        }
        for (std::size_t d = 0; d < dimension; ++d) {
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
    for (std::size_t d = 0; d < dimension; ++d) {
        residual[d] = point[d] - origin[d];
        point_norm_sq += residual[d] * residual[d];
    }
    for (std::size_t basis_index = 0; basis_index < rank; ++basis_index) {
        Real projection = Real{0};
        for (std::size_t d = 0; d < dimension; ++d) {
            projection += residual[d] * tangents[basis_index][d];
        }
        for (std::size_t d = 0; d < dimension; ++d) {
            residual[d] -= projection * tangents[basis_index][d];
        }
    }
    Real residual_norm_sq = Real{0};
    for (std::size_t d = 0; d < dimension; ++d) {
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

// A candidate whose cut feature holds full cells only beyond the root-path
// guard, or whose every root proposal fails the extrapolation or coefficient
// guards, has no admissible extension.  By default the refresh fails closed
// and names the opt-in.  With SmallCutAggregationGuardOptions::
// rootless_fallback the candidate receives the rootless-island policy of a
// feature without full cells (decision D33); every such case is logged once
// per refresh (by the first rank) and counted in the run summary
// (diagnostics::aggregationGuardRootlessSummary), so a result that relies on
// it is flagged.
constexpr const char* kGuardRootlessOptInHint =
    "the candidate's cut feature has full cells only beyond the aggregation "
    "guards (for example a wetting film or wedge at a wall); set "
    "<Small_cut_aggregation_rootless_fallback>true</Small_cut_aggregation_rootless_fallback> "
    "on the free-surface boundary condition to continue with the "
    "rootless-island policy (opt-in: it pins velocity and pressure of those "
    "candidates, which degrades results inside connected liquid; every case "
    "is logged and counted in the run summary)";

void logGuardRootlessCandidate(const systems::FESystem& system,
                               std::string_view field_name,
                               std::string_view reason,
                               GlobalIndex candidate_dof,
                               const std::array<Real, 3>& coordinates,
                               std::size_t feature_cells,
                               const std::string& detail)
{
    int rank = 0;
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Comm_rank(system.dofHandler().mpiComm(), &rank);
    }
#endif
    if (rank != 0) {
        return;
    }
    std::ostringstream oss;
    oss << std::setprecision(17)
        << "SmallCutAggregationConstraint: diagnostic=aggregation_guard_rootless"
        << " reason=" << reason << " field='" << field_name << "'"
        << " candidate_dof=" << candidate_dof << " xyz=(" << coordinates[0]
        << "," << coordinates[1] << "," << coordinates[2] << ")"
        << " feature_cells=" << feature_cells;
    const auto time = system.lastConstraintUpdateTime();
    if (time.has_value()) {
        oss << " constraint_time=" << *time;
    }
    oss << " " << detail << " policy=rootless_island";
    FE_LOG_WARNING(oss.str());
}

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

// Diagnostic only: SVMP_AGGREGATION_TRACE_PROPOSALS=1 logs every root
// proposal and its outcome.  It changes no constraint.
[[nodiscard]] bool aggregationProposalTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* value = std::getenv("SVMP_AGGREGATION_TRACE_PROPOSALS");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
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
    std::size_t canonical_halo_limited_root_choices{0u};
    std::size_t canonical_row_coupled_slaves_beyond_halo{0u};
    // Globally rooted candidates without any proposal inside the guards that
    // received the rootless-island policy (D33).
    std::size_t canonical_guard_rootless_candidate_vertices{0u};
    std::vector<SmallCutAggregationProlongationRow> canonical_rows{};
};

struct GatheredInt64Words {
    std::vector<std::int64_t> words{};
    std::vector<int> counts{};
    std::vector<int> displacements{};
};

// ---------------------------------------------------------------------------
// Diagnostic profile of one refresh (SVMP_AGGREGATION_PROFILE=1).  It times
// the phases of apply() and counts the words each distributed exchange sends
// and receives, and prints one communicator-reduced line per refresh from
// rank 0.  It changes no constraint.  The environment variable must have the
// same value on every rank, because the reduction is collective.

enum class AggregationProfilePhase : std::size_t {
    Preflight,
    ReuseKey,
    ReusePublication,
    Classification,
    CellClassExchange,
    CellOwnerExchange,
    FaceGraph,
    Features,
    CandidateSupport,
    RootTraversal,
    RootProposals,
    Resolution,
    Publication,
    Count,
};

enum class AggregationProfileExchange : std::size_t {
    CandidateSupport,
    RootProposals,
    ProposalVisibility,
    LineVisibility,
    ReusedMeasures,
    CellClasses,
    CellOwners,
    Faces,
    OwnerRows,
    CellCensus,
    CellCensusReplies,
    FaceCensus,
    ActiveCellLedger,
    CandidateCensus,
    MergedCandidates,
    ProposalOutcomes,
    PreferredProposals,
    ResolutionVisibility,
    Count,
};

enum class AggregationProfileCount : std::size_t {
    LocalCells,
    OwnedCells,
    LocalClassifiedCells,
    CommunicatorClassifiedCells,
    LocalCandidateSupports,
    CommunicatorCandidates,
    LocalRootProposals,
    OwnerSourcedRefresh,
    Count,
};

constexpr std::array<const char*,
                     static_cast<std::size_t>(AggregationProfilePhase::Count)>
    kAggregationProfilePhaseNames{{
        "preflight",
        "reuse_key",
        "reuse_publication",
        "classification",
        "cell_class_exchange",
        "cell_owner_exchange",
        "face_graph",
        "features",
        "candidate_support",
        "root_traversal",
        "root_proposals",
        "resolution",
        "publication",
    }};

constexpr std::array<const char*,
                     static_cast<std::size_t>(AggregationProfileExchange::Count)>
    kAggregationProfileExchangeNames{{
        "candidate_support",
        "root_proposals",
        "proposal_visibility",
        "line_visibility",
        "reused_measures",
        "cell_classes",
        "cell_owners",
        "faces",
        "owner_rows",
        "cell_census",
        "cell_census_replies",
        "face_census",
        "active_cell_ledger",
        "candidate_census",
        "merged_candidates",
        "proposal_outcomes",
        "preferred_proposals",
        "resolution_visibility",
    }};

constexpr std::array<const char*,
                     static_cast<std::size_t>(AggregationProfileCount::Count)>
    kAggregationProfileCountNames{{
        "local_cells",
        "owned_cells",
        "local_classified_cells",
        "communicator_classified_cells",
        "local_candidate_supports",
        "communicator_candidates",
        "local_root_proposals",
        "owner_sourced_refresh",
    }};

[[nodiscard]] bool aggregationProfileEnabled() noexcept
{
    static const bool enabled = [] {
        const char* value = std::getenv("SVMP_AGGREGATION_PROFILE");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
}

class AggregationRefreshProfile {
public:
    using Clock = std::chrono::steady_clock;

    void begin() noexcept
    {
        phase_ms_.fill(0.0);
        exchange_ms_.fill(0.0);
        sent_words_.fill(0u);
        received_words_.fill(0u);
        exchange_partners_.fill(0u);
        counts_.fill(0u);
        start_ = Clock::now();
        last_ = start_;
        active_ = aggregationProfileEnabled();
    }

    [[nodiscard]] bool active() const noexcept { return active_; }

    void mark(AggregationProfilePhase phase) noexcept
    {
        if (!active_) {
            return;
        }
        const auto now = Clock::now();
        phase_ms_[static_cast<std::size_t>(phase)] +=
            std::chrono::duration<double, std::milli>(now - last_).count();
        last_ = now;
    }

    void recordExchange(AggregationProfileExchange exchange,
                        std::uint64_t sent,
                        std::uint64_t received,
                        std::uint64_t partners,
                        double milliseconds) noexcept
    {
        if (!active_) {
            return;
        }
        const auto index = static_cast<std::size_t>(exchange);
        sent_words_[index] += sent;
        received_words_[index] += received;
        exchange_partners_[index] =
            std::max(exchange_partners_[index], partners);
        exchange_ms_[index] += milliseconds;
    }

    void setCount(AggregationProfileCount count, std::uint64_t value) noexcept
    {
        if (active_) {
            counts_[static_cast<std::size_t>(count)] = value;
        }
    }

    // Collective when active on a distributed communicator.
    void report(const systems::FESystem& system,
                std::string_view field_name,
                std::string_view kind);

private:
    bool active_{false};
    Clock::time_point start_{};
    Clock::time_point last_{};
    std::array<double, static_cast<std::size_t>(AggregationProfilePhase::Count)>
        phase_ms_{};
    std::array<double,
               static_cast<std::size_t>(AggregationProfileExchange::Count)>
        exchange_ms_{};
    std::array<std::uint64_t,
               static_cast<std::size_t>(AggregationProfileExchange::Count)>
        sent_words_{};
    std::array<std::uint64_t,
               static_cast<std::size_t>(AggregationProfileExchange::Count)>
        received_words_{};
    std::array<std::uint64_t,
               static_cast<std::size_t>(AggregationProfileExchange::Count)>
        exchange_partners_{};
    std::array<std::uint64_t,
               static_cast<std::size_t>(AggregationProfileCount::Count)>
        counts_{};
};

AggregationRefreshProfile& aggregationRefreshProfile()
{
    static AggregationRefreshProfile profile;
    return profile;
}

void AggregationRefreshProfile::report(const systems::FESystem& system,
                                       std::string_view field_name,
                                       std::string_view kind)
{
    if (!active_) {
        return;
    }
    active_ = false;
    const double total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start_)
            .count();
    constexpr auto n_phases =
        static_cast<std::size_t>(AggregationProfilePhase::Count);
    constexpr auto n_exchanges =
        static_cast<std::size_t>(AggregationProfileExchange::Count);
    constexpr auto n_counts =
        static_cast<std::size_t>(AggregationProfileCount::Count);
    // [total, phases..., exchange times...] reduced by MAX and SUM;
    // [sent..., received..., partners..., counts...] reduced by MAX and SUM.
    std::vector<double> times;
    times.reserve(1u + n_phases + n_exchanges);
    times.push_back(total_ms);
    times.insert(times.end(), phase_ms_.begin(), phase_ms_.end());
    times.insert(times.end(), exchange_ms_.begin(), exchange_ms_.end());
    std::vector<std::uint64_t> words;
    words.reserve(3u * n_exchanges + n_counts);
    words.insert(words.end(), sent_words_.begin(), sent_words_.end());
    words.insert(words.end(), received_words_.begin(), received_words_.end());
    words.insert(words.end(), exchange_partners_.begin(),
                 exchange_partners_.end());
    words.insert(words.end(), counts_.begin(), counts_.end());
    auto times_max = times;
    auto times_sum = times;
    auto words_max = words;
    auto words_sum = words;
    int rank = 0;
    int world_size = 1;
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            MPI_Allreduce(times.data(), times_max.data(),
                          static_cast<int>(times.size()), MPI_DOUBLE,
                          MPI_MAX, comm);
            MPI_Allreduce(times.data(), times_sum.data(),
                          static_cast<int>(times.size()), MPI_DOUBLE,
                          MPI_SUM, comm);
            MPI_Allreduce(words.data(), words_max.data(),
                          static_cast<int>(words.size()), MPI_UINT64_T,
                          MPI_MAX, comm);
            MPI_Allreduce(words.data(), words_sum.data(),
                          static_cast<int>(words.size()), MPI_UINT64_T,
                          MPI_SUM, comm);
        }
    }
#else
    static_cast<void>(system);
#endif
    if (rank != 0) {
        return;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "SmallCutAggregationConstraint: diagnostic=aggregation_profile"
        << " field='" << field_name << "' kind=" << kind
        << " ranks=" << world_size << " total_ms_max=" << times_max[0]
        << " total_ms_mean=" << times_sum[0] / world_size;
    for (std::size_t i = 0; i < n_phases; ++i) {
        if (times_max[1u + i] == 0.0) {
            continue;
        }
        out << " phase." << kAggregationProfilePhaseNames[i]
            << "_ms_max=" << times_max[1u + i];
    }
    for (std::size_t i = 0; i < n_exchanges; ++i) {
        if (words_sum[i] == 0u && words_sum[n_exchanges + i] == 0u &&
            times_max[1u + n_phases + i] == 0.0) {
            continue;
        }
        const char* name = kAggregationProfileExchangeNames[i];
        out << " exchange." << name << "_ms_max="
            << times_max[1u + n_phases + i] << " exchange." << name
            << "_sent_words_sum=" << words_sum[i] << " exchange." << name
            << "_sent_words_max=" << words_max[i] << " exchange." << name
            << "_received_words_max=" << words_max[n_exchanges + i]
            << " exchange." << name << "_partners_max="
            << words_max[2u * n_exchanges + i];
    }
    for (std::size_t i = 0; i < n_counts; ++i) {
        out << " count." << kAggregationProfileCountNames[i]
            << "_max=" << words_max[3u * n_exchanges + i] << " count."
            << kAggregationProfileCountNames[i]
            << "_sum=" << words_sum[3u * n_exchanges + i];
    }
    FE_LOG_INFO(out.str());
}

#if FE_HAS_MPI

[[nodiscard]] GatheredInt64Words allGatherInt64Words(
    MPI_Comm comm,
    std::span<const std::int64_t> local_words,
    AggregationProfileExchange exchange)
{
    const auto profile_start = AggregationRefreshProfile::Clock::now();
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
    aggregationRefreshProfile().recordExchange(
        exchange,
        static_cast<std::uint64_t>(local_count),
        static_cast<std::uint64_t>(total),
        static_cast<std::uint64_t>(world_size - 1),
        std::chrono::duration<double, std::milli>(
            AggregationRefreshProfile::Clock::now() - profile_start)
            .count());
    return gathered;
}

/**
 * Sparse exchange: `outgoing[d]` is sent to rank d.  The result holds the
 * received words grouped by source rank in ascending order, laid out like an
 * all-gather (`counts` and `displacements` per source rank).  Collective;
 * allocation failures are coordinated like allGatherInt64Words.
 */
[[nodiscard]] GatheredInt64Words exchangeInt64Words(
    MPI_Comm comm,
    const std::vector<std::vector<std::int64_t>>& outgoing,
    AggregationProfileExchange exchange)
{
    const auto profile_start = AggregationRefreshProfile::Clock::now();
    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    const auto n_ranks = static_cast<std::size_t>(world_size);
    int local_overflow = outgoing.size() != n_ranks ? 1 : 0;
    std::uint64_t sent_total = 0u;
    std::uint64_t partners = 0u;
    for (const auto& words : outgoing) {
        sent_total += words.size();
        partners += words.empty() ? 0u : 1u;
        if (words.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            local_overflow = 1;
        }
    }
    if (sent_total >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        local_overflow = 1;
    }
    int global_overflow = 0;
    MPI_Allreduce(&local_overflow, &global_overflow, 1, MPI_INT, MPI_MAX, comm);
    if (global_overflow != 0) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: distributed aggregation exchange "
            "exceeds the MPI count range");
    }
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
            std::string("SmallCutAggregationConstraint: distributed exchange ") +
            phase + " failed on another communicator rank");
    };
    std::vector<int> send_counts;
    std::vector<int> send_displacements;
    std::vector<std::int64_t> send_words;
    GatheredInt64Words received;
    std::exception_ptr send_exception;
    try {
        send_counts.assign(n_ranks, 0);
        send_displacements.assign(n_ranks, 0);
        send_words.reserve(static_cast<std::size_t>(sent_total));
        for (std::size_t rank = 0u; rank < n_ranks; ++rank) {
            send_counts[rank] = static_cast<int>(outgoing[rank].size());
            send_displacements[rank] = static_cast<int>(send_words.size());
            send_words.insert(send_words.end(),
                              outgoing[rank].begin(),
                              outgoing[rank].end());
        }
        received.counts.assign(n_ranks, 0);
        received.displacements.assign(n_ranks, 0);
    } catch (...) {
        send_exception = std::current_exception();
    }
    coordinate_allocation(send_exception, "send-layout allocation");
    MPI_Alltoall(send_counts.data(),
                 1,
                 MPI_INT,
                 received.counts.data(),
                 1,
                 MPI_INT,
                 comm);
    std::int64_t total = 0;
    std::exception_ptr receive_exception;
    try {
        for (std::size_t rank = 0u; rank < n_ranks; ++rank) {
            if (total > std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: received distributed "
                    "aggregation exchange exceeds the MPI displacement range");
            }
            received.displacements[rank] = static_cast<int>(total);
            total += received.counts[rank];
        }
        if (total > std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: received distributed "
                "aggregation exchange exceeds the MPI displacement range");
        }
        received.words.resize(static_cast<std::size_t>(total));
    } catch (...) {
        receive_exception = std::current_exception();
    }
    coordinate_allocation(receive_exception, "receive-layout allocation");
    MPI_Alltoallv(send_words.data(),
                  send_counts.data(),
                  send_displacements.data(),
                  MPI_INT64_T,
                  received.words.data(),
                  received.counts.data(),
                  received.displacements.data(),
                  MPI_INT64_T,
                  comm);
    aggregationRefreshProfile().recordExchange(
        exchange,
        sent_total,
        static_cast<std::uint64_t>(total),
        partners,
        std::chrono::duration<double, std::milli>(
            AggregationRefreshProfile::Clock::now() - profile_start)
            .count());
    return received;
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
                comm, std::span<const std::int64_t>(local_words),
                AggregationProfileExchange::CandidateSupport);
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

// Merges the line entries of a rooted proposal and validates it: a nonnegative
// root cell, one valid partition-of-unity line per component DOF.
[[nodiscard]] bool normalizeRootProposal(AggregationDeclaration& normalized)
{
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
    return valid;
}

// Normalizes the local root proposals of communicator candidates (dropping
// invalid ones: another root or provider may still furnish the candidate)
// and serializes them in (candidate, distance, root cell) order:
// [base, state, distance, root_cell_gid, n_components, n_roots, n_lines,
//  components..., roots..., {slave,n_entries,{master,weight_bits}...}...]
// `payload_valid` turns false on a non-finite weight.
void serializeRootProposals(const LocalAggregationDeclarationMap& local_candidates,
                            const GlobalCandidateMap& global_candidates,
                            std::vector<std::int64_t>& local_candidate_words,
                            bool& local_candidate_payload_valid)
{
    std::size_t local_declaration_count = 0u;
    std::vector<AggregationDeclaration> normalized_local_declarations;
    std::vector<std::pair<GlobalIndex, const AggregationDeclaration*>> ordered;
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
            const bool valid = normalizeRootProposal(normalized);
            // A bad root proposal is not a communicator failure if another
            // root/provider can furnish a valid extension for the candidate.
            if (!valid) {
                if (aggregationProposalTraceEnabled()) {
                    FE_LOG_INFO(
                        "SmallCutAggregationConstraint: diagnostic="
                        "aggregation_proposal_trace stage=normalization "
                        "candidate_dof=" + std::to_string(dof) +
                        " root_cell_gid=" +
                        std::to_string(normalized.root_cell_gid) +
                        " outcome=dropped");
                }
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
}

struct RankedDeclaration {
    int rank{-1};
    AggregationDeclaration declaration{};
};
using RankedDeclarationMap =
    std::map<GlobalIndex, std::vector<RankedDeclaration>>;

// Decodes serialized root proposals (serializeRootProposals) of every source
// rank, in source-rank order, revalidating each one; valid ones are appended
// to their candidate.  `provider_rank_of(source rank, root cell ID)` names
// the provider rank recorded with the proposal.
template <class ProviderRankOf>
void decodeRootProposals(const GatheredInt64Words& gathered_candidates,
                         int world_size,
                         const ProviderRankOf& provider_rank_of,
                         RankedDeclarationMap& declarations_by_dof)
{
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
            ranked.declaration.state =
                static_cast<AggregationDeclarationState>(raw_state);
            ranked.declaration.root_distance =
                raw_distance < 0
                    ? std::numeric_limits<std::size_t>::max()
                    : static_cast<std::size_t>(raw_distance);
            ranked.declaration.root_cell_gid =
                static_cast<GlobalIndex>(raw_root_cell_gid);
            ranked.rank = provider_rank_of(
                rank, ranked.declaration.root_cell_gid);
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
}

struct ProposalDofVisibility {
    int rank{-1};
    bool owned{false};
    bool preconstrained{false};
    bool needed{false};
};

/**
 * Visibility of proposal DOFs gathered from every rank on which they are
 * locally relevant: (rank, owned, preconstrained, assembled-with) per rank,
 * in rank order.
 */
class GatheredProposalVisibility {
public:
    explicit GatheredProposalVisibility(
        const std::map<GlobalIndex, std::vector<ProposalDofVisibility>>&
            by_dof)
        : by_dof_(by_dof)
    {
    }

    // A strong constraint on any relevant copy.
    [[nodiscard]] bool preconstrained(GlobalIndex dof) const
    {
        const auto found = by_dof_.find(dof);
        return found != by_dof_.end() &&
               std::any_of(
                   found->second.begin(), found->second.end(),
                   [](const ProposalDofVisibility& visibility) {
                       return visibility.preconstrained;
                   });
    }

    [[nodiscard]] bool relevantOnRank(GlobalIndex dof, int rank) const
    {
        const auto found = by_dof_.find(dof);
        return found != by_dof_.end() &&
               std::any_of(
                   found->second.begin(), found->second.end(),
                   [&](const ProposalDofVisibility& visibility) {
                       return visibility.rank == rank;
                   });
    }

    [[nodiscard]] std::size_t ownerCount(GlobalIndex dof) const
    {
        const auto found = by_dof_.find(dof);
        return found == by_dof_.end()
                   ? std::size_t{0}
                   : static_cast<std::size_t>(std::count_if(
                         found->second.begin(), found->second.end(),
                         [](const ProposalDofVisibility& visibility) {
                             return visibility.owned;
                         }));
    }

    [[nodiscard]] int ownerRank(GlobalIndex dof) const
    {
        const auto found = by_dof_.find(dof);
        if (found == by_dof_.end()) {
            return -1;
        }
        int owner = -1;
        for (const auto& visibility : found->second) {
            if (!visibility.owned) {
                continue;
            }
            if (owner != -1) {
                return -1;
            }
            owner = visibility.rank;
        }
        return owner;
    }

    // Ranks on which the slave is relevant, in rank order; null when none.
    [[nodiscard]] const std::vector<ProposalDofVisibility>* slaveVisibility(
        GlobalIndex dof) const
    {
        const auto found = by_dof_.find(dof);
        return found == by_dof_.end() ? nullptr : &found->second;
    }

private:
    const std::map<GlobalIndex, std::vector<ProposalDofVisibility>>& by_dof_;
};

struct CanonicalSelection {
    std::map<GlobalIndex, ConstraintLine> canonical_lines{};
    std::map<GlobalIndex, SmallCutAggregationProlongationRow>
        canonical_row_metadata{};
    std::set<GlobalIndex> expected_slaves{};
    std::ostringstream failures{};
    std::size_t failure_count{0u};
    // Globally rooted candidates without any proposal inside the guards
    // (fail closed unless rootless_fallback): not a halo defect.
    std::ostringstream guard_failures{};
    std::size_t guard_failure_count{0u};
    std::size_t halo_limited_root_choices{0u};
    std::size_t guard_rootless_candidate_vertices{0u};
};

/**
 * Canonical root choice of every communicator candidate from its root
 * proposals and the proposal-DOF visibility `oracle` (see
 * GatheredProposalVisibility for the queries).  The choice depends only on
 * the partition-independent (distance, physical root cell ID) order and on
 * which assembling ranks carry the masters; failures are counted in the
 * selection (throwCanonicalSelectionFailures reports them).
 */
template <class Oracle>
void selectCanonicalAggregationLines(
    std::string_view field_name,
    const GlobalCandidateMap& global_candidates,
    const RankedDeclarationMap& declarations_by_dof,
    const Oracle& oracle,
    bool slave_all_cut,
    bool allow_unaggregated,
    bool rootless_fallback,
    const GuardRootlessLogger& log_guard_rootless,
    CanonicalSelection& selection)
{
    static_cast<void>(field_name);
    auto& canonical_lines = selection.canonical_lines;
    auto& canonical_row_metadata = selection.canonical_row_metadata;
    auto& expected_slaves = selection.expected_slaves;
    auto& failures = selection.failures;
    auto& failure_count = selection.failure_count;
    auto& guard_failures = selection.guard_failures;
    auto& guard_failure_count = selection.guard_failure_count;
    constexpr double line_tolerance = 1.0e-12;
    for (const auto& [dof, support] : global_candidates) {
        const auto candidate_dof = dof;
        const auto* candidate_components =
            &support.component_dofs;
        auto make_row =
            [candidate_dof, candidate_components](
                GlobalIndex slave,
                SmallCutAggregationProvisionalRowKind kind) {
                const auto component_it = std::find(
                    candidate_components->begin(),
                    candidate_components->end(),
                    slave);
                if (component_it == candidate_components->end()) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: canonical slave is "
                        "absent from its candidate component list");
                }
                SmallCutAggregationProlongationRow row;
                row.candidate_dof = candidate_dof;
                row.component = static_cast<std::size_t>(
                    std::distance(candidate_components->begin(),
                                  component_it));
                row.slave_dof = slave;
                row.provisional_kind = kind;
                return row;
            };
        auto record_row =
            [&](SmallCutAggregationProlongationRow row) {
                const auto [it, inserted] =
                    canonical_row_metadata.emplace(
                        row.slave_dof, std::move(row));
                if (!inserted) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: duplicate canonical "
                        "candidate-row provenance for slave " +
                        std::to_string(it->first));
                }
            };
        const bool all_components_preconstrained = std::all_of(
            support.component_dofs.begin(), support.component_dofs.end(),
            [&](GlobalIndex slave) {
                return oracle.preconstrained(slave);
            });
        if (all_components_preconstrained) {
            expected_slaves.insert(support.component_dofs.begin(),
                                   support.component_dofs.end());
            for (const auto slave : support.component_dofs) {
                auto row = make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::
                        PreexistingConstraint);
                row.preconstrained_at_apply = true;
                record_row(std::move(row));
            }
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
                    for (const auto slave : support.component_dofs) {
                        record_row(make_row(
                            slave,
                            SmallCutAggregationProvisionalRowKind::
                                UnaggregatedFreeIdentity));
                    }
                    continue;
                }
                if (!rootless_fallback) {
                    ++guard_failure_count;
                    if (guard_failure_count <= 4u) {
                        guard_failures
                            << " dof=" << dof
                            << " reason=no_valid_root_proposal xyz=("
                            << support.coordinates[0] << ","
                            << support.coordinates[1] << ","
                            << support.coordinates[2] << ")"
                            << " feature_cells=" << support.feature_cells
                            << ";";
                    }
                    continue;
                }
                // Opt-in fallback: rootless-island policy below (D33).
                ++selection.guard_rootless_candidate_vertices;
                log_guard_rootless(
                    "proposal_guard",
                    dof,
                    support,
                    "detail=no_root_proposal_inside_guards");
            }
            if (slave_all_cut && support.has_full_support) {
                // Globally supported slave-all candidate with no foreign root:
                // it remains free by the documented experimental policy.
                for (const auto slave : support.component_dofs) {
                    record_row(make_row(
                        slave,
                        SmallCutAggregationProvisionalRowKind::
                            SupportedFreeIdentity));
                }
                continue;
            }
            if (!allow_unaggregated) {
                for (const auto slave : support.component_dofs) {
                    expected_slaves.insert(slave);
                    canonical_lines.emplace(
                        slave, ConstraintLine{.slave_dof = slave});
                    record_row(make_row(
                        slave,
                        SmallCutAggregationProvisionalRowKind::
                            RootlessHomogeneousPin));
                }
            } else {
                for (const auto slave : support.component_dofs) {
                    record_row(make_row(
                        slave,
                        SmallCutAggregationProvisionalRowKind::
                            UnaggregatedFreeIdentity));
                }
            }
            continue;
        }

        std::vector<const RankedDeclaration*> available_rooted;
        // Proposals that are complete but whose masters some assembling rank
        // cannot see: the geometrically preferred root may be among them.
        std::vector<const RankedDeclaration*> halo_eligible_rooted;
        std::string first_unavailable_reason;
        // Per-proposal reasons, reported only when no proposal is available.
        std::ostringstream unavailable_proposals;
        std::size_t unavailable_proposal_count = 0u;
        for (const auto* declaration : rooted) {
            bool available = true;
            bool halo_eligible = false;
            const auto reason_count_before = first_unavailable_reason.size();
            std::string proposal_reason;
            for (const auto slave : support.component_dofs) {
                if (oracle.preconstrained(slave)) {
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
                        oracle.ownerCount(entry.master_dof);
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
                const auto* slave_visibility =
                    oracle.slaveVisibility(slave);
                if (slave_visibility == nullptr) {
                    available = false;
                    if (first_unavailable_reason.empty()) {
                        first_unavailable_reason =
                            " reason=canonical_slave_not_relevant";
                    }
                    break;
                }
                // Every rank that assembles with the slave's line must carry
                // all masters. Ranks that merely see the slave deeper in the
                // halo carry no line and impose nothing. (The legacy
                // fail-open debug mode keeps its stricter all-relevant-ranks
                // rule.)
                for (const auto& visibility : *slave_visibility) {
                    if (!allow_unaggregated && !visibility.needed) {
                        continue;
                    }
                    const bool masters_visible = std::all_of(
                        line_it->entries.begin(), line_it->entries.end(),
                        [&](const auto& entry) {
                            return oracle.relevantOnRank(entry.master_dof,
                                                  visibility.rank);
                        });
                    if (!masters_visible) {
                        available = false;
                        halo_eligible = true;
                        proposal_reason =
                            "master_not_relevant:rank" +
                            std::to_string(visibility.rank);
                        if (first_unavailable_reason.empty()) {
                            first_unavailable_reason =
                                " reason=canonical_master_not_relevant rank=" +
                                std::to_string(visibility.rank);
                        }
                        break;
                    }
                }
                if (!available) {
                    break;
                }
            }
            if (available) {
                available_rooted.push_back(declaration);
            } else if (halo_eligible) {
                halo_eligible_rooted.push_back(declaration);
            }
            if (!available && unavailable_proposal_count < 6u) {
                ++unavailable_proposal_count;
                const auto key = physicalRootKey(declaration->declaration);
                unavailable_proposals
                    << " [root_cell_gid=" << key.cell_gid
                    << " distance=" << key.distance
                    << " provider_rank=" << declaration->rank << " "
                    << (!proposal_reason.empty()
                            ? proposal_reason
                            : first_unavailable_reason.size() >
                                      reason_count_before
                                  ? first_unavailable_reason.substr(1)
                                  : std::string("unavailable"))
                    << "]";
            }
        }
        if (available_rooted.empty()) {
            if (allow_unaggregated) {
                for (const auto slave : support.component_dofs) {
                    record_row(make_row(
                        slave,
                        SmallCutAggregationProvisionalRowKind::
                            UnaggregatedFreeIdentity));
                }
                continue;
            }
            ++failure_count;
            if (failure_count <= 4u) {
                failures << " dof=" << dof
                         << (first_unavailable_reason.empty()
                                 ? " reason=no_globally_available_root"
                                 : first_unavailable_reason)
                         << " xyz=(" << support.coordinates[0] << ","
                         << support.coordinates[1] << ","
                         << support.coordinates[2] << ")"
                         << " proposals=" << rooted.size()
                         << unavailable_proposals.str() << ";";
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
        // The (distance, root cell GID) order is partition independent; when
        // the halo hides the preferred root from an assembling rank, the
        // next visible root is used and the choice becomes partition
        // dependent. Count it so the diagnostic says so.
        if (std::any_of(halo_eligible_rooted.begin(),
                        halo_eligible_rooted.end(),
                        [&](const auto* hidden) {
                            return detail::smallCutAggregationPhysicalRootLess(
                                physicalRootKey(hidden->declaration),
                                hidden->rank,
                                physicalRootKey(
                                    available_rooted.front()->declaration),
                                available_rooted.front()->rank);
                        })) {
            ++selection.halo_limited_root_choices;
        }
        const auto* chosen_provider = available_rooted.front();
        const auto& chosen = chosen_provider->declaration;
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
                auto row = make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::RootedExtension);
                row.root_cell_gid = chosen.root_cell_gid;
                row.root_provider_rank = chosen_provider->rank;
                row.root_distance = chosen.root_distance;
                row.provisional_entries = selected_line->entries;
                record_row(std::move(row));
            } else if (!allow_unaggregated) {
                // Keep the slave in the availability phase: a globally
                // pre-existing strong constraint legitimately explains the
                // missing aggregation line; otherwise this is a fail-closed
                // machinery/overlap error.
                expected_slaves.insert(slave);
                auto row = make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::RootedExtension);
                row.root_cell_gid = chosen.root_cell_gid;
                row.root_provider_rank = chosen_provider->rank;
                row.root_distance = chosen.root_distance;
                record_row(std::move(row));
            } else {
                record_row(make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::
                        UnaggregatedFreeIdentity));
            }
        }
    }

}

void throwCanonicalSelectionFailures(std::string_view field_name,
                                     const CanonicalSelection& selection)
{
    if (selection.failure_count > 0u) {
        throw InsufficientGhostHaloError(
            "SmallCutAggregationConstraint: diagnostic="
            "incomplete_distributed_aggregation_halo field='" +
            std::string(field_name) + "' inconsistent_candidate_dofs=" +
            std::to_string(selection.failure_count) + selection.failures.str() +
            " Communicator-global candidate/root construction produced "
            "inconsistent component/weight data or no proposed root whose "
            "masters are relevant on every slave-relevant rank; refusing "
            "owner-wins constraint resolution." +
            (selection.guard_failure_count > 0u
                 ? " Also " + std::to_string(selection.guard_failure_count) +
                       " candidate(s) without a root proposal inside the "
                       "guards:" + selection.guard_failures.str()
                 : std::string{}));
    }
    if (selection.guard_failure_count > 0u) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "aggregation_no_root_inside_guards field='" +
            std::string(field_name) + "' candidates=" +
            std::to_string(selection.guard_failure_count) + selection.guard_failures.str() +
            " Every root proposal of these globally rooted candidates fails "
            "the aggregation guards or the reference-coordinate inversion "
            "(the aggregation_root_proposal_outcomes lines name each "
            "proposal and why it failed): " + kGuardRootlessOptInHint + ".");
    }
}

struct CanonicalLineVisibility {
    int rank{-1};
    bool owned{false};
    bool preconstrained{false};
    bool needed{false};
    std::vector<GlobalIndex> missing_masters{};
};

/**
 * Visibility of the canonical slaves gathered from every rank on which they
 * are locally relevant, with the masters of the canonical line that rank
 * lacks.
 */
class GatheredLineVisibility {
public:
    explicit GatheredLineVisibility(
        const std::map<GlobalIndex, std::vector<CanonicalLineVisibility>>&
            by_slave)
        : by_slave_(by_slave)
    {
    }

    [[nodiscard]] bool anyPreconstrained(GlobalIndex slave) const
    {
        const auto found = by_slave_.find(slave);
        return found != by_slave_.end() &&
               std::any_of(found->second.begin(), found->second.end(),
                           [](const CanonicalLineVisibility& visibility) {
                               return visibility.preconstrained;
                           });
    }

    // The first rank (in rank order) that assembles with the slave and lacks
    // a master of its canonical line.
    [[nodiscard]] std::optional<CanonicalLineVisibility>
    firstNeededMissingMasters(GlobalIndex slave, const ConstraintLine&) const
    {
        const auto found = by_slave_.find(slave);
        if (found == by_slave_.end()) {
            return std::nullopt;
        }
        for (const auto& visibility : found->second) {
            if (visibility.needed && !visibility.missing_masters.empty()) {
                return visibility;
            }
        }
        return std::nullopt;
    }

private:
    const std::map<GlobalIndex, std::vector<CanonicalLineVisibility>>&
        by_slave_;
};

/**
 * Final validation of the canonical lines before installation: every slave
 * has exactly one owner, every non-suppressed slave has its line, every
 * master has exactly one owner, and every rank that assembles with a slave
 * carries all of its masters.  Returns the failure count (with the first
 * four reasons in `failures`); when it is zero, fills the canonical rows,
 * slaves, counts and the locally relevant lines of `result`.
 */
template <class Partition, class Oracle, class LineOracle>
[[nodiscard]] std::size_t validateCanonicalAggregationLines(
    const Partition& partition,
    const Oracle& oracle,
    const LineOracle& line_visibility,
    CanonicalSelection& selection,
    DistributedAggregationResult& result,
    std::ostringstream& failures)
{
    auto& canonical_lines = selection.canonical_lines;
    auto& canonical_row_metadata = selection.canonical_row_metadata;
    const auto& expected_slaves = selection.expected_slaves;
    std::size_t failure_count = 0u;
    std::set<GlobalIndex> globally_preconstrained;
    for (const auto& [slave, row] : canonical_row_metadata) {
        static_cast<void>(row);
        const auto owner_count = oracle.ownerCount(slave);
        if (owner_count != 1u) {
            ++failure_count;
            if (failure_count <= 4u) {
                failures << " dof=" << slave
                         << " reason=canonical_slave_owner_count:"
                         << owner_count << ";";
            }
        }
    }
    for (const auto slave : expected_slaves) {
        const bool any_preconstrained =
            line_visibility.anyPreconstrained(slave);
        if (any_preconstrained) {
            globally_preconstrained.insert(slave);
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
            const auto master_owner_count = oracle.ownerCount(entry.master_dof);
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
        // A rank that assembles with the line must carry every master.
        if (const auto visibility = line_visibility.firstNeededMissingMasters(
                slave, line_it->second)) {
            ++failure_count;
            if (failure_count <= 4u) {
                failures << " dof=" << slave
                         << " reason=canonical_master_not_relevant rank="
                         << visibility->rank << " masters=";
                for (const auto master : visibility->missing_masters) {
                    failures << master << ",";
                }
                failures << ";";
            }
        }
    }
    if (failure_count > 0u) {
        return failure_count;
    }

    result.canonical_strong_suppressed_dofs =
        globally_preconstrained.size();
    result.canonical_rows.reserve(canonical_row_metadata.size());
    for (auto& [slave, row] : canonical_row_metadata) {
        row.slave_owner_rank = oracle.ownerRank(slave);
        if (globally_preconstrained.count(slave) != 0u) {
            row.preconstrained_at_apply = true;
        }
        result.canonical_rows.push_back(std::move(row));
    }
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
        // Install where the slave is relevant and every master is present;
        // the validation above proved that no rank assembling with the line
        // lacks a master.
        if (partition.isRelevant(slave) &&
            std::all_of(line.entries.begin(), line.entries.end(),
                        [&](const auto& entry) {
                            return partition.isRelevant(entry.master_dof);
                        })) {
            result.relevant_lines.push_back(line);
        }
    }
    return 0u;
}

void throwCanonicalValidationFailures(std::string_view field_name,
                                      std::size_t failure_count,
                                      const std::ostringstream& failures)
{
    if (failure_count > 0u) {
        throw InsufficientGhostHaloError(
            "SmallCutAggregationConstraint: diagnostic="
            "incomplete_distributed_aggregation_halo field='" +
            std::string(field_name) + "' inconsistent_candidate_dofs=" +
            std::to_string(failure_count) + failures.str() +
            " Increase mesh/DOF overlap (<Ghost_layers>) until every rank "
            "that assembles with a canonical slave carries all of its "
            "nonzero aggregate masters; refusing owner-wins constraint "
            "resolution.");
    }
}

// Slaves that share a cell with an owned row but whose masters lie outside
// this rank's halo: the assembling ranks condense them correctly (the
// residual is unaffected), but the owned rows cannot store the condensed
// master columns, so the Jacobian loses those entries.  Communicator sum
// (collective unless `comm` is MPI_COMM_NULL).
template <class Partition>
[[nodiscard]] std::size_t countRowCoupledSlavesBeyondHalo(
    const std::set<GlobalIndex>& row_coupled_candidate_dofs,
    const std::map<GlobalIndex, ConstraintLine>& canonical_lines,
    const Partition& partition,
    MPI_Comm comm)
{
    long long local_row_coupled_beyond_halo = 0;
    for (const auto dof : row_coupled_candidate_dofs) {
        const auto line_it = canonical_lines.find(dof);
        if (line_it == canonical_lines.end()) {
            continue;
        }
        if (!std::all_of(line_it->second.entries.begin(),
                         line_it->second.entries.end(),
                         [&](const ConstraintEntry& entry) {
                             return partition.isRelevant(entry.master_dof);
                         })) {
            ++local_row_coupled_beyond_halo;
        }
    }
    long long row_coupled_beyond_halo = local_row_coupled_beyond_halo;
    if (comm != MPI_COMM_NULL) {
        MPI_Allreduce(&local_row_coupled_beyond_halo,
                      &row_coupled_beyond_halo,
                      1,
                      MPI_LONG_LONG,
                      MPI_SUM,
                      comm);
    }
    return static_cast<std::size_t>(row_coupled_beyond_halo);
}

// ---------------------------------------------------------------------------
// Owner-sourced refresh (see SmallCutAggregationConstraint::apply).
//
// The communicator-global refresh gathers every classified cell from every
// rank that stores it, every owned cell's face signatures, and every root
// proposal from every rank that stores the root, so each rank receives data
// that grows with the mesh and with the rank count.  The owner-sourced
// refresh computes the same canonical ledger from one record per entity:
//  - a census routed to a home rank per cell (cell ID modulo the rank count)
//    proves that every stored cell has exactly one owner, that its owner
//    classified it and that every rank classifying it agrees on whether it
//    is active; it also yields the canonical measure provider (lowest
//    classifying rank) and the holders (ranks storing the cell) of every
//    active cell;
//  - each owner matches the faces of its cells against its local cells; a
//    census of the faces without a local neighbor and of the faces shared
//    with another rank's cell proves that no owner misses a neighbor;
//  - the owners of the active cells all-gather one record per active cell
//    (class, measure, field DOFs in node order, a digest of the node
//    coordinates and the face-neighbor IDs); every rank checks its own
//    classification, DOFs and coordinates of every stored active cell
//    against it, which makes every copy of a cell interchangeable;
//  - candidate facts are merged at a home rank per candidate DOF in source
//    rank order, as the global merge does, and the merged candidates are
//    all-gathered;
//  - each root cell's proposals are built once, by the owner, and stand for
//    the identical proposals of every holder: the provider rank is the
//    lowest holder rank and the communicator counters are weighted by the
//    holder count; only each candidate's preferred valid proposal is
//    all-gathered with its lines;
//  - the visibility queries of the canonical selection are answered from
//    owner, preconstrained, assembling-rank and missing-master reports.
// Any inconsistency, failure or debug option returns to the global refresh,
// which then reports exactly what it always reported.

[[nodiscard]] std::uint64_t ownerSourcedMix(std::uint64_t digest,
                                            std::uint64_t word) noexcept
{
    word += 0x9e3779b97f4a7c15ull;
    word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ull;
    word = (word ^ (word >> 27)) * 0x94d049bb133111ebull;
    word ^= word >> 31;
    digest ^= word;
    digest = std::rotl(digest, 23) * 0x9e3779b185ebca87ull;
    return digest;
}

[[nodiscard]] std::uint64_t ownerSourcedKeyHash(
    std::span<const GlobalIndex> key) noexcept
{
    std::uint64_t digest = 0x5356534b45594831ull;
    digest = ownerSourcedMix(digest, static_cast<std::uint64_t>(key.size()));
    for (const auto word : key) {
        digest = ownerSourcedMix(digest, static_cast<std::uint64_t>(word));
    }
    return digest;
}

[[nodiscard]] int ownerSourcedHome(std::uint64_t key, int world_size) noexcept
{
    return static_cast<int>(key % static_cast<std::uint64_t>(world_size));
}

// Cell census flags.
constexpr std::int64_t kCellCensusOwned = 1;
constexpr std::int64_t kCellCensusClassified = 2;
constexpr std::int64_t kCellCensusActive = 4;

struct CellCensusReply {
    int provider_rank{-1};
    int holder_count{0};
    int min_holder_rank{-1};
};

/**
 * Home side of the cell census.  `received` holds [cell ID, flags] pairs per
 * source rank.  Returns false when a stored cell does not have exactly one
 * owner, when its owner did not classify it, or when the classifying ranks
 * disagree on whether it is active.  For every active cell, appends
 * [cell ID, provider rank, holder count, lowest holder rank] to the reply
 * for its owner.
 */
[[nodiscard]] bool resolveCellCensus(
    const GatheredInt64Words& received,
    int world_size,
    std::vector<std::vector<std::int64_t>>& replies)
{
    struct Entry {
        GlobalIndex gid{INVALID_GLOBAL_INDEX};
        int source{-1};
        std::int64_t flags{0};
    };
    std::vector<Entry> entries;
    entries.reserve(received.words.size() / 2u);
    for (int rank = 0; rank < world_size; ++rank) {
        const auto begin = static_cast<std::size_t>(
            received.displacements[static_cast<std::size_t>(rank)]);
        const auto count = static_cast<std::size_t>(
            received.counts[static_cast<std::size_t>(rank)]);
        if (count % 2u != 0u) {
            return false;
        }
        for (std::size_t i = 0u; i < count; i += 2u) {
            entries.push_back(Entry{
                .gid = static_cast<GlobalIndex>(received.words[begin + i]),
                .source = rank,
                .flags = received.words[begin + i + 1u]});
        }
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [](const Entry& lhs, const Entry& rhs) {
                         return lhs.gid < rhs.gid;
                     });
    replies.assign(static_cast<std::size_t>(world_size), {});
    for (std::size_t first = 0u; first < entries.size();) {
        std::size_t last = first + 1u;
        while (last < entries.size() && entries[last].gid == entries[first].gid) {
            ++last;
        }
        int owner = -1;
        std::size_t owners = 0u;
        bool owner_classified = false;
        int provider = -1;
        int classified_active = -1;
        for (std::size_t k = first; k < last; ++k) {
            const auto& entry = entries[k];
            if (entry.gid < 0 || entry.flags < 0 || entry.flags > 7 ||
                (k > first && entry.source == entries[k - 1u].source)) {
                return false;
            }
            const bool classified = (entry.flags & kCellCensusClassified) != 0;
            const int active = (entry.flags & kCellCensusActive) != 0 ? 1 : 0;
            if ((entry.flags & kCellCensusOwned) != 0) {
                ++owners;
                owner = entry.source;
                owner_classified = classified;
            }
            if (classified) {
                if (provider < 0) {
                    provider = entry.source;
                }
                if (classified_active < 0) {
                    classified_active = active;
                } else if (classified_active != active) {
                    return false;
                }
            } else if (active != 0) {
                return false;
            }
        }
        if (owners != 1u || !owner_classified) {
            return false;
        }
        if (classified_active == 1) {
            auto& reply = replies[static_cast<std::size_t>(owner)];
            reply.push_back(static_cast<std::int64_t>(entries[first].gid));
            reply.push_back(provider);
            reply.push_back(static_cast<std::int64_t>(last - first));
            reply.push_back(entries[first].source);
        }
        first = last;
    }
    return true;
}

// Face census kinds.
constexpr std::int64_t kFaceCensusOpen = 1;
constexpr std::int64_t kFaceCensusShared = 2;

/**
 * Home side of the face census.  `received` holds [face hash, kind] pairs:
 * an owner reports each face of an owned cell that has no local neighbor
 * (open) and each face it shares with a local cell owned by another rank
 * (shared).  A domain-boundary face is reported once as open; a face
 * between two owners twice as shared, by both owners.  Anything else means
 * that an owner misses a neighbor (or a face is non-manifold, or two
 * signatures collide) and returns false.
 */
[[nodiscard]] bool resolveFaceCensus(const GatheredInt64Words& received,
                                     int world_size)
{
    struct Entry {
        std::uint64_t hash{0u};
        int source{-1};
        std::int64_t kind{0};
    };
    std::vector<Entry> entries;
    entries.reserve(received.words.size() / 2u);
    for (int rank = 0; rank < world_size; ++rank) {
        const auto begin = static_cast<std::size_t>(
            received.displacements[static_cast<std::size_t>(rank)]);
        const auto count = static_cast<std::size_t>(
            received.counts[static_cast<std::size_t>(rank)]);
        if (count % 2u != 0u) {
            return false;
        }
        for (std::size_t i = 0u; i < count; i += 2u) {
            entries.push_back(Entry{
                .hash = static_cast<std::uint64_t>(received.words[begin + i]),
                .source = rank,
                .kind = received.words[begin + i + 1u]});
        }
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [](const Entry& lhs, const Entry& rhs) {
                         return lhs.hash < rhs.hash;
                     });
    for (std::size_t first = 0u; first < entries.size();) {
        std::size_t last = first + 1u;
        while (last < entries.size() &&
               entries[last].hash == entries[first].hash) {
            ++last;
        }
        const auto count = last - first;
        const bool single_open =
            count == 1u && entries[first].kind == kFaceCensusOpen;
        const bool shared_pair =
            count == 2u && entries[first].kind == kFaceCensusShared &&
            entries[first + 1u].kind == kFaceCensusShared &&
            entries[first].source != entries[first + 1u].source;
        if (!single_open && !shared_pair) {
            return false;
        }
        first = last;
    }
    return true;
}

/**
 * One owner-sourced active cell: its class and measure as its owner
 * classified it, its field DOFs in node order (node-major over the field
 * nodes), a digest of its cell type and node coordinates, and the IDs of all
 * its face neighbors (active or not).
 */
struct OwnerSourcedActiveCell {
    GlobalIndex gid{INVALID_GLOBAL_INDEX};
    int owner_rank{-1};
    CellClass klass{};
    int provider_rank{-1};
    int holder_count{0};
    int min_holder_rank{-1};
    std::size_t rule_count{0u};
    Real volume{0.0};
    std::vector<std::uint64_t> rule_ids{};
    std::vector<GlobalIndex> node_dofs{};
    std::uint64_t geometry_digest{0u};
    std::vector<GlobalIndex> neighbor_gids{};
    CellKey key{};
};

// [gid, flags, provider, holders, lowest holder, rule count, volume bits,
//  rule IDs..., n node DOFs, node DOFs..., geometry digest, n neighbors,
//  neighbor IDs...]
void appendOwnerSourcedActiveCell(std::vector<std::int64_t>& words,
                                  const OwnerSourcedActiveCell& cell)
{
    words.push_back(static_cast<std::int64_t>(cell.gid));
    words.push_back((cell.klass.full_active ? 1 : 0) |
                    (cell.klass.cut ? 2 : 0));
    words.push_back(cell.provider_rank);
    words.push_back(cell.holder_count);
    words.push_back(cell.min_holder_rank);
    words.push_back(static_cast<std::int64_t>(cell.rule_count));
    words.push_back(std::bit_cast<std::int64_t>(
        static_cast<double>(cell.volume)));
    for (const auto id : cell.rule_ids) {
        words.push_back(std::bit_cast<std::int64_t>(id));
    }
    words.push_back(static_cast<std::int64_t>(cell.node_dofs.size()));
    for (const auto dof : cell.node_dofs) {
        words.push_back(static_cast<std::int64_t>(dof));
    }
    words.push_back(std::bit_cast<std::int64_t>(cell.geometry_digest));
    words.push_back(static_cast<std::int64_t>(cell.neighbor_gids.size()));
    for (const auto gid : cell.neighbor_gids) {
        words.push_back(static_cast<std::int64_t>(gid));
    }
}

/**
 * Decodes the all-gathered owner-sourced active cells, sorted by cell ID.
 * Returns false on a malformed record or a cell declared by two owners.
 */
[[nodiscard]] bool decodeOwnerSourcedActiveCells(
    const GatheredInt64Words& gathered,
    int world_size,
    std::vector<OwnerSourcedActiveCell>& cells)
{
    cells.clear();
    for (int rank = 0; rank < world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered.displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered.counts[static_cast<std::size_t>(rank)]);
        const auto take = [&](std::int64_t& word) {
            if (position >= end) {
                return false;
            }
            word = gathered.words[position++];
            return true;
        };
        while (position < end) {
            OwnerSourcedActiveCell cell;
            cell.owner_rank = rank;
            std::int64_t gid = 0, flags = 0, provider = 0, holders = 0,
                         min_holder = 0, rule_count = 0, volume_bits = 0;
            if (!take(gid) || !take(flags) || !take(provider) ||
                !take(holders) || !take(min_holder) || !take(rule_count) ||
                !take(volume_bits)) {
                return false;
            }
            if (gid < 0 || (flags != 1 && flags != 2) || provider < 0 ||
                provider >= world_size || holders <= 0 ||
                holders > world_size || min_holder < 0 ||
                min_holder >= world_size || rule_count <= 0 ||
                static_cast<std::uint64_t>(rule_count) >
                    static_cast<std::uint64_t>(end - position)) {
                return false;
            }
            cell.gid = static_cast<GlobalIndex>(gid);
            cell.klass.full_active = flags == 1;
            cell.klass.cut = flags == 2;
            cell.provider_rank = static_cast<int>(provider);
            cell.holder_count = static_cast<int>(holders);
            cell.min_holder_rank = static_cast<int>(min_holder);
            cell.rule_count = static_cast<std::size_t>(rule_count);
            cell.volume = static_cast<Real>(std::bit_cast<double>(volume_bits));
            cell.rule_ids.reserve(cell.rule_count);
            for (std::int64_t i = 0; i < rule_count; ++i) {
                cell.rule_ids.push_back(
                    std::bit_cast<std::uint64_t>(gathered.words[position++]));
            }
            std::int64_t n_dofs = 0;
            if (!take(n_dofs) || n_dofs <= 0 ||
                static_cast<std::uint64_t>(n_dofs) >
                    static_cast<std::uint64_t>(end - position)) {
                return false;
            }
            cell.node_dofs.reserve(static_cast<std::size_t>(n_dofs));
            for (std::int64_t i = 0; i < n_dofs; ++i) {
                cell.node_dofs.push_back(
                    static_cast<GlobalIndex>(gathered.words[position++]));
            }
            std::int64_t digest = 0, n_neighbors = 0;
            if (!take(digest) || !take(n_neighbors) || n_neighbors < 0 ||
                static_cast<std::uint64_t>(n_neighbors) >
                    static_cast<std::uint64_t>(end - position)) {
                return false;
            }
            cell.geometry_digest = std::bit_cast<std::uint64_t>(digest);
            cell.neighbor_gids.reserve(static_cast<std::size_t>(n_neighbors));
            for (std::int64_t i = 0; i < n_neighbors; ++i) {
                cell.neighbor_gids.push_back(
                    static_cast<GlobalIndex>(gathered.words[position++]));
            }
            cell.key = cell.node_dofs;
            std::sort(cell.key.begin(), cell.key.end());
            cells.push_back(std::move(cell));
        }
    }
    std::sort(cells.begin(), cells.end(),
              [](const OwnerSourcedActiveCell& lhs,
                 const OwnerSourcedActiveCell& rhs) {
                  return lhs.gid < rhs.gid;
              });
    return std::adjacent_find(cells.begin(), cells.end(),
                              [](const OwnerSourcedActiveCell& lhs,
                                 const OwnerSourcedActiveCell& rhs) {
                                  return lhs.gid == rhs.gid;
                              }) == cells.end();
}

[[nodiscard]] const OwnerSourcedActiveCell* findOwnerSourcedActiveCell(
    const std::vector<OwnerSourcedActiveCell>& cells,
    GlobalIndex gid)
{
    const auto it = std::lower_bound(
        cells.begin(), cells.end(), gid,
        [](const OwnerSourcedActiveCell& cell, GlobalIndex key) {
            return cell.gid < key;
        });
    return it != cells.end() && it->gid == gid ? &*it : nullptr;
}

// Candidate census flags (LocalCandidateSupport facts).
constexpr std::int64_t kCandidateCensusCut = 1;
constexpr std::int64_t kCandidateCensusFull = 2;
constexpr std::int64_t kCandidateCensusExcluded = 4;
// The record carries only the flags: the vertex touches a full-active cell
// or is excluded on its source rank, so it is no candidate.
constexpr std::int64_t kCandidateCensusMarker = 8;

/**
 * Home side of the candidate census: merges the facts of every candidate
 * DOF in source-rank order exactly like resolveGlobalCandidateSupport
 * (coordinates and component DOFs from the lowest rank, flags OR-combined)
 * and appends [DOF, n components, x, y, z, components...] for every
 * candidate to `merged`.  Returns false when two full records disagree or a
 * record is malformed.
 */
[[nodiscard]] bool mergeCandidateCensus(const GatheredInt64Words& received,
                                        int world_size,
                                        std::vector<std::int64_t>& merged)
{
    struct Entry {
        GlobalIndex dof{INVALID_GLOBAL_INDEX};
        std::int64_t flags{0};
        std::size_t position{0u};
        std::size_t component_count{0u};
    };
    std::vector<Entry> entries;
    for (int rank = 0; rank < world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            received.displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            received.counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            if (end - position < 2u) {
                return false;
            }
            Entry entry;
            entry.dof = static_cast<GlobalIndex>(received.words[position++]);
            entry.flags = received.words[position++];
            if (entry.dof < 0 || entry.flags < 0 || entry.flags > 15) {
                return false;
            }
            if ((entry.flags & kCandidateCensusMarker) == 0) {
                if (end - position < 4u) {
                    return false;
                }
                const auto count = received.words[position++];
                if (count <= 0 ||
                    static_cast<std::uint64_t>(count) + 3u >
                        static_cast<std::uint64_t>(end - position)) {
                    return false;
                }
                entry.component_count = static_cast<std::size_t>(count);
                entry.position = position;
                position += 3u + entry.component_count;
            }
            entries.push_back(entry);
        }
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [](const Entry& lhs, const Entry& rhs) {
                         return lhs.dof < rhs.dof;
                     });
    for (std::size_t first = 0u; first < entries.size();) {
        std::size_t last = first + 1u;
        while (last < entries.size() && entries[last].dof == entries[first].dof) {
            ++last;
        }
        std::int64_t flags = 0;
        const Entry* canonical = nullptr;
        for (std::size_t k = first; k < last; ++k) {
            const auto& entry = entries[k];
            flags |= entry.flags & (kCandidateCensusCut | kCandidateCensusFull |
                                    kCandidateCensusExcluded);
            if ((entry.flags & kCandidateCensusMarker) != 0) {
                continue;
            }
            if (canonical == nullptr) {
                canonical = &entry;
                continue;
            }
            if (entry.component_count != canonical->component_count ||
                !std::equal(
                    received.words.begin() +
                        static_cast<std::ptrdiff_t>(entry.position + 3u),
                    received.words.begin() +
                        static_cast<std::ptrdiff_t>(entry.position + 3u +
                                                    entry.component_count),
                    received.words.begin() +
                        static_cast<std::ptrdiff_t>(canonical->position + 3u))) {
                return false;
            }
            for (std::size_t d = 0u; d < 3u; ++d) {
                const auto lhs = static_cast<Real>(std::bit_cast<double>(
                    received.words[canonical->position + d]));
                const auto rhs = static_cast<Real>(std::bit_cast<double>(
                    received.words[entry.position + d]));
                const auto scale =
                    std::max({Real{1}, std::abs(lhs), std::abs(rhs)});
                if (std::abs(lhs - rhs) > Real{1e-12} * scale) {
                    return false;
                }
            }
        }
        const bool candidate = (flags & kCandidateCensusCut) != 0 &&
                               (flags & kCandidateCensusExcluded) == 0 &&
                               (flags & kCandidateCensusFull) == 0;
        if (candidate) {
            if (canonical == nullptr) {
                return false;
            }
            merged.push_back(static_cast<std::int64_t>(entries[first].dof));
            merged.push_back(
                static_cast<std::int64_t>(canonical->component_count));
            for (std::size_t w = 0u; w < 3u + canonical->component_count; ++w) {
                merged.push_back(received.words[canonical->position + w]);
            }
        }
        first = last;
    }
    return true;
}

/**
 * Decodes the all-gathered merged candidates (mergeCandidateCensus) into the
 * communicator candidate map.  Returns false on a malformed or repeated
 * record.
 */
[[nodiscard]] bool decodeMergedCandidates(const GatheredInt64Words& gathered,
                                          GlobalCandidateMap& candidates)
{
    candidates.clear();
    std::size_t position = 0u;
    const auto end = gathered.words.size();
    while (position < end) {
        if (end - position < 5u) {
            return false;
        }
        const auto dof = static_cast<GlobalIndex>(gathered.words[position++]);
        const auto count = gathered.words[position++];
        if (dof < 0 || count <= 0 ||
            static_cast<std::uint64_t>(count) + 3u >
                static_cast<std::uint64_t>(end - position)) {
            return false;
        }
        GlobalCandidateSupport support;
        for (auto& coordinate : support.coordinates) {
            coordinate = static_cast<Real>(
                std::bit_cast<double>(gathered.words[position++]));
        }
        support.component_dofs.reserve(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            support.component_dofs.push_back(
                static_cast<GlobalIndex>(gathered.words[position++]));
        }
        if (support.component_dofs.front() != dof ||
            !candidates.emplace(dof, std::move(support)).second) {
            return false;
        }
    }
    return true;
}

// Root proposal outcome flags (one record per owner-built attempt).
constexpr std::int64_t kProposalEmitted = 1;
constexpr std::int64_t kProposalDropped = 2;
constexpr std::int64_t kProposalInversionFailed = 4;
constexpr std::int64_t kProposalRoundOff = 8;
constexpr std::int64_t kProposalExtrapolationRejected = 16;
constexpr std::int64_t kProposalEmptyLine = 32;
constexpr std::int64_t kProposalLineGuardRejected = 64;

/**
 * Proposal-DOF visibility of the owner-sourced refresh, built from reports
 * of the ranks that own a DOF, that hold it strongly constrained, that
 * assemble with a candidate DOF, and of the preferred-proposal masters that
 * such an assembling rank lacks.  The canonical selection asks only for the
 * assembling ranks of a slave and for masters of the preferred proposal on
 * those ranks, which these reports answer exactly.
 */
class OwnerSourcedVisibility {
public:
    std::map<GlobalIndex, std::vector<int>> owners{};
    std::set<GlobalIndex> preconstrained_dofs{};
    std::map<GlobalIndex, std::vector<ProposalDofVisibility>> assembling{};
    std::map<int, std::set<GlobalIndex>> missing_masters{};

    [[nodiscard]] bool preconstrained(GlobalIndex dof) const
    {
        return preconstrained_dofs.count(dof) != 0u;
    }

    [[nodiscard]] bool relevantOnRank(GlobalIndex dof, int rank) const
    {
        const auto missing = missing_masters.find(rank);
        return missing == missing_masters.end() ||
               missing->second.count(dof) == 0u;
    }

    [[nodiscard]] std::size_t ownerCount(GlobalIndex dof) const
    {
        const auto found = owners.find(dof);
        return found == owners.end() ? 0u : found->second.size();
    }

    [[nodiscard]] int ownerRank(GlobalIndex dof) const
    {
        const auto found = owners.find(dof);
        return found == owners.end() || found->second.size() != 1u
                   ? -1
                   : found->second.front();
    }

    [[nodiscard]] const std::vector<ProposalDofVisibility>* slaveVisibility(
        GlobalIndex dof) const
    {
        const auto found = assembling.find(dof);
        return found == assembling.end() ? nullptr : &found->second;
    }

    [[nodiscard]] bool anyPreconstrained(GlobalIndex slave) const
    {
        return preconstrained(slave);
    }

    [[nodiscard]] std::optional<CanonicalLineVisibility>
    firstNeededMissingMasters(GlobalIndex slave,
                              const ConstraintLine& line) const
    {
        const auto found = assembling.find(slave);
        if (found == assembling.end()) {
            return std::nullopt;
        }
        for (const auto& visibility : found->second) {
            CanonicalLineVisibility result;
            result.rank = visibility.rank;
            result.needed = true;
            for (const auto& entry : line.entries) {
                if (!relevantOnRank(entry.master_dof, visibility.rank)) {
                    result.missing_masters.push_back(entry.master_dof);
                }
            }
            if (!result.missing_masters.empty()) {
                return result;
            }
        }
        return std::nullopt;
    }
};

[[nodiscard]] DistributedAggregationResult
resolveDistributedAggregationDeclarations(
    const systems::FESystem& system,
    const AffineConstraints& existing_constraints,
    std::string_view field_name,
    const GlobalCandidateMap& global_candidates,
    const LocalAggregationDeclarationMap& local_candidates,
    const std::set<GlobalIndex>& assembled_candidate_dofs,
    const std::set<GlobalIndex>& row_coupled_candidate_dofs,
    bool slave_all_cut,
    bool allow_unaggregated,
    bool rootless_fallback)
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

    std::vector<std::int64_t> local_candidate_words;
    bool local_candidate_payload_valid = true;
    std::exception_ptr local_candidate_serialization_exception;
    try {
    serializeRootProposals(local_candidates,
                           global_candidates,
                           local_candidate_words,
                           local_candidate_payload_valid);
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
            comm, std::span<const std::int64_t>(local_candidate_words),
            AggregationProfileExchange::RootProposals);
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

    RankedDeclarationMap declarations_by_dof;
    std::exception_ptr local_declaration_decode_exception;
    try {
    decodeRootProposals(gathered_candidates,
                        world_size,
                        [](int source_rank, GlobalIndex) { return source_rank; },
                        declarations_by_dof);
    } catch (...) {
        local_declaration_decode_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_declaration_decode_exception,
                                      "root_proposal_decode");

    // Gather ownership, relevance and pre-existing constraints for every DOF
    // mentioned by any proposal. The root itself is chosen by the
    // partition-independent (distance, physical root cell GID) ordering
    // alone; choosing among roots by which ranks can see their masters would
    // change the discretization with the partition. Algebraic master DOFs are
    // deliberately excluded from the ordering because owner-contiguous
    // numbering changes when the physical mesh is repartitioned.
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

    // [dof, owned, preconstrained, assembled] for locally relevant proposal
    // DOFs; `assembled` marks candidate DOFs this rank assembles with.
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
        local_proposal_visibility_words.push_back(
            (partition.isOwned(dof) ||
             assembled_candidate_dofs.count(dof) != 0u)
                ? 1
                : 0);
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
            std::span<const std::int64_t>(local_proposal_visibility_words),
            AggregationProfileExchange::ProposalVisibility);
    } else {
        gathered_proposal_visibility.words =
            local_proposal_visibility_words;
        gathered_proposal_visibility.counts = {
            static_cast<int>(local_proposal_visibility_words.size())};
        gathered_proposal_visibility.displacements = {0};
    }

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
            if (end - position < 4u) {
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
            visibility.needed =
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

    const GatheredProposalVisibility proposal_visibility(
        proposal_visibility_by_dof);
    CanonicalSelection selection;
    auto& canonical_lines = selection.canonical_lines;
    auto& canonical_row_metadata = selection.canonical_row_metadata;
    auto& expected_slaves = selection.expected_slaves;
    std::vector<std::int64_t> local_visibility_words;
    std::exception_ptr local_canonical_selection_exception;
    try {
    selectCanonicalAggregationLines(
        field_name,
        global_candidates,
        declarations_by_dof,
        proposal_visibility,
        slave_all_cut,
        allow_unaggregated,
        rootless_fallback,
        [&](std::string_view reason,
            GlobalIndex dof,
            const GlobalCandidateSupport& support,
            const std::string& detail) {
            logGuardRootlessCandidate(system,
                                      field_name,
                                      reason,
                                      dof,
                                      support.coordinates,
                                      support.feature_cells,
                                      detail);
        },
        selection);
    result.canonical_halo_limited_root_choices =
        selection.halo_limited_root_choices;
    result.canonical_guard_rootless_candidate_vertices =
        selection.guard_rootless_candidate_vertices;
    throwCanonicalSelectionFailures(field_name, selection);

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
        local_visibility_words.push_back(
            (partition.isOwned(slave) ||
             assembled_candidate_dofs.count(slave) != 0u)
                ? 1
                : 0);
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
            comm, std::span<const std::int64_t>(local_visibility_words),
            AggregationProfileExchange::LineVisibility);
    } else {
        gathered_visibility.words = local_visibility_words;
        gathered_visibility.counts = {
            static_cast<int>(local_visibility_words.size())};
        gathered_visibility.displacements = {0};
    }

    std::exception_ptr local_final_validation_exception;
    try {
    std::map<GlobalIndex, std::vector<CanonicalLineVisibility>>
        visibility_by_slave;
    for (int rank = 0; rank < world_size; ++rank) {
        std::size_t position = static_cast<std::size_t>(
            gathered_visibility.displacements[static_cast<std::size_t>(rank)]);
        const auto end = position + static_cast<std::size_t>(
            gathered_visibility.counts[static_cast<std::size_t>(rank)]);
        while (position < end) {
            if (end - position < 5u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed canonical-line "
                    "visibility declaration");
            }
            const auto slave = static_cast<GlobalIndex>(
                gathered_visibility.words[position++]);
            CanonicalLineVisibility visibility;
            visibility.rank = rank;
            visibility.owned = gathered_visibility.words[position++] != 0;
            visibility.preconstrained =
                gathered_visibility.words[position++] != 0;
            visibility.needed = gathered_visibility.words[position++] != 0;
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

    const GatheredLineVisibility line_visibility(visibility_by_slave);
    std::ostringstream validation_failures;
    const auto validation_failure_count = validateCanonicalAggregationLines(
        partition,
        proposal_visibility,
        line_visibility,
        selection,
        result,
        validation_failures);
    throwCanonicalValidationFailures(
        field_name, validation_failure_count, validation_failures);
    } catch (...) {
        local_final_validation_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_final_validation_exception,
                                      "canonical_line_final_validation");

    result.canonical_row_coupled_slaves_beyond_halo =
        countRowCoupledSlavesBeyondHalo(
            row_coupled_candidate_dofs,
            canonical_lines,
            partition,
            initialized != 0 && world_size > 1 ? comm : MPI_COMM_NULL);
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
    std::string_view field_name,
    const GlobalCandidateMap& global_candidates,
    const LocalAggregationDeclarationMap& local_candidates,
    const std::set<GlobalIndex>&,
    const std::set<GlobalIndex>&,
    bool slave_all_cut,
    bool allow_unaggregated,
    bool rootless_fallback)
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
        const auto candidate_dof = dof;
        const auto* candidate_components =
            &support.component_dofs;
        // With the opt-in fallback, a globally rooted candidate without any
        // proposal inside the guards gets the rootless-island policy below
        // (D33).
        const auto recordGuardRootless = [&]() {
            ++result.canonical_guard_rootless_candidate_vertices;
            logGuardRootlessCandidate(system,
                                      field_name,
                                      "proposal_guard",
                                      candidate_dof,
                                      support.coordinates,
                                      support.feature_cells,
                                      "detail=no_root_proposal_inside_guards");
        };
        auto make_row =
            [candidate_dof, candidate_components](
                GlobalIndex slave,
                SmallCutAggregationProvisionalRowKind kind) {
                const auto component_it = std::find(
                    candidate_components->begin(),
                    candidate_components->end(),
                    slave);
                if (component_it == candidate_components->end()) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: serial canonical "
                        "slave is absent from its candidate component list");
                }
                SmallCutAggregationProlongationRow row;
                row.candidate_dof = candidate_dof;
                row.component = static_cast<std::size_t>(
                    std::distance(candidate_components->begin(),
                                  component_it));
                row.slave_dof = slave;
                row.slave_owner_rank = 0;
                row.provisional_kind = kind;
                return row;
            };
        const bool all_components_preconstrained = std::all_of(
            support.component_dofs.begin(), support.component_dofs.end(),
            [&](GlobalIndex slave) {
                return existing_constraints.isConstrained(slave);
            });
        if (all_components_preconstrained) {
            result.canonical_strong_suppressed_dofs +=
                support.component_dofs.size();
            for (const auto slave : support.component_dofs) {
                auto row = make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::
                        PreexistingConstraint);
                row.preconstrained_at_apply = true;
                result.canonical_rows.push_back(std::move(row));
            }
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
            const auto& chosen = valid_declarations.front();
            for (const auto& line : valid_declarations.front().lines) {
                auto row = make_row(
                    line.slave_dof,
                    SmallCutAggregationProvisionalRowKind::RootedExtension);
                row.root_cell_gid = chosen.root_cell_gid;
                row.root_cell_owner_rank = 0;
                row.root_provider_rank = 0;
                row.root_distance = chosen.root_distance;
                row.provisional_entries = line.entries;
                if (existing_constraints.isConstrained(line.slave_dof)) {
                    ++result.canonical_strong_suppressed_dofs;
                    row.preconstrained_at_apply = true;
                    result.canonical_rows.push_back(std::move(row));
                    continue;
                }
                result.canonical_rows.push_back(std::move(row));
                result.canonical_slaves.push_back(line.slave_dof);
                ++result.canonical_owned_aggregate_dofs;
                if (partition.isRelevant(line.slave_dof)) {
                    result.relevant_lines.push_back(line);
                }
            }
        } else if (support.globally_rooted && allow_unaggregated) {
            for (const auto slave : support.component_dofs) {
                result.canonical_rows.push_back(make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::
                        UnaggregatedFreeIdentity));
            }
            continue;
        } else if (support.globally_rooted && !rootless_fallback) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "aggregation_no_root_inside_guards field='" +
                std::string(field_name) + "' candidates=1 dof=" +
                std::to_string(dof) +
                " reason=no_valid_root_proposal: " + kGuardRootlessOptInHint +
                ".");
        } else if (slave_all_cut && support.has_full_support) {
            if (support.globally_rooted) {
                recordGuardRootless();
            }
            for (const auto slave : support.component_dofs) {
                result.canonical_rows.push_back(make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::
                        SupportedFreeIdentity));
            }
        } else if (!allow_unaggregated) {
            if (support.globally_rooted) {
                recordGuardRootless();
            }
            for (const auto slave : support.component_dofs) {
                auto row = make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::
                        RootlessHomogeneousPin);
                if (existing_constraints.isConstrained(slave)) {
                    ++result.canonical_strong_suppressed_dofs;
                    row.preconstrained_at_apply = true;
                    result.canonical_rows.push_back(std::move(row));
                    continue;
                }
                result.canonical_rows.push_back(std::move(row));
                result.canonical_slaves.push_back(slave);
                ++result.canonical_owned_pinned_dofs;
                if (partition.isRelevant(slave)) {
                    result.relevant_lines.push_back(
                        ConstraintLine{.slave_dof = slave});
                }
            }
        } else {
            for (const auto slave : support.component_dofs) {
                result.canonical_rows.push_back(make_row(
                    slave,
                    SmallCutAggregationProvisionalRowKind::
                        UnaggregatedFreeIdentity));
            }
        }
    }
    std::sort(result.canonical_slaves.begin(),
              result.canonical_slaves.end());
    std::sort(
        result.canonical_rows.begin(),
        result.canonical_rows.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.slave_dof < rhs.slave_dof;
        });
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

// A retained volume rule is partial when its cell is cut on that side.  A
// rule that is not full-cell equivalent is a cut rule even when its fraction
// rounds to exactly 1 (a sub-round-off complement): its cell still carries
// the interface fragment and the non-full flag that define the cut topology.
// Classifying it as cut, like the same rule one ulp below 1, keeps the
// aggregation classification a function of the cut topology; treating it as
// neither cut nor full let the roots, and with them the constraints, switch
// between outer refreshes while the topology stayed the same.  Fractions are
// validated to lie in [0, 1] before this test.
[[nodiscard]] constexpr bool retainedPartialVolumeRule(
    bool full_cell_equivalent,
    Real fraction) noexcept
{
    return fraction > Real(0) && (fraction < Real(1) || !full_cell_equivalent);
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
        // The measure reads the parent cell from the rule provenance; copy
        // the rule only when that cell has to be substituted.
        if (rule.provenance.parent_entity == static_cast<MeshIndex>(cell)) {
            measure = geometry::physicalCutQuadratureMeasure(mesh, rule);
        } else {
            auto mapped_rule = rule;
            mapped_rule.provenance.parent_entity =
                static_cast<MeshIndex>(cell);
            measure =
                geometry::physicalCutQuadratureMeasure(mesh, mapped_rule);
        }
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
///
/// When the physical tolerance is the machine floor (64 eps times the
/// coordinate magnitude), the iteration stops once the residual is at that
/// floor and the last correction is below 64 eps max(1, |xi|).  The second
/// test assumes that xi can be resolved to relative round-off, but the
/// coordinates carry an absolute round-off of eps |x|, whose image in
/// reference space is about eps |x| ||J^{-1}|| ~ eps |x| / h.  On fine meshes
/// far from the origin (|x| / h above about 64 |xi|) the corrections then stall
/// at that image: Newton cycles between round-off neighbours of the solution
/// and never meets the test.  On the capillary-rise level-40 mesh this
/// rejected 15% of the root/vertex pairs within 4.5 cells at y = 20-24 mm
/// (8% with fused multiply-add), and the run stopped at a wall vertex whose
/// only root failed this way.  If the iteration budget
/// ends with the residual at the physical tolerance and a last correction
/// within ||J^{-1}|| times that tolerance, the iterate is as accurate as the
/// coordinates allow and is accepted; every inversion that converged before
/// is unchanged.  stalled_at_round_off reports such an acceptance.
[[nodiscard]] bool invertMapping(const geometry::GeometryMapping& mapping,
                                 const std::array<Real, 3>& physical,
                                 int dimension,
                                 Vector<Real, 3>& xi,
                                 bool* stalled_at_round_off = nullptr)
{
    if (stalled_at_round_off != nullptr) {
        *stalled_at_round_off = false;
    }
    const auto& nodes = mapping.nodes();
    if (nodes.empty() || dimension < 1 || dimension > 3) {
        return false;
    }
    const std::size_t dimension_size =
        static_cast<std::size_t>(dimension);
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
    for (std::size_t d = 0; d < dimension_size; ++d) {
        xi[d] = Real(0.25);
    }
    bool have_update = false;
    Real last_update_norm = std::numeric_limits<Real>::infinity();
    // Frobenius norm of the active block of J^{-1} at the last correction.
    Real last_inverse_norm = std::numeric_limits<Real>::infinity();
    for (int iteration = 0; iteration < 25; ++iteration) {
        const auto mapped = mapping.map_to_physical(xi);
        Vector<Real, 3> residual{};
        for (std::size_t d = 0; d < 3u; ++d) {
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
            for (std::size_t d = dimension_size; d < 3u; ++d) {
                for (std::size_t k = 0; k < 3u; ++k) {
                    j_inv(d, k) = Real(0);
                    j_inv(k, d) = Real(0);
                }
                j_inv(d, d) = Real(1);
            }
        }
        const auto inverse =
            geometry::scaleConditionedJacobianInverse(j_inv);
        {
            Real inverse_norm_sq = Real(0);
            for (std::size_t r = 0; r < dimension_size; ++r) {
                for (std::size_t c = 0; c < dimension_size; ++c) {
                    inverse_norm_sq += inverse(r, c) * inverse(r, c);
                }
            }
            last_inverse_norm = std::sqrt(inverse_norm_sq);
        }
        Vector<Real, 3> reference_update{};
        for (std::size_t r = 0; r < 3u; ++r) {
            Real update = Real(0);
            for (std::size_t c = 0; c < 3u; ++c) {
                update += inverse(r, c) * residual[c];
            }
            reference_update[r] = update;
            xi[r] += reference_update[r];
        }
        last_update_norm =
            geometry::detail::stable_vector_norm(reference_update);
        have_update = true;
        for (std::size_t d = dimension_size; d < 3u; ++d) {
            xi[d] = Real(0);
        }
        if (!std::isfinite(last_update_norm) ||
            !std::isfinite(xi[0]) || !std::isfinite(xi[1]) ||
            !std::isfinite(xi[2])) {
            return false;
        }
    }
    // Iteration budget exhausted: accept the iterate only if it reproduces the
    // point to the physical tolerance and the last correction lies within the
    // reference-space image of that tolerance (see the function comment).
    const auto mapped = mapping.map_to_physical(xi);
    Vector<Real, 3> residual{};
    for (std::size_t d = 0; d < 3u; ++d) {
        residual[d] = physical[d] - mapped[d];
    }
    const Real residual_norm = geometry::detail::stable_vector_norm(residual);
    const Real attainable_reference_accuracy =
        last_inverse_norm * physical_tolerance;
    if (have_update && std::isfinite(residual_norm) &&
        std::isfinite(attainable_reference_accuracy) &&
        residual_norm <= physical_tolerance &&
        last_update_norm <= attainable_reference_accuracy) {
        if (stalled_at_round_off != nullptr) {
            *stalled_at_round_off = true;
        }
        return true;
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

// unitSimplexExtrapolationDistance measures in the chart whose origin is the
// simplex vertex the mesh lists first, so the same physical point gets
// different distances under different vertex orders: three cells above a
// right-triangle root it measures 3.0 in the chart of the right-angle vertex
// and 4.24 or 5.0 in the sheared charts of the other two.  The guard uses the
// smallest distance over the d + 1 vertex charts.  It depends only on the
// barycentric coordinates (the extension weights), so it does not depend on
// the vertex order, and it never exceeds the first-vertex chart distance:
// every proposal that passed before still passes.
[[nodiscard]] Real vertexOrderInvariantSimplexExtrapolationDistance(
    const Vector<Real, 3>& xi,
    int dimension)
{
    Real distance = unitSimplexExtrapolationDistance(xi, dimension);
    if (!(distance > Real{0.0})) {
        return distance;
    }
    std::array<Real, 4> barycentric{};
    Real coordinate_sum = 0.0;
    for (int d = 0; d < dimension; ++d) {
        barycentric[static_cast<std::size_t>(d + 1)] =
            xi[static_cast<std::size_t>(d)];
        coordinate_sum += xi[static_cast<std::size_t>(d)];
    }
    barycentric[0] = Real{1.0} - coordinate_sum;
    for (int origin = 1; origin <= dimension; ++origin) {
        Vector<Real, 3> chart{};
        std::size_t slot = 0u;
        for (int vertex = 0; vertex <= dimension; ++vertex) {
            if (vertex != origin) {
                chart[slot++] = barycentric[static_cast<std::size_t>(vertex)];
            }
        }
        distance = std::min(
            distance, unitSimplexExtrapolationDistance(chart, dimension));
    }
    return distance;
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
        return vertexOrderInvariantSimplexExtrapolationDistance(xi, 2);
    case ElementType::Tetra4:
        return vertexOrderInvariantSimplexExtrapolationDistance(xi, 3);
    case ElementType::Wedge6: {
        const Real in_plane =
            vertexOrderInvariantSimplexExtrapolationDistance(xi, 2);
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

inline void aggregationDigestMix(std::uint64_t& digest,
                                 std::uint64_t word) noexcept
{
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::size_t byte = 0u; byte < sizeof(word); ++byte) {
        digest ^= (word >> (8u * byte)) & 0xffu;
        digest *= prime;
    }
}

inline void aggregationDigestMixString(std::uint64_t& digest,
                                       std::string_view value) noexcept
{
    constexpr std::uint64_t prime = 1099511628211ull;
    aggregationDigestMix(digest, static_cast<std::uint64_t>(value.size()));
    for (const unsigned char byte : value) {
        digest ^= byte;
        digest *= prime;
    }
}

[[nodiscard]] std::uint64_t aggregationDigestReal(double value)
{
    if (!std::isfinite(value)) {
        throw std::logic_error(
            "SmallCutAggregationConstraint: cannot digest a non-finite "
            "prolongation value");
    }
    const double canonical = value == 0.0 ? 0.0 : value;
    return std::bit_cast<std::uint64_t>(canonical);
}

[[nodiscard]] std::uint64_t canonicalFeatureClassFingerprint(
    std::span<const SmallCutAggregationActiveFeatureReport> features) noexcept
{
    std::uint64_t digest = 14695981039346656037ull;
    aggregationDigestMix(digest, 0x4147464541545552ull);
    aggregationDigestMix(digest,
                         static_cast<std::uint64_t>(features.size()));
    for (const auto& feature : features) {
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(feature.stable_feature_id));
        aggregationDigestMix(digest, feature.canonical_cell_gid_digest);
        aggregationDigestMix(
            digest, feature.canonical_full_active_cell_gid_digest);
        aggregationDigestMix(digest, feature.canonical_cut_cell_gid_digest);
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(feature.canonical_cell_count));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                feature.canonical_full_active_cell_count));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(feature.canonical_cut_cell_count));
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(feature.disposition));
    }
    return digest;
}

[[nodiscard]] std::uint64_t canonicalSlaveSetFingerprint(
    std::span<const GlobalIndex> slaves) noexcept
{
    std::uint64_t digest = 14695981039346656037ull;
    aggregationDigestMix(digest, 0x414747534c415645ull);
    aggregationDigestMix(digest, static_cast<std::uint64_t>(slaves.size()));
    for (const auto slave : slaves) {
        aggregationDigestMix(digest, static_cast<std::uint64_t>(slave));
    }
    return digest;
}

void finalizeGeometryIdentityFingerprint(
    SmallCutAggregationCanonicalGeometryIdentity& identity)
{
    std::uint64_t digest = 14695981039346656037ull;
    aggregationDigestMix(digest, 0x41474747454f4d45ull);
    aggregationDigestMix(digest, static_cast<std::uint64_t>(identity.kind));
    aggregationDigestMix(digest, identity.available ? 1u : 0u);
    aggregationDigestMixString(digest, identity.source_id);
    aggregationDigestMixString(digest, identity.domain_id);
    aggregationDigestMix(
        digest, static_cast<std::uint64_t>(identity.interface_marker));
    aggregationDigestMix(
        digest, aggregationDigestReal(static_cast<double>(identity.isovalue)));
    aggregationDigestMix(digest, identity.source_layout_revision);
    aggregationDigestMix(digest, identity.source_value_revision);
    aggregationDigestMix(digest, identity.quadrature_policy_key);
    aggregationDigestMix(digest, identity.snapshot_revision_key);
    aggregationDigestMix(
        digest, identity.distributed_mesh_geometry_revision);
    aggregationDigestMix(
        digest, identity.distributed_mesh_topology_revision);
    aggregationDigestMix(digest, identity.distributed_ownership_revision);
    aggregationDigestMix(digest, identity.distributed_numbering_revision);
    identity.canonical_fingerprint = digest;
}

struct SmallCutAggregationPreparedProvenance {
    SmallCutAggregationCanonicalGeometryIdentity geometry_identity{};
    SmallCutAggregationLocalPublicationLineage local_lineage{};
};

[[nodiscard]] SmallCutAggregationPreparedProvenance
prepareRefreshProvenance(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext& cut_context,
    int interface_marker,
    std::uint64_t successful_publication_ordinal,
    const AffineConstraints& staged_constraints)
{
    SmallCutAggregationPreparedProvenance result;
    auto& identity = result.geometry_identity;
    auto& lineage = result.local_lineage;
    lineage.successful_publication_ordinal =
        successful_publication_ordinal;
    lineage.cut_context_content_revision = cut_context.contentRevision();

    const auto& mesh = system.meshAccess();
    lineage.mesh_revision_tracking_available =
        mesh.revisionTrackingAvailable();
    if (lineage.mesh_revision_tracking_available) {
        lineage.mesh_geometry_revision = mesh.geometryRevision();
        lineage.mesh_topology_revision = mesh.topologyRevision();
        lineage.mesh_ownership_revision = mesh.ownershipRevision();
        lineage.mesh_numbering_revision = mesh.numberingRevision();
        lineage.mesh_field_layout_revision = mesh.fieldLayoutRevision();
        lineage.mesh_label_revision = mesh.labelRevision();
        lineage.mesh_active_configuration_epoch =
            mesh.activeConfigurationEpoch();
    }
    lineage.fe_space_revision = system.spaceRevision();
    lineage.fe_dof_layout_revision = system.dofLayoutRevision();
    lineage.fe_constraint_layout_revision =
        system.constraintLayoutRevision();
    lineage.fe_block_layout_revision = system.blockLayoutRevision();
    lineage.affine_constraint_build_revision =
        staged_constraints.constraintLayoutRevision();

    const auto* publication =
        cut_context.findGeneratedLevelSetInterfacePublicationProvenance(
            interface_marker);
    if (publication != nullptr) {
        lineage.has_generated_publication_request = true;
        lineage.generated_request_mesh_geometry_revision =
            publication->request.mesh_geometry_revision;
        lineage.generated_request_mesh_topology_revision =
            publication->request.mesh_topology_revision;
        lineage.generated_request_ownership_revision =
            publication->request.ownership_revision;
    }

    if (cut_context.hasFreeSurfaceGeometrySnapshotForMarker(
            interface_marker)) {
        const auto bound_revision =
            cut_context.freeSurfaceGeometrySnapshotRevisionForMarker(
                interface_marker);
        const interfaces::FreeSurfaceGeometrySnapshot* selected = nullptr;
        for (const auto& snapshot :
             cut_context.freeSurfaceGeometrySnapshots()) {
            if (snapshot != nullptr &&
                snapshot->revision().snapshot_revision_key ==
                    bound_revision) {
                if (selected != nullptr) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: authoritative "
                        "free-surface snapshot identity is ambiguous");
                }
                selected = snapshot.get();
            }
        }
        if (selected == nullptr || !selected->revision().complete() ||
            selected->revision().interface_marker != interface_marker) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: authoritative free-surface "
                "snapshot identity is unavailable or incomplete");
        }
        const auto& revision = selected->revision();
        identity.kind = SmallCutAggregationGeometryIdentityKind::
            AuthoritativeFreeSurfaceSnapshot;
        identity.available = true;
        identity.source_id = revision.source_id;
        identity.domain_id = revision.domain_id;
        identity.interface_marker = revision.interface_marker;
        identity.isovalue = revision.isovalue;
        identity.source_layout_revision =
            revision.source_layout_revision;
        identity.source_value_revision = revision.source_value_revision;
        identity.quadrature_policy_key = revision.quadrature_policy_key;
        identity.snapshot_revision_key = revision.snapshot_revision_key;
        identity.distributed_mesh_geometry_revision =
            revision.mesh_geometry_revision;
        identity.distributed_mesh_topology_revision =
            revision.mesh_topology_revision;
        identity.distributed_ownership_revision =
            revision.ownership_revision;
        identity.distributed_numbering_revision =
            revision.numbering_revision;
        const auto& local_mesh = selected->localMeshRevision();
        lineage.has_snapshot_local_mesh_revision = true;
        lineage.snapshot_local_mesh_geometry_revision =
            local_mesh.mesh_geometry_revision;
        lineage.snapshot_local_mesh_topology_revision =
            local_mesh.mesh_topology_revision;
        lineage.snapshot_local_ownership_revision =
            local_mesh.ownership_revision;
        lineage.snapshot_local_numbering_revision =
            local_mesh.numbering_revision;
    } else if (publication != nullptr) {
        const auto& request = publication->request;
        const auto source_id = request.source.identifier();
        const auto domain_id =
            publication->publication_domain_id.empty()
                ? (request.generated_domain_id.empty()
                       ? source_id
                       : request.generated_domain_id)
                : publication->publication_domain_id;
        if (publication->generated_interface_marker == interface_marker &&
            request.interface_marker == interface_marker &&
            request.source.valid() && !source_id.empty() &&
            !domain_id.empty() && std::isfinite(request.isovalue) &&
            request.source.value_revision != 0u) {
            identity.kind = SmallCutAggregationGeometryIdentityKind::
                GeneratedPublicationSource;
            identity.available = true;
            identity.source_id = source_id;
            identity.domain_id = domain_id;
            identity.interface_marker = request.interface_marker;
            identity.isovalue = request.isovalue;
            identity.source_layout_revision =
                request.source.layout_revision;
            identity.source_value_revision = request.source.value_revision;
            identity.quadrature_policy_key =
                request.quadrature_policy_key;
        }
    }
    finalizeGeometryIdentityFingerprint(identity);
    return result;
}

void coordinateRefreshPublicationConsensus(
    const systems::FESystem& system,
    const std::exception_ptr& local_exception,
    std::uint64_t canonical_feature_class_fingerprint,
    std::uint64_t canonical_slave_set_fingerprint,
    std::uint64_t canonical_geometry_fingerprint,
    std::uint64_t successful_publication_ordinal)
{
    bool any_failed = local_exception != nullptr;
    bool fingerprints_differ = false;
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        int world_size = 1;
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            const std::array<std::uint64_t, 9> local{{
                local_exception == nullptr ? 1u : 0u,
                canonical_feature_class_fingerprint,
                ~canonical_feature_class_fingerprint,
                canonical_slave_set_fingerprint,
                ~canonical_slave_set_fingerprint,
                canonical_geometry_fingerprint,
                ~canonical_geometry_fingerprint,
                successful_publication_ordinal,
                ~successful_publication_ordinal,
            }};
            std::array<std::uint64_t, 9> minimum{};
            MPI_Allreduce(local.data(),
                          minimum.data(),
                          static_cast<int>(minimum.size()),
                          MPI_UINT64_T,
                          MPI_MIN,
                          comm);
            any_failed = minimum[0] == 0u;
            fingerprints_differ =
                !any_failed &&
                (minimum[1] != ~minimum[2] ||
                 minimum[3] != ~minimum[4] ||
                 minimum[5] != ~minimum[6] ||
                 minimum[7] != ~minimum[8]);
        }
    }
#endif
    if (any_failed) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "distributed_local_phase_failure phase='post_resolver_refresh_"
            "publication' another communicator rank rejected its local "
            "aggregation publication");
    }
    if (fingerprints_differ) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: canonical feature-class, slave, "
            "geometry, or publication-sequence fingerprint differs across "
            "communicator ranks");
    }
}

[[nodiscard]] SmallCutAggregationTopologyTransitionReport
buildCanonicalTopologyTransitionReport(
    const SmallCutAggregationRefreshReport& previous,
    const SmallCutAggregationRefreshReport& current,
    std::span<const GlobalIndex> previous_canonical_slaves,
    std::span<const GlobalIndex> current_canonical_slaves)
{
    if (previous.field != current.field ||
        previous.active_side != current.active_side ||
        previous.interface_marker != current.interface_marker) {
        throw std::logic_error(
            "SmallCutAggregationConstraint: cannot compare refresh reports "
            "from different field/marker/side constraints");
    }
    if (previous.local_lineage.successful_publication_ordinal == 0u ||
        previous.local_lineage.successful_publication_ordinal ==
            std::numeric_limits<std::uint64_t>::max() ||
        current.local_lineage.successful_publication_ordinal == 0u ||
        current.local_lineage.successful_publication_ordinal !=
            previous.local_lineage.successful_publication_ordinal + 1u) {
        throw std::logic_error(
            "SmallCutAggregationConstraint: topology transition requires "
            "adjacent successful publication ordinals");
    }
    const auto feature_id_less = [](const auto& lhs, const auto& rhs) {
        return lhs.stable_feature_id < rhs.stable_feature_id;
    };
    const auto valid_features = [&](const auto& features) {
        return std::is_sorted(
                   features.begin(), features.end(), feature_id_less) &&
               std::adjacent_find(
                   features.begin(),
                   features.end(),
                   [](const auto& lhs, const auto& rhs) {
                       return lhs.stable_feature_id ==
                              rhs.stable_feature_id;
                   }) == features.end();
    };
    if (!valid_features(previous.canonical_active_features) ||
        !valid_features(current.canonical_active_features) ||
        previous.canonical_active_feature_count !=
            previous.canonical_active_features.size() ||
        current.canonical_active_feature_count !=
            current.canonical_active_features.size() ||
        !std::is_sorted(previous_canonical_slaves.begin(),
                        previous_canonical_slaves.end()) ||
        !std::is_sorted(current_canonical_slaves.begin(),
                        current_canonical_slaves.end()) ||
        std::adjacent_find(previous_canonical_slaves.begin(),
                           previous_canonical_slaves.end()) !=
            previous_canonical_slaves.end() ||
        std::adjacent_find(current_canonical_slaves.begin(),
                           current_canonical_slaves.end()) !=
            current_canonical_slaves.end() ||
        previous.canonical_feature_class_fingerprint !=
            canonicalFeatureClassFingerprint(
                previous.canonical_active_features) ||
        current.canonical_feature_class_fingerprint !=
            canonicalFeatureClassFingerprint(
                current.canonical_active_features) ||
        previous.canonical_slave_set_fingerprint !=
            canonicalSlaveSetFingerprint(previous_canonical_slaves) ||
        current.canonical_slave_set_fingerprint !=
            canonicalSlaveSetFingerprint(current_canonical_slaves)) {
        throw std::logic_error(
            "SmallCutAggregationConstraint: non-canonical refresh history "
            "cannot form a topology transition ledger");
    }

    SmallCutAggregationTopologyTransitionReport result;
    result.geometry_identity_before = previous.geometry_identity;
    result.geometry_identity_after = current.geometry_identity;
    result.local_lineage_before = previous.local_lineage;
    result.local_lineage_after = current.local_lineage;
    result.canonical_feature_class_fingerprint_before =
        previous.canonical_feature_class_fingerprint;
    result.canonical_feature_class_fingerprint_after =
        current.canonical_feature_class_fingerprint;
    result.canonical_slave_set_fingerprint_before =
        previous.canonical_slave_set_fingerprint;
    result.canonical_slave_set_fingerprint_after =
        current.canonical_slave_set_fingerprint;
    result.canonical_active_feature_count_before =
        previous.canonical_active_features.size();
    result.canonical_active_feature_count_after =
        current.canonical_active_features.size();
    result.canonical_aggregate_slaves_before =
        previous_canonical_slaves.size();
    result.canonical_aggregate_slaves_after =
        current_canonical_slaves.size();
    result.canonical_rootless_active_physical_volume_before =
        previous.canonical_rootless_active_physical_volume;
    result.canonical_rootless_active_physical_volume_after =
        current.canonical_rootless_active_physical_volume;
    result.canonical_rootless_active_physical_volume_delta =
        result.canonical_rootless_active_physical_volume_after -
        result.canonical_rootless_active_physical_volume_before;
    result.canonical_feature_transitions.reserve(
        previous.canonical_active_features.size() +
        current.canonical_active_features.size());

    std::size_t previous_index = 0u;
    std::size_t current_index = 0u;
    while (previous_index < previous.canonical_active_features.size() ||
           current_index < current.canonical_active_features.size()) {
        const auto* before =
            previous_index < previous.canonical_active_features.size()
                ? &previous.canonical_active_features[previous_index]
                : nullptr;
        const auto* after =
            current_index < current.canonical_active_features.size()
                ? &current.canonical_active_features[current_index]
                : nullptr;
        const bool before_only =
            before != nullptr &&
            (after == nullptr ||
             before->stable_feature_id < after->stable_feature_id);
        const bool after_only =
            after != nullptr &&
            (before == nullptr ||
             after->stable_feature_id < before->stable_feature_id);

        SmallCutAggregationActiveFeatureTransitionReport transition;
        if (before_only) {
            transition.stable_feature_id = before->stable_feature_id;
            transition.present_before = true;
            transition.canonical_cell_gid_digest_before =
                before->canonical_cell_gid_digest;
            transition.canonical_full_active_cell_gid_digest_before =
                before->canonical_full_active_cell_gid_digest;
            transition.canonical_cut_cell_gid_digest_before =
                before->canonical_cut_cell_gid_digest;
            transition.disposition_before = before->disposition;
            transition.canonical_cell_count_before =
                before->canonical_cell_count;
            transition.canonical_full_active_cell_count_before =
                before->canonical_full_active_cell_count;
            transition.canonical_cut_cell_count_before =
                before->canonical_cut_cell_count;
            transition.canonical_retained_physical_volume_before =
                before->canonical_retained_physical_volume;
            ++result.canonical_features_exited;
            ++previous_index;
        } else if (after_only) {
            transition.stable_feature_id = after->stable_feature_id;
            transition.present_after = true;
            transition.canonical_cell_gid_digest_after =
                after->canonical_cell_gid_digest;
            transition.canonical_full_active_cell_gid_digest_after =
                after->canonical_full_active_cell_gid_digest;
            transition.canonical_cut_cell_gid_digest_after =
                after->canonical_cut_cell_gid_digest;
            transition.disposition_after = after->disposition;
            transition.canonical_cell_count_after =
                after->canonical_cell_count;
            transition.canonical_full_active_cell_count_after =
                after->canonical_full_active_cell_count;
            transition.canonical_cut_cell_count_after =
                after->canonical_cut_cell_count;
            transition.canonical_retained_physical_volume_after =
                after->canonical_retained_physical_volume;
            ++result.canonical_features_entered;
            ++current_index;
        } else {
            if (before == nullptr || after == nullptr ||
                before->stable_feature_id != after->stable_feature_id) {
                throw std::logic_error(
                    "SmallCutAggregationConstraint: invalid feature merge "
                    "while forming a topology transition ledger");
            }
            transition.stable_feature_id = before->stable_feature_id;
            transition.present_before = true;
            transition.present_after = true;
            transition.canonical_cell_gid_digest_before =
                before->canonical_cell_gid_digest;
            transition.canonical_cell_gid_digest_after =
                after->canonical_cell_gid_digest;
            transition.canonical_full_active_cell_gid_digest_before =
                before->canonical_full_active_cell_gid_digest;
            transition.canonical_full_active_cell_gid_digest_after =
                after->canonical_full_active_cell_gid_digest;
            transition.canonical_cut_cell_gid_digest_before =
                before->canonical_cut_cell_gid_digest;
            transition.canonical_cut_cell_gid_digest_after =
                after->canonical_cut_cell_gid_digest;
            transition.disposition_before = before->disposition;
            transition.disposition_after = after->disposition;
            transition.canonical_cell_count_before =
                before->canonical_cell_count;
            transition.canonical_cell_count_after =
                after->canonical_cell_count;
            transition.canonical_full_active_cell_count_before =
                before->canonical_full_active_cell_count;
            transition.canonical_full_active_cell_count_after =
                after->canonical_full_active_cell_count;
            transition.canonical_cut_cell_count_before =
                before->canonical_cut_cell_count;
            transition.canonical_cut_cell_count_after =
                after->canonical_cut_cell_count;
            transition.canonical_retained_physical_volume_before =
                before->canonical_retained_physical_volume;
            transition.canonical_retained_physical_volume_after =
                after->canonical_retained_physical_volume;
            transition.cell_classification_changed =
                before->canonical_cell_gid_digest !=
                    after->canonical_cell_gid_digest ||
                before->canonical_full_active_cell_gid_digest !=
                    after->canonical_full_active_cell_gid_digest ||
                before->canonical_cut_cell_gid_digest !=
                    after->canonical_cut_cell_gid_digest ||
                before->canonical_cell_count !=
                    after->canonical_cell_count ||
                before->canonical_full_active_cell_count !=
                    after->canonical_full_active_cell_count ||
                before->canonical_cut_cell_count !=
                    after->canonical_cut_cell_count;
            transition.disposition_changed =
                before->disposition != after->disposition;
            ++result.canonical_features_persisted;
            if (transition.cell_classification_changed) {
                ++result.canonical_feature_classification_changes;
            }
            if (transition.disposition_changed) {
                if (after->disposition ==
                    SmallCutAggregationActiveFeatureDisposition::Rooted) {
                    ++result.canonical_features_became_rooted;
                } else {
                    ++result.canonical_features_became_rootless;
                }
            }
            ++previous_index;
            ++current_index;
        }
        result.canonical_feature_transitions.push_back(
            std::move(transition));
    }

    std::size_t previous_slave_index = 0u;
    std::size_t current_slave_index = 0u;
    while (previous_slave_index < previous_canonical_slaves.size() ||
           current_slave_index < current_canonical_slaves.size()) {
        if (current_slave_index >= current_canonical_slaves.size() ||
            (previous_slave_index < previous_canonical_slaves.size() &&
             previous_canonical_slaves[previous_slave_index] <
                 current_canonical_slaves[current_slave_index])) {
            ++result.canonical_aggregate_slaves_left;
            ++previous_slave_index;
        } else if (
            previous_slave_index >= previous_canonical_slaves.size() ||
            current_canonical_slaves[current_slave_index] <
                previous_canonical_slaves[previous_slave_index]) {
            ++result.canonical_aggregate_slaves_entered;
            ++current_slave_index;
        } else {
            ++previous_slave_index;
            ++current_slave_index;
        }
    }

    result.canonical_topology_changed =
        result.canonical_features_entered > 0u ||
        result.canonical_features_exited > 0u ||
        result.canonical_feature_classification_changes > 0u ||
        result.canonical_features_became_rooted > 0u ||
        result.canonical_features_became_rootless > 0u ||
        result.canonical_aggregate_slaves_entered > 0u ||
        result.canonical_aggregate_slaves_left > 0u;
    return result;
}

// ---------------------------------------------------------------------------
// Reuse of an unchanged refresh.

[[nodiscard]] bool aggregationReuseDisabled()
{
    const char* value = std::getenv("SVMP_DISABLE_SMALL_CUT_AGGREGATION_REUSE");
    const char* dump = std::getenv("SVMP_AGGREGATION_DUMP");
    const auto set = [](const char* v) {
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    };
    return set(value) || set(dump);
}

// Two independent 64-bit digests of one word sequence.
class AggregationReuseDigest {
public:
    void mix(std::uint64_t word) noexcept
    {
        a_ ^= word;
        a_ *= 1099511628211ull;
        word ^= word >> 30;
        word *= 0xbf58476d1ce4e5b9ull;
        word ^= word >> 27;
        word *= 0x94d049bb133111ebull;
        word ^= word >> 31;
        b_ = std::rotl(b_ ^ word, 27) * 0x9e3779b185ebca87ull +
             0x632be59bd9b4e019ull;
    }

    void mixReal(double value) noexcept
    {
        mix(std::bit_cast<std::uint64_t>(value));
    }

    [[nodiscard]] std::array<std::uint64_t, 2> value() const noexcept
    {
        return {{a_, b_}};
    }

private:
    std::uint64_t a_{14695981039346656037ull};
    std::uint64_t b_{0x9e3779b97f4a7c15ull};
};

[[nodiscard]] std::uint64_t aggregationReuseAvalanche(
    std::uint64_t word) noexcept
{
    word += 0x9e3779b97f4a7c15ull;
    word ^= word >> 30;
    word *= 0xbf58476d1ce4e5b9ull;
    word ^= word >> 27;
    word *= 0x94d049bb133111ebull;
    word ^= word >> 31;
    return word;
}

/**
 * Computes the rank-local reuse key and the retained-volume digests. The
 * classification pass mirrors the retained-rule classification of apply():
 * it reads the same rules in the same order and applies the same validity
 * tests, so equal digests imply equal cell classes, retained-rule identities
 * and per-cell rule sequences. Returns nullopt when reuse is not admissible
 * (debug knobs, missing revision tracking, or rule data that apply() would
 * reject, so that the full path reports the failure).
 */
[[nodiscard]] std::optional<detail::SmallCutAggregationReuseInputs>
computeAggregationReuseInputs(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext& cut_context,
    const AffineConstraints& constraints,
    FieldId field,
    geometry::CutIntegrationSide active_side,
    int interface_marker,
    std::span<const int> excluded_boundary_markers,
    std::span<const GlobalIndex> excluded_vertices,
    const SmallCutAggregationGuardOptions& guards,
    const AggregationRuntimeOptions& runtime_options)
{
    if (aggregationReuseDisabled() || !runtime_options.flags_valid ||
        !runtime_options.max_lines_valid ||
        runtime_options.allow_unaggregated ||
        runtime_options.max_lines !=
            std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    const auto& mesh = system.meshAccess();
    if (!mesh.revisionTrackingAvailable()) {
        return std::nullopt;
    }

    AggregationReuseDigest classification;
    AggregationReuseDigest geometry_digest;
    const auto& metadata = cut_context.metadata();
    const auto& rules = cut_context.volumeRules();
    const auto inactive_side =
        active_side == geometry::CutIntegrationSide::Negative
            ? geometry::CutIntegrationSide::Positive
            : geometry::CutIntegrationSide::Negative;
    const bool global_ids = mesh.globalEntityIdsAvailable();
    std::vector<GlobalIndex> nodes;
    classification.mix(static_cast<std::uint64_t>(
        static_cast<std::uint32_t>(interface_marker)));
    for (const auto side : {active_side, inactive_side}) {
        const bool active = side == active_side;
        classification.mix(active ? 0x41435456u : 0x494e4143u);
        for (const auto index :
             cut_context.generatedVolumeRuleIndicesForMarkerAndSide(
                 interface_marker, side)) {
            if (index >= rules.size() || index >= metadata.size()) {
                classification.mix(0x534b4950u);
                continue;
            }
            const auto& meta = metadata[index];
            const auto& rule = rules[index];
            const auto fraction = rule.volume_fraction;
            const auto metadata_fraction = meta.volume_fraction;
            if (!std::isfinite(fraction) || fraction < Real(0) ||
                fraction > Real(1) || !std::isfinite(metadata_fraction) ||
                metadata_fraction < Real(0) || metadata_fraction > Real(1)) {
                return std::nullopt;
            }
            const auto fraction_scale =
                std::max({Real(1), std::abs(fraction),
                          std::abs(metadata_fraction)});
            if (std::abs(fraction - metadata_fraction) >
                Real(64) * std::numeric_limits<Real>::epsilon() *
                    fraction_scale) {
                return std::nullopt;
            }
            const auto cell = meta.cell >= 0 ? meta.cell : meta.parent_entity;
            if (cell < 0) {
                classification.mix(0x4e4f4345u);
                continue;
            }
            const bool full = rule.full_cell_equivalent;
            const bool partial = retainedPartialVolumeRule(full, fraction);
            const bool retained = active && (full || partial);
            classification.mix(static_cast<std::uint64_t>(cell));
            classification.mix((full ? 1u : 0u) | (partial ? 2u : 0u) |
                               (retained ? 4u : 0u) | (active ? 8u : 0u));
            if (!retained) {
                continue;
            }
            // Retained-rule identities embed the source value revision and
            // change with every rebuild; only their availability is part of
            // the classification. The identities themselves are refreshed on
            // reuse together with the retained volumes.
            classification.mix(
                rule.provenance.cut_topology_revision == 0u ? 0u : 1u);

            const auto local_cell = static_cast<GlobalIndex>(cell);
            geometry_digest.mix(static_cast<std::uint64_t>(local_cell));
            geometry_digest.mix(static_cast<std::uint64_t>(
                global_ids ? mesh.getCellGlobalId(local_cell) : local_cell));
            geometry_digest.mix(
                static_cast<std::uint64_t>(mesh.getCellType(local_cell)));
            mesh.getCellNodes(local_cell, nodes);
            geometry_digest.mix(static_cast<std::uint64_t>(nodes.size()));
            for (const auto node : nodes) {
                const auto xyz = mesh.getNodeCoordinates(node);
                geometry_digest.mixReal(static_cast<double>(xyz[0]));
                geometry_digest.mixReal(static_cast<double>(xyz[1]));
                geometry_digest.mixReal(static_cast<double>(xyz[2]));
            }
            geometry_digest.mix(rule.provenance.cut_topology_revision);
            geometry_digest.mix(static_cast<std::uint64_t>(rule.kind));
            geometry_digest.mix(static_cast<std::uint64_t>(rule.frame));
            geometry_digest.mix(full ? 1u : 0u);
            geometry_digest.mixReal(static_cast<double>(rule.measure));
            geometry_digest.mix(static_cast<std::uint64_t>(rule.points.size()));
            for (const auto& point : rule.points) {
                for (std::size_t d = 0; d < 3u; ++d) {
                    geometry_digest.mixReal(static_cast<double>(point.point[d]));
                    geometry_digest.mixReal(
                        static_cast<double>(point.parent_coordinate[d]));
                }
                geometry_digest.mixReal(static_cast<double>(point.weight));
            }
        }
    }

    AggregationReuseDigest inputs;
    const auto& record = system.fieldRecord(field);
    const auto& dofs = system.fieldDofHandler(field);
    const auto* entity_map = dofs.getEntityDofMap();
    if (!record.space || entity_map == nullptr) {
        return std::nullopt;
    }
    inputs.mix(static_cast<std::uint64_t>(field));
    inputs.mix(system.spaceRevision());
    inputs.mix(static_cast<std::uint64_t>(system.fieldDofOffset(field)));
    inputs.mix(static_cast<std::uint64_t>(record.components));
    inputs.mix(static_cast<std::uint64_t>(
        record.space->element().basis().size()));
    inputs.mix(static_cast<std::uint64_t>(dofs.getNumDofs()));
    const auto n_cells = dofs.getDofMap().getNumCells();
    inputs.mix(static_cast<std::uint64_t>(n_cells));
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        const auto cell_dofs = dofs.getCellDofs(cell);
        inputs.mix(static_cast<std::uint64_t>(cell_dofs.size()));
        for (const auto dof : cell_dofs) {
            inputs.mix(static_cast<std::uint64_t>(dof));
        }
    }
    const auto n_vertices = entity_map->numVertices();
    inputs.mix(static_cast<std::uint64_t>(n_vertices));
    for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        inputs.mix(static_cast<std::uint64_t>(vertex_dofs.size()));
        for (const auto dof : vertex_dofs) {
            inputs.mix(static_cast<std::uint64_t>(dof));
        }
    }
    const auto& partition = system.dofHandler().getPartition();
    inputs.mix(static_cast<std::uint64_t>(partition.locallyOwned().size()));
    for (const auto dof : partition.locallyOwned()) {
        inputs.mix(static_cast<std::uint64_t>(dof));
    }
    inputs.mix(static_cast<std::uint64_t>(partition.ghost().size()));
    for (const auto dof : partition.ghost()) {
        inputs.mix(static_cast<std::uint64_t>(dof));
    }
    // The assembly ghost policy selects which non-owned candidate DOFs this
    // rank condenses. It cannot change the result when every relevant DOF is
    // owned (one rank), and it differs between the constraint pass inside
    // setup() and later rebuilds, so it is keyed only where ghosts exist.
    inputs.mix(partition.ghost().size() > 0
                   ? static_cast<std::uint64_t>(system.assemblyGhostPolicy()) +
                         1u
                   : 0u);

    inputs.mix(mesh.geometryRevision());
    inputs.mix(mesh.topologyRevision());
    inputs.mix(mesh.ownershipRevision());
    inputs.mix(mesh.numberingRevision());
    inputs.mix(mesh.labelRevision());
    inputs.mix(mesh.activeConfigurationEpoch());
    inputs.mix(mesh.coordinateConfigurationKey());
    inputs.mix(static_cast<std::uint64_t>(mesh.numCells()));
    inputs.mix(global_ids ? 1u : 0u);
    inputs.mix(static_cast<std::uint64_t>(mesh.parallelRank()));
    inputs.mix(static_cast<std::uint64_t>(mesh.parallelSize()));

    // apply() queries the incoming constraints only through isConstrained(),
    // so the constrained set (order-independent) is the complete dependency.
    std::uint64_t constrained_count = 0u;
    std::uint64_t constrained_sum = 0u;
    std::uint64_t constrained_xor = 0u;
    constraints.forEach([&](const AffineConstraints::ConstraintView& line) {
        const auto word = static_cast<std::uint64_t>(line.slave_dof);
        ++constrained_count;
        constrained_sum += aggregationReuseAvalanche(word);
        constrained_xor ^= aggregationReuseAvalanche(word ^ 0x5a5a5a5aull);
    });
    inputs.mix(constrained_count);
    inputs.mix(constrained_sum);
    inputs.mix(constrained_xor);

    inputs.mix(static_cast<std::uint64_t>(excluded_boundary_markers.size()));
    for (const auto marker : excluded_boundary_markers) {
        inputs.mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(marker)));
    }
    inputs.mix(static_cast<std::uint64_t>(excluded_vertices.size()));
    for (const auto vertex : excluded_vertices) {
        inputs.mix(static_cast<std::uint64_t>(vertex));
    }
    inputs.mix(static_cast<std::uint64_t>(guards.maximum_root_path_length));
    inputs.mixReal(static_cast<double>(
        guards.maximum_reference_extrapolation_distance));
    inputs.mixReal(static_cast<double>(guards.maximum_absolute_coefficient));
    inputs.mixReal(static_cast<double>(guards.maximum_row_l1_norm));
    inputs.mix(runtime_options.slave_all_cut ? 1u : 0u);
    inputs.mix(runtime_options.linear_extension ? 1u : 0u);

    detail::SmallCutAggregationReuseInputs result;
    result.classification = classification.value();
    result.geometry_digest = geometry_digest.value();
    result.key.topology = result.classification;
    result.key.inputs = inputs.value();
    return result;
}

// Collective reuse decision: a rank reuses only when every rank of the field
// communicator can. Returns {reuse record, reuse shared retained measures}.
[[nodiscard]] std::array<bool, 2> agreeOnAggregationReuse(
    const systems::FESystem& system,
    bool local_record_matches,
    bool local_shared_measures_available)
{
    std::array<int, 2> agreed{{
        local_record_matches ? 1 : 0,
        local_record_matches && local_shared_measures_available ? 1 : 0}};
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        int world_size = 1;
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            const auto local = agreed;
            MPI_Allreduce(local.data(),
                          agreed.data(),
                          static_cast<int>(agreed.size()),
                          MPI_INT,
                          MPI_MIN,
                          comm);
        }
    }
#else
    static_cast<void>(system);
#endif
    return {{agreed[0] != 0, agreed[1] != 0}};
}

// Canonical retained measures of the last reuse, shared by the constraints of
// the other fields of the same marker (velocity and pressure apply the same
// classification back to back). Keyed only by content digests, so an entry
// can never describe another mesh or context.
struct SharedRetainedMeasureEntry {
    std::array<std::uint64_t, 2> classification{};
    std::array<std::uint64_t, 2> geometry_digest{};
    int interface_marker{-1};
    geometry::CutIntegrationSide active_side{
        geometry::CutIntegrationSide::Negative};
    int rank{0};
    int world_size{1};
    std::vector<GlobalIndex> cell_gids{};
    detail::SmallCutAggregationRetainedMeasures measures{};
};

std::mutex& sharedRetainedMeasureMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::optional<SharedRetainedMeasureEntry>& sharedRetainedMeasureEntry()
{
    static std::optional<SharedRetainedMeasureEntry> entry;
    return entry;
}

[[nodiscard]] std::optional<detail::SmallCutAggregationRetainedMeasures>
lookupSharedRetainedMeasures(
    const systems::FESystem& system,
    const detail::SmallCutAggregationReuseInputs& inputs,
    int interface_marker,
    geometry::CutIntegrationSide active_side,
    std::span<const SmallCutAggregationProlongationCell> active_cells)
{
    const auto& mesh = system.meshAccess();
    std::lock_guard<std::mutex> lock(sharedRetainedMeasureMutex());
    const auto& entry = sharedRetainedMeasureEntry();
    if (!entry.has_value() ||
        entry->classification != inputs.classification ||
        entry->geometry_digest != inputs.geometry_digest ||
        entry->interface_marker != interface_marker ||
        entry->active_side != active_side ||
        entry->rank != mesh.parallelRank() ||
        entry->world_size != mesh.parallelSize() ||
        entry->cell_gids.size() != active_cells.size()) {
        return std::nullopt;
    }
    for (std::size_t i = 0u; i < active_cells.size(); ++i) {
        if (entry->cell_gids[i] != active_cells[i].cell_gid) {
            return std::nullopt;
        }
    }
    return entry->measures;
}

void storeSharedRetainedMeasures(
    const systems::FESystem& system,
    const detail::SmallCutAggregationReuseInputs& inputs,
    int interface_marker,
    geometry::CutIntegrationSide active_side,
    std::span<const SmallCutAggregationProlongationCell> active_cells,
    const detail::SmallCutAggregationRetainedMeasures& measures)
{
    SharedRetainedMeasureEntry entry;
    entry.classification = inputs.classification;
    entry.geometry_digest = inputs.geometry_digest;
    entry.interface_marker = interface_marker;
    entry.active_side = active_side;
    entry.rank = system.meshAccess().parallelRank();
    entry.world_size = system.meshAccess().parallelSize();
    entry.cell_gids.reserve(active_cells.size());
    for (const auto& cell : active_cells) {
        entry.cell_gids.push_back(cell.cell_gid);
    }
    entry.measures = measures;
    std::lock_guard<std::mutex> lock(sharedRetainedMeasureMutex());
    sharedRetainedMeasureEntry() = std::move(entry);
}

/**
 * Recomputes the communicator-canonical retained physical volume and the
 * retained-rule identities of every canonical active cell exactly as apply()
 * does: per-cell sums over the retained active rules in rule order, sorted
 * identities with unavailable (zero) identities first, the declaration of the
 * recorded (lowest-rank) provider, and agreement of every provider.
 * Collective on a distributed field communicator.
 */
[[nodiscard]] detail::SmallCutAggregationRetainedMeasures
refreshRetainedActiveCellMeasures(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext& cut_context,
    int interface_marker,
    geometry::CutIntegrationSide active_side,
    std::span<const SmallCutAggregationProlongationCell> active_cells)
{
    const auto& mesh = system.meshAccess();
    int my_rank = 0;
#if FE_HAS_MPI
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized != 0) {
            MPI_Comm_rank(system.dofHandler().mpiComm(), &my_rank);
        }
    }
#endif
    // Only the recorded canonical provider of an active cell sends its
    // measure; every other rank that retains the cell compares its own
    // measure with the provider's exactly after the gather.  The canonical
    // measure, and every disagreement check, is the same as when every
    // retaining rank sent its measure.
    const auto provider_of = [&](GlobalIndex gid) -> int {
        const auto it = std::lower_bound(
            active_cells.begin(), active_cells.end(), gid,
            [](const SmallCutAggregationProlongationCell& cell, GlobalIndex key) {
                return cell.cell_gid < key;
            });
        if (it == active_cells.end() || it->cell_gid != gid) {
            return -1;
        }
        return it->retained_measure_provider_rank;
    };
    struct RetainedCellMeasure {
        Real volume{0.0};
        std::vector<std::uint64_t> stable_rule_ids{};
    };
    std::vector<std::pair<GlobalIndex, RetainedCellMeasure>> local_non_provider;
    std::vector<std::int64_t> local_words;
    std::exception_ptr local_measure_exception;
    try {
        if (!std::is_sorted(active_cells.begin(), active_cells.end(),
                            [](const auto& lhs, const auto& rhs) {
                                return lhs.cell_gid < rhs.cell_gid;
                            })) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: reused active-cell ledger is "
                "not sorted by physical cell ID");
        }
        struct LocalMeasure {
            Real physical_volume{0.0};
            std::set<std::uint64_t> stable_rule_ids{};
            std::size_t missing_stable_rule_ids{0u};
        };
        std::unordered_map<GlobalIndex, LocalMeasure> local_measures;
        std::vector<GlobalIndex> local_order;
        const auto& metadata = cut_context.metadata();
        const auto& rules = cut_context.volumeRules();
        for (const auto index :
             cut_context.generatedVolumeRuleIndicesForMarkerAndSide(
                 interface_marker, active_side)) {
            if (index >= rules.size() || index >= metadata.size()) {
                continue;
            }
            const auto& meta = metadata[index];
            const auto& rule = rules[index];
            const auto fraction = rule.volume_fraction;
            const auto cell = meta.cell >= 0 ? meta.cell : meta.parent_entity;
            if (cell < 0) {
                continue;
            }
            if (!rule.full_cell_equivalent &&
                !retainedPartialVolumeRule(rule.full_cell_equivalent,
                                           fraction)) {
                continue;
            }
            const auto local_cell = static_cast<GlobalIndex>(cell);
            const auto physical_volume =
                physicalRetainedVolumeRuleMeasure(mesh, local_cell, rule);
            if (!(physical_volume > Real{0.0}) ||
                !std::isfinite(physical_volume)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume retained active-side "
                    "rule has no positive physical measure");
            }
            const auto [measure, inserted] =
                local_measures.try_emplace(local_cell);
            if (inserted) {
                local_order.push_back(local_cell);
            }
            const auto stable_rule_id = rule.provenance.cut_topology_revision;
            if (stable_rule_id == 0u) {
                ++measure->second.missing_stable_rule_ids;
            } else if (!measure->second.stable_rule_ids.insert(
                           stable_rule_id).second) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "duplicate_active_feature_volume_rule retained rules "
                    "for one cell repeat a nonzero stable identity");
            }
            measure->second.physical_volume += physical_volume;
            if (!std::isfinite(measure->second.physical_volume)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume per-cell physical "
                    "measure overflow");
            }
        }
        const bool global_ids = mesh.globalEntityIdsAvailable();
        for (const auto cell : local_order) {
            const auto gid = global_ids ? mesh.getCellGlobalId(cell) : cell;
            if (gid < 0) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume physical cell ID is "
                    "unavailable");
            }
            const auto& measure = local_measures.at(cell);
            const int provider = provider_of(gid);
            if (provider != my_rank) {
                if (provider >= 0) {
                    RetainedCellMeasure own;
                    own.volume = measure.physical_volume;
                    own.stable_rule_ids.assign(
                        measure.missing_stable_rule_ids, std::uint64_t{0});
                    own.stable_rule_ids.insert(own.stable_rule_ids.end(),
                                               measure.stable_rule_ids.begin(),
                                               measure.stable_rule_ids.end());
                    local_non_provider.emplace_back(gid, std::move(own));
                }
                continue;
            }
            local_words.push_back(static_cast<std::int64_t>(gid));
            local_words.push_back(std::bit_cast<std::int64_t>(
                static_cast<double>(measure.physical_volume)));
            local_words.push_back(static_cast<std::int64_t>(
                measure.missing_stable_rule_ids +
                measure.stable_rule_ids.size()));
            for (std::size_t missing = 0u;
                 missing < measure.missing_stable_rule_ids;
                 ++missing) {
                local_words.push_back(0);
            }
            for (const auto stable_id : measure.stable_rule_ids) {
                local_words.push_back(std::bit_cast<std::int64_t>(stable_id));
            }
        }
    } catch (...) {
        local_measure_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_measure_exception,
                                      "reused_retained_measure_refresh");

    GatheredInt64Words gathered;
    int world_size = 1;
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            gathered = allGatherInt64Words(
                comm, std::span<const std::int64_t>(local_words),
                AggregationProfileExchange::ReusedMeasures);
        }
    }
#endif
    if (world_size == 1) {
        gathered.words = local_words;
        gathered.counts = {static_cast<int>(local_words.size())};
        gathered.displacements = {0};
    }

    detail::SmallCutAggregationRetainedMeasures measures;
    std::exception_ptr local_decode_exception;
    try {
        struct ProviderMeasure {
            int rank{-1};
            Real volume{0.0};
            std::vector<std::uint64_t> stable_rule_ids{};
        };
        std::unordered_map<GlobalIndex, std::vector<ProviderMeasure>>
            providers_by_gid;
        for (int rank = 0; rank < world_size; ++rank) {
            auto position = static_cast<std::size_t>(
                gathered.displacements[static_cast<std::size_t>(rank)]);
            const auto end = position + static_cast<std::size_t>(
                gathered.counts[static_cast<std::size_t>(rank)]);
            while (position < end) {
                if (end - position < 3u) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: malformed "
                        "distributed reused retained-measure payload");
                }
                ProviderMeasure provider;
                provider.rank = rank;
                const auto gid =
                    static_cast<GlobalIndex>(gathered.words[position++]);
                provider.volume = static_cast<Real>(
                    std::bit_cast<double>(gathered.words[position++]));
                const auto id_count = gathered.words[position++];
                if (id_count < 0 ||
                    static_cast<std::uint64_t>(id_count) >
                        static_cast<std::uint64_t>(end - position)) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: malformed "
                        "distributed reused retained-rule identity "
                        "payload");
                }
                provider.stable_rule_ids.reserve(
                    static_cast<std::size_t>(id_count));
                for (std::int64_t i = 0; i < id_count; ++i) {
                    provider.stable_rule_ids.push_back(
                        std::bit_cast<std::uint64_t>(
                            gathered.words[position++]));
                }
                providers_by_gid[gid].push_back(std::move(provider));
            }
        }
        measures.volumes.reserve(active_cells.size());
        measures.stable_rule_ids.reserve(active_cells.size());
        for (const auto& cell : active_cells) {
            const auto providers = providers_by_gid.find(cell.cell_gid);
            if (providers == providers_by_gid.end()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume active cell has no "
                    "communicator-visible measure provider");
            }
            const auto canonical = std::find_if(
                providers->second.begin(),
                providers->second.end(),
                [&](const ProviderMeasure& provider) {
                    return provider.rank ==
                           cell.retained_measure_provider_rank;
                });
            if (canonical == providers->second.end()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume canonical measure "
                    "provider is missing");
            }
            for (const auto& provider : providers->second) {
                if (provider.volume != canonical->volume ||
                    provider.stable_rule_ids != canonical->stable_rule_ids) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: diagnostic="
                        "inconsistent_distributed_active_feature_volume "
                        "providers disagree for the same physical cell "
                        "cell_gid=" + std::to_string(cell.cell_gid));
                }
            }
            if (!(canonical->volume > Real{0.0}) ||
                !std::isfinite(canonical->volume) ||
                canonical->stable_rule_ids.empty()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume active cell has no "
                    "complete physical measure declaration");
            }
            measures.volumes.push_back(canonical->volume);
            measures.stable_rule_ids.push_back(canonical->stable_rule_ids);
        }
        // Exact comparison of this rank's own measures of cells provided by
        // another rank (the cross-check every retaining rank used to make on
        // the gathered declarations).
        for (const auto& [gid, own] : local_non_provider) {
            const auto providers = providers_by_gid.find(gid);
            if (providers == providers_by_gid.end()) {
                continue;  // reported above for the active cell
            }
            for (const auto& provider : providers->second) {
                if (provider.volume != own.volume ||
                    provider.stable_rule_ids != own.stable_rule_ids) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: diagnostic="
                        "inconsistent_distributed_active_feature_volume "
                        "providers disagree for the same physical cell "
                        "cell_gid=" + std::to_string(gid) +
                        " retaining_rank=" + std::to_string(my_rank) +
                        " provider_rank=" + std::to_string(provider.rank));
                }
            }
        }
    } catch (...) {
        local_decode_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(system,
                                      local_decode_exception,
                                      "reused_retained_measure_decode");
    return measures;
}

// Per-cell view of communicator-classified cells in ascending key order (the
// index of a cell in every array).  The communicator-global path fills it
// with every classified cell; the owner-sourced path with the active cells
// and their inactive face neighbors (see ownerSourcedActiveCellGraph).
struct AggregationCellGraph {
    const ClassifiedCellIndex& cells;
    const std::vector<CellClass>& klass;
    const std::vector<GlobalIndex>& physical_gid;
    const std::vector<Real>& volume;
    const std::vector<std::vector<std::uint64_t>>& rule_ids;
    const std::vector<int>& provider_rank;
    const std::vector<int>& owner_rank;
    const std::vector<std::vector<std::size_t>>& neighbors;
};

struct ActiveFeatureLedger {
    std::vector<SmallCutAggregationActiveFeatureReport> features{};
    // Canonical feature ID per graph cell (INVALID for inactive cells).
    std::vector<GlobalIndex> feature_of{};
    std::vector<SmallCutAggregationProlongationCell> cells{};
    std::size_t rooted{0u};
    std::size_t rootless{0u};
    Real rootless_volume{0.0};
};

// Canonical retained active features: face-connected components of the
// active cells of the graph, with their digests, counts and volumes, and the
// canonical active-cell records.  Only active cells and their active
// neighbors are read.
void buildActiveFeatureLedger(const AggregationCellGraph& graph,
                              ActiveFeatureLedger& ledger)
{
    const auto& classified_cells = graph.cells;
    const auto& global_cell_class_of = graph.klass;
    const auto& global_physical_cell_gid_of = graph.physical_gid;
    const auto& global_active_cell_physical_volume_of = graph.volume;
    const auto& global_active_cell_rule_ids_of = graph.rule_ids;
    const auto& global_active_cell_measure_provider_rank_of =
        graph.provider_rank;
    const auto& global_cell_owner_rank_of = graph.owner_rank;
    const auto& global_neighbors = graph.neighbors;
    auto& canonical_active_features = ledger.features;
    auto& active_feature_of = ledger.feature_of;
    auto& canonical_active_cells = ledger.cells;
    auto& canonical_rooted_active_feature_count = ledger.rooted;
    auto& canonical_rootless_active_feature_count = ledger.rootless;
    auto& canonical_rootless_active_physical_volume = ledger.rootless_volume;
    const auto n_classified = classified_cells.size();
    const auto is_active_cell = [&](std::size_t cell) {
        return global_cell_class_of[cell].full_active ||
               global_cell_class_of[cell].cut;
    };
    std::set<GlobalIndex> active_physical_cell_gids;
    for (std::size_t cell = 0u; cell < n_classified; ++cell) {
        if (!is_active_cell(cell)) {
            continue;
        }
        const auto gid = global_physical_cell_gid_of[cell];
        if (gid < 0 || !active_physical_cell_gids.insert(gid).second) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "invalid_active_feature_volume communicator active cells "
                "have missing or duplicate physical cell IDs");
        }
    }
    active_feature_of.assign(n_classified, INVALID_GLOBAL_INDEX);
    std::vector<char> visited_active_cells(n_classified, 0);
    std::vector<std::size_t> queue;
    std::vector<std::size_t> component_cells;
    std::vector<std::pair<GlobalIndex, std::size_t>> component_order;
    for (std::size_t seed = 0u; seed < n_classified; ++seed) {
        if (!is_active_cell(seed) || visited_active_cells[seed] != 0) {
            continue;
        }

        queue.clear();
        component_cells.clear();
        queue.push_back(seed);
        visited_active_cells[seed] = 1;
        for (std::size_t head = 0u; head < queue.size(); ++head) {
            const auto cell = queue[head];
            component_cells.push_back(cell);
            for (const auto neighbor : global_neighbors[cell]) {
                if (!is_active_cell(neighbor)) {
                    continue;
                }
                if (visited_active_cells[neighbor] == 0) {
                    visited_active_cells[neighbor] = 1;
                    queue.push_back(neighbor);
                }
            }
        }

        // Order by (physical cell GID, key); index order is key order and
        // the pairs are unique, so the order is the same for any sort.
        component_order.clear();
        for (const auto cell : component_cells) {
            component_order.emplace_back(global_physical_cell_gid_of[cell],
                                         cell);
        }
        std::sort(component_order.begin(), component_order.end());
        for (std::size_t i = 0u; i < component_order.size(); ++i) {
            component_cells[i] = component_order[i].second;
        }
        if (component_cells.empty()) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: active feature traversal "
                "produced an empty component");
        }

        SmallCutAggregationActiveFeatureReport feature;
        feature.stable_feature_id =
            global_physical_cell_gid_of[component_cells.front()];
        for (const auto cell : component_cells) {
            if (active_feature_of[cell] != INVALID_GLOBAL_INDEX) {
                throw std::logic_error(
                    "SmallCutAggregationConstraint: active cell was assigned "
                    "to more than one canonical feature");
            }
            active_feature_of[cell] = feature.stable_feature_id;
        }
        feature.canonical_cell_count = component_cells.size();
        feature.canonical_cell_gid_digest = 14695981039346656037ull;
        feature.canonical_full_active_cell_gid_digest =
            14695981039346656037ull;
        feature.canonical_cut_cell_gid_digest =
            14695981039346656037ull;
        aggregationDigestMix(
            feature.canonical_full_active_cell_gid_digest,
            0x46554c4c41435449ull);
        aggregationDigestMix(
            feature.canonical_cut_cell_gid_digest,
            0x43555443454c4c53ull);
        long double physical_volume = 0.0L;
        GlobalIndex previous_gid = INVALID_GLOBAL_INDEX;
        for (const auto cell : component_cells) {
            const auto physical_gid = global_physical_cell_gid_of[cell];
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
            const auto& klass = global_cell_class_of[cell];
            if (klass.full_active) {
                aggregationDigestMix(
                    feature.canonical_full_active_cell_gid_digest,
                    static_cast<std::uint64_t>(physical_gid));
            }
            if (klass.cut) {
                aggregationDigestMix(
                    feature.canonical_cut_cell_gid_digest,
                    static_cast<std::uint64_t>(physical_gid));
            }
            feature.canonical_full_active_cell_count +=
                klass.full_active ? 1u : 0u;
            feature.canonical_cut_cell_count += klass.cut ? 1u : 0u;
            physical_volume += static_cast<long double>(
                global_active_cell_physical_volume_of[cell]);
        }
        if (feature.canonical_cell_count !=
            feature.canonical_full_active_cell_count +
                feature.canonical_cut_cell_count) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: active feature class "
                "accounting is inconsistent");
        }
        aggregationDigestMix(
            feature.canonical_full_active_cell_gid_digest,
            static_cast<std::uint64_t>(
                feature.canonical_full_active_cell_count));
        aggregationDigestMix(
            feature.canonical_cut_cell_gid_digest,
            static_cast<std::uint64_t>(
                feature.canonical_cut_cell_count));
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

    std::size_t active_cell_count = 0u;
    for (const auto feature_id : active_feature_of) {
        active_cell_count += feature_id != INVALID_GLOBAL_INDEX ? 1u : 0u;
    }
    canonical_active_cells.reserve(active_cell_count);
    for (std::size_t cell = 0u; cell < n_classified; ++cell) {
        const auto feature_id = active_feature_of[cell];
        if (feature_id == INVALID_GLOBAL_INDEX) {
            continue;
        }
        const auto& klass = global_cell_class_of[cell];
        if (klass.full_active == klass.cut) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: canonical active cell does "
                "not have exactly one active class");
        }
        SmallCutAggregationProlongationCell record;
        record.cell_gid = global_physical_cell_gid_of[cell];
        record.owner_rank = global_cell_owner_rank_of[cell];
        record.retained_measure_provider_rank =
            global_active_cell_measure_provider_rank_of[cell];
        record.kind =
            klass.full_active
                ? SmallCutAggregationActiveCellKind::FullActive
                : SmallCutAggregationActiveCellKind::Cut;
        record.active_feature_id = feature_id;
        record.retained_physical_volume =
            global_active_cell_physical_volume_of[cell];
        record.retained_rule_stable_ids =
            global_active_cell_rule_ids_of[cell];
        record.field_dofs = classified_cells.key(cell);
        for (const auto neighbor : global_neighbors[cell]) {
            if (!is_active_cell(neighbor)) {
                continue;
            }
            record.active_face_neighbor_cell_gids.push_back(
                global_physical_cell_gid_of[neighbor]);
        }
        std::sort(record.active_face_neighbor_cell_gids.begin(),
                  record.active_face_neighbor_cell_gids.end());
        record.active_face_neighbor_cell_gids.erase(
            std::unique(
                record.active_face_neighbor_cell_gids.begin(),
                record.active_face_neighbor_cell_gids.end()),
            record.active_face_neighbor_cell_gids.end());
        canonical_active_cells.push_back(std::move(record));
    }
    std::sort(
        canonical_active_cells.begin(),
        canonical_active_cells.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.cell_gid < rhs.cell_gid;
        });
}

struct CandidateRootTraversal {
    std::map<GlobalIndex, std::vector<GlobalRootCandidate>>
        roots_by_candidate{};
    std::size_t maximum_observed_root_path{0u};
    std::size_t root_path_guard_rejections{0u};
    std::size_t root_path_search_cell_visits{0u};
    std::size_t root_path_seed_index_entries{0u};
};

// Bounded breadth-first root search of every candidate over the graph: the
// full-active cells reachable through cut cells within the root-path guard,
// sorted by (distance, cell index).  Sets each candidate's globally_rooted
// and feature_cells.  Throws when a candidate has no cut seed, or when its
// feature holds full cells only beyond the guard and the rootless fallback
// is off.
void traverseCandidateRoots(const AggregationCellGraph& graph,
                            const std::vector<GlobalIndex>& active_feature_of,
                            GlobalCandidateMap& global_candidates,
                            const SmallCutAggregationGuardOptions& guards,
                            bool slave_all_cut,
                            const GuardRootlessLogger& log_guard_rootless,
                            CandidateRootTraversal& traversal)
{
    const auto& classified_cells = graph.cells;
    const auto& global_cell_class_of = graph.klass;
    const auto& global_neighbors = graph.neighbors;
    const auto& guards_ = guards;
    auto& global_roots_by_candidate = traversal.roots_by_candidate;
    auto& maximum_observed_root_path = traversal.maximum_observed_root_path;
    auto& root_path_guard_rejections = traversal.root_path_guard_rejections;
    auto& root_path_search_cell_visits =
        traversal.root_path_search_cell_visits;
    auto& root_path_seed_index_entries =
        traversal.root_path_seed_index_entries;
    std::set<GlobalIndex> all_candidate_component_dofs;
    for (const auto& [dof, support] : global_candidates) {
        static_cast<void>(dof);
        all_candidate_component_dofs.insert(support.component_dofs.begin(),
                                            support.component_dofs.end());
    }

    // Index cut-cell seeds once. The former traversal searched every
    // classified cell for every candidate and then walked the complete
    // connected cut band even though roots beyond the configured path guard
    // can never contribute an admissible row. On a closed three-dimensional
    // interface that work grows with candidate_count * cut_band_size and can
    // dominate every geometry refresh. The index and bounded traversal below
    // retain exactly the roots that can pass the path guard.
    // Seeds are classified-cell indices, ascending like the keys they name.
    const auto n_classified = classified_cells.size();
    std::map<GlobalIndex, std::vector<std::size_t>>
        cut_seed_cells_by_dof;
    for (std::size_t cell = 0u; cell < n_classified; ++cell) {
        if (!global_cell_class_of[cell].cut) {
            continue;
        }
        for (const auto dof : classified_cells.key(cell)) {
            cut_seed_cells_by_dof[dof].push_back(cell);
            ++root_path_seed_index_entries;
        }
    }
    for (auto& [dof, seeds] : cut_seed_cells_by_dof) {
        static_cast<void>(dof);
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
    }

    // A feature is root-eligible when it owns at least one full-active cell
    // that satisfies the selected aggregation policy. For the strong
    // all-cut experiment, a full cell containing any candidate DOF is not a
    // valid root for any candidate because it would introduce slave/master
    // cycles. This policy is cell-global, so it can be classified once.
    std::set<GlobalIndex> root_eligible_feature_ids;
    for (std::size_t cell = 0u; cell < n_classified; ++cell) {
        if (!global_cell_class_of[cell].full_active) {
            continue;
        }
        bool eligible = true;
        if (slave_all_cut) {
            const auto& key = classified_cells.key(cell);
            eligible = std::none_of(
                key.begin(), key.end(), [&](GlobalIndex root_dof) {
                    return all_candidate_component_dofs.count(root_dof) > 0u;
                });
        }
        if (eligible) {
            if (active_feature_of[cell] == INVALID_GLOBAL_INDEX) {
                throw std::logic_error(
                    "SmallCutAggregationConstraint: full-active root cell "
                    "is missing its canonical active feature");
            }
            root_eligible_feature_ids.insert(active_feature_of[cell]);
        }
    }

    // Canonical active feature sizes, for the guard-rootless diagnostics.
    std::map<GlobalIndex, std::size_t> feature_cell_counts;
    for (const auto feature_id : active_feature_of) {
        if (feature_id != INVALID_GLOBAL_INDEX) {
            ++feature_cell_counts[feature_id];
        }
    }

    // Breadth-first search per candidate; a per-candidate stamp marks the
    // visited cells without clearing a set for every candidate.
    std::vector<std::size_t> visit_stamp(n_classified, 0u);
    std::size_t current_stamp = 0u;
    std::vector<std::pair<std::size_t, std::size_t>> queue;
    for (auto& [dof, support] : global_candidates) {
        ++current_stamp;
        queue.clear();
        const auto seed_it = cut_seed_cells_by_dof.find(dof);
        if (seed_it != cut_seed_cells_by_dof.end()) {
            for (const auto seed : seed_it->second) {
                queue.emplace_back(seed, 0u);
                visit_stamp[seed] = current_stamp;
            }
        }
        if (queue.empty()) {
            throw InsufficientGhostHaloError(
                "SmallCutAggregationConstraint: diagnostic="
                "incomplete_distributed_aggregation_halo reason="
                "global_candidate_has_no_cut_seed dof=" +
                std::to_string(dof));
        }

        bool feature_has_eligible_root = false;
        for (const auto seed : seed_it->second) {
            if (active_feature_of[seed] == INVALID_GLOBAL_INDEX) {
                throw std::logic_error(
                    "SmallCutAggregationConstraint: candidate cut seed is "
                    "missing its canonical active feature");
            }
            feature_has_eligible_root =
                feature_has_eligible_root ||
                root_eligible_feature_ids.count(active_feature_of[seed]) > 0u;
            support.feature_cells = std::max(
                support.feature_cells,
                feature_cell_counts[active_feature_of[seed]]);
        }

        auto& roots = global_roots_by_candidate[dof];
        for (std::size_t head = 0u; head < queue.size(); ++head) {
            const auto [cell, distance] = queue[head];
            ++root_path_search_cell_visits;
            const auto& klass = global_cell_class_of[cell];
            bool acceptable_root = klass.full_active;
            if (acceptable_root && slave_all_cut) {
                const auto& key = classified_cells.key(cell);
                acceptable_root = std::none_of(
                    key.begin(), key.end(), [&](GlobalIndex root_dof) {
                        return all_candidate_component_dofs.count(root_dof) >
                               0u;
                    });
            }
            if (acceptable_root) {
                roots.push_back(GlobalRootCandidate{cell, distance});
                continue;
            }
            if (!klass.cut && !(slave_all_cut && klass.full_active)) {
                continue;
            }
            if (distance >= guards_.maximum_root_path_length) {
                continue;
            }
            for (const auto next : global_neighbors[cell]) {
                if (visit_stamp[next] != current_stamp) {
                    visit_stamp[next] = current_stamp;
                    queue.emplace_back(next, distance + 1u);
                }
            }
        }
        std::sort(roots.begin(), roots.end(),
                  [](const GlobalRootCandidate& a,
                     const GlobalRootCandidate& b) {
                      return std::tie(a.distance, a.cell) <
                             std::tie(b.distance, b.cell);
                  });
        roots.erase(std::unique(
                        roots.begin(), roots.end(),
                        [](const GlobalRootCandidate& a,
                           const GlobalRootCandidate& b) {
                            return a.cell == b.cell;
                        }),
                    roots.end());
        for (const auto& root : roots) {
            maximum_observed_root_path =
                std::max(maximum_observed_root_path, root.distance);
        }
        if (feature_has_eligible_root && roots.empty()) {
            // The feature holds full cells, but none within the root-path
            // guard: no admissible extension.  Fail closed unless the
            // opt-in fallback gives the candidate the rootless-island
            // policy (D33).
            ++root_path_guard_rejections;
            maximum_observed_root_path = std::max(
                maximum_observed_root_path,
                guards_.maximum_root_path_length ==
                        std::numeric_limits<std::size_t>::max()
                    ? guards_.maximum_root_path_length
                    : guards_.maximum_root_path_length + 1u);
            if (!guards_.rootless_fallback) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "root_path_guard_rejection candidate_dof=" +
                    std::to_string(dof) + " maximum_observed_path=" +
                    std::to_string(maximum_observed_root_path) +
                    " maximum_allowed_path=" +
                    std::to_string(guards_.maximum_root_path_length) +
                    ": " + kGuardRootlessOptInHint + ".");
            }
            log_guard_rootless(
                "root_path_guard",
                dof,
                support,
                "maximum_allowed_path=" +
                    std::to_string(guards_.maximum_root_path_length));
        }
        support.globally_rooted = !roots.empty();
    }
}

[[nodiscard]] std::shared_ptr<const detail::SmallCutAggregationReuseRecord>
makeAggregationReuseRecord(
    const detail::SmallCutAggregationReuseKey& key,
    const std::string& field_name,
    std::vector<ConstraintLine> added_lines,
    const std::string& diagnostic,
    const SmallCutAggregationRefreshReport& report,
    std::shared_ptr<const detail::SmallCutAggregationPendingProlongation>
        pending,
    const std::vector<GlobalIndex>& canonical_slaves)
{
    static constexpr std::string_view volume_key =
        " canonical_rootless_active_physical_volume=";
    static constexpr std::string_view middle_key =
        " local_relevant_lines_prepared=";
    static constexpr std::string_view pruned_key = " pruned_volume_rules=";
    const auto volume_position = diagnostic.find(volume_key);
    if (volume_position == std::string::npos || !pending) {
        return nullptr;
    }
    const auto head_end = volume_position + volume_key.size();
    const auto middle_position = diagnostic.find(middle_key, head_end);
    if (middle_position == std::string::npos) {
        return nullptr;
    }
    const auto pruned_position = diagnostic.find(pruned_key, middle_position);
    if (pruned_position == std::string::npos) {
        return nullptr;
    }
    auto record = std::make_shared<detail::SmallCutAggregationReuseRecord>();
    record->key = key;
    record->field_name = field_name;
    record->added_lines = std::move(added_lines);
    record->diagnostic_head = diagnostic.substr(0u, head_end);
    record->diagnostic_middle = diagnostic.substr(
        middle_position, pruned_position - middle_position);
    record->report = report;
    record->report.geometry_identity = {};
    record->report.local_lineage = {};
    record->report.canonical_topology_transition.reset();
    record->pending = std::move(pending);
    record->canonical_slaves = canonical_slaves;
    return record;
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
    auto& profile = aggregationRefreshProfile();
    profile.begin();
    // Moving the prior report is allocation-free. This occurs before any
    // communicator collective below, so enabling transition telemetry cannot
    // introduce a rank-asymmetric allocation failure ahead of a collective.
    auto previous_completed_refresh_report =
        std::move(completed_refresh_report_);
    completed_refresh_report_.reset();
    pending_prolongation_.reset();
    auto previous_reuse_record = std::move(reuse_record_);
    reuse_record_.reset();
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
                throw InsufficientGhostHaloError(
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

    // This validation is rank-local and may throw. Coordinate its outcome
    // before any rank can enter the marker collective below.
    std::exception_ptr local_snapshot_exception;
    try {
        cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
    } catch (...) {
        local_snapshot_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(
        system,
        local_snapshot_exception,
        "free_surface_snapshot_validation");

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
    // globalEntityIdsAvailable() scans every cell and face ID; the mesh is
    // fixed for the duration of this call, so query it once instead of per
    // classified cell and per root proposal.
    const bool mesh_global_entity_ids_available =
        mesh.globalEntityIdsAvailable();
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

    // Reuse of an unchanged refresh. Each rank keys every input of the steps
    // below (computeAggregationReuseInputs); the decision is collective, so
    // either every rank publishes its recorded refresh with recomputed
    // retained volumes and rule identities, or every rank recomputes. A key
    // failure only disables reuse.
    profile.mark(AggregationProfilePhase::Preflight);
    std::optional<detail::SmallCutAggregationReuseInputs> reuse_inputs;
    try {
        reuse_inputs = computeAggregationReuseInputs(
            system,
            *cut_context,
            constraints,
            field_,
            active_side_,
            interface_marker_,
            excluded_boundary_markers_,
            excluded_vertices_,
            guards_,
            runtime_options);
    } catch (...) {
        reuse_inputs.reset();
    }
    {
        const bool local_record_matches =
            reuse_inputs.has_value() && previous_reuse_record != nullptr &&
            previous_reuse_record->key == reuse_inputs->key;
        std::optional<detail::SmallCutAggregationRetainedMeasures>
            shared_measures;
        if (local_record_matches) {
            shared_measures = lookupSharedRetainedMeasures(
                system,
                *reuse_inputs,
                interface_marker_,
                active_side_,
                previous_reuse_record->pending->active_cells);
        }
        const auto [reuse_record, reuse_shared_measures] =
            agreeOnAggregationReuse(
                system, local_record_matches, shared_measures.has_value());
        profile.mark(AggregationProfilePhase::ReuseKey);
        if (reuse_record) {
            detail::SmallCutAggregationRetainedMeasures measures;
            if (reuse_shared_measures) {
                measures = std::move(*shared_measures);
            } else {
                measures = refreshRetainedActiveCellMeasures(
                    system,
                    *cut_context,
                    interface_marker_,
                    active_side_,
                    previous_reuse_record->pending->active_cells);
                storeSharedRetainedMeasures(
                    system,
                    *reuse_inputs,
                    interface_marker_,
                    active_side_,
                    previous_reuse_record->pending->active_cells,
                    measures);
            }
            publishReusedRefresh(system,
                                 constraints,
                                 *cut_context,
                                 std::move(previous_reuse_record),
                                 std::move(previous_completed_refresh_report),
                                 measures,
                                 reuse_shared_measures);
            profile.mark(AggregationProfilePhase::ReusePublication);
            profile.report(system, rec.name, "reused");
            return;
        }
        // Detailed diagnostic, printed (at INFO) only when FE_LOG_LEVEL=DEBUG.
        if (Logger::instance().get_level() <= LogLevel::DEBUG) {
            std::ostringstream decision;
            decision << "SmallCutAggregationConstraint: diagnostic="
                        "small_cut_aggregation_reuse field='"
                     << rec.name << "' decision=rebuilt reason="
                     << (!reuse_inputs.has_value()
                             ? "not_admissible"
                             : previous_reuse_record == nullptr
                                   ? "no_previous_refresh"
                                   : !local_record_matches
                                         ? (previous_reuse_record->key
                                                        .topology !=
                                                    reuse_inputs->key.topology
                                                ? "cut_topology_changed"
                                                : "inputs_changed")
                                         : "another_rank_changed");
            FE_LOG_INFO(decision.str());
        }
    }

    // The owner-sourced refresh (see ownerSourcedMix above) replaces the
    // communicator-global exchanges on distributed communicators; debug and
    // experiment options keep the communicator-global refresh, which also
    // remains the fallback whenever the owner-sourced checks fail.
    bool owner_sourced_eligible = false;
#if FE_HAS_MPI
    if (mpi_initialized != 0 && distributed_aggregation &&
        detail::smallCutAggregationOwnerSourcedRefreshEnabled() &&
        !runtime_options.slave_all_cut && !runtime_options.allow_unaggregated &&
        runtime_options.max_lines == std::numeric_limits<std::size_t>::max() &&
        !aggregationProposalTraceEnabled() && mesh_global_entity_ids_available) {
        owner_sourced_eligible = true;
    }
#endif

    // 1. Classify cells from this marker's retained volume rules.
    std::unordered_map<GlobalIndex, CellClass> cell_class;
    struct LocalActiveCellMeasure {
        std::size_t rule_count{0u};
        Real physical_volume{0.0};
        std::set<std::uint64_t> stable_rule_ids{};
        std::size_t missing_stable_rule_ids{0u};
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
    LocalCellKeyTable local_cell_by_key;
    struct OrderedCellClass {
        const CellKey* key{nullptr};
        GlobalIndex cell{-1};
        CellClass klass{};
    };
    std::vector<OrderedCellClass> ordered_cell_classes;
    std::vector<std::int64_t> local_cell_class_words;
    // The owner-sourced refresh does not all-gather the classified cells;
    // it checks them (the same checks) and serializes them only if it
    // returns to the communicator-global refresh.
    bool serialize_cell_classes = !owner_sourced_eligible;
    const auto push_cell_class_word = [&](std::int64_t word) {
        if (serialize_cell_classes) {
            local_cell_class_words.push_back(word);
        }
    };
    const auto declare_ordered_cell_classes = [&]() {
        for (const auto& ordered : ordered_cell_classes) {
            const auto& key = *ordered.key;
            const auto& klass = ordered.klass;
            const std::int64_t flags =
                (klass.full_active ? 1 : 0) | (klass.cut ? 2 : 0);
            push_cell_class_word(flags);
            push_cell_class_word(
                static_cast<std::int64_t>(key.size()));
            for (const auto dof : key) {
                push_cell_class_word(static_cast<std::int64_t>(dof));
            }
            const auto cell = ordered.cell;
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
                mesh_global_entity_ids_available
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
            push_cell_class_word(
                static_cast<std::int64_t>(physical_cell_gid));
            push_cell_class_word(
                static_cast<std::int64_t>(rule_count));
            push_cell_class_word(
                std::bit_cast<std::int64_t>(
                    static_cast<double>(physical_volume)));
            if (measure != local_active_cell_measures.end()) {
                if (measure->second.stable_rule_ids.size() +
                        measure->second.missing_stable_rule_ids !=
                    rule_count) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: retained rule identity "
                        "count does not match the cell measure count");
                }
                for (std::size_t missing = 0u;
                     missing < measure->second.missing_stable_rule_ids;
                     ++missing) {
                    push_cell_class_word(0);
                }
                for (const auto stable_id :
                     measure->second.stable_rule_ids) {
                    push_cell_class_word(
                        std::bit_cast<std::int64_t>(stable_id));
                }
            }
        }
    };
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
                const bool partial = retainedPartialVolumeRule(
                    rule.full_cell_equivalent, fraction);
                if (side == active_side_ && !rule.full_cell_equivalent &&
                    partial) {
                    entry.cut = true;
                }
                if (side == active_side_ &&
                    (rule.full_cell_equivalent || partial)) {
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
                        rule.provenance.cut_topology_revision;
                    if (stable_rule_id == 0u) {
                        ++measure.missing_stable_rule_ids;
                    } else if (!measure.stable_rule_ids.insert(
                                   stable_rule_id).second) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: diagnostic="
                            "duplicate_active_feature_volume_rule retained "
                            "rules for one cell repeat a nonzero stable "
                            "identity");
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

    local_cell_by_key.reserve(static_cast<std::size_t>(
        std::max<GlobalIndex>(GlobalIndex{0}, mesh.numCells())));
    mesh.forEachCell([&](GlobalIndex cell) {
        auto key = cell_key(cell);
        if (key.empty()) {
            return;
        }
        local_cell_by_key.append(std::move(key), cell);
    });
    if (!local_cell_by_key.finalize()) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: duplicate local cells have "
            "the same global field-DOF support");
    }

    // Classified cells in ascending key order, read from the sorted local
    // table instead of recomputing and sorting their keys. Every classified
    // cell with field DOFs must be found there.
    std::size_t classified_cells_with_support = 0u;
    for (const auto& [cell, klass] : cell_class) {
        static_cast<void>(klass);
        if (!dh.getCellDofs(cell).empty()) {
            ++classified_cells_with_support;
        }
    }
    ordered_cell_classes.reserve(classified_cells_with_support);
    for (const auto& [key, cell] : local_cell_by_key) {
        const auto klass = cell_class.find(cell);
        if (klass != cell_class.end()) {
            ordered_cell_classes.push_back(
                OrderedCellClass{&key, cell, klass->second});
        }
    }
    if (ordered_cell_classes.size() != classified_cells_with_support) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: classified cell has no "
            "local cell-key resolution");
    }

    declare_ordered_cell_classes();
    } catch (...) {
        local_cell_declaration_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_cell_declaration_exception,
                                "cell_class_and_key_serialization");
    profile.setCount(AggregationProfileCount::LocalCells,
                     static_cast<std::uint64_t>(mesh.numCells()));
    profile.setCount(AggregationProfileCount::OwnedCells,
                     static_cast<std::uint64_t>(mesh.numOwnedCells()));
    profile.setCount(AggregationProfileCount::LocalClassifiedCells,
                     ordered_cell_classes.size());
    profile.mark(AggregationProfilePhase::Classification);

    // Outputs of the refresh consumed by the publication below; filled by
    // the owner-sourced refresh or by the communicator-global refresh.
    const bool slave_all_cut = runtime_options.slave_all_cut;
    const bool allow_unaggregated = runtime_options.allow_unaggregated;
    const std::size_t max_lines = runtime_options.max_lines;
    std::vector<SmallCutAggregationActiveFeatureReport>
        canonical_active_features;
    std::vector<SmallCutAggregationProlongationCell>
        canonical_active_cells;
    std::size_t canonical_rooted_active_feature_count = 0u;
    std::size_t canonical_rootless_active_feature_count = 0u;
    Real canonical_rootless_active_physical_volume = 0.0;
    std::size_t maximum_observed_root_path = 0u;
    std::size_t root_path_guard_rejections = 0u;
    std::size_t root_path_search_cell_visits = 0u;
    std::size_t root_path_seed_index_entries = 0u;
    std::size_t constrained_vertex_count = 0u;
    std::size_t aggregated = 0u;
    std::size_t no_root = 0u;
    std::size_t island_pinned_dofs = 0u;
    std::size_t inversion_failed = 0u;
    // Inversions accepted at the round-off floor (see invertMapping).
    std::size_t inversion_round_off_acceptances = 0u;
    std::size_t non_field_nodes = 0u;
    std::size_t empty_lines = 0u;
    std::size_t extrapolation_guard_rejections = 0u;
    std::size_t line_guard_rejections = 0u;
    Real maximum_attempted_reference_extrapolation = 0.0;
    Real maximum_observed_reference_extrapolation = 0.0;
    Real maximum_attempted_absolute_coefficient = 0.0;
    Real maximum_observed_absolute_coefficient = 0.0;
    Real maximum_attempted_row_l1_norm = 0.0;
    Real maximum_observed_row_l1_norm = 0.0;
    std::size_t communicator_root_proposal_failures = 0u;
    std::size_t communicator_extrapolation_guard_rejections = 0u;
    std::size_t communicator_line_guard_rejections = 0u;
    std::vector<GlobalIndex> current_slaves;
    LocalAggregationDeclarationMap local_aggregation_declarations;
    DistributedAggregationResult distributed_result;

    // 3. Vertices of cut cells + their incident band cells.
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> vertex_cells;
    std::vector<GlobalIndex> cell_nodes;
    std::unordered_set<GlobalIndex> excluded_vertices;
    std::vector<LocalCandidateSupport> local_candidate_support;
    // Local candidate facts of every vertex of a locally stored active cell.
    // Collective (coordinated local failure).
    const auto build_local_candidate_support = [&]() {
        vertex_cells.clear();
        excluded_vertices.clear();
        local_candidate_support.clear();
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
                            const auto& ref = elements::ReferenceElement::shared(cell_type);
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
    };

    // The root provider needs the field's complete nodal support. Reject a
    // sub-parametric cell before proposal generation instead of emitting a
    // partial polynomial extension.  Throws.
    const auto check_subparametric_active_cells = [&]() {
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
    };

    std::vector<Real> values;
    std::vector<GlobalIndex> root_check_nodes;
    // Outcome of every local root proposal, reported only if the resolver
    // finds a globally rooted candidate without a valid proposal.
    struct RootProposalOutcome {
        GlobalIndex candidate_dof{INVALID_GLOBAL_INDEX};
        GlobalIndex root_cell_gid{INVALID_GLOBAL_INDEX};
        std::size_t distance{0u};
        Real extrapolation{std::numeric_limits<Real>::quiet_NaN()};
        const char* outcome{"unknown"};
    };
    std::vector<RootProposalOutcome> root_proposal_outcomes;
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

    // One root proposal: the polynomial extension of local root cell `root`
    // to the candidate vertex, or the guard or inversion that rejected it.
    // The counters and maxima of the refresh diagnostic are applied by the
    // caller from the returned flags.
    struct RootProposalAttempt {
        bool emitted{false};
        // Inversion or basis-size failure.
        bool inversion_failed{false};
        bool round_off_acceptance{false};
        bool extrapolation_rejected{false};
        bool empty_line{false};
        bool line_guard_rejected{false};
        Real attempted_reference_extrapolation{0.0};
        Real observed_reference_extrapolation{0.0};
        Real attempted_absolute_coefficient{0.0};
        Real observed_absolute_coefficient{0.0};
        Real attempted_row_l1_norm{0.0};
        Real observed_row_l1_norm{0.0};
        AggregationDeclaration declaration{};
    };
    auto build_root_proposal =
        [&](GlobalIndex declaration_dof,
            const GlobalCandidateSupport& candidate,
            std::size_t root_distance,
            GlobalIndex root) {
        RootProposalAttempt attempt;
        Vector<Real, 3> traced_xi{};
        Real traced_extrapolation =
            std::numeric_limits<Real>::quiet_NaN();
        auto trace_proposal = [&](const char* outcome) {
            root_proposal_outcomes.push_back(RootProposalOutcome{
                .candidate_dof = declaration_dof,
                .root_cell_gid = mesh_global_entity_ids_available
                                     ? mesh.getCellGlobalId(root)
                                     : root,
                .distance = root_distance,
                .extrapolation = traced_extrapolation,
                .outcome = outcome});
            if (!aggregationProposalTraceEnabled()) {
                return;
            }
            std::ostringstream trace;
            trace << std::setprecision(17)
                  << "SmallCutAggregationConstraint: diagnostic="
                     "aggregation_proposal_trace stage=construction"
                  << " field='" << rec.name << "'"
                  << " candidate_dof=" << declaration_dof
                  << " xyz=(" << candidate.coordinates[0] << ","
                  << candidate.coordinates[1] << ","
                  << candidate.coordinates[2] << ")"
                  << " root_cell=" << root
                  << " root_distance=" << root_distance
                  << " xi=(" << traced_xi[0] << "," << traced_xi[1]
                  << "," << traced_xi[2] << ")"
                  << " extrapolation=" << traced_extrapolation
                  << " outcome=" << outcome;
            FE_LOG_INFO(trace.str());
        };
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
        bool inversion_stalled_at_round_off = false;
        if (mapping == nullptr ||
            !invertMapping(*mapping, candidate.coordinates,
                           mesh.dimension(), xi,
                           &inversion_stalled_at_round_off)) {
            attempt.inversion_failed = true;
            traced_xi = xi;
            trace_proposal("inversion_failed");
            return attempt;
        }
        if (inversion_stalled_at_round_off) {
            attempt.round_off_acceptance = true;
        }
        const Real reference_extrapolation =
            normalizedReferenceExtrapolationDistance(
                mesh.getCellType(root), xi);
        traced_xi = xi;
        traced_extrapolation = reference_extrapolation;
        attempt.attempted_reference_extrapolation = std::max(
            attempt.attempted_reference_extrapolation,
            reference_extrapolation);
        if (!std::isfinite(reference_extrapolation) ||
            reference_extrapolation >
                guards_.maximum_reference_extrapolation_distance +
                    Real{128.0} * std::numeric_limits<Real>::epsilon() *
                        std::max(
                            Real{1.0},
                            guards_
                                .maximum_reference_extrapolation_distance)) {
            attempt.extrapolation_rejected = true;
            trace_proposal("extrapolation_guard");
            return attempt;
        }
        attempt.observed_reference_extrapolation = std::max(
            attempt.observed_reference_extrapolation,
            reference_extrapolation);
        extension_basis.evaluate_values(xi, values);
        mesh.getCellNodes(root, cell_nodes);
        const auto n_root_field_nodes = std::min(
            {cell_nodes.size(), n_field_basis, n_extension});
        if (values.size() < n_root_field_nodes ||
            (!field_mesh_to_basis.empty() &&
             field_mesh_to_basis.size() < n_root_field_nodes)) {
            attempt.inversion_failed = true;
            trace_proposal("basis_size");
            return attempt;
        }

        auto& declaration = attempt.declaration;
        declaration = AggregationDeclaration{
            .state = AggregationDeclarationState::Rooted,
            .root_distance = root_distance,
            .root_cell_gid = mesh_global_entity_ids_available
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
            attempt.attempted_absolute_coefficient = std::max(
                attempt.attempted_absolute_coefficient,
                line_guard.maximum_absolute_coefficient);
            attempt.attempted_row_l1_norm = std::max(
                attempt.attempted_row_l1_norm,
                line_guard.row_l1_norm);
            if (!line_guard.valid) {
                attempt.line_guard_rejected = true;
                proposal_valid = false;
                break;
            }
            attempt.observed_absolute_coefficient = std::max(
                attempt.observed_absolute_coefficient,
                line_guard.maximum_absolute_coefficient);
            attempt.observed_row_l1_norm = std::max(
                attempt.observed_row_l1_norm,
                line_guard.row_l1_norm);
            declaration.lines.push_back(std::move(line));
        }
        if (!proposal_valid ||
            declaration.lines.size() != candidate.component_dofs.size()) {
            attempt.empty_line = true;
            trace_proposal(proposal_valid ? "line_count" : "line_invalid");
            return attempt;
        }
        trace_proposal("emitted");
        attempt.emitted = true;
        return attempt;
    };
    auto apply_attempt_counters = [&](const RootProposalAttempt& attempt) {
        inversion_failed += attempt.inversion_failed ? 1u : 0u;
        inversion_round_off_acceptances +=
            attempt.round_off_acceptance ? 1u : 0u;
        extrapolation_guard_rejections +=
            attempt.extrapolation_rejected ? 1u : 0u;
        empty_lines += attempt.empty_line ? 1u : 0u;
        line_guard_rejections += attempt.line_guard_rejected ? 1u : 0u;
        maximum_attempted_reference_extrapolation = std::max(
            maximum_attempted_reference_extrapolation,
            attempt.attempted_reference_extrapolation);
        maximum_observed_reference_extrapolation = std::max(
            maximum_observed_reference_extrapolation,
            attempt.observed_reference_extrapolation);
        maximum_attempted_absolute_coefficient = std::max(
            maximum_attempted_absolute_coefficient,
            attempt.attempted_absolute_coefficient);
        maximum_observed_absolute_coefficient = std::max(
            maximum_observed_absolute_coefficient,
            attempt.observed_absolute_coefficient);
        maximum_attempted_row_l1_norm = std::max(
            maximum_attempted_row_l1_norm, attempt.attempted_row_l1_norm);
        maximum_observed_row_l1_norm = std::max(
            maximum_observed_row_l1_norm, attempt.observed_row_l1_norm);
    };

    // Candidate DOFs this rank condenses: DOFs of its owned cells (assembled
    // here under reverse scatter) and of their face neighbors (reached by
    // facet terms). Such a rank needs the canonical line and all masters, or
    // the discretization would change. Under reverse scatter, candidate DOFs
    // of other local cells holding an owned DOF only couple to owned rows;
    // missing masters there cost Jacobian entries, not correctness.
    const auto compute_assembled_candidate_dofs =
        [&](const GlobalCandidateMap& global_candidates,
            std::set<GlobalIndex>& assembled_candidate_dofs,
            std::set<GlobalIndex>& row_coupled_candidate_dofs) {
            // Candidate component DOFs, sorted and distinct, with dense
            // per-DOF marks over their range: one array read per cell DOF
            // instead of a set probe, and one ordered insertion per marked
            // candidate instead of one per incident cell.
            std::vector<GlobalIndex> candidate_component_dofs;
            for (const auto& [dof, candidate] : global_candidates) {
                static_cast<void>(dof);
                candidate_component_dofs.insert(
                    candidate_component_dofs.end(),
                    candidate.component_dofs.begin(),
                    candidate.component_dofs.end());
            }
            std::sort(candidate_component_dofs.begin(),
                      candidate_component_dofs.end());
            candidate_component_dofs.erase(
                std::unique(candidate_component_dofs.begin(),
                            candidate_component_dofs.end()),
                candidate_component_dofs.end());
            const GlobalIndex candidate_dof_base =
                candidate_component_dofs.empty()
                    ? GlobalIndex{0}
                    : candidate_component_dofs.front();
            const std::size_t candidate_dof_span =
                candidate_component_dofs.empty()
                    ? std::size_t{0}
                    : static_cast<std::size_t>(
                          candidate_component_dofs.back() -
                          candidate_dof_base) + 1u;
            constexpr unsigned char candidate_bit = 1u;
            constexpr unsigned char assembled_bit = 2u;
            constexpr unsigned char row_coupled_bit = 4u;
            std::vector<unsigned char> candidate_marks(candidate_dof_span, 0u);
            for (const auto dof : candidate_component_dofs) {
                candidate_marks[static_cast<std::size_t>(
                    dof - candidate_dof_base)] = candidate_bit;
            }
            auto candidate_mark = [&](GlobalIndex dof) -> unsigned char* {
                if (dof < candidate_dof_base ||
                    static_cast<std::size_t>(dof - candidate_dof_base) >=
                        candidate_dof_span) {
                    return nullptr;
                }
                auto* mark = &candidate_marks[static_cast<std::size_t>(
                    dof - candidate_dof_base)];
                return (*mark & candidate_bit) != 0u ? mark : nullptr;
            };
            const auto& assembly_partition =
                system.dofHandler().getPartition();
            const auto n_local_cells = mesh.numCells();
            std::vector<unsigned char> assembled_cell(
                static_cast<std::size_t>(std::max<GlobalIndex>(0, n_local_cells)),
                static_cast<unsigned char>(0));
            auto mark_cell = [&](GlobalIndex cell) {
                if (cell >= 0 && cell < n_local_cells) {
                    assembled_cell[static_cast<std::size_t>(cell)] = 1u;
                }
            };
            for (GlobalIndex cell = 0; cell < n_local_cells; ++cell) {
                if (mesh.isOwnedCell(cell)) {
                    mark_cell(cell);
                }
            }
            mesh.forEachInteriorFace(
                [&](GlobalIndex, GlobalIndex first, GlobalIndex second) {
                    if (first >= 0 && first < n_local_cells &&
                        mesh.isOwnedCell(first)) {
                        mark_cell(second);
                    }
                    if (second >= 0 && second < n_local_cells &&
                        mesh.isOwnedCell(second)) {
                        mark_cell(first);
                    }
                });
            // Owned-row assembly (any policy but reverse scatter) condenses
            // every local cell that holds an owned row as well.
            const bool owned_cells_only =
                system.assemblyGhostPolicy() ==
                assembly::GhostPolicy::ReverseScatter;
            for (GlobalIndex cell = 0; cell < n_local_cells; ++cell) {
                const auto cell_dofs = dh.getCellDofs(cell);
                const bool row_coupled =
                    std::any_of(cell_dofs.begin(), cell_dofs.end(),
                                [&](GlobalIndex dof) {
                                    return assembly_partition.isOwned(offset + dof);
                                });
                const bool assembled =
                    assembled_cell[static_cast<std::size_t>(cell)] != 0u ||
                    (!owned_cells_only && row_coupled);
                if (!assembled && !row_coupled) {
                    continue;
                }
                for (const auto dof : cell_dofs) {
                    auto* mark = candidate_mark(offset + dof);
                    if (mark == nullptr) {
                        continue;
                    }
                    if (assembled) {
                        *mark |= assembled_bit;
                    }
                    if (row_coupled) {
                        *mark |= row_coupled_bit;
                    }
                }
            }
            for (const auto dof : candidate_component_dofs) {
                const auto mark = candidate_marks[static_cast<std::size_t>(
                    dof - candidate_dof_base)];
                if ((mark & assembled_bit) != 0u) {
                    assembled_candidate_dofs.insert(
                        assembled_candidate_dofs.end(), dof);
                }
                if ((mark & row_coupled_bit) != 0u) {
                    row_coupled_candidate_dofs.insert(
                        row_coupled_candidate_dofs.end(), dof);
                }
            }
    };

#if FE_HAS_MPI
    // Owner-sourced refresh (see the description above resolveCellCensus).
    // It fills the outputs above exactly as the communicator-global refresh
    // below does and returns true, or returns false, with no output changed,
    // when any rank finds an inconsistency, a failure or a debug condition;
    // the communicator-global refresh then runs and reports what it always
    // reported.  Collective: every rank takes the same branch at every
    // all_ok() checkpoint, and no collective call is skipped by a local
    // exception.
    const auto run_owner_sourced_refresh = [&]() -> bool {
        const auto comm = system.dofHandler().mpiComm();
        int my_rank = 0;
        int world_size = 1;
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &world_size);
        const auto n_ranks = static_cast<std::size_t>(world_size);
        const auto all_ok = [&](bool ok) {
            const int local = ok ? 1 : 0;
            int global = 0;
            MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
            return global != 0;
        };
        const auto& partition = system.dofHandler().getPartition();
        bool ok = true;

        // Every stored cell with field DOFs, by physical cell ID.  Every
        // owned one must carry this rank's own classification (its rules).
        std::vector<std::pair<GlobalIndex, GlobalIndex>> local_cells_by_gid;
        try {
            local_cells_by_gid.reserve(local_cell_by_key.size());
            for (const auto& [key, cell] : local_cell_by_key) {
                static_cast<void>(key);
                const auto gid = mesh.getCellGlobalId(cell);
                if (gid < 0 ||
                    (mesh.isOwnedCell(cell) && cell_class.count(cell) == 0u)) {
                    ok = false;
                    break;
                }
                local_cells_by_gid.emplace_back(gid, cell);
            }
            std::sort(local_cells_by_gid.begin(), local_cells_by_gid.end());
            ok = ok &&
                 std::adjacent_find(local_cells_by_gid.begin(),
                                    local_cells_by_gid.end(),
                                    [](const auto& lhs, const auto& rhs) {
                                        return lhs.first == rhs.first;
                                    }) == local_cells_by_gid.end();
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        const auto local_cell_of = [&](GlobalIndex gid) -> GlobalIndex {
            const auto it = std::lower_bound(
                local_cells_by_gid.begin(), local_cells_by_gid.end(), gid,
                [](const auto& entry, GlobalIndex key) {
                    return entry.first < key;
                });
            return it != local_cells_by_gid.end() && it->first == gid
                       ? it->second
                       : INVALID_GLOBAL_INDEX;
        };
        const auto class_of = [&](GlobalIndex cell) -> const CellClass* {
            const auto it = cell_class.find(cell);
            return it == cell_class.end() ? nullptr : &it->second;
        };

        // ---- Cell census: ownership, classification, provider, holders ----
        std::vector<std::vector<std::int64_t>> outgoing(n_ranks);
        try {
            for (const auto& [gid, cell] : local_cells_by_gid) {
                std::int64_t flags =
                    mesh.isOwnedCell(cell) ? kCellCensusOwned : 0;
                if (const auto* klass = class_of(cell)) {
                    flags |= kCellCensusClassified;
                    if (klass->full_active || klass->cut) {
                        flags |= kCellCensusActive;
                    }
                }
                auto& words = outgoing[static_cast<std::size_t>(
                    ownerSourcedHome(static_cast<std::uint64_t>(gid),
                                     world_size))];
                words.push_back(static_cast<std::int64_t>(gid));
                words.push_back(flags);
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words census;
        try {
            census = exchangeInt64Words(
                comm, outgoing, AggregationProfileExchange::CellCensus);
        } catch (...) {
            return false;
        }
        std::vector<std::vector<std::int64_t>> census_replies;
        try {
            ok = resolveCellCensus(census, world_size, census_replies);
        } catch (...) {
            ok = false;
        }
        census = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words replies;
        try {
            replies = exchangeInt64Words(
                comm,
                census_replies,
                AggregationProfileExchange::CellCensusReplies);
        } catch (...) {
            return false;
        }
        std::unordered_map<GlobalIndex, CellCensusReply> owned_active_census;
        try {
            if (replies.words.size() % 4u != 0u) {
                ok = false;
            }
            for (std::size_t i = 0u; ok && i < replies.words.size(); i += 4u) {
                const auto gid = static_cast<GlobalIndex>(replies.words[i]);
                const CellCensusReply reply{
                    .provider_rank = static_cast<int>(replies.words[i + 1u]),
                    .holder_count = static_cast<int>(replies.words[i + 2u]),
                    .min_holder_rank =
                        static_cast<int>(replies.words[i + 3u])};
                const auto cell = local_cell_of(gid);
                const auto* klass = cell == INVALID_GLOBAL_INDEX
                                        ? nullptr
                                        : class_of(cell);
                if (klass == nullptr || !mesh.isOwnedCell(cell) ||
                    !(klass->full_active || klass->cut) ||
                    !owned_active_census.emplace(gid, reply).second) {
                    ok = false;
                }
            }
            std::size_t owned_active = 0u;
            for (const auto& [gid, cell] : local_cells_by_gid) {
                static_cast<void>(gid);
                const auto* klass = class_of(cell);
                if (mesh.isOwnedCell(cell) && klass != nullptr &&
                    (klass->full_active || klass->cut)) {
                    ++owned_active;
                }
            }
            ok = ok && owned_active == owned_active_census.size();
        } catch (...) {
            ok = false;
        }
        replies = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        profile.mark(AggregationProfilePhase::CellClassExchange);

        // ---- Faces: owners match the faces of their cells locally ----
        using FaceKey = std::vector<GlobalIndex>;
        std::vector<GlobalIndex> face_cell_nodes;
        // Sorted face-DOF signatures of a cell, as the communicator-global
        // owner face signatures; throws when they cannot be formed.
        const auto cell_face_keys = [&](GlobalIndex cell,
                                        std::vector<FaceKey>& keys) {
            mesh.getCellNodes(cell, face_cell_nodes);
            if (layout == CellDofLayout::Unknown && components > 1) {
                detect_layout(cell, face_cell_nodes);
            }
            if (layout == CellDofLayout::Unknown && components > 1) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: cannot determine "
                    "product-field cell DOF ordering for face signatures");
            }
            check_cell_pairable(cell);
            const auto& ref =
                elements::ReferenceElement::shared(mesh.getCellType(cell));
            const auto n_cell_faces = static_cast<std::size_t>(ref.num_faces());
            keys.assign(n_cell_faces, FaceKey{});
            for (std::size_t face = 0; face < n_cell_faces; ++face) {
                auto& face_key = keys[face];
                const auto& face_nodes = ref.face_nodes(face);
                face_key.reserve(face_nodes.size() * components);
                for (const auto local_node : face_nodes) {
                    if (local_node < 0 ||
                        static_cast<std::size_t>(local_node) >=
                            face_cell_nodes.size() ||
                        static_cast<std::size_t>(local_node) >=
                            n_field_basis) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: incomplete "
                            "face-DOF signature");
                    }
                    const auto node_dofs = cell_node_dofs(
                        cell, static_cast<std::size_t>(local_node));
                    if (node_dofs.size() != components) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: incomplete nodal "
                            "DOFs in a face signature");
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
                        "SmallCutAggregationConstraint: invalid face-DOF "
                        "signature");
                }
            }
            std::sort(keys.begin(), keys.end());
            if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: cell has duplicate "
                    "face-DOF signatures");
            }
        };
        // Face neighbors (physical IDs) of every owned cell.
        std::unordered_map<GlobalIndex, std::vector<GlobalIndex>>
            owned_neighbor_gids;
        std::vector<std::vector<std::int64_t>> face_outgoing(n_ranks);
        try {
            struct FaceIncidence {
                FaceKey key{};
                GlobalIndex cell{INVALID_GLOBAL_INDEX};
                GlobalIndex gid{INVALID_GLOBAL_INDEX};
                bool owned{false};
            };
            std::vector<FaceIncidence> incidences;
            std::unordered_set<GlobalIndex> owned_nodes;
            std::vector<GlobalIndex> nodes;
            for (const auto& [gid, cell] : local_cells_by_gid) {
                static_cast<void>(gid);
                if (!mesh.isOwnedCell(cell)) {
                    continue;
                }
                mesh.getCellNodes(cell, nodes);
                owned_nodes.insert(nodes.begin(), nodes.end());
            }
            std::vector<FaceKey> keys;
            for (const auto& [gid, cell] : local_cells_by_gid) {
                const bool owned = mesh.isOwnedCell(cell);
                if (!owned) {
                    mesh.getCellNodes(cell, nodes);
                    if (std::none_of(nodes.begin(), nodes.end(),
                                     [&](GlobalIndex node) {
                                         return owned_nodes.count(node) != 0u;
                                     })) {
                        continue;
                    }
                }
                cell_face_keys(cell, keys);
                for (auto& key : keys) {
                    incidences.push_back(FaceIncidence{
                        .key = std::move(key),
                        .cell = cell,
                        .gid = gid,
                        .owned = owned});
                }
            }
            std::sort(incidences.begin(), incidences.end(),
                      [](const FaceIncidence& lhs, const FaceIncidence& rhs) {
                          if (lhs.key != rhs.key) {
                              return lhs.key < rhs.key;
                          }
                          return lhs.cell < rhs.cell;
                      });
            for (const auto& [gid, cell] : local_cells_by_gid) {
                if (mesh.isOwnedCell(cell)) {
                    owned_neighbor_gids.try_emplace(gid);
                }
            }
            for (std::size_t first = 0u; ok && first < incidences.size();) {
                std::size_t last = first + 1u;
                while (last < incidences.size() &&
                       incidences[last].key == incidences[first].key) {
                    ++last;
                }
                const auto count = last - first;
                const bool any_owned = std::any_of(
                    incidences.begin() + static_cast<std::ptrdiff_t>(first),
                    incidences.begin() + static_cast<std::ptrdiff_t>(last),
                    [](const FaceIncidence& incidence) {
                        return incidence.owned;
                    });
                if (any_owned) {
                    if (count > 2u) {
                        ok = false;
                    } else {
                        const auto hash = static_cast<std::int64_t>(
                            ownerSourcedKeyHash(incidences[first].key));
                        auto& words = face_outgoing[static_cast<std::size_t>(
                            ownerSourcedHome(static_cast<std::uint64_t>(hash),
                                             world_size))];
                        if (count == 1u) {
                            words.push_back(hash);
                            words.push_back(kFaceCensusOpen);
                        } else {
                            const auto& a = incidences[first];
                            const auto& b = incidences[first + 1u];
                            if (a.owned) {
                                owned_neighbor_gids[a.gid].push_back(b.gid);
                            }
                            if (b.owned) {
                                owned_neighbor_gids[b.gid].push_back(a.gid);
                            }
                            if (a.owned != b.owned) {
                                words.push_back(hash);
                                words.push_back(kFaceCensusShared);
                            }
                        }
                    }
                }
                first = last;
            }
            for (auto& [gid, neighbors] : owned_neighbor_gids) {
                static_cast<void>(gid);
                std::sort(neighbors.begin(), neighbors.end());
                if (std::adjacent_find(neighbors.begin(), neighbors.end()) !=
                    neighbors.end()) {
                    ok = false;
                }
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words face_census;
        try {
            face_census = exchangeInt64Words(
                comm, face_outgoing, AggregationProfileExchange::FaceCensus);
        } catch (...) {
            return false;
        }
        try {
            ok = resolveFaceCensus(face_census, world_size);
        } catch (...) {
            ok = false;
        }
        face_census = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        profile.mark(AggregationProfilePhase::FaceGraph);

        // ---- Active-cell ledger: one record per active cell, by its owner ----
        std::vector<std::array<Real, 3>> geometry_coordinates;
        const auto geometry_digest = [&](GlobalIndex cell) {
            mesh.getCellCoordinates(cell, geometry_coordinates);
            std::uint64_t digest = 0x534f475f47454f4dull;
            digest = ownerSourcedMix(
                digest, static_cast<std::uint64_t>(mesh.getCellType(cell)));
            digest = ownerSourcedMix(
                digest,
                static_cast<std::uint64_t>(geometry_coordinates.size()));
            for (const auto& point : geometry_coordinates) {
                for (const auto coordinate : point) {
                    digest = ownerSourcedMix(
                        digest, std::bit_cast<std::uint64_t>(
                                    static_cast<double>(coordinate)));
                }
            }
            return digest;
        };
        // Field DOFs over the field nodes in node order (node-major).
        const auto node_ordered_dofs = [&](GlobalIndex cell) {
            check_cell_pairable(cell);
            std::vector<GlobalIndex> dofs;
            dofs.reserve(n_field_basis * components);
            for (std::size_t node = 0u; node < n_field_basis; ++node) {
                const auto node_dofs = cell_node_dofs(cell, node);
                if (node_dofs.size() != components) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: incomplete nodal "
                        "DOFs");
                }
                for (const auto dof : node_dofs) {
                    dofs.push_back(offset + dof);
                }
            }
            return dofs;
        };
        // Measure of a locally classified active cell as the classification
        // declares it: rule count, volume and identities (unavailable first).
        const auto local_measure =
            [&](GlobalIndex cell, std::size_t& rule_count, Real& volume,
                std::vector<std::uint64_t>& rule_ids) {
                const auto& measure = local_active_cell_measures.at(cell);
                rule_count = measure.rule_count;
                volume = measure.physical_volume;
                rule_ids.assign(measure.missing_stable_rule_ids,
                                std::uint64_t{0});
                rule_ids.insert(rule_ids.end(),
                                measure.stable_rule_ids.begin(),
                                measure.stable_rule_ids.end());
            };
        std::vector<std::int64_t> ledger_words;
        try {
            for (const auto& [gid, cell] : local_cells_by_gid) {
                if (!mesh.isOwnedCell(cell)) {
                    continue;
                }
                const auto* klass = class_of(cell);
                if (klass == nullptr || !(klass->full_active || klass->cut)) {
                    continue;
                }
                if (klass->full_active && klass->cut) {
                    ok = false;
                    break;
                }
                OwnerSourcedActiveCell record;
                record.gid = gid;
                record.klass = *klass;
                const auto& census_reply = owned_active_census.at(gid);
                record.provider_rank = census_reply.provider_rank;
                record.holder_count = census_reply.holder_count;
                record.min_holder_rank = census_reply.min_holder_rank;
                local_measure(cell, record.rule_count, record.volume,
                              record.rule_ids);
                if (record.rule_count != record.rule_ids.size()) {
                    ok = false;
                    break;
                }
                record.node_dofs = node_ordered_dofs(cell);
                record.geometry_digest = geometry_digest(cell);
                record.neighbor_gids = owned_neighbor_gids.at(gid);
                appendOwnerSourcedActiveCell(ledger_words, record);
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words ledger_gathered;
        try {
            ledger_gathered = allGatherInt64Words(
                comm,
                std::span<const std::int64_t>(ledger_words),
                AggregationProfileExchange::ActiveCellLedger);
        } catch (...) {
            return false;
        }
        std::vector<OwnerSourcedActiveCell> ledger;
        try {
            ok = decodeOwnerSourcedActiveCells(
                ledger_gathered, world_size, ledger);
        } catch (...) {
            ok = false;
        }
        ledger_gathered = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }

        // Every stored copy of an active cell must be the owner's: the same
        // classification and measure where this rank classified it, and the
        // same field DOFs and node coordinates.  A classified cell absent
        // from the ledger must be inactive here too.
        try {
            std::size_t rule_count = 0u;
            Real volume = 0.0;
            std::vector<std::uint64_t> rule_ids;
            for (const auto& [gid, cell] : local_cells_by_gid) {
                const auto* canonical = findOwnerSourcedActiveCell(ledger, gid);
                const auto* klass = class_of(cell);
                const bool active =
                    klass != nullptr && (klass->full_active || klass->cut);
                if (canonical == nullptr) {
                    if (active) {
                        ok = false;
                        break;
                    }
                    continue;
                }
                if (klass != nullptr) {
                    if (klass->full_active != canonical->klass.full_active ||
                        klass->cut != canonical->klass.cut) {
                        ok = false;
                        break;
                    }
                    local_measure(cell, rule_count, volume, rule_ids);
                    if (rule_count != canonical->rule_count ||
                        volume != canonical->volume ||
                        rule_ids != canonical->rule_ids) {
                        ok = false;
                        break;
                    }
                }
                if (node_ordered_dofs(cell) != canonical->node_dofs ||
                    geometry_digest(cell) != canonical->geometry_digest) {
                    ok = false;
                    break;
                }
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        // Communicator classes of every stored cell, as the global refresh
        // copies them onto ghosts whose rules were owner-filtered.
        for (const auto& [gid, cell] : local_cells_by_gid) {
            const auto* canonical = findOwnerSourcedActiveCell(ledger, gid);
            cell_class[cell] =
                canonical == nullptr ? CellClass{} : canonical->klass;
        }

        // ---- Graph of the active cells and their face neighbors ----
        // Active cells in key order (the order of the classified cells),
        // then their inactive neighbors, which only count traversal visits.
        std::vector<std::size_t> active_order(ledger.size());
        std::iota(active_order.begin(), active_order.end(), std::size_t{0});
        std::vector<GlobalIndex> inactive_neighbors;
        std::vector<CellKey> graph_keys;
        std::vector<CellClass> graph_class;
        std::vector<GlobalIndex> graph_gid;
        std::vector<Real> graph_volume;
        std::vector<std::vector<std::uint64_t>> graph_rule_ids;
        std::vector<int> graph_provider;
        std::vector<int> graph_owner;
        std::vector<std::vector<std::size_t>> graph_neighbors;
        std::vector<std::size_t> graph_index_of_ledger(ledger.size());
        ActiveFeatureLedger feature_ledger;
        std::optional<ClassifiedCellIndex> graph_cells;
        try {
            std::sort(active_order.begin(), active_order.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                          return ledger[lhs].key < ledger[rhs].key;
                      });
            for (std::size_t i = 1u; i < active_order.size(); ++i) {
                if (ledger[active_order[i - 1u]].key ==
                    ledger[active_order[i]].key) {
                    ok = false;
                }
            }
            for (const auto& cell : ledger) {
                for (const auto neighbor : cell.neighbor_gids) {
                    if (findOwnerSourcedActiveCell(ledger, neighbor) ==
                        nullptr) {
                        inactive_neighbors.push_back(neighbor);
                    }
                }
            }
            std::sort(inactive_neighbors.begin(), inactive_neighbors.end());
            inactive_neighbors.erase(
                std::unique(inactive_neighbors.begin(),
                            inactive_neighbors.end()),
                inactive_neighbors.end());
            const auto n_active = ledger.size();
            const auto n_graph = n_active + inactive_neighbors.size();
            graph_keys.reserve(n_graph);
            graph_class.reserve(n_graph);
            graph_gid.reserve(n_graph);
            graph_volume.reserve(n_graph);
            graph_rule_ids.reserve(n_graph);
            graph_provider.reserve(n_graph);
            graph_owner.reserve(n_graph);
            for (std::size_t g = 0u; g < n_active; ++g) {
                const auto& cell = ledger[active_order[g]];
                graph_index_of_ledger[active_order[g]] = g;
                graph_keys.push_back(cell.key);
                graph_class.push_back(cell.klass);
                graph_gid.push_back(cell.gid);
                graph_volume.push_back(cell.volume);
                graph_rule_ids.push_back(cell.rule_ids);
                graph_provider.push_back(cell.provider_rank);
                graph_owner.push_back(cell.owner_rank);
            }
            for (const auto gid : inactive_neighbors) {
                // Sorts after every field-DOF key and is unique per cell.
                graph_keys.push_back(
                    CellKey{std::numeric_limits<GlobalIndex>::max(), gid});
                graph_class.push_back(CellClass{});
                graph_gid.push_back(gid);
                graph_volume.push_back(Real{0.0});
                graph_rule_ids.emplace_back();
                graph_provider.push_back(-1);
                graph_owner.push_back(-1);
            }
            const auto graph_index_of = [&](GlobalIndex gid) {
                const auto it = std::lower_bound(
                    ledger.begin(), ledger.end(), gid,
                    [](const OwnerSourcedActiveCell& cell, GlobalIndex key) {
                        return cell.gid < key;
                    });
                if (it != ledger.end() && it->gid == gid) {
                    return graph_index_of_ledger[static_cast<std::size_t>(
                        it - ledger.begin())];
                }
                const auto inactive = std::lower_bound(
                    inactive_neighbors.begin(), inactive_neighbors.end(), gid);
                return n_active + static_cast<std::size_t>(
                                      inactive - inactive_neighbors.begin());
            };
            graph_neighbors.assign(n_graph, {});
            for (std::size_t g = 0u; g < n_active; ++g) {
                auto& neighbors = graph_neighbors[g];
                for (const auto neighbor :
                     ledger[active_order[g]].neighbor_gids) {
                    neighbors.push_back(graph_index_of(neighbor));
                }
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                                neighbors.end());
            }
            // Face adjacency between active cells is symmetric.
            for (std::size_t g = 0u; ok && g < n_active; ++g) {
                for (const auto neighbor : graph_neighbors[g]) {
                    if (neighbor < n_active &&
                        !std::binary_search(graph_neighbors[neighbor].begin(),
                                            graph_neighbors[neighbor].end(),
                                            g)) {
                        ok = false;
                        break;
                    }
                }
            }
            graph_cells.emplace(std::move(graph_keys));
            if (ok) {
                buildActiveFeatureLedger(
                    AggregationCellGraph{
                        .cells = *graph_cells,
                        .klass = graph_class,
                        .physical_gid = graph_gid,
                        .volume = graph_volume,
                        .rule_ids = graph_rule_ids,
                        .provider_rank = graph_provider,
                        .owner_rank = graph_owner,
                        .neighbors = graph_neighbors,
                    },
                    feature_ledger);
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        const AggregationCellGraph graph{
            .cells = *graph_cells,
            .klass = graph_class,
            .physical_gid = graph_gid,
            .volume = graph_volume,
            .rule_ids = graph_rule_ids,
            .provider_rank = graph_provider,
            .owner_rank = graph_owner,
            .neighbors = graph_neighbors,
        };
        const auto ledger_cell_of_graph =
            [&](std::size_t graph_cell) -> const OwnerSourcedActiveCell& {
            return *findOwnerSourcedActiveCell(ledger, graph_gid[graph_cell]);
        };
        profile.mark(AggregationProfilePhase::Features);

        // ---- Candidates: merged at a home rank per DOF ----
        build_local_candidate_support();
        std::vector<std::vector<std::int64_t>> candidate_outgoing(n_ranks);
        try {
            for (const auto& support : local_candidate_support) {
                if (support.declaration_dof < 0 ||
                    support.component_dofs.empty() ||
                    support.component_dofs.front() != support.declaration_dof ||
                    !std::all_of(support.coordinates.begin(),
                                 support.coordinates.end(),
                                 [](Real value) {
                                     return std::isfinite(value);
                                 })) {
                    ok = false;
                    break;
                }
                const std::int64_t flags =
                    (support.touches_cut ? kCandidateCensusCut : 0) |
                    (support.touches_full_active ? kCandidateCensusFull : 0) |
                    (support.excluded ? kCandidateCensusExcluded : 0);
                auto& words = candidate_outgoing[static_cast<std::size_t>(
                    ownerSourcedHome(
                        static_cast<std::uint64_t>(support.declaration_dof),
                        world_size))];
                words.push_back(
                    static_cast<std::int64_t>(support.declaration_dof));
                if (!support.touches_cut || support.touches_full_active ||
                    support.excluded) {
                    words.push_back(flags | kCandidateCensusMarker);
                    continue;
                }
                words.push_back(flags);
                words.push_back(
                    static_cast<std::int64_t>(support.component_dofs.size()));
                for (const auto coordinate : support.coordinates) {
                    words.push_back(std::bit_cast<std::int64_t>(
                        static_cast<double>(coordinate)));
                }
                for (const auto dof : support.component_dofs) {
                    words.push_back(static_cast<std::int64_t>(dof));
                }
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words candidate_census;
        try {
            candidate_census = exchangeInt64Words(
                comm,
                candidate_outgoing,
                AggregationProfileExchange::CandidateCensus);
        } catch (...) {
            return false;
        }
        std::vector<std::int64_t> merged_words;
        try {
            ok = mergeCandidateCensus(candidate_census, world_size, merged_words);
        } catch (...) {
            ok = false;
        }
        candidate_census = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words merged_gathered;
        try {
            merged_gathered = allGatherInt64Words(
                comm,
                std::span<const std::int64_t>(merged_words),
                AggregationProfileExchange::MergedCandidates);
        } catch (...) {
            return false;
        }
        GlobalCandidateMap candidates;
        try {
            ok = decodeMergedCandidates(merged_gathered, candidates);
        } catch (...) {
            ok = false;
        }
        merged_gathered = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        profile.setCount(AggregationProfileCount::LocalCandidateSupports,
                         local_candidate_support.size());
        profile.setCount(AggregationProfileCount::CommunicatorCandidates,
                         candidates.size());
        profile.mark(AggregationProfilePhase::CandidateSupport);

        // ---- Bounded root search (replicated, as the global refresh) ----
        struct DeferredGuardRootlessLog {
            std::string reason{};
            GlobalIndex dof{INVALID_GLOBAL_INDEX};
            GlobalCandidateSupport support{};
            std::string detail{};
        };
        std::vector<DeferredGuardRootlessLog> deferred_logs;
        const GuardRootlessLogger defer_log =
            [&](std::string_view reason,
                GlobalIndex dof,
                const GlobalCandidateSupport& support,
                const std::string& detail) {
                deferred_logs.push_back(DeferredGuardRootlessLog{
                    .reason = std::string(reason),
                    .dof = dof,
                    .support = support,
                    .detail = detail});
            };
        CandidateRootTraversal traversal;
        try {
            traverseCandidateRoots(graph,
                                   feature_ledger.feature_of,
                                   candidates,
                                   guards_,
                                   /*slave_all_cut=*/false,
                                   defer_log,
                                   traversal);
            check_subparametric_active_cells();
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        profile.mark(AggregationProfilePhase::RootTraversal);

        // ---- Root proposals: built once, by the root's owner ----
        std::uint64_t weighted_failures = 0u;
        std::uint64_t weighted_extrapolation_rejections = 0u;
        std::uint64_t weighted_line_rejections = 0u;
        std::array<double, 6> owner_maxima{};
        std::vector<std::int64_t> outcome_words;
        std::map<std::pair<GlobalIndex, GlobalIndex>, AggregationDeclaration>
            owned_proposals;
        std::uint64_t owner_attempts = 0u;
        try {
            for (const auto& [dof, candidate] : candidates) {
                if (!candidate.globally_rooted) {
                    continue;
                }
                const auto roots = traversal.roots_by_candidate.find(dof);
                if (roots == traversal.roots_by_candidate.end()) {
                    continue;
                }
                for (const auto& root : roots->second) {
                    const auto& root_cell = ledger_cell_of_graph(root.cell);
                    if (root_cell.owner_rank != my_rank) {
                        continue;
                    }
                    const auto local_root = local_cell_of(root_cell.gid);
                    if (local_root == INVALID_GLOBAL_INDEX) {
                        ok = false;
                        break;
                    }
                    auto attempt = build_root_proposal(
                        dof, candidate, root.distance, local_root);
                    ++owner_attempts;
                    std::int64_t code =
                        (attempt.emitted ? kProposalEmitted : 0) |
                        (attempt.inversion_failed ? kProposalInversionFailed
                                                  : 0) |
                        (attempt.round_off_acceptance ? kProposalRoundOff : 0) |
                        (attempt.extrapolation_rejected
                             ? kProposalExtrapolationRejected
                             : 0) |
                        (attempt.empty_line ? kProposalEmptyLine : 0) |
                        (attempt.line_guard_rejected
                             ? kProposalLineGuardRejected
                             : 0);
                    if (attempt.emitted) {
                        if (normalizeRootProposal(attempt.declaration)) {
                            owned_proposals.emplace(
                                std::make_pair(dof, root_cell.gid),
                                std::move(attempt.declaration));
                        } else {
                            code |= kProposalDropped;
                        }
                    }
                    const auto weight =
                        static_cast<std::uint64_t>(root_cell.holder_count);
                    weighted_failures +=
                        weight *
                        ((attempt.inversion_failed ? 1u : 0u) +
                         (attempt.empty_line ? 1u : 0u) +
                         (attempt.extrapolation_rejected ? 1u : 0u) +
                         (attempt.line_guard_rejected ? 1u : 0u));
                    weighted_extrapolation_rejections +=
                        weight * (attempt.extrapolation_rejected ? 1u : 0u);
                    weighted_line_rejections +=
                        weight * (attempt.line_guard_rejected ? 1u : 0u);
                    owner_maxima[0] = std::max(
                        owner_maxima[0],
                        static_cast<double>(
                            attempt.attempted_reference_extrapolation));
                    owner_maxima[1] = std::max(
                        owner_maxima[1],
                        static_cast<double>(
                            attempt.observed_reference_extrapolation));
                    owner_maxima[2] = std::max(
                        owner_maxima[2],
                        static_cast<double>(
                            attempt.attempted_absolute_coefficient));
                    owner_maxima[3] = std::max(
                        owner_maxima[3],
                        static_cast<double>(
                            attempt.observed_absolute_coefficient));
                    owner_maxima[4] = std::max(
                        owner_maxima[4],
                        static_cast<double>(attempt.attempted_row_l1_norm));
                    owner_maxima[5] = std::max(
                        owner_maxima[5],
                        static_cast<double>(attempt.observed_row_l1_norm));
                    outcome_words.push_back(static_cast<std::int64_t>(dof));
                    outcome_words.push_back(
                        static_cast<std::int64_t>(root_cell.gid));
                    outcome_words.push_back(code);
                }
                if (!ok) {
                    break;
                }
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words outcomes;
        try {
            outcomes = allGatherInt64Words(
                comm,
                std::span<const std::int64_t>(outcome_words),
                AggregationProfileExchange::ProposalOutcomes);
        } catch (...) {
            return false;
        }
        // Rank-local diagnostic counters (each holder of a root counts the
        // attempt, as when every holder built it), the preferred valid
        // proposal of every candidate, and its owner.
        std::size_t local_inversion_failed = 0u;
        std::size_t local_round_off = 0u;
        std::size_t local_extrapolation_rejections = 0u;
        std::size_t local_empty_lines = 0u;
        std::size_t local_line_rejections = 0u;
        std::size_t local_aggregated = 0u;
        struct PreferredProposal {
            std::size_t distance{std::numeric_limits<std::size_t>::max()};
            GlobalIndex root_gid{INVALID_GLOBAL_INDEX};
        };
        std::map<GlobalIndex, PreferredProposal> preferred;
        try {
            if (outcomes.words.size() % 3u != 0u) {
                ok = false;
            }
            std::set<GlobalIndex> locally_emitted_candidates;
            std::map<GlobalIndex, std::map<GlobalIndex, std::size_t>>
                root_distances;
            for (std::size_t i = 0u; ok && i < outcomes.words.size(); i += 3u) {
                const auto dof = static_cast<GlobalIndex>(outcomes.words[i]);
                const auto root_gid =
                    static_cast<GlobalIndex>(outcomes.words[i + 1u]);
                const auto code = outcomes.words[i + 2u];
                if (local_cell_of(root_gid) != INVALID_GLOBAL_INDEX) {
                    local_inversion_failed +=
                        (code & kProposalInversionFailed) != 0 ? 1u : 0u;
                    local_round_off += (code & kProposalRoundOff) != 0 ? 1u : 0u;
                    local_extrapolation_rejections +=
                        (code & kProposalExtrapolationRejected) != 0 ? 1u : 0u;
                    local_empty_lines +=
                        (code & kProposalEmptyLine) != 0 ? 1u : 0u;
                    local_line_rejections +=
                        (code & kProposalLineGuardRejected) != 0 ? 1u : 0u;
                    if ((code & kProposalEmitted) != 0) {
                        locally_emitted_candidates.insert(dof);
                    }
                }
                if ((code & kProposalEmitted) == 0 ||
                    (code & kProposalDropped) != 0) {
                    continue;
                }
                auto distances = root_distances.find(dof);
                if (distances == root_distances.end()) {
                    auto& by_root = root_distances[dof];
                    const auto roots = traversal.roots_by_candidate.find(dof);
                    if (roots != traversal.roots_by_candidate.end()) {
                        for (const auto& root : roots->second) {
                            by_root.emplace(graph_gid[root.cell], root.distance);
                        }
                    }
                    distances = root_distances.find(dof);
                }
                const auto distance = distances->second.find(root_gid);
                if (distance == distances->second.end()) {
                    ok = false;
                    break;
                }
                auto& best = preferred[dof];
                if (std::tie(distance->second, root_gid) <
                    std::tie(best.distance, best.root_gid)) {
                    best.distance = distance->second;
                    best.root_gid = root_gid;
                }
            }
            local_aggregated = locally_emitted_candidates.size();
        } catch (...) {
            ok = false;
        }
        outcomes = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        std::vector<std::int64_t> preferred_words;
        bool preferred_payload_valid = true;
        try {
            LocalAggregationDeclarationMap preferred_owned;
            for (const auto& [dof, best] : preferred) {
                const auto owned =
                    owned_proposals.find(std::make_pair(dof, best.root_gid));
                if (owned != owned_proposals.end()) {
                    preferred_owned[dof].push_back(owned->second);
                }
            }
            serializeRootProposals(preferred_owned,
                                   candidates,
                                   preferred_words,
                                   preferred_payload_valid);
            ok = preferred_payload_valid;
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words preferred_gathered;
        try {
            preferred_gathered = allGatherInt64Words(
                comm,
                std::span<const std::int64_t>(preferred_words),
                AggregationProfileExchange::PreferredProposals);
        } catch (...) {
            return false;
        }
        RankedDeclarationMap declarations_by_dof;
        try {
            decodeRootProposals(
                preferred_gathered,
                world_size,
                [&](int, GlobalIndex root_gid) {
                    const auto* cell =
                        findOwnerSourcedActiveCell(ledger, root_gid);
                    return cell == nullptr ? -1 : cell->min_holder_rank;
                },
                declarations_by_dof);
            for (const auto& [dof, best] : preferred) {
                const auto found = declarations_by_dof.find(dof);
                if (found == declarations_by_dof.end() ||
                    found->second.size() != 1u ||
                    found->second.front().rank < 0 ||
                    found->second.front().declaration.root_cell_gid !=
                        best.root_gid) {
                    ok = false;
                    break;
                }
            }
            ok = ok && declarations_by_dof.size() == preferred.size();
        } catch (...) {
            ok = false;
        }
        preferred_gathered = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        const std::array<std::uint64_t, 3> local_weighted{{
            weighted_failures,
            weighted_extrapolation_rejections,
            weighted_line_rejections}};
        std::array<std::uint64_t, 3> summed_weighted{};
        MPI_Allreduce(local_weighted.data(),
                      summed_weighted.data(),
                      static_cast<int>(summed_weighted.size()),
                      MPI_UINT64_T,
                      MPI_SUM,
                      comm);
        std::array<double, 6> global_maxima{};
        MPI_Allreduce(owner_maxima.data(),
                      global_maxima.data(),
                      static_cast<int>(global_maxima.size()),
                      MPI_DOUBLE,
                      MPI_MAX,
                      comm);
        profile.setCount(AggregationProfileCount::LocalRootProposals,
                         owner_attempts);
        profile.mark(AggregationProfilePhase::RootProposals);

        // ---- Visibility reports and the canonical selection ----
        std::set<GlobalIndex> assembled_candidate_dofs;
        std::set<GlobalIndex> row_coupled_candidate_dofs;
        compute_assembled_candidate_dofs(candidates,
                                         assembled_candidate_dofs,
                                         row_coupled_candidate_dofs);
        // [tag, DOF] pairs: 1 owned, 2 strongly constrained, 3 assembled
        // with (a candidate DOF), 4 preferred master this rank lacks while it
        // assembles with that master's slave.
        std::vector<std::int64_t> visibility_words;
        try {
            std::set<GlobalIndex> owned_dofs;
            std::set<GlobalIndex> missing_masters;
            for (const auto& [dof, candidate] : candidates) {
                const GlobalIndex candidate_dof = dof;
                const auto preferred_line =
                    [&](GlobalIndex slave) -> const ConstraintLine* {
                    const auto found = declarations_by_dof.find(candidate_dof);
                    if (found == declarations_by_dof.end()) {
                        return nullptr;
                    }
                    for (const auto& line :
                         found->second.front().declaration.lines) {
                        if (line.slave_dof == slave) {
                            return &line;
                        }
                    }
                    return nullptr;
                };
                for (const auto slave : candidate.component_dofs) {
                    if (partition.isOwned(slave)) {
                        owned_dofs.insert(slave);
                    }
                    if (const auto* line = preferred_line(slave)) {
                        for (const auto& entry : line->entries) {
                            if (partition.isOwned(entry.master_dof)) {
                                owned_dofs.insert(entry.master_dof);
                            }
                        }
                    }
                    if (!partition.isRelevant(slave)) {
                        continue;
                    }
                    if (constraints.isConstrained(slave)) {
                        visibility_words.push_back(2);
                        visibility_words.push_back(
                            static_cast<std::int64_t>(slave));
                    }
                    if (!partition.isOwned(slave) &&
                        assembled_candidate_dofs.count(slave) == 0u) {
                        continue;
                    }
                    visibility_words.push_back(3);
                    visibility_words.push_back(
                        static_cast<std::int64_t>(slave));
                    if (const auto* line = preferred_line(slave)) {
                        for (const auto& entry : line->entries) {
                            if (!partition.isRelevant(entry.master_dof)) {
                                missing_masters.insert(entry.master_dof);
                            }
                        }
                    }
                }
            }
            for (const auto dof : owned_dofs) {
                visibility_words.push_back(1);
                visibility_words.push_back(static_cast<std::int64_t>(dof));
            }
            for (const auto dof : missing_masters) {
                visibility_words.push_back(4);
                visibility_words.push_back(static_cast<std::int64_t>(dof));
            }
        } catch (...) {
            ok = false;
        }
        if (!all_ok(ok)) {
            return false;
        }
        GatheredInt64Words visibility_gathered;
        try {
            visibility_gathered = allGatherInt64Words(
                comm,
                std::span<const std::int64_t>(visibility_words),
                AggregationProfileExchange::ResolutionVisibility);
        } catch (...) {
            return false;
        }
        OwnerSourcedVisibility visibility;
        DistributedAggregationResult result;
        CanonicalSelection selection;
        try {
            for (int rank = 0; ok && rank < world_size; ++rank) {
                const auto begin = static_cast<std::size_t>(
                    visibility_gathered
                        .displacements[static_cast<std::size_t>(rank)]);
                const auto count = static_cast<std::size_t>(
                    visibility_gathered
                        .counts[static_cast<std::size_t>(rank)]);
                if (count % 2u != 0u) {
                    ok = false;
                    break;
                }
                for (std::size_t i = 0u; i < count; i += 2u) {
                    const auto tag = visibility_gathered.words[begin + i];
                    const auto dof = static_cast<GlobalIndex>(
                        visibility_gathered.words[begin + i + 1u]);
                    switch (tag) {
                    case 1:
                        visibility.owners[dof].push_back(rank);
                        break;
                    case 2:
                        visibility.preconstrained_dofs.insert(dof);
                        break;
                    case 3:
                        visibility.assembling[dof].push_back(
                            ProposalDofVisibility{
                                .rank = rank, .needed = true});
                        break;
                    case 4:
                        visibility.missing_masters[rank].insert(dof);
                        break;
                    default:
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) {
                selectCanonicalAggregationLines(rec.name,
                                                candidates,
                                                declarations_by_dof,
                                                visibility,
                                                /*slave_all_cut=*/false,
                                                /*allow_unaggregated=*/false,
                                                guards_.rootless_fallback,
                                                defer_log,
                                                selection);
                ok = selection.failure_count == 0u &&
                     selection.guard_failure_count == 0u;
            }
            if (ok) {
                result.validation = DistributedAggregationValidation::Passed;
                result.canonical_candidate_vertices = candidates.size();
                result.canonical_rooted_candidate_vertices =
                    static_cast<std::size_t>(std::count_if(
                        candidates.begin(), candidates.end(),
                        [](const auto& entry) {
                            return entry.second.globally_rooted;
                        }));
                result.canonical_rootless_candidate_vertices =
                    candidates.size() -
                    result.canonical_rooted_candidate_vertices;
                result.canonical_halo_limited_root_choices =
                    selection.halo_limited_root_choices;
                result.canonical_guard_rootless_candidate_vertices =
                    selection.guard_rootless_candidate_vertices;
                std::ostringstream validation_failures;
                ok = validateCanonicalAggregationLines(partition,
                                                       visibility,
                                                       visibility,
                                                       selection,
                                                       result,
                                                       validation_failures) ==
                     0u;
            }
        } catch (...) {
            ok = false;
        }
        visibility_gathered = GatheredInt64Words{};
        if (!all_ok(ok)) {
            return false;
        }
        result.canonical_row_coupled_slaves_beyond_halo =
            countRowCoupledSlavesBeyondHalo(row_coupled_candidate_dofs,
                                            selection.canonical_lines,
                                            partition,
                                            comm);

        // ---- Commit: the refresh is the communicator-global one ----
        for (const auto& log : deferred_logs) {
            logGuardRootlessCandidate(system,
                                      rec.name,
                                      log.reason,
                                      log.dof,
                                      log.support.coordinates,
                                      log.support.feature_cells,
                                      log.detail);
        }
        canonical_active_features = std::move(feature_ledger.features);
        canonical_active_cells = std::move(feature_ledger.cells);
        canonical_rooted_active_feature_count = feature_ledger.rooted;
        canonical_rootless_active_feature_count = feature_ledger.rootless;
        canonical_rootless_active_physical_volume =
            feature_ledger.rootless_volume;
        maximum_observed_root_path = traversal.maximum_observed_root_path;
        root_path_guard_rejections = traversal.root_path_guard_rejections;
        root_path_search_cell_visits = traversal.root_path_search_cell_visits;
        root_path_seed_index_entries = traversal.root_path_seed_index_entries;
        constrained_vertex_count = 0u;
        for (const auto& support : local_candidate_support) {
            if (candidates.count(support.declaration_dof) != 0u &&
                support.touches_cut) {
                ++constrained_vertex_count;
            }
        }
        no_root = static_cast<std::size_t>(std::count_if(
            candidates.begin(), candidates.end(),
            [](const auto& entry) { return !entry.second.globally_rooted; }));
        aggregated = local_aggregated;
        inversion_failed = local_inversion_failed;
        inversion_round_off_acceptances = local_round_off;
        empty_lines = local_empty_lines;
        extrapolation_guard_rejections = local_extrapolation_rejections;
        line_guard_rejections = local_line_rejections;
        communicator_root_proposal_failures =
            static_cast<std::size_t>(summed_weighted[0]);
        communicator_extrapolation_guard_rejections =
            static_cast<std::size_t>(summed_weighted[1]);
        communicator_line_guard_rejections =
            static_cast<std::size_t>(summed_weighted[2]);
        maximum_attempted_reference_extrapolation =
            static_cast<Real>(global_maxima[0]);
        maximum_observed_reference_extrapolation =
            static_cast<Real>(global_maxima[1]);
        maximum_attempted_absolute_coefficient =
            static_cast<Real>(global_maxima[2]);
        maximum_observed_absolute_coefficient =
            static_cast<Real>(global_maxima[3]);
        maximum_attempted_row_l1_norm = static_cast<Real>(global_maxima[4]);
        maximum_observed_row_l1_norm = static_cast<Real>(global_maxima[5]);
        distributed_result = std::move(result);
        // Communicator-global counts, as the global refresh records them.
        diagnostics::recordAggregationHalo(
            distributed_result.canonical_halo_limited_root_choices,
            distributed_result.canonical_row_coupled_slaves_beyond_halo);
        if (guards_.rootless_fallback) {
            diagnostics::noteAggregationGuardRootlessFallbackEnabled();
        }
        diagnostics::recordAggregationGuardRootless(
            root_path_guard_rejections,
            distributed_result.canonical_guard_rootless_candidate_vertices);
        return true;
    };
#endif

    bool owner_sourced_committed = false;
#if FE_HAS_MPI
    if (owner_sourced_eligible) {
        owner_sourced_committed = run_owner_sourced_refresh();
        detail::noteSmallCutAggregationOwnerSourcedRefresh(
            owner_sourced_committed);
    }
    profile.setCount(AggregationProfileCount::OwnerSourcedRefresh,
                     owner_sourced_committed ? 1u : 0u);
#endif
    if (!owner_sourced_committed) {
    if (!serialize_cell_classes) {
        // Returning from the owner-sourced refresh: serialize the classified
        // cells it checked.
        serialize_cell_classes = true;
        std::exception_ptr local_serialization_exception;
        try {
            declare_ordered_cell_classes();
        } catch (...) {
            local_serialization_exception = std::current_exception();
        }
        coordinateLocalPhaseFailure(local_serialization_exception,
                                    "cell_class_and_key_serialization");
    }

    GatheredInt64Words gathered_cell_classes;
    int aggregation_world_size = 1;
#if FE_HAS_MPI
    if (mpi_initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        MPI_Comm_size(comm, &aggregation_world_size);
        if (aggregation_world_size > 1) {
            gathered_cell_classes = allGatherInt64Words(
                comm,
                std::span<const std::int64_t>(local_cell_class_words),
                AggregationProfileExchange::CellClasses);
        }
    }
#endif
    if (aggregation_world_size == 1) {
        gathered_cell_classes.words = local_cell_class_words;
        gathered_cell_classes.counts = {
            static_cast<int>(local_cell_class_words.size())};
        gathered_cell_classes.displacements = {0};
    }

    struct ActiveCellMeasureDeclaration {
        GlobalIndex physical_cell_gid{INVALID_GLOBAL_INDEX};
        std::size_t rule_count{0u};
        Real physical_volume{0.0};
        std::vector<std::uint64_t> stable_rule_ids{};
        int provider_rank{-1};
    };
    struct DecodedCellDeclaration {
        CellKey key{};
        std::int64_t flags{0};
        ActiveCellMeasureDeclaration measure{};
    };
    // Communicator-global classified cells (sorted keys) and per-cell
    // arrays indexed like them. The measure arrays hold the canonical
    // provider's declaration for active cells only.
    ClassifiedCellIndex classified_cells;
    // Global index of each local_cell_by_key entry (npos when unclassified).
    std::vector<std::size_t> local_cell_global_index;
    std::vector<CellClass> global_cell_class_of;
    std::vector<GlobalIndex> global_physical_cell_gid_of;
    std::vector<Real> global_active_cell_physical_volume_of;
    std::vector<std::vector<std::uint64_t>> global_active_cell_rule_ids_of;
    std::vector<int> global_active_cell_measure_provider_rank_of;
    std::uint64_t local_owned_mesh_cells = 0u;
    std::exception_ptr local_class_decode_exception;
    try {
    std::vector<DecodedCellDeclaration> decoded_cell_declarations;
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
            if (static_cast<std::uint64_t>(rule_count_word) >
                static_cast<std::uint64_t>(end - position)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: malformed distributed "
                    "retained-rule identity payload");
            }
            std::vector<std::uint64_t> stable_rule_ids;
            stable_rule_ids.reserve(
                static_cast<std::size_t>(rule_count_word));
            for (std::int64_t rule = 0;
                 rule < rule_count_word;
                 ++rule) {
                stable_rule_ids.push_back(
                    std::bit_cast<std::uint64_t>(
                        gathered_cell_classes.words[position++]));
            }
            const auto repeated_nonzero_identity =
                std::adjacent_find(
                    stable_rule_ids.begin(),
                    stable_rule_ids.end(),
                    [](std::uint64_t lhs, std::uint64_t rhs) {
                        return lhs != 0u && lhs == rhs;
                    });
            if (!std::is_sorted(stable_rule_ids.begin(),
                                stable_rule_ids.end()) ||
                repeated_nonzero_identity != stable_rule_ids.end()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: non-canonical retained "
                    "rule identity payload");
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
            decoded_cell_declarations.push_back(DecodedCellDeclaration{
                .key = std::move(key),
                .flags = flags,
                .measure = ActiveCellMeasureDeclaration{
                    .physical_cell_gid = physical_cell_gid,
                    .rule_count = static_cast<std::size_t>(rule_count_word),
                    .physical_volume = physical_volume,
                    .stable_rule_ids = std::move(stable_rule_ids),
                    .provider_rank = rank,
                }});
        }
    }
    // Group the declarations of each cell; the stable sort keeps them in
    // ascending provider rank within a cell.
    std::stable_sort(decoded_cell_declarations.begin(),
                     decoded_cell_declarations.end(),
                     [](const DecodedCellDeclaration& lhs,
                        const DecodedCellDeclaration& rhs) {
                         return lhs.key < rhs.key;
                     });
    std::vector<std::pair<std::size_t, std::size_t>> declaration_groups;
    for (std::size_t first = 0u; first < decoded_cell_declarations.size();) {
        std::size_t last = first + 1u;
        while (last < decoded_cell_declarations.size() &&
               decoded_cell_declarations[last].key ==
                   decoded_cell_declarations[first].key) {
            ++last;
        }
        declaration_groups.emplace_back(first, last);
        first = last;
    }
    // A declaration is a positive classification fact even when flags==0
    // (inactive-full). Every rank retaining the same cell must therefore
    // report the exact same fact; OR-combining flags would silently accept,
    // for example, inactive-full on one rank and cut on another.
    for (const auto& [first, last] : declaration_groups) {
        for (std::size_t k = first + 1u; k < last; ++k) {
            if (decoded_cell_declarations[k].flags !=
                decoded_cell_declarations[first].flags) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "inconsistent_distributed_cell_classification field='" +
                    rec.name + "' communicator ranks reported different "
                    "class flags for the same cell");
            }
        }
    }
    for (const auto& [first, last] : declaration_groups) {
        static_cast<void>(last);
        if (decoded_cell_declarations[first].flags == 3) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: diagnostic="
                "inconsistent_distributed_cell_classification field='" +
                rec.name + "' communicator ranks classified the same cell as "
                "both full-active and cut");
        }
    }

    {
        std::vector<CellKey> classified_keys;
        classified_keys.reserve(declaration_groups.size());
        for (const auto& [first, last] : declaration_groups) {
            static_cast<void>(last);
            classified_keys.push_back(decoded_cell_declarations[first].key);
        }
        classified_cells = ClassifiedCellIndex(std::move(classified_keys));
    }
    const auto n_classified = classified_cells.size();
    global_cell_class_of.assign(n_classified, CellClass{});
    global_physical_cell_gid_of.assign(n_classified, INVALID_GLOBAL_INDEX);
    global_active_cell_physical_volume_of.assign(n_classified, Real{0.0});
    global_active_cell_rule_ids_of.assign(n_classified, {});
    global_active_cell_measure_provider_rank_of.assign(n_classified, -1);
    for (std::size_t index = 0u; index < n_classified; ++index) {
        const auto [first, last] = declaration_groups[index];
        const auto flags = decoded_cell_declarations[first].flags;
        auto& klass = global_cell_class_of[index];
        klass.full_active = (flags & 1) != 0;
        klass.cut = (flags & 2) != 0;
        if (!klass.full_active && !klass.cut) {
            continue;
        }
        const auto& canonical = decoded_cell_declarations[first].measure;
        for (std::size_t k = first; k < last; ++k) {
            const auto& provider = decoded_cell_declarations[k].measure;
            if (provider.physical_cell_gid !=
                    canonical.physical_cell_gid ||
                provider.rule_count != canonical.rule_count ||
                provider.physical_volume !=
                    canonical.physical_volume ||
                provider.stable_rule_ids !=
                    canonical.stable_rule_ids) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "inconsistent_distributed_active_feature_volume "
                    "providers disagree for the same physical cell");
            }
        }
        global_physical_cell_gid_of[index] = canonical.physical_cell_gid;
        global_active_cell_physical_volume_of[index] =
            canonical.physical_volume;
        global_active_cell_rule_ids_of[index] = canonical.stable_rule_ids;
        global_active_cell_measure_provider_rank_of[index] =
            canonical.provider_rank;
    }
    {
        // Both sequences are sorted by key: one merge pass.
        local_cell_global_index.assign(local_cell_by_key.size(),
                                       ClassifiedCellIndex::npos);
        std::size_t global_index = 0u;
        std::size_t local_index = 0u;
        for (const auto& [key, cell] : local_cell_by_key) {
            static_cast<void>(cell);
            while (global_index < n_classified &&
                   classified_cells.key(global_index) < key) {
                ++global_index;
            }
            if (global_index < n_classified &&
                classified_cells.key(global_index) == key) {
                local_cell_global_index[local_index] = global_index;
            }
            ++local_index;
        }
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
    if (classified_cells.empty()) {
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
        static_cast<std::uint64_t>(classified_cells.size());
    if (!communicator_cell_classification_complete &&
        runtime_options.max_lines ==
            std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "SmallCutAggregationConstraint: diagnostic="
            "incomplete_distributed_aggregation_context reason="
            "global_root_traversal_requires_all_owned_cell_classification "
            "classified_cells=" +
            std::to_string(classified_cells.size()) +
            " communicator_owned_cells=" +
            std::to_string(communicator_owned_mesh_cells));
    }

    profile.setCount(AggregationProfileCount::CommunicatorClassifiedCells,
                     classified_cells.size());
    profile.mark(AggregationProfilePhase::CellClassExchange);
    // Prove that every classified global cell has exactly one owner. This is
    // necessary for accepting owner-sourced topology. A separate all-owned
    // cell-count equality above proves that flags=0 inactive-full declarations
    // from the opposite-side traversal exhaust the mesh before any rootless
    // candidate may be pinned.
    std::vector<std::int64_t> local_cell_visibility_words;
    std::exception_ptr local_cell_visibility_exception;
    try {
    std::size_t visibility_local_index = 0u;
    for (const auto& [key, cell] : local_cell_by_key) {
        if (local_cell_global_index[visibility_local_index++] ==
            ClassifiedCellIndex::npos) {
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
            std::span<const std::int64_t>(local_cell_visibility_words),
            AggregationProfileExchange::CellOwners);
    }
#endif
    if (aggregation_world_size == 1) {
        gathered_cell_visibility.words = local_cell_visibility_words;
        gathered_cell_visibility.counts = {
            static_cast<int>(local_cell_visibility_words.size())};
        gathered_cell_visibility.displacements = {0};
    }
    profile.mark(AggregationProfilePhase::CellOwnerExchange);
    std::vector<std::size_t> global_cell_owner_count_of;
    std::vector<int> global_cell_owner_rank_of;
    using FaceKey = std::vector<GlobalIndex>;
    std::vector<FaceKey> cell_face_keys;
    std::vector<GlobalIndex> graph_cell_nodes;
    std::vector<std::int64_t> local_face_words;
    std::exception_ptr local_owner_face_exception;
    try {
    global_cell_owner_count_of.assign(classified_cells.size(), 0u);
    global_cell_owner_rank_of.assign(classified_cells.size(), -1);
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
            const std::span<const GlobalIndex> key(
                gathered_cell_visibility.words.data() + position,
                static_cast<std::size_t>(key_count));
            position += static_cast<std::size_t>(key_count);
            // Only classified cells are declared; the first owner is kept.
            const auto index = classified_cells.find(key);
            if (index != ClassifiedCellIndex::npos) {
                global_cell_owner_count_of[index] += owned ? 1u : 0u;
                if (owned && global_cell_owner_rank_of[index] < 0) {
                    global_cell_owner_rank_of[index] = rank;
                }
            }
        }
    }
    for (std::size_t index = 0u; index < classified_cells.size(); ++index) {
        if (global_cell_owner_count_of[index] != 1u) {
            throw InsufficientGhostHaloError(
                "SmallCutAggregationConstraint: diagnostic="
                "incomplete_distributed_aggregation_halo reason="
                "global_cell_owner_count:" +
                std::to_string(global_cell_owner_count_of[index]));
        }
        if (global_cell_owner_rank_of[index] < 0) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: canonical cell owner rank "
                "was not retained");
        }
    }

    // Copy communicator-global classifications onto every local geometry
    // holder, including ghosts whose cut rules were owner-filtered.
    {
        std::size_t local_index = 0u;
        for (const auto& [key, cell] : local_cell_by_key) {
            static_cast<void>(key);
            const auto index = local_cell_global_index[local_index++];
            if (index != ClassifiedCellIndex::npos) {
                cell_class[cell] = global_cell_class_of[index];
            }
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
    std::size_t face_local_index = 0u;
    for (const auto& [key, cell] : local_cell_by_key) {
        if (local_cell_global_index[face_local_index++] ==
                ClassifiedCellIndex::npos ||
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
        const auto& ref =
            elements::ReferenceElement::shared(mesh.getCellType(cell));
        // Face-key buffers are reused from cell to cell.
        const auto n_cell_faces = static_cast<std::size_t>(ref.num_faces());
        if (cell_face_keys.size() < n_cell_faces) {
            cell_face_keys.resize(n_cell_faces);
        }
        for (std::size_t face = 0; face < n_cell_faces; ++face) {
            auto& face_key = cell_face_keys[face];
            face_key.clear();
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
        }
        const auto cell_faces_end =
            cell_face_keys.begin() +
            static_cast<std::ptrdiff_t>(n_cell_faces);
        std::sort(cell_face_keys.begin(), cell_faces_end);
        if (std::adjacent_find(cell_face_keys.begin(), cell_faces_end) !=
            cell_faces_end) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: classified cell has duplicate "
                "face-DOF signatures");
        }

        // Cells are visited in ascending key order, the order of the
        // declarations: [cell_key_count, cell_key..., face_count,
        // {face_key_count, face_key...}...]
        local_face_words.push_back(static_cast<std::int64_t>(key.size()));
        for (const auto dof : key) {
            local_face_words.push_back(static_cast<std::int64_t>(dof));
        }
        local_face_words.push_back(static_cast<std::int64_t>(n_cell_faces));
        for (auto face = cell_face_keys.begin(); face != cell_faces_end;
             ++face) {
            local_face_words.push_back(static_cast<std::int64_t>(face->size()));
            for (const auto dof : *face) {
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
            std::span<const std::int64_t>(local_face_words),
            AggregationProfileExchange::Faces);
    }
#endif
    if (aggregation_world_size == 1) {
        gathered_faces.words = local_face_words;
        gathered_faces.counts = {static_cast<int>(local_face_words.size())};
        gathered_faces.displacements = {0};
    }

    // Face-neighbor lists per classified cell, in ascending index (= key)
    // order like the std::set<CellKey> values they replace.
    std::vector<std::vector<std::size_t>> global_neighbors;
    std::exception_ptr local_face_decode_exception;
    try {
    global_neighbors.assign(classified_cells.size(), {});
    std::vector<char> has_owned_faces(classified_cells.size(), 0);
    // Keys are read as views into the gathered words (GlobalIndex is the
    // wire type), so no key is copied.
    using KeyView = std::span<const GlobalIndex>;
    std::vector<std::pair<KeyView, std::size_t>> face_incidences;
    auto read_serialized_key = [](const GatheredInt64Words& gathered,
                                  std::size_t& position,
                                  std::size_t end,
                                  std::string_view kind) -> KeyView {
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
        const KeyView key(gathered.words.data() + position,
                          static_cast<std::size_t>(count));
        position += static_cast<std::size_t>(count);
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
            std::vector<KeyView> faces;
            faces.reserve(static_cast<std::size_t>(face_count));
            for (std::int64_t face = 0; face < face_count; ++face) {
                faces.push_back(read_serialized_key(
                    gathered_faces, position, end, "owner-face signature"));
            }
            // Owners declare only communicator-classified cells.
            const auto cell_index = classified_cells.find(cell);
            if (cell_index == ClassifiedCellIndex::npos) {
                throw std::logic_error(
                    "SmallCutAggregationConstraint: owner-face declaration "
                    "names an unclassified cell");
            }
            if (has_owned_faces[cell_index] != 0) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: classified cell has "
                    "multiple owner-face declarations");
            }
            has_owned_faces[cell_index] = 1;
            for (const auto face : faces) {
                face_incidences.emplace_back(face, cell_index);
            }
        }
    }
    for (std::size_t index = 0u; index < classified_cells.size(); ++index) {
        if (has_owned_faces[index] == 0) {
            throw InsufficientGhostHaloError(
                "SmallCutAggregationConstraint: diagnostic="
                "incomplete_distributed_aggregation_halo reason="
                "missing_owned_face_signature");
        }
    }

    // Faces in ascending signature order, each with its incident cells in
    // ascending index order.
    const auto same_key = [](KeyView a, KeyView b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end());
    };
    std::sort(face_incidences.begin(), face_incidences.end(),
              [&](const auto& a, const auto& b) {
                  if (!same_key(a.first, b.first)) {
                      return std::lexicographical_compare(
                          a.first.begin(), a.first.end(),
                          b.first.begin(), b.first.end());
                  }
                  return a.second < b.second;
              });
    for (std::size_t first = 0u; first < face_incidences.size();) {
        std::size_t last = first + 1u;
        while (last < face_incidences.size() &&
               same_key(face_incidences[last].first,
                        face_incidences[first].first)) {
            ++last;
        }
        std::size_t distinct_cells = 0u;
        std::array<std::size_t, 2> incident{};
        for (std::size_t k = first; k < last; ++k) {
            if (k > first &&
                face_incidences[k].second == face_incidences[k - 1u].second) {
                continue;
            }
            if (distinct_cells == 2u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: non-manifold classified "
                    "band face has more than two incident cells");
            }
            incident[distinct_cells++] = face_incidences[k].second;
        }
        if (distinct_cells == 2u) {
            global_neighbors[incident[0]].push_back(incident[1]);
            global_neighbors[incident[1]].push_back(incident[0]);
        }
        first = last;
    }
    for (auto& neighbors : global_neighbors) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
    }
    } catch (...) {
        local_face_decode_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_face_decode_exception,
                                "face_ledger_decode_and_graph_construction");
    profile.mark(AggregationProfilePhase::FaceGraph);

    // 2b. Canonical retained active features. The graph and per-cell measure
    // declarations are communicator-global on every rank, so component
    // construction and sorted-GID accumulation require no replicated MPI
    // reduction.
    // Canonical feature ID per classified cell (INVALID for inactive cells).
    std::vector<GlobalIndex> active_feature_of;
    std::exception_ptr local_feature_exception;
    try {
    ActiveFeatureLedger feature_ledger;
    buildActiveFeatureLedger(
        AggregationCellGraph{
            .cells = classified_cells,
            .klass = global_cell_class_of,
            .physical_gid = global_physical_cell_gid_of,
            .volume = global_active_cell_physical_volume_of,
            .rule_ids = global_active_cell_rule_ids_of,
            .provider_rank = global_active_cell_measure_provider_rank_of,
            .owner_rank = global_cell_owner_rank_of,
            .neighbors = global_neighbors,
        },
        feature_ledger);
    canonical_active_features = std::move(feature_ledger.features);
    active_feature_of = std::move(feature_ledger.feature_of);
    canonical_active_cells = std::move(feature_ledger.cells);
    canonical_rooted_active_feature_count = feature_ledger.rooted;
    canonical_rootless_active_feature_count = feature_ledger.rootless;
    canonical_rootless_active_physical_volume = feature_ledger.rootless_volume;
    } catch (...) {
        local_feature_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_feature_exception,
                                "active_feature_ledger_construction");
    profile.mark(AggregationProfilePhase::Features);

    build_local_candidate_support();

    auto global_candidates = resolveGlobalCandidateSupport(
        system,
        rec.name,
        std::span<const LocalCandidateSupport>(local_candidate_support),
        slave_all_cut);
    profile.setCount(AggregationProfileCount::LocalCandidateSupports,
                     local_candidate_support.size());
    profile.setCount(AggregationProfileCount::CommunicatorCandidates,
                     global_candidates.size());
    profile.mark(AggregationProfilePhase::CandidateSupport);

    std::map<GlobalIndex, std::vector<GlobalRootCandidate>>
        global_roots_by_candidate;
    std::exception_ptr local_global_graph_exception;
    try {
    CandidateRootTraversal traversal;
    traverseCandidateRoots(
        AggregationCellGraph{
            .cells = classified_cells,
            .klass = global_cell_class_of,
            .physical_gid = global_physical_cell_gid_of,
            .volume = global_active_cell_physical_volume_of,
            .rule_ids = global_active_cell_rule_ids_of,
            .provider_rank = global_active_cell_measure_provider_rank_of,
            .owner_rank = global_cell_owner_rank_of,
            .neighbors = global_neighbors,
        },
        active_feature_of,
        global_candidates,
        guards_,
        slave_all_cut,
        [&](std::string_view reason,
            GlobalIndex dof,
            const GlobalCandidateSupport& support,
            const std::string& detail) {
            logGuardRootlessCandidate(system,
                                      rec.name,
                                      reason,
                                      dof,
                                      support.coordinates,
                                      support.feature_cells,
                                      detail);
        },
        traversal);
    global_roots_by_candidate = std::move(traversal.roots_by_candidate);
    maximum_observed_root_path = traversal.maximum_observed_root_path;
    root_path_guard_rejections = traversal.root_path_guard_rejections;
    root_path_search_cell_visits = traversal.root_path_search_cell_visits;
    root_path_seed_index_entries = traversal.root_path_seed_index_entries;
    } catch (...) {
        local_global_graph_exception = std::current_exception();
    }
    coordinateLocalPhaseFailure(local_global_graph_exception,
                                "global_candidate_root_traversal");
    profile.mark(AggregationProfilePhase::RootTraversal);

    std::exception_ptr local_proposal_exception;
    try {

    for (const auto& support : local_candidate_support) {
        const auto global_it = global_candidates.find(support.declaration_dof);
        if (global_it != global_candidates.end() && support.touches_cut) {
            ++constrained_vertex_count;
        }
    }

    check_subparametric_active_cells();

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
    root_proposal_outcomes.clear();
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
            const auto local_root = local_cell_by_key.find(
                classified_cells.key(root_candidate.cell));
            if (local_root == local_cell_by_key.end()) {
                continue;
            }
            auto attempt = build_root_proposal(declaration_dof,
                                               candidate,
                                               root_candidate.distance,
                                               local_root->second);
            apply_attempt_counters(attempt);
            if (!attempt.emitted) {
                continue;
            }
            local_aggregation_declarations[declaration_dof].push_back(
                std::move(attempt.declaration));
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

    communicator_root_proposal_failures =
        inversion_failed + empty_lines + extrapolation_guard_rejections +
        line_guard_rejections;
    communicator_extrapolation_guard_rejections =
        extrapolation_guard_rejections;
    communicator_line_guard_rejections = line_guard_rejections;
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
            const std::array<double, 6> local_guard_maxima{{
                static_cast<double>(
                    maximum_attempted_reference_extrapolation),
                static_cast<double>(
                    maximum_observed_reference_extrapolation),
                static_cast<double>(maximum_attempted_absolute_coefficient),
                static_cast<double>(maximum_observed_absolute_coefficient),
                static_cast<double>(maximum_attempted_row_l1_norm),
                static_cast<double>(maximum_observed_row_l1_norm),
            }};
            std::array<double, 6> global_guard_maxima{};
            MPI_Allreduce(local_guard_maxima.data(),
                          global_guard_maxima.data(),
                          static_cast<int>(global_guard_maxima.size()),
                          MPI_DOUBLE,
                          MPI_MAX,
                          comm);
            maximum_attempted_reference_extrapolation =
                static_cast<Real>(global_guard_maxima[0]);
            maximum_observed_reference_extrapolation =
                static_cast<Real>(global_guard_maxima[1]);
            maximum_attempted_absolute_coefficient =
                static_cast<Real>(global_guard_maxima[2]);
            maximum_observed_absolute_coefficient =
                static_cast<Real>(global_guard_maxima[3]);
            maximum_attempted_row_l1_norm =
                static_cast<Real>(global_guard_maxima[4]);
            maximum_observed_row_l1_norm =
                static_cast<Real>(global_guard_maxima[5]);
        }
    }
#endif

    if (profile.active()) {
        std::uint64_t local_root_proposals = 0u;
        for (const auto& [dof, declarations] :
             local_aggregation_declarations) {
            static_cast<void>(dof);
            local_root_proposals += declarations.size();
        }
        profile.setCount(AggregationProfileCount::LocalRootProposals,
                         local_root_proposals);
    }
    profile.mark(AggregationProfilePhase::RootProposals);
    if (max_lines == std::numeric_limits<std::size_t>::max()) {
        // Candidate DOFs this rank condenses: DOFs of its owned cells
        // (assembled here under reverse scatter) and of their face neighbors
        // (reached by facet terms). Such a rank needs the canonical line and
        // all masters, or the discretization would change. Under reverse
        // scatter, candidate DOFs of other local cells holding an owned DOF
        // only couple to owned rows; missing masters there cost Jacobian
        // entries, not correctness.
        std::set<GlobalIndex> assembled_candidate_dofs;
        std::set<GlobalIndex> row_coupled_candidate_dofs;
        compute_assembled_candidate_dofs(global_candidates,
                                         assembled_candidate_dofs,
                                         row_coupled_candidate_dofs);
        try {
            distributed_result = resolveDistributedAggregationDeclarations(
                system,
                constraints,
                rec.name,
                global_candidates,
                local_aggregation_declarations,
                assembled_candidate_dofs,
                row_coupled_candidate_dofs,
                slave_all_cut,
                allow_unaggregated,
                guards_.rootless_fallback);
        } catch (const std::exception& failure) {
            // Name every root of a candidate without a valid proposal and,
            // per rank, the outcome of each proposal that rank built, so the
            // log says whether a guard or the inversion rejected it.
            const std::string message = failure.what();
            if (message.find("aggregation_no_root_inside_guards") !=
                std::string::npos) {
                int outcome_rank = 0;
#if FE_HAS_MPI
                if (mpi_initialized != 0) {
                    MPI_Comm_rank(system.dofHandler().mpiComm(),
                                  &outcome_rank);
                }
#endif
                for (const auto& [dof, candidate] : global_candidates) {
                    if (message.find(" dof=" + std::to_string(dof) + " ") ==
                        std::string::npos) {
                        continue;
                    }
                    std::ostringstream detail;
                    detail << std::setprecision(10)
                           << "SmallCutAggregationConstraint: diagnostic="
                              "aggregation_root_proposal_outcomes rank="
                           << outcome_rank << " field='" << rec.name
                           << "' candidate_dof=" << dof << " xyz=("
                           << candidate.coordinates[0] << ","
                           << candidate.coordinates[1] << ","
                           << candidate.coordinates[2] << ") roots=";
                    const auto roots = global_roots_by_candidate.find(dof);
                    if (roots != global_roots_by_candidate.end()) {
                        for (const auto& root : roots->second) {
                            detail << "[gid="
                                   << global_physical_cell_gid_of[root.cell]
                                   << " distance=" << root.distance
                                   << " owner="
                                   << global_cell_owner_rank_of[root.cell]
                                   << "]";
                        }
                    }
                    detail << " local_proposals=";
                    bool any_local = false;
                    for (const auto& note : root_proposal_outcomes) {
                        if (note.candidate_dof != dof) {
                            continue;
                        }
                        any_local = true;
                        detail << "[gid=" << note.root_cell_gid
                               << " distance=" << note.distance
                               << " extrapolation=" << note.extrapolation
                               << " outcome=" << note.outcome << "]";
                    }
                    if (!any_local) {
                        detail << "none";
                    }
                    FE_LOG_WARNING(detail.str());
                }
            }
            throw;
        }
        // Communicator-global counts; the run summary reports them even when
        // the per-refresh diagnostic line is not printed.
        diagnostics::recordAggregationHalo(
            distributed_result.canonical_halo_limited_root_choices,
            distributed_result.canonical_row_coupled_slaves_beyond_halo);
        // Candidates resolved by the opt-in rootless-island fallback because
        // no root lay inside the guards (D33); canonical on every rank.
        if (guards_.rootless_fallback) {
            diagnostics::noteAggregationGuardRootlessFallbackEnabled();
        }
        diagnostics::recordAggregationGuardRootless(
            root_path_guard_rejections,
            distributed_result.canonical_guard_rootless_candidate_vertices);
    } else {
        distributed_result.validation =
            DistributedAggregationValidation::DebugBypass;
    }

    }

    profile.mark(AggregationProfilePhase::Resolution);
    const auto distributed_validation = distributed_result.validation;
    std::vector<SmallCutAggregationProlongationPatch>
        canonical_prolongation_patches;
    std::optional<AffineConstraints> next_constraints;
    std::vector<ConstraintLine> reuse_added_lines;
    std::shared_ptr<const detail::SmallCutAggregationReuseRecord>
        next_reuse_record;
    std::vector<GlobalIndex> next_previous_canonical_slaves;
    std::optional<SmallCutAggregationRefreshReport>
        next_completed_report;
    std::shared_ptr<
        const detail::SmallCutAggregationPendingProlongation>
        next_pending_prolongation;
    std::uint64_t next_successful_publication_ordinal =
        successful_publication_ordinal_;
    std::uint64_t next_feature_class_fingerprint = 0u;
    std::uint64_t next_slave_set_fingerprint = 0u;
    std::uint64_t next_geometry_fingerprint = 0u;
    std::exception_ptr local_publication_exception;
    try {
    next_constraints.emplace(constraints);
    auto& staged_constraints = *next_constraints;
    if (distributed_validation ==
        DistributedAggregationValidation::DebugBypass) {
        // Preserve the debug line-cap contract: no global completion is
        // attempted while intentional partial coverage is requested.
        for (const auto& [dof, declarations] :
             local_aggregation_declarations) {
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
                if (!staged_constraints.isConstrained(line.slave_dof)) {
                    distributed_result.relevant_lines.push_back(line);
                }
            }
        }
    }
    if (distributed_result.validation !=
        DistributedAggregationValidation::DebugBypass) {
        std::map<GlobalIndex, const SmallCutAggregationProlongationCell*>
            active_cell_by_gid;
        std::map<GlobalIndex,
                 std::vector<
                     const SmallCutAggregationProlongationCell*>>
            cut_cells_by_dof;
        std::map<GlobalIndex, std::set<GlobalIndex>>
            active_cell_gids_by_feature;
        for (const auto& cell : canonical_active_cells) {
            active_cell_by_gid.emplace(cell.cell_gid, &cell);
            active_cell_gids_by_feature[cell.active_feature_id].insert(
                cell.cell_gid);
            if (cell.kind ==
                SmallCutAggregationActiveCellKind::Cut) {
                for (const auto dof : cell.field_dofs) {
                    cut_cells_by_dof[dof].push_back(&cell);
                }
            }
        }

        struct PatchAccumulator {
            SmallCutAggregationProlongationPatch patch{};
            std::set<GlobalIndex> active_features{};
            std::set<GlobalIndex> members{};
            std::set<GlobalIndex> support{};
            std::set<GlobalIndex> slaves{};
        };
        std::map<std::pair<int, GlobalIndex>, PatchAccumulator>
            patch_accumulators;
        for (auto& row : distributed_result.canonical_rows) {
            if (row.provisional_kind ==
                SmallCutAggregationProvisionalRowKind::RootedExtension) {
                const auto root = active_cell_by_gid.find(
                    row.root_cell_gid);
                if (root == active_cell_by_gid.end() ||
                    root->second->kind !=
                        SmallCutAggregationActiveCellKind::FullActive) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: selected physical "
                        "root is absent from the canonical full-active cell "
                        "ledger");
                }
                row.root_cell_owner_rank = root->second->owner_rank;
                auto& accumulator = patch_accumulators[{
                    static_cast<int>(SmallCutAggregationPatchKind::Rooted),
                    row.root_cell_gid}];
                accumulator.patch.kind =
                    SmallCutAggregationPatchKind::Rooted;
                accumulator.patch.root_cell_gid = row.root_cell_gid;
                accumulator.patch.root_cell_owner_rank =
                    row.root_cell_owner_rank;
                accumulator.active_features.insert(
                    root->second->active_feature_id);
                accumulator.members.insert(row.root_cell_gid);
                accumulator.slaves.insert(row.slave_dof);
                bool found_cut_support = false;
                const auto cut_support =
                    cut_cells_by_dof.find(row.slave_dof);
                if (cut_support != cut_cells_by_dof.end()) {
                    for (const auto* cell : cut_support->second) {
                        accumulator.members.insert(cell->cell_gid);
                        accumulator.active_features.insert(
                            cell->active_feature_id);
                        found_cut_support = true;
                    }
                }
                if (!found_cut_support) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: rooted candidate has "
                        "no canonical cut-cell support");
                }
            } else if (
                row.provisional_kind ==
                SmallCutAggregationProvisionalRowKind::
                    RootlessHomogeneousPin) {
                std::set<GlobalIndex> feature_ids;
                const auto cut_support =
                    cut_cells_by_dof.find(row.slave_dof);
                if (cut_support != cut_cells_by_dof.end()) {
                    for (const auto* cell : cut_support->second) {
                        feature_ids.insert(cell->active_feature_id);
                    }
                }
                if (feature_ids.empty()) {
                    throw std::logic_error(
                        "SmallCutAggregationConstraint: rootless candidate "
                        "has no canonical active-feature support");
                }
                for (const auto feature_id : feature_ids) {
                    auto& accumulator = patch_accumulators[{
                        static_cast<int>(
                            SmallCutAggregationPatchKind::Rootless),
                        feature_id}];
                    accumulator.patch.kind =
                        SmallCutAggregationPatchKind::Rootless;
                    accumulator.active_features.insert(feature_id);
                    accumulator.slaves.insert(row.slave_dof);
                    const auto members =
                        active_cell_gids_by_feature.find(feature_id);
                    if (members !=
                        active_cell_gids_by_feature.end()) {
                        accumulator.members.insert(
                            members->second.begin(),
                            members->second.end());
                    }
                }
            }
        }

        std::set<GlobalIndex> covered_cut_cells;
        for (const auto& [key, accumulator] : patch_accumulators) {
            static_cast<void>(key);
            for (const auto gid : accumulator.members) {
                const auto cell = active_cell_by_gid.find(gid);
                if (cell != active_cell_by_gid.end() &&
                    cell->second->kind ==
                        SmallCutAggregationActiveCellKind::Cut) {
                    covered_cut_cells.insert(gid);
                }
            }
        }
        for (const auto& cell : canonical_active_cells) {
            if (cell.kind !=
                    SmallCutAggregationActiveCellKind::Cut ||
                covered_cut_cells.count(cell.cell_gid) != 0u) {
                continue;
            }
            auto& accumulator = patch_accumulators[{
                static_cast<int>(
                    SmallCutAggregationPatchKind::SupportedCut),
                cell.cell_gid}];
            accumulator.patch.kind =
                SmallCutAggregationPatchKind::SupportedCut;
            accumulator.active_features.insert(
                cell.active_feature_id);
            accumulator.members.insert(cell.cell_gid);
            accumulator.support.insert(cell.cell_gid);
            accumulator.support.insert(
                cell.active_face_neighbor_cell_gids.begin(),
                cell.active_face_neighbor_cell_gids.end());
        }
        canonical_prolongation_patches.reserve(
            patch_accumulators.size());
        for (auto& [key, accumulator] : patch_accumulators) {
            static_cast<void>(key);
            accumulator.patch.active_feature_ids.assign(
                accumulator.active_features.begin(),
                accumulator.active_features.end());
            accumulator.patch.member_cell_gids.assign(
                accumulator.members.begin(), accumulator.members.end());
            accumulator.support.insert(
                accumulator.members.begin(), accumulator.members.end());
            accumulator.patch.support_cell_gids.assign(
                accumulator.support.begin(),
                accumulator.support.end());
            accumulator.patch.slave_dofs.assign(
                accumulator.slaves.begin(), accumulator.slaves.end());
            canonical_prolongation_patches.push_back(
                std::move(accumulator.patch));
        }
    }
    for (const auto& line : distributed_result.relevant_lines) {
        if (staged_constraints.isConstrained(line.slave_dof)) {
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
        staged_constraints.addConstraintLine(line);
        if (reuse_inputs.has_value()) {
            reuse_added_lines.push_back(line);
        }
        current_slaves.push_back(line.slave_dof);
        if (line.entries.empty()) {
            ++island_pinned_dofs;
        }
    }
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

    // Detailed diagnostics, printed (at INFO) only when FE_LOG_LEVEL=DEBUG.
    // The fail-open warning below also quotes the first line, and the reuse
    // record keeps parts of it for later reused refreshes, so it is built
    // whenever either needs it.
    const bool log_aggregation_diagnostics =
        Logger::instance().get_level() <= LogLevel::DEBUG;
    std::ostringstream oss;
    if (!log_aggregation_diagnostics && !allow_unaggregated &&
        !reuse_inputs.has_value()) {
        oss.setstate(std::ios::badbit);  // the insertions below format nothing
    }
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
        << " local_root_proposal_inversion_round_off_acceptances="
        << inversion_round_off_acceptances
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
        << " guard_rootless_candidates="
        << (root_path_guard_rejections +
            distributed_result.canonical_guard_rootless_candidate_vertices)
        << " root_path_search=bounded_candidate_neighborhood"
        << " root_path_seed_index_entries="
        << root_path_seed_index_entries
        << " root_path_search_cell_visits="
        << root_path_search_cell_visits
        << " maximum_reference_extrapolation_distance="
        << guards_.maximum_reference_extrapolation_distance
        << " maximum_attempted_reference_extrapolation="
        << maximum_attempted_reference_extrapolation
        << " maximum_observed_reference_extrapolation="
        << maximum_observed_reference_extrapolation
        << " communicator_extrapolation_guard_rejections="
        << communicator_extrapolation_guard_rejections
        << " maximum_absolute_coefficient="
        << guards_.maximum_absolute_coefficient
        << " maximum_attempted_absolute_coefficient="
        << maximum_attempted_absolute_coefficient
        << " maximum_observed_absolute_coefficient="
        << maximum_observed_absolute_coefficient
        << " maximum_row_l1_norm="
        << guards_.maximum_row_l1_norm
        << " maximum_attempted_row_l1_norm="
        << maximum_attempted_row_l1_norm
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
        << " canonical_halo_limited_root_choices="
        << distributed_result.canonical_halo_limited_root_choices
        << " canonical_row_coupled_slaves_beyond_halo="
        << distributed_result.canonical_row_coupled_slaves_beyond_halo
        << " canonical_active_feature_count="
        << canonical_active_features.size()
        << " canonical_rooted_active_feature_count="
        << canonical_rooted_active_feature_count
        << " canonical_rootless_active_feature_count="
        << canonical_rootless_active_feature_count
        << " canonical_rootless_active_physical_volume="
        << canonical_rootless_active_physical_volume
        << " local_relevant_lines_prepared=" << current_slaves.size()
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
    if (log_aggregation_diagnostics) {
        FE_LOG_INFO(oss.str());
    }

    next_previous_canonical_slaves =
        distributed_result.canonical_slaves;
    if (log_aggregation_diagnostics) {
        // Churn is computed from the replicated canonical slave set, not the
        // overlap-dependent locally relevant set. State belongs to this
        // constraint instance, whose lifetime is tied to the FESystem; this
        // avoids stale process-global state when a system address is reused.
        const auto& canonical_slaves =
            next_previous_canonical_slaves;
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
            if (successful_publication_ordinal_ ==
                std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error(
                    "SmallCutAggregationConstraint: successful publication "
                    "ordinal is exhausted");
            }
            next_successful_publication_ordinal =
                successful_publication_ordinal_ + 1u;
            next_feature_class_fingerprint =
                canonicalFeatureClassFingerprint(
                    canonical_active_features);
            next_slave_set_fingerprint =
                canonicalSlaveSetFingerprint(
                    next_previous_canonical_slaves);
            auto prepared_provenance = prepareRefreshProvenance(
                system,
                *cut_context,
                interface_marker_,
                next_successful_publication_ordinal,
                staged_constraints);
            next_geometry_fingerprint =
                prepared_provenance
                    .geometry_identity.canonical_fingerprint;
            next_completed_report.emplace(
                SmallCutAggregationRefreshReport{
                    .field = field_,
                    .active_side = active_side_,
                    .interface_marker = interface_marker_,
                    .geometry_identity =
                        std::move(
                            prepared_provenance.geometry_identity),
                    .local_lineage =
                        prepared_provenance.local_lineage,
                    .canonical_feature_class_fingerprint =
                        next_feature_class_fingerprint,
                    .canonical_slave_set_fingerprint =
                        next_slave_set_fingerprint,
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
                        distributed_result
                            .canonical_rooted_candidate_vertices,
                    .canonical_rootless_candidate_vertices =
                        distributed_result
                            .canonical_rootless_candidate_vertices,
                    .canonical_owned_aggregate_dofs =
                        distributed_result.canonical_owned_aggregate_dofs,
                    .canonical_owned_pinned_dofs =
                        distributed_result.canonical_owned_pinned_dofs,
                    .canonical_strong_suppressed_dofs =
                        distributed_result
                            .canonical_strong_suppressed_dofs,
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
                });
            if (previous_completed_refresh_report.has_value()) {
                next_completed_report->canonical_topology_transition =
                    buildCanonicalTopologyTransitionReport(
                        *previous_completed_refresh_report,
                        *next_completed_report,
                        previous_canonical_slaves_,
                        next_previous_canonical_slaves);
            }

            auto pending = std::make_shared<
                detail::SmallCutAggregationPendingProlongation>();
            pending->field = field_;
            pending->active_side = active_side_;
            pending->interface_marker = interface_marker_;
            pending->slave_all_cut = slave_all_cut;
            pending->linear_extension = runtime_options.linear_extension;
            pending->allow_unaggregated = allow_unaggregated;
            pending->trace_bound_eligible =
                !allow_unaggregated &&
                std::all_of(
                    canonical_active_cells.begin(),
                    canonical_active_cells.end(),
                    [](const auto& cell) {
                        return !cell.retained_rule_stable_ids.empty() &&
                            std::none_of(
                                cell.retained_rule_stable_ids.begin(),
                                cell.retained_rule_stable_ids.end(),
                                [](std::uint64_t stable_id) {
                                    return stable_id == 0u;
                                });
                    }) &&
                std::none_of(
                    distributed_result.canonical_rows.begin(),
                    distributed_result.canonical_rows.end(),
                    [](const auto& row) {
                        return row.provisional_kind ==
                            SmallCutAggregationProvisionalRowKind::
                                UnaggregatedFreeIdentity;
                    });
            pending->guards = guards_;
            pending->cut_context_content_revision =
                cut_context->contentRevision();
            pending->has_free_surface_snapshot_revision =
                cut_context->hasFreeSurfaceGeometrySnapshotForMarker(
                    interface_marker_);
            if (pending->has_free_surface_snapshot_revision) {
                pending->free_surface_snapshot_revision =
                    cut_context
                        ->freeSurfaceGeometrySnapshotRevisionForMarker(
                            interface_marker_);
            }
            pending->has_source_value_revision =
                cut_context->hasExpectedGeneratedSourceValueRevision(
                    interface_marker_);
            if (pending->has_source_value_revision) {
                pending->source_value_revision =
                    cut_context->expectedGeneratedSourceValueRevision(
                        interface_marker_);
            }
            pending->rows =
                std::move(distributed_result.canonical_rows);
            pending->active_cells = std::move(canonical_active_cells);
            pending->patches =
                std::move(canonical_prolongation_patches);
            next_pending_prolongation = std::move(pending);
    }
    if (reuse_inputs.has_value() && next_completed_report.has_value() &&
        next_pending_prolongation) {
        next_reuse_record = makeAggregationReuseRecord(
            reuse_inputs->key,
            rec.name,
            std::move(reuse_added_lines),
            oss.str(),
            *next_completed_report,
            next_pending_prolongation,
            next_previous_canonical_slaves);
    }
    } catch (...) {
        local_publication_exception = std::current_exception();
    }

    // Exactly one post-resolver collective coordinates every allocation,
    // validation, diagnostic, staged-constraint, report, and pending-state
    // operation above. It also validates the rank-invariant canonical
    // fingerprints before any caller-owned or lifecycle state is published.
    coordinateRefreshPublicationConsensus(
        system,
        local_publication_exception,
        next_feature_class_fingerprint,
        next_slave_set_fingerprint,
        next_geometry_fingerprint,
        next_successful_publication_ordinal);

    // All operations after the final consensus are statically no-throw moves,
    // swaps, and scalar writes. In particular, aggregation lines were added
    // only to `next_constraints`; a coordinated failure leaves the caller-owned
    // AffineConstraints object untouched by this constraint instance.
    constraints = std::move(*next_constraints);
    if (distributed_validation !=
        DistributedAggregationValidation::DebugBypass) {
        next_completed_report->geometry_identity
            .communicator_fingerprint_consensus_validated =
                next_completed_report->geometry_identity.available;
        if (next_completed_report->canonical_topology_transition.has_value()) {
            auto& transition =
                *next_completed_report->canonical_topology_transition;
            transition.geometry_identity_before
                .communicator_fingerprint_consensus_validated =
                    transition.geometry_identity_before.available;
            transition.geometry_identity_after
                .communicator_fingerprint_consensus_validated =
                    transition.geometry_identity_after.available;
        }
        pending_prolongation_ =
            std::move(next_pending_prolongation);
        completed_refresh_report_ =
            std::move(next_completed_report);
        previous_canonical_slaves_.swap(
            next_previous_canonical_slaves);
        successful_publication_ordinal_ =
            next_successful_publication_ordinal;
        reuse_record_ = std::move(next_reuse_record);
    }
    profile.mark(AggregationProfilePhase::Publication);
    profile.report(system, rec.name, "full");
}

std::shared_ptr<const SmallCutAggregationProlongationReport>
SmallCutAggregationConstraint::finalizeProlongationReport(
    const systems::FESystem& system,
    const AffineConstraints& closed_constraints) const
{
    auto& profile = aggregationRefreshProfile();
    profile.begin();
    int rank = 0;
    int world_size = 1;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        const auto comm = system.dofHandler().mpiComm();
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &world_size);
        if (world_size > 1) {
            const int local_has_pending =
                pending_prolongation_ != nullptr ? 1 : 0;
            int minimum_has_pending = 0;
            int maximum_has_pending = 0;
            MPI_Allreduce(&local_has_pending,
                          &minimum_has_pending,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          comm);
            MPI_Allreduce(&local_has_pending,
                          &maximum_has_pending,
                          1,
                          MPI_INT,
                          MPI_MAX,
                          comm);
            if (minimum_has_pending != maximum_has_pending) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: pending prolongation "
                    "metadata is inconsistent across communicator ranks");
            }
            if (maximum_has_pending == 0) {
                return {};
            }
        }
    }
#endif
    if (pending_prolongation_ == nullptr) {
        return {};
    }
    const auto& pending = *pending_prolongation_;

    std::exception_ptr local_validation_exception;
    try {
        if (!closed_constraints.isClosed()) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: prolongation finalization "
                "requires a closed AffineConstraints object");
        }
        const auto* cut_context = system.cutIntegrationContext();
        if (cut_context == nullptr ||
            cut_context->contentRevision() !=
                pending.cut_context_content_revision) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: cut-context revision changed "
                "between aggregation and prolongation finalization");
        }
        const bool has_snapshot =
            cut_context->hasFreeSurfaceGeometrySnapshotForMarker(
                pending.interface_marker);
        if (has_snapshot !=
                pending.has_free_surface_snapshot_revision ||
            (has_snapshot &&
             cut_context
                     ->freeSurfaceGeometrySnapshotRevisionForMarker(
                         pending.interface_marker) !=
                 pending.free_surface_snapshot_revision)) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: free-surface snapshot "
                "revision changed during prolongation finalization");
        }
        const bool has_source_revision =
            cut_context->hasExpectedGeneratedSourceValueRevision(
                pending.interface_marker);
        if (has_source_revision !=
                pending.has_source_value_revision ||
            (has_source_revision &&
             cut_context->expectedGeneratedSourceValueRevision(
                 pending.interface_marker) !=
                 pending.source_value_revision)) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: source-value revision "
                "changed during prolongation finalization");
        }
    } catch (...) {
        local_validation_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(
        system,
        local_validation_exception,
        "prolongation_revision_validation");

    struct ClosedRow {
        bool constrained{false};
        std::vector<ConstraintEntry> entries{};
        Real inhomogeneity{0.0};
    };
    const auto global_dofs = system.dofHandler().getNumDofs();
    const auto& partition = system.dofHandler().getPartition();
    auto extract_closed_row = [&](GlobalIndex slave) {
        if (slave < 0 || slave >= global_dofs) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: finalized slave is outside "
                "the global DOF range");
        }
        ClosedRow row;
        const auto view = closed_constraints.getConstraint(slave);
        if (!view.has_value()) {
            row.entries.push_back(
                ConstraintEntry{.master_dof = slave, .weight = 1.0});
            return row;
        }
        row.constrained = true;
        row.inhomogeneity = static_cast<Real>(view->inhomogeneity);
        if (!std::isfinite(row.inhomogeneity)) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: finalized row has a "
                "non-finite inhomogeneity");
        }
        row.entries.assign(view->entries.begin(), view->entries.end());
        GlobalIndex previous_master = INVALID_GLOBAL_INDEX;
        for (const auto& entry : row.entries) {
            if (entry.master_dof < 0 ||
                entry.master_dof >= global_dofs ||
                entry.master_dof <= previous_master ||
                !std::isfinite(entry.weight) ||
                !partition.isRelevant(entry.master_dof) ||
                closed_constraints.isConstrained(entry.master_dof)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: finalized row does not "
                    "contain sorted terminal master DOFs");
            }
            previous_master = entry.master_dof;
        }
        return row;
    };

    std::map<
        GlobalIndex,
        const SmallCutAggregationProlongationRow*>
        pending_row_by_slave;
    std::vector<std::int64_t> local_owner_words;
    std::exception_ptr local_owner_exception;
    try {
        for (const auto& row : pending.rows) {
            if (!pending_row_by_slave.emplace(
                    row.slave_dof, &row).second) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: pending prolongation "
                    "contains a duplicate canonical slave");
            }
            if (row.slave_owner_rank < 0 ||
                row.slave_owner_rank >= world_size) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: finalized row has an "
                    "invalid canonical owner rank");
            }
            if (row.slave_owner_rank != rank) {
                continue;
            }
            if (!partition.isRelevant(row.slave_dof) ||
                !partition.isOwned(row.slave_dof)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: canonical owner cannot "
                    "materialize its finalized slave row");
            }
            const auto final_row = extract_closed_row(row.slave_dof);
            local_owner_words.push_back(
                static_cast<std::int64_t>(row.slave_dof));
            local_owner_words.push_back(
                final_row.constrained ? 1 : 0);
            local_owner_words.push_back(
                static_cast<std::int64_t>(
                    final_row.entries.size()));
            local_owner_words.push_back(std::bit_cast<std::int64_t>(
                static_cast<double>(final_row.inhomogeneity)));
            for (const auto& entry : final_row.entries) {
                local_owner_words.push_back(
                    static_cast<std::int64_t>(entry.master_dof));
                local_owner_words.push_back(
                    std::bit_cast<std::int64_t>(entry.weight));
            }
        }
    } catch (...) {
        local_owner_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(
        system,
        local_owner_exception,
        "prolongation_owner_row_serialization");

    GatheredInt64Words gathered_owner_rows;
#if FE_HAS_MPI
    if (world_size > 1) {
        gathered_owner_rows = allGatherInt64Words(
            system.dofHandler().mpiComm(),
            std::span<const std::int64_t>(local_owner_words),
            AggregationProfileExchange::OwnerRows);
    }
#endif
    if (world_size == 1) {
        gathered_owner_rows.words = local_owner_words;
        gathered_owner_rows.counts = {
            static_cast<int>(local_owner_words.size())};
        gathered_owner_rows.displacements = {0};
    }

    std::map<GlobalIndex, ClosedRow> canonical_closed_rows;
    std::exception_ptr local_decode_exception;
    try {
        for (int provider = 0; provider < world_size; ++provider) {
            std::size_t position = static_cast<std::size_t>(
                gathered_owner_rows
                    .displacements[static_cast<std::size_t>(provider)]);
            const auto end = position + static_cast<std::size_t>(
                gathered_owner_rows
                    .counts[static_cast<std::size_t>(provider)]);
            while (position < end) {
                if (end - position < 4u) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: malformed finalized "
                        "owner-row header");
                }
                const auto slave = static_cast<GlobalIndex>(
                    gathered_owner_rows.words[position++]);
                const auto pending_row =
                    pending_row_by_slave.find(slave);
                if (pending_row == pending_row_by_slave.end() ||
                    pending_row->second->slave_owner_rank !=
                        provider) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: finalized row was "
                        "not emitted by its declared canonical owner");
                }
                const auto constrained_word =
                    gathered_owner_rows.words[position++];
                const auto entry_count =
                    gathered_owner_rows.words[position++];
                const auto inhomogeneity = static_cast<Real>(
                    std::bit_cast<double>(
                        gathered_owner_rows.words[position++]));
                if ((constrained_word != 0 &&
                     constrained_word != 1) ||
                    entry_count < 0 ||
                    static_cast<std::uint64_t>(entry_count) * 2u >
                        static_cast<std::uint64_t>(end - position) ||
                    !std::isfinite(inhomogeneity)) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: malformed finalized "
                        "owner-row payload");
                }
                ClosedRow row;
                row.constrained = constrained_word != 0;
                row.inhomogeneity = inhomogeneity;
                row.entries.reserve(
                    static_cast<std::size_t>(entry_count));
                for (std::int64_t entry = 0;
                     entry < entry_count;
                     ++entry) {
                    const auto master = static_cast<GlobalIndex>(
                        gathered_owner_rows.words[position++]);
                    const auto weight = std::bit_cast<double>(
                        gathered_owner_rows.words[position++]);
                    if (master < 0 || master >= global_dofs ||
                        !std::isfinite(weight)) {
                        throw std::runtime_error(
                            "SmallCutAggregationConstraint: malformed "
                            "finalized owner-row entry");
                    }
                    row.entries.push_back(
                        ConstraintEntry{.master_dof = master,
                                        .weight = weight});
                }
                const auto [stored, inserted] =
                    canonical_closed_rows.emplace(
                        slave, std::move(row));
                if (!inserted) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: finalized row has "
                        "more than one canonical owner");
                }
                static_cast<void>(stored);
            }
        }
        if (canonical_closed_rows.size() != pending.rows.size()) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: finalized owner-row ledger "
                "does not cover every canonical candidate");
        }
        for (const auto& row : pending.rows) {
            if (canonical_closed_rows.count(row.slave_dof) != 1u) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: finalized owner-row "
                    "ledger is missing a canonical slave");
            }
        }
    } catch (...) {
        local_decode_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(
        system,
        local_decode_exception,
        "prolongation_owner_row_decode");

    // Rank-local description of a finalized row that disagrees with its
    // canonical owner, for the failure message.
    const auto describe_row_disagreement =
        [&](const SmallCutAggregationProlongationRow& pending_row,
            const ClosedRow& local,
            const ClosedRow& canonical) {
            std::ostringstream out;
            out.precision(17);
            const auto write_row = [&](const char* name,
                                       const ClosedRow& row) {
                out << ' ' << name << "_constrained="
                    << (row.constrained ? 1 : 0) << ' ' << name
                    << "_inhomogeneity=" << row.inhomogeneity << ' '
                    << name << "_entries=";
                for (const auto& entry : row.entries) {
                    out << entry.master_dof << ':' << entry.weight << ',';
                }
            };
            out << " rank=" << rank << " slave=" << pending_row.slave_dof
                << " slave_owner_rank=" << pending_row.slave_owner_rank
                << " slave_owned="
                << (partition.isOwned(pending_row.slave_dof) ? 1 : 0)
                << " provisional_kind="
                << static_cast<int>(pending_row.provisional_kind)
                << " preconstrained_at_apply="
                << (pending_row.preconstrained_at_apply ? 1 : 0);
            write_row("local", local);
            write_row("canonical", canonical);
            out << " provisional_entries(master:weight:relevant:constrained)=";
            for (const auto& entry : pending_row.provisional_entries) {
                out << entry.master_dof << ':' << entry.weight << ':'
                    << (partition.isRelevant(entry.master_dof) ? 1 : 0)
                    << ':'
                    << (closed_constraints.isConstrained(entry.master_dof)
                            ? 1
                            : 0)
                    << ',';
            }
            return out.str();
        };

    std::exception_ptr local_consistency_exception;
    try {
        for (const auto& pending_row : pending.rows) {
            if (!partition.isRelevant(pending_row.slave_dof)) {
                continue;
            }
            const auto& canonical =
                canonical_closed_rows.at(pending_row.slave_dof);
            // A rank that sees the slave only deeper in the halo, where some
            // master is absent, carries no line for it by design: a line is
            // installed (resolveDistributedAggregationDeclarations) and kept
            // (ParallelConstraints::synchronize) on a non-owning rank only
            // where every master of the unclosed canonical line is relevant,
            // and canonical_line_final_validation proved that every rank
            // assembling with the slave has them all. Closure can eliminate
            // the absent master (a strongly constrained DOF), so the closed
            // canonical entries alone do not identify such a rank; the
            // unclosed (provisional) entries do.
            const auto some_master_absent =
                [&](const std::vector<ConstraintEntry>& entries) {
                    return !std::all_of(
                        entries.begin(),
                        entries.end(),
                        [&](const ConstraintEntry& entry) {
                            return partition.isRelevant(entry.master_dof);
                        });
                };
            if (canonical.constrained &&
                !closed_constraints.isConstrained(pending_row.slave_dof) &&
                (some_master_absent(canonical.entries) ||
                 (!partition.isOwned(pending_row.slave_dof) &&
                  some_master_absent(pending_row.provisional_entries)))) {
                continue;
            }
            const auto local =
                extract_closed_row(pending_row.slave_dof);
            if (local.constrained != canonical.constrained ||
                aggregationDigestReal(local.inhomogeneity) !=
                    aggregationDigestReal(
                        canonical.inhomogeneity) ||
                local.entries.size() != canonical.entries.size()) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: relevant finalized row "
                    "disagrees with its canonical owner" +
                    describe_row_disagreement(pending_row, local, canonical));
            }
            for (std::size_t i = 0u;
                 i < local.entries.size();
                 ++i) {
                if (local.entries[i].master_dof !=
                        canonical.entries[i].master_dof ||
                    aggregationDigestReal(local.entries[i].weight) !=
                        aggregationDigestReal(
                            canonical.entries[i].weight)) {
                    throw std::runtime_error(
                        "SmallCutAggregationConstraint: relevant finalized "
                        "entries disagree with their canonical owner" +
                        describe_row_disagreement(
                            pending_row, local, canonical));
                }
            }
        }
    } catch (...) {
        local_consistency_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(
        system,
        local_consistency_exception,
        "prolongation_relevant_row_validation");

    std::shared_ptr<SmallCutAggregationProlongationReport> report;
    std::exception_ptr local_report_exception;
    try {
    report =
        std::make_shared<SmallCutAggregationProlongationReport>();
    report->field = pending.field;
    report->active_side = pending.active_side;
    report->interface_marker = pending.interface_marker;
    report->slave_all_cut = pending.slave_all_cut;
    report->linear_extension = pending.linear_extension;
    report->allow_unaggregated = pending.allow_unaggregated;
    report->trace_bound_eligible = pending.trace_bound_eligible;
    report->guards = pending.guards;
    report->revision.local_rank = rank;
    report->revision.communicator_size = world_size;
    report->revision.constraint =
        system.constraintRevisionSnapshot();
    report->revision.affine_constraint_layout_revision =
        closed_constraints.constraintLayoutRevision();
    report->revision.cut_context_content_revision =
        pending.cut_context_content_revision;
    report->revision.has_free_surface_snapshot_revision =
        pending.has_free_surface_snapshot_revision;
    report->revision.free_surface_snapshot_revision =
        pending.free_surface_snapshot_revision;
    report->revision.has_source_value_revision =
        pending.has_source_value_revision;
    report->revision.source_value_revision =
        pending.source_value_revision;
    report->rows = pending.rows;
    report->active_cells = pending.active_cells;
    report->patches = pending.patches;

    for (auto& row : report->rows) {
        const auto& final =
            canonical_closed_rows.at(row.slave_dof);
        row.final_entries = final.entries;
        row.final_inhomogeneity = final.inhomogeneity;
        if (!final.constrained) {
            row.final_kind =
                SmallCutAggregationFinalRowKind::Identity;
        } else if (!final.entries.empty()) {
            row.final_kind =
                SmallCutAggregationFinalRowKind::MasterBearing;
        } else if (final.inhomogeneity == Real{0.0}) {
            row.final_kind =
                SmallCutAggregationFinalRowKind::HomogeneousPin;
        } else {
            row.final_kind =
                SmallCutAggregationFinalRowKind::FixedValue;
        }

        switch (row.provisional_kind) {
        case SmallCutAggregationProvisionalRowKind::RootedExtension:
            if (row.final_kind ==
                    SmallCutAggregationFinalRowKind::Identity ||
                (row.preconstrained_at_apply &&
                 row.final_kind ==
                     SmallCutAggregationFinalRowKind::MasterBearing)) {
                report->trace_bound_eligible = false;
            }
            break;
        case SmallCutAggregationProvisionalRowKind::
                RootlessHomogeneousPin:
            if (row.final_kind !=
                    SmallCutAggregationFinalRowKind::HomogeneousPin &&
                row.final_kind !=
                    SmallCutAggregationFinalRowKind::FixedValue) {
                report->trace_bound_eligible = false;
            }
            break;
        case SmallCutAggregationProvisionalRowKind::
                SupportedFreeIdentity:
            if (row.final_kind ==
                SmallCutAggregationFinalRowKind::MasterBearing) {
                report->trace_bound_eligible = false;
            }
            break;
        case SmallCutAggregationProvisionalRowKind::
                UnaggregatedFreeIdentity:
            report->trace_bound_eligible = false;
            break;
        case SmallCutAggregationProvisionalRowKind::
                PreexistingConstraint:
            if (row.final_kind ==
                    SmallCutAggregationFinalRowKind::Identity ||
                row.final_kind ==
                    SmallCutAggregationFinalRowKind::MasterBearing) {
                report->trace_bound_eligible = false;
            }
            break;
        }
    }

    std::map<GlobalIndex, const SmallCutAggregationProlongationRow*>
        finalized_row_by_slave;
    for (const auto& row : report->rows) {
        finalized_row_by_slave.emplace(row.slave_dof, &row);
    }
    // Sorted (DOF, active cell GID) incidences and (cell GID, feature)
    // pairs instead of node-based maps: the stable sort keeps each DOF's
    // cells in ledger order, and lookups are binary searches.  Built once
    // per refresh and shared by its reuses (see ActiveCellIndex).
    if (!pending.active_cell_index ||
        pending.active_cell_index->cell_count != report->active_cells.size()) {
    auto built_index = std::make_shared<
        detail::SmallCutAggregationPendingProlongation::ActiveCellIndex>();
    built_index->cell_count = report->active_cells.size();
    auto& active_cell_gids_by_dof = built_index->cell_gids_by_dof;
    auto& active_feature_by_cell_gid = built_index->feature_by_cell_gid;
    {
        std::size_t incidence_count = 0u;
        for (const auto& cell : report->active_cells) {
            incidence_count += cell.field_dofs.size();
        }
        active_cell_gids_by_dof.reserve(incidence_count);
        active_feature_by_cell_gid.reserve(report->active_cells.size());
        for (const auto& cell : report->active_cells) {
            active_feature_by_cell_gid.emplace_back(
                cell.cell_gid, cell.active_feature_id);
            for (const auto dof : cell.field_dofs) {
                active_cell_gids_by_dof.emplace_back(dof, cell.cell_gid);
            }
        }
        const auto by_first = [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        };
        std::stable_sort(active_feature_by_cell_gid.begin(),
                         active_feature_by_cell_gid.end(),
                         by_first);
        if (std::adjacent_find(
                active_feature_by_cell_gid.begin(),
                active_feature_by_cell_gid.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.first == rhs.first;
                }) != active_feature_by_cell_gid.end()) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: canonical active-cell "
                "ledger contains duplicate cell identities");
        }
        std::stable_sort(active_cell_gids_by_dof.begin(),
                         active_cell_gids_by_dof.end(),
                         by_first);
    }
    pending.active_cell_index = std::move(built_index);
    }
    const auto& active_cell_gids_by_dof =
        pending.active_cell_index->cell_gids_by_dof;
    const auto& active_feature_by_cell_gid =
        pending.active_cell_index->feature_by_cell_gid;
    const auto first_key_less = [](const auto& pair, GlobalIndex key) {
        return pair.first < key;
    };
    // Supports, feature sets and the covered cut cells are gathered in
    // vectors and made sorted and distinct, the content of the sets.
    std::vector<GlobalIndex> covered_cut_cell_gids;
    std::vector<GlobalIndex> support;
    std::vector<GlobalIndex> support_active_features;
    for (auto& patch : report->patches) {
        if (patch.active_feature_ids.empty() ||
            !std::is_sorted(
                patch.active_feature_ids.begin(),
                patch.active_feature_ids.end()) ||
            std::adjacent_find(
                patch.active_feature_ids.begin(),
                patch.active_feature_ids.end()) !=
                patch.active_feature_ids.end()) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: prolongation patch has "
                "non-canonical active-feature identities");
        }
        support.assign(patch.support_cell_gids.begin(),
                       patch.support_cell_gids.end());
        support.insert(support.end(),
                       patch.member_cell_gids.begin(),
                       patch.member_cell_gids.end());
        for (const auto slave : patch.slave_dofs) {
            const auto row_it = finalized_row_by_slave.find(slave);
            if (row_it == finalized_row_by_slave.end()) {
                throw std::logic_error(
                    "SmallCutAggregationConstraint: prolongation patch "
                    "references an unknown canonical slave");
            }
            const auto& row = *row_it->second;
            if (patch.kind ==
                    SmallCutAggregationPatchKind::Rootless &&
                !row.final_entries.empty()) {
                report->trace_bound_eligible = false;
            }
            for (const auto& entry : row.final_entries) {
                auto master_support = std::lower_bound(
                    active_cell_gids_by_dof.begin(),
                    active_cell_gids_by_dof.end(),
                    entry.master_dof,
                    first_key_less);
                if (master_support == active_cell_gids_by_dof.end() ||
                    master_support->first != entry.master_dof) {
                    report->trace_bound_eligible = false;
                    continue;
                }
                for (; master_support != active_cell_gids_by_dof.end() &&
                       master_support->first == entry.master_dof;
                     ++master_support) {
                    support.push_back(master_support->second);
                }
            }
        }
        std::sort(support.begin(), support.end());
        support.erase(std::unique(support.begin(), support.end()),
                      support.end());
        // Closed affine rows can replace a provisional root master with a
        // master supported by another active feature. Derive feature identity
        // from the complete final support instead of retaining the provisional
        // patch ledger.
        support_active_features.clear();
        for (const auto gid : support) {
            const auto active_cell = std::lower_bound(
                active_feature_by_cell_gid.begin(),
                active_feature_by_cell_gid.end(),
                gid,
                first_key_less);
            if (active_cell == active_feature_by_cell_gid.end() ||
                active_cell->first != gid) {
                report->trace_bound_eligible = false;
                continue;
            }
            support_active_features.push_back(
                active_cell->second);
        }
        std::sort(support_active_features.begin(),
                  support_active_features.end());
        support_active_features.erase(
            std::unique(support_active_features.begin(),
                        support_active_features.end()),
            support_active_features.end());
        patch.active_feature_ids.assign(
            support_active_features.begin(),
            support_active_features.end());
        patch.support_cell_gids.assign(
            support.begin(), support.end());
        covered_cut_cell_gids.insert(covered_cut_cell_gids.end(),
                                     patch.member_cell_gids.begin(),
                                     patch.member_cell_gids.end());
    }
    std::sort(covered_cut_cell_gids.begin(), covered_cut_cell_gids.end());
    covered_cut_cell_gids.erase(
        std::unique(covered_cut_cell_gids.begin(),
                    covered_cut_cell_gids.end()),
        covered_cut_cell_gids.end());
    for (const auto& cell : report->active_cells) {
        if (cell.kind != SmallCutAggregationActiveCellKind::Cut) {
            continue;
        }
        if (!std::binary_search(covered_cut_cell_gids.begin(),
                                covered_cut_cell_gids.end(),
                                cell.cell_gid)) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: canonical cut cell has no "
                "prolongation patch");
        }
    }

    std::uint64_t digest = 14695981039346656037ull;
    aggregationDigestMix(digest, 0x53564d5041474731ull);
    aggregationDigestMix(
        digest, static_cast<std::uint64_t>(report->field));
    aggregationDigestMix(
        digest, static_cast<std::uint64_t>(report->active_side));
    aggregationDigestMix(
        digest,
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(report->interface_marker)));
    aggregationDigestMix(digest, report->slave_all_cut ? 1u : 0u);
    aggregationDigestMix(digest, report->linear_extension ? 1u : 0u);
    aggregationDigestMix(digest, report->allow_unaggregated ? 1u : 0u);
    aggregationDigestMix(
        digest, report->trace_bound_eligible ? 1u : 0u);
    aggregationDigestMix(
        digest,
        aggregationDigestReal(
            static_cast<double>(
                report->guards
                    .maximum_reference_extrapolation_distance)));
    aggregationDigestMix(
        digest,
        static_cast<std::uint64_t>(
            report->guards.maximum_root_path_length));
    aggregationDigestMix(
        digest,
        aggregationDigestReal(
            static_cast<double>(
                report->guards.maximum_absolute_coefficient)));
    aggregationDigestMix(
        digest,
        aggregationDigestReal(
            static_cast<double>(
                report->guards.maximum_row_l1_norm)));
    aggregationDigestMix(
        digest, static_cast<std::uint64_t>(report->rows.size()));
    for (const auto& row : report->rows) {
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(row.candidate_dof));
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(row.component));
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(row.slave_dof));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(row.slave_owner_rank)));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(row.provisional_kind));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(row.final_kind));
        aggregationDigestMix(
            digest, row.preconstrained_at_apply ? 1u : 0u);
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(row.root_cell_gid));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(
                    row.root_cell_owner_rank)));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(
                    row.root_provider_rank)));
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(row.root_distance));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                row.provisional_entries.size()));
        for (const auto& entry : row.provisional_entries) {
            aggregationDigestMix(
                digest,
                static_cast<std::uint64_t>(entry.master_dof));
            aggregationDigestMix(
                digest, aggregationDigestReal(entry.weight));
        }
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                row.final_entries.size()));
        for (const auto& entry : row.final_entries) {
            aggregationDigestMix(
                digest,
                static_cast<std::uint64_t>(entry.master_dof));
            aggregationDigestMix(
                digest, aggregationDigestReal(entry.weight));
        }
        aggregationDigestMix(
            digest,
            aggregationDigestReal(
                static_cast<double>(row.final_inhomogeneity)));
    }
    aggregationDigestMix(
        digest,
        static_cast<std::uint64_t>(
            report->active_cells.size()));
    for (const auto& cell : report->active_cells) {
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(cell.cell_gid));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(cell.owner_rank)));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(
                    cell.retained_measure_provider_rank)));
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(cell.kind));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(cell.active_feature_id));
        aggregationDigestMix(
            digest,
            aggregationDigestReal(
                static_cast<double>(
                    cell.retained_physical_volume)));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                cell.retained_rule_stable_ids.size()));
        for (const auto stable_id :
             cell.retained_rule_stable_ids) {
            aggregationDigestMix(digest, stable_id);
        }
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                cell.field_dofs.size()));
        for (const auto dof : cell.field_dofs) {
            aggregationDigestMix(
                digest, static_cast<std::uint64_t>(dof));
        }
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                cell.active_face_neighbor_cell_gids.size()));
        for (const auto gid :
             cell.active_face_neighbor_cell_gids) {
            aggregationDigestMix(
                digest, static_cast<std::uint64_t>(gid));
        }
    }
    aggregationDigestMix(
        digest,
        static_cast<std::uint64_t>(report->patches.size()));
    for (const auto& patch : report->patches) {
        aggregationDigestMix(
            digest, static_cast<std::uint64_t>(patch.kind));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(patch.root_cell_gid));
        aggregationDigestMix(
            digest,
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(
                    patch.root_cell_owner_rank)));
        const auto mix_indices =
            [&](const std::vector<GlobalIndex>& indices) {
                aggregationDigestMix(
                    digest,
                    static_cast<std::uint64_t>(indices.size()));
                for (const auto index : indices) {
                    aggregationDigestMix(
                        digest,
                    static_cast<std::uint64_t>(index));
                }
            };
        mix_indices(patch.active_feature_ids);
        mix_indices(patch.member_cell_gids);
        mix_indices(patch.support_cell_gids);
        mix_indices(patch.slave_dofs);
    }
    report->canonical_content_digest = digest;
    } catch (...) {
        local_report_exception = std::current_exception();
    }
    coordinateDistributedLocalFailure(
        system,
        local_report_exception,
        "prolongation_report_construction");

#if FE_HAS_MPI
    if (world_size > 1) {
        const auto local_digest =
            report->canonical_content_digest;
        std::uint64_t minimum_digest = 0u;
        std::uint64_t maximum_digest = 0u;
        MPI_Allreduce(&local_digest,
                      &minimum_digest,
                      1,
                      MPI_UINT64_T,
                      MPI_MIN,
                      system.dofHandler().mpiComm());
        MPI_Allreduce(&local_digest,
                      &maximum_digest,
                      1,
                      MPI_UINT64_T,
                      MPI_MAX,
                      system.dofHandler().mpiComm());
        if (minimum_digest != maximum_digest) {
            throw std::runtime_error(
                "SmallCutAggregationConstraint: canonical finalized "
                "prolongation digest differs across communicator ranks");
        }
    }
#endif
    profile.mark(AggregationProfilePhase::Publication);
    profile.report(system,
                   system.fieldRecord(pending.field).name,
                   "finalize");
    return report;
}

std::shared_ptr<
    const SmallCutAggregationConstraint::LifecycleCheckpoint>
SmallCutAggregationConstraint::captureLifecycleCheckpoint() const
{
    auto checkpoint = std::shared_ptr<LifecycleCheckpoint>(
        new LifecycleCheckpoint());
    checkpoint->previous_canonical_slaves =
        previous_canonical_slaves_;
    checkpoint->completed_refresh_report =
        completed_refresh_report_;
    checkpoint->pending_prolongation = pending_prolongation_;
    checkpoint->successful_publication_ordinal =
        successful_publication_ordinal_;
    return checkpoint;
}

void SmallCutAggregationConstraint::restoreLifecycleCheckpoint(
    const LifecycleCheckpoint& checkpoint)
{
    // Finish every allocation-bearing copy before mutating live lifecycle
    // state. The following swaps/moves/scalar write then restore the report,
    // its comparison ledger, and its successful-publication ordinal as one
    // exception-safe local publication.
    auto restored_previous_canonical_slaves =
        checkpoint.previous_canonical_slaves;
    auto restored_completed_refresh_report =
        checkpoint.completed_refresh_report;
    auto restored_pending_prolongation =
        checkpoint.pending_prolongation;
    previous_canonical_slaves_.swap(
        restored_previous_canonical_slaves);
    completed_refresh_report_ =
        std::move(restored_completed_refresh_report);
    pending_prolongation_ =
        std::move(restored_pending_prolongation);
    successful_publication_ordinal_ =
        checkpoint.successful_publication_ordinal;
}

void SmallCutAggregationConstraint::publishReusedRefresh(
    const systems::FESystem& system,
    AffineConstraints& constraints,
    const assembly::CutIntegrationContext& cut_context,
    std::shared_ptr<const detail::SmallCutAggregationReuseRecord> record,
    std::optional<SmallCutAggregationRefreshReport>
        previous_completed_refresh_report,
    const detail::SmallCutAggregationRetainedMeasures& active_cell_measures,
    bool active_cell_measures_shared)
{
    // Mirrors the publication tail of apply(): every allocation, validation,
    // diagnostic and staged state is prepared first, one collective
    // consensus coordinates it, and the commit below is no-throw.
    std::optional<AffineConstraints> next_constraints;
    std::vector<GlobalIndex> next_previous_canonical_slaves;
    std::optional<SmallCutAggregationRefreshReport> next_completed_report;
    std::shared_ptr<const detail::SmallCutAggregationPendingProlongation>
        next_pending_prolongation;
    std::uint64_t next_successful_publication_ordinal =
        successful_publication_ordinal_;
    std::uint64_t next_feature_class_fingerprint = 0u;
    std::uint64_t next_slave_set_fingerprint = 0u;
    std::uint64_t next_geometry_fingerprint = 0u;
    std::exception_ptr local_publication_exception;
    try {
        const auto& active_cell_volumes = active_cell_measures.volumes;
        if (!record || !record->pending ||
            active_cell_volumes.size() !=
                record->pending->active_cells.size() ||
            active_cell_measures.stable_rule_ids.size() !=
                record->pending->active_cells.size()) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: reused refresh record is "
                "incomplete");
        }
        next_constraints.emplace(constraints);
        auto& staged_constraints = *next_constraints;
        for (const auto& line : record->added_lines) {
            staged_constraints.addConstraintLine(line);
        }

        // Retained volumes: per active cell, then per feature in ascending
        // physical cell ID (the component order of apply()), then the
        // rootless total in ascending feature ID.
        auto pending = std::make_shared<
            detail::SmallCutAggregationPendingProlongation>(*record->pending);
        std::map<GlobalIndex, long double> feature_volumes;
        for (std::size_t i = 0u; i < pending->active_cells.size(); ++i) {
            auto& cell = pending->active_cells[i];
            cell.retained_physical_volume = active_cell_volumes[i];
            cell.retained_rule_stable_ids =
                active_cell_measures.stable_rule_ids[i];
            feature_volumes[cell.active_feature_id] +=
                static_cast<long double>(active_cell_volumes[i]);
        }
        // As in apply(); allow_unaggregated is never reused, so it is false.
        pending->trace_bound_eligible =
            std::all_of(
                pending->active_cells.begin(),
                pending->active_cells.end(),
                [](const auto& cell) {
                    return !cell.retained_rule_stable_ids.empty() &&
                        std::none_of(
                            cell.retained_rule_stable_ids.begin(),
                            cell.retained_rule_stable_ids.end(),
                            [](std::uint64_t stable_id) {
                                return stable_id == 0u;
                            });
                }) &&
            std::none_of(
                pending->rows.begin(),
                pending->rows.end(),
                [](const auto& row) {
                    return row.provisional_kind ==
                        SmallCutAggregationProvisionalRowKind::
                            UnaggregatedFreeIdentity;
                });
        auto report = record->report;
        long double rootless_physical_volume = 0.0L;
        for (auto& feature : report.canonical_active_features) {
            const auto volume = feature_volumes.find(feature.stable_feature_id);
            feature.canonical_retained_physical_volume =
                volume == feature_volumes.end()
                    ? Real{0.0}
                    : static_cast<Real>(volume->second);
            if (!(feature.canonical_retained_physical_volume > Real{0.0}) ||
                !std::isfinite(feature.canonical_retained_physical_volume)) {
                throw std::runtime_error(
                    "SmallCutAggregationConstraint: diagnostic="
                    "invalid_active_feature_volume active component has no "
                    "finite positive retained physical volume");
            }
            if (feature.disposition ==
                SmallCutAggregationActiveFeatureDisposition::Rootless) {
                rootless_physical_volume += static_cast<long double>(
                    feature.canonical_retained_physical_volume);
            }
        }
        report.canonical_rootless_active_physical_volume =
            static_cast<Real>(rootless_physical_volume);
        if (!std::isfinite(report.canonical_rootless_active_physical_volume) ||
            report.canonical_rootless_active_physical_volume < Real{0.0}) {
            throw std::logic_error(
                "SmallCutAggregationConstraint: active feature totals are "
                "inconsistent");
        }

        // Detailed diagnostics, printed (at INFO) only when
        // FE_LOG_LEVEL=DEBUG, as in apply().
        const bool log_aggregation_diagnostics =
            Logger::instance().get_level() <= LogLevel::DEBUG;
        if (log_aggregation_diagnostics) {
            std::ostringstream oss;
            oss << record->diagnostic_head
                << report.canonical_rootless_active_physical_volume
                << record->diagnostic_middle
                << " pruned_volume_rules="
                << cut_context.generatedPrunedVolumeRuleCount()
                << " pruned_volume_measure="
                << cut_context.generatedPrunedVolumeMeasure();
            FE_LOG_INFO(oss.str());
        }

        next_previous_canonical_slaves = record->canonical_slaves;
        if (log_aggregation_diagnostics) {
            const auto& canonical_slaves = next_previous_canonical_slaves;
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
            churn << "SmallCutAggregationConstraint: diagnostic="
                     "small_cut_aggregation_churn"
                  << " field='" << record->field_name << "'"
                  << " canonical_slaves=" << canonical_slaves.size()
                  << " slaves=" << canonical_slaves.size()
                  << " entered=" << entered
                  << " left=" << left;
            FE_LOG_INFO(churn.str());
        }

        if (successful_publication_ordinal_ ==
            std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "SmallCutAggregationConstraint: successful publication "
                "ordinal is exhausted");
        }
        next_successful_publication_ordinal =
            successful_publication_ordinal_ + 1u;
        next_feature_class_fingerprint =
            report.canonical_feature_class_fingerprint;
        next_slave_set_fingerprint = report.canonical_slave_set_fingerprint;
        auto prepared_provenance = prepareRefreshProvenance(
            system,
            cut_context,
            interface_marker_,
            next_successful_publication_ordinal,
            staged_constraints);
        next_geometry_fingerprint =
            prepared_provenance.geometry_identity.canonical_fingerprint;
        report.geometry_identity =
            std::move(prepared_provenance.geometry_identity);
        report.local_lineage = prepared_provenance.local_lineage;
        report.canonical_topology_transition.reset();
        next_completed_report.emplace(std::move(report));
        if (previous_completed_refresh_report.has_value()) {
            next_completed_report->canonical_topology_transition =
                buildCanonicalTopologyTransitionReport(
                    *previous_completed_refresh_report,
                    *next_completed_report,
                    previous_canonical_slaves_,
                    next_previous_canonical_slaves);
        }

        pending->cut_context_content_revision = cut_context.contentRevision();
        pending->has_free_surface_snapshot_revision =
            cut_context.hasFreeSurfaceGeometrySnapshotForMarker(
                interface_marker_);
        pending->free_surface_snapshot_revision =
            pending->has_free_surface_snapshot_revision
                ? cut_context.freeSurfaceGeometrySnapshotRevisionForMarker(
                      interface_marker_)
                : 0u;
        pending->has_source_value_revision =
            cut_context.hasExpectedGeneratedSourceValueRevision(
                interface_marker_);
        pending->source_value_revision =
            pending->has_source_value_revision
                ? cut_context.expectedGeneratedSourceValueRevision(
                      interface_marker_)
                : 0u;
        next_pending_prolongation = std::move(pending);

        if (log_aggregation_diagnostics) {
            std::ostringstream decision;
            decision << "SmallCutAggregationConstraint: diagnostic="
                        "small_cut_aggregation_reuse field='"
                     << record->field_name << "' decision=reused"
                     << " retained_measures="
                     << (active_cell_measures_shared ? "shared" : "recomputed")
                     << " reused_lines=" << record->added_lines.size()
                     << " canonical_active_cells="
                     << next_pending_prolongation->active_cells.size();
            FE_LOG_INFO(decision.str());
        }
    } catch (...) {
        local_publication_exception = std::current_exception();
    }

    coordinateRefreshPublicationConsensus(
        system,
        local_publication_exception,
        next_feature_class_fingerprint,
        next_slave_set_fingerprint,
        next_geometry_fingerprint,
        next_successful_publication_ordinal);

    constraints = std::move(*next_constraints);
    next_completed_report->geometry_identity
        .communicator_fingerprint_consensus_validated =
            next_completed_report->geometry_identity.available;
    if (next_completed_report->canonical_topology_transition.has_value()) {
        auto& transition =
            *next_completed_report->canonical_topology_transition;
        transition.geometry_identity_before
            .communicator_fingerprint_consensus_validated =
                transition.geometry_identity_before.available;
        transition.geometry_identity_after
            .communicator_fingerprint_consensus_validated =
                transition.geometry_identity_after.available;
    }
    pending_prolongation_ = std::move(next_pending_prolongation);
    completed_refresh_report_ = std::move(next_completed_report);
    previous_canonical_slaves_.swap(next_previous_canonical_slaves);
    successful_publication_ordinal_ = next_successful_publication_ordinal;
    reuse_record_ = std::move(record);
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
    out.structural.active_configuration = true;
    // apply() reads no mesh field values: its geometry comes from the
    // installed cut integration context, and installing a new context
    // rebuilds the constraint state (setCutIntegrationContext() is followed
    // by rebuildConstraintState()).
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
