#include "LevelSet/LevelSetCurvatureProjection.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] std::string normalizedCurvatureToken(std::string_view value)
{
    std::string token(value);
    token.erase(token.begin(),
                std::find_if(token.begin(), token.end(), [](unsigned char c) {
                    return !std::isspace(c);
                }));
    token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char c) {
                    return !std::isspace(c);
                }).base(),
                token.end());
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) {
                                   return c == '_' || c == '-' ||
                                          std::isspace(c);
                               }),
                token.end());
    return token;
}

} // namespace

const char* levelSetCurvatureSmoothingModeName(
    LevelSetCurvatureSmoothingMode mode) noexcept
{
    switch (mode) {
        case LevelSetCurvatureSmoothingMode::LocalGraph:
            return "local_graph";
        case LevelSetCurvatureSmoothingMode::MassStiffnessOperator:
            return "mass_stiffness_operator";
    }
    return "unknown";
}

LevelSetCurvatureSmoothingMode parseLevelSetCurvatureSmoothingMode(
    std::string_view value)
{
    const auto token = normalizedCurvatureToken(value);
    if (token.empty() || token == "local" || token == "graph" ||
        token == "localgraph") {
        return LevelSetCurvatureSmoothingMode::LocalGraph;
    }
    if (token == "global" || token == "operator" ||
        token == "operatorprojection" || token == "massstiffness" ||
        token == "massstiffnessoperator" || token == "helmholtz" ||
        token == "helmholtzprojection" || token == "feprojection") {
        return LevelSetCurvatureSmoothingMode::MassStiffnessOperator;
    }
    throw std::invalid_argument(
        "level-set curvature projection smoothing mode '" +
        std::string(value) +
        "' must be local_graph or mass_stiffness_operator");
}

const char* levelSetCurvatureRecoveryModeName(
    LevelSetCurvatureRecoveryMode mode) noexcept
{
    switch (mode) {
        case LevelSetCurvatureRecoveryMode::LevelSetQuadratic:
            return "level_set_quadratic";
        case LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch:
            return "generated_interface_patch";
        case LevelSetCurvatureRecoveryMode::KinematicAreaGradient:
            return "kinematic_area_gradient";
    }
    return "unknown";
}

LevelSetCurvatureRecoveryMode parseLevelSetCurvatureRecoveryMode(
    std::string_view value)
{
    const auto token = normalizedCurvatureToken(value);
    if (token.empty() || token == "levelset" ||
        token == "levelsetquadratic" || token == "quadratic") {
        return LevelSetCurvatureRecoveryMode::LevelSetQuadratic;
    }
    if (token == "interface" || token == "interfacepatch" ||
        token == "generatedinterface" ||
        token == "generatedinterfacepatch") {
        return LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch;
    }
    if (token == "areagradient" || token == "kinematicarea" ||
        token == "kinematicareagradient" ||
        token == "discreteareagradient") {
        return LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    }
    throw std::invalid_argument(
        "level-set curvature recovery mode '" + std::string(value) +
        "' must be level_set_quadratic, generated_interface_patch, or kinematic_area_gradient");
}

namespace {

struct KinematicProjectionCollectiveContext {
    int rank{0};
    int size{1};
    bool active{false};
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
};

struct PackedKinematicCell {
    std::int64_t global_cell{-1};
    std::int32_t element_type{0};
    std::int32_t corner_count{0};
    std::array<std::int64_t, 4> global_nodes{{-1, -1, -1, -1}};
    std::array<std::array<Real, 3>, 4> coordinates{};
    std::array<Real, 4> level_set_values{};
};

struct PackedKinematicBoundaryFace {
    std::int64_t global_face{-1};
    std::int64_t global_parent_cell{-1};
    std::uint32_t local_face{INVALID_LOCAL_INDEX};
    std::int32_t marker{-1};
};

static_assert(std::is_trivially_copyable_v<PackedKinematicCell>);
static_assert(std::is_trivially_copyable_v<PackedKinematicBoundaryFace>);

struct ReplicatedKinematicCell {
    GlobalIndex global_id{-1};
    ElementType type{ElementType::Triangle3};
    std::array<GlobalIndex, 4> nodes{{-1, -1, -1, -1}};
    std::size_t corner_count{0u};
};

struct ReplicatedKinematicBoundaryFace {
    GlobalIndex global_id{-1};
    GlobalIndex parent_cell{-1};
    LocalIndex local_face{INVALID_LOCAL_INDEX};
    int marker{-1};
};

class ReplicatedKinematicMeshAccess final
    : public assembly::IMeshAccess {
public:
    ReplicatedKinematicMeshAccess(
        int dimension,
        std::vector<std::array<Real, 3>> coordinates,
        std::vector<ReplicatedKinematicCell> cells,
        std::vector<ReplicatedKinematicBoundaryFace> boundary_faces)
        : dimension_(dimension),
          coordinates_(std::move(coordinates)),
          cells_(std::move(cells)),
          boundary_faces_(std::move(boundary_faces))
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] GlobalIndex numVertices() const override
    {
        return static_cast<GlobalIndex>(coordinates_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedVertices() const override
    {
        return numVertices();
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return static_cast<GlobalIndex>(boundary_faces_.size());
    }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return dimension_; }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] GlobalIndex getCellGlobalId(
        GlobalIndex cell) const override
    {
        return cells_.at(static_cast<std::size_t>(cell)).global_id;
    }
    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(
        GlobalIndex face) const override
    {
        return boundary_faces_.at(static_cast<std::size_t>(face)).global_id;
    }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return cell >= 0 && cell < numCells();
    }
    [[nodiscard]] ElementType getCellType(GlobalIndex cell) const override
    {
        return cells_.at(static_cast<std::size_t>(cell)).type;
    }
    void getCellNodes(GlobalIndex cell,
                      std::vector<GlobalIndex>& nodes) const override
    {
        const auto& record = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(
            record.nodes.begin(),
            record.nodes.begin() +
                static_cast<std::ptrdiff_t>(record.corner_count));
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(
        GlobalIndex node) const override
    {
        return coordinates_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        GlobalIndex cell,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell, nodes);
        coordinates.clear();
        coordinates.reserve(nodes.size());
        for (const auto node : nodes) {
            coordinates.push_back(getNodeCoordinates(node));
        }
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(
        GlobalIndex face, GlobalIndex cell) const override
    {
        const auto& record =
            boundary_faces_.at(static_cast<std::size_t>(face));
        return record.parent_cell == cell ? record.local_face
                                          : INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        GlobalIndex face) const override
    {
        return boundary_faces_.at(static_cast<std::size_t>(face)).marker;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex) const override
    {
        return {-1, -1};
    }
    void forEachCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }
    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(GlobalIndex, GlobalIndex)> callback)
        const override
    {
        for (std::size_t face = 0; face < boundary_faces_.size(); ++face) {
            const auto& record = boundary_faces_[face];
            if (marker < 0 || marker == record.marker) {
                callback(static_cast<GlobalIndex>(face),
                         record.parent_cell);
            }
        }
    }
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>)
        const override
    {
    }

private:
    int dimension_{0};
    std::vector<std::array<Real, 3>> coordinates_{};
    std::vector<ReplicatedKinematicCell> cells_{};
    std::vector<ReplicatedKinematicBoundaryFace> boundary_faces_{};
};

struct ReplicatedKinematicProjectionInput {
    bool success{false};
    std::vector<Real> level_set_values{};
    std::vector<std::size_t> local_vertex_to_global_dof{};
    std::vector<ReplicatedKinematicCell> cells{};
    std::vector<std::array<Real, 3>> coordinates{};
    std::vector<ReplicatedKinematicBoundaryFace> boundary_faces{};
    std::map<GlobalIndex, GlobalIndex> global_cell_to_replicated_cell{};
    std::string diagnostic{};
};

void mixKinematicProjectionSignature(
    std::uint64_t& signature, std::uint64_t value) noexcept
{
    signature ^= value + 0x9e3779b97f4a7c15ull +
                 (signature << 6u) + (signature >> 2u);
}

void mixKinematicProjectionReal(
    std::uint64_t& signature, Real value) noexcept
{
    std::uint64_t bits{0u};
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    mixKinematicProjectionSignature(signature, bits);
}

[[nodiscard]] std::uint64_t kinematicProjectionOptionsSignature(
    const LevelSetCurvatureProjectionOptions& options) noexcept
{
    std::uint64_t signature{1469598103934665603ull};
    mixKinematicProjectionReal(signature, options.isovalue);
    mixKinematicProjectionReal(signature, options.gradient_tolerance);
    mixKinematicProjectionReal(signature,
                               options.normal_equation_tolerance);
    mixKinematicProjectionReal(signature,
                               options.max_normalized_fit_residual);
    mixKinematicProjectionSignature(
        signature, static_cast<std::uint64_t>(options.max_neighbor_rings));
    mixKinematicProjectionSignature(
        signature,
        static_cast<std::uint64_t>(options.max_neighbor_fallback_vertices));
    mixKinematicProjectionSignature(
        signature,
        static_cast<std::uint64_t>(options.max_zero_fallback_vertices));
    mixKinematicProjectionReal(signature,
                               options.supplemental_sample_weight);
    mixKinematicProjectionSignature(
        signature, static_cast<std::uint64_t>(options.recovery_mode));
    mixKinematicProjectionReal(
        signature, options.kinematic_area_gradient_filter_coefficient);
    mixKinematicProjectionSignature(
        signature,
        options.kinematic_area_gradient_negative_liquid_side ? 1u : 0u);
    mixKinematicProjectionSignature(
        signature,
        static_cast<std::uint64_t>(
            options.kinematic_area_gradient_young_walls.size()));
    for (const auto& wall :
         options.kinematic_area_gradient_young_walls) {
        mixKinematicProjectionSignature(
            signature,
            static_cast<std::uint64_t>(wall.boundary_marker));
        mixKinematicProjectionReal(
            signature, wall.equilibrium_contact_angle_radians);
    }
    mixKinematicProjectionReal(signature, options.narrow_band_width);
    mixKinematicProjectionSignature(
        signature,
        static_cast<std::uint64_t>(options.smoothing_iterations));
    mixKinematicProjectionReal(signature, options.smoothing_relaxation);
    mixKinematicProjectionSignature(
        signature, static_cast<std::uint64_t>(options.smoothing_mode));
    return signature;
}

#if FE_HAS_MPI
[[nodiscard]] MPI_Datatype kinematicMpiSigned64Type() noexcept
{
#ifdef MPI_INT64_T
    return MPI_INT64_T;
#else
    if constexpr (std::is_same_v<std::int64_t, long>) {
        return MPI_LONG;
    }
    return MPI_LONG_LONG;
#endif
}

[[nodiscard]] MPI_Datatype kinematicMpiUnsigned64Type() noexcept
{
#ifdef MPI_UINT64_T
    return MPI_UINT64_T;
#else
    if constexpr (std::is_same_v<std::uint64_t, unsigned long>) {
        return MPI_UNSIGNED_LONG;
    }
    return MPI_UNSIGNED_LONG_LONG;
#endif
}
#endif

[[nodiscard]] bool nearlyEqualKinematicInput(Real lhs, Real rhs) noexcept
{
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const Real scale =
        std::max({Real{1.0}, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <=
           Real{256.0} * std::numeric_limits<Real>::epsilon() * scale;
}

[[nodiscard]] bool synchronizeKinematicProjectionFailure(
    const KinematicProjectionCollectiveContext& context,
    bool local_success,
    const std::string& local_diagnostic,
    std::string& collective_diagnostic)
{
#if FE_HAS_MPI
    if (context.active) {
        const int local_failed_rank =
            local_success ? context.size : context.rank;
        int first_failed_rank = context.size;
        MPI_Allreduce(&local_failed_rank, &first_failed_rank, 1,
                      MPI_INT, MPI_MIN, context.communicator);
        if (first_failed_rank < context.size) {
            constexpr int maximum_diagnostic_bytes = 4096;
            int length = context.rank == first_failed_rank
                ? static_cast<int>(std::min<std::size_t>(
                      local_diagnostic.size(),
                      static_cast<std::size_t>(
                          maximum_diagnostic_bytes - 1)))
                : 0;
            MPI_Bcast(&length, 1, MPI_INT, first_failed_rank,
                      context.communicator);
            collective_diagnostic.assign(
                static_cast<std::size_t>(length), '\0');
            if (context.rank == first_failed_rank && length > 0) {
                std::copy_n(local_diagnostic.begin(), length,
                            collective_diagnostic.begin());
            }
            if (length > 0) {
                MPI_Bcast(collective_diagnostic.data(), length,
                          MPI_CHAR, first_failed_rank,
                          context.communicator);
            }
            collective_diagnostic =
                "collective kinematic-area-gradient input failed on rank " +
                std::to_string(first_failed_rank) + ": " +
                collective_diagnostic;
            return false;
        }
    }
#else
    (void)context;
#endif
    if (!local_success) {
        collective_diagnostic = local_diagnostic;
        return false;
    }
    return true;
}

template <typename Packed>
[[nodiscard]] bool allGatherKinematicRecords(
    const KinematicProjectionCollectiveContext& context,
    std::span<const Packed> local,
    std::vector<Packed>& gathered,
    std::string& diagnostic)
{
    static_assert(std::is_trivially_copyable_v<Packed>);
#if FE_HAS_MPI
    if (context.active) {
        const auto local_bytes_size = local.size_bytes();
        const bool local_payload_valid =
            local_bytes_size <= static_cast<std::size_t>(
                                    std::numeric_limits<int>::max());
        const std::string local_payload_diagnostic = local_payload_valid
            ? std::string{}
            : "kinematic-area-gradient local gather payload exceeds the MPI count range";
        if (!synchronizeKinematicProjectionFailure(
                context, local_payload_valid,
                local_payload_diagnostic, diagnostic)) {
            return false;
        }
        const int local_bytes = static_cast<int>(local_bytes_size);
        std::vector<int> counts(static_cast<std::size_t>(context.size), 0);
        MPI_Allgather(&local_bytes, 1, MPI_INT,
                      counts.data(), 1, MPI_INT,
                      context.communicator);
        std::vector<int> displacements(
            static_cast<std::size_t>(context.size), 0);
        std::size_t total_bytes{0u};
        for (int rank = 0; rank < context.size; ++rank) {
            if (counts[static_cast<std::size_t>(rank)] < 0 ||
                total_bytes >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max())) {
                diagnostic =
                    "kinematic-area-gradient collective gather payload exceeds the MPI displacement range";
                return false;
            }
            displacements[static_cast<std::size_t>(rank)] =
                static_cast<int>(total_bytes);
            total_bytes += static_cast<std::size_t>(
                counts[static_cast<std::size_t>(rank)]);
        }
        if (total_bytes % sizeof(Packed) != 0u ||
            total_bytes > static_cast<std::size_t>(
                              std::numeric_limits<int>::max())) {
            diagnostic =
                "kinematic-area-gradient collective gather payload has an invalid size";
            return false;
        }
        gathered.resize(total_bytes / sizeof(Packed));
        const int status = MPI_Allgatherv(
            local.empty() ? nullptr : local.data(),
            local_bytes,
            MPI_BYTE,
            gathered.empty() ? nullptr : gathered.data(),
            counts.data(),
            displacements.data(),
            MPI_BYTE,
            context.communicator);
        if (status != MPI_SUCCESS) {
            diagnostic =
                "kinematic-area-gradient collective gather failed";
            return false;
        }
        return true;
    }
#else
    (void)context;
#endif
    gathered.assign(local.begin(), local.end());
    return true;
}

[[nodiscard]] ReplicatedKinematicProjectionInput
buildReplicatedKinematicProjectionInput(
    const systems::FESystem& system,
    FieldId level_set_field,
    std::span<const Real> local_level_set_values,
    KinematicProjectionCollectiveContext& context)
{
    ReplicatedKinematicProjectionInput result;
    const auto& mesh = system.meshAccess();
    context.rank = mesh.parallelRank();
    context.size = mesh.parallelSize();
    if (context.rank < 0 || context.size < 1 ||
        context.rank >= context.size) {
        result.diagnostic =
            "kinematic-area-gradient collective projection received invalid mesh rank metadata";
        return result;
    }
    const auto& dofs = system.fieldDofHandler(level_set_field);
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (context.size > 1) {
        if (initialized == 0 || finalized != 0 ||
            dofs.mpiComm() == MPI_COMM_NULL) {
            result.diagnostic =
                "kinematic-area-gradient collective projection requires an active field communicator";
            return result;
        }
        int communicator_rank = 0;
        int communicator_size = 1;
        MPI_Comm_rank(dofs.mpiComm(), &communicator_rank);
        MPI_Comm_size(dofs.mpiComm(), &communicator_size);
        if (communicator_rank != context.rank ||
            communicator_size != context.size) {
            result.diagnostic =
                "kinematic-area-gradient collective projection mesh and field communicators disagree";
            return result;
        }
        context.active = true;
        context.communicator = dofs.mpiComm();
    }
#else
    if (context.size > 1) {
        result.diagnostic =
            "kinematic-area-gradient collective projection cannot use a multi-rank mesh without MPI support";
        return result;
    }
#endif

    bool local_success = true;
    std::string local_diagnostic;
    std::vector<PackedKinematicCell> local_cells;
    std::vector<PackedKinematicBoundaryFace> local_boundary_faces;
    GlobalIndex global_dof_count{0};
    std::uint64_t dof_layout_revision{0u};
    try {
        const auto& field = system.fieldRecord(level_set_field);
        if (!field.space || field.components != 1 ||
            field.space->space_type() != spaces::SpaceType::H1 ||
            field.space->field_type() != FieldType::Scalar ||
            field.space->continuity() != Continuity::C0 ||
            field.space->value_dimension() != 1 ||
            field.space->is_variable_order() ||
            field.space->polynomial_order() != 1) {
            throw std::invalid_argument(
                "kinematic-area-gradient collective projection requires a fixed-order scalar P1 H1 field");
        }
        if (!mesh.globalEntityIdsAvailable()) {
            throw std::invalid_argument(
                "kinematic-area-gradient collective projection requires global cell and boundary-face identities");
        }
        if (mesh.numVertices() < 0 ||
            local_level_set_values.size() !=
                static_cast<std::size_t>(mesh.numVertices())) {
            throw std::invalid_argument(
                "kinematic-area-gradient collective projection requires one local value per visible mesh vertex");
        }
        global_dof_count = dofs.getNumDofs();
        if (global_dof_count <= 0) {
            throw std::invalid_argument(
                "kinematic-area-gradient collective projection requires a nonempty global field layout");
        }
        dof_layout_revision = dofs.dofLayoutRevision();
        const auto* entity_map = dofs.getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::invalid_argument(
                "kinematic-area-gradient collective projection requires vertex-to-DOF metadata");
        }
        result.local_vertex_to_global_dof.resize(
            static_cast<std::size_t>(mesh.numVertices()));
        for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
            const auto vertex_dofs = entity_map->getVertexDofs(vertex);
            if (vertex_dofs.size() != 1u || vertex_dofs.front() < 0 ||
                vertex_dofs.front() >= global_dof_count) {
                throw std::invalid_argument(
                    "kinematic-area-gradient collective projection requires one valid scalar field DOF per visible vertex");
            }
            result.local_vertex_to_global_dof[
                static_cast<std::size_t>(vertex)] =
                static_cast<std::size_t>(vertex_dofs.front());
        }

        mesh.forEachOwnedCell([&](GlobalIndex cell) {
            const auto type = mesh.getCellType(cell);
            const int dimension = mesh.dimension();
            const bool supported =
                (dimension == 2 && type == ElementType::Triangle3) ||
                (dimension == 3 && type == ElementType::Tetra4);
            if (!supported || mesh.getCellGeometryOrder(cell) != 1 ||
                field.space->polynomial_order(cell) != 1) {
                throw std::invalid_argument(
                    "kinematic-area-gradient collective projection requires affine P1 Triangle3 or Tetra4 cells");
            }
            std::vector<GlobalIndex> nodes;
            mesh.getCellNodes(cell, nodes);
            const auto corner_count =
                static_cast<std::size_t>(dimension + 1);
            if (nodes.size() != corner_count) {
                throw std::invalid_argument(
                    "kinematic-area-gradient collective projection found incomplete simplex connectivity");
            }
            PackedKinematicCell packed;
            packed.global_cell = static_cast<std::int64_t>(
                mesh.getCellGlobalId(cell));
            packed.element_type = static_cast<std::int32_t>(type);
            packed.corner_count = static_cast<std::int32_t>(corner_count);
            for (std::size_t corner = 0; corner < corner_count; ++corner) {
                const auto node = nodes[corner];
                if (node < 0 || node >= mesh.numVertices()) {
                    throw std::invalid_argument(
                        "kinematic-area-gradient collective projection found a cell vertex outside the local mesh layout");
                }
                const auto global_node =
                    result.local_vertex_to_global_dof[
                        static_cast<std::size_t>(node)];
                packed.global_nodes[corner] =
                    static_cast<std::int64_t>(global_node);
                packed.coordinates[corner] =
                    mesh.getNodeCoordinates(node);
                packed.level_set_values[corner] =
                    local_level_set_values[
                        static_cast<std::size_t>(node)];
                if (!std::isfinite(packed.coordinates[corner][0]) ||
                    !std::isfinite(packed.coordinates[corner][1]) ||
                    !std::isfinite(packed.coordinates[corner][2]) ||
                    !std::isfinite(packed.level_set_values[corner])) {
                    throw std::invalid_argument(
                        "kinematic-area-gradient collective projection found nonfinite cell data");
                }
            }
            local_cells.push_back(packed);
        });

        mesh.forEachBoundaryFace(
            -1,
            [&](GlobalIndex face, GlobalIndex parent_cell) {
                if (!mesh.isOwnedCell(parent_cell) ||
                    mesh.getCellOwnerRank(parent_cell) != context.rank ||
                    mesh.getBoundaryFaceOwnerRank(face, parent_cell) !=
                        context.rank) {
                    throw std::invalid_argument(
                        "kinematic-area-gradient collective projection requires each emitted boundary face to be owned with its parent cell");
                }
                PackedKinematicBoundaryFace packed;
                packed.global_face = static_cast<std::int64_t>(
                    mesh.getBoundaryFaceGlobalId(face));
                packed.global_parent_cell = static_cast<std::int64_t>(
                    mesh.getCellGlobalId(parent_cell));
                packed.local_face =
                    mesh.getLocalFaceIndex(face, parent_cell);
                packed.marker = static_cast<std::int32_t>(
                    mesh.getBoundaryFaceMarker(face));
                if (packed.local_face == INVALID_LOCAL_INDEX) {
                    throw std::invalid_argument(
                        "kinematic-area-gradient collective projection found an invalid boundary local-face index");
                }
                local_boundary_faces.push_back(packed);
            });
    } catch (const std::exception& exception) {
        local_success = false;
        local_diagnostic = exception.what();
    }
    if (!synchronizeKinematicProjectionFailure(
            context, local_success, local_diagnostic,
            result.diagnostic)) {
        return result;
    }

    int minimum_dimension = mesh.dimension();
    int maximum_dimension = mesh.dimension();
    GlobalIndex minimum_dof_count = global_dof_count;
    GlobalIndex maximum_dof_count = global_dof_count;
    std::uint64_t minimum_layout_revision = dof_layout_revision;
    std::uint64_t maximum_layout_revision = dof_layout_revision;
#if FE_HAS_MPI
    if (context.active) {
        MPI_Allreduce(MPI_IN_PLACE, &minimum_dimension, 1,
                      MPI_INT, MPI_MIN, context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_dimension, 1,
                      MPI_INT, MPI_MAX, context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_dof_count, 1,
                      kinematicMpiSigned64Type(), MPI_MIN,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_dof_count, 1,
                      kinematicMpiSigned64Type(), MPI_MAX,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_layout_revision, 1,
                      kinematicMpiUnsigned64Type(), MPI_MIN,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_layout_revision, 1,
                      kinematicMpiUnsigned64Type(), MPI_MAX,
                      context.communicator);
    }
#endif
    if (minimum_dimension != maximum_dimension ||
        minimum_dof_count != maximum_dof_count ||
        minimum_layout_revision != maximum_layout_revision) {
        result.diagnostic =
            "kinematic-area-gradient collective projection requires identical dimension, global field size, and field-layout revision on every rank";
        return result;
    }

    std::vector<PackedKinematicCell> gathered_cells;
    if (!allGatherKinematicRecords(
            context,
            std::span<const PackedKinematicCell>(
                local_cells.data(), local_cells.size()),
            gathered_cells,
            result.diagnostic)) {
        return result;
    }
    std::vector<PackedKinematicBoundaryFace> gathered_boundary_faces;
    if (!allGatherKinematicRecords(
            context,
            std::span<const PackedKinematicBoundaryFace>(
                local_boundary_faces.data(),
                local_boundary_faces.size()),
            gathered_boundary_faces,
            result.diagnostic)) {
        return result;
    }

    try {
        std::sort(gathered_cells.begin(), gathered_cells.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.global_cell < rhs.global_cell;
                  });
        if (gathered_cells.empty()) {
            throw std::invalid_argument(
                "kinematic-area-gradient collective projection gathered no owned cells");
        }
        result.level_set_values.assign(
            static_cast<std::size_t>(minimum_dof_count), Real{0.0});
        result.coordinates.assign(
            static_cast<std::size_t>(minimum_dof_count),
            std::array<Real, 3>{});
        std::vector<unsigned char> node_seen(
            static_cast<std::size_t>(minimum_dof_count), 0u);
        result.cells.reserve(gathered_cells.size());
        for (std::size_t cell_index = 0;
             cell_index < gathered_cells.size();
             ++cell_index) {
            const auto& packed = gathered_cells[cell_index];
            if (cell_index > 0u &&
                packed.global_cell ==
                    gathered_cells[cell_index - 1u].global_cell) {
                throw std::invalid_argument(
                    "kinematic-area-gradient collective projection gathered duplicate owned cell identities");
            }
            const auto type = static_cast<ElementType>(packed.element_type);
            const std::size_t expected_corners =
                minimum_dimension == 2 ? 3u : 4u;
            if (packed.global_cell < 0 ||
                packed.corner_count !=
                    static_cast<std::int32_t>(expected_corners) ||
                (minimum_dimension == 2 &&
                 type != ElementType::Triangle3) ||
                (minimum_dimension == 3 &&
                 type != ElementType::Tetra4)) {
                throw std::invalid_argument(
                    "kinematic-area-gradient collective projection gathered an invalid simplex cell record");
            }
            ReplicatedKinematicCell cell;
            cell.global_id =
                static_cast<GlobalIndex>(packed.global_cell);
            cell.type = type;
            cell.corner_count = expected_corners;
            for (std::size_t corner = 0;
                 corner < expected_corners;
                 ++corner) {
                const auto global_node = packed.global_nodes[corner];
                if (global_node < 0 ||
                    global_node >= minimum_dof_count) {
                    throw std::invalid_argument(
                        "kinematic-area-gradient collective projection gathered a node outside the global field layout");
                }
                const auto node = static_cast<std::size_t>(global_node);
                cell.nodes[corner] =
                    static_cast<GlobalIndex>(global_node);
                if (node_seen[node] == 0u) {
                    result.coordinates[node] =
                        packed.coordinates[corner];
                    result.level_set_values[node] =
                        packed.level_set_values[corner];
                    node_seen[node] = 1u;
                } else {
                    for (std::size_t component = 0;
                         component < 3u;
                         ++component) {
                        if (!nearlyEqualKinematicInput(
                                result.coordinates[node][component],
                                packed.coordinates[corner][component])) {
                            throw std::invalid_argument(
                                "kinematic-area-gradient collective projection found inconsistent shared-node coordinates");
                        }
                    }
                    if (!nearlyEqualKinematicInput(
                            result.level_set_values[node],
                            packed.level_set_values[corner])) {
                        throw std::invalid_argument(
                            "kinematic-area-gradient collective projection found inconsistent shared-node values");
                    }
                }
            }
            result.global_cell_to_replicated_cell.emplace(
                cell.global_id,
                static_cast<GlobalIndex>(result.cells.size()));
            result.cells.push_back(cell);
        }
        if (std::find(node_seen.begin(), node_seen.end(),
                      static_cast<unsigned char>(0u)) != node_seen.end()) {
            throw std::invalid_argument(
                "kinematic-area-gradient collective projection found a global field DOF outside every owned cell");
        }

        std::sort(gathered_boundary_faces.begin(),
                  gathered_boundary_faces.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.global_face < rhs.global_face;
                  });
        result.boundary_faces.reserve(gathered_boundary_faces.size());
        for (std::size_t face_index = 0;
             face_index < gathered_boundary_faces.size();
             ++face_index) {
            const auto& packed = gathered_boundary_faces[face_index];
            if (face_index > 0u &&
                packed.global_face ==
                    gathered_boundary_faces[face_index - 1u].global_face) {
                throw std::invalid_argument(
                    "kinematic-area-gradient collective projection gathered duplicate owned boundary-face identities");
            }
            const auto parent = result.global_cell_to_replicated_cell.find(
                static_cast<GlobalIndex>(packed.global_parent_cell));
            if (packed.global_face < 0 ||
                parent == result.global_cell_to_replicated_cell.end() ||
                packed.local_face == INVALID_LOCAL_INDEX) {
                throw std::invalid_argument(
                    "kinematic-area-gradient collective projection gathered an invalid boundary-face record");
            }
            result.boundary_faces.push_back({
                static_cast<GlobalIndex>(packed.global_face),
                parent->second,
                static_cast<LocalIndex>(packed.local_face),
                static_cast<int>(packed.marker)});
        }
    } catch (const std::exception& exception) {
        local_success = false;
        local_diagnostic = exception.what();
    }
    if (!synchronizeKinematicProjectionFailure(
            context, local_success, local_diagnostic,
            result.diagnostic)) {
        return result;
    }
    result.success = true;
    return result;
}

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const std::array<Real, 3>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

[[nodiscard]] std::size_t fitSize(int dim)
{
    return dim == 2 ? 5u : 9u;
}

struct FitObservation {
    std::array<Real, 9> row{};
    Real rhs{0.0};
    Real weight{1.0};
};

struct RawFitObservation {
    std::array<Real, 3> displacement{};
    Real rhs{0.0};
    Real relative_weight{1.0};
};

struct FitResidualMetrics {
    Real rms{0.0};
    Real normalized{0.0};
};

[[nodiscard]] std::array<Real, 9> quadraticRow(
    const std::array<Real, 3>& dx,
    int dim) noexcept;

[[nodiscard]] FitResidualMetrics computeFitResidualMetrics(
    std::span<const FitObservation> observations,
    const std::array<Real, 9>& coefficients,
    std::size_t n,
    Real scale_floor)
{
    Real weighted_residual2 = Real{0.0};
    Real weighted_rhs2 = Real{0.0};
    Real weight_sum = Real{0.0};
    for (const auto& obs : observations) {
        Real predicted = Real{0.0};
        for (std::size_t i = 0; i < n; ++i) {
            predicted += obs.row[i] * coefficients[i];
        }
        const Real residual = predicted - obs.rhs;
        weighted_residual2 += obs.weight * residual * residual;
        weighted_rhs2 += obs.weight * obs.rhs * obs.rhs;
        weight_sum += obs.weight;
    }
    if (!(weight_sum > Real{0.0}) || !std::isfinite(weight_sum)) {
        return {};
    }

    const Real rms = std::sqrt(weighted_residual2 / weight_sum);
    const Real rhs_rms = std::sqrt(weighted_rhs2 / weight_sum);
    const Real normalized =
        rms / std::max(std::max(rhs_rms, scale_floor), Real{1.0e-300});
    if (!std::isfinite(rms) || !std::isfinite(normalized)) {
        return {std::numeric_limits<Real>::infinity(),
                std::numeric_limits<Real>::infinity()};
    }
    return {rms, normalized};
}

/**
 * Solve the weighted rectangular fit directly, without forming A^T A.
 *
 * Column-pivoted Householder QR avoids squaring the stencil condition number.
 * The rank threshold is relative to the largest initial weighted column norm;
 * callers first nondimensionalize the coordinates, so this test is invariant
 * under a change of mesh length units.
 */
[[nodiscard]] bool solveWeightedLeastSquares(
    std::span<const FitObservation> observations,
    std::size_t n,
    Real relative_rank_tolerance,
    std::array<Real, 9>& x)
{
    const std::size_t m = observations.size();
    if (m < n || n == 0u) {
        return false;
    }

    std::vector<std::array<Real, 9>> a(m);
    std::vector<Real> b(m, Real{0.0});
    std::array<std::size_t, 9> permutation{};
    std::iota(permutation.begin(), permutation.end(), std::size_t{0u});

    for (std::size_t r = 0; r < m; ++r) {
        const auto& observation = observations[r];
        if (!(observation.weight > Real{0.0}) ||
            !std::isfinite(observation.weight) ||
            !std::isfinite(observation.rhs)) {
            return false;
        }
        const Real sqrt_weight = std::sqrt(observation.weight);
        b[r] = sqrt_weight * observation.rhs;
        for (std::size_t c = 0; c < n; ++c) {
            a[r][c] = sqrt_weight * observation.row[c];
            if (!std::isfinite(a[r][c])) {
                return false;
            }
        }
    }

    Real reference_column_norm = Real{0.0};
    for (std::size_t c = 0; c < n; ++c) {
        Real norm2 = Real{0.0};
        for (std::size_t r = 0; r < m; ++r) {
            norm2 += a[r][c] * a[r][c];
        }
        reference_column_norm =
            std::max(reference_column_norm, std::sqrt(norm2));
    }
    if (!(reference_column_norm > Real{0.0}) ||
        !std::isfinite(reference_column_norm)) {
        return false;
    }
    const Real rank_threshold =
        relative_rank_tolerance * reference_column_norm;

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot_column = k;
        Real pivot_norm2 = Real{-1.0};
        for (std::size_t c = k; c < n; ++c) {
            Real column_norm2 = Real{0.0};
            for (std::size_t r = k; r < m; ++r) {
                column_norm2 += a[r][c] * a[r][c];
            }
            if (column_norm2 > pivot_norm2) {
                pivot_norm2 = column_norm2;
                pivot_column = c;
            }
        }
        const Real pivot_norm = std::sqrt(std::max(pivot_norm2, Real{0.0}));
        if (!(pivot_norm > rank_threshold) || !std::isfinite(pivot_norm)) {
            return false;
        }
        if (pivot_column != k) {
            for (std::size_t r = 0; r < m; ++r) {
                std::swap(a[r][k], a[r][pivot_column]);
            }
            std::swap(permutation[k], permutation[pivot_column]);
        }

        const Real alpha = -std::copysign(pivot_norm, a[k][k]);
        std::vector<Real> reflector(m - k, Real{0.0});
        for (std::size_t r = k; r < m; ++r) {
            reflector[r - k] = a[r][k];
        }
        reflector[0] -= alpha;
        Real reflector_norm2 = Real{0.0};
        for (const Real value : reflector) {
            reflector_norm2 += value * value;
        }
        if (!(reflector_norm2 > Real{0.0}) ||
            !std::isfinite(reflector_norm2)) {
            return false;
        }

        for (std::size_t c = k; c < n; ++c) {
            Real projection = Real{0.0};
            for (std::size_t r = k; r < m; ++r) {
                projection += reflector[r - k] * a[r][c];
            }
            const Real factor = Real{2.0} * projection / reflector_norm2;
            for (std::size_t r = k; r < m; ++r) {
                a[r][c] -= factor * reflector[r - k];
            }
        }
        Real rhs_projection = Real{0.0};
        for (std::size_t r = k; r < m; ++r) {
            rhs_projection += reflector[r - k] * b[r];
        }
        const Real rhs_factor =
            Real{2.0} * rhs_projection / reflector_norm2;
        for (std::size_t r = k; r < m; ++r) {
            b[r] -= rhs_factor * reflector[r - k];
        }

        a[k][k] = alpha;
        for (std::size_t r = k + 1u; r < m; ++r) {
            a[r][k] = Real{0.0};
        }
    }

    std::array<Real, 9> pivoted_solution{};
    for (std::size_t reverse = n; reverse > 0u; --reverse) {
        const std::size_t row = reverse - 1u;
        Real rhs = b[row];
        for (std::size_t c = row + 1u; c < n; ++c) {
            rhs -= a[row][c] * pivoted_solution[c];
        }
        const Real diagonal = a[row][row];
        if (!(std::abs(diagonal) > rank_threshold) ||
            !std::isfinite(diagonal)) {
            return false;
        }
        pivoted_solution[row] = rhs / diagonal;
        if (!std::isfinite(pivoted_solution[row])) {
            return false;
        }
    }

    x.fill(Real{0.0});
    for (std::size_t c = 0; c < n; ++c) {
        x[permutation[c]] = pivoted_solution[c];
    }
    return true;
}

[[nodiscard]] std::array<Real, 3> fitCoordinateScales(
    std::span<const RawFitObservation> observations,
    int dim) noexcept
{
    std::array<Real, 3> scales{{Real{0.0}, Real{0.0}, Real{1.0}}};
    for (const auto& observation : observations) {
        for (int d = 0; d < dim; ++d) {
            scales[static_cast<std::size_t>(d)] =
                std::max(scales[static_cast<std::size_t>(d)],
                         std::abs(observation.displacement[
                             static_cast<std::size_t>(d)]));
        }
    }
    for (int d = 0; d < dim; ++d) {
        if (!(scales[static_cast<std::size_t>(d)] > Real{0.0}) ||
            !std::isfinite(scales[static_cast<std::size_t>(d)])) {
            scales[static_cast<std::size_t>(d)] = Real{1.0};
        }
    }
    return scales;
}

[[nodiscard]] std::vector<FitObservation> nondimensionalizeObservations(
    std::span<const RawFitObservation> raw_observations,
    const std::array<Real, 3>& coordinate_scales,
    int dim)
{
    std::vector<FitObservation> observations;
    observations.reserve(raw_observations.size());
    for (const auto& raw : raw_observations) {
        std::array<Real, 3> nondimensional_displacement{};
        for (int d = 0; d < dim; ++d) {
            nondimensional_displacement[static_cast<std::size_t>(d)] =
                raw.displacement[static_cast<std::size_t>(d)] /
                coordinate_scales[static_cast<std::size_t>(d)];
        }
        const Real distance2 = dot(nondimensional_displacement,
                                   nondimensional_displacement);
        if (!(distance2 > Real{0.0}) || !std::isfinite(distance2)) {
            continue;
        }
        const Real weight = raw.relative_weight /
                            std::max(distance2, Real{1.0e-24});
        observations.push_back(FitObservation{
            quadraticRow(nondimensional_displacement, dim),
            raw.rhs,
            weight});
    }
    return observations;
}

void dimensionalizeFitCoefficients(
    std::array<Real, 9>& coefficients,
    const std::array<Real, 3>& scales,
    int dim) noexcept
{
    coefficients[0] /= scales[0];
    coefficients[1] /= scales[1];
    if (dim == 2) {
        coefficients[2] /= scales[0] * scales[0];
        coefficients[3] /= scales[0] * scales[1];
        coefficients[4] /= scales[1] * scales[1];
        return;
    }

    coefficients[2] /= scales[2];
    coefficients[3] /= scales[0] * scales[0];
    coefficients[4] /= scales[0] * scales[1];
    coefficients[5] /= scales[0] * scales[2];
    coefficients[6] /= scales[1] * scales[1];
    coefficients[7] /= scales[1] * scales[2];
    coefficients[8] /= scales[2] * scales[2];
}

[[nodiscard]] std::vector<std::vector<GlobalIndex>> buildVertexAdjacency(
    const assembly::IMeshAccess& mesh)
{
    const auto n_vertices = static_cast<std::size_t>(mesh.numVertices());
    std::vector<std::vector<GlobalIndex>> adjacency(n_vertices);
    std::vector<GlobalIndex> nodes;
    mesh.forEachCell([&](GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        for (const auto a : nodes) {
            if (a < 0 || static_cast<std::size_t>(a) >= n_vertices) {
                continue;
            }
            auto& row = adjacency[static_cast<std::size_t>(a)];
            for (const auto b : nodes) {
                if (b == a || b < 0 ||
                    static_cast<std::size_t>(b) >= n_vertices) {
                    continue;
                }
                row.push_back(b);
            }
        }
    });
    for (auto& row : adjacency) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return adjacency;
}

[[nodiscard]] std::vector<std::vector<std::size_t>> buildVertexSupplementalSampleAdjacency(
    const assembly::IMeshAccess& mesh,
    std::span<const LevelSetCurvatureProjectionSample> samples)
{
    const auto n_vertices = static_cast<std::size_t>(mesh.numVertices());
    std::vector<std::vector<std::size_t>> sample_adjacency(n_vertices);
    std::vector<GlobalIndex> nodes;
    for (std::size_t sample_index = 0; sample_index < samples.size();
         ++sample_index) {
        const auto& sample = samples[sample_index];
        if (sample.parent_cell >= static_cast<MeshIndex>(0) &&
            sample.parent_cell < mesh.numCells()) {
            mesh.getCellNodes(static_cast<GlobalIndex>(sample.parent_cell), nodes);
            for (const auto node : nodes) {
                if (node < 0 || static_cast<std::size_t>(node) >= n_vertices) {
                    continue;
                }
                sample_adjacency[static_cast<std::size_t>(node)].push_back(
                    sample_index);
            }
            continue;
        }

        Real best_distance2 = std::numeric_limits<Real>::infinity();
        GlobalIndex best_vertex = static_cast<GlobalIndex>(-1);
        for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
            const auto x = mesh.getNodeCoordinates(vertex);
            const std::array<Real, 3> dx{
                x[0] - sample.coordinate[0],
                x[1] - sample.coordinate[1],
                x[2] - sample.coordinate[2],
            };
            const Real distance2 = dot(dx, dx);
            if (distance2 < best_distance2) {
                best_distance2 = distance2;
                best_vertex = vertex;
            }
        }
        if (best_vertex >= 0) {
            sample_adjacency[static_cast<std::size_t>(best_vertex)].push_back(
                sample_index);
        }
    }

    for (auto& row : sample_adjacency) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return sample_adjacency;
}

void mixSignature(std::uint64_t& seed, std::uint64_t value) noexcept
{
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

[[nodiscard]] std::uint64_t realBitsForSignature(Real value) noexcept
{
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits),
                  "level-set curvature projection signature expects Real <= 64 bits");
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

[[nodiscard]] std::uint64_t supplementalSampleAdjacencySignature(
    std::span<const LevelSetCurvatureProjectionSample> samples) noexcept
{
    std::uint64_t seed = 0xcbf29ce484222325ull;
    mixSignature(seed, static_cast<std::uint64_t>(samples.size()));
    for (const auto& sample : samples) {
        mixSignature(seed, static_cast<std::uint64_t>(sample.parent_cell));
        mixSignature(seed, sample.free_surface_snapshot_revision_key);
        mixSignature(seed, sample.source_value_revision);
        mixSignature(seed, sample.cut_topology_revision);
        mixSignature(seed, sample.generated_interface_geometry ? 1u : 0u);
        if (sample.parent_cell < static_cast<MeshIndex>(0)) {
            for (const auto coordinate : sample.coordinate) {
                mixSignature(seed, realBitsForSignature(coordinate));
            }
        }
    }
    return seed;
}

struct SupplementalSampleRevisionIdentity {
    std::uint64_t free_surface_snapshot_revision_key{0};
    std::uint64_t source_value_revision{0};
};

[[nodiscard]] SupplementalSampleRevisionIdentity
supplementalSampleRevisionIdentity(
    std::span<const LevelSetCurvatureProjectionSample> samples)
{
    SupplementalSampleRevisionIdentity identity;
    bool found_revisioned_sample = false;
    bool found_unrevisioned_sample = false;
    for (const auto& sample : samples) {
        const bool has_snapshot_revision =
            sample.free_surface_snapshot_revision_key != 0u;
        const bool has_source_revision = sample.source_value_revision != 0u;
        if (has_snapshot_revision != has_source_revision) {
            throw std::invalid_argument(
                "level-set curvature supplemental sample has incomplete free-surface revision identity");
        }
        if (!has_snapshot_revision) {
            found_unrevisioned_sample = true;
            if (sample.cut_topology_revision != 0u) {
                throw std::invalid_argument(
                    "level-set curvature supplemental sample has a cut revision without a free-surface snapshot revision");
            }
            continue;
        }
        if (sample.cut_topology_revision == 0u) {
            throw std::invalid_argument(
                "level-set curvature supplemental sample has a free-surface revision without a cut topology revision");
        }
        found_revisioned_sample = true;
        if (identity.free_surface_snapshot_revision_key == 0u) {
            identity.free_surface_snapshot_revision_key =
                sample.free_surface_snapshot_revision_key;
            identity.source_value_revision = sample.source_value_revision;
            continue;
        }
        if (identity.free_surface_snapshot_revision_key !=
                sample.free_surface_snapshot_revision_key ||
            identity.source_value_revision != sample.source_value_revision) {
            throw std::invalid_argument(
                "level-set curvature supplemental samples mix free-surface snapshot revisions");
        }
    }
    if (found_revisioned_sample && found_unrevisioned_sample) {
        throw std::invalid_argument(
            "level-set curvature supplemental samples mix revisioned and unrevisioned geometry");
    }
    return identity;
}

[[nodiscard]] bool workspaceMatchesMesh(
    const LevelSetCurvatureProjectionWorkspace& workspace,
    const assembly::IMeshAccess& mesh) noexcept
{
    return workspace.mesh_vertices == mesh.numVertices() &&
           workspace.mesh_cells == mesh.numCells() &&
           workspace.mesh_dimension == mesh.dimension();
}

void recordWorkspaceMeshIdentity(LevelSetCurvatureProjectionWorkspace& workspace,
                                 const assembly::IMeshAccess& mesh) noexcept
{
    workspace.mesh_vertices = mesh.numVertices();
    workspace.mesh_cells = mesh.numCells();
    workspace.mesh_dimension = mesh.dimension();
    workspace.mesh_revision_tracking_available =
        mesh.revisionTrackingAvailable();
    workspace.mesh_geometry_revision =
        workspace.mesh_revision_tracking_available ? mesh.geometryRevision() : 0u;
    workspace.mesh_topology_revision =
        workspace.mesh_revision_tracking_available ? mesh.topologyRevision() : 0u;
    workspace.mesh_ownership_revision =
        workspace.mesh_revision_tracking_available ? mesh.ownershipRevision() : 0u;
    workspace.mesh_numbering_revision =
        workspace.mesh_revision_tracking_available ? mesh.numberingRevision() : 0u;
    workspace.mesh_coordinate_configuration_key =
        workspace.mesh_revision_tracking_available
            ? mesh.coordinateConfigurationKey()
            : 0u;
}

[[nodiscard]] bool workspaceMatchesTopology(
    const LevelSetCurvatureProjectionWorkspace& workspace,
    const assembly::IMeshAccess& mesh) noexcept
{
    if (!workspaceMatchesMesh(workspace, mesh)) {
        return false;
    }
    if (!mesh.revisionTrackingAvailable()) {
        return true;
    }
    return workspace.mesh_revision_tracking_available &&
           workspace.mesh_topology_revision == mesh.topologyRevision() &&
           workspace.mesh_ownership_revision == mesh.ownershipRevision() &&
           workspace.mesh_numbering_revision == mesh.numberingRevision();
}

[[nodiscard]] bool workspaceMatchesSampleGeometry(
    const LevelSetCurvatureProjectionWorkspace& workspace,
    const assembly::IMeshAccess& mesh) noexcept
{
    if (!workspaceMatchesTopology(workspace, mesh) ||
        !mesh.revisionTrackingAvailable()) {
        return false;
    }
    return workspace.mesh_revision_tracking_available &&
           workspace.mesh_geometry_revision == mesh.geometryRevision() &&
           workspace.mesh_coordinate_configuration_key ==
               mesh.coordinateConfigurationKey();
}

[[nodiscard]] const std::vector<std::vector<GlobalIndex>>&
cachedVertexAdjacency(const assembly::IMeshAccess& mesh,
                      LevelSetCurvatureProjectionWorkspace* workspace,
                      LevelSetCurvatureProjectionResult& result,
                      std::vector<std::vector<GlobalIndex>>& local_adjacency)
{
    if (workspace != nullptr &&
        workspace->vertex_adjacency_valid &&
        workspaceMatchesTopology(*workspace, mesh)) {
        result.reused_vertex_adjacency = true;
        result.vertex_adjacency_builds = workspace->vertex_adjacency_builds;
        return workspace->vertex_adjacency;
    }

    auto adjacency = buildVertexAdjacency(mesh);
    if (workspace != nullptr) {
        recordWorkspaceMeshIdentity(*workspace, mesh);
        workspace->vertex_adjacency = std::move(adjacency);
        workspace->vertex_adjacency_valid = true;
        workspace->sample_adjacency_valid = false;
        ++workspace->vertex_adjacency_builds;
        result.vertex_adjacency_builds = workspace->vertex_adjacency_builds;
        return workspace->vertex_adjacency;
    }

    local_adjacency = std::move(adjacency);
    result.vertex_adjacency_builds = 1u;
    return local_adjacency;
}

[[nodiscard]] const std::vector<std::vector<std::size_t>>&
cachedSampleAdjacency(
    const assembly::IMeshAccess& mesh,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    LevelSetCurvatureProjectionWorkspace* workspace,
    LevelSetCurvatureProjectionResult& result,
    std::vector<std::vector<std::size_t>>& local_sample_adjacency)
{
    const auto sample_signature =
        supplementalSampleAdjacencySignature(supplemental_samples);
    if (workspace != nullptr &&
        workspace->sample_adjacency_valid &&
        workspaceMatchesSampleGeometry(*workspace, mesh) &&
        workspace->sample_signature == sample_signature) {
        result.reused_sample_adjacency = true;
        result.sample_adjacency_builds = workspace->sample_adjacency_builds;
        return workspace->sample_adjacency;
    }

    auto sample_adjacency =
        buildVertexSupplementalSampleAdjacency(mesh, supplemental_samples);
    if (workspace != nullptr) {
        recordWorkspaceMeshIdentity(*workspace, mesh);
        workspace->sample_signature = sample_signature;
        workspace->sample_adjacency = std::move(sample_adjacency);
        workspace->sample_adjacency_valid = true;
        ++workspace->sample_adjacency_builds;
        result.sample_adjacency_builds = workspace->sample_adjacency_builds;
        return workspace->sample_adjacency;
    }

    local_sample_adjacency = std::move(sample_adjacency);
    result.sample_adjacency_builds = 1u;
    return local_sample_adjacency;
}

[[nodiscard]] std::vector<GlobalIndex> collectNeighbors(
    GlobalIndex center,
    const std::vector<std::vector<GlobalIndex>>& adjacency,
    int max_rings)
{
    const auto n_vertices = adjacency.size();
    if (center < 0 || static_cast<std::size_t>(center) >= n_vertices) {
        return {};
    }

    std::vector<unsigned char> visited(n_vertices, 0u);
    std::queue<std::pair<GlobalIndex, int>> queue;
    visited[static_cast<std::size_t>(center)] = 1u;
    queue.push({center, 0});

    std::vector<GlobalIndex> result;
    while (!queue.empty()) {
        const auto [vertex, ring] = queue.front();
        queue.pop();
        if (ring >= max_rings) {
            continue;
        }
        for (const auto neighbor :
             adjacency[static_cast<std::size_t>(vertex)]) {
            const auto idx = static_cast<std::size_t>(neighbor);
            if (visited[idx] != 0u) {
                continue;
            }
            visited[idx] = 1u;
            result.push_back(neighbor);
            queue.push({neighbor, ring + 1});
        }
    }
    return result;
}

[[nodiscard]] std::array<Real, 9> quadraticRow(const std::array<Real, 3>& dx,
                                               int dim) noexcept
{
    std::array<Real, 9> row{};
    row[0] = dx[0];
    row[1] = dx[1];
    if (dim == 2) {
        row[2] = Real{0.5} * dx[0] * dx[0];
        row[3] = dx[0] * dx[1];
        row[4] = Real{0.5} * dx[1] * dx[1];
        return row;
    }
    row[2] = dx[2];
    row[3] = Real{0.5} * dx[0] * dx[0];
    row[4] = dx[0] * dx[1];
    row[5] = dx[0] * dx[2];
    row[6] = Real{0.5} * dx[1] * dx[1];
    row[7] = dx[1] * dx[2];
    row[8] = Real{0.5} * dx[2] * dx[2];
    return row;
}

[[nodiscard]] Real curvatureFromFit(const std::array<Real, 9>& c,
                                    int dim,
                                    Real gradient_tolerance,
                                    bool& small_gradient) noexcept
{
    std::array<Real, 3> g{c[0], c[1], dim == 2 ? Real{0.0} : c[2]};
    std::array<std::array<Real, 3>, 3> h{};
    if (dim == 2) {
        h[0][0] = c[2];
        h[0][1] = c[3];
        h[1][0] = c[3];
        h[1][1] = c[4];
    } else {
        h[0][0] = c[3];
        h[0][1] = c[4];
        h[0][2] = c[5];
        h[1][0] = c[4];
        h[1][1] = c[6];
        h[1][2] = c[7];
        h[2][0] = c[5];
        h[2][1] = c[7];
        h[2][2] = c[8];
    }

    const Real g_norm = norm(g);
    if (!(g_norm > gradient_tolerance) || !std::isfinite(g_norm)) {
        small_gradient = true;
        return Real{0.0};
    }

    const Real trace_h = h[0][0] + h[1][1] + (dim == 2 ? Real{0.0} : h[2][2]);
    Real ghg = Real{0.0};
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            ghg += g[static_cast<std::size_t>(i)] *
                   h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                   g[static_cast<std::size_t>(j)];
        }
    }
    return ((g_norm * g_norm) * trace_h - ghg) /
           (g_norm * g_norm * g_norm);
}

struct WeightedNeighbor {
    std::size_t vertex{0};
    Real weight{0.0};
};

[[nodiscard]] std::array<Real, 3> subtract(const std::array<Real, 3>& a,
                                           const std::array<Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] std::array<Real, 3> cross(const std::array<Real, 3>& a,
                                        const std::array<Real, 3>& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

struct SimplexAffineGeometry {
    int dimension{0};
    std::size_t corner_count{0u};
    std::array<Real, 3> origin{};
    std::array<std::array<Real, 3>, 3> edges{};
    std::array<std::array<Real, 3>, 3> inverse_gram{};
    Real gradient_norm{0.0};
};

[[nodiscard]] bool invertSimplexGram(
    std::array<std::array<Real, 3>, 3> matrix,
    int dimension,
    std::array<std::array<Real, 3>, 3>& inverse) noexcept
{
    inverse = {};
    if (dimension == 2) {
        const Real determinant =
            matrix[0][0] * matrix[1][1] -
            matrix[0][1] * matrix[1][0];
        const Real scale = std::max(
            std::abs(matrix[0][0] * matrix[1][1]),
            std::abs(matrix[0][1] * matrix[1][0]));
        if (!(std::abs(determinant) >
              Real{128.0} * std::numeric_limits<Real>::epsilon() *
                  std::max(scale, std::numeric_limits<Real>::min())) ||
            !std::isfinite(determinant)) {
            return false;
        }
        inverse[0][0] = matrix[1][1] / determinant;
        inverse[0][1] = -matrix[0][1] / determinant;
        inverse[1][0] = -matrix[1][0] / determinant;
        inverse[1][1] = matrix[0][0] / determinant;
        return true;
    }
    if (dimension != 3) {
        return false;
    }

    const Real determinant =
        matrix[0][0] *
            (matrix[1][1] * matrix[2][2] -
             matrix[1][2] * matrix[2][1]) -
        matrix[0][1] *
            (matrix[1][0] * matrix[2][2] -
             matrix[1][2] * matrix[2][0]) +
        matrix[0][2] *
            (matrix[1][0] * matrix[2][1] -
             matrix[1][1] * matrix[2][0]);
    Real scale{0.0};
    for (int i = 0; i < 3; ++i) {
        scale = std::max(
            scale,
            std::abs(matrix[0][static_cast<std::size_t>(i)]) *
                std::max(
                    std::abs(matrix[1][(static_cast<std::size_t>(i) + 1u) % 3u] *
                             matrix[2][(static_cast<std::size_t>(i) + 2u) % 3u]),
                    std::abs(matrix[1][(static_cast<std::size_t>(i) + 2u) % 3u] *
                             matrix[2][(static_cast<std::size_t>(i) + 1u) % 3u])));
    }
    if (!(std::abs(determinant) >
          Real{256.0} * std::numeric_limits<Real>::epsilon() *
              std::max(scale, std::numeric_limits<Real>::min())) ||
        !std::isfinite(determinant)) {
        return false;
    }

    inverse[0][0] =
        (matrix[1][1] * matrix[2][2] -
         matrix[1][2] * matrix[2][1]) /
        determinant;
    inverse[0][1] =
        (matrix[0][2] * matrix[2][1] -
         matrix[0][1] * matrix[2][2]) /
        determinant;
    inverse[0][2] =
        (matrix[0][1] * matrix[1][2] -
         matrix[0][2] * matrix[1][1]) /
        determinant;
    inverse[1][0] =
        (matrix[1][2] * matrix[2][0] -
         matrix[1][0] * matrix[2][2]) /
        determinant;
    inverse[1][1] =
        (matrix[0][0] * matrix[2][2] -
         matrix[0][2] * matrix[2][0]) /
        determinant;
    inverse[1][2] =
        (matrix[0][2] * matrix[1][0] -
         matrix[0][0] * matrix[1][2]) /
        determinant;
    inverse[2][0] =
        (matrix[1][0] * matrix[2][1] -
         matrix[1][1] * matrix[2][0]) /
        determinant;
    inverse[2][1] =
        (matrix[0][1] * matrix[2][0] -
         matrix[0][0] * matrix[2][1]) /
        determinant;
    inverse[2][2] =
        (matrix[0][0] * matrix[1][1] -
         matrix[0][1] * matrix[1][0]) /
        determinant;
    return true;
}

[[nodiscard]] bool makeSimplexAffineGeometry(
    std::span<const std::array<Real, 3>> coordinates,
    std::span<const Real> values,
    int dimension,
    SimplexAffineGeometry& geometry) noexcept
{
    const auto corner_count = static_cast<std::size_t>(dimension + 1);
    if ((dimension != 2 && dimension != 3) ||
        coordinates.size() < corner_count || values.size() < corner_count) {
        return false;
    }
    geometry = {};
    geometry.dimension = dimension;
    geometry.corner_count = corner_count;
    geometry.origin = coordinates.front();
    std::array<std::array<Real, 3>, 3> gram{};
    for (int i = 0; i < dimension; ++i) {
        geometry.edges[static_cast<std::size_t>(i)] =
            subtract(coordinates[static_cast<std::size_t>(i + 1)],
                     geometry.origin);
        for (int j = 0; j < dimension; ++j) {
            gram[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                dot(geometry.edges[static_cast<std::size_t>(i)],
                    subtract(coordinates[static_cast<std::size_t>(j + 1)],
                             geometry.origin));
        }
    }
    if (!invertSimplexGram(gram, dimension, geometry.inverse_gram)) {
        return false;
    }

    std::array<Real, 3> gradient{};
    for (int edge = 0; edge < dimension; ++edge) {
        Real coefficient{0.0};
        for (int value = 0; value < dimension; ++value) {
            coefficient +=
                (values[static_cast<std::size_t>(value + 1)] - values[0]) *
                geometry.inverse_gram[static_cast<std::size_t>(value)]
                                     [static_cast<std::size_t>(edge)];
        }
        for (int component = 0; component < 3; ++component) {
            gradient[static_cast<std::size_t>(component)] +=
                coefficient *
                geometry.edges[static_cast<std::size_t>(edge)]
                              [static_cast<std::size_t>(component)];
        }
    }
    geometry.gradient_norm = norm(gradient);
    return geometry.gradient_norm > Real{0.0} &&
           std::isfinite(geometry.gradient_norm);
}

[[nodiscard]] bool simplexShapeValues(
    const SimplexAffineGeometry& geometry,
    const std::array<Real, 3>& point,
    std::array<Real, 4>& shape) noexcept
{
    shape = {};
    const auto displacement = subtract(point, geometry.origin);
    Real sum{0.0};
    for (int i = 0; i < geometry.dimension; ++i) {
        Real value{0.0};
        for (int j = 0; j < geometry.dimension; ++j) {
            value +=
                geometry.inverse_gram[static_cast<std::size_t>(i)]
                                     [static_cast<std::size_t>(j)] *
                dot(geometry.edges[static_cast<std::size_t>(j)],
                    displacement);
        }
        shape[static_cast<std::size_t>(i + 1)] = value;
        sum += value;
    }
    shape[0] = Real{1.0} - sum;
    return std::all_of(
        shape.begin(),
        shape.begin() + static_cast<std::ptrdiff_t>(geometry.corner_count),
        [](Real value) { return std::isfinite(value); });
}

[[nodiscard]] Real activeInterfaceMeasure(
    const interfaces::CutInterfaceDomainRequest& request,
    const interfaces::LevelSetCellCutInput& input,
    int dimension,
    bool& success)
{
    const auto cut = dimension == 2
        ? interfaces::cutLinearLevelSetCell2D(request, input)
        : interfaces::cutLinearLevelSetCell3D(request, input);
    success = cut.supported;
    Real measure{0.0};
    for (const auto& fragment : cut.fragments) {
        if (fragment.active()) {
            measure += fragment.measure;
        }
    }
    success = success && std::isfinite(measure) && measure >= Real{0.0};
    return measure;
}

struct DifferentiatedInterfacePoint {
    std::array<Real, 3> point{};
    std::array<std::array<Real, 3>, 4> derivative{};
};

[[nodiscard]] bool makeDifferentiatedInterfaceEdgeRoot(
    std::span<const std::array<Real, 3>> coordinates,
    std::span<const Real> signed_values,
    std::size_t first,
    std::size_t second,
    DifferentiatedInterfacePoint& root) noexcept
{
    if (first >= coordinates.size() || second >= coordinates.size() ||
        first >= signed_values.size() || second >= signed_values.size()) {
        return false;
    }
    const Real first_value = signed_values[first];
    const Real second_value = signed_values[second];
    if (!((first_value < Real{0.0} && second_value > Real{0.0}) ||
          (first_value > Real{0.0} && second_value < Real{0.0}))) {
        return false;
    }
    const Real denominator = first_value - second_value;
    const Real denominator_squared = denominator * denominator;
    if (!(denominator_squared > Real{0.0}) ||
        !std::isfinite(denominator_squared)) {
        return false;
    }
    const Real fraction = first_value / denominator;
    if (!(fraction > Real{0.0} && fraction < Real{1.0}) ||
        !std::isfinite(fraction)) {
        return false;
    }

    root = {};
    const auto edge = subtract(coordinates[second], coordinates[first]);
    const Real first_factor = -second_value / denominator_squared;
    const Real second_factor = first_value / denominator_squared;
    for (std::size_t component = 0u; component < 3u; ++component) {
        root.point[component] =
            coordinates[first][component] + fraction * edge[component];
        root.derivative[first][component] =
            first_factor * edge[component];
        root.derivative[second][component] =
            second_factor * edge[component];
    }
    return std::all_of(
        root.point.begin(), root.point.end(),
        [](Real value) { return std::isfinite(value); });
}

void accumulateDifferentiatedSegmentMeasure(
    const DifferentiatedInterfacePoint& first,
    const DifferentiatedInterfacePoint& second,
    std::size_t coefficient_count,
    std::span<Real> gradient,
    Real scale = Real{1.0}) noexcept
{
    const auto tangent = subtract(second.point, first.point);
    const Real length = norm(tangent);
    if (!(length > Real{0.0}) || !std::isfinite(length)) {
        std::fill(gradient.begin(), gradient.end(),
                  std::numeric_limits<Real>::quiet_NaN());
        return;
    }
    for (std::size_t coefficient = 0u;
         coefficient < coefficient_count;
         ++coefficient) {
        const auto derivative = subtract(
            second.derivative[coefficient],
            first.derivative[coefficient]);
        gradient[coefficient] +=
            scale * dot(tangent, derivative) / length;
    }
}

void accumulateDifferentiatedTriangleMeasure(
    const DifferentiatedInterfacePoint& first,
    const DifferentiatedInterfacePoint& second,
    const DifferentiatedInterfacePoint& third,
    std::size_t coefficient_count,
    std::span<Real> gradient,
    Real scale = Real{1.0}) noexcept
{
    const auto edge0 = subtract(second.point, first.point);
    const auto edge1 = subtract(third.point, first.point);
    const auto area_vector = cross(edge0, edge1);
    const Real doubled_area = norm(area_vector);
    if (!(doubled_area > Real{0.0}) || !std::isfinite(doubled_area)) {
        std::fill(gradient.begin(), gradient.end(),
                  std::numeric_limits<Real>::quiet_NaN());
        return;
    }
    for (std::size_t coefficient = 0u;
         coefficient < coefficient_count;
         ++coefficient) {
        const auto edge0_derivative = subtract(
            second.derivative[coefficient],
            first.derivative[coefficient]);
        const auto edge1_derivative = subtract(
            third.derivative[coefficient],
            first.derivative[coefficient]);
        const auto area_vector_derivative =
            cross(edge0_derivative, edge1);
        const auto second_term = cross(edge0, edge1_derivative);
        const std::array<Real, 3> combined{{
            area_vector_derivative[0] + second_term[0],
            area_vector_derivative[1] + second_term[1],
            area_vector_derivative[2] + second_term[2]}};
        gradient[coefficient] +=
            scale * Real{0.5} * dot(area_vector, combined) /
            doubled_area;
    }
}

[[nodiscard]] bool differentiateSimplexInterfaceMeasure(
    const interfaces::LevelSetCellCutInput& input,
    Real isovalue,
    const interfaces::LevelSetCellCutResult& cut,
    int dimension,
    std::span<Real> gradient) noexcept
{
    const auto corner_count = static_cast<std::size_t>(dimension + 1);
    if ((dimension != 2 && dimension != 3) ||
        input.node_coordinates.size() < corner_count ||
        input.level_set_values.size() < corner_count ||
        gradient.size() < corner_count) {
        return false;
    }
    std::fill(gradient.begin(), gradient.end(), Real{0.0});
    std::array<Real, 4> signed_values{};
    for (std::size_t corner = 0u; corner < corner_count; ++corner) {
        signed_values[corner] = input.level_set_values[corner] - isovalue;
    }

    constexpr std::array<std::array<std::size_t, 2>, 6> edges{{
        {{0u, 1u}}, {{1u, 2u}}, {{2u, 0u}},
        {{0u, 2u}}, {{0u, 3u}}, {{1u, 3u}}}};
    const std::array<std::array<std::size_t, 2>, 6> tetrahedron_edges{{
        {{0u, 1u}}, {{0u, 2u}}, {{0u, 3u}},
        {{1u, 2u}}, {{1u, 3u}}, {{2u, 3u}}}};
    const auto& active_edges = dimension == 2 ? edges : tetrahedron_edges;
    const std::size_t edge_count = dimension == 2 ? 3u : 6u;
    std::vector<DifferentiatedInterfacePoint> roots;
    roots.reserve(edge_count);
    for (std::size_t edge = 0u; edge < edge_count; ++edge) {
        const auto first = active_edges[edge][0];
        const auto second = active_edges[edge][1];
        if (!((signed_values[first] < Real{0.0} &&
               signed_values[second] > Real{0.0}) ||
              (signed_values[first] > Real{0.0} &&
               signed_values[second] < Real{0.0}))) {
            continue;
        }
        DifferentiatedInterfacePoint root;
        if (!makeDifferentiatedInterfaceEdgeRoot(
                std::span<const std::array<Real, 3>>(
                    input.node_coordinates.data(), corner_count),
                std::span<const Real>(signed_values.data(), corner_count),
                first, second, root)) {
            return false;
        }
        roots.push_back(root);
    }

    std::size_t active_fragment_count{0u};
    for (const auto& fragment : cut.fragments) {
        if (!fragment.active()) {
            continue;
        }
        ++active_fragment_count;
        if ((dimension == 2 && fragment.vertices.size() != 2u) ||
            (dimension == 3 &&
             (fragment.vertices.size() < 3u ||
              fragment.vertices.size() > 4u)) ||
            fragment.vertices.size() != roots.size()) {
            return false;
        }
        std::vector<DifferentiatedInterfacePoint> ordered_roots;
        ordered_roots.reserve(fragment.vertices.size());
        std::vector<unsigned char> used(roots.size(), 0u);
        for (const auto& vertex : fragment.vertices) {
            std::size_t best = roots.size();
            Real best_distance = std::numeric_limits<Real>::infinity();
            for (std::size_t candidate = 0u;
                 candidate < roots.size();
                 ++candidate) {
                if (used[candidate] != 0u) {
                    continue;
                }
                const Real distance =
                    norm(subtract(vertex.point, roots[candidate].point));
                if (distance < best_distance) {
                    best = candidate;
                    best_distance = distance;
                }
            }
            Real coordinate_scale{1.0};
            for (const auto value : vertex.point) {
                coordinate_scale =
                    std::max(coordinate_scale, std::abs(value));
            }
            if (best >= roots.size() ||
                best_distance >
                    Real{4096.0} *
                        std::numeric_limits<Real>::epsilon() *
                        coordinate_scale) {
                return false;
            }
            used[best] = 1u;
            ordered_roots.push_back(roots[best]);
        }

        if (dimension == 2) {
            accumulateDifferentiatedSegmentMeasure(
                ordered_roots[0], ordered_roots[1], corner_count,
                gradient);
        } else {
            for (std::size_t triangle = 1u;
                 triangle + 1u < ordered_roots.size();
                 ++triangle) {
                accumulateDifferentiatedTriangleMeasure(
                    ordered_roots[0], ordered_roots[triangle],
                    ordered_roots[triangle + 1u], corner_count,
                    gradient);
            }
        }
    }
    return active_fragment_count == 1u &&
           std::all_of(
               gradient.begin(),
               gradient.begin() + static_cast<std::ptrdiff_t>(corner_count),
               [](Real value) { return std::isfinite(value); });
}

[[nodiscard]] bool simplexBoundaryFaceCorners(
    ElementType type,
    LocalIndex local_face,
    std::array<std::size_t, 3>& corners,
    std::size_t& corner_count) noexcept
{
    corners = {};
    corner_count = 0u;
    const int face = static_cast<int>(local_face);
    if (type == ElementType::Triangle3) {
        constexpr std::array<std::array<std::size_t, 2>, 3> faces{{
            {{0u, 1u}},
            {{1u, 2u}},
            {{2u, 0u}},
        }};
        if (face < 0 || face >= static_cast<int>(faces.size())) {
            return false;
        }
        corners[0] = faces[static_cast<std::size_t>(face)][0];
        corners[1] = faces[static_cast<std::size_t>(face)][1];
        corner_count = 2u;
        return true;
    }
    if (type == ElementType::Tetra4) {
        constexpr std::array<std::array<std::size_t, 3>, 4> faces{{
            {{0u, 2u, 1u}},
            {{0u, 1u, 3u}},
            {{1u, 2u, 3u}},
            {{2u, 0u, 3u}},
        }};
        if (face < 0 || face >= static_cast<int>(faces.size())) {
            return false;
        }
        corners = faces[static_cast<std::size_t>(face)];
        corner_count = 3u;
        return true;
    }
    return false;
}

[[nodiscard]] Real activeBoundaryMeasure(
    std::span<const std::array<Real, 3>> coordinates,
    std::span<const Real> signed_values,
    bool negative_side,
    bool& success) noexcept
{
    success = false;
    if (coordinates.size() != signed_values.size() ||
        (coordinates.size() != 2u && coordinates.size() != 3u)) {
        return Real{0.0};
    }
    const auto inside = [negative_side](Real value) noexcept {
        return negative_side ? value <= Real{0.0} : value >= Real{0.0};
    };
    if (coordinates.size() == 2u) {
        const Real length = norm(subtract(coordinates[1], coordinates[0]));
        if (!(length > Real{0.0}) || !std::isfinite(length)) {
            return Real{0.0};
        }
        const bool inside0 = inside(signed_values[0]);
        const bool inside1 = inside(signed_values[1]);
        success = true;
        if (inside0 == inside1) {
            return inside0 ? length : Real{0.0};
        }
        const Real denominator = signed_values[0] - signed_values[1];
        if (!(std::abs(denominator) >
              std::numeric_limits<Real>::min()) ||
            !std::isfinite(denominator)) {
            success = false;
            return Real{0.0};
        }
        const Real fraction = signed_values[0] / denominator;
        if (!std::isfinite(fraction) || fraction < Real{0.0} ||
            fraction > Real{1.0}) {
            success = false;
            return Real{0.0};
        }
        return length * (inside0 ? fraction : Real{1.0} - fraction);
    }

    struct SignedBoundaryPoint {
        std::array<Real, 3> coordinate{};
        Real value{0.0};
    };
    std::vector<SignedBoundaryPoint> polygon;
    polygon.reserve(4u);
    for (std::size_t vertex = 0; vertex < coordinates.size(); ++vertex) {
        polygon.push_back({coordinates[vertex], signed_values[vertex]});
    }
    std::vector<SignedBoundaryPoint> clipped;
    clipped.reserve(4u);
    for (std::size_t edge = 0; edge < polygon.size(); ++edge) {
        const auto& a = polygon[edge];
        const auto& b = polygon[(edge + 1u) % polygon.size()];
        const bool inside_a = inside(a.value);
        const bool inside_b = inside(b.value);
        if (inside_a) {
            clipped.push_back(a);
        }
        if (inside_a == inside_b) {
            continue;
        }
        const Real denominator = a.value - b.value;
        if (!(std::abs(denominator) >
              std::numeric_limits<Real>::min()) ||
            !std::isfinite(denominator)) {
            return Real{0.0};
        }
        const Real fraction = a.value / denominator;
        if (!std::isfinite(fraction) || fraction < Real{0.0} ||
            fraction > Real{1.0}) {
            return Real{0.0};
        }
        clipped.push_back({{
                               a.coordinate[0] +
                                   fraction *
                                       (b.coordinate[0] - a.coordinate[0]),
                               a.coordinate[1] +
                                   fraction *
                                       (b.coordinate[1] - a.coordinate[1]),
                               a.coordinate[2] +
                                   fraction *
                                       (b.coordinate[2] - a.coordinate[2]),
                           },
                           Real{0.0}});
    }
    if (clipped.size() < 3u) {
        success = true;
        return Real{0.0};
    }
    Real area{0.0};
    const auto& origin = clipped.front().coordinate;
    for (std::size_t triangle = 1u;
         triangle + 1u < clipped.size();
         ++triangle) {
        area += Real{0.5} *
                norm(cross(subtract(clipped[triangle].coordinate, origin),
                           subtract(clipped[triangle + 1u].coordinate,
                                    origin)));
    }
    success = std::isfinite(area) && area >= Real{0.0};
    return area;
}

[[nodiscard]] bool differentiateActiveBoundaryMeasure(
    std::span<const std::array<Real, 3>> coordinates,
    std::span<const Real> signed_values,
    bool negative_side,
    std::span<Real> gradient) noexcept
{
    const auto count = coordinates.size();
    if (signed_values.size() != count || gradient.size() < count ||
        (count != 2u && count != 3u)) {
        return false;
    }
    std::fill(gradient.begin(), gradient.end(), Real{0.0});
    const auto inside = [negative_side](Real value) noexcept {
        return negative_side ? value < Real{0.0} : value > Real{0.0};
    };
    std::array<std::size_t, 3> inside_vertices{};
    std::array<std::size_t, 3> outside_vertices{};
    std::size_t inside_count{0u};
    std::size_t outside_count{0u};
    for (std::size_t vertex = 0u; vertex < count; ++vertex) {
        if (signed_values[vertex] == Real{0.0}) {
            return false;
        }
        if (inside(signed_values[vertex])) {
            inside_vertices[inside_count++] = vertex;
        } else {
            outside_vertices[outside_count++] = vertex;
        }
    }
    if (inside_count == 0u || outside_count == 0u) {
        return false;
    }

    if (count == 2u) {
        DifferentiatedInterfacePoint root;
        if (!makeDifferentiatedInterfaceEdgeRoot(
                coordinates, signed_values, 0u, 1u, root)) {
            return false;
        }
        DifferentiatedInterfacePoint first;
        first.point = coordinates[0];
        DifferentiatedInterfacePoint second;
        second.point = coordinates[1];
        if (inside(signed_values[0])) {
            accumulateDifferentiatedSegmentMeasure(
                first, root, count, gradient);
        } else {
            accumulateDifferentiatedSegmentMeasure(
                root, second, count, gradient);
        }
    } else if (inside_count == 1u) {
        const auto inside_vertex = inside_vertices[0];
        DifferentiatedInterfacePoint fixed;
        fixed.point = coordinates[inside_vertex];
        DifferentiatedInterfacePoint root0;
        DifferentiatedInterfacePoint root1;
        if (!makeDifferentiatedInterfaceEdgeRoot(
                coordinates, signed_values, inside_vertex,
                outside_vertices[0], root0) ||
            !makeDifferentiatedInterfaceEdgeRoot(
                coordinates, signed_values, inside_vertex,
                outside_vertices[1], root1)) {
            return false;
        }
        accumulateDifferentiatedTriangleMeasure(
            fixed, root0, root1, count, gradient);
    } else if (inside_count == 2u && outside_count == 1u) {
        const auto outside_vertex = outside_vertices[0];
        DifferentiatedInterfacePoint fixed;
        fixed.point = coordinates[outside_vertex];
        DifferentiatedInterfacePoint root0;
        DifferentiatedInterfacePoint root1;
        if (!makeDifferentiatedInterfaceEdgeRoot(
                coordinates, signed_values, outside_vertex,
                inside_vertices[0], root0) ||
            !makeDifferentiatedInterfaceEdgeRoot(
                coordinates, signed_values, outside_vertex,
                inside_vertices[1], root1)) {
            return false;
        }
        accumulateDifferentiatedTriangleMeasure(
            fixed, root0, root1, count, gradient, Real{-1.0});
    } else {
        return false;
    }
    return std::all_of(
        gradient.begin(),
        gradient.begin() + static_cast<std::ptrdiff_t>(count),
        [](Real value) { return std::isfinite(value); });
}

[[nodiscard]] bool accumulateKinematicAreaMass(
    const interfaces::LevelSetCellCutResult& cut,
    const SimplexAffineGeometry& geometry,
    std::span<Real> local_mass,
    std::array<std::array<Real, 4>, 4>& local_matrix) noexcept
{
    if (local_mass.size() < geometry.corner_count) {
        return false;
    }
    for (const auto& fragment : cut.fragments) {
        if (!fragment.active()) {
            continue;
        }
        if (geometry.dimension == 2) {
            if (fragment.vertices.size() != 2u) {
                return false;
            }
            const auto& a = fragment.vertices[0].point;
            const auto& b = fragment.vertices[1].point;
            const Real length = norm(subtract(b, a));
            std::array<Real, 4> shape_a{};
            std::array<Real, 4> shape_b{};
            if (!(length > Real{0.0}) || !std::isfinite(length) ||
                !simplexShapeValues(geometry, a, shape_a) ||
                !simplexShapeValues(geometry, b, shape_b)) {
                return false;
            }
            for (std::size_t i = 0; i < geometry.corner_count; ++i) {
                local_mass[i] +=
                    length * (shape_a[i] + shape_b[i]) /
                    (Real{2.0} * geometry.gradient_norm);
                for (std::size_t j = 0; j < geometry.corner_count; ++j) {
                    local_matrix[i][j] +=
                        length *
                        (Real{2.0} * shape_a[i] * shape_a[j] +
                         shape_a[i] * shape_b[j] +
                         shape_b[i] * shape_a[j] +
                         Real{2.0} * shape_b[i] * shape_b[j]) /
                        (Real{6.0} * geometry.gradient_norm);
                }
            }
            continue;
        }

        if (fragment.vertices.size() < 3u) {
            return false;
        }
        const auto& a = fragment.vertices.front().point;
        for (std::size_t triangle = 1u;
             triangle + 1u < fragment.vertices.size();
             ++triangle) {
            const auto& b = fragment.vertices[triangle].point;
            const auto& c = fragment.vertices[triangle + 1u].point;
            const Real area =
                Real{0.5} * norm(cross(subtract(b, a), subtract(c, a)));
            std::array<Real, 4> shape_a{};
            std::array<Real, 4> shape_b{};
            std::array<Real, 4> shape_c{};
            if (!(area > Real{0.0}) || !std::isfinite(area) ||
                !simplexShapeValues(geometry, a, shape_a) ||
                !simplexShapeValues(geometry, b, shape_b) ||
                !simplexShapeValues(geometry, c, shape_c)) {
                return false;
            }
            for (std::size_t i = 0; i < geometry.corner_count; ++i) {
                local_mass[i] +=
                    area * (shape_a[i] + shape_b[i] + shape_c[i]) /
                    (Real{3.0} * geometry.gradient_norm);
                for (std::size_t j = 0; j < geometry.corner_count; ++j) {
                    const Real diagonal =
                        Real{2.0} *
                        (shape_a[i] * shape_a[j] +
                         shape_b[i] * shape_b[j] +
                         shape_c[i] * shape_c[j]);
                    const Real cross_terms =
                        shape_a[i] * shape_b[j] +
                        shape_b[i] * shape_a[j] +
                        shape_a[i] * shape_c[j] +
                        shape_c[i] * shape_a[j] +
                        shape_b[i] * shape_c[j] +
                        shape_c[i] * shape_b[j];
                    local_matrix[i][j] +=
                        area * (diagonal + cross_terms) /
                        (Real{12.0} * geometry.gradient_norm);
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool solveKinematicAreaMassSystem(
    const std::vector<std::map<std::size_t, Real>>& matrix,
    std::span<const Real> rhs,
    std::span<const Real> null_direction,
    bool use_diagonal_preconditioner,
    std::vector<Real>& solution,
    std::size_t& iterations,
    Real& relative_residual) noexcept
{
    const auto n = rhs.size();
    solution.assign(n, Real{0.0});
    std::vector<Real> residual(rhs.begin(), rhs.end());
    std::vector<Real> preconditioned(n, Real{0.0});
    std::vector<Real> direction(n, Real{0.0});
    std::vector<Real> applied(n, Real{0.0});

    const auto dot_product = [](std::span<const Real> a,
                                std::span<const Real> b) noexcept {
        Real value{0.0};
        for (std::size_t i = 0; i < a.size(); ++i) {
            value += a[i] * b[i];
        }
        return value;
    };
    const auto project_null = [&](std::vector<Real>& vector) noexcept {
        if (null_direction.size() != vector.size()) {
            return;
        }
        const Real denominator =
            dot_product(null_direction, null_direction);
        if (!(denominator > Real{0.0}) || !std::isfinite(denominator)) {
            return;
        }
        const Real coefficient =
            dot_product(vector, null_direction) / denominator;
        for (std::size_t i = 0; i < vector.size(); ++i) {
            vector[i] -= coefficient * null_direction[i];
        }
    };
    const auto apply = [&](std::span<const Real> x,
                           std::vector<Real>& y) noexcept {
        std::fill(y.begin(), y.end(), Real{0.0});
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            for (const auto& [column, value] : matrix[row]) {
                if (column < x.size()) {
                    y[row] += value * x[column];
                }
            }
        }
    };
    const auto precondition = [&](std::span<const Real> input,
                                  std::vector<Real>& output) noexcept {
        if (!use_diagonal_preconditioner) {
            std::copy(input.begin(), input.end(), output.begin());
            return;
        }
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            const auto diagonal = matrix[row].find(row);
            output[row] =
                diagonal != matrix[row].end() && diagonal->second > Real{0.0}
                    ? input[row] / diagonal->second
                    : Real{0.0};
        }
    };

    project_null(residual);
    const Real rhs_norm =
        std::sqrt(std::max(dot_product(residual, residual), Real{0.0}));
    if (!(rhs_norm > Real{0.0}) || !std::isfinite(rhs_norm)) {
        relative_residual = Real{0.0};
        return std::isfinite(rhs_norm);
    }
    precondition(residual, preconditioned);
    project_null(preconditioned);
    direction = preconditioned;
    Real rz = dot_product(residual, preconditioned);
    if (!(rz > Real{0.0}) || !std::isfinite(rz)) {
        return false;
    }

    const std::size_t max_iterations =
        std::max<std::size_t>(200u, 20u * std::max<std::size_t>(1u, n));
    const Real tolerance = Real{1.0e-10};
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        apply(direction, applied);
        const Real p_ap = dot_product(direction, applied);
        if (!(p_ap > Real{0.0}) || !std::isfinite(p_ap)) {
            return false;
        }
        const Real alpha = rz / p_ap;
        for (std::size_t i = 0; i < n; ++i) {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * applied[i];
        }
        project_null(residual);
        const Real residual_norm = std::sqrt(
            std::max(dot_product(residual, residual), Real{0.0}));
        iterations = iteration + 1u;
        relative_residual = residual_norm / rhs_norm;
        if (!std::isfinite(relative_residual)) {
            return false;
        }
        if (relative_residual <= tolerance) {
            return true;
        }
        precondition(residual, preconditioned);
        project_null(preconditioned);
        const Real next_rz = dot_product(residual, preconditioned);
        if (!(next_rz > Real{0.0}) || !std::isfinite(next_rz)) {
            return false;
        }
        const Real beta = next_rz / rz;
        for (std::size_t i = 0; i < n; ++i) {
            direction[i] = preconditioned[i] + beta * direction[i];
        }
        project_null(direction);
        rz = next_rz;
    }
    return false;
}

[[nodiscard]] bool solveKinematicAreaMassMinimumNorm(
    const std::vector<std::map<std::size_t, Real>>& matrix,
    std::span<const Real> rhs,
    std::vector<Real>& solution,
    std::size_t& iterations,
    Real& relative_residual) noexcept
{
    const auto n = rhs.size();
    if (matrix.size() != n) {
        return false;
    }
    solution.assign(n, Real{0.0});
    iterations = 0u;
    relative_residual = std::numeric_limits<Real>::infinity();

    const auto dot_product = [](std::span<const Real> a,
                                std::span<const Real> b) noexcept {
        Real value{0.0};
        for (std::size_t i = 0u; i < a.size(); ++i) {
            value += a[i] * b[i];
        }
        return value;
    };
    const auto vector_norm = [&](std::span<const Real> values) noexcept {
        const Real norm2 = dot_product(values, values);
        return norm2 >= Real{0.0} && std::isfinite(norm2)
            ? std::sqrt(norm2)
            : std::numeric_limits<Real>::infinity();
    };
    const auto apply = [&](std::span<const Real> x,
                           std::vector<Real>& y) noexcept {
        std::fill(y.begin(), y.end(), Real{0.0});
        for (std::size_t row = 0u; row < matrix.size(); ++row) {
            for (const auto& [column, value] : matrix[row]) {
                if (column < x.size()) {
                    y[row] += value * x[column];
                }
            }
        }
    };

    const Real rhs_norm = vector_norm(rhs);
    if (!std::isfinite(rhs_norm)) {
        return false;
    }
    if (rhs_norm == Real{0.0}) {
        relative_residual = Real{0.0};
        return true;
    }

    std::vector<Real> u(rhs.begin(), rhs.end());
    for (auto& value : u) {
        value /= rhs_norm;
    }
    std::vector<Real> v(n, Real{0.0});
    apply(u, v);
    Real alpha = vector_norm(v);
    if (!std::isfinite(alpha)) {
        return false;
    }
    const Real initial_normal_residual = alpha * rhs_norm;
    if (alpha == Real{0.0}) {
        relative_residual = Real{1.0};
        return true;
    }
    for (auto& value : v) {
        value /= alpha;
    }

    std::vector<Real> w = v;
    std::vector<Real> next_u(n, Real{0.0});
    std::vector<Real> next_v(n, Real{0.0});
    std::vector<Real> applied(n, Real{0.0});
    std::vector<Real> residual(n, Real{0.0});
    std::vector<Real> normal_residual(n, Real{0.0});
    Real rho_bar = alpha;
    Real phi_bar = rhs_norm;
    constexpr Real tolerance{5.0e-13};

    const auto converged = [&]() noexcept {
        apply(solution, applied);
        for (std::size_t i = 0u; i < n; ++i) {
            residual[i] = rhs[i] - applied[i];
        }
        const Real residual_norm = vector_norm(residual);
        if (!std::isfinite(residual_norm)) {
            return false;
        }
        relative_residual = residual_norm / rhs_norm;
        apply(residual, normal_residual);
        const Real normal_norm = vector_norm(normal_residual);
        if (!std::isfinite(relative_residual) ||
            !std::isfinite(normal_norm)) {
            return false;
        }
        const Real relative_normal_residual =
            initial_normal_residual > Real{0.0}
                ? normal_norm / initial_normal_residual
                : normal_norm;
        return relative_residual <= tolerance ||
               relative_normal_residual <= tolerance;
    };

    const std::size_t max_iterations =
        std::max<std::size_t>(200u, 20u * std::max<std::size_t>(1u, n));
    for (std::size_t iteration = 0u;
         iteration < max_iterations;
         ++iteration) {
        apply(v, next_u);
        for (std::size_t i = 0u; i < n; ++i) {
            next_u[i] -= alpha * u[i];
        }
        const Real beta = vector_norm(next_u);
        if (!std::isfinite(beta)) {
            return false;
        }
        if (beta > Real{0.0}) {
            for (auto& value : next_u) {
                value /= beta;
            }
        } else {
            std::fill(next_u.begin(), next_u.end(), Real{0.0});
        }

        apply(next_u, next_v);
        for (std::size_t i = 0u; i < n; ++i) {
            next_v[i] -= beta * v[i];
        }
        const Real next_alpha = vector_norm(next_v);
        if (!std::isfinite(next_alpha)) {
            return false;
        }
        if (next_alpha > Real{0.0}) {
            for (auto& value : next_v) {
                value /= next_alpha;
            }
        } else {
            std::fill(next_v.begin(), next_v.end(), Real{0.0});
        }

        const Real rho = std::hypot(rho_bar, beta);
        if (!(rho > Real{0.0}) || !std::isfinite(rho)) {
            return converged();
        }
        const Real cosine = rho_bar / rho;
        const Real sine = beta / rho;
        const Real theta = sine * next_alpha;
        rho_bar = -cosine * next_alpha;
        const Real phi = cosine * phi_bar;
        phi_bar = sine * phi_bar;
        const Real solution_scale = phi / rho;
        const Real direction_scale = theta / rho;
        for (std::size_t i = 0u; i < n; ++i) {
            solution[i] += solution_scale * w[i];
            w[i] = next_v[i] - direction_scale * w[i];
        }
        iterations = iteration + 1u;
        u.swap(next_u);
        v.swap(next_v);
        alpha = next_alpha;

        const Real estimated_relative_residual =
            std::abs(phi_bar) / rhs_norm;
        const Real estimated_normal_residual =
            alpha * std::abs(sine * phi);
        const Real estimated_relative_normal_residual =
            initial_normal_residual > Real{0.0}
                ? estimated_normal_residual /
                      initial_normal_residual
                : estimated_normal_residual;
        if (estimated_relative_residual <= tolerance ||
            estimated_relative_normal_residual <= tolerance ||
            alpha == Real{0.0} || beta == Real{0.0}) {
            if (converged()) {
                return true;
            }
        }
    }
    return converged();
}

[[nodiscard]] bool recoverKinematicAreaGradientCurvature(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature,
    std::vector<unsigned char>& active_vertices,
    std::vector<unsigned char>& fitted,
    LevelSetCurvatureProjectionResult& result)
{
    const int dimension = mesh.dimension();
    const auto n_vertices = curvature.size();
    active_vertices.assign(n_vertices, 0u);
    fitted.assign(n_vertices, 0u);
    std::vector<Real> area_gradient(n_vertices, Real{0.0});
    std::vector<Real> young_wall_gradient(n_vertices, Real{0.0});
    std::vector<Real> lumped_kinematic_mass(n_vertices, Real{0.0});
    std::vector<Real> lumped_interface_measure(n_vertices, Real{0.0});
    std::vector<std::map<std::size_t, Real>> kinematic_mass(n_vertices);
    std::string failure;

    Real global_value_scale{0.0};
    for (const auto value : level_set_vertex_values) {
        global_value_scale =
            std::max(global_value_scale, std::abs(value - options.isovalue));
    }
    if (!(global_value_scale > Real{0.0}) ||
        !std::isfinite(global_value_scale)) {
        result.diagnostic =
            "kinematic-area-gradient curvature recovery requires a nonconstant finite level-set field";
        return false;
    }

    const Real zero_tolerance =
        Real{512.0} * std::numeric_limits<Real>::epsilon() *
        global_value_scale;
    std::size_t negative_vertices{0u};
    std::size_t positive_vertices{0u};
    Real signed_value_sum{0.0};
    std::vector<Real> tie_break_scales(n_vertices, Real{0.0});
    for (const auto value : level_set_vertex_values) {
        const Real signed_value = value - options.isovalue;
        signed_value_sum += signed_value;
        negative_vertices += signed_value < -zero_tolerance ? 1u : 0u;
        positive_vertices += signed_value > zero_tolerance ? 1u : 0u;
    }
    mesh.forEachCell([&](GlobalIndex cell) {
        std::vector<GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        Real cell_scale{0.0};
        for (const auto node : nodes) {
            if (node < 0 || static_cast<std::size_t>(node) >= n_vertices) {
                continue;
            }
            cell_scale = std::max(
                cell_scale,
                std::abs(level_set_vertex_values[
                             static_cast<std::size_t>(node)] -
                         options.isovalue));
        }
        for (const auto node : nodes) {
            if (node < 0 || static_cast<std::size_t>(node) >= n_vertices) {
                continue;
            }
            const auto index = static_cast<std::size_t>(node);
            if (std::abs(level_set_vertex_values[index] - options.isovalue) <=
                zero_tolerance) {
                tie_break_scales[index] =
                    std::max(tie_break_scales[index], cell_scale);
            }
        }
    });
    int tie_break_sign = 1;
    if (negative_vertices > positive_vertices ||
        (negative_vertices == positive_vertices &&
         signed_value_sum < Real{0.0})) {
        tie_break_sign = -1;
    }
    std::vector<Real> working_level_set_values(
        level_set_vertex_values.begin(), level_set_vertex_values.end());
    const Real tie_break_factor =
        std::pow(std::numeric_limits<Real>::epsilon(), Real{0.25});
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        if (tie_break_scales[vertex] == Real{0.0}) {
            continue;
        }
        const Real tie_break_value =
            tie_break_factor * tie_break_scales[vertex];
        if (!(tie_break_value > zero_tolerance) ||
            !std::isfinite(tie_break_value)) {
            result.diagnostic =
                "kinematic-area-gradient curvature recovery could not resolve an isovalue vertex";
            return false;
        }
        working_level_set_values[vertex] =
            options.isovalue +
            static_cast<Real>(tie_break_sign) * tie_break_value;
        ++result.kinematic_area_gradient_tie_break_vertices;
        result.kinematic_area_gradient_max_tie_break_value = std::max(
            result.kinematic_area_gradient_max_tie_break_value,
            tie_break_value);
    }
    if (result.kinematic_area_gradient_tie_break_vertices > 0u) {
        result.kinematic_area_gradient_tie_break_sign = tie_break_sign;
    }

    interfaces::CutInterfaceDomainRequest cut_request;
    cut_request.source = interfaces::LevelSetInterfaceSource::fromEvaluator(
        "kinematic-area-gradient-local-measure");
    cut_request.interface_marker = 1;
    cut_request.isovalue = options.isovalue;
    cut_request.tolerance =
        Real{64.0} * std::numeric_limits<Real>::epsilon() *
        global_value_scale;
    cut_request.quadrature_order = 1;
    cut_request.frame = geometry::CutGeometryFrame::Current;
    cut_request.implicit_geometry_mode = "LinearCorner";
    cut_request.implicit_quadrature_backend = "LinearCorner";

    mesh.forEachCell([&](GlobalIndex cell) {
        if (!failure.empty()) {
            return;
        }
        if (!mesh.isOwnedCell(cell)) {
            failure =
                "kinematic-area-gradient curvature recovery is not yet available on meshes with nonowned cells";
            return;
        }
        const auto type = mesh.getCellType(cell);
        const bool supported =
            (dimension == 2 && type == ElementType::Triangle3) ||
            (dimension == 3 && type == ElementType::Tetra4);
        if (!supported) {
            failure =
                "kinematic-area-gradient curvature recovery requires affine Triangle3 or Tetra4 cells";
            return;
        }

        std::vector<GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        const auto corner_count = static_cast<std::size_t>(dimension + 1);
        if (nodes.size() < corner_count) {
            failure =
                "kinematic-area-gradient curvature recovery found incomplete simplex connectivity";
            return;
        }

        interfaces::LevelSetCellCutInput input;
        input.parent_cell = cell;
        input.element_type = type;
        input.node_coordinates.reserve(corner_count);
        input.level_set_values.reserve(corner_count);
        bool has_negative = false;
        bool has_positive = false;
        Real cell_value_scale{0.0};
        for (std::size_t i = 0; i < corner_count; ++i) {
            const auto node = nodes[i];
            if (node < 0 || static_cast<std::size_t>(node) >= n_vertices) {
                failure =
                    "kinematic-area-gradient curvature recovery found invalid simplex connectivity";
                return;
            }
            input.node_coordinates.push_back(mesh.getNodeCoordinates(node));
            const Real value =
                working_level_set_values[static_cast<std::size_t>(node)];
            input.level_set_values.push_back(value);
            const Real signed_value = value - options.isovalue;
            cell_value_scale =
                std::max(cell_value_scale, std::abs(signed_value));
            has_negative = has_negative || signed_value < -cut_request.tolerance;
            has_positive = has_positive || signed_value > cut_request.tolerance;
        }
        if (!has_negative || !has_positive) {
            return;
        }

        SimplexAffineGeometry simplex;
        if (!makeSimplexAffineGeometry(
                std::span<const std::array<Real, 3>>(
                    input.node_coordinates.data(),
                    input.node_coordinates.size()),
                std::span<const Real>(input.level_set_values.data(),
                                      input.level_set_values.size()),
                dimension,
                simplex) ||
            !(simplex.gradient_norm > options.gradient_tolerance)) {
            failure =
                "kinematic-area-gradient curvature recovery found a degenerate simplex or level-set gradient";
            return;
        }

        const auto baseline_cut = dimension == 2
            ? interfaces::cutLinearLevelSetCell2D(cut_request, input)
            : interfaces::cutLinearLevelSetCell3D(cut_request, input);
        if (!baseline_cut.supported || !baseline_cut.hasActiveFragments()) {
            failure =
                "kinematic-area-gradient curvature recovery could not reproduce a strict simplex cut";
            return;
        }
        std::array<Real, 4> local_mass{};
        std::array<std::array<Real, 4>, 4> local_matrix{};
        if (!accumulateKinematicAreaMass(
                baseline_cut,
                simplex,
                std::span<Real>(local_mass.data(), corner_count),
                local_matrix)) {
            failure =
                "kinematic-area-gradient curvature recovery could not assemble its local kinematic mass";
            return;
        }

        // Richardson extrapolation cancels the fourth-order term in the
        // centered stencil, leaving a sixth-order truncation error.  Balance
        // that error against floating-point subtraction error with h~eps^(1/7).
        std::array<Real, 4> local_gradient{};
        if (!differentiateSimplexInterfaceMeasure(
                input, options.isovalue, baseline_cut, dimension,
                std::span<Real>(local_gradient.data(), corner_count))) {
            failure =
                "kinematic-area-gradient curvature recovery could not differentiate the fixed simplex cut";
            return;
        }

        const Real nominal_step =
            std::pow(std::numeric_limits<Real>::epsilon(), Real{1.0 / 7.0}) *
            cell_value_scale;
        for (std::size_t i = 0; i < corner_count; ++i) {
            const Real margin =
                std::abs(input.level_set_values[i] - options.isovalue);
            const Real smooth_margin_floor =
                Real{512.0} * std::numeric_limits<Real>::epsilon() *
                cell_value_scale;
            const Real step =
                std::min(nominal_step, Real{0.20} * margin);
            if (!(step >
                  smooth_margin_floor) ||
                !std::isfinite(step)) {
                failure =
                    "kinematic-area-gradient curvature recovery could not choose a finite derivative step";
                return;
            }

            std::array<Real, 6> measures{};
            const std::array<Real, 6> offsets{{
                Real{-2.0},
                Real{-1.0},
                Real{-0.5},
                Real{0.5},
                Real{1.0},
                Real{2.0}}};
            for (std::size_t sample = 0; sample < offsets.size(); ++sample) {
                auto perturbed = input;
                perturbed.level_set_values[i] += offsets[sample] * step;
                bool measure_success = false;
                measures[sample] = activeInterfaceMeasure(
                    cut_request, perturbed, dimension, measure_success);
                ++result.kinematic_area_gradient_measure_evaluations;
                if (!measure_success || !(measures[sample] > Real{0.0})) {
                    failure =
                        "kinematic-area-gradient curvature recovery derivative left the fixed cut topology";
                    return;
                }
            }
            const Real fourth_order_full =
                (measures[0] - Real{8.0} * measures[1] +
                 Real{8.0} * measures[4] - measures[5]) /
                (Real{12.0} * step);
            const Real fourth_order_half =
                (measures[1] - Real{8.0} * measures[2] +
                 Real{8.0} * measures[3] - measures[4]) /
                (Real{6.0} * step);
            const Real richardson =
                (Real{16.0} * fourth_order_half - fourth_order_full) /
                Real{15.0};
            if (!std::isfinite(fourth_order_full) ||
                !std::isfinite(fourth_order_half) ||
                !std::isfinite(richardson)) {
                failure =
                    "kinematic-area-gradient curvature recovery produced a nonfinite local derivative";
                return;
            }
            const Real derivative_scale = std::max(
                std::abs(local_gradient[i]),
                baseline_cut.fragments.front().measure /
                    std::max(cell_value_scale,
                             std::numeric_limits<Real>::min()));
            const Real disagreement = std::max(
                std::abs(fourth_order_half - fourth_order_full),
                std::abs(richardson - local_gradient[i])) /
                std::max(derivative_scale,
                         std::numeric_limits<Real>::min());
            result.kinematic_area_gradient_max_relative_fd_disagreement =
                std::max(
                    result.kinematic_area_gradient_max_relative_fd_disagreement,
                    disagreement);
        }

        for (std::size_t i = 0; i < corner_count; ++i) {
            const auto index = static_cast<std::size_t>(nodes[i]);
            if (!(local_mass[i] > Real{0.0}) ||
                !std::isfinite(local_mass[i])) {
                failure =
                    "kinematic-area-gradient curvature recovery produced a nonpositive local kinematic mass";
                return;
            }
            area_gradient[index] += local_gradient[i];
            lumped_kinematic_mass[index] += local_mass[i];
            lumped_interface_measure[index] +=
                local_mass[i] * simplex.gradient_norm;
            for (std::size_t j = 0; j < corner_count; ++j) {
                const auto column = static_cast<std::size_t>(nodes[j]);
                if (local_matrix[i][j] != Real{0.0}) {
                    kinematic_mass[index][column] += local_matrix[i][j];
                }
            }
        }
        ++result.kinematic_area_gradient_cut_cells;
    });

    if (!failure.empty()) {
        result.diagnostic = std::move(failure);
        return false;
    }
    if (result.kinematic_area_gradient_cut_cells == 0u) {
        result.diagnostic =
            "kinematic-area-gradient curvature recovery found no strict cut cells";
        return false;
    }

    result.kinematic_area_gradient_young_wall_count =
        options.kinematic_area_gradient_young_walls.size();
    for (const auto& wall :
         options.kinematic_area_gradient_young_walls) {
        const Real raw_coefficient =
            -std::cos(wall.equilibrium_contact_angle_radians);
        const Real coefficient =
            std::abs(raw_coefficient) <=
                    Real{64.0} * std::numeric_limits<Real>::epsilon()
                ? Real{0.0}
                : raw_coefficient;
        std::size_t wall_boundary_faces{0u};
        mesh.forEachBoundaryFace(
            wall.boundary_marker,
            [&](GlobalIndex face, GlobalIndex cell) {
                if (!failure.empty()) {
                    return;
                }
                if (!mesh.isOwnedCell(cell)) {
                    failure =
                        "kinematic-area-gradient Young wall recovery found a nonowned boundary parent cell";
                    return;
                }
                ++wall_boundary_faces;
                ++result
                      .kinematic_area_gradient_young_wall_boundary_faces;
                const auto type = mesh.getCellType(cell);
                std::array<std::size_t, 3> face_corners{};
                std::size_t face_corner_count{0u};
                if (!simplexBoundaryFaceCorners(
                        type,
                        mesh.getLocalFaceIndex(face, cell),
                        face_corners,
                        face_corner_count) ||
                    face_corner_count !=
                        static_cast<std::size_t>(dimension)) {
                    failure =
                        "kinematic-area-gradient Young wall recovery requires an affine simplex boundary face";
                    return;
                }
                std::vector<GlobalIndex> nodes;
                mesh.getCellNodes(cell, nodes);
                std::array<std::array<Real, 3>, 3> coordinates{};
                std::array<Real, 3> signed_values{};
                Real face_value_scale{0.0};
                bool has_negative = false;
                bool has_positive = false;
                for (std::size_t local = 0; local < face_corner_count;
                     ++local) {
                    const auto corner = face_corners[local];
                    if (corner >= nodes.size() || nodes[corner] < 0 ||
                        static_cast<std::size_t>(nodes[corner]) >=
                            n_vertices) {
                        failure =
                            "kinematic-area-gradient Young wall recovery found invalid boundary connectivity";
                        return;
                    }
                    const auto node =
                        static_cast<std::size_t>(nodes[corner]);
                    coordinates[local] = mesh.getNodeCoordinates(
                        static_cast<GlobalIndex>(node));
                    signed_values[local] =
                        working_level_set_values[node] - options.isovalue;
                    face_value_scale = std::max(
                        face_value_scale,
                        std::abs(signed_values[local]));
                    has_negative = has_negative ||
                                   signed_values[local] <
                                       -cut_request.tolerance;
                    has_positive = has_positive ||
                                   signed_values[local] >
                                       cut_request.tolerance;
                }
                if (!has_negative || !has_positive) {
                    return;
                }
                ++result.kinematic_area_gradient_young_wall_cut_faces;
                if (coefficient == Real{0.0}) {
                    return;
                }

                std::array<Real, 3> local_wall_gradient{};
                if (!differentiateActiveBoundaryMeasure(
                        std::span<const std::array<Real, 3>>(
                            coordinates.data(), face_corner_count),
                        std::span<const Real>(
                            signed_values.data(), face_corner_count),
                        options
                            .kinematic_area_gradient_negative_liquid_side,
                        std::span<Real>(local_wall_gradient.data(),
                                        face_corner_count))) {
                    failure =
                        "kinematic-area-gradient Young wall recovery could not differentiate the fixed contact topology";
                    return;
                }

                const Real nominal_step =
                    std::pow(std::numeric_limits<Real>::epsilon(),
                             Real{1.0 / 7.0}) *
                    face_value_scale;
                for (std::size_t local = 0; local < face_corner_count;
                     ++local) {
                    const Real margin = std::abs(signed_values[local]);
                    const Real step =
                        std::min(nominal_step, Real{0.20} * margin);
                    if (!(step >
                          Real{512.0} *
                              std::numeric_limits<Real>::epsilon() *
                              face_value_scale) ||
                        !std::isfinite(step)) {
                        failure =
                            "kinematic-area-gradient Young wall recovery could not choose a topology-preserving derivative step";
                        return;
                    }
                    std::array<Real, 6> measures{};
                    const std::array<Real, 6> offsets{{
                        Real{-2.0},
                        Real{-1.0},
                        Real{-0.5},
                        Real{0.5},
                        Real{1.0},
                        Real{2.0}}};
                    for (std::size_t sample = 0;
                         sample < offsets.size();
                         ++sample) {
                        auto perturbed_values = signed_values;
                        perturbed_values[local] +=
                            offsets[sample] * step;
                        bool measure_success = false;
                        measures[sample] = activeBoundaryMeasure(
                            std::span<const std::array<Real, 3>>(
                                coordinates.data(), face_corner_count),
                            std::span<const Real>(
                                perturbed_values.data(),
                                face_corner_count),
                            options
                                .kinematic_area_gradient_negative_liquid_side,
                            measure_success);
                        ++result
                              .kinematic_area_gradient_young_wall_measure_evaluations;
                        if (!measure_success ||
                            !(measures[sample] > Real{0.0})) {
                            failure =
                                "kinematic-area-gradient Young wall derivative left the fixed contact topology";
                            return;
                        }
                    }
                    const Real fourth_order_full =
                        (measures[0] - Real{8.0} * measures[1] +
                         Real{8.0} * measures[4] - measures[5]) /
                        (Real{12.0} * step);
                    const Real fourth_order_half =
                        (measures[1] - Real{8.0} * measures[2] +
                         Real{8.0} * measures[3] - measures[4]) /
                        (Real{6.0} * step);
                    const Real richardson =
                        (Real{16.0} * fourth_order_half -
                         fourth_order_full) /
                        Real{15.0};
                    if (!std::isfinite(richardson)) {
                        failure =
                            "kinematic-area-gradient Young wall recovery produced a nonfinite derivative";
                        return;
                    }
                    Real measure_scale{0.0};
                    for (const auto measure : measures) {
                        measure_scale =
                            std::max(measure_scale, std::abs(measure));
                    }
                    const Real derivative_scale = std::max(
                        std::abs(local_wall_gradient[local]),
                        measure_scale /
                            std::max(face_value_scale,
                                     std::numeric_limits<Real>::min()));
                    const Real disagreement = std::max(
                        std::abs(fourth_order_half - fourth_order_full),
                        std::abs(richardson -
                                 local_wall_gradient[local])) /
                        std::max(derivative_scale,
                                 std::numeric_limits<Real>::min());
                    result
                        .kinematic_area_gradient_max_relative_fd_disagreement =
                        std::max(
                            result
                                .kinematic_area_gradient_max_relative_fd_disagreement,
                            disagreement);
                    const auto node = static_cast<std::size_t>(
                        nodes[face_corners[local]]);
                    young_wall_gradient[node] +=
                        coefficient * local_wall_gradient[local];
                }
            });
        if (!failure.empty()) {
            result.diagnostic = std::move(failure);
            return false;
        }
        if (wall_boundary_faces == 0u) {
            result.diagnostic =
                "kinematic-area-gradient Young wall marker has no owned boundary faces";
            return false;
        }
    }

    Real surface_gradient_norm2{0.0};
    Real young_wall_gradient_norm2{0.0};
    Real total_gradient_norm2{0.0};
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        surface_gradient_norm2 +=
            area_gradient[vertex] * area_gradient[vertex];
        young_wall_gradient_norm2 +=
            young_wall_gradient[vertex] * young_wall_gradient[vertex];
        area_gradient[vertex] += young_wall_gradient[vertex];
        total_gradient_norm2 +=
            area_gradient[vertex] * area_gradient[vertex];
    }
    result.kinematic_area_gradient_surface_gradient_norm =
        std::sqrt(surface_gradient_norm2);
    result.kinematic_area_gradient_young_wall_gradient_norm =
        std::sqrt(young_wall_gradient_norm2);
    result.kinematic_area_gradient_total_energy_gradient_norm =
        std::sqrt(total_gradient_norm2);

    std::vector<Real> rhs(n_vertices, Real{0.0});
    std::vector<std::size_t> active_degree(n_vertices, 0u);
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        const Real mass = lumped_kinematic_mass[vertex];
        if (mass == Real{0.0}) {
            continue;
        }
        if (!(mass > Real{0.0}) || !std::isfinite(mass) ||
            !std::isfinite(area_gradient[vertex])) {
            result.diagnostic =
                "kinematic-area-gradient curvature recovery produced an invalid assembled operator row";
            return false;
        }
        rhs[vertex] = -area_gradient[vertex];
        ++result.kinematic_area_gradient_operator_vertices;
        result.kinematic_area_gradient_operator_nonzeros +=
            kinematic_mass[vertex].size();
        for (const auto& [column, value] : kinematic_mass[vertex]) {
            if (column != vertex && value != Real{0.0} &&
                column < n_vertices &&
                lumped_kinematic_mass[column] > Real{0.0}) {
                ++active_degree[vertex];
            }
        }
    }

    std::vector<Real> solved_curvature(n_vertices, Real{0.0});
    auto regularized_mass = kinematic_mass;
    struct FilterComponent {
        Real interface_measure{0.0};
        Real edge_length2_sum{0.0};
        std::size_t edge_count{0u};
        Real characteristic_radius{0.0};
        Real mean_edge_length{0.0};
        Real filter_radius{0.0};
        Real filter_strength{0.0};
    };
    const auto no_component = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> component_ids(n_vertices, no_component);
    std::vector<FilterComponent> components;
    std::queue<std::size_t> pending_vertices;
    for (std::size_t seed = 0; seed < n_vertices; ++seed) {
        if (lumped_kinematic_mass[seed] == Real{0.0} ||
            component_ids[seed] != no_component) {
            continue;
        }
        const auto component = components.size();
        components.emplace_back();
        component_ids[seed] = component;
        pending_vertices.push(seed);
        while (!pending_vertices.empty()) {
            const auto vertex = pending_vertices.front();
            pending_vertices.pop();
            const Real measure = lumped_interface_measure[vertex];
            if (!(measure > Real{0.0}) || !std::isfinite(measure)) {
                result.diagnostic =
                    "kinematic-area-gradient curvature recovery produced an invalid component measure";
                return false;
            }
            components[component].interface_measure += measure;
            for (const auto& [column, value] : kinematic_mass[vertex]) {
                if (value == Real{0.0} || column == vertex ||
                    column >= n_vertices ||
                    lumped_kinematic_mass[column] == Real{0.0}) {
                    continue;
                }
                if (component_ids[column] == no_component) {
                    component_ids[column] = component;
                    pending_vertices.push(column);
                } else if (component_ids[column] != component) {
                    result.diagnostic =
                        "kinematic-area-gradient curvature recovery found inconsistent component connectivity";
                    return false;
                }
            }
        }
    }
    if (components.empty()) {
        result.diagnostic =
            "kinematic-area-gradient curvature recovery found no active interface components";
        return false;
    }

    for (std::size_t row = 0; row < n_vertices; ++row) {
        if (lumped_kinematic_mass[row] == Real{0.0}) {
            continue;
        }
        const auto component = component_ids[row];
        if (component >= components.size()) {
            result.diagnostic =
                "kinematic-area-gradient curvature recovery found an unassigned active vertex";
            return false;
        }
        const auto x =
            mesh.getNodeCoordinates(static_cast<GlobalIndex>(row));
        for (const auto& [column, value] : kinematic_mass[row]) {
            if (value == Real{0.0} || column <= row ||
                column >= n_vertices ||
                lumped_kinematic_mass[column] == Real{0.0}) {
                continue;
            }
            if (component_ids[column] != component) {
                result.diagnostic =
                    "kinematic-area-gradient curvature recovery found a cross-component operator edge";
                return false;
            }
            const auto y =
                mesh.getNodeCoordinates(static_cast<GlobalIndex>(column));
            const Real edge_length2 =
                dot(subtract(y, x), subtract(y, x));
            if (!(edge_length2 > Real{0.0}) ||
                !std::isfinite(edge_length2)) {
                result.diagnostic =
                    "kinematic-area-gradient curvature recovery found an invalid active edge";
                return false;
            }
            components[component].edge_length2_sum += edge_length2;
            ++components[component].edge_count;
        }
    }

    const Real pi = std::acos(Real{-1.0});
    result.kinematic_area_gradient_components = components.size();
    result.kinematic_area_gradient_filter_coefficient =
        options.kinematic_area_gradient_filter_coefficient;
    result.kinematic_area_gradient_min_characteristic_radius =
        std::numeric_limits<Real>::infinity();
    result.kinematic_area_gradient_min_filter_radius_cells =
        std::numeric_limits<Real>::infinity();
    result.kinematic_area_gradient_min_filter_radius =
        std::numeric_limits<Real>::infinity();
    for (auto& component : components) {
        if (component.edge_count == 0u ||
            !(component.interface_measure > Real{0.0}) ||
            !std::isfinite(component.interface_measure)) {
            result.diagnostic =
                "kinematic-area-gradient curvature recovery found an invalid interface component";
            return false;
        }
        component.mean_edge_length = std::sqrt(
            component.edge_length2_sum /
            static_cast<Real>(component.edge_count));
        component.characteristic_radius = dimension == 2
            ? component.interface_measure / (Real{2.0} * pi)
            : std::sqrt(component.interface_measure / (Real{4.0} * pi));
        component.filter_radius =
            options.kinematic_area_gradient_filter_coefficient *
            std::sqrt(component.mean_edge_length *
                      component.characteristic_radius);
        component.filter_strength =
            component.filter_radius * component.filter_radius;
        if (!(component.mean_edge_length > Real{0.0}) ||
            !(component.characteristic_radius > Real{0.0}) ||
            !std::isfinite(component.mean_edge_length) ||
            !std::isfinite(component.characteristic_radius) ||
            !std::isfinite(component.filter_radius)) {
            result.diagnostic =
                "kinematic-area-gradient curvature recovery produced an invalid component filter scale";
            return false;
        }
        const Real filter_radius_cells =
            component.filter_radius / component.mean_edge_length;
        result.kinematic_area_gradient_interface_measure +=
            component.interface_measure;
        result.kinematic_area_gradient_min_characteristic_radius = std::min(
            result.kinematic_area_gradient_min_characteristic_radius,
            component.characteristic_radius);
        result.kinematic_area_gradient_max_characteristic_radius = std::max(
            result.kinematic_area_gradient_max_characteristic_radius,
            component.characteristic_radius);
        result.kinematic_area_gradient_min_filter_radius_cells = std::min(
            result.kinematic_area_gradient_min_filter_radius_cells,
            filter_radius_cells);
        result.kinematic_area_gradient_max_filter_radius_cells = std::max(
            result.kinematic_area_gradient_max_filter_radius_cells,
            filter_radius_cells);
        result.kinematic_area_gradient_min_filter_radius = std::min(
            result.kinematic_area_gradient_min_filter_radius,
            component.filter_radius);
        result.kinematic_area_gradient_max_filter_radius = std::max(
            result.kinematic_area_gradient_max_filter_radius,
            component.filter_radius);
    }

    if (options.kinematic_area_gradient_filter_coefficient > Real{0.0}) {
        for (std::size_t row = 0; row < n_vertices; ++row) {
            if (lumped_kinematic_mass[row] == Real{0.0}) {
                continue;
            }
            const auto x =
                mesh.getNodeCoordinates(static_cast<GlobalIndex>(row));
            for (const auto& [column, value] : kinematic_mass[row]) {
                if (value == Real{0.0} || column <= row ||
                    column >= n_vertices ||
                    lumped_kinematic_mass[column] == Real{0.0}) {
                    continue;
                }
                const auto y =
                    mesh.getNodeCoordinates(static_cast<GlobalIndex>(column));
                const Real edge_length2 =
                    dot(subtract(y, x), subtract(y, x));
                const Real row_share =
                    lumped_kinematic_mass[row] /
                    static_cast<Real>(std::max<std::size_t>(
                        1u, active_degree[row]));
                const Real column_share =
                    lumped_kinematic_mass[column] /
                    static_cast<Real>(std::max<std::size_t>(
                        1u, active_degree[column]));
                const Real weight =
                    Real{0.5} * (row_share + column_share) /
                    edge_length2;
                if (!(weight > Real{0.0}) || !std::isfinite(weight)) {
                    result.diagnostic =
                        "kinematic-area-gradient curvature recovery produced an invalid filter weight";
                    return false;
                }
                const Real filter_strength =
                    components[component_ids[row]].filter_strength;
                regularized_mass[row][row] += filter_strength * weight;
                regularized_mass[column][column] += filter_strength * weight;
                regularized_mass[row][column] -= filter_strength * weight;
                regularized_mass[column][row] -= filter_strength * weight;
            }
        }
    }
    result.kinematic_area_gradient_minimum_norm_solver =
        options.kinematic_area_gradient_filter_coefficient == Real{0.0};
    const bool mass_system_solved =
        result.kinematic_area_gradient_minimum_norm_solver
        ? solveKinematicAreaMassMinimumNorm(
              regularized_mass,
              std::span<const Real>(rhs.data(), rhs.size()),
              solved_curvature,
              result.kinematic_area_gradient_linear_iterations,
              result.kinematic_area_gradient_relative_linear_residual)
        : solveKinematicAreaMassSystem(
              regularized_mass,
              std::span<const Real>(rhs.data(), rhs.size()),
              std::span<const Real>{},
              /*use_diagonal_preconditioner=*/true,
              solved_curvature,
              result.kinematic_area_gradient_linear_iterations,
              result.kinematic_area_gradient_relative_linear_residual);
    if (!mass_system_solved) {
        result.diagnostic =
            "kinematic-area-gradient curvature recovery could not solve its consistent interface system";
        return false;
    }

    Real mass_weighted_curvature{0.0};
    Real mass_weighted_curvature_squared{0.0};
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        if (lumped_kinematic_mass[vertex] == Real{0.0}) {
            continue;
        }
        const Real kappa = solved_curvature[vertex];
        if (!std::isfinite(kappa)) {
            result.diagnostic =
                "kinematic-area-gradient curvature recovery produced nonfinite curvature";
            return false;
        }
        curvature[vertex] = kappa;
        result.kinematic_area_gradient_kinematic_mass +=
            lumped_kinematic_mass[vertex];
        mass_weighted_curvature +=
            lumped_kinematic_mass[vertex] * kappa;
        for (const auto& [column, value] : kinematic_mass[vertex]) {
            mass_weighted_curvature_squared +=
                kappa * value * solved_curvature[column];
        }
        active_vertices[vertex] = 1u;
        fitted[vertex] = 1u;
        ++result.fitted_vertices;
        Real applied{0.0};
        Real identity_scale = std::abs(area_gradient[vertex]);
        for (const auto& [column, value] : regularized_mass[vertex]) {
            applied += value * solved_curvature[column];
            identity_scale += std::abs(value * solved_curvature[column]);
        }
        const Real identity_residual =
            std::abs(area_gradient[vertex] + applied);
        result.kinematic_area_gradient_max_regularized_identity_residual =
            std::max(
                result.kinematic_area_gradient_max_regularized_identity_residual,
                identity_residual);
        result
            .kinematic_area_gradient_max_relative_regularized_identity_residual =
            std::max(
                result
                    .kinematic_area_gradient_max_relative_regularized_identity_residual,
                identity_residual /
                    std::max(identity_scale,
                             std::numeric_limits<Real>::min()));
    }
    if (!(result.kinematic_area_gradient_kinematic_mass > Real{0.0}) ||
        !std::isfinite(result.kinematic_area_gradient_kinematic_mass)) {
        result.diagnostic =
            "kinematic-area-gradient curvature recovery produced an invalid global kinematic mass";
        return false;
    }
    result.kinematic_area_gradient_mass_weighted_mean_curvature =
        mass_weighted_curvature /
        result.kinematic_area_gradient_kinematic_mass;
    const Real mass_weighted_second_moment =
        mass_weighted_curvature_squared /
        result.kinematic_area_gradient_kinematic_mass;
    result.kinematic_area_gradient_mass_weighted_rms_deviation = std::sqrt(
        std::max(
            mass_weighted_second_moment -
                result.kinematic_area_gradient_mass_weighted_mean_curvature *
                    result.kinematic_area_gradient_mass_weighted_mean_curvature,
            Real{0.0}));
    if (!std::isfinite(
            result.kinematic_area_gradient_mass_weighted_mean_curvature) ||
        !std::isfinite(
            result.kinematic_area_gradient_mass_weighted_rms_deviation)) {
        result.diagnostic =
            "kinematic-area-gradient curvature recovery produced invalid weighted statistics";
        return false;
    }
    result.narrow_band_vertices = result.fitted_vertices;
    result.skipped_far_vertices = n_vertices - result.fitted_vertices;
    return result.fitted_vertices > 0u;
}

[[nodiscard]] bool recoverGeneratedInterfacePatchCurvature(
    const std::array<Real, 3>& center,
    const std::array<Real, 3>& level_set_gradient,
    int dim,
    std::span<const std::size_t> sample_indices,
    std::span<const LevelSetCurvatureProjectionSample> samples,
    Real relative_rank_tolerance,
    Real gradient_tolerance,
    Real& curvature,
    FitResidualMetrics& residual,
    std::size_t& geometry_sample_count)
{
    geometry_sample_count = 0u;
    const Real gradient_norm = norm(level_set_gradient);
    if (!(gradient_norm > gradient_tolerance) ||
        !std::isfinite(gradient_norm)) {
        return false;
    }
    std::array<Real, 3> normal{
        level_set_gradient[0] / gradient_norm,
        level_set_gradient[1] / gradient_norm,
        dim == 2 ? Real{0.0}
                 : level_set_gradient[2] / gradient_norm,
    };

    std::array<Real, 3> tangent0{};
    std::array<Real, 3> tangent1{};
    if (dim == 2) {
        tangent0 = {{-normal[1], normal[0], Real{0.0}}};
    } else {
        const std::array<Real, 3> axis =
            std::abs(normal[0]) <= std::abs(normal[1]) &&
                    std::abs(normal[0]) <= std::abs(normal[2])
                ? std::array<Real, 3>{{Real{1.0}, Real{0.0}, Real{0.0}}}
                : (std::abs(normal[1]) <= std::abs(normal[2])
                       ? std::array<Real, 3>{{Real{0.0}, Real{1.0}, Real{0.0}}}
                       : std::array<Real, 3>{{Real{0.0}, Real{0.0}, Real{1.0}}});
        tangent0 = cross(axis, normal);
        const Real tangent0_norm = norm(tangent0);
        if (!(tangent0_norm > gradient_tolerance) ||
            !std::isfinite(tangent0_norm)) {
            return false;
        }
        for (auto& value : tangent0) {
            value /= tangent0_norm;
        }
        tangent1 = cross(normal, tangent0);
    }

    struct PatchPoint {
        Real u{0.0};
        Real v{0.0};
        Real w{0.0};
    };
    std::vector<PatchPoint> points;
    points.reserve(sample_indices.size());
    for (const auto sample_index : sample_indices) {
        if (sample_index >= samples.size() ||
            !samples[sample_index].generated_interface_geometry) {
            continue;
        }
        const auto displacement =
            subtract(samples[sample_index].coordinate, center);
        const PatchPoint point{
            dot(displacement, tangent0),
            dim == 2 ? Real{0.0} : dot(displacement, tangent1),
            dot(displacement, normal),
        };
        if (!std::isfinite(point.u) || !std::isfinite(point.v) ||
            !std::isfinite(point.w)) {
            continue;
        }
        points.push_back(point);
    }
    geometry_sample_count = points.size();
    const std::size_t coefficient_count = dim == 2 ? 3u : 6u;
    if (points.size() < coefficient_count) {
        return false;
    }

    Real scale_u = Real{0.0};
    Real scale_v = dim == 2 ? Real{1.0} : Real{0.0};
    for (const auto& point : points) {
        scale_u = std::max(scale_u, std::abs(point.u));
        if (dim == 3) {
            scale_v = std::max(scale_v, std::abs(point.v));
        }
    }
    if (!(scale_u > gradient_tolerance) || !std::isfinite(scale_u) ||
        !(scale_v > gradient_tolerance) || !std::isfinite(scale_v)) {
        return false;
    }

    std::vector<FitObservation> observations;
    observations.reserve(points.size());
    // A five-to-one decay at half the normalized patch radius limits the
    // truncation bias from outer generated segments while retaining their
    // rank information in the quadratic curve fit.
    constexpr Real generated_curve_patch_distance_decay = Real{16.0};
    for (const auto& point : points) {
        const Real u = point.u / scale_u;
        const Real v = dim == 2 ? Real{0.0} : point.v / scale_v;
        FitObservation observation;
        observation.row[0] = Real{1.0};
        observation.row[1] = u;
        if (dim == 2) {
            observation.row[2] = u * u;
        } else {
            observation.row[2] = v;
            observation.row[3] = u * u;
            observation.row[4] = u * v;
            observation.row[5] = v * v;
        }
        observation.rhs = point.w;
        observation.weight = Real{1.0};
        if (dim == 2) {
            observation.weight =
                Real{1.0} /
                (Real{1.0} +
                 generated_curve_patch_distance_decay * u * u);
        }
        observations.push_back(observation);
    }

    std::array<Real, 9> coefficients{};
    if (!solveWeightedLeastSquares(
            observations,
            coefficient_count,
            relative_rank_tolerance,
            coefficients)) {
        return false;
    }
    residual = computeFitResidualMetrics(
        observations,
        coefficients,
        coefficient_count,
        gradient_tolerance * std::max(scale_u, scale_v));
    if (!std::isfinite(residual.rms) ||
        !std::isfinite(residual.normalized)) {
        return false;
    }

    if (dim == 2) {
        const Real slope = coefficients[1] / scale_u;
        const Real second =
            Real{2.0} * coefficients[2] / (scale_u * scale_u);
        const Real denominator =
            std::pow(Real{1.0} + slope * slope, Real{1.5});
        curvature = -second / denominator;
        return std::isfinite(curvature);
    }

    const Real slope_u = coefficients[1] / scale_u;
    const Real slope_v = coefficients[2] / scale_v;
    const Real second_uu =
        Real{2.0} * coefficients[3] / (scale_u * scale_u);
    const Real second_uv = coefficients[4] / (scale_u * scale_v);
    const Real second_vv =
        Real{2.0} * coefficients[5] / (scale_v * scale_v);
    const Real slope_norm2 = slope_u * slope_u + slope_v * slope_v;
    const Real numerator =
        (Real{1.0} + slope_v * slope_v) * second_uu -
        Real{2.0} * slope_u * slope_v * second_uv +
        (Real{1.0} + slope_u * slope_u) * second_vv;
    curvature =
        -numerator / std::pow(Real{1.0} + slope_norm2, Real{1.5});
    return std::isfinite(curvature);
}

[[nodiscard]] Real tripleProduct(const std::array<Real, 3>& a,
                                 const std::array<Real, 3>& b,
                                 const std::array<Real, 3>& c) noexcept
{
    return dot(a, cross(b, c));
}

[[nodiscard]] Real triangleMeasure(const std::array<Real, 3>& a,
                                   const std::array<Real, 3>& b,
                                   const std::array<Real, 3>& c) noexcept
{
    return Real{0.5} * norm(cross(subtract(b, a), subtract(c, a)));
}

[[nodiscard]] Real tetraMeasure(const std::array<Real, 3>& a,
                                const std::array<Real, 3>& b,
                                const std::array<Real, 3>& c,
                                const std::array<Real, 3>& d) noexcept
{
    return std::abs(
               tripleProduct(subtract(b, a), subtract(c, a), subtract(d, a))) /
           Real{6.0};
}

[[nodiscard]] std::size_t primaryVertexCount(ElementType type,
                                             std::size_t nodes) noexcept
{
    switch (type) {
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return std::min<std::size_t>(3u, nodes);
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
            return std::min<std::size_t>(4u, nodes);
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return std::min<std::size_t>(4u, nodes);
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return std::min<std::size_t>(8u, nodes);
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            return std::min<std::size_t>(6u, nodes);
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return std::min<std::size_t>(5u, nodes);
        default:
            return nodes;
    }
}

[[nodiscard]] Real estimateCellMeasure(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell,
    std::span<const GlobalIndex> nodes,
    int dim)
{
    const auto primary =
        primaryVertexCount(mesh.getCellType(cell), nodes.size());
    if (primary == 0u) {
        return Real{0.0};
    }

    std::vector<std::array<Real, 3>> x(primary);
    for (std::size_t i = 0; i < primary; ++i) {
        if (nodes[i] < 0) {
            return Real{0.0};
        }
        x[i] = mesh.getNodeCoordinates(nodes[i]);
    }

    Real measure = Real{0.0};
    if (dim == 2) {
        if (primary == 3u) {
            measure = triangleMeasure(x[0], x[1], x[2]);
        } else if (primary >= 4u) {
            for (std::size_t i = 1u; i + 1u < primary; ++i) {
                measure += triangleMeasure(x[0], x[i], x[i + 1u]);
            }
        }
    } else {
        if (primary == 4u) {
            measure = tetraMeasure(x[0], x[1], x[2], x[3]);
        } else if (primary >= 8u) {
            measure += tetraMeasure(x[0], x[1], x[3], x[4]);
            measure += tetraMeasure(x[1], x[2], x[3], x[6]);
            measure += tetraMeasure(x[1], x[4], x[5], x[6]);
            measure += tetraMeasure(x[3], x[4], x[6], x[7]);
            measure += tetraMeasure(x[1], x[3], x[4], x[6]);
        } else if (primary >= 5u) {
            for (std::size_t i = 1u; i + 2u < primary; ++i) {
                measure += tetraMeasure(x[0], x[i], x[i + 1u], x[i + 2u]);
            }
        }
    }
    return std::isfinite(measure) && measure > Real{0.0}
        ? measure
        : Real{0.0};
}

void assembleLumpedMass(
    const assembly::IMeshAccess& mesh,
    std::span<const unsigned char> active_vertices,
    std::vector<Real>& mass)
{
    std::fill(mass.begin(), mass.end(), Real{0.0});
    std::vector<GlobalIndex> nodes;
    mesh.forEachCell([&](GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        const Real measure =
            estimateCellMeasure(mesh,
                                cell,
                                std::span<const GlobalIndex>(nodes.data(),
                                                             nodes.size()),
                                mesh.dimension());
        if (!(measure > Real{0.0})) {
            return;
        }

        std::size_t active_count = 0u;
        for (const auto node : nodes) {
            const auto index = static_cast<std::size_t>(node);
            if (node >= 0 && index < mass.size() &&
                active_vertices[index] != 0u) {
                ++active_count;
            }
        }
        if (active_count == 0u) {
            return;
        }
        const Real lump = measure / static_cast<Real>(active_count);
        for (const auto node : nodes) {
            const auto index = static_cast<std::size_t>(node);
            if (node >= 0 && index < mass.size() &&
                active_vertices[index] != 0u) {
                mass[index] += lump;
            }
        }
    });

    for (std::size_t vertex = 0; vertex < mass.size(); ++vertex) {
        if (active_vertices[vertex] != 0u &&
            (!(mass[vertex] > Real{0.0}) || !std::isfinite(mass[vertex]))) {
            mass[vertex] = Real{1.0};
        }
    }
}

[[nodiscard]] Real vectorNorm2(std::span<const Real> values) noexcept
{
    Real sum = Real{0.0};
    for (const auto value : values) {
        sum += value * value;
    }
    return sum;
}

void applyMassStiffnessOperator(
    std::span<const Real> mass,
    const std::vector<std::vector<WeightedNeighbor>>& stiffness,
    std::span<const Real> stiffness_diag,
    Real strength,
    std::span<const Real> x,
    std::vector<Real>& y)
{
    y.assign(x.size(), Real{0.0});
    for (std::size_t row = 0; row < x.size(); ++row) {
        Real value =
            mass[row] * x[row] + strength * stiffness_diag[row] * x[row];
        for (const auto& entry : stiffness[row]) {
            value -= strength * entry.weight * x[entry.vertex];
        }
        y[row] = value;
    }
}

[[nodiscard]] bool solveMassStiffnessOperatorCG(
    std::span<const Real> mass,
    const std::vector<std::vector<WeightedNeighbor>>& stiffness,
    std::span<const Real> stiffness_diag,
    Real strength,
    std::span<const Real> rhs,
    std::span<const Real> initial_guess,
    std::vector<Real>& solution)
{
    const auto n = rhs.size();
    solution.assign(initial_guess.begin(), initial_guess.end());
    std::vector<Real> applied;
    std::vector<Real> residual(n, Real{0.0});
    std::vector<Real> direction(n, Real{0.0});
    std::vector<Real> next_applied;

    applyMassStiffnessOperator(
        mass, stiffness, stiffness_diag, strength, solution, applied);
    for (std::size_t i = 0; i < n; ++i) {
        residual[i] = rhs[i] - applied[i];
        direction[i] = residual[i];
    }

    Real rr = vectorNorm2(residual);
    const Real rhs_norm = std::sqrt(std::max(vectorNorm2(rhs), Real{0.0}));
    const Real tolerance =
        Real{1.0e-12} * std::max(rhs_norm, Real{1.0});
    if (std::sqrt(std::max(rr, Real{0.0})) <= tolerance) {
        return true;
    }

    const std::size_t max_iterations =
        std::max<std::size_t>(100u, 4u * std::max<std::size_t>(1u, n));
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        applyMassStiffnessOperator(
            mass, stiffness, stiffness_diag, strength, direction, next_applied);
        Real p_ap = Real{0.0};
        for (std::size_t i = 0; i < n; ++i) {
            p_ap += direction[i] * next_applied[i];
        }
        if (!(p_ap > Real{0.0}) || !std::isfinite(p_ap)) {
            return false;
        }
        const Real alpha = rr / p_ap;
        for (std::size_t i = 0; i < n; ++i) {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * next_applied[i];
        }
        const Real rr_next = vectorNorm2(residual);
        if (!std::isfinite(rr_next)) {
            return false;
        }
        if (std::sqrt(std::max(rr_next, Real{0.0})) <= tolerance) {
            return true;
        }
        const Real beta = rr_next / rr;
        for (std::size_t i = 0; i < n; ++i) {
            direction[i] = residual[i] + beta * direction[i];
        }
        rr = rr_next;
    }
    return false;
}

void smoothCurvatureOnVertexGraph(
    const std::vector<std::vector<GlobalIndex>>& adjacency,
    std::span<const unsigned char> active_vertices,
    int iterations,
    Real relaxation,
    std::vector<Real>& curvature,
    LevelSetCurvatureProjectionResult& result)
{
    if (iterations <= 0 || !(relaxation > Real{0.0})) {
        return;
    }

    std::vector<Real> current = curvature;
    std::vector<Real> next = current;
    Real total_abs_update = Real{0.0};
    std::size_t update_count = 0u;

    for (int iter = 0; iter < iterations; ++iter) {
        Real iteration_max_update = Real{0.0};
        for (std::size_t vertex = 0; vertex < current.size(); ++vertex) {
            if (!active_vertices.empty() && active_vertices[vertex] == 0u) {
                next[vertex] = current[vertex];
                continue;
            }
            const auto& neighbors = adjacency[vertex];
            if (neighbors.empty()) {
                next[vertex] = current[vertex];
                continue;
            }

            Real sum = Real{0.0};
            std::size_t count = 0u;
            for (const auto neighbor : neighbors) {
                const auto index = static_cast<std::size_t>(neighbor);
                if (index >= current.size() ||
                    (!active_vertices.empty() &&
                     active_vertices[index] == 0u) ||
                    !std::isfinite(current[index])) {
                    continue;
                }
                sum += current[index];
                ++count;
            }
            if (count == 0u || !std::isfinite(current[vertex])) {
                next[vertex] = current[vertex];
                continue;
            }

            const Real average = sum / static_cast<Real>(count);
            next[vertex] =
                current[vertex] + relaxation * (average - current[vertex]);
            if (!std::isfinite(next[vertex])) {
                next[vertex] = current[vertex];
                continue;
            }
            const Real update = std::abs(next[vertex] - current[vertex]);
            total_abs_update += update;
            iteration_max_update = std::max(iteration_max_update, update);
            ++update_count;
        }
        current.swap(next);
        result.smoothing_max_abs_update =
            std::max(result.smoothing_max_abs_update, iteration_max_update);
        ++result.smoothing_iterations_applied;
    }

    if (update_count > 0u) {
        result.smoothing_mean_abs_update =
            total_abs_update / static_cast<Real>(update_count);
    }
    curvature = std::move(current);
}

void smoothCurvatureWithMassStiffnessOperator(
    const assembly::IMeshAccess& mesh,
    const std::vector<std::vector<GlobalIndex>>& adjacency,
    std::span<const unsigned char> active_vertices,
    int iterations,
    Real relaxation,
    std::vector<Real>& curvature,
    LevelSetCurvatureProjectionResult& result)
{
    if (iterations <= 0 || !(relaxation > Real{0.0})) {
        return;
    }

    const auto n_vertices = curvature.size();
    std::vector<std::size_t> active;
    active.reserve(n_vertices);
    std::vector<std::size_t> local_index(n_vertices,
                                         std::numeric_limits<std::size_t>::max());
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        if ((active_vertices.empty() || active_vertices[vertex] != 0u) &&
            std::isfinite(curvature[vertex])) {
            local_index[vertex] = active.size();
            active.push_back(vertex);
        }
    }
    if (active.size() <= 1u) {
        return;
    }

    std::vector<Real> full_mass(n_vertices, Real{0.0});
    assembleLumpedMass(mesh, active_vertices, full_mass);
    std::vector<std::size_t> degree(n_vertices, 0u);
    for (const auto vertex : active) {
        for (const auto neighbor : adjacency[vertex]) {
            const auto neighbor_index = static_cast<std::size_t>(neighbor);
            if (neighbor >= 0 && neighbor_index < n_vertices &&
                local_index[neighbor_index] !=
                    std::numeric_limits<std::size_t>::max()) {
                ++degree[vertex];
            }
        }
    }

    std::vector<Real> mass(active.size(), Real{1.0});
    for (std::size_t i = 0; i < active.size(); ++i) {
        mass[i] = full_mass[active[i]];
    }

    std::vector<std::vector<WeightedNeighbor>> stiffness(active.size());
    std::vector<Real> stiffness_diag(active.size(), Real{0.0});
    Real edge_length2_sum = Real{0.0};
    std::size_t edge_count = 0u;
    for (const auto vertex : active) {
        const auto row = local_index[vertex];
        const auto x = mesh.getNodeCoordinates(static_cast<GlobalIndex>(vertex));
        for (const auto neighbor : adjacency[vertex]) {
            if (neighbor < 0) {
                continue;
            }
            const auto neighbor_index = static_cast<std::size_t>(neighbor);
            if (neighbor_index <= vertex || neighbor_index >= n_vertices) {
                continue;
            }
            const auto col = local_index[neighbor_index];
            if (col == std::numeric_limits<std::size_t>::max()) {
                continue;
            }
            const auto y =
                mesh.getNodeCoordinates(static_cast<GlobalIndex>(neighbor_index));
            const Real edge_length2 = dot(subtract(y, x), subtract(y, x));
            if (!(edge_length2 > Real{0.0}) ||
                !std::isfinite(edge_length2)) {
                continue;
            }
            const Real vertex_share =
                mass[row] / static_cast<Real>(std::max<std::size_t>(
                                1u, degree[vertex]));
            const Real neighbor_share =
                mass[col] / static_cast<Real>(std::max<std::size_t>(
                                1u, degree[neighbor_index]));
            const Real weight =
                Real{0.5} * (vertex_share + neighbor_share) / edge_length2;
            if (!(weight > Real{0.0}) || !std::isfinite(weight)) {
                continue;
            }
            stiffness[row].push_back(WeightedNeighbor{col, weight});
            stiffness[col].push_back(WeightedNeighbor{row, weight});
            stiffness_diag[row] += weight;
            stiffness_diag[col] += weight;
            edge_length2_sum += edge_length2;
            ++edge_count;
        }
    }
    result.smoothing_operator_edges = edge_count;
    if (edge_count == 0u) {
        return;
    }

    const Real mean_edge_length2 =
        edge_length2_sum / static_cast<Real>(edge_count);
    const Real strength = relaxation * mean_edge_length2;
    if (!(strength > Real{0.0}) || !std::isfinite(strength)) {
        return;
    }

    std::vector<Real> current(active.size(), Real{0.0});
    for (std::size_t i = 0; i < active.size(); ++i) {
        current[i] = curvature[active[i]];
    }

    std::vector<Real> rhs(active.size(), Real{0.0});
    std::vector<Real> next(active.size(), Real{0.0});
    Real total_abs_update = Real{0.0};
    std::size_t update_count = 0u;
    for (int iter = 0; iter < iterations; ++iter) {
        for (std::size_t i = 0; i < active.size(); ++i) {
            rhs[i] = mass[i] * current[i];
        }
        if (!solveMassStiffnessOperatorCG(
                std::span<const Real>(mass.data(), mass.size()),
                stiffness,
                std::span<const Real>(stiffness_diag.data(),
                                      stiffness_diag.size()),
                strength,
                std::span<const Real>(rhs.data(), rhs.size()),
                std::span<const Real>(current.data(), current.size()),
                next)) {
            break;
        }

        Real iteration_max_update = Real{0.0};
        for (std::size_t i = 0; i < active.size(); ++i) {
            if (!std::isfinite(next[i])) {
                next[i] = current[i];
            }
            const Real update = std::abs(next[i] - current[i]);
            total_abs_update += update;
            iteration_max_update = std::max(iteration_max_update, update);
            ++update_count;
        }
        current.swap(next);
        result.smoothing_max_abs_update =
            std::max(result.smoothing_max_abs_update, iteration_max_update);
        ++result.smoothing_iterations_applied;
    }

    if (update_count > 0u) {
        result.smoothing_mean_abs_update =
            total_abs_update / static_cast<Real>(update_count);
    }
    for (std::size_t i = 0; i < active.size(); ++i) {
        curvature[active[i]] = current[i];
    }
}

} // namespace

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace* workspace);

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values)
{
    return projectLevelSetMeanCurvatureToVertices(
        mesh,
        level_set_vertex_values,
        std::span<const LevelSetCurvatureProjectionSample>{},
        options,
        curvature_vertex_values);
}

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values)
{
    return projectLevelSetMeanCurvatureToVertices(
        mesh,
        level_set_vertex_values,
        supplemental_samples,
        options,
        curvature_vertex_values,
        /*workspace=*/nullptr);
}

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace& workspace)
{
    return projectLevelSetMeanCurvatureToVertices(
        mesh,
        level_set_vertex_values,
        supplemental_samples,
        options,
        curvature_vertex_values,
        &workspace);
}

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace* workspace)
{
    const auto n_vertices = static_cast<std::size_t>(mesh.numVertices());
    if (level_set_vertex_values.size() != n_vertices) {
        throw std::invalid_argument(
            "level-set curvature projection requires one level-set value per mesh vertex");
    }
    const int dim = mesh.dimension();
    if (dim != 2 && dim != 3) {
        throw std::invalid_argument(
            "level-set curvature projection supports two- and three-dimensional meshes");
    }
    if (!(options.gradient_tolerance > Real{0.0}) ||
        !std::isfinite(options.gradient_tolerance) ||
        !(options.normal_equation_tolerance > Real{0.0}) ||
        !(options.normal_equation_tolerance < Real{1.0}) ||
        !std::isfinite(options.normal_equation_tolerance) ||
        options.max_normalized_fit_residual < Real{0.0} ||
        !std::isfinite(options.max_normalized_fit_residual) ||
        !(options.supplemental_sample_weight > Real{0.0}) ||
        !std::isfinite(options.supplemental_sample_weight) ||
        options.kinematic_area_gradient_filter_coefficient < Real{0.0} ||
        !std::isfinite(
            options.kinematic_area_gradient_filter_coefficient) ||
        options.narrow_band_width < Real{0.0} ||
        !std::isfinite(options.narrow_band_width) ||
        options.smoothing_iterations < 0 ||
        options.smoothing_relaxation < Real{0.0} ||
        options.smoothing_relaxation > Real{1.0} ||
        !std::isfinite(options.smoothing_relaxation)) {
        throw std::invalid_argument(
            "level-set curvature projection requires a finite positive gradient tolerance, a finite relative rank tolerance in (0,1), a nonnegative residual limit, a positive supplemental sample weight, a finite nonnegative kinematic-area-gradient filter coefficient, a nonnegative narrow-band width, nonnegative smoothing iterations, and smoothing relaxation in [0,1]");
    }
    for (const auto value : level_set_vertex_values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "level-set curvature projection received a non-finite level-set value");
        }
    }

    const auto sample_revision =
        supplementalSampleRevisionIdentity(supplemental_samples);
    LevelSetCurvatureProjectionResult result;
    result.free_surface_snapshot_revision_key =
        sample_revision.free_surface_snapshot_revision_key;
    result.source_value_revision = sample_revision.source_value_revision;
    result.vertices = n_vertices;
    result.supplemental_samples = supplemental_samples.size();
    result.generated_interface_geometry_samples =
        static_cast<std::size_t>(std::count_if(
            supplemental_samples.begin(),
            supplemental_samples.end(),
            [](const LevelSetCurvatureProjectionSample& sample) {
                return sample.generated_interface_geometry;
            }));
    result.supplemental_sample_weight = options.supplemental_sample_weight;
    result.recovery_mode = options.recovery_mode;
    result.narrow_band_width = options.narrow_band_width;
    result.smoothing_mode = options.smoothing_mode;
    switch (options.recovery_mode) {
        case LevelSetCurvatureRecoveryMode::LevelSetQuadratic:
        case LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch:
        case LevelSetCurvatureRecoveryMode::KinematicAreaGradient:
            break;
        default:
            throw std::invalid_argument(
                "level-set curvature projection received an unknown recovery mode");
    }
    if (!options.kinematic_area_gradient_young_walls.empty() &&
        options.recovery_mode !=
            LevelSetCurvatureRecoveryMode::KinematicAreaGradient) {
        throw std::invalid_argument(
            "kinematic-area-gradient Young walls require kinematic-area-gradient curvature recovery");
    }
    const Real pi = std::acos(Real{-1.0});
    for (std::size_t wall_index = 0;
         wall_index < options.kinematic_area_gradient_young_walls.size();
         ++wall_index) {
        const auto& wall =
            options.kinematic_area_gradient_young_walls[wall_index];
        if (wall.boundary_marker < 0 ||
            !(wall.equilibrium_contact_angle_radians > Real{0.0}) ||
            !(wall.equilibrium_contact_angle_radians < pi) ||
            !std::isfinite(wall.equilibrium_contact_angle_radians)) {
            throw std::invalid_argument(
                "kinematic-area-gradient Young walls require a nonnegative boundary marker and a finite contact angle strictly between zero and pi radians");
        }
        for (std::size_t previous = 0; previous < wall_index; ++previous) {
            if (options.kinematic_area_gradient_young_walls[previous]
                    .boundary_marker == wall.boundary_marker) {
                throw std::invalid_argument(
                    "kinematic-area-gradient Young wall boundary markers must be unique");
            }
        }
    }
    if (options.recovery_mode ==
            LevelSetCurvatureRecoveryMode::KinematicAreaGradient &&
        options.smoothing_iterations != 0) {
        throw std::invalid_argument(
            "kinematic-area-gradient curvature recovery owns its regularization and cannot be combined with post-projection smoothing");
    }
    if (workspace != nullptr) {
        workspace->free_surface_snapshot_revision_key =
            sample_revision.free_surface_snapshot_revision_key;
        workspace->source_value_revision = sample_revision.source_value_revision;
        workspace->cut_rule_signature = 0u;
    }
    curvature_vertex_values.assign(n_vertices, Real{0.0});
    if (n_vertices == 0u) {
        result.diagnostic = "level-set curvature projection received an empty mesh";
        return result;
    }
    if (options.recovery_mode ==
            LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch &&
        result.generated_interface_geometry_samples == 0u) {
        result.diagnostic =
            "generated-interface-patch curvature recovery requires generated interface geometry samples";
        return result;
    }

    std::vector<std::vector<GlobalIndex>> local_adjacency;
    const auto& adjacency =
        cachedVertexAdjacency(mesh, workspace, result, local_adjacency);
    std::vector<std::vector<std::size_t>> local_sample_adjacency;
    const auto& sample_adjacency =
        cachedSampleAdjacency(mesh,
                              supplemental_samples,
                              workspace,
                              result,
                              local_sample_adjacency);
    const auto n_fit = fitSize(dim);
    const int rings = std::max(1, options.max_neighbor_rings);
    const bool use_narrow_band = options.narrow_band_width > Real{0.0};
    std::vector<unsigned char> active_vertices(n_vertices, 1u);
    if (use_narrow_band) {
        active_vertices.assign(n_vertices, 0u);
        for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
            const auto index = static_cast<std::size_t>(vertex);
            const Real distance_to_interface =
                std::abs(level_set_vertex_values[index] - options.isovalue);
            if (distance_to_interface <= options.narrow_band_width ||
                !sample_adjacency[index].empty()) {
                active_vertices[index] = 1u;
            }
        }
    }
    result.narrow_band_vertices =
        static_cast<std::size_t>(std::count(active_vertices.begin(),
                                            active_vertices.end(),
                                            static_cast<unsigned char>(1u)));
    result.skipped_far_vertices = n_vertices - result.narrow_band_vertices;
    std::vector<unsigned char> fitted(n_vertices, 0u);

    if (options.recovery_mode ==
        LevelSetCurvatureRecoveryMode::KinematicAreaGradient) {
        (void)recoverKinematicAreaGradientCurvature(
            mesh,
            level_set_vertex_values,
            options,
            curvature_vertex_values,
            active_vertices,
            fitted,
            result);
    } else {
    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        if (active_vertices[static_cast<std::size_t>(vertex)] == 0u) {
            continue;
        }
        const auto center = mesh.getNodeCoordinates(vertex);
        const auto neighbors = collectNeighbors(vertex, adjacency, rings);

        std::vector<RawFitObservation> raw_observations;
        std::size_t rows = 0u;
        const auto center_value =
            level_set_vertex_values[static_cast<std::size_t>(vertex)];
        for (const auto neighbor : neighbors) {
            const auto neighbor_index = static_cast<std::size_t>(neighbor);
            const auto x = mesh.getNodeCoordinates(neighbor);
            std::array<Real, 3> dx{
                x[0] - center[0],
                x[1] - center[1],
                dim == 2 ? Real{0.0} : x[2] - center[2],
            };
            const Real distance2 = dot(dx, dx);
            if (!(distance2 > Real{0.0}) || !std::isfinite(distance2)) {
                continue;
            }
            const Real rhs = level_set_vertex_values[neighbor_index] -
                             center_value;
            raw_observations.push_back(
                RawFitObservation{dx, rhs, Real{1.0}});
            ++rows;
        }
        std::vector<std::size_t> sample_indices =
            sample_adjacency[static_cast<std::size_t>(vertex)];
        for (const auto neighbor : neighbors) {
            const auto& neighbor_samples =
                sample_adjacency[static_cast<std::size_t>(neighbor)];
            sample_indices.insert(sample_indices.end(),
                                  neighbor_samples.begin(),
                                  neighbor_samples.end());
        }
        std::sort(sample_indices.begin(), sample_indices.end());
        sample_indices.erase(
            std::unique(sample_indices.begin(), sample_indices.end()),
            sample_indices.end());

        std::size_t supplemental_rows = 0u;
        for (const auto sample_index : sample_indices) {
            if (sample_index >= supplemental_samples.size()) {
                continue;
            }
            const auto& sample = supplemental_samples[sample_index];
            if (!std::isfinite(sample.value) ||
                !std::isfinite(sample.coordinate[0]) ||
                !std::isfinite(sample.coordinate[1]) ||
                !std::isfinite(sample.coordinate[2])) {
                throw std::invalid_argument(
                    "level-set curvature projection received a non-finite supplemental sample");
            }
            if (options.recovery_mode ==
                    LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch &&
                sample.generated_interface_geometry) {
                continue;
            }
            std::array<Real, 3> dx{
                sample.coordinate[0] - center[0],
                sample.coordinate[1] - center[1],
                dim == 2 ? Real{0.0} : sample.coordinate[2] - center[2],
            };
            const Real distance2 = dot(dx, dx);
            if (!(distance2 > Real{0.0}) || !std::isfinite(distance2)) {
                continue;
            }
            const Real rhs = sample.value - center_value;
            raw_observations.push_back(RawFitObservation{
                dx, rhs, options.supplemental_sample_weight});
            ++rows;
            ++supplemental_rows;
        }
        result.supplemental_sample_rows += supplemental_rows;
        if (supplemental_rows > 0u) {
            ++result.vertices_with_supplemental_samples;
        }
        if (rows < n_fit) {
            ++result.insufficient_stencil_vertices;
            continue;
        }

        const auto coordinate_scales = fitCoordinateScales(
            std::span<const RawFitObservation>(raw_observations.data(),
                                               raw_observations.size()),
            dim);
        const auto observations = nondimensionalizeObservations(
            std::span<const RawFitObservation>(raw_observations.data(),
                                               raw_observations.size()),
            coordinate_scales,
            dim);
        if (observations.size() < n_fit) {
            ++result.insufficient_stencil_vertices;
            continue;
        }

        std::array<Real, 9> normalized_coefficients{};
        if (!solveWeightedLeastSquares(
                std::span<const FitObservation>(observations.data(),
                                                observations.size()),
                n_fit,
                options.normal_equation_tolerance,
                normalized_coefficients)) {
            ++result.singular_stencil_vertices;
            continue;
        }
        auto residual = computeFitResidualMetrics(
            std::span<const FitObservation>(observations.data(),
                                            observations.size()),
            normalized_coefficients,
            n_fit,
            options.gradient_tolerance);
        if (!(std::isfinite(residual.rms) &&
              std::isfinite(residual.normalized))) {
            ++result.singular_stencil_vertices;
            continue;
        }
        if (options.recovery_mode ==
                LevelSetCurvatureRecoveryMode::LevelSetQuadratic &&
            options.max_normalized_fit_residual > Real{0.0} &&
            residual.normalized > options.max_normalized_fit_residual) {
            ++result.fit_residual_failure_vertices;
            continue;
        }
        auto coefficients = normalized_coefficients;
        dimensionalizeFitCoefficients(coefficients, coordinate_scales, dim);
        Real kappa = Real{0.0};
        if (options.recovery_mode ==
            LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch) {
            const std::array<Real, 3> level_set_gradient{{
                coefficients[0],
                coefficients[1],
                dim == 2 ? Real{0.0} : coefficients[2],
            }};
            const Real gradient_norm = norm(level_set_gradient);
            if (!(gradient_norm > options.gradient_tolerance) ||
                !std::isfinite(gradient_norm)) {
                ++result.small_gradient_vertices;
                continue;
            }
            FitResidualMetrics patch_residual{};
            std::size_t geometry_sample_count = 0u;
            std::vector<std::size_t> local_patch_sample_indices;
            if (dim == 2) {
                // Curve patches need samples from several generated segments.
                // The distance-weighted configured stencil retains those
                // segments without giving its outer points equal influence.
                local_patch_sample_indices = sample_indices;
            } else {
                // Surface quadrature normally supplies several non-collinear
                // points per cut cell.  Prefer the immediate one-ring patch,
                // then retry the configured wider stencil when needed.
                local_patch_sample_indices =
                    sample_adjacency[static_cast<std::size_t>(vertex)];
                for (const auto neighbor :
                     adjacency[static_cast<std::size_t>(vertex)]) {
                    const auto& neighbor_samples =
                        sample_adjacency[static_cast<std::size_t>(neighbor)];
                    local_patch_sample_indices.insert(
                        local_patch_sample_indices.end(),
                        neighbor_samples.begin(),
                        neighbor_samples.end());
                }
            }
            std::sort(local_patch_sample_indices.begin(),
                      local_patch_sample_indices.end());
            local_patch_sample_indices.erase(
                std::unique(local_patch_sample_indices.begin(),
                            local_patch_sample_indices.end()),
                local_patch_sample_indices.end());
            bool recovered_patch = recoverGeneratedInterfacePatchCurvature(
                    center,
                    level_set_gradient,
                    dim,
                    std::span<const std::size_t>(
                        local_patch_sample_indices.data(),
                        local_patch_sample_indices.size()),
                    supplemental_samples,
                    options.normal_equation_tolerance,
                    options.gradient_tolerance,
                    kappa,
                    patch_residual,
                    geometry_sample_count);
            if (!recovered_patch &&
                local_patch_sample_indices != sample_indices) {
                recovered_patch = recoverGeneratedInterfacePatchCurvature(
                    center,
                    level_set_gradient,
                    dim,
                    std::span<const std::size_t>(sample_indices.data(),
                                                 sample_indices.size()),
                    supplemental_samples,
                    options.normal_equation_tolerance,
                    options.gradient_tolerance,
                    kappa,
                    patch_residual,
                    geometry_sample_count);
                if (recovered_patch) {
                    ++result.generated_interface_patch_expanded_vertices;
                }
            }
            if (!recovered_patch) {
                const std::size_t patch_fit_size = dim == 2 ? 3u : 6u;
                if (geometry_sample_count < patch_fit_size) {
                    ++result.insufficient_stencil_vertices;
                } else {
                    ++result.singular_stencil_vertices;
                }
                continue;
            }
            residual = patch_residual;
        } else {
            bool small_gradient = false;
            kappa = curvatureFromFit(
                coefficients, dim, options.gradient_tolerance, small_gradient);
            if (small_gradient) {
                ++result.small_gradient_vertices;
                continue;
            }
        }
        if (options.max_normalized_fit_residual > Real{0.0} &&
            residual.normalized > options.max_normalized_fit_residual) {
            ++result.fit_residual_failure_vertices;
            continue;
        }
        if (!std::isfinite(kappa)) {
            ++result.singular_stencil_vertices;
            continue;
        }
        curvature_vertex_values[static_cast<std::size_t>(vertex)] = kappa;
        fitted[static_cast<std::size_t>(vertex)] = 1u;
        ++result.fitted_vertices;
        if (options.recovery_mode ==
            LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch) {
            ++result.generated_interface_patch_fitted_vertices;
        }
        result.mean_fit_rms_residual += residual.rms;
        result.mean_normalized_fit_residual += residual.normalized;
        result.max_fit_rms_residual =
            std::max(result.max_fit_rms_residual, residual.rms);
        result.max_normalized_fit_residual =
            std::max(result.max_normalized_fit_residual, residual.normalized);
    }

    std::vector<unsigned char> recovered = fitted;
    std::vector<GlobalIndex> pending;
    pending.reserve(n_vertices);
    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        if (active_vertices[index] != 0u && recovered[index] == 0u) {
            pending.push_back(vertex);
        }
    }

    std::vector<GlobalIndex> next_pending;
    next_pending.reserve(pending.size());
    while (!pending.empty()) {
        bool made_progress = false;
        next_pending.clear();
        for (const auto vertex : pending) {
            const auto index = static_cast<std::size_t>(vertex);
            Real sum = Real{0.0};
            std::size_t count = 0u;
            for (const auto neighbor : adjacency[index]) {
                const auto neighbor_index = static_cast<std::size_t>(neighbor);
                if (recovered[neighbor_index] == 0u) {
                    continue;
                }
                sum += curvature_vertex_values[neighbor_index];
                ++count;
            }
            if (count > 0u) {
                curvature_vertex_values[index] = sum / static_cast<Real>(count);
                recovered[index] = 1u;
                ++result.fallback_vertices;
                made_progress = true;
            } else {
                next_pending.push_back(vertex);
            }
        }
        if (!made_progress) {
            break;
        }
        pending.swap(next_pending);
    }
    for (const auto vertex : pending) {
        const auto index = static_cast<std::size_t>(vertex);
        if (recovered[index] == 0u) {
            curvature_vertex_values[index] = Real{0.0};
            ++result.zero_fallback_vertices;
        }
    }
    }

    if (result.fitted_vertices == 0u) {
        if (!result.diagnostic.empty()) {
            return result;
        }
        result.diagnostic = result.narrow_band_vertices == 0u
            ? "level-set curvature projection found no vertices in the requested narrow band"
            : result.fit_residual_failure_vertices > 0u
            ? "level-set curvature projection exceeded the normalized fit residual limit"
            : "level-set curvature projection could not fit any vertex stencil";
        return result;
    }
    if (options.max_neighbor_fallback_vertices >= 0 &&
        result.fallback_vertices >
            static_cast<std::size_t>(options.max_neighbor_fallback_vertices)) {
        result.diagnostic =
            "level-set curvature projection neighbor fallback vertices " +
            std::to_string(result.fallback_vertices) +
            " exceed configured limit " +
            std::to_string(options.max_neighbor_fallback_vertices);
        return result;
    }
    if (options.max_zero_fallback_vertices >= 0 &&
        result.zero_fallback_vertices >
            static_cast<std::size_t>(options.max_zero_fallback_vertices)) {
        result.diagnostic =
            "level-set curvature projection zero fallback vertices " +
            std::to_string(result.zero_fallback_vertices) +
            " exceed configured limit " +
            std::to_string(options.max_zero_fallback_vertices);
        return result;
    }

    const auto fitted_count = static_cast<Real>(result.fitted_vertices);
    result.mean_fit_rms_residual /= fitted_count;
    result.mean_normalized_fit_residual /= fitted_count;

    const auto active_span =
        std::span<const unsigned char>(active_vertices.data(),
                                       active_vertices.size());
    switch (options.smoothing_mode) {
        case LevelSetCurvatureSmoothingMode::LocalGraph:
            smoothCurvatureOnVertexGraph(
                adjacency,
                active_span,
                options.smoothing_iterations,
                options.smoothing_relaxation,
                curvature_vertex_values,
                result);
            break;
        case LevelSetCurvatureSmoothingMode::MassStiffnessOperator:
            smoothCurvatureWithMassStiffnessOperator(
                mesh,
                adjacency,
                active_span,
                options.smoothing_iterations,
                options.smoothing_relaxation,
                curvature_vertex_values,
                result);
            break;
    }

    result.min_curvature = std::numeric_limits<Real>::infinity();
    result.max_curvature = -std::numeric_limits<Real>::infinity();
    for (const auto kappa : curvature_vertex_values) {
        result.min_curvature = std::min(result.min_curvature, kappa);
        result.max_curvature = std::max(result.max_curvature, kappa);
        result.max_abs_curvature =
            std::max(result.max_abs_curvature, std::abs(kappa));
    }
    if (!std::isfinite(result.min_curvature)) {
        result.min_curvature = Real{0.0};
    }
    if (!std::isfinite(result.max_curvature)) {
        result.max_curvature = Real{0.0};
    }
    if (result.fit_residual_failure_vertices > 0u) {
        result.diagnostic =
            "level-set curvature projection used neighbor fallback for stencils "
            "that exceeded the normalized fit residual limit";
    }
    result.success = true;
    return result;
}

namespace {

LevelSetCurvatureProjectionResult
projectKinematicAreaGradientCollectively(
    const systems::FESystem& system,
    FieldId level_set_field,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace* workspace)
{
    const auto& local_mesh = system.meshAccess();
    if (local_mesh.parallelSize() <= 1) {
        return workspace != nullptr
            ? projectLevelSetMeanCurvatureToVertices(
                  local_mesh,
                  level_set_vertex_values,
                  supplemental_samples,
                  options,
                  curvature_vertex_values,
                  *workspace)
            : projectLevelSetMeanCurvatureToVertices(
                  local_mesh,
                  level_set_vertex_values,
                  supplemental_samples,
                  options,
                  curvature_vertex_values);
    }

    LevelSetCurvatureProjectionResult failure_result;
    failure_result.vertices = level_set_vertex_values.size();
    failure_result.recovery_mode = options.recovery_mode;
    failure_result.kinematic_area_gradient_parallel_size =
        local_mesh.parallelSize();
    curvature_vertex_values.assign(
        level_set_vertex_values.size(), Real{0.0});

    KinematicProjectionCollectiveContext context;
    auto replicated = buildReplicatedKinematicProjectionInput(
        system, level_set_field, level_set_vertex_values, context);
    failure_result.kinematic_area_gradient_parallel_size = context.size;
    if (!replicated.success) {
        failure_result.diagnostic = std::move(replicated.diagnostic);
        return failure_result;
    }

    const auto local_options_signature =
        kinematicProjectionOptionsSignature(options);
    auto minimum_options_signature = local_options_signature;
    auto maximum_options_signature = local_options_signature;
    SupplementalSampleRevisionIdentity local_sample_revision;
    bool local_revision_success = true;
    std::string local_revision_diagnostic;
    try {
        local_sample_revision =
            supplementalSampleRevisionIdentity(supplemental_samples);
    } catch (const std::exception& exception) {
        local_revision_success = false;
        local_revision_diagnostic = exception.what();
    }
    if (!synchronizeKinematicProjectionFailure(
            context, local_revision_success,
            local_revision_diagnostic, failure_result.diagnostic)) {
        return failure_result;
    }
    auto minimum_snapshot_revision =
        local_sample_revision.free_surface_snapshot_revision_key == 0u
            ? std::numeric_limits<std::uint64_t>::max()
            : local_sample_revision.free_surface_snapshot_revision_key;
    auto maximum_snapshot_revision =
        local_sample_revision.free_surface_snapshot_revision_key;
    auto minimum_source_revision =
        local_sample_revision.source_value_revision == 0u
            ? std::numeric_limits<std::uint64_t>::max()
            : local_sample_revision.source_value_revision;
    auto maximum_source_revision =
        local_sample_revision.source_value_revision;
    int any_revisioned_samples =
        local_sample_revision.free_surface_snapshot_revision_key != 0u
            ? 1
            : 0;
    int any_unrevisioned_samples =
        !supplemental_samples.empty() &&
                local_sample_revision.free_surface_snapshot_revision_key == 0u
            ? 1
            : 0;
    std::uint64_t global_sample_count = supplemental_samples.size();
    std::uint64_t global_geometry_sample_count =
        static_cast<std::uint64_t>(std::count_if(
            supplemental_samples.begin(),
            supplemental_samples.end(),
            [](const auto& sample) {
                return sample.generated_interface_geometry;
            }));
#if FE_HAS_MPI
    if (context.active) {
        MPI_Allreduce(MPI_IN_PLACE, &minimum_options_signature, 1,
                      kinematicMpiUnsigned64Type(), MPI_MIN,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_options_signature, 1,
                      kinematicMpiUnsigned64Type(), MPI_MAX,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_snapshot_revision, 1,
                      kinematicMpiUnsigned64Type(), MPI_MIN,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_snapshot_revision, 1,
                      kinematicMpiUnsigned64Type(), MPI_MAX,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_source_revision, 1,
                      kinematicMpiUnsigned64Type(), MPI_MIN,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_source_revision, 1,
                      kinematicMpiUnsigned64Type(), MPI_MAX,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &any_revisioned_samples, 1,
                      MPI_INT, MPI_MAX, context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &any_unrevisioned_samples, 1,
                      MPI_INT, MPI_MAX, context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &global_sample_count, 1,
                      kinematicMpiUnsigned64Type(), MPI_SUM,
                      context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &global_geometry_sample_count, 1,
                      kinematicMpiUnsigned64Type(), MPI_SUM,
                      context.communicator);
    }
#endif
    if (minimum_options_signature != maximum_options_signature) {
        failure_result.diagnostic =
            "kinematic-area-gradient collective projection requires identical options on every rank";
        return failure_result;
    }
    if (any_revisioned_samples != 0 &&
        (any_unrevisioned_samples != 0 ||
         minimum_snapshot_revision != maximum_snapshot_revision ||
         minimum_source_revision != maximum_source_revision)) {
        failure_result.diagnostic =
            "kinematic-area-gradient collective projection requires one complete supplemental-sample revision identity";
        return failure_result;
    }
    if (any_revisioned_samples == 0) {
        minimum_snapshot_revision = 0u;
        minimum_source_revision = 0u;
    }

    std::vector<LevelSetCurvatureProjectionSample> translated_samples(
        supplemental_samples.begin(), supplemental_samples.end());
    for (auto& sample : translated_samples) {
        if (sample.parent_cell < 0 ||
            sample.parent_cell >= local_mesh.numCells()) {
            sample.parent_cell = static_cast<MeshIndex>(-1);
            continue;
        }
        const auto global_cell =
            local_mesh.getCellGlobalId(sample.parent_cell);
        const auto translated =
            replicated.global_cell_to_replicated_cell.find(global_cell);
        sample.parent_cell = translated ==
                                     replicated
                                         .global_cell_to_replicated_cell.end()
            ? static_cast<MeshIndex>(-1)
            : static_cast<MeshIndex>(translated->second);
    }

    const auto gathered_cell_count = replicated.cells.size();
    const auto gathered_boundary_face_count =
        replicated.boundary_faces.size();
    ReplicatedKinematicMeshAccess replicated_mesh(
        local_mesh.dimension(),
        std::move(replicated.coordinates),
        std::move(replicated.cells),
        std::move(replicated.boundary_faces));
    std::vector<Real> global_curvature;
    auto projection = workspace != nullptr
        ? projectLevelSetMeanCurvatureToVertices(
              replicated_mesh,
              std::span<const Real>(replicated.level_set_values.data(),
                                    replicated.level_set_values.size()),
              std::span<const LevelSetCurvatureProjectionSample>(
                  translated_samples.data(), translated_samples.size()),
              options,
              global_curvature,
              *workspace)
        : projectLevelSetMeanCurvatureToVertices(
              replicated_mesh,
              std::span<const Real>(replicated.level_set_values.data(),
                                    replicated.level_set_values.size()),
              std::span<const LevelSetCurvatureProjectionSample>(
                  translated_samples.data(), translated_samples.size()),
              options,
              global_curvature);
    projection.free_surface_snapshot_revision_key =
        minimum_snapshot_revision;
    projection.source_value_revision = minimum_source_revision;
    projection.supplemental_samples =
        static_cast<std::size_t>(global_sample_count);
    projection.generated_interface_geometry_samples =
        static_cast<std::size_t>(global_geometry_sample_count);
    projection.kinematic_area_gradient_collective_replication = true;
    projection.kinematic_area_gradient_parallel_size = context.size;
    projection.kinematic_area_gradient_gathered_owned_cells =
        gathered_cell_count;
    projection
        .kinematic_area_gradient_gathered_owned_boundary_faces =
        gathered_boundary_face_count;
    if (!projection.success) {
        curvature_vertex_values.assign(
            level_set_vertex_values.size(), Real{0.0});
        return projection;
    }
    if (global_curvature.size() != replicated.level_set_values.size()) {
        projection.success = false;
        projection.diagnostic =
            "kinematic-area-gradient collective projection produced an invalid global curvature layout";
        curvature_vertex_values.assign(
            level_set_vertex_values.size(), Real{0.0});
        return projection;
    }
    curvature_vertex_values.resize(
        replicated.local_vertex_to_global_dof.size());
    for (std::size_t local_vertex = 0;
         local_vertex < replicated.local_vertex_to_global_dof.size();
         ++local_vertex) {
        curvature_vertex_values[local_vertex] = global_curvature[
            replicated.local_vertex_to_global_dof[local_vertex]];
    }
    return projection;
}

} // namespace

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const systems::FESystem& system,
    FieldId level_set_field,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values)
{
    return projectKinematicAreaGradientCollectively(
        system,
        level_set_field,
        level_set_vertex_values,
        supplemental_samples,
        options,
        curvature_vertex_values,
        nullptr);
}

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const systems::FESystem& system,
    FieldId level_set_field,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace& workspace)
{
    return projectKinematicAreaGradientCollectively(
        system,
        level_set_field,
        level_set_vertex_values,
        supplemental_samples,
        options,
        curvature_vertex_values,
        &workspace);
}

} // namespace svmp::FE::level_set
