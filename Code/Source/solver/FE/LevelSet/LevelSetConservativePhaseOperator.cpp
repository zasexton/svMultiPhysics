#include "LevelSet/LevelSetConservativePhaseOperator.h"

#include "Basis/BasisFunction.h"
#include "Geometry/MappingFactory.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp::FE::level_set {
namespace {

using Vector3 = std::array<Real, 3>;

struct MutableGradientEdge {
    Vector3 first_test_second_gradient{};
    Vector3 second_test_first_gradient{};
};

struct PhaseGraphCollectiveContext {
    bool active{false};
    int rank{0};
    int size{1};
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
};

#if FE_HAS_MPI
struct PackedGradientEdge {
    std::int64_t first_node{-1};
    std::int64_t second_node{-1};
    std::array<Real, 3> first_test_second_gradient{};
    std::array<Real, 3> second_test_first_gradient{};
};

static_assert(std::is_trivially_copyable_v<PackedGradientEdge>);
static_assert(sizeof(PackedGradientEdge) ==
              2u * sizeof(std::int64_t) + 6u * sizeof(Real));

[[nodiscard]] MPI_Datatype mpiRealType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<Real, double>) {
        return MPI_DOUBLE;
    }
    return MPI_LONG_DOUBLE;
}

[[nodiscard]] MPI_Datatype mpiUnsigned64Type() noexcept
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

[[nodiscard]] PhaseGraphCollectiveContext phaseGraphCollectiveContext(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& dofs)
{
    PhaseGraphCollectiveContext context;
    context.rank = mesh.parallelRank();
    context.size = mesh.parallelSize();
    if (context.rank < 0 || context.size < 1 ||
        context.rank >= context.size) {
        throw std::invalid_argument(
            "P1 conservative phase graph received invalid mesh rank metadata");
    }
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
            throw std::invalid_argument(
                "P1 conservative phase graph requires an active field communicator for multi-rank assembly");
        }
        int communicator_rank = 0;
        int communicator_size = 1;
        MPI_Comm_rank(dofs.mpiComm(), &communicator_rank);
        MPI_Comm_size(dofs.mpiComm(), &communicator_size);
        if (communicator_rank != context.rank ||
            communicator_size != context.size) {
            throw std::invalid_argument(
                "P1 conservative phase graph mesh and field communicators disagree");
        }
        context.active = true;
        context.communicator = dofs.mpiComm();
    }
#else
    (void)dofs;
    if (context.size > 1) {
        throw std::invalid_argument(
            "P1 conservative phase graph cannot assemble a multi-rank mesh without MPI support");
    }
#endif
    return context;
}

[[nodiscard]] bool synchronizeLocalFailure(
    const PhaseGraphCollectiveContext& context,
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
            constexpr std::size_t maximum_diagnostic_bytes = 4096u;
            int length = 0;
            if (context.rank == first_failed_rank) {
                length = static_cast<int>(std::min(
                    maximum_diagnostic_bytes, local_diagnostic.size()));
            }
            MPI_Bcast(&length, 1, MPI_INT, first_failed_rank,
                      context.communicator);
            std::vector<char> bytes(static_cast<std::size_t>(length));
            if (context.rank == first_failed_rank && length > 0) {
                std::copy_n(local_diagnostic.data(), length, bytes.data());
            }
            if (length > 0) {
                MPI_Bcast(bytes.data(), length, MPI_CHAR,
                          first_failed_rank, context.communicator);
            }
            collective_diagnostic =
                "P1 conservative phase graph failed on rank " +
                std::to_string(first_failed_rank) + ": " +
                std::string(bytes.begin(), bytes.end());
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

void allReduceRealBufferSum(const PhaseGraphCollectiveContext& context,
                            std::vector<Real>& values)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        constexpr std::size_t maximum_chunk =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        std::vector<Real> reduced(values.size(), Real{0.0});
        for (std::size_t offset = 0u; offset < values.size();) {
            const std::size_t count =
                std::min(maximum_chunk, values.size() - offset);
            MPI_Allreduce(
                values.data() + static_cast<std::ptrdiff_t>(offset),
                reduced.data() + static_cast<std::ptrdiff_t>(offset),
                static_cast<int>(count), mpiRealType(), MPI_SUM,
                context.communicator);
            offset += count;
        }
        values.swap(reduced);
    }
#else
    (void)context;
    (void)values;
#endif
}

void allReduceVector3Sum(const PhaseGraphCollectiveContext& context,
                         std::vector<Vector3>& values)
{
    std::vector<Real> flat(values.size() * 3u, Real{0.0});
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t d = 0; d < 3u; ++d) {
            flat[3u * i + d] = values[i][d];
        }
    }
    allReduceRealBufferSum(context, flat);
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t d = 0; d < 3u; ++d) {
            values[i][d] = flat[3u * i + d];
        }
    }
}

#if FE_HAS_MPI
[[nodiscard]] Real allReduceReal(const PhaseGraphCollectiveContext& context,
                                 Real local,
                                 MPI_Op operation)
{
    if (context.active) {
        Real global{0.0};
        MPI_Allreduce(&local, &global, 1, mpiRealType(), operation,
                      context.communicator);
        return global;
    }
    return local;
}

[[nodiscard]] int allReduceInt(const PhaseGraphCollectiveContext& context,
                               int local,
                               MPI_Op operation)
{
    if (context.active) {
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, operation,
                      context.communicator);
        return global;
    }
    return local;
}

[[nodiscard]] std::uint64_t allReduceUnsigned64(
    const PhaseGraphCollectiveContext& context,
    std::uint64_t local,
    MPI_Op operation
)
{
    if (context.active) {
        std::uint64_t global = 0u;
        MPI_Allreduce(&local, &global, 1, mpiUnsigned64Type(), operation,
                      context.communicator);
        return global;
    }
    return local;
}
#endif

[[nodiscard]] Real allReduceRealSum(
    const PhaseGraphCollectiveContext& context, Real local)
{
#if FE_HAS_MPI
    return allReduceReal(context, local, MPI_SUM);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] Real allReduceRealMin(
    const PhaseGraphCollectiveContext& context, Real local)
{
#if FE_HAS_MPI
    return allReduceReal(context, local, MPI_MIN);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] Real allReduceRealMax(
    const PhaseGraphCollectiveContext& context, Real local)
{
#if FE_HAS_MPI
    return allReduceReal(context, local, MPI_MAX);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] int allReduceIntMin(
    const PhaseGraphCollectiveContext& context, int local)
{
#if FE_HAS_MPI
    return allReduceInt(context, local, MPI_MIN);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] int allReduceIntMax(
    const PhaseGraphCollectiveContext& context, int local)
{
#if FE_HAS_MPI
    return allReduceInt(context, local, MPI_MAX);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Sum(
    const PhaseGraphCollectiveContext& context, std::uint64_t local)
{
#if FE_HAS_MPI
    return allReduceUnsigned64(context, local, MPI_SUM);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Min(
    const PhaseGraphCollectiveContext& context, std::uint64_t local)
{
#if FE_HAS_MPI
    return allReduceUnsigned64(context, local, MPI_MIN);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Max(
    const PhaseGraphCollectiveContext& context, std::uint64_t local)
{
#if FE_HAS_MPI
    return allReduceUnsigned64(context, local, MPI_MAX);
#else
    (void)context;
    return local;
#endif
}

void allReduceIntBufferMinMax(
    const PhaseGraphCollectiveContext& context,
    const std::vector<int>& local,
    std::vector<int>& minimum,
    std::vector<int>& maximum)
{
    minimum = local;
    maximum = local;
#if FE_HAS_MPI
    if (context.active && !local.empty()) {
        constexpr std::size_t maximum_chunk =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        for (std::size_t offset = 0u; offset < local.size();) {
            const std::size_t count =
                std::min(maximum_chunk, local.size() - offset);
            const auto displacement = static_cast<std::ptrdiff_t>(offset);
            MPI_Allreduce(local.data() + displacement,
                          minimum.data() + displacement,
                          static_cast<int>(count), MPI_INT, MPI_MIN,
                          context.communicator);
            MPI_Allreduce(local.data() + displacement,
                          maximum.data() + displacement,
                          static_cast<int>(count), MPI_INT, MPI_MAX,
                          context.communicator);
            offset += count;
        }
    }
#else
    (void)context;
#endif
}

[[nodiscard]] std::map<std::pair<GlobalIndex, GlobalIndex>,
                       MutableGradientEdge>
globalizeGradientEdges(
    const PhaseGraphCollectiveContext& context,
    const std::map<std::pair<GlobalIndex, GlobalIndex>, MutableGradientEdge>&
        local_edges)
{
#if FE_HAS_MPI
    if (context.active) {
        const int local_capacity_valid =
            local_edges.size() <=
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()) /
                        sizeof(PackedGradientEdge)
                ? 1
                : 0;
        int global_capacity_valid = 0;
        MPI_Allreduce(&local_capacity_valid, &global_capacity_valid, 1,
                      MPI_INT, MPI_MIN, context.communicator);
        if (global_capacity_valid == 0) {
            throw std::overflow_error(
                "P1 conservative phase graph local edge snapshot exceeds MPI count capacity");
        }
        std::vector<PackedGradientEdge> packed;
        packed.reserve(local_edges.size());
        for (const auto& [endpoints, edge] : local_edges) {
            packed.push_back(PackedGradientEdge{
                .first_node = static_cast<std::int64_t>(endpoints.first),
                .second_node = static_cast<std::int64_t>(endpoints.second),
                .first_test_second_gradient =
                    edge.first_test_second_gradient,
                .second_test_first_gradient =
                    edge.second_test_first_gradient,
            });
        }
        const std::size_t local_bytes_size =
            packed.size() * sizeof(PackedGradientEdge);
        const int local_bytes = static_cast<int>(local_bytes_size);
        std::vector<int> counts(static_cast<std::size_t>(context.size), 0);
        MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1,
                      MPI_INT, context.communicator);
        std::vector<int> displacements(counts.size(), 0);
        std::size_t total_bytes = 0u;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            if (counts[rank] < 0 ||
                counts[rank] % static_cast<int>(sizeof(PackedGradientEdge)) !=
                    0 ||
                total_bytes >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()) -
                        static_cast<std::size_t>(counts[rank])) {
                throw std::overflow_error(
                    "P1 conservative phase graph aggregate edge snapshot exceeds MPI displacement capacity");
            }
            displacements[rank] = static_cast<int>(total_bytes);
            total_bytes += static_cast<std::size_t>(counts[rank]);
        }
        std::vector<std::byte> gathered(total_bytes);
        MPI_Allgatherv(
            packed.empty() ? nullptr : packed.data(), local_bytes, MPI_BYTE,
            gathered.empty() ? nullptr : gathered.data(), counts.data(),
            displacements.data(), MPI_BYTE, context.communicator);

        std::map<std::pair<GlobalIndex, GlobalIndex>, MutableGradientEdge>
            global_edges;
        for (std::size_t offset = 0u; offset < gathered.size();
             offset += sizeof(PackedGradientEdge)) {
            PackedGradientEdge packed_edge{};
            std::memcpy(&packed_edge, gathered.data() + offset,
                        sizeof(PackedGradientEdge));
            const GlobalIndex first =
                static_cast<GlobalIndex>(packed_edge.first_node);
            const GlobalIndex second =
                static_cast<GlobalIndex>(packed_edge.second_node);
            if (first < 0 || first >= second) {
                throw std::runtime_error(
                    "P1 conservative phase graph received a malformed distributed edge record");
            }
            auto& edge = global_edges[{first, second}];
            for (std::size_t d = 0; d < 3u; ++d) {
                edge.first_test_second_gradient[d] +=
                    packed_edge.first_test_second_gradient[d];
                edge.second_test_first_gradient[d] +=
                    packed_edge.second_test_first_gradient[d];
            }
        }
        return global_edges;
    }
#else
    (void)context;
#endif
    return local_edges;
}

[[nodiscard]] Real scaledTolerance(Real tolerance,
                                   std::initializer_list<Real> values)
{
    Real scale = Real{1.0};
    for (const Real value : values) {
        scale = std::max(scale, std::abs(value));
    }
    return tolerance * scale;
}

[[nodiscard]] Real dot(const Vector3& first,
                       const Vector3& second,
                       int dimension) noexcept
{
    Real value{0.0};
    for (int d = 0; d < dimension; ++d) {
        value += first[static_cast<std::size_t>(d)] *
                 second[static_cast<std::size_t>(d)];
    }
    return value;
}

[[nodiscard]] Real norm(const Vector3& value, int dimension) noexcept
{
    return std::sqrt(std::max(Real{0.0}, dot(value, value, dimension)));
}

void addScaled(Vector3& target,
               const math::Vector<Real, 3>& source,
               Real scale,
               int dimension) noexcept
{
    for (int d = 0; d < dimension; ++d) {
        target[static_cast<std::size_t>(d)] +=
            scale * source[static_cast<std::size_t>(d)];
    }
}

void add(Vector3& target, const Vector3& source, int dimension) noexcept
{
    for (int d = 0; d < dimension; ++d) {
        target[static_cast<std::size_t>(d)] +=
            source[static_cast<std::size_t>(d)];
    }
}

[[nodiscard]] int resolvedQuadratureOrder(int requested,
                                          int geometry_order)
{
    if (requested > 0) {
        return requested;
    }
    return std::max(4, 3 * std::max(1, geometry_order));
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> makeCellMapping(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell)
{
    std::vector<std::array<Real, 3>> coordinates;
    mesh.getCellCoordinates(cell, coordinates);
    if (coordinates.empty()) {
        throw std::invalid_argument(
            "P1 conservative phase graph found a cell without geometry nodes");
    }

    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        nodes.push_back(math::Vector<Real, 3>{
            coordinate[0], coordinate[1], coordinate[2]});
    }

    geometry::MappingRequest request;
    request.element_type = mesh.getCellType(cell);
    request.geometry_order = mesh.getCellGeometryOrder(cell);
    request.use_affine = request.geometry_order <= 1;
    return geometry::MappingFactory::create(request, nodes);
}

} // namespace

LevelSetP1PhaseTransportGraph buildLevelSetP1PhaseTransportGraph(
    const systems::FESystem& system,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseGraphOptions& options)
{
    LevelSetP1PhaseTransportGraph result;
    try {
        if (liquid_indicator_field == INVALID_FIELD_ID) {
            result.diagnostic =
                "P1 conservative phase graph received an invalid field";
            return result;
        }

        const auto& mesh = system.meshAccess();
        result.dimension = mesh.dimension();
        const auto& record = system.fieldRecord(liquid_indicator_field);
        const auto& dofs = system.fieldDofHandler(liquid_indicator_field);
        const auto collective = phaseGraphCollectiveContext(mesh, dofs);
        result.parallel_rank = collective.rank;
        result.parallel_size = collective.size;
        result.distributed = collective.active;
        result.replicated_sparse_graph = collective.active;

        result.geometry_revision = mesh.geometryRevision();
        result.topology_revision = mesh.topologyRevision();
        result.ownership_revision = mesh.ownershipRevision();
        result.numbering_revision = mesh.numberingRevision();
        result.dof_layout_revision = dofs.dofLayoutRevision();

        bool local_preflight_success = true;
        std::string local_preflight_diagnostic;
        const auto reject_preflight = [&](std::string diagnostic) {
            if (local_preflight_success) {
                local_preflight_success = false;
                local_preflight_diagnostic = std::move(diagnostic);
            }
        };
        if (!std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0} ||
            options.quadrature_order < 0) {
            reject_preflight(
                "P1 conservative phase graph requires a nonnegative quadrature order and finite nonnegative tolerance");
        }
        if (result.dimension < 2 || result.dimension > 3) {
            reject_preflight(
                "P1 conservative phase graph supports two- and three-dimensional meshes");
        }
        if (record.components != 1 || !record.space ||
            record.space->space_type() != spaces::SpaceType::H1 ||
            record.space->field_type() != FieldType::Scalar ||
            record.space->continuity() != Continuity::C0 ||
            record.space->value_dimension() != 1 ||
            record.space->is_variable_order() ||
            record.space->polynomial_order() != 1) {
            reject_preflight(
                "P1 conservative phase graph requires a fixed-order scalar P1 H1 field");
        }
        if (!system.fieldParticipatesInUnknownVector(liquid_indicator_field)) {
            reject_preflight(
                "P1 conservative phase graph requires a transported unknown field");
        }
        if (dofs.getNumDofs() <= 0 || dofs.getNumLocalDofs() < 0 ||
            dofs.getNumLocalDofs() > dofs.getNumDofs()) {
            reject_preflight(
                "P1 conservative phase graph requires a valid nonempty field layout");
        }
        if (mesh.numOwnedCells() < 0 ||
            mesh.numOwnedCells() > mesh.numCells()) {
            reject_preflight(
                "P1 conservative phase graph requires valid owned-cell metadata");
        }
        if (!synchronizeLocalFailure(
                collective, local_preflight_success,
                local_preflight_diagnostic, result.diagnostic)) {
            return result;
        }

        const int minimum_dimension = allReduceIntMin(
            collective, result.dimension);
        const int maximum_dimension = allReduceIntMax(
            collective, result.dimension);
        const int minimum_requested_order = allReduceIntMin(
            collective, options.quadrature_order);
        const int maximum_requested_order = allReduceIntMax(
            collective, options.quadrature_order);
        const Real minimum_requested_tolerance = allReduceRealMin(
            collective, options.invariant_tolerance);
        const Real maximum_requested_tolerance = allReduceRealMax(
            collective, options.invariant_tolerance);
        const auto local_node_count =
            static_cast<std::uint64_t>(dofs.getNumDofs());
        const auto minimum_node_count = allReduceUnsigned64Min(
            collective, local_node_count);
        const auto maximum_node_count = allReduceUnsigned64Max(
            collective, local_node_count);
        if (minimum_dimension != maximum_dimension ||
            minimum_requested_order != maximum_requested_order ||
            minimum_requested_tolerance != maximum_requested_tolerance ||
            minimum_node_count != maximum_node_count) {
            result.diagnostic =
                "P1 conservative phase graph requires identical dimension, options, and global field size on every rank";
            return result;
        }
        result.dimension = minimum_dimension;
        result.nodes = static_cast<std::size_t>(minimum_node_count);

        const auto geometry_revision_min = allReduceUnsigned64Min(
            collective, result.geometry_revision);
        const auto geometry_revision_max = allReduceUnsigned64Max(
            collective, result.geometry_revision);
        const auto topology_revision_min = allReduceUnsigned64Min(
            collective, result.topology_revision);
        const auto topology_revision_max = allReduceUnsigned64Max(
            collective, result.topology_revision);
        const auto ownership_revision_min = allReduceUnsigned64Min(
            collective, result.ownership_revision);
        const auto ownership_revision_max = allReduceUnsigned64Max(
            collective, result.ownership_revision);
        const auto numbering_revision_min = allReduceUnsigned64Min(
            collective, result.numbering_revision);
        const auto numbering_revision_max = allReduceUnsigned64Max(
            collective, result.numbering_revision);
        const auto dof_revision_min = allReduceUnsigned64Min(
            collective, result.dof_layout_revision);
        const auto dof_revision_max = allReduceUnsigned64Max(
            collective, result.dof_layout_revision);
        if (geometry_revision_min != geometry_revision_max ||
            topology_revision_min != topology_revision_max ||
            ownership_revision_min != ownership_revision_max ||
            numbering_revision_min != numbering_revision_max ||
            dof_revision_min != dof_revision_max) {
            result.diagnostic =
                "P1 conservative phase graph requires synchronized mesh and field revisions on every rank";
            return result;
        }
        result.geometry_revision = geometry_revision_min;
        result.topology_revision = topology_revision_min;
        result.ownership_revision = ownership_revision_min;
        result.numbering_revision = numbering_revision_min;
        result.dof_layout_revision = dof_revision_min;

        result.lumped_control_volume.assign(result.nodes, Real{0.0});
        result.diagonal_gradient.assign(result.nodes, Vector3{});
        result.boundary_column_sum.assign(result.nodes, Vector3{});

        bool local_constraint_success = true;
        std::string local_constraint_diagnostic;
        try {
            const GlobalIndex field_offset =
                system.fieldDofOffset(liquid_indicator_field);
            for (std::size_t i = 0; i < result.nodes; ++i) {
                if (system.constraints().isConstrained(
                        field_offset + static_cast<GlobalIndex>(i))) {
                    local_constraint_success = false;
                    local_constraint_diagnostic =
                        "P1 conservative phase graph does not accept constrained indicator nodes; boundary transport must enter through phase fluxes";
                    break;
                }
            }
        } catch (const std::exception& exception) {
            local_constraint_success = false;
            local_constraint_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective, local_constraint_success,
                local_constraint_diagnostic, result.diagnostic)) {
            return result;
        }

        result.minimum_jacobian_determinant =
            std::numeric_limits<Real>::infinity();

        std::map<std::pair<GlobalIndex, GlobalIndex>, MutableGradientEdge>
            assembled_edges;
        std::vector<Vector3> assembled_row_sum(result.nodes, Vector3{});
        long double local_physical_measure_accumulator = 0.0L;
        Real maximum_gradient_coefficient{0.0};
        Real maximum_physical_basis_gradient{0.0};

        bool local_assembly_success = true;
        std::string local_assembly_diagnostic;
        try {
            mesh.forEachOwnedCell([&](GlobalIndex cell) {
                if (record.space->polynomial_order(cell) != 1) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph found a non-P1 cell");
                }
                const auto cell_dofs = dofs.getCellDofs(cell);
                const auto& element = record.space->getElement(
                    mesh.getCellType(cell), cell);
                const auto& basis = element.basis();
                if (basis.is_vector_valued() ||
                    basis.basis_type() != BasisType::Lagrange ||
                    basis.order() != 1 || basis.size() != cell_dofs.size() ||
                    cell_dofs.size() <
                        static_cast<std::size_t>(result.dimension + 1)) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph requires one scalar linear Lagrange basis value per cell DOF");
                }
                for (const auto dof : cell_dofs) {
                    if (dof < 0 ||
                        static_cast<std::size_t>(dof) >= result.nodes) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph found a cell DOF outside the field layout");
                    }
                }

                const auto mapping = makeCellMapping(mesh, cell);
                if (!mapping || mapping->dimension() != result.dimension) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph found an incompatible geometry mapping");
                }
                const int quadrature_order = resolvedQuadratureOrder(
                    options.quadrature_order,
                    mesh.getCellGeometryOrder(cell));
                result.maximum_quadrature_order = std::max(
                    result.maximum_quadrature_order, quadrature_order);
                const auto quadrature = quadrature::QuadratureFactory::create(
                    mesh.getCellType(cell), quadrature_order);
                if (!quadrature || quadrature->num_points() == 0u) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph received an empty quadrature rule");
                }

                const std::size_t local_size = cell_dofs.size();
                std::vector<Real> local_mass(local_size, Real{0.0});
                std::vector<Vector3> local_gradient(
                    local_size * local_size, Vector3{});
                std::vector<Real> values;
                std::vector<basis::Gradient> reference_gradients;
                for (std::size_t q = 0; q < quadrature->num_points(); ++q) {
                    const auto xi = quadrature->point(q);
                    const Real determinant = mapping->jacobian_determinant(xi);
                    const Real quadrature_weight = quadrature->weight(q);
                    if (!std::isfinite(determinant) ||
                        !(determinant > Real{0.0}) ||
                        !std::isfinite(quadrature_weight) ||
                        !(quadrature_weight > Real{0.0})) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph found a nonpositive or non-finite mapped quadrature weight");
                    }
                    result.minimum_jacobian_determinant = std::min(
                        result.minimum_jacobian_determinant, determinant);
                    const Real weight = determinant * quadrature_weight;
                    local_physical_measure_accumulator +=
                        static_cast<long double>(weight);

                    basis.evaluate_values(xi, values);
                    basis.evaluate_gradients(xi, reference_gradients);
                    if (values.size() != local_size ||
                        reference_gradients.size() != local_size) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph received inconsistent basis evaluations");
                    }
                    std::vector<math::Vector<Real, 3>> physical_gradients;
                    physical_gradients.reserve(local_size);
                    const auto inverse = mapping->jacobian_inverse(xi);
                    Real value_sum{0.0};
                    Vector3 gradient_sum{};
                    for (std::size_t local = 0; local < local_size; ++local) {
                        if (!std::isfinite(values[local])) {
                            throw std::invalid_argument(
                                "P1 conservative phase graph found a non-finite basis value");
                        }
                        const auto physical_gradient =
                            mapping->transform_gradient(
                                reference_gradients[local], inverse);
                        for (int d = 0; d < result.dimension; ++d) {
                            if (!std::isfinite(
                                    physical_gradient[static_cast<std::size_t>(d)])) {
                                throw std::invalid_argument(
                                    "P1 conservative phase graph found a non-finite physical basis gradient");
                            }
                        }
                        physical_gradients.push_back(physical_gradient);
                        Vector3 physical_gradient_array{};
                        for (int d = 0; d < result.dimension; ++d) {
                            physical_gradient_array[static_cast<std::size_t>(d)] =
                                physical_gradient[static_cast<std::size_t>(d)];
                        }
                        maximum_physical_basis_gradient = std::max(
                            maximum_physical_basis_gradient,
                            norm(physical_gradient_array, result.dimension));
                        value_sum += values[local];
                        addScaled(gradient_sum, physical_gradient,
                                  Real{1.0}, result.dimension);
                        local_mass[local] += weight * values[local];
                    }
                    result.maximum_partition_of_unity_residual = std::max(
                        result.maximum_partition_of_unity_residual,
                        std::abs(value_sum - Real{1.0}));
                    result.maximum_gradient_partition_residual = std::max(
                        result.maximum_gradient_partition_residual,
                        norm(gradient_sum, result.dimension));

                    for (std::size_t i = 0; i < local_size; ++i) {
                        for (std::size_t j = 0; j < local_size; ++j) {
                            addScaled(local_gradient[i * local_size + j],
                                      physical_gradients[j],
                                      weight * values[i], result.dimension);
                        }
                    }
                }

                for (std::size_t i = 0; i < local_size; ++i) {
                    const auto global_i = cell_dofs[i];
                    const auto node_i = static_cast<std::size_t>(global_i);
                    if (!std::isfinite(local_mass[i]) ||
                        !(local_mass[i] > Real{0.0})) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph found a nonpositive local lumped control volume");
                    }
                    result.lumped_control_volume[node_i] += local_mass[i];
                    for (std::size_t j = 0; j < local_size; ++j) {
                        const auto global_j = cell_dofs[j];
                        const auto& coefficient =
                            local_gradient[i * local_size + j];
                        add(assembled_row_sum[node_i], coefficient,
                            result.dimension);
                        add(result.boundary_column_sum[
                                static_cast<std::size_t>(global_j)],
                            coefficient, result.dimension);
                        maximum_gradient_coefficient = std::max(
                            maximum_gradient_coefficient,
                            norm(coefficient, result.dimension));
                        if (i == j) {
                            add(result.diagonal_gradient[node_i], coefficient,
                                result.dimension);
                            continue;
                        }
                        const auto endpoints = std::minmax(global_i, global_j);
                        auto& edge = assembled_edges[{endpoints.first,
                                                      endpoints.second}];
                        if (global_i == endpoints.first) {
                            add(edge.first_test_second_gradient, coefficient,
                                result.dimension);
                        } else {
                            add(edge.second_test_first_gradient, coefficient,
                                result.dimension);
                        }
                    }
                }
                ++result.cells;
            });
        } catch (const std::exception& exception) {
            local_assembly_success = false;
            local_assembly_diagnostic = exception.what();
        }
        if (local_assembly_success &&
            result.cells !=
                static_cast<std::size_t>(mesh.numOwnedCells())) {
            local_assembly_success = false;
            local_assembly_diagnostic =
                "P1 conservative phase graph owned-cell traversal count does not match mesh metadata";
        }
        if (!synchronizeLocalFailure(
                collective, local_assembly_success,
                local_assembly_diagnostic, result.diagnostic)) {
            return result;
        }
        result.local_owned_cells = result.cells;
        result.physical_measure = static_cast<Real>(
            local_physical_measure_accumulator);

        const auto global_cell_count = allReduceUnsigned64Sum(
            collective, static_cast<std::uint64_t>(result.cells));
        if (global_cell_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            result.diagnostic =
                "P1 conservative phase graph global cell count exceeds local index capacity";
            return result;
        }
        result.cells = static_cast<std::size_t>(global_cell_count);
        result.physical_measure = allReduceRealSum(
            collective, result.physical_measure);
        result.minimum_jacobian_determinant = allReduceRealMin(
            collective, result.minimum_jacobian_determinant);
        result.maximum_partition_of_unity_residual = allReduceRealMax(
            collective, result.maximum_partition_of_unity_residual);
        result.maximum_gradient_partition_residual = allReduceRealMax(
            collective, result.maximum_gradient_partition_residual);
        maximum_gradient_coefficient = allReduceRealMax(
            collective, maximum_gradient_coefficient);
        maximum_physical_basis_gradient = allReduceRealMax(
            collective, maximum_physical_basis_gradient);
        result.maximum_quadrature_order = allReduceIntMax(
            collective, result.maximum_quadrature_order);
        allReduceRealBufferSum(collective,
                               result.lumped_control_volume);
        allReduceVector3Sum(collective, result.diagonal_gradient);
        allReduceVector3Sum(collective, result.boundary_column_sum);
        allReduceVector3Sum(collective, assembled_row_sum);
        assembled_edges = globalizeGradientEdges(
            collective, assembled_edges);
        maximum_gradient_coefficient = Real{0.0};
        for (const auto& coefficient : result.diagonal_gradient) {
            maximum_gradient_coefficient = std::max(
                maximum_gradient_coefficient,
                norm(coefficient, result.dimension));
        }
        for (const auto& [endpoints, edge] : assembled_edges) {
            (void)endpoints;
            maximum_gradient_coefficient = std::max(
                maximum_gradient_coefficient,
                std::max(norm(edge.first_test_second_gradient,
                              result.dimension),
                         norm(edge.second_test_first_gradient,
                              result.dimension)));
        }

        if (result.cells == 0u ||
            !std::isfinite(result.physical_measure) ||
            !(result.physical_measure > Real{0.0}) ||
            !std::isfinite(result.minimum_jacobian_determinant)) {
            result.diagnostic =
                "P1 conservative phase graph found no valid owned-cell measure";
            return result;
        }

        result.minimum_lumped_control_volume =
            std::numeric_limits<Real>::infinity();
        long double volume_sum = 0.0L;
        result.positive_control_volumes_satisfied = true;
        for (std::size_t node = 0; node < result.nodes; ++node) {
            const Real volume = result.lumped_control_volume[node];
            result.positive_control_volumes_satisfied =
                result.positive_control_volumes_satisfied &&
                std::isfinite(volume) && volume > Real{0.0};
            result.minimum_lumped_control_volume = std::min(
                result.minimum_lumped_control_volume, volume);
            volume_sum += static_cast<long double>(volume);
            result.maximum_gradient_row_sum_residual = std::max(
                result.maximum_gradient_row_sum_residual,
                norm(assembled_row_sum[node], result.dimension));
        }
        if (!result.positive_control_volumes_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph left a node without positive owned-cell measure";
            return result;
        }
        result.total_lumped_control_volume = static_cast<Real>(volume_sum);
        result.measure_closure_residual =
            result.total_lumped_control_volume - result.physical_measure;

        const Real partition_tolerance =
            scaledTolerance(options.invariant_tolerance, {Real{1.0}});
        const Real gradient_partition_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {maximum_physical_basis_gradient});
        const Real row_sum_tolerance = scaledTolerance(
            options.invariant_tolerance, {maximum_gradient_coefficient});
        const Real measure_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {result.physical_measure, result.total_lumped_control_volume});
        result.partition_of_unity_satisfied =
            result.maximum_partition_of_unity_residual <=
            partition_tolerance;
        result.gradient_partition_satisfied =
            result.maximum_gradient_partition_residual <=
            gradient_partition_tolerance;
        result.gradient_row_sum_satisfied =
            result.maximum_gradient_row_sum_residual <= row_sum_tolerance;
        result.measure_closure_satisfied =
            std::abs(result.measure_closure_residual) <= measure_tolerance;
        if (!result.partition_of_unity_satisfied ||
            !result.gradient_partition_satisfied ||
            !result.gradient_row_sum_satisfied ||
            !result.measure_closure_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph failed a partition, gradient, or measure identity";
            return result;
        }

        std::vector<int> local_edge_owners;
        local_edge_owners.reserve(assembled_edges.size());
        bool local_ownership_success = true;
        std::string ownership_diagnostic;
        try {
            for (const auto& [endpoints, edge] : assembled_edges) {
                (void)edge;
                int owner_rank = 0;
                if (collective.active) {
                    const int first_owner =
                        dofs.getDofMap().getDofOwner(endpoints.first);
                    const int second_owner =
                        dofs.getDofMap().getDofOwner(endpoints.second);
                    if (first_owner < 0 || second_owner < 0 ||
                        first_owner >= collective.size ||
                        second_owner >= collective.size) {
                        local_ownership_success = false;
                        ownership_diagnostic =
                            "P1 conservative phase graph could not resolve a valid owner for every algebraic edge";
                        break;
                    }
                    owner_rank = std::min(first_owner, second_owner);
                }
                local_edge_owners.push_back(owner_rank);
            }
        } catch (const std::exception& exception) {
            local_ownership_success = false;
            ownership_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective, local_ownership_success,
                ownership_diagnostic, result.diagnostic)) {
            result.edge_ownership_satisfied = false;
            return result;
        }

        std::vector<int> minimum_edge_owners;
        std::vector<int> maximum_edge_owners;
        allReduceIntBufferMinMax(collective, local_edge_owners,
                                 minimum_edge_owners,
                                 maximum_edge_owners);
        result.edge_ownership_satisfied =
            minimum_edge_owners == maximum_edge_owners;
        if (!result.edge_ownership_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph found inconsistent algebraic edge ownership across ranks";
            return result;
        }

        result.edges.reserve(assembled_edges.size());
        std::size_t edge_index = 0u;
        for (const auto& [endpoints, edge] : assembled_edges) {
            const int owner_rank = minimum_edge_owners[edge_index++];
            result.edges.push_back(LevelSetP1PhaseGradientEdge{
                .first_node = endpoints.first,
                .second_node = endpoints.second,
                .owner_rank = owner_rank,
                .first_test_second_gradient =
                    edge.first_test_second_gradient,
                .second_test_first_gradient =
                    edge.second_test_first_gradient,
            });
            if (owner_rank == collective.rank) {
                ++result.locally_owned_edges;
            }
        }
        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

LevelSetP1PhaseTransportStageResult
advanceLevelSetP1ConservativePhaseStage(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    std::span<const Real> lower_liquid_indicator,
    std::span<const Real> upper_liquid_indicator,
    std::span<const std::array<Real, 3>> nodal_velocity,
    Real time_step,
    const LevelSetP1PhaseStageOptions& options)
{
    LevelSetP1PhaseTransportStageResult result;
    try {
        if (!graph.success) {
            result.diagnostic =
                "P1 conservative phase stage requires a valid assembled graph";
            return result;
        }
        const std::size_t node_count = graph.nodes;
        if (node_count == 0u ||
            graph.lumped_control_volume.size() != node_count ||
            graph.diagonal_gradient.size() != node_count ||
            graph.boundary_column_sum.size() != node_count ||
            previous_liquid_indicator.size() != node_count ||
            lower_liquid_indicator.size() != node_count ||
            upper_liquid_indicator.size() != node_count ||
            nodal_velocity.size() != node_count) {
            result.diagnostic =
                "P1 conservative phase stage received inconsistent nodal spans";
            return result;
        }
        if (!(time_step > Real{0.0}) || !std::isfinite(time_step) ||
            !std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0} ||
            !std::isfinite(options.component_activity_tolerance) ||
            !(options.component_activity_tolerance > Real{0.0}) ||
            options.component_activity_tolerance > Real{1.0} ||
            !(options.maximum_courant > Real{0.0}) ||
            options.maximum_courant > Real{1.0} ||
            !std::isfinite(options.maximum_courant)) {
            result.diagnostic =
                "P1 conservative phase stage requires a positive finite time step, nonnegative finite invariant tolerance, component activity tolerance in (0,1], and Courant limit in (0,1]";
            return result;
        }

        result.nodal_courant.assign(node_count, Real{0.0});
        result.physical_boundary_mass_transfer.assign(
            node_count, Real{0.0});
        result.discrete_divergence_mass_source.assign(
            node_count, Real{0.0});
        std::vector<Real> row_divergence(node_count, Real{0.0});
        std::vector<Real> direct_strong_rate(node_count, Real{0.0});
        std::vector<Real> decomposed_strong_rate(node_count, Real{0.0});
        result.flux_edges.reserve(graph.edges.size());
        result.minimum_low_order_coefficient =
            std::numeric_limits<Real>::infinity();
        result.low_order_coefficients_nonnegative = true;

        for (std::size_t node = 0; node < node_count; ++node) {
            const auto& velocity = nodal_velocity[node];
            for (int d = 0; d < graph.dimension; ++d) {
                if (!std::isfinite(velocity[static_cast<std::size_t>(d)])) {
                    result.diagnostic =
                        "P1 conservative phase stage found a non-finite nodal velocity";
                    return result;
                }
            }
            const Real diagonal_velocity = dot(
                graph.diagonal_gradient[node], velocity, graph.dimension);
            row_divergence[node] = diagonal_velocity;
            direct_strong_rate[node] =
                -diagonal_velocity * previous_liquid_indicator[node];
        }

        for (const auto& edge : graph.edges) {
            if (edge.first_node < 0 || edge.second_node < 0 ||
                edge.first_node >= edge.second_node ||
                static_cast<std::size_t>(edge.second_node) >= node_count) {
                result.diagnostic =
                    "P1 conservative phase stage found a malformed graph edge";
                return result;
            }
            const auto first = static_cast<std::size_t>(edge.first_node);
            const auto second = static_cast<std::size_t>(edge.second_node);
            const Real first_to_second_speed = dot(
                edge.first_test_second_gradient,
                nodal_velocity[second], graph.dimension);
            const Real second_to_first_speed = dot(
                edge.second_test_first_gradient,
                nodal_velocity[first], graph.dimension);
            const Real diffusion = std::max(
                std::abs(first_to_second_speed),
                std::abs(second_to_first_speed));
            const Real first_coefficient =
                diffusion - first_to_second_speed;
            const Real second_coefficient =
                diffusion - second_to_first_speed;
            result.minimum_low_order_coefficient = std::min(
                result.minimum_low_order_coefficient,
                std::min(first_coefficient, second_coefficient));
            const Real coefficient_tolerance = scaledTolerance(
                options.invariant_tolerance,
                {diffusion, first_to_second_speed,
                 second_to_first_speed});
            result.low_order_coefficients_nonnegative =
                result.low_order_coefficients_nonnegative &&
                first_coefficient >= -coefficient_tolerance &&
                second_coefficient >= -coefficient_tolerance;

            result.nodal_courant[first] +=
                time_step * std::max(Real{0.0}, first_coefficient) /
                graph.lumped_control_volume[first];
            result.nodal_courant[second] +=
                time_step * std::max(Real{0.0}, second_coefficient) /
                graph.lumped_control_volume[second];

            const Real first_indicator = previous_liquid_indicator[first];
            const Real second_indicator = previous_liquid_indicator[second];
            const Real central_rate =
                second_to_first_speed * first_indicator -
                first_to_second_speed * second_indicator;
            const Real diffusive_rate =
                diffusion * (second_indicator - first_indicator);
            result.flux_edges.push_back(LevelSetPhaseFluxEdge{
                .first_node = edge.first_node,
                .second_node = edge.second_node,
                .low_order_mass_transfer =
                    time_step * (central_rate + diffusive_rate),
                .raw_antidiffusive_mass_transfer =
                    -time_step * diffusive_rate,
            });
            decomposed_strong_rate[first] += central_rate;
            decomposed_strong_rate[second] -= central_rate;

            row_divergence[first] += first_to_second_speed;
            row_divergence[second] += second_to_first_speed;
            direct_strong_rate[first] -=
                first_to_second_speed * second_indicator;
            direct_strong_rate[second] -=
                second_to_first_speed * first_indicator;
        }

        if (graph.edges.empty()) {
            result.minimum_low_order_coefficient = Real{0.0};
        }
        if (!result.low_order_coefficients_nonnegative) {
            result.diagnostic =
                "P1 conservative phase stage produced a negative low-order graph coefficient";
            return result;
        }

        result.maximum_courant = Real{0.0};
        Real maximum_integrated_strong_rate{0.0};
        for (std::size_t node = 0; node < node_count; ++node) {
            if (!std::isfinite(result.nodal_courant[node])) {
                result.diagnostic =
                    "P1 conservative phase stage produced a non-finite nodal Courant number";
                return result;
            }
            result.maximum_courant = std::max(
                result.maximum_courant, result.nodal_courant[node]);
            const Real indicator = previous_liquid_indicator[node];
            const Real boundary_rate =
                -indicator * dot(graph.boundary_column_sum[node],
                                 nodal_velocity[node], graph.dimension);
            const Real divergence_rate = indicator * row_divergence[node];
            result.physical_boundary_mass_transfer[node] =
                time_step * boundary_rate;
            result.discrete_divergence_mass_source[node] =
                time_step * divergence_rate;
            decomposed_strong_rate[node] +=
                boundary_rate + divergence_rate;
            direct_strong_rate[node] += divergence_rate;

            const Real residual = time_step *
                (decomposed_strong_rate[node] -
                 direct_strong_rate[node]);
            maximum_integrated_strong_rate = std::max(
                maximum_integrated_strong_rate,
                time_step * std::max(
                    std::abs(decomposed_strong_rate[node]),
                    std::abs(direct_strong_rate[node])));
            result.maximum_strong_form_decomposition_residual = std::max(
                result.maximum_strong_form_decomposition_residual,
                std::abs(residual));
        }

        const Real decomposition_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {maximum_integrated_strong_rate});
        result.strong_form_decomposition_satisfied =
            result.maximum_strong_form_decomposition_residual <=
            decomposition_tolerance;
        if (!result.strong_form_decomposition_satisfied) {
            result.diagnostic =
                "P1 conservative phase stage failed the strong-CG edge/boundary decomposition identity";
            return result;
        }

        result.courant_satisfied =
            !options.enforce_courant_limit ||
            result.maximum_courant <=
                options.maximum_courant +
                    scaledTolerance(options.invariant_tolerance,
                                    {result.maximum_courant,
                                     options.maximum_courant});
        if (!result.courant_satisfied) {
            result.diagnostic =
                "P1 conservative phase stage rejected a time step outside its low-order Courant contract";
            return result;
        }

        result.correction =
            applyLevelSetConservativePhaseFluxCorrection(
                LevelSetPhaseFluxStageView{
                    .lumped_control_volume = graph.lumped_control_volume,
                    .previous_liquid_indicator =
                        previous_liquid_indicator,
                    .lower_liquid_indicator = lower_liquid_indicator,
                    .upper_liquid_indicator = upper_liquid_indicator,
                    .interior_edges = result.flux_edges,
                    .physical_boundary_mass_transfer =
                        result.physical_boundary_mass_transfer,
                    .discrete_divergence_mass_source =
                        result.discrete_divergence_mass_source,
                    .invariant_tolerance = options.invariant_tolerance,
                    .component_activity_tolerance =
                        options.component_activity_tolerance,
                    .require_constant_preservation =
                        options.require_constant_preservation,
                });
        if (!result.correction.success) {
            result.diagnostic = result.correction.diagnostic;
            return result;
        }

        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

} // namespace svmp::FE::level_set
