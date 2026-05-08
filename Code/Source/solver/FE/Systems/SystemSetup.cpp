/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FESystem.h"
#include "Systems/GlobalKernelStateProvider.h"
#include "Systems/MaterialStateProvider.h"
#include "Systems/OperatorBackends.h"

#include "Backends/Interfaces/DofPermutation.h"

#include "Assembly/Assembler.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/AssemblerSelection.h"
#include "Assembly/MatrixFreeAssembler.h"

#include "Basis/BasisCache.h"
#include "Basis/VectorBasis.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Assembly/MeshAccess.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Systems/MeshSearchAccess.h"
#endif

#include "Constraints/GaugeDiagnostics.h"
#include "Constraints/GlobalConstraint.h"
#include "Constraints/ParallelConstraints.h"

#include "Constraints/SystemConstraints.h"

#include "Analysis/TopologyAnalysisContext.h"

#include "Sparsity/ConstraintSparsityAugmenter.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Systems/SystemsExceptions.h"

#include "Spaces/FunctionSpace.h"
#include "Spaces/CompositeSpace.h"
#include "Spaces/MixedSpace.h"
#include "Spaces/MortarSpace.h"

#include "Dofs/EntityDofMap.h"

#include "Elements/ReferenceElement.h"

#include "Quadrature/QuadratureFactory.h"

#include "Core/FEConfig.h"
#include "Core/KernelTrace.h"
#include "Forms/MixedBlockKernelSet.h"
#include "Forms/MonolithicCellKernel.h"
#include "Forms/JIT/JITCacheKey.h"
#include "Forms/JIT/JITKernelWrapper.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <numeric>
#include <thread>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] std::optional<assembly::GhostPolicy> assemblyGhostPolicyOverrideFromEnv()
{
    const char* env = std::getenv("SVMP_ASSEMBLY_GHOST_POLICY");
    if (env == nullptr || *env == '\0') {
        return std::nullopt;
    }

    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0 || ch == '_' || ch == '-';
    }), value.end());

    if (value == "ownedrowsonly" || value == "owned") {
        return assembly::GhostPolicy::OwnedRowsOnly;
    }
    if (value == "reversescatter" || value == "reverse") {
        return assembly::GhostPolicy::ReverseScatter;
    }
    return std::nullopt;
}

[[nodiscard]] bool dofPermutationTraceEnabled() noexcept
{
    const char* env = std::getenv("SVMP_TRACE_DOF_PERMUTATION");
    if (env == nullptr || *env == '\0') {
        return false;
    }
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return !(value == "0" || value == "false" || value == "off" || value == "no");
}

void insertSortedUniqueIndex(std::vector<GlobalIndex>& values, GlobalIndex value)
{
    const auto it = std::lower_bound(values.begin(), values.end(), value);
    if (it == values.end() || *it != value) {
        values.insert(it, value);
    }
}

[[nodiscard]] int referenceVertexCount(ElementType type) noexcept;

void getCellCornerNodes(const assembly::IMeshAccess& access,
                        GlobalIndex cell,
                        std::vector<GlobalIndex>& nodes)
{
    access.getCellNodes(cell, nodes);
    const int n_corners = referenceVertexCount(access.getCellType(cell));
    if (n_corners > 0 && nodes.size() > static_cast<std::size_t>(n_corners)) {
        nodes.resize(static_cast<std::size_t>(n_corners));
    }
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] dofs::MeshTopologyInfo meshTopologyFromAccess(
    const assembly::IMeshAccess& access,
    int owner_rank)
{
    dofs::MeshTopologyInfo topology;
    topology.n_cells = access.numCells();
    topology.n_vertices = access.numVertices();
    topology.dim = access.dimension();

    FE_THROW_IF(topology.n_cells <= 0, InvalidArgumentException,
                "FESystem::setup: meshAccess() has no cells");
    FE_THROW_IF(topology.n_vertices <= 0, InvalidArgumentException,
                "FESystem::setup: meshAccess() has no vertices");
    FE_THROW_IF(topology.dim <= 0, InvalidArgumentException,
                "FESystem::setup: meshAccess() has invalid spatial dimension");

    topology.cell2vertex_offsets.resize(static_cast<std::size_t>(topology.n_cells) + 1u, 0);
    topology.cell_gids.reserve(static_cast<std::size_t>(topology.n_cells));
    topology.cell_owner_ranks.assign(static_cast<std::size_t>(topology.n_cells), owner_rank);

    std::vector<GlobalIndex> nodes;
    for (GlobalIndex cell = 0; cell < topology.n_cells; ++cell) {
        getCellCornerNodes(access, cell, nodes);
        topology.cell_gids.push_back(static_cast<dofs::gid_t>(cell));
        topology.cell2vertex_offsets[static_cast<std::size_t>(cell + 1)] =
            topology.cell2vertex_offsets[static_cast<std::size_t>(cell)] +
            static_cast<MeshOffset>(nodes.size());
        for (const auto node : nodes) {
            topology.cell2vertex_data.push_back(static_cast<MeshIndex>(node));
        }
    }

    topology.vertex_gids.reserve(static_cast<std::size_t>(topology.n_vertices));
    topology.vertex_coords.resize(static_cast<std::size_t>(topology.n_vertices) *
                                  static_cast<std::size_t>(topology.dim));
    for (GlobalIndex vertex = 0; vertex < topology.n_vertices; ++vertex) {
        topology.vertex_gids.push_back(static_cast<dofs::gid_t>(vertex));
        const auto x = access.getNodeCoordinates(vertex);
        for (int d = 0; d < topology.dim; ++d) {
            topology.vertex_coords[static_cast<std::size_t>(vertex) *
                                       static_cast<std::size_t>(topology.dim) +
                                   static_cast<std::size_t>(d)] = x[static_cast<std::size_t>(d)];
        }
    }

    return topology;
}

[[nodiscard]] dofs::MeshTopologyInfo participantTopologyFromAccess(
    const assembly::IMeshAccess& access,
    const MeshParticipantInfo& participant,
    int owner_rank)
{
    dofs::MeshTopologyInfo topology;
    topology.n_cells = participant.num_cells;
    topology.n_vertices = participant.num_vertices;
    topology.dim = access.dimension();

    FE_THROW_IF(participant.num_cells <= 0, InvalidArgumentException,
                "FESystem::setup: participant-scoped field has no participant cells");
    FE_THROW_IF(participant.num_vertices <= 0, InvalidArgumentException,
                "FESystem::setup: participant-scoped field has no participant vertices");

    topology.cell2vertex_offsets.resize(static_cast<std::size_t>(topology.n_cells) + 1u, 0);
    topology.cell_gids.reserve(static_cast<std::size_t>(topology.n_cells));
    topology.cell_owner_ranks.assign(static_cast<std::size_t>(topology.n_cells), owner_rank);

    std::vector<GlobalIndex> global_nodes;
    for (GlobalIndex local_cell = 0; local_cell < topology.n_cells; ++local_cell) {
        const auto global_cell = participant.cell_offset + local_cell;
        getCellCornerNodes(access, global_cell, global_nodes);
        topology.cell_gids.push_back(static_cast<dofs::gid_t>(global_cell));
        topology.cell2vertex_offsets[static_cast<std::size_t>(local_cell + 1)] =
            topology.cell2vertex_offsets[static_cast<std::size_t>(local_cell)] +
            static_cast<MeshOffset>(global_nodes.size());
        for (const auto global_node : global_nodes) {
            FE_THROW_IF(global_node < participant.vertex_offset ||
                            global_node >= participant.vertex_offset + participant.num_vertices,
                        InvalidArgumentException,
                        "FESystem::setup: participant cell references a vertex outside the participant");
            topology.cell2vertex_data.push_back(
                static_cast<MeshIndex>(global_node - participant.vertex_offset));
        }
    }

    topology.vertex_gids.reserve(static_cast<std::size_t>(topology.n_vertices));
    topology.vertex_coords.resize(static_cast<std::size_t>(topology.n_vertices) *
                                  static_cast<std::size_t>(topology.dim));
    for (GlobalIndex local_vertex = 0; local_vertex < topology.n_vertices; ++local_vertex) {
        const auto global_vertex = participant.vertex_offset + local_vertex;
        topology.vertex_gids.push_back(static_cast<dofs::gid_t>(global_vertex));
        const auto x = access.getNodeCoordinates(global_vertex);
        for (int d = 0; d < topology.dim; ++d) {
            topology.vertex_coords[static_cast<std::size_t>(local_vertex) *
                                       static_cast<std::size_t>(topology.dim) +
                                   static_cast<std::size_t>(d)] = x[static_cast<std::size_t>(d)];
        }
    }

    return topology;
}

[[nodiscard]] dofs::DofHandler remapParticipantDofHandler(
    const dofs::DofHandler& local_handler,
    const MeshParticipantInfo& participant,
    const assembly::IMeshAccess& access,
    const dofs::DofDistributionOptions& options)
{
    dofs::DofMap global_map(access.numCells(),
                            local_handler.getNumDofs(),
                            local_handler.getDofMap().getMaxDofsPerCell());
    global_map.setNumDofs(local_handler.getNumDofs());
    global_map.setNumLocalDofs(local_handler.getPartition().localOwnedSize());
    global_map.setMyRank(options.my_rank);

    std::vector<GlobalIndex> mapped_cell_dofs;
    for (GlobalIndex global_cell = 0; global_cell < access.numCells(); ++global_cell) {
        if (global_cell >= participant.cell_offset &&
            global_cell < participant.cell_offset + participant.num_cells) {
            const auto local_cell = global_cell - participant.cell_offset;
            const auto dofs = local_handler.getDofMap().getCellDofs(local_cell);
            mapped_cell_dofs.assign(dofs.begin(), dofs.end());
        } else {
            mapped_cell_dofs.clear();
        }
        global_map.setCellDofs(global_cell, mapped_cell_dofs);
    }

    std::vector<int> dof_owner_by_local_id(static_cast<std::size_t>(local_handler.getNumDofs()), 0);
    for (GlobalIndex dof = 0; dof < local_handler.getNumDofs(); ++dof) {
        dof_owner_by_local_id[static_cast<std::size_t>(dof)] =
            local_handler.getDofMap().getDofOwner(dof);
    }
    global_map.setDofOwnership(
        [owners = std::move(dof_owner_by_local_id)](GlobalIndex dof) {
            if (dof < 0 || static_cast<std::size_t>(dof) >= owners.size()) {
                return 0;
            }
            return owners[static_cast<std::size_t>(dof)];
        });

    dofs::DofHandler global_handler;
    global_handler.setDofMap(std::move(global_map));

    auto partition = local_handler.getPartition();
    partition.setGlobalSize(local_handler.getNumDofs());
    global_handler.setPartition(std::move(partition));
    global_handler.setRankInfo(options.my_rank, options.world_size);

    if (const auto* local_entities = local_handler.getEntityDofMap()) {
        auto global_entities = std::make_unique<dofs::EntityDofMap>();
        global_entities->reserve(access.numVertices(), 0, 0, access.numCells());

        std::vector<GlobalIndex> entity_dofs;
        for (GlobalIndex global_vertex = 0; global_vertex < access.numVertices(); ++global_vertex) {
            entity_dofs.clear();
            if (global_vertex >= participant.vertex_offset &&
                global_vertex < participant.vertex_offset + participant.num_vertices) {
                const auto local_vertex = global_vertex - participant.vertex_offset;
                if (local_vertex < local_entities->numVertices()) {
                    const auto dofs = local_entities->getVertexDofs(local_vertex);
                    entity_dofs.assign(dofs.begin(), dofs.end());
                }
            }
            global_entities->setVertexDofs(global_vertex, entity_dofs);
        }

        for (GlobalIndex global_cell = 0; global_cell < access.numCells(); ++global_cell) {
            entity_dofs.clear();
            if (global_cell >= participant.cell_offset &&
                global_cell < participant.cell_offset + participant.num_cells) {
                const auto local_cell = global_cell - participant.cell_offset;
                if (local_cell < local_entities->numCells()) {
                    const auto dofs = local_entities->getCellInteriorDofs(local_cell);
                    entity_dofs.assign(dofs.begin(), dofs.end());
                }
            }
            global_entities->setCellInteriorDofs(global_cell, entity_dofs);
        }

        global_entities->buildReverseMapping();
        global_entities->finalize();
        global_handler.setEntityDofMap(std::move(global_entities));
    }

    global_handler.finalize();
    return global_handler;
}
#endif

#if FE_HAS_MPI
constexpr std::uint64_t kConsistencyHashSeed = 1469598103934665603ULL;

[[nodiscard]] std::uint64_t mixConsistencyHash(std::uint64_t hash,
                                               std::uint64_t value) noexcept
{
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    return hash;
}

template <class T>
void hashConsistencyBytes(std::uint64_t& hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        hash = mixConsistencyHash(hash, static_cast<std::uint64_t>(bytes[i]));
    }
}

template <class T>
void hashConsistencySpan(std::uint64_t& hash, std::span<const T> values)
{
    hashConsistencyBytes(hash, values.size());
    for (const auto& value : values) {
        hashConsistencyBytes(hash, value);
    }
}

[[nodiscard]] std::vector<int> gatherGlobalBoundaryMarkers(std::span<const int> local_markers,
                                                           MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        std::vector<int> markers(local_markers.begin(), local_markers.end());
        std::sort(markers.begin(), markers.end());
        markers.erase(std::unique(markers.begin(), markers.end()), markers.end());
        return markers;
    }

    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    if (world_size <= 1) {
        std::vector<int> markers(local_markers.begin(), local_markers.end());
        std::sort(markers.begin(), markers.end());
        markers.erase(std::unique(markers.begin(), markers.end()), markers.end());
        return markers;
    }

    const int local_n = static_cast<int>(local_markers.size());
    std::vector<int> counts(static_cast<std::size_t>(world_size), 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<std::size_t>(world_size), 0);
    for (int r = 1; r < world_size; ++r) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r - 1)] +
            counts[static_cast<std::size_t>(r - 1)];
    }

    const int total_n = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> gathered(static_cast<std::size_t>(total_n), 0);
    MPI_Allgatherv(local_markers.empty() ? nullptr : local_markers.data(),
                   local_n,
                   MPI_INT,
                   gathered.empty() ? nullptr : gathered.data(),
                   counts.data(),
                   displs.data(),
                   MPI_INT,
                   comm);

    std::sort(gathered.begin(), gathered.end());
    gathered.erase(std::unique(gathered.begin(), gathered.end()), gathered.end());
    return gathered;
}

[[nodiscard]] bool gaugeResolutionConsistentAcrossRanks(const gauge::GaugeRegistry& registry,
                                                        MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return true;
    }

    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    if (world_size <= 1) {
        return true;
    }

    struct ResolvedModeKey {
        int field{0};
        int component{0};
        int region{0};
        int family{0};
        int status{0};
        int policy{0};

        [[nodiscard]] bool operator<(const ResolvedModeKey& other) const noexcept
        {
            return std::tie(field, component, region, family, status, policy) <
                   std::tie(other.field, other.component, other.region,
                            other.family, other.status, other.policy);
        }
    };

    std::vector<ResolvedModeKey> keys;
    keys.reserve(registry.resolvedModes().size());
    for (const auto& mode : registry.resolvedModes()) {
        keys.push_back(ResolvedModeKey{
            static_cast<int>(mode.candidate.field),
            mode.candidate.component,
            mode.candidate.region,
            static_cast<int>(mode.candidate.family),
            static_cast<int>(mode.status),
            static_cast<int>(mode.policy)});
    }
    std::sort(keys.begin(), keys.end());

    unsigned long long local_hash = kConsistencyHashSeed;
    local_hash = mixConsistencyHash(local_hash, static_cast<unsigned long long>(keys.size()));
    for (const auto& key : keys) {
        local_hash = mixConsistencyHash(local_hash, static_cast<unsigned long long>(key.field));
        local_hash = mixConsistencyHash(local_hash, static_cast<unsigned long long>(key.component + 4096));
        local_hash = mixConsistencyHash(local_hash, static_cast<unsigned long long>(key.region + 4096));
        local_hash = mixConsistencyHash(local_hash, static_cast<unsigned long long>(key.family));
        local_hash = mixConsistencyHash(local_hash, static_cast<unsigned long long>(key.status));
        local_hash = mixConsistencyHash(local_hash, static_cast<unsigned long long>(key.policy));
    }

    int local_count = static_cast<int>(keys.size());
    int min_count = 0;
    int max_count = 0;
    MPI_Allreduce(&local_count, &min_count, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&local_count, &max_count, 1, MPI_INT, MPI_MAX, comm);

    unsigned long long min_hash = 0ULL;
    unsigned long long max_hash = 0ULL;
    MPI_Allreduce(&local_hash, &min_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(&local_hash, &max_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);

    return min_count == max_count && min_hash == max_hash;
}

[[nodiscard]] bool globalConstraintDefinitionsConsistentAcrossRanks(
    std::span<const std::unique_ptr<constraints::Constraint>> constraint_defs,
    MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return true;
    }

    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    if (world_size <= 1) {
        return true;
    }

    std::vector<std::uint64_t> constraint_hashes;
    constraint_hashes.reserve(constraint_defs.size());

    for (const auto& constraint : constraint_defs) {
        if (!constraint || constraint->getType() != constraints::ConstraintType::Global) {
            continue;
        }

        const auto* global =
            dynamic_cast<const constraints::GlobalConstraint*>(constraint.get());
        if (global == nullptr) {
            return false;
        }

        std::uint64_t hash = kConsistencyHashSeed;
        hashConsistencyBytes(hash, static_cast<std::uint8_t>(global->getType()));
        hashConsistencyBytes(hash, static_cast<std::uint8_t>(global->getGlobalType()));
        hashConsistencyBytes(hash, static_cast<std::uint8_t>(global->getStrategy()));
        hashConsistencyBytes(hash, global->getPinnedDof());
        hashConsistencyBytes(hash, global->getTargetValue());
        hashConsistencySpan(hash, std::span<const GlobalIndex>(global->getDofs()));
        hashConsistencySpan(hash, std::span<const double>(global->getWeights()));
        constraint_hashes.push_back(hash);
    }

    std::sort(constraint_hashes.begin(), constraint_hashes.end());

    std::uint64_t local_hash = kConsistencyHashSeed;
    hashConsistencySpan(local_hash, std::span<const std::uint64_t>(constraint_hashes));

    const int local_count = static_cast<int>(constraint_hashes.size());
    int min_count = 0;
    int max_count = 0;
    MPI_Allreduce(&local_count, &min_count, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&local_count, &max_count, 1, MPI_INT, MPI_MAX, comm);

    std::uint64_t min_hash = 0ULL;
    std::uint64_t max_hash = 0ULL;
    MPI_Allreduce(&local_hash, &min_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(&local_hash, &max_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);

    return min_count == max_count && min_hash == max_hash;
}
#endif

[[nodiscard]] const std::vector<std::vector<int>>* localBoundaryFaceVertexMap(ElementType type) noexcept
{
    static const std::vector<std::vector<int>> tet_faces =
        {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    static const std::vector<std::vector<int>> tri_faces =
        {{0, 1}, {1, 2}, {2, 0}};
    static const std::vector<std::vector<int>> hex_faces =
        {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
         {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    static const std::vector<std::vector<int>> quad_faces =
        {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

    switch (type) {
        case ElementType::Triangle6:
            type = ElementType::Triangle3;
            break;
        case ElementType::Quad8:
        case ElementType::Quad9:
            type = ElementType::Quad4;
            break;
        case ElementType::Tetra10:
            type = ElementType::Tetra4;
            break;
        case ElementType::Hex20:
        case ElementType::Hex27:
            type = ElementType::Hex8;
            break;
        default:
            break;
    }

    if (type == ElementType::Tetra4) return &tet_faces;
    if (type == ElementType::Triangle3) return &tri_faces;
    if (type == ElementType::Hex8) return &hex_faces;
    if (type == ElementType::Quad4) return &quad_faces;
    return nullptr;
}

[[nodiscard]] std::vector<int> boundaryMarkersFromMeshAccess(
    const assembly::IMeshAccess& mesh)
{
    std::unordered_set<int> markers;
    mesh.forEachBoundaryFace(/*marker=*/-1,
        [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
            const int marker = mesh.getBoundaryFaceMarker(face_id);
            if (marker >= 0) {
                markers.insert(marker);
            }
        });

    std::vector<int> result(markers.begin(), markers.end());
    std::sort(result.begin(), result.end());
    return result;
}

[[nodiscard]] std::vector<GlobalIndex> boundaryDofsByMarkerFromMeshAccess(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& dof_handler,
    int marker)
{
    if (!dof_handler.isFinalized()) {
        return {};
    }

    const auto* entity_map = dof_handler.getEntityDofMap();
    if (entity_map == nullptr) {
        return {};
    }

    std::unordered_set<GlobalIndex> visited;
    std::vector<GlobalIndex> dofs;
    auto push_dof = [&](GlobalIndex dof) {
        if (dof >= 0 && visited.insert(dof).second) {
            dofs.push_back(dof);
        }
    };
    auto push_span = [&](std::span<const GlobalIndex> span) {
        for (const auto dof : span) {
            push_dof(dof);
        }
    };

    std::vector<GlobalIndex> cell_nodes;
    mesh.forEachBoundaryFace(marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            const auto local_face = mesh.getLocalFaceIndex(face_id, cell_id);
            const auto* face_map = localBoundaryFaceVertexMap(mesh.getCellType(cell_id));
            if (face_map == nullptr ||
                static_cast<std::size_t>(local_face) >= face_map->size()) {
                return;
            }

            cell_nodes.clear();
            mesh.getCellNodes(cell_id, cell_nodes);

            if (mesh.dimension() >= 3 &&
                face_id >= 0 &&
                face_id < entity_map->numFaces()) {
                push_span(entity_map->getFaceDofs(face_id));
            }

            for (const int local_vertex : (*face_map)[static_cast<std::size_t>(local_face)]) {
                if (local_vertex < 0 ||
                    static_cast<std::size_t>(local_vertex) >= cell_nodes.size()) {
                    continue;
                }
                push_span(entity_map->getVertexDofs(
                    cell_nodes[static_cast<std::size_t>(local_vertex)]));
            }
        });

    std::sort(dofs.begin(), dofs.end());
    return dofs;
}

[[nodiscard]] int referenceVertexCount(ElementType type) noexcept
{
    switch (type) {
        case ElementType::Point1: return 1;
        case ElementType::Line2:
        case ElementType::Line3: return 2;
        case ElementType::Triangle3:
        case ElementType::Triangle6: return 3;
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9: return 4;
        case ElementType::Tetra4:
        case ElementType::Tetra10: return 4;
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27: return 8;
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18: return 6;
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14: return 5;
        case ElementType::Unknown:
        default: return 0;
    }
}

[[nodiscard]] SetupStorageRequirements storageRequirementsForSpace(
    const spaces::FunctionSpace& space,
    const assembly::IMeshAccess* mesh_access = nullptr)
{
    if (space.space_type() == spaces::SpaceType::Mixed) {
        const auto* mixed = dynamic_cast<const spaces::MixedSpace*>(&space);
        FE_THROW_IF(mixed == nullptr, InvalidArgumentException,
                    "FESystem::computeSetupStoragePlan: mixed space does not expose MixedSpace interface");
        FE_THROW_IF(mixed->num_components() == 0u, InvalidArgumentException,
                    "FESystem::computeSetupStoragePlan: mixed space has no component spaces");
        SetupStorageRequirements req;
        for (std::size_t i = 0; i < mixed->num_components(); ++i) {
            const auto& component = mixed->component(i);
            FE_CHECK_NOT_NULL(component.space.get(),
                              "FESystem::computeSetupStoragePlan: mixed component space");
            req.merge(storageRequirementsForSpace(*component.space, mesh_access));
        }
        return req;
    }

    if (space.space_type() == spaces::SpaceType::Composite) {
        const auto* composite = dynamic_cast<const spaces::CompositeSpace*>(&space);
        FE_THROW_IF(composite == nullptr, InvalidArgumentException,
                    "FESystem::computeSetupStoragePlan: composite space does not expose CompositeSpace interface");
        FE_THROW_IF(composite->num_regions() == 0u, InvalidArgumentException,
                    "FESystem::computeSetupStoragePlan: composite space has no region spaces");
        SetupStorageRequirements req;
        for (std::size_t i = 0; i < composite->num_regions(); ++i) {
            const auto& region = composite->region(i);
            FE_CHECK_NOT_NULL(region.space.get(),
                              "FESystem::computeSetupStoragePlan: composite region space");
            req.merge(storageRequirementsForSpace(*region.space, mesh_access));
        }
        return req;
    }

    FE_THROW_IF(space.is_variable_order() && mesh_access == nullptr,
                InvalidArgumentException,
                "FESystem::computeSetupStoragePlan: variable-order storage requirements need mesh access");

    SetupStorageRequirements req;
    req.vertex_topology = true;
    req.cell_topology = true;

    const auto continuity = space.continuity();
    if (continuity == Continuity::C0 || continuity == Continuity::C1) {
        if (mesh_access != nullptr && mesh_access->numCells() > 0) {
            std::vector<GlobalIndex> nodes;
            for (GlobalIndex cell_id = 0; cell_id < mesh_access->numCells(); ++cell_id) {
                getCellCornerNodes(*mesh_access, cell_id, nodes);
                FE_THROW_IF(nodes.empty(), InvalidArgumentException,
                            "FESystem::computeSetupStoragePlan: cell has no vertices");
                const auto layout = dofs::DofLayoutInfo::Lagrange(
                    space.polynomial_order(cell_id),
                    mesh_access->dimension(),
                    static_cast<int>(nodes.size()),
                    space.value_dimension());
                req.edge_topology = req.edge_topology || layout.dofs_per_edge > 0;
                req.interior_face_topology = req.interior_face_topology || layout.has_face_dofs();
            }
        } else {
            const int n_vertices = referenceVertexCount(space.element_type());
            FE_THROW_IF(n_vertices <= 0, InvalidArgumentException,
                        "FESystem::computeSetupStoragePlan: cannot infer reference vertex count for field space");
            const auto layout = dofs::DofLayoutInfo::Lagrange(
                space.polynomial_order(),
                space.topological_dimension(),
                n_vertices,
                space.value_dimension());
            req.edge_topology = layout.dofs_per_edge > 0;
            req.interior_face_topology = layout.has_face_dofs();
        }
        return req;
    }

    if (continuity == Continuity::L2) {
        return req;
    }

    if (continuity == Continuity::H_curl || continuity == Continuity::H_div) {
        const auto& basis = space.element().basis();
        FE_THROW_IF(!basis.is_vector_valued(), InvalidArgumentException,
                    "FESystem::computeSetupStoragePlan: H(curl)/H(div) space must use a vector-valued basis");
        const auto* vb = dynamic_cast<const basis::VectorBasisFunction*>(&basis);
        FE_THROW_IF(vb == nullptr, InvalidArgumentException,
                    "FESystem::computeSetupStoragePlan: vector basis does not expose DOF associations");

        for (const auto& assoc : vb->dof_associations()) {
            switch (assoc.entity_type) {
                case basis::DofEntity::Edge:
                    req.edge_topology = true;
                    break;
                case basis::DofEntity::Face:
                    req.interior_face_topology = true;
                    req.face_gids = true;
                    break;
                case basis::DofEntity::Vertex:
                case basis::DofEntity::Interior:
                    break;
            }
        }
        return req;
    }

    FE_THROW(InvalidArgumentException,
             "FESystem::computeSetupStoragePlan: unsupported field-space continuity for exact storage planning");
}

void mergeKernelDomainRequirements(const assembly::AssemblyKernel* kernel,
                                   SetupStorageRequirements& req)
{
    if (kernel == nullptr) {
        return;
    }
    if (kernel->hasBoundaryFace()) {
        req.boundary_face_topology = true;
    }
    if (kernel->hasInteriorFace()) {
        req.interior_face_topology = true;
    }
    if (kernel->hasSingleSidedInterfaceFace()) {
        req.interface_face_topology = true;
    }
}

void mergeQuantityRequirements(const FEQuantityDefinition& def,
                               SetupStorageRequirements& req)
{
    switch (def.kind) {
        case FEQuantityKind::BoundaryIntegral:
        case FEQuantityKind::BoundaryAverage:
            req.boundary_face_topology = true;
            break;
        case FEQuantityKind::BoundaryNodalSum:
            req.boundary_face_topology = true;
            req.entity_dof_map = true;
            break;
        case FEQuantityKind::SampledField:
            req.entity_dof_map = true;
            req.vertex_topology = true;
            break;
        case FEQuantityKind::DomainIntegral:
        case FEQuantityKind::DomainAverage:
        case FEQuantityKind::RegionIntegral:
        case FEQuantityKind::RegionAverage:
        case FEQuantityKind::FEExpression:
        case FEQuantityKind::DerivedCallback:
        default:
            break;
    }
}

class AffineConstraintsQuery final : public sparsity::IConstraintQuery {
public:
    explicit AffineConstraintsQuery(constraints::AffineConstraints constraints)
        : constraints_(std::move(constraints))
    {}

    [[nodiscard]] bool isConstrained(GlobalIndex dof) const override
    {
        return constraints_.isConstrained(dof);
    }

    [[nodiscard]] std::vector<GlobalIndex> getMasterDofs(GlobalIndex constrained_dof) const override
    {
        auto view = constraints_.getConstraint(constrained_dof);
        if (!view.has_value()) {
            return {};
        }
        std::vector<GlobalIndex> masters;
        masters.reserve(view->entries.size());
        for (const auto& entry : view->entries) {
            masters.push_back(entry.master_dof);
        }
        return masters;
    }

    [[nodiscard]] std::vector<GlobalIndex> getAllConstrainedDofs() const override
    {
        return constraints_.getConstrainedDofs();
    }

    [[nodiscard]] std::size_t numConstraints() const override
    {
        return constraints_.numConstraints();
    }

private:
    constraints::AffineConstraints constraints_;
};

class PermutedAffineConstraintsQuery final : public sparsity::IConstraintQuery {
public:
    PermutedAffineConstraintsQuery(const constraints::AffineConstraints& constraints,
                                   std::span<const GlobalIndex> forward,
                                   std::span<const GlobalIndex> inverse)
        : constraints_(&constraints)
        , forward_(forward)
        , inverse_(inverse)
    {}

    [[nodiscard]] bool isConstrained(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        if (dof < 0 || static_cast<std::size_t>(dof) >= inverse_.size()) {
            return false;
        }
        const auto fe = inverse_[static_cast<std::size_t>(dof)];
        if (fe < 0) {
            return false;
        }
        return constraints_->isConstrained(fe);
    }

    [[nodiscard]] std::vector<GlobalIndex> getMasterDofs(GlobalIndex constrained_dof) const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        if (constrained_dof < 0 || static_cast<std::size_t>(constrained_dof) >= inverse_.size()) {
            return {};
        }
        const auto fe_constrained = inverse_[static_cast<std::size_t>(constrained_dof)];
        if (fe_constrained < 0) {
            return {};
        }

        auto view = constraints_->getConstraint(fe_constrained);
        if (!view.has_value()) {
            return {};
        }

        std::vector<GlobalIndex> masters;
        masters.reserve(view->entries.size());
        for (const auto& entry : view->entries) {
            const auto fe_master = entry.master_dof;
            if (fe_master < 0 || static_cast<std::size_t>(fe_master) >= forward_.size()) {
                continue;
            }
            const auto fs_master = forward_[static_cast<std::size_t>(fe_master)];
            if (fs_master < 0) {
                continue;
            }
            masters.push_back(fs_master);
        }
        return masters;
    }

    [[nodiscard]] std::vector<GlobalIndex> getAllConstrainedDofs() const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        const auto fe_dofs = constraints_->getConstrainedDofs();
        std::vector<GlobalIndex> fs_dofs;
        fs_dofs.reserve(fe_dofs.size());
        for (const auto fe : fe_dofs) {
            if (fe < 0 || static_cast<std::size_t>(fe) >= forward_.size()) {
                continue;
            }
            const auto fs = forward_[static_cast<std::size_t>(fe)];
            if (fs < 0) {
                continue;
            }
            fs_dofs.push_back(fs);
        }
        std::sort(fs_dofs.begin(), fs_dofs.end());
        fs_dofs.erase(std::unique(fs_dofs.begin(), fs_dofs.end()), fs_dofs.end());
        return fs_dofs;
    }

    [[nodiscard]] std::size_t numConstraints() const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        return constraints_->numConstraints();
    }

private:
    const constraints::AffineConstraints* constraints_{nullptr};
    std::span<const GlobalIndex> forward_{};
    std::span<const GlobalIndex> inverse_{};
};

struct NodalInterleavedDofMap {
    int dof_per_node{0};
    GlobalIndex n_nodes{0};
    sparsity::IndexRange owned_range{};

    std::vector<GlobalIndex> fe_to_fs{};
    std::vector<GlobalIndex> fs_to_fe{};

    std::vector<int> ghost_nodes{};
    std::vector<unsigned char> node_is_relevant{};
    std::vector<unsigned char> node_is_ghost{};
    std::vector<int> node_owner_rank{};

    [[nodiscard]] bool isGhostNode(int node) const noexcept
    {
        if (node < 0 || static_cast<std::size_t>(node) >= node_is_ghost.size()) {
            return false;
        }
        return node_is_ghost[static_cast<std::size_t>(node)] != 0;
    }

    [[nodiscard]] bool isRelevantDof(GlobalIndex dof) const noexcept
    {
        if (dof_per_node <= 0 || dof < 0) {
            return false;
        }
        const auto node = static_cast<GlobalIndex>(dof / dof_per_node);
        if (node < 0 || static_cast<std::size_t>(node) >= node_is_relevant.size()) {
            return false;
        }
        return node_is_relevant[static_cast<std::size_t>(node)] != 0;
    }

    [[nodiscard]] GlobalIndex mapFeToFs(GlobalIndex fe_dof) const noexcept
    {
        if (fe_dof < 0 || static_cast<std::size_t>(fe_dof) >= fe_to_fs.size()) {
            return INVALID_GLOBAL_INDEX;
        }
        return fe_to_fs[static_cast<std::size_t>(fe_dof)];
    }

    [[nodiscard]] int ownerRankForDof(GlobalIndex dof) const noexcept
    {
        if (dof_per_node <= 0 || dof < 0) {
            return -1;
        }
        const auto node = static_cast<GlobalIndex>(dof / dof_per_node);
        if (node < 0 || static_cast<std::size_t>(node) >= node_owner_rank.size()) {
            return -1;
        }
        return node_owner_rank[static_cast<std::size_t>(node)];
    }
};

namespace {

constexpr std::uint32_t kSfcBits = 21u;
constexpr std::uint64_t kSfcMaxCoord = (1ULL << kSfcBits) - 1ULL;

[[nodiscard]] std::uint64_t morton3D(std::uint32_t xi, std::uint32_t yi, std::uint32_t zi)
{
    auto spread = [](std::uint64_t v) -> std::uint64_t {
        v = (v | (v << 32)) & 0x1f00000000ffffULL;
        v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
        v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
        v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
        v = (v | (v << 2)) & 0x1249249249249249ULL;
        return v;
    };

    const std::uint64_t x = spread(xi);
    const std::uint64_t y = spread(yi);
    const std::uint64_t z = spread(zi);
    return x | (y << 1) | (z << 2);
}

[[nodiscard]] std::uint64_t hilbertIndexND(std::span<const std::uint32_t> coords, std::uint32_t bits)
{
    const int n = static_cast<int>(coords.size());
    if (n <= 0 || bits == 0u) {
        return 0;
    }
    FE_THROW_IF(n > 3, InvalidArgumentException, "hilbertIndexND: only 2D/3D supported");
    FE_THROW_IF(bits > 21u, InvalidArgumentException, "hilbertIndexND: bits must be <= 21 for uint64 index");

    std::array<std::uint32_t, 3> x{0u, 0u, 0u};
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = coords[static_cast<std::size_t>(i)];
    }

    // John Skilling, "Programming the Hilbert curve" (2004): coord -> Hilbert index (transpose form).
    std::uint32_t M = 1u << (bits - 1u);
    for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
        const std::uint32_t P = Q - 1u;
        for (int i = 0; i < n; ++i) {
            if ((x[static_cast<std::size_t>(i)] & Q) != 0u) {
                x[0] ^= P;
            } else {
                const std::uint32_t t = (x[0] ^ x[static_cast<std::size_t>(i)]) & P;
                x[0] ^= t;
                x[static_cast<std::size_t>(i)] ^= t;
            }
        }
    }

    for (int i = 1; i < n; ++i) {
        x[static_cast<std::size_t>(i)] ^= x[static_cast<std::size_t>(i - 1)];
    }

    std::uint32_t t = 0u;
    for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
        if ((x[static_cast<std::size_t>(n - 1)] & Q) != 0u) {
            t ^= (Q - 1u);
        }
    }
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] ^= t;
    }

    // Interleave transpose bits into a single integer.
    std::uint64_t index = 0;
    for (int b = static_cast<int>(bits) - 1; b >= 0; --b) {
        for (int i = 0; i < n; ++i) {
            index <<= 1u;
            index |= static_cast<std::uint64_t>((x[static_cast<std::size_t>(i)] >> static_cast<std::uint32_t>(b)) & 1u);
        }
    }
    return index;
}

[[nodiscard]] std::uint64_t sfcCodeNormalized(double x, double y, double z, dofs::SpatialCurveType curve)
{
    auto normalize = [](double v) -> std::uint32_t {
        v = std::max(0.0, std::min(1.0, v));
        return static_cast<std::uint32_t>(v * static_cast<double>(kSfcMaxCoord));
    };

    const std::uint32_t xi = normalize(x);
    const std::uint32_t yi = normalize(y);
    const std::uint32_t zi = normalize(z);

    if (curve == dofs::SpatialCurveType::Hilbert) {
        const std::array<std::uint32_t, 3> c{xi, yi, zi};
        return hilbertIndexND(std::span<const std::uint32_t>(c.data(), 3), kSfcBits);
    }
    return morton3D(xi, yi, zi);
}

} // namespace

[[nodiscard]] std::optional<NodalInterleavedDofMap> tryBuildNodalInterleavedDofMap(const dofs::DofHandler& dof_handler,
                                                                                   const dofs::FieldDofMap& field_map,
                                                                                   const assembly::IMeshAccess& mesh,
                                                                                   const dofs::DofDistributionOptions& dof_options)
{
    auto fail = [&](const std::string& reason) -> std::optional<NodalInterleavedDofMap> {
        if (dofPermutationTraceEnabled()) {
            std::ostringstream oss;
            oss << "FESystem::setup: nodal dof permutation unavailable: " << reason;
            FE_LOG_INFO(oss.str());
        }
        return std::nullopt;
    };

    const GlobalIndex total_dofs = dof_handler.getNumDofs();
    if (total_dofs <= 0) {
        return fail("total_dofs<=0");
    }

    const std::size_t n_fields = field_map.numFields();
    if (n_fields == 0u) {
        return fail("n_fields==0");
    }

    GlobalIndex n_nodes = -1;
    int dof_per_node = 0;
    for (std::size_t f = 0; f < n_fields; ++f) {
        const auto& field = field_map.getField(f);
        // Vector-basis spaces such as H(div)/H(curl) do not expose a meaningful node-interleaved
        // backend permutation. In that case, simply report that no nodal map is available and let
        // the caller fall back to the natural owner-contiguous FE numbering when possible.
        if (field.component_dof_layout != dofs::FieldComponentDofLayout::ComponentWise) {
            return fail("field component_dof_layout is not ComponentWise");
        }
        FE_THROW_IF(field.n_components <= 0, InvalidStateException,
                    "FESystem::setup: invalid field components for node-interleaved distributed sparsity");
        FE_THROW_IF(field.n_dofs % field.n_components != 0, InvalidStateException,
                    "FESystem::setup: field DOF count must be divisible by components for node-interleaved distributed sparsity");

        const GlobalIndex n_per_component = field.n_dofs / field.n_components;
        if (n_nodes < 0) {
            n_nodes = n_per_component;
        } else if (n_nodes != n_per_component) {
            return fail("fields have different dofs-per-component counts");
        }
        dof_per_node += field.n_components;
    }
    if (n_nodes <= 0 || dof_per_node <= 0) {
        return fail("n_nodes<=0 or dof_per_node<=0");
    }
    if (total_dofs != static_cast<GlobalIndex>(dof_per_node) * n_nodes) {
        return fail("total_dofs != dof_per_node*n_nodes");
    }

#if !FE_HAS_MPI
    (void)dof_handler;
    (void)field_map;
    (void)mesh;
    (void)dof_options;
    return fail("FE_HAS_MPI disabled");
#else
    // In MPI, overlap backends (FSILS) require that owned rows form a contiguous range in the
    // backend (node-interleaved) indexing. When FE global numbering is process-count independent
    // (e.g., DenseGlobalIds), owned nodes are typically non-contiguous in node space. To keep the
    // backend happy without changing FE numbering, build a backend node permutation that groups
    // nodes by owner rank (owner-contiguous), and optionally orders nodes spatially within each
    // owner block (Morton/Hilbert).

    const int my_rank = dof_options.my_rank;
    const int world_size = dof_options.world_size;
    if (my_rank < 0 || world_size <= 0 || my_rank >= world_size) {
        return fail("invalid my_rank/world_size");
    }

    // Spatial ordering within the owner block is optional but default-on (note 18).
    const bool explicit_spatial =
        dof_options.numbering == dofs::DofNumberingStrategy::Morton ||
        dof_options.numbering == dofs::DofNumberingStrategy::Hilbert;
    const bool default_spatial =
        dof_options.enable_spatial_locality_ordering &&
        dof_options.numbering == dofs::DofNumberingStrategy::Sequential;
    const bool want_spatial = explicit_spatial || default_spatial;
    const dofs::SpatialCurveType curve =
        explicit_spatial
            ? (dof_options.numbering == dofs::DofNumberingStrategy::Hilbert ? dofs::SpatialCurveType::Hilbert
                                                                           : dofs::SpatialCurveType::Morton)
            : dof_options.spatial_curve;

    enum class NodeLayout { Block, Interleaved };
    // Decode per-field node/component indices using the actual monolithic field-map layout
    // (which may differ from dof_options.numbering when default spatial locality ordering is enabled).
    const NodeLayout node_layout =
        (field_map.layout() == dofs::FieldLayout::Block) ? NodeLayout::Block : NodeLayout::Interleaved;

    const auto& part = dof_handler.getPartition();
    const auto owned_size = part.localOwnedSize();
    if (owned_size < 0) {
        return fail("owned_size<0");
    }

    // Representative field/component used to identify node ownership. Prefer the field with the
    // fewest components so mixed systems can use a scalar nodal field (e.g. pressure) instead of a
    // block-numbered vector field whose components may be partitioned separately.
    std::size_t rep_field_idx = 0u;
    LocalIndex rep_components = std::numeric_limits<LocalIndex>::max();
    for (std::size_t f = 0; f < n_fields; ++f) {
        const auto& field = field_map.getField(f);
        if (field.component_dof_layout != dofs::FieldComponentDofLayout::ComponentWise ||
            field.n_components <= 0) {
            continue;
        }
        if (field.n_components < rep_components) {
            rep_components = field.n_components;
            rep_field_idx = f;
        }
    }
    const auto& rep_field = field_map.getField(rep_field_idx);
    if (rep_field.component_dof_layout != dofs::FieldComponentDofLayout::ComponentWise ||
        rep_field.n_components <= 0) {
        return fail("representative field is not component-wise");
    }

    const GlobalIndex rep_n_nodes = rep_field.n_dofs / std::max<GlobalIndex>(1, rep_field.n_components);
    if (rep_n_nodes != n_nodes) {
        return fail("representative field node count mismatch");
    }

    std::vector<int> node_owner(static_cast<std::size_t>(n_nodes), -1);
    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const GlobalIndex fe0 = field_map.componentToGlobal(rep_field_idx, 0, node);
        if (fe0 < 0 || fe0 >= total_dofs) {
            return fail("representative componentToGlobal(rep_field,0,node) out of range");
        }

        const int owner = dof_handler.getDofMap().getDofOwner(fe0);
        if (owner < 0 || owner >= world_size) {
            return fail("representative dof owner outside [0,world_size)");
        }
        node_owner[static_cast<std::size_t>(node)] = owner;
    }

    auto decode_node_comp = [&](const dofs::FieldDescriptor& field,
                                GlobalIndex local_dof) -> std::optional<std::pair<GlobalIndex, LocalIndex>> {
        if (local_dof < 0 || local_dof >= field.n_dofs || field.n_components <= 0) {
            return std::nullopt;
        }
        if (node_layout == NodeLayout::Interleaved) {
            const auto c = static_cast<LocalIndex>(local_dof % field.n_components);
            const auto node = local_dof / field.n_components;
            return std::make_pair(node, c);
        }
        const GlobalIndex n_per_component = field.n_dofs / field.n_components;
        if (n_per_component <= 0) {
            return std::nullopt;
        }
        const auto c = static_cast<LocalIndex>(local_dof / n_per_component);
	        const auto node = local_dof % n_per_component;
	        return std::make_pair(node, c);
	    };

	    // Derive locally relevant nodes across all fields/components. Node ownership comes from the
	    // representative component above so that block-numbered vector fields can still build a nodal
	    // backend map even when different components of the same physical node are owned by different
	    // FE ranks.
	    std::vector<GlobalIndex> relevant_nodes_fe;
	    {
	        const auto relevant_dofs = part.locallyRelevant().toVector();
	        relevant_nodes_fe.reserve(relevant_dofs.size() /
	                                      static_cast<std::size_t>(std::max<GlobalIndex>(1, dof_per_node)) +
	                                  8u);

	        for (const auto fe : relevant_dofs) {
            const auto fld = field_map.globalToField(fe);
            if (!fld.has_value()) {
                continue;
            }
	            const auto& field = field_map.getField(static_cast<std::size_t>(fld->first));
            const auto decoded = decode_node_comp(field, fld->second);
            if (!decoded.has_value()) {
                return fail("decode_node_comp failed");
            }
            const auto node = decoded->first;
            if (node < 0 || node >= n_nodes) {
                return fail("decoded node outside [0,n_nodes)");
            }

	            relevant_nodes_fe.push_back(node);
	        }
	    }

    std::sort(relevant_nodes_fe.begin(), relevant_nodes_fe.end());
    relevant_nodes_fe.erase(std::unique(relevant_nodes_fe.begin(), relevant_nodes_fe.end()), relevant_nodes_fe.end());

    FE_THROW_IF(owned_size % static_cast<GlobalIndex>(dof_per_node) != 0,
	                InvalidArgumentException,
	                "FESystem::setup: nodal interleaved distributed sparsity requires owned DOFs to be a multiple of dof_per_node");
	    const GlobalIndex expected_owned_nodes = owned_size / static_cast<GlobalIndex>(dof_per_node);

	    std::vector<GlobalIndex> owned_nodes_fe;
	    std::vector<GlobalIndex> ghost_nodes_fe;
	    owned_nodes_fe.reserve(static_cast<std::size_t>(expected_owned_nodes) + 8u);
	    ghost_nodes_fe.reserve(relevant_nodes_fe.size());
	    for (const auto node : relevant_nodes_fe) {
	        const int owner = node_owner[static_cast<std::size_t>(node)];
	        if (owner < 0 || owner >= world_size) {
	            return fail("owned/ghost node missing valid owner entry");
	        }
	        if (owner == my_rank) {
	            owned_nodes_fe.push_back(node);
	        } else {
	            ghost_nodes_fe.push_back(node);
	        }
	    }

	    if (static_cast<GlobalIndex>(owned_nodes_fe.size()) != expected_owned_nodes) {
	        std::ostringstream oss;
	        oss << "owned_nodes_fe.size()!=expected_owned_nodes"
	            << " owned_nodes_fe=" << owned_nodes_fe.size()
	            << " expected=" << expected_owned_nodes
	            << " relevant_nodes_fe=" << relevant_nodes_fe.size()
	            << " owned_size=" << owned_size
	            << " dof_per_node=" << dof_per_node;
	        return fail(oss.str());
	    }

    // Compute a spatial (Morton/Hilbert) ordering for owned nodes within this rank's owner block.
    struct OwnedNodeRec {
        std::uint64_t code{0};
        GlobalIndex node{-1};
        std::array<double, 3> xyz{0.0, 0.0, 0.0};
    };
    std::vector<OwnedNodeRec> owned_recs;
    owned_recs.reserve(owned_nodes_fe.size());

    const auto* emap = dof_handler.getEntityDofMap();
    const int dim = std::max(2, mesh.dimension());

    std::array<double, 3> min_xyz{std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};
    std::array<double, 3> max_xyz{-std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity()};

    for (const auto node : owned_nodes_fe) {
        const GlobalIndex fe0 = field_map.componentToGlobal(0, 0, node);
        if (fe0 < 0 || fe0 >= total_dofs) {
            return fail("componentToGlobal(0,0,node) out of range");
        }

        std::array<double, 3> xyz{static_cast<double>(node), static_cast<double>(node), static_cast<double>(node)};
        bool have_xyz = false;
        if (want_spatial && emap) {
            if (const auto ent = emap->getDofEntity(fe0); ent && ent->kind == dofs::EntityKind::Vertex) {
                const auto p = mesh.getNodeCoordinates(ent->id);
                xyz = {static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
                have_xyz = true;
            }
        }
        (void)have_xyz;

        for (int a = 0; a < dim && a < 3; ++a) {
            const auto ax = static_cast<std::size_t>(a);
            min_xyz[ax] = std::min(min_xyz[ax], xyz[ax]);
            max_xyz[ax] = std::max(max_xyz[ax], xyz[ax]);
        }

        owned_recs.push_back(OwnedNodeRec{0u, node, xyz});
    }

    auto normalize_axis = [&](double v, int axis) -> double {
        const auto ax = static_cast<std::size_t>(axis);
        const double lo = min_xyz[ax];
        const double hi = max_xyz[ax];
        if (!(hi > lo)) {
            return 0.0;
        }
        return (v - lo) / (hi - lo);
    };

    if (want_spatial) {
        for (auto& rec : owned_recs) {
            const double x = normalize_axis(rec.xyz[0], 0);
            const double y = (dim >= 2) ? normalize_axis(rec.xyz[1], 1) : 0.0;
            const double z = (dim >= 3) ? normalize_axis(rec.xyz[2], 2) : 0.0;
            rec.code = sfcCodeNormalized(x, y, z, curve);
        }
    }

    std::sort(owned_recs.begin(), owned_recs.end(), [&](const OwnedNodeRec& a, const OwnedNodeRec& b) {
        if (a.code != b.code) {
            return a.code < b.code;
        }
        return a.node < b.node;
    });

    const std::int64_t owned_count_local = static_cast<std::int64_t>(owned_recs.size());
    std::vector<std::int64_t> owned_counts(static_cast<std::size_t>(world_size), 0);
    MPI_Allgather(&owned_count_local, 1, MPI_INT64_T,
                  owned_counts.data(), 1, MPI_INT64_T, dof_options.mpi_comm);

    std::vector<std::int64_t> node_offsets(static_cast<std::size_t>(world_size) + 1u, 0);
    std::int64_t node_total = 0;
    for (int r = 0; r < world_size; ++r) {
        node_offsets[static_cast<std::size_t>(r)] = node_total;
        node_total += owned_counts[static_cast<std::size_t>(r)];
    }
    node_offsets[static_cast<std::size_t>(world_size)] = node_total;
    const std::int64_t node_offset = node_offsets[static_cast<std::size_t>(my_rank)];
    if (node_total != static_cast<std::int64_t>(n_nodes)) {
        std::ostringstream oss;
        oss << "sum owned node counts != n_nodes"
            << " node_total=" << node_total
            << " n_nodes=" << n_nodes;
        return fail(oss.str());
    }

    std::vector<int> owned_counts_int(static_cast<std::size_t>(world_size), 0);
    std::vector<int> owned_displs_int(static_cast<std::size_t>(world_size), 0);
    for (int r = 0; r < world_size; ++r) {
        if (owned_counts[static_cast<std::size_t>(r)] >
            static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            return fail("owned node count exceeds MPI int range");
        }
        if (node_offsets[static_cast<std::size_t>(r)] >
            static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            return fail("owned node displacement exceeds MPI int range");
        }
        owned_counts_int[static_cast<std::size_t>(r)] =
            static_cast<int>(owned_counts[static_cast<std::size_t>(r)]);
        owned_displs_int[static_cast<std::size_t>(r)] =
            static_cast<int>(node_offsets[static_cast<std::size_t>(r)]);
    }

    std::vector<GlobalIndex> local_owned_nodes(owned_recs.size(), INVALID_GLOBAL_INDEX);
    for (std::size_t i = 0; i < owned_recs.size(); ++i) {
        local_owned_nodes[i] = owned_recs[i].node;
    }

    std::vector<GlobalIndex> all_owned_nodes(static_cast<std::size_t>(node_total), INVALID_GLOBAL_INDEX);
    MPI_Allgatherv(local_owned_nodes.data(),
                   static_cast<int>(local_owned_nodes.size()),
                   MPI_INT64_T,
                   all_owned_nodes.data(),
                   owned_counts_int.data(),
                   owned_displs_int.data(),
                   MPI_INT64_T,
                   dof_options.mpi_comm);

    std::unordered_map<GlobalIndex, GlobalIndex> fe_node_to_backend;
    fe_node_to_backend.reserve(static_cast<std::size_t>(n_nodes));
    for (int r = 0; r < world_size; ++r) {
        const auto begin = static_cast<std::size_t>(node_offsets[static_cast<std::size_t>(r)]);
        const auto count = static_cast<std::size_t>(owned_counts[static_cast<std::size_t>(r)]);
        for (std::size_t i = 0; i < count; ++i) {
            const auto node = all_owned_nodes[begin + i];
            if (node < 0 || node >= n_nodes) {
                return fail("gathered owned FE node outside [0,n_nodes)");
            }
            const GlobalIndex backend_node =
                static_cast<GlobalIndex>(node_offsets[static_cast<std::size_t>(r)] +
                                         static_cast<std::int64_t>(i));
            const auto [it, inserted] = fe_node_to_backend.emplace(node, backend_node);
            if (!inserted && it->second != backend_node) {
                return fail("global FE node to backend node map conflict");
            }
        }
    }

	    // Resolve backend node ids for ghost nodes by querying their owner ranks.
	    std::vector<std::vector<GlobalIndex>> requests(static_cast<std::size_t>(world_size));
	    for (const auto node : ghost_nodes_fe) {
	        const int owner = node_owner[static_cast<std::size_t>(node)];
	        if (owner < 0 || owner >= world_size) {
	            return fail("ghost node owner outside [0,world_size)");
	        }
	        if (owner == my_rank) {
            continue;
        }
        requests[static_cast<std::size_t>(owner)].push_back(node);
    }
    for (auto& v : requests) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    std::vector<int> send_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> recv_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> send_displs(static_cast<std::size_t>(world_size), 0);
    std::vector<int> recv_displs(static_cast<std::size_t>(world_size), 0);

    std::vector<GlobalIndex> send_nodes;
    send_nodes.reserve(ghost_nodes_fe.size());
    for (int r = 0; r < world_size; ++r) {
        const auto& v = requests[static_cast<std::size_t>(r)];
        send_counts[static_cast<std::size_t>(r)] = static_cast<int>(v.size());
    }
    int total_send = 0;
    for (int r = 0; r < world_size; ++r) {
        send_displs[static_cast<std::size_t>(r)] = total_send;
        total_send += send_counts[static_cast<std::size_t>(r)];
    }
    send_nodes.resize(static_cast<std::size_t>(total_send));
    for (int r = 0; r < world_size; ++r) {
        const auto& v = requests[static_cast<std::size_t>(r)];
        std::copy(v.begin(), v.end(), send_nodes.begin() + send_displs[static_cast<std::size_t>(r)]);
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, dof_options.mpi_comm);
    int total_recv = 0;
    for (int r = 0; r < world_size; ++r) {
        recv_displs[static_cast<std::size_t>(r)] = total_recv;
        total_recv += recv_counts[static_cast<std::size_t>(r)];
    }

    std::vector<GlobalIndex> recv_nodes(static_cast<std::size_t>(total_recv));
    MPI_Alltoallv(send_nodes.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  recv_nodes.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                  dof_options.mpi_comm);

    std::vector<GlobalIndex> send_backends(static_cast<std::size_t>(total_recv), INVALID_GLOBAL_INDEX);
    for (std::size_t i = 0; i < recv_nodes.size(); ++i) {
        const auto node = recv_nodes[i];
        const auto it = fe_node_to_backend.find(node);
        if (it == fe_node_to_backend.end()) {
            return fail("owner did not resolve requested ghost node backend id");
        }
        send_backends[i] = it->second;
    }

    std::vector<GlobalIndex> recv_backends(static_cast<std::size_t>(total_send), INVALID_GLOBAL_INDEX);
    MPI_Alltoallv(send_backends.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                  recv_backends.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  dof_options.mpi_comm);

    std::vector<int> ghost_nodes_backend;
    ghost_nodes_backend.reserve(ghost_nodes_fe.size());
    for (std::size_t i = 0; i < send_nodes.size(); ++i) {
        const auto node = send_nodes[i];
        const auto backend = recv_backends[i];
        if (backend < 0 || backend >= n_nodes) {
            return fail("received ghost backend id outside [0,n_nodes)");
        }
        fe_node_to_backend.emplace(node, backend);
        ghost_nodes_backend.push_back(static_cast<int>(backend));
    }

    std::sort(ghost_nodes_backend.begin(), ghost_nodes_backend.end());
    ghost_nodes_backend.erase(std::unique(ghost_nodes_backend.begin(), ghost_nodes_backend.end()),
                              ghost_nodes_backend.end());

    NodalInterleavedDofMap map;
    map.dof_per_node = dof_per_node;
    map.n_nodes = n_nodes;
    map.owned_range.first = static_cast<GlobalIndex>(node_offset) * dof_per_node;
    map.owned_range.last = static_cast<GlobalIndex>(node_offset + owned_count_local) * dof_per_node;
    map.ghost_nodes = std::move(ghost_nodes_backend);
    map.node_is_ghost.assign(static_cast<std::size_t>(n_nodes), 0);
    map.node_is_relevant.assign(static_cast<std::size_t>(n_nodes), 0);
    map.node_owner_rank.assign(static_cast<std::size_t>(n_nodes), -1);

    const GlobalIndex owned_node_start = static_cast<GlobalIndex>(node_offset);
    const GlobalIndex owned_node_end = static_cast<GlobalIndex>(node_offset + owned_count_local);
    for (GlobalIndex node = owned_node_start; node < owned_node_end; ++node) {
        if (node < 0 || node >= n_nodes) {
            return std::nullopt;
        }
        map.node_is_relevant[static_cast<std::size_t>(node)] = 1;
    }
    for (const int node : map.ghost_nodes) {
        if (node < 0 || static_cast<GlobalIndex>(node) >= n_nodes) {
            return fail("ghost backend node outside [0,n_nodes)");
        }
        map.node_is_ghost[static_cast<std::size_t>(node)] = 1;
        map.node_is_relevant[static_cast<std::size_t>(node)] = 1;
    }

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const auto it = fe_node_to_backend.find(node);
        if (it == fe_node_to_backend.end()) {
            return fail("FE node missing backend mapping for owner ranks");
        }
        const GlobalIndex backend_node = it->second;
        if (backend_node < 0 || backend_node >= n_nodes) {
            return fail("backend owner node outside [0,n_nodes)");
        }
        const int owner = node_owner[static_cast<std::size_t>(node)];
        if (owner < 0 || owner >= world_size) {
            return fail("backend owner rank outside [0,world_size)");
        }
        map.node_owner_rank[static_cast<std::size_t>(backend_node)] = owner;
    }

	    map.fe_to_fs.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);
	    map.fs_to_fe.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);

	    // Fill a globally complete FE<->backend permutation. The FSILS matrix
        // still stores only owned rows plus the local halo, but PETSc-like
        // remote row insertion can carry arbitrary global FE columns that the
        // receiving rank must be able to map into backend column IDs.
	    for (GlobalIndex node = 0; node < n_nodes; ++node) {
	        const auto it = fe_node_to_backend.find(node);
	        if (it == fe_node_to_backend.end()) {
	            return fail("FE node missing backend mapping");
	        }
        const GlobalIndex backend_node = it->second;
        if (backend_node < 0 || backend_node >= n_nodes) {
            return fail("backend node outside [0,n_nodes)");
        }

        int comp_offset = 0;
        for (std::size_t f = 0; f < n_fields; ++f) {
            const auto& field = field_map.getField(f);
            for (LocalIndex c = 0; c < field.n_components; ++c) {
                const GlobalIndex fe = field_map.componentToGlobal(f, c, node);
                const GlobalIndex fs =
                    backend_node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp_offset);
                if (fe < 0 || fe >= total_dofs || fs < 0 || fs >= total_dofs) {
                    return fail("fe or fs index outside [0,total_dofs)");
                }
                auto& fwd = map.fe_to_fs[static_cast<std::size_t>(fe)];
                auto& inv = map.fs_to_fe[static_cast<std::size_t>(fs)];
                if (fwd != INVALID_GLOBAL_INDEX && fwd != fs) {
                    return fail("forward permutation conflict");
                }
                if (inv != INVALID_GLOBAL_INDEX && inv != fe) {
                    return fail("inverse permutation conflict");
                }
                fwd = fs;
                inv = fe;
                ++comp_offset;
            }
        }
        if (comp_offset != dof_per_node) {
            return fail("comp_offset != dof_per_node");
        }
    }

    if (dofPermutationTraceEnabled()) {
        std::ostringstream oss;
        oss << "FESystem::setup: nodal dof permutation built"
            << " total_dofs=" << total_dofs
            << " n_nodes=" << n_nodes
            << " dof_per_node=" << dof_per_node
            << " owned_nodes=" << owned_nodes_fe.size()
            << " ghost_nodes=" << map.ghost_nodes.size()
            << " owned_range=[" << map.owned_range.first << "," << map.owned_range.last << ")";
        FE_LOG_INFO(oss.str());

        const std::size_t sample_nodes =
            std::min<std::size_t>(owned_recs.size(), std::size_t{4});
        for (std::size_t i = 0; i < sample_nodes; ++i) {
            const auto fe_node = owned_recs[i].node;
            const auto backend_it = fe_node_to_backend.find(fe_node);
            if (backend_it == fe_node_to_backend.end()) {
                continue;
            }
            std::ostringstream sample;
            sample << "FESystem::setup: nodal dof permutation sample"
                   << " rank=" << my_rank
                   << " fe_node=" << fe_node
                   << " backend_node=" << backend_it->second;
            int comp_offset = 0;
            for (std::size_t f = 0; f < n_fields; ++f) {
                const auto& field = field_map.getField(f);
                for (LocalIndex c = 0; c < field.n_components; ++c) {
                    const GlobalIndex fe = field_map.componentToGlobal(f, c, fe_node);
                    const GlobalIndex fs =
                        backend_it->second * static_cast<GlobalIndex>(dof_per_node) +
                        static_cast<GlobalIndex>(comp_offset);
                    sample << " [backend_comp=" << comp_offset
                           << " field=" << field.name
                           << " field_comp=" << static_cast<int>(c)
                           << " fe=" << fe
                           << " fs=" << fs << "]";
                    ++comp_offset;
                }
            }
            FE_LOG_INFO(sample.str());
        }
    }
    return map;
#endif // FE_HAS_MPI
}

[[nodiscard]] bool fieldHandlerGroupsComponentsByPoint(const dofs::DofDistributionOptions& dof_options)
{
    // The monolithic DofMap is assembled by offsetting each finalized field
    // DofHandler. FieldDofMap must therefore describe the field handler's actual
    // numbering, not just the requested high-level numbering strategy.
    //
    // Spatial renumbering sorts all component DOFs by their physical coordinate.
    // Component-wise H1/ProductSpace fields assign the same coordinate to each
    // component of a scalar DOF, so the stable (curve, old-dof) ordering groups
    // components adjacent to each other: [u0,v0], [u1,v1], ...
    switch (dof_options.numbering) {
        case dofs::DofNumberingStrategy::Interleaved:
        case dofs::DofNumberingStrategy::Morton:
        case dofs::DofNumberingStrategy::Hilbert:
            return true;
        case dofs::DofNumberingStrategy::Sequential: {
            bool applies_default_spatial =
                dof_options.enable_spatial_locality_ordering;
            if (dof_options.world_size > 1 &&
                (dof_options.global_numbering != dofs::GlobalNumberingMode::OwnerContiguous ||
                 dof_options.reproducible_across_communicators)) {
                applies_default_spatial = false;
            }
            return applies_default_spatial;
        }
        case dofs::DofNumberingStrategy::Block:
        case dofs::DofNumberingStrategy::Hierarchical:
        case dofs::DofNumberingStrategy::CuthillMcKee:
            return false;
    }
    return false;
}

[[nodiscard]] dofs::FieldLayout fieldLayoutForSystem(std::size_t n_fields,
                                                     const dofs::DofDistributionOptions& dof_options)
{
    const bool interleaved = fieldHandlerGroupsComponentsByPoint(dof_options);
    if (n_fields > 1u) {
        return interleaved ? dofs::FieldLayout::FieldBlock : dofs::FieldLayout::Block;
    }
    return interleaved ? dofs::FieldLayout::Interleaved : dofs::FieldLayout::Block;
}

LocalIndex maxCellQuadraturePoints(const assembly::IMeshAccess& mesh,
                                   const spaces::FunctionSpace& test_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachCell([&](GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);

        auto quad_rule = test_element.quadrature();
        if (!quad_rule) {
            const int quad_order = quadrature::QuadratureFactory::recommended_order(
                test_element.polynomial_order(), false);
            quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
        }

        const auto n = static_cast<LocalIndex>(quad_rule->num_points());
        max_qpts = std::max(max_qpts, n);
    });

    return max_qpts;
}

ElementType faceTypeForFace(const assembly::IMeshAccess& mesh,
                            GlobalIndex face_id,
                            GlobalIndex cell_id)
{
    const ElementType cell_type = mesh.getCellType(cell_id);
    const LocalIndex local_face_id = mesh.getLocalFaceIndex(face_id, cell_id);

    elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face_id));

    switch (face_nodes.size()) {
        case 2:
            return ElementType::Line2;
        case 3:
            return ElementType::Triangle3;
        case 4:
            return ElementType::Quad4;
        default:
            throw FEException("FESystem::setup: unsupported face topology",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

LocalIndex maxBoundaryFaceQuadraturePoints(const assembly::IMeshAccess& mesh,
                                           const spaces::FunctionSpace& test_space,
                                           const spaces::FunctionSpace& trial_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);
        const auto& trial_element = trial_space.getElement(cell_type, cell_id);

        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            std::max(test_element.polynomial_order(), trial_element.polynomial_order()), false);

        const ElementType face_type = faceTypeForFace(mesh, face_id, cell_id);
        auto quad_rule = quadrature::QuadratureFactory::create(face_type, quad_order);
        max_qpts = std::max(max_qpts, static_cast<LocalIndex>(quad_rule->num_points()));
    });

    return max_qpts;
}

LocalIndex maxInteriorFaceQuadraturePoints(const assembly::IMeshAccess& mesh,
                                           const spaces::FunctionSpace& test_space,
                                           const spaces::FunctionSpace& trial_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
        const ElementType cell_type_minus = mesh.getCellType(cell_minus);
        const ElementType cell_type_plus = mesh.getCellType(cell_plus);

        const auto& test_element_minus = test_space.getElement(cell_type_minus, cell_minus);
        const auto& trial_element_minus = trial_space.getElement(cell_type_minus, cell_minus);
        const auto& test_element_plus = test_space.getElement(cell_type_plus, cell_plus);
        const auto& trial_element_plus = trial_space.getElement(cell_type_plus, cell_plus);

        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            std::max({test_element_minus.polynomial_order(),
                      trial_element_minus.polynomial_order(),
                      test_element_plus.polynomial_order(),
                      trial_element_plus.polynomial_order()}),
            false);

        const ElementType face_type_minus = faceTypeForFace(mesh, face_id, cell_minus);
        const ElementType face_type_plus = faceTypeForFace(mesh, face_id, cell_plus);
        FE_THROW_IF(face_type_minus != face_type_plus, InvalidStateException,
                    "FESystem::setup: interior face has mismatched face topology between adjacent cells");

        auto quad_rule = quadrature::QuadratureFactory::create(face_type_minus, quad_order);
        max_qpts = std::max(max_qpts, static_cast<LocalIndex>(quad_rule->num_points()));
    });

    return max_qpts;
}

} // namespace

SetupStoragePlan FESystem::computeSetupStoragePlan() const
{
    SetupStoragePlan plan;

    for (const auto& rec : field_registry_.records()) {
        FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::computeSetupStoragePlan: field.space");
        const auto* field_mesh_access =
            (rec.scope == FieldScope::VolumeCell) ? mesh_access_.get() : nullptr;
        auto req = storageRequirementsForSpace(*rec.space, field_mesh_access);
        if (rec.scope == FieldScope::InterfaceFace) {
            req.interface_face_topology = true;
        }
        plan.merge(req, "field:" + rec.name);
    }

    for (const auto& tag : operator_registry_.list()) {
        const auto& def = operator_registry_.get(tag);

        SetupStorageRequirements req;
        for (const auto& term : def.cells) {
            FE_CHECK_NOT_NULL(term.kernel.get(),
                              "FESystem::computeSetupStoragePlan: null cell kernel");
            mergeKernelDomainRequirements(term.kernel.get(), req);
        }
        for (const auto& term : def.boundary) {
            FE_CHECK_NOT_NULL(term.kernel.get(),
                              "FESystem::computeSetupStoragePlan: null boundary kernel");
            req.boundary_face_topology = true;
            mergeKernelDomainRequirements(term.kernel.get(), req);
        }
        for (const auto& term : def.interior) {
            FE_CHECK_NOT_NULL(term.kernel.get(),
                              "FESystem::computeSetupStoragePlan: null interior-face kernel");
            req.interior_face_topology = true;
            mergeKernelDomainRequirements(term.kernel.get(), req);
        }
        for (const auto& term : def.interface_faces) {
            FE_CHECK_NOT_NULL(term.kernel.get(),
                              "FESystem::computeSetupStoragePlan: null interface-face kernel");
            req.interface_face_topology = true;
            mergeKernelDomainRequirements(term.kernel.get(), req);
        }
        if (!def.global.empty()) {
            req.cell_topology = true;
        }
        plan.merge(req, "operator:" + tag);
    }

    if (operator_backends_) {
        plan.merge(operator_backends_->storageRequirements(), "operator_backends");
    }

    for (const auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::computeSetupStoragePlan: system constraint");
        plan.merge(c->storageRequirements(), "system_constraint");
    }

    if (fe_quantity_registry_) {
        SetupStorageRequirements req;
        for (const auto& def : fe_quantity_registry_->all()) {
            mergeQuantityRequirements(def, req);
        }
        plan.merge(req, "fe_quantities");
    }

    std::size_t unknown_volume_fields = 0;
    std::size_t unknown_fields = 0;
    for (const auto& rec : field_registry_.records()) {
        if (rec.source_kind != FieldSourceKind::Unknown) {
            continue;
        }
        ++unknown_fields;
        if (rec.scope == FieldScope::VolumeCell) {
            ++unknown_volume_fields;
        }
    }
    const bool single_volume_field =
        unknown_fields == 1u && unknown_volume_fields == 1u;
    plan.can_alias_single_field_dof_map = single_volume_field;

    return plan;
}

void FESystem::setup(const SetupOptions& user_opts, const SetupInputs& inputs)
{
    SetupOptions opts = user_opts;

    invalidateSetup();

    FE_THROW_IF(field_registry_.size() == 0u, InvalidStateException,
                "FESystem::setup: no fields registered");

    // ---------------------------------------------------------------------
    // DOFs (multi-field)
    // ---------------------------------------------------------------------
    dof_handler_ = dofs::DofHandler{};
    field_dof_handlers_.clear();
    field_dof_offsets_.clear();

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (!mesh_ && inputs.mesh) {
        mesh_ = inputs.mesh;
        coord_cfg_ = inputs.coord_cfg;
        mesh_access_ = std::make_shared<assembly::MeshAccess>(*mesh_, coord_cfg_);
        if (!search_access_) {
            search_access_ = std::make_shared<MeshSearchAccess>(*mesh_, coord_cfg_);
        }
    }

    if (mesh_) {
        opts.dof_options.my_rank = mesh_->rank();
        opts.dof_options.world_size = mesh_->world_size();
#  if FE_HAS_MPI && defined(MESH_HAS_MPI)
        opts.dof_options.mpi_comm = mesh_->mpi_comm();
#  endif
    }
#endif

    last_setup_options_ = user_opts;
    last_setup_inputs_ = inputs;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (!last_setup_inputs_.mesh && mesh_) {
        last_setup_inputs_.mesh = mesh_;
        last_setup_inputs_.coord_cfg = coord_cfg_;
    }
#endif
    has_last_setup_ = true;

    setup_storage_plan_ = computeSetupStoragePlan();

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (mesh_) {
        opts.dof_options.topology_completion = dofs::TopologyCompletion::RequireComplete;
        const auto& req = setup_storage_plan_.requirements;
        auto& mesh_base = const_cast<svmp::Mesh&>(*mesh_).base();

        MeshFinalizeOptions mesh_storage{};
        mesh_storage.edge_storage = req.edge_topology;
        if (req.interior_face_topology ||
            (mesh_base.dim() == 2 && req.edge_topology)) {
            mesh_storage.codim1_storage = MeshCodim1StorageMode::Full;
        } else if (req.boundary_face_topology) {
            mesh_storage.codim1_storage = MeshCodim1StorageMode::BoundaryOnly;
        } else {
            mesh_storage.codim1_storage = MeshCodim1StorageMode::None;
        }

        const bool needs_faces =
            mesh_storage.codim1_storage != MeshCodim1StorageMode::None;
        const bool boundary_faces_already_planned =
            mesh_storage.codim1_storage == MeshCodim1StorageMode::BoundaryOnly &&
            mesh_base.codim1_storage_mode() == MeshCodim1StorageMode::BoundaryOnly;
        const bool faces_missing =
            needs_faces && mesh_base.n_faces() == 0u && !boundary_faces_already_planned;
        const bool needs_full_faces =
            mesh_storage.codim1_storage == MeshCodim1StorageMode::Full;
        FE_THROW_IF(needs_full_faces &&
                    mesh_base.codim1_storage_mode() == MeshCodim1StorageMode::BoundaryOnly,
                    InvalidStateException,
                    "FESystem::setup: storage plan requires full interior-face topology, "
                    "but the mesh currently holds boundary-only face topology");

        const bool edges_missing =
            mesh_storage.edge_storage && mesh_base.n_edges() == 0u;
        if (faces_missing || edges_missing) {
            if (!faces_missing) {
                mesh_storage.codim1_storage = mesh_base.codim1_storage_mode();
            }
            mesh_base.finalize(mesh_storage);
        }
    }
#endif

#if FE_HAS_MPI
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) {
            int comm_rank = 0;
            int comm_size = 1;
            MPI_Comm_rank(opts.dof_options.mpi_comm, &comm_rank);
            MPI_Comm_size(opts.dof_options.mpi_comm, &comm_size);

            if (opts.dof_options.world_size == 1 &&
                opts.dof_options.my_rank == 0 &&
                comm_size > 1) {
                opts.dof_options.my_rank = comm_rank;
                opts.dof_options.world_size = comm_size;
            }
        }
    }
#endif

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<dofs::MeshTopologyInfo> mesh_access_topology_override;
    if (!mesh_ && !inputs.topology_override.has_value() && mesh_access_) {
        mesh_access_topology_override = meshTopologyFromAccess(*mesh_access_, opts.dof_options.my_rank);
    }
#endif
    const dofs::MeshTopologyInfo* active_topology_override = nullptr;
    if (inputs.topology_override.has_value()) {
        active_topology_override = &*inputs.topology_override;
    }
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    else if (mesh_access_topology_override.has_value()) {
        active_topology_override = &*mesh_access_topology_override;
    }
#endif

    const auto n_fields = field_registry_.size();
    field_dof_handlers_.resize(n_fields);
    field_dof_offsets_.assign(n_fields, 0);
    std::vector<const FieldRecord*> unknown_field_records;
    unknown_field_records.reserve(n_fields);
    for (const auto& rec : field_registry_.records()) {
        if (rec.source_kind == FieldSourceKind::Unknown) {
            unknown_field_records.push_back(&rec);
        }
    }

    auto interface_mesh_for_field = [&](const FieldRecord& rec) -> const svmp::InterfaceMesh* {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        if (rec.scope != FieldScope::InterfaceFace) {
            return nullptr;
        }
        auto iface_it = interface_meshes_.find(rec.interface_marker);
        FE_THROW_IF(iface_it == interface_meshes_.end() || !iface_it->second,
                    InvalidArgumentException,
                    "FESystem::setup: missing InterfaceMesh for interface-scoped field '" +
                        rec.name + "' on interface marker " +
                        std::to_string(rec.interface_marker));
        return iface_it->second.get();
#else
        (void)rec;
        return nullptr;
#endif
    };

    auto distribute_field = [&](const FieldRecord& rec) {
        FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::setup: field.space");

        dofs::DofHandler dh;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        if (mesh_) {
            if (rec.space->space_type() == spaces::SpaceType::Mortar) {
                const auto* mortar_space =
                    dynamic_cast<const spaces::MortarSpace*>(rec.space.get());
                FE_CHECK_NOT_NULL(mortar_space,
                                  "FESystem::setup: mortar field must use MortarSpace");
                FE_THROW_IF(rec.scope != FieldScope::InterfaceFace, InvalidArgumentException,
                            "FESystem::setup: mortar field '" + rec.name +
                                "' must be registered as an interface-scoped field");
                const auto* iface = interface_mesh_for_field(rec);
                FE_CHECK_NOT_NULL(iface, "FESystem::setup: mortar field interface mesh");
                dh.distributeDofs(*mesh_, *iface, *rec.space, opts.dof_options);
            } else if (rec.scope == FieldScope::InterfaceFace) {
                const auto* iface = interface_mesh_for_field(rec);
                FE_CHECK_NOT_NULL(iface, "FESystem::setup: interface field interface mesh");
                dh.distributeDofs(*mesh_, *iface, *rec.space, opts.dof_options);
            } else {
                dh.distributeDofs(*mesh_, *rec.space, opts.dof_options);
            }
            dh.finalize();
            return dh;
        }
#endif

        if (rec.participant_name.has_value() || rec.participant_domain_id.has_value()) {
            FE_THROW_IF(rec.scope == FieldScope::InterfaceFace, InvalidArgumentException,
                        "FESystem::setup: participant-scoped interface fields are not supported");
            const auto* participant = fieldMeshParticipant(rec.id);
            FE_CHECK_NOT_NULL(participant, "FESystem::setup: participant-scoped field participant");
            FE_THROW_IF(mesh_access_ == nullptr, InvalidArgumentException,
                        "FESystem::setup: participant-scoped fields require meshAccess()");

            auto participant_topology =
                participantTopologyFromAccess(*mesh_access_, *participant, opts.dof_options.my_rank);

            dofs::DofHandler local_handler;
            local_handler.distributeDofs(participant_topology, *rec.space, opts.dof_options);
            local_handler.finalize();

            return remapParticipantDofHandler(
                local_handler,
                *participant,
                *mesh_access_,
                opts.dof_options);
        }

        if (active_topology_override != nullptr) {
            FE_THROW_IF(rec.scope == FieldScope::InterfaceFace, InvalidArgumentException,
                        "FESystem::setup: interface-scoped field '" + rec.name +
                            "' requires Mesh and InterfaceMesh input; topology_override is volume-only");
            const auto& topology = *active_topology_override;
            FE_THROW_IF(topology.n_cells <= 0, InvalidArgumentException,
                        "FESystem::setup: topology_override has no cells");

            dh.distributeDofs(topology, *rec.space, opts.dof_options);
            dh.finalize();
            return dh;
        }

        FE_THROW(InvalidArgumentException,
                 "FESystem::setup: need Mesh (SVMP_FE_WITH_MESH) or topology_override for DOF distribution");
    };

    for (const auto& rec : field_registry_.records()) {
        if (rec.participant_name.has_value() || rec.participant_domain_id.has_value()) {
            (void)fieldMeshParticipant(rec.id);
        }
    }

    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        FE_THROW_IF(rec.id < 0 || idx >= field_dof_handlers_.size(), InvalidStateException,
                    "FESystem::setup: field registry contains invalid FieldId");
        field_dof_handlers_[idx] = distribute_field(rec);
    }

    // Assign monolithic offsets in field registration order.
    GlobalIndex total_dofs = 0;
    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        field_dof_offsets_[idx] = total_dofs;
        if (rec.source_kind == FieldSourceKind::Unknown) {
            total_dofs += field_dof_handlers_[idx].getNumDofs();
        }
    }

    if (setup_storage_plan_.can_alias_single_field_dof_map) {
        FE_THROW_IF(unknown_field_records.size() != 1u, InvalidStateException,
                    "FESystem::setup: single-field alias expected exactly one unknown field");
        const auto& rec = *unknown_field_records.front();
        const auto idx = static_cast<std::size_t>(rec.id);
        FE_THROW_IF(idx >= field_dof_handlers_.size(), InvalidStateException,
                    "FESystem::setup: single-field alias references invalid FieldId");
        field_dof_offsets_[idx] = 0;
        dof_handler_ = dofs::DofHandler{};
        dof_handler_.setReadOnlyAlias(field_dof_handlers_[idx]);
        setup_storage_plan_.uses_single_field_alias = true;
        bumpDofLayoutRevision();
    } else {
        // Build a monolithic DofMap + EntityDofMap so Systems can expose a single global DofHandler.
        const auto n_cells = meshAccess().numCells();
        LocalIndex approx_dofs_per_cell = 0;
        for (const auto& rec : field_registry_.records()) {
            if (rec.source_kind != FieldSourceKind::Unknown) {
                continue;
            }
            const auto idx = static_cast<std::size_t>(rec.id);
            approx_dofs_per_cell = static_cast<LocalIndex>(
                approx_dofs_per_cell + field_dof_handlers_[idx].getDofMap().getMaxDofsPerCell());
        }

        dofs::DofMap monolithic_map(n_cells, total_dofs, approx_dofs_per_cell);
        monolithic_map.setNumDofs(total_dofs);
        monolithic_map.setNumLocalDofs(0);

        auto append_field_cell_dofs = [&](const FieldRecord& rec,
                                          GlobalIndex cell,
                                          std::vector<GlobalIndex>& out) {
            const auto idx = static_cast<std::size_t>(rec.id);
            const auto offset = field_dof_offsets_[idx];

            if (rec.scope != FieldScope::InterfaceFace) {
                auto local = field_dof_handlers_[idx].getDofMap().getCellDofs(cell);
                out.reserve(out.size() + local.size());
                for (auto d : local) {
                    out.push_back(d + offset);
                }
                return;
            }

    #if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
            const auto* iface = interface_mesh_for_field(rec);
            FE_CHECK_NOT_NULL(iface, "FESystem::setup: interface field interface mesh");
            for (std::size_t lf = 0; lf < iface->n_faces(); ++lf) {
                const auto local_face = static_cast<svmp::index_t>(lf);
                const auto minus_cell = static_cast<GlobalIndex>(iface->volume_cell_minus(local_face));
                const auto plus_cell = static_cast<GlobalIndex>(iface->volume_cell_plus(local_face));
                if (minus_cell != cell && plus_cell != cell) {
                    continue;
                }
                auto mortar_local =
                    field_dof_handlers_[idx].getDofMap().getCellDofs(static_cast<GlobalIndex>(lf));
                out.reserve(out.size() + mortar_local.size());
                for (auto d : mortar_local) {
                    out.push_back(d + offset);
                }
            }
            return;
    #else
            FE_THROW(InvalidStateException,
                     "FESystem::setup: interface-scoped fields require Mesh support");
    #endif
        };

        std::vector<GlobalIndex> cell_dofs;
        for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
            cell_dofs.clear();
            for (const auto& rec : field_registry_.records()) {
                if (rec.source_kind != FieldSourceKind::Unknown) {
                    continue;
                }
                append_field_cell_dofs(rec, cell, cell_dofs);
            }
            if (!cell_dofs.empty()) {
                std::vector<GlobalIndex> unique_dofs;
                unique_dofs.reserve(cell_dofs.size());
                std::unordered_set<GlobalIndex> seen;
                seen.reserve(cell_dofs.size());
                for (auto dof : cell_dofs) {
                    if (seen.insert(dof).second) {
                        unique_dofs.push_back(dof);
                    }
                }
                monolithic_map.setCellDofs(cell, unique_dofs);
            } else {
                monolithic_map.setCellDofs(cell, cell_dofs);
            }
        }

        // Merge entity-DOF maps if available (Mesh-driven workflows).
        std::unique_ptr<dofs::EntityDofMap> merged_entity_map;
        {
            bool all_have_maps = true;
            GlobalIndex max_vertices = 0;
            GlobalIndex max_edges = 0;
            GlobalIndex max_faces = 0;
            GlobalIndex max_cells = 0;
            for (const auto& rec : field_registry_.records()) {
                if (rec.source_kind != FieldSourceKind::Unknown) {
                    continue;
                }
                if (rec.scope == FieldScope::InterfaceFace &&
                    rec.space->space_type() != spaces::SpaceType::Mortar) {
                    continue;
                }
                const auto idx = static_cast<std::size_t>(rec.id);
                auto* map = field_dof_handlers_[idx].getEntityDofMap();
                if (map == nullptr) {
                    all_have_maps = false;
                    break;
                }
                max_vertices = std::max(max_vertices, map->numVertices());
                max_edges = std::max(max_edges, map->numEdges());
                max_faces = std::max(max_faces, map->numFaces());
                max_cells = std::max(max_cells, map->numCells());
            }

            if (all_have_maps) {
                merged_entity_map = std::make_unique<dofs::EntityDofMap>();
                merged_entity_map->reserve(max_vertices, max_edges, max_faces, max_cells);

                std::vector<GlobalIndex> entity_dofs;

                for (GlobalIndex v = 0; v < max_vertices; ++v) {
                    entity_dofs.clear();
                    for (const auto& rec : field_registry_.records()) {
                        if (rec.source_kind != FieldSourceKind::Unknown) {
                            continue;
                        }
                        if (rec.scope == FieldScope::InterfaceFace &&
                            rec.space->space_type() != spaces::SpaceType::Mortar) {
                            continue;
                        }
                        const auto idx = static_cast<std::size_t>(rec.id);
                        const auto offset = field_dof_offsets_[idx];
                        const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                        FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                        if (v >= emap->numVertices()) {
                            continue;
                        }
                        auto vdofs = emap->getVertexDofs(v);
                        entity_dofs.reserve(entity_dofs.size() + vdofs.size());
                        for (auto d : vdofs) {
                            entity_dofs.push_back(d + offset);
                        }
                    }
                    merged_entity_map->setVertexDofs(v, entity_dofs);
                }

                for (GlobalIndex e = 0; e < max_edges; ++e) {
                    entity_dofs.clear();
                    for (const auto& rec : field_registry_.records()) {
                        if (rec.source_kind != FieldSourceKind::Unknown) {
                            continue;
                        }
                        if (rec.scope == FieldScope::InterfaceFace &&
                            rec.space->space_type() != spaces::SpaceType::Mortar) {
                            continue;
                        }
                        const auto idx = static_cast<std::size_t>(rec.id);
                        const auto offset = field_dof_offsets_[idx];
                        const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                        FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                        if (e >= emap->numEdges()) {
                            continue;
                        }
                        auto edofs = emap->getEdgeDofs(e);
                        entity_dofs.reserve(entity_dofs.size() + edofs.size());
                        for (auto d : edofs) {
                            entity_dofs.push_back(d + offset);
                        }
                    }
                    merged_entity_map->setEdgeDofs(e, entity_dofs);
                }

                for (GlobalIndex f = 0; f < max_faces; ++f) {
                    entity_dofs.clear();
                    for (const auto& rec : field_registry_.records()) {
                        if (rec.source_kind != FieldSourceKind::Unknown) {
                            continue;
                        }
                        if (rec.scope == FieldScope::InterfaceFace &&
                            rec.space->space_type() != spaces::SpaceType::Mortar) {
                            continue;
                        }
                        const auto idx = static_cast<std::size_t>(rec.id);
                        const auto offset = field_dof_offsets_[idx];
                        const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                        FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                        if (f >= emap->numFaces()) {
                            continue;
                        }
                        auto fdofs = emap->getFaceDofs(f);
                        entity_dofs.reserve(entity_dofs.size() + fdofs.size());
                        for (auto d : fdofs) {
                            entity_dofs.push_back(d + offset);
                        }
                    }
                    merged_entity_map->setFaceDofs(f, entity_dofs);
                }

                for (GlobalIndex c = 0; c < max_cells; ++c) {
                    entity_dofs.clear();
                    for (const auto& rec : field_registry_.records()) {
                        if (rec.source_kind != FieldSourceKind::Unknown) {
                            continue;
                        }
                        if (rec.scope == FieldScope::InterfaceFace &&
                            rec.space->space_type() != spaces::SpaceType::Mortar) {
                            continue;
                        }
                        const auto idx = static_cast<std::size_t>(rec.id);
                        const auto offset = field_dof_offsets_[idx];
                        const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                        FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                        if (c >= emap->numCells()) {
                            continue;
                        }
                        auto cdofs = emap->getCellInteriorDofs(c);
                        entity_dofs.reserve(entity_dofs.size() + cdofs.size());
                        for (auto d : cdofs) {
                            entity_dofs.push_back(d + offset);
                        }
                    }
                    merged_entity_map->setCellInteriorDofs(c, entity_dofs);
                }

                merged_entity_map->buildReverseMapping();
                merged_entity_map->finalize();
            }
        }

        dofs::DofPartition part;
        {
            std::vector<dofs::IndexInterval> owned_intervals;
            std::vector<GlobalIndex> owned_explicit;
            std::vector<GlobalIndex> ghost_explicit;

            for (const auto& rec : field_registry_.records()) {
                if (rec.source_kind != FieldSourceKind::Unknown) {
                    continue;
                }
                const auto idx = static_cast<std::size_t>(rec.id);
                const auto offset = field_dof_offsets_[idx];
                const auto& fpart = field_dof_handlers_[idx].getPartition();

                if (auto range = fpart.locallyOwned().contiguousRange()) {
                    owned_intervals.push_back(dofs::IndexInterval{range->begin + offset, range->end + offset});
                } else {
                    auto vec = fpart.locallyOwned().toVector();
                    owned_explicit.reserve(owned_explicit.size() + vec.size());
                    for (auto d : vec) {
                        owned_explicit.push_back(d + offset);
                    }
                }

                auto ghosts = fpart.ghost().toVector();
                ghost_explicit.reserve(ghost_explicit.size() + ghosts.size());
                for (auto d : ghosts) {
                    ghost_explicit.push_back(d + offset);
                }
            }

            dofs::IndexSet owned;
            if (!owned_intervals.empty()) {
                owned = dofs::IndexSet(std::move(owned_intervals));
            }
            if (!owned_explicit.empty()) {
                owned = owned.unionWith(dofs::IndexSet(std::move(owned_explicit)));
            }

            dofs::IndexSet ghost;
            if (!ghost_explicit.empty()) {
                ghost = dofs::IndexSet(std::move(ghost_explicit));
            }

            part = dofs::DofPartition(std::move(owned), std::move(ghost));
            part.setGlobalSize(total_dofs);
        }

        monolithic_map.setNumLocalDofs(part.localOwnedSize());

        // Systems assembly relies on DOF ownership queries for both OwnedRowsOnly and ReverseScatter
        // ghost policies. The monolithic DofMap is built by offsetting per-field DOF indices, so we
        // re-expose ownership by forwarding to each field's DofMap ownership function.
        monolithic_map.setMyRank(opts.dof_options.my_rank);
        {
            std::vector<GlobalIndex> offsets;
            std::vector<GlobalIndex> sizes;
            std::vector<const dofs::DofMap*> maps;
            offsets.reserve(field_registry_.size());
            sizes.reserve(field_registry_.size());
            maps.reserve(field_registry_.size());

            for (const auto& rec : field_registry_.records()) {
                if (rec.source_kind != FieldSourceKind::Unknown) {
                    continue;
                }
                const auto idx = static_cast<std::size_t>(rec.id);
                offsets.push_back(field_dof_offsets_[idx]);
                sizes.push_back(field_dof_handlers_[idx].getNumDofs());
                maps.push_back(&field_dof_handlers_[idx].getDofMap());
            }

            monolithic_map.setDofOwnership(
                [offsets = std::move(offsets),
                 sizes = std::move(sizes),
                 maps = std::move(maps)](GlobalIndex global_dof) -> int {
                    if (global_dof < 0 || offsets.empty()) {
                        return 0;
                    }

                    const auto it = std::upper_bound(offsets.begin(), offsets.end(), global_dof);
                    const std::size_t block =
                        (it == offsets.begin()) ? 0u : static_cast<std::size_t>(std::distance(offsets.begin(), it) - 1);
                    if (block >= offsets.size() || block >= sizes.size() || block >= maps.size()) {
                        return 0;
                    }

                    const GlobalIndex local = global_dof - offsets[block];
                    if (local < 0 || local >= sizes[block]) {
                        return 0;
                    }

                    const auto* map = maps[block];
                    if (!map) {
                        return 0;
                    }
                    return map->getDofOwner(local);
                });
        }

        dof_handler_ = dofs::DofHandler{};
        dof_handler_.setDofMap(std::move(monolithic_map));
        dof_handler_.setPartition(std::move(part));
        dof_handler_.setRankInfo(opts.dof_options.my_rank, opts.dof_options.world_size);
        if (merged_entity_map) {
            dof_handler_.setEntityDofMap(std::move(merged_entity_map));
        }
        // Preserve per-cell orientation metadata (H(curl)/H(div)) by copying it from any field handler
        // that computed it during DOF distribution.
        for (const auto& rec : field_registry_.records()) {
            if (rec.source_kind != FieldSourceKind::Unknown) {
                continue;
            }
            const auto idx = static_cast<std::size_t>(rec.id);
            if (idx < field_dof_handlers_.size() && field_dof_handlers_[idx].hasCellOrientations()) {
                dof_handler_.copyCellOrientationsFrom(field_dof_handlers_[idx]);
                break;
            }
        }
        dof_handler_.finalize();
        bumpDofLayoutRevision();
    }

    // ---------------------------------------------------------------------
    // Field/block metadata (monolithic across fields)
    // ---------------------------------------------------------------------
    field_map_ = dofs::FieldDofMap{};
    field_map_.setLayout(fieldLayoutForSystem(unknown_field_records.size(), opts.dof_options));

    for (const auto& rec : field_registry_.records()) {
        if (rec.source_kind != FieldSourceKind::Unknown) {
            continue;
        }
        const auto idx = static_cast<std::size_t>(rec.id);
        const auto n_components = rec.space->value_dimension();
        const auto n_dofs_field = field_dof_handlers_[idx].getNumDofs();

        if (n_components <= 1) {
            field_map_.addScalarField(rec.name, n_dofs_field);
        } else {
            const bool is_vector_basis =
                (rec.space->continuity() == Continuity::H_curl || rec.space->continuity() == Continuity::H_div);
            if (is_vector_basis) {
                field_map_.addVectorBasisField(rec.name,
                                               static_cast<LocalIndex>(n_components),
                                               n_dofs_field);
            } else {
                FE_THROW_IF(n_dofs_field % n_components != 0, InvalidStateException,
                            "FESystem::setup: vector-valued field has non-divisible DOF count");
                field_map_.addVectorField(rec.name, static_cast<LocalIndex>(n_components),
                                          n_dofs_field / n_components);
            }
        }
    }
    field_map_.finalize();
    block_map_.reset();
    if (unknown_field_records.size() > 1u) {
        auto blocks = std::make_unique<dofs::BlockDofMap>();
        for (const auto& rec : field_registry_.records()) {
            if (rec.source_kind != FieldSourceKind::Unknown) {
                continue;
            }
            const auto idx = static_cast<std::size_t>(rec.id);
            blocks->addBlock(rec.name, field_dof_handlers_[idx].getNumDofs());
        }
        blocks->finalize();
        block_map_ = std::move(blocks);
    }
    bumpBlockLayoutRevision();

    // ---------------------------------------------------------------------
    // Constraints
    // ---------------------------------------------------------------------
    affine_constraints_.clear();
    for (auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::setup: system constraint");
        c->apply(*this, affine_constraints_);
    }
    for (const auto& c : constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::setup: constraint");
        c->apply(affine_constraints_);
    }

#if FE_HAS_MPI
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized &&
            !globalConstraintDefinitionsConsistentAcrossRanks(constraint_defs_,
                                                              dof_handler_.mpiComm())) {
            FE_THROW(FEException,
                     "FESystem::setup: explicit GlobalConstraint definitions differ across "
                     "MPI ranks. Global algebraic constraints must be defined identically on "
                     "every rank.");
        }
    }
#endif

    // -----------------------------------------------------------------
    // Gauge / nullspace auto-detection and enforcement
    // -----------------------------------------------------------------
    // Collect contributions from non-Forms kernels via analysisContributions().
    {
        const auto op_tags_gauge = operator_registry_.list();
        for (const auto& tag : op_tags_gauge) {
            const auto& def = operator_registry_.get(tag);

            // Helper: collect contributions from a kernel.
            auto collectContributions = [this](const auto& kernel) {
                auto contribs = kernel->analysisContributions();
                for (auto& c : contribs) {
                    addContribution(std::move(c));
                }
            };

            // Cell kernels
            for (const auto& term : def.cells) {
                if (!term.kernel) continue;
                collectContributions(term.kernel);
            }

            // Boundary kernels
            for (const auto& term : def.boundary) {
                if (!term.kernel) continue;
                collectContributions(term.kernel);
            }

            // Interior face kernels
            for (const auto& term : def.interior) {
                if (!term.kernel) continue;
                collectContributions(term.kernel);
            }

            // Interface face kernels
            for (const auto& term : def.interface_faces) {
                if (!term.kernel) continue;
                collectContributions(term.kernel);
            }

            // Global kernels
            for (const auto& gk : def.global) {
                if (!gk) continue;
                collectContributions(gk);
            }
        }
    }

    const auto add_gauge_candidate =
        [this](FieldId field,
               int component,
               analysis::NullspaceFamily family,
               analysis::AnalysisConfidence confidence,
               gauge::CandidateSource source,
               const std::string& reason) {
            if (field == INVALID_FIELD_ID) {
                return;
            }

            gauge::GaugeCandidate c;
            c.field = field;
            c.component = component;
            c.source = source;
            c.reason = reason;

            switch (confidence) {
                case analysis::AnalysisConfidence::High:   c.confidence = gauge::Confidence::High; break;
                case analysis::AnalysisConfidence::Medium: c.confidence = gauge::Confidence::Medium; break;
                case analysis::AnalysisConfidence::Low:    c.confidence = gauge::Confidence::Low; break;
            }
            switch (family) {
                case analysis::NullspaceFamily::ScalarConstant:
                    c.family = gauge::NullspaceModeFamily::ScalarConstant; break;
                case analysis::NullspaceFamily::ComponentwiseConstant:
                    c.family = gauge::NullspaceModeFamily::ComponentwiseConstant; break;
                case analysis::NullspaceFamily::KernelOfSymGrad:
                    c.family = gauge::NullspaceModeFamily::KernelOfSymGrad; break;
                case analysis::NullspaceFamily::UserDefined:
                    c.family = gauge::NullspaceModeFamily::ScalarConstant; break;
            }
            gaugeRegistry().addCandidate(std::move(c));
        };

    // Convert NullspaceHints from contributions into GaugeRegistry candidates.
    // This is the primary path for gauge candidate population via
    // FormContributionLowerer NullspaceHint emission.
    for (const auto& contrib : contributions_) {
        for (const auto& hint : contrib.nullspace_hints) {
            add_gauge_candidate(hint.field,
                                hint.component,
                                hint.family,
                                hint.confidence,
                                gauge::CandidateSource::ExplicitDeclaration,
                                hint.reason);
        }
    }

    // MixedOperatorAnalyzer and other analysis passes can also emit field-level
    // nullspace claims directly into the analysis report. These must feed the
    // GaugeRegistry too; otherwise mixed saddle-point pressure gauges are
    // inferred structurally but never enforced during setup.
    for (const auto& claim : analysisReport().claims) {
        if (claim.kind != analysis::PropertyKind::Nullspace ||
            !claim.nullspace_family.has_value()) {
            continue;
        }
        add_gauge_candidate(claim.field,
                            claim.component,
                            *claim.nullspace_family,
                            claim.confidence,
                            gauge::CandidateSource::FormsInference,
                            !claim.description.empty() ? claim.description : claim.claim_origin);
    }

    // If the GaugeRegistry has candidates (populated by FormContributionLowerer
    // or by kernel contributions), resolve them
    // against anchoring evidence from the constraints already applied above.
    //
    // StrongDirichlet BCs that constrained any DOF of a field are treated
    // as anchoring evidence for the constant-mode nullspace of that field.
    if (gauge_registry_ && !gauge_registry_->candidates().empty()) {
        // Provide a DOF-lookup callback for the resolver
        auto get_field_dofs = [this](FieldId fid, int component) -> std::vector<GlobalIndex> {
            const auto idx = static_cast<std::size_t>(fid);
            if (idx >= field_dof_handlers_.size()) return {};

            // When component >= 0, return only DOFs for that component
            if (component >= 0 && idx < field_map_.numFields()) {
                const auto n_comp = field_map_.numComponents(idx);
                if (n_comp > 1 && static_cast<LocalIndex>(component) < n_comp) {
                    return field_map_.getComponentDofs(idx, static_cast<LocalIndex>(component)).toVector();
                }
            }

            const auto offset = field_dof_offsets_[idx];
            const auto n_dofs = field_dof_handlers_[idx].getNumDofs();

            std::vector<GlobalIndex> dofs;
            dofs.reserve(static_cast<std::size_t>(n_dofs));
            for (GlobalIndex d = offset; d < offset + n_dofs; ++d) {
                dofs.push_back(d);
            }
            return dofs;
        };

        // Build RegionProvider: DOF → connected-component ID.
        // Must be built BEFORE the Dirichlet scan so that Dirichlet evidence
        // can be tagged with the correct region.
        //
        // Limitation: this assumes vertex-based DOF spaces (H1/P1) when mapping
        // a DOF back to a region.  Connectivity itself is mesh-based so mixed or
        // vector-valued field ordering does not distort region detection.
        gauge::GaugeRegistry::RegionProvider region_provider;
        std::shared_ptr<std::map<int, std::vector<int>>> marker_regions;
        bool allow_local_region_scoping = true;
#if FE_HAS_MPI
        {
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized) {
                int world_size = 1;
                int rank = 0;
                MPI_Comm_size(dof_handler_.mpiComm(), &world_size);
                MPI_Comm_rank(dof_handler_.mpiComm(), &rank);
                if (world_size > 1) {
                    allow_local_region_scoping = false;
                    if (rank == 0) {
                        std::fprintf(stderr,
                            "[GaugeRegistry] MPI run: using global gauge scoping; "
                            "per-region gauge scoping requires globally consistent "
                            "connected-component labels.\n");
                    }
                }
            }
        }
#endif
        {
            if (mesh_access_ && allow_local_region_scoping) {
                const auto& mesh_ref = meshAccess();
                auto topo = analysis::TopologyAnalysisContext::build(mesh_ref);
                if (topo.numRegions() > 1) {
                    auto vertex_region = std::make_shared<std::vector<int>>();
                    std::vector<GlobalIndex> nodes;
                    mesh_ref.forEachCell([&](GlobalIndex cell_id) {
                        const int region = topo.regionForCell(cell_id);
                        if (region < 0) {
                            return;
                        }
                        nodes.clear();
                        mesh_ref.getCellNodes(cell_id, nodes);
                        for (const auto n : nodes) {
                            if (n < 0) {
                                continue;
                            }
                            const auto idx = static_cast<std::size_t>(n);
                            if (idx >= vertex_region->size()) {
                                vertex_region->resize(idx + 1, -1);
                            }
                            auto& slot = (*vertex_region)[idx];
                            if (slot < 0) {
                                slot = region;
                            }
                        }
                    });

                    const auto* emap = dof_handler_.getEntityDofMap();
                    if (emap && !vertex_region->empty()) {
                        region_provider = [vertex_region, emap](GlobalIndex dof) -> int {
                            auto ent = emap->getDofEntity(dof);
                            if (ent && ent->kind == dofs::EntityKind::Vertex) {
                                const auto vid = static_cast<std::size_t>(ent->id);
                                if (vid < vertex_region->size()) {
                                    const int region = (*vertex_region)[vid];
                                    if (region >= 0) {
                                        return region;
                                    }
                                }
                            }
                            return 0;
                        };
                        marker_regions = std::make_shared<std::map<int, std::vector<int>>>(
                            topo.boundary_mapping.marker_to_regions);
                        std::fprintf(stderr,
                            "[GaugeRegistry] Mesh has %lld connected components — "
                            "enabling per-region gauge scoping\n",
                            static_cast<long long>(topo.numRegions()));
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // Retag pre-region BC evidence: convert global (region=-1) anchoring
        // evidence from the BC manager into per-region evidence using the
        // boundary_marker → region mapping.  This prevents a Robin/custom BC
        // on one disconnected region from anchoring all regions.
        //
        // Evidence without a boundary_marker (e.g., explicit physics-module
        // anchors like the NS natural-pressure anchor) stays global.  The
        // resolver blocks unresolved global Anchored evidence from matching
        // per-region candidates, so those anchors require the physics module
        // to be updated to emit per-marker evidence for full region support.
        // -----------------------------------------------------------------
        if (region_provider && marker_regions) {
            gauge_registry_->retagEvidenceRegions(
                [marker_regions](int marker) -> std::vector<int> {
                    auto it = marker_regions->find(marker);
                    if (it == marker_regions->end()) {
                        return {};
                    }
                    return it->second;
                });
        }

        // -----------------------------------------------------------------
        // Dirichlet scan: detect which fields/components/regions have
        // Dirichlet-constrained DOFs and record per-region anchoring evidence.
        //
        // When region_provider is active, each constrained DOF is tagged with
        // its region so that a Dirichlet on region 0 does NOT anchor region 1.
        // -----------------------------------------------------------------
        {
            std::vector<FieldId> seen_fields;
            for (const auto& candidate : gauge_registry_->candidates()) {
                bool already = false;
                for (auto f : seen_fields) {
                    if (f == candidate.field) { already = true; break; }
                }
                if (!already) seen_fields.push_back(candidate.field);
            }

            for (const auto fid : seen_fields) {
                const auto idx = static_cast<std::size_t>(fid);
                if (idx >= field_dof_handlers_.size()) continue;

                // Helper: given a set of DOFs, add per-region Dirichlet evidence
                // when region_provider is active, otherwise add global evidence.
                auto addDirichletEvidence = [&](FieldId field, int comp,
                                                const std::vector<GlobalIndex>& dofs) {
                    if (region_provider) {
                        // Group constrained DOFs by region
                        std::unordered_map<int, bool> region_has_dirichlet;
                        for (const auto d : dofs) {
                            if (affine_constraints_.isConstrained(d)) {
                                region_has_dirichlet[region_provider(d)] = true;
                            }
                        }
                        for (const auto& [region, _] : region_has_dirichlet) {
                            gauge_registry_->addAnchoring(gauge::AnchoringEvidence{
                                field, comp, region,
                                std::nullopt,
                                gauge::AnchoringVerdict::Anchored,
                                "StrongDirichlet constraint (region " +
                                    std::to_string(region) + ")"});
                        }
                    } else {
                        bool has_dirichlet = false;
                        for (const auto d : dofs) {
                            if (affine_constraints_.isConstrained(d)) {
                                has_dirichlet = true;
                                break;
                            }
                        }
                        if (has_dirichlet) {
                            gauge_registry_->addAnchoring(gauge::AnchoringEvidence{
                                field, comp, -1,
                                std::nullopt,
                                gauge::AnchoringVerdict::Anchored,
                                "StrongDirichlet constraint on DOFs"});
                        }
                    }
                };

                // Multi-component field: check each component separately
                if (idx < field_map_.numFields()) {
                    const auto n_comp = field_map_.numComponents(idx);
                    if (n_comp > 1) {
                        for (LocalIndex comp = 0; comp < n_comp; ++comp) {
                            auto comp_dofs = field_map_.getComponentDofs(idx, comp).toVector();
                            addDirichletEvidence(fid, static_cast<int>(comp), comp_dofs);
                        }
                        continue;
                    }
                }

                // Scalar field: check all DOFs
                const auto offset = field_dof_offsets_[idx];
                const auto n_dofs = field_dof_handlers_[idx].getNumDofs();
                std::vector<GlobalIndex> all_dofs;
                all_dofs.reserve(static_cast<std::size_t>(n_dofs));
                for (GlobalIndex d = offset; d < offset + n_dofs; ++d) {
                    all_dofs.push_back(d);
                }
                addDirichletEvidence(fid, -1, all_dofs);
            }
        }

        // -----------------------------------------------------------------
        // IBP coupling analysis: detect cross-field nullspace anchoring
        // from integration-by-parts boundary terms.
        //
        // Mathematical argument:
        // If field P has a nullspace candidate and appears undifferentiated
        // (ConstraintPair) in another field V's equation, then IBP produces
        // a boundary integral containing P undifferentiated on faces where
        // V has natural (non-Dirichlet) BCs.  For ScalarConstant nullspace,
        // the constant mode contributes non-trivially to this integral,
        // anchoring the nullspace.
        //
        // Example: incompressible NS has VP block ∫ p div(v) dΩ.  By the
        // divergence theorem, the boundary term ∫ p (v·n) dΓ exists on
        // faces where v is unrestricted.  A constant pressure p=c gives
        // c·∫ v·n dΓ ≠ 0, so the constant mode is anchored.
        //
        // This is physics-agnostic: it works for any saddle-point system
        // where a Lagrange multiplier field enters algebraically into the
        // coupled equation (Stokes, mixed Darcy, electromagnetics, etc.).
        //
        // Limitation: periodic BCs on field V constrain slave DOFs but
        // leave master DOFs unconstrained.  The IBP boundary terms on
        // periodic face pairs cancel (opposite normals), so they do NOT
        // anchor P.  Currently periodic markers are not excluded; a future
        // refinement could detect periodic constraints and skip those markers.
        // -----------------------------------------------------------------
        if (mesh_access_) {
            // Step 1: Collect nullspace candidate fields and their families.
            std::unordered_map<FieldId, gauge::NullspaceModeFamily> candidate_fields;
            for (const auto& c : gauge_registry_->candidates()) {
                candidate_fields.emplace(c.field, c.family);
            }

            if (!candidate_fields.empty()) {
                // Step 2: Find ConstraintPair couplings: trial field P (nullspace
                // candidate) enters algebraically into test field V's equation.
                // (col_var = trial = P, row_var = test = V)
                struct CouplingPair {
                    FieldId nullspace_field;   // P
                    FieldId coupled_field;     // V
                    gauge::NullspaceModeFamily family;
                };
                std::vector<CouplingPair> couplings;

                std::size_t total_pairings = 0;
                for (const auto& contrib : contributions_) {
                    for (const auto& pairing : contrib.pairings) {
                        // The trial field must appear undifferentiated in the
                        // coupling block.  This is true for ConstraintPair (always)
                        // and for FormalAdjointPair when the trial also has an
                        // undifferentiated path (e.g., NS-VMS VP block with both
                        // `p div(v)` and `τ_m grad(v) · grad(p)`).
                        if (!pairing.trial_has_undifferentiated) continue;

                        // col_var is the trial (column) variable — the field that enters
                        // algebraically (undifferentiated) into the test equation.
                        if (pairing.col_var.kind != analysis::VariableKind::FieldComponent) continue;
                        if (pairing.row_var.kind != analysis::VariableKind::FieldComponent) continue;

                        const FieldId trial_fid = pairing.col_var.field_id;
                        const FieldId test_fid  = pairing.row_var.field_id;
                        if (trial_fid == test_fid) continue;  // Skip diagonal

                        auto it = candidate_fields.find(trial_fid);
                        if (it == candidate_fields.end()) continue;

                        // Avoid duplicates
                        bool already = false;
                        for (const auto& cp : couplings) {
                            if (cp.nullspace_field == trial_fid && cp.coupled_field == test_fid) {
                                already = true;
                                break;
                            }
                        }
                        if (!already) {
                            couplings.push_back({trial_fid, test_fid, it->second});
                        }
                    }
                }

                // Step 3: For each coupling, find markers where the coupled field V
                // has natural (non-Dirichlet) BCs, and emit anchoring evidence for P.
                if (!couplings.empty()) {
                    // Enumerate boundary markers globally across MPI ranks.
                    // Whether a coupled field has a natural boundary on marker m
                    // is a global property of the problem. Resolving this per-rank
                    // leads to inconsistent pressure-gauge decisions when only a
                    // subset of ranks owns the outlet/inlet faces.
                    std::vector<int> all_markers =
                        boundaryMarkersFromMeshAccess(meshAccess());
#if FE_HAS_MPI
                    {
                        int mpi_initialized = 0;
                        MPI_Initialized(&mpi_initialized);
                        if (mpi_initialized) {
                            all_markers =
                                gatherGlobalBoundaryMarkers(all_markers, dof_handler_.mpiComm());
                        }
                    }
#endif

                    for (const auto& cp : couplings) {
                        const auto v_idx = static_cast<std::size_t>(cp.coupled_field);
                        if (v_idx >= field_dof_handlers_.size()) continue;

                        const GlobalIndex v_offset = field_dof_offsets_[v_idx];
                        const GlobalIndex v_end    = v_offset + field_dof_handlers_[v_idx].getNumDofs();

                        for (const int marker : all_markers) {
                            // Get all DOFs on this marker (combined DOF numbering).
                            std::vector<GlobalIndex> marker_dofs;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
                            if (mesh_) {
                                marker_dofs =
                                    constraints::boundaryDofsByMarker(*mesh_, dof_handler_, marker);
                            } else
#endif
                            {
                                marker_dofs = boundaryDofsByMarkerFromMeshAccess(
                                    meshAccess(), dof_handler_, marker);
                            }

                            // Filter to DOFs belonging to field V.
                            bool has_unconstrained_v_dof = false;
                            bool has_any_v_dof = false;
                            for (const auto d : marker_dofs) {
                                if (d >= v_offset && d < v_end) {
                                    has_any_v_dof = true;
                                    if (!affine_constraints_.isConstrained(d)) {
                                        has_unconstrained_v_dof = true;
                                        break;
                                    }
                                }
                            }

#if FE_HAS_MPI
                            {
                                int mpi_initialized = 0;
                                MPI_Initialized(&mpi_initialized);
                                if (mpi_initialized) {
                                    const int local_flags[2] = {
                                        has_any_v_dof ? 1 : 0,
                                        has_unconstrained_v_dof ? 1 : 0};
                                    int global_flags[2] = {0, 0};
                                    MPI_Allreduce(local_flags,
                                                  global_flags,
                                                  2,
                                                  MPI_INT,
                                                  MPI_MAX,
                                                  dof_handler_.mpiComm());
                                    has_any_v_dof = (global_flags[0] != 0);
                                    has_unconstrained_v_dof = (global_flags[1] != 0);
                                }
                            }
#endif

                            if (has_any_v_dof && has_unconstrained_v_dof) {
                                gauge_registry_->addAnchoring(gauge::AnchoringEvidence{
                                    cp.nullspace_field,
                                    -1,  // all components
                                    -1,  // global (retagEvidenceRegions handles per-region)
                                    cp.family,
                                    gauge::AnchoringVerdict::Anchored,
                                    "IBP coupling: field " + std::to_string(cp.nullspace_field) +
                                        " enters algebraically in field " +
                                        std::to_string(cp.coupled_field) +
                                        "'s equation; natural BC on marker " +
                                        std::to_string(marker) +
                                        " produces anchoring boundary integral",
                                    marker});
                            }
                        }
                    }
                }
            }
        }
        // Retag the newly added IBP evidence for disconnected-region support.
        // The first retagEvidenceRegions() ran before the IBP pass; this second
        // call processes only the new global evidence (existing per-region evidence
        // from the first call is left untouched since its region is already >= 0).
        if (region_provider && marker_regions) {
            gauge_registry_->retagEvidenceRegions(
                [marker_regions](int marker) -> std::vector<int> {
                    auto it = marker_regions->find(marker);
                    if (it == marker_regions->end()) {
                        return {};
                    }
                    return it->second;
                });
        }

        // Build CoordinateProvider: DOF → physical coordinates.
        // Maps a DOF to its owning vertex via EntityDofMap, then to coordinates.
        gauge::GaugeRegistry::CoordinateProvider coord_provider;
        {
            const auto* emap = dof_handler_.getEntityDofMap();
            if (emap && mesh_access_) {
                coord_provider = [this, emap](FieldId /*fid*/, GlobalIndex dof)
                    -> std::array<double, 3> {
                    auto ent = emap->getDofEntity(dof);
                    if (ent && ent->kind == dofs::EntityKind::Vertex) {
                        auto p = meshAccess().getNodeCoordinates(ent->id);
                        return {static_cast<double>(p[0]),
                                static_cast<double>(p[1]),
                                static_cast<double>(p[2])};
                    }
                    return {0.0, 0.0, 0.0};
                };
            }
        }

        // Build MassWeightProvider: lumped mass weights w_i = ∫ φ_i dΩ.
        // For P1 Lagrange on simplices, w_i = Σ_{cells touching i} |V_cell| / n_verts.
        // This gives FE-correct mean-zero constraints on non-uniform meshes.
        //
        // Limitation: only Tri3 (3-node) and Tet4 (4-node) cells are supported.
        // Meshes containing Quad4 or higher-order cells produce zero-weight DOFs;
        // the provider returns empty for such fields, causing automatic fallback
        // to uniform weights.  A general solution would use quadrature-based
        // mass assembly (e.g., FESystem::assembleMass()).
        gauge::GaugeRegistry::MassWeightProvider mass_weight_provider;
        if (mesh_access_) {
            const auto& mesh = meshAccess();
            const auto n_cells = mesh.numCells();

            // Pre-compute per-vertex lumped mass in a flat array indexed by vertex ID.
            // We accumulate |V_cell|/n_verts for each vertex.
            std::vector<GlobalIndex> cell_nodes;
            std::vector<std::array<Real, 3>> cell_coords;
            const auto total = dof_handler_.getNumDofs();
            auto vertex_mass = std::make_shared<std::vector<double>>(
                static_cast<std::size_t>(total), 0.0);

            const auto* emap = dof_handler_.getEntityDofMap();
            const int mesh_dim = mesh.dimension();
            bool has_unsupported_cells = false;
            if (emap && n_cells > 0) {
                for (GlobalIndex c = 0; c < n_cells; ++c) {
                    mesh.getCellNodes(c, cell_nodes);
                    mesh.getCellCoordinates(c, cell_coords);
                    const auto nv = cell_nodes.size();
                    if (nv < 2 || cell_coords.size() != nv) continue;

                    // Compute simplex volume.  Only Tri3 and Tet4 are supported.
                    // Quad4 (nv==4, dim==2) and higher-order cells are skipped.
                    double vol = 0.0;
                    if (nv == 3 && mesh_dim == 2) {
                        // 2D triangle: 0.5 * |cross(v1-v0, v2-v0)|
                        const double ax = cell_coords[1][0] - cell_coords[0][0];
                        const double ay = cell_coords[1][1] - cell_coords[0][1];
                        const double bx = cell_coords[2][0] - cell_coords[0][0];
                        const double by = cell_coords[2][1] - cell_coords[0][1];
                        vol = 0.5 * std::abs(ax * by - ay * bx);
                    } else if (nv == 4 && mesh_dim == 3) {
                        // 3D tetrahedron: (1/6) * |det[v1-v0, v2-v0, v3-v0]|
                        const double ax = cell_coords[1][0] - cell_coords[0][0];
                        const double ay = cell_coords[1][1] - cell_coords[0][1];
                        const double az = cell_coords[1][2] - cell_coords[0][2];
                        const double bx = cell_coords[2][0] - cell_coords[0][0];
                        const double by = cell_coords[2][1] - cell_coords[0][1];
                        const double bz = cell_coords[2][2] - cell_coords[0][2];
                        const double cx = cell_coords[3][0] - cell_coords[0][0];
                        const double cy = cell_coords[3][1] - cell_coords[0][1];
                        const double cz = cell_coords[3][2] - cell_coords[0][2];
                        vol = std::abs(ax*(by*cz-bz*cy) - ay*(bx*cz-bz*cx) + az*(bx*cy-by*cx)) / 6.0;
                    } else {
                        // Unsupported cell type (Quad4, Hex8, higher-order): mark
                        // so we don't build a provider with partial/zero weights.
                        has_unsupported_cells = true;
                        break;
                    }

                    const double w_per_vert = vol / static_cast<double>(nv);
                    for (const auto node : cell_nodes) {
                        // Map mesh vertex → all DOFs at this vertex, accumulate weight
                        auto vdofs = emap->getVertexDofs(node);
                        for (const auto d : vdofs) {
                            if (d >= 0 && static_cast<std::size_t>(d) < vertex_mass->size()) {
                                (*vertex_mass)[static_cast<std::size_t>(d)] += w_per_vert;
                            }
                        }
                    }
                }

                // Only build the provider if all cells were supported simplices.
                // If any unsupported cell was encountered, leave mass_weight_provider
                // null so that applyEnforcement falls back to uniform weights.
                if (!has_unsupported_cells) {
                auto field_offsets = field_dof_offsets_;
                auto n_fields = field_dof_handlers_.size();
                auto field_handlers_ptr = &field_dof_handlers_;
                auto field_map_ptr = &field_map_;
                mass_weight_provider = [vertex_mass, field_offsets, n_fields,
                                        field_handlers_ptr, field_map_ptr]
                                       (FieldId fid, int component) -> std::vector<double> {
                    const auto idx = static_cast<std::size_t>(fid);
                    if (idx >= n_fields) return {};

                    // Get DOFs for this field/component (same logic as get_field_dofs)
                    std::vector<GlobalIndex> dofs;
                    if (component >= 0 && idx < field_map_ptr->numFields()) {
                        const auto n_comp = field_map_ptr->numComponents(idx);
                        if (n_comp > 1 && static_cast<LocalIndex>(component) < n_comp) {
                            dofs = field_map_ptr->getComponentDofs(idx, static_cast<LocalIndex>(component)).toVector();
                        }
                    }
                    if (dofs.empty()) {
                        const auto offset = field_offsets[idx];
                        const auto nd = (*field_handlers_ptr)[idx].getNumDofs();
                        dofs.reserve(static_cast<std::size_t>(nd));
                        for (GlobalIndex d = offset; d < offset + nd; ++d) {
                            dofs.push_back(d);
                        }
                    }

                    // Extract weights and validate: if any DOF has zero weight
                    // (unsupported cell type like Quad4), return empty to trigger
                    // the uniform-weight fallback in applyEnforcement.
                    std::vector<double> weights;
                    weights.reserve(dofs.size());
                    bool all_positive = true;
                    for (const auto d : dofs) {
                        double w = 0.0;
                        if (d >= 0 && static_cast<std::size_t>(d) < vertex_mass->size()) {
                            w = (*vertex_mass)[static_cast<std::size_t>(d)];
                        }
                        if (w <= 0.0) {
                            all_positive = false;
                            break;
                        }
                        weights.push_back(w);
                    }
                    if (!all_positive) return {};  // fallback to uniform
                    return weights;
                };
                } // if (!has_unsupported_cells)
            }
        }

        gauge_registry_->resolve(get_field_dofs, region_provider, coord_provider);

#if FE_HAS_MPI
        {
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized &&
                !gaugeResolutionConsistentAcrossRanks(*gauge_registry_, dof_handler_.mpiComm())) {
                FE_THROW(FEException,
                         "FESystem::setup: GaugeRegistry resolved modes differ across MPI ranks. "
                         "This would create inconsistent replicated algebraic constraints.");
            }
        }
#endif

        // Apply enforcement: auto-create GlobalConstraint objects for
        // resolved exact-nullspace modes.
        const int n_gauge_constraints =
            gauge_registry_->applyEnforcement(affine_constraints_, get_field_dofs,
                                               mass_weight_provider);

        if (n_gauge_constraints > 0) {
            std::fprintf(stderr,
                "[GaugeRegistry] Applied %d automatic gauge constraint(s)\n",
                n_gauge_constraints);
        }

        // Diagnostic report (opt-in via SetupOptions or environment variable)
        if (opts.gauge_diagnostics || gauge::isNullspaceValidationEnabled()) {
            std::fprintf(stderr, "%s", gauge_registry_->diagnosticReport().c_str());
        }
    }

    // Synchronize in MPI before closing.
    //
    // NOTE: The partition-only ParallelConstraints ctor is a serial/no-op helper (world_size=1).
    // In MPI runs we must construct with an MPI communicator so constraints on shared/ghost DOFs
    // are imported from the owning rank before close()/assembly.
#if FE_HAS_MPI
    int mpi_initialized_constraints = 0;
    MPI_Initialized(&mpi_initialized_constraints);
    std::optional<constraints::ParallelConstraints> parallel;
    if (mpi_initialized_constraints) {
        parallel.emplace(dof_handler_.mpiComm(), dof_handler_.getPartition());
    } else {
        parallel.emplace(dof_handler_.getPartition());
    }
#else
    std::optional<constraints::ParallelConstraints> parallel;
    parallel.emplace(dof_handler_.getPartition());
#endif
    if (parallel && parallel->isParallel()) {
        parallel->synchronize(affine_constraints_);
        if (!parallel->validateConsistency(affine_constraints_)) {
            FE_THROW(FEException,
                     "FESystem::setup: algebraic constraints are inconsistent across MPI ranks "
                     "after synchronization.");
        }
    }

    affine_constraints_.close();
    bumpConstraintLayoutRevision();
    constraint_revision_snapshot_ = captureConstraintRevisionSnapshot();

    // ---------------------------------------------------------------------
    // Analysis subsystem: topology + constraint summary
    // ---------------------------------------------------------------------
    if (mesh_access_) {
        buildTopologyContext();
    }
    buildInterfaceTopologyContext();
    buildConstraintSummary();

    // ---------------------------------------------------------------------
    // Sparsity
    // ---------------------------------------------------------------------
    sparsity_by_op_.clear();
    distributed_sparsity_by_op_.clear();
    const auto op_tags = operator_registry_.list();
    const auto n_total_dofs = dof_handler_.getNumDofs();
    const auto n_cells_sparsity = meshAccess().numCells();

    const auto& partition = dof_handler_.getPartition();
    bool mpi_parallel =
        (partition.globalSize() > 0) && (partition.globalSize() > partition.localOwnedSize());
#if FE_HAS_MPI
    {
        int mpi_initialized_parallel = 0;
        MPI_Initialized(&mpi_initialized_parallel);
        if (mpi_initialized_parallel && partition.globalSize() > 0) {
            int comm_size = 1;
            MPI_Comm_size(dof_handler_.mpiComm(), &comm_size);
            if (comm_size > 1) {
                const int local_partial =
                    (partition.globalSize() > partition.localOwnedSize()) ? 1 : 0;
                int any_partial = 0;
                MPI_Allreduce(&local_partial,
                              &any_partial,
                              1,
                              MPI_INT,
                              MPI_MAX,
                              dof_handler_.mpiComm());
                mpi_parallel = (any_partial != 0);
            }
        }
    }
#endif

    enum class DistSparsityMode { None, ContiguousRange, NodalInterleaved };

    DistSparsityMode dist_mode = DistSparsityMode::None;
    sparsity::IndexRange owned_range{};
    // Backend DOF permutation (FSILS overlap) is needed in MPI even when FE owned DOFs are already
    // owner-contiguous. When assembly uses backend row ownership, the distributed sparsity must use
    // that same backend row ownership/indexing; otherwise the symbolic graph and numeric assembly
    // route rows through different owners.
    std::optional<NodalInterleavedDofMap> backend_map{};
    std::optional<NodalInterleavedDofMap> nodal_map{};

#if FE_HAS_MPI
    int mpi_initialized_dof_map = 0;
    MPI_Initialized(&mpi_initialized_dof_map);
    if (mpi_initialized_dof_map) {
        backend_map = tryBuildNodalInterleavedDofMap(dof_handler_, field_map_, meshAccess(), opts.dof_options);
    }
#endif

    const auto owned_iv_opt = partition.locallyOwned().contiguousRange();
    if (mpi_parallel && backend_map.has_value() && opts.use_backend_row_ownership_for_assembly) {
        dist_mode = DistSparsityMode::NodalInterleaved;
        owned_range = backend_map->owned_range;
        nodal_map = std::move(backend_map);
    } else if (mpi_parallel && owned_iv_opt.has_value()) {
        dist_mode = DistSparsityMode::ContiguousRange;
        owned_range = sparsity::IndexRange{owned_iv_opt->begin, owned_iv_opt->end};
    } else if (mpi_parallel && backend_map.has_value()) {
        dist_mode = DistSparsityMode::NodalInterleaved;
        owned_range = backend_map->owned_range;
        nodal_map = std::move(backend_map);
    }

    const auto& ghost_set = partition.ghost();
    const auto& relevant_set = partition.locallyRelevant();

    for (const auto& tag : op_tags) {
        const auto& def = operator_registry_.get(tag);
        const bool build_serial_pattern =
            opts.retain_serial_sparsity || dist_mode == DistSparsityMode::None || !def.global.empty();

        std::unique_ptr<sparsity::SparsityPattern> pattern;
        if (build_serial_pattern) {
            pattern = std::make_unique<sparsity::SparsityPattern>(
                n_total_dofs, n_total_dofs);
        }

        std::unique_ptr<sparsity::DistributedSparsityPattern> dist_pattern;
        std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> ghost_row_cols;
        if (dist_mode != DistSparsityMode::None) {
            dist_pattern = std::make_unique<sparsity::DistributedSparsityPattern>(
                owned_range, owned_range, n_total_dofs, n_total_dofs);
            dist_pattern->setDofIndexing(dist_mode == DistSparsityMode::NodalInterleaved
                                             ? sparsity::DistributedSparsityPattern::DofIndexing::NodalInterleaved
                                             : sparsity::DistributedSparsityPattern::DofIndexing::Natural);
        }

#if FE_HAS_MPI
        int pattern_rank = opts.dof_options.my_rank;
        int pattern_size = opts.dof_options.world_size;
        bool remote_pattern_exchange_enabled = false;
        using RemotePatternRows = std::unordered_map<GlobalIndex, std::vector<GlobalIndex>>;
        std::vector<RemotePatternRows> remote_pattern_by_rank;
        if (dist_pattern) {
            int mpi_initialized_pattern = 0;
            MPI_Initialized(&mpi_initialized_pattern);
            if (mpi_initialized_pattern) {
                MPI_Comm_rank(opts.dof_options.mpi_comm, &pattern_rank);
                MPI_Comm_size(opts.dof_options.mpi_comm, &pattern_size);
                remote_pattern_exchange_enabled = pattern_size > 1;
                if (remote_pattern_exchange_enabled) {
                    remote_pattern_by_rank.resize(static_cast<std::size_t>(pattern_size));
                }
            }
        }

        auto queue_remote_pattern_entry = [&](int owner_rank, GlobalIndex row, GlobalIndex col) {
            if (!remote_pattern_exchange_enabled || !dist_pattern) {
                return;
            }
            if (row < 0 || row >= n_total_dofs || col < 0 || col >= n_total_dofs) {
                return;
            }
            if (owner_rank < 0 || owner_rank >= pattern_size) {
                return;
            }
            if (owner_rank == pattern_rank) {
                if (owned_range.contains(row)) {
                    dist_pattern->addEntry(row, col);
                }
                return;
            }
            auto& cols_for_row =
                remote_pattern_by_rank[static_cast<std::size_t>(owner_rank)][row];
            insertSortedUniqueIndex(cols_for_row, col);
        };

        auto flush_remote_pattern_entries = [&]() {
            if (!remote_pattern_exchange_enabled || !dist_pattern) {
                return;
            }

            std::vector<int> send_counts(static_cast<std::size_t>(pattern_size), 0);
            std::vector<int> recv_counts(static_cast<std::size_t>(pattern_size), 0);
            std::vector<int> send_displs(static_cast<std::size_t>(pattern_size), 0);
            std::vector<int> recv_displs(static_cast<std::size_t>(pattern_size), 0);
            std::vector<std::vector<GlobalIndex>> sorted_remote_rows(static_cast<std::size_t>(pattern_size));

            int total_send = 0;
            for (int r = 0; r < pattern_size; ++r) {
                const auto& rows = remote_pattern_by_rank[static_cast<std::size_t>(r)];
                auto& sorted_rows = sorted_remote_rows[static_cast<std::size_t>(r)];
                sorted_rows.reserve(rows.size());
                int n_values = 0;
                for (const auto& [row, cols] : rows) {
                    sorted_rows.push_back(row);
                    n_values += static_cast<int>(cols.size() * 2u);
                }
                std::sort(sorted_rows.begin(), sorted_rows.end());

                send_displs[static_cast<std::size_t>(r)] = total_send;
                send_counts[static_cast<std::size_t>(r)] = n_values;
                total_send += n_values;
            }

            std::vector<GlobalIndex> send_data(static_cast<std::size_t>(total_send));
            for (int r = 0; r < pattern_size; ++r) {
                auto offset = send_displs[static_cast<std::size_t>(r)];
                const auto& rows = remote_pattern_by_rank[static_cast<std::size_t>(r)];
                for (const auto row : sorted_remote_rows[static_cast<std::size_t>(r)]) {
                    const auto it = rows.find(row);
                    if (it == rows.end()) {
                        continue;
                    }
                    for (const auto col : it->second) {
                        send_data[static_cast<std::size_t>(offset++)] = row;
                        send_data[static_cast<std::size_t>(offset++)] = col;
                    }
                }
            }

            MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                         recv_counts.data(), 1, MPI_INT,
                         opts.dof_options.mpi_comm);

            int total_recv = 0;
            for (int r = 0; r < pattern_size; ++r) {
                recv_displs[static_cast<std::size_t>(r)] = total_recv;
                total_recv += recv_counts[static_cast<std::size_t>(r)];
            }

            std::vector<GlobalIndex> recv_data(static_cast<std::size_t>(total_recv));
            MPI_Alltoallv(send_data.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                          recv_data.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                          opts.dof_options.mpi_comm);

            FE_THROW_IF((recv_data.size() % 2u) != 0u, InvalidStateException,
                        "FESystem::setup: received malformed remote distributed sparsity payload");
            for (std::size_t i = 0; i < recv_data.size(); i += 2u) {
                const GlobalIndex row = recv_data[i];
                const GlobalIndex col = recv_data[i + 1u];
                if (owned_range.contains(row) && col >= 0 && col < n_total_dofs) {
                    dist_pattern->addEntry(row, col);
                }
            }
            for (auto& rows : remote_pattern_by_rank) {
                rows.clear();
            }
        };
#else
        auto queue_remote_pattern_entry = [](int, GlobalIndex, GlobalIndex) {};
        auto flush_remote_pattern_entries = []() {};
#endif

		        std::vector<std::pair<FieldId, FieldId>> cell_pairs;
		        std::vector<std::tuple<int, FieldId, FieldId>> boundary_pairs;
		        std::vector<std::pair<FieldId, FieldId>> interior_pairs;
		        std::vector<std::tuple<int, FieldId, FieldId>> interface_pairs;

		        cell_pairs.reserve(def.cells.size());
		        boundary_pairs.reserve(def.boundary.size());
		        interior_pairs.reserve(def.interior.size());
		        interface_pairs.reserve(def.interface_faces.size());

	        auto maybe_add_cell_pair =
	            [&](FieldId test, FieldId trial, const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
	                if (!kernel) {
	                    return;
	                }

                    if (kernel->semanticKernelKind() == assembly::SemanticKernelKind::MonolithicCell) {
                        const auto* monolithic =
                            dynamic_cast<const forms::MonolithicCellKernel*>(kernel.get());
                        FE_CHECK_NOT_NULL(monolithic, "FESystem::setup: monolithic cell kernel in sparsity build");
                        for (std::size_t bi = 0; bi < monolithic->numBlocks(); ++bi) {
                            const auto& bs = monolithic->blockSpec(bi);
                            if (!bs.want_matrix) {
                                continue;
                            }
                            cell_pairs.emplace_back(bs.test_field, bs.trial_field);
                        }
                        return;
                    }

                    if (kernel->semanticKernelKind() == assembly::SemanticKernelKind::MixedBlockSet) {
                        const auto* mixed_block =
                            dynamic_cast<const forms::MixedBlockKernelSet*>(kernel.get());
                        FE_CHECK_NOT_NULL(mixed_block, "FESystem::setup: mixed block cell kernel in sparsity build");
                        for (std::size_t bi = 0; bi < mixed_block->numBlocks(); ++bi) {
                            const auto& bs = mixed_block->blockSpec(bi);
                            if (!bs.want_matrix) {
                                continue;
                            }
                            cell_pairs.emplace_back(bs.test_field, bs.trial_field);
                        }
                        return;
                    }

	                if (kernel->isVectorOnly()) {
	                    return;
	                }
	                cell_pairs.emplace_back(test, trial);
	            };

	        auto maybe_add_boundary_pair =
	            [&](int marker,
	                FieldId test,
	                FieldId trial,
	                const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
	                if (!kernel || kernel->isVectorOnly()) {
	                    return;
	                }
	                boundary_pairs.emplace_back(marker, test, trial);
	            };

		        auto maybe_add_interior_pair =
		            [&](FieldId test, FieldId trial, const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
		                if (!kernel || kernel->isVectorOnly()) {
		                    return;
		                }
		                interior_pairs.emplace_back(test, trial);
		            };

		        auto maybe_add_interface_pair =
		            [&](int marker,
		                FieldId test,
		                FieldId trial,
		                const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
		                if (!kernel || kernel->isVectorOnly()) {
		                    return;
		                }
		                interface_pairs.emplace_back(marker, test, trial);
		            };

	        for (const auto& term : def.cells) {
	            maybe_add_cell_pair(term.test_field, term.trial_field, term.kernel);
	        }
	        for (const auto& term : def.boundary) {
	            maybe_add_boundary_pair(term.marker, term.test_field, term.trial_field, term.kernel);
	        }
		        for (const auto& term : def.interior) {
		            maybe_add_interior_pair(term.test_field, term.trial_field, term.kernel);
		        }
		        for (const auto& term : def.interface_faces) {
		            maybe_add_interface_pair(term.marker, term.test_field, term.trial_field, term.kernel);
		        }

	        std::sort(cell_pairs.begin(), cell_pairs.end());
	        cell_pairs.erase(std::unique(cell_pairs.begin(), cell_pairs.end()), cell_pairs.end());

	        std::sort(boundary_pairs.begin(), boundary_pairs.end());
	        boundary_pairs.erase(std::unique(boundary_pairs.begin(), boundary_pairs.end()), boundary_pairs.end());

		        std::sort(interior_pairs.begin(), interior_pairs.end());
		        interior_pairs.erase(std::unique(interior_pairs.begin(), interior_pairs.end()), interior_pairs.end());

		        std::sort(interface_pairs.begin(), interface_pairs.end());
		        interface_pairs.erase(std::unique(interface_pairs.begin(), interface_pairs.end()), interface_pairs.end());

		        std::vector<GlobalIndex> row_dofs;
		        std::vector<GlobalIndex> col_dofs;

				        auto add_element_couplings = [&](std::span<const GlobalIndex> rows,
				                                         std::span<const GlobalIndex> cols) {
				            if (pattern) {
				                pattern->addElementCouplings(rows, cols);
				            }
				            if (!dist_pattern) {
				                return;
			            }

				            if (dist_mode == DistSparsityMode::ContiguousRange) {
				                for (const auto global_row : rows) {
				                    if (owned_range.contains(global_row)) {
				                        dist_pattern->addEntries(global_row, cols);
				                        continue;
				                    }

				                    const int owner = dof_handler_.getDofMap().getDofOwner(global_row);
				                    for (const auto global_col : cols) {
			                        queue_remote_pattern_entry(owner, global_row, global_col);
			                    }

			                    if (!ghost_set.contains(global_row)) {
			                        continue;
			                    }

				                    auto& cols_for_row = ghost_row_cols[global_row];
				                    for (const auto global_col : cols) {
				                        if (relevant_set.contains(global_col)) {
				                            insertSortedUniqueIndex(cols_for_row, global_col);
				                        }
				                    }
				                }
				                return;
				            }

			            FE_THROW_IF(dist_mode != DistSparsityMode::NodalInterleaved || !nodal_map.has_value(),
			                        InvalidStateException,
			                        "FESystem::setup: missing nodal interleaved mapping for distributed sparsity build");

				            const auto& nodal = *nodal_map;
				            const int dof = nodal.dof_per_node;
				            std::vector<GlobalIndex> cols_fs;
				            cols_fs.reserve(cols.size());
				            for (const auto global_col_fe : cols) {
				                const auto global_col_fs = nodal.mapFeToFs(global_col_fe);
				                if (global_col_fs != INVALID_GLOBAL_INDEX) {
				                    insertSortedUniqueIndex(cols_fs, global_col_fs);
				                }
				            }

				            for (const auto global_row_fe : rows) {
				                const auto global_row_fs = nodal.mapFeToFs(global_row_fe);
				                if (global_row_fs == INVALID_GLOBAL_INDEX) {
				                    continue;
				                }

				                if (owned_range.contains(global_row_fs)) {
				                    dist_pattern->addEntries(global_row_fs, cols_fs);
				                    continue;
				                }

				                const int owner = nodal.ownerRankForDof(global_row_fs);
				                for (const auto global_col_fs : cols_fs) {
				                    queue_remote_pattern_entry(owner, global_row_fs, global_col_fs);
				                }

			                const int node = static_cast<int>(global_row_fs / dof);
			                if (node < 0 || static_cast<GlobalIndex>(node) >= nodal.n_nodes) {
			                    continue;
			                }

				                auto& cols_for_row = ghost_row_cols[global_row_fs];
				                for (const auto global_col_fs : cols_fs) {
				                    if (nodal.isRelevantDof(global_col_fs)) {
				                        insertSortedUniqueIndex(cols_for_row, global_col_fs);
				                    }
				                }
				            }
				        };

		        auto add_cell_couplings = [&](GlobalIndex cell_id,
		                                      const dofs::DofMap& row_map,
		                                      const dofs::DofMap& col_map,
	                                      GlobalIndex row_offset,
	                                      GlobalIndex col_offset) {
	            auto row_local = row_map.getCellDofs(cell_id);
	            auto col_local = col_map.getCellDofs(cell_id);

	            row_dofs.resize(row_local.size());
	            for (std::size_t i = 0; i < row_local.size(); ++i) {
	                row_dofs[i] = row_local[i] + row_offset;
	            }

	            col_dofs.resize(col_local.size());
	            for (std::size_t j = 0; j < col_local.size(); ++j) {
	                col_dofs[j] = col_local[j] + col_offset;
	            }

		            add_element_couplings(row_dofs, col_dofs);
		        };

	        // Cell terms: all cells participate.
	        for (const auto& [test_field, trial_field] : cell_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in sparsity build");

            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

		            auto add_sparse_cell = [&](GlobalIndex cell) {
		                add_cell_couplings(cell, row_map, col_map, row_offset, col_offset);
		            };
		            if (dist_pattern) {
		                meshAccess().forEachOwnedCell(add_sparse_cell);
		            } else {
		                for (GlobalIndex cell = 0; cell < n_cells_sparsity; ++cell) {
		                    add_sparse_cell(cell);
		                }
		            }
		        }

	        // Boundary terms: only cells adjacent to the requested marker participate.
	        std::vector<GlobalIndex> marker_cells;
	        for (const auto& [marker, test_field, trial_field] : boundary_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in boundary sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in boundary sparsity build");

	            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            marker_cells.clear();
	            meshAccess().forEachBoundaryFace(marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
	                marker_cells.push_back(cell_id);
	            });
	            std::sort(marker_cells.begin(), marker_cells.end());
	            marker_cells.erase(std::unique(marker_cells.begin(), marker_cells.end()), marker_cells.end());

	            for (const auto cell_id : marker_cells) {
	                add_cell_couplings(cell_id, row_map, col_map, row_offset, col_offset);
	            }
	        }

	        // Interior face terms (DG): include both self and cross-cell couplings.
	        std::vector<GlobalIndex> row_dofs_minus, row_dofs_plus;
	        std::vector<GlobalIndex> col_dofs_minus, col_dofs_plus;
	        for (const auto& [test_field, trial_field] : interior_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in interior-face sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in interior-face sparsity build");

	            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            meshAccess().forEachInteriorFace(
	                [&](GlobalIndex /*face_id*/, GlobalIndex cell_minus, GlobalIndex cell_plus) {
	                    auto minus_row_local = row_map.getCellDofs(cell_minus);
	                    auto plus_row_local = row_map.getCellDofs(cell_plus);
	                    auto minus_col_local = col_map.getCellDofs(cell_minus);
	                    auto plus_col_local = col_map.getCellDofs(cell_plus);

	                    row_dofs_minus.resize(minus_row_local.size());
	                    for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
	                        row_dofs_minus[i] = minus_row_local[i] + row_offset;
	                    }
	                    row_dofs_plus.resize(plus_row_local.size());
	                    for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
	                        row_dofs_plus[i] = plus_row_local[i] + row_offset;
	                    }

	                    col_dofs_minus.resize(minus_col_local.size());
	                    for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
	                        col_dofs_minus[j] = minus_col_local[j] + col_offset;
	                    }
	                    col_dofs_plus.resize(plus_col_local.size());
	                    for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
	                        col_dofs_plus[j] = plus_col_local[j] + col_offset;
	                    }

		                    add_element_couplings(row_dofs_minus, col_dofs_minus);
		                    add_element_couplings(row_dofs_plus, col_dofs_plus);
		                    add_element_couplings(row_dofs_minus, col_dofs_plus);
		                    add_element_couplings(row_dofs_plus, col_dofs_minus);
		                });
		        }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
		        // Interface face terms (InterfaceMesh subset): include both self and cross-cell couplings.
		        std::vector<GlobalIndex> iface_row_dofs_minus, iface_row_dofs_plus;
		        std::vector<GlobalIndex> iface_col_dofs_minus, iface_col_dofs_plus;
		        for (const auto& [marker, test_field, trial_field] : interface_pairs) {
		            const auto test_idx = static_cast<std::size_t>(test_field);
		            const auto trial_idx = static_cast<std::size_t>(trial_field);

		            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
		                        "FESystem::setup: invalid test field in interface-face sparsity build");
		            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
		                        "FESystem::setup: invalid trial field in interface-face sparsity build");

		            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
		            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
		            const auto row_offset = field_dof_offsets_[test_idx];
		            const auto col_offset = field_dof_offsets_[trial_idx];
                    const bool test_is_interface_field =
                        field_registry_.get(test_field).scope == FieldScope::InterfaceFace;
                    const bool trial_is_interface_field =
                        field_registry_.get(trial_field).scope == FieldScope::InterfaceFace;

		            auto add_interface_mesh_couplings = [&](const svmp::InterfaceMesh& iface) {
		                for (std::size_t lf = 0; lf < iface.n_faces(); ++lf) {
		                    const auto local_face = static_cast<svmp::index_t>(lf);
		                    const GlobalIndex cell_minus = static_cast<GlobalIndex>(iface.volume_cell_minus(local_face));
		                    const GlobalIndex cell_plus = static_cast<GlobalIndex>(iface.volume_cell_plus(local_face));

                            iface_row_dofs_minus.clear();
                            iface_row_dofs_plus.clear();
                            iface_col_dofs_minus.clear();
                            iface_col_dofs_plus.clear();

                            if (test_is_interface_field) {
                                auto local_face_row = row_map.getCellDofs(static_cast<GlobalIndex>(lf));
                                iface_row_dofs_minus.resize(local_face_row.size());
                                for (std::size_t i = 0; i < local_face_row.size(); ++i) {
                                    iface_row_dofs_minus[i] = local_face_row[i] + row_offset;
                                }
                            } else {
                                if (cell_minus != INVALID_GLOBAL_INDEX) {
                                    auto minus_row_local = row_map.getCellDofs(cell_minus);
                                    iface_row_dofs_minus.resize(minus_row_local.size());
                                    for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
                                        iface_row_dofs_minus[i] = minus_row_local[i] + row_offset;
                                    }
                                }
                                if (cell_plus != INVALID_GLOBAL_INDEX) {
                                    auto plus_row_local = row_map.getCellDofs(cell_plus);
                                    iface_row_dofs_plus.resize(plus_row_local.size());
                                    for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
                                        iface_row_dofs_plus[i] = plus_row_local[i] + row_offset;
                                    }
                                }
                            }

                            if (trial_is_interface_field) {
                                auto local_face_col = col_map.getCellDofs(static_cast<GlobalIndex>(lf));
                                iface_col_dofs_minus.resize(local_face_col.size());
                                for (std::size_t j = 0; j < local_face_col.size(); ++j) {
                                    iface_col_dofs_minus[j] = local_face_col[j] + col_offset;
                                }
                            } else {
                                if (cell_minus != INVALID_GLOBAL_INDEX) {
                                    auto minus_col_local = col_map.getCellDofs(cell_minus);
                                    iface_col_dofs_minus.resize(minus_col_local.size());
                                    for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
                                        iface_col_dofs_minus[j] = minus_col_local[j] + col_offset;
                                    }
                                }
                                if (cell_plus != INVALID_GLOBAL_INDEX) {
                                    auto plus_col_local = col_map.getCellDofs(cell_plus);
                                    iface_col_dofs_plus.resize(plus_col_local.size());
                                    for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
                                        iface_col_dofs_plus[j] = plus_col_local[j] + col_offset;
                                    }
                                }
                            }

                            if (test_is_interface_field && trial_is_interface_field) {
                                add_element_couplings(iface_row_dofs_minus, iface_col_dofs_minus);
                            } else if (test_is_interface_field) {
                                if (!iface_col_dofs_minus.empty()) {
                                    add_element_couplings(iface_row_dofs_minus, iface_col_dofs_minus);
                                }
                                if (!iface_col_dofs_plus.empty()) {
                                    add_element_couplings(iface_row_dofs_minus, iface_col_dofs_plus);
                                }
                            } else if (trial_is_interface_field) {
                                if (!iface_row_dofs_minus.empty()) {
                                    add_element_couplings(iface_row_dofs_minus, iface_col_dofs_minus);
                                }
                                if (!iface_row_dofs_plus.empty()) {
                                    add_element_couplings(iface_row_dofs_plus, iface_col_dofs_minus);
                                }
                            } else {
                                if (cell_minus == INVALID_GLOBAL_INDEX || cell_plus == INVALID_GLOBAL_INDEX) {
                                    continue;
                                }
                                add_element_couplings(iface_row_dofs_minus, iface_col_dofs_minus);
                                add_element_couplings(iface_row_dofs_plus, iface_col_dofs_plus);
                                add_element_couplings(iface_row_dofs_minus, iface_col_dofs_plus);
                                add_element_couplings(iface_row_dofs_plus, iface_col_dofs_minus);
                            }
		                }
		            };

		            if (marker < 0) {
		                FE_THROW_IF(interface_meshes_.empty(), InvalidStateException,
		                            "FESystem::setup: interface-face kernels registered for all markers, but no InterfaceMesh was set");
		                for (const auto& kv : interface_meshes_) {
		                    if (!kv.second) continue;
		                    add_interface_mesh_couplings(*kv.second);
		                }
		            } else {
		                auto it = interface_meshes_.find(marker);
		                FE_THROW_IF(it == interface_meshes_.end() || !it->second, InvalidStateException,
		                            "FESystem::setup: missing InterfaceMesh for interface marker " + std::to_string(marker));
		                add_interface_mesh_couplings(*it->second);
		            }
		        }
#endif

	        // Global terms: allow kernels to conservatively augment sparsity.
	        for (const auto& kernel : def.global) {
	            if (kernel) {
	                FE_CHECK_NOT_NULL(pattern.get(), "FESystem::setup: serial sparsity pattern for global kernel");
	                kernel->addSparsityCouplings(*this, *pattern);
	            }
        }

        flush_remote_pattern_entries();

		        if (pattern && opts.sparsity_options.ensure_diagonal) {
		            pattern->ensureDiagonal();
		        }
	        if (pattern && opts.sparsity_options.ensure_non_empty_rows) {
	            pattern->ensureNonEmptyRows();
	        }
	        if (dist_pattern) {
	            if (opts.sparsity_options.ensure_diagonal) {
	                dist_pattern->ensureDiagonal();
	            }
	            if (opts.sparsity_options.ensure_non_empty_rows) {
	                dist_pattern->ensureNonEmptyRows();
	            }
	        }
	
		        if (opts.use_constraints_in_assembly && !affine_constraints_.empty()) {
		            if (pattern) {
		                auto query = std::make_shared<AffineConstraintsQuery>(affine_constraints_);
		                sparsity::ConstraintSparsityAugmenter augmenter(std::move(query));
		                augmenter.augment(*pattern, sparsity::AugmentationMode::EliminationFill);
		            }
		            if (dist_pattern) {
		                if (dist_mode == DistSparsityMode::NodalInterleaved) {
		                    FE_THROW_IF(!nodal_map.has_value(), InvalidStateException,
		                                "FESystem::setup: missing nodal mapping for constraint sparsity augmentation");
		                    auto dist_query = std::make_shared<PermutedAffineConstraintsQuery>(
		                        affine_constraints_,
		                        std::span<const GlobalIndex>(nodal_map->fe_to_fs),
		                        std::span<const GlobalIndex>(nodal_map->fs_to_fe));
		                    sparsity::ConstraintSparsityAugmenter dist_augmenter(std::move(dist_query));
		                    dist_augmenter.augment(*dist_pattern, sparsity::AugmentationMode::EliminationFill);
		                } else {
		                    auto dist_query = std::make_shared<AffineConstraintsQuery>(affine_constraints_);
		                    sparsity::ConstraintSparsityAugmenter dist_augmenter(std::move(dist_query));
		                    dist_augmenter.augment(*dist_pattern, sparsity::AugmentationMode::EliminationFill);
		                }
		            }
		        }
	
	        if (pattern && opts.sparsity_options.ensure_diagonal) {
	            pattern->ensureDiagonal();
	        }
	        if (pattern && opts.sparsity_options.ensure_non_empty_rows) {
	            pattern->ensureNonEmptyRows();
	        }
	        if (dist_pattern) {
	            if (opts.sparsity_options.ensure_diagonal) {
	                dist_pattern->ensureDiagonal();
	            }
	            if (opts.sparsity_options.ensure_non_empty_rows) {
	                dist_pattern->ensureNonEmptyRows();
	            }
	        }
	
		        if (pattern) {
		            pattern->finalize();
		            sparsity_by_op_.emplace(tag, std::move(pattern));
		        }
		        const auto* full_pattern = [&]() -> const sparsity::SparsityPattern* {
		            auto it = sparsity_by_op_.find(tag);
		            return (it != sparsity_by_op_.end() && it->second) ? it->second.get() : nullptr;
		        }();

		        if (dist_pattern) {
		            dist_pattern->finalize();

		            // Store optional ghost-row sparsity using the locally relevant (owned+ghost) overlap.
		            // This is required by overlap-style MPI backends (e.g., FSILS) so all column nodes
		            // referenced by owned rows are present locally.
		            if (dist_mode == DistSparsityMode::NodalInterleaved) {
		                FE_THROW_IF(!nodal_map.has_value(), InvalidStateException,
		                            "FESystem::setup: missing nodal mapping for ghost-row sparsity storage");
		                const auto& nodal = *nodal_map;
		                const int dof = nodal.dof_per_node;

		                std::vector<int> ghost_nodes_closed = nodal.ghost_nodes;
		                for (const auto col_fs : dist_pattern->getGhostColMap()) {
		                    if (col_fs < 0 || col_fs >= n_total_dofs) {
		                        continue;
		                    }
		                    const int node = static_cast<int>(col_fs / dof);
		                    if (node < 0 || static_cast<GlobalIndex>(node) >= nodal.n_nodes) {
		                        continue;
		                    }
		                    const GlobalIndex first_dof =
		                        static_cast<GlobalIndex>(node) * static_cast<GlobalIndex>(dof);
		                    if (owned_range.contains(first_dof)) {
		                        continue;
		                    }
		                    ghost_nodes_closed.push_back(node);
		                }
		                std::sort(ghost_nodes_closed.begin(), ghost_nodes_closed.end());
		                ghost_nodes_closed.erase(std::unique(ghost_nodes_closed.begin(), ghost_nodes_closed.end()),
		                                         ghost_nodes_closed.end());

		                std::vector<unsigned char> ghost_node_in_overlap(
		                    static_cast<std::size_t>(std::max<GlobalIndex>(0, nodal.n_nodes)), 0u);
		                for (const int node : ghost_nodes_closed) {
		                    if (node >= 0 && static_cast<GlobalIndex>(node) < nodal.n_nodes) {
		                        ghost_node_in_overlap[static_cast<std::size_t>(node)] = 1u;
		                    }
		                }

		                auto col_is_in_overlap = [&](GlobalIndex col_fs) {
		                    if (col_fs < 0 || col_fs >= n_total_dofs) {
		                        return false;
		                    }
		                    const auto node = static_cast<GlobalIndex>(col_fs / dof);
		                    if (node < 0 || static_cast<std::size_t>(node) >= ghost_node_in_overlap.size()) {
		                        return false;
		                    }
		                    return nodal.isRelevantDof(col_fs) ||
		                           ghost_node_in_overlap[static_cast<std::size_t>(node)] != 0u;
		                };

		                std::vector<GlobalIndex> ghost_row_map;
		                ghost_row_map.reserve(ghost_nodes_closed.size() * static_cast<std::size_t>(std::max(1, dof)));
		                for (const int node : ghost_nodes_closed) {
		                    for (int c = 0; c < dof; ++c) {
		                        ghost_row_map.push_back(static_cast<GlobalIndex>(node) * dof + c);
		                    }
		                }

		                if (!ghost_row_map.empty()) {
		                    std::vector<GlobalIndex> ghost_row_ptr;
		                    std::vector<GlobalIndex> ghost_row_cols_flat;
		                    ghost_row_ptr.reserve(ghost_row_map.size() + 1);
		                    ghost_row_ptr.push_back(0);

			                    for (const auto row_fs : ghost_row_map) {
			                        std::vector<GlobalIndex> cols_vec;
			                        cols_vec.reserve(32);

			                        if (full_pattern != nullptr && row_fs >= 0 && row_fs < n_total_dofs &&
			                            static_cast<std::size_t>(row_fs) < nodal.fs_to_fe.size()) {
			                            const auto row_fe = nodal.fs_to_fe[static_cast<std::size_t>(row_fs)];
			                            if (row_fe >= 0 && row_fe < n_total_dofs) {
			                                const auto cols_fe = full_pattern->getRowSpan(row_fe);
			                                for (const auto col_fe : cols_fe) {
			                                    const auto col_fs = nodal.mapFeToFs(col_fe);
			                                    if (col_fs == INVALID_GLOBAL_INDEX) {
			                                        continue;
			                                    }
			                                    if (col_is_in_overlap(col_fs)) {
			                                        cols_vec.push_back(col_fs);
			                                    }
			                                }
			                            }
			                        } else {
			                            const auto it_cols = ghost_row_cols.find(row_fs);
			                            if (it_cols != ghost_row_cols.end()) {
			                                for (const auto col_fs : it_cols->second) {
			                                    if (col_is_in_overlap(col_fs)) {
			                                        cols_vec.push_back(col_fs);
			                                    }
			                                }
			                            }
			                        }

			                        cols_vec.push_back(row_fs); // ensure diagonal

			                        cols_vec.erase(std::remove_if(cols_vec.begin(),
			                                                     cols_vec.end(),
			                                                     [&](GlobalIndex col) {
			                                                         return !col_is_in_overlap(col);
			                                                     }),
			                                       cols_vec.end());

		                        std::sort(cols_vec.begin(), cols_vec.end());
		                        cols_vec.erase(std::unique(cols_vec.begin(), cols_vec.end()), cols_vec.end());

			                        ghost_row_cols_flat.insert(ghost_row_cols_flat.end(), cols_vec.begin(), cols_vec.end());
			                        ghost_row_ptr.push_back(static_cast<GlobalIndex>(ghost_row_cols_flat.size()));
			                    }

		                    dist_pattern->setGhostRows(std::move(ghost_row_map),
		                                              std::move(ghost_row_ptr),
		                                              std::move(ghost_row_cols_flat));
		                } else {
		                    dist_pattern->clearGhostRows();
		                }
		            } else {
		                auto ghost_rows_all = ghost_set.toVector();
		                std::vector<GlobalIndex> ghost_row_map;
		                ghost_row_map.reserve(ghost_rows_all.size());
		                for (const auto row : ghost_rows_all) {
		                    if (row < 0 || row >= n_total_dofs) {
		                        continue;
		                    }
		                    if (owned_range.contains(row)) {
		                        continue;
		                    }
		                    ghost_row_map.push_back(row);
		                }

		                if (!ghost_row_map.empty()) {
		                    // ghost_set.toVector() is already sorted/unique; keep it that way after filtering.
		                    std::vector<GlobalIndex> ghost_row_ptr;
		                    std::vector<GlobalIndex> ghost_row_cols_flat;
		                    ghost_row_ptr.reserve(ghost_row_map.size() + 1);
		                    ghost_row_ptr.push_back(0);

			                    for (const auto row : ghost_row_map) {
			                        std::vector<GlobalIndex> cols_vec;
			                        cols_vec.reserve(32);

			                        if (full_pattern != nullptr && row >= 0 && row < n_total_dofs) {
			                            const auto cols = full_pattern->getRowSpan(row);
			                            for (const auto col : cols) {
			                                if (relevant_set.contains(col)) {
			                                    cols_vec.push_back(col);
			                                }
			                            }
			                        } else {
			                            const auto it_cols = ghost_row_cols.find(row);
			                            if (it_cols != ghost_row_cols.end()) {
			                                for (const auto col : it_cols->second) {
			                                    if (relevant_set.contains(col)) {
			                                        cols_vec.push_back(col);
			                                    }
			                                }
			                            }
			                        }

			                        cols_vec.push_back(row); // ensure diagonal

		                        // Only store columns that are locally present (owned+ghost) so overlap backends can map them.
		                        cols_vec.erase(std::remove_if(cols_vec.begin(),
		                                                     cols_vec.end(),
		                                                     [&](GlobalIndex col) {
		                                                         return (col < 0) || (col >= n_total_dofs) || !relevant_set.contains(col);
		                                                     }),
		                                       cols_vec.end());

		                        std::sort(cols_vec.begin(), cols_vec.end());
		                        cols_vec.erase(std::unique(cols_vec.begin(), cols_vec.end()), cols_vec.end());

		                        ghost_row_cols_flat.insert(ghost_row_cols_flat.end(), cols_vec.begin(), cols_vec.end());
		                        ghost_row_ptr.push_back(static_cast<GlobalIndex>(ghost_row_cols_flat.size()));
		                    }

		                    dist_pattern->setGhostRows(std::move(ghost_row_map),
		                                              std::move(ghost_row_ptr),
		                                              std::move(ghost_row_cols_flat));
		                } else {
		                    dist_pattern->clearGhostRows();
		                }
		            }

		            distributed_sparsity_by_op_.emplace(tag, std::move(dist_pattern));
		        }
			    }

    // Persist a node-interleaved backend permutation for overlap backends (FSILS). This is needed
    // in MPI even when we used Natural indexing for distributed sparsity (owner-contiguous FE IDs).
    if (nodal_map.has_value() || backend_map.has_value()) {
        auto& map = nodal_map.has_value() ? *nodal_map : *backend_map;
        auto perm = std::make_shared<backends::DofPermutation>();
        perm->owner_rank.assign(map.fe_to_fs.size(), -1);
        if (map.dof_per_node > 0) {
            for (GlobalIndex backend_dof = 0;
                 backend_dof < static_cast<GlobalIndex>(perm->owner_rank.size());
                 ++backend_dof) {
                perm->owner_rank[static_cast<std::size_t>(backend_dof)] =
                    map.ownerRankForDof(backend_dof);
            }
        }
        perm->forward = std::move(map.fe_to_fs);
        perm->inverse = std::move(map.fs_to_fe);
        dof_permutation_ = std::move(perm);
    } else {
        dof_permutation_.reset();
    }

    // ---------------------------------------------------------------------
    // Per-cell material state storage (optional; for RequiredData::MaterialState)
    // ---------------------------------------------------------------------
    const auto requires_material_state_storage = [&]() {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (!term.kernel) continue;

                if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MonolithicCell) {
                    const auto* monolithic =
                        dynamic_cast<const forms::MonolithicCellKernel*>(term.kernel.get());
                    FE_CHECK_NOT_NULL(monolithic, "FESystem::setup: monolithic cell kernel");
                    for (std::size_t bi = 0; bi < monolithic->numBlocks(); ++bi) {
                        const auto& bs = monolithic->blockSpec(bi);
                        if (bs.fallback_kernel &&
                            assembly::hasFlag(bs.fallback_kernel->getRequiredData(),
                                              assembly::RequiredData::MaterialState)) {
                            return true;
                        }
                    }
                    continue;
                }

                if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MixedBlockSet) {
                    const auto* mixed =
                        dynamic_cast<const forms::MixedBlockKernelSet*>(term.kernel.get());
                    FE_CHECK_NOT_NULL(mixed, "FESystem::setup: mixed block cell kernel");
                    for (std::size_t bi = 0; bi < mixed->numBlocks(); ++bi) {
                        const auto& bs = mixed->blockSpec(bi);
                        if (bs.fallback_kernel &&
                            assembly::hasFlag(bs.fallback_kernel->getRequiredData(),
                                              assembly::RequiredData::MaterialState)) {
                            return true;
                        }
                    }
                    continue;
                }

                if (assembly::hasFlag(term.kernel->getRequiredData(),
                                      assembly::RequiredData::MaterialState)) {
                    return true;
                }
            }

            for (const auto& term : def.boundary) {
                if (term.kernel &&
                    assembly::hasFlag(term.kernel->getRequiredData(),
                                      assembly::RequiredData::MaterialState)) {
                    return true;
                }
            }
            for (const auto& term : def.interior) {
                if (term.kernel &&
                    assembly::hasFlag(term.kernel->getRequiredData(),
                                      assembly::RequiredData::MaterialState)) {
                    return true;
                }
            }
        }
        return false;
    };

    material_state_provider_.reset();
    if (requires_material_state_storage()) {
        std::vector<GlobalIndex> boundary_faces;
        meshAccess().forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
            boundary_faces.push_back(face_id);
        });

        std::vector<GlobalIndex> interior_faces;
        meshAccess().forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex /*cell_minus*/, GlobalIndex /*cell_plus*/) {
            interior_faces.push_back(face_id);
        });

        auto provider = std::make_unique<MaterialStateProvider>(meshAccess().numCells(),
                                                                std::move(boundary_faces),
                                                                std::move(interior_faces));
        bool any = false;

        const auto register_cell_material_kernel =
            [&](const assembly::AssemblyKernel& kernel, FieldId test_field_id) {
                const auto spec = kernel.materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0");

                const auto& test_field = field_registry_.get(test_field_id);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: test_field.space");
                const auto max_qpts = maxCellQuadraturePoints(meshAccess(), *test_field.space);
                FE_THROW_IF(max_qpts <= 0, InvalidStateException,
                            "FESystem::setup: failed to determine max quadrature points for MaterialState allocation");

                provider->addKernel(kernel, spec, max_qpts);
                any = true;
            };

        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (!term.kernel) continue;

                if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MonolithicCell) {
                    const auto* monolithic =
                        dynamic_cast<const forms::MonolithicCellKernel*>(term.kernel.get());
                    FE_CHECK_NOT_NULL(monolithic, "FESystem::setup: monolithic cell kernel");
                    for (std::size_t bi = 0; bi < monolithic->numBlocks(); ++bi) {
                        const auto& bs = monolithic->blockSpec(bi);
                        if (!bs.fallback_kernel) {
                            continue;
                        }
                        const auto required = bs.fallback_kernel->getRequiredData();
                        if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                            continue;
                        }
                        register_cell_material_kernel(*bs.fallback_kernel, bs.test_field);
                    }
                    continue;
                }

                if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MixedBlockSet) {
                    const auto* mixed =
                        dynamic_cast<const forms::MixedBlockKernelSet*>(term.kernel.get());
                    FE_CHECK_NOT_NULL(mixed, "FESystem::setup: mixed block cell kernel");
                    for (std::size_t bi = 0; bi < mixed->numBlocks(); ++bi) {
                        const auto& bs = mixed->blockSpec(bi);
                        if (!bs.fallback_kernel) {
                            continue;
                        }
                        const auto required = bs.fallback_kernel->getRequiredData();
                        if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                            continue;
                        }
                        register_cell_material_kernel(*bs.fallback_kernel, bs.test_field);
                    }
                    continue;
                }

                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                register_cell_material_kernel(*term.kernel, term.test_field);
            }

            for (const auto& term : def.boundary) {
                if (!term.kernel) continue;
                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0 (boundary)");

                const auto& test_field = field_registry_.get(term.test_field);
                const auto& trial_field = field_registry_.get(term.trial_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: boundary test_field.space");
                FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::setup: boundary trial_field.space");

                const auto max_qpts = maxBoundaryFaceQuadraturePoints(meshAccess(), *test_field.space, *trial_field.space);
                if (max_qpts > 0) {
                    provider->addKernel(*term.kernel, spec,
                                        /*max_cell_qpts=*/0,
                                        /*max_boundary_face_qpts=*/max_qpts,
                                        /*max_interior_face_qpts=*/0);
                    any = true;
                }
            }
            for (const auto& term : def.interior) {
                if (!term.kernel) continue;
                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0 (interior)");

                const auto& test_field = field_registry_.get(term.test_field);
                const auto& trial_field = field_registry_.get(term.trial_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: interior test_field.space");
                FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::setup: interior trial_field.space");

                const auto max_qpts = maxInteriorFaceQuadraturePoints(meshAccess(), *test_field.space, *trial_field.space);
                if (max_qpts > 0) {
                    provider->addKernel(*term.kernel, spec,
                                        /*max_cell_qpts=*/0,
                                        /*max_boundary_face_qpts=*/0,
                                        /*max_interior_face_qpts=*/max_qpts);
                    any = true;
                }
            }
        }

        if (any) {
            material_state_provider_ = std::move(provider);
        }
    }

    // ---------------------------------------------------------------------
    // Global-kernel persistent state storage (optional; for GlobalStateSpec)
    // ---------------------------------------------------------------------
    const auto requires_global_kernel_state_storage = [&]() {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);
            for (const auto& kernel : def.global) {
                if (kernel && !kernel->globalStateSpec().empty()) {
                    return true;
                }
            }
        }
        return false;
    };

    global_kernel_state_provider_.reset();
    if (requires_global_kernel_state_storage()) {
        std::vector<GlobalIndex> boundary_faces;
        meshAccess().forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
            boundary_faces.push_back(face_id);
        });

        std::vector<GlobalIndex> interior_faces;
        meshAccess().forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex /*cell_minus*/, GlobalIndex /*cell_plus*/) {
            interior_faces.push_back(face_id);
        });

        auto provider = std::make_unique<GlobalKernelStateProvider>(meshAccess().numCells(),
                                                                    std::move(boundary_faces),
                                                                    std::move(interior_faces));

        bool any = false;
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);
            for (const auto& kernel : def.global) {
                if (!kernel) continue;
                const auto spec = kernel->globalStateSpec();
                if (spec.empty()) continue;

                provider->addKernel(*kernel, spec);
                any = true;
            }
        }

        if (any) {
            global_kernel_state_provider_ = std::move(provider);
        }
    }

    // ---------------------------------------------------------------------
    // Parameter requirements (optional)
    // ---------------------------------------------------------------------
    parameter_registry_.clear();
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& term : def.boundary) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& term : def.interior) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& term : def.interface_faces) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& kernel : def.global) {
                if (!kernel) continue;
                parameter_registry_.addAll(kernel->parameterSpecs(), kernel->name());
            }
        }

	    }

    // ---------------------------------------------------------------------
    // Resolve FE/Forms constitutive calls for JIT-fast mode (setup-time)
    // ---------------------------------------------------------------------
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
            for (const auto& term : def.boundary) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
            for (const auto& term : def.interior) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
            for (const auto& term : def.interface_faces) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Resolve FE/Forms ParameterSymbol -> ParameterRef(slot) (setup-time)
    // ---------------------------------------------------------------------
    {
        const auto slot_of_real_param = [&](std::string_view key) -> std::optional<std::uint32_t> {
            return parameter_registry_.slotOf(key);
        };

        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.boundary) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.interior) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.interface_faces) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
        }

	    }

    // ---------------------------------------------------------------------
    // Assembler configuration
    // ---------------------------------------------------------------------
    assembly::FormCharacteristics form_chars{};
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            form_chars.has_cell_terms = form_chars.has_cell_terms || !def.cells.empty();
            form_chars.has_boundary_terms = form_chars.has_boundary_terms || !def.boundary.empty();
            form_chars.has_interior_face_terms = form_chars.has_interior_face_terms || !def.interior.empty();
            form_chars.has_interface_face_terms = form_chars.has_interface_face_terms || !def.interface_faces.empty();
            form_chars.has_global_terms = form_chars.has_global_terms || !def.global.empty();

            const auto mergeKernelMeta = [&](const std::shared_ptr<assembly::AssemblyKernel>& k) {
                if (!k) return;
                form_chars.required_data |= k->getRequiredData();
                form_chars.max_time_derivative_order =
                    std::max(form_chars.max_time_derivative_order, k->maxTemporalDerivativeOrder());
                form_chars.has_explicit_time_dependency =
                    form_chars.has_explicit_time_dependency || k->hasExplicitTimeDependency();
                form_chars.has_field_requirements = form_chars.has_field_requirements || !k->fieldRequirements().empty();
                form_chars.has_parameter_specs = form_chars.has_parameter_specs || !k->parameterSpecs().empty();
            };

            for (const auto& term : def.cells) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.boundary) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.interior) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.interface_faces) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& k : def.global) {
                if (!k) continue;
                form_chars.has_parameter_specs = form_chars.has_parameter_specs || !k->parameterSpecs().empty();
            }
        }
    }

    auto assembly_options = opts.assembly_options;
    const auto ghost_policy_override = assemblyGhostPolicyOverrideFromEnv();
    if (ghost_policy_override.has_value()) {
        assembly_options.ghost_policy = *ghost_policy_override;
    }
    if (opts.use_backend_row_ownership_for_assembly &&
        dof_permutation_ != nullptr &&
        !dof_permutation_->empty() &&
        !dof_permutation_->owner_rank.empty()) {
        auto perm = dof_permutation_;
        const auto* default_owner_map = &dof_handler_.getDofMap();
        assembly_options.row_owner_rank =
            [perm = std::move(perm), default_owner_map](GlobalIndex fe_dof) -> int {
                if (fe_dof < 0 ||
                    static_cast<std::size_t>(fe_dof) >= perm->forward.size()) {
                    return default_owner_map ? default_owner_map->getDofOwner(fe_dof) : 0;
                }
                const GlobalIndex backend_dof = perm->forward[static_cast<std::size_t>(fe_dof)];
                if (backend_dof < 0 ||
                    static_cast<std::size_t>(backend_dof) >= perm->owner_rank.size()) {
                    return default_owner_map ? default_owner_map->getDofOwner(fe_dof) : 0;
                }
                const int owner = perm->owner_rank[static_cast<std::size_t>(backend_dof)];
                return owner >= 0 ? owner
                                  : (default_owner_map ? default_owner_map->getDofOwner(fe_dof) : 0);
            };
        if (!ghost_policy_override.has_value()) {
            // Backend row ownership defines the target owner of each FE row,
            // but numeric element assembly still has to route off-owner rows
            // from owned cells to those owners. ReverseScatter traverses owned
            // cells only and sends off-owner rows, matching the distributed
            // sparsity construction above without double-counting ghost cells.
            assembly_options.ghost_policy = assembly::GhostPolicy::ReverseScatter;
        }
    }

    assembly::SystemCharacteristics sys_chars{};
    sys_chars.num_fields = field_registry_.size();
    sys_chars.num_cells = meshAccess().numCells();
    sys_chars.dimension = meshAccess().dimension();
    sys_chars.num_dofs_total = dof_handler_.getDofMap().getNumDofs();
    sys_chars.max_dofs_per_cell = dof_handler_.getDofMap().getMaxDofsPerCell();

    for (const auto& rec : field_registry_.records()) {
        if (!rec.space) continue;
        sys_chars.max_polynomial_order = std::max(sys_chars.max_polynomial_order, rec.space->polynomial_order());
    }

    // Resolve thread count for reporting/heuristics (0 means "auto").
    sys_chars.num_threads = assembly_options.num_threads;
    if (sys_chars.num_threads <= 0) {
        const auto hw = std::max(1u, std::thread::hardware_concurrency());
        sys_chars.num_threads = static_cast<int>(hw);
    }

    sys_chars.mpi_world_size = 1;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_size(dof_handler_.mpiComm(), &sys_chars.mpi_world_size);
    }
#endif

    assembler_selection_report_.clear();
    assembler_ = assembly::createAssembler(assembly_options, opts.assembler_name,
                                           form_chars, sys_chars, &assembler_selection_report_);
    FE_CHECK_NOT_NULL(assembler_.get(), "FESystem::setup: assembler");
    assembler_->setDofHandler(dof_handler_);
    use_constraints_in_assembly_ = opts.use_constraints_in_assembly;
    use_backend_row_ownership_for_assembly_ =
        opts.use_backend_row_ownership_for_assembly;

    if (opts.use_constraints_in_assembly) {
        assembler_->setConstraints(&affine_constraints_);
    } else {
        assembler_->setConstraints(nullptr);
    }

    assembler_->setMaterialStateProvider(material_state_provider_.get());
    assembler_->setOptions(assembly_options);
    assembler_->initialize();

    // ---------------------------------------------------------------------
    // Optional: Auto-register matrix-free operators (explicit opt-in)
    // ---------------------------------------------------------------------
    if (opts.auto_register_matrix_free) {
        FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::setup: operator_backends");
        FE_THROW_IF(field_registry_.size() != 1u, NotImplementedException,
                    "FESystem::setup: auto_register_matrix_free currently requires a single-field system");

        std::size_t registered = 0;
        for (const auto& tag : operator_registry_.list()) {
            if (operator_backends_->hasMatrixFree(tag)) {
                continue; // Respect explicit user registration.
            }

            const auto& def = operator_registry_.get(tag);
            if (def.cells.empty()) continue;
            if (!def.boundary.empty() || !def.interior.empty() || !def.global.empty()) {
                continue; // Cell-only operators only (initial conservative scope).
            }

            // Build a (possibly composite) cell kernel for this operator.
            std::shared_ptr<assembly::AssemblyKernel> kernel_to_wrap;
            if (def.cells.size() == 1u) {
                kernel_to_wrap = def.cells.front().kernel;
            } else {
                auto composite = std::make_shared<assembly::CompositeKernel>();
                for (const auto& term : def.cells) {
                    if (!term.kernel) continue;
                    composite->addKernel(term.kernel);
                }
                kernel_to_wrap = std::move(composite);
            }

            if (!kernel_to_wrap) continue;

            // Conservative eligibility: linear, steady, cell-only.
            const auto required = kernel_to_wrap->getRequiredData();
            if (assembly::hasFlag(required, assembly::RequiredData::SolutionCoefficients) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionValues) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionGradients) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionHessians) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionLaplacians) ||
                assembly::hasFlag(required, assembly::RequiredData::MaterialState) ||
                kernel_to_wrap->hasExplicitTimeDependency() ||
                kernel_to_wrap->maxTemporalDerivativeOrder() > 0) {
                continue;
            }

            auto mf_unique = assembly::wrapAsMatrixFreeKernel(kernel_to_wrap);
            std::shared_ptr<assembly::IMatrixFreeKernel> mf_kernel = std::move(mf_unique);
            operator_backends_->registerMatrixFree(tag, std::move(mf_kernel));
            ++registered;
        }

        if (registered > 0u) {
            if (!assembler_selection_report_.empty()) {
                assembler_selection_report_.append("\n");
            }
            assembler_selection_report_.append("Auto-registered matrix-free operators: ");
            assembler_selection_report_.append(std::to_string(registered));
        }
    }

    // ---------------------------------------------------------------------
    // Setup-time JIT priming for cell kernels.
    //
    // Generic residual/tangent kernels are compiled on first use, and the
    // size-specialized cell variants are normally compiled on the first batch
    // that reaches a given (n_qpts, n_test_dofs, n_trial_dofs) signature.
    // Prime those variants here so the transient loop does not pay that cold
    // compile cost on its first nonlinear assemblies.
    // ---------------------------------------------------------------------
    {
        std::vector<std::pair<ElementType, GlobalIndex>> cell_exemplars;
        const auto collect_cell_exemplar = [&](GlobalIndex cell_id) {
            const auto cell_type = meshAccess().getCellType(cell_id);
            const auto it = std::find_if(
                cell_exemplars.begin(), cell_exemplars.end(),
                [&](const auto& entry) { return entry.first == cell_type; });
            if (it == cell_exemplars.end()) {
                cell_exemplars.emplace_back(cell_type, cell_id);
            }
        };

        meshAccess().forEachOwnedCell(collect_cell_exemplar);
        if (cell_exemplars.empty()) {
            meshAccess().forEachCell(collect_cell_exemplar);
        }

        std::vector<std::tuple<int, ElementType, ElementType, GlobalIndex, GlobalIndex>> boundary_exemplars;
        meshAccess().forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex cell_id) {
            const int marker = meshAccess().getBoundaryFaceMarker(face_id);
            if (marker < 0) {
                return;
            }

            const auto cell_type = meshAccess().getCellType(cell_id);
            const auto face_type = faceTypeForFace(meshAccess(), face_id, cell_id);
            const auto it = std::find_if(
                boundary_exemplars.begin(), boundary_exemplars.end(),
                [&](const auto& exemplar) {
                    return std::get<0>(exemplar) == marker &&
                           std::get<1>(exemplar) == cell_type &&
                           std::get<2>(exemplar) == face_type;
                });
            if (it == boundary_exemplars.end()) {
                boundary_exemplars.emplace_back(marker, cell_type, face_type, face_id, cell_id);
            }
        });

        auto resolve_cell_quadrature =
            [&](const spaces::FunctionSpace& qpt_space,
                GlobalIndex cell_id,
                ElementType cell_type) {
                const auto& test_element = qpt_space.getElement(cell_type, cell_id);

                // For P1 Tet4 elements, use position-based Gaussian rules
                // (4 QPs) matching StandardAssembler::resolveQuadratureRule.
                // Tri3 keeps the Duffy rule (4 QPs) for better NS-VMS stability.
                const int basis_order = test_element.polynomial_order();
                if (basis_order <= 1 && cell_type == ElementType::Tetra4) {
                    const auto default_mod = quadrature::QuadratureFactory::default_legacy_modifier(cell_type);
                    return quadrature::QuadratureFactory::create_legacy_compatible(
                        cell_type, default_mod);
                }

                auto quad_rule = test_element.quadrature();
                if (!quad_rule) {
                    const int quad_order = quadrature::QuadratureFactory::recommended_order(
                        basis_order, false);
                    quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
                }
                return quad_rule;
            };

        auto resolve_boundary_quadrature =
            [&](const spaces::FunctionSpace& test_space,
                const spaces::FunctionSpace& trial_space,
                GlobalIndex face_id,
                GlobalIndex cell_id) {
                const auto cell_type = meshAccess().getCellType(cell_id);
                const auto& test_element = test_space.getElement(cell_type, cell_id);
                const auto& trial_element = trial_space.getElement(cell_type, cell_id);

                const int quad_order = quadrature::QuadratureFactory::recommended_order(
                    std::max(test_element.polynomial_order(), trial_element.polynomial_order()), false);

                const auto face_type = faceTypeForFace(meshAccess(), face_id, cell_id);
                return quadrature::QuadratureFactory::create(face_type, quad_order);
            };

        auto build_cell_hints =
            [&](const spaces::FunctionSpace& qpt_space,
                const spaces::FunctionSpace& test_space,
                const spaces::FunctionSpace& trial_space,
                const forms::JITBasisBakeOptions* bake_options) {
                auto mix_double = [](std::uint64_t& h, double value) {
                    std::uint64_t bits = 0;
                    static_assert(sizeof(bits) == sizeof(value));
                    std::memcpy(&bits, &value, sizeof(bits));
                    forms::jit::mixCacheKey(h, bits);
                };

                auto build_baked_side =
                    [&](const spaces::FunctionSpace& space,
                        const elements::Element& element,
                        const quadrature::QuadratureRule& quad_rule,
                        GlobalIndex cell_id,
                        const forms::JITBasisBakeOptions& opt) {
                    forms::jit::JITBakedBasisSide side;
                    const auto& basis = element.basis();
                    if (basis.is_vector_valued()) {
                        return side;
                    }

                    const auto n_qpts = quad_rule.num_points();
                    const auto n_space_dofs = space.dofs_per_element(cell_id);
                    const auto n_basis_dofs = basis.size();
                    if (n_qpts == 0u || n_space_dofs == 0u || n_basis_dofs == 0u) {
                        return side;
                    }
                    if (n_qpts > opt.max_baked_qpts ||
                        n_space_dofs > opt.max_baked_dofs ||
                        n_qpts * n_space_dofs > opt.max_baked_entries) {
                        return side;
                    }
                    if (n_space_dofs != n_basis_dofs) {
                        const bool product_scalar =
                            space.space_type() == spaces::SpaceType::Product &&
                            n_space_dofs % n_basis_dofs == 0u;
                        if (!product_scalar) {
                            return side;
                        }
                    }

                    const auto& entry =
                        basis::BasisCache::instance().get_or_compute(basis, quad_rule,
                                                                      /*gradients=*/true,
                                                                      /*hessians=*/true);
                    if (entry.num_qpts != n_qpts || entry.num_dofs != n_basis_dofs) {
                        return side;
                    }

                    side.enabled = true;
                    side.scalar_basis = true;
                    side.n_qpts = static_cast<std::uint32_t>(n_qpts);
                    side.n_dofs = static_cast<std::uint32_t>(n_space_dofs);
                    side.basis_hash = forms::jit::hashStringForCacheKey(basis.cache_identity());
                    side.quadrature_hash = forms::jit::hashStringForCacheKey(quad_rule.cache_identity());

                    side.scalar_values_qmajor.resize(n_qpts * n_space_dofs);
                    side.ref_gradients_qmajor.resize(n_qpts * n_space_dofs * 3u);
                    side.ref_hessians_qmajor.resize(n_qpts * n_space_dofs * 9u);

                    for (std::size_t q = 0; q < n_qpts; ++q) {
                        for (std::size_t dof = 0; dof < n_space_dofs; ++dof) {
                            const std::size_t scalar_dof = dof % n_basis_dofs;
                            const std::size_t qmajor = q * n_space_dofs + dof;
                            side.scalar_values_qmajor[qmajor] = entry.scalarValue(scalar_dof, q);

                            const auto& g = entry.gradients[q][scalar_dof];
                            for (std::size_t d = 0; d < 3u; ++d) {
                                side.ref_gradients_qmajor[qmajor * 3u + d] = g[d];
                            }

                            const auto& H = entry.hessians[q][scalar_dof];
                            for (std::size_t r = 0; r < 3u; ++r) {
                                for (std::size_t c = 0; c < 3u; ++c) {
                                    side.ref_hessians_qmajor[qmajor * 9u + r * 3u + c] = H(r, c);
                                }
                            }
                        }
                    }

                    side.ref_gradients_qp_constant = true;
                    for (std::size_t q = 1; q < n_qpts && side.ref_gradients_qp_constant; ++q) {
                        for (std::size_t dof = 0; dof < n_space_dofs && side.ref_gradients_qp_constant; ++dof) {
                            for (std::size_t d = 0; d < 3u; ++d) {
                                const double first = side.ref_gradients_qmajor[dof * 3u + d];
                                const double current =
                                    side.ref_gradients_qmajor[(q * n_space_dofs + dof) * 3u + d];
                                if (current != first) {
                                    side.ref_gradients_qp_constant = false;
                                    break;
                                }
                            }
                        }
                    }

                    std::uint64_t h = forms::jit::kCacheKeyFNVOffset;
                    forms::jit::mixCacheKey(h, side.basis_hash);
                    forms::jit::mixCacheKey(h, side.quadrature_hash);
                    forms::jit::mixCacheKey(h, side.n_qpts);
                    forms::jit::mixCacheKey(h, side.n_dofs);
                    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(side.ref_gradients_qp_constant ? 1u : 0u));
                    for (const double value : side.scalar_values_qmajor) {
                        mix_double(h, value);
                    }
                    for (const double value : side.ref_gradients_qmajor) {
                        mix_double(h, value);
                    }
                    for (const double value : side.ref_hessians_qmajor) {
                        mix_double(h, value);
                    }
                    side.table_hash = h;
                    return side;
                };

                auto build_baked_spec =
                    [&](const elements::Element& test_element,
                        const elements::Element& trial_element,
                        const quadrature::QuadratureRule& quad_rule,
                        GlobalIndex cell_id) {
                    forms::jit::JITBakedBasisSpec spec;
                    if (bake_options == nullptr || !bake_options->enable) {
                        return spec;
                    }

                    spec.geometry_affine = meshAccess().getCellGeometryOrder(cell_id) <= 1;
                    spec.test = build_baked_side(test_space, test_element, quad_rule, cell_id, *bake_options);
                    spec.trial = build_baked_side(trial_space, trial_element, quad_rule, cell_id, *bake_options);
                    spec.enabled = spec.test.enabled || spec.trial.enabled;
                    if (!spec.enabled) {
                        return spec;
                    }

                    std::uint64_t h = forms::jit::kCacheKeyFNVOffset;
                    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(spec.geometry_affine ? 1u : 0u));
                    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(spec.test.enabled ? 1u : 0u));
                    forms::jit::mixCacheKey(h, spec.test.table_hash);
                    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(spec.trial.enabled ? 1u : 0u));
                    forms::jit::mixCacheKey(h, spec.trial.table_hash);
                    spec.hash = h;
                    return spec;
                };

                std::vector<forms::jit::JITKernelWrapper::CellSpecializationHint> hints;
                hints.reserve(cell_exemplars.size());
                for (const auto& [cell_type, cell_id] : cell_exemplars) {
                    const auto quad_rule = resolve_cell_quadrature(qpt_space, cell_id, cell_type);
                    if (!quad_rule || quad_rule->num_points() == 0u) {
                        continue;
                    }

                    forms::jit::JITKernelWrapper::CellSpecializationHint hint;
                    hint.n_qpts = static_cast<std::uint32_t>(quad_rule->num_points());
                    hint.n_test_dofs = static_cast<std::uint32_t>(test_space.dofs_per_element(cell_id));
                    hint.n_trial_dofs = static_cast<std::uint32_t>(trial_space.dofs_per_element(cell_id));

                    // Detect P1 simplices for QP-constant term hoisting
                    const auto& test_element = test_space.getElement(cell_type, cell_id);
                    const auto& trial_element = trial_space.getElement(cell_type, cell_id);
                    hint.is_affine = (test_element.polynomial_order() <= 1) &&
                        (cell_type == ElementType::Tetra4 || cell_type == ElementType::Triangle3);
                    hint.baked_basis = build_baked_spec(test_element, trial_element, *quad_rule, cell_id);

                    hints.push_back(hint);
                }
                return hints;
            };

        auto build_boundary_hints =
            [&](const spaces::FunctionSpace& test_space,
                const spaces::FunctionSpace& trial_space,
                int boundary_marker) {
                std::vector<forms::jit::JITKernelWrapper::BoundarySpecializationHint> hints;
                hints.reserve(boundary_exemplars.size());
                for (const auto& [marker, cell_type_unused, face_type_unused, face_id, cell_id] : boundary_exemplars) {
                    (void)cell_type_unused;
                    (void)face_type_unused;
                    if (boundary_marker >= 0 && marker != boundary_marker) {
                        continue;
                    }

                    const auto quad_rule = resolve_boundary_quadrature(test_space, trial_space, face_id, cell_id);
                    if (!quad_rule || quad_rule->num_points() == 0u) {
                        continue;
                    }

                    forms::jit::JITKernelWrapper::BoundarySpecializationHint hint;
                    hint.boundary_marker = marker;
                    hint.n_qpts = static_cast<std::uint32_t>(quad_rule->num_points());
                    hint.n_test_dofs = static_cast<std::uint32_t>(test_space.dofs_per_element());
                    hint.n_trial_dofs = static_cast<std::uint32_t>(trial_space.dofs_per_element());
                    hints.push_back(hint);
                }
                return hints;
            };

        std::unordered_set<const assembly::AssemblyKernel*> primed_cell_jit_kernels;
        auto prime_cell_jit_kernel =
            [&](const std::shared_ptr<assembly::AssemblyKernel>& kernel,
                const spaces::FunctionSpace& qpt_space,
                const spaces::FunctionSpace& test_space,
                const spaces::FunctionSpace& trial_space) {
                if (!kernel) {
                    return;
                }

                auto* jit_kernel = dynamic_cast<forms::jit::JITKernelWrapper*>(kernel.get());
                if (!jit_kernel) {
                    return;
                }
                if (!primed_cell_jit_kernels.insert(jit_kernel).second) {
                    return;
                }

                const auto* bake_options =
                    jit_kernel->wantsBasisBakingHints() ? &jit_kernel->basisBakeOptions() : nullptr;
                const auto hints = build_cell_hints(qpt_space, test_space, trial_space, bake_options);
                if (!hints.empty()) {
                    jit_kernel->primeCellSpecializations(hints);
                }
            };

        std::vector<std::pair<const assembly::AssemblyKernel*, int>> primed_boundary_jit_kernels;
        auto prime_boundary_jit_kernel =
            [&](const std::shared_ptr<assembly::AssemblyKernel>& kernel,
                const spaces::FunctionSpace& test_space,
                const spaces::FunctionSpace& trial_space,
                int boundary_marker) {
                if (!kernel) {
                    return;
                }

                auto* jit_kernel = dynamic_cast<forms::jit::JITKernelWrapper*>(kernel.get());
                if (!jit_kernel) {
                    return;
                }
                const auto it = std::find(primed_boundary_jit_kernels.begin(),
                                          primed_boundary_jit_kernels.end(),
                                          std::make_pair(static_cast<const assembly::AssemblyKernel*>(jit_kernel),
                                                         boundary_marker));
                if (it != primed_boundary_jit_kernels.end()) {
                    return;
                }
                primed_boundary_jit_kernels.emplace_back(jit_kernel, boundary_marker);

                const auto hints = build_boundary_hints(test_space, trial_space, boundary_marker);
                if (!hints.empty()) {
                    jit_kernel->primeBoundarySpecializations(hints);
                }
            };

        if (!cell_exemplars.empty()) {
            for (const auto& tag : operator_registry_.list()) {
                const auto& def = operator_registry_.get(tag);
                for (const auto& term : def.cells) {
                    if (!term.kernel) {
                        continue;
                    }

                    if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MonolithicCell) {
                        auto* monolithic = dynamic_cast<forms::MonolithicCellKernel*>(term.kernel.get());
                        FE_CHECK_NOT_NULL(monolithic, "FESystem::setup: monolithic cell kernel");
                        for (std::size_t bi = 0; bi < monolithic->numBlocks(); ++bi) {
                            const auto& bs = monolithic->blockSpec(bi);
                            if (!bs.fallback_kernel) {
                                continue;
                            }

                            const auto& test_field = field_registry_.get(bs.test_field);
                            const auto& trial_field = field_registry_.get(bs.trial_field);
                            FE_CHECK_NOT_NULL(test_field.space.get(),
                                              "FESystem::setup: monolithic priming test space");
                            FE_CHECK_NOT_NULL(trial_field.space.get(),
                                              "FESystem::setup: monolithic priming trial space");

                            prime_cell_jit_kernel(bs.fallback_kernel,
                                                  *test_field.space,
                                                  *test_field.space,
                                                  *trial_field.space);
                        }
                        continue;
                    }

                    if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MixedBlockSet) {
                        auto* mixed_block = dynamic_cast<forms::MixedBlockKernelSet*>(term.kernel.get());
                        FE_CHECK_NOT_NULL(mixed_block, "FESystem::setup: mixed block kernel set");
                        if (mixed_block->numBlocks() == 0u) {
                            continue;
                        }

                        const auto& ref_block = mixed_block->blockSpec(0);
                        const auto& qpt_field = field_registry_.get(ref_block.test_field);
                        FE_CHECK_NOT_NULL(qpt_field.space.get(),
                                          "FESystem::setup: mixed-block priming quadrature space");

                        for (std::size_t bi = 0; bi < mixed_block->numBlocks(); ++bi) {
                            const auto& bs = mixed_block->blockSpec(bi);
                            if (!bs.fallback_kernel) {
                                continue;
                            }

                            const auto& test_field = field_registry_.get(bs.test_field);
                            const auto& trial_field = field_registry_.get(bs.trial_field);
                            FE_CHECK_NOT_NULL(test_field.space.get(),
                                              "FESystem::setup: mixed-block priming test space");
                            FE_CHECK_NOT_NULL(trial_field.space.get(),
                                              "FESystem::setup: mixed-block priming trial space");

                            prime_cell_jit_kernel(bs.fallback_kernel,
                                                  *qpt_field.space,
                                                  *test_field.space,
                                                  *trial_field.space);
                        }

                        mixed_block->primeColocatedTextLayout();

                        continue;
                    }

                    const auto& test_field = field_registry_.get(term.test_field);
                    const auto& trial_field = field_registry_.get(term.trial_field);
                    FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: priming test space");
                    FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::setup: priming trial space");

                    prime_cell_jit_kernel(term.kernel,
                                          *test_field.space,
                                          *test_field.space,
                                          *trial_field.space);
                }

                if (!boundary_exemplars.empty()) {
                    for (const auto& term : def.boundary) {
                        if (!term.kernel) {
                            continue;
                        }

                        const auto& test_field = field_registry_.get(term.test_field);
                        const auto& trial_field = field_registry_.get(term.trial_field);
                        FE_CHECK_NOT_NULL(test_field.space.get(),
                                          "FESystem::setup: boundary priming test space");
                        FE_CHECK_NOT_NULL(trial_field.space.get(),
                                          "FESystem::setup: boundary priming trial space");

                        prime_boundary_jit_kernel(term.kernel,
                                                  *test_field.space,
                                                  *trial_field.space,
                                                  term.marker);
                    }
                }
            }
        }
    }

    buildAssemblyPlans();
    is_setup_ = true;
}

void FESystem::buildAssemblyPlans()
{
    assembly_plan_by_op_.clear();

    for (const auto& tag : operator_registry_.list()) {
        const auto& def = operator_registry_.get(tag);
        auto& plan = assembly_plan_by_op_[tag];

        auto common_participant_scope =
            [&](const FieldRecord& test_field,
                const FieldRecord& trial_field,
                std::string_view context) -> std::string {
            const auto* test_participant = fieldMeshParticipant(test_field.id);
            const auto* trial_participant = fieldMeshParticipant(trial_field.id);
            if (test_participant != nullptr && trial_participant != nullptr) {
                FE_THROW_IF(test_participant->name != trial_participant->name,
                            InvalidArgumentException,
                            "FESystem::buildAssemblyPlans: " + std::string(context) +
                                " term couples fields on different mesh participants");
                return test_participant->name;
            }
            if (test_participant != nullptr) {
                return test_participant->name;
            }
            if (trial_participant != nullptr) {
                return trial_participant->name;
            }
            return {};
        };

        auto merge_participant_scope =
            [](std::string current,
               std::string next,
               std::string_view context) -> std::string {
            if (next.empty()) {
                return current;
            }
            FE_THROW_IF(!current.empty() && current != next,
                        InvalidArgumentException,
                        "FESystem::buildAssemblyPlans: " + std::string(context) +
                            " term references fields on different mesh participants");
            return next;
        };

        auto field_participant_scope =
            [&](FieldId field) -> std::string {
            if (field == INVALID_FIELD_ID || !field_registry_.has(field)) {
                return {};
            }
            const auto* participant = fieldMeshParticipant(field);
            if (participant == nullptr) {
                return {};
            }
            return participant->name;
        };

        auto kernel_participant_scope =
            [&](const assembly::AssemblyKernel& kernel,
                std::string_view context) -> std::string {
            std::string scope;
            for (const auto& req : kernel.fieldRequirements()) {
                scope = merge_participant_scope(
                    std::move(scope),
                    field_participant_scope(req.field),
                    context);
            }
            return scope;
        };

        auto validate_boundary_scope =
            [&](int marker, std::string_view participant_name) {
            if (participant_name.empty()) {
                return;
            }
            meshAccess().forEachBoundaryFace(marker, [&](GlobalIndex, GlobalIndex cell_id) {
                const auto* owner = meshParticipantForCell(cell_id);
                FE_THROW_IF(owner == nullptr || owner->name != participant_name,
                            InvalidArgumentException,
                            "FESystem::buildAssemblyPlans: boundary marker " +
                                std::to_string(marker) +
                                " does not belong to mesh participant '" +
                                std::string(participant_name) + "'");
            });
        };

        plan.cell_terms.reserve(def.cells.size());
        for (const auto& term : def.cells) {
            FE_CHECK_NOT_NULL(term.kernel.get(), "FESystem::buildAssemblyPlans: cell kernel");

            if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MonolithicCell) {
                auto* monolithic = dynamic_cast<forms::MonolithicCellKernel*>(term.kernel.get());
                FE_CHECK_NOT_NULL(monolithic, "FESystem::buildAssemblyPlans: monolithic cell kernel");

                if (!monolithic->isResolved()) {
                    for (std::size_t bi = 0; bi < monolithic->numBlocks(); ++bi) {
                        auto& bs = monolithic->mutableBlockSpec(bi);
                        const auto& b_test_field = field_registry_.get(bs.test_field);
                        const auto& b_trial_field = field_registry_.get(bs.trial_field);
                        FE_CHECK_NOT_NULL(b_test_field.space.get(),
                                          "FESystem::buildAssemblyPlans: monolithic block test space");
                        FE_CHECK_NOT_NULL(b_trial_field.space.get(),
                                          "FESystem::buildAssemblyPlans: monolithic block trial space");

                        const auto b_test_idx = static_cast<std::size_t>(b_test_field.id);
                        const auto b_trial_idx = static_cast<std::size_t>(b_trial_field.id);
                        FE_THROW_IF(b_test_field.id < 0 || b_test_idx >= field_dof_handlers_.size(),
                                    InvalidStateException,
                                    "FESystem::buildAssemblyPlans: invalid monolithic block test field");
                        FE_THROW_IF(b_trial_field.id < 0 || b_trial_idx >= field_dof_handlers_.size(),
                                    InvalidStateException,
                                    "FESystem::buildAssemblyPlans: invalid monolithic block trial field");

                        bs.test_space = b_test_field.space.get();
                        bs.trial_space = b_trial_field.space.get();
                        bs.row_dof_map = &field_dof_handlers_[b_test_idx].getDofMap();
                        bs.col_dof_map = &field_dof_handlers_[b_trial_idx].getDofMap();
                        bs.row_dof_offset = field_dof_offsets_[b_test_idx];
                        bs.col_dof_offset = field_dof_offsets_[b_trial_idx];
                    }
                    monolithic->setResolved();
                    monolithic->ensureCompiled();
                    if (!monolithic->hasCompiledDispatch() &&
                        !monolithic->compileMessage().empty() &&
                        core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
                        core::kernelTraceLog(
                            core::KernelTraceChannel::Selection,
                            "FESystem::buildAssemblyPlans: monolithic JIT fallback: " +
                                monolithic->compileMessage());
                    }
                }

                bool any_active = false;
                bool matrix_capable = false;
                bool vector_capable = false;
                std::string participant_scope;
                for (std::size_t bi = 0; bi < monolithic->numBlocks(); ++bi) {
                    const auto& bs = monolithic->blockSpec(bi);
                    if (!bs.fallback_kernel || !bs.fallback_kernel->hasCell()) {
                        continue;
                    }
                    const auto& b_test_field = field_registry_.get(bs.test_field);
                    const auto& b_trial_field = field_registry_.get(bs.trial_field);
                    const auto block_scope =
                        common_participant_scope(b_test_field, b_trial_field, "monolithic cell");
                    participant_scope = merge_participant_scope(
                        std::move(participant_scope),
                        block_scope,
                        "monolithic cell");
                    participant_scope = merge_participant_scope(
                        std::move(participant_scope),
                        kernel_participant_scope(*bs.fallback_kernel, "monolithic cell"),
                        "monolithic cell");
                    any_active = true;
                    matrix_capable = matrix_capable || bs.want_matrix;
                    vector_capable = vector_capable || bs.want_vector;
                }
                if (!any_active) {
                    continue;
                }

                const auto& first_bs = monolithic->blockSpec(0);
                plan.cell_terms.push_back(PlannedCellTerm{
                    term.test_field,
                    term.trial_field,
                    first_bs.test_space,
                    first_bs.trial_space,
                    monolithic,
                    first_bs.row_dof_map,
                    first_bs.col_dof_map,
                    first_bs.row_dof_offset,
                    first_bs.col_dof_offset,
                    participant_scope,
                    assembly::SemanticKernelKind::MonolithicCell,
                    matrix_capable,
                    vector_capable});
                continue;
            }

            if (term.kernel->semanticKernelKind() == assembly::SemanticKernelKind::MixedBlockSet) {
                auto* mixed_block = dynamic_cast<forms::MixedBlockKernelSet*>(term.kernel.get());
                FE_CHECK_NOT_NULL(mixed_block, "FESystem::buildAssemblyPlans: mixed block kernel set");
                if (!mixed_block->isResolved()) {
                    for (std::size_t bi = 0; bi < mixed_block->numBlocks(); ++bi) {
                        auto& bs = mixed_block->mutableBlockSpec(bi);
                        const auto& b_test_field = field_registry_.get(bs.test_field);
                        const auto& b_trial_field = field_registry_.get(bs.trial_field);
                        FE_CHECK_NOT_NULL(b_test_field.space.get(),
                                          "FESystem::buildAssemblyPlans: mixed-block test space");
                        FE_CHECK_NOT_NULL(b_trial_field.space.get(),
                                          "FESystem::buildAssemblyPlans: mixed-block trial space");

                        const auto b_test_idx = static_cast<std::size_t>(b_test_field.id);
                        const auto b_trial_idx = static_cast<std::size_t>(b_trial_field.id);
                        FE_THROW_IF(b_test_field.id < 0 || b_test_idx >= field_dof_handlers_.size(),
                                    InvalidStateException,
                                    "FESystem::buildAssemblyPlans: invalid mixed-block test field");
                        FE_THROW_IF(b_trial_field.id < 0 || b_trial_idx >= field_dof_handlers_.size(),
                                    InvalidStateException,
                                    "FESystem::buildAssemblyPlans: invalid mixed-block trial field");

                        bs.test_space = b_test_field.space.get();
                        bs.trial_space = b_trial_field.space.get();
                        bs.row_dof_map = &field_dof_handlers_[b_test_idx].getDofMap();
                        bs.col_dof_map = &field_dof_handlers_[b_trial_idx].getDofMap();
                        bs.row_dof_offset = field_dof_offsets_[b_test_idx];
                        bs.col_dof_offset = field_dof_offsets_[b_trial_idx];
                    }
                    mixed_block->setResolved();
                }

                bool any_active = false;
                bool matrix_capable = false;
                bool vector_capable = false;
                std::string participant_scope;
                for (std::size_t bi = 0; bi < mixed_block->numBlocks(); ++bi) {
                    const auto& bs = mixed_block->blockSpec(bi);
                    if (!bs.fallback_kernel || !bs.fallback_kernel->hasCell()) {
                        continue;
                    }
                    const auto& b_test_field = field_registry_.get(bs.test_field);
                    const auto& b_trial_field = field_registry_.get(bs.trial_field);
                    const auto block_scope =
                        common_participant_scope(b_test_field, b_trial_field, "mixed-block cell");
                    participant_scope = merge_participant_scope(
                        std::move(participant_scope),
                        block_scope,
                        "mixed-block cell");
                    participant_scope = merge_participant_scope(
                        std::move(participant_scope),
                        kernel_participant_scope(*bs.fallback_kernel, "mixed-block cell"),
                        "mixed-block cell");
                    any_active = true;
                    matrix_capable = matrix_capable || bs.want_matrix;
                    vector_capable = vector_capable || bs.want_vector;
                }
                if (!any_active) {
                    continue;
                }

                const auto& first_bs = mixed_block->blockSpec(0);
                plan.cell_terms.push_back(PlannedCellTerm{
                    term.test_field,
                    term.trial_field,
                    first_bs.test_space,
                    first_bs.trial_space,
                    mixed_block,
                    first_bs.row_dof_map,
                    first_bs.col_dof_map,
                    first_bs.row_dof_offset,
                    first_bs.col_dof_offset,
                    participant_scope,
                    assembly::SemanticKernelKind::MixedBlockSet,
                    matrix_capable,
                    vector_capable});
                continue;
            }

            const auto& test_field = field_registry_.get(term.test_field);
            const auto& trial_field = field_registry_.get(term.trial_field);
            FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::buildAssemblyPlans: cell test space");
            FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::buildAssemblyPlans: cell trial space");

            const auto test_idx = static_cast<std::size_t>(test_field.id);
            const auto trial_idx = static_cast<std::size_t>(trial_field.id);
            auto participant_scope =
                common_participant_scope(test_field, trial_field, "cell");
            participant_scope = merge_participant_scope(
                std::move(participant_scope),
                kernel_participant_scope(*term.kernel, "cell"),
                "cell");
            FE_THROW_IF(test_field.id < 0 || test_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid cell test field");
            FE_THROW_IF(trial_field.id < 0 || trial_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid cell trial field");

            plan.cell_terms.push_back(PlannedCellTerm{
                term.test_field,
                term.trial_field,
                test_field.space.get(),
                trial_field.space.get(),
                term.kernel.get(),
                &field_dof_handlers_[test_idx].getDofMap(),
                &field_dof_handlers_[trial_idx].getDofMap(),
                field_dof_offsets_[test_idx],
                field_dof_offsets_[trial_idx],
                participant_scope,
                term.kernel->semanticKernelKind(),
                !term.kernel->isVectorOnly(),
                !term.kernel->isMatrixOnly()});
        }

        plan.boundary_terms.reserve(def.boundary.size());
        for (const auto& term : def.boundary) {
            FE_CHECK_NOT_NULL(term.kernel.get(), "FESystem::buildAssemblyPlans: boundary kernel");
            const auto& test_field = field_registry_.get(term.test_field);
            const auto& trial_field = field_registry_.get(term.trial_field);
            FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::buildAssemblyPlans: boundary test space");
            FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::buildAssemblyPlans: boundary trial space");

            const auto test_idx = static_cast<std::size_t>(test_field.id);
            const auto trial_idx = static_cast<std::size_t>(trial_field.id);
            auto participant_scope =
                common_participant_scope(test_field, trial_field, "boundary");
            participant_scope = merge_participant_scope(
                std::move(participant_scope),
                kernel_participant_scope(*term.kernel, "boundary"),
                "boundary");
            validate_boundary_scope(term.marker, participant_scope);
            FE_THROW_IF(test_field.id < 0 || test_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid boundary test field");
            FE_THROW_IF(trial_field.id < 0 || trial_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid boundary trial field");

            plan.boundary_terms.push_back(PlannedBoundaryTerm{
                term.marker,
                term.test_field,
                term.trial_field,
                test_field.space.get(),
                trial_field.space.get(),
                term.kernel.get(),
                &field_dof_handlers_[test_idx].getDofMap(),
                &field_dof_handlers_[trial_idx].getDofMap(),
                field_dof_offsets_[test_idx],
                field_dof_offsets_[trial_idx],
                participant_scope,
                !term.kernel->isVectorOnly(),
                !term.kernel->isMatrixOnly()});
        }

        plan.interior_terms.reserve(def.interior.size());
        for (const auto& term : def.interior) {
            FE_CHECK_NOT_NULL(term.kernel.get(), "FESystem::buildAssemblyPlans: interior kernel");
            const auto& test_field = field_registry_.get(term.test_field);
            const auto& trial_field = field_registry_.get(term.trial_field);
            FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::buildAssemblyPlans: interior test space");
            FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::buildAssemblyPlans: interior trial space");

            const auto test_idx = static_cast<std::size_t>(test_field.id);
            const auto trial_idx = static_cast<std::size_t>(trial_field.id);
            FE_THROW_IF(test_field.id < 0 || test_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid interior test field");
            FE_THROW_IF(trial_field.id < 0 || trial_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid interior trial field");

            plan.interior_terms.push_back(PlannedInteriorFaceTerm{
                term.test_field,
                term.trial_field,
                test_field.space.get(),
                trial_field.space.get(),
                term.kernel.get(),
                &field_dof_handlers_[test_idx].getDofMap(),
                &field_dof_handlers_[trial_idx].getDofMap(),
                field_dof_offsets_[test_idx],
                field_dof_offsets_[trial_idx],
                !term.kernel->isVectorOnly(),
                !term.kernel->isMatrixOnly()});
        }

        plan.interface_terms.reserve(def.interface_faces.size());
        for (const auto& term : def.interface_faces) {
            FE_CHECK_NOT_NULL(term.kernel.get(), "FESystem::buildAssemblyPlans: interface kernel");
            const auto& test_field = field_registry_.get(term.test_field);
            const auto& trial_field = field_registry_.get(term.trial_field);
            FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::buildAssemblyPlans: interface test space");
            FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::buildAssemblyPlans: interface trial space");

            const auto test_idx = static_cast<std::size_t>(test_field.id);
            const auto trial_idx = static_cast<std::size_t>(trial_field.id);
            FE_THROW_IF(test_field.id < 0 || test_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid interface test field");
            FE_THROW_IF(trial_field.id < 0 || trial_idx >= field_dof_handlers_.size(),
                        InvalidStateException,
                        "FESystem::buildAssemblyPlans: invalid interface trial field");

            plan.interface_terms.push_back(PlannedInterfaceFaceTerm{
                term.marker,
                term.test_field,
                term.trial_field,
                test_field.space.get(),
                trial_field.space.get(),
                term.kernel.get(),
                &field_dof_handlers_[test_idx].getDofMap(),
                &field_dof_handlers_[trial_idx].getDofMap(),
                field_dof_offsets_[test_idx],
                field_dof_offsets_[trial_idx],
                !term.kernel->isVectorOnly(),
                !term.kernel->isMatrixOnly()});
        }

        plan.global_terms.reserve(def.global.size());
        for (const auto& kernel : def.global) {
            FE_CHECK_NOT_NULL(kernel.get(), "FESystem::buildAssemblyPlans: global kernel");
            plan.global_terms.push_back(kernel.get());
        }

        bool has_matrix_terms = false;
        bool matrix_state_independent = true;
        const auto consider_matrix_term =
            [&](const assembly::AssemblyKernel* kernel, bool matrix_capable) {
                if (!matrix_capable) {
                    return;
                }
                has_matrix_terms = true;
                matrix_state_independent =
                    matrix_state_independent &&
                    kernel != nullptr &&
                    kernel->hasStateIndependentMatrix();
            };

        for (const auto& term : plan.cell_terms) {
            consider_matrix_term(term.kernel, term.matrix_capable);
        }
        for (const auto& term : plan.boundary_terms) {
            consider_matrix_term(term.kernel, term.matrix_capable);
        }
        for (const auto& term : plan.interior_terms) {
            consider_matrix_term(term.kernel, term.matrix_capable);
        }
        for (const auto& term : plan.interface_terms) {
            consider_matrix_term(term.kernel, term.matrix_capable);
        }

        if (!plan.global_terms.empty()) {
            // Global kernels do not yet expose matrix-independence metadata, so
            // any global contribution keeps the operator on the conservative path.
            has_matrix_terms = true;
            matrix_state_independent = false;
        }

        plan.matrix_state_independent = has_matrix_terms && matrix_state_independent;
        if (core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
            core::kernelTraceLog(
                core::KernelTraceChannel::Selection,
                "FESystem::buildAssemblyPlans: operator '" + tag +
                    "' matrix_state_independent=" +
                    (plan.matrix_state_independent ? "true" : "false"));
        }
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
