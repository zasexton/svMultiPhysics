/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GeneratedBoundaryAggregateTraceCertificateMPI.cpp
 * @brief Distributed generated-boundary aggregate trace certification.
 */

#include <gtest/gtest.h>

#include "Analysis/GeneratedBoundaryAggregateTraceCertificate.h"
#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/BoundaryConditions.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemAssembly.h"
#include "Systems/SystemSetup.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::analysis::test {
namespace {

constexpr int kInterfaceMarker = 811;
constexpr int kWallMarker = 29;
constexpr int kContactMarker = 812;
constexpr int kActiveBoundaryMarker = 813;
constexpr int kRevisionOnlyMarker = 814;
constexpr std::uint64_t kSourceLayoutRevision = 1u;
constexpr std::uint64_t kSourceValueRevision = 1u;
constexpr std::uint64_t kQuadraturePolicyKey = 815u;
constexpr GlobalIndex kRootCellGid = 10;
constexpr GlobalIndex kCutCellGid = 11;

void installFormBoundTracePolicy(
    systems::FESystem& system,
    FieldId velocity,
    const spaces::FunctionSpace& velocity_space)
{
    const auto u =
        forms::FormExpr::stateField(
            velocity, velocity_space, "u_trace_policy");
    const auto v =
        forms::FormExpr::testFunction(
            velocity, velocity_space, "v_trace_policy");
    std::vector<forms::FormExpr> zero_components(
        static_cast<std::size_t>(
            velocity_space.value_dimension()),
        forms::FormExpr::constant(Real{0.0}));
    const auto zero =
        forms::FormExpr::asVector(
            std::move(zero_components));
    auto terms =
        forms::bc::
            buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                u,
                v,
                zero,
                forms::FormExpr::constant(Real{2.5}),
                kWallMarker,
                kActiveBoundaryMarker,
                forms::bc::TraceNitscheOptions{
                    .gamma = Real{1.0},
                    .variant =
                        forms::bc::NitscheVariant::Symmetric,
                    .scale_with_p = false});
    systems::FormInstallOptions install;
    install.generated_boundary_nitsche_trace_requests.push_back(
        systems::GeneratedBoundaryNitscheTraceInstallRequest{
            .binding = std::move(terms.binding),
            .volume_interface_marker = kInterfaceMarker,
        });
    (void)systems::installFormulation(
        system,
        "velocity",
        {velocity},
        terms.route_contribution,
        install);
}

class ScopedEnvironmentVariable {
public:
    ScopedEnvironmentVariable(const char* key, const char* value)
        : key_(key)
    {
        if (const char* prior = std::getenv(key_)) {
            prior_ = std::string(prior);
        }
        if (value == nullptr) {
            ::unsetenv(key_);
        } else {
            ::setenv(key_, value, 1);
        }
    }

    ~ScopedEnvironmentVariable()
    {
        if (prior_.has_value()) {
            ::setenv(key_, prior_->c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(
        const ScopedEnvironmentVariable&) = delete;

private:
    const char* key_;
    std::optional<std::string> prior_{};
};

class TwoTriangleAggregationMeshAccess final
    : public assembly::IMeshAccess {
public:
    TwoTriangleAggregationMeshAccess(int rank, int size)
        : rank_(rank)
        , size_(size)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 4; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7u; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11u; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 31u; }
    [[nodiscard]] std::uint64_t numberingRevision() const override { return 13u; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17u; }
    [[nodiscard]] std::uint64_t labelRevision() const override { return 19u; }

    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }

    [[nodiscard]] GlobalIndex getCellGlobalId(
        GlobalIndex cell) const override
    {
        return kRootCellGid + cell;
    }

    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(
        GlobalIndex face) const override
    {
        return 100 + face;
    }

    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }

    [[nodiscard]] int getCellOwnerRank(
        GlobalIndex cell) const override
    {
        return static_cast<int>(cell);
    }

    [[nodiscard]] int getBoundaryFaceOwnerRank(
        GlobalIndex,
        GlobalIndex parent_cell) const override
    {
        return getCellOwnerRank(parent_cell);
    }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return getCellOwnerRank(cell) == rank_;
    }

    [[nodiscard]] ElementType getCellType(
        GlobalIndex) const override
    {
        return ElementType::Triangle3;
    }

    void getCellNodes(
        GlobalIndex cell,
        std::vector<GlobalIndex>& nodes) const override
    {
        nodes = cell == 0
                    ? std::vector<GlobalIndex>{0, 1, 2}
                    : std::vector<GlobalIndex>{0, 2, 3};
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
        GlobalIndex face,
        GlobalIndex cell) const override
    {
        for (const auto& boundary : boundary_faces_) {
            if (boundary.face == face &&
                boundary.cell == cell) {
                return boundary.local_face;
            }
        }
        if (face == 4 && cell == 0) {
            return 2;
        }
        if (face == 4 && cell == 1) {
            return 0;
        }
        return INVALID_LOCAL_INDEX;
    }

    [[nodiscard]] int getBoundaryFaceMarker(
        GlobalIndex face) const override
    {
        return face == 2 ? kWallMarker : 0;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face) const override
    {
        return face == 4
                    ? std::pair<GlobalIndex, GlobalIndex>{0, 1}
                    : std::pair<GlobalIndex, GlobalIndex>{
                          INVALID_GLOBAL_INDEX,
                          INVALID_GLOBAL_INDEX};
    }

    void forEachCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        callback(static_cast<GlobalIndex>(rank_));
    }

    void forEachBoundaryFace(
        int marker,
        std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        for (const auto& boundary : boundary_faces_) {
            if (marker < 0 ||
                getBoundaryFaceMarker(boundary.face) == marker) {
                callback(boundary.face, boundary.cell);
            }
        }
    }

    void forEachInteriorFace(
        std::function<void(
            GlobalIndex,
            GlobalIndex,
            GlobalIndex)> callback) const override
    {
        callback(4, 0, 1);
    }

private:
    struct BoundaryFace {
        GlobalIndex face;
        GlobalIndex cell;
        LocalIndex local_face;
    };

    int rank_{0};
    int size_{1};
    std::array<std::array<Real, 3>, 4> coordinates_{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
    }};
    std::array<BoundaryFace, 4> boundary_faces_{{
        {0, 0, 0},
        {1, 0, 1},
        {2, 1, 1},
        {3, 1, 2},
    }};
};

[[nodiscard]] dofs::MeshTopologyInfo twoTriangleTopology(int rank)
{
    dofs::MeshTopologyInfo topology;
    topology.n_cells = 2;
    topology.n_vertices = 4;
    topology.dim = 2;
    topology.cell2vertex_offsets = {0, 3, 6};
    topology.cell2vertex_data = {0, 1, 2, 0, 2, 3};
    topology.vertex_gids = {0, 1, 2, 3};
    topology.cell_gids = {kRootCellGid, kCutCellGid};
    topology.cell_owner_ranks = {0, 1};
    topology.neighbor_ranks = {1 - rank};
    return topology;
}

[[nodiscard]] systems::SetupOptions setupOptions(
    int rank,
    int size,
    MPI_Comm communicator)
{
    systems::SetupOptions options;
    options.dof_options.global_numbering =
        dofs::GlobalNumberingMode::OwnerContiguous;
    options.dof_options.ownership =
        dofs::OwnershipStrategy::CellOwner;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = size;
    options.dof_options.mpi_comm = communicator;
    return options;
}

[[nodiscard]] std::vector<std::uint64_t> allGatherUnsignedValues(
    std::span<const std::uint64_t> local)
{
    const int local_count =
        static_cast<int>(local.size());
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<int> counts(
        static_cast<std::size_t>(size), 0);
    MPI_Allgather(
        &local_count,
        1,
        MPI_INT,
        counts.data(),
        1,
        MPI_INT,
        MPI_COMM_WORLD);

    std::vector<int> displacements(
        static_cast<std::size_t>(size), 0);
    int global_count = 0;
    for (int peer = 0; peer < size; ++peer) {
        displacements[static_cast<std::size_t>(peer)] =
            global_count;
        global_count +=
            counts[static_cast<std::size_t>(peer)];
    }
    std::vector<std::uint64_t> gathered(
        static_cast<std::size_t>(global_count));
    MPI_Allgatherv(
        local.empty() ? nullptr : local.data(),
        local_count,
        MPI_UINT64_T,
        gathered.empty() ? nullptr : gathered.data(),
        counts.data(),
        displacements.data(),
        MPI_UINT64_T,
        MPI_COMM_WORLD);
    return gathered;
}

[[nodiscard]]
interfaces::FreeSurfaceGeometryOwnershipCollective
mpiOwnershipCollective(int rank, int size)
{
    interfaces::FreeSurfaceGeometryOwnershipCollective collective;
    collective.rank = rank;
    collective.size = size;
    collective.all_gather_owned_rule_identity_values =
        allGatherUnsignedValues;
    collective.all_gather_revision_values =
        allGatherUnsignedValues;
    return collective;
}

struct CollectiveOutcome {
    int minimum_threw{0};
    int maximum_threw{0};
    std::string local_message{};

    [[nodiscard]] bool allSucceeded() const noexcept
    {
        return maximum_threw == 0;
    }
};

template <typename Callable>
CollectiveOutcome invokeCollectively(
    MPI_Comm communicator,
    Callable&& callable)
{
    int local_threw = 0;
    std::string message;
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception& error) {
        local_threw = 1;
        message = error.what();
    } catch (...) {
        local_threw = 1;
        message = "non-std exception";
    }

    CollectiveOutcome outcome;
    outcome.local_message = std::move(message);
    MPI_Allreduce(
        &local_threw,
        &outcome.minimum_threw,
        1,
        MPI_INT,
        MPI_MIN,
        communicator);
    MPI_Allreduce(
        &local_threw,
        &outcome.maximum_threw,
        1,
        MPI_INT,
        MPI_MAX,
        communicator);
    return outcome;
}

[[nodiscard]] interfaces::LevelSetInterfaceSource levelSetSource()
{
    return interfaces::LevelSetInterfaceSource::fromEvaluator(
        "aggregate-trace-rooted-mpi",
        kSourceLayoutRevision,
        kSourceValueRevision);
}

template <typename Request>
void setRequestRevisions(
    const assembly::IMeshAccess& mesh,
    Request& request)
{
    request.mesh_geometry_revision =
        mesh.geometryRevision();
    request.mesh_topology_revision =
        mesh.topologyRevision();
    request.ownership_revision =
        mesh.ownershipRevision();
}

[[nodiscard]] interfaces::CutInterfaceDomainRequest
interfaceRequest(const assembly::IMeshAccess& mesh)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = levelSetSource();
    request.interface_marker = kInterfaceMarker;
    request.quadrature_order = 2;
    request.interface_quadrature_order = 2;
    request.volume_quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.implicit_geometry_mode =
        "LinearCorner";
    request.implicit_quadrature_backend =
        "LinearCorner";
    request.implicit_fallback_status = "None";
    setRequestRevisions(mesh, request);
    return request;
}

[[nodiscard]] std::array<Real, 3> levelSetValues(
    GlobalIndex cell)
{
    return cell == 0
               ? std::array<Real, 3>{{-1.0, -1.0, -1.0}}
               : std::array<Real, 3>{{-1.0, -1.0, 7.0}};
}

[[nodiscard]] interfaces::LevelSetInterfaceDomain
rootedInterfaceDomain(const assembly::IMeshAccess& mesh)
{
    const auto request = interfaceRequest(mesh);
    interfaces::LevelSetInterfaceDomain domain(request);
    const std::vector<std::array<Real, 3>> reference_nodes{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
    };
    for (GlobalIndex cell = 0;
         cell < mesh.numCells();
         ++cell) {
        const auto values = levelSetValues(cell);
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = cell;
        input.element_type = ElementType::Triangle3;
        input.node_coordinates = reference_nodes;
        input.level_set_values.assign(
            values.begin(), values.end());
        auto cut =
            interfaces::cutLinearLevelSetCell2D(
                request, input);
        for (auto& fragment : cut.fragments) {
            fragment.parent_cell_global_id =
                mesh.getCellGlobalId(cell);
            fragment.owner_rank =
                mesh.getCellOwnerRank(cell);
            fragment.stable_id = 0u;
            domain.addFragment(std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            region.parent_cell_global_id =
                mesh.getCellGlobalId(cell);
            region.owner_rank =
                mesh.getCellOwnerRank(cell);
            region.stable_id = 0u;
            domain.addVolumeRegion(std::move(region));
        }
    }
    return domain;
}

[[nodiscard]]
interfaces::GeneratedInterfaceBoundaryIntersectionRequest
contactRequest(const assembly::IMeshAccess& mesh)
{
    interfaces::GeneratedInterfaceBoundaryIntersectionRequest
        request;
    request.source = levelSetSource();
    request.generated_domain_id =
        "aggregate-trace-rooted-mpi";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.intersection_marker = kContactMarker;
    request.quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.source_value_revision =
        kSourceValueRevision;
    setRequestRevisions(mesh, request);
    return request;
}

[[nodiscard]] interfaces::GeneratedActiveBoundaryRequest
activeBoundaryRequest(const assembly::IMeshAccess& mesh)
{
    interfaces::GeneratedActiveBoundaryRequest request;
    request.source = levelSetSource();
    request.generated_domain_id =
        "aggregate-trace-rooted-mpi";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.active_boundary_marker =
        kActiveBoundaryMarker;
    request.side =
        geometry::CutIntegrationSide::Negative;
    request.quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.source_value_revision =
        kSourceValueRevision;
    setRequestRevisions(mesh, request);
    return request;
}

[[nodiscard]] Real nodalLevelSetValue(GlobalIndex node)
{
    static constexpr std::array<Real, 4> values{{
        -1.0, -1.0, -1.0, 7.0}};
    return values.at(static_cast<std::size_t>(node));
}

[[nodiscard]]
interfaces::FreeSurfaceGeometryScalarEvaluator
snapshotScalar()
{
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value =
        [](GlobalIndex parent_cell,
           const std::array<Real, 3>& xi,
           const geometry::CutQuadratureProvenance&) {
            const auto values =
                levelSetValues(parent_cell);
            return values[0] *
                       (Real{1.0} - xi[0] - xi[1]) +
                   values[1] * xi[0] +
                   values[2] * xi[1];
        };
    scalar.reference_gradient =
        [](GlobalIndex parent_cell,
           const std::array<Real, 3>&,
           const geometry::CutQuadratureProvenance&) {
            const auto values =
                levelSetValues(parent_cell);
            return std::array<Real, 3>{{
                values[1] - values[0],
                values[2] - values[0],
                Real{0.0},
            }};
        };
    return scalar;
}

[[nodiscard]]
std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
rootedSnapshot(
    const assembly::IMeshAccess& mesh,
    int rank,
    int size)
{
    auto interface_domain =
        rootedInterfaceDomain(mesh);
    auto contact_domain =
        interfaces::
            buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(mesh),
                interface_domain,
                mesh);
    interfaces::GeneratedActiveBoundaryScalarField
        boundary_scalar;
    boundary_scalar.value_at_node =
        [](GlobalIndex node) {
            return nodalLevelSetValue(node);
        };
    auto active_domain =
        interfaces::buildGeneratedActiveBoundaryDomain(
            activeBoundaryRequest(mesh),
            interface_domain,
            contact_domain,
            mesh,
            boundary_scalar);

    std::vector<
        interfaces::
            GeneratedInterfaceBoundaryIntersectionDomain>
        contact_domains;
    contact_domains.push_back(
        std::move(contact_domain));
    std::vector<
        interfaces::GeneratedActiveBoundaryDomain>
        active_domains;
    active_domains.push_back(
        std::move(active_domain));
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition =
        false;
    policy.minimum_retained_volume_fraction =
        assembly::CutIntegrationContext::
            minGeneratedCutVolumeFraction();
    return interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        std::move(contact_domains),
        std::move(active_domains),
        mesh,
        policy,
        snapshotScalar(),
        "aggregate-trace-rooted-mpi",
        mpiOwnershipCollective(rank, size));
}

[[nodiscard]] std::vector<GlobalIndex> vertexDofs(
    const systems::FESystem& system,
    FieldId field,
    GlobalIndex vertex)
{
    const auto* entity_map =
        system.fieldDofHandler(field)
            .getEntityDofMap();
    if (entity_map == nullptr) {
        return {};
    }
    const auto local =
        entity_map->getVertexDofs(vertex);
    std::vector<GlobalIndex> result;
    result.reserve(local.size());
    for (const auto dof : local) {
        result.push_back(
            system.fieldDofOffset(field) + dof);
    }
    return result;
}

TEST(GeneratedBoundaryAggregateTraceCertificateMPI,
     RootedCrossRankAggregateHasAnalyticBoundThirtyTwoOverSeventyNine)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    const ScopedEnvironmentVariable slave_all_cut(
        "SVMP_AGGREGATION_SLAVE_ALL_CUT",
        "0");
    const ScopedEnvironmentVariable linear_extension(
        "SVMP_AGGREGATION_LINEAR_EXTENSION",
        "0");
    const ScopedEnvironmentVariable allow_unaggregated(
        "SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
        "0");
    const ScopedEnvironmentVariable maximum_lines(
        "SVMP_AGGREGATION_MAX_LINES",
        nullptr);

    auto mesh =
        std::make_shared<
            TwoTriangleAggregationMeshAccess>(
                rank, size);
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            2);
    systems::FESystem system(mesh);
    const auto velocity =
        system.addField(
            systems::FieldSpec{
                .name = "velocity",
                .space = velocity_space,
                .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(
        std::make_unique<
            constraints::SmallCutAggregationConstraint>(
            velocity,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    installFormBoundTracePolicy(
        system, velocity, *velocity_space);
    const auto policies =
        system.generatedBoundaryNitscheTracePolicies();
    const int local_binding_ready =
        policies.size() == 1u &&
                policies.front().id !=
                    systems::
                        INVALID_GENERATED_BOUNDARY_NITSCHE_TRACE_POLICY_ID &&
                policies.front().form_binding_digest != 0u &&
                policies.front()
                        .source_formulation_record_index <
                    system.formulationRecords().size()
            ? 1
            : 0;
    int all_bindings_ready = 0;
    MPI_Allreduce(
        &local_binding_ready,
        &all_bindings_ready,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);
    ASSERT_EQ(all_bindings_ready, 1);
    const auto policy_id = policies.front().id;

    systems::SetupInputs inputs;
    inputs.topology_override =
        twoTriangleTopology(rank);
    const auto setup_outcome =
        invokeCollectively(
            MPI_COMM_WORLD,
            [&] {
                system.setup(
                    setupOptions(
                        rank,
                        size,
                        MPI_COMM_WORLD),
                    inputs);
            });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;
    EXPECT_TRUE(
        system.generatedBoundaryNitscheTraceCertificates()
            .empty());

    std::shared_ptr<
        const interfaces::FreeSurfaceGeometrySnapshot>
        snapshot;
    const auto snapshot_outcome =
        invokeCollectively(
            MPI_COMM_WORLD,
            [&] {
                snapshot =
                    rootedSnapshot(
                        system.meshAccess(),
                        rank,
                        size);
            });
    ASSERT_TRUE(snapshot_outcome.allSucceeded())
        << snapshot_outcome.local_message;
    const int local_snapshot_ready =
        snapshot != nullptr ? 1 : 0;
    int all_snapshots_ready = 0;
    MPI_Allreduce(
        &local_snapshot_ready,
        &all_snapshots_ready,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);
    ASSERT_EQ(all_snapshots_ready, 1);

    auto context =
        std::make_shared<
            assembly::CutIntegrationContext>();
    if (rank == 1) {
        context
            ->setExpectedGeneratedSourceValueRevision(
                kRevisionOnlyMarker,
                99u);
    }
    context->addFreeSurfaceGeometrySnapshot(
        snapshot,
        geometry::CutIntegrationSide::Negative);
    system.setCutIntegrationContext(context);

    const auto rebuild_outcome =
        invokeCollectively(
            MPI_COMM_WORLD,
            [&] {
                system.rebuildConstraintState();
            });
    ASSERT_TRUE(rebuild_outcome.allSucceeded())
        << rebuild_outcome.local_message;

    const auto reports =
        system
            .finalizedSmallCutAggregationProlongations();
    const auto* report =
        reports.size() == 1u &&
                reports.front()
            ? reports.front().get()
            : nullptr;

    GeneratedBoundaryAggregateTraceCertificationOptions
        options;
    options.field = velocity;
    options.physical_boundary_marker = kWallMarker;
    options.volume_interface_marker =
        kInterfaceMarker;
    options.generated_active_boundary_marker =
        kActiveBoundaryMarker;
    options.dynamic_viscosity = Real{2.5};

    GeneratedBoundaryAggregateTraceCertificate
        certificate;
    const auto certification_outcome =
        invokeCollectively(
            MPI_COMM_WORLD,
            [&] {
                certificate =
                    certifyGeneratedBoundaryAggregateTrace(
                        system,
                        options);
            });
    ASSERT_TRUE(
        certification_outcome.allSucceeded())
        << certification_outcome.local_message;

    const auto eager =
        system.generatedBoundaryNitscheTraceCertificates();
    const int local_eager_ready =
        eager.size() == 1u &&
                eager.front()
                    .symmetric_energy_ratio_lower_bound
                    .has_value()
            ? 1
            : 0;
    int all_eager_ready = 0;
    MPI_Allreduce(
        &local_eager_ready,
        &all_eager_ready,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);
    ASSERT_EQ(all_eager_ready, 1);
    EXPECT_EQ(eager.front().policy.id, policy_id);
    EXPECT_EQ(
        eager.front().certificate.canonical_certificate_digest,
        certificate.canonical_certificate_digest);
    EXPECT_EQ(
        eager.front().effective_penalty_multiplier,
        Real{1.0});
    EXPECT_LT(
        eager.front()
            .grouped_symmetric_trace_to_penalty_ratio,
        Real{1.0});
    EXPECT_TRUE(
        eager.front()
            .symmetric_energy_ratio_lower_bound
            .has_value());
    EXPECT_GT(
        *eager.front()
             .symmetric_energy_ratio_lower_bound,
        Real{0.0});

    assembly::DenseMatrixView preflight_matrix(
        system.dofHandler().getNumDofs());
    systems::AssemblyRequest assembly_request;
    assembly_request.op = "velocity";
    assembly_request.want_matrix = true;
    systems::SystemStateView state;
    const auto assembly_outcome =
        invokeCollectively(
            MPI_COMM_WORLD,
            [&] {
                (void)systems::assembleOperator(
                    system,
                    assembly_request,
                    state,
                    &preflight_matrix,
                    nullptr);
            });
    ASSERT_TRUE(assembly_outcome.allSucceeded())
        << assembly_outcome.local_message;

    const std::array<std::uint64_t, 4>
        local_report_shape{{
            static_cast<std::uint64_t>(
                reports.size()),
            report != nullptr
                ? static_cast<std::uint64_t>(
                      report->rows.size())
                : 0u,
            report != nullptr
                ? static_cast<std::uint64_t>(
                      report->active_cells.size())
                : 0u,
            report != nullptr
                ? static_cast<std::uint64_t>(
                      report->patches.size())
                : 0u,
        }};
    std::array<std::uint64_t, 4>
        minimum_report_shape{};
    std::array<std::uint64_t, 4>
        maximum_report_shape{};
    MPI_Allreduce(
        local_report_shape.data(),
        minimum_report_shape.data(),
        static_cast<int>(
            local_report_shape.size()),
        MPI_UINT64_T,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        local_report_shape.data(),
        maximum_report_shape.data(),
        static_cast<int>(
            local_report_shape.size()),
        MPI_UINT64_T,
        MPI_MAX,
        MPI_COMM_WORLD);

    const std::uint64_t local_report_digest =
        report != nullptr
            ? report->canonical_content_digest
            : 0u;
    std::uint64_t minimum_report_digest = 0u;
    std::uint64_t maximum_report_digest = 0u;
    MPI_Allreduce(
        &local_report_digest,
        &minimum_report_digest,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_report_digest,
        &maximum_report_digest,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        MPI_COMM_WORLD);

    const std::uint64_t local_certificate_digest =
        certificate.canonical_certificate_digest;
    std::uint64_t minimum_certificate_digest = 0u;
    std::uint64_t maximum_certificate_digest = 0u;
    MPI_Allreduce(
        &local_certificate_digest,
        &minimum_certificate_digest,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_certificate_digest,
        &maximum_certificate_digest,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        MPI_COMM_WORLD);

    const std::uint64_t local_context_revision =
        certificate.cut_context_content_revision;
    std::uint64_t minimum_context_revision = 0u;
    std::uint64_t maximum_context_revision = 0u;
    MPI_Allreduce(
        &local_context_revision,
        &minimum_context_revision,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_context_revision,
        &maximum_context_revision,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        MPI_COMM_WORLD);

    const std::array<std::uint64_t, 4>
        expected_report_shape{{1u, 2u, 2u, 1u}};
    EXPECT_EQ(
        minimum_report_shape,
        maximum_report_shape);
    EXPECT_EQ(
        maximum_report_shape,
        expected_report_shape);
    EXPECT_NE(minimum_report_digest, 0u);
    EXPECT_EQ(
        minimum_report_digest,
        maximum_report_digest);
    EXPECT_NE(minimum_certificate_digest, 0u);
    EXPECT_EQ(
        minimum_certificate_digest,
        maximum_certificate_digest);
    EXPECT_LT(
        minimum_context_revision,
        maximum_context_revision);

    ASSERT_NE(snapshot, nullptr);
    ASSERT_NE(report, nullptr);
    ASSERT_EQ(
        snapshot->activeBoundaryDomains().size(),
        1u);
    const auto& local_wet_fragments =
        snapshot->activeBoundaryDomains()
            .front()
            .fragments();
    if (rank == 0) {
        EXPECT_TRUE(local_wet_fragments.empty());
    } else {
        ASSERT_EQ(local_wet_fragments.size(), 1u);
        EXPECT_EQ(
            local_wet_fragments.front()
                .parent_cell_global_id,
            kCutCellGid);
        EXPECT_EQ(
            local_wet_fragments.front().owner_rank,
            1);
    }
    EXPECT_EQ(report->revision.local_rank, rank);
    EXPECT_EQ(
        report->revision.communicator_size,
        size);
    EXPECT_TRUE(report->trace_bound_eligible);
    EXPECT_EQ(
        report->revision
            .cut_context_content_revision,
        context->contentRevision());
    EXPECT_EQ(
        certificate
            .cut_context_content_revision,
        context->contentRevision());
    EXPECT_EQ(
        certificate
            .cut_context_content_revision,
        report->revision
            .cut_context_content_revision);
    EXPECT_EQ(
        certificate
            .free_surface_snapshot_revision,
        snapshot->revision()
            .snapshot_revision_key);
    EXPECT_EQ(
        certificate
            .free_surface_snapshot_revision,
        report->revision
            .free_surface_snapshot_revision);
    EXPECT_EQ(
        certificate.source_value_revision,
        kSourceValueRevision);
    EXPECT_EQ(
        certificate.source_value_revision,
        report->revision.source_value_revision);
    EXPECT_EQ(
        certificate
            .affine_constraint_layout_revision,
        system.constraints()
            .constraintLayoutRevision());
    EXPECT_EQ(
        certificate
            .affine_constraint_layout_revision,
        report->revision
            .affine_constraint_layout_revision);
    EXPECT_EQ(
        certificate.aggregation_content_digest,
        report->canonical_content_digest);

    ASSERT_EQ(report->active_cells.size(), 2u);
    const auto root_cell =
        std::find_if(
            report->active_cells.begin(),
            report->active_cells.end(),
            [](const auto& cell) {
                return cell.cell_gid ==
                       kRootCellGid;
            });
    const auto cut_cell =
        std::find_if(
            report->active_cells.begin(),
            report->active_cells.end(),
            [](const auto& cell) {
                return cell.cell_gid ==
                       kCutCellGid;
            });
    ASSERT_NE(
        root_cell,
        report->active_cells.end());
    ASSERT_NE(
        cut_cell,
        report->active_cells.end());
    EXPECT_EQ(root_cell->owner_rank, 0);
    EXPECT_EQ(
        root_cell->kind,
        constraints::
            SmallCutAggregationActiveCellKind::
                FullActive);
    EXPECT_EQ(
        root_cell->active_feature_id,
        kRootCellGid);
    EXPECT_NEAR(
        root_cell->retained_physical_volume,
        Real{0.5},
        Real{1.0e-14});
    EXPECT_EQ(
        root_cell
            ->active_face_neighbor_cell_gids,
        (std::vector<GlobalIndex>{
            kCutCellGid}));
    EXPECT_EQ(cut_cell->owner_rank, 1);
    EXPECT_EQ(
        cut_cell->kind,
        constraints::
            SmallCutAggregationActiveCellKind::
                Cut);
    EXPECT_EQ(
        cut_cell->active_feature_id,
        kRootCellGid);
    EXPECT_NEAR(
        cut_cell->retained_physical_volume,
        Real{15.0} / Real{128.0},
        Real{1.0e-14});
    EXPECT_EQ(
        cut_cell
            ->active_face_neighbor_cell_gids,
        (std::vector<GlobalIndex>{
            kRootCellGid}));
    EXPECT_FALSE(
        root_cell
            ->retained_rule_stable_ids.empty());
    EXPECT_FALSE(
        cut_cell
            ->retained_rule_stable_ids.empty());
    EXPECT_EQ(root_cell->field_dofs.size(), 6u);
    EXPECT_EQ(cut_cell->field_dofs.size(), 6u);

    const auto v0 =
        vertexDofs(system, velocity, 0);
    const auto v1 =
        vertexDofs(system, velocity, 1);
    const auto v2 =
        vertexDofs(system, velocity, 2);
    const auto v3 =
        vertexDofs(system, velocity, 3);
    ASSERT_EQ(v0.size(), 2u);
    ASSERT_EQ(v1.size(), 2u);
    ASSERT_EQ(v2.size(), 2u);
    ASSERT_EQ(v3.size(), 2u);
    ASSERT_EQ(report->rows.size(), 2u);
    for (std::size_t component = 0u;
         component < 2u;
         ++component) {
        const auto row =
            std::find_if(
                report->rows.begin(),
                report->rows.end(),
                [&](const auto& candidate) {
                    return candidate.slave_dof ==
                           v3[component];
                });
        ASSERT_NE(row, report->rows.end());
        EXPECT_EQ(row->candidate_dof, v3.front());
        EXPECT_EQ(row->component, component);
        EXPECT_EQ(row->slave_owner_rank, 1);
        EXPECT_EQ(
            row->provisional_kind,
            constraints::
                SmallCutAggregationProvisionalRowKind::
                    RootedExtension);
        EXPECT_EQ(
            row->final_kind,
            constraints::
                SmallCutAggregationFinalRowKind::
                    MasterBearing);
        EXPECT_FALSE(row->preconstrained_at_apply);
        EXPECT_EQ(
            row->root_cell_gid,
            kRootCellGid);
        EXPECT_EQ(row->root_cell_owner_rank, 0);
        EXPECT_EQ(row->root_distance, 1u);
        EXPECT_EQ(row->final_inhomogeneity, Real{0.0});

        std::vector<
            std::pair<GlobalIndex, Real>>
            expected_entries{
                {v0[component], Real{1.0}},
                {v1[component], Real{-1.0}},
                {v2[component], Real{1.0}},
            };
        std::sort(
            expected_entries.begin(),
            expected_entries.end(),
            [](const auto& left,
               const auto& right) {
                return left.first < right.first;
            });
        ASSERT_EQ(
            row->provisional_entries.size(),
            expected_entries.size());
        ASSERT_EQ(
            row->final_entries.size(),
            expected_entries.size());
        for (std::size_t entry = 0u;
             entry < expected_entries.size();
             ++entry) {
            EXPECT_EQ(
                row->provisional_entries[entry]
                    .master_dof,
                expected_entries[entry].first);
            EXPECT_EQ(
                row->provisional_entries[entry]
                    .weight,
                expected_entries[entry].second);
            EXPECT_EQ(
                row->final_entries[entry]
                    .master_dof,
                expected_entries[entry].first);
            EXPECT_EQ(
                row->final_entries[entry]
                    .weight,
                expected_entries[entry].second);
            EXPECT_EQ(
                system.dofHandler()
                    .getDofMap()
                    .getDofOwner(
                        row->final_entries[entry]
                            .master_dof),
                0);
        }
    }

    ASSERT_EQ(report->patches.size(), 1u);
    const auto& aggregate_patch =
        report->patches.front();
    EXPECT_EQ(
        aggregate_patch.kind,
        constraints::
            SmallCutAggregationPatchKind::Rooted);
    EXPECT_EQ(
        aggregate_patch.root_cell_gid,
        kRootCellGid);
    EXPECT_EQ(
        aggregate_patch.root_cell_owner_rank,
        0);
    EXPECT_EQ(
        aggregate_patch.active_feature_ids,
        (std::vector<GlobalIndex>{
            kRootCellGid}));
    EXPECT_EQ(
        aggregate_patch.member_cell_gids,
        (std::vector<GlobalIndex>{
            kRootCellGid,
            kCutCellGid}));
    EXPECT_EQ(
        aggregate_patch.support_cell_gids,
        (std::vector<GlobalIndex>{
            kRootCellGid,
            kCutCellGid}));
    auto expected_slaves = v3;
    std::sort(
        expected_slaves.begin(),
        expected_slaves.end());
    EXPECT_EQ(
        aggregate_patch.slave_dofs,
        expected_slaves);

    const auto refresh_reports =
        system
            .completedSmallCutAggregationRefreshReports();
    ASSERT_EQ(refresh_reports.size(), 1u);
    const auto& refresh = refresh_reports.front();
    EXPECT_EQ(
        refresh.canonical_candidate_vertices,
        1u);
    EXPECT_EQ(
        refresh
            .canonical_rooted_candidate_vertices,
        1u);
    EXPECT_EQ(
        refresh
            .canonical_rootless_candidate_vertices,
        0u);
    EXPECT_EQ(
        refresh.canonical_owned_aggregate_dofs,
        2u);
    EXPECT_EQ(
        refresh.canonical_owned_pinned_dofs,
        0u);
    EXPECT_EQ(
        refresh
            .canonical_strong_suppressed_dofs,
        0u);
    ASSERT_EQ(
        refresh.canonical_active_features.size(),
        1u);
    EXPECT_EQ(
        refresh.canonical_active_features.front()
            .stable_feature_id,
        kRootCellGid);
    EXPECT_EQ(
        refresh.canonical_active_features.front()
            .disposition,
        constraints::
            SmallCutAggregationActiveFeatureDisposition::
                Rooted);
    EXPECT_EQ(
        refresh.canonical_active_features.front()
            .canonical_cell_count,
        2u);
    EXPECT_EQ(
        refresh.canonical_active_features.front()
            .canonical_full_active_cell_count,
        1u);
    EXPECT_EQ(
        refresh.canonical_active_features.front()
            .canonical_cut_cell_count,
        1u);
    EXPECT_NEAR(
        refresh.canonical_active_features.front()
            .canonical_retained_physical_volume,
        Real{79.0} / Real{128.0},
        Real{1.0e-14});

    EXPECT_EQ(certificate.field, velocity);
    EXPECT_EQ(
        certificate.physical_boundary_marker,
        kWallMarker);
    EXPECT_EQ(certificate.communicator_size, 2);
    EXPECT_EQ(certificate.active_cell_count, 2u);
    EXPECT_EQ(
        certificate.generated_boundary_rule_count,
        1u);
    EXPECT_EQ(
        certificate.certified_patch_count,
        1u);
    EXPECT_EQ(
        certificate.maximum_support_overlap,
        1u);
    EXPECT_EQ(
        certificate
            .maximum_terminal_tangent_dimension,
        6u);
    EXPECT_NEAR(
        certificate.retained_active_physical_volume,
        Real{79.0} / Real{128.0},
        Real{1.0e-14});
    EXPECT_NEAR(
        certificate
            .generated_boundary_physical_measure,
        Real{1.0} / Real{8.0},
        Real{1.0e-14});

    ASSERT_EQ(certificate.patches.size(), 1u);
    const auto& patch =
        certificate.patches.front();
    EXPECT_FALSE(
        patch.synthetic_full_active_patch);
    EXPECT_EQ(patch.canonical_patch_index, 0u);
    EXPECT_EQ(
        patch.root_cell_gid,
        kRootCellGid);
    EXPECT_EQ(
        patch.support_cell_gids,
        (std::vector<GlobalIndex>{
            kRootCellGid,
            kCutCellGid}));
    ASSERT_EQ(
        patch.boundary_rule_stable_ids.size(),
        1u);
    EXPECT_NE(
        patch.boundary_rule_stable_ids.front(),
        0u);
    EXPECT_EQ(patch.raw_support_dof_count, 8u);
    EXPECT_EQ(
        patch.terminal_tangent_dof_count,
        6u);
    EXPECT_EQ(
        patch.rigid_mode_candidate_count,
        3u);
    EXPECT_EQ(
        patch.structural_rigid_mode_count,
        3u);
    EXPECT_EQ(
        patch.rigid_mode_constraint_rank,
        0u);
    EXPECT_EQ(
        patch.rigid_mode_quotient_status,
        GeneratedBoundaryRigidModeQuotientStatus::
            Applied);
    EXPECT_EQ(
        patch.maximum_cell_support_overlap,
        1u);
    EXPECT_NEAR(
        patch.retained_support_physical_volume,
        Real{79.0} / Real{128.0},
        Real{1.0e-14});
    EXPECT_NEAR(
        patch.generated_boundary_physical_measure,
        Real{1.0} / Real{8.0},
        Real{1.0e-14});

    const Real analytic_bound =
        Real{32.0} / Real{79.0};
    const auto& bound =
        patch.generalized_bound;
    EXPECT_EQ(bound.dimension, 6u);
    EXPECT_EQ(bound.positive_rank, 3u);
    EXPECT_EQ(bound.nullity, 3u);
    EXPECT_TRUE(bound.denominator_converged);
    EXPECT_TRUE(bound.quotient_converged);
    EXPECT_TRUE(
        bound.explicit_nullspace.applied);
    EXPECT_EQ(
        bound.explicit_nullspace
            .supplied_nullity,
        3u);
    EXPECT_EQ(
        bound.explicit_nullspace
            .reduced_dimension,
        3u);
    EXPECT_TRUE(
        bound.explicit_nullspace
            .exact_binary64_actions_proven);
    EXPECT_TRUE(
        bound.explicit_nullspace
            .exact_binary64_anchor_rank_proven);
    EXPECT_TRUE(bound.exact_dyadic.applied);
    EXPECT_TRUE(
        bound.exact_dyadic
            .denominator_positive_definite_proven);
    EXPECT_TRUE(
        bound.exact_dyadic
            .numerator_positive_semidefinite_proven);
    EXPECT_TRUE(
        bound.exact_dyadic.upper_inequality_proven);
    EXPECT_EQ(bound.exact_dyadic.dimension, 3u);
    EXPECT_EQ(bound.exact_dyadic.denominator_rank, 3u);
    EXPECT_GE(
        bound.conservative_upper_bound,
        bound.exact_dyadic.directly_proven_upper_bound);
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue,
        analytic_bound,
        Real{1.0e-10});
    EXPECT_GE(
        bound.conservative_upper_bound,
        analytic_bound);
    EXPECT_NEAR(
        bound.conservative_upper_bound,
        analytic_bound,
        Real{1.0e-9});
    EXPECT_EQ(
        certificate
            .maximum_patch_conservative_upper_bound,
        bound.conservative_upper_bound);
    EXPECT_GE(
        certificate
            .global_conservative_upper_bound,
        bound.conservative_upper_bound);
    EXPECT_NEAR(
        certificate
            .global_conservative_upper_bound,
        analytic_bound,
        Real{1.0e-9});
#endif
}

} // namespace
} // namespace svmp::FE::analysis::test
