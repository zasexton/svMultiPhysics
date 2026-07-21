#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetConservativePhaseState.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Dofs/EntityDofMap.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class TwoRankQuadPhaseMeshAccess final : public FE::assembly::IMeshAccess {
public:
    using FE::assembly::IMeshAccess::getCellCoordinates;

    explicit TwoRankQuadPhaseMeshAccess(int rank,
                                        bool fault_rank_one = false)
        : rank_(rank), fault_rank_one_(fault_rank_one)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 6; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return 2; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] FE::GlobalIndex getCellGlobalId(
        FE::GlobalIndex cell) const override
    {
        return 10 + cell;
    }
    [[nodiscard]] int getCellOwnerRank(
        FE::GlobalIndex cell) const override
    {
        return static_cast<int>(cell);
    }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7u; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11u; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13u; }
    [[nodiscard]] std::uint64_t numberingRevision() const override { return 17u; }

    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override
    {
        return cell == static_cast<FE::GlobalIndex>(rank_);
    }

    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex /*cell*/) const override
    {
        return FE::ElementType::Quad4;
    }

    void getCellNodes(FE::GlobalIndex cell,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& cell_nodes = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(cell_nodes.begin(), cell_nodes.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node) const override
    {
        return coordinates_.at(static_cast<std::size_t>(node));
    }

    void getCellCoordinates(
        FE::GlobalIndex cell,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        coordinates.clear();
        for (const auto node : cells_.at(static_cast<std::size_t>(cell))) {
            coordinates.push_back(getNodeCoordinates(node));
        }
        if (fault_rank_one_ && rank_ == 1 && cell == 1) {
            coordinates.front()[0] =
                std::numeric_limits<FE::Real>::quiet_NaN();
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face*/, FE::GlobalIndex /*cell*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex /*face*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face*/) const override
    {
        return {0, 1};
    }

    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(static_cast<FE::GlobalIndex>(rank_));
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex,
                           FE::GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int rank_{0};
    bool fault_rank_one_{false};
    std::array<std::array<FE::Real, 3>, 6> coordinates_{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{2.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{2.0, 1.0, 0.0}},
    }};
    std::array<std::array<FE::GlobalIndex, 4>, 2> cells_{{
        {{0, 1, 4, 3}},
        {{1, 2, 5, 4}},
    }};
};

[[nodiscard]] FE::systems::SetupInputs phaseSetupInputs()
{
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = 2;
    topology.n_vertices = 6;
    topology.dim = 2;
    topology.cell2vertex_offsets = {0, 4, 8};
    topology.cell2vertex_data = {0, 1, 4, 3, 1, 2, 5, 4};
    topology.vertex_gids = {0, 1, 2, 3, 4, 5};
    topology.cell_gids = {10, 11};
    topology.cell_owner_ranks = {0, 1};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

[[nodiscard]] FE::systems::SetupOptions phaseSetupOptions(
    int rank,
    MPI_Comm communicator)
{
    FE::systems::SetupOptions options;
    options.dof_options.global_numbering =
        FE::dofs::GlobalNumberingMode::GlobalIds;
    options.dof_options.ownership =
        FE::dofs::OwnershipStrategy::LowestRank;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = 2;
    options.dof_options.mpi_comm = communicator;
    return options;
}

[[nodiscard]] FE::GlobalIndex vertexDof(
    const FE::dofs::DofHandler& dofs,
    FE::GlobalIndex vertex)
{
    const auto* entity_map = dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        return -1;
    }
    const auto vertex_dofs = entity_map->getVertexDofs(vertex);
    return vertex_dofs.size() == 1u ? vertex_dofs.front() : -1;
}

[[nodiscard]] FE::Real graphChecksum(
    const level_set::LevelSetP1PhaseTransportGraph& graph)
{
    FE::Real checksum = graph.physical_measure;
    for (std::size_t i = 0; i < graph.lumped_control_volume.size(); ++i) {
        checksum += static_cast<FE::Real>(i + 1u) *
                    graph.lumped_control_volume[i];
        for (std::size_t d = 0; d < 3u; ++d) {
            checksum += static_cast<FE::Real>(7u + 3u * i + d) *
                        graph.boundary_column_sum[i][d];
        }
    }
    for (std::size_t e = 0; e < graph.edges.size(); ++e) {
        const auto& edge = graph.edges[e];
        const FE::Real weight = static_cast<FE::Real>(e + 17u);
        checksum += weight * static_cast<FE::Real>(edge.first_node + 1);
        checksum += (weight + 1.0) *
                    static_cast<FE::Real>(edge.second_node + 1);
        checksum += (weight + 2.0) *
                    static_cast<FE::Real>(edge.owner_rank + 1);
        for (std::size_t d = 0; d < 3u; ++d) {
            checksum += (weight + static_cast<FE::Real>(3u + d)) *
                        edge.first_test_second_gradient[d];
            checksum += (weight + static_cast<FE::Real>(6u + d)) *
                        edge.second_test_first_gradient[d];
        }
    }
    return checksum;
}

TEST(LevelSetConservativePhaseOperatorMPI,
     MergesOwnedCellFragmentsAndTransportsAcrossThePartition)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";

    auto mesh = std::make_shared<TwoRankQuadPhaseMeshAccess>(rank);
    FE::systems::FESystem system(mesh);
    const auto indicator_field = system.addField(FE::systems::FieldSpec{
        .name = "liquid_indicator",
        .space = std::make_shared<FE::spaces::H1Space>(
            FE::ElementType::Quad4, /*order=*/1),
        .components = 1,
    });
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = std::make_shared<FE::spaces::H1Space>(
            FE::ElementType::Quad4, /*order=*/1),
        .components = 1,
    });
    system.setup(phaseSetupOptions(rank, MPI_COMM_WORLD),
                 phaseSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& phi_dofs = system.fieldDofHandler(phi_field);
    const auto phi_offset = static_cast<std::size_t>(
        system.fieldDofOffset(phi_field));
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto dof = vertexDof(phi_dofs, vertex);
        ASSERT_GE(dof, 0);
        solution[phi_offset + static_cast<std::size_t>(dof)] =
            mesh->getNodeCoordinates(vertex)[0] - FE::Real{0.75};
    }
    constexpr int interface_marker = 8124;
    level_set::LevelSetGeneratedInterfaceOptions interface_options;
    interface_options.level_set_field_name = "phi";
    interface_options.domain_id = "distributed_phase_state";
    interface_options.requested_interface_marker = interface_marker;
    interface_options.interface_quadrature_order = 2;
    interface_options.volume_quadrature_order = 2;
    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated = lifecycle.build(
        system, interface_options, solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;
    auto cut_context =
        std::make_shared<FE::assembly::CutIntegrationContext>();
    cut_context->addGeneratedInterfaceDomain(generated.domain);
    system.setCutIntegrationContext(std::move(cut_context));

    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        system, indicator_field);
    ASSERT_TRUE(graph.success) << graph.diagnostic;
    EXPECT_TRUE(graph.distributed);
    EXPECT_TRUE(graph.replicated_sparse_graph);
    EXPECT_TRUE(graph.edge_ownership_satisfied);
    EXPECT_EQ(graph.parallel_rank, rank);
    EXPECT_EQ(graph.parallel_size, 2);
    EXPECT_EQ(graph.cells, 2u);
    EXPECT_EQ(graph.local_owned_cells, 1u);
    EXPECT_EQ(graph.nodes, 6u);
    EXPECT_EQ(graph.edges.size(), 11u);
    EXPECT_NEAR(graph.physical_measure, 2.0, 2.0e-14);
    EXPECT_NEAR(graph.total_lumped_control_volume, 2.0, 2.0e-14);

    level_set::LevelSetP1PhaseProjectionOptions projection_options;
    projection_options.interface_marker = interface_marker;
    const auto projection =
        level_set::projectLevelSetP1PhaseIndicatorFromCutContext(
            system, indicator_field, graph, projection_options);
    ASSERT_TRUE(projection.success) << projection.diagnostic;
    EXPECT_TRUE(projection.phase_bounds_satisfied);
    EXPECT_TRUE(projection.global_measure_closure_satisfied);
    EXPECT_NEAR(projection.retained_liquid_measure, 0.75, 3.0e-14);
    EXPECT_NEAR(projection.projected_liquid_measure, 0.75, 3.0e-14);

    const auto& dofs = system.fieldDofHandler(indicator_field);
    std::vector<FE::Real> indicator = projection.liquid_indicator;
    std::vector<FE::Real> lower(graph.nodes, 0.0);
    std::vector<FE::Real> upper(graph.nodes, 1.0);
    std::vector<std::array<FE::Real, 3>> velocity(graph.nodes);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto dof = vertexDof(dofs, vertex);
        ASSERT_GE(dof, 0);
        const auto index = static_cast<std::size_t>(dof);
        const auto point = mesh->getNodeCoordinates(vertex);
        velocity[index] = {
            FE::Real{0.8} + FE::Real{0.1} * point[0],
            FE::Real{-0.2} + FE::Real{0.05} * point[1],
            FE::Real{0.0}};

        const FE::Real expected_mass =
            point[0] == FE::Real{1.0} ? FE::Real{0.5}
                                      : FE::Real{0.25};
        EXPECT_NEAR(graph.lumped_control_volume[index], expected_mass,
                    2.0e-14);
        if (point[0] == FE::Real{1.0}) {
            EXPECT_NEAR(graph.boundary_column_sum[index][0], 0.0,
                        2.0e-14);
        }
    }

    std::size_t cross_owner_edges = 0u;
    for (const auto& edge : graph.edges) {
        EXPECT_GE(edge.owner_rank, 0);
        EXPECT_LT(edge.owner_rank, 2);
        const int first_owner = dofs.getDofMap().getDofOwner(edge.first_node);
        const int second_owner = dofs.getDofMap().getDofOwner(edge.second_node);
        EXPECT_EQ(edge.owner_rank, std::min(first_owner, second_owner));
        if (first_owner != second_owner) {
            ++cross_owner_edges;
        }
    }
    EXPECT_GT(cross_owner_edges, 0u);
    const auto local_owned_edges =
        static_cast<unsigned long long>(graph.locally_owned_edges);
    unsigned long long global_owned_edges = 0u;
    MPI_Allreduce(&local_owned_edges, &global_owned_edges, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(global_owned_edges,
              static_cast<unsigned long long>(graph.edges.size()));

    const FE::Real local_checksum = graphChecksum(graph);
    FE::Real minimum_checksum = 0.0;
    FE::Real maximum_checksum = 0.0;
    MPI_Allreduce(&local_checksum, &minimum_checksum, 1,
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_checksum, &maximum_checksum, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_checksum, maximum_checksum);

    std::vector<FE::Real> minimum_projected(graph.nodes, 0.0);
    std::vector<FE::Real> maximum_projected(graph.nodes, 0.0);
    MPI_Allreduce(indicator.data(), minimum_projected.data(),
                  static_cast<int>(graph.nodes), MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(indicator.data(), maximum_projected.data(),
                  static_cast<int>(graph.nodes), MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_EQ(minimum_projected, maximum_projected);

    const auto stage = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, indicator, lower, upper, velocity, /*time_step=*/0.02);
    ASSERT_TRUE(stage.success) << stage.diagnostic;
    EXPECT_TRUE(stage.courant_satisfied);
    EXPECT_TRUE(stage.strong_form_decomposition_satisfied);
    EXPECT_TRUE(stage.correction.interior_cancellation_satisfied);
    EXPECT_TRUE(stage.correction.local_balance_satisfied);
    EXPECT_TRUE(stage.correction.global_balance_satisfied);
    EXPECT_GE(stage.correction.minimum_limited_liquid_indicator, 0.0);
    EXPECT_LE(stage.correction.maximum_limited_liquid_indicator, 1.0);
    EXPECT_EQ(stage.correction.maximum_edge_pair_cancellation_residual, 0.0);

    std::vector<FE::Real> limited(graph.nodes, 0.0);
    for (std::size_t i = 0; i < graph.nodes; ++i) {
        limited[i] = stage.correction.nodes[i].limited_liquid_indicator;
    }
    std::vector<FE::Real> minimum_limited(graph.nodes, 0.0);
    std::vector<FE::Real> maximum_limited(graph.nodes, 0.0);
    MPI_Allreduce(limited.data(), minimum_limited.data(),
                  static_cast<int>(graph.nodes), MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(limited.data(), maximum_limited.data(),
                  static_cast<int>(graph.nodes), MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    for (std::size_t i = 0; i < graph.nodes; ++i) {
        EXPECT_EQ(minimum_limited[i], maximum_limited[i]);
    }

    std::fill(indicator.begin(), indicator.end(), FE::Real{0.4});
    const auto constant_stage =
        level_set::advanceLevelSetP1ConservativePhaseStage(
            graph, indicator, lower, upper, velocity,
            /*time_step=*/0.02);
    ASSERT_TRUE(constant_stage.success) << constant_stage.diagnostic;
    EXPECT_TRUE(constant_stage.correction.constant_state_input);
    EXPECT_TRUE(constant_stage.correction.constant_preservation_satisfied);
    for (const auto& node : constant_stage.correction.nodes) {
        EXPECT_NEAR(node.limited_liquid_indicator, 0.4, 2.0e-14);
    }
}

TEST(LevelSetConservativePhaseOperatorMPI,
     ReportsTheSameOwnedCellAssemblyFailureOnEveryRank)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";

    auto mesh = std::make_shared<TwoRankQuadPhaseMeshAccess>(
        rank, /*fault_rank_one=*/true);
    FE::systems::FESystem system(mesh);
    const auto indicator_field = system.addField(FE::systems::FieldSpec{
        .name = "liquid_indicator",
        .space = std::make_shared<FE::spaces::H1Space>(
            FE::ElementType::Quad4, /*order=*/1),
        .components = 1,
    });
    system.setup(phaseSetupOptions(rank, MPI_COMM_WORLD),
                 phaseSetupInputs());

    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        system, indicator_field);
    EXPECT_FALSE(graph.success);
    EXPECT_NE(graph.diagnostic.find("rank 1"), std::string::npos);
    EXPECT_NE(graph.diagnostic.find("nonpositive or non-finite"),
              std::string::npos);

    const int local_length = static_cast<int>(graph.diagnostic.size());
    int minimum_length = 0;
    int maximum_length = 0;
    MPI_Allreduce(&local_length, &minimum_length, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_length, &maximum_length, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_length, maximum_length);
}

TEST(LevelSetConservativePhaseOperatorMPI,
     ReportsRankLocalPreflightFailureWithoutStrandingPeers)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";

    auto mesh = std::make_shared<TwoRankQuadPhaseMeshAccess>(rank);
    FE::systems::FESystem system(mesh);
    const auto indicator_field = system.addField(FE::systems::FieldSpec{
        .name = "liquid_indicator",
        .space = std::make_shared<FE::spaces::H1Space>(
            FE::ElementType::Quad4, /*order=*/1),
        .components = 1,
    });
    system.setup(phaseSetupOptions(rank, MPI_COMM_WORLD),
                 phaseSetupInputs());

    level_set::LevelSetP1PhaseGraphOptions graph_options;
    if (rank == 1) {
        graph_options.invariant_tolerance = -1.0;
    }
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        system, indicator_field, graph_options);
    EXPECT_FALSE(graph.success);
    EXPECT_NE(graph.diagnostic.find("rank 1"), std::string::npos);
    EXPECT_NE(graph.diagnostic.find("finite nonnegative tolerance"),
              std::string::npos);

    const int local_length = static_cast<int>(graph.diagnostic.size());
    int minimum_length = 0;
    int maximum_length = 0;
    MPI_Allreduce(&local_length, &minimum_length, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_length, &maximum_length, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_length, maximum_length);
}

} // namespace
