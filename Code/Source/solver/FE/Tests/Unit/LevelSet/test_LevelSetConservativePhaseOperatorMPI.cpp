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
#include <string_view>
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

    const auto sensitivity =
        level_set::buildLevelSetP1PhaseGeometrySensitivity(
            system,
            phi_field,
            indicator_field,
            graph,
            projection_options,
            solution);
    ASSERT_TRUE(sensitivity.success) << sensitivity.diagnostic;
    EXPECT_TRUE(sensitivity.field_layouts_identical);
    EXPECT_TRUE(sensitivity.level_set_null_space_satisfied);
    EXPECT_EQ(sensitivity.owned_rules, 1u);
    EXPECT_EQ(sensitivity.active_nodes, 4u);
    EXPECT_NEAR(sensitivity.interface_measure, 1.0, 3.0e-14);
    EXPECT_NEAR(sensitivity.minimum_level_set_gradient, 1.0, 3.0e-14);
    EXPECT_NEAR(sensitivity.minimum_cell_node_distance, 1.0, 3.0e-14);

    std::vector<FE::Real> sensitivity_values = sensitivity.diagonal;
    for (const auto& edge : sensitivity.edges) {
        sensitivity_values.push_back(edge.coefficient);
    }
    std::vector<FE::Real> minimum_sensitivity(
        sensitivity_values.size(), 0.0);
    std::vector<FE::Real> maximum_sensitivity(
        sensitivity_values.size(), 0.0);
    MPI_Allreduce(sensitivity_values.data(), minimum_sensitivity.data(),
                  static_cast<int>(sensitivity_values.size()), MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(sensitivity_values.data(), maximum_sensitivity.data(),
                  static_cast<int>(sensitivity_values.size()), MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_sensitivity, maximum_sensitivity);

    std::vector<FE::Real> phi(graph.nodes, 0.0);
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        phi[node] = solution[phi_offset + node];
    }
    std::vector<FE::Real> expected_increment(
        graph.nodes, FE::Real{0.0});
    expected_increment[0] = FE::Real{0.01};
    expected_increment[1] = FE::Real{-0.02};
    expected_increment[3] = FE::Real{0.015};
    expected_increment[4] = FE::Real{-0.005};
    FE::Real scaling_projection{0.0};
    FE::Real scaling_norm_squared{0.0};
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        if (sensitivity.diagonal[node] > FE::Real{0.0}) {
            scaling_projection += expected_increment[node] * phi[node];
            scaling_norm_squared += phi[node] * phi[node];
        }
    }
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        if (sensitivity.diagonal[node] > FE::Real{0.0}) {
            expected_increment[node] -=
                scaling_projection / scaling_norm_squared * phi[node];
        }
    }
    std::vector<FE::Real> matrix_increment(
        graph.nodes, FE::Real{0.0});
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        matrix_increment[node] =
            sensitivity.diagonal[node] * expected_increment[node];
    }
    for (const auto& edge : sensitivity.edges) {
        const auto first = static_cast<std::size_t>(edge.first_node);
        const auto second = static_cast<std::size_t>(edge.second_node);
        matrix_increment[first] +=
            edge.coefficient * expected_increment[second];
        matrix_increment[second] +=
            edge.coefficient * expected_increment[first];
    }
    auto target_mass = projection.liquid_phase_mass;
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        target_mass[node] -= matrix_increment[node];
    }
    const auto geometry_correction =
        level_set::solveLevelSetP1PhaseGeometryCorrection(
            sensitivity,
            FE::geometry::CutIntegrationSide::Negative,
            phi,
            projection.liquid_phase_mass,
            target_mass);
    ASSERT_TRUE(geometry_correction.success)
        << geometry_correction.diagnostic;
    EXPECT_TRUE(geometry_correction.linear_solve_converged);
    EXPECT_EQ(geometry_correction.interface_components, 1u);
    ASSERT_EQ(geometry_correction.level_set_increment.size(), graph.nodes);
    ASSERT_EQ(geometry_correction.predicted_liquid_mass_change.size(),
              graph.nodes);
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        EXPECT_NEAR(geometry_correction.predicted_liquid_mass_change[node],
                    target_mass[node] - projection.liquid_phase_mass[node],
                    3.0e-12);
    }
    std::vector<FE::Real> minimum_increment(graph.nodes, 0.0);
    std::vector<FE::Real> maximum_increment(graph.nodes, 0.0);
    MPI_Allreduce(geometry_correction.level_set_increment.data(),
                  minimum_increment.data(), static_cast<int>(graph.nodes),
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(geometry_correction.level_set_increment.data(),
                  maximum_increment.data(), static_cast<int>(graph.nodes),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_increment, maximum_increment);

    const auto& dofs = system.fieldDofHandler(indicator_field);
    std::vector<FE::Real> indicator = projection.liquid_indicator;
    std::vector<FE::Real> lower = indicator;
    std::vector<FE::Real> upper = indicator;
    for (const auto& edge : graph.edges) {
        const auto first = static_cast<std::size_t>(edge.first_node);
        const auto second = static_cast<std::size_t>(edge.second_node);
        lower[first] = std::min(lower[first], indicator[second]);
        lower[second] = std::min(lower[second], indicator[first]);
        upper[first] = std::max(upper[first], indicator[second]);
        upper[second] = std::max(upper[second], indicator[first]);
    }
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        lower[node] = std::clamp(lower[node], FE::Real{0.0}, FE::Real{1.0});
        upper[node] = std::clamp(upper[node], FE::Real{0.0}, FE::Real{1.0});
    }
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
    ASSERT_TRUE(stage.replicated_stage_inputs_satisfied);
    ASSERT_EQ(stage.sampled_nodal_velocity.size(), graph.nodes);
    EXPECT_NE(stage.sampled_nodal_velocity.front()[0], FE::Real{0.0});
    EXPECT_TRUE(stage.courant_satisfied);
    EXPECT_TRUE(stage.strong_form_decomposition_satisfied);
    EXPECT_TRUE(stage.correction.interior_cancellation_satisfied);
    EXPECT_TRUE(stage.correction.local_balance_satisfied);
    EXPECT_TRUE(stage.correction.global_balance_satisfied);
    EXPECT_TRUE(stage.correction.component_balance_satisfied);
    EXPECT_TRUE(stage.correction.component_measure_closure_satisfied);
    EXPECT_GE(stage.correction.minimum_limited_liquid_indicator, 0.0);
    EXPECT_LE(stage.correction.maximum_limited_liquid_indicator, 1.0);
    EXPECT_EQ(stage.correction.maximum_edge_pair_cancellation_residual, 0.0);
    ASSERT_EQ(stage.correction.node_component_ids.size(), graph.nodes);
    ASSERT_FALSE(stage.correction.components.empty());

    // Local mesh cache stamps may legitimately differ across partitions. The
    // fixed-background contract compares them from q^n to the operator stage
    // on each rank, while the execution-layout-sensitive graph content/layout
    // remains replicated within this communicator.
    auto validation_graph = graph;
    validation_graph.geometry_revision +=
        static_cast<std::uint64_t>(rank);
    validation_graph.topology_revision +=
        static_cast<std::uint64_t>(2 * rank);
    validation_graph.ownership_revision +=
        static_cast<std::uint64_t>(3 * rank);
    validation_graph.numbering_revision +=
        static_cast<std::uint64_t>(4 * rank);
    const auto graph_identity =
        level_set::levelSetP1PhaseGraphIdentity(validation_graph);
    const FE::Real split_start_time{1.0};
    const auto provenance =
        level_set::LevelSetP1PhaseSplitStageProvenance{
            .scheme = level_set::LevelSetP1PhaseSplitScheme::
                BackwardEulerExplicitIndicatorEndpointVelocity,
            .transport_mesh_policy =
                level_set::LevelSetP1PhaseTransportMeshPolicy::
                    FixedBackground,
            .temporal_order = 1,
            .prospective_step = 9u,
            .attempt = 3u,
            .step_start_time = split_start_time,
            .step_end_time = split_start_time + stage.time_step,
            .q_input_time = split_start_time,
            .velocity_state_time = split_start_time + stage.time_step,
            .time_step = stage.time_step,
            .operator_state_revision = 0xb701u,
            .previous_q_revision =
                level_set::levelSetP1PhaseScalarContentRevision(indicator),
            .nodal_velocity_revision =
                level_set::levelSetP1PhaseVelocityContentRevision(
                    stage.sampled_nodal_velocity),
            .previous_graph_identity = graph_identity,
            .operator_graph_identity = graph_identity,
            .final_flux_ledger_digest =
                level_set::levelSetP1PhaseFluxLedgerDigest(stage),
            .stage_options = stage.executed_options,
        };
    const auto split_validation =
        level_set::validateLevelSetP1PhaseSplitStage(
            validation_graph, indicator, stage, provenance);
    ASSERT_TRUE(split_validation.valid) << split_validation.diagnostic;
    const auto local_flux_digest = static_cast<unsigned long long>(
        split_validation.computed_flux_ledger_digest);
    unsigned long long minimum_flux_digest = 0u;
    unsigned long long maximum_flux_digest = 0u;
    MPI_Allreduce(&local_flux_digest, &minimum_flux_digest, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_digest, &maximum_flux_digest, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_NE(minimum_flux_digest, 0u);
    EXPECT_EQ(minimum_flux_digest, maximum_flux_digest);

    const auto expect_collective_provenance_rejection =
        [&](const auto& candidate_stage,
            const auto& candidate_provenance,
            std::string_view expected_text) {
            const auto validation =
                level_set::validateLevelSetP1PhaseSplitStage(
                    validation_graph,
                    indicator,
                    candidate_stage,
                    candidate_provenance);
            const int local_rejected = validation.valid ? 0 : 1;
            int every_rank_rejected = 0;
            MPI_Allreduce(&local_rejected, &every_rank_rejected, 1,
                          MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            EXPECT_EQ(every_rank_rejected, 1);
            EXPECT_NE(validation.diagnostic.find(expected_text),
                      std::string::npos)
                << validation.diagnostic;
            int root_length =
                rank == 0
                    ? static_cast<int>(validation.diagnostic.size())
                    : 0;
            MPI_Bcast(&root_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::string root_diagnostic(
                static_cast<std::size_t>(root_length), '\0');
            if (rank == 0) {
                std::copy(validation.diagnostic.begin(),
                          validation.diagnostic.end(),
                          root_diagnostic.begin());
            }
            MPI_Bcast(root_diagnostic.data(), root_length, MPI_CHAR, 0,
                      MPI_COMM_WORLD);
            EXPECT_EQ(validation.diagnostic, root_diagnostic);
        };
    {
        auto drifted = provenance;
        if (rank == 1) {
            ++drifted.attempt;
        }
        expect_collective_provenance_rejection(
            stage, drifted, "identical attempt");
    }
    {
        auto drifted_stage = stage;
        auto drifted = provenance;
        if (rank == 1) {
            drifted_stage.executed_options.maximum_courant =
                std::nextafter(
                    drifted_stage.executed_options.maximum_courant,
                    FE::Real{0.0});
            drifted.stage_options = drifted_stage.executed_options;
            drifted.final_flux_ledger_digest =
                level_set::levelSetP1PhaseFluxLedgerDigest(
                    drifted_stage);
        }
        expect_collective_provenance_rejection(
            drifted_stage, drifted,
            "identical maximum Courant option");
    }
    {
        auto drifted_stage = stage;
        auto drifted = provenance;
        if (rank == 1) {
            drifted_stage.sampled_nodal_velocity.front()[0] =
                std::nextafter(
                    drifted_stage.sampled_nodal_velocity.front()[0],
                    std::numeric_limits<FE::Real>::infinity());
        }
        expect_collective_provenance_rejection(
            drifted_stage, drifted, "failed on rank 1");
    }
    {
        auto drifted_stage = stage;
        auto drifted = provenance;
        if (rank == 1) {
            drifted_stage.sampled_nodal_velocity.front()[2] =
                std::nextafter(
                    drifted_stage.sampled_nodal_velocity.front()[2],
                    std::numeric_limits<FE::Real>::infinity());
            drifted.nodal_velocity_revision =
                level_set::levelSetP1PhaseVelocityContentRevision(
                    drifted_stage.sampled_nodal_velocity);
        }
        expect_collective_provenance_rejection(
            drifted_stage, drifted,
            "identical nodal velocity revision");
    }
    {
        auto drifted_stage = stage;
        auto drifted = provenance;
        if (rank == 1) {
            drifted_stage.correction.components.front()
                .limited_balance_residual = std::nextafter(
                    drifted_stage.correction.components.front()
                        .limited_balance_residual,
                    std::numeric_limits<FE::Real>::infinity());
            drifted.final_flux_ledger_digest =
                level_set::levelSetP1PhaseFluxLedgerDigest(
                    drifted_stage);
        }
        expect_collective_provenance_rejection(
            drifted_stage, drifted,
            "failed on rank 1");
    }

    const auto local_component_count = static_cast<unsigned long long>(
        stage.correction.components.size());
    unsigned long long minimum_component_count = 0u;
    unsigned long long maximum_component_count = 0u;
    MPI_Allreduce(&local_component_count, &minimum_component_count, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_component_count, &maximum_component_count, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_component_count, maximum_component_count);
    std::vector<FE::GlobalIndex> minimum_component_ids(graph.nodes, 0);
    std::vector<FE::GlobalIndex> maximum_component_ids(graph.nodes, 0);
    MPI_Allreduce(stage.correction.node_component_ids.data(),
                  minimum_component_ids.data(),
                  static_cast<int>(graph.nodes), MPI_INT64_T, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(stage.correction.node_component_ids.data(),
                  maximum_component_ids.data(),
                  static_cast<int>(graph.nodes), MPI_INT64_T, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_EQ(minimum_component_ids, maximum_component_ids);

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
    EXPECT_TRUE(constant_stage.correction.component_balance_satisfied);
    EXPECT_TRUE(
        constant_stage.correction.component_measure_closure_satisfied);
    ASSERT_EQ(constant_stage.correction.components.size(), 1u);
    EXPECT_EQ(constant_stage.correction.components.front().component_id, 0);
    EXPECT_EQ(constant_stage.correction.components.front().nodes,
              graph.nodes);
    for (const auto& node : constant_stage.correction.nodes) {
        EXPECT_NEAR(node.limited_liquid_indicator, 0.4, 2.0e-14);
    }
}

TEST(LevelSetConservativePhaseOperatorMPI,
     RejectsEveryReplicatedStageInputMismatchCollectively)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";

    MPI_Comm source_communicator = MPI_COMM_NULL;
    ASSERT_EQ(MPI_Comm_dup(MPI_COMM_WORLD, &source_communicator),
              MPI_SUCCESS);
    level_set::LevelSetP1PhaseTransportGraph graph;
    {
        auto mesh =
            std::make_shared<TwoRankQuadPhaseMeshAccess>(rank);
        FE::systems::FESystem system(mesh);
        const auto indicator_field =
            system.addField(FE::systems::FieldSpec{
                .name = "liquid_indicator",
                .space = std::make_shared<FE::spaces::H1Space>(
                    FE::ElementType::Quad4, /*order=*/1),
                .components = 1,
            });
        system.setup(phaseSetupOptions(rank, source_communicator),
                     phaseSetupInputs());
        graph = level_set::buildLevelSetP1PhaseTransportGraph(
            system, indicator_field);
    }
    EXPECT_EQ(MPI_Comm_free(&source_communicator), MPI_SUCCESS);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    const std::vector<FE::Real> base_indicator(graph.nodes, 0.4);
    const std::vector<FE::Real> base_lower(graph.nodes, 0.0);
    const std::vector<FE::Real> base_upper(graph.nodes, 1.0);
    const std::vector<std::array<FE::Real, 3>> base_velocity(
        graph.nodes, {0.0, 0.0, 0.0});
    const level_set::LevelSetP1PhaseStageOptions base_options{};

    const auto expect_collective_rejection =
        [&](const level_set::LevelSetP1PhaseTransportStageResult& stage,
            const std::string& expected_text) {
            EXPECT_FALSE(stage.success);
            EXPECT_FALSE(stage.replicated_stage_inputs_satisfied);
            EXPECT_NE(stage.diagnostic.find(expected_text),
                      std::string::npos);
            int root_length =
                rank == 0
                    ? static_cast<int>(stage.diagnostic.size())
                    : 0;
            MPI_Bcast(&root_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::string root_diagnostic(
                static_cast<std::size_t>(root_length), '\0');
            if (rank == 0) {
                std::copy(stage.diagnostic.begin(),
                          stage.diagnostic.end(),
                          root_diagnostic.begin());
            }
            MPI_Bcast(root_diagnostic.data(), root_length, MPI_CHAR, 0,
                      MPI_COMM_WORLD);
            EXPECT_EQ(stage.diagnostic, root_diagnostic);
        };

    {
        auto indicator = base_indicator;
        if (rank == 1) {
            indicator[2] = 0.45;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, indicator, base_lower, base_upper,
                base_velocity, 0.02, base_options),
            "previous liquid indicator");
    }
    {
        auto lower = base_lower;
        if (rank == 1) {
            lower[1] = 0.1;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, lower, base_upper,
                base_velocity, 0.02, base_options),
            "lower liquid-indicator bound");
    }
    {
        auto upper = base_upper;
        if (rank == 1) {
            upper[3] = 0.9;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, upper,
                base_velocity, 0.02, base_options),
            "upper liquid-indicator bound");
    }
    {
        auto velocity = base_velocity;
        if (rank == 1) {
            velocity[4][1] = 0.2;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                velocity, 0.02, base_options),
            "nodal velocity");
    }
    {
        auto velocity = base_velocity;
        if (rank == 1) {
            velocity[4][1] = -FE::Real{0.0};
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                velocity, 0.02, base_options),
            "nodal velocity");
    }
    {
        const FE::Real time_step = rank == 1 ? 0.03 : 0.02;
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                base_velocity, time_step, base_options),
            "time step");
    }
    {
        auto options = base_options;
        if (rank == 1) {
            options.invariant_tolerance = 2.0e-12;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                base_velocity, 0.02, options),
            "invariant tolerance");
    }
    {
        auto options = base_options;
        if (rank == 1) {
            options.component_activity_tolerance = 2.0e-8;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                base_velocity, 0.02, options),
            "component activity tolerance");
    }
    {
        auto options = base_options;
        if (rank == 1) {
            options.maximum_courant = 0.9;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                base_velocity, 0.02, options),
            "maximum Courant number");
    }
    {
        auto options = base_options;
        if (rank == 1) {
            options.enforce_courant_limit = false;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                base_velocity, 0.02, options),
            "Courant-limit enforcement");
    }
    {
        auto options = base_options;
        if (rank == 1) {
            options.require_constant_preservation = false;
        }
        expect_collective_rejection(
            level_set::advanceLevelSetP1ConservativePhaseStage(
                graph, base_indicator, base_lower, base_upper,
                base_velocity, 0.02, options),
            "constant-preservation requirements");
    }

    const auto consistent_stage =
        level_set::advanceLevelSetP1ConservativePhaseStage(
            graph, base_indicator, base_lower, base_upper,
            base_velocity, 0.02, base_options);
    EXPECT_TRUE(consistent_stage.success)
        << consistent_stage.diagnostic;
    EXPECT_TRUE(consistent_stage.replicated_stage_inputs_satisfied);
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
