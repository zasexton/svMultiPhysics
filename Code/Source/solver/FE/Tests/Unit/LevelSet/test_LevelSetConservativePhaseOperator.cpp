#include "LevelSet/LevelSetConservativePhaseOperator.h"

#include "Assembly/Assembler.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SingleCellPhaseMeshAccess final : public FE::assembly::IMeshAccess {
public:
    using FE::assembly::IMeshAccess::getCellCoordinates;

    SingleCellPhaseMeshAccess(
        FE::ElementType type,
        int dimension,
        std::vector<std::array<FE::Real, 3>> coordinates)
        : type_(type), dimension_(dimension), coordinates_(std::move(coordinates))
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return dimension_; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell*/) const override
    {
        return true;
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex /*cell*/) const override
    {
        return type_;
    }
    void getCellNodes(FE::GlobalIndex /*cell*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.resize(coordinates_.size());
        for (std::size_t i = 0; i < coordinates_.size(); ++i) {
            nodes[i] = static_cast<FE::GlobalIndex>(i);
        }
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node) const override
    {
        return coordinates_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        FE::GlobalIndex /*cell*/,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        coordinates = coordinates_;
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
        return {0, 0};
    }
    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
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
    FE::ElementType type_{FE::ElementType::Unknown};
    int dimension_{0};
    std::vector<std::array<FE::Real, 3>> coordinates_{};
};

[[nodiscard]] FE::systems::SetupInputs makeSingleCellSetupInputs(
    std::size_t node_count,
    int dimension)
{
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = 1;
    topology.n_vertices = static_cast<FE::GlobalIndex>(node_count);
    topology.dim = dimension;
    topology.cell2vertex_offsets = {
        0, static_cast<FE::MeshOffset>(node_count)};
    topology.cell2vertex_data.resize(node_count);
    topology.vertex_gids.resize(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
        topology.cell2vertex_data[i] = static_cast<FE::MeshIndex>(i);
        topology.vertex_gids[i] = static_cast<FE::dofs::gid_t>(i);
    }
    topology.cell_gids = {0};
    topology.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

struct PhaseSystemFixture {
    std::shared_ptr<SingleCellPhaseMeshAccess> mesh;
    FE::systems::FESystem system;
    FE::FieldId indicator{FE::INVALID_FIELD_ID};

    PhaseSystemFixture(
        FE::ElementType type,
        int dimension,
        std::vector<std::array<FE::Real, 3>> coordinates,
        int order = 1)
        : mesh(std::make_shared<SingleCellPhaseMeshAccess>(
              type, dimension, std::move(coordinates))),
          system(mesh)
    {
        indicator = system.addField(FE::systems::FieldSpec{
            .name = "liquid_indicator",
            .space = std::make_shared<FE::spaces::H1Space>(type, order),
            .components = 1,
        });
        system.setup({}, makeSingleCellSetupInputs(
                             static_cast<std::size_t>(mesh->numVertices()),
                             dimension));
    }
};

[[nodiscard]] PhaseSystemFixture makeUnitTriangleFixture(int order = 1)
{
    return PhaseSystemFixture(
        FE::ElementType::Triangle3,
        /*dimension=*/2,
        {{0.0, 0.0, 0.0},
         {1.0, 0.0, 0.0},
         {0.0, 1.0, 0.0}},
        order);
}

[[nodiscard]] FE::Real polygonArea(
    const std::vector<std::array<FE::Real, 3>>& coordinates)
{
    FE::Real twice_area{0.0};
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        const auto& first = coordinates[i];
        const auto& second = coordinates[(i + 1u) % coordinates.size()];
        twice_area += first[0] * second[1] - second[0] * first[1];
    }
    return FE::Real{0.5} * std::abs(twice_area);
}

[[nodiscard]] level_set::LevelSetP1PhaseSplitStageProvenance
makeBackwardEulerSplitStageProvenance(
    const level_set::LevelSetP1PhaseTransportGraph& graph,
    std::span<const FE::Real> previous_q,
    const level_set::LevelSetP1PhaseTransportStageResult& stage,
    FE::Real step_start_time = FE::Real{1.0})
{
    const auto graph_identity =
        level_set::levelSetP1PhaseGraphIdentity(graph);
    return level_set::LevelSetP1PhaseSplitStageProvenance{
        .scheme = level_set::LevelSetP1PhaseSplitScheme::
            BackwardEulerExplicitIndicatorEndpointVelocity,
        .transport_mesh_policy =
            level_set::LevelSetP1PhaseTransportMeshPolicy::FixedBackground,
        .temporal_order = 1,
        .prospective_step = 7u,
        .attempt = 2u,
        .step_start_time = step_start_time,
        .step_end_time = step_start_time + stage.time_step,
        .q_input_time = step_start_time,
        .velocity_state_time = step_start_time + stage.time_step,
        .time_step = stage.time_step,
        .operator_state_revision = 0xa501u,
        .previous_q_revision =
            level_set::levelSetP1PhaseScalarContentRevision(previous_q),
        .nodal_velocity_revision =
            level_set::levelSetP1PhaseVelocityContentRevision(
                stage.sampled_nodal_velocity),
        .previous_graph_identity = graph_identity,
        .operator_graph_identity = graph_identity,
        .final_flux_ledger_digest =
            level_set::levelSetP1PhaseFluxLedgerDigest(stage),
        .stage_options = stage.executed_options,
    };
}

TEST(LevelSetConservativePhaseOperator,
     AssemblesExactUnitTriangleMassAndGradientCoefficients)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);

    ASSERT_TRUE(graph.success) << graph.diagnostic;
    EXPECT_EQ(graph.dimension, 2);
    EXPECT_EQ(graph.cells, 1u);
    EXPECT_EQ(graph.nodes, 3u);
    ASSERT_EQ(graph.edges.size(), 3u);
    EXPECT_NEAR(graph.physical_measure, 0.5, 2.0e-14);
    EXPECT_NEAR(graph.total_lumped_control_volume, 0.5, 2.0e-14);
    for (const auto mass : graph.lumped_control_volume) {
        EXPECT_NEAR(mass, 1.0 / 6.0, 2.0e-14);
    }

    const auto& edge01 = graph.edges[0];
    EXPECT_EQ(edge01.first_node, 0);
    EXPECT_EQ(edge01.second_node, 1);
    EXPECT_NEAR(edge01.first_test_second_gradient[0], 1.0 / 6.0,
                2.0e-14);
    EXPECT_NEAR(edge01.first_test_second_gradient[1], 0.0, 2.0e-14);
    EXPECT_NEAR(edge01.second_test_first_gradient[0], -1.0 / 6.0,
                2.0e-14);
    EXPECT_NEAR(edge01.second_test_first_gradient[1], -1.0 / 6.0,
                2.0e-14);

    ASSERT_EQ(graph.boundary_column_sum.size(), 3u);
    EXPECT_NEAR(graph.boundary_column_sum[0][0], -0.5, 2.0e-14);
    EXPECT_NEAR(graph.boundary_column_sum[0][1], -0.5, 2.0e-14);
    EXPECT_NEAR(graph.boundary_column_sum[1][0], 0.5, 2.0e-14);
    EXPECT_NEAR(graph.boundary_column_sum[1][1], 0.0, 2.0e-14);
    EXPECT_NEAR(graph.boundary_column_sum[2][0], 0.0, 2.0e-14);
    EXPECT_NEAR(graph.boundary_column_sum[2][1], 0.5, 2.0e-14);
    EXPECT_TRUE(graph.partition_of_unity_satisfied);
    EXPECT_TRUE(graph.gradient_partition_satisfied);
    EXPECT_TRUE(graph.gradient_row_sum_satisfied);
    EXPECT_TRUE(graph.measure_closure_satisfied);
}

TEST(LevelSetConservativePhaseOperator,
     UsesPointwiseMappingAndClosesIdentitiesOnADistortedQuadrilateral)
{
    const std::vector<std::array<FE::Real, 3>> coordinates{
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {1.7, 1.2, 0.0},
        {-0.2, 0.8, 0.0},
    };
    PhaseSystemFixture fixture(
        FE::ElementType::Quad4, /*dimension=*/2, coordinates);
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);

    ASSERT_TRUE(graph.success) << graph.diagnostic;
    EXPECT_EQ(graph.edges.size(), 6u);
    EXPECT_NEAR(graph.physical_measure, polygonArea(coordinates), 2.0e-13);
    EXPECT_NEAR(graph.total_lumped_control_volume,
                polygonArea(coordinates), 2.0e-13);
    EXPECT_GT(graph.minimum_jacobian_determinant, 0.0);
    EXPECT_GT(graph.minimum_lumped_control_volume, 0.0);
    EXPECT_LE(graph.maximum_partition_of_unity_residual, 2.0e-14);
    EXPECT_LE(graph.maximum_gradient_partition_residual, 2.0e-14);
    EXPECT_LE(graph.maximum_gradient_row_sum_residual, 2.0e-14);
    EXPECT_LE(std::abs(graph.measure_closure_residual), 2.0e-14);
}

TEST(LevelSetConservativePhaseOperator,
     PreservesAConstantWithVariableVelocityByBoundaryDivergenceCancellation)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    const std::array<FE::Real, 3> indicator{0.37, 0.37, 0.37};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<std::array<FE::Real, 3>, 3> velocity{{
        {0.3, -0.2, 0.0},
        {1.1, 0.4, 0.0},
        {-0.5, 0.8, 0.0},
    }};
    const auto stage = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, indicator, lower, upper, velocity, /*time_step=*/0.01);

    ASSERT_TRUE(stage.success) << stage.diagnostic;
    EXPECT_TRUE(stage.courant_satisfied);
    EXPECT_TRUE(stage.low_order_coefficients_nonnegative);
    EXPECT_TRUE(stage.strong_form_decomposition_satisfied);
    EXPECT_TRUE(stage.correction.constant_state_input);
    EXPECT_TRUE(stage.correction.constant_preservation_satisfied);
    EXPECT_FALSE(stage.correction.applied);
    EXPECT_LE(stage.maximum_strong_form_decomposition_residual, 2.0e-17);
    ASSERT_EQ(stage.correction.nodes.size(), indicator.size());
    for (const auto& node : stage.correction.nodes) {
        EXPECT_NEAR(node.low_order_liquid_indicator, 0.37, 2.0e-14);
        EXPECT_NEAR(node.limited_liquid_indicator, 0.37, 2.0e-14);
    }
    FE::Real boundary_total{0.0};
    FE::Real divergence_total{0.0};
    for (std::size_t i = 0; i < indicator.size(); ++i) {
        boundary_total += stage.physical_boundary_mass_transfer[i];
        divergence_total += stage.discrete_divergence_mass_source[i];
    }
    EXPECT_GT(std::abs(boundary_total), 1.0e-8);
    EXPECT_NEAR(boundary_total + divergence_total, 0.0, 2.0e-16);
}

TEST(LevelSetConservativePhaseOperator,
     RejectsRawTargetDriftHiddenByFluxLimitingForAConstantState)
{
    level_set::LevelSetP1PhaseTransportGraph graph;
    graph.success = true;
    graph.dimension = 2;
    graph.nodes = 2u;
    graph.lumped_control_volume = {1.0, 1.0};
    graph.diagonal_gradient = {{
        {-1.0, 0.0, 0.0},
        {-1.0, 0.0, 0.0},
    }};
    graph.boundary_column_sum.resize(2u);
    graph.edges.push_back(level_set::LevelSetP1PhaseGradientEdge{
        .first_node = 0,
        .second_node = 1,
        .owner_rank = 0,
        .first_test_second_gradient = {1.0, 0.0, 0.0},
        .second_test_first_gradient = {1.0, 0.0, 0.0},
    });

    const std::array<FE::Real, 2> indicator{
        0.5, 0.5 + 5.0e-13};
    const std::array<FE::Real, 2> lower = indicator;
    const std::array<FE::Real, 2> upper = indicator;
    const std::array<std::array<FE::Real, 3>, 2> velocity{{
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    }};
    const auto stage = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, indicator, lower, upper, velocity, /*time_step=*/10.0);

    EXPECT_FALSE(stage.success);
    EXPECT_TRUE(stage.replicated_stage_inputs_satisfied);
    EXPECT_TRUE(stage.correction.constant_state_input);
    EXPECT_FALSE(stage.correction.constant_preservation_satisfied);
    EXPECT_FALSE(stage.correction.success);
    EXPECT_GT(stage.correction.maximum_constant_preservation_error,
              4.0e-12);
    EXPECT_NE(stage.diagnostic.find("raw target"),
              std::string::npos);
    ASSERT_EQ(stage.correction.nodes.size(), indicator.size());
    for (std::size_t node = 0u; node < indicator.size(); ++node) {
        EXPECT_EQ(
            stage.correction.nodes[node].low_order_liquid_indicator,
            indicator[node]);
        EXPECT_EQ(
            stage.correction.nodes[node].limited_liquid_indicator,
            indicator[node]);
    }
}

TEST(LevelSetConservativePhaseOperator,
     LimitsTheLumpedCentralTargetWithoutChangingExternalPhaseBalance)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    const std::array<FE::Real, 3> indicator{1.0, 0.0, 0.0};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<std::array<FE::Real, 3>, 3> velocity{{
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    }};
    const auto stage = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, indicator, lower, upper, velocity, /*time_step=*/0.1);

    ASSERT_TRUE(stage.success) << stage.diagnostic;
    EXPECT_TRUE(stage.correction.applied);
    EXPECT_GT(stage.correction.limited_edges, 0u);
    EXPECT_TRUE(stage.correction.interior_cancellation_satisfied);
    EXPECT_TRUE(stage.correction.local_balance_satisfied);
    EXPECT_TRUE(stage.correction.global_balance_satisfied);
    EXPECT_LE(stage.correction.minimum_limited_liquid_indicator, 1.0);
    EXPECT_GE(stage.correction.minimum_limited_liquid_indicator, 0.0);
    EXPECT_LE(stage.correction.maximum_limited_liquid_indicator, 1.0);
    EXPECT_EQ(stage.correction.maximum_edge_pair_cancellation_residual, 0.0);
    EXPECT_NEAR(
        stage.correction.total_limited_liquid_measure -
            stage.correction.total_previous_liquid_measure,
        stage.correction.total_physical_boundary_mass_transfer +
            stage.correction.total_discrete_divergence_mass_source,
                2.0e-14);
}

TEST(LevelSetConservativePhaseOperator,
     ValidatesBackwardEulerExplicitIndicatorEndpointVelocityProvenance)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    const std::array<FE::Real, 3> previous_q{1.0, 0.0, 0.0};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<std::array<FE::Real, 3>, 3> endpoint_velocity{{
        {0.25, -0.125, 0.0},
        {0.5, 0.25, 0.0},
        {-0.25, 0.375, 0.0},
    }};
    const auto stage = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, previous_q, lower, upper, endpoint_velocity,
        /*time_step=*/FE::Real{0.0625});
    ASSERT_TRUE(stage.success) << stage.diagnostic;
    EXPECT_TRUE(stage.replicated_stage_inputs_satisfied);
    EXPECT_EQ(stage.time_step, FE::Real{0.0625});
    const std::vector<std::array<FE::Real, 3>>
        expected_endpoint_velocity(
            endpoint_velocity.begin(), endpoint_velocity.end());
    EXPECT_EQ(stage.sampled_nodal_velocity, expected_endpoint_velocity);
    EXPECT_NE(stage.sampled_nodal_velocity.front()[0], FE::Real{0.0});

    const auto provenance = makeBackwardEulerSplitStageProvenance(
        graph, previous_q, stage);
    const auto validation = level_set::validateLevelSetP1PhaseSplitStage(
        graph, previous_q, stage, provenance);

    ASSERT_TRUE(validation.valid) << validation.diagnostic;
    EXPECT_STREQ(
        level_set::levelSetP1PhaseSplitSchemeName(provenance.scheme),
        "backward_euler_explicit_indicator_endpoint_velocity");
    EXPECT_EQ(validation.computed_previous_q_revision,
              provenance.previous_q_revision);
    EXPECT_EQ(validation.computed_nodal_velocity_revision,
              provenance.nodal_velocity_revision);
    EXPECT_EQ(validation.computed_flux_ledger_digest,
              provenance.final_flux_ledger_digest);
    EXPECT_NE(validation.computed_flux_ledger_digest, 0u);
    EXPECT_EQ(validation.actual_operator_graph_identity.content_revision,
              provenance.operator_graph_identity.content_revision);
}

TEST(LevelSetConservativePhaseOperator,
     RejectsSplitStageTimeRevisionMetadataAndFixedGraphDrift)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;
    const std::array<FE::Real, 3> previous_q{0.8, 0.1, 0.0};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{0.8, 0.8, 0.8};
    const std::array<std::array<FE::Real, 3>, 3> endpoint_velocity{{
        {0.125, 0.0, 0.0},
        {0.25, 0.125, 0.0},
        {-0.125, 0.25, 0.0},
    }};
    const auto stage = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, previous_q, lower, upper, endpoint_velocity,
        /*time_step=*/FE::Real{0.03125});
    ASSERT_TRUE(stage.success) << stage.diagnostic;
    const auto baseline = makeBackwardEulerSplitStageProvenance(
        graph, previous_q, stage);

    const auto expect_invalid = [&](auto mutate, std::string_view text) {
        auto provenance = baseline;
        mutate(provenance);
        const auto validation =
            level_set::validateLevelSetP1PhaseSplitStage(
                graph, previous_q, stage, provenance);
        EXPECT_FALSE(validation.valid);
        EXPECT_NE(validation.diagnostic.find(text), std::string::npos)
            << validation.diagnostic;
    };
    expect_invalid(
        [](auto& provenance) {
            provenance.scheme = level_set::LevelSetP1PhaseSplitScheme::
                GeneralizedAlphaUnsupported;
        },
        "only the Backward-Euler");
    expect_invalid(
        [](auto& provenance) {
            provenance.transport_mesh_policy =
                level_set::LevelSetP1PhaseTransportMeshPolicy::
                    MovingMeshUnsupported;
        },
        "fixed background");
    expect_invalid(
        [](auto& provenance) { provenance.temporal_order = 2; },
        "temporal order one");
    expect_invalid(
        [](auto& provenance) { provenance.prospective_step = 0u; },
        "positive prospective-step");
    expect_invalid(
        [](auto& provenance) { provenance.attempt = 0u; },
        "positive prospective-step");
    expect_invalid(
        [](auto& provenance) {
            provenance.q_input_time = std::nextafter(
                provenance.q_input_time,
                std::numeric_limits<FE::Real>::infinity());
        },
        "exact q^n/start");
    expect_invalid(
        [](auto& provenance) {
            provenance.velocity_state_time = std::nextafter(
                provenance.velocity_state_time,
                std::numeric_limits<FE::Real>::infinity());
        },
        "exact q^n/start");
    expect_invalid(
        [](auto& provenance) {
            provenance.time_step = FE::Real{0.0};
        },
        "positive finite time step");
    expect_invalid(
        [](auto& provenance) {
            provenance.step_end_time =
                std::numeric_limits<FE::Real>::infinity();
        },
        "finite times");
    expect_invalid(
        [](auto& provenance) {
            provenance.operator_state_revision = 0u;
        },
        "nonzero operator");
    expect_invalid(
        [](auto& provenance) { provenance.previous_q_revision = 0u; },
        "nonzero operator");
    expect_invalid(
        [](auto& provenance) { provenance.nodal_velocity_revision = 0u; },
        "nonzero operator");
    expect_invalid(
        [](auto& provenance) {
            provenance.final_flux_ledger_digest = 0u;
        },
        "nonzero operator");
    expect_invalid(
        [](auto& provenance) {
            ++provenance.previous_graph_identity.topology_revision;
        },
        "graph geometry/topology");
    expect_invalid(
        [](auto& provenance) {
            ++provenance.operator_graph_identity.dof_layout_revision;
        },
        "graph geometry/topology");
}

TEST(LevelSetConservativePhaseOperator,
     SplitStageRequiresExactClampedProductionOneRingBounds)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;
    const std::array<std::array<FE::Real, 3>, 3> zero_velocity{};

    const std::array<FE::Real, 3> previous_q{0.8, 0.1, 0.0};
    const std::array<FE::Real, 3> generic_lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> generic_upper{1.0, 1.0, 1.0};
    const auto generic_stage =
        level_set::advanceLevelSetP1ConservativePhaseStage(
            graph, previous_q, generic_lower, generic_upper,
            zero_velocity, FE::Real{0.03125});
    ASSERT_TRUE(generic_stage.success) << generic_stage.diagnostic;
    const auto generic_provenance =
        makeBackwardEulerSplitStageProvenance(
            graph, previous_q, generic_stage);
    const auto generic_validation =
        level_set::validateLevelSetP1PhaseSplitStage(
            graph, previous_q, generic_stage, generic_provenance);
    EXPECT_FALSE(generic_validation.valid);
    EXPECT_NE(generic_validation.diagnostic.find(
                  "production one-ring bound mismatch"),
              std::string::npos)
        << generic_validation.diagnostic;

    const std::array<FE::Real, 3> near_unit_q{
        -FE::Real{5.0e-13}, FE::Real{0.4}, FE::Real{0.8}};
    const std::array<FE::Real, 3> clamped_lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> clamped_upper{0.8, 0.8, 0.8};
    const auto clamped_stage =
        level_set::advanceLevelSetP1ConservativePhaseStage(
            graph, near_unit_q, clamped_lower, clamped_upper,
            zero_velocity, FE::Real{0.03125});
    ASSERT_TRUE(clamped_stage.success) << clamped_stage.diagnostic;
    const auto clamped_provenance =
        makeBackwardEulerSplitStageProvenance(
            graph, near_unit_q, clamped_stage);
    const auto clamped_validation =
        level_set::validateLevelSetP1PhaseSplitStage(
            graph, near_unit_q, clamped_stage, clamped_provenance);
    EXPECT_TRUE(clamped_validation.valid)
        << clamped_validation.diagnostic;
}

TEST(LevelSetConservativePhaseOperator,
     ExactInputAndCompleteFluxLedgerDigestsAreMutationSensitive)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;
    const std::array<FE::Real, 3> previous_q{1.0, 0.0, 0.0};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<std::array<FE::Real, 3>, 3> endpoint_velocity{{
        {0.5, 0.0, 0.0},
        {0.5, 0.25, 0.0},
        {-0.25, 0.5, 0.0},
    }};
    const auto stage = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, previous_q, lower, upper, endpoint_velocity,
        /*time_step=*/FE::Real{0.03125});
    ASSERT_TRUE(stage.success) << stage.diagnostic;
    ASSERT_FALSE(stage.correction.nodes.empty());
    ASSERT_FALSE(stage.correction.edges.empty());
    ASSERT_FALSE(stage.correction.components.empty());
    const auto provenance = makeBackwardEulerSplitStageProvenance(
        graph, previous_q, stage);

    auto drifted_options_provenance = provenance;
    drifted_options_provenance.stage_options.maximum_courant =
        std::nextafter(
            drifted_options_provenance.stage_options.maximum_courant,
            std::numeric_limits<FE::Real>::infinity());
    const auto drifted_options_validation =
        level_set::validateLevelSetP1PhaseSplitStage(
            graph, previous_q, stage, drifted_options_provenance);
    EXPECT_FALSE(drifted_options_validation.valid);
    EXPECT_NE(drifted_options_validation.diagnostic.find(
                  "options do not exactly match"),
              std::string::npos)
        << drifted_options_validation.diagnostic;

    auto changed_options_stage = stage;
    changed_options_stage.executed_options.enforce_courant_limit =
        !changed_options_stage.executed_options.enforce_courant_limit;
    EXPECT_NE(level_set::levelSetP1PhaseFluxLedgerDigest(
                  changed_options_stage),
              provenance.final_flux_ledger_digest);
    EXPECT_FALSE(level_set::validateLevelSetP1PhaseSplitStage(
                     graph, previous_q, changed_options_stage, provenance)
                     .valid);

    auto changed_q = previous_q;
    changed_q[1] = std::nextafter(
        changed_q[1], std::numeric_limits<FE::Real>::infinity());
    EXPECT_NE(level_set::levelSetP1PhaseScalarContentRevision(changed_q),
              provenance.previous_q_revision);
    EXPECT_FALSE(level_set::validateLevelSetP1PhaseSplitStage(
                     graph, changed_q, stage, provenance)
                     .valid);

    auto changed_velocity_stage = stage;
    changed_velocity_stage.sampled_nodal_velocity[1][2] =
        std::nextafter(
            changed_velocity_stage.sampled_nodal_velocity[1][2],
            std::numeric_limits<FE::Real>::infinity());
    EXPECT_NE(level_set::levelSetP1PhaseVelocityContentRevision(
                  changed_velocity_stage.sampled_nodal_velocity),
              provenance.nodal_velocity_revision);
    EXPECT_FALSE(level_set::validateLevelSetP1PhaseSplitStage(
                     graph, previous_q, changed_velocity_stage, provenance)
                     .valid);

    const auto baseline_digest =
        level_set::levelSetP1PhaseFluxLedgerDigest(stage);
    const auto expect_digest_change = [&](auto mutate) {
        auto changed = stage;
        mutate(changed);
        EXPECT_NE(level_set::levelSetP1PhaseFluxLedgerDigest(changed),
                  baseline_digest);
        EXPECT_FALSE(level_set::validateLevelSetP1PhaseSplitStage(
                         graph, previous_q, changed, provenance)
                         .valid);
    };

    const auto expect_invariant_rejection = [&](auto mutate) {
        auto changed = stage;
        mutate(changed);
        auto changed_provenance = provenance;
        changed_provenance.final_flux_ledger_digest =
            level_set::levelSetP1PhaseFluxLedgerDigest(changed);
        const auto validation =
            level_set::validateLevelSetP1PhaseSplitStage(
                graph, previous_q, changed, changed_provenance);
        EXPECT_FALSE(validation.valid);
        EXPECT_NE(validation.diagnostic.find(
                      "successful replicated transport stage"),
                  std::string::npos)
            << validation.diagnostic;
    };
    expect_invariant_rejection(
        [](auto& changed) { changed.courant_satisfied = false; });
    expect_invariant_rejection([](auto& changed) {
        changed.low_order_coefficients_nonnegative = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.strong_form_decomposition_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.low_order_bounds_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.limited_bounds_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.interior_cancellation_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.local_balance_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.global_balance_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.component_balance_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.component_measure_closure_satisfied = false;
    });
    expect_invariant_rejection([](auto& changed) {
        changed.correction.constant_preservation_satisfied = false;
    });
    const auto expect_equation_rejection = [&](auto mutate) {
        auto changed = stage;
        mutate(changed);
        auto changed_provenance = provenance;
        changed_provenance.final_flux_ledger_digest =
            level_set::levelSetP1PhaseFluxLedgerDigest(changed);
        const auto validation =
            level_set::validateLevelSetP1PhaseSplitStage(
                graph, previous_q, changed, changed_provenance);
        EXPECT_FALSE(validation.valid);
        EXPECT_NE(validation.diagnostic.find(
                      "independent ledger verification"),
                  std::string::npos)
            << validation.diagnostic;
    };
    expect_equation_rejection([](auto& changed) {
        changed.maximum_courant =
            changed.executed_options.maximum_courant + FE::Real{0.25};
    });
    expect_equation_rejection([](auto& changed) {
        changed.correction.nodes.front().local_mass_balance_residual =
            FE::Real{0.125};
    });
    expect_equation_rejection([](auto& changed) {
        changed.correction.nodes[1].upper_liquid_indicator =
            std::nextafter(
                changed.correction.nodes[1].upper_liquid_indicator,
                FE::Real{0.0});
    });
    expect_digest_change([](auto& changed) {
        changed.correction.nodes.front().local_mass_balance_residual =
            std::nextafter(
                changed.correction.nodes.front().local_mass_balance_residual,
                std::numeric_limits<FE::Real>::infinity());
    });
    expect_digest_change([](auto& changed) {
        changed.correction.edges.front().limited_pair_cancellation_residual =
            std::nextafter(
                changed.correction.edges.front()
                    .limited_pair_cancellation_residual,
                std::numeric_limits<FE::Real>::infinity());
    });
    expect_digest_change([](auto& changed) {
        changed.correction.components.front().limited_balance_residual =
            std::nextafter(
                changed.correction.components.front()
                    .limited_balance_residual,
                std::numeric_limits<FE::Real>::infinity());
    });
    expect_digest_change([](auto& changed) {
        changed.correction.subthreshold_component
            .limited_balance_residual = std::nextafter(
                changed.correction.subthreshold_component
                    .limited_balance_residual,
                std::numeric_limits<FE::Real>::infinity());
    });

    auto changed_graph = graph;
    changed_graph.lumped_control_volume.front() = std::nextafter(
        changed_graph.lumped_control_volume.front(),
        std::numeric_limits<FE::Real>::infinity());
    EXPECT_NE(level_set::levelSetP1PhaseGraphIdentity(changed_graph)
                  .content_revision,
              provenance.operator_graph_identity.content_revision);
    EXPECT_FALSE(level_set::validateLevelSetP1PhaseSplitStage(
                     changed_graph, previous_q, stage, provenance)
                     .valid);
}

TEST(LevelSetConservativePhaseOperator,
     RejectsAnUnsafeTimeStepAndAFieldOutsideTheP1Contract)
{
    auto fixture = makeUnitTriangleFixture();
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    const std::array<FE::Real, 3> indicator{1.0, 0.0, 0.0};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<std::array<FE::Real, 3>, 3> velocity{{
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    }};
    const auto unsafe = level_set::advanceLevelSetP1ConservativePhaseStage(
        graph, indicator, lower, upper, velocity, /*time_step=*/10.0);
    EXPECT_FALSE(unsafe.success);
    EXPECT_FALSE(unsafe.courant_satisfied);
    EXPECT_NE(unsafe.diagnostic.find("Courant"), std::string::npos);

    auto quadratic_fixture = makeUnitTriangleFixture(/*order=*/2);
    const auto invalid_graph =
        level_set::buildLevelSetP1PhaseTransportGraph(
            quadratic_fixture.system, quadratic_fixture.indicator);
    EXPECT_FALSE(invalid_graph.success);
    EXPECT_NE(invalid_graph.diagnostic.find("P1"), std::string::npos);
}

} // namespace
