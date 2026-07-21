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
#include <memory>
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
