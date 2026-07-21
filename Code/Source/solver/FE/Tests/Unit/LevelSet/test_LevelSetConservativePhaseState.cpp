#include "LevelSet/LevelSetConservativePhaseState.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SingleCellPhaseStateMesh final
    : public FE::assembly::IMeshAccess {
public:
    using FE::assembly::IMeshAccess::getCellCoordinates;

    SingleCellPhaseStateMesh(
        FE::ElementType type,
        int dimension,
        std::vector<std::array<FE::Real, 3>> coordinates)
        : type_(type),
          dimension_(dimension),
          coordinates_(std::move(coordinates))
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(coordinates_.size());
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return 0;
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override
    {
        return 0;
    }
    [[nodiscard]] int dimension() const override { return dimension_; }
    [[nodiscard]] bool revisionTrackingAvailable() const override
    {
        return true;
    }
    [[nodiscard]] std::uint64_t geometryRevision() const override
    {
        return 3u;
    }
    [[nodiscard]] std::uint64_t topologyRevision() const override
    {
        return 5u;
    }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return 7u;
    }
    [[nodiscard]] std::uint64_t numberingRevision() const override
    {
        return 11u;
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override
    {
        return cell == 0;
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
        coordinates.assign(coordinates_.begin(), coordinates_.end());
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

[[nodiscard]] FE::systems::SetupInputs phaseStateSetupInputs(
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

struct PhaseStateFixture {
    static constexpr int interface_marker = 8123;

    std::shared_ptr<SingleCellPhaseStateMesh> mesh;
    FE::systems::FESystem system;
    FE::FieldId indicator{FE::INVALID_FIELD_ID};
    FE::FieldId phi{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution{};

    PhaseStateFixture()
        : PhaseStateFixture(
              FE::ElementType::Triangle3,
              /*dimension=*/2,
              {{0.0, 0.0, 0.0},
               {1.0, 0.0, 0.0},
               {0.0, 1.0, 0.0}},
              {-0.25, 0.75, 0.75})
    {
    }

    PhaseStateFixture(
        FE::ElementType type,
        int dimension,
        std::vector<std::array<FE::Real, 3>> coordinates,
        std::vector<FE::Real> phi_values)
        : mesh(std::make_shared<SingleCellPhaseStateMesh>(
              type, dimension, std::move(coordinates))),
          system(mesh)
    {
        if (phi_values.size() !=
            static_cast<std::size_t>(mesh->numVertices())) {
            throw std::invalid_argument(
                "phase-state fixture requires one level-set value per vertex");
        }
        const auto space = std::make_shared<FE::spaces::H1Space>(
            type, /*order=*/1);
        indicator = system.addField(FE::systems::FieldSpec{
            .name = "liquid_indicator",
            .space = space,
            .components = 1,
        });
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = space,
            .components = 1,
        });
        system.setup({}, phaseStateSetupInputs(
                             static_cast<std::size_t>(mesh->numVertices()),
                             dimension));
        solution.assign(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()),
            FE::Real{0.0});
        const auto phi_offset = static_cast<std::size_t>(
            system.fieldDofOffset(phi));
        const auto phi_dofs = system.fieldDofHandler(phi).getCellDofs(0);
        for (std::size_t i = 0; i < phi_dofs.size(); ++i) {
            solution[phi_offset +
                     static_cast<std::size_t>(phi_dofs[i])] = phi_values[i];
        }

        level_set::LevelSetGeneratedInterfaceOptions options;
        options.level_set_field_name = "phi";
        options.domain_id = "phase_state_projection";
        options.requested_interface_marker = interface_marker;
        options.interface_quadrature_order = 2;
        options.volume_quadrature_order = 2;
        level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        const auto generated = lifecycle.build(system, options, solution);
        if (!generated.success) {
            throw std::runtime_error(generated.diagnostic);
        }
        auto context =
            std::make_shared<FE::assembly::CutIntegrationContext>();
        context->addGeneratedInterfaceDomain(generated.domain);
        system.setCutIntegrationContext(std::move(context));
    }
};

TEST(LevelSetConservativePhaseState,
     ProjectsExactRetainedMomentsAndComplementOnACutTriangle)
{
    PhaseStateFixture fixture;
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    level_set::LevelSetP1PhaseProjectionOptions negative_options;
    negative_options.interface_marker =
        PhaseStateFixture::interface_marker;
    const auto negative =
        level_set::projectLevelSetP1PhaseIndicatorFromCutContext(
            fixture.system, fixture.indicator, graph, negative_options);
    ASSERT_TRUE(negative.success) << negative.diagnostic;
    EXPECT_TRUE(negative.phase_bounds_satisfied);
    EXPECT_TRUE(negative.rule_moment_closure_satisfied);
    EXPECT_TRUE(negative.global_measure_closure_satisfied);
    EXPECT_TRUE(negative.complement_bounds_satisfied);
    EXPECT_EQ(negative.owned_rules, 1u);
    EXPECT_GT(negative.quadrature_points, 0u);
    EXPECT_GT(negative.cut_context_revision, 0u);
    EXPECT_GT(negative.source_value_revision, 0u);
    EXPECT_NEAR(negative.retained_liquid_measure, 1.0 / 32.0,
                2.0e-14);
    EXPECT_NEAR(negative.projected_liquid_measure, 1.0 / 32.0,
                2.0e-14);
    ASSERT_EQ(negative.liquid_indicator.size(), 3u);
    EXPECT_NEAR(negative.liquid_indicator[0], 5.0 / 32.0,
                2.0e-14);
    EXPECT_NEAR(negative.liquid_indicator[1], 1.0 / 64.0,
                2.0e-14);
    EXPECT_NEAR(negative.liquid_indicator[2], 1.0 / 64.0,
                2.0e-14);

    auto positive_options = negative_options;
    positive_options.liquid_side =
        FE::geometry::CutIntegrationSide::Positive;
    const auto positive =
        level_set::projectLevelSetP1PhaseIndicatorFromCutContext(
            fixture.system, fixture.indicator, graph, positive_options);
    ASSERT_TRUE(positive.success) << positive.diagnostic;
    EXPECT_NEAR(positive.retained_liquid_measure, 15.0 / 32.0,
                2.0e-14);
    ASSERT_EQ(positive.liquid_indicator.size(), 3u);
    for (std::size_t i = 0; i < negative.liquid_indicator.size(); ++i) {
        EXPECT_NEAR(negative.liquid_indicator[i] +
                        positive.liquid_indicator[i],
                    1.0, 3.0e-14);
        EXPECT_NEAR(negative.liquid_phase_mass[i] +
                        positive.liquid_phase_mass[i],
                    graph.lumped_control_volume[i], 3.0e-14);
    }
    EXPECT_NEAR(negative.retained_liquid_measure +
                    positive.retained_liquid_measure,
                graph.physical_measure, 3.0e-14);
}

TEST(LevelSetConservativePhaseState,
     RejectsMissingCutProvenanceAndAStaleGraph)
{
    PhaseStateFixture fixture;
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    level_set::LevelSetP1PhaseProjectionOptions options;
    options.interface_marker = 99;
    const auto missing =
        level_set::projectLevelSetP1PhaseIndicatorFromCutContext(
            fixture.system, fixture.indicator, graph, options);
    EXPECT_FALSE(missing.success);
    EXPECT_NE(missing.diagnostic.find("authoritative"), std::string::npos);

    options.interface_marker = PhaseStateFixture::interface_marker;
    auto stale_graph = graph;
    ++stale_graph.geometry_revision;
    const auto stale =
        level_set::projectLevelSetP1PhaseIndicatorFromCutContext(
            fixture.system, fixture.indicator, stale_graph, options);
    EXPECT_FALSE(stale.success);
    EXPECT_NE(stale.diagnostic.find("does not match"), std::string::npos);
}

TEST(LevelSetConservativePhaseState,
     UsesPointwiseGeometryAndClosesComplementOnADistortedQuadrilateral)
{
    PhaseStateFixture fixture(
        FE::ElementType::Quad4,
        /*dimension=*/2,
        {{0.0, 0.0, 0.0},
         {2.0, 0.0, 0.0},
         {1.7, 1.2, 0.0},
         {-0.2, 0.8, 0.0}},
        {-1.0, 1.0, 1.0, -1.0});
    const auto graph = level_set::buildLevelSetP1PhaseTransportGraph(
        fixture.system, fixture.indicator);
    ASSERT_TRUE(graph.success) << graph.diagnostic;

    level_set::LevelSetP1PhaseProjectionOptions options;
    options.interface_marker = PhaseStateFixture::interface_marker;
    const auto negative =
        level_set::projectLevelSetP1PhaseIndicatorFromCutContext(
            fixture.system, fixture.indicator, graph, options);
    ASSERT_TRUE(negative.success) << negative.diagnostic;
    options.liquid_side = FE::geometry::CutIntegrationSide::Positive;
    const auto positive =
        level_set::projectLevelSetP1PhaseIndicatorFromCutContext(
            fixture.system, fixture.indicator, graph, options);
    ASSERT_TRUE(positive.success) << positive.diagnostic;

    EXPECT_GT(negative.retained_liquid_measure, 0.0);
    EXPECT_GT(positive.retained_liquid_measure, 0.0);
    EXPECT_NEAR(negative.retained_liquid_measure +
                    positive.retained_liquid_measure,
                graph.physical_measure, 2.0e-13);
    ASSERT_EQ(negative.liquid_indicator.size(), graph.nodes);
    ASSERT_EQ(positive.liquid_indicator.size(), graph.nodes);
    for (std::size_t i = 0; i < graph.nodes; ++i) {
        EXPECT_GE(negative.liquid_indicator[i], 0.0);
        EXPECT_LE(negative.liquid_indicator[i], 1.0);
        EXPECT_NEAR(negative.liquid_indicator[i] +
                        positive.liquid_indicator[i],
                    1.0, 2.0e-13);
        EXPECT_NEAR(negative.liquid_phase_mass[i] +
                        positive.liquid_phase_mass[i],
                    graph.lumped_control_volume[i], 2.0e-13);
    }
}

} // namespace
