#include "LevelSet/LevelSetConservativePhaseOperator.h"

#include "Assembly/Assembler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class StructuredQuadPhaseMeshAccess final
    : public FE::assembly::IMeshAccess {
public:
    using FE::assembly::IMeshAccess::getCellCoordinates;

    explicit StructuredQuadPhaseMeshAccess(std::size_t cells_per_axis)
        : cells_per_axis_(cells_per_axis)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(
            cells_per_axis_ * cells_per_axis_);
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        const auto nodes_per_axis = cells_per_axis_ + 1u;
        return static_cast<FE::GlobalIndex>(
            nodes_per_axis * nodes_per_axis);
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return 0;
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override
    {
        return 0;
    }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] bool isOwnedCell(
        FE::GlobalIndex cell) const override
    {
        return cell >= 0 && cell < numCells();
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex /*cell*/) const override
    {
        return FE::ElementType::Quad4;
    }
    void getCellNodes(
        FE::GlobalIndex cell,
        std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto cell_index = static_cast<std::size_t>(cell);
        const auto i = cell_index % cells_per_axis_;
        const auto j = cell_index / cells_per_axis_;
        nodes = {
            node(i, j),
            node(i + 1u, j),
            node(i + 1u, j + 1u),
            node(i, j + 1u),
        };
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        const auto index = static_cast<std::size_t>(node_id);
        const auto nodes_per_axis = cells_per_axis_ + 1u;
        const auto i = index % nodes_per_axis;
        const auto j = index / nodes_per_axis;
        const FE::Real spacing = FE::Real{1.0} /
                                 static_cast<FE::Real>(cells_per_axis_);
        return {
            spacing * static_cast<FE::Real>(i),
            spacing * static_cast<FE::Real>(j),
            FE::Real{0.0},
        };
    }
    void getCellCoordinates(
        FE::GlobalIndex cell,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        std::vector<FE::GlobalIndex> nodes;
        getCellNodes(cell, nodes);
        coordinates.clear();
        coordinates.reserve(nodes.size());
        for (const auto node_id : nodes) {
            coordinates.push_back(getNodeCoordinates(node_id));
        }
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face*/,
        FE::GlobalIndex /*cell*/) const override
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
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
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
    [[nodiscard]] FE::GlobalIndex node(
        std::size_t i,
        std::size_t j) const
    {
        return static_cast<FE::GlobalIndex>(
            j * (cells_per_axis_ + 1u) + i);
    }

    std::size_t cells_per_axis_{0u};
};

[[nodiscard]] FE::systems::SetupInputs structuredSetupInputs(
    std::size_t cells_per_axis)
{
    const auto nodes_per_axis = cells_per_axis + 1u;
    const auto cell_count = cells_per_axis * cells_per_axis;
    const auto node_count = nodes_per_axis * nodes_per_axis;
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = static_cast<FE::GlobalIndex>(cell_count);
    topology.n_vertices = static_cast<FE::GlobalIndex>(node_count);
    topology.dim = 2;
    topology.cell2vertex_offsets.resize(cell_count + 1u, 0);
    topology.cell2vertex_data.reserve(4u * cell_count);
    topology.cell_gids.resize(cell_count);
    topology.cell_owner_ranks.assign(cell_count, 0);
    for (std::size_t j = 0u; j < cells_per_axis; ++j) {
        for (std::size_t i = 0u; i < cells_per_axis; ++i) {
            const auto cell = j * cells_per_axis + i;
            const auto lower_left = j * nodes_per_axis + i;
            topology.cell2vertex_offsets[cell] =
                static_cast<FE::MeshOffset>(4u * cell);
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(lower_left));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(lower_left + 1u));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(lower_left + nodes_per_axis + 1u));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(lower_left + nodes_per_axis));
            topology.cell_gids[cell] =
                static_cast<FE::dofs::gid_t>(cell);
        }
    }
    topology.cell2vertex_offsets[cell_count] =
        static_cast<FE::MeshOffset>(4u * cell_count);
    topology.vertex_gids.resize(node_count);
    for (std::size_t node = 0u; node < node_count; ++node) {
        topology.vertex_gids[node] =
            static_cast<FE::dofs::gid_t>(node);
    }
    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

struct StructuredPhaseFixture {
    std::shared_ptr<StructuredQuadPhaseMeshAccess> mesh;
    FE::systems::FESystem system;
    FE::FieldId phase{FE::INVALID_FIELD_ID};
    level_set::LevelSetP1PhaseTransportGraph graph{};
    std::vector<std::array<FE::Real, 3>> node_coordinates{};

    explicit StructuredPhaseFixture(std::size_t cells_per_axis)
        : mesh(std::make_shared<StructuredQuadPhaseMeshAccess>(
              cells_per_axis)),
          system(mesh)
    {
        phase = system.addField(FE::systems::FieldSpec{
            .name = "liquid_indicator",
            .space = std::make_shared<FE::spaces::H1Space>(
                FE::ElementType::Quad4, /*order=*/1),
            .components = 1,
        });
        system.setup({}, structuredSetupInputs(cells_per_axis));
        graph = level_set::buildLevelSetP1PhaseTransportGraph(
            system, phase);
        if (!graph.success) {
            return;
        }
        node_coordinates.resize(graph.nodes);
        const auto* entity_map =
            system.fieldDofHandler(phase).getEntityDofMap();
        if (entity_map == nullptr) {
            graph.success = false;
            graph.diagnostic = "structured phase fixture has no entity map";
            return;
        }
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh->numVertices(); ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            if (dofs.size() != 1u || dofs.front() < 0 ||
                static_cast<std::size_t>(dofs.front()) >= graph.nodes) {
                graph.success = false;
                graph.diagnostic =
                    "structured phase fixture has an invalid vertex map";
                return;
            }
            node_coordinates[static_cast<std::size_t>(dofs.front())] =
                mesh->getNodeCoordinates(vertex);
        }
    }
};

[[nodiscard]] std::pair<std::vector<FE::Real>, std::vector<FE::Real>>
oneRingBounds(
    const level_set::LevelSetP1PhaseTransportGraph& graph,
    const std::vector<FE::Real>& phase)
{
    std::vector<FE::Real> lower = phase;
    std::vector<FE::Real> upper = phase;
    for (const auto& edge : graph.edges) {
        const auto first = static_cast<std::size_t>(edge.first_node);
        const auto second = static_cast<std::size_t>(edge.second_node);
        lower[first] = std::min(lower[first], phase[second]);
        lower[second] = std::min(lower[second], phase[first]);
        upper[first] = std::max(upper[first], phase[second]);
        upper[second] = std::max(upper[second], phase[first]);
    }
    return {std::move(lower), std::move(upper)};
}

using PhaseInitializer = std::function<FE::Real(
    const std::array<FE::Real, 3>&)>;
using VelocityField = std::function<std::array<FE::Real, 3>(
    FE::Real, const std::array<FE::Real, 3>&)>;

struct TransportRun {
    bool success{false};
    std::string diagnostic{};
    std::vector<FE::Real> initial_phase{};
    std::vector<FE::Real> final_phase{};
    FE::Real initial_measure{0.0};
    FE::Real final_measure{0.0};
    FE::Real maximum_measure_error{0.0};
    FE::Real cumulative_boundary_transfer{0.0};
    FE::Real cumulative_divergence_source{0.0};
    FE::Real maximum_accounted_balance_error{0.0};
    FE::Real minimum_indicator{1.0};
    FE::Real maximum_indicator{0.0};
    FE::Real maximum_courant{0.0};
    FE::Real maximum_local_balance_residual{0.0};
    FE::Real maximum_component_balance_residual{0.0};
    std::size_t minimum_components{std::numeric_limits<std::size_t>::max()};
    std::size_t maximum_components{0u};
};

[[nodiscard]] FE::Real phaseMeasure(
    const level_set::LevelSetP1PhaseTransportGraph& graph,
    const std::vector<FE::Real>& phase)
{
    long double measure = 0.0L;
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
        measure += static_cast<long double>(
            graph.lumped_control_volume[node] * phase[node]);
    }
    return static_cast<FE::Real>(measure);
}

[[nodiscard]] TransportRun runTransport(
    const StructuredPhaseFixture& fixture,
    const PhaseInitializer& initialize,
    const VelocityField& velocity_field,
    FE::Real final_time,
    int steps,
    FE::Real component_activity_tolerance = 1.0e-8)
{
    TransportRun run;
    if (!fixture.graph.success || steps <= 0 ||
        !(final_time > FE::Real{0.0})) {
        run.diagnostic = fixture.graph.success
            ? "transport run has an invalid horizon"
            : fixture.graph.diagnostic;
        return run;
    }
    run.initial_phase.resize(fixture.graph.nodes, FE::Real{0.0});
    for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
        run.initial_phase[node] = std::clamp(
            initialize(fixture.node_coordinates[node]),
            FE::Real{0.0}, FE::Real{1.0});
    }
    run.final_phase = run.initial_phase;
    run.initial_measure = phaseMeasure(fixture.graph, run.initial_phase);
    const FE::Real dt = final_time / static_cast<FE::Real>(steps);
    std::vector<std::array<FE::Real, 3>> velocity(fixture.graph.nodes);
    level_set::LevelSetP1PhaseStageOptions options;
    options.invariant_tolerance = 1.0e-12;
    options.component_activity_tolerance =
        component_activity_tolerance;
    options.maximum_courant = 0.8;
    long double cumulative_boundary_transfer = 0.0L;
    long double cumulative_divergence_source = 0.0L;
    for (int step = 0; step < steps; ++step) {
        const FE::Real stage_time =
            (static_cast<FE::Real>(step) + FE::Real{0.5}) * dt;
        for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
            velocity[node] = velocity_field(
                stage_time, fixture.node_coordinates[node]);
        }
        auto [lower, upper] = oneRingBounds(
            fixture.graph, run.final_phase);
        const auto stage =
            level_set::advanceLevelSetP1ConservativePhaseStage(
                fixture.graph,
                run.final_phase,
                lower,
                upper,
                velocity,
                dt,
                options);
        if (!stage.success) {
            run.diagnostic = stage.diagnostic;
            return run;
        }
        if (!stage.correction.interior_cancellation_satisfied ||
            !stage.correction.local_balance_satisfied ||
            !stage.correction.global_balance_satisfied ||
            !stage.correction.component_balance_satisfied ||
            !stage.correction.component_measure_closure_satisfied) {
            run.diagnostic =
                "transport run failed a phase-flux ledger invariant";
            return run;
        }
        for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
            run.final_phase[node] =
                stage.correction.nodes[node].limited_liquid_indicator;
        }
        const FE::Real measure = phaseMeasure(
            fixture.graph, run.final_phase);
        cumulative_boundary_transfer += static_cast<long double>(
            stage.correction.total_physical_boundary_mass_transfer);
        cumulative_divergence_source += static_cast<long double>(
            stage.correction.total_discrete_divergence_mass_source);
        const FE::Real accounted_measure = static_cast<FE::Real>(
            static_cast<long double>(run.initial_measure) +
            cumulative_boundary_transfer +
            cumulative_divergence_source);
        run.maximum_measure_error = std::max(
            run.maximum_measure_error,
            std::abs(measure - run.initial_measure));
        run.maximum_accounted_balance_error = std::max(
            run.maximum_accounted_balance_error,
            std::abs(measure - accounted_measure));
        run.minimum_indicator = std::min(
            run.minimum_indicator,
            stage.correction.minimum_limited_liquid_indicator);
        run.maximum_indicator = std::max(
            run.maximum_indicator,
            stage.correction.maximum_limited_liquid_indicator);
        run.maximum_courant = std::max(
            run.maximum_courant, stage.maximum_courant);
        run.maximum_local_balance_residual = std::max(
            run.maximum_local_balance_residual,
            stage.correction.maximum_local_mass_balance_residual);
        run.maximum_component_balance_residual = std::max(
            run.maximum_component_balance_residual,
            stage.correction.maximum_component_balance_residual);
        run.minimum_components = std::min(
            run.minimum_components, stage.correction.components.size());
        run.maximum_components = std::max(
            run.maximum_components, stage.correction.components.size());
    }
    run.final_measure = phaseMeasure(fixture.graph, run.final_phase);
    run.cumulative_boundary_transfer =
        static_cast<FE::Real>(cumulative_boundary_transfer);
    run.cumulative_divergence_source =
        static_cast<FE::Real>(cumulative_divergence_source);
    run.success = true;
    run.diagnostic = "ok";
    return run;
}

[[nodiscard]] FE::Real weightedL1Error(
    const StructuredPhaseFixture& fixture,
    const std::vector<FE::Real>& phase,
    const PhaseInitializer& exact)
{
    long double error = 0.0L;
    long double normalization = 0.0L;
    for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
        const FE::Real weight = fixture.graph.lumped_control_volume[node];
        error += static_cast<long double>(
            weight * std::abs(
                         phase[node] -
                         exact(fixture.node_coordinates[node])));
        normalization += static_cast<long double>(weight);
    }
    return static_cast<FE::Real>(error / normalization);
}

[[nodiscard]] FE::Real weightedL1Difference(
    const StructuredPhaseFixture& fixture,
    const std::vector<FE::Real>& first,
    const std::vector<FE::Real>& second)
{
    if (first.size() != fixture.graph.nodes ||
        second.size() != fixture.graph.nodes) {
        return std::numeric_limits<FE::Real>::infinity();
    }
    long double difference = 0.0L;
    long double normalization = 0.0L;
    for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
        const FE::Real weight = fixture.graph.lumped_control_volume[node];
        difference += static_cast<long double>(
            weight * std::abs(first[node] - second[node]));
        normalization += static_cast<long double>(weight);
    }
    return static_cast<FE::Real>(difference / normalization);
}

[[nodiscard]] FE::Real observedOrder(
    FE::Real coarse_error,
    FE::Real fine_error,
    FE::Real refinement_ratio = FE::Real{2.0})
{
    if (!(coarse_error > FE::Real{0.0}) ||
        !(fine_error > FE::Real{0.0}) ||
        !(refinement_ratio > FE::Real{1.0})) {
        return std::numeric_limits<FE::Real>::quiet_NaN();
    }
    return std::log(coarse_error / fine_error) /
           std::log(refinement_ratio);
}

[[nodiscard]] std::string serializeReal(FE::Real value)
{
    std::ostringstream stream;
    stream << std::setprecision(
                  std::numeric_limits<FE::Real>::max_digits10)
           << value;
    return stream.str();
}

[[nodiscard]] std::array<FE::Real, 2> phaseCentroid(
    const StructuredPhaseFixture& fixture,
    const std::vector<FE::Real>& phase)
{
    long double measure = 0.0L;
    std::array<long double, 2> first_moment{0.0L, 0.0L};
    for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
        const long double nodal_measure = static_cast<long double>(
            fixture.graph.lumped_control_volume[node] * phase[node]);
        measure += nodal_measure;
        first_moment[0] += nodal_measure *
                           fixture.node_coordinates[node][0];
        first_moment[1] += nodal_measure *
                           fixture.node_coordinates[node][1];
    }
    return {
        static_cast<FE::Real>(first_moment[0] / measure),
        static_cast<FE::Real>(first_moment[1] / measure),
    };
}

[[nodiscard]] PhaseInitializer disk(
    FE::Real center_x,
    FE::Real center_y,
    FE::Real radius)
{
    return [=](const std::array<FE::Real, 3>& point) {
        const FE::Real dx = point[0] - center_x;
        const FE::Real dy = point[1] - center_y;
        return dx * dx + dy * dy <= radius * radius
            ? FE::Real{1.0}
            : FE::Real{0.0};
    };
}

void expectConservativeRun(const TransportRun& run)
{
    ASSERT_TRUE(run.success) << run.diagnostic;
    EXPECT_LE(run.maximum_accounted_balance_error, 2.0e-11);
    EXPECT_GE(run.minimum_indicator, -2.0e-12);
    EXPECT_LE(run.maximum_indicator, 1.0 + 2.0e-12);
    EXPECT_LE(run.maximum_courant, 0.8 + 2.0e-12);
    EXPECT_LE(run.maximum_local_balance_residual, 2.0e-12);
    EXPECT_LE(run.maximum_component_balance_residual, 2.0e-12);
}

TEST(LevelSetConservativePhaseBenchmarks,
     TranslatingDiskConservesAndRefines)
{
    constexpr FE::Real speed = 0.2;
    constexpr FE::Real final_time = 0.5;
    std::vector<FE::Real> errors;
    std::vector<FE::Real> centroid_errors;
    for (const std::size_t cells_per_axis : {16u, 32u, 64u}) {
        StructuredPhaseFixture fixture(cells_per_axis);
        ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
        const auto initial = disk(0.30, 0.50, 0.14);
        const auto exact = disk(
            0.30 + speed * final_time, 0.50, 0.14);
        const auto run = runTransport(
            fixture,
            initial,
            [](FE::Real /*time*/,
               const std::array<FE::Real, 3>& /*point*/) {
                return std::array<FE::Real, 3>{speed, 0.0, 0.0};
            },
            final_time,
            static_cast<int>(5u * cells_per_axis));
        expectConservativeRun(run);
        EXPECT_LE(run.maximum_measure_error, 1.0e-8);
        errors.push_back(weightedL1Error(
            fixture, run.final_phase, exact));
        const auto initial_centroid = phaseCentroid(
            fixture, run.initial_phase);
        const auto final_centroid = phaseCentroid(
            fixture, run.final_phase);
        centroid_errors.push_back(std::abs(
            final_centroid[0] - initial_centroid[0] -
            speed * final_time));
        const std::string suffix = "_N" +
                                   std::to_string(cells_per_axis);
        RecordProperty("coupled_l1" + suffix,
                       serializeReal(errors.back()));
        RecordProperty("coupled_centroid_error" + suffix,
                       serializeReal(centroid_errors.back()));
        RecordProperty("coupled_measure_error" + suffix,
                       serializeReal(run.maximum_measure_error));
        RecordProperty("coupled_accounted_balance_error" + suffix,
                       serializeReal(run.maximum_accounted_balance_error));
        EXPECT_EQ(run.minimum_components, 1u);
        EXPECT_EQ(run.maximum_components, 1u);
    }
    ASSERT_EQ(errors.size(), 3u);
    EXPECT_LT(errors[1], errors[0]);
    EXPECT_LT(errors[2], errors[1]);
    EXPECT_LT(centroid_errors[1], centroid_errors[0]);
    EXPECT_LT(centroid_errors[2], centroid_errors[1]);
    EXPECT_LT(errors.back(), 0.08);
    EXPECT_LT(centroid_errors.back(), 0.01);
}

TEST(LevelSetConservativePhaseBenchmarks,
     TranslatingDiskSeparatesSpaceAndTimeRefinement)
{
    constexpr FE::Real speed = 0.2;
    constexpr FE::Real final_time = 0.5;
    const auto initial = disk(0.30, 0.50, 0.14);
    const auto exact = disk(
        0.30 + speed * final_time, 0.50, 0.14);
    const VelocityField velocity = [](
        FE::Real /*time*/,
        const std::array<FE::Real, 3>& /*point*/) {
        return std::array<FE::Real, 3>{speed, 0.0, 0.0};
    };

    constexpr int fixed_space_steps = 64;
    std::vector<FE::Real> space_errors;
    std::vector<FE::Real> space_centroid_errors;
    for (const std::size_t cells_per_axis : {16u, 32u, 64u}) {
        StructuredPhaseFixture fixture(cells_per_axis);
        ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
        const auto run = runTransport(
            fixture,
            initial,
            velocity,
            final_time,
            fixed_space_steps);
        expectConservativeRun(run);
        const FE::Real error = weightedL1Error(
            fixture, run.final_phase, exact);
        const auto initial_centroid = phaseCentroid(
            fixture, run.initial_phase);
        const auto final_centroid = phaseCentroid(
            fixture, run.final_phase);
        const FE::Real centroid_error = std::abs(
            final_centroid[0] - initial_centroid[0] -
            speed * final_time);
        space_errors.push_back(error);
        space_centroid_errors.push_back(centroid_error);
        const std::string suffix = "_N" +
                                   std::to_string(cells_per_axis);
        RecordProperty("space_l1" + suffix, serializeReal(error));
        RecordProperty("space_centroid_error" + suffix,
                       serializeReal(centroid_error));
        RecordProperty("space_measure_error" + suffix,
                       serializeReal(run.maximum_measure_error));
        RecordProperty("space_accounted_balance_error" + suffix,
                       serializeReal(run.maximum_accounted_balance_error));
        RecordProperty("space_maximum_courant" + suffix,
                       serializeReal(run.maximum_courant));
    }
    ASSERT_EQ(space_errors.size(), 3u);
    EXPECT_LT(space_errors[1], space_errors[0]);
    EXPECT_LT(space_errors[2], space_errors[1]);
    EXPECT_LT(space_centroid_errors[1], space_centroid_errors[0]);
    EXPECT_LT(space_centroid_errors[2], space_centroid_errors[1]);
    const FE::Real first_space_order = observedOrder(
        space_errors[0], space_errors[1]);
    const FE::Real second_space_order = observedOrder(
        space_errors[1], space_errors[2]);
    RecordProperty("space_l1_order_16_to_32",
                   serializeReal(first_space_order));
    RecordProperty("space_l1_order_32_to_64",
                   serializeReal(second_space_order));
    EXPECT_GT(first_space_order, 0.35);
    EXPECT_GT(second_space_order, 0.35);

    StructuredPhaseFixture time_fixture(48u);
    ASSERT_TRUE(time_fixture.graph.success)
        << time_fixture.graph.diagnostic;
    const auto temporal_reference = runTransport(
        time_fixture,
        initial,
        velocity,
        final_time,
        256);
    expectConservativeRun(temporal_reference);
    std::vector<FE::Real> time_errors;
    for (const int steps : {32, 64, 128}) {
        const auto run = runTransport(
            time_fixture,
            initial,
            velocity,
            final_time,
            steps);
        expectConservativeRun(run);
        const FE::Real error = weightedL1Difference(
            time_fixture,
            run.final_phase,
            temporal_reference.final_phase);
        time_errors.push_back(error);
        const std::string suffix = "_steps" + std::to_string(steps);
        RecordProperty("time_reference_l1" + suffix,
                       serializeReal(error));
        RecordProperty("time_measure_error" + suffix,
                       serializeReal(run.maximum_measure_error));
        RecordProperty("time_accounted_balance_error" + suffix,
                       serializeReal(run.maximum_accounted_balance_error));
        RecordProperty("time_maximum_courant" + suffix,
                       serializeReal(run.maximum_courant));
    }
    ASSERT_EQ(time_errors.size(), 3u);
    EXPECT_LT(time_errors[1], time_errors[0]);
    EXPECT_LT(time_errors[2], time_errors[1]);
    const FE::Real first_time_order = observedOrder(
        time_errors[0], time_errors[1]);
    const FE::Real second_time_order = observedOrder(
        time_errors[1], time_errors[2]);
    RecordProperty("time_l1_order_32_to_64_steps",
                   serializeReal(first_time_order));
    RecordProperty("time_l1_order_64_to_128_steps",
                   serializeReal(second_time_order));
    EXPECT_GT(first_time_order, 0.5);
    EXPECT_GT(second_time_order, 0.5);
}

TEST(LevelSetConservativePhaseBenchmarks,
     RotatesASlottedDiskThroughOnePeriod)
{
    StructuredPhaseFixture fixture(32u);
    ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
    const PhaseInitializer slotted_disk = [](
        const std::array<FE::Real, 3>& point) {
        const FE::Real dx = point[0] - 0.5;
        const FE::Real dy = point[1] - 0.65;
        const bool in_disk = dx * dx + dy * dy <= 0.15 * 0.15;
        const bool in_slot = std::abs(dx) < 0.025 && point[1] >= 0.65;
        return in_disk && !in_slot ? FE::Real{1.0} : FE::Real{0.0};
    };
    const FE::Real final_time = FE::Real{2.0} *
                                std::numbers::pi_v<FE::Real>;
    const auto run = runTransport(
        fixture,
        slotted_disk,
        [](FE::Real /*time*/, const std::array<FE::Real, 3>& point) {
            return std::array<FE::Real, 3>{
                -(point[1] - FE::Real{0.5}),
                point[0] - FE::Real{0.5},
                FE::Real{0.0},
            };
        },
        final_time,
        1600);
    expectConservativeRun(run);
    EXPECT_LE(run.maximum_measure_error, 1.0e-4);
    EXPECT_GE(run.minimum_components, 1u);
    const FE::Real l1_error = weightedL1Error(
        fixture, run.final_phase, slotted_disk);
    RecordProperty("zalesak_l1", serializeReal(l1_error));
    RecordProperty("zalesak_measure_error",
                   serializeReal(run.maximum_measure_error));
    RecordProperty("zalesak_accounted_balance_error",
                   serializeReal(run.maximum_accounted_balance_error));
    RecordProperty("zalesak_minimum_components",
                   std::to_string(run.minimum_components));
    RecordProperty("zalesak_maximum_components",
                   std::to_string(run.maximum_components));
    EXPECT_LT(l1_error, 0.12);
}

TEST(LevelSetConservativePhaseBenchmarks,
     ReversibleDeformationReturnsWithoutMassDrift)
{
    StructuredPhaseFixture fixture(32u);
    ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
    const auto initial = disk(0.35, 0.50, 0.15);
    constexpr FE::Real final_time = 1.0;
    const auto run = runTransport(
        fixture,
        initial,
        [](FE::Real time, const std::array<FE::Real, 3>& point) {
            const FE::Real pi = std::numbers::pi_v<FE::Real>;
            const FE::Real amplitude = std::cos(pi * time);
            const FE::Real sin_x = std::sin(pi * point[0]);
            const FE::Real sin_y = std::sin(pi * point[1]);
            return std::array<FE::Real, 3>{
                amplitude * sin_x * sin_x *
                    std::sin(FE::Real{2.0} * pi * point[1]),
                -amplitude * sin_y * sin_y *
                    std::sin(FE::Real{2.0} * pi * point[0]),
                FE::Real{0.0},
            };
        },
        final_time,
        512);
    expectConservativeRun(run);
    EXPECT_LE(run.maximum_measure_error, 1.0e-8);
    EXPECT_EQ(run.minimum_components, 1u);
    EXPECT_EQ(run.maximum_components, 1u);
    const FE::Real l1_error = weightedL1Error(
        fixture, run.final_phase, initial);
    RecordProperty("deformation_l1", serializeReal(l1_error));
    RecordProperty("deformation_measure_error",
                   serializeReal(run.maximum_measure_error));
    RecordProperty("deformation_accounted_balance_error",
                   serializeReal(run.maximum_accounted_balance_error));
    EXPECT_LT(l1_error, 0.10);
}

TEST(LevelSetConservativePhaseBenchmarks,
     AdvectsAThinWallFilmTangentially)
{
    StructuredPhaseFixture fixture(32u);
    ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
    const PhaseInitializer initial = [](
        const std::array<FE::Real, 3>& point) {
        return point[0] >= 0.20 && point[0] <= 0.45 &&
                       point[1] <= 0.08
            ? FE::Real{1.0}
            : FE::Real{0.0};
    };
    constexpr FE::Real speed = 0.15;
    constexpr FE::Real final_time = 0.5;
    const PhaseInitializer exact = [](
        const std::array<FE::Real, 3>& point) {
        return point[0] >= 0.20 + speed * final_time &&
                       point[0] <= 0.45 + speed * final_time &&
                       point[1] <= 0.08
            ? FE::Real{1.0}
            : FE::Real{0.0};
    };
    const auto run = runTransport(
        fixture,
        initial,
        [](FE::Real /*time*/,
           const std::array<FE::Real, 3>& /*point*/) {
            return std::array<FE::Real, 3>{speed, 0.0, 0.0};
        },
        final_time,
        192);
    expectConservativeRun(run);
    EXPECT_LE(run.maximum_measure_error, 1.0e-7);
    EXPECT_EQ(run.minimum_components, 1u);
    EXPECT_EQ(run.maximum_components, 1u);
    const FE::Real l1_error = weightedL1Error(
        fixture, run.final_phase, exact);
    EXPECT_LT(l1_error, 0.06);
    const auto initial_centroid = phaseCentroid(
        fixture, run.initial_phase);
    const auto final_centroid = phaseCentroid(
        fixture, run.final_phase);
    EXPECT_NEAR(final_centroid[0] - initial_centroid[0],
                speed * final_time, 0.015);
    EXPECT_NEAR(final_centroid[1], initial_centroid[1], 0.002);
    RecordProperty("wall_film_l1", serializeReal(l1_error));
    RecordProperty("wall_film_measure_error",
                   serializeReal(run.maximum_measure_error));
    RecordProperty("wall_film_centroid_x_error",
                   serializeReal(std::abs(
                       final_centroid[0] - initial_centroid[0] -
                       speed * final_time)));
    RecordProperty("wall_film_centroid_y_error",
                   serializeReal(std::abs(
                       final_centroid[1] - initial_centroid[1])));
}

TEST(LevelSetConservativePhaseBenchmarks,
     KeepsSeparatedDropsInDistinctFluxComponents)
{
    StructuredPhaseFixture fixture(32u);
    ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
    const PhaseInitializer initial = [](
        const std::array<FE::Real, 3>& point) {
        const FE::Real first_x = point[0] - 0.25;
        const FE::Real second_x = point[0] - 0.65;
        const FE::Real y = point[1] - 0.50;
        return first_x * first_x + y * y <= 0.08 * 0.08 ||
                       second_x * second_x + y * y <= 0.08 * 0.08
            ? FE::Real{1.0}
            : FE::Real{0.0};
    };
    constexpr FE::Real speed = 0.16;
    constexpr FE::Real final_time = 0.5;
    const auto run = runTransport(
        fixture,
        initial,
        [](FE::Real /*time*/,
           const std::array<FE::Real, 3>& /*point*/) {
            return std::array<FE::Real, 3>{speed, 0.0, 0.0};
        },
        final_time,
        192,
        1.0e-2);
    expectConservativeRun(run);
    EXPECT_LE(run.maximum_measure_error, 1.0e-6);
    EXPECT_EQ(run.minimum_components, 2u);
    EXPECT_EQ(run.maximum_components, 2u);
    const auto initial_centroid = phaseCentroid(
        fixture, run.initial_phase);
    const auto final_centroid = phaseCentroid(
        fixture, run.final_phase);
    EXPECT_NEAR(final_centroid[0] - initial_centroid[0],
                speed * final_time, 0.015);
    EXPECT_NEAR(final_centroid[1], initial_centroid[1], 2.0e-12);
    RecordProperty("separated_drops_measure_error",
                   serializeReal(run.maximum_measure_error));
    RecordProperty("separated_drops_accounted_balance_error",
                   serializeReal(run.maximum_accounted_balance_error));
    RecordProperty("separated_drops_minimum_components",
                   std::to_string(run.minimum_components));
    RecordProperty("separated_drops_maximum_components",
                   std::to_string(run.maximum_components));
    RecordProperty("separated_drops_centroid_x_error",
                   serializeReal(std::abs(
                       final_centroid[0] - initial_centroid[0] -
                       speed * final_time)));
}

} // namespace
