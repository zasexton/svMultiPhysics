#include "LevelSet/LevelSetVolume.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"
#include "Systems/TimeIntegrator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SingleTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.0, 1.0},
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra4;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords = nodes_;
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

[[nodiscard]] FE::systems::SetupInputs makeSingleTetraSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

struct ScalarFieldFixture {
    std::shared_ptr<SingleTetraMeshAccess> mesh{};
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};

    ScalarFieldFixture()
        : mesh(std::make_shared<SingleTetraMeshAccess>()),
          system(mesh)
    {
        auto scalar_space =
            FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        system.setup({}, makeSingleTetraSetupInputs());
    }
};

[[nodiscard]] std::vector<FE::Real> planeCoefficients(const ScalarFieldFixture& fixture,
                                                      FE::Real offset)
{
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("planeCoefficients: field has no entity DOF map");
    }

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error("planeCoefficients: expected one vertex DOF");
        }
        const auto x = fixture.mesh->getNodeCoordinates(vertex);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] + x[1] + x[2] - offset;
    }
    return coefficients;
}

} // namespace

TEST(LevelSetVolume, CutCellVolumeUsesGeneratedInterfaceFractions)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});

    level_set::LevelSetVolumeOptions volume_opts{};
    volume_opts.tolerance = 1.0e-12;
    const auto result = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        volume_opts,
        coefficients);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cells, 1u);
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_EQ(result.full_negative_cells, 0u);
    EXPECT_EQ(result.full_positive_cells, 0u);
    EXPECT_NEAR(result.total_volume, 1.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(result.negative_volume, 1.0 / 48.0, 1.0e-12);
    EXPECT_NEAR(result.positive_volume, 7.0 / 48.0, 1.0e-12);
}

TEST(LevelSetVolume, FESystemOverloadUsesFieldSlice)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 2.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    const auto result = level_set::computeLevelSetCutCellVolume(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_NEAR(result.negative_volume, 1.0 / 48.0, 1.0e-12);
}

TEST(LevelSetVolume, HandlesUncutCells)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), -1.0);

    const auto result = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        coefficients);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cells, 1u);
    EXPECT_EQ(result.cut_cells, 0u);
    EXPECT_EQ(result.full_negative_cells, 1u);
    EXPECT_EQ(result.full_positive_cells, 0u);
    EXPECT_NEAR(result.total_volume, 1.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(result.negative_volume, 1.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(result.positive_volume, 0.0, 1.0e-12);
}

TEST(LevelSetVolume, GlobalShiftCorrectionMatchesTargetVolume)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.iterations, 0);
    EXPECT_NEAR(result.applied_shift, 0.25, 1.0e-8);
    EXPECT_NEAR(result.initial_negative_volume, 1.0 / 48.0, 1.0e-12);
    EXPECT_GT(std::abs(result.initial_negative_volume -
                       result.target_negative_volume),
              correction_opts.volume_tolerance);
    EXPECT_NEAR(result.corrected_negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
    EXPECT_NEAR(result.volume_error, 0.0, correction_opts.volume_tolerance);
    ASSERT_EQ(corrected.size(), coefficients.size());
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        EXPECT_NEAR(corrected[i], coefficients[i] + result.applied_shift, 1.0e-12);
    }
}

TEST(LevelSetVolume, GlobalShiftCorrectionLeavesMatchedVolumeUnchanged)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), -1.0);

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 6.0;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.iterations, 0);
    EXPECT_DOUBLE_EQ(result.applied_shift, 0.0);
    EXPECT_NEAR(result.volume_error, 0.0, correction_opts.volume_tolerance);
    EXPECT_EQ(corrected, coefficients);
}

TEST(LevelSetVolume, VolumeCorrectionUpdatesOutputTimeActiveVolume)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 0.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    const auto initial_volume = level_set::computeLevelSetCutCellVolume(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        solution);
    ASSERT_TRUE(initial_volume.success) << initial_volume.diagnostic;

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    const auto output_time_volume = level_set::computeLevelSetCutCellVolume(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        corrected_solution);
    ASSERT_TRUE(output_time_volume.success) << output_time_volume.diagnostic;

    EXPECT_NEAR(initial_volume.negative_volume, 1.0 / 48.0, 1.0e-12);
    EXPECT_NEAR(output_time_volume.negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
    EXPECT_NE(output_time_volume.negative_volume, initial_volume.negative_volume);
}

TEST(LevelSetVolume, VolumeCorrectionSynchronizesHistoryAndCutContext)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 0.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    const auto accepted_solution = corrected_solution;
    const auto previous_solution = corrected_solution;
    ASSERT_EQ(accepted_solution.size(), previous_solution.size());
    for (std::size_t i = 0; i < accepted_solution.size(); ++i) {
        EXPECT_NEAR(accepted_solution[i], previous_solution[i], 1.0e-15);
    }

    level_set::LevelSetGeneratedInterfaceOptions interface_opts{};
    interface_opts.level_set_field_name = "phi";
    interface_opts.domain_id = "maintained-fluid";
    interface_opts.requested_interface_marker = 812;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto accepted_context =
        lifecycle.build(fixture.system, interface_opts, accepted_solution);
    const auto previous_context =
        lifecycle.build(fixture.system, interface_opts, previous_solution);

    ASSERT_TRUE(accepted_context.success) << accepted_context.diagnostic;
    ASSERT_TRUE(previous_context.success) << previous_context.diagnostic;
    EXPECT_EQ(previous_context.interface_marker, accepted_context.interface_marker);
    EXPECT_EQ(previous_context.domain.marker(), accepted_context.domain.marker());
    EXPECT_EQ(previous_context.value_revision, accepted_context.value_revision + 1u);
    EXPECT_NEAR(accepted_context.summary.negative_volume_measure,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
    EXPECT_NEAR(previous_context.summary.negative_volume_measure,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
}

TEST(LevelSetVolume, MaintainedPreviousStateLeavesBDF1ResidualNeutral)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 0.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    const auto accepted_solution = corrected_solution;
    const auto previous_solution = corrected_solution;
    auto older_history = solution;
    for (auto& value : older_history) {
        value -= FE::Real{10.0};
    }

    std::array<std::span<const FE::Real>, 2> history_spans = {
        std::span<const FE::Real>(previous_solution.data(), previous_solution.size()),
        std::span<const FE::Real>(older_history.data(), older_history.size()),
    };
    const std::array<double, 2> dt_history = {0.1, 0.1};
    FE::systems::SystemStateView state{};
    state.dt = 0.1;
    state.dt_prev = 0.1;
    state.u = std::span<const FE::Real>(
        accepted_solution.data(), accepted_solution.size());
    state.u_prev = history_spans[0];
    state.u_prev2 = history_spans[1];
    state.u_history = std::span<const std::span<const FE::Real>>(
        history_spans.data(), history_spans.size());
    state.dt_history = std::span<const double>(
        dt_history.data(), dt_history.size());

    const FE::systems::BDFIntegrator bdf1(1);
    const auto context = bdf1.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(context.dt1.has_value());
    ASSERT_EQ(context.dt1->a.size(), 2u);

    for (std::size_t i = 0; i < accepted_solution.size(); ++i) {
        const auto derivative =
            context.dt1->a[0] * accepted_solution[i] +
            context.dt1->a[1] * previous_solution[i];
        EXPECT_NEAR(derivative, 0.0, 1.0e-12);
    }

    auto alternate_older_history = older_history;
    for (auto& value : alternate_older_history) {
        value += FE::Real{25.0};
    }
    history_spans[1] = std::span<const FE::Real>(
        alternate_older_history.data(), alternate_older_history.size());
    state.u_prev2 = history_spans[1];
    state.u_history = std::span<const std::span<const FE::Real>>(
        history_spans.data(), history_spans.size());
    const auto alternate_context =
        bdf1.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(alternate_context.dt1.has_value());
    ASSERT_EQ(alternate_context.dt1->a.size(), context.dt1->a.size());
    EXPECT_EQ(alternate_context.dt1->a, context.dt1->a);
}
