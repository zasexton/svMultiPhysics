#include "LevelSet/LevelSetVolume.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
    EXPECT_NEAR(result.corrected_negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
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
    EXPECT_EQ(corrected, coefficients);
}
