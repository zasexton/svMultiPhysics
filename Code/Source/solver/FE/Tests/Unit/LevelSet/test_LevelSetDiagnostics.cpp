#include "LevelSet/LevelSetDiagnostics.h"

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
#include <string_view>
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

[[nodiscard]] std::vector<FE::Real> distortedPlaneCoefficients(
    const ScalarFieldFixture& fixture)
{
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("distortedPlaneCoefficients: field has no entity DOF map");
    }

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error("distortedPlaneCoefficients: expected one vertex DOF");
        }
        const auto x = fixture.mesh->getNodeCoordinates(vertex);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            FE::Real{4.0} * (x[0] - FE::Real{0.25});
    }
    return coefficients;
}

[[nodiscard]] const FE::Real* findScalarDiagnostic(
    const std::vector<level_set::LevelSetScalarDiagnostic>& scalars,
    std::string_view name)
{
    for (const auto& scalar : scalars) {
        if (scalar.name == name) {
            return &scalar.value;
        }
    }
    return nullptr;
}

} // namespace

TEST(LevelSetDiagnostics, ReportsVolumeAndReferenceVolumeLoss)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto distorted = distortedPlaneCoefficients(fixture);

    level_set::LevelSetOutputDiagnosticsOptions options{};
    options.has_reference_negative_volume = true;
    options.reference_negative_volume = 0.125;

    const auto result = level_set::computeLevelSetOutputDiagnostics(
        *fixture.mesh,
        field_dofs,
        options,
        distorted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.volume.success) << result.volume.diagnostic;
    EXPECT_NEAR(result.negative_volume_loss,
                options.reference_negative_volume - result.volume.negative_volume,
                1.0e-12);

    const auto* negative_volume =
        findScalarDiagnostic(result.scalars, "level_set.negative_volume");
    const auto* volume_loss =
        findScalarDiagnostic(result.scalars, "level_set.negative_volume_loss");
    ASSERT_NE(negative_volume, nullptr);
    ASSERT_NE(volume_loss, nullptr);
    EXPECT_NEAR(*negative_volume, result.volume.negative_volume, 1.0e-12);
    EXPECT_NEAR(*volume_loss, result.negative_volume_loss, 1.0e-12);
    EXPECT_EQ(result.scalars.size(), 6u);
}

TEST(LevelSetDiagnostics, FESystemOverloadReportsTheFieldSliceVolume)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = distortedPlaneCoefficients(fixture);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 2.0);
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    level_set::LevelSetOutputDiagnosticsOptions options{};

    const auto result = level_set::computeLevelSetOutputDiagnostics(
        fixture.system,
        fixture.phi,
        options,
        solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    const auto direct = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        fixture.system.fieldDofHandler(fixture.phi),
        level_set::LevelSetVolumeOptions{},
        coefficients);
    ASSERT_TRUE(direct.success) << direct.diagnostic;
    EXPECT_DOUBLE_EQ(result.volume.negative_volume, direct.negative_volume);
    EXPECT_DOUBLE_EQ(result.volume.positive_volume, direct.positive_volume);
    ASSERT_NE(findScalarDiagnostic(result.scalars, "level_set.total_volume"), nullptr);
    EXPECT_EQ(findScalarDiagnostic(result.scalars, "level_set.negative_volume_loss"),
              nullptr);
}

TEST(LevelSetDiagnostics, PureTangentialAdvectionPreservesVolumeDiagnostics)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);

    const std::array<FE::Real, 3> velocity{0.0, 0.35, -0.20};
    const auto exact_advected_plane = [&](FE::Real time) {
        std::vector<FE::Real> coefficients(
            static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            const auto x = fixture.mesh->getNodeCoordinates(vertex);
            const std::array<FE::Real, 3> departure{
                x[0] - velocity[0] * time,
                x[1] - velocity[1] * time,
                x[2] - velocity[2] * time,
            };
            coefficients[static_cast<std::size_t>(dofs.front())] =
                departure[0] - FE::Real(0.25);
        }
        return coefficients;
    };

    const auto initial = exact_advected_plane(0.0);
    const auto advected = exact_advected_plane(2.0);
    const auto initial_volume = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        initial);
    ASSERT_TRUE(initial_volume.success) << initial_volume.diagnostic;

    level_set::LevelSetOutputDiagnosticsOptions options{};
    options.has_reference_negative_volume = true;
    options.reference_negative_volume = initial_volume.negative_volume;

    const auto diagnostics = level_set::computeLevelSetOutputDiagnostics(
        *fixture.mesh,
        field_dofs,
        options,
        advected);

    ASSERT_TRUE(diagnostics.success) << diagnostics.diagnostic;
    EXPECT_NEAR(diagnostics.volume.negative_volume,
                initial_volume.negative_volume,
                1.0e-12);
    EXPECT_NEAR(diagnostics.negative_volume_loss, 0.0, 1.0e-12);
    EXPECT_NEAR(diagnostics.relative_negative_volume_loss, 0.0, 1.0e-12);
    const auto* volume_loss =
        findScalarDiagnostic(diagnostics.scalars, "level_set.negative_volume_loss");
    ASSERT_NE(volume_loss, nullptr);
    EXPECT_NEAR(*volume_loss, 0.0, 1.0e-12);
}

