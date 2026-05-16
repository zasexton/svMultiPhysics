#include "LevelSet/LevelSetReinitialization.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

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

class SingleQuadMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleQuadMeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Quad4;
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

[[nodiscard]] FE::systems::SetupInputs makeSingleQuadSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

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

struct QuadScalarFieldFixture {
    std::shared_ptr<SingleQuadMeshAccess> mesh{};
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};

    QuadScalarFieldFixture()
        : mesh(std::make_shared<SingleQuadMeshAccess>()),
          system(mesh)
    {
        auto scalar_space =
            FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        system.setup({}, makeSingleQuadSetupInputs());
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

[[nodiscard]] std::vector<FE::Real> signedDistancePlaneCoefficients(
    const ScalarFieldFixture& fixture)
{
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("signedDistancePlaneCoefficients: field has no entity DOF map");
    }

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error("signedDistancePlaneCoefficients: expected one vertex DOF");
        }
        const auto x = fixture.mesh->getNodeCoordinates(vertex);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] - FE::Real{0.25};
    }
    return coefficients;
}

[[nodiscard]] std::vector<FE::Real> verticalSignedDistanceCoefficients(
    const QuadScalarFieldFixture& fixture)
{
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("verticalSignedDistanceCoefficients: field has no entity DOF map");
    }

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error("verticalSignedDistanceCoefficients: expected one vertex DOF");
        }
        const auto x = fixture.mesh->getNodeCoordinates(vertex);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] - FE::Real{0.25};
    }
    return coefficients;
}

[[nodiscard]] FE::Real vertexValue(const FE::dofs::EntityDofMap& entity_map,
                                   const std::vector<FE::Real>& coefficients,
                                   FE::GlobalIndex vertex)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::runtime_error("vertexValue: expected one vertex DOF");
    }
    return coefficients[static_cast<std::size_t>(dofs.front())];
}

} // namespace

TEST(LevelSetReinitialization, ProjectionRepairsNodalField)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto distorted = distortedPlaneCoefficients(fixture);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        options,
        distorted,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.method, level_set::LevelSetReinitializationMethod::Projection);
    EXPECT_EQ(result.repaired_dofs, 4u);
    EXPECT_EQ(result.interface_fragments, 1u);
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_EQ(result.interface_displacement_samples, 3u);
    EXPECT_GT(result.max_abs_update, 0.0);
    EXPECT_GT(result.max_interface_displacement, 0.0);
    EXPECT_GT(result.l2_interface_displacement, 0.0);
    EXPECT_LE(result.max_interface_displacement, result.max_abs_update);

    EXPECT_NEAR(vertexValue(*entity_map, repaired, 0), -0.25, 1.0e-12);
    EXPECT_NEAR(vertexValue(*entity_map, repaired, 1), 0.75, 1.0e-12);
    EXPECT_NEAR(vertexValue(*entity_map, repaired, 2), -std::sqrt(0.125), 1.0e-12);
    EXPECT_NEAR(vertexValue(*entity_map, repaired, 3), -std::sqrt(0.125), 1.0e-12);
}

TEST(LevelSetReinitialization, ProjectionReportsInterfaceMovementBeyondTolerance)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto signed_distance = signedDistancePlaneCoefficients(fixture);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        options,
        signed_distance,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_EQ(result.interface_displacement_samples, 4u);
    EXPECT_GT(result.max_interface_displacement,
              options.signed_distance_tolerance);
    EXPECT_NEAR(result.max_interface_displacement,
                std::sqrt(0.125) - 0.25,
                1.0e-12);
    ASSERT_EQ(repaired.size(), signed_distance.size());
    EXPECT_NEAR(vertexValue(*entity_map, repaired, 0), -0.25, 1.0e-12);
    EXPECT_NEAR(vertexValue(*entity_map, repaired, 1), 0.75, 1.0e-12);
    EXPECT_NEAR(vertexValue(*entity_map, repaired, 2), -std::sqrt(0.125), 1.0e-12);
    EXPECT_NEAR(vertexValue(*entity_map, repaired, 3), -std::sqrt(0.125), 1.0e-12);
}

TEST(LevelSetReinitialization, ProjectionPreservesZeroContourWithinTolerance)
{
    const QuadScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto signed_distance = verticalSignedDistanceCoefficients(fixture);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        options,
        signed_distance,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(repaired.size(), signed_distance.size());
    const auto tolerance_gate =
        std::max(options.signed_distance_tolerance, FE::Real{0.05});
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_EQ(result.interface_displacement_samples, 4u);
    EXPECT_LE(result.max_interface_displacement, tolerance_gate);
    EXPECT_LE(result.l2_interface_displacement, tolerance_gate);
    EXPECT_NEAR(result.max_interface_displacement, 0.0, 1.0e-12);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        EXPECT_NEAR(vertexValue(*entity_map, repaired, vertex),
                    vertexValue(*entity_map, signed_distance, vertex),
                    1.0e-12);
    }
}

TEST(LevelSetReinitialization, FESystemOverloadRepairsFieldSlice)
{
    const ScalarFieldFixture fixture;
    const auto distorted = distortedPlaneCoefficients(fixture);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 2.0);
    std::copy(distorted.begin(),
              distorted.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;

    std::vector<FE::Real> repaired_solution;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        fixture.system,
        fixture.phi,
        options,
        solution,
        repaired_solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(repaired_solution.size(), solution.size());
    EXPECT_NE(repaired_solution, solution);
}

TEST(LevelSetReinitialization, ProjectionReportsMissingInterface)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    std::vector<FE::Real> input(static_cast<std::size_t>(field_dofs.getNumDofs()), 1.0);

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetReinitializationOptions{},
        input,
        repaired);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.interface_fragments, 0u);
    EXPECT_EQ(repaired, input);
}
