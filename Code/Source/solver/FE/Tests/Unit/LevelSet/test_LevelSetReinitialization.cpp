#include "LevelSet/LevelSetReinitialization.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/SpaceFactory.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::shared_ptr<svmp::Mesh> buildNativeQuad9Mesh()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        0.5, 0.0,
        1.0, 0.5,
        0.5, 1.0,
        0.0, 0.5,
        0.5, 0.5,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 9};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    base->build_from_arrays(/*spatial_dim=*/2,
                            X_ref,
                            cell2vertex_offsets,
                            cell2vertex,
                            {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}
#endif

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

class SingleQuad9MeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleQuad9MeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.5, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.5, 0.0},
            std::array<FE::Real, 3>{0.5, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.5, 0.0},
            std::array<FE::Real, 3>{0.5, 0.5, 0.0},
        };
        cell_ = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Quad9;
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
    std::array<FE::GlobalIndex, 9> cell_{};
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

[[nodiscard]] FE::systems::SetupInputs makeSingleQuad9SetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 9;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 9};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
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

struct Quad9ScalarFieldFixture {
    std::shared_ptr<SingleQuad9MeshAccess> mesh{};
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};

    Quad9ScalarFieldFixture()
        : mesh(std::make_shared<SingleQuad9MeshAccess>()),
          system(mesh)
    {
        auto scalar_space =
            FE::spaces::Space(FE::spaces::SpaceType::H1, *mesh, /*order=*/2, /*components=*/1);
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        system.setup({}, makeSingleQuad9SetupInputs());
    }
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
struct NativeQuad9ScalarFieldFixture {
    std::shared_ptr<svmp::Mesh> mesh{};
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};

    NativeQuad9ScalarFieldFixture()
        : mesh(buildNativeQuad9Mesh()),
          system(mesh)
    {
        auto scalar_space =
            std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4, /*order=*/2);
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        system.setup();
    }
};
#endif

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

[[nodiscard]] std::vector<FE::Real> distortedPlaneCoefficients(
    const Quad9ScalarFieldFixture& fixture)
{
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    std::vector<FE::GlobalIndex> cell_nodes;
    fixture.mesh->getCellNodes(0, cell_nodes);
    if (cell_dofs.size() != cell_nodes.size()) {
        throw std::runtime_error("distortedPlaneCoefficients: expected nodal cell DOFs");
    }

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < cell_nodes.size(); ++i) {
        const auto x = fixture.mesh->getNodeCoordinates(cell_nodes[i]);
        coefficients[static_cast<std::size_t>(cell_dofs[i])] =
            FE::Real{4.0} * (x[0] - FE::Real{0.25});
    }
    return coefficients;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::vector<FE::Real> planeCoefficients(
    const NativeQuad9ScalarFieldFixture& fixture,
    FE::Real scale)
{
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    std::vector<FE::Real> mesh_values(fixture.mesh->n_vertices(), 0.0);
    for (std::size_t vertex = 0; vertex < fixture.mesh->n_vertices(); ++vertex) {
        const auto x =
            fixture.mesh->get_vertex_coords(static_cast<svmp::index_t>(vertex));
        mesh_values[vertex] = scale * (x[0] - FE::Real{0.25});
    }

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    std::vector<std::uint8_t> assigned(coefficients.size(), 0u);
    const auto projection =
        fixture.system.projectMeshVertexValuesToFieldCoefficients(
            fixture.phi,
            std::span<const FE::Real>(mesh_values.data(), mesh_values.size()),
            /*mesh_components=*/1,
            std::span<FE::Real>(coefficients.data(), coefficients.size()),
            std::span<std::uint8_t>(assigned.data(), assigned.size()),
            "LevelSetReinitialization test projection");
    if (projection.unassigned_dofs != 0u) {
        throw std::runtime_error(
            "planeCoefficients: native high-order projection left unassigned coefficients");
    }
    return coefficients;
}
#endif

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

[[nodiscard]] FE::Real cellNodeValue(const Quad9ScalarFieldFixture& fixture,
                                     const std::vector<FE::Real>& coefficients,
                                     std::size_t local_node)
{
    std::vector<FE::GlobalIndex> cell_nodes;
    fixture.mesh->getCellNodes(0, cell_nodes);
    if (local_node >= cell_nodes.size()) {
        throw std::runtime_error("cellNodeValue: local node out of range");
    }
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("cellNodeValue: expected entity DOF metadata");
    }
    return vertexValue(*entity_map, coefficients, cell_nodes[local_node]);
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

TEST(LevelSetReinitialization, GenericProjectionFailsClosedForHighOrderCellNodeDofs)
{
    const Quad9ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
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

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("without an entity-aware mesh-node binding"),
              std::string::npos);
    EXPECT_EQ(result.repaired_dofs, 4u);
    EXPECT_EQ(result.interface_fragments, 1u);
    EXPECT_EQ(result.cut_cells, 1u);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(LevelSetReinitialization, FESystemOverloadRepairsNativeHighOrderCellNodeDofs)
{
    const NativeQuad9ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto distorted = planeCoefficients(fixture, FE::Real{4.0});
    const auto expected = planeCoefficients(fixture, FE::Real{1.0});
    const auto offset = static_cast<std::size_t>(
        fixture.system.fieldDofOffset(fixture.phi));

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 2.0);
    std::copy(distorted.begin(),
              distorted.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    std::vector<FE::Real> repaired_solution;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        fixture.system,
        fixture.phi,
        options,
        solution,
        repaired_solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.repaired_dofs, 9u);
    EXPECT_EQ(result.interface_fragments, 1u);
    EXPECT_EQ(result.cut_cells, 1u);
    ASSERT_EQ(repaired_solution.size(), solution.size());
    ASSERT_EQ(expected.size(), static_cast<std::size_t>(field_dofs.getNumDofs()));

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(repaired_solution[offset + i], expected[i], 1.0e-12)
            << "coefficient " << i;
    }
}
#endif

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
