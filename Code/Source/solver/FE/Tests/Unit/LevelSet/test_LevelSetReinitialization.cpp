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
#include <optional>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::shared_ptr<svmp::Mesh> buildNativeTetra4Mesh()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/3,
                            X_ref,
                            cell2vertex_offsets,
                            cell2vertex,
                            {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

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

[[nodiscard]] std::shared_ptr<svmp::Mesh> buildNativeTwoQuadColumnMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        0.0, 2.0,
        1.0, 2.0,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 3, 2,
        2, 3, 5, 4,
    };

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2,
                            X_ref,
                            cell2vertex_offsets,
                            cell2vertex,
                            {shape, shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

[[nodiscard]] std::shared_ptr<svmp::Mesh> buildNativeStructuredQuadMesh(
    int subdivisions)
{
    if (subdivisions <= 0) {
        throw std::invalid_argument(
            "structured level-set mesh requires positive subdivisions");
    }

    auto base = std::make_shared<svmp::MeshBase>();
    const auto vertex_extent = static_cast<std::size_t>(subdivisions + 1);
    std::vector<svmp::real_t> X_ref;
    X_ref.reserve(2u * vertex_extent * vertex_extent);
    for (int row = 0; row <= subdivisions; ++row) {
        for (int column = 0; column <= subdivisions; ++column) {
            X_ref.push_back(
                static_cast<svmp::real_t>(column) / subdivisions);
            X_ref.push_back(
                static_cast<svmp::real_t>(row) / subdivisions);
        }
    }

    std::vector<svmp::offset_t> cell2vertex_offsets;
    std::vector<svmp::index_t> cell2vertex;
    cell2vertex_offsets.reserve(
        static_cast<std::size_t>(subdivisions * subdivisions) + 1u);
    cell2vertex.reserve(
        4u * static_cast<std::size_t>(subdivisions * subdivisions));
    cell2vertex_offsets.push_back(0);
    for (int row = 0; row < subdivisions; ++row) {
        for (int column = 0; column < subdivisions; ++column) {
            const auto lower_left = static_cast<svmp::index_t>(
                static_cast<std::size_t>(row) * vertex_extent +
                static_cast<std::size_t>(column));
            const auto lower_right = lower_left + 1;
            const auto upper_left =
                lower_left + static_cast<svmp::index_t>(vertex_extent);
            const auto upper_right = upper_left + 1;
            cell2vertex.insert(cell2vertex.end(),
                               {lower_left,
                                lower_right,
                                upper_right,
                                upper_left});
            cell2vertex_offsets.push_back(
                static_cast<svmp::offset_t>(cell2vertex.size()));
        }
    }

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    std::vector<svmp::CellShape> shapes(
        static_cast<std::size_t>(subdivisions * subdivisions), shape);
    base->build_from_arrays(/*spatial_dim=*/2,
                            X_ref,
                            cell2vertex_offsets,
                            cell2vertex,
                            shapes);
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
struct NativeLinearTetraP2ScalarFieldFixture {
    std::shared_ptr<svmp::Mesh> mesh{};
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};

    NativeLinearTetraP2ScalarFieldFixture()
        : mesh(buildNativeTetra4Mesh()),
          system(mesh)
    {
        auto scalar_space =
            std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, /*order=*/2);
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        system.setup();
    }
};

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

[[nodiscard]] std::vector<FE::Real> planeCoefficients(
    const NativeLinearTetraP2ScalarFieldFixture& fixture,
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
            "LevelSetReinitialization linear-tetra P2 projection");
    if (projection.unassigned_dofs != 0u) {
        throw std::runtime_error(
            "planeCoefficients: native P2 tetra projection left unassigned coefficients");
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
    options.max_iterations = 100;

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
    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    EXPECT_LE(result.max_iteration_residual,
              options.signed_distance_tolerance);
    EXPECT_GT(result.max_signed_distance_error, 0.0);
    EXPECT_LE(result.max_interface_displacement,
              options.max_zero_set_displacement);
    EXPECT_LE(result.l2_interface_displacement,
              options.max_zero_set_displacement);

    // All coefficients in a connected cut-cell patch must receive one common
    // positive scale.  This improves the distance magnitude without moving
    // any zero of the finite-element interpolant.
    const auto scale = repaired[0] / distorted[0];
    EXPECT_GT(scale, 0.0);
    EXPECT_NE(scale, 1.0);
    EXPECT_LT(std::abs(FE::Real{4.0} * scale - FE::Real{1.0}),
              std::abs(FE::Real{4.0} - FE::Real{1.0}));
    for (std::size_t i = 0; i < repaired.size(); ++i) {
        EXPECT_NEAR(repaired[i], scale * distorted[i], 1.0e-11)
            << "coefficient " << i;
    }
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(LevelSetReinitialization,
     ProjectionRepairsPlanarSpatiallyNonuniformNearInterfaceGradient)
{
    auto mesh = buildNativeTwoQuadColumnMesh();
    auto scalar_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4,
                                              /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    // This is a deliberately representable special case: all cells share the
    // same planar zero set x=0.25, even though the positive multiplier varies
    // with y.  It demonstrates that the repair is not restricted to one
    // global coefficient scale; it does not imply that arbitrary curved P1
    // interfaces can be exactly preserved and exactly redistanced.
    std::vector<FE::Real> distorted(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices(); ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        const FE::Real varying_scale = FE::Real{2.0} + FE::Real{2.0} * x[1];
        distorted[static_cast<std::size_t>(dofs.front())] =
            varying_scale * (x[0] - FE::Real{0.25});
    }

    const auto row_gradient_error = [&](std::span<const FE::Real> values) {
        FE::Real error = 0.0;
        for (FE::GlobalIndex row = 0; row < 3; ++row) {
            const auto left = entity_map->getVertexDofs(2 * row);
            const auto right = entity_map->getVertexDofs(2 * row + 1);
            if (left.size() != 1u || right.size() != 1u) {
                throw std::runtime_error(
                    "nonuniform-gradient test expected scalar row DOFs");
            }
            const FE::Real gradient =
                values[static_cast<std::size_t>(right.front())] -
                values[static_cast<std::size_t>(left.front())];
            error = std::max(error, std::abs(gradient - FE::Real{1.0}));
        }
        return error;
    };
    const FE::Real initial_error = row_gradient_error(distorted);
    ASSERT_GT(initial_error, 1.0);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-11;
    options.interface_band_width = 3.0;
    options.max_iterations = 100;
    options.pseudo_time_step_scale = 0.5;
    options.max_zero_set_displacement = 1.0e-12;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        system.meshAccess(),
        field_dofs,
        options,
        distorted,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.converged) << result.diagnostic;
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    EXPECT_LE(result.max_interface_displacement,
              options.max_zero_set_displacement + 1.0e-14);
    EXPECT_LE(result.max_signed_distance_error,
              options.signed_distance_tolerance);
    EXPECT_LT(row_gradient_error(repaired), 1.0e-9);
    EXPECT_LT(row_gradient_error(repaired), initial_error);
}

TEST(LevelSetReinitialization,
     ProjectionManufacturedPlanarRefinementConvergesWithoutMovingWallContacts)
{
    constexpr FE::Real interface_x = FE::Real{0.37};
    std::vector<FE::Real> converged_l2_errors;
    for (const int subdivisions : {8, 16, 32}) {
        SCOPED_TRACE("subdivisions=" + std::to_string(subdivisions));
        auto mesh = buildNativeStructuredQuadMesh(subdivisions);
        auto scalar_space =
            std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4,
                                                  /*order=*/1);
        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        ASSERT_NO_THROW(system.setup());

        const auto& field_dofs = system.fieldDofHandler(phi);
        const auto* entity_map = field_dofs.getEntityDofMap();
        ASSERT_NE(entity_map, nullptr);
        std::vector<FE::Real> distorted(
            static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0;
             vertex < entity_map->numVertices();
             ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            ASSERT_EQ(dofs.size(), 1u);
            const auto x = system.meshAccess().getNodeCoordinates(vertex);
            // This Q1-manufactured field has the exact vertical zero set
            // x=interface_x, but a non-distance multiplier varying along the
            // interface.  It is the representable class for which projection
            // redistancing can converge without relaxing its geometric gate.
            const FE::Real multiplier = FE::Real{2.0} + x[1];
            distorted[static_cast<std::size_t>(dofs.front())] =
                multiplier * (x[0] - interface_x);
        }

        level_set::LevelSetReinitializationOptions options{};
        options.signed_distance_tolerance = 1.0e-11;
        options.interface_band_width = 2.0;
        options.max_iterations = 100;
        options.pseudo_time_step_scale = 0.5;
        options.max_zero_set_displacement = 1.0e-12;

        std::vector<FE::Real> repaired;
        const auto result =
            level_set::repairLevelSetSignedDistanceByProjection(
                system.meshAccess(),
                field_dofs,
                options,
                distorted,
                repaired);

        ASSERT_TRUE(result.success) << result.diagnostic;
        ASSERT_TRUE(result.converged) << result.diagnostic;
        EXPECT_TRUE(result.zero_set_bound_satisfied);
        EXPECT_EQ(result.cut_cells,
                  static_cast<std::size_t>(subdivisions));
        EXPECT_LE(result.max_signed_distance_error,
                  options.signed_distance_tolerance);
        EXPECT_LE(result.max_interface_displacement,
                  options.max_zero_set_displacement + 1.0e-14);

        FE::Real squared_error = 0.0;
        for (FE::GlobalIndex vertex = 0;
             vertex < entity_map->numVertices();
             ++vertex) {
            const auto x = system.meshAccess().getNodeCoordinates(vertex);
            const FE::Real error =
                vertexValue(*entity_map, repaired, vertex) -
                (x[0] - interface_x);
            squared_error += error * error;
        }
        const auto vertex_count =
            static_cast<FE::Real>(entity_map->numVertices());
        const FE::Real l2_error = std::sqrt(squared_error / vertex_count);
        EXPECT_LE(l2_error, options.signed_distance_tolerance);
        converged_l2_errors.push_back(l2_error);
        RecordProperty("redistance_l2_N" + std::to_string(subdivisions),
                       ::testing::PrintToString(l2_error));
        RecordProperty(
            "redistance_zero_set_linf_N" + std::to_string(subdivisions),
            ::testing::PrintToString(result.max_interface_displacement));
        RecordProperty(
            "redistance_signed_distance_linf_N" +
                std::to_string(subdivisions),
            ::testing::PrintToString(result.max_signed_distance_error));

        const auto contact_root = [&](const std::vector<FE::Real>& values,
                                      int row) {
            const FE::Real scaled =
                interface_x * static_cast<FE::Real>(subdivisions);
            const int left_column = static_cast<int>(std::floor(scaled));
            const auto vertex_extent =
                static_cast<FE::GlobalIndex>(subdivisions + 1);
            const FE::GlobalIndex left_vertex =
                static_cast<FE::GlobalIndex>(row) * vertex_extent +
                left_column;
            const FE::GlobalIndex right_vertex = left_vertex + 1;
            const FE::Real left_value =
                vertexValue(*entity_map, values, left_vertex);
            const FE::Real right_value =
                vertexValue(*entity_map, values, right_vertex);
            const FE::Real denominator = right_value - left_value;
            if (!(std::abs(denominator) > 1.0e-14)) {
                throw std::runtime_error(
                    "manufactured contact edge has no resolvable crossing");
            }
            const FE::Real left_x =
                static_cast<FE::Real>(left_column) / subdivisions;
            const FE::Real h = FE::Real{1.0} / subdivisions;
            return left_x - h * left_value / denominator;
        };
        for (const int wall_row : {0, subdivisions}) {
            const FE::Real original_contact =
                contact_root(distorted, wall_row);
            const FE::Real repaired_contact =
                contact_root(repaired, wall_row);
            EXPECT_NEAR(original_contact, interface_x, 1.0e-13);
            EXPECT_NEAR(repaired_contact, interface_x, 1.0e-12);
            EXPECT_LE(std::abs(repaired_contact - original_contact),
                      options.max_zero_set_displacement + 1.0e-14);
        }

        if (subdivisions == 16) {
            std::array<FE::Real, 3> iteration_errors{};
            for (int iterations = 1; iterations <= 3; ++iterations) {
                auto iterative_options = options;
                iterative_options.signed_distance_tolerance = 1.0e-14;
                iterative_options.max_iterations = iterations;
                iterative_options.pseudo_time_step_scale = 0.25;
                std::vector<FE::Real> candidate;
                const auto iterative_result =
                    level_set::repairLevelSetSignedDistanceByProjection(
                        system.meshAccess(),
                        field_dofs,
                        iterative_options,
                        distorted,
                        candidate);
                ASSERT_TRUE(iterative_result.success)
                    << iterative_result.diagnostic;
                EXPECT_FALSE(iterative_result.converged);
                EXPECT_LE(iterative_result.max_interface_displacement,
                          iterative_options.max_zero_set_displacement +
                              1.0e-14);
                iteration_errors[static_cast<std::size_t>(iterations - 1)] =
                    iterative_result.max_signed_distance_error;
            }
            ASSERT_GT(iteration_errors[0], 0.0);
            ASSERT_GT(iteration_errors[1], 0.0);
            ASSERT_GT(iteration_errors[2], 0.0);
            // The manufactured update is not displacement-limited, so the
            // measured error must exhibit the relaxation operator's exact
            // linear convergence factor 1-pseudo_time_step_scale = 0.75.
            EXPECT_NEAR(iteration_errors[1] / iteration_errors[0],
                        0.75,
                        1.0e-12);
            EXPECT_NEAR(iteration_errors[2] / iteration_errors[1],
                        0.75,
                        1.0e-12);
            RecordProperty("redistance_iteration_factor_1_to_2",
                           ::testing::PrintToString(iteration_errors[1] /
                                                    iteration_errors[0]));
            RecordProperty("redistance_iteration_factor_2_to_3",
                           ::testing::PrintToString(iteration_errors[2] /
                                                    iteration_errors[1]));
        }
    }

    ASSERT_EQ(converged_l2_errors.size(), 3u);
    for (const auto error : converged_l2_errors) {
        EXPECT_LE(error, 1.0e-11);
    }
}
#endif

TEST(LevelSetReinitialization, GenericProjectionFailsClosedForHighOrderCellNodeDofs)
{
    const Quad9ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto distorted = distortedPlaneCoefficients(fixture);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;
    options.preserve_band_width = 0.0;

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
    EXPECT_EQ(result.repaired_dofs, 0u);
    EXPECT_EQ(result.interface_fragments, 1u);
    EXPECT_EQ(result.cut_cells, 1u);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(LevelSetReinitialization, FESystemOverloadRepairsP2EdgeDofsOnLinearTetraMesh)
{
    const NativeLinearTetraP2ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto distorted = planeCoefficients(fixture, FE::Real{4.0});
    const auto offset = static_cast<std::size_t>(
        fixture.system.fieldDofOffset(fixture.phi));

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;
    options.max_iterations = 100;

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
    EXPECT_EQ(result.repaired_dofs, 10u);
    EXPECT_EQ(result.interface_fragments, 1u);
    EXPECT_EQ(result.cut_cells, 1u);
    ASSERT_EQ(repaired_solution.size(), solution.size());

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    EXPECT_LE(result.max_interface_displacement,
              options.max_zero_set_displacement);
    const auto first_dof = static_cast<std::size_t>(
        entity_map->getVertexDofs(0).front());
    const auto common_scale = repaired_solution[offset + first_dof] /
                              distorted[first_dof];
    EXPECT_GT(common_scale, 0.0);

    for (FE::GlobalIndex edge = 0;
         edge < static_cast<FE::GlobalIndex>(fixture.mesh->local_mesh().n_edges());
         ++edge) {
        const auto edge_dofs = entity_map->getEdgeDofs(edge);
        ASSERT_EQ(edge_dofs.size(), 1u);
        const auto dof = static_cast<std::size_t>(edge_dofs.front());
        EXPECT_TRUE(std::isfinite(repaired_solution[offset + dof]));
        EXPECT_NEAR(repaired_solution[offset + dof],
                    common_scale * distorted[dof],
                    1.0e-11)
            << "edge=" << edge << " dof=" << dof;
    }
}

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
    options.max_iterations = 100;

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

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    EXPECT_LE(result.max_interface_displacement,
              options.max_zero_set_displacement);
    std::optional<FE::Real> common_scale;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(distorted[i]) <= 1.0e-14) {
            EXPECT_NEAR(repaired_solution[offset + i], 0.0, 1.0e-12);
            continue;
        }
        const auto scale = repaired_solution[offset + i] / distorted[i];
        if (!common_scale.has_value()) {
            common_scale = scale;
        }
        EXPECT_NEAR(scale, *common_scale, 1.0e-11)
            << "coefficient " << i;
    }
    ASSERT_TRUE(common_scale.has_value());
    EXPECT_GT(*common_scale, 0.0);
    EXPECT_LT(std::abs(*common_scale - FE::Real{0.25}),
              std::abs(FE::Real{1.0} - FE::Real{0.25}));
}
#endif

TEST(LevelSetReinitialization, ProjectionPreservesDistortedZeroSet)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto signed_distance = signedDistancePlaneCoefficients(fixture);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;
    options.max_iterations = 100;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        options,
        signed_distance,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_EQ(result.interface_displacement_samples, 3u);
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    EXPECT_LE(result.max_interface_displacement,
              options.max_zero_set_displacement);
    ASSERT_EQ(repaired.size(), signed_distance.size());
    const auto common_scale = repaired[0] / signed_distance[0];
    EXPECT_GT(common_scale, 0.0);
    for (std::size_t i = 0; i < repaired.size(); ++i) {
        EXPECT_NEAR(repaired[i], common_scale * signed_distance[i], 1.0e-11)
            << "coefficient " << i;
    }
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
    options.preserve_band_width = 0.0;

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
    EXPECT_EQ(result.interface_displacement_samples, 2u);
    EXPECT_LE(result.max_interface_displacement, tolerance_gate);
    EXPECT_LE(result.l2_interface_displacement, tolerance_gate);
    EXPECT_NEAR(result.max_interface_displacement, 0.0, 1.0e-12);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        EXPECT_NEAR(vertexValue(*entity_map, repaired, vertex),
                    vertexValue(*entity_map, signed_distance, vertex),
                    1.0e-12);
    }
}

TEST(LevelSetReinitialization,
     ProjectionPreservesObliqueWallContactAngleWhileRedistancing)
{
    const QuadScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);

    constexpr FE::Real nx = 0.5;
    constexpr FE::Real ny = 0.86602540378443864676;
    std::vector<FE::Real> distorted(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = fixture.mesh->getNodeCoordinates(vertex);
        distorted[static_cast<std::size_t>(dofs.front())] =
            FE::Real{4.0} *
            (nx * x[0] + ny * x[1] - FE::Real{0.6});
    }

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 2.0;
    options.max_iterations = 100;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        options,
        distorted,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.converged) << result.diagnostic;
    EXPECT_LE(result.max_interface_displacement,
              options.max_zero_set_displacement);
    const FE::Real gx =
        vertexValue(*entity_map, repaired, 1) -
        vertexValue(*entity_map, repaired, 0);
    const FE::Real gy =
        vertexValue(*entity_map, repaired, 3) -
        vertexValue(*entity_map, repaired, 0);
    const FE::Real gradient_norm = std::sqrt(gx * gx + gy * gy);
    ASSERT_GT(gradient_norm, 0.0);
    EXPECT_NEAR(gradient_norm, 1.0, 1.0e-10);
    EXPECT_NEAR(gx / gradient_norm, nx, 1.0e-10);
    EXPECT_NEAR(gy / gradient_norm, ny, 1.0e-10);
}

TEST(LevelSetReinitialization,
     ProjectionPreservesThreeDimensionalWallContactAngleWhileRedistancing)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);

    constexpr FE::Real nx = 0.3;
    constexpr FE::Real ny = 0.4;
    constexpr FE::Real nz = 0.86602540378443864676;
    std::vector<FE::Real> distorted(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = fixture.mesh->getNodeCoordinates(vertex);
        distorted[static_cast<std::size_t>(dofs.front())] =
            FE::Real{4.0} *
            (nx * x[0] + ny * x[1] + nz * x[2] - FE::Real{0.25});
    }

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 2.0;
    options.max_iterations = 100;
    options.max_zero_set_displacement = 1.0e-12;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        options,
        distorted,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.converged) << result.diagnostic;
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    EXPECT_LE(result.max_interface_displacement,
              options.max_zero_set_displacement + 1.0e-14);

    // Vertex 0 is the origin and vertices 1--3 lie on the coordinate axes,
    // so these differences recover the physical gradient directly.  The
    // interface intersects the z=0 boundary face in a contact line; preserving
    // its unit normal preserves that wall-contact angle.
    const FE::Real origin = vertexValue(*entity_map, repaired, 0);
    const FE::Real gx = vertexValue(*entity_map, repaired, 1) - origin;
    const FE::Real gy = vertexValue(*entity_map, repaired, 2) - origin;
    const FE::Real gz = vertexValue(*entity_map, repaired, 3) - origin;
    const FE::Real gradient_norm =
        std::sqrt(gx * gx + gy * gy + gz * gz);
    ASSERT_GT(gradient_norm, 0.0);
    EXPECT_NEAR(gradient_norm, 1.0, 1.0e-10);
    EXPECT_NEAR(gx / gradient_norm, nx, 1.0e-10);
    EXPECT_NEAR(gy / gradient_norm, ny, 1.0e-10);
    EXPECT_NEAR(gz / gradient_norm, nz, 1.0e-10);
    EXPECT_NEAR(std::acos(std::abs(gz) / gradient_norm),
                std::acos(nz),
                1.0e-10);
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
    options.preserve_band_width = 0.0;

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

TEST(LevelSetReinitialization, ProjectionRepairsInterfaceNeighborhoodByDefault)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto distorted = distortedPlaneCoefficients(fixture);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;
    options.max_iterations = 100;

    std::vector<FE::Real> repaired;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh,
        field_dofs,
        options,
        distorted,
        repaired);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_DOUBLE_EQ(result.preserve_band_width, 0.0);
    EXPECT_EQ(result.preserved_dofs, 0u);
    EXPECT_EQ(result.repaired_dofs, 4u);
    EXPECT_GT(result.max_abs_update, 0.0);
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    ASSERT_EQ(repaired.size(), distorted.size());
    EXPECT_NE(repaired, distorted);
}

TEST(LevelSetReinitialization, ProjectionRejectsLegacyPreserveBand)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto distorted = distortedPlaneCoefficients(fixture);

    level_set::LevelSetReinitializationOptions options{};
    options.signed_distance_tolerance = 1.0e-12;
    options.interface_band_width = 1.0;
    options.preserve_band_width = 1.0e-9;

    std::vector<FE::Real> repaired;
    EXPECT_THROW(
        (void)level_set::repairLevelSetSignedDistanceByProjection(
            *fixture.mesh,
            field_dofs,
            options,
            distorted,
            repaired),
        std::invalid_argument);
}

TEST(LevelSetReinitialization, IterationControlsChangeRepairConvergence)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto distorted = distortedPlaneCoefficients(fixture);

    level_set::LevelSetReinitializationOptions one_step{};
    one_step.signed_distance_tolerance = 1.0e-14;
    one_step.interface_band_width = 1.0;
    one_step.max_iterations = 1;
    one_step.pseudo_time_step_scale = 0.25;

    auto four_steps = one_step;
    four_steps.max_iterations = 4;

    std::vector<FE::Real> repaired_one;
    std::vector<FE::Real> repaired_four;
    const auto result_one = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh, field_dofs, one_step, distorted, repaired_one);
    const auto result_four = level_set::repairLevelSetSignedDistanceByProjection(
        *fixture.mesh, field_dofs, four_steps, distorted, repaired_four);

    ASSERT_TRUE(result_one.success) << result_one.diagnostic;
    ASSERT_TRUE(result_four.success) << result_four.diagnostic;
    EXPECT_EQ(result_one.iterations, 1);
    EXPECT_EQ(result_four.iterations, 4);
    EXPECT_LT(result_four.max_iteration_residual,
              result_one.max_iteration_residual);
    EXPECT_NE(repaired_four, repaired_one);
    EXPECT_LE(result_one.max_interface_displacement,
              one_step.max_zero_set_displacement);
    EXPECT_LE(result_four.max_interface_displacement,
              four_steps.max_zero_set_displacement);
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
