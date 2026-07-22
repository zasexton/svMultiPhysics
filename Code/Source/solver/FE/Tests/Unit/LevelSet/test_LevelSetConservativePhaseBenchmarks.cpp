#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetVolume.h"

#include "Assembly/Assembler.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
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

    explicit StructuredQuadPhaseMeshAccess(
        std::size_t cells_per_axis,
        FE::Real distortion_amplitude = FE::Real{0.0})
        : cells_per_axis_(cells_per_axis)
        , distortion_amplitude_(distortion_amplitude)
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
        const FE::Real xi = spacing * static_cast<FE::Real>(i);
        const FE::Real eta = spacing * static_cast<FE::Real>(j);
        const FE::Real pi = std::numbers::pi_v<FE::Real>;
        return {
            xi + distortion_amplitude_ *
                     std::sin(FE::Real{2.0} * pi * xi) *
                     std::sin(pi * eta),
            eta + distortion_amplitude_ *
                      std::sin(pi * xi) *
                      std::sin(FE::Real{2.0} * pi * eta),
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
    FE::Real distortion_amplitude_{0.0};
};

class StructuredTriLevelSetMeshAccess final
    : public FE::assembly::IMeshAccess {
public:
    using FE::assembly::IMeshAccess::getCellCoordinates;

    explicit StructuredTriLevelSetMeshAccess(
        std::size_t cells_per_axis,
        FE::Real distortion_amplitude)
        : cells_per_axis_(cells_per_axis)
        , distortion_amplitude_(distortion_amplitude)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(
            2u * cells_per_axis_ * cells_per_axis_);
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
        return FE::ElementType::Triangle3;
    }
    void getCellNodes(
        FE::GlobalIndex cell,
        std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto cell_index = static_cast<std::size_t>(cell);
        const auto square = cell_index / 2u;
        const auto i = square % cells_per_axis_;
        const auto j = square / cells_per_axis_;
        if (cell_index % 2u == 0u) {
            nodes = {
                node(i, j),
                node(i + 1u, j),
                node(i + 1u, j + 1u),
            };
        } else {
            nodes = {
                node(i, j),
                node(i + 1u, j + 1u),
                node(i, j + 1u),
            };
        }
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
        const FE::Real xi = spacing * static_cast<FE::Real>(i);
        const FE::Real eta = spacing * static_cast<FE::Real>(j);
        const FE::Real pi = std::numbers::pi_v<FE::Real>;
        return {
            xi + distortion_amplitude_ *
                     std::sin(FE::Real{2.0} * pi * xi) *
                     std::sin(pi * eta),
            eta + distortion_amplitude_ *
                      std::sin(pi * xi) *
                      std::sin(FE::Real{2.0} * pi * eta),
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
    FE::Real distortion_amplitude_{0.0};
};

class StructuredHexPhaseMeshAccess final
    : public FE::assembly::IMeshAccess {
public:
    using FE::assembly::IMeshAccess::getCellCoordinates;

    explicit StructuredHexPhaseMeshAccess(std::size_t cells_per_axis)
        : cells_per_axis_(cells_per_axis)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(
            cells_per_axis_ * cells_per_axis_ * cells_per_axis_);
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        const auto nodes_per_axis = cells_per_axis_ + 1u;
        return static_cast<FE::GlobalIndex>(
            nodes_per_axis * nodes_per_axis * nodes_per_axis);
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return 0;
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override
    {
        return 0;
    }
    [[nodiscard]] int dimension() const override { return 3; }
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
        return FE::ElementType::Hex8;
    }
    void getCellNodes(
        FE::GlobalIndex cell,
        std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto cell_index = static_cast<std::size_t>(cell);
        const auto i = cell_index % cells_per_axis_;
        const auto j =
            (cell_index / cells_per_axis_) % cells_per_axis_;
        const auto k = cell_index /
                       (cells_per_axis_ * cells_per_axis_);
        nodes = {
            node(i, j, k),
            node(i + 1u, j, k),
            node(i + 1u, j + 1u, k),
            node(i, j + 1u, k),
            node(i, j, k + 1u),
            node(i + 1u, j, k + 1u),
            node(i + 1u, j + 1u, k + 1u),
            node(i, j + 1u, k + 1u),
        };
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        const auto index = static_cast<std::size_t>(node_id);
        const auto nodes_per_axis = cells_per_axis_ + 1u;
        const auto i = index % nodes_per_axis;
        const auto j =
            (index / nodes_per_axis) % nodes_per_axis;
        const auto k = index /
                       (nodes_per_axis * nodes_per_axis);
        const FE::Real spacing = FE::Real{1.0} /
                                 static_cast<FE::Real>(cells_per_axis_);
        return {
            spacing * static_cast<FE::Real>(i),
            spacing * static_cast<FE::Real>(j),
            spacing * static_cast<FE::Real>(k),
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
        std::size_t j,
        std::size_t k) const
    {
        const auto nodes_per_axis = cells_per_axis_ + 1u;
        return static_cast<FE::GlobalIndex>(
            (k * nodes_per_axis + j) * nodes_per_axis + i);
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

[[nodiscard]] FE::systems::SetupInputs structuredTriSetupInputs(
    std::size_t cells_per_axis)
{
    const auto nodes_per_axis = cells_per_axis + 1u;
    const auto square_count = cells_per_axis * cells_per_axis;
    const auto cell_count = 2u * square_count;
    const auto node_count = nodes_per_axis * nodes_per_axis;
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = static_cast<FE::GlobalIndex>(cell_count);
    topology.n_vertices = static_cast<FE::GlobalIndex>(node_count);
    topology.dim = 2;
    topology.cell2vertex_offsets.resize(cell_count + 1u, 0);
    topology.cell2vertex_data.reserve(3u * cell_count);
    topology.cell_gids.resize(cell_count);
    topology.cell_owner_ranks.assign(cell_count, 0);
    for (std::size_t j = 0u; j < cells_per_axis; ++j) {
        for (std::size_t i = 0u; i < cells_per_axis; ++i) {
            const auto square = j * cells_per_axis + i;
            const auto first_cell = 2u * square;
            const auto lower_left = j * nodes_per_axis + i;
            const auto lower_right = lower_left + 1u;
            const auto upper_left = lower_left + nodes_per_axis;
            const auto upper_right = upper_left + 1u;
            topology.cell2vertex_offsets[first_cell] =
                static_cast<FE::MeshOffset>(3u * first_cell);
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(lower_left));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(lower_right));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(upper_right));
            topology.cell_gids[first_cell] =
                static_cast<FE::dofs::gid_t>(first_cell);

            topology.cell2vertex_offsets[first_cell + 1u] =
                static_cast<FE::MeshOffset>(3u * (first_cell + 1u));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(lower_left));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(upper_right));
            topology.cell2vertex_data.push_back(
                static_cast<FE::MeshIndex>(upper_left));
            topology.cell_gids[first_cell + 1u] =
                static_cast<FE::dofs::gid_t>(first_cell + 1u);
        }
    }
    topology.cell2vertex_offsets[cell_count] =
        static_cast<FE::MeshOffset>(3u * cell_count);
    topology.vertex_gids.resize(node_count);
    for (std::size_t vertex = 0u; vertex < node_count; ++vertex) {
        topology.vertex_gids[vertex] =
            static_cast<FE::dofs::gid_t>(vertex);
    }
    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

[[nodiscard]] FE::systems::SetupInputs structuredHexSetupInputs(
    std::size_t cells_per_axis)
{
    const auto nodes_per_axis = cells_per_axis + 1u;
    const auto cell_count =
        cells_per_axis * cells_per_axis * cells_per_axis;
    const auto node_count =
        nodes_per_axis * nodes_per_axis * nodes_per_axis;
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = static_cast<FE::GlobalIndex>(cell_count);
    topology.n_vertices = static_cast<FE::GlobalIndex>(node_count);
    topology.dim = 3;
    topology.cell2vertex_offsets.resize(cell_count + 1u, 0);
    topology.cell2vertex_data.reserve(8u * cell_count);
    topology.cell_gids.resize(cell_count);
    topology.cell_owner_ranks.assign(cell_count, 0);
    const auto node = [nodes_per_axis](
                          std::size_t i,
                          std::size_t j,
                          std::size_t k) {
        return (k * nodes_per_axis + j) * nodes_per_axis + i;
    };
    for (std::size_t k = 0u; k < cells_per_axis; ++k) {
        for (std::size_t j = 0u; j < cells_per_axis; ++j) {
            for (std::size_t i = 0u; i < cells_per_axis; ++i) {
                const auto cell =
                    (k * cells_per_axis + j) * cells_per_axis + i;
                topology.cell2vertex_offsets[cell] =
                    static_cast<FE::MeshOffset>(8u * cell);
                for (const auto vertex : {
                         node(i, j, k),
                         node(i + 1u, j, k),
                         node(i + 1u, j + 1u, k),
                         node(i, j + 1u, k),
                         node(i, j, k + 1u),
                         node(i + 1u, j, k + 1u),
                         node(i + 1u, j + 1u, k + 1u),
                         node(i, j + 1u, k + 1u)}) {
                    topology.cell2vertex_data.push_back(
                        static_cast<FE::MeshIndex>(vertex));
                }
                topology.cell_gids[cell] =
                    static_cast<FE::dofs::gid_t>(cell);
            }
        }
    }
    topology.cell2vertex_offsets[cell_count] =
        static_cast<FE::MeshOffset>(8u * cell_count);
    topology.vertex_gids.resize(node_count);
    for (std::size_t vertex = 0u; vertex < node_count; ++vertex) {
        topology.vertex_gids[vertex] =
            static_cast<FE::dofs::gid_t>(vertex);
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

    explicit StructuredPhaseFixture(
        std::size_t cells_per_axis,
        FE::Real distortion_amplitude = FE::Real{0.0})
        : mesh(std::make_shared<StructuredQuadPhaseMeshAccess>(
              cells_per_axis, distortion_amplitude)),
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

struct StructuredTriLevelSetFixture {
    std::shared_ptr<StructuredTriLevelSetMeshAccess> mesh;
    FE::systems::FESystem system;
    FE::FieldId level_set_field{FE::INVALID_FIELD_ID};
    std::vector<FE::GlobalIndex> vertex_dofs{};

    explicit StructuredTriLevelSetFixture(
        std::size_t cells_per_axis,
        FE::Real distortion_amplitude)
        : mesh(std::make_shared<StructuredTriLevelSetMeshAccess>(
              cells_per_axis, distortion_amplitude))
        , system(mesh)
    {
        level_set_field = system.addField(FE::systems::FieldSpec{
            .name = "level_set",
            .space = std::make_shared<FE::spaces::H1Space>(
                FE::ElementType::Triangle3, /*order=*/1),
            .components = 1,
        });
        system.setup({}, structuredTriSetupInputs(cells_per_axis));
        const auto* entity_map =
            system.fieldDofHandler(level_set_field).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error(
                "structured triangular level-set fixture has no entity map");
        }
        vertex_dofs.resize(
            static_cast<std::size_t>(mesh->numVertices()));
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh->numVertices(); ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            if (dofs.size() != 1u || dofs.front() < 0) {
                throw std::runtime_error(
                    "structured triangular level-set fixture has an invalid vertex map");
            }
            vertex_dofs[static_cast<std::size_t>(vertex)] = dofs.front();
        }
    }
};

struct StructuredHexPhaseFixture {
    std::shared_ptr<StructuredHexPhaseMeshAccess> mesh;
    FE::systems::FESystem system;
    FE::FieldId phase{FE::INVALID_FIELD_ID};
    level_set::LevelSetP1PhaseTransportGraph graph{};
    std::vector<std::array<FE::Real, 3>> node_coordinates{};

    explicit StructuredHexPhaseFixture(std::size_t cells_per_axis)
        : mesh(std::make_shared<StructuredHexPhaseMeshAccess>(
              cells_per_axis)),
          system(mesh)
    {
        phase = system.addField(FE::systems::FieldSpec{
            .name = "liquid_indicator",
            .space = std::make_shared<FE::spaces::H1Space>(
                FE::ElementType::Hex8, /*order=*/1),
            .components = 1,
        });
        system.setup({}, structuredHexSetupInputs(cells_per_axis));
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
            graph.diagnostic =
                "structured hex phase fixture has no entity map";
            return;
        }
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh->numVertices(); ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            if (dofs.size() != 1u || dofs.front() < 0 ||
                static_cast<std::size_t>(dofs.front()) >= graph.nodes) {
                graph.success = false;
                graph.diagnostic =
                    "structured hex phase fixture has an invalid vertex map";
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
    struct HistoryPoint {
        int step{0};
        FE::Real time{0.0};
        FE::Real previous_measure{0.0};
        FE::Real low_order_measure{0.0};
        FE::Real raw_target_measure{0.0};
        FE::Real limited_measure{0.0};
        FE::Real cumulative_boundary_transfer{0.0};
        FE::Real cumulative_divergence_source{0.0};
        FE::Real accounted_measure{0.0};
        FE::Real raw_measure_error{0.0};
        FE::Real accounted_balance_error{0.0};
        FE::Real minimum_indicator{0.0};
        FE::Real maximum_indicator{0.0};
        FE::Real maximum_courant{0.0};
        FE::Real maximum_local_balance_residual{0.0};
        FE::Real maximum_component_balance_residual{0.0};
        std::size_t components{0u};
    };
    std::vector<HistoryPoint> history{};
    std::optional<level_set::LevelSetP1PhaseTransportStageResult>
        final_stage{};
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

template <typename Fixture>
[[nodiscard]] TransportRun runTransport(
    const Fixture& fixture,
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
        auto stage =
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
        run.history.push_back(TransportRun::HistoryPoint{
            .step = step + 1,
            .time = static_cast<FE::Real>(step + 1) * dt,
            .previous_measure =
                stage.correction.total_previous_liquid_measure,
            .low_order_measure =
                stage.correction.total_low_order_liquid_measure,
            .raw_target_measure =
                stage.correction.total_raw_target_liquid_measure,
            .limited_measure =
                stage.correction.total_limited_liquid_measure,
            .cumulative_boundary_transfer =
                static_cast<FE::Real>(cumulative_boundary_transfer),
            .cumulative_divergence_source =
                static_cast<FE::Real>(cumulative_divergence_source),
            .accounted_measure = accounted_measure,
            .raw_measure_error = measure - run.initial_measure,
            .accounted_balance_error = measure - accounted_measure,
            .minimum_indicator =
                stage.correction.minimum_limited_liquid_indicator,
            .maximum_indicator =
                stage.correction.maximum_limited_liquid_indicator,
            .maximum_courant = stage.maximum_courant,
            .maximum_local_balance_residual =
                stage.correction.maximum_local_mass_balance_residual,
            .maximum_component_balance_residual =
                stage.correction.maximum_component_balance_residual,
            .components = stage.correction.components.size(),
        });
        if (step + 1 == steps) {
            run.final_stage = std::move(stage);
        }
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

template <typename Fixture>
[[nodiscard]] FE::Real weightedL1Error(
    const Fixture& fixture,
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

template <typename Fixture>
[[nodiscard]] FE::Real weightedL1Difference(
    const Fixture& fixture,
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

template <typename Fixture>
[[nodiscard]] std::array<FE::Real, 3> phaseCentroid(
    const Fixture& fixture,
    const std::vector<FE::Real>& phase)
{
    long double measure = 0.0L;
    std::array<long double, 3> first_moment{0.0L, 0.0L, 0.0L};
    for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
        const long double nodal_measure = static_cast<long double>(
            fixture.graph.lumped_control_volume[node] * phase[node]);
        measure += nodal_measure;
        first_moment[0] += nodal_measure *
                           fixture.node_coordinates[node][0];
        first_moment[1] += nodal_measure *
                           fixture.node_coordinates[node][1];
        first_moment[2] += nodal_measure *
                           fixture.node_coordinates[node][2];
    }
    return {
        static_cast<FE::Real>(first_moment[0] / measure),
        static_cast<FE::Real>(first_moment[1] / measure),
        static_cast<FE::Real>(first_moment[2] / measure),
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

[[nodiscard]] PhaseInitializer sphere(
    FE::Real center_x,
    FE::Real center_y,
    FE::Real center_z,
    FE::Real radius)
{
    return [=](const std::array<FE::Real, 3>& point) {
        const FE::Real dx = point[0] - center_x;
        const FE::Real dy = point[1] - center_y;
        const FE::Real dz = point[2] - center_z;
        return dx * dx + dy * dy + dz * dz <= radius * radius
            ? FE::Real{1.0}
            : FE::Real{0.0};
    };
}

[[nodiscard]] VelocityField reversibleThreeDimensionalVelocity(
    FE::Real final_time)
{
    return [final_time](
               FE::Real time,
               const std::array<FE::Real, 3>& point) {
        const FE::Real pi = std::numbers::pi_v<FE::Real>;
        const FE::Real amplitude = std::cos(pi * time / final_time);
        const FE::Real sin_x = std::sin(pi * point[0]);
        const FE::Real sin_y = std::sin(pi * point[1]);
        const FE::Real sin_z = std::sin(pi * point[2]);
        const FE::Real sin_2x = std::sin(FE::Real{2.0} * pi * point[0]);
        const FE::Real sin_2y = std::sin(FE::Real{2.0} * pi * point[1]);
        const FE::Real sin_2z = std::sin(FE::Real{2.0} * pi * point[2]);
        return std::array<FE::Real, 3>{
            FE::Real{2.0} * amplitude * sin_x * sin_x *
                sin_2y * sin_2z,
            -amplitude * sin_2x * sin_y * sin_y * sin_2z,
            -amplitude * sin_2x * sin_2y * sin_z * sin_z,
        };
    };
}

[[nodiscard]] FE::Real vectorDot(
    const std::array<FE::Real, 3>& first,
    const std::array<FE::Real, 3>& second,
    int dimension)
{
    FE::Real value{0.0};
    for (int d = 0; d < dimension; ++d) {
        value += first[static_cast<std::size_t>(d)] *
                 second[static_cast<std::size_t>(d)];
    }
    return value;
}

template <typename Fixture>
[[nodiscard]] FE::Real maximumCourantRate(
    const Fixture& fixture,
    const VelocityField& velocity_field,
    FE::Real time)
{
    std::vector<std::array<FE::Real, 3>> velocity(fixture.graph.nodes);
    for (std::size_t node = 0u; node < fixture.graph.nodes; ++node) {
        velocity[node] = velocity_field(
            time, fixture.node_coordinates[node]);
    }
    std::vector<FE::Real> rate(fixture.graph.nodes, FE::Real{0.0});
    for (const auto& edge : fixture.graph.edges) {
        const auto first = static_cast<std::size_t>(edge.first_node);
        const auto second = static_cast<std::size_t>(edge.second_node);
        const FE::Real first_speed = vectorDot(
            edge.first_test_second_gradient,
            velocity[second], fixture.graph.dimension);
        const FE::Real second_speed = vectorDot(
            edge.second_test_first_gradient,
            velocity[first], fixture.graph.dimension);
        const FE::Real diffusion = std::max(
            std::abs(first_speed), std::abs(second_speed));
        rate[first] += std::max(FE::Real{0.0},
                                diffusion - first_speed) /
                       fixture.graph.lumped_control_volume[first];
        rate[second] += std::max(FE::Real{0.0},
                                 diffusion - second_speed) /
                        fixture.graph.lumped_control_volume[second];
    }
    return *std::max_element(rate.begin(), rate.end());
}

[[nodiscard]] int stepsForCourant(
    FE::Real final_time,
    FE::Real maximum_courant_rate,
    FE::Real requested_courant)
{
    if (!(final_time > FE::Real{0.0}) ||
        !(maximum_courant_rate > FE::Real{0.0}) ||
        !(requested_courant > FE::Real{0.0})) {
        throw std::invalid_argument(
            "release transport case requires positive time and Courant data");
    }
    const long double required = std::ceil(
        static_cast<long double>(final_time) *
        static_cast<long double>(maximum_courant_rate) /
        static_cast<long double>(requested_courant));
    if (required > static_cast<long double>(
                       std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "release transport case requires too many time steps");
    }
    return std::max(1, static_cast<int>(required));
}

[[nodiscard]] std::optional<std::string> environmentValue(
    const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::string(value);
}

[[nodiscard]] std::size_t parsePositiveSize(
    const std::string& value,
    const char* name)
{
    std::size_t consumed = 0u;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0u ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            std::string(name) + " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

[[nodiscard]] FE::Real parsePositiveReal(
    const std::string& value,
    const char* name)
{
    std::size_t consumed = 0u;
    const FE::Real parsed = static_cast<FE::Real>(
        std::stod(value, &consumed));
    if (consumed != value.size() || !std::isfinite(parsed) ||
        !(parsed > FE::Real{0.0})) {
        throw std::invalid_argument(
            std::string(name) + " must be a positive finite number");
    }
    return parsed;
}

[[nodiscard]] bool isReleaseCourant(FE::Real value)
{
    return value == FE::Real{0.5} || value == FE::Real{0.25} ||
           value == FE::Real{0.125};
}

template <typename Writer>
void publishNewTextFile(
    const std::filesystem::path& path,
    Writer&& writer)
{
    if (path.empty()) {
        throw std::invalid_argument(
            "release transport artifact path must not be empty");
    }
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code directory_error;
        std::filesystem::create_directories(parent, directory_error);
        if (directory_error) {
            throw std::runtime_error(
                "could not create release transport artifact directory: " +
                directory_error.message());
        }
    }
    const std::filesystem::path temporary(path.string() + ".tmp");
    if (std::filesystem::exists(path) ||
        std::filesystem::exists(temporary)) {
        throw std::runtime_error(
            "release transport artifact refuses to replace an existing path");
    }
    {
        std::ofstream output(temporary, std::ios::out | std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error(
                "could not open release transport temporary artifact");
        }
        output << std::setprecision(
            std::numeric_limits<FE::Real>::max_digits10);
        writer(output);
        output.flush();
        if (!output.good()) {
            throw std::runtime_error(
                "could not write release transport temporary artifact");
        }
    }
    std::error_code link_error;
    std::filesystem::create_hard_link(temporary, path, link_error);
    if (link_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw std::runtime_error(
            "could not publish release transport artifact: " +
            link_error.message());
    }
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    if (cleanup_error) {
        throw std::runtime_error(
            "could not remove release transport temporary artifact: " +
            cleanup_error.message());
    }
}

void writeTransportHistory(
    const std::filesystem::path& path,
    const TransportRun& run)
{
    if (!run.success || run.history.empty()) {
        throw std::invalid_argument(
            "release transport history requires a successful nonempty run");
    }
    publishNewTextFile(path, [&run](std::ostream& output) {
        output
            << "step,time,previous_measure,low_order_measure,"
               "raw_target_measure,limited_measure,"
               "cumulative_boundary_transfer,"
               "cumulative_discrete_divergence_source,"
               "accounted_measure,raw_measure_error,"
               "accounted_balance_error,minimum_indicator,"
               "maximum_indicator,maximum_courant,"
               "maximum_local_balance_residual,"
               "maximum_component_balance_residual,components\n";
        for (const auto& point : run.history) {
            output << point.step << ',' << point.time << ','
                   << point.previous_measure << ','
                   << point.low_order_measure << ','
                   << point.raw_target_measure << ','
                   << point.limited_measure << ','
                   << point.cumulative_boundary_transfer << ','
                   << point.cumulative_divergence_source << ','
                   << point.accounted_measure << ','
                   << point.raw_measure_error << ','
                   << point.accounted_balance_error << ','
                   << point.minimum_indicator << ','
                   << point.maximum_indicator << ','
                   << point.maximum_courant << ','
                   << point.maximum_local_balance_residual << ','
                   << point.maximum_component_balance_residual << ','
                   << point.components << '\n';
        }
    });
}

template <typename Fixture>
void writeFinalTransportDetails(
    const std::filesystem::path& directory,
    const Fixture& fixture,
    const TransportRun& run)
{
    if (!run.success || !run.final_stage.has_value()) {
        throw std::invalid_argument(
            "release transport details require a successful final stage");
    }
    std::error_code parent_error;
    if (!directory.parent_path().empty()) {
        std::filesystem::create_directories(
            directory.parent_path(), parent_error);
    }
    if (parent_error || !std::filesystem::create_directory(directory)) {
        throw std::runtime_error(
            "release transport detail directory must be new");
    }
    const auto& stage = *run.final_stage;
    const auto& correction = stage.correction;
    publishNewTextFile(
        directory / "control_volumes.csv",
        [&fixture, &stage, &correction](std::ostream& output) {
            output
                << "node,x,y,z,lumped_control_volume,previous_indicator,"
                   "low_order_indicator,raw_target_indicator,"
                   "limited_indicator,physical_boundary_transfer,"
                   "discrete_divergence_source,"
                   "low_order_interior_transfer,"
                   "raw_antidiffusive_transfer,"
                   "limited_antidiffusive_transfer,"
                   "limited_balance_residual,courant,component_id\n";
            for (std::size_t index = 0u;
                 index < correction.nodes.size(); ++index) {
                const auto& node = correction.nodes[index];
                const auto& coordinate = fixture.node_coordinates[index];
                output << node.node << ','
                       << coordinate[0] << ',' << coordinate[1] << ','
                       << coordinate[2] << ','
                       << node.lumped_control_volume << ','
                       << node.previous_liquid_indicator << ','
                       << node.low_order_liquid_indicator << ','
                       << node.raw_target_liquid_indicator << ','
                       << node.limited_liquid_indicator << ','
                       << node.physical_boundary_mass_transfer << ','
                       << node.discrete_divergence_mass_source << ','
                       << node.low_order_interior_mass_transfer << ','
                       << node.raw_antidiffusive_mass_transfer << ','
                       << node.limited_antidiffusive_mass_transfer << ','
                       << node.local_mass_balance_residual << ','
                       << stage.nodal_courant[index] << ','
                       << correction.node_component_ids[index] << '\n';
            }
        });
    publishNewTextFile(
        directory / "edges.csv",
        [&correction](std::ostream& output) {
            output
                << "first_node,second_node,low_order_transfer,"
                   "raw_antidiffusive_transfer,correction_factor,"
                   "limited_antidiffusive_transfer,"
                   "low_order_pair_residual,raw_pair_residual,"
                   "limited_pair_residual\n";
            for (const auto& edge : correction.edges) {
                output << edge.first_node << ',' << edge.second_node << ','
                       << edge.low_order_mass_transfer << ','
                       << edge.raw_antidiffusive_mass_transfer << ','
                       << edge.correction_factor << ','
                       << edge.limited_antidiffusive_mass_transfer << ','
                       << edge.low_order_pair_cancellation_residual << ','
                       << edge.raw_pair_cancellation_residual << ','
                       << edge.limited_pair_cancellation_residual << '\n';
            }
        });
    publishNewTextFile(
        directory / "components.csv",
        [&correction](std::ostream& output) {
            output
                << "classification,component_id,nodes,previous_measure,"
                   "low_order_measure,raw_target_measure,limited_measure,"
                   "physical_boundary_transfer,"
                   "discrete_divergence_source,low_order_transfer,"
                   "raw_antidiffusive_transfer,"
                   "limited_antidiffusive_transfer,"
                   "low_order_balance_residual,raw_target_balance_residual,"
                   "limited_balance_residual\n";
            const auto write_component = [&output](
                const char* classification,
                const level_set::LevelSetPhaseFluxComponentLedger& component) {
                output << classification << ',' << component.component_id
                       << ',' << component.nodes << ','
                       << component.previous_liquid_measure << ','
                       << component.low_order_liquid_measure << ','
                       << component.raw_target_liquid_measure << ','
                       << component.limited_liquid_measure << ','
                       << component.physical_boundary_mass_transfer << ','
                       << component.discrete_divergence_mass_source << ','
                       << component.low_order_interior_mass_transfer << ','
                       << component.raw_antidiffusive_mass_transfer << ','
                       << component.limited_antidiffusive_mass_transfer << ','
                       << component.low_order_balance_residual << ','
                       << component.raw_target_balance_residual << ','
                       << component.limited_balance_residual << '\n';
            };
            for (const auto& component : correction.components) {
                write_component("resolved", component);
            }
            if (correction.subthreshold_component_present) {
                write_component(
                    "subthreshold", correction.subthreshold_component);
            }
        });
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
     DistortedMeshTranslatingDiskConservesAndRefines)
{
    constexpr FE::Real speed = 0.2;
    constexpr FE::Real final_time = 0.5;
    constexpr FE::Real distortion_amplitude = 0.04;
    std::vector<FE::Real> errors;
    std::vector<FE::Real> centroid_errors;
    for (const std::size_t cells_per_axis : {16u, 32u, 64u}) {
        StructuredPhaseFixture fixture(
            cells_per_axis, distortion_amplitude);
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
        RecordProperty("distorted_l1" + suffix,
                       serializeReal(errors.back()));
        RecordProperty("distorted_centroid_error" + suffix,
                       serializeReal(centroid_errors.back()));
        RecordProperty("distorted_measure_error" + suffix,
                       serializeReal(run.maximum_measure_error));
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
     DistortedTranslatingDiskGlobalVolumeCorrectionClosesRuntimeDrift)
{
    constexpr FE::Real speed = 0.2;
    constexpr FE::Real representative_time = 0.37;
    constexpr FE::Real initial_center_x = 0.317;
    constexpr FE::Real center_y = 0.473;
    constexpr FE::Real radius = 0.137;
    constexpr FE::Real distortion_amplitude = 0.03;
    const FE::Real translated_center_x =
        initial_center_x + speed * representative_time;
    const FE::Real analytic_measure =
        std::numbers::pi_v<FE::Real> * radius * radius;
    std::vector<FE::Real> target_measure_errors;
    FE::Real maximum_snapshot_measure_difference{0.0};
    FE::Real maximum_absolute_post_correction_drift{0.0};
    FE::Real minimum_absolute_pre_correction_drift =
        std::numeric_limits<FE::Real>::max();

    for (const std::size_t cells_per_axis : {12u, 24u, 48u}) {
        StructuredTriLevelSetFixture fixture(
            cells_per_axis, distortion_amplitude);
        const auto field_offset = static_cast<std::size_t>(
            fixture.system.fieldDofOffset(fixture.level_set_field));
        std::vector<FE::Real> target_solution(
            static_cast<std::size_t>(
                fixture.system.dofHandler().getNumDofs()),
            FE::Real{0.0});
        FE::Real minimum_absolute_vertex_value =
            std::numeric_limits<FE::Real>::max();
        for (FE::GlobalIndex vertex = 0;
             vertex < fixture.mesh->numVertices(); ++vertex) {
            const auto point = fixture.mesh->getNodeCoordinates(vertex);
            const FE::Real dx = point[0] - translated_center_x;
            const FE::Real dy = point[1] - center_y;
            const FE::Real value = std::sqrt(dx * dx + dy * dy) - radius;
            const auto local_dof = fixture.vertex_dofs.at(
                static_cast<std::size_t>(vertex));
            target_solution[field_offset +
                            static_cast<std::size_t>(local_dof)] = value;
            minimum_absolute_vertex_value = std::min(
                minimum_absolute_vertex_value, std::abs(value));
        }
        ASSERT_GT(minimum_absolute_vertex_value, FE::Real{1.0e-8});

        level_set::LevelSetVolumeOptions volume_options{};
        volume_options.use_generated_interface_quadrature = true;
        volume_options.level_set_field_name = "level_set";
        volume_options.generated_domain_id =
            "distorted_translating_disk_correction_" +
            std::to_string(cells_per_axis);
        volume_options.interface_quadrature_order = 4;
        volume_options.volume_quadrature_order = 4;

        const auto target_volume =
            level_set::computeLevelSetCutCellVolume(
                fixture.system,
                fixture.level_set_field,
                volume_options,
                target_solution);
        ASSERT_TRUE(target_volume.success) << target_volume.diagnostic;
        target_measure_errors.push_back(std::abs(
            target_volume.negative_volume - analytic_measure));

        level_set::LevelSetGeneratedInterfaceOptions interface_options{};
        interface_options.level_set_field_name = "level_set";
        interface_options.domain_id = volume_options.generated_domain_id;
        interface_options.requested_interface_marker =
            4700 + static_cast<int>(cells_per_axis);
        interface_options.interface_quadrature_order = 4;
        interface_options.volume_quadrature_order = 4;
        level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        auto generated = lifecycle.build(
            fixture.system, interface_options, target_solution);
        ASSERT_TRUE(generated.success) << generated.diagnostic;

        FE::interfaces::FreeSurfaceGeometrySnapshotPolicy snapshot_policy;
        snapshot_policy.require_complete_exterior_boundary_partition = false;
        FE::interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
        scalar.value = [mesh = fixture.mesh,
                        vertex_dofs = fixture.vertex_dofs,
                        target_solution,
                        field_offset](
                           FE::GlobalIndex cell,
                           const std::array<FE::Real, 3>& xi,
                           const FE::geometry::CutQuadratureProvenance&) {
            std::vector<FE::GlobalIndex> nodes;
            mesh->getCellNodes(cell, nodes);
            std::array<FE::Real, 3> values{};
            for (std::size_t local = 0u; local < values.size(); ++local) {
                const auto vertex = static_cast<std::size_t>(nodes.at(local));
                values[local] = target_solution[
                    field_offset + static_cast<std::size_t>(
                                       vertex_dofs.at(vertex))];
            }
            return (FE::Real{1.0} - xi[0] - xi[1]) * values[0] +
                   xi[0] * values[1] + xi[1] * values[2];
        };
        scalar.reference_gradient =
            [mesh = fixture.mesh,
             vertex_dofs = fixture.vertex_dofs,
             target_solution,
             field_offset](
                FE::GlobalIndex cell,
                const std::array<FE::Real, 3>&,
                const FE::geometry::CutQuadratureProvenance&) {
                std::vector<FE::GlobalIndex> nodes;
                mesh->getCellNodes(cell, nodes);
                std::array<FE::Real, 3> values{};
                for (std::size_t local = 0u; local < values.size(); ++local) {
                    const auto vertex =
                        static_cast<std::size_t>(nodes.at(local));
                    values[local] = target_solution[
                        field_offset + static_cast<std::size_t>(
                                           vertex_dofs.at(vertex))];
                }
                return std::array<FE::Real, 3>{
                    values[1] - values[0],
                    values[2] - values[0],
                    FE::Real{0.0},
                };
            };
        const auto snapshot =
            FE::interfaces::buildFreeSurfaceGeometrySnapshot(
                std::move(generated.domain),
                {},
                {},
                fixture.system.meshAccess(),
                snapshot_policy,
                std::move(scalar),
                volume_options.generated_domain_id);
        ASSERT_TRUE(snapshot);
        const FE::Real snapshot_measure =
            snapshot->ledger().owned_retained_negative_physical_volume;
        maximum_snapshot_measure_difference = std::max(
            maximum_snapshot_measure_difference,
            std::abs(snapshot_measure - target_volume.negative_volume));
        EXPECT_NEAR(snapshot_measure,
                    target_volume.negative_volume,
                    FE::Real{2.0e-11});
        EXPECT_NEAR(snapshot->ledger().maximum_constant_moment_error,
                    FE::Real{0.0},
                    FE::Real{2.0e-12});

        const FE::Real spacing = FE::Real{1.0} /
                                 static_cast<FE::Real>(cells_per_axis);
        const FE::Real imposed_drift = std::min(
            FE::Real{0.04} * spacing,
            FE::Real{0.25} * minimum_absolute_vertex_value);
        ASSERT_GT(imposed_drift, FE::Real{1.0e-8});
        auto drifted_solution = target_solution;
        for (const auto local_dof : fixture.vertex_dofs) {
            drifted_solution[
                field_offset + static_cast<std::size_t>(local_dof)] +=
                imposed_drift;
        }
        const auto drifted_volume =
            level_set::computeLevelSetCutCellVolume(
                fixture.system,
                fixture.level_set_field,
                volume_options,
                drifted_solution);
        ASSERT_TRUE(drifted_volume.success) << drifted_volume.diagnostic;
        const FE::Real pre_correction_drift =
            drifted_volume.negative_volume - target_volume.negative_volume;
        minimum_absolute_pre_correction_drift = std::min(
            minimum_absolute_pre_correction_drift,
            std::abs(pre_correction_drift));
        EXPECT_LT(pre_correction_drift, FE::Real{0.0});

        level_set::LevelSetGlobalShiftCorrectionOptions correction_options{};
        correction_options.target_negative_volume =
            target_volume.negative_volume;
        correction_options.volume_tolerance = FE::Real{1.0e-10};
        correction_options.max_iterations = 80;
        correction_options.maximum_interface_displacement_fraction = 0.2;
        std::vector<FE::Real> corrected_solution;
        const auto correction =
            level_set::applyGlobalLevelSetShiftCorrection(
                fixture.system,
                fixture.level_set_field,
                volume_options,
                correction_options,
                drifted_solution,
                corrected_solution);
        ASSERT_TRUE(correction.success) << correction.diagnostic;
        ASSERT_TRUE(correction.correction_triggered);
        ASSERT_TRUE(correction.correction_applied);
        ASSERT_TRUE(correction.target_reached);
        EXPECT_FALSE(correction.limited_by_displacement_bound);
        EXPECT_NEAR(correction.applied_shift,
                    -imposed_drift,
                    FE::Real{2.0e-9});
        const FE::Real post_correction_drift =
            correction.corrected_negative_volume -
            target_volume.negative_volume;
        maximum_absolute_post_correction_drift = std::max(
            maximum_absolute_post_correction_drift,
            std::abs(post_correction_drift));
        EXPECT_LE(std::abs(post_correction_drift),
                  correction_options.volume_tolerance);

        const std::string suffix =
            "_N" + std::to_string(cells_per_axis);
        RecordProperty("runtime_target_measure" + suffix,
                       serializeReal(target_volume.negative_volume));
        RecordProperty("runtime_snapshot_measure" + suffix,
                       serializeReal(snapshot_measure));
        RecordProperty("runtime_drifted_measure" + suffix,
                       serializeReal(drifted_volume.negative_volume));
        RecordProperty("runtime_pre_correction_drift" + suffix,
                       serializeReal(pre_correction_drift));
        RecordProperty("runtime_imposed_shift" + suffix,
                       serializeReal(imposed_drift));
        RecordProperty("runtime_correction_shift" + suffix,
                       serializeReal(correction.applied_shift));
        RecordProperty("runtime_corrected_measure" + suffix,
                       serializeReal(correction.corrected_negative_volume));
        RecordProperty("runtime_post_correction_drift" + suffix,
                       serializeReal(post_correction_drift));
        RecordProperty("runtime_target_analytic_error" + suffix,
                       serializeReal(target_measure_errors.back()));
        RecordProperty("runtime_correction_iterations" + suffix,
                       std::to_string(correction.iterations));
    }

    ASSERT_EQ(target_measure_errors.size(), 3u);
    EXPECT_LT(target_measure_errors[1], target_measure_errors[0]);
    EXPECT_LT(target_measure_errors[2], target_measure_errors[1]);
    const FE::Real first_measure_order = observedOrder(
        target_measure_errors[0], target_measure_errors[1]);
    const FE::Real second_measure_order = observedOrder(
        target_measure_errors[1], target_measure_errors[2]);
    RecordProperty("runtime_target_measure_order_12_to_24",
                   serializeReal(first_measure_order));
    RecordProperty("runtime_target_measure_order_24_to_48",
                   serializeReal(second_measure_order));
    RecordProperty("runtime_maximum_snapshot_measure_difference",
                   serializeReal(maximum_snapshot_measure_difference));
    RecordProperty("runtime_minimum_absolute_pre_correction_drift",
                   serializeReal(minimum_absolute_pre_correction_drift));
    RecordProperty("runtime_maximum_absolute_post_correction_drift",
                   serializeReal(maximum_absolute_post_correction_drift));
    EXPECT_GT(first_measure_order, FE::Real{1.0});
    EXPECT_GT(second_measure_order, FE::Real{1.0});
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
     ThreeDimensionalReversibleDeformationSmoke)
{
    StructuredHexPhaseFixture fixture(6u);
    ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
    constexpr FE::Real final_time = 0.2;
    const auto initial = sphere(0.40, 0.40, 0.40, 0.26);
    const auto velocity = reversibleThreeDimensionalVelocity(final_time);
    const FE::Real courant_rate = maximumCourantRate(
        fixture, velocity, FE::Real{0.0});
    const int steps = stepsForCourant(
        final_time, courant_rate, FE::Real{0.4});
    const auto run = runTransport(
        fixture, initial, velocity, final_time, steps);
    expectConservativeRun(run);
    const FE::Real l1_error = weightedL1Error(
        fixture, run.final_phase, initial);
    EXPECT_LE(run.maximum_measure_error, 1.0e-4);
    EXPECT_LE(run.maximum_accounted_balance_error, 2.0e-11);
    EXPECT_LT(l1_error, 0.12);
    EXPECT_EQ(run.minimum_components, 1u);
    EXPECT_EQ(run.maximum_components, 1u);
    RecordProperty("three_dimensional_smoke_l1",
                   serializeReal(l1_error));
    RecordProperty("three_dimensional_smoke_measure_error",
                   serializeReal(run.maximum_measure_error));
    RecordProperty("three_dimensional_smoke_divergence_source",
                   serializeReal(run.cumulative_divergence_source));
    RecordProperty("three_dimensional_smoke_maximum_courant",
                   serializeReal(run.maximum_courant));
    RecordProperty("three_dimensional_smoke_steps",
                   std::to_string(steps));
}

TEST(LevelSetConservativePhaseBenchmarks,
     ThreeDimensionalGraphClosesMeasureAtReleaseEntryResolution)
{
    StructuredHexPhaseFixture fixture(32u);
    ASSERT_TRUE(fixture.graph.success)
        << fixture.graph.diagnostic
        << " measure_residual="
        << fixture.graph.measure_closure_residual
        << " physical_measure=" << fixture.graph.physical_measure
        << " lumped_measure="
        << fixture.graph.total_lumped_control_volume;
    EXPECT_EQ(fixture.graph.cells, 32u * 32u * 32u);
    EXPECT_EQ(fixture.graph.nodes, 33u * 33u * 33u);
    EXPECT_NEAR(fixture.graph.physical_measure, 1.0, 2.0e-14);
    EXPECT_NEAR(fixture.graph.total_lumped_control_volume,
                1.0, 2.0e-14);
    EXPECT_LE(std::abs(fixture.graph.measure_closure_residual),
              2.0e-14);
    RecordProperty("three_dimensional_graph_measure_residual",
                   serializeReal(fixture.graph.measure_closure_residual));
    RecordProperty("three_dimensional_graph_nodes",
                   std::to_string(fixture.graph.nodes));
    RecordProperty("three_dimensional_graph_edges",
                   std::to_string(fixture.graph.edges.size()));
}

TEST(LevelSetConservativePhaseQualification,
     RunsOneExplicitReleaseMatrixPoint)
{
    constexpr const char* case_variable =
        "SVMP_PHASE_TRANSPORT_RELEASE_CASE";
    constexpr const char* resolution_variable =
        "SVMP_PHASE_TRANSPORT_RELEASE_RESOLUTION";
    constexpr const char* courant_variable =
        "SVMP_PHASE_TRANSPORT_RELEASE_CFL";
    constexpr const char* history_variable =
        "SVMP_PHASE_TRANSPORT_RELEASE_HISTORY";
    constexpr const char* details_variable =
        "SVMP_PHASE_TRANSPORT_RELEASE_DETAILS";
    const auto requested_case = environmentValue(case_variable);
    const auto requested_resolution = environmentValue(resolution_variable);
    const auto requested_courant = environmentValue(courant_variable);
    const auto requested_history = environmentValue(history_variable);
    const auto requested_details = environmentValue(details_variable);
    if (!requested_case.has_value()) {
        EXPECT_FALSE(requested_resolution.has_value())
            << resolution_variable << " is set without " << case_variable;
        EXPECT_FALSE(requested_courant.has_value())
            << courant_variable << " is set without " << case_variable;
        EXPECT_FALSE(requested_history.has_value())
            << history_variable << " is set without " << case_variable;
        EXPECT_FALSE(requested_details.has_value())
            << details_variable << " is set without " << case_variable;
        GTEST_SKIP()
            << "Set the five release-matrix environment variables to run "
               "one scheduled qualification point.";
    }
    ASSERT_TRUE(requested_resolution.has_value())
        << resolution_variable << " is required when "
        << case_variable << " is set";
    ASSERT_TRUE(requested_courant.has_value())
        << courant_variable << " is required when "
        << case_variable << " is set";
    ASSERT_TRUE(requested_history.has_value())
        << history_variable << " is required when "
        << case_variable << " is set";
    ASSERT_TRUE(requested_details.has_value())
        << details_variable << " is required when "
        << case_variable << " is set";

    std::size_t resolution = 0u;
    FE::Real courant = 0.0;
    try {
        resolution = parsePositiveSize(
            *requested_resolution, resolution_variable);
        courant = parsePositiveReal(
            *requested_courant, courant_variable);
    } catch (const std::exception& exception) {
        FAIL() << exception.what();
    }
    ASSERT_TRUE(isReleaseCourant(courant))
        << courant_variable << " must be one of 0.5, 0.25, or 0.125";

    const auto record_common = [courant](
                                   const auto& fixture,
                                   const TransportRun& run,
                                   int steps,
                                   FE::Real l1_error) {
        ::testing::Test::RecordProperty(
            "requested_cfl", serializeReal(courant));
        ::testing::Test::RecordProperty(
            "achieved_graph_cfl", serializeReal(run.maximum_courant));
        ::testing::Test::RecordProperty(
            "time_steps", std::to_string(steps));
        ::testing::Test::RecordProperty(
            "graph_nodes", std::to_string(fixture.graph.nodes));
        ::testing::Test::RecordProperty(
            "graph_edges", std::to_string(fixture.graph.edges.size()));
        ::testing::Test::RecordProperty(
            "interface_l1", serializeReal(l1_error));
        ::testing::Test::RecordProperty(
            "maximum_raw_measure_error",
            serializeReal(run.maximum_measure_error));
        ::testing::Test::RecordProperty(
            "maximum_accounted_balance_error",
            serializeReal(run.maximum_accounted_balance_error));
        ::testing::Test::RecordProperty(
            "cumulative_boundary_transfer",
            serializeReal(run.cumulative_boundary_transfer));
        ::testing::Test::RecordProperty(
            "cumulative_discrete_divergence_source",
            serializeReal(run.cumulative_divergence_source));
        ::testing::Test::RecordProperty(
            "minimum_indicator", serializeReal(run.minimum_indicator));
        ::testing::Test::RecordProperty(
            "maximum_indicator", serializeReal(run.maximum_indicator));
        ::testing::Test::RecordProperty(
            "maximum_local_balance_residual",
            serializeReal(run.maximum_local_balance_residual));
        ::testing::Test::RecordProperty(
            "minimum_components",
            std::to_string(run.minimum_components));
        ::testing::Test::RecordProperty(
            "maximum_components",
            std::to_string(run.maximum_components));
    };

    RecordProperty("matrix_case", *requested_case);
    RecordProperty("resolution", std::to_string(resolution));
    if (*requested_case == "translating_drop_2d") {
        ASSERT_TRUE(resolution == 64u || resolution == 128u ||
                    resolution == 256u)
            << "translating_drop_2d requires resolution 64, 128, or 256";
        StructuredPhaseFixture fixture(resolution);
        ASSERT_TRUE(fixture.graph.success) << fixture.graph.diagnostic;
        constexpr FE::Real final_time = 0.5;
        constexpr FE::Real speed = 0.2;
        constexpr FE::Real diameter = 0.25;
        const auto initial = disk(0.30, 0.50, diameter / FE::Real{2.0});
        const auto exact = disk(
            0.30 + speed * final_time,
            0.50,
            diameter / FE::Real{2.0});
        const VelocityField velocity = [](
            FE::Real /*time*/,
            const std::array<FE::Real, 3>& /*point*/) {
            return std::array<FE::Real, 3>{speed, 0.0, 0.0};
        };
        const int steps = stepsForCourant(
            final_time,
            maximumCourantRate(fixture, velocity, FE::Real{0.0}),
            courant);
        const auto run = runTransport(
            fixture, initial, velocity, final_time, steps);
        expectConservativeRun(run);
        EXPECT_LE(run.maximum_courant, courant + 2.0e-12);
        EXPECT_LE(run.maximum_measure_error, 1.0e-8);
        EXPECT_EQ(run.minimum_components, 1u);
        EXPECT_EQ(run.maximum_components, 1u);
        const FE::Real l1_error = weightedL1Error(
            fixture, run.final_phase, exact);
        EXPECT_LT(l1_error, 0.12);
        ASSERT_NO_THROW(writeTransportHistory(
            *requested_history, run));
        ASSERT_NO_THROW(writeFinalTransportDetails(
            *requested_details, fixture, run));
        record_common(fixture, run, steps, l1_error);
        RecordProperty("history_points",
                       std::to_string(run.history.size()));
        RecordProperty("detail_control_volumes",
                       std::to_string(
                           run.final_stage->correction.nodes.size()));
        RecordProperty("detail_edges",
                       std::to_string(
                           run.final_stage->correction.edges.size()));
        const auto initial_centroid = phaseCentroid(
            fixture, run.initial_phase);
        const auto final_centroid = phaseCentroid(
            fixture, run.final_phase);
        RecordProperty("diameter_over_dx",
                       serializeReal(diameter * resolution));
        RecordProperty("centroid_x_error",
                       serializeReal(std::abs(
                           final_centroid[0] - initial_centroid[0] -
                           speed * final_time)));
        RecordProperty("centroid_y_error",
                       serializeReal(std::abs(
                           final_centroid[1] - initial_centroid[1])));
        return;
    }

    ASSERT_EQ(*requested_case, "enright_3d")
        << case_variable
        << " must be translating_drop_2d or enright_3d";
    ASSERT_TRUE(resolution == 32u || resolution == 64u ||
                resolution == 128u)
        << "enright_3d requires resolution 32, 64, or 128";
    StructuredHexPhaseFixture fixture(resolution);
    ASSERT_TRUE(fixture.graph.success)
        << fixture.graph.diagnostic
        << " partition_residual="
        << fixture.graph.maximum_partition_of_unity_residual
        << " gradient_partition_residual="
        << fixture.graph.maximum_gradient_partition_residual
        << " row_sum_residual="
        << fixture.graph.maximum_gradient_row_sum_residual
        << " measure_residual="
        << fixture.graph.measure_closure_residual
        << " physical_measure=" << fixture.graph.physical_measure
        << " lumped_measure="
        << fixture.graph.total_lumped_control_volume;
    constexpr FE::Real final_time = 3.0;
    const auto initial = sphere(0.35, 0.35, 0.35, 0.15);
    const auto velocity = reversibleThreeDimensionalVelocity(final_time);
    const int steps = stepsForCourant(
        final_time,
        maximumCourantRate(fixture, velocity, FE::Real{0.0}),
        courant);
    const auto run = runTransport(
        fixture, initial, velocity, final_time, steps);
    expectConservativeRun(run);
    EXPECT_LE(run.maximum_courant, courant + 2.0e-12);
    EXPECT_LE(run.maximum_measure_error, 5.0e-4);
    EXPECT_LE(run.maximum_accounted_balance_error, 2.0e-11);
    const FE::Real l1_error = weightedL1Error(
        fixture, run.final_phase, initial);
    EXPECT_LT(l1_error, 0.35);
    ASSERT_NO_THROW(writeTransportHistory(
        *requested_history, run));
    ASSERT_NO_THROW(writeFinalTransportDetails(
        *requested_details, fixture, run));
    record_common(fixture, run, steps, l1_error);
    RecordProperty("history_points",
                   std::to_string(run.history.size()));
    RecordProperty("detail_control_volumes",
                   std::to_string(
                       run.final_stage->correction.nodes.size()));
    RecordProperty("detail_edges",
                   std::to_string(
                       run.final_stage->correction.edges.size()));
    const auto initial_centroid = phaseCentroid(
        fixture, run.initial_phase);
    const auto final_centroid = phaseCentroid(
        fixture, run.final_phase);
    RecordProperty("centroid_return_error",
                   serializeReal(std::sqrt(
                       std::pow(final_centroid[0] - initial_centroid[0], 2) +
                       std::pow(final_centroid[1] - initial_centroid[1], 2) +
                       std::pow(final_centroid[2] - initial_centroid[2], 2))));
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
