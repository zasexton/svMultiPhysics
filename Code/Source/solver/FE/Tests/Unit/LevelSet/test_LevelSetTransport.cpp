#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVelocityExtensionConstraint.h"

#include "Assembly/Assembler.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/SpaceFactory.h"
#include "Sparsity/SparsityPattern.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"
#include "Systems/TimeIntegrator.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/TimeSteppingUtils.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

using FE::forms::FormExpr;
using FE::forms::FormExprNode;
using FE::forms::FormExprType;

class ScopedEnvironmentVariable final {
public:
    ScopedEnvironmentVariable(const char* key, const char* value)
        : key_(key)
    {
        if (const char* current = std::getenv(key); current != nullptr) {
            prior_ = std::string(current);
        }
        ::setenv(key_.c_str(), value, 1);
    }

    ~ScopedEnvironmentVariable()
    {
        if (prior_.has_value()) {
            ::setenv(key_.c_str(), prior_->c_str(), 1);
        } else {
            ::unsetenv(key_.c_str());
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(
        const ScopedEnvironmentVariable&) = delete;

private:
    std::string key_;
    std::optional<std::string> prior_;
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::shared_ptr<svmp::Mesh> buildNativeQuad9Mesh()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
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
                            x_ref,
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

class SingleTriangleMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTriangleMeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{2.0, 0.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
        };
        cell_ = {0, 1, 2};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override
    {
        return FE::ElementType::Triangle3;
    }

    void getCellNodes(FE::GlobalIndex,
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
        FE::GlobalIndex,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords = nodes_;
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex,
        FE::GlobalIndex) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override
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
        int,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)>) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex,
                           FE::GlobalIndex,
                           FE::GlobalIndex)>) const override
    {
    }

private:
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::array<FE::GlobalIndex, 3> cell_{};
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

class Quad9Patch2x2MeshAccess final : public FE::assembly::IMeshAccess {
public:
    Quad9Patch2x2MeshAccess()
    {
        nodes_.reserve(25u);
        for (int j = 0; j < 5; ++j) {
            for (int i = 0; i < 5; ++i) {
                nodes_.push_back(std::array<FE::Real, 3>{
                    FE::Real{0.25} * static_cast<FE::Real>(i),
                    FE::Real{0.25} * static_cast<FE::Real>(j),
                    0.0});
            }
        }

        for (int cy = 0; cy < 2; ++cy) {
            for (int cx = 0; cx < 2; ++cx) {
                const auto node = [](int ix, int iy) -> FE::GlobalIndex {
                    return static_cast<FE::GlobalIndex>(5 * iy + ix);
                };
                const int i = 2 * cx;
                const int j = 2 * cy;
                cells_.push_back(std::array<FE::GlobalIndex, 9>{
                    node(i, j),
                    node(i + 2, j),
                    node(i + 2, j + 2),
                    node(i, j + 2),
                    node(i + 1, j),
                    node(i + 2, j + 1),
                    node(i + 1, j + 2),
                    node(i, j + 1),
                    node(i + 1, j + 1)});
            }
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(cells_.size());
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Quad9;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex cell_id,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        std::vector<FE::GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.clear();
        coords.reserve(nodes.size());
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
        }
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
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
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
    std::vector<std::array<FE::GlobalIndex, 9>> cells_{};
};

class StructuredQuadTransportMeshAccess final
    : public FE::assembly::IMeshAccess {
public:
    static constexpr int kWallMarker = 31;
    static constexpr int kInflowMarker = 32;
    static constexpr int kOutflowMarker = 33;

    explicit StructuredQuadTransportMeshAccess(int cells_per_axis)
        : n_(cells_per_axis)
    {
        if (n_ <= 0) {
            throw std::invalid_argument(
                "StructuredQuadTransportMeshAccess requires a positive cell count");
        }
        const int nodes_per_axis = n_ + 1;
        nodes_.resize(static_cast<std::size_t>(nodes_per_axis * nodes_per_axis));
        for (int j = 0; j < nodes_per_axis; ++j) {
            for (int i = 0; i < nodes_per_axis; ++i) {
                nodes_[static_cast<std::size_t>(nodeId(i, j))] = {
                    static_cast<FE::Real>(i) / static_cast<FE::Real>(n_),
                    static_cast<FE::Real>(j) / static_cast<FE::Real>(n_),
                    0.0};
            }
        }
        cells_.resize(static_cast<std::size_t>(n_ * n_));
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < n_; ++i) {
                cells_[static_cast<std::size_t>(cellId(i, j))] = {
                    nodeId(i, j), nodeId(i + 1, j),
                    nodeId(i + 1, j + 1), nodeId(i, j + 1)};
            }
        }
        // Quad4 local faces: bottom, right, top, left.
        for (int i = 0; i < n_; ++i) {
            boundary_faces_.push_back(
                {cellId(i, 0), FE::LocalIndex{0}, kWallMarker});
            boundary_faces_.push_back(
                {cellId(i, n_ - 1), FE::LocalIndex{2}, kWallMarker});
        }
        for (int j = 0; j < n_; ++j) {
            boundary_faces_.push_back(
                {cellId(0, j), FE::LocalIndex{3}, kInflowMarker});
            boundary_faces_.push_back(
                {cellId(n_ - 1, j), FE::LocalIndex{1}, kOutflowMarker});
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(cells_.size());
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return static_cast<FE::GlobalIndex>(boundary_faces_.size());
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell*/) const override
    {
        return true;
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex /*cell*/) const override
    {
        return FE::ElementType::Quad4;
    }
    void getCellNodes(FE::GlobalIndex cell,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& record = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(record.begin(), record.end());
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node) const override
    {
        return nodes_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        FE::GlobalIndex cell,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        const auto& record = cells_.at(static_cast<std::size_t>(cell));
        coordinates.clear();
        coordinates.reserve(record.size());
        for (const auto node : record) {
            coordinates.push_back(getNodeCoordinates(node));
        }
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face,
        FE::GlobalIndex cell) const override
    {
        const auto& record = boundary_faces_.at(static_cast<std::size_t>(face));
        if (record.cell != cell) {
            throw std::invalid_argument(
                "StructuredQuadTransportMeshAccess face/cell mismatch");
        }
        return record.local_face;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex face) const override
    {
        return boundary_faces_.at(static_cast<std::size_t>(face)).marker;
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
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex face = 0; face < numBoundaryFaces(); ++face) {
            const auto& record =
                boundary_faces_[static_cast<std::size_t>(face)];
            if (marker < 0 || marker == record.marker) {
                callback(face, record.cell);
            }
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex,
                           FE::GlobalIndex)> /*callback*/) const override
    {
    }
    [[nodiscard]] int cellsPerAxis() const noexcept { return n_; }
    [[nodiscard]] FE::GlobalIndex nodeId(int i, int j) const
    {
        return static_cast<FE::GlobalIndex>(i + (n_ + 1) * j);
    }

private:
    struct BoundaryFace {
        FE::GlobalIndex cell{-1};
        FE::LocalIndex local_face{0};
        int marker{-1};
    };
    [[nodiscard]] FE::GlobalIndex cellId(int i, int j) const
    {
        return static_cast<FE::GlobalIndex>(i + n_ * j);
    }

    int n_{1};
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::vector<std::array<FE::GlobalIndex, 4>> cells_{};
    std::vector<BoundaryFace> boundary_faces_{};
};

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> scalarSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
}

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> scalarSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh,
    int order)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, order, /*components=*/1);
}

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> vectorSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
}

bool containsExprType(const FormExprNode* node, FormExprType target)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == target) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsExprType(child, target)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsContain(const FE::systems::FESystem& system,
                               FormExprType target)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsExprType(record.residual_expr.get(), target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsExprType(expr.get(), target)) {
                return true;
            }
        }
    }
    return false;
}

void addScalarAndVelocityFields(FE::systems::FESystem& system,
                                const std::shared_ptr<const FE::spaces::FunctionSpace>& scalar_space,
                                const std::shared_ptr<const FE::spaces::FunctionSpace>& velocity_space,
                                FE::systems::FieldSourceKind velocity_source =
                                    FE::systems::FieldSourceKind::PrescribedData)
{
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
        .source_kind = velocity_source,
    });
}

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

[[nodiscard]] FE::systems::SetupInputs makeSingleTriangleSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 3;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 3};
    topo.cell2vertex_data = {0, 1, 2};
    topo.vertex_gids = {0, 1, 2};
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

[[nodiscard]] FE::systems::SetupInputs makeQuad9Patch2x2SetupInputs()
{
    const Quad9Patch2x2MeshAccess mesh;
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 4;
    topo.n_vertices = 25;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets.reserve(5u);
    topo.cell2vertex_offsets.push_back(0);
    for (FE::GlobalIndex cell = 0; cell < 4; ++cell) {
        std::vector<FE::GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        topo.cell2vertex_data.insert(
            topo.cell2vertex_data.end(),
            nodes.begin(),
            nodes.end());
        topo.cell2vertex_offsets.push_back(
            static_cast<FE::GlobalIndex>(topo.cell2vertex_data.size()));
    }
    for (FE::GlobalIndex vertex = 0; vertex < 25; ++vertex) {
        topo.vertex_gids.push_back(vertex);
    }
    for (FE::GlobalIndex cell = 0; cell < 4; ++cell) {
        topo.cell_gids.push_back(cell);
        topo.cell_owner_ranks.push_back(0);
    }

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupInputs makeStructuredQuadTransportSetupInputs(
    const StructuredQuadTransportMeshAccess& mesh)
{
    const int n = mesh.cellsPerAxis();
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = static_cast<FE::GlobalIndex>(n * n);
    topology.n_vertices = static_cast<FE::GlobalIndex>((n + 1) * (n + 1));
    topology.dim = 2;
    topology.cell2vertex_offsets.reserve(
        static_cast<std::size_t>(topology.n_cells) + 1u);
    topology.cell2vertex_offsets.push_back(0);
    for (FE::GlobalIndex cell = 0; cell < topology.n_cells; ++cell) {
        std::vector<FE::GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        topology.cell2vertex_data.insert(topology.cell2vertex_data.end(),
                                         nodes.begin(), nodes.end());
        topology.cell2vertex_offsets.push_back(
            static_cast<FE::MeshOffset>(topology.cell2vertex_data.size()));
        topology.cell_gids.push_back(cell);
        topology.cell_owner_ranks.push_back(0);
    }
    for (FE::GlobalIndex vertex = 0; vertex < topology.n_vertices; ++vertex) {
        topology.vertex_gids.push_back(vertex);
    }
    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

std::vector<FE::Real> constantVectorTetraCoefficients(FE::Real x,
                                                      FE::Real y,
                                                      FE::Real z)
{
    std::vector<FE::Real> coefficients(12u, 0.0);
    for (std::size_t node = 0; node < 4u; ++node) {
        coefficients[node] = x;
        coefficients[4u + node] = y;
        coefficients[8u + node] = z;
    }
    return coefficients;
}

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            int component,
                            FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("setFieldComponentValue: component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        dofs[static_cast<std::size_t>(component)] + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

void setScalarVertexValue(std::vector<FE::Real>& solution,
                          const FE::systems::FESystem& system,
                          FE::FieldId field,
                          FE::GlobalIndex vertex,
                          FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setScalarVertexValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::runtime_error("setScalarVertexValue: expected one scalar vertex DOF");
    }
    const auto index = static_cast<std::size_t>(dofs.front() + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setScalarVertexValue: DOF index is out of range");
    }
    solution[index] = value;
}

[[nodiscard]] std::vector<FE::Real> assembleLevelSetResidual(
    FE::systems::FESystem& system,
    const FE::systems::SystemStateView& state)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();

    FE::systems::AssemblyRequest request;
    request.op = "level_set";
    request.want_vector = true;
    const auto result = system.assemble(request, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    std::vector<FE::Real> out(static_cast<std::size_t>(n), 0.0);
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = residual.getVectorEntry(i);
    }
    return out;
}

[[nodiscard]] std::vector<FE::Real> solveDenseSystem(
    std::vector<FE::Real> matrix,
    std::vector<FE::Real> rhs)
{
    const std::size_t n = rhs.size();
    if (matrix.size() != n * n) {
        throw std::invalid_argument("solveDenseSystem matrix size mismatch");
    }
    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        FE::Real pivot_magnitude =
            std::abs(matrix[column * n + column]);
        for (std::size_t row = column + 1u; row < n; ++row) {
            const FE::Real magnitude =
                std::abs(matrix[row * n + column]);
            if (magnitude > pivot_magnitude) {
                pivot = row;
                pivot_magnitude = magnitude;
            }
        }
        if (!(pivot_magnitude > 1.0e-14) || !std::isfinite(pivot_magnitude)) {
            throw std::runtime_error("solveDenseSystem encountered a singular pivot");
        }
        if (pivot != column) {
            for (std::size_t j = column; j < n; ++j) {
                std::swap(matrix[column * n + j], matrix[pivot * n + j]);
            }
            std::swap(rhs[column], rhs[pivot]);
        }
        const FE::Real diagonal = matrix[column * n + column];
        for (std::size_t row = column + 1u; row < n; ++row) {
            const FE::Real multiplier = matrix[row * n + column] / diagonal;
            matrix[row * n + column] = 0.0;
            for (std::size_t j = column + 1u; j < n; ++j) {
                matrix[row * n + j] -= multiplier * matrix[column * n + j];
            }
            rhs[row] -= multiplier * rhs[column];
        }
    }
    std::vector<FE::Real> solution(n, 0.0);
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const std::size_t row = n - 1u - reverse;
        FE::Real value = rhs[row];
        for (std::size_t column = row + 1u; column < n; ++column) {
            value -= matrix[row * n + column] * solution[column];
        }
        solution[row] = value / matrix[row * n + row];
    }
    return solution;
}

[[nodiscard]] std::vector<FE::Real> solveLinearLevelSetStep(
    FE::systems::FESystem& system,
    std::span<const FE::Real> previous,
    FE::Real dt)
{
    const auto n = system.dofHandler().getNumDofs();
    if (n <= 0 || previous.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "solveLinearLevelSetStep incompatible previous solution");
    }
    std::vector<FE::Real> candidate(previous.begin(), previous.end());
    FE::systems::SystemStateView state;
    state.dt = dt;
    state.u = std::span<const FE::Real>(candidate);
    state.u_prev = previous;
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    FE::assembly::DenseMatrixView jacobian(n);
    FE::assembly::DenseVectorView residual(n);
    jacobian.zero();
    residual.zero();
    FE::systems::AssemblyRequest request;
    request.op = "level_set";
    request.want_matrix = true;
    request.want_vector = true;
    const auto assembled = system.assemble(
        request, state, &jacobian, &residual);
    if (!assembled.success) {
        throw std::runtime_error(assembled.error_message);
    }
    std::vector<FE::Real> matrix(static_cast<std::size_t>(n * n), 0.0);
    std::vector<FE::Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (FE::GlobalIndex row = 0; row < n; ++row) {
        rhs[static_cast<std::size_t>(row)] = residual.getVectorEntry(row);
        for (FE::GlobalIndex column = 0; column < n; ++column) {
            matrix[static_cast<std::size_t>(row * n + column)] =
                jacobian(row, column);
        }
    }
    const auto update = solveDenseSystem(std::move(matrix), std::move(rhs));
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        candidate[i] -= update[i];
    }
    return candidate;
}

template <typename Function>
void fillScalarFieldAtStructuredVertices(
    std::vector<FE::Real>& solution,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    const StructuredQuadTransportMeshAccess& mesh,
    Function&& value)
{
    const int n = mesh.cellsPerAxis();
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            const auto node = mesh.nodeId(i, j);
            const auto point = mesh.getNodeCoordinates(node);
            setScalarVertexValue(solution, system, field, node,
                                 value(point[0], point[1]));
        }
    }
}

[[nodiscard]] FE::Real scalarVertexValue(
    std::span<const FE::Real> solution,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    FE::GlobalIndex vertex)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("scalarVertexValue missing entity map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::runtime_error("scalarVertexValue expected a scalar vertex DOF");
    }
    const auto index = static_cast<std::size_t>(
        system.fieldDofOffset(field) + dofs.front());
    return solution[index];
}

[[nodiscard]] FE::Real l2Norm(std::span<const FE::Real> values)
{
    FE::Real sum = 0.0;
    for (const auto value : values) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

void expectOperatorJacobianMatchesCentralFD(FE::systems::FESystem& system,
                                            const FE::systems::SystemStateView& base_state,
                                            FE::Real eps,
                                            FE::Real rtol,
                                            FE::Real atol)
{
    const auto n = system.dofHandler().getNumDofs();
    ASSERT_GT(n, 0);

    const std::vector<FE::Real> base_u(base_state.u.begin(), base_state.u.end());
    ASSERT_EQ(static_cast<FE::GlobalIndex>(base_u.size()), n);

    FE::assembly::DenseMatrixView jacobian(n);
    {
        FE::systems::AssemblyRequest request;
        request.op = "level_set";
        request.want_matrix = true;
        const auto result = system.assemble(request, base_state, &jacobian, nullptr);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    for (FE::GlobalIndex column = 0; column < n; ++column) {
        std::vector<FE::Real> u_plus = base_u;
        std::vector<FE::Real> u_minus = base_u;
        u_plus[static_cast<std::size_t>(column)] += eps;
        u_minus[static_cast<std::size_t>(column)] -= eps;

        FE::systems::SystemStateView state_plus = base_state;
        FE::systems::SystemStateView state_minus = base_state;
        state_plus.u = std::span<const FE::Real>(u_plus);
        state_minus.u = std::span<const FE::Real>(u_minus);

        FE::assembly::DenseVectorView r_plus(n);
        FE::assembly::DenseVectorView r_minus(n);
        {
            FE::systems::AssemblyRequest request;
            request.op = "level_set";
            request.want_vector = true;
            const auto result = system.assemble(request, state_plus, nullptr, &r_plus);
            ASSERT_TRUE(result.success) << result.error_message;
        }
        {
            FE::systems::AssemblyRequest request;
            request.op = "level_set";
            request.want_vector = true;
            const auto result = system.assemble(request, state_minus, nullptr, &r_minus);
            ASSERT_TRUE(result.success) << result.error_message;
        }

        for (FE::GlobalIndex row = 0; row < n; ++row) {
            const FE::Real finite_difference = (r_plus[row] - r_minus[row]) / (2.0 * eps);
            const FE::Real assembled = jacobian(row, column);
            const FE::Real tolerance =
                atol + rtol * std::max<FE::Real>(1.0, std::abs(finite_difference));
            EXPECT_NEAR(assembled, finite_difference, tolerance)
                << "Mismatch at (row=" << row << ", column=" << column << ")";
        }
    }
}

} // namespace

TEST(LevelSetTransport, ValidatesFieldOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem scalar_system(mesh);
    scalar_system.addField(FE::systems::FieldSpec{
        .name = "level_set",
        .space = phi_space,
        .components = 1,
    });
    scalar_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(scalar_system, phi_space, {}));

    FE::systems::FESystem vector_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(vector_system, velocity_space, {}),
        std::invalid_argument);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name.clear();
    FE::systems::FESystem empty_name_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(empty_name_system, phi_space, options),
        std::invalid_argument);

    options.level_set.field_name = "phi";
    options.velocity.field_name.clear();
    FE::systems::FESystem empty_velocity_name_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            empty_velocity_name_system,
            phi_space,
            options),
        std::invalid_argument);
}

TEST(LevelSetTransport, AutoRegistersConfiguredFields)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(phi).source_kind, FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(velocity).source_kind,
              FE::systems::FieldSourceKind::PrescribedData);
    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport,
     RegistersConservativePhaseStateWithAnAlgebraicEndpointHold)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    FE::systems::FESystem system(mesh);
    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    options.conservative_phase.enabled = true;
    options.conservative_phase.liquid_indicator.field_name = "phase";

    const auto kernels = level_set::installLevelSetTransport(
        system, phi_space, options);

    const auto phi = system.findFieldByName("phi");
    const auto phase = system.findFieldByName("phase");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(phase, FE::INVALID_FIELD_ID);
    EXPECT_NE(phi, phase);
    EXPECT_TRUE(system.fieldParticipatesInUnknownVector(phase));
    const auto& phase_record = system.fieldRecord(phase);
    ASSERT_TRUE(phase_record.space);
    EXPECT_EQ(phase_record.space->space_type(), FE::spaces::SpaceType::H1);
    EXPECT_EQ(phase_record.space->polynomial_order(), 1);
    EXPECT_TRUE(formulationRecordsContain(
        system, FormExprType::PreviousSolutionRef));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport,
     ConservativePhaseEndpointHoldAssemblesOnlyTheCurrentMinusPreviousState)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    FE::systems::FESystem system(mesh);
    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    options.conservative_phase.enabled = true;
    options.conservative_phase.liquid_indicator.field_name = "phase";
    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto phase = system.findFieldByName("phase");
    ASSERT_NE(phase, FE::INVALID_FIELD_ID);
    const auto phase_offset = static_cast<std::size_t>(
        system.fieldDofOffset(phase));
    const auto phase_count = static_cast<std::size_t>(
        system.fieldDofHandler(phase).getNumDofs());
    ASSERT_EQ(phase_count, 4u);

    const auto total_dofs = static_cast<std::size_t>(
        system.dofHandler().getNumDofs());
    std::vector<FE::Real> previous(total_dofs, FE::Real{0.25});
    for (std::size_t i = 0u; i < phase_count; ++i) {
        previous[phase_offset + i] =
            FE::Real{0.1} + FE::Real{0.2} * static_cast<FE::Real>(i);
    }
    auto current = previous;
    FE::systems::SystemStateView state;
    state.dt = FE::Real{0.1};
    state.u = std::span<const FE::Real>(current);
    state.u_prev = std::span<const FE::Real>(previous);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    const auto held_residual = assembleLevelSetResidual(system, state);
    for (std::size_t i = 0u; i < phase_count; ++i) {
        EXPECT_NEAR(held_residual[phase_offset + i], FE::Real{0.0},
                    FE::Real{1.0e-13});
    }

    current[phase_offset] += FE::Real{0.4};
    state.u = std::span<const FE::Real>(current);
    const auto perturbed_residual = assembleLevelSetResidual(system, state);
    FE::Real phase_residual_norm = FE::Real{0.0};
    for (std::size_t i = 0u; i < phase_count; ++i) {
        const auto value = perturbed_residual[phase_offset + i];
        phase_residual_norm += value * value;
    }
    EXPECT_GT(std::sqrt(phase_residual_norm), FE::Real{1.0e-8});
}

TEST(LevelSetTransport,
     ConservativePhaseConfigurationFailsBeforeFieldMutation)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    const auto expect_rejected_without_fields = [&](auto configure) {
        FE::systems::FESystem system(mesh);
        level_set::LevelSetTransportOptions options{};
        options.level_set.field_name = "phi";
        options.velocity.source =
            level_set::LevelSetVelocitySource::ConstantVector;
        options.conservative_phase.enabled = true;
        configure(options);
        EXPECT_THROW(
            (void)level_set::installLevelSetTransport(
                system, phi_space, options),
            std::invalid_argument);
        EXPECT_EQ(system.findFieldByName("phi"), FE::INVALID_FIELD_ID);
        EXPECT_EQ(system.findFieldByName("liquid_indicator"),
                  FE::INVALID_FIELD_ID);
        EXPECT_FALSE(system.hasOperator(options.operator_tag));
    };

    expect_rejected_without_fields([](auto& options) {
        options.conservative_phase.liquid_indicator.field_name = "phi";
    });
    expect_rejected_without_fields([](auto& options) {
        options.boundaries.outflow.push_back(
            level_set::LevelSetOutflowBoundary{.boundary_marker = 4});
    });
    expect_rejected_without_fields([](auto& options) {
        options.bound_preserving.enabled = true;
    });
    expect_rejected_without_fields([](auto& options) {
        options.volume_correction.enabled = true;
    });
    expect_rejected_without_fields([](auto& options) {
        options.conservative_phase.maximum_courant = 1.01;
    });
    expect_rejected_without_fields([](auto& options) {
        options.conservative_phase.component_activity_tolerance = 0.0;
    });
    expect_rejected_without_fields([](auto& options) {
        options.conservative_phase.flux_artifact_cadence_steps = 0;
    });
    expect_rejected_without_fields([](auto& options) {
        options.conservative_phase
            .pointwise_impermeable_velocity_tolerance_explicitly_requested =
            true;
    });
    expect_rejected_without_fields([](auto& options) {
        options.conservative_phase.fixed_flux_regions.push_back(
            level_set::LevelSetPhaseRegionBox{});
    });
    expect_rejected_without_fields([](auto& options) {
        options.conservative_phase.geometry_measure_tolerance = 0.0;
    });
}

TEST(LevelSetTransport, InstallsOnConfiguredOperatorTag)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.operator_tag = "equations";
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "Velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_FALSE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, AutoRegistersCoupledVelocityAsUnknownWhenRequested)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "Velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("Velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(phi).source_kind, FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(velocity).source_kind,
              FE::systems::FieldSourceKind::Unknown);
    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, InstallsResidualFormStructure)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::Divergence));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CellDiameter));
}

TEST(LevelSetTransport, ConservativeDivergenceTransportUsesDivergenceResidual)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.transport_form = level_set::LevelSetTransportForm::ConservativeDivergence;
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Divergence));
}

TEST(LevelSetTransport, SUPGUsesTransientDirectionalMetricAndControlledCapturing)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.supg.enabled = true;
    options.supg.discontinuity_capturing_enabled = true;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellDiameter));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::EffectiveTimeStep));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::JacobianInverse));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::SmoothAbsoluteValue));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Minimum));
}

TEST(LevelSetTransport, DiscontinuityCapturingJitRequestUsesCompiledPath)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.operator_tag = "equations";
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.supg.enabled = true;
    options.supg.discontinuity_capturing_enabled = true;

    FE::systems::FormInstallOptions install_options{};
    install_options.compiler_options.jit.enable = true;

    FE::systems::FESystem dc_system(mesh);
    addScalarAndVelocityFields(
        dc_system,
        phi_space,
        velocity_space,
        FE::systems::FieldSourceKind::Unknown);
    const auto dc_kernels = level_set::installLevelSetTransport(
        dc_system, phi_space, options, install_options);

    ASSERT_EQ(dc_kernels.residual.size(), 1u);
    ASSERT_TRUE(dc_kernels.residual.front());
    EXPECT_NE(
        dynamic_cast<const FE::forms::jit::JITKernelWrapper*>(
            dc_kernels.residual.front().get()),
        nullptr);
    ASSERT_TRUE(dc_kernels.mixed_plan);
    EXPECT_TRUE(dc_kernels.mixed_plan->jit_requested);
    EXPECT_TRUE(dc_kernels.mixed_plan->monolithic_cell_requested);

    // Ordinary SUPG and residual-based discontinuity capturing must both
    // remain eligible for the compiled mixed-cell path.
    FE::systems::FESystem supg_system(mesh);
    addScalarAndVelocityFields(
        supg_system,
        phi_space,
        velocity_space,
        FE::systems::FieldSourceKind::Unknown);
    options.supg.discontinuity_capturing_enabled = false;
    const auto supg_kernels = level_set::installLevelSetTransport(
        supg_system, phi_space, options, install_options);

    ASSERT_EQ(supg_kernels.residual.size(), 1u);
    ASSERT_TRUE(supg_kernels.residual.front());
    EXPECT_NE(
        dynamic_cast<const FE::forms::jit::JITKernelWrapper*>(
            supg_kernels.residual.front().get()),
        nullptr);
    ASSERT_TRUE(supg_kernels.mixed_plan);
    EXPECT_TRUE(supg_kernels.mixed_plan->jit_requested);
}

TEST(LevelSetTransport,
     DiscontinuityCapturingJitCombinedAndVectorOnlyResidualsMatch)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    const auto velocity = system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.supg.enabled = true;
    options.supg.discontinuity_capturing_enabled = true;

    FE::systems::FormInstallOptions install_options{};
    install_options.compiler_options.jit.enable = true;
    (void)level_set::installLevelSetTransport(
        system, phi_space, options, install_options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto n = system.dofHandler().getNumDofs();
    std::vector<FE::Real> solution(static_cast<std::size_t>(n), 0.0);
    std::vector<FE::Real> previous = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto point = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, 0,
                               0.12 + 0.03 * point[0] - 0.04 * point[1]);
        setFieldComponentValue(previous, system, phi, vertex, 0,
                               0.08 - 0.02 * point[0] + 0.01 * point[2]);
        setFieldComponentValue(solution, system, velocity, vertex, 0,
                               0.40 + 0.015 * point[0]);
        setFieldComponentValue(solution, system, velocity, vertex, 1,
                               -0.20 + 0.010 * point[1]);
        setFieldComponentValue(solution, system, velocity, vertex, 2,
                               0.30 - 0.005 * point[2]);
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = solution;
    state.u_prev = previous;
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    FE::assembly::DenseMatrixView jacobian(n);
    FE::assembly::DenseVectorView combined_residual(n);
    FE::assembly::DenseVectorView vector_only_residual(n);
    FE::systems::AssemblyRequest combined_request;
    combined_request.op = "level_set";
    combined_request.want_matrix = true;
    combined_request.want_vector = true;
    const auto combined = system.assemble(
        combined_request, state, &jacobian, &combined_residual);
    ASSERT_TRUE(combined.success) << combined.error_message;

    FE::systems::AssemblyRequest vector_request;
    vector_request.op = "level_set";
    vector_request.want_vector = true;
    const auto vector_only = system.assemble(
        vector_request, state, nullptr, &vector_only_residual);
    ASSERT_TRUE(vector_only.success) << vector_only.error_message;

    FE::systems::FESystem interpreter_system(mesh);
    (void)interpreter_system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    (void)interpreter_system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    auto interpreter_install_options = install_options;
    interpreter_install_options.compiler_options.jit.enable = false;
    (void)level_set::installLevelSetTransport(
        interpreter_system,
        phi_space,
        options,
        interpreter_install_options);
    ASSERT_NO_THROW(
        interpreter_system.setup({}, makeSingleTetraSetupInputs()));
    ASSERT_EQ(interpreter_system.dofHandler().getNumDofs(), n);
    FE::assembly::DenseVectorView interpreter_residual(n);
    const auto interpreted = interpreter_system.assemble(
        vector_request, state, nullptr, &interpreter_residual);
    ASSERT_TRUE(interpreted.success) << interpreted.error_message;

    for (FE::GlobalIndex row = 0; row < n; ++row) {
        EXPECT_NEAR(combined_residual[row], vector_only_residual[row], 1.0e-12)
            << "residual row=" << row;
        EXPECT_NEAR(combined_residual[row], interpreter_residual[row], 1.0e-12)
            << "interpreter residual row=" << row;
    }
}

TEST(LevelSetTransport,
     DiscontinuityCapturingCompiledGeneralizedAlphaMatchesInterpreterOnTriangle)
{
    ScopedEnvironmentVariable enable_compiled(
        "SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH", "1");
    ScopedEnvironmentVariable allow_compiled(
        "SVMP_FE_DISABLE_MONOLITHIC_COMPILED_DISPATCH", "0");
    ScopedEnvironmentVariable disable_compare(
        "SVMP_FE_COMPARE_MONOLITHIC_COMPILED", "0");

    const auto mesh = std::make_shared<SingleTriangleMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        mesh,
        /*order=*/1,
        /*components=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    const auto velocity = system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source =
        level_set::LevelSetVelocitySource::CoupledField;
    options.supg.enabled = true;
    options.supg.discontinuity_capturing_enabled = true;

    FE::systems::FormInstallOptions install_options{};
    install_options.compiler_options.jit.enable = true;
    const auto kernels = level_set::installLevelSetTransport(
        system, phi_space, options, install_options);
    ASSERT_TRUE(kernels.mixed_plan);
    EXPECT_TRUE(kernels.mixed_plan->jit_requested);
    EXPECT_TRUE(kernels.mixed_plan->monolithic_cell_requested);
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    const auto n = system.dofHandler().getNumDofs();
    std::vector<FE::Real> solution(static_cast<std::size_t>(n), 0.0);
    std::vector<FE::Real> previous(solution.size(), 0.0);
    std::vector<FE::Real> injected_rate(solution.size(), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 3; ++vertex) {
        const auto point = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(
            solution,
            system,
            phi,
            vertex,
            0,
            0.18 + 0.07 * point[0] - 0.05 * point[1]);
        setFieldComponentValue(
            previous,
            system,
            phi,
            vertex,
            0,
            0.11 - 0.03 * point[0] + 0.02 * point[1]);
        setFieldComponentValue(
            injected_rate,
            system,
            phi,
            vertex,
            0,
            -0.04 + 0.01 * point[0]);
        const FE::Real velocity_x = 0.35 + 0.02 * point[0];
        const FE::Real velocity_y = -0.16 + 0.03 * point[1];
        setFieldComponentValue(
            solution, system, velocity, vertex, 0, velocity_x);
        setFieldComponentValue(
            solution, system, velocity, vertex, 1, velocity_y);
        setFieldComponentValue(
            previous, system, velocity, vertex, 0, velocity_x);
        setFieldComponentValue(
            previous, system, velocity, vertex, 1, velocity_y);
    }

    constexpr double dt = 0.05;
    constexpr double dt_prev = 0.04;
    const std::vector<std::span<const FE::Real>> history = {
        std::span<const FE::Real>(previous),
        std::span<const FE::Real>(injected_rate),
    };
    const std::array<double, 2> dt_history = {dt_prev, dt_prev};
    FE::systems::SystemStateView state;
    state.time = 0.125;
    state.dt = dt;
    state.effective_dt = dt;
    state.dt_prev = dt_prev;
    state.u = solution;
    state.u_prev = previous;
    state.u_prev2 = injected_rate;
    state.u_history = history;
    state.dt_history = dt_history;

    const auto ga = FE::timestepping::utils::
        generalizedAlphaFirstOrderFromRhoInf(0.5);
    const FE::timestepping::GeneralizedAlphaFirstOrderIntegrator integrator({
        .alpha_m = ga.alpha_m,
        .alpha_f = ga.alpha_f,
        .gamma = ga.gamma,
        .history_rate_order = 0,
    });
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    FE::assembly::DenseMatrixView jacobian(n);
    FE::assembly::DenseVectorView compiled_residual(n);
    FE::systems::AssemblyRequest combined_request;
    combined_request.op = "level_set";
    combined_request.want_matrix = true;
    combined_request.want_vector = true;
    const auto compiled = system.assemble(
        combined_request, state, &jacobian, &compiled_residual);
    ASSERT_TRUE(compiled.success) << compiled.error_message;

    FE::Real matrix_l1 = 0.0;
    for (const auto value : jacobian.data()) {
        EXPECT_TRUE(std::isfinite(value));
        matrix_l1 += std::abs(value);
    }
    EXPECT_GT(matrix_l1, 0.0);
    for (FE::GlobalIndex row = 0; row < n; ++row) {
        EXPECT_TRUE(std::isfinite(compiled_residual[row]));
    }

    FE::systems::FESystem interpreter_system(mesh);
    (void)interpreter_system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    (void)interpreter_system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    auto interpreter_install_options = install_options;
    interpreter_install_options.compiler_options.jit.enable = false;
    (void)level_set::installLevelSetTransport(
        interpreter_system,
        phi_space,
        options,
        interpreter_install_options);
    ASSERT_NO_THROW(
        interpreter_system.setup({}, makeSingleTriangleSetupInputs()));
    ASSERT_EQ(interpreter_system.dofHandler().getNumDofs(), n);

    FE::assembly::DenseVectorView interpreter_residual(n);
    FE::systems::AssemblyRequest residual_request;
    residual_request.op = "level_set";
    residual_request.want_vector = true;
    const auto interpreted = interpreter_system.assemble(
        residual_request, state, nullptr, &interpreter_residual);
    ASSERT_TRUE(interpreted.success) << interpreted.error_message;

    for (FE::GlobalIndex row = 0; row < n; ++row) {
        ASSERT_TRUE(std::isfinite(interpreter_residual[row]));
        EXPECT_NEAR(
            compiled_residual[row],
            interpreter_residual[row],
            1.0e-11)
            << "residual row=" << row;
    }
}

TEST(LevelSetTransport,
     DiscontinuityCapturingPrescribedVelocityCompiledGeneralizedAlphaMatchesSeparateResidual)
{
    ScopedEnvironmentVariable enable_compiled(
        "SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH", "1");
    ScopedEnvironmentVariable allow_compiled(
        "SVMP_FE_DISABLE_MONOLITHIC_COMPILED_DISPATCH", "0");
    ScopedEnvironmentVariable disable_compare(
        "SVMP_FE_COMPARE_MONOLITHIC_COMPILED", "0");

    const auto mesh = std::make_shared<SingleTriangleMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        mesh,
        /*order=*/1,
        /*components=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    const auto velocity = system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source =
        level_set::LevelSetVelocitySource::PrescribedData;
    options.supg.enabled = true;
    options.supg.discontinuity_capturing_enabled = true;
    options.supg.tau_scale = 0.5;
    options.supg.transient_scale = 2.0;
    options.supg.discontinuity_capturing_scale = 0.1;
    options.supg.gradient_epsilon = 1.0e-12;
    options.supg.discontinuity_capturing_residual_epsilon = 1.0e-12;
    options.supg.discontinuity_capturing_max_courant = 0.5;

    FE::systems::FormInstallOptions install_options{};
    install_options.compiler_options.jit.enable = true;
    install_options.compiler_options.jit.cache_kernels = false;
    install_options.compiler_options.jit.optimization_level = 3;
    install_options.compiler_options.jit.specialization.enable = true;
    install_options.compiler_options.jit.specialization.specialize_n_qpts = true;
    install_options.compiler_options.jit.specialization.specialize_dofs = true;
    install_options.compiler_options.jit.specialization.text_budget_bytes =
        4u * 1024u * 1024u;
    install_options.compiler_options.jit.specialization
        .helper_text_budget_bytes = 1u;
    install_options.compiler_options.jit.basis_baking.enable = true;
    install_options.compiler_options.jit.basis_baking.force_dof_specialization =
        true;
    const auto kernels = level_set::installLevelSetTransport(
        system, phi_space, options, install_options);
    ASSERT_EQ(kernels.residual.size(), 1u);
    const auto* compiled_kernel =
        dynamic_cast<const FE::forms::jit::JITKernelWrapper*>(
            kernels.residual.front().get());
    ASSERT_NE(compiled_kernel, nullptr);
    EXPECT_TRUE(compiled_kernel->jitOptions().specialization.enable);
    EXPECT_TRUE(compiled_kernel->wantsBasisBakingHints());
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    std::vector<FE::Real> prescribed_velocity(6u, 0.0);
    for (std::size_t node = 0; node < 3u; ++node) {
        prescribed_velocity[node] = 0.70 - 0.08 * static_cast<FE::Real>(node);
        prescribed_velocity[3u + node] =
            -0.25 + 0.06 * static_cast<FE::Real>(node);
    }
    system.setPrescribedFieldCoefficients(velocity, prescribed_velocity);

    const auto n = system.dofHandler().getNumDofs();
    std::vector<FE::Real> solution(static_cast<std::size_t>(n), 0.0);
    std::vector<FE::Real> previous(solution.size(), 0.0);
    std::vector<FE::Real> injected_rate(solution.size(), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 3; ++vertex) {
        const auto index = static_cast<FE::Real>(vertex);
        setFieldComponentValue(
            solution, system, phi, vertex, 0, -0.012 + 0.021 * index);
        setFieldComponentValue(
            previous, system, phi, vertex, 0, -0.011 + 0.020 * index);
        setFieldComponentValue(
            injected_rate,
            system,
            phi,
            vertex,
            0,
            -900.0 + 850.0 * index);
    }

    constexpr double dt = 5.0e-4;
    constexpr double dt_prev = 5.0e-4;
    const std::vector<std::span<const FE::Real>> history = {
        std::span<const FE::Real>(previous),
        std::span<const FE::Real>(injected_rate),
    };
    const std::array<double, 2> dt_history = {dt_prev, dt_prev};
    FE::systems::SystemStateView state;
    state.time = dt;
    state.dt = dt;
    state.effective_dt = dt;
    state.dt_prev = dt_prev;
    state.u = solution;
    state.u_prev = previous;
    state.u_prev2 = injected_rate;
    state.u_history = history;
    state.dt_history = dt_history;

    const auto ga = FE::timestepping::utils::
        generalizedAlphaFirstOrderFromRhoInf(0.5);
    const FE::timestepping::GeneralizedAlphaFirstOrderIntegrator integrator({
        .alpha_m = ga.alpha_m,
        .alpha_f = ga.alpha_f,
        .gamma = ga.gamma,
        .history_rate_order = 0,
    });
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    FE::assembly::DenseMatrixView jacobian(n);
    FE::assembly::DenseVectorView combined_residual(n);
    FE::systems::AssemblyRequest combined_request;
    combined_request.op = "level_set";
    combined_request.want_matrix = true;
    combined_request.want_vector = true;
    const auto combined = system.assemble(
        combined_request, state, &jacobian, &combined_residual);
    ASSERT_TRUE(combined.success) << combined.error_message;
    EXPECT_TRUE(compiled_kernel->isJITReady());

    FE::assembly::DenseVectorView separate_residual(n);
    FE::systems::AssemblyRequest residual_request;
    residual_request.op = "level_set";
    residual_request.want_vector = true;
    const auto separate = system.assemble(
        residual_request, state, nullptr, &separate_residual);
    ASSERT_TRUE(separate.success) << separate.error_message;

    FE::systems::FESystem interpreter_system(mesh);
    (void)interpreter_system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    const auto interpreter_velocity =
        interpreter_system.addField(FE::systems::FieldSpec{
            .name = "advecting_velocity",
            .space = velocity_space,
            .components = velocity_space->value_dimension(),
            .source_kind = FE::systems::FieldSourceKind::PrescribedData,
        });
    auto interpreter_install_options = install_options;
    interpreter_install_options.compiler_options.jit.enable = false;
    (void)level_set::installLevelSetTransport(
        interpreter_system,
        phi_space,
        options,
        interpreter_install_options);
    ASSERT_NO_THROW(
        interpreter_system.setup({}, makeSingleTriangleSetupInputs()));
    interpreter_system.setPrescribedFieldCoefficients(
        interpreter_velocity, prescribed_velocity);
    ASSERT_EQ(interpreter_system.dofHandler().getNumDofs(), n);

    FE::assembly::DenseVectorView interpreter_residual(n);
    const auto interpreted = interpreter_system.assemble(
        residual_request, state, nullptr, &interpreter_residual);
    ASSERT_TRUE(interpreted.success) << interpreted.error_message;

    for (FE::GlobalIndex row = 0; row < n; ++row) {
        ASSERT_TRUE(std::isfinite(combined_residual[row]));
        ASSERT_TRUE(std::isfinite(separate_residual[row]));
        ASSERT_TRUE(std::isfinite(interpreter_residual[row]));
        EXPECT_NEAR(combined_residual[row], separate_residual[row], 1.0e-11)
            << "separate residual row=" << row;
        EXPECT_NEAR(combined_residual[row], interpreter_residual[row], 1.0e-11)
            << "interpreter residual row=" << row;
    }
}

TEST(LevelSetTransport,
     AlgebraicVelocityExtensionJitRequestUsesCompiledPath)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });
    const auto physical_velocity = system.addField(FE::systems::FieldSpec{
        .name = "physical_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });

    level_set::LevelSetTransportOptions options{};
    options.operator_tag = "equations";
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "extension_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;
    options.velocity.algebraic_extension_source_field_name =
        "physical_velocity";

    FE::systems::FormInstallOptions install_options{};
    install_options.compiler_options.jit.enable = true;
    const auto kernels = level_set::installLevelSetTransport(
        system, phi_space, options, install_options);

    const auto extension_velocity =
        system.findFieldByName("extension_velocity");
    ASSERT_NE(extension_velocity, FE::INVALID_FIELD_ID);
    EXPECT_NE(extension_velocity, physical_velocity);
    EXPECT_TRUE(level_set::findLevelSetVelocityExtensionConstraintKernel(
        system, options.operator_tag, extension_velocity));
    ASSERT_EQ(kernels.residual.size(), 1u);
    ASSERT_TRUE(kernels.residual.front());
    EXPECT_NE(
        dynamic_cast<const FE::forms::jit::JITKernelWrapper*>(
            kernels.residual.front().get()),
        nullptr);
    ASSERT_TRUE(kernels.mixed_plan);
    EXPECT_TRUE(kernels.mixed_plan->jit_requested);
    EXPECT_TRUE(kernels.mixed_plan->monolithic_cell_requested);
}

TEST(LevelSetTransport,
     AutoRegistrationPreflightRejectsMissingExtensionSourceWithoutMutation)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    level_set::LevelSetTransportOptions options{};
    options.operator_tag = "equations";
    options.level_set.field_name = "phi_pending";
    options.level_set.auto_register_field = true;
    options.velocity.field_name = "extension_pending";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;
    options.velocity.algebraic_extension_source_field_name =
        "missing_physical_velocity";

    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            system, phi_space, options),
        std::invalid_argument);
    EXPECT_EQ(system.findFieldByName("phi_pending"), FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("extension_pending"),
              FE::INVALID_FIELD_ID);
    EXPECT_FALSE(system.hasOperator("equations"));
    EXPECT_TRUE(system.formulationRecords().empty());
}

TEST(LevelSetTransport,
     AlgebraicExtensionSparsityUsesQuadEdgesWithoutCellDiagonals)
{
    const auto mesh =
        std::make_shared<StructuredQuadTransportMeshAccess>(1);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto source = system.addField(FE::systems::FieldSpec{
        .name = "physical_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    const auto extension = system.addField(FE::systems::FieldSpec{
        .name = "extension_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    ASSERT_NO_THROW(
        system.setup({}, makeStructuredQuadTransportSetupInputs(*mesh)));

    level_set::LevelSetVelocityExtensionConstraintKernel kernel({
        .extension_field = extension,
        .source_velocity_field = source,
        .components = 3,
        .operator_tag = "equations",
    });
    FE::sparsity::SparsityPattern pattern(
        system.dofHandler().getNumDofs());
    ASSERT_NO_THROW(kernel.addSparsityCouplings(system, pattern));
    pattern.finalize();

    const auto vertex_dof = [&](FE::FieldId field,
                                FE::GlobalIndex vertex,
                                int component) {
        const auto* entity_map =
            system.fieldDofHandler(field).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error("test field has no vertex DOF map");
        }
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (component < 0 ||
            static_cast<std::size_t>(component) >= dofs.size()) {
            throw std::runtime_error("test field component is out of range");
        }
        return system.fieldDofOffset(field) +
               dofs[static_cast<std::size_t>(component)];
    };

    const auto row = vertex_dof(extension, 0, 0);
    for (int component = 0; component < 3; ++component) {
        EXPECT_TRUE(pattern.hasEntry(
            row, vertex_dof(extension, 0, component)));
        EXPECT_TRUE(pattern.hasEntry(
            row, vertex_dof(extension, 1, component)));
        EXPECT_TRUE(pattern.hasEntry(
            row, vertex_dof(extension, 2, component)));
        EXPECT_FALSE(pattern.hasEntry(
            row, vertex_dof(extension, 3, component)));
    }
    EXPECT_TRUE(pattern.hasEntry(row, vertex_dof(source, 0, 0)));
    EXPECT_FALSE(pattern.hasEntry(row, vertex_dof(source, 1, 0)));
}

TEST(LevelSetTransport,
     DuplicateExtensionPreflightDoesNotInstallAnotherFormulation)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });
    system.addField(FE::systems::FieldSpec{
        .name = "physical_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });

    level_set::LevelSetTransportOptions options{};
    options.operator_tag = "equations";
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "extension_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;
    options.velocity.algebraic_extension_source_field_name =
        "physical_velocity";

    ASSERT_NO_THROW((void)level_set::installLevelSetTransport(
        system, phi_space, options));
    const auto record_count = system.formulationRecords().size();
    ASSERT_GT(record_count, 0u);

    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            system, phi_space, options),
        std::invalid_argument);
    EXPECT_EQ(system.formulationRecords().size(), record_count);
}

TEST(LevelSetTransport, SUPGRejectsInvalidTransientAndCapturingControls)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    auto make_options = [] {
        level_set::LevelSetTransportOptions options{};
        options.level_set.field_name = "phi";
        options.level_set.auto_register_field = false;
        options.velocity.field_name = "advecting_velocity";
        options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
        options.supg.enabled = true;
        options.supg.discontinuity_capturing_enabled = true;
        return options;
    };

    {
        FE::systems::FESystem system(mesh);
        addScalarAndVelocityFields(system, phi_space, velocity_space);
        auto options = make_options();
        options.supg.transient_scale = 0.0;
        EXPECT_THROW((void)level_set::installLevelSetTransport(system, phi_space, options),
                     std::invalid_argument);
    }
    {
        FE::systems::FESystem system(mesh);
        addScalarAndVelocityFields(system, phi_space, velocity_space);
        auto options = make_options();
        options.supg.discontinuity_capturing_scale = -1.0;
        EXPECT_THROW((void)level_set::installLevelSetTransport(system, phi_space, options),
                     std::invalid_argument);
    }
    {
        FE::systems::FESystem system(mesh);
        addScalarAndVelocityFields(system, phi_space, velocity_space);
        auto options = make_options();
        options.supg.gradient_epsilon = 0.0;
        EXPECT_THROW((void)level_set::installLevelSetTransport(system, phi_space, options),
                     std::invalid_argument);
    }
    {
        FE::systems::FESystem system(mesh);
        addScalarAndVelocityFields(system, phi_space, velocity_space);
        auto options = make_options();
        options.supg.discontinuity_capturing_residual_epsilon = 0.0;
        EXPECT_THROW((void)level_set::installLevelSetTransport(
                         system, phi_space, options),
                     std::invalid_argument);
    }
    {
        FE::systems::FESystem system(mesh);
        addScalarAndVelocityFields(system, phi_space, velocity_space);
        auto options = make_options();
        options.supg.discontinuity_capturing_max_courant = 0.0;
        EXPECT_THROW((void)level_set::installLevelSetTransport(system, phi_space, options),
                     std::invalid_argument);
    }
    {
        FE::systems::FESystem system(mesh);
        addScalarAndVelocityFields(system, phi_space, velocity_space);
        auto options = make_options();
        options.supg.enabled = false;
        EXPECT_THROW((void)level_set::installLevelSetTransport(system, phi_space, options),
                     std::invalid_argument);
    }
}

TEST(LevelSetTransport, Quad9FlatHorizontalNullModeHasZeroSpatialResidual)
{
    const auto run_case =
        [](const std::shared_ptr<FE::assembly::IMeshAccess>& mesh,
           const FE::systems::SetupInputs& setup) {
          for (const bool stabilized : {false, true}) {
            SCOPED_TRACE(stabilized ? "SUPG and discontinuity capturing"
                                    : "unstabilized Galerkin");
            auto phi_space = scalarSpace(mesh, /*order=*/2);

            FE::systems::FESystem system(mesh);
            const auto phi = system.addField(FE::systems::FieldSpec{
                .name = "phi",
                .space = phi_space,
                .components = 1,
            });

            level_set::LevelSetTransportOptions options{};
            options.level_set.field_name = "phi";
            options.level_set.auto_register_field = false;
            options.velocity.source = level_set::LevelSetVelocitySource::ConstantVector;
            options.velocity.constant_value = {0.1, 0.0, 0.0};
            options.supg.enabled = stabilized;

            (void)level_set::installLevelSetTransport(system, phi_space, options);
            ASSERT_NO_THROW(system.setup({}, setup));

            std::vector<FE::Real> solution(
                static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
            const auto& phi_dofs = system.fieldDofHandler(phi);
            const auto offset = system.fieldDofOffset(phi);
            for (FE::GlobalIndex cell = 0; cell < mesh->numCells(); ++cell) {
                std::vector<FE::GlobalIndex> nodes;
                mesh->getCellNodes(cell, nodes);
                const auto dofs = phi_dofs.getCellDofs(cell);
                ASSERT_EQ(dofs.size(), nodes.size());
                for (std::size_t local = 0; local < nodes.size(); ++local) {
                    const auto x = mesh->getNodeCoordinates(nodes[local]);
                    const auto index = static_cast<std::size_t>(dofs[local] + offset);
                    ASSERT_LT(index, solution.size());
                    solution[index] = x[1] - FE::Real{0.375};
                }
            }
            const auto previous_solution = solution;

            FE::systems::SystemStateView state;
            state.dt = 0.1;
            state.u = std::span<const FE::Real>(solution);
            state.u_prev = std::span<const FE::Real>(previous_solution);
            const FE::systems::BackwardDifferenceIntegrator integrator;
            const auto time_context =
                integrator.buildContext(/*max_time_derivative_order=*/1, state);
            state.time_integration = &time_context;

            const auto residual = assembleLevelSetResidual(system, state);
            EXPECT_LT(l2Norm(std::span<const FE::Real>(residual)), 1.0e-12);
          }
        };

    run_case(
        std::make_shared<SingleQuad9MeshAccess>(),
        makeSingleQuad9SetupInputs());
    run_case(
        std::make_shared<Quad9Patch2x2MeshAccess>(),
        makeQuad9Patch2x2SetupInputs());
}

TEST(LevelSetTransport, Quad9TravelingLinearModeHasZeroStabilizedResidual)
{
    const auto run_case =
        [](const std::shared_ptr<FE::assembly::IMeshAccess>& mesh,
           const FE::systems::SetupInputs& setup) {
          constexpr FE::Real speed = FE::Real{0.35};
          constexpr FE::Real dt = FE::Real{0.08};
          constexpr FE::Real interface_offset = FE::Real{0.37};

          auto phi_space = scalarSpace(mesh, /*order=*/2);
          FE::systems::FESystem system(mesh);
          const auto phi = system.addField(FE::systems::FieldSpec{
              .name = "phi",
              .space = phi_space,
              .components = 1,
          });

          level_set::LevelSetTransportOptions options{};
          options.level_set.field_name = "phi";
          options.level_set.auto_register_field = false;
          options.velocity.source = level_set::LevelSetVelocitySource::ConstantVector;
          options.velocity.constant_value = {speed, 0.0, 0.0};
          options.supg.enabled = true;
          options.supg.discontinuity_capturing_enabled = true;

          (void)level_set::installLevelSetTransport(system, phi_space, options);
          ASSERT_NO_THROW(system.setup({}, setup));

          std::vector<FE::Real> solution(
              static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
          auto previous_solution = solution;
          const auto& phi_dofs = system.fieldDofHandler(phi);
          const auto offset = system.fieldDofOffset(phi);
          for (FE::GlobalIndex cell = 0; cell < mesh->numCells(); ++cell) {
              std::vector<FE::GlobalIndex> nodes;
              mesh->getCellNodes(cell, nodes);
              const auto dofs = phi_dofs.getCellDofs(cell);
              ASSERT_EQ(dofs.size(), nodes.size());
              for (std::size_t local = 0; local < nodes.size(); ++local) {
                  const auto x = mesh->getNodeCoordinates(nodes[local]);
                  const auto index = static_cast<std::size_t>(dofs[local] + offset);
                  ASSERT_LT(index, solution.size());
                  previous_solution[index] = x[0] - interface_offset;
                  solution[index] =
                      x[0] - speed * dt - interface_offset;
              }
          }

          FE::systems::SystemStateView state;
          state.dt = dt;
          state.u = std::span<const FE::Real>(solution);
          state.u_prev = std::span<const FE::Real>(previous_solution);
          const FE::systems::BackwardDifferenceIntegrator integrator;
          const auto time_context =
              integrator.buildContext(/*max_time_derivative_order=*/1, state);
          state.time_integration = &time_context;

          const auto residual = assembleLevelSetResidual(system, state);
          EXPECT_LT(l2Norm(std::span<const FE::Real>(residual)), 1.0e-12);
        };

    run_case(
        std::make_shared<SingleQuad9MeshAccess>(),
        makeSingleQuad9SetupInputs());
    run_case(
        std::make_shared<Quad9Patch2x2MeshAccess>(),
        makeQuad9Patch2x2SetupInputs());
}

TEST(LevelSetTransport,
     Quad9CurvedInterfaceTransportResidualConvergesUnderTimeRefinement)
{
    constexpr FE::Real speed_x = FE::Real{0.35};
    constexpr FE::Real speed_y = FE::Real{-0.2};
    constexpr FE::Real center_x = FE::Real{0.65};
    constexpr FE::Real center_y = FE::Real{0.85};

    const auto mesh = std::make_shared<Quad9Patch2x2MeshAccess>();
    auto phi_space = scalarSpace(mesh, /*order=*/2);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.source = level_set::LevelSetVelocitySource::ConstantVector;
    options.velocity.constant_value = {speed_x, speed_y, 0.0};
    options.supg.enabled = true;
    options.supg.discontinuity_capturing_enabled = true;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeQuad9Patch2x2SetupInputs()));

    std::vector<FE::Real> current(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& phi_dofs = system.fieldDofHandler(phi);
    const auto offset = system.fieldDofOffset(phi);
    const auto curved_phi = [](FE::Real x, FE::Real y) {
        const FE::Real dx = x - center_x;
        const FE::Real dy = y - center_y;
        return dx * dx + FE::Real{0.6} * dy * dy - FE::Real{0.22};
    };
    for (FE::GlobalIndex cell = 0; cell < mesh->numCells(); ++cell) {
        std::vector<FE::GlobalIndex> nodes;
        mesh->getCellNodes(cell, nodes);
        const auto dofs = phi_dofs.getCellDofs(cell);
        ASSERT_EQ(dofs.size(), nodes.size());
        for (std::size_t local = 0; local < nodes.size(); ++local) {
            const auto x = mesh->getNodeCoordinates(nodes[local]);
            current[static_cast<std::size_t>(dofs[local] + offset)] =
                curved_phi(x[0], x[1]);
        }
    }

    const auto residual_norm = [&](FE::Real dt) {
        std::vector<FE::Real> previous = current;
        for (FE::GlobalIndex cell = 0; cell < mesh->numCells(); ++cell) {
            std::vector<FE::GlobalIndex> nodes;
            mesh->getCellNodes(cell, nodes);
            const auto dofs = phi_dofs.getCellDofs(cell);
            for (std::size_t local = 0; local < nodes.size(); ++local) {
                const auto x = mesh->getNodeCoordinates(nodes[local]);
                // Exact previous-time value for the translating curved field
                // phi(x,t)=phi_0(x-u*t), with the current time set to zero.
                previous[static_cast<std::size_t>(dofs[local] + offset)] =
                    curved_phi(x[0] + speed_x * dt,
                               x[1] + speed_y * dt);
            }
        }

        FE::systems::SystemStateView state;
        state.dt = dt;
        state.u = std::span<const FE::Real>(current);
        state.u_prev = std::span<const FE::Real>(previous);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto time_context =
            integrator.buildContext(/*max_time_derivative_order=*/1, state);
        state.time_integration = &time_context;
        return l2Norm(assembleLevelSetResidual(system, state));
    };

    const FE::Real coarse = residual_norm(FE::Real{0.08});
    const FE::Real fine = residual_norm(FE::Real{0.04});
    EXPECT_GT(coarse, 1.0e-7);
    EXPECT_LT(fine, FE::Real{0.7} * coarse);
    EXPECT_GT(fine, FE::Real{0.3} * coarse);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(LevelSetTransport,
     NativeQuad9ProjectedFlatHorizontalNullModeHasZeroSpatialResidual)
{
    auto mesh = buildNativeQuad9Mesh();
    auto phi_space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4,
        /*order=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.source = level_set::LevelSetVelocitySource::ConstantVector;
    options.velocity.constant_value = {0.1, 0.0, 0.0};
    options.supg.enabled = false;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup());

    std::vector<FE::Real> mesh_values(mesh->n_vertices(), FE::Real{0});
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
        const auto point =
            mesh->get_vertex_coords(static_cast<svmp::index_t>(vertex));
        mesh_values[vertex] = point[1] - FE::Real{0.375};
    }

    std::vector<FE::Real> phi_coefficients(
        static_cast<std::size_t>(system.fieldDofHandler(phi).getNumDofs()),
        FE::Real{0});
    std::vector<std::uint8_t> assigned(phi_coefficients.size(), 0u);
    const auto projection = system.projectMeshVertexValuesToFieldCoefficients(
        phi,
        std::span<const FE::Real>(mesh_values.data(), mesh_values.size()),
        /*mesh_components=*/1,
        std::span<FE::Real>(phi_coefficients.data(), phi_coefficients.size()),
        std::span<std::uint8_t>(assigned.data(), assigned.size()),
        "LevelSetTransport native Quad9 projection invariant");
    ASSERT_EQ(projection.unassigned_dofs, 0u);
    ASSERT_EQ(projection.values_written, phi_coefficients.size());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0});
    const auto offset = system.fieldDofOffset(phi);
    ASSERT_GE(offset, 0);
    ASSERT_LE(static_cast<std::size_t>(offset) + phi_coefficients.size(),
              solution.size());
    std::copy(phi_coefficients.begin(),
              phi_coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));
    const auto previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    const auto residual = assembleLevelSetResidual(system, state);
    EXPECT_LT(l2Norm(std::span<const FE::Real>(residual)), 1.0e-12);
}
#endif

TEST(LevelSetTransport,
     BoundPreservingLimiterEnforcesPreviousOneRingBoundsAndDrySign)
{
    constexpr int n = 4;
    const auto mesh =
        std::make_shared<StructuredQuadTransportMeshAccess>(n);
    auto phi_space = scalarSpace(mesh);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});

    level_set::LevelSetTransportOptions transport{};
    transport.level_set.field_name = "phi";
    transport.level_set.auto_register_field = false;
    transport.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    transport.velocity.constant_value = {0.25, 0.0, 0.0};
    transport.bound_preserving.enabled = true;
    (void)level_set::installLevelSetTransport(system, phi_space, transport);
    ASSERT_NO_THROW(system.setup(
        {}, makeStructuredQuadTransportSetupInputs(*mesh)));

    std::vector<FE::Real> previous(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    fillScalarFieldAtStructuredVertices(
        previous, system, phi, *mesh,
        [](FE::Real x, FE::Real /*y*/) { return x - FE::Real{0.55}; });
    auto candidate = previous;
    const auto corrupted_vertex = mesh->nodeId(n, n);
    setScalarVertexValue(candidate, system, phi, corrupted_vertex, -0.25);

    std::vector<FE::Real> limited;
    const auto result = level_set::applyLevelSetBoundPreservingLimiter(
        system, phi, transport.boundaries, transport.bound_preserving,
        previous, candidate, /*observed_courant=*/0.25, limited);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.bounds_satisfied);
    EXPECT_TRUE(result.sign_preservation_satisfied);
    EXPECT_EQ(result.limited_dofs, 1u);
    EXPECT_EQ(result.positive_patch_sign_flips_prevented, 1u);
    EXPECT_GE(result.maximum_unrelaxed_bound_violation,
              result.maximum_bound_violation);
    EXPECT_GT(result.maximum_bound_violation, 0.0);
    EXPECT_GT(scalarVertexValue(limited, system, phi, corrupted_vertex), 0.0);
    EXPECT_NEAR(scalarVertexValue(limited, system, phi, corrupted_vertex),
                0.2, 2.0e-12);
}

TEST(LevelSetTransport,
     BoundPreservingLimiterRejectsSignFlipInsideRelaxedCoefficientBounds)
{
    constexpr int n = 4;
    const auto mesh =
        std::make_shared<StructuredQuadTransportMeshAccess>(n);
    auto phi_space = scalarSpace(mesh);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});

    level_set::LevelSetTransportOptions transport{};
    transport.level_set.field_name = "phi";
    transport.level_set.auto_register_field = false;
    transport.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    transport.velocity.constant_value = {0.25, 0.0, 0.0};
    transport.bound_preserving.enabled = true;
    // Deliberately make the coefficient bound permissive enough that the raw
    // negative value is not clamped.  Same-sign patch preservation must remain
    // an independent fail-closed invariant.
    transport.bound_preserving.bound_tolerance = 1.0;
    transport.bound_preserving.sign_tolerance = 1.0e-12;
    (void)level_set::installLevelSetTransport(system, phi_space, transport);
    ASSERT_NO_THROW(system.setup(
        {}, makeStructuredQuadTransportSetupInputs(*mesh)));

    std::vector<FE::Real> previous(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    fillScalarFieldAtStructuredVertices(
        previous, system, phi, *mesh,
        [](FE::Real x, FE::Real /*y*/) { return x - FE::Real{0.55}; });
    auto candidate = previous;
    const auto corrupted_vertex = mesh->nodeId(n, n);
    setScalarVertexValue(candidate, system, phi, corrupted_vertex, -0.25);

    std::vector<FE::Real> limited;
    const auto result = level_set::applyLevelSetBoundPreservingLimiter(
        system, phi, transport.boundaries, transport.bound_preserving,
        previous, candidate, /*observed_courant=*/0.25, limited);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.bounds_satisfied);
    EXPECT_FALSE(result.sign_preservation_satisfied);
    EXPECT_EQ(result.positive_patch_sign_flips_prevented, 1u);
    EXPECT_EQ(result.limited_dofs, 0u);
    EXPECT_GT(result.maximum_unrelaxed_bound_violation, 0.0);
    EXPECT_EQ(result.maximum_bound_violation, 0.0);
    EXPECT_LT(scalarVertexValue(limited, system, phi, corrupted_vertex), 0.0);
    EXPECT_NE(result.diagnostic.find("post-projection invariant"),
              std::string::npos);

    std::vector<FE::Real> courant_limited;
    const auto courant_result = level_set::applyLevelSetBoundPreservingLimiter(
        system, phi, transport.boundaries, transport.bound_preserving,
        previous, previous,
        transport.bound_preserving.maximum_courant + 1.0e-6,
        courant_limited);
    EXPECT_FALSE(courant_result.success);
    EXPECT_NE(courant_result.diagnostic.find("Courant"), std::string::npos);
}

TEST(LevelSetTransport,
     TransportSafetyAcceptsTangentialWallsAndRejectsNormalWallFlux)
{
    constexpr int n = 8;
    const auto mesh =
        std::make_shared<StructuredQuadTransportMeshAccess>(n);
    auto phi_space = scalarSpace(mesh);
    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});

    level_set::LevelSetTransportOptions transport{};
    transport.level_set.field_name = "phi";
    transport.level_set.auto_register_field = false;
    transport.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    transport.velocity.constant_value = {0.5, 0.0, 0.0};
    transport.boundaries.inflow.push_back(
        level_set::LevelSetInflowBoundary{
            .boundary_marker =
                StructuredQuadTransportMeshAccess::kInflowMarker,
            .value = 1.0});
    transport.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{
            .boundary_marker =
                StructuredQuadTransportMeshAccess::kOutflowMarker});
    transport.bound_preserving.enabled = true;
    (void)level_set::installLevelSetTransport(system, phi_space, transport);
    ASSERT_NO_THROW(system.setup(
        {}, makeStructuredQuadTransportSetupInputs(*mesh)));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = solution;

    const auto tangential = level_set::evaluateLevelSetTransportSafety(
        system, transport.velocity, transport.boundaries,
        transport.bound_preserving, state, /*dt=*/0.125);
    ASSERT_TRUE(tangential.success) << tangential.diagnostic;
    EXPECT_TRUE(tangential.courant_satisfied);
    EXPECT_TRUE(tangential.impermeable_boundaries_satisfied);
    EXPECT_EQ(tangential.impermeable_boundary_faces_checked,
              static_cast<std::size_t>(2 * n));
    EXPECT_NEAR(tangential.maximum_courant, 0.5, 1.0e-12);
    EXPECT_NEAR(tangential.maximum_boundary_normal_velocity, 0.0, 1.0e-14);

    auto normal_velocity = transport.velocity;
    normal_velocity.constant_value = {0.0, 0.5, 0.0};
    const auto incompatible = level_set::evaluateLevelSetTransportSafety(
        system, normal_velocity, transport.boundaries,
        transport.bound_preserving, state, /*dt=*/0.125);
    EXPECT_FALSE(incompatible.success);
    EXPECT_TRUE(incompatible.courant_satisfied);
    EXPECT_FALSE(incompatible.impermeable_boundaries_satisfied);
    EXPECT_NEAR(incompatible.maximum_boundary_normal_velocity, 0.5, 1.0e-12);
}

TEST(LevelSetTransport,
     TransportSafetyRejectsNodalWallFluxThatCancelsAtFaceCentroid)
{
    const auto mesh =
        std::make_shared<StructuredQuadTransportMeshAccess>(/*cells_per_axis=*/1);
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    const auto velocity = system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
        .source_kind = FE::systems::FieldSourceKind::PrescribedData});

    level_set::LevelSetTransportOptions transport{};
    transport.level_set.field_name = "phi";
    transport.level_set.auto_register_field = false;
    transport.velocity.field_name = "advecting_velocity";
    transport.velocity.source =
        level_set::LevelSetVelocitySource::PrescribedData;
    transport.boundaries.inflow.push_back(
        level_set::LevelSetInflowBoundary{
            .boundary_marker =
                StructuredQuadTransportMeshAccess::kInflowMarker,
            .value = 1.0});
    transport.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{
            .boundary_marker =
                StructuredQuadTransportMeshAccess::kOutflowMarker});
    transport.bound_preserving.enabled = true;
    (void)level_set::installLevelSetTransport(system, phi_space, transport);
    ASSERT_NO_THROW(system.setup(
        {}, makeStructuredQuadTransportSetupInputs(*mesh)));

    const auto& handler = system.fieldDofHandler(velocity);
    const auto* entity_map = handler.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(handler.getNumDofs()), 0.0);
    for (int j = 0; j <= 1; ++j) {
        for (int i = 0; i <= 1; ++i) {
            const auto dofs = entity_map->getVertexDofs(mesh->nodeId(i, j));
            ASSERT_GE(dofs.size(), 2u);
            ASSERT_GE(dofs[1], 0);
            ASSERT_LT(static_cast<std::size_t>(dofs[1]), coefficients.size());
            coefficients[static_cast<std::size_t>(dofs[1])] =
                i == 0 ? -1.0 : 1.0;
        }
    }
    system.setPrescribedFieldCoefficients(velocity, coefficients);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = solution;
    const auto safety = level_set::evaluateLevelSetTransportSafety(
        system, transport.velocity, transport.boundaries,
        transport.bound_preserving, state, /*dt=*/0.125);

    EXPECT_FALSE(safety.success);
    EXPECT_TRUE(safety.courant_satisfied);
    EXPECT_FALSE(safety.impermeable_boundaries_satisfied);
    EXPECT_EQ(safety.impermeable_boundary_faces_checked, 2u);
    EXPECT_NEAR(safety.maximum_boundary_normal_velocity, 1.0, 1.0e-12);
    EXPECT_EQ(safety.worst_boundary_marker,
              StructuredQuadTransportMeshAccess::kWallMarker);
}

TEST(LevelSetTransport,
     AssembledSmoothTravelingWaveSolveConvergesUnderSpaceTimeRefinement)
{
    const auto solve_error = [](int n) {
        const FE::Real speed = 0.35;
        const FE::Real dt = 0.2 / static_cast<FE::Real>(n);
        const auto mesh =
            std::make_shared<StructuredQuadTransportMeshAccess>(n);
        auto phi_space = scalarSpace(mesh);
        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi", .space = phi_space, .components = 1});

        level_set::LevelSetTransportOptions options{};
        options.level_set.field_name = "phi";
        options.level_set.auto_register_field = false;
        options.velocity.source =
            level_set::LevelSetVelocitySource::ConstantVector;
        options.velocity.constant_value = {speed, 0.0, 0.0};
        options.supg.enabled = true;
        options.supg.discontinuity_capturing_enabled = false;
        options.boundaries.inflow.push_back(
            level_set::LevelSetInflowBoundary{
                .boundary_marker =
                    StructuredQuadTransportMeshAccess::kInflowMarker,
                .value = std::sin(-2.0 * std::acos(-1.0) * speed * dt),
                .penalty_scale = 4.0});
        options.boundaries.outflow.push_back(
            level_set::LevelSetOutflowBoundary{
                .boundary_marker =
                    StructuredQuadTransportMeshAccess::kOutflowMarker});
        (void)level_set::installLevelSetTransport(system, phi_space, options);
        system.setup({}, makeStructuredQuadTransportSetupInputs(*mesh));

        std::vector<FE::Real> previous(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        fillScalarFieldAtStructuredVertices(
            previous, system, phi, *mesh,
            [](FE::Real x, FE::Real /*y*/) {
                return std::sin(2.0 * std::acos(-1.0) * x);
            });
        const auto candidate =
            solveLinearLevelSetStep(system, previous, dt);

        FE::Real squared_error = 0.0;
        std::size_t samples = 0u;
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                const auto node = mesh->nodeId(i, j);
                const auto point = mesh->getNodeCoordinates(node);
                const FE::Real exact = std::sin(
                    2.0 * std::acos(-1.0) * (point[0] - speed * dt));
                const FE::Real error =
                    scalarVertexValue(candidate, system, phi, node) - exact;
                squared_error += error * error;
                ++samples;
            }
        }
        return std::sqrt(squared_error / static_cast<FE::Real>(samples));
    };

    const FE::Real coarse = solve_error(4);
    const FE::Real medium = solve_error(8);
    const FE::Real fine = solve_error(16);
    EXPECT_GT(coarse, 0.0);
    EXPECT_LT(medium, 0.8 * coarse);
    EXPECT_LT(fine, 0.8 * medium);
}

TEST(LevelSetTransport,
     AssembledSteepFrontLimiterPreventsFalseDryWallSignChange)
{
    constexpr int n = 16;
    constexpr FE::Real speed = 0.5;
    const FE::Real h = 1.0 / static_cast<FE::Real>(n);
    const FE::Real dt = 0.8 * h / speed;
    const auto mesh =
        std::make_shared<StructuredQuadTransportMeshAccess>(n);
    auto phi_space = scalarSpace(mesh);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    options.velocity.constant_value = {speed, 0.0, 0.0};
    options.supg.enabled = false;
    options.boundaries.inflow.push_back(
        level_set::LevelSetInflowBoundary{
            .boundary_marker =
                StructuredQuadTransportMeshAccess::kInflowMarker,
            .value = -1.0});
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{
            .boundary_marker =
                StructuredQuadTransportMeshAccess::kOutflowMarker});
    options.bound_preserving.enabled = true;
    options.bound_preserving.maximum_courant = 1.0;
    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup(
        {}, makeStructuredQuadTransportSetupInputs(*mesh)));

    std::vector<FE::Real> previous(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    fillScalarFieldAtStructuredVertices(
        previous, system, phi, *mesh,
        [](FE::Real x, FE::Real /*y*/) {
            // A signed interface at x=0.30 followed by a dry-side gate cliff.
            // The 0.01-to-1 jump is entirely dry; a monotone transport update
            // must not turn its small positive upstream plateau negative.
            if (x < FE::Real{0.30}) {
                return x - FE::Real{0.30};
            }
            return x < FE::Real{0.625} ? FE::Real{0.01} : FE::Real{1.0};
        });
    const auto candidate = solveLinearLevelSetStep(system, previous, dt);

    FE::systems::SystemStateView state;
    state.u = candidate;
    const auto safety = level_set::evaluateLevelSetTransportSafety(
        system, options.velocity, options.boundaries,
        options.bound_preserving, state, dt);
    ASSERT_TRUE(safety.success) << safety.diagnostic;
    EXPECT_NEAR(safety.maximum_courant, 0.8, 1.0e-12);
    EXPECT_NEAR(safety.maximum_boundary_normal_velocity, 0.0, 1.0e-14);

    std::vector<FE::Real> limited;
    const auto limited_result = level_set::applyLevelSetBoundPreservingLimiter(
        system, phi, options.boundaries, options.bound_preserving,
        previous, candidate, safety.maximum_courant, limited);
    ASSERT_TRUE(limited_result.success) << limited_result.diagnostic;
    EXPECT_TRUE(limited_result.applied);
    EXPECT_GT(limited_result.positive_patch_sign_flips_prevented, 0u);

    std::size_t false_dry_wall_flips = 0u;
    std::size_t remaining_false_dry_wall_flips = 0u;
    for (const int wall_j : {0, n}) {
        for (int i = 1; i < n; ++i) {
            const auto vertex = mesh->nodeId(i, wall_j);
            const FE::Real left_old = scalarVertexValue(
                previous, system, phi, mesh->nodeId(i - 1, wall_j));
            const FE::Real center_old = scalarVertexValue(
                previous, system, phi, vertex);
            const FE::Real right_old = scalarVertexValue(
                previous, system, phi, mesh->nodeId(i + 1, wall_j));
            const FE::Real patch_minimum =
                std::min({left_old, center_old, right_old});
            if (patch_minimum <= options.bound_preserving.sign_tolerance) {
                continue;
            }
            if (scalarVertexValue(candidate, system, phi, vertex) <
                -options.bound_preserving.sign_tolerance) {
                ++false_dry_wall_flips;
            }
            if (scalarVertexValue(limited, system, phi, vertex) <
                -options.bound_preserving.sign_tolerance) {
                ++remaining_false_dry_wall_flips;
            }
        }
    }
    EXPECT_GT(false_dry_wall_flips, 0u);
    EXPECT_EQ(remaining_false_dry_wall_flips, 0u);
}

TEST(LevelSetTransport, InterfaceKinematicAddsInterfaceResidual)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.interface_kinematic.enabled = true;
    options.interface_kinematic.interface_marker = 77;
    options.interface_kinematic.weight_scale = 2.0;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellDiameter));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
}

TEST(LevelSetTransport, ValidatesSUPGOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.supg.enabled = true;

    options.supg.tau_scale = 0.0;
    FE::systems::FESystem tau_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tau_system, phi_space, options),
        std::invalid_argument);

    options.supg.tau_scale = 0.5;
    options.supg.velocity_epsilon = 0.0;
    FE::systems::FESystem epsilon_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(epsilon_system, phi_space, options),
        std::invalid_argument);
}

TEST(LevelSetTransport, ValidatesInterfaceKinematicOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.interface_kinematic.enabled = true;

    FE::systems::FESystem marker_system(mesh);
    addScalarAndVelocityFields(marker_system, phi_space, velocity_space);
    options.interface_kinematic.interface_marker = -1;
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(marker_system, phi_space, options),
        std::invalid_argument);

    FE::systems::FESystem weight_system(mesh);
    addScalarAndVelocityFields(weight_system, phi_space, velocity_space);
    options.interface_kinematic.interface_marker = 77;
    options.interface_kinematic.weight_scale = 0.0;
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(weight_system, phi_space, options),
        std::invalid_argument);

    FE::systems::FESystem valid_system(mesh);
    addScalarAndVelocityFields(valid_system, phi_space, velocity_space);
    options.interface_kinematic.weight_scale = 1.0;
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, ValidatesReinitializationOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.reinitialization.enabled = true;

    options.reinitialization.cadence_steps = 0;
    FE::systems::FESystem cadence_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(cadence_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.cadence_steps = 1;
    options.reinitialization.max_iterations = 0;
    FE::systems::FESystem iterations_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(iterations_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.max_iterations = 10;
    options.reinitialization.pseudo_time_step_scale = 0.0;
    FE::systems::FESystem pseudo_time_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(pseudo_time_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.pseudo_time_step_scale = 0.3;
    options.reinitialization.preserve_band_width = 1.0e-3;
    FE::systems::FESystem preserve_band_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            preserve_band_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.preserve_band_width = 0.0;
    options.reinitialization.max_zero_set_displacement = -1.0;
    FE::systems::FESystem displacement_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            displacement_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.max_zero_set_displacement = 1.0e-10;
    options.reinitialization.interface_band_width = 0.0;
    FE::systems::FESystem band_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(band_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.interface_band_width = 3.0;
    options.reinitialization.signed_distance_tolerance = 0.0;
    FE::systems::FESystem tolerance_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tolerance_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.signed_distance_tolerance = 1.0e-6;
    options.reinitialization.method =
        level_set::LevelSetReinitializationMethod::FastMarching;
    FE::systems::FESystem unsupported_method_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            unsupported_method_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.method =
        level_set::LevelSetReinitializationMethod::Projection;
    FE::systems::FESystem valid_system(mesh);
    valid_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, ValidatesVolumeCorrectionOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.volume_correction.enabled = true;

    options.volume_correction.cadence_steps = 0;
    FE::systems::FESystem cadence_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(cadence_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.cadence_steps = 1;
    options.volume_correction.volume_tolerance = 0.0;
    FE::systems::FESystem tolerance_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tolerance_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.volume_tolerance = 1.0e-10;
    options.volume_correction.max_iterations = 0;
    FE::systems::FESystem iterations_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(iterations_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.max_iterations = 50;
    options.volume_correction.minimum_relative_volume_error = -1.0;
    FE::systems::FESystem relative_error_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            relative_error_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.minimum_relative_volume_error = 1.0e-6;
    options.volume_correction.maximum_interface_displacement_fraction = 0.0;
    FE::systems::FESystem displacement_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            displacement_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.maximum_interface_displacement_fraction = 0.1;
    options.volume_correction
        .maximum_cumulative_interface_displacement_fraction = 0.0;
    FE::systems::FESystem cumulative_displacement_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            cumulative_displacement_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction
        .maximum_cumulative_interface_displacement_fraction = 1.0;
    options.volume_correction.use_initial_negative_volume_as_target = false;
    options.volume_correction.target_negative_volume = -1.0;
    FE::systems::FESystem target_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(target_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.target_negative_volume = 0.125;
    FE::systems::FESystem valid_system(mesh);
    valid_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, InflowBoundaryAddsUpwindPenalty)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 4,
        .value = FE::Real{1.25},
        .penalty_scale = 2.0,
    });

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Normal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::AbsoluteValue));
    const auto policies = system.exteriorBoundaryMeasurePolicies();
    ASSERT_EQ(policies.size(), 1u);
    EXPECT_EQ(
        policies.front().intent,
        FE::systems::ExteriorBoundaryMeasureIntent::FullPhysical);
    EXPECT_EQ(policies.front().physical_boundary_marker, 4);
}

TEST(LevelSetTransport, OutflowBoundaryIsNatural)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 5});

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
}

TEST(LevelSetTransport, ValidatesBoundaryOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{});
    FE::systems::FESystem missing_marker_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            missing_marker_system,
            phi_space,
            options),
        std::invalid_argument);

    options.boundaries.inflow.clear();
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 4,
        .penalty_scale = 0.0,
    });
    FE::systems::FESystem penalty_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(penalty_system, phi_space, options),
        std::invalid_argument);

    options.boundaries.inflow.clear();
    options.boundaries.inflow.push_back(
        level_set::LevelSetInflowBoundary{.boundary_marker = 4});
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 4});
    FE::systems::FESystem duplicate_marker_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            duplicate_marker_system,
            phi_space,
            options),
        std::invalid_argument);
}

TEST(LevelSetTransport, PrescribedVelocityJacobianMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    system.setPrescribedFieldCoefficients(
        velocity,
        constantVectorTetraCoefficients(0.70, -0.15, 0.25));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(
            solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.20) + FE::Real(0.035) * x);
        setFieldComponentValue(
            previous_solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.18) + FE::Real(0.025) * x);
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        1.0e-6,
        2.0e-5,
        1.0e-8);
}

TEST(LevelSetTransport, CoupledVelocityJacobianMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(
        system,
        phi_space,
        velocity_space,
        FE::systems::FieldSourceKind::Unknown);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(
            solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.15) + FE::Real(0.04) * x);
        setFieldComponentValue(
            previous_solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.12) + FE::Real(0.03) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            0,
            FE::Real(0.40) + FE::Real(0.015) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            1,
            FE::Real(-0.20) + FE::Real(0.010) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            2,
            FE::Real(0.30) - FE::Real(0.005) * x);
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        1.0e-6,
        5.0e-5,
        1.0e-8);
}

TEST(LevelSetTransport,
     DiscontinuityCapturingZeroStrongResidualJacobianMatchesCentralDifference)
{
    constexpr FE::Real speed = 0.35;
    constexpr FE::Real dt = 0.1;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    options.velocity.constant_value = {0.0, speed, 0.0};
    options.supg.enabled = true;
    options.supg.discontinuity_capturing_enabled = true;

    FE::systems::FormInstallOptions install_options{};
    install_options.compiler_options.jit.enable = true;
    (void)level_set::installLevelSetTransport(
        system, phi_space, options, install_options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const FE::Real y = mesh->getNodeCoordinates(vertex)[1];
        // phi(y,t_n)=y and phi(y,t_{n-1})=y+u_y*dt gives
        // dt(phi)+u.grad(phi)=0 pointwise on this affine P1 cell.
        setFieldComponentValue(solution, system, phi, vertex, 0, y);
        setFieldComponentValue(
            previous, system, phi, vertex, 0, y + speed * dt);
    }

    FE::systems::SystemStateView state;
    state.dt = dt;
    state.u = solution;
    state.u_prev = previous;
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        /*eps=*/1.0e-7,
        /*rtol=*/2.0e-5,
        /*atol=*/2.0e-8);
}

TEST(LevelSetTransport,
     AlgebraicWetExtensionCarriesVelocityYToPhiForDryCutSupportSourceRow)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = phi_space, .components = 1});
    const auto physical_velocity = system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "LevelSetAdvectionVelocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;
    options.velocity.algebraic_extension_source_field_name = "Velocity";

    FE::systems::FormInstallOptions install_options{};
    install_options.compiler_options.jit.enable = true;
    (void)level_set::installLevelSetTransport(
        system, phi_space, options, install_options);
    const auto extension_velocity =
        system.findFieldByName("LevelSetAdvectionVelocity");
    ASSERT_NE(extension_velocity, FE::INVALID_FIELD_ID);
    auto extension_kernel =
        level_set::findLevelSetVelocityExtensionConstraintKernel(
            system, "level_set", extension_velocity);
    ASSERT_TRUE(extension_kernel);

    std::vector<level_set::VelocityExtensionConstraintRow> rows;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        for (int component = 0; component < 3; ++component) {
            rows.push_back(level_set::VelocityExtensionConstraintRow{
                .vertex = vertex,
                .component = component,
                .dependencies = {
                    level_set::VelocityExtensionDependency{
                        .field = level_set::
                            VelocityExtensionDependencyField::SourceVelocity,
                        .vertex = vertex,
                        .component = component,
                        .coefficient = 1.0,
                    }},
            });
        }
    }
    // phi=y-0.25 cuts this P1 tetra.  Vertex 2 (y=1) is dry, but it supports
    // the cut-cell trace and therefore must carry an exact E_y=u_y source row
    // rather than a graph-extension dependency.
    constexpr FE::GlobalIndex dry_trace_vertex = 2;
    ASSERT_GT(mesh->getNodeCoordinates(dry_trace_vertex)[1] - 0.25, 0.0);
    const auto dry_trace_row = std::find_if(
        rows.begin(), rows.end(), [](const auto& row) {
          return row.vertex == dry_trace_vertex && row.component == 1;
        });
    ASSERT_NE(dry_trace_row, rows.end());
    ASSERT_EQ(dry_trace_row->dependencies.size(), 1u);
    EXPECT_EQ(dry_trace_row->dependencies.front().field,
              level_set::
                  VelocityExtensionDependencyField::SourceVelocity);
    EXPECT_EQ(dry_trace_row->dependencies.front().vertex,
              dry_trace_vertex);
    EXPECT_EQ(dry_trace_row->dependencies.front().component, 1);
    EXPECT_DOUBLE_EQ(dry_trace_row->dependencies.front().coefficient, 1.0);
    extension_kernel->setFrozenRows(std::move(rows), 1u);
    EXPECT_TRUE(extension_kernel->hasFrozenMap());
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto system_dof = [&](FE::FieldId field,
                                FE::GlobalIndex vertex,
                                int component) {
        const auto* entity_map =
            system.fieldDofHandler(field).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error("test field has no vertex DOF map");
        }
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (component < 0 ||
            static_cast<std::size_t>(component) >= dofs.size()) {
            throw std::runtime_error("test field component is out of range");
        }
        return system.fieldDofOffset(field) +
               dofs[static_cast<std::size_t>(component)];
    };

    const auto n = system.dofHandler().getNumDofs();
    std::vector<FE::Real> solution(static_cast<std::size_t>(n), 0.0);
    std::vector<FE::Real> previous = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto coordinates = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(
            solution, system, phi, vertex, 0, coordinates[1] - 0.25);
        setFieldComponentValue(
            previous, system, phi, vertex, 0, coordinates[1] - 0.25);
        for (int component = 0; component < 3; ++component) {
            const FE::Real value = component == 1 ? FE::Real{0.2} : 0.0;
            setFieldComponentValue(solution, system, physical_velocity,
                                   vertex, component, value);
            setFieldComponentValue(solution, system, extension_velocity,
                                   vertex, component, value);
        }
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = solution;
    state.u_prev = previous;
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    FE::assembly::DenseMatrixView jacobian(n);
    FE::systems::AssemblyRequest matrix_request;
    matrix_request.op = "level_set";
    matrix_request.want_matrix = true;
    const auto matrix_result =
        system.assemble(matrix_request, state, &jacobian, nullptr);
    ASSERT_TRUE(matrix_result.success) << matrix_result.error_message;

    constexpr FE::Real eps = 1.0e-7;
    FE::Real maximum_chain_entry = 0.0;
    for (FE::GlobalIndex source_vertex = 0; source_vertex < 4;
         ++source_vertex) {
        const auto source_y =
            system_dof(physical_velocity, source_vertex, 1);
        const auto extension_y =
            system_dof(extension_velocity, source_vertex, 1);

        std::vector<FE::Real> plus = solution;
        std::vector<FE::Real> minus = solution;
        // Perturb the physical velocity and move E along the exact active-row
        // constraint E=u.  This is the condensed monolithic u_y direction.
        plus[static_cast<std::size_t>(source_y)] += eps;
        plus[static_cast<std::size_t>(extension_y)] += eps;
        minus[static_cast<std::size_t>(source_y)] -= eps;
        minus[static_cast<std::size_t>(extension_y)] -= eps;

        auto state_plus = state;
        auto state_minus = state;
        state_plus.u = plus;
        state_minus.u = minus;
        const auto residual_plus = assembleLevelSetResidual(system, state_plus);
        const auto residual_minus =
            assembleLevelSetResidual(system, state_minus);

        for (FE::GlobalIndex row_vertex = 0; row_vertex < 4; ++row_vertex) {
            const auto phi_row = system_dof(phi, row_vertex, 0);
            const FE::Real finite_difference =
                (residual_plus[static_cast<std::size_t>(phi_row)] -
                 residual_minus[static_cast<std::size_t>(phi_row)]) /
                (2.0 * eps);
            const FE::Real assembled_chain =
                jacobian(phi_row, extension_y);
            maximum_chain_entry =
                std::max(maximum_chain_entry, std::abs(assembled_chain));
            EXPECT_NEAR(assembled_chain, finite_difference, 2.0e-8)
                << "phi row vertex=" << row_vertex
                << " source y vertex=" << source_vertex;
            // The active extension row itself must annihilate the same
            // constrained direction: d(E_y-u_y)=0.
            const auto extension_row = extension_y;
            EXPECT_NEAR(jacobian(extension_row, extension_y) +
                            jacobian(extension_row, source_y),
                        0.0,
                        1.0e-13);
        }
    }
    EXPECT_GT(maximum_chain_entry, 1.0e-6)
        << "Regression guard: the D18 u_y -> phi block must not be zero";
    extension_kernel->invalidateFrozenMap();
    EXPECT_FALSE(extension_kernel->hasFrozenMap());
}
