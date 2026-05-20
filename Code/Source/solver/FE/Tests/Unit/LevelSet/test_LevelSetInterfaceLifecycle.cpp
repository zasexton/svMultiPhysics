#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutDomainAssembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Basis/NodeOrderingConventions.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/FrameGeometry.h"
#include "Geometry/MappingFactory.h"
#include "Interfaces/LevelSetInterfaceGeometryWriter.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SingleTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
        : SingleTetraMeshAccess({
              std::array<FE::Real, 3>{0.0, 0.0, 0.0},
              std::array<FE::Real, 3>{1.0, 0.0, 0.0},
              std::array<FE::Real, 3>{0.0, 1.0, 0.0},
              std::array<FE::Real, 3>{0.0, 0.0, 1.0},
          })
    {
    }

    explicit SingleTetraMeshAccess(std::vector<std::array<FE::Real, 3>> nodes)
        : nodes_(std::move(nodes))
    {
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
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

class SingleTetra10GeometryMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetra10GeometryMeshAccess()
        : nodes_{
              std::array<FE::Real, 3>{0.0, 0.0, 0.0},
              std::array<FE::Real, 3>{1.0, 0.0, 0.0},
              std::array<FE::Real, 3>{0.0, 1.0, 0.0},
              std::array<FE::Real, 3>{0.0, 0.0, 1.0},
              std::array<FE::Real, 3>{0.5, 0.0, 0.0},
              std::array<FE::Real, 3>{0.5, 0.5, 0.0},
              std::array<FE::Real, 3>{0.0, 0.5, 0.0},
              std::array<FE::Real, 3>{0.0, 0.0, 0.5},
              std::array<FE::Real, 3>{0.5, 0.0, 0.5},
              std::array<FE::Real, 3>{0.0, 0.5, 0.5},
          }
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra10;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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
        coords.assign(nodes_.begin(), nodes_.end());
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
    std::array<std::array<FE::Real, 3>, 10> nodes_{};
};

class SingleQuadMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleQuadMeshAccess(FE::ElementType type = FE::ElementType::Quad4)
        : type_(type)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return type_;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (type_ == FE::ElementType::Quad4) {
            nodes = {0, 1, 2, 3};
        } else if (type_ == FE::ElementType::Quad8) {
            nodes = {0, 1, 2, 3, 4, 5, 6, 7};
        } else {
            nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        }
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
        if (type_ == FE::ElementType::Quad4) {
            coords.assign(nodes_.begin(), nodes_.begin() + 4);
        } else if (type_ == FE::ElementType::Quad8) {
            coords.assign(nodes_.begin(), nodes_.begin() + 8);
        } else {
            coords.assign(nodes_.begin(), nodes_.end());
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
    FE::ElementType type_{FE::ElementType::Quad4};
    std::array<std::array<FE::Real, 3>, 9> nodes_{{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
        {{0.0, -1.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{-1.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
    }};
};

class SingleHexMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleHexMeshAccess(FE::ElementType type = FE::ElementType::Hex8)
        : type_(type)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 8; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return type_;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (type_ == FE::ElementType::Hex8) {
            nodes = {0, 1, 2, 3, 4, 5, 6, 7};
        } else if (type_ == FE::ElementType::Hex20) {
            nodes.resize(20u);
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                nodes[i] = static_cast<FE::GlobalIndex>(i);
            }
        } else {
            nodes.resize(nodes_.size());
            for (std::size_t i = 0; i < nodes_.size(); ++i) {
                nodes[i] = static_cast<FE::GlobalIndex>(i);
            }
        }
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
        if (type_ == FE::ElementType::Hex8) {
            coords.assign(nodes_.begin(), nodes_.begin() + 8);
        } else if (type_ == FE::ElementType::Hex20) {
            coords.assign(nodes_.begin(), nodes_.begin() + 20);
        } else {
            coords.assign(nodes_.begin(), nodes_.end());
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
    FE::ElementType type_{FE::ElementType::Hex8};
    std::array<std::array<FE::Real, 3>, 27> nodes_{{
        {{-1.0, -1.0, -1.0}},
        {{1.0, -1.0, -1.0}},
        {{1.0, 1.0, -1.0}},
        {{-1.0, 1.0, -1.0}},
        {{-1.0, -1.0, 1.0}},
        {{1.0, -1.0, 1.0}},
        {{1.0, 1.0, 1.0}},
        {{-1.0, 1.0, 1.0}},
        {{0.0, -1.0, -1.0}},
        {{1.0, 0.0, -1.0}},
        {{0.0, 1.0, -1.0}},
        {{-1.0, 0.0, -1.0}},
        {{0.0, -1.0, 1.0}},
        {{1.0, 0.0, 1.0}},
        {{0.0, 1.0, 1.0}},
        {{-1.0, 0.0, 1.0}},
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
        {{0.0, 0.0, -1.0}},
        {{0.0, 0.0, 1.0}},
        {{0.0, -1.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{-1.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
    }};
};

class SingleUnsupportedMixedCellMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleUnsupportedMixedCellMeshAccess(FE::ElementType type)
        : type_(type)
    {
        if (type_ == FE::ElementType::Wedge6) {
            nodes_ = {
                std::array<FE::Real, 3>{0.0, 0.0, 0.0},
                std::array<FE::Real, 3>{1.0, 0.0, 0.0},
                std::array<FE::Real, 3>{0.0, 1.0, 0.0},
                std::array<FE::Real, 3>{0.0, 0.0, 1.0},
                std::array<FE::Real, 3>{1.0, 0.0, 1.0},
                std::array<FE::Real, 3>{0.0, 1.0, 1.0},
            };
        } else {
            nodes_ = {
                std::array<FE::Real, 3>{-1.0, -1.0, 0.0},
                std::array<FE::Real, 3>{1.0, -1.0, 0.0},
                std::array<FE::Real, 3>{1.0, 1.0, 0.0},
                std::array<FE::Real, 3>{-1.0, 1.0, 0.0},
                std::array<FE::Real, 3>{0.0, 0.0, 1.0},
            };
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(nodes_.size());
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return type_;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.resize(nodes_.size());
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            nodes[i] = static_cast<FE::GlobalIndex>(i);
        }
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
    FE::ElementType type_{FE::ElementType::Wedge6};
    std::vector<std::array<FE::Real, 3>> nodes_{};
};

class SingleTriangleMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleTriangleMeshAccess(
        FE::ElementType type = FE::ElementType::Triangle3)
        : type_(type)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 3; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return type_;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (type_ == FE::ElementType::Triangle3) {
            nodes = {0, 1, 2};
        } else {
            nodes = {0, 1, 2, 3, 4, 5};
        }
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
        if (type_ == FE::ElementType::Triangle3) {
            coords.assign(nodes_.begin(), nodes_.begin() + 3);
        } else {
            coords.assign(nodes_.begin(), nodes_.end());
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
    FE::ElementType type_{FE::ElementType::Triangle3};
    std::array<std::array<FE::Real, 3>, 6> nodes_{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.5, 0.0, 0.0}},
        {{0.5, 0.5, 0.0}},
        {{0.0, 0.5, 0.0}},
    }};
};

class MixedQuadTriangleMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit MixedQuadTriangleMeshAccess(bool reverse_cell_iteration = false)
        : reverse_cell_iteration_(reverse_cell_iteration)
    {
    }

    void setFieldLayoutRevision(std::uint64_t revision)
    {
        field_layout_revision_ = revision;
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 7; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override
    {
        return field_layout_revision_;
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex cell_id) const override
    {
        return cell_id == 0 ? FE::ElementType::Quad4 : FE::ElementType::Triangle3;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (cell_id == 0) {
            nodes = {0, 1, 2, 3};
        } else {
            nodes = {4, 5, 6};
        }
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
        if (reverse_cell_iteration_) {
            callback(1);
            callback(0);
        } else {
            callback(0);
            callback(1);
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
    bool reverse_cell_iteration_{false};
    std::uint64_t field_layout_revision_{17};
    std::array<std::array<FE::Real, 3>, 7> nodes_{{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
        {{2.0, 0.0, 0.0}},
        {{3.0, 0.0, 0.0}},
        {{2.0, 1.0, 0.0}},
    }};
};

class MixedQuadTriangleLinearH1Space final : public FE::spaces::FunctionSpace {
public:
    MixedQuadTriangleLinearH1Space()
        : quad_(makeElement(FE::ElementType::Quad4))
        , triangle_(makeElement(FE::ElementType::Triangle3))
    {
    }

    [[nodiscard]] FE::spaces::SpaceType space_type() const noexcept override
    {
        return FE::spaces::SpaceType::H1;
    }
    [[nodiscard]] FE::FieldType field_type() const noexcept override
    {
        return FE::FieldType::Scalar;
    }
    [[nodiscard]] FE::Continuity continuity() const noexcept override
    {
        return FE::Continuity::C0;
    }
    [[nodiscard]] int value_dimension() const noexcept override { return 1; }
    [[nodiscard]] int topological_dimension() const noexcept override { return 2; }
    [[nodiscard]] int polynomial_order() const noexcept override { return 1; }
    [[nodiscard]] int polynomial_order(FE::GlobalIndex /*cell_id*/) const noexcept override
    {
        return 1;
    }
    [[nodiscard]] bool is_variable_order() const noexcept override { return true; }
    [[nodiscard]] FE::ElementType element_type() const noexcept override
    {
        return FE::ElementType::Unknown;
    }
    [[nodiscard]] const FE::elements::Element& element() const noexcept override
    {
        return *quad_;
    }
    [[nodiscard]] const FE::elements::Element& getElement(
        FE::ElementType cell_type,
        FE::GlobalIndex /*cell_id*/) const noexcept override
    {
        return cell_type == FE::ElementType::Triangle3 ? *triangle_ : *quad_;
    }
    [[nodiscard]] std::shared_ptr<const FE::elements::Element> element_ptr()
        const noexcept override
    {
        return quad_;
    }
    [[nodiscard]] std::size_t dofs_per_element() const noexcept override
    {
        return quad_->num_dofs();
    }
    [[nodiscard]] std::size_t dofs_per_element(FE::GlobalIndex cell_id)
        const noexcept override
    {
        return cell_id == 1 ? triangle_->num_dofs() : quad_->num_dofs();
    }

    [[nodiscard]] Value evaluate(const Value& xi,
                                 const std::vector<FE::Real>& coefficients)
        const override
    {
        const auto& elem = elementForCoefficients(coefficients.size());
        std::vector<FE::Real> values(elem.num_dofs(), FE::Real{0});
        elem.basis().evaluate_values(xi, values);

        Value result{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            result[0] += values[i] * coefficients[i];
        }
        return result;
    }

    [[nodiscard]] Gradient evaluate_gradient(
        const Value& xi,
        const std::vector<FE::Real>& coefficients) const override
    {
        const auto& elem = elementForCoefficients(coefficients.size());
        std::vector<FE::basis::Gradient> gradients(elem.num_dofs());
        elem.basis().evaluate_gradients(xi, gradients);

        Gradient result{};
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            for (std::size_t d = 0; d < 3; ++d) {
                result[d] += gradients[i][d] * coefficients[i];
            }
        }
        return result;
    }

private:
    [[nodiscard]] static std::shared_ptr<FE::elements::Element> makeElement(
        FE::ElementType element_type)
    {
        FE::elements::ElementRequest request;
        request.element_type = element_type;
        request.field_type = FE::FieldType::Scalar;
        request.continuity = FE::Continuity::C0;
        request.basis_type = FE::BasisType::Lagrange;
        request.order = 1;
        return FE::elements::ElementFactory::create(request);
    }

    [[nodiscard]] const FE::elements::Element& elementForCoefficients(
        std::size_t coefficient_count) const
    {
        if (coefficient_count == quad_->num_dofs()) {
            return *quad_;
        }
        if (coefficient_count == triangle_->num_dofs()) {
            return *triangle_;
        }
        throw std::invalid_argument(
            "MixedQuadTriangleLinearH1Space: unexpected coefficient count");
    }

    std::shared_ptr<FE::elements::Element> quad_{};
    std::shared_ptr<FE::elements::Element> triangle_{};
};

class MixedHexTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit MixedHexTetraMeshAccess(bool reverse_cell_iteration = false)
        : reverse_cell_iteration_(reverse_cell_iteration)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 12; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex cell_id) const override
    {
        return cell_id == 0 ? FE::ElementType::Hex8 : FE::ElementType::Tetra4;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (cell_id == 0) {
            nodes = {0, 1, 2, 3, 4, 5, 6, 7};
        } else {
            nodes = {8, 9, 10, 11};
        }
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
        if (reverse_cell_iteration_) {
            callback(1);
            callback(0);
        } else {
            callback(0);
            callback(1);
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
    bool reverse_cell_iteration_{false};
    std::array<std::array<FE::Real, 3>, 12> nodes_{{
        {{-1.0, -1.0, -1.0}},
        {{1.0, -1.0, -1.0}},
        {{1.0, 1.0, -1.0}},
        {{-1.0, 1.0, -1.0}},
        {{-1.0, -1.0, 1.0}},
        {{1.0, -1.0, 1.0}},
        {{1.0, 1.0, 1.0}},
        {{-1.0, 1.0, 1.0}},
        {{2.0, 0.0, 0.0}},
        {{3.0, 0.0, 0.0}},
        {{2.0, 1.0, 0.0}},
        {{2.0, 0.0, 1.0}},
    }};
};

class MixedHexTetraLinearH1Space final : public FE::spaces::FunctionSpace {
public:
    MixedHexTetraLinearH1Space()
        : hex_(makeElement(FE::ElementType::Hex8))
        , tetra_(makeElement(FE::ElementType::Tetra4))
    {
    }

    [[nodiscard]] FE::spaces::SpaceType space_type() const noexcept override
    {
        return FE::spaces::SpaceType::H1;
    }
    [[nodiscard]] FE::FieldType field_type() const noexcept override
    {
        return FE::FieldType::Scalar;
    }
    [[nodiscard]] FE::Continuity continuity() const noexcept override
    {
        return FE::Continuity::C0;
    }
    [[nodiscard]] int value_dimension() const noexcept override { return 1; }
    [[nodiscard]] int topological_dimension() const noexcept override { return 3; }
    [[nodiscard]] int polynomial_order() const noexcept override { return 1; }
    [[nodiscard]] int polynomial_order(FE::GlobalIndex /*cell_id*/) const noexcept override
    {
        return 1;
    }
    [[nodiscard]] bool is_variable_order() const noexcept override { return true; }
    [[nodiscard]] FE::ElementType element_type() const noexcept override
    {
        return FE::ElementType::Unknown;
    }
    [[nodiscard]] const FE::elements::Element& element() const noexcept override
    {
        return *hex_;
    }
    [[nodiscard]] const FE::elements::Element& getElement(
        FE::ElementType cell_type,
        FE::GlobalIndex /*cell_id*/) const noexcept override
    {
        return cell_type == FE::ElementType::Tetra4 ? *tetra_ : *hex_;
    }
    [[nodiscard]] std::shared_ptr<const FE::elements::Element> element_ptr()
        const noexcept override
    {
        return hex_;
    }
    [[nodiscard]] std::size_t dofs_per_element() const noexcept override
    {
        return hex_->num_dofs();
    }
    [[nodiscard]] std::size_t dofs_per_element(FE::GlobalIndex cell_id)
        const noexcept override
    {
        return cell_id == 1 ? tetra_->num_dofs() : hex_->num_dofs();
    }

    [[nodiscard]] Value evaluate(const Value& xi,
                                 const std::vector<FE::Real>& coefficients)
        const override
    {
        const auto& elem = elementForCoefficients(coefficients.size());
        std::vector<FE::Real> values(elem.num_dofs(), FE::Real{0});
        elem.basis().evaluate_values(xi, values);

        Value result{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            result[0] += values[i] * coefficients[i];
        }
        return result;
    }

    [[nodiscard]] Gradient evaluate_gradient(
        const Value& xi,
        const std::vector<FE::Real>& coefficients) const override
    {
        const auto& elem = elementForCoefficients(coefficients.size());
        std::vector<FE::basis::Gradient> gradients(elem.num_dofs());
        elem.basis().evaluate_gradients(xi, gradients);

        Gradient result{};
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            for (std::size_t d = 0; d < 3; ++d) {
                result[d] += gradients[i][d] * coefficients[i];
            }
        }
        return result;
    }

private:
    [[nodiscard]] static std::shared_ptr<FE::elements::Element> makeElement(
        FE::ElementType element_type)
    {
        FE::elements::ElementRequest request;
        request.element_type = element_type;
        request.field_type = FE::FieldType::Scalar;
        request.continuity = FE::Continuity::C0;
        request.basis_type = FE::BasisType::Lagrange;
        request.order = 1;
        return FE::elements::ElementFactory::create(request);
    }

    [[nodiscard]] const FE::elements::Element& elementForCoefficients(
        std::size_t coefficient_count) const
    {
        if (coefficient_count == hex_->num_dofs()) {
            return *hex_;
        }
        if (coefficient_count == tetra_->num_dofs()) {
            return *tetra_;
        }
        throw std::invalid_argument(
            "MixedHexTetraLinearH1Space: unexpected coefficient count");
    }

    std::shared_ptr<FE::elements::Element> hex_{};
    std::shared_ptr<FE::elements::Element> tetra_{};
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

[[nodiscard]] FE::systems::SetupInputs makeSingleHexSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 8;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};
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

[[nodiscard]] FE::systems::SetupInputs makeMixedQuadTriangleSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 7;
    topo.n_edges = 7;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4, 7};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6};
    topo.cell2edge_offsets = {0, 4, 7};
    topo.cell2edge_data = {0, 1, 2, 3, 4, 5, 6};
    topo.edge2vertex_data = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        4, 5,
        5, 6,
        6, 4,
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6};
    topo.edge_gids = {0, 1, 2, 3, 4, 5, 6};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupInputs makeSingleUnsupportedMixedCellSetupInputs(
    FE::ElementType type)
{
    const FE::GlobalIndex vertex_count =
        type == FE::ElementType::Wedge6 ? 6 : 5;

    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = vertex_count;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, vertex_count};
    topo.cell2vertex_data.reserve(static_cast<std::size_t>(vertex_count));
    topo.vertex_gids.reserve(static_cast<std::size_t>(vertex_count));
    for (FE::GlobalIndex vertex = 0; vertex < vertex_count; ++vertex) {
        topo.cell2vertex_data.push_back(vertex);
        topo.vertex_gids.push_back(vertex);
    }
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupInputs makeMixedHexTetraSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 12;
    topo.n_edges = 18;
    topo.n_faces = 10;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 8, 12};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    topo.cell2edge_offsets = {0, 12, 18};
    topo.cell2edge_data = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17,
    };
    topo.edge2vertex_data = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        4, 5,
        5, 6,
        6, 7,
        7, 4,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        8, 9,
        9, 10,
        10, 8,
        8, 11,
        9, 11,
        10, 11,
    };
    topo.cell2face_offsets = {0, 6, 10};
    topo.cell2face_data = {
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9,
    };
    topo.face2vertex_offsets = {0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36};
    topo.face2vertex_data = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7,
        8, 9, 10,
        8, 9, 11,
        9, 10, 11,
        10, 8, 11,
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    topo.edge_gids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 11, 12, 13, 14, 15, 16, 17};
    topo.face_gids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.empty()) {
        throw std::runtime_error("setFieldComponentValue: missing vertex DOF");
    }
    const auto index = static_cast<std::size_t>(dofs.front() + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

std::vector<std::string> ruleSignatures(
    const std::vector<FE::geometry::CutQuadratureRule>& rules)
{
    std::vector<std::string> signatures;
    signatures.reserve(rules.size());
    for (const auto& rule : rules) {
        const auto& provenance = rule.provenance;
        signatures.push_back(
            std::to_string(provenance.parent_entity) + "|" +
            std::to_string(static_cast<int>(rule.side)) + "|" +
            provenance.implicit_quadrature_backend + "|" +
            provenance.selected_implicit_quadrature_backend + "|" +
            provenance.cut_topology_id + "|" +
            std::to_string(provenance.cut_topology_revision) + "|" +
            std::to_string(provenance.requested_quadrature_order) + "|" +
            std::to_string(provenance.achieved_quadrature_order) + "|" +
            std::to_string(rule.points.size()));
    }
    return signatures;
}

level_set::LevelSetGeneratedInterfaceResult buildMixedQuadTriangleResult(
    bool reverse_cell_iteration,
    level_set::ImplicitCutQuadratureBackend backend,
    level_set::ImplicitCutFallbackPolicy fallback_policy =
        level_set::ImplicitCutFallbackPolicy::Fail)
{
    const auto mesh =
        std::make_shared<MixedQuadTriangleMeshAccess>(reverse_cell_iteration);
    auto scalar_space = std::make_shared<MixedQuadTriangleLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, makeMixedQuadTriangleSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 4 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 902;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        backend;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 1;
    options.implicit_cut_fallback_policy = fallback_policy;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

level_set::LevelSetGeneratedInterfaceResult buildMixedQuadTriangleAutoResult(
    bool reverse_cell_iteration)
{
    return buildMixedQuadTriangleResult(
        reverse_cell_iteration,
        level_set::ImplicitCutQuadratureBackend::Auto);
}

level_set::LevelSetGeneratedInterfaceResult buildMixedHexTetraAutoResult(
    bool reverse_cell_iteration)
{
    const auto mesh =
        std::make_shared<MixedHexTetraMeshAccess>(reverse_cell_iteration);
    auto scalar_space = std::make_shared<MixedHexTetraLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, makeMixedHexTetraSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 8 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 903;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

FE::Real integrateVolumeMoment(
    const FE::interfaces::LevelSetInterfaceDomain& domain,
    FE::geometry::CutIntegrationSide side,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& moment)
{
    FE::Real value = 0.0;
    for (const auto& rule : domain.volumeQuadratureRules()) {
        if (rule.side != side) {
            continue;
        }
        for (const auto& point : rule.points) {
            value += moment(point.point) * point.weight;
        }
    }
    return value;
}

FE::Real integrateInterfaceMoment(
    const FE::interfaces::LevelSetInterfaceDomain& domain,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& moment)
{
    FE::Real value = 0.0;
    for (const auto& rule : domain.interfaceQuadratureRules()) {
        for (const auto& point : rule.points) {
            value += moment(point.point) * point.weight;
        }
    }
    return value;
}

FE::Real vectorNorm(const std::array<FE::Real, 3>& vector)
{
    return std::sqrt(vector[0] * vector[0] +
                     vector[1] * vector[1] +
                     vector[2] * vector[2]);
}

FE::Real vectorDot(const std::array<FE::Real, 3>& a,
                   const std::array<FE::Real, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void expectInterfaceNormalsAlignWithGradient(
    const FE::interfaces::LevelSetInterfaceDomain& domain,
    const std::function<std::array<FE::Real, 3>(
        const std::array<FE::Real, 3>&)>& gradient,
    FE::Real minimum_alignment)
{
    const auto interface_rules = domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());

    std::size_t checked_points = 0u;
    for (const auto& rule : interface_rules) {
        EXPECT_TRUE(rule.curved_geometry);
        for (const auto& point : rule.points) {
            const auto grad_phi = gradient(point.point);
            const FE::Real grad_norm = vectorNorm(grad_phi);
            const FE::Real normal_norm = vectorNorm(point.normal);
            ASSERT_GT(grad_norm, 1.0e-12);
            ASSERT_GT(normal_norm, 1.0e-12);
            EXPECT_NEAR(normal_norm, 1.0, 1.0e-12);
            EXPECT_NEAR(point.gradient_norm, grad_norm, 1.0e-10);
            const FE::Real alignment =
                vectorDot(point.normal, grad_phi) / (normal_norm * grad_norm);
            EXPECT_GE(alignment, minimum_alignment)
                << "point=(" << point.point[0] << ", " << point.point[1]
                << ", " << point.point[2] << ")";
            ++checked_points;
        }
    }
    EXPECT_GT(checked_points, 0u);
}

void expectSingleParentVolumeRulesPartitionMeasure(
    const FE::interfaces::LevelSetInterfaceDomain& domain,
    FE::Real tolerance)
{
    const auto rules = domain.volumeQuadratureRules();
    ASSERT_FALSE(rules.empty());

    const auto parent_cell = rules.front().provenance.parent_entity;
    FE::Real parent_measure = 0.0;
    FE::Real total_measure = 0.0;
    for (const auto& rule : rules) {
        EXPECT_EQ(rule.provenance.parent_entity, parent_cell);
        EXPECT_TRUE(std::isfinite(rule.measure));
        EXPECT_TRUE(std::isfinite(rule.parent_measure));
        EXPECT_GE(rule.measure, 0.0);
        EXPECT_GE(rule.parent_measure, 0.0);
        parent_measure = std::max(parent_measure, rule.parent_measure);
        total_measure += rule.measure;
    }

    EXPECT_GT(parent_measure, 0.0);
    EXPECT_NEAR(total_measure, parent_measure, tolerance);
}

void expectGeneratedCutRulesAreFinite(
    const FE::interfaces::LevelSetInterfaceDomain& domain)
{
    const auto check_rule =
        [](const FE::geometry::CutQuadratureRule& rule) {
            EXPECT_TRUE(std::isfinite(rule.measure));
            EXPECT_TRUE(std::isfinite(rule.parent_measure));
            EXPECT_TRUE(std::isfinite(rule.volume_fraction));
            EXPECT_GE(rule.measure, 0.0);
            EXPECT_GE(rule.volume_fraction, 0.0);
            EXPECT_LE(rule.volume_fraction, 1.0);
            ASSERT_FALSE(rule.points.empty());
            for (const auto& point : rule.points) {
                EXPECT_TRUE(std::isfinite(point.weight));
                EXPECT_GT(point.weight, 0.0);
                for (int d = 0; d < 3; ++d) {
                    EXPECT_TRUE(std::isfinite(point.point[static_cast<std::size_t>(d)]));
                    EXPECT_TRUE(std::isfinite(point.normal[static_cast<std::size_t>(d)]));
                }
            }
        };

    const auto volume_rules = domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    for (const auto& rule : volume_rules) {
        check_rule(rule);
    }

    const auto interface_rules = domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    for (const auto& rule : interface_rules) {
        check_rule(rule);
    }
}

FE::geometry::Matrix3x3 toGeometryMatrix(
    const FE::math::Matrix<FE::Real, 3, 3>& matrix)
{
    FE::geometry::Matrix3x3 out{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            out[i][j] = matrix(i, j);
        }
    }
    return out;
}

FE::math::Vector<FE::Real, 3> toMathPoint(
    const std::array<FE::Real, 3>& point)
{
    FE::math::Vector<FE::Real, 3> out{};
    out[0] = point[0];
    out[1] = point[1];
    out[2] = point[2];
    return out;
}

FE::math::Vector<FE::Real, 3> curvedHexPhysicalPoint(
    const FE::math::Vector<FE::Real, 3>& xi)
{
    constexpr FE::Real sx = 1.25;
    constexpr FE::Real sy = 0.75;
    constexpr FE::Real sz = 1.5;
    constexpr FE::Real bend = 0.35;
    FE::math::Vector<FE::Real, 3> out{};
    out[0] = sx * xi[0];
    out[1] = sy * xi[1];
    out[2] = sz * xi[2] + bend * xi[0] * xi[1];
    return out;
}

std::shared_ptr<FE::geometry::GeometryMapping> makeCurvedHex27Mapping()
{
    std::vector<FE::math::Vector<FE::Real, 3>> nodes;
    nodes.reserve(FE::basis::NodeOrdering::num_nodes(FE::ElementType::Hex27));
    for (std::size_t i = 0;
         i < FE::basis::NodeOrdering::num_nodes(FE::ElementType::Hex27);
         ++i) {
        nodes.push_back(curvedHexPhysicalPoint(
            FE::basis::NodeOrdering::get_node_coords(FE::ElementType::Hex27, i)));
    }

    FE::geometry::MappingRequest request{};
    request.element_type = FE::ElementType::Hex27;
    request.geometry_order = 2;
    request.use_affine = false;
    return FE::geometry::MappingFactory::create(request, nodes);
}

FE::Real mappedVolumeMeasure(
    const FE::geometry::CutQuadratureRule& rule,
    const FE::geometry::GeometryMapping& mapping)
{
    FE::Real measure = 0.0;
    for (const auto& point : rule.points) {
        measure += point.weight *
                   std::abs(mapping.jacobian_determinant(toMathPoint(point.point)));
    }
    return measure;
}

FE::Real mappedInterfaceMeasure(
    const FE::geometry::CutQuadratureRule& rule,
    const FE::geometry::GeometryMapping& mapping)
{
    FE::Real measure = 0.0;
    for (const auto& point : rule.points) {
        const auto xi = toMathPoint(point.point);
        const auto transform =
            FE::geometry::surfaceTransformFromJacobianInverse(
                point.normal,
                point.weight,
                toGeometryMatrix(mapping.jacobian_inverse(xi)),
                mapping.jacobian_determinant(xi));
        measure += transform.measure;
    }
    return measure;
}

FE::Real expectedCurvedHexMidplaneArea()
{
    constexpr FE::Real sx = 1.25;
    constexpr FE::Real sy = 0.75;
    constexpr FE::Real bend = 0.35;
    constexpr std::array<FE::Real, 5> points{{
        -0.9061798459386640,
        -0.5384693101056831,
        0.0,
        0.5384693101056831,
        0.9061798459386640,
    }};
    constexpr std::array<FE::Real, 5> weights{{
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891,
    }};

    FE::Real area = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = 0; j < points.size(); ++j) {
            const FE::Real xi = points[i];
            const FE::Real eta = points[j];
            const FE::Real density =
                std::sqrt((sx * sy) * (sx * sy) +
                          (bend * sy * eta) * (bend * sy * eta) +
                          (bend * sx * xi) * (bend * sx * xi));
            area += weights[i] * weights[j] * density;
        }
    }
    return area;
}

FE::Real ellipsePerimeterReference(FE::Real semi_major, FE::Real semi_minor)
{
    constexpr int intervals = 4096;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const FE::Real h = 2.0 * pi / static_cast<FE::Real>(intervals);
    const auto density = [semi_major, semi_minor](FE::Real theta) {
        const FE::Real s = std::sin(theta);
        const FE::Real c = std::cos(theta);
        return std::sqrt(semi_major * semi_major * s * s +
                         semi_minor * semi_minor * c * c);
    };

    FE::Real sum = density(0.0) + density(2.0 * pi);
    for (int i = 1; i < intervals; ++i) {
        sum += (i % 2 == 0 ? 2.0 : 4.0) * density(h * static_cast<FE::Real>(i));
    }
    return h * sum / 3.0;
}

FE::Real prolateSpheroidSurfaceArea(FE::Real axis_x, FE::Real axis_y)
{
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const FE::Real eccentricity =
        std::sqrt(FE::Real{1.0} - (axis_y * axis_y) / (axis_x * axis_x));
    if (eccentricity < FE::Real{1.0e-14}) {
        return FE::Real{4.0} * pi * axis_x * axis_x;
    }
    return FE::Real{2.0} * pi * axis_y * axis_y *
           (FE::Real{1.0} +
            axis_x * std::asin(eccentricity) / (axis_y * eccentricity));
}

level_set::LevelSetGeneratedInterfaceResult buildSingleQuadCircleCut(
    FE::ElementType element_type,
    int level_set_order,
    int subdivision_depth,
    int interface_order,
    int volume_order,
    int interface_marker,
    FE::Real radius = 0.5,
    level_set::ImplicitCutFallbackPolicy fallback_policy =
        level_set::ImplicitCutFallbackPolicy::Fail)
{
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(element_type);
    auto scalar_space = FE::spaces::Space(FE::spaces::SpaceType::H1,
                                          mesh,
                                          level_set_order,
                                          /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, makeSingleQuadSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = subdivision_depth;
    options.interface_quadrature_order = interface_order;
    options.volume_quadrature_order = volume_order;
    options.implicit_cut_fallback_policy = fallback_policy;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

level_set::LevelSetGeneratedInterfaceResult buildSingleQuadEllipseCut(
    FE::Real semi_major,
    FE::Real semi_minor,
    int subdivision_depth,
    int interface_order,
    int volume_order,
    int interface_marker,
    bool require_production_qualified_backend = false)
{
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, makeSingleQuadSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] / (semi_major * semi_major) +
            x[1] * x[1] / (semi_minor * semi_minor) - 1.0;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = subdivision_depth;
    options.interface_quadrature_order = interface_order;
    options.volume_quadrature_order = volume_order;
    options.require_production_qualified_implicit_cut_backend =
        require_production_qualified_backend;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

level_set::LevelSetGeneratedInterfaceResult buildSingleHexSayeCut(
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& level_set_function,
    int subdivision_depth,
    int interface_order,
    int volume_order,
    int interface_marker,
    level_set::ImplicitCutFallbackPolicy fallback_policy =
        level_set::ImplicitCutFallbackPolicy::LinearCorner)
{
    const auto mesh = std::make_shared<SingleHexMeshAccess>(FE::ElementType::Hex27);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, makeSingleHexSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            level_set_function(x);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = subdivision_depth;
    options.interface_quadrature_order = interface_order;
    options.volume_quadrature_order = volume_order;
    options.implicit_cut_fallback_policy = fallback_policy;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

level_set::LevelSetGeneratedInterfaceResult buildSingleTriangleHighOrderSubcellCut(
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& level_set_function,
    int interface_marker,
    int subdivision_depth = 3,
    int interface_order = 1,
    int volume_order = 2)
{
    const auto mesh =
        std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle6);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, makeSingleTriangleSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            level_set_function(x);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = subdivision_depth;
    options.interface_quadrature_order = interface_order;
    options.volume_quadrature_order = volume_order;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

level_set::LevelSetGeneratedInterfaceResult buildSingleTetraHighOrderSubcellCut(
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& level_set_function,
    int interface_marker,
    int subdivision_depth,
    int interface_order,
    int volume_order)
{
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            level_set_function(x);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = subdivision_depth;
    options.interface_quadrature_order = interface_order;
    options.volume_quadrature_order = volume_order;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

int affineFixtureLevelSetOrder(FE::ElementType element_type)
{
    switch (element_type) {
    case FE::ElementType::Quad8:
    case FE::ElementType::Quad9:
    case FE::ElementType::Triangle6:
    case FE::ElementType::Tetra10:
    case FE::ElementType::Hex20:
    case FE::ElementType::Hex27:
        return 2;
    default:
        return 1;
    }
}

std::shared_ptr<FE::spaces::FunctionSpace> makeAffineFixtureScalarSpace(
    FE::ElementType element_type,
    const std::shared_ptr<FE::assembly::IMeshAccess>& mesh)
{
    if (element_type == FE::ElementType::Quad8 ||
        element_type == FE::ElementType::Hex20) {
        FE::spaces::SpaceRequest request{};
        request.space_type = FE::spaces::SpaceType::H1;
        request.element.element_type = element_type;
        request.element.basis_type = FE::BasisType::Serendipity;
        request.element.field_type = FE::FieldType::Scalar;
        request.element.continuity = FE::Continuity::C0;
        request.element.order = affineFixtureLevelSetOrder(element_type);
        return FE::spaces::Space(request);
    }

    return FE::spaces::Space(FE::spaces::SpaceType::H1,
                             mesh,
                             affineFixtureLevelSetOrder(element_type),
                             /*components=*/1);
}

level_set::LevelSetGeneratedInterfaceResult buildSayeAffineHyperrectangleCut(
    FE::ElementType element_type,
    int interface_marker,
    FE::Real cut_coordinate,
    int subdivision_depth = 3,
    int interface_order = 1,
    int volume_order = 2)
{
    const bool is_2d = element_type == FE::ElementType::Quad4 ||
                       element_type == FE::ElementType::Quad8 ||
                       element_type == FE::ElementType::Quad9;
    std::shared_ptr<FE::assembly::IMeshAccess> mesh =
        is_2d ? std::static_pointer_cast<FE::assembly::IMeshAccess>(
                    std::make_shared<SingleQuadMeshAccess>(element_type))
              : std::static_pointer_cast<FE::assembly::IMeshAccess>(
                    std::make_shared<SingleHexMeshAccess>(element_type));
    auto scalar_space = makeAffineFixtureScalarSpace(element_type, mesh);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, is_2d ? makeSingleQuadSetupInputs()
                           : makeSingleHexSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] - cut_coordinate;
    }
    const FE::GlobalIndex vertex_count = is_2d ? 4 : 8;
    for (FE::GlobalIndex vertex = 0; vertex < vertex_count; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] - cut_coordinate);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = subdivision_depth;
    options.interface_quadrature_order = interface_order;
    options.volume_quadrature_order = volume_order;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

level_set::LevelSetGeneratedInterfaceResult buildHighOrderSubcellAffineSimplexCut(
    FE::ElementType element_type,
    int interface_marker,
    FE::Real cut_coordinate,
    int subdivision_depth = 3,
    int interface_order = 1,
    int volume_order = 2)
{
    const bool is_2d = element_type == FE::ElementType::Triangle3 ||
                       element_type == FE::ElementType::Triangle6;
    std::shared_ptr<FE::assembly::IMeshAccess> mesh;
    if (element_type == FE::ElementType::Triangle3 ||
        element_type == FE::ElementType::Triangle6) {
        mesh = std::make_shared<SingleTriangleMeshAccess>(element_type);
    } else if (element_type == FE::ElementType::Tetra10) {
        mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    } else {
        mesh = std::make_shared<SingleTetraMeshAccess>();
    }

    auto scalar_space = makeAffineFixtureScalarSpace(element_type, mesh);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.setup({}, is_2d ? makeSingleTriangleSetupInputs()
                           : makeSingleTetraSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] - cut_coordinate;
    }
    const FE::GlobalIndex vertex_count = is_2d ? 3 : 4;
    for (FE::GlobalIndex vertex = 0; vertex < vertex_count; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] - cut_coordinate);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = subdivision_depth;
    options.interface_quadrature_order = interface_order;
    options.volume_quadrature_order = volume_order;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

class CutMeasureAssemblyKernel final : public FE::assembly::AssemblyKernel {
public:
    [[nodiscard]] FE::assembly::RequiredData getRequiredData() const override
    {
        return FE::assembly::RequiredData::IntegrationWeights;
    }

    void computeCell(const FE::assembly::AssemblyContext& ctx,
                     FE::assembly::KernelOutput& output) override
    {
        integrate(ctx, output);
    }

    void computeBoundaryFace(const FE::assembly::AssemblyContext& ctx,
                             int /*boundary_marker*/,
                             FE::assembly::KernelOutput& output) override
    {
        integrate(ctx, output);
    }

    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

private:
    static void integrate(const FE::assembly::AssemblyContext& ctx,
                          FE::assembly::KernelOutput& output)
    {
        output.reserve(/*n_test=*/1, /*n_trial=*/0, /*need_matrix=*/false,
                       /*need_vector=*/true);
        for (FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            output.vectorEntry(0) += ctx.integrationWeight(q);
        }
    }
};

void populateMeasureAssemblyContext(
    const FE::geometry::CutQuadratureRule& rule,
    FE::assembly::AssemblyContext& ctx)
{
    std::vector<std::array<FE::Real, 3>> points;
    std::vector<std::array<FE::Real, 3>> normals;
    std::vector<FE::Real> weights;
    points.reserve(rule.points.size());
    normals.reserve(rule.points.size());
    weights.reserve(rule.points.size());
    for (const auto& point : rule.points) {
        points.push_back(point.point);
        normals.push_back(point.normal);
        weights.push_back(point.weight);
    }
    ctx.setQuadratureData(points, weights);
    ctx.setPhysicalPoints(points);
    ctx.setIntegrationWeights(weights);
    ctx.setNormals(normals);
}

} // namespace

TEST(LevelSetInterfaceLifecycle, BuildsDomainFromScalarField)
{
    constexpr int interface_marker = 73;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.tolerance = 1.0e-12;
    options.interface_quadrature_order = 0;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.value_revision, 1u);
    EXPECT_EQ(lifecycle.valueRevision(), 1u);
    EXPECT_EQ(result.cell_count, 1u);
    EXPECT_EQ(result.cell_cache_hits, 0u);
    EXPECT_EQ(result.cell_cache_misses, 1u);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.max_cell_node_count, 4u);
    EXPECT_EQ(result.max_corner_node_count, 4u);
    EXPECT_EQ(result.domain.marker(), interface_marker);
    EXPECT_EQ(result.domain.request().source.field_id, phi);
    EXPECT_EQ(result.domain.request().source.layout_revision,
              system.fieldDofHandler(phi).getDofStateRevision());
    EXPECT_EQ(result.domain.request().source.value_revision, result.value_revision);
    EXPECT_EQ(result.domain.request().mesh_geometry_revision, 7u);
    EXPECT_EQ(result.domain.request().mesh_topology_revision, 11u);
    EXPECT_EQ(result.domain.request().ownership_revision, 13u);
    EXPECT_NE(result.domain.request().quadrature_policy_key, 0u);
    EXPECT_EQ(result.domain.request().implicit_geometry_mode, "LinearCorner");
    EXPECT_EQ(result.domain.request().implicit_quadrature_backend, "LinearCorner");
    EXPECT_EQ(result.domain.request().implicit_fallback_policy, "Fail");
    EXPECT_EQ(result.domain.request().implicit_fallback_status, "None");
    EXPECT_EQ(result.domain.request().geometry_tangent_policy,
              "RefreshedFrozenQuadrature");
    EXPECT_DOUBLE_EQ(result.domain.request().implicit_cut_root_coordinate_tolerance,
                     1.0e-12);
    EXPECT_EQ(result.domain.request().implicit_cut_root_max_iterations, 48);
    EXPECT_EQ(result.geometry_tangent_policy,
              level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
    EXPECT_EQ(result.domain.request().resolvedInterfaceQuadratureOrder(), 0);
    EXPECT_EQ(result.domain.request().resolvedVolumeQuadratureOrder(), 1);
    EXPECT_EQ(result.summary.interface_marker, interface_marker);
    EXPECT_EQ(result.summary.active_fragment_count, 1u);
    EXPECT_EQ(result.summary.active_volume_region_count, 2u);
    EXPECT_EQ(result.summary.quadrature_point_count, 1u);
    EXPECT_GT(result.summary.volume_quadrature_point_count, 0u);
    EXPECT_EQ(result.summary.total_quadrature_point_count,
              result.summary.quadrature_point_count +
                  result.summary.volume_quadrature_point_count);
    EXPECT_GT(result.summary.measure, 0.0);
    EXPECT_GT(result.summary.negative_volume_measure, 0.0);
    EXPECT_GT(result.summary.positive_volume_measure, 0.0);
    ASSERT_EQ(result.domain.fragments().size(), 1u);
    EXPECT_EQ(result.domain.fragments().front().interface_marker, interface_marker);
    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 1u);
    EXPECT_EQ(interface_rules.front().exact_polynomial_order, 0);
    EXPECT_EQ(interface_rules.front().provenance.predicate_policy_key,
              result.domain.request().quadrature_policy_key);
    EXPECT_EQ(interface_rules.front().provenance.implicit_quadrature_backend,
              "LinearCorner");
    EXPECT_EQ(interface_rules.front().provenance.implicit_fallback_status,
              "None");
    EXPECT_EQ(interface_rules.front().provenance.implicit_cut_root_tolerance,
              result.domain.request().implicit_cut_root_tolerance);
    EXPECT_EQ(interface_rules.front().provenance.implicit_cut_root_coordinate_tolerance,
              result.domain.request().implicit_cut_root_coordinate_tolerance);
    EXPECT_EQ(interface_rules.front().provenance.implicit_cut_root_max_iterations,
              result.domain.request().implicit_cut_root_max_iterations);
    EXPECT_EQ(interface_rules.front().provenance.geometry_tangent_policy,
              "RefreshedFrozenQuadrature");
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 0);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 0);
    ASSERT_EQ(result.domain.volumeRegions().size(), 2u);
    EXPECT_EQ(result.domain.volumeRegions().front().interface_marker, interface_marker);
    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 2u);
    std::size_t exported_volume_quadrature_points = 0u;
    for (const auto& rule : volume_rules) {
        exported_volume_quadrature_points += rule.points.size();
    }
    EXPECT_EQ(result.summary.volume_quadrature_point_count,
              exported_volume_quadrature_points);
    EXPECT_EQ(volume_rules.front().exact_polynomial_order, 1);
    EXPECT_EQ(volume_rules.front().provenance.predicate_policy_key,
              result.domain.request().quadrature_policy_key);
    EXPECT_EQ(volume_rules.front().provenance.implicit_geometry_mode,
              "LinearCorner");
    EXPECT_EQ(volume_rules.front().provenance.geometry_tangent_policy,
              "RefreshedFrozenQuadrature");
    EXPECT_EQ(volume_rules.front().provenance.implicit_fallback_status,
              "None");
    EXPECT_EQ(volume_rules.front().provenance.implicit_cut_root_tolerance,
              result.domain.request().implicit_cut_root_tolerance);
    EXPECT_EQ(volume_rules.front().provenance.implicit_cut_root_coordinate_tolerance,
              result.domain.request().implicit_cut_root_coordinate_tolerance);
    EXPECT_EQ(volume_rules.front().provenance.implicit_cut_root_max_iterations,
              result.domain.request().implicit_cut_root_max_iterations);
    EXPECT_EQ(volume_rules.front().provenance.requested_quadrature_order, 1);
    EXPECT_EQ(volume_rules.front().provenance.achieved_quadrature_order, 1);
}

TEST(LevelSetInterfaceLifecycle, ReusesUnchangedGeneratedInterfaceCells)
{
    constexpr int interface_marker = 874;
    const auto mesh = std::make_shared<MixedQuadTriangleMeshAccess>();
    auto scalar_space = std::make_shared<MixedQuadTriangleLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeMixedQuadTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 4 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto initial = lifecycle.build(system, options, solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;
    EXPECT_EQ(initial.cell_count, 2u);
    EXPECT_EQ(initial.cell_cache_hits, 0u);
    EXPECT_EQ(initial.cell_cache_misses, 2u);
    EXPECT_EQ(initial.domain_cache_hits, 0u);
    EXPECT_GT(initial.backend_elapsed_seconds, 0.0);

    const auto repeated = lifecycle.build(system, options, solution);
    ASSERT_TRUE(repeated.success) << repeated.diagnostic;
    EXPECT_EQ(repeated.value_revision, initial.value_revision + 1u);
    EXPECT_EQ(repeated.cell_count, 2u);
    EXPECT_EQ(repeated.cell_cache_hits, 2u);
    EXPECT_EQ(repeated.cell_cache_misses, 0u);
    EXPECT_EQ(repeated.domain_cache_hits, 1u);
    EXPECT_EQ(repeated.backend_elapsed_seconds, 0.0);
    EXPECT_NEAR(repeated.summary.negative_volume_measure,
                initial.summary.negative_volume_measure,
                1.0e-12);
    const auto repeated_rules = repeated.domain.volumeQuadratureRules();
    ASSERT_FALSE(repeated_rules.empty());
    EXPECT_EQ(repeated_rules.front().provenance.source_value_revision,
              repeated.value_revision);

    setFieldComponentValue(solution, system, phi, 0, FE::Real{-0.75});
    const auto updated = lifecycle.build(system, options, solution);
    ASSERT_TRUE(updated.success) << updated.diagnostic;
    EXPECT_EQ(updated.value_revision, repeated.value_revision + 1u);
    EXPECT_EQ(updated.cell_count, 2u);
    EXPECT_EQ(updated.cell_cache_hits, 1u);
    EXPECT_EQ(updated.cell_cache_misses, 1u);
    EXPECT_EQ(updated.domain_cache_hits, 0u);
    EXPECT_GT(updated.backend_elapsed_seconds, 0.0);
    EXPECT_GT(std::abs(updated.summary.negative_volume_measure -
                       repeated.summary.negative_volume_measure),
              1.0e-4);
}

TEST(LevelSetInterfaceLifecycle, ReusesCellsAcrossUnrelatedMeshFieldLayoutChanges)
{
    constexpr int interface_marker = 878;
    const auto mesh = std::make_shared<MixedQuadTriangleMeshAccess>();
    auto scalar_space = std::make_shared<MixedQuadTriangleLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeMixedQuadTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, x[0]);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto initial = lifecycle.build(system, options, solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;
    EXPECT_EQ(initial.cell_cache_hits, 0u);
    EXPECT_EQ(initial.cell_cache_misses, 2u);
    EXPECT_EQ(initial.domain_cache_hits, 0u);

    mesh->setFieldLayoutRevision(99u);
    const auto repeated = lifecycle.build(system, options, solution);
    ASSERT_TRUE(repeated.success) << repeated.diagnostic;
    EXPECT_EQ(repeated.cell_cache_hits, 2u);
    EXPECT_EQ(repeated.cell_cache_misses, 0u);
    EXPECT_EQ(repeated.domain_cache_hits, 1u);
    EXPECT_EQ(repeated.backend_elapsed_seconds, 0.0);
}

TEST(LevelSetInterfaceLifecycle, ReusesLinearFullCellsAcrossMagnitudeChanges)
{
    constexpr int interface_marker = 875;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(solution, system, phi, vertex, FE::Real{1.0});
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.tolerance = 1.0e-12;
    options.interface_quadrature_order = 0;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto initial = lifecycle.build(system, options, solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;
    EXPECT_EQ(initial.cell_cache_hits, 0u);
    EXPECT_EQ(initial.cell_cache_misses, 1u);
    EXPECT_EQ(initial.linear_full_cell_fast_path_count, 1u);
    EXPECT_EQ(initial.backend_elapsed_seconds, 0.0);
    EXPECT_NEAR(initial.summary.negative_volume_measure, 0.0, 1.0e-12);
    EXPECT_GT(initial.summary.positive_volume_measure, 0.0);

    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(solution, system, phi, vertex, FE::Real{2.0});
    }
    const auto shifted = lifecycle.build(system, options, solution);
    ASSERT_TRUE(shifted.success) << shifted.diagnostic;
    EXPECT_EQ(shifted.cell_cache_hits, 1u);
    EXPECT_EQ(shifted.cell_cache_misses, 0u);
    EXPECT_EQ(shifted.linear_full_cell_fast_path_count, 1u);
    EXPECT_EQ(shifted.backend_elapsed_seconds, 0.0);
    EXPECT_NEAR(shifted.summary.negative_volume_measure, 0.0, 1.0e-12);
    EXPECT_NEAR(shifted.summary.positive_volume_measure,
                initial.summary.positive_volume_measure,
                1.0e-12);
    ASSERT_EQ(shifted.domain.volumeRegions().size(), 1u);
    EXPECT_DOUBLE_EQ(shifted.domain.volumeRegions().front().min_level_set_value,
                     2.0);
    EXPECT_DOUBLE_EQ(shifted.domain.volumeRegions().front().max_level_set_value,
                     2.0);
}

TEST(LevelSetInterfaceLifecycle, HighOrderImplicitBypassesBackendForLinearFullCells)
{
    constexpr int interface_marker = 879;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(solution, system, phi, vertex, FE::Real{-1.0});
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cell_count, 1u);
    EXPECT_EQ(result.cell_cache_hits, 0u);
    EXPECT_EQ(result.cell_cache_misses, 1u);
    EXPECT_EQ(result.linear_full_cell_fast_path_count, 1u);
    EXPECT_EQ(result.backend_elapsed_seconds, 0.0);
    EXPECT_EQ(result.summary.active_fragment_count, 0u);
    EXPECT_EQ(result.summary.active_volume_region_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure, 4.0, 1.0e-12);
    EXPECT_NEAR(result.summary.positive_volume_measure, 0.0, 1.0e-12);

    ASSERT_EQ(result.domain.volumeRegions().size(), 1u);
    EXPECT_TRUE(result.domain.volumeRegions().front().full_cell_equivalent);
    const auto rules = result.domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_TRUE(rules.front().full_cell_equivalent);
    EXPECT_EQ(rules.front().provenance.implicit_quadrature_backend,
              "SayeHyperrectangle");
    EXPECT_EQ(rules.front().provenance.implicit_fallback_status, "None");
}

TEST(LevelSetInterfaceLifecycle, LinearCorner2DSegmentSupportsHigherInterfaceQuadrature)
{
    constexpr int interface_marker = 74;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.tolerance = 1.0e-12;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.achieved_interface_quadrature_order, 2);
    EXPECT_EQ(result.domain.request().resolvedInterfaceQuadratureOrder(), 2);
    EXPECT_EQ(result.summary.quadrature_point_count, 2u);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 1u);
    EXPECT_EQ(interface_rules.front().exact_polynomial_order, 2);
    EXPECT_EQ(interface_rules.front().policy.name,
              "gauss-segment-level-set-interface");
    EXPECT_EQ(interface_rules.front().points.size(), 2u);
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 2);
}

TEST(LevelSetInterfaceLifecycle, QuadraturePolicyKeyChangesWithBackendOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions base_options{};
    base_options.level_set_field_name = "phi";
    base_options.requested_interface_marker = 74;
    base_options.domain_id = "water-air";
    base_options.interface_quadrature_order = 0;
    base_options.volume_quadrature_order = 1;

    const auto build_policy_key =
        [&](const level_set::LevelSetGeneratedInterfaceOptions& options)
            -> std::uint64_t {
        level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        const auto result = lifecycle.build(system, options, solution);
        EXPECT_TRUE(result.success) << result.diagnostic;
        return result.domain.request().quadrature_policy_key;
    };

    const auto base_key = build_policy_key(base_options);
    ASSERT_NE(base_key, 0u);

    auto changed_tolerance = base_options;
    changed_tolerance.implicit_cut_root_tolerance = 1.0e-8;
    EXPECT_NE(base_key, build_policy_key(changed_tolerance));

    auto changed_coordinate_tolerance = base_options;
    changed_coordinate_tolerance.implicit_cut_root_coordinate_tolerance = 5.0e-11;
    EXPECT_NE(base_key, build_policy_key(changed_coordinate_tolerance));

    auto changed_root_iterations = base_options;
    changed_root_iterations.implicit_cut_root_max_iterations = 32;
    EXPECT_NE(base_key, build_policy_key(changed_root_iterations));

    auto changed_depth = base_options;
    changed_depth.implicit_cut_max_subdivision_depth = 5;
    EXPECT_NE(base_key, build_policy_key(changed_depth));

    auto changed_fallback = base_options;
    changed_fallback.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::LinearCorner;
    EXPECT_NE(base_key, build_policy_key(changed_fallback));

    auto changed_interface_order = base_options;
    changed_interface_order.interface_quadrature_order = 1;
    EXPECT_NE(base_key, build_policy_key(changed_interface_order));

    auto changed_volume_order = base_options;
    changed_volume_order.volume_quadrature_order = 2;
    EXPECT_NE(base_key, build_policy_key(changed_volume_order));

    auto changed_degenerate_policy = base_options;
    changed_degenerate_policy.keep_degenerate_fragments = true;
    EXPECT_NE(base_key, build_policy_key(changed_degenerate_policy));

    auto changed_corner_policy = base_options;
    changed_corner_policy.allow_corner_linearized_geometry = true;
    EXPECT_NE(base_key, build_policy_key(changed_corner_policy));
}

TEST(LevelSetInterfaceLifecycle,
     RejectsDifferentiatedQuadratureTangentPolicyUntilSensitivitiesExist)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.geometry_tangent_policy =
        level_set::GeometryTangentPolicy::DifferentiatedQuadrature;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected differentiated quadrature tangent policy rejection";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("DifferentiatedQuadrature"), std::string::npos);
        EXPECT_NE(message.find("point"), std::string::npos);
        EXPECT_NE(message.find("weight"), std::string::npos);
        EXPECT_NE(message.find("measure"), std::string::npos);
        EXPECT_NE(message.find("normal"), std::string::npos);
        EXPECT_NE(message.find("topology"), std::string::npos);
        EXPECT_NE(message.find("sensitivities"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, RejectsInvalidImplicitRootPolishingControls)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);
    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    level_set::LevelSetGeneratedInterfaceOptions options{};

    auto bad_coordinate_tolerance = options;
    bad_coordinate_tolerance.implicit_cut_root_coordinate_tolerance = 0.0;
    try {
        (void)lifecycle.build(system,
                              bad_coordinate_tolerance,
                              std::span<const FE::Real>{});
        FAIL() << "Expected invalid root coordinate tolerance to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("implicit_cut_root_coordinate_tolerance"),
                  std::string::npos);
        EXPECT_NE(message.find("positive"), std::string::npos);
    }

    auto bad_max_iterations = options;
    bad_max_iterations.implicit_cut_root_max_iterations = 0;
    try {
        (void)lifecycle.build(system,
                              bad_max_iterations,
                              std::span<const FE::Real>{});
        FAIL() << "Expected invalid root max iterations to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("implicit_cut_root_max_iterations"),
                  std::string::npos);
        EXPECT_NE(message.find("positive"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, ReportsQuadratureSensitivityCapabilityStub)
{
    const auto refreshed =
        level_set::geometryQuadratureSensitivitySupport(
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
    EXPECT_EQ(refreshed.policy,
              level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
    EXPECT_FALSE(refreshed.complete());
    EXPECT_FALSE(refreshed.point_location_sensitivity_available);
    EXPECT_FALSE(refreshed.quadrature_weight_sensitivity_available);
    EXPECT_FALSE(refreshed.measure_sensitivity_available);
    EXPECT_FALSE(refreshed.normal_sensitivity_available);
    EXPECT_FALSE(refreshed.topology_transition_sensitivity_available);
    EXPECT_NE(refreshed.diagnostic.find("fixed during tangent assembly"),
              std::string::npos);

    const auto differentiated =
        level_set::geometryQuadratureSensitivitySupport(
            level_set::GeometryTangentPolicy::DifferentiatedQuadrature);
    EXPECT_EQ(differentiated.policy,
              level_set::GeometryTangentPolicy::DifferentiatedQuadrature);
    EXPECT_FALSE(differentiated.complete());
    EXPECT_FALSE(differentiated.point_location_sensitivity_available);
    EXPECT_FALSE(differentiated.quadrature_weight_sensitivity_available);
    EXPECT_FALSE(differentiated.measure_sensitivity_available);
    EXPECT_FALSE(differentiated.normal_sensitivity_available);
    EXPECT_FALSE(differentiated.topology_transition_sensitivity_available);
    EXPECT_NE(differentiated.diagnostic.find("reserved"),
              std::string::npos);
    EXPECT_NE(differentiated.diagnostic.find("point"), std::string::npos);
    EXPECT_NE(differentiated.diagnostic.find("weight"), std::string::npos);
    EXPECT_NE(differentiated.diagnostic.find("measure"), std::string::npos);
    EXPECT_NE(differentiated.diagnostic.find("normal"), std::string::npos);
    EXPECT_NE(differentiated.diagnostic.find("topology"), std::string::npos);
    EXPECT_NE(differentiated.diagnostic.find("sensitivities"),
              std::string::npos);
}

TEST(LevelSetCellEvaluator, P1ReproducesCornerValuesAndReferenceGradient)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    const auto at_origin = evaluator.evaluate(0, {{0.0, 0.0, 0.0}});
    EXPECT_EQ(at_origin.interpolation_order, 1);
    EXPECT_EQ(at_origin.implicit_geometry_order, 1);
    EXPECT_NEAR(at_origin.value, -0.5, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[0], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[1], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[2], 1.0, 1.0e-12);

    const auto at_vertex_one = evaluator.evaluate(0, {{1.0, 0.0, 0.0}});
    EXPECT_NEAR(at_vertex_one.value, 0.5, 1.0e-12);
}

TEST(LevelSetCellEvaluator, P2RespondsToEdgeDofsAtInteriorNodes)
{
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 10u);

    const auto offset = system.fieldDofOffset(phi);
    solution[static_cast<std::size_t>(offset + cell_dofs[4])] = 2.0;

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    const auto at_edge_node = evaluator.evaluate(0, {{0.5, 0.0, 0.0}});
    EXPECT_EQ(at_edge_node.interpolation_order, 2);
    EXPECT_EQ(at_edge_node.implicit_geometry_order, 2);
    EXPECT_NEAR(at_edge_node.value, 2.0, 1.0e-12);

    const auto at_vertex = evaluator.evaluate(0, {{0.0, 0.0, 0.0}});
    EXPECT_NEAR(at_vertex.value, 0.0, 1.0e-12);
}

TEST(LevelSetCellEvaluator, P3RespondsToInteriorDofsWhereAvailable)
{
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/3, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_EQ(cell_dofs.size(), 16u);

    const auto offset = system.fieldDofOffset(phi);
    constexpr std::size_t first_cell_interior_dof = 12u;
    solution[static_cast<std::size_t>(
        offset + cell_dofs[first_cell_interior_dof])] = 7.0;

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    const auto at_interior = evaluator.evaluate(
        0, {{FE::Real(-1.0) / FE::Real(3.0),
             FE::Real(-1.0) / FE::Real(3.0),
             0.0}});
    EXPECT_EQ(at_interior.interpolation_order, 3);
    EXPECT_EQ(at_interior.implicit_geometry_order, 3);
    EXPECT_NEAR(at_interior.value, 7.0, 1.0e-12);

    const auto at_corner = evaluator.evaluate(0, {{-1.0, -1.0, 0.0}});
    EXPECT_NEAR(at_corner.value, 0.0, 1.0e-12);
}

TEST(LevelSetCellEvaluator, ReferenceGradientMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 10u);

    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            FE::Real(0.1) * static_cast<FE::Real>(i + 1u);
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    const std::array<FE::Real, 3> xi{{0.2, 0.2, 0.2}};
    const auto evaluation = evaluator.evaluate(0, xi);
    constexpr FE::Real eps = 1.0e-6;
    for (std::size_t d = 0; d < 3u; ++d) {
        auto plus = xi;
        auto minus = xi;
        plus[d] += eps;
        minus[d] -= eps;
        const auto value_plus = evaluator.evaluate(0, plus).value;
        const auto value_minus = evaluator.evaluate(0, minus).value;
        const auto finite_difference =
            (value_plus - value_minus) / (FE::Real{2.0} * eps);
        EXPECT_NEAR(evaluation.reference_gradient[d],
                    finite_difference,
                    1.0e-8);
    }
}

TEST(LevelSetCellEvaluator, UsesFieldOffsetInFullStateVector)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto velocity_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    (void)system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = 3,
    });
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 1000.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);
    const auto at_origin = evaluator.evaluate(0, {{0.0, 0.0, 0.0}});
    EXPECT_NEAR(at_origin.value, -0.5, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[0], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[1], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[2], 1.0, 1.0e-12);
}

TEST(LevelSetCellEvaluator, ReportsCellEvaluationFailure)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    try {
        (void)evaluator.evaluate(99, {{0.0, 0.0, 0.0}});
        FAIL() << "Expected invalid cell evaluation to fail";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("could not evaluate cell 99"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, RejectsHighOrderImplicitModeOnUnsupportedTetra)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected unsupported high-order tetrahedron backend to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("global high-order backend validation"),
                  std::string::npos);
        EXPECT_NE(message.find("backend=SayeHyperrectangle"), std::string::npos);
        EXPECT_NE(message.find("element_type=Tetra4"), std::string::npos);
        EXPECT_NE(message.find("possible_interface_order=-1"),
                  std::string::npos);
        EXPECT_NE(message.find("implicit_cut_quadrature_backend=Auto"),
                  std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, MixedElementUnsupportedBackendReportsCellAndElement)
{
    const auto mesh = std::make_shared<MixedQuadTriangleMeshAccess>();
    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/42,
        /*layout_revision=*/1,
        /*value_revision=*/2);
    request.interface_marker = 901;
    request.quadrature_order = 1;
    request.interface_quadrature_order = 1;
    request.volume_quadrature_order = 1;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_policy = "Fail";

    FE::interfaces::LevelSetInterfaceDomain domain(request);
    std::vector<std::array<FE::Real, 3>> quad_coordinates;
    mesh->getCellCoordinates(/*cell_id=*/0, quad_coordinates);
    FE::interfaces::appendLinearLevelSetCellCut2D(
        domain,
        FE::interfaces::LevelSetCellCutInput{
            .parent_cell = 0,
            .element_type = FE::ElementType::Quad4,
            .node_coordinates = quad_coordinates,
            .level_set_values = {-1.0, 1.0, 1.0, -1.0}});

    std::vector<std::array<FE::Real, 3>> triangle_coordinates;
    mesh->getCellCoordinates(/*cell_id=*/1, triangle_coordinates);
    FE::interfaces::appendLinearLevelSetCellCut2D(
        domain,
        FE::interfaces::LevelSetCellCutInput{
            .parent_cell = 1,
            .element_type = FE::ElementType::Triangle3,
            .node_coordinates = triangle_coordinates,
            .level_set_values = {-0.5, 0.5, -0.5}});

    const auto interface_rules =
        domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 2u);
    EXPECT_EQ(interface_rules[0].provenance.parent_entity, 0);
    EXPECT_EQ(interface_rules[1].provenance.parent_entity, 1);
    EXPECT_EQ(interface_rules[0].provenance.implicit_quadrature_backend,
              "LinearCorner");
    EXPECT_EQ(interface_rules[1].provenance.implicit_quadrature_backend,
              "LinearCorner");

    const auto volume_rules = domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 4u);
    EXPECT_EQ(volume_rules[0].provenance.parent_entity, 0);
    EXPECT_EQ(volume_rules[1].provenance.parent_entity, 0);
    EXPECT_EQ(volume_rules[2].provenance.parent_entity, 1);
    EXPECT_EQ(volume_rules[3].provenance.parent_entity, 1);
    EXPECT_EQ(volume_rules[0].side, FE::geometry::CutIntegrationSide::Negative);
    EXPECT_EQ(volume_rules[1].side, FE::geometry::CutIntegrationSide::Positive);
    EXPECT_EQ(volume_rules[2].side, FE::geometry::CutIntegrationSide::Negative);
    EXPECT_EQ(volume_rules[3].side, FE::geometry::CutIntegrationSide::Positive);

    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input.parent_cell = 1;
    backend_input.linearized_input.element_type = FE::ElementType::Triangle3;
    backend_input.linearized_input.node_coordinates = triangle_coordinates;
    backend_input.linearized_input.level_set_values = {-0.5, 0.5, -0.5};
    const auto& saye_backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    const auto saye_result = saye_backend.cut(/*mesh_dimension=*/2,
                                             request,
                                             backend_input);
    ASSERT_FALSE(saye_result.cut.supported);
    const std::string saye_message =
        level_set::levelSetImplicitCutBackendCellDiagnostic(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            /*cell_id=*/1,
            FE::ElementType::Triangle3,
            saye_result.cut.diagnostic);
    EXPECT_NE(saye_message.find("backend=SayeHyperrectangle"), std::string::npos);
    EXPECT_NE(saye_message.find("cell=1"), std::string::npos);
    EXPECT_NE(saye_message.find("element_type=Triangle3"), std::string::npos);
    EXPECT_NE(saye_message.find("quadrilateral"), std::string::npos);

    backend_input.linearized_input.parent_cell = 0;
    backend_input.linearized_input.element_type = FE::ElementType::Quad4;
    backend_input.linearized_input.node_coordinates = quad_coordinates;
    backend_input.linearized_input.level_set_values = {-1.0, 1.0, 1.0, -1.0};
    const auto& subcell_backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    const auto subcell_result = subcell_backend.cut(/*mesh_dimension=*/2,
                                                   request,
                                                   backend_input);
    ASSERT_FALSE(subcell_result.cut.supported);
    const std::string subcell_message =
        level_set::levelSetImplicitCutBackendCellDiagnostic(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
            /*cell_id=*/0,
            FE::ElementType::Quad4,
            subcell_result.cut.diagnostic);
    EXPECT_NE(subcell_message.find("backend=HighOrderSubcell"), std::string::npos);
    EXPECT_NE(subcell_message.find("cell=0"), std::string::npos);
    EXPECT_NE(subcell_message.find("element_type=Quad4"), std::string::npos);
    EXPECT_NE(subcell_message.find("triangular"), std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, AutoBackendDispatchesMixedQuadTriangleCells)
{
    const auto mesh = std::make_shared<MixedQuadTriangleMeshAccess>();
    auto scalar_space = std::make_shared<MixedQuadTriangleLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeMixedQuadTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 4 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 902;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::Auto);
    EXPECT_EQ(result.domain.request().implicit_quadrature_backend, "Auto");
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 2u);
    EXPECT_EQ(interface_rules[0].provenance.parent_entity, 0);
    EXPECT_EQ(interface_rules[0].provenance.implicit_quadrature_backend,
              "SayeHyperrectangle");
    EXPECT_EQ(interface_rules[0].provenance.selected_implicit_quadrature_backend,
              "SayeHyperrectangle");
    EXPECT_EQ(interface_rules[1].provenance.parent_entity, 1);
    EXPECT_EQ(interface_rules[1].provenance.implicit_quadrature_backend,
              "HighOrderSubcell");
    EXPECT_EQ(interface_rules[1].provenance.selected_implicit_quadrature_backend,
              "HighOrderSubcell");

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 4u);
    for (const auto& rule : volume_rules) {
        if (rule.provenance.parent_entity == 0) {
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "SayeHyperrectangle");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "SayeHyperrectangle");
        } else {
            EXPECT_EQ(rule.provenance.parent_entity, 1);
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "HighOrderSubcell");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "HighOrderSubcell");
        }
    }
}

TEST(LevelSetInterfaceLifecycle,
     ProductionQualificationRequirementAcceptsQualifiedSayeQuad)
{
    const auto result = buildSingleQuadEllipseCut(
        /*semi_major=*/0.75,
        /*semi_minor=*/0.50,
        /*subdivision_depth=*/5,
        /*interface_order=*/2,
        /*volume_order=*/2,
        /*interface_marker=*/914,
        /*require_production_qualified_backend=*/true);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.domain.request().required_implicit_cut_backend_qualification,
              "ProductionQualified");
    EXPECT_EQ(result.domain.request().implicit_quadrature_backend,
              "SayeHyperrectangle");
    EXPECT_EQ(result.selected_implicit_cut_quadrature_backend_counts[
                  static_cast<std::size_t>(
                      level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle)],
              1u);
}

TEST(LevelSetInterfaceLifecycle,
     ProductionQualificationRequirementRejectsExperimentalAutoCell)
{
    const auto mesh = std::make_shared<MixedQuadTriangleMeshAccess>();
    auto scalar_space = std::make_shared<MixedQuadTriangleLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeMixedQuadTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 4 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 915;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 1;
    options.require_production_qualified_implicit_cut_backend = true;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "expected production qualification guard to reject Auto on mixed quad/triangle cells";
    } catch (const std::invalid_argument& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("required=ProductionQualified"),
                  std::string::npos);
        EXPECT_NE(message.find("capability_state=Experimental"),
                  std::string::npos);
        EXPECT_NE(message.find("element_type=Triangle3"),
                  std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle,
     GlobalBackendValidationRejectsUnsupportedMixedCellsWithoutFallback)
{
    struct Case {
        level_set::ImplicitCutQuadratureBackend backend;
        const char* backend_name;
        const char* cell;
        const char* element_type;
    };

    for (const auto& test_case :
         {Case{level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
               "backend=SayeHyperrectangle",
               "cell=1",
               "element_type=Triangle3"},
          Case{level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
               "backend=HighOrderSubcell",
               "cell=0",
               "element_type=Quad4"}}) {
        try {
            (void)buildMixedQuadTriangleResult(
                /*reverse_cell_iteration=*/false,
                test_case.backend);
            FAIL() << "Expected unsupported global backend to fail validation";
        } catch (const std::invalid_argument& ex) {
            const std::string message = ex.what();
            EXPECT_NE(message.find("global high-order backend validation"),
                      std::string::npos);
            EXPECT_NE(message.find(test_case.backend_name), std::string::npos);
            EXPECT_NE(message.find(test_case.cell), std::string::npos);
            EXPECT_NE(message.find(test_case.element_type), std::string::npos);
            EXPECT_NE(message.find("fallback_policy=Fail"), std::string::npos);
            EXPECT_NE(message.find("linear_corner_fallback_supported=false"),
                      std::string::npos);
            EXPECT_NE(message.find("possible_interface_order=-1"),
                      std::string::npos);
            EXPECT_NE(message.find("possible_volume_order=-1"),
                      std::string::npos);
            EXPECT_NE(message.find("implicit_cut_quadrature_backend=Auto"),
                      std::string::npos);
            EXPECT_NE(message.find("Implicit_cut_fallback_policy=LinearCorner"),
                      std::string::npos);
        }
    }
}

TEST(LevelSetInterfaceLifecycle,
     GlobalBackendUsesExplicitLinearCornerFallbackForUnsupportedMixedCells)
{
    const auto result =
        buildMixedQuadTriangleResult(
            /*reverse_cell_iteration=*/false,
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            level_set::ImplicitCutFallbackPolicy::LinearCorner);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(result.domain.request().implicit_quadrature_backend,
              "SayeHyperrectangle");
    EXPECT_EQ(result.domain.request().implicit_fallback_policy,
              "LinearCorner");
    EXPECT_EQ(result.domain.request().implicit_fallback_status, "Used");
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 1u);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 2u);
    for (const auto& rule : interface_rules) {
        if (rule.provenance.parent_entity == 0) {
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "SayeHyperrectangle");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "SayeHyperrectangle");
            EXPECT_EQ(rule.provenance.implicit_fallback_status, "None");
        } else {
            EXPECT_EQ(rule.provenance.parent_entity, 1);
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "LinearCorner");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "LinearCorner");
            EXPECT_EQ(rule.provenance.implicit_fallback_status, "Used");
        }
    }

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    std::size_t fallback_volume_rules = 0u;
    for (const auto& rule : volume_rules) {
        if (rule.provenance.parent_entity == 1) {
            ++fallback_volume_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "LinearCorner");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "LinearCorner");
            EXPECT_EQ(rule.provenance.implicit_fallback_status, "Used");
        }
    }
    EXPECT_GT(fallback_volume_rules, 0u);
}

TEST(LevelSetInterfaceLifecycle, AutoBackendDispatchesMixedHexTetraCells)
{
    const auto mesh = std::make_shared<MixedHexTetraMeshAccess>();
    auto scalar_space = std::make_shared<MixedHexTetraLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeMixedHexTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 8 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 903;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::Auto);
    EXPECT_EQ(result.domain.request().implicit_quadrature_backend, "Auto");
    EXPECT_EQ(result.domain.request().implicit_fallback_status, "None");
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    std::size_t hex_interface_rules = 0u;
    std::size_t tetra_interface_rules = 0u;
    for (const auto& rule : interface_rules) {
        EXPECT_EQ(rule.provenance.implicit_fallback_status, "None");
        if (rule.provenance.parent_entity == 0) {
            ++hex_interface_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "SayeHyperrectangle");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "SayeHyperrectangle");
        } else {
            EXPECT_EQ(rule.provenance.parent_entity, 1);
            ++tetra_interface_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "HighOrderSubcell");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "HighOrderSubcell");
        }
    }
    EXPECT_GT(hex_interface_rules, 0u);
    EXPECT_GT(tetra_interface_rules, 0u);

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    std::size_t hex_volume_rules = 0u;
    std::size_t tetra_volume_rules = 0u;
    for (const auto& rule : volume_rules) {
        EXPECT_EQ(rule.provenance.implicit_fallback_status, "None");
        if (rule.provenance.parent_entity == 0) {
            ++hex_volume_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "SayeHyperrectangle");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "SayeHyperrectangle");
        } else {
            EXPECT_EQ(rule.provenance.parent_entity, 1);
            ++tetra_volume_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "HighOrderSubcell");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "HighOrderSubcell");
        }
    }
    EXPECT_GT(hex_volume_rules, 0u);
    EXPECT_GT(tetra_volume_rules, 0u);
}

TEST(LevelSetInterfaceLifecycle,
     AutoBackendMixedQuadTriangleRuleOrderIndependentOfCellIteration)
{
    const auto forward = buildMixedQuadTriangleAutoResult(
        /*reverse_cell_iteration=*/false);
    const auto reverse = buildMixedQuadTriangleAutoResult(
        /*reverse_cell_iteration=*/true);

    ASSERT_TRUE(forward.success) << forward.diagnostic;
    ASSERT_TRUE(reverse.success) << reverse.diagnostic;
    EXPECT_EQ(ruleSignatures(forward.domain.interfaceQuadratureRules()),
              ruleSignatures(reverse.domain.interfaceQuadratureRules()));
    EXPECT_EQ(ruleSignatures(forward.domain.volumeQuadratureRules()),
              ruleSignatures(reverse.domain.volumeQuadratureRules()));
    EXPECT_EQ(forward.summary.active_fragment_count,
              reverse.summary.active_fragment_count);
    EXPECT_EQ(forward.summary.active_volume_region_count,
              reverse.summary.active_volume_region_count);
}

TEST(LevelSetInterfaceLifecycle,
     AutoBackendMixedHexTetraRuleOrderIndependentOfCellIteration)
{
    const auto forward = buildMixedHexTetraAutoResult(
        /*reverse_cell_iteration=*/false);
    const auto reverse = buildMixedHexTetraAutoResult(
        /*reverse_cell_iteration=*/true);

    ASSERT_TRUE(forward.success) << forward.diagnostic;
    ASSERT_TRUE(reverse.success) << reverse.diagnostic;
    EXPECT_EQ(ruleSignatures(forward.domain.interfaceQuadratureRules()),
              ruleSignatures(reverse.domain.interfaceQuadratureRules()));
    EXPECT_EQ(ruleSignatures(forward.domain.volumeQuadratureRules()),
              ruleSignatures(reverse.domain.volumeQuadratureRules()));
    EXPECT_EQ(forward.summary.active_fragment_count,
              reverse.summary.active_fragment_count);
    EXPECT_EQ(forward.summary.active_volume_region_count,
              reverse.summary.active_volume_region_count);
}

TEST(LevelSetInterfaceLifecycle, AutoBackendFailsClosedForWedgeAndPyramidCells)
{
    for (const auto type : {FE::ElementType::Wedge6,
                            FE::ElementType::Pyramid5}) {
        const auto mesh = std::make_shared<SingleUnsupportedMixedCellMeshAccess>(type);
        auto scalar_space =
            FE::spaces::Space(FE::spaces::SpaceType::H1,
                              mesh,
                              /*order=*/1,
                              /*components=*/1);

        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        ASSERT_NO_THROW(
            system.setup({}, makeSingleUnsupportedMixedCellSetupInputs(type)));

        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
            const auto x = mesh->getNodeCoordinates(vertex);
            setFieldComponentValue(solution, system, phi, vertex, x[0]);
        }

        level_set::LevelSetGeneratedInterfaceOptions options{};
        options.level_set_field_name = "phi";
        options.requested_interface_marker = 904;
        options.domain_id = "water-air";
        options.geometry_mode =
            level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
        options.implicit_cut_quadrature_backend =
            level_set::ImplicitCutQuadratureBackend::Auto;
        options.interface_quadrature_order = 1;
        options.volume_quadrature_order = 1;

        level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        try {
            (void)lifecycle.build(system, options, solution);
            FAIL() << "Expected Auto backend to reject unsupported mixed cell";
        } catch (const std::invalid_argument& ex) {
            const std::string message = ex.what();
            EXPECT_NE(message.find("backend=Auto"), std::string::npos);
            EXPECT_NE(message.find("cell=0"), std::string::npos);
            EXPECT_NE(message.find(type == FE::ElementType::Wedge6
                                       ? "element_type=Wedge6"
                                       : "element_type=Pyramid5"),
                      std::string::npos);
            EXPECT_NE(message.find("cannot select"), std::string::npos);
            EXPECT_NE(message.find("possible_interface_order=-1"),
                      std::string::npos);
            EXPECT_NE(message.find("possible_volume_order=-1"),
                      std::string::npos);
        }
    }
}

TEST(LevelSetInterfaceLifecycle, HighOrderImplicitDoesNotSilentlyUseLinearFallback)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::LinearCorner;
    options.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::LinearCorner;
    options.allow_corner_linearized_geometry = true;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, std::span<const FE::Real>{});
        FAIL() << "Expected high-order implicit geometry to reject linear fallback";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("high-order implicit"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, RejectsNonlinearBackendForLinearCornerMode)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::LinearCorner;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, std::span<const FE::Real>{});
        FAIL() << "Expected nonlinear backend to be rejected for linear geometry";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("LinearCorner"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, LinearBackendDriverReportsSupportAndOrders)
{
    const auto& backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::LinearCorner);
    EXPECT_EQ(backend.kind(),
              level_set::ImplicitCutQuadratureBackend::LinearCorner);
    EXPECT_STREQ(backend.name(), "LinearCorner");
    EXPECT_TRUE(backend.supports(2, FE::ElementType::Quad4));
    EXPECT_TRUE(backend.supports(3, FE::ElementType::Tetra4));
    EXPECT_FALSE(backend.supports(3, FE::ElementType::Hex8));

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "test-level-set", 0, 1);
    request.interface_marker = 10;
    request.quadrature_order = 3;
    request.interface_quadrature_order = 2;
    request.volume_quadrature_order = 3;

    EXPECT_EQ(backend.achievedInterfaceQuadratureOrder(
                  2, FE::ElementType::Quad4, request),
              2);
    EXPECT_EQ(backend.achievedVolumeQuadratureOrder(
                  2, FE::ElementType::Quad4, request),
              3);
    EXPECT_EQ(backend.achievedInterfaceQuadratureOrder(
                  3, FE::ElementType::Tetra4, request),
              1);
    EXPECT_EQ(backend.achievedVolumeQuadratureOrder(
                  3, FE::ElementType::Tetra4, request),
              2);
    EXPECT_EQ(backend.achievedInterfaceQuadratureOrder(
                  3, FE::ElementType::Hex8, request),
              -1);

    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input.parent_cell = 0;
    backend_input.linearized_input.element_type = FE::ElementType::Quad4;
    backend_input.linearized_input.node_coordinates = {
        std::array<FE::Real, 3>{0.0, 0.0, 0.0},
        std::array<FE::Real, 3>{1.0, 0.0, 0.0},
        std::array<FE::Real, 3>{1.0, 1.0, 0.0},
        std::array<FE::Real, 3>{0.0, 1.0, 0.0},
    };
    backend_input.linearized_input.level_set_values = {-0.5, 0.5, 1.5, 0.5};

    const auto result = backend.cut(/*mesh_dimension=*/2, request, backend_input);
    ASSERT_TRUE(result.cut.supported) << result.cut.diagnostic;
    EXPECT_EQ(result.requested_interface_quadrature_order, 2);
    EXPECT_EQ(result.requested_volume_quadrature_order, 3);
    EXPECT_EQ(result.possible_interface_quadrature_order, 2);
    EXPECT_EQ(result.possible_volume_quadrature_order, 3);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 2);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 3);
    EXPECT_EQ(result.verified_interface_quadrature_order, 2);
    EXPECT_EQ(result.verified_volume_quadrature_order, 3);
    EXPECT_GT(result.volume_quadrature_point_count, 0u);
    EXPECT_GT(result.interface_quadrature_point_count, 0u);
    EXPECT_GE(result.backend_elapsed_seconds, 0.0);
}

TEST(LevelSetInterfaceLifecycle, AutoBackendDriverReportsDispatchSupport)
{
    const auto& backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::Auto);
    EXPECT_EQ(backend.kind(),
              level_set::ImplicitCutQuadratureBackend::Auto);
    EXPECT_STREQ(backend.name(), "Auto");
    EXPECT_TRUE(backend.supports(2, FE::ElementType::Quad4));
    EXPECT_TRUE(backend.supports(2, FE::ElementType::Triangle3));
    EXPECT_TRUE(backend.supports(3, FE::ElementType::Hex8));
    EXPECT_TRUE(backend.supports(3, FE::ElementType::Tetra4));
    EXPECT_FALSE(backend.supports(3, FE::ElementType::Wedge6));
    EXPECT_FALSE(backend.supports(3, FE::ElementType::Pyramid5));

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "auto-level-set", 0, 1);
    request.interface_marker = 11;
    request.interface_quadrature_order = 3;
    request.volume_quadrature_order = 3;

    level_set::ImplicitCutQuadratureBackendCellInput input{};
    input.linearized_input.parent_cell = 0;
    input.linearized_input.element_type = FE::ElementType::Wedge6;
    const auto result = backend.cut(/*mesh_dimension=*/3, request, input);
    EXPECT_FALSE(result.cut.supported);
    EXPECT_EQ(result.selected_backend, level_set::ImplicitCutQuadratureBackend::Auto);
    EXPECT_EQ(result.requested_interface_quadrature_order, 3);
    EXPECT_EQ(result.requested_volume_quadrature_order, 3);
    EXPECT_EQ(result.possible_interface_quadrature_order, -1);
    EXPECT_EQ(result.possible_volume_quadrature_order, -1);
    EXPECT_EQ(result.achieved_interface_quadrature_order, -1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, -1);
    EXPECT_EQ(result.verified_interface_quadrature_order, -1);
    EXPECT_EQ(result.verified_volume_quadrature_order, -1);
    EXPECT_EQ(result.volume_quadrature_point_count, 0u);
    EXPECT_EQ(result.interface_quadrature_point_count, 0u);
    EXPECT_GE(result.backend_elapsed_seconds, 0.0);
    EXPECT_NE(result.cut.diagnostic.find("possible_interface_order=-1"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("volume_quadrature_points=0"),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, BackendCapabilityReportsMilestoneContract)
{
    const auto linear_quad =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::LinearCorner,
            2,
            FE::ElementType::Quad4);
    EXPECT_TRUE(linear_quad.implemented);
    EXPECT_TRUE(linear_quad.supports_element_type);
    EXPECT_FALSE(linear_quad.supports_high_order_geometry);
    EXPECT_TRUE(linear_quad.requires_scalar_h1_c0_level_set);
    EXPECT_EQ(linear_quad.minimum_level_set_order, 1);
    EXPECT_EQ(linear_quad.validation_level_set_order, 1);
    EXPECT_EQ(linear_quad.maximum_reported_interface_order, 5);
    EXPECT_EQ(linear_quad.maximum_reported_volume_order, 5);
    EXPECT_TRUE(linear_quad.returns_reference_frame_rules);
    EXPECT_TRUE(linear_quad.requires_positive_volume_weights);
    EXPECT_TRUE(linear_quad.requires_deterministic_rule_order);
    EXPECT_TRUE(linear_quad.prunes_tiny_slivers_in_context);
    EXPECT_TRUE(linear_quad.near_tangent_requires_diagnostic);
    EXPECT_GT(linear_quad.tiny_sliver_volume_fraction, 0.0);

    const auto linear_tet =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::LinearCorner,
            3,
            FE::ElementType::Tetra4);
    EXPECT_TRUE(linear_tet.supports_element_type);
    EXPECT_EQ(linear_tet.maximum_reported_interface_order, 1);
    EXPECT_EQ(linear_tet.maximum_reported_volume_order, 2);

    const auto linear_hex =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::LinearCorner,
            3,
            FE::ElementType::Hex8);
    EXPECT_FALSE(linear_hex.supports_element_type);
    EXPECT_EQ(linear_hex.maximum_reported_interface_order, -1);
    EXPECT_EQ(linear_hex.maximum_reported_volume_order, -1);

    const auto saye_quad =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            2,
            FE::ElementType::Quad9);
    EXPECT_TRUE(saye_quad.implemented);
    EXPECT_TRUE(saye_quad.supports_element_type);
    EXPECT_TRUE(saye_quad.supports_high_order_geometry);
    EXPECT_EQ(saye_quad.minimum_level_set_order, 1);
    EXPECT_EQ(saye_quad.validation_level_set_order, 3);
    EXPECT_EQ(saye_quad.qualification,
              level_set::ImplicitCutQuadratureBackendQualification::ProductionQualified);
    EXPECT_NE(saye_quad.qualification_diagnostic.find("2D quadrilateral"),
              std::string::npos);
    EXPECT_EQ(saye_quad.maximum_reported_interface_order, 5);
    EXPECT_EQ(saye_quad.maximum_reported_volume_order, 5);
    EXPECT_TRUE(saye_quad.requires_scalar_h1_c0_level_set);
    EXPECT_TRUE(saye_quad.prunes_tiny_slivers_in_context);
    EXPECT_GT(saye_quad.tiny_sliver_volume_fraction, 0.0);

    const auto saye_tri =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            2,
            FE::ElementType::Triangle6);
    EXPECT_FALSE(saye_tri.supports_element_type);
    EXPECT_EQ(saye_tri.maximum_reported_interface_order, -1);
    EXPECT_EQ(saye_tri.maximum_reported_volume_order, -1);

    const auto subcell_tri =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
            2,
            FE::ElementType::Triangle6);
    EXPECT_TRUE(subcell_tri.implemented);
    EXPECT_TRUE(subcell_tri.supports_element_type);
    EXPECT_TRUE(subcell_tri.supports_high_order_geometry);
    EXPECT_EQ(subcell_tri.minimum_level_set_order, 1);
    EXPECT_EQ(subcell_tri.validation_level_set_order, 3);
    EXPECT_EQ(subcell_tri.maximum_reported_interface_order, 5);
    EXPECT_EQ(subcell_tri.maximum_reported_volume_order, 5);
    EXPECT_TRUE(subcell_tri.requires_scalar_h1_c0_level_set);
    EXPECT_TRUE(subcell_tri.prunes_tiny_slivers_in_context);
    EXPECT_GT(subcell_tri.tiny_sliver_volume_fraction, 0.0);

    const auto subcell_tet =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
            3,
            FE::ElementType::Tetra10);
    EXPECT_TRUE(subcell_tet.implemented);
    EXPECT_TRUE(subcell_tet.supports_element_type);
    EXPECT_TRUE(subcell_tet.supports_high_order_geometry);
    EXPECT_EQ(subcell_tet.minimum_level_set_order, 1);
    EXPECT_EQ(subcell_tet.validation_level_set_order, 3);
    EXPECT_EQ(subcell_tet.maximum_reported_interface_order, 2);
    EXPECT_EQ(subcell_tet.maximum_reported_volume_order, 2);
    EXPECT_TRUE(subcell_tet.requires_scalar_h1_c0_level_set);
    EXPECT_TRUE(subcell_tet.prunes_tiny_slivers_in_context);
    EXPECT_GT(subcell_tet.tiny_sliver_volume_fraction, 0.0);

    const auto subcell_quad =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
            2,
            FE::ElementType::Quad9);
    EXPECT_FALSE(subcell_quad.supports_element_type);
    EXPECT_EQ(subcell_quad.maximum_reported_interface_order, -1);
    EXPECT_EQ(subcell_quad.maximum_reported_volume_order, -1);

    const auto saye_hex =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            3,
            FE::ElementType::Hex27);
    EXPECT_TRUE(saye_hex.implemented);
    EXPECT_TRUE(saye_hex.supports_element_type);
    EXPECT_TRUE(saye_hex.supports_high_order_geometry);
    EXPECT_EQ(saye_hex.qualification,
              level_set::ImplicitCutQuadratureBackendQualification::Experimental);
    EXPECT_EQ(saye_hex.maximum_reported_interface_order, 1);
    EXPECT_EQ(saye_hex.maximum_reported_volume_order, 2);
    EXPECT_TRUE(saye_hex.prunes_tiny_slivers_in_context);
    EXPECT_GT(saye_hex.tiny_sliver_volume_fraction, 0.0);

    const auto auto_quad =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::Auto,
            2,
            FE::ElementType::Quad9);
    EXPECT_TRUE(auto_quad.implemented);
    EXPECT_TRUE(auto_quad.supports_element_type);
    EXPECT_TRUE(auto_quad.supports_high_order_geometry);
    EXPECT_EQ(auto_quad.qualification,
              level_set::ImplicitCutQuadratureBackendQualification::ProductionQualified);
    EXPECT_EQ(auto_quad.maximum_reported_interface_order, 5);
    EXPECT_EQ(auto_quad.maximum_reported_volume_order, 5);

    const auto auto_tri =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::Auto,
            2,
            FE::ElementType::Triangle6);
    EXPECT_TRUE(auto_tri.implemented);
    EXPECT_TRUE(auto_tri.supports_element_type);
    EXPECT_TRUE(auto_tri.supports_high_order_geometry);
    EXPECT_EQ(auto_tri.qualification,
              level_set::ImplicitCutQuadratureBackendQualification::Experimental);

    const auto auto_wedge =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::Auto,
            3,
            FE::ElementType::Wedge6);
    EXPECT_TRUE(auto_wedge.implemented);
    EXPECT_FALSE(auto_wedge.supports_element_type);
    EXPECT_EQ(auto_wedge.maximum_reported_interface_order, -1);
    EXPECT_EQ(auto_wedge.maximum_reported_volume_order, -1);

    const auto moment_fit =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::MomentFit,
            2,
            FE::ElementType::Quad9);
    EXPECT_FALSE(moment_fit.implemented);
    EXPECT_FALSE(moment_fit.supports_element_type);
    EXPECT_TRUE(moment_fit.supports_high_order_geometry);
    EXPECT_EQ(moment_fit.qualification,
              level_set::ImplicitCutQuadratureBackendQualification::Unavailable);
    EXPECT_NE(moment_fit.qualification_diagnostic.find("MomentFit"),
              std::string::npos);
    EXPECT_NE(moment_fit.qualification_diagnostic.find("unavailable"),
              std::string::npos);

    FE::interfaces::CutInterfaceDomainRequest high_order_request{};
    high_order_request.source =
        FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
            "capability-order-level-set", 0, 1);
    high_order_request.interface_marker = 907;
    high_order_request.interface_quadrature_order = 5;
    high_order_request.volume_quadrature_order = 5;

    const auto& saye_backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(saye_backend.achievedInterfaceQuadratureOrder(
                  2, FE::ElementType::Quad9, high_order_request),
              saye_quad.maximum_reported_interface_order);
    EXPECT_EQ(saye_backend.achievedVolumeQuadratureOrder(
                  2, FE::ElementType::Quad9, high_order_request),
              saye_quad.maximum_reported_volume_order);
    EXPECT_EQ(saye_backend.achievedInterfaceQuadratureOrder(
                  2, FE::ElementType::Triangle6, high_order_request),
              -1);

    const auto& subcell_backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(subcell_backend.achievedInterfaceQuadratureOrder(
                  3, FE::ElementType::Tetra10, high_order_request),
              subcell_tet.maximum_reported_interface_order);
    EXPECT_EQ(subcell_backend.achievedVolumeQuadratureOrder(
                  3, FE::ElementType::Tetra10, high_order_request),
              subcell_tet.maximum_reported_volume_order);

    const auto& auto_backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::Auto);
    EXPECT_EQ(auto_backend.achievedInterfaceQuadratureOrder(
                  2, FE::ElementType::Quad9, high_order_request),
              auto_quad.maximum_reported_interface_order);
    EXPECT_EQ(auto_backend.achievedVolumeQuadratureOrder(
                  2, FE::ElementType::Quad9, high_order_request),
              auto_quad.maximum_reported_volume_order);
    EXPECT_EQ(auto_backend.achievedInterfaceQuadratureOrder(
                  3, FE::ElementType::Wedge6, high_order_request),
              -1);
}

TEST(LevelSetInterfaceLifecycle, BackendCapabilityCoversAllElementFamilies)
{
    struct Expectation {
        FE::ElementType type{FE::ElementType::Unknown};
        bool linear_2d{false};
        bool linear_3d{false};
        bool saye_2d{false};
        bool saye_3d{false};
        bool subcell_2d{false};
        bool subcell_3d{false};
    };

    const std::array<Expectation, 20> expectations{{
        {FE::ElementType::Line2},
        {FE::ElementType::Triangle3, true, false, false, false, true, false},
        {FE::ElementType::Quad4, true, false, true, false, false, false},
        {FE::ElementType::Tetra4, false, true, false, false, false, true},
        {FE::ElementType::Hex8, false, false, false, true, false, false},
        {FE::ElementType::Wedge6},
        {FE::ElementType::Pyramid5},
        {FE::ElementType::Line3},
        {FE::ElementType::Triangle6, true, false, false, false, true, false},
        {FE::ElementType::Quad9, true, false, true, false, false, false},
        {FE::ElementType::Quad8, true, false, true, false, false, false},
        {FE::ElementType::Tetra10, false, true, false, false, false, true},
        {FE::ElementType::Hex27, false, false, false, true, false, false},
        {FE::ElementType::Hex20, false, false, false, true, false, false},
        {FE::ElementType::Wedge15},
        {FE::ElementType::Wedge18},
        {FE::ElementType::Pyramid13},
        {FE::ElementType::Pyramid14},
        {FE::ElementType::Point1},
        {FE::ElementType::Unknown},
    }};

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source =
        FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
            "full-capability-matrix-level-set", 0, 1);
    request.interface_marker = 912;
    request.interface_quadrature_order = 5;
    request.volume_quadrature_order = 5;

    const auto check_backend =
        [&](level_set::ImplicitCutQuadratureBackend backend,
            int mesh_dimension,
            FE::ElementType type,
            bool supported,
            int max_interface_order,
            int max_volume_order) {
            SCOPED_TRACE(static_cast<int>(backend));
            SCOPED_TRACE(mesh_dimension);
            SCOPED_TRACE(static_cast<int>(type));
            const auto capability =
                level_set::implicitCutQuadratureBackendCapability(
                    backend, mesh_dimension, type);
            const auto& driver =
                level_set::implicitCutQuadratureBackendDriver(backend);
            EXPECT_EQ(driver.supports(mesh_dimension, type), supported);
            EXPECT_EQ(capability.supports_element_type, supported);
            EXPECT_EQ(capability.maximum_reported_interface_order,
                      supported ? max_interface_order : -1);
            EXPECT_EQ(capability.maximum_reported_volume_order,
                      supported ? max_volume_order : -1);
            EXPECT_EQ(driver.achievedInterfaceQuadratureOrder(
                          mesh_dimension, type, request),
                      supported ? max_interface_order : -1);
            EXPECT_EQ(driver.achievedVolumeQuadratureOrder(
                          mesh_dimension, type, request),
                      supported ? max_volume_order : -1);
        };

    for (const auto& expected : expectations) {
        check_backend(level_set::ImplicitCutQuadratureBackend::LinearCorner,
                      2,
                      expected.type,
                      expected.linear_2d,
                      5,
                      5);
        check_backend(level_set::ImplicitCutQuadratureBackend::LinearCorner,
                      3,
                      expected.type,
                      expected.linear_3d,
                      1,
                      2);
        check_backend(level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
                      2,
                      expected.type,
                      expected.saye_2d,
                      5,
                      5);
        check_backend(level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
                      3,
                      expected.type,
                      expected.saye_3d,
                      1,
                      2);
        check_backend(level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
                      2,
                      expected.type,
                      expected.subcell_2d,
                      5,
                      5);
        check_backend(level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
                      3,
                      expected.type,
                      expected.subcell_3d,
                      2,
                      2);

        const bool auto_2d = expected.saye_2d || expected.subcell_2d;
        const bool auto_3d = expected.saye_3d || expected.subcell_3d;
        check_backend(level_set::ImplicitCutQuadratureBackend::Auto,
                      2,
                      expected.type,
                      auto_2d,
                      5,
                      5);
        check_backend(level_set::ImplicitCutQuadratureBackend::Auto,
                      3,
                      expected.type,
                      auto_3d,
                      expected.saye_3d ? 1 : 2,
                      2);

        check_backend(level_set::ImplicitCutQuadratureBackend::MomentFit,
                      2,
                      expected.type,
                      false,
                      -1,
                      -1);
        check_backend(level_set::ImplicitCutQuadratureBackend::MomentFit,
                      3,
                      expected.type,
                      false,
                      -1,
                      -1);
    }
}

TEST(LevelSetInterfaceLifecycle, BackendDiagnosticStatusNamesAreStable)
{
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::ExactNoCut),
        "ExactNoCut");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Cut),
        "Cut");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Tangent),
        "Tangent");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Degenerate),
        "Degenerate");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Fallback),
        "Fallback");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Unsupported),
        "Unsupported");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Failed),
        "Failed");
}

TEST(LevelSetInterfaceLifecycle, BackendDiagnosticIdentifiesFallbackCell)
{
    const std::string diagnostic =
        level_set::levelSetImplicitCutBackendCellDiagnostic(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            /*cell_id=*/42,
            FE::ElementType::Quad9,
            "fallback_used=true; status=Fallback; max_depth_reached=7");

    EXPECT_NE(diagnostic.find("backend=SayeHyperrectangle"), std::string::npos);
    EXPECT_NE(diagnostic.find("cell=42"), std::string::npos);
    EXPECT_NE(diagnostic.find("element_type=Quad9"), std::string::npos);
    EXPECT_NE(diagnostic.find("fallback_used=true"), std::string::npos);
    EXPECT_NE(diagnostic.find("status=Fallback"), std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, RecursiveBackendReportsStructuredMilestoneCounters)
{
    constexpr FE::Real radius = 0.5;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "diagnostic-level-set", 0, 1);
    request.interface_marker = 904;
    request.quadrature_order = 3;
    request.interface_quadrature_order = 3;
    request.volume_quadrature_order = 6;
    request.implicit_cut_root_coordinate_tolerance = 5.0e-13;
    request.implicit_cut_root_max_iterations = 64;
    request.implicit_cut_max_subdivision_depth = 2;

    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input.parent_cell = 0;
    backend_input.linearized_input.element_type = FE::ElementType::Quad9;
    backend_input.linearized_input.node_coordinates = {
        mesh->getNodeCoordinates(0),
        mesh->getNodeCoordinates(1),
        mesh->getNodeCoordinates(2),
        mesh->getNodeCoordinates(3),
    };
    for (const auto& x : backend_input.linearized_input.node_coordinates) {
        backend_input.linearized_input.level_set_values.push_back(
            x[0] * x[0] + x[1] * x[1] - radius * radius);
    }
    backend_input.evaluator = &evaluator;
    backend_input.reference_min = {{-1.0, -1.0, 0.0}};
    backend_input.reference_max = {{1.0, 1.0, 0.0}};

    const auto& backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    const auto result = backend.cut(/*mesh_dimension=*/2, request, backend_input);

    ASSERT_TRUE(result.cut.supported) << result.cut.diagnostic;
    EXPECT_EQ(result.selected_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(result.requested_interface_quadrature_order, 3);
    EXPECT_EQ(result.requested_volume_quadrature_order, 6);
    EXPECT_EQ(result.possible_interface_quadrature_order, 3);
    EXPECT_EQ(result.possible_volume_quadrature_order, 5);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 3);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 5);
    EXPECT_EQ(result.verified_interface_quadrature_order, 3);
    EXPECT_EQ(result.verified_volume_quadrature_order, 5);
    EXPECT_TRUE(result.fallback_used);
    EXPECT_TRUE(result.requested_high_order_downgrade);
    EXPECT_EQ(result.diagnostic_status,
              level_set::ImplicitCutQuadratureDiagnosticStatus::Fallback);
    EXPECT_GT(result.subdivision_count, 0);
    EXPECT_GT(result.max_subdivision_depth_reached, 0);
    EXPECT_GT(result.linearized_leaf_count, 0);
    EXPECT_GT(result.interface_fragment_count, 0);
    EXPECT_GT(result.root_branch_count, 0);
    EXPECT_GT(result.root_finder_iteration_count, 0);
    EXPECT_GT(result.curved_fragment_count, 0);
    EXPECT_GT(result.volume_quadrature_point_count, 0u);
    EXPECT_GT(result.interface_quadrature_point_count, 0u);
    EXPECT_GE(result.backend_elapsed_seconds, 0.0);
    EXPECT_NE(result.fallback_reason.find("downgraded"), std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("root_branches="), std::string::npos);
    EXPECT_EQ(result.cut.diagnostic.find("root_branches=0"), std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("root_finder_iterations="),
              std::string::npos);
    EXPECT_EQ(result.cut.diagnostic.find("root_finder_iterations=0"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("root_coordinate_tolerance="),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("root_max_iterations=64"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("curved_fragments="),
              std::string::npos);
    EXPECT_EQ(result.cut.diagnostic.find("curved_fragments=0"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("requested_interface_order=3"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("requested_volume_order=6"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("possible_interface_order=3"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("possible_volume_order=5"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("verified_interface_order=3"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("verified_volume_order=5"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("fallback_used=true"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("high_order_downgrade=true"),
              std::string::npos);
    EXPECT_EQ(result.cut.diagnostic.find("root_polishing=not_performed"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("status=Fallback"), std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("fallback_reason="), std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("volume_quadrature_points="),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("interface_quadrature_points="),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("backend_elapsed_seconds="),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, HighOrderImplicitFallbackPolicyFailsClosed)
{
    try {
        (void)buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                       /*level_set_order=*/2,
                                       /*subdivision_depth=*/2,
                                       /*interface_order=*/3,
                                       /*volume_order=*/6,
                                       /*interface_marker=*/1896);
        FAIL() << "Expected fallback policy Fail to reject downgraded high-order cut";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("fallback policy Fail"), std::string::npos);
        EXPECT_NE(message.find("fallback_reason="), std::string::npos);
        EXPECT_NE(message.find("downgraded"), std::string::npos);
        EXPECT_NE(message.find("high_order_downgrade=true"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, HighOrderImplicitExplicitFallbackPolicyCountsCell)
{
    const auto result =
        buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                 /*level_set_order=*/2,
                                 /*subdivision_depth=*/2,
                                 /*interface_order=*/3,
                                 /*volume_order=*/6,
                                 /*interface_marker=*/1897,
                                 /*radius=*/0.5,
                                 level_set::ImplicitCutFallbackPolicy::LinearCorner);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 1u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 3);
    EXPECT_NE(result.diagnostic.find("high_order_downgrade=true"),
              std::string::npos);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 5);
    EXPECT_EQ(result.domain.request().implicit_fallback_policy, "LinearCorner");
    EXPECT_EQ(result.domain.request().implicit_fallback_status, "Used");
    EXPECT_NE(result.diagnostic.find("fallback_used=true"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("status=Fallback"), std::string::npos);
    EXPECT_GT(result.backend_volume_quadrature_point_count, 0u);
    EXPECT_GT(result.backend_interface_quadrature_point_count, 0u);
    EXPECT_GE(result.backend_elapsed_seconds, 0.0);
    EXPECT_NE(result.diagnostic.find("volume_quadrature_points="),
              std::string::npos);
    EXPECT_NE(result.diagnostic.find("backend_elapsed_seconds="),
              std::string::npos);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    EXPECT_EQ(interface_rules.front().provenance.implicit_fallback_status,
              "Used");
    EXPECT_EQ(interface_rules.front().provenance.implicit_cut_root_tolerance,
              result.domain.request().implicit_cut_root_tolerance);
    EXPECT_EQ(interface_rules.front().provenance.implicit_cut_root_coordinate_tolerance,
              result.domain.request().implicit_cut_root_coordinate_tolerance);
    EXPECT_EQ(interface_rules.front().provenance.implicit_cut_root_max_iterations,
              result.domain.request().implicit_cut_root_max_iterations);
}

TEST(LevelSetInterfaceLifecycle,
     HighOrderFullCellEquivalentVolumeRulesAreCompact)
{
    const auto result =
        buildSingleTetraHighOrderSubcellCut(
            [](const std::array<FE::Real, 3>&) { return FE::Real{-1.0}; },
            /*interface_marker=*/1898,
            /*subdivision_depth=*/3,
            /*interface_order=*/1,
            /*volume_order=*/2);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.summary.active_fragment_count, 0u);
    EXPECT_EQ(result.summary.active_volume_region_count, 1u);
    EXPECT_EQ(result.backend_volume_quadrature_point_count, 1u);
    ASSERT_EQ(result.domain.volumeRegions().size(), 1u);
    const auto& region = result.domain.volumeRegions().front();
    EXPECT_TRUE(region.full_cell_equivalent);
    EXPECT_TRUE(region.quadrature_points.empty());
    EXPECT_EQ(region.achieved_quadrature_order, 2);

    const auto rules = result.domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_TRUE(rules.front().full_cell_equivalent);
    EXPECT_EQ(rules.front().exact_polynomial_order, 2);
    ASSERT_EQ(rules.front().points.size(), 1u);
    EXPECT_NEAR(rules.front().points.front().weight,
                rules.front().measure,
                FE::Real{1.0e-14});
}

TEST(LevelSetInterfaceLifecycle, LinearBackendOutputPassesCommonValidation)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "validation-level-set", 0, 1);
    request.interface_marker = 99;
    request.tolerance = 1.0e-12;
    request.quadrature_order = 2;
    request.interface_quadrature_order = 1;
    request.volume_quadrature_order = 2;

    FE::interfaces::LevelSetCellCutInput input{};
    input.parent_cell = 0;
    input.element_type = FE::ElementType::Tetra4;
    input.node_coordinates = {
        std::array<FE::Real, 3>{0.0, 0.0, 0.0},
        std::array<FE::Real, 3>{1.0, 0.0, 0.0},
        std::array<FE::Real, 3>{0.0, 1.0, 0.0},
        std::array<FE::Real, 3>{0.0, 0.0, 1.0},
    };
    input.level_set_values = {-0.25, 0.75, 0.75, 0.75};

    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;
    backend_input.isovalue = request.isovalue;
    backend_input.reference_min = {{0.0, 0.0, 0.0}};
    backend_input.reference_max = {{1.0, 1.0, 1.0}};

    const auto& backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::LinearCorner);
    const auto result = backend.cut(3, request, backend_input);
    const auto validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, result);
    EXPECT_TRUE(validation.ok) << validation.diagnostic;
    EXPECT_EQ(validation.status,
              level_set::ImplicitCutQuadratureDiagnosticStatus::Cut);

    FE::Real parent_measure = 0.0;
    FE::Real side_measure = 0.0;
    for (const auto& region : result.cut.volume_regions) {
        parent_measure = std::max(parent_measure, region.parent_measure);
        side_measure += region.measure;
    }
    EXPECT_NEAR(side_measure, parent_measure, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, InvalidBackendOutputIsRejected)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "validation-level-set", 0, 1);
    request.interface_marker = 12;
    request.tolerance = 1.0e-12;

    FE::interfaces::LevelSetCellCutInput input{};
    input.parent_cell = 2;
    input.element_type = FE::ElementType::Tetra4;
    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;

    level_set::ImplicitCutQuadratureBackendCellResult result{};
    result.cut.supported = true;
    result.achieved_interface_quadrature_order = 1;
    result.achieved_volume_quadrature_order = 1;
    result.diagnostic_status =
        level_set::ImplicitCutQuadratureDiagnosticStatus::Cut;

    FE::interfaces::CutInterfaceVolumeRegion bad_region{};
    bad_region.interface_marker = request.interface_marker;
    bad_region.parent_cell = input.parent_cell;
    bad_region.side = FE::geometry::CutIntegrationSide::Negative;
    bad_region.parent_measure = 1.0;
    bad_region.measure = std::numeric_limits<FE::Real>::quiet_NaN();
    bad_region.volume_fraction = 0.5;
    result.cut.volume_regions.push_back(bad_region);

    const auto validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, result);
    EXPECT_FALSE(validation.ok);
    EXPECT_NE(validation.diagnostic.find("invalid volume region"),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, BackendValidationRejectsInconsistentInterfaceNormals)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2]);
    }
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "validation-level-set", 0, 1);
    request.interface_marker = 13;
    request.tolerance = 1.0e-12;

    FE::interfaces::LevelSetCellCutInput input{};
    input.parent_cell = 0;
    input.element_type = FE::ElementType::Tetra4;
    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;

    level_set::ImplicitCutQuadratureBackendCellResult result{};
    result.cut.supported = true;
    result.achieved_interface_quadrature_order = 1;
    result.achieved_volume_quadrature_order = 1;
    result.diagnostic_status =
        level_set::ImplicitCutQuadratureDiagnosticStatus::Cut;

    FE::interfaces::CutInterfaceFragment fragment{};
    fragment.interface_marker = request.interface_marker;
    fragment.parent_cell = input.parent_cell;
    fragment.measure = 1.0;
    fragment.quadrature_points.push_back(
        FE::interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.25, 0.25, 0.25}},
            .parent_coordinate = {{0.25, 0.25, 0.25}},
            .normal = {{-1.0, -1.0, -1.0}},
            .weight = 1.0});
    result.cut.fragments.push_back(fragment);

    const auto validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, result);
    EXPECT_FALSE(validation.ok);
    EXPECT_NE(validation.diagnostic.find("inconsistent interface quadrature normal"),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, BackendValidationRejectsInconsistentQuadratureWeights)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2]);
    }
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "validation-level-set", 0, 1);
    request.interface_marker = 14;
    request.tolerance = 1.0e-12;

    FE::interfaces::LevelSetCellCutInput input{};
    input.parent_cell = 0;
    input.element_type = FE::ElementType::Tetra4;
    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;

    level_set::ImplicitCutQuadratureBackendCellResult interface_result{};
    interface_result.cut.supported = true;
    interface_result.achieved_interface_quadrature_order = 2;
    interface_result.achieved_volume_quadrature_order = 2;
    interface_result.diagnostic_status =
        level_set::ImplicitCutQuadratureDiagnosticStatus::Cut;

    FE::interfaces::CutInterfaceFragment fragment{};
    fragment.interface_marker = request.interface_marker;
    fragment.parent_cell = input.parent_cell;
    fragment.kind = FE::interfaces::CutInterfaceFragmentKind::CurvedPatch;
    fragment.measure = 1.0;
    fragment.quadrature_points.push_back(
        FE::interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.25, 0.25, 0.25}},
            .parent_coordinate = {{0.25, 0.25, 0.25}},
            .normal = {{1.0, 1.0, 1.0}},
            .weight = 0.25,
            .reference_measure_factor = 0.25,
            .level_set_residual = 0.0,
            .gradient_norm = 1.0});
    interface_result.cut.fragments.push_back(fragment);

    const auto interface_validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, interface_result);
    EXPECT_FALSE(interface_validation.ok);
    EXPECT_NE(interface_validation.diagnostic.find(
                  "interface quadrature weights do not sum"),
              std::string::npos);

    level_set::ImplicitCutQuadratureBackendCellResult volume_result{};
    volume_result.cut.supported = true;
    volume_result.achieved_interface_quadrature_order = 2;
    volume_result.achieved_volume_quadrature_order = 2;
    volume_result.diagnostic_status =
        level_set::ImplicitCutQuadratureDiagnosticStatus::Cut;

    FE::interfaces::CutInterfaceVolumeRegion region{};
    region.interface_marker = request.interface_marker;
    region.parent_cell = input.parent_cell;
    region.side = FE::geometry::CutIntegrationSide::Negative;
    region.parent_measure = 2.0;
    region.measure = 1.0;
    region.volume_fraction = 0.5;
    region.quadrature_points.push_back(FE::geometry::CutQuadraturePoint{
        .point = {{0.25, 0.25, 0.25}},
        .normal = {{1.0, 0.0, 0.0}},
        .weight = 0.25,
        .parent_coordinate = {{0.25, 0.25, 0.25}},
        .reference_measure_factor = 0.25,
        .level_set_residual = 0.0,
        .gradient_norm = 1.0});
    volume_result.cut.volume_regions.push_back(region);

    const auto volume_validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, volume_result);
    EXPECT_FALSE(volume_validation.ok);
    EXPECT_NE(volume_validation.diagnostic.find(
                  "volume quadrature weights do not sum"),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, BackendValidationRejectsBadRootPolishMetadata)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2]);
    }
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "validation-level-set", 0, 1);
    request.interface_marker = 15;
    request.tolerance = 1.0e-12;
    request.implicit_cut_root_tolerance = 1.0e-10;

    FE::interfaces::LevelSetCellCutInput input{};
    input.parent_cell = 0;
    input.element_type = FE::ElementType::Tetra4;
    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;

    level_set::ImplicitCutQuadratureBackendCellResult result{};
    result.cut.supported = true;
    result.achieved_interface_quadrature_order = 2;
    result.achieved_volume_quadrature_order = 2;
    result.diagnostic_status =
        level_set::ImplicitCutQuadratureDiagnosticStatus::Cut;

    FE::interfaces::CutInterfaceFragment fragment{};
    fragment.interface_marker = request.interface_marker;
    fragment.parent_cell = input.parent_cell;
    fragment.kind = FE::interfaces::CutInterfaceFragmentKind::CurvedPatch;
    fragment.measure = 1.0;
    fragment.root_polished = true;
    fragment.max_root_residual = 1.0e-5;
    fragment.min_gradient_norm = 1.0;
    fragment.quadrature_points.push_back(
        FE::interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.25, 0.25, 0.25}},
            .parent_coordinate = {{0.25, 0.25, 0.25}},
            .normal = {{1.0, 1.0, 1.0}},
            .weight = 1.0,
            .reference_measure_factor = 1.0,
            .level_set_residual = 1.0e-5,
            .gradient_norm = 1.0});
    result.cut.fragments.push_back(fragment);

    const auto validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, result);
    EXPECT_FALSE(validation.ok);
    EXPECT_NE(validation.diagnostic.find("root-polished interface"),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, BackendMetadataReachesCutIntegrationContext)
{
    constexpr int interface_marker = 83;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.interface_quadrature_order = 0;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;

    FE::assembly::CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(result.domain);

    ASSERT_FALSE(context.volumeRules().empty());
    EXPECT_EQ(context.volumeRules().front().provenance.implicit_geometry_mode,
              "LinearCorner");
    EXPECT_EQ(context.volumeRules().front().provenance.implicit_quadrature_backend,
              "LinearCorner");
    EXPECT_EQ(context.volumeRules().front().provenance.geometry_tangent_policy,
              "RefreshedFrozenQuadrature");
    EXPECT_EQ(context.volumeRules().front().provenance.predicate_policy_key,
              result.domain.request().quadrature_policy_key);
    ASSERT_FALSE(context.interfaceRules().empty());
    EXPECT_EQ(context.interfaceRules().front().provenance.marker,
              interface_marker);
    EXPECT_EQ(context.interfaceRules().front().provenance.implicit_fallback_policy,
              "Fail");
    EXPECT_EQ(context.interfaceRules().front().provenance.geometry_tangent_policy,
              "RefreshedFrozenQuadrature");
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP1LineMatchesLinearMeasures)
{
    constexpr int interface_marker = 84;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, x[0]);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_NEAR(result.summary.negative_volume_measure, 2.0, 1.0e-12);
    EXPECT_NEAR(result.summary.positive_volume_measure, 2.0, 1.0e-12);
    EXPECT_NEAR(result.summary.measure, 2.0, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleSeededLineCutsMatchAnalyticAreas)
{
    constexpr int interface_marker = 184;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 4u);
    const auto offset = system.fieldDofOffset(phi);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    std::mt19937_64 rng(0x5eed1234ull);
    std::uniform_real_distribution<FE::Real> cut_distribution(-0.75, 0.75);
    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    for (int sample = 0; sample < 6; ++sample) {
        const FE::Real cut = cut_distribution(rng);
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (std::size_t i = 0; i < 4u; ++i) {
            const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
            solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
                x[0] - cut;
        }

        const auto result = lifecycle.build(system, options, solution);
        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.corner_linearized_cell_count, 0u);
        EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);
        EXPECT_NEAR(result.summary.negative_volume_measure,
                    2.0 * (cut + 1.0),
                    1.0e-12);
        EXPECT_NEAR(result.summary.positive_volume_measure,
                    2.0 * (1.0 - cut),
                    1.0e-12);
        EXPECT_NEAR(result.summary.measure, 2.0, 1.0e-12);
        expectSingleParentVolumeRulesPartitionMeasure(result.domain, 1.0e-12);
        expectGeneratedCutRulesAreFinite(result.domain);
    }
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP1PlaneMatchesHexMeasures)
{
    constexpr int interface_marker = 91;
    const auto mesh = std::make_shared<SingleHexMeshAccess>(FE::ElementType::Hex8);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleHexSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 8; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, x[0]);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_NEAR(result.summary.negative_volume_measure, 4.0, 1.0e-12);
    EXPECT_NEAR(result.summary.positive_volume_measure, 4.0, 1.0e-12);
    EXPECT_NEAR(result.summary.measure, 4.0, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellP1LineMatchesLinearTriangleMeasures)
{
    constexpr int interface_marker = 87;
    const auto mesh =
        std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle3);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 3; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, x[0] - FE::Real{0.5});
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_NEAR(result.summary.negative_volume_measure, 0.375, 1.0e-12);
    EXPECT_NEAR(result.summary.positive_volume_measure, 0.125, 1.0e-12);
    EXPECT_NEAR(result.summary.measure, 0.5, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellP2CircleSegmentApproximatesAreaAndLength)
{
    constexpr int interface_marker = 88;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh =
        std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle6);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 6u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 6u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 6;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 2);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("HighOrderSubcell"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("linearized_leaves="), std::string::npos);
    EXPECT_EQ(result.diagnostic.find("root_finder_iterations=0"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("curved_fragments="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("fallback_used=false"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("achieved_interface_order=2"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("achieved_volume_order=2"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("status="), std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure,
                pi * radius * radius / 4.0,
                2.0e-2);
    EXPECT_NEAR(result.summary.positive_volume_measure,
                0.5 - pi * radius * radius / 4.0,
                2.0e-2);
    EXPECT_NEAR(result.summary.measure,
                pi * radius / 2.0,
                5.0e-2);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    std::set<std::string> interface_rule_topology_ids;
    for (const auto& rule : interface_rules) {
        EXPECT_TRUE(interface_rule_topology_ids
                        .insert(rule.provenance.cut_topology_id)
                        .second)
            << rule.provenance.cut_topology_id;
    }
    EXPECT_EQ(interface_rules.front().provenance.implicit_quadrature_backend,
              "HighOrderSubcell");
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 2);
    EXPECT_TRUE(interface_rules.front().curved_geometry);
    EXPECT_LT(interface_rules.front().points.front().level_set_residual,
              options.implicit_cut_root_tolerance * 10.0);

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    EXPECT_EQ(volume_rules.front().provenance.implicit_quadrature_backend,
              "HighOrderSubcell");
    EXPECT_EQ(volume_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(volume_rules.front().provenance.achieved_quadrature_order, 2);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellTriangleCircleErrorsDecreaseUnderHRefinement)
{
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const FE::Real expected_area = pi * radius * radius / 4.0;
    const FE::Real expected_length = pi * radius / 2.0;
    const auto circle = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0] + x[1] * x[1] - radius * radius;
    };

    const auto coarse = buildSingleTriangleHighOrderSubcellCut(
        circle,
        /*interface_marker=*/1896,
        /*subdivision_depth=*/3,
        /*interface_order=*/1,
        /*volume_order=*/2);
    const auto fine = buildSingleTriangleHighOrderSubcellCut(
        circle,
        /*interface_marker=*/1897,
        /*subdivision_depth=*/6,
        /*interface_order=*/1,
        /*volume_order=*/2);
    ASSERT_TRUE(coarse.success) << coarse.diagnostic;
    ASSERT_TRUE(fine.success) << fine.diagnostic;

    const FE::Real coarse_area_error =
        std::abs(coarse.summary.negative_volume_measure - expected_area);
    const FE::Real fine_area_error =
        std::abs(fine.summary.negative_volume_measure - expected_area);
    const FE::Real coarse_length_error =
        std::abs(coarse.summary.measure - expected_length);
    const FE::Real fine_length_error =
        std::abs(fine.summary.measure - expected_length);

    EXPECT_LT(fine_area_error, coarse_area_error);
    EXPECT_LT(fine_length_error, coarse_length_error);
    EXPECT_EQ(fine.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(fine.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(fine.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(fine.achieved_volume_quadrature_order, 2);
    EXPECT_GT(fine.summary.active_fragment_count,
              coarse.summary.active_fragment_count);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellTriangleP2EdgeDofMovesCutVolume)
{
    constexpr int interface_marker = 188;
    const auto mesh =
        std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle6);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 6u);
    const auto offset = system.fieldDofOffset(phi);

    const auto make_solution = [&](FE::Real edge_mid_value) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (std::size_t i = 0; i < 6u; ++i) {
            const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
            solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
                x[0] - FE::Real{0.5};
        }
        solution[static_cast<std::size_t>(offset + cell_dofs[3])] = edge_mid_value;
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 5;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto linear_edge = lifecycle.build(system, options, make_solution(0.0));
    const auto curved_edge = lifecycle.build(system, options, make_solution(-0.35));
    const auto restored_edge = lifecycle.build(system, options, make_solution(0.0));

    ASSERT_TRUE(linear_edge.success) << linear_edge.diagnostic;
    ASSERT_TRUE(curved_edge.success) << curved_edge.diagnostic;
    ASSERT_TRUE(restored_edge.success) << restored_edge.diagnostic;
    EXPECT_EQ(linear_edge.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(curved_edge.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(restored_edge.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(linear_edge.value_revision, 1u);
    EXPECT_EQ(curved_edge.value_revision, linear_edge.value_revision + 1u);
    EXPECT_EQ(restored_edge.value_revision, curved_edge.value_revision + 1u);
    EXPECT_EQ(linear_edge.domain.request().source.value_revision,
              linear_edge.value_revision);
    EXPECT_EQ(curved_edge.domain.request().source.value_revision,
              curved_edge.value_revision);
    EXPECT_EQ(restored_edge.domain.request().source.value_revision,
              restored_edge.value_revision);
    EXPECT_GT(std::abs(curved_edge.summary.negative_volume_measure -
                       linear_edge.summary.negative_volume_measure),
              1.0e-3);
    EXPECT_NEAR(restored_edge.summary.negative_volume_measure,
                linear_edge.summary.negative_volume_measure,
                1.0e-12);
    EXPECT_GT(std::abs(curved_edge.summary.negative_volume_measure -
                       restored_edge.summary.negative_volume_measure),
              1.0e-3);

    const auto linear_rules = linear_edge.domain.volumeQuadratureRules();
    const auto curved_rules = curved_edge.domain.volumeQuadratureRules();
    const auto restored_rules = restored_edge.domain.volumeQuadratureRules();
    ASSERT_FALSE(linear_rules.empty());
    ASSERT_FALSE(curved_rules.empty());
    ASSERT_FALSE(restored_rules.empty());
    EXPECT_EQ(linear_rules.front().provenance.source_value_revision,
              linear_edge.value_revision);
    EXPECT_EQ(curved_rules.front().provenance.source_value_revision,
              curved_edge.value_revision);
    EXPECT_EQ(restored_rules.front().provenance.source_value_revision,
              restored_edge.value_revision);
    EXPECT_NE(linear_rules.front().provenance.cut_topology_revision,
              curved_rules.front().provenance.cut_topology_revision);
    EXPECT_NE(curved_rules.front().provenance.cut_topology_revision,
              restored_rules.front().provenance.cut_topology_revision);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellTriangleVertexAndEdgeTouchesAreDegenerate)
{
    const auto vertex_touch =
        buildSingleTriangleHighOrderSubcellCut(
            [](const std::array<FE::Real, 3>& x) {
                return x[0] + x[1];
            },
            /*interface_marker=*/1891);
    ASSERT_TRUE(vertex_touch.success) << vertex_touch.diagnostic;
    EXPECT_EQ(vertex_touch.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(vertex_touch.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(vertex_touch.summary.active_fragment_count, 0u);
    EXPECT_NEAR(vertex_touch.summary.negative_volume_measure, 0.0, 1.0e-12);
    EXPECT_NEAR(vertex_touch.summary.positive_volume_measure, 0.5, 1.0e-12);
    EXPECT_NE(vertex_touch.diagnostic.find("status=Degenerate"),
              std::string::npos);
    EXPECT_NE(vertex_touch.diagnostic.find("fallback_used=false"),
              std::string::npos);

    const auto edge_touch =
        buildSingleTriangleHighOrderSubcellCut(
            [](const std::array<FE::Real, 3>& x) {
                return x[1];
            },
            /*interface_marker=*/1892);
    ASSERT_TRUE(edge_touch.success) << edge_touch.diagnostic;
    EXPECT_EQ(edge_touch.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(edge_touch.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(edge_touch.summary.active_fragment_count, 0u);
    EXPECT_NEAR(edge_touch.summary.negative_volume_measure, 0.0, 1.0e-12);
    EXPECT_NEAR(edge_touch.summary.positive_volume_measure, 0.5, 1.0e-12);
    EXPECT_NE(edge_touch.diagnostic.find("status=Degenerate"),
              std::string::npos);
    EXPECT_NE(edge_touch.diagnostic.find("fallback_used=false"),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, BackendIndependentValidationFixturesCheckRuleInvariants)
{
    const auto saye_circle =
        buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                 /*level_set_order=*/2,
                                 /*subdivision_depth=*/6,
                                 /*interface_order=*/1,
                                 /*volume_order=*/2,
                                 /*interface_marker=*/1893);
    ASSERT_TRUE(saye_circle.success) << saye_circle.diagnostic;
    expectSingleParentVolumeRulesPartitionMeasure(saye_circle.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(saye_circle.domain);

    const auto subcell_circle =
        buildSingleTriangleHighOrderSubcellCut(
            [](const std::array<FE::Real, 3>& x) {
                constexpr FE::Real radius = 0.5;
                return x[0] * x[0] + x[1] * x[1] - radius * radius;
            },
            /*interface_marker=*/1894);
    ASSERT_TRUE(subcell_circle.success) << subcell_circle.diagnostic;
    expectSingleParentVolumeRulesPartitionMeasure(subcell_circle.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(subcell_circle.domain);
}

TEST(LevelSetInterfaceLifecycle, GeneratedOrderSummariesMatchRuleProvenance)
{
    const auto result =
        buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                 /*level_set_order=*/2,
                                 /*subdivision_depth=*/6,
                                 /*interface_order=*/3,
                                 /*volume_order=*/5,
                                 /*interface_marker=*/1895);

    ASSERT_TRUE(result.success) << result.diagnostic;
    const auto interface_rules = result.domain.interfaceQuadratureRules();
    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    ASSERT_FALSE(volume_rules.empty());

    int min_interface_order = std::numeric_limits<int>::max();
    for (const auto& rule : interface_rules) {
        min_interface_order =
            std::min(min_interface_order,
                     rule.provenance.achieved_quadrature_order);
        EXPECT_LE(rule.provenance.achieved_quadrature_order,
                  rule.provenance.requested_quadrature_order);
    }

    int min_volume_order = std::numeric_limits<int>::max();
    for (const auto& rule : volume_rules) {
        min_volume_order =
            std::min(min_volume_order,
                     rule.provenance.achieved_quadrature_order);
        EXPECT_LE(rule.provenance.achieved_quadrature_order,
                  rule.provenance.requested_quadrature_order);
    }

    EXPECT_EQ(result.achieved_interface_quadrature_order, min_interface_order);
    EXPECT_EQ(result.achieved_volume_quadrature_order, min_volume_order);
    EXPECT_EQ(result.domain.request().achieved_interface_quadrature_order,
              min_interface_order);
    EXPECT_EQ(result.domain.request().achieved_volume_quadrature_order,
              min_volume_order);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellP1PlaneMatchesLinearTetraMeasures)
{
    constexpr int interface_marker = 89;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, x[0] - FE::Real{0.5});
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NEAR(result.summary.negative_volume_measure,
                7.0 / 48.0,
                1.0e-12);
    EXPECT_NEAR(result.summary.positive_volume_measure,
                1.0 / 48.0,
                1.0e-12);
    EXPECT_NEAR(result.summary.measure, 0.125, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, AffineCutsMatchLinearMeasuresForSupportedHighOrderCells)
{
    constexpr FE::Real cut = 0.37;
    constexpr FE::Real quad_negative = 2.0 * (1.0 + cut);
    constexpr FE::Real quad_positive = 2.0 * (1.0 - cut);
    constexpr FE::Real hex_negative = 4.0 * (1.0 + cut);
    constexpr FE::Real hex_positive = 4.0 * (1.0 - cut);
    constexpr FE::Real tri_negative = cut - 0.5 * cut * cut;
    constexpr FE::Real tri_positive = 0.5 - tri_negative;
    constexpr FE::Real tri_interface = 1.0 - cut;
    constexpr FE::Real tet_positive =
        (1.0 - cut) * (1.0 - cut) * (1.0 - cut) / 6.0;
    constexpr FE::Real tet_negative = 1.0 / 6.0 - tet_positive;
    constexpr FE::Real tet_interface = 0.5 * (1.0 - cut) * (1.0 - cut);

    struct Case {
        FE::ElementType element_type{FE::ElementType::Unknown};
        level_set::ImplicitCutQuadratureBackend backend{
            level_set::ImplicitCutQuadratureBackend::Auto};
        FE::Real negative_measure{0.0};
        FE::Real positive_measure{0.0};
        FE::Real interface_measure{0.0};
    };

    const std::array<Case, 10> cases{{
        {FE::ElementType::Quad4,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         quad_negative,
         quad_positive,
         2.0},
        {FE::ElementType::Quad8,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         quad_negative,
         quad_positive,
         2.0},
        {FE::ElementType::Quad9,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         quad_negative,
         quad_positive,
         2.0},
        {FE::ElementType::Hex8,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         hex_negative,
         hex_positive,
         4.0},
        {FE::ElementType::Hex20,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         hex_negative,
         hex_positive,
         4.0},
        {FE::ElementType::Hex27,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         hex_negative,
         hex_positive,
         4.0},
        {FE::ElementType::Triangle3,
         level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
         tri_negative,
         tri_positive,
         tri_interface},
        {FE::ElementType::Triangle6,
         level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
         tri_negative,
         tri_positive,
         tri_interface},
        {FE::ElementType::Tetra4,
         level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
         tet_negative,
         tet_positive,
         tet_interface},
        {FE::ElementType::Tetra10,
         level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
         tet_negative,
         tet_positive,
         tet_interface},
    }};

    int interface_marker = 2100;
    for (const auto& entry : cases) {
        SCOPED_TRACE(static_cast<int>(entry.element_type));
        level_set::LevelSetGeneratedInterfaceResult result{};
        try {
            result =
                entry.backend == level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle
                    ? buildSayeAffineHyperrectangleCut(entry.element_type,
                                                       interface_marker++,
                                                       cut)
                    : buildHighOrderSubcellAffineSimplexCut(entry.element_type,
                                                            interface_marker++,
                                                            cut);
        } catch (const std::exception& ex) {
            FAIL() << "element_type=" << static_cast<int>(entry.element_type)
                   << " threw: " << ex.what();
        }
        SCOPED_TRACE(result.diagnostic);
        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.corner_linearized_cell_count, 0u);
        EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u)
            << result.diagnostic;
        EXPECT_EQ(result.implicit_cut_quadrature_backend, entry.backend);
        EXPECT_NEAR(result.summary.negative_volume_measure,
                    entry.negative_measure,
                    1.0e-12);
        EXPECT_NEAR(result.summary.positive_volume_measure,
                    entry.positive_measure,
                    1.0e-12);
        EXPECT_NEAR(result.summary.measure, entry.interface_measure, 1.0e-12);
        expectSingleParentVolumeRulesPartitionMeasure(result.domain, 1.0e-12);
        expectGeneratedCutRulesAreFinite(result.domain);
    }
}

TEST(LevelSetInterfaceLifecycle, AffinePlaneCutsStayPlanarOnHighOrderPhysicalCells)
{
    constexpr FE::Real cut = 0.37;
    constexpr FE::Real hex_negative = 4.0 * (1.0 + cut);
    constexpr FE::Real hex_positive = 4.0 * (1.0 - cut);
    constexpr FE::Real tet_positive =
        (1.0 - cut) * (1.0 - cut) * (1.0 - cut) / 6.0;
    constexpr FE::Real tet_negative = 1.0 / 6.0 - tet_positive;

    struct Case {
        FE::ElementType element_type{FE::ElementType::Unknown};
        level_set::ImplicitCutQuadratureBackend backend{
            level_set::ImplicitCutQuadratureBackend::Auto};
        FE::Real negative_measure{0.0};
        FE::Real positive_measure{0.0};
        FE::Real interface_measure{0.0};
    };

    const std::array<Case, 3> cases{{
        {FE::ElementType::Quad9,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         2.0 * (1.0 + cut),
         2.0 * (1.0 - cut),
         2.0},
        {FE::ElementType::Hex27,
         level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
         hex_negative,
         hex_positive,
         4.0},
        {FE::ElementType::Tetra10,
         level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
         tet_negative,
         tet_positive,
         0.5 * (1.0 - cut) * (1.0 - cut)},
    }};

    int interface_marker = 2120;
    for (const auto& entry : cases) {
        SCOPED_TRACE(static_cast<int>(entry.element_type));
        const auto result =
            entry.backend == level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle
                ? buildSayeAffineHyperrectangleCut(entry.element_type,
                                                   interface_marker++,
                                                   cut)
                : buildHighOrderSubcellAffineSimplexCut(entry.element_type,
                                                        interface_marker++,
                                                        cut);
        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.implicit_cut_quadrature_backend, entry.backend);
        EXPECT_EQ(result.corner_linearized_cell_count, 0u);
        EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);
        EXPECT_NEAR(result.summary.negative_volume_measure,
                    entry.negative_measure,
                    1.0e-12);
        EXPECT_NEAR(result.summary.positive_volume_measure,
                    entry.positive_measure,
                    1.0e-12);
        EXPECT_NEAR(result.summary.measure, entry.interface_measure, 1.0e-12);

        const auto interface_rules = result.domain.interfaceQuadratureRules();
        ASSERT_FALSE(interface_rules.empty());
        for (const auto& rule : interface_rules) {
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      entry.backend == level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle
                          ? "SayeHyperrectangle"
                          : "HighOrderSubcell");
            for (const auto& point : rule.points) {
                EXPECT_NEAR(point.point[0], cut, 1.0e-12);
                EXPECT_LE(point.level_set_residual, 1.0e-12);
                EXPECT_GT(point.weight, 0.0);
            }
        }
    }
}

TEST(LevelSetInterfaceLifecycle, Affine2DCutRulesIntegratePolynomialMomentsToAdvertisedOrder)
{
    constexpr FE::Real cut = 0.37;

    const auto x5 = [](const std::array<FE::Real, 3>& x) {
        return std::pow(x[0], 5);
    };
    const auto shifted_y5 = [](const std::array<FE::Real, 3>& x) {
        return std::pow(x[1] + FE::Real{0.25}, 5);
    };
    const auto saye_quad =
        buildSayeAffineHyperrectangleCut(FE::ElementType::Quad9,
                                         /*interface_marker=*/2130,
                                         cut,
                                         /*subdivision_depth=*/3,
                                         /*interface_order=*/5,
                                         /*volume_order=*/5);
    ASSERT_TRUE(saye_quad.success) << saye_quad.diagnostic;
    EXPECT_EQ(saye_quad.achieved_interface_quadrature_order, 5);
    EXPECT_EQ(saye_quad.achieved_volume_quadrature_order, 5);
    EXPECT_NEAR(integrateVolumeMoment(saye_quad.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      x5),
                (std::pow(cut, 6) - 1.0) / 3.0,
                1.0e-12);
    EXPECT_NEAR(integrateVolumeMoment(saye_quad.domain,
                                      FE::geometry::CutIntegrationSide::Positive,
                                      x5),
                (1.0 - std::pow(cut, 6)) / 3.0,
                1.0e-12);
    EXPECT_NEAR(integrateInterfaceMoment(saye_quad.domain, shifted_y5),
                (std::pow(1.25, 6) - std::pow(-0.75, 6)) / 6.0,
                1.0e-12);

    const auto subcell_triangle =
        buildHighOrderSubcellAffineSimplexCut(FE::ElementType::Triangle6,
                                              /*interface_marker=*/2131,
                                              cut,
                                              /*subdivision_depth=*/3,
                                              /*interface_order=*/5,
                                              /*volume_order=*/5);
    ASSERT_TRUE(subcell_triangle.success) << subcell_triangle.diagnostic;
    EXPECT_EQ(subcell_triangle.achieved_interface_quadrature_order, 5);
    EXPECT_EQ(subcell_triangle.achieved_volume_quadrature_order, 5);
    const FE::Real triangle_negative_x5 =
        std::pow(cut, 6) / 6.0 - std::pow(cut, 7) / 7.0;
    EXPECT_NEAR(integrateVolumeMoment(subcell_triangle.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      x5),
                triangle_negative_x5,
                1.0e-12);
    EXPECT_NEAR(integrateVolumeMoment(subcell_triangle.domain,
                                      FE::geometry::CutIntegrationSide::Positive,
                                      x5),
                1.0 / 42.0 - triangle_negative_x5,
                1.0e-12);
    EXPECT_NEAR(integrateInterfaceMoment(subcell_triangle.domain, shifted_y5),
                (std::pow(1.0 - cut + 0.25, 6) - std::pow(0.25, 6)) / 6.0,
                1.0e-11);
}

TEST(LevelSetInterfaceLifecycle, Affine3DCutRulesIntegratePolynomialMomentsToAdvertisedOrder)
{
    constexpr FE::Real cut = 0.37;
    constexpr FE::Real shift = 0.25;

    const auto x2 = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0];
    };
    const auto shifted_y = [](const std::array<FE::Real, 3>& x) {
        return x[1] + shift;
    };
    const auto shifted_y2 = [](const std::array<FE::Real, 3>& x) {
        return (x[1] + shift) * (x[1] + shift);
    };

    const auto saye_hex =
        buildSayeAffineHyperrectangleCut(FE::ElementType::Hex27,
                                         /*interface_marker=*/2132,
                                         cut,
                                         /*subdivision_depth=*/3,
                                         /*interface_order=*/1,
                                         /*volume_order=*/2);
    ASSERT_TRUE(saye_hex.success) << saye_hex.diagnostic;
    EXPECT_EQ(saye_hex.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(saye_hex.achieved_volume_quadrature_order, 2);
    EXPECT_NEAR(integrateVolumeMoment(saye_hex.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      x2),
                4.0 * (std::pow(cut, 3) + 1.0) / 3.0,
                1.0e-12);
    EXPECT_NEAR(integrateVolumeMoment(saye_hex.domain,
                                      FE::geometry::CutIntegrationSide::Positive,
                                      x2),
                4.0 * (1.0 - std::pow(cut, 3)) / 3.0,
                1.0e-12);
    EXPECT_NEAR(integrateInterfaceMoment(saye_hex.domain, shifted_y),
                1.0,
                1.0e-12);

    const auto subcell_tetra =
        buildHighOrderSubcellAffineSimplexCut(FE::ElementType::Tetra10,
                                              /*interface_marker=*/2133,
                                              cut,
                                              /*subdivision_depth=*/3,
                                              /*interface_order=*/2,
                                              /*volume_order=*/2);
    ASSERT_TRUE(subcell_tetra.success) << subcell_tetra.diagnostic;
    EXPECT_EQ(subcell_tetra.achieved_interface_quadrature_order, 2);
    EXPECT_EQ(subcell_tetra.achieved_volume_quadrature_order, 2);
    const FE::Real tetra_negative_x2 =
        0.5 * (std::pow(cut, 3) / 3.0 -
               std::pow(cut, 4) / 2.0 +
               std::pow(cut, 5) / 5.0);
    EXPECT_NEAR(integrateVolumeMoment(subcell_tetra.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      x2),
                tetra_negative_x2,
                1.0e-12);
    EXPECT_NEAR(integrateVolumeMoment(subcell_tetra.domain,
                                      FE::geometry::CutIntegrationSide::Positive,
                                      x2),
                1.0 / 60.0 - tetra_negative_x2,
                1.0e-12);
    const FE::Real tetra_interface_edge = 1.0 - cut;
    const FE::Real tetra_interface_upper = tetra_interface_edge + shift;
    const FE::Real tetra_interface_y2 =
        tetra_interface_upper *
            (std::pow(tetra_interface_upper, 3) - std::pow(shift, 3)) / 3.0 -
        (std::pow(tetra_interface_upper, 4) - std::pow(shift, 4)) / 4.0;
    EXPECT_NEAR(integrateInterfaceMoment(subcell_tetra.domain, shifted_y2),
                tetra_interface_y2,
                2.0e-12);
}

TEST(LevelSetInterfaceLifecycle, Affine2DCutRulesImprovePolynomialMomentsUnderQRefinement)
{
    constexpr FE::Real cut = 0.37;
    const auto x5 = [](const std::array<FE::Real, 3>& x) {
        return std::pow(x[0], 5);
    };
    const auto shifted_y5 = [](const std::array<FE::Real, 3>& x) {
        return std::pow(x[1] + FE::Real{0.25}, 5);
    };

    const FE::Real quad_exact_volume = (std::pow(cut, 6) - 1.0) / 3.0;
    const FE::Real quad_exact_interface =
        (std::pow(1.25, 6) - std::pow(-0.75, 6)) / 6.0;
    const auto quad_low =
        buildSayeAffineHyperrectangleCut(FE::ElementType::Quad9,
                                         /*interface_marker=*/2134,
                                         cut,
                                         /*subdivision_depth=*/3,
                                         /*interface_order=*/1,
                                         /*volume_order=*/1);
    const auto quad_high =
        buildSayeAffineHyperrectangleCut(FE::ElementType::Quad9,
                                         /*interface_marker=*/2135,
                                         cut,
                                         /*subdivision_depth=*/3,
                                         /*interface_order=*/5,
                                         /*volume_order=*/5);
    ASSERT_TRUE(quad_low.success) << quad_low.diagnostic;
    ASSERT_TRUE(quad_high.success) << quad_high.diagnostic;
    const FE::Real quad_low_volume_error =
        std::abs(integrateVolumeMoment(quad_low.domain,
                                       FE::geometry::CutIntegrationSide::Negative,
                                       x5) -
                 quad_exact_volume);
    const FE::Real quad_high_volume_error =
        std::abs(integrateVolumeMoment(quad_high.domain,
                                       FE::geometry::CutIntegrationSide::Negative,
                                       x5) -
                 quad_exact_volume);
    const FE::Real quad_low_interface_error =
        std::abs(integrateInterfaceMoment(quad_low.domain, shifted_y5) -
                 quad_exact_interface);
    const FE::Real quad_high_interface_error =
        std::abs(integrateInterfaceMoment(quad_high.domain, shifted_y5) -
                 quad_exact_interface);
    EXPECT_GT(quad_low_volume_error, 1.0e-4);
    EXPECT_GT(quad_low_interface_error, 1.0e-4);
    EXPECT_LT(quad_high_volume_error, 1.0e-11);
    EXPECT_LT(quad_high_interface_error, 1.0e-11);

    const FE::Real triangle_exact_volume =
        std::pow(cut, 6) / 6.0 - std::pow(cut, 7) / 7.0;
    const FE::Real triangle_exact_interface =
        (std::pow(1.0 - cut + 0.25, 6) - std::pow(0.25, 6)) / 6.0;
    const auto triangle_low =
        buildHighOrderSubcellAffineSimplexCut(FE::ElementType::Triangle6,
                                              /*interface_marker=*/2136,
                                              cut,
                                              /*subdivision_depth=*/3,
                                              /*interface_order=*/1,
                                              /*volume_order=*/1);
    const auto triangle_high =
        buildHighOrderSubcellAffineSimplexCut(FE::ElementType::Triangle6,
                                              /*interface_marker=*/2137,
                                              cut,
                                              /*subdivision_depth=*/3,
                                              /*interface_order=*/5,
                                              /*volume_order=*/5);
    ASSERT_TRUE(triangle_low.success) << triangle_low.diagnostic;
    ASSERT_TRUE(triangle_high.success) << triangle_high.diagnostic;
    const FE::Real triangle_low_volume_error =
        std::abs(integrateVolumeMoment(triangle_low.domain,
                                       FE::geometry::CutIntegrationSide::Negative,
                                       x5) -
                 triangle_exact_volume);
    const FE::Real triangle_high_volume_error =
        std::abs(integrateVolumeMoment(triangle_high.domain,
                                       FE::geometry::CutIntegrationSide::Negative,
                                       x5) -
                 triangle_exact_volume);
    const FE::Real triangle_low_interface_error =
        std::abs(integrateInterfaceMoment(triangle_low.domain, shifted_y5) -
                 triangle_exact_interface);
    const FE::Real triangle_high_interface_error =
        std::abs(integrateInterfaceMoment(triangle_high.domain, shifted_y5) -
                 triangle_exact_interface);
    EXPECT_GT(triangle_low_volume_error, 1.0e-7);
    EXPECT_GT(triangle_low_interface_error, 1.0e-4);
    EXPECT_LT(triangle_high_volume_error, 1.0e-11);
    EXPECT_LT(triangle_high_interface_error, 1.0e-10);
}

TEST(LevelSetInterfaceLifecycle, GeneratedRulesConserveParentMeasureAndUsePositiveWeights)
{
    constexpr FE::Real radius = 0.5;
    const auto circle2d = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0] + x[1] * x[1] - radius * radius;
    };
    const auto sphere3d = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] -
               radius * radius;
    };

    const auto saye_quad =
        buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                 /*level_set_order=*/2,
                                 /*subdivision_depth=*/6,
                                 /*interface_order=*/2,
                                 /*volume_order=*/2,
                                 /*interface_marker=*/2138);
    ASSERT_TRUE(saye_quad.success) << saye_quad.diagnostic;
    expectSingleParentVolumeRulesPartitionMeasure(saye_quad.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(saye_quad.domain);

    const auto subcell_triangle =
        buildSingleTriangleHighOrderSubcellCut(circle2d,
                                               /*interface_marker=*/2139,
                                               /*subdivision_depth=*/6,
                                               /*interface_order=*/2,
                                               /*volume_order=*/2);
    ASSERT_TRUE(subcell_triangle.success) << subcell_triangle.diagnostic;
    expectSingleParentVolumeRulesPartitionMeasure(subcell_triangle.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(subcell_triangle.domain);

    const auto saye_hex =
        buildSingleHexSayeCut(sphere3d,
                              /*subdivision_depth=*/4,
                              /*interface_order=*/2,
                              /*volume_order=*/2,
                              /*interface_marker=*/2140);
    ASSERT_TRUE(saye_hex.success) << saye_hex.diagnostic;
    expectSingleParentVolumeRulesPartitionMeasure(saye_hex.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(saye_hex.domain);

    const auto subcell_tetra =
        buildSingleTetraHighOrderSubcellCut(sphere3d,
                                            /*interface_marker=*/2141,
                                            /*subdivision_depth=*/5,
                                            /*interface_order=*/2,
                                            /*volume_order=*/2);
    ASSERT_TRUE(subcell_tetra.success) << subcell_tetra.diagnostic;
    expectSingleParentVolumeRulesPartitionMeasure(subcell_tetra.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(subcell_tetra.domain);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellP2SphereCapApproximatesVolumeAndArea)
{
    constexpr int interface_marker = 90;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 10u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 10u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 5;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u) << result.diagnostic;
    EXPECT_EQ(result.achieved_interface_quadrature_order, 2) << result.diagnostic;
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("HighOrderSubcell"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("linearized_leaves="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("curved_fragment_failures=0"),
              std::string::npos);
    EXPECT_NE(result.diagnostic.find("curved_edge_root_mismatches=0"),
              std::string::npos);
    EXPECT_NE(result.diagnostic.find("curved_boundary_degenerate="),
              std::string::npos);
    EXPECT_NE(result.diagnostic.find("fallback_used=false"),
              std::string::npos);
    EXPECT_NE(result.diagnostic.find("achieved_interface_order=2"),
              std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure,
                pi * radius * radius * radius / 6.0,
                1.5e-2);
    EXPECT_NEAR(result.summary.positive_volume_measure,
                1.0 / 6.0 - pi * radius * radius * radius / 6.0,
                1.5e-2);
    EXPECT_NEAR(result.summary.measure,
                pi * radius * radius / 2.0,
                8.0e-2);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    EXPECT_EQ(interface_rules.front().provenance.implicit_quadrature_backend,
              "HighOrderSubcell");
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 2);
    EXPECT_TRUE(interface_rules.front().curved_geometry);
    EXPECT_EQ(interface_rules.front().policy.name,
              "root-polished-level-set-interface");
    expectInterfaceNormalsAlignWithGradient(
        result.domain,
        [](const std::array<FE::Real, 3>& x) {
            return std::array<FE::Real, 3>{
                2.0 * x[0], 2.0 * x[1], 2.0 * x[2]};
        },
        0.999999);

    std::size_t root_polished_active_fragment_count = 0u;
    std::size_t boundary_degenerate_fragment_count = 0u;
    for (const auto& fragment : result.domain.fragments()) {
        if (fragment.conditioning_diagnostic ==
            "boundary-only-curved-degenerate") {
            EXPECT_FALSE(fragment.active());
            EXPECT_TRUE(fragment.root_polished);
            EXPECT_EQ(fragment.kind,
                      FE::interfaces::CutInterfaceFragmentKind::CurvedPatch);
            EXPECT_EQ(fragment.measure, 0.0);
            ++boundary_degenerate_fragment_count;
            continue;
        }
        if (!fragment.active()) {
            continue;
        }
        EXPECT_TRUE(fragment.root_polished);
        EXPECT_EQ(fragment.kind,
                  FE::interfaces::CutInterfaceFragmentKind::CurvedPatch);
        EXPECT_EQ(fragment.conditioning_diagnostic, "ok");
        EXPECT_GE(fragment.root_finder_iterations, 0);
        EXPECT_LE(fragment.max_root_residual,
                  options.implicit_cut_root_tolerance * 10.0);
        EXPECT_GT(fragment.min_gradient_norm, 1.0e-12);
        ++root_polished_active_fragment_count;
    }
    EXPECT_EQ(root_polished_active_fragment_count,
              result.summary.active_fragment_count);
    EXPECT_GT(boundary_degenerate_fragment_count, 0u);

    for (const auto& rule : interface_rules) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.provenance.achieved_quadrature_order, 2);
        for (const auto& point : rule.points) {
            EXPECT_GT(point.weight, 0.0);
            EXPECT_GT(point.reference_measure_factor, 0.0);
            EXPECT_LE(point.level_set_residual,
                      options.implicit_cut_root_tolerance * 10.0);
            EXPECT_GT(point.gradient_norm, 1.0e-12);
        }
    }

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    EXPECT_EQ(volume_rules.front().provenance.implicit_quadrature_backend,
              "HighOrderSubcell");
    EXPECT_EQ(volume_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(volume_rules.front().provenance.achieved_quadrature_order, 2);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellTetraSphereCapErrorsDecreaseUnderHRefinement)
{
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const FE::Real expected_volume = pi * radius * radius * radius / 6.0;
    const FE::Real expected_area = pi * radius * radius / 2.0;
    const auto sphere = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] -
               radius * radius;
    };

    const auto coarse = buildSingleTetraHighOrderSubcellCut(
        sphere,
        /*interface_marker=*/1898,
        /*subdivision_depth=*/3,
        /*interface_order=*/2,
        /*volume_order=*/2);
    const auto fine = buildSingleTetraHighOrderSubcellCut(
        sphere,
        /*interface_marker=*/1899,
        /*subdivision_depth=*/5,
        /*interface_order=*/2,
        /*volume_order=*/2);
    ASSERT_TRUE(coarse.success) << coarse.diagnostic;
    ASSERT_TRUE(fine.success) << fine.diagnostic;

    const FE::Real coarse_volume_error =
        std::abs(coarse.summary.negative_volume_measure - expected_volume);
    const FE::Real fine_volume_error =
        std::abs(fine.summary.negative_volume_measure - expected_volume);
    const FE::Real coarse_area_error =
        std::abs(coarse.summary.measure - expected_area);
    const FE::Real fine_area_error =
        std::abs(fine.summary.measure - expected_area);

    EXPECT_LT(fine_volume_error, coarse_volume_error);
    EXPECT_LT(fine_area_error, coarse_area_error);
    EXPECT_EQ(fine.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(fine.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(fine.achieved_interface_quadrature_order, 2);
    EXPECT_EQ(fine.achieved_volume_quadrature_order, 2);
    EXPECT_GT(fine.summary.active_fragment_count,
              coarse.summary.active_fragment_count);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellTetraP2EdgeDofMovesCutVolume)
{
    constexpr int interface_marker = 190;
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 10u);
    const auto offset = system.fieldDofOffset(phi);

    const auto make_solution = [&](FE::Real edge_mid_value) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (std::size_t i = 0; i < 10u; ++i) {
            const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
            solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
                x[0] - FE::Real{0.5};
        }
        solution[static_cast<std::size_t>(offset + cell_dofs[4])] = edge_mid_value;
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 4;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto linear_edge = lifecycle.build(system, options, make_solution(0.0));
    const auto curved_edge = lifecycle.build(system, options, make_solution(-0.35));

    ASSERT_TRUE(linear_edge.success) << linear_edge.diagnostic;
    ASSERT_TRUE(curved_edge.success) << curved_edge.diagnostic;
    EXPECT_EQ(linear_edge.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(curved_edge.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(linear_edge.value_revision, 1u);
    EXPECT_EQ(curved_edge.value_revision, linear_edge.value_revision + 1u);
    EXPECT_EQ(linear_edge.domain.request().source.value_revision,
              linear_edge.value_revision);
    EXPECT_EQ(curved_edge.domain.request().source.value_revision,
              curved_edge.value_revision);
    EXPECT_GT(std::abs(curved_edge.summary.negative_volume_measure -
                       linear_edge.summary.negative_volume_measure),
              1.0e-4);

    const auto linear_rules = linear_edge.domain.volumeQuadratureRules();
    const auto curved_rules = curved_edge.domain.volumeQuadratureRules();
    ASSERT_FALSE(linear_rules.empty());
    ASSERT_FALSE(curved_rules.empty());
    EXPECT_EQ(linear_rules.front().provenance.source_value_revision,
              linear_edge.value_revision);
    EXPECT_EQ(curved_rules.front().provenance.source_value_revision,
              curved_edge.value_revision);
    EXPECT_NE(linear_rules.front().provenance.cut_topology_revision,
              curved_rules.front().provenance.cut_topology_revision);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleQuadP3DofMovesGeometryAndQuadrature)
{
    constexpr int interface_marker = 191;
    constexpr FE::Real cut = 0.37;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/3, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto dof_coordinates =
        FE::basis::NodeOrdering::get_lagrange_node_coords(FE::ElementType::Quad4, 3);
    ASSERT_EQ(cell_dofs.size(), dof_coordinates.size());
    const auto offset = system.fieldDofOffset(phi);

    const auto make_solution = [&](FE::Real edge_dof_delta) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (std::size_t i = 0; i < dof_coordinates.size(); ++i) {
            const auto& xi = dof_coordinates[i];
            solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
                xi[0] - cut;
        }
        constexpr std::size_t lower_edge_x_plus_third_dof = 5u;
        solution[static_cast<std::size_t>(
            offset + cell_dofs[lower_edge_x_plus_third_dof])] += edge_dof_delta;
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 6;
    options.interface_quadrature_order = 3;
    options.volume_quadrature_order = 3;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto affine = lifecycle.build(system, options, make_solution(0.0));
    const auto perturbed = lifecycle.build(system, options, make_solution(0.55));

    ASSERT_TRUE(affine.success) << affine.diagnostic;
    ASSERT_TRUE(perturbed.success) << perturbed.diagnostic;
    EXPECT_EQ(affine.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(perturbed.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(affine.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(perturbed.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(affine.value_revision, 1u);
    EXPECT_EQ(perturbed.value_revision, affine.value_revision + 1u);

    const auto y_moment = [](const std::array<FE::Real, 3>& x) {
        return x[1];
    };
    EXPECT_GT(std::abs(perturbed.summary.negative_volume_measure -
                       affine.summary.negative_volume_measure),
              1.0e-3);
    EXPECT_GT(std::abs(integrateInterfaceMoment(perturbed.domain, y_moment) -
                       integrateInterfaceMoment(affine.domain, y_moment)),
              1.0e-3);

    const auto affine_interface_rules = affine.domain.interfaceQuadratureRules();
    const auto perturbed_interface_rules = perturbed.domain.interfaceQuadratureRules();
    ASSERT_FALSE(affine_interface_rules.empty());
    ASSERT_FALSE(perturbed_interface_rules.empty());
    EXPECT_NE(ruleSignatures(affine_interface_rules),
              ruleSignatures(perturbed_interface_rules));
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellTriangleP3DofMovesGeometryAndQuadrature)
{
    constexpr int interface_marker = 192;
    constexpr FE::Real cut = 0.37;
    const auto mesh =
        std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle3);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/3, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    const auto dof_coordinates =
        FE::basis::NodeOrdering::get_lagrange_node_coords(FE::ElementType::Triangle3, 3);
    ASSERT_EQ(cell_dofs.size(), dof_coordinates.size());
    const auto offset = system.fieldDofOffset(phi);

    const auto make_solution = [&](FE::Real edge_dof_delta) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (std::size_t i = 0; i < dof_coordinates.size(); ++i) {
            const auto& xi = dof_coordinates[i];
            solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
                xi[0] - cut;
        }
        constexpr std::size_t lower_edge_x_one_third_dof = 3u;
        solution[static_cast<std::size_t>(
            offset + cell_dofs[lower_edge_x_one_third_dof])] += edge_dof_delta;
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 5;
    options.interface_quadrature_order = 3;
    options.volume_quadrature_order = 3;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto affine = lifecycle.build(system, options, make_solution(0.0));
    const auto perturbed = lifecycle.build(system, options, make_solution(0.55));

    ASSERT_TRUE(affine.success) << affine.diagnostic;
    ASSERT_TRUE(perturbed.success) << perturbed.diagnostic;
    EXPECT_EQ(affine.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(perturbed.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(affine.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(perturbed.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(affine.value_revision, 1u);
    EXPECT_EQ(perturbed.value_revision, affine.value_revision + 1u);

    const auto y_moment = [](const std::array<FE::Real, 3>& x) {
        return x[1];
    };
    EXPECT_GT(std::abs(perturbed.summary.negative_volume_measure -
                       affine.summary.negative_volume_measure),
              1.0e-3);
    EXPECT_GT(std::abs(integrateInterfaceMoment(perturbed.domain, y_moment) -
                       integrateInterfaceMoment(affine.domain, y_moment)),
              1.0e-4);

    const auto affine_interface_rules = affine.domain.interfaceQuadratureRules();
    const auto perturbed_interface_rules = perturbed.domain.interfaceQuadratureRules();
    ASSERT_FALSE(affine_interface_rules.empty());
    ASSERT_FALSE(perturbed_interface_rules.empty());
    EXPECT_NE(ruleSignatures(affine_interface_rules),
              ruleSignatures(perturbed_interface_rules));
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP2CircleApproximatesAreaAndLength)
{
    constexpr int interface_marker = 85;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 6;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 2);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("SayeHyperrectangle"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("max_depth_reached="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("linearized_leaves="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("interface_fragments="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("root_finder_iterations="), std::string::npos);
    EXPECT_EQ(result.diagnostic.find("root_finder_iterations=0"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("fallback_used=false"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("achieved_interface_order=2"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("achieved_volume_order=2"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("status="), std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure,
                pi * radius * radius,
                6.0e-2);
    EXPECT_NEAR(result.summary.measure,
                2.0 * pi * radius,
                1.2e-1);

    const auto x_moment = [](const std::array<FE::Real, 3>& x) { return x[0]; };
    const auto y_moment = [](const std::array<FE::Real, 3>& x) { return x[1]; };
    const auto r2_moment = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      x_moment),
                0.0,
                2.0e-2);
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      y_moment),
                0.0,
                2.0e-2);
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      r2_moment),
                pi * radius * radius * radius * radius / 2.0,
                3.0e-2);
    EXPECT_NEAR(integrateInterfaceMoment(result.domain, x_moment), 0.0, 2.0e-2);
    EXPECT_NEAR(integrateInterfaceMoment(result.domain, y_moment), 0.0, 2.0e-2);
    EXPECT_NEAR(integrateInterfaceMoment(result.domain, r2_moment),
                2.0 * pi * radius * radius * radius,
                6.0e-2);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    std::set<std::string> interface_rule_topology_ids;
    for (const auto& rule : interface_rules) {
        EXPECT_TRUE(interface_rule_topology_ids
                        .insert(rule.provenance.cut_topology_id)
                        .second)
            << rule.provenance.cut_topology_id;
    }
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 2);
    EXPECT_TRUE(interface_rules.front().curved_geometry);
    EXPECT_EQ(interface_rules.front().policy.name,
              "root-polished-level-set-interface");
    expectInterfaceNormalsAlignWithGradient(
        result.domain,
        [](const std::array<FE::Real, 3>& x) {
            return std::array<FE::Real, 3>{2.0 * x[0], 2.0 * x[1], 0.0};
        },
        0.999999);

    std::size_t root_polished_fragment_count = 0u;
    int recorded_root_iterations = 0;
    std::set<std::string> active_fragment_topology_ids;
    for (const auto& fragment : result.domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        EXPECT_TRUE(active_fragment_topology_ids.insert(fragment.topology_id).second)
            << fragment.topology_id;
        EXPECT_TRUE(fragment.root_polished);
        EXPECT_EQ(fragment.kind, FE::interfaces::CutInterfaceFragmentKind::CurvedPatch);
        EXPECT_FALSE(fragment.branch_id.empty());
        EXPECT_EQ(fragment.branch_id, fragment.topology_id);
        EXPECT_EQ(fragment.conditioning_diagnostic, "ok");
        EXPECT_GE(fragment.root_finder_iterations, 0);
        EXPECT_LE(fragment.max_root_residual, 1.0e-8);
        EXPECT_GT(fragment.min_gradient_norm, 1.0e-12);
        recorded_root_iterations += fragment.root_finder_iterations;
        ++root_polished_fragment_count;
    }
    EXPECT_EQ(root_polished_fragment_count, result.summary.active_fragment_count);
    EXPECT_GT(recorded_root_iterations, 0);

    for (const auto& rule : interface_rules) {
        EXPECT_TRUE(rule.curved_geometry);
        for (const auto& point : rule.points) {
            EXPECT_GT(point.weight, 0.0);
            EXPECT_GT(point.reference_measure_factor, 0.0);
            EXPECT_LE(point.level_set_residual, 1.0e-8);
            EXPECT_GT(point.gradient_norm, 1.0e-12);
        }
    }

    const std::string vtp =
        FE::interfaces::levelSetInterfaceGeometryVtpString(result.domain);
    EXPECT_NE(vtp.find("<VTKFile type=\"PolyData\""), std::string::npos);
    EXPECT_EQ(vtp.find("NumberOfLines=\"0\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"level_set_value\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"interface_normal\" NumberOfComponents=\"3\""),
              std::string::npos);
    EXPECT_NE(vtp.find("Name=\"negative_volume_fraction\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"interface_marker\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"root_polished\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"root_finder_iterations\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"max_root_residual\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"min_gradient_norm\""), std::string::npos);

    const std::string debug_json =
        FE::interfaces::levelSetInterfaceQuadratureDebugJsonString(result.domain, 0);
    EXPECT_NE(debug_json.find("\"parent_coordinate\""), std::string::npos);
    EXPECT_NE(debug_json.find("\"reference_measure_factor\""), std::string::npos);
    EXPECT_NE(debug_json.find("\"level_set_residual\""), std::string::npos);
    EXPECT_NE(debug_json.find("\"gradient_norm\""), std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleVolumeNormalsFollowSideConvention)
{
    constexpr int interface_marker = 94;
    const auto result = buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                                 /*level_set_order=*/2,
                                                 /*subdivision_depth=*/6,
                                                 /*interface_order=*/1,
                                                 /*volume_order=*/2,
                                                 interface_marker);

    ASSERT_TRUE(result.success) << result.diagnostic;
    std::size_t checked_negative_regions = 0u;
    std::size_t checked_positive_regions = 0u;
    for (const auto& region : result.domain.volumeRegions()) {
        if (!region.active()) {
            continue;
        }
        const FE::Real radius =
            std::sqrt(region.centroid[0] * region.centroid[0] +
                      region.centroid[1] * region.centroid[1]);
        if (radius <= FE::Real{1.0e-10}) {
            continue;
        }
        const FE::Real radial_dot =
            region.normal[0] * region.centroid[0] +
            region.normal[1] * region.centroid[1];
        if (region.side == FE::geometry::CutIntegrationSide::Negative) {
            EXPECT_GT(radial_dot, 0.0);
            ++checked_negative_regions;
        } else if (region.side == FE::geometry::CutIntegrationSide::Positive) {
            EXPECT_LT(radial_dot, 0.0);
            ++checked_positive_regions;
        }
    }

    EXPECT_GT(checked_negative_regions, 0u);
    EXPECT_GT(checked_positive_regions, 0u);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP2EllipseIntegratesAreaAndArcLength)
{
    constexpr int interface_marker = 87;
    constexpr FE::Real semi_major = 0.65;
    constexpr FE::Real semi_minor = 0.35;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] / (semi_major * semi_major) +
            x[1] * x[1] / (semi_minor * semi_minor) - 1.0;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 7;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("SayeHyperrectangle"), std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);

    const FE::Real expected_area = pi * semi_major * semi_minor;
    const FE::Real expected_length =
        ellipsePerimeterReference(semi_major, semi_minor);
    EXPECT_NEAR(result.summary.negative_volume_measure, expected_area, 2.0e-2);
    EXPECT_NEAR(result.summary.positive_volume_measure,
                4.0 - expected_area,
                2.0e-2);
    EXPECT_NEAR(result.summary.measure, expected_length, 8.0e-2);

    const auto x2_moment = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0];
    };
    const auto y2_moment = [](const std::array<FE::Real, 3>& x) {
        return x[1] * x[1];
    };
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      x2_moment),
                pi * semi_major * semi_major * semi_major * semi_minor / 4.0,
                1.5e-2);
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      y2_moment),
                pi * semi_major * semi_minor * semi_minor * semi_minor / 4.0,
                1.5e-2);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 1);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 1);

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    EXPECT_EQ(volume_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(volume_rules.front().provenance.achieved_quadrature_order, 2);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleCircleErrorsDecreaseUnderHAndPRefinement)
{
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const FE::Real expected_area = pi * radius * radius;
    const FE::Real expected_length = 2.0 * pi * radius;

    const auto coarse_h = buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                                  /*level_set_order=*/2,
                                                  /*subdivision_depth=*/3,
                                                  /*interface_order=*/1,
                                                  /*volume_order=*/2,
                                                  /*interface_marker=*/1881);
    const auto fine_h = buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                                /*level_set_order=*/2,
                                                /*subdivision_depth=*/6,
                                                /*interface_order=*/1,
                                                /*volume_order=*/2,
                                                /*interface_marker=*/1882);
    ASSERT_TRUE(coarse_h.success) << coarse_h.diagnostic;
    ASSERT_TRUE(fine_h.success) << fine_h.diagnostic;

    const FE::Real coarse_h_area_error =
        std::abs(coarse_h.summary.negative_volume_measure - expected_area);
    const FE::Real fine_h_area_error =
        std::abs(fine_h.summary.negative_volume_measure - expected_area);
    const FE::Real coarse_h_length_error =
        std::abs(coarse_h.summary.measure - expected_length);
    const FE::Real fine_h_length_error =
        std::abs(fine_h.summary.measure - expected_length);

    EXPECT_LT(fine_h_area_error, coarse_h_area_error);
    EXPECT_LT(fine_h_length_error, coarse_h_length_error);
    EXPECT_EQ(fine_h.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(fine_h.achieved_volume_quadrature_order, 2);
    EXPECT_GT(fine_h.summary.active_fragment_count,
              coarse_h.summary.active_fragment_count);

    const auto coarse_p = buildSingleQuadCircleCut(FE::ElementType::Quad4,
                                                  /*level_set_order=*/1,
                                                  /*subdivision_depth=*/6,
                                                  /*interface_order=*/1,
                                                  /*volume_order=*/1,
                                                  /*interface_marker=*/1883);
    const auto fine_p = buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                                /*level_set_order=*/2,
                                                /*subdivision_depth=*/6,
                                                /*interface_order=*/1,
                                                /*volume_order=*/2,
                                                /*interface_marker=*/1884);
    ASSERT_TRUE(coarse_p.success) << coarse_p.diagnostic;
    ASSERT_TRUE(fine_p.success) << fine_p.diagnostic;

    const FE::Real coarse_p_area_error =
        std::abs(coarse_p.summary.negative_volume_measure - expected_area);
    const FE::Real fine_p_area_error =
        std::abs(fine_p.summary.negative_volume_measure - expected_area);
    const FE::Real coarse_p_length_error =
        std::abs(coarse_p.summary.measure - expected_length);
    const FE::Real fine_p_length_error =
        std::abs(fine_p.summary.measure - expected_length);

    EXPECT_LT(fine_p_area_error, 0.25 * coarse_p_area_error);
    EXPECT_LT(fine_p_length_error, 0.25 * coarse_p_length_error);
    EXPECT_EQ(coarse_p.summary.active_fragment_count, 0u);
    EXPECT_GT(fine_p.summary.active_fragment_count, 0u);
    EXPECT_EQ(fine_p.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(fine_p.achieved_volume_quadrature_order, 2);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleEllipseErrorsDecreaseUnderHRefinement)
{
    constexpr FE::Real semi_major = 0.65;
    constexpr FE::Real semi_minor = 0.35;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const FE::Real expected_area = pi * semi_major * semi_minor;
    const FE::Real expected_length =
        ellipsePerimeterReference(semi_major, semi_minor);

    const auto coarse = buildSingleQuadEllipseCut(semi_major,
                                                  semi_minor,
                                                  /*subdivision_depth=*/3,
                                                  /*interface_order=*/1,
                                                  /*volume_order=*/2,
                                                  /*interface_marker=*/1886);
    const auto fine = buildSingleQuadEllipseCut(semi_major,
                                                semi_minor,
                                                /*subdivision_depth=*/7,
                                                /*interface_order=*/1,
                                                /*volume_order=*/2,
                                                /*interface_marker=*/1887);
    ASSERT_TRUE(coarse.success) << coarse.diagnostic;
    ASSERT_TRUE(fine.success) << fine.diagnostic;

    const FE::Real coarse_area_error =
        std::abs(coarse.summary.negative_volume_measure - expected_area);
    const FE::Real fine_area_error =
        std::abs(fine.summary.negative_volume_measure - expected_area);
    const FE::Real coarse_length_error =
        std::abs(coarse.summary.measure - expected_length);
    const FE::Real fine_length_error =
        std::abs(fine.summary.measure - expected_length);

    EXPECT_LT(fine_area_error, coarse_area_error);
    EXPECT_LT(fine_length_error, coarse_length_error);
    EXPECT_EQ(fine.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(fine.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(fine.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(fine.achieved_volume_quadrature_order, 2);
    EXPECT_GT(fine.summary.active_fragment_count,
              coarse.summary.active_fragment_count);
}

TEST(LevelSetInterfaceLifecycle, SmallVolumeCutsReturnFinitePositiveRules)
{
    constexpr FE::Real radius = 0.05;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;

    const auto saye_circle =
        buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                 /*level_set_order=*/2,
                                 /*subdivision_depth=*/7,
                                 /*interface_order=*/1,
                                 /*volume_order=*/2,
                                 /*interface_marker=*/1902,
                                 radius);
    ASSERT_TRUE(saye_circle.success) << saye_circle.diagnostic;
    EXPECT_EQ(saye_circle.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(saye_circle.implicit_cut_fallback_cell_count, 0u);
    EXPECT_GT(saye_circle.summary.negative_volume_measure, 0.0);
    EXPECT_LT(saye_circle.summary.negative_volume_measure, 0.01);
    EXPECT_NEAR(saye_circle.summary.negative_volume_measure,
                pi * radius * radius,
                2.0e-3);
    EXPECT_NEAR(saye_circle.summary.measure,
                2.0 * pi * radius,
                2.0e-2);
    expectSingleParentVolumeRulesPartitionMeasure(saye_circle.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(saye_circle.domain);

    const auto subcell_circle =
        buildSingleTriangleHighOrderSubcellCut(
            [](const std::array<FE::Real, 3>& x) {
                return x[0] * x[0] + x[1] * x[1] -
                       radius * radius;
            },
            /*interface_marker=*/1903,
            /*subdivision_depth=*/7,
            /*interface_order=*/1,
            /*volume_order=*/2);
    ASSERT_TRUE(subcell_circle.success) << subcell_circle.diagnostic;
    EXPECT_EQ(subcell_circle.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_EQ(subcell_circle.implicit_cut_fallback_cell_count, 0u);
    EXPECT_GT(subcell_circle.summary.negative_volume_measure, 0.0);
    EXPECT_LT(subcell_circle.summary.negative_volume_measure, 0.003);
    EXPECT_NEAR(subcell_circle.summary.negative_volume_measure,
                pi * radius * radius / 4.0,
                8.0e-4);
    EXPECT_NEAR(subcell_circle.summary.measure,
                pi * radius / 2.0,
                1.0e-2);
    expectSingleParentVolumeRulesPartitionMeasure(subcell_circle.domain, 1.0e-10);
    expectGeneratedCutRulesAreFinite(subcell_circle.domain);
}

TEST(LevelSetInterfaceLifecycle, MultipleComponentsInSingleParentCellProduceSeparatedRules)
{
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;

    {
        constexpr FE::Real radius = 0.35;
        const auto two_circles = [=](const std::array<FE::Real, 3>& x) {
            const FE::Real left =
                (x[0] + 1.0) * (x[0] + 1.0) +
                (x[1] + 1.0) * (x[1] + 1.0) - radius * radius;
            const FE::Real right =
                (x[0] - 1.0) * (x[0] - 1.0) +
                (x[1] - 1.0) * (x[1] - 1.0) - radius * radius;
            return left * right;
        };

        const auto mesh =
            std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
        auto scalar_space =
            FE::spaces::Space(FE::spaces::SpaceType::H1,
                              mesh,
                              /*order=*/4,
                              /*components=*/1);

        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        const auto& field_dofs = system.fieldDofHandler(phi);
        const auto cell_dofs = field_dofs.getCellDofs(0);
        const auto dof_coordinates =
            FE::basis::NodeOrdering::get_lagrange_node_coords(
                FE::ElementType::Quad4,
                4);
        ASSERT_EQ(cell_dofs.size(), dof_coordinates.size());
        const auto offset = system.fieldDofOffset(phi);
        for (std::size_t i = 0; i < dof_coordinates.size(); ++i) {
            const auto& xi = dof_coordinates[i];
            solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
                two_circles({{xi[0], xi[1], 0.0}});
        }

        level_set::LevelSetGeneratedInterfaceOptions options{};
        options.level_set_field_name = "phi";
        options.requested_interface_marker = 1904;
        options.domain_id = "water-air";
        options.geometry_mode =
            level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
        options.implicit_cut_quadrature_backend =
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
        options.implicit_cut_max_subdivision_depth = 7;
        options.interface_quadrature_order = 1;
        options.volume_quadrature_order = 2;

        level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        const auto result = lifecycle.build(system, options, solution);

        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.implicit_cut_quadrature_backend,
                  level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
        EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);
        EXPECT_GT(result.summary.active_fragment_count, 1u);
        EXPECT_NEAR(result.summary.negative_volume_measure,
                    0.5 * pi * radius * radius,
                    3.0e-2);
        EXPECT_NEAR(result.summary.measure,
                    pi * radius,
                    8.0e-2);

        const auto left_component = [](const std::array<FE::Real, 3>& x) {
            return x[0] < 0.0 ? FE::Real{1.0} : FE::Real{0.0};
        };
        const auto right_component = [](const std::array<FE::Real, 3>& x) {
            return x[0] > 0.0 ? FE::Real{1.0} : FE::Real{0.0};
        };
        EXPECT_NEAR(integrateInterfaceMoment(result.domain, left_component),
                    0.5 * pi * radius,
                    5.0e-2);
        EXPECT_NEAR(integrateInterfaceMoment(result.domain, right_component),
                    0.5 * pi * radius,
                    5.0e-2);
        expectSingleParentVolumeRulesPartitionMeasure(result.domain, 1.0e-10);
        expectGeneratedCutRulesAreFinite(result.domain);
    }

    {
        constexpr FE::Real radius = 0.18;
        const auto two_circles = [=](const std::array<FE::Real, 3>& x) {
            const FE::Real left =
                x[0] * x[0] + x[1] * x[1] - radius * radius;
            const FE::Real right =
                (x[0] - 1.0) * (x[0] - 1.0) + x[1] * x[1] -
                radius * radius;
            return left * right;
        };

        const auto mesh =
            std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle3);
        auto scalar_space =
            FE::spaces::Space(FE::spaces::SpaceType::H1,
                              mesh,
                              /*order=*/4,
                              /*components=*/1);

        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        const auto& field_dofs = system.fieldDofHandler(phi);
        const auto cell_dofs = field_dofs.getCellDofs(0);
        const auto dof_coordinates =
            FE::basis::NodeOrdering::get_lagrange_node_coords(
                FE::ElementType::Triangle3,
                4);
        ASSERT_EQ(cell_dofs.size(), dof_coordinates.size());
        const auto offset = system.fieldDofOffset(phi);
        for (std::size_t i = 0; i < dof_coordinates.size(); ++i) {
            const auto& xi = dof_coordinates[i];
            solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
                two_circles({{xi[0], xi[1], 0.0}});
        }

        level_set::LevelSetGeneratedInterfaceOptions options{};
        options.level_set_field_name = "phi";
        options.requested_interface_marker = 1905;
        options.domain_id = "water-air";
        options.geometry_mode =
            level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
        options.implicit_cut_quadrature_backend =
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
        options.implicit_cut_max_subdivision_depth = 7;
        options.interface_quadrature_order = 1;
        options.volume_quadrature_order = 2;

        level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        const auto result = lifecycle.build(system, options, solution);

        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.implicit_cut_quadrature_backend,
                  level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
        EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);
        EXPECT_GT(result.summary.active_fragment_count, 1u);
        EXPECT_NEAR(result.summary.negative_volume_measure,
                    3.0 * pi * radius * radius / 8.0,
                    1.5e-2);
        EXPECT_NEAR(result.summary.measure,
                    3.0 * pi * radius / 4.0,
                    8.0e-2);

        const auto left_component = [](const std::array<FE::Real, 3>& x) {
            return x[0] < 0.5 ? FE::Real{1.0} : FE::Real{0.0};
        };
        const auto right_component = [](const std::array<FE::Real, 3>& x) {
            return x[0] > 0.5 ? FE::Real{1.0} : FE::Real{0.0};
        };
        EXPECT_NEAR(integrateInterfaceMoment(result.domain, left_component),
                    0.5 * pi * radius,
                    5.0e-2);
        EXPECT_NEAR(integrateInterfaceMoment(result.domain, right_component),
                    0.25 * pi * radius,
                    5.0e-2);
        expectSingleParentVolumeRulesPartitionMeasure(result.domain, 1.0e-10);
        expectGeneratedCutRulesAreFinite(result.domain);
    }
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleNearTangentCircleReturnsRulesOrDiagnostic)
{
    const auto result = buildSingleQuadCircleCut(FE::ElementType::Quad9,
                                                /*level_set_order=*/2,
                                                /*subdivision_depth=*/7,
                                                /*interface_order=*/1,
                                                /*volume_order=*/2,
                                                /*interface_marker=*/1885,
                                                /*radius=*/1.0);

    EXPECT_NE(result.diagnostic.find("status="), std::string::npos);
    if (!result.success) {
        const bool explicit_diagnostic =
            result.diagnostic.find("Tangent") != std::string::npos ||
            result.diagnostic.find("Fallback") != std::string::npos ||
            result.diagnostic.find("Failed") != std::string::npos ||
            result.diagnostic.find("Degenerate") != std::string::npos;
        EXPECT_TRUE(explicit_diagnostic) << result.diagnostic;
        return;
    }

    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    ASSERT_FALSE(volume_rules.empty());
    for (const auto& rule : interface_rules) {
        EXPECT_GT(rule.measure, 0.0);
        EXPECT_FALSE(rule.points.empty());
        EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                  "SayeHyperrectangle");
        for (const auto& point : rule.points) {
            EXPECT_TRUE(std::isfinite(point.weight));
            EXPECT_GT(point.weight, 0.0);
            EXPECT_TRUE(std::isfinite(point.point[0]));
            EXPECT_TRUE(std::isfinite(point.point[1]));
            EXPECT_TRUE(std::isfinite(point.normal[0]));
            EXPECT_TRUE(std::isfinite(point.normal[1]));
        }
    }
    for (const auto& rule : volume_rules) {
        EXPECT_GE(rule.measure, 0.0);
        EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                  "SayeHyperrectangle");
        for (const auto& point : rule.points) {
            EXPECT_TRUE(std::isfinite(point.weight));
            EXPECT_GT(point.weight, 0.0);
            EXPECT_TRUE(std::isfinite(point.point[0]));
            EXPECT_TRUE(std::isfinite(point.point[1]));
            EXPECT_TRUE(std::isfinite(point.normal[0]));
            EXPECT_TRUE(std::isfinite(point.normal[1]));
        }
    }
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP2SphereApproximatesVolumeAndArea)
{
    constexpr int interface_marker = 92;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleHexMeshAccess>(FE::ElementType::Hex27);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleHexSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 27u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 27u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 4;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;
    options.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::LinearCorner;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 1u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("SayeHyperrectangle"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("max_depth_reached="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("linearized_leaves="), std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure,
                4.0 * pi * radius * radius * radius / 3.0,
                8.0e-2);
    EXPECT_NEAR(result.summary.measure,
                4.0 * pi * radius * radius,
                2.5e-1);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    EXPECT_EQ(interface_rules.front().provenance.implicit_quadrature_backend,
              "SayeHyperrectangle");
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 1);

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    EXPECT_EQ(volume_rules.front().provenance.implicit_quadrature_backend,
              "SayeHyperrectangle");
    EXPECT_EQ(volume_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(volume_rules.front().provenance.achieved_quadrature_order, 2);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP2EllipsoidApproximatesVolumeAndArea)
{
    constexpr int interface_marker = 93;
    constexpr FE::Real axis_x = 0.65;
    constexpr FE::Real axis_y = 0.4;
    constexpr FE::Real axis_z = axis_y;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleHexMeshAccess>(FE::ElementType::Hex27);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleHexSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 27u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 27u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            (x[0] * x[0]) / (axis_x * axis_x) +
            (x[1] * x[1]) / (axis_y * axis_y) +
            (x[2] * x[2]) / (axis_z * axis_z) - FE::Real{1.0};
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 5;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;
    options.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::LinearCorner;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    const FE::Real eccentricity =
        std::sqrt(FE::Real{1.0} - (axis_y * axis_y) / (axis_x * axis_x));
    const FE::Real expected_volume =
        FE::Real{4.0} * pi * axis_x * axis_y * axis_z / FE::Real{3.0};
    const FE::Real expected_area =
        FE::Real{2.0} * pi * axis_y * axis_y *
        (FE::Real{1.0} +
         axis_x * std::asin(eccentricity) / (axis_y * eccentricity));

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 1u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("SayeHyperrectangle"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("max_depth_reached="), std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure, expected_volume, 5.0e-2);
    EXPECT_NEAR(result.summary.measure, expected_area, 2.0e-1);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleSphereAndEllipsoidErrorsDecreaseUnderHRefinement)
{
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;

    {
        constexpr FE::Real radius = 0.5;
        const FE::Real expected_volume =
            FE::Real{4.0} * pi * radius * radius * radius / FE::Real{3.0};
        const FE::Real expected_area = FE::Real{4.0} * pi * radius * radius;
        const auto sphere = [](const std::array<FE::Real, 3>& x) {
            return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] -
                   radius * radius;
        };

        const auto coarse = buildSingleHexSayeCut(sphere,
                                                  /*subdivision_depth=*/3,
                                                  /*interface_order=*/2,
                                                  /*volume_order=*/2,
                                                  /*interface_marker=*/1888);
        const auto fine = buildSingleHexSayeCut(sphere,
                                                /*subdivision_depth=*/5,
                                                /*interface_order=*/2,
                                                /*volume_order=*/2,
                                                /*interface_marker=*/1889);
        ASSERT_TRUE(coarse.success) << coarse.diagnostic;
        ASSERT_TRUE(fine.success) << fine.diagnostic;

        const FE::Real coarse_volume_error =
            std::abs(coarse.summary.negative_volume_measure - expected_volume);
        const FE::Real fine_volume_error =
            std::abs(fine.summary.negative_volume_measure - expected_volume);
        const FE::Real coarse_area_error =
            std::abs(coarse.summary.measure - expected_area);
        const FE::Real fine_area_error =
            std::abs(fine.summary.measure - expected_area);

        EXPECT_LT(fine_volume_error, coarse_volume_error);
        EXPECT_LT(fine_area_error, coarse_area_error);
        EXPECT_EQ(fine.implicit_cut_quadrature_backend,
                  level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
        EXPECT_LE(fine.implicit_cut_fallback_cell_count,
                  coarse.implicit_cut_fallback_cell_count);
        EXPECT_EQ(fine.achieved_interface_quadrature_order, 1);
        EXPECT_EQ(fine.achieved_volume_quadrature_order, 2);
        EXPECT_GT(fine.summary.active_fragment_count,
                  coarse.summary.active_fragment_count);
    }

    {
        constexpr FE::Real axis_x = 0.65;
        constexpr FE::Real axis_y = 0.4;
        constexpr FE::Real axis_z = axis_y;
        const FE::Real expected_volume =
            FE::Real{4.0} * pi * axis_x * axis_y * axis_z / FE::Real{3.0};
        const FE::Real expected_area =
            prolateSpheroidSurfaceArea(axis_x, axis_y);
        const auto ellipsoid = [](const std::array<FE::Real, 3>& x) {
            return (x[0] * x[0]) / (axis_x * axis_x) +
                   (x[1] * x[1]) / (axis_y * axis_y) +
                   (x[2] * x[2]) / (axis_z * axis_z) - FE::Real{1.0};
        };

        const auto coarse = buildSingleHexSayeCut(ellipsoid,
                                                  /*subdivision_depth=*/2,
                                                  /*interface_order=*/2,
                                                  /*volume_order=*/2,
                                                  /*interface_marker=*/1900);
        const auto fine = buildSingleHexSayeCut(ellipsoid,
                                                /*subdivision_depth=*/5,
                                                /*interface_order=*/2,
                                                /*volume_order=*/2,
                                                /*interface_marker=*/1901);
        ASSERT_TRUE(coarse.success) << coarse.diagnostic;
        ASSERT_TRUE(fine.success) << fine.diagnostic;

        const FE::Real coarse_volume_error =
            std::abs(coarse.summary.negative_volume_measure - expected_volume);
        const FE::Real fine_volume_error =
            std::abs(fine.summary.negative_volume_measure - expected_volume);
        const FE::Real coarse_area_error =
            std::abs(coarse.summary.measure - expected_area);
        const FE::Real fine_area_error =
            std::abs(fine.summary.measure - expected_area);

        EXPECT_LT(fine_volume_error, coarse_volume_error);
        EXPECT_LT(fine_area_error, coarse_area_error);
        EXPECT_EQ(fine.implicit_cut_quadrature_backend,
                  level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
        EXPECT_LE(fine.implicit_cut_fallback_cell_count,
                  coarse.implicit_cut_fallback_cell_count);
        EXPECT_EQ(fine.achieved_interface_quadrature_order, 1);
        EXPECT_EQ(fine.achieved_volume_quadrature_order, 2);
        EXPECT_GT(fine.summary.active_fragment_count,
                  coarse.summary.active_fragment_count);
    }
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleRulesAssembleFixedGeometryMeasures)
{
    constexpr int interface_marker = 86;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 6;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;

    FE::assembly::CutIntegrationContext context;
    ASSERT_NO_THROW(context.addGeneratedInterfaceDomain(result.domain));

    CutMeasureAssemblyKernel kernel;
    const auto build_context =
        [](const FE::assembly::CutRuleAssemblyRequest& request,
           FE::assembly::AssemblyContext& ctx) {
            ASSERT_NE(request.rule, nullptr);
            populateMeasureAssemblyContext(*request.rule, ctx);
        };

    FE::assembly::CutDomainAssemblyOptions volume_options;
    volume_options.include_interface_rules = false;
    volume_options.volume_marker = interface_marker;
    volume_options.volume_side = FE::geometry::CutIntegrationSide::Negative;
    const auto volume_summary =
        FE::assembly::assembleCutDomains(
            context, kernel, build_context, volume_options);

    ASSERT_GT(volume_summary.volume_rule_count, 0u);
    ASSERT_TRUE(volume_summary.hasVector());
    ASSERT_EQ(volume_summary.total_output.local_vector.size(), 1u);
    EXPECT_NEAR(volume_summary.total_output.local_vector[0],
                result.summary.negative_volume_measure,
                1.0e-12);
    EXPECT_NEAR(volume_summary.total_output.local_vector[0],
                pi * radius * radius,
                6.0e-2);

    FE::assembly::CutDomainAssemblyOptions interface_options;
    interface_options.include_volume_rules = false;
    interface_options.interface_marker = interface_marker;
    const auto interface_summary =
        FE::assembly::assembleCutDomains(
            context, kernel, build_context, interface_options);

    ASSERT_GT(interface_summary.interface_rule_count, 0u);
    ASSERT_TRUE(interface_summary.hasVector());
    ASSERT_EQ(interface_summary.total_output.local_vector.size(), 1u);
    EXPECT_NEAR(interface_summary.total_output.local_vector[0],
                result.summary.measure,
                1.0e-12);
    EXPECT_NEAR(interface_summary.total_output.local_vector[0],
                2.0 * pi * radius,
                1.2e-1);
}

TEST(LevelSetInterfaceLifecycle, BackendFactoryReturnsFailClosedDrivers)
{
    EXPECT_NO_THROW(
        (void)level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle));
    EXPECT_NO_THROW(
        (void)level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell));
    EXPECT_NO_THROW(
        (void)level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::Auto));
    const auto& moment_fit =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::MomentFit);
    EXPECT_EQ(moment_fit.kind(),
              level_set::ImplicitCutQuadratureBackend::MomentFit);
    EXPECT_STREQ(moment_fit.name(), "MomentFit");
    EXPECT_FALSE(moment_fit.supports(2, FE::ElementType::Quad9));
}

TEST(LevelSetInterfaceLifecycle, MomentFitBackendReportsUnavailableCapability)
{
    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "moment-fit-level-set", 0, 1);
    request.interface_marker = 905;
    request.interface_quadrature_order = 3;
    request.volume_quadrature_order = 3;

    level_set::ImplicitCutQuadratureBackendCellInput input{};
    input.linearized_input.parent_cell = 7;
    input.linearized_input.element_type = FE::ElementType::Quad9;
    input.linearized_input.node_coordinates = {
        std::array<FE::Real, 3>{-1.0, -1.0, 0.0},
        std::array<FE::Real, 3>{1.0, -1.0, 0.0},
        std::array<FE::Real, 3>{1.0, 1.0, 0.0},
        std::array<FE::Real, 3>{-1.0, 1.0, 0.0},
    };
    input.linearized_input.level_set_values = {-1.0, 1.0, 1.0, -1.0};

    const auto& backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::MomentFit);
    const auto result = backend.cut(/*mesh_dimension=*/2, request, input);

    EXPECT_FALSE(result.cut.supported);
    EXPECT_EQ(result.diagnostic_status,
              level_set::ImplicitCutQuadratureDiagnosticStatus::Unsupported);
    EXPECT_EQ(result.requested_interface_quadrature_order, 3);
    EXPECT_EQ(result.requested_volume_quadrature_order, 3);
    EXPECT_EQ(result.possible_interface_quadrature_order, -1);
    EXPECT_EQ(result.possible_volume_quadrature_order, -1);
    EXPECT_EQ(result.achieved_interface_quadrature_order, -1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, -1);
    EXPECT_EQ(result.verified_interface_quadrature_order, -1);
    EXPECT_EQ(result.verified_volume_quadrature_order, -1);
    EXPECT_EQ(result.volume_quadrature_point_count, 0u);
    EXPECT_EQ(result.interface_quadrature_point_count, 0u);
    EXPECT_GE(result.backend_elapsed_seconds, 0.0);
    EXPECT_EQ(result.selected_backend,
              level_set::ImplicitCutQuadratureBackend::MomentFit);
    EXPECT_NE(result.cut.diagnostic.find("capability_state=Unavailable"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("requested_interface_order=3"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("possible_interface_order=-1"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("verified_volume_order=-1"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("production_qualified=false"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("positive-weight policy"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("fallback_reason=MomentFit backend unavailable"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("root_polishing=not_performed"),
              std::string::npos);
    EXPECT_EQ(result.cut.diagnostic.find("root_finder_iterations=0"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("high_order_downgrade=false"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("volume_quadrature_points=0"),
              std::string::npos);
    EXPECT_NE(result.cut.diagnostic.find("backend_elapsed_seconds="),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, MomentFitConfigurationFailsClosedWithCapabilityDiagnostic)
{
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - FE::Real{0.25};
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::MomentFit;
    options.interface_quadrature_order = 3;
    options.volume_quadrature_order = 3;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected unavailable MomentFit backend to fail closed";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("backend=MomentFit"), std::string::npos);
        EXPECT_NE(message.find("capability_state=Unavailable"), std::string::npos);
        EXPECT_NE(message.find("production_qualified=false"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, RejectsNonH1LevelSetField)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::L2, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected non-H1 level-set field to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("H1/C0"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, FullSideVolumeRegionSucceedsWithoutInterfaceFragment)
{
    constexpr int interface_marker = 76;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(solution, system, phi, vertex, FE::Real(-1.0));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water";
    options.tolerance = 1.0e-12;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.summary.active_fragment_count, 0u);
    EXPECT_EQ(result.summary.active_volume_region_count, 1u);
    EXPECT_GT(result.summary.negative_volume_measure, 0.0);
    EXPECT_EQ(result.summary.positive_volume_measure, 0.0);
}

TEST(LevelSetInterfaceLifecycle, GeneratedRulesUseReferenceCoordinatesOnPhysicalCells)
{
    constexpr int interface_marker = 79;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>(
        std::vector<std::array<FE::Real, 3>>{
            std::array<FE::Real, 3>{10.0, 20.0, 30.0},
            std::array<FE::Real, 3>{12.0, 20.0, 30.0},
            std::array<FE::Real, 3>{10.0, 23.0, 30.0},
            std::array<FE::Real, 3>{10.0, 20.0, 34.0},
        });
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    setFieldComponentValue(solution, system, phi, 0, -0.5);
    setFieldComponentValue(solution, system, phi, 1, 0.5);
    setFieldComponentValue(solution, system, phi, 2, 0.5);
    setFieldComponentValue(solution, system, phi, 3, 0.5);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.tolerance = 1.0e-12;
    options.interface_quadrature_order = 0;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 1u);
    EXPECT_EQ(interface_rules.front().frame, FE::geometry::CutGeometryFrame::Reference);
    for (const auto& qp : interface_rules.front().points) {
        EXPECT_GE(qp.point[0], 0.0);
        EXPECT_GE(qp.point[1], 0.0);
        EXPECT_GE(qp.point[2], 0.0);
        EXPECT_LE(qp.point[0] + qp.point[1] + qp.point[2], 1.0);
    }

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 2u);
    FE::Real measure = 0.0;
    for (const auto& rule : volume_rules) {
        EXPECT_EQ(rule.frame, FE::geometry::CutGeometryFrame::Reference);
        measure += rule.measure;
        for (const auto& qp : rule.points) {
            EXPECT_GE(qp.point[0], 0.0);
            EXPECT_GE(qp.point[1], 0.0);
            EXPECT_GE(qp.point[2], 0.0);
            EXPECT_LE(qp.point[0] + qp.point[1] + qp.point[2], 1.0);
        }
    }
    EXPECT_NEAR(measure, FE::Real(1.0) / FE::Real(6.0), 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleReferenceRulesMapToCurvedHexGeometry)
{
    constexpr int interface_marker = 94;
    constexpr FE::Real sx = 1.25;
    constexpr FE::Real sy = 0.75;
    constexpr FE::Real sz = 1.5;
    constexpr FE::Real cut_zeta = 0.125;
    constexpr FE::Real physical_jacobian = sx * sy * sz;
    constexpr FE::Real reference_negative_volume = 4.0 * (1.0 + cut_zeta);
    constexpr FE::Real reference_positive_volume = 4.0 * (1.0 - cut_zeta);
    const auto mesh = std::make_shared<SingleHexMeshAccess>(FE::ElementType::Hex27);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleHexSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 27u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 27u; ++i) {
        const auto xi =
            FE::basis::NodeOrdering::get_node_coords(FE::ElementType::Hex27, i);
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            xi[2] - cut_zeta;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 3;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_NEAR(result.summary.negative_volume_measure,
                reference_negative_volume,
                1.0e-12);
    ASSERT_NEAR(result.summary.positive_volume_measure,
                reference_positive_volume,
                1.0e-12);
    ASSERT_NEAR(result.summary.measure, 4.0, 1.0e-12);

    const auto mapping = makeCurvedHex27Mapping();
    ASSERT_FALSE(mapping->isAffine());

    FE::Real physical_negative_volume = 0.0;
    FE::Real physical_positive_volume = 0.0;
    for (const auto& rule : result.domain.volumeQuadratureRules()) {
        EXPECT_EQ(rule.frame, FE::geometry::CutGeometryFrame::Reference);
        if (rule.side == FE::geometry::CutIntegrationSide::Negative) {
            physical_negative_volume += mappedVolumeMeasure(rule, *mapping);
        } else if (rule.side == FE::geometry::CutIntegrationSide::Positive) {
            physical_positive_volume += mappedVolumeMeasure(rule, *mapping);
        }
    }
    EXPECT_NEAR(physical_negative_volume,
                physical_jacobian * reference_negative_volume,
                1.0e-10);
    EXPECT_NEAR(physical_positive_volume,
                physical_jacobian * reference_positive_volume,
                1.0e-10);

    FE::Real physical_interface_measure = 0.0;
    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    for (const auto& rule : interface_rules) {
        EXPECT_EQ(rule.frame, FE::geometry::CutGeometryFrame::Reference);
        EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                  "SayeHyperrectangle");
        EXPECT_EQ(rule.provenance.requested_quadrature_order, 1);
        EXPECT_EQ(rule.provenance.achieved_quadrature_order, 1);
        physical_interface_measure += mappedInterfaceMeasure(rule, *mapping);
    }
    EXPECT_NE(physical_interface_measure, result.summary.measure);
    EXPECT_GT(physical_interface_measure, sx * sy * 4.0);
    EXPECT_NEAR(physical_interface_measure, expectedCurvedHexMidplaneArea(), 5.0e-3);
}

TEST(LevelSetInterfaceLifecycle, PreservesMarkerIdentity)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto make_solution = [&](FE::Real offset) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            const auto x = mesh->getNodeCoordinates(vertex);
            setFieldComponentValue(solution, system, phi, vertex,
                                   x[0] + x[1] + x[2] - offset);
        }
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto initial = lifecycle.build(system, options, make_solution(0.5));
    const auto updated = lifecycle.build(system, options, make_solution(0.75));

    ASSERT_TRUE(initial.success) << initial.diagnostic;
    ASSERT_TRUE(updated.success) << updated.diagnostic;
    EXPECT_GE(initial.interface_marker, 1000000);
    EXPECT_EQ(initial.interface_marker, updated.interface_marker);
    EXPECT_EQ(initial.domain.marker(), updated.domain.marker());
    EXPECT_EQ(updated.value_revision, initial.value_revision + 1u);
}

TEST(LevelSetInterfaceLifecycle, UpdatesGeometryAfterFieldChange)
{
    constexpr int interface_marker = 74;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto make_solution = [&](FE::Real offset) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            const auto x = mesh->getNodeCoordinates(vertex);
            setFieldComponentValue(solution, system, phi, vertex,
                                   x[0] + x[1] + x[2] - offset);
        }
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto initial = lifecycle.build(system, options, make_solution(0.5));
    const auto updated = lifecycle.build(system, options, make_solution(0.75));

    ASSERT_TRUE(initial.success) << initial.diagnostic;
    ASSERT_TRUE(updated.success) << updated.diagnostic;
    EXPECT_EQ(initial.value_revision, 1u);
    EXPECT_EQ(updated.value_revision, 2u);
    EXPECT_GT(initial.summary.measure, 0.0);
    EXPECT_GT(updated.summary.measure, 0.0);
    EXPECT_NE(initial.summary.measure, updated.summary.measure);
    ASSERT_EQ(initial.domain.fragments().size(), 1u);
    ASSERT_EQ(updated.domain.fragments().size(), 1u);
    EXPECT_NE(initial.domain.fragments().front().stable_id,
              updated.domain.fragments().front().stable_id);
}

TEST(LevelSetInterfaceLifecycle, CornerLinearizedHighOrderGeometryIsRejectedByDefault)
{
    constexpr int interface_marker = 81;
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected high-order generated interface geometry to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("corner-linearize"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, CornerLinearizedHighOrderGeometryReportsDiagnosticsWhenAllowed)
{
    constexpr int interface_marker = 82;
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.allow_corner_linearized_geometry = true;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cell_count, 1u);
    EXPECT_EQ(result.corner_linearized_cell_count, 1u);
    EXPECT_EQ(result.max_cell_node_count, 10u);
    EXPECT_EQ(result.max_corner_node_count, 4u);
    EXPECT_EQ(result.interface_marker, interface_marker);
}
