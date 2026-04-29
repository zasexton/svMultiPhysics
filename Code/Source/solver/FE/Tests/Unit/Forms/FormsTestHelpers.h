#ifndef SVMP_FE_TESTS_UNIT_FORMS_FORMS_TEST_HELPERS_H
#define SVMP_FE_TESTS_UNIT_FORMS_FORMS_TEST_HELPERS_H

#include "Assembly/Assembler.h"
#include "Basis/NodeOrderingConventions.h"
#include "Core/Types.h"
#include "Dofs/DofMap.h"
#include "Geometry/FrameGeometry.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

class SingleTetraMeshAccess final : public assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame /*frame*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 4> cell_{};
};

class CurvedTetra10MeshAccess final : public assembly::IMeshAccess {
public:
    CurvedTetra10MeshAccess()
    {
        const auto n_nodes = basis::NodeOrdering::num_nodes(ElementType::Tetra10);
        nodes_.reserve(n_nodes);
        cell_.reserve(n_nodes);
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto xi = basis::NodeOrdering::get_node_coords(ElementType::Tetra10, i);
            std::array<Real, 3> x{xi[0], xi[1], xi[2]};
            if (i == 4) x[2] += Real(0.030);
            if (i == 5) x[0] += Real(0.020);
            if (i == 6) x[1] -= Real(0.020);
            if (i == 7) x[1] += Real(0.025);
            if (i == 8) x[2] += Real(0.020);
            if (i == 9) x[0] -= Real(0.015);
            nodes_.push_back(x);
            cell_.push_back(static_cast<GlobalIndex>(i));
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra10; }
    [[nodiscard]] int getCellGeometryOrder(GlobalIndex) const override { return 2; }

    void getCellNodes(GlobalIndex, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<GlobalIndex> cell_;
};

class CurvedSingleCellMeshAccess final : public assembly::IMeshAccess {
public:
    CurvedSingleCellMeshAccess(ElementType cell_type, std::string name = {})
        : cell_type_(cell_type),
          geometry_order_(geometry::defaultGeometryOrder(cell_type)),
          dimension_(element_dimension(cell_type)),
          name_(std::move(name))
    {
        const auto n_nodes = basis::NodeOrdering::num_nodes(cell_type_);
        nodes_.reserve(n_nodes);
        cell_.reserve(n_nodes);
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto xi = basis::NodeOrdering::get_node_coords(cell_type_, i);
            nodes_.push_back(curvedNode(xi));
            cell_.push_back(static_cast<GlobalIndex>(i));
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return dimension_; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return cell_type_; }
    [[nodiscard]] int getCellGeometryOrder(GlobalIndex) const override { return geometry_order_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    void getCellNodes(GlobalIndex, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    [[nodiscard]] std::array<Real, 3> curvedNode(const math::Vector<Real, 3>& xi) const
    {
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];

        if (dimension_ == 1) {
            return {x,
                    Real(0.040) * (Real(1) - x * x),
                    Real(0.012) * x};
        }
        if (dimension_ == 2) {
            return {x + Real(0.035) * x * y,
                    y + Real(0.025) * x * x,
                    Real(0.030) * (x * x - y * y + Real(0.5) * x * y)};
        }
        return {x + Real(0.030) * x * y + Real(0.018) * z * z,
                y + Real(0.024) * y * z + Real(0.014) * x * x,
                z + Real(0.020) * x * z + Real(0.016) * y * y};
    }

    ElementType cell_type_{ElementType::Unknown};
    int geometry_order_{1};
    int dimension_{0};
    std::string name_;
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<GlobalIndex> cell_;
};

class SingleTetraOneBoundaryFaceMeshAccess final : public assembly::IMeshAccess {
public:
    explicit SingleTetraOneBoundaryFaceMeshAccess(int boundary_marker)
        : boundary_marker_(boundary_marker)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame /*frame*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        // Face 0 for Tetra4 is the {0,1,2} triangle.
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return boundary_marker_; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker < 0 || marker == boundary_marker_) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int boundary_marker_{-1};
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 4> cell_{};
};

class SingleTetraTwoBoundaryFaceMeshAccess final : public assembly::IMeshAccess {
public:
    SingleTetraTwoBoundaryFaceMeshAccess(int marker_face0, int marker_face1)
        : markers_{marker_face0, marker_face1}
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex /*cell_id*/) const override
    {
        switch (face_id) {
        case 0: return 0;  // {0,1,2}
        case 1: return 1;  // {0,1,3}
        default: return 0;
        }
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return markers_.at(static_cast<std::size_t>(face_id));
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        for (GlobalIndex face_id = 0; face_id < 2; ++face_id) {
            if (marker < 0 || marker == markers_[static_cast<std::size_t>(face_id)]) {
                callback(face_id, 0);
            }
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::array<int, 2> markers_{-1, -1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<GlobalIndex, 4> cell_{};
};

class TwoTetraSharedFaceMeshAccess final : public assembly::IMeshAccess {
public:
    TwoTetraSharedFaceMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0}   // 4
        };
        cells_ = {
            {0, 1, 2, 3},  // cell 0
            {1, 2, 3, 4}   // cell 1
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame /*frame*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        // Interior face 0 is shared between:
        // - cell 0 local face 2 (nodes {1,2,3})
        // - cell 1 local face 0 (nodes {0,1,2} in cell-local indexing => global {1,2,3})
        if (face_id == 0 && cell_id == 0) return 2;
        if (face_id == 0 && cell_id == 1) return 0;
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override {
        if (face_id == 0) return {0, 1};
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(0, 0, 1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<std::array<GlobalIndex, 4>> cells_;
};

class TwoTetraSharedFacePermutedPlusMeshAccess final : public assembly::IMeshAccess {
public:
    TwoTetraSharedFacePermutedPlusMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0}   // 4
        };
        cells_ = {
            {0, 1, 2, 3},  // cell 0
            {2, 3, 1, 4}   // cell 1 (shared face nodes {2,3,1} permuted)
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        // Interior face 0 is shared between:
        // - cell 0 local face 2 (nodes {1,2,3})
        // - cell 1 local face 0 (nodes {0,1,2} in cell-local indexing => global {2,3,1})
        if (face_id == 0 && cell_id == 0) return 2;
        if (face_id == 0 && cell_id == 1) return 0;
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override {
        if (face_id == 0) return {0, 1};
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(0, 0, 1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<std::array<GlobalIndex, 4>> cells_;
};

class TwoCellMixedTypeMeshAccess final : public assembly::IMeshAccess {
public:
    TwoCellMixedTypeMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 0.0},  // 4
            {1.0, 0.0, 1.0},  // 5
            {0.0, 1.0, 1.0},  // 6
            {1.0, 1.0, 1.0}   // 7
        };
        tetra_ = {0, 1, 2, 3};
        hex_ = {0, 1, 4, 2, 3, 5, 7, 6};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override
    {
        return (cell_id == 0) ? ElementType::Tetra4 : ElementType::Hex8;
    }

    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override
    {
        // Domain IDs:
        //  - Cell 0 (Tetra4) => 1
        //  - Cell 1 (Hex8)   => 2
        return (cell_id == 0) ? 1 : 2;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        if (cell_id == 0) {
            nodes.assign(tetra_.begin(), tetra_.end());
        } else {
            nodes.assign(hex_.begin(), hex_.end());
        }
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(nodes[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 4> tetra_{};
    std::array<GlobalIndex, 8> hex_{};
};

class SingleTriangleMeshAccess final : public assembly::IMeshAccess {
public:
    SingleTriangleMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0}   // 2
        };
        cell_ = {0, 1, 2};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Triangle3;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 3> cell_{};
};

inline dofs::DofMap createSingleTetraDofMap()
{
    dofs::DofMap dof_map(1, 4, 4);
    std::vector<GlobalIndex> cell_dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();
    return dof_map;
}

inline dofs::DofMap createSingleTriangleDofMap()
{
    dofs::DofMap dof_map(1, 3, 3);
    std::vector<GlobalIndex> cell_dofs = {0, 1, 2};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(3);
    dof_map.setNumLocalDofs(3);
    dof_map.finalize();
    return dof_map;
}

inline dofs::DofMap createTwoTetraDG_DofMap()
{
    dofs::DofMap dof_map(2, 8, 4);
    std::vector<GlobalIndex> cell0 = {0, 1, 2, 3};
    std::vector<GlobalIndex> cell1 = {4, 5, 6, 7};
    dof_map.setCellDofs(0, cell0);
    dof_map.setCellDofs(1, cell1);
    dof_map.setNumDofs(8);
    dof_map.setNumLocalDofs(8);
    dof_map.finalize();
    return dof_map;
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TESTS_UNIT_FORMS_FORMS_TEST_HELPERS_H
