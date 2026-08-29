#include "LevelSet/LevelSetCurvatureProjection.h"

#include "Assembly/Assembler.h"
#include "Basis/NodeOrderingConventions.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class StructuredQuadMeshAccess final : public FE::assembly::IMeshAccess {
public:
    StructuredQuadMeshAccess(int nx, int ny, FE::Real h)
        : StructuredQuadMeshAccess(nx, ny, h, h)
    {
    }

    StructuredQuadMeshAccess(int nx,
                             int ny,
                             FE::Real hx,
                             FE::Real hy)
        : nx_(nx), ny_(ny), hx_(hx), hy_(hy)
    {
        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                nodes_.push_back({{
                    (static_cast<FE::Real>(i) -
                     static_cast<FE::Real>(nx_) * FE::Real{0.5}) * hx_,
                    (static_cast<FE::Real>(j) -
                     static_cast<FE::Real>(ny_) * FE::Real{0.5}) * hy_,
                    FE::Real{0.0},
                }});
            }
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(nx_ * ny_);
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(nodes_.size());
    }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override
    {
        return geometry_revision_;
    }
    [[nodiscard]] std::uint64_t topologyRevision() const override
    {
        return topology_revision_;
    }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return 1u;
    }
    [[nodiscard]] std::uint64_t numberingRevision() const override
    {
        return 1u;
    }
    [[nodiscard]] std::uint64_t coordinateConfigurationKey() const override
    {
        return 1u;
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex) const override { return true; }
    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override
    {
        return FE::ElementType::Quad4;
    }

    void translate(FE::Real dx, FE::Real dy)
    {
        for (auto& node : nodes_) {
            node[0] += dx;
            node[1] += dy;
        }
        ++geometry_revision_;
    }

    void rotate(FE::Real angle)
    {
        const FE::Real cosine = std::cos(angle);
        const FE::Real sine = std::sin(angle);
        for (auto& node : nodes_) {
            const FE::Real x = node[0];
            const FE::Real y = node[1];
            node[0] = cosine * x - sine * y;
            node[1] = sine * x + cosine * y;
        }
        ++geometry_revision_;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const int cell = static_cast<int>(cell_id);
        if (cell < 0 || cell >= nx_ * ny_) {
            throw std::out_of_range("invalid cell");
        }
        const int i = cell % nx_;
        const int j = cell / nx_;
        nodes = {
            vertex(i, j),
            vertex(i + 1, j),
            vertex(i + 1, j + 1),
            vertex(i, j + 1),
        };
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
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex,
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

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }
    void forEachBoundaryFace(
        int,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)>) const override
    {
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>)
        const override
    {
    }

private:
    [[nodiscard]] FE::GlobalIndex vertex(int i, int j) const
    {
        return static_cast<FE::GlobalIndex>(j * (nx_ + 1) + i);
    }

    int nx_{0};
    int ny_{0};
    FE::Real hx_{1.0};
    FE::Real hy_{1.0};
    std::uint64_t geometry_revision_{1u};
    std::uint64_t topology_revision_{1u};
    std::vector<std::array<FE::Real, 3>> nodes_{};
};

class StructuredHexMeshAccess final : public FE::assembly::IMeshAccess {
public:
    StructuredHexMeshAccess(int nx, int ny, int nz, FE::Real h)
        : nx_(nx), ny_(ny), nz_(nz), h_(h)
    {
        for (int k = 0; k <= nz_; ++k) {
            for (int j = 0; j <= ny_; ++j) {
                for (int i = 0; i <= nx_; ++i) {
                    nodes_.push_back({{
                        (static_cast<FE::Real>(i) -
                         static_cast<FE::Real>(nx_) * FE::Real{0.5}) * h_,
                        (static_cast<FE::Real>(j) -
                         static_cast<FE::Real>(ny_) * FE::Real{0.5}) * h_,
                        (static_cast<FE::Real>(k) -
                         static_cast<FE::Real>(nz_) * FE::Real{0.5}) * h_,
                    }});
                }
            }
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(nx_ * ny_ * nz_);
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(nodes_.size());
    }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex) const override { return true; }
    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override
    {
        return FE::ElementType::Hex8;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const int cell = static_cast<int>(cell_id);
        if (cell < 0 || cell >= nx_ * ny_ * nz_) {
            throw std::out_of_range("invalid cell");
        }
        const int i = cell % nx_;
        const int j = (cell / nx_) % ny_;
        const int k = cell / (nx_ * ny_);
        nodes = {
            vertex(i, j, k),
            vertex(i + 1, j, k),
            vertex(i + 1, j + 1, k),
            vertex(i, j + 1, k),
            vertex(i, j, k + 1),
            vertex(i + 1, j, k + 1),
            vertex(i + 1, j + 1, k + 1),
            vertex(i, j + 1, k + 1),
        };
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
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex,
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

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }
    void forEachBoundaryFace(
        int,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)>) const override
    {
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>)
        const override
    {
    }

private:
    [[nodiscard]] FE::GlobalIndex vertex(int i, int j, int k) const
    {
        return static_cast<FE::GlobalIndex>(
            (k * (ny_ + 1) + j) * (nx_ + 1) + i);
    }

    int nx_{0};
    int ny_{0};
    int nz_{0};
    FE::Real h_{1.0};
    std::vector<std::array<FE::Real, 3>> nodes_{};
};

class SimplexMeshAccess final : public FE::assembly::IMeshAccess {
public:
    struct BoundaryFace {
        FE::GlobalIndex parent_cell{FE::GlobalIndex{-1}};
        FE::LocalIndex local_face{FE::INVALID_LOCAL_INDEX};
        int marker{-1};
    };

    SimplexMeshAccess(
        int dimension,
        std::vector<std::array<FE::Real, 3>> nodes,
        std::vector<std::array<FE::GlobalIndex, 4>> cells,
        std::vector<BoundaryFace> boundary_faces = {})
        : dimension_(dimension),
          nodes_(std::move(nodes)),
          cells_(std::move(cells)),
          boundary_faces_(std::move(boundary_faces))
    {
        if (dimension_ != 2 && dimension_ != 3) {
            throw std::invalid_argument("simplex test mesh dimension must be two or three");
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
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(nodes_.size());
    }
    [[nodiscard]] int dimension() const override { return dimension_; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t numberingRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t coordinateConfigurationKey() const override
    {
        return 1u;
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex) const override { return true; }
    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override
    {
        return dimension_ == 2 ? FE::ElementType::Triangle3
                               : FE::ElementType::Tetra4;
    }

    void getCellNodes(FE::GlobalIndex cell,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& simplex = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(simplex.begin(),
                     simplex.begin() + static_cast<std::ptrdiff_t>(dimension_ + 1));
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
        std::vector<FE::GlobalIndex> nodes;
        getCellNodes(cell, nodes);
        coordinates.clear();
        for (const auto node : nodes) {
            coordinates.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face,
        FE::GlobalIndex cell) const override
    {
        const auto& boundary_face =
            boundary_faces_.at(static_cast<std::size_t>(face));
        return boundary_face.parent_cell == cell
            ? boundary_face.local_face
            : FE::INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex face) const override
    {
        return boundary_faces_.at(static_cast<std::size_t>(face)).marker;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override
    {
        return {0, 0};
    }
    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
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
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback)
        const override
    {
        for (std::size_t face = 0; face < boundary_faces_.size(); ++face) {
            const auto& boundary_face = boundary_faces_[face];
            if (marker < 0 || boundary_face.marker == marker) {
                callback(static_cast<FE::GlobalIndex>(face),
                         boundary_face.parent_cell);
            }
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>)
        const override
    {
    }

    void transformCoordinates(
        const std::function<std::array<FE::Real, 3>(
            const std::array<FE::Real, 3>&)>& transform)
    {
        for (auto& node : nodes_) {
            node = transform(node);
        }
    }

private:
    int dimension_{0};
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::vector<std::array<FE::GlobalIndex, 4>> cells_{};
    std::vector<BoundaryFace> boundary_faces_{};
};

SimplexMeshAccess makeStructuredTriangleMesh(int subdivisions,
                                             FE::Real minimum,
                                             FE::Real maximum)
{
    const FE::Real h =
        (maximum - minimum) / static_cast<FE::Real>(subdivisions);
    std::vector<std::array<FE::Real, 3>> nodes;
    nodes.reserve(static_cast<std::size_t>((subdivisions + 1) *
                                           (subdivisions + 1)));
    for (int j = 0; j <= subdivisions; ++j) {
        for (int i = 0; i <= subdivisions; ++i) {
            nodes.push_back({{
                minimum + static_cast<FE::Real>(i) * h,
                minimum + static_cast<FE::Real>(j) * h,
                FE::Real{0.0}}});
        }
    }
    const auto vertex = [subdivisions](int i, int j) {
        return static_cast<FE::GlobalIndex>(j * (subdivisions + 1) + i);
    };
    std::vector<std::array<FE::GlobalIndex, 4>> cells;
    std::vector<SimplexMeshAccess::BoundaryFace> boundary_faces;
    cells.reserve(static_cast<std::size_t>(2 * subdivisions * subdivisions));
    boundary_faces.reserve(static_cast<std::size_t>(4 * subdivisions));
    for (int j = 0; j < subdivisions; ++j) {
        for (int i = 0; i < subdivisions; ++i) {
            const auto v0 = vertex(i, j);
            const auto v1 = vertex(i + 1, j);
            const auto v2 = vertex(i + 1, j + 1);
            const auto v3 = vertex(i, j + 1);
            const auto lower_cell =
                static_cast<FE::GlobalIndex>(cells.size());
            cells.push_back({{v0, v1, v2, FE::GlobalIndex{-1}}});
            const auto upper_cell =
                static_cast<FE::GlobalIndex>(cells.size());
            cells.push_back({{v0, v2, v3, FE::GlobalIndex{-1}}});
            if (j == 0) {
                boundary_faces.push_back(
                    {lower_cell, FE::LocalIndex{0}, 1});
            }
            if (i + 1 == subdivisions) {
                boundary_faces.push_back(
                    {lower_cell, FE::LocalIndex{1}, 2});
            }
            if (j + 1 == subdivisions) {
                boundary_faces.push_back(
                    {upper_cell, FE::LocalIndex{1}, 3});
            }
            if (i == 0) {
                boundary_faces.push_back(
                    {upper_cell, FE::LocalIndex{2}, 4});
            }
        }
    }
    return SimplexMeshAccess(2,
                             std::move(nodes),
                             std::move(cells),
                             std::move(boundary_faces));
}

SimplexMeshAccess makeStructuredTetrahedronMesh(int subdivisions,
                                                FE::Real minimum,
                                                FE::Real maximum)
{
    const FE::Real h =
        (maximum - minimum) / static_cast<FE::Real>(subdivisions);
    std::vector<std::array<FE::Real, 3>> nodes;
    const auto side = static_cast<std::size_t>(subdivisions + 1);
    nodes.reserve(side * side * side);
    for (int k = 0; k <= subdivisions; ++k) {
        for (int j = 0; j <= subdivisions; ++j) {
            for (int i = 0; i <= subdivisions; ++i) {
                nodes.push_back({{
                    minimum + static_cast<FE::Real>(i) * h,
                    minimum + static_cast<FE::Real>(j) * h,
                    minimum + static_cast<FE::Real>(k) * h}});
            }
        }
    }
    const auto vertex = [subdivisions](int i, int j, int k) {
        return static_cast<FE::GlobalIndex>(
            (k * (subdivisions + 1) + j) * (subdivisions + 1) + i);
    };
    constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra{{
        {{0u, 1u, 2u, 6u}},
        {{0u, 2u, 3u, 6u}},
        {{0u, 3u, 7u, 6u}},
        {{0u, 7u, 4u, 6u}},
        {{0u, 4u, 5u, 6u}},
        {{0u, 5u, 1u, 6u}},
    }};
    std::vector<std::array<FE::GlobalIndex, 4>> cells;
    cells.reserve(static_cast<std::size_t>(
        6 * subdivisions * subdivisions * subdivisions));
    for (int k = 0; k < subdivisions; ++k) {
        for (int j = 0; j < subdivisions; ++j) {
            for (int i = 0; i < subdivisions; ++i) {
                const std::array<FE::GlobalIndex, 8> box{{
                    vertex(i, j, k),
                    vertex(i + 1, j, k),
                    vertex(i + 1, j + 1, k),
                    vertex(i, j + 1, k),
                    vertex(i, j, k + 1),
                    vertex(i + 1, j, k + 1),
                    vertex(i + 1, j + 1, k + 1),
                    vertex(i, j + 1, k + 1),
                }};
                for (const auto& tetrahedron : tetrahedra) {
                    cells.push_back({{
                        box[tetrahedron[0]],
                        box[tetrahedron[1]],
                        box[tetrahedron[2]],
                        box[tetrahedron[3]],
                    }});
                }
            }
        }
    }
    return SimplexMeshAccess(3, std::move(nodes), std::move(cells));
}

struct KinematicCurvatureEvaluation {
    level_set::LevelSetCurvatureProjectionResult result{};
    std::vector<FE::Real> curvature{};
    FE::Real mean_absolute_error{0.0};
    FE::Real root_mean_square_error{0.0};
    FE::Real maximum_absolute_error{0.0};
    FE::Real mean_curvature{0.0};
    FE::Real mass_weighted_root_mean_square_error{0.0};
    std::size_t samples{0u};
};

KinematicCurvatureEvaluation evaluateKinematicCurvature(
    SimplexMeshAccess& mesh,
    FE::Real radius,
    const std::array<FE::Real, 3>& center,
    FE::Real signed_level_set_scale = FE::Real{1.0},
    FE::Real filter_coefficient = FE::Real{1.0})
{
    std::vector<FE::Real> phi(
        static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto point = mesh.getNodeCoordinates(vertex);
        FE::Real distance2{0.0};
        for (int component = 0; component < mesh.dimension(); ++component) {
            const FE::Real delta =
                point[static_cast<std::size_t>(component)] -
                center[static_cast<std::size_t>(component)];
            distance2 += delta * delta;
        }
        phi[static_cast<std::size_t>(vertex)] =
            signed_level_set_scale * (std::sqrt(distance2) - radius);
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    options.kinematic_area_gradient_filter_coefficient = filter_coefficient;
    KinematicCurvatureEvaluation evaluation;
    evaluation.result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, evaluation.curvature);

    const FE::Real orientation =
        signed_level_set_scale > FE::Real{0.0} ? FE::Real{1.0}
                                                : FE::Real{-1.0};
    const FE::Real exact =
        orientation * static_cast<FE::Real>(mesh.dimension() - 1) / radius;
    for (const auto value : evaluation.curvature) {
        if (value == FE::Real{0.0}) {
            continue;
        }
        const FE::Real error = std::abs(value - exact);
        evaluation.mean_absolute_error += error;
        evaluation.root_mean_square_error += error * error;
        evaluation.maximum_absolute_error =
            std::max(evaluation.maximum_absolute_error, error);
        evaluation.mean_curvature += value;
        ++evaluation.samples;
    }
    if (evaluation.samples > 0u) {
        const FE::Real denominator =
            static_cast<FE::Real>(evaluation.samples);
        evaluation.mean_absolute_error /= denominator;
        evaluation.root_mean_square_error =
            std::sqrt(evaluation.root_mean_square_error / denominator);
        evaluation.mean_curvature /= denominator;
    }
    const FE::Real weighted_mean_error =
        evaluation.result
            .kinematic_area_gradient_mass_weighted_mean_curvature -
        exact;
    evaluation.mass_weighted_root_mean_square_error = std::sqrt(
        evaluation.result
                .kinematic_area_gradient_mass_weighted_rms_deviation *
            evaluation.result
                .kinematic_area_gradient_mass_weighted_rms_deviation +
        weighted_mean_error * weighted_mean_error);
    return evaluation;
}

KinematicCurvatureEvaluation evaluateSessileKinematicCurvature(
    SimplexMeshAccess& mesh,
    FE::Real radius,
    FE::Real contact_angle,
    FE::Real wall_coordinate,
    bool include_young_wall,
    FE::Real signed_level_set_scale = FE::Real{1.0},
    FE::Real filter_coefficient = FE::Real{1.0})
{
    const std::array<FE::Real, 3> center{{
        FE::Real{0.0},
        wall_coordinate - radius * std::cos(contact_angle),
        FE::Real{0.0}}};
    std::vector<FE::Real> phi(
        static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto point = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            signed_level_set_scale *
            (std::hypot(point[0] - center[0], point[1] - center[1]) -
             radius);
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    options.kinematic_area_gradient_filter_coefficient = filter_coefficient;
    options.kinematic_area_gradient_negative_liquid_side =
        signed_level_set_scale > FE::Real{0.0};
    if (include_young_wall) {
        options.kinematic_area_gradient_young_walls.push_back(
            {1, contact_angle});
    }

    KinematicCurvatureEvaluation evaluation;
    evaluation.result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, evaluation.curvature);
    const FE::Real orientation =
        signed_level_set_scale > FE::Real{0.0} ? FE::Real{1.0}
                                                : FE::Real{-1.0};
    const FE::Real exact = orientation / radius;
    for (const auto value : evaluation.curvature) {
        if (value == FE::Real{0.0}) {
            continue;
        }
        const FE::Real error = std::abs(value - exact);
        evaluation.mean_absolute_error += error;
        evaluation.root_mean_square_error += error * error;
        evaluation.maximum_absolute_error =
            std::max(evaluation.maximum_absolute_error, error);
        evaluation.mean_curvature += value;
        ++evaluation.samples;
    }
    if (evaluation.samples > 0u) {
        const FE::Real denominator =
            static_cast<FE::Real>(evaluation.samples);
        evaluation.mean_absolute_error /= denominator;
        evaluation.root_mean_square_error =
            std::sqrt(evaluation.root_mean_square_error / denominator);
        evaluation.mean_curvature /= denominator;
    }
    const FE::Real weighted_mean_error =
        evaluation.result
            .kinematic_area_gradient_mass_weighted_mean_curvature -
        exact;
    evaluation.mass_weighted_root_mean_square_error = std::sqrt(
        evaluation.result
                .kinematic_area_gradient_mass_weighted_rms_deviation *
            evaluation.result
                .kinematic_area_gradient_mass_weighted_rms_deviation +
        weighted_mean_error * weighted_mean_error);
    return evaluation;
}

FE::Real directSessileDiscreteEnergy(
    const SimplexMeshAccess& mesh,
    const std::vector<FE::Real>& phi,
    FE::Real level_set_offset,
    FE::Real contact_angle,
    bool negative_liquid_side)
{
    FE::Real interface_measure{0.0};
    mesh.forEachCell([&](FE::GlobalIndex cell) {
        std::vector<FE::GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        if (nodes.size() != 3u) {
            throw std::runtime_error(
                "sessile discrete-energy check requires triangles");
        }
        std::array<FE::Real, 3> values{};
        for (std::size_t corner = 0; corner < nodes.size(); ++corner) {
            values[corner] =
                phi[static_cast<std::size_t>(nodes[corner])] +
                level_set_offset;
            if (values[corner] == FE::Real{0.0}) {
                throw std::runtime_error(
                    "sessile discrete-energy check encountered an isovalue vertex");
            }
        }
        std::array<std::array<FE::Real, 3>, 2> intersections{};
        std::size_t intersection_count{0u};
        constexpr std::array<std::array<std::size_t, 2>, 3> edges{{
            {{0u, 1u}}, {{1u, 2u}}, {{2u, 0u}}}};
        for (const auto& edge : edges) {
            const auto a = edge[0];
            const auto b = edge[1];
            if ((values[a] < FE::Real{0.0}) ==
                (values[b] < FE::Real{0.0})) {
                continue;
            }
            if (intersection_count >= intersections.size()) {
                throw std::runtime_error(
                    "sessile discrete-energy check found invalid cut topology");
            }
            const FE::Real fraction = values[a] / (values[a] - values[b]);
            const auto x_a = mesh.getNodeCoordinates(nodes[a]);
            const auto x_b = mesh.getNodeCoordinates(nodes[b]);
            for (std::size_t component = 0; component < 3u; ++component) {
                intersections[intersection_count][component] =
                    x_a[component] +
                    fraction * (x_b[component] - x_a[component]);
            }
            ++intersection_count;
        }
        if (intersection_count == 2u) {
            interface_measure += std::hypot(
                intersections[1][0] - intersections[0][0],
                intersections[1][1] - intersections[0][1]);
        } else if (intersection_count != 0u) {
            throw std::runtime_error(
                "sessile discrete-energy check found an incomplete cut");
        }
    });

    FE::Real wetted_measure{0.0};
    mesh.forEachBoundaryFace(
        1,
        [&](FE::GlobalIndex face, FE::GlobalIndex cell) {
            std::vector<FE::GlobalIndex> nodes;
            mesh.getCellNodes(cell, nodes);
            constexpr std::array<std::array<std::size_t, 2>, 3> faces{{
                {{0u, 1u}}, {{1u, 2u}}, {{2u, 0u}}}};
            const auto local_face =
                static_cast<std::size_t>(mesh.getLocalFaceIndex(face, cell));
            if (local_face >= faces.size()) {
                throw std::runtime_error(
                    "sessile discrete-energy check found an invalid boundary face");
            }
            const auto node0 = nodes[faces[local_face][0]];
            const auto node1 = nodes[faces[local_face][1]];
            const FE::Real value0 =
                phi[static_cast<std::size_t>(node0)] + level_set_offset;
            const FE::Real value1 =
                phi[static_cast<std::size_t>(node1)] + level_set_offset;
            const bool inside0 = negative_liquid_side
                ? value0 < FE::Real{0.0}
                : value0 > FE::Real{0.0};
            const bool inside1 = negative_liquid_side
                ? value1 < FE::Real{0.0}
                : value1 > FE::Real{0.0};
            const auto x0 = mesh.getNodeCoordinates(node0);
            const auto x1 = mesh.getNodeCoordinates(node1);
            const FE::Real length =
                std::hypot(x1[0] - x0[0], x1[1] - x0[1]);
            if (inside0 == inside1) {
                wetted_measure += inside0 ? length : FE::Real{0.0};
                return;
            }
            const FE::Real fraction = value0 / (value0 - value1);
            wetted_measure +=
                length * (inside0 ? fraction : FE::Real{1.0} - fraction);
        });
    return interface_measure - std::cos(contact_angle) * wetted_measure;
}

FE::Real curvatureGraphTotalVariation(
    const FE::assembly::IMeshAccess& mesh,
    const std::vector<FE::Real>& values)
{
    std::vector<std::pair<FE::GlobalIndex, FE::GlobalIndex>> edges;
    std::vector<FE::GlobalIndex> nodes;
    mesh.forEachCell([&](FE::GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (std::size_t j = i + 1u; j < nodes.size(); ++j) {
                auto a = nodes[i];
                auto b = nodes[j];
                if (a == b || a < 0 || b < 0) {
                    continue;
                }
                if (b < a) {
                    std::swap(a, b);
                }
                edges.push_back({a, b});
            }
        }
    });
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    FE::Real total = FE::Real{0.0};
    for (const auto& [a, b] : edges) {
        total += std::abs(values[static_cast<std::size_t>(a)] -
                          values[static_cast<std::size_t>(b)]);
    }
    return total;
}

FE::Real maxAbsDifference(const std::vector<FE::Real>& lhs,
                          const std::vector<FE::Real>& rhs)
{
    FE::Real value = FE::Real{0.0};
    const auto n = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < n; ++i) {
        value = std::max(value, std::abs(lhs[i] - rhs[i]));
    }
    return value;
}

FE::Real circleCurvatureMeanError(int nx, FE::Real h, FE::Real radius)
{
    StructuredQuadMeshAccess mesh(nx, nx, h);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);
    if (!result.success) {
        throw std::runtime_error(result.diagnostic);
    }

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        if (r < radius - FE::Real{0.05} || r > radius + FE::Real{0.05}) {
            continue;
        }
        error_sum += std::abs(
            curvature[static_cast<std::size_t>(v)] - FE::Real{1.0} / r);
        ++samples;
    }
    if (samples == 0u) {
        throw std::runtime_error("no circle curvature samples in narrow band");
    }
    return error_sum / static_cast<FE::Real>(samples);
}

struct CurvatureStudyResult {
    FE::Real mean_error{0.0};
    FE::Real max_error{0.0};
    std::size_t error_samples{0u};
    std::size_t supplemental_samples{0u};
    std::size_t generated_geometry_samples{0u};
    std::size_t generated_patch_fitted_vertices{0u};
    std::size_t generated_patch_expanded_vertices{0u};
    std::size_t narrow_band_vertices{0u};
    std::size_t fallback_vertices{0u};
    std::size_t zero_fallback_vertices{0u};
    std::size_t insufficient_stencil_vertices{0u};
    std::size_t singular_stencil_vertices{0u};
    FE::Real max_fit_residual{0.0};
    FE::Real max_interpolation_defect{0.0};
};

std::vector<level_set::LevelSetCurvatureProjectionSample>
makeGeneratedCircleInterfaceSamples(const FE::assembly::IMeshAccess& mesh,
                                    FE::Real radius,
                                    const std::array<FE::Real, 3>& center)
{
    std::vector<level_set::LevelSetCurvatureProjectionSample> samples;
    std::vector<FE::GlobalIndex> nodes;
    constexpr std::array<std::array<std::size_t, 2>, 4> edges{{
        {{0u, 1u}}, {{1u, 2u}}, {{2u, 3u}}, {{3u, 0u}},
    }};
    constexpr std::array<FE::Real, 2> segment_coordinates{{
        FE::Real{0.21132486540518713},
        FE::Real{0.78867513459481287},
    }};
    mesh.forEachCell([&](FE::GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        if (nodes.size() != 4u) {
            throw std::runtime_error(
                "generated circle interface expects quadrilateral cells");
        }
        std::vector<std::array<FE::Real, 3>> intersections;
        for (const auto& edge : edges) {
            const auto x0 = mesh.getNodeCoordinates(nodes[edge[0]]);
            const auto x1 = mesh.getNodeCoordinates(nodes[edge[1]]);
            const FE::Real phi0 =
                std::hypot(x0[0] - center[0], x0[1] - center[1]) -
                radius;
            const FE::Real phi1 =
                std::hypot(x1[0] - center[0], x1[1] - center[1]) -
                radius;
            if (phi0 * phi1 >= FE::Real{0.0}) {
                continue;
            }
            const FE::Real t = phi0 / (phi0 - phi1);
            intersections.push_back({{
                x0[0] + t * (x1[0] - x0[0]),
                x0[1] + t * (x1[1] - x0[1]),
                FE::Real{0.0},
            }});
        }
        if (intersections.empty()) {
            return;
        }
        if (intersections.size() != 2u) {
            throw std::runtime_error(
                "generated circle interface produced an ambiguous cut cell");
        }
        for (const auto t : segment_coordinates) {
            std::array<FE::Real, 3> point{};
            for (std::size_t d = 0; d < point.size(); ++d) {
                point[d] = (FE::Real{1.0} - t) * intersections[0][d] +
                           t * intersections[1][d];
            }
            samples.push_back(
                level_set::LevelSetCurvatureProjectionSample{
                    .parent_cell = static_cast<FE::MeshIndex>(cell),
                    .coordinate = point,
                    .value = FE::Real{0.0},
                    .generated_interface_geometry = true,
                });
        }
    });
    return samples;
}

CurvatureStudyResult generatedCirclePatchCurvatureError(
    int subdivisions,
    std::array<FE::Real, 3> center = {},
    FE::Real phi_scale = FE::Real{1.0},
    FE::Real rigid_rotation = FE::Real{0.0},
    std::array<FE::Real, 3> rigid_translation = {})
{
    constexpr FE::Real extent = FE::Real{0.8};
    constexpr FE::Real radius = FE::Real{0.24};
    const FE::Real h = extent / static_cast<FE::Real>(subdivisions);
    StructuredQuadMeshAccess mesh(subdivisions, subdivisions, h);
    const FE::Real cosine = std::cos(rigid_rotation);
    const FE::Real sine = std::sin(rigid_rotation);
    center = {{
        cosine * center[0] - sine * center[1] + rigid_translation[0],
        sine * center[0] + cosine * center[1] + rigid_translation[1],
        rigid_translation[2],
    }};
    mesh.rotate(rigid_rotation);
    mesh.translate(rigid_translation[0], rigid_translation[1]);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()));
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto x = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            phi_scale *
            (std::hypot(x[0] - center[0], x[1] - center[1]) - radius);
    }
    const auto samples =
        makeGeneratedCircleInterfaceSamples(mesh, radius, center);

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch;
    options.max_neighbor_rings = 2;
    options.narrow_band_width =
        std::abs(phi_scale) * FE::Real{1.5} * h;
    std::vector<FE::Real> curvature;
    const auto projection = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);
    if (!projection.success) {
        throw std::runtime_error(projection.diagnostic);
    }

    CurvatureStudyResult study;
    study.supplemental_samples = projection.supplemental_samples;
    study.generated_geometry_samples =
        projection.generated_interface_geometry_samples;
    study.generated_patch_fitted_vertices =
        projection.generated_interface_patch_fitted_vertices;
    study.generated_patch_expanded_vertices =
        projection.generated_interface_patch_expanded_vertices;
    study.max_fit_residual = projection.max_normalized_fit_residual;
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        if (std::abs(phi[index]) > std::abs(phi_scale) * h) {
            continue;
        }
        const FE::Real expected_curvature =
            std::copysign(FE::Real{1.0} / radius, phi_scale);
        const FE::Real error =
            std::abs(curvature[index] - expected_curvature);
        study.mean_error += error;
        study.max_error = std::max(study.max_error, error);
        ++study.error_samples;
    }
    if (study.error_samples == 0u) {
        throw std::runtime_error(
            "generated circle patch study has no interface-band vertices");
    }
    study.mean_error /= static_cast<FE::Real>(study.error_samples);
    return study;
}

std::vector<level_set::LevelSetCurvatureProjectionSample>
makeSphereInterfacePointCloudSamples(int subdivisions,
                                     FE::Real h,
                                     FE::Real radius)
{
    // Exact points isolate the three-dimensional graph-fit formula from the
    // separate approximation error of a particular generated-surface backend.
    constexpr FE::Real pi =
        FE::Real{3.141592653589793238462643383279502884};
    const std::size_t sample_count =
        static_cast<std::size_t>(24 * subdivisions * subdivisions);
    const FE::Real minimum_coordinate =
        -FE::Real{0.5} * static_cast<FE::Real>(subdivisions) * h;
    const FE::Real golden_angle =
        pi * (FE::Real{3.0} - std::sqrt(FE::Real{5.0}));
    std::vector<level_set::LevelSetCurvatureProjectionSample> samples;
    samples.reserve(sample_count);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const FE::Real z =
            FE::Real{1.0} -
            FE::Real{2.0} *
                (static_cast<FE::Real>(sample) + FE::Real{0.5}) /
                static_cast<FE::Real>(sample_count);
        const FE::Real radial =
            std::sqrt(std::max(FE::Real{0.0}, FE::Real{1.0} - z * z));
        const FE::Real theta =
            golden_angle * static_cast<FE::Real>(sample);
        const std::array<FE::Real, 3> point{{
            radius * radial * std::cos(theta),
            radius * radial * std::sin(theta),
            radius * z,
        }};
        std::array<int, 3> cell_coordinate{};
        for (std::size_t d = 0; d < cell_coordinate.size(); ++d) {
            cell_coordinate[d] = std::clamp(
                static_cast<int>(
                    std::floor((point[d] - minimum_coordinate) / h)),
                0,
                subdivisions - 1);
        }
        const FE::MeshIndex parent_cell = static_cast<FE::MeshIndex>(
            (cell_coordinate[2] * subdivisions + cell_coordinate[1]) *
                subdivisions +
            cell_coordinate[0]);
        samples.push_back(level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = parent_cell,
            .coordinate = point,
            .value = FE::Real{0.0},
            .generated_interface_geometry = true,
        });
    }
    return samples;
}

std::vector<level_set::LevelSetCurvatureProjectionSample>
makeGeneratedSphereFacetSamples(const FE::assembly::IMeshAccess& mesh,
                                FE::Real radius)
{
    // Match the LinearCorner SayeHyperrectangle box decomposition and its
    // order-one polygon-centroid interface rule.
    constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra{{
        {{0u, 1u, 2u, 6u}},
        {{0u, 2u, 3u, 6u}},
        {{0u, 3u, 7u, 6u}},
        {{0u, 7u, 4u, 6u}},
        {{0u, 4u, 5u, 6u}},
        {{0u, 5u, 1u, 6u}},
    }};
    constexpr std::array<std::array<std::size_t, 2>, 6> tetrahedron_edges{{
        {{0u, 1u}}, {{0u, 2u}}, {{0u, 3u}},
        {{1u, 2u}}, {{1u, 3u}}, {{2u, 3u}},
    }};
    constexpr FE::Real duplicate_tolerance2 = FE::Real{1.0e-28};

    const auto subtract = [](const auto& a, const auto& b) {
        return std::array<FE::Real, 3>{{
            a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
    };
    const auto dot = [](const auto& a, const auto& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    const auto cross = [](const auto& a, const auto& b) {
        return std::array<FE::Real, 3>{{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]}};
    };
    const auto norm = [&](const auto& value) {
        return std::sqrt(dot(value, value));
    };

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples;
    std::vector<FE::GlobalIndex> nodes;
    mesh.forEachCell([&](FE::GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        if (nodes.size() != 8u) {
            throw std::runtime_error(
                "generated sphere facets expect hexahedral cells");
        }
        std::array<std::array<FE::Real, 3>, 8> coordinates{};
        std::array<FE::Real, 8> values{};
        for (std::size_t node = 0u; node < nodes.size(); ++node) {
            coordinates[node] = mesh.getNodeCoordinates(nodes[node]);
            values[node] =
                norm(coordinates[node]) - radius;
        }

        for (const auto& tetrahedron : tetrahedra) {
            std::vector<std::array<FE::Real, 3>> intersections;
            intersections.reserve(4u);
            const auto append_intersection = [&](const auto& point) {
                const bool duplicate = std::any_of(
                    intersections.begin(), intersections.end(),
                    [&](const auto& existing) {
                        const auto delta = subtract(existing, point);
                        return dot(delta, delta) <= duplicate_tolerance2;
                    });
                if (!duplicate) {
                    intersections.push_back(point);
                }
            };
            for (const auto& edge : tetrahedron_edges) {
                const auto local_a = tetrahedron[edge[0]];
                const auto local_b = tetrahedron[edge[1]];
                const FE::Real value_a = values[local_a];
                const FE::Real value_b = values[local_b];
                if ((value_a < FE::Real{0.0}) ==
                    (value_b < FE::Real{0.0})) {
                    continue;
                }
                const FE::Real fraction = value_a / (value_a - value_b);
                std::array<FE::Real, 3> point{};
                for (std::size_t component = 0u;
                     component < point.size();
                     ++component) {
                    point[component] =
                        coordinates[local_a][component] +
                        fraction *
                            (coordinates[local_b][component] -
                             coordinates[local_a][component]);
                }
                append_intersection(point);
            }
            if (intersections.size() < 3u) {
                continue;
            }
            if (intersections.size() > 4u) {
                throw std::runtime_error(
                    "generated sphere facet has an invalid intersection polygon");
            }

            std::array<FE::Real, 3> centroid{};
            for (const auto& point : intersections) {
                for (std::size_t component = 0u;
                     component < centroid.size();
                     ++component) {
                    centroid[component] += point[component];
                }
            }
            for (auto& component : centroid) {
                component /= static_cast<FE::Real>(intersections.size());
            }
            auto normal = cross(subtract(intersections[1], intersections[0]),
                                subtract(intersections[2], intersections[0]));
            const FE::Real normal_norm = norm(normal);
            if (!(normal_norm > FE::Real{0.0})) {
                throw std::runtime_error(
                    "generated sphere facet is degenerate");
            }
            for (auto& component : normal) {
                component /= normal_norm;
            }
            const std::array<FE::Real, 3> axis =
                std::abs(normal[0]) <= std::abs(normal[1]) &&
                        std::abs(normal[0]) <= std::abs(normal[2])
                    ? std::array<FE::Real, 3>{{1.0, 0.0, 0.0}}
                    : (std::abs(normal[1]) <= std::abs(normal[2])
                           ? std::array<FE::Real, 3>{{0.0, 1.0, 0.0}}
                           : std::array<FE::Real, 3>{{0.0, 0.0, 1.0}});
            auto tangent0 = cross(axis, normal);
            const FE::Real tangent0_norm = norm(tangent0);
            for (auto& component : tangent0) {
                component /= tangent0_norm;
            }
            const auto tangent1 = cross(normal, tangent0);
            std::sort(
                intersections.begin(), intersections.end(),
                [&](const auto& lhs, const auto& rhs) {
                    const auto lhs_delta = subtract(lhs, centroid);
                    const auto rhs_delta = subtract(rhs, centroid);
                    const FE::Real lhs_angle =
                        std::atan2(dot(lhs_delta, tangent1),
                                   dot(lhs_delta, tangent0));
                    const FE::Real rhs_angle =
                        std::atan2(dot(rhs_delta, tangent1),
                                   dot(rhs_delta, tangent0));
                    return lhs_angle < rhs_angle;
                });

            std::array<FE::Real, 3> polygon_centroid{};
            FE::Real polygon_area = FE::Real{0.0};
            for (std::size_t triangle = 1u;
                 triangle + 1u < intersections.size();
                 ++triangle) {
                const auto triangle_cross = cross(
                    subtract(intersections[triangle], intersections[0]),
                    subtract(intersections[triangle + 1u], intersections[0]));
                const FE::Real area = FE::Real{0.5} * norm(triangle_cross);
                std::array<FE::Real, 3> triangle_centroid{};
                for (std::size_t component = 0u;
                     component < triangle_centroid.size();
                     ++component) {
                    triangle_centroid[component] =
                        (intersections[0][component] +
                         intersections[triangle][component] +
                         intersections[triangle + 1u][component]) /
                        FE::Real{3.0};
                    polygon_centroid[component] +=
                        area * triangle_centroid[component];
                }
                polygon_area += area;
            }
            if (!(polygon_area > FE::Real{0.0})) {
                throw std::runtime_error(
                    "generated sphere facet has zero polygon area");
            }
            for (auto& component : polygon_centroid) {
                component /= polygon_area;
            }
            samples.push_back(
                level_set::LevelSetCurvatureProjectionSample{
                    .parent_cell = static_cast<FE::MeshIndex>(cell),
                    .coordinate = polygon_centroid,
                    .value = FE::Real{0.0},
                    .generated_interface_geometry = true,
                });
        }
    });
    return samples;
}

CurvatureStudyResult generatedSphereFacetPatchCurvatureError(int subdivisions)
{
    constexpr FE::Real extent = FE::Real{0.8};
    constexpr FE::Real radius = FE::Real{0.24};
    const FE::Real h = extent / static_cast<FE::Real>(subdivisions);
    StructuredHexMeshAccess mesh(
        subdivisions, subdivisions, subdivisions, h);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()));
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto x = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) -
            radius;
    }
    const auto samples = makeGeneratedSphereFacetSamples(mesh, radius);

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = FE::Real{1.5} * h;
    std::vector<FE::Real> curvature;
    const auto projection = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);
    if (!projection.success) {
        throw std::runtime_error(projection.diagnostic);
    }

    CurvatureStudyResult study;
    study.supplemental_samples = projection.supplemental_samples;
    study.generated_geometry_samples =
        projection.generated_interface_geometry_samples;
    study.generated_patch_fitted_vertices =
        projection.generated_interface_patch_fitted_vertices;
    study.generated_patch_expanded_vertices =
        projection.generated_interface_patch_expanded_vertices;
    study.narrow_band_vertices = projection.narrow_band_vertices;
    study.fallback_vertices = projection.fallback_vertices;
    study.zero_fallback_vertices = projection.zero_fallback_vertices;
    study.insufficient_stencil_vertices =
        projection.insufficient_stencil_vertices;
    study.singular_stencil_vertices = projection.singular_stencil_vertices;
    study.max_fit_residual = projection.max_normalized_fit_residual;
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        if (std::abs(phi[index]) > h) {
            continue;
        }
        const FE::Real error =
            std::abs(curvature[index] - FE::Real{2.0} / radius);
        study.mean_error += error;
        study.max_error = std::max(study.max_error, error);
        ++study.error_samples;
    }
    if (study.error_samples == 0u) {
        throw std::runtime_error(
            "generated sphere facet study has no interface-band vertices");
    }
    study.mean_error /= static_cast<FE::Real>(study.error_samples);
    return study;
}

CurvatureStudyResult spherePointCloudPatchCurvatureError(int subdivisions)
{
    constexpr FE::Real extent = FE::Real{0.8};
    constexpr FE::Real radius = FE::Real{0.24};
    const FE::Real h = extent / static_cast<FE::Real>(subdivisions);
    StructuredHexMeshAccess mesh(
        subdivisions, subdivisions, subdivisions, h);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()));
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto x = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) -
            radius;
    }
    const auto samples =
        makeSphereInterfacePointCloudSamples(subdivisions, h, radius);

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = FE::Real{1.5} * h;
    std::vector<FE::Real> curvature;
    const auto projection = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);
    if (!projection.success) {
        throw std::runtime_error(projection.diagnostic);
    }

    CurvatureStudyResult study;
    study.supplemental_samples = projection.supplemental_samples;
    study.generated_geometry_samples =
        projection.generated_interface_geometry_samples;
    study.generated_patch_fitted_vertices =
        projection.generated_interface_patch_fitted_vertices;
    study.generated_patch_expanded_vertices =
        projection.generated_interface_patch_expanded_vertices;
    study.max_fit_residual = projection.max_normalized_fit_residual;
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        if (std::abs(phi[index]) > h) {
            continue;
        }
        const FE::Real error =
            std::abs(curvature[index] - FE::Real{2.0} / radius);
        study.mean_error += error;
        study.max_error = std::max(study.max_error, error);
        ++study.error_samples;
    }
    if (study.error_samples == 0u) {
        throw std::runtime_error(
            "sphere point-cloud patch study has no interface-band vertices");
    }
    study.mean_error /= static_cast<FE::Real>(study.error_samples);
    return study;
}

template <typename ValueFunction>
std::vector<level_set::LevelSetCurvatureProjectionSample>
makeQ3InteriorSamples(const FE::assembly::IMeshAccess& mesh,
                      const ValueFunction& value_function,
                      FE::Real* max_interpolation_defect = nullptr)
{
    const auto element_type =
        mesh.dimension() == 3 ? FE::ElementType::Hex8
                              : FE::ElementType::Quad4;
    FE::spaces::H1Space q3_space(element_type, /*order=*/3);
    const auto q3_nodes =
        FE::basis::ReferenceNodeLayout::get_lagrange_node_coords(
            element_type, /*order=*/3);
    const FE::Real gauss_coordinate =
        FE::Real{1.0} / std::sqrt(FE::Real{3.0});
    const std::array<FE::Real, 2> coordinates{
        -gauss_coordinate,
        gauss_coordinate,
    };
    if (max_interpolation_defect != nullptr) {
        *max_interpolation_defect = FE::Real{0.0};
    }
    std::vector<level_set::LevelSetCurvatureProjectionSample> samples;
    const auto samples_per_cell = mesh.dimension() == 3 ? 8u : 4u;
    samples.reserve(static_cast<std::size_t>(mesh.numCells()) *
                    samples_per_cell);

    std::vector<std::array<FE::Real, 3>> cell_coordinates;
    mesh.forEachCell([&](FE::GlobalIndex cell) {
        mesh.getCellCoordinates(cell, cell_coordinates);
        if (mesh.dimension() == 2 && cell_coordinates.size() != 4u) {
            throw std::runtime_error("Q3 curvature study expects Quad4 cells");
        }
        if (mesh.dimension() == 3 && cell_coordinates.size() != 8u) {
            throw std::runtime_error("Q3 curvature study expects Hex8 cells");
        }

        const auto map_to_physical =
            [&](const auto& xi) -> std::array<FE::Real, 3> {
                const FE::Real u = FE::Real{0.5} * (xi[0] + FE::Real{1.0});
                const FE::Real v = FE::Real{0.5} * (xi[1] + FE::Real{1.0});
                const FE::Real w =
                    FE::Real{0.5} * (xi[2] + FE::Real{1.0});
                std::array<FE::Real, 3> x{{0.0, 0.0, 0.0}};
                if (mesh.dimension() == 2) {
                    const std::array<FE::Real, 4> weights{
                        (FE::Real{1.0} - u) * (FE::Real{1.0} - v),
                        u * (FE::Real{1.0} - v),
                        u * v,
                        (FE::Real{1.0} - u) * v,
                    };
                    for (std::size_t node = 0; node < weights.size(); ++node) {
                        for (int component = 0; component < 3; ++component) {
                            x[static_cast<std::size_t>(component)] +=
                                weights[node] *
                                cell_coordinates[node]
                                                [static_cast<std::size_t>(
                                                    component)];
                        }
                    }
                } else {
                    const std::array<FE::Real, 8> weights{
                        (FE::Real{1.0} - u) * (FE::Real{1.0} - v) *
                            (FE::Real{1.0} - w),
                        u * (FE::Real{1.0} - v) * (FE::Real{1.0} - w),
                        u * v * (FE::Real{1.0} - w),
                        (FE::Real{1.0} - u) * v * (FE::Real{1.0} - w),
                        (FE::Real{1.0} - u) * (FE::Real{1.0} - v) * w,
                        u * (FE::Real{1.0} - v) * w,
                        u * v * w,
                        (FE::Real{1.0} - u) * v * w,
                    };
                    for (std::size_t node = 0; node < weights.size(); ++node) {
                        for (int component = 0; component < 3; ++component) {
                            x[static_cast<std::size_t>(component)] +=
                                weights[node] *
                                cell_coordinates[node]
                                                [static_cast<std::size_t>(
                                                    component)];
                        }
                    }
                }
                return x;
            };

        std::vector<FE::Real> coefficients;
        coefficients.reserve(q3_nodes.size());
        for (const auto& q3_node : q3_nodes) {
            coefficients.push_back(value_function(map_to_physical(q3_node)));
        }

        for (const auto xi0 : coordinates) {
            for (const auto xi1 : coordinates) {
                const auto append_sample = [&](FE::Real xi2) {
                    FE::spaces::FunctionSpace::Value xi{};
                    xi[0] = xi0;
                    xi[1] = xi1;
                    xi[2] = xi2;
                    const auto x = map_to_physical(xi);
                    const auto interpolated_value =
                        q3_space.evaluate_scalar(xi, coefficients);
                    if (max_interpolation_defect != nullptr) {
                        *max_interpolation_defect = std::max(
                            *max_interpolation_defect,
                            std::abs(interpolated_value - value_function(x)));
                    }
                    samples.push_back(
                        level_set::LevelSetCurvatureProjectionSample{
                            .parent_cell = static_cast<FE::MeshIndex>(cell),
                            .coordinate = x,
                            .value = interpolated_value,
                        });
                };

                if (mesh.dimension() == 2) {
                    append_sample(FE::Real{0.0});
                } else {
                    for (const auto w : coordinates) {
                        append_sample(w);
                    }
                }
            }
        }
    });
    return samples;
}

CurvatureStudyResult staticDropQ3CurvatureError(int subdivisions)
{
    constexpr FE::Real extent = 0.8;
    constexpr FE::Real radius = 0.24;
    const FE::Real h = extent / static_cast<FE::Real>(subdivisions);
    StructuredQuadMeshAccess mesh(subdivisions, subdivisions, h);
    const auto signed_distance = [=](const std::array<FE::Real, 3>& x) {
        return std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()));
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        phi[static_cast<std::size_t>(vertex)] =
            signed_distance(mesh.getNodeCoordinates(vertex));
    }
    FE::Real max_interpolation_defect = 0.0;
    const auto samples = makeQ3InteriorSamples(
        mesh, signed_distance, &max_interpolation_defect);

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = FE::Real{1.5} * h;
    std::vector<FE::Real> curvature;
    const auto projection = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);
    if (!projection.success) {
        throw std::runtime_error(projection.diagnostic);
    }

    CurvatureStudyResult study;
    study.supplemental_samples = projection.supplemental_samples;
    study.max_fit_residual = projection.max_normalized_fit_residual;
    study.max_interpolation_defect = max_interpolation_defect;
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        if (std::abs(phi[index]) > h) {
            continue;
        }
        const auto x = mesh.getNodeCoordinates(vertex);
        const FE::Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        const FE::Real error = std::abs(curvature[index] - FE::Real{1.0} / r);
        study.mean_error += error;
        study.max_error = std::max(study.max_error, error);
        ++study.error_samples;
    }
    if (study.error_samples == 0u) {
        throw std::runtime_error("static-drop curvature study has no samples");
    }
    study.mean_error /= static_cast<FE::Real>(study.error_samples);
    return study;
}

CurvatureStudyResult sphereQ3CurvatureError(int subdivisions)
{
    constexpr FE::Real extent = 0.8;
    constexpr FE::Real radius = 0.24;
    const FE::Real h = extent / static_cast<FE::Real>(subdivisions);
    StructuredHexMeshAccess mesh(
        subdivisions, subdivisions, subdivisions, h);
    const auto signed_distance = [=](const std::array<FE::Real, 3>& x) {
        return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) -
               radius;
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()));
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        phi[static_cast<std::size_t>(vertex)] =
            signed_distance(mesh.getNodeCoordinates(vertex));
    }
    FE::Real max_interpolation_defect = 0.0;
    const auto samples = makeQ3InteriorSamples(
        mesh, signed_distance, &max_interpolation_defect);

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = FE::Real{1.5} * h;
    std::vector<FE::Real> curvature;
    const auto projection = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);
    if (!projection.success) {
        throw std::runtime_error(projection.diagnostic);
    }

    CurvatureStudyResult study;
    study.supplemental_samples = projection.supplemental_samples;
    study.max_fit_residual = projection.max_normalized_fit_residual;
    study.max_interpolation_defect = max_interpolation_defect;
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        if (std::abs(phi[index]) > h) {
            continue;
        }
        const auto x = mesh.getNodeCoordinates(vertex);
        const FE::Real r =
            std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        const FE::Real error = std::abs(curvature[index] - FE::Real{2.0} / r);
        study.mean_error += error;
        study.max_error = std::max(study.max_error, error);
        ++study.error_samples;
    }
    if (study.error_samples == 0u) {
        throw std::runtime_error("sphere curvature study has no samples");
    }
    study.mean_error /= static_cast<FE::Real>(study.error_samples);
    return study;
}

CurvatureStudyResult capillaryWaveQ3CurvatureError(int subdivisions)
{
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    constexpr FE::Real amplitude = 0.01;
    constexpr FE::Real wave_number = FE::Real{2.0} * pi;
    const int vertical_subdivisions = subdivisions / 2;
    const FE::Real h = FE::Real{1.0} / static_cast<FE::Real>(subdivisions);
    StructuredQuadMeshAccess mesh(
        subdivisions, vertical_subdivisions, h, h);
    const auto level_set_value = [](const std::array<FE::Real, 3>& x) {
        return x[1] - amplitude * std::cos(wave_number * x[0]);
    };
    const auto exact_curvature = [](FE::Real x) {
        const FE::Real slope =
            -amplitude * wave_number * std::sin(wave_number * x);
        const FE::Real denominator =
            std::pow(FE::Real{1.0} + slope * slope, FE::Real{1.5});
        return amplitude * wave_number * wave_number *
               std::cos(wave_number * x) / denominator;
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()));
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        phi[static_cast<std::size_t>(vertex)] =
            level_set_value(mesh.getNodeCoordinates(vertex));
    }
    FE::Real max_interpolation_defect = 0.0;
    const auto samples = makeQ3InteriorSamples(
        mesh, level_set_value, &max_interpolation_defect);

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = FE::Real{1.5} * h;
    std::vector<FE::Real> curvature;
    const auto projection = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);
    if (!projection.success) {
        throw std::runtime_error(projection.diagnostic);
    }

    CurvatureStudyResult study;
    study.supplemental_samples = projection.supplemental_samples;
    study.max_fit_residual = projection.max_normalized_fit_residual;
    study.max_interpolation_defect = max_interpolation_defect;
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        const auto x = mesh.getNodeCoordinates(vertex);
        if (std::abs(phi[index]) > h ||
            std::abs(x[0]) > FE::Real{0.5} - FE::Real{2.0} * h) {
            continue;
        }
        const FE::Real error =
            std::abs(curvature[index] - exact_curvature(x[0]));
        study.mean_error += error;
        study.max_error = std::max(study.max_error, error);
        ++study.error_samples;
    }
    if (study.error_samples == 0u) {
        throw std::runtime_error("capillary-wave curvature study has no samples");
    }
    study.mean_error /= static_cast<FE::Real>(study.error_samples);
    return study;
}

FE::Real observedOrder(FE::Real coarse_error, FE::Real fine_error)
{
    return std::log(coarse_error / fine_error) / std::log(FE::Real{2.0});
}

FE::Real recoveredQuadraticCurvatureAtOrigin(FE::Real hx, FE::Real hy)
{
    StructuredQuadMeshAccess mesh(/*nx=*/6, /*ny=*/6, hx, hy);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{0.0});
    FE::GlobalIndex origin = -1;
    FE::Real origin_radius2 = std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto x = mesh.getNodeCoordinates(vertex);
        // At the origin, grad(phi)=(1,0) and Hessian(phi)_yy=2, hence
        // div(grad(phi)/|grad(phi)|)=2 exactly.
        phi[static_cast<std::size_t>(vertex)] =
            x[0] + FE::Real{0.5} * FE::Real{2.0} * x[1] * x[1];
        const FE::Real radius2 = x[0] * x[0] + x[1] * x[1];
        if (radius2 < origin_radius2) {
            origin_radius2 = radius2;
            origin = vertex;
        }
    }
    if (origin < 0 || origin_radius2 != FE::Real{0.0}) {
        throw std::runtime_error("structured curvature test has no origin vertex");
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);
    if (!result.success) {
        throw std::runtime_error(result.diagnostic);
    }
    return curvature.at(static_cast<std::size_t>(origin));
}

} // namespace

TEST(LevelSetCurvatureProjection, RecoversCircleCurvatureFromSignedDistance)
{
    StructuredQuadMeshAccess mesh(/*nx=*/16, /*ny=*/16, /*h=*/0.05);
    constexpr FE::Real radius = 0.30;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_GT(result.fitted_vertices, 200u);
    EXPECT_LT(result.zero_fallback_vertices, 12u);
    EXPECT_GE(result.max_normalized_fit_residual, FE::Real{0.0});
    EXPECT_GE(result.mean_normalized_fit_residual, FE::Real{0.0});

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        if (r < 0.25 || r > 0.35) {
            continue;
        }
        error_sum += std::abs(
            curvature[static_cast<std::size_t>(v)] - FE::Real{1.0} / r);
        ++samples;
    }
    ASSERT_GT(samples, 0u);
    EXPECT_LT(error_sum / static_cast<FE::Real>(samples), 0.08);
}

TEST(LevelSetCurvatureProjection, CircleCurvatureErrorImprovesWithRefinement)
{
    constexpr FE::Real radius = 0.30;
    const auto coarse_error =
        circleCurvatureMeanError(/*nx=*/8, /*h=*/0.10, radius);
    const auto fine_error =
        circleCurvatureMeanError(/*nx=*/16, /*h=*/0.05, radius);

    EXPECT_LT(fine_error, coarse_error);
    EXPECT_LT(fine_error, 0.08);
}

TEST(LevelSetCurvatureProjection,
     GeneratedCircleInterfacePatchCurvatureImprovesWithRefinement)
{
    const auto very_coarse = generatedCirclePatchCurvatureError(8);
    const auto coarse = generatedCirclePatchCurvatureError(16);
    const auto medium = generatedCirclePatchCurvatureError(32);
    const auto fine = generatedCirclePatchCurvatureError(64);
    const FE::Real coarse_order =
        observedOrder(coarse.mean_error, medium.mean_error);
    const FE::Real fine_order =
        observedOrder(medium.mean_error, fine.mean_error);

    RecordProperty("generated_circle_patch_mean_error_N8",
                   ::testing::PrintToString(very_coarse.mean_error));
    RecordProperty("generated_circle_patch_mean_error_N16",
                   ::testing::PrintToString(coarse.mean_error));
    RecordProperty("generated_circle_patch_mean_error_N32",
                   ::testing::PrintToString(medium.mean_error));
    RecordProperty("generated_circle_patch_mean_error_N64",
                   ::testing::PrintToString(fine.mean_error));
    RecordProperty("generated_circle_patch_max_error_N64",
                   ::testing::PrintToString(fine.max_error));
    RecordProperty("generated_circle_patch_order_16_to_32",
                   ::testing::PrintToString(coarse_order));
    RecordProperty("generated_circle_patch_order_32_to_64",
                   ::testing::PrintToString(fine_order));

    EXPECT_GT(very_coarse.supplemental_samples, 0u);
    EXPECT_GT(coarse.supplemental_samples, 0u);
    EXPECT_EQ(coarse.generated_geometry_samples, coarse.supplemental_samples);
    EXPECT_GT(coarse.generated_patch_fitted_vertices, 0u);
    EXPECT_EQ(coarse.generated_patch_expanded_vertices, 0u);
    EXPECT_GT(coarse.supplemental_samples,
              very_coarse.supplemental_samples);
    EXPECT_GT(medium.supplemental_samples, coarse.supplemental_samples);
    EXPECT_GT(fine.supplemental_samples, medium.supplemental_samples);
    EXPECT_LT(coarse.mean_error, very_coarse.mean_error);
    EXPECT_LT(medium.mean_error, coarse.mean_error);
    EXPECT_LT(fine.mean_error, medium.mean_error);
    EXPECT_GT(coarse_order, FE::Real{1.5});
    EXPECT_GT(fine_order, FE::Real{1.5});
    EXPECT_LT(fine.mean_error, FE::Real{0.05});
    EXPECT_LT(fine.max_error, FE::Real{0.10});
}

TEST(LevelSetCurvatureProjection,
     SpherePointCloudPatchCurvatureImprovesWithRefinement)
{
    const auto coarse = spherePointCloudPatchCurvatureError(8);
    const auto fine = spherePointCloudPatchCurvatureError(16);
    const FE::Real order =
        observedOrder(coarse.mean_error, fine.mean_error);

    RecordProperty("sphere_point_cloud_patch_mean_error_N8",
                   ::testing::PrintToString(coarse.mean_error));
    RecordProperty("sphere_point_cloud_patch_mean_error_N16",
                   ::testing::PrintToString(fine.mean_error));
    RecordProperty("sphere_point_cloud_patch_max_error_N16",
                   ::testing::PrintToString(fine.max_error));
    RecordProperty("sphere_point_cloud_patch_order_8_to_16",
                   ::testing::PrintToString(order));

    EXPECT_GT(coarse.supplemental_samples, 0u);
    EXPECT_EQ(coarse.generated_geometry_samples, coarse.supplemental_samples);
    EXPECT_GT(coarse.generated_patch_fitted_vertices, 0u);
    EXPECT_GT(fine.supplemental_samples, coarse.supplemental_samples);
    EXPECT_LT(fine.mean_error, coarse.mean_error);
    EXPECT_GT(order, FE::Real{1.5});
    EXPECT_LT(fine.mean_error,
              FE::Real{0.10} * FE::Real{2.0} / FE::Real{0.24});
}

TEST(LevelSetCurvatureProjection,
     GeneratedSphereFacetPatchCurvatureImprovesWithRefinement)
{
    const auto coarse = generatedSphereFacetPatchCurvatureError(8);
    const auto fine = generatedSphereFacetPatchCurvatureError(16);
    const FE::Real order =
        observedOrder(coarse.mean_error, fine.mean_error);

    RecordProperty("generated_sphere_facet_patch_mean_error_N8",
                   ::testing::PrintToString(coarse.mean_error));
    RecordProperty("generated_sphere_facet_patch_mean_error_N16",
                   ::testing::PrintToString(fine.mean_error));
    RecordProperty("generated_sphere_facet_patch_max_error_N16",
                   ::testing::PrintToString(fine.max_error));
    RecordProperty("generated_sphere_facet_patch_order_8_to_16",
                   ::testing::PrintToString(order));
    RecordProperty("generated_sphere_facet_patch_samples_N8",
                   coarse.supplemental_samples);
    RecordProperty("generated_sphere_facet_patch_samples_N16",
                   fine.supplemental_samples);
    RecordProperty("generated_sphere_facet_patch_band_vertices_N16",
                   fine.narrow_band_vertices);
    RecordProperty("generated_sphere_facet_patch_fitted_vertices_N16",
                   fine.generated_patch_fitted_vertices);
    RecordProperty("generated_sphere_facet_patch_expanded_vertices_N16",
                   fine.generated_patch_expanded_vertices);
    RecordProperty("generated_sphere_facet_patch_max_fit_residual_N16",
                   ::testing::PrintToString(fine.max_fit_residual));

    EXPECT_EQ(coarse.supplemental_samples, 516u);
    EXPECT_EQ(fine.supplemental_samples, 1920u);
    EXPECT_EQ(coarse.generated_geometry_samples, coarse.supplemental_samples);
    EXPECT_GT(coarse.generated_patch_fitted_vertices, 0u);
    EXPECT_GT(fine.supplemental_samples, coarse.supplemental_samples);
    EXPECT_EQ(fine.generated_patch_fitted_vertices,
              fine.narrow_band_vertices);
    EXPECT_EQ(fine.fallback_vertices, 0u);
    EXPECT_EQ(fine.zero_fallback_vertices, 0u);
    EXPECT_EQ(fine.insufficient_stencil_vertices, 0u);
    EXPECT_EQ(fine.singular_stencil_vertices, 0u);
    EXPECT_LT(fine.max_fit_residual, FE::Real{0.10});
    EXPECT_LT(fine.mean_error, coarse.mean_error);
    EXPECT_GT(order, FE::Real{1.5});
    EXPECT_LT(fine.mean_error,
              FE::Real{0.10} * FE::Real{2.0} / FE::Real{0.24});
    EXPECT_LT(fine.max_error,
              FE::Real{0.15} * FE::Real{2.0} / FE::Real{0.24});
}

TEST(LevelSetCurvatureProjection,
     GeneratedCircleInterfacePatchIsRigidMotionScaleAndSignInvariant)
{
    constexpr int subdivisions = 32;
    constexpr FE::Real h = FE::Real{0.8} / subdivisions;
    const std::array<FE::Real, 3> center{{
        FE::Real{0.31} * h,
        -FE::Real{0.27} * h,
        FE::Real{0.0},
    }};
    const auto baseline =
        generatedCirclePatchCurvatureError(subdivisions, center);
    const auto scaled = generatedCirclePatchCurvatureError(
        subdivisions, center, FE::Real{8.0});
    const auto reversed = generatedCirclePatchCurvatureError(
        subdivisions, center, -FE::Real{0.125});
    const auto transformed = generatedCirclePatchCurvatureError(
        subdivisions,
        center,
        FE::Real{1.0},
        FE::Real{0.37},
        {{FE::Real{1.75}, -FE::Real{0.85}, FE::Real{0.0}}});

    RecordProperty("generated_circle_patch_offset_mean_error_N32",
                   ::testing::PrintToString(baseline.mean_error));
    EXPECT_LT(baseline.mean_error, FE::Real{0.20});
    EXPECT_EQ(scaled.supplemental_samples, baseline.supplemental_samples);
    EXPECT_EQ(reversed.supplemental_samples, baseline.supplemental_samples);
    EXPECT_EQ(transformed.supplemental_samples,
              baseline.supplemental_samples);
    EXPECT_NEAR(scaled.mean_error, baseline.mean_error, FE::Real{1.0e-12});
    EXPECT_NEAR(reversed.mean_error,
                baseline.mean_error,
                FE::Real{1.0e-12});
    EXPECT_NEAR(transformed.mean_error,
                baseline.mean_error,
                FE::Real{1.0e-8});
    EXPECT_NEAR(scaled.max_error, baseline.max_error, FE::Real{1.0e-12});
    EXPECT_NEAR(reversed.max_error,
                baseline.max_error,
                FE::Real{1.0e-12});
    EXPECT_NEAR(transformed.max_error,
                baseline.max_error,
                FE::Real{1.0e-8});
}

TEST(LevelSetCurvatureProjection,
     StaticDropQ3SamplesProduceQuantifiedCurvatureRefinement)
{
    const auto coarse = staticDropQ3CurvatureError(16);
    const auto medium = staticDropQ3CurvatureError(32);
    const auto fine = staticDropQ3CurvatureError(64);
    const FE::Real coarse_order =
        observedOrder(coarse.mean_error, medium.mean_error);
    const FE::Real fine_order =
        observedOrder(medium.mean_error, fine.mean_error);

    RecordProperty("static_drop_mean_error_N16",
                   ::testing::PrintToString(coarse.mean_error));
    RecordProperty("static_drop_mean_error_N32",
                   ::testing::PrintToString(medium.mean_error));
    RecordProperty("static_drop_mean_error_N64",
                   ::testing::PrintToString(fine.mean_error));
    RecordProperty("static_drop_max_error_N64",
                   ::testing::PrintToString(fine.max_error));
    RecordProperty("static_drop_order_16_to_32",
                   ::testing::PrintToString(coarse_order));
    RecordProperty("static_drop_order_32_to_64",
                   ::testing::PrintToString(fine_order));
    RecordProperty("static_drop_Q3_samples_N64",
                   std::to_string(fine.supplemental_samples));
    RecordProperty("static_drop_fit_residual_N64",
                   ::testing::PrintToString(fine.max_fit_residual));
    RecordProperty("static_drop_Q3_interpolation_defect_N64",
                   ::testing::PrintToString(
                       fine.max_interpolation_defect));

    EXPECT_EQ(coarse.supplemental_samples, 4u * 16u * 16u);
    EXPECT_EQ(medium.supplemental_samples, 4u * 32u * 32u);
    EXPECT_EQ(fine.supplemental_samples, 4u * 64u * 64u);
    EXPECT_LT(medium.mean_error, coarse.mean_error);
    EXPECT_LT(fine.mean_error, medium.mean_error);
    EXPECT_GT(coarse_order, FE::Real{1.0});
    EXPECT_GT(fine_order, FE::Real{1.0});
    EXPECT_GT(fine.max_interpolation_defect, FE::Real{1.0e-12});
    EXPECT_LT(fine.mean_error, FE::Real{0.015});
}

TEST(LevelSetCurvatureProjection,
     SphereQ3SamplesProduceQuantifiedCurvatureRefinement)
{
    const auto coarse = sphereQ3CurvatureError(8);
    const auto fine = sphereQ3CurvatureError(16);
    const FE::Real order = observedOrder(coarse.mean_error, fine.mean_error);

    RecordProperty("sphere_mean_error_N8",
                   ::testing::PrintToString(coarse.mean_error));
    RecordProperty("sphere_mean_error_N16",
                   ::testing::PrintToString(fine.mean_error));
    RecordProperty("sphere_max_error_N16",
                   ::testing::PrintToString(fine.max_error));
    RecordProperty("sphere_order_8_to_16", ::testing::PrintToString(order));
    RecordProperty("sphere_Q3_samples_N16",
                   std::to_string(fine.supplemental_samples));
    RecordProperty("sphere_fit_residual_N16",
                   ::testing::PrintToString(fine.max_fit_residual));

    EXPECT_EQ(coarse.supplemental_samples, 8u * 8u * 8u * 8u);
    EXPECT_EQ(fine.supplemental_samples, 8u * 16u * 16u * 16u);
    EXPECT_LT(fine.mean_error, coarse.mean_error);
    EXPECT_GT(order, FE::Real{0.5});
    EXPECT_LT(fine.mean_error, FE::Real{0.12});
}

TEST(LevelSetCurvatureProjection,
     CapillaryWaveQ3SamplesProduceQuantifiedCurvatureRefinement)
{
    const auto coarse = capillaryWaveQ3CurvatureError(16);
    const auto medium = capillaryWaveQ3CurvatureError(32);
    const auto fine = capillaryWaveQ3CurvatureError(64);
    const FE::Real coarse_order =
        observedOrder(coarse.mean_error, medium.mean_error);
    const FE::Real fine_order =
        observedOrder(medium.mean_error, fine.mean_error);

    RecordProperty("wave_mean_error_N16",
                   ::testing::PrintToString(coarse.mean_error));
    RecordProperty("wave_mean_error_N32",
                   ::testing::PrintToString(medium.mean_error));
    RecordProperty("wave_mean_error_N64",
                   ::testing::PrintToString(fine.mean_error));
    RecordProperty("wave_max_error_N64",
                   ::testing::PrintToString(fine.max_error));
    RecordProperty("wave_order_16_to_32",
                   ::testing::PrintToString(coarse_order));
    RecordProperty("wave_order_32_to_64",
                   ::testing::PrintToString(fine_order));
    RecordProperty("wave_Q3_samples_N64",
                   std::to_string(fine.supplemental_samples));
    RecordProperty("wave_fit_residual_N64",
                   ::testing::PrintToString(fine.max_fit_residual));

    EXPECT_EQ(coarse.supplemental_samples, 4u * 16u * 8u);
    EXPECT_EQ(medium.supplemental_samples, 4u * 32u * 16u);
    EXPECT_EQ(fine.supplemental_samples, 4u * 64u * 32u);
    EXPECT_LT(medium.mean_error, coarse.mean_error);
    EXPECT_LT(fine.mean_error, medium.mean_error);
    EXPECT_GT(coarse_order, FE::Real{0.5});
    EXPECT_GT(fine_order, FE::Real{0.5});
    EXPECT_LT(fine.mean_error, FE::Real{0.002});
}

TEST(LevelSetCurvatureProjection,
     QuadraticRecoveryIsInvariantUnderMeshLengthUnits)
{
    EXPECT_NEAR(recoveredQuadraticCurvatureAtOrigin(1.0e-6, 1.0e-6),
                2.0,
                2.0e-5);
    EXPECT_NEAR(recoveredQuadraticCurvatureAtOrigin(1.0, 1.0),
                2.0,
                2.0e-11);
    EXPECT_NEAR(recoveredQuadraticCurvatureAtOrigin(1.0e6, 1.0e6),
                2.0,
                2.0e-5);
}

TEST(LevelSetCurvatureProjection,
     QuadraticRecoveryHandlesStrongCoordinateAnisotropy)
{
    // At a 10^6 aspect ratio the y^2 signal in mixed x/y nodal values is only
    // about four decimal digits above double-precision roundoff.  The bound is
    // therefore set by information already lost in the input coefficients,
    // not by the scaled QR solve.
    EXPECT_NEAR(recoveredQuadraticCurvatureAtOrigin(1.0e-6, 1.0),
                2.0,
                3.0e-5);
    EXPECT_NEAR(recoveredQuadraticCurvatureAtOrigin(1.0, 1.0e-6),
                2.0,
                3.0e-5);
}

TEST(LevelSetCurvatureProjection, NarrowBandRestrictsRecoveryToInterfaceVertices)
{
    StructuredQuadMeshAccess mesh(/*nx=*/16, /*ny=*/16, /*h=*/0.05);
    constexpr FE::Real radius = 0.30;
    constexpr FE::Real band_width = 0.055;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = band_width;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_DOUBLE_EQ(result.narrow_band_width, band_width);
    EXPECT_GT(result.narrow_band_vertices, 0u);
    EXPECT_GT(result.skipped_far_vertices, 0u);
    EXPECT_EQ(result.narrow_band_vertices + result.skipped_far_vertices,
              result.vertices);
    EXPECT_LT(result.fitted_vertices, result.vertices);

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto index = static_cast<std::size_t>(v);
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        if (std::abs(phi[index]) <= band_width) {
            error_sum += std::abs(curvature[index] - FE::Real{1.0} / r);
            ++samples;
        } else {
            EXPECT_DOUBLE_EQ(curvature[index], FE::Real{0.0});
        }
    }
    ASSERT_GT(samples, 0u);
    EXPECT_LT(error_sum / static_cast<FE::Real>(samples), 0.10);
}

TEST(LevelSetCurvatureProjection, RecoversSphereCurvatureFromSignedDistance)
{
    StructuredHexMeshAccess mesh(/*nx=*/10, /*ny=*/10, /*nz=*/10, /*h=*/0.06);
    constexpr FE::Real radius = 0.30;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_GT(result.fitted_vertices, 900u);
    EXPECT_EQ(result.zero_fallback_vertices, 0u);
    EXPECT_GE(result.max_normalized_fit_residual, FE::Real{0.0});
    EXPECT_GE(result.mean_normalized_fit_residual, FE::Real{0.0});

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        if (r < 0.24 || r > 0.36) {
            continue;
        }
        error_sum += std::abs(
            curvature[static_cast<std::size_t>(v)] - FE::Real{2.0} / r);
        ++samples;
    }
    ASSERT_GT(samples, 0u);
    EXPECT_LT(error_sum / static_cast<FE::Real>(samples), 0.35);
}

TEST(LevelSetCurvatureProjection, ReportsFailureForUnderresolvedStencil)
{
    StructuredQuadMeshAccess mesh(/*nx=*/1, /*ny=*/1, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, {}, curvature);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.fitted_vertices, 0u);
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_NE(result.diagnostic.find("could not fit"), std::string::npos);
}

TEST(LevelSetCurvatureProjection,
     GeneratedInterfacePatchFailsClosedWithoutGeometrySamples)
{
    StructuredQuadMeshAccess mesh(/*nx=*/4, /*ny=*/4, /*h=*/0.2);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()));
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto x = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] = x[0];
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::GeneratedInterfacePatch;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.generated_interface_geometry_samples, 0u);
    EXPECT_NE(result.diagnostic.find("requires generated interface geometry"),
              std::string::npos);
}

TEST(LevelSetCurvatureProjection, SupplementalSamplesAllowUnderresolvedQuadraticStencil)
{
    StructuredQuadMeshAccess mesh(/*nx=*/1, /*ny=*/1, /*h=*/1.0);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] = phi_function(mesh.getNodeCoordinates(v));
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}}})},
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{0.20}, FE::Real{-0.15}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.20}, FE::Real{-0.15}, FE::Real{0.0}}})},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 1;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_EQ(result.supplemental_samples, samples.size());
    EXPECT_DOUBLE_EQ(result.supplemental_sample_weight, FE::Real{1.0});
    EXPECT_EQ(result.vertices_with_supplemental_samples, phi.size());
    EXPECT_GE(result.supplemental_sample_rows, samples.size() * phi.size());
    EXPECT_EQ(result.fitted_vertices, phi.size());
    EXPECT_GT(result.max_abs_curvature, FE::Real{0.0});
    EXPECT_LT(result.max_normalized_fit_residual, FE::Real{1.0e-10});
    EXPECT_LT(result.mean_normalized_fit_residual, FE::Real{1.0e-10});
}

TEST(LevelSetCurvatureProjection, SupplementalSampleWeightControlsFitInfluence)
{
    StructuredQuadMeshAccess mesh(/*nx=*/3, /*ny=*/3, /*h=*/0.25);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] = phi_function(mesh.getNodeCoordinates(v));
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 4,
            .coordinate = {{FE::Real{0.04}, FE::Real{-0.03}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.04},
                                    FE::Real{-0.03},
                                    FE::Real{0.0}}}) +
                     FE::Real{0.08}},
    };

    level_set::LevelSetCurvatureProjectionOptions low_weight_options;
    low_weight_options.max_neighbor_rings = 2;
    low_weight_options.supplemental_sample_weight = FE::Real{1.0e-4};
    std::vector<FE::Real> low_weight_curvature;
    const auto low_weight = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, low_weight_options, low_weight_curvature);
    ASSERT_TRUE(low_weight.success) << low_weight.diagnostic;
    EXPECT_DOUBLE_EQ(low_weight.supplemental_sample_weight,
                     low_weight_options.supplemental_sample_weight);

    auto high_weight_options = low_weight_options;
    high_weight_options.supplemental_sample_weight = FE::Real{1.0e4};
    std::vector<FE::Real> high_weight_curvature;
    const auto high_weight = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, high_weight_options, high_weight_curvature);
    ASSERT_TRUE(high_weight.success) << high_weight.diagnostic;
    EXPECT_DOUBLE_EQ(high_weight.supplemental_sample_weight,
                     high_weight_options.supplemental_sample_weight);

    EXPECT_GT(maxAbsDifference(low_weight_curvature, high_weight_curvature),
              FE::Real{1.0e-2});
}

TEST(LevelSetCurvatureProjection, RejectsNonpositiveSupplementalSampleWeight)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{0.0});

    level_set::LevelSetCurvatureProjectionOptions options;
    options.supplemental_sample_weight = FE::Real{0.0};
    std::vector<FE::Real> curvature;
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection, RejectsNegativeNarrowBandWidth)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{0.0});

    level_set::LevelSetCurvatureProjectionOptions options;
    options.narrow_band_width = FE::Real{-1.0e-3};
    std::vector<FE::Real> curvature;
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection,
     RejectsInvalidKinematicAreaGradientFilterCoefficient)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{0.0});
    std::vector<FE::Real> curvature;

    level_set::LevelSetCurvatureProjectionOptions options;
    options.kinematic_area_gradient_filter_coefficient = FE::Real{-1.0};
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);

    options.kinematic_area_gradient_filter_coefficient =
        std::numeric_limits<FE::Real>::infinity();
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection,
     RejectsInvalidKinematicAreaGradientYoungWalls)
{
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/12, FE::Real{-0.70}, FE::Real{0.70});
    std::vector<FE::Real> phi(
        static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto point = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            std::hypot(point[0], point[1]) - FE::Real{0.347};
    }
    const FE::Real pi = std::acos(FE::Real{-1.0});
    std::vector<FE::Real> curvature;
    level_set::LevelSetCurvatureProjectionOptions options;
    options.kinematic_area_gradient_young_walls.push_back(
        {1, FE::Real{0.5} * pi});
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);

    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    options.kinematic_area_gradient_young_walls.front().boundary_marker = -1;
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);

    options.kinematic_area_gradient_young_walls.front().boundary_marker = 1;
    options.kinematic_area_gradient_young_walls.front()
        .equilibrium_contact_angle_radians = FE::Real{0.0};
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
    options.kinematic_area_gradient_young_walls.front()
        .equilibrium_contact_angle_radians = pi;
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
    options.kinematic_area_gradient_young_walls.front()
        .equilibrium_contact_angle_radians =
        std::numeric_limits<FE::Real>::infinity();
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);

    options.kinematic_area_gradient_young_walls = {
        {1, FE::Real{0.4} * pi}, {1, FE::Real{0.6} * pi}};
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);

    options.kinematic_area_gradient_young_walls = {
        {99, FE::Real{0.5} * pi}};
    const auto missing_marker =
        level_set::projectLevelSetMeanCurvatureToVertices(
            mesh, phi, options, curvature);
    EXPECT_FALSE(missing_marker.success);
    EXPECT_NE(missing_marker.diagnostic.find("no owned boundary faces"),
              std::string::npos);
}

TEST(LevelSetCurvatureProjection,
     RejectsPostSmoothingOfKinematicAreaGradientRecovery)
{
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/8, FE::Real{-0.70}, FE::Real{0.70});
    std::vector<FE::Real> phi(
        static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto point = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            std::hypot(point[0], point[1]) - FE::Real{0.35};
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    options.smoothing_iterations = 1;
    std::vector<FE::Real> curvature;
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection, RejectsInvalidLeastSquaresTolerances)
{
    StructuredQuadMeshAccess mesh(/*nx=*/6, /*ny=*/6, /*h=*/0.1);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{0.0});
    std::vector<FE::Real> curvature;

    level_set::LevelSetCurvatureProjectionOptions options;
    options.gradient_tolerance =
        std::numeric_limits<FE::Real>::infinity();
    EXPECT_THROW(
        (void)level_set::projectLevelSetMeanCurvatureToVertices(
            mesh, phi, options, curvature),
        std::invalid_argument);

    options = level_set::LevelSetCurvatureProjectionOptions{};
    options.normal_equation_tolerance =
        std::numeric_limits<FE::Real>::quiet_NaN();
    EXPECT_THROW(
        (void)level_set::projectLevelSetMeanCurvatureToVertices(
            mesh, phi, options, curvature),
        std::invalid_argument);

    options.normal_equation_tolerance = FE::Real{1.0};
    EXPECT_THROW(
        (void)level_set::projectLevelSetMeanCurvatureToVertices(
            mesh, phi, options, curvature),
        std::invalid_argument);
}

TEST(LevelSetCurvatureProjection, FailsClosedWhenNarrowBandHasNoVertices)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{1.0});

    level_set::LevelSetCurvatureProjectionOptions options;
    options.narrow_band_width = FE::Real{1.0e-3};
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.narrow_band_vertices, 0u);
    EXPECT_EQ(result.skipped_far_vertices, result.vertices);
    EXPECT_NE(result.diagnostic.find("narrow band"), std::string::npos);
}

TEST(LevelSetCurvatureProjection, FailsClosedWhenFitResidualLimitIsExceeded)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] = phi_function(mesh.getNodeCoordinates(v));
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{0.1}, FE::Real{0.2}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.1}, FE::Real{0.2}, FE::Real{0.0}}}) +
                     FE::Real{0.5}},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.max_normalized_fit_residual = 1.0e-6;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    EXPECT_FALSE(result.success);
    EXPECT_GT(result.fit_residual_failure_vertices, 0u);
    EXPECT_NE(result.diagnostic.find("residual"), std::string::npos);
}

TEST(LevelSetCurvatureProjection, FailsClosedWhenZeroFallbackLimitIsExceeded)
{
    StructuredQuadMeshAccess mesh(/*nx=*/16, /*ny=*/16, /*h=*/0.05);
    constexpr FE::Real radius = 0.25;
    constexpr FE::Real corner_plateau = 10.0;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
        if (x[0] <= FE::Real{-0.25} && x[1] <= FE::Real{-0.25}) {
            phi[static_cast<std::size_t>(v)] = corner_plateau;
        }
    }

    const std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{-0.375},
                            FE::Real{-0.375},
                            FE::Real{0.0}}},
            .value = corner_plateau},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = FE::Real{0.055};
    std::vector<FE::Real> curvature;
    const auto allowed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    ASSERT_TRUE(allowed.success) << allowed.diagnostic;
    ASSERT_GT(allowed.fitted_vertices, 0u);
    ASSERT_GT(allowed.zero_fallback_vertices, 0u);

    options.max_zero_fallback_vertices = 0;
    const auto rejected = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(rejected.zero_fallback_vertices, allowed.zero_fallback_vertices);
    EXPECT_NE(rejected.diagnostic.find("zero fallback vertices"),
              std::string::npos);
    EXPECT_NE(rejected.diagnostic.find("configured limit"),
              std::string::npos);
}

TEST(LevelSetCurvatureProjection,
     UsesBoundedFallbackWhenResidualOutliersHaveFittedNeighbors)
{
    StructuredQuadMeshAccess mesh(/*nx=*/8, /*ny=*/8, /*h=*/0.10);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    FE::GlobalIndex center_vertex = 0;
    FE::Real center_distance2 = std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] = phi_function(x);
        const auto distance2 = x[0] * x[0] + x[1] * x[1];
        if (distance2 < center_distance2) {
            center_distance2 = distance2;
            center_vertex = v;
        }
    }
    phi[static_cast<std::size_t>(center_vertex)] += FE::Real{0.20};

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.max_normalized_fit_residual = 2.0e-2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.fitted_vertices, 0u);
    EXPECT_GT(result.fit_residual_failure_vertices, 0u);
    EXPECT_GT(result.fallback_vertices, 0u);
    EXPECT_LT(result.zero_fallback_vertices, result.vertices);
    EXPECT_NE(result.diagnostic.find("neighbor fallback"), std::string::npos);
    for (const auto value : curvature) {
        EXPECT_TRUE(std::isfinite(value));
    }

    options.max_neighbor_fallback_vertices = 0;
    const auto rejected = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);
    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(rejected.fallback_vertices, result.fallback_vertices);
    EXPECT_NE(rejected.diagnostic.find("neighbor fallback vertices"),
              std::string::npos);
    EXPECT_NE(rejected.diagnostic.find("configured limit"),
              std::string::npos);
}

TEST(LevelSetCurvatureProjection, OptionalSmoothingReducesCurvatureGraphVariation)
{
    StructuredQuadMeshAccess mesh(/*nx=*/8, /*ny=*/8, /*h=*/0.08);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.20} * x[0] * x[0] +
               FE::Real{0.35} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    FE::GlobalIndex center_vertex = 0;
    FE::Real center_distance2 = std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] = phi_function(x);
        const FE::Real distance2 = x[0] * x[0] + x[1] * x[1];
        if (distance2 < center_distance2) {
            center_distance2 = distance2;
            center_vertex = v;
        }
    }
    phi[static_cast<std::size_t>(center_vertex)] += FE::Real{0.03};

    level_set::LevelSetCurvatureProjectionOptions unsmoothed_options;
    unsmoothed_options.max_neighbor_rings = 2;
    std::vector<FE::Real> unsmoothed_curvature;
    const auto unsmoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, unsmoothed_options, unsmoothed_curvature);
    ASSERT_TRUE(unsmoothed.success) << unsmoothed.diagnostic;
    EXPECT_EQ(unsmoothed.smoothing_iterations_applied, 0u);

    auto smoothed_options = unsmoothed_options;
    smoothed_options.smoothing_iterations = 3;
    smoothed_options.smoothing_relaxation = 0.5;
    std::vector<FE::Real> smoothed_curvature;
    const auto smoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, smoothed_options, smoothed_curvature);

    ASSERT_TRUE(smoothed.success) << smoothed.diagnostic;
    EXPECT_EQ(smoothed.smoothing_iterations_applied, 3u);
    EXPECT_GT(smoothed.smoothing_mean_abs_update, FE::Real{0.0});
    EXPECT_GT(smoothed.smoothing_max_abs_update, FE::Real{0.0});
    EXPECT_LT(curvatureGraphTotalVariation(mesh, smoothed_curvature),
              curvatureGraphTotalVariation(mesh, unsmoothed_curvature));
}

TEST(LevelSetCurvatureProjection,
     ParsesCurvatureSmoothingModesAndRejectsUnknownTokens)
{
    EXPECT_EQ(level_set::parseLevelSetCurvatureSmoothingMode("local_graph"),
              level_set::LevelSetCurvatureSmoothingMode::LocalGraph);
    EXPECT_EQ(level_set::parseLevelSetCurvatureSmoothingMode("mass-stiffness"),
              level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator);
    EXPECT_EQ(level_set::parseLevelSetCurvatureSmoothingMode("helmholtz"),
              level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator);
    EXPECT_STREQ(level_set::levelSetCurvatureSmoothingModeName(
                     level_set::LevelSetCurvatureSmoothingMode::
                         MassStiffnessOperator),
                 "mass_stiffness_operator");
    EXPECT_THROW((void)level_set::parseLevelSetCurvatureSmoothingMode(
                     "unsupported"),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection,
     ParsesCurvatureRecoveryModesAndRejectsUnknownTokens)
{
    EXPECT_EQ(level_set::parseLevelSetCurvatureRecoveryMode(
                  "level_set_quadratic"),
              level_set::LevelSetCurvatureRecoveryMode::LevelSetQuadratic);
    EXPECT_EQ(level_set::parseLevelSetCurvatureRecoveryMode(
                  "generated-interface-patch"),
              level_set::LevelSetCurvatureRecoveryMode::
                  GeneratedInterfacePatch);
    EXPECT_STREQ(level_set::levelSetCurvatureRecoveryModeName(
                     level_set::LevelSetCurvatureRecoveryMode::
                         GeneratedInterfacePatch),
                 "generated_interface_patch");
    EXPECT_EQ(level_set::parseLevelSetCurvatureRecoveryMode(
                  "kinematic-area-gradient"),
              level_set::LevelSetCurvatureRecoveryMode::
                  KinematicAreaGradient);
    EXPECT_STREQ(level_set::levelSetCurvatureRecoveryModeName(
                     level_set::LevelSetCurvatureRecoveryMode::
                         KinematicAreaGradient),
                 "kinematic_area_gradient");
    EXPECT_THROW((void)level_set::parseLevelSetCurvatureRecoveryMode(
                     "unsupported"),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientRecoversCircleAndSphereOnSimplexMeshes)
{
    const auto check_contract = [](const auto& mesh,
                                   const auto& evaluation) {
        const auto& result = evaluation.result;
        EXPECT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(evaluation.samples, result.fitted_vertices);
        EXPECT_EQ(result.kinematic_area_gradient_operator_vertices,
                  result.fitted_vertices);
        EXPECT_GT(result.kinematic_area_gradient_cut_cells, 0u);
        EXPECT_EQ(result.kinematic_area_gradient_measure_evaluations,
                  static_cast<std::size_t>(6 * (mesh.dimension() + 1)) *
                      result.kinematic_area_gradient_cut_cells);
        EXPECT_GT(result.kinematic_area_gradient_linear_iterations, 0u);
        EXPECT_EQ(result.kinematic_area_gradient_components, 1u);
        EXPECT_GT(result.kinematic_area_gradient_interface_measure,
                  FE::Real{0.0});
        EXPECT_DOUBLE_EQ(
            result.kinematic_area_gradient_filter_coefficient,
            FE::Real{1.0});
        EXPECT_GT(
            result.kinematic_area_gradient_min_characteristic_radius,
            FE::Real{0.0});
        EXPECT_GT(result.kinematic_area_gradient_min_filter_radius_cells,
                  FE::Real{0.0});
        EXPECT_GE(result.kinematic_area_gradient_max_filter_radius_cells,
                  result.kinematic_area_gradient_min_filter_radius_cells);
        EXPECT_GT(result.kinematic_area_gradient_min_filter_radius,
                  FE::Real{0.0});
        EXPECT_GE(result.kinematic_area_gradient_max_filter_radius,
                  result.kinematic_area_gradient_min_filter_radius);
        EXPECT_LT(result.kinematic_area_gradient_max_relative_fd_disagreement,
                  FE::Real{2.0e-5});
        EXPECT_LT(result.kinematic_area_gradient_relative_linear_residual,
                  FE::Real{1.0e-9});
        EXPECT_LT(
            result
                .kinematic_area_gradient_max_relative_regularized_identity_residual,
            FE::Real{1.0e-7});
    };

    auto circle_mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/24, FE::Real{-0.70}, FE::Real{0.70});
    const auto circle = evaluateKinematicCurvature(
        circle_mesh,
        FE::Real{0.437},
        {{FE::Real{0.013}, FE::Real{-0.021}, FE::Real{0.0}}});
    auto sphere_mesh = makeStructuredTetrahedronMesh(
        /*subdivisions=*/10, FE::Real{-0.65}, FE::Real{0.65});
    const auto sphere = evaluateKinematicCurvature(
        sphere_mesh,
        FE::Real{0.391},
        {{FE::Real{0.017}, FE::Real{-0.011}, FE::Real{0.009}}});

    check_contract(circle_mesh, circle);
    check_contract(sphere_mesh, sphere);
    RecordProperty("kinematic_area_gradient_circle_mean_error",
                   circle.mean_absolute_error);
    RecordProperty("kinematic_area_gradient_circle_max_error",
                   circle.maximum_absolute_error);
    RecordProperty("kinematic_area_gradient_circle_mean_value",
                   circle.mean_curvature);
    RecordProperty("kinematic_area_gradient_sphere_mean_error",
                   sphere.mean_absolute_error);
    RecordProperty("kinematic_area_gradient_sphere_max_error",
                   sphere.maximum_absolute_error);
    RecordProperty("kinematic_area_gradient_sphere_mean_value",
                   sphere.mean_curvature);
    EXPECT_LT(circle.mean_absolute_error, FE::Real{0.20});
    EXPECT_LT(circle.maximum_absolute_error, FE::Real{0.90});
    EXPECT_LT(sphere.mean_absolute_error, FE::Real{0.45});
    EXPECT_LT(sphere.maximum_absolute_error, FE::Real{1.70});
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientCurvatureConvergesUnderSimplexRefinement)
{
    std::array<FE::Real, 4> circle_errors{};
    constexpr std::array<int, 4> circle_levels{{12, 24, 48, 96}};
    for (std::size_t level = 0; level < circle_levels.size(); ++level) {
        auto mesh = makeStructuredTriangleMesh(
            circle_levels[level], FE::Real{-0.70}, FE::Real{0.70});
        const auto evaluation = evaluateKinematicCurvature(
            mesh,
            FE::Real{0.437},
            {{FE::Real{0.013}, FE::Real{-0.021}, FE::Real{0.0}}});
        ASSERT_TRUE(evaluation.result.success)
            << evaluation.result.diagnostic;
        circle_errors[level] =
            evaluation.mass_weighted_root_mean_square_error;
    }

    std::array<FE::Real, 3> sphere_errors{};
    constexpr std::array<int, 3> sphere_levels{{6, 10, 14}};
    for (std::size_t level = 0; level < sphere_levels.size(); ++level) {
        auto mesh = makeStructuredTetrahedronMesh(
            sphere_levels[level], FE::Real{-0.65}, FE::Real{0.65});
        const auto evaluation = evaluateKinematicCurvature(
            mesh,
            FE::Real{0.391},
            {{FE::Real{0.017}, FE::Real{-0.011}, FE::Real{0.009}}});
        ASSERT_TRUE(evaluation.result.success)
            << evaluation.result.diagnostic;
        sphere_errors[level] =
            evaluation.mass_weighted_root_mean_square_error;
    }

    const auto order = [](FE::Real coarse_error,
                          FE::Real fine_error,
                          int coarse_subdivisions,
                          int fine_subdivisions) {
        return std::log(coarse_error / fine_error) /
               std::log(static_cast<FE::Real>(fine_subdivisions) /
                        static_cast<FE::Real>(coarse_subdivisions));
    };
    const FE::Real circle_order_0 = order(
        circle_errors[0], circle_errors[1], circle_levels[0], circle_levels[1]);
    const FE::Real circle_order_1 = order(
        circle_errors[1], circle_errors[2], circle_levels[1], circle_levels[2]);
    const FE::Real circle_order_2 = order(
        circle_errors[2], circle_errors[3], circle_levels[2], circle_levels[3]);
    const FE::Real sphere_order_0 = order(
        sphere_errors[0], sphere_errors[1], sphere_levels[0], sphere_levels[1]);
    const FE::Real sphere_order_1 = order(
        sphere_errors[1], sphere_errors[2], sphere_levels[1], sphere_levels[2]);

    RecordProperty("kinematic_area_gradient_circle_rms_error_n12",
                   circle_errors[0]);
    RecordProperty("kinematic_area_gradient_circle_rms_error_n24",
                   circle_errors[1]);
    RecordProperty("kinematic_area_gradient_circle_rms_error_n48",
                   circle_errors[2]);
    RecordProperty("kinematic_area_gradient_circle_rms_error_n96",
                   circle_errors[3]);
    RecordProperty("kinematic_area_gradient_circle_order_12_24",
                   circle_order_0);
    RecordProperty("kinematic_area_gradient_circle_order_24_48",
                   circle_order_1);
    RecordProperty("kinematic_area_gradient_circle_order_48_96",
                   circle_order_2);
    RecordProperty("kinematic_area_gradient_sphere_rms_error_n6",
                   sphere_errors[0]);
    RecordProperty("kinematic_area_gradient_sphere_rms_error_n10",
                   sphere_errors[1]);
    RecordProperty("kinematic_area_gradient_sphere_rms_error_n14",
                   sphere_errors[2]);
    RecordProperty("kinematic_area_gradient_sphere_order_6_10",
                   sphere_order_0);
    RecordProperty("kinematic_area_gradient_sphere_order_10_14",
                   sphere_order_1);

    EXPECT_GT(circle_order_0, FE::Real{0.50});
    EXPECT_GT(circle_order_1, FE::Real{0.20});
    EXPECT_GT(circle_order_2, FE::Real{0.50});
    EXPECT_GT(sphere_order_0, FE::Real{0.35});
    EXPECT_GT(sphere_order_1, FE::Real{0.35});
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientCurvatureRespectsOrientationScaleAndRigidMotion)
{
    constexpr FE::Real radius{0.437};
    constexpr std::array<FE::Real, 3> center{{0.013, -0.021, 0.0}};
    auto base_mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/20, FE::Real{-0.70}, FE::Real{0.70});
    const auto base =
        evaluateKinematicCurvature(base_mesh, radius, center);
    const auto phi_scaled = evaluateKinematicCurvature(
        base_mesh, radius, center, FE::Real{7.25});
    const auto orientation_reversed = evaluateKinematicCurvature(
        base_mesh, radius, center, FE::Real{-3.40});

    constexpr FE::Real angle{0.47};
    const FE::Real cosine = std::cos(angle);
    const FE::Real sine = std::sin(angle);
    constexpr std::array<FE::Real, 3> shift{{0.31, -0.27, 0.0}};
    const auto rigid_transform = [&](const std::array<FE::Real, 3>& point) {
        return std::array<FE::Real, 3>{{
            cosine * point[0] - sine * point[1] + shift[0],
            sine * point[0] + cosine * point[1] + shift[1],
            point[2]}};
    };
    auto rigid_mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/20, FE::Real{-0.70}, FE::Real{0.70});
    rigid_mesh.transformCoordinates(rigid_transform);
    const auto rigid = evaluateKinematicCurvature(
        rigid_mesh, radius, rigid_transform(center));

    constexpr FE::Real physical_scale{2.30};
    const auto scale_transform = [](const std::array<FE::Real, 3>& point) {
        return std::array<FE::Real, 3>{{
            physical_scale * point[0],
            physical_scale * point[1],
            physical_scale * point[2]}};
    };
    auto physical_scale_mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/20, FE::Real{-0.70}, FE::Real{0.70});
    physical_scale_mesh.transformCoordinates(scale_transform);
    const auto physically_scaled = evaluateKinematicCurvature(
        physical_scale_mesh,
        physical_scale * radius,
        scale_transform(center));

    ASSERT_TRUE(base.result.success) << base.result.diagnostic;
    ASSERT_TRUE(phi_scaled.result.success) << phi_scaled.result.diagnostic;
    ASSERT_TRUE(orientation_reversed.result.success)
        << orientation_reversed.result.diagnostic;
    ASSERT_TRUE(rigid.result.success) << rigid.result.diagnostic;
    ASSERT_TRUE(physically_scaled.result.success)
        << physically_scaled.result.diagnostic;
    ASSERT_EQ(phi_scaled.curvature.size(), base.curvature.size());
    ASSERT_EQ(orientation_reversed.curvature.size(), base.curvature.size());
    ASSERT_EQ(rigid.curvature.size(), base.curvature.size());
    ASSERT_EQ(physically_scaled.curvature.size(), base.curvature.size());

    FE::Real maximum_phi_scale_difference{0.0};
    FE::Real maximum_orientation_sum{0.0};
    FE::Real maximum_rigid_difference{0.0};
    FE::Real maximum_physical_scale_difference{0.0};
    for (std::size_t vertex = 0; vertex < base.curvature.size(); ++vertex) {
        maximum_phi_scale_difference = std::max(
            maximum_phi_scale_difference,
            std::abs(phi_scaled.curvature[vertex] -
                     base.curvature[vertex]));
        maximum_orientation_sum = std::max(
            maximum_orientation_sum,
            std::abs(orientation_reversed.curvature[vertex] +
                     base.curvature[vertex]));
        maximum_rigid_difference = std::max(
            maximum_rigid_difference,
            std::abs(rigid.curvature[vertex] - base.curvature[vertex]));
        maximum_physical_scale_difference = std::max(
            maximum_physical_scale_difference,
            std::abs(physical_scale * physically_scaled.curvature[vertex] -
                     base.curvature[vertex]));
    }

    RecordProperty("kinematic_area_gradient_max_phi_scale_difference",
                   maximum_phi_scale_difference);
    RecordProperty("kinematic_area_gradient_max_orientation_sum",
                   maximum_orientation_sum);
    RecordProperty("kinematic_area_gradient_max_rigid_difference",
                   maximum_rigid_difference);
    RecordProperty("kinematic_area_gradient_max_physical_scale_difference",
                   maximum_physical_scale_difference);
    EXPECT_LT(maximum_phi_scale_difference, FE::Real{2.0e-6});
    EXPECT_LT(maximum_orientation_sum, FE::Real{2.0e-6});
    EXPECT_LT(maximum_rigid_difference, FE::Real{2.0e-6});
    EXPECT_LT(maximum_physical_scale_difference, FE::Real{2.0e-6});
    EXPECT_NEAR(
        physically_scaled.result.kinematic_area_gradient_max_filter_radius,
        physical_scale *
            base.result.kinematic_area_gradient_max_filter_radius,
        FE::Real{1.0e-13});
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientUsesIndependentDisconnectedComponentScales)
{
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/48, FE::Real{-1.20}, FE::Real{1.20});
    constexpr std::array<FE::Real, 3> center0{{-0.547, 0.031, 0.0}};
    constexpr std::array<FE::Real, 3> center1{{0.503, -0.043, 0.0}};
    constexpr FE::Real radius0{0.283};
    constexpr FE::Real radius1{0.193};
    std::vector<FE::Real> phi(
        static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto point = mesh.getNodeCoordinates(vertex);
        const auto distance = [&](const std::array<FE::Real, 3>& center) {
            const FE::Real dx = point[0] - center[0];
            const FE::Real dy = point[1] - center[1];
            return std::sqrt(dx * dx + dy * dy);
        };
        phi[static_cast<std::size_t>(vertex)] =
            std::min(distance(center0) - radius0,
                     distance(center1) - radius1);
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);
    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(result.kinematic_area_gradient_components, 2u);

    std::array<FE::Real, 2> mean_errors{};
    std::array<std::size_t, 2> samples{};
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const FE::Real value = curvature[static_cast<std::size_t>(vertex)];
        if (value == FE::Real{0.0}) {
            continue;
        }
        const auto point = mesh.getNodeCoordinates(vertex);
        const FE::Real d0 = std::hypot(point[0] - center0[0],
                                      point[1] - center0[1]);
        const FE::Real d1 = std::hypot(point[0] - center1[0],
                                      point[1] - center1[1]);
        const std::size_t component =
            std::abs(d0 - radius0) <= std::abs(d1 - radius1) ? 0u : 1u;
        const FE::Real exact =
            component == 0u ? FE::Real{1.0} / radius0
                            : FE::Real{1.0} / radius1;
        mean_errors[component] += std::abs(value - exact);
        ++samples[component];
    }
    ASSERT_GT(samples[0], 0u);
    ASSERT_GT(samples[1], 0u);
    mean_errors[0] /= static_cast<FE::Real>(samples[0]);
    mean_errors[1] /= static_cast<FE::Real>(samples[1]);

    const FE::Real expected_measure =
        FE::Real{2.0} * std::acos(FE::Real{-1.0}) * (radius0 + radius1);
    RecordProperty("kinematic_area_gradient_two_component_error_large",
                   mean_errors[0]);
    RecordProperty("kinematic_area_gradient_two_component_error_small",
                   mean_errors[1]);
    EXPECT_NEAR(result.kinematic_area_gradient_interface_measure,
                expected_measure,
                FE::Real{0.015} * expected_measure);
    EXPECT_NEAR(result.kinematic_area_gradient_min_characteristic_radius,
                radius1,
                FE::Real{0.015} * radius1);
    EXPECT_NEAR(result.kinematic_area_gradient_max_characteristic_radius,
                radius0,
                FE::Real{0.015} * radius0);
    EXPECT_LT(mean_errors[0], FE::Real{0.30});
    EXPECT_LT(mean_errors[1], FE::Real{0.50});
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientCanDisableRegularizationExplicitly)
{
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/16, FE::Real{-0.70}, FE::Real{0.70});
    const auto evaluation = evaluateKinematicCurvature(
        mesh,
        FE::Real{0.437},
        {{FE::Real{0.013}, FE::Real{-0.021}, FE::Real{0.0}}},
        FE::Real{1.0},
        FE::Real{0.0});
    ASSERT_TRUE(evaluation.result.success)
        << evaluation.result.diagnostic;
    EXPECT_EQ(evaluation.result.kinematic_area_gradient_linear_iterations, 0u);
    EXPECT_DOUBLE_EQ(
        evaluation.result.kinematic_area_gradient_min_filter_radius,
        FE::Real{0.0});
    EXPECT_DOUBLE_EQ(
        evaluation.result.kinematic_area_gradient_max_filter_radius,
        FE::Real{0.0});
    EXPECT_DOUBLE_EQ(
        evaluation.result.kinematic_area_gradient_min_filter_radius_cells,
        FE::Real{0.0});
    EXPECT_DOUBLE_EQ(
        evaluation.result.kinematic_area_gradient_max_filter_radius_cells,
        FE::Real{0.0});
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientRecordsAndResolvesExactVertexCrossings)
{
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/20, FE::Real{-0.70}, FE::Real{0.70});
    constexpr FE::Real radius{0.35};
    const auto evaluation = evaluateKinematicCurvature(
        mesh,
        radius,
        {{FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}}});
    ASSERT_TRUE(evaluation.result.success)
        << evaluation.result.diagnostic;
    EXPECT_GT(
        evaluation.result.kinematic_area_gradient_tie_break_vertices,
        0u);
    EXPECT_NE(evaluation.result.kinematic_area_gradient_tie_break_sign, 0);
    EXPECT_GT(evaluation.result.kinematic_area_gradient_max_tie_break_value,
              FE::Real{0.0});
    EXPECT_EQ(evaluation.result.kinematic_area_gradient_components, 1u);
    EXPECT_NEAR(
        evaluation.result.kinematic_area_gradient_max_characteristic_radius,
        radius,
        FE::Real{0.02} * radius);
    EXPECT_LT(evaluation.mean_absolute_error, FE::Real{0.30});
    EXPECT_LT(evaluation.maximum_absolute_error, FE::Real{1.20});
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientRegularizationPreservesMeanAndReducesVariation)
{
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/20, FE::Real{-0.70}, FE::Real{0.70});
    constexpr FE::Real radius{0.437};
    constexpr std::array<FE::Real, 3> center{{0.013, -0.021, 0.0}};
    constexpr std::array<FE::Real, 4> coefficients{{0.0, 0.5, 1.0, 2.0}};
    std::array<KinematicCurvatureEvaluation, coefficients.size()>
        evaluations{};
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        evaluations[index] = evaluateKinematicCurvature(
            mesh,
            radius,
            center,
            FE::Real{1.0},
            coefficients[index]);
        ASSERT_TRUE(evaluations[index].result.success)
            << evaluations[index].result.diagnostic;
    }

    const FE::Real reference_mean =
        evaluations.front()
            .result.kinematic_area_gradient_mass_weighted_mean_curvature;
    for (const auto& evaluation : evaluations) {
        EXPECT_NEAR(
            evaluation.result
                .kinematic_area_gradient_mass_weighted_mean_curvature,
            reference_mean,
            FE::Real{2.0e-8});
    }
    for (std::size_t index = 1; index < evaluations.size(); ++index) {
        EXPECT_LT(
            evaluations[index]
                .result.kinematic_area_gradient_mass_weighted_rms_deviation,
            evaluations[index - 1u]
                .result.kinematic_area_gradient_mass_weighted_rms_deviation);
    }
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientYoungWallRecoversSessileCircularCaps)
{
    const FE::Real pi = std::acos(FE::Real{-1.0});
    constexpr FE::Real radius{0.347};
    constexpr FE::Real wall_coordinate{-0.70};
    constexpr std::array<int, 3> levels{{32, 64, 128}};
    const std::array<FE::Real, 2> contact_angles{{
        pi / FE::Real{3.0}, FE::Real{2.0} * pi / FE::Real{3.0}}};
    for (std::size_t angle_index = 0;
         angle_index < contact_angles.size();
         ++angle_index) {
        std::array<FE::Real, levels.size()> young_errors{};
        std::array<FE::Real, levels.size()> surface_only_errors{};
        for (std::size_t level = 0; level < levels.size(); ++level) {
            auto mesh = makeStructuredTriangleMesh(
                levels[level], wall_coordinate, FE::Real{0.70});
            const auto young = evaluateSessileKinematicCurvature(
                mesh,
                radius,
                contact_angles[angle_index],
                wall_coordinate,
                true);
            const auto surface_only = evaluateSessileKinematicCurvature(
                mesh,
                radius,
                contact_angles[angle_index],
                wall_coordinate,
                false);
            ASSERT_TRUE(young.result.success) << young.result.diagnostic;
            ASSERT_TRUE(surface_only.result.success)
                << surface_only.result.diagnostic;
            EXPECT_EQ(young.result.kinematic_area_gradient_young_wall_count,
                      1u);
            EXPECT_EQ(
                young.result
                    .kinematic_area_gradient_young_wall_boundary_faces,
                static_cast<std::size_t>(levels[level]));
            EXPECT_EQ(
                young.result.kinematic_area_gradient_young_wall_cut_faces,
                2u);
            EXPECT_EQ(
                young.result
                    .kinematic_area_gradient_young_wall_measure_evaluations,
                24u);
            EXPECT_GT(
                young.result
                    .kinematic_area_gradient_surface_gradient_norm,
                FE::Real{0.0});
            EXPECT_GT(
                young.result
                    .kinematic_area_gradient_young_wall_gradient_norm,
                FE::Real{0.0});
            EXPECT_GT(
                young.result
                    .kinematic_area_gradient_total_energy_gradient_norm,
                FE::Real{0.0});
            EXPECT_LT(
                young.result
                    .kinematic_area_gradient_max_relative_regularized_identity_residual,
                FE::Real{1.0e-7});
            young_errors[level] =
                young.mass_weighted_root_mean_square_error;
            surface_only_errors[level] =
                surface_only.mass_weighted_root_mean_square_error;
            const std::string prefix = angle_index == 0u
                ? "kinematic_young_sessile_60"
                : "kinematic_young_sessile_120";
            RecordProperty(
                prefix + "_rms_error_n" +
                    std::to_string(levels[level]),
                young_errors[level]);
            RecordProperty(
                prefix + "_surface_only_rms_error_n" +
                    std::to_string(levels[level]),
                surface_only_errors[level]);
        }
        const FE::Real order_0 =
            std::log(young_errors[0] / young_errors[1]) /
            std::log(FE::Real{2.0});
        const FE::Real order_1 =
            std::log(young_errors[1] / young_errors[2]) /
            std::log(FE::Real{2.0});
        const std::string prefix = angle_index == 0u
            ? "kinematic_young_sessile_60"
            : "kinematic_young_sessile_120";
        RecordProperty(prefix + "_order_32_64", order_0);
        RecordProperty(prefix + "_order_64_128", order_1);
        EXPECT_LT(young_errors[1], young_errors[0]);
        EXPECT_LT(young_errors[2], young_errors[1]);
        EXPECT_LT(young_errors[2], FE::Real{0.025});
        EXPECT_LT(young_errors[2],
                  FE::Real{0.01} * surface_only_errors[2]);
        EXPECT_GT(order_0, FE::Real{0.60});
        EXPECT_GT(order_1, FE::Real{0.75});
    }
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientYoungWallIsNeutralAtRightAngle)
{
    const FE::Real pi = std::acos(FE::Real{-1.0});
    constexpr FE::Real wall_coordinate{-0.70};
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/48, wall_coordinate, FE::Real{0.70});
    const auto young = evaluateSessileKinematicCurvature(
        mesh,
        FE::Real{0.347},
        FE::Real{0.5} * pi,
        wall_coordinate,
        true);
    const auto surface_only = evaluateSessileKinematicCurvature(
        mesh,
        FE::Real{0.347},
        FE::Real{0.5} * pi,
        wall_coordinate,
        false);
    ASSERT_TRUE(young.result.success) << young.result.diagnostic;
    ASSERT_TRUE(surface_only.result.success)
        << surface_only.result.diagnostic;
    EXPECT_EQ(young.result.kinematic_area_gradient_young_wall_count, 1u);
    EXPECT_EQ(
        young.result.kinematic_area_gradient_young_wall_boundary_faces,
        48u);
    EXPECT_EQ(young.result.kinematic_area_gradient_young_wall_cut_faces, 2u);
    EXPECT_EQ(
        young.result.kinematic_area_gradient_young_wall_measure_evaluations,
        0u);
    EXPECT_DOUBLE_EQ(
        young.result.kinematic_area_gradient_young_wall_gradient_norm,
        FE::Real{0.0});
    EXPECT_DOUBLE_EQ(
        young.result.kinematic_area_gradient_total_energy_gradient_norm,
        young.result.kinematic_area_gradient_surface_gradient_norm);
    EXPECT_DOUBLE_EQ(maxAbsDifference(young.curvature,
                                     surface_only.curvature),
                     FE::Real{0.0});
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientYoungWallRespectsLevelSetOrientation)
{
    const FE::Real pi = std::acos(FE::Real{-1.0});
    constexpr FE::Real wall_coordinate{-0.70};
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/64, wall_coordinate, FE::Real{0.70});
    const auto positive = evaluateSessileKinematicCurvature(
        mesh,
        FE::Real{0.347},
        FE::Real{0.41} * pi,
        wall_coordinate,
        true,
        FE::Real{1.0});
    const auto negative = evaluateSessileKinematicCurvature(
        mesh,
        FE::Real{0.347},
        FE::Real{0.41} * pi,
        wall_coordinate,
        true,
        FE::Real{-2.75});
    ASSERT_TRUE(positive.result.success) << positive.result.diagnostic;
    ASSERT_TRUE(negative.result.success) << negative.result.diagnostic;
    ASSERT_EQ(positive.curvature.size(), negative.curvature.size());
    FE::Real maximum_orientation_sum{0.0};
    for (std::size_t vertex = 0; vertex < positive.curvature.size();
         ++vertex) {
        maximum_orientation_sum = std::max(
            maximum_orientation_sum,
            std::abs(positive.curvature[vertex] +
                     negative.curvature[vertex]));
    }
    RecordProperty("kinematic_young_max_orientation_sum",
                   maximum_orientation_sum);
    EXPECT_LT(maximum_orientation_sum, FE::Real{2.0e-6});
    EXPECT_EQ(
        positive.result.kinematic_area_gradient_young_wall_cut_faces,
        negative.result.kinematic_area_gradient_young_wall_cut_faces);
}

TEST(LevelSetCurvatureProjection,
     KinematicAreaGradientYoungWallMatchesTotalEnergyDerivative)
{
    const FE::Real pi = std::acos(FE::Real{-1.0});
    constexpr FE::Real wall_coordinate{-0.70};
    constexpr FE::Real radius{0.347};
    const FE::Real contact_angle = FE::Real{67.0} * pi / FE::Real{180.0};
    const FE::Real center_y =
        wall_coordinate - radius * std::cos(contact_angle);
    auto mesh = makeStructuredTriangleMesh(
        /*subdivisions=*/64, wall_coordinate, FE::Real{0.70});
    std::vector<FE::Real> phi(
        static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto point = mesh.getNodeCoordinates(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            std::hypot(point[0], point[1] - center_y) - radius;
    }
    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    options.kinematic_area_gradient_filter_coefficient = FE::Real{0.0};
    options.kinematic_area_gradient_negative_liquid_side = true;
    options.kinematic_area_gradient_young_walls.push_back(
        {1, contact_angle});
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);
    ASSERT_TRUE(result.success) << result.diagnostic;

    constexpr FE::Real step{2.0e-5};
    const FE::Real energy_m2 = directSessileDiscreteEnergy(
        mesh, phi, FE::Real{-2.0} * step, contact_angle, true);
    const FE::Real energy_m1 = directSessileDiscreteEnergy(
        mesh, phi, -step, contact_angle, true);
    const FE::Real energy_p1 = directSessileDiscreteEnergy(
        mesh, phi, step, contact_angle, true);
    const FE::Real energy_p2 = directSessileDiscreteEnergy(
        mesh, phi, FE::Real{2.0} * step, contact_angle, true);
    const FE::Real finite_difference =
        (energy_m2 - FE::Real{8.0} * energy_m1 +
         FE::Real{8.0} * energy_p1 - energy_p2) /
        (FE::Real{12.0} * step);
    const FE::Real assembled_derivative =
        -result.kinematic_area_gradient_kinematic_mass *
        result.kinematic_area_gradient_mass_weighted_mean_curvature;
    const FE::Real relative_difference =
        std::abs(finite_difference - assembled_derivative) /
        std::max(std::abs(finite_difference), FE::Real{1.0});
    RecordProperty("kinematic_young_energy_fd_derivative",
                   finite_difference);
    RecordProperty("kinematic_young_energy_assembled_derivative",
                   assembled_derivative);
    RecordProperty("kinematic_young_energy_relative_difference",
                   relative_difference);
    EXPECT_LT(relative_difference, FE::Real{2.0e-5});
}

TEST(LevelSetCurvatureProjection,
     MassStiffnessOperatorSmoothingReducesCurvatureGraphVariation)
{
    StructuredQuadMeshAccess mesh(/*nx=*/8, /*ny=*/8, /*h=*/0.08);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.20} * x[0] * x[0] +
               FE::Real{0.35} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    FE::GlobalIndex center_vertex = 0;
    FE::Real center_distance2 = std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] = phi_function(x);
        const FE::Real distance2 = x[0] * x[0] + x[1] * x[1];
        if (distance2 < center_distance2) {
            center_distance2 = distance2;
            center_vertex = v;
        }
    }
    phi[static_cast<std::size_t>(center_vertex)] += FE::Real{0.03};

    level_set::LevelSetCurvatureProjectionOptions unsmoothed_options;
    unsmoothed_options.max_neighbor_rings = 2;
    std::vector<FE::Real> unsmoothed_curvature;
    const auto unsmoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, unsmoothed_options, unsmoothed_curvature);
    ASSERT_TRUE(unsmoothed.success) << unsmoothed.diagnostic;

    auto smoothed_options = unsmoothed_options;
    smoothed_options.smoothing_mode =
        level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator;
    smoothed_options.smoothing_iterations = 2;
    smoothed_options.smoothing_relaxation = 0.5;
    std::vector<FE::Real> smoothed_curvature;
    const auto smoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, smoothed_options, smoothed_curvature);

    ASSERT_TRUE(smoothed.success) << smoothed.diagnostic;
    EXPECT_EQ(smoothed.smoothing_mode,
              level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator);
    EXPECT_EQ(smoothed.smoothing_iterations_applied, 2u);
    EXPECT_GT(smoothed.smoothing_operator_edges, 0u);
    EXPECT_GT(smoothed.smoothing_mean_abs_update, FE::Real{0.0});
    EXPECT_GT(smoothed.smoothing_max_abs_update, FE::Real{0.0});
    EXPECT_LT(curvatureGraphTotalVariation(mesh, smoothed_curvature),
              curvatureGraphTotalVariation(mesh, unsmoothed_curvature));
}

TEST(LevelSetCurvatureProjection, WorkspaceReusesMeshAndSampleAdjacency)
{
    StructuredQuadMeshAccess mesh(/*nx=*/4, /*ny=*/4, /*h=*/0.20);
    const auto phi_function = [](const std::array<FE::Real, 3>& x,
                                 FE::Real offset) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1] + offset;
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] =
            phi_function(mesh.getNodeCoordinates(v), FE::Real{0.0});
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}},
            .value = phi_function(
                {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}},
                FE::Real{0.0})},
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 5,
            .coordinate = {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}},
            .value = phi_function(
                {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}},
                FE::Real{0.0})},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    level_set::LevelSetCurvatureProjectionWorkspace workspace;

    std::vector<FE::Real> curvature;
    const auto first = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(first.success) << first.diagnostic;
    EXPECT_FALSE(first.reused_vertex_adjacency);
    EXPECT_FALSE(first.reused_sample_adjacency);
    EXPECT_EQ(first.vertex_adjacency_builds, 1u);
    EXPECT_EQ(first.sample_adjacency_builds, 1u);

    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] =
            phi_function(mesh.getNodeCoordinates(v), FE::Real{0.01});
    }
    for (auto& sample : samples) {
        sample.value += FE::Real{0.01};
    }

    const auto second = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(second.success) << second.diagnostic;
    EXPECT_TRUE(second.reused_vertex_adjacency);
    EXPECT_TRUE(second.reused_sample_adjacency);
    EXPECT_EQ(second.vertex_adjacency_builds, 1u);
    EXPECT_EQ(second.sample_adjacency_builds, 1u);

    samples.front().coordinate[0] += FE::Real{0.01};
    const auto third = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(third.success) << third.diagnostic;
    EXPECT_TRUE(third.reused_vertex_adjacency);
    EXPECT_TRUE(third.reused_sample_adjacency);
    EXPECT_EQ(third.vertex_adjacency_builds, 1u);
    EXPECT_EQ(third.sample_adjacency_builds, 1u);

    samples.front().parent_cell = static_cast<FE::MeshIndex>(-1);
    samples.front().value =
        phi_function(samples.front().coordinate, FE::Real{0.01});
    const auto fourth = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(fourth.success) << fourth.diagnostic;
    EXPECT_TRUE(fourth.reused_vertex_adjacency);
    EXPECT_FALSE(fourth.reused_sample_adjacency);
    EXPECT_EQ(fourth.vertex_adjacency_builds, 1u);
    EXPECT_EQ(fourth.sample_adjacency_builds, 2u);

    samples.front().coordinate[0] += FE::Real{0.01};
    samples.front().value =
        phi_function(samples.front().coordinate, FE::Real{0.01});
    const auto fifth = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(fifth.success) << fifth.diagnostic;
    EXPECT_TRUE(fifth.reused_vertex_adjacency);
    EXPECT_FALSE(fifth.reused_sample_adjacency);
    EXPECT_EQ(fifth.vertex_adjacency_builds, 1u);
    EXPECT_EQ(fifth.sample_adjacency_builds, 3u);

    samples.front().generated_interface_geometry = true;
    const auto sixth = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(sixth.success) << sixth.diagnostic;
    EXPECT_TRUE(sixth.reused_vertex_adjacency);
    EXPECT_FALSE(sixth.reused_sample_adjacency);
    EXPECT_EQ(sixth.vertex_adjacency_builds, 1u);
    EXPECT_EQ(sixth.sample_adjacency_builds, 4u);
}

TEST(LevelSetCurvatureProjection,
     WorkspaceRebuildsSampleAdjacencyWhenMeshGeometryRevisionChanges)
{
    StructuredQuadMeshAccess mesh(/*nx=*/4, /*ny=*/4, /*h=*/0.20);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };
    auto sample_value = [&](const std::array<FE::Real, 3>& x) {
        return phi_function(x);
    };
    auto vertex_phi = [&]() {
        std::vector<FE::Real> phi(
            static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
        for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
            phi[static_cast<std::size_t>(v)] =
                phi_function(mesh.getNodeCoordinates(v));
        }
        return phi;
    };

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}},
            .value = sample_value(
                {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}})},
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 5,
            .coordinate = {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}},
            .value = sample_value(
                {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}})},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    level_set::LevelSetCurvatureProjectionWorkspace workspace;

    std::vector<FE::Real> curvature;
    auto phi = vertex_phi();
    const auto first = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(first.success) << first.diagnostic;
    EXPECT_FALSE(first.reused_vertex_adjacency);
    EXPECT_FALSE(first.reused_sample_adjacency);
    EXPECT_EQ(first.vertex_adjacency_builds, 1u);
    EXPECT_EQ(first.sample_adjacency_builds, 1u);

    mesh.translate(FE::Real{0.02}, FE::Real{-0.01});
    phi = vertex_phi();
    const auto second = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(second.success) << second.diagnostic;
    EXPECT_TRUE(second.reused_vertex_adjacency);
    EXPECT_FALSE(second.reused_sample_adjacency);
    EXPECT_EQ(second.vertex_adjacency_builds, 1u);
    EXPECT_EQ(second.sample_adjacency_builds, 2u);
}

TEST(LevelSetCurvatureProjection,
     CarriesSnapshotRevisionAndRejectsMixedSamples)
{
    StructuredQuadMeshAccess mesh(/*nx=*/4, /*ny=*/4, /*h=*/0.20);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };
    std::vector<FE::Real> phi(
        static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        phi[static_cast<std::size_t>(vertex)] =
            phi_function(mesh.getNodeCoordinates(vertex));
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}},
            .value = phi_function(
                {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}}),
            .free_surface_snapshot_revision_key = 8101u,
            .source_value_revision = 9101u,
            .cut_topology_revision = 101u},
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 5,
            .coordinate = {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}},
            .value = phi_function(
                {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}}),
            .free_surface_snapshot_revision_key = 8101u,
            .source_value_revision = 9101u,
            .cut_topology_revision = 102u},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    level_set::LevelSetCurvatureProjectionWorkspace workspace;
    std::vector<FE::Real> curvature;
    const auto first = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(first.success) << first.diagnostic;
    EXPECT_EQ(first.free_surface_snapshot_revision_key, 8101u);
    EXPECT_EQ(first.source_value_revision, 9101u);
    EXPECT_EQ(workspace.free_surface_snapshot_revision_key, 8101u);
    EXPECT_EQ(workspace.source_value_revision, 9101u);
    EXPECT_EQ(first.sample_adjacency_builds, 1u);

    for (auto& sample : samples) {
        sample.free_surface_snapshot_revision_key = 8102u;
        sample.source_value_revision = 9102u;
    }
    const auto second = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(second.success) << second.diagnostic;
    EXPECT_EQ(second.free_surface_snapshot_revision_key, 8102u);
    EXPECT_EQ(second.source_value_revision, 9102u);
    EXPECT_EQ(workspace.free_surface_snapshot_revision_key, 8102u);
    EXPECT_EQ(workspace.source_value_revision, 9102u);
    EXPECT_FALSE(second.reused_sample_adjacency);
    EXPECT_EQ(second.sample_adjacency_builds, 2u);

    samples.back().free_surface_snapshot_revision_key = 8103u;
    EXPECT_THROW(
        (void)level_set::projectLevelSetMeanCurvatureToVertices(
            mesh, phi, samples, options, curvature, workspace),
        std::invalid_argument);

    samples.back().free_surface_snapshot_revision_key = 0u;
    samples.back().source_value_revision = 0u;
    samples.back().cut_topology_revision = 0u;
    EXPECT_THROW(
        (void)level_set::projectLevelSetMeanCurvatureToVertices(
            mesh, phi, samples, options, curvature, workspace),
        std::invalid_argument);
    samples.back().free_surface_snapshot_revision_key = 8102u;
    samples.back().source_value_revision = 9102u;
    samples.back().cut_topology_revision = 0u;
    EXPECT_THROW(
        (void)level_set::projectLevelSetMeanCurvatureToVertices(
            mesh, phi, samples, options, curvature, workspace),
        std::invalid_argument);
    RecordProperty("curvature_snapshot_mismatch_rejections", 2);
    RecordProperty("curvature_zero_topology_rejections", 1);
}
