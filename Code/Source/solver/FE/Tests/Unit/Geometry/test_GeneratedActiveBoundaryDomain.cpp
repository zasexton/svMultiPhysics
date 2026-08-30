#include "Assembly/CutDomainAssembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/StandardAssembler.h"
#include "Basis/NodeOrderingConventions.h"
#include "Dofs/DofMap.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace interfaces = svmp::FE::interfaces;

class SingleQuadBoundaryMesh final : public FE::assembly::IMeshAccess {
public:
    explicit SingleQuadBoundaryMesh(int marker = 7,
                                    int rank = 0,
                                    int size = 1,
                                    int owner_rank = 0,
                                    bool owned = true,
                                    FE::ElementType type = FE::ElementType::Quad4,
                                    std::vector<std::array<FE::Real, 3>>
                                        coordinates = {},
                                    bool expose_opposite_face = false,
                                    int geometry_order = 0)
        : marker_(marker)
        , rank_(rank)
        , size_(size)
        , owner_rank_(owner_rank)
        , owned_(owned)
        , type_(type)
        , coordinates_(std::move(coordinates))
        , expose_opposite_face_(expose_opposite_face)
        , geometry_order_(geometry_order > 0
                              ? geometry_order
                              : (type == FE::ElementType::Quad8 ||
                                         type == FE::ElementType::Quad9
                                     ? 2
                                     : 1))
    {
        if (coordinates_.empty()) {
            coordinates_ = {
                {{0.0, 0.0, 0.0}},
                {{1.0, 0.0, 0.0}},
                {{1.0, 1.0, 0.0}},
                {{0.0, 1.0, 0.0}},
            };
        }
    }

    void enableRevisionTracking(std::uint64_t geometry_revision = 11u,
                                std::uint64_t topology_revision = 12u,
                                std::uint64_t ownership_revision = 13u,
                                std::uint64_t numbering_revision = 0u)
    {
        track_revisions_ = true;
        geometry_revision_ = geometry_revision;
        topology_revision_ = topology_revision;
        ownership_revision_ = ownership_revision;
        numbering_revision_ = numbering_revision;
    }

    [[nodiscard]] bool revisionTrackingAvailable() const override {
        return track_revisions_;
    }
    [[nodiscard]] std::uint64_t geometryRevision() const override {
        return geometry_revision_;
    }
    [[nodiscard]] std::uint64_t topologyRevision() const override {
        return topology_revision_;
    }
    [[nodiscard]] std::uint64_t ownershipRevision() const override {
        return ownership_revision_;
    }
    [[nodiscard]] std::uint64_t numberingRevision() const override {
        return numbering_revision_;
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override {
        return owned_ ? 1 : 0;
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override {
        return expose_opposite_face_ ? 2 : 1;
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override {
        return size_ > 1;
    }
    [[nodiscard]] FE::GlobalIndex getCellGlobalId(
        FE::GlobalIndex cell) const override {
        return cell == 0 ? FE::GlobalIndex{7001} : FE::INVALID_GLOBAL_INDEX;
    }
    [[nodiscard]] FE::GlobalIndex getBoundaryFaceGlobalId(
        FE::GlobalIndex face) const override {
        if (face == 0) {
            return FE::GlobalIndex{9001};
        }
        return expose_opposite_face_ && face == 1
                   ? FE::GlobalIndex{9002}
                   : FE::INVALID_GLOBAL_INDEX;
    }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }
    [[nodiscard]] int getCellOwnerRank(FE::GlobalIndex) const override {
        return owner_rank_;
    }
    [[nodiscard]] int getBoundaryFaceOwnerRank(
        FE::GlobalIndex, FE::GlobalIndex) const override {
        return owner_rank_;
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override {
        return owned_ && cell == 0;
    }
    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override {
        return type_;
    }
    [[nodiscard]] int getCellGeometryOrder(FE::GlobalIndex) const override {
        return geometry_order_;
    }
    void getCellNodes(FE::GlobalIndex,
                      std::vector<FE::GlobalIndex>& nodes) const override {
        nodes.resize(coordinates_.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            nodes[i] = static_cast<FE::GlobalIndex>(i);
        }
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node) const override {
        return coordinates_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        FE::GlobalIndex,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override {
        coordinates = coordinates_;
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face,
        FE::GlobalIndex) const override {
        if (face == 0) {
            return FE::LocalIndex{0};
        }
        return expose_opposite_face_ && face == 1
                   ? FE::LocalIndex{2}
                   : FE::INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex face) const override {
        return face == 0 || (expose_opposite_face_ && face == 1)
                   ? marker_
                   : -1;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override {
        return {-1, -1};
    }
    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override {
        callback(0);
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override {
        if (owned_) {
            callback(0);
        }
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback)
        const override {
        if (marker < 0 || marker == marker_) {
            callback(0, 0);
            if (expose_opposite_face_) {
                callback(1, 0);
            }
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex,
                           FE::GlobalIndex,
                           FE::GlobalIndex)>) const override {
    }

private:
    int marker_{7};
    int rank_{0};
    int size_{1};
    int owner_rank_{0};
    bool owned_{true};
    FE::ElementType type_{FE::ElementType::Quad4};
    std::vector<std::array<FE::Real, 3>> coordinates_{};
    bool expose_opposite_face_{false};
    int geometry_order_{1};
    bool track_revisions_{false};
    std::uint64_t geometry_revision_{0u};
    std::uint64_t topology_revision_{0u};
    std::uint64_t ownership_revision_{0u};
    std::uint64_t numbering_revision_{0u};
};

class SingleHexBoundaryMesh final : public FE::assembly::IMeshAccess {
public:
    explicit SingleHexBoundaryMesh(int marker = 17)
        : marker_(marker)
    {
        coordinates_.reserve(8u);
        for (std::size_t node = 0u; node < 8u; ++node) {
            const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
                FE::ElementType::Hex8, node);
            coordinates_.push_back({{xi[0], xi[1], xi[2]}});
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override {
        return cell == 0;
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex) const override {
        return FE::ElementType::Hex8;
    }
    void getCellNodes(FE::GlobalIndex,
                      std::vector<FE::GlobalIndex>& nodes) const override {
        nodes.resize(coordinates_.size());
        std::iota(nodes.begin(), nodes.end(), FE::GlobalIndex{0});
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node) const override {
        return coordinates_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        FE::GlobalIndex,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override {
        coordinates = coordinates_;
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face,
        FE::GlobalIndex cell) const override {
        return face == 0 && cell == 0 ? FE::LocalIndex{0}
                                     : FE::INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex face) const override {
        return face == 0 ? marker_ : -1;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override {
        return {-1, -1};
    }
    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override {
        callback(0);
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override {
        callback(0);
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback)
        const override {
        if (marker < 0 || marker == marker_) {
            callback(0, 0);
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex,
                           FE::GlobalIndex,
                           FE::GlobalIndex)>) const override {
    }

private:
    int marker_{17};
    std::vector<std::array<FE::Real, 3>> coordinates_{};
};

class SingleTetraBoundaryMesh final : public FE::assembly::IMeshAccess {
public:
    explicit SingleTetraBoundaryMesh(int marker = 17)
        : marker_(marker)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override {
        return cell == 0;
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex) const override {
        return FE::ElementType::Tetra4;
    }
    void getCellNodes(FE::GlobalIndex,
                      std::vector<FE::GlobalIndex>& nodes) const override {
        nodes = {0, 1, 2, 3};
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node) const override {
        const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
            FE::ElementType::Tetra4,
            static_cast<std::size_t>(node));
        return {{xi[0], xi[1], xi[2]}};
    }
    void getCellCoordinates(
        FE::GlobalIndex,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override {
        coordinates.resize(4u);
        for (std::size_t node = 0u; node < coordinates.size(); ++node) {
            const auto xi =
                FE::basis::ReferenceNodeLayout::get_node_coords(
                    FE::ElementType::Tetra4, node);
            coordinates[node] = {{xi[0], xi[1], xi[2]}};
        }
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face,
        FE::GlobalIndex cell) const override {
        return face == 0 && cell == 0 ? FE::LocalIndex{0}
                                     : FE::INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex face) const override {
        return face == 0 ? marker_ : -1;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override {
        return {-1, -1};
    }
    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override {
        callback(0);
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override {
        callback(0);
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback)
        const override {
        if (marker < 0 || marker == marker_) {
            callback(0, 0);
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex,
                           FE::GlobalIndex,
                           FE::GlobalIndex)>) const override {
    }

private:
    int marker_{17};
};

interfaces::CutInterfaceDomainRequest interfaceRequest(int marker)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/4, /*layout_revision=*/3, /*value_revision=*/9);
    request.interface_marker = marker;
    request.quadrature_order = 2;
    request.interface_quadrature_order = 2;
    request.volume_quadrature_order = 2;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_status = "None";
    request.mesh_geometry_revision = 11;
    request.mesh_topology_revision = 12;
    request.ownership_revision = 13;
    request.quadrature_policy_key = 14;
    return request;
}

interfaces::LevelSetInterfaceDomain verticalInterface(
    int marker,
    FE::Real represented_root_residual = 0.0)
{
    interfaces::LevelSetInterfaceDomain domain(interfaceRequest(marker));
    interfaces::CutInterfaceFragment fragment;
    fragment.parent_cell = 0;
    fragment.kind = interfaces::CutInterfaceFragmentKind::Segment;
    fragment.measure = 2.0;
    fragment.normal = {{1.0, 0.0, 0.0}};
    fragment.min_gradient_norm = 1.0;
    fragment.vertices = {
        interfaces::CutInterfaceVertex{
            .point = {{0.0, -1.0, 0.0}},
            .parent_coordinate = {{0.0, -1.0, 0.0}}},
        interfaces::CutInterfaceVertex{
            .point = {{0.0, 1.0, 0.0}},
            .parent_coordinate = {{0.0, 1.0, 0.0}}},
    };
    constexpr FE::Real gauss_offset =
        FE::Real{0.57735026918962576451};
    fragment.quadrature_points = {
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.0, -gauss_offset, 0.0}},
            .parent_coordinate = {{0.0, -gauss_offset, 0.0}},
            .normal = fragment.normal,
            .weight = 1.0,
            .reference_measure_factor = 2.0,
            .level_set_residual = represented_root_residual,
            .gradient_norm = 1.0},
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.0, gauss_offset, 0.0}},
            .parent_coordinate = {{0.0, gauss_offset, 0.0}},
            .normal = fragment.normal,
            .weight = 1.0,
            .reference_measure_factor = 2.0,
            .level_set_residual = represented_root_residual,
            .gradient_norm = 1.0},
    };
    domain.addFragment(std::move(fragment));
    return domain;
}

interfaces::LevelSetInterfaceDomain verticalInterfaceAtX(
    int marker,
    FE::Real root)
{
    auto domain = interfaces::LevelSetInterfaceDomain(interfaceRequest(marker));
    interfaces::CutInterfaceFragment fragment;
    fragment.parent_cell = 0;
    fragment.kind = interfaces::CutInterfaceFragmentKind::Segment;
    fragment.measure = 2.0;
    fragment.normal = {{1.0, 0.0, 0.0}};
    fragment.min_gradient_norm = 1.0;
    fragment.vertices = {
        interfaces::CutInterfaceVertex{
            .point = {{root, -1.0, 0.0}},
            .parent_coordinate = {{root, -1.0, 0.0}}},
        interfaces::CutInterfaceVertex{
            .point = {{root, 1.0, 0.0}},
            .parent_coordinate = {{root, 1.0, 0.0}}},
    };
    fragment.quadrature_points = {
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{root, 0.0, 0.0}},
            .parent_coordinate = {{root, 0.0, 0.0}},
            .normal = fragment.normal,
            .weight = 2.0,
            .reference_measure_factor = 2.0,
            .gradient_norm = 1.0},
    };
    domain.addFragment(std::move(fragment));
    return domain;
}

interfaces::LevelSetInterfaceDomain diagonalHexInterface(int marker)
{
    auto domain = interfaces::LevelSetInterfaceDomain(interfaceRequest(marker));
    interfaces::CutInterfaceFragment fragment;
    fragment.parent_cell = 0;
    fragment.kind = interfaces::CutInterfaceFragmentKind::Polygon;
    fragment.measure = FE::Real{4.0} * std::sqrt(FE::Real{2.0});
    const FE::Real inverse_sqrt_two =
        FE::Real{1.0} / std::sqrt(FE::Real{2.0});
    fragment.normal = {{inverse_sqrt_two, inverse_sqrt_two, 0.0}};
    fragment.min_gradient_norm = std::sqrt(FE::Real{2.0});
    fragment.negative_volume_fraction = 0.5;
    fragment.positive_volume_fraction = 0.5;
    fragment.min_level_set_value = -2.0;
    fragment.max_level_set_value = 2.0;
    fragment.vertices = {
        interfaces::CutInterfaceVertex{
            .point = {{-1.0, 1.0, -1.0}},
            .parent_coordinate = {{-1.0, 1.0, -1.0}}},
        interfaces::CutInterfaceVertex{
            .point = {{1.0, -1.0, -1.0}},
            .parent_coordinate = {{1.0, -1.0, -1.0}}},
        interfaces::CutInterfaceVertex{
            .point = {{1.0, -1.0, 1.0}},
            .parent_coordinate = {{1.0, -1.0, 1.0}}},
        interfaces::CutInterfaceVertex{
            .point = {{-1.0, 1.0, 1.0}},
            .parent_coordinate = {{-1.0, 1.0, 1.0}}},
    };
    fragment.quadrature_points = {
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.0, 0.0, 0.0}},
            .parent_coordinate = {{0.0, 0.0, 0.0}},
            .normal = fragment.normal,
            .weight = fragment.measure,
            .reference_measure_factor = fragment.measure,
            .gradient_norm = fragment.min_gradient_norm},
    };
    domain.addFragment(std::move(fragment));
    return domain;
}

void addVerticalHalfVolumes(
    interfaces::LevelSetInterfaceDomain& domain,
    FE::Real negative_point_x = -0.5,
    FE::Real negative_weight = 2.0,
    int negative_achieved_order = 1,
    FE::Real positive_point_x = 0.5)
{
    const auto add_volume = [&domain,
                             negative_point_x,
                             negative_weight,
                             negative_achieved_order,
                             positive_point_x](
                                FE::geometry::CutIntegrationSide side,
                                FE::Real x,
                                FE::LocalIndex local_index) {
        if (side == FE::geometry::CutIntegrationSide::Negative) {
            x = negative_point_x;
        } else {
            x = positive_point_x;
        }
        interfaces::CutInterfaceVolumeRegion region;
        region.parent_cell = 0;
        region.local_region_index = local_index;
        region.side = side;
        region.centroid = {{x, 0.0, 0.0}};
        region.normal = side == FE::geometry::CutIntegrationSide::Negative
                            ? std::array<FE::Real, 3>{{1.0, 0.0, 0.0}}
                            : std::array<FE::Real, 3>{{-1.0, 0.0, 0.0}};
        region.measure = 2.0;
        region.parent_measure = 4.0;
        region.volume_fraction = 0.5;
        region.min_level_set_value =
            side == FE::geometry::CutIntegrationSide::Negative ? -1.0 : 0.0;
        region.max_level_set_value =
            side == FE::geometry::CutIntegrationSide::Negative ? 0.0 : 1.0;
        region.topology_id =
            side == FE::geometry::CutIntegrationSide::Negative
                ? "negative-half"
                : "positive-half";
        region.achieved_quadrature_order =
            side == FE::geometry::CutIntegrationSide::Negative
                ? negative_achieved_order
                : 1;
        region.quadrature_points = {
            FE::geometry::CutQuadraturePoint{
                .point = {{x, 0.0, 0.0}},
                .normal = region.normal,
                .weight = side == FE::geometry::CutIntegrationSide::Negative
                              ? negative_weight
                              : 2.0,
                .parent_coordinate = {{x, 0.0, 0.0}},
                .reference_measure_factor = 2.0}};
        domain.addVolumeRegion(std::move(region));
    };
    add_volume(FE::geometry::CutIntegrationSide::Negative, -0.5, 0u);
    add_volume(FE::geometry::CutIntegrationSide::Positive, 0.5, 1u);
}

interfaces::LevelSetInterfaceDomain verticalInterfaceSegment(
    int marker,
    FE::Real lower,
    FE::Real upper)
{
    auto domain = interfaces::LevelSetInterfaceDomain(interfaceRequest(marker));
    interfaces::CutInterfaceFragment fragment;
    fragment.parent_cell = 0;
    fragment.kind = interfaces::CutInterfaceFragmentKind::Segment;
    fragment.measure = upper - lower;
    fragment.normal = {{1.0, 0.0, 0.0}};
    fragment.min_gradient_norm = 1.0;
    fragment.vertices = {
        interfaces::CutInterfaceVertex{
            .point = {{0.0, lower, 0.0}},
            .parent_coordinate = {{0.0, lower, 0.0}}},
        interfaces::CutInterfaceVertex{
            .point = {{0.0, upper, 0.0}},
            .parent_coordinate = {{0.0, upper, 0.0}}},
    };
    constexpr FE::Real gauss_offset =
        FE::Real{0.57735026918962576451};
    const FE::Real midpoint = FE::Real{0.5} * (lower + upper);
    const FE::Real half_length = FE::Real{0.5} * (upper - lower);
    fragment.quadrature_points = {
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.0,
                       midpoint - half_length * gauss_offset,
                       0.0}},
            .parent_coordinate = {{0.0,
                                   midpoint - half_length * gauss_offset,
                                   0.0}},
            .normal = fragment.normal,
            .weight = half_length,
            .reference_measure_factor = fragment.measure,
            .gradient_norm = 1.0},
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.0,
                       midpoint + half_length * gauss_offset,
                       0.0}},
            .parent_coordinate = {{0.0,
                                   midpoint + half_length * gauss_offset,
                                   0.0}},
            .normal = fragment.normal,
            .weight = half_length,
            .reference_measure_factor = fragment.measure,
            .gradient_norm = 1.0},
    };
    domain.addFragment(std::move(fragment));
    addVerticalHalfVolumes(domain);
    return domain;
}

interfaces::LevelSetInterfaceDomain verticalInterfaceWithVolumes(
    int marker,
    FE::Real negative_point_x = -0.5,
    FE::Real negative_weight = 2.0,
    int negative_achieved_order = 1,
    FE::Real positive_point_x = 0.5)
{
    auto domain = verticalInterface(marker);
    addVerticalHalfVolumes(domain,
                           negative_point_x,
                           negative_weight,
                           negative_achieved_order,
                           positive_point_x);
    return domain;
}

interfaces::LevelSetInterfaceDomain distributedVerticalInterfaceWithVolumes(
    int marker,
    int owner_rank = 0)
{
    const auto source = verticalInterfaceWithVolumes(marker);
    interfaces::LevelSetInterfaceDomain distributed(source.request());
    for (auto fragment : source.fragments()) {
        fragment.parent_cell_global_id = 7001;
        fragment.owner_rank = owner_rank;
        fragment.stable_id = 0u;
        distributed.addFragment(std::move(fragment));
    }
    for (auto region : source.volumeRegions()) {
        region.parent_cell_global_id = 7001;
        region.owner_rank = owner_rank;
        region.stable_id = 0u;
        distributed.addVolumeRegion(std::move(region));
    }
    return distributed;
}

interfaces::LevelSetInterfaceDomain verticalInterfaceWithTinyNegativeVolume(
    int marker,
    FE::Real negative_fraction)
{
    auto domain = verticalInterface(marker);
    const auto add_volume = [&](FE::geometry::CutIntegrationSide side,
                                FE::Real fraction,
                                FE::Real x,
                                FE::LocalIndex local_index) {
        interfaces::CutInterfaceVolumeRegion region;
        region.parent_cell = 0;
        region.local_region_index = local_index;
        region.side = side;
        region.centroid = {{x, 0.0, 0.0}};
        region.normal = side == FE::geometry::CutIntegrationSide::Negative
                            ? std::array<FE::Real, 3>{{1.0, 0.0, 0.0}}
                            : std::array<FE::Real, 3>{{-1.0, 0.0, 0.0}};
        region.parent_measure = 4.0;
        region.measure = 4.0 * fraction;
        region.volume_fraction = fraction;
        region.min_level_set_value =
            side == FE::geometry::CutIntegrationSide::Negative ? -1.0 : 0.0;
        region.max_level_set_value =
            side == FE::geometry::CutIntegrationSide::Negative ? 0.0 : 1.0;
        region.topology_id =
            side == FE::geometry::CutIntegrationSide::Negative
                ? "tiny-negative"
                : "positive-complement";
        region.achieved_quadrature_order = 0;
        region.quadrature_points = {
            FE::geometry::CutQuadraturePoint{
                .point = {{x, 0.0, 0.0}},
                .normal = region.normal,
                .weight = region.measure,
                .parent_coordinate = {{x, 0.0, 0.0}},
                .reference_measure_factor = region.measure}};
        domain.addVolumeRegion(std::move(region));
    };
    add_volume(FE::geometry::CutIntegrationSide::Negative,
               negative_fraction,
               -0.999,
               0u);
    add_volume(FE::geometry::CutIntegrationSide::Positive,
               1.0 - negative_fraction,
               0.0,
               1u);
    return domain;
}

interfaces::LevelSetInterfaceDomain fullNegativeCell(int marker)
{
    interfaces::LevelSetInterfaceDomain domain(interfaceRequest(marker));
    interfaces::CutInterfaceVolumeRegion region;
    region.parent_cell = 0;
    region.local_region_index = 0u;
    region.side = FE::geometry::CutIntegrationSide::Negative;
    region.centroid = {{0.0, 0.0, 0.0}};
    region.normal = {{1.0, 0.0, 0.0}};
    region.parent_measure = 4.0;
    region.measure = 4.0;
    region.volume_fraction = 1.0;
    region.min_level_set_value = -1.0;
    region.max_level_set_value = -1.0;
    region.topology_id = "full-negative-cell";
    region.full_cell_equivalent = true;
    region.achieved_quadrature_order = 2;
    domain.addVolumeRegion(std::move(region));
    return domain;
}

interfaces::LevelSetInterfaceDomain linearQuadCutDomain(
    int marker,
    const std::array<FE::Real, 4>& level_set_values,
    FE::Real tolerance = 1.0e-12,
    FE::geometry::CutIntegrationSide aligned_parent_side =
        FE::geometry::CutIntegrationSide::Interface)
{
    auto request = interfaceRequest(marker);
    request.tolerance = tolerance;
    request.aligned_zero_interface_parent_side = aligned_parent_side;
    interfaces::LevelSetCellCutInput input;
    input.parent_cell = 0;
    input.element_type = FE::ElementType::Quad4;
    input.node_coordinates = {
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
    };
    input.level_set_values.assign(level_set_values.begin(),
                                  level_set_values.end());
    auto cut = interfaces::cutLinearLevelSetCell2D(request, input);
    interfaces::LevelSetInterfaceDomain domain(request);
    for (auto& fragment : cut.fragments) {
        domain.addFragment(std::move(fragment));
    }
    for (auto& region : cut.volume_regions) {
        domain.addVolumeRegion(std::move(region));
    }
    return domain;
}

interfaces::FreeSurfaceGeometryScalarEvaluator bilinearQuadScalar(
    std::array<FE::Real, 4> values)
{
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value =
        [values](FE::GlobalIndex,
                 const std::array<FE::Real, 3>& point,
                 const FE::geometry::CutQuadratureProvenance&) {
            const FE::Real x = point[0];
            const FE::Real y = point[1];
            return FE::Real{0.25} *
                   ((FE::Real{1.0} - x) * (FE::Real{1.0} - y) * values[0] +
                    (FE::Real{1.0} + x) * (FE::Real{1.0} - y) * values[1] +
                    (FE::Real{1.0} + x) * (FE::Real{1.0} + y) * values[2] +
                    (FE::Real{1.0} - x) * (FE::Real{1.0} + y) * values[3]);
        };
    scalar.reference_gradient =
        [values](FE::GlobalIndex,
                 const std::array<FE::Real, 3>& point,
                 const FE::geometry::CutQuadratureProvenance&) {
            const FE::Real x = point[0];
            const FE::Real y = point[1];
            return std::array<FE::Real, 3>{{
                FE::Real{0.25} *
                    (-(FE::Real{1.0} - y) * values[0] +
                     (FE::Real{1.0} - y) * values[1] +
                     (FE::Real{1.0} + y) * values[2] -
                     (FE::Real{1.0} + y) * values[3]),
                FE::Real{0.25} *
                    (-(FE::Real{1.0} - x) * values[0] -
                     (FE::Real{1.0} + x) * values[1] +
                     (FE::Real{1.0} + x) * values[2] +
                     (FE::Real{1.0} - x) * values[3]),
                FE::Real{0.0}}};
        };
    return scalar;
}

interfaces::LevelSetInterfaceDomain verticalHalfCutWithQuadrature(
    int marker,
    FE::ElementType element_type,
    int geometry_order)
{
    auto domain = verticalInterface(marker);
    const auto quadrature = FE::quadrature::QuadratureFactory::create(
        element_type, 2 * geometry_order);
    const auto add_region = [&](FE::geometry::CutIntegrationSide side,
                                FE::LocalIndex local_index) {
        const bool negative =
            side == FE::geometry::CutIntegrationSide::Negative;
        const FE::Real lower = negative ? FE::Real{-1.0} : FE::Real{0.0};
        const FE::Real upper = negative ? FE::Real{0.0} : FE::Real{1.0};
        interfaces::CutInterfaceVolumeRegion region;
        region.parent_cell = 0;
        region.local_region_index = local_index;
        region.side = side;
        region.centroid = {{FE::Real{0.5} * (lower + upper), 0.0, 0.0}};
        region.normal = negative
                            ? std::array<FE::Real, 3>{{1.0, 0.0, 0.0}}
                            : std::array<FE::Real, 3>{{-1.0, 0.0, 0.0}};
        region.parent_measure = 4.0;
        region.measure = 2.0;
        region.volume_fraction = 0.5;
        region.min_level_set_value = lower;
        region.max_level_set_value = upper;
        region.topology_id = negative ? "curved-negative-half"
                                      : "curved-positive-half";
        region.achieved_quadrature_order = 2;

        const auto add_triangle = [&](std::array<FE::Real, 3> a,
                                      std::array<FE::Real, 3> b,
                                      std::array<FE::Real, 3> c) {
            interfaces::CutInterfaceReferenceSimplex triangle;
            triangle.vertex_count = 3u;
            triangle.vertices[0] = a;
            triangle.vertices[1] = b;
            triangle.vertices[2] = c;
            triangle.has_represented_signed_values = true;
            triangle.represented_signed_values[0] = a[0];
            triangle.represented_signed_values[1] = b[0];
            triangle.represented_signed_values[2] = c[0];
            region.reference_subcells.push_back(std::move(triangle));
        };
        add_triangle({{lower, -1.0, 0.0}},
                     {{upper, -1.0, 0.0}},
                     {{upper, 1.0, 0.0}});
        add_triangle({{lower, -1.0, 0.0}},
                     {{upper, 1.0, 0.0}},
                     {{lower, 1.0, 0.0}});

        region.quadrature_points.reserve(quadrature->num_points());
        for (std::size_t q = 0u; q < quadrature->num_points(); ++q) {
            const auto point = quadrature->point(q);
            const FE::Real x = FE::Real{0.5} *
                               ((upper - lower) * point[0] + lower + upper);
            const FE::Real weight =
                FE::Real{0.5} * (upper - lower) * quadrature->weight(q);
            region.quadrature_points.push_back(
                FE::geometry::CutQuadraturePoint{
                    .point = {{x, point[1], point[2]}},
                    .normal = region.normal,
                    .weight = weight,
                    .parent_coordinate = {{x, point[1], point[2]}},
                    .reference_measure_factor = weight});
        }
        domain.addVolumeRegion(std::move(region));
    };
    add_region(FE::geometry::CutIntegrationSide::Negative, 0u);
    add_region(FE::geometry::CutIntegrationSide::Positive, 1u);
    return domain;
}

std::vector<std::array<FE::Real, 3>> curvedQuadCoordinates(int order)
{
    const auto reference_nodes =
        FE::basis::ReferenceNodeLayout::get_lagrange_node_coords(
            FE::ElementType::Quad4, order);
    std::vector<std::array<FE::Real, 3>> coordinates;
    coordinates.reserve(reference_nodes.size());
    for (const auto& xi : reference_nodes) {
        const FE::Real quadratic_scale =
            FE::Real{0.18} * (FE::Real{1.0} - xi[0] * xi[0]);
        const FE::Real cubic_scale =
            order == 3
                ? FE::Real{0.07} * xi[0] *
                      (FE::Real{1.0} - xi[0] * xi[0])
                : FE::Real{0.0};
        coordinates.push_back(
            {{xi[0], xi[1] * (FE::Real{1.0} + quadratic_scale +
                              cubic_scale), 0.0}});
    }
    return coordinates;
}

FE::dofs::DofMap singleCellDofMap(std::size_t dof_count)
{
    const auto local_count = static_cast<FE::LocalIndex>(dof_count);
    const auto global_count = static_cast<FE::GlobalIndex>(dof_count);
    FE::dofs::DofMap dof_map(1, global_count, local_count);
    std::vector<FE::GlobalIndex> cell_dofs(dof_count);
    std::iota(cell_dofs.begin(), cell_dofs.end(), FE::GlobalIndex{0});
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(global_count);
    dof_map.setNumLocalDofs(global_count);
    dof_map.finalize();
    return dof_map;
}

class ConstantOneCutVolumeKernel final
    : public FE::assembly::AssemblyKernel {
public:
    [[nodiscard]] FE::assembly::RequiredData getRequiredData() const override
    {
        return FE::assembly::RequiredData::BasisValues |
               FE::assembly::RequiredData::IntegrationWeights;
    }

    void computeCell(const FE::assembly::AssemblyContext& context,
                     FE::assembly::KernelOutput& output) override
    {
        const auto test_dof_count = context.numTestDofs();
        output.reserve(test_dof_count,
                       context.numTrialDofs(),
                       /*need_matrix=*/false,
                       /*need_vector=*/true);
        for (FE::LocalIndex q = 0; q < context.numQuadraturePoints(); ++q) {
            const FE::Real weight = context.integrationWeight(q);
            for (FE::LocalIndex i = 0; i < test_dof_count; ++i) {
                output.vectorEntry(i) +=
                    weight * context.basisValue(i, q);
            }
        }
    }
};

interfaces::FreeSurfaceGeometrySnapshotPolicy snapshotPolicyWithoutBoundary()
{
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    return policy;
}

interfaces::FreeSurfaceGeometryScalarEvaluator verticalScalar(
    FE::Real sign = 1.0,
    std::array<FE::Real, 3> gradient = {{1.0, 0.0, 0.0}})
{
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [sign](FE::GlobalIndex,
                          const std::array<FE::Real, 3>& xi,
                          const FE::geometry::CutQuadratureProvenance&) {
        return sign * xi[0];
    };
    scalar.reference_gradient =
        [gradient](FE::GlobalIndex,
                   const std::array<FE::Real, 3>&,
                   const FE::geometry::CutQuadratureProvenance&) {
            return gradient;
        };
    return scalar;
}

interfaces::GeneratedInterfaceBoundaryIntersectionRequest contactRequest(
    int interface_marker,
    int wall_marker)
{
    interfaces::GeneratedInterfaceBoundaryIntersectionRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/4, /*layout_revision=*/3, /*value_revision=*/9);
    request.generated_domain_id = "sharp_boundary_test";
    request.interface_marker = interface_marker;
    request.boundary_marker = wall_marker;
    request.quadrature_order = 2;
    request.mesh_geometry_revision = 11;
    request.mesh_topology_revision = 12;
    request.ownership_revision = 13;
    request.quadrature_policy_key = 14;
    request.source_value_revision = 9;
    return request;
}

interfaces::GeneratedActiveBoundaryRequest activeRequest(
    int interface_marker,
    int wall_marker,
    FE::geometry::CutIntegrationSide side)
{
    interfaces::GeneratedActiveBoundaryRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/4, /*layout_revision=*/3, /*value_revision=*/9);
    request.generated_domain_id = "sharp_boundary_test";
    request.interface_marker = interface_marker;
    request.boundary_marker = wall_marker;
    request.side = side;
    request.quadrature_order = 3;
    request.mesh_geometry_revision = 11;
    request.mesh_topology_revision = 12;
    request.ownership_revision = 13;
    request.quadrature_policy_key = 14;
    request.source_value_revision = 9;
    return request;
}

TEST(GeneratedActiveBoundaryDomain,
     RejectsNonfiniteRequestScalars)
{
    auto nonfinite_isovalue =
        activeRequest(
            101,
            8,
            FE::geometry::CutIntegrationSide::Negative);
    nonfinite_isovalue.isovalue =
        std::numeric_limits<FE::Real>::quiet_NaN();
    EXPECT_THROW(
        interfaces::GeneratedActiveBoundaryDomain(
            std::move(nonfinite_isovalue)),
        std::invalid_argument);

    auto nonfinite_tolerance =
        activeRequest(
            101,
            8,
            FE::geometry::CutIntegrationSide::Negative);
    nonfinite_tolerance.tolerance =
        std::numeric_limits<FE::Real>::infinity();
    EXPECT_THROW(
        interfaces::GeneratedActiveBoundaryDomain(
            std::move(nonfinite_tolerance)),
        std::invalid_argument);
}

struct ActiveBoundaryMoments {
    FE::Real constant{0.0};
    FE::Real x{0.0};
    FE::Real y{0.0};
    FE::Real x_squared{0.0};
    FE::Real x_y{0.0};
    FE::Real y_squared{0.0};
};

ActiveBoundaryMoments activeBoundaryMoments(
    const interfaces::GeneratedActiveBoundaryDomain& domain)
{
    ActiveBoundaryMoments moments;
    for (const auto& rule : domain.boundaryQuadratureRules()) {
        for (const auto& point : rule.points) {
            const FE::Real x = point.parent_coordinate[0];
            const FE::Real y = point.parent_coordinate[1];
            moments.constant += point.weight;
            moments.x += point.weight * x;
            moments.y += point.weight * y;
            moments.x_squared += point.weight * x * x;
            moments.x_y += point.weight * x * y;
            moments.y_squared += point.weight * y * y;
        }
    }
    return moments;
}

FE::Real intervalMonomialMoment(FE::Real lower,
                                FE::Real upper,
                                int degree)
{
    const int exponent = degree + 1;
    return (std::pow(upper, exponent) - std::pow(lower, exponent)) /
           static_cast<FE::Real>(exponent);
}

} // namespace

TEST(GeneratedActiveBoundaryDomain,
     HalfWetEdgeUsesAuthoritativeContactAndPartitionsExactly)
{
    constexpr int interface_marker = 101;
    constexpr int wall_marker = 7;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    const auto interface_domain = verticalInterface(interface_marker);
    const auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    ASSERT_EQ(contact_domain.summary().active_fragment_count, 1u);

    const std::array<FE::Real, 4> values{{-1.0, 1.0, 1.0, -1.0}};
    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [&values](FE::GlobalIndex node) {
        return values.at(static_cast<std::size_t>(node));
    };
    const auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Negative),
        interface_domain,
        contact_domain,
        mesh,
        field);
    const auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Positive),
        interface_domain,
        contact_domain,
        mesh,
        field);

    ASSERT_EQ(negative.fragments().size(), 1u);
    ASSERT_EQ(positive.fragments().size(), 1u);
    EXPECT_NEAR(negative.summary().measure, 1.0, 1.0e-14);
    EXPECT_NEAR(positive.summary().measure, 1.0, 1.0e-14);
    EXPECT_EQ(negative.fragments().front().source_contact_stable_ids.size(), 1u);
    EXPECT_EQ(negative.fragments().front().source_interface_stable_ids.size(), 1u);
    const auto negative_rules = negative.boundaryQuadratureRules();
    ASSERT_EQ(negative_rules.size(), 1u);
    EXPECT_EQ(negative_rules.front().provenance.implicit_geometry_mode,
              "LinearCorner");
    EXPECT_EQ(negative_rules.front()
                  .provenance.selected_implicit_quadrature_backend,
              "LinearCorner");
    const auto partition = interfaces::validateGeneratedActiveBoundaryPartition(
        negative, positive, interface_domain, contact_domain, mesh);
    EXPECT_EQ(partition.boundary_face_count, 1u);
    EXPECT_EQ(partition.orphan_source_reference_count, 0u);
    EXPECT_NEAR(partition.total_boundary_measure, 2.0, 1.0e-14);
    EXPECT_NEAR(partition.negative_boundary_measure, 1.0, 1.0e-14);
    EXPECT_NEAR(partition.positive_boundary_measure, 1.0, 1.0e-14);
    EXPECT_NEAR(partition.max_partition_error, 0.0, 1.0e-14);

    FE::assembly::CutIntegrationContext context;
    context.addGeneratedActiveBoundaryDomain(negative);
    context.addGeneratedActiveBoundaryDomain(positive);
    EXPECT_EQ(context.interfaceRulesForMarker(negative.marker()).size(), 1u);
    EXPECT_EQ(context.interfaceRulesForMarker(positive.marker()).size(), 1u);
}

TEST(GeneratedActiveBoundaryDomain,
     PlanarHalfAndQuarterFacesIntegratePolynomialsForBothSides)
{
    constexpr int wall_marker = 18;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    struct Scenario {
        FE::Real negative_fraction;
        int interface_marker;
    };
    const std::array<Scenario, 2> scenarios{{
        {0.5, 281},
        {0.25, 282},
    }};

    FE::Real maximum_constant_error{0.0};
    FE::Real maximum_polynomial_error{0.0};
    FE::Real maximum_partition_error{0.0};
    std::size_t side_count{0u};
    for (const auto& scenario : scenarios) {
        SCOPED_TRACE(scenario.negative_fraction);
        const FE::Real root =
            -1.0 + 2.0 * scenario.negative_fraction;
        const auto interface_domain =
            verticalInterfaceAtX(scenario.interface_marker, root);
        const auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(scenario.interface_marker, wall_marker),
                interface_domain,
                mesh);
        ASSERT_EQ(contact_domain.summary().active_fragment_count, 1u);

        const std::array<FE::Real, 4> nodal_values{{
            -scenario.negative_fraction,
            1.0 - scenario.negative_fraction,
            1.0 - scenario.negative_fraction,
            -scenario.negative_fraction}};
        interfaces::GeneratedActiveBoundaryScalarField field;
        field.value_at_node = [&nodal_values](FE::GlobalIndex node) {
            return nodal_values.at(static_cast<std::size_t>(node));
        };
        const auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(scenario.interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Negative),
            interface_domain,
            contact_domain,
            mesh,
            field);
        const auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(scenario.interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Positive),
            interface_domain,
            contact_domain,
            mesh,
            field);

        const auto check_side = [&](const auto& domain,
                                    FE::Real lower,
                                    FE::Real upper) {
            const auto rules = domain.boundaryQuadratureRules();
            ASSERT_EQ(rules.size(), 1u);
            const auto moments = activeBoundaryMoments(domain);
            const FE::Real expected_constant =
                intervalMonomialMoment(lower, upper, 0);
            const FE::Real expected_linear =
                intervalMonomialMoment(lower, upper, 1);
            const FE::Real expected_quadratic =
                intervalMonomialMoment(lower, upper, 2);
            const FE::Real constant_error =
                std::abs(moments.constant - expected_constant);
            const FE::Real polynomial_error = std::max(
                std::abs(moments.x - expected_linear),
                std::abs(moments.x_squared - expected_quadratic));
            EXPECT_NEAR(moments.constant, expected_constant, 1.0e-14);
            EXPECT_NEAR(moments.x, expected_linear, 1.0e-14);
            EXPECT_NEAR(moments.x_squared, expected_quadratic, 1.0e-14);
            maximum_constant_error =
                std::max(maximum_constant_error, constant_error);
            maximum_polynomial_error =
                std::max(maximum_polynomial_error, polynomial_error);
            ++side_count;
        };
        check_side(negative, -1.0, root);
        check_side(positive, root, 1.0);

        const auto partition =
            interfaces::validateGeneratedActiveBoundaryPartition(
                negative,
                positive,
                interface_domain,
                contact_domain,
                mesh);
        EXPECT_NEAR(partition.total_boundary_measure, 2.0, 1.0e-14);
        maximum_partition_error =
            std::max(maximum_partition_error, partition.max_partition_error);
    }

    EXPECT_EQ(side_count, 4u);
    RecordProperty("active_boundary_planar_fraction_case_count",
                   scenarios.size());
    RecordProperty("active_boundary_planar_fraction_side_count", side_count);
    RecordProperty("active_boundary_planar_max_constant_moment_error",
                   ::testing::PrintToString(maximum_constant_error));
    RecordProperty("active_boundary_planar_max_polynomial_moment_error",
                   ::testing::PrintToString(maximum_polynomial_error));
    RecordProperty("active_boundary_planar_max_partition_error",
                   ::testing::PrintToString(maximum_partition_error));
}

TEST(GeneratedActiveBoundaryDomain,
     ObliqueHexFaceIntegratesQuadraticMomentsForBothSides)
{
    constexpr int interface_marker = 283;
    constexpr int wall_marker = 20;
    const SingleHexBoundaryMesh mesh(wall_marker);
    const auto interface_domain = diagonalHexInterface(interface_marker);
    const auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    ASSERT_EQ(contact_domain.summary().active_fragment_count, 1u);
    EXPECT_NEAR(contact_domain.summary().measure,
                2.0 * std::sqrt(2.0),
                1.0e-14);

    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [](FE::GlobalIndex node) {
        const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
            FE::ElementType::Hex8, static_cast<std::size_t>(node));
        return xi[0] + xi[1];
    };
    const auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Negative),
        interface_domain,
        contact_domain,
        mesh,
        field);
    const auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Positive),
        interface_domain,
        contact_domain,
        mesh,
        field);
    ASSERT_EQ(negative.fragments().size(), 1u);
    ASSERT_EQ(positive.fragments().size(), 1u);
    EXPECT_EQ(negative.fragments().front().vertices.size(), 3u);
    EXPECT_EQ(positive.fragments().front().vertices.size(), 3u);

    const auto negative_moments = activeBoundaryMoments(negative);
    const auto positive_moments = activeBoundaryMoments(positive);
    struct ExpectedMoments {
        FE::Real constant;
        FE::Real x;
        FE::Real y;
        FE::Real x_squared;
        FE::Real x_y;
        FE::Real y_squared;
    };
    const std::array<ExpectedMoments, 2> expected{{
        {2.0, -2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 0.0, 2.0 / 3.0},
        {2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0, 2.0 / 3.0},
    }};
    const std::array<ActiveBoundaryMoments, 2> observed{{
        negative_moments,
        positive_moments,
    }};
    FE::Real maximum_constant_error{0.0};
    FE::Real maximum_polynomial_error{0.0};
    for (std::size_t side = 0u; side < observed.size(); ++side) {
        SCOPED_TRACE(side);
        const auto& actual = observed[side];
        const auto& reference = expected[side];
        maximum_constant_error = std::max(
            maximum_constant_error,
            std::abs(actual.constant - reference.constant));
        maximum_polynomial_error = std::max(
            maximum_polynomial_error,
            std::max({std::abs(actual.x - reference.x),
                      std::abs(actual.y - reference.y),
                      std::abs(actual.x_squared - reference.x_squared),
                      std::abs(actual.x_y - reference.x_y),
                      std::abs(actual.y_squared - reference.y_squared)}));
        EXPECT_NEAR(actual.constant, reference.constant, 1.0e-14);
        EXPECT_NEAR(actual.x, reference.x, 1.0e-14);
        EXPECT_NEAR(actual.y, reference.y, 1.0e-14);
        EXPECT_NEAR(actual.x_squared, reference.x_squared, 1.0e-14);
        EXPECT_NEAR(actual.x_y, reference.x_y, 1.0e-14);
        EXPECT_NEAR(actual.y_squared, reference.y_squared, 1.0e-14);
    }

    const auto partition =
        interfaces::validateGeneratedActiveBoundaryPartition(
            negative,
            positive,
            interface_domain,
            contact_domain,
            mesh);
    EXPECT_EQ(partition.boundary_face_count, 1u);
    EXPECT_EQ(partition.cut_boundary_face_count, 2u);
    EXPECT_NEAR(partition.total_boundary_measure, 4.0, 1.0e-14);
    EXPECT_NEAR(partition.negative_boundary_measure, 2.0, 1.0e-14);
    EXPECT_NEAR(partition.positive_boundary_measure, 2.0, 1.0e-14);

    RecordProperty("active_boundary_oblique_face_spatial_dimension", 3);
    RecordProperty("active_boundary_oblique_face_side_count", observed.size());
    RecordProperty("active_boundary_oblique_face_max_constant_moment_error",
                   ::testing::PrintToString(maximum_constant_error));
    RecordProperty("active_boundary_oblique_face_max_polynomial_moment_error",
                   ::testing::PrintToString(maximum_polynomial_error));
    RecordProperty("active_boundary_oblique_face_max_partition_error",
                   ::testing::PrintToString(partition.max_partition_error));
}

TEST(GeneratedActiveBoundaryDomain, CompletelyDrySideHasExactlyZeroRules)
{
    constexpr int interface_marker = 102;
    constexpr int wall_marker = 8;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    auto source_request = interfaceRequest(interface_marker);
    interfaces::LevelSetInterfaceDomain interface_domain(source_request);
    interfaces::LevelSetCellCutInput input;
    input.parent_cell = 0;
    input.element_type = FE::ElementType::Quad4;
    input.node_coordinates.reserve(4u);
    input.level_set_values.assign(4u, FE::Real{1.0});
    for (std::size_t node = 0u; node < 4u; ++node) {
        const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
            FE::ElementType::Quad4, node);
        input.node_coordinates.push_back({{xi[0], xi[1], xi[2]}});
    }
    interfaces::appendLinearLevelSetCellCut2D(interface_domain, input);
    ASSERT_TRUE(interface_domain.fragments().empty());
    ASSERT_EQ(interface_domain.volumeRegions().size(), 1u);
    EXPECT_TRUE(
        interface_domain.volumeRegions().front().full_cell_equivalent);
    EXPECT_EQ(interface_domain.volumeRegions().front().side,
              FE::geometry::CutIntegrationSide::Positive);
    const auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [](FE::GlobalIndex) { return FE::Real{1.0}; };

    const auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Negative),
        interface_domain,
        contact_domain,
        mesh,
        field);
    const auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Positive),
        interface_domain,
        contact_domain,
        mesh,
        field);
    EXPECT_TRUE(negative.empty());
    EXPECT_TRUE(negative.boundaryQuadratureRules().empty());
    EXPECT_EQ(negative.summary().measure, FE::Real{0.0});
    ASSERT_EQ(positive.fragments().size(), 1u);
    EXPECT_TRUE(positive.fragments().front().full_face_equivalent);
    EXPECT_NEAR(positive.summary().measure, 2.0, 1.0e-14);
    FE::assembly::CutIntegrationContext context;
    const auto revision_before_dry_import = context.contentRevision();
    context.addGeneratedActiveBoundaryDomain(negative);
    EXPECT_GT(context.contentRevision(), revision_before_dry_import);
    EXPECT_TRUE(context.hasGeneratedInterfaceMarker(negative.marker()));
    EXPECT_TRUE(context.interfaceRulesForMarker(negative.marker()).empty());
    EXPECT_NO_THROW(interfaces::validateGeneratedActiveBoundaryPartition(
        negative, positive, interface_domain, contact_domain, mesh));
    RecordProperty("active_boundary_dry_rule_count",
                   negative.boundaryQuadratureRules().size());
    RecordProperty("active_boundary_dry_measure",
                   ::testing::PrintToString(negative.summary().measure));
    RecordProperty("active_boundary_full_measure",
                   ::testing::PrintToString(positive.summary().measure));
}

TEST(GeneratedActiveBoundaryDomain,
     RejectsBoundaryClippingWithoutAuthoritativeCellPhase)
{
    constexpr int interface_marker = 103;
    constexpr int wall_marker = 8;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    interfaces::LevelSetInterfaceDomain interface_domain(
        interfaceRequest(interface_marker));
    const auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [](FE::GlobalIndex) { return FE::Real{1.0}; };

    EXPECT_THROW(
        (void)interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Positive),
            interface_domain,
            contact_domain,
            mesh,
            field),
        std::invalid_argument);
}

TEST(GeneratedActiveBoundaryDomain,
     EmptyActiveMarkerCannotBeReusedByContactDomain)
{
    constexpr int interface_marker = 102;
    constexpr int wall_marker = 8;
    constexpr int shared_generated_marker = 3102102;

    auto active_request = activeRequest(
        interface_marker,
        wall_marker,
        FE::geometry::CutIntegrationSide::Negative);
    active_request.active_boundary_marker = shared_generated_marker;
    interfaces::GeneratedActiveBoundaryDomain active_domain(
        std::move(active_request));
    ASSERT_TRUE(active_domain.empty());

    FE::assembly::CutIntegrationContext context;
    context.addGeneratedActiveBoundaryDomain(active_domain);
    const auto revision_before_contact = context.contentRevision();
    const auto interface_rule_count_before_contact =
        context.interfaceRules().size();
    const auto generated_markers_before_contact =
        context.generatedInterfaceMarkers();
    ASSERT_TRUE(
        context.hasGeneratedActiveBoundaryMarker(shared_generated_marker));
    ASSERT_NE(
        context.findGeneratedActiveBoundaryProvenance(
            shared_generated_marker),
        nullptr);
    ASSERT_EQ(
        context.generatedActiveBoundaryMarkersForPhysicalBoundary(
                   wall_marker)
            .size(),
        1u);

    auto contact_request =
        contactRequest(interface_marker, wall_marker);
    contact_request.intersection_marker = shared_generated_marker;
    interfaces::GeneratedInterfaceBoundaryIntersectionDomain
        contact_domain(std::move(contact_request));

    EXPECT_THROW(
        context.addGeneratedInterfaceBoundaryIntersectionDomain(
            contact_domain),
        std::invalid_argument);
    EXPECT_EQ(context.contentRevision(), revision_before_contact);
    EXPECT_EQ(context.interfaceRules().size(),
              interface_rule_count_before_contact);
    EXPECT_EQ(context.generatedInterfaceMarkers(),
              generated_markers_before_contact);
    EXPECT_TRUE(
        context.hasGeneratedActiveBoundaryMarker(shared_generated_marker));
    const auto* provenance =
        context.findGeneratedActiveBoundaryProvenance(
            shared_generated_marker);
    ASSERT_NE(provenance, nullptr);
    EXPECT_EQ(provenance->owner.boundary_marker, wall_marker);
    EXPECT_EQ(
        provenance->clipping_tolerance,
        active_domain.request().tolerance);
    const auto inverse_markers =
        context.generatedActiveBoundaryMarkersForPhysicalBoundary(
            wall_marker);
    ASSERT_EQ(inverse_markers.size(), 1u);
    EXPECT_EQ(inverse_markers.front(), shared_generated_marker);
}

TEST(GeneratedActiveBoundaryDomain,
     VolumeOnlyInterfaceMarkerCannotBeReusedByBoundaryDomains)
{
    constexpr int shared_generated_marker = 104;
    constexpr int wall_marker = 8;

    FE::assembly::CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(
        fullNegativeCell(shared_generated_marker));
    ASSERT_TRUE(
        context.hasGeneratedVolumeMarker(shared_generated_marker));
    ASSERT_FALSE(
        context.hasGeneratedInterfaceMarker(shared_generated_marker));
    const auto revision_before_boundary_domains =
        context.contentRevision();
    const auto volume_rule_count_before_boundary_domains =
        context.volumeRules().size();

    auto contact_request =
        contactRequest(shared_generated_marker, wall_marker);
    contact_request.intersection_marker =
        shared_generated_marker;
    interfaces::GeneratedInterfaceBoundaryIntersectionDomain
        contact_domain(std::move(contact_request));
    EXPECT_THROW(
        context.addGeneratedInterfaceBoundaryIntersectionDomain(
            contact_domain),
        std::invalid_argument);

    auto active_request = activeRequest(
        shared_generated_marker,
        wall_marker,
        FE::geometry::CutIntegrationSide::Negative);
    active_request.active_boundary_marker =
        shared_generated_marker;
    interfaces::GeneratedActiveBoundaryDomain active_domain(
        std::move(active_request));
    EXPECT_THROW(
        context.addGeneratedActiveBoundaryDomain(active_domain),
        std::invalid_argument);

    EXPECT_EQ(context.contentRevision(),
              revision_before_boundary_domains);
    EXPECT_EQ(context.volumeRules().size(),
              volume_rule_count_before_boundary_domains);
    EXPECT_TRUE(
        context.hasGeneratedVolumeMarker(shared_generated_marker));
    EXPECT_FALSE(
        context.hasGeneratedInterfaceMarker(shared_generated_marker));
    EXPECT_FALSE(
        context.hasGeneratedActiveBoundaryMarker(
            shared_generated_marker));
}

TEST(GeneratedActiveBoundaryDomain,
     EmptyLevelSetInterfaceMarkerCannotBeReusedByBoundaryDomains)
{
    constexpr int shared_generated_marker = 105;
    constexpr int wall_marker = 8;

    interfaces::LevelSetInterfaceDomain empty_interface(
        interfaceRequest(shared_generated_marker));
    ASSERT_TRUE(empty_interface.empty());

    FE::assembly::CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(empty_interface);
    ASSERT_TRUE(
        context.hasGeneratedLevelSetInterfaceMarker(
            shared_generated_marker));
    ASSERT_FALSE(
        context.hasGeneratedVolumeMarker(shared_generated_marker));
    ASSERT_FALSE(
        context.hasGeneratedInterfaceMarker(shared_generated_marker));
    const auto revision_before_boundary_domains =
        context.contentRevision();

    auto contact_request =
        contactRequest(shared_generated_marker, wall_marker);
    contact_request.intersection_marker =
        shared_generated_marker;
    interfaces::GeneratedInterfaceBoundaryIntersectionDomain
        contact_domain(std::move(contact_request));
    EXPECT_THROW(
        context.addGeneratedInterfaceBoundaryIntersectionDomain(
            contact_domain),
        std::invalid_argument);

    auto active_request = activeRequest(
        shared_generated_marker,
        wall_marker,
        FE::geometry::CutIntegrationSide::Negative);
    active_request.active_boundary_marker =
        shared_generated_marker;
    interfaces::GeneratedActiveBoundaryDomain active_domain(
        std::move(active_request));
    EXPECT_THROW(
        context.addGeneratedActiveBoundaryDomain(active_domain),
        std::invalid_argument);

    EXPECT_EQ(context.contentRevision(),
              revision_before_boundary_domains);
    EXPECT_TRUE(
        context.hasGeneratedLevelSetInterfaceMarker(
            shared_generated_marker));
    EXPECT_FALSE(
        context.hasGeneratedActiveBoundaryMarker(
            shared_generated_marker));
}

TEST(GeneratedActiveBoundaryDomain,
     GeneratedInterfaceMarkerNamespacesControlLaterVolumeRuleReuse)
{
    constexpr int wall_marker = 8;
    FE::assembly::CutIntegrationContext donor;
    donor.addGeneratedInterfaceDomain(
        fullNegativeCell(190));
    ASSERT_FALSE(donor.metadata().empty());
    ASSERT_FALSE(donor.volumeRules().empty());
    const auto metadata =
        donor.metadata().front();
    const auto rule =
        donor.volumeRules().front();

    const auto expect_volume_rejected =
        [&](FE::assembly::CutIntegrationContext&
                context,
            int marker) {
            const auto revision =
                context.contentRevision();
            const auto volume_rule_count =
                context.volumeRules().size();
            auto candidate_rule = rule;
            candidate_rule.provenance.marker =
                marker;
            EXPECT_THROW(
                context.addGeneratedVolumeRule(
                    marker,
                    metadata,
                    std::move(candidate_rule)),
                std::invalid_argument);
            EXPECT_EQ(
                context.contentRevision(),
                revision);
            EXPECT_EQ(
                context.volumeRules().size(),
                volume_rule_count);
        };

    constexpr int base_marker = 191;
    interfaces::LevelSetInterfaceDomain
        empty_interface(
            interfaceRequest(base_marker));
    FE::assembly::CutIntegrationContext
        base_context;
    base_context.addGeneratedInterfaceDomain(
        empty_interface);
    const auto base_revision =
        base_context.contentRevision();
    const auto base_volume_rule_count =
        base_context.volumeRules().size();
    auto base_rule = rule;
    base_rule.provenance.marker = base_marker;
    EXPECT_NO_THROW(
        base_context.addGeneratedVolumeRule(
            base_marker,
            metadata,
            std::move(base_rule)));
    EXPECT_GT(base_context.contentRevision(),
              base_revision);
    EXPECT_EQ(base_context.volumeRules().size(),
              base_volume_rule_count + 1u);
    EXPECT_TRUE(
        base_context.hasGeneratedVolumeMarker(
            base_marker));

    constexpr int contact_marker = 192;
    auto contact_request =
        contactRequest(
            contact_marker, wall_marker);
    contact_request.intersection_marker =
        contact_marker;
    interfaces::
        GeneratedInterfaceBoundaryIntersectionDomain
            contact_domain(
                std::move(contact_request));
    FE::assembly::CutIntegrationContext
        contact_context;
    contact_context
        .addGeneratedInterfaceBoundaryIntersectionDomain(
            contact_domain);
    expect_volume_rejected(
        contact_context, contact_marker);

    constexpr int active_marker = 193;
    auto active_request = activeRequest(
        active_marker,
        wall_marker,
        FE::geometry::CutIntegrationSide::
            Negative);
    active_request.active_boundary_marker =
        active_marker;
    interfaces::GeneratedActiveBoundaryDomain
        active_domain(
            std::move(active_request));
    FE::assembly::CutIntegrationContext
        active_context;
    active_context
        .addGeneratedActiveBoundaryDomain(
            active_domain);
    expect_volume_rejected(
        active_context, active_marker);
}

TEST(GeneratedActiveBoundaryDomain,
     DirectGeneratedVolumeRuleCanonicalizesAndChecksMarker)
{
    FE::assembly::CutIntegrationContext donor;
    donor.addGeneratedInterfaceDomain(
        fullNegativeCell(194));
    ASSERT_FALSE(donor.metadata().empty());
    ASSERT_FALSE(donor.volumeRules().empty());

    auto unbound_rule =
        donor.volumeRules().front();
    unbound_rule.provenance.marker = -1;
    FE::assembly::CutIntegrationContext context;
    context.addGeneratedVolumeRule(
        195,
        donor.metadata().front(),
        std::move(unbound_rule));
    ASSERT_EQ(context.volumeRules().size(), 1u);
    EXPECT_EQ(
        context.volumeRules().front().provenance.marker,
        195);

    auto mismatched_rule =
        donor.volumeRules().front();
    ASSERT_NE(
        mismatched_rule.provenance.marker,
        196);
    const auto revision = context.contentRevision();
    const auto rule_count =
        context.volumeRules().size();
    EXPECT_THROW(
        context.addGeneratedVolumeRule(
            196,
            donor.metadata().front(),
            std::move(mismatched_rule)),
        std::invalid_argument);
    EXPECT_EQ(context.contentRevision(), revision);
    EXPECT_EQ(context.volumeRules().size(), rule_count);
}

TEST(GeneratedActiveBoundaryDomain,
     RejectsScalarRootThatDoesNotMatchAuthoritativeContactTrace)
{
    constexpr int interface_marker = 103;
    constexpr int wall_marker = 9;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    const auto interface_domain = verticalInterface(interface_marker);
    const auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    const std::array<FE::Real, 4> values{{-0.5, 1.5, 1.5, -0.5}};
    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [&values](FE::GlobalIndex node) {
        return values.at(static_cast<std::size_t>(node));
    };
    EXPECT_THROW(
        interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Negative),
            interface_domain,
            contact_domain,
            mesh,
            field),
        std::invalid_argument);
}

TEST(GeneratedActiveBoundaryDomain,
     CollapsedTinyTetraCutUsesAuthoritativeFullCellBoundaryPhase)
{
    constexpr int interface_marker = 104;
    constexpr int wall_marker = 10;
    constexpr FE::Real epsilon = FE::Real{1.0e-8};
    const SingleTetraBoundaryMesh mesh(wall_marker);

    for (const FE::Real phase_sign : {FE::Real{1.0}, FE::Real{-1.0}}) {
        SCOPED_TRACE(phase_sign);
        auto request = interfaceRequest(interface_marker);
        FE::interfaces::LevelSetCellCutInput input;
        input.parent_cell = 0;
        input.element_type = FE::ElementType::Tetra4;
        input.node_coordinates.reserve(4u);
        for (std::size_t node = 0u; node < 4u; ++node) {
            const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
                FE::ElementType::Tetra4, node);
            input.node_coordinates.push_back({{xi[0], xi[1], xi[2]}});
        }
        const std::array<FE::Real, 4> values{{
            -phase_sign * epsilon,
            phase_sign * (FE::Real{1.0} - epsilon),
            phase_sign * (FE::Real{1.0} - epsilon),
            phase_sign * (FE::Real{1.0} - epsilon),
        }};
        input.level_set_values.assign(values.begin(), values.end());
        auto cut = interfaces::cutLinearLevelSetCell3D(request, input);
        ASSERT_TRUE(cut.supported);
        EXPECT_TRUE(cut.fragments.empty());
        ASSERT_EQ(cut.volume_regions.size(), 1u);
        EXPECT_TRUE(cut.volume_regions.front().full_cell_equivalent);
        const auto dominant_side =
            phase_sign > FE::Real{0.0}
                ? FE::geometry::CutIntegrationSide::Positive
                : FE::geometry::CutIntegrationSide::Negative;
        EXPECT_EQ(cut.volume_regions.front().side, dominant_side);

        interfaces::LevelSetInterfaceDomain interface_domain(request);
        interface_domain.addVolumeRegion(
            std::move(cut.volume_regions.front()));
        const auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(interface_marker, wall_marker),
                interface_domain,
                mesh);
        EXPECT_TRUE(contact_domain.empty());

        interfaces::GeneratedActiveBoundaryScalarField field;
        field.value_at_node = [values](FE::GlobalIndex node) {
            return values.at(static_cast<std::size_t>(node));
        };
        const auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Negative),
            interface_domain,
            contact_domain,
            mesh,
            field);
        const auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Positive),
            interface_domain,
            contact_domain,
            mesh,
            field);
        const auto& dominant =
            dominant_side == FE::geometry::CutIntegrationSide::Negative
                ? negative
                : positive;
        const auto& collapsed =
            dominant_side == FE::geometry::CutIntegrationSide::Negative
                ? positive
                : negative;
        ASSERT_EQ(dominant.fragments().size(), 1u);
        EXPECT_TRUE(dominant.fragments().front().full_face_equivalent);
        EXPECT_GT(dominant.summary().measure, FE::Real{0.0});
        EXPECT_EQ(collapsed.summary().measure, FE::Real{0.0});
        const auto partition =
            interfaces::validateGeneratedActiveBoundaryPartition(
                negative,
                positive,
                interface_domain,
                contact_domain,
                mesh);
        EXPECT_EQ(partition.max_partition_error, FE::Real{0.0});
    }

    RecordProperty("collapsed_tiny_tetra_phase_reversal_count", 2);
}

TEST(GeneratedActiveBoundaryDomain,
     SnapshotPrunesBoundaryTraceWithoutRetainedParentVolume)
{
    constexpr int interface_marker = 106;
    constexpr int wall_marker = 12;
    constexpr FE::Real epsilon = FE::Real{1.0e-8};
    const SingleTetraBoundaryMesh mesh(wall_marker);
    auto request = interfaceRequest(interface_marker);
    request.interface_quadrature_order = 1;
    FE::interfaces::LevelSetCellCutInput input;
    input.parent_cell = 0;
    input.element_type = FE::ElementType::Tetra4;
    input.node_coordinates.reserve(4u);
    for (std::size_t node = 0u; node < 4u; ++node) {
        const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
            FE::ElementType::Tetra4, node);
        input.node_coordinates.push_back({{xi[0], xi[1], xi[2]}});
    }
    const std::array<FE::Real, 4> values{{
        -epsilon,
        -epsilon,
        FE::Real{1.0} - epsilon,
        FE::Real{1.0} - epsilon,
    }};
    input.level_set_values.assign(values.begin(), values.end());
    auto cut = interfaces::cutLinearLevelSetCell3D(request, input);
    ASSERT_TRUE(cut.supported);
    ASSERT_EQ(cut.fragments.size(), 1u);
    ASSERT_EQ(cut.volume_regions.size(), 2u);

    interfaces::LevelSetInterfaceDomain interface_domain(request);
    for (auto& fragment : cut.fragments) {
        interface_domain.addFragment(std::move(fragment));
    }
    for (auto& region : cut.volume_regions) {
        interface_domain.addVolumeRegion(std::move(region));
    }
    auto contact =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    ASSERT_EQ(contact.summary().active_fragment_count, 1u);
    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [values](FE::GlobalIndex node) {
        return values.at(static_cast<std::size_t>(node));
    };
    auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Negative),
        interface_domain,
        contact,
        mesh,
        field);
    auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Positive),
        interface_domain,
        contact,
        mesh,
        field);
    ASSERT_EQ(negative.fragments().size(), 1u);
    ASSERT_EQ(positive.fragments().size(), 1u);
    const int negative_marker = negative.marker();
    const int positive_marker = positive.marker();

    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [epsilon](FE::GlobalIndex,
                            const std::array<FE::Real, 3>& point,
                            const FE::geometry::CutQuadratureProvenance&) {
        return point[1] + point[2] - epsilon;
    };
    scalar.reference_gradient =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return std::array<FE::Real, 3>{{0.0, 1.0, 1.0}};
        };
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.minimum_retained_volume_fraction =
        FE::assembly::CutIntegrationContext::
            minGeneratedCutVolumeFraction();
    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        {std::move(contact)},
        {std::move(negative), std::move(positive)},
        mesh,
        policy,
        std::move(scalar),
        "sharp_boundary_test");
    ASSERT_TRUE(snapshot);
    const auto negative_record = std::find_if(
        snapshot->rules().begin(),
        snapshot->rules().end(),
        [](const auto& record) {
            return record.role == interfaces::
                                      FreeSurfaceGeometryRuleRole::
                                          NegativeExteriorBoundary;
        });
    ASSERT_NE(negative_record, snapshot->rules().end());
    EXPECT_EQ(negative_record->retention,
              interfaces::FreeSurfaceGeometryRetention::PrunedSmallVolume);
    const auto positive_record = std::find_if(
        snapshot->rules().begin(),
        snapshot->rules().end(),
        [](const auto& record) {
            return record.role == interfaces::
                                      FreeSurfaceGeometryRuleRole::
                                          PositiveExteriorBoundary;
        });
    ASSERT_NE(positive_record, snapshot->rules().end());
    EXPECT_EQ(positive_record->retention,
              interfaces::FreeSurfaceGeometryRetention::Retained);

    FE::assembly::CutIntegrationContext context;
    context.addFreeSurfaceGeometrySnapshot(snapshot);
    EXPECT_TRUE(context.interfaceRulesForMarker(negative_marker).empty());
    EXPECT_EQ(context.interfaceRulesForMarker(positive_marker).size(), 1u);
    RecordProperty("pruned_boundary_parent_volume_case_count", 1);
}

TEST(FreeSurfaceGeometrySnapshot,
     AcceptsParallelTetrahedralCutsUnderActiveSideReversal)
{
    constexpr int interface_marker = 107;
    constexpr std::array<FE::geometry::CutIntegrationSide, 2>
        active_sides{{
            FE::geometry::CutIntegrationSide::Negative,
            FE::geometry::CutIntegrationSide::Positive,
        }};
    constexpr std::array<FE::Real, 3> interface_heights{{
        FE::Real{1.0e-6}, FE::Real{0.49}, FE::Real{0.5}}};
    const SingleTetraBoundaryMesh mesh;

    for (const auto active_side : active_sides) {
        for (const auto interface_height : interface_heights) {
            SCOPED_TRACE(static_cast<unsigned>(active_side));
            SCOPED_TRACE(interface_height);
            auto request = interfaceRequest(interface_marker);
            request.interface_quadrature_order = 1;
            request.aligned_zero_interface_parent_side = active_side;
            interfaces::LevelSetCellCutInput input;
            input.parent_cell = 0;
            input.element_type = FE::ElementType::Tetra4;
            input.node_coordinates.reserve(4u);
            input.level_set_values.reserve(4u);
            const FE::Real active_sign =
                active_side == FE::geometry::CutIntegrationSide::Negative
                    ? FE::Real{1.0}
                    : FE::Real{-1.0};
            for (std::size_t node = 0u; node < 4u; ++node) {
                const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
                    FE::ElementType::Tetra4, node);
                input.node_coordinates.push_back(
                    {{xi[0], xi[1], xi[2]}});
                input.level_set_values.push_back(
                    active_sign *
                    (FE::Real{0.5} * (xi[0] + xi[1] + xi[2]) -
                     interface_height));
            }
            auto cut =
                interfaces::cutLinearLevelSetCell3D(request, input);
            ASSERT_TRUE(cut.supported);
            ASSERT_EQ(cut.fragments.size(), 1u);
            if (interface_height < FE::Real{0.5}) {
                ASSERT_EQ(cut.volume_regions.size(), 2u);
            } else {
                ASSERT_EQ(cut.volume_regions.size(), 1u);
                EXPECT_TRUE(cut.volume_regions.front().full_cell_equivalent);
                EXPECT_EQ(cut.volume_regions.front().side, active_side);
            }

            interfaces::LevelSetInterfaceDomain interface_domain(request);
            for (auto& fragment : cut.fragments) {
                interface_domain.addFragment(std::move(fragment));
            }
            for (auto& region : cut.volume_regions) {
                interface_domain.addVolumeRegion(std::move(region));
            }
            interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
            scalar.value =
                [active_sign, interface_height](
                    FE::GlobalIndex,
                    const std::array<FE::Real, 3>& point,
                    const FE::geometry::CutQuadratureProvenance&) {
                    return active_sign *
                           (FE::Real{0.5} *
                                (point[0] + point[1] + point[2]) -
                            interface_height);
                };
            scalar.reference_gradient =
                [active_sign](
                    FE::GlobalIndex,
                    const std::array<FE::Real, 3>&,
                    const FE::geometry::CutQuadratureProvenance&) {
                    return std::array<FE::Real, 3>{{
                        FE::Real{0.5} * active_sign,
                        FE::Real{0.5} * active_sign,
                        FE::Real{0.5} * active_sign,
                    }};
                };
            EXPECT_NO_THROW(
                (void)interfaces::buildFreeSurfaceGeometrySnapshot(
                    std::move(interface_domain),
                    {},
                    {},
                    mesh,
                    snapshotPolicyWithoutBoundary(),
                    std::move(scalar),
                    "parallel_tetrahedral_cut"));
        }
    }

    RecordProperty("parallel_tetrahedral_active_side_case_count", 6);
}

TEST(GeneratedActiveBoundaryDomain,
     RejectsUnimplementedHighOrderImplicitBoundaryRestriction)
{
    constexpr int interface_marker = 105;
    constexpr int wall_marker = 11;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    auto source_request = interfaceRequest(interface_marker);
    source_request.implicit_geometry_mode = "HighOrderImplicit";
    source_request.implicit_quadrature_backend = "SayeHyperrectangle";
    interfaces::LevelSetInterfaceDomain interface_domain(
        std::move(source_request));
    const auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [](FE::GlobalIndex) { return FE::Real{1.0}; };
    EXPECT_THROW(
        interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Positive),
            interface_domain,
            contact_domain,
            mesh,
            field),
        std::invalid_argument);
}

TEST(GeneratedActiveBoundaryDomain,
     WetFractionSweepIntegratesBoundaryPolynomialsForBothPhases)
{
    constexpr int wall_marker = 12;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    const std::array<FE::Real, 8> fractions{{
        1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 0.1, 0.25, 0.49, 1.0}};
    int marker = 200;
    for (const FE::Real fraction : fractions) {
        const FE::Real root = -1.0 + 2.0 * fraction;
        auto interface_domain = verticalInterfaceAtX(marker, root);
        auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(marker, wall_marker),
                interface_domain,
                mesh);
        const std::array<FE::Real, 4> values{{
            -fraction,
            1.0 - fraction,
            1.0 - fraction,
            -fraction}};
        interfaces::GeneratedActiveBoundaryScalarField field;
        field.value_at_node = [&values](FE::GlobalIndex node) {
            return values.at(static_cast<std::size_t>(node));
        };
        const auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Negative),
            interface_domain,
            contact_domain,
            mesh,
            field);
        const auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Positive),
            interface_domain,
            contact_domain,
            mesh,
            field);
        const auto negative_rules = negative.boundaryQuadratureRules();
        ASSERT_EQ(negative_rules.size(), 1u);
        const auto& rule = negative_rules.front();
        FE::Real constant_moment = 0.0;
        FE::Real linear_moment = 0.0;
        FE::Real quadratic_moment = 0.0;
        for (const auto& point : rule.points) {
            constant_moment += point.weight;
            linear_moment += point.weight * point.parent_coordinate[0];
            quadratic_moment += point.weight * point.parent_coordinate[0] *
                                point.parent_coordinate[0];
        }
        const FE::Real a = -1.0;
        const FE::Real b = root;
        EXPECT_NEAR(constant_moment, b - a, 2.0e-13);
        EXPECT_NEAR(linear_moment,
                    (b * b - a * a) / 2.0,
                    2.0e-13);
        EXPECT_NEAR(quadratic_moment,
                    (b * b * b - a * a * a) / 3.0,
                    2.0e-13);
        EXPECT_NEAR(negative.summary().measure,
                    2.0 * fraction,
                    2.0e-13);
        EXPECT_NEAR(positive.summary().measure,
                    2.0 * (1.0 - fraction),
                    2.0e-13);
        const auto partition =
            interfaces::validateGeneratedActiveBoundaryPartition(
                negative,
                positive,
                interface_domain,
                contact_domain,
                mesh);
        EXPECT_NEAR(partition.max_partition_error, 0.0, 2.0e-13);
        ++marker;
    }
}

TEST(GeneratedActiveBoundaryDomain,
     CurvedQuadraticParentUsesPointwisePhysicalBoundaryMapping)
{
    constexpr int interface_marker = 260;
    constexpr int wall_marker = 19;
    std::vector<std::array<FE::Real, 3>> coordinates;
    coordinates.reserve(9u);
    for (std::size_t node = 0u; node < 9u; ++node) {
        const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
            FE::ElementType::Quad9, node);
        const FE::Real x = xi[0];
        const FE::Real y = xi[1] + FE::Real{0.1} *
                                      (FE::Real{1.0} - x * x) *
                                      (FE::Real{1.0} - xi[1]);
        coordinates.push_back({{x, y, 0.0}});
    }
    const SingleQuadBoundaryMesh mesh(
        wall_marker,
        /*rank=*/0,
        /*size=*/1,
        /*owner_rank=*/0,
        /*owned=*/true,
        FE::ElementType::Quad9,
        std::move(coordinates));
    auto interface_domain = verticalInterfaceWithVolumes(interface_marker);
    auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);

    std::array<FE::Real, 9> values{};
    for (std::size_t node = 0u; node < values.size(); ++node) {
        values[node] = FE::basis::ReferenceNodeLayout::get_node_coords(
                           FE::ElementType::Quad9, node)[0];
    }
    interfaces::GeneratedActiveBoundaryScalarField field;
    field.value_at_node = [&values](FE::GlobalIndex node) {
        return values.at(static_cast<std::size_t>(node));
    };
    auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Negative),
        interface_domain,
        contact_domain,
        mesh,
        field);
    auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Positive),
        interface_domain,
        contact_domain,
        mesh,
        field);
    EXPECT_NO_THROW(interfaces::validateGeneratedActiveBoundaryPartition(
        negative, positive, interface_domain, contact_domain, mesh));

    auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        {std::move(contact_domain)},
        {std::move(negative), std::move(positive)},
        mesh,
        {},
        verticalScalar(),
        "curved_quadratic_boundary");
    const auto wet_rules = snapshot->retainedRules(
        interfaces::FreeSurfaceGeometryRuleRole::NegativeExteriorBoundary);
    ASSERT_EQ(wet_rules.size(), 1u);
    const FE::Real expected_arc =
        FE::Real{0.5} *
        (std::sqrt(FE::Real{1.16}) +
         std::asinh(FE::Real{0.4}) / FE::Real{0.4});
    EXPECT_GT(wet_rules.front()->physical_rule.physical_measure,
              wet_rules.front()->reference_rule.measure);
    EXPECT_NEAR(wet_rules.front()->physical_rule.physical_measure,
                expected_arc,
                5.0e-5);
    EXPECT_NEAR(snapshot->ledger().maximum_boundary_partition_error,
                0.0,
                1.0e-12);
}

TEST(FreeSurfaceGeometrySnapshot,
     CurvedQ2AndQ3ConstantOneAssemblyMatchesRetainedPhysicalVolume)
{
    struct Scenario {
        int order;
        FE::ElementType element_type;
        int interface_marker;
        const char* property_name;
    };
    const std::array<Scenario, 2> scenarios{{
        {2,
         FE::ElementType::Quad9,
         273,
         "curved_q2_snapshot_constant_one_assembly_error"},
        {3,
         FE::ElementType::Quad4,
         274,
         "curved_q3_snapshot_constant_one_assembly_error"},
    }};

    for (const auto& scenario : scenarios) {
        SCOPED_TRACE(scenario.order);
        auto coordinates = curvedQuadCoordinates(scenario.order);
        ASSERT_EQ(coordinates.size(),
                  static_cast<std::size_t>(
                      (scenario.order + 1) * (scenario.order + 1)));
        const SingleQuadBoundaryMesh mesh(
            /*marker=*/7,
            /*rank=*/0,
            /*size=*/1,
            /*owner_rank=*/0,
            /*owned=*/true,
            scenario.element_type,
            std::move(coordinates),
            /*expose_opposite_face=*/false,
            scenario.order);
        const auto snapshot =
            interfaces::buildFreeSurfaceGeometrySnapshot(
                verticalHalfCutWithQuadrature(
                    scenario.interface_marker,
                    scenario.element_type,
                    scenario.order),
                {},
                {},
                mesh,
                snapshotPolicyWithoutBoundary(),
                verticalScalar(),
                scenario.order == 2 ? "curved_q2_constant_one"
                                    : "curved_q3_constant_one");
        ASSERT_NE(snapshot, nullptr);

        const auto retained = snapshot->retainedRules(
            interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume);
        ASSERT_EQ(retained.size(), 1u);
        ASSERT_NE(retained.front(), nullptr);
        const FE::Real rule_measure =
            retained.front()->physical_rule.physical_measure;
        const FE::Real ledger_measure =
            snapshot->ledger().owned_retained_negative_physical_volume;
        EXPECT_NEAR(ledger_measure, rule_measure, 2.0e-14);
        EXPECT_GT(std::abs(rule_measure - FE::Real{4.0}), FE::Real{0.1});

        FE::assembly::CutIntegrationContext context;
        context.addFreeSurfaceGeometrySnapshot(
            snapshot, FE::geometry::CutIntegrationSide::Negative);
        EXPECT_TRUE(context.hasFreeSurfaceGeometrySnapshotForMarker(
            scenario.interface_marker));

        FE::spaces::H1Space space(scenario.element_type, scenario.order);
        auto dof_map = singleCellDofMap(space.dofs_per_element());
        FE::assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);
        assembler.initialize();
        FE::assembly::DenseVectorView assembled(
            static_cast<FE::GlobalIndex>(space.dofs_per_element()));
        assembled.zero();
        ConstantOneCutVolumeKernel kernel;
        const auto result = assembler.assembleCutVolumes(
            mesh,
            context,
            scenario.interface_marker,
            FE::geometry::CutIntegrationSide::Negative,
            space,
            space,
            kernel,
            /*matrix_view=*/nullptr,
            &assembled,
            /*assemble_matrix=*/false,
            /*assemble_vector=*/true);
        ASSERT_TRUE(result.success) << result.error_message;
        ASSERT_EQ(result.elements_assembled, FE::GlobalIndex{1});
        const FE::Real assembled_measure = std::accumulate(
            assembled.data().begin(), assembled.data().end(), FE::Real{0.0});
        const FE::Real error = std::abs(assembled_measure - rule_measure);
        EXPECT_NEAR(assembled_measure, rule_measure, 2.0e-13);
        EXPECT_NEAR(assembled_measure, ledger_measure, 2.0e-13);
        RecordProperty(scenario.property_name,
                       ::testing::PrintToString(error));
    }
}

TEST(FreeSurfaceGeometrySnapshot,
     OwnsOneRevisionAndPointwisePhysicalMappingForEveryRuleFamily)
{
    constexpr int interface_marker = 104;
    constexpr int wall_marker = 10;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    auto interface_domain = verticalInterfaceWithVolumes(interface_marker);
    auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    const std::array<FE::Real, 4> values{{-1.0, 1.0, 1.0, -1.0}};
    interfaces::GeneratedActiveBoundaryScalarField nodal_field;
    nodal_field.value_at_node = [&values](FE::GlobalIndex node) {
        return values.at(static_cast<std::size_t>(node));
    };
    auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Negative),
        interface_domain,
        contact_domain,
        mesh,
        nodal_field);
    auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Positive),
        interface_domain,
        contact_domain,
        mesh,
        nodal_field);

    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [](FE::GlobalIndex,
                      const std::array<FE::Real, 3>& xi,
                      const FE::geometry::CutQuadratureProvenance&) {
        return xi[0];
    };
    scalar.reference_gradient =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return std::array<FE::Real, 3>{{1.0, 0.0, 0.0}};
        };
    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        {std::move(contact_domain)},
        {std::move(negative), std::move(positive)},
        mesh,
        {},
        std::move(scalar),
        "sharp_boundary_test");

    ASSERT_TRUE(snapshot);
    EXPECT_TRUE(snapshot->revision().complete());
    EXPECT_EQ(snapshot->revision().source_value_revision, 9u);
    EXPECT_EQ(snapshot->ledger().rule_count, 6u);
    EXPECT_EQ(snapshot->ledger().retained_rule_count, 6u);
    EXPECT_EQ(snapshot->ledger().certified_rule_count,
              snapshot->ledger().rule_count);
    EXPECT_EQ(snapshot->ledger().parent_cell_moment_certificate_count +
                  snapshot->ledger().centroid_moment_certificate_count +
                  snapshot->ledger()
                      .piecewise_affine_moment_certificate_count +
                  snapshot->ledger()
                      .backend_reference_moment_certificate_count +
                  snapshot->ledger()
                      .stored_generated_moment_certificate_count,
              snapshot->ledger().rule_count);
    EXPECT_EQ(snapshot->ledger().stored_generated_moment_certificate_count,
              0u);
    EXPECT_GT(snapshot->ledger().validated_rule_polynomial_moment_count,
              snapshot->ledger().rule_count);
    EXPECT_LE(snapshot->ledger().maximum_polynomial_moment_scaled_error,
              1.0);
    EXPECT_EQ(snapshot->ledger().global_owned_rule_count,
              snapshot->ledger().owned_rule_count);
    EXPECT_EQ(snapshot->ledger().orphan_contact_fragment_count, 0u);
    EXPECT_NEAR(snapshot->ledger().retained_negative_reference_volume,
                2.0,
                1.0e-14);
    EXPECT_NEAR(snapshot->ledger().retained_positive_reference_volume,
                2.0,
                1.0e-14);
    EXPECT_NEAR(snapshot->ledger().retained_negative_physical_volume,
                0.5,
                1.0e-14);
    EXPECT_NEAR(snapshot->ledger().retained_positive_physical_volume,
                0.5,
                1.0e-14);
    EXPECT_NEAR(
        snapshot->ledger().owned_retained_negative_physical_volume,
        snapshot->ledger().retained_negative_physical_volume,
        1.0e-14);
    EXPECT_NEAR(
        snapshot->ledger().owned_retained_positive_physical_volume,
        snapshot->ledger().retained_positive_physical_volume,
        1.0e-14);
    EXPECT_NEAR(snapshot->ledger().maximum_volume_partition_error,
                0.0,
                1.0e-14);
    EXPECT_NEAR(snapshot->ledger().maximum_boundary_partition_error,
                0.0,
                1.0e-14);

    FE::Real expected_negative_wall_area{0.0};
    FE::Real expected_contact_measure{0.0};
    std::map<std::uint64_t, std::int64_t> surface_component_by_source;
    for (const auto& record : snapshot->rules()) {
        if (record.role == interfaces::FreeSurfaceGeometryRuleRole::Interface) {
            ASSERT_EQ(record.source_fragment_stable_ids.size(), 1u);
            surface_component_by_source.emplace(
                record.source_fragment_stable_ids.front(),
                record.component_id);
        }
    }
    for (const auto& record : snapshot->rules()) {
        EXPECT_GE(record.component_id, 0);
        EXPECT_EQ(record.moment_certificate.polynomial_order,
                  record.reference_rule.exact_polynomial_order);
        EXPECT_EQ(record.moment_certificate.ambient_dimension,
                  mesh.dimension());
        EXPECT_FALSE(record.moment_certificate.moments.empty());
        EXPECT_EQ(record.reference_rule.provenance
                      .free_surface_snapshot_revision_key,
                  snapshot->revision().snapshot_revision_key);
        EXPECT_EQ(record.physical_rule.free_surface_snapshot_revision_key,
                  snapshot->revision().snapshot_revision_key);
        ASSERT_EQ(record.reference_rule.points.size(),
                  record.physical_rule.points.size());
        EXPECT_GT(record.physical_rule.physical_measure, 0.0);
        if (record.locally_owned &&
            record.retention ==
                interfaces::FreeSurfaceGeometryRetention::Retained) {
            if (record.role == interfaces::FreeSurfaceGeometryRuleRole::
                                   NegativeExteriorBoundary) {
                EXPECT_EQ(record.physical_boundary_marker, 10);
                expected_negative_wall_area +=
                    record.physical_rule.physical_measure;
            } else if (record.role ==
                       interfaces::FreeSurfaceGeometryRuleRole::Contact) {
                EXPECT_EQ(record.physical_boundary_marker, 10);
                ASSERT_EQ(record.source_fragment_stable_ids.size(), 1u);
                const auto source = surface_component_by_source.find(
                    record.source_fragment_stable_ids.front());
                ASSERT_NE(source, surface_component_by_source.end());
                EXPECT_EQ(record.component_id, source->second);
                expected_contact_measure +=
                    record.physical_rule.physical_measure;
            } else if (record.role == interfaces::FreeSurfaceGeometryRuleRole::
                                          PositiveExteriorBoundary) {
                EXPECT_EQ(record.physical_boundary_marker, 10);
            } else {
                EXPECT_EQ(record.physical_boundary_marker, -1);
            }
        }
        for (const auto& point : record.physical_rule.points) {
            EXPECT_GT(point.physical_weight, 0.0);
            EXPECT_GT(point.absolute_jacobian_determinant, 0.0);
        }
    }

    interfaces::FreeSurfaceDiscreteFunctionalParameters functional_parameters;
    functional_parameters.liquid_side =
        FE::geometry::CutIntegrationSide::Negative;
    functional_parameters.surface_tension = 2.0;
    functional_parameters.young_wall_coefficients.push_back(
        {10, std::acos(FE::Real{0.5})});
    functional_parameters.dynamic_contact_coefficients.push_back(
        {.boundary_marker = 10,
         .equilibrium_contact_angle_radians =
             std::acos(FE::Real{0.5}),
         .mobility = FE::Real{0.5},
         .slip_length = FE::Real{0.2},
         .dynamic_viscosity = FE::Real{0.4}});
    functional_parameters.volume_multiplier = 3.0;
    const auto functional =
        interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *snapshot, functional_parameters);
    EXPECT_EQ(functional.snapshot_revision_key,
              snapshot->revision().snapshot_revision_key);
    EXPECT_NEAR(functional.owned_liquid_volume,
                snapshot->ledger().owned_retained_negative_physical_volume,
                1.0e-14);
    EXPECT_NEAR(functional.owned_liquid_gas_area,
                snapshot->ledger().interface_physical_measure,
                1.0e-14);
    EXPECT_NEAR(functional.owned_wetted_wall_area,
                expected_negative_wall_area,
                1.0e-14);
    EXPECT_NEAR(functional.owned_contact_measure,
                expected_contact_measure,
                1.0e-14);
    EXPECT_NEAR(functional.liquid_gas_surface_energy,
                2.0 * functional.owned_liquid_gas_area,
                1.0e-14);
    EXPECT_NEAR(functional.young_wall_energy,
                -functional.owned_wetted_wall_area,
                1.0e-14);
    ASSERT_EQ(functional.walls.size(), 1u);
    EXPECT_EQ(functional.walls.front().boundary_marker, 10);
    ASSERT_TRUE(functional.walls.front()
                    .equilibrium_contact_angle_radians.has_value());
    EXPECT_NEAR(*functional.walls.front()
                     .equilibrium_contact_angle_radians,
                std::acos(FE::Real{0.5}),
                1.0e-14);
    EXPECT_NEAR(functional.walls.front().owned_wetted_wall_area,
                functional.owned_wetted_wall_area,
                1.0e-14);
    EXPECT_NEAR(functional.walls.front().owned_contact_measure,
                functional.owned_contact_measure,
                1.0e-14);
    EXPECT_NEAR(functional.walls.front().young_wall_energy,
                functional.young_wall_energy,
                1.0e-14);
    EXPECT_NEAR(functional.volume_constraint_potential,
                3.0 * functional.owned_liquid_volume,
                1.0e-14);
    EXPECT_NEAR(functional.total_potential,
                functional.liquid_gas_surface_energy +
                    functional.young_wall_energy +
                    functional.volume_constraint_potential,
                1.0e-14);

    interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator velocity;
    velocity.value = [](FE::GlobalIndex,
                        const std::array<FE::Real, 3>&,
                        const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{0.25, 0.0, 0.0}};
    };
    const auto dynamic_contact =
        interfaces::evaluateFreeSurfaceDynamicContactState(
            *snapshot, functional_parameters, velocity);
    EXPECT_EQ(dynamic_contact.snapshot_revision_key,
              snapshot->revision().snapshot_revision_key);
    ASSERT_EQ(dynamic_contact.walls.size(), 1u);
    const auto& contact = dynamic_contact.walls.front();
    EXPECT_EQ(contact.boundary_marker, wall_marker);
    EXPECT_EQ(contact.owned_quadrature_point_count, 1u);
    EXPECT_EQ(contact.owned_advancing_point_count, 1u);
    EXPECT_EQ(contact.owned_receding_point_count, 0u);
    EXPECT_EQ(contact.motion,
              interfaces::FreeSurfaceContactMotion::Advancing);
    EXPECT_NEAR(contact.owned_contact_measure,
                expected_contact_measure,
                1.0e-14);
    ASSERT_TRUE(contact.mean_dynamic_angle_radians.has_value());
    ASSERT_TRUE(contact.mean_dynamic_cosine.has_value());
    ASSERT_TRUE(contact.mean_contact_speed.has_value());
    ASSERT_TRUE(contact.mean_constitutive_residual.has_value());
    ASSERT_TRUE(contact.mean_absolute_constitutive_residual.has_value());
    EXPECT_NEAR(*contact.mean_dynamic_angle_radians,
                std::numbers::pi_v<FE::Real> / FE::Real{2.0},
                1.0e-14);
    EXPECT_NEAR(*contact.mean_dynamic_cosine, 0.0, 1.0e-14);
    EXPECT_NEAR(*contact.mean_contact_speed, 0.25, 1.0e-14);
    EXPECT_NEAR(contact.contact_speed_squared_integral,
                FE::Real{0.0625} * expected_contact_measure,
                1.0e-14);
    EXPECT_NEAR(*contact.mean_constitutive_residual, -0.5, 1.0e-14);
    EXPECT_NEAR(*contact.mean_absolute_constitutive_residual,
                0.5,
                1.0e-14);
    EXPECT_NEAR(contact.line_friction_dissipation,
                FE::Real{0.125} * expected_contact_measure,
                1.0e-14);
    EXPECT_EQ(contact.owned_wetted_wall_quadrature_point_count, 2u);
    EXPECT_NEAR(contact.owned_wetted_wall_measure,
                expected_negative_wall_area,
                1.0e-14);
    EXPECT_NEAR(contact.wall_slip_speed_squared_integral,
                FE::Real{0.0625} * expected_negative_wall_area,
                1.0e-14);
    EXPECT_NEAR(contact.wall_slip_dissipation,
                FE::Real{0.125} * expected_negative_wall_area,
                1.0e-14);
    ASSERT_TRUE(contact.mean_wall_slip_speed.has_value());
    EXPECT_NEAR(*contact.mean_wall_slip_speed, 0.25, 1.0e-14);
    EXPECT_NEAR(contact.mean_wall_tangential_velocity[0],
                0.25,
                1.0e-14);
    EXPECT_NEAR(contact.mean_wall_tangential_velocity[1],
                0.0,
                1.0e-14);
    EXPECT_NEAR(contact.mean_wall_normal[0], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_wall_normal[1], -1.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_position[0], 0.5, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_position[1], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_footprint_direction[0], 1.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_footprint_direction[1], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.contact_line_tangent_integral[2],
                expected_contact_measure,
                1.0e-14);
    EXPECT_NEAR(contact.mean_contact_line_tangent[0], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_line_tangent[1], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_line_tangent[2], 1.0, 1.0e-14);
    EXPECT_NEAR(dynamic_contact.owned_contact_measure,
                contact.owned_contact_measure,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.line_friction_dissipation,
                contact.line_friction_dissipation,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.owned_wetted_wall_measure,
                contact.owned_wetted_wall_measure,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.wall_slip_dissipation,
                contact.wall_slip_dissipation,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.total_dissipation,
                contact.line_friction_dissipation +
                    contact.wall_slip_dissipation,
                1.0e-14);

    auto prescribed_slip_parameters = functional_parameters;
    auto& prescribed_slip_coefficient =
        prescribed_slip_parameters.dynamic_contact_coefficients.front();
    prescribed_slip_coefficient.law =
        interfaces::FreeSurfaceContactLaw::PrescribedAngle;
    prescribed_slip_coefficient.mobility = 0.0;
    const auto prescribed_slip =
        interfaces::evaluateFreeSurfaceDynamicContactState(
            *snapshot, prescribed_slip_parameters, velocity);
    ASSERT_EQ(prescribed_slip.walls.size(), 1u);
    const auto& prescribed_wall = prescribed_slip.walls.front();
    EXPECT_EQ(prescribed_wall.law,
              interfaces::FreeSurfaceContactLaw::PrescribedAngle);
    EXPECT_DOUBLE_EQ(prescribed_wall.mobility, 0.0);
    EXPECT_NEAR(prescribed_wall.wall_slip_dissipation,
                contact.wall_slip_dissipation,
                1.0e-14);
    EXPECT_DOUBLE_EQ(prescribed_wall.line_friction_dissipation, 0.0);
    EXPECT_DOUBLE_EQ(prescribed_slip.line_friction_dissipation, 0.0);
    ASSERT_TRUE(
        prescribed_wall.mean_constitutive_residual.has_value());
    EXPECT_NEAR(*prescribed_wall.mean_constitutive_residual,
                -functional_parameters.surface_tension * 0.5,
                1.0e-14);
    EXPECT_NEAR(prescribed_slip.total_dissipation,
                prescribed_wall.wall_slip_dissipation,
                1.0e-14);

    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceDynamicContactState(
            *snapshot, functional_parameters, {}),
        std::invalid_argument);
    auto invalid_contact_parameters = functional_parameters;
    invalid_contact_parameters.dynamic_contact_coefficients.front()
        .slip_length = 0.0;
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceDynamicContactState(
            *snapshot, invalid_contact_parameters, velocity),
        std::invalid_argument);
    invalid_contact_parameters = functional_parameters;
    invalid_contact_parameters.dynamic_contact_coefficients.front()
        .dynamic_viscosity = 0.0;
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceDynamicContactState(
            *snapshot, invalid_contact_parameters, velocity),
        std::invalid_argument);
    invalid_contact_parameters = prescribed_slip_parameters;
    invalid_contact_parameters.dynamic_contact_coefficients.front()
        .mobility = 0.5;
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceDynamicContactState(
            *snapshot, invalid_contact_parameters, velocity),
        std::invalid_argument);

    auto positive_parameters = functional_parameters;
    positive_parameters.liquid_side =
        FE::geometry::CutIntegrationSide::Positive;
    const auto positive_contact =
        interfaces::evaluateFreeSurfaceDynamicContactState(
            *snapshot, positive_parameters, velocity);
    ASSERT_EQ(positive_contact.walls.size(), 1u);
    ASSERT_TRUE(
        positive_contact.walls.front().mean_contact_speed.has_value());
    EXPECT_NEAR(*positive_contact.walls.front().mean_contact_speed,
                -0.25,
                1.0e-14);
    EXPECT_EQ(positive_contact.walls.front().motion,
              interfaces::FreeSurfaceContactMotion::Receding);

    auto invalid_parameters = functional_parameters;
    invalid_parameters.surface_tension = -1.0;
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *snapshot, invalid_parameters),
        std::invalid_argument);
    invalid_parameters = functional_parameters;
    invalid_parameters.young_wall_coefficients.front()
        .equilibrium_contact_angle_radians = 0.0;
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *snapshot, invalid_parameters),
        std::invalid_argument);
    invalid_parameters = functional_parameters;
    invalid_parameters.young_wall_coefficients.push_back(
        invalid_parameters.young_wall_coefficients.front());
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *snapshot, invalid_parameters),
        std::invalid_argument);

    FE::assembly::CutIntegrationContext context;
    context.addFreeSurfaceGeometrySnapshot(snapshot);
    EXPECT_EQ(context.freeSurfaceGeometrySnapshots().size(), 1u);
    const auto* interface_publication =
        context
            .findGeneratedLevelSetInterfacePublicationProvenance(
                interface_marker);
    ASSERT_NE(interface_publication, nullptr);
    EXPECT_EQ(
        interface_publication
            ->publication_domain_id,
        snapshot->revision().domain_id);
    for (const auto& contact :
         snapshot->contactDomains()) {
        const auto* contact_publication =
            context
                .findGeneratedInterfaceBoundaryPublicationProvenance(
                    contact.marker());
        ASSERT_NE(contact_publication, nullptr);
        EXPECT_EQ(
            contact_publication
                ->request
                .generated_domain_id,
            contact.request().generated_domain_id);
    }
    EXPECT_EQ(context.freeSurfaceGeometrySnapshotRevisionForMarker(
                  interface_marker),
              snapshot->revision().snapshot_revision_key);
    EXPECT_NO_THROW(
        context.assertFreeSurfaceGeometrySnapshotCurrentForMarker(
            interface_marker));
    for (const auto& rule : context.volumeRules()) {
        EXPECT_EQ(rule.provenance.free_surface_snapshot_revision_key,
                  snapshot->revision().snapshot_revision_key);
    }
    for (const auto& rule : context.interfaceRules()) {
        EXPECT_EQ(rule.provenance.free_surface_snapshot_revision_key,
                  snapshot->revision().snapshot_revision_key);
    }
    for (const auto& metadata : context.metadata()) {
        EXPECT_EQ(metadata.free_surface_snapshot_revision_key,
                  snapshot->revision().snapshot_revision_key);
    }
    for (const auto& binding : context.bindings()) {
        EXPECT_EQ(binding.marker, interface_marker);
        EXPECT_EQ(binding.free_surface_snapshot_revision_key,
                  snapshot->revision().snapshot_revision_key);
    }
    for (const auto& active : snapshot->activeBoundaryDomains()) {
        EXPECT_EQ(context.freeSurfaceGeometrySnapshotRevisionForMarker(
                      active.marker()),
                  snapshot->revision().snapshot_revision_key);
        EXPECT_NO_THROW(
            context.assertFreeSurfaceGeometrySnapshotCurrentForMarker(
                active.marker()));
    }
    const auto revision_before_duplicate_domain =
        context.contentRevision();
    EXPECT_THROW(
        context.addGeneratedInterfaceDomain(
            snapshot->interfaceDomain()),
        std::invalid_argument);
    EXPECT_EQ(context.contentRevision(),
              revision_before_duplicate_domain);
    EXPECT_NO_THROW(
        context.assertFreeSurfaceGeometrySnapshotCurrentForMarker(
            interface_marker));
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsDuplicateClosedSnapshotImportWithoutMutation)
{
    constexpr int interface_marker = 125;
    const SingleQuadBoundaryMesh mesh;
    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        verticalInterfaceWithVolumes(interface_marker),
        {},
        {},
        mesh,
        {},
        verticalScalar(),
        "duplicate_closed_snapshot_import");
    ASSERT_NE(snapshot, nullptr);

    FE::assembly::CutIntegrationContext context;
    context.addFreeSurfaceGeometrySnapshot(snapshot);
    const auto volume_rule_count = context.volumeRules().size();
    const auto interface_rule_count = context.interfaceRules().size();
    const auto metadata_count = context.metadata().size();
    const auto binding_count = context.bindings().size();
    const auto snapshot_count = context.freeSurfaceGeometrySnapshots().size();
    const auto content_revision = context.contentRevision();

    EXPECT_THROW(
        context.addFreeSurfaceGeometrySnapshot(snapshot),
        std::invalid_argument);
    EXPECT_EQ(context.volumeRules().size(), volume_rule_count);
    EXPECT_EQ(context.interfaceRules().size(), interface_rule_count);
    EXPECT_EQ(context.metadata().size(), metadata_count);
    EXPECT_EQ(context.bindings().size(), binding_count);
    EXPECT_EQ(context.freeSurfaceGeometrySnapshots().size(), snapshot_count);
    EXPECT_EQ(context.contentRevision(), content_revision);
    EXPECT_NO_THROW(
        context.assertAllFreeSurfaceGeometrySnapshotsCurrent());
    RecordProperty("duplicate_snapshot_import_rejection_count", 1);
}

TEST(FreeSurfaceGeometrySnapshot,
     CompletedSnapshotPublicationRejectsDirectVolumeExtension)
{
    constexpr int interface_marker = 278;
    const SingleQuadBoundaryMesh mesh;
    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        verticalInterfaceWithVolumes(interface_marker),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        verticalScalar(),
        "completed_snapshot_direct_volume_extension");
    ASSERT_NE(snapshot, nullptr);

    FE::assembly::CutIntegrationContext context;
    ASSERT_NO_THROW(context.addFreeSurfaceGeometrySnapshot(snapshot));
    ASSERT_TRUE(
        context.hasGeneratedLevelSetInterfaceMarker(interface_marker));
    ASSERT_FALSE(context.volumeRules().empty());
    ASSERT_EQ(context.metadata().size(), context.volumeRules().size());
    EXPECT_NO_THROW(
        context.assertAllFreeSurfaceGeometrySnapshotsCurrent());

    const auto stored_rule = context.volumeRules().front();
    const auto stored_metadata = context.metadata().front();
    const auto content_revision = context.contentRevision();
    const auto volume_rule_count = context.volumeRules().size();
    const auto metadata_count = context.metadata().size();
    const auto binding_count = context.bindings().size();
    const auto volume_markers = context.generatedVolumeMarkers();
    const auto snapshot_revision =
        snapshot->revision().snapshot_revision_key;
    ASSERT_NE(snapshot_revision, 0u);

    const auto expect_rejected =
        [&](std::uint64_t candidate_snapshot_revision) {
            auto rule = stored_rule;
            auto metadata = stored_metadata;
            rule.provenance.free_surface_snapshot_revision_key =
                candidate_snapshot_revision;
            metadata.free_surface_snapshot_revision_key =
                candidate_snapshot_revision;
            EXPECT_THROW(
                context.addGeneratedVolumeRule(
                    interface_marker,
                    std::move(metadata),
                    std::move(rule)),
                std::invalid_argument);
            EXPECT_EQ(context.contentRevision(), content_revision);
            EXPECT_EQ(context.volumeRules().size(), volume_rule_count);
            EXPECT_EQ(context.metadata().size(), metadata_count);
            EXPECT_EQ(context.bindings().size(), binding_count);
            EXPECT_EQ(context.generatedVolumeMarkers(), volume_markers);
            EXPECT_NO_THROW(
                context.assertAllFreeSurfaceGeometrySnapshotsCurrent());
        };

    expect_rejected(0u);
    expect_rejected(snapshot_revision + 1u);
    expect_rejected(snapshot_revision);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsStaleTrackedMeshAtBuildAndConsumption)
{
    constexpr int interface_marker = 128;
    const auto expect_build_rejection = [&](std::uint64_t geometry_revision,
                                            std::uint64_t topology_revision,
                                            std::uint64_t ownership_revision) {
        SingleQuadBoundaryMesh mesh;
        mesh.enableRevisionTracking(geometry_revision,
                                    topology_revision,
                                    ownership_revision);
        EXPECT_THROW(
            (void)interfaces::buildFreeSurfaceGeometrySnapshot(
                verticalInterfaceWithVolumes(interface_marker),
                {},
                {},
                mesh,
                snapshotPolicyWithoutBoundary(),
                verticalScalar(),
                "stale_mesh_build"),
            std::invalid_argument);
    };
    expect_build_rejection(10u, 12u, 13u);
    expect_build_rejection(11u, 14u, 13u);
    expect_build_rejection(11u, 12u, 15u);

    SingleQuadBoundaryMesh mesh;
    mesh.enableRevisionTracking();
    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        verticalInterfaceWithVolumes(interface_marker),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        verticalScalar(),
        "tracked_mesh_consumption");
    FE::assembly::CutIntegrationContext context;
    context.addFreeSurfaceGeometrySnapshot(snapshot);
    EXPECT_TRUE(
        context.freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(mesh));
    EXPECT_NO_THROW(
        context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh));

    mesh.enableRevisionTracking(12u, 12u, 13u, 0u);
    EXPECT_FALSE(
        context.freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(mesh));
    try {
        context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
        ADD_FAILURE() << "stale geometry revision was accepted";
    } catch (const std::invalid_argument& error) {
        const std::string diagnostic = error.what();
        EXPECT_NE(diagnostic.find("snapshot_index=0"), std::string::npos);
        EXPECT_NE(diagnostic.find("snapshot_revision_key="),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("snapshot_geometry_revision=11"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("current_geometry_revision=12"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("snapshot_topology_revision=12"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("current_topology_revision=12"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("snapshot_ownership_revision=13"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("current_ownership_revision=13"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("snapshot_numbering_revision=0"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("current_numbering_revision=0"),
                  std::string::npos);
    }
    mesh.enableRevisionTracking(11u, 14u, 13u, 0u);
    EXPECT_FALSE(
        context.freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(mesh));
    EXPECT_THROW(context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh),
                 std::invalid_argument);
    mesh.enableRevisionTracking(11u, 12u, 15u, 0u);
    EXPECT_FALSE(
        context.freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(mesh));
    EXPECT_THROW(context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh),
                 std::invalid_argument);
    mesh.enableRevisionTracking(11u, 12u, 13u, 1u);
    EXPECT_FALSE(
        context.freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(mesh));
    EXPECT_THROW(context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh),
                 std::invalid_argument);

    mesh.enableRevisionTracking();
    EXPECT_TRUE(
        context.freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(mesh));
    EXPECT_NO_THROW(
        context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh));
    RecordProperty("snapshot_stale_mesh_build_rejection_count", 3);
    RecordProperty("snapshot_stale_mesh_consumption_rejection_count", 4);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsIncompleteAuthoritativeCutFamilies)
{
    constexpr int interface_marker = 131;
    const SingleQuadBoundaryMesh mesh;
    std::uint64_t rejection_count = 0u;
    const auto expect_rejection = [&](interfaces::LevelSetInterfaceDomain domain,
                                      const char* domain_id) {
        try {
            (void)interfaces::buildFreeSurfaceGeometrySnapshot(
                std::move(domain),
                {},
                {},
                mesh,
                snapshotPolicyWithoutBoundary(),
                verticalScalar(),
                domain_id);
            ADD_FAILURE() << "incomplete authoritative cut families were accepted";
        } catch (const std::invalid_argument&) {
            ++rejection_count;
        }
    };

    expect_rejection(verticalInterface(interface_marker),
                     "cut_parent_without_volume_regions");

    const auto complete = verticalInterfaceWithVolumes(interface_marker);
    interfaces::LevelSetInterfaceDomain missing_positive(complete.request());
    for (auto fragment : complete.fragments()) {
        missing_positive.addFragment(std::move(fragment));
    }
    for (auto region : complete.volumeRegions()) {
        if (region.side == FE::geometry::CutIntegrationSide::Negative) {
            missing_positive.addVolumeRegion(std::move(region));
        }
    }
    expect_rejection(std::move(missing_positive),
                     "cut_parent_without_positive_region");

    interfaces::LevelSetInterfaceDomain missing_interface(complete.request());
    for (auto region : complete.volumeRegions()) {
        missing_interface.addVolumeRegion(std::move(region));
    }
    expect_rejection(std::move(missing_interface),
                     "non_full_regions_without_interface");

    EXPECT_EQ(rejection_count, 3u);
    const auto incomplete_family_rejection_count = rejection_count;

    interfaces::LevelSetInterfaceDomain wrong_marker(complete.request());
    for (auto fragment : complete.fragments()) {
        wrong_marker.addFragment(std::move(fragment));
    }
    for (auto region : complete.volumeRegions()) {
        if (region.side == FE::geometry::CutIntegrationSide::Positive) {
            region.interface_marker = interface_marker + 1;
        }
        wrong_marker.addVolumeRegion(std::move(region));
    }
    expect_rejection(std::move(wrong_marker),
                     "cross_family_interface_marker_mismatch");

    EXPECT_EQ(rejection_count, 4u);
    RecordProperty("authoritative_cut_family_incomplete_rejection_count",
                   incomplete_family_rejection_count);
    RecordProperty("authoritative_cross_family_marker_rejection_count", 1);
    RecordProperty("authoritative_cut_family_total_rejection_count",
                   rejection_count);
}

TEST(FreeSurfaceGeometrySnapshot,
     AcceptsCompleteAndDegenerateAuthoritativeFamilies)
{
    constexpr int interface_marker = 132;
    const SingleQuadBoundaryMesh mesh;
    std::uint64_t acceptance_count = 0u;
    std::uint64_t two_sided_cut_acceptance_count = 0u;
    std::uint64_t one_sided_full_cell_acceptance_count = 0u;
    std::uint64_t one_sided_aligned_cut_acceptance_count = 0u;
    std::uint64_t empty_full_zero_acceptance_count = 0u;

    const auto complete_cut = interfaces::buildFreeSurfaceGeometrySnapshot(
        verticalInterfaceWithVolumes(interface_marker),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        verticalScalar(),
        "complete_authoritative_cut_families");
    ASSERT_TRUE(complete_cut);
    ++acceptance_count;
    ++two_sided_cut_acceptance_count;

    interfaces::FreeSurfaceGeometryScalarEvaluator value_only;
    value_only.value = [](FE::GlobalIndex,
                          const std::array<FE::Real, 3>&,
                          const FE::geometry::CutQuadratureProvenance&) {
        return FE::Real{-1.0};
    };
    const auto full_cell = interfaces::buildFreeSurfaceGeometrySnapshot(
        fullNegativeCell(interface_marker),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        std::move(value_only),
        "full_cell_without_interface_family");
    ASSERT_TRUE(full_cell);
    EXPECT_TRUE(full_cell->interfaceDomain().cutCells().empty());
    ++acceptance_count;
    ++one_sided_full_cell_acceptance_count;

    struct Scenario {
        std::array<FE::Real, 4> values;
        const char* domain_id;
        bool expect_active_interface;
        std::size_t expected_volume_region_count;
    };
    const std::array<Scenario, 5> scenarios{{
        {{{-1.0, 1.0, 1.0, -1.0}},
         "aligned_complete_cut_family",
         true,
         2u},
        {{{-1.0e-7, 1.0, 1.0, 1.0}},
         "nearly_tangent_complete_cut_family",
         true,
         2u},
        {{{0.0, 0.0, 1.0, 1.0}},
         "edge_touch_one_sided_full_cell_family",
         false,
         1u},
        {{{0.0, 1.0, 1.0, 1.0}},
         "vertex_touch_one_sided_full_cell_family",
         false,
         1u},
        {{{0.0, 0.0, 0.0, 0.0}},
         "full_zero_empty_authoritative_family",
         false,
         0u},
    }};
    for (const auto& scenario : scenarios) {
        SCOPED_TRACE(scenario.domain_id);
        auto domain = linearQuadCutDomain(interface_marker, scenario.values);
        EXPECT_EQ(!domain.cutCells().empty(),
                  scenario.expect_active_interface);
        ASSERT_EQ(domain.volumeRegions().size(),
                  scenario.expected_volume_region_count);
        if (!scenario.expect_active_interface &&
            !domain.volumeRegions().empty()) {
            EXPECT_TRUE(std::all_of(
                domain.volumeRegions().begin(),
                domain.volumeRegions().end(),
                [](const auto& region) {
                    return region.full_cell_equivalent;
                }));
        }
        const auto snapshot =
            interfaces::buildFreeSurfaceGeometrySnapshot(
                std::move(domain),
                {},
                {},
                mesh,
                snapshotPolicyWithoutBoundary(),
                bilinearQuadScalar(scenario.values),
                scenario.domain_id);
        ASSERT_TRUE(snapshot);
        ++acceptance_count;
        if (scenario.expect_active_interface) {
            ++two_sided_cut_acceptance_count;
        } else if (scenario.expected_volume_region_count == 0u) {
            ++empty_full_zero_acceptance_count;
        } else {
            ++one_sided_full_cell_acceptance_count;
        }
    }

    constexpr std::array<FE::geometry::CutIntegrationSide, 2>
        aligned_parent_sides{{
            FE::geometry::CutIntegrationSide::Negative,
            FE::geometry::CutIntegrationSide::Positive,
        }};
    for (const auto parent_side : aligned_parent_sides) {
        const FE::Real parent_value =
            parent_side == FE::geometry::CutIntegrationSide::Negative
                ? FE::Real{-1.0}
                : FE::Real{1.0};
        const std::array<FE::Real, 4> values{{
            FE::Real{0.0},
            FE::Real{0.0},
            parent_value,
            parent_value,
        }};
        auto domain = linearQuadCutDomain(
            interface_marker,
            values,
            FE::Real{1.0e-12},
            parent_side);
        ASSERT_EQ(domain.cutCells().size(), 1u);
        ASSERT_EQ(domain.fragments().size(), 1u);
        ASSERT_EQ(domain.volumeRegions().size(), 1u);
        EXPECT_EQ(domain.volumeRegions().front().side, parent_side);
        EXPECT_NEAR(domain.volumeRegions().front().volume_fraction,
                    FE::Real{1.0},
                    FE::Real{1.0e-14});
        const auto snapshot =
            interfaces::buildFreeSurfaceGeometrySnapshot(
                std::move(domain),
                {},
                {},
                mesh,
                snapshotPolicyWithoutBoundary(),
                bilinearQuadScalar(values),
                parent_side ==
                        FE::geometry::CutIntegrationSide::Negative
                    ? "aligned_negative_parent_family"
                    : "aligned_positive_parent_family");
        ASSERT_TRUE(snapshot);
        ++acceptance_count;
        ++one_sided_aligned_cut_acceptance_count;
    }

    EXPECT_EQ(acceptance_count, 9u);
    EXPECT_EQ(two_sided_cut_acceptance_count, 3u);
    EXPECT_EQ(one_sided_full_cell_acceptance_count, 3u);
    EXPECT_EQ(one_sided_aligned_cut_acceptance_count, 2u);
    EXPECT_EQ(empty_full_zero_acceptance_count, 1u);
    RecordProperty("authoritative_cut_family_valid_acceptance_count",
                   acceptance_count);
    RecordProperty("authoritative_two_sided_cut_acceptance_count",
                   two_sided_cut_acceptance_count);
    RecordProperty("authoritative_one_sided_full_cell_acceptance_count",
                   one_sided_full_cell_acceptance_count);
    RecordProperty("authoritative_one_sided_aligned_cut_acceptance_count",
                   one_sided_aligned_cut_acceptance_count);
    RecordProperty("authoritative_empty_full_zero_acceptance_count",
                   empty_full_zero_acceptance_count);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsUncertifiedVolumeWithoutMatchingScalarValue)
{
    constexpr int interface_marker = 133;
    const SingleQuadBoundaryMesh mesh;
    std::uint64_t missing_value_rejection_count = 0u;
    std::uint64_t wrong_side_rejection_count = 0u;
    try {
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            fullNegativeCell(interface_marker),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            {},
            "uncertified_volume_without_scalar_value");
        ADD_FAILURE() << "uncertified volume rule was accepted without a scalar value evaluator";
    } catch (const std::invalid_argument&) {
        ++missing_value_rejection_count;
    }
    interfaces::FreeSurfaceGeometryScalarEvaluator wrong_side;
    wrong_side.value = [](FE::GlobalIndex,
                          const std::array<FE::Real, 3>&,
                          const FE::geometry::CutQuadratureProvenance&) {
        return FE::Real{1.0};
    };
    try {
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            fullNegativeCell(interface_marker),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            std::move(wrong_side),
            "uncertified_negative_volume_with_positive_scalar");
        ADD_FAILURE() << "uncertified negative volume rule was accepted with a positive scalar value";
    } catch (const std::invalid_argument&) {
        ++wrong_side_rejection_count;
    }
    EXPECT_EQ(missing_value_rejection_count, 1u);
    EXPECT_EQ(wrong_side_rejection_count, 1u);
    RecordProperty(
        "uncertified_volume_missing_scalar_value_rejection_count",
        missing_value_rejection_count);
    RecordProperty(
        "uncertified_volume_wrong_side_scalar_rejection_count",
        wrong_side_rejection_count);
    RecordProperty("uncertified_volume_scalar_rejection_count",
                   missing_value_rejection_count +
                       wrong_side_rejection_count);
}

TEST(FreeSurfaceGeometrySnapshot,
     AcceptsVolumeOnlyValueEvaluatorWithoutGradient)
{
    constexpr int interface_marker = 134;
    const SingleQuadBoundaryMesh mesh;
    interfaces::FreeSurfaceGeometryScalarEvaluator value_only;
    value_only.value = [](FE::GlobalIndex,
                          const std::array<FE::Real, 3>&,
                          const FE::geometry::CutQuadratureProvenance&) {
        return FE::Real{-1.0};
    };
    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        fullNegativeCell(interface_marker),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        std::move(value_only),
        "volume_value_without_gradient");
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(snapshot->retainedRules(
                  interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume)
                  .size(),
              1u);
    RecordProperty("volume_only_value_without_gradient_acceptance_count", 1);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsIncompleteRevisionKeysAcrossEveryImportedRuleFamily)
{
    constexpr int interface_marker = 126;
    constexpr int wall_marker = 23;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    auto interface_domain = verticalInterfaceWithVolumes(interface_marker);
    auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker),
            interface_domain,
            mesh);
    const std::array<FE::Real, 4> values{{-1.0, 1.0, 1.0, -1.0}};
    interfaces::GeneratedActiveBoundaryScalarField nodal_field;
    nodal_field.value_at_node = [&values](FE::GlobalIndex node) {
        return values.at(static_cast<std::size_t>(node));
    };
    auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Negative),
        interface_domain,
        contact_domain,
        mesh,
        nodal_field);
    auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
        activeRequest(interface_marker,
                      wall_marker,
                      FE::geometry::CutIntegrationSide::Positive),
        interface_domain,
        contact_domain,
        mesh,
        nodal_field);
    const int contact_marker = contact_domain.marker();
    const std::array<int, 2> active_markers{{negative.marker(),
                                             positive.marker()}};
    auto stale_active_request = negative.request();
    ++stale_active_request.source.value_revision;
    ++stale_active_request.source_value_revision;
    interfaces::GeneratedActiveBoundaryDomain stale_active(
        stale_active_request);
    for (auto fragment : negative.fragments()) {
        stale_active.addFragment(std::move(fragment));
    }
    try {
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            interface_domain,
            {contact_domain},
            {std::move(stale_active), positive},
            mesh,
            {},
            verticalScalar(),
            "stale_active_boundary_revision");
        FAIL() << "stale active-boundary source revision was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("revision"),
                  std::string::npos);
    }
    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        {std::move(contact_domain)},
        {std::move(negative), std::move(positive)},
        mesh,
        {},
        verticalScalar(),
        "complete_revision_contract");
    ASSERT_NE(snapshot, nullptr);
    const auto revision_key = snapshot->revision().snapshot_revision_key;
    ASSERT_NE(revision_key, 0u);
    const auto mismatched_key =
        revision_key == std::numeric_limits<std::uint64_t>::max()
            ? revision_key - 1u
            : revision_key + 1u;
    ASSERT_NE(mismatched_key, 0u);
    ASSERT_NE(mismatched_key, revision_key);

    FE::assembly::CutIntegrationContext pristine;
    pristine.addFreeSurfaceGeometrySnapshot(snapshot);
    ASSERT_FALSE(pristine.volumeRules().empty());
    ASSERT_EQ(pristine.volumeRules().size(), pristine.metadata().size());
    ASSERT_EQ(pristine.volumeRules().size(), pristine.bindings().size());
    EXPECT_NO_THROW(
        pristine.assertAllFreeSurfaceGeometrySnapshotsCurrent());

    auto stale_source_snapshot = pristine;
    stale_source_snapshot.setExpectedGeneratedSourceValueRevision(
        interface_marker,
        snapshot->revision().source_value_revision + 1u);
    EXPECT_THROW(
        stale_source_snapshot.assertFreeSurfaceGeometrySnapshotCurrentForMarker(
            interface_marker),
        std::invalid_argument);
    EXPECT_THROW(
        stale_source_snapshot.assertFreeSurfaceGeometrySnapshotCurrentForMarker(
            contact_marker),
        std::invalid_argument);
    for (const auto marker : active_markers) {
        EXPECT_THROW(
            stale_source_snapshot
                .assertFreeSurfaceGeometrySnapshotCurrentForMarker(marker),
            std::invalid_argument);
    }
    EXPECT_THROW(
        stale_source_snapshot.assertAllFreeSurfaceGeometrySnapshotsCurrent(),
        std::invalid_argument);
    EXPECT_THROW(
        stale_source_snapshot
            .freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(mesh),
        std::invalid_argument);

    const auto interface_rule_index =
        [&pristine](int marker) {
            const auto& rules = pristine.interfaceRules();
            const auto found = std::find_if(
                rules.begin(), rules.end(), [marker](const auto& rule) {
                    return rule.provenance.marker == marker;
                });
            return static_cast<std::size_t>(
                std::distance(rules.begin(), found));
        };
    const auto base_interface_index = interface_rule_index(interface_marker);
    const auto contact_index = interface_rule_index(contact_marker);
    const auto negative_active_index =
        interface_rule_index(active_markers[0]);
    const auto positive_active_index =
        interface_rule_index(active_markers[1]);
    ASSERT_LT(base_interface_index, pristine.interfaceRules().size());
    ASSERT_LT(contact_index, pristine.interfaceRules().size());
    ASSERT_LT(negative_active_index, pristine.interfaceRules().size());
    ASSERT_LT(positive_active_index, pristine.interfaceRules().size());

    const auto expect_rejected = [](const auto& candidate, int marker) {
        EXPECT_THROW(
            candidate.assertFreeSurfaceGeometrySnapshotCurrentForMarker(
                marker),
            std::invalid_argument);
    };
    const auto mutate_volume_rule =
        [&](std::uint64_t key) {
            auto candidate = pristine;
            auto& rules = const_cast<
                std::vector<FE::geometry::CutQuadratureRule>&>(
                candidate.volumeRules());
            rules.front().provenance.free_surface_snapshot_revision_key = key;
            expect_rejected(candidate, interface_marker);
        };
    const auto mutate_metadata =
        [&](std::uint64_t key) {
            auto candidate = pristine;
            auto& metadata = const_cast<std::vector<
                FE::assembly::CutCellAssemblyMetadata>&>(candidate.metadata());
            metadata.front().free_surface_snapshot_revision_key = key;
            expect_rejected(candidate, interface_marker);
        };
    const auto mutate_binding =
        [&](std::uint64_t key) {
            auto candidate = pristine;
            auto& bindings = const_cast<std::vector<
                FE::assembly::CutIntegrationBinding>&>(candidate.bindings());
            bindings.front().free_surface_snapshot_revision_key = key;
            expect_rejected(candidate, interface_marker);
        };
    const auto mutate_interface_rule =
        [&](std::size_t index, int marker, std::uint64_t key) {
            auto candidate = pristine;
            auto& rules = const_cast<
                std::vector<FE::geometry::CutQuadratureRule>&>(
                candidate.interfaceRules());
            rules[index].provenance.free_surface_snapshot_revision_key = key;
            expect_rejected(candidate, marker);
        };

    mutate_volume_rule(0u);
    mutate_volume_rule(mismatched_key);
    mutate_metadata(0u);
    mutate_metadata(mismatched_key);
    mutate_binding(0u);
    mutate_binding(mismatched_key);
    mutate_interface_rule(base_interface_index, interface_marker, 0u);
    mutate_interface_rule(
        base_interface_index, interface_marker, mismatched_key);
    mutate_interface_rule(contact_index, contact_marker, 0u);
    mutate_interface_rule(contact_index, contact_marker, mismatched_key);
    mutate_interface_rule(negative_active_index, active_markers[0], 0u);
    mutate_interface_rule(
        negative_active_index, active_markers[0], mismatched_key);
    mutate_interface_rule(positive_active_index, active_markers[1], 0u);
    mutate_interface_rule(
        positive_active_index, active_markers[1], mismatched_key);

    FE::assembly::CutIntegrationContext orphan_volume;
    orphan_volume.addGeneratedVolumeRule(
        interface_marker,
        pristine.metadata().front(),
        pristine.volumeRules().front());
    EXPECT_THROW(
        (void)orphan_volume.generatedVolumeRulesForMarker(interface_marker),
        std::invalid_argument);

    auto unkeyed_metadata = pristine.metadata().front();
    unkeyed_metadata.free_surface_snapshot_revision_key = 0u;
    FE::assembly::CutIntegrationContext orphan_volume_rule;
    orphan_volume_rule.addVolumeRule(
        unkeyed_metadata, pristine.volumeRules().front());
    EXPECT_THROW(
        orphan_volume_rule.assertAllFreeSurfaceGeometrySnapshotsCurrent(),
        std::invalid_argument);

    auto unkeyed_volume_rule = pristine.volumeRules().front();
    unkeyed_volume_rule.provenance.free_surface_snapshot_revision_key = 0u;
    FE::assembly::CutIntegrationContext orphan_metadata;
    orphan_metadata.addVolumeRule(
        pristine.metadata().front(), std::move(unkeyed_volume_rule));
    EXPECT_THROW(
        orphan_metadata.assertAllFreeSurfaceGeometrySnapshotsCurrent(),
        std::invalid_argument);

    FE::assembly::CutIntegrationContext orphan_binding;
    orphan_binding.addBinding(pristine.bindings().front());
    EXPECT_THROW(
        orphan_binding.assertAllFreeSurfaceGeometrySnapshotsCurrent(),
        std::invalid_argument);

    FE::assembly::CutIntegrationContext orphan_interface;
    orphan_interface.addInterfaceRule(
        pristine.interfaceRules()[base_interface_index]);
    EXPECT_THROW(
        orphan_interface.assertGeneratedInterfaceRulesCurrentForMarker(
            interface_marker),
        std::invalid_argument);
    EXPECT_THROW(
        orphan_interface.assertAllFreeSurfaceGeometrySnapshotsCurrent(),
        std::invalid_argument);
    class BoundaryKernel final : public FE::assembly::AssemblyKernel {
    public:
        [[nodiscard]] FE::assembly::RequiredData getRequiredData()
            const override {
            return FE::assembly::RequiredData::None;
        }
        void computeCell(const FE::assembly::AssemblyContext&,
                         FE::assembly::KernelOutput&) override {}
        [[nodiscard]] bool hasCell() const noexcept override { return false; }
        void computeBoundaryFace(const FE::assembly::AssemblyContext&,
                                 int,
                                 FE::assembly::KernelOutput&) override {}
        [[nodiscard]] bool hasBoundaryFace() const noexcept override {
            return true;
        }
    } boundary_kernel;
    FE::assembly::CutDomainAssemblyOptions assembly_options;
    assembly_options.include_volume_rules = false;
    assembly_options.interface_marker = interface_marker;
    EXPECT_THROW(
        (void)FE::assembly::assembleCutDomains(
            orphan_interface,
            boundary_kernel,
            [](const FE::assembly::CutRuleAssemblyRequest&,
               FE::assembly::AssemblyContext&) {},
            assembly_options),
        std::invalid_argument);

    RecordProperty("snapshot_revision_negative_case_count", 27);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsStoredOffInterfaceResidual)
{
    constexpr int interface_marker = 105;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterface(interface_marker, 1.0e-4);
    addVerticalHalfVolumes(domain);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "bad_root"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsOffChordPointWithZeroStoredResidual)
{
    constexpr int interface_marker = 130;
    const SingleQuadBoundaryMesh mesh;
    const auto source = verticalInterfaceWithVolumes(interface_marker);
    interfaces::LevelSetInterfaceDomain defect(source.request());
    for (auto fragment : source.fragments()) {
        ASSERT_FALSE(fragment.quadrature_points.empty());
        fragment.quadrature_points.front().point[0] += 1.0e-4;
        fragment.quadrature_points.front().parent_coordinate[0] += 1.0e-4;
        fragment.quadrature_points.front().level_set_residual = 0.0;
        defect.addFragment(std::move(fragment));
    }
    for (auto region : source.volumeRegions()) {
        defect.addVolumeRegion(std::move(region));
    }
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(defect),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "off_chord_zero_stored_residual"),
        std::invalid_argument);
    RecordProperty("off_chord_zero_residual_rejection_count", 1);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsWrongNormalWhenScalarEvaluatorIsMissing)
{
    constexpr int interface_marker = 127;
    const SingleQuadBoundaryMesh mesh;
    const auto source = verticalInterfaceWithVolumes(interface_marker);
    interfaces::LevelSetInterfaceDomain wrong_normal(source.request());
    for (auto fragment : source.fragments()) {
        for (auto& component : fragment.normal) {
            component = -component;
        }
        for (auto& point : fragment.quadrature_points) {
            for (auto& component : point.normal) {
                component = -component;
            }
        }
        wrong_normal.addFragment(std::move(fragment));
    }
    for (auto region : source.volumeRegions()) {
        wrong_normal.addVolumeRegion(std::move(region));
    }

    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(wrong_normal),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            {},
            "missing_normal_evaluator"),
        std::invalid_argument);
    RecordProperty("missing_normal_evaluator_rejection_count", 1);
}

TEST(FreeSurfaceGeometrySnapshot,
     SeparatesUnprunedLifecycleVolumeFromRetainedAssemblyVolume)
{
    constexpr int interface_marker = 119;
    constexpr FE::Real negative_fraction = 1.0e-8;
    const SingleQuadBoundaryMesh mesh;
    auto policy = snapshotPolicyWithoutBoundary();
    policy.minimum_retained_volume_fraction = 1.0e-6;

    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        verticalInterfaceWithTinyNegativeVolume(
            interface_marker, negative_fraction),
        {},
        {},
        mesh,
        policy,
        verticalScalar(),
        "unpruned_volume_ledger");
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(snapshot->ledger().pruned_rule_count, 1u);
    EXPECT_NEAR(snapshot->ledger().unpruned_negative_reference_volume,
                4.0 * negative_fraction,
                1.0e-15);
    EXPECT_NEAR(snapshot->ledger().unpruned_positive_reference_volume,
                4.0 * (1.0 - negative_fraction),
                1.0e-14);
    EXPECT_NEAR(snapshot->ledger().unpruned_negative_physical_volume,
                negative_fraction,
                1.0e-15);
    EXPECT_NEAR(snapshot->ledger().unpruned_positive_physical_volume,
                1.0 - negative_fraction,
                1.0e-14);
    EXPECT_EQ(snapshot->ledger().retained_negative_reference_volume, 0.0);
    EXPECT_EQ(snapshot->ledger().retained_negative_physical_volume, 0.0);
    EXPECT_NEAR(
        snapshot->ledger().owned_unpruned_negative_physical_volume,
        snapshot->ledger().unpruned_negative_physical_volume,
        1.0e-15);
    EXPECT_NEAR(
        snapshot->ledger().owned_unpruned_positive_physical_volume,
        snapshot->ledger().unpruned_positive_physical_volume,
        1.0e-14);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsStaleGlobalIdentityAndOwner)
{
    constexpr int interface_marker = 115;
    const SingleQuadBoundaryMesh mesh;
    const auto source = verticalInterfaceWithVolumes(interface_marker);

    auto stale_id_fragment = source.fragments().front();
    stale_id_fragment.parent_cell_global_id = 99;
    stale_id_fragment.stable_id = 0u;
    interfaces::LevelSetInterfaceDomain stale_id_domain(
        interfaceRequest(interface_marker));
    stale_id_domain.addFragment(std::move(stale_id_fragment));
    for (auto region : source.volumeRegions()) {
        stale_id_domain.addVolumeRegion(std::move(region));
    }
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(stale_id_domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            {},
            "stale_global_id"),
        std::invalid_argument);

    auto stale_owner_fragment = source.fragments().front();
    stale_owner_fragment.owner_rank = 1;
    stale_owner_fragment.stable_id = 0u;
    interfaces::LevelSetInterfaceDomain stale_owner_domain(
        interfaceRequest(interface_marker));
    stale_owner_domain.addFragment(std::move(stale_owner_fragment));
    for (auto region : source.volumeRegions()) {
        stale_owner_domain.addVolumeRegion(std::move(region));
    }
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(stale_owner_domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            {},
            "stale_owner"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot,
     CollectiveValidatorRejectsDuplicateAndMissingRuleOwners)
{
    constexpr int interface_marker = 116;
    const SingleQuadBoundaryMesh distributed_mesh(
        /*marker=*/7,
        /*rank=*/0,
        /*size=*/2,
        /*owner_rank=*/0,
        /*owned=*/true);
    const auto build = [&](interfaces::FreeSurfaceGeometryOwnershipCollective
                               collective,
                           std::string domain_id) {
        return interfaces::buildFreeSurfaceGeometrySnapshot(
            distributedVerticalInterfaceWithVolumes(interface_marker),
            {},
            {},
            distributed_mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            std::move(domain_id),
            std::move(collective));
    };

    interfaces::FreeSurfaceGeometryOwnershipCollective valid;
    valid.rank = 0;
    valid.size = 2;
    valid.all_gather_owned_rule_identity_values =
        [](std::span<const std::uint64_t> local) {
            return std::vector<std::uint64_t>(local.begin(), local.end());
        };
    valid.all_gather_revision_values =
        [](std::span<const std::uint64_t> local) {
            std::vector<std::uint64_t> gathered(local.begin(), local.end());
            gathered.insert(gathered.end(), local.begin(), local.end());
            return gathered;
        };
    const auto snapshot = build(valid, "valid_distributed_ownership");
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(snapshot->ledger().global_owned_rule_count,
              snapshot->ledger().owned_rule_count);

    auto duplicate = valid;
    duplicate.all_gather_owned_rule_identity_values =
        [](std::span<const std::uint64_t> local) {
            std::vector<std::uint64_t> gathered(local.begin(), local.end());
            gathered.insert(gathered.end(), local.begin(), local.end());
            return gathered;
        };
    EXPECT_THROW(
        (void)build(std::move(duplicate), "duplicate_distributed_owner"),
        std::invalid_argument);

    auto missing = valid;
    missing.all_gather_owned_rule_identity_values =
        [](std::span<const std::uint64_t>) {
            return std::vector<std::uint64_t>{};
        };
    EXPECT_THROW((void)build(std::move(missing), "missing_distributed_owner"),
                 std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot,
     DiscreteFunctionalFirstVariationMatchesCentralDifference)
{
    constexpr int interface_marker = 118;
    constexpr int wall_marker = 19;
    constexpr FE::Real interface_intercept = 0.40;
    constexpr FE::Real interface_slope = 0.20;
    constexpr FE::Real translation = 0.12;
    constexpr FE::Real deformation_x_gradient = 0.30;
    constexpr FE::Real deformation_y_gradient = 0.10;
    constexpr FE::Real epsilon = 2.0e-6;
    const SingleQuadBoundaryMesh mesh(wall_marker);

    const auto nodal_values = [](FE::Real perturbation) {
        constexpr std::array<std::array<FE::Real, 2>, 4> points{{
            {{0.0, 0.0}},
            {{1.0, 0.0}},
            {{1.0, 1.0}},
            {{0.0, 1.0}},
        }};
        std::array<FE::Real, 4> values{};
        for (std::size_t node = 0u; node < points.size(); ++node) {
            const FE::Real x = points[node][0];
            const FE::Real y = points[node][1];
            const FE::Real moved_interface_x =
                (FE::Real{1.0} +
                 perturbation * deformation_x_gradient) *
                    (interface_intercept + interface_slope * y) +
                perturbation *
                    (deformation_y_gradient * y + translation);
            values[node] = x - moved_interface_x;
        }
        return values;
    };
    const auto build_snapshot = [&](FE::Real perturbation,
                                    std::string domain_id) {
        const auto values = nodal_values(perturbation);
        auto interface_domain =
            linearQuadCutDomain(interface_marker, values);
        auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(interface_marker, wall_marker),
                interface_domain,
                mesh);
        interfaces::GeneratedActiveBoundaryScalarField nodal_field;
        nodal_field.value_at_node = [values](FE::GlobalIndex node) {
            return values.at(static_cast<std::size_t>(node));
        };
        auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Negative),
            interface_domain,
            contact_domain,
            mesh,
            nodal_field);
        auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Positive),
            interface_domain,
            contact_domain,
            mesh,
            nodal_field);
        return interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(interface_domain),
            {std::move(contact_domain)},
            {std::move(negative), std::move(positive)},
            mesh,
            {},
            bilinearQuadScalar(values),
            std::move(domain_id));
    };

    const auto base_snapshot = build_snapshot(0.0, "functional_fd_base");
    const auto plus_snapshot =
        build_snapshot(epsilon, "functional_fd_plus");
    const auto minus_snapshot =
        build_snapshot(-epsilon, "functional_fd_minus");

    interfaces::FreeSurfaceDiscreteFunctionalDeformationEvaluator
        deformation;
    deformation.value =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>& xi,
           const FE::geometry::CutQuadratureProvenance&) {
            const FE::Real x = FE::Real{0.5} * (xi[0] + FE::Real{1.0});
            const FE::Real y = FE::Real{0.5} * (xi[1] + FE::Real{1.0});
            return std::array<FE::Real, 3>{{
                translation + deformation_x_gradient * x +
                    deformation_y_gradient * y,
                0.0,
                0.0,
            }};
        };
    deformation.physical_gradient =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return interfaces::FreeSurfaceDiscreteFunctionalPhysicalGradient{{
                {{deformation_x_gradient, deformation_y_gradient, 0.0}},
                {{0.0, 0.0, 0.0}},
                {{0.0, 0.0, 0.0}},
            }};
        };

    const auto central_difference = [](FE::Real plus, FE::Real minus) {
        return (plus - minus) / (FE::Real{2.0} * epsilon);
    };
    const auto relative_error = [](FE::Real actual, FE::Real expected) {
        return std::abs(actual - expected) /
               std::max({FE::Real{1.0e-12},
                         std::abs(actual),
                         std::abs(expected)});
    };
    FE::Real maximum_surface_area_error{0.0};
    FE::Real maximum_surface_energy_error{0.0};
    FE::Real maximum_wall_area_error{0.0};
    FE::Real maximum_young_wall_error{0.0};
    FE::Real maximum_volume_error{0.0};
    FE::Real maximum_volume_potential_error{0.0};
    FE::Real maximum_total_error{0.0};
    std::size_t case_count{0u};

    for (const auto liquid_side : {
             FE::geometry::CutIntegrationSide::Negative,
             FE::geometry::CutIntegrationSide::Positive}) {
        SCOPED_TRACE(static_cast<int>(liquid_side));
        interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
        parameters.liquid_side = liquid_side;
        parameters.surface_tension = FE::Real{1.7};
        parameters.young_wall_coefficients.push_back(
            {wall_marker, std::numbers::pi_v<FE::Real> / FE::Real{3.0}});
        parameters.volume_multiplier = FE::Real{-0.8};

        const auto base =
            interfaces::evaluateFreeSurfaceDiscreteFunctional(
                *base_snapshot, parameters);
        const auto plus =
            interfaces::evaluateFreeSurfaceDiscreteFunctional(
                *plus_snapshot, parameters);
        const auto minus =
            interfaces::evaluateFreeSurfaceDiscreteFunctional(
                *minus_snapshot, parameters);
        const auto variation =
            interfaces::evaluateFreeSurfaceDiscreteFunctionalFirstVariation(
                *base_snapshot, parameters, deformation);

        EXPECT_EQ(variation.snapshot_revision_key,
                  base.snapshot_revision_key);
        EXPECT_EQ(variation.liquid_side, liquid_side);
        EXPECT_DOUBLE_EQ(variation.surface_tension,
                         parameters.surface_tension);
        EXPECT_DOUBLE_EQ(variation.volume_multiplier,
                         parameters.volume_multiplier);
        ASSERT_EQ(base.walls.size(), 1u);
        ASSERT_EQ(plus.walls.size(), 1u);
        ASSERT_EQ(minus.walls.size(), 1u);
        ASSERT_EQ(variation.walls.size(), 1u);
        EXPECT_EQ(variation.walls.front().boundary_marker, wall_marker);
        EXPECT_EQ(variation.walls.front()
                      .equilibrium_contact_angle_radians,
                  base.walls.front().equilibrium_contact_angle_radians);

        const FE::Real surface_area_fd = central_difference(
            plus.owned_liquid_gas_area,
            minus.owned_liquid_gas_area);
        const FE::Real surface_energy_fd = central_difference(
            plus.liquid_gas_surface_energy,
            minus.liquid_gas_surface_energy);
        const FE::Real wall_area_fd = central_difference(
            plus.owned_wetted_wall_area,
            minus.owned_wetted_wall_area);
        const FE::Real young_wall_fd = central_difference(
            plus.young_wall_energy,
            minus.young_wall_energy);
        const FE::Real volume_fd = central_difference(
            plus.owned_liquid_volume,
            minus.owned_liquid_volume);
        const FE::Real volume_potential_fd = central_difference(
            plus.volume_constraint_potential,
            minus.volume_constraint_potential);
        const FE::Real total_fd = central_difference(
            plus.total_potential,
            minus.total_potential);

        EXPECT_NEAR(
            variation.walls.front().owned_wetted_wall_area_variation,
            wall_area_fd,
            1.0e-10);
        EXPECT_NEAR(variation.walls.front().young_wall_energy_variation,
                    young_wall_fd,
                    1.0e-10);
        EXPECT_DOUBLE_EQ(
            variation.walls.front().owned_wetted_wall_area_variation,
            variation.owned_wetted_wall_area_variation);
        EXPECT_DOUBLE_EQ(
            variation.walls.front().young_wall_energy_variation,
            variation.young_wall_energy_variation);

        const FE::Real surface_area_error = relative_error(
            variation.owned_liquid_gas_area_variation,
            surface_area_fd);
        const FE::Real surface_energy_error = relative_error(
            variation.liquid_gas_surface_energy_variation,
            surface_energy_fd);
        const FE::Real wall_area_error = relative_error(
            variation.owned_wetted_wall_area_variation,
            wall_area_fd);
        const FE::Real young_wall_error = relative_error(
            variation.young_wall_energy_variation,
            young_wall_fd);
        const FE::Real volume_error = relative_error(
            variation.owned_liquid_volume_variation,
            volume_fd);
        const FE::Real volume_potential_error = relative_error(
            variation.volume_constraint_potential_variation,
            volume_potential_fd);
        const FE::Real total_error = relative_error(
            variation.total_potential_variation,
            total_fd);

        EXPECT_LT(surface_area_error, 2.0e-7);
        EXPECT_LT(surface_energy_error, 2.0e-7);
        EXPECT_LT(wall_area_error, 2.0e-7);
        EXPECT_LT(young_wall_error, 2.0e-7);
        EXPECT_LT(volume_error, 2.0e-7);
        EXPECT_LT(volume_potential_error, 2.0e-7);
        EXPECT_LT(total_error, 2.0e-7);
        EXPECT_NE(variation.owned_liquid_volume_variation, 0.0);
        EXPECT_NE(variation.owned_liquid_gas_area_variation, 0.0);
        EXPECT_NE(variation.owned_wetted_wall_area_variation, 0.0);
        EXPECT_NE(variation.total_potential_variation, 0.0);

        EXPECT_NEAR(
            variation.liquid_gas_surface_energy_variation,
            parameters.surface_tension *
                variation.owned_liquid_gas_area_variation,
            1.0e-14);
        EXPECT_NEAR(
            variation.volume_constraint_potential_variation,
            parameters.volume_multiplier *
                variation.owned_liquid_volume_variation,
            1.0e-14);
        EXPECT_NEAR(
            variation.total_potential_variation,
            variation.liquid_gas_surface_energy_variation +
                variation.young_wall_energy_variation +
                variation.volume_constraint_potential_variation,
            1.0e-14);

        maximum_surface_area_error =
            std::max(maximum_surface_area_error, surface_area_error);
        maximum_surface_energy_error =
            std::max(maximum_surface_energy_error, surface_energy_error);
        maximum_wall_area_error =
            std::max(maximum_wall_area_error, wall_area_error);
        maximum_young_wall_error =
            std::max(maximum_young_wall_error, young_wall_error);
        maximum_volume_error =
            std::max(maximum_volume_error, volume_error);
        maximum_volume_potential_error =
            std::max(maximum_volume_potential_error,
                     volume_potential_error);
        maximum_total_error =
            std::max(maximum_total_error, total_error);
        ++case_count;
    }

    RecordProperty("functional_first_variation_fd_case_count", case_count);
    RecordProperty("functional_surface_area_fd_max_relative_error",
                   ::testing::PrintToString(maximum_surface_area_error));
    RecordProperty("functional_surface_energy_fd_max_relative_error",
                   ::testing::PrintToString(maximum_surface_energy_error));
    RecordProperty("functional_wetted_wall_area_fd_max_relative_error",
                   ::testing::PrintToString(maximum_wall_area_error));
    RecordProperty("functional_young_wall_fd_max_relative_error",
                   ::testing::PrintToString(maximum_young_wall_error));
    RecordProperty("functional_volume_fd_max_relative_error",
                   ::testing::PrintToString(maximum_volume_error));
    RecordProperty("functional_volume_potential_fd_max_relative_error",
                   ::testing::PrintToString(maximum_volume_potential_error));
    RecordProperty("functional_total_fd_max_relative_error",
                   ::testing::PrintToString(maximum_total_error));
}

TEST(FreeSurfaceGeometrySnapshot,
     GeneralFiniteElementPressureMatchesMaterialDomainVariation)
{
    constexpr int interface_marker = 124;
    constexpr int wall_marker = 21;
    constexpr FE::Real epsilon = FE::Real{1.0e-6};
    constexpr std::array<std::array<FE::Real, 3>, 4> base_coordinates{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
    }};

    // Both p_h and V_h are nonconstant Q1 fields.  Pressure coefficients are
    // held fixed in parent coordinates while x_t = x + t V_h(x), which is the
    // material-transport convention for
    //
    //   d/dt integral_{Omega_t} p_h,t dx
    //       = integral_{Omega_h} p_h div(V_h) dx.
    //
    // This is deliberately separate from the constant lambda V_h term in the
    // capillary functional.
    const auto deformation_value =
        [](const std::array<FE::Real, 3>& point) {
            const FE::Real x = point[0];
            const FE::Real y = point[1];
            return std::array<FE::Real, 3>{{
                FE::Real{0.07} + FE::Real{0.20} * x -
                    FE::Real{0.11} * y + FE::Real{0.13} * x * y,
                FE::Real{-0.04} + FE::Real{0.08} * x +
                    FE::Real{0.17} * y - FE::Real{0.09} * x * y,
                FE::Real{0.0},
            }};
        };
    const auto deformation_divergence =
        [](const std::array<FE::Real, 3>& point) {
            return FE::Real{0.37} + FE::Real{0.13} * point[1] -
                   FE::Real{0.09} * point[0];
        };
    const auto reference_to_base_point =
        [](const std::array<FE::Real, 3>& xi) {
            return std::array<FE::Real, 3>{{
                FE::Real{0.5} * (xi[0] + FE::Real{1.0}),
                FE::Real{0.5} * (xi[1] + FE::Real{1.0}),
                FE::Real{0.0},
            }};
        };
    const auto material_pressure =
        [&reference_to_base_point](
            const std::array<FE::Real, 3>& parent_coordinate) {
            const auto point =
                reference_to_base_point(parent_coordinate);
            const FE::Real x = point[0];
            const FE::Real y = point[1];
            return FE::Real{0.60} + FE::Real{0.70} * x -
                   FE::Real{0.40} * y + FE::Real{0.50} * x * y;
        };

    const std::array<FE::Real, 4> level_set_values{{
        FE::Real{-0.42},
        FE::Real{0.58},
        FE::Real{0.40},
        FE::Real{-0.60},
    }};
    const auto build_snapshot =
        [&](FE::Real perturbation, std::string domain_id) {
            std::vector<std::array<FE::Real, 3>> coordinates;
            coordinates.reserve(base_coordinates.size());
            for (const auto& point : base_coordinates) {
                const auto displacement = deformation_value(point);
                coordinates.push_back({{
                    point[0] + perturbation * displacement[0],
                    point[1] + perturbation * displacement[1],
                    FE::Real{0.0},
                }});
            }
            const SingleQuadBoundaryMesh mesh(
                wall_marker,
                /*rank=*/0,
                /*size=*/1,
                /*owner_rank=*/0,
                /*owned=*/true,
                FE::ElementType::Quad4,
                std::move(coordinates));
            auto interface_domain =
                linearQuadCutDomain(interface_marker, level_set_values);
            auto contact_domain =
                interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                    contactRequest(interface_marker, wall_marker),
                    interface_domain,
                    mesh);
            interfaces::GeneratedActiveBoundaryScalarField nodal_field;
            nodal_field.value_at_node =
                [level_set_values](FE::GlobalIndex node) {
                    return level_set_values.at(
                        static_cast<std::size_t>(node));
                };
            auto negative =
                interfaces::buildGeneratedActiveBoundaryDomain(
                    activeRequest(
                        interface_marker,
                        wall_marker,
                        FE::geometry::CutIntegrationSide::Negative),
                    interface_domain,
                    contact_domain,
                    mesh,
                    nodal_field);
            auto positive =
                interfaces::buildGeneratedActiveBoundaryDomain(
                    activeRequest(
                        interface_marker,
                        wall_marker,
                        FE::geometry::CutIntegrationSide::Positive),
                    interface_domain,
                    contact_domain,
                    mesh,
                    nodal_field);
            return interfaces::buildFreeSurfaceGeometrySnapshot(
                std::move(interface_domain),
                {std::move(contact_domain)},
                {std::move(negative), std::move(positive)},
                mesh,
                {},
                bilinearQuadScalar(level_set_values),
                std::move(domain_id));
        };

    const auto base_snapshot =
        build_snapshot(FE::Real{0.0}, "pressure_domain_variation_base");
    const auto plus_snapshot =
        build_snapshot(epsilon, "pressure_domain_variation_plus");
    const auto minus_snapshot =
        build_snapshot(-epsilon, "pressure_domain_variation_minus");

    const auto volume_role =
        [](FE::geometry::CutIntegrationSide side) {
            return side == FE::geometry::CutIntegrationSide::Negative
                ? interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume
                : interfaces::FreeSurfaceGeometryRuleRole::PositiveVolume;
        };
    const auto integrate_owned_pressure =
        [&](const interfaces::FreeSurfaceGeometrySnapshot& snapshot,
            FE::geometry::CutIntegrationSide side) {
            FE::Real integral{0.0};
            for (const auto& record : snapshot.rules()) {
                if (!record.locally_owned ||
                    record.retention !=
                        interfaces::FreeSurfaceGeometryRetention::Retained ||
                    record.role != volume_role(side)) {
                    continue;
                }
                if (record.reference_rule.points.size() !=
                    record.physical_rule.points.size()) {
                    throw std::logic_error(
                        "pressure-domain regression found mismatched quadrature points");
                }
                for (std::size_t q = 0u;
                     q < record.reference_rule.points.size();
                     ++q) {
                    integral +=
                        material_pressure(
                            record.reference_rule.points[q]
                                .parent_coordinate) *
                        record.physical_rule.points[q].physical_weight;
                }
            }
            return integral;
        };
    const auto integrate_owned_pressure_divergence =
        [&](FE::geometry::CutIntegrationSide side) {
            FE::Real integral{0.0};
            for (const auto& record : base_snapshot->rules()) {
                if (!record.locally_owned ||
                    record.retention !=
                        interfaces::FreeSurfaceGeometryRetention::Retained ||
                    record.role != volume_role(side)) {
                    continue;
                }
                for (std::size_t q = 0u;
                     q < record.reference_rule.points.size();
                     ++q) {
                    const auto& parent_coordinate =
                        record.reference_rule.points[q].parent_coordinate;
                    const auto point =
                        reference_to_base_point(parent_coordinate);
                    integral +=
                        material_pressure(parent_coordinate) *
                        deformation_divergence(point) *
                        record.physical_rule.points[q].physical_weight;
                }
            }
            return integral;
        };

    FE::Real maximum_relative_error{0.0};
    std::size_t case_count{0u};
    for (const auto liquid_side : {
             FE::geometry::CutIntegrationSide::Negative,
             FE::geometry::CutIntegrationSide::Positive}) {
        SCOPED_TRACE(static_cast<int>(liquid_side));
        const FE::Real domain_variation =
            (integrate_owned_pressure(*plus_snapshot, liquid_side) -
             integrate_owned_pressure(*minus_snapshot, liquid_side)) /
            (FE::Real{2.0} * epsilon);
        const FE::Real pressure_divergence =
            integrate_owned_pressure_divergence(liquid_side);
        const FE::Real scale =
            std::max({FE::Real{1.0e-12},
                      std::abs(domain_variation),
                      std::abs(pressure_divergence)});
        const FE::Real relative_error =
            std::abs(domain_variation - pressure_divergence) / scale;

        EXPECT_GT(std::abs(domain_variation), FE::Real{1.0e-8});
        EXPECT_LT(relative_error, FE::Real{2.0e-8});
        EXPECT_NEAR(-pressure_divergence,
                    -domain_variation,
                    FE::Real{2.0e-8} * scale)
            << "the pressure virtual-work sign is the negative material "
               "domain variation";
        maximum_relative_error =
            std::max(maximum_relative_error, relative_error);
        ++case_count;
    }

    EXPECT_NE(material_pressure({{-1.0, -1.0, 0.0}}),
              material_pressure({{1.0, 1.0, 0.0}}));
    EXPECT_NE(deformation_divergence({{0.0, 0.0, 0.0}}),
              deformation_divergence({{1.0, 1.0, 0.0}}));
    RecordProperty("general_pressure_domain_variation_case_count",
                   case_count);
    RecordProperty(
        "general_pressure_domain_variation_max_relative_error",
        ::testing::PrintToString(maximum_relative_error));
}

TEST(FreeSurfaceGeometrySnapshot,
     DiscreteFunctionalFirstVariationMatchesThreeDimensionalCentralDifference)
{
    constexpr int interface_marker = 125;
    constexpr int wall_marker = 22;
    constexpr FE::Real interface_intercept = 0.35;
    constexpr FE::Real interface_y_slope = -0.30;
    constexpr FE::Real interface_z_slope = -0.20;
    // V = (0.14 * (1 - x - y - z), 0, 0) is wall tangential and has
    // zero normal flux on every non-wall tetrahedral face reached by the cut.
    constexpr FE::Real translation = 0.14;
    constexpr FE::Real deformation_x_gradient = -0.14;
    constexpr FE::Real deformation_y_gradient = -0.14;
    constexpr FE::Real deformation_z_gradient = -0.14;
    constexpr FE::Real epsilon = 2.0e-6;
    const SingleTetraBoundaryMesh mesh(wall_marker);

    const auto moved_interface_x = [](FE::Real y,
                                      FE::Real z,
                                      FE::Real perturbation) {
        return (FE::Real{1.0} +
                perturbation * deformation_x_gradient) *
                   (interface_intercept + interface_y_slope * y +
                    interface_z_slope * z) +
               perturbation *
                   (translation + deformation_y_gradient * y +
                    deformation_z_gradient * z);
    };
    const auto nodal_values = [&](FE::Real perturbation) {
        std::array<FE::Real, 4> values{};
        for (std::size_t node = 0u; node < values.size(); ++node) {
            const auto xi =
                FE::basis::ReferenceNodeLayout::get_node_coords(
                    FE::ElementType::Tetra4, node);
            values[node] =
                xi[0] - moved_interface_x(xi[1], xi[2], perturbation);
        }
        return values;
    };
    const auto scalar_evaluator = [&](FE::Real perturbation) {
        interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
        scalar.value =
            [perturbation, moved_interface_x](
                FE::GlobalIndex,
                const std::array<FE::Real, 3>& point,
                const FE::geometry::CutQuadratureProvenance&) {
                return point[0] -
                       moved_interface_x(
                           point[1], point[2], perturbation);
            };
        scalar.reference_gradient =
            [perturbation](
                FE::GlobalIndex,
                const std::array<FE::Real, 3>&,
                const FE::geometry::CutQuadratureProvenance&) {
                const FE::Real scale =
                    FE::Real{1.0} +
                    perturbation * deformation_x_gradient;
                return std::array<FE::Real, 3>{{
                    FE::Real{1.0},
                    -(scale * interface_y_slope +
                      perturbation * deformation_y_gradient),
                    -(scale * interface_z_slope +
                      perturbation * deformation_z_gradient),
                }};
            };
        return scalar;
    };
    const auto build_snapshot = [&](FE::Real perturbation,
                                    std::string domain_id) {
        const auto values = nodal_values(perturbation);
        auto request = interfaceRequest(interface_marker);
        request.interface_quadrature_order = 1;
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = 0;
        input.element_type = FE::ElementType::Tetra4;
        input.node_coordinates.reserve(values.size());
        for (std::size_t node = 0u; node < values.size(); ++node) {
            const auto xi =
                FE::basis::ReferenceNodeLayout::get_node_coords(
                    FE::ElementType::Tetra4, node);
            input.node_coordinates.push_back({{xi[0], xi[1], xi[2]}});
        }
        input.level_set_values.assign(values.begin(), values.end());
        auto cut = interfaces::cutLinearLevelSetCell3D(request, input);
        if (!cut.supported || cut.fragments.empty() ||
            cut.volume_regions.empty()) {
            throw std::runtime_error(
                "three-dimensional functional regression requires a supported tetrahedral cut");
        }
        interfaces::LevelSetInterfaceDomain interface_domain(request);
        for (auto& fragment : cut.fragments) {
            interface_domain.addFragment(std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            interface_domain.addVolumeRegion(std::move(region));
        }

        auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(interface_marker, wall_marker),
                interface_domain,
                mesh);
        interfaces::GeneratedActiveBoundaryScalarField nodal_field;
        nodal_field.value_at_node = [values](FE::GlobalIndex node) {
            return values.at(static_cast<std::size_t>(node));
        };
        auto negative = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Negative),
            interface_domain,
            contact_domain,
            mesh,
            nodal_field);
        auto positive = interfaces::buildGeneratedActiveBoundaryDomain(
            activeRequest(interface_marker,
                          wall_marker,
                          FE::geometry::CutIntegrationSide::Positive),
            interface_domain,
            contact_domain,
            mesh,
            nodal_field);
        return interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(interface_domain),
            {std::move(contact_domain)},
            {std::move(negative), std::move(positive)},
            mesh,
            {},
            scalar_evaluator(perturbation),
            std::move(domain_id));
    };

    const auto base_snapshot =
        build_snapshot(0.0, "functional_fd_3d_base");
    const auto plus_snapshot =
        build_snapshot(epsilon, "functional_fd_3d_plus");
    const auto minus_snapshot =
        build_snapshot(-epsilon, "functional_fd_3d_minus");

    interfaces::FreeSurfaceDiscreteFunctionalDeformationEvaluator
        deformation;
    deformation.value =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>& xi,
           const FE::geometry::CutQuadratureProvenance&) {
            return std::array<FE::Real, 3>{{
                translation + deformation_x_gradient * xi[0] +
                    deformation_y_gradient * xi[1] +
                    deformation_z_gradient * xi[2],
                0.0,
                0.0,
            }};
        };
    deformation.physical_gradient =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return interfaces::FreeSurfaceDiscreteFunctionalPhysicalGradient{{
                {{deformation_x_gradient,
                  deformation_y_gradient,
                  deformation_z_gradient}},
                {{0.0, 0.0, 0.0}},
                {{0.0, 0.0, 0.0}},
            }};
        };

    const auto central_difference = [](FE::Real plus, FE::Real minus) {
        return (plus - minus) / (FE::Real{2.0} * epsilon);
    };
    const auto relative_error = [](FE::Real actual, FE::Real expected) {
        return std::abs(actual - expected) /
               std::max({FE::Real{1.0e-12},
                         std::abs(actual),
                         std::abs(expected)});
    };
    FE::Real maximum_relative_error{0.0};
    std::size_t case_count{0u};

    for (const auto liquid_side : {
             FE::geometry::CutIntegrationSide::Negative,
             FE::geometry::CutIntegrationSide::Positive}) {
        SCOPED_TRACE(static_cast<int>(liquid_side));
        interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
        parameters.liquid_side = liquid_side;
        parameters.surface_tension = FE::Real{1.6};
        parameters.young_wall_coefficients.push_back(
            {wall_marker, std::numbers::pi_v<FE::Real> / FE::Real{4.0}});
        parameters.volume_multiplier = FE::Real{-0.7};

        const auto base =
            interfaces::evaluateFreeSurfaceDiscreteFunctional(
                *base_snapshot, parameters);
        const auto plus =
            interfaces::evaluateFreeSurfaceDiscreteFunctional(
                *plus_snapshot, parameters);
        const auto minus =
            interfaces::evaluateFreeSurfaceDiscreteFunctional(
                *minus_snapshot, parameters);
        const auto variation =
            interfaces::evaluateFreeSurfaceDiscreteFunctionalFirstVariation(
                *base_snapshot, parameters, deformation);

        EXPECT_EQ(variation.snapshot_revision_key,
                  base.snapshot_revision_key);
        EXPECT_EQ(variation.liquid_side, liquid_side);
        ASSERT_EQ(base.walls.size(), 1u);
        ASSERT_EQ(plus.walls.size(), 1u);
        ASSERT_EQ(minus.walls.size(), 1u);
        ASSERT_EQ(variation.walls.size(), 1u);
        EXPECT_EQ(variation.walls.front().boundary_marker, wall_marker);

        const auto check_central_difference =
            [&](FE::Real actual,
                FE::Real plus_value,
                FE::Real minus_value,
                const char* quantity) {
                const FE::Real expected =
                    central_difference(plus_value, minus_value);
                const FE::Real error = relative_error(actual, expected);
                EXPECT_LT(error, 5.0e-7)
                    << quantity << ": actual=" << actual
                    << ", central_difference=" << expected;
                maximum_relative_error =
                    std::max(maximum_relative_error, error);
            };
        check_central_difference(
            variation.owned_liquid_gas_area_variation,
            plus.owned_liquid_gas_area,
            minus.owned_liquid_gas_area,
            "liquid-gas area");
        check_central_difference(
            variation.liquid_gas_surface_energy_variation,
            plus.liquid_gas_surface_energy,
            minus.liquid_gas_surface_energy,
            "liquid-gas surface energy");
        check_central_difference(
            variation.owned_wetted_wall_area_variation,
            plus.owned_wetted_wall_area,
            minus.owned_wetted_wall_area,
            "wetted-wall area");
        check_central_difference(
            variation.young_wall_energy_variation,
            plus.young_wall_energy,
            minus.young_wall_energy,
            "Young wall energy");
        check_central_difference(
            variation.owned_liquid_volume_variation,
            plus.owned_liquid_volume,
            minus.owned_liquid_volume,
            "liquid volume");
        check_central_difference(
            variation.volume_constraint_potential_variation,
            plus.volume_constraint_potential,
            minus.volume_constraint_potential,
            "volume constraint potential");
        check_central_difference(
            variation.total_potential_variation,
            plus.total_potential,
            minus.total_potential,
            "total potential");

        EXPECT_DOUBLE_EQ(
            variation.walls.front().owned_wetted_wall_area_variation,
            variation.owned_wetted_wall_area_variation);
        EXPECT_DOUBLE_EQ(
            variation.walls.front().young_wall_energy_variation,
            variation.young_wall_energy_variation);
        EXPECT_NE(variation.owned_liquid_volume_variation, 0.0);
        EXPECT_NE(variation.owned_liquid_gas_area_variation, 0.0);
        EXPECT_NE(variation.owned_wetted_wall_area_variation, 0.0);
        EXPECT_NE(variation.total_potential_variation, 0.0);
        EXPECT_NEAR(
            variation.total_potential_variation,
            variation.liquid_gas_surface_energy_variation +
                variation.young_wall_energy_variation +
                variation.volume_constraint_potential_variation,
            1.0e-14);
        ++case_count;
    }

    RecordProperty("functional_first_variation_3d_spatial_dimension", 3);
    RecordProperty("functional_first_variation_3d_fd_case_count",
                   case_count);
    RecordProperty(
        "functional_first_variation_3d_fd_max_relative_error",
        ::testing::PrintToString(maximum_relative_error));
}

TEST(FreeSurfaceGeometrySnapshot,
     GravitationalPotentialPowerMatchesMaterialTranslationDifference)
{
    constexpr int interface_marker = 116;
    constexpr FE::Real epsilon = FE::Real{1.0e-6};
    constexpr std::array<FE::Real, 3> translation_velocity{
        {FE::Real{0.37}, FE::Real{-0.21}, FE::Real{0.0}}};
    const std::vector<std::array<FE::Real, 3>> base_coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
    };
    const auto translated_coordinates =
        [&](FE::Real signed_step) {
            auto coordinates = base_coordinates;
            for (auto& point : coordinates) {
                for (std::size_t component = 0u;
                     component < point.size();
                     ++component) {
                    point[component] +=
                        signed_step *
                        translation_velocity[component];
                }
            }
            return coordinates;
        };

    const SingleQuadBoundaryMesh base_mesh(
        7, 0, 1, 0, true, FE::ElementType::Quad4,
        base_coordinates);
    const SingleQuadBoundaryMesh plus_mesh(
        7, 0, 1, 0, true, FE::ElementType::Quad4,
        translated_coordinates(epsilon));
    const SingleQuadBoundaryMesh minus_mesh(
        7, 0, 1, 0, true, FE::ElementType::Quad4,
        translated_coordinates(-epsilon));
    const auto build_snapshot = [&](const auto& mesh,
                                    std::string owner) {
        return interfaces::buildFreeSurfaceGeometrySnapshot(
            verticalInterfaceWithVolumes(interface_marker),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            std::move(owner));
    };
    const auto base_snapshot = build_snapshot(base_mesh, "gravity_base");
    const auto plus_snapshot = build_snapshot(plus_mesh, "gravity_plus");
    const auto minus_snapshot = build_snapshot(minus_mesh, "gravity_minus");

    interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator velocity;
    velocity.value =
        [translation_velocity](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return translation_velocity;
        };

    for (const auto side : {
             FE::geometry::CutIntegrationSide::Negative,
             FE::geometry::CutIntegrationSide::Positive}) {
        interfaces::FreeSurfaceActiveVolumeEnergyParameters parameters;
        parameters.liquid_side = side;
        parameters.density = FE::Real{2.7};
        parameters.gravitational_acceleration =
            {{FE::Real{1.4}, FE::Real{-2.3}, FE::Real{0.0}}};
        parameters.gravitational_reference_point =
            {{FE::Real{-0.8}, FE::Real{0.45}, FE::Real{0.0}}};

        const auto base =
            interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
                *base_snapshot, parameters, velocity);
        const auto plus =
            interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
                *plus_snapshot, parameters, velocity);
        const auto minus =
            interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
                *minus_snapshot, parameters, velocity);
        const FE::Real central_difference =
            (plus.gravitational_energy - minus.gravitational_energy) /
            (FE::Real{2.0} * epsilon);

        EXPECT_GT(base.owned_liquid_volume, FE::Real{0.0});
        EXPECT_NE(base.gravitational_potential_power, FE::Real{0.0});
        EXPECT_NEAR(
            central_difference,
            base.gravitational_potential_power,
            FE::Real{2.0e-9});
        EXPECT_NEAR(
            plus.owned_liquid_volume,
            base.owned_liquid_volume,
            FE::Real{1.0e-14});
        EXPECT_NEAR(
            minus.owned_liquid_volume,
            base.owned_liquid_volume,
            FE::Real{1.0e-14});
    }
}

TEST(FreeSurfaceGeometrySnapshot,
     DiscreteFunctionalExcludesGhostRuleContributions)
{
    constexpr int interface_marker = 117;
    const SingleQuadBoundaryMesh owner_mesh(
        /*marker=*/7,
        /*rank=*/0,
        /*size=*/2,
        /*owner_rank=*/0,
        /*owned=*/true);
    std::vector<std::uint64_t> owned_identities;
    interfaces::FreeSurfaceGeometryOwnershipCollective owner_collective;
    owner_collective.rank = 0;
    owner_collective.size = 2;
    owner_collective.all_gather_owned_rule_identity_values =
        [&owned_identities](std::span<const std::uint64_t> local) {
            owned_identities.assign(local.begin(), local.end());
            return owned_identities;
        };
    owner_collective.all_gather_revision_values =
        [](std::span<const std::uint64_t> local) {
            std::vector<std::uint64_t> gathered(local.begin(), local.end());
            gathered.insert(gathered.end(), local.begin(), local.end());
            return gathered;
        };
    const auto owner_snapshot =
        interfaces::buildFreeSurfaceGeometrySnapshot(
            distributedVerticalInterfaceWithVolumes(interface_marker),
            {},
            {},
            owner_mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "functional_owner",
            std::move(owner_collective));
    ASSERT_FALSE(owned_identities.empty());

    interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
    parameters.surface_tension = 2.0;
    parameters.volume_multiplier = 3.0;
    interfaces::FreeSurfaceDiscreteFunctionalDeformationEvaluator
        deformation;
    deformation.value =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return std::array<FE::Real, 3>{{0.25, 0.0, 0.0}};
        };
    deformation.physical_gradient =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return interfaces::FreeSurfaceDiscreteFunctionalPhysicalGradient{{
                {{0.0, 0.0, 0.0}},
                {{0.0, 0.2, 0.0}},
                {{0.0, 0.0, 0.0}},
            }};
        };
    const auto owner_state =
        interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *owner_snapshot, parameters);
    const auto owner_variation =
        interfaces::evaluateFreeSurfaceDiscreteFunctionalFirstVariation(
            *owner_snapshot, parameters, deformation);
    interfaces::FreeSurfaceActiveVolumeEnergyParameters
        volume_energy_parameters;
    volume_energy_parameters.density = 3.0;
    volume_energy_parameters.gravitational_acceleration =
        {{0.0, -2.0, 0.0}};
    interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator
        volume_velocity;
    volume_velocity.value =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return std::array<FE::Real, 3>{{2.0, -1.0, 0.0}};
        };
    volume_velocity.physical_gradient =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return FE::interfaces::
                FreeSurfaceDiscreteFunctionalPhysicalGradient{{
                    {{1.0, 2.0, 0.0}},
                    {{0.0, -1.0, 0.0}},
                    {{0.0, 0.0, 0.0}},
                }};
        };
    const auto owner_volume_energy =
        interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
            *owner_snapshot,
            volume_energy_parameters,
            volume_velocity);
    interfaces::FreeSurfaceActiveVolumeDissipationParameters
        volume_dissipation_parameters;
    volume_dissipation_parameters.dynamic_viscosity = 0.5;
    const auto owner_volume_dissipation =
        interfaces::evaluateFreeSurfaceActiveVolumeDissipation(
            *owner_snapshot,
            volume_dissipation_parameters,
            volume_velocity);
    interfaces::FreeSurfaceExternalPressurePowerParameters
        exterior_pressure_parameters;
    exterior_pressure_parameters.external_pressure = 2.5;
    const auto owner_exterior_pressure_power =
        interfaces::evaluateFreeSurfaceExternalPressurePower(
            *owner_snapshot,
            exterior_pressure_parameters,
            volume_velocity);
    interfaces::FreeSurfaceDiscreteFunctionalDeformationEvaluator
        endpoint_velocity_deformation;
    endpoint_velocity_deformation.value = volume_velocity.value;
    endpoint_velocity_deformation.physical_gradient =
        volume_velocity.physical_gradient;
    const auto endpoint_velocity_variation =
        interfaces::evaluateFreeSurfaceDiscreteFunctionalFirstVariation(
            *owner_snapshot,
            parameters,
            endpoint_velocity_deformation);
    interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator
        previous_volume_velocity;
    previous_volume_velocity.value =
        [](FE::GlobalIndex,
           const std::array<FE::Real, 3>&,
           const FE::geometry::CutQuadratureProvenance&) {
            return std::array<FE::Real, 3>{{1.0, 0.0, 0.0}};
        };
    const auto owner_kinetic_work =
        interfaces::evaluateFreeSurfaceBackwardEulerKineticWork(
            *owner_snapshot,
            volume_energy_parameters.liquid_side,
            volume_energy_parameters.density,
            /*previous_velocity_revision=*/17u,
            /*endpoint_velocity_revision=*/18u,
            previous_volume_velocity,
            volume_velocity);
    EXPECT_GT(owner_state.owned_liquid_volume, 0.0);
    EXPECT_GT(owner_state.owned_liquid_gas_area, 0.0);
    EXPECT_GT(owner_variation.owned_liquid_volume_variation, 0.0);
    EXPECT_GT(owner_variation.owned_liquid_gas_area_variation, 0.0);
    EXPECT_GT(owner_variation.total_potential_variation, 0.0);
    EXPECT_EQ(
        owner_volume_energy.snapshot_revision_key,
        owner_snapshot->revision().snapshot_revision_key);
    EXPECT_GT(owner_volume_energy.owned_quadrature_point_count, 0u);
    EXPECT_NEAR(
        owner_volume_energy.owned_liquid_volume,
        owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_volume_energy.kinetic_energy,
        FE::Real{0.5} * FE::Real{3.0} * FE::Real{5.0} *
            owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_volume_energy.gravitational_energy,
        1.5,
        1.0e-14);
    EXPECT_NEAR(
        owner_volume_energy.gravitational_potential_power,
        FE::Real{-6.0} * owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_volume_energy.total_energy,
        owner_volume_energy.kinetic_energy +
            owner_volume_energy.gravitational_energy,
        1.0e-14);
    EXPECT_EQ(
        owner_volume_dissipation.snapshot_revision_key,
        owner_snapshot->revision().snapshot_revision_key);
    EXPECT_EQ(
        owner_volume_dissipation.owned_quadrature_point_count,
        owner_volume_energy.owned_quadrature_point_count);
    EXPECT_NEAR(
        owner_volume_dissipation.owned_liquid_volume,
        owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_volume_dissipation.bulk_viscous_dissipation_rate,
        FE::Real{4.0} * owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_EQ(
        owner_exterior_pressure_power.snapshot_revision_key,
        owner_snapshot->revision().snapshot_revision_key);
    EXPECT_GT(
        owner_exterior_pressure_power.owned_quadrature_point_count,
        0u);
    EXPECT_NEAR(
        owner_exterior_pressure_power.owned_liquid_gas_area,
        owner_state.owned_liquid_gas_area,
        1.0e-14);
    EXPECT_NEAR(
        owner_exterior_pressure_power
            .outward_liquid_volume_flux_rate,
        endpoint_velocity_variation
            .owned_liquid_volume_variation,
        1.0e-14);
    EXPECT_NEAR(
        owner_exterior_pressure_power.external_pressure_power,
        -exterior_pressure_parameters.external_pressure *
            owner_exterior_pressure_power
                .outward_liquid_volume_flux_rate,
        1.0e-14);
    EXPECT_EQ(
        owner_kinetic_work.snapshot_revision_key,
        owner_snapshot->revision().snapshot_revision_key);
    EXPECT_EQ(owner_kinetic_work.previous_velocity_revision, 17u);
    EXPECT_EQ(owner_kinetic_work.endpoint_velocity_revision, 18u);
    EXPECT_EQ(
        owner_kinetic_work.owned_quadrature_point_count,
        owner_volume_energy.owned_quadrature_point_count);
    EXPECT_NEAR(
        owner_kinetic_work.owned_liquid_volume,
        owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_kinetic_work.kinetic_energy_before_on_endpoint_domain,
        FE::Real{1.5} * owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_kinetic_work.kinetic_energy_after,
        FE::Real{7.5} * owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_kinetic_work.kinetic_energy_change_on_endpoint_domain,
        FE::Real{6.0} * owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_kinetic_work.step_integrated_inertia_work,
        FE::Real{9.0} * owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(
        owner_kinetic_work.time_discretization_loss,
        FE::Real{3.0} * owner_state.owned_liquid_volume,
        1.0e-14);
    EXPECT_NEAR(owner_kinetic_work.identity_residual, 0.0, 1.0e-14);
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
            *owner_snapshot,
            volume_energy_parameters,
            {}),
        std::invalid_argument);
    auto invalid_volume_energy_parameters =
        volume_energy_parameters;
    invalid_volume_energy_parameters.density = 0.0;
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
            *owner_snapshot,
            invalid_volume_energy_parameters,
            volume_velocity),
        std::invalid_argument);
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceBackwardEulerKineticWork(
            *owner_snapshot,
            volume_energy_parameters.liquid_side,
            volume_energy_parameters.density,
            /*previous_velocity_revision=*/17u,
            /*endpoint_velocity_revision=*/18u,
            previous_volume_velocity,
            {}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceBackwardEulerKineticWork(
            *owner_snapshot,
            volume_energy_parameters.liquid_side,
            volume_energy_parameters.density,
            /*previous_velocity_revision=*/0u,
            /*endpoint_velocity_revision=*/18u,
            previous_volume_velocity,
            volume_velocity),
        std::invalid_argument);
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceActiveVolumeDissipation(
            *owner_snapshot,
            volume_dissipation_parameters,
            previous_volume_velocity),
        std::invalid_argument);
    auto invalid_volume_dissipation_parameters =
        volume_dissipation_parameters;
    invalid_volume_dissipation_parameters.dynamic_viscosity = 0.0;
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceActiveVolumeDissipation(
            *owner_snapshot,
            invalid_volume_dissipation_parameters,
            volume_velocity),
        std::invalid_argument);
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceExternalPressurePower(
            *owner_snapshot,
            exterior_pressure_parameters,
            {}),
        std::invalid_argument);
    auto invalid_exterior_pressure_parameters =
        exterior_pressure_parameters;
    invalid_exterior_pressure_parameters.external_pressure =
        std::numeric_limits<FE::Real>::quiet_NaN();
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceExternalPressurePower(
            *owner_snapshot,
            invalid_exterior_pressure_parameters,
            volume_velocity),
        std::invalid_argument);

    const SingleQuadBoundaryMesh ghost_mesh(
        /*marker=*/7,
        /*rank=*/1,
        /*size=*/2,
        /*owner_rank=*/0,
        /*owned=*/false);
    interfaces::FreeSurfaceGeometryOwnershipCollective ghost_collective;
    ghost_collective.rank = 1;
    ghost_collective.size = 2;
    ghost_collective.all_gather_owned_rule_identity_values =
        [&owned_identities](std::span<const std::uint64_t> local) {
            EXPECT_TRUE(local.empty());
            return owned_identities;
        };
    ghost_collective.all_gather_revision_values =
        [](std::span<const std::uint64_t> local) {
            std::vector<std::uint64_t> gathered(local.begin(), local.end());
            gathered.insert(gathered.end(), local.begin(), local.end());
            return gathered;
        };
    const auto ghost_snapshot =
        interfaces::buildFreeSurfaceGeometrySnapshot(
            distributedVerticalInterfaceWithVolumes(interface_marker),
            {},
            {},
            ghost_mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "functional_owner",
            std::move(ghost_collective));
    const auto ghost_state =
        interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *ghost_snapshot, parameters);
    const auto ghost_variation =
        interfaces::evaluateFreeSurfaceDiscreteFunctionalFirstVariation(
            *ghost_snapshot, parameters, deformation);
    const auto ghost_volume_energy =
        interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
            *ghost_snapshot,
            volume_energy_parameters,
            volume_velocity);
    const auto ghost_volume_dissipation =
        interfaces::evaluateFreeSurfaceActiveVolumeDissipation(
            *ghost_snapshot,
            volume_dissipation_parameters,
            volume_velocity);
    const auto ghost_exterior_pressure_power =
        interfaces::evaluateFreeSurfaceExternalPressurePower(
            *ghost_snapshot,
            exterior_pressure_parameters,
            volume_velocity);
    const auto ghost_kinetic_work =
        interfaces::evaluateFreeSurfaceBackwardEulerKineticWork(
            *ghost_snapshot,
            volume_energy_parameters.liquid_side,
            volume_energy_parameters.density,
            /*previous_velocity_revision=*/17u,
            /*endpoint_velocity_revision=*/18u,
            previous_volume_velocity,
            volume_velocity);
    EXPECT_EQ(ghost_snapshot->revision().snapshot_revision_key,
              owner_snapshot->revision().snapshot_revision_key);
    EXPECT_EQ(ghost_state.owned_liquid_volume, 0.0);
    EXPECT_EQ(ghost_state.owned_liquid_gas_area, 0.0);
    EXPECT_EQ(ghost_state.owned_wetted_wall_area, 0.0);
    EXPECT_EQ(ghost_state.owned_contact_measure, 0.0);
    EXPECT_EQ(ghost_state.total_potential, 0.0);
    EXPECT_EQ(ghost_variation.owned_liquid_volume_variation, 0.0);
    EXPECT_EQ(ghost_variation.owned_liquid_gas_area_variation, 0.0);
    EXPECT_EQ(ghost_variation.owned_wetted_wall_area_variation, 0.0);
    EXPECT_EQ(ghost_variation.liquid_gas_surface_energy_variation, 0.0);
    EXPECT_EQ(ghost_variation.young_wall_energy_variation, 0.0);
    EXPECT_EQ(ghost_variation.volume_constraint_potential_variation, 0.0);
    EXPECT_EQ(ghost_variation.total_potential_variation, 0.0);
    EXPECT_EQ(ghost_volume_energy.owned_quadrature_point_count, 0u);
    EXPECT_EQ(ghost_volume_energy.owned_liquid_volume, 0.0);
    EXPECT_EQ(ghost_volume_energy.kinetic_energy, 0.0);
    EXPECT_EQ(ghost_volume_energy.gravitational_energy, 0.0);
    EXPECT_EQ(
        ghost_volume_energy.gravitational_potential_power, 0.0);
    EXPECT_EQ(ghost_volume_energy.total_energy, 0.0);
    EXPECT_EQ(
        ghost_volume_dissipation.owned_quadrature_point_count, 0u);
    EXPECT_EQ(ghost_volume_dissipation.owned_liquid_volume, 0.0);
    EXPECT_EQ(
        ghost_volume_dissipation.bulk_viscous_dissipation_rate,
        0.0);
    EXPECT_EQ(
        ghost_exterior_pressure_power.owned_quadrature_point_count,
        0u);
    EXPECT_EQ(
        ghost_exterior_pressure_power.owned_liquid_gas_area,
        0.0);
    EXPECT_EQ(
        ghost_exterior_pressure_power
            .outward_liquid_volume_flux_rate,
        0.0);
    EXPECT_EQ(
        ghost_exterior_pressure_power.external_pressure_power,
        0.0);
    EXPECT_EQ(ghost_kinetic_work.owned_quadrature_point_count, 0u);
    EXPECT_EQ(ghost_kinetic_work.previous_velocity_revision, 17u);
    EXPECT_EQ(ghost_kinetic_work.endpoint_velocity_revision, 18u);
    EXPECT_EQ(ghost_kinetic_work.owned_liquid_volume, 0.0);
    EXPECT_EQ(
        ghost_kinetic_work.kinetic_energy_before_on_endpoint_domain,
        0.0);
    EXPECT_EQ(ghost_kinetic_work.kinetic_energy_after, 0.0);
    EXPECT_EQ(
        ghost_kinetic_work.kinetic_energy_change_on_endpoint_domain,
        0.0);
    EXPECT_EQ(ghost_kinetic_work.step_integrated_inertia_work, 0.0);
    EXPECT_EQ(ghost_kinetic_work.time_discretization_loss, 0.0);
    EXPECT_EQ(ghost_kinetic_work.identity_residual, 0.0);
    RecordProperty("functional_first_variation_ghost_contribution_count", 0);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsPointOutsideParentReferenceCell)
{
    constexpr int interface_marker = 106;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(
        interface_marker, /*negative_point_x=*/2.0);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "outside_parent"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsQuadraturePointOnWrongPhase)
{
    constexpr int interface_marker = 107;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(interface_marker);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(/*sign=*/-1.0),
            "wrong_phase"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsNormalOpposedToRepresentedGradient)
{
    constexpr int interface_marker = 108;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(interface_marker);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(
                /*sign=*/1.0,
                /*gradient=*/{{-1.0, 0.0, 0.0}}),
            "wrong_normal"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsNearlyOrthogonalNormalToRepresentedGradient)
{
    constexpr int interface_marker = 125;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(interface_marker);
    try {
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(
                /*sign=*/1.0,
                /*gradient=*/{{1.0e-12, 1.0, 0.0}}),
            "nearly_orthogonal_normal");
        FAIL() << "nearly orthogonal represented normal was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("normal disagrees"),
                  std::string::npos);
    }
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsDegenerateOrNonFiniteGradientAndNormal)
{
    constexpr int interface_marker = 129;
    const SingleQuadBoundaryMesh mesh;
    const auto expect_gradient_rejection =
        [&](std::array<FE::Real, 3> gradient, const char* domain_id) {
            auto scalar = verticalScalar();
            scalar.reference_gradient =
                [gradient](FE::GlobalIndex,
                           const std::array<FE::Real, 3>&,
                           const FE::geometry::CutQuadratureProvenance&) {
                    return gradient;
                };
            EXPECT_THROW(
                (void)interfaces::buildFreeSurfaceGeometrySnapshot(
                    verticalInterfaceWithVolumes(interface_marker),
                    {},
                    {},
                    mesh,
                    snapshotPolicyWithoutBoundary(),
                    std::move(scalar),
                    domain_id),
                std::invalid_argument);
        };
    expect_gradient_rejection({{0.0, 0.0, 0.0}}, "zero_gradient");
    expect_gradient_rejection(
        {{std::numeric_limits<FE::Real>::quiet_NaN(), 0.0, 0.0}},
        "nonfinite_gradient");

    const auto expect_normal_rejection =
        [&](std::array<FE::Real, 3> normal, const char* domain_id) {
            const auto source = verticalInterfaceWithVolumes(interface_marker);
            interfaces::LevelSetInterfaceDomain defect(source.request());
            for (auto fragment : source.fragments()) {
                fragment.normal = normal;
                for (auto& point : fragment.quadrature_points) {
                    point.normal = normal;
                }
                defect.addFragment(std::move(fragment));
            }
            for (auto region : source.volumeRegions()) {
                defect.addVolumeRegion(std::move(region));
            }
            EXPECT_THROW(
                (void)interfaces::buildFreeSurfaceGeometrySnapshot(
                    std::move(defect),
                    {},
                    {},
                    mesh,
                    snapshotPolicyWithoutBoundary(),
                    verticalScalar(),
                    domain_id),
                std::invalid_argument);
        };
    expect_normal_rejection({{0.0, 0.0, 0.0}}, "zero_normal");
    expect_normal_rejection(
        {{std::numeric_limits<FE::Real>::quiet_NaN(), 0.0, 0.0}},
        "nonfinite_normal");
    RecordProperty("invalid_gradient_normal_rejection_count", 4);
}

TEST(FreeSurfaceGeometrySnapshot,
     AcceptsZeroOneAndMultipleContactsPerSourceAndRecordsLedger)
{
    constexpr int wall_marker = 21;
    const SingleQuadBoundaryMesh mesh(
        wall_marker,
        /*rank=*/0,
        /*size=*/1,
        /*owner_rank=*/0,
        /*owned=*/true,
        FE::ElementType::Quad4,
        {},
        /*expose_opposite_face=*/true);
    struct Scenario {
        int interface_marker;
        FE::Real lower;
        FE::Real upper;
        std::size_t expected_contact_count;
        const char* label;
    };
    const std::array<Scenario, 3> scenarios{{
        {270, -0.5, 0.5, 0u, "zero"},
        {271, -1.0, 0.5, 1u, "one"},
        {272, -1.0, 1.0, 2u, "multiple"},
    }};

    std::size_t total_rule_count{0u};
    std::size_t total_contact_count{0u};
    std::size_t total_orphan_count{0u};
    std::size_t total_missing_count{0u};
    std::size_t total_stale_count{0u};
    FE::Real maximum_root_error{0.0};
    FE::Real maximum_normal_error{0.0};
    FE::Real maximum_constant_error{0.0};
    FE::Real maximum_polynomial_error{0.0};
    FE::Real maximum_scaled_polynomial_error{0.0};

    for (const auto& scenario : scenarios) {
        SCOPED_TRACE(scenario.label);
        auto interface_domain = verticalInterfaceSegment(
            scenario.interface_marker, scenario.lower, scenario.upper);
        ASSERT_EQ(interface_domain.fragments().size(), 1u);
        const auto source_id = interface_domain.fragments().front().stable_id;
        ASSERT_NE(source_id, 0u);
        auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(scenario.interface_marker, wall_marker),
                interface_domain,
                mesh);
        EXPECT_EQ(contact_domain.summary().active_fragment_count,
                  scenario.expected_contact_count);
        for (const auto& contact : contact_domain.fragments()) {
            if (!contact.active()) {
                continue;
            }
            const auto matching_source_count = std::count_if(
                interface_domain.fragments().begin(),
                interface_domain.fragments().end(),
                [&contact](const auto& source) {
                    return source.stable_id ==
                           contact.source_interface_stable_id;
                });
            EXPECT_EQ(matching_source_count, 1);
            EXPECT_EQ(contact.source_interface_stable_id, source_id);
        }

        const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(interface_domain),
            {std::move(contact_domain)},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            std::string("contact_cardinality_") + scenario.label);
        ASSERT_NE(snapshot, nullptr);
        const auto& ledger = snapshot->ledger();
        EXPECT_EQ(ledger.rule_count,
                  3u + scenario.expected_contact_count);
        EXPECT_EQ(ledger.contact_fragment_count,
                  scenario.expected_contact_count);
        EXPECT_EQ(ledger.referenced_surface_fragment_count,
                  scenario.expected_contact_count == 0u ? 0u : 1u);
        EXPECT_EQ(ledger.orphan_contact_fragment_count, 0u);
        EXPECT_EQ(ledger.missing_contact_fragment_count, 0u);
        EXPECT_EQ(ledger.stale_revision_count, 0u);
        EXPECT_LE(ledger.maximum_polynomial_moment_scaled_error, 1.0);

        total_rule_count += ledger.rule_count;
        total_contact_count += ledger.contact_fragment_count;
        total_orphan_count += ledger.orphan_contact_fragment_count;
        total_missing_count += ledger.missing_contact_fragment_count;
        total_stale_count += ledger.stale_revision_count;
        maximum_root_error =
            std::max(maximum_root_error, ledger.maximum_root_residual);
        maximum_normal_error = std::max(
            maximum_normal_error, ledger.maximum_normal_angular_error);
        maximum_constant_error = std::max(
            maximum_constant_error, ledger.maximum_constant_moment_error);
        maximum_polynomial_error = std::max(
            maximum_polynomial_error,
            ledger.maximum_polynomial_moment_error);
        maximum_scaled_polynomial_error = std::max(
            maximum_scaled_polynomial_error,
            ledger.maximum_polynomial_moment_scaled_error);
    }

    RecordProperty("source_contact_count_zero", 0);
    RecordProperty("source_contact_count_one", 1);
    RecordProperty("source_contact_count_multiple", 2);
    RecordProperty("geometry_ledger_rule_count",
                   static_cast<int>(total_rule_count));
    RecordProperty("geometry_ledger_contact_fragment_count",
                   static_cast<int>(total_contact_count));
    RecordProperty("geometry_ledger_orphan_contact_count",
                   static_cast<int>(total_orphan_count));
    RecordProperty("geometry_ledger_missing_contact_count",
                   static_cast<int>(total_missing_count));
    RecordProperty("geometry_ledger_stale_revision_count",
                   static_cast<int>(total_stale_count));
    RecordProperty("geometry_ledger_maximum_root_residual",
                   ::testing::PrintToString(maximum_root_error));
    RecordProperty("geometry_ledger_maximum_normal_angular_error",
                   ::testing::PrintToString(maximum_normal_error));
    RecordProperty("geometry_ledger_maximum_constant_moment_error",
                   ::testing::PrintToString(maximum_constant_error));
    RecordProperty("geometry_ledger_maximum_polynomial_moment_error",
                   ::testing::PrintToString(maximum_polynomial_error));
    RecordProperty("geometry_ledger_maximum_scaled_polynomial_moment_error",
                   ::testing::PrintToString(maximum_scaled_polynomial_error));
}

TEST(FreeSurfaceGeometrySnapshot, RejectsIncorrectConstantMoment)
{
    constexpr int interface_marker = 109;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(
        interface_marker,
        /*negative_point_x=*/-0.5,
        /*negative_weight=*/1.5);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "wrong_moment"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsIncorrectLinearPartitionMoment)
{
    constexpr int interface_marker = 120;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(
        interface_marker,
        /*negative_point_x=*/-0.25);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "wrong_linear_partition_moment"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsCompensatingPerPhaseMomentDefects)
{
    constexpr int interface_marker = 122;
    const SingleQuadBoundaryMesh mesh;
    const auto source = verticalInterfaceWithVolumes(interface_marker);
    interfaces::LevelSetInterfaceDomain defect(source.request());
    for (auto fragment : source.fragments()) {
        defect.addFragment(std::move(fragment));
    }
    for (auto region : source.volumeRegions()) {
        ASSERT_EQ(region.quadrature_points.size(), 1u);
        const FE::Real shifted_x =
            region.side == FE::geometry::CutIntegrationSide::Negative
                ? FE::Real{-0.25}
                : FE::Real{0.25};
        region.quadrature_points.front().point[0] = shifted_x;
        region.quadrature_points.front().parent_coordinate[0] = shifted_x;
        defect.addVolumeRegion(std::move(region));
    }
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(defect),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "compensating_phase_moment_defects"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot,
     CertifiesQuadraticVolumeRulesFromSourceSubcells)
{
    constexpr int interface_marker = 124;
    const SingleQuadBoundaryMesh mesh;
    auto make_domain = []() {
        auto request = interfaceRequest(interface_marker);
        request.interface_quadrature_order = 1;
        request.volume_quadrature_order = 2;
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = 0;
        input.element_type = FE::ElementType::Quad4;
        input.node_coordinates = {
            {{-1.0, -1.0, 0.0}},
            {{1.0, -1.0, 0.0}},
            {{1.0, 1.0, 0.0}},
            {{-1.0, 1.0, 0.0}},
        };
        input.level_set_values = {-1.0, 1.0, 1.0, -1.0};
        auto cut = interfaces::cutLinearLevelSetCell2D(request, input);
        interfaces::LevelSetInterfaceDomain domain(request);
        for (auto& fragment : cut.fragments) {
            domain.addFragment(std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            domain.addVolumeRegion(std::move(region));
        }
        return domain;
    };

    auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        make_domain(),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        verticalScalar(),
        "source_subcell_quadratic_volume");
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(snapshot->ledger().stored_generated_moment_certificate_count,
              0u);
    EXPECT_GT(snapshot->ledger().represented_phase_point_count, 0u);
    EXPECT_EQ(snapshot->ledger().represented_phase_disagreement_count, 0u);
    std::size_t volume_rule_count = 0u;
    for (const auto& record : snapshot->rules()) {
        if (record.reference_rule.kind !=
            FE::geometry::CutQuadratureKind::Volume) {
            continue;
        }
        ++volume_rule_count;
        EXPECT_EQ(record.reference_rule.exact_polynomial_order, 2);
        EXPECT_EQ(record.moment_certificate.source,
                  interfaces::FreeSurfaceGeometryMomentCertificateSource::
                      PiecewiseAffineGeometry);
        EXPECT_TRUE(record.moment_certificate.phase_sign_certified);
    }
    EXPECT_EQ(volume_rule_count, 2u);

    auto defect = make_domain();
    interfaces::LevelSetInterfaceDomain mutated(defect.request());
    for (auto fragment : defect.fragments()) {
        mutated.addFragment(std::move(fragment));
    }
    bool changed = false;
    for (auto region : defect.volumeRegions()) {
        if (!changed &&
            region.side == FE::geometry::CutIntegrationSide::Negative) {
            ASSERT_GE(region.quadrature_points.size(), 2u);
            ASSERT_NEAR(region.quadrature_points[0].weight,
                        region.quadrature_points[1].weight,
                        1.0e-15);
            region.quadrature_points[0].point[1] += 0.05;
            region.quadrature_points[0].parent_coordinate[1] += 0.05;
            region.quadrature_points[1].point[1] -= 0.05;
            region.quadrature_points[1].parent_coordinate[1] -= 0.05;
            changed = true;
        }
        mutated.addVolumeRegion(std::move(region));
    }
    ASSERT_TRUE(changed);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(mutated),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "source_subcell_quadratic_volume_defect"),
        std::invalid_argument);

    auto swapped_source = make_domain();
    interfaces::LevelSetInterfaceDomain swapped(swapped_source.request());
    for (auto fragment : swapped_source.fragments()) {
        swapped.addFragment(std::move(fragment));
    }
    std::size_t swapped_phase_count = 0u;
    for (auto region : swapped_source.volumeRegions()) {
        region.side =
            region.side == FE::geometry::CutIntegrationSide::Negative
                ? FE::geometry::CutIntegrationSide::Positive
                : FE::geometry::CutIntegrationSide::Negative;
        ++swapped_phase_count;
        swapped.addVolumeRegion(std::move(region));
    }
    ASSERT_EQ(swapped_phase_count, 2u);
    try {
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(swapped),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "source_subcell_swapped_phase");
        FAIL() << "swapped represented phase was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find(
                      "wrong represented source phase"),
                  std::string::npos);
    }
}

TEST(FreeSurfaceGeometrySnapshot,
     RejectsInterfaceMomentDefectWithValidRootsAndMeasure)
{
    constexpr int interface_marker = 123;
    const SingleQuadBoundaryMesh mesh;
    const auto source = verticalInterfaceWithVolumes(interface_marker);
    interfaces::LevelSetInterfaceDomain defect(source.request());
    for (auto fragment : source.fragments()) {
        for (auto& point : fragment.quadrature_points) {
            point.point[1] += FE::Real{0.125};
            point.parent_coordinate[1] += FE::Real{0.125};
        }
        defect.addFragment(std::move(fragment));
    }
    for (auto region : source.volumeRegions()) {
        defect.addVolumeRegion(std::move(region));
    }
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(defect),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "interface_moment_defect"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot,
     MaterializesFullCurvedCellRuleForMappingAndMomentValidation)
{
    constexpr int interface_marker = 121;
    std::vector<std::array<FE::Real, 3>> coordinates;
    coordinates.reserve(9u);
    for (std::size_t node = 0u; node < 9u; ++node) {
        const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(
            FE::ElementType::Quad9, node);
        coordinates.push_back({{
            xi[0],
            xi[1] + FE::Real{0.1} *
                        (FE::Real{1.0} - xi[0] * xi[0]) *
                        (FE::Real{1.0} - xi[1]),
            0.0,
        }});
    }
    const SingleQuadBoundaryMesh mesh(
        /*marker=*/7,
        /*rank=*/0,
        /*size=*/1,
        /*owner_rank=*/0,
        /*owned=*/true,
        FE::ElementType::Quad9,
        std::move(coordinates));
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [](FE::GlobalIndex,
                      const std::array<FE::Real, 3>&,
                      const FE::geometry::CutQuadratureProvenance&) {
        return FE::Real{-1.0};
    };

    const auto snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
        fullNegativeCell(interface_marker),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        std::move(scalar),
        "curved_full_cell");
    const auto rules = snapshot->retainedRules(
        interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume);
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_GT(rules.front()->reference_rule.points.size(), 1u);
    EXPECT_NEAR(rules.front()->physical_rule.physical_measure,
                FE::Real{56.0} / FE::Real{15.0},
                1.0e-12);
    EXPECT_NEAR(snapshot->ledger().retained_negative_physical_volume,
                rules.front()->physical_rule.physical_measure,
                1.0e-14);
    EXPECT_EQ(snapshot->ledger().certified_rule_count, 1u);
    EXPECT_EQ(snapshot->ledger().parent_cell_moment_certificate_count, 1u);
    EXPECT_EQ(snapshot->ledger().validated_rule_polynomial_moment_count, 6u);
    EXPECT_EQ(snapshot->ledger().validated_polynomial_moment_count, 12u);
    EXPECT_LE(snapshot->ledger().maximum_polynomial_moment_scaled_error,
              1.0);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsImpossibleClaimedQuadraticOrder)
{
    constexpr int interface_marker = 110;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(
        interface_marker,
        /*negative_point_x=*/-0.5,
        /*negative_weight=*/2.0,
        /*negative_achieved_order=*/2);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "false_order"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsAchievedOrderBelowPolicy)
{
    constexpr int interface_marker = 111;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterfaceWithVolumes(interface_marker);
    auto policy = snapshotPolicyWithoutBoundary();
    policy.minimum_achieved_quadrature_order = 2;
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            policy,
            verticalScalar(),
            "insufficient_order"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsStaleAndDuplicateContactDomains)
{
    constexpr int interface_marker = 112;
    constexpr int wall_marker = 11;
    const SingleQuadBoundaryMesh mesh(wall_marker);

    auto stale_request = contactRequest(interface_marker, wall_marker);
    stale_request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/4, /*layout_revision=*/3, /*value_revision=*/8);
    stale_request.source_value_revision = 8;
    interfaces::GeneratedInterfaceBoundaryIntersectionDomain stale_contact(
        std::move(stale_request));
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            verticalInterfaceWithVolumes(interface_marker),
            {std::move(stale_contact)},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "stale_contact"),
        std::invalid_argument);

    auto domain = verticalInterfaceWithVolumes(interface_marker);
    const auto contact =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            contactRequest(interface_marker, wall_marker), domain, mesh);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {contact, contact},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "duplicate_contact"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsMissingExpectedContactTrace)
{
    constexpr int interface_marker = 118;
    constexpr int wall_marker = 11;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    interfaces::GeneratedInterfaceBoundaryIntersectionDomain missing_contact(
        contactRequest(interface_marker, wall_marker));

    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            verticalInterfaceWithVolumes(interface_marker),
            {std::move(missing_contact)},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "missing_contact_trace"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshotCache,
     HoldsWeakReferencesAndEvictsOnlyAfterConsumersRelease)
{
    constexpr int interface_marker = 113;
    const SingleQuadBoundaryMesh mesh;
    std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot> snapshot =
        interfaces::buildFreeSurfaceGeometrySnapshot(
            verticalInterfaceWithVolumes(interface_marker),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            verticalScalar(),
            "cache_lifetime");
    const auto key = snapshot->revision().snapshot_revision_key;

    interfaces::FreeSurfaceGeometrySnapshotCache cache;
    cache.insert(snapshot);
    EXPECT_EQ(cache.find(key), snapshot);
    auto live = cache.statistics();
    EXPECT_EQ(live.live_snapshot_count, 1u);
    EXPECT_GE(live.live_resident_bytes, snapshot->residentBytes());
    EXPECT_EQ(live.hit_count, 1u);

    snapshot.reset();
    cache.evictExpired();
    EXPECT_FALSE(cache.find(key));
    const auto expired = cache.statistics();
    EXPECT_EQ(expired.live_snapshot_count, 0u);
    EXPECT_GE(expired.expired_eviction_count, 1u);
    EXPECT_EQ(expired.miss_count, 1u);
    EXPECT_GE(expired.peak_live_snapshot_count, 1u);
    RecordProperty("peak_live_snapshot_count",
                   std::to_string(expired.peak_live_snapshot_count));
    RecordProperty("peak_live_resident_bytes",
                   std::to_string(expired.peak_live_resident_bytes));
    RecordProperty("final_live_snapshot_count",
                   std::to_string(expired.live_snapshot_count));
    RecordProperty("expired_eviction_count",
                   std::to_string(expired.expired_eviction_count));
    RecordProperty("snapshot_cache_hit_count",
                   std::to_string(expired.hit_count));
    RecordProperty("snapshot_cache_miss_count",
                   std::to_string(expired.miss_count));
}

TEST(FreeSurfaceGeometrySnapshotCache,
     RevisionKeyIncludesAuthoritativeQuadratureContent)
{
    constexpr int interface_marker = 114;
    const SingleQuadBoundaryMesh mesh;
    const auto first = interfaces::buildFreeSurfaceGeometrySnapshot(
        verticalInterfaceWithVolumes(
            interface_marker,
            /*negative_point_x=*/-0.5),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        verticalScalar(),
        "cache_content_key");
    const auto second = interfaces::buildFreeSurfaceGeometrySnapshot(
        verticalInterfaceWithVolumes(
            interface_marker,
            /*negative_point_x=*/-0.25,
            /*negative_weight=*/2.0,
            /*negative_achieved_order=*/1,
            /*positive_point_x=*/0.25),
        {},
        {},
        mesh,
        snapshotPolicyWithoutBoundary(),
        verticalScalar(),
        "cache_content_key");
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_NE(first->revision().snapshot_revision_key,
              second->revision().snapshot_revision_key);
}
