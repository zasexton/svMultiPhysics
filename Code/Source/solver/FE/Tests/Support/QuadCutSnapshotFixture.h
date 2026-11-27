#pragma once

#include "FE/Assembly/Assembler.h"
#include "FE/Interfaces/GeneratedActiveBoundaryDomain.h"
#include "FE/Interfaces/LevelSetGeometrySnapshot.h"

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace svmp::FE::testing::quad_cut {

namespace FE = svmp::FE;
namespace interfaces = FE::interfaces;

inline FE::Real ownedRuleMeasure(const interfaces::LevelSetGeometrySnapshot& snapshot,
                          interfaces::LevelSetGeometryRuleRole role,
                          int boundary_marker = -1)
{
    FE::Real measure{0.0};
    for (const auto& record : snapshot.rules()) {
        if (!record.locally_owned ||
            record.retention != interfaces::LevelSetGeometryRetention::Retained ||
            record.role != role ||
            (boundary_marker >= 0 && record.physical_boundary_marker != boundary_marker)) {
            continue;
        }
        measure += record.physical_rule.physical_measure;
    }
    return measure;
}

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

inline interfaces::CutInterfaceDomainRequest interfaceRequest(int marker)
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

inline interfaces::LevelSetInterfaceDomain verticalInterface(
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

inline void addVerticalHalfVolumes(
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

inline interfaces::LevelSetInterfaceDomain verticalInterfaceWithVolumes(
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

inline interfaces::GeneratedInterfaceBoundaryIntersectionRequest contactRequest(
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

inline interfaces::GeneratedActiveBoundaryRequest activeRequest(
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

inline std::shared_ptr<const interfaces::LevelSetGeometrySnapshot> verticalCutSnapshot(
    const SingleQuadBoundaryMesh& mesh, int interface_marker, int wall_marker)
{
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

    interfaces::LevelSetGeometryScalarEvaluator scalar;
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
    return interfaces::buildLevelSetGeometrySnapshot(
        std::move(interface_domain),
        {std::move(contact_domain)},
        {std::move(negative), std::move(positive)},
        mesh,
        {},
        std::move(scalar),
        "sharp_boundary_test");
}

} // namespace svmp::FE::testing::quad_cut
