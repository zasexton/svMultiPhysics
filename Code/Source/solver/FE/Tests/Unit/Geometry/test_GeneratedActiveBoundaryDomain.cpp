#include "Assembly/CutIntegrationContext.h"
#include "Basis/NodeOrderingConventions.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <functional>
#include <memory>
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
                                        coordinates = {})
        : marker_(marker)
        , rank_(rank)
        , size_(size)
        , owner_rank_(owner_rank)
        , owned_(owned)
        , type_(type)
        , coordinates_(std::move(coordinates))
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

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override {
        return owned_ ? 1 : 0;
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 1; }
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
        return face == 0 ? FE::GlobalIndex{9001} : FE::INVALID_GLOBAL_INDEX;
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
        return face == 0 ? FE::LocalIndex{0} : FE::INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex face) const override {
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

interfaces::LevelSetInterfaceDomain verticalInterfaceWithVolumes(
    int marker,
    FE::Real negative_point_x = -0.5,
    FE::Real negative_weight = 2.0,
    int negative_achieved_order = 1)
{
    auto domain = verticalInterface(marker);
    const auto add_volume = [&domain,
                             negative_point_x,
                             negative_weight,
                             negative_achieved_order](
                                FE::geometry::CutIntegrationSide side,
                                FE::Real x,
                                FE::LocalIndex local_index) {
        if (side == FE::geometry::CutIntegrationSide::Negative) {
            x = negative_point_x;
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
        region.min_level_set_value = side == FE::geometry::CutIntegrationSide::Negative
                                         ? -1.0
                                         : 0.0;
        region.max_level_set_value = side == FE::geometry::CutIntegrationSide::Negative
                                         ? 0.0
                                         : 1.0;
        region.topology_id = side == FE::geometry::CutIntegrationSide::Negative
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

TEST(GeneratedActiveBoundaryDomain, CompletelyDrySideHasExactlyZeroRules)
{
    constexpr int interface_marker = 102;
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
    ASSERT_EQ(positive.fragments().size(), 1u);
    EXPECT_TRUE(positive.fragments().front().full_face_equivalent);
    EXPECT_NEAR(positive.summary().measure, 2.0, 1.0e-14);
    EXPECT_NO_THROW(interfaces::validateGeneratedActiveBoundaryPartition(
        negative, positive, interface_domain, contact_domain, mesh));
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
    for (const auto& record : snapshot->rules()) {
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
    EXPECT_EQ(context.freeSurfaceGeometrySnapshotRevisionForMarker(
                  interface_marker),
              snapshot->revision().snapshot_revision_key);
    for (const auto& active : snapshot->activeBoundaryDomains()) {
        EXPECT_EQ(context.freeSurfaceGeometrySnapshotRevisionForMarker(
                      active.marker()),
                  snapshot->revision().snapshot_revision_key);
    }
}

TEST(FreeSurfaceGeometrySnapshot, RejectsStoredOffInterfaceResidual)
{
    constexpr int interface_marker = 105;
    const SingleQuadBoundaryMesh mesh;
    auto domain = verticalInterface(interface_marker, 1.0e-4);
    EXPECT_THROW(
        (void)interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain),
            {},
            {},
            mesh,
            snapshotPolicyWithoutBoundary(),
            {},
            "bad_root"),
        std::invalid_argument);
}

TEST(FreeSurfaceGeometrySnapshot, RejectsStaleGlobalIdentityAndOwner)
{
    constexpr int interface_marker = 115;
    const SingleQuadBoundaryMesh mesh;
    const auto source = verticalInterface(interface_marker);

    auto stale_id_fragment = source.fragments().front();
    stale_id_fragment.parent_cell_global_id = 99;
    stale_id_fragment.stable_id = 0u;
    interfaces::LevelSetInterfaceDomain stale_id_domain(
        interfaceRequest(interface_marker));
    stale_id_domain.addFragment(std::move(stale_id_fragment));
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
    const auto owner_state =
        interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *owner_snapshot, parameters);
    EXPECT_GT(owner_state.owned_liquid_volume, 0.0);
    EXPECT_GT(owner_state.owned_liquid_gas_area, 0.0);

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
    EXPECT_EQ(ghost_snapshot->revision().snapshot_revision_key,
              owner_snapshot->revision().snapshot_revision_key);
    EXPECT_EQ(ghost_state.owned_liquid_volume, 0.0);
    EXPECT_EQ(ghost_state.owned_liquid_gas_area, 0.0);
    EXPECT_EQ(ghost_state.owned_wetted_wall_area, 0.0);
    EXPECT_EQ(ghost_state.owned_contact_measure, 0.0);
    EXPECT_EQ(ghost_state.total_potential, 0.0);
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
            /*negative_point_x=*/-0.25),
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
