#include "Assembly/CutIntegrationContext.h"
#include "Assembly/Assembler.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

class BoundaryOnlyMesh final : public assembly::IMeshAccess {
public:
    struct Face {
        GlobalIndex id{0};
        LocalIndex local_face{0};
        int marker{0};
        GlobalIndex global_id{INVALID_GLOBAL_INDEX};
    };

    BoundaryOnlyMesh(int dimension,
                     ElementType type,
                     std::vector<std::array<Real, 3>> coordinates,
                     std::vector<Face> faces,
                     bool owned = true,
                     GlobalIndex global_cell_id = 0,
                     int rank = 0,
                     int size = 1,
                     int owner_rank = -1)
        : dimension_(dimension)
        , type_(type)
        , coordinates_(std::move(coordinates))
        , faces_(std::move(faces))
        , owned_(owned)
        , global_cell_id_(global_cell_id)
        , rank_(rank)
        , size_(size)
        , owner_rank_(owner_rank >= 0 ? owner_rank : (owned ? rank : -1))
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        return owned_ ? 1 : 0;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return static_cast<GlobalIndex>(faces_.size());
    }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return dimension_; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] GlobalIndex getCellGlobalId(GlobalIndex cell_id) const override
    {
        return cell_id == 0 ? global_cell_id_ : INVALID_GLOBAL_INDEX;
    }
    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(
        GlobalIndex face_id) const override
    {
        for (const auto& face : faces_) {
            if (face.id == face_id) {
                return face.global_id != INVALID_GLOBAL_INDEX
                           ? face.global_id
                           : face.id;
            }
        }
        return INVALID_GLOBAL_INDEX;
    }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }
    [[nodiscard]] int getCellOwnerRank(GlobalIndex) const override
    {
        return owner_rank_;
    }
    [[nodiscard]] int getBoundaryFaceOwnerRank(
        GlobalIndex, GlobalIndex) const override
    {
        return owner_rank_;
    }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return owned_ && cell_id == 0;
    }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return type_;
    }
    void getCellNodes(GlobalIndex, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.resize(coordinates_.size());
        for (std::size_t i = 0; i < coordinates_.size(); ++i) {
            nodes[i] = static_cast<GlobalIndex>(i);
        }
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id)
        const override
    {
        return coordinates_.at(static_cast<std::size_t>(node_id));
    }
    void getCellCoordinates(
        GlobalIndex,
        std::vector<std::array<Real, 3>>& coords) const override
    {
        coords = coordinates_;
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex) const override
    {
        for (const auto& face : faces_) {
            if (face.id == face_id) {
                return face.local_face;
            }
        }
        return INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        for (const auto& face : faces_) {
            if (face.id == face_id) {
                return face.marker;
            }
        }
        return -1;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex) const override
    {
        return {-1, -1};
    }
    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        if (owned_) {
            callback(0);
        }
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        for (const auto& face : faces_) {
            if (marker < 0 || face.marker == marker) {
                callback(face.id, 0);
            }
        }
    }
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override
    {
    }

private:
    int dimension_{2};
    ElementType type_{ElementType::Quad4};
    std::vector<std::array<Real, 3>> coordinates_{};
    std::vector<Face> faces_{};
    bool owned_{true};
    GlobalIndex global_cell_id_{0};
    int rank_{0};
    int size_{1};
    int owner_rank_{0};
};

CutInterfaceDomainRequest interfaceRequest(int marker)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/4,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/9);
    request.interface_marker = marker;
    request.isovalue = 0.0;
    request.quadrature_order = 1;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_status = "None";
    return request;
}

GeneratedInterfaceBoundaryIntersectionRequest intersectionRequest(
    int interface_marker,
    int boundary_marker)
{
    GeneratedInterfaceBoundaryIntersectionRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/4,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/9);
    request.generated_domain_id = "test_domain";
    request.interface_marker = interface_marker;
    request.boundary_marker = boundary_marker;
    request.isovalue = 0.0;
    request.quadrature_order = 1;
    request.source_value_revision = 9u;
    return request;
}

} // namespace

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     StableMarkerKeysAreCanonicalAndUnambiguous)
{
    GeneratedInterfaceBoundaryIntersectionMarkerKey field_key;
    field_key.source =
        LevelSetInterfaceSource::fromField(
            /*field_id=*/8);
    field_key.domain_id = "contact";
    field_key.interface_marker = 10;
    field_key.boundary_marker = 20;

    GeneratedInterfaceBoundaryIntersectionMarkerKey evaluator_key =
        field_key;
    evaluator_key.source =
        LevelSetInterfaceSource::fromEvaluator(
            "field:8");
    EXPECT_NE(
        field_key.stableKey(),
        evaluator_key.stableKey());

    GeneratedInterfaceBoundaryIntersectionMarkerKey delimiter_left =
        field_key;
    delimiter_left.source =
        LevelSetInterfaceSource::fromEvaluator(
            "source|part");
    delimiter_left.domain_id = "domain";
    GeneratedInterfaceBoundaryIntersectionMarkerKey delimiter_right =
        delimiter_left;
    delimiter_right.source =
        LevelSetInterfaceSource::fromEvaluator(
            "source");
    delimiter_right.domain_id = "part|domain";
    EXPECT_NE(
        delimiter_left.stableKey(),
        delimiter_right.stableKey());

    GeneratedInterfaceBoundaryIntersectionMarkerKey fine_isovalue =
        field_key;
    fine_isovalue.isovalue = Real{1.0e-7};
    GeneratedInterfaceBoundaryIntersectionMarkerKey other_fine_isovalue =
        fine_isovalue;
    other_fine_isovalue.isovalue =
        Real{2.0e-7};
    EXPECT_NE(
        fine_isovalue.stableKey(),
        other_fine_isovalue.stableKey());

    GeneratedInterfaceBoundaryIntersectionMarkerKey negative_zero =
        field_key;
    negative_zero.isovalue = -Real{0.0};
    EXPECT_EQ(
        field_key.stableKey(),
        negative_zero.stableKey());
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain, Builds2DPointOnMarkedBoundaryEdge)
{
    constexpr int interface_marker = 101;
    constexpr int wall_marker = 7;
    BoundaryOnlyMesh mesh(
        2,
        ElementType::Quad4,
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{1.0, 1.0, 0.0}},
         {{0.0, 1.0, 0.0}}},
        {{0, 0, wall_marker}, {1, 1, 9}, {2, 2, 10}, {3, 3, 11}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    CutInterfaceFragment fragment;
    fragment.interface_marker = interface_marker;
    fragment.parent_cell = 0;
    fragment.kind = CutInterfaceFragmentKind::Segment;
    fragment.measure = 1.0;
    fragment.normal = {{1.0, 0.0, 0.0}};
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, -1.0, 0.0}},
                           .parent_coordinate = {{0.25, -1.0, 0.0}}});
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 1.0, 0.0}},
                           .parent_coordinate = {{0.25, 1.0, 0.0}}});
    fragment.quadrature_points.push_back(
        CutInterfaceQuadraturePoint{.point = {{0.25, 0.5, 0.0}},
                                    .normal = fragment.normal,
                                    .weight = 1.0});
    interface_domain.addFragment(std::move(fragment));

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, wall_marker), interface_domain, mesh);

    const auto summary = domain.summary();
    EXPECT_EQ(summary.boundary_marker, wall_marker);
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_DOUBLE_EQ(summary.measure, 1.0);
    ASSERT_EQ(domain.fragments().size(), 1u);
    EXPECT_EQ(domain.fragments().front().degeneracy,
              GeneratedInterfaceBoundaryIntersectionDegeneracy::None);
    const auto& point = domain.fragments().front().quadrature_points.front();
    EXPECT_NEAR(point.point[0], 0.25, 1.0e-12);
    EXPECT_NEAR(point.point[1], -1.0, 1.0e-12);
    EXPECT_NEAR(point.interface_normal[0], 1.0, 1.0e-12);
    EXPECT_NEAR(point.boundary_normal[1], -1.0, 1.0e-12);
    const auto rules = domain.intersectionQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().geometric_dimension, 0);
    EXPECT_EQ(rules.front().frame, geometry::CutGeometryFrame::Reference);
    const auto provenance = validateGeneratedInterfaceBoundaryProvenance(
        domain, interface_domain);
    EXPECT_EQ(provenance.active_contact_fragment_count, 1u);
    EXPECT_EQ(provenance.referenced_source_surface_fragment_count, 1u);
    EXPECT_EQ(provenance.orphan_contact_fragment_count, 0u);
    EXPECT_EQ(rules.front().provenance.source_stable_id,
              interface_domain.fragments().front().stable_id);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     GlobalIdentityIsIndependentOfRepartitionedLocalFaceNumbering)
{
    constexpr int interface_marker = 111;
    constexpr int wall_marker = 17;
    constexpr GlobalIndex global_cell = 7001;
    constexpr GlobalIndex global_face = 9003;
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    BoundaryOnlyMesh owner_mesh(
        2,
        ElementType::Quad4,
        coordinates,
        {{10, 0, wall_marker, global_face}},
        true,
        global_cell,
        0,
        2,
        0);
    BoundaryOnlyMesh repartitioned_mesh(
        2,
        ElementType::Quad4,
        coordinates,
        {{91, 0, wall_marker, global_face}},
        true,
        global_cell,
        1,
        2,
        1);

    const auto make_interface = [&](int owner_rank) {
        LevelSetInterfaceDomain domain(interfaceRequest(interface_marker));
        CutInterfaceFragment fragment;
        fragment.interface_marker = interface_marker;
        fragment.parent_cell = 0;
        fragment.parent_cell_global_id = global_cell;
        fragment.owner_rank = owner_rank;
        fragment.kind = CutInterfaceFragmentKind::Segment;
        fragment.measure = 1.0;
        fragment.normal = {{1.0, 0.0, 0.0}};
        fragment.vertices = {
            CutInterfaceVertex{
                .point = {{0.25, -1.0, 0.0}},
                .parent_coordinate = {{0.25, -1.0, 0.0}}},
            CutInterfaceVertex{
                .point = {{0.25, 1.0, 0.0}},
                .parent_coordinate = {{0.25, 1.0, 0.0}}}};
        fragment.quadrature_points.push_back(
            CutInterfaceQuadraturePoint{
                .point = {{0.25, 0.0, 0.0}},
                .normal = fragment.normal,
                .weight = 1.0});
        domain.addFragment(std::move(fragment));
        return domain;
    };

    const auto owner_interface = make_interface(0);
    const auto repartitioned_interface = make_interface(1);
    const auto owner_contact =
        buildGeneratedInterfaceBoundaryIntersectionDomain(
            intersectionRequest(interface_marker, wall_marker),
            owner_interface,
            owner_mesh);
    const auto repartitioned_contact =
        buildGeneratedInterfaceBoundaryIntersectionDomain(
            intersectionRequest(interface_marker, wall_marker),
            repartitioned_interface,
            repartitioned_mesh);

    ASSERT_EQ(owner_contact.fragments().size(), 1u);
    ASSERT_EQ(repartitioned_contact.fragments().size(), 1u);
    const auto& owner = owner_contact.fragments().front();
    const auto& repartitioned = repartitioned_contact.fragments().front();
    EXPECT_NE(owner.parent_face, repartitioned.parent_face);
    EXPECT_EQ(owner.parent_cell_global_id, global_cell);
    EXPECT_EQ(repartitioned.parent_cell_global_id, global_cell);
    EXPECT_EQ(owner.parent_face_global_id, global_face);
    EXPECT_EQ(repartitioned.parent_face_global_id, global_face);
    EXPECT_EQ(owner.owner_rank, 0);
    EXPECT_EQ(repartitioned.owner_rank, 1);
    EXPECT_EQ(owner.source_interface_stable_id,
              repartitioned.source_interface_stable_id);
    EXPECT_EQ(owner.stable_id, repartitioned.stable_id);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain, WrongBoundaryMarkerProducesZeroMeasure)
{
    constexpr int interface_marker = 101;
    BoundaryOnlyMesh mesh(
        2,
        ElementType::Quad4,
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{1.0, 1.0, 0.0}},
         {{0.0, 1.0, 0.0}}},
        {{0, 0, 7}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    CutInterfaceFragment fragment;
    fragment.interface_marker = interface_marker;
    fragment.parent_cell = 0;
    fragment.kind = CutInterfaceFragmentKind::Segment;
    fragment.measure = 1.0;
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, -1.0, 0.0}},
                           .parent_coordinate = {{0.25, -1.0, 0.0}}});
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 1.0, 0.0}},
                           .parent_coordinate = {{0.25, 1.0, 0.0}}});
    interface_domain.addFragment(std::move(fragment));

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, 99), interface_domain, mesh);

    EXPECT_EQ(domain.summary().active_fragment_count, 0u);
    EXPECT_DOUBLE_EQ(domain.summary().measure, 0.0);
    EXPECT_TRUE(domain.intersectionQuadratureRules().empty());
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain, Builds3DSegmentOnMarkedBoundaryFace)
{
    constexpr int interface_marker = 202;
    constexpr int wall_marker = 8;
    BoundaryOnlyMesh mesh(
        3,
        ElementType::Tetra4,
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{0.0, 1.0, 0.0}},
         {{0.0, 0.0, 1.0}}},
        {{1, 1, wall_marker}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    CutInterfaceFragment fragment;
    fragment.interface_marker = interface_marker;
    fragment.parent_cell = 0;
    fragment.kind = CutInterfaceFragmentKind::Polygon;
    fragment.measure = 0.5;
    fragment.normal = {{1.0, 0.0, 0.0}};
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 0.0, 0.0}},
                           .parent_coordinate = {{0.25, 0.0, 0.0}}});
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 0.0, 0.75}},
                           .parent_coordinate = {{0.25, 0.0, 0.75}}});
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 0.75, 0.0}},
                           .parent_coordinate = {{0.25, 0.75, 0.0}}});
    fragment.quadrature_points.push_back(
        CutInterfaceQuadraturePoint{.point = {{0.25, 0.25, 0.25}},
                                    .normal = fragment.normal,
                                    .weight = 0.5});
    interface_domain.addFragment(std::move(fragment));

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, wall_marker), interface_domain, mesh);

    const auto summary = domain.summary();
    ASSERT_EQ(summary.active_fragment_count, 1u);
    EXPECT_NEAR(summary.measure, 0.75, 1.0e-12);
    const auto rules = domain.intersectionQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    ASSERT_EQ(rules.front().points.size(), 1u);
    EXPECT_EQ(rules.front().geometric_dimension, 1);
    EXPECT_NEAR(rules.front().points.front().point[0], 0.25, 1.0e-12);
    EXPECT_NEAR(rules.front().points.front().point[1], 0.0, 1.0e-12);
    EXPECT_NEAR(rules.front().points.front().weight, 0.75, 1.0e-12);
    EXPECT_NEAR(rules.front().points.front().normal[0], 1.0, 1.0e-12);
    EXPECT_NEAR(rules.front().points.front().normal[1], 0.0, 1.0e-12);
    EXPECT_NEAR(rules.front().points.front().normal[2], 0.0, 1.0e-12);
    EXPECT_EQ(rules.front().provenance.implicit_geometry_mode,
              "LinearCorner");
    EXPECT_EQ(rules.front().provenance.selected_implicit_quadrature_backend,
              "LinearCorner");
    EXPECT_NEAR(rules.front().points.front().boundary_normal[1], -1.0, 1.0e-12);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     ScalarFieldCannotInventContactWithoutSourceSurfaceFragment)
{
    constexpr int interface_marker = 205;
    constexpr int wall_marker = 8;
    BoundaryOnlyMesh mesh(
        3,
        ElementType::Tetra4,
        {{{0.0, 0.0, 0.0}},
         {{2.0, 0.0, 0.0}},
         {{0.0, 1.0, 0.0}},
         {{0.0, 0.0, 4.0}}},
        {{1, 1, wall_marker}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    GeneratedInterfaceBoundaryIntersectionScalarField scalar_field;
    scalar_field.value_at_node = [](GlobalIndex node) -> Real {
        return node == 1 ? 1.5 : -0.5;
    };

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, wall_marker),
        interface_domain,
        mesh,
        scalar_field);

    EXPECT_EQ(domain.summary().active_fragment_count, 0u);
    EXPECT_TRUE(domain.fragments().empty());
    EXPECT_TRUE(domain.intersectionQuadratureRules().empty());
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain, ImportsRulesIntoCutContextByGeneratedMarker)
{
    constexpr int interface_marker = 203;
    constexpr int wall_marker = 8;
    BoundaryOnlyMesh mesh(
        2,
        ElementType::Quad4,
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{1.0, 1.0, 0.0}},
         {{0.0, 1.0, 0.0}}},
        {{0, 0, wall_marker}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    CutInterfaceFragment fragment;
    fragment.interface_marker = interface_marker;
    fragment.parent_cell = 0;
    fragment.kind = CutInterfaceFragmentKind::Segment;
    fragment.measure = 1.0;
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.5, -1.0, 0.0}},
                           .parent_coordinate = {{0.5, -1.0, 0.0}}});
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.5, 1.0, 0.0}},
                           .parent_coordinate = {{0.5, 1.0, 0.0}}});
    interface_domain.addFragment(std::move(fragment));

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, wall_marker), interface_domain, mesh);
    assembly::CutIntegrationContext context;
    context.addGeneratedInterfaceBoundaryIntersectionDomain(domain);

    EXPECT_TRUE(context.hasGeneratedInterfaceMarker(domain.marker()));
    EXPECT_EQ(context.interfaceRulesForMarker(domain.marker()).size(), 1u);
    EXPECT_TRUE(context.interfaceRulesForMarker(interface_marker).empty());
    const auto* stable_key =
        context.findGeneratedInterfaceBoundaryMarkerKey(
            domain.marker());
    ASSERT_NE(stable_key, nullptr);
    GeneratedInterfaceBoundaryIntersectionMarkerKey expected_key;
    expected_key.source = domain.request().source;
    expected_key.domain_id =
        domain.request().generated_domain_id;
    expected_key.isovalue =
        domain.request().isovalue;
    expected_key.interface_marker =
        domain.request().interface_marker;
    expected_key.boundary_marker =
        domain.request().boundary_marker;
    EXPECT_EQ(
        *stable_key,
        expected_key.stableKey());
    EXPECT_EQ(
        context.findGeneratedInterfaceBoundaryMarkerKey(
            interface_marker),
        nullptr);
    const auto* publication_provenance =
        context
            .findGeneratedInterfaceBoundaryPublicationProvenance(
                domain.marker());
    ASSERT_NE(publication_provenance, nullptr);
    EXPECT_EQ(
        publication_provenance
            ->generated_interface_boundary_marker,
        domain.marker());
    EXPECT_EQ(
        publication_provenance
            ->stable_owner_key,
        expected_key.stableKey());
    EXPECT_EQ(
        publication_provenance
            ->request
            .generated_domain_id,
        domain.request().generated_domain_id);
    EXPECT_EQ(
        publication_provenance
            ->request
            .source
            .layout_revision,
        domain.request().source.layout_revision);
    EXPECT_EQ(
        publication_provenance
            ->request
            .source_value_revision,
        domain.request().source_value_revision);
    EXPECT_EQ(
        publication_provenance
            ->request
            .quadrature_order,
        domain.request().quadrature_order);
    context.clear();
    EXPECT_EQ(
        context
            .findGeneratedInterfaceBoundaryPublicationProvenance(
                domain.marker()),
        nullptr);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     RestrictsAuthoritativeFragmentOnHighOrderParentGeometry)
{
    constexpr int interface_marker = 204;
    constexpr int wall_marker = 8;
    BoundaryOnlyMesh mesh(
        3,
        ElementType::Tetra10,
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{0.0, 1.0, 0.0}},
         {{0.0, 0.0, 1.0}}},
        {{1, 1, wall_marker}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    CutInterfaceFragment fragment;
    fragment.interface_marker = interface_marker;
    fragment.parent_cell = 0;
    fragment.kind = CutInterfaceFragmentKind::Polygon;
    fragment.measure = 0.5;
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 0.0, 0.0}},
                           .parent_coordinate = {{0.25, 0.0, 0.0}}});
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 0.0, 0.75}},
                           .parent_coordinate = {{0.25, 0.0, 0.75}}});
    fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 0.75, 0.0}},
                           .parent_coordinate = {{0.25, 0.75, 0.0}}});
    interface_domain.addFragment(std::move(fragment));

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, wall_marker), interface_domain, mesh);

    const auto summary = domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.skipped_fragment_count, 0u);
    ASSERT_EQ(domain.fragments().size(), 1u);
    EXPECT_EQ(domain.fragments().front().degeneracy,
              GeneratedInterfaceBoundaryIntersectionDegeneracy::None);
    EXPECT_NEAR(domain.fragments().front().measure, 0.75, 1.0e-14);
    EXPECT_EQ(domain.fragments().front().source_interface_stable_id,
              interface_domain.fragments().front().stable_id);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     ScalarFieldCannotInventHighOrderContactWithoutSourceFragment)
{
    constexpr int interface_marker = 206;
    constexpr int wall_marker = 8;
    BoundaryOnlyMesh mesh(
        3,
        ElementType::Tetra10,
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{0.0, 1.0, 0.0}},
         {{0.0, 0.0, 1.0}}},
        {{1, 1, wall_marker}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    GeneratedInterfaceBoundaryIntersectionScalarField scalar_field;
    scalar_field.value_at_node = [](GlobalIndex node) -> Real {
        return node == 1 ? 1.0 : -1.0;
    };

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, wall_marker),
        interface_domain,
        mesh,
        scalar_field);

    EXPECT_EQ(domain.summary().active_fragment_count, 0u);
    EXPECT_TRUE(domain.fragments().empty());
    EXPECT_NO_THROW(validateGeneratedInterfaceBoundaryProvenance(
        domain, interface_domain));
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     ScalarFieldCannotInventAmbiguousContactWithoutSourceFragment)
{
    constexpr int interface_marker = 207;
    constexpr int wall_marker = 12;
    BoundaryOnlyMesh mesh(
        3,
        ElementType::Hex8,
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{1.0, 1.0, 0.0}},
         {{0.0, 1.0, 0.0}},
         {{0.0, 0.0, 1.0}},
         {{1.0, 0.0, 1.0}},
         {{1.0, 1.0, 1.0}},
         {{0.0, 1.0, 1.0}}},
        {{0, 0, wall_marker}});

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    GeneratedInterfaceBoundaryIntersectionScalarField scalar_field;
    scalar_field.value_at_node = [](GlobalIndex node) -> Real {
        static constexpr std::array<Real, 8> values{
            1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0};
        return values.at(static_cast<std::size_t>(node));
    };

    const auto domain = buildGeneratedInterfaceBoundaryIntersectionDomain(
        intersectionRequest(interface_marker, wall_marker),
        interface_domain,
        mesh,
        scalar_field);

    const auto summary = domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 0u);
    EXPECT_EQ(summary.ambiguous_topology_count, 0u);
    EXPECT_TRUE(domain.fragments().empty());
    EXPECT_TRUE(domain.intersectionQuadratureRules().empty());
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     ProvenanceValidatorRejectsOrphanContactFragment)
{
    constexpr int interface_marker = 212;
    auto request = intersectionRequest(interface_marker, 24);
    GeneratedInterfaceBoundaryIntersectionDomain contact_domain(request);
    GeneratedInterfaceBoundaryIntersectionFragment contact;
    contact.parent_cell = 0;
    contact.parent_face = 0;
    contact.source_interface_stable_id = 999999u;
    contact.kind = GeneratedInterfaceBoundaryIntersectionKind::Point;
    contact.measure = 1.0;
    contact.quadrature_points.push_back(
        GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
            .point = {{0.0, 0.0, 0.0}},
            .parent_coordinate = {{0.0, 0.0, 0.0}},
            .interface_normal = {{1.0, 0.0, 0.0}},
            .boundary_normal = {{0.0, 1.0, 0.0}},
            .tangent = {{1.0, 0.0, 0.0}},
            .weight = 1.0});
    contact_domain.addFragment(std::move(contact));

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    CutInterfaceFragment source;
    source.interface_marker = interface_marker;
    source.parent_cell = 0;
    source.kind = CutInterfaceFragmentKind::Segment;
    source.measure = 1.0;
    source.normal = {{1.0, 0.0, 0.0}};
    source.quadrature_points.push_back(
        CutInterfaceQuadraturePoint{.point = {{0.0, 0.0, 0.0}},
                                    .normal = source.normal,
                                    .weight = 1.0});
    interface_domain.addFragment(std::move(source));

    EXPECT_THROW(validateGeneratedInterfaceBoundaryProvenance(
                     contact_domain, interface_domain),
                 std::invalid_argument);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     RejectsCurrentFrameToPreventMixedGeometryContracts)
{
    auto request = intersectionRequest(/*interface_marker=*/208,
                                       /*boundary_marker=*/13);
    request.frame = geometry::CutGeometryFrame::Current;
    EXPECT_FALSE(request.valid());
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     ReportsLinearCornerQuadratureOrderHonestly)
{
    auto request = intersectionRequest(/*interface_marker=*/209,
                                       /*boundary_marker=*/14);
    request.intersection_marker = 909;
    request.quadrature_order = 12;
    GeneratedInterfaceBoundaryIntersectionDomain domain(request);
    GeneratedInterfaceBoundaryIntersectionFragment fragment;
    fragment.parent_cell = 0;
    fragment.parent_face = 1;
    fragment.kind = GeneratedInterfaceBoundaryIntersectionKind::Segment;
    fragment.measure = 0.5;
    fragment.tangent = {{1.0, 0.0, 0.0}};
    fragment.quadrature_points.push_back(
        GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
            .point = {{0.1, 0.0, 0.0}},
            .parent_coordinate = {{0.1, 0.0, 0.0}},
            .tangent = fragment.tangent,
            .weight = 5.0 / 36.0});
    fragment.quadrature_points.push_back(
        GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
            .point = {{0.25, 0.0, 0.0}},
            .parent_coordinate = {{0.25, 0.0, 0.0}},
            .tangent = fragment.tangent,
            .weight = 2.0 / 9.0});
    fragment.quadrature_points.push_back(
        GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
            .point = {{0.4, 0.0, 0.0}},
            .parent_coordinate = {{0.4, 0.0, 0.0}},
            .tangent = fragment.tangent,
            .weight = 5.0 / 36.0});
    domain.addFragment(std::move(fragment));

    const auto rules = domain.intersectionQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().provenance.requested_quadrature_order, 12);
    EXPECT_EQ(rules.front().provenance.achieved_quadrature_order, 5);
    EXPECT_EQ(rules.front().exact_polynomial_order, 5);
    EXPECT_EQ(rules.front().geometric_dimension, 1);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     CutContextRejectsGeneratedMarkerCollisions)
{
    auto make_domain = [](int boundary_marker) {
        auto request = intersectionRequest(/*interface_marker=*/210,
                                           boundary_marker);
        request.intersection_marker = 777;
        GeneratedInterfaceBoundaryIntersectionDomain domain(request);
        GeneratedInterfaceBoundaryIntersectionFragment fragment;
        fragment.parent_cell = 0;
        fragment.parent_face = boundary_marker;
        fragment.kind = GeneratedInterfaceBoundaryIntersectionKind::Point;
        fragment.measure = 1.0;
        fragment.quadrature_points.push_back(
            GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
                .point = {{0.0, 0.0, 0.0}},
                .parent_coordinate = {{0.0, 0.0, 0.0}},
                .weight = 1.0});
        domain.addFragment(std::move(fragment));
        return domain;
    };

    assembly::CutIntegrationContext context;
    const auto first = make_domain(21);
    const auto second = make_domain(22);
    context.addGeneratedInterfaceBoundaryIntersectionDomain(first);
    EXPECT_THROW(
        context.addGeneratedInterfaceBoundaryIntersectionDomain(second),
        std::invalid_argument);
}

TEST(GeneratedInterfaceBoundaryIntersectionDomain,
     OwnedAndGhostViewsYieldOneGlobalPointMultiplicity)
{
    constexpr int interface_marker = 211;
    constexpr int wall_marker = 23;
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    const std::vector<BoundaryOnlyMesh::Face> duplicate_faces{
        {0, 0, wall_marker}, {0, 0, wall_marker}};
    BoundaryOnlyMesh owner_mesh(
        2, ElementType::Quad4, coordinates, duplicate_faces, true);
    BoundaryOnlyMesh ghost_mesh(
        2, ElementType::Quad4, coordinates, duplicate_faces, false);

    LevelSetInterfaceDomain interface_domain(interfaceRequest(interface_marker));
    CutInterfaceFragment interface_fragment;
    interface_fragment.interface_marker = interface_marker;
    interface_fragment.parent_cell = 0;
    interface_fragment.kind = CutInterfaceFragmentKind::Segment;
    interface_fragment.measure = 2.0;
    interface_fragment.normal = {{1.0, 0.0, 0.0}};
    interface_fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, -1.0, 0.0}},
                           .parent_coordinate = {{0.25, -1.0, 0.0}}});
    interface_fragment.vertices.push_back(
        CutInterfaceVertex{.point = {{0.25, 1.0, 0.0}},
                           .parent_coordinate = {{0.25, 1.0, 0.0}}});
    interface_fragment.quadrature_points.push_back(
        CutInterfaceQuadraturePoint{.point = {{0.25, 0.0, 0.0}},
                                    .parent_coordinate = {{0.25, 0.0, 0.0}},
                                    .weight = 2.0});
    interface_domain.addFragment(std::move(interface_fragment));

    const auto request = intersectionRequest(interface_marker, wall_marker);
    const auto owner = buildGeneratedInterfaceBoundaryIntersectionDomain(
        request, interface_domain, owner_mesh);
    const auto ghost = buildGeneratedInterfaceBoundaryIntersectionDomain(
        request, interface_domain, ghost_mesh);

    const auto owner_summary = owner.summary();
    const auto ghost_summary = ghost.summary();
    EXPECT_EQ(owner_summary.active_fragment_count, 1u);
    EXPECT_NEAR(owner_summary.measure, 1.0, 1.0e-12);
    EXPECT_EQ(ghost_summary.active_fragment_count, 0u);
    EXPECT_NEAR(ghost_summary.measure, 0.0, 1.0e-12);
    EXPECT_NEAR(owner_summary.measure + ghost_summary.measure,
                1.0,
                1.0e-12);
}
