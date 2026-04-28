#include "Geometry/CutQuadrature.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

namespace {

constexpr Real kPi = 3.141592653589793238462643383279502884;

Real sphere_volume(Real radius)
{
    return (4.0 / 3.0) * kPi * radius * radius * radius;
}

Real spherical_cap_volume(Real radius, Real height)
{
    return kPi * height * height * (radius - height / 3.0);
}

Real spherical_cap_surface_area(Real radius, Real height)
{
    return 2.0 * kPi * radius * height;
}

Real circular_disk_area(Real radius)
{
    return kPi * radius * radius;
}

} // namespace

TEST(CutQuadrature, AxisAlignedBoxVolumeAndInterfaceMeasuresAreExactForConstants)
{
    const std::array<Real, 3> lo{{0.0, 0.0, 0.0}};
    const std::array<Real, 3> hi{{1.0, 1.0, 1.0}};

    const auto negative = makeAxisAlignedBoxCutVolumeQuadrature(
        lo, hi, 0, 0.25, CutIntegrationSide::Negative, "cut-box");
    ASSERT_EQ(negative.points.size(), 1u);
    EXPECT_DOUBLE_EQ(negative.measure, 0.25);
    EXPECT_DOUBLE_EQ(negative.volume_fraction, 0.25);
    EXPECT_DOUBLE_EQ(negative.points[0].weight, 0.25);
    EXPECT_DOUBLE_EQ(negative.points[0].point[0], 0.125);

    const auto positive = makeAxisAlignedBoxCutVolumeQuadrature(
        lo, hi, 0, 0.25, CutIntegrationSide::Positive, "cut-box");
    EXPECT_DOUBLE_EQ(positive.measure, 0.75);
    EXPECT_DOUBLE_EQ(positive.volume_fraction, 0.75);

    const auto iface = makeAxisAlignedBoxCutInterfaceQuadrature(lo, hi, 0, 0.25, "cut-box");
    ASSERT_EQ(iface.points.size(), 1u);
    EXPECT_DOUBLE_EQ(iface.measure, 1.0);
    EXPECT_DOUBLE_EQ(iface.points[0].point[0], 0.25);
}

TEST(CutQuadrature, SegmentCutFaceQuadraturePreservesLengthFractions)
{
    const auto rule = makeSegmentCutFaceQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.25, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        CutIntegrationSide::Negative,
        "cut-segment");

    ASSERT_EQ(rule.points.size(), 1u);
    EXPECT_DOUBLE_EQ(rule.measure, 0.25);
    EXPECT_DOUBLE_EQ(rule.volume_fraction, 0.25);
    EXPECT_DOUBLE_EQ(rule.points[0].point[0], 0.125);
}

TEST(CutQuadrature, SegmentCutFaceQuadratureHandlesNoCrossingTangentAndZeroLengthCases)
{
    const auto negative_full = makeSegmentCutFaceQuadrature(
        {{-2.0, 0.0, 0.0}},
        {{-1.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        CutIntegrationSide::Negative);
    EXPECT_DOUBLE_EQ(negative_full.measure, 1.0);
    ASSERT_EQ(negative_full.points.size(), 1u);
    EXPECT_DOUBLE_EQ(negative_full.points[0].point[0], -1.5);

    const auto negative_empty = makeSegmentCutFaceQuadrature(
        {{1.0, 0.0, 0.0}},
        {{2.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        CutIntegrationSide::Negative);
    EXPECT_DOUBLE_EQ(negative_empty.measure, 0.0);
    EXPECT_TRUE(negative_empty.points.empty());

    const auto positive_full = makeSegmentCutFaceQuadrature(
        {{1.0, 0.0, 0.0}},
        {{2.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        CutIntegrationSide::Positive);
    EXPECT_DOUBLE_EQ(positive_full.measure, 1.0);

    const auto tangent = makeSegmentCutFaceQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        CutIntegrationSide::Negative);
    EXPECT_DOUBLE_EQ(tangent.measure, 0.0);

    const auto zero_length = makeSegmentCutFaceQuadrature(
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        CutIntegrationSide::Negative);
    EXPECT_DOUBLE_EQ(zero_length.parent_measure, 0.0);
    EXPECT_DOUBLE_EQ(zero_length.measure, 0.0);
}

TEST(CutQuadrature, AxisAlignedBoxCutsHandleOutsideAndBoundaryPlanes)
{
    const std::array<Real, 3> lo{{0.0, 0.0, 0.0}};
    const std::array<Real, 3> hi{{1.0, 1.0, 1.0}};

    const auto negative_before = makeAxisAlignedBoxCutVolumeQuadrature(
        lo, hi, 0, -0.5, CutIntegrationSide::Negative);
    EXPECT_DOUBLE_EQ(negative_before.measure, 0.0);
    EXPECT_TRUE(negative_before.points.empty());

    const auto positive_before = makeAxisAlignedBoxCutVolumeQuadrature(
        lo, hi, 0, -0.5, CutIntegrationSide::Positive);
    EXPECT_DOUBLE_EQ(positive_before.measure, 1.0);

    const auto negative_at_boundary = makeAxisAlignedBoxCutVolumeQuadrature(
        lo, hi, 0, 0.0, CutIntegrationSide::Negative);
    EXPECT_DOUBLE_EQ(negative_at_boundary.measure, 0.0);

    const auto iface_outside = makeAxisAlignedBoxCutInterfaceQuadrature(lo, hi, 0, 2.0);
    EXPECT_DOUBLE_EQ(iface_outside.measure, 0.0);
    EXPECT_TRUE(iface_outside.points.empty());

    const auto iface_boundary = makeAxisAlignedBoxCutInterfaceQuadrature(lo, hi, 0, 1.0);
    EXPECT_DOUBLE_EQ(iface_boundary.measure, 1.0);
    ASSERT_EQ(iface_boundary.points.size(), 1u);
    EXPECT_DOUBLE_EQ(iface_boundary.points[0].point[0], 1.0);
}

TEST(CutQuadrature, PolygonAndConservativeSplitRulesCarryPolicyAndProvenance)
{
    CutQuadratureConstructionPolicy policy;
    policy.polynomial_order = 1;
    policy.name = "linear-moment-fit";
    CutQuadratureProvenance provenance;
    provenance.embedded_geometry_id = "embedded-plane";
    provenance.cut_topology_id = "poly-1";
    provenance.cut_topology_revision = 44;
    provenance.predicate_policy_key = 55;

    const auto polygon = makePolygonInterfaceQuadrature(
        {{{0.0, 0.0, 0.0}},
         {{1.0, 0.0, 0.0}},
         {{1.0, 1.0, 0.0}},
         {{0.0, 1.0, 0.0}}},
        {{0.0, 0.0, 1.0}},
        policy,
        provenance);
    EXPECT_DOUBLE_EQ(polygon.measure, 1.0);
    EXPECT_EQ(polygon.exact_polynomial_order, 1);
    EXPECT_EQ(polygon.provenance.cut_topology_revision, 44u);

    const auto split = makeConservativeSplitCutVolumeQuadrature(
        2.0,
        0.25,
        {{0.5, 0.5, 0.5}},
        {{1.0, 0.0, 0.0}},
        policy,
        provenance);
    EXPECT_DOUBLE_EQ(split[0].measure, 0.5);
    EXPECT_DOUBLE_EQ(split[1].measure, 1.5);
    const auto conservation = checkCutVolumeConservation(split[0], split[1]);
    EXPECT_TRUE(conservation.ok);
    EXPECT_DOUBLE_EQ(conservation.residual, 0.0);
}

TEST(CutQuadrature, TopologyDerivedRulesConserveLinearCutMeasuresForAdvertisedFamilies)
{
    struct Case {
        const char* name;
        Real parent_measure;
        Real negative_fraction;
        std::vector<std::array<Real, 3>> polygon;
        std::array<Real, 3> normal;
        Real interface_measure;
    };
    const std::array<Case, 4> cases{{
        {"tri", 0.5, 0.25,
         {{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}},
         {{0.0, 0.0, 1.0}},
         0.5},
        {"quad", 1.0, 0.5,
         {{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{1.0, 1.0, 0.0}}, {{0.0, 1.0, 0.0}}},
         {{0.0, 0.0, 1.0}},
         1.0},
        {"tet", 1.0 / 6.0, 0.125,
         {{{0.0, 0.0, 0.0}}, {{0.5, 0.0, 0.0}}, {{0.0, 0.5, 0.0}}},
         {{0.0, 0.0, 1.0}},
         0.125},
        {"hex", 1.0, 0.25,
         {{{0.25, 0.0, 0.0}}, {{0.25, 1.0, 0.0}}, {{0.25, 1.0, 1.0}}, {{0.25, 0.0, 1.0}}},
         {{1.0, 0.0, 0.0}},
         1.0},
    }};

    for (const auto& c : cases) {
        CutTopologyQuadratureInput input;
        input.parent_measure = c.parent_measure;
        input.negative_fraction = c.negative_fraction;
        input.representative_point = {{0.25, 0.25, 0.25}};
        input.interface_normal = c.normal;
        input.ordered_interface_points = c.polygon;
        input.provenance.embedded_geometry_id = c.name;
        input.provenance.cut_topology_revision = 42;

        const auto rules = makeTopologyDerivedCutQuadrature(input);
        EXPECT_TRUE(rules.conservation.ok) << c.name;
        EXPECT_DOUBLE_EQ(rules.negative_volume.measure, c.parent_measure * c.negative_fraction);
        EXPECT_DOUBLE_EQ(rules.positive_volume.measure, c.parent_measure * (1.0 - c.negative_fraction));
        EXPECT_NEAR(rules.interface_rule.measure, c.interface_measure, 1.0e-12) << c.name;
        EXPECT_EQ(rules.negative_volume.policy.kind, CutQuadratureConstructionKind::TopologySubdivision);
        EXPECT_EQ(rules.interface_rule.provenance.cut_topology_revision, 42u);
    }
}

TEST(CutQuadrature, ClosedTopologyRulesUseSubcellMeasuresForFamilyNeutralQuadrature)
{
    CutClosedTopologyQuadratureInput input;
    input.parent_measure = 1.0;
    input.interface_normal = {{1.0, 0.0, 0.0}};
    input.ordered_interface_points = {
        {{0.5, 0.0, 0.0}},
        {{0.5, 1.0, 0.0}},
        {{0.5, 1.0, 1.0}},
        {{0.5, 0.0, 1.0}}};
    input.provenance.embedded_geometry_id = "closed-topology-plane";
    input.provenance.cut_topology_id = "hex-subcells";
    input.provenance.cut_topology_revision = 81;
    input.policy.polynomial_order = 0;

    input.negative_subcells.push_back({
        "neg-wedge-piece",
        {{{0.0, 0.0, 0.0}}, {{0.5, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}},
        0.25,
        {{0.125, 0.25, 0.25}}});
    input.negative_subcells.push_back({
        "neg-pyramid-piece",
        {{{0.5, 0.0, 0.0}}, {{0.5, 1.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}},
        0.25,
        {{0.25, 0.5, 0.25}}});
    input.positive_subcells.push_back({
        "pos-polyhedron-piece",
        {{{0.5, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{1.0, 1.0, 0.0}}, {{0.5, 1.0, 0.0}}},
        0.50,
        {{0.75, 0.5, 0.5}}});

    const auto rules = makeClosedTopologyCutQuadrature(input);
    EXPECT_TRUE(rules.conservation.ok);
    EXPECT_DOUBLE_EQ(rules.negative_volume.measure, 0.5);
    EXPECT_DOUBLE_EQ(rules.positive_volume.measure, 0.5);
    ASSERT_EQ(rules.negative_volume.points.size(), 2u);
    ASSERT_EQ(rules.positive_volume.points.size(), 1u);
    EXPECT_DOUBLE_EQ(rules.negative_volume.points[0].weight, 0.25);
    EXPECT_DOUBLE_EQ(rules.positive_volume.points[0].point[0], 0.75);
    EXPECT_EQ(rules.negative_volume.policy.kind, CutQuadratureConstructionKind::TopologySubdivision);
    EXPECT_EQ(rules.negative_volume.provenance.cut_topology_revision, 81u);
    EXPECT_NEAR(rules.interface_rule.measure, 1.0, 1.0e-12);
}

TEST(CutQuadrature, ClosedTopologyRulesPreserveCurvedIsoparametricSubcellAndInterfaceData)
{
    CutClosedTopologyQuadratureInput input;
    input.parent_measure = 1.0;
    input.curved_geometry = true;
    input.frame = CutGeometryFrame::Current;
    input.interface_normal = {{0.0, 0.0, 1.0}};
    input.provenance.embedded_geometry_id = "curved-topology-plane";
    input.provenance.cut_topology_id = "quadratic-triangle-slice";
    input.provenance.cut_topology_revision = 128;
    input.policy.polynomial_order = 2;

    input.negative_subcells.push_back({
        "curved-neg-triangle",
        {{{0.0, 0.0, 0.0}}, {{0.5, 0.0, 0.0}}, {{0.0, 0.5, 0.0}}},
        0.125,
        {{0.2, 0.15, 0.01}},
        true,
        true});
    input.positive_subcells.push_back({
        "curved-pos-triangle",
        {{{0.5, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}},
        0.875,
        {{0.55, 0.25, 0.02}},
        true,
        true});
    input.curved_interface_points = {
        CutQuadraturePoint{{{0.25, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}, 0.2},
        CutQuadraturePoint{{{0.25, 0.5, 0.02}}, {{0.0, 0.0, 1.0}}, 0.3}};

    const auto rules = makeClosedTopologyCutQuadrature(input);
    EXPECT_TRUE(rules.conservation.ok);
    EXPECT_TRUE(rules.negative_volume.curved_geometry);
    EXPECT_TRUE(rules.interface_rule.curved_geometry);
    EXPECT_EQ(rules.negative_volume.policy.kind,
              CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(rules.interface_rule.policy.kind,
              CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(rules.interface_rule.frame, CutGeometryFrame::Current);
    EXPECT_DOUBLE_EQ(rules.interface_rule.measure, 0.5);
    ASSERT_EQ(rules.interface_rule.points.size(), 2u);
    EXPECT_EQ(rules.negative_volume.provenance.cut_topology_revision, 128u);
}

TEST(CutQuadrature, CurvedAndMomentFittedRulesCarryFrameAndExactnessMetadata)
{
    CutQuadratureProvenance provenance;
    provenance.embedded_geometry_id = "sphere-cut";
    provenance.cut_topology_id = "curved-face";
    provenance.frame = CutGeometryFrame::Current;

    CutQuadratureConstructionPolicy policy;
    policy.polynomial_order = 2;
    policy.name = "curved-surface-rule";

    const auto curved = makeCurvedInterfaceQuadrature(
        {{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 2.0}}, 0.25},
         {{{1.0, 0.0, 0.0}}, {{0.0, 0.0, 2.0}}, 0.25},
         {{{0.0, 1.0, 0.0}}, {{0.0, 0.0, 2.0}}, 0.50}},
        CutGeometryFrame::Current,
        policy,
        provenance);
    EXPECT_TRUE(curved.curved_geometry);
    EXPECT_EQ(curved.frame, CutGeometryFrame::Current);
    EXPECT_DOUBLE_EQ(curved.measure, 1.0);
    EXPECT_EQ(curved.exact_polynomial_order, 2);
    ASSERT_EQ(curved.points.size(), 3u);
    EXPECT_DOUBLE_EQ(curved.points[0].normal[2], 1.0);

    const auto moment = makeMomentFittedCutVolumeQuadrature(
        4.0,
        0.25,
        CutIntegrationSide::Positive,
        {{1.0, 2.0, 3.0}},
        {{1.0, 0.0, 0.0}},
        3,
        provenance);
    EXPECT_DOUBLE_EQ(moment.measure, 1.0);
    EXPECT_TRUE(moment.policy.moment_fitted);
    EXPECT_EQ(moment.policy.kind, CutQuadratureConstructionKind::MomentFittedImplicit);
    EXPECT_EQ(moment.exact_polynomial_order, 3);
    EXPECT_EQ(moment.frame, CutGeometryFrame::Current);
}

TEST(CutQuadrature, SphericalCapReferencesValidateCurvedAndMomentFittedRules)
{
    const Real radius = 2.0;
    const Real cut_coordinate = 0.5;
    const Real cap_height = radius - cut_coordinate;
    const Real disk_radius = std::sqrt(radius * radius - cut_coordinate * cut_coordinate);
    const Real total_volume = sphere_volume(radius);
    const Real cap_volume = spherical_cap_volume(radius, cap_height);
    const Real complement_volume = total_volume - cap_volume;
    const Real cap_fraction = cap_volume / total_volume;
    const Real complement_fraction = complement_volume / total_volume;
    const Real cap_area = spherical_cap_surface_area(radius, cap_height);
    const Real disk_area = circular_disk_area(disk_radius);

    CutQuadratureProvenance provenance;
    provenance.embedded_geometry_id = "analytic-sphere";
    provenance.cut_topology_id = "sphere-plane-cap";
    provenance.cut_topology_revision = 812;
    provenance.predicate_policy_key = 144;
    provenance.frame = CutGeometryFrame::Current;

    const auto positive_cap = makeMomentFittedCutVolumeQuadrature(
        total_volume,
        cap_fraction,
        CutIntegrationSide::Positive,
        {{cut_coordinate + 0.5 * cap_height, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        2,
        provenance);
    const auto negative_complement = makeMomentFittedCutVolumeQuadrature(
        total_volume,
        complement_fraction,
        CutIntegrationSide::Negative,
        {{-0.25 * radius, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        2,
        provenance);

    EXPECT_FALSE(positive_cap.curved_geometry);
    EXPECT_TRUE(positive_cap.policy.moment_fitted);
    EXPECT_EQ(positive_cap.policy.kind, CutQuadratureConstructionKind::MomentFittedImplicit);
    EXPECT_EQ(positive_cap.frame, CutGeometryFrame::Current);
    EXPECT_EQ(positive_cap.provenance.cut_topology_revision, 812u);
    EXPECT_NEAR(positive_cap.measure, cap_volume, 1.0e-12);
    EXPECT_NEAR(positive_cap.volume_fraction, cap_fraction, 1.0e-14);
    EXPECT_NEAR(negative_complement.measure, complement_volume, 1.0e-12);
    EXPECT_NEAR(negative_complement.volume_fraction, complement_fraction, 1.0e-14);

    const auto conservation = checkCutVolumeConservation(
        negative_complement,
        positive_cap,
        1.0e-12);
    EXPECT_TRUE(conservation.ok);
    EXPECT_NEAR(conservation.residual, 0.0, 1.0e-12);

    CutQuadratureConstructionPolicy curved_policy;
    curved_policy.kind = CutQuadratureConstructionKind::CurvedInterface;
    curved_policy.polynomial_order = 2;
    curved_policy.name = "analytic-spherical-reference";

    const auto cap_surface = makeCurvedInterfaceQuadrature(
        {{{{cut_coordinate + 0.5 * cap_height, 0.0, 0.0}},
          {{2.0, 0.0, 0.0}},
          cap_area}},
        CutGeometryFrame::Current,
        curved_policy,
        provenance);
    EXPECT_TRUE(cap_surface.curved_geometry);
    EXPECT_EQ(cap_surface.policy.kind, CutQuadratureConstructionKind::CurvedInterface);
    EXPECT_EQ(cap_surface.frame, CutGeometryFrame::Current);
    EXPECT_EQ(cap_surface.exact_polynomial_order, 2);
    EXPECT_NEAR(cap_surface.measure, cap_area, 1.0e-12);
    ASSERT_EQ(cap_surface.points.size(), 1u);
    EXPECT_NEAR(cap_surface.points[0].weight, cap_area, 1.0e-12);
    EXPECT_NEAR(cap_surface.points[0].normal[0], 1.0, 1.0e-14);

    const auto disk_interface = makeCurvedInterfaceQuadrature(
        {{{{cut_coordinate, 0.0, 0.0}},
          {{1.0, 0.0, 0.0}},
          disk_area}},
        CutGeometryFrame::Current,
        curved_policy,
        provenance);
    EXPECT_NEAR(disk_interface.measure, disk_area, 1.0e-12);
    EXPECT_NEAR(disk_interface.points.front().weight, disk_area, 1.0e-12);
    EXPECT_EQ(disk_interface.provenance.predicate_policy_key, 144u);
}

TEST(CutQuadrature, CurvedReferenceDiagnosticsRejectDegenerateAnalyticRules)
{
    CutQuadratureProvenance provenance;
    provenance.embedded_geometry_id = "degenerate-sphere-reference";
    provenance.cut_topology_id = "zero-height-cap";
    provenance.frame = CutGeometryFrame::Reference;

    CutQuadratureConstructionPolicy policy;
    policy.kind = CutQuadratureConstructionKind::CurvedInterface;
    policy.polynomial_order = 2;

    const auto degenerate_cap = makeCurvedInterfaceQuadrature(
        {{{{1.0, 0.0, 0.0}},
          {{1.0, 0.0, 0.0}},
          0.0}},
        CutGeometryFrame::Reference,
        policy,
        provenance);
    EXPECT_TRUE(degenerate_cap.points.empty());
    EXPECT_DOUBLE_EQ(degenerate_cap.measure, 0.0);

    CutQuadratureValidityPolicy validity;
    validity.min_measure = 1.0e-10;
    const auto diagnostic = diagnoseCutQuadrature(degenerate_cap, validity);
    EXPECT_FALSE(diagnostic.ok);
    EXPECT_TRUE(diagnostic.degenerate);

    EXPECT_THROW((void)makeMomentFittedCutVolumeQuadrature(
                     1.0,
                     0.5,
                     CutIntegrationSide::Interface,
                     {{0.0, 0.0, 0.0}},
                     {{1.0, 0.0, 0.0}},
                     2,
                     provenance),
                 std::invalid_argument);
}

TEST(CutQuadrature, AnalyticAxisAlignedBoxSensitivityAvoidsProductionFiniteDifferences)
{
    const auto sensitivity = makeAxisAlignedBoxCutSensitivity(
        {{0.0, 0.0, 0.0}},
        {{1.0, 2.0, 3.0}},
        0,
        0.5,
        CutIntegrationSide::Negative);
    ASSERT_TRUE(sensitivity.available);
    EXPECT_DOUBLE_EQ(sensitivity.d_location_d_plane_origin_diagonal[0], 1.0);
    EXPECT_DOUBLE_EQ(sensitivity.d_measure_d_plane_origin[0], 6.0);
    EXPECT_DOUBLE_EQ(sensitivity.d_volume_fraction_d_cut_coordinate, 1.0);

    const auto outside = makeAxisAlignedBoxCutSensitivity(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        0,
        2.0,
        CutIntegrationSide::Negative);
    EXPECT_FALSE(outside.available);
    EXPECT_FALSE(outside.capability_diagnostic.empty());

    const auto location = makePlaneCutLocationSensitivity({{0.0, 0.0, 2.0}});
    ASSERT_TRUE(location.available);
    EXPECT_DOUBLE_EQ(location.d_location_d_plane_origin_diagonal[2], 1.0);
    EXPECT_DOUBLE_EQ(location.d_location_d_mesh_point_diagonal[0], 1.0);
}

TEST(CutQuadrature, DiagnosticsFlagDegenerateAndSmallCutFractions)
{
    auto rule = makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        0,
        1.0e-12,
        CutIntegrationSide::Negative,
        "small-cut");

    CutQuadratureValidityPolicy policy;
    policy.min_fraction = 1.0e-6;
    policy.min_measure = 1.0e-14;
    const auto diagnostic = diagnoseCutQuadrature(rule, policy);
    EXPECT_TRUE(diagnostic.ok);
    EXPECT_TRUE(diagnostic.small_fraction);

    rule.measure = 0.0;
    const auto degenerate = diagnoseCutQuadrature(rule, policy);
    EXPECT_FALSE(degenerate.ok);
    EXPECT_TRUE(degenerate.degenerate);
}

TEST(CutQuadrature, DiagnosticsRejectMalformedCurvedQuadratureRules)
{
    CutQuadratureRule inverted_normals;
    inverted_normals.kind = CutQuadratureKind::Interface;
    inverted_normals.side = CutIntegrationSide::Interface;
    inverted_normals.curved_geometry = true;
    inverted_normals.measure = 1.0;
    inverted_normals.parent_measure = 1.0;
    inverted_normals.points = {
        CutQuadraturePoint{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}, 0.5},
        CutQuadraturePoint{{{1.0, 0.0, 0.0}}, {{0.0, 0.0, -1.0}}, 0.5}};

    const auto inverted = diagnoseCutQuadrature(inverted_normals);
    EXPECT_FALSE(inverted.ok);
    EXPECT_TRUE(inverted.inconsistent_normals);
    EXPECT_TRUE(inverted.degenerate);

    CutQuadratureRule nonfinite_normal = inverted_normals;
    nonfinite_normal.points[1].normal = {{0.0, std::nan(""), 1.0}};
    const auto nonfinite = diagnoseCutQuadrature(nonfinite_normal);
    EXPECT_FALSE(nonfinite.ok);
    EXPECT_TRUE(nonfinite.nonfinite_geometry);

    CutQuadratureRule mismatched_weights = inverted_normals;
    mismatched_weights.points = {
        CutQuadraturePoint{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}, 0.25},
        CutQuadraturePoint{{{1.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}, 0.25}};
    const auto mismatched = diagnoseCutQuadrature(mismatched_weights);
    EXPECT_FALSE(mismatched.ok);
    EXPECT_TRUE(mismatched.conservation_failure);
}
