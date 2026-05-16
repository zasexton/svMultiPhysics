#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::geometry;
using namespace svmp::FE::interfaces;

namespace {

Real integrateCoordinate(const CutQuadratureRule& rule, std::size_t component)
{
    Real value = 0.0;
    for (const auto& point : rule.points) {
        value += point.weight * point.point[component];
    }
    return value;
}

Real integrateSquaredRadius(const CutQuadratureRule& rule, std::size_t dimension)
{
    Real value = 0.0;
    for (const auto& point : rule.points) {
        Real radius_squared = 0.0;
        for (std::size_t component = 0; component < dimension; ++component) {
            radius_squared += point.point[component] * point.point[component];
        }
        value += point.weight * radius_squared;
    }
    return value;
}

Real integrateWeight(const CutQuadratureRule& rule)
{
    Real value = 0.0;
    for (const auto& point : rule.points) {
        value += point.weight;
    }
    return value;
}

Real integrateQuadraticCutVolumeOnUnitTriangle(int cells_per_axis)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromEvaluator("quadratic-convergence-source");
    request.interface_marker = 89;
    request.volume_quadrature_order = 2;

    LevelSetInterfaceDomain domain(request);
    MeshIndex parent_cell = 0;
    const Real h = Real{1.0} / static_cast<Real>(cells_per_axis);
    for (int j = 0; j < cells_per_axis; ++j) {
        for (int i = 0; i < cells_per_axis; ++i) {
            const Real x0 = h * static_cast<Real>(i);
            const Real y0 = h * static_cast<Real>(j);
            const Real x1 = x0 + h;
            const Real y1 = y0 + h;
            const std::vector<std::array<Real, 3>> nodes{
                {{x0, y0, 0.0}},
                {{x1, y0, 0.0}},
                {{x1, y1, 0.0}},
                {{x0, y1, 0.0}}};
            appendLinearLevelSetCellCut2D(
                domain,
                LevelSetCellCutInput{
                    .parent_cell = parent_cell++,
                    .element_type = ElementType::Quad4,
                    .node_coordinates = nodes,
                    .level_set_values = {x0 + y0 - 1.0,
                                         x1 + y0 - 1.0,
                                         x1 + y1 - 1.0,
                                         x0 + y1 - 1.0}});
        }
    }

    Real value = 0.0;
    for (const auto& rule : domain.volumeQuadratureRules()) {
        if (rule.side != CutIntegrationSide::Negative) {
            continue;
        }
        for (const auto& point : rule.points) {
            const Real x = point.point[0];
            const Real y = point.point[1];
            value += point.weight * (x * x + y * y);
        }
    }
    return value;
}

} // namespace

TEST(LevelSetInterfaceDomain, RequestCarriesFieldSourceAndMarker)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/3,
                                                        /*layout_revision=*/11,
                                                        /*value_revision=*/17);
    request.interface_marker = 42;
    request.isovalue = 0.25;

    LevelSetInterfaceDomain domain(request);

    EXPECT_TRUE(request.valid());
    EXPECT_EQ(domain.marker(), 42);
    EXPECT_EQ(domain.request().source.field_id, 3);
    EXPECT_EQ(domain.request().source.identifier(), "field:3");
    EXPECT_TRUE(domain.empty());

    const auto summary = domain.summary();
    EXPECT_EQ(summary.interface_marker, 42);
    EXPECT_EQ(summary.fragment_count, 0u);
    EXPECT_DOUBLE_EQ(summary.measure, 0.0);
}

TEST(LevelSetInterfaceDomain, AccumulatesFragmentsAndExportsCutQuadrature)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromEvaluator("plane-level-set",
                                                            /*layout_revision=*/5,
                                                            /*value_revision=*/9);
    request.interface_marker = 7;
    request.quadrature_policy_key = 101;

    LevelSetInterfaceDomain domain(request);

    CutInterfaceFragment fragment;
    fragment.parent_cell = 4;
    fragment.kind = CutInterfaceFragmentKind::Segment;
    fragment.normal = {{0.0, 1.0, 0.0}};
    fragment.measure = 1.0;
    fragment.min_level_set_value = -0.5;
    fragment.max_level_set_value = 0.5;
    fragment.topology_id = "cell-4-fragment-0";
    fragment.vertices = {
        CutInterfaceVertex{.point = {{0.0, 0.5, 0.0}},
                           .parent_coordinate = {{0.0, 0.5, 0.0}},
                           .level_set_value = 0.0,
                           .stable_id = 1},
        CutInterfaceVertex{.point = {{1.0, 0.5, 0.0}},
                           .parent_coordinate = {{1.0, 0.5, 0.0}},
                           .level_set_value = 0.0,
                           .stable_id = 2}};
    fragment.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = {{0.5, 0.5, 0.0}},
                                    .parent_coordinate = {{0.5, 0.5, 0.0}},
                                    .normal = {{0.0, 1.0, 0.0}},
                                    .weight = 1.0}};

    domain.addFragment(fragment);

    const auto summary = domain.summary();
    EXPECT_EQ(summary.fragment_count, 1u);
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.quadrature_point_count, 1u);
    EXPECT_DOUBLE_EQ(summary.measure, 1.0);

    ASSERT_EQ(domain.fragments().size(), 1u);
    EXPECT_EQ(domain.fragments().front().interface_marker, 7);
    EXPECT_EQ(domain.fragments().front().local_fragment_index, 0u);
    EXPECT_NE(domain.fragments().front().stable_id, 0u);

    const auto rules = domain.interfaceQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().kind, svmp::FE::geometry::CutQuadratureKind::Interface);
    EXPECT_EQ(rules.front().side, svmp::FE::geometry::CutIntegrationSide::Interface);
    EXPECT_EQ(rules.front().provenance.embedded_geometry_id, "plane-level-set");
    EXPECT_EQ(rules.front().provenance.cut_topology_id, "cell-4-fragment-0");
    EXPECT_EQ(rules.front().provenance.parent_entity, 4);
    EXPECT_EQ(rules.front().provenance.predicate_policy_key, 101u);
    ASSERT_EQ(rules.front().points.size(), 1u);
    EXPECT_DOUBLE_EQ(rules.front().points.front().weight, 1.0);
    EXPECT_DOUBLE_EQ(rules.front().points.front().point[0], 0.5);
    EXPECT_DOUBLE_EQ(rules.front().points.front().normal[1], 1.0);
}

TEST(LevelSetInterfaceDomain, IdentifiesUniqueCutCellsFromActiveFragments)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/5,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 12;

    LevelSetInterfaceDomain domain(request);

    CutInterfaceFragment fragment;
    fragment.parent_cell = 8;
    fragment.measure = 0.5;
    domain.addFragment(fragment);

    CutInterfaceFragment duplicate = fragment;
    duplicate.parent_cell = 8;
    domain.addFragment(duplicate);

    CutInterfaceFragment earlier_cell = fragment;
    earlier_cell.parent_cell = 3;
    domain.addFragment(earlier_cell);

    CutInterfaceFragment inactive;
    inactive.parent_cell = 2;
    inactive.degeneracy = CutInterfaceDegeneracy::NoCut;
    domain.addFragment(inactive);

    const auto cells = domain.cutCells();
    ASSERT_EQ(cells.size(), 2u);
    EXPECT_EQ(cells[0], 3);
    EXPECT_EQ(cells[1], 8);
}

TEST(LevelSetInterfaceDomain, StableIdsTrackMarkerCellFragmentAndRevision)
{
    const auto base = cutInterfaceStableId(/*interface_marker=*/3,
                                           /*parent_cell=*/4,
                                           /*local_fragment_index=*/0,
                                           /*source_revision=*/9);
    EXPECT_EQ(base, cutInterfaceStableId(3, 4, 0, 9));
    EXPECT_NE(base, cutInterfaceStableId(8, 4, 0, 9));
    EXPECT_NE(base, cutInterfaceStableId(3, 5, 0, 9));
    EXPECT_NE(base, cutInterfaceStableId(3, 4, 1, 9));
    EXPECT_NE(base, cutInterfaceStableId(3, 4, 0, 10));
}

TEST(LevelSetInterfaceDomain, SummaryCountsDegenerateInactiveFragments)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/2,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 6;

    LevelSetInterfaceDomain domain(request);

    CutInterfaceFragment no_cut;
    no_cut.parent_cell = 2;
    no_cut.degeneracy = CutInterfaceDegeneracy::NoCut;
    domain.addFragment(no_cut);

    CutInterfaceFragment small_fragment;
    small_fragment.parent_cell = 3;
    small_fragment.degeneracy = CutInterfaceDegeneracy::SmallFragment;
    small_fragment.measure = 1.0e-16;
    small_fragment.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = {{0.0, 0.0, 0.0}},
                                    .parent_coordinate = {{0.0, 0.0, 0.0}},
                                    .normal = {{1.0, 0.0, 0.0}},
                                    .weight = 1.0e-16}};
    domain.addFragment(small_fragment);

    const auto summary = domain.summary();
    EXPECT_EQ(summary.fragment_count, 2u);
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.degenerate_fragment_count, 2u);
    EXPECT_EQ(summary.quadrature_point_count, 1u);
    EXPECT_DOUBLE_EQ(summary.measure, 1.0e-16);
}

TEST(LevelSetInterfaceDomain, GeneratedInterfaceMarkersAreStable)
{
    GeneratedInterfaceMarkerKey key;
    key.source = LevelSetInterfaceSource::fromField(/*field_id=*/8,
                                                    /*layout_revision=*/1,
                                                    /*value_revision=*/11);
    key.domain_id = "fluid-interface";
    key.isovalue = 0.0;

    GeneratedInterfaceMarkerKey same_domain_new_values = key;
    same_domain_new_values.source = LevelSetInterfaceSource::fromField(
        /*field_id=*/8,
        /*layout_revision=*/1,
        /*value_revision=*/12);

    EXPECT_EQ(stableGeneratedInterfaceMarker(key),
              stableGeneratedInterfaceMarker(same_domain_new_values));

    GeneratedInterfaceMarkerRegistry registry;
    const int marker = registry.assign(key);
    EXPECT_EQ(registry.assign(same_domain_new_values), marker);
    EXPECT_TRUE(registry.contains(key));
    EXPECT_TRUE(registry.containsMarker(marker));
    EXPECT_EQ(registry.size(), 1u);

    GeneratedInterfaceMarkerKey other_domain = key;
    other_domain.domain_id = "secondary-interface";
    const int other_marker = registry.assign(other_domain);
    EXPECT_NE(other_marker, marker);
    EXPECT_EQ(registry.size(), 2u);
}

TEST(LevelSetInterfaceDomain, GeneratedInterfaceMarkerRegistryHonorsExplicitMarkers)
{
    GeneratedInterfaceMarkerKey key;
    key.source = LevelSetInterfaceSource::fromEvaluator("phi-evaluator");
    key.domain_id = "explicit-interface";
    key.isovalue = 0.0;
    key.requested_marker = 17;

    GeneratedInterfaceMarkerRegistry registry;
    EXPECT_EQ(registry.assign(key), 17);
    EXPECT_EQ(registry.assign(key), 17);

    GeneratedInterfaceMarkerKey colliding = key;
    colliding.domain_id = "different-interface";
    EXPECT_THROW((void)registry.assign(colliding), std::invalid_argument);
}

TEST(LevelSetInterfaceDomain, GeneratedInterfaceMarkerRegistryResolvesHashCollisions)
{
    GeneratedInterfaceMarkerRegistry registry(/*marker_base=*/200,
                                             /*marker_range=*/2);

    GeneratedInterfaceMarkerKey first;
    first.source = LevelSetInterfaceSource::fromEvaluator("first");
    first.domain_id = "interface";

    GeneratedInterfaceMarkerKey second;
    second.source = LevelSetInterfaceSource::fromEvaluator("second");
    second.domain_id = "interface";

    const int first_marker = registry.assign(first);
    const int second_marker = registry.assign(second);
    EXPECT_NE(first_marker, second_marker);
    EXPECT_GE(first_marker, 200);
    EXPECT_LT(first_marker, 202);
    EXPECT_GE(second_marker, 200);
    EXPECT_LT(second_marker, 202);
    EXPECT_EQ(registry.size(), 2u);
}

TEST(LevelSetInterfaceDomain, LinearFragmentQuadratureRulesCarryCentroidWeightsAndNormals)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromEvaluator("linear-fragment-source");
    request.interface_marker = 81;
    request.quadrature_policy_key = 23;

    LevelSetInterfaceDomain domain(request);
    CutInterfaceFragment segment;
    segment.parent_cell = 1;
    segment.kind = CutInterfaceFragmentKind::Segment;
    segment.normal = {{0.0, 1.0, 0.0}};
    segment.measure = 2.0;
    segment.topology_id = "segment-fragment";
    segment.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = {{1.0, 0.0, 0.0}},
                                    .parent_coordinate = {{1.0, 0.0, 0.0}},
                                    .normal = {{0.0, 1.0, 0.0}},
                                    .weight = 2.0}};
    domain.addFragment(segment);

    CutInterfaceFragment polygon;
    polygon.parent_cell = 2;
    polygon.kind = CutInterfaceFragmentKind::Polygon;
    polygon.normal = {{0.0, 0.0, 1.0}};
    polygon.measure = 0.5;
    polygon.topology_id = "polygon-fragment";
    polygon.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = {{1.0 / 3.0, 1.0 / 3.0, 0.0}},
                                    .parent_coordinate = {{1.0 / 3.0, 1.0 / 3.0, 0.0}},
                                    .normal = {{0.0, 0.0, 1.0}},
                                    .weight = 0.5}};
    domain.addFragment(polygon);

    const auto rules = domain.interfaceQuadratureRules();
    ASSERT_EQ(rules.size(), 2u);

    EXPECT_EQ(rules[0].kind, svmp::FE::geometry::CutQuadratureKind::Interface);
    EXPECT_EQ(rules[0].exact_polynomial_order, 1);
    ASSERT_EQ(rules[0].points.size(), 1u);
    EXPECT_DOUBLE_EQ(rules[0].points.front().point[0], 1.0);
    EXPECT_DOUBLE_EQ(rules[0].points.front().weight, 2.0);
    EXPECT_DOUBLE_EQ(rules[0].points.front().normal[1], 1.0);
    EXPECT_EQ(rules[0].provenance.predicate_policy_key, 23u);

    EXPECT_EQ(rules[1].kind, svmp::FE::geometry::CutQuadratureKind::Interface);
    EXPECT_EQ(rules[1].exact_polynomial_order, 1);
    ASSERT_EQ(rules[1].points.size(), 1u);
    EXPECT_DOUBLE_EQ(rules[1].points.front().point[0], 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(rules[1].points.front().point[1], 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(rules[1].points.front().weight, 0.5);
    EXPECT_DOUBLE_EQ(rules[1].points.front().normal[2], 1.0);
}

TEST(LevelSetInterfaceDomain, CutVolumeMetadataIsDeterministicForOwnedAndGhostRegions)
{
    const auto make_domain = [](std::uint64_t ownership_revision, bool reverse_order) {
        CutInterfaceDomainRequest request;
        request.source = LevelSetInterfaceSource::fromEvaluator("deterministic-volume-source",
                                                                /*layout_revision=*/3,
                                                                /*value_revision=*/17);
        request.interface_marker = 87;
        request.ownership_revision = ownership_revision;

        std::vector<CutInterfaceVolumeRegion> regions;
        auto make_region = [](MeshIndex parent_cell,
                              LocalIndex local_region_index,
                              CutIntegrationSide side,
                              Real measure) {
            CutInterfaceVolumeRegion region;
            region.parent_cell = parent_cell;
            region.local_region_index = local_region_index;
            region.side = side;
            region.parent_measure = 1.0;
            region.volume_fraction = measure;
            region.measure = measure;
            region.centroid = {{measure, 0.25, 0.0}};
            region.topology_id = "cell-" + std::to_string(parent_cell) + "-region-" +
                                 std::to_string(local_region_index);
            return region;
        };
        regions.push_back(make_region(/*parent_cell=*/7,
                                      /*local_region_index=*/0,
                                      CutIntegrationSide::Negative,
                                      0.25));
        regions.push_back(make_region(/*parent_cell=*/4,
                                      /*local_region_index=*/1,
                                      CutIntegrationSide::Positive,
                                      0.75));
        regions.push_back(make_region(/*parent_cell=*/4,
                                      /*local_region_index=*/0,
                                      CutIntegrationSide::Negative,
                                      0.25));
        if (reverse_order) {
            std::reverse(regions.begin(), regions.end());
        }

        LevelSetInterfaceDomain domain(request);
        for (auto& region : regions) {
            domain.addVolumeRegion(std::move(region));
        }
        return domain;
    };

    const auto owned_rules =
        make_domain(/*ownership_revision=*/5, /*reverse_order=*/false).volumeQuadratureRules();
    const auto ghost_rules =
        make_domain(/*ownership_revision=*/6, /*reverse_order=*/true).volumeQuadratureRules();

    ASSERT_EQ(owned_rules.size(), 3u);
    ASSERT_EQ(ghost_rules.size(), owned_rules.size());
    EXPECT_EQ(owned_rules[0].provenance.parent_entity, 4);
    EXPECT_EQ(owned_rules[0].side, CutIntegrationSide::Negative);
    EXPECT_EQ(owned_rules[1].provenance.parent_entity, 4);
    EXPECT_EQ(owned_rules[1].side, CutIntegrationSide::Positive);
    EXPECT_EQ(owned_rules[2].provenance.parent_entity, 7);
    EXPECT_EQ(owned_rules[2].side, CutIntegrationSide::Negative);

    for (std::size_t i = 0; i < owned_rules.size(); ++i) {
        EXPECT_EQ(ghost_rules[i].provenance.parent_entity,
                  owned_rules[i].provenance.parent_entity);
        EXPECT_EQ(ghost_rules[i].side, owned_rules[i].side);
        EXPECT_EQ(ghost_rules[i].provenance.cut_topology_revision,
                  owned_rules[i].provenance.cut_topology_revision);
        EXPECT_EQ(ghost_rules[i].provenance.cut_topology_id,
                  owned_rules[i].provenance.cut_topology_id);
        EXPECT_NEAR(ghost_rules[i].measure, owned_rules[i].measure, 1.0e-14);
    }
}

TEST(LevelSetInterfaceDomain, InterfaceRulesUseDeterministicMetadataOrdering)
{
    const auto make_domain = [](bool reverse_order) {
        CutInterfaceDomainRequest request;
        request.source = LevelSetInterfaceSource::fromEvaluator("deterministic-interface-source",
                                                                /*layout_revision=*/3,
                                                                /*value_revision=*/19);
        request.interface_marker = 86;

        std::vector<CutInterfaceFragment> fragments;
        auto make_fragment = [](MeshIndex parent_cell,
                                LocalIndex local_fragment_index,
                                Real x_coordinate) {
            CutInterfaceFragment fragment;
            fragment.parent_cell = parent_cell;
            fragment.local_fragment_index = local_fragment_index;
            fragment.measure = 1.0;
            fragment.topology_id = "cell-" + std::to_string(parent_cell) +
                                   "-fragment-" +
                                   std::to_string(local_fragment_index);
            fragment.quadrature_points = {
                CutInterfaceQuadraturePoint{
                    .point = {{x_coordinate, 0.25, 0.0}},
                    .parent_coordinate = {{x_coordinate, 0.25, 0.0}},
                    .normal = {{0.0, 1.0, 0.0}},
                    .weight = 1.0}};
            return fragment;
        };
        fragments.push_back(make_fragment(/*parent_cell=*/7,
                                          /*local_fragment_index=*/0,
                                          0.75));
        fragments.push_back(make_fragment(/*parent_cell=*/4,
                                          /*local_fragment_index=*/1,
                                          0.5));
        fragments.push_back(make_fragment(/*parent_cell=*/4,
                                          /*local_fragment_index=*/0,
                                          0.25));
        if (reverse_order) {
            std::reverse(fragments.begin(), fragments.end());
        }

        LevelSetInterfaceDomain domain(request);
        for (auto& fragment : fragments) {
            domain.addFragment(std::move(fragment));
        }
        return domain;
    };

    const auto forward_rules =
        make_domain(/*reverse_order=*/false).interfaceQuadratureRules();
    const auto reversed_rules =
        make_domain(/*reverse_order=*/true).interfaceQuadratureRules();

    ASSERT_EQ(forward_rules.size(), 3u);
    ASSERT_EQ(reversed_rules.size(), forward_rules.size());
    EXPECT_EQ(forward_rules[0].provenance.parent_entity, 4);
    EXPECT_EQ(forward_rules[0].provenance.cut_topology_id,
              "cell-4-fragment-0");
    EXPECT_EQ(forward_rules[1].provenance.parent_entity, 4);
    EXPECT_EQ(forward_rules[1].provenance.cut_topology_id,
              "cell-4-fragment-1");
    EXPECT_EQ(forward_rules[2].provenance.parent_entity, 7);
    EXPECT_EQ(forward_rules[2].provenance.cut_topology_id,
              "cell-7-fragment-0");

    for (std::size_t i = 0; i < forward_rules.size(); ++i) {
        EXPECT_EQ(reversed_rules[i].provenance.parent_entity,
                  forward_rules[i].provenance.parent_entity);
        EXPECT_EQ(reversed_rules[i].provenance.cut_topology_revision,
                  forward_rules[i].provenance.cut_topology_revision);
        EXPECT_EQ(reversed_rules[i].provenance.cut_topology_id,
                  forward_rules[i].provenance.cut_topology_id);
        ASSERT_EQ(reversed_rules[i].points.size(), forward_rules[i].points.size());
        EXPECT_NEAR(reversed_rules[i].points.front().point[0],
                    forward_rules[i].points.front().point[0],
                    1.0e-14);
    }
}

TEST(LevelSetInterfaceDomain, LinearCellCutsExportFullSideVolumeRules)
{
    CutInterfaceDomainRequest negative_request;
    negative_request.source = LevelSetInterfaceSource::fromEvaluator("full-negative-source");
    negative_request.interface_marker = 83;
    negative_request.quadrature_policy_key = 29;

    LevelSetInterfaceDomain negative_domain(negative_request);
    const LevelSetCellCutInput negative_input{
        .parent_cell = 11,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-1.0, -1.0, -1.0, -1.0}};
    appendLinearLevelSetCellCut2D(negative_domain, negative_input);

    auto summary = negative_domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 0u);
    EXPECT_EQ(summary.active_volume_region_count, 1u);
    EXPECT_NEAR(summary.negative_volume_measure, 1.0, 1.0e-14);
    EXPECT_NEAR(summary.positive_volume_measure, 0.0, 1.0e-14);

    auto rules = negative_domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().kind, CutQuadratureKind::Volume);
    EXPECT_EQ(rules.front().side, CutIntegrationSide::Negative);
    EXPECT_EQ(rules.front().provenance.parent_entity, 11);
    EXPECT_EQ(rules.front().provenance.marker, 83);
    EXPECT_EQ(rules.front().provenance.predicate_policy_key, 29u);
    EXPECT_NEAR(rules.front().parent_measure, 1.0, 1.0e-14);
    EXPECT_NEAR(rules.front().measure, 1.0, 1.0e-14);
    EXPECT_TRUE(rules.front().full_cell_equivalent);
    ASSERT_EQ(rules.front().points.size(), 1u);
    EXPECT_NEAR(rules.front().points.front().weight, 1.0, 1.0e-14);

    CutInterfaceDomainRequest positive_request;
    positive_request.source = LevelSetInterfaceSource::fromEvaluator("full-positive-source");
    positive_request.interface_marker = 84;

    LevelSetInterfaceDomain positive_domain(positive_request);
    const LevelSetCellCutInput positive_input{
        .parent_cell = 12,
        .element_type = ElementType::Quad4,
        .node_coordinates = negative_input.node_coordinates,
        .level_set_values = {1.0, 1.0, 1.0, 1.0}};
    appendLinearLevelSetCellCut2D(positive_domain, positive_input);

    summary = positive_domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 0u);
    EXPECT_EQ(summary.active_volume_region_count, 1u);
    EXPECT_NEAR(summary.negative_volume_measure, 0.0, 1.0e-14);
    EXPECT_NEAR(summary.positive_volume_measure, 1.0, 1.0e-14);

    rules = positive_domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().side, CutIntegrationSide::Positive);
    EXPECT_EQ(rules.front().provenance.parent_entity, 12);
    EXPECT_EQ(rules.front().provenance.marker, 84);
    EXPECT_NEAR(rules.front().measure, 1.0, 1.0e-14);
    EXPECT_TRUE(rules.front().full_cell_equivalent);
}

TEST(LevelSetInterfaceDomain, LinearCellCutsExportCutSideVolumeRules)
{
    CutInterfaceDomainRequest square_request;
    square_request.source = LevelSetInterfaceSource::fromEvaluator("cut-square-source");
    square_request.interface_marker = 85;

    LevelSetInterfaceDomain square_domain(square_request);
    const LevelSetCellCutInput square_input{
        .parent_cell = 13,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};
    appendLinearLevelSetCellCut2D(square_domain, square_input);

    auto summary = square_domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.active_volume_region_count, 2u);
    EXPECT_NEAR(summary.negative_volume_measure, 0.5, 1.0e-14);
    EXPECT_NEAR(summary.positive_volume_measure, 0.5, 1.0e-14);

    auto rules = square_domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_EQ(rules[0].side, CutIntegrationSide::Negative);
    EXPECT_EQ(rules[1].side, CutIntegrationSide::Positive);
    EXPECT_FALSE(rules[0].full_cell_equivalent);
    EXPECT_FALSE(rules[1].full_cell_equivalent);
    EXPECT_NEAR(rules[0].measure, 0.5, 1.0e-14);
    EXPECT_NEAR(rules[1].measure, 0.5, 1.0e-14);
    EXPECT_NEAR(integrateWeight(rules[0]) + integrateWeight(rules[1]), 1.0, 1.0e-14);
    EXPECT_EQ(rules[0].exact_polynomial_order, 2);
    EXPECT_EQ(rules[0].policy.name, "quadratic-subcell-level-set-volume");
    ASSERT_EQ(rules[0].points.size(), 6u);
    ASSERT_EQ(rules[1].points.size(), 6u);
    EXPECT_NEAR(integrateCoordinate(rules[0], 0), 0.125, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[0], 1), 0.25, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[1], 0), 0.375, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[1], 1), 0.25, 1.0e-14);
    EXPECT_NEAR(integrateSquaredRadius(rules[0], 2), 5.0 / 24.0, 1.0e-14);
    EXPECT_NEAR(integrateSquaredRadius(rules[1], 2), 11.0 / 24.0, 1.0e-14);

    CutInterfaceDomainRequest tetra_request;
    tetra_request.source = LevelSetInterfaceSource::fromEvaluator("cut-tetra-source");
    tetra_request.interface_marker = 86;

    LevelSetInterfaceDomain tetra_domain(tetra_request);
    const LevelSetCellCutInput tetra_input{
        .parent_cell = 14,
        .element_type = ElementType::Tetra4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}},
                             {{0.0, 0.0, 1.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, 0.5}};
    appendLinearLevelSetCellCut3D(tetra_domain, tetra_input);

    summary = tetra_domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.active_volume_region_count, 2u);

    rules = tetra_domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_EQ(rules[0].side, CutIntegrationSide::Negative);
    EXPECT_EQ(rules[1].side, CutIntegrationSide::Positive);
    EXPECT_FALSE(rules[0].full_cell_equivalent);
    EXPECT_FALSE(rules[1].full_cell_equivalent);
    EXPECT_NEAR(rules[0].parent_measure, 1.0 / 6.0, 1.0e-14);
    EXPECT_NEAR(rules[0].volume_fraction, 1.0 / 8.0, 1.0e-14);
    EXPECT_NEAR(rules[1].volume_fraction, 7.0 / 8.0, 1.0e-14);
    EXPECT_NEAR(rules[0].measure, 1.0 / 48.0, 1.0e-14);
    EXPECT_NEAR(rules[1].measure, 7.0 / 48.0, 1.0e-14);
    EXPECT_EQ(rules[0].exact_polynomial_order, 2);
    EXPECT_EQ(rules[0].policy.name, "quadratic-subcell-level-set-volume");
    EXPECT_GT(rules[0].points.size(), 1u);
    EXPECT_GT(rules[1].points.size(), 1u);
    EXPECT_NEAR(integrateWeight(rules[0]), 1.0 / 48.0, 1.0e-14);
    EXPECT_NEAR(integrateWeight(rules[1]), 7.0 / 48.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[0], 0), 1.0 / 384.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[0], 1), 1.0 / 384.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[0], 2), 1.0 / 384.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[1], 0), 5.0 / 128.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[1], 1), 5.0 / 128.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[1], 2), 5.0 / 128.0, 1.0e-14);
    EXPECT_NEAR(integrateSquaredRadius(rules[0], 3), 1.0 / 640.0, 1.0e-14);
    EXPECT_NEAR(integrateSquaredRadius(rules[1], 3), 31.0 / 640.0, 1.0e-14);
}

TEST(LevelSetInterfaceDomain, CutVolumeRulesConserveConstantsForSupportedElementTypes)
{
    struct Case {
        ElementType element_type{ElementType::Unknown};
        bool three_dimensional{false};
        std::vector<std::array<Real, 3>> node_coordinates{};
        std::vector<Real> level_set_values{};
        Real parent_measure{0.0};
    };

    const std::vector<std::array<Real, 3>> triangle_nodes{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    const std::vector<std::array<Real, 3>> quad_nodes{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    const std::vector<std::array<Real, 3>> tetra_nodes{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}};

    const std::vector<Case> cases{
        Case{ElementType::Triangle3, false, triangle_nodes, {-0.5, 0.5, -0.5}, 0.5},
        Case{ElementType::Triangle6, false, triangle_nodes, {-0.5, 0.5, -0.5}, 0.5},
        Case{ElementType::Quad4, false, quad_nodes, {-0.5, 0.5, 0.5, -0.5}, 1.0},
        Case{ElementType::Quad8, false, quad_nodes, {-0.5, 0.5, 0.5, -0.5}, 1.0},
        Case{ElementType::Quad9, false, quad_nodes, {-0.5, 0.5, 0.5, -0.5}, 1.0},
        Case{ElementType::Tetra4, true, tetra_nodes, {-0.5, 0.5, 0.5, 0.5}, 1.0 / 6.0},
        Case{ElementType::Tetra10, true, tetra_nodes, {-0.5, 0.5, 0.5, 0.5}, 1.0 / 6.0}};

    int marker = 91;
    MeshIndex parent_cell = 100;
    for (const auto& c : cases) {
        CutInterfaceDomainRequest request;
        request.source = LevelSetInterfaceSource::fromEvaluator("constant-volume-source");
        request.interface_marker = marker++;
        request.volume_quadrature_order = 3;

        LevelSetInterfaceDomain domain(request);
        const LevelSetCellCutInput input{
            .parent_cell = parent_cell++,
            .element_type = c.element_type,
            .node_coordinates = c.node_coordinates,
            .level_set_values = c.level_set_values};
        if (c.three_dimensional) {
            appendLinearLevelSetCellCut3D(domain, input);
        } else {
            appendLinearLevelSetCellCut2D(domain, input);
        }

        const auto rules = domain.volumeQuadratureRules();
        ASSERT_EQ(rules.size(), 2u);
        Real measure_sum = 0.0;
        Real weight_sum = 0.0;
        for (const auto& rule : rules) {
            EXPECT_TRUE(rule.exact_for_constants);
            EXPECT_EQ(rule.exact_polynomial_order, 2);
            EXPECT_NEAR(rule.parent_measure, c.parent_measure, 1.0e-14);
            measure_sum += rule.measure;
            for (const auto& point : rule.points) {
                weight_sum += point.weight;
            }
        }
        EXPECT_NEAR(measure_sum, c.parent_measure, 1.0e-14);
        EXPECT_NEAR(weight_sum, c.parent_measure, 1.0e-14);
    }
}

TEST(LevelSetInterfaceDomain, CutVolumeQuadratureConvergesForQuadraticFields)
{
    constexpr Real exact = Real{1.0} / Real{6.0};
    const Real coarse_error =
        std::abs(integrateQuadraticCutVolumeOnUnitTriangle(/*cells_per_axis=*/8) - exact);
    const Real fine_error =
        std::abs(integrateQuadraticCutVolumeOnUnitTriangle(/*cells_per_axis=*/16) - exact);

    EXPECT_LT(coarse_error, 1.0e-12);
    EXPECT_LT(fine_error, 1.0e-12);
}

TEST(LevelSetInterfaceDomain, LinearCellCutsExportTriangleVolumeCentroids)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromEvaluator("cut-triangle-source");
    request.interface_marker = 87;

    LevelSetInterfaceDomain domain(request);
    const LevelSetCellCutInput input{
        .parent_cell = 15,
        .element_type = ElementType::Triangle3,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, -0.5}};
    appendLinearLevelSetCellCut2D(domain, input);

    const auto rules = domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_EQ(rules[0].side, CutIntegrationSide::Negative);
    EXPECT_EQ(rules[1].side, CutIntegrationSide::Positive);
    EXPECT_NEAR(rules[0].measure, 3.0 / 8.0, 1.0e-14);
    EXPECT_NEAR(rules[1].measure, 1.0 / 8.0, 1.0e-14);
    EXPECT_NEAR(rules[0].measure + rules[1].measure, 0.5, 1.0e-14);
    ASSERT_EQ(rules[0].points.size(), 6u);
    ASSERT_EQ(rules[1].points.size(), 3u);
    EXPECT_NEAR(integrateCoordinate(rules[0], 0), 1.0 / 12.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[0], 1), 7.0 / 48.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[1], 0), 1.0 / 12.0, 1.0e-14);
    EXPECT_NEAR(integrateCoordinate(rules[1], 1), 1.0 / 48.0, 1.0e-14);
}

TEST(LevelSetInterfaceDomain, ConfigurableQuadratureOrderIsRecordedAndValidated)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromEvaluator("quadrature-order-source");
    request.interface_marker = 82;
    request.quadrature_order = 0;

    LevelSetInterfaceDomain domain(request);
    CutInterfaceFragment segment;
    segment.parent_cell = 3;
    segment.kind = CutInterfaceFragmentKind::Segment;
    segment.measure = 1.0;
    segment.topology_id = "constant-order-segment";
    segment.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = {{0.5, 0.0, 0.0}},
                                    .parent_coordinate = {{0.5, 0.0, 0.0}},
                                    .normal = {{1.0, 0.0, 0.0}},
                                    .weight = 1.0}};
    domain.addFragment(segment);

    const auto constant_rules = domain.interfaceQuadratureRules();
    ASSERT_EQ(constant_rules.size(), 1u);
    EXPECT_EQ(constant_rules.front().exact_polynomial_order, 0);
    EXPECT_EQ(constant_rules.front().policy.polynomial_order, 0);
    EXPECT_EQ(constant_rules.front().policy.name, "constant-level-set-interface");

    CutInterfaceDomainRequest linear_request = request;
    linear_request.quadrature_order = 1;
    const auto linear_rule = domain.fragments().front().toCutQuadratureRule(linear_request);
    EXPECT_EQ(linear_rule.exact_polynomial_order, 1);
    EXPECT_EQ(linear_rule.policy.polynomial_order, 1);
    EXPECT_EQ(linear_rule.policy.name, "linear-level-set-interface");

    CutInterfaceDomainRequest high_order_request = request;
    high_order_request.quadrature_order = 2;
    EXPECT_THROW((void)domain.fragments().front().toCutQuadratureRule(high_order_request),
                 std::invalid_argument);

    CutInterfaceVolumeRegion region;
    region.parent_cell = 8;
    region.side = CutIntegrationSide::Negative;
    region.parent_measure = 2.0;
    region.measure = 0.5;
    region.volume_fraction = 0.25;
    region.centroid = {{0.25, 0.5, 0.0}};
    region.topology_id = "cut-volume-region";
    domain.addVolumeRegion(region);

    const auto volume_rule = domain.volumeRegions().front().toCutQuadratureRule(linear_request);
    EXPECT_EQ(volume_rule.exact_polynomial_order, 1);
    EXPECT_EQ(volume_rule.policy.polynomial_order, 1);
    EXPECT_EQ(volume_rule.policy.name, "linear-moment-fitted-level-set-volume");
    EXPECT_TRUE(volume_rule.policy.moment_fitted);

    const auto high_order_volume_rule =
        domain.volumeRegions().front().toCutQuadratureRule(high_order_request);
    EXPECT_EQ(high_order_volume_rule.exact_polynomial_order, 1);
    EXPECT_EQ(high_order_volume_rule.policy.polynomial_order, 1);
    EXPECT_EQ(high_order_volume_rule.policy.name, "linear-moment-fitted-level-set-volume");
    EXPECT_TRUE(high_order_volume_rule.policy.moment_fitted);
    EXPECT_NEAR(high_order_volume_rule.measure, 0.5, 1.0e-14);
    ASSERT_EQ(high_order_volume_rule.points.size(), 1u);
    EXPECT_NEAR(high_order_volume_rule.points.front().weight, 0.5, 1.0e-14);
}

TEST(LevelSetInterfaceDomain, DefinesCutVolumeQuadratureOrderPolicy)
{
    EXPECT_EQ(minimumLevelSetCutVolumeQuadratureOrder(/*geometry_order=*/1,
                                                     /*field_order=*/1,
                                                     /*form_order=*/1),
              1);
    EXPECT_EQ(defaultLevelSetCutVolumeQuadratureOrder(/*geometry_order=*/1,
                                                     /*field_order=*/1,
                                                     /*form_order=*/1),
              1);
    EXPECT_EQ(minimumLevelSetCutVolumeQuadratureOrder(/*geometry_order=*/2,
                                                     /*field_order=*/1,
                                                     /*form_order=*/1),
              2);
    EXPECT_EQ(minimumLevelSetCutVolumeQuadratureOrder(/*geometry_order=*/1,
                                                     /*field_order=*/2,
                                                     /*form_order=*/1),
              2);
    EXPECT_EQ(defaultLevelSetCutVolumeQuadratureOrder(/*geometry_order=*/1,
                                                     /*field_order=*/1,
                                                     /*form_order=*/3),
              3);
    EXPECT_EQ(implementedLevelSetCutVolumeExactOrder(0), 0);
    EXPECT_EQ(implementedLevelSetCutVolumeExactOrder(1), 1);
    EXPECT_EQ(implementedLevelSetCutVolumeExactOrder(3), 2);
}

TEST(LevelSetInterfaceDomain, SeparateInterfaceAndVolumeQuadratureOrders)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromEvaluator("split-order-source");
    request.interface_marker = 88;
    request.quadrature_order = 1;
    request.interface_quadrature_order = 0;
    request.volume_quadrature_order = 1;

    LevelSetInterfaceDomain domain(request);
    CutInterfaceFragment segment;
    segment.parent_cell = 4;
    segment.kind = CutInterfaceFragmentKind::Segment;
    segment.measure = 1.0;
    segment.topology_id = "split-order-segment";
    segment.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = {{0.5, 0.0, 0.0}},
                                    .parent_coordinate = {{0.5, 0.0, 0.0}},
                                    .normal = {{1.0, 0.0, 0.0}},
                                    .weight = 1.0}};
    domain.addFragment(segment);

    CutInterfaceVolumeRegion region;
    region.parent_cell = 4;
    region.side = CutIntegrationSide::Negative;
    region.parent_measure = 1.0;
    region.measure = 0.5;
    region.volume_fraction = 0.5;
    region.centroid = {{0.25, 0.5, 0.0}};
    region.topology_id = "split-order-volume";
    domain.addVolumeRegion(region);

    const auto interface_rules = domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 1u);
    EXPECT_EQ(interface_rules.front().exact_polynomial_order, 0);
    EXPECT_EQ(interface_rules.front().policy.name, "constant-level-set-interface");

    const auto volume_rules = domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 1u);
    EXPECT_EQ(volume_rules.front().exact_polynomial_order, 1);
    EXPECT_EQ(volume_rules.front().policy.name,
              "linear-moment-fitted-level-set-volume");
}
