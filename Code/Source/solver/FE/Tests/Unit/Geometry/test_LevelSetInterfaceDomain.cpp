#include "Interfaces/LevelSetInterfaceDomain.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

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
