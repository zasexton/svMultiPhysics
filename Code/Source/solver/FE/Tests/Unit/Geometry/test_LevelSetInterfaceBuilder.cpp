#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

CutInterfaceDomainRequest make_request(int marker)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/4,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/22);
    request.interface_marker = marker;
    request.isovalue = 0.0;
    request.tolerance = 1.0e-12;
    request.quadrature_policy_key = 31;
    return request;
}

} // namespace

TEST(LevelSetInterfaceBuilder, CutsLinearTriangleWithSingleSegment)
{
    const auto request = make_request(/*marker=*/17);
    const LevelSetCellCutInput input{
        .parent_cell = 5,
        .element_type = ElementType::Triangle3,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.25, 0.75, -0.25}};

    const auto result = cutLinearLevelSetCell2D(request, input);

    ASSERT_TRUE(result.supported);
    ASSERT_EQ(result.fragments.size(), 1u);
    const auto& fragment = result.fragments.front();
    EXPECT_TRUE(fragment.active());
    EXPECT_EQ(fragment.interface_marker, 17);
    EXPECT_EQ(fragment.parent_cell, 5);
    EXPECT_EQ(fragment.kind, CutInterfaceFragmentKind::Segment);
    EXPECT_EQ(fragment.degeneracy, CutInterfaceDegeneracy::None);
    EXPECT_NE(fragment.stable_id, 0u);
    ASSERT_EQ(fragment.vertices.size(), 2u);
    ASSERT_EQ(fragment.quadrature_points.size(), 1u);
    EXPECT_NEAR(fragment.measure, 0.75, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().point[0], 0.25, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().point[1], 0.375, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().normal[0], 1.0, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().normal[1], 0.0, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().weight, 0.75, 1.0e-14);
}

TEST(LevelSetInterfaceBuilder, CutsLinearQuadWithSingleSegment)
{
    LevelSetInterfaceDomain domain(make_request(/*marker=*/21));
    const LevelSetCellCutInput input{
        .parent_cell = 9,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};

    appendLinearLevelSetCellCut2D(domain, input);

    const auto summary = domain.summary();
    EXPECT_EQ(summary.fragment_count, 1u);
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.quadrature_point_count, 1u);
    EXPECT_NEAR(summary.measure, 1.0, 1.0e-14);

    const auto rules = domain.interfaceQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().provenance.parent_entity, 9);
    EXPECT_EQ(rules.front().provenance.predicate_policy_key, 31u);
    ASSERT_EQ(rules.front().points.size(), 1u);
    EXPECT_NEAR(rules.front().points.front().point[0], 0.5, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().point[1], 0.5, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().normal[0], 1.0, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().normal[1], 0.0, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().weight, 1.0, 1.0e-14);
}

TEST(LevelSetInterfaceBuilder, ReportsNoCutAndUnsupportedElement)
{
    const auto request = make_request(/*marker=*/3);
    const LevelSetCellCutInput no_cut{
        .parent_cell = 1,
        .element_type = ElementType::Triangle3,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {1.0, 2.0, 3.0}};
    const auto no_cut_result = cutLinearLevelSetCell2D(request, no_cut);
    EXPECT_TRUE(no_cut_result.supported);
    EXPECT_FALSE(no_cut_result.hasActiveFragments());
    EXPECT_EQ(no_cut_result.degeneracy, CutInterfaceDegeneracy::NoCut);

    const LevelSetCellCutInput unsupported{
        .parent_cell = 2,
        .element_type = ElementType::Tetra4,
        .node_coordinates = {},
        .level_set_values = {}};
    const auto unsupported_result = cutLinearLevelSetCell2D(request, unsupported);
    EXPECT_FALSE(unsupported_result.supported);
    EXPECT_FALSE(unsupported_result.diagnostic.empty());
}

TEST(LevelSetInterfaceBuilder, RejectsFullZeroCellAsDegenerate)
{
    const auto request = make_request(/*marker=*/8);
    const LevelSetCellCutInput input{
        .parent_cell = 6,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {0.0, 0.0, 0.0, 0.0}};

    const auto result = cutLinearLevelSetCell2D(request, input);
    EXPECT_TRUE(result.supported);
    EXPECT_FALSE(result.hasActiveFragments());
    EXPECT_EQ(result.degeneracy, CutInterfaceDegeneracy::FullZeroCell);
    EXPECT_FALSE(result.diagnostic.empty());
}
