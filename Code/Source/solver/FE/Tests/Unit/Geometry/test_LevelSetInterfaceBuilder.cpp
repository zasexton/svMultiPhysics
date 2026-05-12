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

TEST(LevelSetInterfaceBuilder, CutsLinearTetrahedronWithTriangularPatch)
{
    const auto request = make_request(/*marker=*/12);
    const LevelSetCellCutInput input{
        .parent_cell = 14,
        .element_type = ElementType::Tetra4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}},
                             {{0.0, 0.0, 1.0}}},
        .level_set_values = {-0.25, 0.75, -0.25, -0.25}};

    const auto result = cutLinearLevelSetCell3D(request, input);

    ASSERT_TRUE(result.supported);
    ASSERT_EQ(result.fragments.size(), 1u);
    const auto& fragment = result.fragments.front();
    EXPECT_TRUE(fragment.active());
    EXPECT_EQ(fragment.kind, CutInterfaceFragmentKind::Polygon);
    EXPECT_EQ(fragment.degeneracy, CutInterfaceDegeneracy::None);
    ASSERT_EQ(fragment.vertices.size(), 3u);
    ASSERT_EQ(fragment.quadrature_points.size(), 1u);
    EXPECT_NEAR(fragment.measure, 0.28125, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().point[0], 0.25, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().point[1], 0.25, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().point[2], 0.25, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().normal[0], 1.0, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().normal[1], 0.0, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().normal[2], 0.0, 1.0e-14);
    EXPECT_NEAR(fragment.quadrature_points.front().weight, 0.28125, 1.0e-14);
}

TEST(LevelSetInterfaceBuilder, CutsLinearTetrahedronWithQuadrilateralPatch)
{
    LevelSetInterfaceDomain domain(make_request(/*marker=*/13));
    const LevelSetCellCutInput input{
        .parent_cell = 15,
        .element_type = ElementType::Tetra4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}},
                             {{0.0, 0.0, 1.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};

    appendLinearLevelSetCellCut3D(domain, input);

    const auto summary = domain.summary();
    EXPECT_EQ(summary.fragment_count, 1u);
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.quadrature_point_count, 1u);
    EXPECT_NEAR(summary.measure, std::sqrt(0.125), 1.0e-14);

    const auto rules = domain.interfaceQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().provenance.parent_entity, 15);
    ASSERT_EQ(rules.front().points.size(), 1u);
    EXPECT_NEAR(rules.front().points.front().point[0], 0.25, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().point[1], 0.25, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().point[2], 0.25, 1.0e-14);
    const Real inv_sqrt2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(rules.front().points.front().normal[0], inv_sqrt2, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().normal[1], inv_sqrt2, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().normal[2], 0.0, 1.0e-14);
    EXPECT_NEAR(rules.front().points.front().weight, std::sqrt(0.125), 1.0e-14);
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

TEST(LevelSetInterfaceBuilder, ProvidesExtensionPointsForHexWedgeAndPyramid)
{
    EXPECT_TRUE(isLevelSetCellCutExtensionElement(ElementType::Hex8));
    EXPECT_TRUE(isLevelSetCellCutExtensionElement(ElementType::Hex20));
    EXPECT_TRUE(isLevelSetCellCutExtensionElement(ElementType::Wedge6));
    EXPECT_TRUE(isLevelSetCellCutExtensionElement(ElementType::Pyramid5));
    EXPECT_FALSE(isLevelSetCellCutExtensionElement(ElementType::Tetra4));

    LevelSetCellCutExtensionRegistry registry;
    const auto make_extension = [](ElementType type, const char* name) {
        LevelSetCellCutExtension extension;
        extension.element_type = type;
        extension.dimension = 3;
        extension.name = name;
        extension.cutter = [](const CutInterfaceDomainRequest& request,
                              const LevelSetCellCutInput& input) {
            LevelSetCellCutResult result;
            CutInterfaceFragment fragment;
            fragment.interface_marker = request.interface_marker;
            fragment.parent_cell = input.parent_cell;
            fragment.kind = CutInterfaceFragmentKind::Polygon;
            fragment.measure = 2.0;
            fragment.quadrature_points = {
                CutInterfaceQuadraturePoint{.point = {{0.0, 0.0, 0.0}},
                                            .parent_coordinate = {{0.0, 0.0, 0.0}},
                                            .normal = {{1.0, 0.0, 0.0}},
                                            .weight = 2.0}};
            result.fragments.push_back(fragment);
            return result;
        };
        return extension;
    };

    registry.registerCutter(make_extension(ElementType::Hex8, "hex-cutter"));
    registry.registerCutter(make_extension(ElementType::Wedge6, "wedge-cutter"));
    registry.registerCutter(make_extension(ElementType::Pyramid5, "pyramid-cutter"));

    EXPECT_TRUE(registry.hasCutter(ElementType::Hex8));
    EXPECT_TRUE(registry.hasCutter(ElementType::Wedge6));
    EXPECT_TRUE(registry.hasCutter(ElementType::Pyramid5));
    EXPECT_FALSE(registry.hasCutter(ElementType::Tetra4));

    const auto types = registry.registeredElementTypes();
    ASSERT_EQ(types.size(), 3u);
    EXPECT_EQ(types[0], ElementType::Hex8);
    EXPECT_EQ(types[1], ElementType::Wedge6);
    EXPECT_EQ(types[2], ElementType::Pyramid5);

    const LevelSetCellCutInput input{
        .parent_cell = 99,
        .element_type = ElementType::Hex8,
        .node_coordinates = {},
        .level_set_values = {}};
    const auto result = registry.cut(make_request(/*marker=*/44), input);
    ASSERT_TRUE(result.supported);
    ASSERT_EQ(result.fragments.size(), 1u);
    EXPECT_EQ(result.fragments.front().interface_marker, 44);
    EXPECT_EQ(result.fragments.front().parent_cell, 99);
    EXPECT_DOUBLE_EQ(result.fragments.front().measure, 2.0);

    const LevelSetCellCutInput missing{
        .parent_cell = 100,
        .element_type = ElementType::Hex27,
        .node_coordinates = {},
        .level_set_values = {}};
    const auto missing_result = registry.cut(make_request(/*marker=*/45), missing);
    EXPECT_FALSE(missing_result.supported);
    EXPECT_FALSE(missing_result.diagnostic.empty());
}
