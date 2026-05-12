#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

#include <array>
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

void expect_normal_near(const std::array<Real, 3>& actual,
                        const std::array<Real, 3>& expected,
                        Real tolerance)
{
    EXPECT_NEAR(actual[0], expected[0], tolerance);
    EXPECT_NEAR(actual[1], expected[1], tolerance);
    EXPECT_NEAR(actual[2], expected[2], tolerance);
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
    EXPECT_NEAR(fragment.negative_volume_fraction, 0.4375, 1.0e-14);
    EXPECT_NEAR(fragment.positive_volume_fraction, 0.5625, 1.0e-14);
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
    ASSERT_EQ(domain.fragments().size(), 1u);
    EXPECT_NEAR(domain.fragments().front().negative_volume_fraction, 0.5, 1.0e-14);
    EXPECT_NEAR(domain.fragments().front().positive_volume_fraction, 0.5, 1.0e-14);

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
    EXPECT_NEAR(fragment.negative_volume_fraction, 0.578125, 1.0e-14);
    EXPECT_NEAR(fragment.positive_volume_fraction, 0.421875, 1.0e-14);
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
    ASSERT_EQ(domain.fragments().size(), 1u);
    EXPECT_NEAR(domain.fragments().front().negative_volume_fraction, 0.5, 1.0e-14);
    EXPECT_NEAR(domain.fragments().front().positive_volume_fraction, 0.5, 1.0e-14);

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

TEST(LevelSetInterfaceBuilder, GeneratedNormalsMatchLinearLevelSetGradients)
{
    const auto request = make_request(/*marker=*/24);
    const LevelSetCellCutInput quad_input{
        .parent_cell = 16,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-1.25, -0.25, 1.75, 0.75}};
    const auto quad_result = cutLinearLevelSetCell2D(request, quad_input);
    ASSERT_TRUE(quad_result.hasActiveFragments());
    ASSERT_EQ(quad_result.fragments.size(), 1u);
    const Real inv_sqrt5 = Real{1.0} / std::sqrt(Real{5.0});
    const std::array<Real, 3> expected_quad_normal{
        {inv_sqrt5, Real{2.0} * inv_sqrt5, Real{0.0}}};
    expect_normal_near(quad_result.fragments.front().normal,
                       expected_quad_normal,
                       1.0e-14);
    expect_normal_near(quad_result.fragments.front().quadrature_points.front().normal,
                       expected_quad_normal,
                       1.0e-14);

    const LevelSetCellCutInput tetra_input{
        .parent_cell = 17,
        .element_type = ElementType::Tetra4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}},
                             {{0.0, 0.0, 1.0}}},
        .level_set_values = {-1.25, -0.25, 0.75, 1.75}};
    const auto tetra_result = cutLinearLevelSetCell3D(request, tetra_input);
    ASSERT_TRUE(tetra_result.hasActiveFragments());
    ASSERT_EQ(tetra_result.fragments.size(), 1u);
    const Real inv_sqrt14 = Real{1.0} / std::sqrt(Real{14.0});
    const std::array<Real, 3> expected_tetra_normal{
        {inv_sqrt14, Real{2.0} * inv_sqrt14, Real{3.0} * inv_sqrt14}};
    expect_normal_near(tetra_result.fragments.front().normal,
                       expected_tetra_normal,
                       1.0e-14);
    expect_normal_near(tetra_result.fragments.front().quadrature_points.front().normal,
                       expected_tetra_normal,
                       1.0e-14);
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

TEST(LevelSetInterfaceBuilder, ClassifiesCutEdgeCases)
{
    auto request = make_request(/*marker=*/30);

    const LevelSetCellCutInput tetra_no_cut{
        .parent_cell = 1,
        .element_type = ElementType::Tetra4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}},
                             {{0.0, 0.0, 1.0}}},
        .level_set_values = {1.0, 2.0, 3.0, 4.0}};
    const auto no_cut = cutLinearLevelSetCell3D(request, tetra_no_cut);
    EXPECT_FALSE(no_cut.hasActiveFragments());
    EXPECT_EQ(no_cut.degeneracy, CutInterfaceDegeneracy::NoCut);

    LevelSetCellCutInput tetra_full_zero = tetra_no_cut;
    tetra_full_zero.level_set_values = {0.0, 0.0, 0.0, 0.0};
    const auto full_zero = cutLinearLevelSetCell3D(request, tetra_full_zero);
    EXPECT_FALSE(full_zero.hasActiveFragments());
    EXPECT_EQ(full_zero.degeneracy, CutInterfaceDegeneracy::FullZeroCell);
    EXPECT_FALSE(full_zero.diagnostic.empty());

    const LevelSetCellCutInput vertex_touch{
        .parent_cell = 2,
        .element_type = ElementType::Triangle3,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {0.0, 1.0, 1.0}};
    const auto vertex_touch_result = cutLinearLevelSetCell2D(request, vertex_touch);
    EXPECT_FALSE(vertex_touch_result.hasActiveFragments());
    EXPECT_EQ(vertex_touch_result.degeneracy, CutInterfaceDegeneracy::VertexTouch);

    LevelSetCellCutInput vertex_cut = vertex_touch;
    vertex_cut.level_set_values = {0.0, 1.0, -1.0};
    const auto vertex_cut_result = cutLinearLevelSetCell2D(request, vertex_cut);
    ASSERT_TRUE(vertex_cut_result.hasActiveFragments());
    ASSERT_EQ(vertex_cut_result.fragments.size(), 1u);
    EXPECT_EQ(vertex_cut_result.fragments.front().degeneracy,
              CutInterfaceDegeneracy::VertexTouch);

    const LevelSetCellCutInput edge_touch{
        .parent_cell = 3,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {0.0, 0.0, 1.0, 1.0}};
    const auto edge_touch_result = cutLinearLevelSetCell2D(request, edge_touch);
    EXPECT_FALSE(edge_touch_result.hasActiveFragments());
    EXPECT_EQ(edge_touch_result.degeneracy, CutInterfaceDegeneracy::EdgeTouch);

    request.tolerance = 1.0e-12;
    const LevelSetCellCutInput nearly_tangent{
        .parent_cell = 4,
        .element_type = ElementType::Triangle3,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-1.0e-7, 1.0, 1.0}};
    const auto nearly_tangent_result = cutLinearLevelSetCell2D(request, nearly_tangent);
    ASSERT_TRUE(nearly_tangent_result.hasActiveFragments());
    ASSERT_EQ(nearly_tangent_result.fragments.size(), 1u);
    EXPECT_EQ(nearly_tangent_result.fragments.front().degeneracy,
              CutInterfaceDegeneracy::NearlyTangent);

    const LevelSetCellCutInput small_physical_fragment{
        .parent_cell = 5,
        .element_type = ElementType::Triangle3,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0e-13, 0.0, 0.0}},
                             {{0.0, 1.0e-13, 0.0}}},
        .level_set_values = {-1.0, 1.0, -1.0}};
    const auto small_fragment_result =
        cutLinearLevelSetCell2D(request, small_physical_fragment);
    EXPECT_FALSE(small_fragment_result.hasActiveFragments());
    EXPECT_EQ(small_fragment_result.degeneracy, CutInterfaceDegeneracy::SmallFragment);
    EXPECT_FALSE(small_fragment_result.diagnostic.empty());
}

TEST(LevelSetInterfaceBuilder, SerialGeneratedInterfaceFragmentCounts)
{
    LevelSetInterfaceDomain triangle_domain(make_request(/*marker=*/60));
    appendLinearLevelSetCellCut2D(
        triangle_domain,
        LevelSetCellCutInput{.parent_cell = 1,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25}});
    appendLinearLevelSetCellCut2D(
        triangle_domain,
        LevelSetCellCutInput{.parent_cell = 2,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {1.0, 1.0, 1.0}});
    auto summary = triangle_domain.summary();
    EXPECT_EQ(summary.fragment_count, 1u);
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.quadrature_point_count, 1u);
    EXPECT_NEAR(summary.measure, 0.75, 1.0e-14);

    LevelSetInterfaceDomain quad_domain(make_request(/*marker=*/61));
    appendLinearLevelSetCellCut2D(
        quad_domain,
        LevelSetCellCutInput{.parent_cell = 3,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    appendLinearLevelSetCellCut2D(
        quad_domain,
        LevelSetCellCutInput{.parent_cell = 4,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.25, 0.75, 0.75, -0.25}});
    summary = quad_domain.summary();
    EXPECT_EQ(summary.fragment_count, 2u);
    EXPECT_EQ(summary.active_fragment_count, 2u);
    EXPECT_EQ(summary.quadrature_point_count, 2u);
    EXPECT_NEAR(summary.measure, 2.0, 1.0e-14);

    LevelSetInterfaceDomain tetra_domain(make_request(/*marker=*/62));
    appendLinearLevelSetCellCut3D(
        tetra_domain,
        LevelSetCellCutInput{.parent_cell = 5,
                             .element_type = ElementType::Tetra4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}},
                                                  {{0.0, 0.0, 1.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25, -0.25}});
    appendLinearLevelSetCellCut3D(
        tetra_domain,
        LevelSetCellCutInput{.parent_cell = 6,
                             .element_type = ElementType::Tetra4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}},
                                                  {{0.0, 0.0, 1.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    summary = tetra_domain.summary();
    EXPECT_EQ(summary.fragment_count, 2u);
    EXPECT_EQ(summary.active_fragment_count, 2u);
    EXPECT_EQ(summary.quadrature_point_count, 2u);
    EXPECT_NEAR(summary.measure, 0.28125 + std::sqrt(0.125), 1.0e-14);
}
