#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

void expect_observation(const LevelSetCellCutResult& result,
                        LinearCornerStrictBranch expected)
{
    EXPECT_EQ(result.construction_observation, expected);
    for (const auto& fragment : result.fragments) {
        EXPECT_EQ(fragment.construction_observation, expected);
    }
    for (const auto& region : result.volume_regions) {
        EXPECT_EQ(region.construction_observation, expected);
    }
}

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

TEST(ProducerObservation, FiniteDyadicGradientNormOverflowStampsEveryRecord)
{
    const auto request = make_request(70);
    const Real magnitude = std::ldexp(Real{1.0}, 600);
    ASSERT_TRUE(std::isfinite(magnitude));
    ASSERT_TRUE(std::isfinite(Real{2.0} * magnitude));
    for (const Real sign : {-1.0, 1.0}) {
        SCOPED_TRACE(sign);
        const LevelSetCellCutInput input{
            .parent_cell = 0,
            .element_type = ElementType::Triangle3,
            .node_coordinates = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}},
            .level_set_values = {-sign * magnitude, sign * magnitude, sign * magnitude}};
        for (const auto value : input.level_set_values) {
            ASSERT_TRUE(std::isfinite(value));
        }
        const Real crossing_denominator = input.level_set_values[0] - input.level_set_values[1];
        ASSERT_TRUE(std::isfinite(crossing_denominator));
        EXPECT_EQ(input.level_set_values[0] / crossing_denominator, 0.5);

        const auto result = cutLinearLevelSetCell2D(request, input);
        ASSERT_TRUE(result.supported);
        ASSERT_EQ(result.fragments.size(), 1u);
        ASSERT_EQ(result.volume_regions.size(), 2u);
        const auto& fragment = result.fragments.front();
        ASSERT_EQ(fragment.vertices.size(), 2u);
        EXPECT_EQ(fragment.vertices[0].point, (std::array<Real, 3>{{0.5, 0, 0}}));
        EXPECT_EQ(fragment.vertices[1].point, (std::array<Real, 3>{{0, 0.5, 0}}));
        EXPECT_EQ(fragment.measure, std::sqrt(0.5));
        EXPECT_EQ(fragment.negative_volume_fraction, sign > 0 ? 0.25 : 0.75);
        EXPECT_EQ(fragment.positive_volume_fraction, sign > 0 ? 0.75 : 0.25);
        EXPECT_EQ(result.volume_regions[0].side, geometry::CutIntegrationSide::Negative);
        EXPECT_EQ(result.volume_regions[1].side, geometry::CutIntegrationSide::Positive);
        EXPECT_EQ(result.volume_regions[0].measure, sign > 0 ? 0.125 : 0.375);
        EXPECT_EQ(result.volume_regions[1].measure, sign > 0 ? 0.375 : 0.125);
        for (const auto& region : result.volume_regions) {
            EXPECT_FALSE(region.quadrature_points.empty());
            EXPECT_FALSE(region.reference_subcells.empty());
        }
        // The existing overflowing gradient normalization loses its orientation
        // direction under either sign. This observation-only fix must not repair it.
        const Real normal_component = 0.5 * (1.0 / std::sqrt(0.5));
        EXPECT_EQ(fragment.normal, (std::array<Real, 3>{{normal_component, normal_component, 0}}));
        expect_observation(result, LinearCornerStrictBranch::ModifiedOrUnresolved);
    }
}

TEST(ProducerObservation, DistinctRootCollapseSurvivesFullPhaseReplacement)
{
    const auto request = make_request(70);
    for (const Real sign : {-1.0, 1.0}) {
        const LevelSetCellCutInput input{
            .parent_cell = 0,
            .element_type = ElementType::Tetra4,
            .node_coordinates = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}},
            .level_set_values = {-sign, sign * 1e14, sign * 1e14, sign * 1e14}};
        const auto result = cutLinearLevelSetCell3D(request, input);
        ASSERT_TRUE(result.supported);
        ASSERT_TRUE(result.fragments.empty());
        ASSERT_EQ(result.volume_regions.size(), 1u);
        EXPECT_TRUE(result.volume_regions.front().full_cell_equivalent);
        EXPECT_EQ(result.volume_regions.front().volume_fraction, 1.0);
        EXPECT_EQ(result.volume_regions.front().measure, 1.0 / 6.0);
        expect_observation(result, LinearCornerStrictBranch::ModifiedOrUnresolved);
    }
}

TEST(ProducerObservation, DyadicAndFullPhaseControlsRemainUnchecked)
{
    const auto request = make_request(70);
    for (const auto type : {ElementType::Triangle3, ElementType::Tetra4}) {
        for (const int negative_count : {0, 1, 2}) {
            for (const Real sign : {-1.0, 1.0}) {
                LevelSetCellCutInput input{
                    .parent_cell = 0, .element_type = type,
                    .node_coordinates = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}},
                    .level_set_values = {sign, sign, sign}};
                if (type == ElementType::Tetra4) {
                    input.node_coordinates.push_back({{0, 0, 1}});
                    input.level_set_values.push_back(sign);
                }
                for (int i = 0; i < negative_count; ++i) {
                    input.level_set_values[static_cast<std::size_t>(i)] = -sign;
                }
                const auto result = type == ElementType::Triangle3
                    ? cutLinearLevelSetCell2D(request, input)
                    : cutLinearLevelSetCell3D(request, input);
                ASSERT_TRUE(result.supported);
                ASSERT_EQ(result.volume_regions.size(), negative_count == 0 ? 1u : 2u);
                expect_observation(result, LinearCornerStrictBranch::Unchecked);
                Real volume = 0;
                for (const auto& region : result.volume_regions) {
                    volume += region.measure;
                    EXPECT_GT(region.quadrature_points.size(), 0u);
                }
                EXPECT_NEAR(volume, type == ElementType::Triangle3 ? 0.5 : 1.0 / 6.0, 1e-15);
                for (const auto& fragment : result.fragments) {
                    for (const auto& vertex : fragment.vertices) {
                        for (const auto coordinate : vertex.point) {
                            EXPECT_TRUE(coordinate == 0.0 || coordinate == 0.5);
                        }
                    }
                    EXPECT_EQ(fragment.quadrature_points.front().weight, fragment.measure);
                }
                if (type == ElementType::Triangle3 && negative_count == 1) {
                    ASSERT_EQ(result.fragments.size(), 1u);
                    EXPECT_EQ(result.fragments.front().vertices[0].point,
                              (std::array<Real, 3>{{0.5, 0, 0}}));
                    EXPECT_EQ(result.fragments.front().vertices[1].point,
                              (std::array<Real, 3>{{0, 0.5, 0}}));
                    EXPECT_EQ(result.fragments.front().measure, std::sqrt(0.5));
                    EXPECT_EQ(result.volume_regions[0].measure, sign > 0 ? 0.125 : 0.375);
                }
            }
        }
    }
}

TEST(ProducerObservation, FloatingRootRepeatsAndZeroBranchesStayUnresolved)
{
    auto request = make_request(70);
    for (const auto& values : {std::vector<Real>{0.25, -1, -1, -1},
                              std::vector<Real>{0.25, 1, -1, -1},
                              std::vector<Real>{0, 0, 0, 1}}) {
        LevelSetCellCutInput input{
            .parent_cell = 0, .element_type = ElementType::Tetra4,
            .node_coordinates = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}},
            .level_set_values = values};
        request.aligned_zero_interface_parent_side = geometry::CutIntegrationSide::Positive;
        const auto result = cutLinearLevelSetCell3D(request, input);
        ASSERT_TRUE(result.supported);
        ASSERT_EQ(result.fragments.size(), 1u);
        expect_observation(result, LinearCornerStrictBranch::ModifiedOrUnresolved);
        if (values[0] == 0.25 && values[1] == -1) {
            Real minimum_root = 1.0, maximum_root = 0.0;
            std::size_t appearances = 0;
            for (const auto& region : result.volume_regions) {
                for (const auto& simplex : region.reference_subcells) {
                    for (std::size_t i = 0; i < simplex.vertex_count; ++i) {
                        const auto& point = simplex.vertices[i];
                        // Face (0,1,2) crosses (2,0); face (0,2,3)
                        // crosses (0,2). The x-axis edge is not reversed.
                        if (point[0] == 0 && point[2] == 0 && point[1] > 0 && point[1] < 0.3) {
                            minimum_root = std::min(minimum_root, point[1]);
                            maximum_root = std::max(maximum_root, point[1]);
                            ++appearances;
                        }
                    }
                }
            }
            ASSERT_GT(appearances, 1u);
            EXPECT_GT(maximum_root, minimum_root);
            RecordProperty("floating_original_edge_root_residual",
                           ::testing::PrintToString(maximum_root - minimum_root));
        }
    }
    for (const auto& values : {std::vector<Real>{0, -1, 1},
                              std::vector<Real>{0, 0, 1},
                              std::vector<Real>{0.75e-12, -1, 1},
                              std::vector<Real>{0, 0, 0}}) {
        const LevelSetCellCutInput input{
            .parent_cell = 0, .element_type = ElementType::Triangle3,
            .node_coordinates = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}},
            .level_set_values = values};
        const auto result = cutLinearLevelSetCell2D(request, input);
        ASSERT_TRUE(result.supported);
        expect_observation(result, LinearCornerStrictBranch::ModifiedOrUnresolved);
    }
    expect_observation(LevelSetCellCutResult{}, LinearCornerStrictBranch::Unchecked);
    EXPECT_EQ(CutInterfaceFragment{}.construction_observation, LinearCornerStrictBranch::Unchecked);
    EXPECT_EQ(CutInterfaceVolumeRegion{}.construction_observation, LinearCornerStrictBranch::Unchecked);
    const LevelSetCellCutInput triangle{
        .parent_cell = 0, .element_type = ElementType::Triangle3,
        .node_coordinates = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}},
        .level_set_values = {0.25, -1, 1}};
    const auto triangle_result = cutLinearLevelSetCell2D(request, triangle);
    ASSERT_EQ(triangle_result.fragments.size(), 1u);
    ASSERT_EQ(triangle_result.volume_regions.size(), 2u);
    // This path has no compared repeated-root appearances. It remains
    // unassessed, not a successful numerical-margin certificate.
    expect_observation(triangle_result, LinearCornerStrictBranch::Unchecked);
}

TEST(ProducerObservation, PositivePieceFilteringStampsBothPhasesAndInterface)
{
    auto request = make_request(70);
    request.tolerance = 0.05;
    for (const Real sign : {-1.0, 1.0}) {
        const LevelSetCellCutInput input{
            .parent_cell = 0,
            .element_type = ElementType::Triangle3,
            .node_coordinates = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}},
            .level_set_values = {sign * 0.1, -sign * 0.1, sign * 1e6}};
        const auto result = cutLinearLevelSetCell2D(request, input);
        ASSERT_TRUE(result.supported);
        ASSERT_EQ(result.fragments.size(), 1u);
        ASSERT_EQ(result.volume_regions.size(), 2u);
        const auto side = sign > 0 ? geometry::CutIntegrationSide::Negative
                                   : geometry::CutIntegrationSide::Positive;
        std::size_t selected_regions = 0;
        for (const auto& region : result.volume_regions) {
            if (region.side == side) {
                ++selected_regions;
                EXPECT_GT(region.measure, 0.0);
                EXPECT_LT(region.measure, request.tolerance * request.tolerance);
                EXPECT_TRUE(region.reference_subcells.empty());
                EXPECT_TRUE(region.quadrature_points.empty());
            }
        }
        ASSERT_EQ(selected_regions, 1u);
        expect_observation(result, LinearCornerStrictBranch::ModifiedOrUnresolved);
    }
}

void expect_normal_near(const std::array<Real, 3>& actual,
                        const std::array<Real, 3>& expected,
                        Real tolerance)
{
    EXPECT_NEAR(actual[0], expected[0], tolerance);
    EXPECT_NEAR(actual[1], expected[1], tolerance);
    EXPECT_NEAR(actual[2], expected[2], tolerance);
}

Real circle_phi(const std::array<Real, 3>& p, Real radius)
{
    return std::sqrt(p[0] * p[0] + p[1] * p[1]) - radius;
}

Real sphere_phi(const std::array<Real, 3>& p, Real radius)
{
    return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - radius;
}

std::array<Real, 3> normalized_position(const std::array<Real, 3>& p)
{
    const Real norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    return {{p[0] / norm, p[1] / norm, p[2] / norm}};
}

Real vector_error(const std::array<Real, 3>& a, const std::array<Real, 3>& b)
{
    const Real dx = a[0] - b[0];
    const Real dy = a[1] - b[1];
    const Real dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Real reference_tetrahedron_measure(
    const CutInterfaceReferenceSimplex& simplex)
{
    if (simplex.vertex_count != 4u) {
        return 0.0;
    }
    const auto& a = simplex.vertices[0];
    const auto& b = simplex.vertices[1];
    const auto& c = simplex.vertices[2];
    const auto& d = simplex.vertices[3];
    const std::array<Real, 3> ab{{
        b[0] - a[0], b[1] - a[1], b[2] - a[2]}};
    const std::array<Real, 3> ac{{
        c[0] - a[0], c[1] - a[1], c[2] - a[2]}};
    const std::array<Real, 3> ad{{
        d[0] - a[0], d[1] - a[1], d[2] - a[2]}};
    const Real determinant =
        ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) -
        ab[1] * (ac[0] * ad[2] - ac[2] * ad[0]) +
        ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
    return std::abs(determinant) * simplex.measure_scale / Real{6.0};
}

Real reference_triangle_measure(
    const CutInterfaceReferenceSimplex& simplex)
{
    if (simplex.vertex_count != 3u) {
        return 0.0;
    }
    const auto& a = simplex.vertices[0];
    const auto& b = simplex.vertices[1];
    const auto& c = simplex.vertices[2];
    const std::array<Real, 3> ab{{
        b[0] - a[0], b[1] - a[1], b[2] - a[2]}};
    const std::array<Real, 3> ac{{
        c[0] - a[0], c[1] - a[1], c[2] - a[2]}};
    const std::array<Real, 3> product{{
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0]}};
    const Real magnitude = std::sqrt(
        product[0] * product[0] +
        product[1] * product[1] +
        product[2] * product[2]);
    return Real{0.5} * magnitude * simplex.measure_scale;
}

Real integrate_volume_weight(const svmp::FE::geometry::CutQuadratureRule& rule)
{
    Real value = 0.0;
    for (const auto& point : rule.points) {
        value += point.weight;
    }
    return value;
}

void expect_volume_rule_constant_exact(
    const svmp::FE::geometry::CutQuadratureRule& rule,
    Real tolerance)
{
    EXPECT_NEAR(integrate_volume_weight(rule), rule.measure, tolerance);
}

Real integrate_volume_coordinate(const svmp::FE::geometry::CutQuadratureRule& rule,
                                 std::size_t component)
{
    Real value = 0.0;
    for (const auto& point : rule.points) {
        value += point.weight * point.point[component];
    }
    return value;
}

struct InterfaceApproximation {
    Real measure{0.0};
    Real weighted_normal_error{0.0};
    std::size_t active_fragments{0};
};

void accumulate_fragment_metrics(InterfaceApproximation& metrics,
                                 const CutInterfaceFragment& fragment)
{
    if (!fragment.active()) {
        return;
    }
    metrics.measure += fragment.measure;
    ++metrics.active_fragments;
    for (const auto& qp : fragment.quadrature_points) {
        metrics.weighted_normal_error +=
            qp.weight * vector_error(qp.normal, normalized_position(qp.point));
    }
}

InterfaceApproximation approximate_circle(int cells_per_axis, Real radius)
{
    const auto request = make_request(/*marker=*/70);
    const Real min_coord = -1.0;
    const Real h = 2.0 / static_cast<Real>(cells_per_axis);
    InterfaceApproximation metrics;
    MeshIndex parent_cell = 0;

    const auto append_triangle = [&](const std::array<Real, 3>& a,
                                     const std::array<Real, 3>& b,
                                     const std::array<Real, 3>& c) {
        const LevelSetCellCutInput input{
            .parent_cell = parent_cell++,
            .element_type = ElementType::Triangle3,
            .node_coordinates = {a, b, c},
            .level_set_values = {
                circle_phi(a, radius),
                circle_phi(b, radius),
                circle_phi(c, radius)}};
        const auto result = cutLinearLevelSetCell2D(request, input);
        for (const auto& fragment : result.fragments) {
            accumulate_fragment_metrics(metrics, fragment);
        }
    };

    for (int j = 0; j < cells_per_axis; ++j) {
        for (int i = 0; i < cells_per_axis; ++i) {
            const Real x0 = min_coord + h * static_cast<Real>(i);
            const Real y0 = min_coord + h * static_cast<Real>(j);
            const Real x1 = x0 + h;
            const Real y1 = y0 + h;
            const std::array<Real, 3> p00{{x0, y0, 0.0}};
            const std::array<Real, 3> p10{{x1, y0, 0.0}};
            const std::array<Real, 3> p11{{x1, y1, 0.0}};
            const std::array<Real, 3> p01{{x0, y1, 0.0}};
            append_triangle(p00, p10, p11);
            append_triangle(p00, p11, p01);
        }
    }

    if (metrics.measure > 0.0) {
        metrics.weighted_normal_error /= metrics.measure;
    }
    return metrics;
}

InterfaceApproximation approximate_sphere(int cells_per_axis, Real radius)
{
    const auto request = make_request(/*marker=*/71);
    const Real min_coord = -1.0;
    const Real h = 2.0 / static_cast<Real>(cells_per_axis);
    InterfaceApproximation metrics;
    MeshIndex parent_cell = 0;

    const auto append_tetra = [&](const std::array<Real, 3>& a,
                                  const std::array<Real, 3>& b,
                                  const std::array<Real, 3>& c,
                                  const std::array<Real, 3>& d) {
        const LevelSetCellCutInput input{
            .parent_cell = parent_cell++,
            .element_type = ElementType::Tetra4,
            .node_coordinates = {a, b, c, d},
            .level_set_values = {
                sphere_phi(a, radius),
                sphere_phi(b, radius),
                sphere_phi(c, radius),
                sphere_phi(d, radius)}};
        const auto result = cutLinearLevelSetCell3D(request, input);
        for (const auto& fragment : result.fragments) {
            accumulate_fragment_metrics(metrics, fragment);
        }
    };

    for (int k = 0; k < cells_per_axis; ++k) {
        for (int j = 0; j < cells_per_axis; ++j) {
            for (int i = 0; i < cells_per_axis; ++i) {
                const Real x0 = min_coord + h * static_cast<Real>(i);
                const Real y0 = min_coord + h * static_cast<Real>(j);
                const Real z0 = min_coord + h * static_cast<Real>(k);
                const Real x1 = x0 + h;
                const Real y1 = y0 + h;
                const Real z1 = z0 + h;

                const std::array<Real, 3> v000{{x0, y0, z0}};
                const std::array<Real, 3> v100{{x1, y0, z0}};
                const std::array<Real, 3> v010{{x0, y1, z0}};
                const std::array<Real, 3> v110{{x1, y1, z0}};
                const std::array<Real, 3> v001{{x0, y0, z1}};
                const std::array<Real, 3> v101{{x1, y0, z1}};
                const std::array<Real, 3> v011{{x0, y1, z1}};
                const std::array<Real, 3> v111{{x1, y1, z1}};

                append_tetra(v000, v100, v110, v111);
                append_tetra(v000, v110, v010, v111);
                append_tetra(v000, v010, v011, v111);
                append_tetra(v000, v011, v001, v111);
                append_tetra(v000, v001, v101, v111);
                append_tetra(v000, v101, v100, v111);
            }
        }
    }

    if (metrics.measure > 0.0) {
        metrics.weighted_normal_error /= metrics.measure;
    }
    return metrics;
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

TEST(LevelSetInterfaceBuilder, CircleInterfaceLengthConvergesUnderRefinement)
{
    const Real radius = 0.53;
    const Real exact_length = 2.0 * M_PI * radius;
    const auto coarse = approximate_circle(/*cells_per_axis=*/16, radius);
    const auto fine = approximate_circle(/*cells_per_axis=*/32, radius);

    ASSERT_GT(coarse.active_fragments, 0u);
    ASSERT_GT(fine.active_fragments, coarse.active_fragments);
    const Real coarse_error = std::abs(coarse.measure - exact_length);
    const Real fine_error = std::abs(fine.measure - exact_length);
    EXPECT_LT(fine_error, 0.65 * coarse_error);
    EXPECT_LT(fine_error / exact_length, 0.015);
}

TEST(LevelSetInterfaceBuilder, SphereInterfaceAreaConvergesUnderRefinement)
{
    const Real radius = 0.53;
    const Real exact_area = 4.0 * M_PI * radius * radius;
    const auto coarse = approximate_sphere(/*cells_per_axis=*/8, radius);
    const auto fine = approximate_sphere(/*cells_per_axis=*/12, radius);

    ASSERT_GT(coarse.active_fragments, 0u);
    ASSERT_GT(fine.active_fragments, coarse.active_fragments);
    const Real coarse_error = std::abs(coarse.measure - exact_area);
    const Real fine_error = std::abs(fine.measure - exact_area);
    EXPECT_LT(fine_error, 0.85 * coarse_error);
    EXPECT_LT(fine_error / exact_area, 0.05);
}

TEST(LevelSetInterfaceBuilder, CurvedInterfaceNormalsConvergeToAnalyticNormals)
{
    const Real radius = 0.53;
    const auto circle_coarse = approximate_circle(/*cells_per_axis=*/16, radius);
    const auto circle_fine = approximate_circle(/*cells_per_axis=*/32, radius);
    const auto sphere_coarse = approximate_sphere(/*cells_per_axis=*/8, radius);
    const auto sphere_fine = approximate_sphere(/*cells_per_axis=*/12, radius);

    EXPECT_LT(circle_fine.weighted_normal_error,
              0.70 * circle_coarse.weighted_normal_error);
    EXPECT_LT(circle_fine.weighted_normal_error, 0.035);
    EXPECT_LT(sphere_fine.weighted_normal_error,
              0.90 * sphere_coarse.weighted_normal_error);
    EXPECT_LT(sphere_fine.weighted_normal_error, 0.09);
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

    for (const auto element_type :
         {ElementType::Hex8, ElementType::Wedge6, ElementType::Pyramid5}) {
        const LevelSetCellCutInput input{
            .parent_cell = 98,
            .element_type = element_type,
            .node_coordinates = {},
            .level_set_values = {}};
        const auto direct_result =
            cutLinearLevelSetCell3D(make_request(/*marker=*/43), input);
        EXPECT_FALSE(direct_result.supported);
        EXPECT_EQ(direct_result.degeneracy, CutInterfaceDegeneracy::NoCut);
        EXPECT_FALSE(direct_result.diagnostic.empty());
    }

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

TEST(LevelSetInterfaceBuilder,
     PublishesAlignedFacetFromRequestedParentSideExactlyOnce)
{
    constexpr std::array<geometry::CutIntegrationSide, 2> parent_sides{{
        geometry::CutIntegrationSide::Negative,
        geometry::CutIntegrationSide::Positive,
    }};
    int marker = 130;
    for (const auto parent_side : parent_sides) {
        auto request = make_request(marker++);
        request.aligned_zero_interface_parent_side = parent_side;
        const Real parent_value =
            parent_side == geometry::CutIntegrationSide::Negative
                ? Real{-1.0}
                : Real{1.0};
        const Real other_value = -parent_value;

        LevelSetCellCutInput triangle{
            .parent_cell = 20,
            .element_type = ElementType::Triangle3,
            .node_coordinates = {{{0.0, 0.0, 0.0}},
                                 {{1.0, 0.0, 0.0}},
                                 {{0.0, 1.0, 0.0}}},
            .level_set_values = {0.0, 0.0, parent_value}};
        const auto triangle_owner =
            cutLinearLevelSetCell2D(request, triangle);
        ASSERT_EQ(triangle_owner.fragments.size(), 1u);
        ASSERT_EQ(triangle_owner.volume_regions.size(), 1u);
        EXPECT_EQ(triangle_owner.fragments.front().degeneracy,
                  CutInterfaceDegeneracy::EdgeTouch);
        EXPECT_NEAR(triangle_owner.fragments.front().measure, 1.0, 1.0e-14);
        EXPECT_EQ(triangle_owner.volume_regions.front().side, parent_side);
        EXPECT_TRUE(
            triangle_owner.volume_regions.front().full_cell_equivalent);
        EXPECT_NEAR(triangle_owner.volume_regions.front().volume_fraction,
                    1.0,
                    1.0e-14);
        triangle.parent_cell = 21;
        triangle.level_set_values = {0.0, 0.0, other_value};
        const auto triangle_other =
            cutLinearLevelSetCell2D(request, triangle);
        EXPECT_FALSE(triangle_other.hasActiveFragments());
        ASSERT_EQ(triangle_other.volume_regions.size(), 1u);
        EXPECT_NE(triangle_other.volume_regions.front().side, parent_side);

        LevelSetCellCutInput tetrahedron{
            .parent_cell = 22,
            .element_type = ElementType::Tetra4,
            .node_coordinates = {{{0.0, 0.0, 0.0}},
                                 {{1.0, 0.0, 0.0}},
                                 {{0.0, 1.0, 0.0}},
                                 {{0.0, 0.0, 1.0}}},
            .level_set_values = {0.0, 0.0, 0.0, parent_value}};
        const auto tetrahedron_owner =
            cutLinearLevelSetCell3D(request, tetrahedron);
        ASSERT_EQ(tetrahedron_owner.fragments.size(), 1u);
        ASSERT_EQ(tetrahedron_owner.volume_regions.size(), 1u);
        EXPECT_EQ(tetrahedron_owner.fragments.front().degeneracy,
                  CutInterfaceDegeneracy::EdgeTouch);
        EXPECT_NEAR(tetrahedron_owner.fragments.front().measure, 0.5, 1.0e-14);
        EXPECT_EQ(tetrahedron_owner.volume_regions.front().side, parent_side);
        EXPECT_TRUE(
            tetrahedron_owner.volume_regions.front().full_cell_equivalent);
        EXPECT_NEAR(tetrahedron_owner.volume_regions.front().volume_fraction,
                    1.0,
                    1.0e-14);
        EXPECT_NEAR(tetrahedron_owner.volume_regions.front().measure,
                    1.0 / 6.0,
                    1.0e-14);
        Real reference_subcell_measure = 0.0;
        for (const auto& subcell :
             tetrahedron_owner.volume_regions.front().reference_subcells) {
            reference_subcell_measure +=
                reference_tetrahedron_measure(subcell);
        }
        EXPECT_NEAR(reference_subcell_measure, 1.0 / 6.0, 1.0e-14);
        tetrahedron.parent_cell = 23;
        tetrahedron.level_set_values = {0.0, 0.0, 0.0, other_value};
        const auto tetrahedron_other =
            cutLinearLevelSetCell3D(request, tetrahedron);
        EXPECT_FALSE(tetrahedron_other.hasActiveFragments());
        ASSERT_EQ(tetrahedron_other.volume_regions.size(), 1u);
        EXPECT_NE(tetrahedron_other.volume_regions.front().side, parent_side);
    }

    RecordProperty("aligned_facet_parent_side_case_count", 4);
}

TEST(LevelSetInterfaceBuilder, PreservesSmallVolumeFractionsNearVertexAndEdge)
{
    constexpr Real eps = 1.0e-7;
    const Real t = eps / (Real{1.0} + eps);

    LevelSetInterfaceDomain vertex_domain(make_request(/*marker=*/31));
    appendLinearLevelSetCellCut2D(
        vertex_domain,
        LevelSetCellCutInput{.parent_cell = 6,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-eps, 1.0, 1.0}});

    auto summary = vertex_domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.active_volume_region_count, 2u);
    const Real expected_vertex_volume = Real{0.5} * t * t;
    EXPECT_GT(summary.negative_volume_measure, 0.0);
    EXPECT_NEAR(summary.negative_volume_measure,
                expected_vertex_volume,
                1.0e-20);
    ASSERT_EQ(vertex_domain.fragments().size(), 1u);
    EXPECT_EQ(vertex_domain.fragments().front().degeneracy,
              CutInterfaceDegeneracy::NearlyTangent);

    auto rules = vertex_domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_EQ(rules[0].side, svmp::FE::geometry::CutIntegrationSide::Negative);
    ASSERT_EQ(rules[0].points.size(), 3u);
    EXPECT_NEAR(integrate_volume_weight(rules[0]),
                expected_vertex_volume,
                1.0e-20);
    EXPECT_NEAR(integrate_volume_coordinate(rules[0], 0),
                expected_vertex_volume * t / Real{3.0},
                1.0e-27);
    EXPECT_NEAR(integrate_volume_coordinate(rules[0], 1),
                expected_vertex_volume * t / Real{3.0},
                1.0e-27);

    LevelSetInterfaceDomain edge_domain(make_request(/*marker=*/32));
    appendLinearLevelSetCellCut2D(
        edge_domain,
        LevelSetCellCutInput{.parent_cell = 7,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-eps, -eps, 1.0, 1.0}});

    summary = edge_domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.active_volume_region_count, 2u);
    EXPECT_GT(summary.negative_volume_measure, 0.0);
    EXPECT_NEAR(summary.negative_volume_measure, t, 1.0e-14);

    rules = edge_domain.volumeQuadratureRules();
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_EQ(rules[0].side, svmp::FE::geometry::CutIntegrationSide::Negative);
    ASSERT_EQ(rules[0].points.size(), 6u);
    EXPECT_NEAR(integrate_volume_weight(rules[0]), t, 1.0e-14);
}

TEST(LevelSetInterfaceBuilder,
     PreservesSmallPlanarCornerVolumeUnderSignReversal)
{
    constexpr Real cut_ratio = 1.0e-8;
    constexpr Real expected_small_fraction = cut_ratio * cut_ratio;
    constexpr Real parent_measure = 0.5;
    constexpr std::array<geometry::CutIntegrationSide, 2> small_sides{{
        geometry::CutIntegrationSide::Negative,
        geometry::CutIntegrationSide::Positive,
    }};

    for (std::size_t case_index = 0u;
         case_index < small_sides.size();
         ++case_index) {
        const auto small_side = small_sides[case_index];
        const Real sign =
            small_side == geometry::CutIntegrationSide::Negative
                ? Real{-1.0}
                : Real{1.0};
        const LevelSetCellCutInput input{
            .parent_cell = static_cast<MeshIndex>(23u + case_index),
            .element_type = ElementType::Triangle3,
            .node_coordinates = {{{0.0, 0.0, 0.0}},
                                 {{1.0, 0.0, 0.0}},
                                 {{0.0, 1.0, 0.0}}},
            .level_set_values = {
                sign * cut_ratio,
                sign * (cut_ratio - Real{1.0}),
                sign * (cut_ratio - Real{1.0})}};

        const auto result =
            cutLinearLevelSetCell2D(make_request(/*marker=*/132), input);
        ASSERT_EQ(result.fragments.size(), 1u);
        ASSERT_EQ(result.volume_regions.size(), 2u)
            << "small_side=" << static_cast<unsigned>(small_side);
        const CutInterfaceVolumeRegion* small_region = nullptr;
        const CutInterfaceVolumeRegion* large_region = nullptr;
        for (const auto& region : result.volume_regions) {
            if (region.side == small_side) {
                small_region = &region;
            } else {
                large_region = &region;
            }
        }
        ASSERT_NE(small_region, nullptr);
        ASSERT_NE(large_region, nullptr);
        EXPECT_NEAR(small_region->volume_fraction /
                        expected_small_fraction,
                    1.0,
                    1.0e-8);
        EXPECT_NEAR(small_region->measure /
                        (parent_measure * expected_small_fraction),
                    1.0,
                    1.0e-8);
        EXPECT_NEAR(small_region->volume_fraction +
                        large_region->volume_fraction,
                    1.0,
                    1.0e-14);
        ASSERT_FALSE(small_region->quadrature_points.empty());
        Real quadrature_measure = 0.0;
        for (const auto& point : small_region->quadrature_points) {
            quadrature_measure += point.weight;
        }
        EXPECT_NEAR(quadrature_measure / small_region->measure,
                    1.0,
                    1.0e-12);
        ASSERT_FALSE(small_region->reference_subcells.empty());
        Real reference_measure = 0.0;
        for (const auto& subcell : small_region->reference_subcells) {
            reference_measure += reference_triangle_measure(subcell);
        }
        EXPECT_NEAR(reference_measure / small_region->measure,
                    1.0,
                    1.0e-8);
    }
}

TEST(LevelSetInterfaceBuilder,
     PreservesSmallTetrahedralCornerVolumeUnderSignReversal)
{
    constexpr Real interface_height = 1.0e-6;
    constexpr Real parent_height = 0.5;
    constexpr Real cut_ratio = interface_height / parent_height;
    constexpr Real expected_small_fraction =
        cut_ratio * cut_ratio * cut_ratio;
    constexpr Real parent_measure = 1.0 / 12.0;
    constexpr std::array<geometry::CutIntegrationSide, 2> small_sides{{
        geometry::CutIntegrationSide::Negative,
        geometry::CutIntegrationSide::Positive,
    }};

    for (std::size_t case_index = 0u;
         case_index < small_sides.size();
         ++case_index) {
        const auto small_side = small_sides[case_index];
        const Real sign =
            small_side == geometry::CutIntegrationSide::Negative
                ? Real{-1.0}
                : Real{1.0};
        const LevelSetCellCutInput input{
            .parent_cell = static_cast<MeshIndex>(25u + case_index),
            .element_type = ElementType::Tetra4,
            .node_coordinates = {{{0.0, 0.0, 0.0}},
                                 {{1.0, parent_height, 0.0}},
                                 {{0.0, parent_height, 0.0}},
                                 {{1.0, parent_height, 1.0}}},
            .level_set_values = {
                sign * interface_height,
                sign * (interface_height - parent_height),
                sign * (interface_height - parent_height),
                sign * (interface_height - parent_height)}};

        const auto result =
            cutLinearLevelSetCell3D(make_request(/*marker=*/133), input);
        ASSERT_EQ(result.fragments.size(), 1u);
        ASSERT_EQ(result.volume_regions.size(), 2u)
            << "small_side=" << static_cast<unsigned>(small_side);
        const CutInterfaceVolumeRegion* small_region = nullptr;
        const CutInterfaceVolumeRegion* large_region = nullptr;
        for (const auto& region : result.volume_regions) {
            if (region.side == small_side) {
                small_region = &region;
            } else {
                large_region = &region;
            }
        }
        ASSERT_NE(small_region, nullptr);
        ASSERT_NE(large_region, nullptr);
        EXPECT_GT(small_region->measure, 0.0);
        EXPECT_NEAR(small_region->volume_fraction /
                        expected_small_fraction,
                    1.0,
                    1.0e-10);
        EXPECT_NEAR(small_region->measure /
                        (parent_measure * expected_small_fraction),
                    1.0,
                    1.0e-10);
        EXPECT_NEAR(small_region->volume_fraction +
                        large_region->volume_fraction,
                    1.0,
                    1.0e-14);
        ASSERT_FALSE(small_region->quadrature_points.empty());
        Real quadrature_measure = 0.0;
        for (const auto& point : small_region->quadrature_points) {
            quadrature_measure += point.weight;
        }
        EXPECT_NEAR(quadrature_measure / small_region->measure,
                    1.0,
                    1.0e-12);
        ASSERT_FALSE(small_region->reference_subcells.empty());
        Real reference_measure = 0.0;
        for (const auto& subcell : small_region->reference_subcells) {
            reference_measure += reference_tetrahedron_measure(subcell);
        }
        EXPECT_NEAR(reference_measure / small_region->measure,
                    1.0,
                    1.0e-10);
    }
}

TEST(LevelSetInterfaceBuilder, VolumeQuadratureMatchesConservativeToleranceBandFractions)
{
    auto request = make_request(/*marker=*/33);
    request.tolerance = 1.0e-12;
    LevelSetInterfaceDomain domain(request);
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 8,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-2.0e-12,
                                                  1.0,
                                                  2.0e-12,
                                                  -5.0e-13}});

    const auto summary = domain.summary();
    EXPECT_EQ(summary.active_fragment_count, 1u);
    EXPECT_EQ(summary.active_volume_region_count, 2u);
    ASSERT_EQ(domain.fragments().size(), 1u);
    const auto& fragment = domain.fragments().front();
    EXPECT_NEAR(fragment.negative_volume_fraction +
                    fragment.positive_volume_fraction,
                1.0,
                1.0e-14);
    EXPECT_GT(fragment.negative_volume_fraction, 0.0);
    EXPECT_GT(fragment.positive_volume_fraction, 0.0);

    const auto volume_rules = domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 2u);
    Real total_measure = 0.0;
    Real total_weight = 0.0;
    bool saw_negative = false;
    bool saw_positive = false;
    for (const auto& rule : volume_rules) {
        expect_volume_rule_constant_exact(rule, 1.0e-14);
        total_measure += rule.measure;
        total_weight += integrate_volume_weight(rule);
        saw_negative =
            saw_negative ||
            rule.side == svmp::FE::geometry::CutIntegrationSide::Negative;
        saw_positive =
            saw_positive ||
            rule.side == svmp::FE::geometry::CutIntegrationSide::Positive;
    }
    EXPECT_TRUE(saw_negative);
    EXPECT_TRUE(saw_positive);
    EXPECT_NEAR(total_measure, 1.0, 1.0e-14);
    EXPECT_NEAR(total_weight, 1.0, 1.0e-14);
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
