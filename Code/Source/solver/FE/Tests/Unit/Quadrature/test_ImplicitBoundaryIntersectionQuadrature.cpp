#include "Quadrature/ImplicitBoundaryIntersectionQuadrature.h"

#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

namespace {

ImplicitBoundaryIntersectionRequest makeRequest(
    ElementType type,
    int dimension,
    LocalIndex subentity,
    std::vector<std::array<Real, 3>> coordinates,
    std::vector<Real> values,
    int order = 1)
{
    ImplicitBoundaryIntersectionRequest request;
    request.parent_element = type;
    request.parent_dimension = dimension;
    request.local_subentity = subentity;
    request.parent_node_coordinates = std::move(coordinates);
    request.scalar_values = std::move(values);
    request.isovalue = 0.0;
    request.quadrature_order = order;
    request.tolerance.zero = 1.0e-12;
    request.tolerance.duplicate = 1.0e-12;
    request.tolerance.measure = 1.0e-14;
    return request;
}

std::vector<Real> xMinus(
    const std::vector<std::array<Real, 3>>& coordinates,
    Real offset)
{
    std::vector<Real> values;
    values.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        values.push_back(coordinate[0] - offset);
    }
    return values;
}

} // namespace

TEST(ImplicitBoundaryIntersectionQuadrature, Builds2DQuadPointRuleOnEdge)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    auto request = makeRequest(ElementType::Quad4,
                               2,
                               0,
                               coordinates,
                               xMinus(coordinates, 0.25));

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    ASSERT_TRUE(result.hasActiveFragments());
    ASSERT_EQ(result.fragments.size(), 1u);
    const auto& fragment = result.fragments.front();
    EXPECT_EQ(fragment.kind, ImplicitBoundaryIntersectionKind::Point);
    EXPECT_DOUBLE_EQ(fragment.measure, 1.0);
    ASSERT_EQ(fragment.quadrature_points.size(), 1u);
    EXPECT_NEAR(fragment.quadrature_points.front().physical_coordinate[0],
                0.25,
                1.0e-12);
    EXPECT_NEAR(fragment.quadrature_points.front().physical_coordinate[1],
                0.0,
                1.0e-12);
    EXPECT_NEAR(fragment.quadrature_points.front().weight, 1.0, 1.0e-12);
}

TEST(ImplicitBoundaryIntersectionQuadrature, Builds2DTrianglePointRuleOnEdge)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    auto request = makeRequest(ElementType::Triangle3,
                               2,
                               0,
                               coordinates,
                               xMinus(coordinates, 0.5));

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    ASSERT_TRUE(result.hasActiveFragments());
    const auto& point = result.fragments.front().quadrature_points.front();
    EXPECT_NEAR(point.physical_coordinate[0], 0.5, 1.0e-12);
    EXPECT_NEAR(point.physical_coordinate[1], 0.0, 1.0e-12);
}

TEST(ImplicitBoundaryIntersectionQuadrature, Builds3DTetraSegmentOnFace)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}};
    auto request = makeRequest(ElementType::Tetra4,
                               3,
                               1,
                               coordinates,
                               xMinus(coordinates, 0.25));

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    ASSERT_TRUE(result.hasActiveFragments());
    ASSERT_EQ(result.fragments.size(), 1u);
    const auto& fragment = result.fragments.front();
    EXPECT_EQ(fragment.kind, ImplicitBoundaryIntersectionKind::Segment);
    EXPECT_NEAR(fragment.measure, 0.75, 1.0e-12);
    ASSERT_EQ(fragment.quadrature_points.size(), 1u);
    EXPECT_NEAR(fragment.quadrature_points.front().physical_coordinate[0],
                0.25,
                1.0e-12);
    EXPECT_NEAR(fragment.quadrature_points.front().physical_coordinate[1],
                0.0,
                1.0e-12);
    EXPECT_NEAR(fragment.quadrature_points.front().physical_coordinate[2],
                0.375,
                1.0e-12);
    EXPECT_NEAR(fragment.quadrature_points.front().weight, 0.75, 1.0e-12);
}

TEST(ImplicitBoundaryIntersectionQuadrature, UsesReferenceLengthOnStretchedHexFace)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{2.0, 0.0, 0.0}},
        {{2.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 4.0}},
        {{2.0, 0.0, 4.0}},
        {{2.0, 1.0, 4.0}},
        {{0.0, 1.0, 4.0}}};
    auto request = makeRequest(ElementType::Hex8,
                               3,
                               2,
                               coordinates,
                               xMinus(coordinates, 1.0),
                               3);

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    ASSERT_TRUE(result.hasActiveFragments());
    const auto& fragment = result.fragments.front();
    EXPECT_NEAR(fragment.measure, 2.0, 1.0e-12);
    ASSERT_EQ(fragment.quadrature_points.size(), 2u);
    EXPECT_NEAR(fragment.quadrature_points[0].weight, 1.0, 1.0e-12);
    EXPECT_NEAR(fragment.quadrature_points[1].weight, 1.0, 1.0e-12);
    EXPECT_NEAR(result.measure(), 2.0, 1.0e-12);
}

TEST(ImplicitBoundaryIntersectionQuadrature, SupportsWedgeFaceSegment)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, -1.0}},
        {{1.0, 0.0, -1.0}},
        {{0.0, 1.0, -1.0}},
        {{0.0, 0.0, 1.0}},
        {{1.0, 0.0, 1.0}},
        {{0.0, 1.0, 1.0}}};
    auto request = makeRequest(ElementType::Wedge6,
                               3,
                               2,
                               coordinates,
                               xMinus(coordinates, 0.5));

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    ASSERT_TRUE(result.hasActiveFragments());
    EXPECT_NEAR(result.measure(), 2.0, 1.0e-12);
}

TEST(ImplicitBoundaryIntersectionQuadrature, SupportsPyramidBaseSegment)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}};
    auto request = makeRequest(ElementType::Pyramid5,
                               3,
                               0,
                               coordinates,
                               xMinus(coordinates, 0.0));

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    ASSERT_TRUE(result.hasActiveFragments());
    EXPECT_NEAR(result.measure(), 2.0, 1.0e-12);
}

TEST(ImplicitBoundaryIntersectionQuadrature, EmptyWhenSelectedSubentityDoesNotCross)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    auto request = makeRequest(ElementType::Quad4,
                               2,
                               0,
                               coordinates,
                               std::vector<Real>{1.0, 1.0, -1.0, -1.0});

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    EXPECT_FALSE(result.hasActiveFragments());
    EXPECT_TRUE(result.fragments.empty());
    EXPECT_EQ(result.status, ImplicitBoundaryIntersectionStatus::Empty);
}

TEST(ImplicitBoundaryIntersectionQuadrature, HandlesVertexTouchDeterministically)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}};
    auto request = makeRequest(ElementType::Tetra4,
                               3,
                               1,
                               coordinates,
                               std::vector<Real>{0.0, 1.0, 1.0, 1.0});

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    EXPECT_FALSE(result.hasActiveFragments());
    ASSERT_EQ(result.fragments.size(), 1u);
    EXPECT_EQ(result.status, ImplicitBoundaryIntersectionStatus::VertexTouch);
    EXPECT_EQ(result.fragments.front().status,
              ImplicitBoundaryIntersectionStatus::VertexTouch);
}

TEST(ImplicitBoundaryIntersectionQuadrature, HandlesEdgeAlignedZeroWithoutDuplicates)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    auto request = makeRequest(ElementType::Quad4,
                               2,
                               0,
                               coordinates,
                               std::vector<Real>{0.0, 0.0, 1.0, 1.0});

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    EXPECT_FALSE(result.hasActiveFragments());
    ASSERT_EQ(result.fragments.size(), 1u);
    EXPECT_EQ(result.fragments.front().status,
              ImplicitBoundaryIntersectionStatus::EdgeAlignedZero);
    EXPECT_TRUE(result.fragments.front().quadrature_points.empty());
}

TEST(ImplicitBoundaryIntersectionQuadrature,
     RejectsTwoDimensionalVertexTouchWithoutLocalDoubleCounting)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    auto request = makeRequest(ElementType::Quad4,
                               2,
                               0,
                               coordinates,
                               std::vector<Real>{0.0, 1.0, 1.0, 1.0});

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    EXPECT_EQ(result.status, ImplicitBoundaryIntersectionStatus::VertexTouch);
    EXPECT_FALSE(result.hasActiveFragments());
    ASSERT_EQ(result.fragments.size(), 1u);
    EXPECT_TRUE(result.fragments.front().quadrature_points.empty());
}

TEST(ImplicitBoundaryIntersectionQuadrature,
     RejectsAlignedThreeDimensionalBoundaryEdge)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}};
    auto request = makeRequest(ElementType::Tetra4,
                               3,
                               1,
                               coordinates,
                               std::vector<Real>{0.0, 0.0, 1.0, 1.0});

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    EXPECT_EQ(result.status,
              ImplicitBoundaryIntersectionStatus::EdgeAlignedZero);
    EXPECT_FALSE(result.hasActiveFragments());
    ASSERT_EQ(result.fragments.size(), 1u);
    EXPECT_TRUE(result.fragments.front().quadrature_points.empty());
}

TEST(ImplicitBoundaryIntersectionQuadrature,
     RejectsAmbiguousSaddleFaceInsteadOfPairingRoots)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
        {{1.0, 0.0, 1.0}},
        {{1.0, 1.0, 1.0}},
        {{0.0, 1.0, 1.0}}};
    auto request = makeRequest(ElementType::Hex8,
                               3,
                               0,
                               coordinates,
                               std::vector<Real>{1.0, -1.0, 1.0, -1.0,
                                                 1.0, 1.0, 1.0, 1.0});

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    EXPECT_EQ(result.status, ImplicitBoundaryIntersectionStatus::Ambiguous);
    EXPECT_FALSE(result.hasActiveFragments());
    ASSERT_EQ(result.fragments.size(), 1u);
    EXPECT_TRUE(result.fragments.front().quadrature_points.empty());
}

TEST(ImplicitBoundaryIntersectionQuadrature,
     RejectsHighOrderParentInsteadOfClaimingCornerLinearAccuracy)
{
    const std::vector<std::array<Real, 3>> coordinates{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}};
    auto request = makeRequest(ElementType::Tetra10,
                               3,
                               1,
                               coordinates,
                               std::vector<Real>{-1.0, 1.0, -1.0, -1.0});

    const auto result = buildImplicitBoundaryIntersectionQuadrature(request);

    EXPECT_EQ(result.status,
              ImplicitBoundaryIntersectionStatus::UnsupportedElement);
    EXPECT_FALSE(result.hasActiveFragments());
}
