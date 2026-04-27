#include "Geometry/CutQuadrature.h"

#include <gtest/gtest.h>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

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
