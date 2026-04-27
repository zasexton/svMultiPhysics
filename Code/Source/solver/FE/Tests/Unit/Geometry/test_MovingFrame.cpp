#include "Geometry/MovingFrame.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace svmp::FE::geometry;

TEST(MovingFrame, TransformsPointsVectorsNormalsAndRankTwoTensors)
{
    CoordinateFrameDescriptor frame;
    frame.name = "rotating_frame";
    frame.kind = CoordinateFrameKind::Moving;
    frame.basis = FrameMatrix3{{
        {{0.0, -1.0, 0.0}},
        {{1.0,  0.0, 0.0}},
        {{0.0,  0.0, 1.0}}}};
    frame.origin = FrameVector3{{1.0, 0.0, 0.0}};

    const auto validation = MovingFrameTransform::validate(frame);
    EXPECT_TRUE(validation.ok);

    const auto p = MovingFrameTransform::pointToFrame(frame, FrameVector3{{1.0, 1.0, 0.0}});
    EXPECT_NEAR(p[0], 1.0, 1e-14);
    EXPECT_NEAR(p[1], 0.0, 1e-14);
    EXPECT_NEAR(p[2], 0.0, 1e-14);
    const auto back = MovingFrameTransform::pointFromFrame(frame, p);
    EXPECT_NEAR(back[0], 1.0, 1e-14);
    EXPECT_NEAR(back[1], 1.0, 1e-14);

    const auto n = MovingFrameTransform::normalToFrame(frame, FrameVector3{{0.0, 1.0, 0.0}});
    EXPECT_NEAR(n[0], 1.0, 1e-14);
    EXPECT_NEAR(n[1], 0.0, 1e-14);

    FrameMatrix3 tensor{{
        {{2.0, 0.0, 0.0}},
        {{0.0, 3.0, 0.0}},
        {{0.0, 0.0, 4.0}}}};
    const auto tf = MovingFrameTransform::rank2TensorToFrame(frame, tensor);
    EXPECT_NEAR(tf[0][0], 3.0, 1e-14);
    EXPECT_NEAR(tf[1][1], 2.0, 1e-14);
    EXPECT_NEAR(tf[2][2], 4.0, 1e-14);
}

TEST(MovingFrame, ComputesAnalyticRelativeVelocityAndAcceleration)
{
    CoordinateFrameDescriptor frame;
    frame.kind = CoordinateFrameKind::Moving;
    frame.angular_velocity = FrameVector3{{0.0, 0.0, 2.0}};
    frame.angular_acceleration = FrameVector3{{0.0, 0.0, 3.0}};
    frame.linear_velocity = FrameVector3{{1.0, 0.0, 0.0}};
    frame.linear_acceleration = FrameVector3{{0.0, 5.0, 0.0}};

    const FrameVector3 point{{1.0, 0.0, 0.0}};
    const auto v_frame = MovingFrameTransform::frameVelocityAtPoint(frame, point);
    EXPECT_DOUBLE_EQ(v_frame[0], 1.0);
    EXPECT_DOUBLE_EQ(v_frame[1], 2.0);

    const auto a_frame = MovingFrameTransform::frameAccelerationAtPoint(frame, point);
    EXPECT_DOUBLE_EQ(a_frame[0], -4.0);
    EXPECT_DOUBLE_EQ(a_frame[1], 8.0);

    const auto rel = MovingFrameTransform::relativeVelocity(
        frame, FrameVector3{{1.0, 4.0, 0.0}}, point);
    EXPECT_DOUBLE_EQ(rel[0], 0.0);
    EXPECT_DOUBLE_EQ(rel[1], 2.0);
}
