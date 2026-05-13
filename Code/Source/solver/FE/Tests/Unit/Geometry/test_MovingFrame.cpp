#include "Core/FEException.h"
#include "Geometry/MovingFrame.h"
#include "Geometry/RegionFrameBinding.h"

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

TEST(MovingFrame, RegionFrameRegistryLooksUpFramesByRegion)
{
    RegionFrameRegistry registry;
    auto frame = MovingFrameTransform::inertial();
    frame.name = "boundary_frame";
    frame.kind = CoordinateFrameKind::Moving;
    frame.linear_velocity = FrameVector3{{1.0, 0.0, 0.0}};
    registry.addFrame(frame);

    FrameRegionDescriptor boundary;
    boundary.kind = FrameRegionKind::BoundaryMarker;
    boundary.marker = 4;
    registry.bindRegion(RegionFrameBinding{.region = boundary,
                                           .frame_name = "boundary_frame"});

    const auto* found = registry.findFrameForRegion(boundary);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name, "boundary_frame");

    FrameRegionDescriptor missing;
    missing.kind = FrameRegionKind::BoundaryMarker;
    missing.marker = 5;
    EXPECT_EQ(registry.findFrameForRegion(missing), nullptr);
    EXPECT_THROW(static_cast<void>(registry.requireFrameForRegion(missing)),
                 svmp::FE::InvalidArgumentException);
}

TEST(MovingFrame, RegionFrameVelocityAtQuadraturePoint)
{
    RegionFrameRegistry registry;
    auto frame = MovingFrameTransform::inertial();
    frame.name = "cell_frame";
    frame.kind = CoordinateFrameKind::Moving;
    frame.angular_velocity = FrameVector3{{0.0, 0.0, 2.0}};
    registry.addFrame(frame);

    FrameRegionDescriptor cells;
    cells.kind = FrameRegionKind::CellSet;
    cells.entity_ids = {10, 11};
    registry.bindRegion(RegionFrameBinding{.region = cells,
                                           .frame_name = "cell_frame"});

    const auto& bound_frame = registry.requireFrameForRegion(cells);
    const FrameVector3 quadrature_point{{1.0, 0.0, 0.0}};
    const auto velocity = MovingFrameTransform::frameVelocityAtPoint(
        bound_frame, quadrature_point);

    EXPECT_DOUBLE_EQ(velocity[0], 0.0);
    EXPECT_DOUBLE_EQ(velocity[1], 2.0);
    EXPECT_DOUBLE_EQ(velocity[2], 0.0);
}
