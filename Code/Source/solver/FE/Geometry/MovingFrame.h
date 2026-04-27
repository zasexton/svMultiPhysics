/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_GEOMETRY_MOVINGFRAME_H
#define SVMP_FE_GEOMETRY_MOVINGFRAME_H

/**
 * @file MovingFrame.h
 * @brief Physics-agnostic coordinate-frame descriptors for moving domains.
 */

#include "Core/Types.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace geometry {

using FrameVector3 = std::array<Real, 3>;
using FrameMatrix3 = std::array<std::array<Real, 3>, 3>;

enum class CoordinateFrameKind : std::uint8_t {
    Inertial,
    UserDefined,
    Moving
};

enum class CoordinateFrameTimeLevel : std::uint8_t {
    TrialIterate,
    AcceptedNonlinearState,
    AcceptedTimeStep,
    AcceptedRemeshOrRezoneState
};

struct CoordinateFrameDescriptor {
    std::string name{"inertial"};
    CoordinateFrameKind kind{CoordinateFrameKind::Inertial};
    CoordinateFrameTimeLevel time_level{CoordinateFrameTimeLevel::TrialIterate};
    FrameVector3 origin{{0.0, 0.0, 0.0}};
    FrameMatrix3 basis{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}}};
    FrameVector3 linear_velocity{{0.0, 0.0, 0.0}};
    FrameVector3 angular_velocity{{0.0, 0.0, 0.0}};
    FrameVector3 linear_acceleration{{0.0, 0.0, 0.0}};
    FrameVector3 angular_acceleration{{0.0, 0.0, 0.0}};
    Real time{0.0};
    std::uint64_t epoch{0};
    bool orthonormal_basis{true};
};

struct FrameValidationResult {
    bool ok{true};
    std::vector<std::string> messages{};
};

class MovingFrameTransform {
public:
    [[nodiscard]] static CoordinateFrameDescriptor inertial();

    [[nodiscard]] static FrameValidationResult validate(
        const CoordinateFrameDescriptor& frame,
        Real tolerance = Real(1e-10));

    [[nodiscard]] static FrameVector3 pointToFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& inertial_point);

    [[nodiscard]] static FrameVector3 pointFromFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& frame_point);

    [[nodiscard]] static FrameVector3 vectorToFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& inertial_vector);

    [[nodiscard]] static FrameVector3 vectorFromFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& frame_vector);

    [[nodiscard]] static FrameVector3 normalToFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& inertial_normal);

    [[nodiscard]] static FrameVector3 normalFromFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& frame_normal);

    [[nodiscard]] static FrameMatrix3 rank2TensorToFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameMatrix3& inertial_tensor);

    [[nodiscard]] static FrameMatrix3 rank2TensorFromFrame(
        const CoordinateFrameDescriptor& frame,
        const FrameMatrix3& frame_tensor);

    [[nodiscard]] static Real measureToFrame(
        const CoordinateFrameDescriptor& frame,
        Real inertial_measure);

    [[nodiscard]] static FrameVector3 frameVelocityAtPoint(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& inertial_point);

    [[nodiscard]] static FrameVector3 frameAccelerationAtPoint(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& inertial_point);

    [[nodiscard]] static FrameVector3 relativeVelocity(
        const CoordinateFrameDescriptor& frame,
        const FrameVector3& inertial_field_velocity,
        const FrameVector3& inertial_point);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_MOVINGFRAME_H
