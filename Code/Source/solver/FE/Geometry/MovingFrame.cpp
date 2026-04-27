/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Geometry/MovingFrame.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {
namespace {

FrameVector3 add(const FrameVector3& a, const FrameVector3& b) noexcept
{
    return FrameVector3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

FrameVector3 sub(const FrameVector3& a, const FrameVector3& b) noexcept
{
    return FrameVector3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

FrameVector3 scale(const FrameVector3& a, Real s) noexcept
{
    return FrameVector3{s * a[0], s * a[1], s * a[2]};
}

Real dot(const FrameVector3& a, const FrameVector3& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

FrameVector3 cross(const FrameVector3& a, const FrameVector3& b) noexcept
{
    return FrameVector3{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]};
}

Real norm(const FrameVector3& v) noexcept
{
    return std::sqrt(dot(v, v));
}

FrameVector3 unitOrZero(const FrameVector3& v) noexcept
{
    const Real n = norm(v);
    if (n <= Real(1e-30)) {
        return FrameVector3{};
    }
    return scale(v, Real(1) / n);
}

FrameVector3 matVec(const FrameMatrix3& a, const FrameVector3& x) noexcept
{
    return FrameVector3{
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2]};
}

FrameVector3 matTransposeVec(const FrameMatrix3& a, const FrameVector3& x) noexcept
{
    return FrameVector3{
        a[0][0] * x[0] + a[1][0] * x[1] + a[2][0] * x[2],
        a[0][1] * x[0] + a[1][1] * x[1] + a[2][1] * x[2],
        a[0][2] * x[0] + a[1][2] * x[1] + a[2][2] * x[2]};
}

FrameMatrix3 matMul(const FrameMatrix3& a, const FrameMatrix3& b) noexcept
{
    FrameMatrix3 out{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            for (std::size_t k = 0; k < 3u; ++k) {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return out;
}

FrameMatrix3 transpose(const FrameMatrix3& a) noexcept
{
    FrameMatrix3 out{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            out[i][j] = a[j][i];
        }
    }
    return out;
}

Real determinant(const FrameMatrix3& a) noexcept
{
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
           a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
           a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

} // namespace

CoordinateFrameDescriptor MovingFrameTransform::inertial()
{
    return CoordinateFrameDescriptor{};
}

FrameValidationResult MovingFrameTransform::validate(
    const CoordinateFrameDescriptor& frame,
    Real tolerance)
{
    FrameValidationResult result;
    const Real det = determinant(frame.basis);
    if (!std::isfinite(det) || std::abs(det) <= tolerance) {
        result.ok = false;
        result.messages.push_back("coordinate frame basis is singular");
    }
    if (frame.orthonormal_basis) {
        for (std::size_t i = 0; i < 3u; ++i) {
            FrameVector3 ei{frame.basis[0][i], frame.basis[1][i], frame.basis[2][i]};
            const Real nii = dot(ei, ei);
            if (std::abs(nii - Real(1)) > tolerance) {
                result.ok = false;
                result.messages.push_back("coordinate frame basis column is not unit length");
                break;
            }
            for (std::size_t j = i + 1u; j < 3u; ++j) {
                FrameVector3 ej{frame.basis[0][j], frame.basis[1][j], frame.basis[2][j]};
                if (std::abs(dot(ei, ej)) > tolerance) {
                    result.ok = false;
                    result.messages.push_back("coordinate frame basis columns are not orthogonal");
                    break;
                }
            }
        }
    }
    return result;
}

FrameVector3 MovingFrameTransform::pointToFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& inertial_point)
{
    return matTransposeVec(frame.basis, sub(inertial_point, frame.origin));
}

FrameVector3 MovingFrameTransform::pointFromFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& frame_point)
{
    return add(frame.origin, matVec(frame.basis, frame_point));
}

FrameVector3 MovingFrameTransform::vectorToFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& inertial_vector)
{
    return matTransposeVec(frame.basis, inertial_vector);
}

FrameVector3 MovingFrameTransform::vectorFromFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& frame_vector)
{
    return matVec(frame.basis, frame_vector);
}

FrameVector3 MovingFrameTransform::normalToFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& inertial_normal)
{
    return unitOrZero(vectorToFrame(frame, inertial_normal));
}

FrameVector3 MovingFrameTransform::normalFromFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& frame_normal)
{
    return unitOrZero(vectorFromFrame(frame, frame_normal));
}

FrameMatrix3 MovingFrameTransform::rank2TensorToFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameMatrix3& inertial_tensor)
{
    return matMul(transpose(frame.basis), matMul(inertial_tensor, frame.basis));
}

FrameMatrix3 MovingFrameTransform::rank2TensorFromFrame(
    const CoordinateFrameDescriptor& frame,
    const FrameMatrix3& frame_tensor)
{
    return matMul(frame.basis, matMul(frame_tensor, transpose(frame.basis)));
}

Real MovingFrameTransform::measureToFrame(
    const CoordinateFrameDescriptor& frame,
    Real inertial_measure)
{
    return inertial_measure * std::abs(determinant(frame.basis));
}

FrameVector3 MovingFrameTransform::frameVelocityAtPoint(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& inertial_point)
{
    return add(frame.linear_velocity,
               cross(frame.angular_velocity, sub(inertial_point, frame.origin)));
}

FrameVector3 MovingFrameTransform::frameAccelerationAtPoint(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& inertial_point)
{
    const auto r = sub(inertial_point, frame.origin);
    return add(frame.linear_acceleration,
               add(cross(frame.angular_acceleration, r),
                   cross(frame.angular_velocity, cross(frame.angular_velocity, r))));
}

FrameVector3 MovingFrameTransform::relativeVelocity(
    const CoordinateFrameDescriptor& frame,
    const FrameVector3& inertial_field_velocity,
    const FrameVector3& inertial_point)
{
    return sub(inertial_field_velocity, frameVelocityAtPoint(frame, inertial_point));
}

} // namespace geometry
} // namespace FE
} // namespace svmp
