/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_MOVINGFRAMEFORMS_H
#define SVMP_FE_FORMS_MOVINGFRAMEFORMS_H

/**
 * @file MovingFrameForms.h
 * @brief Form-expression helpers for physics-agnostic moving-frame terminals.
 *
 * A moving frame is named, and its data are the named runtime parameters
 * frame.<name>.<quantity>_<x|y|z> (movingFrameParameterNames()), which the
 * system resolves to their own slots at setup like any FormExpr::parameter,
 * so they never alias another parameter.
 */

#include "Forms/FormExpr.h"

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

/// The quantities of a moving frame, in the order of movingFrameParameterNames().
inline constexpr std::array<std::string_view, 5> kMovingFrameQuantities{
    "origin", "linear_velocity", "angular_velocity", "linear_acceleration", "angular_acceleration"};

/// frame.<frame>.<quantity>_<axis>, the name of one component of a frame's data.
[[nodiscard]] inline std::string movingFrameParameterName(std::string_view frame,
                                                          std::string_view quantity,
                                                          char axis)
{
    return "frame." + std::string(frame) + "." + std::string(quantity) + "_" + std::string(1, axis);
}

/// The 15 parameter names of @p frame: each quantity's x, y and z components.
[[nodiscard]] inline std::vector<std::string> movingFrameParameterNames(std::string_view frame)
{
    std::vector<std::string> names;
    names.reserve(3u * kMovingFrameQuantities.size());
    for (const auto quantity : kMovingFrameQuantities) {
        for (const char axis : {'x', 'y', 'z'}) {
            names.push_back(movingFrameParameterName(frame, quantity, axis));
        }
    }
    return names;
}

/// The vector of the named parameters frame.<frame>.<quantity>_{x,y,z}.
[[nodiscard]] inline FormExpr frameVectorParameter(std::string_view frame, std::string_view quantity)
{
    return FormExpr::asVector({FormExpr::parameter(movingFrameParameterName(frame, quantity, 'x')),
                               FormExpr::parameter(movingFrameParameterName(frame, quantity, 'y')),
                               FormExpr::parameter(movingFrameParameterName(frame, quantity, 'z'))});
}

[[nodiscard]] inline FormExpr frameOrigin(std::string_view frame)
{
    return frameVectorParameter(frame, "origin");
}

[[nodiscard]] inline FormExpr frameLinearVelocity(std::string_view frame)
{
    return frameVectorParameter(frame, "linear_velocity");
}

[[nodiscard]] inline FormExpr frameAngularVelocity(std::string_view frame)
{
    return frameVectorParameter(frame, "angular_velocity");
}

[[nodiscard]] inline FormExpr frameLinearAcceleration(std::string_view frame)
{
    return frameVectorParameter(frame, "linear_acceleration");
}

[[nodiscard]] inline FormExpr frameAngularAcceleration(std::string_view frame)
{
    return frameVectorParameter(frame, "angular_acceleration");
}

/// v_0 + omega x (x - x_0), the frame's velocity at currentCoordinate().
[[nodiscard]] inline FormExpr frameVelocityAtCurrentCoordinate(std::string_view frame)
{
    return frameLinearVelocity(frame) +
           cross(frameAngularVelocity(frame), FormExpr::currentCoordinate() - frameOrigin(frame));
}

/// a_0 + alpha x r + omega x (omega x r) with r = x - x_0, the frame's acceleration at currentCoordinate().
[[nodiscard]] inline FormExpr frameAccelerationAtCurrentCoordinate(std::string_view frame)
{
    const auto r = FormExpr::currentCoordinate() - frameOrigin(frame);
    const auto omega = frameAngularVelocity(frame);
    return frameLinearAcceleration(frame) + cross(frameAngularAcceleration(frame), r) + cross(omega, cross(omega, r));
}

/// meshVelocity() - frameVelocityAtCurrentCoordinate(frame).
[[nodiscard]] inline FormExpr relativeMeshVelocity(std::string_view frame)
{
    return FormExpr::meshVelocity() - frameVelocityAtCurrentCoordinate(frame);
}

struct MovingFrameFormTerminals {
    FormExpr current_coordinate{};
    FormExpr reference_coordinate{};
    FormExpr mesh_velocity{};
    FormExpr mesh_acceleration{};
    FormExpr frame_origin{};
    FormExpr frame_linear_velocity{};
    FormExpr frame_angular_velocity{};
    FormExpr frame_linear_acceleration{};
    FormExpr frame_angular_acceleration{};
    FormExpr relative_velocity{};
};

[[nodiscard]] inline MovingFrameFormTerminals movingFrameTerminals(std::string_view frame)
{
    MovingFrameFormTerminals terminals;
    terminals.current_coordinate = FormExpr::currentCoordinate();
    terminals.reference_coordinate = FormExpr::referenceCoordinatePhysical();
    terminals.mesh_velocity = FormExpr::meshVelocity();
    terminals.mesh_acceleration = FormExpr::meshAcceleration();
    terminals.frame_origin = frameOrigin(frame);
    terminals.frame_linear_velocity = frameLinearVelocity(frame);
    terminals.frame_angular_velocity = frameAngularVelocity(frame);
    terminals.frame_linear_acceleration = frameLinearAcceleration(frame);
    terminals.frame_angular_acceleration = frameAngularAcceleration(frame);
    terminals.relative_velocity = relativeMeshVelocity(frame);
    return terminals;
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_MOVINGFRAMEFORMS_H
