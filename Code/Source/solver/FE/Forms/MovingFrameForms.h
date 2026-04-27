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
 */

#include "Forms/FormExpr.h"

#include <array>

namespace svmp {
namespace FE {
namespace forms {

struct MovingFrameParameterSlots {
    std::array<std::uint32_t, 3> origin{{0u, 1u, 2u}};
    std::array<std::uint32_t, 3> linear_velocity{{3u, 4u, 5u}};
    std::array<std::uint32_t, 3> angular_velocity{{6u, 7u, 8u}};
    std::array<std::uint32_t, 3> linear_acceleration{{9u, 10u, 11u}};
    std::array<std::uint32_t, 3> angular_acceleration{{12u, 13u, 14u}};
};

[[nodiscard]] inline FormExpr frameVectorParameter(
    const std::array<std::uint32_t, 3>& slots)
{
    return FormExpr::asVector({
        FormExpr::parameterRef(slots[0]),
        FormExpr::parameterRef(slots[1]),
        FormExpr::parameterRef(slots[2])});
}

[[nodiscard]] inline FormExpr frameOrigin(
    const MovingFrameParameterSlots& slots = {})
{
    return frameVectorParameter(slots.origin);
}

[[nodiscard]] inline FormExpr frameLinearVelocity(
    const MovingFrameParameterSlots& slots = {})
{
    return frameVectorParameter(slots.linear_velocity);
}

[[nodiscard]] inline FormExpr frameAngularVelocity(
    const MovingFrameParameterSlots& slots = {})
{
    return frameVectorParameter(slots.angular_velocity);
}

[[nodiscard]] inline FormExpr frameLinearAcceleration(
    const MovingFrameParameterSlots& slots = {})
{
    return frameVectorParameter(slots.linear_acceleration);
}

[[nodiscard]] inline FormExpr frameAngularAcceleration(
    const MovingFrameParameterSlots& slots = {})
{
    return frameVectorParameter(slots.angular_acceleration);
}

[[nodiscard]] inline FormExpr frameVelocityAtCurrentCoordinate(
    const MovingFrameParameterSlots& slots = {})
{
    return frameLinearVelocity(slots) +
           cross(frameAngularVelocity(slots),
                 FormExpr::currentCoordinate() - frameOrigin(slots));
}

[[nodiscard]] inline FormExpr frameAccelerationAtCurrentCoordinate(
    const MovingFrameParameterSlots& slots = {})
{
    const auto r = FormExpr::currentCoordinate() - frameOrigin(slots);
    const auto omega = frameAngularVelocity(slots);
    return frameLinearAcceleration(slots) +
           cross(frameAngularAcceleration(slots), r) +
           cross(omega, cross(omega, r));
}

[[nodiscard]] inline FormExpr relativeDomainVelocity(
    const MovingFrameParameterSlots& slots = {})
{
    return FormExpr::domainVelocity() - frameVelocityAtCurrentCoordinate(slots);
}

struct MovingFrameFormTerminals {
    FormExpr current_coordinate{};
    FormExpr reference_coordinate{};
    FormExpr domain_velocity{};
    FormExpr domain_acceleration{};
    FormExpr frame_origin{};
    FormExpr frame_linear_velocity{};
    FormExpr frame_angular_velocity{};
    FormExpr frame_linear_acceleration{};
    FormExpr frame_angular_acceleration{};
    FormExpr relative_velocity{};
};

[[nodiscard]] inline MovingFrameFormTerminals movingFrameTerminals(
    const MovingFrameParameterSlots& slots = {})
{
    MovingFrameFormTerminals terminals;
    terminals.current_coordinate = FormExpr::currentCoordinate();
    terminals.reference_coordinate = FormExpr::referenceCoordinatePhysical();
    terminals.domain_velocity = FormExpr::domainVelocity();
    terminals.domain_acceleration = FormExpr::meshAcceleration();
    terminals.frame_origin = frameOrigin(slots);
    terminals.frame_linear_velocity = frameLinearVelocity(slots);
    terminals.frame_angular_velocity = frameAngularVelocity(slots);
    terminals.frame_linear_acceleration = frameLinearAcceleration(slots);
    terminals.frame_angular_acceleration = frameAngularAcceleration(slots);
    terminals.relative_velocity = relativeDomainVelocity(slots);
    return terminals;
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_MOVINGFRAMEFORMS_H
