/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_GEOMETRIC_INTEGRAL_CONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_GEOMETRIC_INTEGRAL_CONSTRAINT_H

/**
 * @file GeometricIntegralConstraint.h
 * @brief Physics-neutral contracts for nonlinear geometric integral constraints.
 */

#include "Core/Types.h"
#include "Forms/FiniteDeformationForms.h"

#include <algorithm>
#include <cstdint>

namespace svmp {
namespace FE {
namespace constraints {

enum class GeometricIntegralQuantity : std::uint8_t {
    EnclosedVolume,
    SurfaceArea,
    CenterOfMass,
    AverageBoundaryDisplacement,
    GeometricMoment
};

enum class GeometricConstraintSensitivityPath : std::uint8_t {
    Analytic,
    Symbolic,
    AD,
    JIT,
    VerificationFiniteDifference
};

enum class GeometricConstraintStateLevel : std::uint8_t {
    TrialIterate,
    AcceptedNonlinearState,
    AcceptedTimeStep,
    AcceptedRemeshOrRezoneState
};

struct GeometricIntegralConstraintSpec {
    GeometricIntegralQuantity quantity{GeometricIntegralQuantity::EnclosedVolume};
    int boundary_marker{-1};
    int component{-1};
    int moment_order{0};
    Real target_value{0.0};
    GeometricConstraintSensitivityPath sensitivity{
        GeometricConstraintSensitivityPath::Analytic};
    GeometricConstraintStateLevel state_level{
        GeometricConstraintStateLevel::TrialIterate};
    bool contributes_to_residual{true};
    bool contributes_to_tangent{true};
};

[[nodiscard]] constexpr bool isProductionSensitivity(
    GeometricConstraintSensitivityPath path) noexcept
{
    return path != GeometricConstraintSensitivityPath::VerificationFiniteDifference;
}

[[nodiscard]] constexpr bool geometricIntegralConstraintSpecIsValid(
    const GeometricIntegralConstraintSpec& spec) noexcept
{
    if (spec.boundary_marker < -1) {
        return false;
    }
    if (spec.component < -1 || spec.component > 2) {
        return false;
    }
    if (spec.moment_order < 0) {
        return false;
    }
    if (spec.contributes_to_tangent && !isProductionSensitivity(spec.sensitivity)) {
        return false;
    }
    return spec.contributes_to_residual || spec.contributes_to_tangent;
}

[[nodiscard]] constexpr bool quantityRequiresBoundary(
    GeometricIntegralQuantity quantity) noexcept
{
    return quantity == GeometricIntegralQuantity::SurfaceArea ||
           quantity == GeometricIntegralQuantity::AverageBoundaryDisplacement ||
           quantity == GeometricIntegralQuantity::GeometricMoment;
}

[[nodiscard]] constexpr const char* geometricIntegralQuantityName(
    GeometricIntegralQuantity quantity) noexcept
{
    switch (quantity) {
        case GeometricIntegralQuantity::EnclosedVolume: return "EnclosedVolume";
        case GeometricIntegralQuantity::SurfaceArea: return "SurfaceArea";
        case GeometricIntegralQuantity::CenterOfMass: return "CenterOfMass";
        case GeometricIntegralQuantity::AverageBoundaryDisplacement:
            return "AverageBoundaryDisplacement";
        case GeometricIntegralQuantity::GeometricMoment: return "GeometricMoment";
    }
    return "Unknown";
}

[[nodiscard]] constexpr const char* geometricConstraintSensitivityPathName(
    GeometricConstraintSensitivityPath path) noexcept
{
    switch (path) {
        case GeometricConstraintSensitivityPath::Analytic: return "Analytic";
        case GeometricConstraintSensitivityPath::Symbolic: return "Symbolic";
        case GeometricConstraintSensitivityPath::AD: return "AD";
        case GeometricConstraintSensitivityPath::JIT: return "JIT";
        case GeometricConstraintSensitivityPath::VerificationFiniteDifference:
            return "VerificationFiniteDifference";
    }
    return "Unknown";
}

[[nodiscard]] constexpr const char* geometricConstraintStateLevelName(
    GeometricConstraintStateLevel state_level) noexcept
{
    switch (state_level) {
        case GeometricConstraintStateLevel::TrialIterate: return "TrialIterate";
        case GeometricConstraintStateLevel::AcceptedNonlinearState:
            return "AcceptedNonlinearState";
        case GeometricConstraintStateLevel::AcceptedTimeStep:
            return "AcceptedTimeStep";
        case GeometricConstraintStateLevel::AcceptedRemeshOrRezoneState:
            return "AcceptedRemeshOrRezoneState";
    }
    return "Unknown";
}

[[nodiscard]] inline forms::FormExpr enclosedVolumeChangeResidual(
    const forms::FormExpr& deformation_gradient,
    Real target_volume_ratio)
{
    return (forms::finite_deformation::jacobian(deformation_gradient) -
            forms::FormExpr::constant(target_volume_ratio))
        .dx();
}

[[nodiscard]] inline forms::FormExpr surfaceAreaChangeResidual(
    const forms::FormExpr& deformation_gradient,
    Real target_area_ratio,
    int boundary_marker = -1)
{
    const auto reference_normal = forms::FormExpr::normal();
    const auto current_measure =
        forms::norm(forms::finite_deformation::nansonMeasureVector(
            deformation_gradient, reference_normal));
    return (current_measure - forms::FormExpr::constant(target_area_ratio))
        .ds(boundary_marker);
}

[[nodiscard]] inline forms::FormExpr averageBoundaryDisplacementResidual(
    const forms::FormExpr& displacement,
    int component,
    Real target_value,
    int boundary_marker = -1)
{
    const auto value =
        component >= 0 ? forms::component(displacement, component) : forms::norm(displacement);
    return (value - forms::FormExpr::constant(target_value)).ds(boundary_marker);
}

[[nodiscard]] inline forms::FormExpr centerOfMassResidual(
    const forms::FormExpr& deformation_gradient,
    int component,
    Real target_coordinate)
{
    const auto x = forms::FormExpr::currentCoordinate();
    const auto value =
        component >= 0 ? forms::component(x, component) : forms::norm(x);
    const auto J = forms::finite_deformation::jacobian(deformation_gradient);
    return ((value - forms::FormExpr::constant(target_coordinate)) * J).dx();
}

[[nodiscard]] inline forms::FormExpr geometricMomentResidual(
    const forms::FormExpr& deformation_gradient,
    int component,
    int moment_order,
    Real target_moment,
    int boundary_marker = -1)
{
    const auto x = forms::FormExpr::currentCoordinate();
    auto value = component >= 0 ? forms::component(x, component) : forms::norm(x);
    const int order = std::max(moment_order, 1);
    if (order > 1) {
        value = value.pow(forms::FormExpr::constant(static_cast<Real>(order)));
    }
    const auto reference_normal = forms::FormExpr::normal();
    const auto current_measure =
        forms::norm(forms::finite_deformation::nansonMeasureVector(
            deformation_gradient, reference_normal));
    return (value * current_measure - forms::FormExpr::constant(target_moment))
        .ds(boundary_marker);
}

[[nodiscard]] inline forms::FormExpr geometricIntegralResidual(
    const GeometricIntegralConstraintSpec& spec,
    const forms::FormExpr& displacement,
    const forms::FormExpr& deformation_gradient)
{
    switch (spec.quantity) {
        case GeometricIntegralQuantity::EnclosedVolume:
            return enclosedVolumeChangeResidual(deformation_gradient, spec.target_value);
        case GeometricIntegralQuantity::SurfaceArea:
            return surfaceAreaChangeResidual(deformation_gradient,
                                             spec.target_value,
                                             spec.boundary_marker);
        case GeometricIntegralQuantity::AverageBoundaryDisplacement:
            return averageBoundaryDisplacementResidual(displacement,
                                                       spec.component,
                                                       spec.target_value,
                                                       spec.boundary_marker);
        case GeometricIntegralQuantity::CenterOfMass:
            return centerOfMassResidual(deformation_gradient,
                                        spec.component,
                                        spec.target_value);
        case GeometricIntegralQuantity::GeometricMoment:
            return geometricMomentResidual(deformation_gradient,
                                           spec.component,
                                           spec.moment_order,
                                           spec.target_value,
                                           spec.boundary_marker);
    }
    return forms::FormExpr::constant(0.0).dx();
}

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_GEOMETRIC_INTEGRAL_CONSTRAINT_H
