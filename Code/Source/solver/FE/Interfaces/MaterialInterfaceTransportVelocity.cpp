/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/MaterialInterfaceTransportVelocity.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace svmp::FE::interfaces {
namespace {

[[nodiscard]] bool finiteVector(const std::array<Real, 3>& value) noexcept
{
    return std::all_of(value.begin(), value.end(), [](Real component) {
        return std::isfinite(component);
    });
}

} // namespace

void validateMaterialInterfaceTransportVelocityDeclaration(
    const MaterialInterfaceTransportVelocityDeclaration& declaration)
{
    const Real weight_sum =
        declaration.negative_trace_weight +
        declaration.positive_trace_weight;
    const Real weight_tolerance =
        Real{128.0} * std::numeric_limits<Real>::epsilon();
    if ((declaration.dimension != 2 && declaration.dimension != 3) ||
        declaration.interface_marker < 0 ||
        declaration.level_set_field == INVALID_FIELD_ID ||
        declaration.negative_velocity_field == INVALID_FIELD_ID ||
        declaration.positive_velocity_field == INVALID_FIELD_ID ||
        declaration.level_set_field ==
            declaration.negative_velocity_field ||
        declaration.level_set_field ==
            declaration.positive_velocity_field ||
        declaration.negative_velocity_field ==
            declaration.positive_velocity_field ||
        !std::isfinite(declaration.level_set_isovalue) ||
        !std::isfinite(declaration.negative_trace_weight) ||
        !std::isfinite(declaration.positive_trace_weight) ||
        !(declaration.negative_trace_weight > Real{0.0}) ||
        !(declaration.positive_trace_weight > Real{0.0}) ||
        std::abs(weight_sum - Real{1.0}) > weight_tolerance ||
        declaration.geometry_domain_id.find_first_not_of(" \t\r\n") ==
            std::string::npos ||
        declaration.owner_component.find_first_not_of(" \t\r\n") ==
            std::string::npos) {
        throw std::invalid_argument(
            "material-interface transport velocity declaration is incomplete or inconsistent");
    }
}

std::array<Real, 3> materialInterfaceCommonTraceVelocity(
    const MaterialInterfaceTransportVelocityDeclaration& declaration,
    const std::array<Real, 3>& negative_velocity,
    const std::array<Real, 3>& positive_velocity)
{
    validateMaterialInterfaceTransportVelocityDeclaration(declaration);
    if (!finiteVector(negative_velocity) ||
        !finiteVector(positive_velocity)) {
        throw std::invalid_argument(
            "material-interface transport velocity requires finite phase values");
    }
    std::array<Real, 3> value{};
    for (int component = 0; component < declaration.dimension; ++component) {
        const auto index = static_cast<std::size_t>(component);
        value[index] =
            declaration.negative_trace_weight * negative_velocity[index] +
            declaration.positive_trace_weight * positive_velocity[index];
    }
    return value;
}

MaterialInterfaceVelocitySample selectMaterialInterfaceTransportVelocity(
    const MaterialInterfaceTransportVelocityDeclaration& declaration,
    Real level_set_value,
    const std::array<Real, 3>& negative_velocity,
    const std::array<Real, 3>& positive_velocity)
{
    validateMaterialInterfaceTransportVelocityDeclaration(declaration);
    if (!std::isfinite(level_set_value) ||
        !finiteVector(negative_velocity) ||
        !finiteVector(positive_velocity)) {
        throw std::invalid_argument(
            "material-interface transport velocity requires finite level-set and phase values");
    }
    if (level_set_value < declaration.level_set_isovalue) {
        return MaterialInterfaceVelocitySample{
            .value = negative_velocity,
            .region = MaterialInterfaceVelocityRegion::NegativeBulk,
        };
    }
    if (level_set_value > declaration.level_set_isovalue) {
        return MaterialInterfaceVelocitySample{
            .value = positive_velocity,
            .region = MaterialInterfaceVelocityRegion::PositiveBulk,
        };
    }
    return MaterialInterfaceVelocitySample{
        .value = materialInterfaceCommonTraceVelocity(
            declaration, negative_velocity, positive_velocity),
        .region = MaterialInterfaceVelocityRegion::InterfaceTrace,
    };
}

} // namespace svmp::FE::interfaces
