/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_CUTCELLFORMS_H
#define SVMP_FE_FORMS_CUTCELLFORMS_H

/**
 * @file CutCellForms.h
 * @brief Form-expression helpers for physics-neutral cut-cell metadata.
 */

#include "Core/AlignedAllocator.h"
#include "Core/Alignment.h"
#include "Core/Types.h"
#include "Forms/FormExpr.h"
#include "Geometry/CutQuadrature.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

struct CutCellParameterSlots {
    std::uint32_t volume_fraction{0u};
    std::uint32_t side_indicator{1u};
    std::array<std::uint32_t, 3> embedded_normal{{2u, 3u, 4u}};
    std::uint32_t stabilization_scale{5u};
    std::array<std::uint32_t, 3> measure_sensitivity{{6u, 7u, 8u}};
    std::array<std::uint32_t, 3> normal_sensitivity{{9u, 10u, 11u}};
    std::uint32_t quadrature_weight_sensitivity{12u};
};

using CutCellParameterVector =
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>>;

[[nodiscard]] inline std::uint32_t cutCellParameterCount(
    const CutCellParameterSlots& slots = {}) noexcept
{
    std::uint32_t max_slot = slots.volume_fraction;
    max_slot = std::max(max_slot, slots.side_indicator);
    max_slot = std::max(max_slot, slots.stabilization_scale);
    max_slot = std::max(max_slot, slots.quadrature_weight_sensitivity);
    for (const auto slot : slots.embedded_normal) {
        max_slot = std::max(max_slot, slot);
    }
    for (const auto slot : slots.measure_sensitivity) {
        max_slot = std::max(max_slot, slot);
    }
    for (const auto slot : slots.normal_sensitivity) {
        max_slot = std::max(max_slot, slot);
    }
    return max_slot + 1u;
}

[[nodiscard]] inline Real cutSideIndicatorValue(geometry::CutIntegrationSide side) noexcept
{
    switch (side) {
        case geometry::CutIntegrationSide::Negative:
            return Real(-1.0);
        case geometry::CutIntegrationSide::Positive:
            return Real(1.0);
        case geometry::CutIntegrationSide::Interface:
            return Real(0.0);
    }
    return Real(0.0);
}

[[nodiscard]] inline CutCellParameterVector cutCellParametersForRule(
    const geometry::CutQuadratureRule& rule,
    const CutCellParameterSlots& slots = {},
    Real stabilization_scale = Real(0.0),
    Real quadrature_weight_sensitivity = Real(0.0))
{
    CutCellParameterVector parameters(cutCellParameterCount(slots), Real(0.0));
    parameters[slots.volume_fraction] = rule.volume_fraction;
    parameters[slots.side_indicator] = cutSideIndicatorValue(rule.side);
    if (!rule.points.empty()) {
        parameters[slots.embedded_normal[0]] = rule.points.front().normal[0];
        parameters[slots.embedded_normal[1]] = rule.points.front().normal[1];
        parameters[slots.embedded_normal[2]] = rule.points.front().normal[2];
    }
    parameters[slots.stabilization_scale] = stabilization_scale;
    parameters[slots.quadrature_weight_sensitivity] = quadrature_weight_sensitivity;
    return parameters;
}

[[nodiscard]] inline FormExpr cutVolumeFraction(
    const CutCellParameterSlots& slots = {})
{
    return FormExpr::parameterRef(slots.volume_fraction);
}

[[nodiscard]] inline FormExpr cutSideIndicator(
    const CutCellParameterSlots& slots = {})
{
    return FormExpr::parameterRef(slots.side_indicator);
}

[[nodiscard]] inline FormExpr cutEmbeddedNormal(
    const CutCellParameterSlots& slots = {})
{
    return FormExpr::asVector({
        FormExpr::parameterRef(slots.embedded_normal[0]),
        FormExpr::parameterRef(slots.embedded_normal[1]),
        FormExpr::parameterRef(slots.embedded_normal[2])});
}

[[nodiscard]] inline FormExpr cutStabilizationScale(
    const CutCellParameterSlots& slots = {})
{
    return FormExpr::parameterRef(slots.stabilization_scale);
}

[[nodiscard]] inline FormExpr cutMeasureSensitivity(
    const CutCellParameterSlots& slots = {})
{
    return FormExpr::asVector({
        FormExpr::parameterRef(slots.measure_sensitivity[0]),
        FormExpr::parameterRef(slots.measure_sensitivity[1]),
        FormExpr::parameterRef(slots.measure_sensitivity[2])});
}

[[nodiscard]] inline FormExpr cutNormalSensitivity(
    const CutCellParameterSlots& slots = {})
{
    return FormExpr::asVector({
        FormExpr::parameterRef(slots.normal_sensitivity[0]),
        FormExpr::parameterRef(slots.normal_sensitivity[1]),
        FormExpr::parameterRef(slots.normal_sensitivity[2])});
}

[[nodiscard]] inline FormExpr cutQuadratureWeightSensitivity(
    const CutCellParameterSlots& slots = {})
{
    return FormExpr::parameterRef(slots.quadrature_weight_sensitivity);
}

struct CutCellFormTerminals {
    FormExpr volume_fraction{};
    FormExpr side_indicator{};
    FormExpr embedded_normal{};
    FormExpr stabilization_scale{};
    FormExpr measure_sensitivity{};
    FormExpr normal_sensitivity{};
    FormExpr quadrature_weight_sensitivity{};
};

[[nodiscard]] inline CutCellFormTerminals cutCellTerminals(
    const CutCellParameterSlots& slots = {})
{
    CutCellFormTerminals terminals;
    terminals.volume_fraction = cutVolumeFraction(slots);
    terminals.side_indicator = cutSideIndicator(slots);
    terminals.embedded_normal = cutEmbeddedNormal(slots);
    terminals.stabilization_scale = cutStabilizationScale(slots);
    terminals.measure_sensitivity = cutMeasureSensitivity(slots);
    terminals.normal_sensitivity = cutNormalSensitivity(slots);
    terminals.quadrature_weight_sensitivity = cutQuadratureWeightSensitivity(slots);
    return terminals;
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_CUTCELLFORMS_H
