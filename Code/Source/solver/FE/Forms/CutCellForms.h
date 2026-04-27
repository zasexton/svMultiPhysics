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

#include "Forms/FormExpr.h"

#include <array>
#include <cstdint>

namespace svmp {
namespace FE {
namespace forms {

struct CutCellParameterSlots {
    std::uint32_t volume_fraction{0u};
    std::uint32_t side_indicator{1u};
    std::array<std::uint32_t, 3> embedded_normal{{2u, 3u, 4u}};
    std::uint32_t stabilization_scale{5u};
};

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

struct CutCellFormTerminals {
    FormExpr volume_fraction{};
    FormExpr side_indicator{};
    FormExpr embedded_normal{};
    FormExpr stabilization_scale{};
};

[[nodiscard]] inline CutCellFormTerminals cutCellTerminals(
    const CutCellParameterSlots& slots = {})
{
    CutCellFormTerminals terminals;
    terminals.volume_fraction = cutVolumeFraction(slots);
    terminals.side_indicator = cutSideIndicator(slots);
    terminals.embedded_normal = cutEmbeddedNormal(slots);
    terminals.stabilization_scale = cutStabilizationScale(slots);
    return terminals;
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_CUTCELLFORMS_H
