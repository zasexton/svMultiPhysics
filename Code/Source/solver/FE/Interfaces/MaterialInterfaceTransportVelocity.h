/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_MATERIALINTERFACETRANSPORTVELOCITY_H
#define SVMP_FE_INTERFACES_MATERIALINTERFACETRANSPORTVELOCITY_H

/**
 * @file MaterialInterfaceTransportVelocity.h
 * @brief Declared phase-pair velocity used by material-interface transport.
 */

#include "Core/Types.h"

#include <array>
#include <cstdint>
#include <string>

namespace svmp::FE::interfaces {

/**
 * One immutable owner declaration for a sharp material-interface velocity.
 * The bulk value is selected from the matching phase. At the zero level set,
 * the trace is the declared complementary weighted phase average.
 */
struct MaterialInterfaceTransportVelocityDeclaration {
    int dimension{0};
    int interface_marker{-1};
    FieldId level_set_field{INVALID_FIELD_ID};
    FieldId negative_velocity_field{INVALID_FIELD_ID};
    FieldId positive_velocity_field{INVALID_FIELD_ID};
    Real level_set_isovalue{0.0};
    Real negative_trace_weight{0.5};
    Real positive_trace_weight{0.5};
    std::string geometry_domain_id{};
    std::string owner_component{};

    [[nodiscard]] friend bool operator==(
        const MaterialInterfaceTransportVelocityDeclaration&,
        const MaterialInterfaceTransportVelocityDeclaration&) = default;
};

enum class MaterialInterfaceVelocityRegion : std::uint8_t {
    NegativeBulk = 1u,
    InterfaceTrace = 2u,
    PositiveBulk = 3u,
};

struct MaterialInterfaceVelocitySample {
    std::array<Real, 3> value{{0.0, 0.0, 0.0}};
    MaterialInterfaceVelocityRegion region{
        MaterialInterfaceVelocityRegion::InterfaceTrace};
};

/** Validate declaration metadata independently of any field registry. */
void validateMaterialInterfaceTransportVelocityDeclaration(
    const MaterialInterfaceTransportVelocityDeclaration& declaration);

/** Evaluate the exact complementary weighted interface trace. */
[[nodiscard]] std::array<Real, 3> materialInterfaceCommonTraceVelocity(
    const MaterialInterfaceTransportVelocityDeclaration& declaration,
    const std::array<Real, 3>& negative_velocity,
    const std::array<Real, 3>& positive_velocity);

/** Select a sharp bulk value or the exact weighted trace at the isovalue. */
[[nodiscard]] MaterialInterfaceVelocitySample
selectMaterialInterfaceTransportVelocity(
    const MaterialInterfaceTransportVelocityDeclaration& declaration,
    Real level_set_value,
    const std::array<Real, 3>& negative_velocity,
    const std::array<Real, 3>& positive_velocity);

} // namespace svmp::FE::interfaces

#endif // SVMP_FE_INTERFACES_MATERIALINTERFACETRANSPORTVELOCITY_H
