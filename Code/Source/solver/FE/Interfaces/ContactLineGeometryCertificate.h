/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_CONTACTLINEGEOMETRYCERTIFICATE_H
#define SVMP_FE_INTERFACES_CONTACTLINEGEOMETRYCERTIFICATE_H

/**
 * @file ContactLineGeometryCertificate.h
 * @brief Geometry certificate of a contact line of a generated interface on a planar wall
 *
 * A contact-line term or law on the wall footprint of a generated interface
 * needs (a) a wall whose faces are first-order planes with the configured
 * outward normal, and (b) contact rules whose physical boundary normal is
 * that wall normal and whose interface normal is transverse to the wall, so
 * that the wall-tangential conormal is well defined.  Both are checked with
 * the tolerances below; (a) once, summed over the ranks, (b) on every cut
 * context the caller passes (typically at every refresh).
 */

#include "Core/Types.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace assembly {
class CutIntegrationContext;
} // namespace assembly

namespace systems {
class FESystem;
} // namespace systems

namespace interfaces {

/// 1 - n_configured . n_physical, per wall face and per contact point.
inline constexpr Real kContactLineAlignmentTolerance = Real{1.0e-8};
/// Smallest length of the wall-tangential part of the unit interface normal.
inline constexpr Real kContactLineMinimumTransverseSine = Real{1.0e-6};

/// Evidence that every owned face on a wall marker is a first-order plane
/// with the configured outward normal.
struct PlanarWallEvidence {
    std::uint64_t face_count{0u};
    std::uint64_t invalid_face_count{0u};
    Real minimum_alignment{Real{1.0}};
    std::string first_failure{};   ///< this rank's first failure, if any
};

/**
 * @brief Planar-wall evidence of @p wall_marker on the system's mesh: face
 * counts summed and the alignment minimized over the active communicator
 * (when MPI is initialized and not finalized); the first failure stays local.
 */
[[nodiscard]] PlanarWallEvidence planarWallEvidence(
    const systems::FESystem& system,
    int wall_marker,
    const std::array<Real, 3>& configured_normal);

/// One contact line: the generated contact marker on a wall.
struct ContactLineGeometryCheck {
    int contact_marker{-1};
    int wall_marker{-1};
    std::array<Real, 3> wall_normal{};   ///< configured unit normal, out of the fluid
    std::string owner{"ContactLineGeometryCertificate"};   ///< message prefix
};

/**
 * @brief Checks the contact rules of @p check in @p context.
 *
 * Nothing is checked without a context or when the marker has no rules (the
 * interface need not meet every wall).  Every rule must be a nonempty
 * codimension-two rule whose physical boundary normal (reference-frame rules
 * mapped to the physical frame) is aligned with the wall normal and whose
 * interface normal has a wall-tangential part of length at least
 * kContactLineMinimumTransverseSine; otherwise std::invalid_argument.  At
 * FE_LOG_LEVEL=DEBUG one line reports the samples.
 */
void certifyContactLineRules(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext* context,
    const ContactLineGeometryCheck& check);

/**
 * @brief certifyContactLineRules with its failure agreed on over the ranks:
 * systems::agreeOnLocalPhase(system, failure, @p local_followup_available,
 * @p phase, owner).
 */
[[nodiscard]] bool certifyContactLineRulesCollectively(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext* context,
    const ContactLineGeometryCheck& check,
    bool local_followup_available,
    std::string_view phase);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_CONTACTLINEGEOMETRYCERTIFICATE_H
