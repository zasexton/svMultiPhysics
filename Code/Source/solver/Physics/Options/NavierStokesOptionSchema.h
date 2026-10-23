/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#ifndef SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_SCHEMA_H
#define SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_SCHEMA_H

/**
 * @file NavierStokesOptionSchema.h
 * @brief The Navier-Stokes option schema: names of option values
 *
 * The input, artifact and log names of the free-surface option values, the
 * mapping of the fitted tangential mesh policy to the FE system policy, and
 * the descriptor texts of the fitted rows.  Moved unchanged from
 * FreeSurface/FreeSurfaceQueries (decisions D50 and D52); the option schema
 * of D47 P2 extends this file.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace svmp::FE::systems {
enum class MeshTangentialBoundaryPolicy : std::uint8_t;
} // namespace svmp::FE::systems

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept;

[[nodiscard]] const char* surfaceTensionFormName(
    const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] const char* freeSurfaceSurfaceTensionSemiImplicitName(
    FreeSurfaceSurfaceTensionSemiImplicit value) noexcept;

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc);

[[nodiscard]] std::string contactWallLawName(
    const FreeSurfaceContactLine& contact_line);

[[nodiscard]] const char* pressureStabilizationPolicyName(
    FreeSurfacePressureStabilizationPolicy policy) noexcept;

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept;

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept;

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept;

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept;

[[nodiscard]] FE::systems::MeshTangentialBoundaryPolicy
systemTangentialMeshPolicy(FreeSurfaceTangentialMeshPolicy policy);

[[nodiscard]] std::string fittedTangentialOperatorSource(
    int boundary_marker,
    FreeSurfaceTangentialMeshPolicy policy);

[[nodiscard]] std::string fittedFluidNormalDescriptorSource(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] const char* freeSurfaceImplementationName(
    FreeSurfaceImplementation implementation) noexcept;

[[nodiscard]] const char* freeSurfaceBoundaryRoleName(
    FreeSurfaceBoundaryRole role) noexcept;

[[nodiscard]] const char* freeSurfacePhysicalModelName(
    FreeSurfacePhysicalModel model) noexcept;

[[nodiscard]] const char* normalKinematicPolicyName(
    FreeSurfaceNormalKinematicPolicy policy) noexcept;

[[nodiscard]] const char* tangentialMeshPolicyName(
    FreeSurfaceTangentialMeshPolicy policy) noexcept;

[[nodiscard]] const char* contactLineKindName(ContactLineKind kind) noexcept;

[[nodiscard]] const char* geometryTangentPathName(
    FE::forms::GeometryTangentPath path) noexcept;

[[nodiscard]] std::string activePhaseSignName(
    FreeSurfaceActiveDomain domain);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_SCHEMA_H
