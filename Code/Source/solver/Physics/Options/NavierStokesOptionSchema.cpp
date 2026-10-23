/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#include "Physics/Options/NavierStokesOptionSchema.h"

#include "FE/Systems/FESystem.h"

#include <stdexcept>
#include <string>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] const char* surfaceTensionFormName(
    const FreeSurfaceBoundary& bc) noexcept
{
    switch (bc.surface_tension_form) {
    case FreeSurfaceSurfaceTensionForm::Automatic:
        return "SurfaceStress";
    case FreeSurfaceSurfaceTensionForm::SurfaceStress:
        return "SurfaceStress";
    }
    return "Unsupported";
}

const char* freeSurfaceSurfaceTensionSemiImplicitName(
    FreeSurfaceSurfaceTensionSemiImplicit value) noexcept
{
    switch (value) {
    case FreeSurfaceSurfaceTensionSemiImplicit::None:
        return "None";
    case FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement:
        return "NormalIncrement";
    }
    return "Unsupported";
}

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc)
{
    const char* kind = isUnfittedLevelSet(bc) ? "_i" : "_b";
    return std::string(prefix) + kind + std::to_string(freeSurfaceMarker(bc));
}

// Input-facing name of a wall-owning contact law, used in validation messages
// shared by PrescribedContactAngle and DynamicContactAngle walls.
[[nodiscard]] std::string contactWallLawName(
    const FreeSurfaceContactLine& contact_line)
{
    return contactLineKind(contact_line) == ContactLineKind::PrescribedAngle
               ? "PrescribedContactAngle"
               : "DynamicContactAngle";
}

[[nodiscard]] const char* pressureStabilizationPolicyName(
    FreeSurfacePressureStabilizationPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfacePressureStabilizationPolicy::Enabled:
        return "Enabled";
    case FreeSurfacePressureStabilizationPolicy::Incremental:
        return "Incremental";
    case FreeSurfacePressureStabilizationPolicy::Disabled:
        return "Disabled";
    case FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder:
        return "DisabledForRefreshedFrozenHighOrder";
    }
    return "Unknown";
}

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "inactive_zero_surface_tension";
    }
    // SurfaceStress, the only form (decision D48).
    return "variational_surface_stress_generated_geometry";
}

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept
{
    switch (method) {
    case FreeSurfaceActiveDomainMethod::CutVolume:
        return "CutVolume";
    }
    return "Unknown";
}

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept
{
    switch (enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return "None";
    case FreeSurfaceKinematicEnforcement::MeshNitsche:
        return "MeshNitsche";
    }
    return "Unknown";
}

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative ? "Negative" : "Positive";
}

[[nodiscard]] FE::systems::MeshTangentialBoundaryPolicy
systemTangentialMeshPolicy(FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return FE::systems::MeshTangentialBoundaryPolicy::Free;
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return FE::systems::MeshTangentialBoundaryPolicy::SmoothingOnly;
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return FE::systems::MeshTangentialBoundaryPolicy::Prescribed;
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted "
        "free-surface tangential mesh policy");
}

[[nodiscard]] std::string fittedTangentialOperatorSource(
    int boundary_marker,
    FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Fitted free-surface natural tangential state on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "Fitted free-surface tangential surface smoothing on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Fitted free-surface prescribed tangential mesh velocity on "
               "marker " +
               std::to_string(boundary_marker);
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted tangential "
        "mesh policy source");
}

[[nodiscard]] std::string fittedFluidNormalDescriptorSource(
    const FreeSurfaceBoundary& bc)
{
    // MeshNitsche, the only normal relation, adds no fluid kinematic row.
    return "Fitted free-surface fluid natural dynamic condition on marker " +
           std::to_string(bc.boundary_marker);
}

[[nodiscard]] const char* freeSurfaceImplementationName(
    FreeSurfaceImplementation implementation) noexcept
{
    return implementation == FreeSurfaceImplementation::FittedALE
               ? "FittedALE"
               : "UnfittedLevelSet";
}

[[nodiscard]] const char* freeSurfaceBoundaryRoleName(
    FreeSurfaceBoundaryRole role) noexcept
{
    switch (role) {
    case FreeSurfaceBoundaryRole::ExteriorOnePhaseBoundary:
        return "ExteriorOnePhaseBoundary";
    case FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume:
        return "InternalMaterialInterfaceVolume";
    }
    return "Unknown";
}

[[nodiscard]] const char* freeSurfacePhysicalModelName(
    FreeSurfacePhysicalModel model) noexcept
{
    switch (model) {
    case FreeSurfacePhysicalModel::
        OnePhaseLiquidPrescribedExteriorPressure:
        return "one_phase_liquid_prescribed_exterior_pressure";
    }
    return nullptr;
}

[[nodiscard]] const char* normalKinematicPolicyName(
    FreeSurfaceNormalKinematicPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceNormalKinematicPolicy::None:
        return "None";
    case FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity:
        return "MatchFluidNormalVelocity";
    }
    return "Unknown";
}

[[nodiscard]] const char* tangentialMeshPolicyName(
    FreeSurfaceTangentialMeshPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Free";
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "SmoothingOnly";
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Prescribed";
    }
    return "Unknown";
}

[[nodiscard]] const char* contactLineKindName(ContactLineKind kind) noexcept
{
    switch (kind) {
    case ContactLineKind::None:
        return "None";
    case ContactLineKind::Pinned:
        return "Pinned";
    case ContactLineKind::PrescribedAngle:
        return "PrescribedAngle";
    case ContactLineKind::DynamicRenE:
        return "DynamicRenE";
    }
    return "Unknown";
}

[[nodiscard]] const char* geometryTangentPathName(
    FE::forms::GeometryTangentPath path) noexcept
{
    switch (path) {
    case FE::forms::GeometryTangentPath::Auto:
        return "Auto";
    case FE::forms::GeometryTangentPath::ADReference:
        return "ADReference";
    case FE::forms::GeometryTangentPath::SymbolicRequired:
        return "SymbolicRequired";
    case FE::forms::GeometryTangentPath::SymbolicWithADCheck:
        return "SymbolicWithADCheck";
    }
    return "Unknown";
}

[[nodiscard]] std::string activePhaseSignName(
    FreeSurfaceActiveDomain domain)
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "full_domain";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "negative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "positive";
    }
    return "unknown";
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
