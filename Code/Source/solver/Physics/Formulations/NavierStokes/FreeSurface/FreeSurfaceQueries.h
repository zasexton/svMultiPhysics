/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_QUERIES_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_QUERIES_H

/**
 * @file FreeSurfaceQueries.h
 * @brief What a free-surface declaration selects: predicates, contact-line accessors, names
 *
 * Pure functions of the free-surface options, moved from
 * IncompressibleNavierStokesVMSModule.cpp: the implementation, role and
 * surface-tension form, the contact-line configuration, literal values and
 * their checks, the cut pressure-stabilization policy, and the names of
 * option values that logs, messages and the configuration artifact print.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

enum class ContactLineKind : std::uint8_t {
    None,
    Pinned,
    PrescribedAngle,
    DynamicRenE
};

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept;

[[nodiscard]] bool isUnfittedLevelSet(const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] bool isExteriorOnePhaseBoundary(
    const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] bool isInternalMaterialInterfaceVolume(
    const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] bool usesSurfaceStress(
    const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] const char* surfaceTensionFormName(
    const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] bool surfaceStressActive(
    const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] int freeSurfaceMarker(const FreeSurfaceBoundary& bc);

[[nodiscard]] bool useFittedCurrentGeometry(const FreeSurfaceBoundary& bc,
                                            bool ale_enabled) noexcept;

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc);

[[nodiscard]] ContactLineKind contactLineKind(
    const FreeSurfaceContactLine& contact_line) noexcept;

[[nodiscard]] std::string contactWallLawName(
    const FreeSurfaceContactLine& contact_line);

[[nodiscard]] int contactLineWallBoundaryMarker(
    const FreeSurfaceContactLine& contact_line) noexcept;

[[nodiscard]] int contactLineMarker(
    const FreeSurfaceContactLine& contact_line) noexcept;

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineAngleRadians(const FreeSurfaceContactLine& contact_line);

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineMobility(const FreeSurfaceContactLine& contact_line);

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineSlipLength(const FreeSurfaceContactLine& contact_line);

[[nodiscard]] bool contactLineHasNavierSlip(
    const FreeSurfaceContactLine& contact_line) noexcept;

[[nodiscard]] int contactLineConstraintMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line);

[[nodiscard]] FE::Real constantScalarValueOrThrow(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

[[nodiscard]] bool hasWallAwareContactLaw(
    const FreeSurfaceBoundary& boundary);

[[nodiscard]] std::array<FE::Real, 3> normalizedWallNormal(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line);

void validateWallNormalDimension(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim,
    std::string_view context);

void validateFiniteConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

void validateNonnegativeConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

void validatePositiveConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

[[nodiscard]] FE::forms::FormExpr wallNormalExpression(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim);

[[nodiscard]] std::string normalizedFreeSurfaceOptionToken(std::string value);

[[nodiscard]] bool isRefreshedFrozenGeometryTangent(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] const char* pressureStabilizationPolicyName(
    FreeSurfacePressureStabilizationPolicy policy) noexcept;

[[nodiscard]] const char* pressureStabilizationDisabledReason(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] bool pressureStabilizationActive(const FreeSurfaceBoundary& bc);

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept;

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept;

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept;

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept;

[[nodiscard]] FE::forms::CutVolumeSide activeDomainSide(
    FreeSurfaceActiveDomain domain) noexcept;

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

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_QUERIES_H
