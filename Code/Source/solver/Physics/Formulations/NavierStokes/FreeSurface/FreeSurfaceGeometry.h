/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_GEOMETRY_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_GEOMETRY_H

/**
 * @file FreeSurfaceGeometry.h
 * @brief The level-set geometry of a free surface and the fields its terms read
 *
 * An unfitted free surface Gamma_h = {phi_h = c} is a FE level-set geometry
 * (FE/LevelSet/LevelSetGeometry.h) bound from the free-surface options; its
 * active side is the liquid.  Moved from IncompressibleNavierStokesVMSModule.cpp:
 * the binding and its generated markers, the active (liquid) domain, the
 * integration on the free surface and the liquid, the fields the
 * free-surface terms read (level set, curvature, the semi-implicit reference
 * velocity), and the extra trial fields those reads add to the residual.
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

struct ActivePressureDomain {
    const FreeSurfaceBoundary* boundary{nullptr};
    FreeSurfaceActiveDomain active_domain{FreeSurfaceActiveDomain::None};
};

struct ActiveVolumeDomain {
    int interface_marker{-1};
    FE::forms::CutVolumeSide side{FE::forms::CutVolumeSide::Negative};
    FreeSurfaceActiveDomainMethod method{FreeSurfaceActiveDomainMethod::CutVolume};
};

[[nodiscard]] std::optional<ActivePressureDomain>
activePressureDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces);

[[nodiscard]] FE::forms::FormExpr generatedInterfaceOutwardNormal(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] FE::FieldId resolveLevelSetFieldId(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] FE::level_set::LevelSetGeometry freeSurfaceLevelSetGeometry(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey
generatedUnfittedContactLineMarkerKey(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] int generatedUnfittedContactLineMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] int generatedActiveBoundaryMarker(
    const FreeSurfaceBoundary& bc,
    int physical_boundary_marker,
    const FE::systems::FESystem& system);

[[nodiscard]] std::optional<int> sharpActiveBoundaryMarkerFor(
    int physical_boundary_marker,
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system);

[[nodiscard]] FreeSurfaceBoundary withResolvedInterfaceMarker(
    FreeSurfaceBoundary bc,
    const FE::systems::FESystem& system);

[[nodiscard]] std::string freeSurfaceCurvatureTangentPolicyName(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] FE::forms::FormExpr unfittedInterfaceNormal(
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& phi);

[[nodiscard]] FE::forms::FormExpr integrateOnFreeSurface(
    const FE::forms::FormExpr& integrand,
    const FreeSurfaceBoundary& bc,
    bool ale_enabled);

[[nodiscard]] std::optional<ActiveVolumeDomain> activeVolumeDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces);

[[nodiscard]] FE::forms::FormExpr integrateOnActiveVolume(
    const FE::forms::FormExpr& integrand,
    const std::optional<ActiveVolumeDomain>& active_domain);

[[nodiscard]] FE::forms::FormExpr freeSurfaceLevelSet(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] FE::forms::FormExpr freeSurfaceSemiImplicitReferenceVelocity(
    const FE::systems::FESystem& system);

[[nodiscard]] FE::forms::FormExpr unfittedTractionCurvature(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi);

void appendUnfittedFreeSurfaceLevelSetTrialFields(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system,
    FE::systems::FormInstallOptions& install);

void appendUnfittedFreeSurfaceCurvatureTrialFields(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system,
    FE::systems::FormInstallOptions& install);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_GEOMETRY_H
