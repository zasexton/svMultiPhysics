/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_PRESSURE_GAUGE_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_PRESSURE_GAUGE_H

/**
 * @file FreeSurfacePressureGauge.h
 * @brief The free-surface traction anchors the pressure of a cut liquid
 *
 * With a natural traction on Gamma_h the pressure constant is fixed by the
 * exterior pressure, provided the generated interface has a positive
 * measure.  Moved from IncompressibleNavierStokesVMSModule.cpp: the measure
 * evidence (collective reductions), the global kernel that checks it, and
 * the registration of the anchor.
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

/// The natural traction on an unfitted CutVolume free surface anchors the
/// pressure constant: the gauge-registry evidence, and the global kernel and
/// cut-context callback that certify |Gamma_h| > 0 at every refresh.
void anchorPressureOnFreeSurface(
    FE::systems::FESystem& system,
    const std::string& operator_tag,
    FE::FieldId p_id,
    const FreeSurfaceBoundary& boundary);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_PRESSURE_GAUGE_H
