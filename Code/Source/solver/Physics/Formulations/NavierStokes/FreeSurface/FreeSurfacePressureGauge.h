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
 * measure: an FE::constraints::InterfaceMeasureGaugeAnchor with the
 * module's texts (gauge source, kernel and callback names, messages, and the
 * kernel's manifest label).
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Core/Types.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <string>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// The anchor of the pressure constant by the natural traction on the
/// unfitted CutVolume free surface @p boundary (ConstraintPlan::anchorGauge).
[[nodiscard]] FE::constraints::InterfaceMeasureGaugeAnchor freeSurfacePressureAnchor(
    FE::FieldId p_id,
    const FreeSurfaceBoundary& boundary);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_PRESSURE_GAUGE_H
