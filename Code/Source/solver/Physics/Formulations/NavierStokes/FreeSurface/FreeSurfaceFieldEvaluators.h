/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

/**
 * @file
 * @brief Point evaluators of FE fields for the free-surface discrete
 *        functionals and accepted-stage diagnostics.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRunSupport.h"

#include "FE/Core/Types.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"

#include <span>
#include <string_view>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator
makeFreeSurfaceVelocityEvaluator(
    const FreeSurfaceSystemHandle& sim,
    svmp::FE::FieldId velocity_field,
    std::span<const svmp::FE::Real> solution,
    std::string_view diagnostic_context,
    bool allow_scalar_product_fast_path = true);

svmp::FE::interfaces::FreeSurfaceGeometryScalarEvaluator
makeFreeSurfaceScalarValueEvaluator(
    const FreeSurfaceSystemHandle& sim,
    svmp::FE::FieldId scalar_field,
    std::span<const svmp::FE::Real> solution,
    std::string_view diagnostic_context);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
