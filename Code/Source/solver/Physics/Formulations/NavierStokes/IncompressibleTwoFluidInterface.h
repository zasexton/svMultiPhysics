/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_TWO_FLUID_INTERFACE_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_TWO_FLUID_INTERFACE_H

/**
 * @file IncompressibleTwoFluidInterface.h
 * @brief Weighted symmetric coupling on a generated material interface.
 */

#include "FE/Core/Types.h"
#include "FE/Forms/FormExpr.h"

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

struct IncompressibleTwoFluidInterfaceParameters {
  int dimension{0};
  int interface_marker{-1};
  FE::Real negative_density{1.0};
  FE::Real positive_density{1.0};
  FE::Real negative_viscosity{1.0};
  FE::Real positive_viscosity{1.0};
  FE::Real nitsche_gamma{20.0};
  FE::Real surface_tension{0.0};
  bool include_transient_penalty{true};
};

struct IncompressibleTwoFluidInterfaceWeights {
  FE::Real negative_traction{0.5};
  FE::Real positive_traction{0.5};
  FE::Real negative_complement{0.5};
  FE::Real positive_complement{0.5};
  FE::Real harmonic_viscosity{1.0};
  FE::Real harmonic_density{1.0};
};

/**
 * Each returned channel is already integrated on the generated interface.
 * Keeping the channels separate permits accepted-stage work and consistency
 * diagnostics to use the identical production expressions.
 */
struct IncompressibleTwoFluidInterfaceForms {
  IncompressibleTwoFluidInterfaceWeights weights{};
  FE::forms::FormExpr consistency{};
  FE::forms::FormExpr adjoint{};
  FE::forms::FormExpr penalty{};
  FE::forms::FormExpr surface_energy{};
  FE::forms::FormExpr residual{};
};

[[nodiscard]] IncompressibleTwoFluidInterfaceWeights
incompressibleTwoFluidInterfaceWeights(
    const IncompressibleTwoFluidInterfaceParameters &parameters);

[[nodiscard]] IncompressibleTwoFluidInterfaceForms
buildIncompressibleTwoFluidInterfaceForms(
    const FE::forms::FormExpr &negative_velocity,
    const FE::forms::FormExpr &negative_pressure,
    const FE::forms::FormExpr &negative_velocity_test,
    const FE::forms::FormExpr &negative_pressure_test,
    const FE::forms::FormExpr &positive_velocity,
    const FE::forms::FormExpr &positive_pressure,
    const FE::forms::FormExpr &positive_velocity_test,
    const FE::forms::FormExpr &positive_pressure_test,
    const IncompressibleTwoFluidInterfaceParameters &parameters);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_TWO_FLUID_INTERFACE_H
