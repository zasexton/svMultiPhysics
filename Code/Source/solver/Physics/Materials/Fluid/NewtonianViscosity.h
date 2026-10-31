#ifndef SVMP_PHYSICS_MATERIALS_FLUID_NEWTONIAN_VISCOSITY_H
#define SVMP_PHYSICS_MATERIALS_FLUID_NEWTONIAN_VISCOSITY_H

/**
 * @file NewtonianViscosity.h
 * @brief Constant (Newtonian) viscosity model: mu = const
 *
 * Input: scalar shear rate gamma (ignored; accepted for uniform call sites)
 * Output: scalar viscosity mu
 *
 * The law is written as forms (forms::law) and publishes its dynamic
 * viscosity metadata.
 */

#include "FE/Constitutive/ExpressionLaw.h"

namespace svmp {
namespace Physics {
namespace materials {
namespace fluid {

/// The law, for mu > 0; its kind id is that of the former class.
[[nodiscard]] FE::forms::Law newtonianViscosity(FE::Real mu);

} // namespace fluid
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_FLUID_NEWTONIAN_VISCOSITY_H
