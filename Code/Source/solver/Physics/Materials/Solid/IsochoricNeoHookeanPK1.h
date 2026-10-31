#ifndef SVMP_PHYSICS_MATERIALS_SOLID_ISOCHORIC_NEO_HOOKEAN_PK1_H
#define SVMP_PHYSICS_MATERIALS_SOLID_ISOCHORIC_NEO_HOOKEAN_PK1_H

/**
 * @file IsochoricNeoHookeanPK1.h
 * @brief Isochoric Neo-Hookean first Piola stress for mixed solid formulations
 *
 * Input: deformation gradient F.
 * Output: deviatoric/isochoric first Piola stress
 *
 *   Pdev = F S_iso,   S_iso = mu J^{-2/d} (I - tr(C)/d C^{-1}),   C = F^T F,   J = det F.
 *
 * The volumetric response is intentionally omitted. Mixed formulations such as
 * ustruct provide the pressure/volumetric part through a separate pressure
 * unknown.  The law is written as forms (forms::law): setup expands each call,
 * so the symbolic tangent and the JIT see the law itself.
 */

#include "FE/Constitutive/ExpressionLaw.h"

namespace svmp {
namespace Physics {
namespace materials {
namespace solid {

/// The law, for mu > 0; its kind id is that of the former class.
[[nodiscard]] FE::forms::Law isochoricNeoHookeanPK1(FE::Real mu);

} // namespace solid
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_SOLID_ISOCHORIC_NEO_HOOKEAN_PK1_H
