/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Materials/Fluid/NewtonianViscosity.h"

#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace fluid {

FE::forms::Law newtonianViscosity(FE::Real mu)
{
    if (!(mu > 0.0)) {
        throw std::invalid_argument("NewtonianViscosity: mu must be > 0");
    }
    using Kind = FE::forms::ConstitutiveModel::ValueKind;
    return FE::forms::law(
        "Physics::materials::fluid::NewtonianViscosity", Kind::Scalar, Kind::Scalar,
        [mu](const FE::forms::FormExpr& /*gamma*/, int /*dim*/) { return FE::forms::FormExpr::constant(mu); },
        FE::analysis::dynamicViscosityMetadata(FE::INVALID_FIELD_ID, mu));
}

} // namespace fluid
} // namespace materials
} // namespace Physics
} // namespace svmp
