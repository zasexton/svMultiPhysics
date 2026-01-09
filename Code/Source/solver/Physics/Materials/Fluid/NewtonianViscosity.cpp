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

NewtonianViscosity::NewtonianViscosity(FE::Real mu)
    : mu_(mu)
{
    if (!(mu_ > 0.0)) {
        throw std::invalid_argument("NewtonianViscosity: mu must be > 0");
    }
}

} // namespace fluid
} // namespace materials
} // namespace Physics
} // namespace svmp

