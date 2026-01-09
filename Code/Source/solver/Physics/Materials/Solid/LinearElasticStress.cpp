/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Materials/Solid/LinearElasticStress.h"

namespace svmp {
namespace Physics {
namespace materials {
namespace solid {

LinearElasticStress::LinearElasticStress(FE::Real lambda, FE::Real mu)
    : lambda_(lambda)
    , mu_(mu)
{
    if (mu_ <= 0.0) {
        throw std::invalid_argument("LinearElasticStress: mu must be > 0");
    }
    if (lambda_ < 0.0) {
        throw std::invalid_argument("LinearElasticStress: lambda must be >= 0");
    }
}

} // namespace solid
} // namespace materials
} // namespace Physics
} // namespace svmp

