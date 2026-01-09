/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"

namespace svmp {
namespace Physics {
namespace materials {
namespace fluid {

CarreauYasudaViscosity::CarreauYasudaViscosity(FE::Real mu0,
                                               FE::Real mu_inf,
                                               FE::Real lambda,
                                               FE::Real n,
                                               FE::Real a)
    : mu0_(mu0)
    , mu_inf_(mu_inf)
    , lambda_(lambda)
    , n_(n)
    , a_(a)
{
    if (!(mu0_ > 0.0)) {
        throw std::invalid_argument("CarreauYasudaViscosity: mu0 must be > 0");
    }
    if (!(mu_inf_ > 0.0)) {
        throw std::invalid_argument("CarreauYasudaViscosity: mu_inf must be > 0");
    }
    if (mu0_ < mu_inf_) {
        throw std::invalid_argument("CarreauYasudaViscosity: expected mu0 >= mu_inf");
    }
    if (!(lambda_ > 0.0)) {
        throw std::invalid_argument("CarreauYasudaViscosity: lambda must be > 0");
    }
    if (!(n_ > 0.0)) {
        throw std::invalid_argument("CarreauYasudaViscosity: n must be > 0");
    }
    if (!(a_ > 0.0)) {
        throw std::invalid_argument("CarreauYasudaViscosity: a must be > 0");
    }
}

} // namespace fluid
} // namespace materials
} // namespace Physics
} // namespace svmp

