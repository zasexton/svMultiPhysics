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

FE::forms::Law carreauYasudaViscosityLaw(FE::Real mu0,
                                         FE::Real mu_inf,
                                         FE::Real lambda,
                                         FE::Real n,
                                         FE::Real a)
{
    const CarreauYasudaViscosity validated(mu0, mu_inf, lambda, n, a);
    using Kind = FE::forms::ConstitutiveModel::ValueKind;
    return FE::forms::law(
        "Physics::materials::fluid::CarreauYasudaViscosity", Kind::Scalar, Kind::Scalar,
        [=](const FE::forms::FormExpr& gamma, int /*dim*/) {
            using FE::forms::FormExpr;
            const auto base = FormExpr::constant(1.0) + FE::forms::pow(FormExpr::constant(lambda) * gamma,
                                                                       FormExpr::constant(a));
            return FormExpr::constant(mu_inf) +
                   (FormExpr::constant(mu0) - FormExpr::constant(mu_inf)) *
                       FE::forms::pow(base, FormExpr::constant((n - 1.0) / a));
        },
        validated.constitutiveLawMetadata(0u));
}

} // namespace fluid
} // namespace materials
} // namespace Physics
} // namespace svmp

