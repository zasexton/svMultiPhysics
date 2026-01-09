#ifndef SVMP_PHYSICS_MATERIALS_FLUID_CARREAU_YASUDA_VISCOSITY_H
#define SVMP_PHYSICS_MATERIALS_FLUID_CARREAU_YASUDA_VISCOSITY_H

/**
 * @file CarreauYasudaViscosity.h
 * @brief Generalized-Newtonian Carreau–Yasuda viscosity model
 *
 * Input: scalar shear rate gamma
 * Output: scalar viscosity mu(gamma)
 *
 *   mu(gamma) = mu_inf + (mu0 - mu_inf) * (1 + (lambda*gamma)^a)^((n-1)/a)
 */

#include "Physics/Materials/Common/TensorOps.h"

#include "FE/Constitutive/ModelCRTP.h"

#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace fluid {

class CarreauYasudaViscosity final : public FE::constitutive::ModelCRTP<CarreauYasudaViscosity> {
public:
    using ValueKind = FE::forms::ConstitutiveModel::ValueKind;

    static constexpr ValueKind kExpectedInputKind = ValueKind::Scalar;

    CarreauYasudaViscosity(FE::Real mu0,
                           FE::Real mu_inf,
                           FE::Real lambda,
                           FE::Real n,
                           FE::Real a);

    template <class Scalar, class Workspace>
    [[nodiscard]] FE::forms::Value<Scalar> evaluateImpl(const FE::forms::Value<Scalar>& input,
                                                        int /*dim*/,
                                                        Workspace& ws) const
    {
        if (input.kind != FE::forms::Value<Scalar>::Kind::Scalar) {
            throw std::invalid_argument("CarreauYasudaViscosity: expected scalar input");
        }

        const auto gamma = input.s;
        if (common::scalarValue(gamma) < 0.0) {
            throw std::invalid_argument("CarreauYasudaViscosity: shear rate must be >= 0");
        }

        const auto mu0 = common::constant<Scalar>(mu0_, ws);
        const auto mu_inf = common::constant<Scalar>(mu_inf_, ws);
        const auto lam = common::constant<Scalar>(lambda_, ws);

        const auto lam_gamma = common::mul(lam, gamma, ws);
        const auto lam_gamma_a = common::pow(lam_gamma, a_, ws);

        const auto one = common::constant<Scalar>(1.0, ws);
        const auto base = common::add(one, lam_gamma_a, ws);
        const auto exponent = (n_ - 1.0) / a_;
        const auto base_exp = common::pow(base, exponent, ws);

        const auto mu_delta = common::sub(mu0, mu_inf, ws);
        const auto scaled = common::mul(mu_delta, base_exp, ws);

        FE::forms::Value<Scalar> out;
        out.kind = FE::forms::Value<Scalar>::Kind::Scalar;
        out.s = common::add(mu_inf, scaled, ws);
        return out;
    }

private:
    FE::Real mu0_{0.0};
    FE::Real mu_inf_{0.0};
    FE::Real lambda_{0.0};
    FE::Real n_{1.0};
    FE::Real a_{2.0};
};

} // namespace fluid
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_FLUID_CARREAU_YASUDA_VISCOSITY_H

