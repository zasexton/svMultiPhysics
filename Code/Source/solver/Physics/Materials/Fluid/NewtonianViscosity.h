#ifndef SVMP_PHYSICS_MATERIALS_FLUID_NEWTONIAN_VISCOSITY_H
#define SVMP_PHYSICS_MATERIALS_FLUID_NEWTONIAN_VISCOSITY_H

/**
 * @file NewtonianViscosity.h
 * @brief Constant (Newtonian) viscosity model: mu = const
 *
 * Input: scalar shear rate gamma (ignored; accepted for uniform call sites)
 * Output: scalar viscosity mu
 */

#include "Physics/Materials/Common/TensorOps.h"

#include "FE/Constitutive/ModelCRTP.h"

#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace fluid {

class NewtonianViscosity final : public FE::constitutive::ModelCRTP<NewtonianViscosity> {
public:
    using ValueKind = FE::forms::ConstitutiveModel::ValueKind;

    static constexpr ValueKind kExpectedInputKind = ValueKind::Scalar;

    explicit NewtonianViscosity(FE::Real mu);

    template <class Scalar, class Workspace>
    [[nodiscard]] FE::forms::Value<Scalar> evaluateImpl(const FE::forms::Value<Scalar>& input,
                                                        int /*dim*/,
                                                        Workspace& ws) const
    {
        if (input.kind != FE::forms::Value<Scalar>::Kind::Scalar) {
            throw std::invalid_argument("NewtonianViscosity: expected scalar input");
        }

        FE::forms::Value<Scalar> out;
        out.kind = FE::forms::Value<Scalar>::Kind::Scalar;
        out.s = common::constant<Scalar>(mu_, ws);
        return out;
    }

private:
    FE::Real mu_{0.0};
};

} // namespace fluid
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_FLUID_NEWTONIAN_VISCOSITY_H

