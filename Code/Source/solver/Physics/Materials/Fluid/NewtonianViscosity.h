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
#include "FE/Forms/JIT/InlinableConstitutiveModel.h"

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace fluid {

class NewtonianViscosity final
    : public FE::constitutive::ModelCRTP<NewtonianViscosity>
    , public FE::forms::InlinableConstitutiveModel {
public:
    using ValueKind = FE::forms::ConstitutiveModel::ValueKind;

    static constexpr ValueKind kExpectedInputKind = ValueKind::Scalar;

    explicit NewtonianViscosity(FE::Real mu);

    [[nodiscard]] const FE::forms::InlinableConstitutiveModel*
    inlinable() const noexcept override
    {
        return this;
    }

    [[nodiscard]] std::uint64_t kindId() const noexcept override
    {
        return FE::forms::InlinableConstitutiveModel::fnv1a64(
            "Physics::materials::fluid::NewtonianViscosity");
    }

    [[nodiscard]] FE::forms::MaterialStateAccess stateAccess() const noexcept override
    {
        return FE::forms::MaterialStateAccess::None;
    }

    [[nodiscard]] FE::forms::InlinedConstitutiveExpansion inlineExpand(
        std::span<const FE::forms::FormExpr> inputs,
        const FE::forms::InlinableConstitutiveContext& /*ctx*/) const override
    {
        if (inputs.size() != 1u) {
            throw std::invalid_argument(
                "NewtonianViscosity::inlineExpand: expected exactly 1 input");
        }
        FE::forms::InlinedConstitutiveExpansion out;
        out.outputs.push_back(FE::forms::FormExpr::constant(mu_));
        return out;
    }

    [[nodiscard]] std::optional<FE::analysis::ConstitutiveLawMetadata>
    constitutiveLawMetadata(std::size_t output_index) const override
    {
        if (output_index != 0u) {
            throw std::invalid_argument(
                "NewtonianViscosity::constitutiveLawMetadata: output_index out of range");
        }
        return FE::analysis::dynamicViscosityMetadata(
            FE::INVALID_FIELD_ID,
            mu_);
    }

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
