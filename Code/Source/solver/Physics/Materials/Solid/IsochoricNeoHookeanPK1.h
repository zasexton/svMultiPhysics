#ifndef SVMP_PHYSICS_MATERIALS_SOLID_ISOCHORIC_NEO_HOOKEAN_PK1_H
#define SVMP_PHYSICS_MATERIALS_SOLID_ISOCHORIC_NEO_HOOKEAN_PK1_H

/**
 * @file IsochoricNeoHookeanPK1.h
 * @brief Isochoric Neo-Hookean first Piola stress for mixed solid formulations
 *
 * Input: deformation gradient F.
 * Output: deviatoric/isochoric first Piola stress Pdev.
 *
 * The volumetric response is intentionally omitted. Mixed formulations such as
 * ustruct provide the pressure/volumetric part through a separate pressure
 * unknown.
 */

#include "Physics/Materials/Common/TensorOps.h"

#include "FE/Constitutive/ModelCRTP.h"
#include "FE/Forms/JIT/InlinableConstitutiveModel.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace solid {

class IsochoricNeoHookeanPK1 final
    : public FE::constitutive::ModelCRTP<IsochoricNeoHookeanPK1>
    , public FE::forms::InlinableConstitutiveModel {
public:
    using ValueKind = FE::forms::ConstitutiveModel::ValueKind;

    static constexpr ValueKind kExpectedInputKind = ValueKind::Matrix;

    explicit IsochoricNeoHookeanPK1(FE::Real mu);

    [[nodiscard]] const FE::forms::InlinableConstitutiveModel*
    inlinable() const noexcept override
    {
        return this;
    }

    [[nodiscard]] FE::forms::ConstitutiveModel::OutputSpec
    outputSpec(std::size_t output_index) const override
    {
        if (output_index != 0u) {
            throw std::invalid_argument("IsochoricNeoHookeanPK1::outputSpec: output_index out of range");
        }
        return FE::forms::ConstitutiveModel::OutputSpec{ValueKind::Matrix};
    }

    [[nodiscard]] std::uint64_t kindId() const noexcept override
    {
        return FE::forms::InlinableConstitutiveModel::fnv1a64(
            "Physics::materials::solid::IsochoricNeoHookeanPK1");
    }

    [[nodiscard]] FE::forms::MaterialStateAccess stateAccess() const noexcept override
    {
        return FE::forms::MaterialStateAccess::None;
    }

    [[nodiscard]] FE::forms::InlinedConstitutiveExpansion inlineExpand(
        std::span<const FE::forms::FormExpr> inputs,
        const FE::forms::InlinableConstitutiveContext& ctx) const override
    {
        if (inputs.size() != 1u) {
            throw std::invalid_argument("IsochoricNeoHookeanPK1::inlineExpand: expected exactly 1 input");
        }
        if (ctx.dim != 2 && ctx.dim != 3) {
            throw std::invalid_argument("IsochoricNeoHookeanPK1::inlineExpand: only dim=2 or dim=3 supported");
        }

        const auto constant = [](FE::Real value) {
            return FE::forms::FormExpr::constant(value);
        };

        const auto& F = inputs.front();
        const auto C = FE::forms::transpose(F) * F;
        const auto Cinv = FE::forms::inv(C);
        const auto J = FE::forms::det(F);
        const auto Jm2d = FE::forms::pow(J, constant(-2.0 / static_cast<FE::Real>(ctx.dim)));
        const auto I = FE::forms::FormExpr::identity(ctx.dim);
        const auto Siso =
            constant(mu_) * Jm2d *
            (I - (FE::forms::trace(C) / constant(static_cast<FE::Real>(ctx.dim))) * Cinv);

        FE::forms::InlinedConstitutiveExpansion out;
        out.outputs.push_back(F * Siso);
        return out;
    }

    template <class Scalar, class Workspace>
    [[nodiscard]] FE::forms::Value<Scalar> evaluateImpl(const FE::forms::Value<Scalar>& input,
                                                        int dim,
                                                        Workspace& ws) const
    {
        if (!common::isMatrixLike<Scalar>(input.kind)) {
            throw std::invalid_argument("IsochoricNeoHookeanPK1: expected matrix-like input");
        }
        if (dim != 2 && dim != 3) {
            throw std::invalid_argument("IsochoricNeoHookeanPK1: only dim=2 or dim=3 supported");
        }

        const auto J = common::det(input, dim, ws);
        if (common::scalarValue(J) <= 0.0) {
            throw std::invalid_argument("IsochoricNeoHookeanPK1: det(F) must be > 0");
        }

        FE::forms::Value<Scalar> C;
        C.kind = FE::forms::Value<Scalar>::Kind::Matrix;
        C.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                auto cij = common::constant<Scalar>(0.0, ws);
                for (int k = 0; k < dim; ++k) {
                    const auto Fki = input.matrixAt(static_cast<std::size_t>(k), static_cast<std::size_t>(i));
                    const auto Fkj = input.matrixAt(static_cast<std::size_t>(k), static_cast<std::size_t>(j));
                    cij = common::add(cij, common::mul(Fki, Fkj, ws), ws);
                }
                C.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = cij;
            }
        }

        const auto Cinv = common::inverse(C, dim, ws);

        auto trC = common::constant<Scalar>(0.0, ws);
        for (int i = 0; i < dim; ++i) {
            trC = common::add(trC, C.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(i)), ws);
        }

        const auto mu = common::constant<Scalar>(mu_, ws);
        const auto Jm2d = common::pow(J, -2.0 / static_cast<FE::Real>(dim), ws);
        const auto scale = common::mul(mu, Jm2d, ws);
        const auto trC_over_dim = common::div(trC, common::constant<Scalar>(static_cast<FE::Real>(dim), ws), ws);

        FE::forms::Value<Scalar> Siso;
        Siso.kind = FE::forms::Value<Scalar>::Kind::Matrix;
        Siso.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                const auto delta = common::constant<Scalar>(i == j ? 1.0 : 0.0, ws);
                const auto spherical =
                    common::mul(trC_over_dim,
                                Cinv.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                ws);
                Siso.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    common::mul(scale, common::sub(delta, spherical, ws), ws);
            }
        }

        FE::forms::Value<Scalar> P;
        P.kind = FE::forms::Value<Scalar>::Kind::Matrix;
        P.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                auto pij = common::constant<Scalar>(0.0, ws);
                for (int k = 0; k < dim; ++k) {
                    const auto Fik = input.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(k));
                    const auto Skj = Siso.matrixAt(static_cast<std::size_t>(k), static_cast<std::size_t>(j));
                    pij = common::add(pij, common::mul(Fik, Skj, ws), ws);
                }
                P.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = pij;
            }
        }

        return P;
    }

private:
    FE::Real mu_{0.0};
};

} // namespace solid
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_SOLID_ISOCHORIC_NEO_HOOKEAN_PK1_H
