#ifndef SVMP_PHYSICS_MATERIALS_SOLID_NEO_HOOKEAN_PK1_H
#define SVMP_PHYSICS_MATERIALS_SOLID_NEO_HOOKEAN_PK1_H

/**
 * @file NeoHookeanPK1.h
 * @brief Compressible Neo-Hookean first Piola–Kirchhoff stress (PK1)
 *
 * Input: deformation gradient F (matrix)
 * Output: first Piola stress P (matrix)
 *
 *   J = det(F)
 *   F^{-T} = (inv(F))^T
 *   P = μ (F - F^{-T}) + λ ln(J) F^{-T}
 *
 * This model is intended for use in weak forms such as:
 *   ∫ P(F(u)) : grad(v) dx
 */

#include "Physics/Materials/Common/TensorOps.h"

#include "FE/Constitutive/ModelCRTP.h"

#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace solid {

class NeoHookeanPK1 final : public FE::constitutive::ModelCRTP<NeoHookeanPK1> {
public:
    using ValueKind = FE::forms::ConstitutiveModel::ValueKind;

    static constexpr ValueKind kExpectedInputKind = ValueKind::Matrix;

    NeoHookeanPK1(FE::Real lambda, FE::Real mu);

    template <class Scalar, class Workspace>
    [[nodiscard]] FE::forms::Value<Scalar> evaluateImpl(const FE::forms::Value<Scalar>& input,
                                                        int dim,
                                                        Workspace& ws) const
    {
        if (!common::isMatrixLike<Scalar>(input.kind)) {
            throw std::invalid_argument("NeoHookeanPK1: expected matrix-like input");
        }
        if (dim != 2 && dim != 3) {
            throw std::invalid_argument("NeoHookeanPK1: only dim=2 or dim=3 supported");
        }

        const auto J = common::det(input, dim, ws);
        if (common::scalarValue(J) <= 0.0) {
            throw std::invalid_argument("NeoHookeanPK1: det(F) must be > 0");
        }

        const auto Finv = common::inverse(input, dim, ws);
        const auto logJ = common::log(J, ws);

        const auto lambda = common::constant<Scalar>(lambda_, ws);
        const auto mu = common::constant<Scalar>(mu_, ws);

        FE::forms::Value<Scalar> P;
        P.kind = FE::forms::Value<Scalar>::Kind::Matrix;
        P.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                const auto Fij = input.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
                const auto FinvTij = Finv.matrixAt(static_cast<std::size_t>(j), static_cast<std::size_t>(i));

                const auto diff = common::sub(Fij, FinvTij, ws);
                const auto term1 = common::mul(mu, diff, ws);

                const auto lam_logJ = common::mul(lambda, logJ, ws);
                const auto term2 = common::mul(lam_logJ, FinvTij, ws);

                P.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    common::add(term1, term2, ws);
            }
        }

        return P;
    }

private:
    FE::Real lambda_{0.0};
    FE::Real mu_{0.0};
};

} // namespace solid
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_SOLID_NEO_HOOKEAN_PK1_H

