#ifndef SVMP_PHYSICS_MATERIALS_SOLID_LINEAR_ELASTIC_STRESS_H
#define SVMP_PHYSICS_MATERIALS_SOLID_LINEAR_ELASTIC_STRESS_H

/**
 * @file LinearElasticStress.h
 * @brief Small-strain linear elastic stress model (isotropic)
 *
 * Input: symmetric strain tensor ε (matrix-like Value)
 * Output: Cauchy stress σ (symmetric matrix)
 *
 *   σ = λ tr(ε) I + 2 μ ε
 */

#include "Physics/Materials/Common/TensorOps.h"

#include "FE/Constitutive/ModelCRTP.h"

#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace solid {

class LinearElasticStress final : public FE::constitutive::ModelCRTP<LinearElasticStress> {
public:
    using ValueKind = FE::forms::ConstitutiveModel::ValueKind;

    static constexpr ValueKind kExpectedInputKind = ValueKind::SymmetricMatrix;

    LinearElasticStress(FE::Real lambda, FE::Real mu);

    template <class Scalar, class Workspace>
    [[nodiscard]] FE::forms::Value<Scalar> evaluateImpl(const FE::forms::Value<Scalar>& input,
                                                        int dim,
                                                        Workspace& ws) const
    {
        if (!common::isMatrixLike<Scalar>(input.kind)) {
            throw std::invalid_argument("LinearElasticStress: expected matrix-like input");
        }
        if (dim != 2 && dim != 3) {
            throw std::invalid_argument("LinearElasticStress: only dim=2 or dim=3 supported");
        }

        const auto lambda = common::constant<Scalar>(lambda_, ws);
        const auto mu = common::constant<Scalar>(mu_, ws);
        const auto two_mu = common::mul(mu, common::constant<Scalar>(2.0, ws), ws);

        Scalar tr = common::constant<Scalar>(0.0, ws);
        for (int d = 0; d < dim; ++d) {
            tr = common::add(tr, input.matrixAt(static_cast<std::size_t>(d), static_cast<std::size_t>(d)), ws);
        }

        FE::forms::Value<Scalar> out;
        out.kind = FE::forms::Value<Scalar>::Kind::SymmetricMatrix;
        out.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                auto val = common::mul(two_mu,
                                       input.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                       ws);
                if (i == j) {
                    val = common::add(val, common::mul(lambda, tr, ws), ws);
                }
                out.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = val;
            }
        }

        return out;
    }

private:
    FE::Real lambda_{0.0};
    FE::Real mu_{0.0};
};

} // namespace solid
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_SOLID_LINEAR_ELASTIC_STRESS_H

