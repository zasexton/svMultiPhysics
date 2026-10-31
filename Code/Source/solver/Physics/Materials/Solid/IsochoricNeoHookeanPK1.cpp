/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Materials/Solid/IsochoricNeoHookeanPK1.h"

#include "FE/Forms/Vocabulary.h"

#include <stdexcept>

namespace svmp {
namespace Physics {
namespace materials {
namespace solid {

FE::forms::Law isochoricNeoHookeanPK1(FE::Real mu)
{
    if (mu <= 0.0) {
        throw std::invalid_argument("IsochoricNeoHookeanPK1: mu must be > 0");
    }
    using Kind = FE::forms::ConstitutiveModel::ValueKind;
    return FE::forms::law(
        "Physics::materials::solid::IsochoricNeoHookeanPK1", Kind::Matrix, Kind::Matrix,
        [mu](const FE::forms::FormExpr& F, int dim) {
            if (dim != 2 && dim != 3) {
                throw std::invalid_argument("IsochoricNeoHookeanPK1::inlineExpand: only dim=2 or dim=3 supported");
            }
            const auto constant = [](FE::Real value) {
                return FE::forms::FormExpr::constant(value);
            };
            const auto C = FE::forms::transpose(F) * F;
            const auto Cinv = FE::forms::inv(C);
            const auto J = FE::forms::det(F);
            const auto Jm2d = FE::forms::pow(J, constant(-2.0 / static_cast<FE::Real>(dim)));
            const auto I = FE::forms::FormExpr::identity(dim);
            const auto Siso =
                constant(mu) * Jm2d *
                (I - (FE::forms::trace(C) / constant(static_cast<FE::Real>(dim))) * Cinv);
            return F * Siso;
        });
}

} // namespace solid
} // namespace materials
} // namespace Physics
} // namespace svmp
