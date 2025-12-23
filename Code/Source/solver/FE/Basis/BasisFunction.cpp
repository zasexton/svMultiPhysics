/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BasisFunction.h"
#include <algorithm>

namespace svmp {
namespace FE {
namespace basis {

void BasisFunction::numerical_gradient(const math::Vector<Real, 3>& xi,
                                       std::vector<Gradient>& gradients,
                                       Real eps) const {
    std::vector<Real> base;
    evaluate_values(xi, base);
    gradients.assign(base.size(), Gradient{});

    for (int d = 0; d < dimension(); ++d) {
        math::Vector<Real, 3> forward = xi;
        math::Vector<Real, 3> backward = xi;
        const std::size_t idx = static_cast<std::size_t>(d);
        forward[idx] += eps;
        backward[idx] -= eps;

        std::vector<Real> fwd, bwd;
        evaluate_values(forward, fwd);
        evaluate_values(backward, bwd);

        for (std::size_t i = 0; i < base.size(); ++i) {
            gradients[i][idx] = (fwd[i] - bwd[i]) / (Real(2) * eps);
        }
    }
}

void BasisFunction::numerical_hessian(const math::Vector<Real, 3>& xi,
                                      std::vector<Hessian>& hessians,
                                      Real eps) const {
    std::vector<Gradient> base_grad;
    evaluate_gradients(xi, base_grad);
    hessians.assign(base_grad.size(), Hessian{});

    for (int d = 0; d < dimension(); ++d) {
        math::Vector<Real, 3> forward = xi;
        math::Vector<Real, 3> backward = xi;
        const std::size_t col = static_cast<std::size_t>(d);
        forward[col] += eps;
        backward[col] -= eps;

        std::vector<Gradient> g_forward, g_backward;
        evaluate_gradients(forward, g_forward);
        evaluate_gradients(backward, g_backward);

        for (std::size_t i = 0; i < base_grad.size(); ++i) {
            for (int k = 0; k < dimension(); ++k) {
                const std::size_t row = static_cast<std::size_t>(k);
                hessians[i](row, col) = (g_forward[i][row] - g_backward[i][row]) / (Real(2) * eps);
            }
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
