/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Basis/BSplineBasis.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace svmp {
namespace FE {
namespace basis {

namespace {

int find_span_index(const std::vector<Real>& knots,
                    int degree,
                    std::size_t num_basis,
                    Real u) {
    const int n = static_cast<int>(num_basis) - 1;
    const std::size_t p = static_cast<std::size_t>(degree);

    const Real u_min = knots[p];
    const Real u_max = knots[static_cast<std::size_t>(n) + 1];

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(32);
    if (u >= u_max - eps) {
        return n;
    }
    if (u <= u_min + eps) {
        return degree;
    }

    int low = degree;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < knots[static_cast<std::size_t>(mid)] ||
           u >= knots[static_cast<std::size_t>(mid) + 1]) {
        if (u < knots[static_cast<std::size_t>(mid)]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

void basis_funs(const std::vector<Real>& knots,
                int degree,
                int span,
                Real u,
                std::vector<Real>& N) {
    N.assign(static_cast<std::size_t>(degree) + 1, Real(0));
    N[0] = Real(1);

    std::vector<Real> left(static_cast<std::size_t>(degree) + 1, Real(0));
    std::vector<Real> right(static_cast<std::size_t>(degree) + 1, Real(0));

    for (int j = 1; j <= degree; ++j) {
        left[static_cast<std::size_t>(j)] =
            u - knots[static_cast<std::size_t>(span + 1 - j)];
        right[static_cast<std::size_t>(j)] =
            knots[static_cast<std::size_t>(span + j)] - u;

        Real saved = Real(0);
        for (int r = 0; r < j; ++r) {
            const Real denom = right[static_cast<std::size_t>(r) + 1] +
                               left[static_cast<std::size_t>(j - r)];
            const Real temp = (std::abs(denom) > Real(0)) ? (N[static_cast<std::size_t>(r)] / denom) : Real(0);
            N[static_cast<std::size_t>(r)] =
                saved + right[static_cast<std::size_t>(r) + 1] * temp;
            saved = left[static_cast<std::size_t>(j - r)] * temp;
        }
        N[static_cast<std::size_t>(j)] = saved;
    }
}

void basis_funs_and_derivatives(const std::vector<Real>& knots,
                                int degree,
                                int span,
                                Real u,
                                int max_derivative,
                                std::vector<std::vector<Real>>& ders) {
    const int p = degree;
    const int ndu_size = p + 1;
    std::vector<std::vector<Real>> ndu(static_cast<std::size_t>(ndu_size),
                                       std::vector<Real>(static_cast<std::size_t>(ndu_size), Real(0)));
    std::vector<Real> left(static_cast<std::size_t>(ndu_size), Real(0));
    std::vector<Real> right(static_cast<std::size_t>(ndu_size), Real(0));

    ndu[0][0] = Real(1);
    for (int j = 1; j <= p; ++j) {
        left[static_cast<std::size_t>(j)] =
            u - knots[static_cast<std::size_t>(span + 1 - j)];
        right[static_cast<std::size_t>(j)] =
            knots[static_cast<std::size_t>(span + j)] - u;

        Real saved = Real(0);
        for (int r = 0; r < j; ++r) {
            ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)] =
                right[static_cast<std::size_t>(r) + 1] + left[static_cast<std::size_t>(j - r)];
            const Real temp =
                (std::abs(ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)]) > Real(0))
                    ? ndu[static_cast<std::size_t>(r)][static_cast<std::size_t>(j - 1)] /
                          ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)]
                    : Real(0);
            ndu[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] =
                saved + right[static_cast<std::size_t>(r) + 1] * temp;
            saved = left[static_cast<std::size_t>(j - r)] * temp;
        }
        ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = saved;
    }

    ders.assign(static_cast<std::size_t>(max_derivative + 1),
                std::vector<Real>(static_cast<std::size_t>(ndu_size), Real(0)));
    for (int j = 0; j <= p; ++j) {
        ders[0][static_cast<std::size_t>(j)] = ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(p)];
    }

    std::vector<std::vector<Real>> a(2u, std::vector<Real>(static_cast<std::size_t>(ndu_size), Real(0)));
    for (int r = 0; r <= p; ++r) {
        int s1 = 0;
        int s2 = 1;
        a[0][0] = Real(1);

        for (int k = 1; k <= max_derivative; ++k) {
            Real d = Real(0);
            const int rk = r - k;
            const int pk = p - k;

            if (r >= k) {
                a[static_cast<std::size_t>(s2)][0] =
                    a[static_cast<std::size_t>(s1)][0] /
                    ndu[static_cast<std::size_t>(pk + 1)][static_cast<std::size_t>(rk)];
                d = a[static_cast<std::size_t>(s2)][0] *
                    ndu[static_cast<std::size_t>(rk)][static_cast<std::size_t>(pk)];
            }

            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : p - r;

            for (int j = j1; j <= j2; ++j) {
                a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(j)] =
                    (a[static_cast<std::size_t>(s1)][static_cast<std::size_t>(j)] -
                     a[static_cast<std::size_t>(s1)][static_cast<std::size_t>(j - 1)]) /
                    ndu[static_cast<std::size_t>(pk + 1)][static_cast<std::size_t>(rk + j)];
                d += a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(j)] *
                     ndu[static_cast<std::size_t>(rk + j)][static_cast<std::size_t>(pk)];
            }

            if (r <= pk) {
                a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(k)] =
                    -a[static_cast<std::size_t>(s1)][static_cast<std::size_t>(k - 1)] /
                    ndu[static_cast<std::size_t>(pk + 1)][static_cast<std::size_t>(r)];
                d += a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(k)] *
                     ndu[static_cast<std::size_t>(r)][static_cast<std::size_t>(pk)];
            }

            ders[static_cast<std::size_t>(k)][static_cast<std::size_t>(r)] = d;
            std::swap(s1, s2);
        }
    }

    int scale = p;
    for (int k = 1; k <= max_derivative; ++k) {
        for (int j = 0; j <= p; ++j) {
            ders[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] *= Real(scale);
        }
        scale *= (p - k);
    }
}

} // namespace

BSplineBasis::BSplineBasis(int degree, std::vector<Real> knots)
    : degree_(degree),
      knots_(std::move(knots)) {
    if (degree_ < 0) {
        throw BasisConfigurationException("BSplineBasis requires non-negative degree",
                                          __FILE__, __LINE__, __func__);
    }
    if (knots_.size() < static_cast<std::size_t>(degree_) + 2) {
        throw BasisConfigurationException("BSplineBasis: knot vector too short for degree",
                                          __FILE__, __LINE__, __func__);
    }
    if (!std::is_sorted(knots_.begin(), knots_.end())) {
        throw BasisConfigurationException("BSplineBasis: knot vector must be non-decreasing",
                                          __FILE__, __LINE__, __func__);
    }

    num_basis_ = knots_.size() - static_cast<std::size_t>(degree_) - 1;
    if (num_basis_ == 0u) {
        throw BasisConfigurationException("BSplineBasis: invalid number of basis functions",
                                          __FILE__, __LINE__, __func__);
    }

    u_min_ = knots_[static_cast<std::size_t>(degree_)];
    u_max_ = knots_[num_basis_];
    if (!(u_max_ > u_min_)) {
        throw BasisConfigurationException("BSplineBasis: invalid parametric domain from knots",
                                          __FILE__, __LINE__, __func__);
    }
}

BSplineBasis::BSplineBasis(int degree, std::vector<Real> knots, std::vector<Real> weights)
    : BSplineBasis(degree, std::move(knots)) {
    if (weights.size() != num_basis_) {
        throw BasisConfigurationException("BSplineBasis: weights size must equal number of basis functions",
                                          __FILE__, __LINE__, __func__);
    }
    semantic_type_ = BasisType::NURBS;
    weights_ = std::move(weights);
}

std::string BSplineBasis::cache_identity() const {
    std::ostringstream oss;
    oss << BasisFunction::cache_identity()
        << "|degree=" << degree_
        << "|knots=" << knots_.size()
        << "|weights=" << weights_.size();

    oss << std::setprecision(std::numeric_limits<Real>::max_digits10);
    for (Real knot : knots_) {
        oss << "|k=" << knot;
    }
    for (Real weight : weights_) {
        oss << "|w=" << weight;
    }
    return oss.str();
}

void BSplineBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                  std::vector<Real>& values) const {
    values.assign(num_basis_, Real(0));
    if (num_basis_ == 0) {
        return;
    }

    const Real xi0 = std::clamp(xi[0], Real(-1), Real(1));
    const Real u = u_min_ + (xi0 + Real(1)) * Real(0.5) * (u_max_ - u_min_);

    const int span = find_span_index(knots_, degree_, num_basis_, u);

    std::vector<Real> N;
    basis_funs(knots_, degree_, span, u, N);

    const int p = degree_;
    const int first = span - p;
    for (int r = 0; r <= p; ++r) {
        const int i = first + r;
        if (i >= 0 && i < static_cast<int>(num_basis_)) {
            values[static_cast<std::size_t>(i)] = N[static_cast<std::size_t>(r)];
        }
    }

    // Apply NURBS rational weighting: R_i = N_i * w_i / W, where W = sum(N_j * w_j)
    if (!weights_.empty()) {
        Real W = Real(0);
        for (std::size_t i = 0; i < num_basis_; ++i) {
            values[i] *= weights_[i];
            W += values[i];
        }
        if (std::abs(W) > Real(0)) {
            const Real inv_W = Real(1) / W;
            for (std::size_t i = 0; i < num_basis_; ++i) {
                values[i] *= inv_W;
            }
        }
    }
}

void BSplineBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                      std::vector<Gradient>& gradients) const {
    gradients.assign(num_basis_, Gradient{});
    if (num_basis_ == 0) {
        return;
    }
    if (degree_ == 0) {
        return;
    }

    const Real xi0 = std::clamp(xi[0], Real(-1), Real(1));
    const Real du_dxi = (u_max_ - u_min_) * Real(0.5);
    const Real u = u_min_ + (xi0 + Real(1)) * Real(0.5) * (u_max_ - u_min_);

    const int span = find_span_index(knots_, degree_, num_basis_, u);

    std::vector<Real> Npm1;
    basis_funs(knots_, degree_ - 1, span, u, Npm1);

    const int p = degree_;
    const int first = span - p;

    // d/du of the non-zero B-splines via recurrence:
    // dN_{i,p} = p/(U_{i+p}-U_i)*N_{i,p-1} - p/(U_{i+p+1}-U_{i+1})*N_{i+1,p-1}
    for (int r = 0; r <= p; ++r) {
        const int i = first + r;
        if (i < 0 || i >= static_cast<int>(num_basis_)) {
            continue;
        }

        const Real n_i = (r - 1 >= 0) ? Npm1[static_cast<std::size_t>(r - 1)] : Real(0);
        const Real n_ip1 = (r <= p - 1) ? Npm1[static_cast<std::size_t>(r)] : Real(0);

        const Real den1 = knots_[static_cast<std::size_t>(i + p)] - knots_[static_cast<std::size_t>(i)];
        const Real den2 = knots_[static_cast<std::size_t>(i + p + 1)] - knots_[static_cast<std::size_t>(i + 1)];

        const Real term1 = (std::abs(den1) > Real(0)) ? (Real(p) * n_i / den1) : Real(0);
        const Real term2 = (std::abs(den2) > Real(0)) ? (Real(p) * n_ip1 / den2) : Real(0);

        gradients[static_cast<std::size_t>(i)][0] = (term1 - term2) * du_dxi;
    }

    // NURBS quotient rule: dR_i/dxi = (dN_i*w_i*W - N_i*w_i*dW) / W^2
    // where W = sum(N_j*w_j), dW = sum(dN_j*w_j)
    if (!weights_.empty()) {
        // Evaluate B-spline values for the weight function
        std::vector<Real> N_vals(num_basis_, Real(0));
        {
            std::vector<Real> Np;
            basis_funs(knots_, degree_, span, u, Np);
            const int first_v = span - p;
            for (int r = 0; r <= p; ++r) {
                const int i = first_v + r;
                if (i >= 0 && i < static_cast<int>(num_basis_)) {
                    N_vals[static_cast<std::size_t>(i)] = Np[static_cast<std::size_t>(r)];
                }
            }
        }

        Real W = Real(0);
        Real dW = Real(0);
        for (std::size_t i = 0; i < num_basis_; ++i) {
            W += N_vals[i] * weights_[i];
            dW += gradients[i][0] * weights_[i];
        }

        if (std::abs(W) > Real(0)) {
            const Real inv_W2 = Real(1) / (W * W);
            for (std::size_t i = 0; i < num_basis_; ++i) {
                const Real dNw = gradients[i][0] * weights_[i];
                const Real Nw = N_vals[i] * weights_[i];
                gradients[i][0] = (dNw * W - Nw * dW) * inv_W2;
            }
        }
    }
}

void BSplineBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                     std::vector<Hessian>& hessians) const {
    hessians.assign(num_basis_, Hessian{});
    if (num_basis_ == 0 || degree_ == 0) {
        return;
    }

    const Real xi0 = std::clamp(xi[0], Real(-1), Real(1));
    const Real du_dxi = (u_max_ - u_min_) * Real(0.5);
    const Real du_dxi_sq = du_dxi * du_dxi;
    const Real u = u_min_ + (xi0 + Real(1)) * Real(0.5) * (u_max_ - u_min_);

    const int span = find_span_index(knots_, degree_, num_basis_, u);
    std::vector<std::vector<Real>> ders;
    basis_funs_and_derivatives(knots_, degree_, span, u, 2, ders);

    std::vector<Real> values(num_basis_, Real(0));
    std::vector<Real> first(num_basis_, Real(0));
    std::vector<Real> second(num_basis_, Real(0));
    const int p = degree_;
    const int first_index = span - p;
    for (int r = 0; r <= p; ++r) {
        const int i = first_index + r;
        if (i < 0 || i >= static_cast<int>(num_basis_)) {
            continue;
        }
        const std::size_t si = static_cast<std::size_t>(i);
        values[si] = ders[0][static_cast<std::size_t>(r)];
        first[si] = ders[1][static_cast<std::size_t>(r)] * du_dxi;
        second[si] = ders[2][static_cast<std::size_t>(r)] * du_dxi_sq;
    }

    if (weights_.empty()) {
        for (std::size_t i = 0; i < num_basis_; ++i) {
            hessians[i](0, 0) = second[i];
        }
        return;
    }

    Real W = Real(0);
    Real dW = Real(0);
    Real ddW = Real(0);
    for (std::size_t i = 0; i < num_basis_; ++i) {
        const Real wi = weights_[i];
        W += values[i] * wi;
        dW += first[i] * wi;
        ddW += second[i] * wi;
    }

    if (std::abs(W) <= Real(0)) {
        return;
    }

    const Real inv_W3 = Real(1) / (W * W * W);
    for (std::size_t i = 0; i < num_basis_; ++i) {
        const Real wi = weights_[i];
        const Real A = values[i] * wi;
        const Real dA = first[i] * wi;
        const Real ddA = second[i] * wi;
        hessians[i](0, 0) =
            (ddA * W * W - A * ddW * W - Real(2) * dA * W * dW + Real(2) * A * dW * dW) *
            inv_W3;
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
