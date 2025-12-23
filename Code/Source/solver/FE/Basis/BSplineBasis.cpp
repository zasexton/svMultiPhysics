/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Basis/BSplineBasis.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

} // namespace

BSplineBasis::BSplineBasis(int degree, std::vector<Real> knots)
    : degree_(degree),
      knots_(std::move(knots)) {
    FE_CHECK_ARG(degree_ >= 0, "BSplineBasis requires non-negative degree");
    FE_CHECK_ARG(knots_.size() >= static_cast<std::size_t>(degree_) + 2,
                 "BSplineBasis: knot vector too short for degree");
    FE_CHECK_ARG(std::is_sorted(knots_.begin(), knots_.end()),
                 "BSplineBasis: knot vector must be non-decreasing");

    num_basis_ = knots_.size() - static_cast<std::size_t>(degree_) - 1;
    FE_CHECK_ARG(num_basis_ > 0, "BSplineBasis: invalid number of basis functions");

    u_min_ = knots_[static_cast<std::size_t>(degree_)];
    u_max_ = knots_[num_basis_];
    FE_CHECK_ARG(u_max_ > u_min_, "BSplineBasis: invalid parametric domain from knots");
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
}

} // namespace basis
} // namespace FE
} // namespace svmp
