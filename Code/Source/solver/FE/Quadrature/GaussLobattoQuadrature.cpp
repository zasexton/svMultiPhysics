/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "GaussLobattoQuadrature.h"
#include "Math/MathConstants.h"

#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

std::pair<Real, Real> legendre_with_derivative(int n, Real x) {
    if (n == 0) {
        return {Real(1), Real(0)};
    }
    if (n == 1) {
        return {x, Real(1)};
    }

    Real p_nm2 = Real(1);
    Real p_nm1 = x;
    Real dp_nm2 = Real(0);
    Real dp_nm1 = Real(1);

    for (int k = 2; k <= n; ++k) {
        const Real rk = static_cast<Real>(k);
        const Real p =
            ((Real(2) * rk - Real(1)) * x * p_nm1 -
             (rk - Real(1)) * p_nm2) / rk;
        const Real dp =
            ((Real(2) * rk - Real(1)) * (p_nm1 + x * dp_nm1) -
             (rk - Real(1)) * dp_nm2) / rk;
        p_nm2 = p_nm1;
        p_nm1 = p;
        dp_nm2 = dp_nm1;
        dp_nm1 = dp;
    }

    return {p_nm1, dp_nm1};
}

} // namespace

std::pair<std::vector<Real>, std::vector<Real>> GaussLobattoQuadrature1D::generate_raw(int num_points) {
    if (num_points < 2) {
        throw FEException("GaussLobattoQuadrature1D requires at least 2 points (endpoints)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (num_points > 128) {
        throw FEException("GaussLobattoQuadrature1D: num_points exceeds safe limit (128)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const int n = num_points;
    std::vector<Real> nodes(static_cast<std::size_t>(n));
    std::vector<Real> weights(static_cast<std::size_t>(n));

    nodes.front() = Real(-1);
    nodes.back() = Real(1);

    const Real endpoint_weight = Real(2) / (Real(n) * Real(n - 1));
    weights.front() = endpoint_weight;
    weights.back() = endpoint_weight;

    if (n == 2) {
        return {nodes, weights};
    }

    const Real tolerance = Real(1e-14);
    const int interior = n - 2;
    const int half = (interior + 1) / 2;

    for (int i = 0; i < half; ++i) {
        Real x_root =
            -std::cos(math::constants::PI * Real(i + 1) / Real(n - 1));
        Real x_prev = std::numeric_limits<Real>::max();

        while (std::abs(x_root - x_prev) > tolerance) {
            x_prev = x_root;
            const auto [Pn1, dPn1] = legendre_with_derivative(n - 1, x_root);
            const auto [Pn2, dPn2] = legendre_with_derivative(n - 2, x_root);

            const Real f = x_root * Pn1 - Pn2;
            const Real fp = Pn1 + x_root * dPn1 - dPn2;
            x_root = x_prev - f / fp;
        }

        const auto [Pn1, dPn1] = legendre_with_derivative(n - 1, x_root);
        (void)dPn1;
        const Real w = endpoint_weight / (Pn1 * Pn1);

        const std::size_t left_index = static_cast<std::size_t>(i + 1);
        const std::size_t right_index = static_cast<std::size_t>(n - 2 - i);
        nodes[left_index] = x_root;
        nodes[right_index] = -x_root;
        weights[left_index] = w;
        weights[right_index] = w;
    }

    return {nodes, weights};
}

GaussLobattoQuadrature1D::GaussLobattoQuadrature1D(int num_points)
    : QuadratureRule(svmp::CellFamily::Line, 1, 2 * num_points - 3) {
    auto [nodes, weights] = generate_raw(num_points);

    std::vector<QuadPoint> pts;
    pts.reserve(nodes.size());
    std::vector<Real> wts;
    wts.reserve(weights.size());

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        pts.push_back(QuadPoint{nodes[i], Real(0), Real(0)});
        wts.push_back(weights[i]);
    }

    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
