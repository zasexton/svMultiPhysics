/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "GaussQuadrature.h"
#include "Math/MathConstants.h"
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

// Compute Legendre polynomial P_n(x) and its derivative using recurrence
inline std::pair<Real, Real> legendre_with_derivative(int n, Real x) {
    Real p0 = Real(1);
    Real p1 = x;

    if (n == 0) {
        return {p0, Real(0)};
    }
    if (n == 1) {
        return {p1, p0};
    }

    for (int k = 2; k <= n; ++k) {
        Real pk = ((Real(2 * k - 1) * x * p1) - Real(k - 1) * p0) / Real(k);
        p0 = p1;
        p1 = pk;
    }

    // Derivative from recurrence relation:
    // P'_n(x) = n/(1 - x^2) [P_{n-1}(x) - x P_n(x)]
    Real derivative = Real(n) / (Real(1) - x * x) * (p0 - x * p1);
    return {p1, derivative};
}

} // namespace

std::pair<std::vector<Real>, std::vector<Real>> GaussQuadrature1D::generate_raw(int num_points) {
    if (num_points < 1) {
        throw FEException("GaussQuadrature1D: num_points must be positive",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (num_points > 128) {
        throw FEException("GaussQuadrature1D: num_points exceeds safe limit (128)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const int n = num_points;
    const int m = (n + 1) / 2;
    const Real tolerance = Real(1e-14);
    std::vector<Real> nodes(static_cast<std::size_t>(n));
    std::vector<Real> weights(static_cast<std::size_t>(n));

    for (int i = 0; i < m; ++i) {
        // Initial guess via asymptotic approximation
        const Real PI = math::constants::PI;
        Real z = std::cos(PI * (Real(i) + Real(0.75)) / (Real(n) + Real(0.5)));
        Real z_prev = std::numeric_limits<Real>::max();

        // Newton iteration to refine root
        while (std::abs(z - z_prev) > tolerance) {
            z_prev = z;
            auto [P, dP] = legendre_with_derivative(n, z);
            z = z_prev - P / dP;
        }

        // Compute weight
        auto [P, dP] = legendre_with_derivative(n, z);
        Real w = Real(2) / ((Real(1) - z * z) * dP * dP);

        nodes[static_cast<std::size_t>(i)] = -z;
        nodes[static_cast<std::size_t>(n - 1 - i)] = z;
        weights[static_cast<std::size_t>(i)] = w;
        weights[static_cast<std::size_t>(n - 1 - i)] = w;
    }

    return {nodes, weights};
}

GaussQuadrature1D::GaussQuadrature1D(int num_points)
    : QuadratureRule(svmp::CellFamily::Line, 1, 2 * num_points - 1) {
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
