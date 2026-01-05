/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_MATH_FINITE_DIFFERENCE_H
#define SVMP_FE_MATH_FINITE_DIFFERENCE_H

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief Fornberg finite-difference weights for arbitrary node distributions.
 *
 * Computes weights `w_j` such that:
 *   f^{(m)}(x0) ≈ sum_j w_j f(x_j)
 *
 * @param derivative_order m >= 0
 * @param x0 evaluation point
 * @param nodes node locations (size >= 1)
 */
[[nodiscard]] inline std::vector<double>
finiteDifferenceWeights(int derivative_order, double x0, std::span<const double> nodes)
{
    FE_THROW_IF(derivative_order < 0, InvalidArgumentException,
                "finiteDifferenceWeights: derivative_order must be >= 0");
    FE_THROW_IF(nodes.empty(), InvalidArgumentException,
                "finiteDifferenceWeights: node list must be non-empty");
    FE_THROW_IF(!std::isfinite(x0), InvalidArgumentException,
                "finiteDifferenceWeights: x0 must be finite");

    for (double xi : nodes) {
        FE_THROW_IF(!std::isfinite(xi), InvalidArgumentException,
                    "finiteDifferenceWeights: nodes must be finite");
    }

    const int n = static_cast<int>(nodes.size()) - 1;
    const int m = derivative_order;

    std::vector<std::vector<double>> c(nodes.size(), std::vector<double>(static_cast<std::size_t>(m + 1), 0.0));
    c[0][0] = 1.0;

    double c1 = 1.0;
    double c4 = nodes[0] - x0;

    for (int i = 1; i <= n; ++i) {
        const int mn = std::min(i, m);
        double c2 = 1.0;
        double c5 = c4;
        c4 = nodes[static_cast<std::size_t>(i)] - x0;

        for (int j = 0; j <= i - 1; ++j) {
            const double c3 =
                nodes[static_cast<std::size_t>(i)] - nodes[static_cast<std::size_t>(j)];
            FE_THROW_IF(c3 == 0.0, InvalidArgumentException,
                        "finiteDifferenceWeights: duplicate nodes are not allowed");
            c2 *= c3;

            if (j == i - 1) {
                for (int k = mn; k >= 1; --k) {
                    c[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
                        c1 * (k * c[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(k - 1)] -
                              c5 * c[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(k)]) /
                        c2;
                }
                c[static_cast<std::size_t>(i)][0] = -c1 * c5 * c[static_cast<std::size_t>(i - 1)][0] / c2;
            }

            for (int k = mn; k >= 1; --k) {
                c[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
                    (c4 * c[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] -
                     k * c[static_cast<std::size_t>(j)][static_cast<std::size_t>(k - 1)]) /
                    c3;
            }
            c[static_cast<std::size_t>(j)][0] = c4 * c[static_cast<std::size_t>(j)][0] / c3;
        }
        c1 = c2;
    }

    std::vector<double> w(nodes.size(), 0.0);
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        w[j] = c[j][static_cast<std::size_t>(m)];
    }
    return w;
}

/**
 * @brief Lagrange interpolation/extrapolation weights.
 *
 * Computes weights `w_j` such that:
 *   f(x) ≈ sum_j w_j f(x_j)
 *
 * @param x evaluation point
 * @param nodes node locations (size >= 1)
 */
[[nodiscard]] inline std::vector<double>
lagrangeWeights(double x, std::span<const double> nodes)
{
    FE_THROW_IF(nodes.empty(), InvalidArgumentException,
                "lagrangeWeights: node list must be non-empty");
    FE_THROW_IF(!std::isfinite(x), InvalidArgumentException,
                "lagrangeWeights: x must be finite");

    for (double xi : nodes) {
        FE_THROW_IF(!std::isfinite(xi), InvalidArgumentException,
                    "lagrangeWeights: nodes must be finite");
    }

    std::vector<double> w(nodes.size(), 0.0);
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        double num = 1.0;
        double den = 1.0;
        const double xj = nodes[j];
        for (std::size_t m = 0; m < nodes.size(); ++m) {
            if (m == j) {
                continue;
            }
            const double xm = nodes[m];
            const double d = xj - xm;
            FE_THROW_IF(d == 0.0, InvalidArgumentException,
                        "lagrangeWeights: duplicate nodes are not allowed");
            num *= (x - xm);
            den *= d;
        }
        w[j] = num / den;
    }
    return w;
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_FINITE_DIFFERENCE_H

