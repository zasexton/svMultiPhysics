/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/CollocationMethods.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace svmp {
namespace FE {
namespace timestepping {
namespace collocation {

namespace {

std::pair<double, double> legendreWithDerivative(int n, double x)
{
    double p0 = 1.0;
    double p1 = x;

    if (n == 0) {
        return {p0, 0.0};
    }
    if (n == 1) {
        return {p1, p0};
    }

    for (int k = 2; k <= n; ++k) {
        const double pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / static_cast<double>(k);
        p0 = p1;
        p1 = pk;
    }

    const double denom = 1.0 - x * x;
    FE_THROW_IF(denom == 0.0, InvalidArgumentException,
                "TimeLoop: Legendre derivative singularity at |x|=1");
    const double dp = static_cast<double>(n) / denom * (p0 - x * p1);
    return {p1, dp};
}

std::vector<double> gaussNodesMinusOneOne(int n)
{
    FE_THROW_IF(n <= 0, InvalidArgumentException, "TimeLoop: Gauss nodes require n > 0");
    FE_THROW_IF(n > 64, InvalidArgumentException, "TimeLoop: Gauss nodes n too large (max 64)");

    const int m = (n + 1) / 2;
    std::vector<double> nodes(static_cast<std::size_t>(n), 0.0);

    const double pi = std::acos(-1.0);
    constexpr double tol = 1e-14;
    for (int i = 0; i < m; ++i) {
        double z = std::cos(pi * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));
        double z_prev = std::numeric_limits<double>::max();
        for (int it = 0; it < 64; ++it) {
            if (std::abs(z - z_prev) <= tol) {
                break;
            }
            z_prev = z;
            const auto [P, dP] = legendreWithDerivative(n, z);
            FE_THROW_IF(dP == 0.0, InvalidArgumentException, "TimeLoop: Gauss node Newton derivative is zero");
            z = z_prev - P / dP;
        }

        nodes[static_cast<std::size_t>(i)] = -z;
        nodes[static_cast<std::size_t>(n - 1 - i)] = z;
    }

    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

} // namespace

std::vector<double> gaussNodesUnit(int stages)
{
    const auto x = gaussNodesMinusOneOne(stages);
    std::vector<double> c;
    c.reserve(x.size());
    for (double xi : x) {
        c.push_back(0.5 * (xi + 1.0));
    }
    return c;
}

std::vector<double> radauIIANodesUnit(int stages)
{
    FE_THROW_IF(stages <= 0, InvalidArgumentException, "TimeLoop: Radau IIA requires stages > 0");
    if (stages == 1) {
        return {1.0};
    }

    const auto guesses = gaussNodesMinusOneOne(stages - 1);
    std::vector<double> roots;
    roots.reserve(static_cast<std::size_t>(stages));

    constexpr double tol = 1e-14;
    for (double z0 : guesses) {
        double z = z0;
        double z_prev = std::numeric_limits<double>::max();
        for (int it = 0; it < 64; ++it) {
            if (std::abs(z - z_prev) <= tol) {
                break;
            }
            z_prev = z;
            const auto [Ps, dPs] = legendreWithDerivative(stages, z);
            const auto [Ps1, dPs1] = legendreWithDerivative(stages - 1, z);
            const double f = Ps - Ps1;
            const double df = dPs - dPs1;
            FE_THROW_IF(df == 0.0, InvalidArgumentException, "TimeLoop: Radau node Newton derivative is zero");
            z = z_prev - f / df;
        }
        roots.push_back(z);
    }
    std::sort(roots.begin(), roots.end());
    roots.push_back(1.0);

    std::vector<double> c;
    c.reserve(roots.size());
    for (double xi : roots) {
        c.push_back(0.5 * (xi + 1.0));
    }
    return c;
}

std::vector<double> invertDenseMatrix(const std::vector<double>& A, int n)
{
    FE_THROW_IF(n <= 0, InvalidArgumentException, "TimeLoop: invalid dense matrix size");
    FE_THROW_IF(static_cast<int>(A.size()) != n * n, InvalidArgumentException,
                "TimeLoop: dense matrix size mismatch");

    std::vector<double> M = A;
    std::vector<double> inv(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        inv[static_cast<std::size_t>(i * n + i)] = 1.0;
    }

    auto rowSwap = [&](std::vector<double>& mat, int r1, int r2) {
        for (int j = 0; j < n; ++j) {
            std::swap(mat[static_cast<std::size_t>(r1 * n + j)],
                      mat[static_cast<std::size_t>(r2 * n + j)]);
        }
    };

    for (int k = 0; k < n; ++k) {
        int piv = k;
        double piv_abs = std::abs(M[static_cast<std::size_t>(k * n + k)]);
        for (int r = k + 1; r < n; ++r) {
            const double a = std::abs(M[static_cast<std::size_t>(r * n + k)]);
            if (a > piv_abs) {
                piv_abs = a;
                piv = r;
            }
        }
        FE_THROW_IF(piv_abs == 0.0, InvalidArgumentException, "TimeLoop: singular dense matrix");
        if (piv != k) {
            rowSwap(M, piv, k);
            rowSwap(inv, piv, k);
        }

        const double diag = M[static_cast<std::size_t>(k * n + k)];
        const double inv_diag = 1.0 / diag;
        for (int j = 0; j < n; ++j) {
            M[static_cast<std::size_t>(k * n + j)] *= inv_diag;
            inv[static_cast<std::size_t>(k * n + j)] *= inv_diag;
        }

        for (int r = 0; r < n; ++r) {
            if (r == k) continue;
            const double fac = M[static_cast<std::size_t>(r * n + k)];
            if (fac == 0.0) continue;
            for (int j = 0; j < n; ++j) {
                M[static_cast<std::size_t>(r * n + j)] -= fac * M[static_cast<std::size_t>(k * n + j)];
                inv[static_cast<std::size_t>(r * n + j)] -= fac * inv[static_cast<std::size_t>(k * n + j)];
            }
        }
    }

    return inv;
}

CollocationMethod buildCollocationMethod(CollocationFamily family, int stages)
{
    FE_THROW_IF(stages <= 0, InvalidArgumentException, "TimeLoop: invalid collocation stage count");

    CollocationMethod method;
    method.stages = stages;
    method.order = (family == CollocationFamily::Gauss) ? 2 * stages : 2 * stages - 1;
    method.stiffly_accurate = (family == CollocationFamily::RadauIIA);
    method.final_stage = (family == CollocationFamily::RadauIIA) ? (stages - 1) : 0;

    method.c = (family == CollocationFamily::Gauss)
        ? gaussNodesUnit(stages)
        : radauIIANodesUnit(stages);

    auto lagrangeCoeff = [&](int j) -> std::vector<double> {
        std::vector<double> coeff(static_cast<std::size_t>(stages), 0.0);
        coeff[0] = 1.0;
        int deg = 0;
        const double cj = method.c[static_cast<std::size_t>(j)];

        for (int m = 0; m < stages; ++m) {
            if (m == j) continue;
            const double cm = method.c[static_cast<std::size_t>(m)];
            const double denom = cj - cm;
            FE_THROW_IF(denom == 0.0, InvalidArgumentException, "TimeLoop: duplicate collocation nodes");

            std::vector<double> next(static_cast<std::size_t>(stages), 0.0);
            for (int k = 0; k <= deg; ++k) {
                next[static_cast<std::size_t>(k)] += (-cm / denom) * coeff[static_cast<std::size_t>(k)];
                next[static_cast<std::size_t>(k + 1)] += (1.0 / denom) * coeff[static_cast<std::size_t>(k)];
            }
            coeff = std::move(next);
            ++deg;
        }
        return coeff;
    };

    std::vector<double> A(static_cast<std::size_t>(stages) * static_cast<std::size_t>(stages), 0.0);
    std::vector<double> b(static_cast<std::size_t>(stages), 0.0);

    for (int j = 0; j < stages; ++j) {
        const auto coeff = lagrangeCoeff(j);

        auto integratePoly = [&](double x) -> double {
            double sum = 0.0;
            double xpow = x;
            for (int k = 0; k < stages; ++k) {
                sum += coeff[static_cast<std::size_t>(k)] * xpow / static_cast<double>(k + 1);
                xpow *= x;
            }
            return sum;
        };

        b[static_cast<std::size_t>(j)] = integratePoly(1.0);
        for (int i = 0; i < stages; ++i) {
            const double ci = method.c[static_cast<std::size_t>(i)];
            A[static_cast<std::size_t>(i * stages + j)] = integratePoly(ci);
        }
    }

    method.ainv = invertDenseMatrix(A, stages);
    method.row_sums.resize(static_cast<std::size_t>(stages), 0.0);
    for (int i = 0; i < stages; ++i) {
        double sum = 0.0;
        for (int j = 0; j < stages; ++j) {
            sum += method.ainv[static_cast<std::size_t>(i * stages + j)];
        }
        method.row_sums[static_cast<std::size_t>(i)] = sum;
    }

    if (!method.stiffly_accurate) {
        method.final_w.resize(static_cast<std::size_t>(stages), 0.0);
        for (int j = 0; j < stages; ++j) {
            double sum = 0.0;
            for (int i = 0; i < stages; ++i) {
                sum += b[static_cast<std::size_t>(i)] * method.ainv[static_cast<std::size_t>(i * stages + j)];
            }
            method.final_w[static_cast<std::size_t>(j)] = sum;
        }
    }

    return method;
}

SecondOrderCollocationData buildSecondOrderCollocationData(const CollocationMethod& method)
{
    SecondOrderCollocationData data;
    data.stages = method.stages;
    data.n_constraints = method.stages + 2;
    const int s = method.stages;
    const int n = data.n_constraints;

    std::vector<double> V(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    V[0] = 1.0;                                  // u(0)
    V[static_cast<std::size_t>(n + 1)] = 1.0;    // u'(0)

    for (int j = 0; j < s; ++j) {
        const double cj = method.c[static_cast<std::size_t>(j)];
        double pow = 1.0;
        const int row = 2 + j;
        for (int k = 0; k < n; ++k) {
            V[static_cast<std::size_t>(row * n + k)] = pow;
            pow *= cj;
        }
    }

    const auto Vinv = invertDenseMatrix(V, n);

    auto evalRow = [&](double tau, int deriv) -> std::vector<double> {
        std::vector<double> row(static_cast<std::size_t>(n), 0.0);
        if (deriv == 0) {
            double pow = 1.0;
            for (int k = 0; k < n; ++k) {
                row[static_cast<std::size_t>(k)] = pow;
                pow *= tau;
            }
            return row;
        }
        if (deriv == 1) {
            double pow = 1.0;
            for (int k = 1; k < n; ++k) {
                row[static_cast<std::size_t>(k)] = static_cast<double>(k) * pow;
                pow *= tau;
            }
            return row;
        }
        FE_THROW_IF(deriv != 2, InvalidArgumentException, "TimeLoop: invalid Hermite derivative order");
        double pow = 1.0;
        for (int k = 2; k < n; ++k) {
            row[static_cast<std::size_t>(k)] = static_cast<double>(k) * static_cast<double>(k - 1) * pow;
            pow *= tau;
        }
        return row;
    };

    data.d1.resize(static_cast<std::size_t>(s) * static_cast<std::size_t>(s), 0.0);
    data.d2.resize(static_cast<std::size_t>(s) * static_cast<std::size_t>(s), 0.0);
    data.d1_u0.resize(static_cast<std::size_t>(s), 0.0);
    data.d1_dv0.resize(static_cast<std::size_t>(s), 0.0);
    data.d2_u0.resize(static_cast<std::size_t>(s), 0.0);
    data.d2_dv0.resize(static_cast<std::size_t>(s), 0.0);

    for (int i = 0; i < s; ++i) {
        const double ci = method.c[static_cast<std::size_t>(i)];
        const auto row1 = evalRow(ci, 1);
        const auto row2 = evalRow(ci, 2);

        for (int col = 0; col < n; ++col) {
            double c1 = 0.0;
            double c2 = 0.0;
            for (int k = 0; k < n; ++k) {
                c1 += row1[static_cast<std::size_t>(k)] * Vinv[static_cast<std::size_t>(k * n + col)];
                c2 += row2[static_cast<std::size_t>(k)] * Vinv[static_cast<std::size_t>(k * n + col)];
            }

            if (col == 0) {
                data.d1_u0[static_cast<std::size_t>(i)] = c1;
                data.d2_u0[static_cast<std::size_t>(i)] = c2;
            } else if (col == 1) {
                data.d1_dv0[static_cast<std::size_t>(i)] = c1;
                data.d2_dv0[static_cast<std::size_t>(i)] = c2;
            } else {
                const int j = col - 2;
                data.d1[static_cast<std::size_t>(i * s + j)] = c1;
                data.d2[static_cast<std::size_t>(i * s + j)] = c2;
            }
        }
    }

    // Final-time coefficients (τ=1).
    const auto row0 = evalRow(1.0, 0);
    const auto row1 = evalRow(1.0, 1);
    const auto row2 = evalRow(1.0, 2);

    data.u1.resize(static_cast<std::size_t>(s), 0.0);
    data.du1.resize(static_cast<std::size_t>(s), 0.0);
    data.ddu1.resize(static_cast<std::size_t>(s), 0.0);

    for (int col = 0; col < n; ++col) {
        double c0 = 0.0;
        double c1 = 0.0;
        double c2 = 0.0;
        for (int k = 0; k < n; ++k) {
            const double vik = Vinv[static_cast<std::size_t>(k * n + col)];
            c0 += row0[static_cast<std::size_t>(k)] * vik;
            c1 += row1[static_cast<std::size_t>(k)] * vik;
            c2 += row2[static_cast<std::size_t>(k)] * vik;
        }

        if (col == 0) {
            data.u1_u0 = c0;
            data.du1_u0 = c1;
            data.ddu1_u0 = c2;
        } else if (col == 1) {
            data.u1_dv0 = c0;
            data.du1_dv0 = c1;
            data.ddu1_dv0 = c2;
        } else {
            const int j = col - 2;
            data.u1[static_cast<std::size_t>(j)] = c0;
            data.du1[static_cast<std::size_t>(j)] = c1;
            data.ddu1[static_cast<std::size_t>(j)] = c2;
        }
    }

    return data;
}

} // namespace collocation
} // namespace timestepping
} // namespace FE
} // namespace svmp

