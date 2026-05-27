/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "OrthogonalPolynomials.h"
#include "BasisTolerance.h"
#include "Math/IntegerMath.h"
#include "Math/MathConstants.h"
#include "ReferenceDerivativeJet.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace basis {
namespace orthopoly {

using math::pow_int;

namespace {

using detail::Jet3;

std::size_t legendre_sequence_count(int n, const char* name) {
    if (n < 0) {
        throw std::invalid_argument(name);
    }
    return static_cast<std::size_t>(n + 1);
}

void check_legendre_span(int n, std::span<Real> values, const char* name) {
    const auto required = legendre_sequence_count(n, name);
    if (values.size() < required) {
        throw std::invalid_argument(name);
    }
}

std::size_t jacobi_sequence_count(int n, const char* name) {
    if (n < 0) {
        throw std::invalid_argument(name);
    }
    return static_cast<std::size_t>(n + 1);
}

void check_jacobi_span(int n, std::span<Real> values, const char* name) {
    const auto required = jacobi_sequence_count(n, name);
    if (values.size() < required) {
        throw std::invalid_argument(name);
    }
}

Jet3 lift_jacobi(int n, Real alpha, Real beta, const Jet3& arg) {
    const auto poly = jacobi_with_second_derivative(n, alpha, beta, arg.value);
    return detail::compose_univariate(arg,
                                      poly.value,
                                      poly.derivative,
                                      poly.second_derivative);
}

Real collapsed_coordinate_tolerance() noexcept {
    return detail::basis_scaled_tolerance(Real(1), Real(10));
}

} // namespace

Real legendre(int n, Real x) {
    if (n == 0) {
        return Real(1);
    }
    if (n == 1) {
        return x;
    }

    Real p0 = Real(1);
    Real p1 = x;
    for (int k = 2; k <= n; ++k) {
        Real pk = ((Real(2 * k - 1) * x * p1) - Real(k - 1) * p0) / Real(k);
        p0 = p1;
        p1 = pk;
    }
    return p1;
}

UnivariateFirstDerivative legendre_derivative(int n, Real x) {
    if (n == 0) {
        return {Real(1), Real(0)};
    }
    if (n == 1) {
        return {x, Real(1)};
    }

    Real p0 = Real(1);
    Real p1 = x;
    Real dp0 = Real(0);
    Real dp1 = Real(1);
    for (int k = 2; k <= n; ++k) {
        const Real a = Real(2 * k - 1);
        const Real b = Real(k - 1);
        Real pk = (a * x * p1 - b * p0) / Real(k);
        Real dpk = (a * (p1 + x * dp1) - b * dp0) / Real(k);
        p0 = p1;
        p1 = pk;
        dp0 = dp1;
        dp1 = dpk;
    }

    return {p1, dp1};
}

Real integrated_legendre(int n, Real x) {
    if (n == 0) {
        return x + Real(1);
    }
    const Real denom = Real(2 * n + 1);
    // ∫ P_n = (P_{n+1} - P_{n-1}) / (2n+1), with P_{-1} = 0
    Real p_np1 = legendre(n + 1, x);
    Real p_nm1 = legendre(n - 1, x);
    return (p_np1 - p_nm1) / denom;
}

Real jacobi(int n, Real alpha, Real beta, Real x) {
    if (n == 0) {
        return Real(1);
    }
    if (n == 1) {
        return Real(0.5) * ((alpha - beta) + (alpha + beta + Real(2)) * x);
    }

    // Three-term recurrence (see e.g. Abramowitz & Stegun 22.7.15):
    // 2(n+1)(n+a+b+1)(2n+a+b) P_{n+1} =
    //   (2n+a+b+1) * [ (2n+a+b)(2n+a+b+2) x + (a^2-b^2) ] P_n
    //   - 2(n+a)(n+b)(2n+a+b+2) P_{n-1}.
    Real pnm1 = Real(1);
    Real pn = Real(0.5) * ((alpha - beta) + (alpha + beta + Real(2)) * x);

    for (int k = 1; k < n; ++k) {
        const Real kf = static_cast<Real>(k);
        const Real two_k_ab = Real(2) * kf + alpha + beta;

        const Real denom = Real(2) * (kf + Real(1)) * (kf + alpha + beta + Real(1)) * two_k_ab;
        const Real b = (two_k_ab + Real(1)) *
                       ((two_k_ab + Real(2)) * two_k_ab * x + (alpha * alpha - beta * beta));
        const Real c = Real(2) * (kf + alpha) * (kf + beta) * (two_k_ab + Real(2));

        const Real pnp1 = (b * pn - c * pnm1) / denom;
        pnm1 = pn;
        pn = pnp1;
    }

    return pn;
}

Real jacobi_derivative(int n, Real alpha, Real beta, Real x) {
    if (n == 0) {
        return Real(0);
    }
    // d/dx P_n^{(a,b)} = 0.5*(a+b+n+1) * P_{n-1}^{(a+1,b+1)}(x)
    const Real factor = Real(0.5) * (alpha + beta + Real(n) + Real(1));
    return factor * jacobi(n - 1, alpha + Real(1), beta + Real(1), x);
}

Real jacobi_second_derivative(int n, Real alpha, Real beta, Real x) {
    if (n <= 1) {
        return Real(0);
    }
    const Real factor = Real(0.25) *
                        (alpha + beta + Real(n) + Real(1)) *
                        (alpha + beta + Real(n) + Real(2));
    return factor * jacobi(n - 2, alpha + Real(2), beta + Real(2), x);
}

UnivariateDerivatives jacobi_with_second_derivative(int n, Real alpha, Real beta, Real x) {
    if (n < 0) {
        throw std::invalid_argument("jacobi_with_second_derivative order");
    }
    if (n == 0) {
        return {Real(1), Real(0), Real(0)};
    }

    Real pkm1 = Real(1);
    Real pk = Real(0.5) * ((alpha - beta) + (alpha + beta + Real(2)) * x);
    Real dpkm1 = Real(0);
    Real dpk = Real(0.5) * (alpha + beta + Real(2));
    Real ddpkm1 = Real(0);
    Real ddpk = Real(0);

    if (n == 1) {
        return {pk, dpk, ddpk};
    }

    for (int k = 1; k < n; ++k) {
        const Real kf = static_cast<Real>(k);
        const Real two_k_ab = Real(2) * kf + alpha + beta;

        const Real denom = Real(2) * (kf + Real(1)) * (kf + alpha + beta + Real(1)) * two_k_ab;
        const Real b_slope = (two_k_ab + Real(1)) * (two_k_ab + Real(2)) * two_k_ab;
        const Real b = b_slope * x + (two_k_ab + Real(1)) * (alpha * alpha - beta * beta);
        const Real c = Real(2) * (kf + alpha) * (kf + beta) * (two_k_ab + Real(2));

        const Real pnext = (b * pk - c * pkm1) / denom;
        const Real dpnext = (b_slope * pk + b * dpk - c * dpkm1) / denom;
        const Real ddpnext = (Real(2) * b_slope * dpk + b * ddpk - c * ddpkm1) / denom;

        pkm1 = pk;
        pk = pnext;
        dpkm1 = dpk;
        dpk = dpnext;
        ddpkm1 = ddpk;
        ddpk = ddpnext;
    }

    return {pk, dpk, ddpk};
}

void jacobi_sequence_with_second_derivatives_to(int n,
                                                Real alpha,
                                                Real beta,
                                                Real x,
                                                std::span<Real> values,
                                                std::span<Real> derivatives,
                                                std::span<Real> second_derivatives) {
    check_jacobi_span(n, values, "jacobi_sequence_with_second_derivatives_to values");
    check_jacobi_span(n, derivatives, "jacobi_sequence_with_second_derivatives_to derivatives");
    check_jacobi_span(n, second_derivatives,
                      "jacobi_sequence_with_second_derivatives_to second derivatives");

    values[0] = Real(1);
    derivatives[0] = Real(0);
    second_derivatives[0] = Real(0);
    if (n == 0) {
        return;
    }

    values[1] = Real(0.5) * ((alpha - beta) + (alpha + beta + Real(2)) * x);
    derivatives[1] = Real(0.5) * (alpha + beta + Real(2));
    second_derivatives[1] = Real(0);

    for (int k = 1; k < n; ++k) {
        const std::size_t prev = static_cast<std::size_t>(k);
        const std::size_t prev2 = static_cast<std::size_t>(k - 1);
        const std::size_t next = static_cast<std::size_t>(k + 1);
        const Real kf = static_cast<Real>(k);
        const Real two_k_ab = Real(2) * kf + alpha + beta;

        const Real denom = Real(2) * (kf + Real(1)) * (kf + alpha + beta + Real(1)) * two_k_ab;
        const Real b_slope = (two_k_ab + Real(1)) * (two_k_ab + Real(2)) * two_k_ab;
        const Real b = b_slope * x + (two_k_ab + Real(1)) * (alpha * alpha - beta * beta);
        const Real c = Real(2) * (kf + alpha) * (kf + beta) * (two_k_ab + Real(2));

        values[next] = (b * values[prev] - c * values[prev2]) / denom;
        derivatives[next] =
            (b_slope * values[prev] + b * derivatives[prev] - c * derivatives[prev2]) / denom;
        second_derivatives[next] =
            (Real(2) * b_slope * derivatives[prev] +
             b * second_derivatives[prev] -
             c * second_derivatives[prev2]) / denom;
    }
}

Real dubiner(int p, int q, Real xi, Real eta) {
    // Map to collapsed coordinates on reference triangle (xi>=0, eta>=0, xi+eta<=1)
    const Real eps = collapsed_coordinate_tolerance();
    const Real one_minus_eta = Real(1) - eta;
    Real a = (std::abs(one_minus_eta) > eps)
             ? (Real(2) * xi / one_minus_eta - Real(1))
             : Real(-1);
    Real b = Real(2) * eta - Real(1);

    Real factor = pow_int(one_minus_eta, p);
    return factor * jacobi(p, Real(0), Real(0), a) * jacobi(q, Real(2 * p + 1), Real(0), b);
}

Real proriol(int p, int q, int r, Real xi, Real eta, Real zeta) {
    // Collapsed coordinates on tetrahedron (xi,eta,zeta >=0, xi+eta+zeta<=1)
    const Real eps = collapsed_coordinate_tolerance();
    const Real one_minus_eta_zeta = Real(1) - eta - zeta;
    Real a = (std::abs(one_minus_eta_zeta) > eps)
             ? (Real(2) * xi / one_minus_eta_zeta - Real(1))
             : Real(-1);

    const Real one_minus_zeta = Real(1) - zeta;
    Real b = (std::abs(one_minus_zeta) > eps)
             ? (Real(2) * eta / one_minus_zeta - Real(1))
             : Real(-1);

    Real c = Real(2) * zeta - Real(1);

    Real factor = pow_int(one_minus_eta_zeta, p) *
                  pow_int(one_minus_zeta, q);

    return factor *
           jacobi(p, Real(0), Real(0), a) *
           jacobi(q, Real(2 * p + 1), Real(0), b) *
           jacobi(r, Real(2 * p + 2 * q + 2), Real(0), c);
}

void legendre_sequence_to(int n, Real x, std::span<Real> values) {
    check_legendre_span(n, values, "legendre_sequence_to values");
    values[0] = Real(1);
    if (n == 0) {
        return;
    }
    values[1] = x;
    for (int k = 2; k <= n; ++k) {
        values[static_cast<std::size_t>(k)] =
            ((Real(2 * k - 1) * x * values[static_cast<std::size_t>(k - 1)]) -
             Real(k - 1) * values[static_cast<std::size_t>(k - 2)]) / Real(k);
    }
}

std::vector<Real> legendre_sequence(int n, Real x) {
    std::vector<Real> seq(legendre_sequence_count(n, "legendre_sequence order"), Real(0));
    legendre_sequence_to(n, x, std::span<Real>(seq.data(), seq.size()));
    return seq;
}

void legendre_sequence_with_derivatives_to(int n,
                                           Real x,
                                           std::span<Real> values,
                                           std::span<Real> derivatives) {
    check_legendre_span(n, values, "legendre_sequence_with_derivatives_to values");
    check_legendre_span(n, derivatives, "legendre_sequence_with_derivatives_to derivatives");
    values[0] = Real(1);
    derivatives[0] = Real(0);

    if (n == 0) {
        return;
    }

    values[1] = x;
    derivatives[1] = Real(1);

    // Use recurrence: P_k = ((2k-1)*x*P_{k-1} - (k-1)*P_{k-2}) / k
    // Derivative: P'_k = ((2k-1)*(P_{k-1} + x*P'_{k-1}) - (k-1)*P'_{k-2}) / k
    for (int k = 2; k <= n; ++k) {
        const std::size_t ku = static_cast<std::size_t>(k);
        const Real c1 = Real(2 * k - 1);
        const Real c2 = Real(k - 1);
        const Real c3 = Real(k);

        values[ku] = (c1 * x * values[ku - 1] - c2 * values[ku - 2]) / c3;
        derivatives[ku] =
            (c1 * (values[ku - 1] + x * derivatives[ku - 1]) - c2 * derivatives[ku - 2]) / c3;
    }
}

PolynomialSequenceDerivatives legendre_sequence_derivatives(int n, Real x) {
    const auto count = legendre_sequence_count(n, "legendre_sequence_with_derivatives order");
    std::vector<Real> vals(count, Real(0));
    std::vector<Real> derivs(count, Real(0));
    legendre_sequence_with_derivatives_to(n,
                                          x,
                                          std::span<Real>(vals.data(), vals.size()),
                                          std::span<Real>(derivs.data(), derivs.size()));
    return {std::move(vals), std::move(derivs)};
}

void legendre_sequence_with_second_derivatives_to(int n,
                                                  Real x,
                                                  std::span<Real> values,
                                                  std::span<Real> derivatives,
                                                  std::span<Real> second_derivatives) {
    check_legendre_span(n, values, "legendre_sequence_with_second_derivatives_to values");
    check_legendre_span(n, derivatives, "legendre_sequence_with_second_derivatives_to derivatives");
    check_legendre_span(n, second_derivatives,
                        "legendre_sequence_with_second_derivatives_to second derivatives");
    values[0] = Real(1);
    derivatives[0] = Real(0);
    second_derivatives[0] = Real(0);
    if (n == 0) {
        return;
    }

    values[1] = x;
    derivatives[1] = Real(1);
    second_derivatives[1] = Real(0);

    for (int k = 2; k <= n; ++k) {
        const std::size_t ku = static_cast<std::size_t>(k);
        const Real c1 = Real(2 * k - 1);
        const Real c2 = Real(k - 1);
        const Real c3 = Real(k);

        values[ku] = (c1 * x * values[ku - 1] - c2 * values[ku - 2]) / c3;
        derivatives[ku] =
            (c1 * (values[ku - 1] + x * derivatives[ku - 1]) - c2 * derivatives[ku - 2]) / c3;
        second_derivatives[ku] =
            (c1 * (Real(2) * derivatives[ku - 1] + x * second_derivatives[ku - 1]) -
             c2 * second_derivatives[ku - 2]) / c3;
    }
}

PolynomialSequenceSecondDerivatives legendre_sequence_second_derivatives(int n, Real x) {
    const auto count = legendre_sequence_count(n, "legendre_sequence_with_second_derivatives order");
    std::vector<Real> vals(count, Real(0));
    std::vector<Real> derivs(count, Real(0));
    std::vector<Real> second_derivs(count, Real(0));
    legendre_sequence_with_second_derivatives_to(
        n,
        x,
        std::span<Real>(vals.data(), vals.size()),
        std::span<Real>(derivs.data(), derivs.size()),
        std::span<Real>(second_derivs.data(), second_derivs.size()));
    return {std::move(vals), std::move(derivs), std::move(second_derivs)};
}

std::vector<Real> gll_nodes(int num_points) {
    if (num_points < 2) {
        throw std::invalid_argument(
            "gll_nodes requires at least 2 points");
    }
    if (num_points > 128) {
        throw std::invalid_argument(
            "gll_nodes num_points exceeds safe limit");
    }

    const int n = num_points;
    std::vector<Real> nodes(static_cast<std::size_t>(n));

    nodes.front() = Real(-1);
    nodes.back() = Real(1);

    if (n == 2) {
        return nodes;
    }

    const Real tolerance = Real(1e-14);
    const int interior = n - 2;
    const int half = (interior + 1) / 2;

    for (int i = 0; i < half; ++i) {
        Real x_root = -std::cos(math::constants::PI * Real(i + 1) / Real(n - 1));
        Real x_prev = std::numeric_limits<Real>::max();

        while (std::abs(x_root - x_prev) > tolerance) {
            x_prev = x_root;
            const auto pn1 = legendre_derivative(n - 1, x_root);
            const auto pn2 = legendre_derivative(n - 2, x_root);

            const Real f = x_root * pn1.value - pn2.value;
            const Real fp = pn1.value + x_root * pn1.derivative - pn2.derivative;
            x_root = x_prev - f / fp;
        }

        const std::size_t left_index = static_cast<std::size_t>(i + 1);
        const std::size_t right_index = static_cast<std::size_t>(n - 2 - i);
        nodes[left_index] = x_root;
        nodes[right_index] = -x_root;
    }

    return nodes;
}

BivariateFirstDerivatives dubiner_derivatives(int p, int q, Real xi, Real eta) {
    // Dubiner basis on reference triangle: (xi>=0, eta>=0, xi+eta<=1)
    // psi_{p,q}(xi,eta) = (1-eta)^p * P_p^{0,0}(a) * P_q^{2p+1,0}(b)
    // where a = 2*xi/(1-eta) - 1, b = 2*eta - 1

    const Real eps = collapsed_coordinate_tolerance();
    const Real one_minus_eta = Real(1) - eta;

    Real a, da_dxi, da_deta;
    if (std::abs(one_minus_eta) > eps) {
        a = Real(2) * xi / one_minus_eta - Real(1);
        da_dxi = Real(2) / one_minus_eta;
        da_deta = Real(2) * xi / (one_minus_eta * one_minus_eta);
    } else {
        // At eta = 1 (apex), use limit: a -> -1 if xi=0
        a = Real(-1);
        da_dxi = Real(0);
        da_deta = Real(0);
    }

    Real b = Real(2) * eta - Real(1);
    Real db_deta = Real(2);

    const auto pa = jacobi_with_second_derivative(p, Real(0), Real(0), a);
    const auto qb = jacobi_with_second_derivative(q, Real(2 * p + 1), Real(0), b);

    // factor = (1-eta)^p
    Real factor = (p > 0) ? pow_int(one_minus_eta, p) : Real(1);
    Real dfactor_deta = (p > 0) ? Real(-p) * pow_int(one_minus_eta, p - 1) : Real(0);

    // psi = factor * Pa * Qb
    Real value = factor * pa.value * qb.value;

    // d(psi)/d(xi) = factor * dPa/da * da/dxi * Qb
    Real dxi = factor * pa.derivative * da_dxi * qb.value;

    // d(psi)/d(eta) = dfactor/deta * Pa * Qb + factor * dPa/da * da/deta * Qb + factor * Pa * dQb/db * db/deta
    Real deta = dfactor_deta * pa.value * qb.value
              + factor * pa.derivative * da_deta * qb.value
              + factor * pa.value * qb.derivative * db_deta;

    return {value, dxi, deta};
}

BivariateDerivatives dubiner_with_second_derivatives(int p, int q, Real xi, Real eta) {
    const Real eps = collapsed_coordinate_tolerance();
    const Real one_minus_eta = Real(1) - eta;

    if (std::abs(one_minus_eta) <= eps) {
        const auto first = dubiner_derivatives(p, q, xi, eta);
        BivariateDerivatives out;
        out.value = first.value;
        out.dxi = first.dxi;
        out.deta = first.deta;
        return out;
    }

    const Jet3 xi_jet = detail::variable_jet(0, xi);
    const Jet3 eta_jet = detail::variable_jet(1, eta);
    const Jet3 one_minus_eta_jet = detail::constant_jet(Real(1)) - eta_jet;
    const Jet3 a = detail::constant_jet(Real(2)) * xi_jet / one_minus_eta_jet - Real(1);
    const Jet3 b = detail::constant_jet(Real(2)) * eta_jet - Real(1);
    const Jet3 factor = detail::pow_int(one_minus_eta_jet, p);
    const Jet3 pa = lift_jacobi(p, Real(0), Real(0), a);
    const Jet3 qb = lift_jacobi(q, Real(2 * p + 1), Real(0), b);
    const Jet3 psi = factor * pa * qb;

    BivariateDerivatives out;
    out.value = psi.value;
    out.dxi = psi.gradient[0];
    out.deta = psi.gradient[1];
    out.dxx = psi.hessian(0, 0);
    out.dxy = psi.hessian(0, 1);
    out.dyy = psi.hessian(1, 1);
    return out;
}

TrivariateFirstDerivatives proriol_derivatives(int p, int q, int r,
                                               Real xi, Real eta, Real zeta) {
    // Proriol basis on reference tetrahedron: (xi,eta,zeta>=0, xi+eta+zeta<=1)
    // psi_{p,q,r} = (1-eta-zeta)^p * (1-zeta)^q * P_p^{0,0}(a) * P_q^{2p+1,0}(b) * P_r^{2p+2q+2,0}(c)
    // where a = 2*xi/(1-eta-zeta) - 1, b = 2*eta/(1-zeta) - 1, c = 2*zeta - 1

    const Real eps = collapsed_coordinate_tolerance();

    const Real one_minus_eta_zeta = Real(1) - eta - zeta;
    const Real one_minus_zeta = Real(1) - zeta;

    // Collapsed coordinate a and its derivatives
    Real a, da_dxi, da_deta, da_dzeta;
    if (std::abs(one_minus_eta_zeta) > eps) {
        a = Real(2) * xi / one_minus_eta_zeta - Real(1);
        da_dxi = Real(2) / one_minus_eta_zeta;
        da_deta = Real(2) * xi / (one_minus_eta_zeta * one_minus_eta_zeta);
        da_dzeta = Real(2) * xi / (one_minus_eta_zeta * one_minus_eta_zeta);
    } else {
        a = Real(-1);
        da_dxi = Real(0);
        da_deta = Real(0);
        da_dzeta = Real(0);
    }

    // Collapsed coordinate b and its derivatives
    Real b, db_deta, db_dzeta;
    if (std::abs(one_minus_zeta) > eps) {
        b = Real(2) * eta / one_minus_zeta - Real(1);
        db_deta = Real(2) / one_minus_zeta;
        db_dzeta = Real(2) * eta / (one_minus_zeta * one_minus_zeta);
    } else {
        b = Real(-1);
        db_deta = Real(0);
        db_dzeta = Real(0);
    }

    Real c = Real(2) * zeta - Real(1);
    Real dc_dzeta = Real(2);

    const auto pa = jacobi_with_second_derivative(p, Real(0), Real(0), a);
    const auto qb = jacobi_with_second_derivative(q, Real(2 * p + 1), Real(0), b);
    const auto rc = jacobi_with_second_derivative(r, Real(2 * p + 2 * q + 2), Real(0), c);

    // factor1 = (1-eta-zeta)^p
    Real factor1 = (p > 0) ? pow_int(one_minus_eta_zeta, p) : Real(1);
    Real df1_deta = (p > 0) ? Real(-p) * pow_int(one_minus_eta_zeta, p - 1) : Real(0);
    Real df1_dzeta = df1_deta; // same since d/deta and d/dzeta of (1-eta-zeta) are both -1

    // factor2 = (1-zeta)^q
    Real factor2 = (q > 0) ? pow_int(one_minus_zeta, q) : Real(1);
    Real df2_dzeta = (q > 0) ? Real(-q) * pow_int(one_minus_zeta, q - 1) : Real(0);

    // psi = factor1 * factor2 * Pa * Qb * Rc
    Real value = factor1 * factor2 * pa.value * qb.value * rc.value;

    // d(psi)/d(xi) = factor1 * factor2 * dPa/da * da/dxi * Qb * Rc
    Real dxi = factor1 * factor2 * pa.derivative * da_dxi * qb.value * rc.value;

    // d(psi)/d(eta): chain rule on factor1, Pa(a), Qb(b)
    Real deta = df1_deta * factor2 * pa.value * qb.value * rc.value
              + factor1 * factor2 * pa.derivative * da_deta * qb.value * rc.value
              + factor1 * factor2 * pa.value * qb.derivative * db_deta * rc.value;

    // d(psi)/d(zeta): chain rule on factor1, factor2, Pa(a), Qb(b), Rc(c)
    Real dzeta = df1_dzeta * factor2 * pa.value * qb.value * rc.value
               + factor1 * df2_dzeta * pa.value * qb.value * rc.value
               + factor1 * factor2 * pa.derivative * da_dzeta * qb.value * rc.value
               + factor1 * factor2 * pa.value * qb.derivative * db_dzeta * rc.value
               + factor1 * factor2 * pa.value * qb.value * rc.derivative * dc_dzeta;

    TrivariateFirstDerivatives out;
    out.value = value;
    out.gradient[0] = dxi;
    out.gradient[1] = deta;
    out.gradient[2] = dzeta;
    return out;
}

TrivariateDerivatives proriol_with_second_derivatives(int p, int q, int r,
                                                      Real xi, Real eta, Real zeta) {
    const Real eps = collapsed_coordinate_tolerance();
    const Real one_minus_eta_zeta = Real(1) - eta - zeta;
    const Real one_minus_zeta = Real(1) - zeta;

    if (std::abs(one_minus_eta_zeta) <= eps || std::abs(one_minus_zeta) <= eps) {
        const auto first = proriol_derivatives(p, q, r, xi, eta, zeta);
        TrivariateDerivatives out;
        out.value = first.value;
        out.gradient = first.gradient;
        return out;
    }

    const Jet3 xi_jet = detail::variable_jet(0, xi);
    const Jet3 eta_jet = detail::variable_jet(1, eta);
    const Jet3 zeta_jet = detail::variable_jet(2, zeta);
    const Jet3 s1 = detail::constant_jet(Real(1)) - eta_jet - zeta_jet;
    const Jet3 s2 = detail::constant_jet(Real(1)) - zeta_jet;
    const Jet3 a = detail::constant_jet(Real(2)) * xi_jet / s1 - Real(1);
    const Jet3 b = detail::constant_jet(Real(2)) * eta_jet / s2 - Real(1);
    const Jet3 c = detail::constant_jet(Real(2)) * zeta_jet - Real(1);

    const Jet3 factor1 = detail::pow_int(s1, p);
    const Jet3 factor2 = detail::pow_int(s2, q);
    const Jet3 pa = lift_jacobi(p, Real(0), Real(0), a);
    const Jet3 qb = lift_jacobi(q, Real(2 * p + 1), Real(0), b);
    const Jet3 rc = lift_jacobi(r, Real(2 * p + 2 * q + 2), Real(0), c);
    const Jet3 psi = factor1 * factor2 * pa * qb * rc;

    TrivariateDerivatives out;
    out.value = psi.value;
    out.gradient = psi.gradient;
    out.hessian = psi.hessian;
    return out;
}

} // namespace orthopoly
} // namespace basis
} // namespace FE
} // namespace svmp
