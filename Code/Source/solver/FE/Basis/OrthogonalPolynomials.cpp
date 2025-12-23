/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "OrthogonalPolynomials.h"
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace basis {
namespace orthopoly {

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

std::pair<Real, Real> legendre_with_derivative(int n, Real x) {
    if (n == 0) {
        return {Real(1), Real(0)};
    }
    if (n == 1) {
        return {x, Real(1)};
    }

    Real p0 = Real(1);
    Real p1 = x;
    for (int k = 2; k <= n; ++k) {
        Real pk = ((Real(2 * k - 1) * x * p1) - Real(k - 1) * p0) / Real(k);
        p0 = p1;
        p1 = pk;
    }

    Real derivative = Real(n) / (Real(1) - x * x) * (p0 - x * p1);
    return {p1, derivative};
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

Real dubiner(int p, int q, Real xi, Real eta) {
    // Map to collapsed coordinates on reference triangle (xi>=0, eta>=0, xi+eta<=1)
    const Real one_minus_eta = Real(1) - eta;
    Real a = (std::abs(one_minus_eta) > std::numeric_limits<Real>::epsilon())
             ? (Real(2) * xi / one_minus_eta - Real(1))
             : Real(-1);
    Real b = Real(2) * eta - Real(1);

    Real factor = std::pow(one_minus_eta, static_cast<Real>(p));
    return factor * jacobi(p, Real(0), Real(0), a) * jacobi(q, Real(2 * p + 1), Real(0), b);
}

Real proriol(int p, int q, int r, Real xi, Real eta, Real zeta) {
    // Collapsed coordinates on tetrahedron (xi,eta,zeta >=0, xi+eta+zeta<=1)
    const Real one_minus_eta_zeta = Real(1) - eta - zeta;
    Real a = (std::abs(one_minus_eta_zeta) > std::numeric_limits<Real>::epsilon())
             ? (Real(2) * xi / one_minus_eta_zeta - Real(1))
             : Real(-1);

    const Real one_minus_zeta = Real(1) - zeta;
    Real b = (std::abs(one_minus_zeta) > std::numeric_limits<Real>::epsilon())
             ? (Real(2) * eta / one_minus_zeta - Real(1))
             : Real(-1);

    Real c = Real(2) * zeta - Real(1);

    Real factor = std::pow(one_minus_eta_zeta, static_cast<Real>(p)) *
                  std::pow(one_minus_zeta, static_cast<Real>(q));

    return factor *
           jacobi(p, Real(0), Real(0), a) *
           jacobi(q, Real(2 * p + 1), Real(0), b) *
           jacobi(r, Real(2 * p + 2 * q + 2), Real(0), c);
}

std::vector<Real> legendre_sequence(int n, Real x) {
    std::vector<Real> seq(static_cast<std::size_t>(n + 1), Real(0));
    seq[0] = Real(1);
    if (n == 0) {
        return seq;
    }
    seq[1] = x;
    for (int k = 2; k <= n; ++k) {
        seq[static_cast<std::size_t>(k)] =
            ((Real(2 * k - 1) * x * seq[static_cast<std::size_t>(k - 1)]) -
             Real(k - 1) * seq[static_cast<std::size_t>(k - 2)]) / Real(k);
    }
    return seq;
}

std::pair<std::vector<Real>, std::vector<Real>> legendre_sequence_with_derivatives(int n, Real x) {
    std::vector<Real> vals(static_cast<std::size_t>(n + 1), Real(0));
    std::vector<Real> derivs(static_cast<std::size_t>(n + 1), Real(0));

    vals[0] = Real(1);
    derivs[0] = Real(0);

    if (n == 0) {
        return {vals, derivs};
    }

    vals[1] = x;
    derivs[1] = Real(1);

    // Use recurrence: P_k = ((2k-1)*x*P_{k-1} - (k-1)*P_{k-2}) / k
    // Derivative: P'_k = ((2k-1)*(P_{k-1} + x*P'_{k-1}) - (k-1)*P'_{k-2}) / k
    for (int k = 2; k <= n; ++k) {
        const std::size_t ku = static_cast<std::size_t>(k);
        const Real c1 = Real(2 * k - 1);
        const Real c2 = Real(k - 1);
        const Real c3 = Real(k);

        vals[ku] = (c1 * x * vals[ku - 1] - c2 * vals[ku - 2]) / c3;
        derivs[ku] = (c1 * (vals[ku - 1] + x * derivs[ku - 1]) - c2 * derivs[ku - 2]) / c3;
    }

    return {vals, derivs};
}

std::tuple<Real, Real, Real> dubiner_with_derivatives(int p, int q, Real xi, Real eta) {
    // Dubiner basis on reference triangle: (xi>=0, eta>=0, xi+eta<=1)
    // psi_{p,q}(xi,eta) = (1-eta)^p * P_p^{0,0}(a) * P_q^{2p+1,0}(b)
    // where a = 2*xi/(1-eta) - 1, b = 2*eta - 1

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(10);
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

    // Compute Jacobi polynomials and derivatives
    Real Pa = jacobi(p, Real(0), Real(0), a);
    Real Pa_deriv = jacobi_derivative(p, Real(0), Real(0), a);

    Real Qb = jacobi(q, Real(2 * p + 1), Real(0), b);
    Real Qb_deriv = jacobi_derivative(q, Real(2 * p + 1), Real(0), b);

    // factor = (1-eta)^p
    Real factor = (p > 0) ? std::pow(one_minus_eta, static_cast<Real>(p)) : Real(1);
    Real dfactor_deta = (p > 0) ? Real(-p) * std::pow(one_minus_eta, static_cast<Real>(p - 1)) : Real(0);

    // psi = factor * Pa * Qb
    Real value = factor * Pa * Qb;

    // d(psi)/d(xi) = factor * dPa/da * da/dxi * Qb
    Real dxi = factor * Pa_deriv * da_dxi * Qb;

    // d(psi)/d(eta) = dfactor/deta * Pa * Qb + factor * dPa/da * da/deta * Qb + factor * Pa * dQb/db * db/deta
    Real deta = dfactor_deta * Pa * Qb
              + factor * Pa_deriv * da_deta * Qb
              + factor * Pa * Qb_deriv * db_deta;

    return {value, dxi, deta};
}

std::tuple<Real, Real, Real, Real> proriol_with_derivatives(int p, int q, int r,
                                                             Real xi, Real eta, Real zeta) {
    // Proriol basis on reference tetrahedron: (xi,eta,zeta>=0, xi+eta+zeta<=1)
    // psi_{p,q,r} = (1-eta-zeta)^p * (1-zeta)^q * P_p^{0,0}(a) * P_q^{2p+1,0}(b) * P_r^{2p+2q+2,0}(c)
    // where a = 2*xi/(1-eta-zeta) - 1, b = 2*eta/(1-zeta) - 1, c = 2*zeta - 1

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(10);

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

    // Jacobi polynomials and derivatives
    Real Pa = jacobi(p, Real(0), Real(0), a);
    Real Pa_deriv = jacobi_derivative(p, Real(0), Real(0), a);

    Real Qb = jacobi(q, Real(2 * p + 1), Real(0), b);
    Real Qb_deriv = jacobi_derivative(q, Real(2 * p + 1), Real(0), b);

    Real Rc = jacobi(r, Real(2 * p + 2 * q + 2), Real(0), c);
    Real Rc_deriv = jacobi_derivative(r, Real(2 * p + 2 * q + 2), Real(0), c);

    // factor1 = (1-eta-zeta)^p
    Real factor1 = (p > 0) ? std::pow(one_minus_eta_zeta, static_cast<Real>(p)) : Real(1);
    Real df1_deta = (p > 0) ? Real(-p) * std::pow(one_minus_eta_zeta, static_cast<Real>(p - 1)) : Real(0);
    Real df1_dzeta = df1_deta; // same since d/deta and d/dzeta of (1-eta-zeta) are both -1

    // factor2 = (1-zeta)^q
    Real factor2 = (q > 0) ? std::pow(one_minus_zeta, static_cast<Real>(q)) : Real(1);
    Real df2_dzeta = (q > 0) ? Real(-q) * std::pow(one_minus_zeta, static_cast<Real>(q - 1)) : Real(0);

    // psi = factor1 * factor2 * Pa * Qb * Rc
    Real value = factor1 * factor2 * Pa * Qb * Rc;

    // d(psi)/d(xi) = factor1 * factor2 * dPa/da * da/dxi * Qb * Rc
    Real dxi = factor1 * factor2 * Pa_deriv * da_dxi * Qb * Rc;

    // d(psi)/d(eta): chain rule on factor1, Pa(a), Qb(b)
    Real deta = df1_deta * factor2 * Pa * Qb * Rc
              + factor1 * factor2 * Pa_deriv * da_deta * Qb * Rc
              + factor1 * factor2 * Pa * Qb_deriv * db_deta * Rc;

    // d(psi)/d(zeta): chain rule on factor1, factor2, Pa(a), Qb(b), Rc(c)
    Real dzeta = df1_dzeta * factor2 * Pa * Qb * Rc
               + factor1 * df2_dzeta * Pa * Qb * Rc
               + factor1 * factor2 * Pa_deriv * da_dzeta * Qb * Rc
               + factor1 * factor2 * Pa * Qb_deriv * db_dzeta * Rc
               + factor1 * factor2 * Pa * Qb * Rc_deriv * dc_dzeta;

    return {value, dxi, deta, dzeta};
}

} // namespace orthopoly
} // namespace basis
} // namespace FE
} // namespace svmp
