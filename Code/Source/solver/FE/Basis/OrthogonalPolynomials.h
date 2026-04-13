/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_ORTHOGONALPOLYNOMIALS_H
#define SVMP_FE_BASIS_ORTHOGONALPOLYNOMIALS_H

/**
 * @file OrthogonalPolynomials.h
 * @brief Lightweight orthogonal polynomial utilities used by basis families
 *
 * Provides Legendre, Jacobi, Dubiner, and Proriol polynomial evaluations with
 * recurrence-based implementations suitable for element-level computations.
 */

#include "Core/Types.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"
#include <tuple>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {
namespace orthopoly {

struct BivariateDerivatives {
    Real value{Real(0)};
    Real dxi{Real(0)};
    Real deta{Real(0)};
    Real dxx{Real(0)};
    Real dxy{Real(0)};
    Real dyy{Real(0)};
};

struct TrivariateDerivatives {
    Real value{Real(0)};
    math::Vector<Real, 3> gradient{};
    math::Matrix<Real, 3, 3> hessian{};
};

/// Evaluate Legendre polynomial P_n(x)
Real legendre(int n, Real x);

/// Evaluate Legendre polynomial and its derivative (P_n, P'_n)
std::pair<Real, Real> legendre_with_derivative(int n, Real x);

/// Integral of Legendre polynomial from -1 to x (used for hierarchical bases)
Real integrated_legendre(int n, Real x);

/// Evaluate Jacobi polynomial P_n^{(alpha,beta)}(x)
Real jacobi(int n, Real alpha, Real beta, Real x);

/// Evaluate derivative of Jacobi polynomial
Real jacobi_derivative(int n, Real alpha, Real beta, Real x);

/// Evaluate second derivative of Jacobi polynomial
Real jacobi_second_derivative(int n, Real alpha, Real beta, Real x);

/// Dubiner basis function on the reference triangle
Real dubiner(int p, int q, Real xi, Real eta);

/// Proriol polynomial on the reference tetrahedron
Real proriol(int p, int q, int r, Real xi, Real eta, Real zeta);

/// Convenience: compute sequence P_0 ... P_n at x
std::vector<Real> legendre_sequence(int n, Real x);

/// Compute sequence P_0...P_n and their derivatives at x
/// Returns pair of vectors: (values, derivatives)
std::pair<std::vector<Real>, std::vector<Real>> legendre_sequence_with_derivatives(int n, Real x);

/// Compute sequence P_0...P_n and their first/second derivatives at x
std::tuple<std::vector<Real>, std::vector<Real>, std::vector<Real>>
legendre_sequence_with_second_derivatives(int n, Real x);

/// Dubiner basis function with derivatives on the reference triangle
/// Returns (value, dxi, deta)
std::tuple<Real, Real, Real> dubiner_with_derivatives(int p, int q, Real xi, Real eta);

/// Dubiner basis function with first and second derivatives on the reference triangle
BivariateDerivatives dubiner_with_second_derivatives(int p, int q, Real xi, Real eta);

/// Proriol polynomial with derivatives on the reference tetrahedron
/// Returns (value, dxi, deta, dzeta)
std::tuple<Real, Real, Real, Real> proriol_with_derivatives(int p, int q, int r, Real xi, Real eta, Real zeta);

/// Proriol polynomial with first and second derivatives on the reference tetrahedron
TrivariateDerivatives proriol_with_second_derivatives(int p, int q, int r,
                                                      Real xi, Real eta, Real zeta);

} // namespace orthopoly
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_ORTHOGONALPOLYNOMIALS_H
