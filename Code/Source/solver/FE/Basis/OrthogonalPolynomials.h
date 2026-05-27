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
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {
namespace orthopoly {

/// Accuracy envelope covered by regression tests for the forward Jacobi
/// recurrence used by Dubiner/Proriol modal bases. These are validation limits,
/// not runtime guards; inputs outside this envelope are accepted but do not carry
/// the same documented accuracy contract.
inline constexpr int kMaxValidatedJacobiRecurrenceOrder = 12;
inline constexpr int kMaxValidatedJacobiAlphaBetaSum = 40;
inline constexpr int kMaxValidatedSimplexModalTotalOrder = 12;

struct BivariateDerivatives {
    Real value{Real(0)};
    Real dxi{Real(0)};
    Real deta{Real(0)};
    Real dxx{Real(0)};
    Real dxy{Real(0)};
    Real dyy{Real(0)};
};

struct UnivariateDerivatives {
    Real value{Real(0)};
    Real derivative{Real(0)};
    Real second_derivative{Real(0)};
};

struct UnivariateFirstDerivative {
    Real value{Real(0)};
    Real derivative{Real(0)};
};

struct PolynomialSequenceDerivatives {
    std::vector<Real> values;
    std::vector<Real> derivatives;
};

struct PolynomialSequenceSecondDerivatives {
    std::vector<Real> values;
    std::vector<Real> derivatives;
    std::vector<Real> second_derivatives;
};

struct BivariateFirstDerivatives {
    Real value{Real(0)};
    Real dxi{Real(0)};
    Real deta{Real(0)};
};

struct TrivariateFirstDerivatives {
    Real value{Real(0)};
    math::Vector<Real, 3> gradient{};
};

struct TrivariateDerivatives {
    Real value{Real(0)};
    math::Vector<Real, 3> gradient{};
    math::Matrix<Real, 3, 3> hessian{};
};

/// Evaluate Legendre polynomial P_n(x)
[[nodiscard]] Real legendre(int n, Real x);

/// Evaluate Legendre polynomial and its derivative with named fields
[[nodiscard]] UnivariateFirstDerivative legendre_derivative(int n, Real x);

/// Integral of Legendre polynomial from -1 to x (used for hierarchical bases)
[[nodiscard]] Real integrated_legendre(int n, Real x);

/// Evaluate Jacobi polynomial P_n^{(alpha,beta)}(x)
[[nodiscard]] Real jacobi(int n, Real alpha, Real beta, Real x);

/// Evaluate derivative of Jacobi polynomial
[[nodiscard]] Real jacobi_derivative(int n, Real alpha, Real beta, Real x);

/// Evaluate second derivative of Jacobi polynomial
[[nodiscard]] Real jacobi_second_derivative(int n, Real alpha, Real beta, Real x);

/// Evaluate Jacobi polynomial and its first two derivatives in one recurrence
[[nodiscard]] UnivariateDerivatives jacobi_with_second_derivative(int n, Real alpha, Real beta, Real x);

/// Compute Jacobi sequence P_0...P_n and first/second derivatives into caller-owned storage
void jacobi_sequence_with_second_derivatives_to(int n,
                                                Real alpha,
                                                Real beta,
                                                Real x,
                                                std::span<Real> values,
                                                std::span<Real> derivatives,
                                                std::span<Real> second_derivatives);

/// Dubiner basis function on the reference triangle
[[nodiscard]] Real dubiner(int p, int q, Real xi, Real eta);

/// Proriol polynomial on the reference tetrahedron
[[nodiscard]] Real proriol(int p, int q, int r, Real xi, Real eta, Real zeta);

/// Convenience: compute sequence P_0 ... P_n at x
[[nodiscard]] std::vector<Real> legendre_sequence(int n, Real x);

/// Compute sequence P_0 ... P_n at x into caller-owned storage
void legendre_sequence_to(int n, Real x, std::span<Real> values);

/// Compute sequence P_0...P_n and their derivatives at x
[[nodiscard]] PolynomialSequenceDerivatives legendre_sequence_derivatives(int n, Real x);

/// Compute sequence P_0...P_n and their derivatives into caller-owned storage
void legendre_sequence_with_derivatives_to(int n,
                                           Real x,
                                           std::span<Real> values,
                                           std::span<Real> derivatives);

/// Compute sequence P_0...P_n and their first/second derivatives at x
[[nodiscard]] PolynomialSequenceSecondDerivatives legendre_sequence_second_derivatives(int n, Real x);

/// Compute sequence P_0...P_n and their first/second derivatives into caller-owned storage
void legendre_sequence_with_second_derivatives_to(int n,
                                                  Real x,
                                                  std::span<Real> values,
                                                  std::span<Real> derivatives,
                                                  std::span<Real> second_derivatives);

/// Generate Gauss-Lobatto-Legendre nodes on [-1, 1].
[[nodiscard]] std::vector<Real> gll_nodes(int num_points);

/// Dubiner basis function with derivatives on the reference triangle
/// Returns named value and first derivatives
[[nodiscard]] BivariateFirstDerivatives dubiner_derivatives(int p, int q, Real xi, Real eta);

/// Dubiner basis function with first and second derivatives on the reference triangle
[[nodiscard]] BivariateDerivatives dubiner_with_second_derivatives(int p, int q, Real xi, Real eta);

/// Proriol polynomial with derivatives on the reference tetrahedron
/// Returns named value and first derivatives
[[nodiscard]] TrivariateFirstDerivatives proriol_derivatives(int p, int q, int r,
                                                             Real xi, Real eta, Real zeta);

/// Proriol polynomial with first and second derivatives on the reference tetrahedron
[[nodiscard]] TrivariateDerivatives proriol_with_second_derivatives(int p, int q, int r,
                                                                    Real xi, Real eta, Real zeta);

} // namespace orthopoly
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_ORTHOGONALPOLYNOMIALS_H
