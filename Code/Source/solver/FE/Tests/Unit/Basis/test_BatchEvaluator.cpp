/**
 * @file test_BatchEvaluator.cpp
 * @brief Unit tests for BatchEvaluator SIMD batch basis evaluation
 */

#include <gtest/gtest.h>

#include "FE/Basis/BatchEvaluator.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"

#include <cmath>
#include <vector>

using namespace svmp::FE;

TEST(BatchEvaluator, WeightedSumMatchesPointwise) {
    basis::LagrangeBasis basis(ElementType::Quad4, 2);
    quadrature::QuadrilateralQuadrature quad(3);

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/true, /*compute_hessians=*/false);

    const std::size_t n = basis.size();
    const std::size_t nq = quad.num_points();

    std::vector<Real> coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        coeffs[i] = Real(0.1) * static_cast<Real>(static_cast<double>(i) + 1.0);
    }

    std::vector<Real> weights(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        const Real x = quad.point(q)[0];
        weights[q] = quad.weight(q) * (Real(1) + Real(0.2) * std::abs(x));
    }

    std::vector<Real> result_batch(nq, Real(0));
    batch.weighted_sum(coeffs.data(), weights.data(), result_batch.data());

    std::vector<Real> result_ref(nq, Real(0));
    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<Real> vals;
        basis.evaluate_values(quad.point(q), vals);
        ASSERT_EQ(vals.size(), n);

        Real sum = Real(0);
        for (std::size_t i = 0; i < n; ++i) {
            sum += coeffs[i] * vals[i] * weights[q];
        }
        result_ref[q] = sum;
    }

    for (std::size_t q = 0; q < nq; ++q) {
        EXPECT_NEAR(result_batch[q], result_ref[q], 1e-12);
    }
}

TEST(BatchEvaluator, WeightedGradientSumMatchesPointwise) {
    basis::LagrangeBasis basis(ElementType::Quad4, 2);
    quadrature::QuadrilateralQuadrature quad(3);

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/true, /*compute_hessians=*/false);

    const std::size_t n = basis.size();
    const std::size_t nq = quad.num_points();
    const int dim = basis.dimension();

    std::vector<Real> coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        coeffs[i] = Real(0.05) * static_cast<Real>(static_cast<double>(i) + 1.0);
    }

    std::vector<Real> weights(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        const Real x = quad.point(q)[0];
        const Real y = quad.point(q)[1];
        weights[q] = quad.weight(q) * (Real(1) + Real(0.1) * std::abs(x) + Real(0.05) * std::abs(y));
    }

    std::vector<Real> result_batch(3 * nq, Real(0));
    batch.weighted_gradient_sum(coeffs.data(), weights.data(), result_batch.data());

    std::vector<Real> result_ref(3 * nq, Real(0));
    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<basis::Gradient> grads;
        basis.evaluate_gradients(quad.point(q), grads);
        ASSERT_EQ(grads.size(), n);

        for (int d = 0; d < dim; ++d) {
            Real sum = Real(0);
            const std::size_t sd = static_cast<std::size_t>(d);
            for (std::size_t i = 0; i < n; ++i) {
                sum += coeffs[i] * grads[i][sd] * weights[q];
            }
            result_ref[sd * nq + q] = sum;
        }
    }

    for (int d = 0; d < 3; ++d) {
        const std::size_t sd = static_cast<std::size_t>(d);
        for (std::size_t q = 0; q < nq; ++q) {
            const double tol = (d < dim) ? 1e-12 : 1e-14;
            EXPECT_NEAR(result_batch[sd * nq + q], result_ref[sd * nq + q], tol);
        }
    }
}

TEST(BatchEvaluator, AssembleStiffnessMatchesPointwise) {
    basis::LagrangeBasis basis(ElementType::Quad4, 2);
    quadrature::QuadrilateralQuadrature quad(4);

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/true, /*compute_hessians=*/false);

    const std::size_t n = basis.size();
    const std::size_t nq = quad.num_points();
    const int dim = basis.dimension();

    ASSERT_EQ(dim, 2);

    // Symmetric positive definite material tensor
    const Real D[4] = {Real(2.0), Real(0.3),
                       Real(0.3), Real(1.5)};

    std::vector<Real> weights(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        weights[q] = quad.weight(q) * (Real(1) + Real(0.2) * std::abs(quad.point(q)[0]));
    }

    std::vector<Real> K_batch(n * n, Real(0));
    batch.assemble_stiffness_contribution(D, weights.data(), K_batch.data());

    // Point-by-point reference using the same reference-space gradients
    std::vector<std::vector<basis::Gradient>> grads_q(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        basis.evaluate_gradients(quad.point(q), grads_q[q]);
        ASSERT_EQ(grads_q[q].size(), n);
    }

    std::vector<Real> K_ref(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Real kij = Real(0);
            for (std::size_t q = 0; q < nq; ++q) {
                Real contribution = Real(0);
                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        contribution += grads_q[q][i][static_cast<std::size_t>(d1)] *
                                        D[static_cast<std::size_t>(d1 * dim + d2)] *
                                        grads_q[q][j][static_cast<std::size_t>(d2)];
                    }
                }
                kij += contribution * weights[q];
            }
            K_ref[i * n + j] = kij;
        }
    }

    for (std::size_t idx = 0; idx < n * n; ++idx) {
        EXPECT_NEAR(K_batch[idx], K_ref[idx], 1e-12);
    }
}

TEST(BatchEvaluator, ThrowsForVectorBases) {
    basis::RaviartThomasBasis rt(ElementType::Quad4, 0);
    quadrature::QuadrilateralQuadrature quad(2);
    EXPECT_THROW(basis::BatchEvaluator(rt, quad, true, false), FEException);
}

TEST(BatchEvaluator, ThrowsWhenGradientsNotComputed) {
    basis::LagrangeBasis basis(ElementType::Quad4, 1);
    quadrature::QuadrilateralQuadrature quad(2);

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/false, /*compute_hessians=*/false);

    const std::size_t n = basis.size();
    const std::size_t nq = quad.num_points();

    std::vector<Real> coeffs(n, Real(1));
    std::vector<Real> weights(nq, Real(1));
    std::vector<Real> grad_result(3 * nq, Real(0));
    std::vector<Real> K(n * n, Real(0));

    const Real D[4] = {Real(1), Real(0), Real(0), Real(1)};

    EXPECT_THROW(batch.weighted_gradient_sum(coeffs.data(), weights.data(), grad_result.data()), FEException);
    EXPECT_THROW(batch.assemble_stiffness_contribution(D, weights.data(), K.data()), FEException);
}

