/**
 * @file test_BatchEvaluator.cpp
 * @brief Unit tests for BatchEvaluator SIMD batch basis evaluation
 */

#include <gtest/gtest.h>

#include "FE/Assembly/BatchedProjection.h"
#include "FE/Assembly/BatchedStiffness.h"
#include "FE/Basis/BatchEvaluator.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/HexahedronQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/QuadratureRule.h"

#include <cmath>
#include <vector>

using namespace svmp::FE;

namespace {

class CustomQuadratureRule final : public quadrature::QuadratureRule {
public:
    CustomQuadratureRule(svmp::CellFamily family,
                         int dimension,
                         int order,
                         std::vector<quadrature::QuadPoint> points,
                         std::vector<Real> weights)
        : QuadratureRule(family, dimension, order) {
        set_data(std::move(points), std::move(weights));
    }
};

CustomQuadratureRule make_pyramid_quadrature_with_apex() {
    return CustomQuadratureRule(
        svmp::CellFamily::Pyramid, 3, 4,
        {
            quadrature::QuadPoint{Real(0), Real(0), Real(1)},
            quadrature::QuadPoint{Real(0.08), Real(-0.06), Real(0.35)},
            quadrature::QuadPoint{Real(-0.12), Real(0.1), Real(0.5)}
        },
        {Real(0.2), Real(0.5), Real(0.6333333333333333)});
}

CustomQuadratureRule make_pyramid_interior_quadrature() {
    return CustomQuadratureRule(
        svmp::CellFamily::Pyramid, 3, 4,
        {
            quadrature::QuadPoint{Real(0.0), Real(0.0), Real(0.15)},
            quadrature::QuadPoint{Real(0.18), Real(-0.12), Real(0.3)},
            quadrature::QuadPoint{Real(-0.2), Real(0.1), Real(0.42)},
            quadrature::QuadPoint{Real(0.04), Real(-0.03), Real(0.78)}
        },
        {Real(0.2), Real(0.3), Real(0.4), Real(0.4333333333333333)});
}

template <typename ScalarBasis>
void expect_pyramid_value_only_batch_succeeds(const ScalarBasis& basis,
                                              const quadrature::QuadratureRule& quad) {
    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/false, /*compute_hessians=*/false);
    const auto& data = batch.data();
    ASSERT_FALSE(data.has_gradients);
    ASSERT_FALSE(data.has_hessians);

    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<Real> values;
        basis.evaluate_values(quad.point(q), values);
        ASSERT_EQ(values.size(), basis.size());
        for (std::size_t i = 0; i < basis.size(); ++i) {
            EXPECT_NEAR(data.value(i, q), values[i], 1e-12);
        }
    }
}

} // namespace

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
    assembly::weighted_sum(batch, coeffs.data(), weights.data(), result_batch.data());

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

TEST(BatchEvaluator, UsesPaddedAlignedPrimaryStorage) {
    basis::LagrangeBasis basis(ElementType::Line2, 2);
    auto quad = CustomQuadratureRule(
        svmp::CellFamily::Line, 1, 3,
        {
            quadrature::QuadPoint{Real(-0.5), Real(0), Real(0)},
            quadrature::QuadPoint{Real(0.0), Real(0), Real(0)},
            quadrature::QuadPoint{Real(0.5), Real(0), Real(0)}
        },
        {Real(1.0), Real(1.0), Real(1.0)});

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/true, /*compute_hessians=*/true);
    const auto& data = batch.data();

    ASSERT_GE(data.quad_stride, data.num_quad_points);
    EXPECT_EQ(data.values.size(), data.num_basis * data.quad_stride);
    EXPECT_EQ(data.gradients.size(), data.num_basis * 3u * data.quad_stride);
    EXPECT_EQ(data.hessians.size(), data.num_basis * 9u * data.quad_stride);

    for (std::size_t i = 0; i < data.num_basis; ++i) {
        for (std::size_t q = data.num_quad_points; q < data.quad_stride; ++q) {
            EXPECT_EQ(data.values_for_basis(i)[q], Real(0));
            EXPECT_EQ(data.gradients_for_basis(i, 0u)[q], Real(0));
            EXPECT_EQ(data.hessian(i, 0u, 0u, q), Real(0));
        }
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
    assembly::weighted_gradient_sum(batch, coeffs.data(), weights.data(), result_batch.data());

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
    assembly::assemble_stiffness_contribution(batch, D, weights.data(), K_batch.data());

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

TEST(BatchEvaluator, AssembleStiffnessMatchesPointwise_1D) {
    basis::LagrangeBasis basis(ElementType::Line2, 3);
    quadrature::GaussQuadrature1D quad(5);

    basis::BatchEvaluator batch(basis, quad, true, false);

    const std::size_t n = basis.size();
    const std::size_t nq = quad.num_points();

    const Real D[1] = {Real(3.5)};

    std::vector<Real> weights(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        weights[q] = quad.weight(q);
    }

    std::vector<Real> K_batch(n * n, Real(0));
    assembly::assemble_stiffness_contribution(batch, D, weights.data(), K_batch.data());

    std::vector<Real> K_ref(n * n, Real(0));
    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<basis::Gradient> grads;
        basis.evaluate_gradients(quad.point(q), grads);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                K_ref[i * n + j] += grads[i][0] * D[0] * grads[j][0] * weights[q];
            }
        }
    }

    for (std::size_t idx = 0; idx < n * n; ++idx) {
        EXPECT_NEAR(K_batch[idx], K_ref[idx], 1e-12);
    }
}

TEST(BatchEvaluator, AssembleStiffnessSupportsNonsymmetricTensor) {
    basis::LagrangeBasis basis(ElementType::Quad4, 1);
    quadrature::QuadrilateralQuadrature quad(3);

    basis::BatchEvaluator batch(basis, quad, true, false);

    const std::size_t n = basis.size();
    const std::size_t nq = quad.num_points();
    const int dim = basis.dimension();

    ASSERT_EQ(dim, 2);

    const Real D[4] = {Real(1.75), Real(0.8),
                       Real(-0.25), Real(0.9)};

    std::vector<Real> weights(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        weights[q] = quad.weight(q);
    }

    std::vector<Real> K_batch(n * n, Real(42));
    assembly::assemble_stiffness_contribution(batch, D, weights.data(), K_batch.data());

    std::vector<Real> K_ref(n * n, Real(0));
    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<basis::Gradient> grads;
        basis.evaluate_gradients(quad.point(q), grads);
        ASSERT_EQ(grads.size(), n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Real contribution = Real(0);
                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        contribution += grads[i][static_cast<std::size_t>(d1)] *
                                        D[static_cast<std::size_t>(d1 * dim + d2)] *
                                        grads[j][static_cast<std::size_t>(d2)];
                    }
                }
                K_ref[i * n + j] += contribution * weights[q];
            }
        }
    }

    bool found_nonsymmetric_entry = false;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_NEAR(K_batch[i * n + j], K_ref[i * n + j], 1e-12);
            if (i != j && std::abs(K_ref[i * n + j] - K_ref[j * n + i]) > Real(1e-12)) {
                found_nonsymmetric_entry = true;
            }
        }
    }
    EXPECT_TRUE(found_nonsymmetric_entry);
}

TEST(BatchEvaluator, AssembleStiffnessMatchesPointwise_3D) {
    basis::LagrangeBasis basis(ElementType::Hex8, 1);
    quadrature::HexahedronQuadrature quad(2);

    basis::BatchEvaluator batch(basis, quad, true, false);

    const std::size_t n = basis.size();
    const std::size_t nq = quad.num_points();
    const int dim = 3;

    // Non-diagonal SPD material tensor
    const Real D[9] = {
        Real(2.0), Real(0.3), Real(0.1),
        Real(0.3), Real(1.5), Real(0.2),
        Real(0.1), Real(0.2), Real(1.8)
    };

    std::vector<Real> weights(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        weights[q] = quad.weight(q);
    }

    std::vector<Real> K_batch(n * n, Real(0));
    assembly::assemble_stiffness_contribution(batch, D, weights.data(), K_batch.data());

    std::vector<Real> K_ref(n * n, Real(0));
    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<basis::Gradient> grads;
        basis.evaluate_gradients(quad.point(q), grads);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Real kij = Real(0);
                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        kij += grads[i][static_cast<std::size_t>(d1)] *
                               D[static_cast<std::size_t>(d1 * dim + d2)] *
                               grads[j][static_cast<std::size_t>(d2)];
                    }
                }
                K_ref[i * n + j] += kij * weights[q];
            }
        }
    }

    for (std::size_t idx = 0; idx < n * n; ++idx) {
        EXPECT_NEAR(K_batch[idx], K_ref[idx], 1e-12);
    }
}

TEST(BatchEvaluator, HessiansMatchPointwiseOnQuadrilateralLagrange) {
    basis::LagrangeBasis basis(ElementType::Quad4, 2);
    quadrature::QuadrilateralQuadrature quad(4);

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/true, /*compute_hessians=*/true);
    const auto& data = batch.data();

    ASSERT_TRUE(data.has_gradients);
    ASSERT_TRUE(data.has_hessians);
    ASSERT_EQ(data.num_basis, basis.size());
    ASSERT_EQ(data.num_quad_points, quad.num_points());

    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<basis::Hessian> hessians;
        basis.evaluate_hessians(quad.point(q), hessians);
        ASSERT_EQ(hessians.size(), basis.size());

        for (std::size_t i = 0; i < basis.size(); ++i) {
            for (std::size_t d1 = 0; d1 < 3; ++d1) {
                for (std::size_t d2 = 0; d2 < 3; ++d2) {
                    EXPECT_NEAR(data.hessian(i, d1, d2, q), hessians[i](d1, d2), 1e-12)
                        << "i=" << i << ", q=" << q << ", d1=" << d1 << ", d2=" << d2;
                }
            }
        }
    }
}

TEST(BatchEvaluator, HessiansMatchPointwiseOnPyramidLagrange) {
    basis::LagrangeBasis basis(ElementType::Pyramid5, 4);
    const auto quad = make_pyramid_interior_quadrature();

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/false, /*compute_hessians=*/true);
    const auto& data = batch.data();

    ASSERT_FALSE(data.has_gradients);
    ASSERT_TRUE(data.has_hessians);
    ASSERT_EQ(data.num_basis, basis.size());
    ASSERT_EQ(data.num_quad_points, quad.num_points());

    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<basis::Hessian> hessians;
        basis.evaluate_hessians(quad.point(q), hessians);
        ASSERT_EQ(hessians.size(), basis.size());

        for (std::size_t i = 0; i < basis.size(); ++i) {
            for (std::size_t d1 = 0; d1 < 3; ++d1) {
                for (std::size_t d2 = 0; d2 < 3; ++d2) {
                    EXPECT_NEAR(data.hessian(i, d1, d2, q), hessians[i](d1, d2), 1e-10)
                        << "i=" << i << ", q=" << q << ", d1=" << d1 << ", d2=" << d2;
                }
            }
        }
    }
}

TEST(BatchEvaluator, UnusedDimensionsRemainZeroForLineHessians) {
    basis::LagrangeBasis basis(ElementType::Line2, 3);
    quadrature::GaussQuadrature1D quad(5);

    basis::BatchEvaluator batch(basis, quad, /*compute_gradients=*/false, /*compute_hessians=*/true);
    const auto& data = batch.data();

    ASSERT_TRUE(data.has_hessians);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t q = 0; q < quad.num_points(); ++q) {
            EXPECT_NEAR(data.hessian(i, 0, 1, q), 0.0, 1e-14);
            EXPECT_NEAR(data.hessian(i, 1, 0, q), 0.0, 1e-14);
            EXPECT_NEAR(data.hessian(i, 0, 2, q), 0.0, 1e-14);
            EXPECT_NEAR(data.hessian(i, 2, 0, q), 0.0, 1e-14);
            EXPECT_NEAR(data.hessian(i, 1, 1, q), 0.0, 1e-14);
            EXPECT_NEAR(data.hessian(i, 1, 2, q), 0.0, 1e-14);
            EXPECT_NEAR(data.hessian(i, 2, 1, q), 0.0, 1e-14);
            EXPECT_NEAR(data.hessian(i, 2, 2, q), 0.0, 1e-14);
        }
    }
}

TEST(BatchEvaluator, PyramidApexValueOnlyConstructionSucceeds) {
    const auto quad = make_pyramid_quadrature_with_apex();

    basis::LagrangeBasis pyramid5(ElementType::Pyramid5, 2);
    expect_pyramid_value_only_batch_succeeds(pyramid5, quad);

    basis::LagrangeBasis pyramid14(ElementType::Pyramid14, 2);
    expect_pyramid_value_only_batch_succeeds(pyramid14, quad);

    basis::SerendipityBasis pyramid13(ElementType::Pyramid13, 2);
    expect_pyramid_value_only_batch_succeeds(pyramid13, quad);
}

TEST(BatchEvaluator, PyramidApexGradientConstructionThrows) {
    const auto quad = make_pyramid_quadrature_with_apex();

    basis::LagrangeBasis pyramid5(ElementType::Pyramid5, 2);
    EXPECT_THROW((void)basis::BatchEvaluator(pyramid5, quad, /*compute_gradients=*/true, /*compute_hessians=*/false),
                 basis::BasisEvaluationException);

    basis::LagrangeBasis pyramid14(ElementType::Pyramid14, 2);
    EXPECT_THROW((void)basis::BatchEvaluator(pyramid14, quad, /*compute_gradients=*/true, /*compute_hessians=*/false),
                 basis::BasisEvaluationException);

    basis::SerendipityBasis pyramid13(ElementType::Pyramid13, 2);
    EXPECT_THROW((void)basis::BatchEvaluator(pyramid13, quad, /*compute_gradients=*/true, /*compute_hessians=*/false),
                 basis::BasisEvaluationException);
}

TEST(BatchEvaluator, PyramidApexHessianConstructionThrows) {
    const auto quad = make_pyramid_quadrature_with_apex();

    basis::LagrangeBasis pyramid5(ElementType::Pyramid5, 2);
    EXPECT_THROW((void)basis::BatchEvaluator(pyramid5, quad, /*compute_gradients=*/false, /*compute_hessians=*/true),
                 basis::BasisEvaluationException);

    basis::LagrangeBasis pyramid14(ElementType::Pyramid14, 2);
    EXPECT_THROW((void)basis::BatchEvaluator(pyramid14, quad, /*compute_gradients=*/false, /*compute_hessians=*/true),
                 basis::BasisEvaluationException);

    basis::SerendipityBasis pyramid13(ElementType::Pyramid13, 2);
    EXPECT_THROW((void)basis::BatchEvaluator(pyramid13, quad, /*compute_gradients=*/false, /*compute_hessians=*/true),
                 basis::BasisEvaluationException);
}

TEST(BatchEvaluator, ThrowsForVectorBases) {
    basis::RaviartThomasBasis rt(ElementType::Quad4, 0);
    quadrature::QuadrilateralQuadrature quad(2);
    EXPECT_THROW(basis::BatchEvaluator(rt, quad, true, false), basis::BasisConfigurationException);
}

TEST(BatchEvaluator, ThrowsForVectorBasesWhenHessiansRequested) {
    basis::RaviartThomasBasis rt(ElementType::Quad4, 0);
    quadrature::QuadrilateralQuadrature quad(2);
    EXPECT_THROW((void)basis::BatchEvaluator(rt, quad, false, true), basis::BasisConfigurationException);
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

    EXPECT_THROW(assembly::weighted_gradient_sum(batch, coeffs.data(), weights.data(), grad_result.data()),
                 basis::BasisEvaluationException);
    EXPECT_THROW(assembly::assemble_stiffness_contribution(batch, D, weights.data(), K.data()),
                 basis::BasisEvaluationException);
}
