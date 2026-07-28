/**
 * @file test_DenseLinearAlgebra.cpp
 * @brief Tests for shared dense linear algebra utilities.
 */

#include <gtest/gtest.h>

#include "FE/Core/FEException.h"
#include "FE/Math/DenseLinearAlgebra.h"

#include <cmath>
#include <limits>
#include <span>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::math;

namespace {

Real multiply_entry(const std::vector<Real>& A,
                    const std::vector<Real>& B,
                    std::size_t n,
                    std::size_t row,
                    std::size_t col) {
    Real sum = Real(0);
    for (std::size_t k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    return sum;
}

} // namespace

TEST(DenseLinearAlgebra, InvertsScaledMatrix) {
    const std::vector<Real> A{
        Real(1.0e9), Real(2.0e6),
        Real(3.0e3), Real(4.0)
    };

    const auto inv = invert_dense_matrix(A, 2u, "scaled 2x2");
    for (std::size_t row = 0; row < 2u; ++row) {
        for (std::size_t col = 0; col < 2u; ++col) {
            const Real expected = (row == col) ? Real(1) : Real(0);
            EXPECT_NEAR(multiply_entry(A, inv, 2u, row, col), expected, Real(1.0e-10));
        }
    }
}

TEST(DenseLinearAlgebra, FactorizationSolvesMultipleRightHandSides) {
    const std::vector<Real> A{
        Real(4), Real(2), Real(0),
        Real(2), Real(5), Real(1),
        Real(0), Real(1), Real(3)
    };

    const auto solver = factor_dense_matrix(A, 3u, "symmetric 3x3");
    EXPECT_EQ(solver.diagnostics.rank, 3u);

    const std::vector<Real> rhs{Real(2), Real(4), Real(6)};
    const auto x = solver.solve(std::span<const Real>(rhs.data(), rhs.size()));
    ASSERT_EQ(x.size(), 3u);

    for (std::size_t row = 0; row < 3u; ++row) {
        Real ax = Real(0);
        for (std::size_t col = 0; col < 3u; ++col) {
            ax += A[row * 3u + col] * x[col];
        }
        EXPECT_NEAR(ax, rhs[row], Real(1.0e-12));
    }

    std::vector<Real> second_rhs{Real(1), Real(-2), Real(0.5)};
    const auto original_second_rhs = second_rhs;
    solver.solve_in_place(std::span<Real>(second_rhs.data(), second_rhs.size()));
    for (std::size_t row = 0; row < 3u; ++row) {
        Real ax = Real(0);
        for (std::size_t col = 0; col < 3u; ++col) {
            ax += A[row * 3u + col] * second_rhs[col];
        }
        EXPECT_NEAR(ax, original_second_rhs[row], Real(1.0e-12));
    }
}

TEST(DenseLinearAlgebra, FactorizationSolvesDenseRightHandSideBlock) {
    const std::vector<Real> A{
        Real(4), Real(2), Real(0),
        Real(2), Real(5), Real(1),
        Real(0), Real(1), Real(3)
    };

    const auto solver = factor_dense_matrix(A, 3u, "symmetric 3x3 block");

    std::vector<Real> rhs{
        Real(2), Real(1),
        Real(4), Real(-2),
        Real(6), Real(0.5)
    };
    const auto original_rhs = rhs;
    solver.solve_in_place(std::span<Real>(rhs.data(), rhs.size()), 2u);

    for (std::size_t rhs_col = 0; rhs_col < 2u; ++rhs_col) {
        for (std::size_t row = 0; row < 3u; ++row) {
            Real ax = Real(0);
            for (std::size_t col = 0; col < 3u; ++col) {
                ax += A[row * 3u + col] * rhs[col * 2u + rhs_col];
            }
            EXPECT_NEAR(ax, original_rhs[row * 2u + rhs_col], Real(1.0e-12));
        }
    }
}

TEST(DenseLinearAlgebra, HighConditionInverseUsesSvdFallback) {
    const std::vector<Real> high_condition{
        Real(1), Real(0),
        Real(0), Real(1.0e-13)
    };

    const auto result =
        invert_dense_matrix_with_diagnostics(high_condition, 2u, "high-condition diagonal");
    EXPECT_EQ(result.diagnostics.rank, 2u);
#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    EXPECT_GT(result.diagnostics.condition_estimate,
              dense_matrix_condition_fallback_threshold());
    EXPECT_TRUE(result.used_svd_fallback);
#else
    EXPECT_FALSE(result.used_svd_fallback);
#endif

    for (std::size_t row = 0; row < 2u; ++row) {
        for (std::size_t col = 0; col < 2u; ++col) {
            const Real expected = (row == col) ? Real(1) : Real(0);
            EXPECT_NEAR(multiply_entry(high_condition, result.inverse, 2u, row, col),
                        expected,
                        Real(1.0e-12));
        }
    }
}

TEST(DenseLinearAlgebra, DiagnosticValidationRejectsUnsupportedCondition) {
    DenseInverseResult result;
    result.diagnostics.rank = 2u;
    result.diagnostics.condition_estimate =
        dense_matrix_condition_error_threshold() * Real(10);

    EXPECT_GT(result.diagnostics.condition_estimate,
              dense_matrix_condition_error_threshold());
    EXPECT_THROW(validate_dense_inverse_diagnostics(
                     result, 2u, "excessive-condition diagonal"),
                 FEException);
}

TEST(DenseLinearAlgebra, ThrowsForScaleAwareSingularPivot) {
    const std::vector<Real> singular{
        Real(1.0e12), Real(2.0e12),
        Real(0.5e12), Real(1.0e12)
    };

    EXPECT_THROW((void)invert_dense_matrix(singular, 2u, "singular 2x2"),
                 FEException);
}

TEST(DenseLinearAlgebra, FactorizationThrowsForRankDeficientMatrix) {
    const std::vector<Real> singular{
        Real(1), Real(2),
        Real(2), Real(4)
    };

    EXPECT_THROW((void)factor_dense_matrix(singular, 2u, "rank-one 2x2"),
                 FEException);
}

TEST(DenseLinearAlgebra, RankUsesScaleAwareTolerance) {
    const std::vector<Real> rank_one{
        Real(1.0e8), Real(2.0e8),
        Real(3.0e8), Real(6.0e8)
    };
    EXPECT_EQ(dense_matrix_rank(rank_one, 2u, 2u), 1u);

    const std::vector<Real> full_rank{
        Real(1.0e8), Real(2.0e8),
        Real(3.0e8), Real(6.1e8)
    };
    EXPECT_EQ(dense_matrix_rank(full_rank, 2u, 2u), 2u);
}

TEST(DenseLinearAlgebra, DiagnosticsReportRankAndConditionEstimate) {
    const std::vector<Real> diagonal{
        Real(4), Real(0),
        Real(0), Real(0.5)
    };
    const auto full =
        dense_matrix_diagnostics(diagonal, 2u, 2u, "diagonal 2x2");
    EXPECT_EQ(full.rank, 2u);
    EXPECT_NEAR(full.largest_singular_value, Real(4), Real(1.0e-14));
    EXPECT_NEAR(full.smallest_retained_singular_value, Real(0.5), Real(1.0e-14));
    EXPECT_NEAR(full.condition_estimate, Real(8), Real(1.0e-14));

    const std::vector<Real> rank_one{
        Real(1), Real(2),
        Real(2), Real(4)
    };
    const auto deficient =
        dense_matrix_diagnostics(rank_one, 2u, 2u, "rank-one 2x2");
    EXPECT_EQ(deficient.rank, 1u);
    EXPECT_TRUE(std::isinf(deficient.condition_estimate));
}

TEST(DenseLinearAlgebra, SymmetricEigenvalueBoundsDoNotRequireOptionalBackend) {
    const std::vector<Real> matrix{
        Real(2), Real(1), Real(0),
        Real(1), Real(2), Real(0),
        Real(0), Real(0), Real(0.25),
    };
    const auto bounds = dense_symmetric_eigenvalue_bounds(
        matrix, 3u, "known symmetric spectrum");
    EXPECT_TRUE(bounds.converged);
    EXPECT_NEAR(bounds.smallest_eigenvalue, Real(0.25), Real(1.0e-13));
    EXPECT_NEAR(bounds.largest_eigenvalue, Real(3.0), Real(1.0e-13));
    EXPECT_LE(bounds.maximum_off_diagonal, bounds.tolerance);
}

TEST(DenseLinearAlgebra, SymmetricEigenvalueBoundsRejectAsymmetricInput) {
    const std::vector<Real> matrix{
        Real(1), Real(0.1),
        Real(0), Real(1),
    };
    EXPECT_THROW(
        (void)dense_symmetric_eigenvalue_bounds(
            matrix, 2u, "asymmetric matrix"),
        FEException);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundHandlesCompatibleNullspace) {
    const std::vector<Real> denominator{
        Real(4), Real(0), Real(0),
        Real(0), Real(1), Real(0),
        Real(0), Real(0), Real(0),
    };
    const std::vector<Real> numerator{
        Real(8), Real(0), Real(0),
        Real(0), Real(3), Real(0),
        Real(0), Real(0), Real(0),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator, denominator, 3u, "compatible singular pencil");
    EXPECT_TRUE(bound.denominator_converged);
    EXPECT_TRUE(bound.quotient_converged);
    EXPECT_EQ(bound.dimension, 3u);
    EXPECT_EQ(bound.positive_rank, 2u);
    EXPECT_EQ(bound.nullity, 1u);
    EXPECT_NEAR(
        bound.smallest_quotient_eigenvalue, Real(2), Real(1.0e-11));
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue, Real(3), Real(1.0e-11));
    EXPECT_GE(
        bound.conservative_upper_bound,
        bound.largest_quotient_eigenvalue);
    EXPECT_NEAR(
        bound.conservative_upper_bound, Real(3), Real(1.0e-10));
    EXPECT_LE(
        bound.maximum_nullspace_residual,
        bound.nullspace_compatibility_tolerance);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundHandlesRotatedNullspace) {
    const std::vector<Real> denominator{
        Real(1), Real(1),
        Real(1), Real(1),
    };
    const std::vector<Real> numerator{
        Real(3), Real(3),
        Real(3), Real(3),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator, denominator, 2u, "rotated rank-one pencil");
    EXPECT_EQ(bound.positive_rank, 1u);
    EXPECT_EQ(bound.nullity, 1u);
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue, Real(3), Real(1.0e-11));
    EXPECT_GE(
        bound.conservative_upper_bound,
        bound.largest_quotient_eigenvalue);
    EXPECT_NEAR(
        bound.conservative_upper_bound, Real(3), Real(1.0e-10));
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundHandlesNoncommutingQuotient) {
    const std::vector<Real> denominator{
        Real(4), Real(0), Real(0), Real(0),
        Real(0), Real(1), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(0),
    };
    const std::vector<Real> numerator{
        Real(8), Real(1), Real(0), Real(0),
        Real(1), Real(3), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(0),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator,
        denominator,
        4u,
        "noncommuting singular pencil");
    const Real expected_largest =
        (Real(5) + std::sqrt(Real(2))) / Real(2);
    EXPECT_EQ(bound.positive_rank, 2u);
    EXPECT_EQ(bound.nullity, 2u);
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue,
        expected_largest,
        Real(1.0e-10));
    EXPECT_GE(
        bound.conservative_upper_bound,
        bound.largest_quotient_eigenvalue);
    EXPECT_NEAR(
        bound.conservative_upper_bound,
        Real(3.5),
        Real(1.0e-9));
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundIsCommonScaleInvariant) {
    const std::vector<Real> base_denominator{
        Real(4), Real(0), Real(0),
        Real(0), Real(1), Real(0),
        Real(0), Real(0), Real(0),
    };
    const std::vector<Real> base_numerator{
        Real(8), Real(0), Real(0),
        Real(0), Real(3), Real(0),
        Real(0), Real(0), Real(0),
    };

    for (const Real scale : {Real(1.0e-120), Real(1.0e120)}) {
        auto denominator = base_denominator;
        auto numerator = base_numerator;
        for (auto& value : denominator) {
            value *= scale;
        }
        for (auto& value : numerator) {
            value *= scale;
        }
        const auto bound = dense_psd_generalized_eigenvalue_bound(
            numerator, denominator, 3u, "commonly scaled pencil");
        EXPECT_EQ(bound.positive_rank, 2u);
        EXPECT_EQ(bound.nullity, 1u);
        EXPECT_NEAR(
            bound.largest_quotient_eigenvalue, Real(3), Real(1.0e-10));
        EXPECT_NEAR(
            bound.conservative_upper_bound, Real(3), Real(1.0e-9));
    }
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundHasInverseLengthScaling) {
    const std::vector<Real> denominator{
        Real(16), Real(0), Real(0),
        Real(0), Real(4), Real(0),
        Real(0), Real(0), Real(0),
    };
    const std::vector<Real> numerator{
        Real(8), Real(0), Real(0),
        Real(0), Real(3), Real(0),
        Real(0), Real(0), Real(0),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator, denominator, 3u, "length-scaled pencil");
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue, Real(0.75), Real(1.0e-11));
    EXPECT_NEAR(
        bound.conservative_upper_bound, Real(0.75), Real(1.0e-10));
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundAcceptsZeroPencil) {
    const std::vector<Real> zero(9u, Real(0));
    const auto bound = dense_psd_generalized_eigenvalue_bound(
        zero, zero, 3u, "zero pencil");
    EXPECT_EQ(bound.positive_rank, 0u);
    EXPECT_EQ(bound.nullity, 3u);
    EXPECT_TRUE(bound.denominator_converged);
    EXPECT_TRUE(bound.quotient_converged);
    EXPECT_EQ(bound.conservative_upper_bound, Real(0));
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundRejectsIncompatibleNullspace) {
    const std::vector<Real> denominator{
        Real(1), Real(0),
        Real(0), Real(0),
    };
    const std::vector<Real> numerator{
        Real(1), Real(0),
        Real(0), Real(1),
    };
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            numerator, denominator, 2u, "incompatible pencil"),
        FEException);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundChecksFullNullspaceResidual) {
    const std::vector<Real> denominator{
        Real(1), Real(0),
        Real(0), Real(0),
    };
    const std::vector<Real> numerator{
        Real(1), Real(1),
        Real(1), Real(0),
    };

    // The scalar nullspace quadratic form is zero, but N*q0 is not.  A
    // compatibility check that only inspected q0^T*N*q0 would miss this.
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            numerator, denominator, 2u, "cross-nullspace pencil"),
        FEException);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundRejectsSmallNullspaceAction) {
    const std::vector<Real> denominator{
        Real(1), Real(0),
        Real(0), Real(0),
    };
    const std::vector<Real> numerator{
        Real(1), Real(0),
        Real(0), Real(1.0e-14),
    };

    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            numerator,
            denominator,
            2u,
            "small incompatible nullspace action"),
        FEException);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundRetainsTinyPositiveMode) {
    const std::vector<Real> denominator{
        Real(1), Real(0),
        Real(0), Real(1.0e-300),
    };
    const std::vector<Real> numerator{
        Real(1), Real(0),
        Real(0), Real(1.0e-14),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator, denominator, 2u, "tiny positive denominator mode");
    const Real analytic_quotient =
        numerator[3] / denominator[3];
    EXPECT_EQ(bound.positive_rank, 2u);
    EXPECT_EQ(bound.nullity, 0u);
    EXPECT_GE(bound.conservative_upper_bound, analytic_quotient);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundRecomputesRotatedDenominator) {
    const Real quarter_above =
        std::nextafter(
            Real(0.25), std::numeric_limits<Real>::infinity());
    const std::vector<Real> denominator{
        Real(1), Real(0.5),
        Real(0.5), quarter_above,
    };
    const std::vector<Real> numerator{
        Real(1), Real(0),
        Real(0), Real(1),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator,
        denominator,
        2u,
        "rounded Jacobi rotation");
    const long double determinant =
        static_cast<long double>(denominator[0]) *
            static_cast<long double>(denominator[3]) -
        static_cast<long double>(denominator[1]) *
            static_cast<long double>(denominator[2]);
    const long double trace =
        static_cast<long double>(denominator[0]) +
        static_cast<long double>(denominator[3]);
    const long double difference =
        static_cast<long double>(denominator[0]) -
        static_cast<long double>(denominator[3]);
    const long double largest_denominator_eigenvalue =
        (trace +
         std::hypot(
             difference,
             2.0L *
                 static_cast<long double>(denominator[1]))) /
        2.0L;
    const long double analytic_quotient =
        largest_denominator_eigenvalue / determinant;
    EXPECT_GE(
        static_cast<long double>(bound.conservative_upper_bound),
        analytic_quotient);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundUsesWideNormalization) {
    const Real below_three =
        std::nextafter(
            std::nextafter(Real(3), Real(0)),
            Real(0));
    const std::vector<Real> denominator{
        Real(3), below_three,
        below_three, Real(3),
    };
    const std::vector<Real> numerator{
        Real(1), Real(0),
        Real(0), Real(1),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator,
        denominator,
        2u,
        "wide denominator normalization");
    const long double analytic_quotient =
        1.0L /
        (static_cast<long double>(denominator[0]) -
         static_cast<long double>(denominator[1]));
    EXPECT_GE(
        static_cast<long double>(bound.conservative_upper_bound),
        analytic_quotient);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundRoundsTinyOutputUp) {
    const std::vector<Real> denominator{
        std::numeric_limits<Real>::max(),
    };
    const std::vector<Real> numerator{
        std::numeric_limits<Real>::denorm_min(),
    };

    const auto bound = dense_psd_generalized_eigenvalue_bound(
        numerator, denominator, 1u, "subnormal generalized bound");
    const long double analytic_quotient =
        static_cast<long double>(numerator.front()) /
        static_cast<long double>(denominator.front());
    EXPECT_EQ(
        bound.conservative_upper_bound,
        std::numeric_limits<Real>::denorm_min());
    EXPECT_GE(
        static_cast<long double>(bound.conservative_upper_bound),
        analytic_quotient);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundRejectsIndefiniteInputs) {
    const std::vector<Real> identity{
        Real(1), Real(0),
        Real(0), Real(1),
    };
    const std::vector<Real> indefinite{
        Real(1), Real(0),
        Real(0), Real(-0.25),
    };
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            identity, indefinite, 2u, "indefinite denominator"),
        FEException);
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            indefinite, identity, 2u, "indefinite numerator"),
        FEException);

    const std::vector<Real> tiny_negative{
        Real(1), Real(0),
        Real(0), Real(-1.0e-20),
    };
    const std::vector<Real> compatible_numerator{
        Real(1), Real(0),
        Real(0), Real(0),
    };
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            compatible_numerator,
            tiny_negative,
            2u,
            "tiny indefinite denominator"),
        FEException);
}

TEST(DenseLinearAlgebra, PsdGeneralizedBoundRejectsInvalidEntries) {
    const std::vector<Real> identity{
        Real(1), Real(0),
        Real(0), Real(1),
    };
    const std::vector<Real> asymmetric{
        Real(1), Real(0.1),
        Real(0), Real(1),
    };
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            asymmetric, identity, 2u, "asymmetric numerator"),
        FEException);

    auto nonfinite = identity;
    nonfinite[0] = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            identity, nonfinite, 2u, "nonfinite denominator"),
        FEException);
}

TEST(DenseLinearAlgebra, PseudoInverseHandlesSingularMatrixWithoutNormalEquations) {
#if !(defined(FE_HAS_EIGEN) && FE_HAS_EIGEN)
    GTEST_SKIP() << "rank-revealing pseudo-inverse requires FE_ENABLE_EIGEN";
#endif
    const std::vector<Real> rank_one{
        Real(1), Real(2),
        Real(2), Real(4)
    };

    const auto pinv =
        rank_revealing_pseudo_inverse(rank_one, 2u, 2u, "rank-one 2x2");
    EXPECT_EQ(pinv.rank, 1u);
    EXPECT_NEAR(pinv.inverse[0], Real(0.04), Real(1.0e-13));
    EXPECT_NEAR(pinv.inverse[1], Real(0.08), Real(1.0e-13));
    EXPECT_NEAR(pinv.inverse[2], Real(0.08), Real(1.0e-13));
    EXPECT_NEAR(pinv.inverse[3], Real(0.16), Real(1.0e-13));

    std::vector<Real> projection(4u, Real(0));
    for (std::size_t row = 0; row < 2u; ++row) {
        for (std::size_t col = 0; col < 2u; ++col) {
            for (std::size_t a = 0; a < 2u; ++a) {
                for (std::size_t b = 0; b < 2u; ++b) {
                    projection[row * 2u + col] +=
                        rank_one[row * 2u + a] * pinv.inverse[a * 2u + b] *
                        rank_one[b * 2u + col];
                }
            }
            EXPECT_NEAR(projection[row * 2u + col],
                        rank_one[row * 2u + col],
                        Real(1.0e-12));
        }
    }
}

TEST(DenseLinearAlgebra, PseudoInverseDropsNearZeroSingularValues) {
#if !(defined(FE_HAS_EIGEN) && FE_HAS_EIGEN)
    GTEST_SKIP() << "rank-revealing pseudo-inverse requires FE_ENABLE_EIGEN";
#endif
    const std::vector<Real> near_singular{
        Real(1), Real(0),
        Real(0), Real(1.0e-18)
    };

    const auto pinv =
        rank_revealing_pseudo_inverse(near_singular, 2u, 2u, "near-singular 2x2");
    EXPECT_EQ(pinv.rank, 1u);
    EXPECT_GT(pinv.tolerance, Real(1.0e-18));
    EXPECT_NEAR(pinv.inverse[0], Real(1), Real(1.0e-14));
    EXPECT_NEAR(pinv.inverse[1], Real(0), Real(1.0e-14));
    EXPECT_NEAR(pinv.inverse[2], Real(0), Real(1.0e-14));
    EXPECT_NEAR(pinv.inverse[3], Real(0), Real(1.0e-14));
}
