/**
 * @file test_DenseLinearAlgebra.cpp
 * @brief Tests for shared dense linear algebra utilities.
 */

#include <gtest/gtest.h>

#include "FE/Core/FEException.h"
#include "FE/Math/DenseLinearAlgebra.h"

#include <array>
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

std::vector<Real> p1_triangle_symmetric_gradient_matrix() {
    constexpr std::size_t scalar_dofs = 3u;
    constexpr std::size_t components = 2u;
    constexpr std::size_t dimension = scalar_dofs * components;
    const std::array<std::array<Real, 2>, scalar_dofs> gradients{{
        {{Real(-1), Real(-1)}},
        {{Real(1), Real(0)}},
        {{Real(0), Real(1)}},
    }};

    std::vector<Real> matrix(dimension * dimension, Real(0));
    for (std::size_t row = 0; row < dimension; ++row) {
        const std::size_t row_component = row / scalar_dofs;
        const std::size_t row_basis = row % scalar_dofs;
        for (std::size_t column = row;
             column < dimension;
             ++column) {
            const std::size_t column_component =
                column / scalar_dofs;
            const std::size_t column_basis =
                column % scalar_dofs;
            Real gradient_inner = Real(0);
            for (std::size_t direction = 0;
                 direction < components;
                 ++direction) {
                gradient_inner +=
                    gradients[row_basis][direction] *
                    gradients[column_basis][direction];
            }
            // On the unit right triangle, area = 1/2 and
            // 2 eps(phi_i e_c):eps(phi_j e_d)
            //   = delta_cd grad(phi_i).grad(phi_j)
            //     + d_c(phi_j) d_d(phi_i).
            const Real value =
                Real(0.5) *
                ((row_component == column_component
                      ? gradient_inner
                      : Real(0)) +
                 gradients[row_basis][column_component] *
                     gradients[column_basis][row_component]);
            matrix[row * dimension + column] = value;
            matrix[column * dimension + row] = value;
        }
    }
    return matrix;
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

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspaceQuotientsScalarP1Constant) {
    // Unit-right-triangle H1 seminorm and the normal-gradient trace on x=0.
    // Quotienting the constant leaves diag(1/2, 1/2) in the denominator
    // and diag(1, 0) in the numerator, so the exact largest quotient is 2.
    const std::vector<Real> denominator{
        Real(1), Real(-0.5), Real(-0.5),
        Real(-0.5), Real(0.5), Real(0),
        Real(-0.5), Real(0), Real(0.5),
    };
    const std::vector<Real> numerator{
        Real(1), Real(-1), Real(0),
        Real(-1), Real(1), Real(0),
        Real(0), Real(0), Real(0),
    };
    const std::vector<Real> constant_mode{
        Real(1), Real(1), Real(1),
    };

    const auto bound =
        dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            numerator,
            denominator,
            3u,
            constant_mode,
            1u,
            "scalar P1 trace quotient");

    EXPECT_EQ(bound.dimension, 3u);
    EXPECT_EQ(bound.positive_rank, 2u);
    EXPECT_EQ(bound.nullity, 1u);
    EXPECT_TRUE(bound.explicit_nullspace.applied);
    EXPECT_EQ(bound.explicit_nullspace.supplied_nullity, 1u);
    EXPECT_EQ(bound.explicit_nullspace.reduced_dimension, 2u);
    ASSERT_EQ(
        bound.explicit_nullspace.eliminated_coordinates.size(), 1u);
    EXPECT_EQ(
        bound.explicit_nullspace.eliminated_coordinates.front(), 0u);
    EXPECT_EQ(
        bound.explicit_nullspace.maximum_denominator_action, Real(0));
    EXPECT_EQ(
        bound.explicit_nullspace.maximum_numerator_action, Real(0));
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue, Real(2), Real(1.0e-12));
    EXPECT_GE(bound.conservative_upper_bound, Real(2));
    EXPECT_NEAR(
        bound.conservative_upper_bound, Real(2), Real(1.0e-10));
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspaceQuotientsP1RigidModes) {
    auto denominator = p1_triangle_symmetric_gradient_matrix();
    auto numerator = denominator;
    for (auto& value : numerator) {
        value *= Real(3);
    }
    // Component-major coefficients at (0,0), (1,0), (0,1):
    // x translation, y translation, and u=(-y,x).
    const std::vector<Real> rigid_modes{
        Real(1), Real(0), Real(0),
        Real(1), Real(0), Real(0),
        Real(1), Real(0), Real(-1),
        Real(0), Real(1), Real(0),
        Real(0), Real(1), Real(1),
        Real(0), Real(1), Real(0),
    };

    const auto bound =
        dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            numerator,
            denominator,
            6u,
            rigid_modes,
            3u,
            "P1 symmetric-gradient rigid quotient");

    EXPECT_EQ(bound.dimension, 6u);
    EXPECT_EQ(bound.positive_rank, 3u);
    EXPECT_EQ(bound.nullity, 3u);
    EXPECT_EQ(bound.explicit_nullspace.supplied_nullity, 3u);
    EXPECT_EQ(bound.explicit_nullspace.reduced_dimension, 3u);
    EXPECT_EQ(
        bound.explicit_nullspace.eliminated_coordinates.size(), 3u);
    EXPECT_EQ(
        bound.explicit_nullspace.maximum_denominator_action, Real(0));
    EXPECT_EQ(
        bound.explicit_nullspace.maximum_numerator_action, Real(0));
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue, Real(3), Real(1.0e-10));
    EXPECT_GE(bound.conservative_upper_bound, Real(3));
    EXPECT_NEAR(
        bound.conservative_upper_bound, Real(3), Real(1.0e-9));
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspaceHandlesMixedPermutedModes) {
    const std::vector<Real> denominator{
        Real(0), Real(0), Real(0), Real(0),
        Real(0), Real(4), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(1),
    };
    const std::vector<Real> numerator{
        Real(0), Real(0), Real(0), Real(0),
        Real(0), Real(8), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(0),
        Real(0), Real(0), Real(0), Real(3),
    };
    // The common kernel occupies noncontiguous coordinates 0 and 2. Its
    // columns are independently scaled and nonsingularly mixed.
    const std::vector<Real> mixed_scaled_modes{
        Real(2.0e-200), Real(1.0e200),
        Real(0), Real(0),
        Real(1.0e-200), Real(1.0e200),
        Real(0), Real(0),
    };

    const auto bound =
        dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            numerator,
            denominator,
            4u,
            mixed_scaled_modes,
            2u,
            "mixed scaled and permuted nullspace");

    EXPECT_EQ(bound.dimension, 4u);
    EXPECT_EQ(bound.positive_rank, 2u);
    EXPECT_EQ(bound.nullity, 2u);
    ASSERT_EQ(
        bound.explicit_nullspace.eliminated_coordinates.size(), 2u);
    EXPECT_EQ(
        bound.explicit_nullspace.eliminated_coordinates[0], 0u);
    EXPECT_EQ(
        bound.explicit_nullspace.eliminated_coordinates[1], 2u);
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue, Real(3), Real(1.0e-11));
    EXPECT_GE(bound.conservative_upper_bound, Real(3));
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspaceAuditsStructuralRoundoff) {
    const Real small_perturbation =
        Real(32) * std::numeric_limits<Real>::epsilon();
    const Real large_perturbation =
        Real(8192) * std::numeric_limits<Real>::epsilon();
    const std::vector<Real> numerator{
        Real(1), Real(-1), Real(0),
        Real(-1), Real(1), Real(0),
        Real(0), Real(0), Real(0),
    };
    const std::vector<Real> constant_mode{
        Real(1), Real(1), Real(1),
    };
    std::vector<Real> accepted_denominator{
        Real(1), Real(-0.5), Real(-0.5),
        Real(-0.5), Real(0.5), Real(0),
        Real(-0.5), Real(0), Real(0.5),
    };
    accepted_denominator[0] += small_perturbation;

    const auto accepted =
        dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            numerator,
            accepted_denominator,
            3u,
            constant_mode,
            1u,
            "roundoff-sized structural residual");
    EXPECT_GT(
        accepted.explicit_nullspace.maximum_denominator_action,
        Real(0));
    EXPECT_LE(
        accepted.explicit_nullspace.maximum_denominator_action,
        accepted.explicit_nullspace.denominator_action_tolerance);
    EXPECT_GE(accepted.conservative_upper_bound, Real(2));

    auto rejected_denominator = accepted_denominator;
    rejected_denominator[0] =
        Real(1) + large_perturbation;
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            numerator,
            rejected_denominator,
            3u,
            constant_mode,
            1u,
            "oversized structural residual"),
        FEException);
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspaceRejectsInvalidBasisAndAction) {
    const std::vector<Real> zero_pencil(9u, Real(0));
    const std::vector<Real> dependent_modes{
        Real(1), Real(1),
        Real(0), Real(0),
        Real(0), Real(0),
    };
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            zero_pencil,
            zero_pencil,
            3u,
            dependent_modes,
            2u,
            "dependent explicit modes"),
        FEException);

    const std::vector<Real> zero_mode(3u, Real(0));
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            zero_pencil,
            zero_pencil,
            3u,
            zero_mode,
            1u,
            "zero explicit mode"),
        FEException);

    auto nonfinite_mode = std::vector<Real>{
        Real(1), Real(0), Real(0)};
    nonfinite_mode[1] =
        std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            zero_pencil,
            zero_pencil,
            3u,
            nonfinite_mode,
            1u,
            "nonfinite explicit mode"),
        FEException);

    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            zero_pencil,
            zero_pencil,
            3u,
            std::span<const Real>{},
            1u,
            "wrong-sized explicit nullspace"),
        FEException);
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            zero_pencil,
            zero_pencil,
            3u,
            std::span<const Real>{},
            4u,
            "oversized explicit nullity"),
        FEException);

    const std::vector<Real> denominator{
        Real(1), Real(0),
        Real(0), Real(0),
    };
    const std::vector<Real> cross_numerator{
        Real(1), Real(1.0e-8),
        Real(1.0e-8), Real(0),
    };
    const std::vector<Real> null_mode{
        Real(0), Real(1),
    };
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            cross_numerator,
            denominator,
            2u,
            null_mode,
            1u,
            "explicit mode with numerator cross action"),
        FEException);
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspaceValidatesDroppedCoordinates) {
    const std::vector<Real> denominator{
        Real(1), Real(-0.5), Real(-0.5),
        Real(-0.5), Real(0.5), Real(0),
        Real(-0.5), Real(0), Real(0.5),
    };
    std::vector<Real> asymmetric_numerator{
        Real(1), Real(-1), Real(0),
        Real(-1), Real(1), Real(0),
        Real(0), Real(0), Real(0),
    };
    asymmetric_numerator[1] =
        std::nextafter(
            asymmetric_numerator[1],
            std::numeric_limits<Real>::infinity());
    const std::vector<Real> constant_mode{
        Real(1), Real(1), Real(1),
    };

    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            asymmetric_numerator,
            denominator,
            3u,
            constant_mode,
            1u,
            "asymmetry in eliminated coordinate"),
        FEException);

    auto nonfinite_numerator = asymmetric_numerator;
    nonfinite_numerator[1] = Real(-1);
    nonfinite_numerator[0] =
        std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            nonfinite_numerator,
            denominator,
            3u,
            constant_mode,
            1u,
            "nonfinite eliminated coordinate"),
        FEException);
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspacePreservesTinyPositiveFreeMode) {
    const std::vector<Real> denominator{
        Real(0), Real(0), Real(0),
        Real(0), Real(1), Real(0),
        Real(0), Real(0), Real(1.0e-300),
    };
    const std::vector<Real> numerator{
        Real(0), Real(0), Real(0),
        Real(0), Real(1), Real(0),
        Real(0), Real(0), Real(1.0e-14),
    };
    const std::vector<Real> explicit_mode{
        Real(1), Real(0), Real(0),
    };

    const auto bound =
        dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            numerator,
            denominator,
            3u,
            explicit_mode,
            1u,
            "explicit quotient with tiny positive mode");
    const Real analytic_quotient =
        numerator[8] / denominator[8];

    EXPECT_EQ(bound.positive_rank, 2u);
    EXPECT_EQ(bound.nullity, 1u);
    EXPECT_GE(bound.conservative_upper_bound, analytic_quotient);
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundExplicitNullspaceHandlesEmptyAndFullBasis) {
    const std::vector<Real> denominator{
        Real(2), Real(0),
        Real(0), Real(1),
    };
    const std::vector<Real> numerator{
        Real(4), Real(0),
        Real(0), Real(3),
    };
    const auto direct = dense_psd_generalized_eigenvalue_bound(
        numerator, denominator, 2u, "direct no-nullspace pencil");
    const auto empty =
        dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            numerator,
            denominator,
            2u,
            std::span<const Real>{},
            0u,
            "explicit empty-nullspace pencil");
    EXPECT_EQ(
        empty.conservative_upper_bound,
        direct.conservative_upper_bound);
    EXPECT_EQ(empty.dimension, 2u);
    EXPECT_FALSE(empty.explicit_nullspace.applied);
    EXPECT_EQ(empty.explicit_nullspace.supplied_nullity, 0u);
    EXPECT_EQ(empty.explicit_nullspace.reduced_dimension, 2u);

    const std::vector<Real> zero_pencil(4u, Real(0));
    const std::vector<Real> full_nullspace{
        Real(1), Real(0),
        Real(0), Real(1),
    };
    const auto full =
        dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            zero_pencil,
            zero_pencil,
            2u,
            full_nullspace,
            2u,
            "all-null structural pencil");
    EXPECT_EQ(full.dimension, 2u);
    EXPECT_EQ(full.positive_rank, 0u);
    EXPECT_EQ(full.nullity, 2u);
    EXPECT_EQ(full.conservative_upper_bound, Real(0));
    EXPECT_TRUE(full.denominator_converged);
    EXPECT_TRUE(full.quotient_converged);
    EXPECT_EQ(full.explicit_nullspace.reduced_dimension, 0u);
}

TEST(DenseLinearAlgebra,
     PsdGeneralizedBoundRejectsOverflowingDimensionProducts) {
    const std::size_t overflowing_dimension =
        std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound(
            std::span<const Real>{},
            std::span<const Real>{},
            overflowing_dimension,
            "overflowing square pencil"),
        FEException);
    EXPECT_THROW(
        (void)dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
            std::span<const Real>{},
            std::span<const Real>{},
            overflowing_dimension,
            std::span<const Real>{},
            0u,
            "overflowing explicit-nullspace pencil"),
        FEException);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundProvesDiagonalEquality) {
    const std::vector<Real> denominator{
        Real(2), Real(0),
        Real(0), Real(4),
    };
    const std::vector<Real> numerator{
        Real(6), Real(0),
        Real(0), Real(8),
    };

    const auto bound =
        dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            2u,
            "exact diagonal quotient");
    EXPECT_TRUE(bound.applied);
    EXPECT_TRUE(bound.denominator_positive_definite_proven);
    EXPECT_TRUE(bound.numerator_positive_semidefinite_proven);
    EXPECT_TRUE(bound.upper_inequality_proven);
    EXPECT_EQ(bound.denominator_rank, 2u);
    EXPECT_EQ(bound.numerator_rank, 2u);
    EXPECT_TRUE(bound.failing_lower_bound_proven);
    EXPECT_EQ(bound.directly_proven_upper_bound, Real(3));
    EXPECT_EQ(
        bound.largest_failing_lower_bound,
        std::nextafter(Real(3), Real(0)));
    EXPECT_GT(bound.psd_oracle_calls, 2u);
    EXPECT_LE(bound.binary64_search_steps, 64u);
    EXPECT_GT(bound.maximum_integer_bits, 0u);

    const std::vector<Real> zero_numerator(4u, Real(0));
    const auto zero_bound =
        dense_exact_dyadic_spd_generalized_upper_bound(
            zero_numerator,
            denominator,
            2u,
            "zero exact numerator");
    EXPECT_EQ(zero_bound.numerator_rank, 0u);
    EXPECT_FALSE(zero_bound.failing_lower_bound_proven);
    EXPECT_EQ(zero_bound.directly_proven_upper_bound, Real(0));
    EXPECT_TRUE(zero_bound.upper_inequality_proven);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundRetainsTinyPositiveMode) {
    const Real adjacent_to_one =
        std::nextafter(Real(1), Real(0));
    const std::vector<Real> denominator{
        Real(1), adjacent_to_one,
        adjacent_to_one, Real(1),
    };
    const std::vector<Real> numerator{
        Real(2), Real(2) * adjacent_to_one,
        Real(2) * adjacent_to_one, Real(2),
    };

    const auto bound =
        dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            2u,
            "one-ulp exact quotient");
    EXPECT_EQ(bound.denominator_rank, 2u);
    EXPECT_EQ(bound.numerator_rank, 2u);
    EXPECT_EQ(bound.directly_proven_upper_bound, Real(2));
    EXPECT_TRUE(bound.upper_inequality_proven);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundExercisesThreeByThreeBareissDivision) {
    const std::vector<Real> denominator{
        Real(4), Real(1), Real(1),
        Real(1), Real(3), Real(1),
        Real(1), Real(1), Real(2),
    };
    std::vector<Real> numerator = denominator;
    for (auto& value : numerator) {
        value *= Real(2);
    }

    const auto bound =
        dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            3u,
            "three-by-three Bareiss quotient");
    EXPECT_EQ(bound.denominator_rank, 3u);
    EXPECT_EQ(bound.numerator_rank, 3u);
    EXPECT_EQ(bound.directly_proven_upper_bound, Real(2));
    EXPECT_GT(bound.exact_update_count, 0u);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundExercisesSymmetricPsdPivotSwap) {
    const std::vector<Real> denominator{
        Real(1), Real(0), Real(0),
        Real(0), Real(1), Real(0),
        Real(0), Real(0), Real(1),
    };
    const std::vector<Real> numerator{
        Real(0), Real(0), Real(0),
        Real(0), Real(2), Real(1),
        Real(0), Real(1), Real(2),
    };

    const auto bound =
        dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            3u,
            "pivoted semidefinite numerator");
    EXPECT_EQ(bound.denominator_rank, 3u);
    EXPECT_EQ(bound.numerator_rank, 2u);
    EXPECT_EQ(bound.directly_proven_upper_bound, Real(3));
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundRejectsLateIndefinitePivot) {
    const std::vector<Real> denominator{
        Real(1), Real(0), Real(0),
        Real(0), Real(1), Real(0),
        Real(0), Real(0), Real(1),
    };
    const std::vector<Real> numerator{
        Real(2), Real(1), Real(0),
        Real(1), Real(2), Real(2),
        Real(0), Real(2), Real(1),
    };
    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            3u,
            "late indefinite exact numerator"),
        FEException);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundRejectsSemidefiniteDenominator) {
    const std::vector<Real> denominator{
        Real(1), Real(1),
        Real(1), Real(1),
    };
    const std::vector<Real> numerator{
        Real(1), Real(0),
        Real(0), Real(1),
    };
    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            2u,
            "singular exact quotient"),
        FEException);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundRejectsIndefiniteNumerator) {
    const std::vector<Real> denominator{
        Real(1), Real(0),
        Real(0), Real(1),
    };
    const std::vector<Real> numerator{
        Real(1), Real(0),
        Real(0), Real(-1),
    };
    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            2u,
            "indefinite exact numerator"),
        FEException);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundRejectsUnrepresentableUpperBound) {
    const std::vector<Real> denominator{
        std::numeric_limits<Real>::denorm_min(),
    };
    const std::vector<Real> numerator{
        std::numeric_limits<Real>::max(),
    };
    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            numerator,
            denominator,
            1u,
            "unrepresentable exact upper bound"),
        FEException);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundRejectsDimensionAboveCap) {
    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            std::span<const Real>{},
            std::span<const Real>{},
            33u,
            "oversized exact quotient"),
        FEException);
}

TEST(DenseLinearAlgebra,
     ExactDyadicSpdGeneralizedBoundRejectsMalformedInputs) {
    const std::vector<Real> identity{
        Real(1), Real(0),
        Real(0), Real(1),
    };
    const std::vector<Real> asymmetric{
        Real(1), Real(1),
        Real(0), Real(1),
    };
    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            asymmetric,
            identity,
            2u,
            "asymmetric exact numerator"),
        FEException);

    auto nonfinite = identity;
    nonfinite.front() =
        std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            identity,
            nonfinite,
            2u,
            "nonfinite exact denominator"),
        FEException);

    EXPECT_THROW(
        (void)dense_exact_dyadic_spd_generalized_upper_bound(
            std::span<const Real>(identity.data(), 3u),
            identity,
            2u,
            "wrong-sized exact numerator"),
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
