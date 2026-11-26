/**
 * @file test_DGOperators.cpp
 * @brief Unit tests for DGOperators
 */

#include <gtest/gtest.h>
#include "FE/Spaces/DGOperators.h"
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::spaces;
using Vec3 = DGOperators::Vec3;
using Vec2 = DGOperators::Vec2;

class DGOperatorsTest : public ::testing::Test {
protected:
    static constexpr Real tol = 1e-12;
};

// =============================================================================
// Scalar Jump and Average Tests
// =============================================================================

TEST_F(DGOperatorsTest, ScalarJumpContinuousField) {
    // If field is continuous, jump should be zero
    std::vector<Real> plus = {1.0, 2.0, 3.0};
    std::vector<Real> minus = {1.0, 2.0, 3.0};

    auto result = DGOperators::jump(plus, minus);

    ASSERT_EQ(result.size(), 3u);
    for (auto& v : result) {
        EXPECT_NEAR(v, 0.0, tol);
    }
}

TEST_F(DGOperatorsTest, ScalarJumpDiscontinuous) {
    std::vector<Real> plus = {1.0, 2.0};
    std::vector<Real> minus = {3.0, 5.0};

    auto result = DGOperators::jump(plus, minus);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], -2.0, tol);  // 1 - 3
    EXPECT_NEAR(result[1], -3.0, tol);  // 2 - 5
}

TEST_F(DGOperatorsTest, ScalarAverageSymmetric) {
    std::vector<Real> plus = {0.0, 4.0};
    std::vector<Real> minus = {2.0, 6.0};

    auto result = DGOperators::average(plus, minus);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 1.0, tol);   // (0 + 2) / 2
    EXPECT_NEAR(result[1], 5.0, tol);   // (4 + 6) / 2
}

TEST_F(DGOperatorsTest, WeightedAverage) {
    std::vector<Real> plus = {10.0};
    std::vector<Real> minus = {0.0};

    auto result = DGOperators::weighted_average(plus, minus, 0.75, 0.25);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 7.5, tol);  // 0.75 * 10 + 0.25 * 0
}

// =============================================================================
// Vector Jump and Average Tests
// =============================================================================

TEST_F(DGOperatorsTest, VectorJump3D) {
    std::vector<Vec3> plus = {Vec3{1.0, 2.0, 3.0}};
    std::vector<Vec3> minus = {Vec3{0.0, 1.0, 1.0}};

    auto result = DGOperators::jump_vector(plus, minus);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0][0], 1.0, tol);
    EXPECT_NEAR(result[0][1], 1.0, tol);
    EXPECT_NEAR(result[0][2], 2.0, tol);
}

TEST_F(DGOperatorsTest, VectorAverage3D) {
    std::vector<Vec3> plus = {Vec3{2.0, 4.0, 6.0}};
    std::vector<Vec3> minus = {Vec3{0.0, 2.0, 2.0}};

    auto result = DGOperators::average_vector(plus, minus);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0][0], 1.0, tol);
    EXPECT_NEAR(result[0][1], 3.0, tol);
    EXPECT_NEAR(result[0][2], 4.0, tol);
}

TEST_F(DGOperatorsTest, VectorJump2D) {
    std::vector<Vec2> plus = {Vec2{1.0, 2.0}};
    std::vector<Vec2> minus = {Vec2{0.5, 1.0}};

    auto result = DGOperators::jump_vector_2d(plus, minus);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0][0], 0.5, tol);
    EXPECT_NEAR(result[0][1], 1.0, tol);
}

// =============================================================================
// H(div) and H(curl) Tests
// =============================================================================

TEST_F(DGOperatorsTest, NormalJumpHdivConforming) {
    // For H(div) conforming fields, normal jump should be zero
    Vec3 n{0.0, 0.0, 1.0};
    std::vector<Vec3> plus = {Vec3{1.0, 2.0, 5.0}};
    std::vector<Vec3> minus = {Vec3{3.0, 4.0, 5.0}};  // Same normal component

    auto result = DGOperators::normal_jump(plus, minus, n);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 0.0, tol);  // (5 - 5) = 0
}

TEST_F(DGOperatorsTest, NormalJumpNonConforming) {
    Vec3 n{0.0, 0.0, 1.0};
    std::vector<Vec3> plus = {Vec3{1.0, 2.0, 5.0}};
    std::vector<Vec3> minus = {Vec3{1.0, 2.0, 3.0}};

    auto result = DGOperators::normal_jump(plus, minus, n);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 2.0, tol);  // (5 - 3) = 2
}

TEST_F(DGOperatorsTest, NormalAverage) {
    Vec3 n{0.0, 0.0, 1.0};
    std::vector<Vec3> plus = {Vec3{0.0, 0.0, 4.0}};
    std::vector<Vec3> minus = {Vec3{0.0, 0.0, 2.0}};

    auto result = DGOperators::normal_average(plus, minus, n);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 3.0, tol);  // 0.5 * (4 + 2)
}

TEST_F(DGOperatorsTest, TangentialJumpHcurlConforming) {
    // For H(curl) conforming fields, tangential jump should be zero
    Vec3 n{0.0, 0.0, 1.0};
    std::vector<Vec3> plus = {Vec3{1.0, 2.0, 5.0}};
    std::vector<Vec3> minus = {Vec3{1.0, 2.0, 3.0}};  // Same tangential components

    auto result = DGOperators::tangential_jump(plus, minus, n);

    ASSERT_EQ(result.size(), 1u);
    // [v × n] = (v+ - v-) × n = (0, 0, 2) × (0, 0, 1) = (0, 0, 0)
    EXPECT_NEAR(result[0].norm(), 0.0, tol);
}

TEST_F(DGOperatorsTest, TangentialJumpNonConforming) {
    Vec3 n{0.0, 0.0, 1.0};
    std::vector<Vec3> plus = {Vec3{2.0, 0.0, 0.0}};
    std::vector<Vec3> minus = {Vec3{0.0, 0.0, 0.0}};

    auto result = DGOperators::tangential_jump(plus, minus, n);

    ASSERT_EQ(result.size(), 1u);
    // [v × n] = (2, 0, 0) × (0, 0, 1) = (0, -2, 0)
    EXPECT_NEAR(result[0][1], -2.0, tol);
}

// =============================================================================
// Gradient Operators
// =============================================================================

TEST_F(DGOperatorsTest, GradientNormalJump) {
    Vec3 n{1.0, 0.0, 0.0};
    std::vector<Vec3> grad_plus = {Vec3{3.0, 1.0, 2.0}};
    std::vector<Vec3> grad_minus = {Vec3{1.0, 1.0, 2.0}};

    auto result = DGOperators::gradient_normal_jump(grad_plus, grad_minus, n);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 2.0, tol);  // (3 - 1) · (1, 0, 0) = 2
}

TEST_F(DGOperatorsTest, GradientNormalAverage) {
    Vec3 n{1.0, 0.0, 0.0};
    std::vector<Vec3> grad_plus = {Vec3{4.0, 0.0, 0.0}};
    std::vector<Vec3> grad_minus = {Vec3{2.0, 0.0, 0.0}};

    auto result = DGOperators::gradient_normal_average(grad_plus, grad_minus, n);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 3.0, tol);  // 0.5 * (4 + 2)
}

// =============================================================================
// Penalty and Upwind Tests
// =============================================================================

TEST_F(DGOperatorsTest, PenaltyParameter) {
    Real eta = DGOperators::penalty_parameter(2, 0.1, 1.0);
    // η = C * (p+1)² / h = 1 * 9 / 0.1 = 90
    EXPECT_NEAR(eta, 90.0, tol);
}

TEST_F(DGOperatorsTest, HarmonicAverage) {
    Real avg = DGOperators::harmonic_average(2.0, 8.0);
    // 2 * 2 * 8 / (2 + 8) = 32 / 10 = 3.2
    EXPECT_NEAR(avg, 3.2, tol);
}

TEST_F(DGOperatorsTest, HarmonicAverageEqual) {
    Real avg = DGOperators::harmonic_average(4.0, 4.0);
    // 2 * 4 * 4 / (4 + 4) = 32 / 8 = 4
    EXPECT_NEAR(avg, 4.0, tol);
}

TEST_F(DGOperatorsTest, UpwindPositiveVelocity) {
    std::vector<Real> plus = {10.0};
    std::vector<Real> minus = {5.0};
    Vec3 velocity{1.0, 0.0, 0.0};
    Vec3 normal{1.0, 0.0, 0.0};

    auto result = DGOperators::upwind(plus, minus, velocity, normal);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 10.0, tol);  // Upwind from plus side
}

TEST_F(DGOperatorsTest, UpwindNegativeVelocity) {
    std::vector<Real> plus = {10.0};
    std::vector<Real> minus = {5.0};
    Vec3 velocity{-1.0, 0.0, 0.0};
    Vec3 normal{1.0, 0.0, 0.0};

    auto result = DGOperators::upwind(plus, minus, velocity, normal);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 5.0, tol);  // Upwind from minus side
}

TEST_F(DGOperatorsTest, UpwindParallelVelocity) {
    std::vector<Real> plus = {10.0};
    std::vector<Real> minus = {6.0};
    Vec3 velocity{0.0, 1.0, 0.0};
    Vec3 normal{1.0, 0.0, 0.0};

    auto result = DGOperators::upwind(plus, minus, velocity, normal);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 8.0, tol);  // Average when parallel
}

TEST_F(DGOperatorsTest, LaxFriedrichsFluxIsUpwindForLinearAdvection) {
    const std::vector<Real> plus = {2.0, -3.0};
    const std::vector<Real> minus = {1.0, 4.0};
    for (const Real normal_speed : {Real(2), Real(-2), Real(0)}) {
        const std::vector<Real> flux_plus = {normal_speed * plus[0], normal_speed * plus[1]};
        const std::vector<Real> flux_minus = {normal_speed * minus[0], normal_speed * minus[1]};
        const auto result = DGOperators::lax_friedrichs_flux(
            plus, minus, flux_plus, flux_minus, std::abs(normal_speed));
        ASSERT_EQ(result.size(), plus.size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            const Real upstream = normal_speed > 0 ? plus[i] : minus[i];
            EXPECT_DOUBLE_EQ(result[i], normal_speed * upstream);
        }
    }
}
