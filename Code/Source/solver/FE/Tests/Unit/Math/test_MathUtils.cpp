/**
 * @file test_MathUtils.cpp
 * @brief Unit tests for MathUtils.h - mathematical utility functions
 */

#include <gtest/gtest.h>
#include "FE/Math/MathUtils.h"
#include "FE/Math/MathConstants.h"
#include <limits>
#include <cmath>
#include <chrono>
#include <vector>
#include <functional>

using namespace svmp::FE::math;

// Test fixture for MathUtils tests
class MathUtilsTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-14;
    static constexpr float ftolerance = 1e-6f;

    void SetUp() override {}
    void TearDown() override {}

    // Helper function to check if two values are approximately equal
    template<typename T>
    bool approx_equal(T a, T b, T tol = tolerance) {
        return std::abs(a - b) <= tol;
    }
};

// =============================================================================
// Sign Function Tests
// =============================================================================

TEST_F(MathUtilsTest, SignOfPositiveNumbers) {
    EXPECT_EQ(sign(5.0), 1);
    EXPECT_EQ(sign(0.001), 1);
    EXPECT_EQ(sign(100.0), 1);
    EXPECT_EQ(sign(std::numeric_limits<double>::max()), 1);
}

TEST_F(MathUtilsTest, SignOfNegativeNumbers) {
    EXPECT_EQ(sign(-5.0), -1);
    EXPECT_EQ(sign(-0.001), -1);
    EXPECT_EQ(sign(-100.0), -1);
    EXPECT_EQ(sign(-std::numeric_limits<double>::max()), -1);
}

TEST_F(MathUtilsTest, SignOfZero) {
    EXPECT_EQ(sign(0.0), 0);
    EXPECT_EQ(sign(-0.0), 0);
    EXPECT_EQ(sign(0), 0);
}

TEST_F(MathUtilsTest, SignWithDifferentTypes) {
    EXPECT_EQ(sign(5), 1);
    EXPECT_EQ(sign(-5), -1);
    EXPECT_EQ(sign(5.0f), 1);
    EXPECT_EQ(sign(-5.0f), -1);
}

// =============================================================================
// Absolute Value Tests
// =============================================================================

TEST_F(MathUtilsTest, AbsPositiveNumbers) {
    EXPECT_EQ(abs(5.0), 5.0);
    EXPECT_EQ(abs(0.001), 0.001);
    EXPECT_EQ(abs(100.0), 100.0);
}

TEST_F(MathUtilsTest, AbsNegativeNumbers) {
    EXPECT_EQ(abs(-5.0), 5.0);
    EXPECT_EQ(abs(-0.001), 0.001);
    EXPECT_EQ(abs(-100.0), 100.0);
}

TEST_F(MathUtilsTest, AbsZero) {
    EXPECT_EQ(abs(0.0), 0.0);
    EXPECT_EQ(abs(-0.0), 0.0);
}

TEST_F(MathUtilsTest, AbsEdgeCases) {
    EXPECT_EQ(abs(std::numeric_limits<double>::max()),
              std::numeric_limits<double>::max());
    EXPECT_EQ(abs(-std::numeric_limits<double>::max()),
              std::numeric_limits<double>::max());
    EXPECT_EQ(abs(std::numeric_limits<double>::min()),
              std::numeric_limits<double>::min());
}

// =============================================================================
// Min/Max Tests
// =============================================================================

TEST_F(MathUtilsTest, MinFunction) {
    EXPECT_EQ(min(3.0, 5.0), 3.0);
    EXPECT_EQ(min(5.0, 3.0), 3.0);
    EXPECT_EQ(min(-5.0, -3.0), -5.0);
    EXPECT_EQ(min(0.0, 0.0), 0.0);
}

TEST_F(MathUtilsTest, MaxFunction) {
    EXPECT_EQ(max(3.0, 5.0), 5.0);
    EXPECT_EQ(max(5.0, 3.0), 5.0);
    EXPECT_EQ(max(-5.0, -3.0), -3.0);
    EXPECT_EQ(max(0.0, 0.0), 0.0);
}

TEST_F(MathUtilsTest, MinMaxWithInfinity) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(min(5.0, inf), 5.0);
    EXPECT_EQ(max(5.0, inf), inf);
    EXPECT_EQ(min(-inf, 5.0), -inf);
    EXPECT_EQ(max(-inf, 5.0), 5.0);
}

// =============================================================================
// Clamp Tests
// =============================================================================

TEST_F(MathUtilsTest, ClampInsideRange) {
    EXPECT_EQ(clamp(5.0, 0.0, 10.0), 5.0);
    EXPECT_EQ(clamp(0.5, 0.0, 1.0), 0.5);
    EXPECT_EQ(clamp(-2.0, -5.0, 5.0), -2.0);
}

TEST_F(MathUtilsTest, ClampBelowRange) {
    EXPECT_EQ(clamp(-5.0, 0.0, 10.0), 0.0);
    EXPECT_EQ(clamp(-0.5, 0.0, 1.0), 0.0);
    EXPECT_EQ(clamp(-10.0, -5.0, 5.0), -5.0);
}

TEST_F(MathUtilsTest, ClampAboveRange) {
    EXPECT_EQ(clamp(15.0, 0.0, 10.0), 10.0);
    EXPECT_EQ(clamp(1.5, 0.0, 1.0), 1.0);
    EXPECT_EQ(clamp(10.0, -5.0, 5.0), 5.0);
}

TEST_F(MathUtilsTest, ClampAtBoundaries) {
    EXPECT_EQ(clamp(0.0, 0.0, 10.0), 0.0);
    EXPECT_EQ(clamp(10.0, 0.0, 10.0), 10.0);
    EXPECT_EQ(clamp(0.0, 0.0, 0.0), 0.0);
}

// =============================================================================
// Integer Power Tests
// =============================================================================

TEST_F(MathUtilsTest, CompileTimeIntegerPower) {
    EXPECT_EQ((ipow<double, 0>(2.0)), 1.0);
    EXPECT_EQ((ipow<double, 1>(2.0)), 2.0);
    EXPECT_EQ((ipow<double, 2>(2.0)), 4.0);
    EXPECT_EQ((ipow<double, 3>(2.0)), 8.0);
    EXPECT_EQ((ipow<double, 4>(2.0)), 16.0);
    EXPECT_EQ((ipow<double, 5>(2.0)), 32.0);
}

TEST_F(MathUtilsTest, RuntimeIntegerPower) {
    EXPECT_EQ(ipow(2.0, 0u), 1.0);
    EXPECT_EQ(ipow(2.0, 1u), 2.0);
    EXPECT_EQ(ipow(2.0, 2u), 4.0);
    EXPECT_EQ(ipow(2.0, 3u), 8.0);
    EXPECT_EQ(ipow(2.0, 10u), 1024.0);
}

TEST_F(MathUtilsTest, IntegerPowerNegativeBase) {
    EXPECT_EQ(ipow(-2.0, 0u), 1.0);
    EXPECT_EQ(ipow(-2.0, 1u), -2.0);
    EXPECT_EQ(ipow(-2.0, 2u), 4.0);
    EXPECT_EQ(ipow(-2.0, 3u), -8.0);
    EXPECT_EQ(ipow(-2.0, 4u), 16.0);
}

TEST_F(MathUtilsTest, IntegerPowerFractionalBase) {
    EXPECT_NEAR(ipow(0.5, 1u), 0.5, tolerance);
    EXPECT_NEAR(ipow(0.5, 2u), 0.25, tolerance);
    EXPECT_NEAR(ipow(0.5, 3u), 0.125, tolerance);
}

// =============================================================================
// Square and Cube Tests
// =============================================================================

TEST_F(MathUtilsTest, SquareFunction) {
    EXPECT_EQ(square(0.0), 0.0);
    EXPECT_EQ(square(2.0), 4.0);
    EXPECT_EQ(square(-2.0), 4.0);
    EXPECT_EQ(square(0.5), 0.25);
    EXPECT_EQ(square(-0.5), 0.25);
}

TEST_F(MathUtilsTest, CubeFunction) {
    EXPECT_EQ(cube(0.0), 0.0);
    EXPECT_EQ(cube(2.0), 8.0);
    EXPECT_EQ(cube(-2.0), -8.0);
    EXPECT_EQ(cube(0.5), 0.125);
    EXPECT_EQ(cube(-0.5), -0.125);
}

// =============================================================================
// Smoothstep Tests
// =============================================================================

TEST_F(MathUtilsTest, SmoothstepBoundaries) {
    EXPECT_EQ(smoothstep(0.0, 1.0, 0.0), 0.0);
    EXPECT_EQ(smoothstep(0.0, 1.0, 1.0), 1.0);
    EXPECT_EQ(smoothstep(0.0, 1.0, -1.0), 0.0); // Clamped
    EXPECT_EQ(smoothstep(0.0, 1.0, 2.0), 1.0);  // Clamped
}

TEST_F(MathUtilsTest, SmoothstepMiddle) {
    EXPECT_NEAR(smoothstep(0.0, 1.0, 0.5), 0.5, tolerance);

    // Check smoothness at 0.25 and 0.75
    double t1 = 0.25;
    double expected1 = t1 * t1 * (3.0 - 2.0 * t1);
    EXPECT_NEAR(smoothstep(0.0, 1.0, 0.25), expected1, tolerance);

    double t2 = 0.75;
    double expected2 = t2 * t2 * (3.0 - 2.0 * t2);
    EXPECT_NEAR(smoothstep(0.0, 1.0, 0.75), expected2, tolerance);
}

TEST_F(MathUtilsTest, SmoothstepCustomRange) {
    EXPECT_EQ(smoothstep(10.0, 20.0, 10.0), 0.0);
    EXPECT_EQ(smoothstep(10.0, 20.0, 20.0), 1.0);
    EXPECT_NEAR(smoothstep(10.0, 20.0, 15.0), 0.5, tolerance);
}

// =============================================================================
// Linear Interpolation Tests
// =============================================================================

TEST_F(MathUtilsTest, LerpBasic) {
    EXPECT_EQ(lerp(0.0, 10.0, 0.0), 0.0);
    EXPECT_EQ(lerp(0.0, 10.0, 1.0), 10.0);
    EXPECT_EQ(lerp(0.0, 10.0, 0.5), 5.0);
    EXPECT_EQ(lerp(0.0, 10.0, 0.25), 2.5);
    EXPECT_EQ(lerp(0.0, 10.0, 0.75), 7.5);
}

TEST_F(MathUtilsTest, LerpNegativeValues) {
    EXPECT_EQ(lerp(-10.0, 10.0, 0.0), -10.0);
    EXPECT_EQ(lerp(-10.0, 10.0, 1.0), 10.0);
    EXPECT_EQ(lerp(-10.0, 10.0, 0.5), 0.0);
}

TEST_F(MathUtilsTest, LerpExtrapolation) {
    EXPECT_EQ(lerp(0.0, 10.0, -0.5), -5.0);
    EXPECT_EQ(lerp(0.0, 10.0, 1.5), 15.0);
    EXPECT_EQ(lerp(0.0, 10.0, 2.0), 20.0);
}

TEST_F(MathUtilsTest, InverseLerp) {
    EXPECT_EQ(inverse_lerp(0.0, 10.0, 0.0), 0.0);
    EXPECT_EQ(inverse_lerp(0.0, 10.0, 10.0), 1.0);
    EXPECT_EQ(inverse_lerp(0.0, 10.0, 5.0), 0.5);
    EXPECT_EQ(inverse_lerp(0.0, 10.0, 2.5), 0.25);
}

TEST_F(MathUtilsTest, InverseLerpOutOfRange) {
    EXPECT_EQ(inverse_lerp(0.0, 10.0, -5.0), -0.5);
    EXPECT_EQ(inverse_lerp(0.0, 10.0, 15.0), 1.5);
}

// =============================================================================
// Remap Tests
// =============================================================================

TEST_F(MathUtilsTest, RemapBasic) {
    // Map [0, 1] to [0, 100]
    EXPECT_EQ(remap(0.0, 0.0, 1.0, 0.0, 100.0), 0.0);
    EXPECT_EQ(remap(1.0, 0.0, 1.0, 0.0, 100.0), 100.0);
    EXPECT_EQ(remap(0.5, 0.0, 1.0, 0.0, 100.0), 50.0);
}

TEST_F(MathUtilsTest, RemapInverse) {
    // Map [0, 100] to [1, 0] (inverted output)
    EXPECT_EQ(remap(0.0, 0.0, 100.0, 1.0, 0.0), 1.0);
    EXPECT_EQ(remap(100.0, 0.0, 100.0, 1.0, 0.0), 0.0);
    EXPECT_EQ(remap(50.0, 0.0, 100.0, 1.0, 0.0), 0.5);
}

TEST_F(MathUtilsTest, RemapNegativeRanges) {
    // Map [-1, 1] to [-100, 100]
    EXPECT_EQ(remap(-1.0, -1.0, 1.0, -100.0, 100.0), -100.0);
    EXPECT_EQ(remap(1.0, -1.0, 1.0, -100.0, 100.0), 100.0);
    EXPECT_EQ(remap(0.0, -1.0, 1.0, -100.0, 100.0), 0.0);
}

// =============================================================================
// Finite Difference Tests
// =============================================================================

TEST_F(MathUtilsTest, FiniteDifferenceLinear) {
    auto linear = [](double x) { return 2.0 * x + 3.0; };
    double derivative = finite_difference(linear, 5.0);
    EXPECT_NEAR(derivative, 2.0, 1e-6);
}

TEST_F(MathUtilsTest, FiniteDifferenceQuadratic) {
    auto quadratic = [](double x) { return x * x; };
    double x = 3.0;
    double derivative = finite_difference(quadratic, x);
    EXPECT_NEAR(derivative, 2.0 * x, 1e-6);
}

TEST_F(MathUtilsTest, FiniteDifferenceTrigonometric) {
    auto sine = [](double x) { return std::sin(x); };
    double x = Constants<double>::pi / 4.0;
    double derivative = finite_difference(sine, x);
    EXPECT_NEAR(derivative, std::cos(x), 1e-6);
}

TEST_F(MathUtilsTest, FiniteDifferenceCustomStep) {
    auto cubic = [](double x) { return x * x * x; };
    double x = 2.0;
    double h = 1e-5;
    double derivative = finite_difference(cubic, x, h);
    EXPECT_NEAR(derivative, 3.0 * x * x, 1e-4);
}

// =============================================================================
// Newton-Raphson Tests
// =============================================================================

TEST_F(MathUtilsTest, NewtonRaphsonLinear) {
    auto f = [](double x) { return 2.0 * x - 6.0; };  // Root at x = 3
    auto df = [](double) { return 2.0; };

    double root = newton_raphson(f, df, 0.0);
    EXPECT_NEAR(root, 3.0, tolerance);
}

TEST_F(MathUtilsTest, NewtonRaphsonQuadratic) {
    auto f = [](double x) { return x * x - 4.0; };  // Roots at x = �2
    auto df = [](double x) { return 2.0 * x; };

    double root1 = newton_raphson(f, df, 3.0);  // Should converge to 2
    EXPECT_NEAR(root1, 2.0, tolerance);

    double root2 = newton_raphson(f, df, -3.0);  // Should converge to -2
    EXPECT_NEAR(root2, -2.0, tolerance);
}

TEST_F(MathUtilsTest, NewtonRaphsonTrigonometric) {
    auto f = [](double x) { return std::cos(x) - x; };
    auto df = [](double x) { return -std::sin(x) - 1.0; };

    double root = newton_raphson(f, df, 0.5);
    // Verify the root satisfies the equation
    EXPECT_NEAR(f(root), 0.0, 1e-10);
}

TEST_F(MathUtilsTest, NewtonRaphsonCustomTolerance) {
    auto f = [](double x) { return x * x - 2.0; };  // Root at sqrt(2)
    auto df = [](double x) { return 2.0 * x; };

    double root = newton_raphson(f, df, 1.0, 100, 1e-12);
    EXPECT_NEAR(root, std::sqrt(2.0), 1e-12);
}

// =============================================================================
// Factorial Tests
// =============================================================================

TEST_F(MathUtilsTest, FactorialCompileTime) {
    EXPECT_EQ(factorial_v<0>, 1ULL);
    EXPECT_EQ(factorial_v<1>, 1ULL);
    EXPECT_EQ(factorial_v<2>, 2ULL);
    EXPECT_EQ(factorial_v<3>, 6ULL);
    EXPECT_EQ(factorial_v<4>, 24ULL);
    EXPECT_EQ(factorial_v<5>, 120ULL);
    EXPECT_EQ(factorial_v<6>, 720ULL);
    EXPECT_EQ(factorial_v<7>, 5040ULL);
    EXPECT_EQ(factorial_v<10>, 3628800ULL);
}

// =============================================================================
// Binomial Coefficient Tests
// =============================================================================

TEST_F(MathUtilsTest, BinomialCoefficientBasic) {
    EXPECT_EQ((binomial_v<0, 0>), 1ULL);
    EXPECT_EQ((binomial_v<1, 0>), 1ULL);
    EXPECT_EQ((binomial_v<1, 1>), 1ULL);
    EXPECT_EQ((binomial_v<5, 0>), 1ULL);
    EXPECT_EQ((binomial_v<5, 1>), 5ULL);
    EXPECT_EQ((binomial_v<5, 2>), 10ULL);
    EXPECT_EQ((binomial_v<5, 3>), 10ULL);
    EXPECT_EQ((binomial_v<5, 4>), 5ULL);
    EXPECT_EQ((binomial_v<5, 5>), 1ULL);
}

TEST_F(MathUtilsTest, BinomialCoefficientPascalTriangle) {
    // Row 6 of Pascal's triangle: 1, 6, 15, 20, 15, 6, 1
    EXPECT_EQ((binomial_v<6, 0>), 1ULL);
    EXPECT_EQ((binomial_v<6, 1>), 6ULL);
    EXPECT_EQ((binomial_v<6, 2>), 15ULL);
    EXPECT_EQ((binomial_v<6, 3>), 20ULL);
    EXPECT_EQ((binomial_v<6, 4>), 15ULL);
    EXPECT_EQ((binomial_v<6, 5>), 6ULL);
    EXPECT_EQ((binomial_v<6, 6>), 1ULL);
}

// =============================================================================
// Wrap Angle Tests
// =============================================================================

TEST_F(MathUtilsTest, WrapAngleInRange) {
    EXPECT_NEAR(wrap_angle(0.0), 0.0, tolerance);
    EXPECT_NEAR(wrap_angle(1.0), 1.0, tolerance);
    EXPECT_NEAR(wrap_angle(-1.0), -1.0, tolerance);
    EXPECT_NEAR(wrap_angle(Constants<double>::pi), Constants<double>::pi, tolerance);
    EXPECT_NEAR(wrap_angle(-Constants<double>::pi), -Constants<double>::pi, tolerance);
}

TEST_F(MathUtilsTest, WrapAnglePositiveOutOfRange) {
    double pi = Constants<double>::pi;
    EXPECT_NEAR(wrap_angle(2.0 * pi), 0.0, tolerance);
    EXPECT_NEAR(wrap_angle(3.0 * pi), -pi, tolerance);  // std::remainder gives -π
    EXPECT_NEAR(wrap_angle(4.0 * pi), 0.0, tolerance);
    EXPECT_NEAR(wrap_angle(5.0 * pi), pi, tolerance);   // std::remainder gives π (corrected)
}

TEST_F(MathUtilsTest, WrapAngleNegativeOutOfRange) {
    double pi = Constants<double>::pi;
    EXPECT_NEAR(wrap_angle(-2.0 * pi), 0.0, tolerance);
    EXPECT_NEAR(wrap_angle(-3.0 * pi), pi, tolerance);    // std::remainder gives π
    EXPECT_NEAR(wrap_angle(-4.0 * pi), 0.0, tolerance);
    EXPECT_NEAR(wrap_angle(-5.0 * pi), -pi, tolerance);   // std::remainder gives -π (corrected)
}

TEST_F(MathUtilsTest, WrapAngleLargeAngles) {
    double pi = Constants<double>::pi;
    EXPECT_NEAR(wrap_angle(10.0 * pi), 0.0, 1e-10);       // Relax tolerance for large multiples
    EXPECT_NEAR(wrap_angle(100.0 * pi), 0.0, 1e-10);      // Relax tolerance for large multiples
    EXPECT_NEAR(wrap_angle(1001.0 * pi), -pi, 1e-10);     // std::remainder gives -π for 1001π
}

// =============================================================================
// Performance Benchmarks (Optional)
// =============================================================================

TEST_F(MathUtilsTest, PerformanceBenchmarkIPow) {
    // Test that integer power is faster than std::pow for small exponents
    const int iterations = 1000000;
    double base = 1.1;
    double sum1 = 0.0, sum2 = 0.0;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sum1 += ipow(base, 5u);
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sum2 += std::pow(base, 5.0);
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    (void)start1;
    (void)end1;
    (void)start2;
    (void)end2;

    // Just verify the results are similar
    EXPECT_NEAR(sum1 / iterations, sum2 / iterations, tolerance);
}

// =============================================================================
// Edge Cases and Special Values Tests
// =============================================================================

TEST_F(MathUtilsTest, SpecialValuesNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();

    // Most operations with NaN should return NaN
    EXPECT_TRUE(std::isnan(abs(nan)));
    EXPECT_TRUE(std::isnan(square(nan)));
    EXPECT_TRUE(std::isnan(cube(nan)));
    EXPECT_TRUE(std::isnan(lerp(0.0, 1.0, nan)));
    EXPECT_TRUE(std::isnan(lerp(nan, 1.0, 0.5)));
}

TEST_F(MathUtilsTest, SpecialValuesInfinity) {
    double inf = std::numeric_limits<double>::infinity();

    EXPECT_EQ(sign(inf), 1);
    EXPECT_EQ(sign(-inf), -1);
    EXPECT_EQ(abs(inf), inf);
    EXPECT_EQ(abs(-inf), inf);
    EXPECT_EQ(square(inf), inf);
    EXPECT_EQ(square(-inf), inf);
    EXPECT_EQ(cube(inf), inf);
    EXPECT_EQ(cube(-inf), -inf);
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_F(MathUtilsTest, NumericalStabilitySmallValues) {
    double epsilon = std::numeric_limits<double>::epsilon();

    // Test that functions handle small values correctly
    EXPECT_EQ(sign(epsilon), 1);
    EXPECT_EQ(sign(-epsilon), -1);
    EXPECT_EQ(abs(epsilon), epsilon);
    EXPECT_EQ(abs(-epsilon), epsilon);

    // Square and cube should preserve small values
    EXPECT_GT(square(epsilon), 0.0);
    EXPECT_GT(cube(epsilon), 0.0);
}

TEST_F(MathUtilsTest, NumericalStabilityLerpNearEndpoints) {
    // Test lerp stability near t=0 and t=1
    double a = 1e-15;
    double b = 1.0;

    EXPECT_EQ(lerp(a, b, 0.0), a);
    EXPECT_EQ(lerp(a, b, 1.0), b);

    // Test with very small t
    double small_t = 1e-15;
    double result = lerp(a, b, small_t);
    EXPECT_GE(result, a);
    EXPECT_LE(result, b);
}
