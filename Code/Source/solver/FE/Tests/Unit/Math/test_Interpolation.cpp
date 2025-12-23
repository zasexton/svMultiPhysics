/**
 * @file test_Interpolation.cpp
 * @brief Unit tests for Interpolation.h - interpolation utilities
 */

#include <gtest/gtest.h>
#include "FE/Math/Interpolation.h"
#include "FE/Math/Vector.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/MathConstants.h"
#include <limits>
#include <cmath>
#include <array>
#include <vector>
#include <random>

using namespace svmp::FE::math;

// Test fixture for Interpolation tests
class InterpolationTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-14;
    static constexpr double loose_tolerance = 1e-10;

    void SetUp() override {
        rng.seed(42);  // Fixed seed for reproducibility
    }

    void TearDown() override {}

    // Helper to check if vectors are approximately equal
    template<typename T, std::size_t N>
    bool vector_approx_equal(const Vector<T, N>& a, const Vector<T, N>& b, T tol = tolerance) {
        for (std::size_t i = 0; i < N; ++i) {
            if (std::abs(a[i] - b[i]) > tol) {
                return false;
            }
        }
        return true;
    }

    std::mt19937 rng;
};

// =============================================================================
// Barycentric Coordinates Tests
// =============================================================================

TEST_F(InterpolationTest, BarycentricCoords2D_EquilateralTriangle) {
    // Equilateral triangle vertices
    Vector2<double> v0{0.0, 0.0};
    Vector2<double> v1{1.0, 0.0};
    Vector2<double> v2{0.5, std::sqrt(3.0) / 2.0};

    // Test at vertices
    auto bary0 = BarycentricCoords<double>::compute_2d(v0, v0, v1, v2);
    EXPECT_NEAR(bary0[0], 1.0, tolerance);
    EXPECT_NEAR(bary0[1], 0.0, tolerance);
    EXPECT_NEAR(bary0[2], 0.0, tolerance);

    auto bary1 = BarycentricCoords<double>::compute_2d(v1, v0, v1, v2);
    EXPECT_NEAR(bary1[0], 0.0, tolerance);
    EXPECT_NEAR(bary1[1], 1.0, tolerance);
    EXPECT_NEAR(bary1[2], 0.0, tolerance);

    auto bary2 = BarycentricCoords<double>::compute_2d(v2, v0, v1, v2);
    EXPECT_NEAR(bary2[0], 0.0, tolerance);
    EXPECT_NEAR(bary2[1], 0.0, tolerance);
    EXPECT_NEAR(bary2[2], 1.0, tolerance);

    // Test at centroid
    Vector2<double> centroid = (v0 + v1 + v2) / 3.0;
    auto bary_centroid = BarycentricCoords<double>::compute_2d(centroid, v0, v1, v2);
    EXPECT_NEAR(bary_centroid[0], 1.0 / 3.0, tolerance);
    EXPECT_NEAR(bary_centroid[1], 1.0 / 3.0, tolerance);
    EXPECT_NEAR(bary_centroid[2], 1.0 / 3.0, tolerance);
}

TEST_F(InterpolationTest, BarycentricCoords2D_RightTriangle) {
    // Right triangle vertices
    Vector2<double> v0{0.0, 0.0};
    Vector2<double> v1{1.0, 0.0};
    Vector2<double> v2{0.0, 1.0};

    // Test at midpoint of hypotenuse
    Vector2<double> midpoint{0.5, 0.5};
    auto bary = BarycentricCoords<double>::compute_2d(midpoint, v0, v1, v2);
    EXPECT_NEAR(bary[0], 0.0, tolerance);
    EXPECT_NEAR(bary[1], 0.5, tolerance);
    EXPECT_NEAR(bary[2], 0.5, tolerance);

    // Verify that barycentric coordinates sum to 1
    EXPECT_NEAR(bary[0] + bary[1] + bary[2], 1.0, tolerance);
}

TEST_F(InterpolationTest, BarycentricCoords2D_OutsideTriangle) {
    Vector2<double> v0{0.0, 0.0};
    Vector2<double> v1{1.0, 0.0};
    Vector2<double> v2{0.0, 1.0};

    // Point outside triangle
    Vector2<double> outside{2.0, 2.0};
    auto bary = BarycentricCoords<double>::compute_2d(outside, v0, v1, v2);

    // Should still sum to 1, but some coordinates will be negative
    EXPECT_NEAR(bary[0] + bary[1] + bary[2], 1.0, tolerance);
    EXPECT_FALSE(BarycentricCoords<double>::is_inside(bary));
}

TEST_F(InterpolationTest, BarycentricCoords2D_DegenerateTriangle) {
    // Degenerate triangle (collinear points)
    Vector2<double> v0{0.0, 0.0};
    Vector2<double> v1{1.0, 0.0};
    Vector2<double> v2{2.0, 0.0};

    Vector2<double> p{0.5, 0.0};
    auto bary = BarycentricCoords<double>::compute_2d(p, v0, v1, v2);

    // Should return reasonable values even for degenerate case
    EXPECT_NEAR(bary[0] + bary[1] + bary[2], 1.0, tolerance);
}

TEST_F(InterpolationTest, BarycentricCoords3D_RegularTetrahedron) {
    // Regular tetrahedron vertices
    Vector3<double> v0{0.0, 0.0, 0.0};
    Vector3<double> v1{1.0, 0.0, 0.0};
    Vector3<double> v2{0.5, std::sqrt(3.0) / 2.0, 0.0};
    Vector3<double> v3{0.5, std::sqrt(3.0) / 6.0, std::sqrt(2.0 / 3.0)};

    // Test at vertices
    auto bary0 = BarycentricCoords<double>::compute_3d(v0, v0, v1, v2, v3);
    EXPECT_NEAR(bary0[0], 1.0, tolerance);
    EXPECT_NEAR(bary0[1], 0.0, tolerance);
    EXPECT_NEAR(bary0[2], 0.0, tolerance);
    EXPECT_NEAR(bary0[3], 0.0, tolerance);

    // Test at centroid
    Vector3<double> centroid = (v0 + v1 + v2 + v3) / 4.0;
    auto bary_centroid = BarycentricCoords<double>::compute_3d(centroid, v0, v1, v2, v3);
    EXPECT_NEAR(bary_centroid[0], 0.25, loose_tolerance);
    EXPECT_NEAR(bary_centroid[1], 0.25, loose_tolerance);
    EXPECT_NEAR(bary_centroid[2], 0.25, loose_tolerance);
    EXPECT_NEAR(bary_centroid[3], 0.25, loose_tolerance);
}

TEST_F(InterpolationTest, BarycentricInterpolation) {
    // Triangle vertices
    Vector2<double> v0{0.0, 0.0};
    Vector2<double> v1{1.0, 0.0};
    Vector2<double> v2{0.0, 1.0};

    // Values at vertices
    std::array<double, 3> values{1.0, 2.0, 3.0};

    // Test at vertices
    Vector3<double> bary0{1.0, 0.0, 0.0};
    double interp0 = BarycentricCoords<double>::interpolate(bary0, values);
    EXPECT_NEAR(interp0, values[0], tolerance);

    // Test at centroid
    Vector3<double> bary_centroid{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    double interp_centroid = BarycentricCoords<double>::interpolate(bary_centroid, values);
    EXPECT_NEAR(interp_centroid, 2.0, tolerance);  // Average of 1, 2, 3
}

TEST_F(InterpolationTest, BarycentricIsInside) {
    // Valid barycentric coordinates (inside)
    Vector3<double> bary_inside{0.3, 0.4, 0.3};
    EXPECT_TRUE(BarycentricCoords<double>::is_inside(bary_inside));

    // On boundary
    Vector3<double> bary_boundary{0.5, 0.5, 0.0};
    EXPECT_TRUE(BarycentricCoords<double>::is_inside(bary_boundary));

    // Outside
    Vector3<double> bary_outside{0.5, 0.6, -0.1};
    EXPECT_FALSE(BarycentricCoords<double>::is_inside(bary_outside));

    // With tolerance
    Vector3<double> bary_near_boundary{0.5, 0.5, -1e-10};
    EXPECT_TRUE(BarycentricCoords<double>::is_inside(bary_near_boundary, 1e-9));
}

// =============================================================================
// Basic Interpolation Functions Tests
// =============================================================================

TEST_F(InterpolationTest, Lerp_Basic) {
    // Scalar interpolation
    double a = 10.0;
    double b = 20.0;

    EXPECT_EQ(lerp(a, b, 0.0), a);
    EXPECT_EQ(lerp(a, b, 1.0), b);
    EXPECT_EQ(lerp(a, b, 0.5), 15.0);
    EXPECT_EQ(lerp(a, b, 0.25), 12.5);
    EXPECT_EQ(lerp(a, b, 0.75), 17.5);

    // Extrapolation
    EXPECT_EQ(lerp(a, b, -0.5), 5.0);
    EXPECT_EQ(lerp(a, b, 1.5), 25.0);
}

TEST_F(InterpolationTest, Lerp_Vector) {
    Vector3<double> v1{1.0, 2.0, 3.0};
    Vector3<double> v2{4.0, 5.0, 6.0};

    auto v_mid = lerp(v1, v2, 0.5);
    EXPECT_NEAR(v_mid[0], 2.5, tolerance);
    EXPECT_NEAR(v_mid[1], 3.5, tolerance);
    EXPECT_NEAR(v_mid[2], 4.5, tolerance);
}

TEST_F(InterpolationTest, Smoothstep) {
    // Basic smoothstep
    EXPECT_EQ(smoothstep(0.0), 0.0);
    EXPECT_EQ(smoothstep(1.0), 1.0);
    EXPECT_NEAR(smoothstep(0.5), 0.5, tolerance);

    // Check smooth derivative at endpoints
    double epsilon = 1e-8;
    double deriv_at_0 = (smoothstep(epsilon) - smoothstep(0.0)) / epsilon;
    double deriv_at_1 = (smoothstep(1.0) - smoothstep(1.0 - epsilon)) / epsilon;
    EXPECT_NEAR(deriv_at_0, 0.0, 1e-6);
    EXPECT_NEAR(deriv_at_1, 0.0, 1e-6);

    // Clamping
    EXPECT_EQ(smoothstep(-0.5), 0.0);
    EXPECT_EQ(smoothstep(1.5), 1.0);
}

TEST_F(InterpolationTest, Smootherstep) {
    // Basic smootherstep
    EXPECT_EQ(smootherstep(0.0), 0.0);
    EXPECT_EQ(smootherstep(1.0), 1.0);
    EXPECT_NEAR(smootherstep(0.5), 0.5, tolerance);

    // Check second derivative at endpoints is zero
    double h = 1e-8;
    double second_deriv_at_0 = (smootherstep(h) - 2 * smootherstep(0.0) + smootherstep(-h)) / (h * h);
    double second_deriv_at_1 = (smootherstep(1.0 + h) - 2 * smootherstep(1.0) + smootherstep(1.0 - h)) / (h * h);
    EXPECT_NEAR(second_deriv_at_0, 0.0, 1e-4);
    EXPECT_NEAR(second_deriv_at_1, 0.0, 1e-4);
}

TEST_F(InterpolationTest, Bilinear) {
    // Unit square interpolation
    double f00 = 1.0;  // Value at (0, 0)
    double f10 = 2.0;  // Value at (1, 0)
    double f01 = 3.0;  // Value at (0, 1)
    double f11 = 4.0;  // Value at (1, 1)

    // Test at corners
    EXPECT_EQ(bilinear(0.0, 0.0, f00, f10, f01, f11), f00);
    EXPECT_EQ(bilinear(1.0, 0.0, f00, f10, f01, f11), f10);
    EXPECT_EQ(bilinear(0.0, 1.0, f00, f10, f01, f11), f01);
    EXPECT_EQ(bilinear(1.0, 1.0, f00, f10, f01, f11), f11);

    // Test at center
    EXPECT_EQ(bilinear(0.5, 0.5, f00, f10, f01, f11), 2.5);

    // Test along edges
    EXPECT_EQ(bilinear(0.5, 0.0, f00, f10, f01, f11), 1.5);  // Bottom edge
    EXPECT_EQ(bilinear(0.5, 1.0, f00, f10, f01, f11), 3.5);  // Top edge
    EXPECT_EQ(bilinear(0.0, 0.5, f00, f10, f01, f11), 2.0);  // Left edge
    EXPECT_EQ(bilinear(1.0, 0.5, f00, f10, f01, f11), 3.0);  // Right edge
}

TEST_F(InterpolationTest, Trilinear) {
    // Unit cube interpolation
    double f000 = 1.0, f100 = 2.0, f010 = 3.0, f110 = 4.0;
    double f001 = 5.0, f101 = 6.0, f011 = 7.0, f111 = 8.0;

    // Test at corners
    EXPECT_EQ(trilinear(0.0, 0.0, 0.0, f000, f100, f010, f110, f001, f101, f011, f111), f000);
    EXPECT_EQ(trilinear(1.0, 0.0, 0.0, f000, f100, f010, f110, f001, f101, f011, f111), f100);
    EXPECT_EQ(trilinear(0.0, 1.0, 0.0, f000, f100, f010, f110, f001, f101, f011, f111), f010);
    EXPECT_EQ(trilinear(1.0, 1.0, 1.0, f000, f100, f010, f110, f001, f101, f011, f111), f111);

    // Test at center
    double center = trilinear(0.5, 0.5, 0.5, f000, f100, f010, f110, f001, f101, f011, f111);
    EXPECT_EQ(center, 4.5);  // Average of all 8 values

    // Test along an edge
    double edge = trilinear(0.5, 0.0, 0.0, f000, f100, f010, f110, f001, f101, f011, f111);
    EXPECT_EQ(edge, 1.5);  // Average of f000 and f100
}

// =============================================================================
// Shape Function Helpers Tests
// =============================================================================

TEST_F(InterpolationTest, ShapeFunctions_Linear1D) {
    // Test at left node
    auto N_left = ShapeFunctionHelpers<double>::linear_1d(-1.0);
    EXPECT_NEAR(N_left[0], 1.0, tolerance);
    EXPECT_NEAR(N_left[1], 0.0, tolerance);

    // Test at right node
    auto N_right = ShapeFunctionHelpers<double>::linear_1d(1.0);
    EXPECT_NEAR(N_right[0], 0.0, tolerance);
    EXPECT_NEAR(N_right[1], 1.0, tolerance);

    // Test at center
    auto N_center = ShapeFunctionHelpers<double>::linear_1d(0.0);
    EXPECT_NEAR(N_center[0], 0.5, tolerance);
    EXPECT_NEAR(N_center[1], 0.5, tolerance);

    // Verify partition of unity
    for (double xi = -1.0; xi <= 1.0; xi += 0.1) {
        auto N = ShapeFunctionHelpers<double>::linear_1d(xi);
        EXPECT_NEAR(N[0] + N[1], 1.0, tolerance);
    }
}

TEST_F(InterpolationTest, ShapeFunctions_Quadratic1D) {
    // Test at nodes
    auto N_left = ShapeFunctionHelpers<double>::quadratic_1d(-1.0);
    EXPECT_NEAR(N_left[0], 1.0, tolerance);
    EXPECT_NEAR(N_left[1], 0.0, tolerance);
    EXPECT_NEAR(N_left[2], 0.0, tolerance);

    auto N_mid = ShapeFunctionHelpers<double>::quadratic_1d(0.0);
    EXPECT_NEAR(N_mid[0], 0.0, tolerance);
    EXPECT_NEAR(N_mid[1], 1.0, tolerance);
    EXPECT_NEAR(N_mid[2], 0.0, tolerance);

    auto N_right = ShapeFunctionHelpers<double>::quadratic_1d(1.0);
    EXPECT_NEAR(N_right[0], 0.0, tolerance);
    EXPECT_NEAR(N_right[1], 0.0, tolerance);
    EXPECT_NEAR(N_right[2], 1.0, tolerance);

    // Verify partition of unity
    for (double xi = -1.0; xi <= 1.0; xi += 0.1) {
        auto N = ShapeFunctionHelpers<double>::quadratic_1d(xi);
        EXPECT_NEAR(N[0] + N[1] + N[2], 1.0, tolerance);
    }
}

TEST_F(InterpolationTest, ShapeFunctions_Bilinear2D) {
    // Test at corners
    auto N_00 = ShapeFunctionHelpers<double>::bilinear_2d(-1.0, -1.0);
    EXPECT_NEAR(N_00[0], 1.0, tolerance);
    EXPECT_NEAR(N_00[1], 0.0, tolerance);
    EXPECT_NEAR(N_00[2], 0.0, tolerance);
    EXPECT_NEAR(N_00[3], 0.0, tolerance);

    // Test at center
    auto N_center = ShapeFunctionHelpers<double>::bilinear_2d(0.0, 0.0);
    EXPECT_NEAR(N_center[0], 0.25, tolerance);
    EXPECT_NEAR(N_center[1], 0.25, tolerance);
    EXPECT_NEAR(N_center[2], 0.25, tolerance);
    EXPECT_NEAR(N_center[3], 0.25, tolerance);

    // Verify partition of unity
    for (double xi = -1.0; xi <= 1.0; xi += 0.5) {
        for (double eta = -1.0; eta <= 1.0; eta += 0.5) {
            auto N = ShapeFunctionHelpers<double>::bilinear_2d(xi, eta);
            double sum = N[0] + N[1] + N[2] + N[3];
            EXPECT_NEAR(sum, 1.0, tolerance);
        }
    }
}

TEST_F(InterpolationTest, ShapeFunctions_LinearTriangle) {
    // Test at vertices
    auto N_v0 = ShapeFunctionHelpers<double>::linear_triangle(0.0, 0.0);
    EXPECT_NEAR(N_v0[0], 1.0, tolerance);
    EXPECT_NEAR(N_v0[1], 0.0, tolerance);
    EXPECT_NEAR(N_v0[2], 0.0, tolerance);

    auto N_v1 = ShapeFunctionHelpers<double>::linear_triangle(1.0, 0.0);
    EXPECT_NEAR(N_v1[0], 0.0, tolerance);
    EXPECT_NEAR(N_v1[1], 1.0, tolerance);
    EXPECT_NEAR(N_v1[2], 0.0, tolerance);

    auto N_v2 = ShapeFunctionHelpers<double>::linear_triangle(0.0, 1.0);
    EXPECT_NEAR(N_v2[0], 0.0, tolerance);
    EXPECT_NEAR(N_v2[1], 0.0, tolerance);
    EXPECT_NEAR(N_v2[2], 1.0, tolerance);

    // Test at centroid
    auto N_center = ShapeFunctionHelpers<double>::linear_triangle(1.0 / 3.0, 1.0 / 3.0);
    EXPECT_NEAR(N_center[0], 1.0 / 3.0, tolerance);
    EXPECT_NEAR(N_center[1], 1.0 / 3.0, tolerance);
    EXPECT_NEAR(N_center[2], 1.0 / 3.0, tolerance);

    // Verify partition of unity
    EXPECT_NEAR(N_center[0] + N_center[1] + N_center[2], 1.0, tolerance);
}

TEST_F(InterpolationTest, ShapeFunctions_Trilinear3D) {
    // Test at corners
    auto N_000 = ShapeFunctionHelpers<double>::trilinear_3d(-1.0, -1.0, -1.0);
    EXPECT_NEAR(N_000[0], 1.0, tolerance);
    for (std::size_t i = 1; i < 8; ++i) {
        EXPECT_NEAR(N_000[i], 0.0, tolerance);
    }

    // Test at center
    auto N_center = ShapeFunctionHelpers<double>::trilinear_3d(0.0, 0.0, 0.0);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(N_center[i], 0.125, tolerance);
    }

    // Verify partition of unity
    double sum = 0.0;
    for (std::size_t i = 0; i < 8; ++i) {
        sum += N_center[i];
    }
    EXPECT_NEAR(sum, 1.0, tolerance);
}

TEST_F(InterpolationTest, ShapeFunctions_LinearTetrahedron) {
    // Test at vertices
    auto N_v0 = ShapeFunctionHelpers<double>::linear_tetrahedron(0.0, 0.0, 0.0);
    EXPECT_NEAR(N_v0[0], 1.0, tolerance);
    EXPECT_NEAR(N_v0[1], 0.0, tolerance);
    EXPECT_NEAR(N_v0[2], 0.0, tolerance);
    EXPECT_NEAR(N_v0[3], 0.0, tolerance);

    // Test at centroid
    auto N_center = ShapeFunctionHelpers<double>::linear_tetrahedron(0.25, 0.25, 0.25);
    EXPECT_NEAR(N_center[0], 0.25, tolerance);
    EXPECT_NEAR(N_center[1], 0.25, tolerance);
    EXPECT_NEAR(N_center[2], 0.25, tolerance);
    EXPECT_NEAR(N_center[3], 0.25, tolerance);

    // Verify partition of unity
    EXPECT_NEAR(N_center[0] + N_center[1] + N_center[2] + N_center[3], 1.0, tolerance);
}

// =============================================================================
// Lagrange Interpolation Tests
// =============================================================================

TEST_F(InterpolationTest, LagrangeBasis1D) {
    // Three-point interpolation nodes
    double nodes[] = {-1.0, 0.0, 1.0};
    size_t n = 3;

    // Test basis functions at nodes (Kronecker delta property)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double basis = LagrangeInterpolation<double>::basis_1d(i, nodes[j], nodes, n);
            if (i == j) {
                EXPECT_NEAR(basis, 1.0, tolerance);
            } else {
                EXPECT_NEAR(basis, 0.0, tolerance);
            }
        }
    }

    // Test partition of unity
    double x = 0.5;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += LagrangeInterpolation<double>::basis_1d(i, x, nodes, n);
    }
    EXPECT_NEAR(sum, 1.0, tolerance);
}

TEST_F(InterpolationTest, LagrangeInterpolate1D) {
    // Interpolate a quadratic function
    double nodes[] = {-1.0, 0.0, 1.0};
    double values[] = {1.0, 0.0, 1.0};  // f(x) = x^2
    size_t n = 3;

    // Test at nodes
    for (size_t i = 0; i < n; ++i) {
        double interp = LagrangeInterpolation<double>::interpolate_1d(nodes[i], nodes, values, n);
        EXPECT_NEAR(interp, values[i], tolerance);
    }

    // Test at intermediate points
    double x = 0.5;
    double interp = LagrangeInterpolation<double>::interpolate_1d(x, nodes, values, n);
    EXPECT_NEAR(interp, x * x, tolerance);  // Should match x^2
}

TEST_F(InterpolationTest, LagrangeDerivative1D) {
    // Interpolate a linear function f(x) = 2x + 1
    double nodes[] = {-1.0, 0.0, 1.0};
    double values[] = {-1.0, 1.0, 3.0};
    size_t n = 3;

    // Derivative should be constant = 2
    for (double x = -0.8; x <= 0.8; x += 0.2) {
        double deriv = LagrangeInterpolation<double>::derivative_1d(x, nodes, values, n);
        EXPECT_NEAR(deriv, 2.0, tolerance);
    }
}

// =============================================================================
// Hermite Interpolation Tests
// =============================================================================

TEST_F(InterpolationTest, HermiteCubic_Scalar) {
    // Interpolate between two points with specified derivatives
    double p0 = 1.0;  // Value at t=0
    double p1 = 2.0;  // Value at t=1
    double m0 = 0.0;  // Derivative at t=0
    double m1 = 0.0;  // Derivative at t=1

    // Test at endpoints
    EXPECT_NEAR(HermiteInterpolation<double>::cubic(0.0, p0, p1, m0, m1), p0, tolerance);
    EXPECT_NEAR(HermiteInterpolation<double>::cubic(1.0, p0, p1, m0, m1), p1, tolerance);

    // Test at midpoint (should be average for zero derivatives)
    EXPECT_NEAR(HermiteInterpolation<double>::cubic(0.5, p0, p1, m0, m1), 1.5, tolerance);

    // Test with non-zero derivatives
    m0 = 1.0;
    m1 = -1.0;
    double mid = HermiteInterpolation<double>::cubic(0.5, p0, p1, m0, m1);
    // The curve should overshoot the linear interpolation
    EXPECT_GT(mid, 1.5);
}

TEST_F(InterpolationTest, HermiteCubic_Vector) {
    Vector3<double> p0{0.0, 0.0, 0.0};
    Vector3<double> p1{1.0, 1.0, 1.0};
    Vector3<double> m0{1.0, 0.0, 0.0};
    Vector3<double> m1{0.0, 1.0, 0.0};

    auto result_0 = HermiteInterpolation<double>::cubic(0.0, p0, p1, m0, m1);
    EXPECT_TRUE(vector_approx_equal(result_0, p0));

    auto result_1 = HermiteInterpolation<double>::cubic(1.0, p0, p1, m0, m1);
    EXPECT_TRUE(vector_approx_equal(result_1, p1));

    // Test smoothness - derivatives should match at endpoints
    double h = 1e-8;
    auto result_h = HermiteInterpolation<double>::cubic(h, p0, p1, m0, m1);
    // Force materialization of the difference before division to avoid dangling reference
    Vector3<double> diff = result_h - p0;
    Vector3<double> deriv_0 = diff / h;
    EXPECT_NEAR(deriv_0[0], m0[0], 1e-6);
    EXPECT_NEAR(deriv_0[1], m0[1], 1e-6);
    EXPECT_NEAR(deriv_0[2], m0[2], 1e-6);
}

TEST_F(InterpolationTest, CatmullRomSpline) {
    // Test with four control points
    double p0 = 0.0;
    double p1 = 1.0;
    double p2 = 2.0;
    double p3 = 1.5;

    // At t=0, should equal p1
    EXPECT_NEAR(HermiteInterpolation<double>::catmull_rom(0.0, p0, p1, p2, p3), p1, tolerance);

    // At t=1, should equal p2
    EXPECT_NEAR(HermiteInterpolation<double>::catmull_rom(1.0, p0, p1, p2, p3), p2, tolerance);

    // Test continuity
    double t = 0.5;
    double result = HermiteInterpolation<double>::catmull_rom(t, p0, p1, p2, p3);
    EXPECT_GE(result, std::min(p1, p2));
    EXPECT_LE(result, std::max(p1, p2));
}

// =============================================================================
// Inverse Distance Weighting Tests
// =============================================================================

TEST_F(InterpolationTest, IDWInterpolation_ExactAtPoints) {
    // 2D points and values
    Vector2<double> points[] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };
    double values[] = {1.0, 2.0, 3.0};
    size_t n = 3;

    // Test exact interpolation at data points
    for (size_t i = 0; i < n; ++i) {
        double result = IDWInterpolation<double, 2>::interpolate(
            points[i], points, values, n, 2.0
        );
        EXPECT_NEAR(result, values[i], tolerance);
    }
}

TEST_F(InterpolationTest, IDWInterpolation_Weighted) {
    // Test that closer points have more influence
    Vector2<double> points[] = {
        {0.0, 0.0},
        {10.0, 0.0}
    };
    double values[] = {0.0, 10.0};
    size_t n = 2;

    // Query point closer to first point
    Vector2<double> query1{2.0, 0.0};
    double result1 = IDWInterpolation<double, 2>::interpolate(
        query1, points, values, n, 2.0
    );
    EXPECT_LT(result1, 5.0);  // Should be less than midpoint value

    // Query point closer to second point
    Vector2<double> query2{8.0, 0.0};
    double result2 = IDWInterpolation<double, 2>::interpolate(
        query2, points, values, n, 2.0
    );
    EXPECT_GT(result2, 5.0);  // Should be greater than midpoint value
}

TEST_F(InterpolationTest, IDWInterpolation_PowerParameter) {
    Vector2<double> points[] = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    double values[] = {0.0, 1.0};
    size_t n = 2;

    Vector2<double> query{0.5, 0.0};  // Midpoint

    // With power = 0, all points have equal weight
    double result_p0 = IDWInterpolation<double, 2>::interpolate(
        query, points, values, n, 0.0
    );
    EXPECT_NEAR(result_p0, 0.5, tolerance);  // Average

    // With higher power, interpolation becomes more local
    double result_p1 = IDWInterpolation<double, 2>::interpolate(
        query, points, values, n, 1.0
    );
    double result_p2 = IDWInterpolation<double, 2>::interpolate(
        query, points, values, n, 2.0
    );
    double result_p4 = IDWInterpolation<double, 2>::interpolate(
        query, points, values, n, 4.0
    );

    // All should be 0.5 at midpoint due to symmetry
    EXPECT_NEAR(result_p1, 0.5, tolerance);
    EXPECT_NEAR(result_p2, 0.5, tolerance);
    EXPECT_NEAR(result_p4, 0.5, tolerance);
}

// =============================================================================
// Edge Cases and Numerical Stability Tests
// =============================================================================

TEST_F(InterpolationTest, NumericalStability_SmallDenominators) {
    // Test barycentric coordinates with nearly degenerate triangle
    Vector2<double> v0{0.0, 0.0};
    Vector2<double> v1{1.0, 0.0};
    Vector2<double> v2{0.5, 1e-15};  // Nearly collinear

    Vector2<double> p{0.5, 0.0};
    auto bary = BarycentricCoords<double>::compute_2d(p, v0, v1, v2);

    // Should still sum to 1
    EXPECT_NEAR(bary[0] + bary[1] + bary[2], 1.0, tolerance);
}

TEST_F(InterpolationTest, NumericalStability_LargeCoordinates) {
    // Test with large coordinate values
    double scale = 1e10;
    Vector2<double> v0{0.0 * scale, 0.0 * scale};
    Vector2<double> v1{1.0 * scale, 0.0 * scale};
    Vector2<double> v2{0.0 * scale, 1.0 * scale};

    Vector2<double> p{0.3 * scale, 0.3 * scale};
    auto bary = BarycentricCoords<double>::compute_2d(p, v0, v1, v2);

    // Check sum
    EXPECT_NEAR(bary[0] + bary[1] + bary[2], 1.0, 1e-10);

    // Check values are reasonable
    EXPECT_GT(bary[0], 0.0);
    EXPECT_GT(bary[1], 0.0);
    EXPECT_GT(bary[2], 0.0);
}

TEST_F(InterpolationTest, EdgeCase_ZeroDistanceIDW) {
    // Test IDW when query point coincides with data point
    Vector2<double> points[] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };
    double values[] = {1.0, 2.0, 3.0};
    size_t n = 3;

    // Query at exact data point
    double result = IDWInterpolation<double, 2>::interpolate(
        points[0], points, values, n, 2.0
    );
    EXPECT_NEAR(result, values[0], tolerance);
}

// =============================================================================
// Property-Based Tests
// =============================================================================

TEST_F(InterpolationTest, PropertyBased_BarycentricReconstruction) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int iter = 0; iter < 100; ++iter) {
        // Random triangle
        Vector2<double> v0{dist(rng), dist(rng)};
        Vector2<double> v1{dist(rng) + 1.0, dist(rng)};
        Vector2<double> v2{dist(rng), dist(rng) + 1.0};

        // Random point inside triangle
        double u = dist(rng);
        double v = dist(rng);
        if (u + v > 1.0) {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        Vector2<double> p = v0 + u * (v1 - v0) + v * (v2 - v0);

        // Compute barycentric coordinates
        auto bary = BarycentricCoords<double>::compute_2d(p, v0, v1, v2);

        // Reconstruct point
        Vector2<double> p_reconstructed = bary[0] * v0 + bary[1] * v1 + bary[2] * v2;

        EXPECT_NEAR(p_reconstructed[0], p[0], 1e-10);
        EXPECT_NEAR(p_reconstructed[1], p[1], 1e-10);
    }
}

TEST_F(InterpolationTest, PropertyBased_InterpolationMonotonicity) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int iter = 0; iter < 50; ++iter) {
        double a = dist(rng);
        double b = a + dist(rng);  // Ensure b > a

        // Linear interpolation should be monotonic
        for (double t = 0.0; t <= 1.0; t += 0.1) {
            double result = lerp(a, b, t);
            EXPECT_GE(result, a);
            EXPECT_LE(result, b);
        }

        // Smoothstep should also preserve bounds
        for (double t = 0.0; t <= 1.0; t += 0.1) {
            double s = smoothstep(t);
            double result = lerp(a, b, s);
            EXPECT_GE(result, a - 1e-10);
            EXPECT_LE(result, b + 1e-10);
        }
    }
}
