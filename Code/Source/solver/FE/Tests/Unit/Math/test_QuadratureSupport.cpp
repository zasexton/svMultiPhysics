/**
 * @file test_QuadratureSupport.cpp
 * @brief Unit tests for QuadratureSupport.h - quadrature and integration support
 */

#include <gtest/gtest.h>
#include "FE/Math/QuadratureSupport.h"
#include "FE/Math/Vector.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/MathConstants.h"
#include <limits>
#include <cmath>
#include <array>
#include <vector>
#include <random>

using namespace svmp::FE::math;

// Test fixture for QuadratureSupport tests
class QuadratureSupportTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-14;
    static constexpr double loose_tolerance = 1e-10;

    void SetUp() override {
        rng.seed(42);  // Fixed seed for reproducibility
    }

    void TearDown() override {}

    // Helper to check if matrices are approximately equal
    template<typename T, std::size_t M, std::size_t N>
    bool matrix_approx_equal(const Matrix<T, M, N>& A,
                             const Matrix<T, M, N>& B,
                             T tol = tolerance) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                if (std::abs(A(i, j) - B(i, j)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    std::mt19937 rng;
};

// =============================================================================
// Jacobian Operations Tests
// =============================================================================

TEST_F(QuadratureSupportTest, JacobianOperations2D_Identity) {
    // Identity mapping (reference = physical)
    Matrix<double, 2, 4> grad_shape;  // Shape function gradients for 4-node quad
    grad_shape(0, 0) = -0.25; grad_shape(0, 1) = 0.25;
    grad_shape(0, 2) = 0.25;  grad_shape(0, 3) = -0.25;
    grad_shape(1, 0) = -0.25; grad_shape(1, 1) = -0.25;
    grad_shape(1, 2) = 0.25;  grad_shape(1, 3) = 0.25;

    Matrix<double, 2, 4> node_coords;  // Unit square nodes
    node_coords(0, 0) = -1.0; node_coords(1, 0) = -1.0;  // Node 0
    node_coords(0, 1) = 1.0;  node_coords(1, 1) = -1.0;  // Node 1
    node_coords(0, 2) = 1.0;  node_coords(1, 2) = 1.0;   // Node 2
    node_coords(0, 3) = -1.0; node_coords(1, 3) = 1.0;   // Node 3

    auto J = JacobianOperations<double, 2>::compute_jacobian(grad_shape, node_coords);

    // Should be identity matrix (scaled by 2 due to [-1,1] to [-1,1] mapping)
    EXPECT_NEAR(J(0, 0), 1.0, tolerance);
    EXPECT_NEAR(J(0, 1), 0.0, tolerance);
    EXPECT_NEAR(J(1, 0), 0.0, tolerance);
    EXPECT_NEAR(J(1, 1), 1.0, tolerance);

    // Determinant should be 1
    double det_J = JacobianOperations<double, 2>::determinant(J);
    EXPECT_NEAR(det_J, 1.0, tolerance);

    // Inverse should also be identity
    auto J_inv = JacobianOperations<double, 2>::inverse(J);
    EXPECT_NEAR(J_inv(0, 0), 1.0, tolerance);
    EXPECT_NEAR(J_inv(0, 1), 0.0, tolerance);
    EXPECT_NEAR(J_inv(1, 0), 0.0, tolerance);
    EXPECT_NEAR(J_inv(1, 1), 1.0, tolerance);

    // Check validity
    EXPECT_TRUE((JacobianOperations<double, 2>::is_valid(J)));

    // Condition number should be 1 for identity
    double cond = JacobianOperations<double, 2>::condition_number(J);
    EXPECT_NEAR(cond, 1.0, loose_tolerance);
}

TEST_F(QuadratureSupportTest, JacobianOperations2D_PathologicalGradientSigns) {
    // Intentionally inconsistent gradient signs (e.g., upstream shape gradient bug).
    // compute_jacobian should not try to "fix" this; it should surface the degeneracy.
    Matrix<double, 2, 4> grad_shape;
    grad_shape(0, 0) = -0.25; grad_shape(0, 1) = -0.25;
    grad_shape(0, 2) = -0.25; grad_shape(0, 3) = -0.25;
    grad_shape(1, 0) = -0.25; grad_shape(1, 1) = -0.25;
    grad_shape(1, 2) = 0.25;  grad_shape(1, 3) = 0.25;

    Matrix<double, 2, 4> node_coords;
    node_coords(0, 0) = -1.0; node_coords(1, 0) = -1.0;
    node_coords(0, 1) = 1.0;  node_coords(1, 1) = -1.0;
    node_coords(0, 2) = 1.0;  node_coords(1, 2) = 1.0;
    node_coords(0, 3) = -1.0; node_coords(1, 3) = 1.0;

    auto J = JacobianOperations<double, 2>::compute_jacobian(grad_shape, node_coords);
    double det_J = JacobianOperations<double, 2>::determinant(J);

    EXPECT_NEAR(det_J, 0.0, tolerance);
    EXPECT_FALSE((JacobianOperations<double, 2>::is_valid(J)));
}

TEST_F(QuadratureSupportTest, JacobianOperations2D_Scaling) {
    // Scaled square (2x3 rectangle)
    Matrix<double, 2, 4> grad_shape;
    grad_shape(0, 0) = -0.25; grad_shape(0, 1) = 0.25;
    grad_shape(0, 2) = 0.25;  grad_shape(0, 3) = -0.25;
    grad_shape(1, 0) = -0.25; grad_shape(1, 1) = -0.25;
    grad_shape(1, 2) = 0.25;  grad_shape(1, 3) = 0.25;

    Matrix<double, 2, 4> node_coords;
    node_coords(0, 0) = 0.0; node_coords(1, 0) = 0.0;  // Node 0
    node_coords(0, 1) = 2.0; node_coords(1, 1) = 0.0;  // Node 1
    node_coords(0, 2) = 2.0; node_coords(1, 2) = 3.0;  // Node 2
    node_coords(0, 3) = 0.0; node_coords(1, 3) = 3.0;  // Node 3

    auto J = JacobianOperations<double, 2>::compute_jacobian(grad_shape, node_coords);

    // Jacobian should reflect scaling
    EXPECT_NEAR(J(0, 0), 1.0, tolerance);  // dx/dxi = 2/2 = 1
    EXPECT_NEAR(J(0, 1), 0.0, tolerance);
    EXPECT_NEAR(J(1, 0), 0.0, tolerance);
    EXPECT_NEAR(J(1, 1), 1.5, tolerance);  // dy/deta = 3/2 = 1.5

    // Determinant should be area scaling factor
    double det_J = JacobianOperations<double, 2>::determinant(J);
    EXPECT_NEAR(det_J, 1.5, tolerance);  // 1.0 * 1.5

    // Integration weight
    double quad_weight = 0.5;  // Example quadrature weight
    double int_weight = JacobianOperations<double, 2>::integration_weight(det_J, quad_weight);
    EXPECT_NEAR(int_weight, 0.75, tolerance);  // |1.5| * 0.5
}

TEST_F(QuadratureSupportTest, JacobianOperations2D_Distorted) {
    // Distorted quadrilateral
    Matrix<double, 2, 4> grad_shape;
    grad_shape(0, 0) = -0.25; grad_shape(0, 1) = 0.25;
    grad_shape(0, 2) = 0.25;  grad_shape(0, 3) = -0.25;
    grad_shape(1, 0) = -0.25; grad_shape(1, 1) = -0.25;
    grad_shape(1, 2) = 0.25;  grad_shape(1, 3) = 0.25;

    Matrix<double, 2, 4> node_coords;
    node_coords(0, 0) = 0.0; node_coords(1, 0) = 0.0;  // Node 0
    node_coords(0, 1) = 2.0; node_coords(1, 1) = 0.5;  // Node 1 (shifted up)
    node_coords(0, 2) = 2.5; node_coords(1, 2) = 2.0;  // Node 2 (shifted right)
    node_coords(0, 3) = 0.0; node_coords(1, 3) = 2.0;  // Node 3

    auto J = JacobianOperations<double, 2>::compute_jacobian(grad_shape, node_coords);

    // Check that Jacobian is non-singular
    double det_J = JacobianOperations<double, 2>::determinant(J);
    EXPECT_GT(det_J, 0.0);  // Should be positive for valid element

    // Check validity
    EXPECT_TRUE((JacobianOperations<double, 2>::is_valid(J)));

    // Condition number should be reasonable
    double cond = JacobianOperations<double, 2>::condition_number(J);
    EXPECT_GT(cond, 1.0);  // Greater than 1 for distorted element
    EXPECT_LT(cond, 100.0);  // But not too large
}

TEST_F(QuadratureSupportTest, JacobianOperations3D_Hexahedron) {
    // Unit cube
    Matrix<double, 3, 8> grad_shape;  // Shape function gradients at center
    Matrix<double, 3, 8> node_coords;  // Unit cube nodes
    node_coords(0, 0) = 0; node_coords(1, 0) = 0; node_coords(2, 0) = 0;
    node_coords(0, 1) = 1; node_coords(1, 1) = 0; node_coords(2, 1) = 0;
    node_coords(0, 2) = 1; node_coords(1, 2) = 1; node_coords(2, 2) = 0;
    node_coords(0, 3) = 0; node_coords(1, 3) = 1; node_coords(2, 3) = 0;
    node_coords(0, 4) = 0; node_coords(1, 4) = 0; node_coords(2, 4) = 1;
    node_coords(0, 5) = 1; node_coords(1, 5) = 0; node_coords(2, 5) = 1;
    node_coords(0, 6) = 1; node_coords(1, 6) = 1; node_coords(2, 6) = 1;
    node_coords(0, 7) = 0; node_coords(1, 7) = 1; node_coords(2, 7) = 1;

    // Trilinear hexahedron gradients at the element center.
    // For a node at reference coordinates (±1, ±1, ±1), we have:
    // dN/dxi = sign_xi/8, dN/deta = sign_eta/8, dN/dzeta = sign_zeta/8.
    // Here we infer the reference signs from the physical cube mapping [0,1]^3.
    for (std::size_t a = 0; a < 8; ++a) {
        grad_shape(0, a) = node_coords(0, a) > 0.5 ? 0.125 : -0.125;
        grad_shape(1, a) = node_coords(1, a) > 0.5 ? 0.125 : -0.125;
        grad_shape(2, a) = node_coords(2, a) > 0.5 ? 0.125 : -0.125;
    }

    auto J = JacobianOperations<double, 3>::compute_jacobian(grad_shape, node_coords);

    // For unit cube, Jacobian should be diagonal with 0.5 values
    // (mapping from [-1,1]^3 to [0,1]^3)
    EXPECT_NEAR(std::abs(J(0, 0)), 0.5, loose_tolerance);
    EXPECT_NEAR(std::abs(J(1, 1)), 0.5, loose_tolerance);
    EXPECT_NEAR(std::abs(J(2, 2)), 0.5, loose_tolerance);

    // Determinant should be 0.125 (0.5^3)
    double det_J = JacobianOperations<double, 3>::determinant(J);
    EXPECT_NEAR(std::abs(det_J), 0.125, loose_tolerance);

    // Check validity
    EXPECT_TRUE((JacobianOperations<double, 3>::is_valid(J, 1e-10)));
}

TEST_F(QuadratureSupportTest, JacobianOperations1D) {
    // 1D line element
    Vector<double, 2> grad_shape{-0.5, 0.5};  // dN/dxi at center
    Vector<double, 2> node_coords{2.0, 5.0};  // Line from x=2 to x=5

    double J = JacobianOperations<double, 1>::compute_jacobian(grad_shape, node_coords);

    // Jacobian should be dx/dxi = (5-2)/2 = 1.5
    EXPECT_NEAR(J, 1.5, tolerance);

    // Determinant (absolute value for 1D)
    double det_J = JacobianOperations<double, 1>::determinant(J);
    EXPECT_NEAR(det_J, 1.5, tolerance);

    // Inverse
    double J_inv = JacobianOperations<double, 1>::inverse(J);
    EXPECT_NEAR(J_inv, 1.0 / 1.5, tolerance);

    // Transform gradient
    double grad_ref = 2.0;  // Gradient in reference space
    double grad_phys = JacobianOperations<double, 1>::transform_gradient(grad_ref, J_inv);
    EXPECT_NEAR(grad_phys, 2.0 / 1.5, tolerance);

    // Integration weight
    double quad_weight = 1.0;
    double int_weight = JacobianOperations<double, 1>::integration_weight(J, quad_weight);
    EXPECT_NEAR(int_weight, 1.5, tolerance);
}

TEST_F(QuadratureSupportTest, GradientTransformation) {
    // Test gradient transformation from reference to physical
    Matrix<double, 2, 2> J;
    J(0, 0) = 2.0; J(0, 1) = 0.0;  // Scaling by 2 in x
    J(1, 0) = 0.0; J(1, 1) = 3.0;  // Scaling by 3 in y

    auto J_inv = JacobianOperations<double, 2>::inverse(J);

    Vector<double, 2> grad_ref{1.0, 1.0};  // Gradient in reference space
    auto grad_phys = JacobianOperations<double, 2>::transform_gradient(grad_ref, J_inv);

    // grad_phys = J^{-T} * grad_ref
    EXPECT_NEAR(grad_phys[0], 0.5, tolerance);  // 1/2
    EXPECT_NEAR(grad_phys[1], 1.0 / 3.0, tolerance);  // 1/3
}

TEST_F(QuadratureSupportTest, ReferenceToPhysicalMapping) {
    // Map point from reference to physical element (2D triangle)
    Vector<double, 3> shape_values{0.2, 0.3, 0.5};  // Barycentric coordinates

    Matrix<double, 2, 3> node_coords;
    node_coords(0, 0) = 0.0; node_coords(1, 0) = 0.0;  // Node 0
    node_coords(0, 1) = 4.0; node_coords(1, 1) = 0.0;  // Node 1
    node_coords(0, 2) = 0.0; node_coords(1, 2) = 3.0;  // Node 2

    auto x_phys = JacobianOperations<double, 2>::reference_to_physical(
        shape_values, node_coords);

    // x = sum(N_i * x_i)
    double expected_x = 0.2 * 0.0 + 0.3 * 4.0 + 0.5 * 0.0;
    double expected_y = 0.2 * 0.0 + 0.3 * 0.0 + 0.5 * 3.0;

    EXPECT_NEAR(x_phys[0], expected_x, tolerance);
    EXPECT_NEAR(x_phys[1], expected_y, tolerance);
}

// =============================================================================
// Integration Helpers Tests
// =============================================================================

TEST_F(QuadratureSupportTest, Integrate1D_Constant) {
    // Integrate constant function
    std::vector<double> values{2.0, 2.0, 2.0};
    std::vector<double> weights{0.5, 1.0, 0.5};  // Simpson's rule weights

    double result = IntegrationHelpers<double>::integrate_1d(values, weights);
    EXPECT_NEAR(result, 4.0, tolerance);  // 2.0 * (0.5 + 1.0 + 0.5)
}

TEST_F(QuadratureSupportTest, Integrate1D_Linear) {
    // Integrate linear function f(x) = x on [0, 1]
    // Using 3-point Gauss quadrature
    std::vector<double> values{
        -std::sqrt(3.0/5.0),  // f at first Gauss point
        0.0,                   // f at second Gauss point
        std::sqrt(3.0/5.0)     // f at third Gauss point
    };
    std::vector<double> weights{5.0/9.0, 8.0/9.0, 5.0/9.0};

    double result = IntegrationHelpers<double>::integrate_1d(values, weights);
    // Integral of x from -1 to 1 is 0
    EXPECT_NEAR(result, 0.0, tolerance);
}

TEST_F(QuadratureSupportTest, Integrate1D_Quadratic) {
    // Integrate f(x) = x^2 on [-1, 1]
    // Using 2-point Gauss quadrature
    double xi1 = -1.0 / std::sqrt(3.0);
    double xi2 = 1.0 / std::sqrt(3.0);
    std::vector<double> values{xi1 * xi1, xi2 * xi2};
    std::vector<double> weights{1.0, 1.0};

    double result = IntegrationHelpers<double>::integrate_1d(values, weights);
    // Integral of x^2 from -1 to 1 is 2/3
    EXPECT_NEAR(result, 2.0 / 3.0, tolerance);
}

TEST_F(QuadratureSupportTest, IntegrateND_2D) {
    // Integrate over 2D tensor product quadrature
    // f(x,y) = 1 over unit square
    std::vector<double> values(4, 1.0);  // 2x2 quadrature points

    std::array<std::vector<double>, 2> weights;
    weights[0] = {1.0, 1.0};  // Weights in x-direction
    weights[1] = {1.0, 1.0};  // Weights in y-direction

    std::array<std::size_t, 2> dims{2, 2};

    double result = IntegrationHelpers<double>::integrate_nd(values, weights, dims);
    EXPECT_NEAR(result, 4.0, tolerance);  // 1.0 * (1+1) * (1+1)
}

TEST_F(QuadratureSupportTest, IntegrateND_3D) {
    // Integrate over 3D tensor product quadrature
    // f(x,y,z) = x*y*z
    std::vector<double> values;
    std::array<std::vector<double>, 3> weights;
    std::array<std::size_t, 3> dims{2, 2, 2};

    // 2x2x2 Gauss points
    double xi = 1.0 / std::sqrt(3.0);
    std::vector<double> pts{-xi, xi};
    weights[0] = weights[1] = weights[2] = {1.0, 1.0};

    // Generate function values
    for (double z : pts) {
        for (double y : pts) {
            for (double x : pts) {
                values.push_back(x * y * z);
            }
        }
    }

    double result = IntegrationHelpers<double>::integrate_nd(values, weights, dims);
    // Integral of x*y*z over [-1,1]^3 is 0 (odd function)
    EXPECT_NEAR(result, 0.0, tolerance);
}

TEST_F(QuadratureSupportTest, ComputeElementVolume) {
    // Compute volume/area of element
    std::vector<double> det_jacobians{2.0, 2.0, 2.0, 2.0};  // Constant Jacobian
    std::vector<double> weights{1.0, 1.0, 1.0, 1.0};  // Unit weights

    double volume = IntegrationHelpers<double>::compute_element_volume(det_jacobians, weights);
    EXPECT_NEAR(volume, 8.0, tolerance);  // 2.0 * 4

    // With varying Jacobians
    std::vector<double> det_jacobians2{1.0, 2.0, 3.0};
    std::vector<double> weights2{0.5, 1.0, 0.5};

    double volume2 = IntegrationHelpers<double>::compute_element_volume(det_jacobians2, weights2);
    EXPECT_NEAR(volume2, 4.0, tolerance);  // 1.0*0.5 + 2.0*1.0 + 3.0*0.5
}

// =============================================================================
// Reference Element Mappings Tests
// =============================================================================

TEST_F(QuadratureSupportTest, StandardToUnitMapping) {
    // Test mapping from [-1,1] to [0,1]
    EXPECT_NEAR(ReferenceElementMappings<double>::standard_to_unit(-1.0), 0.0, tolerance);
    EXPECT_NEAR(ReferenceElementMappings<double>::standard_to_unit(0.0), 0.5, tolerance);
    EXPECT_NEAR(ReferenceElementMappings<double>::standard_to_unit(1.0), 1.0, tolerance);

    // Test inverse mapping
    EXPECT_NEAR(ReferenceElementMappings<double>::unit_to_standard(0.0), -1.0, tolerance);
    EXPECT_NEAR(ReferenceElementMappings<double>::unit_to_standard(0.5), 0.0, tolerance);
    EXPECT_NEAR(ReferenceElementMappings<double>::unit_to_standard(1.0), 1.0, tolerance);

    // Round-trip test
    double xi = 0.3;
    double x = ReferenceElementMappings<double>::standard_to_unit(xi);
    double xi_back = ReferenceElementMappings<double>::unit_to_standard(x);
    EXPECT_NEAR(xi_back, xi, tolerance);
}

TEST_F(QuadratureSupportTest, SquareToTriangleMapping) {
    // Test Duffy transformation
    // Map corners of square to triangle
    auto p00 = ReferenceElementMappings<double>::square_to_triangle(-1.0, -1.0);
    EXPECT_NEAR(p00[0], 0.0, tolerance);
    EXPECT_NEAR(p00[1], 0.0, tolerance);

    auto p10 = ReferenceElementMappings<double>::square_to_triangle(1.0, -1.0);
    EXPECT_NEAR(p10[0], 1.0, tolerance);
    EXPECT_NEAR(p10[1], 0.0, tolerance);

    auto p01 = ReferenceElementMappings<double>::square_to_triangle(-1.0, 1.0);
    EXPECT_NEAR(p01[0], 0.0, tolerance);
    EXPECT_NEAR(p01[1], 1.0, tolerance);

    // Test Jacobian
    double jac = ReferenceElementMappings<double>::square_to_triangle_jacobian(0.0, 0.0);
    EXPECT_NEAR(jac, 1.0 / 8.0, tolerance);

    // Test at different point
    double jac2 = ReferenceElementMappings<double>::square_to_triangle_jacobian(0.0, 0.5);
    EXPECT_NEAR(jac2, (1.0 - 0.5) / 8.0, tolerance);
}

TEST_F(QuadratureSupportTest, CubeToTetrahedronMapping) {
    // Test mapping from cube to tetrahedron
    auto p000 = ReferenceElementMappings<double>::cube_to_tetrahedron(-1, -1, -1);
    EXPECT_NEAR(p000[0], 0.0, tolerance);
    EXPECT_NEAR(p000[1], 0.0, tolerance);
    EXPECT_NEAR(p000[2], 0.0, tolerance);

    auto p111 = ReferenceElementMappings<double>::cube_to_tetrahedron(1, 1, 1);
    // This point maps to (1, 1, 1) which is outside standard tetrahedron
    // but the mapping is still valid for integration
    (void)p111;

    // Test Jacobian
    double jac = ReferenceElementMappings<double>::cube_to_tetrahedron_jacobian(0, 0, 0);
    EXPECT_GT(jac, 0.0);  // Should be positive
}

TEST_F(QuadratureSupportTest, LineToCircleMapping) {
    // Map from [0,1] line to circle
    auto p0 = ReferenceElementMappings<double>::line_to_circle(0.0, 1.0);
    EXPECT_NEAR(p0[0], 1.0, tolerance);  // cos(0)
    EXPECT_NEAR(p0[1], 0.0, tolerance);  // sin(0)

    auto p_quarter = ReferenceElementMappings<double>::line_to_circle(0.25, 1.0);
    EXPECT_NEAR(p_quarter[0], 0.0, tolerance);  // cos(pi/2)
    EXPECT_NEAR(p_quarter[1], 1.0, tolerance);  // sin(pi/2)

    auto p_half = ReferenceElementMappings<double>::line_to_circle(0.5, 1.0);
    EXPECT_NEAR(p_half[0], -1.0, tolerance);  // cos(pi)
    EXPECT_NEAR(p_half[1], 0.0, tolerance);   // sin(pi)

    // Test with different radius
    auto p_r2 = ReferenceElementMappings<double>::line_to_circle(0.25, 2.0);
    EXPECT_NEAR(p_r2[0], 0.0, tolerance);   // 2*cos(pi/2)
    EXPECT_NEAR(p_r2[1], 2.0, tolerance);   // 2*sin(pi/2)
}

// =============================================================================
// Tensor Transformations Tests
// =============================================================================

TEST_F(QuadratureSupportTest, TensorTransformations_PushForwardVector) {
    // Test contravariant transformation
    Matrix<double, 2, 2> J;
    J(0, 0) = 2.0; J(0, 1) = 0.0;
    J(1, 0) = 0.0; J(1, 1) = 3.0;

    Vector<double, 2> v_ref{1.0, 1.0};
    auto v_phys = TensorTransformations<double, 2>::push_forward_vector(v_ref, J);

    EXPECT_NEAR(v_phys[0], 2.0, tolerance);  // 2*1 + 0*1
    EXPECT_NEAR(v_phys[1], 3.0, tolerance);  // 0*1 + 3*1
}

TEST_F(QuadratureSupportTest, TensorTransformations_PullBackVector) {
    // Test covariant transformation
    Matrix<double, 2, 2> J;
    J(0, 0) = 2.0; J(0, 1) = 0.0;
    J(1, 0) = 0.0; J(1, 1) = 3.0;

    auto J_inv = J.inverse();

    Vector<double, 2> v_phys{2.0, 3.0};
    auto v_ref = TensorTransformations<double, 2>::pull_back_vector(v_phys, J_inv);

    EXPECT_NEAR(v_ref[0], 1.0, tolerance);  // 2/2
    EXPECT_NEAR(v_ref[1], 1.0, tolerance);  // 3/3
}

TEST_F(QuadratureSupportTest, TensorTransformations_PiolaTransform) {
    // Test Piola transformation for stress tensor
    Matrix<double, 2, 2> sigma_ref;
    sigma_ref(0, 0) = 1.0; sigma_ref(0, 1) = 0.5;
    sigma_ref(1, 0) = 0.5; sigma_ref(1, 1) = 2.0;

    Matrix<double, 2, 2> J;
    J(0, 0) = 2.0; J(0, 1) = 0.0;
    J(1, 0) = 0.0; J(1, 1) = 2.0;

    double det_J = J.determinant();

    auto sigma_phys = TensorTransformations<double, 2>::piola_transform(sigma_ref, J, det_J);

    // �_phys = (1/det(J)) * J * �_ref * J^T
    // For diagonal J with factor 2, and det(J) = 4:
    // �_phys = (1/4) * 2 * �_ref * 2 = �_ref
    EXPECT_NEAR(sigma_phys(0, 0), 1.0, tolerance);
    EXPECT_NEAR(sigma_phys(0, 1), 0.5, tolerance);
    EXPECT_NEAR(sigma_phys(1, 0), 0.5, tolerance);
    EXPECT_NEAR(sigma_phys(1, 1), 2.0, tolerance);
}

TEST_F(QuadratureSupportTest, TensorTransformations_InversePiolaTransform) {
    // Test inverse Piola transformation
    Matrix<double, 2, 2> sigma_phys;
    sigma_phys(0, 0) = 1.0; sigma_phys(0, 1) = 0.5;
    sigma_phys(1, 0) = 0.5; sigma_phys(1, 1) = 2.0;

    Matrix<double, 2, 2> J;
    J(0, 0) = 2.0; J(0, 1) = 0.0;
    J(1, 0) = 0.0; J(1, 1) = 2.0;

    auto J_inv = J.inverse();
    double det_J = J.determinant();

    auto sigma_ref = TensorTransformations<double, 2>::inverse_piola_transform(
        sigma_phys, J_inv, det_J);

    // Should recover original tensor
    EXPECT_NEAR(sigma_ref(0, 0), 1.0, tolerance);
    EXPECT_NEAR(sigma_ref(0, 1), 0.5, tolerance);
    EXPECT_NEAR(sigma_ref(1, 0), 0.5, tolerance);
    EXPECT_NEAR(sigma_ref(1, 1), 2.0, tolerance);
}

TEST_F(QuadratureSupportTest, TensorTransformations_CovariantTransform) {
    // Test covariant transformation of rank-2 tensor
    Matrix<double, 2, 2> T_ref;
    T_ref(0, 0) = 1.0; T_ref(0, 1) = 0.0;
    T_ref(1, 0) = 0.0; T_ref(1, 1) = 1.0;

    Matrix<double, 2, 2> J;
    J(0, 0) = 2.0; J(0, 1) = 0.0;
    J(1, 0) = 0.0; J(1, 1) = 3.0;

    auto J_inv = J.inverse();

    auto T_phys = TensorTransformations<double, 2>::covariant_transform(T_ref, J_inv);

    // T_phys = J^{-T} * T_ref * J^{-1}
    EXPECT_NEAR(T_phys(0, 0), 0.25, tolerance);  // 1/4
    EXPECT_NEAR(T_phys(0, 1), 0.0, tolerance);
    EXPECT_NEAR(T_phys(1, 0), 0.0, tolerance);
    EXPECT_NEAR(T_phys(1, 1), 1.0 / 9.0, tolerance);
}

TEST_F(QuadratureSupportTest, TensorTransformations_ContravariantTransform) {
    // Test contravariant transformation of rank-2 tensor
    Matrix<double, 2, 2> T_ref;
    T_ref(0, 0) = 1.0; T_ref(0, 1) = 0.0;
    T_ref(1, 0) = 0.0; T_ref(1, 1) = 1.0;

    Matrix<double, 2, 2> J;
    J(0, 0) = 2.0; J(0, 1) = 0.0;
    J(1, 0) = 0.0; J(1, 1) = 3.0;

    auto T_phys = TensorTransformations<double, 2>::contravariant_transform(T_ref, J);

    // T_phys = J * T_ref * J^T
    EXPECT_NEAR(T_phys(0, 0), 4.0, tolerance);  // 2^2
    EXPECT_NEAR(T_phys(0, 1), 0.0, tolerance);
    EXPECT_NEAR(T_phys(1, 0), 0.0, tolerance);
    EXPECT_NEAR(T_phys(1, 1), 9.0, tolerance);  // 3^2
}

// =============================================================================
// Surface Integration Tests
// =============================================================================

TEST_F(QuadratureSupportTest, SurfaceIntegration2D_LineElement) {
    // 2D surface is a line
    Matrix<double, 2, 1> tangent_vectors;
    tangent_vectors(0, 0) = 3.0;  // Tangent in x
    tangent_vectors(1, 0) = 4.0;  // Tangent in y

    auto [normal, area] = SurfaceIntegration<double, 2>::compute_normal_and_area(tangent_vectors);

    // Normal should be perpendicular to tangent
    EXPECT_NEAR(normal[0] * tangent_vectors(0, 0) + normal[1] * tangent_vectors(1, 0),
                0.0, tolerance);

    // Normal should be unit vector
    EXPECT_NEAR(normal.norm(), 1.0, tolerance);

    // Area element should be length of tangent
    EXPECT_NEAR(area, 5.0, tolerance);  // sqrt(3^2 + 4^2)
}

TEST_F(QuadratureSupportTest, SurfaceIntegration3D_TriangleElement) {
    // 3D surface is a triangle
    Matrix<double, 3, 2> tangent_vectors;
    tangent_vectors(0, 0) = 1.0; tangent_vectors(0, 1) = 0.0;  // First tangent
    tangent_vectors(1, 0) = 0.0; tangent_vectors(1, 1) = 1.0;  // Second tangent
    tangent_vectors(2, 0) = 0.0; tangent_vectors(2, 1) = 0.0;

    auto [normal, area] = SurfaceIntegration<double, 3>::compute_normal_and_area(tangent_vectors);

    // Normal should be (0, 0, 1) for xy-plane triangle
    EXPECT_NEAR(normal[0], 0.0, tolerance);
    EXPECT_NEAR(normal[1], 0.0, tolerance);
    EXPECT_NEAR(normal[2], 1.0, tolerance);

    // Area element should be 1 (unit parallelogram)
    EXPECT_NEAR(area, 1.0, tolerance);
}

TEST_F(QuadratureSupportTest, SurfaceJacobian) {
    // Compute surface Jacobian for a quadrilateral face
    Matrix<double, 1, 4> grad_shape;  // 1D shape function gradients on edge
    grad_shape(0, 0) = -0.5; grad_shape(0, 1) = 0.5;
    grad_shape(0, 2) = 0.5;  grad_shape(0, 3) = -0.5;

    Matrix<double, 2, 4> node_coords;  // 2D coordinates of edge nodes
    node_coords(0, 0) = 0.0; node_coords(1, 0) = 0.0;
    node_coords(0, 1) = 2.0; node_coords(1, 1) = 0.0;
    node_coords(0, 2) = 2.0; node_coords(1, 2) = 1.0;
    node_coords(0, 3) = 0.0; node_coords(1, 3) = 1.0;

    auto J_surf = SurfaceIntegration<double, 2>::compute_surface_jacobian(grad_shape, node_coords);

    // Check dimensions
    EXPECT_EQ(J_surf.rows(), 2);
    EXPECT_EQ(J_surf.cols(), 1);

    // Should give tangent vector along the edge
    double length = std::sqrt(J_surf(0, 0) * J_surf(0, 0) + J_surf(1, 0) * J_surf(1, 0));
    EXPECT_GT(length, 0.0);
}

// =============================================================================
// Edge Cases and Numerical Stability Tests
// =============================================================================

TEST_F(QuadratureSupportTest, NearSingularJacobian) {
    // Test with nearly singular Jacobian
    Matrix<double, 2, 2> J;
    J(0, 0) = 1e-10; J(0, 1) = 0.0;
    J(1, 0) = 0.0;   J(1, 1) = 1.0;

    double det_J = JacobianOperations<double, 2>::determinant(J);
    EXPECT_NEAR(det_J, 1e-10, 1e-20);

    // Should still be considered invalid with default tolerance
    EXPECT_FALSE((JacobianOperations<double, 2>::is_valid(J)));

    // But valid with looser tolerance
    EXPECT_TRUE((JacobianOperations<double, 2>::is_valid(J, 1e-15)));

    // Condition number should be very large
    double cond = JacobianOperations<double, 2>::condition_number(J);
    EXPECT_GT(cond, 1e9);
}

TEST_F(QuadratureSupportTest, NegativeJacobian) {
    // Test with negative Jacobian (inverted element)
    Matrix<double, 2, 2> J;
    J(0, 0) = -1.0; J(0, 1) = 0.0;
    J(1, 0) = 0.0;  J(1, 1) = 1.0;

    double det_J = JacobianOperations<double, 2>::determinant(J);
    EXPECT_NEAR(det_J, -1.0, tolerance);

    // Should be invalid (negative determinant)
    EXPECT_FALSE((JacobianOperations<double, 2>::is_valid(J)));

    // But integration weight uses absolute value
    double int_weight = JacobianOperations<double, 2>::integration_weight(det_J, 1.0);
    EXPECT_NEAR(int_weight, 1.0, tolerance);
}

TEST_F(QuadratureSupportTest, InvalidInputSizes) {
    // Test error handling for mismatched sizes
    std::vector<double> values{1.0, 2.0, 3.0};
    std::vector<double> weights{1.0, 1.0};  // Wrong size

    EXPECT_THROW(IntegrationHelpers<double>::integrate_1d(values, weights),
                 std::invalid_argument);

    EXPECT_THROW(IntegrationHelpers<double>::compute_element_volume(values, weights),
                 std::invalid_argument);
}

// =============================================================================
// Property-Based Tests
// =============================================================================

TEST_F(QuadratureSupportTest, PropertyBased_JacobianInvariance) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int iter = 0; iter < 100; ++iter) {
        // Random 2x2 matrix (ensure non-singular)
        Matrix<double, 2, 2> J;
        J(0, 0) = dist(rng) + 2.0;  // Ensure positive diagonal
        J(0, 1) = dist(rng) * 0.5;  // Small off-diagonal
        J(1, 0) = dist(rng) * 0.5;
        J(1, 1) = dist(rng) + 2.0;

        // Check fundamental properties
        double det_J = JacobianOperations<double, 2>::determinant(J);
        auto J_inv = JacobianOperations<double, 2>::inverse(J);

        // J * J^{-1} = I
        auto I = J * J_inv;
        EXPECT_NEAR(I(0, 0), 1.0, loose_tolerance);
        EXPECT_NEAR(I(0, 1), 0.0, loose_tolerance);
        EXPECT_NEAR(I(1, 0), 0.0, loose_tolerance);
        EXPECT_NEAR(I(1, 1), 1.0, loose_tolerance);

        // det(J^{-1}) = 1/det(J)
        double det_J_inv = J_inv.determinant();
        EXPECT_NEAR(det_J_inv, 1.0 / det_J, loose_tolerance);

        // Gradient transformation consistency
        Vector<double, 2> grad_ref{dist(rng), dist(rng)};
        auto grad_phys = JacobianOperations<double, 2>::transform_gradient(grad_ref, J_inv);
        auto grad_back = J.transpose() * grad_phys;
        EXPECT_NEAR(grad_back[0], grad_ref[0], loose_tolerance);
        EXPECT_NEAR(grad_back[1], grad_ref[1], loose_tolerance);
    }
}

TEST_F(QuadratureSupportTest, PropertyBased_IntegrationConsistency) {
    std::uniform_real_distribution<double> dist(0.1, 2.0);

    // Test that tensor product integration is consistent
    for (int iter = 0; iter < 50; ++iter) {
        // Generate random function values on 2x2 grid
        std::vector<double> values(4);
        for (auto& v : values) {
            v = dist(rng);
        }

        // Equal weights
        std::array<std::vector<double>, 2> weights;
        weights[0] = {1.0, 1.0};
        weights[1] = {1.0, 1.0};
        std::array<std::size_t, 2> dims{2, 2};

        double result_nd = IntegrationHelpers<double>::integrate_nd(values, weights, dims);

        // Manual computation
        double result_manual = 0.0;
        for (const auto& v : values) {
            result_manual += v;
        }

        EXPECT_NEAR(result_nd, result_manual, tolerance);
    }
}

TEST_F(QuadratureSupportTest, PropertyBased_MappingConsistency) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int iter = 0; iter < 50; ++iter) {
        // Random point in reference square
        double xi = dist(rng);
        double eta = dist(rng);

        // Map to triangle and check bounds
        auto p_tri = ReferenceElementMappings<double>::square_to_triangle(xi, eta);

        // Should be in unit triangle (with some tolerance for numerical errors)
        EXPECT_GE(p_tri[0], -1e-10);
        EXPECT_GE(p_tri[1], -1e-10);
        EXPECT_LE(p_tri[0] + p_tri[1], 1.0 + 1e-10);

        // Check Jacobian is positive
        double jac = ReferenceElementMappings<double>::square_to_triangle_jacobian(xi, eta);
        EXPECT_GT(jac, 0.0);
    }
}
