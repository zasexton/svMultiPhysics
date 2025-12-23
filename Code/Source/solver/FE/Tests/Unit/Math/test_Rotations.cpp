/**
 * @file test_Rotations.cpp
 * @brief Unit tests for Rotations.h - rotation matrices and transformations
 */

#include <gtest/gtest.h>
#include "FE/Math/Rotations.h"
#include "FE/Math/Vector.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/Tensor.h"
#include "FE/Math/MathConstants.h"
#include <limits>
#include <cmath>
#include <random>

using namespace svmp::FE::math;

// Test fixture for Rotation tests
class RotationsTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-14;
    static constexpr double loose_tolerance = 1e-10;

    void SetUp() override {
        // Initialize random number generator for property-based tests
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

    // Check if a matrix is orthogonal (R * R^T = I)
    template<typename T, std::size_t N>
    bool is_orthogonal(const Matrix<T, N, N>& R, T tol = tolerance) {
        auto I = Matrix<T, N, N>::identity();
        auto RRt = R * R.transpose();
        return matrix_approx_equal(RRt, I, tol);
    }

    // Check if determinant equals 1 (proper rotation)
    template<typename T, std::size_t N>
    bool is_proper_rotation(const Matrix<T, N, N>& R, T tol = tolerance) {
        return std::abs(R.determinant() - T(1)) < tol;
    }

    std::mt19937 rng;
};

// =============================================================================
// 2D Rotation Tests
// =============================================================================

TEST_F(RotationsTest, Rotation2D_DefaultConstructor) {
    Rotation2D<double> rot;
    EXPECT_NEAR(rot.cos(), 1.0, tolerance);
    EXPECT_NEAR(rot.sin(), 0.0, tolerance);
    EXPECT_NEAR(rot.angle(), 0.0, tolerance);

    // Identity rotation should not change vectors
    Vector2<double> v{3.0, 4.0};
    auto v_rot = rot.rotate(v);
    EXPECT_NEAR(v_rot[0], v[0], tolerance);
    EXPECT_NEAR(v_rot[1], v[1], tolerance);
}

TEST_F(RotationsTest, Rotation2D_FromAngle) {
    double angle = Constants<double>::pi / 4.0;  // 45 degrees
    Rotation2D<double> rot(angle);

    EXPECT_NEAR(rot.cos(), std::sqrt(2.0) / 2.0, tolerance);
    EXPECT_NEAR(rot.sin(), std::sqrt(2.0) / 2.0, tolerance);
    EXPECT_NEAR(rot.angle(), angle, tolerance);
}

TEST_F(RotationsTest, Rotation2D_SpecialAngles) {
    // 90 degrees
    Rotation2D<double> rot90(Constants<double>::half_pi);
    EXPECT_NEAR(rot90.cos(), 0.0, tolerance);
    EXPECT_NEAR(rot90.sin(), 1.0, tolerance);

    // 180 degrees
    Rotation2D<double> rot180(Constants<double>::pi);
    EXPECT_NEAR(rot180.cos(), -1.0, tolerance);
    EXPECT_NEAR(rot180.sin(), 0.0, tolerance);

    // 270 degrees (-90 degrees)
    Rotation2D<double> rot270(-Constants<double>::half_pi);
    EXPECT_NEAR(rot270.cos(), 0.0, tolerance);
    EXPECT_NEAR(rot270.sin(), -1.0, tolerance);
}

TEST_F(RotationsTest, Rotation2D_FromCosSin) {
    // Test normalization
    double c = 3.0, s = 4.0;  // Not normalized
    Rotation2D<double> rot(c, s);

    double norm = std::sqrt(c * c + s * s);
    EXPECT_NEAR(rot.cos(), c / norm, tolerance);
    EXPECT_NEAR(rot.sin(), s / norm, tolerance);

    // Should be normalized
    EXPECT_NEAR(rot.cos() * rot.cos() + rot.sin() * rot.sin(), 1.0, tolerance);
}

TEST_F(RotationsTest, Rotation2D_FromVector) {
    Vector2<double> v{3.0, 4.0};
    auto rot = Rotation2D<double>::from_vector(v);

    double norm = v.norm();
    EXPECT_NEAR(rot.cos(), v[0] / norm, tolerance);
    EXPECT_NEAR(rot.sin(), v[1] / norm, tolerance);

    // Zero vector should give identity
    Vector2<double> zero{0.0, 0.0};
    auto rot_zero = Rotation2D<double>::from_vector(zero);
    EXPECT_NEAR(rot_zero.cos(), 1.0, tolerance);
    EXPECT_NEAR(rot_zero.sin(), 0.0, tolerance);
}

TEST_F(RotationsTest, Rotation2D_Matrix) {
    double angle = Constants<double>::pi / 3.0;  // 60 degrees
    Rotation2D<double> rot(angle);
    auto R = rot.matrix();

    // Check matrix elements
    EXPECT_NEAR(R(0, 0), std::cos(angle), tolerance);
    EXPECT_NEAR(R(0, 1), -std::sin(angle), tolerance);
    EXPECT_NEAR(R(1, 0), std::sin(angle), tolerance);
    EXPECT_NEAR(R(1, 1), std::cos(angle), tolerance);

    // Check orthogonality
    EXPECT_TRUE(is_orthogonal(R));

    // Check determinant = 1
    EXPECT_TRUE(is_proper_rotation(R));
}

TEST_F(RotationsTest, Rotation2D_RotateVector) {
    // Rotate unit vector along x by 90 degrees
    Rotation2D<double> rot(Constants<double>::half_pi);
    Vector2<double> v{1.0, 0.0};
    auto v_rot = rot.rotate(v);

    EXPECT_NEAR(v_rot[0], 0.0, tolerance);
    EXPECT_NEAR(v_rot[1], 1.0, tolerance);

    // Rotate arbitrary vector
    Vector2<double> v2{3.0, 4.0};
    auto v2_rot = rot.rotate(v2);
    EXPECT_NEAR(v2_rot[0], -4.0, tolerance);
    EXPECT_NEAR(v2_rot[1], 3.0, tolerance);
}

TEST_F(RotationsTest, Rotation2D_InverseRotation) {
    double angle = Constants<double>::pi / 6.0;
    Rotation2D<double> rot(angle);
    Vector2<double> v{5.0, 3.0};

    auto v_rot = rot.rotate(v);
    auto v_back = rot.rotate_inverse(v_rot);

    EXPECT_NEAR(v_back[0], v[0], tolerance);
    EXPECT_NEAR(v_back[1], v[1], tolerance);

    // Check inverse object
    auto rot_inv = rot.inverse();
    auto v_inv = rot_inv.rotate(v_rot);
    EXPECT_NEAR(v_inv[0], v[0], tolerance);
    EXPECT_NEAR(v_inv[1], v[1], tolerance);
}

TEST_F(RotationsTest, Rotation2D_Composition) {
    double angle1 = Constants<double>::pi / 6.0;  // 30 degrees
    double angle2 = Constants<double>::pi / 4.0;  // 45 degrees
    Rotation2D<double> rot1(angle1);
    Rotation2D<double> rot2(angle2);

    auto rot_combined = rot1.compose(rot2);
    EXPECT_NEAR(rot_combined.angle(), angle1 + angle2, tolerance);

    // Test on vector
    Vector2<double> v{1.0, 0.0};
    auto v1 = rot2.rotate(v);
    auto v2 = rot1.rotate(v1);
    auto v_combined = rot_combined.rotate(v);

    EXPECT_NEAR(v_combined[0], v2[0], tolerance);
    EXPECT_NEAR(v_combined[1], v2[1], tolerance);
}

TEST_F(RotationsTest, Rotation2D_TensorRotation) {
    double angle = Constants<double>::pi / 4.0;
    Rotation2D<double> rot(angle);

    // Create a simple tensor
    Tensor2<double, 2> T;
    T(0, 0) = 1.0; T(0, 1) = 2.0;
    T(1, 0) = 2.0; T(1, 1) = 3.0;

    auto T_rot = rot.rotate_tensor(T);

    // Check that trace is invariant
    double trace_orig = T(0, 0) + T(1, 1);
    double trace_rot = T_rot(0, 0) + T_rot(1, 1);
    EXPECT_NEAR(trace_rot, trace_orig, tolerance);

    // Check symmetry is preserved
    EXPECT_NEAR(T_rot(0, 1), T_rot(1, 0), tolerance);
}

TEST_F(RotationsTest, Rotation2D_TensorRotation_NonSymmetric) {
    double angle = Constants<double>::pi / 6.0;
    Rotation2D<double> rot(angle);

    // Non-symmetric rank-2 tensor.
    Tensor2<double, 2> T;
    T(0, 0) = 1.0; T(0, 1) = 2.0;
    T(1, 0) = 3.0; T(1, 1) = 4.0;

    auto T_rot = rot.rotate_tensor(T);

    // Reference using explicit matrix multiplication: R*T*R^T.
    Matrix<double, 2, 2> Tm;
    Tm(0, 0) = T(0, 0); Tm(0, 1) = T(0, 1);
    Tm(1, 0) = T(1, 0); Tm(1, 1) = T(1, 1);

    auto R = rot.matrix();
    auto T_ref = R * Tm * R.transpose();

    EXPECT_NEAR(T_rot(0, 0), T_ref(0, 0), tolerance);
    EXPECT_NEAR(T_rot(0, 1), T_ref(0, 1), tolerance);
    EXPECT_NEAR(T_rot(1, 0), T_ref(1, 0), tolerance);
    EXPECT_NEAR(T_rot(1, 1), T_ref(1, 1), tolerance);
}

// =============================================================================
// 3D Rotation Tests
// =============================================================================

TEST_F(RotationsTest, Rotation3D_DefaultConstructor) {
    Rotation3D<double> rot;
    auto R = rot.matrix();

    // Should be identity
    auto I = Matrix<double, 3, 3>::identity();
    EXPECT_TRUE(matrix_approx_equal(R, I));

    // Test on vector
    Vector3<double> v{1.0, 2.0, 3.0};
    auto v_rot = rot.rotate(v);
    EXPECT_NEAR(v_rot[0], v[0], tolerance);
    EXPECT_NEAR(v_rot[1], v[1], tolerance);
    EXPECT_NEAR(v_rot[2], v[2], tolerance);
}

TEST_F(RotationsTest, Rotation3D_FromMatrix) {
    // Create a rotation matrix manually (rotation about z-axis)
    double angle = Constants<double>::pi / 3.0;
    Matrix<double, 3, 3> R;
    R(0, 0) = std::cos(angle);  R(0, 1) = -std::sin(angle); R(0, 2) = 0.0;
    R(1, 0) = std::sin(angle);  R(1, 1) = std::cos(angle);  R(1, 2) = 0.0;
    R(2, 0) = 0.0;              R(2, 1) = 0.0;              R(2, 2) = 1.0;

    Rotation3D<double> rot(R);
    auto R_result = rot.matrix();

    // Should be orthogonalized and normalized
    EXPECT_TRUE(is_orthogonal(R_result));
    EXPECT_TRUE(is_proper_rotation(R_result));
}

TEST_F(RotationsTest, Rotation3D_FromEulerAngles) {
    double phi = Constants<double>::pi / 6.0;    // 30 degrees about z
    double theta = Constants<double>::pi / 4.0;  // 45 degrees about y
    double psi = Constants<double>::pi / 3.0;    // 60 degrees about x

    auto rot = Rotation3D<double>::from_euler(phi, theta, psi);
    auto R = rot.matrix();

    EXPECT_TRUE(is_orthogonal(R));
    EXPECT_TRUE(is_proper_rotation(R));

    // Convert back to Euler angles
    auto euler = rot.to_euler();
    (void)euler;
    EXPECT_NEAR(euler[0], phi, loose_tolerance);
    EXPECT_NEAR(euler[1], theta, loose_tolerance);
    EXPECT_NEAR(euler[2], psi, loose_tolerance);
}

TEST_F(RotationsTest, Rotation3D_FromAxisAngle) {
    // Rotation about x-axis
    Vector3<double> axis_x{1.0, 0.0, 0.0};
    double angle = Constants<double>::pi / 2.0;
    auto rot_x = Rotation3D<double>::from_axis_angle(axis_x, angle);

    Vector3<double> v{0.0, 1.0, 0.0};
    auto v_rot = rot_x.rotate(v);
    EXPECT_NEAR(v_rot[0], 0.0, tolerance);
    EXPECT_NEAR(v_rot[1], 0.0, tolerance);
    EXPECT_NEAR(v_rot[2], 1.0, tolerance);

    // Rotation about arbitrary axis
    Vector3<double> axis{1.0, 1.0, 1.0};
    auto rot = Rotation3D<double>::from_axis_angle(axis, Constants<double>::pi / 3.0);
    auto R = rot.matrix();
    EXPECT_TRUE(is_orthogonal(R));
    EXPECT_TRUE(is_proper_rotation(R));
}

TEST_F(RotationsTest, Rotation3D_FromQuaternion) {
    // Identity quaternion
    Vector4<double> q_identity{1.0, 0.0, 0.0, 0.0};
    auto rot_identity = Rotation3D<double>::from_quaternion(q_identity);
    auto R_identity = rot_identity.matrix();
    auto I = Matrix<double, 3, 3>::identity();
    EXPECT_TRUE(matrix_approx_equal(R_identity, I));

    // 180 degree rotation about z-axis
    Vector4<double> q_180z{0.0, 0.0, 0.0, 1.0};
    auto rot_180z = Rotation3D<double>::from_quaternion(q_180z);
    Vector3<double> v{1.0, 0.0, 0.0};
    auto v_rot = rot_180z.rotate(v);
    EXPECT_NEAR(v_rot[0], -1.0, tolerance);
    EXPECT_NEAR(v_rot[1], 0.0, tolerance);
    EXPECT_NEAR(v_rot[2], 0.0, tolerance);
}

TEST_F(RotationsTest, Rotation3D_FromTwoVectors) {
    // Parallel vectors (identity)
    Vector3<double> v1{1.0, 0.0, 0.0};
    Vector3<double> v2{2.0, 0.0, 0.0};
    auto rot_parallel = Rotation3D<double>::from_two_vectors(v1, v2);
    auto R_parallel = rot_parallel.matrix();
    auto I = Matrix<double, 3, 3>::identity();
    EXPECT_TRUE(matrix_approx_equal(R_parallel, I, loose_tolerance));

    // Opposite vectors (180 degree rotation)
    Vector3<double> v3{1.0, 0.0, 0.0};
    Vector3<double> v4{-1.0, 0.0, 0.0};
    auto rot_opposite = Rotation3D<double>::from_two_vectors(v3, v4);
    auto v_test = rot_opposite.rotate(v3);
    EXPECT_NEAR(v_test[0], v4[0], tolerance);

    // General case
    Vector3<double> from{1.0, 0.0, 0.0};
    Vector3<double> to{0.0, 1.0, 0.0};
    auto rot = Rotation3D<double>::from_two_vectors(from, to);
    auto result = rot.rotate(from.normalized());
    auto expected = to.normalized();
    EXPECT_NEAR(result[0], expected[0], tolerance);
    EXPECT_NEAR(result[1], expected[1], tolerance);
    EXPECT_NEAR(result[2], expected[2], tolerance);
}

TEST_F(RotationsTest, Rotation3D_FromTwoVectors_NearlyOpposite) {
    Vector3<double> from{1.0, 0.0, 0.0};
    Vector3<double> to{-1.0, 1e-7, 0.0};  // Nearly opposite, but not exact.

    auto rot = Rotation3D<double>::from_two_vectors(from, to);
    auto result = rot.rotate(from.normalized());
    auto expected = to.normalized();

    EXPECT_NEAR(result.norm(), expected.norm(), loose_tolerance);
    EXPECT_NEAR(result.dot(expected), 1.0, 1e-12);
    EXPECT_TRUE(is_orthogonal(rot.matrix(), loose_tolerance));
    EXPECT_TRUE(is_proper_rotation(rot.matrix(), loose_tolerance));
}

TEST_F(RotationsTest, Rotation3D_ToEulerAngles) {
    // Test various angles
    double phi = 0.5;
    double theta = 0.3;
    double psi = 0.7;

    auto rot = Rotation3D<double>::from_euler(phi, theta, psi);
    auto euler = rot.to_euler();
    (void)euler;
// Create rotation from recovered angles
    auto rot2 = Rotation3D<double>::from_euler(euler[0], euler[1], euler[2]);
    auto R1 = rot.matrix();
    auto R2 = rot2.matrix();

    EXPECT_TRUE(matrix_approx_equal(R1, R2, loose_tolerance));
}

TEST_F(RotationsTest, Rotation3D_ToAxisAngle) {
    // Test known rotation
    Vector3<double> axis{0.0, 0.0, 1.0};
    double angle = Constants<double>::pi / 3.0;
    auto rot = Rotation3D<double>::from_axis_angle(axis, angle);

    auto [recovered_axis, recovered_angle] = rot.to_axis_angle();
    EXPECT_NEAR(recovered_angle, angle, tolerance);
    EXPECT_NEAR(std::abs(recovered_axis.dot(axis)), 1.0, tolerance);

    // Test identity
    Rotation3D<double> rot_identity;
    auto [axis_id, angle_id] = rot_identity.to_axis_angle();
    EXPECT_NEAR(angle_id, 0.0, tolerance);

    // Test 180 degree rotation
    auto rot_180 = Rotation3D<double>::from_axis_angle(axis, Constants<double>::pi);
    auto [axis_180, angle_180] = rot_180.to_axis_angle();
    EXPECT_NEAR(angle_180, Constants<double>::pi, tolerance);
    EXPECT_NEAR(std::abs(axis_180.dot(axis)), 1.0, tolerance);

    // Test 180 degree rotation about an arbitrary axis (exercises the pi-branch reconstruction)
    Vector3<double> axis_diag{1.0, 1.0, 1.0};
    auto rot_180_diag = Rotation3D<double>::from_axis_angle(axis_diag, Constants<double>::pi);
    auto [axis_180_diag, angle_180_diag] = rot_180_diag.to_axis_angle();
    EXPECT_NEAR(angle_180_diag, Constants<double>::pi, tolerance);
    EXPECT_NEAR(std::abs(axis_180_diag.dot(axis_diag.normalized())), 1.0, tolerance);
}

TEST_F(RotationsTest, Rotation3D_ToQuaternion) {
    // Test various rotations
    Vector3<double> axis{1.0, 1.0, 0.0};
    double angle = Constants<double>::pi / 4.0;
    auto rot = Rotation3D<double>::from_axis_angle(axis, angle);

    auto q = rot.to_quaternion();

    // Convert back to rotation
    auto rot2 = Rotation3D<double>::from_quaternion(q);
    auto R1 = rot.matrix();
    auto R2 = rot2.matrix();

    EXPECT_TRUE(matrix_approx_equal(R1, R2, tolerance));

    // Quaternion should be normalized
    double q_norm = q.norm();
    EXPECT_NEAR(q_norm, 1.0, tolerance);
}

TEST_F(RotationsTest, Rotation3D_VectorRotation) {
    // Test rotation of basis vectors
    double angle = Constants<double>::half_pi;

    // Rotation about z-axis
    Vector3<double> z_axis{0.0, 0.0, 1.0};
    auto rot_z = Rotation3D<double>::from_axis_angle(z_axis, angle);

    Vector3<double> x_unit{1.0, 0.0, 0.0};
    auto x_rot = rot_z.rotate(x_unit);
    EXPECT_NEAR(x_rot[0], 0.0, tolerance);
    EXPECT_NEAR(x_rot[1], 1.0, tolerance);
    EXPECT_NEAR(x_rot[2], 0.0, tolerance);

    // Test inverse rotation
    auto x_back = rot_z.rotate_inverse(x_rot);
    EXPECT_NEAR(x_back[0], x_unit[0], tolerance);
    EXPECT_NEAR(x_back[1], x_unit[1], tolerance);
    EXPECT_NEAR(x_back[2], x_unit[2], tolerance);
}

TEST_F(RotationsTest, Rotation3D_TensorRotation) {
    // Create a rotation
    Vector3<double> axis{1.0, 0.0, 0.0};
    double angle = Constants<double>::pi / 4.0;
    auto rot = Rotation3D<double>::from_axis_angle(axis, angle);

    // Create a rank-2 tensor
    Tensor2<double, 3> T;
    T(0, 0) = 1.0; T(0, 1) = 0.0; T(0, 2) = 0.0;
    T(1, 0) = 0.0; T(1, 1) = 2.0; T(1, 2) = 0.0;
    T(2, 0) = 0.0; T(2, 1) = 0.0; T(2, 2) = 3.0;

    auto T_rot = rot.rotate_tensor(T);

    // Check invariants
    double trace_orig = T(0, 0) + T(1, 1) + T(2, 2);
    double trace_rot = T_rot(0, 0) + T_rot(1, 1) + T_rot(2, 2);
    EXPECT_NEAR(trace_rot, trace_orig, tolerance);

    // Check symmetry preservation
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_NEAR(T_rot(i, j), T_rot(j, i), tolerance);
        }
    }
}

TEST_F(RotationsTest, Rotation3D_Rank4TensorRotation) {
    // Create a simple rotation
    Vector3<double> axis{0.0, 0.0, 1.0};
    double angle = Constants<double>::pi / 6.0;
    auto rot = Rotation3D<double>::from_axis_angle(axis, angle);

    // Create a rank-4 tensor (simplified elasticity tensor)
    Tensor4<double, 3> C;

    // Set some non-zero values
    C(0, 0, 0, 0) = 1.0;
    C(1, 1, 1, 1) = 2.0;
    C(2, 2, 2, 2) = 3.0;
    C(0, 1, 0, 1) = 0.5;
    C(0, 2, 0, 2) = 0.5;
    C(1, 2, 1, 2) = 0.5;

    auto C_rot = rot.rotate_tensor(C);

    // Check that rotation preserves tensor structure
    // The rotated tensor should still be a valid elasticity tensor
    // (This is a simplified check)
    EXPECT_TRUE(std::isfinite(C_rot(0, 0, 0, 0)));
    EXPECT_TRUE(std::isfinite(C_rot(1, 1, 1, 1)));
    EXPECT_TRUE(std::isfinite(C_rot(2, 2, 2, 2)));
}

TEST_F(RotationsTest, Rotation3D_Composition) {
    // Create two rotations
    Vector3<double> axis1{1.0, 0.0, 0.0};
    Vector3<double> axis2{0.0, 1.0, 0.0};
    double angle1 = Constants<double>::pi / 4.0;
    double angle2 = Constants<double>::pi / 3.0;

    auto rot1 = Rotation3D<double>::from_axis_angle(axis1, angle1);
    auto rot2 = Rotation3D<double>::from_axis_angle(axis2, angle2);

    auto rot_combined = rot1.compose(rot2);

    // Test on vector
    Vector3<double> v{1.0, 1.0, 1.0};
    auto v1 = rot2.rotate(v);
    auto v2 = rot1.rotate(v1);
    auto v_combined = rot_combined.rotate(v);

    EXPECT_NEAR(v_combined[0], v2[0], tolerance);
    EXPECT_NEAR(v_combined[1], v2[1], tolerance);
    EXPECT_NEAR(v_combined[2], v2[2], tolerance);
}

TEST_F(RotationsTest, Rotation3D_Inverse) {
    // Create a rotation
    Vector3<double> axis{1.0, 2.0, 3.0};
    double angle = 1.234;
    auto rot = Rotation3D<double>::from_axis_angle(axis, angle);

    auto rot_inv = rot.inverse();

    // R * R^{-1} = I
    auto rot_identity = rot.compose(rot_inv);
    auto R_identity = rot_identity.matrix();
    auto I = Matrix<double, 3, 3>::identity();
    EXPECT_TRUE(matrix_approx_equal(R_identity, I, tolerance));

    // Test on vector
    Vector3<double> v{5.0, 4.0, 3.0};
    auto v_rot = rot.rotate(v);
    auto v_back = rot_inv.rotate(v_rot);
    EXPECT_NEAR(v_back[0], v[0], tolerance);
    EXPECT_NEAR(v_back[1], v[1], tolerance);
    EXPECT_NEAR(v_back[2], v[2], tolerance);
}

TEST_F(RotationsTest, Rotation3D_Slerp) {
    // Create two rotations
    auto rot1 = Rotation3D<double>::from_euler(0.0, 0.0, 0.0);  // Identity
    auto rot2 = Rotation3D<double>::from_euler(0.0, 0.0, Constants<double>::half_pi);  // 90 degrees about z

    // Interpolate at t=0
    auto rot_0 = rot1.slerp(rot2, 0.0);
    auto R_0 = rot_0.matrix();
    auto R1 = rot1.matrix();
    EXPECT_TRUE(matrix_approx_equal(R_0, R1, tolerance));

    // Interpolate at t=1
    auto rot_1 = rot1.slerp(rot2, 1.0);
    auto R_1 = rot_1.matrix();
    auto R2 = rot2.matrix();
    EXPECT_TRUE(matrix_approx_equal(R_1, R2, tolerance));

    // Interpolate at t=0.5 (should be 45 degrees)
    auto rot_half = rot1.slerp(rot2, 0.5);
    Vector3<double> v{1.0, 0.0, 0.0};
    auto v_half = rot_half.rotate(v);
    double expected_angle = Constants<double>::pi / 4.0;
    EXPECT_NEAR(v_half[0], std::cos(expected_angle), loose_tolerance);
    EXPECT_NEAR(v_half[1], std::sin(expected_angle), loose_tolerance);
    EXPECT_NEAR(v_half[2], 0.0, tolerance);
}

TEST_F(RotationsTest, Rotation3D_Slerp_NearlyParallel) {
    auto rot1 = Rotation3D<double>::from_euler(0.0, 0.0, 0.0);  // Identity
    Vector3<double> axis{0.0, 0.0, 1.0};
    double angle = 0.01;  // Small angle => quaternion dot ~ cos(angle/2) > 0.9995
    auto rot2 = Rotation3D<double>::from_axis_angle(axis, angle);

    auto rot_half = rot1.slerp(rot2, 0.5);
    Vector3<double> v{1.0, 0.0, 0.0};
    auto v_half = rot_half.rotate(v);

    double expected_angle = 0.5 * angle;
    EXPECT_NEAR(v_half[0], std::cos(expected_angle), loose_tolerance);
    EXPECT_NEAR(v_half[1], std::sin(expected_angle), loose_tolerance);
    EXPECT_NEAR(v_half[2], 0.0, tolerance);
}

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST_F(RotationsTest, RotationBetweenVectors) {
    Vector3<double> from{1.0, 0.0, 0.0};
    Vector3<double> to{0.0, 1.0, 0.0};

    auto R = rotation_between_vectors(from, to);
    auto result = R * from;

    EXPECT_NEAR(result[0], to[0], tolerance);
    EXPECT_NEAR(result[1], to[1], tolerance);
    EXPECT_NEAR(result[2], to[2], tolerance);

    EXPECT_TRUE(is_orthogonal(R));
    EXPECT_TRUE(is_proper_rotation(R));
}

TEST_F(RotationsTest, CoordinateSystemRotation) {
    Vector3<double> x_new{1.0, 1.0, 0.0};  // New x-axis
    Vector3<double> y_new{-1.0, 1.0, 0.0}; // New y-axis

    auto R = coordinate_system_rotation(x_new, y_new);

    EXPECT_TRUE(is_orthogonal(R));
    EXPECT_TRUE(is_proper_rotation(R));

    // Check that the new axes are orthogonal
    Vector3<double> x_result{R(0, 0), R(1, 0), R(2, 0)};
    Vector3<double> y_result{R(0, 1), R(1, 1), R(2, 1)};
    Vector3<double> z_result{R(0, 2), R(1, 2), R(2, 2)};

    EXPECT_NEAR(x_result.dot(y_result), 0.0, tolerance);
    EXPECT_NEAR(x_result.dot(z_result), 0.0, tolerance);
    EXPECT_NEAR(y_result.dot(z_result), 0.0, tolerance);
}

// =============================================================================
// Edge Cases and Numerical Stability Tests
// =============================================================================

TEST_F(RotationsTest, GimbalLockHandling) {
    // Test gimbal lock situation (theta = ±90 degrees)
    double phi = 0.5;
    double theta = Constants<double>::half_pi;  // 90 degrees
    double psi = 0.3;

    auto rot = Rotation3D<double>::from_euler(phi, theta, psi);
    auto euler = rot.to_euler();
    (void)euler;
// In gimbal lock, we lose one degree of freedom
    // The important thing is that the rotation matrix is still valid
    auto R = rot.matrix();
    EXPECT_TRUE(is_orthogonal(R));
    EXPECT_TRUE(is_proper_rotation(R));
}

TEST_F(RotationsTest, SmallAngleStability) {
    // Test with very small angles
    double small_angle = 1e-10;

    Rotation2D<double> rot2d(small_angle);
    EXPECT_NEAR(rot2d.cos(), 1.0, tolerance);
    EXPECT_NEAR(rot2d.sin(), small_angle, tolerance);

    Vector3<double> axis{0.0, 0.0, 1.0};
    auto rot3d = Rotation3D<double>::from_axis_angle(axis, small_angle);
    auto R = rot3d.matrix();
    EXPECT_TRUE(is_orthogonal(R));
    EXPECT_TRUE(is_proper_rotation(R));
}

TEST_F(RotationsTest, LargeAngleStability) {
    // Test with large angles
    double large_angle = 100.0 * Constants<double>::pi;

    Rotation2D<double> rot2d(large_angle);
    // Should wrap around to equivalent angle
    double norm = rot2d.cos() * rot2d.cos() + rot2d.sin() * rot2d.sin();
    EXPECT_NEAR(norm, 1.0, tolerance);

    Vector3<double> axis{1.0, 0.0, 0.0};
    auto rot3d = Rotation3D<double>::from_axis_angle(axis, large_angle);
    auto R = rot3d.matrix();
    EXPECT_TRUE(is_orthogonal(R));
    EXPECT_TRUE(is_proper_rotation(R));
}

// =============================================================================
// Property-Based Tests
// =============================================================================

TEST_F(RotationsTest, RandomRotationProperties) {
    std::uniform_real_distribution<double> dist(-Constants<double>::pi,
                                                Constants<double>::pi);

    for (int i = 0; i < 100; ++i) {
        // Generate random Euler angles
        double phi = dist(rng);
        double theta = dist(rng) / 2.0;  // Keep theta in [-pi/2, pi/2]
        double psi = dist(rng);

        auto rot = Rotation3D<double>::from_euler(phi, theta, psi);
        auto R = rot.matrix();

        // Check fundamental properties
        EXPECT_TRUE(is_orthogonal(R, loose_tolerance));
        EXPECT_TRUE(is_proper_rotation(R, loose_tolerance));

        // Check that rotation preserves vector length
        Vector3<double> v{dist(rng), dist(rng), dist(rng)};
        auto v_rot = rot.rotate(v);
        EXPECT_NEAR(v_rot.norm(), v.norm(), loose_tolerance);

        // Check inverse
        auto v_back = rot.rotate_inverse(v_rot);
        EXPECT_NEAR(v_back[0], v[0], loose_tolerance);
        EXPECT_NEAR(v_back[1], v[1], loose_tolerance);
        EXPECT_NEAR(v_back[2], v[2], loose_tolerance);
    }
}

TEST_F(RotationsTest, QuaternionNormalization) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < 50; ++i) {
        // Generate random quaternion (not normalized)
        Vector4<double> q{dist(rng), dist(rng), dist(rng), dist(rng)};

        // Avoid zero quaternion
        if (q.norm() < 1e-10) {
            q[0] = 1.0;
        }

        auto rot = Rotation3D<double>::from_quaternion(q);
        auto R = rot.matrix();

        // Should still produce valid rotation
        EXPECT_TRUE(is_orthogonal(R, loose_tolerance));
        EXPECT_TRUE(is_proper_rotation(R, loose_tolerance));

        // Convert back to quaternion should be normalized
        auto q_back = rot.to_quaternion();
        EXPECT_NEAR(q_back.norm(), 1.0, tolerance);
    }
}
