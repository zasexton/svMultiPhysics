/**
 * @file test_Eigensolvers.cpp
 * @brief Unit tests for Eigensolvers.h - analytical eigensolvers for small matrices
 */

#include <gtest/gtest.h>
#include "FE/Math/Eigensolvers.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/Vector.h"
#include "FE/Math/MathConstants.h"
#include "FE/Math/Tensor.h"  // For von_mises_stress and other tensor operations
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstddef>

using namespace svmp::FE::math;

// Test fixture for Eigensolvers tests
class EigensolversTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-12;
    std::mt19937 rng{42};

    void SetUp() override {}
    void TearDown() override {}

    // Helper to verify eigenvalue/eigenvector pair
    template<size_t N>
    void verify_eigenpair(const Matrix<double, N, N>& A,
                         double eigenvalue,
                         const Vector<double, N>& eigenvector) {
        Vector<double, N> Av = A * eigenvector;
        Vector<double, N> lambda_v = eigenvalue * eigenvector;

        for (size_t i = 0; i < N; ++i) {
            EXPECT_NEAR(Av[i], lambda_v[i], tolerance)
                << "Failed for eigenvalue " << eigenvalue;
        }
    }
};

// =============================================================================
// 2x2 Symmetric Eigensolvers
// =============================================================================

TEST_F(EigensolversTest, Symmetric2x2Identity) {
    Matrix<double, 2, 2> I = Matrix<double, 2, 2>::identity();

    auto [eigenvalues, eigenvectors] = eigen_symmetric_2x2(I);

    // Identity matrix has eigenvalues 1, 1
    EXPECT_NEAR(eigenvalues[0], 1.0, tolerance);
    EXPECT_NEAR(eigenvalues[1], 1.0, tolerance);

    // Verify eigenvectors (any orthonormal basis works)
    for (std::size_t i = 0; i < 2; ++i) {
        Vector<double, 2> v = eigenvectors.col(i);
        verify_eigenpair(I, eigenvalues[static_cast<std::size_t>(i)], v);
        EXPECT_NEAR(v.norm(), 1.0, tolerance);  // Unit vectors
    }
}

TEST_F(EigensolversTest, Symmetric2x2Diagonal) {
    Matrix<double, 2, 2> D{{3.0, 0.0},
                           {0.0, 5.0}};

    auto [eigenvalues, eigenvectors] = eigen_symmetric_2x2(D);

    // Eigenvalues should be sorted in descending order
    EXPECT_NEAR(eigenvalues[0], 5.0, tolerance);
    EXPECT_NEAR(eigenvalues[1], 3.0, tolerance);

    // Eigenvectors should be standard basis
    // Eigenvalue 5.0 corresponds to [0, 1] direction
    // Eigenvalue 3.0 corresponds to [1, 0] direction
    Vector<double, 2> v1 = eigenvectors.col(0);  // For eigenvalue 5.0
    Vector<double, 2> v2 = eigenvectors.col(1);  // For eigenvalue 3.0

    EXPECT_NEAR(v1[0], 0.0, tolerance);
    EXPECT_NEAR(std::abs(v1[1]), 1.0, tolerance);  // [0, ±1] for eigenvalue 5.0
    EXPECT_NEAR(std::abs(v2[0]), 1.0, tolerance);  // [±1, 0] for eigenvalue 3.0
    EXPECT_NEAR(v2[1], 0.0, tolerance);
}

TEST_F(EigensolversTest, Symmetric2x2General) {
    Matrix<double, 2, 2> A{{4.0, 1.0},
                           {1.0, 3.0}};

    auto [eigenvalues, eigenvectors] = eigen_symmetric_2x2(A);

    // Analytical eigenvalues: (4+3 ± sqrt((4-3)^2 + 4))/2 = (7 ± sqrt(5))/2
    double lambda1 = (7.0 - std::sqrt(5.0)) / 2.0;  // Smaller eigenvalue
    double lambda2 = (7.0 + std::sqrt(5.0)) / 2.0;  // Larger eigenvalue

    // Eigenvalues should be in descending order
    EXPECT_NEAR(eigenvalues[0], lambda2, tolerance);  // Larger first
    EXPECT_NEAR(eigenvalues[1], lambda1, tolerance);  // Smaller second

    // Verify eigenvectors
    for (std::size_t i = 0; i < 2; ++i) {
        Vector<double, 2> v = eigenvectors.col(i);
        verify_eigenpair(A, eigenvalues[static_cast<std::size_t>(i)], v);
    }

    // Check orthogonality of eigenvectors
    Vector<double, 2> v1 = eigenvectors.col(0);
    Vector<double, 2> v2 = eigenvectors.col(1);
    EXPECT_NEAR(v1.dot(v2), 0.0, tolerance);
}

// =============================================================================
// 3x3 Symmetric Eigensolvers
// =============================================================================

TEST_F(EigensolversTest, Symmetric3x3Identity) {
    Matrix<double, 3, 3> I = Matrix<double, 3, 3>::identity();

    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(I);

    // All eigenvalues should be 1
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(eigenvalues[i], 1.0, tolerance);
    }

    // Verify eigenvectors form orthonormal basis
    for (std::size_t i = 0; i < 3; ++i) {
        Vector<double, 3> vi = eigenvectors.col(i);
        EXPECT_NEAR(vi.norm(), 1.0, tolerance);

        for (std::size_t j = i + 1; j < 3; ++j) {
            Vector<double, 3> vj = eigenvectors.col(j);
            EXPECT_NEAR(vi.dot(vj), 0.0, tolerance);
        }
    }
}

TEST_F(EigensolversTest, Symmetric3x3Diagonal) {
    Matrix<double, 3, 3> D{{2.0, 0.0, 0.0},
                           {0.0, 5.0, 0.0},
                           {0.0, 0.0, 3.0}};

    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(D);

    // Eigenvalues should be sorted in ascending order
    EXPECT_NEAR(eigenvalues[0], 2.0, tolerance);
    EXPECT_NEAR(eigenvalues[1], 3.0, tolerance);
    EXPECT_NEAR(eigenvalues[2], 5.0, tolerance);

    // Verify eigenpairs
    for (std::size_t i = 0; i < 3; ++i) {
        Vector<double, 3> v = eigenvectors.col(i);
        verify_eigenpair(D, eigenvalues[static_cast<std::size_t>(i)], v);
    }
}

TEST_F(EigensolversTest, Symmetric3x3StressMatrix) {
    // Typical stress matrix from FE computation
    Matrix<double, 3, 3> stress{{100.0, 30.0, 20.0},
                                {30.0, 150.0, 40.0},
                                {20.0, 40.0, 200.0}};

    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(stress);

    // Principal stresses should be positive and sorted
    EXPECT_GT(eigenvalues[0], 0.0);
    EXPECT_GT(eigenvalues[1], 0.0);
    EXPECT_GT(eigenvalues[2], 0.0);
    EXPECT_LE(eigenvalues[0], eigenvalues[1]);
    EXPECT_LE(eigenvalues[1], eigenvalues[2]);

    // Trace invariant: sum of eigenvalues = trace
    double trace = stress.trace();
    double eigen_sum = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    EXPECT_NEAR(eigen_sum, trace, tolerance);

    // Determinant invariant: product of eigenvalues = determinant
    double det = stress.determinant();
    double eigen_prod = eigenvalues[0] * eigenvalues[1] * eigenvalues[2];
    EXPECT_NEAR(eigen_prod, det, std::abs(det) * 1e-10);

    // Verify eigenpairs
    for (std::size_t i = 0; i < 3; ++i) {
        Vector<double, 3> v = eigenvectors.col(i);
        verify_eigenpair(stress, eigenvalues[static_cast<std::size_t>(i)], v);
    }
}

// =============================================================================
// Principal Stress/Strain Tests
// =============================================================================

TEST_F(EigensolversTest, PrincipalStresses) {
    // Create a stress tensor
    Matrix<double, 3, 3> sigma{{50.0, 20.0, 10.0},
                               {20.0, 30.0, 15.0},
                               {10.0, 15.0, 40.0}};

    auto [principal, directions] = principal_stresses(sigma);

    // Check sorted order
    EXPECT_LE(principal[0], principal[1]);
    EXPECT_LE(principal[1], principal[2]);

    // Verify they are eigenvalues
    for (std::size_t i = 0; i < 3; ++i) {
        Vector<double, 3> dir = directions.col(i);
        verify_eigenpair(sigma, principal[static_cast<std::size_t>(i)], dir);
    }
}

TEST_F(EigensolversTest, PrincipalStrains) {
    // Create a strain tensor
    Matrix<double, 3, 3> epsilon{{0.001, 0.0002, 0.0001},
                                 {0.0002, 0.0015, 0.0003},
                                 {0.0001, 0.0003, 0.002}};

    auto [principal, directions] = principal_strains(epsilon);

    // Check sorted order
    EXPECT_LE(principal[0], principal[1]);
    EXPECT_LE(principal[1], principal[2]);

    // Principal directions should be orthonormal
    for (std::size_t i = 0; i < 3; ++i) {
        Vector<double, 3> vi = directions.col(i);
        EXPECT_NEAR(vi.norm(), 1.0, tolerance);

        for (std::size_t j = i + 1; j < 3; ++j) {
            Vector<double, 3> vj = directions.col(j);
            EXPECT_NEAR(vi.dot(vj), 0.0, tolerance);
        }
    }
}

// =============================================================================
// Von Mises Stress Tests
// =============================================================================

TEST_F(EigensolversTest, VonMisesStressUniaxial) {
    // Uniaxial stress state
    Matrix<double, 3, 3> sigma = Matrix<double, 3, 3>::zero();
    sigma(0, 0) = 100.0;  // Only stress in x-direction

    double vm = von_mises_stress(sigma);

    // For uniaxial stress, von Mises = |σ_x|
    EXPECT_NEAR(vm, 100.0, tolerance);
}

TEST_F(EigensolversTest, VonMisesStressHydrostatic) {
    // Pure hydrostatic stress (should give zero von Mises)
    Matrix<double, 3, 3> sigma{{100.0, 0.0, 0.0},
                               {0.0, 100.0, 0.0},
                               {0.0, 0.0, 100.0}};

    double vm = von_mises_stress(sigma);

    // Hydrostatic stress has zero von Mises stress
    EXPECT_NEAR(vm, 0.0, tolerance);
}

TEST_F(EigensolversTest, VonMisesStressPureShear) {
    // Pure shear stress
    Matrix<double, 3, 3> sigma{{0.0, 50.0, 0.0},
                               {50.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0}};

    double vm = von_mises_stress(sigma);

    // For pure shear τ_xy, von Mises = √3 * |τ_xy|
    EXPECT_NEAR(vm, std::sqrt(3.0) * 50.0, tolerance);
}

TEST_F(EigensolversTest, VonMisesStressGeneral) {
    // General stress state
    Matrix<double, 3, 3> sigma{{100.0, 30.0, 20.0},
                               {30.0, 80.0, 10.0},
                               {20.0, 10.0, 60.0}};

    double vm = von_mises_stress(sigma);

    // Calculate using principal stresses
    auto [principal, dirs] = principal_stresses(sigma);
    double s1 = principal[2], s2 = principal[1], s3 = principal[0];

    double vm_from_principal = std::sqrt(
        0.5 * ((s1-s2)*(s1-s2) + (s2-s3)*(s2-s3) + (s3-s1)*(s3-s1))
    );

    EXPECT_NEAR(vm, vm_from_principal, tolerance);
}

// =============================================================================
// Stress Invariants Tests
// =============================================================================

TEST_F(EigensolversTest, StressInvariants) {
    Matrix<double, 3, 3> sigma{{50.0, 20.0, 10.0},
                               {20.0, 30.0, 15.0},
                               {10.0, 15.0, 40.0}};

    auto [I1, I2, I3] = stress_invariants(sigma);

    // I1 = trace
    EXPECT_NEAR(I1, sigma.trace(), tolerance);

    // I3 = determinant
    EXPECT_NEAR(I3, sigma.determinant(), std::abs(I3) * 1e-10);

    // Verify using characteristic polynomial
    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(sigma);
    double lambda1 = eigenvalues[0];
    double lambda2 = eigenvalues[1];
    double lambda3 = eigenvalues[2];

    // Check invariants match eigenvalues
    EXPECT_NEAR(I1, lambda1 + lambda2 + lambda3, tolerance);
    EXPECT_NEAR(I2, lambda1*lambda2 + lambda2*lambda3 + lambda3*lambda1,
                std::abs(I2) * 1e-10);
    EXPECT_NEAR(I3, lambda1 * lambda2 * lambda3, std::abs(I3) * 1e-10);
}

// =============================================================================
// Deviatoric Stress Tests
// =============================================================================

TEST_F(EigensolversTest, DeviatoricStress) {
    Matrix<double, 3, 3> sigma{{100.0, 30.0, 20.0},
                               {30.0, 80.0, 10.0},
                               {20.0, 10.0, 60.0}};

    Matrix<double, 3, 3> dev = deviatoric_stress(sigma);

    // Check trace of deviatoric stress is zero
    EXPECT_NEAR(dev.trace(), 0.0, tolerance);

    // Check symmetry is preserved
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(dev(i, j), dev(j, i), tolerance);
        }
    }

    // Verify decomposition: σ = σ_dev + σ_hydrostatic
    double p = sigma.trace() / 3.0;  // Hydrostatic pressure
    Matrix<double, 3, 3> hydro = p * Matrix<double, 3, 3>::identity();
    Matrix<double, 3, 3> reconstructed = dev + hydro;

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(reconstructed(i, j), sigma(i, j), tolerance);
        }
    }
}

// =============================================================================
// Special Cases and Edge Cases
// =============================================================================

TEST_F(EigensolversTest, ZeroMatrix) {
    Matrix<double, 3, 3> zero = Matrix<double, 3, 3>::zero();

    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(zero);

    // All eigenvalues should be zero
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(eigenvalues[i], 0.0, tolerance);
    }
}

TEST_F(EigensolversTest, RepeatedEigenvalues) {
    // Matrix with repeated eigenvalue
    Matrix<double, 3, 3> A{{2.0, 0.0, 0.0},
                           {0.0, 2.0, 0.0},
                           {0.0, 0.0, 5.0}};

    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(A);

    EXPECT_NEAR(eigenvalues[0], 2.0, tolerance);
    EXPECT_NEAR(eigenvalues[1], 2.0, tolerance);
    EXPECT_NEAR(eigenvalues[2], 5.0, tolerance);

    // Verify eigenpairs
    for (std::size_t i = 0; i < 3; ++i) {
        Vector<double, 3> v = eigenvectors.col(i);
        verify_eigenpair(A, eigenvalues[static_cast<std::size_t>(i)], v);
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST_F(EigensolversTest, Performance2x2) {
    Matrix<double, 2, 2> A{{4.0, 1.0},
                           {1.0, 3.0}};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000000; ++i) {
        auto [eigenvalues, eigenvectors] = eigen_symmetric_2x2(A);
        A(0, 0) = eigenvalues[0] * 0.999;  // Prevent optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Adjust threshold based on build type (debug builds are ~5x slower)
#ifdef NDEBUG
    const int threshold_us = 100000;  // 100ms for release
#else
    const int threshold_us = 500000;  // 500ms for debug
#endif
    EXPECT_LT(duration.count(), threshold_us) << "2x2 eigen should be < " << threshold_us/1000 << "ms for 1M iterations";
}

TEST_F(EigensolversTest, Performance3x3) {
    Matrix<double, 3, 3> A{{4.0, 1.0, 1.0},
                           {1.0, 3.0, 0.0},
                           {1.0, 0.0, 2.0}};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100000; ++i) {
        auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(A);
        A(0, 0) = eigenvalues[0] * 0.999;  // Prevent optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Adjust threshold based on build type (debug builds are ~5x slower)
#ifdef NDEBUG
    const int threshold_us = 100000;  // 100ms for release
#else
    const int threshold_us = 500000;  // 500ms for debug
#endif
    EXPECT_LT(duration.count(), threshold_us) << "3x3 eigen should be < " << threshold_us/1000 << "ms for 100K iterations";
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_F(EigensolversTest, NearlyDiagonal) {
    // Nearly diagonal matrix
    double eps = 1e-14;
    Matrix<double, 3, 3> A{{1.0, eps, 0.0},
                           {eps, 2.0, eps},
                           {0.0, eps, 3.0}};

    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(A);

    // Eigenvalues should be very close to diagonal elements
    EXPECT_NEAR(eigenvalues[0], 1.0, 1e-12);
    EXPECT_NEAR(eigenvalues[1], 2.0, 1e-12);
    EXPECT_NEAR(eigenvalues[2], 3.0, 1e-12);
}

TEST_F(EigensolversTest, LargeValueStability) {
    // Test with large values
    double scale = 1e10;
    Matrix<double, 3, 3> A{{2.0*scale, 0.5*scale, 0.0},
                           {0.5*scale, 3.0*scale, 1.0*scale},
                           {0.0, 1.0*scale, 4.0*scale}};

    auto [eigenvalues, eigenvectors] = eigen_symmetric_3x3(A);

    // Check relative accuracy of trace
    double trace = A.trace();
    double eigen_sum = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    EXPECT_NEAR(eigen_sum / trace, 1.0, 1e-10);
}
