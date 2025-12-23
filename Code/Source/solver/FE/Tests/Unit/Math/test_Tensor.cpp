/**
 * @file test_Tensor.cpp
 * @brief Unit tests for Tensor.h - rank-2 and rank-4 tensors
 */

#include <gtest/gtest.h>
#include "FE/Math/Tensor.h"
#include "FE/Math/Vector.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/MathConstants.h"
#include <cmath>
#include <random>
#include <chrono>

using namespace svmp::FE::math;

// Test fixture for Tensor tests
class TensorTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-14;
    std::mt19937 rng{42};

    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Tensor2 (Rank-2 Tensor) Tests
// =============================================================================

TEST_F(TensorTest, Tensor2DefaultConstruction) {
    Tensor2<double, 3> T;

    // Should be zero-initialized
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(T(i, j), 0.0);
        }
    }
}

TEST_F(TensorTest, Tensor2ElementAccess) {
    Tensor2<double, 3> T;

    // Set elements
    T(0, 0) = 1.0;
    T(0, 1) = 2.0;
    T(1, 0) = 3.0;
    T(1, 1) = 4.0;

    // Get elements
    EXPECT_EQ(T(0, 0), 1.0);
    EXPECT_EQ(T(0, 1), 2.0);
    EXPECT_EQ(T(1, 0), 3.0);
    EXPECT_EQ(T(1, 1), 4.0);
}

TEST_F(TensorTest, Tensor2FromMatrix) {
    Matrix<double, 3, 3> M{{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};

    Tensor2<double, 3> T(M);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(T(i, j), M(i, j));
        }
    }
}

TEST_F(TensorTest, Tensor2ToMatrix) {
    Tensor2<double, 3> T;
    T(0, 0) = 1.0; T(0, 1) = 2.0; T(0, 2) = 3.0;
    T(1, 0) = 4.0; T(1, 1) = 5.0; T(1, 2) = 6.0;
    T(2, 0) = 7.0; T(2, 1) = 8.0; T(2, 2) = 9.0;

    Matrix<double, 3, 3> M = T.to_matrix();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(M(i, j), T(i, j));
        }
    }
}

TEST_F(TensorTest, Tensor2Identity) {
    Tensor2<double, 3> I = Tensor2<double, 3>::identity();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_EQ(I(i, j), expected);
        }
    }
}

TEST_F(TensorTest, Tensor2Arithmetic) {
    Tensor2<double, 2> A, B;
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;

    B(0, 0) = 5.0; B(0, 1) = 6.0;
    B(1, 0) = 7.0; B(1, 1) = 8.0;

    // Addition
    Tensor2<double, 2> C = A + B;
    EXPECT_EQ(C(0, 0), 6.0);
    EXPECT_EQ(C(0, 1), 8.0);
    EXPECT_EQ(C(1, 0), 10.0);
    EXPECT_EQ(C(1, 1), 12.0);

    // Subtraction
    Tensor2<double, 2> D = B - A;
    EXPECT_EQ(D(0, 0), 4.0);
    EXPECT_EQ(D(0, 1), 4.0);
    EXPECT_EQ(D(1, 0), 4.0);
    EXPECT_EQ(D(1, 1), 4.0);

    // Scalar multiplication
    Tensor2<double, 2> E = 2.0 * A;
    EXPECT_EQ(E(0, 0), 2.0);
    EXPECT_EQ(E(0, 1), 4.0);
    EXPECT_EQ(E(1, 0), 6.0);
    EXPECT_EQ(E(1, 1), 8.0);
}

TEST_F(TensorTest, Tensor2Contraction) {
    Tensor2<double, 3> T;
    T(0, 0) = 1.0; T(1, 1) = 2.0; T(2, 2) = 3.0;

    Vector<double, 3> v{1.0, 2.0, 3.0};

    // T · v (contraction)
    Vector<double, 3> result = T.contract(v);

    EXPECT_EQ(result[0], 1.0);  // T(0,j) * v[j]
    EXPECT_EQ(result[1], 4.0);  // T(1,j) * v[j]
    EXPECT_EQ(result[2], 9.0);  // T(2,j) * v[j]
}

TEST_F(TensorTest, Tensor2DoubleContraction) {
    Tensor2<double, 2> A, B;
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;

    B(0, 0) = 2.0; B(0, 1) = 0.0;
    B(1, 0) = 0.0; B(1, 1) = 2.0;

    // A : B (double contraction)
    double result = A.double_contract(B);

    // A_ij * B_ij = 1*2 + 2*0 + 3*0 + 4*2 = 2 + 8 = 10
    EXPECT_EQ(result, 10.0);
}

TEST_F(TensorTest, Tensor2Transpose) {
    Tensor2<double, 3> T;
    T(0, 1) = 1.0;
    T(0, 2) = 2.0;
    T(1, 2) = 3.0;

    Tensor2<double, 3> Tt = T.transpose();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(Tt(i, j), T(j, i));
        }
    }
}

TEST_F(TensorTest, Tensor2Trace) {
    Tensor2<double, 3> T;
    T(0, 0) = 1.0;
    T(1, 1) = 2.0;
    T(2, 2) = 3.0;
    T(0, 1) = 4.0;  // Off-diagonal

    double tr = T.trace();
    EXPECT_EQ(tr, 6.0);  // 1 + 2 + 3
}

TEST_F(TensorTest, Tensor2Invariants) {
    Tensor2<double, 3> T;
    T(0, 0) = 2.0; T(0, 1) = 1.0; T(0, 2) = 0.0;
    T(1, 0) = 1.0; T(1, 1) = 3.0; T(1, 2) = 1.0;
    T(2, 0) = 0.0; T(2, 1) = 1.0; T(2, 2) = 4.0;

    auto [I1, I2, I3] = T.invariants();

    // First invariant = trace
    EXPECT_NEAR(I1, T.trace(), tolerance);

    // Third invariant = determinant
    Matrix<double, 3, 3> M = T.to_matrix();
    EXPECT_NEAR(I3, M.determinant(), std::abs(I3) * 1e-10);
}

TEST_F(TensorTest, Tensor2EigenvaluesRepeatedRoot) {
    // Diagonal tensor with a repeated eigenvalue: diag(2, 1, 1).
    Tensor2<double, 3> T;
    T(0, 0) = 2.0;
    T(1, 1) = 1.0;
    T(2, 2) = 1.0;

    auto eig = T.eigenvalues();
    EXPECT_NEAR(eig[0], 2.0, tolerance);
    EXPECT_NEAR(eig[1], 1.0, tolerance);
    EXPECT_NEAR(eig[2], 1.0, tolerance);

    // Cross-check against the eigendecomposition path.
    const auto [vals, vecs] = T.eigen_decomposition();
    // Note: repeated eigenvalues are numerically sensitive; allow a looser tolerance here.
    constexpr double decomp_tol = 1e-6;
    EXPECT_NEAR(vals[0], 2.0, decomp_tol);
    EXPECT_NEAR(vals[1], 1.0, decomp_tol);
    EXPECT_NEAR(vals[2], 1.0, decomp_tol);
}

// =============================================================================
// Tensor4 (Rank-4 Tensor) Tests
// =============================================================================

TEST_F(TensorTest, Tensor4DefaultConstruction) {
    Tensor4<double, 3> C;

    // Should be zero-initialized
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    EXPECT_EQ(C(i, j, k, l), 0.0);
                }
            }
        }
    }
}

TEST_F(TensorTest, Tensor4ElementAccess) {
    Tensor4<double, 2> C;

    C(0, 0, 0, 0) = 1.0;
    C(0, 1, 0, 1) = 2.0;
    C(1, 0, 1, 0) = 3.0;
    C(1, 1, 1, 1) = 4.0;

    EXPECT_EQ(C(0, 0, 0, 0), 1.0);
    EXPECT_EQ(C(0, 1, 0, 1), 2.0);
    EXPECT_EQ(C(1, 0, 1, 0), 3.0);
    EXPECT_EQ(C(1, 1, 1, 1), 4.0);
}

TEST_F(TensorTest, Tensor4IsotropicIdentity) {
    // Create isotropic 4th order identity tensor
    // C_ijkl = 0.5 * (δ_ik δ_jl + δ_il δ_jk)
    Tensor4<double, 3> I = Tensor4<double, 3>::symmetric_identity();

    // Test key components
    EXPECT_EQ(I(0, 0, 0, 0), 1.0);
    EXPECT_EQ(I(1, 1, 1, 1), 1.0);
    EXPECT_EQ(I(2, 2, 2, 2), 1.0);
    EXPECT_EQ(I(0, 1, 0, 1), 0.5);
    EXPECT_EQ(I(0, 1, 1, 0), 0.5);
}

TEST_F(TensorTest, Tensor4DoubleContraction) {
    // Test C : ε = σ (Hooke's law)
    Tensor4<double, 2> C;
    Tensor2<double, 2> epsilon;

    // Simple isotropic material
    double lambda = 2.0, mu = 1.0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    double delta_kl = (k == l) ? 1.0 : 0.0;
                    double delta_ik = (i == k) ? 1.0 : 0.0;
                    double delta_jl = (j == l) ? 1.0 : 0.0;
                    double delta_il = (i == l) ? 1.0 : 0.0;
                    double delta_jk = (j == k) ? 1.0 : 0.0;

                    C(i, j, k, l) = lambda * delta_ij * delta_kl +
                                   mu * (delta_ik * delta_jl + delta_il * delta_jk);
                }
            }
        }
    }

    // Apply strain
    epsilon(0, 0) = 0.01;
    epsilon(1, 1) = 0.02;
    epsilon(0, 1) = epsilon(1, 0) = 0.005;

    // Compute stress
    Tensor2<double, 2> sigma = C.double_contract(epsilon);

    // Check diagonal stresses
    double expected_s00 = lambda * (epsilon(0, 0) + epsilon(1, 1)) + 2 * mu * epsilon(0, 0);
    double expected_s11 = lambda * (epsilon(0, 0) + epsilon(1, 1)) + 2 * mu * epsilon(1, 1);
    double expected_s01 = 2 * mu * epsilon(0, 1);

    EXPECT_NEAR(sigma(0, 0), expected_s00, tolerance);
    EXPECT_NEAR(sigma(1, 1), expected_s11, tolerance);
    EXPECT_NEAR(sigma(0, 1), expected_s01, tolerance);
    EXPECT_NEAR(sigma(1, 0), expected_s01, tolerance);  // Symmetry
}

TEST_F(TensorTest, Tensor4MajorSymmetry) {
    Tensor4<double, 3> C;

    // Set some values
    C(0, 1, 2, 0) = 5.0;

    // Apply major symmetry
    C.apply_major_symmetry();

    // Check C_ijkl = C_klij
    EXPECT_EQ(C(0, 1, 2, 0), C(2, 0, 0, 1));
}

TEST_F(TensorTest, Tensor4MinorSymmetries) {
    Tensor4<double, 3> C;

    // Set some values (using valid indices 0, 1, 2 for dimension 3)
    C(0, 1, 1, 2) = 7.0;

    // Apply minor symmetries
    C.apply_minor_symmetries();

    // Check C_ijkl = C_jikl = C_ijlk
    EXPECT_EQ(C(0, 1, 1, 2), C(1, 0, 1, 2));  // First minor symmetry
    EXPECT_EQ(C(0, 1, 1, 2), C(0, 1, 2, 1));  // Second minor symmetry
}

// =============================================================================
// Tensor Operations Tests
// =============================================================================

TEST_F(TensorTest, TensorProduct) {
    Vector<double, 3> a{1.0, 2.0, 3.0};
    Vector<double, 3> b{4.0, 5.0, 6.0};

    Tensor2<double, 3> T = tensor_product(a, b);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(T(i, j), a[i] * b[j]);
        }
    }
}

TEST_F(TensorTest, SymmetricPart) {
    Tensor2<double, 3> T;
    T(0, 1) = 2.0;
    T(1, 0) = 4.0;
    T(0, 2) = 3.0;
    T(2, 0) = 5.0;

    Tensor2<double, 3> S = symmetric_part(T);

    // Check symmetry
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(S(i, j), S(j, i));
            EXPECT_EQ(S(i, j), 0.5 * (T(i, j) + T(j, i)));
        }
    }
}

TEST_F(TensorTest, SkewSymmetricPart) {
    Tensor2<double, 3> T;
    T(0, 1) = 2.0;
    T(1, 0) = 4.0;
    T(0, 2) = 3.0;
    T(2, 0) = 5.0;

    Tensor2<double, 3> W = skew_symmetric_part(T);

    // Check skew-symmetry
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(W(i, j), -W(j, i));
            EXPECT_EQ(W(i, j), 0.5 * (T(i, j) - T(j, i)));
        }
    }
}

TEST_F(TensorTest, Decomposition) {
    // Any tensor can be decomposed into symmetric and skew-symmetric parts
    Tensor2<double, 3> T;
    T(0, 0) = 1.0; T(0, 1) = 2.0; T(0, 2) = 3.0;
    T(1, 0) = 4.0; T(1, 1) = 5.0; T(1, 2) = 6.0;
    T(2, 0) = 7.0; T(2, 1) = 8.0; T(2, 2) = 9.0;

    Tensor2<double, 3> S = symmetric_part(T);
    Tensor2<double, 3> W = skew_symmetric_part(T);

    // T = S + W
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(T(i, j), S(i, j) + W(i, j), tolerance);
        }
    }
}

// =============================================================================
// Special Tensor Tests
// =============================================================================

TEST_F(TensorTest, StressTensor) {
    // Create a typical stress tensor
    Tensor2<double, 3> stress;
    stress(0, 0) = 100.0;  // σ_xx
    stress(1, 1) = 50.0;   // σ_yy
    stress(2, 2) = 75.0;   // σ_zz
    stress(0, 1) = stress(1, 0) = 25.0;  // σ_xy
    stress(1, 2) = stress(2, 1) = 30.0;  // σ_yz
    stress(0, 2) = stress(2, 0) = 20.0;  // σ_xz

    // Check symmetry
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(stress(i, j), stress(j, i));
        }
    }

    // Compute hydrostatic and deviatoric parts
    double p = stress.trace() / 3.0;  // Hydrostatic pressure
    Tensor2<double, 3> dev = stress - p * Tensor2<double, 3>::identity();

    // Deviatoric stress should have zero trace
    EXPECT_NEAR(dev.trace(), 0.0, tolerance);
}

TEST_F(TensorTest, StrainTensor) {
    // Create a strain tensor from displacement gradient
    Tensor2<double, 3> grad_u;
    grad_u(0, 0) = 0.001;
    grad_u(0, 1) = 0.0005;
    grad_u(1, 0) = 0.0003;
    grad_u(1, 1) = 0.002;

    // Small strain tensor: ε = 0.5 * (grad_u + grad_u^T)
    Tensor2<double, 3> strain = symmetric_part(grad_u);

    // Check symmetry
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(strain(i, j), strain(j, i));
        }
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST_F(TensorTest, PerformanceContraction) {
    Tensor2<double, 3> T = Tensor2<double, 3>::identity();
    Vector<double, 3> v{1.0, 2.0, 3.0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000000; ++i) {
        Vector<double, 3> result = T.contract(v);
        v[0] = result[0] * 0.999;  // Prevent optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Adjust threshold based on build type (debug builds are ~5x slower)
#ifdef NDEBUG
    const int threshold_us = 100000;  // 100ms for release
#else
    const int threshold_us = 500000;  // 500ms for debug
#endif
    EXPECT_LT(duration.count(), threshold_us) << "Tensor contraction should be < " << threshold_us/1000 << "ms for 1M iterations";
}

TEST_F(TensorTest, PerformanceDoubleContraction) {
    Tensor4<double, 3> C = Tensor4<double, 3>::symmetric_identity();
    Tensor2<double, 3> epsilon = Tensor2<double, 3>::identity();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100000; ++i) {
        Tensor2<double, 3> sigma = C.double_contract(epsilon);
        epsilon(0, 0) = sigma(0, 0) * 0.999;  // Prevent optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_LT(duration.count(), 200000) << "Double contraction should be < 200ms for 100K iterations";
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_F(TensorTest, LargeValueStability) {
    double large = 1e15;
    Tensor2<double, 3> T;
    T(0, 0) = large;
    T(1, 1) = large;
    T(2, 2) = large;

    double trace = T.trace();
    EXPECT_EQ(trace, 3.0 * large);

    // Scaling
    Tensor2<double, 3> T2 = T / large;
    EXPECT_NEAR(T2(0, 0), 1.0, tolerance);
    EXPECT_NEAR(T2(1, 1), 1.0, tolerance);
    EXPECT_NEAR(T2(2, 2), 1.0, tolerance);
}

TEST_F(TensorTest, SmallValueStability) {
    double tiny = 1e-15;
    Tensor2<double, 3> T;
    T(0, 1) = tiny;
    T(1, 0) = tiny;

    Tensor2<double, 3> T2 = T * 1e15;
    EXPECT_NEAR(T2(0, 1), 1.0, tolerance);
    EXPECT_NEAR(T2(1, 0), 1.0, tolerance);
}
