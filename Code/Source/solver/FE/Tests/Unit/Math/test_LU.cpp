/**
 * @file test_LU.cpp
 * @brief Unit tests for LU.h - analytical solvers for small matrices
 */

#include <gtest/gtest.h>
#include "FE/Math/LU.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/Vector.h"
#include "FE/Math/MathConstants.h"
#include <cmath>
#include <random>
#include <chrono>

using namespace svmp::FE::math;

// Test fixture for LU tests
class LUTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-13;
    std::mt19937 rng{42};  // Fixed seed for reproducibility

    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// 2x2 Analytical Solver Tests
// =============================================================================

TEST_F(LUTest, Solve2x2Simple) {
    Matrix<double, 2, 2> A{{2.0, 1.0},
                           {1.0, 3.0}};
    Vector<double, 2> b{5.0, 6.0};

    Vector<double, 2> x = solve_2x2(A, b);

    // Verify solution: A*x = b
    Vector<double, 2> Ax = A * x;
    EXPECT_NEAR(Ax[0], b[0], tolerance);
    EXPECT_NEAR(Ax[1], b[1], tolerance);

    // Check exact solution
    EXPECT_NEAR(x[0], 9.0/5.0, tolerance);
    EXPECT_NEAR(x[1], 7.0/5.0, tolerance);
}

TEST_F(LUTest, Solve2x2Identity) {
    Matrix<double, 2, 2> I = Matrix<double, 2, 2>::identity();
    Vector<double, 2> b{3.0, 4.0};

    Vector<double, 2> x = solve_2x2(I, b);

    EXPECT_EQ(x[0], 3.0);
    EXPECT_EQ(x[1], 4.0);
}

TEST_F(LUTest, Solve2x2Diagonal) {
    Matrix<double, 2, 2> D{{2.0, 0.0},
                           {0.0, 3.0}};
    Vector<double, 2> b{4.0, 9.0};

    Vector<double, 2> x = solve_2x2(D, b);

    EXPECT_EQ(x[0], 2.0);
    EXPECT_EQ(x[1], 3.0);
}

TEST_F(LUTest, Solve2x2Singular) {
    Matrix<double, 2, 2> A{{1.0, 2.0},
                           {2.0, 4.0}};  // Singular (det = 0)
    Vector<double, 2> b{3.0, 6.0};

    EXPECT_THROW(solve_2x2(A, b), std::runtime_error);
}

TEST_F(LUTest, Solve2x2NearSingular) {
    double eps = 1e-15;
    Matrix<double, 2, 2> A{{1.0, 1.0},
                           {1.0, 1.0 + eps}};
    Vector<double, 2> b{2.0, 2.0 + eps};

    Vector<double, 2> x = solve_2x2(A, b);

    // Solution should be approximately [1, 1]
    // For ill-conditioned matrix with det~1e-15, expect reduced accuracy
    // Condition number ~ 1/det ~ 1e15, so accuracy ~ eps * cond ~ 1e-16 * 1e15 = 1e-1
    EXPECT_NEAR(x[0], 1.0, 0.5);
    EXPECT_NEAR(x[1], 1.0, 0.5);
}

// =============================================================================
// 3x3 Analytical Solver Tests
// =============================================================================

TEST_F(LUTest, Solve3x3Simple) {
    Matrix<double, 3, 3> A{{2.0, -1.0, 0.0},
                           {-1.0, 2.0, -1.0},
                           {0.0, -1.0, 2.0}};
    Vector<double, 3> b{1.0, 0.0, 1.0};

    Vector<double, 3> x = solve_3x3(A, b);

    // Verify solution
    Vector<double, 3> Ax = A * x;
    EXPECT_NEAR(Ax[0], b[0], tolerance);
    EXPECT_NEAR(Ax[1], b[1], tolerance);
    EXPECT_NEAR(Ax[2], b[2], tolerance);
}

TEST_F(LUTest, Solve3x3Identity) {
    Matrix<double, 3, 3> I = Matrix<double, 3, 3>::identity();
    Vector<double, 3> b{1.0, 2.0, 3.0};

    Vector<double, 3> x = solve_3x3(I, b);

    EXPECT_EQ(x[0], 1.0);
    EXPECT_EQ(x[1], 2.0);
    EXPECT_EQ(x[2], 3.0);
}

TEST_F(LUTest, Solve3x3Jacobian) {
    // Typical Jacobian matrix from FE computation
    Matrix<double, 3, 3> J{{2.0, 0.5, 0.0},
                           {0.5, 2.0, 0.5},
                           {0.0, 0.5, 2.0}};
    Vector<double, 3> grad{1.0, 1.0, 1.0};

    Vector<double, 3> x = solve_3x3(J, grad);

    // Verify solution
    Vector<double, 3> Jx = J * x;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(Jx[i], grad[i], tolerance);
    }
}

TEST_F(LUTest, Solve3x3Singular) {
    Matrix<double, 3, 3> A{{1.0, 2.0, 3.0},
                           {4.0, 5.0, 6.0},
                           {7.0, 8.0, 9.0}};  // Singular matrix
    Vector<double, 3> b{1.0, 2.0, 3.0};

    EXPECT_THROW(solve_3x3(A, b), std::runtime_error);
}

// =============================================================================
// 4x4 Gauss Elimination Tests
// =============================================================================

TEST_F(LUTest, Solve4x4Simple) {
    Matrix<double, 4, 4> A{{4.0, -1.0, 0.0, 0.0},
                           {-1.0, 4.0, -1.0, 0.0},
                           {0.0, -1.0, 4.0, -1.0},
                           {0.0, 0.0, -1.0, 4.0}};
    Vector<double, 4> b{3.0, 2.0, 2.0, 3.0};

    Vector<double, 4> x = solve_4x4(A, b);

    // Verify solution
    Vector<double, 4> Ax = A * x;
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(Ax[i], b[i], tolerance);
    }
}

TEST_F(LUTest, Solve4x4Diagonal) {
    Matrix<double, 4, 4> D = Matrix<double, 4, 4>::zero();
    D(0, 0) = 1.0;
    D(1, 1) = 2.0;
    D(2, 2) = 3.0;
    D(3, 3) = 4.0;

    Vector<double, 4> b{1.0, 4.0, 9.0, 16.0};

    Vector<double, 4> x = solve_4x4(D, b);

    EXPECT_EQ(x[0], 1.0);
    EXPECT_EQ(x[1], 2.0);
    EXPECT_EQ(x[2], 3.0);
    EXPECT_EQ(x[3], 4.0);
}

// =============================================================================
// General NxN Solver Tests (for N <= 4)
// =============================================================================

TEST_F(LUTest, SolveGeneral2x2) {
    Matrix<double, 2, 2> A{{3.0, 1.0},
                           {1.0, 2.0}};
    Vector<double, 2> b{9.0, 8.0};

    Vector<double, 2> x = solve(A, b);

    Vector<double, 2> Ax = A * x;
    EXPECT_NEAR(Ax[0], b[0], tolerance);
    EXPECT_NEAR(Ax[1], b[1], tolerance);
}

TEST_F(LUTest, SolveGeneral3x3) {
    Matrix<double, 3, 3> A{{3.0, 1.0, 1.0},
                           {1.0, 3.0, 1.0},
                           {1.0, 1.0, 3.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    Vector<double, 3> x = solve(A, b);

    Vector<double, 3> Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(Ax[i], b[i], tolerance);
    }
}

TEST_F(LUTest, SolveGeneral4x4) {
    Matrix<double, 4, 4> A{{5.0, 1.0, 0.0, 1.0},
                           {1.0, 5.0, 1.0, 0.0},
                           {0.0, 1.0, 5.0, 1.0},
                           {1.0, 0.0, 1.0, 5.0}};
    Vector<double, 4> b{7.0, 7.0, 7.0, 7.0};

    Vector<double, 4> x = solve(A, b);

    Vector<double, 4> Ax = A * x;
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(Ax[i], b[i], tolerance);
    }
}

// =============================================================================
// LU Factorization Tests
// =============================================================================

TEST_F(LUTest, LUFactorization3x3) {
    Matrix<double, 3, 3> A{{2.0, 1.0, 1.0},
                           {4.0, -6.0, 0.0},
                           {-2.0, 7.0, 2.0}};

    auto [L, U, P] = lu_factorize(A);

    // Verify P*A = L*U
    Matrix<double, 3, 3> PA = P * A;
    Matrix<double, 3, 3> LU_result = L * U;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(PA(i, j), LU_result(i, j), tolerance);
        }
    }

    // Verify L is lower triangular with unit diagonal
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(L(i, i), 1.0);
        for (size_t j = i + 1; j < 3; ++j) {
            EXPECT_EQ(L(i, j), 0.0);
        }
    }

    // Verify U is upper triangular
    for (size_t i = 1; i < 3; ++i) {
        for (size_t j = 0; j < i; ++j) {
            EXPECT_NEAR(U(i, j), 0.0, tolerance);
        }
    }
}

TEST_F(LUTest, LUSolve) {
    Matrix<double, 3, 3> A{{1.0, 2.0, 3.0},
                           {2.0, 5.0, 3.0},
                           {1.0, 0.0, 8.0}};
    Vector<double, 3> b{1.0, 2.0, 3.0};

    // Factorize
    auto [L, U, P] = lu_factorize(A);

    // Solve using factorization
    Vector<double, 3> x = lu_solve(L, U, P, b);

    // Verify solution
    Vector<double, 3> Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(Ax[i], b[i], tolerance);
    }
}

TEST_F(LUTest, LUDecompositionPivotingPermutesRHS) {
    // Matrix requiring row pivoting at the first step (A(0,0)=0, A(1,0)=1).
    // This is a permutation matrix swapping the first two unknowns.
    Matrix<double, 5, 5> A = Matrix<double, 5, 5>::zero();
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    for (std::size_t i = 2; i < 5; ++i) {
        A(i, i) = 1.0;
    }

    Vector<double, 5> b{1.0, 2.0, 3.0, 4.0, 5.0};

    LUDecomposition<double, 5> lu(A);
    Vector<double, 5> x = lu.solve(b);

    // Exact solution for this permutation matrix.
    EXPECT_EQ(x[0], 2.0);
    EXPECT_EQ(x[1], 1.0);
    EXPECT_EQ(x[2], 3.0);
    EXPECT_EQ(x[3], 4.0);
    EXPECT_EQ(x[4], 5.0);

    // Verify residual A*x = b.
    Vector<double, 5> Ax = A * x;
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(Ax[i], b[i], tolerance);
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST_F(LUTest, Performance2x2) {
    Matrix<double, 2, 2> A{{2.0, 1.0},
                           {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000000; ++i) {
        Vector<double, 2> x = solve_2x2(A, b);
        A(0, 0) = x[0] * 0.999;  // Prevent optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Adjust threshold based on build type (debug builds are ~5x slower)
#ifdef NDEBUG
    const int threshold_us = 50000;  // 50ms for release
#else
    const int threshold_us = 250000;  // 250ms for debug
#endif
    EXPECT_LT(duration.count(), threshold_us) << "2x2 solve should be < " << threshold_us/1000 << "ms for 1M iterations";
}

TEST_F(LUTest, Performance3x3) {
    Matrix<double, 3, 3> A{{2.0, 1.0, 0.0},
                           {1.0, 3.0, 1.0},
                           {0.0, 1.0, 2.0}};
    Vector<double, 3> b{1.0, 2.0, 3.0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100000; ++i) {
        Vector<double, 3> x = solve_3x3(A, b);
        A(0, 0) = x[0] * 0.999;  // Prevent optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Adjust threshold based on build type (debug builds are ~5x slower)
#ifdef NDEBUG
    const int threshold_us = 50000;  // 50ms for release
#else
    const int threshold_us = 250000;  // 250ms for debug
#endif
    EXPECT_LT(duration.count(), threshold_us) << "3x3 solve should be < " << threshold_us/1000 << "ms for 100K iterations";
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_F(LUTest, IllConditioned2x2) {
    // Hilbert matrix (ill-conditioned)
    Matrix<double, 2, 2> H{{1.0, 1.0/2.0},
                           {1.0/2.0, 1.0/3.0}};
    Vector<double, 2> b{1.0, 1.0};

    Vector<double, 2> x = solve_2x2(H, b);

    // Verify solution accuracy despite ill-conditioning
    Vector<double, 2> Hx = H * x;
    EXPECT_NEAR(Hx[0], b[0], 1e-12);
    EXPECT_NEAR(Hx[1], b[1], 1e-12);
}

TEST_F(LUTest, IllConditioned3x3) {
    // Ill-conditioned (not singular) matrix
    // Use a different structure to avoid catastrophic cancellation
    double eps = 1e-12;
    Matrix<double, 3, 3> A{{1.0, 0.0, 0.0},
                           {0.0, 1.0, 0.0},
                           {0.0, 0.0, eps}};
    Vector<double, 3> b{1.0, 1.0, eps};

    Vector<double, 3> x = solve_3x3(A, b);

    // Solution should be [1, 1, 1]
    // Diagonal matrix, so no cancellation issues despite small determinant
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0, 1e-10);
    EXPECT_NEAR(x[2], 1.0, 1e-10);
}

// =============================================================================
// Random Matrix Tests
// =============================================================================

TEST_F(LUTest, RandomMatrices2x2) {
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int test = 0; test < 100; ++test) {
        // Generate random non-singular matrix
        Matrix<double, 2, 2> A;
        do {
            for (std::size_t i = 0; i < 2; ++i) {
                for (std::size_t j = 0; j < 2; ++j) {
                    A(i, j) = dist(rng);
                }
            }
        } while (std::abs(A.determinant()) < 1e-10);

        Vector<double, 2> b{dist(rng), dist(rng)};

        Vector<double, 2> x = solve_2x2(A, b);

        // Verify solution
        Vector<double, 2> Ax = A * x;
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_NEAR(Ax[i], b[i], 1e-10);
        }
    }
}

TEST_F(LUTest, RandomMatrices3x3) {
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int test = 0; test < 100; ++test) {
        // Generate random non-singular matrix
        Matrix<double, 3, 3> A;
        do {
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    A(i, j) = dist(rng);
                }
            }
        } while (std::abs(A.determinant()) < 1e-10);

        Vector<double, 3> b{dist(rng), dist(rng), dist(rng)};

        Vector<double, 3> x = solve_3x3(A, b);

        // Verify solution
        Vector<double, 3> Ax = A * x;
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(Ax[i], b[i], 1e-10);
        }
    }
}

// =============================================================================
// Special System Tests
// =============================================================================

TEST_F(LUTest, TridiagonalSystem) {
    // Tridiagonal matrix (common in FD/FE)
    Matrix<double, 4, 4> A = Matrix<double, 4, 4>::zero();
    A(0, 0) = 2.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = 2.0; A(1, 2) = -1.0;
    A(2, 1) = -1.0; A(2, 2) = 2.0; A(2, 3) = -1.0;
    A(3, 2) = -1.0; A(3, 3) = 2.0;

    Vector<double, 4> b{1.0, 0.0, 0.0, 1.0};

    Vector<double, 4> x = solve_4x4(A, b);

    // Verify solution
    Vector<double, 4> Ax = A * x;
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(Ax[i], b[i], tolerance);
    }
}

TEST_F(LUTest, SymmetricPositiveDefinite) {
    // SPD matrix (common in FE stiffness matrices)
    Matrix<double, 3, 3> A{{4.0, 1.0, 1.0},
                           {1.0, 3.0, 0.0},
                           {1.0, 0.0, 2.0}};
    Vector<double, 3> b{6.0, 4.0, 3.0};

    Vector<double, 3> x = solve_3x3(A, b);

    // Verify solution
    Vector<double, 3> Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(Ax[i], b[i], tolerance);
    }
}
