/**
 * @file test_SIMD.cpp
 * @brief Unit tests for SIMD.h - SIMD optimized operations
 */

#include <gtest/gtest.h>
#include "FE/Math/SIMD.h"
#include "FE/Math/Vector.h"
#include "FE/Math/Matrix.h"
#include <immintrin.h>
#include <array>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

using namespace svmp::FE::math;

// Test fixture for SIMD tests
class SIMDTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-14;
    std::mt19937 rng{42};

    void SetUp() override {
        // SIMD.h provides scalar fallbacks; tests should run even when AVX isn't enabled.
    }

    void TearDown() override {}

    // Helper to generate aligned data
    template<typename T>
    T* allocate_aligned(size_t count) {
        return static_cast<T*>(_mm_malloc(count * sizeof(T), 32));
    }

    template<typename T>
    void free_aligned(T* ptr) {
        _mm_free(ptr);
    }
};

// =============================================================================
// SIMD Vector Operations Tests
// =============================================================================

TEST_F(SIMDTest, VectorAdd2D) {
    Vector<double, 2> a{1.0, 2.0};
    Vector<double, 2> b{3.0, 4.0};

    // SIMD version
    Vector<double, 2> c_simd = simd::add(a, b);

    // Reference version
    Vector<double, 2> c_ref = a + b;

    EXPECT_NEAR(c_simd[0], c_ref[0], tolerance);
    EXPECT_NEAR(c_simd[1], c_ref[1], tolerance);
}

TEST_F(SIMDTest, VectorAdd3D) {
    Vector<double, 3> a{1.0, 2.0, 3.0};
    Vector<double, 3> b{4.0, 5.0, 6.0};

    Vector<double, 3> c_simd = simd::add(a, b);
    Vector<double, 3> c_ref = a + b;

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(c_simd[i], c_ref[i], tolerance);
    }
}

TEST_F(SIMDTest, VectorAdd4D) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{5.0, 6.0, 7.0, 8.0};

    Vector<double, 4> c_simd = simd::add(a, b);
    Vector<double, 4> c_ref = a + b;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(c_simd[i], c_ref[i], tolerance);
    }
}

TEST_F(SIMDTest, VectorSubtract) {
    Vector<double, 4> a{10.0, 20.0, 30.0, 40.0};
    Vector<double, 4> b{1.0, 2.0, 3.0, 4.0};

    Vector<double, 4> c_simd = simd::subtract(a, b);
    Vector<double, 4> c_ref = a - b;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(c_simd[i], c_ref[i], tolerance);
    }
}

TEST_F(SIMDTest, VectorMultiply) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{2.0, 3.0, 4.0, 5.0};

    Vector<double, 4> c_simd = simd::multiply(a, b);

    EXPECT_NEAR(c_simd[0], 2.0, tolerance);
    EXPECT_NEAR(c_simd[1], 6.0, tolerance);
    EXPECT_NEAR(c_simd[2], 12.0, tolerance);
    EXPECT_NEAR(c_simd[3], 20.0, tolerance);
}

TEST_F(SIMDTest, VectorDivide) {
    Vector<double, 4> a{10.0, 20.0, 30.0, 40.0};
    Vector<double, 4> b{2.0, 4.0, 5.0, 8.0};

    Vector<double, 4> c_simd = simd::divide(a, b);

    EXPECT_NEAR(c_simd[0], 5.0, tolerance);
    EXPECT_NEAR(c_simd[1], 5.0, tolerance);
    EXPECT_NEAR(c_simd[2], 6.0, tolerance);
    EXPECT_NEAR(c_simd[3], 5.0, tolerance);
}

// =============================================================================
// SIMD Dot Product Tests
// =============================================================================

TEST_F(SIMDTest, DotProduct2D) {
    Vector<double, 2> a{3.0, 4.0};
    Vector<double, 2> b{1.0, 2.0};

    double dot_simd = simd::dot(a, b);
    double dot_ref = a.dot(b);

    EXPECT_NEAR(dot_simd, dot_ref, tolerance);
    EXPECT_NEAR(dot_simd, 11.0, tolerance);  // 3*1 + 4*2 = 11
}

TEST_F(SIMDTest, DotProduct3D) {
    Vector<double, 3> a{1.0, 2.0, 3.0};
    Vector<double, 3> b{4.0, 5.0, 6.0};

    double dot_simd = simd::dot(a, b);
    double dot_ref = a.dot(b);

    EXPECT_NEAR(dot_simd, dot_ref, tolerance);
    EXPECT_NEAR(dot_simd, 32.0, tolerance);  // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(SIMDTest, DotProduct4D) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{5.0, 6.0, 7.0, 8.0};

    double dot_simd = simd::dot(a, b);
    double dot_ref = a.dot(b);

    EXPECT_NEAR(dot_simd, dot_ref, tolerance);
    EXPECT_NEAR(dot_simd, 70.0, tolerance);  // 1*5 + 2*6 + 3*7 + 4*8 = 70
}

// =============================================================================
// SIMD Cross Product Tests (3D only)
// =============================================================================

TEST_F(SIMDTest, CrossProduct3D) {
    Vector<double, 3> a{1.0, 2.0, 3.0};
    Vector<double, 3> b{4.0, 5.0, 6.0};

    Vector<double, 3> c_simd = simd::cross(a, b);
    Vector<double, 3> c_ref = a.cross(b);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(c_simd[i], c_ref[i], tolerance);
    }
}

TEST_F(SIMDTest, CrossProductBasisVectors) {
    Vector<double, 3> x{1.0, 0.0, 0.0};
    Vector<double, 3> y{0.0, 1.0, 0.0};
    Vector<double, 3> z{0.0, 0.0, 1.0};

    // x × y = z
    Vector<double, 3> xy = simd::cross(x, y);
    EXPECT_NEAR(xy[0], 0.0, tolerance);
    EXPECT_NEAR(xy[1], 0.0, tolerance);
    EXPECT_NEAR(xy[2], 1.0, tolerance);

    // y × z = x
    Vector<double, 3> yz = simd::cross(y, z);
    EXPECT_NEAR(yz[0], 1.0, tolerance);
    EXPECT_NEAR(yz[1], 0.0, tolerance);
    EXPECT_NEAR(yz[2], 0.0, tolerance);

    // z × x = y
    Vector<double, 3> zx = simd::cross(z, x);
    EXPECT_NEAR(zx[0], 0.0, tolerance);
    EXPECT_NEAR(zx[1], 1.0, tolerance);
    EXPECT_NEAR(zx[2], 0.0, tolerance);
}

// =============================================================================
// SIMD Norm Tests
// =============================================================================

TEST_F(SIMDTest, Norm2D) {
    Vector<double, 2> v{3.0, 4.0};

    double norm_simd = simd::norm(v);
    double norm_ref = v.norm();

    EXPECT_NEAR(norm_simd, norm_ref, tolerance);
    EXPECT_NEAR(norm_simd, 5.0, tolerance);
}

TEST_F(SIMDTest, Norm3D) {
    Vector<double, 3> v{2.0, 6.0, 9.0};

    double norm_simd = simd::norm(v);
    double norm_ref = v.norm();

    EXPECT_NEAR(norm_simd, norm_ref, tolerance);
    EXPECT_NEAR(norm_simd, 11.0, tolerance);  // sqrt(4 + 36 + 81) = 11
}

TEST_F(SIMDTest, Norm4D) {
    Vector<double, 4> v{1.0, 2.0, 2.0, 0.0};

    double norm_simd = simd::norm(v);
    double norm_ref = v.norm();

    EXPECT_NEAR(norm_simd, norm_ref, tolerance);
    EXPECT_NEAR(norm_simd, 3.0, tolerance);  // sqrt(1 + 4 + 4 + 0) = 3
}

// =============================================================================
// SIMD Matrix Operations Tests
// =============================================================================

TEST_F(SIMDTest, Matrix2x2Multiply) {
    Matrix<double, 2, 2> A{{1.0, 2.0},
                           {3.0, 4.0}};
    Matrix<double, 2, 2> B{{5.0, 6.0},
                           {7.0, 8.0}};

    Matrix<double, 2, 2> C_simd = simd::multiply(A, B);
    Matrix<double, 2, 2> C_ref = A * B;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(C_simd(i, j), C_ref(i, j), tolerance);
        }
    }
}

TEST_F(SIMDTest, Matrix3x3Multiply) {
    Matrix<double, 3, 3> A{{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};
    Matrix<double, 3, 3> B{{9, 8, 7},
                           {6, 5, 4},
                           {3, 2, 1}};

    Matrix<double, 3, 3> C_simd = simd::multiply(A, B);
    Matrix<double, 3, 3> C_ref = A * B;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(C_simd(i, j), C_ref(i, j), tolerance);
        }
    }
}

TEST_F(SIMDTest, Matrix4x4Multiply) {
    Matrix<double, 4, 4> A = Matrix<double, 4, 4>::identity();
    Matrix<double, 4, 4> B{{1, 2, 3, 4},
                           {5, 6, 7, 8},
                           {9, 10, 11, 12},
                           {13, 14, 15, 16}};

    Matrix<double, 4, 4> C_simd = simd::multiply(A, B);

    // Identity * B = B
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(C_simd(i, j), B(i, j), tolerance);
        }
    }
}

TEST_F(SIMDTest, MatrixVectorMultiply) {
    Matrix<double, 3, 3> A{{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};
    Vector<double, 3> v{1.0, 2.0, 3.0};

    Vector<double, 3> result_simd = simd::multiply(A, v);
    Vector<double, 3> result_ref = A * v;

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(result_simd[i], result_ref[i], tolerance);
    }
}

// =============================================================================
// SIMD Pointer-Kernel Tests
// =============================================================================

TEST_F(SIMDTest, ScaleSIMD) {
    std::array<double, 7> a{{1.0, -2.0, 3.5, 0.0, 4.25, -1.0, 2.0}};
    std::array<double, 7> b{};
    const double alpha = 2.5;

    simd::scale_simd(alpha, a.data(), b.data(), a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(b[i], alpha * a[i], tolerance);
    }
}

TEST_F(SIMDTest, AxpySIMD) {
    std::array<double, 7> x{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}};
    std::array<double, 7> y{{-1.0, 0.5, 2.0, -3.0, 0.0, 1.0, -2.5}};
    auto y_ref = y;
    const double alpha = -0.25;

    for (std::size_t i = 0; i < x.size(); ++i) {
        y_ref[i] += alpha * x[i];
    }

    simd::axpy_simd(alpha, x.data(), y.data(), x.size());

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], y_ref[i], tolerance);
    }
}

TEST_F(SIMDTest, GemvSIMD) {
    constexpr std::size_t M = 3;
    constexpr std::size_t N = 5;

    // Row-major A (M x N).
    std::array<double, M * N> A{{1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 10.0,
                                 -1.0, 0.5, 0.0, 2.0, -3.0}};
    std::array<double, N> x{{1.0, -1.0, 2.0, 0.5, -2.0}};
    std::array<double, M> y{};

    simd::gemv_simd(A.data(), x.data(), y.data(), M, N);

    std::array<double, M> y_ref{};
    for (std::size_t i = 0; i < M; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        y_ref[i] = sum;
    }

    for (std::size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y[i], y_ref[i], tolerance);
    }
}

TEST_F(SIMDTest, MatrixSIMDOpsGemv) {
    Matrix<double, 3, 5> A{{1.0, 2.0, 3.0, 4.0, 5.0},
                           {6.0, 7.0, 8.0, 9.0, 10.0},
                           {-1.0, 0.5, 0.0, 2.0, -3.0}};
    Vector<double, 5> x{1.0, -1.0, 2.0, 0.5, -2.0};

    auto y_simd = simd::MatrixSIMDOps<double, 3, 5>::gemv(A, x);
    auto y_ref = A * x;

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(y_simd[i], y_ref[i], tolerance);
    }
}

// =============================================================================
// SIMD Fused Multiply-Add Tests
// =============================================================================

TEST_F(SIMDTest, FusedMultiplyAdd) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{2.0, 3.0, 4.0, 5.0};
    Vector<double, 4> c{10.0, 10.0, 10.0, 10.0};

    // result = a * b + c
    Vector<double, 4> result = simd::fma(a, b, c);

    EXPECT_NEAR(result[0], 12.0, tolerance);  // 1*2 + 10
    EXPECT_NEAR(result[1], 16.0, tolerance);  // 2*3 + 10
    EXPECT_NEAR(result[2], 22.0, tolerance);  // 3*4 + 10
    EXPECT_NEAR(result[3], 30.0, tolerance);  // 4*5 + 10
}

TEST_F(SIMDTest, FusedMultiplySubtract) {
    Vector<double, 4> a{5.0, 6.0, 7.0, 8.0};
    Vector<double, 4> b{2.0, 2.0, 2.0, 2.0};
    Vector<double, 4> c{1.0, 2.0, 3.0, 4.0};

    // result = a * b - c
    Vector<double, 4> result = simd::fms(a, b, c);

    EXPECT_NEAR(result[0], 9.0, tolerance);   // 5*2 - 1
    EXPECT_NEAR(result[1], 10.0, tolerance);  // 6*2 - 2
    EXPECT_NEAR(result[2], 11.0, tolerance);  // 7*2 - 3
    EXPECT_NEAR(result[3], 12.0, tolerance);  // 8*2 - 4
}

// =============================================================================
// SIMD Special Operations Tests
// =============================================================================

TEST_F(SIMDTest, HorizontalSum) {
    Vector<double, 4> v{1.0, 2.0, 3.0, 4.0};

    double sum = simd::horizontal_sum(v);

    EXPECT_NEAR(sum, 10.0, tolerance);
}

TEST_F(SIMDTest, Broadcast) {
    double value = 5.0;

    Vector<double, 4> v = simd::broadcast<4>(value);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(v[i], 5.0);
    }
}

TEST_F(SIMDTest, Gather) {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int indices[] = {0, 2, 4, 6};

    Vector<double, 4> v = simd::gather<4>(data, indices);

    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 3.0);
    EXPECT_EQ(v[2], 5.0);
    EXPECT_EQ(v[3], 7.0);
}

TEST_F(SIMDTest, Scatter) {
    Vector<double, 4> v{10.0, 20.0, 30.0, 40.0};
    double data[8] = {0.0};
    int indices[] = {1, 3, 5, 7};

    simd::scatter(v, data, indices);

    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 10.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], 20.0);
    EXPECT_EQ(data[4], 0.0);
    EXPECT_EQ(data[5], 30.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 40.0);
}

// =============================================================================
// SIMD Comparison Tests
// =============================================================================

TEST_F(SIMDTest, CompareEqual) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> c{1.0, 2.0, 3.0, 5.0};

    auto mask_ab = simd::compare_eq(a, b);
    auto mask_ac = simd::compare_eq(a, c);

    // All elements equal for a and b
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(mask_ab[i]);
    }

    // First 3 elements equal for a and c
    EXPECT_TRUE(mask_ac[0]);
    EXPECT_TRUE(mask_ac[1]);
    EXPECT_TRUE(mask_ac[2]);
    EXPECT_FALSE(mask_ac[3]);
}

TEST_F(SIMDTest, CompareLessThan) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{2.0, 2.0, 2.0, 2.0};

    auto mask = simd::compare_lt(a, b);

    EXPECT_TRUE(mask[0]);   // 1 < 2
    EXPECT_FALSE(mask[1]);  // 2 < 2 is false
    EXPECT_FALSE(mask[2]);  // 3 < 2 is false
    EXPECT_FALSE(mask[3]);  // 4 < 2 is false
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST_F(SIMDTest, DISABLED_PerformanceVectorOps) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{5.0, 6.0, 7.0, 8.0};

    auto start_simd = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000000; ++i) {
        Vector<double, 4> c = simd::add(a, b);
        a[0] = c[0] * 0.999;  // Prevent optimization
    }
    auto end_simd = std::chrono::high_resolution_clock::now();

    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000000; ++i) {
        Vector<double, 4> c = a + b;
        a[0] = c[0] * 0.999;  // Prevent optimization
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();

    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end_simd - start_simd);
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar);

    // SIMD should be at least as fast as scalar
    EXPECT_LE(simd_time.count(), scalar_time.count() * 1.1);
}

TEST_F(SIMDTest, DISABLED_PerformanceDotProduct) {
    Vector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    Vector<double, 4> b{5.0, 6.0, 7.0, 8.0};

    auto start_simd = std::chrono::high_resolution_clock::now();
    double sum_simd = 0.0;
    for (int i = 0; i < 10000000; ++i) {
        sum_simd += simd::dot(a, b);
    }
    auto end_simd = std::chrono::high_resolution_clock::now();

    auto start_scalar = std::chrono::high_resolution_clock::now();
    double sum_scalar = 0.0;
    for (int i = 0; i < 10000000; ++i) {
        sum_scalar += a.dot(b);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();

    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end_simd - start_simd);
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar);

    // SIMD should provide speedup for dot product
    EXPECT_LT(simd_time.count(), scalar_time.count());
}

// =============================================================================
// AVX512 Tests (if available)
// =============================================================================

TEST_F(SIMDTest, AVX512Operations) {
    if (!simd::is_avx512_available()) {
        GTEST_SKIP() << "AVX512 not available on this platform";
    }

    // Test 8-wide operations
    Vector<double, 8> a{1, 2, 3, 4, 5, 6, 7, 8};
    Vector<double, 8> b{8, 7, 6, 5, 4, 3, 2, 1};

    Vector<double, 8> c = simd::add_avx512(a, b);

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(c[i], 9.0, tolerance);
    }
}

// =============================================================================
// Float Operations Tests
// =============================================================================

TEST_F(SIMDTest, FloatOperations) {
    Vector<float, 4> a{1.0f, 2.0f, 3.0f, 4.0f};
    Vector<float, 4> b{5.0f, 6.0f, 7.0f, 8.0f};

    Vector<float, 4> c = simd::add(a, b);

    EXPECT_NEAR(c[0], 6.0f, 1e-6f);
    EXPECT_NEAR(c[1], 8.0f, 1e-6f);
    EXPECT_NEAR(c[2], 10.0f, 1e-6f);
    EXPECT_NEAR(c[3], 12.0f, 1e-6f);
}

TEST_F(SIMDTest, FloatDotProduct) {
    Vector<float, 4> a{1.0f, 2.0f, 3.0f, 4.0f};
    Vector<float, 4> b{5.0f, 6.0f, 7.0f, 8.0f};

    float dot = simd::dot(a, b);

    EXPECT_NEAR(dot, 70.0f, 1e-6f);
}

// =============================================================================
// Memory Alignment Tests
// =============================================================================

TEST_F(SIMDTest, AlignmentCheck) {
    Vector<double, 4> v;

    // Check that Vector data is properly aligned for SIMD
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(v.data());
    EXPECT_EQ(addr % 32, 0) << "Vector data should be 32-byte aligned for AVX";
}

TEST_F(SIMDTest, AlignedAllocatorAlignment) {
    simd::AlignedAllocator<double> alloc;
    double* ptr = alloc.allocate(16);
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    EXPECT_EQ(addr % 64, 0u);
    alloc.deallocate(ptr, 16);
}

TEST_F(SIMDTest, UnalignedLoad) {
    // Create unaligned data
    double* data = allocate_aligned<double>(9);
    for (int i = 0; i < 9; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Load from unaligned position
    Vector<double, 4> v = simd::load_unaligned<4>(data + 1);

    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
    EXPECT_EQ(v[3], 4.0);

    free_aligned(data);
}
