#ifndef SVMP_FE_MATH_SIMD_H
#define SVMP_FE_MATH_SIMD_H

/**
 * @file SIMD.h
 * @brief SIMD wrappers for vectorized mathematical operations
 *
 * This header provides portable SIMD intrinsics wrappers for accelerating
 * small-scale linear algebra operations. Supports AVX2 and AVX512 instruction
 * sets with automatic fallback to scalar operations when SIMD is unavailable.
 * Primary focus is on operations commonly used in FE assembly.
 */

#include "Vector.h"
#include "Matrix.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>
#include <type_traits>

// Detect available SIMD instruction sets
// Note: Include order matters - more specific architectures include their dependencies
#if defined(__AVX512F__)
    #define SVMP_HAS_AVX512 1
    #define SVMP_HAS_AVX2 1
    #define SVMP_HAS_AVX 1
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define SVMP_HAS_AVX2 1
    #define SVMP_HAS_AVX 1
    #include <immintrin.h>
#elif defined(__AVX__)
    #define SVMP_HAS_AVX 1
    #include <immintrin.h>
#elif defined(__SSE4_2__)
    #define SVMP_HAS_SSE42 1
    #include <nmmintrin.h>
#elif defined(__SSE3__)
    #define SVMP_HAS_SSE3 1
    #include <pmmintrin.h>
#elif defined(__SSE2__)
    #define SVMP_HAS_SSE2 1
    #include <emmintrin.h>
#endif

// Always include pmmintrin.h if we're using any AVX (for _mm_movehdup_ps)
#if defined(SVMP_HAS_AVX) || defined(SVMP_HAS_AVX2) || defined(SVMP_HAS_AVX512)
    #include <pmmintrin.h>
#endif

namespace svmp {
namespace FE {
namespace math {
namespace simd {

/**
 * @brief SIMD capabilities detection
 */
struct SIMDCapabilities {
    static constexpr bool has_sse2 =
        #if defined(SVMP_HAS_SSE2) || defined(SVMP_HAS_SSE3) || defined(SVMP_HAS_SSE42) || defined(SVMP_HAS_AVX) || defined(SVMP_HAS_AVX2) || defined(SVMP_HAS_AVX512)
            true;
        #else
            false;
        #endif

    static constexpr bool has_sse3 =
        #if defined(SVMP_HAS_SSE3) || defined(SVMP_HAS_SSE42) || defined(SVMP_HAS_AVX) || defined(SVMP_HAS_AVX2) || defined(SVMP_HAS_AVX512)
            true;
        #else
            false;
        #endif

    static constexpr bool has_sse42 =
        #if defined(SVMP_HAS_SSE42) || defined(SVMP_HAS_AVX) || defined(SVMP_HAS_AVX2) || defined(SVMP_HAS_AVX512)
            true;
        #else
            false;
        #endif

    static constexpr bool has_avx =
        #ifdef SVMP_HAS_AVX
            true;
        #else
            false;
        #endif

    static constexpr bool has_avx2 =
        #ifdef SVMP_HAS_AVX2
            true;
        #else
            false;
        #endif

    static constexpr bool has_avx512 =
        #ifdef SVMP_HAS_AVX512
            true;
        #else
            false;
        #endif

    /**
     * @brief Get vector width for float type
     * @return Number of floats that fit in a SIMD register
     */
    static constexpr std::size_t float_width() {
        if constexpr (has_avx512) return 16;
        else if constexpr (has_avx2 || has_avx) return 8;
        else if constexpr (has_sse2) return 4;
        else return 1;
    }

    /**
     * @brief Get vector width for double type
     * @return Number of doubles that fit in a SIMD register
     */
    static constexpr std::size_t double_width() {
        if constexpr (has_avx512) return 8;
        else if constexpr (has_avx2 || has_avx) return 4;
        else if constexpr (has_sse2) return 2;
        else return 1;
    }
};

/**
 * @brief SIMD vector type wrapper
 * @tparam T Scalar type (float or double)
 */
template<typename T>
struct SIMDVector {
    static_assert(std::is_floating_point_v<T>, "T must be float or double");
};

// Specialization for float
template<>
struct SIMDVector<float> {
#ifdef SVMP_HAS_AVX512
    using type = __m512;
    static constexpr std::size_t size = 16;
#elif defined(SVMP_HAS_AVX2) || defined(SVMP_HAS_AVX)
    using type = __m256;
    static constexpr std::size_t size = 8;
#elif defined(SVMP_HAS_SSE2)
    using type = __m128;
    static constexpr std::size_t size = 4;
#else
    using type = float;
    static constexpr std::size_t size = 1;
#endif
};

// Specialization for double
template<>
struct SIMDVector<double> {
#ifdef SVMP_HAS_AVX512
    using type = __m512d;
    static constexpr std::size_t size = 8;
#elif defined(SVMP_HAS_AVX2) || defined(SVMP_HAS_AVX)
    using type = __m256d;
    static constexpr std::size_t size = 4;
#elif defined(SVMP_HAS_SSE2)
    using type = __m128d;
    static constexpr std::size_t size = 2;
#else
    using type = double;
    static constexpr std::size_t size = 1;
#endif
};

template<typename T>
using simd_vec = typename SIMDVector<T>::type;

/**
 * @brief SIMD operations wrapper
 * @tparam T Scalar type (float or double)
 *
 * Provides a uniform interface for SIMD operations with automatic
 * fallback to scalar operations when SIMD is not available.
 */
template<typename T>
class SIMDOps {
    static_assert(std::is_floating_point_v<T>, "T must be float or double");

public:
    using vec_type = simd_vec<T>;
    static constexpr std::size_t vec_size = SIMDVector<T>::size;

    /**
     * @brief Load aligned data into SIMD register
     * @param ptr Pointer to aligned data (32-byte alignment required)
     * @return SIMD vector
     */
    static inline vec_type load(const T* ptr) {
        if constexpr (vec_size == 1) {
            return *ptr;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_load_ps(ptr);
#elif defined(SVMP_HAS_AVX)
            return _mm256_load_ps(ptr);
#elif defined(SVMP_HAS_SSE2)
            return _mm_load_ps(ptr);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_load_pd(ptr);
#elif defined(SVMP_HAS_AVX)
            return _mm256_load_pd(ptr);
#elif defined(SVMP_HAS_SSE2)
            return _mm_load_pd(ptr);
#endif
        }
    }

    /**
     * @brief Load unaligned data into SIMD register
     * @param ptr Pointer to unaligned data
     * @return SIMD vector
     */
    static inline vec_type loadu(const T* ptr) {
        if constexpr (vec_size == 1) {
            return *ptr;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_loadu_ps(ptr);
#elif defined(SVMP_HAS_AVX)
            return _mm256_loadu_ps(ptr);
#elif defined(SVMP_HAS_SSE2)
            return _mm_loadu_ps(ptr);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_loadu_pd(ptr);
#elif defined(SVMP_HAS_AVX)
            return _mm256_loadu_pd(ptr);
#elif defined(SVMP_HAS_SSE2)
            return _mm_loadu_pd(ptr);
#endif
        }
    }

    /**
     * @brief Store SIMD register to aligned memory
     * @param ptr Pointer to aligned memory (32-byte alignment required)
     * @param v SIMD vector to store
     */
    static inline void store(T* ptr, vec_type v) {
        if constexpr (vec_size == 1) {
            *ptr = v;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            _mm512_store_ps(ptr, v);
#elif defined(SVMP_HAS_AVX)
            _mm256_store_ps(ptr, v);
#elif defined(SVMP_HAS_SSE2)
            _mm_store_ps(ptr, v);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            _mm512_store_pd(ptr, v);
#elif defined(SVMP_HAS_AVX)
            _mm256_store_pd(ptr, v);
#elif defined(SVMP_HAS_SSE2)
            _mm_store_pd(ptr, v);
#endif
        }
    }

    /**
     * @brief Store SIMD register to unaligned memory
     * @param ptr Pointer to unaligned memory
     * @param v SIMD vector to store
     */
    static inline void storeu(T* ptr, vec_type v) {
        if constexpr (vec_size == 1) {
            *ptr = v;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            _mm512_storeu_ps(ptr, v);
#elif defined(SVMP_HAS_AVX)
            _mm256_storeu_ps(ptr, v);
#elif defined(SVMP_HAS_SSE2)
            _mm_storeu_ps(ptr, v);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            _mm512_storeu_pd(ptr, v);
#elif defined(SVMP_HAS_AVX)
            _mm256_storeu_pd(ptr, v);
#elif defined(SVMP_HAS_SSE2)
            _mm_storeu_pd(ptr, v);
#endif
        }
    }

    /**
     * @brief Add two SIMD vectors
     */
    static inline vec_type add(vec_type a, vec_type b) {
        if constexpr (vec_size == 1) {
            return a + b;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_add_ps(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_add_ps(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_add_ps(a, b);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_add_pd(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_add_pd(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_add_pd(a, b);
#endif
        }
    }

    /**
     * @brief Subtract two SIMD vectors
     */
    static inline vec_type sub(vec_type a, vec_type b) {
        if constexpr (vec_size == 1) {
            return a - b;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_sub_ps(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_sub_ps(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_sub_ps(a, b);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_sub_pd(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_sub_pd(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_sub_pd(a, b);
#endif
        }
    }

    /**
     * @brief Multiply two SIMD vectors element-wise
     */
    static inline vec_type mul(vec_type a, vec_type b) {
        if constexpr (vec_size == 1) {
            return a * b;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_mul_ps(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_mul_ps(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_mul_ps(a, b);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_mul_pd(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_mul_pd(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_mul_pd(a, b);
#endif
        }
    }

    /**
     * @brief Divide two SIMD vectors element-wise
     */
    static inline vec_type div(vec_type a, vec_type b) {
        if constexpr (vec_size == 1) {
            return a / b;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_div_ps(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_div_ps(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_div_ps(a, b);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_div_pd(a, b);
#elif defined(SVMP_HAS_AVX)
            return _mm256_div_pd(a, b);
#elif defined(SVMP_HAS_SSE2)
            return _mm_div_pd(a, b);
#endif
        }
    }

    /**
     * @brief Fused multiply-add: a*b + c
     */
    static inline vec_type fma(vec_type a, vec_type b, vec_type c) {
        if constexpr (vec_size == 1) {
            return std::fma(a, b, c);
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
    #if defined(__FMA__)
            return _mm512_fmadd_ps(a, b, c);
    #else
            return _mm512_add_ps(_mm512_mul_ps(a, b), c);
    #endif
#elif defined(SVMP_HAS_AVX2)
    #if defined(__FMA__)
            return _mm256_fmadd_ps(a, b, c);
    #else
            return _mm256_add_ps(_mm256_mul_ps(a, b), c);
    #endif
#elif defined(SVMP_HAS_AVX)
            return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#elif defined(SVMP_HAS_SSE2)
            return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
    #if defined(__FMA__)
            return _mm512_fmadd_pd(a, b, c);
    #else
            return _mm512_add_pd(_mm512_mul_pd(a, b), c);
    #endif
#elif defined(SVMP_HAS_AVX2)
    #if defined(__FMA__)
            return _mm256_fmadd_pd(a, b, c);
    #else
            return _mm256_add_pd(_mm256_mul_pd(a, b), c);
    #endif
#elif defined(SVMP_HAS_AVX)
            return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#elif defined(SVMP_HAS_SSE2)
            return _mm_add_pd(_mm_mul_pd(a, b), c);
#endif
        }
    }

    /**
     * @brief Broadcast single value to all elements
     */
    static inline vec_type broadcast(T value) {
        if constexpr (vec_size == 1) {
            return value;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_set1_ps(value);
#elif defined(SVMP_HAS_AVX)
            return _mm256_set1_ps(value);
#elif defined(SVMP_HAS_SSE2)
            return _mm_set1_ps(value);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_set1_pd(value);
#elif defined(SVMP_HAS_AVX)
            return _mm256_set1_pd(value);
#elif defined(SVMP_HAS_SSE2)
            return _mm_set1_pd(value);
#endif
        }
    }

    /**
     * @brief Set to zero
     */
    static inline vec_type zero() {
        if constexpr (vec_size == 1) {
            return T(0);
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_setzero_ps();
#elif defined(SVMP_HAS_AVX)
            return _mm256_setzero_ps();
#elif defined(SVMP_HAS_SSE2)
            return _mm_setzero_ps();
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_setzero_pd();
#elif defined(SVMP_HAS_AVX)
            return _mm256_setzero_pd();
#elif defined(SVMP_HAS_SSE2)
            return _mm_setzero_pd();
#endif
        }
    }

    /**
     * @brief Horizontal sum of all elements
     */
    static inline T horizontal_sum(vec_type v) {
        if constexpr (vec_size == 1) {
            return v;
        } else if constexpr (std::is_same_v<T, float>) {
#if defined(SVMP_HAS_AVX512)
            return _mm512_reduce_add_ps(v);
#elif defined(SVMP_HAS_AVX) || defined(SVMP_HAS_AVX2)
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            vlow = _mm_add_ps(vlow, vhigh);
            // Use shuffle instead of movehdup for SSE2 compatibility
            __m128 shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(vlow, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#elif defined(SVMP_HAS_SSE3) || defined(SVMP_HAS_SSE42) || defined(SVMP_HAS_SSE2)
            // SSE2-compatible horizontal sum
            __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#endif
        } else { // double
#if defined(SVMP_HAS_AVX512)
            return _mm512_reduce_add_pd(v);
#elif defined(SVMP_HAS_AVX) || defined(SVMP_HAS_AVX2)
            __m128d vlow = _mm256_castpd256_pd128(v);
            __m128d vhigh = _mm256_extractf128_pd(v, 1);
            vlow = _mm_add_pd(vlow, vhigh);
            __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
            return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
#elif defined(SVMP_HAS_SSE3) || defined(SVMP_HAS_SSE42) || defined(SVMP_HAS_SSE2)
            __m128d high64 = _mm_unpackhi_pd(v, v);
            return _mm_cvtsd_f64(_mm_add_sd(v, high64));
#endif
        }
    }

    /**
     * @brief Square root of all elements
     */
    static inline vec_type sqrt(vec_type v) {
        if constexpr (vec_size == 1) {
            return std::sqrt(v);
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_sqrt_ps(v);
#elif defined(SVMP_HAS_AVX)
            return _mm256_sqrt_ps(v);
#elif defined(SVMP_HAS_SSE2)
            return _mm_sqrt_ps(v);
#endif
        } else { // double
#ifdef SVMP_HAS_AVX512
            return _mm512_sqrt_pd(v);
#elif defined(SVMP_HAS_AVX)
            return _mm256_sqrt_pd(v);
#elif defined(SVMP_HAS_SSE2)
            return _mm_sqrt_pd(v);
#endif
        }
    }

    /**
     * @brief Reciprocal approximation (1/x)
     */
    static inline vec_type reciprocal(vec_type v) {
        if constexpr (vec_size == 1) {
            return T(1) / v;
        } else if constexpr (std::is_same_v<T, float>) {
#ifdef SVMP_HAS_AVX512
            return _mm512_rcp14_ps(v);
#elif defined(SVMP_HAS_AVX)
            return _mm256_rcp_ps(v);
#elif defined(SVMP_HAS_SSE2)
            return _mm_rcp_ps(v);
#endif
        } else { // double - no fast reciprocal for double
            return div(broadcast(T(1)), v);
        }
    }
};

/**
 * @brief Vectorized dot product
 * @tparam T Scalar type
 * @param a First vector data
 * @param b Second vector data
 * @param n Vector length
 * @return Dot product result
 */
template<typename T>
inline T dot_simd(const T* a, const T* b, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        // Scalar fallback
        T sum = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    } else {
        vec_t sum = ops::zero();
        std::size_t vec_end = n - (n % vec_size);

        // Vectorized portion
        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t va = ops::loadu(&a[i]);
            vec_t vb = ops::loadu(&b[i]);
            sum = ops::fma(va, vb, sum);
        }

        T result = ops::horizontal_sum(sum);

        // Handle remaining elements
        for (std::size_t i = vec_end; i < n; ++i) {
            result += a[i] * b[i];
        }

        return result;
    }
}

/**
 * @brief Vectorized norm computation
 * @tparam T Scalar type
 * @param a Vector data
 * @param n Vector length
 * @return L2 norm
 */
template<typename T>
inline T norm_simd(const T* a, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        // Scalar fallback
        T sum = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            sum += a[i] * a[i];
        }
        return std::sqrt(sum);
    } else {
        vec_t sum = ops::zero();
        std::size_t vec_end = n - (n % vec_size);

        // Vectorized portion
        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t va = ops::loadu(&a[i]);
            sum = ops::fma(va, va, sum);
        }

        T result = ops::horizontal_sum(sum);

        // Handle remaining elements
        for (std::size_t i = vec_end; i < n; ++i) {
            result += a[i] * a[i];
        }

        return std::sqrt(result);
    }
}

/**
 * @brief Vectorized vector addition: c = a + b
 * @tparam T Scalar type
 * @param a First vector
 * @param b Second vector
 * @param c Result vector
 * @param n Vector length
 */
template<typename T>
inline void add_simd(const T* a, const T* b, T* c, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        // Scalar fallback
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    } else {
        std::size_t vec_end = n - (n % vec_size);

        // Vectorized portion
        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t va = ops::loadu(&a[i]);
            vec_t vb = ops::loadu(&b[i]);
            vec_t vc = ops::add(va, vb);
            ops::storeu(&c[i], vc);
        }

        // Handle remaining elements
        for (std::size_t i = vec_end; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

/**
 * @brief Vectorized vector subtraction: c = a - b
 * @tparam T Scalar type
 * @param a First vector
 * @param b Second vector
 * @param c Result vector
 * @param n Vector length
 */
template<typename T>
inline void sub_simd(const T* a, const T* b, T* c, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] - b[i];
        }
    } else {
        std::size_t vec_end = n - (n % vec_size);

        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t va = ops::loadu(&a[i]);
            vec_t vb = ops::loadu(&b[i]);
            vec_t vc = ops::sub(va, vb);
            ops::storeu(&c[i], vc);
        }

        for (std::size_t i = vec_end; i < n; ++i) {
            c[i] = a[i] - b[i];
        }
    }
}

/**
 * @brief Vectorized element-wise multiplication: c = a * b
 * @tparam T Scalar type
 * @param a First vector
 * @param b Second vector
 * @param c Result vector
 * @param n Vector length
 */
template<typename T>
inline void mul_simd(const T* a, const T* b, T* c, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] * b[i];
        }
    } else {
        std::size_t vec_end = n - (n % vec_size);

        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t va = ops::loadu(&a[i]);
            vec_t vb = ops::loadu(&b[i]);
            vec_t vc = ops::mul(va, vb);
            ops::storeu(&c[i], vc);
        }

        for (std::size_t i = vec_end; i < n; ++i) {
            c[i] = a[i] * b[i];
        }
    }
}

/**
 * @brief Vectorized element-wise division: c = a / b
 * @tparam T Scalar type
 * @param a First vector
 * @param b Second vector
 * @param c Result vector
 * @param n Vector length
 */
template<typename T>
inline void div_simd(const T* a, const T* b, T* c, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] / b[i];
        }
    } else {
        std::size_t vec_end = n - (n % vec_size);

        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t va = ops::loadu(&a[i]);
            vec_t vb = ops::loadu(&b[i]);
            vec_t vc = ops::div(va, vb);
            ops::storeu(&c[i], vc);
        }

        for (std::size_t i = vec_end; i < n; ++i) {
            c[i] = a[i] / b[i];
        }
    }
}

/**
 * @brief Vectorized scalar multiplication: b = alpha * a
 * @tparam T Scalar type
 * @param alpha Scalar multiplier
 * @param a Input vector
 * @param b Result vector
 * @param n Vector length
 */
template<typename T>
inline void scale_simd(T alpha, const T* a, T* b, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        // Scalar fallback
        for (std::size_t i = 0; i < n; ++i) {
            b[i] = alpha * a[i];
        }
    } else {
        vec_t valpha = ops::broadcast(alpha);
        std::size_t vec_end = n - (n % vec_size);

        // Vectorized portion
        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t va = ops::loadu(&a[i]);
            vec_t vb = ops::mul(valpha, va);
            ops::storeu(&b[i], vb);
        }

        // Handle remaining elements
        for (std::size_t i = vec_end; i < n; ++i) {
            b[i] = alpha * a[i];
        }
    }
}

/**
 * @brief Vectorized AXPY: y = alpha * x + y
 * @tparam T Scalar type
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param y Input/output vector
 * @param n Vector length
 */
template<typename T>
inline void axpy_simd(T alpha, const T* x, T* y, std::size_t n) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        // Scalar fallback
        for (std::size_t i = 0; i < n; ++i) {
            y[i] += alpha * x[i];
        }
    } else {
        vec_t valpha = ops::broadcast(alpha);
        std::size_t vec_end = n - (n % vec_size);

        // Vectorized portion
        for (std::size_t i = 0; i < vec_end; i += vec_size) {
            vec_t vx = ops::loadu(&x[i]);
            vec_t vy = ops::loadu(&y[i]);
            vy = ops::fma(valpha, vx, vy);
            ops::storeu(&y[i], vy);
        }

        // Handle remaining elements
        for (std::size_t i = vec_end; i < n; ++i) {
            y[i] += alpha * x[i];
        }
    }
}

/**
 * @brief Vectorized matrix-vector multiplication (small matrices)
 * @tparam T Scalar type
 * @param A Matrix data (row-major, M x N)
 * @param x Input vector (length N)
 * @param y Output vector (length M)
 * @param M Number of rows
 * @param N Number of columns
 */
template<typename T>
inline void gemv_simd(const T* A, const T* x, T* y,
                      std::size_t M, std::size_t N) {
    using ops = SIMDOps<T>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size == 1) {
        // Scalar fallback
        for (std::size_t i = 0; i < M; ++i) {
            T sum = T(0);
            for (std::size_t j = 0; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
    } else if (N < vec_size) {
        // Scalar fallback for small matrices
        for (std::size_t i = 0; i < M; ++i) {
            T sum = T(0);
            for (std::size_t j = 0; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
    } else {
        std::size_t vec_end = N - (N % vec_size);

        for (std::size_t i = 0; i < M; ++i) {
            vec_t sum = ops::zero();
            const T* row = &A[i * N];

            // Vectorized portion
            for (std::size_t j = 0; j < vec_end; j += vec_size) {
                vec_t va = ops::loadu(&row[j]);
                vec_t vx = ops::loadu(&x[j]);
                sum = ops::fma(va, vx, sum);
            }

            T result = ops::horizontal_sum(sum);

            // Handle remaining elements
            for (std::size_t j = vec_end; j < N; ++j) {
                result += row[j] * x[j];
            }

            y[i] = result;
        }
    }
}

/**
 * @brief Helper to enable SIMD operations for Vector class
 * @tparam T Scalar type
 * @tparam N Vector dimension
 */
template<typename T, std::size_t N>
class VectorSIMDOps {
public:
    /**
     * @brief Compute dot product using SIMD
     */
    static T dot(const Vector<T, N>& a, const Vector<T, N>& b) {
        return dot_simd(a.data(), b.data(), N);
    }

    /**
     * @brief Compute norm using SIMD
     */
    static T norm(const Vector<T, N>& a) {
        return norm_simd(a.data(), N);
    }

    /**
     * @brief Add vectors using SIMD
     */
    static Vector<T, N> add(const Vector<T, N>& a, const Vector<T, N>& b) {
        Vector<T, N> result;
        add_simd(a.data(), b.data(), result.data(), N);
        return result;
    }

    /**
     * @brief Subtract vectors using SIMD
     */
    static Vector<T, N> subtract(const Vector<T, N>& a, const Vector<T, N>& b) {
        Vector<T, N> result;
        sub_simd(a.data(), b.data(), result.data(), N);
        return result;
    }

    /**
     * @brief Multiply vectors element-wise using SIMD
     */
    static Vector<T, N> multiply(const Vector<T, N>& a, const Vector<T, N>& b) {
        Vector<T, N> result;
        mul_simd(a.data(), b.data(), result.data(), N);
        return result;
    }

    /**
     * @brief Divide vectors element-wise using SIMD
     */
    static Vector<T, N> divide(const Vector<T, N>& a, const Vector<T, N>& b) {
        Vector<T, N> result;
        div_simd(a.data(), b.data(), result.data(), N);
        return result;
    }

    /**
     * @brief Scale vector using SIMD
     */
    static Vector<T, N> scale(T alpha, const Vector<T, N>& a) {
        Vector<T, N> result;
        scale_simd(alpha, a.data(), result.data(), N);
        return result;
    }
};

//=============================================================================
// Convenience wrapper functions in simd namespace
//=============================================================================

/**
 * @brief Check if AVX is available at runtime
 */
inline bool is_avx_available() {
    return SIMDCapabilities::has_avx || SIMDCapabilities::has_avx2;
}

/**
 * @brief Check if AVX512 is available at runtime
 */
inline bool is_avx512_available() {
    return SIMDCapabilities::has_avx512;
}

/**
 * @brief Add two vectors using SIMD
 */
template<typename T, std::size_t N>
inline Vector<T, N> add(const Vector<T, N>& a, const Vector<T, N>& b) {
    return VectorSIMDOps<T, N>::add(a, b);
}

/**
 * @brief Subtract two vectors using SIMD
 */
template<typename T, std::size_t N>
inline Vector<T, N> subtract(const Vector<T, N>& a, const Vector<T, N>& b) {
    return VectorSIMDOps<T, N>::subtract(a, b);
}

/**
 * @brief Multiply two vectors element-wise using SIMD
 */
template<typename T, std::size_t N>
inline Vector<T, N> multiply(const Vector<T, N>& a, const Vector<T, N>& b) {
    return VectorSIMDOps<T, N>::multiply(a, b);
}

/**
 * @brief Divide two vectors element-wise using SIMD
 */
template<typename T, std::size_t N>
inline Vector<T, N> divide(const Vector<T, N>& a, const Vector<T, N>& b) {
    return VectorSIMDOps<T, N>::divide(a, b);
}

/**
 * @brief Compute dot product using SIMD
 */
template<typename T, std::size_t N>
inline T dot(const Vector<T, N>& a, const Vector<T, N>& b) {
    return VectorSIMDOps<T, N>::dot(a, b);
}

/**
 * @brief Compute 3D cross product using SIMD
 */
template<typename T>
inline Vector<T, 3> cross(const Vector<T, 3>& a, const Vector<T, 3>& b) {
    return a.cross(b);  // Delegate to Vector's cross product
}

/**
 * @brief Compute norm using SIMD
 */
template<typename T, std::size_t N>
inline T norm(const Vector<T, N>& a) {
    return VectorSIMDOps<T, N>::norm(a);
}

/**
 * @brief Fused multiply-add: a*b + c
 */
template<typename T, std::size_t N>
inline Vector<T, N> fma(const Vector<T, N>& a, const Vector<T, N>& b, const Vector<T, N>& c) {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = std::fma(a[i], b[i], c[i]);
    }
    return result;
}

/**
 * @brief Fused multiply-subtract: a*b - c
 */
template<typename T, std::size_t N>
inline Vector<T, N> fms(const Vector<T, N>& a, const Vector<T, N>& b, const Vector<T, N>& c) {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = std::fma(a[i], b[i], -c[i]);
    }
    return result;
}

/**
 * @brief Horizontal sum of vector elements
 */
template<typename T, std::size_t N>
inline T horizontal_sum(const Vector<T, N>& v) {
    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        sum += v[i];
    }
    return sum;
}

/**
 * @brief Broadcast scalar to all elements
 */
template<std::size_t N, typename T>
inline Vector<T, N> broadcast(T value) {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = value;
    }
    return result;
}

/**
 * @brief Gather elements from array using indices
 */
template<std::size_t N, typename T>
inline Vector<T, N> gather(const T* data, const int* indices) {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = data[indices[i]];
    }
    return result;
}

/**
 * @brief Scatter vector elements to array using indices
 */
template<typename T, std::size_t N>
inline void scatter(const Vector<T, N>& v, T* data, const int* indices) {
    for (std::size_t i = 0; i < N; ++i) {
        data[indices[i]] = v[i];
    }
}

/**
 * @brief Compare vectors for equality element-wise
 */
template<typename T, std::size_t N>
inline std::array<bool, N> compare_eq(const Vector<T, N>& a, const Vector<T, N>& b) {
    std::array<bool, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = (a[i] == b[i]);
    }
    return result;
}

/**
 * @brief Compare vectors for less-than element-wise
 */
template<typename T, std::size_t N>
inline std::array<bool, N> compare_lt(const Vector<T, N>& a, const Vector<T, N>& b) {
    std::array<bool, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = (a[i] < b[i]);
    }
    return result;
}

// Forward declaration used by the simd::multiply(Matrix, Vector) wrapper.
template<typename T, std::size_t M, std::size_t N>
class MatrixSIMDOps;

/**
 * @brief Matrix multiplication using SIMD
 */
template<typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> multiply(const Matrix<T, M, N>& A, const Matrix<T, M, N>& B) {
    return A * B;  // Delegate to Matrix's operator*
}

/**
 * @brief Matrix-vector multiplication using SIMD
 */
template<typename T, std::size_t M, std::size_t N>
inline Vector<T, M> multiply(const Matrix<T, M, N>& A, const Vector<T, N>& v) {
    return MatrixSIMDOps<T, M, N>::gemv(A, v);
}

/**
 * @brief AVX512-specific addition (8-wide double vectors)
 */
template<typename T>
inline Vector<T, 8> add_avx512(const Vector<T, 8>& a, const Vector<T, 8>& b) {
    return add(a, b);
}

/**
 * @brief Load vector from unaligned memory
 */
template<std::size_t N, typename T>
inline Vector<T, N> load_unaligned(const T* data) {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = data[i];
    }
    return result;
}

/**
 * @brief Helper to enable SIMD operations for Matrix class
 * @tparam T Scalar type
 * @tparam M Number of rows
 * @tparam N Number of columns
 */
template<typename T, std::size_t M, std::size_t N>
class MatrixSIMDOps {
public:
    /**
     * @brief Matrix-vector multiplication using SIMD
     */
    static Vector<T, M> gemv(const Matrix<T, M, N>& A, const Vector<T, N>& x) {
        Vector<T, M> y;
        gemv_simd(A.data(), x.data(), y.data(), M, N);
        return y;
    }
};

/**
 * @brief Aligned allocator for SIMD-friendly containers
 *
 * Provides aligned allocations suitable for aligned SIMD loads/stores.
 * The default alignment is 64 bytes to satisfy AVX512 aligned load
 * requirements; it also works for AVX/SSE targets.
 */
template<typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template<class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > (static_cast<std::size_t>(-1) / sizeof(T))) {
            throw std::bad_array_new_length();
        }

        void* ptr = nullptr;
#ifdef _MSC_VER
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#ifdef _MSC_VER
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template<class U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template<class T1, class T2, std::size_t A1, std::size_t A2>
constexpr bool operator==(const AlignedAllocator<T1, A1>&,
                          const AlignedAllocator<T2, A2>&) noexcept {
    return A1 == A2;
}

template<class T1, class T2, std::size_t A1, std::size_t A2>
constexpr bool operator!=(const AlignedAllocator<T1, A1>&,
                          const AlignedAllocator<T2, A2>&) noexcept {
    return A1 != A2;
}

} // namespace simd
} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_SIMD_H
