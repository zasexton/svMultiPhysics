#ifndef SVMP_FE_MATH_UTILS_H
#define SVMP_FE_MATH_UTILS_H

/**
 * @file MathUtils.h
 * @brief Common mathematical utility functions for FE computations
 *
 * This header provides essential mathematical utility functions including
 * sign operations, clamping, interpolation helpers, and numerical algorithms.
 * All functions are constexpr where possible for compile-time evaluation.
 */

#include "MathConstants.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief Compute the sign of a value
 * @tparam T Arithmetic type
 * @param x Input value
 * @return -1 for negative, 0 for zero, +1 for positive
 */
template<typename T>
constexpr int sign(T x) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    return (T(0) < x) - (x < T(0));
}

/**
 * @brief Compute absolute value
 * @tparam T Arithmetic type
 * @param x Input value
 * @return Absolute value of x
 */
template<typename T>
constexpr T abs(T x) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    return x < T(0) ? -x : x;
}

/**
 * @brief Compute minimum of two values
 * @tparam T Comparable type
 * @param a First value
 * @param b Second value
 * @return Smaller of a and b
 */
template<typename T>
constexpr const T& min(const T& a, const T& b) noexcept {
    return (b < a) ? b : a;
}

/**
 * @brief Compute maximum of two values
 * @tparam T Comparable type
 * @param a First value
 * @param b Second value
 * @return Larger of a and b
 */
template<typename T>
constexpr const T& max(const T& a, const T& b) noexcept {
    return (a < b) ? b : a;
}

/**
 * @brief Clamp value to a range [low, high]
 * @tparam T Comparable type
 * @param x Value to clamp
 * @param low Lower bound
 * @param high Upper bound
 * @return Clamped value
 */
template<typename T>
constexpr const T& clamp(const T& x, const T& low, const T& high) noexcept {
    return min(max(x, low), high);
}

/**
 * @brief Compute integer power at compile time
 * @tparam T Arithmetic type
 * @tparam N Power exponent
 * @param base Base value
 * @return base^N
 */
template<typename T, std::size_t N>
constexpr T ipow(T base) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    if constexpr (N == 0) {
        return T(1);
    } else if constexpr (N == 1) {
        return base;
    } else if constexpr (N % 2 == 0) {
        T half = ipow<T, N/2>(base);
        return half * half;
    } else {
        return base * ipow<T, N-1>(base);
    }
}

/**
 * @brief Runtime integer power function
 * @tparam T Arithmetic type
 * @param base Base value
 * @param exp Exponent (must be non-negative)
 * @return base^exp
 */
template<typename T>
constexpr T ipow(T base, unsigned int exp) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    T result = T(1);
    while (exp) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

/**
 * @brief Compute square of a value
 * @tparam T Arithmetic type
 * @param x Input value
 * @return x^2
 */
template<typename T>
constexpr T square(T x) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    return x * x;
}

/**
 * @brief Compute cube of a value
 * @tparam T Arithmetic type
 * @param x Input value
 * @return x^3
 */
template<typename T>
constexpr T cube(T x) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    return x * x * x;
}

/**
 * @brief Smooth Hermite interpolation between 0 and 1
 * @tparam T Floating-point type
 * @param edge0 Lower edge
 * @param edge1 Upper edge
 * @param x Value to interpolate
 * @return Smoothly interpolated value in [0,1]
 */
template<typename T>
constexpr T smoothstep(T edge0, T edge1, T x) noexcept {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point");
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}

/**
 * @brief Linear interpolation between two values
 * @tparam T Arithmetic type
 * @tparam U Interpolant type (typically floating-point)
 * @param a Start value (at t=0)
 * @param b End value (at t=1)
 * @param t Interpolation parameter [0,1]
 * @return Interpolated value
 */
template<typename T, typename U = T>
constexpr T lerp(const T& a, const T& b, U t) noexcept {
    static_assert(std::is_arithmetic_v<U>, "Interpolant must be arithmetic");
    return a + t * (b - a);
}

/**
 * @brief Inverse linear interpolation (find t given value)
 * @tparam T Arithmetic type
 * @param a Start value
 * @param b End value
 * @param value Value to find t for
 * @return Parameter t such that lerp(a,b,t) = value
 */
template<typename T>
constexpr T inverse_lerp(T a, T b, T value) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    return (value - a) / (b - a);
}

/**
 * @brief Map value from one range to another
 * @tparam T Arithmetic type
 * @param value Value to remap
 * @param in_min Input range minimum
 * @param in_max Input range maximum
 * @param out_min Output range minimum
 * @param out_max Output range maximum
 * @return Remapped value
 */
template<typename T>
constexpr T remap(T value, T in_min, T in_max, T out_min, T out_max) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    T t = (value - in_min) / (in_max - in_min);
    return lerp(out_min, out_max, t);
}

/**
 * @brief Compute finite difference approximation of derivative
 * @tparam F Function type
 * @tparam T Floating-point type
 * @param f Function to differentiate
 * @param x Point at which to compute derivative
 * @param h Step size (defaults to sqrt(epsilon))
 * @return Approximate derivative df/dx at x
 */
template<typename F, typename T>
inline T finite_difference(F&& f, T x, T h = std::sqrt(epsilon<T>)) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point");
    // Central difference formula for better accuracy
    return (f(x + h) - f(x - h)) / (T(2) * h);
}

/**
 * @brief Newton-Raphson method for finding roots
 * @tparam F Function type
 * @tparam DF Derivative function type
 * @tparam T Floating-point type
 * @param f Function to find root of
 * @param df Derivative of function
 * @param x0 Initial guess
 * @param max_iter Maximum iterations
 * @param tol Convergence tolerance
 * @return Root of function (where f(x) = 0)
 */
template<typename F, typename DF, typename T>
inline T newton_raphson(F&& f, DF&& df, T x0,
                       int max_iter = 100,
                       T tol = tolerance<T>) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point");

    T x = x0;
    for (int iter = 0; iter < max_iter; ++iter) {
        T fx = f(x);
        if (abs(fx) < tol) {
            return x;  // Converged
        }

        T dfx = df(x);
        if (abs(dfx) < epsilon<T>) {
            break;  // Derivative too small, can't continue
        }

        x = x - fx / dfx;
    }

    return x;  // Return best estimate
}

/**
 * @brief Compute factorial at compile time
 * @tparam N Input value
 * @return N!
 */
template<unsigned int N>
struct Factorial {
    static constexpr unsigned long long value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static constexpr unsigned long long value = 1;
};

template<unsigned int N>
inline constexpr unsigned long long factorial_v = Factorial<N>::value;

/**
 * @brief Compute binomial coefficient at compile time
 * @tparam N Total items
 * @tparam K Items to choose
 * @return C(N,K) = N!/(K!(N-K)!)
 */
template<unsigned int N, unsigned int K>
struct BinomialCoefficient {
    static constexpr unsigned long long value =
        Factorial<N>::value / (Factorial<K>::value * Factorial<N-K>::value);
};

template<unsigned int N, unsigned int K>
inline constexpr unsigned long long binomial_v = BinomialCoefficient<N, K>::value;

/**
 * @brief Wrap angle to [-π, π] range
 * @tparam T Floating-point type
 * @param angle Angle in radians
 * @return Wrapped angle in [-π, π]
 */
template<typename T>
inline T wrap_angle(T angle) noexcept {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point");
    // Wrap angle to [-π, π] using std::remainder
    // std::remainder returns the IEEE remainder which is in the range [-π, π]
    // and chooses the value closest to 0 when equidistant
    return std::remainder(angle, two_pi<T>);
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_UTILS_H