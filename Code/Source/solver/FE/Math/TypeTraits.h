#ifndef SVMP_FE_MATH_TYPE_TRAITS_H
#define SVMP_FE_MATH_TYPE_TRAITS_H

/**
 * @file TypeTraits.h
 * @brief Type traits and SFINAE helpers for mathematical types
 *
 * This header provides compile-time type checking and trait detection
 * for the mathematical types in the FE library. It includes SFINAE
 * helpers for enabling/disabling template functions based on type properties.
 */

#include <cstddef>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace math {

// Forward declarations
template<typename T, std::size_t N> class Vector;
template<typename T, std::size_t M, std::size_t N> class Matrix;
template<typename T, std::size_t Rank, std::size_t Dim> class TensorBase;
template<typename T, std::size_t Dim> class Tensor2;
template<typename T, std::size_t Dim> class Tensor4;
template<typename Derived> class VectorExpr;
template<typename Derived> class MatrixExpr;

/**
 * @brief Check if type is a Vector
 */
template<typename T>
struct is_vector : std::false_type {};

template<typename T, std::size_t N>
struct is_vector<Vector<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

/**
 * @brief Check if type is a VectorExpr
 */
template<typename T>
struct is_vector_expr : std::bool_constant<
    std::is_base_of_v<VectorExpr<std::decay_t<T>>, std::decay_t<T>> &&
    !is_vector_v<std::decay_t<T>>
> {};

template<typename T>
inline constexpr bool is_vector_expr_v = is_vector_expr<T>::value;

/**
 * @brief Check if type is vector-like (Vector or VectorExpr)
 */
template<typename T>
struct is_vector_like : std::bool_constant<is_vector_v<T> || is_vector_expr_v<T>> {};

template<typename T>
inline constexpr bool is_vector_like_v = is_vector_like<T>::value;

/**
 * @brief Check if type is a Matrix
 */
template<typename T>
struct is_matrix : std::false_type {};

template<typename T, std::size_t M, std::size_t N>
struct is_matrix<Matrix<T, M, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_matrix_v = is_matrix<T>::value;

/**
 * @brief Check if type is a MatrixExpr
 */
template<typename T>
struct is_matrix_expr : std::bool_constant<
    std::is_base_of_v<MatrixExpr<std::decay_t<T>>, std::decay_t<T>> &&
    !is_matrix_v<std::decay_t<T>>
> {};

template<typename T>
inline constexpr bool is_matrix_expr_v = is_matrix_expr<T>::value;

/**
 * @brief Check if type is matrix-like (Matrix or MatrixExpr)
 */
template<typename T>
struct is_matrix_like : std::bool_constant<is_matrix_v<T> || is_matrix_expr_v<T>> {};

template<typename T>
inline constexpr bool is_matrix_like_v = is_matrix_like<T>::value;

/**
 * @brief Check if type is a square matrix
 */
template<typename T>
struct is_square_matrix : std::false_type {};

template<typename T, std::size_t N>
struct is_square_matrix<Matrix<T, N, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_square_matrix_v = is_square_matrix<T>::value;

/**
 * @brief Check if type is a Tensor
 */
template<typename T>
struct is_tensor : std::false_type {};

template<typename T, std::size_t R, std::size_t D>
struct is_tensor<TensorBase<T, R, D>> : std::true_type {};

template<typename T, std::size_t D>
struct is_tensor<Tensor2<T, D>> : std::true_type {};

template<typename T, std::size_t D>
struct is_tensor<Tensor4<T, D>> : std::true_type {};

template<typename T>
inline constexpr bool is_tensor_v = is_tensor<T>::value;

/**
 * @brief Extract scalar type from mathematical types
 */
template<typename T>
struct scalar_type {
    using type = T;
};

template<typename T, std::size_t N>
struct scalar_type<Vector<T, N>> {
    using type = T;
};

template<typename T, std::size_t M, std::size_t N>
struct scalar_type<Matrix<T, M, N>> {
    using type = T;
};

template<typename T, std::size_t R, std::size_t D>
struct scalar_type<TensorBase<T, R, D>> {
    using type = T;
};

template<typename T, std::size_t D>
struct scalar_type<Tensor2<T, D>> {
    using type = T;
};

template<typename T, std::size_t D>
struct scalar_type<Tensor4<T, D>> {
    using type = T;
};

template<typename T>
using scalar_type_t = typename scalar_type<T>::type;

/**
 * @brief Get dimensions of mathematical types
 */
template<typename T>
struct dimensions {
    static constexpr std::size_t rank = 0;
    static constexpr std::size_t rows = 0;
    static constexpr std::size_t cols = 0;
    static constexpr std::size_t size = 0;
};

template<typename T, std::size_t N>
struct dimensions<Vector<T, N>> {
    static constexpr std::size_t rank = 1;
    static constexpr std::size_t rows = N;
    static constexpr std::size_t cols = 1;
    static constexpr std::size_t size = N;
};

template<typename T, std::size_t M, std::size_t N>
struct dimensions<Matrix<T, M, N>> {
    static constexpr std::size_t rank = 2;
    static constexpr std::size_t rows = M;
    static constexpr std::size_t cols = N;
    static constexpr std::size_t size = M * N;
};

template<typename T, std::size_t R, std::size_t D>
struct dimensions<TensorBase<T, R, D>> {
    static constexpr std::size_t rank = R;
    static constexpr std::size_t rows = 0;
    static constexpr std::size_t cols = 0;
    static constexpr std::size_t dim = D;

    // Compile-time power function for tensor size
    static constexpr std::size_t ipow(std::size_t base, std::size_t exp) {
        return exp == 0 ? 1 : base * ipow(base, exp - 1);
    }

    static constexpr std::size_t size = ipow(D, R);
};

template<typename T, std::size_t D>
struct dimensions<Tensor2<T, D>> : dimensions<TensorBase<T, 2, D>> {};

template<typename T, std::size_t D>
struct dimensions<Tensor4<T, D>> : dimensions<TensorBase<T, 4, D>> {};

/**
 * @brief Convenience variable templates for dimension access
 */
template<typename T>
inline constexpr std::size_t vector_dimension_v = dimensions<T>::size;

template<typename T>
inline constexpr std::size_t matrix_rows_v = dimensions<T>::rows;

template<typename T>
inline constexpr std::size_t matrix_cols_v = dimensions<T>::cols;

/**
 * @brief Convenience type aliases for standard library traits
 */
template<typename T>
inline constexpr bool is_scalar_v = std::is_arithmetic_v<T>;

template<typename T>
inline constexpr bool is_floating_point_v = std::is_floating_point_v<T>;

template<typename T>
inline constexpr bool is_integral_v = std::is_integral_v<T>;

template<typename T1, typename T2>
using common_type_t = std::common_type_t<T1, T2>;

/**
 * @brief Check if two vectors can be added (same dimension)
 */
template<typename V1, typename V2>
inline constexpr bool can_add_vectors_v =
    is_vector_v<V1> && is_vector_v<V2> &&
    (dimensions<V1>::size == dimensions<V2>::size);

/**
 * @brief Check if two matrices can be multiplied
 */
template<typename M1, typename M2>
inline constexpr bool can_multiply_matrices_v =
    is_matrix_v<M1> && is_matrix_v<M2> &&
    (dimensions<M1>::cols == dimensions<M2>::rows);

/**
 * @brief Enable if type is arithmetic
 */
template<typename T>
using enable_if_arithmetic_t = std::enable_if_t<std::is_arithmetic_v<T>>;

/**
 * @brief Enable if type is floating point
 */
template<typename T>
using enable_if_floating_point_t = std::enable_if_t<std::is_floating_point_v<T>>;

/**
 * @brief Enable if type is integral
 */
template<typename T>
using enable_if_integral_t = std::enable_if_t<std::is_integral_v<T>>;

/**
 * @brief Enable if type is a vector
 */
template<typename T>
using enable_if_vector_t = std::enable_if_t<is_vector_v<T>>;

/**
 * @brief Enable if type is a matrix
 */
template<typename T>
using enable_if_matrix_t = std::enable_if_t<is_matrix_v<T>>;

/**
 * @brief Enable if type is a square matrix
 */
template<typename T>
using enable_if_square_matrix_t = std::enable_if_t<is_square_matrix_v<T>>;

/**
 * @brief Enable if type is a tensor
 */
template<typename T>
using enable_if_tensor_t = std::enable_if_t<is_tensor_v<T>>;

/**
 * @brief Check if two types have compatible dimensions for operations
 */
template<typename T1, typename T2>
struct are_compatible_vectors : std::false_type {};

template<typename T, std::size_t N>
struct are_compatible_vectors<Vector<T, N>, Vector<T, N>> : std::true_type {};

template<typename T1, typename T2>
inline constexpr bool are_compatible_vectors_v = are_compatible_vectors<T1, T2>::value;

/**
 * @brief Check if matrix and vector are compatible for multiplication
 */
template<typename MatType, typename VecType>
struct are_compatible_for_mult : std::false_type {};

template<typename T, std::size_t M, std::size_t N>
struct are_compatible_for_mult<Matrix<T, M, N>, Vector<T, N>> : std::true_type {};

template<typename MatType, typename VecType>
inline constexpr bool are_compatible_for_mult_v = are_compatible_for_mult<MatType, VecType>::value;

/**
 * @brief Check if two matrices are compatible for multiplication
 */
template<typename Mat1, typename Mat2>
struct are_compatible_matrices_for_mult : std::false_type {};

template<typename T, std::size_t M, std::size_t N, std::size_t P>
struct are_compatible_matrices_for_mult<Matrix<T, M, N>, Matrix<T, N, P>> : std::true_type {};

template<typename Mat1, typename Mat2>
inline constexpr bool are_compatible_matrices_for_mult_v = are_compatible_matrices_for_mult<Mat1, Mat2>::value;

/**
 * @brief Detect if type has member function
 */
template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>> : std::true_type {};

template<typename T>
inline constexpr bool has_size_v = has_size<T>::value;

/**
 * @brief Detect if type has subscript operator
 */
template<typename T, typename = void>
struct has_subscript : std::false_type {};

template<typename T>
struct has_subscript<T, std::void_t<decltype(std::declval<T>()[0])>> : std::true_type {};

template<typename T>
inline constexpr bool has_subscript_v = has_subscript<T>::value;

/**
 * @brief Detect if type has call operator (for matrices)
 */
template<typename T, typename = void>
struct has_call_operator : std::false_type {};

template<typename T>
struct has_call_operator<T, std::void_t<decltype(std::declval<T>()(0, 0))>> : std::true_type {};

template<typename T>
inline constexpr bool has_call_operator_v = has_call_operator<T>::value;

/**
 * @brief Result type of binary operations
 */
template<typename T1, typename T2>
using binary_op_result_t = std::decay_t<decltype(std::declval<T1>() + std::declval<T2>())>;

/**
 * @brief Check if type supports SIMD operations
 */
template<typename T>
struct supports_simd : std::false_type {};

// Specialize for types that support SIMD
template<>
struct supports_simd<float> : std::true_type {};

template<>
struct supports_simd<double> : std::true_type {};

template<typename T>
inline constexpr bool supports_simd_v = supports_simd<T>::value;

/**
 * @brief Get SIMD alignment for type
 */
template<typename T>
struct simd_alignment {
    static constexpr std::size_t value = alignof(T);
};

// Specialize for SIMD-capable types (32-byte alignment for AVX)
template<>
struct simd_alignment<float> {
    static constexpr std::size_t value = 32;
};

template<>
struct simd_alignment<double> {
    static constexpr std::size_t value = 32;
};

template<typename T>
inline constexpr std::size_t simd_alignment_v = simd_alignment<T>::value;

/**
 * @brief Detect expression template types
 */
template<typename T>
struct is_expression_template : std::bool_constant<
    is_vector_expr_v<T> || is_matrix_expr_v<T>
> {};

template<typename T>
inline constexpr bool is_expression_template_v = is_expression_template<T>::value;

/**
 * @brief Common type for mixed precision operations
 */
template<typename T1, typename T2>
using common_floating_point_t = std::common_type_t<
    std::conditional_t<std::is_floating_point_v<T1>, T1, double>,
    std::conditional_t<std::is_floating_point_v<T2>, T2, double>
>;

/**
 * @brief Check if type is complex (for future extension)
 */
template<typename T>
struct is_complex : std::false_type {};

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

/**
 * @brief Check if type supports autodiff (for future extension)
 */
template<typename T>
struct supports_autodiff : std::false_type {};

template<typename T>
inline constexpr bool supports_autodiff_v = supports_autodiff<T>::value;

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_TYPE_TRAITS_H
