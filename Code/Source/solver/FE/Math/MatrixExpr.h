#ifndef SVMP_FE_MATH_MATRIX_EXPR_H
#define SVMP_FE_MATH_MATRIX_EXPR_H

/**
 * @file MatrixExpr.h
 * @brief Expression template infrastructure for lazy evaluation of matrix operations
 *
 * This header provides expression templates that enable compound matrix operations
 * without creating temporary objects. Operations are evaluated lazily at the point
 * of assignment, eliminating intermediate allocations and improving performance.
 */

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>
#include "ExpressionOps.h"

namespace svmp {
namespace FE {
namespace math {

template<typename T, std::size_t M, std::size_t N>
class Matrix;

/**
 * @brief Base class for all matrix expressions using CRTP
 * @tparam Derived The derived expression type
 *
 * This uses the Curiously Recurring Template Pattern (CRTP) to provide
 * static polymorphism for expression templates.
 */
template<typename Derived>
class MatrixExpr {
public:
    /**
     * @brief Get the derived expression
     * @return Reference to the derived type
     */
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    /**
     * @brief Get the derived expression (non-const)
     * @return Reference to the derived type
     */
    Derived& derived() {
        return static_cast<Derived&>(*this);
    }

    /**
     * @brief Access element by row and column indices
     * @param i Row index
     * @param j Column index
     * @return Value at (i,j)
     */
    auto operator()(std::size_t i, std::size_t j) const {
        return derived()(i, j);
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    std::size_t rows() const {
        return derived().rows();
    }

    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    std::size_t cols() const {
        return derived().cols();
    }
};

namespace detail {

template<typename T>
using MatrixExprValue = std::remove_cv_t<std::remove_reference_t<T>>;

template<typename T>
inline constexpr bool is_matrix_expression_v =
    std::is_base_of_v<MatrixExpr<MatrixExprValue<T>>, MatrixExprValue<T>>;

template<typename T>
struct IsConcreteMatrix : std::false_type {};

template<typename T, std::size_t M, std::size_t N>
struct IsConcreteMatrix<Matrix<T, M, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_concrete_matrix_v =
    IsConcreteMatrix<MatrixExprValue<T>>::value;

/**
 * @brief Stores lvalue operands by reference and temporary operands by value.
 *
 * Owning temporary child nodes keeps a stored matrix expression valid after
 * the full expression that created it has ended, while matrix lvalues remain
 * lazily referenced.
 */
template<typename Operand>
class MatrixExprOperand {
private:
    using value_type = MatrixExprValue<Operand>;
    static constexpr bool stores_reference =
        std::is_lvalue_reference_v<Operand>;
    using storage_type =
        std::conditional_t<stores_reference, const value_type*, value_type>;

    static constexpr storage_type makeStorage(Operand&& operand)
    {
        if constexpr (stores_reference) {
            return std::addressof(operand);
        } else {
            return std::forward<Operand>(operand);
        }
    }

    storage_type storage_;

public:
    constexpr explicit MatrixExprOperand(Operand&& operand)
        : storage_(makeStorage(std::forward<Operand>(operand)))
    {
    }

    [[nodiscard]] constexpr const value_type& get() const noexcept
    {
        if constexpr (stores_reference) {
            return *storage_;
        } else {
            return storage_;
        }
    }
};

} // namespace detail

/**
 * @brief Binary expression for element-wise operations between two matrix expressions
 * @tparam LHS Left-hand side expression type
 * @tparam RHS Right-hand side expression type
 * @tparam Op Binary operation functor
 */
template<typename LHS, typename RHS, typename Op>
class MatrixBinaryExpr : public MatrixExpr<MatrixBinaryExpr<LHS, RHS, Op>> {
private:
    detail::MatrixExprOperand<LHS> lhs_;
    detail::MatrixExprOperand<RHS> rhs_;
    Op op_;

public:
    /**
     * @brief Construct binary expression
     * @param lhs Left operand
     * @param rhs Right operand
     * @param op Operation to apply
     */
    constexpr MatrixBinaryExpr(LHS&& lhs, RHS&& rhs, Op op = Op{})
        : lhs_(std::forward<LHS>(lhs)),
          rhs_(std::forward<RHS>(rhs)),
          op_(op)
    {
    }

    /**
     * @brief Access element at (i,j)
     * @param i Row index
     * @param j Column index
     * @return Result of operation on elements at (i,j)
     */
    constexpr auto operator()(std::size_t i, std::size_t j) const {
        return op_(lhs_.get()(i, j), rhs_.get()(i, j));
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    constexpr std::size_t rows() const {
        return lhs_.get().rows();
    }

    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    constexpr std::size_t cols() const {
        return lhs_.get().cols();
    }
};

/**
 * @brief Unary expression for element-wise operations on a single matrix expression
 * @tparam Expr Expression type
 * @tparam Op Unary operation functor
 */
template<typename Expr, typename Op>
class MatrixUnaryExpr : public MatrixExpr<MatrixUnaryExpr<Expr, Op>> {
private:
    detail::MatrixExprOperand<Expr> expr_;
    Op op_;

public:
    /**
     * @brief Construct unary expression
     * @param expr Operand expression
     * @param op Operation to apply
     */
    constexpr MatrixUnaryExpr(Expr&& expr, Op op = Op{})
        : expr_(std::forward<Expr>(expr)), op_(op)
    {
    }

    /**
     * @brief Access element at (i,j)
     * @param i Row index
     * @param j Column index
     * @return Result of operation on element at (i,j)
     */
    constexpr auto operator()(std::size_t i, std::size_t j) const {
        return op_(expr_.get()(i, j));
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    constexpr std::size_t rows() const {
        return expr_.get().rows();
    }

    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    constexpr std::size_t cols() const {
        return expr_.get().cols();
    }
};

/**
 * @brief Scalar multiplication expression
 * @tparam Expr Matrix expression type
 * @tparam Scalar Scalar type
 */
template<typename Expr, typename Scalar>
class MatrixScalarExpr : public MatrixExpr<MatrixScalarExpr<Expr, Scalar>> {
private:
    detail::MatrixExprOperand<Expr> expr_;
    Scalar scalar_;

public:
    /**
     * @brief Construct scalar multiplication expression
     * @param expr Matrix expression
     * @param scalar Scalar value
     */
    constexpr MatrixScalarExpr(Expr&& expr, Scalar scalar)
        : expr_(std::forward<Expr>(expr)), scalar_(scalar)
    {
    }

    /**
     * @brief Access element at (i,j)
     * @param i Row index
     * @param j Column index
     * @return Element multiplied by scalar
     */
    constexpr auto operator()(std::size_t i, std::size_t j) const {
        return expr_.get()(i, j) * scalar_;
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    constexpr std::size_t rows() const {
        return expr_.get().rows();
    }

    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    constexpr std::size_t cols() const {
        return expr_.get().cols();
    }
};

/**
 * @brief Scalar division expression
 * @tparam Expr Matrix expression type
 * @tparam Scalar Scalar type
 */
template<typename Expr, typename Scalar>
class MatrixScalarDivExpr : public MatrixExpr<MatrixScalarDivExpr<Expr, Scalar>> {
private:
    detail::MatrixExprOperand<Expr> expr_;
    Scalar scalar_;

public:
    /**
     * @brief Construct scalar division expression
     * @param expr Matrix expression
     * @param scalar Scalar divisor
     */
    constexpr MatrixScalarDivExpr(Expr&& expr, Scalar scalar)
        : expr_(std::forward<Expr>(expr)), scalar_(scalar)
    {
    }

    /**
     * @brief Access element at (i,j)
     * @param i Row index
     * @param j Column index
     * @return Element divided by scalar
     */
    constexpr auto operator()(std::size_t i, std::size_t j) const {
        return expr_.get()(i, j) / scalar_;
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    constexpr std::size_t rows() const {
        return expr_.get().rows();
    }

    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    constexpr std::size_t cols() const {
        return expr_.get().cols();
    }
};

/**
 * @brief Matrix multiplication expression (lazy evaluation)
 * @tparam LHS Left matrix expression type
 * @tparam RHS Right matrix expression type
 *
 * Computes matrix multiplication A*B lazily
 */
template<typename LHS, typename RHS>
class MatrixMulExpr : public MatrixExpr<MatrixMulExpr<LHS, RHS>> {
private:
    detail::MatrixExprOperand<LHS> lhs_;
    detail::MatrixExprOperand<RHS> rhs_;

public:
    /**
     * @brief Construct matrix multiplication expression
     * @param lhs Left matrix
     * @param rhs Right matrix
     */
    constexpr MatrixMulExpr(LHS&& lhs, RHS&& rhs)
        : lhs_(std::forward<LHS>(lhs)), rhs_(std::forward<RHS>(rhs))
    {
    }

    /**
     * @brief Compute element at (i,j)
     * @param i Row index
     * @param j Column index
     * @return Dot product of row i of lhs and column j of rhs
     */
    constexpr auto operator()(std::size_t i, std::size_t j) const {
        using result_type =
            decltype(lhs_.get()(0, 0) * rhs_.get()(0, 0));
        result_type sum = result_type{0};
        const auto n = lhs_.get().cols();
        for (std::size_t k = 0; k < n; ++k) {
            sum += lhs_.get()(i, k) * rhs_.get()(k, j);
        }
        return sum;
    }

    /**
     * @brief Get number of rows (from left matrix)
     * @return Number of rows
     */
    constexpr std::size_t rows() const {
        return lhs_.get().rows();
    }

    /**
     * @brief Get number of columns (from right matrix)
     * @return Number of columns
     */
    constexpr std::size_t cols() const {
        return rhs_.get().cols();
    }
};

/**
 * @brief Transpose expression (lazy evaluation)
 * @tparam Expr Matrix expression type
 */
template<typename Expr>
class TransposeExpr : public MatrixExpr<TransposeExpr<Expr>> {
private:
    detail::MatrixExprOperand<Expr> expr_;

public:
    /**
     * @brief Construct transpose expression
     * @param expr Matrix expression to transpose
     */
    constexpr explicit TransposeExpr(Expr&& expr)
        : expr_(std::forward<Expr>(expr))
    {
    }

    /**
     * @brief Access transposed element
     * @param i Row index (becomes column in original)
     * @param j Column index (becomes row in original)
     * @return Element at (j,i) of original matrix
     */
    constexpr auto operator()(std::size_t i, std::size_t j) const {
        return expr_.get()(j, i);
    }

    /**
     * @brief Get number of rows (columns of original)
     * @return Number of rows
     */
    constexpr std::size_t rows() const {
        return expr_.get().cols();
    }

    /**
     * @brief Get number of columns (rows of original)
     * @return Number of columns
     */
    constexpr std::size_t cols() const {
        return expr_.get().rows();
    }
};

/**
 * @brief Diagonal matrix expression (creates diagonal matrix from vector)
 * @tparam VecExpr Vector expression type
 */
template<typename VecExpr>
class DiagonalExpr : public MatrixExpr<DiagonalExpr<VecExpr>> {
private:
    const VecExpr& vec_;
    std::size_t n_;

public:
    /**
     * @brief Construct diagonal matrix from vector
     * @param vec Vector of diagonal elements
     * @param n Matrix dimension (default: vector size)
     */
    constexpr explicit DiagonalExpr(const VecExpr& vec, std::size_t n = 0)
        : vec_(vec), n_(n > 0 ? n : vec.size()) {}

    /**
     * @brief Access element
     * @param i Row index
     * @param j Column index
     * @return Diagonal element if i==j, zero otherwise
     */
    constexpr auto operator()(std::size_t i, std::size_t j) const {
        using result_type = decltype(vec_[0]);
        return (i == j && i < vec_.size()) ? vec_[i] : result_type{0};
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    constexpr std::size_t rows() const {
        return n_;
    }

    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    constexpr std::size_t cols() const {
        return n_;
    }
};

/**
 * @brief Addition operator for matrix expressions
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_matrix_expression_v<LHS> &&
             detail::is_matrix_expression_v<RHS>, int> = 0>
constexpr auto operator+(LHS&& lhs, RHS&& rhs) {
    return MatrixBinaryExpr<LHS, RHS, detail::ops::Add>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Add{}
    );
}

/**
 * @brief Subtraction operator for matrix expressions
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_matrix_expression_v<LHS> &&
             detail::is_matrix_expression_v<RHS>, int> = 0>
constexpr auto operator-(LHS&& lhs, RHS&& rhs) {
    return MatrixBinaryExpr<LHS, RHS, detail::ops::Sub>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Sub{}
    );
}

/**
 * @brief Matrix multiplication operator
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_matrix_expression_v<LHS> &&
             detail::is_matrix_expression_v<RHS> &&
             !(detail::is_concrete_matrix_v<LHS> &&
               detail::is_concrete_matrix_v<RHS>), int> = 0>
constexpr auto operator*(LHS&& lhs, RHS&& rhs) {
    return MatrixMulExpr<LHS, RHS>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

/**
 * @brief Element-wise multiplication (Hadamard product)
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_matrix_expression_v<LHS> &&
             detail::is_matrix_expression_v<RHS>, int> = 0>
constexpr auto hadamard(LHS&& lhs, RHS&& rhs) {
    return MatrixBinaryExpr<LHS, RHS, detail::ops::Mul>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Mul{}
    );
}

/**
 * @brief Element-wise division
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_matrix_expression_v<LHS> &&
             detail::is_matrix_expression_v<RHS>, int> = 0>
constexpr auto hadamard_div(LHS&& lhs, RHS&& rhs) {
    return MatrixBinaryExpr<LHS, RHS, detail::ops::Div>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Div{}
    );
}

/**
 * @brief Negation operator for matrix expressions
 */
template<typename Expr,
         std::enable_if_t<detail::is_matrix_expression_v<Expr>, int> = 0>
constexpr auto operator-(Expr&& expr) {
    return MatrixUnaryExpr<Expr, detail::ops::Negate>(
        std::forward<Expr>(expr), detail::ops::Negate{}
    );
}

/**
 * @brief Scalar multiplication operator (matrix * scalar)
 */
template<typename Expr, typename Scalar,
         std::enable_if_t<
             detail::is_matrix_expression_v<Expr> &&
             std::is_arithmetic_v<Scalar>, int> = 0>
constexpr auto operator*(Expr&& expr, Scalar scalar) {
    return MatrixScalarExpr<Expr, Scalar>(std::forward<Expr>(expr), scalar);
}

/**
 * @brief Scalar multiplication operator (scalar * matrix)
 */
template<typename Scalar, typename Expr,
         std::enable_if_t<
             std::is_arithmetic_v<Scalar> &&
             detail::is_matrix_expression_v<Expr>, int> = 0>
constexpr auto operator*(Scalar scalar, Expr&& expr) {
    return MatrixScalarExpr<Expr, Scalar>(std::forward<Expr>(expr), scalar);
}

/**
 * @brief Scalar division operator (matrix / scalar)
 */
template<typename Expr, typename Scalar,
         std::enable_if_t<
             detail::is_matrix_expression_v<Expr> &&
             std::is_arithmetic_v<Scalar>, int> = 0>
constexpr auto operator/(Expr&& expr, Scalar scalar) {
    return MatrixScalarDivExpr<Expr, Scalar>(std::forward<Expr>(expr), scalar);
}

/**
 * @brief Transpose function
 */
template<typename Expr,
         std::enable_if_t<
             detail::is_matrix_expression_v<Expr> &&
             !detail::is_concrete_matrix_v<Expr>, int> = 0>
constexpr auto transpose(Expr&& expr) {
    return TransposeExpr<Expr>(std::forward<Expr>(expr));
}

/**
 * @brief Element-wise absolute value
 */
template<typename Expr,
         std::enable_if_t<
             detail::is_matrix_expression_v<Expr> &&
             !detail::is_concrete_matrix_v<Expr>, int> = 0>
constexpr auto abs(Expr&& expr) {
    return MatrixUnaryExpr<Expr, detail::ops::Abs>(
        std::forward<Expr>(expr), detail::ops::Abs{});
}

/**
 * @brief Element-wise square root
 */
template<typename Expr,
         std::enable_if_t<detail::is_matrix_expression_v<Expr>, int> = 0>
constexpr auto sqrt(Expr&& expr) {
    return MatrixUnaryExpr<Expr, detail::ops::Sqrt>(
        std::forward<Expr>(expr), detail::ops::Sqrt{});
}

/**
 * @brief Compute Frobenius norm squared of matrix expression
 * @tparam Expr Matrix expression type
 * @param expr Matrix expression
 * @return Square of the Frobenius norm
 */
template<typename Expr,
         typename = std::enable_if_t<
             std::is_base_of_v<MatrixExpr<Expr>, Expr>
         >>
constexpr auto frobenius_norm_squared(const MatrixExpr<Expr>& expr) {
    using result_type = decltype(expr.derived()(0, 0) * expr.derived()(0, 0));
    result_type sum = result_type{0};
    const auto m = expr.rows();
    const auto n = expr.cols();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto val = expr.derived()(i, j);
            sum += val * val;
        }
    }
    return sum;
}

/**
 * @brief Compute Frobenius norm of matrix expression
 * @tparam Expr Matrix expression type
 * @param expr Matrix expression
 * @return Frobenius norm
 */
template<typename Expr,
         typename = std::enable_if_t<
             std::is_base_of_v<MatrixExpr<Expr>, Expr>
         >>
constexpr auto frobenius_norm(const MatrixExpr<Expr>& expr) {
    using std::sqrt;
    return sqrt(frobenius_norm_squared(expr));
}

/**
 * @brief Compute trace of square matrix expression
 * @tparam Expr Matrix expression type
 * @param expr Matrix expression
 * @return Sum of diagonal elements
 */
template<typename Expr,
         typename = std::enable_if_t<
             std::is_base_of_v<MatrixExpr<Expr>, Expr>
         >>
constexpr auto trace(const MatrixExpr<Expr>& expr) {
    using result_type = decltype(expr.derived()(0, 0));
    result_type sum = result_type{0};
    const auto n = std::min(expr.rows(), expr.cols());
    for (std::size_t i = 0; i < n; ++i) {
        sum += expr.derived()(i, i);
    }
    return sum;
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_MATRIX_EXPR_H
