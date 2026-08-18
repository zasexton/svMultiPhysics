#ifndef SVMP_FE_MATH_VECTOR_EXPR_H
#define SVMP_FE_MATH_VECTOR_EXPR_H

/**
 * @file VectorExpr.h
 * @brief Expression template infrastructure for lazy evaluation of vector operations
 *
 * This header provides expression templates that enable compound vector operations
 * without creating temporary objects. Operations are evaluated lazily at the point
 * of assignment, eliminating intermediate allocations and improving performance.
 */

#include <cstddef>
#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>
#include "ExpressionOps.h"

namespace svmp {
namespace FE {
namespace math {

template<typename T, std::size_t N>
class Vector;

/**
 * @brief Base class for all vector expressions using CRTP
 * @tparam Derived The derived expression type
 *
 * This uses the Curiously Recurring Template Pattern (CRTP) to provide
 * static polymorphism for expression templates.
 */
template<typename Derived>
class VectorExpr {
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
     * @brief Access element by index
     * @param i Element index
     * @return Value at index i
     */
    auto operator[](std::size_t i) const {
        return derived()[i];
    }

    /**
     * @brief Get the size of the vector expression
     * @return Number of elements
     */
    std::size_t size() const {
        return derived().size();
    }
};

namespace detail {

template<typename T>
using VectorExprValue = std::remove_cv_t<std::remove_reference_t<T>>;

template<typename T>
inline constexpr bool is_vector_expression_v =
    std::is_base_of_v<VectorExpr<VectorExprValue<T>>, VectorExprValue<T>>;

template<typename T>
struct IsConcreteVector : std::false_type {};

template<typename T, std::size_t N>
struct IsConcreteVector<Vector<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_concrete_vector_v =
    IsConcreteVector<VectorExprValue<T>>::value;

/**
 * @brief Stores lvalue operands by reference and temporary operands by value.
 *
 * Expression nodes may outlive the full expression that created their child
 * nodes (for example, `auto expr = a + b - c`).  Owning temporary children
 * keeps those chains valid without copying lvalue vector leaves.
 */
template<typename Operand>
class VectorExprOperand {
private:
    using value_type = VectorExprValue<Operand>;
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
    constexpr explicit VectorExprOperand(Operand&& operand)
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
 * @brief Binary expression for element-wise operations between two vector expressions
 * @tparam LHS Left-hand side expression type
 * @tparam RHS Right-hand side expression type
 * @tparam Op Binary operation functor
 */
template<typename LHS, typename RHS, typename Op>
class VectorBinaryExpr : public VectorExpr<VectorBinaryExpr<LHS, RHS, Op>> {
private:
    detail::VectorExprOperand<LHS> lhs_;
    detail::VectorExprOperand<RHS> rhs_;
    Op op_;

public:
    /**
     * @brief Construct binary expression
     * @param lhs Left operand
     * @param rhs Right operand
     * @param op Operation to apply
     */
    constexpr VectorBinaryExpr(LHS&& lhs, RHS&& rhs, Op op = Op{})
        : lhs_(std::forward<LHS>(lhs)),
          rhs_(std::forward<RHS>(rhs)),
          op_(op)
    {
    }

    /**
     * @brief Access element at index
     * @param i Element index
     * @return Result of operation on elements at index i
     */
    constexpr auto operator[](std::size_t i) const {
        return op_(lhs_.get()[i], rhs_.get()[i]);
    }

    /**
     * @brief Get size of expression (from left operand)
     * @return Number of elements
     */
    constexpr std::size_t size() const {
        return lhs_.get().size();
    }
};

/**
 * @brief Unary expression for element-wise operations on a single vector expression
 * @tparam Expr Expression type
 * @tparam Op Unary operation functor
 */
template<typename Expr, typename Op>
class VectorUnaryExpr : public VectorExpr<VectorUnaryExpr<Expr, Op>> {
private:
    detail::VectorExprOperand<Expr> expr_;
    Op op_;

public:
    /**
     * @brief Construct unary expression
     * @param expr Operand expression
     * @param op Operation to apply
     */
    constexpr VectorUnaryExpr(Expr&& expr, Op op = Op{})
        : expr_(std::forward<Expr>(expr)), op_(op)
    {
    }

    /**
     * @brief Access element at index
     * @param i Element index
     * @return Result of operation on element at index i
     */
    constexpr auto operator[](std::size_t i) const {
        return op_(expr_.get()[i]);
    }

    /**
     * @brief Get size of expression
     * @return Number of elements
     */
    constexpr std::size_t size() const {
        return expr_.get().size();
    }
};

/**
 * @brief Scalar multiplication expression
 * @tparam Expr Vector expression type
 * @tparam Scalar Scalar type
 */
template<typename Expr, typename Scalar>
class VectorScalarExpr : public VectorExpr<VectorScalarExpr<Expr, Scalar>> {
private:
    detail::VectorExprOperand<Expr> expr_;
    Scalar scalar_;

public:
    /**
     * @brief Construct scalar multiplication expression
     * @param expr Vector expression
     * @param scalar Scalar value
     */
    constexpr VectorScalarExpr(Expr&& expr, Scalar scalar)
        : expr_(std::forward<Expr>(expr)), scalar_(scalar)
    {
    }

    /**
     * @brief Access element at index
     * @param i Element index
     * @return Element multiplied by scalar
     */
    constexpr auto operator[](std::size_t i) const {
        return expr_.get()[i] * scalar_;
    }

    /**
     * @brief Get size of expression
     * @return Number of elements
     */
    constexpr std::size_t size() const {
        return expr_.get().size();
    }
};

/**
 * @brief Scalar division expression
 * @tparam Expr Vector expression type
 * @tparam Scalar Scalar type
 */
template<typename Expr, typename Scalar>
class VectorScalarDivExpr : public VectorExpr<VectorScalarDivExpr<Expr, Scalar>> {
private:
    detail::VectorExprOperand<Expr> expr_;
    Scalar scalar_;

public:
    /**
     * @brief Construct scalar division expression
     * @param expr Vector expression
     * @param scalar Scalar divisor
     */
    constexpr VectorScalarDivExpr(Expr&& expr, Scalar scalar)
        : expr_(std::forward<Expr>(expr)), scalar_(scalar)
    {
    }

    /**
     * @brief Access element at index
     * @param i Element index
     * @return Element divided by scalar
     */
    constexpr auto operator[](std::size_t i) const {
        return expr_.get()[i] / scalar_;
    }

    /**
     * @brief Get size of expression
     * @return Number of elements
     */
    constexpr std::size_t size() const {
        return expr_.get().size();
    }
};

/**
 * @brief Addition operator for vector expressions
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_vector_expression_v<LHS> &&
             detail::is_vector_expression_v<RHS>, int> = 0>
constexpr auto operator+(LHS&& lhs, RHS&& rhs) {
    return VectorBinaryExpr<LHS, RHS, detail::ops::Add>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Add{}
    );
}

/**
 * @brief Subtraction operator for vector expressions
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_vector_expression_v<LHS> &&
             detail::is_vector_expression_v<RHS>, int> = 0>
constexpr auto operator-(LHS&& lhs, RHS&& rhs) {
    return VectorBinaryExpr<LHS, RHS, detail::ops::Sub>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Sub{}
    );
}

/**
 * @brief Element-wise multiplication operator for vector expressions
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_vector_expression_v<LHS> &&
             detail::is_vector_expression_v<RHS>, int> = 0>
constexpr auto hadamard(LHS&& lhs, RHS&& rhs) {
    return VectorBinaryExpr<LHS, RHS, detail::ops::Mul>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Mul{}
    );
}

/**
 * @brief Element-wise division operator for vector expressions
 */
template<typename LHS, typename RHS,
         std::enable_if_t<
             detail::is_vector_expression_v<LHS> &&
             detail::is_vector_expression_v<RHS>, int> = 0>
constexpr auto hadamard_div(LHS&& lhs, RHS&& rhs) {
    return VectorBinaryExpr<LHS, RHS, detail::ops::Div>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), detail::ops::Div{}
    );
}

/**
 * @brief Negation operator for vector expressions
 */
template<typename Expr,
         std::enable_if_t<detail::is_vector_expression_v<Expr>, int> = 0>
constexpr auto operator-(Expr&& expr) {
    return VectorUnaryExpr<Expr, detail::ops::Negate>(
        std::forward<Expr>(expr), detail::ops::Negate{}
    );
}

/**
 * @brief Scalar multiplication operator (vector * scalar)
 */
template<typename Expr, typename Scalar,
         std::enable_if_t<
             detail::is_vector_expression_v<Expr> &&
             std::is_arithmetic_v<Scalar>, int> = 0>
constexpr auto operator*(Expr&& expr, Scalar scalar) {
    return VectorScalarExpr<Expr, Scalar>(std::forward<Expr>(expr), scalar);
}

/**
 * @brief Scalar multiplication operator (scalar * vector)
 */
template<typename Scalar, typename Expr,
         std::enable_if_t<
             std::is_arithmetic_v<Scalar> &&
             detail::is_vector_expression_v<Expr>, int> = 0>
constexpr auto operator*(Scalar scalar, Expr&& expr) {
    return VectorScalarExpr<Expr, Scalar>(std::forward<Expr>(expr), scalar);
}

/**
 * @brief Scalar division operator (vector / scalar)
 */
template<typename Expr, typename Scalar,
         std::enable_if_t<
             detail::is_vector_expression_v<Expr> &&
             std::is_arithmetic_v<Scalar>, int> = 0>
constexpr auto operator/(Expr&& expr, Scalar scalar) {
    return VectorScalarDivExpr<Expr, Scalar>(std::forward<Expr>(expr), scalar);
}

/**
 * @brief Element-wise absolute value
 */
template<typename Expr,
         std::enable_if_t<
             detail::is_vector_expression_v<Expr> &&
             !detail::is_concrete_vector_v<Expr>, int> = 0>
constexpr auto abs(Expr&& expr) {
    return VectorUnaryExpr<Expr, detail::ops::Abs>(
        std::forward<Expr>(expr), detail::ops::Abs{});
}

/**
 * @brief Element-wise square root
 */
template<typename Expr,
         std::enable_if_t<detail::is_vector_expression_v<Expr>, int> = 0>
constexpr auto sqrt(Expr&& expr) {
    return VectorUnaryExpr<Expr, detail::ops::Sqrt>(
        std::forward<Expr>(expr), detail::ops::Sqrt{});
}

/**
 * @brief Dot product for vector expressions
 * @tparam LHS Left vector expression type
 * @tparam RHS Right vector expression type
 * @param lhs Left operand
 * @param rhs Right operand
 * @return Dot product result
 */
template<typename LHS, typename RHS,
         typename = std::enable_if_t<
             std::is_base_of_v<VectorExpr<LHS>, LHS> &&
             std::is_base_of_v<VectorExpr<RHS>, RHS>
         >>
constexpr auto dot(const VectorExpr<LHS>& lhs, const VectorExpr<RHS>& rhs) {
    using result_type = decltype(lhs.derived()[0] * rhs.derived()[0]);
    result_type sum = result_type{0};
    const auto n = lhs.size();
    for (std::size_t i = 0; i < n; ++i) {
        sum += lhs.derived()[i] * rhs.derived()[i];
    }
    return sum;
}

/**
 * @brief Compute norm squared of vector expression
 * @tparam Expr Vector expression type
 * @param expr Vector expression
 * @return Square of the Euclidean norm
 */
template<typename Expr,
         typename = std::enable_if_t<
             std::is_base_of_v<VectorExpr<Expr>, Expr>
         >>
constexpr auto norm_squared(const VectorExpr<Expr>& expr) {
    return dot(expr, expr);
}

/**
 * @brief Compute norm of vector expression
 * @tparam Expr Vector expression type
 * @param expr Vector expression
 * @return Euclidean norm
 */
template<typename Expr,
         typename = std::enable_if_t<
             std::is_base_of_v<VectorExpr<Expr>, Expr>
         >>
constexpr auto norm(const VectorExpr<Expr>& expr) {
    using std::sqrt;
    return sqrt(norm_squared(expr));
}

/**
 * @brief Normalize vector expression
 * @tparam Expr Vector expression type
 * @param expr Vector expression
 * @return Normalized vector expression
 */
template<typename Expr,
         typename = std::enable_if_t<
             std::is_base_of_v<VectorExpr<Expr>, Expr>
         >>
constexpr auto normalize(const VectorExpr<Expr>& expr) {
    return expr.derived() / norm(expr);
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_VECTOR_EXPR_H
