#ifndef SVMP_FE_CONSTITUTIVE_EXPRESSION_LAW_H
#define SVMP_FE_CONSTITUTIVE_EXPRESSION_LAW_H

#include "Constitutive/LawAdapters.h"

#include "Forms/Dual.h"

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constitutive {

/**
 * @brief Tiny expression wrapper for scalar (Real or Dual) local laws
 *
 * This enables "expression-style" constitutive authoring with a single code
 * path for Real and Dual by encapsulating Dual workspace allocation.
 *
 * The intent is similar to UFL-style local expressions, but evaluated eagerly
 * (not symbolic): each algebraic operation allocates the next Dual temporary.
 */
template <class Scalar, class Workspace>
class ScalarExpr {
public:
    using ScalarType = Scalar;
    using WorkspaceType = Workspace;

    ScalarExpr(const Scalar& value, Workspace& ws) : value_(value), ws_(&ws) {}

    [[nodiscard]] const Scalar& value() const noexcept { return value_; }
    [[nodiscard]] Workspace& workspace() const noexcept { return *ws_; }

    [[nodiscard]] ScalarExpr constant(Real c) const
    {
        if constexpr (std::is_same_v<Scalar, forms::Dual>) {
            return ScalarExpr(forms::makeDualConstant(c, ws_->alloc()), *ws_);
        } else {
            return ScalarExpr(static_cast<Scalar>(c), *ws_);
        }
    }

private:
    Scalar value_{};
    Workspace* ws_{nullptr};
};

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator-(const ScalarExpr<Scalar, Workspace>& a)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::neg(a.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(-a.value(), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator+(const ScalarExpr<Scalar, Workspace>& a,
                                                             const ScalarExpr<Scalar, Workspace>& b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::add(a.value(), b.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() + b.value(), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator+(const ScalarExpr<Scalar, Workspace>& a, Real b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        forms::Dual out = forms::makeDualConstant(0.0, a.workspace().alloc());
        out.value = a.value().value + b;
        const std::size_t n = out.deriv.size();
        for (std::size_t k = 0; k < n; ++k) out.deriv[k] = a.value().deriv[k];
        return ScalarExpr<Scalar, Workspace>(out, a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() + static_cast<Scalar>(b), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator+(Real a, const ScalarExpr<Scalar, Workspace>& b)
{
    return b + a;
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator-(const ScalarExpr<Scalar, Workspace>& a,
                                                             const ScalarExpr<Scalar, Workspace>& b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::sub(a.value(), b.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() - b.value(), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator-(const ScalarExpr<Scalar, Workspace>& a, Real b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        forms::Dual out = forms::makeDualConstant(0.0, a.workspace().alloc());
        out.value = a.value().value - b;
        const std::size_t n = out.deriv.size();
        for (std::size_t k = 0; k < n; ++k) out.deriv[k] = a.value().deriv[k];
        return ScalarExpr<Scalar, Workspace>(out, a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() - static_cast<Scalar>(b), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator-(Real a, const ScalarExpr<Scalar, Workspace>& b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        forms::Dual out = forms::makeDualConstant(0.0, b.workspace().alloc());
        out.value = a - b.value().value;
        const std::size_t n = out.deriv.size();
        for (std::size_t k = 0; k < n; ++k) out.deriv[k] = -b.value().deriv[k];
        return ScalarExpr<Scalar, Workspace>(out, b.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(static_cast<Scalar>(a) - b.value(), b.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator*(const ScalarExpr<Scalar, Workspace>& a,
                                                             const ScalarExpr<Scalar, Workspace>& b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::mul(a.value(), b.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() * b.value(), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator*(const ScalarExpr<Scalar, Workspace>& a, Real b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::mul(a.value(), b, forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() * static_cast<Scalar>(b), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator*(Real a, const ScalarExpr<Scalar, Workspace>& b)
{
    return b * a;
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator/(const ScalarExpr<Scalar, Workspace>& a,
                                                             const ScalarExpr<Scalar, Workspace>& b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::div(a.value(), b.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() / b.value(), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator/(const ScalarExpr<Scalar, Workspace>& a, Real b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::div(a.value(), b, forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(a.value() / static_cast<Scalar>(b), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> operator/(Real a, const ScalarExpr<Scalar, Workspace>& b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::div(a, b.value(), forms::makeDualConstant(0.0, b.workspace().alloc())),
                                             b.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(static_cast<Scalar>(a) / b.value(), b.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> abs(const ScalarExpr<Scalar, Workspace>& a)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::abs(a.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(std::abs(a.value()), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> sign(const ScalarExpr<Scalar, Workspace>& a)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::sign(a.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        const Scalar v = a.value();
        const Scalar s = (v > 0) ? Scalar(1) : ((v < 0) ? Scalar(-1) : Scalar(0));
        return ScalarExpr<Scalar, Workspace>(s, a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> sqrt(const ScalarExpr<Scalar, Workspace>& a)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::sqrt(a.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(std::sqrt(a.value()), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> exp(const ScalarExpr<Scalar, Workspace>& a)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::exp(a.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(std::exp(a.value()), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> log(const ScalarExpr<Scalar, Workspace>& a)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::log(a.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(std::log(a.value()), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> pow(const ScalarExpr<Scalar, Workspace>& a,
                                                       const ScalarExpr<Scalar, Workspace>& b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::pow(a.value(), b.value(), forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(std::pow(a.value(), b.value()), a.workspace());
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline ScalarExpr<Scalar, Workspace> pow(const ScalarExpr<Scalar, Workspace>& a, Real b)
{
    if constexpr (std::is_same_v<Scalar, forms::Dual>) {
        return ScalarExpr<Scalar, Workspace>(forms::pow(a.value(), b, forms::makeDualConstant(0.0, a.workspace().alloc())),
                                             a.workspace());
    } else {
        return ScalarExpr<Scalar, Workspace>(std::pow(a.value(), static_cast<Scalar>(b)), a.workspace());
    }
}

// ---------------------------------------------------------------------------
// Expression-defined scalar local laws
// ---------------------------------------------------------------------------

template <class Builder>
class ExpressionScalarLaw final : public ScalarLawCRTP<ExpressionScalarLaw<Builder>> {
public:
    explicit ExpressionScalarLaw(std::string name, Builder builder, std::vector<params::Spec> specs = {})
        : name_(std::move(name))
        , builder_(std::move(builder))
        , specs_(std::move(specs))
    {
    }

    template <class Scalar, class Workspace>
    [[nodiscard]] Scalar evalScalar(const Scalar& x, const forms::ConstitutiveEvalContext& ctx, Workspace& ws) const
    {
        const ScalarExpr<Scalar, Workspace> in(x, ws);
        return builder_(in, ctx).value();
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override { return specs_; }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    std::string name_{};
    Builder builder_{};
    std::vector<params::Spec> specs_{};
};

template <class Builder>
[[nodiscard]] inline std::shared_ptr<const forms::ConstitutiveModel> makeExpressionScalarLaw(std::string name,
                                                                                             Builder builder,
                                                                                             std::vector<params::Spec> specs = {})
{
    return std::make_shared<ExpressionScalarLaw<Builder>>(std::move(name), std::move(builder), std::move(specs));
}

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_EXPRESSION_LAW_H
