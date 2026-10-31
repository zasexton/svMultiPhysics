#ifndef SVMP_FE_CONSTITUTIVE_EXPRESSION_LAW_H
#define SVMP_FE_CONSTITUTIVE_EXPRESSION_LAW_H

#include "Constitutive/LawAdapters.h"

#include "Analysis/ConstitutiveLawMetadata.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/Dual.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/InlinableConstitutiveModel.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
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

// ---------------------------------------------------------------------------
// Laws written as forms
// ---------------------------------------------------------------------------

/**
 * @brief A constitutive law written as forms
 *
 * Its body maps the input expressions of a call and the spatial dimension to
 * the output expression with forms operations.  A call of the law is a
 * constitutive node that setup expands into the body's tree
 * (InlinableConstitutiveModel), so the residual, the symbolic tangent and the
 * JIT see the law itself and need no external call.  Such a law has no
 * pointwise evaluation: evaluate() fails.  kindId() is the FNV-1a hash of the
 * kind name, which names the law; its parameters are constants of the tree.
 */
class FormLaw final : public forms::ConstitutiveModel, public forms::InlinableConstitutiveModel {
public:
    using Body = std::function<forms::FormExpr(std::span<const forms::FormExpr> inputs, int dim)>;

    struct Signature {
        std::vector<ValueKind> inputs{};
        ValueKind output{ValueKind::Scalar};
    };

    FormLaw(std::string kind_name,
            Signature signature,
            Body body,
            std::optional<analysis::ConstitutiveLawMetadata> metadata = std::nullopt)
        : kind_name_(std::move(kind_name))
        , signature_(std::move(signature))
        , body_(std::move(body))
        , metadata_(std::move(metadata))
    {
        if (kind_name_.empty()) {
            throw std::invalid_argument("FormLaw: the kind name must be non-empty");
        }
        if (signature_.inputs.empty()) {
            throw std::invalid_argument("FormLaw '" + kind_name_ + "': a law takes at least one input");
        }
        if (!body_) {
            throw std::invalid_argument("FormLaw '" + kind_name_ + "': missing body");
        }
    }

    [[nodiscard]] const std::string& kindName() const noexcept { return kind_name_; }

    [[nodiscard]] const forms::InlinableConstitutiveModel* inlinable() const noexcept override { return this; }
    [[nodiscard]] std::uint64_t kindId() const noexcept override { return fnv1a64(kind_name_); }
    [[nodiscard]] forms::MaterialStateAccess stateAccess() const noexcept override
    {
        return forms::MaterialStateAccess::None;
    }

    [[nodiscard]] forms::InlinedConstitutiveExpansion
    inlineExpand(std::span<const forms::FormExpr> inputs,
                 const forms::InlinableConstitutiveContext& ctx) const override
    {
        if (inputs.size() != signature_.inputs.size()) {
            throw std::invalid_argument("FormLaw '" + kind_name_ + "': expected " +
                                        std::to_string(signature_.inputs.size()) + " input(s), got " +
                                        std::to_string(inputs.size()));
        }
        forms::InlinedConstitutiveExpansion out;
        out.outputs.push_back(body_(inputs, ctx.dim));
        return out;
    }

    [[nodiscard]] forms::Value<Real> evaluate(const forms::Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw noPointwiseEvaluation();
    }

    [[nodiscard]] forms::Value<forms::Dual> evaluate(const forms::Value<forms::Dual>& /*input*/,
                                                     int /*dim*/,
                                                     forms::DualWorkspace& /*workspace*/) const override
    {
        throw noPointwiseEvaluation();
    }

    [[nodiscard]] OutputSpec outputSpec(std::size_t output_index) const override
    {
        if (output_index != 0u) {
            throw std::invalid_argument("FormLaw::outputSpec: output_index out of range");
        }
        return OutputSpec{signature_.output};
    }

    [[nodiscard]] std::optional<analysis::ConstitutiveLawMetadata>
    constitutiveLawMetadata(std::size_t output_index) const override
    {
        if (output_index != 0u) {
            throw std::invalid_argument("FormLaw::constitutiveLawMetadata: output_index out of range");
        }
        return metadata_;
    }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind() const override
    {
        return expectedInputKind(0u);
    }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind(std::size_t input_index) const override
    {
        if (input_index >= signature_.inputs.size()) {
            return std::nullopt;
        }
        return signature_.inputs[input_index];
    }

    [[nodiscard]] std::optional<std::size_t> expectedInputCount() const override
    {
        return signature_.inputs.size();
    }

private:
    [[nodiscard]] std::logic_error noPointwiseEvaluation() const
    {
        return std::logic_error("FormLaw '" + kind_name_ +
                                "' is written as forms; setup expands its calls, and it has no pointwise evaluation");
    }

    std::string kind_name_{};
    Signature signature_{};
    Body body_{};
    std::optional<analysis::ConstitutiveLawMetadata> metadata_{};
};

} // namespace constitutive

namespace forms {

/// A constitutive law written as forms, called like a function: law(x) is
/// the constitutive node FormExpr::constitutive(model, x) that setup expands.
class Law {
public:
    explicit Law(std::shared_ptr<const constitutive::FormLaw> model)
        : model_(std::move(model))
    {
    }

    [[nodiscard]] FormExpr operator()(const FormExpr& input) const { return FormExpr::constitutive(model_, input); }
    [[nodiscard]] FormExpr operator()(std::vector<FormExpr> inputs) const
    {
        return FormExpr::constitutive(model_, std::move(inputs));
    }

    [[nodiscard]] const std::shared_ptr<const constitutive::FormLaw>& model() const noexcept { return model_; }

private:
    std::shared_ptr<const constitutive::FormLaw> model_{};
};

/**
 * @brief The law of one input @p body written as forms
 *
 *   const auto mu = law("carreau_yasuda", Kind::Scalar, Kind::Scalar,
 *                       [=](const FormExpr& gamma, int) { return mu_inf + ...; });
 *   const auto viscous = 2.0 * mu(shear_rate) * inner(symgrad(u), symgrad(v));
 */
[[nodiscard]] inline Law law(std::string kind_name,
                             Value<Real>::Kind input,
                             Value<Real>::Kind output,
                             std::function<FormExpr(const FormExpr& input, int dim)> body,
                             std::optional<analysis::ConstitutiveLawMetadata> metadata = std::nullopt)
{
    if (!body) {
        throw std::invalid_argument("forms::law: missing body");
    }
    return Law(std::make_shared<const constitutive::FormLaw>(
        std::move(kind_name),
        constitutive::FormLaw::Signature{{input}, output},
        [body = std::move(body)](std::span<const FormExpr> inputs, int dim) { return body(inputs.front(), dim); },
        std::move(metadata)));
}

/// The law of several inputs @p body written as forms.
[[nodiscard]] inline Law law(std::string kind_name,
                             constitutive::FormLaw::Signature signature,
                             constitutive::FormLaw::Body body,
                             std::optional<analysis::ConstitutiveLawMetadata> metadata = std::nullopt)
{
    return Law(std::make_shared<const constitutive::FormLaw>(
        std::move(kind_name), std::move(signature), std::move(body), std::move(metadata)));
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_EXPRESSION_LAW_H
