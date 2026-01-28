#ifndef SVMP_FE_CONSTITUTIVE_LAW_ADAPTERS_H
#define SVMP_FE_CONSTITUTIVE_LAW_ADAPTERS_H

#include "Constitutive/ModelCRTP.h"
#include "Constitutive/ValueChecks.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constitutive {

template <class Derived>
class ScalarLawCRTP : public ModelCRTP<Derived> {
public:
    using ValueKind = typename forms::ConstitutiveModel::ValueKind;

    static constexpr ValueKind kExpectedInputKind = forms::Value<Real>::Kind::Scalar;
    static constexpr std::size_t kExpectedInputCount = 1u;

    [[nodiscard]] forms::ConstitutiveModel::OutputSpec outputSpec(std::size_t output_index) const override
    {
        if (output_index != 0u) {
            throw std::invalid_argument("ScalarLawCRTP::outputSpec: output_index out of range");
        }
        return forms::ConstitutiveModel::OutputSpec{ValueKind::Scalar};
    }

    template <class Scalar, class Workspace>
    [[nodiscard]] forms::Value<Scalar> evaluateImpl(const forms::Value<Scalar>& input,
                                                    const forms::ConstitutiveEvalContext& ctx,
                                                    Workspace& ws) const
    {
        requireValueKind(input, forms::Value<Scalar>::Kind::Scalar, "ScalarLawCRTP");
        return forms::Value<Scalar>{forms::Value<Scalar>::Kind::Scalar,
                                    derived().template evalScalar<Scalar>(input.s, ctx, ws)};
    }

private:
    template <class T, class Scalar, class Workspace, class = void>
    struct HasEvalScalar : std::false_type {};
    template <class T, class Scalar, class Workspace>
    struct HasEvalScalar<
        T,
        Scalar,
        Workspace,
        std::void_t<decltype(std::declval<const T&>().template evalScalar<Scalar>(
            std::declval<const Scalar&>(),
            std::declval<const forms::ConstitutiveEvalContext&>(),
            std::declval<Workspace&>()))>> : std::true_type {};

    [[nodiscard]] const Derived& derived() const
    {
        static_assert(std::is_base_of_v<ScalarLawCRTP<Derived>, Derived>,
                      "ScalarLawCRTP<Derived>: Derived must inherit from ScalarLawCRTP<Derived>");
        static_assert(HasEvalScalar<Derived, Real, NoWorkspace>::value,
                      "ScalarLawCRTP<Derived>: Derived must implement "
                      "evalScalar<Real>(Real, ctx, NoWorkspace) -> Real (e.g., via a template)");
        static_assert(HasEvalScalar<Derived, forms::Dual, forms::DualWorkspace>::value,
                      "ScalarLawCRTP<Derived>: Derived must implement "
                      "evalScalar<Dual>(Dual, ctx, DualWorkspace) -> Dual (e.g., via a template)");
        return static_cast<const Derived&>(*this);
    }
};

template <class Scalar>
class MatrixConstRef {
public:
    explicit MatrixConstRef(const forms::Value<Scalar>& v) : v_(&v) {}

    [[nodiscard]] typename forms::Value<Scalar>::Kind kind() const noexcept { return v_->kind; }
    [[nodiscard]] std::size_t rows() const noexcept { return v_->matrixRows(); }
    [[nodiscard]] std::size_t cols() const noexcept { return v_->matrixCols(); }

    [[nodiscard]] const Scalar& operator()(std::size_t r, std::size_t c) const { return v_->matrixAt(r, c); }

private:
    const forms::Value<Scalar>* v_{nullptr};
};

template <class Scalar>
class MatrixRef {
public:
    explicit MatrixRef(forms::Value<Scalar>& v) : v_(&v) {}

    [[nodiscard]] typename forms::Value<Scalar>::Kind kind() const noexcept { return v_->kind; }
    [[nodiscard]] std::size_t rows() const noexcept { return v_->matrixRows(); }
    [[nodiscard]] std::size_t cols() const noexcept { return v_->matrixCols(); }

    [[nodiscard]] Scalar& operator()(std::size_t r, std::size_t c) { return v_->matrixAt(r, c); }
    [[nodiscard]] const Scalar& operator()(std::size_t r, std::size_t c) const { return v_->matrixAt(r, c); }

private:
    forms::Value<Scalar>* v_{nullptr};
};

template <class Derived>
class MatrixLawCRTP : public ModelCRTP<Derived> {
public:
    using forms::ConstitutiveModel::expectedInputKind;

    static constexpr std::size_t kExpectedInputCount = 1u;

    [[nodiscard]] std::optional<typename forms::ConstitutiveModel::ValueKind> expectedInputKind() const override
    {
        // Matrix-like laws may accept Matrix / SymmetricMatrix / SkewMatrix. Leave unspecified by default.
        return std::nullopt;
    }

    template <class Scalar, class Workspace>
    [[nodiscard]] forms::Value<Scalar> evaluateImpl(const forms::Value<Scalar>& input,
                                                    const forms::ConstitutiveEvalContext& ctx,
                                                    Workspace& ws) const
    {
        using Kind = typename forms::Value<Scalar>::Kind;
        const bool is_matrix = (input.kind == Kind::Matrix) ||
                               (input.kind == Kind::SymmetricMatrix) ||
                               (input.kind == Kind::SkewMatrix);
        FE_THROW_IF(!is_matrix, InvalidArgumentException,
                    "MatrixLawCRTP: expected matrix-like Value input");

        forms::Value<Scalar> out;
        out.kind = input.kind;
        out.resizeMatrix(input.matrixRows(), input.matrixCols());

        derived().template evalMatrix<Scalar>(MatrixConstRef<Scalar>(input), ctx, ws, MatrixRef<Scalar>(out));
        return out;
    }

private:
    template <class T, class Scalar, class Workspace, class = void>
    struct HasEvalMatrix : std::false_type {};
    template <class T, class Scalar, class Workspace>
    struct HasEvalMatrix<
        T,
        Scalar,
        Workspace,
        std::void_t<decltype(std::declval<const T&>().template evalMatrix<Scalar>(
            std::declval<const MatrixConstRef<Scalar>&>(),
            std::declval<const forms::ConstitutiveEvalContext&>(),
            std::declval<Workspace&>(),
            std::declval<MatrixRef<Scalar>>()))>> : std::true_type {};

    [[nodiscard]] const Derived& derived() const
    {
        static_assert(std::is_base_of_v<MatrixLawCRTP<Derived>, Derived>,
                      "MatrixLawCRTP<Derived>: Derived must inherit from MatrixLawCRTP<Derived>");
        static_assert(HasEvalMatrix<Derived, Real, NoWorkspace>::value,
                      "MatrixLawCRTP<Derived>: Derived must implement "
                      "evalMatrix<Real>(MatrixConstRef<Real>, ctx, NoWorkspace, MatrixRef<Real>)");
        static_assert(HasEvalMatrix<Derived, forms::Dual, forms::DualWorkspace>::value,
                      "MatrixLawCRTP<Derived>: Derived must implement "
                      "evalMatrix<Dual>(MatrixConstRef<Dual>, ctx, DualWorkspace, MatrixRef<Dual>)");
        return static_cast<const Derived&>(*this);
    }
};

// ---------------------------------------------------------------------------
// Lambda-backed law factories
// ---------------------------------------------------------------------------

template <class Func>
class LambdaScalarLaw final : public ScalarLawCRTP<LambdaScalarLaw<Func>> {
public:
    explicit LambdaScalarLaw(std::string name, Func func, std::vector<params::Spec> specs = {})
        : name_(std::move(name))
        , func_(std::move(func))
        , specs_(std::move(specs))
    {
    }

    template <class Scalar, class Workspace>
    [[nodiscard]] Scalar evalScalar(const Scalar& x, const forms::ConstitutiveEvalContext& ctx, Workspace& ws) const
    {
        return func_(x, ctx, ws);
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override { return specs_; }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    std::string name_{};
    Func func_{};
    std::vector<params::Spec> specs_{};
};

template <class Func>
[[nodiscard]] inline std::shared_ptr<const forms::ConstitutiveModel> makeScalarLaw(std::string name,
                                                                                   Func func,
                                                                                   std::vector<params::Spec> specs = {})
{
    return std::make_shared<LambdaScalarLaw<Func>>(std::move(name), std::move(func), std::move(specs));
}

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_LAW_ADAPTERS_H
