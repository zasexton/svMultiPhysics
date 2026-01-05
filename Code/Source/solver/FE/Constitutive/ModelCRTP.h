#ifndef SVMP_FE_CONSTITUTIVE_MODEL_CRTP_H
#define SVMP_FE_CONSTITUTIVE_MODEL_CRTP_H

#include "Constitutive/StateLayout.h"
#include "Constitutive/ValueChecks.h"

#include "Core/FEException.h"
#include "Core/Types.h"
#include "Forms/ConstitutiveModel.h"

#include <optional>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace constitutive {

struct NoWorkspace {
    void reset(std::size_t /*num_dofs*/) {}
};

template <class Derived>
class ModelCRTP : public forms::ConstitutiveModel {
public:
    using forms::ConstitutiveModel::evaluate;
    using forms::ConstitutiveModel::expectedInputKind;
    using ValueKind = forms::ConstitutiveModel::ValueKind;
    using StateSpec = forms::ConstitutiveModel::StateSpec;

    [[nodiscard]] forms::Value<Real> evaluate(const forms::Value<Real>& input, int dim) const override
    {
        NoWorkspace ws;
        if constexpr (HasEvaluateImplDim<Derived, Real, NoWorkspace>::value) {
            return derived().template evaluateImpl<Real>(input, dim, ws);
        } else if constexpr (HasEvaluateImplCtx<Derived, Real, NoWorkspace>::value) {
            forms::ConstitutiveEvalContext ctx;
            ctx.dim = dim;
            return derived().template evaluateImpl<Real>(input, ctx, ws);
        } else {
            static_assert(HasEvaluateImplDim<Derived, Real, NoWorkspace>::value ||
                              HasEvaluateImplCtx<Derived, Real, NoWorkspace>::value,
                          "ModelCRTP<Derived>: Derived must implement either "
                          "evaluateImpl<Real>(Value<Real>, dim, NoWorkspace) -> Value<Real> "
                          "or evaluateImpl<Real>(Value<Real>, ctx, NoWorkspace) -> Value<Real>");
        }
    }

    [[nodiscard]] forms::Value<Real> evaluate(const forms::Value<Real>& input,
                                              const forms::ConstitutiveEvalContext& ctx) const override
    {
        NoWorkspace ws;
        if constexpr (HasEvaluateImplCtx<Derived, Real, NoWorkspace>::value) {
            return derived().template evaluateImpl<Real>(input, ctx, ws);
        } else if constexpr (HasEvaluateImplDim<Derived, Real, NoWorkspace>::value) {
            return derived().template evaluateImpl<Real>(input, ctx.dim, ws);
        } else {
            static_assert(HasEvaluateImplDim<Derived, Real, NoWorkspace>::value ||
                              HasEvaluateImplCtx<Derived, Real, NoWorkspace>::value,
                          "ModelCRTP<Derived>: Derived must implement either "
                          "evaluateImpl<Real>(Value<Real>, dim, NoWorkspace) -> Value<Real> "
                          "or evaluateImpl<Real>(Value<Real>, ctx, NoWorkspace) -> Value<Real>");
        }
    }

    [[nodiscard]] forms::Value<forms::Dual> evaluate(const forms::Value<forms::Dual>& input,
                                                     int dim,
                                                     forms::DualWorkspace& workspace) const override
    {
        if constexpr (HasEvaluateImplDim<Derived, forms::Dual, forms::DualWorkspace>::value) {
            return derived().template evaluateImpl<forms::Dual>(input, dim, workspace);
        } else if constexpr (HasEvaluateImplCtx<Derived, forms::Dual, forms::DualWorkspace>::value) {
            forms::ConstitutiveEvalContext ctx;
            ctx.dim = dim;
            return derived().template evaluateImpl<forms::Dual>(input, ctx, workspace);
        } else {
            static_assert(HasEvaluateImplDim<Derived, forms::Dual, forms::DualWorkspace>::value ||
                              HasEvaluateImplCtx<Derived, forms::Dual, forms::DualWorkspace>::value,
                          "ModelCRTP<Derived>: Derived must implement either "
                          "evaluateImpl<Dual>(Value<Dual>, dim, DualWorkspace) -> Value<Dual> "
                          "or evaluateImpl<Dual>(Value<Dual>, ctx, DualWorkspace) -> Value<Dual>");
        }
    }

    [[nodiscard]] forms::Value<forms::Dual> evaluate(const forms::Value<forms::Dual>& input,
                                                     const forms::ConstitutiveEvalContext& ctx,
                                                     forms::DualWorkspace& workspace) const override
    {
        if constexpr (HasEvaluateImplCtx<Derived, forms::Dual, forms::DualWorkspace>::value) {
            return derived().template evaluateImpl<forms::Dual>(input, ctx, workspace);
        } else if constexpr (HasEvaluateImplDim<Derived, forms::Dual, forms::DualWorkspace>::value) {
            return derived().template evaluateImpl<forms::Dual>(input, ctx.dim, workspace);
        } else {
            static_assert(HasEvaluateImplDim<Derived, forms::Dual, forms::DualWorkspace>::value ||
                              HasEvaluateImplCtx<Derived, forms::Dual, forms::DualWorkspace>::value,
                          "ModelCRTP<Derived>: Derived must implement either "
                          "evaluateImpl<Dual>(Value<Dual>, dim, DualWorkspace) -> Value<Dual> "
                          "or evaluateImpl<Dual>(Value<Dual>, ctx, DualWorkspace) -> Value<Dual>");
        }
    }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind() const override
    {
        if constexpr (HasStaticExpectedInputKind<Derived>::value) {
            return Derived::kExpectedInputKind;
        }
        return forms::ConstitutiveModel::expectedInputKind();
    }

    [[nodiscard]] std::optional<std::size_t> expectedInputCount() const override
    {
        if constexpr (HasStaticExpectedInputCount<Derived>::value) {
            return static_cast<std::size_t>(Derived::kExpectedInputCount);
        }
        return forms::ConstitutiveModel::expectedInputCount();
    }

    [[nodiscard]] StateSpec stateSpec() const noexcept override
    {
        if constexpr (HasStaticStateSpec<Derived>::value) {
            return Derived::kStateSpec;
        }
        if constexpr (HasStaticStateLayout<Derived>::value) {
            if (Derived::kStateLayout.empty()) return {};
            return StateSpec{Derived::kStateLayout.bytesPerPoint(), Derived::kStateLayout.alignment()};
        }
        return forms::ConstitutiveModel::stateSpec();
    }

    [[nodiscard]] const StateLayout* stateLayout() const noexcept override
    {
        if constexpr (HasStaticStateLayout<Derived>::value) {
            if (Derived::kStateLayout.empty()) return nullptr;
            return &Derived::kStateLayout;
        }
        return forms::ConstitutiveModel::stateLayout();
    }

private:
    template <class T, class Scalar, class Workspace, class = void>
    struct HasEvaluateImplDim : std::false_type {};
    template <class T, class Scalar, class Workspace>
    struct HasEvaluateImplDim<
        T,
        Scalar,
        Workspace,
        std::void_t<decltype(std::declval<const T&>().template evaluateImpl<Scalar>(
            std::declval<const forms::Value<Scalar>&>(),
            0,
            std::declval<Workspace&>()))>> {
        static constexpr bool value =
            std::is_same_v<decltype(std::declval<const T&>().template evaluateImpl<Scalar>(
                              std::declval<const forms::Value<Scalar>&>(),
                              0,
                              std::declval<Workspace&>())),
                          forms::Value<Scalar>>;
    };

    template <class T, class Scalar, class Workspace, class = void>
    struct HasEvaluateImplCtx : std::false_type {};
    template <class T, class Scalar, class Workspace>
    struct HasEvaluateImplCtx<
        T,
        Scalar,
        Workspace,
        std::void_t<decltype(std::declval<const T&>().template evaluateImpl<Scalar>(
            std::declval<const forms::Value<Scalar>&>(),
            std::declval<const forms::ConstitutiveEvalContext&>(),
            std::declval<Workspace&>()))>> {
        static constexpr bool value =
            std::is_same_v<decltype(std::declval<const T&>().template evaluateImpl<Scalar>(
                              std::declval<const forms::Value<Scalar>&>(),
                              std::declval<const forms::ConstitutiveEvalContext&>(),
                              std::declval<Workspace&>())),
                          forms::Value<Scalar>>;
    };

    template <class T, class = void>
    struct HasStaticExpectedInputKind : std::false_type {};
    template <class T>
    struct HasStaticExpectedInputKind<T, std::void_t<decltype(T::kExpectedInputKind)>> {
        static constexpr bool value = std::is_convertible_v<decltype(T::kExpectedInputKind), ValueKind>;
    };

    template <class T, class = void>
    struct HasStaticExpectedInputCount : std::false_type {};
    template <class T>
    struct HasStaticExpectedInputCount<T, std::void_t<decltype(T::kExpectedInputCount)>> {
        static constexpr bool value = std::is_convertible_v<decltype(T::kExpectedInputCount), std::size_t>;
    };

    template <class T, class = void>
    struct HasStaticStateSpec : std::false_type {};
    template <class T>
    struct HasStaticStateSpec<T, std::void_t<decltype(T::kStateSpec)>> {
        static constexpr bool value = std::is_convertible_v<decltype(T::kStateSpec), StateSpec>;
    };

    template <class T, class = void>
    struct HasStaticStateLayout : std::false_type {};
    template <class T>
    struct HasStaticStateLayout<T, std::void_t<decltype(T::kStateLayout)>> {
        static constexpr bool value =
            std::is_convertible_v<decltype(T::kStateLayout), const StateLayout&> ||
            std::is_convertible_v<decltype(T::kStateLayout), StateLayout>;
    };

    [[nodiscard]] const Derived& derived() const
    {
        static_assert(std::is_base_of_v<ModelCRTP<Derived>, Derived>,
                      "ModelCRTP<Derived>: Derived must inherit from ModelCRTP<Derived>");
        return static_cast<const Derived&>(*this);
    }
};

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_MODEL_CRTP_H
