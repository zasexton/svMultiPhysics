#ifndef SVMP_FE_FORMS_CONSTITUTIVE_MODEL_H
#define SVMP_FE_FORMS_CONSTITUTIVE_MODEL_H

/**
 * @file ConstitutiveModel.h
 * @brief Type-erased material-point operator interface for FE/Forms
 *
 * This interface is the integration boundary between FE/Forms and future
 * FE/Constitutive implementations. A ConstitutiveModel is callable at
 * quadrature points and must support both Real and Dual evaluation so that
 * FE/Forms can assemble consistent Jacobians via AD.
 */

#include "Core/Types.h"
#include "Core/ParameterValue.h"
#include "Forms/Dual.h"
#include "Forms/Value.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace svmp {
namespace FE {

namespace constitutive {
class StateLayout;
}

namespace forms {

class InlinableConstitutiveModel;

struct ConstitutiveEvalContext {
    enum class Domain : std::uint8_t {
        Cell,
        BoundaryFace,
        InteriorFace,
    };

    enum class TraceSide : std::uint8_t {
        Minus,
        Plus,
    };

    struct NonlocalAccess {
        const void* self{nullptr};

        Value<Real> (*input_real)(const void* self, std::size_t input_index, LocalIndex qpt){nullptr};
        Value<Dual> (*input_dual)(const void* self,
                                  std::size_t input_index,
                                  LocalIndex qpt,
                                  DualWorkspace& workspace){nullptr};

        std::span<const std::byte> (*state_old)(const void* self, LocalIndex qpt){nullptr};
        std::span<std::byte> (*state_work)(const void* self, LocalIndex qpt){nullptr};

        std::array<Real, 3> (*physical_point)(const void* self, LocalIndex qpt){nullptr};
        Real (*integration_weight)(const void* self, LocalIndex qpt){nullptr};
    };

    Domain domain{Domain::Cell};
    TraceSide side{TraceSide::Minus};

    int dim{0};
    std::array<Real, 3> x{0.0, 0.0, 0.0};
    Real time{0.0};
    Real dt{0.0};

    GlobalIndex cell_id{-1};
    GlobalIndex face_id{-1};
    LocalIndex local_face_id{0};
    int boundary_marker{-1};

    LocalIndex q{0};
    LocalIndex num_qpts{0};

    std::span<const std::byte> state_old{};
    std::span<std::byte> state_work{};

    const std::function<std::optional<Real>(std::string_view)>* get_real_param{nullptr};
    const std::function<std::optional<params::Value>(std::string_view)>* get_param{nullptr};
    const void* user_data{nullptr};
    const NonlocalAccess* nonlocal{nullptr};

    [[nodiscard]] std::optional<Real> realParam(std::string_view key) const
    {
        if (auto v = param(key)) {
            if (auto r = params::get<Real>(*v)) return *r;
        }
        if (get_real_param == nullptr) return std::nullopt;
        if (!static_cast<bool>(*get_real_param)) return std::nullopt;
        return (*get_real_param)(key);
    }

    [[nodiscard]] std::optional<params::Value> param(std::string_view key) const
    {
        if (get_param == nullptr) return std::nullopt;
        if (!static_cast<bool>(*get_param)) return std::nullopt;
        return (*get_param)(key);
    }

    template <class T>
    [[nodiscard]] std::optional<T> paramAs(std::string_view key) const
    {
        const auto v = param(key);
        if (!v) return std::nullopt;
        return params::get<T>(*v);
    }

    template <class T>
    [[nodiscard]] T requireParamAs(std::string_view key) const
    {
        if constexpr (std::is_same_v<T, Real>) {
            const auto v = realParam(key);
            if (!v.has_value()) {
                throw std::invalid_argument(std::string("ConstitutiveEvalContext: missing required parameter '") +
                                            std::string(key) + "'");
            }
            return *v;
        } else {
            const auto v = paramAs<T>(key);
            if (!v.has_value()) {
                throw std::invalid_argument(std::string("ConstitutiveEvalContext: missing required parameter '") +
                                            std::string(key) + "' (or wrong type)");
            }
            return *v;
        }
    }

    template <class T>
    [[nodiscard]] T paramOr(std::string_view key, T default_value) const
    {
        if constexpr (std::is_same_v<T, Real>) {
            const auto v = realParam(key);
            return v.has_value() ? *v : default_value;
        } else {
            const auto v = paramAs<T>(key);
            return v.has_value() ? *v : default_value;
        }
    }

    [[nodiscard]] bool hasNonlocalAccess() const noexcept { return nonlocal != nullptr; }

    [[nodiscard]] Value<Real> inputAt(std::size_t input_index, LocalIndex qpt) const
    {
        if (nonlocal == nullptr || nonlocal->input_real == nullptr) {
            throw std::invalid_argument("ConstitutiveEvalContext::inputAt: nonlocal input access not available");
        }
        return nonlocal->input_real(nonlocal->self, input_index, qpt);
    }

    [[nodiscard]] Value<Dual> inputAt(std::size_t input_index, LocalIndex qpt, DualWorkspace& workspace) const
    {
        if (nonlocal == nullptr || nonlocal->input_dual == nullptr) {
            throw std::invalid_argument("ConstitutiveEvalContext::inputAt: nonlocal input access not available (dual)");
        }
        return nonlocal->input_dual(nonlocal->self, input_index, qpt, workspace);
    }

    [[nodiscard]] std::span<const std::byte> stateOldAt(LocalIndex qpt) const
    {
        if (nonlocal == nullptr || nonlocal->state_old == nullptr) return {};
        return nonlocal->state_old(nonlocal->self, qpt);
    }

    [[nodiscard]] std::span<std::byte> stateWorkAt(LocalIndex qpt) const
    {
        if (nonlocal == nullptr || nonlocal->state_work == nullptr) return {};
        return nonlocal->state_work(nonlocal->self, qpt);
    }

    [[nodiscard]] std::array<Real, 3> physicalPointAt(LocalIndex qpt) const
    {
        if (nonlocal == nullptr || nonlocal->physical_point == nullptr) {
            throw std::invalid_argument("ConstitutiveEvalContext::physicalPointAt: nonlocal geometry access not available");
        }
        return nonlocal->physical_point(nonlocal->self, qpt);
    }

    [[nodiscard]] Real integrationWeightAt(LocalIndex qpt) const
    {
        if (nonlocal == nullptr || nonlocal->integration_weight == nullptr) {
            throw std::invalid_argument("ConstitutiveEvalContext::integrationWeightAt: nonlocal geometry access not available");
        }
        return nonlocal->integration_weight(nonlocal->self, qpt);
    }
};

class ConstitutiveModel {
public:
    using ValueKind = Value<Real>::Kind;

    struct OutputSpec {
        std::optional<ValueKind> kind{};
    };

    struct StateSpec {
        std::size_t bytes_per_qpt{0};
        std::size_t alignment{alignof(std::max_align_t)};

        [[nodiscard]] bool empty() const noexcept { return bytes_per_qpt == 0; }
    };

    virtual ~ConstitutiveModel() = default;

    /**
     * @brief Optional hook: return inlinable interface for JIT-fast lowering
     *
     * Models that support setup-time symbolic expansion should override this to
     * return a non-null pointer (typically `return this;` if also implementing
     * InlinableConstitutiveModel).
     */
    [[nodiscard]] virtual const InlinableConstitutiveModel* inlinable() const noexcept { return nullptr; }

    [[nodiscard]] virtual Value<Real> evaluate(const Value<Real>& input, int dim) const = 0;

    [[nodiscard]] virtual Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const
    {
        return evaluate(input, ctx.dim);
    }

    [[nodiscard]] virtual Value<Dual> evaluate(const Value<Dual>& input,
                                               int dim,
                                               DualWorkspace& workspace) const = 0;

    [[nodiscard]] virtual Value<Dual> evaluate(const Value<Dual>& input,
                                               const ConstitutiveEvalContext& ctx,
                                               DualWorkspace& workspace) const
    {
        return evaluate(input, ctx.dim, workspace);
    }

    [[nodiscard]] virtual Value<Real> evaluateNary(std::span<const Value<Real>> inputs,
                                                   const ConstitutiveEvalContext& ctx) const
    {
        if (inputs.size() != 1u) {
            throw std::invalid_argument("ConstitutiveModel::evaluateNary: model does not support arity != 1");
        }
        return evaluate(inputs[0], ctx);
    }

    [[nodiscard]] virtual Value<Dual> evaluateNary(std::span<const Value<Dual>> inputs,
                                                   const ConstitutiveEvalContext& ctx,
                                                   DualWorkspace& workspace) const
    {
        if (inputs.size() != 1u) {
            throw std::invalid_argument("ConstitutiveModel::evaluateNary: model does not support arity != 1 (dual)");
        }
        return evaluate(inputs[0], ctx, workspace);
    }

    [[nodiscard]] virtual std::size_t outputCount() const noexcept { return 1u; }

    [[nodiscard]] virtual OutputSpec outputSpec(std::size_t output_index) const
    {
        if (output_index >= outputCount()) {
            throw std::invalid_argument("ConstitutiveModel::outputSpec: output_index out of range");
        }
        return {};
    }

    virtual void evaluateNaryOutputs(std::span<const Value<Real>> inputs,
                                     const ConstitutiveEvalContext& ctx,
                                     std::span<Value<Real>> outputs) const
    {
        const auto n = outputCount();
        if (outputs.size() != n) {
            throw std::invalid_argument("ConstitutiveModel::evaluateNaryOutputs: outputs span size mismatch");
        }
        if (n != 1u) {
            throw std::invalid_argument("ConstitutiveModel::evaluateNaryOutputs: multi-output not implemented");
        }
        outputs[0] = evaluateNary(inputs, ctx);
    }

    virtual void evaluateNaryOutputs(std::span<const Value<Dual>> inputs,
                                     const ConstitutiveEvalContext& ctx,
                                     DualWorkspace& workspace,
                                     std::span<Value<Dual>> outputs) const
    {
        const auto n = outputCount();
        if (outputs.size() != n) {
            throw std::invalid_argument("ConstitutiveModel::evaluateNaryOutputs: outputs span size mismatch (dual)");
        }
        if (n != 1u) {
            throw std::invalid_argument("ConstitutiveModel::evaluateNaryOutputs: multi-output not implemented (dual)");
        }
        outputs[0] = evaluateNary(inputs, ctx, workspace);
    }

    /**
     * @brief Optional metadata: expected input kind (shape) for this model
     *
     * Returning nullopt indicates "not specified" or "multiple kinds supported".
     */
    [[nodiscard]] virtual std::optional<ValueKind> expectedInputKind() const { return std::nullopt; }

    [[nodiscard]] virtual std::optional<ValueKind> expectedInputKind(std::size_t input_index) const
    {
        if (input_index == 0u) return expectedInputKind();
        return std::nullopt;
    }

    [[nodiscard]] virtual std::optional<std::size_t> expectedInputCount() const { return std::nullopt; }

    /**
     * @brief Optional metadata: per-integration-point state requirement
     *
     * Returning bytes_per_qpt=0 indicates the model does not require state (or does not specify it).
     */
    [[nodiscard]] virtual StateSpec stateSpec() const noexcept { return {}; }

    /**
     * @brief Optional structured state layout metadata
     *
     * If provided, this describes the byte-level contract for the model's
     * per-integration-point state block. Forms/Assembly use the byte size and
     * alignment from @ref stateSpec to allocate storage; this hook is intended
     * for tooling/introspection and for models that want to define their state
     * in terms of named fields.
     */
    [[nodiscard]] virtual const constitutive::StateLayout* stateLayout() const noexcept { return nullptr; }

    /**
     * @brief Optional metadata: required/optional parameters
     *
     * This is consumed by higher-level setup code (e.g., FE/Systems) for
     * validation and defaults. Returning an empty list indicates that the model
     * does not declare any parameter requirements.
     */
    [[nodiscard]] virtual std::vector<params::Spec> parameterSpecs() const { return {}; }
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_CONSTITUTIVE_MODEL_H
