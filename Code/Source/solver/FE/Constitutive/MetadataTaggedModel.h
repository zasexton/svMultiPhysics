#ifndef SVMP_FE_CONSTITUTIVE_METADATA_TAGGED_MODEL_H
#define SVMP_FE_CONSTITUTIVE_METADATA_TAGGED_MODEL_H

/**
 * @file MetadataTaggedModel.h
 * @brief Lightweight decorator for attaching semantic metadata to constitutive outputs.
 */

#include "Analysis/ConstitutiveLawMetadata.h"
#include "Forms/ConstitutiveModel.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constitutive {

struct ConstitutiveOutputMetadataBinding {
    std::size_t output_index{0u};
    analysis::ConstitutiveLawMetadata metadata{};
};

class MetadataTaggedConstitutiveModel final : public forms::ConstitutiveModel {
public:
    MetadataTaggedConstitutiveModel(
        std::shared_ptr<const forms::ConstitutiveModel> model,
        std::vector<ConstitutiveOutputMetadataBinding> metadata)
        : model_(std::move(model))
        , metadata_(std::move(metadata))
    {
        if (!model_) {
            throw std::invalid_argument("MetadataTaggedConstitutiveModel: null model");
        }
        for (const auto& binding : metadata_) {
            if (binding.output_index >= model_->outputCount()) {
                throw std::invalid_argument(
                    "MetadataTaggedConstitutiveModel: output_index out of range");
            }
        }
    }

    [[nodiscard]] const forms::InlinableConstitutiveModel* inlinable() const noexcept override
    {
        return model_->inlinable();
    }

    [[nodiscard]] forms::Value<Real> evaluate(const forms::Value<Real>& input,
                                              int dim) const override
    {
        return model_->evaluate(input, dim);
    }

    [[nodiscard]] forms::Value<Real> evaluate(
        const forms::Value<Real>& input,
        const forms::ConstitutiveEvalContext& ctx) const override
    {
        return model_->evaluate(input, ctx);
    }

    [[nodiscard]] forms::Value<forms::Dual> evaluate(
        const forms::Value<forms::Dual>& input,
        int dim,
        forms::DualWorkspace& workspace) const override
    {
        return model_->evaluate(input, dim, workspace);
    }

    [[nodiscard]] forms::Value<forms::Dual> evaluate(
        const forms::Value<forms::Dual>& input,
        const forms::ConstitutiveEvalContext& ctx,
        forms::DualWorkspace& workspace) const override
    {
        return model_->evaluate(input, ctx, workspace);
    }

    [[nodiscard]] forms::Value<Real> evaluateNary(
        std::span<const forms::Value<Real>> inputs,
        const forms::ConstitutiveEvalContext& ctx) const override
    {
        return model_->evaluateNary(inputs, ctx);
    }

    [[nodiscard]] forms::Value<forms::Dual> evaluateNary(
        std::span<const forms::Value<forms::Dual>> inputs,
        const forms::ConstitutiveEvalContext& ctx,
        forms::DualWorkspace& workspace) const override
    {
        return model_->evaluateNary(inputs, ctx, workspace);
    }

    [[nodiscard]] std::size_t outputCount() const noexcept override
    {
        return model_->outputCount();
    }

    [[nodiscard]] OutputSpec outputSpec(std::size_t output_index) const override
    {
        return model_->outputSpec(output_index);
    }

    [[nodiscard]] std::optional<analysis::ConstitutiveLawMetadata>
    constitutiveLawMetadata(std::size_t output_index) const override
    {
        if (output_index >= model_->outputCount()) {
            throw std::invalid_argument(
                "MetadataTaggedConstitutiveModel::constitutiveLawMetadata: output_index out of range");
        }

        std::optional<analysis::ConstitutiveLawMetadata> tagged;
        for (const auto& binding : metadata_) {
            if (binding.output_index == output_index) {
                tagged = binding.metadata;
                break;
            }
        }

        const auto base = model_->constitutiveLawMetadata(output_index);
        if (!tagged) {
            return base;
        }
        if (!base) {
            return tagged;
        }

        auto merged = std::move(*tagged);
        if (merged.name.empty()) merged.name = base->name;
        if (merged.role == analysis::ConstitutiveLawRole::Unknown) {
            merged.role = base->role;
        }
        if (merged.input_measure ==
            analysis::ConstitutiveLawInputMeasure::Unspecified) {
            merged.input_measure = base->input_measure;
        }
        if (merged.primary_field == INVALID_FIELD_ID) {
            merged.primary_field = base->primary_field;
        }
        if (!merged.constant_value_available &&
            base->constant_value_available) {
            merged.constant_value_available = true;
            merged.constant_value = base->constant_value;
        }
        if (!merged.model) {
            merged.model = base->model;
        }
        if (merged.source_operator_tag.empty()) {
            merged.source_operator_tag = base->source_operator_tag;
        }
        return merged;
    }

    void evaluateNaryOutputs(std::span<const forms::Value<Real>> inputs,
                             const forms::ConstitutiveEvalContext& ctx,
                             std::span<forms::Value<Real>> outputs) const override
    {
        model_->evaluateNaryOutputs(inputs, ctx, outputs);
    }

    void evaluateNaryOutputs(std::span<const forms::Value<forms::Dual>> inputs,
                             const forms::ConstitutiveEvalContext& ctx,
                             forms::DualWorkspace& workspace,
                             std::span<forms::Value<forms::Dual>> outputs) const override
    {
        model_->evaluateNaryOutputs(inputs, ctx, workspace, outputs);
    }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind() const override
    {
        return model_->expectedInputKind();
    }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind(std::size_t input_index) const override
    {
        return model_->expectedInputKind(input_index);
    }

    [[nodiscard]] std::optional<std::size_t> expectedInputCount() const override
    {
        return model_->expectedInputCount();
    }

    [[nodiscard]] StateSpec stateSpec() const noexcept override
    {
        return model_->stateSpec();
    }

    [[nodiscard]] std::vector<state::StateVariableMetadata> stateVariables() const override
    {
        return model_->stateVariables();
    }

    [[nodiscard]] state::StateFrameTransformHook stateFrameTransformHook() const override
    {
        return model_->stateFrameTransformHook();
    }

    [[nodiscard]] const StateLayout* stateLayout() const noexcept override
    {
        return model_->stateLayout();
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        return model_->parameterSpecs();
    }

private:
    std::shared_ptr<const forms::ConstitutiveModel> model_{};
    std::vector<ConstitutiveOutputMetadataBinding> metadata_{};
};

[[nodiscard]] inline std::shared_ptr<const forms::ConstitutiveModel>
withConstitutiveLawMetadata(
    std::shared_ptr<const forms::ConstitutiveModel> model,
    std::size_t output_index,
    analysis::ConstitutiveLawMetadata metadata)
{
    std::vector<ConstitutiveOutputMetadataBinding> bindings;
    bindings.push_back(ConstitutiveOutputMetadataBinding{
        .output_index = output_index,
        .metadata = std::move(metadata),
    });
    return std::make_shared<MetadataTaggedConstitutiveModel>(
        std::move(model),
        std::move(bindings));
}

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_METADATA_TAGGED_MODEL_H
