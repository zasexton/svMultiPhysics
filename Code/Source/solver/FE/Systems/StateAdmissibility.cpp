/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/StateAdmissibility.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace svmp {
namespace FE {
namespace systems {

StateAdmissibilityTarget StateAdmissibilityTarget::fieldTarget(FieldId field_id) noexcept
{
    StateAdmissibilityTarget target;
    target.kind = StateAdmissibilityTargetKind::Field;
    target.field = field_id;
    return target;
}

StateAdmissibilityTarget StateAdmissibilityTarget::stateGroupTarget(StateGroupId group_id) noexcept
{
    StateAdmissibilityTarget target;
    target.kind = StateAdmissibilityTargetKind::StateGroup;
    target.state_group = group_id;
    return target;
}

namespace {

[[nodiscard]] Real positivePart(Real value) noexcept
{
    return value > 0.0 ? value : 0.0;
}

[[nodiscard]] Real maxFiniteResidual(std::span<const Real> values) noexcept
{
    Real residual = 0.0;
    for (const Real value : values) {
        if (!std::isfinite(value)) {
            residual = 1.0;
        }
    }
    return residual;
}

} // namespace

void StateAdmissibilityEvaluator::validateDescriptor(
    const StateAdmissibilityDescriptor& descriptor)
{
    FE_THROW_IF(descriptor.id.empty(), InvalidArgumentException,
                "StateAdmissibilityEvaluator: descriptor id is empty");
    FE_THROW_IF(descriptor.tolerance < 0.0, InvalidArgumentException,
                "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                    "' has negative tolerance");

    if (descriptor.target.kind == StateAdmissibilityTargetKind::Field) {
        FE_THROW_IF(descriptor.target.field == INVALID_FIELD_ID, InvalidArgumentException,
                    "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                        "' has an invalid field target");
    } else {
        FE_THROW_IF(descriptor.target.state_group == INVALID_STATE_GROUP_ID,
                    InvalidArgumentException,
                    "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                        "' has an invalid state-group target");
    }

    switch (descriptor.check) {
        case StateAdmissibilityCheckKind::LowerBound:
            FE_THROW_IF(!descriptor.lower_bound.has_value(), InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' requires a lower bound");
            break;
        case StateAdmissibilityCheckKind::UpperBound:
            FE_THROW_IF(!descriptor.upper_bound.has_value(), InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' requires an upper bound");
            break;
        case StateAdmissibilityCheckKind::Interval:
            FE_THROW_IF(!descriptor.lower_bound.has_value() ||
                            !descriptor.upper_bound.has_value(),
                        InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' requires lower and upper bounds");
            FE_THROW_IF(*descriptor.lower_bound > *descriptor.upper_bound,
                        InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' has lower bound greater than upper bound");
            break;
        case StateAdmissibilityCheckKind::SumEquality:
            FE_THROW_IF(!descriptor.sum_target.has_value(), InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' requires a sum target");
            break;
        case StateAdmissibilityCheckKind::SumInequality:
            FE_THROW_IF(!descriptor.sum_lower_bound.has_value() &&
                            !descriptor.sum_upper_bound.has_value(),
                        InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' requires a sum lower or upper bound");
            FE_THROW_IF(descriptor.sum_lower_bound.has_value() &&
                            descriptor.sum_upper_bound.has_value() &&
                            *descriptor.sum_lower_bound > *descriptor.sum_upper_bound,
                        InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' has sum lower bound greater than sum upper bound");
            break;
        case StateAdmissibilityCheckKind::Finite:
            break;
        case StateAdmissibilityCheckKind::CallbackResidual:
            FE_THROW_IF(!descriptor.residual_callback, InvalidArgumentException,
                        "StateAdmissibilityEvaluator: descriptor '" + descriptor.id +
                            "' requires a residual callback");
            break;
    }
}

Real StateAdmissibilityEvaluator::sampleResidual(
    const StateAdmissibilityDescriptor& descriptor,
    std::span<const Real> values)
{
    if (values.empty()) {
        return 1.0;
    }

    switch (descriptor.check) {
        case StateAdmissibilityCheckKind::LowerBound: {
            Real residual = 0.0;
            for (const Real value : values) {
                residual = std::max(residual, positivePart(*descriptor.lower_bound - value));
            }
            return residual;
        }
        case StateAdmissibilityCheckKind::UpperBound: {
            Real residual = 0.0;
            for (const Real value : values) {
                residual = std::max(residual, positivePart(value - *descriptor.upper_bound));
            }
            return residual;
        }
        case StateAdmissibilityCheckKind::Interval: {
            Real residual = 0.0;
            for (const Real value : values) {
                residual = std::max(residual, positivePart(*descriptor.lower_bound - value));
                residual = std::max(residual, positivePart(value - *descriptor.upper_bound));
            }
            return residual;
        }
        case StateAdmissibilityCheckKind::SumEquality: {
            const Real sum = std::accumulate(values.begin(), values.end(), Real{0.0});
            return positivePart(std::abs(sum - *descriptor.sum_target) -
                                descriptor.tolerance);
        }
        case StateAdmissibilityCheckKind::SumInequality: {
            const Real sum = std::accumulate(values.begin(), values.end(), Real{0.0});
            Real residual = 0.0;
            if (descriptor.sum_lower_bound.has_value()) {
                residual = std::max(residual,
                                    positivePart(*descriptor.sum_lower_bound - sum -
                                                 descriptor.tolerance));
            }
            if (descriptor.sum_upper_bound.has_value()) {
                residual = std::max(residual,
                                    positivePart(sum - *descriptor.sum_upper_bound -
                                                 descriptor.tolerance));
            }
            return residual;
        }
        case StateAdmissibilityCheckKind::Finite:
            return maxFiniteResidual(values);
        case StateAdmissibilityCheckKind::CallbackResidual:
            return positivePart(descriptor.residual_callback(values) - descriptor.tolerance);
    }
    return 1.0;
}

StateAdmissibilityResult StateAdmissibilityEvaluator::evaluate(
    const StateAdmissibilityDescriptor& descriptor,
    std::span<const StateAdmissibilitySample> samples,
    std::size_t max_recorded_violations)
{
    validateDescriptor(descriptor);

    StateAdmissibilityResult result;
    result.checked_state_count = static_cast<std::uint64_t>(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const Real residual = sampleResidual(descriptor, samples[i].values);
        result.max_residual = std::max(result.max_residual, residual);
        if (residual > 0.0) {
            result.admissible = false;
            ++result.violation_count;
            if (result.violations.size() < max_recorded_violations) {
                result.violations.push_back(
                    StateAdmissibilityViolation{.sample_index = i,
                                                .residual = residual,
                                                .label = samples[i].label});
            }
        }
    }
    return result;
}

analysis::InvariantDomainSummary StateAdmissibilityEvaluator::makeInvariantDomainSummary(
    const StateAdmissibilityDescriptor& descriptor,
    const StateAdmissibilityResult& result,
    const FieldRegistry* registry)
{
    validateDescriptor(descriptor);

    analysis::InvariantDomainSummary summary;
    summary.invariant_set_id = descriptor.id;
    if (descriptor.lower_bound.has_value()) {
        summary.lower_bound = *descriptor.lower_bound;
        summary.lower_bound_active = true;
    }
    if (descriptor.upper_bound.has_value()) {
        summary.upper_bound = *descriptor.upper_bound;
        summary.upper_bound_active = true;
    }
    summary.source_admissibility_evidence_present = true;
    summary.post_step_violation_count = result.violation_count;

    if (descriptor.target.kind == StateAdmissibilityTargetKind::Field) {
        summary.variables.push_back(analysis::VariableKey::field(descriptor.target.field));
    } else if (registry != nullptr && registry->hasStateGroup(descriptor.target.state_group)) {
        const auto& group = registry->getStateGroup(descriptor.target.state_group);
        for (const FieldId field_id : group.fields) {
            summary.variables.push_back(analysis::VariableKey::field(field_id));
        }
    }

    return summary;
}

} // namespace systems
} // namespace FE
} // namespace svmp
