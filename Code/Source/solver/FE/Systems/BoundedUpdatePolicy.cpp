/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/BoundedUpdatePolicy.h"

#include "Core/FEException.h"

#include <algorithm>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] bool descriptorConfigured(const StateAdmissibilityDescriptor& descriptor) noexcept
{
    return !descriptor.id.empty();
}

[[nodiscard]] std::string decisionMessage(BoundedUpdateDecision decision)
{
    switch (decision) {
        case BoundedUpdateDecision::Accepted: return "accepted";
        case BoundedUpdateDecision::AcceptedWithModification:
            return "accepted with modification";
        case BoundedUpdateDecision::Rejected: return "rejected";
    }
    return "unknown";
}

} // namespace

bool BoundedUpdateController::clampSamples(
    const StateAdmissibilityDescriptor& descriptor,
    std::vector<StateAdmissibilitySample>& samples)
{
    FE_THROW_IF(!descriptor.lower_bound.has_value() && !descriptor.upper_bound.has_value(),
                InvalidArgumentException,
                "BoundedUpdateController: clamp policy requires at least one bound");

    bool modified = false;
    for (auto& sample : samples) {
        for (Real& value : sample.values) {
            const Real original = value;
            if (descriptor.lower_bound.has_value()) {
                value = std::max(value, *descriptor.lower_bound);
            }
            if (descriptor.upper_bound.has_value()) {
                value = std::min(value, *descriptor.upper_bound);
            }
            modified = modified || value != original;
        }
    }
    return modified;
}

BoundedUpdateResult BoundedUpdateController::apply(
    BoundedUpdatePolicy policy,
    std::vector<StateAdmissibilitySample>& samples)
{
    if (policy.admissibility_residual_callback) {
        policy.admissibility.check = StateAdmissibilityCheckKind::CallbackResidual;
        policy.admissibility.residual_callback = std::move(policy.admissibility_residual_callback);
    }

    BoundedUpdateResult result;

    if (policy.pre_update_callback) {
        policy.pre_update_callback(samples);
    }

    if (policy.candidate_state_filter &&
        !policy.candidate_state_filter(std::span<const StateAdmissibilitySample>(
            samples.data(), samples.size()))) {
        result.decision = BoundedUpdateDecision::Rejected;
        result.message = "candidate state filter rejected update";
        if (descriptorConfigured(policy.admissibility)) {
            result.before = StateAdmissibilityEvaluator::evaluate(policy.admissibility, samples);
        }
        result.after = result.before;
        return result;
    }

    if (policy.kind == BoundedUpdatePolicyKind::None) {
        if (descriptorConfigured(policy.admissibility)) {
            result.before = StateAdmissibilityEvaluator::evaluate(policy.admissibility, samples);
        }
        result.after = result.before;
        result.decision = BoundedUpdateDecision::Accepted;
        result.message = decisionMessage(result.decision);
        if (policy.post_update_callback) {
            policy.post_update_callback(samples);
        }
        return result;
    }

    result.before = StateAdmissibilityEvaluator::evaluate(policy.admissibility, samples);

    switch (policy.kind) {
        case BoundedUpdatePolicyKind::None:
            break;
        case BoundedUpdatePolicyKind::CheckOnly:
            result.decision = BoundedUpdateDecision::Accepted;
            result.after = result.before;
            break;
        case BoundedUpdatePolicyKind::RejectStepOnViolation:
            result.decision = result.before.admissible
                                  ? BoundedUpdateDecision::Accepted
                                  : BoundedUpdateDecision::Rejected;
            result.after = result.before;
            break;
        case BoundedUpdatePolicyKind::ClampCellAverage:
            result.modified = clampSamples(policy.admissibility, samples);
            result.after = StateAdmissibilityEvaluator::evaluate(policy.admissibility, samples);
            result.decision = result.modified
                                  ? BoundedUpdateDecision::AcceptedWithModification
                                  : BoundedUpdateDecision::Accepted;
            if (!result.after.admissible &&
                policy.admissibility.failure_policy ==
                    StateAdmissibilityFailurePolicy::HardFailure) {
                result.decision = BoundedUpdateDecision::Rejected;
            }
            break;
    }

    if (policy.post_update_callback) {
        policy.post_update_callback(samples);
    }
    result.message = decisionMessage(result.decision);
    return result;
}

BoundedUpdateResult BoundedUpdateController::applyToSpan(
    BoundedUpdatePolicy policy,
    std::span<Real> values,
    std::string label)
{
    StateAdmissibilitySample sample;
    sample.values.assign(values.begin(), values.end());
    sample.label = std::move(label);

    std::vector<StateAdmissibilitySample> samples{std::move(sample)};
    auto result = apply(std::move(policy), samples);
    if (!samples.empty() && samples.front().values.size() == values.size()) {
        std::copy(samples.front().values.begin(), samples.front().values.end(), values.begin());
    }
    return result;
}

} // namespace systems
} // namespace FE
} // namespace svmp
