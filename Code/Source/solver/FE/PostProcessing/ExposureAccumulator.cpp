/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "PostProcessing/ExposureAccumulator.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace post {

namespace {

[[nodiscard]] bool finiteNonnegative(Real value) noexcept
{
    return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool aboveThreshold(Real value, Real threshold, bool inclusive) noexcept
{
    return inclusive ? value >= threshold : value > threshold;
}

[[nodiscard]] bool belowThreshold(Real value, Real threshold, bool inclusive) noexcept
{
    return inclusive ? value <= threshold : value < threshold;
}

[[nodiscard]] Real positiveExcess(const ExposureAccumulatorDefinition& definition,
                                  Real value) noexcept
{
    switch (definition.excess_direction) {
        case ExposureExcessDirection::AboveThreshold:
            return std::max(Real{0.0}, value - definition.threshold);
        case ExposureExcessDirection::BelowThreshold:
            return std::max(Real{0.0}, definition.threshold - value);
    }
    return 0.0;
}

[[nodiscard]] Real entityIncrement(const ExposureAccumulatorDefinition& definition,
                                   Real value,
                                   Real dt) noexcept
{
    switch (definition.kind) {
        case ExposureAccumulationKind::TimeAboveThreshold:
            return aboveThreshold(value, definition.threshold, definition.inclusive) ? dt : 0.0;
        case ExposureAccumulationKind::TimeBelowThreshold:
            return belowThreshold(value, definition.threshold, definition.inclusive) ? dt : 0.0;
        case ExposureAccumulationKind::PositiveExcessIntegral:
            return positiveExcess(definition, value) * dt;
        case ExposureAccumulationKind::SquaredPositiveExcessIntegral: {
            const Real excess = positiveExcess(definition, value);
            return excess * excess * dt;
        }
        case ExposureAccumulationKind::RunningMaximum:
        case ExposureAccumulationKind::RunningMinimum:
            return 0.0;
    }
    return 0.0;
}

void updateRunningExtreme(ExposureAccumulationKind kind, Real value, Real& state_value)
{
    switch (kind) {
        case ExposureAccumulationKind::RunningMaximum:
            state_value = std::max(state_value, value);
            break;
        case ExposureAccumulationKind::RunningMinimum:
            state_value = std::min(state_value, value);
            break;
        default:
            break;
    }
}

} // namespace

Real ExposureAccumulator::initialValue(ExposureAccumulationKind kind) noexcept
{
    switch (kind) {
        case ExposureAccumulationKind::RunningMaximum:
            return -std::numeric_limits<Real>::infinity();
        case ExposureAccumulationKind::RunningMinimum:
            return std::numeric_limits<Real>::infinity();
        case ExposureAccumulationKind::TimeAboveThreshold:
        case ExposureAccumulationKind::TimeBelowThreshold:
        case ExposureAccumulationKind::PositiveExcessIntegral:
        case ExposureAccumulationKind::SquaredPositiveExcessIntegral:
            return 0.0;
    }
    return 0.0;
}

std::size_t ExposureAccumulator::requiredValueCount(
    const ExposureAccumulatorDefinition& definition,
    std::span<const ReductionSample> samples)
{
    if (definition.storage_scope == ExposureStorageScope::RegionAggregate) {
        return 1u;
    }
    return static_cast<std::size_t>(std::count_if(
        samples.begin(), samples.end(),
        [&definition](const ReductionSample& sample) {
            return ReductionEvaluator::sampleInRegion(definition.region, sample);
        }));
}

void ExposureAccumulator::reset(const ExposureAccumulatorDefinition& definition,
                                std::span<const ReductionSample> samples,
                                ExposureAccumulatorState& state)
{
    state.scope = definition.storage_scope;
    state.values.assign(requiredValueCount(definition, samples),
                        initialValue(definition.kind));
    state.elapsed_time = 0.0;
    state.step_count = 0;
    state.initialized = true;
}

void ExposureAccumulator::resetSpan(const ExposureAccumulatorDefinition& definition,
                                    std::span<Real> values)
{
    std::fill(values.begin(), values.end(), initialValue(definition.kind));
}

ExposureAccumulatorAdvanceResult ExposureAccumulator::advance(
    const ExposureAccumulatorDefinition& definition,
    std::span<const ReductionSample> samples,
    Real dt,
    ExposureAccumulatorState& state)
{
    FE_THROW_IF(!finiteNonnegative(dt), InvalidArgumentException,
                "ExposureAccumulator::advance: dt must be finite and nonnegative");
    const std::size_t required_count = requiredValueCount(definition, samples);
    if (!state.initialized) {
        reset(definition, samples, state);
    }
    FE_THROW_IF(state.scope != definition.storage_scope, InvalidArgumentException,
                "ExposureAccumulator::advance: state scope does not match definition");
    FE_THROW_IF(state.values.size() != required_count, InvalidArgumentException,
                "ExposureAccumulator::advance: state value count does not match samples");

    auto result = advanceIntoSpan(definition, samples, dt, state.values);
    state.elapsed_time += dt;
    ++state.step_count;
    return result;
}

ExposureAccumulatorAdvanceResult ExposureAccumulator::advanceIntoSpan(
    const ExposureAccumulatorDefinition& definition,
    std::span<const ReductionSample> samples,
    Real dt,
    std::span<Real> values)
{
    FE_THROW_IF(!finiteNonnegative(dt), InvalidArgumentException,
                "ExposureAccumulator::advanceIntoSpan: dt must be finite and nonnegative");
    FE_THROW_IF(values.size() != requiredValueCount(definition, samples),
                InvalidArgumentException,
                "ExposureAccumulator::advanceIntoSpan: value storage size mismatch");

    ExposureAccumulatorAdvanceResult result;
    result.sample_count = static_cast<std::uint64_t>(samples.size());
    result.dt = dt;

    if (definition.storage_scope == ExposureStorageScope::RegionAggregate) {
        Real aggregate = 0.0;
        bool have_extreme = false;
        Real extreme = initialValue(definition.kind);
        for (const auto& sample : samples) {
            if (!ReductionEvaluator::sampleInRegion(definition.region, sample)) {
                continue;
            }
            ++result.selected_sample_count;
            const Real weight = ReductionEvaluator::sampleWeight(
                sample, definition.weighted_region_aggregate);
            switch (definition.kind) {
                case ExposureAccumulationKind::TimeAboveThreshold:
                    if (aboveThreshold(sample.value, definition.threshold, definition.inclusive)) {
                        aggregate += weight * dt;
                    }
                    break;
                case ExposureAccumulationKind::TimeBelowThreshold:
                    if (belowThreshold(sample.value, definition.threshold, definition.inclusive)) {
                        aggregate += weight * dt;
                    }
                    break;
                case ExposureAccumulationKind::PositiveExcessIntegral:
                    aggregate += weight * positiveExcess(definition, sample.value) * dt;
                    break;
                case ExposureAccumulationKind::SquaredPositiveExcessIntegral: {
                    const Real excess = positiveExcess(definition, sample.value);
                    aggregate += weight * excess * excess * dt;
                    break;
                }
                case ExposureAccumulationKind::RunningMaximum:
                case ExposureAccumulationKind::RunningMinimum:
                    if (!have_extreme) {
                        extreme = sample.value;
                        have_extreme = true;
                    } else {
                        updateRunningExtreme(definition.kind, sample.value, extreme);
                    }
                    break;
            }
        }

        if (definition.kind == ExposureAccumulationKind::RunningMaximum ||
            definition.kind == ExposureAccumulationKind::RunningMinimum) {
            if (have_extreme) {
                updateRunningExtreme(definition.kind, extreme, values[0]);
                result.updated_entry_count = 1;
            }
        } else {
            values[0] += aggregate;
            result.updated_entry_count = 1;
        }
        return result;
    }

    std::size_t value_index = 0;
    for (const auto& sample : samples) {
        if (!ReductionEvaluator::sampleInRegion(definition.region, sample)) {
            continue;
        }
        ++result.selected_sample_count;
        if (definition.kind == ExposureAccumulationKind::RunningMaximum ||
            definition.kind == ExposureAccumulationKind::RunningMinimum) {
            updateRunningExtreme(definition.kind, sample.value, values[value_index]);
        } else {
            values[value_index] += entityIncrement(definition, sample.value, dt);
        }
        ++value_index;
    }
    result.updated_entry_count = static_cast<std::uint64_t>(value_index);
    return result;
}

ExposureAccumulatorRestartRecord ExposureAccumulator::packState(
    const ExposureAccumulatorState& state)
{
    ExposureAccumulatorRestartRecord record;
    record.scope = state.scope;
    record.values = state.values;
    record.elapsed_time = state.elapsed_time;
    record.step_count = state.step_count;
    record.initialized = state.initialized;
    return record;
}

ExposureAccumulatorState ExposureAccumulator::unpackState(
    const ExposureAccumulatorRestartRecord& record)
{
    ExposureAccumulatorState state;
    state.scope = record.scope;
    state.values = record.values;
    state.elapsed_time = record.elapsed_time;
    state.step_count = record.step_count;
    state.initialized = record.initialized;
    return state;
}

} // namespace post
} // namespace FE
} // namespace svmp
