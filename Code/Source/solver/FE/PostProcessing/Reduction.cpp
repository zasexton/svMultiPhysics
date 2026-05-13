/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "PostProcessing/Reduction.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace svmp {
namespace FE {
namespace post {

namespace {

[[nodiscard]] bool matchesThreshold(Real value, ThresholdSense sense, Real threshold) noexcept
{
    switch (sense) {
        case ThresholdSense::Below: return value < threshold;
        case ThresholdSense::BelowOrEqual: return value <= threshold;
        case ThresholdSense::Above: return value > threshold;
        case ThresholdSense::AboveOrEqual: return value >= threshold;
    }
    return false;
}

[[nodiscard]] bool finiteWeight(Real weight) noexcept
{
    return std::isfinite(weight) && weight >= 0.0;
}

} // namespace

bool ReductionEvaluator::sampleInRegion(const ReductionRegion& region,
                                        const ReductionSample& sample)
{
    switch (region.kind) {
        case ReductionRegionKind::WholeDomain:
        case ReductionRegionKind::NamedRegion:
        case ReductionRegionKind::Boundary:
            return true;
        case ReductionRegionKind::CellSet:
            return std::find(region.entity_ids.begin(), region.entity_ids.end(),
                             sample.entity_id) != region.entity_ids.end();
    }
    return true;
}

Real ReductionEvaluator::sampleWeight(const ReductionSample& sample, bool weighted)
{
    if (!weighted) {
        return 1.0;
    }
    FE_THROW_IF(!finiteWeight(sample.weight), InvalidArgumentException,
                "ReductionEvaluator: sample weight must be finite and nonnegative");
    return sample.weight;
}

ThresholdReductionResult ReductionEvaluator::thresholdMeasure(
    const ThresholdReductionDefinition& definition,
    std::span<const ReductionSample> samples)
{
    ThresholdReductionResult result;
    result.sample_count = static_cast<std::uint64_t>(samples.size());
    for (const auto& sample : samples) {
        if (!sampleInRegion(definition.region, sample)) {
            continue;
        }
        if (matchesThreshold(sample.value, definition.sense, definition.threshold)) {
            ++result.matching_count;
            result.measure += sampleWeight(sample, definition.weighted);
        }
    }
    return result;
}

ThresholdReductionResult ReductionEvaluator::thresholdIntegral(
    const ThresholdReductionDefinition& definition,
    std::span<const ReductionSample> samples)
{
    ThresholdReductionResult result = thresholdMeasure(definition, samples);
    result.integral = 0.0;
    for (const auto& sample : samples) {
        if (!sampleInRegion(definition.region, sample)) {
            continue;
        }
        if (matchesThreshold(sample.value, definition.sense, definition.threshold)) {
            result.integral += sampleWeight(sample, definition.weighted) * sample.value;
        }
    }
    return result;
}

HistogramResult ReductionEvaluator::histogram(const HistogramDefinition& definition,
                                              std::span<const ReductionSample> samples)
{
    FE_THROW_IF(definition.bin_count == 0, InvalidArgumentException,
                "ReductionEvaluator::histogram: bin_count must be positive");
    FE_THROW_IF(!(definition.lower_bound < definition.upper_bound),
                InvalidArgumentException,
                "ReductionEvaluator::histogram: lower_bound must be less than upper_bound");

    HistogramResult result;
    result.sample_count = static_cast<std::uint64_t>(samples.size());
    result.bin_edges.resize(definition.bin_count + 1);
    result.bin_weights.assign(definition.bin_count, 0.0);

    const Real width =
        (definition.upper_bound - definition.lower_bound) /
        static_cast<Real>(definition.bin_count);
    for (std::size_t i = 0; i <= definition.bin_count; ++i) {
        result.bin_edges[i] = definition.lower_bound + static_cast<Real>(i) * width;
    }
    result.bin_edges.back() = definition.upper_bound;

    for (const auto& sample : samples) {
        if (!sampleInRegion(definition.region, sample)) {
            continue;
        }
        const Real weight = sampleWeight(sample, definition.weighted);
        if (sample.value < definition.lower_bound) {
            if (definition.include_underflow) {
                result.underflow_weight += weight;
            }
            continue;
        }
        if (sample.value > definition.upper_bound) {
            if (definition.include_overflow) {
                result.overflow_weight += weight;
            }
            continue;
        }
        std::size_t bin = 0;
        if (sample.value == definition.upper_bound) {
            bin = definition.bin_count - 1;
        } else {
            bin = static_cast<std::size_t>(
                (sample.value - definition.lower_bound) / width);
            bin = std::min(bin, definition.bin_count - 1);
        }
        result.bin_weights[bin] += weight;
    }
    return result;
}

std::optional<Real> ReductionEvaluator::percentile(
    const PercentileDefinition& definition,
    std::span<const ReductionSample> samples)
{
    FE_THROW_IF(definition.percentile < 0.0 || definition.percentile > 100.0,
                InvalidArgumentException,
                "ReductionEvaluator::percentile: percentile must be in [0, 100]");

    std::vector<std::pair<Real, Real>> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
        if (!sampleInRegion(definition.region, sample)) {
            continue;
        }
        const Real weight = sampleWeight(sample, definition.weighted);
        if (weight > 0.0) {
            values.emplace_back(sample.value, weight);
        }
    }
    if (values.empty()) {
        return std::nullopt;
    }

    std::stable_sort(values.begin(), values.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.first < rhs.first;
                     });

    if (definition.percentile == 0.0) {
        return values.front().first;
    }
    if (definition.percentile == 100.0) {
        return values.back().first;
    }

    const Real total_weight = std::accumulate(
        values.begin(), values.end(), Real{0.0},
        [](Real sum, const auto& sample) { return sum + sample.second; });
    const Real target = total_weight * definition.percentile / 100.0;
    Real cumulative = 0.0;
    for (const auto& [value, weight] : values) {
        cumulative += weight;
        if (cumulative >= target) {
            return value;
        }
    }
    return values.back().first;
}

MinMaxPercentileSummary ReductionEvaluator::minMaxPercentileSummary(
    const MinMaxPercentileSummaryDefinition& definition,
    std::span<const ReductionSample> samples)
{
    MinMaxPercentileSummary summary;
    summary.percentiles = definition.percentiles;
    summary.sample_count = static_cast<std::uint64_t>(samples.size());

    bool have_value = false;
    for (const auto& sample : samples) {
        if (!sampleInRegion(definition.region, sample)) {
            continue;
        }
        if (!have_value) {
            summary.minimum = sample.value;
            summary.maximum = sample.value;
            have_value = true;
        } else {
            summary.minimum = std::min(*summary.minimum, sample.value);
            summary.maximum = std::max(*summary.maximum, sample.value);
        }
    }

    summary.values.reserve(definition.percentiles.size());
    for (const Real p : definition.percentiles) {
        PercentileDefinition percentile_def;
        percentile_def.input = definition.input;
        percentile_def.region = definition.region;
        percentile_def.percentile = p;
        percentile_def.weighted = definition.weighted;
        const auto value = percentile(percentile_def, samples);
        if (value.has_value()) {
            summary.values.push_back(*value);
        } else {
            summary.values.push_back(0.0);
        }
    }
    return summary;
}

} // namespace post
} // namespace FE
} // namespace svmp
