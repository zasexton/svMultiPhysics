/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_POSTPROCESSING_EXPOSURE_ACCUMULATOR_H
#define SVMP_FE_POSTPROCESSING_EXPOSURE_ACCUMULATOR_H

#include "Core/Types.h"
#include "PostProcessing/Reduction.h"

#include <cstdint>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace post {

enum class ExposureAccumulationKind : std::uint8_t {
    TimeAboveThreshold,
    TimeBelowThreshold,
    PositiveExcessIntegral,
    SquaredPositiveExcessIntegral,
    RunningMaximum,
    RunningMinimum
};

enum class ExposureStorageScope : std::uint8_t {
    Node,
    Cell,
    QuadraturePoint,
    RegionAggregate
};

enum class ExposureExcessDirection : std::uint8_t {
    AboveThreshold,
    BelowThreshold
};

struct ExposureAccumulatorDefinition {
    ReductionInputDescriptor input{};
    ReductionRegion region{};
    ExposureAccumulationKind kind{ExposureAccumulationKind::TimeAboveThreshold};
    ExposureStorageScope storage_scope{ExposureStorageScope::Cell};
    Real threshold{0.0};
    bool inclusive{false};
    ExposureExcessDirection excess_direction{ExposureExcessDirection::AboveThreshold};
    bool weighted_region_aggregate{true};
};

struct ExposureAccumulatorState {
    ExposureStorageScope scope{ExposureStorageScope::Cell};
    std::vector<Real> values;
    Real elapsed_time{0.0};
    std::uint64_t step_count{0};
    bool initialized{false};
};

struct ExposureAccumulatorRestartRecord {
    ExposureStorageScope scope{ExposureStorageScope::Cell};
    std::vector<Real> values;
    Real elapsed_time{0.0};
    std::uint64_t step_count{0};
    bool initialized{false};
};

struct ExposureAccumulatorAdvanceResult {
    std::uint64_t sample_count{0};
    std::uint64_t selected_sample_count{0};
    std::uint64_t updated_entry_count{0};
    Real dt{0.0};
};

class ExposureAccumulator {
public:
    static Real initialValue(ExposureAccumulationKind kind) noexcept;

    static std::size_t requiredValueCount(
        const ExposureAccumulatorDefinition& definition,
        std::span<const ReductionSample> samples);

    static void reset(const ExposureAccumulatorDefinition& definition,
                      std::span<const ReductionSample> samples,
                      ExposureAccumulatorState& state);

    static void resetSpan(const ExposureAccumulatorDefinition& definition,
                          std::span<Real> values);

    static ExposureAccumulatorAdvanceResult advance(
        const ExposureAccumulatorDefinition& definition,
        std::span<const ReductionSample> samples,
        Real dt,
        ExposureAccumulatorState& state);

    static ExposureAccumulatorAdvanceResult advanceIntoSpan(
        const ExposureAccumulatorDefinition& definition,
        std::span<const ReductionSample> samples,
        Real dt,
        std::span<Real> values);

    static ExposureAccumulatorRestartRecord packState(
        const ExposureAccumulatorState& state);

    static ExposureAccumulatorState unpackState(
        const ExposureAccumulatorRestartRecord& record);
};

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_EXPOSURE_ACCUMULATOR_H
