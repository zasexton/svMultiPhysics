/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_POSTPROCESSING_REDUCTION_H
#define SVMP_FE_POSTPROCESSING_REDUCTION_H

#include "Core/Types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace post {

enum class ReductionRegionKind : std::uint8_t {
    WholeDomain,
    NamedRegion,
    Boundary,
    CellSet
};

enum class ReductionInputKind : std::uint8_t {
    FEField,
    DerivedField,
    AuxiliaryField,
    Expression
};

enum class ThresholdSense : std::uint8_t {
    Below,
    BelowOrEqual,
    Above,
    AboveOrEqual
};

struct ReductionRegion {
    ReductionRegionKind kind{ReductionRegionKind::WholeDomain};
    std::string name;
    int marker{-1};
    std::vector<GlobalIndex> entity_ids;
};

struct ReductionInputDescriptor {
    ReductionInputKind kind{ReductionInputKind::FEField};
    std::string name;
    int component{-1};
};

struct ReductionSample {
    Real value{0.0};
    Real weight{1.0};
    GlobalIndex entity_id{INVALID_GLOBAL_INDEX};
};

struct ThresholdReductionDefinition {
    ReductionInputDescriptor input{};
    ReductionRegion region{};
    ThresholdSense sense{ThresholdSense::Below};
    Real threshold{0.0};
    bool weighted{true};
};

struct ThresholdReductionResult {
    Real measure{0.0};
    Real integral{0.0};
    std::uint64_t matching_count{0};
    std::uint64_t sample_count{0};
};

struct HistogramDefinition {
    ReductionInputDescriptor input{};
    ReductionRegion region{};
    Real lower_bound{0.0};
    Real upper_bound{1.0};
    std::size_t bin_count{1};
    bool weighted{true};
    bool include_underflow{true};
    bool include_overflow{true};
};

struct HistogramResult {
    std::vector<Real> bin_edges;
    std::vector<Real> bin_weights;
    Real underflow_weight{0.0};
    Real overflow_weight{0.0};
    std::uint64_t sample_count{0};
};

struct PercentileDefinition {
    ReductionInputDescriptor input{};
    ReductionRegion region{};
    Real percentile{50.0};
    bool weighted{true};
};

struct MinMaxPercentileSummaryDefinition {
    ReductionInputDescriptor input{};
    ReductionRegion region{};
    std::vector<Real> percentiles;
    bool weighted{true};
};

struct MinMaxPercentileSummary {
    std::optional<Real> minimum{};
    std::optional<Real> maximum{};
    std::vector<Real> percentiles;
    std::vector<Real> values;
    std::uint64_t sample_count{0};
};

class ReductionEvaluator {
public:
    static ThresholdReductionResult thresholdMeasure(
        const ThresholdReductionDefinition& definition,
        std::span<const ReductionSample> samples);

    static ThresholdReductionResult thresholdIntegral(
        const ThresholdReductionDefinition& definition,
        std::span<const ReductionSample> samples);

    static HistogramResult histogram(const HistogramDefinition& definition,
                                     std::span<const ReductionSample> samples);

    static std::optional<Real> percentile(const PercentileDefinition& definition,
                                          std::span<const ReductionSample> samples);

    static MinMaxPercentileSummary minMaxPercentileSummary(
        const MinMaxPercentileSummaryDefinition& definition,
        std::span<const ReductionSample> samples);

private:
    static bool inRegion(const ReductionRegion& region, const ReductionSample& sample);
    static Real sampleWeight(const ReductionSample& sample, bool weighted);
};

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_REDUCTION_H
