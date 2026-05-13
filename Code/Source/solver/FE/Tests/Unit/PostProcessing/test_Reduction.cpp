/**
 * @file test_Reduction.cpp
 * @brief Unit tests for generic postprocessing reductions.
 */

#include <gtest/gtest.h>

#include "PostProcessing/Reduction.h"

#include <vector>

using svmp::FE::post::HistogramDefinition;
using svmp::FE::post::ReductionEvaluator;
using svmp::FE::post::ReductionInputDescriptor;
using svmp::FE::post::ReductionInputKind;
using svmp::FE::post::ReductionRegion;
using svmp::FE::post::ReductionRegionKind;
using svmp::FE::post::ReductionSample;
using svmp::FE::post::MinMaxPercentileSummaryDefinition;
using svmp::FE::post::PercentileDefinition;
using svmp::FE::post::ThresholdReductionDefinition;
using svmp::FE::post::ThresholdSense;

namespace {

std::vector<ReductionSample> weightedSamples()
{
    return {
        ReductionSample{.value = 0.1, .weight = 2.0, .entity_id = 0},
        ReductionSample{.value = 0.4, .weight = 1.0, .entity_id = 1},
        ReductionSample{.value = 0.8, .weight = 3.0, .entity_id = 2},
        ReductionSample{.value = 1.0, .weight = 4.0, .entity_id = 3}};
}

ReductionInputDescriptor derivedInput()
{
    return ReductionInputDescriptor{.kind = ReductionInputKind::DerivedField,
                                    .name = "derived_scalar",
                                    .component = 0};
}

} // namespace

TEST(ReductionEvaluator, ComputesThresholdMeasureAndIntegral)
{
    ThresholdReductionDefinition definition;
    definition.input = derivedInput();
    definition.sense = ThresholdSense::Below;
    definition.threshold = 0.5;
    definition.weighted = true;

    const auto samples = weightedSamples();
    const auto measure = ReductionEvaluator::thresholdMeasure(definition, samples);
    const auto integral = ReductionEvaluator::thresholdIntegral(definition, samples);

    EXPECT_EQ(measure.sample_count, 4u);
    EXPECT_EQ(measure.matching_count, 2u);
    EXPECT_DOUBLE_EQ(measure.measure, 3.0);
    EXPECT_DOUBLE_EQ(integral.integral, 0.6);
}

TEST(ReductionEvaluator, RestrictsCellSetRegions)
{
    ThresholdReductionDefinition definition;
    definition.input = derivedInput();
    definition.region = ReductionRegion{.kind = ReductionRegionKind::CellSet,
                                        .entity_ids = {1, 3}};
    definition.sense = ThresholdSense::AboveOrEqual;
    definition.threshold = 0.4;
    definition.weighted = false;

    const auto samples = weightedSamples();
    const auto result = ReductionEvaluator::thresholdMeasure(definition, samples);

    EXPECT_EQ(result.matching_count, 2u);
    EXPECT_DOUBLE_EQ(result.measure, 2.0);
}

TEST(ReductionEvaluator, ComputesDeterministicHistograms)
{
    HistogramDefinition definition;
    definition.input = ReductionInputDescriptor{.kind = ReductionInputKind::FEField,
                                                .name = "u"};
    definition.lower_bound = 0.0;
    definition.upper_bound = 1.0;
    definition.bin_count = 2;
    definition.weighted = false;

    std::vector<ReductionSample> samples = {
        ReductionSample{.value = -0.1},
        ReductionSample{.value = 0.0},
        ReductionSample{.value = 0.49},
        ReductionSample{.value = 0.5},
        ReductionSample{.value = 1.0},
        ReductionSample{.value = 1.2}};

    const auto result = ReductionEvaluator::histogram(definition, samples);
    ASSERT_EQ(result.bin_edges.size(), 3u);
    EXPECT_DOUBLE_EQ(result.bin_edges[0], 0.0);
    EXPECT_DOUBLE_EQ(result.bin_edges[1], 0.5);
    EXPECT_DOUBLE_EQ(result.bin_edges[2], 1.0);
    ASSERT_EQ(result.bin_weights.size(), 2u);
    EXPECT_DOUBLE_EQ(result.bin_weights[0], 2.0);
    EXPECT_DOUBLE_EQ(result.bin_weights[1], 2.0);
    EXPECT_DOUBLE_EQ(result.underflow_weight, 1.0);
    EXPECT_DOUBLE_EQ(result.overflow_weight, 1.0);
}

TEST(ReductionEvaluator, ComputesWeightedPercentiles)
{
    PercentileDefinition definition;
    definition.input = derivedInput();
    definition.percentile = 50.0;
    definition.weighted = true;

    const auto samples = weightedSamples();
    const auto p50 = ReductionEvaluator::percentile(definition, samples);

    ASSERT_TRUE(p50.has_value());
    EXPECT_DOUBLE_EQ(*p50, 0.8);
}

TEST(ReductionEvaluator, HandlesEmptyPercentileRegions)
{
    PercentileDefinition definition;
    definition.input = derivedInput();
    definition.region = ReductionRegion{.kind = ReductionRegionKind::CellSet,
                                        .entity_ids = {99}};
    definition.percentile = 95.0;

    const auto samples = weightedSamples();
    EXPECT_FALSE(ReductionEvaluator::percentile(definition, samples).has_value());
}

TEST(ReductionEvaluator, ComputesMinMaxPercentileSummary)
{
    MinMaxPercentileSummaryDefinition definition;
    definition.input = ReductionInputDescriptor{.kind = ReductionInputKind::Expression,
                                                .name = "expr"};
    definition.percentiles = {0.0, 50.0, 100.0};
    definition.weighted = false;

    const auto samples = weightedSamples();
    const auto summary = ReductionEvaluator::minMaxPercentileSummary(definition, samples);

    ASSERT_TRUE(summary.minimum.has_value());
    ASSERT_TRUE(summary.maximum.has_value());
    EXPECT_DOUBLE_EQ(*summary.minimum, 0.1);
    EXPECT_DOUBLE_EQ(*summary.maximum, 1.0);
    EXPECT_EQ(summary.percentiles, std::vector<double>({0.0, 50.0, 100.0}));
    EXPECT_EQ(summary.values, std::vector<double>({0.1, 0.4, 1.0}));
}
