/**
 * @file test_PostProcessingInfrastructureIntegration.cpp
 * @brief Integration-style tests for generic FE postprocessing infrastructure.
 */

#include <gtest/gtest.h>

#include "PostProcessing/ExposureAccumulator.h"
#include "PostProcessing/Reduction.h"

#include <vector>

using svmp::FE::post::ExposureAccumulationKind;
using svmp::FE::post::ExposureAccumulator;
using svmp::FE::post::ExposureAccumulatorDefinition;
using svmp::FE::post::ExposureAccumulatorState;
using svmp::FE::post::ExposureStorageScope;
using svmp::FE::post::PercentileDefinition;
using svmp::FE::post::ReductionEvaluator;
using svmp::FE::post::ReductionInputDescriptor;
using svmp::FE::post::ReductionInputKind;
using svmp::FE::post::ReductionSample;
using svmp::FE::post::ThresholdReductionDefinition;
using svmp::FE::post::ThresholdSense;

TEST(PostProcessingInfrastructureIntegration, ThresholdAndPercentileShareSamples)
{
    const std::vector<ReductionSample> samples = {
        ReductionSample{.value = 0.1, .weight = 2.0, .entity_id = 0},
        ReductionSample{.value = 0.4, .weight = 1.0, .entity_id = 1},
        ReductionSample{.value = 0.8, .weight = 3.0, .entity_id = 2},
        ReductionSample{.value = 1.0, .weight = 4.0, .entity_id = 3}};

    ReductionInputDescriptor input;
    input.kind = ReductionInputKind::DerivedField;
    input.name = "arbitrary_scalar";

    ThresholdReductionDefinition threshold;
    threshold.input = input;
    threshold.sense = ThresholdSense::Below;
    threshold.threshold = 0.5;
    threshold.weighted = true;

    PercentileDefinition percentile;
    percentile.input = input;
    percentile.percentile = 50.0;
    percentile.weighted = true;

    const auto measure = ReductionEvaluator::thresholdMeasure(threshold, samples);
    const auto p50 = ReductionEvaluator::percentile(percentile, samples);

    EXPECT_DOUBLE_EQ(measure.measure, 3.0);
    ASSERT_TRUE(p50.has_value());
    EXPECT_DOUBLE_EQ(*p50, 0.8);
}

TEST(PostProcessingInfrastructureIntegration, ExposureAccumulatorUsesAcceptedStepSizes)
{
    ExposureAccumulatorDefinition definition;
    definition.input = ReductionInputDescriptor{.kind = ReductionInputKind::FEField,
                                                .name = "arbitrary_field",
                                                .component = 0};
    definition.kind = ExposureAccumulationKind::TimeAboveThreshold;
    definition.storage_scope = ExposureStorageScope::Cell;
    definition.threshold = 0.5;

    ExposureAccumulatorState state;
    const std::vector<double> accepted_dt = {0.1, 0.2, 0.15};
    for (const double dt : accepted_dt) {
        const std::vector<ReductionSample> samples = {
            ReductionSample{.value = 0.6, .weight = 1.0, .entity_id = 0}};
        ExposureAccumulator::advance(definition, samples, dt, state);
    }

    ASSERT_EQ(state.values.size(), 1u);
    EXPECT_DOUBLE_EQ(state.values[0], 0.45);
    EXPECT_DOUBLE_EQ(state.elapsed_time, 0.45);
    EXPECT_EQ(state.step_count, 3u);
}
