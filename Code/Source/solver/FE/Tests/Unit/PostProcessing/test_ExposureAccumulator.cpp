/**
 * @file test_ExposureAccumulator.cpp
 * @brief Unit tests for generic accumulated exposure postprocessing.
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryStateStorage.h"
#include "PostProcessing/ExposureAccumulator.h"

#include <initializer_list>
#include <limits>
#include <vector>

using svmp::FE::post::ExposureAccumulationKind;
using svmp::FE::post::ExposureAccumulator;
using svmp::FE::post::ExposureAccumulatorDefinition;
using svmp::FE::post::ExposureAccumulatorState;
using svmp::FE::post::ExposureExcessDirection;
using svmp::FE::post::ExposureStorageScope;
using svmp::FE::post::ReductionInputDescriptor;
using svmp::FE::post::ReductionInputKind;
using svmp::FE::post::ReductionRegion;
using svmp::FE::post::ReductionRegionKind;
using svmp::FE::post::ReductionSample;
using svmp::FE::systems::AuxiliaryBlockStorage;
using svmp::FE::systems::AuxiliaryStateScope;
using svmp::FE::systems::AuxiliaryStateSpec;

namespace {

std::vector<ReductionSample> samples(std::initializer_list<double> values)
{
    std::vector<ReductionSample> out;
    out.reserve(values.size());
    svmp::FE::GlobalIndex entity_id = 0;
    for (const double value : values) {
        out.push_back(ReductionSample{.value = value,
                                      .weight = 1.0,
                                      .entity_id = entity_id++});
    }
    return out;
}

ExposureAccumulatorDefinition fieldExposure(ExposureAccumulationKind kind,
                                            ExposureStorageScope scope)
{
    ExposureAccumulatorDefinition definition;
    definition.input = ReductionInputDescriptor{.kind = ReductionInputKind::FEField,
                                                .name = "quantity",
                                                .component = 0};
    definition.kind = kind;
    definition.storage_scope = scope;
    definition.threshold = 0.5;
    return definition;
}

} // namespace

TEST(ExposureAccumulator, AccumulatesConstantSignalAboveThreshold)
{
    auto definition = fieldExposure(ExposureAccumulationKind::TimeAboveThreshold,
                                    ExposureStorageScope::Cell);
    ExposureAccumulatorState state;

    const auto step_samples = samples({0.2, 0.8, 1.1});
    const auto result = ExposureAccumulator::advance(definition, step_samples, 0.25, state);

    EXPECT_EQ(result.sample_count, 3u);
    EXPECT_EQ(result.selected_sample_count, 3u);
    ASSERT_EQ(state.values.size(), 3u);
    EXPECT_DOUBLE_EQ(state.values[0], 0.0);
    EXPECT_DOUBLE_EQ(state.values[1], 0.25);
    EXPECT_DOUBLE_EQ(state.values[2], 0.25);
    EXPECT_DOUBLE_EQ(state.elapsed_time, 0.25);
    EXPECT_EQ(state.step_count, 1u);
}

TEST(ExposureAccumulator, AccumulatesThresholdCrossingsAcrossSteps)
{
    auto definition = fieldExposure(ExposureAccumulationKind::TimeBelowThreshold,
                                    ExposureStorageScope::Node);
    ExposureAccumulatorState state;

    ExposureAccumulator::advance(definition, samples({0.4, 0.6}), 0.1, state);
    ExposureAccumulator::advance(definition, samples({0.7, 0.2}), 0.2, state);

    ASSERT_EQ(state.values.size(), 2u);
    EXPECT_DOUBLE_EQ(state.values[0], 0.1);
    EXPECT_DOUBLE_EQ(state.values[1], 0.2);
    EXPECT_DOUBLE_EQ(state.elapsed_time, 0.3);
    EXPECT_EQ(state.step_count, 2u);
}

TEST(ExposureAccumulator, ResetsStateWithKindSpecificInitialValues)
{
    auto definition = fieldExposure(ExposureAccumulationKind::RunningMaximum,
                                    ExposureStorageScope::Cell);
    ExposureAccumulatorState state;
    state.values = {10.0};
    state.initialized = true;

    ExposureAccumulator::reset(definition, samples({0.1, 0.2}), state);

    ASSERT_EQ(state.values.size(), 2u);
    EXPECT_EQ(state.values[0], -std::numeric_limits<double>::infinity());
    EXPECT_EQ(state.values[1], -std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(state.elapsed_time, 0.0);
    EXPECT_EQ(state.step_count, 0u);
}

TEST(ExposureAccumulator, SupportsQuadraturePointExcessStorage)
{
    auto definition = fieldExposure(ExposureAccumulationKind::SquaredPositiveExcessIntegral,
                                    ExposureStorageScope::QuadraturePoint);
    definition.threshold = 1.0;
    definition.excess_direction = ExposureExcessDirection::BelowThreshold;

    ExposureAccumulatorState state;
    ExposureAccumulator::advance(definition, samples({0.5, 1.5}), 2.0, state);

    ASSERT_EQ(state.values.size(), 2u);
    EXPECT_DOUBLE_EQ(state.values[0], 0.5);
    EXPECT_DOUBLE_EQ(state.values[1], 0.0);
}

TEST(ExposureAccumulator, AccumulatesWeightedRegionAggregate)
{
    auto definition = fieldExposure(ExposureAccumulationKind::PositiveExcessIntegral,
                                    ExposureStorageScope::RegionAggregate);
    definition.threshold = 0.5;
    definition.weighted_region_aggregate = true;

    std::vector<ReductionSample> step_samples = {
        ReductionSample{.value = 0.4, .weight = 2.0, .entity_id = 0},
        ReductionSample{.value = 0.7, .weight = 3.0, .entity_id = 1},
        ReductionSample{.value = 1.0, .weight = 4.0, .entity_id = 2}};

    ExposureAccumulatorState state;
    ExposureAccumulator::advance(definition, step_samples, 0.5, state);

    ASSERT_EQ(state.values.size(), 1u);
    EXPECT_DOUBLE_EQ(state.values[0], 1.3);
}

TEST(ExposureAccumulator, RestrictsCellSetRegions)
{
    auto definition = fieldExposure(ExposureAccumulationKind::RunningMinimum,
                                    ExposureStorageScope::Cell);
    definition.region = ReductionRegion{.kind = ReductionRegionKind::CellSet,
                                        .entity_ids = {1, 3}};
    ExposureAccumulatorState state;

    ExposureAccumulator::advance(definition, samples({0.8, 0.3, 0.2, 0.6}), 0.1, state);

    ASSERT_EQ(state.values.size(), 2u);
    EXPECT_DOUBLE_EQ(state.values[0], 0.3);
    EXPECT_DOUBLE_EQ(state.values[1], 0.6);
}

TEST(ExposureAccumulator, PacksAndUnpacksRestartState)
{
    auto definition = fieldExposure(ExposureAccumulationKind::TimeAboveThreshold,
                                    ExposureStorageScope::Cell);
    ExposureAccumulatorState state;
    ExposureAccumulator::advance(definition, samples({0.8}), 0.25, state);

    const auto record = ExposureAccumulator::packState(state);
    const auto restored = ExposureAccumulator::unpackState(record);

    EXPECT_EQ(restored.scope, state.scope);
    EXPECT_EQ(restored.values, state.values);
    EXPECT_DOUBLE_EQ(restored.elapsed_time, state.elapsed_time);
    EXPECT_EQ(restored.step_count, state.step_count);
    EXPECT_EQ(restored.initialized, state.initialized);
}

TEST(ExposureAccumulator, UpdatesAuxiliaryStorageSpan)
{
    AuxiliaryStateSpec spec;
    spec.name = "exposure";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Cell;

    AuxiliaryBlockStorage storage;
    storage.setupFixedStride(spec, 2);

    auto definition = fieldExposure(ExposureAccumulationKind::TimeAboveThreshold,
                                    ExposureStorageScope::Cell);
    ExposureAccumulator::resetSpan(definition, storage.work());
    ExposureAccumulator::advanceIntoSpan(definition, samples({0.6, 0.2}), 0.5,
                                         storage.work());
    storage.commitTimeStep(0.5);

    ASSERT_EQ(storage.committed().size(), 2u);
    EXPECT_DOUBLE_EQ(storage.committed()[0], 0.5);
    EXPECT_DOUBLE_EQ(storage.committed()[1], 0.0);
    ASSERT_EQ(storage.history().depth(), 1u);
    EXPECT_DOUBLE_EQ(storage.history().snapshot(0)[0], 0.0);
}
