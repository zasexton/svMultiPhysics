#include "LevelSet/LevelSetOptions.h"

#include <gtest/gtest.h>

#include <variant>

namespace level_set = svmp::FE::level_set;

TEST(LevelSetOptions, DefaultsAreNeutral)
{
    const level_set::LevelSetTransportOptions options{};

    EXPECT_EQ(options.operator_tag, "level_set");
    EXPECT_EQ(options.level_set.field_name, "level_set");
    EXPECT_EQ(options.level_set.source, level_set::LevelSetFieldSource::Unknown);
    EXPECT_TRUE(options.level_set.auto_register_field);
    EXPECT_EQ(options.velocity.field_name, "Velocity");
    EXPECT_EQ(options.velocity.source, level_set::LevelSetVelocitySource::CoupledField);
    EXPECT_FALSE(options.velocity.auto_register_field);
    EXPECT_EQ(options.velocity.space, nullptr);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[0], 0.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[1], 0.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[2], 0.0);
    EXPECT_FALSE(options.supg.enabled);
    EXPECT_DOUBLE_EQ(options.supg.tau_scale, 0.5);
    EXPECT_DOUBLE_EQ(options.supg.velocity_epsilon, 1.0e-12);
    EXPECT_FALSE(options.reinitialization.enabled);
    EXPECT_EQ(options.reinitialization.method,
              level_set::LevelSetReinitializationMethod::HamiltonJacobiPDE);
    EXPECT_EQ(options.reinitialization.cadence_steps, 1);
    EXPECT_EQ(options.reinitialization.max_iterations, 10);
    EXPECT_DOUBLE_EQ(options.reinitialization.pseudo_time_step_scale, 0.3);
    EXPECT_DOUBLE_EQ(options.reinitialization.interface_band_width, 3.0);
    EXPECT_DOUBLE_EQ(options.reinitialization.signed_distance_tolerance, 1.0e-6);
    EXPECT_FALSE(options.volume_correction.enabled);
    EXPECT_EQ(options.volume_correction.cadence_steps, 1);
    EXPECT_TRUE(options.volume_correction.use_initial_negative_volume_as_target);
    EXPECT_DOUBLE_EQ(options.volume_correction.target_negative_volume, 0.0);
    EXPECT_DOUBLE_EQ(options.volume_correction.volume_tolerance, 1.0e-10);
    EXPECT_EQ(options.volume_correction.max_iterations, 50);
    EXPECT_TRUE(options.boundaries.inflow.empty());
    EXPECT_TRUE(options.boundaries.outflow.empty());
}

TEST(LevelSetOptions, ExplicitTransportOptions)
{
    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.velocity.auto_register_field = true;
    options.velocity.constant_value = {1.0, -2.0, 0.5};
    options.supg.enabled = true;
    options.supg.tau_scale = 0.25;
    options.supg.velocity_epsilon = 1.0e-8;
    options.reinitialization.enabled = true;
    options.reinitialization.method = level_set::LevelSetReinitializationMethod::Projection;
    options.reinitialization.cadence_steps = 4;
    options.reinitialization.max_iterations = 12;
    options.reinitialization.pseudo_time_step_scale = 0.20;
    options.reinitialization.interface_band_width = 2.5;
    options.reinitialization.signed_distance_tolerance = 1.0e-5;
    options.volume_correction.enabled = true;
    options.volume_correction.cadence_steps = 5;
    options.volume_correction.use_initial_negative_volume_as_target = false;
    options.volume_correction.target_negative_volume = 0.375;
    options.volume_correction.volume_tolerance = 1.0e-7;
    options.volume_correction.max_iterations = 24;
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 11,
        .value = svmp::FE::Real{2.5},
        .penalty_scale = 3.0,
    });
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 12});

    EXPECT_EQ(options.level_set.field_name, "phi");
    EXPECT_EQ(options.level_set.source, level_set::LevelSetFieldSource::PrescribedData);
    EXPECT_FALSE(options.level_set.auto_register_field);
    EXPECT_EQ(options.velocity.field_name, "advecting_velocity");
    EXPECT_EQ(options.velocity.source, level_set::LevelSetVelocitySource::PrescribedData);
    EXPECT_TRUE(options.velocity.auto_register_field);
    EXPECT_EQ(options.velocity.space, nullptr);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[0], 1.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[1], -2.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[2], 0.5);
    EXPECT_TRUE(options.supg.enabled);
    EXPECT_DOUBLE_EQ(options.supg.tau_scale, 0.25);
    EXPECT_DOUBLE_EQ(options.supg.velocity_epsilon, 1.0e-8);
    EXPECT_TRUE(options.reinitialization.enabled);
    EXPECT_EQ(options.reinitialization.method, level_set::LevelSetReinitializationMethod::Projection);
    EXPECT_EQ(options.reinitialization.cadence_steps, 4);
    EXPECT_EQ(options.reinitialization.max_iterations, 12);
    EXPECT_DOUBLE_EQ(options.reinitialization.pseudo_time_step_scale, 0.20);
    EXPECT_DOUBLE_EQ(options.reinitialization.interface_band_width, 2.5);
    EXPECT_DOUBLE_EQ(options.reinitialization.signed_distance_tolerance, 1.0e-5);
    EXPECT_TRUE(options.volume_correction.enabled);
    EXPECT_EQ(options.volume_correction.cadence_steps, 5);
    EXPECT_FALSE(options.volume_correction.use_initial_negative_volume_as_target);
    EXPECT_DOUBLE_EQ(options.volume_correction.target_negative_volume, 0.375);
    EXPECT_DOUBLE_EQ(options.volume_correction.volume_tolerance, 1.0e-7);
    EXPECT_EQ(options.volume_correction.max_iterations, 24);
    ASSERT_EQ(options.boundaries.inflow.size(), 1u);
    EXPECT_EQ(options.boundaries.inflow.front().boundary_marker, 11);
    EXPECT_DOUBLE_EQ(std::get<svmp::FE::Real>(options.boundaries.inflow.front().value), 2.5);
    EXPECT_DOUBLE_EQ(options.boundaries.inflow.front().penalty_scale, 3.0);
    ASSERT_EQ(options.boundaries.outflow.size(), 1u);
    EXPECT_EQ(options.boundaries.outflow.front().boundary_marker, 12);
}

TEST(LevelSetOptions, ReinitializationCadence)
{
    level_set::LevelSetReinitializationOptions options{};
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 1));

    options.enabled = true;
    options.cadence_steps = 3;
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, -1));
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 0));
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 1));
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 2));
    EXPECT_TRUE(level_set::shouldReinitializeLevelSet(options, 3));
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 4));
    EXPECT_TRUE(level_set::shouldReinitializeLevelSet(options, 6));

    options.cadence_steps = 0;
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 3));
}

TEST(LevelSetOptions, VolumeCorrectionCadence)
{
    level_set::LevelSetVolumeCorrectionOptions options{};
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 1));

    options.enabled = true;
    options.cadence_steps = 4;
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, -1));
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 0));
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 1));
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 3));
    EXPECT_TRUE(level_set::shouldApplyLevelSetVolumeCorrection(options, 4));
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 5));
    EXPECT_TRUE(level_set::shouldApplyLevelSetVolumeCorrection(options, 8));

    options.cadence_steps = -1;
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 4));
}
