#include "LevelSet/LevelSetOptions.h"

#include <gtest/gtest.h>

namespace level_set = svmp::FE::level_set;

TEST(LevelSetOptions, DefaultsAreNeutral)
{
    const level_set::LevelSetTransportOptions options{};

    EXPECT_EQ(options.level_set.field_name, "level_set");
    EXPECT_EQ(options.level_set.source, level_set::LevelSetFieldSource::Unknown);
    EXPECT_TRUE(options.level_set.auto_register_field);
    EXPECT_EQ(options.velocity.field_name, "Velocity");
    EXPECT_EQ(options.velocity.source, level_set::LevelSetVelocitySource::CoupledField);
    EXPECT_FALSE(options.supg.enabled);
    EXPECT_EQ(options.reinitialization.method,
              level_set::LevelSetReinitializationMethod::HamiltonJacobiPDE);
    EXPECT_TRUE(options.volume_correction.use_initial_negative_volume_as_target);
}

TEST(LevelSetOptions, ReinitializationCadence)
{
    level_set::LevelSetReinitializationOptions options{};
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 1));

    options.enabled = true;
    options.cadence_steps = 3;
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 0));
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 2));
    EXPECT_TRUE(level_set::shouldReinitializeLevelSet(options, 3));

    options.cadence_steps = 0;
    EXPECT_FALSE(level_set::shouldReinitializeLevelSet(options, 3));
}

TEST(LevelSetOptions, VolumeCorrectionCadence)
{
    level_set::LevelSetVolumeCorrectionOptions options{};
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 1));

    options.enabled = true;
    options.cadence_steps = 4;
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 0));
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 3));
    EXPECT_TRUE(level_set::shouldApplyLevelSetVolumeCorrection(options, 4));

    options.cadence_steps = -1;
    EXPECT_FALSE(level_set::shouldApplyLevelSetVolumeCorrection(options, 4));
}
