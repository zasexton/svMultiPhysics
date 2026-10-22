/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h"

#include "Systems/SystemCollectives.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace level_set = svmp::FE::level_set;

namespace {

constexpr std::uint64_t bit(unsigned index) { return 1ull << index; }

} // namespace

TEST(LevelSetMaintenanceRequestSchedule, ActionBitsNameTheConfiguredAndDueActions)
{
    level_set::LevelSetMaintenanceConfiguration configuration{};
    // A field velocity and the default conservative geometry reconciliation
    // flag; nothing is due.
    EXPECT_EQ(level_set::levelSetMaintenanceRequestActionBits(configuration, 3),
              bit(4u) | bit(10u));

    configuration.transport.velocity.source =
        level_set::LevelSetVelocitySource::ConstantVector;
    configuration.transport.conservative_phase.reconcile_geometry = false;
    EXPECT_EQ(level_set::levelSetMaintenanceRequestActionBits(configuration, 3),
              bit(3u));

    configuration.transport.bound_preserving.enabled = true;
    EXPECT_EQ(level_set::levelSetMaintenanceRequestActionBits(configuration, 3),
              bit(1u) | bit(3u) | bit(8u));

    // The conservative phase writes its flux artifact every second step.
    level_set::LevelSetMaintenanceConfiguration conservative{};
    conservative.transport.conservative_phase.enabled = true;
    conservative.transport.conservative_phase.write_flux_artifacts = true;
    conservative.transport.conservative_phase.flux_artifact_cadence_steps = 2;
    const auto even =
        level_set::levelSetMaintenanceRequestActionBits(conservative, 4);
    const auto odd =
        level_set::levelSetMaintenanceRequestActionBits(conservative, 5);
    EXPECT_EQ(even, bit(0u) | bit(4u) | bit(8u) | bit(9u) | bit(10u));
    EXPECT_EQ(odd, even & ~bit(9u));

    // Bit 11 is left to the application; retired bits stay unused.
    for (const auto bits : {even, odd}) {
        EXPECT_EQ(bits & (bit(2u) | bit(5u) | bit(6u) | bit(7u) | bit(11u)), 0u);
    }
}

TEST(LevelSetMaintenanceRequestSchedule, HeaderAndConfigurationWordsAreCanonical)
{
    std::vector<std::uint64_t> header;
    level_set::appendLevelSetMaintenanceScheduleHeader(
        header,
        level_set::LevelSetMaintenanceScheduleStage::ProspectiveAcceptedEndpoint,
        7, 2u, bit(4u));
    EXPECT_EQ(header, (std::vector<std::uint64_t>{4u, 3u, 7u, 2u, bit(4u)}));

    const level_set::LevelSetMaintenanceConfiguration configuration{};
    std::vector<std::uint64_t> first;
    std::vector<std::uint64_t> second;
    EXPECT_TRUE(level_set::appendLevelSetMaintenanceConfigurationScheduleWords(
        first, configuration));
    EXPECT_TRUE(level_set::appendLevelSetMaintenanceConfigurationScheduleWords(
        second, configuration));
    EXPECT_FALSE(first.empty());
    EXPECT_EQ(first, second);

    auto shifted = configuration;
    shifted.isovalue = 0.25;
    std::vector<std::uint64_t> shifted_words;
    EXPECT_TRUE(level_set::appendLevelSetMaintenanceConfigurationScheduleWords(
        shifted_words, shifted));
    EXPECT_EQ(shifted_words.size(), first.size());
    EXPECT_NE(shifted_words, first);

    std::vector<std::uint64_t> curvature;
    level_set::appendLevelSetCurvatureProjectionScheduleWords(
        curvature, configuration, configuration.curvature_projection);
    auto filtered_options = configuration.curvature_projection;
    filtered_options.kinematic_area_gradient_filter_coefficient += 1.0;
    std::vector<std::uint64_t> filtered;
    level_set::appendLevelSetCurvatureProjectionScheduleWords(
        filtered, configuration, filtered_options);
    EXPECT_NE(curvature, filtered);
}

TEST(LevelSetMaintenanceRequestSchedule, SingleRankAgreementRequiresASupportedSchedule)
{
    const svmp::FE::systems::CollectiveCommunicator self{};
    level_set::CanonicalLevelSetMaintenanceSchedule schedule{};
    schedule.words = {4u, 1u, 0u, 0u, 0u};
    EXPECT_NO_THROW(
        level_set::requireCollectiveLevelSetMaintenanceRequestSchedule(
            schedule, self));
    schedule.supported = false;
    EXPECT_THROW(
        level_set::requireCollectiveLevelSetMaintenanceRequestSchedule(
            schedule, self),
        std::runtime_error);
}

TEST(LevelSetMaintenanceRequestSchedule, GeometryStateWithoutASystemIsUnsupported)
{
    const auto state = level_set::canonicalLevelSetMaintenanceGeometryState(
        nullptr, std::span<const level_set::LevelSetMaintenanceRequest>{});
    EXPECT_FALSE(state.supported);
    EXPECT_EQ(state.words, (std::vector<std::uint64_t>{1u, 0u}));

    std::vector<std::uint64_t> section;
    level_set::appendCanonicalLevelSetMaintenanceGeometrySection(section, state);
    ASSERT_EQ(section.size(), 5u);
    EXPECT_EQ(section[1], 0u);
    EXPECT_EQ(section[2], 2u);
}

TEST(LevelSetMaintenanceRequestSchedule, AlgebraicRevisionFollowsTheStateBits)
{
    const std::vector<svmp::FE::Real> state{1.0, -2.0, 0.5};
    auto changed = state;
    changed[1] = -2.0 + 1.0e-15;
    const auto revision = level_set::levelSetMaintenanceAlgebraicRevision(state);
    EXPECT_NE(revision, 0u);
    EXPECT_EQ(revision, level_set::levelSetMaintenanceAlgebraicRevision(state));
    EXPECT_NE(revision, level_set::levelSetMaintenanceAlgebraicRevision(changed));
    EXPECT_NE(level_set::levelSetMaintenanceAlgebraicRevision(
                  std::span<const svmp::FE::Real>{}),
              0u);
    EXPECT_EQ(level_set::collectiveLevelSetMaintenanceAlgebraicRevision(
                  state, svmp::FE::systems::CollectiveCommunicator{}),
              revision);
}
