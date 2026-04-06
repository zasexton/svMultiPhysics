/**
 * @file test_AuxiliaryHistoryBuffer.cpp
 * @brief Unit tests for AuxiliaryHistoryBuffer — time-stamped history snapshots
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryHistoryBuffer.h"

#include <cmath>
#include <vector>

using svmp::FE::Real;
using namespace svmp::FE::systems;

// ---------------------------------------------------------------------------
//  Setup and basic properties
// ---------------------------------------------------------------------------

TEST(AuxiliaryHistoryBuffer, DefaultIsNotSetUp)
{
    AuxiliaryHistoryBuffer buf;
    EXPECT_FALSE(buf.isSetup());
    EXPECT_EQ(buf.depth(), 0u);
    EXPECT_TRUE(buf.empty());
}

TEST(AuxiliaryHistoryBuffer, SetupConfiguresProperties)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(/*storage_size=*/4, /*max_depth=*/3);

    EXPECT_TRUE(buf.isSetup());
    EXPECT_EQ(buf.maxDepth(), 3u);
    EXPECT_EQ(buf.storageSize(), 4u);
    EXPECT_EQ(buf.depth(), 0u);
    EXPECT_TRUE(buf.empty());
}

// ---------------------------------------------------------------------------
//  Push and step-back access
// ---------------------------------------------------------------------------

TEST(AuxiliaryHistoryBuffer, PushAndRetrieve)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(2, 4);

    const std::vector<Real> snap1 = {1.0, 2.0};
    const std::vector<Real> snap2 = {3.0, 4.0};

    buf.push(0.1, snap1);
    EXPECT_EQ(buf.depth(), 1u);
    EXPECT_FALSE(buf.empty());

    buf.push(0.2, snap2);
    EXPECT_EQ(buf.depth(), 2u);

    // Index 0 = most recent (snap2)
    auto s0 = buf.snapshot(0);
    ASSERT_EQ(s0.size(), 2u);
    EXPECT_DOUBLE_EQ(s0[0], 3.0);
    EXPECT_DOUBLE_EQ(s0[1], 4.0);

    // Index 1 = older (snap1)
    auto s1 = buf.snapshot(1);
    EXPECT_DOUBLE_EQ(s1[0], 1.0);
    EXPECT_DOUBLE_EQ(s1[1], 2.0);
}

TEST(AuxiliaryHistoryBuffer, PushDropsOldestWhenFull)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 2); // max 2 snapshots

    buf.push(0.1, std::vector<Real>{10.0});
    buf.push(0.2, std::vector<Real>{20.0});
    EXPECT_EQ(buf.depth(), 2u);

    buf.push(0.3, std::vector<Real>{30.0});
    EXPECT_EQ(buf.depth(), 2u); // Still 2

    // Oldest (t=0.1) was dropped
    EXPECT_DOUBLE_EQ(buf.snapshot(0)[0], 30.0); // newest
    EXPECT_DOUBLE_EQ(buf.snapshot(1)[0], 20.0); // oldest remaining
}

TEST(AuxiliaryHistoryBuffer, ZeroDepthDiscardsAllPushes)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(2, 0); // max 0 snapshots

    buf.push(0.1, std::vector<Real>{1.0, 2.0});
    EXPECT_EQ(buf.depth(), 0u);
    EXPECT_TRUE(buf.empty());
}

// ---------------------------------------------------------------------------
//  Timestamps
// ---------------------------------------------------------------------------

TEST(AuxiliaryHistoryBuffer, SnapshotTimesAreStored)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 5);

    buf.push(1.0, std::vector<Real>{0.0});
    buf.push(2.0, std::vector<Real>{0.0});
    buf.push(3.0, std::vector<Real>{0.0});

    EXPECT_DOUBLE_EQ(buf.snapshotTime(0), 3.0); // newest
    EXPECT_DOUBLE_EQ(buf.snapshotTime(1), 2.0);
    EXPECT_DOUBLE_EQ(buf.snapshotTime(2), 1.0); // oldest
}

TEST(AuxiliaryHistoryBuffer, FindSnapshotAtTimeExact)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 5);

    buf.push(1.0, std::vector<Real>{10.0});
    buf.push(2.0, std::vector<Real>{20.0});
    buf.push(3.0, std::vector<Real>{30.0});

    EXPECT_EQ(buf.findSnapshotAtTime(2.0), 1u);
    EXPECT_EQ(buf.findSnapshotAtTime(1.0), 2u);
    EXPECT_EQ(buf.findSnapshotAtTime(3.0), 0u);
    EXPECT_EQ(buf.findSnapshotAtTime(2.5), static_cast<std::size_t>(-1)); // not found
}

// ---------------------------------------------------------------------------
//  Interpolation
// ---------------------------------------------------------------------------

TEST(AuxiliaryHistoryBuffer, LinearInterpolation)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(2, 5, AuxiliaryHistoryInterpolationPolicy::Linear);

    buf.push(1.0, std::vector<Real>{0.0, 10.0});
    buf.push(3.0, std::vector<Real>{4.0, 20.0});

    std::vector<Real> result(2);
    buf.interpolate(2.0, result); // midpoint

    // Linear interp: alpha = (2-1)/(3-1) = 0.5
    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 15.0);
}

TEST(AuxiliaryHistoryBuffer, LinearInterpolationAtBoundary)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 5, AuxiliaryHistoryInterpolationPolicy::Linear);

    buf.push(1.0, std::vector<Real>{10.0});
    buf.push(3.0, std::vector<Real>{30.0});

    // At exact snapshot time
    std::vector<Real> result(1);
    buf.interpolate(1.0, result);
    EXPECT_NEAR(result[0], 10.0, 1e-12);

    buf.interpolate(3.0, result);
    EXPECT_NEAR(result[0], 30.0, 1e-12);
}

TEST(AuxiliaryHistoryBuffer, FormulationDefinedInterpolation)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 5, AuxiliaryHistoryInterpolationPolicy::FormulationDefined);

    // Custom hook: quadratic interpolation (just for testing)
    buf.setInterpolationHook(
        [](Real t,
           std::span<const Real> before, Real t_before,
           std::span<const Real> after, Real t_after,
           std::span<Real> output) {
            // Just return 999.0 to prove the hook was called
            output[0] = 999.0;
        });

    buf.push(1.0, std::vector<Real>{0.0});
    buf.push(3.0, std::vector<Real>{0.0});

    std::vector<Real> result(1);
    buf.interpolate(2.0, result);
    EXPECT_DOUBLE_EQ(result[0], 999.0);
}

TEST(AuxiliaryHistoryBuffer, InterpolationThrowsWhenDisabled)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 5, AuxiliaryHistoryInterpolationPolicy::None);

    buf.push(1.0, std::vector<Real>{0.0});
    buf.push(2.0, std::vector<Real>{0.0});

    std::vector<Real> result(1);
    EXPECT_THROW(buf.interpolate(1.5, result), svmp::FE::systems::InvalidStateException);
}

// ---------------------------------------------------------------------------
//  Mutation
// ---------------------------------------------------------------------------

TEST(AuxiliaryHistoryBuffer, PopNewestRemovesLatest)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 5);

    buf.push(1.0, std::vector<Real>{10.0});
    buf.push(2.0, std::vector<Real>{20.0});
    buf.push(3.0, std::vector<Real>{30.0});
    EXPECT_EQ(buf.depth(), 3u);

    buf.popNewest();
    EXPECT_EQ(buf.depth(), 2u);
    EXPECT_DOUBLE_EQ(buf.snapshotTime(0), 2.0); // was index 1
}

TEST(AuxiliaryHistoryBuffer, ClearRemovesAll)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 5);

    buf.push(1.0, std::vector<Real>{0.0});
    buf.push(2.0, std::vector<Real>{0.0});

    buf.clear();
    EXPECT_EQ(buf.depth(), 0u);
    EXPECT_TRUE(buf.empty());
    EXPECT_TRUE(buf.isSetup()); // Setup state preserved
}

// ---------------------------------------------------------------------------
//  hasSnapshot
// ---------------------------------------------------------------------------

TEST(AuxiliaryHistoryBuffer, HasSnapshotBoundsCheck)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(1, 3);

    EXPECT_FALSE(buf.hasSnapshot(0)); // empty

    buf.push(1.0, std::vector<Real>{0.0});
    EXPECT_TRUE(buf.hasSnapshot(0));
    EXPECT_FALSE(buf.hasSnapshot(1));

    buf.push(2.0, std::vector<Real>{0.0});
    EXPECT_TRUE(buf.hasSnapshot(0));
    EXPECT_TRUE(buf.hasSnapshot(1));
    EXPECT_FALSE(buf.hasSnapshot(2));
}

// ---------------------------------------------------------------------------
//  Total storage
// ---------------------------------------------------------------------------

TEST(AuxiliaryHistoryBuffer, TotalHistoryStorageAccurate)
{
    AuxiliaryHistoryBuffer buf;
    buf.setup(3, 5);

    EXPECT_EQ(buf.totalHistoryStorage(), 0u);

    buf.push(1.0, std::vector<Real>{1.0, 2.0, 3.0});
    EXPECT_EQ(buf.totalHistoryStorage(), 3u);

    buf.push(2.0, std::vector<Real>{4.0, 5.0, 6.0});
    EXPECT_EQ(buf.totalHistoryStorage(), 6u);
}
