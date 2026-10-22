/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEConfig.h"

#if SVMP_FE_WITH_MESH

#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "LevelSet/LevelSetGeometryService.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemState.h"

#include <map>
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ls = svmp::FE::level_set;

struct HostStorage {
  std::unique_ptr<svmp::FE::interfaces::FreeSurfaceGeometrySnapshotCache>
      snapshot_cache{};
  std::map<std::string,
           std::unique_ptr<
               svmp::FE::interfaces::FreeSurfaceGeometrySnapshotReuseCache>>
      reuse_caches{};

  ls::LevelSetGeometryHost host()
  {
    return ls::LevelSetGeometryHost{
        .fe_system = nullptr,
        .primary_mesh = nullptr,
        .free_surface_geometry_snapshot_cache = snapshot_cache,
        .free_surface_geometry_snapshot_reuse_caches = reuse_caches,
    };
  }
};

} // namespace

TEST(LevelSetGeometryService, RefreshWithoutAnFESystemRejectsConfiguredRequests)
{
  HostStorage storage;
  ls::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto report = ls::refreshActiveCutIntegrationContextFromSolution(
      storage.host(), {}, {}, lifecycle, "test");
  EXPECT_FALSE(report.refreshed);
  EXPECT_EQ(report.request_policy_key, 0u);

  const std::vector<ls::ActiveCutVolumeRequest> requests(1u);
  EXPECT_THROW((void)ls::refreshActiveCutIntegrationContextFromSolution(
                   storage.host(), requests, {}, lifecycle, "test"),
               std::runtime_error);
  ls::ActiveCutContextRefreshCache cache;
  EXPECT_THROW((void)ls::refreshActiveCutIntegrationContextFromSolutionCached(
                   storage.host(), requests, {}, lifecycle, cache, "test"),
               std::runtime_error);
  EXPECT_FALSE(ls::activeCutContextRefreshSignature(
                   storage.host(), requests, std::span<const svmp::FE::Real>{})
                   .has_value());
}

TEST(LevelSetGeometryService, RestoreRetentionArmsAndReleases)
{
  ls::ActiveCutContextRefreshCache cache;
  using Pin = ls::ActiveCutContextRefreshCache::RestorePinState;
  cache.reinstall_window_open = true;
  ls::armActiveCutRestoreRetention(cache);
  EXPECT_EQ(cache.restore_pin_state, Pin::Armed);
  EXPECT_FALSE(cache.reinstall_window_open);
  EXPECT_FALSE(cache.retained_restore_context.has_value());
  ls::releaseActiveCutRestoreRetention(cache);
  EXPECT_EQ(cache.restore_pin_state, Pin::Idle);

  cache.topology_key = 7u;
  cache.evaluated_state_source_revisions[3] = 11u;
  cache.reinstall_window_open = true;
  cache.invalidateGeneratedState();
  EXPECT_FALSE(cache.topology_key.has_value());
  EXPECT_TRUE(cache.evaluated_state_source_revisions.empty());
  EXPECT_FALSE(cache.reinstall_window_open);
  EXPECT_EQ(cache.restore_pin_state, Pin::Idle);
}

TEST(LevelSetGeometryService, SignaturesCompareLayoutCountersOnlyForValueRevisions)
{
  ls::ActiveCutContextRefreshSignature a;
  a.solution_signature_kind = ls::ActiveCutContextRefreshSignature::
      SolutionSignatureKind::LevelSetFieldByteHash;
  a.solution_hash = 5u;
  auto b = a;
  b.system_dof_layout_revision += 1u;
  EXPECT_TRUE(a == b);
  a.solution_signature_kind = ls::ActiveCutContextRefreshSignature::
      SolutionSignatureKind::VectorValueRevision;
  b.solution_signature_kind = a.solution_signature_kind;
  EXPECT_FALSE(a == b);
  b.system_dof_layout_revision = a.system_dof_layout_revision;
  EXPECT_TRUE(a == b);
  b.mesh_field_layout_revision += 1u;
  EXPECT_FALSE(a == b);
}

TEST(LevelSetGeometryService, SerialCollectivesReturnTheLocalValue)
{
  const svmp::FE::systems::CollectiveCommunicator serial;
  EXPECT_EQ(svmp::FE::systems::collectiveSumDouble(2.5, serial), 2.5);
  const auto timing = svmp::FE::systems::collectiveTimingStatistics(1.25, serial);
  EXPECT_EQ(timing.local, 1.25);
  EXPECT_EQ(timing.min, 1.25);
  EXPECT_EQ(timing.mean, 1.25);
  EXPECT_EQ(timing.max, 1.25);
}

TEST(LevelSetGeometryService, ActiveSideNames)
{
  EXPECT_STREQ(ls::activeSideName(ls::LevelSetActiveSide::Negative),
               "LevelSetNegative");
  EXPECT_STREQ(ls::activeSideName(ls::LevelSetActiveSide::Positive),
               "LevelSetPositive");
}

TEST(LevelSetGeometryService, ServiceKeepsTheRunRequestsAndTracksTopology)
{
  HostStorage storage;
  std::vector<ls::ActiveCutVolumeRequest> requests(1u);
  requests.front().domain_id = "service_test";
  ls::LevelSetGeometryService service(storage.host(), requests);
  ASSERT_EQ(service.requests().size(), 1u);
  EXPECT_EQ(service.requests().front().domain_id, "service_test");
  EXPECT_EQ(service.host().fe_system, nullptr);

  // Without an FE system a state-view refresh does nothing and a
  // transaction cannot begin.
  svmp::FE::systems::SystemStateView state;
  EXPECT_FALSE(service.refreshCached(state, "test").refreshed);
  EXPECT_THROW((void)service.beginTransaction(), std::runtime_error);

  // The attempt tracker observes every refresh reported through the cache.
  service.trackTopology();
  service.topologyTracker().beginAttempt();
  ls::ActiveCutContextRefreshReport report;
  report.refreshed = true;
  report.topology_key = 5u;
  ls::observeActiveCutContextRefresh(
      service.refreshCache(), report, "before_physics_solve");
  EXPECT_EQ(service.topologyTracker().acceptedTopologyKey(),
            std::optional<std::uint64_t>(5u));
  EXPECT_FALSE(service.lastLoggedTopologyKey().has_value());
}

#endif // SVMP_FE_WITH_MESH
