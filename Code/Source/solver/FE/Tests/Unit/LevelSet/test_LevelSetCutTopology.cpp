/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "LevelSet/LevelSetCutTopology.h"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

namespace ls = svmp::FE::level_set;

ls::ActiveCutContextRefreshReport reportWithKey(std::uint64_t key)
{
  ls::ActiveCutContextRefreshReport report;
  report.refreshed = true;
  report.topology_key = key;
  return report;
}

} // namespace

TEST(LevelSetCutTopologyTracker, FirstBeforeSolveRefreshSeedsTheAcceptedBaseline)
{
  ls::TransientCutTopologyAttemptTracker tracker;
  EXPECT_THROW(tracker.requireAcceptedBaseline(), std::runtime_error);
  tracker.beginAttempt();
  tracker.observe(reportWithKey(11u), "before_physics_solve");
  EXPECT_NO_THROW(tracker.requireAcceptedBaseline());
  EXPECT_EQ(tracker.acceptedTopologyKey(), std::optional<std::uint64_t>(11u));
  EXPECT_EQ(tracker.attemptTopologyKey(), std::optional<std::uint64_t>(11u));
  EXPECT_FALSE(tracker.attemptTainted());
}

TEST(LevelSetCutTopologyTracker, UnacknowledgedTopologyChangeRejectsTheCandidate)
{
  ls::TransientCutTopologyAttemptTracker tracker;
  tracker.beginAttempt();
  tracker.observe(reportWithKey(11u), "before_physics_solve");
  tracker.observe(reportWithKey(12u), "outer_fixed_point");
  EXPECT_TRUE(tracker.attemptTainted());
  EXPECT_EQ(tracker.firstMismatchedTopologyKey(),
            std::optional<std::uint64_t>(12u));
  EXPECT_EQ(tracker.firstMismatchProvenance(), "outer_fixed_point");
  tracker.observe(reportWithKey(12u), "final_candidate_topology_gate");
  EXPECT_TRUE(tracker.candidateMustReject(std::uint64_t{12u}));
  EXPECT_THROW(tracker.completeAttempt(std::uint64_t{12u}), std::logic_error);
}

TEST(LevelSetCutTopologyTracker, AcknowledgedRestartAdoptsTheNewEpoch)
{
  ls::TransientCutTopologyAttemptTracker tracker;
  tracker.beginAttempt();
  tracker.observe(reportWithKey(11u), "before_physics_solve");
  tracker.observe(reportWithKey(12u), "outer_fixed_point");
  tracker.acknowledgeNonlinearRestart();
  EXPECT_FALSE(tracker.attemptTainted());
  EXPECT_EQ(tracker.attemptTopologyKey(), std::optional<std::uint64_t>(12u));
  tracker.observe(reportWithKey(12u), "final_candidate_topology_gate");
  EXPECT_FALSE(tracker.candidateMustReject(std::uint64_t{12u}));
  tracker.completeAttempt(std::uint64_t{12u});
  EXPECT_EQ(tracker.acceptedTopologyKey(), std::optional<std::uint64_t>(12u));
  EXPECT_FALSE(tracker.attemptActive());
}

TEST(LevelSetCutTopologyTracker, ReturnToAVisitedEpochIsARevisit)
{
  ls::TransientCutTopologyAttemptTracker tracker;
  tracker.beginAttempt();
  tracker.observe(reportWithKey(11u), "before_physics_solve");
  tracker.observe(reportWithKey(12u), "outer_fixed_point");
  EXPECT_FALSE(tracker.revisitPending());
  tracker.acknowledgeNonlinearRestart();
  tracker.observe(reportWithKey(11u), "outer_fixed_point");
  EXPECT_TRUE(tracker.revisitPending());
  // A frozen epoch can only be acknowledged in a consistent attempt.
  EXPECT_THROW(tracker.acknowledgeFrozenEpoch(), std::logic_error);
  tracker.acknowledgeNonlinearRestart();
  tracker.acknowledgeFrozenEpoch();
  EXPECT_TRUE(tracker.frozenEpoch());
  // On a frozen epoch the regenerated topology may differ.
  tracker.observe(reportWithKey(12u), "final_candidate_topology_gate");
  EXPECT_FALSE(tracker.candidateMustReject(std::uint64_t{12u}));
}

TEST(LevelSetCutTopologyTracker, EndpointPhaseMayFinalizeInANewTopology)
{
  ls::TransientCutTopologyAttemptTracker tracker;
  tracker.beginAttempt();
  tracker.observe(reportWithKey(11u), "before_physics_solve");
  tracker.observe(reportWithKey(13u), "endpoint_candidate");
  EXPECT_TRUE(tracker.endpointPhase());
  EXPECT_FALSE(tracker.attemptTainted());
  tracker.observe(reportWithKey(13u), "final_candidate_topology_gate");
  EXPECT_FALSE(tracker.candidateMustReject(std::uint64_t{13u}));
  // A zero key is never a valid topology.
  tracker.observe(reportWithKey(0u), "final_candidate_topology_gate");
  EXPECT_TRUE(tracker.attemptTainted());
}

TEST(LevelSetCutTopologyFingerprint, MultisetIsOrderIndependentAndKeepsMultiplicity)
{
  const std::array<std::uint64_t, 4> words{3u, 0u, 17u, 3u};
  ls::CutTopologyFingerprintMultiset forward;
  ls::CutTopologyFingerprintMultiset backward;
  for (const auto word : words) {
    forward.insert(word);
  }
  for (auto it = words.rbegin(); it != words.rend(); ++it) {
    backward.insert(*it);
  }
  EXPECT_EQ(forward.sums, backward.sums);
  EXPECT_EQ(forward.xors, backward.xors);
  constexpr std::uint64_t tag = 0x5445535431ull;
  EXPECT_EQ(ls::finalizeCutTopologyFingerprintMultiset(forward, tag),
            ls::finalizeCutTopologyFingerprintMultiset(backward, tag));

  // Zero is inserted as one; a repeated word changes the digest.
  ls::CutTopologyFingerprintMultiset zero;
  zero.insert(0u);
  ls::CutTopologyFingerprintMultiset one;
  one.insert(1u);
  EXPECT_EQ(zero.sums, one.sums);
  ls::CutTopologyFingerprintMultiset once;
  once.insert(3u);
  ls::CutTopologyFingerprintMultiset twice;
  twice.insert(3u);
  twice.insert(3u);
  EXPECT_NE(ls::finalizeCutTopologyFingerprintMultiset(once, tag),
            ls::finalizeCutTopologyFingerprintMultiset(twice, tag));

  // merge() equals inserting the other's words.
  ls::CutTopologyFingerprintMultiset merged = once;
  merged.merge(once);
  EXPECT_EQ(merged.sums, twice.sums);
  EXPECT_EQ(merged.xors, twice.xors);
}

TEST(LevelSetCutTopologyFingerprint, SerialFingerprintIgnoresRuleOrder)
{
  const svmp::FE::systems::CollectiveCommunicator serial;
  const std::array<std::uint64_t, 2> requests{101u, 202u};
  const std::array<std::uint64_t, 3> rules_a{5u, 6u, 7u};
  const std::array<std::uint64_t, 3> rules_b{7u, 5u, 6u};
  const auto fingerprint =
      ls::collectivePartitionIndependentCutTopologyFingerprint(
          requests, rules_a, serial);
  EXPECT_NE(fingerprint, 0u);
  EXPECT_EQ(ls::collectivePartitionIndependentCutTopologyFingerprint(
                requests, rules_b, serial),
            fingerprint);
  const std::array<std::uint64_t, 2> other_rules{5u, 6u};
  EXPECT_NE(ls::collectivePartitionIndependentCutTopologyFingerprint(
                requests, other_rules, serial),
            fingerprint);
  const auto global = ls::collectiveCutTopologyFingerprintMultiset(
      ls::CutTopologyFingerprintMultiset{}, serial);
  EXPECT_EQ(global.sums[0], 0u);
}

TEST(LevelSetCutTopologyFingerprint, IncompleteBindingsFailOnEveryRank)
{
  const svmp::FE::systems::CollectiveCommunicator serial;
  const std::array<ls::ActiveCutTopologySnapshotBinding, 1> bindings{{}};
  EXPECT_THROW((void)ls::activeCutContextTopologyFingerprint(bindings, serial),
               std::runtime_error);
}

TEST(LevelSetCutTopologyFingerprint, QualifiedOnlyForLinearCornerRequests)
{
  std::vector<ls::ActiveCutVolumeRequest> requests(2u);
  EXPECT_TRUE(ls::transientCutTopologyFingerprintSupportsRequests(requests));
  requests[1].geometry_mode =
      ls::GeneratedInterfaceGeometryMode::HighOrderImplicit;
  EXPECT_FALSE(ls::transientCutTopologyFingerprintSupportsRequests(requests));
  EXPECT_TRUE(ls::transientCutTopologyFingerprintSupportsRequests({}));
}

TEST(LevelSetCutTopologyFingerprint, HashMixersAreFnv1a)
{
  std::uint64_t h = ls::kCutContextHashOffset;
  ls::mixCutContextHash(h, std::uint64_t{0x61u});
  EXPECT_EQ(h, (ls::kCutContextHashOffset ^ 0x61u) * ls::kCutContextHashPrime);
  std::uint64_t s = ls::kCutContextHashOffset;
  ls::mixCutContextHash(s, std::string("a"));
  std::uint64_t expected = ls::kCutContextHashOffset;
  ls::mixCutContextHash(expected, std::uint64_t{0x61u});
  ls::mixCutContextHash(expected, std::uint64_t{0xffu});
  EXPECT_EQ(s, expected);
  EXPECT_EQ(ls::avalancheCutTopologyFingerprint(0u), 0u);
  EXPECT_NE(ls::avalancheCutTopologyFingerprint(1u), 1u);
}
