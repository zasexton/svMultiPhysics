/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetCutTopology.h
 * @brief Topology identity of generated cut geometry, and the cut-topology
 *        protocol of one time-step attempt
 *
 * - ActiveCutContextRefreshReport: what a refresh of the generated cut
 *   integration context produced (topology key, policy key, source value
 *   revisions, communicator-wide counts and measures).
 * - The partition-independent topology fingerprint of the generated rules
 *   (a probabilistic multiset digest over owned rules, qualified for
 *   LinearCorner cuts) and its FNV-style hash helpers.
 * - TransientCutTopologyAttemptTracker: the epochs of one attempt.
 *
 * Moved from Application/Core/ApplicationDriver.cpp with bodies, messages and
 * collective order unchanged.  The collectives take the FE system's active
 * communicator as a systems::CollectiveCommunicator.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Systems/SystemCollectives.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class CutIntegrationContext;
} // namespace assembly

namespace interfaces {
class FreeSurfaceGeometrySnapshot;
class LevelSetInterfaceDomain;
struct FreeSurfaceGeometryRevision;
struct FreeSurfaceGeometryRuleRecord;
} // namespace interfaces

namespace level_set {

// ---------------------------------------------------------------------------
// Refresh report and attempt tracker
// ---------------------------------------------------------------------------

struct ActiveCutContextRefreshReport {
  bool refreshed{false};
  // Set by a cached refresh that kept the installed context because its
  // content signature proves that it was generated from the same input.
  bool reused_installed_context{false};
  std::uint64_t topology_key{0};
  std::uint64_t request_policy_key{0};
  std::uint64_t value_revision{0};
  std::map<int, std::uint64_t> evaluated_state_source_revisions{};
  // Non-owning handle to the context installed by a rebuild, so that a
  // refresh cache can retain it for restored states without keeping every
  // report's context alive.
  std::weak_ptr<const svmp::FE::assembly::CutIntegrationContext>
      installed_context{};
  std::size_t cell_count{0};
  std::size_t corner_linearized_cell_count{0};
  std::size_t interface_fragments{0};
  std::size_t active_volume_regions{0};
  std::size_t active_cut_cells{0};
  std::size_t active_quadrature_points{0};
  std::size_t domain_interface_quadrature_point_count{0};
  std::size_t domain_volume_quadrature_point_count{0};
  std::size_t domain_total_quadrature_point_count{0};
  std::size_t backend_volume_quadrature_point_count{0};
  std::size_t backend_interface_quadrature_point_count{0};
  std::size_t cut_adjacent_facets{0};
  std::size_t basis_cache_entries{0};
  // Cache/refresh counters and backend elapsed time are summed rank-local work
  // telemetry and deliberately include generated ghost-cell work.  Physical
  // domain counts and measures above are owned-cell global totals.
  std::size_t generated_cell_cache_hits{0};
  std::size_t generated_cell_cache_misses{0};
  std::size_t generated_cell_cache_unchanged_dof_hits{0};
  std::size_t generated_cell_refresh_candidates{0};
  std::size_t generated_cell_directly_affected{0};
  std::size_t generated_cell_affected_neighborhood{0};
  std::size_t generated_domain_cache_hits{0};
  std::size_t linear_full_cell_fast_path_count{0};
  double backend_elapsed_seconds{0.0};
  long process_vm_kb{-1};
  long process_rss_kb{-1};
  svmp::FE::Real negative_volume{0.0};
  svmp::FE::Real positive_volume{0.0};
  svmp::FE::Real negative_physical_volume{0.0};
  svmp::FE::Real positive_physical_volume{0.0};
};

// Tracks the cut topology of one time-step attempt.  A topology change is a
// normal event for a moving interface: the nonlinear solve adopts a new
// attempt epoch through acknowledgeNonlinearRestart whenever an outer
// fixed-point refresh (including the canonicalization of the predicted entry
// state) regenerates a different topology, and the converged state is valid
// in the epoch it was solved in.  An attempt is tainted only by a topology
// that the solve did not acknowledge.
//
// For a stage scheme (generalized-alpha) the nonlinear solve converges at the
// operator stage, and the time loop then finalizes the endpoint
// (endpoint_candidate refresh).  From that point the endpoint may lie in a
// different topology than the solved stage, for example when the interface
// passes a vertex between the stage and the endpoint.  Such an endpoint is
// accepted and becomes the committed topology of the next step; observations
// after the endpoint refresh only have to be self-consistent.
//
// When an outer refresh returns to an epoch already visited in the attempt,
// the fixed point alternates across a nonsmooth switching surface (a
// level-set value at a vertex whose sign flips with the topology it was
// solved in).  The solve then finishes on that frozen epoch
// (acknowledgeFrozenEpoch), and the candidate is accepted although its
// regenerated topology lies on the other side; the next step starts there.
class TransientCutTopologyAttemptTracker {
public:
  void beginAttempt() noexcept
  {
    attempt_active_ = true;
    attempt_tainted_ = false;
    final_candidate_observed_ = false;
    endpoint_phase_ = false;
    frozen_epoch_ = false;
    attempt_topology_key_ = accepted_topology_key_;
    visited_epochs_.clear();
    if (accepted_topology_key_.has_value()) {
      visited_epochs_.push_back(*accepted_topology_key_);
    }
    last_observed_topology_key_.reset();
    first_mismatched_topology_key_.reset();
    first_mismatch_provenance_size_ = 0u;
  }

  void discardAttempt() noexcept
  {
    attempt_active_ = false;
  }

  void observe(
      const ActiveCutContextRefreshReport& report,
      std::string_view provenance) noexcept
  {
    if (!attempt_active_) {
      return;
    }

    const auto normalized_provenance =
        provenance.empty() ? std::string_view{"unknown"} : provenance;
    last_observed_topology_key_ = report.topology_key;
    if (normalized_provenance == "final_candidate_topology_gate") {
      final_candidate_observed_ = true;
    }
    if (normalized_provenance == "endpoint_candidate") {
      endpoint_phase_ = true;
    }
    if ((endpoint_phase_ || frozen_epoch_) &&
        accepted_topology_key_.has_value()) {
      if (report.topology_key == 0u) {
        taint(report.topology_key, normalized_provenance);
      }
      return;
    }

    if (!accepted_topology_key_.has_value()) {
      if (normalized_provenance == "before_physics_solve" &&
          report.topology_key != 0u) {
        accepted_topology_key_ = report.topology_key;
        attempt_topology_key_ = report.topology_key;
        visited_epochs_.assign(1u, report.topology_key);
        return;
      }
      taint(report.topology_key, normalized_provenance);
      return;
    }

    if (!attempt_topology_key_.has_value() ||
        *attempt_topology_key_ == 0u || report.topology_key == 0u ||
        report.topology_key != *attempt_topology_key_) {
      taint(report.topology_key, normalized_provenance);
    }
  }

  void acknowledgeNonlinearRestart()
  {
    if (!attempt_active_ ||
        !last_observed_topology_key_.has_value() ||
        *last_observed_topology_key_ == 0u) {
      throw std::logic_error(
          "[svMultiPhysics::Application] A transient cut-topology nonlinear restart lacked a nonzero observed attempt epoch.");
    }
    attempt_topology_key_ = last_observed_topology_key_;
    attempt_tainted_ = false;
    final_candidate_observed_ = false;
    if (std::find(visited_epochs_.begin(),
                  visited_epochs_.end(),
                  *attempt_topology_key_) == visited_epochs_.end()) {
      visited_epochs_.push_back(*attempt_topology_key_);
    }
  }

  // True when the latest refresh left the attempt epoch for a topology that
  // already served as an epoch of this attempt.
  [[nodiscard]] bool revisitPending() const noexcept
  {
    return attempt_active_ && attempt_tainted_ &&
           last_observed_topology_key_.has_value() &&
           *last_observed_topology_key_ != 0u &&
           std::find(visited_epochs_.begin(),
                     visited_epochs_.end(),
                     *last_observed_topology_key_) != visited_epochs_.end();
  }

  void acknowledgeFrozenEpoch()
  {
    if (!attempt_active_ || attempt_tainted_) {
      throw std::logic_error(
          "[svMultiPhysics::Application] A transient cut-topology frozen epoch was acknowledged outside a consistent attempt.");
    }
    frozen_epoch_ = true;
  }

  void requireAcceptedBaseline() const
  {
    if (!accepted_topology_key_.has_value() ||
        *accepted_topology_key_ == 0u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The transient cut-topology acceptance baseline was not seeded by before_physics_solve.");
    }
  }

  [[nodiscard]] bool candidateMustReject(
      const std::optional<std::uint64_t>& cached_topology_key) const noexcept
  {
    const bool inconsistent =
        !attempt_active_ || !accepted_topology_key_.has_value() ||
        *accepted_topology_key_ == 0u ||
        !attempt_topology_key_.has_value() ||
        *attempt_topology_key_ == 0u || attempt_tainted_ ||
        !final_candidate_observed_ ||
        !last_observed_topology_key_.has_value() ||
        !cached_topology_key.has_value() ||
        *cached_topology_key == 0u ||
        *last_observed_topology_key_ != *cached_topology_key;
    if (inconsistent) {
      return true;
    }
    // A solved-state scheme (backward Euler) accepts exactly the state whose
    // topology the solve converged in, unless the solve finished on a frozen
    // epoch after a cycle.  A stage scheme may finalize its endpoint in a new
    // topology (see the class comment).
    return !endpoint_phase_ && !frozen_epoch_ &&
           *cached_topology_key != *attempt_topology_key_;
  }

  void completeAttempt(
      const std::optional<std::uint64_t>& cached_topology_key)
  {
    if (candidateMustReject(cached_topology_key)) {
      throw std::logic_error(
          "[svMultiPhysics::Application] A transient cut-topology candidate reached commit-ready without a fixed accepted topology.");
    }
    accepted_topology_key_ = *cached_topology_key;
    attempt_topology_key_ = accepted_topology_key_;
    attempt_active_ = false;
  }

  [[nodiscard]] bool attemptActive() const noexcept
  {
    return attempt_active_;
  }

  [[nodiscard]] bool attemptTainted() const noexcept
  {
    return attempt_tainted_;
  }

  [[nodiscard]] bool endpointPhase() const noexcept
  {
    return endpoint_phase_;
  }

  [[nodiscard]] bool frozenEpoch() const noexcept
  {
    return frozen_epoch_;
  }

  [[nodiscard]] const std::optional<std::uint64_t>&
  acceptedTopologyKey() const noexcept
  {
    return accepted_topology_key_;
  }

  [[nodiscard]] const std::optional<std::uint64_t>&
  attemptTopologyKey() const noexcept
  {
    return attempt_topology_key_;
  }

  [[nodiscard]] const std::optional<std::uint64_t>&
  lastObservedTopologyKey() const noexcept
  {
    return last_observed_topology_key_;
  }

  [[nodiscard]] const std::optional<std::uint64_t>&
  firstMismatchedTopologyKey() const noexcept
  {
    return first_mismatched_topology_key_;
  }

  [[nodiscard]] std::string_view firstMismatchProvenance() const noexcept
  {
    return std::string_view{
        first_mismatch_provenance_.data(),
        first_mismatch_provenance_size_};
  }

private:
  void taint(
      std::uint64_t topology_key,
      std::string_view provenance) noexcept
  {
    attempt_tainted_ = true;
    if (!first_mismatched_topology_key_.has_value()) {
      first_mismatched_topology_key_ = topology_key;
      // Provenance is diagnostic only. Production callers use static tokens,
      // but copying into fixed storage also makes arbitrary caller-owned views
      // safe without allocating inside a communicator-collective refresh.
      first_mismatch_provenance_size_ = std::min(
          provenance.size(), first_mismatch_provenance_.size());
      if (first_mismatch_provenance_size_ != 0u) {
        std::copy_n(
            provenance.data(),
            first_mismatch_provenance_size_,
            first_mismatch_provenance_.data());
      }
    }
  }

  std::optional<std::uint64_t> accepted_topology_key_{};
  std::optional<std::uint64_t> attempt_topology_key_{};
  std::optional<std::uint64_t> last_observed_topology_key_{};
  std::optional<std::uint64_t> first_mismatched_topology_key_{};
  std::array<char, 128u> first_mismatch_provenance_{};
  std::size_t first_mismatch_provenance_size_{0u};
  bool attempt_active_{false};
  bool attempt_tainted_{false};
  bool final_candidate_observed_{false};
  bool endpoint_phase_{false};
  bool frozen_epoch_{false};
  std::vector<std::uint64_t> visited_epochs_{};
};

// ---------------------------------------------------------------------------
// Topology keys and fingerprints
// ---------------------------------------------------------------------------

inline constexpr std::uint64_t kCutContextHashOffset = 1469598103934665603ull;
inline constexpr std::uint64_t kCutContextHashPrime = 1099511628211ull;

inline void mixCutContextHash(std::uint64_t& h, std::uint64_t value) noexcept
{
  h ^= value;
  h *= kCutContextHashPrime;
}

inline void mixCutContextHash(std::uint64_t& h, const std::string& value) noexcept
{
  for (const char c : value) {
    mixCutContextHash(h, static_cast<unsigned char>(c));
  }
  mixCutContextHash(h, 0xffu);
}

/// Rank-local key of the active fragments and volume regions of @p domain.
[[nodiscard]] std::uint64_t cutContextTopologyKey(
    const svmp::FE::interfaces::LevelSetInterfaceDomain& domain) noexcept;

inline std::uint64_t avalancheCutTopologyFingerprint(
    std::uint64_t value) noexcept
{
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

struct CutTopologyFingerprintMultiset {
  // This is a bounded probabilistic fingerprint, not an exact set-equality
  // certificate.  Count plus two independently mixed sums and xors makes the
  // reduction insensitive to rule/rank order while retaining multiplicity.
  std::array<std::uint64_t, 3> sums{{0u, 0u, 0u}};
  std::array<std::uint64_t, 2> xors{{0u, 0u}};

  void insert(std::uint64_t fingerprint) noexcept
  {
    fingerprint = fingerprint == 0u ? 1u : fingerprint;
    const auto mixed = avalancheCutTopologyFingerprint(fingerprint);
    ++sums[0];
    sums[1] += fingerprint;
    sums[2] += mixed;
    xors[0] ^= fingerprint;
    xors[1] ^= mixed;
  }

  void merge(const CutTopologyFingerprintMultiset& other) noexcept
  {
    for (std::size_t i = 0u; i < sums.size(); ++i) {
      sums[i] += other.sums[i];
    }
    for (std::size_t i = 0u; i < xors.size(); ++i) {
      xors[i] ^= other.xors[i];
    }
  }
};

[[nodiscard]] std::uint64_t finalizeCutTopologyFingerprintMultiset(
    const CutTopologyFingerprintMultiset& digest,
    std::uint64_t contract_tag) noexcept;

/// Communicator-wide sums and xors of @p local.  Collective.
[[nodiscard]] CutTopologyFingerprintMultiset collectiveCutTopologyFingerprintMultiset(
    const CutTopologyFingerprintMultiset& local,
    const svmp::FE::systems::CollectiveCommunicator& comm);

/// Fingerprint of the request identities (equal on every rank, else throws)
/// and of the owned rule fingerprints of all ranks.  Collective.
[[nodiscard]] std::uint64_t collectivePartitionIndependentCutTopologyFingerprint(
    std::span<const std::uint64_t> request_identities,
    std::span<const std::uint64_t> local_owned_rule_fingerprints,
    const svmp::FE::systems::CollectiveCommunicator& comm);

/// Epoch-free identity of a cut-volume request and its snapshot revision.
[[nodiscard]] std::uint64_t activeCutTopologyRequestIdentity(
    const ActiveCutVolumeRequest& request,
    const svmp::FE::interfaces::FreeSurfaceGeometryRevision& revision);

/// Partition- and epoch-independent identity of one snapshot rule.
[[nodiscard]] std::uint64_t freeSurfaceRuleSemanticTopologyFingerprint(
    const svmp::FE::interfaces::FreeSurfaceGeometryRuleRecord& rule,
    std::uint64_t request_identity);

struct ActiveCutTopologySnapshotBinding {
  const ActiveCutVolumeRequest* request{nullptr};
  const svmp::FE::interfaces::FreeSurfaceGeometrySnapshot* snapshot{nullptr};
};

/// Topology fingerprint of the owned rules of @p bindings.  Collective.
[[nodiscard]] std::uint64_t activeCutContextTopologyFingerprint(
    std::span<const ActiveCutTopologySnapshotBinding> bindings,
    const svmp::FE::systems::CollectiveCommunicator& comm);

/// True when every request generates LinearCorner geometry, the mode for
/// which the fingerprint is qualified.
[[nodiscard]] bool transientCutTopologyFingerprintSupportsRequests(
    std::span<const ActiveCutVolumeRequest> requests) noexcept;

} // namespace level_set
} // namespace FE
} // namespace svmp
