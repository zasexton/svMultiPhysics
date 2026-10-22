/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetCutTopology.h"

#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <exception>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace level_set {

namespace {

// The moved bodies keep the names of the driver's communicator helpers.
// Each forwards to the FE/Systems collective that the driver's helper of the
// same name calls, so every rank issues the same MPI calls in the same order.
using svmp::FE::systems::requireCollectivePhasePreparation;

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMinMaxUint64(local, comm);
}

} // namespace

// Moved from Application/Core/ApplicationDriver.cpp with bodies, messages and
// collective order unchanged.

std::uint64_t cutContextTopologyKey(
    const svmp::FE::interfaces::LevelSetInterfaceDomain& domain) noexcept
{
  std::uint64_t h = kCutContextHashOffset;
  mixCutContextHash(h, static_cast<std::uint64_t>(domain.marker()));
  mixCutContextHash(h, domain.request().quadrature_policy_key);
  for (const auto& fragment : domain.fragments()) {
    if (!fragment.active()) {
      continue;
    }
    mixCutContextHash(h, static_cast<std::uint64_t>(fragment.parent_cell));
    mixCutContextHash(h, static_cast<std::uint64_t>(fragment.kind));
    mixCutContextHash(h, static_cast<std::uint64_t>(fragment.degeneracy));
    mixCutContextHash(h, fragment.topology_id);
  }
  for (const auto& region : domain.volumeRegions()) {
    if (!region.active()) {
      continue;
    }
    mixCutContextHash(h, static_cast<std::uint64_t>(region.parent_cell));
    mixCutContextHash(h, static_cast<std::uint64_t>(region.side));
    mixCutContextHash(h, region.full_cell_equivalent ? 1u : 0u);
    mixCutContextHash(h, region.topology_id);
  }
  return h;
}

std::uint64_t finalizeCutTopologyFingerprintMultiset(
    const CutTopologyFingerprintMultiset& digest,
    std::uint64_t contract_tag) noexcept
{
  std::uint64_t fingerprint = kCutContextHashOffset;
  mixCutContextHash(fingerprint, contract_tag);
  for (const auto word : digest.sums) {
    mixCutContextHash(fingerprint, word);
  }
  for (const auto word : digest.xors) {
    mixCutContextHash(fingerprint, word);
  }
  return fingerprint == 0u ? 1u : fingerprint;
}

CutTopologyFingerprintMultiset collectiveCutTopologyFingerprintMultiset(
    const CutTopologyFingerprintMultiset& local,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  auto global = local;
#if FE_HAS_MPI
  if (comm.isParallel()) {
#ifdef MPI_UINT64_T
    const MPI_Datatype fingerprint_type = MPI_UINT64_T;
#else
    const MPI_Datatype fingerprint_type = MPI_UNSIGNED_LONG_LONG;
#endif
    MPI_Allreduce(local.sums.data(),
                  global.sums.data(),
                  static_cast<int>(local.sums.size()),
                  fingerprint_type,
                  MPI_SUM,
                  comm.native());
    MPI_Allreduce(local.xors.data(),
                  global.xors.data(),
                  static_cast<int>(local.xors.size()),
                  fingerprint_type,
                  MPI_BXOR,
                  comm.native());
  }
#else
  if (comm.isParallel()) {
    throw std::runtime_error(
        "Cut-topology multiset fingerprinting requires MPI in a parallel run.");
  }
#endif
  return global;
}

std::uint64_t collectivePartitionIndependentCutTopologyFingerprint(
    std::span<const std::uint64_t> request_identities,
    std::span<const std::uint64_t> local_owned_rule_fingerprints,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  CutTopologyFingerprintMultiset request_digest;
  for (const auto identity : request_identities) {
    request_digest.insert(identity);
  }
  constexpr std::uint64_t request_contract_tag =
      0x4355545245515331ull; // "CUTREQS1"
  const auto request_set_fingerprint =
      finalizeCutTopologyFingerprintMultiset(
          request_digest, request_contract_tag);
  const auto [minimum_request_set, maximum_request_set] =
      globalMinMaxUint64(request_set_fingerprint, comm);
  if (minimum_request_set != maximum_request_set) {
    throw std::runtime_error(
        "Active-cut topology request identities differ across the FE communicator.");
  }

  CutTopologyFingerprintMultiset local_rule_digest;
  for (const auto fingerprint : local_owned_rule_fingerprints) {
    local_rule_digest.insert(fingerprint);
  }
  const auto global_rule_digest =
      collectiveCutTopologyFingerprintMultiset(local_rule_digest, comm);

  std::uint64_t fingerprint = kCutContextHashOffset;
  constexpr std::uint64_t topology_contract_tag =
      0x41435443544f5031ull; // "ACTCTOP1"
  mixCutContextHash(fingerprint, topology_contract_tag);
  mixCutContextHash(fingerprint, request_set_fingerprint);
  for (const auto word : global_rule_digest.sums) {
    mixCutContextHash(fingerprint, word);
  }
  for (const auto word : global_rule_digest.xors) {
    mixCutContextHash(fingerprint, word);
  }
  return fingerprint == 0u ? 1u : fingerprint;
}

std::uint64_t activeCutTopologyRequestIdentity(
    const ActiveCutVolumeRequest& request,
    const svmp::FE::interfaces::FreeSurfaceGeometryRevision& revision)
{
  if (revision.interface_marker < 0 ||
      revision.domain_id != request.domain_id ||
      revision.isovalue != static_cast<svmp::FE::Real>(request.isovalue) ||
      (request.requested_interface_marker >= 0 &&
       revision.interface_marker != request.requested_interface_marker)) {
    throw std::runtime_error(
        "Active-cut topology fingerprint received a snapshot that does not match its request domain.");
  }

  std::uint64_t identity = kCutContextHashOffset;
  constexpr std::uint64_t request_identity_tag =
      0x4355545245514931ull; // "CUTREQI1"
  mixCutContextHash(identity, request_identity_tag);
  const std::vector<ActiveCutVolumeRequest> single_request{request};
  mixCutContextHash(
      identity,
      activeCutVolumeRequestPolicyKey(single_request));
  mixCutContextHash(identity, request.level_set_field_name);
  mixCutContextHash(identity, request.domain_id);
  mixCutContextHash(identity, revision.source_id);
  mixCutContextHash(identity, revision.domain_id);
  mixCutContextHash(
      identity,
      static_cast<std::uint64_t>(
          static_cast<std::int64_t>(revision.interface_marker)));
  mixCutContextHash(identity, revision.quadrature_policy_key);
  return identity == 0u ? 1u : identity;
}

std::uint64_t freeSurfaceRuleSemanticTopologyFingerprint(
    const svmp::FE::interfaces::FreeSurfaceGeometryRuleRecord& rule,
    std::uint64_t request_identity)
{
  const auto& reference = rule.reference_rule;
  const auto& provenance = reference.provenance;
  if (provenance.parent_entity_global_id ==
          svmp::FE::INVALID_GLOBAL_INDEX ||
      rule.source_topology_key == 0u) {
    throw std::runtime_error(
        "Active-cut topology fingerprint requires canonical global rule identities and an authoritative epoch-free source topology key.");
  }

  std::uint64_t fingerprint = kCutContextHashOffset;
  constexpr std::uint64_t rule_identity_tag =
      0x43555452554c4531ull; // "CUTRULE1"
  mixCutContextHash(fingerprint, rule_identity_tag);
  mixCutContextHash(fingerprint, request_identity);
  mixCutContextHash(
      fingerprint, static_cast<std::uint64_t>(rule.role));
  mixCutContextHash(
      fingerprint, static_cast<std::uint64_t>(rule.retention));
  mixCutContextHash(
      fingerprint,
      static_cast<std::uint64_t>(
          static_cast<std::int64_t>(rule.physical_boundary_marker)));
  mixCutContextHash(
      fingerprint,
      static_cast<std::uint64_t>(
          static_cast<std::int64_t>(provenance.marker)));
  mixCutContextHash(
      fingerprint,
      static_cast<std::uint64_t>(provenance.parent_entity_global_id));
  mixCutContextHash(
      fingerprint,
      static_cast<std::uint64_t>(
          provenance.parent_boundary_entity_global_id));
  mixCutContextHash(
      fingerprint, static_cast<std::uint64_t>(reference.kind));
  mixCutContextHash(
      fingerprint, static_cast<std::uint64_t>(reference.side));
  mixCutContextHash(
      fingerprint,
      static_cast<std::uint64_t>(
          static_cast<std::int64_t>(reference.geometric_dimension)));
  mixCutContextHash(
      fingerprint, reference.full_cell_equivalent ? 1u : 0u);
  mixCutContextHash(fingerprint, rule.source_topology_key);

  // Deliberately exclude rank/local addressing, point coordinates and
  // measures, snapshot/source/mesh epochs, and generated stable IDs.  Despite
  // their names, cut_topology_revision, source_fragment_stable_ids, and
  // component_id all incorporate the source value revision.  Generated
  // topology_id/branch labels are also excluded: high-order backends derive
  // them from traversal counters and local fragment ordering rather than a
  // canonical connectivity identity.
  return fingerprint == 0u ? 1u : fingerprint;
}

std::uint64_t activeCutContextTopologyFingerprint(
    std::span<const ActiveCutTopologySnapshotBinding> bindings,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  std::vector<std::uint64_t> request_identities;
  std::vector<std::uint64_t> local_owned_rule_fingerprints;
  std::exception_ptr local_preparation_failure;
  try {
    request_identities.reserve(bindings.size());
    for (const auto& binding : bindings) {
      if (binding.request == nullptr || binding.snapshot == nullptr) {
        throw std::runtime_error(
            "Active-cut topology fingerprint received an incomplete request/snapshot binding.");
      }
      const auto request_identity = activeCutTopologyRequestIdentity(
          *binding.request, binding.snapshot->revision());
      if (request_identity == 0u) {
        throw std::runtime_error(
            "Active-cut topology fingerprint produced a zero request identity.");
      }
      request_identities.push_back(request_identity);
      for (const auto& rule : binding.snapshot->rules()) {
        if (!rule.locally_owned) {
          continue;
        }
        const auto rule_fingerprint =
            freeSurfaceRuleSemanticTopologyFingerprint(
                rule, request_identity);
        if (rule_fingerprint == 0u) {
          throw std::runtime_error(
              "Active-cut topology fingerprint produced a zero owned-rule identity.");
        }
        local_owned_rule_fingerprints.push_back(rule_fingerprint);
      }
    }
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_preparation_failure,
      comm,
      "active_cut_topology_fingerprint");

  return collectivePartitionIndependentCutTopologyFingerprint(
      request_identities, local_owned_rule_fingerprints, comm);
}

bool transientCutTopologyFingerprintSupportsRequests(
    std::span<const ActiveCutVolumeRequest> requests) noexcept
{
  using Mode =
      svmp::FE::level_set::GeneratedInterfaceGeometryMode;
  return std::all_of(
      requests.begin(), requests.end(), [](const auto& request) {
        return request.geometry_mode == Mode::LinearCorner;
      });
}

} // namespace level_set
} // namespace FE
} // namespace svmp
