/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetCutTopology.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Systems/FESystem.h"

#include <cmath>
#include <exception>
#include <set>
#include <tuple>

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

// The reference facets (hyperplanes) a parent coordinate lies on, as a bit
// mask: for a tensor cell on [-1, 1]^d, bit 2a for x_a = -1 and 2a + 1 for
// x_a = 1; for a simplex, bit a for x_a = 0 and bit d for sum x_a = 1.
std::uint64_t referenceFacetMask(svmp::FE::ElementType type,
                                 const std::array<svmp::FE::Real, 3>& x)
{
  using svmp::FE::ElementType;
  constexpr svmp::FE::Real tolerance = 1.0e-10;
  const auto near = [](svmp::FE::Real a, svmp::FE::Real b) {
    return std::abs(a - b) <= tolerance;
  };
  std::uint64_t mask = 0u;
  switch (type) {
  case ElementType::Quad4:
  case ElementType::Quad8:
  case ElementType::Quad9:
  case ElementType::Hex8:
  case ElementType::Hex20:
  case ElementType::Hex27: {
    const int dim = svmp::FE::element_dimension(type);
    for (int a = 0; a < dim; ++a) {
      const auto c = x[static_cast<std::size_t>(a)];
      mask |= near(c, -1.0) ? (std::uint64_t{1} << (2 * a)) : 0u;
      mask |= near(c, 1.0) ? (std::uint64_t{1} << (2 * a + 1)) : 0u;
    }
    return mask;
  }
  case ElementType::Triangle3:
  case ElementType::Triangle6:
  case ElementType::Tetra4:
  case ElementType::Tetra10: {
    const int dim = svmp::FE::element_dimension(type);
    svmp::FE::Real sum = 0.0;
    for (int a = 0; a < dim; ++a) {
      const auto c = x[static_cast<std::size_t>(a)];
      mask |= near(c, 0.0) ? (std::uint64_t{1} << a) : 0u;
      sum += c;
    }
    mask |= near(sum, 1.0) ? (std::uint64_t{1} << dim) : 0u;
    return mask;
  }
  default:
    throw std::runtime_error(
        "The cut-structure key supports triangles, quadrilaterals, tetrahedra "
        "and hexahedra.");
  }
}

} // namespace

std::map<svmp::FE::GlobalIndex, std::uint64_t> cutStructureCellKeys(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::interfaces::LevelSetInterfaceDomain& domain,
    svmp::FE::FieldId level_set_field,
    std::span<const svmp::FE::Real> solution)
{
  const auto& mesh = system.meshAccess();
  const auto& dofs = system.fieldDofHandler(level_set_field).getDofMap();
  const auto offset =
      static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
  const auto isovalue = domain.request().isovalue;
  const auto band = domain.request().resolvedCoefficientClassificationBand();

  // The parent-boundary facets of the fragment vertices, a set per cell.
  std::map<svmp::FE::GlobalIndex, std::set<std::uint64_t>> facets;
  for (const auto& fragment : domain.fragments()) {
    if (!fragment.active() || fragment.parent_cell < 0 ||
        !mesh.isOwnedCell(fragment.parent_cell)) {
      continue;
    }
    const auto type = mesh.getCellType(fragment.parent_cell);
    auto& cell_facets =
        facets[mesh.getCellGlobalId(fragment.parent_cell)];
    for (const auto& vertex : fragment.vertices) {
      const auto mask = referenceFacetMask(type, vertex.parent_coordinate);
      if (mask != 0u) {
        cell_facets.insert(mask);
      }
    }
  }

  std::map<svmp::FE::GlobalIndex, std::uint64_t> keys;
  for (svmp::FE::GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
    if (!mesh.isOwnedCell(cell)) {
      continue;
    }
    const auto gid = mesh.getCellGlobalId(cell);
    std::uint64_t key = kCutContextHashOffset;
    constexpr std::uint64_t contract_tag =
        0x4355545354525531ull; // "CUTSTRU1"
    mixCutContextHash(key, contract_tag);
    // The sign class of every level-set DOF, in the cell's DOF order.
    const auto cell_dofs = dofs.getCellDofs(cell);
    mixCutContextHash(key, static_cast<std::uint64_t>(cell_dofs.size()));
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
      const auto index = offset + static_cast<std::size_t>(cell_dofs[i]);
      if (index >= solution.size() || !std::isfinite(solution[index])) {
        throw std::runtime_error(
            "The cut-structure key found a level-set DOF without a finite value.");
      }
      const auto signed_value = solution[index] - isovalue;
      mixCutContextHash(
          key, signed_value < -band ? 0u : (signed_value > band ? 2u : 1u));
    }
    if (const auto found = facets.find(gid); found != facets.end()) {
      mixCutContextHash(key, static_cast<std::uint64_t>(found->second.size()));
      for (const auto mask : found->second) {
        mixCutContextHash(key, mask);
      }
    }
    keys.emplace(gid, key == 0u ? 1u : key);
  }
  return keys;
}

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

namespace {

// The cut-structure key of a non-LinearCorner request: one fingerprint per
// owned cell, of the set (not the multiset) of the kinds of its retained
// rules, without the source topology key (it counts the leaves of an
// adaptive subdivision) and without pruned leaves (a sliver leaf comes and
// goes with the subdivision; a side whose every leaf is pruned leaves the
// set), and of the cell's cut-structure key (cutStructureCellKeys).
void appendCutStructureFingerprints(
    const ActiveCutTopologySnapshotBinding& binding,
    std::uint64_t request_identity,
    std::vector<std::uint64_t>& out)
{
  if (binding.cell_structure_keys == nullptr) {
    throw std::runtime_error(
        "Active-cut topology fingerprint of a non-LinearCorner request requires the cut-structure keys of its cells.");
  }
  using RuleKind = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                              std::uint64_t, std::uint64_t, std::uint64_t,
                              std::uint64_t>;
  std::map<svmp::FE::GlobalIndex, std::set<RuleKind>> rules_by_cell;
  for (const auto& rule : binding.snapshot->rules()) {
    if (!rule.locally_owned ||
        rule.retention != svmp::FE::interfaces::FreeSurfaceGeometryRetention::Retained) {
      continue;
    }
    const auto& reference = rule.reference_rule;
    const auto& provenance = reference.provenance;
    if (provenance.parent_entity_global_id ==
        svmp::FE::INVALID_GLOBAL_INDEX) {
      throw std::runtime_error(
          "Active-cut topology fingerprint requires canonical global rule identities.");
    }
    rules_by_cell[provenance.parent_entity_global_id].insert(RuleKind{
        static_cast<std::uint64_t>(rule.role),
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(rule.physical_boundary_marker)),
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(provenance.marker)),
        static_cast<std::uint64_t>(provenance.parent_boundary_entity_global_id),
        static_cast<std::uint64_t>(reference.kind),
        static_cast<std::uint64_t>(reference.side),
        reference.full_cell_equivalent ? 1u : 0u});
  }
  for (const auto& [gid, structure_key] : *binding.cell_structure_keys) {
    std::uint64_t fingerprint = kCutContextHashOffset;
    constexpr std::uint64_t cell_identity_tag =
        0x43555443454c4c31ull; // "CUTCELL1"
    mixCutContextHash(fingerprint, cell_identity_tag);
    mixCutContextHash(fingerprint, request_identity);
    mixCutContextHash(fingerprint, static_cast<std::uint64_t>(gid));
    mixCutContextHash(fingerprint, structure_key);
    if (const auto found = rules_by_cell.find(gid);
        found != rules_by_cell.end()) {
      mixCutContextHash(fingerprint,
                        static_cast<std::uint64_t>(found->second.size()));
      for (const auto& kind : found->second) {
        std::apply(
            [&](auto... words) { (mixCutContextHash(fingerprint, words), ...); },
            kind);
      }
      rules_by_cell.erase(found);
    }
    out.push_back(fingerprint == 0u ? 1u : fingerprint);
  }
  if (!rules_by_cell.empty()) {
    throw std::runtime_error(
        "Active-cut topology fingerprint found owned rules of a cell without a cut-structure key.");
  }
}

} // namespace

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
      if (binding.request->geometry_mode !=
          GeneratedInterfaceGeometryMode::LinearCorner) {
        appendCutStructureFingerprints(binding, request_identity,
                                       local_owned_rule_fingerprints);
        continue;
      }
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
  using Backend = svmp::FE::level_set::ImplicitCutQuadratureBackend;
  return std::all_of(
      requests.begin(), requests.end(), [](const auto& request) {
        if (request.geometry_mode == Mode::LinearCorner) {
          return true;
        }
        return request.geometry_mode == Mode::HighOrderImplicit &&
               (request.implicit_cut_backend ==
                    Backend::SayeHyperrectangle ||
                request.implicit_cut_backend == Backend::HighOrderSubcell ||
                request.implicit_cut_backend == Backend::Auto) &&
               request.implicit_cut_fallback_policy ==
                   ImplicitCutFallbackPolicy::Fail;
      });
}

} // namespace level_set
} // namespace FE
} // namespace svmp
