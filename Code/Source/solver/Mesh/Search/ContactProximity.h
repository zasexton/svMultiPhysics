/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_SEARCH_CONTACT_PROXIMITY_H
#define SVMP_SEARCH_CONTACT_PROXIMITY_H

#include "../Core/DistributedMesh.h"
#include "../Core/MeshBase.h"
#include "../Core/MeshTypes.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace search {

enum class ContactEntityKind {
  Vertex,
  Edge,
  Face,
  Surface,
  Shell
};

enum class ContactPairState {
  Candidate,
  Projected,
  Active,
  Inactive,
  Rejected,
  Stale
};

enum class ContactLifecycleStage {
  BroadPhaseCandidate,
  NarrowPhaseProjected,
  Classified
};

enum class ContactTransactionState {
  Empty,
  TrialIterate,
  AcceptedNonlinearState,
  AcceptedTimeStep,
  AcceptedRemeshRezoneState,
  RolledBack
};

enum class ContactDiagnosticCode {
  None,
  NoContact,
  SearchRadiusMiss,
  ProjectionFailure,
  DuplicatePairRemoved,
  StaleRevision,
  UnsupportedTopology,
  ReinitializedAfterRemeshOrRepartition
};

struct ContactExternalRevisions {
  std::uint64_t fe_space_revision = 0;
  std::uint64_t fe_dof_layout_revision = 0;
  std::uint64_t fe_constraint_layout_revision = 0;
  std::uint64_t fe_block_layout_revision = 0;
  std::uint64_t restart_layout_revision = 0;
};

struct ContactRevisionSnapshot {
  Configuration configuration = Configuration::Reference;
  std::uint64_t geometry_revision = 0;
  std::uint64_t reference_geometry_revision = 0;
  std::uint64_t current_geometry_revision = 0;
  std::uint64_t topology_revision = 0;
  std::uint64_t ownership_revision = 0;
  std::uint64_t numbering_revision = 0;
  std::uint64_t field_layout_revision = 0;
  std::uint64_t label_revision = 0;
  std::uint64_t active_configuration_epoch = 0;
  ContactExternalRevisions external{};

  [[nodiscard]] static ContactRevisionSnapshot capture(
      const MeshBase& mesh,
      Configuration configuration,
      ContactExternalRevisions external = {});

  [[nodiscard]] bool matches(
      const MeshBase& mesh,
      Configuration configuration,
      ContactExternalRevisions external = {}) const noexcept;

  [[nodiscard]] std::uint64_t revision_key() const noexcept;
};

struct ContactSurfaceSpec {
  const MeshBase* mesh = nullptr;
  const DistributedMesh* distributed_mesh = nullptr;
  label_t label = INVALID_LABEL;
  ContactEntityKind entity_kind = ContactEntityKind::Face;
  Configuration configuration = Configuration::Reference;
  std::string name;
  bool two_sided = false;
  real_t shell_thickness = 0.0;
  bool allow_self_contact = false;

  [[nodiscard]] static ContactSurfaceSpec from_mesh(
      const MeshBase& mesh,
      label_t label,
      ContactEntityKind entity_kind = ContactEntityKind::Face,
      Configuration configuration = Configuration::Reference,
      std::string name = {});

  [[nodiscard]] static ContactSurfaceSpec from_distributed_mesh(
      const DistributedMesh& mesh,
      label_t label,
      ContactEntityKind entity_kind = ContactEntityKind::Face,
      Configuration configuration = Configuration::Reference,
      std::string name = {});

  [[nodiscard]] const MeshBase& local_mesh() const;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] rank_t local_rank() const noexcept;
  [[nodiscard]] int world_size() const noexcept;
  [[nodiscard]] rank_t owner_rank(index_t entity) const;
  [[nodiscard]] gid_t entity_gid(index_t entity) const;
};

struct ContactCandidateOptions {
  real_t search_radius = std::numeric_limits<real_t>::infinity();
  real_t activation_distance = 0.0;
  bool only_nearest_per_source = false;
  bool include_inactive_candidates = true;
  bool remove_duplicate_pairs = true;
  bool allow_self_pairs = false;
  std::string generation_policy = "centroid-closest-projection";
};

struct ContactPairProvenance {
  std::uint64_t pair_id = 0;
  std::string source_surface_name;
  std::string target_surface_name;
  label_t source_label = INVALID_LABEL;
  label_t target_label = INVALID_LABEL;
  ContactEntityKind source_kind = ContactEntityKind::Face;
  ContactEntityKind target_kind = ContactEntityKind::Face;
  index_t source_entity = INVALID_INDEX;
  index_t target_entity = INVALID_INDEX;
  gid_t source_gid = INVALID_GID;
  gid_t target_gid = INVALID_GID;
  rank_t source_owner_rank = 0;
  rank_t target_owner_rank = 0;
  rank_t source_local_rank = 0;
  rank_t target_local_rank = 0;
  rank_t canonical_owner_rank = 0;
  Configuration source_configuration = Configuration::Reference;
  Configuration target_configuration = Configuration::Reference;
  real_t time_level = 0.0;
  std::string generation_policy;
};

struct ContactProjectionState {
  bool valid = false;
  std::array<real_t, 3> source_point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> target_point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> source_local_coordinates{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> target_local_coordinates{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> source_normal{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> target_normal{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> tangent0{{1.0, 0.0, 0.0}};
  std::array<real_t, 3> tangent1{{0.0, 1.0, 0.0}};
  std::array<real_t, 3> tangential_reference0{{1.0, 0.0, 0.0}};
  std::array<real_t, 3> tangential_reference1{{0.0, 1.0, 0.0}};
  real_t unsigned_gap = std::numeric_limits<real_t>::infinity();
  real_t signed_gap = std::numeric_limits<real_t>::infinity();
  real_t tangential_slip_magnitude = 0.0;
  real_t shell_thickness_offset = 0.0;
  bool tangential_frame_valid = false;
  bool wrong_side_projection = false;
  std::string side;
};

struct ContactPair {
  ContactLifecycleStage lifecycle_stage = ContactLifecycleStage::BroadPhaseCandidate;
  ContactPairState state = ContactPairState::Candidate;
  ContactPairProvenance provenance;
  ContactProjectionState projection;
  std::vector<ContactDiagnosticCode> diagnostics;
};

struct ContactDiagnostic {
  ContactDiagnosticCode code = ContactDiagnosticCode::None;
  std::string message;
  std::uint64_t pair_id = 0;
};

struct ContactRestartMetadata {
  std::string name;
  std::uint64_t source_revision_key = 0;
  std::uint64_t target_revision_key = 0;
  std::uint64_t contact_revision_key = 0;
  std::uint64_t candidate_generation_epoch = 0;
  std::uint64_t active_set_epoch = 0;
  std::size_t pair_count = 0;
  std::size_t active_pair_count = 0;
  ContactTransactionState accepted_state = ContactTransactionState::Empty;
};

struct ContactProximityMap {
  std::string name;
  ContactSurfaceSpec source;
  ContactSurfaceSpec target;
  ContactCandidateOptions options;
  ContactRevisionSnapshot source_revision;
  ContactRevisionSnapshot target_revision;
  std::vector<ContactPair> pairs;
  std::vector<ContactDiagnostic> diagnostics;
  ContactTransactionState state = ContactTransactionState::Empty;
  std::uint64_t candidate_generation_epoch = 0;
  std::uint64_t active_set_epoch = 0;

  [[nodiscard]] bool valid_for_current_revisions(
      ContactExternalRevisions external = {}) const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
  [[nodiscard]] std::size_t active_pair_count() const noexcept;
  [[nodiscard]] std::vector<const ContactPair*> pairs_in_state(ContactPairState state) const;
  [[nodiscard]] bool has_diagnostic(ContactDiagnosticCode code) const noexcept;
  [[nodiscard]] ContactRestartMetadata restart_metadata() const;

  void accept_trial(ContactTransactionState accepted_state = ContactTransactionState::AcceptedNonlinearState);
  void accept_time_step();
  void rollback_trial();
  void mark_stale(ContactDiagnosticCode code = ContactDiagnosticCode::StaleRevision);
  void reinitialize_after_remesh_or_repartition(std::string reason = {});
};

struct ContactRegistryEntry {
  std::string name;
  ContactSurfaceSpec source;
  ContactSurfaceSpec target;
  ContactCandidateOptions options;
};

class ContactProximityRegistry {
public:
  void register_contact(
      std::string name,
      ContactSurfaceSpec source,
      ContactSurfaceSpec target,
      ContactCandidateOptions options = {});

  [[nodiscard]] bool contains(const std::string& name) const noexcept;
  [[nodiscard]] std::vector<std::string> contact_names() const;
  [[nodiscard]] const ContactRegistryEntry& contact_entry(const std::string& name) const;

  [[nodiscard]] ContactProximityMap build_trial_map(
      const std::string& name,
      ContactExternalRevisions external = {},
      real_t time_level = 0.0) const;

  void commit_map(
      ContactProximityMap map,
      ContactTransactionState accepted_state = ContactTransactionState::AcceptedTimeStep);

  void rollback_committed_map(const std::string& name);
  [[nodiscard]] const ContactProximityMap* committed_map(const std::string& name) const noexcept;
  [[nodiscard]] bool committed_map_valid(
      const std::string& name,
      ContactExternalRevisions external = {}) const noexcept;

private:
  std::unordered_map<std::string, ContactRegistryEntry> entries_;
  std::unordered_map<std::string, ContactProximityMap> committed_maps_;
};

[[nodiscard]] std::string to_string(ContactDiagnosticCode code);
[[nodiscard]] std::string to_string(ContactPairState state);

} // namespace search
} // namespace svmp

#endif // SVMP_SEARCH_CONTACT_PROXIMITY_H
