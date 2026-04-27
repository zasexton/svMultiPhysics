/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ContactProximity.h"

#include "MultiMeshInterface.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace svmp {
namespace search {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

std::uint64_t append_hash(std::uint64_t h, std::uint64_t value) noexcept {
  h ^= value;
  h *= kFnvPrime;
  return h;
}

std::uint64_t append_string_hash(std::uint64_t h, const std::string& value) noexcept {
  for (const unsigned char ch : value) {
    h = append_hash(h, static_cast<std::uint64_t>(ch));
  }
  return h;
}

Configuration normalized_configuration(Configuration cfg) noexcept {
  return cfg == Configuration::Deformed ? Configuration::Current : cfg;
}

std::array<real_t, 3> add(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

std::array<real_t, 3> sub(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<real_t, 3> scale(const std::array<real_t, 3>& a, real_t s) noexcept {
  return {{s * a[0], s * a[1], s * a[2]}};
}

real_t dot(const std::array<real_t, 3>& a,
           const std::array<real_t, 3>& b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<real_t, 3> cross(const std::array<real_t, 3>& a,
                            const std::array<real_t, 3>& b) noexcept {
  return {{a[1] * b[2] - a[2] * b[1],
           a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]}};
}

real_t norm2(const std::array<real_t, 3>& a) noexcept {
  return dot(a, a);
}

real_t norm(const std::array<real_t, 3>& a) noexcept {
  return std::sqrt(norm2(a));
}

std::array<real_t, 3> normalized_or(
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& fallback) noexcept {
  const real_t n = norm(a);
  if (n <= 1.0e-30) {
    return fallback;
  }
  return scale(a, 1.0 / n);
}

real_t distance(const std::array<real_t, 3>& a,
                const std::array<real_t, 3>& b) noexcept {
  return norm(sub(a, b));
}

bool finite_point(const std::array<real_t, 3>& p) noexcept {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

std::pair<std::array<real_t, 3>, std::array<real_t, 3>> tangent_basis(
    const std::array<real_t, 3>& normal) noexcept {
  const auto n = normalized_or(normal, {{0.0, 0.0, 1.0}});
  const std::array<real_t, 3> axis =
      std::abs(n[2]) < 0.9 ? std::array<real_t, 3>{{0.0, 0.0, 1.0}}
                           : std::array<real_t, 3>{{0.0, 1.0, 0.0}};
  const auto t0 = normalized_or(cross(axis, n), {{1.0, 0.0, 0.0}});
  const auto t1 = normalized_or(cross(n, t0), {{0.0, 1.0, 0.0}});
  return {t0, t1};
}

std::vector<index_t> labelled_entities(
    const MeshBase& mesh,
    ContactEntityKind kind,
    label_t label) {
  switch (kind) {
    case ContactEntityKind::Vertex:
      if (label != INVALID_LABEL) {
        return mesh.vertices_with_label(label);
      }
      {
        std::vector<index_t> ids(mesh.n_vertices());
        for (index_t i = 0; i < static_cast<index_t>(ids.size()); ++i) ids[static_cast<std::size_t>(i)] = i;
        return ids;
      }
    case ContactEntityKind::Edge:
      if (label != INVALID_LABEL) {
        return mesh.edges_with_label(label);
      }
      {
        std::vector<index_t> ids(mesh.n_edges());
        for (index_t i = 0; i < static_cast<index_t>(ids.size()); ++i) ids[static_cast<std::size_t>(i)] = i;
        return ids;
      }
    case ContactEntityKind::Face:
    case ContactEntityKind::Surface:
    case ContactEntityKind::Shell:
      if (label != INVALID_LABEL) {
        return mesh.faces_with_label(label);
      }
      {
        std::vector<index_t> ids(mesh.n_faces());
        for (index_t i = 0; i < static_cast<index_t>(ids.size()); ++i) ids[static_cast<std::size_t>(i)] = i;
        return ids;
      }
  }
  return {};
}

std::size_t entity_topology_count(const MeshBase& mesh, ContactEntityKind kind) noexcept {
  switch (kind) {
    case ContactEntityKind::Vertex:
      return mesh.n_vertices();
    case ContactEntityKind::Edge:
      return mesh.n_edges();
    case ContactEntityKind::Face:
    case ContactEntityKind::Surface:
    case ContactEntityKind::Shell:
      return mesh.n_faces();
  }
  return 0;
}

std::array<real_t, 3> edge_point(
    const MeshBase& mesh,
    index_t edge,
    int endpoint,
    Configuration cfg) {
  const auto vertices = mesh.edge_vertices(edge);
  return mesh.geometry_dof_coords(vertices[static_cast<std::size_t>(endpoint)], cfg);
}

std::array<real_t, 3> closest_point_on_segment(
    const std::array<real_t, 3>& point,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b) noexcept {
  const auto ab = sub(b, a);
  const real_t denom = norm2(ab);
  if (denom <= 1.0e-30) {
    return a;
  }
  const real_t t = std::max(real_t{0.0}, std::min(real_t{1.0}, dot(sub(point, a), ab) / denom));
  return add(a, scale(ab, t));
}

std::array<real_t, 3> entity_center(
    const MeshBase& mesh,
    ContactEntityKind kind,
    index_t entity,
    Configuration cfg) {
  switch (kind) {
    case ContactEntityKind::Vertex:
      return mesh.geometry_dof_coords(entity, cfg);
    case ContactEntityKind::Edge:
      return scale(add(edge_point(mesh, entity, 0, cfg), edge_point(mesh, entity, 1, cfg)), 0.5);
    case ContactEntityKind::Face:
    case ContactEntityKind::Surface:
    case ContactEntityKind::Shell:
      return mesh.face_center(entity, cfg);
  }
  return {{0.0, 0.0, 0.0}};
}

std::array<real_t, 3> closest_point_on_entity(
    const MeshBase& mesh,
    ContactEntityKind kind,
    index_t entity,
    const std::array<real_t, 3>& point,
    Configuration cfg) {
  switch (kind) {
    case ContactEntityKind::Vertex:
      return mesh.geometry_dof_coords(entity, cfg);
    case ContactEntityKind::Edge:
      return closest_point_on_segment(point, edge_point(mesh, entity, 0, cfg), edge_point(mesh, entity, 1, cfg));
    case ContactEntityKind::Face:
    case ContactEntityKind::Surface:
    case ContactEntityKind::Shell:
      return closest_point_on_face(mesh, entity, point, cfg);
  }
  return point;
}

std::array<real_t, 3> local_coordinates_on_entity(
    const MeshBase& mesh,
    ContactEntityKind kind,
    index_t entity,
    const std::array<real_t, 3>& point,
    Configuration cfg) {
  switch (kind) {
    case ContactEntityKind::Vertex:
      return {{0.0, 0.0, 0.0}};
    case ContactEntityKind::Edge: {
      const auto a = edge_point(mesh, entity, 0, cfg);
      const auto b = edge_point(mesh, entity, 1, cfg);
      const auto ab = sub(b, a);
      const real_t denom = norm2(ab);
      if (denom <= 1.0e-30) {
        return {{-1.0, 0.0, 0.0}};
      }
      const real_t t = dot(sub(point, a), ab) / denom;
      return {{2.0 * t - 1.0, 0.0, 0.0}};
    }
    case ContactEntityKind::Face:
    case ContactEntityKind::Surface:
    case ContactEntityKind::Shell:
      return face_local_coordinates(mesh, entity, point, cfg);
  }
  return {{0.0, 0.0, 0.0}};
}

std::array<real_t, 3> entity_normal(
    const MeshBase& mesh,
    ContactEntityKind kind,
    index_t entity,
    const std::array<real_t, 3>& fallback,
    Configuration cfg) {
  switch (kind) {
    case ContactEntityKind::Face:
    case ContactEntityKind::Surface:
    case ContactEntityKind::Shell:
      return normalized_or(mesh.face_normal(entity, cfg), fallback);
    case ContactEntityKind::Vertex:
    case ContactEntityKind::Edge:
      return normalized_or(fallback, {{0.0, 0.0, 1.0}});
  }
  return normalized_or(fallback, {{0.0, 0.0, 1.0}});
}

EntityKind mesh_entity_kind(ContactEntityKind kind) noexcept {
  switch (kind) {
    case ContactEntityKind::Vertex:
      return EntityKind::Vertex;
    case ContactEntityKind::Edge:
      return EntityKind::Edge;
    case ContactEntityKind::Face:
    case ContactEntityKind::Surface:
    case ContactEntityKind::Shell:
      return EntityKind::Face;
  }
  return EntityKind::Face;
}

bool same_mesh(const ContactSurfaceSpec& a, const ContactSurfaceSpec& b) noexcept {
  if (a.distributed_mesh != nullptr || b.distributed_mesh != nullptr) {
    return a.distributed_mesh != nullptr && a.distributed_mesh == b.distributed_mesh;
  }
  return a.mesh != nullptr && a.mesh == b.mesh;
}

std::uint64_t canonical_pair_id(const ContactPairProvenance& p) noexcept {
  auto left = std::make_tuple(static_cast<int>(p.source_kind), p.source_gid,
                              static_cast<std::int64_t>(p.source_label),
                              p.source_surface_name);
  auto right = std::make_tuple(static_cast<int>(p.target_kind), p.target_gid,
                               static_cast<std::int64_t>(p.target_label),
                               p.target_surface_name);
  if (right < left) {
    std::swap(left, right);
  }

  std::uint64_t h = kFnvOffset;
  h = append_hash(h, static_cast<std::uint64_t>(std::get<0>(left)));
  h = append_hash(h, static_cast<std::uint64_t>(std::get<1>(left)));
  h = append_hash(h, static_cast<std::uint64_t>(std::get<2>(left)));
  h = append_string_hash(h, std::get<3>(left));
  h = append_hash(h, static_cast<std::uint64_t>(std::get<0>(right)));
  h = append_hash(h, static_cast<std::uint64_t>(std::get<1>(right)));
  h = append_hash(h, static_cast<std::uint64_t>(std::get<2>(right)));
  h = append_string_hash(h, std::get<3>(right));
  return h;
}

std::string diagnostic_message(ContactDiagnosticCode code) {
  switch (code) {
    case ContactDiagnosticCode::None:
      return "no diagnostic";
    case ContactDiagnosticCode::NoContact:
      return "no contact candidates were generated";
    case ContactDiagnosticCode::SearchRadiusMiss:
      return "candidate rejected by search radius";
    case ContactDiagnosticCode::ProjectionFailure:
      return "closest-point projection failed";
    case ContactDiagnosticCode::DuplicatePairRemoved:
      return "duplicate contact pair removed";
    case ContactDiagnosticCode::StaleRevision:
      return "contact state is stale for current revisions";
    case ContactDiagnosticCode::UnsupportedTopology:
      return "unsupported contact entity topology";
    case ContactDiagnosticCode::ReinitializedAfterRemeshOrRepartition:
      return "contact state reinitialized after remesh or repartition";
  }
  return "unknown contact diagnostic";
}

ContactPair make_pair(
    const ContactSurfaceSpec& source,
    const ContactSurfaceSpec& target,
    const ContactCandidateOptions& options,
    index_t source_entity,
    index_t target_entity,
    real_t time_level) {
  const MeshBase& source_mesh = source.local_mesh();
  const MeshBase& target_mesh = target.local_mesh();
  const auto source_cfg = normalized_configuration(source.configuration);
  const auto target_cfg = normalized_configuration(target.configuration);
  const auto source_point = entity_center(source_mesh, source.entity_kind, source_entity, source_cfg);
  const auto target_point =
      closest_point_on_entity(target_mesh, target.entity_kind, target_entity, source_point, target_cfg);
  const auto source_to_target = sub(target_point, source_point);
  const real_t raw_gap = distance(source_point, target_point);
  const real_t shell_offset = std::max(real_t{0.0}, source.shell_thickness) * real_t{0.5} +
                              std::max(real_t{0.0}, target.shell_thickness) * real_t{0.5};
  const auto fallback_normal = normalized_or(source_to_target, {{0.0, 0.0, 1.0}});
  const auto source_normal = entity_normal(source_mesh, source.entity_kind, source_entity, fallback_normal, source_cfg);
  const auto target_normal = entity_normal(target_mesh, target.entity_kind, target_entity, scale(fallback_normal, -1.0), target_cfg);
  const real_t signed_gap = dot(source_to_target, source_normal) - shell_offset;
  const auto tangents = tangent_basis(source_normal);

  ContactPair pair;
  pair.provenance.source_surface_name = source.name;
  pair.provenance.target_surface_name = target.name;
  pair.provenance.source_label = source.label;
  pair.provenance.target_label = target.label;
  pair.provenance.source_kind = source.entity_kind;
  pair.provenance.target_kind = target.entity_kind;
  pair.provenance.source_entity = source_entity;
  pair.provenance.target_entity = target_entity;
  pair.provenance.source_gid = source.entity_gid(source_entity);
  pair.provenance.target_gid = target.entity_gid(target_entity);
  pair.provenance.source_owner_rank = source.owner_rank(source_entity);
  pair.provenance.target_owner_rank = target.owner_rank(target_entity);
  pair.provenance.source_local_rank = source.local_rank();
  pair.provenance.target_local_rank = target.local_rank();
  pair.provenance.canonical_owner_rank =
      std::min(pair.provenance.source_owner_rank, pair.provenance.target_owner_rank);
  pair.provenance.source_configuration = source_cfg;
  pair.provenance.target_configuration = target_cfg;
  pair.provenance.time_level = time_level;
  pair.provenance.generation_policy = options.generation_policy;
  pair.provenance.pair_id = canonical_pair_id(pair.provenance);

  pair.projection.valid = finite_point(source_point) && finite_point(target_point) && std::isfinite(raw_gap);
  pair.projection.source_point = source_point;
  pair.projection.target_point = target_point;
  pair.projection.source_local_coordinates =
      local_coordinates_on_entity(source_mesh, source.entity_kind, source_entity, source_point, source_cfg);
  pair.projection.target_local_coordinates =
      local_coordinates_on_entity(target_mesh, target.entity_kind, target_entity, target_point, target_cfg);
  pair.projection.source_normal = source_normal;
  pair.projection.target_normal = target_normal;
  pair.projection.tangent0 = tangents.first;
  pair.projection.tangent1 = tangents.second;
  pair.projection.tangential_reference0 = tangents.first;
  pair.projection.tangential_reference1 = tangents.second;
  pair.projection.unsigned_gap = std::max(real_t{0.0}, raw_gap - shell_offset);
  pair.projection.signed_gap = signed_gap;
  pair.projection.tangential_slip_magnitude = 0.0;
  pair.projection.shell_thickness_offset = shell_offset;
  pair.projection.tangential_frame_valid = pair.projection.valid;
  pair.projection.wrong_side_projection = !source.two_sided && signed_gap < -1.0e-12;
  pair.projection.side = signed_gap >= 0.0 ? "positive" : "negative";
  pair.lifecycle_stage = ContactLifecycleStage::NarrowPhaseProjected;

  if (!pair.projection.valid) {
    pair.state = ContactPairState::Rejected;
    pair.diagnostics.push_back(ContactDiagnosticCode::ProjectionFailure);
  } else if (pair.projection.unsigned_gap <= options.activation_distance) {
    pair.state = ContactPairState::Active;
  } else if (options.include_inactive_candidates) {
    pair.state = ContactPairState::Inactive;
  } else {
    pair.state = ContactPairState::Projected;
  }
  pair.lifecycle_stage = ContactLifecycleStage::Classified;
  return pair;
}

void add_diagnostic(ContactProximityMap& map, ContactDiagnosticCode code, std::uint64_t pair_id = 0) {
  map.diagnostics.push_back({code, diagnostic_message(code), pair_id});
}

} // namespace

ContactRevisionSnapshot ContactRevisionSnapshot::capture(
    const MeshBase& mesh,
    Configuration configuration,
    ContactExternalRevisions external_revisions) {
  ContactRevisionSnapshot snapshot;
  snapshot.configuration = normalized_configuration(configuration);
  snapshot.geometry_revision = mesh.geometry_revision();
  snapshot.reference_geometry_revision = mesh.reference_geometry_revision();
  snapshot.current_geometry_revision = mesh.current_geometry_revision();
  snapshot.topology_revision = mesh.topology_revision();
  snapshot.ownership_revision = mesh.ownership_revision();
  snapshot.numbering_revision = mesh.numbering_revision();
  snapshot.field_layout_revision = mesh.field_layout_revision();
  snapshot.label_revision = mesh.label_revision();
  snapshot.active_configuration_epoch = mesh.active_configuration_epoch();
  snapshot.external = external_revisions;
  return snapshot;
}

bool ContactRevisionSnapshot::matches(
    const MeshBase& mesh,
    Configuration configuration,
    ContactExternalRevisions external_revisions) const noexcept {
  return this->configuration == normalized_configuration(configuration) &&
         geometry_revision == mesh.geometry_revision() &&
         reference_geometry_revision == mesh.reference_geometry_revision() &&
         current_geometry_revision == mesh.current_geometry_revision() &&
         topology_revision == mesh.topology_revision() &&
         ownership_revision == mesh.ownership_revision() &&
         numbering_revision == mesh.numbering_revision() &&
         field_layout_revision == mesh.field_layout_revision() &&
         label_revision == mesh.label_revision() &&
         active_configuration_epoch == mesh.active_configuration_epoch() &&
         external.fe_space_revision == external_revisions.fe_space_revision &&
         external.fe_dof_layout_revision == external_revisions.fe_dof_layout_revision &&
         external.fe_constraint_layout_revision == external_revisions.fe_constraint_layout_revision &&
         external.fe_block_layout_revision == external_revisions.fe_block_layout_revision &&
         external.restart_layout_revision == external_revisions.restart_layout_revision;
}

std::uint64_t ContactRevisionSnapshot::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, static_cast<std::uint64_t>(configuration));
  h = append_hash(h, geometry_revision);
  h = append_hash(h, reference_geometry_revision);
  h = append_hash(h, current_geometry_revision);
  h = append_hash(h, topology_revision);
  h = append_hash(h, ownership_revision);
  h = append_hash(h, numbering_revision);
  h = append_hash(h, field_layout_revision);
  h = append_hash(h, label_revision);
  h = append_hash(h, active_configuration_epoch);
  h = append_hash(h, external.fe_space_revision);
  h = append_hash(h, external.fe_dof_layout_revision);
  h = append_hash(h, external.fe_constraint_layout_revision);
  h = append_hash(h, external.fe_block_layout_revision);
  h = append_hash(h, external.restart_layout_revision);
  return h;
}

ContactSurfaceSpec ContactSurfaceSpec::from_mesh(
    const MeshBase& mesh,
    label_t label_value,
    ContactEntityKind kind,
    Configuration configuration_value,
    std::string name_value) {
  ContactSurfaceSpec spec;
  spec.mesh = &mesh;
  spec.label = label_value;
  spec.entity_kind = kind;
  spec.configuration = normalized_configuration(configuration_value);
  spec.name = std::move(name_value);
  return spec;
}

ContactSurfaceSpec ContactSurfaceSpec::from_distributed_mesh(
    const DistributedMesh& mesh_value,
    label_t label_value,
    ContactEntityKind kind,
    Configuration configuration_value,
    std::string name_value) {
  ContactSurfaceSpec spec;
  spec.mesh = &mesh_value.local_mesh();
  spec.distributed_mesh = &mesh_value;
  spec.label = label_value;
  spec.entity_kind = kind;
  spec.configuration = normalized_configuration(configuration_value);
  spec.name = std::move(name_value);
  return spec;
}

const MeshBase& ContactSurfaceSpec::local_mesh() const {
  if (distributed_mesh != nullptr) {
    return distributed_mesh->local_mesh();
  }
  if (mesh == nullptr) {
    throw std::logic_error("ContactSurfaceSpec has no mesh");
  }
  return *mesh;
}

bool ContactSurfaceSpec::valid() const noexcept {
  return distributed_mesh != nullptr || mesh != nullptr;
}

rank_t ContactSurfaceSpec::local_rank() const noexcept {
  return distributed_mesh != nullptr ? distributed_mesh->rank() : 0;
}

int ContactSurfaceSpec::world_size() const noexcept {
  return distributed_mesh != nullptr ? distributed_mesh->world_size() : 1;
}

rank_t ContactSurfaceSpec::owner_rank(index_t entity) const {
  if (distributed_mesh == nullptr) {
    return 0;
  }
  switch (mesh_entity_kind(entity_kind)) {
    case EntityKind::Vertex:
      return distributed_mesh->owner_rank_vertex(entity);
    case EntityKind::Edge:
      return distributed_mesh->owner_rank_edge(entity);
    case EntityKind::Face:
      return distributed_mesh->owner_rank_face(entity);
    case EntityKind::Volume:
      return distributed_mesh->owner_rank_cell(entity);
  }
  return 0;
}

gid_t ContactSurfaceSpec::entity_gid(index_t entity) const {
  const auto& local = local_mesh();
  switch (mesh_entity_kind(entity_kind)) {
    case EntityKind::Vertex: {
      const auto& gids = local.vertex_gids();
      return entity >= 0 && static_cast<std::size_t>(entity) < gids.size() ? gids[static_cast<std::size_t>(entity)]
                                                                           : static_cast<gid_t>(entity);
    }
    case EntityKind::Edge: {
      const auto& gids = local.edge_gids();
      return entity >= 0 && static_cast<std::size_t>(entity) < gids.size() ? gids[static_cast<std::size_t>(entity)]
                                                                           : static_cast<gid_t>(entity);
    }
    case EntityKind::Face: {
      const auto& gids = local.face_gids();
      return entity >= 0 && static_cast<std::size_t>(entity) < gids.size() ? gids[static_cast<std::size_t>(entity)]
                                                                           : static_cast<gid_t>(entity);
    }
    case EntityKind::Volume: {
      const auto& gids = local.cell_gids();
      return entity >= 0 && static_cast<std::size_t>(entity) < gids.size() ? gids[static_cast<std::size_t>(entity)]
                                                                           : static_cast<gid_t>(entity);
    }
  }
  return static_cast<gid_t>(entity);
}

bool ContactProximityMap::valid_for_current_revisions(
    ContactExternalRevisions external_revisions) const noexcept {
  if (!source.valid() || !target.valid()) {
    return false;
  }
  return source_revision.matches(source.local_mesh(), source.configuration, external_revisions) &&
         target_revision.matches(target.local_mesh(), target.configuration, external_revisions);
}

std::uint64_t ContactProximityMap::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_string_hash(h, name);
  h = append_hash(h, source_revision.revision_key());
  h = append_hash(h, target_revision.revision_key());
  h = append_hash(h, static_cast<std::uint64_t>(source.label));
  h = append_hash(h, static_cast<std::uint64_t>(target.label));
  h = append_hash(h, static_cast<std::uint64_t>(source.entity_kind));
  h = append_hash(h, static_cast<std::uint64_t>(target.entity_kind));
  h = append_hash(h, candidate_generation_epoch);
  h = append_hash(h, active_set_epoch);
  for (const auto& pair : pairs) {
    h = append_hash(h, pair.provenance.pair_id);
    h = append_hash(h, static_cast<std::uint64_t>(pair.state));
  }
  return h;
}

std::size_t ContactProximityMap::active_pair_count() const noexcept {
  return static_cast<std::size_t>(std::count_if(pairs.begin(), pairs.end(), [](const ContactPair& pair) {
    return pair.state == ContactPairState::Active;
  }));
}

std::vector<const ContactPair*> ContactProximityMap::pairs_in_state(ContactPairState requested) const {
  std::vector<const ContactPair*> result;
  for (const auto& pair : pairs) {
    if (pair.state == requested) {
      result.push_back(&pair);
    }
  }
  return result;
}

bool ContactProximityMap::has_diagnostic(ContactDiagnosticCode code) const noexcept {
  return std::any_of(diagnostics.begin(), diagnostics.end(), [code](const ContactDiagnostic& diagnostic) {
    return diagnostic.code == code;
  });
}

ContactRestartMetadata ContactProximityMap::restart_metadata() const {
  ContactRestartMetadata metadata;
  metadata.name = name;
  metadata.source_revision_key = source_revision.revision_key();
  metadata.target_revision_key = target_revision.revision_key();
  metadata.contact_revision_key = revision_key();
  metadata.candidate_generation_epoch = candidate_generation_epoch;
  metadata.active_set_epoch = active_set_epoch;
  metadata.pair_count = pairs.size();
  metadata.active_pair_count = active_pair_count();
  metadata.accepted_state = state;
  return metadata;
}

void ContactProximityMap::accept_trial(ContactTransactionState accepted_state) {
  if (state == ContactTransactionState::TrialIterate) {
    state = accepted_state;
  }
}

void ContactProximityMap::accept_time_step() {
  state = ContactTransactionState::AcceptedTimeStep;
}

void ContactProximityMap::rollback_trial() {
  if (state == ContactTransactionState::TrialIterate) {
    pairs.clear();
    diagnostics.clear();
    state = ContactTransactionState::RolledBack;
  }
}

void ContactProximityMap::mark_stale(ContactDiagnosticCode code) {
  for (auto& pair : pairs) {
    pair.state = ContactPairState::Stale;
    pair.diagnostics.push_back(code);
  }
  add_diagnostic(*this, code);
}

void ContactProximityMap::reinitialize_after_remesh_or_repartition(std::string reason) {
  pairs.clear();
  state = ContactTransactionState::AcceptedRemeshRezoneState;
  ++candidate_generation_epoch;
  ++active_set_epoch;
  if (!reason.empty()) {
    diagnostics.push_back({ContactDiagnosticCode::ReinitializedAfterRemeshOrRepartition,
                           std::move(reason), 0});
  } else {
    add_diagnostic(*this, ContactDiagnosticCode::ReinitializedAfterRemeshOrRepartition);
  }
}

void ContactProximityRegistry::register_contact(
    std::string name,
    ContactSurfaceSpec source,
    ContactSurfaceSpec target,
    ContactCandidateOptions options) {
  if (name.empty()) {
    throw std::invalid_argument("contact name must not be empty");
  }
  if (!source.valid() || !target.valid()) {
    throw std::invalid_argument("contact surfaces must reference valid meshes");
  }
  if (options.search_radius < real_t{0.0}) {
    throw std::invalid_argument("contact search radius must be non-negative");
  }
  if (source.name.empty()) {
    source.name = name + ":source";
  }
  if (target.name.empty()) {
    target.name = name + ":target";
  }

  ContactRegistryEntry entry;
  entry.name = name;
  entry.source = std::move(source);
  entry.target = std::move(target);
  entry.options = std::move(options);
  entries_[entry.name] = std::move(entry);
  committed_maps_.erase(name);
}

bool ContactProximityRegistry::contains(const std::string& name) const noexcept {
  return entries_.find(name) != entries_.end();
}

std::vector<std::string> ContactProximityRegistry::contact_names() const {
  std::vector<std::string> names;
  names.reserve(entries_.size());
  for (const auto& kv : entries_) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

const ContactRegistryEntry& ContactProximityRegistry::contact_entry(const std::string& name) const {
  const auto it = entries_.find(name);
  if (it == entries_.end()) {
    throw std::out_of_range("unknown contact set: " + name);
  }
  return it->second;
}

ContactProximityMap ContactProximityRegistry::build_trial_map(
    const std::string& name,
    ContactExternalRevisions external,
    real_t time_level) const {
  const auto& entry = contact_entry(name);
  const MeshBase& source_mesh = entry.source.local_mesh();
  const MeshBase& target_mesh = entry.target.local_mesh();
  const auto source_entities = labelled_entities(source_mesh, entry.source.entity_kind, entry.source.label);
  const auto target_entities = labelled_entities(target_mesh, entry.target.entity_kind, entry.target.label);

  ContactProximityMap map;
  map.name = name;
  map.source = entry.source;
  map.target = entry.target;
  map.options = entry.options;
  map.source_revision = ContactRevisionSnapshot::capture(source_mesh, entry.source.configuration, external);
  map.target_revision = ContactRevisionSnapshot::capture(target_mesh, entry.target.configuration, external);
  map.state = ContactTransactionState::TrialIterate;
  map.candidate_generation_epoch = 1;
  map.active_set_epoch = 1;

  if (entity_topology_count(source_mesh, entry.source.entity_kind) == 0u ||
      entity_topology_count(target_mesh, entry.target.entity_kind) == 0u) {
    add_diagnostic(map, ContactDiagnosticCode::UnsupportedTopology);
    return map;
  }

  bool any_radius_miss = false;
  std::unordered_map<std::uint64_t, std::size_t> unique_pair_to_index;

  for (const index_t source_entity : source_entities) {
    ContactPair best_pair;
    bool have_best = false;
    real_t best_gap = std::numeric_limits<real_t>::infinity();

    for (const index_t target_entity : target_entities) {
      if (!entry.options.allow_self_pairs &&
          same_mesh(entry.source, entry.target) &&
          entry.source.entity_kind == entry.target.entity_kind &&
          source_entity == target_entity) {
        continue;
      }

      auto pair = make_pair(entry.source, entry.target, entry.options, source_entity, target_entity, time_level);
      if (!pair.projection.valid) {
        add_diagnostic(map, ContactDiagnosticCode::ProjectionFailure, pair.provenance.pair_id);
        continue;
      }
      if (pair.projection.unsigned_gap > entry.options.search_radius) {
        any_radius_miss = true;
        continue;
      }

      if (entry.options.only_nearest_per_source) {
        if (!have_best ||
            pair.projection.unsigned_gap < best_gap ||
            (std::abs(pair.projection.unsigned_gap - best_gap) <= 1.0e-14 &&
             pair.provenance.target_gid < best_pair.provenance.target_gid)) {
          best_gap = pair.projection.unsigned_gap;
          best_pair = std::move(pair);
          have_best = true;
        }
        continue;
      }

      if (entry.options.remove_duplicate_pairs) {
        const auto existing = unique_pair_to_index.find(pair.provenance.pair_id);
        if (existing != unique_pair_to_index.end()) {
          add_diagnostic(map, ContactDiagnosticCode::DuplicatePairRemoved, pair.provenance.pair_id);
          continue;
        }
        unique_pair_to_index[pair.provenance.pair_id] = map.pairs.size();
      }
      map.pairs.push_back(std::move(pair));
    }

    if (entry.options.only_nearest_per_source && have_best) {
      if (entry.options.remove_duplicate_pairs) {
        const auto existing = unique_pair_to_index.find(best_pair.provenance.pair_id);
        if (existing != unique_pair_to_index.end()) {
          add_diagnostic(map, ContactDiagnosticCode::DuplicatePairRemoved, best_pair.provenance.pair_id);
          continue;
        }
        unique_pair_to_index[best_pair.provenance.pair_id] = map.pairs.size();
      }
      map.pairs.push_back(std::move(best_pair));
    }
  }

  std::sort(map.pairs.begin(), map.pairs.end(), [](const ContactPair& a, const ContactPair& b) {
    if (a.provenance.source_gid != b.provenance.source_gid) {
      return a.provenance.source_gid < b.provenance.source_gid;
    }
    if (a.provenance.target_gid != b.provenance.target_gid) {
      return a.provenance.target_gid < b.provenance.target_gid;
    }
    return a.provenance.pair_id < b.provenance.pair_id;
  });

  if (map.pairs.empty()) {
    add_diagnostic(map, any_radius_miss ? ContactDiagnosticCode::SearchRadiusMiss
                                        : ContactDiagnosticCode::NoContact);
  }

  return map;
}

void ContactProximityRegistry::commit_map(
    ContactProximityMap map,
    ContactTransactionState accepted_state) {
  if (map.name.empty()) {
    throw std::invalid_argument("cannot commit unnamed contact map");
  }
  map.accept_trial(accepted_state);
  if (map.state == ContactTransactionState::Empty) {
    map.state = accepted_state;
  }
  committed_maps_[map.name] = std::move(map);
}

void ContactProximityRegistry::rollback_committed_map(const std::string& name) {
  committed_maps_.erase(name);
}

const ContactProximityMap* ContactProximityRegistry::committed_map(
    const std::string& name) const noexcept {
  const auto it = committed_maps_.find(name);
  return it == committed_maps_.end() ? nullptr : &it->second;
}

bool ContactProximityRegistry::committed_map_valid(
    const std::string& name,
    ContactExternalRevisions external) const noexcept {
  const auto* map = committed_map(name);
  return map != nullptr && map->valid_for_current_revisions(external);
}

std::string to_string(ContactDiagnosticCode code) {
  return diagnostic_message(code);
}

std::string to_string(ContactPairState state) {
  switch (state) {
    case ContactPairState::Candidate:
      return "candidate";
    case ContactPairState::Projected:
      return "projected";
    case ContactPairState::Active:
      return "active";
    case ContactPairState::Inactive:
      return "inactive";
    case ContactPairState::Rejected:
      return "rejected";
    case ContactPairState::Stale:
      return "stale";
  }
  return "unknown";
}

} // namespace search
} // namespace svmp
