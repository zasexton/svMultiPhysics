/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MultiMeshInterface.h"

#include "MeshSearch.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
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

Configuration normalized_configuration(Configuration cfg) noexcept {
  return cfg == Configuration::Deformed ? Configuration::Current : cfg;
}

std::array<real_t, 3> sub(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<real_t, 3> add(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
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

real_t distance(const std::array<real_t, 3>& a,
                const std::array<real_t, 3>& b) noexcept {
  return std::sqrt(norm2(sub(a, b)));
}

real_t clamp01(real_t x) noexcept {
  return std::max(real_t{0.0}, std::min(real_t{1.0}, x));
}

std::vector<std::array<real_t, 3>> face_points(
    const MeshBase& mesh,
    index_t face,
    Configuration cfg) {
  const auto span = mesh.face_vertices_span(face);
  std::vector<std::array<real_t, 3>> points;
  points.reserve(span.second);
  for (size_t i = 0; i < span.second; ++i) {
    points.push_back(mesh.geometry_dof_coords(span.first[i], cfg));
  }
  return points;
}

std::array<real_t, 3> closest_on_segment(
    const std::array<real_t, 3>& p,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b) noexcept {
  const auto ab = sub(b, a);
  const real_t denom = norm2(ab);
  if (denom <= 0.0) {
    return a;
  }
  const real_t t = clamp01(dot(sub(p, a), ab) / denom);
  return add(a, scale(ab, t));
}

std::array<real_t, 3> closest_on_triangle(
    const std::array<real_t, 3>& p,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c) noexcept {
  const auto ab = sub(b, a);
  const auto ac = sub(c, a);
  const auto ap = sub(p, a);
  const real_t d1 = dot(ab, ap);
  const real_t d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return a;

  const auto bp = sub(p, b);
  const real_t d3 = dot(ab, bp);
  const real_t d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return b;

  const real_t vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const real_t v = d1 / (d1 - d3);
    return add(a, scale(ab, v));
  }

  const auto cp = sub(p, c);
  const real_t d5 = dot(ab, cp);
  const real_t d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return c;

  const real_t vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const real_t w = d2 / (d2 - d6);
    return add(a, scale(ac, w));
  }

  const real_t va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return add(b, scale(sub(c, b), w));
  }

  const real_t denom = va + vb + vc;
  if (std::abs(denom) <= 1.0e-30) {
    return a;
  }
  const real_t v = vb / denom;
  const real_t w = vc / denom;
  return add(a, add(scale(ab, v), scale(ac, w)));
}

std::array<real_t, 3> barycentric_on_triangle(
    const std::array<real_t, 3>& p,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c) noexcept {
  const auto v0 = sub(b, a);
  const auto v1 = sub(c, a);
  const auto v2 = sub(p, a);
  const real_t d00 = dot(v0, v0);
  const real_t d01 = dot(v0, v1);
  const real_t d11 = dot(v1, v1);
  const real_t d20 = dot(v2, v0);
  const real_t d21 = dot(v2, v1);
  const real_t denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) <= 1.0e-30) {
    return {{1.0, 0.0, 0.0}};
  }
  const real_t v = (d11 * d20 - d01 * d21) / denom;
  const real_t w = (d00 * d21 - d01 * d20) / denom;
  return {{1.0 - v - w, v, w}};
}

index_t first_incident_cell(const MeshBase& mesh, index_t face) noexcept {
  const auto& face2cell = mesh.face2cell();
  if (face < 0 || static_cast<size_t>(face) >= face2cell.size()) {
    return INVALID_INDEX;
  }
  if (face2cell[static_cast<size_t>(face)][0] != INVALID_INDEX) {
    return face2cell[static_cast<size_t>(face)][0];
  }
  return face2cell[static_cast<size_t>(face)][1];
}

std::vector<index_t> labelled_faces(const MeshBase& mesh, label_t label) {
  if (label != INVALID_LABEL) {
    return mesh.faces_with_label(label);
  }

  std::vector<index_t> faces;
  faces.reserve(mesh.n_boundary_faces());
  for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
    const auto& face2cell = mesh.face2cell();
    const bool boundary = face2cell.empty() ||
        (static_cast<size_t>(f) < face2cell.size() &&
         (face2cell[static_cast<size_t>(f)][0] == INVALID_INDEX ||
          face2cell[static_cast<size_t>(f)][1] == INVALID_INDEX));
    if (boundary) {
      faces.push_back(f);
    }
  }
  return faces;
}

InterfacePair make_pair_record(
    const InterfaceSideSpec& source,
    const InterfaceSideSpec& target,
    index_t source_face,
    index_t target_face,
    const std::array<real_t, 3>& source_point,
    const std::array<real_t, 3>& target_point,
    real_t pair_distance) {
  const MeshBase& source_mesh = source.local_mesh();
  const MeshBase& target_mesh = target.local_mesh();
  InterfacePair pair;
  pair.source_face = source_face;
  pair.target_face = target_face;
  pair.source_cell = first_incident_cell(source_mesh, source_face);
  pair.target_cell = first_incident_cell(target_mesh, target_face);
  pair.source_face_gid = source.face_gid(source_face);
  pair.target_face_gid = target.face_gid(target_face);
  pair.source_owner_rank = source.owner_rank_face(source_face);
  pair.target_owner_rank = target.owner_rank_face(target_face);
  pair.source_local_rank = source.local_rank();
  pair.target_local_rank = target.local_rank();
  pair.source_label = source_mesh.boundary_label(source_face);
  pair.target_label = target_mesh.boundary_label(target_face);
  pair.source_point = source_point;
  pair.target_point = target_point;
  pair.source_face_xi = face_local_coordinates(source_mesh, source_face, source_point, source.configuration);
  pair.target_face_xi = face_local_coordinates(target_mesh, target_face, target_point, target.configuration);
  if (pair.source_cell != INVALID_INDEX) {
    pair.source_cell_xi = MeshSearch::compute_parametric_coords(
        source_mesh, pair.source_cell, source_point, source.configuration);
  }
  if (pair.target_cell != INVALID_INDEX) {
    pair.target_cell_xi = MeshSearch::compute_parametric_coords(
        target_mesh, pair.target_cell, target_point, target.configuration);
  }
  pair.source_normal = source_mesh.face_normal(source_face, source.configuration);
  pair.target_normal = target_mesh.face_normal(target_face, target.configuration);
  pair.source_measure = source_mesh.face_area(source_face, source.configuration);
  pair.target_measure = target_mesh.face_area(target_face, target.configuration);
  pair.distance = pair_distance;
  pair.source_logical_region = source.logical_region;
  pair.target_logical_region = target.logical_region;
  return pair;
}

} // namespace

bool LogicalInterfaceRegionId::compatible_with(
    const LogicalInterfaceRegionId& other) const noexcept {
  if (empty() || other.empty()) {
    return true;
  }
  if (!persistent_id.empty() && !other.persistent_id.empty() &&
      persistent_id != other.persistent_id) {
    return false;
  }
  if (kind != LogicalInterfaceRegionKind::Generic &&
      other.kind != LogicalInterfaceRegionKind::Generic &&
      kind != other.kind) {
    return false;
  }
  return true;
}

InterfaceRevisionSnapshot InterfaceRevisionSnapshot::capture(
    const MeshBase& mesh,
    Configuration configuration) {
  InterfaceRevisionSnapshot snapshot;
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
  return snapshot;
}

bool InterfaceRevisionSnapshot::matches(
    const MeshBase& mesh,
    Configuration configuration) const noexcept {
  return this->configuration == normalized_configuration(configuration) &&
         geometry_revision == mesh.geometry_revision() &&
         reference_geometry_revision == mesh.reference_geometry_revision() &&
         current_geometry_revision == mesh.current_geometry_revision() &&
         topology_revision == mesh.topology_revision() &&
         ownership_revision == mesh.ownership_revision() &&
         numbering_revision == mesh.numbering_revision() &&
         field_layout_revision == mesh.field_layout_revision() &&
         label_revision == mesh.label_revision() &&
         active_configuration_epoch == mesh.active_configuration_epoch();
}

std::uint64_t InterfaceRevisionSnapshot::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, static_cast<std::uint64_t>(normalized_configuration(configuration)));
  h = append_hash(h, geometry_revision);
  h = append_hash(h, reference_geometry_revision);
  h = append_hash(h, current_geometry_revision);
  h = append_hash(h, topology_revision);
  h = append_hash(h, ownership_revision);
  h = append_hash(h, numbering_revision);
  h = append_hash(h, field_layout_revision);
  h = append_hash(h, label_revision);
  h = append_hash(h, active_configuration_epoch);
  return h;
}

InterfaceSideSpec InterfaceSideSpec::from_mesh(
    const MeshBase& mesh,
    label_t boundary_label,
    Configuration configuration,
    std::string name) {
  InterfaceSideSpec spec;
  spec.mesh = &mesh;
  spec.boundary_label = boundary_label;
  spec.configuration = normalized_configuration(configuration);
  spec.name = std::move(name);
  return spec;
}

InterfaceSideSpec InterfaceSideSpec::from_distributed_mesh(
    const DistributedMesh& mesh,
    label_t boundary_label,
    Configuration configuration,
    std::string name) {
  InterfaceSideSpec spec;
  spec.mesh = &mesh.local_mesh();
  spec.distributed_mesh = &mesh;
  spec.boundary_label = boundary_label;
  spec.configuration = normalized_configuration(configuration);
  spec.name = std::move(name);
  return spec;
}

const MeshBase& InterfaceSideSpec::local_mesh() const {
  if (distributed_mesh != nullptr) {
    return distributed_mesh->local_mesh();
  }
  if (mesh == nullptr) {
    throw std::logic_error("InterfaceSideSpec has no mesh");
  }
  return *mesh;
}

bool InterfaceSideSpec::valid() const noexcept {
  return distributed_mesh != nullptr || mesh != nullptr;
}

rank_t InterfaceSideSpec::local_rank() const noexcept {
  return distributed_mesh != nullptr ? distributed_mesh->rank() : 0;
}

int InterfaceSideSpec::world_size() const noexcept {
  return distributed_mesh != nullptr ? distributed_mesh->world_size() : 1;
}

rank_t InterfaceSideSpec::owner_rank_face(index_t face) const {
  return distributed_mesh != nullptr ? distributed_mesh->owner_rank_face(face) : 0;
}

gid_t InterfaceSideSpec::face_gid(index_t face) const {
  const auto& gids = local_mesh().face_gids();
  if (face >= 0 && static_cast<size_t>(face) < gids.size()) {
    return gids[static_cast<size_t>(face)];
  }
  return static_cast<gid_t>(face);
}

bool InterfaceMap::valid_for_current_revisions() const noexcept {
  if (!source.valid() || !target.valid()) {
    return false;
  }
  return source_revision.matches(source.local_mesh(), source.configuration) &&
         target_revision.matches(target.local_mesh(), target.configuration);
}

std::uint64_t InterfaceMap::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, source_revision.revision_key());
  h = append_hash(h, target_revision.revision_key());
  h = append_hash(h, static_cast<std::uint64_t>(source.boundary_label));
  h = append_hash(h, static_cast<std::uint64_t>(target.boundary_label));
  h = append_hash(h, static_cast<std::uint64_t>(pairs.size()));
  return h;
}

void InterfaceMap::accept_trial() {
  if (state == InterfaceMapState::Trial) {
    state = InterfaceMapState::Committed;
  }
}

void InterfaceMap::rollback_trial() {
  if (state == InterfaceMapState::Trial) {
    pairs.clear();
    state = InterfaceMapState::RolledBack;
  }
}

void InterfaceSearchRegistry::register_interface(
    std::string name,
    InterfaceSideSpec source,
    InterfaceSideSpec target,
    real_t max_pair_distance) {
  if (name.empty()) {
    throw std::invalid_argument("interface name must not be empty");
  }
  if (!source.valid() || !target.valid()) {
    throw std::invalid_argument("interface sides must reference valid meshes");
  }

  InterfaceRegistryEntry entry;
  entry.name = name;
  entry.source = std::move(source);
  entry.target = std::move(target);
  entry.max_pair_distance = max_pair_distance;
  entries_[entry.name] = std::move(entry);
  committed_maps_.erase(name);
}

bool InterfaceSearchRegistry::contains(const std::string& name) const noexcept {
  return entries_.find(name) != entries_.end();
}

std::vector<std::string> InterfaceSearchRegistry::interface_names() const {
  std::vector<std::string> names;
  names.reserve(entries_.size());
  for (const auto& kv : entries_) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

const InterfaceRegistryEntry& InterfaceSearchRegistry::interface_entry(
    const std::string& name) const {
  const auto it = entries_.find(name);
  if (it == entries_.end()) {
    throw std::out_of_range("unknown interface: " + name);
  }
  return it->second;
}

InterfaceMap InterfaceSearchRegistry::build_trial_map(const std::string& name) const {
  const auto& entry = interface_entry(name);
  const MeshBase& source_mesh = entry.source.local_mesh();
  const MeshBase& target_mesh = entry.target.local_mesh();
  const auto source_faces = labelled_faces(source_mesh, entry.source.boundary_label);
  const auto target_faces = labelled_faces(target_mesh, entry.target.boundary_label);

  InterfaceMap map;
  map.name = name;
  map.source = entry.source;
  map.target = entry.target;
  map.source_revision = InterfaceRevisionSnapshot::capture(source_mesh, entry.source.configuration);
  map.target_revision = InterfaceRevisionSnapshot::capture(target_mesh, entry.target.configuration);
  map.state = InterfaceMapState::Trial;

  for (const index_t source_face : source_faces) {
    const auto source_point = source_mesh.face_center(source_face, entry.source.configuration);
    index_t best_face = INVALID_INDEX;
    std::array<real_t, 3> best_point{{0.0, 0.0, 0.0}};
    real_t best_distance = std::numeric_limits<real_t>::infinity();

    for (const index_t target_face : target_faces) {
      const auto target_point = closest_point_on_face(
          target_mesh, target_face, source_point, entry.target.configuration);
      const real_t d = distance(source_point, target_point);
      if (d < best_distance ||
          (std::abs(d - best_distance) <= 1.0e-14 && target_face < best_face)) {
        best_face = target_face;
        best_point = target_point;
        best_distance = d;
      }
    }

    if (best_face != INVALID_INDEX && best_distance <= entry.max_pair_distance) {
      map.pairs.push_back(make_pair_record(
          entry.source, entry.target, source_face, best_face, source_point, best_point, best_distance));
    }
  }

  std::sort(map.pairs.begin(), map.pairs.end(), [](const InterfacePair& a, const InterfacePair& b) {
    if (a.source_face_gid != b.source_face_gid) return a.source_face_gid < b.source_face_gid;
    if (a.target_face_gid != b.target_face_gid) return a.target_face_gid < b.target_face_gid;
    if (a.source_face != b.source_face) return a.source_face < b.source_face;
    return a.target_face < b.target_face;
  });

  return map;
}

void InterfaceSearchRegistry::commit_map(InterfaceMap map) {
  if (map.name.empty()) {
    throw std::invalid_argument("cannot commit unnamed interface map");
  }
  map.accept_trial();
  if (map.state == InterfaceMapState::Empty) {
    map.state = InterfaceMapState::Committed;
  }
  committed_maps_[map.name] = std::move(map);
}

void InterfaceSearchRegistry::rollback_committed_map(const std::string& name) {
  committed_maps_.erase(name);
}

const InterfaceMap* InterfaceSearchRegistry::committed_map(const std::string& name) const noexcept {
  const auto it = committed_maps_.find(name);
  return it == committed_maps_.end() ? nullptr : &it->second;
}

bool InterfaceSearchRegistry::committed_map_valid(const std::string& name) const noexcept {
  const auto* map = committed_map(name);
  return map != nullptr && map->valid_for_current_revisions();
}

std::array<real_t, 3> closest_point_on_face(
    const MeshBase& mesh,
    index_t face,
    const std::array<real_t, 3>& point,
    Configuration configuration) {
  const auto cfg = normalized_configuration(configuration);
  const auto vertices = face_points(mesh, face, cfg);
  if (vertices.empty()) {
    return point;
  }
  if (vertices.size() == 1) {
    return vertices.front();
  }
  if (vertices.size() == 2) {
    return closest_on_segment(point, vertices[0], vertices[1]);
  }

  std::array<real_t, 3> best = vertices.front();
  real_t best_d2 = std::numeric_limits<real_t>::infinity();
  for (size_t i = 1; i + 1 < vertices.size(); ++i) {
    const auto candidate = closest_on_triangle(point, vertices[0], vertices[i], vertices[i + 1]);
    const real_t d2 = norm2(sub(point, candidate));
    if (d2 < best_d2) {
      best = candidate;
      best_d2 = d2;
    }
  }
  return best;
}

std::array<real_t, 3> face_local_coordinates(
    const MeshBase& mesh,
    index_t face,
    const std::array<real_t, 3>& point,
    Configuration configuration) {
  const auto cfg = normalized_configuration(configuration);
  const auto vertices = face_points(mesh, face, cfg);
  if (vertices.size() == 2) {
    const auto e = sub(vertices[1], vertices[0]);
    const real_t denom = norm2(e);
    if (denom <= 0.0) {
      return {{-1.0, 0.0, 0.0}};
    }
    const real_t t = dot(sub(point, vertices[0]), e) / denom;
    return {{2.0 * t - 1.0, 0.0, 0.0}};
  }
  if (vertices.size() == 3) {
    const auto bary = barycentric_on_triangle(point, vertices[0], vertices[1], vertices[2]);
    return {{bary[1], bary[2], 0.0}};
  }
  if (vertices.size() >= 4) {
    const auto e0 = sub(vertices[1], vertices[0]);
    const auto e1 = sub(vertices[3], vertices[0]);
    const auto normal = cross(e0, e1);
    const real_t n2 = norm2(normal);
    const auto q = n2 > 0.0
        ? sub(point, scale(normal, dot(sub(point, vertices[0]), normal) / n2))
        : point;
    const real_t s = norm2(e0) > 0.0 ? dot(sub(q, vertices[0]), e0) / norm2(e0) : 0.0;
    const real_t t = norm2(e1) > 0.0 ? dot(sub(q, vertices[0]), e1) / norm2(e1) : 0.0;
    return {{2.0 * s - 1.0, 2.0 * t - 1.0, 0.0}};
  }
  return {{0.0, 0.0, 0.0}};
}

InterfaceProvenanceDiagnostic validate_interface_provenance(
    const InterfaceMap& map) {
  InterfaceProvenanceDiagnostic diagnostic;
  const auto source_region = map.source.logical_region;
  const auto target_region = map.target.logical_region;

  if (!source_region.empty() && source_region.physical_label != INVALID_LABEL &&
      source_region.physical_label != map.source.boundary_label) {
    diagnostic.ok = false;
    diagnostic.messages.push_back(
        "source logical region physical label does not match interface side label");
  }
  if (!target_region.empty() && target_region.physical_label != INVALID_LABEL &&
      target_region.physical_label != map.target.boundary_label) {
    diagnostic.ok = false;
    diagnostic.messages.push_back(
        "target logical region physical label does not match interface side label");
  }

  for (const auto& pair : map.pairs) {
    if (!source_region.compatible_with(pair.source_logical_region)) {
      diagnostic.ok = false;
      diagnostic.messages.push_back(
          "source pair logical region is incompatible with interface side identity");
      break;
    }
  }
  for (const auto& pair : map.pairs) {
    if (!target_region.compatible_with(pair.target_logical_region)) {
      diagnostic.ok = false;
      diagnostic.messages.push_back(
          "target pair logical region is incompatible with interface side identity");
      break;
    }
  }

  return diagnostic;
}

} // namespace search
} // namespace svmp
