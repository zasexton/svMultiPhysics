/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "CutCell.h"

#include "../Core/DistributedMesh.h"
#include "../Core/MeshBase.h"
#include "../Topology/CellTopology.h"

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

real_t norm(const std::array<real_t, 3>& a) noexcept {
  return std::sqrt(dot(a, a));
}

std::array<real_t, 3> unit_or_default(std::array<real_t, 3> n) noexcept {
  const real_t len = norm(n);
  if (len <= real_t{1.0e-30}) {
    return {{1.0, 0.0, 0.0}};
  }
  return scale(n, real_t{1.0} / len);
}

gid_t entity_gid(const MeshBase& mesh, CutEntityKind kind, index_t entity) {
  if (entity < 0) {
    return INVALID_GID;
  }
  const auto i = static_cast<std::size_t>(entity);
  switch (kind) {
    case CutEntityKind::Cell:
      return i < mesh.cell_gids().size() ? mesh.cell_gids()[i] : static_cast<gid_t>(entity);
    case CutEntityKind::Face:
      return i < mesh.face_gids().size() ? mesh.face_gids()[i] : static_cast<gid_t>(entity);
    case CutEntityKind::Edge:
      return i < mesh.edge_gids().size() ? mesh.edge_gids()[i] : static_cast<gid_t>(entity);
  }
  return static_cast<gid_t>(entity);
}

CutEntityRecord make_record(CutEntityKind kind,
                            index_t entity,
                            gid_t global_id,
                            rank_t owner_rank,
                            const std::vector<index_t>& dofs,
                            const std::vector<real_t>& signed_distances,
                            const std::vector<CutIntersectionPoint>& intersections,
                            real_t tolerance,
                            EmbeddedRegionProvenance provenance) {
  CutEntityRecord record;
  record.kind = kind;
  record.entity = entity;
  record.global_id = global_id;
  record.owner_rank = owner_rank;
  record.classification = classify_signed_distances(signed_distances, tolerance);
  if (!signed_distances.empty()) {
    const auto minmax = std::minmax_element(signed_distances.begin(), signed_distances.end());
    record.min_signed_distance = *minmax.first;
    record.max_signed_distance = *minmax.second;
  }
  record.intersections = intersections;
  record.provenance = std::move(provenance);
  (void)dofs;
  return record;
}

std::vector<CutIntersectionPoint> edge_intersections(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    const std::vector<index_t>& dofs,
    const std::vector<std::array<index_t, 2>>& local_edges,
    const std::vector<real_t>& signed_distances,
    Configuration cfg,
    real_t tolerance) {
  std::vector<CutIntersectionPoint> out;
  for (const auto& e : local_edges) {
    const auto ia = static_cast<std::size_t>(e[0]);
    const auto ib = static_cast<std::size_t>(e[1]);
    if (ia >= dofs.size() || ib >= dofs.size()) {
      continue;
    }
    const real_t da = signed_distances[ia];
    const real_t db = signed_distances[ib];
    if ((da > tolerance && db > tolerance) || (da < -tolerance && db < -tolerance)) {
      continue;
    }
    if (std::abs(da - db) <= tolerance && std::abs(da) > tolerance) {
      continue;
    }
    const real_t denom = da - db;
    real_t t = std::abs(denom) <= tolerance ? real_t{0.0} : da / denom;
    t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
    const auto pa = mesh.geometry_dof_coords(dofs[ia], cfg);
    const auto pb = mesh.geometry_dof_coords(dofs[ib], cfg);
    const auto p = add(scale(pa, real_t{1.0} - t), scale(pb, t));

    CutIntersectionPoint hit;
    hit.point = p;
    hit.normal = embedded.outward_normal(p);
    hit.edge_fraction = t;
    hit.endpoint_a = dofs[ia];
    hit.endpoint_b = dofs[ib];
    out.push_back(hit);
  }
  return out;
}

std::vector<std::array<index_t, 2>> cyclic_edges(std::size_t n) {
  std::vector<std::array<index_t, 2>> edges;
  if (n < 2u) {
    return edges;
  }
  edges.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    edges.push_back({{static_cast<index_t>(i), static_cast<index_t>((i + 1u) % n)}});
  }
  return edges;
}

std::vector<real_t> signed_distances_for_dofs(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    const std::vector<index_t>& dofs,
    Configuration cfg) {
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const index_t dof : dofs) {
    distances.push_back(embedded.signed_distance(mesh.geometry_dof_coords(dof, cfg)));
  }
  return distances;
}

std::vector<index_t> span_to_vector(std::pair<const index_t*, std::size_t> span) {
  std::vector<index_t> out;
  out.reserve(span.second);
  for (std::size_t i = 0; i < span.second; ++i) {
    out.push_back(span.first[i]);
  }
  return out;
}

std::uint64_t max_constraint_epoch(const std::vector<EmbeddedKinematicConstraint>& constraints) noexcept {
  std::uint64_t epoch = 0;
  for (const auto& c : constraints) {
    epoch = std::max(epoch, c.constraint_epoch);
  }
  return epoch;
}

void update_distributed_owners(const DistributedMesh& mesh, CutClassificationMap& map) {
  for (auto& record : map.cells) {
    record.owner_rank = mesh.owner_rank_cell(record.entity);
  }
  for (auto& record : map.faces) {
    record.owner_rank = mesh.owner_rank_face(record.entity);
  }
  for (auto& record : map.edges) {
    record.owner_rank = mesh.owner_rank_edge(record.entity);
  }
}

} // namespace

EmbeddedRevisionSnapshot EmbeddedRevisionSnapshot::capture(
    const MeshBase& mesh,
    Configuration configuration,
    std::uint64_t embedded_geometry_epoch,
    std::uint64_t embedded_constraint_epoch,
    std::uint64_t fe_layout_revision) {
  EmbeddedRevisionSnapshot snapshot;
  snapshot.configuration = normalized_configuration(configuration);
  snapshot.geometry_revision = mesh.geometry_revision();
  snapshot.topology_revision = mesh.topology_revision();
  snapshot.ownership_revision = mesh.ownership_revision();
  snapshot.numbering_revision = mesh.numbering_revision();
  snapshot.label_revision = mesh.label_revision();
  snapshot.active_configuration_epoch = mesh.active_configuration_epoch();
  snapshot.embedded_geometry_epoch = embedded_geometry_epoch;
  snapshot.embedded_constraint_epoch = embedded_constraint_epoch;
  snapshot.fe_layout_revision = fe_layout_revision;
  return snapshot;
}

bool EmbeddedRevisionSnapshot::matches(
    const MeshBase& mesh,
    Configuration configuration,
    std::uint64_t embedded_geometry_epoch,
    std::uint64_t embedded_constraint_epoch,
    std::uint64_t fe_layout_revision) const noexcept {
  return this->configuration == normalized_configuration(configuration) &&
         geometry_revision == mesh.geometry_revision() &&
         topology_revision == mesh.topology_revision() &&
         ownership_revision == mesh.ownership_revision() &&
         numbering_revision == mesh.numbering_revision() &&
         label_revision == mesh.label_revision() &&
         active_configuration_epoch == mesh.active_configuration_epoch() &&
         this->embedded_geometry_epoch == embedded_geometry_epoch &&
         this->embedded_constraint_epoch == embedded_constraint_epoch &&
         this->fe_layout_revision == fe_layout_revision;
}

std::uint64_t EmbeddedRevisionSnapshot::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, static_cast<std::uint64_t>(configuration));
  h = append_hash(h, geometry_revision);
  h = append_hash(h, topology_revision);
  h = append_hash(h, ownership_revision);
  h = append_hash(h, numbering_revision);
  h = append_hash(h, label_revision);
  h = append_hash(h, active_configuration_epoch);
  h = append_hash(h, embedded_geometry_epoch);
  h = append_hash(h, embedded_constraint_epoch);
  h = append_hash(h, fe_layout_revision);
  return h;
}

real_t EmbeddedGeometryDescriptor::signed_distance(
    const std::array<real_t, 3>& point) const noexcept {
  if (kind == EmbeddedGeometryKind::Sphere) {
    return norm(sub(point, origin)) - radius;
  }
  return dot(sub(point, origin), unit_or_default(normal));
}

std::array<real_t, 3> EmbeddedGeometryDescriptor::outward_normal(
    const std::array<real_t, 3>& point) const noexcept {
  if (kind == EmbeddedGeometryKind::Sphere) {
    return unit_or_default(sub(point, origin));
  }
  return unit_or_default(normal);
}

bool CutClassificationMap::valid_for(const MeshBase& mesh) const noexcept {
  return revision.matches(mesh,
                          options.configuration,
                          embedded_geometry.geometry_epoch,
                          max_constraint_epoch(kinematic_constraints),
                          options.fe_layout_revision);
}

std::uint64_t CutClassificationMap::revision_key() const noexcept {
  std::uint64_t h = revision.revision_key();
  h = append_hash(h, static_cast<std::uint64_t>(embedded_geometry.kind));
  h = append_hash(h, static_cast<std::uint64_t>(cells.size()));
  h = append_hash(h, static_cast<std::uint64_t>(faces.size()));
  h = append_hash(h, static_cast<std::uint64_t>(edges.size()));
  h = append_hash(h, static_cast<std::uint64_t>(kinematic_constraints.size()));
  return h;
}

void CutClassificationMap::accept_trial() noexcept {
  if (state == CutClassificationState::Trial) {
    state = CutClassificationState::Committed;
  }
}

void CutClassificationMap::rollback_trial() {
  if (state == CutClassificationState::Trial) {
    cells.clear();
    faces.clear();
    edges.clear();
    state = CutClassificationState::RolledBack;
  }
}

CutClassificationTransaction::CutClassificationTransaction(CutClassificationMap& map)
    : map_(&map)
    , backup_(map)
    , state_(CutClassificationState::Trial) {}

void CutClassificationTransaction::stage(CutClassificationMap next) {
  if (!map_) {
    throw std::runtime_error("CutClassificationTransaction::stage: transaction has no map");
  }
  next.state = CutClassificationState::Trial;
  *map_ = std::move(next);
  state_ = CutClassificationState::Trial;
}

void CutClassificationTransaction::accept() {
  if (map_) {
    map_->accept_trial();
  }
  state_ = CutClassificationState::Committed;
}

void CutClassificationTransaction::rollback() {
  if (map_) {
    *map_ = backup_;
    map_->state = CutClassificationState::RolledBack;
  }
  state_ = CutClassificationState::RolledBack;
}

CutClassification classify_signed_distances(
    const std::vector<real_t>& signed_distances,
    real_t tolerance) noexcept {
  if (signed_distances.empty()) {
    return CutClassification::Degenerate;
  }
  bool has_negative = false;
  bool has_positive = false;
  bool has_zero = false;
  for (const real_t d : signed_distances) {
    has_negative = has_negative || d < -tolerance;
    has_positive = has_positive || d > tolerance;
    has_zero = has_zero || std::abs(d) <= tolerance;
  }
  if (has_negative && has_positive) {
    return CutClassification::Cut;
  }
  if (has_zero && (has_negative || has_positive)) {
    return CutClassification::Cut;
  }
  if (has_zero) {
    return CutClassification::Degenerate;
  }
  return has_negative ? CutClassification::Negative : CutClassification::Positive;
}

CutClassificationMap classify_embedded_geometry(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const CutClassificationOptions& options) {
  CutClassificationMap map;
  map.name = embedded_geometry.provenance.name.empty()
                 ? embedded_geometry.provenance.persistent_id
                 : embedded_geometry.provenance.name;
  map.embedded_geometry = embedded_geometry;
  map.options = options;
  map.kinematic_constraints = options.kinematic_constraints;
  map.state = CutClassificationState::Trial;
  map.revision = EmbeddedRevisionSnapshot::capture(mesh,
                                                   options.configuration,
                                                   embedded_geometry.geometry_epoch,
                                                   max_constraint_epoch(map.kinematic_constraints),
                                                   options.fe_layout_revision);

  const Configuration cfg = normalized_configuration(options.configuration);

  if (options.classify_cells) {
    map.cells.reserve(mesh.n_cells());
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
      const auto dofs = mesh.cell_geometry_dofs(c);
      const auto distances = signed_distances_for_dofs(mesh, embedded_geometry, dofs, cfg);
      const auto eview = CellTopology::get_edges_view(mesh.cell_shape(c).family);
      std::vector<std::array<index_t, 2>> local_edges;
      local_edges.reserve(static_cast<std::size_t>(std::max(eview.edge_count, 0)));
      for (int e = 0; e < eview.edge_count; ++e) {
        local_edges.push_back({{eview.pairs_flat[2 * e], eview.pairs_flat[2 * e + 1]}});
      }
      const auto intersections = edge_intersections(
          mesh, embedded_geometry, dofs, local_edges, distances, cfg, options.tolerance);
      map.cells.push_back(make_record(CutEntityKind::Cell,
                                      c,
                                      entity_gid(mesh, CutEntityKind::Cell, c),
                                      0,
                                      dofs,
                                      distances,
                                      intersections,
                                      options.tolerance,
                                      embedded_geometry.provenance));
    }
  }

  if (options.classify_faces) {
    map.faces.reserve(mesh.n_faces());
    for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
      auto dofs = span_to_vector(mesh.face_vertices_span(f));
      const auto& face_shape = mesh.face_shapes().at(static_cast<std::size_t>(f));
      if (face_shape.num_corners > 0 &&
          static_cast<std::size_t>(face_shape.num_corners) < dofs.size()) {
        dofs.resize(static_cast<std::size_t>(face_shape.num_corners));
      }
      const auto distances = signed_distances_for_dofs(mesh, embedded_geometry, dofs, cfg);
      const auto intersections = edge_intersections(
          mesh, embedded_geometry, dofs, cyclic_edges(dofs.size()), distances, cfg, options.tolerance);
      map.faces.push_back(make_record(CutEntityKind::Face,
                                      f,
                                      entity_gid(mesh, CutEntityKind::Face, f),
                                      0,
                                      dofs,
                                      distances,
                                      intersections,
                                      options.tolerance,
                                      embedded_geometry.provenance));
    }
  }

  if (options.classify_edges) {
    map.edges.reserve(mesh.n_edges());
    for (index_t e = 0; e < static_cast<index_t>(mesh.n_edges()); ++e) {
      const auto ev = mesh.edge_vertices(e);
      const std::vector<index_t> dofs{ev[0], ev[1]};
      const auto distances = signed_distances_for_dofs(mesh, embedded_geometry, dofs, cfg);
      const auto intersections = edge_intersections(
          mesh,
          embedded_geometry,
          dofs,
          std::vector<std::array<index_t, 2>>{{{0, 1}}},
          distances,
          cfg,
          options.tolerance);
      map.edges.push_back(make_record(CutEntityKind::Edge,
                                      e,
                                      entity_gid(mesh, CutEntityKind::Edge, e),
                                      0,
                                      dofs,
                                      distances,
                                      intersections,
                                      options.tolerance,
                                      embedded_geometry.provenance));
    }
  }

  return map;
}

CutClassificationMap classify_embedded_geometry(
    const DistributedMesh& mesh,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const CutClassificationOptions& options) {
  auto map = classify_embedded_geometry(mesh.local_mesh(), embedded_geometry, options);
  update_distributed_owners(mesh, map);
  return map;
}

CutClassificationRestartRecord make_cut_classification_restart_record(
    const CutClassificationMap& map) {
  CutClassificationRestartRecord record;
  record.name = map.name;
  record.provenance = map.embedded_geometry.provenance;
  record.embedded_kind = map.embedded_geometry.kind;
  record.revision_key = map.revision_key();
  record.embedded_geometry_epoch = map.embedded_geometry.geometry_epoch;
  record.embedded_constraint_epoch = max_constraint_epoch(map.kinematic_constraints);
  record.fe_layout_revision = map.options.fe_layout_revision;
  record.cut_cell_count = static_cast<std::size_t>(std::count_if(
      map.cells.begin(), map.cells.end(), [](const auto& r) { return r.classification == CutClassification::Cut; }));
  record.cut_face_count = static_cast<std::size_t>(std::count_if(
      map.faces.begin(), map.faces.end(), [](const auto& r) { return r.classification == CutClassification::Cut; }));
  record.cut_edge_count = static_cast<std::size_t>(std::count_if(
      map.edges.begin(), map.edges.end(), [](const auto& r) { return r.classification == CutClassification::Cut; }));
  return record;
}

} // namespace search
} // namespace svmp
