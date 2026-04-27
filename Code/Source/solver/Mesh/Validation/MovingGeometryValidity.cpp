/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MovingGeometryValidity.h"

#include "../Core/DistributedMesh.h"
#include "../Core/MeshBase.h"
#include "../Search/SearchPrimitives.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace validation {
namespace {

struct EntityRecord {
  EntityKind kind = EntityKind::Face;
  index_t local_id = INVALID_INDEX;
  gid_t gid = INVALID_GID;
  label_t label = INVALID_LABEL;
  std::vector<index_t> vertices;
  std::vector<std::array<real_t, 3>> x;
  std::vector<std::array<real_t, 3>> x_ref;
  std::array<real_t, 3> normal{{0.0, 0.0, 1.0}};
  std::array<real_t, 3> ref_normal{{0.0, 0.0, 1.0}};
  std::array<real_t, 3> center{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> min{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> max{{0.0, 0.0, 0.0}};
  real_t measure = 0.0;
  int num_corners = 0;
  int order = 1;
};

struct PackedEntity {
  double data[24]{};
};

std::uint64_t fnv_append(std::uint64_t h, std::uint64_t x)
{
  constexpr std::uint64_t prime = 1099511628211ULL;
  h ^= x;
  h *= prime;
  return h;
}

std::uint64_t canonical_key(const char* check, gid_t a, gid_t b = INVALID_GID)
{
  std::uint64_t h = 1469598103934665603ULL;
  for (const char* p = check; *p; ++p) {
    h = fnv_append(h, static_cast<unsigned char>(*p));
  }
  if (b != INVALID_GID && b < a) {
    std::swap(a, b);
  }
  h = fnv_append(h, static_cast<std::uint64_t>(a));
  h = fnv_append(h, static_cast<std::uint64_t>(b));
  return h;
}

struct IndexVectorHash {
  std::size_t operator()(const std::vector<index_t>& values) const noexcept
  {
    std::size_t h = values.size() + 0x9e3779b97f4a7c15ULL;
    for (const auto value : values) {
      const auto v = static_cast<std::size_t>(value);
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
    }
    return h;
  }
};

struct EdgeIndexKey {
  index_t a{INVALID_INDEX};
  index_t b{INVALID_INDEX};

  bool operator==(const EdgeIndexKey& other) const noexcept
  {
    return a == other.a && b == other.b;
  }
};

struct EdgeIndexKeyHash {
  std::size_t operator()(const EdgeIndexKey& key) const noexcept
  {
    std::size_t h = static_cast<std::size_t>(key.a) + 0x9e3779b97f4a7c15ULL;
    const auto b = static_cast<std::size_t>(key.b);
    h ^= b + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
    return h;
  }
};

std::array<real_t, 3> sub(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b)
{
  return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<real_t, 3> add(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b)
{
  return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

std::array<real_t, 3> scale(const std::array<real_t, 3>& a, real_t s)
{
  return {{a[0] * s, a[1] * s, a[2] * s}};
}

real_t dot(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<real_t, 3> cross(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b)
{
  return {{a[1] * b[2] - a[2] * b[1],
           a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]}};
}

real_t norm(const std::array<real_t, 3>& a)
{
  return std::sqrt(dot(a, a));
}

std::array<real_t, 3> normalized(const std::array<real_t, 3>& a)
{
  const real_t n = norm(a);
  if (n <= std::numeric_limits<real_t>::epsilon()) {
    return {{0.0, 0.0, 0.0}};
  }
  return scale(a, 1.0 / n);
}

std::array<real_t, 3> coords(const MeshBase& mesh, index_t v, Configuration cfg)
{
  const int dim = mesh.dim();
  const bool use_current =
      (cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords();
  const auto& x = use_current ? mesh.X_cur() : mesh.X_ref();
  std::array<real_t, 3> out{{0.0, 0.0, 0.0}};
  if (v < 0 || static_cast<std::size_t>(v) >= mesh.n_vertices()) {
    return out;
  }
  for (int d = 0; d < dim && d < 3; ++d) {
    out[static_cast<std::size_t>(d)] = x[static_cast<std::size_t>(v) * dim + static_cast<std::size_t>(d)];
  }
  return out;
}

bool finite3(const std::array<real_t, 3>& x)
{
  return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}

void include_point(EntityRecord& e, const std::array<real_t, 3>& p)
{
  if (e.x.size() == 1) {
    e.min = p;
    e.max = p;
    e.center = p;
  } else {
    for (int d = 0; d < 3; ++d) {
      e.min[d] = std::min(e.min[d], p[d]);
      e.max[d] = std::max(e.max[d], p[d]);
      e.center[d] += p[d];
    }
  }
}

void finalize_entity(EntityRecord& e)
{
  if (!e.x.empty()) {
    for (int d = 0; d < 3; ++d) {
      e.center[d] /= static_cast<real_t>(e.x.size());
    }
  }
  if (e.x.size() >= 3) {
    e.normal = normalized(cross(sub(e.x[1], e.x[0]), sub(e.x[2], e.x[0])));
    e.ref_normal = normalized(cross(sub(e.x_ref[1], e.x_ref[0]), sub(e.x_ref[2], e.x_ref[0])));
    real_t area = 0.0;
    for (std::size_t i = 1; i + 1 < e.x.size(); ++i) {
      area += 0.5 * norm(cross(sub(e.x[i], e.x[0]), sub(e.x[i + 1], e.x[0])));
    }
    e.measure = area;
  } else if (e.x.size() == 2) {
    e.measure = norm(sub(e.x[1], e.x[0]));
  } else {
    e.measure = 0.0;
  }
}

bool labels_match(const MovingGeometryCheckSpec& spec, label_t label)
{
  return spec.labels.empty() || spec.labels.count(label) > 0;
}

bool pair_labels_match(const MovingGeometryCheckSpec& spec, label_t a, label_t b)
{
  if (!spec.label_pairs.empty()) {
    return std::any_of(spec.label_pairs.begin(), spec.label_pairs.end(),
                       [a, b](const LabelPairScope& scope) { return scope.matches(a, b); });
  }
  if (!spec.labels.empty()) {
    return spec.labels.count(a) > 0 && spec.labels.count(b) > 0;
  }
  return true;
}

bool bbox_possible(const EntityRecord& a, const EntityRecord& b, real_t padding)
{
  for (int d = 0; d < 3; ++d) {
    if (a.max[d] + padding < b.min[d] || b.max[d] + padding < a.min[d]) {
      return false;
    }
  }
  return true;
}

real_t bbox_distance(const EntityRecord& a, const EntityRecord& b)
{
  real_t sq = 0.0;
  for (int d = 0; d < 3; ++d) {
    real_t delta = 0.0;
    if (a.max[d] < b.min[d]) {
      delta = b.min[d] - a.max[d];
    } else if (b.max[d] < a.min[d]) {
      delta = a.min[d] - b.max[d];
    }
    sq += delta * delta;
  }
  return std::sqrt(sq);
}

bool share_vertex(const EntityRecord& a, const EntityRecord& b)
{
  for (const auto va : a.vertices) {
    for (const auto vb : b.vertices) {
      if (va == vb) {
        return true;
      }
    }
  }
  return false;
}

real_t face_distance(const EntityRecord& a, const EntityRecord& b)
{
  real_t best = std::numeric_limits<real_t>::infinity();
  if (a.x.size() == 2 && b.x.size() == 2) {
    std::array<real_t, 3> ca;
    best = search::point_segment_distance(a.x[0], b.x[0], b.x[1], &ca);
    best = std::min(best, search::point_segment_distance(a.x[1], b.x[0], b.x[1], nullptr));
    best = std::min(best, search::point_segment_distance(b.x[0], a.x[0], a.x[1], nullptr));
    best = std::min(best, search::point_segment_distance(b.x[1], a.x[0], a.x[1], nullptr));
    return best;
  }

  const auto point_to_poly = [](const std::array<real_t, 3>& p, const EntityRecord& poly) {
    real_t local_best = std::numeric_limits<real_t>::infinity();
    if (poly.x.size() == 2) {
      return search::point_segment_distance(p, poly.x[0], poly.x[1], nullptr);
    }
    for (std::size_t i = 1; i + 1 < poly.x.size(); ++i) {
      local_best = std::min(local_best,
                            search::point_triangle_distance(p, poly.x[0], poly.x[i], poly.x[i + 1], nullptr));
    }
    return local_best;
  };

  for (const auto& p : a.x) {
    best = std::min(best, point_to_poly(p, b));
  }
  for (const auto& p : b.x) {
    best = std::min(best, point_to_poly(p, a));
  }
  return best;
}

bool segments_intersect_2d_tol(const std::array<real_t, 3>& a0,
                               const std::array<real_t, 3>& a1,
                               const std::array<real_t, 3>& b0,
                               const std::array<real_t, 3>& b1,
                               real_t tol)
{
  const auto orient = [](const std::array<real_t, 3>& p,
                         const std::array<real_t, 3>& q,
                         const std::array<real_t, 3>& r) {
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  };
  const auto on_segment = [tol](const std::array<real_t, 3>& p,
                                const std::array<real_t, 3>& q,
                                const std::array<real_t, 3>& r) {
    return q[0] >= std::min(p[0], r[0]) - tol && q[0] <= std::max(p[0], r[0]) + tol &&
           q[1] >= std::min(p[1], r[1]) - tol && q[1] <= std::max(p[1], r[1]) + tol;
  };

  const real_t o1 = orient(a0, a1, b0);
  const real_t o2 = orient(a0, a1, b1);
  const real_t o3 = orient(b0, b1, a0);
  const real_t o4 = orient(b0, b1, a1);

  if (((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
      ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol))) {
    return true;
  }
  if (std::abs(o1) <= tol && on_segment(a0, b0, a1)) return true;
  if (std::abs(o2) <= tol && on_segment(a0, b1, a1)) return true;
  if (std::abs(o3) <= tol && on_segment(b0, a0, b1)) return true;
  if (std::abs(o4) <= tol && on_segment(b0, a1, b1)) return true;
  return false;
}

bool segment_triangle_intersects(const std::array<real_t, 3>& p0,
                                 const std::array<real_t, 3>& p1,
                                 const std::array<real_t, 3>& a,
                                 const std::array<real_t, 3>& b,
                                 const std::array<real_t, 3>& c,
                                 real_t tol)
{
  const auto dir = sub(p1, p0);
  const real_t len = norm(dir);
  if (len <= tol) {
    return false;
  }
  search::Ray ray(p0, scale(dir, 1.0 / len), 0.0, len);
  ray.max_t = len;
  real_t t = 0.0;
  return search::ray_triangle_intersect(ray, a, b, c, t, nullptr);
}

bool face_edge_intersection(const EntityRecord& a, const EntityRecord& b, real_t tol)
{
  if (a.x.size() < 2 || b.x.size() < 3) {
    return false;
  }
  for (std::size_t ei = 0; ei < a.x.size(); ++ei) {
    const auto& p0 = a.x[ei];
    const auto& p1 = a.x[(ei + 1) % a.x.size()];
    for (std::size_t ti = 1; ti + 1 < b.x.size(); ++ti) {
      if (segment_triangle_intersects(p0, p1, b.x[0], b.x[ti], b.x[ti + 1], tol)) {
        return true;
      }
    }
  }
  return false;
}

bool faces_intersect(const EntityRecord& a, const EntityRecord& b, real_t tol)
{
  if (a.x.size() == 2 && b.x.size() == 2) {
    return segments_intersect_2d_tol(a.x[0], a.x[1], b.x[0], b.x[1], tol);
  }
  return face_edge_intersection(a, b, tol) || face_edge_intersection(b, a, tol);
}

std::vector<EntityRecord> collect_face_entities(const MeshBase& mesh, const MovingGeometryValidityPolicy& policy)
{
  std::vector<EntityRecord> out;
  out.reserve(mesh.n_faces());
  const auto& face_gids = mesh.face_gids();
  const auto& shapes = mesh.face_shapes();
  for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
    auto [ptr, n] = mesh.face_vertices_span(f);
    if (!ptr || n < 2) {
      continue;
    }
    EntityRecord e;
    e.kind = EntityKind::Face;
    e.local_id = f;
    e.gid = (static_cast<std::size_t>(f) < face_gids.size()) ? face_gids[static_cast<std::size_t>(f)]
                                                            : static_cast<gid_t>(f);
    e.label = mesh.boundary_label(f);
    e.num_corners = static_cast<int>(n);
    if (static_cast<std::size_t>(f) < shapes.size()) {
      e.num_corners = shapes[static_cast<std::size_t>(f)].num_corners > 0
                          ? std::min(shapes[static_cast<std::size_t>(f)].num_corners, static_cast<int>(n))
                          : static_cast<int>(n);
      e.order = shapes[static_cast<std::size_t>(f)].order;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const index_t v = ptr[i];
      e.vertices.push_back(v);
      e.x.push_back(coords(mesh, v, policy.configuration));
      e.x_ref.push_back(coords(mesh, v, Configuration::Reference));
      include_point(e, e.x.back());
    }
    finalize_entity(e);
    out.push_back(std::move(e));
  }
  return out;
}

std::vector<EntityRecord> collect_edge_entities(const MeshBase& mesh, const MovingGeometryValidityPolicy& policy)
{
  std::vector<EntityRecord> out;
  out.reserve(mesh.n_edges());
  const auto& edge_gids = mesh.edge_gids();
  for (index_t e_id = 0; e_id < static_cast<index_t>(mesh.n_edges()); ++e_id) {
    const auto ev = mesh.edge_vertices(e_id);
    EntityRecord e;
    e.kind = EntityKind::Edge;
    e.local_id = e_id;
    e.gid = (static_cast<std::size_t>(e_id) < edge_gids.size()) ? edge_gids[static_cast<std::size_t>(e_id)]
                                                               : static_cast<gid_t>(e_id);
    e.label = mesh.edge_label(e_id);
    e.vertices = {ev[0], ev[1]};
    for (const auto v : e.vertices) {
      e.x.push_back(coords(mesh, v, policy.configuration));
      e.x_ref.push_back(coords(mesh, v, Configuration::Reference));
      include_point(e, e.x.back());
    }
    finalize_entity(e);
    out.push_back(std::move(e));
  }
  return out;
}

std::vector<EntityRecord> face_loop_edges(const std::vector<EntityRecord>& faces)
{
  std::vector<EntityRecord> out;
  for (const auto& face : faces) {
    const std::size_t nc = static_cast<std::size_t>(std::max(face.num_corners, 0));
    if (nc < 2 || face.x.size() < nc) {
      continue;
    }
    for (std::size_t i = 0; i < nc; ++i) {
      const std::size_t j = (i + 1) % nc;
      EntityRecord edge;
      edge.kind = EntityKind::Edge;
      edge.local_id = face.local_id;
      edge.gid = face.gid;
      edge.label = face.label;
      edge.vertices = {face.vertices[i], face.vertices[j]};
      edge.x = {face.x[i], face.x[j]};
      edge.x_ref = {face.x_ref[i], face.x_ref[j]};
      include_point(edge, edge.x[0]);
      include_point(edge, edge.x[1]);
      finalize_entity(edge);
      out.push_back(std::move(edge));
    }
  }
  return out;
}

std::vector<std::pair<std::size_t, std::size_t>>
sweep_pairs(const std::vector<EntityRecord>& entities,
            const MovingGeometryCheckSpec& spec,
            real_t padding,
            std::size_t& broad_count)
{
  std::vector<std::size_t> order(entities.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&entities](std::size_t a, std::size_t b) {
    return entities[a].min[0] < entities[b].min[0];
  });

  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t oi = 0; oi < order.size(); ++oi) {
    const std::size_t i = order[oi];
    if (!labels_match(spec, entities[i].label)) {
      continue;
    }
    for (std::size_t oj = oi + 1; oj < order.size(); ++oj) {
      const std::size_t j = order[oj];
      if (entities[j].min[0] > entities[i].max[0] + padding) {
        break;
      }
      if (!labels_match(spec, entities[j].label) ||
          !pair_labels_match(spec, entities[i].label, entities[j].label)) {
        continue;
      }
      ++broad_count;
      if (bbox_possible(entities[i], entities[j], padding)) {
        pairs.emplace_back(i, j);
      }
    }
  }
  return pairs;
}

MovingGeometryValidityFailure make_failure(const MovingGeometryCheckSpec& spec,
                                           const MeshBase& mesh,
                                           const MovingGeometryValidityPolicy& policy,
                                           EntityKind kind,
                                           std::vector<index_t> local_ids,
                                           std::vector<gid_t> gids,
                                           std::vector<label_t> labels,
                                           real_t measured,
                                           real_t threshold,
                                           std::string message,
                                           int owner_rank = 0)
{
  MovingGeometryValidityFailure f;
  f.check_name = spec.name.empty() ? MovingGeometryValidity::check_name(spec.check) : spec.name;
  f.message = std::move(message);
  f.severity = spec.severity;
  f.recommended_action = spec.action;
  f.entity_kind = kind;
  f.local_ids = std::move(local_ids);
  f.global_ids = std::move(gids);
  f.labels = std::move(labels);
  f.measured_value = measured;
  f.threshold = threshold;
  f.configuration = policy.configuration;
  f.revision_state = mesh.revision_state();
  f.time_level = policy.time_level;
  f.owner_rank = owner_rank;
  f.canonical_key = !f.global_ids.empty()
                        ? canonical_key(f.check_name.c_str(), f.global_ids[0],
                                        f.global_ids.size() > 1 ? f.global_ids[1] : INVALID_GID)
                        : canonical_key(f.check_name.c_str(), 0, 0);
  return f;
}

void check_degenerate_boundary(const MeshBase& mesh,
                               const MovingGeometryValidityPolicy& policy,
                               const MovingGeometryCheckSpec& spec,
                               const std::vector<EntityRecord>& faces,
                               const std::vector<EntityRecord>& edges,
                               MovingGeometryValidityReport& report)
{
  const auto& vertex_gids = mesh.vertex_gids();
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const auto p = coords(mesh, v, policy.configuration);
    if (!finite3(p)) {
      report.add_failure(make_failure(
          spec, mesh, policy, EntityKind::Vertex, {v},
          {static_cast<std::size_t>(v) < vertex_gids.size() ? vertex_gids[static_cast<std::size_t>(v)]
                                                            : static_cast<gid_t>(v)},
          {mesh.vertex_label(v)}, std::numeric_limits<real_t>::quiet_NaN(), 0.0,
          "non-finite moved coordinate"));
    }
  }

  auto all_edges = edges;
  const auto loop_edges = face_loop_edges(faces);
  all_edges.insert(all_edges.end(), loop_edges.begin(), loop_edges.end());
  for (const auto& edge : all_edges) {
    if (!labels_match(spec, edge.label)) continue;
    if (edge.measure <= policy.robust.degenerate_tolerance) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Edge, {edge.local_id}, {edge.gid},
                                      {edge.label}, edge.measure, policy.robust.degenerate_tolerance,
                                      "zero or near-zero moved boundary edge"));
    }
  }

  std::unordered_map<std::vector<index_t>, index_t, IndexVectorHash> face_keys;
  std::unordered_map<EdgeIndexKey, int, EdgeIndexKeyHash> edge_incidence;
  for (const auto& face : faces) {
    if (!labels_match(spec, face.label)) continue;
    if (face.x.size() >= 3 && face.measure <= policy.robust.degenerate_tolerance) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {face.local_id}, {face.gid},
                                      {face.label}, face.measure, policy.robust.degenerate_tolerance,
                                      "zero or near-zero moved boundary face"));
    }

    std::vector<index_t> sorted_vertices = face.vertices;
    std::sort(sorted_vertices.begin(), sorted_vertices.end());
    if (auto it = face_keys.find(sorted_vertices); it != face_keys.end()) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {it->second, face.local_id},
                                      {static_cast<gid_t>(it->second), face.gid}, {face.label},
                                      0.0, 0.0, "duplicate or coincident boundary face"));
    } else {
      face_keys.emplace(std::move(sorted_vertices), face.local_id);
    }

    const std::size_t nc = static_cast<std::size_t>(std::max(face.num_corners, 0));
    for (std::size_t i = 0; i < nc && i < face.vertices.size(); ++i) {
      const auto a = std::min(face.vertices[i], face.vertices[(i + 1) % nc]);
      const auto b = std::max(face.vertices[i], face.vertices[(i + 1) % nc]);
      ++edge_incidence[EdgeIndexKey{a, b}];
    }
  }
  for (const auto& kv : edge_incidence) {
    if (kv.second > 2) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Edge, {INVALID_INDEX}, {INVALID_GID},
                                      {INVALID_LABEL}, static_cast<real_t>(kv.second), 2.0,
                                      "nonmanifold moved boundary edge"));
    }
  }
}

void check_self_intersection(const MeshBase& mesh,
                             const MovingGeometryValidityPolicy& policy,
                             const MovingGeometryCheckSpec& spec,
                             const std::vector<EntityRecord>& faces,
                             MovingGeometryValidityReport& report)
{
  std::size_t broad = 0;
  const auto pairs = sweep_pairs(faces, spec, policy.robust.aabb_padding, broad);
  report.broad_phase_candidate_pairs += broad;
  for (const auto& ij : pairs) {
    const auto& a = faces[ij.first];
    const auto& b = faces[ij.second];
    if (share_vertex(a, b)) continue;
    ++report.exact_candidate_pairs;
    const real_t d = face_distance(a, b);
    if (d <= policy.robust.intersection_tolerance ||
        faces_intersect(a, b, policy.robust.intersection_tolerance)) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {a.local_id, b.local_id},
                                      {a.gid, b.gid}, {a.label, b.label}, d,
                                      policy.robust.intersection_tolerance,
                                      "moved boundary faces intersect or overlap"));
    }
  }
}

void check_2d_boundary(const MeshBase& mesh,
                       const MovingGeometryValidityPolicy& policy,
                       const MovingGeometryCheckSpec& spec,
                       const std::vector<EntityRecord>& faces,
                       const std::vector<EntityRecord>& edges,
                       MovingGeometryValidityReport& report)
{
  auto segments = edges;
  const auto loop_edges = face_loop_edges(faces);
  segments.insert(segments.end(), loop_edges.begin(), loop_edges.end());

  std::size_t broad = 0;
  const auto pairs = sweep_pairs(segments, spec, policy.robust.aabb_padding, broad);
  report.broad_phase_candidate_pairs += broad;
  for (const auto& ij : pairs) {
    const auto& a = segments[ij.first];
    const auto& b = segments[ij.second];
    if (share_vertex(a, b)) continue;
    ++report.exact_candidate_pairs;
    if (segments_intersect_2d_tol(a.x[0], a.x[1], b.x[0], b.x[1],
                                  policy.robust.intersection_tolerance)) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Edge, {a.local_id, b.local_id},
                                      {a.gid, b.gid}, {a.label, b.label}, 0.0,
                                      policy.robust.intersection_tolerance,
                                      "2D moved boundary segments cross"));
    }
  }

  for (const auto& face : faces) {
    if (face.x.size() < 3 || !labels_match(spec, face.label)) continue;
    const auto area2 = [](const std::vector<std::array<real_t, 3>>& pts) {
      real_t a = 0.0;
      for (std::size_t i = 0; i < pts.size(); ++i) {
        const auto& p = pts[i];
        const auto& q = pts[(i + 1) % pts.size()];
        a += p[0] * q[1] - q[0] * p[1];
      }
      return a;
    };
    const real_t ref = area2(face.x_ref);
    const real_t cur = area2(face.x);
    if (std::abs(ref) > policy.robust.degenerate_tolerance &&
        ref * cur < -policy.robust.degenerate_tolerance) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {face.local_id}, {face.gid},
                                      {face.label}, cur, 0.0,
                                      "2D polygon winding or orientation changed"));
    }
  }
}

void check_orientation(const MeshBase& mesh,
                       const MovingGeometryValidityPolicy& policy,
                       const MovingGeometryCheckSpec& spec,
                       const std::vector<EntityRecord>& faces,
                       MovingGeometryValidityReport& report)
{
  for (const auto& face : faces) {
    if (face.x.size() < 3 || !labels_match(spec, face.label)) continue;
    const real_t alignment = dot(face.normal, face.ref_normal);
    if (alignment < -policy.robust.coplanar_tolerance) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {face.local_id}, {face.gid},
                                      {face.label}, alignment, 0.0,
                                      "moved boundary normal/orientation flipped"));
    }
  }
}

void check_curved_sampling(const MeshBase& mesh,
                           const MovingGeometryValidityPolicy& policy,
                           const MovingGeometryCheckSpec& spec,
                           const std::vector<EntityRecord>& faces,
                           MovingGeometryValidityReport& report)
{
  for (const auto& face : faces) {
    if (!labels_match(spec, face.label) || face.x.size() <= static_cast<std::size_t>(face.num_corners) ||
        face.num_corners < 3) {
      continue;
    }
    if (!std::isfinite(spec.threshold)) {
      for (std::size_t i = static_cast<std::size_t>(face.num_corners); i < face.x.size(); ++i) {
        if (!finite3(face.x[i])) {
          report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {face.local_id}, {face.gid},
                                          {face.label}, std::numeric_limits<real_t>::quiet_NaN(), 0.0,
                                          "non-finite high-order geometry DOF"));
        }
      }
      continue;
    }
    const auto n = normalized(cross(sub(face.x[1], face.x[0]), sub(face.x[2], face.x[0])));
    for (std::size_t i = static_cast<std::size_t>(face.num_corners); i < face.x.size(); ++i) {
      const real_t signed_distance = dot(sub(face.x[i], face.x[0]), n);
      if (std::abs(signed_distance) > spec.threshold + policy.robust.curved_sampling_tolerance) {
        report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {face.local_id}, {face.gid},
                                        {face.label}, std::abs(signed_distance), spec.threshold,
                                        "high-order geometry DOF violates curved-boundary sampling tolerance"));
      }
    }
  }
}

void check_separation(const MeshBase& mesh,
                      const MovingGeometryValidityPolicy& policy,
                      const MovingGeometryCheckSpec& spec,
                      const std::vector<EntityRecord>& faces,
                      MovingGeometryValidityReport& report)
{
  std::size_t broad = 0;
  const real_t threshold = std::max(spec.threshold, policy.robust.near_contact_tolerance);
  const auto pairs = sweep_pairs(faces, spec, threshold + policy.robust.aabb_padding, broad);
  report.broad_phase_candidate_pairs += broad;
  for (const auto& ij : pairs) {
    const auto& a = faces[ij.first];
    const auto& b = faces[ij.second];
    if (share_vertex(a, b)) continue;
    ++report.exact_candidate_pairs;
    const real_t d = face_distance(a, b);
    bool fail = d < threshold;
    std::string message = "moved boundary label-pair separation below threshold";
    if (spec.check == MovingGeometryCheck::ContactSeparation && a.x.size() >= 3) {
      const real_t signed_gap = dot(sub(b.center, a.center), a.normal);
      if (signed_gap < -policy.robust.near_contact_tolerance) {
        fail = true;
        message = "contact-oriented moved boundary is overlapped or on the wrong projection side";
      }
    }
    if (fail) {
      report.add_failure(make_failure(spec, mesh, policy, EntityKind::Face, {a.local_id, b.local_id},
                                      {a.gid, b.gid}, {a.label, b.label}, d, threshold,
                                      std::move(message)));
    }
  }
}

void check_swept_volume(const MeshBase& mesh,
                        const MovingGeometryValidityPolicy& policy,
                        const MovingGeometryCheckSpec& spec,
                        const std::vector<EntityRecord>& faces,
                        MovingGeometryValidityReport& report)
{
  if (!mesh.has_current_coords()) {
    return;
  }
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const auto p0 = coords(mesh, v, Configuration::Reference);
    const auto p1 = coords(mesh, v, Configuration::Current);
    if (norm(sub(p1, p0)) <= policy.robust.intersection_tolerance) {
      continue;
    }
    for (const auto& face : faces) {
      if (std::find(face.vertices.begin(), face.vertices.end(), v) != face.vertices.end()) {
        continue;
      }
      if (!labels_match(spec, face.label)) {
        continue;
      }
      ++report.broad_phase_candidate_pairs;
      for (std::size_t i = 1; i + 1 < face.x.size(); ++i) {
        ++report.exact_candidate_pairs;
        if (segment_triangle_intersects(p0, p1, face.x[0], face.x[i], face.x[i + 1],
                                        policy.robust.intersection_tolerance)) {
          const auto& vertex_gids = mesh.vertex_gids();
          report.add_failure(make_failure(
              spec, mesh, policy, EntityKind::Vertex, {v, face.local_id},
              {static_cast<std::size_t>(v) < vertex_gids.size() ? vertex_gids[static_cast<std::size_t>(v)]
                                                                : static_cast<gid_t>(v),
               face.gid},
              {mesh.vertex_label(v), face.label}, 0.0, policy.robust.intersection_tolerance,
              "vertex swept path intersects moved boundary face"));
          break;
        }
      }
    }
  }
}

void check_constraints(const MeshBase& mesh,
                       const MovingGeometryValidityPolicy& policy,
                       MovingGeometryValidityReport& report)
{
  const auto& vertex_gids = mesh.vertex_gids();
  for (const auto& constraint : policy.constraints) {
    MovingGeometryCheckSpec spec;
    spec.check = (constraint.kind == MotionConstraintKind::ManifoldPlane)
                     ? MovingGeometryCheck::ManifoldConstraint
                     : (constraint.kind == MotionConstraintKind::MaximumDisplacement ||
                        constraint.kind == MotionConstraintKind::MinimumSeparation)
                           ? MovingGeometryCheck::ActiveInequality
                           : MovingGeometryCheck::DirectionalConstraint;
    spec.name = constraint.name.empty() ? MovingGeometryValidity::check_name(spec.check) : constraint.name;
    spec.action = constraint.action;
    spec.severity = ValiditySeverity::Error;
    spec.threshold = constraint.threshold;
    spec.physics_neutral_constraint_output = constraint.physics_neutral;

    const auto dir = normalized(constraint.direction);
    for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
      if (constraint.label != INVALID_LABEL && mesh.vertex_label(v) != constraint.label) {
        continue;
      }
      const auto xr = coords(mesh, v, Configuration::Reference);
      const auto xc = coords(mesh, v, policy.configuration);
      const auto u = sub(xc, xr);
      real_t measured = 0.0;
      std::string message;
      bool fail = false;

      switch (constraint.kind) {
        case MotionConstraintKind::ManifoldPlane:
        case MotionConstraintKind::SurfaceSliding:
          measured = std::abs(dot(sub(xc, constraint.point), dir));
          fail = measured > constraint.threshold;
          message = "moved geometry violates manifold-following plane constraint";
          break;
        case MotionConstraintKind::NormalOnly:
          measured = norm(sub(u, scale(dir, dot(u, dir))));
          fail = measured > constraint.threshold;
          message = "moved geometry violates normal-only motion constraint";
          break;
        case MotionConstraintKind::TangentialOnly:
          measured = std::abs(dot(u, dir));
          fail = measured > constraint.threshold;
          message = "moved geometry violates tangential-only motion constraint";
          break;
        case MotionConstraintKind::MaximumDisplacement:
          measured = norm(u);
          fail = measured > constraint.threshold;
          message = "moved geometry violates active maximum-displacement inequality";
          break;
        case MotionConstraintKind::MinimumSeparation:
        case MotionConstraintKind::BoundaryLayerPreservation:
          // Pairwise versions are handled through label-pair separation checks.
          break;
      }
      if (fail) {
        report.add_failure(make_failure(
            spec, mesh, policy, EntityKind::Vertex, {v},
            {static_cast<std::size_t>(v) < vertex_gids.size() ? vertex_gids[static_cast<std::size_t>(v)]
                                                              : static_cast<gid_t>(v)},
            {mesh.vertex_label(v)}, measured, constraint.threshold, std::move(message)));
      }
    }
  }
}

MovingGeometryCheckSpec default_spec(MovingGeometryCheck check,
                                     ValidityAction action = ValidityAction::Reject,
                                     real_t threshold = 0.0)
{
  MovingGeometryCheckSpec spec;
  spec.check = check;
  spec.name = MovingGeometryValidity::check_name(check);
  spec.action = action;
  spec.threshold = threshold;
  spec.severity = ValiditySeverity::Error;
  return spec;
}

std::string cfg_name(Configuration cfg)
{
  switch (cfg) {
    case Configuration::Reference: return "reference";
    case Configuration::Current:
    case Configuration::Deformed: return "current";
  }
  return "unknown";
}

} // namespace

bool LabelPairScope::matches(label_t a, label_t b) const noexcept
{
  if (first == INVALID_LABEL && second == INVALID_LABEL) {
    return true;
  }
  const bool direct = (first == INVALID_LABEL || first == a) &&
                      (second == INVALID_LABEL || second == b);
  if (direct) {
    return true;
  }
  return symmetric && (first == INVALID_LABEL || first == b) &&
         (second == INVALID_LABEL || second == a);
}

bool MovingGeometryValidityPolicy::enables(MovingGeometryCheck check) const noexcept
{
  return spec(check) != nullptr;
}

const MovingGeometryCheckSpec* MovingGeometryValidityPolicy::spec(MovingGeometryCheck check) const noexcept
{
  for (const auto& check_spec : checks) {
    if (check_spec.enabled && check_spec.check == check) {
      return &check_spec;
    }
  }
  return nullptr;
}

std::map<std::string, std::string> MovingGeometryValidityPolicy::restart_metadata() const
{
  std::map<std::string, std::string> out;
  out["policy_group"] = group_name;
  out["configuration"] = cfg_name(configuration);
  out["time_level"] = std::to_string(time_level);
  out["intersection_tolerance"] = std::to_string(robust.intersection_tolerance);
  out["near_contact_tolerance"] = std::to_string(robust.near_contact_tolerance);
  out["coplanar_tolerance"] = std::to_string(robust.coplanar_tolerance);
  out["degenerate_tolerance"] = std::to_string(robust.degenerate_tolerance);
  out["curved_sampling_tolerance"] = std::to_string(robust.curved_sampling_tolerance);
  out["check_count"] = std::to_string(checks.size());
  for (std::size_t i = 0; i < checks.size(); ++i) {
    out["check." + std::to_string(i)] = checks[i].name.empty() ? MovingGeometryValidity::check_name(checks[i].check)
                                                               : checks[i].name;
  }
  return out;
}

void MovingGeometryValidityReport::add_failure(MovingGeometryValidityFailure failure)
{
  passed = false;
  failures.push_back(std::move(failure));
}

bool MovingGeometryValidityReport::requires_rejection() const noexcept
{
  return std::any_of(failures.begin(), failures.end(), [](const auto& f) {
    return f.recommended_action == ValidityAction::Reject ||
           f.recommended_action == ValidityAction::Backtrack;
  });
}

bool MovingGeometryValidityReport::recommends_backtrack() const noexcept
{
  return std::any_of(failures.begin(), failures.end(), [](const auto& f) {
    return f.recommended_action == ValidityAction::Backtrack;
  });
}

bool MovingGeometryValidityReport::provides_constraints() const noexcept
{
  return std::any_of(failures.begin(), failures.end(), [](const auto& f) {
    return f.recommended_action == ValidityAction::Constrain;
  });
}

std::string MovingGeometryValidityReport::to_string() const
{
  std::ostringstream os;
  os << "MovingGeometryValidityReport{policy=" << policy_group_name
     << ", passed=" << (passed ? "true" : "false")
     << ", failures=" << failures.size()
     << ", broad_phase_pairs=" << broad_phase_candidate_pairs
     << ", exact_pairs=" << exact_candidate_pairs << "}";
  for (const auto& f : failures) {
    os << "\n  [" << MovingGeometryValidity::severity_name(f.severity) << "] "
       << f.check_name << " action=" << MovingGeometryValidity::action_name(f.recommended_action)
       << " measured=" << f.measured_value << " threshold=" << f.threshold
       << " ids=";
    for (const auto id : f.global_ids) os << id << ' ';
    os << "message=" << f.message;
  }
  return os.str();
}

std::map<std::string, std::string> MovingGeometryValidityReport::restart_metadata() const
{
  std::map<std::string, std::string> out;
  out["policy_group"] = policy_group_name;
  out["configuration"] = cfg_name(configuration);
  out["time_level"] = std::to_string(time_level);
  out["passed"] = passed ? "true" : "false";
  out["failure_count"] = std::to_string(failures.size());
  out["broad_phase_candidate_pairs"] = std::to_string(broad_phase_candidate_pairs);
  out["exact_candidate_pairs"] = std::to_string(exact_candidate_pairs);
  for (std::size_t i = 0; i < failures.size(); ++i) {
    const auto& f = failures[i];
    const auto prefix = "failure." + std::to_string(i) + ".";
    out[prefix + "check"] = f.check_name;
    out[prefix + "action"] = MovingGeometryValidity::action_name(f.recommended_action);
    out[prefix + "severity"] = MovingGeometryValidity::severity_name(f.severity);
    out[prefix + "measured"] = std::to_string(f.measured_value);
    out[prefix + "threshold"] = std::to_string(f.threshold);
    out[prefix + "message"] = f.message;
  }
  return out;
}

MovingGeometryValidityPolicy MovingGeometryValidity::preset(ValidityPolicyGroup group)
{
  switch (group) {
    case ValidityPolicyGroup::ALEBasic: return ale_basic_policy();
    case ValidityPolicyGroup::Contact: return contact_policy();
    case ValidityPolicyGroup::FreeSurface: return free_surface_policy();
    case ValidityPolicyGroup::Shell: return shell_policy();
    case ValidityPolicyGroup::BoundaryLayer: return boundary_layer_policy();
    case ValidityPolicyGroup::LargeStep: return large_step_policy();
    case ValidityPolicyGroup::Custom: break;
  }
  MovingGeometryValidityPolicy policy;
  policy.group = ValidityPolicyGroup::Custom;
  policy.group_name = "Custom";
  return policy;
}

MovingGeometryValidityPolicy MovingGeometryValidity::ale_basic_policy()
{
  MovingGeometryValidityPolicy policy;
  policy.group = ValidityPolicyGroup::ALEBasic;
  policy.group_name = "ALEBasic";
  policy.checks = {
      default_spec(MovingGeometryCheck::DegenerateBoundary, ValidityAction::Reject),
      default_spec(MovingGeometryCheck::SurfaceFolding, ValidityAction::Backtrack),
      default_spec(MovingGeometryCheck::NormalOrientation, ValidityAction::Backtrack),
      default_spec(MovingGeometryCheck::TwoDBoundary, ValidityAction::Backtrack),
  };
  return policy;
}

MovingGeometryValidityPolicy MovingGeometryValidity::contact_policy()
{
  auto policy = ale_basic_policy();
  policy.group = ValidityPolicyGroup::Contact;
  policy.group_name = "Contact";
  policy.checks.push_back(default_spec(MovingGeometryCheck::BoundarySelfIntersection, ValidityAction::Reject));
  policy.checks.push_back(default_spec(MovingGeometryCheck::MinimumSeparation, ValidityAction::Reject,
                                       policy.robust.near_contact_tolerance));
  policy.checks.push_back(default_spec(MovingGeometryCheck::ContactSeparation, ValidityAction::Reject,
                                       policy.robust.near_contact_tolerance));
  policy.checks.push_back(default_spec(MovingGeometryCheck::CurvedBoundarySampling, ValidityAction::Reject,
                                       std::numeric_limits<real_t>::infinity()));
  return policy;
}

MovingGeometryValidityPolicy MovingGeometryValidity::free_surface_policy()
{
  auto policy = ale_basic_policy();
  policy.group = ValidityPolicyGroup::FreeSurface;
  policy.group_name = "FreeSurface";
  policy.checks.push_back(default_spec(MovingGeometryCheck::BoundarySelfIntersection, ValidityAction::Reject));
  policy.checks.push_back(default_spec(MovingGeometryCheck::MinimumSeparation, ValidityAction::Reject,
                                       policy.robust.near_contact_tolerance));
  return policy;
}

MovingGeometryValidityPolicy MovingGeometryValidity::shell_policy()
{
  auto policy = contact_policy();
  policy.group = ValidityPolicyGroup::Shell;
  policy.group_name = "Shell";
  policy.checks.push_back(default_spec(MovingGeometryCheck::ShellThicknessSeparation, ValidityAction::Reject,
                                       policy.robust.near_contact_tolerance));
  return policy;
}

MovingGeometryValidityPolicy MovingGeometryValidity::boundary_layer_policy()
{
  auto policy = ale_basic_policy();
  policy.group = ValidityPolicyGroup::BoundaryLayer;
  policy.group_name = "BoundaryLayer";
  policy.checks.push_back(default_spec(MovingGeometryCheck::BoundaryLayer, ValidityAction::Backtrack,
                                       policy.robust.near_contact_tolerance));
  return policy;
}

MovingGeometryValidityPolicy MovingGeometryValidity::large_step_policy()
{
  auto policy = free_surface_policy();
  policy.group = ValidityPolicyGroup::LargeStep;
  policy.group_name = "LargeStep";
  policy.checks.push_back(default_spec(MovingGeometryCheck::SweptVolume, ValidityAction::Backtrack));
  return policy;
}

MovingGeometryValidityReport MovingGeometryValidity::evaluate(const MeshBase& mesh,
                                                              const MovingGeometryValidityPolicy& policy)
{
  MovingGeometryValidityReport report;
  report.policy_group_name = policy.group_name;
  report.configuration = policy.configuration;
  report.revision_state = mesh.revision_state();
  report.time_level = policy.time_level;

  const auto faces = collect_face_entities(mesh, policy);
  const auto edges = collect_edge_entities(mesh, policy);

  if (const auto* spec = policy.spec(MovingGeometryCheck::DegenerateBoundary)) {
    check_degenerate_boundary(mesh, policy, *spec, faces, edges, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::BoundarySelfIntersection)) {
    check_self_intersection(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::TwoDBoundary)) {
    check_2d_boundary(mesh, policy, *spec, faces, edges, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::SurfaceFolding)) {
    check_orientation(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::NormalOrientation)) {
    check_orientation(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::MinimumSeparation)) {
    check_separation(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::ContactSeparation)) {
    check_separation(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::ShellThicknessSeparation)) {
    check_separation(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::BoundaryLayer)) {
    check_separation(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::CurvedBoundarySampling)) {
    check_curved_sampling(mesh, policy, *spec, faces, report);
  }
  if (const auto* spec = policy.spec(MovingGeometryCheck::SweptVolume)) {
    check_swept_volume(mesh, policy, *spec, faces, report);
  }
  check_constraints(mesh, policy, report);

  return report;
}

MovingGeometryValidityReport MovingGeometryValidity::evaluate(const DistributedMesh& mesh,
                                                              const MovingGeometryValidityPolicy& policy)
{
  auto report = evaluate(mesh.local_mesh(), policy);

#ifdef MESH_HAS_MPI
  if (mesh.world_size() <= 1) {
    return report;
  }

  const auto faces = collect_face_entities(mesh.local_mesh(), policy);
  std::vector<PackedEntity> local;
  local.reserve(faces.size());
  for (const auto& face : faces) {
    PackedEntity p;
    p.data[0] = static_cast<double>(mesh.rank());
    p.data[1] = static_cast<double>(face.local_id);
    p.data[2] = static_cast<double>(face.gid);
    p.data[3] = static_cast<double>(face.label);
    for (int d = 0; d < 3; ++d) {
      p.data[4 + d] = face.min[d];
      p.data[7 + d] = face.max[d];
      p.data[10 + d] = face.center[d];
      p.data[13 + d] = face.normal[d];
    }
    p.data[16] = face.measure;
    p.data[17] = static_cast<double>(face.vertices.size());
    local.push_back(p);
  }

  int local_count = static_cast<int>(local.size());
  int size = mesh.world_size();
  std::vector<int> counts(static_cast<std::size_t>(size), 0);
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, mesh.mpi_comm());
  std::vector<int> byte_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  int total_count = 0;
  for (int r = 0; r < size; ++r) {
    byte_counts[static_cast<std::size_t>(r)] =
        counts[static_cast<std::size_t>(r)] * static_cast<int>(sizeof(PackedEntity));
    displs[static_cast<std::size_t>(r)] =
        total_count * static_cast<int>(sizeof(PackedEntity));
    total_count += counts[static_cast<std::size_t>(r)];
  }
  std::vector<PackedEntity> all(static_cast<std::size_t>(total_count));
  MPI_Allgatherv(local.empty() ? nullptr : local.data(),
                 local_count * static_cast<int>(sizeof(PackedEntity)), MPI_BYTE,
                 all.empty() ? nullptr : all.data(), byte_counts.data(), displs.data(),
                 MPI_BYTE, mesh.mpi_comm());

  const auto add_cross_rank_failure =
      [&](const MovingGeometryCheckSpec& spec, const PackedEntity& a, const PackedEntity& b,
          real_t measured, real_t threshold, const std::string& message) {
        MovingGeometryValidityFailure f;
        f.check_name = spec.name.empty() ? check_name(spec.check) : spec.name;
        f.message = message;
        f.severity = spec.severity;
        f.recommended_action = spec.action;
        f.entity_kind = EntityKind::Face;
        f.local_ids = {static_cast<index_t>(a.data[1]), static_cast<index_t>(b.data[1])};
        f.global_ids = {static_cast<gid_t>(a.data[2]), static_cast<gid_t>(b.data[2])};
        f.labels = {static_cast<label_t>(a.data[3]), static_cast<label_t>(b.data[3])};
        f.measured_value = measured;
        f.threshold = threshold;
        f.configuration = policy.configuration;
        f.revision_state = mesh.local_mesh().revision_state();
        f.time_level = policy.time_level;
        f.owner_rank = std::min(static_cast<int>(a.data[0]), static_cast<int>(b.data[0]));
        f.canonical_key = canonical_key(f.check_name.c_str(), f.global_ids[0], f.global_ids[1]);
        report.add_failure(std::move(f));
      };

  std::unordered_set<std::uint64_t> seen;
  for (std::size_t i = 0; i < all.size(); ++i) {
    for (std::size_t j = i + 1; j < all.size(); ++j) {
      const int ri = static_cast<int>(all[i].data[0]);
      const int rj = static_cast<int>(all[j].data[0]);
      if (ri == rj) {
        continue;
      }
      const label_t li = static_cast<label_t>(all[i].data[3]);
      const label_t lj = static_cast<label_t>(all[j].data[3]);
      const gid_t gi = static_cast<gid_t>(all[i].data[2]);
      const gid_t gj = static_cast<gid_t>(all[j].data[2]);

      auto bbox_dist = [&]() {
        real_t sq = 0.0;
        for (int d = 0; d < 3; ++d) {
          const real_t ai0 = all[i].data[4 + d];
          const real_t ai1 = all[i].data[7 + d];
          const real_t bj0 = all[j].data[4 + d];
          const real_t bj1 = all[j].data[7 + d];
          real_t delta = 0.0;
          if (ai1 < bj0) delta = bj0 - ai1;
          else if (bj1 < ai0) delta = ai0 - bj1;
          sq += delta * delta;
        }
        return std::sqrt(sq);
      };

      if (const auto* spec = policy.spec(MovingGeometryCheck::BoundarySelfIntersection)) {
        if (pair_labels_match(*spec, li, lj) && bbox_dist() <= policy.robust.intersection_tolerance) {
          const auto key = canonical_key("mpi-self-intersection", gi, gj);
          if (seen.insert(key).second) {
            ++report.broad_phase_candidate_pairs;
            ++report.exact_candidate_pairs;
            add_cross_rank_failure(*spec, all[i], all[j], 0.0, policy.robust.intersection_tolerance,
                                   "cross-rank moved boundary faces intersect or overlap");
          }
        }
      }
      if (const auto* spec = policy.spec(MovingGeometryCheck::MinimumSeparation)) {
        const real_t threshold = std::max(spec->threshold, policy.robust.near_contact_tolerance);
        const real_t d = bbox_dist();
        if (pair_labels_match(*spec, li, lj) && d < threshold) {
          const auto key = canonical_key("mpi-minimum-separation", gi, gj);
          if (seen.insert(key).second) {
            ++report.broad_phase_candidate_pairs;
            ++report.exact_candidate_pairs;
            add_cross_rank_failure(*spec, all[i], all[j], d, threshold,
                                   "cross-rank moved boundary label-pair separation below threshold");
          }
        }
      }
    }
  }
#endif

  return report;
}

const char* MovingGeometryValidity::check_name(MovingGeometryCheck check) noexcept
{
  switch (check) {
    case MovingGeometryCheck::DegenerateBoundary: return "DegenerateBoundary";
    case MovingGeometryCheck::BoundarySelfIntersection: return "BoundarySelfIntersection";
    case MovingGeometryCheck::SurfaceFolding: return "SurfaceFolding";
    case MovingGeometryCheck::NormalOrientation: return "NormalOrientation";
    case MovingGeometryCheck::MinimumSeparation: return "MinimumSeparation";
    case MovingGeometryCheck::ContactSeparation: return "ContactSeparation";
    case MovingGeometryCheck::ShellThicknessSeparation: return "ShellThicknessSeparation";
    case MovingGeometryCheck::CurvedBoundarySampling: return "CurvedBoundarySampling";
    case MovingGeometryCheck::SweptVolume: return "SweptVolume";
    case MovingGeometryCheck::BoundaryLayer: return "BoundaryLayer";
    case MovingGeometryCheck::TwoDBoundary: return "TwoDBoundary";
    case MovingGeometryCheck::ManifoldConstraint: return "ManifoldConstraint";
    case MovingGeometryCheck::DirectionalConstraint: return "DirectionalConstraint";
    case MovingGeometryCheck::ActiveInequality: return "ActiveInequality";
  }
  return "Unknown";
}

const char* MovingGeometryValidity::action_name(ValidityAction action) noexcept
{
  switch (action) {
    case ValidityAction::Warn: return "warn";
    case ValidityAction::Reject: return "reject";
    case ValidityAction::Backtrack: return "backtrack";
    case ValidityAction::Constrain: return "constrain";
  }
  return "reject";
}

const char* MovingGeometryValidity::severity_name(ValiditySeverity severity) noexcept
{
  switch (severity) {
    case ValiditySeverity::Info: return "info";
    case ValiditySeverity::Warning: return "warning";
    case ValiditySeverity::Error: return "error";
  }
  return "error";
}

const char* MovingGeometryValidity::policy_group_name(ValidityPolicyGroup group) noexcept
{
  switch (group) {
    case ValidityPolicyGroup::Custom: return "Custom";
    case ValidityPolicyGroup::ALEBasic: return "ALEBasic";
    case ValidityPolicyGroup::Contact: return "Contact";
    case ValidityPolicyGroup::FreeSurface: return "FreeSurface";
    case ValidityPolicyGroup::Shell: return "Shell";
    case ValidityPolicyGroup::BoundaryLayer: return "BoundaryLayer";
    case ValidityPolicyGroup::LargeStep: return "LargeStep";
  }
  return "Custom";
}

} // namespace validation
} // namespace svmp
