/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "MeshSearch.h"
#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"
#include "../Boundary/BoundaryDetector.h"
#include "../Geometry/CurvilinearEval.h"
#include "../Geometry/MeshGeometry.h"
#include "../Topology/DistributedTopology.h"
#include "SearchAccel.h"
#include "SearchBuilders.h"
#include "SearchPrimitives.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <cmath>
#include <unordered_set>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {

namespace {

Configuration canonical_cfg(Configuration cfg) {
  return (cfg == Configuration::Deformed) ? Configuration::Current : cfg;
}

Configuration effective_cfg(const MeshBase& mesh, Configuration cfg) {
  cfg = canonical_cfg(cfg);
  if (cfg == Configuration::Current && !mesh.has_current_coords()) {
    return Configuration::Reference;
  }
  return cfg;
}

bool configs_match(Configuration built, Configuration requested) {
  return canonical_cfg(built) == canonical_cfg(requested);
}

const IAccel* accel_if_built(const MeshBase& mesh, Configuration cfg) {
  const auto* holder = mesh.search_accel();
  if (!holder || !holder->accel || !holder->accel->is_built()) {
    return nullptr;
  }

  cfg = effective_cfg(mesh, cfg);
  if (!configs_match(holder->accel->built_config(), cfg)) {
    return nullptr;
  }

  return holder->accel.get();
}

int topological_dim(const MeshBase& mesh) {
  int tdim = 0;
  for (const auto& s : mesh.cell_shapes()) {
    if (s.is_3d()) return 3;
    if (s.is_2d()) tdim = std::max(tdim, 2);
    else if (s.is_1d()) tdim = std::max(tdim, 1);
  }
  return tdim;
}

#ifdef MESH_HAS_MPI
struct MinLoc {
  double value = 0.0;
  int rank = 0;
};

MinLoc allreduce_minloc(MPI_Comm comm, double value, int rank) {
  MinLoc in;
  in.value = value;
  in.rank = rank;
  MinLoc out;
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
  return out;
}
#endif

} // namespace

// ---- Point location ----

PointLocateResult MeshSearch::locate_point(const MeshBase& mesh,
                                          const std::array<real_t,3>& point,
                                          Configuration cfg,
                                          index_t hint_cell) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->locate_point(mesh, point, hint_cell);
  }

  // Simple linear search implementation
  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;

  size_t n_cells = mesh.n_cells();

  // Try hint cell first if provided
  if (hint_cell >= 0 && hint_cell < static_cast<index_t>(n_cells)) {
    if (point_in_cell(mesh, point, hint_cell, cfg)) {
      result.cell_id = hint_cell;
      result.found = true;
      result.xi = compute_parametric_coords(mesh, hint_cell, point, cfg);
      return result;
    }
  }

  // Linear search through all cells
  for (size_t c = 0; c < n_cells; ++c) {
    if (point_in_cell(mesh, point, static_cast<index_t>(c), cfg)) {
      result.cell_id = static_cast<index_t>(c);
      result.found = true;
      result.xi = compute_parametric_coords(mesh, result.cell_id, point, cfg);
      return result;
    }
  }

  return result;
}

PointLocateResult MeshSearch::locate_point_global(const DistributedMesh& mesh,
                                                 const std::array<real_t,3>& point,
                                                 Configuration cfg) {
  return mesh.locate_point_global(point, cfg);
}

std::vector<PointLocateResult> MeshSearch::locate_points(const MeshBase& mesh,
                                                        const std::vector<std::array<real_t,3>>& points,
                                                        Configuration cfg) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->locate_points(mesh, points);
  }

  std::vector<PointLocateResult> results;
  results.reserve(points.size());

  for (const auto& point : points) {
    results.push_back(locate_point(mesh, point, cfg));
  }

  return results;
}

std::vector<PointLocateResult> MeshSearch::locate_points_global(
    const DistributedMesh& mesh,
    const std::vector<std::array<real_t,3>>& points,
    Configuration cfg) {
  return mesh.locate_points_global(points, cfg);
}

bool MeshSearch::contains_point(const MeshBase& mesh,
                               const std::array<real_t,3>& point,
                               Configuration cfg) {
  return locate_point(mesh, point, cfg).found;
}

bool MeshSearch::contains_point_global(const DistributedMesh& mesh,
                                      const std::array<real_t,3>& point,
                                      Configuration cfg) {
  return locate_point_global(mesh, point, cfg).found;
}

// ---- Nearest neighbor search ----

std::pair<index_t, real_t> MeshSearch::nearest_vertex(const MeshBase& mesh,
                                                     const std::array<real_t,3>& point,
                                                     Configuration cfg) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->nearest_vertex(mesh, point);
  }

  index_t nearest_idx = INVALID_INDEX;
  real_t min_dist = std::numeric_limits<real_t>::max();

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  for (size_t n = 0; n < n_vertices; ++n) {
    real_t dist_sq = 0;
    for (int d = 0; d < dim; ++d) {
      real_t dx = coords[n * dim + d] - point[d];
      dist_sq += dx * dx;
    }

    if (dist_sq < min_dist * min_dist) {
      nearest_idx = static_cast<index_t>(n);
      min_dist = std::sqrt(dist_sq);
    }
  }

  return {nearest_idx, min_dist};
}

std::vector<std::pair<index_t, real_t>> MeshSearch::k_nearest_vertices(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    size_t k,
    Configuration cfg) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->k_nearest_vertices(mesh, point, k);
  }

  using HeapEntry = std::pair<real_t, index_t>;
  std::priority_queue<HeapEntry> max_heap;

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  for (size_t n = 0; n < n_vertices; ++n) {
    real_t dist_sq = 0;
    for (int d = 0; d < dim; ++d) {
      real_t dx = coords[n * dim + d] - point[d];
      dist_sq += dx * dx;
    }
    real_t dist = std::sqrt(dist_sq);

    if (max_heap.size() < k) {
      max_heap.push({dist, static_cast<index_t>(n)});
    } else if (dist < max_heap.top().first) {
      max_heap.pop();
      max_heap.push({dist, static_cast<index_t>(n)});
    }
  }

  // Extract results
  std::vector<std::pair<index_t, real_t>> results;
  while (!max_heap.empty()) {
    auto [dist, idx] = max_heap.top();
    max_heap.pop();
    results.push_back({idx, dist});
  }

  std::reverse(results.begin(), results.end());
  return results;
}

std::vector<index_t> MeshSearch::vertices_in_radius(const MeshBase& mesh,
                                                   const std::array<real_t,3>& point,
                                                   real_t radius,
                                                   Configuration cfg) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->vertices_in_radius(mesh, point, radius);
  }

  std::vector<index_t> vertices;
  real_t radius_sq = radius * radius;

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  for (size_t n = 0; n < n_vertices; ++n) {
    real_t dist_sq = 0;
    for (int d = 0; d < dim; ++d) {
      real_t dx = coords[n * dim + d] - point[d];
      dist_sq += dx * dx;
    }

    if (dist_sq <= radius_sq) {
      vertices.push_back(static_cast<index_t>(n));
    }
  }

  return vertices;
}

std::pair<index_t, real_t> MeshSearch::nearest_cell(const MeshBase& mesh,
                                                   const std::array<real_t,3>& point,
                                                   Configuration cfg) {
  index_t nearest_idx = INVALID_INDEX;
  real_t min_dist = std::numeric_limits<real_t>::max();

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    real_t dist = distance_to_cell(mesh, point, static_cast<index_t>(c), cfg);
    if (dist < min_dist) {
      nearest_idx = static_cast<index_t>(c);
      min_dist = dist;
    }
  }

  return {nearest_idx, min_dist};
}

// ---- Ray intersection ----

RayIntersectResult MeshSearch::intersect_ray(const MeshBase& mesh,
                                            const std::array<real_t,3>& origin,
                                            const std::array<real_t,3>& direction,
                                            Configuration cfg,
                                            real_t max_distance) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->intersect_ray(mesh, origin, direction, max_distance);
  }

  cfg = effective_cfg(mesh, cfg);

  RayIntersectResult result;
  result.found = false;
  result.face_id = -1;
  result.t = -1.0;

  const auto triangles = search::SearchBuilders::triangulate_boundary(mesh, cfg);
  if (triangles.empty()) {
    return result;
  }

  search::Ray ray(origin, search::normalize3(direction), 0.0, max_distance);

  real_t min_t = max_distance;
  index_t hit_face = INVALID_INDEX;
  std::array<real_t,3> hit_point{{0, 0, 0}};

  for (const auto& tri : triangles) {
    real_t t = 0.0;
    if (!search::ray_triangle_intersect(ray, tri.vertices[0], tri.vertices[1], tri.vertices[2], t)) {
      continue;
    }
    if (t < min_t) {
      min_t = t;
      hit_face = tri.face_id;
      hit_point = ray.point_at(t);
    }
  }

  if (hit_face != INVALID_INDEX) {
    result.found = true;
    result.hit = true;
    result.face_id = hit_face;
    result.t = min_t;
    result.distance = min_t;
    result.point = hit_point;
    result.hit_point = hit_point;
  }

  return result;
}

RayIntersectResult MeshSearch::intersect_ray_global(const DistributedMesh& mesh,
                                                    const std::array<real_t,3>& origin,
                                                    const std::array<real_t,3>& direction,
                                                    Configuration cfg,
                                                    real_t max_distance) {
  const auto& local = mesh.local_mesh();
  cfg = effective_cfg(local, cfg);

  RayIntersectResult result;
  result.found = false;
  result.hit = false;
  result.face_id = INVALID_INDEX;

  if (mesh.world_size() <= 1) {
    return intersect_ray(local, origin, direction, cfg, max_distance);
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL) {
    return intersect_ray(local, origin, direction, cfg, max_distance);
  }

  MPI_Comm comm = mesh.mpi_comm();
  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);

  const int local_tdim = topological_dim(local);
  int tdim = local_tdim;
  MPI_Allreduce(&local_tdim, &tdim, 1, MPI_INT, MPI_MAX, comm);

  const auto dir_n = search::normalize3(direction);
  search::Ray ray(origin, dir_n, 0.0, max_distance);

  real_t best_t = std::numeric_limits<real_t>::infinity();
  index_t best_id = INVALID_INDEX;
  std::array<real_t,3> best_pt{{0, 0, 0}};

  if (tdim == 3) {
    const int faces_ok_local = (local_tdim == 3) ? (local.n_faces() != 0 ? 1 : 0) : 1;
    int faces_ok = 1;
    MPI_Allreduce(&faces_ok_local, &faces_ok, 1, MPI_INT, MPI_MIN, comm);

    if (faces_ok != 0) {
      const auto boundary_faces = DistributedTopology::global_boundary_faces(mesh, /*owned_only=*/true);
      for (index_t f : boundary_faces) {
        const auto tris = search::SearchBuilders::triangulate_face(local, f, cfg);
        for (const auto& tri : tris) {
          real_t t = 0.0;
          if (!search::ray_triangle_intersect(ray, tri[0], tri[1], tri[2], t)) {
            continue;
          }
          if (t < best_t) {
            best_t = t;
            best_id = f;
            best_pt = ray.point_at(t);
          }
        }
      }
    } else {
      const auto info = BoundaryDetector::detect_boundary_global(mesh);
      for (const auto& ring : info.oriented_boundary_entities) {
        if (ring.size() < 3) continue;
        const auto p0 = search::SearchBuilders::get_vertex_coord(local, ring[0], cfg);
        for (size_t i = 1; i + 1 < ring.size(); ++i) {
          const auto p1 = search::SearchBuilders::get_vertex_coord(local, ring[i], cfg);
          const auto p2 = search::SearchBuilders::get_vertex_coord(local, ring[i + 1], cfg);
          real_t t = 0.0;
          if (!search::ray_triangle_intersect(ray, p0, p1, p2, t)) {
            continue;
          }
          if (t < best_t) {
            best_t = t;
            best_id = INVALID_INDEX;
            best_pt = ray.point_at(t);
          }
        }
      }
    }
  } else if (tdim == 2) {
    const auto owned_cells = mesh.owned_cells();
    for (index_t c : owned_cells) {
      const auto& shape = local.cell_shape(c);
      if (!shape.is_2d()) continue;
      const auto verts = search::SearchBuilders::get_cell_vertex_coords(local, c, cfg);
      if (verts.size() < 3) continue;
      for (size_t i = 1; i + 1 < verts.size(); ++i) {
        real_t t = 0.0;
        if (!search::ray_triangle_intersect(ray, verts[0], verts[i], verts[i + 1], t)) {
          continue;
        }
        if (t < best_t) {
          best_t = t;
          best_id = c;
          best_pt = ray.point_at(t);
        }
      }
    }
  }

  const double local_t = std::isfinite(best_t) ? static_cast<double>(best_t)
                                               : std::numeric_limits<double>::infinity();
  const MinLoc win = allreduce_minloc(comm, local_t, my_rank);

  if (!std::isfinite(win.value)) {
    return result;
  }

  const int win_rank = win.rank;
  std::array<real_t,3> hit_pt = best_pt;
  MPI_Bcast(hit_pt.data(), 3, MPI_DOUBLE, win_rank, comm);

  result.found = true;
  result.hit = true;
  result.t = static_cast<real_t>(win.value);
  result.distance = result.t;
  result.point = hit_pt;
  result.hit_point = hit_pt;
  if (my_rank == win_rank) {
    result.face_id = best_id;
  } else {
    result.face_id = INVALID_INDEX;
  }

  return result;
#else
  return intersect_ray(local, origin, direction, cfg, max_distance);
#endif
}

std::vector<RayIntersectResult> MeshSearch::intersect_ray_all(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    Configuration cfg,
    real_t max_distance) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->intersect_ray_all(mesh, origin, direction, max_distance);
  }

  cfg = effective_cfg(mesh, cfg);

  std::vector<RayIntersectResult> results;

  const auto triangles = search::SearchBuilders::triangulate_boundary(mesh, cfg);
  if (triangles.empty()) {
    return results;
  }

  search::Ray ray(origin, search::normalize3(direction), 0.0, max_distance);
  results.reserve(triangles.size());

  for (const auto& tri : triangles) {
    real_t t = 0.0;
    if (!search::ray_triangle_intersect(ray, tri.vertices[0], tri.vertices[1], tri.vertices[2], t)) {
      continue;
    }

    RayIntersectResult hit;
    hit.found = true;
    hit.hit = true;
    hit.face_id = tri.face_id;
    hit.t = t;
    hit.distance = t;
    hit.point = ray.point_at(t);
    hit.hit_point = hit.point;
    results.push_back(hit);
  }

  std::sort(results.begin(), results.end(),
            [](const RayIntersectResult& a, const RayIntersectResult& b) { return a.t < b.t; });

  // De-duplicate hits that arise from triangulating the same face, e.g. when the ray intersects
  // exactly on a shared triangle edge. Keep one hit per (face_id, t) within tolerance.
  if (!results.empty()) {
    std::vector<RayIntersectResult> unique;
    unique.reserve(results.size());
    unique.push_back(results.front());

    const auto same_hit = [](const RayIntersectResult& a, const RayIntersectResult& b) {
      if (a.face_id != b.face_id) return false;
      const real_t tol = 1e-10 * std::max(real_t(1.0), std::max(std::abs(a.t), std::abs(b.t)));
      return std::abs(a.t - b.t) <= tol;
    };

    for (size_t i = 1; i < results.size(); ++i) {
      if (!same_hit(results[i], unique.back())) {
        unique.push_back(results[i]);
      }
    }
    results.swap(unique);
  }

  return results;
}

std::vector<RayIntersectResult> MeshSearch::intersect_ray_all_global(
    const DistributedMesh& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    Configuration cfg,
    real_t max_distance) {
  const auto& local = mesh.local_mesh();
  cfg = effective_cfg(local, cfg);

  if (mesh.world_size() <= 1) {
    return intersect_ray_all(local, origin, direction, cfg, max_distance);
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL) {
    return intersect_ray_all(local, origin, direction, cfg, max_distance);
  }

  MPI_Comm comm = mesh.mpi_comm();
  int my_rank = 0;
  int world = 1;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &world);

  const int local_tdim = topological_dim(local);
  int tdim = local_tdim;
  MPI_Allreduce(&local_tdim, &tdim, 1, MPI_INT, MPI_MAX, comm);

  const auto dir_n = search::normalize3(direction);
  search::Ray ray(origin, dir_n, 0.0, max_distance);

  std::vector<RayIntersectResult> local_hits;

  if (tdim == 3) {
    const int faces_ok_local = (local_tdim == 3) ? (local.n_faces() != 0 ? 1 : 0) : 1;
    int faces_ok = 1;
    MPI_Allreduce(&faces_ok_local, &faces_ok, 1, MPI_INT, MPI_MIN, comm);

    if (faces_ok != 0) {
      const auto boundary_faces = DistributedTopology::global_boundary_faces(mesh, /*owned_only=*/true);
      for (index_t f : boundary_faces) {
        const auto tris = search::SearchBuilders::triangulate_face(local, f, cfg);
        for (const auto& tri : tris) {
          real_t t = 0.0;
          if (!search::ray_triangle_intersect(ray, tri[0], tri[1], tri[2], t)) {
            continue;
          }
          RayIntersectResult hit;
          hit.found = true;
          hit.hit = true;
          hit.face_id = f;
          hit.t = t;
          hit.distance = t;
          hit.point = ray.point_at(t);
          hit.hit_point = hit.point;
          local_hits.push_back(hit);
        }
      }
    } else {
      const auto info = BoundaryDetector::detect_boundary_global(mesh);
      for (const auto& ring : info.oriented_boundary_entities) {
        if (ring.size() < 3) continue;
        const auto p0 = search::SearchBuilders::get_vertex_coord(local, ring[0], cfg);
        for (size_t i = 1; i + 1 < ring.size(); ++i) {
          const auto p1 = search::SearchBuilders::get_vertex_coord(local, ring[i], cfg);
          const auto p2 = search::SearchBuilders::get_vertex_coord(local, ring[i + 1], cfg);
          real_t t = 0.0;
          if (!search::ray_triangle_intersect(ray, p0, p1, p2, t)) {
            continue;
          }
          RayIntersectResult hit;
          hit.found = true;
          hit.hit = true;
          hit.face_id = INVALID_INDEX;
          hit.t = t;
          hit.distance = t;
          hit.point = ray.point_at(t);
          hit.hit_point = hit.point;
          local_hits.push_back(hit);
        }
      }
    }
  } else if (tdim == 2) {
    const auto owned_cells = mesh.owned_cells();
    for (index_t c : owned_cells) {
      const auto& shape = local.cell_shape(c);
      if (!shape.is_2d()) continue;
      const auto verts = search::SearchBuilders::get_cell_vertex_coords(local, c, cfg);
      if (verts.size() < 3) continue;
      for (size_t i = 1; i + 1 < verts.size(); ++i) {
        real_t t = 0.0;
        if (!search::ray_triangle_intersect(ray, verts[0], verts[i], verts[i + 1], t)) {
          continue;
        }
        RayIntersectResult hit;
        hit.found = true;
        hit.hit = true;
        hit.face_id = c;
        hit.t = t;
        hit.distance = t;
        hit.point = ray.point_at(t);
        hit.hit_point = hit.point;
        local_hits.push_back(hit);
      }
    }
  }

  std::sort(local_hits.begin(), local_hits.end(),
            [](const RayIntersectResult& a, const RayIntersectResult& b) { return a.t < b.t; });

  // De-duplicate local hits per (face_id, t) within tolerance.
  if (!local_hits.empty()) {
    std::vector<RayIntersectResult> unique;
    unique.reserve(local_hits.size());
    unique.push_back(local_hits.front());

    const auto same_hit = [](const RayIntersectResult& a, const RayIntersectResult& b) {
      if (a.face_id != b.face_id) return false;
      const real_t tol = 1e-10 * std::max(real_t(1.0), std::max(std::abs(a.t), std::abs(b.t)));
      return std::abs(a.t - b.t) <= tol;
    };

    for (size_t i = 1; i < local_hits.size(); ++i) {
      if (!same_hit(local_hits[i], unique.back())) {
        unique.push_back(local_hits[i]);
      }
    }
    local_hits.swap(unique);
  }

  // Gather variable-sized hit lists to rank 0.
  std::vector<int> counts(world, 0);
  const int local_n = static_cast<int>(local_hits.size());
  MPI_Gather(&local_n, 1, MPI_INT,
             counts.data(), 1, MPI_INT,
             0, comm);

  std::vector<int> displs(world + 1, 0);
  int total = 0;
  if (my_rank == 0) {
    for (int r = 0; r < world; ++r) {
      displs[r] = total;
      total += counts[r];
    }
    displs[world] = total;
  }

  struct PackedHit {
    double t;
    double x[3];
  };
  static_assert(sizeof(PackedHit) == 32, "PackedHit must be 32 bytes");

  std::vector<PackedHit> send_buf(static_cast<size_t>(local_hits.size()));
  for (size_t i = 0; i < local_hits.size(); ++i) {
    send_buf[i].t = static_cast<double>(local_hits[i].t);
    send_buf[i].x[0] = static_cast<double>(local_hits[i].point[0]);
    send_buf[i].x[1] = static_cast<double>(local_hits[i].point[1]);
    send_buf[i].x[2] = static_cast<double>(local_hits[i].point[2]);
  }

  std::vector<PackedHit> recv_buf;
  std::vector<int> byte_counts;
  std::vector<int> byte_displs;
  if (my_rank == 0) {
    recv_buf.resize(static_cast<size_t>(total));
    byte_counts.resize(world);
    byte_displs.resize(world);
    for (int r = 0; r < world; ++r) {
      byte_counts[r] = counts[r] * static_cast<int>(sizeof(PackedHit));
      byte_displs[r] = displs[r] * static_cast<int>(sizeof(PackedHit));
    }
  }

  const int send_bytes = local_n * static_cast<int>(sizeof(PackedHit));
  MPI_Gatherv(send_buf.data(),
              send_bytes,
              MPI_BYTE,
              my_rank == 0 ? recv_buf.data() : nullptr,
              my_rank == 0 ? byte_counts.data() : nullptr,
              my_rank == 0 ? byte_displs.data() : nullptr,
              MPI_BYTE,
              0,
              comm);

  if (my_rank != 0) {
    return {};
  }

  std::vector<RayIntersectResult> results;
  results.reserve(recv_buf.size());
  for (const auto& ph : recv_buf) {
    RayIntersectResult hit;
    hit.found = true;
    hit.hit = true;
    hit.face_id = INVALID_INDEX;
    hit.t = static_cast<real_t>(ph.t);
    hit.distance = hit.t;
    hit.point = {{static_cast<real_t>(ph.x[0]), static_cast<real_t>(ph.x[1]), static_cast<real_t>(ph.x[2])}};
    hit.hit_point = hit.point;
    results.push_back(hit);
  }

  std::sort(results.begin(), results.end(),
            [](const RayIntersectResult& a, const RayIntersectResult& b) { return a.t < b.t; });

  return results;
#else
  return intersect_ray_all(local, origin, direction, cfg, max_distance);
#endif
}

// ---- Distance queries ----

real_t MeshSearch::signed_distance(const MeshBase& mesh,
                                  const std::array<real_t,3>& point,
                                  Configuration cfg) {
  cfg = effective_cfg(mesh, cfg);

  auto [closest, face_id] = closest_boundary_point(mesh, point, cfg);
  (void)face_id;

  const auto diff = search::sub3(closest, point);
  const real_t dist = std::sqrt(search::dot3(diff, diff));
  const bool inside = contains_point(mesh, point, cfg);
  return inside ? -dist : dist;
}

real_t MeshSearch::signed_distance_global(const DistributedMesh& mesh,
                                         const std::array<real_t,3>& point,
                                         Configuration cfg) {
  auto [closest, id] = closest_boundary_point_global(mesh, point, cfg);
  (void)id;
  const auto diff = search::sub3(closest, point);
  const real_t dist = std::sqrt(search::dot3(diff, diff));
  const bool inside = contains_point_global(mesh, point, cfg);
  return inside ? -dist : dist;
}

std::pair<std::array<real_t,3>, index_t> MeshSearch::closest_boundary_point(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    Configuration cfg) {
  cfg = effective_cfg(mesh, cfg);

  const int tdim = topological_dim(mesh);

  std::array<real_t,3> closest_pt{{0, 0, 0}};
  index_t closest_id = INVALID_INDEX;
  real_t min_dist = std::numeric_limits<real_t>::max();

  if (tdim == 3) {
    const auto boundary_faces = mesh.boundary_faces();
    for (index_t f : boundary_faces) {
      const auto tris = search::SearchBuilders::triangulate_face(mesh, f, cfg);
      for (const auto& tri : tris) {
        std::array<real_t,3> cand{{0, 0, 0}};
        const real_t d = search::point_triangle_distance(point, tri[0], tri[1], tri[2], &cand);
        if (d < min_dist) {
          min_dist = d;
          closest_pt = cand;
          closest_id = f;
        }
      }
    }
    return {closest_pt, closest_id};
  }

  if (tdim == 2) {
    const auto boundary_edges = mesh.boundary_faces(); // faces are edges in 2D
    for (index_t e : boundary_edges) {
      auto [vptr, nv] = mesh.face_vertices_span(e);
      if (nv < 2) continue;
      const auto a = search::SearchBuilders::get_vertex_coord(mesh, vptr[0], cfg);
      const auto b = search::SearchBuilders::get_vertex_coord(mesh, vptr[1], cfg);
      std::array<real_t,3> cand{{0, 0, 0}};
      const real_t d = search::point_segment_distance(point, a, b, &cand);
      if (d < min_dist) {
        min_dist = d;
        closest_pt = cand;
        closest_id = e;
      }
    }
    return {closest_pt, closest_id};
  }

  if (tdim == 1) {
    // Boundary vertices are endpoints of incident line cells.
    std::vector<int> count(mesh.n_vertices(), 0);
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
      const auto shape = mesh.cell_shape(c);
      if (!shape.is_1d()) continue;
      auto [vptr, nv] = mesh.cell_vertices_span(c);
      if (nv == 0) continue;
      const int nc = (shape.num_corners > 0) ? shape.num_corners : static_cast<int>(nv);
      if (nc >= 1) count[static_cast<size_t>(vptr[0])] += 1;
      if (nc >= 2) count[static_cast<size_t>(vptr[nc - 1])] += 1;
    }

    for (size_t v = 0; v < count.size(); ++v) {
      if (count[v] != 1) continue;
      const auto pv = search::SearchBuilders::get_vertex_coord(mesh, static_cast<index_t>(v), cfg);
      const auto diff = search::sub3(pv, point);
      const real_t d = std::sqrt(search::dot3(diff, diff));
      if (d < min_dist) {
        min_dist = d;
        closest_pt = pv;
        closest_id = static_cast<index_t>(v);
      }
    }
    return {closest_pt, closest_id};
  }

  return {closest_pt, closest_id};
}

std::pair<std::array<real_t,3>, index_t> MeshSearch::closest_boundary_point_global(
    const DistributedMesh& mesh,
    const std::array<real_t,3>& point,
    Configuration cfg) {
  const auto& local = mesh.local_mesh();
  cfg = effective_cfg(local, cfg);

  if (mesh.world_size() <= 1) {
    return closest_boundary_point(local, point, cfg);
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL) {
    return closest_boundary_point(local, point, cfg);
  }

  MPI_Comm comm = mesh.mpi_comm();
  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);

  const int local_tdim = topological_dim(local);
  int tdim = local_tdim;
  MPI_Allreduce(&local_tdim, &tdim, 1, MPI_INT, MPI_MAX, comm);

  std::array<real_t,3> best_pt{{0, 0, 0}};
  index_t best_id = INVALID_INDEX;
  real_t best_d = std::numeric_limits<real_t>::infinity();

  if (tdim == 3) {
    const int faces_ok_local = (local_tdim == 3) ? (local.n_faces() != 0 ? 1 : 0) : 1;
    int faces_ok = 1;
    MPI_Allreduce(&faces_ok_local, &faces_ok, 1, MPI_INT, MPI_MIN, comm);

    if (faces_ok != 0) {
      const auto boundary_faces = DistributedTopology::global_boundary_faces(mesh, /*owned_only=*/true);
      for (index_t f : boundary_faces) {
        const auto tris = search::SearchBuilders::triangulate_face(local, f, cfg);
        for (const auto& tri : tris) {
          std::array<real_t,3> cand{{0, 0, 0}};
          const real_t d = search::point_triangle_distance(point, tri[0], tri[1], tri[2], &cand);
          if (d < best_d) {
            best_d = d;
            best_pt = cand;
            best_id = f;
          }
        }
      }
    } else {
      const auto info = BoundaryDetector::detect_boundary_global(mesh);
      for (const auto& ring : info.oriented_boundary_entities) {
        if (ring.size() < 3) continue;
        const auto p0 = search::SearchBuilders::get_vertex_coord(local, ring[0], cfg);
        for (size_t i = 1; i + 1 < ring.size(); ++i) {
          const auto p1 = search::SearchBuilders::get_vertex_coord(local, ring[i], cfg);
          const auto p2 = search::SearchBuilders::get_vertex_coord(local, ring[i + 1], cfg);
          std::array<real_t,3> cand{{0, 0, 0}};
          const real_t d = search::point_triangle_distance(point, p0, p1, p2, &cand);
          if (d < best_d) {
            best_d = d;
            best_pt = cand;
            best_id = INVALID_INDEX;
          }
        }
      }
    }
  } else if (tdim == 2) {
    const int faces_ok_local = (local_tdim == 2) ? (local.n_faces() != 0 ? 1 : 0) : 1;
    int faces_ok = 1;
    MPI_Allreduce(&faces_ok_local, &faces_ok, 1, MPI_INT, MPI_MIN, comm);

    if (faces_ok != 0) {
      const auto boundary_edges = DistributedTopology::global_boundary_faces(mesh, /*owned_only=*/true);
      for (index_t e : boundary_edges) {
        auto [vptr, nv] = local.face_vertices_span(e);
        if (nv < 2) continue;

        for (size_t i = 0; i + 1 < nv; ++i) {
          const auto a = search::SearchBuilders::get_vertex_coord(local, vptr[i], cfg);
          const auto b = search::SearchBuilders::get_vertex_coord(local, vptr[i + 1], cfg);
          std::array<real_t,3> cand{{0, 0, 0}};
          const real_t d = search::point_segment_distance(point, a, b, &cand);
          if (d < best_d) {
            best_d = d;
            best_pt = cand;
            best_id = e;
          }
        }
      }
    } else {
      const auto info = BoundaryDetector::detect_boundary_global(mesh);
      for (const auto& edge : info.oriented_boundary_entities) {
        if (edge.size() < 2) continue;
        for (size_t i = 0; i + 1 < edge.size(); ++i) {
          const auto a = search::SearchBuilders::get_vertex_coord(local, edge[i], cfg);
          const auto b = search::SearchBuilders::get_vertex_coord(local, edge[i + 1], cfg);
          std::array<real_t,3> cand{{0, 0, 0}};
          const real_t d = search::point_segment_distance(point, a, b, &cand);
          if (d < best_d) {
            best_d = d;
            best_pt = cand;
            best_id = INVALID_INDEX;
          }
        }
      }
    }
  } else if (tdim == 1) {
    const auto info = BoundaryDetector::detect_boundary_global(mesh);
    for (const auto& vtx : info.oriented_boundary_entities) {
      if (vtx.empty()) continue;
      const auto pv = search::SearchBuilders::get_vertex_coord(local, vtx[0], cfg);
      const auto diff = search::sub3(pv, point);
      const real_t d = std::sqrt(search::dot3(diff, diff));
      if (d < best_d) {
        best_d = d;
        best_pt = pv;
        best_id = INVALID_INDEX;
      }
    }
  }

  const double local_d = std::isfinite(best_d) ? static_cast<double>(best_d)
                                               : std::numeric_limits<double>::infinity();
  const MinLoc win = allreduce_minloc(comm, local_d, my_rank);

  if (!std::isfinite(win.value)) {
    return {best_pt, INVALID_INDEX};
  }

  const int win_rank = win.rank;
  std::array<real_t,3> out_pt = best_pt;
  MPI_Bcast(out_pt.data(), 3, MPI_DOUBLE, win_rank, comm);

  if (my_rank != win_rank) {
    best_id = INVALID_INDEX;
  }
  return {out_pt, best_id};
#else
  return closest_boundary_point(local, point, cfg);
#endif
}

// ---- Search structure management ----

void MeshSearch::build_search_structure(const MeshBase& mesh,
                                       const SearchConfig& config,
                                       Configuration cfg) {
  mesh.build_search_structure(config, effective_cfg(mesh, cfg));
}

void MeshSearch::clear_search_structure(const MeshBase& mesh) {
  mesh.clear_search_structure();
}

bool MeshSearch::has_search_structure(const MeshBase& mesh) {
  return mesh.has_search_structure();
}

// ---- Spatial queries ----

std::vector<index_t> MeshSearch::cells_in_box(const MeshBase& mesh,
                                             const std::array<real_t,3>& box_min,
                                             const std::array<real_t,3>& box_max,
                                             Configuration cfg) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->cells_in_box(mesh, box_min, box_max);
  }

  std::vector<index_t> cells;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto center = mesh.cell_center(static_cast<index_t>(c), cfg);

    bool inside = true;
    for (int d = 0; d < 3; ++d) {
      if (center[d] < box_min[d] || center[d] > box_max[d]) {
        inside = false;
        break;
      }
    }

    if (inside) {
      cells.push_back(static_cast<index_t>(c));
    }
  }

  return cells;
}

std::vector<index_t> MeshSearch::cells_in_sphere(const MeshBase& mesh,
                                                const std::array<real_t,3>& center,
                                                real_t radius,
                                                Configuration cfg) {
  if (const auto* accel = accel_if_built(mesh, cfg)) {
    return accel->cells_in_sphere(mesh, center, radius);
  }

  std::vector<index_t> cells;
  real_t radius_sq = radius * radius;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto cell_center = mesh.cell_center(static_cast<index_t>(c), cfg);

    real_t dist_sq = 0;
    for (int d = 0; d < 3; ++d) {
      real_t dx = cell_center[d] - center[d];
      dist_sq += dx * dx;
    }

    if (dist_sq <= radius_sq) {
      cells.push_back(static_cast<index_t>(c));
    }
  }

  return cells;
}

// ---- Parametric coordinates ----

std::array<real_t,3> MeshSearch::compute_parametric_coords(const MeshBase& mesh,
                                                          index_t cell,
                                                          const std::array<real_t,3>& point,
                                                          Configuration cfg) {
  cfg = effective_cfg(mesh, cfg);
  if (cell < 0 || static_cast<size_t>(cell) >= mesh.n_cells()) {
    return {0, 0, 0};
  }

  const auto shape = mesh.cell_shape(cell);
  const bool supported =
      (shape.family == CellFamily::Point ||
       shape.family == CellFamily::Line ||
       shape.family == CellFamily::Triangle ||
       shape.family == CellFamily::Quad ||
       shape.family == CellFamily::Tetra ||
       shape.family == CellFamily::Hex ||
       shape.family == CellFamily::Wedge ||
       shape.family == CellFamily::Pyramid);
  if (!supported) {
    return {0, 0, 0};
  }

  const auto [xi, ok] = CurvilinearEvaluator::inverse_map(mesh, cell, point, cfg);
  if (!ok) {
    // Return best-effort iterate (still useful for diagnostics); callers should
    // validate using is_inside_reference_element().
    return xi;
  }

  return xi;
}

bool MeshSearch::is_inside_reference_element(const CellShape& shape,
                                            const std::array<real_t,3>& xi) {
  // Check if parametric coordinates are inside reference element
  switch (shape.family) {
    case CellFamily::Line:
      return xi[0] >= -1 && xi[0] <= 1;

    case CellFamily::Triangle:
      return xi[0] >= 0 && xi[1] >= 0 && (xi[0] + xi[1]) <= 1;

    case CellFamily::Quad:
      return xi[0] >= -1 && xi[0] <= 1 &&
             xi[1] >= -1 && xi[1] <= 1;

    case CellFamily::Tetra:
      return xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 &&
             (xi[0] + xi[1] + xi[2]) <= 1;

    case CellFamily::Hex:
      return xi[0] >= -1 && xi[0] <= 1 &&
             xi[1] >= -1 && xi[1] <= 1 &&
             xi[2] >= -1 && xi[2] <= 1;

    default:
      return false;
  }
}

// ---- Walking algorithms ----

std::vector<index_t> MeshSearch::walk_to_point(const MeshBase& mesh,
                                              index_t start_cell,
                                              const std::array<real_t,3>& target_point,
                                              Configuration cfg) {
  cfg = effective_cfg(mesh, cfg);

  std::vector<index_t> path;
  path.push_back(start_cell);

  if (start_cell < 0 || static_cast<size_t>(start_cell) >= mesh.n_cells()) {
    return path;
  }

  // Greedy walk: pick the neighbor whose center is closest to the target.
  // This is not guaranteed to find a path for all meshes, but provides a safe,
  // deterministic fallback for coarse point-location heuristics.
  std::unordered_set<index_t> visited;
  visited.insert(start_cell);

  index_t current = start_cell;
  for (size_t iter = 0; iter < mesh.n_cells(); ++iter) {
    if (point_in_cell(mesh, target_point, current, cfg)) {
      break;
    }

    const auto cur_center = mesh.cell_center(current, cfg);
    real_t cur_d2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      const real_t dx = cur_center[d] - target_point[d];
      cur_d2 += dx * dx;
    }

    auto neighbors = mesh.cell_neighbors(current);
    index_t best = INVALID_INDEX;
    real_t best_d2 = cur_d2;
    for (index_t nbr : neighbors) {
      if (nbr < 0 || static_cast<size_t>(nbr) >= mesh.n_cells()) continue;
      if (visited.count(nbr) > 0) continue;
      const auto c = mesh.cell_center(nbr, cfg);
      real_t d2 = 0.0;
      for (int d = 0; d < 3; ++d) {
        const real_t dx = c[d] - target_point[d];
        d2 += dx * dx;
      }
      if (d2 < best_d2) {
        best_d2 = d2;
        best = nbr;
      }
    }

    if (best == INVALID_INDEX) {
      break;
    }

    visited.insert(best);
    path.push_back(best);
    current = best;
  }

  return path;
}

// ---- Helper methods ----

bool MeshSearch::point_in_cell(const MeshBase& mesh,
                              const std::array<real_t,3>& point,
                              index_t cell,
                              Configuration cfg) {
  cfg = effective_cfg(mesh, cfg);

  if (cell < 0 || static_cast<size_t>(cell) >= mesh.n_cells()) {
    return false;
  }

  const auto shape = mesh.cell_shape(cell);
  auto [vptr, nverts] = mesh.cell_vertices_span(cell);

  const size_t n_corners = (shape.num_corners > 0)
                               ? static_cast<size_t>(shape.num_corners)
                               : nverts;
  if (n_corners == 0 || n_corners > nverts) {
    return false;
  }

  const bool has_high_order_nodes =
      (shape.order > 1) || (nverts > n_corners);
  if (has_high_order_nodes &&
      (shape.family == CellFamily::Line ||
       shape.family == CellFamily::Triangle ||
       shape.family == CellFamily::Quad ||
       shape.family == CellFamily::Tetra ||
       shape.family == CellFamily::Hex ||
       shape.family == CellFamily::Wedge ||
       shape.family == CellFamily::Pyramid)) {
    const auto [xi, ok] = CurvilinearEvaluator::inverse_map(mesh, cell, point, cfg);
    if (!ok) {
      return false;
    }
    return CurvilinearEvaluator::is_inside_reference_element(shape, xi);
  }

  const int dim = mesh.dim();
  const auto& coords = (cfg == Configuration::Current && mesh.has_current_coords()) ? mesh.X_cur()
                                                                                    : mesh.X_ref();

  std::vector<std::array<real_t,3>> verts;
  verts.reserve(n_corners);
  for (size_t i = 0; i < n_corners; ++i) {
    const index_t v = vptr[i];
    if (v < 0) {
      return false;
    }
    const size_t vi = static_cast<size_t>(v);
    if (vi * static_cast<size_t>(dim) + static_cast<size_t>(std::max(0, dim - 1)) >= coords.size()) {
      return false;
    }

    std::array<real_t,3> x{{0.0, 0.0, 0.0}};
    for (int d = 0; d < dim && d < 3; ++d) {
      x[static_cast<size_t>(d)] = coords[vi * static_cast<size_t>(dim) + static_cast<size_t>(d)];
    }
    verts.push_back(x);
  }

  return search::point_in_cell(point, shape, verts);
}

real_t MeshSearch::distance_to_cell(const MeshBase& mesh,
                                   const std::array<real_t,3>& point,
                                   index_t cell,
                                   Configuration cfg) {
  // Simple distance to cell center
  auto center = mesh.cell_center(cell, cfg);

  real_t dist_sq = 0;
  for (int d = 0; d < 3; ++d) {
    real_t dx = center[d] - point[d];
    dist_sq += dx * dx;
  }

  return std::sqrt(dist_sq);
}

} // namespace svmp
