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

#include "SearchBuilders.h"
#include "../Core/MeshBase.h"
#include "../Geometry/CurvilinearEval.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace svmp {
namespace search {

// ---- Coordinate extraction ----

std::vector<std::array<real_t,3>> SearchBuilders::extract_vertex_coords(
    const MeshBase& mesh,
    Configuration cfg) {

  const std::vector<real_t>& coords = ((cfg != Configuration::Reference) && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  std::vector<std::array<real_t,3>> result;
  result.reserve(n_vertices);

  for (size_t v = 0; v < n_vertices; ++v) {
    std::array<real_t,3> pt = {0, 0, 0};
    for (int d = 0; d < dim; ++d) {
      pt[d] = coords[v * dim + d];
    }
    result.push_back(pt);
  }

  return result;
}

std::array<real_t,3> SearchBuilders::get_vertex_coord(
    const MeshBase& mesh,
    index_t vertex_id,
    Configuration cfg) {

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int dim = mesh.dim();

  std::array<real_t,3> pt = {0, 0, 0};
  for (int d = 0; d < dim; ++d) {
    pt[d] = coords[vertex_id * dim + d];
  }

  return pt;
}

std::vector<std::array<real_t,3>> SearchBuilders::get_cell_vertex_coords(
    const MeshBase& mesh,
    index_t cell_id,
    Configuration cfg) {

  auto vertices = mesh.cell_vertices(cell_id);
  std::vector<std::array<real_t,3>> coords;
  coords.reserve(vertices.size());

  for (index_t v : vertices) {
    coords.push_back(get_vertex_coord(mesh, v, cfg));
  }

  return coords;
}

std::vector<std::array<real_t,3>> SearchBuilders::get_face_vertex_coords(
    const MeshBase& mesh,
    index_t face_id,
    Configuration cfg) {

  auto vertices = mesh.face_vertices(face_id);
  std::vector<std::array<real_t,3>> coords;
  coords.reserve(vertices.size());

  for (index_t v : vertices) {
    coords.push_back(get_vertex_coord(mesh, v, cfg));
  }

  return coords;
}

// ---- AABB computation ----

AABB SearchBuilders::compute_mesh_aabb(const MeshBase& mesh, Configuration cfg) {
  AABB aabb;
  auto all_coords = extract_vertex_coords(mesh, cfg);

  for (const auto& coord : all_coords) {
    aabb.include(coord);
  }

  return aabb;
}

AABB SearchBuilders::compute_cell_aabb(const MeshBase& mesh,
                                       index_t cell_id,
                                       Configuration cfg) {
  AABB aabb;
  auto coords = get_cell_vertex_coords(mesh, cell_id, cfg);

  for (const auto& coord : coords) {
    aabb.include(coord);
  }

  return aabb;
}

std::vector<AABB> SearchBuilders::compute_all_cell_aabbs(const MeshBase& mesh,
                                                         Configuration cfg) {
  size_t n_cells = mesh.n_cells();
  std::vector<AABB> aabbs;
  aabbs.reserve(n_cells);

  for (size_t c = 0; c < n_cells; ++c) {
    aabbs.push_back(compute_cell_aabb(mesh, static_cast<index_t>(c), cfg));
  }

  return aabbs;
}

std::vector<AABB> SearchBuilders::compute_cell_aabbs(const MeshBase& mesh,
                                                     const std::vector<std::array<real_t,3>>& vertex_coords) {
  size_t n_cells = mesh.n_cells();
  std::vector<AABB> aabbs;
  aabbs.reserve(n_cells);

  for (size_t c = 0; c < n_cells; ++c) {
    AABB aabb;
    auto verts = mesh.cell_vertices(static_cast<index_t>(c));
    for (index_t vid : verts) {
      if (vid >= 0 && static_cast<size_t>(vid) < vertex_coords.size()) {
        aabb.include(vertex_coords[static_cast<size_t>(vid)]);
      }
    }
    aabbs.push_back(aabb);
  }
  return aabbs;
}

AABB SearchBuilders::compute_face_aabb(const MeshBase& mesh,
                                       index_t face_id,
                                       Configuration cfg) {
  AABB aabb;
  auto coords = get_face_vertex_coords(mesh, face_id, cfg);

  for (const auto& coord : coords) {
    aabb.include(coord);
  }

  return aabb;
}

std::vector<AABB> SearchBuilders::compute_boundary_face_aabbs(const MeshBase& mesh,
                                                              Configuration cfg) {
  std::vector<AABB> aabbs;
  auto boundary_faces = get_boundary_faces(mesh);
  aabbs.reserve(boundary_faces.size());

  for (index_t face_id : boundary_faces) {
    aabbs.push_back(compute_face_aabb(mesh, face_id, cfg));
  }

  return aabbs;
}

// ---- Boundary extraction ----

std::vector<index_t> SearchBuilders::get_boundary_faces(const MeshBase& mesh) {
  std::vector<index_t> boundary_faces;
  size_t n_faces = mesh.n_faces();

  for (size_t f = 0; f < n_faces; ++f) {
    if (is_boundary_face(mesh, static_cast<index_t>(f))) {
      boundary_faces.push_back(static_cast<index_t>(f));
    }
  }

  return boundary_faces;
}

bool SearchBuilders::is_boundary_face(const MeshBase& mesh, index_t face_id) {
  auto face_cells = mesh.face_cells(face_id);
  return face_cells[1] == INVALID_INDEX;
}

std::vector<index_t> SearchBuilders::get_cell_boundary_faces(const MeshBase& mesh,
                                                             index_t cell_id) {
  std::vector<index_t> boundary_faces;
  auto cell_faces = mesh.cell_faces(cell_id);

  for (index_t face_id : cell_faces) {
    if (is_boundary_face(mesh, face_id)) {
      boundary_faces.push_back(face_id);
    }
  }

  return boundary_faces;
}

// ---- Face triangulation ----

std::vector<std::array<std::array<real_t,3>,3>> SearchBuilders::triangulate_face(
    const MeshBase& mesh,
    index_t face_id,
    Configuration cfg) {

  auto vertices = get_face_vertex_coords(mesh, face_id, cfg);
  std::vector<std::array<std::array<real_t,3>,3>> triangles;

  if (vertices.size() < 3) {
    return triangles;  // Invalid face
  }

  if (vertices.size() == 3) {
    // Already a triangle
    triangles.push_back({vertices[0], vertices[1], vertices[2]});
  } else {
    // Fan triangulation from first vertex
    for (size_t i = 1; i < vertices.size() - 1; ++i) {
      triangles.push_back({vertices[0], vertices[i], vertices[i+1]});
    }
  }

  return triangles;
}

std::vector<SearchBuilders::TriangleWithFace> SearchBuilders::triangulate_boundary(
    const MeshBase& mesh,
    Configuration cfg) {

  std::vector<TriangleWithFace> triangles;
  auto boundary_faces = get_boundary_faces(mesh);

  // If the mesh exposes surface-like boundary faces (>= 3 vertices), triangulate them.
  // Otherwise (e.g., a surface mesh where "faces" are edges), treat 2D cells as the
  // boundary surface and triangulate cells instead.
  bool has_triangulatable_boundary_faces = false;
  for (index_t face_id : boundary_faces) {
    auto [vptr, nv] = mesh.face_vertices_span(face_id);
    if (nv >= 3) {
      has_triangulatable_boundary_faces = true;
      break;
    }
  }

  if (has_triangulatable_boundary_faces) {
    for (index_t face_id : boundary_faces) {
      auto face_triangles = triangulate_face(mesh, face_id, cfg);

      for (const auto& tri : face_triangles) {
        TriangleWithFace twf;
        twf.vertices = tri;
        twf.face_id = face_id;
        triangles.push_back(twf);
      }
    }
    return triangles;
  }

  // Surface mesh fallback: triangulate 2D cells and use `face_id` as the cell id.
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    const auto& shape = mesh.cell_shape(c);
    if (!shape.is_2d()) continue;

    auto vertices = get_cell_vertex_coords(mesh, c, cfg);
    if (vertices.size() < 3) continue;

    // Fan triangulation from first vertex
    for (size_t i = 1; i + 1 < vertices.size(); ++i) {
      TriangleWithFace tri;
      tri.vertices = {vertices[0], vertices[i], vertices[i + 1]};
      tri.face_id = c;
      triangles.push_back(tri);
    }
  }

  return triangles;
}

std::vector<SearchBuilders::TriangleWithFace> SearchBuilders::extract_boundary_triangles(
    const MeshBase& mesh,
    const std::vector<std::array<real_t,3>>& vertex_coords) {
  std::vector<TriangleWithFace> triangles;
  auto boundary_faces = get_boundary_faces(mesh);
  bool has_triangulatable_boundary_faces = false;
  for (index_t face_id : boundary_faces) {
    auto [vptr, nv] = mesh.face_vertices_span(face_id);
    if (nv >= 3) {
      has_triangulatable_boundary_faces = true;
      break;
    }
  }

  if (has_triangulatable_boundary_faces) {
    for (index_t face_id : boundary_faces) {
      auto verts = mesh.face_vertices(face_id);
      if (verts.size() < 3) continue;
      // Fan triangulation
      for (size_t i = 1; i + 1 < verts.size(); ++i) {
        TriangleWithFace tri;
        auto v0 = static_cast<size_t>(verts[0]);
        auto v1 = static_cast<size_t>(verts[i]);
        auto v2 = static_cast<size_t>(verts[i + 1]);
        if (v0 < vertex_coords.size() && v1 < vertex_coords.size() && v2 < vertex_coords.size()) {
          tri.vertices = {vertex_coords[v0], vertex_coords[v1], vertex_coords[v2]};
          tri.face_id = face_id;
          triangles.push_back(tri);
        }
      }
    }
    return triangles;
  }

  // Surface mesh fallback: triangulate 2D cells and use `face_id` as the cell id.
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    const auto& shape = mesh.cell_shape(c);
    if (!shape.is_2d()) continue;

    auto verts = mesh.cell_vertices(c);
    if (verts.size() < 3) continue;
    // Fan triangulation
    for (size_t i = 1; i + 1 < verts.size(); ++i) {
      TriangleWithFace tri;
      const size_t v0 = static_cast<size_t>(verts[0]);
      const size_t v1 = static_cast<size_t>(verts[i]);
      const size_t v2 = static_cast<size_t>(verts[i + 1]);
      if (v0 < vertex_coords.size() && v1 < vertex_coords.size() && v2 < vertex_coords.size()) {
        tri.vertices = {vertex_coords[v0], vertex_coords[v1], vertex_coords[v2]};
        tri.face_id = c;
        triangles.push_back(tri);
      }
    }
  }
  return triangles;
}

bool SearchBuilders::point_in_cell(const MeshBase& mesh,
                                   const std::vector<std::array<real_t,3>>& vertex_coords,
                                   index_t cell_id,
                                   const std::array<real_t,3>& p,
                                   std::array<real_t,3>& xi) {
  if (cell_id < 0 || static_cast<size_t>(cell_id) >= mesh.n_cells()) return false;
  auto shape = mesh.cell_shape(cell_id);
  auto verts_idx = mesh.cell_vertices(cell_id);
  std::vector<std::array<real_t,3>> verts;
  verts.reserve(verts_idx.size());
  for (auto vid : verts_idx) {
    if (vid < 0 || static_cast<size_t>(vid) >= vertex_coords.size()) return false;
    verts.push_back(vertex_coords[static_cast<size_t>(vid)]);
  }
  bool inside = search::point_in_cell(p, shape, verts);
  if (inside) {
    xi = SearchBuilders::compute_parametric_coords(mesh, cell_id, p, Configuration::Reference);
  }
  return inside;
}

// ---- Cell center computation ----

std::array<real_t,3> SearchBuilders::compute_cell_center(const MeshBase& mesh,
                                                         index_t cell_id,
                                                         Configuration cfg) {
  // Use the mesh's built-in cell_center method if available
  return mesh.cell_center(cell_id, cfg);
}

std::vector<std::array<real_t,3>> SearchBuilders::compute_all_cell_centers(
    const MeshBase& mesh,
    Configuration cfg) {

  size_t n_cells = mesh.n_cells();
  std::vector<std::array<real_t,3>> centers;
  centers.reserve(n_cells);

  for (size_t c = 0; c < n_cells; ++c) {
    centers.push_back(compute_cell_center(mesh, static_cast<index_t>(c), cfg));
  }

  return centers;
}

// ---- Neighbor information ----

std::vector<index_t> SearchBuilders::get_cell_neighbors(const MeshBase& mesh,
                                                        index_t cell_id) {
  std::unordered_set<index_t> neighbors;
  auto cell_faces = mesh.cell_faces(cell_id);

  for (index_t face_id : cell_faces) {
    auto face_cells = mesh.face_cells(face_id);

    if (face_cells[0] != cell_id && face_cells[0] != INVALID_INDEX) {
      neighbors.insert(face_cells[0]);
    }
    if (face_cells[1] != cell_id && face_cells[1] != INVALID_INDEX) {
      neighbors.insert(face_cells[1]);
    }
  }

  return std::vector<index_t>(neighbors.begin(), neighbors.end());
}

std::vector<index_t> SearchBuilders::get_vertex_cells(const MeshBase& mesh,
                                                      index_t vertex_id) {
  // This requires vertex-to-cell connectivity
  // For now, do a linear search
  std::vector<index_t> cells;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto vertices = mesh.cell_vertices(static_cast<index_t>(c));
    if (std::find(vertices.begin(), vertices.end(), vertex_id) != vertices.end()) {
      cells.push_back(static_cast<index_t>(c));
    }
  }

  return cells;
}

std::vector<index_t> SearchBuilders::get_edge_cells(const MeshBase& mesh,
                                                    index_t v0,
                                                    index_t v1) {
  std::vector<index_t> cells;
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto vertices = mesh.cell_vertices(static_cast<index_t>(c));
    bool has_v0 = std::find(vertices.begin(), vertices.end(), v0) != vertices.end();
    bool has_v1 = std::find(vertices.begin(), vertices.end(), v1) != vertices.end();

    if (has_v0 && has_v1) {
      cells.push_back(static_cast<index_t>(c));
    }
  }

  return cells;
}

// ---- Parametric coordinate helpers ----

std::array<real_t,3> SearchBuilders::tetra_parametric_coords(
    const std::array<real_t,3>& p,
    const std::vector<std::array<real_t,3>>& tet_vertices) {

  if (tet_vertices.size() != 4) {
    return {0, 0, 0};
  }

  // Use barycentric coordinates
  auto bary = tetrahedron_barycentric(p, tet_vertices[0], tet_vertices[1],
                                      tet_vertices[2], tet_vertices[3]);

  // Convert to parametric coordinates (xi, eta, zeta)
  // Using standard tetrahedron mapping
  return {bary[1], bary[2], bary[3]};
}

std::array<real_t,3> SearchBuilders::hex_parametric_coords(
    const std::array<real_t,3>& p,
    const std::vector<std::array<real_t,3>>& hex_vertices,
    int max_iter,
    real_t tol) {

  if (hex_vertices.size() != 8) {
    return {0, 0, 0};
  }

  // Initial guess at center of reference element
  std::array<real_t,3> xi = {0, 0, 0};

  // Reference-hex corner sign pattern (VTK/standard ordering):
  // 0:(-,-,-), 1:(+,-,-), 2:(+,+,-), 3:(-,+,-), 4:(-,-,+), 5:(+,-,+), 6:(+,+,+), 7:(-,+,+)
  static constexpr real_t sr[8] = {-1, 1, 1,-1,-1, 1, 1,-1};
  static constexpr real_t ss[8] = {-1,-1, 1, 1,-1,-1, 1, 1};
  static constexpr real_t st[8] = {-1,-1,-1,-1, 1, 1, 1, 1};

  auto solve_3x3 = [](const real_t J[3][3], const real_t b[3], real_t x[3]) -> bool {
    const real_t a00 = J[0][0], a01 = J[0][1], a02 = J[0][2];
    const real_t a10 = J[1][0], a11 = J[1][1], a12 = J[1][2];
    const real_t a20 = J[2][0], a21 = J[2][1], a22 = J[2][2];

    const real_t det =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);

    if (!std::isfinite(det) || std::abs(det) < 1e-14) {
      return false;
    }

    const real_t inv00 =  (a11 * a22 - a12 * a21) / det;
    const real_t inv01 =  (a02 * a21 - a01 * a22) / det;
    const real_t inv02 =  (a01 * a12 - a02 * a11) / det;
    const real_t inv10 =  (a12 * a20 - a10 * a22) / det;
    const real_t inv11 =  (a00 * a22 - a02 * a20) / det;
    const real_t inv12 =  (a02 * a10 - a00 * a12) / det;
    const real_t inv20 =  (a10 * a21 - a11 * a20) / det;
    const real_t inv21 =  (a01 * a20 - a00 * a21) / det;
    const real_t inv22 =  (a00 * a11 - a01 * a10) / det;

    x[0] = inv00 * b[0] + inv01 * b[1] + inv02 * b[2];
    x[1] = inv10 * b[0] + inv11 * b[1] + inv12 * b[2];
    x[2] = inv20 * b[0] + inv21 * b[1] + inv22 * b[2];
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
  };

  for (int iter = 0; iter < max_iter; ++iter) {
    const real_t r = xi[0];
    const real_t s = xi[1];
    const real_t t = xi[2];

    // Shape functions and derivatives at (r,s,t)
    real_t N[8];
    real_t dNdr[8];
    real_t dNds[8];
    real_t dNdt[8];

    for (int i = 0; i < 8; ++i) {
      const real_t ar = 1 + sr[i] * r;
      const real_t as = 1 + ss[i] * s;
      const real_t at = 1 + st[i] * t;
      N[i]    = 0.125 * ar * as * at;
      dNdr[i] = 0.125 * sr[i] * as * at;
      dNds[i] = 0.125 * ar * ss[i] * at;
      dNdt[i] = 0.125 * ar * as * st[i];
    }

    std::array<real_t,3> x_current = {0, 0, 0};
    for (int i = 0; i < 8; ++i) {
      for (int d = 0; d < 3; ++d) {
        x_current[d] += N[i] * hex_vertices[i][d];
      }
    }

    const real_t b[3] = {p[0] - x_current[0], p[1] - x_current[1], p[2] - x_current[2]};
    const real_t res_norm = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    if (res_norm < tol) {
      break;
    }

    real_t J[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (int i = 0; i < 8; ++i) {
      const auto& X = hex_vertices[i];
      J[0][0] += dNdr[i] * X[0];
      J[0][1] += dNds[i] * X[0];
      J[0][2] += dNdt[i] * X[0];

      J[1][0] += dNdr[i] * X[1];
      J[1][1] += dNds[i] * X[1];
      J[1][2] += dNdt[i] * X[1];

      J[2][0] += dNdr[i] * X[2];
      J[2][1] += dNds[i] * X[2];
      J[2][2] += dNdt[i] * X[2];
    }

    real_t delta[3] = {0, 0, 0};
    if (!solve_3x3(J, b, delta)) {
      break;
    }

    const real_t step_norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    xi[0] += delta[0];
    xi[1] += delta[1];
    xi[2] += delta[2];

    // Keep the iteration stable by clamping to the reference cube.
    for (int d = 0; d < 3; ++d) {
      xi[d] = std::max<real_t>(-1.0, std::min<real_t>(1.0, xi[d]));
    }

    if (step_norm < tol) {
      break;
    }
  }

  return xi;
}

std::array<real_t,3> SearchBuilders::compute_parametric_coords(
    const MeshBase& mesh,
    index_t cell_id,
    const std::array<real_t,3>& p,
    Configuration cfg) {

  // Treat Deformed as Current (compatibility alias).
  if (cfg == Configuration::Deformed) {
    cfg = Configuration::Current;
  }
  if (cfg == Configuration::Current && !mesh.has_current_coords()) {
    cfg = Configuration::Reference;
  }
  if (cell_id < 0 || static_cast<size_t>(cell_id) >= mesh.n_cells()) {
    return {0, 0, 0};
  }

  auto shape = mesh.cell_shape(cell_id);
  auto vertices = get_cell_vertex_coords(mesh, cell_id, cfg);

  switch (shape.family) {
    case CellFamily::Tetra:
      return tetra_parametric_coords(p, vertices);

    case CellFamily::Hex:
      return hex_parametric_coords(p, vertices);

    default:
      // Fallback: use a general inverse mapping for all supported families.
      return svmp::CurvilinearEvaluator::inverse_map(mesh, cell_id, p, cfg).first;
  }
}

// ---- Mesh statistics ----

real_t SearchBuilders::compute_mesh_characteristic_length(const MeshBase& mesh,
                                                          Configuration cfg) {
  // Compute average edge length
  real_t total_length = 0;
  size_t edge_count = 0;

  size_t n_edges = mesh.n_edges();
  for (size_t e = 0; e < n_edges; ++e) {
    auto vertices = mesh.edge_vertices(static_cast<index_t>(e));
    if (vertices.size() == 2) {
      auto v0 = get_vertex_coord(mesh, vertices[0], cfg);
      auto v1 = get_vertex_coord(mesh, vertices[1], cfg);

      real_t length = 0;
      for (int d = 0; d < 3; ++d) {
        real_t diff = v1[d] - v0[d];
        length += diff * diff;
      }
      total_length += std::sqrt(length);
      edge_count++;
    }
  }

  if (edge_count > 0) {
    return total_length / edge_count;
  }

  // Fallback: use mesh bounding box
  auto aabb = compute_mesh_aabb(mesh, cfg);
  auto extents = aabb.extents();
  return std::cbrt(extents[0] * extents[1] * extents[2]) / std::cbrt(mesh.n_cells());
}

int SearchBuilders::estimate_grid_resolution(const MeshBase& mesh,
                                             Configuration cfg,
                                             int target_cells_per_bucket) {
  size_t n_cells = mesh.n_cells();

  // Estimate resolution for roughly target_cells_per_bucket cells per grid cell
  int resolution = static_cast<int>(std::cbrt(n_cells / target_cells_per_bucket));

  // Clamp to reasonable range
  resolution = std::max(4, std::min(128, resolution));

  return resolution;
}

bool SearchBuilders::use_linear_search(const MeshBase& mesh,
                                       size_t threshold) {
  return mesh.n_cells() < threshold;
}

} // namespace search
} // namespace svmp
