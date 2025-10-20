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
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace svmp {
namespace search {

// ---- Coordinate extraction ----

std::vector<std::array<real_t,3>> SearchBuilders::extract_vertex_coords(
    const MeshBase& mesh,
    Configuration cfg) {

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
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

  // Initial guess at center
  std::array<real_t,3> xi = {0, 0, 0};

  // Newton-Raphson iteration
  for (int iter = 0; iter < max_iter; ++iter) {
    // Evaluate shape functions and derivatives
    // This is a simplified version - full implementation would use proper shape functions

    // Trilinear interpolation
    real_t N[8];
    N[0] = 0.125 * (1 - xi[0]) * (1 - xi[1]) * (1 - xi[2]);
    N[1] = 0.125 * (1 + xi[0]) * (1 - xi[1]) * (1 - xi[2]);
    N[2] = 0.125 * (1 + xi[0]) * (1 + xi[1]) * (1 - xi[2]);
    N[3] = 0.125 * (1 - xi[0]) * (1 + xi[1]) * (1 - xi[2]);
    N[4] = 0.125 * (1 - xi[0]) * (1 - xi[1]) * (1 + xi[2]);
    N[5] = 0.125 * (1 + xi[0]) * (1 - xi[1]) * (1 + xi[2]);
    N[6] = 0.125 * (1 + xi[0]) * (1 + xi[1]) * (1 + xi[2]);
    N[7] = 0.125 * (1 - xi[0]) * (1 + xi[1]) * (1 + xi[2]);

    // Compute current position
    std::array<real_t,3> x_current = {0, 0, 0};
    for (int i = 0; i < 8; ++i) {
      for (int d = 0; d < 3; ++d) {
        x_current[d] += N[i] * hex_vertices[i][d];
      }
    }

    // Check convergence
    real_t residual = 0;
    for (int d = 0; d < 3; ++d) {
      real_t diff = p[d] - x_current[d];
      residual += diff * diff;
    }

    if (std::sqrt(residual) < tol) {
      break;
    }

    // Update xi (simplified - full version would compute Jacobian)
    // This is a placeholder for proper Newton iteration
    for (int d = 0; d < 3; ++d) {
      xi[d] += 0.1 * (p[d] - x_current[d]);
      xi[d] = std::max(-1.0, std::min(1.0, xi[d]));
    }
  }

  return xi;
}

std::array<real_t,3> SearchBuilders::compute_parametric_coords(
    const MeshBase& mesh,
    index_t cell_id,
    const std::array<real_t,3>& p,
    Configuration cfg) {

  auto shape = mesh.cell_shape(cell_id);
  auto vertices = get_cell_vertex_coords(mesh, cell_id, cfg);

  switch (shape.family) {
    case CellFamily::Tetra:
      return tetra_parametric_coords(p, vertices);

    case CellFamily::Hex:
      return hex_parametric_coords(p, vertices);

    default:
      // For other shapes, return a default
      // Full implementation would handle all shape types
      return {0, 0, 0};
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