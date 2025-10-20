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

#include "MeshGeometry.h"
#include "PolyGeometry.h"
#include "../Core/MeshBase.h"
#include <cmath>
#include <algorithm>

namespace svmp {

// ---- Centers and bounding boxes ----

std::array<real_t,3> MeshGeometry::cell_center(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(cell);
  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  for (size_t i = 0; i < n_vertices; ++i) {
    index_t vertex_id = vertices_ptr[i];
    for (int d = 0; d < spatial_dim; ++d) {
      center[d] += coords[vertex_id * spatial_dim + d];
    }
  }

  if (n_vertices > 0) {
    for (int d = 0; d < 3; ++d) {
      center[d] /= n_vertices;
    }
  }

  return center;
}

std::array<real_t,3> MeshGeometry::face_center(const MeshBase& mesh, index_t face, Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.face_vertices_span(face);
  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  for (size_t i = 0; i < n_vertices; ++i) {
    index_t vertex_id = vertices_ptr[i];
    for (int d = 0; d < spatial_dim; ++d) {
      center[d] += coords[vertex_id * spatial_dim + d];
    }
  }

  if (n_vertices > 0) {
    for (int d = 0; d < 3; ++d) {
      center[d] /= n_vertices;
    }
  }

  return center;
}

BoundingBox MeshGeometry::bounding_box(const MeshBase& mesh, Configuration cfg) {
  BoundingBox box;
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  for (size_t i = 0; i < n_vertices; ++i) {
    for (int d = 0; d < spatial_dim; ++d) {
      real_t val = coords[i * spatial_dim + d];
      box.min[d] = std::min(box.min[d], val);
      box.max[d] = std::max(box.max[d], val);
    }
  }

  return box;
}

// ---- Normals ----

std::array<real_t,3> MeshGeometry::face_normal(const MeshBase& mesh, index_t face, Configuration cfg) {
  auto normal = face_normal_unnormalized(mesh, face, cfg);
  return normalize(normal);
}

std::array<real_t,3> MeshGeometry::face_normal_unnormalized(const MeshBase& mesh, index_t face, Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.face_vertices_span(face);
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();
  std::array<real_t,3> normal = {{0,0,0}};

  if (spatial_dim == 2) {
    // 2D: normal to line segment
    if (n_vertices >= 2) {
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      for (int d = 0; d < spatial_dim; ++d) {
        p0[d] = coords[vertices_ptr[0] * spatial_dim + d];
        p1[d] = coords[vertices_ptr[1] * spatial_dim + d];
      }
      // 90-degree rotation in 2D
      normal[0] = -(p1[1] - p0[1]);
      normal[1] = p1[0] - p0[0];
    }
  } else if (spatial_dim == 3) {
    // 3D: robust Newell normal over the polygon vertices
    if (n_vertices >= 3) {
      std::vector<index_t> ids;
      ids.reserve(n_vertices);
      for (size_t i = 0; i < n_vertices; ++i) ids.push_back(vertices_ptr[i]);
      normal = PolyGeometry::polygon_normal(mesh, ids, cfg);
    }
  }

  return normal;
}

// ---- Measures ----

real_t MeshGeometry::cell_measure(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto [vertices_ptr2, n_vertices2] = mesh.cell_vertices_span(cell);
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  const CellShape& shape = mesh.cell_shape(cell);
  int spatial_dim = mesh.dim();

  // Helper to get coordinates
  auto get_coords = [&](index_t vertex_id) -> std::array<real_t,3> {
    std::array<real_t,3> pt = {{0,0,0}};
    for (int d = 0; d < spatial_dim; ++d) {
      pt[d] = coords[vertex_id * spatial_dim + d];
    }
    return pt;
  };

  switch (shape.family) {
    case CellFamily::Line: {
      if (n_vertices2 >= 2) {
        auto p0 = get_coords(vertices_ptr2[0]);
        auto p1 = get_coords(vertices_ptr2[1]);
        return distance(p0, p1);
      }
      break;
    }

    case CellFamily::Triangle: {
      if (n_vertices2 >= 3) {
        auto p0 = get_coords(vertices_ptr2[0]);
        auto p1 = get_coords(vertices_ptr2[1]);
        auto p2 = get_coords(vertices_ptr2[2]);
        return triangle_area(p0, p1, p2);
      }
      break;
    }

    case CellFamily::Quad: {
      if (n_vertices2 >= 4) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 4; ++i) {
          verts.push_back(get_coords(vertices_ptr2[i]));
        }
        return quad_area(verts);
      }
      break;
    }

    case CellFamily::Tetra: {
      if (n_vertices2 >= 4) {
        auto p0 = get_coords(vertices_ptr2[0]);
        auto p1 = get_coords(vertices_ptr2[1]);
        auto p2 = get_coords(vertices_ptr2[2]);
        auto p3 = get_coords(vertices_ptr2[3]);
        return std::abs(tet_volume(p0, p1, p2, p3));
      }
      break;
    }

    case CellFamily::Hex: {
      if (n_vertices2 >= 8) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 8; ++i) {
          verts.push_back(get_coords(vertices_ptr2[i]));
        }
        return hex_volume(verts);
      }
      break;
    }

    case CellFamily::Wedge: {
      if (n_vertices2 >= 6) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 6; ++i) {
          verts.push_back(get_coords(vertices_ptr2[i]));
        }
        return wedge_volume(verts);
      }
      break;
    }

    case CellFamily::Pyramid: {
      if (n_vertices2 >= 5) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 5; ++i) {
          verts.push_back(get_coords(vertices_ptr2[i]));
        }
        return pyramid_volume(verts);
      }
      break;
    }

    default:
      // For polygons/polyhedra, return 0 as placeholder
      break;
  }

  return 0.0;
}

real_t MeshGeometry::face_area(const MeshBase& mesh, index_t face, Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.face_vertices_span(face);
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  if (spatial_dim == 2) {
    // In 2D, "faces" are edges, so return edge length
    if (n_vertices >= 2) {
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      for (int d = 0; d < spatial_dim; ++d) {
        p0[d] = coords[vertices_ptr[0] * spatial_dim + d];
        p1[d] = coords[vertices_ptr[1] * spatial_dim + d];
      }
      return distance(p0, p1);
    }
  } else if (spatial_dim == 3) {
    // 3D face area via polygon formula for any n >= 3
    if (n_vertices >= 3) {
      std::vector<index_t> ids;
      ids.reserve(n_vertices);
      for (size_t i = 0; i < n_vertices; ++i) ids.push_back(vertices_ptr[i]);
      return PolyGeometry::polygon_area(mesh, ids, cfg);
    }
  }

  return 0.0;
}

// ---- Specialized shape measures ----

real_t MeshGeometry::tet_volume(const std::array<real_t,3>& p0,
                               const std::array<real_t,3>& p1,
                               const std::array<real_t,3>& p2,
                               const std::array<real_t,3>& p3) {
  // Volume = 1/6 * |det([p1-p0, p2-p0, p3-p0])|
  std::array<real_t,3> v1, v2, v3;
  for (int i = 0; i < 3; ++i) {
    v1[i] = p1[i] - p0[i];
    v2[i] = p2[i] - p0[i];
    v3[i] = p3[i] - p0[i];
  }

  // Scalar triple product: v1 · (v2 × v3)
  auto cross_v2_v3 = cross(v2, v3);
  real_t det = dot(v1, cross_v2_v3);

  return det / 6.0;  // Signed volume
}

real_t MeshGeometry::triangle_area(const std::array<real_t,3>& p0,
                                  const std::array<real_t,3>& p1,
                                  const std::array<real_t,3>& p2) {
  std::array<real_t,3> v1, v2;
  for (int i = 0; i < 3; ++i) {
    v1[i] = p1[i] - p0[i];
    v2[i] = p2[i] - p0[i];
  }
  auto c = cross(v1, v2);
  return 0.5 * magnitude(c);
}

real_t MeshGeometry::hex_volume(const std::vector<std::array<real_t,3>>& v) {
  // Decompose hex into 6 tetrahedra
  // Standard hex vertex ordering: 0-1-2-3 bottom, 4-5-6-7 top
  real_t vol = 0;
  vol += std::abs(tet_volume(v[0], v[1], v[3], v[4]));
  vol += std::abs(tet_volume(v[1], v[2], v[3], v[6]));
  vol += std::abs(tet_volume(v[1], v[3], v[4], v[6]));
  vol += std::abs(tet_volume(v[3], v[4], v[6], v[7]));
  vol += std::abs(tet_volume(v[1], v[4], v[5], v[6]));
  vol += std::abs(tet_volume(v[1], v[6], v[4], v[3]));
  return vol;
}

real_t MeshGeometry::wedge_volume(const std::vector<std::array<real_t,3>>& v) {
  // Decompose wedge into 3 tetrahedra
  real_t vol = 0;
  vol += std::abs(tet_volume(v[0], v[1], v[2], v[3]));
  vol += std::abs(tet_volume(v[1], v[2], v[3], v[4]));
  vol += std::abs(tet_volume(v[2], v[3], v[4], v[5]));
  return vol;
}

real_t MeshGeometry::pyramid_volume(const std::vector<std::array<real_t,3>>& v) {
  // Decompose pyramid into 2 tetrahedra
  real_t vol = 0;
  vol += std::abs(tet_volume(v[0], v[1], v[2], v[4]));
  vol += std::abs(tet_volume(v[0], v[2], v[3], v[4]));
  return vol;
}

real_t MeshGeometry::quad_area(const std::vector<std::array<real_t,3>>& v) {
  // Split quad into two triangles
  real_t area = 0;
  area += triangle_area(v[0], v[1], v[2]);
  area += triangle_area(v[0], v[2], v[3]);
  return area;
}

// ---- Vector operations ----

std::array<real_t,3> MeshGeometry::cross(const std::array<real_t,3>& a,
                                        const std::array<real_t,3>& b) {
  return {{
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  }};
}

real_t MeshGeometry::dot(const std::array<real_t,3>& a,
                        const std::array<real_t,3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<real_t,3> MeshGeometry::normalize(const std::array<real_t,3>& v) {
  real_t mag = magnitude(v);
  if (mag < 1e-12) return {{0,0,0}};
  return {{v[0]/mag, v[1]/mag, v[2]/mag}};
}

real_t MeshGeometry::magnitude(const std::array<real_t,3>& v) {
  return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

real_t MeshGeometry::distance(const std::array<real_t,3>& p1,
                             const std::array<real_t,3>& p2) {
  real_t dx = p2[0] - p1[0];
  real_t dy = p2[1] - p1[1];
  real_t dz = p2[2] - p1[2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ---- Boundary-specific geometry ----

std::array<real_t,3> MeshGeometry::compute_normal_from_vertices(
    const MeshBase& mesh,
    const std::vector<index_t>& oriented_vertices,
    Configuration cfg) {

  if (oriented_vertices.size() < 3) {
    return {{0.0, 0.0, 0.0}};
  }

  // Get vertex coordinates from appropriate configuration
  const auto& X = (cfg == Configuration::Current && mesh.has_current_coords())
                  ? mesh.X_cur() : mesh.X_ref();
  const int dim = mesh.dim();

  const index_t v0 = oriented_vertices[0];
  const index_t v1 = oriented_vertices[1];
  const index_t v2 = oriented_vertices[2];

  // Extract coordinates (stored as flat array: x0,y0,z0,x1,y1,z1,...)
  std::array<real_t,3> p0 = {{X[v0*dim + 0], X[v0*dim + 1], dim > 2 ? X[v0*dim + 2] : 0.0}};
  std::array<real_t,3> p1 = {{X[v1*dim + 0], X[v1*dim + 1], dim > 2 ? X[v1*dim + 2] : 0.0}};
  std::array<real_t,3> p2 = {{X[v2*dim + 0], X[v2*dim + 1], dim > 2 ? X[v2*dim + 2] : 0.0}};

  // Compute edge vectors
  std::array<real_t,3> e1, e2;
  for (int d = 0; d < 3; ++d) {
    e1[d] = p1[d] - p0[d];
    e2[d] = p2[d] - p0[d];
  }

  // Compute cross product: normal = e1 × e2
  return cross(e1, e2);
}

std::array<real_t,3> MeshGeometry::compute_edge_normal_from_vertices(
    const MeshBase& mesh,
    const std::vector<index_t>& oriented_vertices,
    Configuration cfg) {

  if (oriented_vertices.size() < 2) {
    return {{0.0, 0.0, 0.0}};
  }

  // Get vertex coordinates from appropriate configuration
  const auto& X = (cfg == Configuration::Current && mesh.has_current_coords())
                  ? mesh.X_cur() : mesh.X_ref();
  const int dim = mesh.dim();

  const index_t v0 = oriented_vertices[0];
  const index_t v1 = oriented_vertices[1];

  // Extract coordinates
  std::array<real_t,3> p0 = {{0.0, 0.0, 0.0}};
  std::array<real_t,3> p1 = {{0.0, 0.0, 0.0}};

  for (int d = 0; d < dim; ++d) {
    p0[d] = X[v0*dim + d];
    p1[d] = X[v1*dim + d];
  }

  // Edge vector
  std::array<real_t,3> edge;
  for (int d = 0; d < 3; ++d) {
    edge[d] = p1[d] - p0[d];
  }

  // Compute normal perpendicular to edge
  if (dim == 2) {
    // 2D: Outward normal (90-degree rotation counterclockwise in xy-plane)
    return {{-edge[1], edge[0], 0.0}};
  } else {
    // 3D: Return a perpendicular vector (arbitrary choice without additional context)
    // Choose a perpendicular direction using cross product with a reference vector
    std::array<real_t,3> ref = {{1.0, 0.0, 0.0}};

    // If edge is nearly parallel to ref, use a different reference
    real_t dot_prod = edge[0] * ref[0] + edge[1] * ref[1] + edge[2] * ref[2];
    real_t edge_mag = magnitude(edge);
    if (std::abs(dot_prod) > 0.9 * edge_mag) {
      ref = {{0.0, 1.0, 0.0}};
    }

    // Return perpendicular via cross product
    return cross(edge, ref);
  }
}

std::array<real_t,3> MeshGeometry::compute_centroid_from_vertices(
    const MeshBase& mesh,
    const std::vector<index_t>& vertices,
    Configuration cfg) {

  std::array<real_t,3> centroid = {{0.0, 0.0, 0.0}};
  if (vertices.empty()) return centroid;
  if (vertices.size() == 1) {
    const auto& X = (cfg == Configuration::Current && mesh.has_current_coords())
                    ? mesh.X_cur() : mesh.X_ref();
    int dim = mesh.dim();
    centroid[0] = X[vertices[0]*dim + 0];
    centroid[1] = (dim > 1 ? X[vertices[0]*dim + 1] : 0.0);
    centroid[2] = (dim > 2 ? X[vertices[0]*dim + 2] : 0.0);
    return centroid;
  }
  if (vertices.size() == 2) {
    const auto& X = (cfg == Configuration::Current && mesh.has_current_coords())
                    ? mesh.X_cur() : mesh.X_ref();
    int dim = mesh.dim();
    centroid[0] = (X[vertices[0]*dim + 0] + X[vertices[1]*dim + 0]) / 2.0;
    centroid[1] = (dim > 1 ? (X[vertices[0]*dim + 1] + X[vertices[1]*dim + 1]) / 2.0 : 0.0);
    centroid[2] = (dim > 2 ? (X[vertices[0]*dim + 2] + X[vertices[1]*dim + 2]) / 2.0 : 0.0);
    return centroid;
  }
  // Polygon centroid (area-weighted)
  return PolyGeometry::polygon_centroid(mesh, vertices, cfg);
}

real_t MeshGeometry::compute_area_from_vertices(
    const MeshBase& mesh,
    const std::vector<index_t>& oriented_vertices,
    Configuration cfg) {

  if (oriented_vertices.size() < 2) {
    return 0.0;
  }

  const auto& X = (cfg == Configuration::Current && mesh.has_current_coords())
                  ? mesh.X_cur() : mesh.X_ref();
  const int dim = mesh.dim();

  // Helper to get coordinates
  auto get_coords = [&](index_t v) -> std::array<real_t,3> {
    std::array<real_t,3> pt = {{0.0, 0.0, 0.0}};
    for (int d = 0; d < dim; ++d) {
      pt[d] = X[v*dim + d];
    }
    return pt;
  };

  if (dim == 2) {
    // 1D boundary (edge): compute length
    if (oriented_vertices.size() >= 2) {
      auto p0 = get_coords(oriented_vertices[0]);
      auto p1 = get_coords(oriented_vertices[1]);
      return distance(p0, p1);
    }
  } else if (dim == 3) {
    // 2D boundary (face): compute area
    if (oriented_vertices.size() == 3) {
      // Triangle
      auto p0 = get_coords(oriented_vertices[0]);
      auto p1 = get_coords(oriented_vertices[1]);
      auto p2 = get_coords(oriented_vertices[2]);
      return triangle_area(p0, p1, p2);
    } else if (oriented_vertices.size() == 4) {
      // Quadrilateral
      std::vector<std::array<real_t,3>> verts;
      for (index_t v : oriented_vertices) {
        verts.push_back(get_coords(v));
      }
      return quad_area(verts);
    } else if (oriented_vertices.size() > 2) {
      // General polygon (tri/quad/poly)
      return PolyGeometry::polygon_area(mesh, oriented_vertices, cfg);
    }
  }

  return 0.0;
}

real_t MeshGeometry::polygon_area(const std::vector<std::array<real_t,3>>& vertices) {
  return PolyGeometry::polygon_area(vertices);
}

BoundingBox MeshGeometry::compute_bounding_box_from_vertices(
    const MeshBase& mesh,
    const std::vector<index_t>& vertices,
    Configuration cfg) {

  BoundingBox box;

  if (vertices.empty()) {
    return box;
  }

  const auto& X = (cfg == Configuration::Current && mesh.has_current_coords())
                  ? mesh.X_cur() : mesh.X_ref();
  const int dim = mesh.dim();

  // Initialize with first vertex
  for (int d = 0; d < dim; ++d) {
    box.min[d] = X[vertices[0]*dim + d];
    box.max[d] = X[vertices[0]*dim + d];
  }

  // Expand to include all vertices
  for (size_t i = 1; i < vertices.size(); ++i) {
    for (int d = 0; d < dim; ++d) {
      real_t val = X[vertices[i]*dim + d];
      box.min[d] = std::min(box.min[d], val);
      box.max[d] = std::max(box.max[d], val);
    }
  }

  return box;
}

} // namespace svmp
