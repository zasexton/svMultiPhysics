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
#include "GeometryConfig.h"
#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"
#include <cmath>
#include <algorithm>

namespace svmp {

namespace {
inline const std::vector<real_t>& coords_for(const MeshBase& mesh, Configuration cfg) {
  return ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
             ? mesh.X_cur()
             : mesh.X_ref();
}

inline std::array<real_t,3> vertex_coords(const std::vector<real_t>& coords, int dim, index_t v) {
  std::array<real_t,3> pt = {{0.0, 0.0, 0.0}};
  const size_t base = static_cast<size_t>(v) * static_cast<size_t>(dim);
  if (dim >= 1) pt[0] = coords[base + 0];
  if (dim >= 2) pt[1] = coords[base + 1];
  if (dim >= 3) pt[2] = coords[base + 2];
  return pt;
}
} // namespace

// ---- Centers and bounding boxes ----

std::array<real_t,3> MeshGeometry::cell_center(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(cell);
  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
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

std::array<real_t,3> MeshGeometry::cell_centroid(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const auto shape = mesh.cell_shape(cell);
  auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(cell);
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
  const int dim = mesh.dim();

  auto get_coords = [&](index_t v) { return vertex_coords(coords, dim, v); };

  if (n_vertices == 0) {
    return {{0.0, 0.0, 0.0}};
  }

  switch (shape.family) {
    case CellFamily::Point:
      return get_coords(vertices_ptr[0]);

    case CellFamily::Line: {
      if (n_vertices >= 2) {
        const auto p0 = get_coords(vertices_ptr[0]);
        const auto p1 = get_coords(vertices_ptr[1]);
        return {{(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5}};
      }
      return get_coords(vertices_ptr[0]);
    }

    case CellFamily::Triangle: {
      if (n_vertices >= 3) {
        const auto p0 = get_coords(vertices_ptr[0]);
        const auto p1 = get_coords(vertices_ptr[1]);
        const auto p2 = get_coords(vertices_ptr[2]);
        return {{(p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0, (p0[2] + p1[2] + p2[2]) / 3.0}};
      }
      break;
    }

    case CellFamily::Quad:
    case CellFamily::Polygon: {
      if (n_vertices >= 3) {
        std::vector<index_t> ids;
        ids.reserve(n_vertices);
        for (size_t i = 0; i < n_vertices; ++i) ids.push_back(vertices_ptr[i]);
        return PolyGeometry::polygon_centroid(mesh, ids, cfg);
      }
      break;
    }

    case CellFamily::Tetra: {
      if (n_vertices >= 4) {
        const auto p0 = get_coords(vertices_ptr[0]);
        const auto p1 = get_coords(vertices_ptr[1]);
        const auto p2 = get_coords(vertices_ptr[2]);
        const auto p3 = get_coords(vertices_ptr[3]);
        return {{(p0[0] + p1[0] + p2[0] + p3[0]) / 4.0,
                 (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0,
                 (p0[2] + p1[2] + p2[2] + p3[2]) / 4.0}};
      }
      break;
    }

    case CellFamily::Hex: {
      if (n_vertices >= 8) {
        std::vector<std::array<real_t,3>> v;
        v.reserve(8);
        for (size_t i = 0; i < 8; ++i) v.push_back(get_coords(vertices_ptr[i]));

        static constexpr int tets[6][4] = {
            {0, 1, 2, 6},
            {0, 2, 3, 6},
            {0, 3, 7, 6},
            {0, 7, 4, 6},
            {0, 4, 5, 6},
            {0, 5, 1, 6},
        };

        real_t vol_sum = 0.0;
        std::array<real_t,3> csum = {{0.0, 0.0, 0.0}};
        for (const auto& tet : tets) {
          const real_t vtet = std::abs(tet_volume(v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]));
          if (vtet <= GeometryConfig::volume_epsilon()) continue;
          const auto c = std::array<real_t,3>{
              (v[tet[0]][0] + v[tet[1]][0] + v[tet[2]][0] + v[tet[3]][0]) / 4.0,
              (v[tet[0]][1] + v[tet[1]][1] + v[tet[2]][1] + v[tet[3]][1]) / 4.0,
              (v[tet[0]][2] + v[tet[1]][2] + v[tet[2]][2] + v[tet[3]][2]) / 4.0,
          };
          csum[0] += vtet * c[0];
          csum[1] += vtet * c[1];
          csum[2] += vtet * c[2];
          vol_sum += vtet;
        }
        if (vol_sum > GeometryConfig::volume_epsilon()) {
          return {{csum[0] / vol_sum, csum[1] / vol_sum, csum[2] / vol_sum}};
        }
      }
      break;
    }

    case CellFamily::Wedge: {
      if (n_vertices >= 6) {
        std::vector<std::array<real_t,3>> v;
        v.reserve(6);
        for (size_t i = 0; i < 6; ++i) v.push_back(get_coords(vertices_ptr[i]));

        static constexpr int tets[3][4] = {
            {0, 1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4, 5},
        };

        real_t vol_sum = 0.0;
        std::array<real_t,3> csum = {{0.0, 0.0, 0.0}};
        for (const auto& tet : tets) {
          const real_t vtet = std::abs(tet_volume(v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]));
          if (vtet <= GeometryConfig::volume_epsilon()) continue;
          const auto c = std::array<real_t,3>{
              (v[tet[0]][0] + v[tet[1]][0] + v[tet[2]][0] + v[tet[3]][0]) / 4.0,
              (v[tet[0]][1] + v[tet[1]][1] + v[tet[2]][1] + v[tet[3]][1]) / 4.0,
              (v[tet[0]][2] + v[tet[1]][2] + v[tet[2]][2] + v[tet[3]][2]) / 4.0,
          };
          csum[0] += vtet * c[0];
          csum[1] += vtet * c[1];
          csum[2] += vtet * c[2];
          vol_sum += vtet;
        }
        if (vol_sum > GeometryConfig::volume_epsilon()) {
          return {{csum[0] / vol_sum, csum[1] / vol_sum, csum[2] / vol_sum}};
        }
      }
      break;
    }

    case CellFamily::Pyramid: {
      if (n_vertices >= 5) {
        std::vector<std::array<real_t,3>> v;
        v.reserve(5);
        for (size_t i = 0; i < 5; ++i) v.push_back(get_coords(vertices_ptr[i]));

        static constexpr int tets[2][4] = {
            {0, 1, 2, 4},
            {0, 2, 3, 4},
        };

        real_t vol_sum = 0.0;
        std::array<real_t,3> csum = {{0.0, 0.0, 0.0}};
        for (const auto& tet : tets) {
          const real_t vtet = std::abs(tet_volume(v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]));
          if (vtet <= GeometryConfig::volume_epsilon()) continue;
          const auto c = std::array<real_t,3>{
              (v[tet[0]][0] + v[tet[1]][0] + v[tet[2]][0] + v[tet[3]][0]) / 4.0,
              (v[tet[0]][1] + v[tet[1]][1] + v[tet[2]][1] + v[tet[3]][1]) / 4.0,
              (v[tet[0]][2] + v[tet[1]][2] + v[tet[2]][2] + v[tet[3]][2]) / 4.0,
          };
          csum[0] += vtet * c[0];
          csum[1] += vtet * c[1];
          csum[2] += vtet * c[2];
          vol_sum += vtet;
        }
        if (vol_sum > GeometryConfig::volume_epsilon()) {
          return {{csum[0] / vol_sum, csum[1] / vol_sum, csum[2] / vol_sum}};
        }
      }
      break;
    }

    case CellFamily::Polyhedron:
      return PolyGeometry::polyhedron_centroid(mesh, cell, cfg);
  }

  // Fallback: vertex-average center.
  return cell_center(mesh, cell, cfg);
}

std::array<real_t,3> MeshGeometry::face_center(const MeshBase& mesh, index_t face, Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.face_vertices_span(face);
  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
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

std::array<real_t,3> MeshGeometry::edge_center(const MeshBase& mesh, index_t edge, Configuration cfg) {
  if (edge < 0 || static_cast<size_t>(edge) >= mesh.n_edges()) {
    return {{0.0, 0.0, 0.0}};
  }

  const auto& coords = coords_for(mesh, cfg);
  const int dim = mesh.dim();
  const auto ev = mesh.edge_vertices(edge);

  const auto p0 = vertex_coords(coords, dim, ev[0]);
  const auto p1 = vertex_coords(coords, dim, ev[1]);
  return {{(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5}};
}

BoundingBox MeshGeometry::bounding_box(const MeshBase& mesh, Configuration cfg) {
  BoundingBox box;
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
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

BoundingBox MeshGeometry::cell_bounding_box(const MeshBase& mesh, index_t cell, Configuration cfg) {
  BoundingBox box;
  auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(cell);
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
  const int dim = mesh.dim();

  for (size_t i = 0; i < n_vertices; ++i) {
    const auto p = vertex_coords(coords, dim, vertices_ptr[i]);
    for (int d = 0; d < dim; ++d) {
      box.min[d] = std::min(box.min[d], p[d]);
      box.max[d] = std::max(box.max[d], p[d]);
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
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
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

std::array<real_t,3> MeshGeometry::edge_normal(const MeshBase& mesh, index_t edge, Configuration cfg) {
  if (edge < 0 || static_cast<size_t>(edge) >= mesh.n_edges()) {
    return {{0.0, 0.0, 0.0}};
  }
  const auto ev = mesh.edge_vertices(edge);
  const std::vector<index_t> ids = {ev[0], ev[1]};
  return normalize(compute_edge_normal_from_vertices(mesh, ids, cfg));
}

// ---- Measures ----

real_t MeshGeometry::cell_measure(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto [vertices_ptr2, n_vertices2] = mesh.cell_vertices_span(cell);
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
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

    case CellFamily::Polygon: {
      if (n_vertices2 >= 3) {
        std::vector<index_t> ids;
        ids.reserve(n_vertices2);
        for (size_t i = 0; i < n_vertices2; ++i) ids.push_back(vertices_ptr2[i]);
        return PolyGeometry::polygon_area(mesh, ids, cfg);
      }
      break;
    }

    case CellFamily::Polyhedron: {
      // Requires explicit face connectivity (mesh.face_vertices_span + mesh.cell_faces).
      return PolyGeometry::polyhedron_volume(mesh, cell, cfg);
    }

    default:
      break;
  }

  return 0.0;
}

real_t MeshGeometry::face_area(const MeshBase& mesh, index_t face, Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.face_vertices_span(face);
  const std::vector<real_t>& coords = coords_for(mesh, cfg);
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
  // Decompose hex into 6 tetrahedra around a body diagonal (0,6).
  // Standard hex vertex ordering: 0-1-2-3 bottom, 4-5-6-7 top.
  static constexpr int tets[6][4] = {
      {0, 1, 2, 6},
      {0, 2, 3, 6},
      {0, 3, 7, 6},
      {0, 7, 4, 6},
      {0, 4, 5, 6},
      {0, 5, 1, 6},
  };
  real_t vol = 0.0;
  for (const auto& tet : tets) {
    vol += std::abs(tet_volume(v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]));
  }
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

  const int dim = mesh.dim();
  if (oriented_vertices.size() < 2 || dim < 2) {
    return {{0.0, 0.0, 0.0}};
  }

  if (dim == 2 || oriented_vertices.size() == 2) {
    return compute_edge_normal_from_vertices(mesh, oriented_vertices, cfg);
  }

  if (oriented_vertices.size() < 3) {
    return {{0.0, 0.0, 0.0}};
  }

  // 3D polygon normal via Newell's method (robust to collinear early vertices).
  return PolyGeometry::polygon_normal(mesh, oriented_vertices, cfg);
}

std::array<real_t,3> MeshGeometry::compute_edge_normal_from_vertices(
    const MeshBase& mesh,
    const std::vector<index_t>& oriented_vertices,
    Configuration cfg) {

  if (oriented_vertices.size() < 2) {
    return {{0.0, 0.0, 0.0}};
  }

  // Get vertex coordinates from appropriate configuration
  const auto& X = coords_for(mesh, cfg);
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
    const auto& X = coords_for(mesh, cfg);
    int dim = mesh.dim();
    centroid[0] = X[vertices[0]*dim + 0];
    centroid[1] = (dim > 1 ? X[vertices[0]*dim + 1] : 0.0);
    centroid[2] = (dim > 2 ? X[vertices[0]*dim + 2] : 0.0);
    return centroid;
  }
  if (vertices.size() == 2) {
    const auto& X = coords_for(mesh, cfg);
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

  const auto& X = coords_for(mesh, cfg);
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

  const auto& X = coords_for(mesh, cfg);
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

real_t MeshGeometry::edge_length(const MeshBase& mesh, index_t edge, Configuration cfg) {
  if (edge < 0 || static_cast<size_t>(edge) >= mesh.n_edges()) {
    return 0.0;
  }
  const auto ev = mesh.edge_vertices(edge);
  const auto& coords = coords_for(mesh, cfg);
  const int dim = mesh.dim();
  const auto p0 = vertex_coords(coords, dim, ev[0]);
  const auto p1 = vertex_coords(coords, dim, ev[1]);
  return distance(p0, p1);
}

real_t MeshGeometry::total_volume(const MeshBase& mesh, Configuration cfg) {
  real_t total = 0.0;
  const index_t n_cells = static_cast<index_t>(mesh.n_cells());
  for (index_t c = 0; c < n_cells; ++c) {
    total += cell_measure(mesh, c, cfg);
  }
  return total;
}

real_t MeshGeometry::total_volume_global(const DistributedMesh& mesh, Configuration cfg) {
  const auto& local = mesh.local_mesh();
  real_t local_total = 0.0;

  const index_t n_cells = static_cast<index_t>(local.n_cells());
  for (index_t c = 0; c < n_cells; ++c) {
    if (!mesh.is_owned_cell(c)) {
      continue;
    }
    local_total += cell_measure(local, c, cfg);
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL || mesh.world_size() <= 1) {
    return local_total;
  }

  real_t global_total = 0.0;
  MPI_Allreduce(&local_total, &global_total, 1, MPI_DOUBLE, MPI_SUM, mesh.mpi_comm());
  return global_total;
#else
  return local_total;
#endif
}

real_t MeshGeometry::boundary_area(const MeshBase& mesh, Configuration cfg) {
  real_t total = 0.0;
  for (index_t f : mesh.boundary_faces()) {
    total += face_area(mesh, f, cfg);
  }
  return total;
}

real_t MeshGeometry::boundary_area_global(const DistributedMesh& mesh, Configuration cfg) {
  const auto& local = mesh.local_mesh();
  real_t local_total = 0.0;

  for (const index_t f : local.boundary_faces()) {
    if (!mesh.is_owned_face(f)) {
      continue;
    }
    local_total += face_area(local, f, cfg);
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL || mesh.world_size() <= 1) {
    return local_total;
  }

  real_t global_total = 0.0;
  MPI_Allreduce(&local_total, &global_total, 1, MPI_DOUBLE, MPI_SUM, mesh.mpi_comm());
  return global_total;
#else
  return local_total;
#endif
}

real_t MeshGeometry::angle(const std::array<real_t,3>& p1,
                           const std::array<real_t,3>& p2,
                           const std::array<real_t,3>& p3) {
  const std::array<real_t,3> v1 = {{p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]}};
  const std::array<real_t,3> v2 = {{p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]}};

  const real_t l1 = magnitude(v1);
  const real_t l2 = magnitude(v2);
  if (l1 < 1e-20 || l2 < 1e-20) return 0.0;

  real_t c = dot(v1, v2) / (l1 * l2);
  c = std::max(static_cast<real_t>(-1.0), std::min(static_cast<real_t>(1.0), c));
  return std::acos(c);
}

real_t MeshGeometry::dihedral_angle(const std::array<real_t,3>& n1,
                                    const std::array<real_t,3>& n2) {
  const real_t l1 = magnitude(n1);
  const real_t l2 = magnitude(n2);
  if (l1 < 1e-20 || l2 < 1e-20) return 0.0;

  real_t c = dot(n1, n2) / (l1 * l2);
  c = std::max(static_cast<real_t>(-1.0), std::min(static_cast<real_t>(1.0), c));
  return std::acos(c);
}

std::vector<std::array<real_t,3>> MeshGeometry::get_cell_vertices(const MeshBase& mesh,
                                                                  index_t cell,
                                                                  Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(cell);
  std::vector<std::array<real_t,3>> pts;
  pts.reserve(n_vertices);
  const auto& coords = coords_for(mesh, cfg);
  const int dim = mesh.dim();
  for (size_t i = 0; i < n_vertices; ++i) {
    pts.push_back(vertex_coords(coords, dim, vertices_ptr[i]));
  }
  return pts;
}

std::vector<std::array<real_t,3>> MeshGeometry::get_face_vertices(const MeshBase& mesh,
                                                                  index_t face,
                                                                  Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.face_vertices_span(face);
  std::vector<std::array<real_t,3>> pts;
  pts.reserve(n_vertices);
  const auto& coords = coords_for(mesh, cfg);
  const int dim = mesh.dim();
  for (size_t i = 0; i < n_vertices; ++i) {
    pts.push_back(vertex_coords(coords, dim, vertices_ptr[i]));
  }
  return pts;
}

} // namespace svmp
