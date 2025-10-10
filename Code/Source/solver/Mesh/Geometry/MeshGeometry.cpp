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
#include "../Mesh.h"
#include <cmath>
#include <algorithm>

namespace svmp {

// ---- Centers and bounding boxes ----

std::array<real_t,3> MeshGeometry::cell_center(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(cell);
  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  for (size_t i = 0; i < n_nodes; ++i) {
    index_t node_id = nodes_ptr[i];
    for (int d = 0; d < spatial_dim; ++d) {
      center[d] += coords[node_id * spatial_dim + d];
    }
  }

  if (n_nodes > 0) {
    for (int d = 0; d < 3; ++d) {
      center[d] /= n_nodes;
    }
  }

  return center;
}

std::array<real_t,3> MeshGeometry::face_center(const MeshBase& mesh, index_t face, Configuration cfg) {
  auto [nodes_ptr, n_nodes] = mesh.face_nodes_span(face);
  std::array<real_t,3> center = {{0,0,0}};
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  for (size_t i = 0; i < n_nodes; ++i) {
    index_t node_id = nodes_ptr[i];
    for (int d = 0; d < spatial_dim; ++d) {
      center[d] += coords[node_id * spatial_dim + d];
    }
  }

  if (n_nodes > 0) {
    for (int d = 0; d < 3; ++d) {
      center[d] /= n_nodes;
    }
  }

  return center;
}

BoundingBox MeshGeometry::bounding_box(const MeshBase& mesh, Configuration cfg) {
  BoundingBox box;
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();
  size_t n_nodes = mesh.n_nodes();

  for (size_t i = 0; i < n_nodes; ++i) {
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
  auto [nodes_ptr, n_nodes] = mesh.face_nodes_span(face);
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();
  std::array<real_t,3> normal = {{0,0,0}};

  if (spatial_dim == 2) {
    // 2D: normal to line segment
    if (n_nodes >= 2) {
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      for (int d = 0; d < spatial_dim; ++d) {
        p0[d] = coords[nodes_ptr[0] * spatial_dim + d];
        p1[d] = coords[nodes_ptr[1] * spatial_dim + d];
      }
      // 90-degree rotation in 2D
      normal[0] = -(p1[1] - p0[1]);
      normal[1] = p1[0] - p0[0];
    }
  } else if (spatial_dim == 3) {
    // 3D: cross product of two edges
    if (n_nodes >= 3) {
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      std::array<real_t,3> p2 = {{0,0,0}};
      for (int d = 0; d < spatial_dim; ++d) {
        p0[d] = coords[nodes_ptr[0] * spatial_dim + d];
        p1[d] = coords[nodes_ptr[1] * spatial_dim + d];
        p2[d] = coords[nodes_ptr[2] * spatial_dim + d];
      }
      std::array<real_t,3> e1, e2;
      for (int d = 0; d < 3; ++d) {
        e1[d] = p1[d] - p0[d];
        e2[d] = p2[d] - p0[d];
      }
      normal = cross(e1, e2);
    }
  }

  return normal;
}

// ---- Measures ----

real_t MeshGeometry::cell_measure(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(cell);
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  const CellShape& shape = mesh.cell_shape(cell);
  int spatial_dim = mesh.dim();

  // Helper to get coordinates
  auto get_coords = [&](index_t node_id) -> std::array<real_t,3> {
    std::array<real_t,3> pt = {{0,0,0}};
    for (int d = 0; d < spatial_dim; ++d) {
      pt[d] = coords[node_id * spatial_dim + d];
    }
    return pt;
  };

  switch (shape.family) {
    case CellFamily::Line: {
      if (n_nodes >= 2) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        return distance(p0, p1);
      }
      break;
    }

    case CellFamily::Triangle: {
      if (n_nodes >= 3) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);
        return triangle_area(p0, p1, p2);
      }
      break;
    }

    case CellFamily::Quad: {
      if (n_nodes >= 4) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 4; ++i) {
          verts.push_back(get_coords(nodes_ptr[i]));
        }
        return quad_area(verts);
      }
      break;
    }

    case CellFamily::Tetra: {
      if (n_nodes >= 4) {
        auto p0 = get_coords(nodes_ptr[0]);
        auto p1 = get_coords(nodes_ptr[1]);
        auto p2 = get_coords(nodes_ptr[2]);
        auto p3 = get_coords(nodes_ptr[3]);
        return std::abs(tet_volume(p0, p1, p2, p3));
      }
      break;
    }

    case CellFamily::Hex: {
      if (n_nodes >= 8) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 8; ++i) {
          verts.push_back(get_coords(nodes_ptr[i]));
        }
        return hex_volume(verts);
      }
      break;
    }

    case CellFamily::Wedge: {
      if (n_nodes >= 6) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 6; ++i) {
          verts.push_back(get_coords(nodes_ptr[i]));
        }
        return wedge_volume(verts);
      }
      break;
    }

    case CellFamily::Pyramid: {
      if (n_nodes >= 5) {
        std::vector<std::array<real_t,3>> verts;
        for (size_t i = 0; i < 5; ++i) {
          verts.push_back(get_coords(nodes_ptr[i]));
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
  auto [nodes_ptr, n_nodes] = mesh.face_nodes_span(face);
  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  if (spatial_dim == 2) {
    // In 2D, "faces" are edges, so return edge length
    if (n_nodes >= 2) {
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      for (int d = 0; d < spatial_dim; ++d) {
        p0[d] = coords[nodes_ptr[0] * spatial_dim + d];
        p1[d] = coords[nodes_ptr[1] * spatial_dim + d];
      }
      return distance(p0, p1);
    }
  } else if (spatial_dim == 3) {
    // 3D face area
    if (n_nodes == 3) {
      // Triangle
      std::array<real_t,3> p0 = {{0,0,0}};
      std::array<real_t,3> p1 = {{0,0,0}};
      std::array<real_t,3> p2 = {{0,0,0}};
      for (int d = 0; d < spatial_dim; ++d) {
        p0[d] = coords[nodes_ptr[0] * spatial_dim + d];
        p1[d] = coords[nodes_ptr[1] * spatial_dim + d];
        p2[d] = coords[nodes_ptr[2] * spatial_dim + d];
      }
      return triangle_area(p0, p1, p2);
    } else if (n_nodes == 4) {
      // Quadrilateral
      std::vector<std::array<real_t,3>> verts;
      for (size_t i = 0; i < 4; ++i) {
        std::array<real_t,3> pt = {{0,0,0}};
        for (int d = 0; d < spatial_dim; ++d) {
          pt[d] = coords[nodes_ptr[i] * spatial_dim + d];
        }
        verts.push_back(pt);
      }
      return quad_area(verts);
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
  // Standard hex node ordering: 0-1-2-3 bottom, 4-5-6-7 top
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

} // namespace svmp