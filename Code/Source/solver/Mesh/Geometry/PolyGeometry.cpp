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

#include "PolyGeometry.h"
#include "GeometryConfig.h"

#include <cmath>
#include <algorithm>

namespace svmp {

namespace {
  inline std::array<real_t,3> make_vec(real_t x, real_t y, real_t z) {
    return {x, y, z};
  }

  inline std::array<real_t,3> add(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
    return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
  }

  inline std::array<real_t,3> sub(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
  }

  inline std::array<real_t,3> cross(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
    return {
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]
    };
  }

  inline real_t dot(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  }

  inline real_t norm(const std::array<real_t,3>& a) {
    return std::sqrt(dot(a, a));
  }

  inline std::array<real_t,3> get_vertex(const MeshBase& mesh, index_t n, Configuration cfg) {
    const auto& coords = (cfg == Configuration::Current && mesh.has_current_coords()) ? mesh.X_cur() : mesh.X_ref();
    int dim = mesh.dim();
    std::array<real_t,3> p = {0.0, 0.0, 0.0};
    if (dim >= 1) p[0] = coords[static_cast<size_t>(n) * dim + 0];
    if (dim >= 2) p[1] = coords[static_cast<size_t>(n) * dim + 1];
    if (dim >= 3) p[2] = coords[static_cast<size_t>(n) * dim + 2];
    return p;
  }
}

// -------------------
// 3D coordinate API
// -------------------

std::array<real_t,3> PolyGeometry::newell_normal(const std::vector<std::array<real_t,3>>& verts) {
  std::array<real_t,3> n = {0.0, 0.0, 0.0};
  const size_t m = verts.size();
  if (m < 3) return n;

  for (size_t i = 0, j = m - 1; i < m; j = i++) {
    const auto& vi = verts[i];
    const auto& vj = verts[j];
    n[0] += (vj[1] - vi[1]) * (vj[2] + vi[2]);
    n[1] += (vj[2] - vi[2]) * (vj[0] + vi[0]);
    n[2] += (vj[0] - vi[0]) * (vj[1] + vi[1]);
  }
  return n;
}

real_t PolyGeometry::polygon_area(const std::vector<std::array<real_t,3>>& verts) {
  auto n = newell_normal(verts);
  const real_t a = static_cast<real_t>(0.5) * norm(n);
  return (a < GeometryConfig::area_epsilon()) ? static_cast<real_t>(0) : a;
}

std::array<real_t,3> PolyGeometry::polygon_centroid(const std::vector<std::array<real_t,3>>& verts) {
  std::array<real_t,3> c = {0.0, 0.0, 0.0};
  const size_t m = verts.size();
  if (m == 0) return c;
  if (m == 1) return verts[0];
  if (m == 2) return make_vec((verts[0][0]+verts[1][0])/2.0, (verts[0][1]+verts[1][1])/2.0, (verts[0][2]+verts[1][2])/2.0);

  // Triangle fan around verts[0]
  const auto& v0 = verts[0];
  real_t area_sum = 0.0;
  std::array<real_t,3> weighted_sum = {0.0, 0.0, 0.0};

  for (size_t i = 1; i + 1 < m; ++i) {
    auto u = sub(verts[i], v0);
    auto v = sub(verts[i+1], v0);
    auto cr = cross(u, v);
    real_t tri_area = static_cast<real_t>(0.5) * norm(cr);
    if (tri_area <= GeometryConfig::area_epsilon()) continue;

    auto tri_c = make_vec((v0[0] + verts[i][0] + verts[i+1][0]) / 3.0,
                          (v0[1] + verts[i][1] + verts[i+1][1]) / 3.0,
                          (v0[2] + verts[i][2] + verts[i+1][2]) / 3.0);
    weighted_sum = add(weighted_sum, make_vec(tri_c[0] * tri_area, tri_c[1] * tri_area, tri_c[2] * tri_area));
    area_sum += tri_area;
  }

  if (area_sum <= GeometryConfig::area_epsilon()) {
    // Fallback: average of vertices
    for (const auto& p : verts) {
      c[0] += p[0]; c[1] += p[1]; c[2] += p[2];
    }
    c[0] /= static_cast<real_t>(m);
    c[1] /= static_cast<real_t>(m);
    c[2] /= static_cast<real_t>(m);
    return c;
  }

  c[0] = weighted_sum[0] / area_sum;
  c[1] = weighted_sum[1] / area_sum;
  c[2] = weighted_sum[2] / area_sum;
  return c;
}

// -------------------
// Mesh-based API
// -------------------

std::array<real_t,3> PolyGeometry::polygon_normal(const MeshBase& mesh,
                                                  const std::vector<index_t>& vertices,
                                                  Configuration cfg) {
  std::vector<std::array<real_t,3>> pts;
  pts.reserve(vertices.size());
  for (auto vid : vertices) pts.push_back(get_vertex(mesh, vid, cfg));
  return newell_normal(pts);
}

real_t PolyGeometry::polygon_area(const MeshBase& mesh,
                                  const std::vector<index_t>& vertices,
                                  Configuration cfg) {
  auto n = polygon_normal(mesh, vertices, cfg);
  const real_t a = static_cast<real_t>(0.5) * norm(n);
  return (a < GeometryConfig::area_epsilon()) ? static_cast<real_t>(0) : a;
}

std::array<real_t,3> PolyGeometry::polygon_centroid(const MeshBase& mesh,
                                                    const std::vector<index_t>& vertices,
                                                    Configuration cfg) {
  std::vector<std::array<real_t,3>> pts;
  pts.reserve(vertices.size());
  for (auto vid : vertices) pts.push_back(get_vertex(mesh, vid, cfg));
  return polygon_centroid(pts);
}

} // namespace svmp

