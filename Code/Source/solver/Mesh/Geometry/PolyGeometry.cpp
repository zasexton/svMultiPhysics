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
#include <unordered_map>
#include <utility>

namespace svmp {

namespace {
  inline std::array<real_t,3> make_vec(real_t x, real_t y, real_t z) {
    return {x, y, z};
  }

  inline std::array<real_t,3> scale(const std::array<real_t,3>& a, real_t s) {
    return {a[0] * s, a[1] * s, a[2] * s};
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

    inline real_t tet_abs_volume(const std::array<real_t,3>& p0,
                                 const std::array<real_t,3>& p1,
                                 const std::array<real_t,3>& p2,
                                 const std::array<real_t,3>& p3) {
      const auto v1 = sub(p1, p0);
      const auto v2 = sub(p2, p0);
      const auto v3 = sub(p3, p0);
      const real_t det = dot(v1, cross(v2, v3));
      return std::abs(det) / static_cast<real_t>(6.0);
    }

    inline std::array<real_t,3> tet_centroid(const std::array<real_t,3>& p0,
                                             const std::array<real_t,3>& p1,
                                             const std::array<real_t,3>& p2,
                                             const std::array<real_t,3>& p3) {
      return make_vec((p0[0] + p1[0] + p2[0] + p3[0]) / static_cast<real_t>(4.0),
                      (p0[1] + p1[1] + p2[1] + p3[1]) / static_cast<real_t>(4.0),
                      (p0[2] + p1[2] + p2[2] + p3[2]) / static_cast<real_t>(4.0));
    }

	  inline std::array<real_t,3> get_vertex(const MeshBase& mesh, index_t n, Configuration cfg) {
	    const auto& coords = ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
	                             ? mesh.X_cur()
	                             : mesh.X_ref();
	    int dim = mesh.dim();
	    std::array<real_t,3> p = {0.0, 0.0, 0.0};
	    if (dim >= 1) p[0] = coords[static_cast<size_t>(n) * dim + 0];
	    if (dim >= 2) p[1] = coords[static_cast<size_t>(n) * dim + 1];
	    if (dim >= 3) p[2] = coords[static_cast<size_t>(n) * dim + 2];
	    return p;
	  }

    struct EdgeKey {
      index_t a = 0;
      index_t b = 0;
      bool operator==(const EdgeKey& o) const noexcept { return a == o.a && b == o.b; }
    };

    struct EdgeKeyHash {
      size_t operator()(const EdgeKey& k) const noexcept {
        size_t h1 = std::hash<index_t>()(k.a);
        size_t h2 = std::hash<index_t>()(k.b);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
      }
    };

    inline EdgeKey make_edge_key(index_t u, index_t v) {
      if (u < v) return {u, v};
      return {v, u};
    }

    inline std::array<real_t,3> sub3(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
      return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }

    inline std::array<real_t,3> add3(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
      return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    }

    inline std::array<real_t,3> scale3(const std::array<real_t,3>& a, real_t s) {
      return {a[0] * s, a[1] * s, a[2] * s};
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

real_t PolyGeometry::polyhedron_volume(const MeshBase& mesh,
                                       index_t cell,
                                       Configuration cfg) {
  const auto props = polyhedron_mass_properties(mesh, cell, cfg);
  if (props.is_valid) return props.volume;
  // Fallback for legacy callers: convex star decomposition.
  const auto faces = mesh.cell_faces(cell);
  if (faces.empty()) return static_cast<real_t>(0.0);
  auto [cell_verts, n_cell_verts] = mesh.cell_vertices_span(cell);
  if (n_cell_verts == 0) return static_cast<real_t>(0.0);
  std::array<real_t,3> p0{0.0, 0.0, 0.0};
  for (size_t i = 0; i < n_cell_verts; ++i) p0 = add(p0, get_vertex(mesh, cell_verts[i], cfg));
  p0 = scale(p0, static_cast<real_t>(1.0) / static_cast<real_t>(n_cell_verts));
  real_t vol_sum = 0.0;
  for (index_t f : faces) {
    auto fverts = mesh.face_vertices(f);
    if (fverts.size() < 3) continue;
    const auto a0 = get_vertex(mesh, fverts[0], cfg);
    for (size_t i = 1; i + 1 < fverts.size(); ++i) {
      const auto a1 = get_vertex(mesh, fverts[i], cfg);
      const auto a2 = get_vertex(mesh, fverts[i + 1], cfg);
      vol_sum += tet_abs_volume(p0, a0, a1, a2);
    }
  }
  return (vol_sum < GeometryConfig::volume_epsilon()) ? static_cast<real_t>(0.0) : vol_sum;
}

std::array<real_t,3> PolyGeometry::polyhedron_centroid(const MeshBase& mesh,
                                                       index_t cell,
                                                       Configuration cfg) {
  const auto props = polyhedron_mass_properties(mesh, cell, cfg);
  if (props.is_valid) return props.centroid;
  // Fallback for legacy callers: convex star decomposition centroid.
  const auto faces = mesh.cell_faces(cell);
  auto [cell_verts, n_cell_verts] = mesh.cell_vertices_span(cell);
  if (faces.empty() || n_cell_verts == 0) return {0.0, 0.0, 0.0};
  std::array<real_t,3> p0{0.0, 0.0, 0.0};
  for (size_t i = 0; i < n_cell_verts; ++i) p0 = add(p0, get_vertex(mesh, cell_verts[i], cfg));
  p0 = scale(p0, static_cast<real_t>(1.0) / static_cast<real_t>(n_cell_verts));
  real_t vol_sum = 0.0;
  std::array<real_t,3> csum{0.0, 0.0, 0.0};
  for (index_t f : faces) {
    auto fverts = mesh.face_vertices(f);
    if (fverts.size() < 3) continue;
    const auto a0 = get_vertex(mesh, fverts[0], cfg);
    for (size_t i = 1; i + 1 < fverts.size(); ++i) {
      const auto a1 = get_vertex(mesh, fverts[i], cfg);
      const auto a2 = get_vertex(mesh, fverts[i + 1], cfg);
      const real_t v = tet_abs_volume(p0, a0, a1, a2);
      if (v <= GeometryConfig::volume_epsilon()) continue;
      const auto ct = tet_centroid(p0, a0, a1, a2);
      csum[0] += v * ct[0];
      csum[1] += v * ct[1];
      csum[2] += v * ct[2];
      vol_sum += v;
    }
  }
  if (vol_sum <= GeometryConfig::volume_epsilon()) return p0;
  return {csum[0] / vol_sum, csum[1] / vol_sum, csum[2] / vol_sum};
}

PolyGeometry::PolyhedronMassProperties PolyGeometry::polyhedron_mass_properties(const MeshBase& mesh,
                                                                                index_t cell,
                                                                                Configuration cfg) {
  PolyhedronMassProperties out;
  out.centroid = {0.0, 0.0, 0.0};

  const auto faces = mesh.cell_faces(cell);
  if (faces.empty()) return out;

  // Vertex-average origin shift to reduce cancellation and improve robustness.
  auto [cell_verts, n_cell_verts] = mesh.cell_vertices_span(cell);
  if (n_cell_verts == 0) return out;
  std::array<real_t,3> origin{0.0, 0.0, 0.0};
  for (size_t i = 0; i < n_cell_verts; ++i) origin = add(origin, get_vertex(mesh, cell_verts[i], cfg));
  origin = scale(origin, static_cast<real_t>(1.0) / static_cast<real_t>(n_cell_verts));

  // Collect face vertex loops.
  struct FaceData {
    index_t face_id = INVALID_INDEX;
    std::vector<index_t> verts;
  };
  std::vector<FaceData> fdata;
  fdata.reserve(faces.size());
  for (index_t f : faces) {
    auto v = mesh.face_vertices(f);
    if (v.size() < 3) return out;
    fdata.push_back({f, std::move(v)});
  }

  // Infer a consistent per-face flip (sense) if the mesh does not provide one.
  // flip = +1 means use face ordering as stored; flip = -1 means reverse.
  std::vector<int> flip(fdata.size(), 0);
  std::vector<std::vector<std::pair<int,int>>> adj(fdata.size()); // (nbr, parity) parity=+1 same, -1 opposite

  struct EdgeAcc {
    int face = -1;
    int dir = 0; // +1 if traverses a->b for key(a<b), -1 otherwise
    int count = 0;
  };
  std::unordered_map<EdgeKey, EdgeAcc, EdgeKeyHash> edges;
  edges.reserve(128);

  for (int fi = 0; fi < static_cast<int>(fdata.size()); ++fi) {
    const auto& verts = fdata[static_cast<size_t>(fi)].verts;
    const size_t m = verts.size();
    for (size_t i = 0; i < m; ++i) {
      const index_t u = verts[i];
      const index_t v = verts[(i + 1) % m];
      const EdgeKey key = make_edge_key(u, v);
      const int dir = (u == key.a) ? +1 : -1;
      auto& acc = edges[key];
      if (acc.count == 0) {
        acc.face = fi;
        acc.dir = dir;
        acc.count = 1;
      } else if (acc.count == 1) {
        const int fj = acc.face;
        const int dirj = acc.dir;
        const int parity = (dir == -dirj) ? +1 : -1;
        adj[static_cast<size_t>(fi)].push_back({fj, parity});
        adj[static_cast<size_t>(fj)].push_back({fi, parity});
        acc.count = 2;
      } else {
        // Non-manifold edge (>2 incident faces).
        return out;
      }
    }
  }

  for (const auto& kv : edges) {
    if (kv.second.count != 2) {
      // Open surface edge (watertightness violation).
      return out;
    }
  }

  // Solve for flips via BFS.
  for (int start = 0; start < static_cast<int>(fdata.size()); ++start) {
    if (flip[static_cast<size_t>(start)] != 0) continue;
    flip[static_cast<size_t>(start)] = +1;
    std::vector<int> stack = {start};
    while (!stack.empty()) {
      const int cur = stack.back();
      stack.pop_back();
      const int cur_flip = flip[static_cast<size_t>(cur)];
      for (const auto& [nbr, parity] : adj[static_cast<size_t>(cur)]) {
        const int want = (parity == +1) ? cur_flip : -cur_flip;
        int& nbr_flip = flip[static_cast<size_t>(nbr)];
        if (nbr_flip == 0) {
          nbr_flip = want;
          stack.push_back(nbr);
        } else if (nbr_flip != want) {
          // Inconsistent orientation constraints.
          return out;
        }
      }
    }
  }

  // Compute signed volume and centroid in translated coordinates.
  real_t vol6 = 0.0;
  std::array<real_t,3> cnum{0.0, 0.0, 0.0};

  for (size_t fi = 0; fi < fdata.size(); ++fi) {
    auto verts = fdata[fi].verts;
    if (flip[fi] < 0) std::reverse(verts.begin(), verts.end());

    const auto p0 = sub3(get_vertex(mesh, verts[0], cfg), origin);
    for (size_t i = 1; i + 1 < verts.size(); ++i) {
      const auto p1 = sub3(get_vertex(mesh, verts[i], cfg), origin);
      const auto p2 = sub3(get_vertex(mesh, verts[i + 1], cfg), origin);
      const real_t d = dot(p0, cross(p1, p2));
      vol6 += d;
      const auto sum = add3(add3(p0, p1), p2);
      cnum[0] += d * sum[0];
      cnum[1] += d * sum[1];
      cnum[2] += d * sum[2];
    }
  }

  if (std::abs(vol6) <= GeometryConfig::volume_epsilon() * 6.0) {
    return out;
  }

  // Ensure positive volume by optionally flipping all faces; centroid is invariant under global flip.
  const real_t volume = vol6 / 6.0;
  out.volume = std::abs(volume);
  const std::array<real_t,3> cent_rel = scale3(cnum, static_cast<real_t>(1.0) / (static_cast<real_t>(4.0) * vol6));
  out.centroid = add3(origin, cent_rel);
  out.is_valid = true;
  return out;
}

} // namespace svmp
