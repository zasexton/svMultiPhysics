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

    inline int dominant_axis(const std::array<real_t,3>& n) {
      const real_t ax = std::abs(n[0]);
      const real_t ay = std::abs(n[1]);
      const real_t az = std::abs(n[2]);
      if (ax >= ay && ax >= az) return 0;
      if (ay >= az) return 1;
      return 2;
    }

    inline std::array<real_t,2> project2(const std::array<real_t,3>& p, int drop_axis) {
      switch (drop_axis) {
        case 0: return {p[1], p[2]}; // yz
        case 1: return {p[0], p[2]}; // xz
        default: return {p[0], p[1]}; // xy
      }
    }

    inline real_t orient2d(const std::array<real_t,2>& a,
                           const std::array<real_t,2>& b,
                           const std::array<real_t,2>& c) {
      return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }

    inline bool on_segment_2d(const std::array<real_t,2>& a,
                              const std::array<real_t,2>& b,
                              const std::array<real_t,2>& p,
                              real_t tol) {
      return (std::min(a[0], b[0]) - tol <= p[0] && p[0] <= std::max(a[0], b[0]) + tol) &&
             (std::min(a[1], b[1]) - tol <= p[1] && p[1] <= std::max(a[1], b[1]) + tol);
    }

    inline bool segments_intersect_2d(const std::array<real_t,2>& p1,
                                      const std::array<real_t,2>& q1,
                                      const std::array<real_t,2>& p2,
                                      const std::array<real_t,2>& q2,
                                      real_t tol) {
      const real_t o1 = orient2d(p1, q1, p2);
      const real_t o2 = orient2d(p1, q1, q2);
      const real_t o3 = orient2d(p2, q2, p1);
      const real_t o4 = orient2d(p2, q2, q1);

      const auto sgn = [&](real_t v) -> int {
        if (v > tol) return 1;
        if (v < -tol) return -1;
        return 0;
      };

      const int s1o1 = sgn(o1);
      const int s1o2 = sgn(o2);
      const int s2o3 = sgn(o3);
      const int s2o4 = sgn(o4);

      if (s1o1 * s1o2 < 0 && s2o3 * s2o4 < 0) return true;

      if (s1o1 == 0 && on_segment_2d(p1, q1, p2, tol)) return true;
      if (s1o2 == 0 && on_segment_2d(p1, q1, q2, tol)) return true;
      if (s2o3 == 0 && on_segment_2d(p2, q2, p1, tol)) return true;
      if (s2o4 == 0 && on_segment_2d(p2, q2, q1, tol)) return true;

      return false;
    }

    inline bool is_simple_polygon_2d(const std::vector<std::array<real_t,2>>& pts,
                                     const std::vector<index_t>& idx,
                                     real_t tol) {
      const size_t n = idx.size();
      if (n < 3) return false;
      for (size_t i = 0; i < n; ++i) {
        const size_t i2 = (i + 1) % n;
        const auto& a0 = pts[static_cast<size_t>(idx[i])];
        const auto& a1 = pts[static_cast<size_t>(idx[i2])];
        for (size_t j = i + 1; j < n; ++j) {
          const size_t j2 = (j + 1) % n;
          if (i == j || i2 == j || i == j2 || i2 == j2) continue; // share endpoint
          const auto& b0 = pts[static_cast<size_t>(idx[j])];
          const auto& b1 = pts[static_cast<size_t>(idx[j2])];
          if (segments_intersect_2d(a0, a1, b0, b1, tol)) return false;
        }
      }
      return true;
    }

    inline bool point_in_triangle_2d(const std::array<real_t,2>& p,
                                     const std::array<real_t,2>& a,
                                     const std::array<real_t,2>& b,
                                     const std::array<real_t,2>& c,
                                     real_t tol) {
      const real_t o = orient2d(a, b, c);
      if (std::abs(o) <= tol) return false;
      const int s = (o > 0) ? 1 : -1;
      const real_t o1 = orient2d(a, b, p);
      const real_t o2 = orient2d(b, c, p);
      const real_t o3 = orient2d(c, a, p);
      if (s > 0) {
        return (o1 >= -tol) && (o2 >= -tol) && (o3 >= -tol);
      }
      return (o1 <= tol) && (o2 <= tol) && (o3 <= tol);
    }

    inline bool ear_clip_triangulate_2d(const std::vector<std::array<real_t,2>>& pts,
                                        std::vector<index_t> idx,
                                        std::vector<std::array<index_t,3>>& triangles,
                                        real_t tol) {
      triangles.clear();
      if (idx.size() < 3) return false;

      // Signed area (twice).
      real_t area2 = 0.0;
      for (size_t i = 0; i < idx.size(); ++i) {
        const size_t j = (i + 1) % idx.size();
        const auto& a = pts[static_cast<size_t>(idx[i])];
        const auto& b = pts[static_cast<size_t>(idx[j])];
        area2 += (a[0] * b[1] - b[0] * a[1]);
      }
      if (std::abs(area2) <= tol) return false;
      const int poly_sign = (area2 > 0) ? 1 : -1;

      const size_t n0 = idx.size();
      triangles.reserve(n0 - 2);

      size_t guard = 0;
      while (idx.size() > 3 && guard++ < 10000) {
        bool found = false;
        const size_t n = idx.size();
        for (size_t pos = 0; pos < n; ++pos) {
          const size_t ip = (pos + n - 1) % n;
          const size_t in = (pos + 1) % n;
          const index_t a = idx[ip];
          const index_t b = idx[pos];
          const index_t c = idx[in];

          const real_t o = orient2d(pts[static_cast<size_t>(a)],
                                    pts[static_cast<size_t>(b)],
                                    pts[static_cast<size_t>(c)]);
          if (!(poly_sign * o > tol)) continue; // reflex or nearly collinear

          bool contains = false;
          for (size_t k = 0; k < n; ++k) {
            const index_t p = idx[k];
            if (p == a || p == b || p == c) continue;
            if (point_in_triangle_2d(pts[static_cast<size_t>(p)],
                                     pts[static_cast<size_t>(a)],
                                     pts[static_cast<size_t>(b)],
                                     pts[static_cast<size_t>(c)], tol)) {
              contains = true;
              break;
            }
          }
          if (contains) continue;

          triangles.push_back({a, b, c});
          idx.erase(idx.begin() + static_cast<std::vector<index_t>::difference_type>(pos));
          found = true;
          break;
        }

        if (!found) {
          // Try to remove a nearly collinear vertex to make progress.
          bool removed = false;
          const size_t n = idx.size();
          for (size_t pos = 0; pos < n; ++pos) {
            const size_t ip = (pos + n - 1) % n;
            const size_t in = (pos + 1) % n;
            const index_t a = idx[ip];
            const index_t b = idx[pos];
            const index_t c = idx[in];
            const real_t o = orient2d(pts[static_cast<size_t>(a)],
                                      pts[static_cast<size_t>(b)],
                                      pts[static_cast<size_t>(c)]);
            if (std::abs(o) <= tol) {
              idx.erase(idx.begin() + static_cast<std::vector<index_t>::difference_type>(pos));
              removed = true;
              break;
            }
          }
          if (!removed) return false;
        }
      }

      if (idx.size() == 3) triangles.push_back({idx[0], idx[1], idx[2]});
      return triangles.size() >= 1;
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

  const auto n = newell_normal(verts);
  const int drop = dominant_axis(n);

  std::vector<std::array<real_t,2>> p2;
  p2.reserve(m);
  for (const auto& p : verts) p2.push_back(project2(p, drop));

  // Projected (signed) area and centroid in 2D.
  real_t area2 = 0.0;
  real_t cx = 0.0;
  real_t cy = 0.0;
  for (size_t i = 0; i < m; ++i) {
    const size_t j = (i + 1) % m;
    const auto& a = p2[i];
    const auto& b = p2[j];
    const real_t cr = (a[0] * b[1] - b[0] * a[1]);
    area2 += cr;
    cx += (a[0] + b[0]) * cr;
    cy += (a[1] + b[1]) * cr;
  }

  if (std::abs(area2) <= GeometryConfig::area_epsilon()) {
    // Fallback: average of vertices
    for (const auto& p : verts) {
      c[0] += p[0]; c[1] += p[1]; c[2] += p[2];
    }
    c[0] /= static_cast<real_t>(m);
    c[1] /= static_cast<real_t>(m);
    c[2] /= static_cast<real_t>(m);
    return c;
  }

  const real_t inv = static_cast<real_t>(1.0) / (static_cast<real_t>(3.0) * area2);
  const real_t u = cx * inv;
  const real_t v = cy * inv;

  const auto& p0 = verts[0];
  if (drop == 0) {
    c[1] = u;
    c[2] = v;
    if (std::abs(n[0]) <= GeometryConfig::normal_epsilon()) return p0;
    c[0] = p0[0] - (n[1] * (c[1] - p0[1]) + n[2] * (c[2] - p0[2])) / n[0];
  } else if (drop == 1) {
    c[0] = u;
    c[2] = v;
    if (std::abs(n[1]) <= GeometryConfig::normal_epsilon()) return p0;
    c[1] = p0[1] - (n[0] * (c[0] - p0[0]) + n[2] * (c[2] - p0[2])) / n[1];
  } else {
    c[0] = u;
    c[1] = v;
    if (std::abs(n[2]) <= GeometryConfig::normal_epsilon()) return p0;
    c[2] = p0[2] - (n[0] * (c[0] - p0[0]) + n[1] * (c[1] - p0[1])) / n[2];
  }

  return c;
}

bool PolyGeometry::triangulate_planar_polygon(const std::vector<std::array<real_t,3>>& verts,
                                              std::vector<std::array<index_t,3>>& triangles) {
  triangles.clear();
  const size_t n = verts.size();
  if (n < 3) return false;
  if (n == 3) {
    triangles.push_back({0, 1, 2});
    return true;
  }

  const auto nrm = newell_normal(verts);
  const real_t nmag = norm(nrm);
  if (!(nmag > GeometryConfig::normal_epsilon())) return false;

  const int drop = dominant_axis(nrm);
  std::vector<std::array<real_t,2>> p2;
  p2.reserve(n);
  for (const auto& p : verts) p2.push_back(project2(p, drop));

  // Index list with a light duplicate cleanup (consecutive duplicates).
  const real_t tol = GeometryConfig::length_epsilon();
  std::vector<index_t> idx;
  idx.reserve(n);

  auto dist2 = [&](const std::array<real_t,2>& a, const std::array<real_t,2>& b) -> real_t {
    const real_t dx = a[0] - b[0];
    const real_t dy = a[1] - b[1];
    return dx * dx + dy * dy;
  };

  const real_t tol2 = tol * tol;
  for (index_t i = 0; i < static_cast<index_t>(n); ++i) {
    if (!idx.empty()) {
      if (dist2(p2[static_cast<size_t>(i)], p2[static_cast<size_t>(idx.back())]) <= tol2) continue;
    }
    idx.push_back(i);
  }
  if (idx.size() >= 2 &&
      dist2(p2[static_cast<size_t>(idx.front())], p2[static_cast<size_t>(idx.back())]) <= tol2) {
    idx.pop_back();
  }

  if (idx.size() < 3) return false;
  if (idx.size() == 3) {
    triangles.push_back({idx[0], idx[1], idx[2]});
    return true;
  }

  if (!is_simple_polygon_2d(p2, idx, tol)) return false;

  return ear_clip_triangulate_2d(p2, std::move(idx), triangles, GeometryConfig::area_epsilon());
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

bool PolyGeometry::triangulate_planar_polygon(const MeshBase& mesh,
                                              const std::vector<index_t>& vertices,
                                              std::vector<std::array<index_t,3>>& triangles,
                                              Configuration cfg) {
  std::vector<std::array<real_t,3>> pts;
  pts.reserve(vertices.size());
  for (auto vid : vertices) pts.push_back(get_vertex(mesh, vid, cfg));
  return triangulate_planar_polygon(pts, triangles);
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

    std::vector<std::array<real_t,3>> face_pts;
    face_pts.reserve(verts.size());
    for (index_t v : verts) face_pts.push_back(sub3(get_vertex(mesh, v, cfg), origin));

    std::vector<std::array<index_t,3>> tris;
    if (!triangulate_planar_polygon(face_pts, tris)) return out;

    for (const auto& t : tris) {
      const auto& p0 = face_pts[static_cast<size_t>(t[0])];
      const auto& p1 = face_pts[static_cast<size_t>(t[1])];
      const auto& p2 = face_pts[static_cast<size_t>(t[2])];
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
