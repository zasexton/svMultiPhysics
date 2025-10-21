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

#include "SearchPrimitives.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace svmp {
namespace search {

// ---- AABB implementation ----

void AABB::include(const std::array<real_t,3>& p) {
  for (int i = 0; i < 3; ++i) {
    min[i] = std::min(min[i], p[i]);
    max[i] = std::max(max[i], p[i]);
  }
}

void AABB::include(const AABB& other) {
  for (int i = 0; i < 3; ++i) {
    min[i] = std::min(min[i], other.min[i]);
    max[i] = std::max(max[i], other.max[i]);
  }
}

bool AABB::overlaps(const AABB& other) const {
  for (int i = 0; i < 3; ++i) {
    if (max[i] < other.min[i] || min[i] > other.max[i]) {
      return false;
    }
  }
  return true;
}

bool AABB::contains(const std::array<real_t,3>& p) const {
  for (int i = 0; i < 3; ++i) {
    if (p[i] < min[i] || p[i] > max[i]) {
      return false;
    }
  }
  return true;
}

// Closest point on AABB to a given point (clamped)
static inline std::array<real_t,3> clamp3(const std::array<real_t,3>& x,
                                          const std::array<real_t,3>& lo,
                                          const std::array<real_t,3>& hi) {
  return {
    std::min(hi[0], std::max(lo[0], x[0])),
    std::min(hi[1], std::max(lo[1], x[1])),
    std::min(hi[2], std::max(lo[2], x[2]))
  };
}

std::array<real_t,3> AABB::center() const {
  return {
    0.5 * (min[0] + max[0]),
    0.5 * (min[1] + max[1]),
    0.5 * (min[2] + max[2])
  };
}

// Provide closest_point convenience expected by some accelerators
std::array<real_t,3> AABB::closest_point(const std::array<real_t,3>& p) const {
  return clamp3(p, min, max);
}

std::array<real_t,3> AABB::extents() const {
  return {
    max[0] - min[0],
    max[1] - min[1],
    max[2] - min[2]
  };
}

real_t AABB::volume() const {
  auto ext = extents();
  return ext[0] * ext[1] * ext[2];
}

real_t AABB::surface_area() const {
  auto ext = extents();
  return 2.0 * (ext[0]*ext[1] + ext[1]*ext[2] + ext[2]*ext[0]);
}

// ---- Ray implementation ----

std::array<real_t,3> Ray::point_at(real_t t) const {
  return add3(origin, scale3(direction, t));
}

// ---- Point-in-shape tests ----

bool point_in_triangle(const std::array<real_t,3>& p,
                       const std::array<real_t,3>& a,
                       const std::array<real_t,3>& b,
                       const std::array<real_t,3>& c,
                       real_t tol) {
  // Compute barycentric coordinates
  auto bary = triangle_barycentric(p, a, b, c);

  // Check if all coordinates are non-negative and sum to ~1
  return bary[0] >= -tol && bary[1] >= -tol && bary[2] >= -tol &&
         std::abs(bary[0] + bary[1] + bary[2] - 1.0) < tol;
}

bool point_in_tetrahedron(const std::array<real_t,3>& p,
                         const std::array<real_t,3>& a,
                         const std::array<real_t,3>& b,
                         const std::array<real_t,3>& c,
                         const std::array<real_t,3>& d,
                         real_t tol) {
  // Compute barycentric coordinates
  auto bary = tetrahedron_barycentric(p, a, b, c, d);

  // Check if all coordinates are non-negative and sum to ~1
  return bary[0] >= -tol && bary[1] >= -tol &&
         bary[2] >= -tol && bary[3] >= -tol &&
         std::abs(bary[0] + bary[1] + bary[2] + bary[3] - 1.0) < tol;
}

bool point_in_hexahedron(const std::array<real_t,3>& p,
                        const std::vector<std::array<real_t,3>>& vertices,
                        real_t tol) {
  // Hexahedron vertices numbered as:
  // Bottom: 0-1-2-3, Top: 4-5-6-7
  // Decompose into 5 tetrahedra
  if (vertices.size() != 8) return false;

  // Central decomposition
  return point_in_tetrahedron(p, vertices[0], vertices[1], vertices[3], vertices[4], tol) ||
         point_in_tetrahedron(p, vertices[1], vertices[2], vertices[3], vertices[6], tol) ||
         point_in_tetrahedron(p, vertices[1], vertices[3], vertices[4], vertices[6], tol) ||
         point_in_tetrahedron(p, vertices[1], vertices[4], vertices[5], vertices[6], tol) ||
         point_in_tetrahedron(p, vertices[3], vertices[4], vertices[6], vertices[7], tol);
}

bool point_in_wedge(const std::array<real_t,3>& p,
                   const std::vector<std::array<real_t,3>>& vertices,
                   real_t tol) {
  // Wedge vertices: bottom triangle 0-1-2, top triangle 3-4-5
  // Decompose into 3 tetrahedra
  if (vertices.size() != 6) return false;

  return point_in_tetrahedron(p, vertices[0], vertices[1], vertices[2], vertices[3], tol) ||
         point_in_tetrahedron(p, vertices[1], vertices[2], vertices[3], vertices[4], tol) ||
         point_in_tetrahedron(p, vertices[2], vertices[3], vertices[4], vertices[5], tol);
}

bool point_in_pyramid(const std::array<real_t,3>& p,
                     const std::vector<std::array<real_t,3>>& vertices,
                     real_t tol) {
  // Pyramid vertices: quad base 0-1-2-3, apex 4
  // Decompose into 2 tetrahedra
  if (vertices.size() != 5) return false;

  return point_in_tetrahedron(p, vertices[0], vertices[1], vertices[2], vertices[4], tol) ||
         point_in_tetrahedron(p, vertices[0], vertices[2], vertices[3], vertices[4], tol);
}

bool point_in_cell(const std::array<real_t,3>& p,
                  const CellShape& shape,
                  const std::vector<std::array<real_t,3>>& vertices,
                  real_t tol) {
  switch (shape.family) {
    case CellFamily::Tetra:
      if (vertices.size() >= 4) {
        return point_in_tetrahedron(p, vertices[0], vertices[1],
                                   vertices[2], vertices[3], tol);
      }
      break;

    case CellFamily::Hex:
      return point_in_hexahedron(p, vertices, tol);

    case CellFamily::Wedge:
      return point_in_wedge(p, vertices, tol);

    case CellFamily::Pyramid:
      return point_in_pyramid(p, vertices, tol);

    default:
      // For other shapes, fallback to linear search or specific implementations
      break;
  }

  return false;
}

// ---- Ray intersection tests ----

bool ray_triangle_intersect(const Ray& ray,
                           const std::array<real_t,3>& a,
                           const std::array<real_t,3>& b,
                           const std::array<real_t,3>& c,
                           real_t& t,
                           std::array<real_t,2>* uv) {
  // Möller-Trumbore algorithm
  const real_t EPSILON = 1e-10;

  auto edge1 = sub3(b, a);
  auto edge2 = sub3(c, a);
  auto h = cross3(ray.direction, edge2);
  real_t det = dot3(edge1, h);

  if (std::abs(det) < EPSILON) {
    return false;  // Ray parallel to triangle
  }

  real_t inv_det = 1.0 / det;
  auto s = sub3(ray.origin, a);
  real_t u = inv_det * dot3(s, h);

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  auto q = cross3(s, edge1);
  real_t v = inv_det * dot3(ray.direction, q);

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  t = inv_det * dot3(edge2, q);

  if (t < ray.t_min || t > ray.t_max) {
    return false;
  }

  if (uv) {
    (*uv)[0] = u;
    (*uv)[1] = v;
  }

  return true;
}

bool ray_aabb_intersect(const Ray& ray,
                       const AABB& aabb,
                       real_t& t_near,
                       real_t& t_far) {
  t_near = ray.t_min;
  t_far = ray.t_max;

  for (int i = 0; i < 3; ++i) {
    if (std::abs(ray.direction[i]) < 1e-10) {
      // Ray parallel to slab
      if (ray.origin[i] < aabb.min[i] || ray.origin[i] > aabb.max[i]) {
        return false;
      }
    } else {
      real_t inv_d = 1.0 / ray.direction[i];
      real_t t1 = (aabb.min[i] - ray.origin[i]) * inv_d;
      real_t t2 = (aabb.max[i] - ray.origin[i]) * inv_d;

      if (t1 > t2) std::swap(t1, t2);

      t_near = std::max(t_near, t1);
      t_far = std::min(t_far, t2);

      if (t_near > t_far) {
        return false;
      }
    }
  }

  return true;
}

bool ray_sphere_intersect(const Ray& ray,
                         const std::array<real_t,3>& center,
                         real_t radius,
                         real_t& t1,
                         real_t& t2) {
  auto oc = sub3(ray.origin, center);
  real_t a = dot3(ray.direction, ray.direction);
  real_t b = 2.0 * dot3(oc, ray.direction);
  real_t c = dot3(oc, oc) - radius * radius;

  real_t discriminant = b * b - 4 * a * c;
  if (discriminant < 0) {
    return false;
  }

  real_t sqrt_disc = std::sqrt(discriminant);
  t1 = (-b - sqrt_disc) / (2.0 * a);
  t2 = (-b + sqrt_disc) / (2.0 * a);

  // Check if intersections are within ray bounds
  bool valid = false;
  if (t1 >= ray.t_min && t1 <= ray.t_max) valid = true;
  if (t2 >= ray.t_min && t2 <= ray.t_max) valid = true;

  return valid;
}

// ---- Distance computations ----

real_t point_segment_distance(const std::array<real_t,3>& p,
                             const std::array<real_t,3>& a,
                             const std::array<real_t,3>& b,
                             std::array<real_t,3>* closest) {
  auto ab = sub3(b, a);
  auto ap = sub3(p, a);

  real_t t = dot3(ap, ab) / dot3(ab, ab);
  t = std::max(0.0, std::min(1.0, t));

  auto closest_point = add3(a, scale3(ab, t));
  if (closest) {
    *closest = closest_point;
  }

  auto diff = sub3(p, closest_point);
  return std::sqrt(dot3(diff, diff));
}

real_t point_triangle_distance(const std::array<real_t,3>& p,
                              const std::array<real_t,3>& a,
                              const std::array<real_t,3>& b,
                              const std::array<real_t,3>& c,
                              std::array<real_t,3>* closest) {
  // Project point onto triangle plane
  auto normal = normalize3(triangle_normal(a, b, c));
  auto ap = sub3(p, a);
  real_t dist_to_plane = dot3(ap, normal);
  auto proj_p = sub3(p, scale3(normal, dist_to_plane));

  // Check if projection is inside triangle
  if (point_in_triangle(proj_p, a, b, c, 1e-10)) {
    if (closest) {
      *closest = proj_p;
    }
    return std::abs(dist_to_plane);
  }

  // Find closest point on triangle edges
  std::array<real_t,3> closest_edge;
  real_t min_dist = point_segment_distance(p, a, b, &closest_edge);

  std::array<real_t,3> temp_closest;
  real_t dist = point_segment_distance(p, b, c, &temp_closest);
  if (dist < min_dist) {
    min_dist = dist;
    closest_edge = temp_closest;
  }

  dist = point_segment_distance(p, c, a, &temp_closest);
  if (dist < min_dist) {
    min_dist = dist;
    closest_edge = temp_closest;
  }

  if (closest) {
    *closest = closest_edge;
  }

  return min_dist;
}

real_t point_aabb_distance(const std::array<real_t,3>& p,
                          const AABB& aabb) {
  real_t dist_sq = 0;

  for (int i = 0; i < 3; ++i) {
    if (p[i] < aabb.min[i]) {
      real_t d = aabb.min[i] - p[i];
      dist_sq += d * d;
    } else if (p[i] > aabb.max[i]) {
      real_t d = p[i] - aabb.max[i];
      dist_sq += d * d;
    }
  }

  return std::sqrt(dist_sq);
}

// ---- Barycentric coordinates ----

std::array<real_t,3> triangle_barycentric(const std::array<real_t,3>& p,
                                         const std::array<real_t,3>& a,
                                         const std::array<real_t,3>& b,
                                         const std::array<real_t,3>& c) {
  auto v0 = sub3(c, a);
  auto v1 = sub3(b, a);
  auto v2 = sub3(p, a);

  real_t d00 = dot3(v0, v0);
  real_t d01 = dot3(v0, v1);
  real_t d11 = dot3(v1, v1);
  real_t d20 = dot3(v2, v0);
  real_t d21 = dot3(v2, v1);

  real_t denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) < 1e-30) {
    return {1.0/3.0, 1.0/3.0, 1.0/3.0};  // Degenerate triangle
  }

  real_t v = (d11 * d20 - d01 * d21) / denom;
  real_t w = (d00 * d21 - d01 * d20) / denom;
  real_t u = 1.0 - v - w;

  return {u, w, v};
}

std::array<real_t,4> tetrahedron_barycentric(const std::array<real_t,3>& p,
                                            const std::array<real_t,3>& a,
                                            const std::array<real_t,3>& b,
                                            const std::array<real_t,3>& c,
                                            const std::array<real_t,3>& d) {
  // Compute volume of sub-tetrahedra
  real_t vol_total = tetrahedron_volume(a, b, c, d);

  if (std::abs(vol_total) < 1e-30) {
    return {0.25, 0.25, 0.25, 0.25};  // Degenerate tetrahedron
  }

  real_t vol_pbcd = tetrahedron_volume(p, b, c, d);
  real_t vol_apcd = tetrahedron_volume(a, p, c, d);
  real_t vol_abpd = tetrahedron_volume(a, b, p, d);
  real_t vol_abcp = tetrahedron_volume(a, b, c, p);

  return {
    vol_pbcd / vol_total,
    vol_apcd / vol_total,
    vol_abpd / vol_total,
    vol_abcp / vol_total
  };
}

// ---- Geometric utilities ----

real_t triangle_area(const std::array<real_t,3>& a,
                    const std::array<real_t,3>& b,
                    const std::array<real_t,3>& c) {
  auto normal = triangle_normal(a, b, c);
  return 0.5 * std::sqrt(dot3(normal, normal));
}

std::array<real_t,3> triangle_normal(const std::array<real_t,3>& a,
                                    const std::array<real_t,3>& b,
                                    const std::array<real_t,3>& c) {
  auto ab = sub3(b, a);
  auto ac = sub3(c, a);
  return cross3(ab, ac);
}

real_t tetrahedron_volume(const std::array<real_t,3>& a,
                         const std::array<real_t,3>& b,
                         const std::array<real_t,3>& c,
                         const std::array<real_t,3>& d) {
  auto ab = sub3(b, a);
  auto ac = sub3(c, a);
  auto ad = sub3(d, a);

  // Volume = 1/6 * |det(ab, ac, ad)|
  return dot3(ab, cross3(ac, ad)) / 6.0;
}

bool aabb_sphere_overlap(const AABB& aabb,
                        const std::array<real_t,3>& center,
                        real_t radius) {
  real_t dist_sq = 0;

  for (int i = 0; i < 3; ++i) {
    if (center[i] < aabb.min[i]) {
      real_t d = aabb.min[i] - center[i];
      dist_sq += d * d;
    } else if (center[i] > aabb.max[i]) {
      real_t d = center[i] - aabb.max[i];
      dist_sq += d * d;
    }
  }

  return dist_sq <= radius * radius;
}

bool segments_intersect_2d(const std::array<real_t,2>& a1,
                          const std::array<real_t,2>& a2,
                          const std::array<real_t,2>& b1,
                          const std::array<real_t,2>& b2,
                          real_t* t) {
  // Check if segments intersect using cross products
  auto cross2d = [](const std::array<real_t,2>& v1, const std::array<real_t,2>& v2) {
    return v1[0] * v2[1] - v1[1] * v2[0];
  };

  std::array<real_t,2> da = {a2[0] - a1[0], a2[1] - a1[1]};
  std::array<real_t,2> db = {b2[0] - b1[0], b2[1] - b1[1]};
  std::array<real_t,2> dc = {b1[0] - a1[0], b1[1] - a1[1]};

  real_t cross_da_db = cross2d(da, db);

  if (std::abs(cross_da_db) < 1e-10) {
    return false;  // Parallel segments
  }

  real_t t1 = cross2d(dc, db) / cross_da_db;
  real_t t2 = cross2d(dc, da) / cross_da_db;

  if (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1) {
    if (t) {
      *t = t1;
    }
    return true;
  }

  return false;
}

} // namespace search
} // namespace svmp
