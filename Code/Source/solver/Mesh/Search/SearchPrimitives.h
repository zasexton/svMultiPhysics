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

#ifndef SVMP_SEARCH_PRIMITIVES_H
#define SVMP_SEARCH_PRIMITIVES_H

#include "../Core/MeshTypes.h"
#include "../Topology/CellShape.h"
#include <array>
#include <vector>
#include <utility>
#include <cmath>

namespace svmp {
namespace search {

/**
 * @brief Axis-aligned bounding box
 */
struct AABB {
  std::array<real_t,3> min = {1e30, 1e30, 1e30};
  std::array<real_t,3> max = {-1e30, -1e30, -1e30};

  AABB() = default;
  AABB(const std::array<real_t,3>& p) : min(p), max(p) {}
  AABB(const std::array<real_t,3>& min_, const std::array<real_t,3>& max_)
      : min(min_), max(max_) {}

  void include(const std::array<real_t,3>& p);
  void include(const AABB& other);
  // Compatibility helpers: some modules use 'expand' terminology
  inline void expand(const std::array<real_t,3>& p) { include(p); }
  inline void expand(const AABB& other) { include(other); }
  bool overlaps(const AABB& other) const;
  bool contains(const std::array<real_t,3>& p) const;
  std::array<real_t,3> center() const;
  std::array<real_t,3> extents() const;
  real_t volume() const;
  real_t surface_area() const;
  // Compute closest point on box to p
  std::array<real_t,3> closest_point(const std::array<real_t,3>& p) const;
};

/**
 * @brief Ray for intersection tests
 */
struct Ray {
  std::array<real_t,3> origin;
  std::array<real_t,3> direction;  // Should be normalized
  real_t t_min = 0.0;
  real_t t_max = 1e30;
  // Compatibility alias used by some accelerators/tests
  real_t max_t = 1e30;

  Ray() = default;
  Ray(const std::array<real_t,3>& o, const std::array<real_t,3>& d,
      real_t tmin = 0.0, real_t tmax = 1e30)
      : origin(o), direction(d), t_min(tmin), t_max(tmax), max_t(tmax) {}

  std::array<real_t,3> point_at(real_t t) const;
};


// ---- Geometric predicates ----

/**
 * @brief Check if point is inside triangle (2D test in plane of triangle)
 * @param p Point to test
 * @param a,b,c Triangle vertices
 * @param tol Tolerance for edge cases
 * @return True if point is inside triangle
 */
bool point_in_triangle(const std::array<real_t,3>& p,
                       const std::array<real_t,3>& a,
                       const std::array<real_t,3>& b,
                       const std::array<real_t,3>& c,
                       real_t tol = 1e-10);

/**
 * @brief Check if point is inside tetrahedron
 * @param p Point to test
 * @param a,b,c,d Tetrahedron vertices
 * @param tol Tolerance
 * @return True if point is inside tetrahedron
 */
bool point_in_tetrahedron(const std::array<real_t,3>& p,
                         const std::array<real_t,3>& a,
                         const std::array<real_t,3>& b,
                         const std::array<real_t,3>& c,
                         const std::array<real_t,3>& d,
                         real_t tol = 1e-10);

/**
 * @brief Check if point is inside hexahedron
 * @param p Point to test
 * @param vertices 8 vertices of hexahedron (standard ordering)
 * @param tol Tolerance
 * @return True if point is inside hexahedron
 */
bool point_in_hexahedron(const std::array<real_t,3>& p,
                        const std::vector<std::array<real_t,3>>& vertices,
                        real_t tol = 1e-10);

/**
 * @brief Check if point is inside wedge (triangular prism)
 * @param p Point to test
 * @param vertices 6 vertices of wedge
 * @param tol Tolerance
 * @return True if point is inside wedge
 */
bool point_in_wedge(const std::array<real_t,3>& p,
                   const std::vector<std::array<real_t,3>>& vertices,
                   real_t tol = 1e-10);

/**
 * @brief Check if point is inside pyramid
 * @param p Point to test
 * @param vertices 5 vertices of pyramid (quad base + apex)
 * @param tol Tolerance
 * @return True if point is inside pyramid
 */
bool point_in_pyramid(const std::array<real_t,3>& p,
                     const std::vector<std::array<real_t,3>>& vertices,
                     real_t tol = 1e-10);

/**
 * @brief Generic point-in-cell test dispatching by shape
 * @param p Point to test
 * @param shape Cell shape
 * @param vertices Cell vertices
 * @param tol Tolerance
 * @return True if point is inside cell
 */
bool point_in_cell(const std::array<real_t,3>& p,
                  const CellShape& shape,
                  const std::vector<std::array<real_t,3>>& vertices,
                  real_t tol = 1e-10);

// ---- Ray intersection tests ----

/**
 * @brief Ray-triangle intersection test (Möller-Trumbore algorithm)
 * @param ray The ray
 * @param a,b,c Triangle vertices
 * @param[out] t Intersection parameter
 * @param[out] uv Barycentric coordinates (optional)
 * @return True if ray intersects triangle
 */
bool ray_triangle_intersect(const Ray& ray,
                           const std::array<real_t,3>& a,
                           const std::array<real_t,3>& b,
                           const std::array<real_t,3>& c,
                           real_t& t,
                           std::array<real_t,2>* uv = nullptr);

/**
 * @brief Ray-AABB intersection test (slab method)
 * @param ray The ray
 * @param aabb The bounding box
 * @param[out] t_near Near intersection parameter
 * @param[out] t_far Far intersection parameter
 * @return True if ray intersects AABB
 */
bool ray_aabb_intersect(const Ray& ray,
                       const AABB& aabb,
                       real_t& t_near,
                       real_t& t_far);

/**
 * @brief Ray-sphere intersection test
 * @param ray The ray
 * @param center Sphere center
 * @param radius Sphere radius
 * @param[out] t1 First intersection parameter
 * @param[out] t2 Second intersection parameter
 * @return True if ray intersects sphere
 */
bool ray_sphere_intersect(const Ray& ray,
                         const std::array<real_t,3>& center,
                         real_t radius,
                         real_t& t1,
                         real_t& t2);

// ---- Distance computations ----

/**
 * @brief Distance from point to line segment
 * @param p Query point
 * @param a,b Segment endpoints
 * @param[out] closest Closest point on segment (optional)
 * @return Distance to segment
 */
real_t point_segment_distance(const std::array<real_t,3>& p,
                             const std::array<real_t,3>& a,
                             const std::array<real_t,3>& b,
                             std::array<real_t,3>* closest = nullptr);

/**
 * @brief Distance from point to triangle
 * @param p Query point
 * @param a,b,c Triangle vertices
 * @param[out] closest Closest point on triangle (optional)
 * @return Distance to triangle
 */
real_t point_triangle_distance(const std::array<real_t,3>& p,
                              const std::array<real_t,3>& a,
                              const std::array<real_t,3>& b,
                              const std::array<real_t,3>& c,
                              std::array<real_t,3>* closest = nullptr);

/**
 * @brief Distance from point to AABB
 * @param p Query point
 * @param aabb The bounding box
 * @return Distance to AABB (0 if inside)
 */
real_t point_aabb_distance(const std::array<real_t,3>& p,
                          const AABB& aabb);

// ---- Barycentric coordinates ----

/**
 * @brief Compute barycentric coordinates for point in triangle
 * @param p Point
 * @param a,b,c Triangle vertices
 * @return Barycentric coordinates (u,v,w) where u+v+w=1
 */
std::array<real_t,3> triangle_barycentric(const std::array<real_t,3>& p,
                                         const std::array<real_t,3>& a,
                                         const std::array<real_t,3>& b,
                                         const std::array<real_t,3>& c);

/**
 * @brief Compute barycentric coordinates for point in tetrahedron
 * @param p Point
 * @param a,b,c,d Tetrahedron vertices
 * @return Barycentric coordinates (u,v,w,t) where u+v+w+t=1
 */
std::array<real_t,4> tetrahedron_barycentric(const std::array<real_t,3>& p,
                                            const std::array<real_t,3>& a,
                                            const std::array<real_t,3>& b,
                                            const std::array<real_t,3>& c,
                                            const std::array<real_t,3>& d);

// ---- Geometric utilities ----

/**
 * @brief Compute triangle area
 * @param a,b,c Triangle vertices
 * @return Triangle area
 */
real_t triangle_area(const std::array<real_t,3>& a,
                    const std::array<real_t,3>& b,
                    const std::array<real_t,3>& c);

/**
 * @brief Compute triangle normal (not normalized)
 * @param a,b,c Triangle vertices
 * @return Triangle normal vector
 */
std::array<real_t,3> triangle_normal(const std::array<real_t,3>& a,
                                    const std::array<real_t,3>& b,
                                    const std::array<real_t,3>& c);

/**
 * @brief Compute tetrahedron volume
 * @param a,b,c,d Tetrahedron vertices
 * @return Tetrahedron volume (signed)
 */
real_t tetrahedron_volume(const std::array<real_t,3>& a,
                         const std::array<real_t,3>& b,
                         const std::array<real_t,3>& c,
                         const std::array<real_t,3>& d);

/**
 * @brief Check if AABB overlaps with sphere
 * @param aabb The bounding box
 * @param center Sphere center
 * @param radius Sphere radius
 * @return True if AABB and sphere overlap
 */
bool aabb_sphere_overlap(const AABB& aabb,
                        const std::array<real_t,3>& center,
                        real_t radius);

/**
 * @brief Check if two segments intersect (2D test)
 * @param a1,a2 First segment endpoints
 * @param b1,b2 Second segment endpoints
 * @param[out] t Parameter of intersection on first segment (optional)
 * @return True if segments intersect
 */
bool segments_intersect_2d(const std::array<real_t,2>& a1,
                          const std::array<real_t,2>& a2,
                          const std::array<real_t,2>& b1,
                          const std::array<real_t,2>& b2,
                          real_t* t = nullptr);

// ---- Vector operations ----

/**
 * @brief Dot product of 3D vectors
 */
inline real_t dot3(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * @brief Cross product of 3D vectors
 */
inline std::array<real_t,3> cross3(const std::array<real_t,3>& a,
                                  const std::array<real_t,3>& b) {
  return {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
  };
}

/**
 * @brief Normalize 3D vector
 */
inline std::array<real_t,3> normalize3(const std::array<real_t,3>& v) {
  real_t len = std::sqrt(dot3(v, v));
  if (len > 1e-30) {
    return {v[0]/len, v[1]/len, v[2]/len};
  }
  return v;
}

/**
 * @brief Vector subtraction
 */
inline std::array<real_t,3> sub3(const std::array<real_t,3>& a,
                                const std::array<real_t,3>& b) {
  return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

/**
 * @brief Vector addition
 */
inline std::array<real_t,3> add3(const std::array<real_t,3>& a,
                                const std::array<real_t,3>& b) {
  return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}

/**
 * @brief Scalar multiplication
 */
inline std::array<real_t,3> scale3(const std::array<real_t,3>& v, real_t s) {
  return {v[0]*s, v[1]*s, v[2]*s};
}

// Vector utility compat wrappers expected by some accelerators
inline std::array<real_t,3> subtract(const std::array<real_t,3>& a,
                                     const std::array<real_t,3>& b) {
  return sub3(a, b);
}

inline real_t dot(const std::array<real_t,3>& a,
                  const std::array<real_t,3>& b) {
  return dot3(a, b);
}

inline std::array<real_t,3> cross(const std::array<real_t,3>& a,
                                  const std::array<real_t,3>& b) {
  return cross3(a, b);
}

inline real_t norm_squared(const std::array<real_t,3>& v) {
  return dot3(v, v);
}

} // namespace search
} // namespace svmp

#endif // SVMP_SEARCH_PRIMITIVES_H
