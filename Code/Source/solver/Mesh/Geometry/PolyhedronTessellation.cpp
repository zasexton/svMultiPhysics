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

#include "PolyhedronTessellation.h"

#include "../Core/MeshBase.h"
#include "GeometryConfig.h"

#include <algorithm>
#include <cstddef>

namespace svmp {

namespace {

inline const std::vector<real_t>& coords_for(const MeshBase& mesh, Configuration cfg) {
  return ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
             ? mesh.X_cur()
             : mesh.X_ref();
}

inline std::array<real_t, 3> get_vertex(const MeshBase& mesh, index_t v, Configuration cfg) {
  std::array<real_t, 3> pt{0, 0, 0};
  const int dim = mesh.dim();
  const auto& X = coords_for(mesh, cfg);
  if (v < 0 || static_cast<size_t>(v) >= mesh.n_vertices()) return pt;
  if (dim >= 1) pt[0] = X[static_cast<size_t>(v * dim + 0)];
  if (dim >= 2) pt[1] = X[static_cast<size_t>(v * dim + 1)];
  if (dim >= 3) pt[2] = X[static_cast<size_t>(v * dim + 2)];
  return pt;
}

inline std::array<real_t, 3> sub3(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline std::array<real_t, 3> cross3(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b) {
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

inline real_t dot3(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline real_t signed_tet_volume6(const std::array<real_t, 3>& p0,
                                 const std::array<real_t, 3>& p1,
                                 const std::array<real_t, 3>& p2,
                                 const std::array<real_t, 3>& p3) {
  const auto a = sub3(p1, p0);
  const auto b = sub3(p2, p0);
  const auto c = sub3(p3, p0);
  return dot3(a, cross3(b, c));
}

inline void append_oriented_tet(std::vector<PolyhedronTet4>& out,
                                std::array<std::array<real_t, 3>, 4> vertices) {
  const real_t vol6 = signed_tet_volume6(vertices[0], vertices[1], vertices[2], vertices[3]);
  if (std::abs(vol6) <= GeometryConfig::volume_epsilon() * 6.0) return;
  if (vol6 < 0) std::swap(vertices[2], vertices[3]);
  PolyhedronTet4 tet;
  tet.vertices = vertices;
  out.push_back(tet);
}

} // namespace

std::vector<PolyhedronTet4> PolyhedronTessellation::linear_cell_tets(
    const MeshBase& mesh,
    index_t cell,
    Configuration cfg) {
  std::vector<PolyhedronTet4> out;
  const auto shape = mesh.cell_shape(cell);
  if (shape.family == CellFamily::Polyhedron) {
    return convex_star_tets(mesh, cell, cfg);
  }

  const auto cell_vertices = mesh.cell_vertices_span(cell);
  const auto* cell_verts = cell_vertices.first;
  const auto n_cell_verts = cell_vertices.second;
  const std::size_t n_corners =
      shape.num_corners > 0
          ? std::min<std::size_t>(static_cast<std::size_t>(shape.num_corners), n_cell_verts)
          : n_cell_verts;
  if (n_corners == 0u) return out;

  std::vector<std::array<real_t, 3>> points;
  points.reserve(n_corners);
  for (std::size_t i = 0; i < n_corners; ++i) {
    points.push_back(get_vertex(mesh, cell_verts[i], cfg));
  }
  return linear_cell_tets(shape.family, points);
}

std::vector<PolyhedronTet4> PolyhedronTessellation::linear_cell_tets(
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& points) {
  std::vector<PolyhedronTet4> out;

  for (const auto& ids : linear_cell_tet_corner_indices(family, points.size())) {
    std::array<std::array<real_t, 3>, 4> vertices{{
        points[ids[0]],
        points[ids[1]],
        points[ids[2]],
        points[ids[3]]}};
    append_oriented_tet(out, vertices);
  }

  return out;
}

std::vector<std::array<std::size_t, 4>> PolyhedronTessellation::linear_cell_tet_corner_indices(
    CellFamily family,
    std::size_t point_count) {
  std::vector<std::array<std::size_t, 4>> out;
  const auto append = [&](std::array<std::size_t, 4> ids) {
    for (const auto id : ids) {
      if (id >= point_count) return;
    }
    out.push_back(ids);
  };
  switch (family) {
    case CellFamily::Tetra:
      append({0u, 1u, 2u, 3u});
      break;
    case CellFamily::Hex:
      append({0u, 1u, 3u, 4u});
      append({1u, 2u, 3u, 6u});
      append({1u, 4u, 5u, 6u});
      append({3u, 4u, 6u, 7u});
      append({1u, 3u, 4u, 6u});
      break;
    case CellFamily::Wedge:
      append({0u, 1u, 2u, 3u});
      append({1u, 4u, 2u, 3u});
      append({2u, 4u, 5u, 3u});
      break;
    case CellFamily::Pyramid:
      append({0u, 1u, 2u, 4u});
      append({0u, 2u, 3u, 4u});
      break;
    default:
      break;
  }

  return out;
}

std::vector<PolyhedronTet4> PolyhedronTessellation::convex_star_tets(
    const MeshBase& mesh,
    index_t cell,
    Configuration cfg) {

  std::vector<PolyhedronTet4> out;
  const auto faces = mesh.cell_faces(cell);
  if (faces.empty()) return out;

  auto [cell_verts, n_cell_verts] = mesh.cell_vertices_span(cell);
  if (n_cell_verts == 0) return out;

  std::array<real_t, 3> p0{0, 0, 0};
  for (size_t i = 0; i < n_cell_verts; ++i) {
    const auto p = get_vertex(mesh, cell_verts[i], cfg);
    p0[0] += p[0];
    p0[1] += p[1];
    p0[2] += p[2];
  }
  const real_t inv_n = static_cast<real_t>(1.0) / static_cast<real_t>(n_cell_verts);
  p0[0] *= inv_n;
  p0[1] *= inv_n;
  p0[2] *= inv_n;

  for (index_t f : faces) {
    const auto v = mesh.face_vertices(f);
    if (v.size() < 3) continue;
    const auto p_face0 = get_vertex(mesh, v[0], cfg);
    for (size_t i = 1; i + 1 < v.size(); ++i) {
      PolyhedronTet4 tet;
      tet.vertices[0] = p0;
      tet.vertices[1] = p_face0;
      tet.vertices[2] = get_vertex(mesh, v[i], cfg);
      tet.vertices[3] = get_vertex(mesh, v[i + 1], cfg);

      const real_t vol6 = signed_tet_volume6(tet.vertices[0],
                                             tet.vertices[1],
                                             tet.vertices[2],
                                             tet.vertices[3]);
      if (std::abs(vol6) <= GeometryConfig::volume_epsilon() * 6.0) continue;
      if (vol6 < 0) std::swap(tet.vertices[2], tet.vertices[3]);
      out.push_back(tet);
    }
  }

  return out;
}

} // namespace svmp
