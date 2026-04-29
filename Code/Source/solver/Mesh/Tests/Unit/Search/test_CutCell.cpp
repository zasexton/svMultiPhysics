#include "Core/MeshBase.h"
#include "Core/DistributedMesh.h"
#include "Geometry/MeshGeometry.h"
#include "Geometry/PolyGeometry.h"
#include "Geometry/PolyhedronTessellation.h"
#include "Search/CutCell.h"
#include "Topology/CellTopology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace svmp;
using namespace svmp::search;

namespace {

MeshBase make_tetra_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0},
      std::vector<offset_t>{0, 4},
      std::vector<index_t>{0, 1, 2, 3},
      std::vector<CellShape>{CellShape{CellFamily::Tetra, 4, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quadratic_line_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.00, 0.0,
          1.0, 0.00, 0.0,
          0.5, 0.08, 0.0},
      std::vector<offset_t>{0, 3},
      std::vector<index_t>{0, 1, 2},
      std::vector<CellShape>{CellShape{CellFamily::Line, 2, 2}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quadratic_tetra_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          0.50, -0.04, 0.0,
         -0.04, 0.50, 0.0,
          0.0, -0.03, 0.50,
          0.50, 0.50, 0.04,
          0.50, 0.0, 0.50,
          0.0, 0.50, 0.50},
      std::vector<offset_t>{0, 10},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      std::vector<CellShape>{CellShape{CellFamily::Tetra, 4, 2}});
  mesh.finalize();
  return mesh;
}

std::vector<std::array<int, 4>> tetra_exponents_vtk_for_test(int p)
{
  std::vector<std::array<int, 4>> exps;
  exps.push_back({p, 0, 0, 0});
  exps.push_back({0, p, 0, 0});
  exps.push_back({0, 0, p, 0});
  exps.push_back({0, 0, 0, p});

  const auto eview = CellTopology::get_edges_view(CellFamily::Tetra);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = eview.pairs_flat[2 * ei + 0];
    const int b = eview.pairs_flat[2 * ei + 1];
    for (int k = 1; k <= p - 1; ++k) {
      std::array<int, 4> e{{0, 0, 0, 0}};
      e[static_cast<std::size_t>(a)] = p - k;
      e[static_cast<std::size_t>(b)] = k;
      exps.push_back(e);
    }
  }

  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Tetra);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    if (e - b != 3) {
      continue;
    }
    const int v0 = fview.indices[b + 0];
    const int v1 = fview.indices[b + 1];
    const int v2 = fview.indices[b + 2];
    for (int i = 1; i <= p - 2; ++i) {
      for (int j = 1; j <= p - 1 - i; ++j) {
        std::array<int, 4> f{{0, 0, 0, 0}};
        f[static_cast<std::size_t>(v0)] = p - i - j;
        f[static_cast<std::size_t>(v1)] = i;
        f[static_cast<std::size_t>(v2)] = j;
        exps.push_back(f);
      }
    }
  }

  for (int i = 1; i <= p - 3; ++i) {
    for (int j = 1; j <= p - 2 - i; ++j) {
      for (int k = 1; k <= p - 1 - i - j; ++k) {
        exps.push_back({p - i - j - k, i, j, k});
      }
    }
  }
  return exps;
}

MeshBase make_cubic_curved_tetra_mesh()
{
  constexpr int order = 3;
  const auto map_point = [](real_t r, real_t s, real_t t) {
    const real_t l0 = real_t{1.0} - r - s - t;
    return std::array<real_t, 3>{{
        r + real_t{0.04} * s * l0 + real_t{0.03} * t * l0,
        s + real_t{0.02} * r * l0,
        t + real_t{0.015} * r * s}};
  };

  std::vector<real_t> coords;
  const auto exps = tetra_exponents_vtk_for_test(order);
  coords.reserve(exps.size() * 3u);
  for (const auto& exp : exps) {
    const real_t r = static_cast<real_t>(exp[1]) / static_cast<real_t>(order);
    const real_t s = static_cast<real_t>(exp[2]) / static_cast<real_t>(order);
    const real_t t = static_cast<real_t>(exp[3]) / static_cast<real_t>(order);
    const auto point = map_point(r, s, t);
    coords.insert(coords.end(), point.begin(), point.end());
  }

  std::vector<index_t> connectivity(exps.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Tetra, 4, order}});
  mesh.finalize();
  return mesh;
}

EmbeddedGeometryDescriptor plane(real_t x, std::uint64_t epoch = 1)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::Plane;
  embedded.origin = {{x, 0.0, 0.0}};
  embedded.normal = {{1.0, 0.0, 0.0}};
  embedded.geometry_epoch = epoch;
  embedded.provenance.persistent_id = "embedded-plane";
  embedded.provenance.name = "plane";
  embedded.provenance.provenance_epoch = epoch;
  return embedded;
}

EmbeddedGeometryDescriptor axis_plane(int axis, real_t coordinate, std::uint64_t epoch)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::Plane;
  embedded.origin = {{0.0, 0.0, 0.0}};
  embedded.origin[static_cast<std::size_t>(axis)] = coordinate;
  embedded.normal = {{0.0, 0.0, 0.0}};
  embedded.normal[static_cast<std::size_t>(axis)] = 1.0;
  embedded.geometry_epoch = epoch;
  embedded.provenance.persistent_id = "axis-plane-" + std::to_string(axis);
  embedded.provenance.name = "axis-plane";
  embedded.provenance.provenance_epoch = epoch;
  return embedded;
}

EmbeddedGeometryDescriptor sphere(
    std::array<real_t, 3> center,
    real_t radius,
    std::uint64_t epoch,
    std::string id)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::Sphere;
  embedded.origin = center;
  embedded.radius = radius;
  embedded.geometry_epoch = epoch;
  embedded.revisions.geometry_epoch = epoch;
  embedded.provenance.persistent_id = std::move(id);
  embedded.provenance.name = embedded.provenance.persistent_id;
  embedded.provenance.provenance_epoch = epoch;
  return embedded;
}

EmbeddedGeometryDescriptor callback_sphere(
    std::array<real_t, 3> center,
    real_t radius,
    std::uint64_t epoch,
    std::string id)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::SignedDistanceCallback;
  embedded.origin = center;
  embedded.radius = radius;
  embedded.geometry_epoch = epoch;
  embedded.revisions.geometry_epoch = epoch;
  embedded.provenance.persistent_id = std::move(id);
  embedded.provenance.name = embedded.provenance.persistent_id;
  embedded.provenance.provenance_epoch = epoch;
  embedded.signed_distance_callback = [center, radius](const std::array<real_t, 3>& point) {
    const real_t dx = point[0] - center[0];
    const real_t dy = point[1] - center[1];
    const real_t dz = point[2] - center[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
  };
  embedded.normal_callback = [center](const std::array<real_t, 3>& point) {
    const std::array<real_t, 3> d{{point[0] - center[0],
                                   point[1] - center[1],
                                   point[2] - center[2]}};
    const real_t n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (n <= real_t{0.0}) {
      return std::array<real_t, 3>{{1.0, 0.0, 0.0}};
    }
    return std::array<real_t, 3>{{d[0] / n, d[1] / n, d[2] / n}};
  };
  embedded.closest_point_callback = [center, radius](const std::array<real_t, 3>& point) {
    const std::array<real_t, 3> d{{point[0] - center[0],
                                   point[1] - center[1],
                                   point[2] - center[2]}};
    const real_t n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const std::array<real_t, 3> unit =
        n <= real_t{0.0}
            ? std::array<real_t, 3>{{1.0, 0.0, 0.0}}
            : std::array<real_t, 3>{{d[0] / n, d[1] / n, d[2] / n}};
    return std::array<real_t, 3>{{center[0] + radius * unit[0],
                                  center[1] + radius * unit[1],
                                  center[2] + radius * unit[2]}};
  };
  return embedded;
}

std::array<real_t, 3> unit_vector_for_test(const std::array<real_t, 3>& v)
{
  const real_t n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (n <= real_t{0.0}) {
    return {{1.0, 0.0, 0.0}};
  }
  return {{v[0] / n, v[1] / n, v[2] / n}};
}

EmbeddedGeometryDescriptor level_set_sphere(
    std::array<real_t, 3> center,
    real_t radius,
    std::uint64_t epoch,
    std::string id)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::LevelSetField;
  embedded.origin = center;
  embedded.radius = radius;
  embedded.geometry_epoch = epoch;
  embedded.revisions.geometry_epoch = epoch;
  embedded.revisions.field_layout_revision = 1;
  embedded.revisions.field_value_revision = epoch;
  embedded.provenance.persistent_id = std::move(id);
  embedded.provenance.name = embedded.provenance.persistent_id;
  embedded.provenance.provenance_epoch = epoch;

  const std::array<std::array<real_t, 3>, 14> directions{{
      {{ 1.0,  0.0,  0.0}}, {{-1.0,  0.0,  0.0}},
      {{ 0.0,  1.0,  0.0}}, {{ 0.0, -1.0,  0.0}},
      {{ 0.0,  0.0,  1.0}}, {{ 0.0,  0.0, -1.0}},
      {{ 1.0,  1.0,  1.0}}, {{ 1.0,  1.0, -1.0}},
      {{ 1.0, -1.0,  1.0}}, {{ 1.0, -1.0, -1.0}},
      {{-1.0,  1.0,  1.0}}, {{-1.0,  1.0, -1.0}},
      {{-1.0, -1.0,  1.0}}, {{-1.0, -1.0, -1.0}}}};
  for (const auto& direction : directions) {
    const auto n = unit_vector_for_test(direction);
    EmbeddedLevelSetSample sample;
    sample.point = {{center[0] + radius * n[0],
                     center[1] + radius * n[1],
                     center[2] + radius * n[2]}};
    sample.value = 0.0;
    sample.gradient = n;
    embedded.level_set_samples.push_back(sample);
  }
  return embedded;
}

EmbeddedGeometryDescriptor triangulated_octahedron_sphere(
    std::array<real_t, 3> center,
    real_t radius,
    std::uint64_t epoch,
    std::string id)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::TriangulatedSurface;
  embedded.origin = center;
  embedded.radius = radius;
  embedded.geometry_epoch = epoch;
  embedded.revisions.geometry_epoch = epoch;
  embedded.revisions.source_surface_revision = epoch;
  embedded.provenance.persistent_id = std::move(id);
  embedded.provenance.name = embedded.provenance.persistent_id;
  embedded.provenance.provenance_epoch = epoch;

  const std::array<real_t, 3> xp{{center[0] + radius, center[1], center[2]}};
  const std::array<real_t, 3> xn{{center[0] - radius, center[1], center[2]}};
  const std::array<real_t, 3> yp{{center[0], center[1] + radius, center[2]}};
  const std::array<real_t, 3> yn{{center[0], center[1] - radius, center[2]}};
  const std::array<real_t, 3> zp{{center[0], center[1], center[2] + radius}};
  const std::array<real_t, 3> zn{{center[0], center[1], center[2] - radius}};

  const auto add_oriented_triangle =
      [&](std::array<real_t, 3> a,
          std::array<real_t, 3> b,
          std::array<real_t, 3> c) {
        const std::array<real_t, 3> ab{{b[0] - a[0], b[1] - a[1], b[2] - a[2]}};
        const std::array<real_t, 3> ac{{c[0] - a[0], c[1] - a[1], c[2] - a[2]}};
        const std::array<real_t, 3> normal{{
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0]}};
        const std::array<real_t, 3> centroid{{
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0}};
        const std::array<real_t, 3> outward{{
            centroid[0] - center[0],
            centroid[1] - center[1],
            centroid[2] - center[2]}};
        const real_t orientation =
            normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2];
        if (orientation < 0.0) {
          std::swap(b, c);
        }
        embedded.surface_triangles.push_back({{{a, b, c}}, embedded.provenance});
      };

  add_oriented_triangle(zp, xp, yp);
  add_oriented_triangle(zp, yp, xn);
  add_oriented_triangle(zp, xn, yn);
  add_oriented_triangle(zp, yn, xp);
  add_oriented_triangle(zn, yp, xp);
  add_oriented_triangle(zn, xn, yp);
  add_oriented_triangle(zn, yn, xn);
  add_oriented_triangle(zn, xp, yn);
  return embedded;
}

EmbeddedGeometryDescriptor boolean_composite(
    EmbeddedGeometryBooleanOperation operation,
    std::vector<EmbeddedGeometryDescriptor> children,
    std::uint64_t epoch,
    std::string id)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::BooleanComposite;
  embedded.boolean_operation = operation;
  embedded.geometry_epoch = epoch;
  embedded.revisions.geometry_epoch = epoch;
  embedded.provenance.persistent_id = std::move(id);
  embedded.provenance.name = embedded.provenance.persistent_id;
  embedded.provenance.provenance_epoch = epoch;
  embedded.children = std::move(children);
  return embedded;
}

MeshBase make_line_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0},
      std::vector<offset_t>{0, 2},
      std::vector<index_t>{0, 1},
      std::vector<CellShape>{CellShape{CellFamily::Line, 2, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_curved_line_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.00,  0.00, 0.0,
          0.22,  0.05, 0.0,
          0.68, -0.03, 0.0,
          1.05,  0.02, 0.0},
      std::vector<offset_t>{0, 4},
      std::vector<index_t>{0, 1, 2, 3},
      std::vector<CellShape>{CellShape{CellFamily::Line, 2, 3}});
  mesh.finalize();
  return mesh;
}

MeshBase make_triangle_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0},
      std::vector<offset_t>{0, 3},
      std::vector<index_t>{0, 1, 2},
      std::vector<CellShape>{CellShape{CellFamily::Triangle, 3, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quadratic_triangle_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.50, -0.10, 0.0,
          0.58, 0.58, 0.0,
         -0.10, 0.50, 0.0},
      std::vector<offset_t>{0, 6},
      std::vector<index_t>{0, 1, 2, 3, 4, 5},
      std::vector<CellShape>{CellShape{CellFamily::Triangle, 3, 2}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_curved_triangle_mesh()
{
  const auto p = [](real_t r, real_t s) {
    const real_t bubble = s * (real_t{1.0} - r - s);
    return std::array<real_t, 3>{{
        r + real_t{0.04} * bubble,
        s + real_t{0.02} * r * (real_t{1.0} - r - s),
        real_t{0.03} * r * s}};
  };
  const std::vector<std::array<real_t, 3>> pts{
      p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0),
      p(1.0 / 3.0, 0.0), p(2.0 / 3.0, 0.0),
      p(2.0 / 3.0, 1.0 / 3.0), p(1.0 / 3.0, 2.0 / 3.0),
      p(0.0, 2.0 / 3.0), p(0.0, 1.0 / 3.0),
      p(1.0 / 3.0, 1.0 / 3.0)};
  std::vector<real_t> coords;
  coords.reserve(pts.size() * 3u);
  for (const auto& point : pts) {
    coords.insert(coords.end(), point.begin(), point.end());
  }
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, 10},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      std::vector<CellShape>{CellShape{CellFamily::Triangle, 3, 3}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_non_graph_triangle_mesh()
{
  const auto p = [](real_t r, real_t s) {
    const real_t centered = r - real_t{0.5};
    return std::array<real_t, 3>{{centered * centered, s, real_t{0.02} * r * s}};
  };
  const std::vector<std::array<real_t, 3>> pts{
      p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0),
      p(1.0 / 3.0, 0.0), p(2.0 / 3.0, 0.0),
      p(2.0 / 3.0, 1.0 / 3.0), p(1.0 / 3.0, 2.0 / 3.0),
      p(0.0, 2.0 / 3.0), p(0.0, 1.0 / 3.0),
      p(1.0 / 3.0, 1.0 / 3.0)};
  std::vector<real_t> coords;
  coords.reserve(pts.size() * 3u);
  for (const auto& point : pts) {
    coords.insert(coords.end(), point.begin(), point.end());
  }
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, 10},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      std::vector<CellShape>{CellShape{CellFamily::Triangle, 3, 3}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quad_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                          1.0, 1.0, 0.0,
                          0.0, 1.0, 0.0},
      std::vector<offset_t>{0, 4},
      std::vector<index_t>{0, 1, 2, 3},
      std::vector<CellShape>{CellShape{CellFamily::Quad, 4, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quadratic_quad_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
         -1.0, -1.0, 0.0,
          1.0, -1.0, 0.0,
          1.0,  1.0, 0.0,
         -1.0,  1.0, 0.0,
          0.0, -1.0, 0.12,
          1.0,  0.0, 0.08,
          0.0,  1.0, 0.10,
         -1.0,  0.0, -0.06},
      std::vector<offset_t>{0, 8},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7},
      std::vector<CellShape>{CellShape{CellFamily::Quad, 4, 2}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_curved_quad_mesh()
{
  const auto p = [](int i, int j) {
    const real_t r = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) / real_t{3.0};
    const real_t s = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(j) / real_t{3.0};
    return std::array<real_t, 3>{{
        real_t{0.5} * (r + real_t{1.0}) + real_t{0.06} * s * (real_t{1.0} - r * r),
        s,
        real_t{0.04} * r * s}};
  };
  const std::vector<std::array<real_t, 3>> pts{
      p(0, 0), p(3, 0), p(3, 3), p(0, 3),
      p(1, 0), p(2, 0),
      p(3, 1), p(3, 2),
      p(2, 3), p(1, 3),
      p(0, 2), p(0, 1),
      p(1, 1), p(1, 2), p(2, 1), p(2, 2)};
  std::vector<real_t> coords;
  coords.reserve(pts.size() * 3u);
  for (const auto& point : pts) {
    coords.insert(coords.end(), point.begin(), point.end());
  }
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, 16},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7,
                           8, 9, 10, 11, 12, 13, 14, 15},
      std::vector<CellShape>{CellShape{CellFamily::Quad, 4, 3}});
  mesh.finalize();
  return mesh;
}

MeshBase make_hex_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                          1.0, 1.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0,
                          1.0, 0.0, 1.0,
                          1.0, 1.0, 1.0,
                          0.0, 1.0, 1.0},
      std::vector<offset_t>{0, 8},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7},
      std::vector<CellShape>{CellShape{CellFamily::Hex, 8, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quadratic_hex_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
         -1.0, -1.0, -1.0,
          1.0, -1.0, -1.0,
          1.0,  1.0, -1.0,
         -1.0,  1.0, -1.0,
         -1.0, -1.0,  1.0,
          1.0, -1.0,  1.0,
          1.0,  1.0,  1.0,
         -1.0,  1.0,  1.0,
          0.0, -1.0, -0.88,
          1.0,  0.0, -0.92,
          0.0,  1.0, -0.90,
         -1.0,  0.0, -1.08,
          0.0, -1.0,  1.06,
          1.0,  0.0,  0.94,
          0.0,  1.0,  1.04,
         -1.0,  0.0,  0.96,
         -1.08, -1.0,  0.0,
          1.06, -1.0,  0.0,
          1.04,  1.0,  0.0,
         -0.96,  1.0,  0.0},
      std::vector<offset_t>{0, 20},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7,
                           8, 9, 10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19},
      std::vector<CellShape>{CellShape{CellFamily::Hex, 8, 2}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_non_graph_quad_mesh()
{
  const auto p = [](int i, int j) {
    const real_t r = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) / real_t{3.0};
    const real_t s = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(j) / real_t{3.0};
    return std::array<real_t, 3>{{r * r, s, real_t{0.02} * r * s}};
  };
  const std::vector<std::array<real_t, 3>> pts{
      p(0, 0), p(3, 0), p(3, 3), p(0, 3),
      p(1, 0), p(2, 0),
      p(3, 1), p(3, 2),
      p(2, 3), p(1, 3),
      p(0, 2), p(0, 1),
      p(1, 1), p(1, 2), p(2, 1), p(2, 2)};
  std::vector<real_t> coords;
  coords.reserve(pts.size() * 3u);
  for (const auto& point : pts) {
    coords.insert(coords.end(), point.begin(), point.end());
  }
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, 16},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7,
                           8, 9, 10, 11, 12, 13, 14, 15},
      std::vector<CellShape>{CellShape{CellFamily::Quad, 4, 3}});
  mesh.finalize();
  return mesh;
}

std::vector<std::array<int, 3>> hex_lagrange_indices_vtk_for_test(int p)
{
  std::vector<std::array<int, 3>> idx;
  const std::array<std::array<int, 3>, 8> corner_grid{{
      {{0, 0, 0}}, {{p, 0, 0}}, {{p, p, 0}}, {{0, p, 0}},
      {{0, 0, p}}, {{p, 0, p}}, {{p, p, p}}, {{0, p, p}}}};
  for (const auto& corner : corner_grid) {
    idx.push_back(corner);
  }

  const auto eview = CellTopology::get_edges_view(CellFamily::Hex);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = eview.pairs_flat[2 * ei + 0];
    const int b = eview.pairs_flat[2 * ei + 1];
    const auto A = corner_grid[static_cast<std::size_t>(a)];
    const auto B = corner_grid[static_cast<std::size_t>(b)];
    for (int k = 1; k <= p - 1; ++k) {
      const real_t u = static_cast<real_t>(k) / static_cast<real_t>(p);
      idx.push_back({{
          static_cast<int>(std::lround((real_t{1.0} - u) * A[0] + u * B[0])),
          static_cast<int>(std::lround((real_t{1.0} - u) * A[1] + u * B[1])),
          static_cast<int>(std::lround((real_t{1.0} - u) * A[2] + u * B[2]))}});
    }
  }

  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    if (e - b != 4) {
      continue;
    }
    const auto A = corner_grid[static_cast<std::size_t>(fview.indices[b + 0])];
    const auto B = corner_grid[static_cast<std::size_t>(fview.indices[b + 1])];
    const auto C = corner_grid[static_cast<std::size_t>(fview.indices[b + 2])];
    const auto D = corner_grid[static_cast<std::size_t>(fview.indices[b + 3])];
    for (int i = 1; i <= p - 1; ++i) {
      for (int j = 1; j <= p - 1; ++j) {
        const real_t u = static_cast<real_t>(i) / static_cast<real_t>(p);
        const real_t v = static_cast<real_t>(j) / static_cast<real_t>(p);
        std::array<int, 3> g{{0, 0, 0}};
        for (int d = 0; d < 3; ++d) {
          const auto dim = static_cast<std::size_t>(d);
          const real_t x =
              (real_t{1.0} - u) * (real_t{1.0} - v) * A[dim] +
              u * (real_t{1.0} - v) * B[dim] +
              u * v * C[dim] +
              (real_t{1.0} - u) * v * D[dim];
          g[dim] = static_cast<int>(std::lround(x));
        }
        idx.push_back(g);
      }
    }
  }

  for (int i = 1; i <= p - 1; ++i) {
    for (int j = 1; j <= p - 1; ++j) {
      for (int k = 1; k <= p - 1; ++k) {
        idx.push_back({{i, j, k}});
      }
    }
  }
  return idx;
}

MeshBase make_cubic_curved_hex_mesh()
{
  constexpr int order = 3;
  const auto map_point = [](real_t r, real_t s, real_t t) {
    return std::array<real_t, 3>{{
        real_t{0.5} * (r + real_t{1.0}) +
            real_t{0.04} * s * (real_t{1.0} - r * r) +
            real_t{0.02} * t * (real_t{1.0} - r * r),
        s + real_t{0.03} * r * (real_t{1.0} - s * s),
        t + real_t{0.02} * r * s * (real_t{1.0} - t * t)}};
  };

  const auto labels = hex_lagrange_indices_vtk_for_test(order);
  std::vector<real_t> coords;
  coords.reserve(labels.size() * 3u);
  for (const auto& label : labels) {
    const real_t r = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[0]) /
                                       static_cast<real_t>(order);
    const real_t s = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[1]) /
                                       static_cast<real_t>(order);
    const real_t t = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[2]) /
                                       static_cast<real_t>(order);
    const auto point = map_point(r, s, t);
    coords.insert(coords.end(), point.begin(), point.end());
  }

  std::vector<index_t> connectivity(labels.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Hex, 8, order}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_affine_hex_mesh()
{
  constexpr int order = 3;
  const auto labels = hex_lagrange_indices_vtk_for_test(order);
  std::vector<real_t> coords;
  coords.reserve(labels.size() * 3u);
  for (const auto& label : labels) {
    const real_t r = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[0]) /
                                       static_cast<real_t>(order);
    const real_t s = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[1]) /
                                       static_cast<real_t>(order);
    const real_t t = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[2]) /
                                       static_cast<real_t>(order);
    coords.insert(coords.end(), {r, s, t});
  }

  std::vector<index_t> connectivity(labels.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Hex, 8, order}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_non_graph_hex_mesh()
{
  constexpr int order = 3;
  const auto labels = hex_lagrange_indices_vtk_for_test(order);
  std::vector<real_t> coords;
  coords.reserve(labels.size() * 3u);
  for (const auto& label : labels) {
    const real_t r = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[0]) /
                                       static_cast<real_t>(order);
    const real_t s = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[1]) /
                                       static_cast<real_t>(order);
    const real_t t = real_t{-1.0} + real_t{2.0} *
                                       static_cast<real_t>(label[2]) /
                                       static_cast<real_t>(order);
    const std::array<real_t, 3> point{{r * r, s, t}};
    coords.insert(coords.end(), point.begin(), point.end());
  }

  std::vector<index_t> connectivity(labels.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Hex, 8, order}});
  mesh.finalize();
  return mesh;
}

struct WedgeLagrangeLabelForTest {
  std::array<int, 3> tri_exp{{0, 0, 0}};
  int kz{0};
};

std::array<real_t, 3> wedge_corner_parametric_for_test(int local_corner)
{
  switch (local_corner) {
    case 0: return {{0.0, 0.0, -1.0}};
    case 1: return {{1.0, 0.0, -1.0}};
    case 2: return {{0.0, 1.0, -1.0}};
    case 3: return {{0.0, 0.0,  1.0}};
    case 4: return {{1.0, 0.0,  1.0}};
    case 5: return {{0.0, 1.0,  1.0}};
    default: return {{0.0, 0.0, 0.0}};
  }
}

WedgeLagrangeLabelForTest wedge_label_from_param_for_test(
    int p,
    const std::array<real_t, 3>& xi)
{
  int a1 = static_cast<int>(std::lround(static_cast<real_t>(p) * xi[0]));
  int a2 = static_cast<int>(std::lround(static_cast<real_t>(p) * xi[1]));
  a1 = std::clamp(a1, 0, p);
  a2 = std::clamp(a2, 0, p);
  int a0 = p - a1 - a2;
  if (a0 < 0) {
    a0 = 0;
  }
  int kz = static_cast<int>(std::lround((xi[2] + real_t{1.0}) *
                                        real_t{0.5} *
                                        static_cast<real_t>(p)));
  kz = std::clamp(kz, 0, p);
  return {{{a0, a1, a2}}, kz};
}

std::vector<WedgeLagrangeLabelForTest> wedge_lagrange_labels_vtk_for_test(int p)
{
  std::vector<WedgeLagrangeLabelForTest> labels;
  labels.reserve(static_cast<std::size_t>((p + 1) * (p + 1) * (p + 2) / 2));

  for (int c = 0; c < 6; ++c) {
    labels.push_back(wedge_label_from_param_for_test(
        p, wedge_corner_parametric_for_test(c)));
  }

  const auto eview = CellTopology::get_edges_view(CellFamily::Wedge);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = eview.pairs_flat[2 * ei + 0];
    const int b = eview.pairs_flat[2 * ei + 1];
    const auto A = wedge_corner_parametric_for_test(a);
    const auto B = wedge_corner_parametric_for_test(b);
    for (int k = 1; k <= p - 1; ++k) {
      const real_t u = static_cast<real_t>(k) / static_cast<real_t>(p);
      labels.push_back(wedge_label_from_param_for_test(
          p,
          {{(real_t{1.0} - u) * A[0] + u * B[0],
            (real_t{1.0} - u) * A[1] + u * B[1],
            (real_t{1.0} - u) * A[2] + u * B[2]}}));
    }
  }

  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    const int fv = e - b;
    if (fv == 3) {
      const auto A = wedge_corner_parametric_for_test(fview.indices[b + 0]);
      const auto B = wedge_corner_parametric_for_test(fview.indices[b + 1]);
      const auto C = wedge_corner_parametric_for_test(fview.indices[b + 2]);
      for (int i = 1; i <= p - 2; ++i) {
        for (int j = 1; j <= p - 1 - i; ++j) {
          const real_t w1 = static_cast<real_t>(i) / static_cast<real_t>(p);
          const real_t w2 = static_cast<real_t>(j) / static_cast<real_t>(p);
          const real_t w0 = real_t{1.0} - w1 - w2;
          labels.push_back(wedge_label_from_param_for_test(
              p,
              {{w0 * A[0] + w1 * B[0] + w2 * C[0],
                w0 * A[1] + w1 * B[1] + w2 * C[1],
                w0 * A[2] + w1 * B[2] + w2 * C[2]}}));
        }
      }
    } else if (fv == 4) {
      const auto A = wedge_corner_parametric_for_test(fview.indices[b + 0]);
      const auto B = wedge_corner_parametric_for_test(fview.indices[b + 1]);
      const auto C = wedge_corner_parametric_for_test(fview.indices[b + 2]);
      const auto D = wedge_corner_parametric_for_test(fview.indices[b + 3]);
      for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
          const real_t u = static_cast<real_t>(i) / static_cast<real_t>(p);
          const real_t v = static_cast<real_t>(j) / static_cast<real_t>(p);
          labels.push_back(wedge_label_from_param_for_test(
              p,
              {{(real_t{1.0} - u) * (real_t{1.0} - v) * A[0] +
                    u * (real_t{1.0} - v) * B[0] + u * v * C[0] +
                    (real_t{1.0} - u) * v * D[0],
                (real_t{1.0} - u) * (real_t{1.0} - v) * A[1] +
                    u * (real_t{1.0} - v) * B[1] + u * v * C[1] +
                    (real_t{1.0} - u) * v * D[1],
                (real_t{1.0} - u) * (real_t{1.0} - v) * A[2] +
                    u * (real_t{1.0} - v) * B[2] + u * v * C[2] +
                    (real_t{1.0} - u) * v * D[2]}}));
        }
      }
    }
  }

  for (int i = 1; i <= p - 2; ++i) {
    for (int j = 1; j <= p - 1 - i; ++j) {
      for (int k = 1; k <= p - 1; ++k) {
        labels.push_back({{{p - i - j, i, j}}, k});
      }
    }
  }
  return labels;
}

MeshBase make_cubic_curved_wedge_mesh()
{
  constexpr int order = 3;
  const auto map_point = [](real_t xi, real_t eta, real_t zeta) {
    const real_t l0 = real_t{1.0} - xi - eta;
    return std::array<real_t, 3>{{
        real_t{0.5} * (zeta + real_t{1.0}) +
            real_t{0.04} * xi * (real_t{1.0} - zeta * zeta) +
            real_t{0.02} * eta * (real_t{1.0} - zeta * zeta),
        xi + real_t{0.02} * zeta * xi * l0,
        eta + real_t{0.015} * zeta * xi * eta}};
  };

  const auto labels = wedge_lagrange_labels_vtk_for_test(order);
  std::vector<real_t> coords;
  coords.reserve(labels.size() * 3u);
  for (const auto& label : labels) {
    const real_t xi = static_cast<real_t>(label.tri_exp[1]) /
                      static_cast<real_t>(order);
    const real_t eta = static_cast<real_t>(label.tri_exp[2]) /
                       static_cast<real_t>(order);
    const real_t zeta = real_t{-1.0} + real_t{2.0} *
                                          static_cast<real_t>(label.kz) /
                                          static_cast<real_t>(order);
    const auto point = map_point(xi, eta, zeta);
    coords.insert(coords.end(), point.begin(), point.end());
  }

  std::vector<index_t> connectivity(labels.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Wedge, 6, order}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_non_graph_wedge_mesh()
{
  constexpr int order = 3;
  const auto labels = wedge_lagrange_labels_vtk_for_test(order);
  std::vector<real_t> coords;
  coords.reserve(labels.size() * 3u);
  for (const auto& label : labels) {
    const real_t xi = static_cast<real_t>(label.tri_exp[1]) /
                      static_cast<real_t>(order);
    const real_t eta = static_cast<real_t>(label.tri_exp[2]) /
                       static_cast<real_t>(order);
    const real_t zeta = real_t{-1.0} + real_t{2.0} *
                                          static_cast<real_t>(label.kz) /
                                          static_cast<real_t>(order);
    const std::array<real_t, 3> point{{zeta * zeta, xi, eta}};
    coords.insert(coords.end(), point.begin(), point.end());
  }

  std::vector<index_t> connectivity(labels.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Wedge, 6, order}});
  mesh.finalize();
  return mesh;
}

std::array<real_t, 3> pyramid_corner_parametric_for_test(int local_corner)
{
  switch (local_corner) {
    case 0: return {{-1.0, -1.0, 0.0}};
    case 1: return {{ 1.0, -1.0, 0.0}};
    case 2: return {{ 1.0,  1.0, 0.0}};
    case 3: return {{-1.0,  1.0, 0.0}};
    case 4: return {{ 0.0,  0.0, 1.0}};
    default: return {{0.0, 0.0, 0.0}};
  }
}

std::array<real_t, 3> interpolate_pyramid_parametric_for_test(
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    real_t u)
{
  return {{(real_t{1.0} - u) * a[0] + u * b[0],
           (real_t{1.0} - u) * a[1] + u * b[1],
           (real_t{1.0} - u) * a[2] + u * b[2]}};
}

std::array<real_t, 3> bilerp_pyramid_parametric_for_test(
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c,
    const std::array<real_t, 3>& d,
    real_t u,
    real_t v)
{
  const real_t w00 = (real_t{1.0} - u) * (real_t{1.0} - v);
  const real_t w10 = u * (real_t{1.0} - v);
  const real_t w11 = u * v;
  const real_t w01 = (real_t{1.0} - u) * v;
  return {{w00 * a[0] + w10 * b[0] + w11 * c[0] + w01 * d[0],
           w00 * a[1] + w10 * b[1] + w11 * c[1] + w01 * d[1],
           w00 * a[2] + w10 * b[2] + w11 * c[2] + w01 * d[2]}};
}

std::vector<std::array<real_t, 3>> pyramid_lagrange_parametric_points_vtk_for_test(int p)
{
  const auto pattern =
      CellTopology::high_order_pattern(
          CellFamily::Pyramid, p, CellTopology::HighOrderKind::Lagrange);
  const auto eview = CellTopology::get_edges_view(CellFamily::Pyramid);
  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);

  std::vector<std::array<real_t, 3>> points;
  points.reserve(pattern.sequence.size());
  for (const auto& role : pattern.sequence) {
    switch (role.role) {
      case CellTopology::HONodeRole::Corner:
        points.push_back(pyramid_corner_parametric_for_test(role.idx0));
        break;
      case CellTopology::HONodeRole::Edge: {
        const int edge = role.idx0;
        const int a = eview.pairs_flat[2 * edge + 0];
        const int b = eview.pairs_flat[2 * edge + 1];
        const real_t u = static_cast<real_t>(role.idx1) / static_cast<real_t>(p);
        points.push_back(interpolate_pyramid_parametric_for_test(
            pyramid_corner_parametric_for_test(a),
            pyramid_corner_parametric_for_test(b),
            u));
        break;
      }
      case CellTopology::HONodeRole::Face: {
        const int face = role.idx0;
        const int begin = fview.offsets[face];
        const int end = fview.offsets[face + 1];
        const int vertices = end - begin;
        if (vertices == 3) {
          const auto a = pyramid_corner_parametric_for_test(fview.indices[begin + 0]);
          const auto b = pyramid_corner_parametric_for_test(fview.indices[begin + 1]);
          const auto c = pyramid_corner_parametric_for_test(fview.indices[begin + 2]);
          const real_t w1 = static_cast<real_t>(role.idx1) / static_cast<real_t>(p);
          const real_t w2 = static_cast<real_t>(role.idx2) / static_cast<real_t>(p);
          const real_t w0 = real_t{1.0} - w1 - w2;
          points.push_back({{w0 * a[0] + w1 * b[0] + w2 * c[0],
                             w0 * a[1] + w1 * b[1] + w2 * c[1],
                             w0 * a[2] + w1 * b[2] + w2 * c[2]}});
        } else if (vertices == 4) {
          const real_t u = static_cast<real_t>(role.idx1) / static_cast<real_t>(p);
          const real_t v = static_cast<real_t>(role.idx2) / static_cast<real_t>(p);
          points.push_back(bilerp_pyramid_parametric_for_test(
              pyramid_corner_parametric_for_test(fview.indices[begin + 0]),
              pyramid_corner_parametric_for_test(fview.indices[begin + 1]),
              pyramid_corner_parametric_for_test(fview.indices[begin + 2]),
              pyramid_corner_parametric_for_test(fview.indices[begin + 3]),
              u,
              v));
        }
        break;
      }
      case CellTopology::HONodeRole::Volume: {
        const real_t zeta = static_cast<real_t>(role.idx2) / static_cast<real_t>(p);
        const real_t scale = real_t{1.0} - zeta;
        const int layer_nodes = (p + 1) - role.idx2;
        const int denominator = layer_nodes - 1;
        const real_t u = real_t{-1.0} +
                         real_t{2.0} * static_cast<real_t>(role.idx0) /
                             static_cast<real_t>(denominator);
        const real_t v = real_t{-1.0} +
                         real_t{2.0} * static_cast<real_t>(role.idx1) /
                             static_cast<real_t>(denominator);
        points.push_back({{scale * u, scale * v, zeta}});
        break;
      }
    }
  }
  return points;
}

MeshBase make_cubic_curved_pyramid_mesh()
{
  constexpr int order = 3;
  const auto map_point = [](real_t xi, real_t eta, real_t zeta) {
    return std::array<real_t, 3>{{
        zeta + real_t{0.035} * xi * (real_t{1.0} - zeta) +
            real_t{0.02} * eta * (real_t{1.0} - zeta),
        xi + real_t{0.02} * zeta * xi * (real_t{1.0} - zeta),
        eta + real_t{0.015} * zeta * xi * eta}};
  };

  const auto points = pyramid_lagrange_parametric_points_vtk_for_test(order);
  std::vector<real_t> coords;
  coords.reserve(points.size() * 3u);
  for (const auto& xi : points) {
    const auto point = map_point(xi[0], xi[1], xi[2]);
    coords.insert(coords.end(), point.begin(), point.end());
  }

  std::vector<index_t> connectivity(points.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Pyramid, 5, order}});
  mesh.finalize();
  return mesh;
}

MeshBase make_cubic_non_graph_pyramid_mesh()
{
  constexpr int order = 3;
  const auto points = pyramid_lagrange_parametric_points_vtk_for_test(order);
  std::vector<real_t> coords;
  coords.reserve(points.size() * 3u);
  for (const auto& xi : points) {
    const real_t centered = xi[2] - real_t{0.5};
    const std::array<real_t, 3> point{{centered * centered, xi[0], xi[1]}};
    coords.insert(coords.end(), point.begin(), point.end());
  }

  std::vector<index_t> connectivity(points.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Pyramid, 5, order}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quadratic_wedge_mesh()
{
  const std::array<std::array<real_t, 3>, 6> corners{{
      {{0.0, 0.0, 0.0}},
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
      {{0.0, 0.0, 1.0}},
      {{1.0, 0.0, 1.0}},
      {{0.0, 1.0, 1.0}}}};
  std::vector<std::array<real_t, 3>> points(corners.begin(), corners.end());
  const auto eview = CellTopology::get_edges_view(CellFamily::Wedge);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const auto& a = corners[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 0])];
    const auto& b = corners[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 1])];
    points.push_back({{real_t{0.5} * (a[0] + b[0]),
                       real_t{0.5} * (a[1] + b[1]),
                       real_t{0.5} * (a[2] + b[2])}});
  }

  std::vector<real_t> coords;
  coords.reserve(points.size() * 3u);
  for (const auto& point : points) {
    coords.insert(coords.end(), point.begin(), point.end());
  }
  std::vector<index_t> connectivity(points.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Wedge, 6, 2}});
  mesh.finalize();
  return mesh;
}

MeshBase make_quadratic_pyramid_mesh()
{
  const std::array<std::array<real_t, 3>, 5> corners{{
      {{0.0, 0.0, 0.0}},
      {{1.0, 0.0, 0.0}},
      {{1.0, 1.0, 0.0}},
      {{0.0, 1.0, 0.0}},
      {{0.5, 0.5, 1.0}}}};
  std::vector<std::array<real_t, 3>> points(corners.begin(), corners.end());
  const auto eview = CellTopology::get_edges_view(CellFamily::Pyramid);
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const auto& a = corners[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 0])];
    const auto& b = corners[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 1])];
    points.push_back({{real_t{0.5} * (a[0] + b[0]),
                       real_t{0.5} * (a[1] + b[1]),
                       real_t{0.5} * (a[2] + b[2])}});
  }

  std::vector<real_t> coords;
  coords.reserve(points.size() * 3u);
  for (const auto& point : points) {
    coords.insert(coords.end(), point.begin(), point.end());
  }
  std::vector<index_t> connectivity(points.size());
  for (std::size_t i = 0; i < connectivity.size(); ++i) {
    connectivity[i] = static_cast<index_t>(i);
  }

  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      coords,
      std::vector<offset_t>{0, static_cast<offset_t>(connectivity.size())},
      connectivity,
      std::vector<CellShape>{CellShape{CellFamily::Pyramid, 5, 2}});
  mesh.finalize();
  return mesh;
}

MeshBase make_wedge_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0,
                          1.0, 0.0, 1.0,
                          0.0, 1.0, 1.0},
      std::vector<offset_t>{0, 6},
      std::vector<index_t>{0, 1, 2, 3, 4, 5},
      std::vector<CellShape>{CellShape{CellFamily::Wedge, 6, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_pyramid_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                          1.0, 1.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.5, 0.5, 1.0},
      std::vector<offset_t>{0, 5},
      std::vector<index_t>{0, 1, 2, 3, 4},
      std::vector<CellShape>{CellShape{CellFamily::Pyramid, 5, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_polygon_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                          1.2, 0.6, 0.0,
                          0.5, 1.2, 0.0,
                         -0.2, 0.6, 0.0},
      std::vector<offset_t>{0, 5},
      std::vector<index_t>{0, 1, 2, 3, 4},
      std::vector<CellShape>{CellShape{CellFamily::Polygon, 5, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_concave_polygon_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          2.0, 0.0, 0.0,
                          2.0, 1.0, 0.0,
                          1.0, 1.0, 0.0,
                          1.0, 2.0, 0.0,
                          0.0, 2.0, 0.0},
      std::vector<offset_t>{0, 6},
      std::vector<index_t>{0, 1, 2, 3, 4, 5},
      std::vector<CellShape>{CellShape{CellFamily::Polygon, 6, 1}});
  mesh.finalize();
  return mesh;
}

MeshBase make_polyhedron_cube_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                          1.0, 1.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0,
                          1.0, 0.0, 1.0,
                          1.0, 1.0, 1.0,
                          0.0, 1.0, 1.0},
      std::vector<offset_t>{0, 8},
      std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7},
      std::vector<CellShape>{CellShape{CellFamily::Polyhedron, 8, 1}});
  const std::vector<CellShape> face_shapes(6, CellShape{CellFamily::Quad, 4, 1});
  mesh.set_faces_from_arrays(
      face_shapes,
      std::vector<offset_t>{0, 4, 8, 12, 16, 20, 24},
      std::vector<index_t>{0, 1, 2, 3,
                           4, 5, 6, 7,
                           0, 1, 5, 4,
                           1, 2, 6, 5,
                           2, 3, 7, 6,
                           3, 0, 4, 7},
      std::vector<std::array<index_t, 2>>{
          {{0, INVALID_INDEX}}, {{0, INVALID_INDEX}}, {{0, INVALID_INDEX}},
          {{0, INVALID_INDEX}}, {{0, INVALID_INDEX}}, {{0, INVALID_INDEX}}});
  mesh.finalize();
  return mesh;
}

void expect_linear_side_regions_for_family(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    const char* name)
{
  SCOPED_TRACE(name);
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  const auto topology = reconstruct_cut_topology(mesh, map);
  EXPECT_TRUE(topology.supported);
  EXPECT_FALSE(topology.vertices.empty());

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t parent_measure = -1.0;
  real_t measure_sum = 0.0;
  real_t fraction_sum = 0.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_EQ(region.integration_family, family);
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_GT(region.parent_measure, 0.0);
    EXPECT_GE(region.measure_estimate, -1.0e-12);
    EXPECT_LE(region.measure_estimate, region.parent_measure + 1.0e-9);
    EXPECT_GE(region.volume_fraction_estimate, -1.0e-12);
    EXPECT_LE(region.volume_fraction_estimate, 1.0 + 1.0e-12);
    EXPECT_FALSE(region.integration_region_vertices.empty());
    EXPECT_FALSE(region.integration_region_faces.empty());
    if (region.measure_estimate > 1.0e-12) {
      EXPECT_TRUE(region.closed_integration_topology);
      EXPECT_FALSE(region.integration_vertices.empty());
      EXPECT_FALSE(region.integration_subcells.empty());
      real_t subcell_measure = 0.0;
      for (const auto& subcell : region.integration_subcells) {
        EXPECT_TRUE(subcell.closed_topology);
        EXPECT_FALSE(subcell.vertices.empty());
        EXPECT_FALSE(subcell.faces.empty());
        EXPECT_GT(subcell.measure, 0.0);
        subcell_measure += subcell.measure;
      }
      EXPECT_NEAR(subcell_measure,
                  region.measure_estimate,
                  std::max(1.0e-9, region.parent_measure * 1.0e-8));
    }
    EXPECT_TRUE(std::isfinite(region.centroid_estimate[0]));
    EXPECT_TRUE(std::isfinite(region.centroid_estimate[1]));
    EXPECT_TRUE(std::isfinite(region.centroid_estimate[2]));
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure, parent_measure, std::max(1.0e-10, parent_measure * 1.0e-10));
    }
    measure_sum += region.measure_estimate;
    fraction_sum += region.volume_fraction_estimate;
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-9, parent_measure * 1.0e-9));
  EXPECT_NEAR(fraction_sum, 1.0, 1.0e-9);

  const auto validity = diagnose_cut_topology_validity(topology);
  EXPECT_TRUE(validity.ok) << (validity.messages.empty() ? "" : validity.messages.front());
}

void expect_linear_tessellation_matches_cut_topology(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    const char* name)
{
  SCOPED_TRACE(name);
  const auto tets = PolyhedronTessellation::linear_cell_tets(mesh, 0, Configuration::Reference);
  ASSERT_FALSE(tets.empty());

  real_t tessellated_measure = 0.0;
  for (const auto& tet : tets) {
    tessellated_measure += std::abs(MeshGeometry::tet_volume(
        tet.vertices[0], tet.vertices[1], tet.vertices[2], tet.vertices[3]));
  }
  const real_t parent_measure = MeshGeometry::cell_measure(mesh, 0, Configuration::Reference);
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(tessellated_measure, parent_measure, std::max(1.0e-10, parent_measure * 1.0e-10));

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  const auto topology = reconstruct_cut_topology(mesh, map);
  ASSERT_TRUE(topology.supported);
  ASSERT_EQ(topology.side_regions.size(), 2u);

  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Tetra ||
                  subcell.family == CellFamily::Polyhedron);
      EXPECT_TRUE(subcell.closed_topology);
    }
  }
}

std::vector<std::array<std::uint64_t, 3>> canonical_triangle_vertex_sets_from_subcells(
    const CutSideRegion& region)
{
  std::vector<std::array<std::uint64_t, 3>> out;
  for (const auto& subcell : region.integration_subcells) {
    if (subcell.family != CellFamily::Triangle || subcell.vertices.size() != 3u) {
      continue;
    }
    std::array<std::uint64_t, 3> ids{{subcell.vertices[0], subcell.vertices[1], subcell.vertices[2]}};
    std::sort(ids.begin(), ids.end());
    out.push_back(ids);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::array<std::uint64_t, 3>> canonical_triangle_vertex_sets_from_poly_geometry(
    const CutSideRegion& region)
{
  std::vector<std::array<real_t, 3>> points;
  points.reserve(region.integration_region_vertices.size());
  for (const auto id : region.integration_region_vertices) {
    const auto it = std::find_if(region.integration_vertices.begin(),
                                 region.integration_vertices.end(),
                                 [&](const auto& vertex) {
                                   return vertex.stable_id == id;
                                 });
    if (it == region.integration_vertices.end()) {
      return {};
    }
    points.push_back(it->point);
  }

  std::vector<std::array<index_t, 3>> triangles;
  if (!PolyGeometry::triangulate_planar_polygon(points, triangles)) {
    return {};
  }

  std::vector<std::array<std::uint64_t, 3>> out;
  out.reserve(triangles.size());
  for (const auto& tri : triangles) {
    std::array<std::uint64_t, 3> ids{{
        region.integration_region_vertices[static_cast<std::size_t>(tri[0])],
        region.integration_region_vertices[static_cast<std::size_t>(tri[1])],
        region.integration_region_vertices[static_cast<std::size_t>(tri[2])]}};
    std::sort(ids.begin(), ids.end());
    out.push_back(ids);
  }
  std::sort(out.begin(), out.end());
  return out;
}

CutTopologyVertex cut_topology_vertex(
    std::uint64_t id,
    std::array<real_t, 3> point,
    std::array<real_t, 3> normal = {{0.0, 0.0, 1.0}})
{
  CutTopologyVertex vertex;
  vertex.stable_id = id;
  vertex.point = point;
  vertex.normal = normal;
  vertex.parent_cell = 0;
  vertex.parent_cell_gid = 100;
  vertex.provenance.persistent_id = "manual-curved-validity";
  return vertex;
}

CutIntegrationSubcell closed_subcell(
    std::uint64_t id,
    real_t measure,
    std::array<real_t, 3> centroid = {{0.25, 0.25, 0.25}})
{
  CutIntegrationSubcell subcell;
  subcell.stable_id = id;
  subcell.family = CellFamily::Tetra;
  subcell.vertices = {id + 1, id + 2, id + 3, id + 4};
  subcell.faces = {
      {id + 1, id + 2, id + 3},
      {id + 1, id + 2, id + 4},
      {id + 1, id + 3, id + 4},
      {id + 2, id + 3, id + 4}};
  subcell.measure = measure;
  subcell.centroid = centroid;
  subcell.closed_topology = true;
  subcell.provenance.persistent_id = "manual-curved-validity";
  return subcell;
}

CutSideRegion closed_side_region(
    CutTopologySide side,
    real_t measure,
    real_t parent_measure,
    std::uint64_t id)
{
  CutSideRegion region;
  region.stable_id = id;
  region.side = side;
  region.parent_cell = 0;
  region.parent_cell_gid = 100;
  region.integration_family = CellFamily::Tetra;
  region.parent_measure = parent_measure;
  region.measure_estimate = measure;
  region.volume_fraction_estimate = parent_measure > 0.0 ? measure / parent_measure : 0.0;
  region.centroid_estimate = {{0.25, 0.25, 0.25}};
  region.measure_from_linear_topology = true;
  region.closed_integration_topology = true;
  region.integration_vertices = {
      CutIntegrationVertex{id + 1, {{0.0, 0.0, 0.0}}},
      CutIntegrationVertex{id + 2, {{1.0, 0.0, 0.0}}},
      CutIntegrationVertex{id + 3, {{0.0, 1.0, 0.0}}},
      CutIntegrationVertex{id + 4, {{0.0, 0.0, 1.0}}}};
  region.integration_subcells.push_back(closed_subcell(id + 10, measure));
  region.provenance.persistent_id = "manual-curved-validity";
  return region;
}

bool has_message_containing(
    const std::vector<std::string>& messages,
    const std::string& needle)
{
  return std::any_of(messages.begin(), messages.end(), [&](const auto& message) {
    return message.find(needle) != std::string::npos;
  });
}

std::string joined_messages(const std::vector<std::string>& messages)
{
  std::string out;
  for (const auto& message : messages) {
    if (!out.empty()) {
      out += "; ";
    }
    out += message;
  }
  return out;
}

std::string support_matrix_key(const CutSupportMatrixEntry& entry)
{
  return std::to_string(static_cast<int>(entry.parent_family)) + ":" +
         std::to_string(entry.geometry_order) + ":" +
         std::to_string(static_cast<int>(entry.embedded_kind)) + ":" +
         (entry.distributed ? "mpi" : "serial") + ":" +
         entry.cut_mode + ":" +
         entry.quadrature_policy + ":" +
         entry.conditioning_policy + ":" +
         entry.fe_execution_path;
}

bool is_quadratic_curved_starter_support_row(const CutSupportMatrixEntry& entry)
{
  return entry.geometry_order == 2 &&
         entry.embedded_kind == EmbeddedGeometryKind::Plane &&
         entry.cut_mode == "curved-isoparametric-cut" &&
         entry.quadrature_policy == "curved-topology-subdivision" &&
         entry.conditioning_policy == "geometric-conditioning-hooks" &&
         (entry.parent_family == CellFamily::Line ||
          entry.parent_family == CellFamily::Triangle ||
          entry.parent_family == CellFamily::Quad ||
          entry.parent_family == CellFamily::Tetra ||
          entry.parent_family == CellFamily::Hex);
}

bool is_first_order_linearized_validation_support_row(const CutSupportMatrixEntry& entry)
{
  return entry.geometry_order == 1 &&
         entry.cut_mode == "linearized-cut" &&
         entry.quadrature_policy == "topology-subdivision" &&
         entry.conditioning_policy == "geometric-conditioning-hooks" &&
         (entry.parent_family == CellFamily::Line ||
          entry.parent_family == CellFamily::Triangle ||
          entry.parent_family == CellFamily::Quad ||
          entry.parent_family == CellFamily::Tetra ||
          entry.parent_family == CellFamily::Hex ||
          entry.parent_family == CellFamily::Wedge ||
          entry.parent_family == CellFamily::Pyramid ||
          entry.parent_family == CellFamily::Polygon ||
          entry.parent_family == CellFamily::Polyhedron) &&
         (entry.embedded_kind == EmbeddedGeometryKind::Plane ||
          entry.embedded_kind == EmbeddedGeometryKind::Sphere ||
          entry.embedded_kind == EmbeddedGeometryKind::TriangulatedSurface ||
          entry.embedded_kind == EmbeddedGeometryKind::LevelSetField ||
          entry.embedded_kind == EmbeddedGeometryKind::BooleanComposite);
}

bool is_first_order_linearized_sensitivity_support_row(const CutSupportMatrixEntry& entry)
{
  return is_first_order_linearized_validation_support_row(entry);
}

bool is_second_order_linearized_validation_support_row(const CutSupportMatrixEntry& entry)
{
  return entry.geometry_order == 2 &&
         !entry.distributed &&
         entry.cut_mode == "linearized-cut" &&
         entry.quadrature_policy == "topology-subdivision" &&
         entry.conditioning_policy == "geometric-conditioning-hooks" &&
         (entry.parent_family == CellFamily::Line ||
          entry.parent_family == CellFamily::Triangle ||
          entry.parent_family == CellFamily::Quad ||
          entry.parent_family == CellFamily::Tetra ||
          entry.parent_family == CellFamily::Hex ||
          entry.parent_family == CellFamily::Wedge ||
          entry.parent_family == CellFamily::Pyramid) &&
         (entry.embedded_kind == EmbeddedGeometryKind::Plane ||
          entry.embedded_kind == EmbeddedGeometryKind::Sphere ||
          entry.embedded_kind == EmbeddedGeometryKind::TriangulatedSurface ||
          entry.embedded_kind == EmbeddedGeometryKind::LevelSetField ||
          entry.embedded_kind == EmbeddedGeometryKind::BooleanComposite);
}

bool is_second_order_linearized_sensitivity_support_row(const CutSupportMatrixEntry& entry)
{
  return is_second_order_linearized_validation_support_row(entry);
}

bool curved_topology_sensitivity_family(CellFamily family)
{
  return family == CellFamily::Line ||
         family == CellFamily::Triangle ||
         family == CellFamily::Quad ||
         family == CellFamily::Tetra ||
         family == CellFamily::Hex ||
         family == CellFamily::Wedge ||
         family == CellFamily::Pyramid;
}

bool true_curved_plane_subdivision_family(CellFamily family)
{
  return family == CellFamily::Triangle ||
         family == CellFamily::Quad ||
         family == CellFamily::Hex ||
         family == CellFamily::Wedge ||
         family == CellFamily::Pyramid;
}

bool true_curved_subdivision_sensitivity_kind(EmbeddedGeometryKind kind)
{
  return kind == EmbeddedGeometryKind::Sphere ||
         kind == EmbeddedGeometryKind::SignedDistanceCallback ||
         kind == EmbeddedGeometryKind::LevelSetField ||
         kind == EmbeddedGeometryKind::TriangulatedSurface ||
         kind == EmbeddedGeometryKind::BooleanComposite;
}

bool is_true_curved_topology_sensitivity_support_row(const CutSupportMatrixEntry& entry)
{
  if (entry.geometry_order >= 0 ||
      entry.cut_mode != "curved-isoparametric-cut" ||
      entry.conditioning_policy != "geometric-conditioning-hooks" ||
      !curved_topology_sensitivity_family(entry.parent_family)) {
    return false;
  }
  if (entry.quadrature_policy == "true-curved-arrangement") {
    return entry.embedded_kind == EmbeddedGeometryKind::Plane;
  }
  if (entry.quadrature_policy == "true-curved-subdivision-arrangement") {
    if (entry.embedded_kind == EmbeddedGeometryKind::Plane) {
      return true_curved_plane_subdivision_family(entry.parent_family);
    }
    return true_curved_subdivision_sensitivity_kind(entry.embedded_kind);
  }
  return false;
}

bool is_curved_topology_sensitivity_support_row(const CutSupportMatrixEntry& entry)
{
  return is_quadratic_curved_starter_support_row(entry) ||
         is_true_curved_topology_sensitivity_support_row(entry);
}

std::string support_evidence_domain_key(CutSupportEvidenceDomain domain)
{
  switch (domain) {
    case CutSupportEvidenceDomain::Topology:
      return "topology";
    case CutSupportEvidenceDomain::Quadrature:
      return "quadrature";
    case CutSupportEvidenceDomain::FeExecution:
      return "fe-execution";
    case CutSupportEvidenceDomain::RestartRollback:
      return "restart-rollback";
    case CutSupportEvidenceDomain::Mpi:
      return "mpi";
    case CutSupportEvidenceDomain::Sensitivity:
      return "sensitivity";
    case CutSupportEvidenceDomain::Diagnostic:
      return "diagnostic";
    case CutSupportEvidenceDomain::FullValidation:
      return "full-validation";
  }
  return "unknown";
}

bool has_missing_domain(
    const CutSupportMatrixAuditRecord& record,
    CutSupportEvidenceDomain domain)
{
  return std::find(record.missing_domains.begin(),
                   record.missing_domains.end(),
                   domain) != record.missing_domains.end();
}

MeshBase make_first_order_mesh_for_family(CellFamily family)
{
  switch (family) {
    case CellFamily::Line:
      return make_line_mesh();
    case CellFamily::Triangle:
      return make_triangle_mesh();
    case CellFamily::Quad:
      return make_quad_mesh();
    case CellFamily::Tetra:
      return make_tetra_mesh();
    case CellFamily::Hex:
      return make_hex_mesh();
    case CellFamily::Wedge:
      return make_wedge_mesh();
    case CellFamily::Pyramid:
      return make_pyramid_mesh();
    case CellFamily::Polygon:
      return make_polygon_mesh();
    case CellFamily::Polyhedron:
      return make_polyhedron_cube_mesh();
    default:
      return make_tetra_mesh();
  }
}

MeshBase make_quadratic_mesh_for_family(CellFamily family)
{
  switch (family) {
    case CellFamily::Line:
      return make_quadratic_line_mesh();
    case CellFamily::Triangle:
      return make_quadratic_triangle_mesh();
    case CellFamily::Quad:
      return make_quadratic_quad_mesh();
    case CellFamily::Tetra:
      return make_quadratic_tetra_mesh();
    case CellFamily::Hex:
      return make_quadratic_hex_mesh();
    case CellFamily::Wedge:
      return make_quadratic_wedge_mesh();
    case CellFamily::Pyramid:
      return make_quadratic_pyramid_mesh();
    default:
      return make_quadratic_tetra_mesh();
  }
}

EmbeddedGeometryDescriptor first_order_linearized_nonplane_embedded(
    EmbeddedGeometryKind kind,
    std::uint64_t epoch,
    const std::string& id);

EmbeddedGeometryDescriptor second_order_linearized_embedded(
    CellFamily family,
    EmbeddedGeometryKind kind,
    std::uint64_t epoch,
    const std::string& id)
{
  if (kind == EmbeddedGeometryKind::Plane) {
    switch (family) {
      case CellFamily::Line:
        return axis_plane(0, 0.4, epoch);
      case CellFamily::Quad:
      case CellFamily::Hex:
        return axis_plane(0, 0.0, epoch);
      default:
        return axis_plane(0, 0.25, epoch);
    }
  }

  constexpr std::array<real_t, 3> center{{0.0, 0.0, 0.0}};
  real_t radius =
      family == CellFamily::Hex ? real_t{1.70}
      : family == CellFamily::Quad ? real_t{1.10}
                                   : real_t{0.55};
  if (family == CellFamily::Hex &&
      kind == EmbeddedGeometryKind::TriangulatedSurface) {
    radius = real_t{2.0};
  }
  switch (kind) {
    case EmbeddedGeometryKind::Sphere:
      return sphere(center, radius, epoch, id);
    case EmbeddedGeometryKind::LevelSetField:
      return level_set_sphere(center, radius, epoch, id);
    case EmbeddedGeometryKind::TriangulatedSurface:
      return triangulated_octahedron_sphere(center, radius, epoch, id);
    case EmbeddedGeometryKind::BooleanComposite: {
      auto active = sphere(center, radius, epoch + 1u, id + "-active-sphere");
      auto enclosing =
          sphere(center, radius * real_t{3.0}, epoch + 2u, id + "-enclosing-sphere");
      return boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                               {active, enclosing},
                               epoch,
                               id);
    }
    default:
      return first_order_linearized_nonplane_embedded(kind, epoch, id);
  }
}

std::size_t min_curved_patch_vertices_for_family(CellFamily family)
{
  switch (family) {
    case CellFamily::Line:
      return 1u;
    case CellFamily::Triangle:
    case CellFamily::Quad:
      return 2u;
    default:
      return 3u;
  }
}

EmbeddedGeometryDescriptor first_order_linearized_nonplane_embedded(
    EmbeddedGeometryKind kind,
    std::uint64_t epoch,
    const std::string& id)
{
  constexpr std::array<real_t, 3> center{{0.0, 0.0, 0.0}};
  constexpr real_t radius = 0.55;
  switch (kind) {
    case EmbeddedGeometryKind::Sphere:
      return sphere(center, radius, epoch, id);
    case EmbeddedGeometryKind::LevelSetField:
      return level_set_sphere(center, radius, epoch, id);
    case EmbeddedGeometryKind::TriangulatedSurface:
      return triangulated_octahedron_sphere(center, radius, epoch, id);
    case EmbeddedGeometryKind::BooleanComposite: {
      auto active = sphere(center, radius, epoch + 1u, id + "-active-sphere");
      auto enclosing = sphere(center, 2.0, epoch + 2u, id + "-enclosing-sphere");
      return boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                               {active, enclosing},
                               epoch,
                               id);
    }
    default:
      return sphere(center, radius, epoch, id);
  }
}

void expect_first_order_linearized_nonplane_validation(
    CellFamily family,
    EmbeddedGeometryKind kind,
    std::uint64_t epoch)
{
  SCOPED_TRACE("family=" + std::to_string(static_cast<int>(family)) +
               " kind=" + std::to_string(static_cast<int>(kind)));
  const auto mesh = make_first_order_mesh_for_family(family);
  const auto embedded = first_order_linearized_nonplane_embedded(
      kind,
      epoch,
      "linearized-validation-" + std::to_string(static_cast<int>(family)) +
          "-" + std::to_string(static_cast<int>(kind)));

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);
  EXPECT_EQ(map.embedded_geometry.kind, kind);

  const auto topology = reconstruct_cut_topology(mesh, map);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  ASSERT_FALSE(topology.vertices.empty());
  ASSERT_EQ(topology.side_regions.size(), 2u);

  real_t parent_measure = -1.0;
  real_t measure_sum = 0.0;
  real_t fraction_sum = 0.0;
  std::set<CutTopologySide> observed_sides;
  for (const auto& region : topology.side_regions) {
    EXPECT_EQ(region.integration_family, family);
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    EXPECT_FALSE(region.provenance.persistent_id.empty());
    observed_sides.insert(region.side);
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(std::isfinite(vertex.point[0]));
      EXPECT_TRUE(std::isfinite(vertex.point[1]));
      EXPECT_TRUE(std::isfinite(vertex.point[2]));
      EXPECT_FALSE(vertex.provenance.persistent_id.empty());
    }
    real_t subcell_measure = 0.0;
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_FALSE(subcell.vertices.empty());
      EXPECT_FALSE(subcell.faces.empty());
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_EQ(subcell.construction_policy, "linear-topology-subdivision");
      subcell_measure += subcell.measure;
    }
    EXPECT_NEAR(subcell_measure,
                region.measure_estimate,
                std::max(real_t{1.0e-9}, region.parent_measure * real_t{1.0e-8}));
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(real_t{1.0e-10}, parent_measure * real_t{1.0e-10}));
    }
    measure_sum += region.measure_estimate;
    fraction_sum += region.volume_fraction_estimate;
  }
  EXPECT_NE(observed_sides.find(CutTopologySide::Negative), observed_sides.end());
  EXPECT_NE(observed_sides.find(CutTopologySide::Positive), observed_sides.end());
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum,
              parent_measure,
              std::max(real_t{1.0e-9}, parent_measure * real_t{1.0e-8}));
  EXPECT_NEAR(fraction_sum, 1.0, 1.0e-9);

  const auto validity = diagnose_cut_topology_validity(topology);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);

  const auto restart = make_cut_classification_restart_record(map, topology);
  EXPECT_EQ(restart.embedded_kind, kind);
  EXPECT_EQ(restart.side_regions.size(), topology.side_regions.size());
  EXPECT_EQ(restart.cut_topology_revision, topology.topology_revision);
  if (kind == EmbeddedGeometryKind::BooleanComposite) {
    EXPECT_TRUE(restart.is_composed_region);
    EXPECT_FALSE(restart.composition_children.empty());
  }

  const auto packet = make_distributed_cut_exchange_packet(map, topology);
  EXPECT_FALSE(packet.entities.empty());
  const auto side_entity_count =
      std::count_if(packet.entities.begin(), packet.entities.end(), [](const auto& entity) {
        return entity.kind == CutTopologyEntityKind::SideRegion;
      });
  EXPECT_EQ(side_entity_count, topology.side_regions.size());

  const auto topology_again = reconstruct_cut_topology(mesh, map);
  EXPECT_EQ(topology_again.topology_revision, topology.topology_revision);
}

bool parametric_coordinate_inside(CellFamily family, const std::array<real_t, 3>& xi)
{
  const real_t tol = 1.0e-8;
  switch (family) {
    case CellFamily::Line:
      return xi[0] >= -1.0 - tol && xi[0] <= 1.0 + tol;
    case CellFamily::Triangle:
      return xi[0] >= -tol && xi[1] >= -tol && xi[0] + xi[1] <= 1.0 + tol;
    case CellFamily::Quad:
      return xi[0] >= -1.0 - tol && xi[0] <= 1.0 + tol &&
             xi[1] >= -1.0 - tol && xi[1] <= 1.0 + tol;
    case CellFamily::Tetra:
      return xi[0] >= -tol && xi[1] >= -tol && xi[2] >= -tol &&
             xi[0] + xi[1] + xi[2] <= 1.0 + tol;
    case CellFamily::Hex:
      return xi[0] >= -1.0 - tol && xi[0] <= 1.0 + tol &&
             xi[1] >= -1.0 - tol && xi[1] <= 1.0 + tol &&
             xi[2] >= -1.0 - tol && xi[2] <= 1.0 + tol;
    case CellFamily::Wedge:
      return xi[0] >= -tol && xi[1] >= -tol &&
             xi[0] + xi[1] <= 1.0 + tol &&
             xi[2] >= -1.0 - tol && xi[2] <= 1.0 + tol;
    case CellFamily::Pyramid: {
      const real_t span = 1.0 - xi[2];
      return xi[2] >= -tol && xi[2] <= 1.0 + tol &&
             std::abs(xi[0]) <= span + tol &&
             std::abs(xi[1]) <= span + tol;
    }
    default:
      return std::isfinite(xi[0]) && std::isfinite(xi[1]) && std::isfinite(xi[2]);
  }
}

void expect_curved_patch_descriptors(
    const CutTopologyRecord& topology,
    CellFamily family,
    std::size_t min_vertices_per_patch)
{
  ASSERT_FALSE(topology.curved_patches.empty());
  bool saw_matching_patch = false;
  for (const auto& patch : topology.curved_patches) {
    if (patch.parent_family != family) {
      continue;
    }
    saw_matching_patch = true;
    EXPECT_EQ(patch.geometry_order, 2);
    EXPECT_TRUE(patch.linearized_surrogate);
    EXPECT_FALSE(patch.exact_topology_available);
    EXPECT_EQ(patch.construction_policy, "curved-isoparametric-topology-subdivision");
    EXPECT_TRUE(patch.isoparametric_quadrature_available);
    EXPECT_GT(patch.quadrature_measure, 0.0);
    ASSERT_EQ(patch.quadrature_points.size(), patch.quadrature_normals.size());
    ASSERT_EQ(patch.quadrature_points.size(), patch.quadrature_weights.size());
    ASSERT_EQ(patch.quadrature_points.size(),
              patch.quadrature_parent_parametric_coordinates.size());
    real_t quadrature_weight_sum = 0.0;
    for (const auto weight : patch.quadrature_weights) {
      EXPECT_GT(weight, 0.0);
      quadrature_weight_sum += weight;
    }
    EXPECT_NEAR(quadrature_weight_sum,
                patch.quadrature_measure,
                std::max(1.0e-12, patch.quadrature_measure * 1.0e-12));
    EXPECT_TRUE(patch.parametric_coordinates_valid);
    EXPECT_TRUE(std::isfinite(patch.max_parent_parametric_residual));
    EXPECT_LE(patch.max_parent_parametric_residual, 5.0e-2);
    ASSERT_GE(patch.ordered_vertices.size(), min_vertices_per_patch);
    ASSERT_EQ(patch.ordered_vertices.size(), patch.parent_parametric_coordinates.size());
    ASSERT_EQ(patch.ordered_vertices.size(), patch.physical_points.size());
    for (const auto& xi : patch.parent_parametric_coordinates) {
      EXPECT_TRUE(parametric_coordinate_inside(family, xi));
    }
    for (const auto& xi : patch.quadrature_parent_parametric_coordinates) {
      EXPECT_TRUE(parametric_coordinate_inside(family, xi));
    }
  }
  EXPECT_TRUE(saw_matching_patch);
}

void expect_sensitivity_sample_shape_identities(
    const CutQuadratureGeometrySensitivityRecord& record,
    CellFamily family)
{
  ASSERT_FALSE(record.samples.empty());
  ASSERT_FALSE(record.parent_geometry_dofs.empty());
  for (const auto& sample : record.samples) {
    EXPECT_TRUE(parametric_coordinate_inside(family, sample.parent_parametric_coordinate));
    ASSERT_EQ(sample.shape_values.size(), record.parent_geometry_dofs.size());
    ASSERT_EQ(sample.shape_gradients.size(), record.parent_geometry_dofs.size());
    real_t value_sum = 0.0;
    std::array<real_t, 3> gradient_sum{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < sample.shape_values.size(); ++i) {
      value_sum += sample.shape_values[i];
      gradient_sum[0] += sample.shape_gradients[i][0];
      gradient_sum[1] += sample.shape_gradients[i][1];
      gradient_sum[2] += sample.shape_gradients[i][2];
    }
    EXPECT_NEAR(value_sum, 1.0, 1.0e-11);
    EXPECT_NEAR(gradient_sum[0], 0.0, 1.0e-10);
    EXPECT_NEAR(gradient_sum[1], 0.0, 1.0e-10);
    EXPECT_NEAR(gradient_sum[2], 0.0, 1.0e-10);
  }
}

void expect_curved_topology_sensitivity_metadata(
    const char* name,
    const CutTopologyRecord& topology,
    const MeshBase& mesh,
    CellFamily family,
    int geometry_order,
    EmbeddedGeometryKind embedded_kind,
    const std::string& construction_policy)
{
  SCOPED_TRACE(name);
  ASSERT_FALSE(topology.sensitivity_records.empty()) << joined_messages(topology.diagnostics);

  const auto parent_dofs = mesh.cell_geometry_dofs(0);
  std::set<std::string> target_kinds;
  std::size_t matching_records = 0u;
  for (const auto& record : topology.sensitivity_records) {
    if (record.parent_family != family ||
        record.construction_policy != construction_policy) {
      continue;
    }
    ++matching_records;
    target_kinds.insert(record.target_kind);
    EXPECT_EQ(record.geometry_order, geometry_order);
    EXPECT_EQ(record.embedded_kind, embedded_kind);
    EXPECT_TRUE(record.ad_compatible);
    EXPECT_TRUE(record.location_sensitivity_available);
    EXPECT_TRUE(record.jacobian_sensitivity_available);
    EXPECT_TRUE(record.measure_sensitivity_available);
    EXPECT_TRUE(record.normal_sensitivity_available);
    EXPECT_TRUE(record.quadrature_weight_sensitivity_available);
    EXPECT_TRUE(record.capability_diagnostic.empty());
    EXPECT_EQ(record.parent_geometry_dofs, parent_dofs);
    expect_sensitivity_sample_shape_identities(record, family);
  }
  EXPECT_GE(matching_records, 3u);
  EXPECT_NE(target_kinds.find("interface-quadrature"), target_kinds.end());
  EXPECT_NE(target_kinds.find("negative-volume-subcell-quadrature"), target_kinds.end());
  EXPECT_NE(target_kinds.find("positive-volume-subcell-quadrature"), target_kinds.end());
}

void expect_first_order_linearized_sensitivity_metadata(
    CellFamily family,
    EmbeddedGeometryKind kind,
    std::uint64_t epoch)
{
  const std::string name =
      "linearized-sensitivity-" + std::to_string(static_cast<int>(family)) +
      "-" + std::to_string(static_cast<int>(kind));
  SCOPED_TRACE(name);
  const auto mesh = make_first_order_mesh_for_family(family);
  const auto embedded = first_order_linearized_nonplane_embedded(kind, epoch, name);

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  const auto topology = reconstruct_cut_topology(mesh, map);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  expect_curved_topology_sensitivity_metadata(name.c_str(),
                                             topology,
                                             mesh,
                                             family,
                                             1,
                                             kind,
                                             "linear-topology-subdivision");

  const auto rebuilt = reconstruct_cut_topology(mesh, map);
  ASSERT_TRUE(rebuilt.supported) << joined_messages(rebuilt.diagnostics);
  ASSERT_EQ(rebuilt.sensitivity_records.size(), topology.sensitivity_records.size());
  EXPECT_EQ(rebuilt.topology_revision, topology.topology_revision);
  for (std::size_t i = 0; i < topology.sensitivity_records.size(); ++i) {
    EXPECT_EQ(rebuilt.sensitivity_records[i].stable_id,
              topology.sensitivity_records[i].stable_id);
    EXPECT_EQ(rebuilt.sensitivity_records[i].samples.size(),
              topology.sensitivity_records[i].samples.size());
  }

  const auto parent_dofs = mesh.cell_geometry_dofs(0);
  const auto packet = make_distributed_cut_exchange_packet(map, topology);
  std::size_t sensitivity_entity_count = 0u;
  for (const auto& entity : packet.entities) {
    if (entity.kind != CutTopologyEntityKind::InterfacePolygon &&
        entity.kind != CutTopologyEntityKind::SideRegion) {
      continue;
    }
    if (!entity.geometry_sensitivity_ad_compatible) {
      continue;
    }
    ++sensitivity_entity_count;
    EXPECT_EQ(entity.embedded_kind, kind);
    EXPECT_EQ(entity.geometry_order, 1);
    EXPECT_EQ(entity.construction_policy, "linear-topology-subdivision");
    EXPECT_EQ(entity.geometry_sensitivity_dof_count, parent_dofs.size());
    EXPECT_GT(entity.geometry_sensitivity_sample_count, 0u);
    EXPECT_TRUE(entity.location_sensitivity_available);
    EXPECT_TRUE(entity.jacobian_sensitivity_available);
    EXPECT_TRUE(entity.measure_sensitivity_available);
    EXPECT_TRUE(entity.normal_sensitivity_available);
    EXPECT_TRUE(entity.quadrature_weight_sensitivity_available);
  }
  EXPECT_GE(sensitivity_entity_count, 3u);
}

void expect_quadratic_curved_starter_sensitivity_metadata(
    const char* name,
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family)
{
  SCOPED_TRACE(name);
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  ASSERT_FALSE(topology.sensitivity_records.empty());

  const auto parent_dofs = mesh.cell_geometry_dofs(0);
  std::set<std::string> target_kinds;
  std::size_t matching_records = 0u;
  for (const auto& record : topology.sensitivity_records) {
    if (record.parent_family != family) {
      continue;
    }
    ++matching_records;
    target_kinds.insert(record.target_kind);
    EXPECT_EQ(record.geometry_order, 2);
    EXPECT_EQ(record.embedded_kind, EmbeddedGeometryKind::Plane);
    EXPECT_EQ(record.construction_policy, "curved-isoparametric-topology-subdivision");
    EXPECT_TRUE(record.ad_compatible);
    EXPECT_TRUE(record.location_sensitivity_available);
    EXPECT_TRUE(record.jacobian_sensitivity_available);
    EXPECT_TRUE(record.measure_sensitivity_available);
    EXPECT_TRUE(record.normal_sensitivity_available);
    EXPECT_TRUE(record.quadrature_weight_sensitivity_available);
    EXPECT_TRUE(record.capability_diagnostic.empty());
    EXPECT_EQ(record.parent_geometry_dofs, parent_dofs);
    expect_sensitivity_sample_shape_identities(record, family);
  }
  EXPECT_GE(matching_records, 3u);
  EXPECT_NE(target_kinds.find("interface-quadrature"), target_kinds.end());
  EXPECT_NE(target_kinds.find("negative-volume-subcell-quadrature"), target_kinds.end());
  EXPECT_NE(target_kinds.find("positive-volume-subcell-quadrature"), target_kinds.end());

  const auto rebuilt = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(rebuilt.supported) << joined_messages(rebuilt.diagnostics);
  ASSERT_EQ(rebuilt.sensitivity_records.size(), topology.sensitivity_records.size());
  EXPECT_EQ(rebuilt.topology_revision, topology.topology_revision);
  for (std::size_t i = 0; i < topology.sensitivity_records.size(); ++i) {
    EXPECT_EQ(rebuilt.sensitivity_records[i].stable_id,
              topology.sensitivity_records[i].stable_id);
    EXPECT_EQ(rebuilt.sensitivity_records[i].samples.size(),
              topology.sensitivity_records[i].samples.size());
  }
}

void expect_quadratic_curved_starter_full_validation(
    const char* name,
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    std::size_t min_vertices_per_patch,
    index_t first_mid_edge_dof,
    index_t last_mid_edge_dof)
{
  SCOPED_TRACE(name);
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_TRUE(topology.linearized_cut_mode);
  expect_curved_patch_descriptors(topology, family, min_vertices_per_patch);
  ASSERT_EQ(topology.side_regions.size(), 2u);

  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  int nonzero_regions = 0;
  std::size_t subcell_count = 0u;
  bool saw_mid_edge_geometry_dof = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    ++nonzero_regions;
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(vertex.curved_isoparametric);
      EXPECT_TRUE(vertex.has_parent_parametric_coordinate);
      EXPECT_TRUE(vertex.parent_parametric_coordinate_valid);
      saw_mid_edge_geometry_dof =
          saw_mid_edge_geometry_dof ||
          (vertex.parent_geometry_dof >= first_mid_edge_dof &&
           vertex.parent_geometry_dof <= last_mid_edge_dof);
    }
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
      ++subcell_count;
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }

  EXPECT_EQ(nonzero_regions, 2);
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum,
              parent_measure,
              std::max(1.0e-6, parent_measure * 1.0e-6));
  EXPECT_GT(subcell_count, 1u);
  EXPECT_TRUE(saw_mid_edge_geometry_dof);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_TRUE(validity.requires_curved_geometry_support);

  const auto rebuilt = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(rebuilt.supported) << joined_messages(rebuilt.diagnostics);
  EXPECT_EQ(rebuilt.topology_revision, topology.topology_revision);
  EXPECT_EQ(rebuilt.curved_patches.size(), topology.curved_patches.size());
  EXPECT_EQ(rebuilt.side_regions.size(), topology.side_regions.size());

  topology_options.allow_linearized_high_order_geometry = false;
  const auto rejected = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(rejected.supported);
}

void expect_second_order_linearized_validation_and_sensitivity(
    CellFamily family,
    EmbeddedGeometryKind kind,
    std::uint64_t epoch)
{
  const std::string name =
      "second-order-linearized-" + std::to_string(static_cast<int>(family)) +
      "-" + std::to_string(static_cast<int>(kind));
  SCOPED_TRACE(name);
  const auto mesh = make_quadratic_mesh_for_family(family);
  ASSERT_TRUE(mesh.has_high_order_geometry());
  ASSERT_EQ(mesh.n_cells(), 1u);
  EXPECT_EQ(mesh.geometry_order(0), 2);
  const auto embedded =
      second_order_linearized_embedded(family, kind, epoch, name);

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);
  EXPECT_EQ(map.embedded_geometry.kind, kind);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_TRUE(topology.linearized_cut_mode);
  EXPECT_TRUE(has_message_containing(topology.diagnostics,
                                     "controlled linearized-cut mode"))
      << joined_messages(topology.diagnostics);
  expect_curved_patch_descriptors(
      topology, family, min_curved_patch_vertices_for_family(family));
  ASSERT_EQ(topology.side_regions.size(), 2u);

  const auto parent_dofs = mesh.cell_geometry_dofs(0);
  ASSERT_GT(parent_dofs.size(), mesh.cell_shape(0).num_corners);
  std::set<index_t> high_order_dofs;
  for (std::size_t i = static_cast<std::size_t>(mesh.cell_shape(0).num_corners);
       i < parent_dofs.size();
       ++i) {
    high_order_dofs.insert(parent_dofs[i]);
  }

  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  int nonzero_regions = 0;
  std::size_t subcell_count = 0u;
  bool saw_high_order_geometry_dof = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    ++nonzero_regions;
    EXPECT_EQ(region.integration_family, family);
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(vertex.curved_isoparametric);
      EXPECT_TRUE(vertex.has_parent_parametric_coordinate);
      EXPECT_TRUE(vertex.parent_parametric_coordinate_valid);
      saw_high_order_geometry_dof =
          saw_high_order_geometry_dof ||
          high_order_dofs.find(vertex.parent_geometry_dof) != high_order_dofs.end();
    }
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.construction_policy,
                "curved-isoparametric-topology-subdivision");
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
      ++subcell_count;
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }

  EXPECT_EQ(nonzero_regions, 2);
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum,
              parent_measure,
              std::max(real_t{1.0e-5}, parent_measure * real_t{1.0e-5}));
  EXPECT_GT(subcell_count, 1u);
  EXPECT_TRUE(saw_high_order_geometry_dof);

  expect_curved_topology_sensitivity_metadata(
      name.c_str(),
      topology,
      mesh,
      family,
      2,
      kind,
      "curved-isoparametric-topology-subdivision");

  const auto packet = make_distributed_cut_exchange_packet(map, topology);
  std::size_t sensitivity_entity_count = 0u;
  for (const auto& entity : packet.entities) {
    if (entity.kind != CutTopologyEntityKind::InterfacePolygon &&
        entity.kind != CutTopologyEntityKind::SideRegion) {
      continue;
    }
    if (!entity.geometry_sensitivity_ad_compatible) {
      continue;
    }
    ++sensitivity_entity_count;
    EXPECT_EQ(entity.embedded_kind, kind);
    EXPECT_EQ(entity.geometry_order, 2);
    EXPECT_EQ(entity.construction_policy,
              "curved-isoparametric-topology-subdivision");
    EXPECT_EQ(entity.geometry_sensitivity_dof_count, parent_dofs.size());
    EXPECT_GT(entity.geometry_sensitivity_sample_count, 0u);
    EXPECT_TRUE(entity.location_sensitivity_available);
    EXPECT_TRUE(entity.jacobian_sensitivity_available);
    EXPECT_TRUE(entity.measure_sensitivity_available);
    EXPECT_TRUE(entity.normal_sensitivity_available);
    EXPECT_TRUE(entity.quadrature_weight_sensitivity_available);
  }
  EXPECT_GE(sensitivity_entity_count, 3u);

  const auto rebuilt = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(rebuilt.supported) << joined_messages(rebuilt.diagnostics);
  ASSERT_EQ(rebuilt.sensitivity_records.size(), topology.sensitivity_records.size());
  EXPECT_EQ(rebuilt.topology_revision, topology.topology_revision);
  EXPECT_EQ(rebuilt.curved_patches.size(), topology.curved_patches.size());
  EXPECT_EQ(rebuilt.side_regions.size(), topology.side_regions.size());
  for (std::size_t i = 0; i < topology.sensitivity_records.size(); ++i) {
    EXPECT_EQ(rebuilt.sensitivity_records[i].stable_id,
              topology.sensitivity_records[i].stable_id);
    EXPECT_EQ(rebuilt.sensitivity_records[i].samples.size(),
              topology.sensitivity_records[i].samples.size());
  }

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_TRUE(validity.requires_curved_geometry_support);
}

std::string unique_path(const std::string& suffix)
{
  const auto stamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return (std::filesystem::temp_directory_path() /
          ("svmp_cut_cell_" + std::to_string(static_cast<long long>(stamp)) + suffix))
      .string();
}

} // namespace

TEST(CutCell, ClassifiesStaticPlaneAndSphereCuts)
{
  auto mesh = make_tetra_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = true;

  const auto plane_map = classify_embedded_geometry(mesh, plane(0.5), options);
  ASSERT_EQ(plane_map.cells.size(), 1u);
  EXPECT_EQ(plane_map.cells[0].classification, CutClassification::Cut);
  EXPECT_FALSE(plane_map.cells[0].intersections.empty());
  EXPECT_TRUE(plane_map.valid_for(mesh));

  EmbeddedGeometryDescriptor sphere;
  sphere.kind = EmbeddedGeometryKind::Sphere;
  sphere.origin = {{0.0, 0.0, 0.0}};
  sphere.radius = 0.5;
  sphere.geometry_epoch = 2;
  sphere.provenance.persistent_id = "embedded-sphere";
  const auto sphere_map = classify_embedded_geometry(mesh, sphere, options);
  ASSERT_EQ(sphere_map.cells.size(), 1u);
  EXPECT_EQ(sphere_map.cells[0].classification, CutClassification::Cut);
}

TEST(CutCell, MovingEmbeddedGeometryTransactionCanRollbackAndAccept)
{
  auto mesh = make_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;

  auto map = classify_embedded_geometry(mesh, plane(0.5, 1), options);
  map.accept_trial();
  ASSERT_EQ(map.state, CutClassificationState::Committed);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutClassificationTransaction tx(map);
  tx.stage(classify_embedded_geometry(mesh, plane(2.0, 2), options));
  ASSERT_EQ(map.cells[0].classification, CutClassification::Negative);
  tx.rollback();
  EXPECT_EQ(tx.state(), CutClassificationState::RolledBack);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutClassificationTransaction tx2(map);
  tx2.stage(classify_embedded_geometry(mesh, plane(-1.0, 3), options));
  tx2.accept();
  EXPECT_EQ(map.state, CutClassificationState::Committed);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Positive);
}

TEST(CutCell, KinematicConstraintProvenanceAndRestartMetadataArePreserved)
{
  auto mesh = make_tetra_mesh();
  auto embedded = plane(0.25, 7);

  EmbeddedKinematicConstraint constraint;
  constraint.kind = EmbeddedKinematicConstraintKind::RelationMap;
  constraint.id = "plane-relation";
  constraint.source_geometry_id = "embedded-plane";
  constraint.relation_map_id = "plane-map";
  constraint.constraint_epoch = 9;
  constraint.provenance.persistent_id = "embedded-plane";
  constraint.source_revision = EmbeddedRevisionSnapshot::capture(
      mesh, Configuration::Reference, embedded.geometry_epoch, constraint.constraint_epoch, 3);

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  options.fe_layout_revision = 3;
  options.kinematic_constraints.push_back(constraint);

  auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.kinematic_constraints.size(), 1u);
  EXPECT_EQ(map.kinematic_constraints[0].relation_map_id, "plane-map");
  EXPECT_TRUE(map.valid_for(mesh));

  const auto restart = make_cut_classification_restart_record(map);
  EXPECT_EQ(restart.provenance.persistent_id, "embedded-plane");
  EXPECT_EQ(restart.embedded_geometry_epoch, 7u);
  EXPECT_EQ(restart.embedded_constraint_epoch, 9u);
  EXPECT_EQ(restart.fe_layout_revision, 3u);
  EXPECT_EQ(restart.cut_cell_count, 1u);
}

TEST(CutCell, DegenerateClassificationsCoverVertexEdgeFaceAndAllOnInterface)
{
  EXPECT_EQ(classify_signed_distances({0.0, 1.0, 1.0}, 1.0e-12), CutClassification::Cut);
  EXPECT_EQ(classify_signed_distances({0.0, 0.0, -1.0, 1.0}, 1.0e-12), CutClassification::Cut);
  EXPECT_EQ(classify_signed_distances({0.0, 0.0, 0.0, 1.0}, 1.0e-12), CutClassification::Cut);
  EXPECT_EQ(classify_signed_distances({0.0, 0.0, 0.0}, 1.0e-12), CutClassification::Degenerate);
  EXPECT_EQ(classify_signed_distances({std::nan("")}, 1.0e-12), CutClassification::Degenerate);

  auto mesh = make_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = true;
  options.classify_edges = true;

  auto face_map = classify_embedded_geometry(mesh, plane(0.0), options);
  EXPECT_EQ(face_map.cells[0].classification, CutClassification::Cut);
  EXPECT_NE(std::find_if(face_map.faces.begin(), face_map.faces.end(), [](const auto& face) {
              return face.classification == CutClassification::Degenerate;
            }),
            face_map.faces.end());

  EmbeddedGeometryDescriptor edge_plane;
  edge_plane.kind = EmbeddedGeometryKind::Plane;
  edge_plane.origin = {{0.0, 0.0, 0.0}};
  edge_plane.normal = {{0.0, 1.0, -1.0}};
  edge_plane.geometry_epoch = 4;
  edge_plane.provenance.persistent_id = "edge-plane";
  auto edge_map = classify_embedded_geometry(mesh, edge_plane, options);
  EXPECT_EQ(edge_map.cells[0].classification, CutClassification::Cut);
  EXPECT_NE(std::find_if(edge_map.edges.begin(), edge_map.edges.end(), [](const auto& edge) {
              return edge.classification == CutClassification::Degenerate;
            }),
            edge_map.edges.end());
}

TEST(CutCell, EmbeddedGeometryRegistrySupportsCallbacksFieldsSurfacesAndMultipleActiveCuts)
{
  auto mesh = make_tetra_mesh();

  EmbeddedGeometryDescriptor callback;
  callback.kind = EmbeddedGeometryKind::SignedDistanceCallback;
  callback.geometry_epoch = 10;
  callback.revisions.geometry_epoch = 10;
  callback.provenance.persistent_id = "callback-plane";
  callback.signed_distance_callback = [](const std::array<real_t, 3>& p) { return p[0] - 0.5; };
  callback.normal_callback = [](const std::array<real_t, 3>&) {
    return std::array<real_t, 3>{{1.0, 0.0, 0.0}};
  };

  EmbeddedGeometryDescriptor level_set;
  level_set.kind = EmbeddedGeometryKind::LevelSetField;
  level_set.provenance.persistent_id = "level-set";
  level_set.revisions.field_layout_revision = 2;
  level_set.revisions.field_value_revision = 3;
  level_set.level_set_samples.push_back({{{0.0, 0.0, 0.0}}, -0.25, {{1.0, 0.0, 0.0}}});
  level_set.level_set_samples.push_back({{{1.0, 0.0, 0.0}}, 0.25, {{1.0, 0.0, 0.0}}});

  EmbeddedGeometryDescriptor surface;
  surface.kind = EmbeddedGeometryKind::TriangulatedSurface;
  surface.provenance.persistent_id = "triangulated-surface";
  surface.revisions.source_surface_revision = 5;
  surface.surface_triangles.push_back({{{{{0.25, -1.0, -1.0}},
                                         {{0.25, 2.0, -1.0}},
                                         {{0.25, -1.0, 2.0}}}},
                                      {}});

  EmbeddedGeometryRegistry registry;
  registry.register_geometry(callback);
  registry.register_geometry(level_set);
  registry.register_geometry(surface);
  ASSERT_TRUE(registry.contains("callback-plane"));
  EXPECT_EQ(registry.active_geometry_ids().size(), 3u);

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto maps = registry.classify_active(mesh, options);
  ASSERT_EQ(maps.size(), 3u);
  EXPECT_TRUE(std::all_of(maps.begin(), maps.end(), [](const auto& map) {
    return !map.cells.empty();
  }));
  EXPECT_EQ(registry.snapshot().field_value_revision, 3u);
  EXPECT_EQ(registry.snapshot().source_surface_revision, 5u);
}

TEST(CutCell, BooleanCompositeAndCutTopologyAreDeterministicAndRestartVisible)
{
  auto mesh = make_tetra_mesh();
  auto left = plane(0.5, 6);
  left.provenance.persistent_id = "boolean-plane";
  EmbeddedGeometryDescriptor right;
  right.kind = EmbeddedGeometryKind::Sphere;
  right.origin = {{0.0, 0.0, 0.0}};
  right.radius = 10.0;
  right.geometry_epoch = 7;
  right.provenance.persistent_id = "boolean-enclosing-sphere";
  right.provenance.provenance_epoch = 7;

  EmbeddedGeometryDescriptor band;
  band.kind = EmbeddedGeometryKind::BooleanComposite;
  band.boolean_operation = EmbeddedGeometryBooleanOperation::Intersection;
  band.provenance.persistent_id = "plane-with-enclosing-region";
  band.provenance.name = "plane with enclosing region";
  band.provenance.provenance_epoch = 8;
  band.children = {left, right};
  band.geometry_epoch = 8;

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  auto map = classify_embedded_geometry(mesh, band, options);
  const auto topology_a = reconstruct_cut_topology(mesh, map);
  const auto topology_b = reconstruct_cut_topology(mesh, map);
  EXPECT_TRUE(topology_a.supported);
  EXPECT_EQ(topology_a.topology_revision, topology_b.topology_revision);
  EXPECT_FALSE(topology_a.vertices.empty());
  EXPECT_FALSE(topology_a.side_regions.empty());

  const auto restart = make_cut_classification_restart_record(map, topology_a);
  EXPECT_NE(restart.predicate_policy_key, 0u);
  EXPECT_NE(restart.cut_topology_revision, 0u);
  EXPECT_TRUE(restart.is_composed_region);
  EXPECT_EQ(restart.composition_operation, EmbeddedGeometryBooleanOperation::Intersection);
  ASSERT_EQ(restart.composition_children.size(), 2u);
  EXPECT_EQ(restart.composition_children[0].parent_persistent_id, "plane-with-enclosing-region");
  EXPECT_EQ(restart.composition_children[0].provenance.persistent_id, "boolean-plane");
  EXPECT_EQ(restart.composition_children[1].provenance.persistent_id, "boolean-enclosing-sphere");
  EXPECT_EQ(restart.side_regions.size(), topology_a.side_regions.size());

  const auto diagnostic = diagnose_embedded_geometry_query_support(band);
  EXPECT_TRUE(diagnostic.ok);
}

TEST(CutCell, LinearFamilySideRegionsConserveMeasuresForAdvertisedFamilies)
{
  expect_linear_side_regions_for_family(make_line_mesh(),
                                        axis_plane(0, 0.4, 51),
                                        CellFamily::Line,
                                        "line");
  expect_linear_side_regions_for_family(make_triangle_mesh(),
                                        axis_plane(0, 0.25, 52),
                                        CellFamily::Triangle,
                                        "triangle");
  expect_linear_side_regions_for_family(make_quad_mesh(),
                                        axis_plane(0, 0.5, 53),
                                        CellFamily::Quad,
                                        "quad");
  expect_linear_side_regions_for_family(make_tetra_mesh(),
                                        axis_plane(0, 0.25, 54),
                                        CellFamily::Tetra,
                                        "tetra");
  expect_linear_side_regions_for_family(make_hex_mesh(),
                                        axis_plane(0, 0.5, 55),
                                        CellFamily::Hex,
                                        "hex");
  expect_linear_side_regions_for_family(make_wedge_mesh(),
                                        axis_plane(2, 0.5, 56),
                                        CellFamily::Wedge,
                                        "wedge");
  expect_linear_side_regions_for_family(make_pyramid_mesh(),
                                        axis_plane(2, 0.4, 57),
                                        CellFamily::Pyramid,
                                        "pyramid");
  expect_linear_side_regions_for_family(make_polygon_mesh(),
                                        axis_plane(0, 0.5, 58),
                                        CellFamily::Polygon,
                                        "polygon");
  expect_linear_side_regions_for_family(make_polyhedron_cube_mesh(),
                                        axis_plane(0, 0.5, 59),
                                        CellFamily::Polyhedron,
                                        "polyhedron");
}

TEST(CutCell, FirstOrderLinearizedNonPlaneRowsHaveRowExactValidationEvidence)
{
  const std::array<CellFamily, 9> families{{
      CellFamily::Line,
      CellFamily::Triangle,
      CellFamily::Quad,
      CellFamily::Tetra,
      CellFamily::Hex,
      CellFamily::Wedge,
      CellFamily::Pyramid,
      CellFamily::Polygon,
      CellFamily::Polyhedron}};
  const std::array<EmbeddedGeometryKind, 4> kinds{{
      EmbeddedGeometryKind::Sphere,
      EmbeddedGeometryKind::TriangulatedSurface,
      EmbeddedGeometryKind::LevelSetField,
      EmbeddedGeometryKind::BooleanComposite}};

  std::uint64_t epoch = 1900;
  for (const auto family : families) {
    for (const auto kind : kinds) {
      expect_first_order_linearized_nonplane_validation(family, kind, epoch++);
    }
  }

  const auto records = qualify_cut_support_matrix();
  std::size_t row_exact_validation_rows = 0u;
  std::size_t nonplane_validation_rows = 0u;
  for (const auto& record : records) {
    if (!is_first_order_linearized_validation_support_row(record.entry)) {
      continue;
    }
    ++row_exact_validation_rows;
    EXPECT_TRUE(record.validation_evidence) << support_matrix_key(record.entry);
    EXPECT_FALSE(has_message_containing(record.missing,
                                        "full support-matrix validation"))
        << support_matrix_key(record.entry) << " :: " << joined_messages(record.missing);
    if (record.entry.embedded_kind != EmbeddedGeometryKind::Plane) {
      ++nonplane_validation_rows;
      EXPECT_TRUE(record.sensitivity_evidence) << support_matrix_key(record.entry);
      EXPECT_TRUE(record.qualified) << support_matrix_key(record.entry)
                                    << " :: " << joined_messages(record.missing);
    }
  }
  EXPECT_EQ(row_exact_validation_rows, 2u * 6u * families.size() * 5u);
  EXPECT_EQ(nonplane_validation_rows, 2u * 6u * families.size() * kinds.size());
}

TEST(CutCell, FirstOrderLinearizedNonPlaneRowsHaveAnalyticADSensitivityEvidence)
{
  const std::array<CellFamily, 9> families{{
      CellFamily::Line,
      CellFamily::Triangle,
      CellFamily::Quad,
      CellFamily::Tetra,
      CellFamily::Hex,
      CellFamily::Wedge,
      CellFamily::Pyramid,
      CellFamily::Polygon,
      CellFamily::Polyhedron}};
  const std::array<EmbeddedGeometryKind, 4> kinds{{
      EmbeddedGeometryKind::Sphere,
      EmbeddedGeometryKind::TriangulatedSurface,
      EmbeddedGeometryKind::LevelSetField,
      EmbeddedGeometryKind::BooleanComposite}};

  std::uint64_t epoch = 2010;
  for (const auto family : families) {
    for (const auto kind : kinds) {
      expect_first_order_linearized_sensitivity_metadata(family, kind, epoch++);
    }
  }

  const auto records = qualify_cut_support_matrix();
  std::size_t sensitivity_rows = 0u;
  std::size_t nonplane_sensitivity_rows = 0u;
  for (const auto& record : records) {
    if (!is_first_order_linearized_sensitivity_support_row(record.entry)) {
      continue;
    }
    ++sensitivity_rows;
    EXPECT_TRUE(record.sensitivity_evidence) << support_matrix_key(record.entry);
    EXPECT_TRUE(record.qualified) << support_matrix_key(record.entry)
                                  << " :: " << joined_messages(record.missing);
    if (record.entry.embedded_kind != EmbeddedGeometryKind::Plane) {
      ++nonplane_sensitivity_rows;
    }
  }
  EXPECT_EQ(sensitivity_rows, 2u * 6u * families.size() * 5u);
  EXPECT_EQ(nonplane_sensitivity_rows, 2u * 6u * families.size() * kinds.size());
}

TEST(CutCell, VolumeClosedTopologyUsesSharedLinearCellTessellationUtility)
{
  expect_linear_tessellation_matches_cut_topology(make_tetra_mesh(),
                                                  axis_plane(0, 0.25, 160),
                                                  "tetra");
  expect_linear_tessellation_matches_cut_topology(make_hex_mesh(),
                                                  axis_plane(0, 0.50, 161),
                                                  "hex");
  expect_linear_tessellation_matches_cut_topology(make_wedge_mesh(),
                                                  axis_plane(2, 0.50, 162),
                                                  "wedge");
  expect_linear_tessellation_matches_cut_topology(make_pyramid_mesh(),
                                                  axis_plane(2, 0.40, 163),
                                                  "pyramid");
  expect_linear_tessellation_matches_cut_topology(make_polyhedron_cube_mesh(),
                                                  axis_plane(0, 0.50, 164),
                                                  "polyhedron");
}

TEST(CutCell, ConcavePolygonClosedTopologyMatchesPolyGeometryTriangulation)
{
  const auto mesh = make_concave_polygon_mesh();
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, axis_plane(0, 1.5, 165), options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  const auto topology = reconstruct_cut_topology(mesh, map);
  ASSERT_TRUE(topology.supported);
  ASSERT_EQ(topology.side_regions.size(), 2u);

  bool checked_concave_side = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12 ||
        region.integration_region_vertices.size() < 4u) {
      continue;
    }
    const auto expected = canonical_triangle_vertex_sets_from_poly_geometry(region);
    ASSERT_FALSE(expected.empty());
    const auto actual = canonical_triangle_vertex_sets_from_subcells(region);
    EXPECT_EQ(actual, expected);
    checked_concave_side = checked_concave_side || region.integration_region_vertices.size() > 4u;
  }
  EXPECT_TRUE(checked_concave_side);
}

TEST(CutCell, ClassificationUsesHighOrderGeometryDofsWithLinearizedTopologyGate)
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
         -1.0, 0.0, 0.0},
      std::vector<offset_t>{0, 3},
      std::vector<index_t>{0, 1, 2},
      std::vector<CellShape>{CellShape{CellFamily::Line, 2, 2}});
  mesh.finalize();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, plane(-0.5, 12), options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_TRUE(topology.supported);
  EXPECT_TRUE(topology.linearized_cut_mode);
  ASSERT_FALSE(topology.vertices.empty());
  expect_curved_patch_descriptors(topology, CellFamily::Line, 1u);
  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  bool saw_midpoint_geometry_dof = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(vertex.curved_isoparametric);
      EXPECT_TRUE(vertex.has_parent_parametric_coordinate);
      EXPECT_TRUE(vertex.parent_parametric_coordinate_valid);
      saw_midpoint_geometry_dof =
          saw_midpoint_geometry_dof || vertex.parent_geometry_dof == 1;
    }
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_EQ(subcell.family, CellFamily::Line);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure, parent_measure, std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-8, parent_measure * 1.0e-8));
  EXPECT_TRUE(saw_midpoint_geometry_dof);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_TRUE(validity.requires_curved_geometry_support);

  topology_options.allow_linearized_high_order_geometry = false;
  const auto rejected = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(rejected.supported);
}

TEST(CutCell, TrueCurvedArrangementLineUsesRootIntervalsWithoutLinearizedSurrogate)
{
  const auto mesh = make_cubic_curved_line_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto embedded = plane(0.50, 166);
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_EQ(topology.curved_patches.size(), 1u);

  const auto& patch = topology.curved_patches.front();
  EXPECT_EQ(patch.parent_family, CellFamily::Line);
  EXPECT_EQ(patch.geometry_order, 3);
  EXPECT_TRUE(patch.exact_topology_available);
  EXPECT_FALSE(patch.linearized_surrogate);
  EXPECT_TRUE(patch.parametric_coordinates_valid);
  EXPECT_TRUE(patch.isoparametric_quadrature_available);
  EXPECT_EQ(patch.construction_policy, "true-curved-isoparametric-arrangement");
  ASSERT_EQ(patch.parent_parametric_coordinates.size(), 1u);
  ASSERT_EQ(patch.physical_points.size(), 1u);
  EXPECT_GT(patch.parent_parametric_coordinates.front()[0], -1.0);
  EXPECT_LT(patch.parent_parametric_coordinates.front()[0], 1.0);
  EXPECT_NEAR(embedded.signed_distance(patch.physical_points.front()), 0.0, 1.0e-8);

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    ASSERT_EQ(region.integration_subcells.size(), 1u);
    const auto& subcell = region.integration_subcells.front();
    EXPECT_EQ(subcell.family, CellFamily::Line);
    EXPECT_TRUE(subcell.closed_topology);
    EXPECT_TRUE(subcell.curved_isoparametric);
    EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
    EXPECT_EQ(subcell.construction_policy, "true-curved-isoparametric-arrangement");
    EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
    EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-8, parent_measure * 1.0e-8));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-line-sensitivity",
      topology,
      mesh,
      CellFamily::Line,
      3,
      embedded.kind,
      "true-curved-isoparametric-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_FALSE(validity.requires_curved_geometry_support);
}

void expect_true_curved_face_arrangement(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily expected_family,
    int expected_order)
{
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_EQ(topology.curved_patches.size(), 1u);

  const auto& patch = topology.curved_patches.front();
  EXPECT_EQ(patch.parent_family, expected_family);
  EXPECT_EQ(patch.geometry_order, expected_order);
  EXPECT_TRUE(patch.exact_topology_available);
  EXPECT_FALSE(patch.linearized_surrogate);
  EXPECT_TRUE(patch.parametric_coordinates_valid);
  EXPECT_TRUE(patch.isoparametric_quadrature_available);
  EXPECT_EQ(patch.construction_policy, "true-curved-isoparametric-arrangement");
  EXPECT_EQ(patch.parent_parametric_coordinates.size(), 2u);
  EXPECT_GT(patch.quadrature_points.size(), 2u);
  EXPECT_GT(patch.quadrature_measure, 0.0);

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Triangle ||
                  subcell.family == CellFamily::Quad);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.construction_policy, "true-curved-isoparametric-arrangement");
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-8, parent_measure * 1.0e-8));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-6, parent_measure * 1.0e-6));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-face-sensitivity",
      topology,
      mesh,
      expected_family,
      expected_order,
      embedded.kind,
      "true-curved-isoparametric-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_FALSE(validity.requires_curved_geometry_support);
}

void expect_true_curved_subdivision_arrangement(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily parent_family)
{
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  EXPECT_TRUE(has_message_containing(topology.diagnostics, "subdivision arrangement"))
      << joined_messages(topology.diagnostics);
  ASSERT_FALSE(topology.curved_patches.empty());
  EXPECT_GT(topology.curved_patches.size(), 1u);

  for (const auto& patch : topology.curved_patches) {
    EXPECT_EQ(patch.parent_family, parent_family);
    EXPECT_EQ(patch.geometry_order, 3);
    EXPECT_TRUE(patch.exact_topology_available);
    EXPECT_FALSE(patch.linearized_surrogate);
    EXPECT_TRUE(patch.parametric_coordinates_valid);
    EXPECT_TRUE(patch.isoparametric_quadrature_available);
    EXPECT_EQ(patch.construction_policy, "true-curved-subdivision-arrangement");
    if (parent_family == CellFamily::Line) {
      EXPECT_GE(patch.parent_parametric_coordinates.size(), 1u);
    } else if (parent_family == CellFamily::Triangle || parent_family == CellFamily::Quad) {
      EXPECT_GE(patch.parent_parametric_coordinates.size(), 2u);
    } else {
      EXPECT_GE(patch.parent_parametric_coordinates.size(), 3u);
    }
    EXPECT_EQ(patch.ordered_vertices.size(), patch.parent_parametric_coordinates.size());
    EXPECT_EQ(patch.ordered_vertices.size(), patch.physical_points.size());
    if (embedded.kind == EmbeddedGeometryKind::BooleanComposite) {
      EXPECT_EQ(patch.embedded_kind, EmbeddedGeometryKind::BooleanComposite);
      EXPECT_EQ(patch.composition_operation, embedded.boolean_operation);
      EXPECT_EQ(patch.active_child_ordinals.size(), patch.ordered_vertices.size());
      EXPECT_EQ(patch.active_child_provenance.size(), patch.ordered_vertices.size());
      EXPECT_FALSE(patch.active_child_ordinals.empty());
      EXPECT_TRUE(std::all_of(patch.active_child_provenance.begin(),
                              patch.active_child_provenance.end(),
                              [](const auto& child) {
                                return !child.persistent_id.empty() || !child.name.empty();
                              }));
    }
    EXPECT_GT(patch.quadrature_measure, 0.0);
    EXPECT_TRUE(std::isfinite(patch.max_parent_parametric_residual));
  }

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& subcell : region.integration_subcells) {
      if (parent_family == CellFamily::Line) {
        EXPECT_EQ(subcell.family, CellFamily::Line);
      } else if (parent_family == CellFamily::Triangle || parent_family == CellFamily::Quad) {
        EXPECT_EQ(subcell.family, CellFamily::Triangle);
      } else {
        EXPECT_TRUE(subcell.family == CellFamily::Tetra ||
                    subcell.family == CellFamily::Polyhedron);
      }
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.construction_policy, "true-curved-subdivision-arrangement");
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-5, parent_measure * 1.0e-5));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-4, parent_measure * 1.0e-4));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-subdivision-sensitivity",
      topology,
      mesh,
      parent_family,
      3,
      embedded.kind,
      "true-curved-subdivision-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_FALSE(validity.requires_curved_geometry_support);
}

real_t pi_for_test()
{
  return std::acos(real_t{-1.0});
}

real_t side_measure(const CutTopologyRecord& topology, CutTopologySide side)
{
  real_t measure = 0.0;
  for (const auto& region : topology.side_regions) {
    if (region.side == side) {
      measure += region.measure_estimate;
    }
  }
  return measure;
}

real_t curved_patch_measure(const CutTopologyRecord& topology)
{
  real_t measure = 0.0;
  for (const auto& patch : topology.curved_patches) {
    measure += patch.quadrature_measure;
  }
  return measure;
}

CutTopologyRecord reconstruct_true_curved_topology_for_validation(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    const CutTopologyOptions* topology_options_override = nullptr)
{
  CutClassificationOptions classification_options;
  classification_options.classify_faces = false;
  classification_options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, classification_options);
  EXPECT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells.front().classification, CutClassification::Cut);

  CutTopologyOptions topology_options =
      topology_options_override ? *topology_options_override : CutTopologyOptions{};
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  return reconstruct_cut_topology(mesh, map, topology_options);
}

void expect_affine_hex_plane_cut_matches_analytic_reference(real_t cut_x,
                                                           real_t expected_negative,
                                                           real_t tolerance)
{
  const auto mesh = make_cubic_affine_hex_mesh();
  const auto topology =
      reconstruct_true_curved_topology_for_validation(mesh, axis_plane(0, cut_x, 3100));
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_EQ(topology.curved_patches.size(), 1u);
  EXPECT_EQ(topology.curved_patches.front().construction_policy,
            "true-curved-isoparametric-arrangement");
  EXPECT_TRUE(topology.curved_patches.front().exact_topology_available);
  EXPECT_FALSE(topology.curved_patches.front().linearized_surrogate);

  const real_t expected_positive = real_t{8.0} - expected_negative;
  EXPECT_NEAR(side_measure(topology, CutTopologySide::Negative),
              expected_negative,
              tolerance);
  EXPECT_NEAR(side_measure(topology, CutTopologySide::Positive),
              expected_positive,
              tolerance);
  EXPECT_NEAR(curved_patch_measure(topology), real_t{4.0}, tolerance);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
}

void expect_curved_restart_roundtrip_and_rollback(
    const std::string& case_name,
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& accepted,
    const EmbeddedGeometryDescriptor& trial,
    const CutTopologyOptions& topology_options,
    CellFamily expected_family,
    const std::string& expected_construction_policy,
    bool expected_exact_topology,
    bool expected_linearized_surrogate)
{
  SCOPED_TRACE(case_name);

  CutClassificationOptions classification_options;
  classification_options.classify_faces = false;
  classification_options.classify_edges = false;

  EmbeddedGeometryRegistry registry;
  registry.register_geometry(accepted);
  const auto accepted_maps = registry.classify_active(mesh, classification_options);
  ASSERT_EQ(accepted_maps.size(), 1u);
  ASSERT_EQ(accepted_maps.front().cells.size(), 1u);
  EXPECT_EQ(accepted_maps.front().cells.front().classification, CutClassification::Cut);

  const auto accepted_topology =
      reconstruct_cut_topology(mesh, accepted_maps.front(), topology_options);
  ASSERT_TRUE(accepted_topology.supported) << joined_messages(accepted_topology.diagnostics);
  ASSERT_FALSE(accepted_topology.curved_patches.empty());
  for (const auto& patch : accepted_topology.curved_patches) {
    EXPECT_EQ(patch.parent_family, expected_family);
    EXPECT_EQ(patch.construction_policy, expected_construction_policy);
    EXPECT_EQ(patch.exact_topology_available, expected_exact_topology);
    EXPECT_EQ(patch.linearized_surrogate, expected_linearized_surrogate);
    EXPECT_TRUE(patch.isoparametric_quadrature_available);
    EXPECT_GT(patch.quadrature_measure, 0.0);
  }
  EXPECT_FALSE(accepted_topology.side_regions.empty());

  const auto records = make_embedded_geometry_restart_records(registry);
  ASSERT_EQ(records.size(), 1u);
  const bool requires_callback_reregistration =
      accepted.kind == EmbeddedGeometryKind::SignedDistanceCallback;
  EXPECT_EQ(records.front().requires_application_reregistration,
            requires_callback_reregistration);

  auto restored_registry = restore_embedded_geometry_registry(records);
  if (requires_callback_reregistration) {
    EXPECT_TRUE(restored_registry.active_geometry_ids().empty());
    const auto* restored_callback =
        restored_registry.find(accepted.provenance.persistent_id);
    ASSERT_NE(restored_callback, nullptr);
    EXPECT_FALSE(restored_callback->active);
    restored_registry.register_geometry(accepted);
  }

  const auto restored_maps = restored_registry.classify_active(mesh, classification_options);
  ASSERT_EQ(restored_maps.size(), 1u);
  const auto restored_topology =
      reconstruct_cut_topology(mesh, restored_maps.front(), topology_options);
  ASSERT_TRUE(restored_topology.supported) << joined_messages(restored_topology.diagnostics);
  EXPECT_EQ(restored_topology.topology_revision, accepted_topology.topology_revision);
  EXPECT_EQ(restored_topology.curved_patches.size(), accepted_topology.curved_patches.size());
  EXPECT_EQ(restored_topology.side_regions.size(), accepted_topology.side_regions.size());

  const auto restart = make_cut_classification_restart_record(
      restored_maps.front(), restored_topology);
  EXPECT_EQ(restart.provenance.persistent_id, accepted.provenance.persistent_id);
  EXPECT_EQ(restart.embedded_kind, accepted.kind);
  EXPECT_EQ(restart.cut_topology_revision, restored_topology.topology_revision);
  EXPECT_EQ(restart.side_regions.size(), restored_topology.side_regions.size());

  auto accepted_state = restored_maps.front();
  accepted_state.accept_trial();
  ASSERT_EQ(accepted_state.state, CutClassificationState::Committed);
  CutClassificationTransaction tx(accepted_state);
  tx.stage(classify_embedded_geometry(mesh, trial, classification_options));
  EXPECT_EQ(accepted_state.embedded_geometry.provenance.persistent_id,
            trial.provenance.persistent_id);
  tx.rollback();
  EXPECT_EQ(tx.state(), CutClassificationState::RolledBack);
  EXPECT_EQ(accepted_state.embedded_geometry.provenance.persistent_id,
            accepted.provenance.persistent_id);

  const auto rolled_back_topology =
      reconstruct_cut_topology(mesh, accepted_state, topology_options);
  ASSERT_TRUE(rolled_back_topology.supported)
      << joined_messages(rolled_back_topology.diagnostics);
  EXPECT_EQ(rolled_back_topology.topology_revision, accepted_topology.topology_revision);
  const auto rolled_back_restart =
      make_cut_classification_restart_record(accepted_state, rolled_back_topology);
  EXPECT_EQ(rolled_back_restart.provenance.persistent_id,
            accepted.provenance.persistent_id);
  EXPECT_NE(rolled_back_restart.provenance.persistent_id,
            trial.provenance.persistent_id);
  EXPECT_EQ(rolled_back_restart.cut_topology_revision,
            accepted_topology.topology_revision);
}

EmbeddedGeometryDescriptor renamed(EmbeddedGeometryDescriptor descriptor, std::string id)
{
  descriptor.provenance.persistent_id = std::move(id);
  descriptor.provenance.name = descriptor.provenance.persistent_id;
  return descriptor;
}

TEST(CutCell, CurvedCutRestartRoundtripAndRollbackQualifiesSupportMatrixRows)
{
  CutTopologyOptions starter_options;
  starter_options.allow_linearized_high_order_geometry = true;

  CutTopologyOptions true_options;
  true_options.allow_linearized_high_order_geometry = false;
  true_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;

  expect_curved_restart_roundtrip_and_rollback(
      "quadratic curved-topology subdivision",
      make_quadratic_tetra_mesh(),
      renamed(axis_plane(0, 0.25, 1801), "restart-quadratic-tetra-plane"),
      renamed(axis_plane(0, 0.40, 1802), "restart-quadratic-tetra-plane-trial"),
      starter_options,
      CellFamily::Tetra,
      "curved-isoparametric-topology-subdivision",
      false,
      true);

  expect_curved_restart_roundtrip_and_rollback(
      "arbitrary-order line true arrangement",
      make_cubic_curved_line_mesh(),
      renamed(plane(0.48, 1810), "restart-cubic-line-plane"),
      renamed(plane(0.62, 1811), "restart-cubic-line-plane-trial"),
      true_options,
      CellFamily::Line,
      "true-curved-isoparametric-arrangement",
      true,
      false);

  expect_curved_restart_roundtrip_and_rollback(
      "arbitrary-order tetra graph true arrangement",
      make_cubic_curved_tetra_mesh(),
      renamed(plane(0.25, 1820), "restart-cubic-tetra-plane"),
      renamed(plane(0.35, 1821), "restart-cubic-tetra-plane-trial"),
      true_options,
      CellFamily::Tetra,
      "true-curved-isoparametric-arrangement",
      true,
      false);

  expect_curved_restart_roundtrip_and_rollback(
      "analytic sphere subdivision",
      make_cubic_curved_tetra_mesh(),
      sphere({{0.32, 0.28, 0.22}}, 0.34, 1830, "restart-tetra-sphere"),
      sphere({{0.34, 0.24, 0.20}}, 0.32, 1831, "restart-tetra-sphere-trial"),
      true_options,
      CellFamily::Tetra,
      "true-curved-subdivision-arrangement",
      true,
      false);

  expect_curved_restart_roundtrip_and_rollback(
      "callback sphere subdivision with application re-registration",
      make_cubic_curved_tetra_mesh(),
      callback_sphere({{0.32, 0.28, 0.22}}, 0.34, 1840, "restart-tetra-callback"),
      callback_sphere({{0.34, 0.24, 0.20}}, 0.32, 1841, "restart-tetra-callback-trial"),
      true_options,
      CellFamily::Tetra,
      "true-curved-subdivision-arrangement",
      true,
      false);

  expect_curved_restart_roundtrip_and_rollback(
      "level-set sphere subdivision",
      make_cubic_curved_tetra_mesh(),
      level_set_sphere({{0.32, 0.28, 0.22}}, 0.34, 1850, "restart-tetra-level-set"),
      level_set_sphere({{0.34, 0.24, 0.20}}, 0.32, 1851, "restart-tetra-level-set-trial"),
      true_options,
      CellFamily::Tetra,
      "true-curved-subdivision-arrangement",
      true,
      false);

  expect_curved_restart_roundtrip_and_rollback(
      "triangulated-surface subdivision",
      make_cubic_curved_tetra_mesh(),
      triangulated_octahedron_sphere({{0.32, 0.28, 0.22}},
                                     0.34,
                                     1860,
                                     "restart-tetra-trisurf"),
      triangulated_octahedron_sphere({{0.34, 0.24, 0.20}},
                                     0.32,
                                     1861,
                                     "restart-tetra-trisurf-trial"),
      true_options,
      CellFamily::Tetra,
      "true-curved-subdivision-arrangement",
      true,
      false);

  expect_curved_restart_roundtrip_and_rollback(
      "active-child Boolean subdivision",
      make_cubic_curved_tetra_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.32, 0.28, 0.22}},
                                0.34,
                                1870,
                                "restart-boolean-active-sphere"),
                         sphere({{0.32, 0.28, 0.22}},
                                10.0,
                                1871,
                                "restart-boolean-enclosing-sphere")},
                        1872,
                        "restart-boolean-intersection"),
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.34, 0.24, 0.20}},
                                0.32,
                                1873,
                                "restart-boolean-active-sphere-trial"),
                         sphere({{0.34, 0.24, 0.20}},
                                10.0,
                                1874,
                                "restart-boolean-enclosing-sphere-trial")},
                        1875,
                        "restart-boolean-intersection-trial"),
      true_options,
      CellFamily::Tetra,
      "true-curved-subdivision-arrangement",
      true,
      false);
}

TEST(CutCell, TrueCurvedArrangementTriangleUsesGraphContourWithoutLinearizedSurrogate)
{
  expect_true_curved_face_arrangement(make_cubic_curved_triangle_mesh(),
                                      plane(0.35, 167),
                                      CellFamily::Triangle,
                                      3);
}

TEST(CutCell, TrueCurvedArrangementQuadUsesGraphContourWithoutLinearizedSurrogate)
{
  expect_true_curved_face_arrangement(make_cubic_curved_quad_mesh(),
                                      plane(0.50, 168),
                                      CellFamily::Quad,
                                      3);
}

TEST(CutCell, TrueCurvedArrangementTriangleUsesSubdivisionForNonGraphPlaneCuts)
{
  expect_true_curved_subdivision_arrangement(make_cubic_non_graph_triangle_mesh(),
                                             plane(0.04, 1671),
                                             CellFamily::Triangle);
}

TEST(CutCell, TrueCurvedArrangementQuadUsesSubdivisionForNonGraphPlaneCuts)
{
  expect_true_curved_subdivision_arrangement(make_cubic_non_graph_quad_mesh(),
                                             plane(0.25, 1681),
                                             CellFamily::Quad);
}

TEST(CutCell, TrueCurvedArrangementSphereUsesSubdivisionForNonPlaneCuts)
{
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_line_mesh(),
      sphere({{0.48, 0.0, 0.0}}, 0.22, 1682, "line-sphere"),
      CellFamily::Line);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_triangle_mesh(),
      sphere({{0.35, 0.35, 0.0}}, 0.30, 1683, "triangle-sphere"),
      CellFamily::Triangle);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_quad_mesh(),
      sphere({{0.50, 0.0, 0.0}}, 0.62, 1684, "quad-sphere"),
      CellFamily::Quad);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_tetra_mesh(),
      sphere({{0.32, 0.28, 0.22}}, 0.34, 1685, "tetra-sphere"),
      CellFamily::Tetra);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_hex_mesh(),
      sphere({{0.50, 0.0, 0.0}}, 0.72, 1686, "hex-sphere"),
      CellFamily::Hex);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_wedge_mesh(),
      sphere({{0.50, 0.35, 0.25}}, 0.46, 1687, "wedge-sphere"),
      CellFamily::Wedge);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_pyramid_mesh(),
      sphere({{0.48, 0.0, 0.0}}, 0.42, 1688, "pyramid-sphere"),
      CellFamily::Pyramid);
}

TEST(CutCell, TrueCurvedArrangementCallbackSurfaceUsesSubdivisionForNonPlaneCuts)
{
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_line_mesh(),
      callback_sphere({{0.48, 0.0, 0.0}}, 0.22, 16891, "line-callback-sphere"),
      CellFamily::Line);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_triangle_mesh(),
      callback_sphere({{0.35, 0.35, 0.0}}, 0.30, 16892, "triangle-callback-sphere"),
      CellFamily::Triangle);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_quad_mesh(),
      callback_sphere({{0.50, 0.0, 0.0}}, 0.62, 16893, "quad-callback-sphere"),
      CellFamily::Quad);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_tetra_mesh(),
      callback_sphere({{0.32, 0.28, 0.22}}, 0.34, 1689, "tetra-callback-sphere"),
      CellFamily::Tetra);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_hex_mesh(),
      callback_sphere({{0.50, 0.0, 0.0}}, 0.72, 1690, "hex-callback-sphere"),
      CellFamily::Hex);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_wedge_mesh(),
      callback_sphere({{0.50, 0.35, 0.25}}, 0.46, 16894, "wedge-callback-sphere"),
      CellFamily::Wedge);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_pyramid_mesh(),
      callback_sphere({{0.48, 0.0, 0.0}}, 0.42, 16895, "pyramid-callback-sphere"),
      CellFamily::Pyramid);
}

TEST(CutCell, TrueCurvedArrangementLevelSetFieldUsesSubdivisionForSampledSurfaces)
{
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_line_mesh(),
      level_set_sphere({{0.48, 0.0, 0.0}}, 0.22, 1691, "line-level-set-sphere"),
      CellFamily::Line);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_triangle_mesh(),
      level_set_sphere({{0.35, 0.35, 0.0}}, 0.30, 1692, "triangle-level-set-sphere"),
      CellFamily::Triangle);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_quad_mesh(),
      level_set_sphere({{0.50, 0.0, 0.0}}, 0.62, 1693, "quad-level-set-sphere"),
      CellFamily::Quad);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_tetra_mesh(),
      level_set_sphere({{0.32, 0.28, 0.22}}, 0.34, 1694, "tetra-level-set-sphere"),
      CellFamily::Tetra);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_hex_mesh(),
      level_set_sphere({{0.50, 0.0, 0.0}}, 0.72, 1695, "hex-level-set-sphere"),
      CellFamily::Hex);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_wedge_mesh(),
      level_set_sphere({{0.50, 0.35, 0.25}}, 0.46, 1696, "wedge-level-set-sphere"),
      CellFamily::Wedge);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_pyramid_mesh(),
      level_set_sphere({{0.48, 0.0, 0.0}}, 0.42, 1697, "pyramid-level-set-sphere"),
      CellFamily::Pyramid);
}

TEST(CutCell, TrueCurvedArrangementTriangulatedSurfaceUsesSubdivisionForSurfaceCuts)
{
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_line_mesh(),
      triangulated_octahedron_sphere({{0.48, 0.0, 0.0}}, 0.30, 1698, "line-trisurf-sphere"),
      CellFamily::Line);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_triangle_mesh(),
      triangulated_octahedron_sphere({{0.35, 0.35, 0.0}}, 0.30, 1699, "triangle-trisurf-sphere"),
      CellFamily::Triangle);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_quad_mesh(),
      triangulated_octahedron_sphere({{0.50, 0.0, 0.0}}, 0.62, 1700, "quad-trisurf-sphere"),
      CellFamily::Quad);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_tetra_mesh(),
      triangulated_octahedron_sphere({{0.32, 0.28, 0.22}}, 0.34, 1701, "tetra-trisurf-sphere"),
      CellFamily::Tetra);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_hex_mesh(),
      triangulated_octahedron_sphere({{0.50, 0.0, 0.0}}, 0.95, 1702, "hex-trisurf-sphere"),
      CellFamily::Hex);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_wedge_mesh(),
      triangulated_octahedron_sphere({{0.50, 0.35, 0.25}}, 0.62, 1703, "wedge-trisurf-sphere"),
      CellFamily::Wedge);
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_pyramid_mesh(),
      triangulated_octahedron_sphere({{0.48, 0.0, 0.0}}, 0.58, 1704, "pyramid-trisurf-sphere"),
      CellFamily::Pyramid);
}

TEST(CutCell, TrueCurvedArrangementBooleanCompositeUsesSubdivisionForActiveChildCuts)
{
  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_line_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.48, 0.0, 0.0}}, 0.22, 1705, "boolean-active-line-sphere"),
                         sphere({{0.48, 0.0, 0.0}}, 10.0, 1706, "boolean-active-line-enclosing")},
                        1707,
                        "boolean-active-line-intersection"),
      CellFamily::Line);

  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_triangle_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.35, 0.35, 0.0}}, 0.30, 1708, "boolean-active-triangle-sphere"),
                         sphere({{0.35, 0.35, 0.0}}, 10.0, 1709, "boolean-active-triangle-enclosing")},
                        17100,
                        "boolean-active-triangle-intersection"),
      CellFamily::Triangle);

  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_quad_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.50, 0.0, 0.0}}, 0.62, 17101, "boolean-active-quad-sphere"),
                         sphere({{0.50, 0.0, 0.0}}, 10.0, 17102, "boolean-active-quad-enclosing")},
                        17103,
                        "boolean-active-quad-intersection"),
      CellFamily::Quad);

  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_tetra_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.32, 0.28, 0.22}}, 0.34, 1710, "boolean-active-tetra-sphere"),
                         sphere({{0.32, 0.28, 0.22}}, 10.0, 1711, "boolean-active-tetra-enclosing")},
                        1712,
                        "boolean-active-tetra-intersection"),
      CellFamily::Tetra);

  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_hex_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Union,
                        {sphere({{0.50, 0.0, 0.0}}, 0.72, 1713, "boolean-active-hex-sphere"),
                         sphere({{4.0, 4.0, 4.0}}, 0.25, 1714, "boolean-active-hex-far-sphere")},
                        1715,
                        "boolean-active-hex-union"),
      CellFamily::Hex);

  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_wedge_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Difference,
                        {sphere({{0.50, 0.35, 0.25}}, 10.0, 1716, "boolean-active-wedge-base"),
                         sphere({{0.50, 0.35, 0.25}}, 0.46, 1717, "boolean-active-wedge-cavity")},
                        1718,
                        "boolean-active-wedge-difference"),
      CellFamily::Wedge);

  expect_true_curved_subdivision_arrangement(
      make_cubic_curved_pyramid_mesh(),
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.48, 0.0, 0.0}}, 0.42, 1719, "boolean-active-pyramid-sphere"),
                         sphere({{0.48, 0.0, 0.0}}, 10.0, 1720, "boolean-active-pyramid-enclosing")},
                        1721,
                        "boolean-active-pyramid-intersection"),
      CellFamily::Pyramid);
}

TEST(CutCell, TrueCurvedArrangementTetraUsesGraphSurfaceWithoutLinearizedSurrogate)
{
  const auto mesh = make_cubic_curved_tetra_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto embedded = plane(0.25, 169);
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_EQ(topology.curved_patches.size(), 1u);

  const auto& patch = topology.curved_patches.front();
  EXPECT_EQ(patch.parent_family, CellFamily::Tetra);
  EXPECT_EQ(patch.geometry_order, 3);
  EXPECT_TRUE(patch.exact_topology_available);
  EXPECT_FALSE(patch.linearized_surrogate);
  EXPECT_TRUE(patch.parametric_coordinates_valid);
  EXPECT_TRUE(patch.isoparametric_quadrature_available);
  EXPECT_EQ(patch.construction_policy, "true-curved-isoparametric-arrangement");
  EXPECT_GE(patch.parent_parametric_coordinates.size(), 3u);
  EXPECT_GT(patch.quadrature_points.size(), 8u);
  EXPECT_GT(patch.quadrature_measure, 0.0);

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Tetra ||
                  subcell.family == CellFamily::Polyhedron);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.construction_policy, "true-curved-isoparametric-arrangement");
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-6, parent_measure * 1.0e-6));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-5, parent_measure * 1.0e-5));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-tetra-sensitivity",
      topology,
      mesh,
      CellFamily::Tetra,
      3,
      embedded.kind,
      "true-curved-isoparametric-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_FALSE(validity.requires_curved_geometry_support);
}

TEST(CutCell, TrueCurvedArrangementHexUsesGraphSurfaceWithoutLinearizedSurrogate)
{
  const auto mesh = make_cubic_curved_hex_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto embedded = plane(0.42, 170);
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_EQ(topology.curved_patches.size(), 1u);

  const auto& patch = topology.curved_patches.front();
  EXPECT_EQ(patch.parent_family, CellFamily::Hex);
  EXPECT_EQ(patch.geometry_order, 3);
  EXPECT_TRUE(patch.exact_topology_available);
  EXPECT_FALSE(patch.linearized_surrogate);
  EXPECT_TRUE(patch.parametric_coordinates_valid);
  EXPECT_TRUE(patch.isoparametric_quadrature_available);
  EXPECT_EQ(patch.construction_policy, "true-curved-isoparametric-arrangement");
  EXPECT_GE(patch.parent_parametric_coordinates.size(), 4u);
  EXPECT_GT(patch.quadrature_points.size(), 8u);
  EXPECT_GT(patch.quadrature_measure, 0.0);

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Hex ||
                  subcell.family == CellFamily::Polyhedron);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.construction_policy, "true-curved-isoparametric-arrangement");
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-6, parent_measure * 1.0e-6));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-5, parent_measure * 1.0e-5));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-hex-sensitivity",
      topology,
      mesh,
      CellFamily::Hex,
      3,
      embedded.kind,
      "true-curved-isoparametric-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_FALSE(validity.requires_curved_geometry_support);
}

TEST(CutCell, TrueCurvedArrangementHexUsesSubdivisionForNonGraphPlaneCuts)
{
  expect_true_curved_subdivision_arrangement(make_cubic_non_graph_hex_mesh(),
                                             plane(0.25, 172),
                                             CellFamily::Hex);
}

TEST(CutCell, TrueCurvedArrangementWedgeUsesGraphSurfaceWithoutLinearizedSurrogate)
{
  const auto mesh = make_cubic_curved_wedge_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto embedded = plane(0.42, 173);
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_EQ(topology.curved_patches.size(), 1u);

  const auto& patch = topology.curved_patches.front();
  EXPECT_EQ(patch.parent_family, CellFamily::Wedge);
  EXPECT_EQ(patch.geometry_order, 3);
  EXPECT_TRUE(patch.exact_topology_available);
  EXPECT_FALSE(patch.linearized_surrogate);
  EXPECT_TRUE(patch.parametric_coordinates_valid);
  EXPECT_TRUE(patch.isoparametric_quadrature_available);
  EXPECT_EQ(patch.construction_policy, "true-curved-isoparametric-arrangement");
  EXPECT_GE(patch.parent_parametric_coordinates.size(), 3u);
  EXPECT_GT(patch.quadrature_points.size(), 8u);
  EXPECT_GT(patch.quadrature_measure, 0.0);

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Wedge ||
                  subcell.family == CellFamily::Polyhedron);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.construction_policy, "true-curved-isoparametric-arrangement");
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-6, parent_measure * 1.0e-6));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-5, parent_measure * 1.0e-5));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-wedge-sensitivity",
      topology,
      mesh,
      CellFamily::Wedge,
      3,
      embedded.kind,
      "true-curved-isoparametric-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_FALSE(validity.requires_curved_geometry_support);
}

TEST(CutCell, TrueCurvedArrangementWedgeUsesSubdivisionForNonGraphPlaneCuts)
{
  expect_true_curved_subdivision_arrangement(make_cubic_non_graph_wedge_mesh(),
                                             plane(0.25, 174),
                                             CellFamily::Wedge);
}

TEST(CutCell, TrueCurvedArrangementPyramidUsesGraphSurfaceWithoutLinearizedSurrogate)
{
  const auto mesh = make_cubic_curved_pyramid_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto embedded = plane(0.42, 175);
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_EQ(topology.curved_patches.size(), 1u);

  const auto& patch = topology.curved_patches.front();
  EXPECT_EQ(patch.parent_family, CellFamily::Pyramid);
  EXPECT_EQ(patch.geometry_order, 3);
  EXPECT_TRUE(patch.exact_topology_available);
  EXPECT_FALSE(patch.linearized_surrogate);
  EXPECT_TRUE(patch.parametric_coordinates_valid);
  EXPECT_TRUE(patch.isoparametric_quadrature_available);
  EXPECT_EQ(patch.construction_policy, "true-curved-isoparametric-arrangement");
  EXPECT_GE(patch.parent_parametric_coordinates.size(), 3u);
  EXPECT_GT(patch.quadrature_points.size(), 8u);
  EXPECT_GT(patch.quadrature_measure, 0.0);

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Pyramid ||
                  subcell.family == CellFamily::Polyhedron);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.construction_policy, "true-curved-isoparametric-arrangement");
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-6, parent_measure * 1.0e-6));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-5, parent_measure * 1.0e-5));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-pyramid-sensitivity",
      topology,
      mesh,
      CellFamily::Pyramid,
      3,
      embedded.kind,
      "true-curved-isoparametric-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_FALSE(validity.requires_curved_geometry_support);
}

TEST(CutCell, TrueCurvedArrangementPyramidUsesSubdivisionForNonGraphPlaneCuts)
{
  expect_true_curved_subdivision_arrangement(make_cubic_non_graph_pyramid_mesh(),
                                             plane(0.04, 176),
                                             CellFamily::Pyramid);
}

TEST(CutCell, TrueCurvedAffineHexPlaneCutMatchesAnalyticSlabReference)
{
  expect_affine_hex_plane_cut_matches_analytic_reference(
      /*cut_x=*/real_t{0.25},
      /*expected_negative=*/real_t{5.0},
      /*tolerance=*/real_t{1.0e-10});
}

TEST(CutCell, TrueCurvedAffineHexNearBoundaryPlaneCutMatchesAnalyticSliverReference)
{
  const real_t eps = real_t{1.0e-5};
  expect_affine_hex_plane_cut_matches_analytic_reference(
      /*cut_x=*/real_t{-1.0} + eps,
      /*expected_negative=*/real_t{4.0} * eps,
      /*tolerance=*/real_t{1.0e-8});
}

TEST(CutCell, TrueCurvedAffineHexInteriorSphereCutConvergesAgainstAnalyticReference)
{
  const auto mesh = make_cubic_affine_hex_mesh();
  const real_t radius = real_t{0.65};
  const auto embedded = sphere({{0.0, 0.0, 0.0}},
                               radius,
                               3110,
                               "analytic-interior-sphere-reference");

  CutTopologyOptions coarse_options;
  coarse_options.true_curved_subdivision_refinement_level = 2;
  coarse_options.true_curved_subdivision_max_refinement_level = 2;
  const auto coarse_topology =
      reconstruct_true_curved_topology_for_validation(mesh, embedded, &coarse_options);

  const auto topology =
      reconstruct_true_curved_topology_for_validation(mesh, embedded);

  CutTopologyOptions fine_options;
  fine_options.true_curved_subdivision_refinement_level = 4;
  fine_options.true_curved_subdivision_max_refinement_level = 4;
  const auto fine_topology =
      reconstruct_true_curved_topology_for_validation(mesh, embedded, &fine_options);

  ASSERT_TRUE(coarse_topology.supported) << joined_messages(coarse_topology.diagnostics);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  ASSERT_TRUE(fine_topology.supported) << joined_messages(fine_topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_FALSE(topology.curved_patches.empty());
  EXPECT_TRUE(has_message_containing(topology.diagnostics, "subdivision arrangement"))
      << joined_messages(topology.diagnostics);
  for (const auto& patch : topology.curved_patches) {
    EXPECT_EQ(patch.construction_policy, "true-curved-subdivision-arrangement");
    EXPECT_TRUE(patch.exact_topology_available);
    EXPECT_FALSE(patch.linearized_surrogate);
    EXPECT_TRUE(patch.isoparametric_quadrature_available);
  }

  const real_t expected_volume =
      real_t{4.0} * pi_for_test() * radius * radius * radius / real_t{3.0};
  const real_t expected_area = real_t{4.0} * pi_for_test() * radius * radius;
  const real_t negative_measure = side_measure(topology, CutTopologySide::Negative);
  const real_t interface_measure = curved_patch_measure(topology);
  const real_t relative_volume_error =
      std::abs(negative_measure - expected_volume) / expected_volume;
  const real_t relative_area_error =
      std::abs(interface_measure - expected_area) / expected_area;
  const real_t coarse_relative_volume_error =
      std::abs(side_measure(coarse_topology, CutTopologySide::Negative) - expected_volume) /
      expected_volume;
  const real_t coarse_relative_area_error =
      std::abs(curved_patch_measure(coarse_topology) - expected_area) / expected_area;
  const real_t fine_relative_volume_error =
      std::abs(side_measure(fine_topology, CutTopologySide::Negative) - expected_volume) /
      expected_volume;
  const real_t fine_relative_area_error =
      std::abs(curved_patch_measure(fine_topology) - expected_area) / expected_area;

  RecordProperty("relative_volume_error", relative_volume_error);
  RecordProperty("relative_interface_area_error", relative_area_error);
  RecordProperty("coarse_relative_volume_error", coarse_relative_volume_error);
  RecordProperty("coarse_relative_interface_area_error", coarse_relative_area_error);
  RecordProperty("fine_relative_volume_error", fine_relative_volume_error);
  RecordProperty("fine_relative_interface_area_error", fine_relative_area_error);
  EXPECT_LT(relative_volume_error, coarse_relative_volume_error);
  EXPECT_LT(relative_area_error, coarse_relative_area_error);
  EXPECT_LT(fine_relative_volume_error, relative_volume_error);
  EXPECT_LT(fine_relative_area_error, relative_area_error);
  EXPECT_LE(relative_volume_error, real_t{1.0e-1});
  EXPECT_LE(relative_area_error, real_t{1.0e-1});
  EXPECT_LE(fine_relative_volume_error, real_t{5.0e-2});
  EXPECT_LE(fine_relative_area_error, real_t{5.0e-2});
  EXPECT_NEAR(negative_measure + side_measure(topology, CutTopologySide::Positive),
              real_t{8.0},
              real_t{1.0e-5});

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
}

TEST(CutCell, TrueCurvedArrangementBooleanCompositeRecordsChildChildIntersectionTopology)
{
  const auto mesh = make_cubic_curved_tetra_mesh();
  const auto embedded =
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.25, 0.28, 0.22}},
                                0.34,
                                1722,
                                "boolean-child-child-sphere-a"),
                         sphere({{0.45, 0.28, 0.22}},
                                0.34,
                                1723,
                                "boolean-child-child-sphere-b")},
                        1724,
                        "boolean-child-child-intersection");

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_FALSE(topology.linearized_cut_mode);
  ASSERT_GT(topology.curved_patches.size(), 1u);

  bool saw_child_zero = false;
  bool saw_child_one = false;
  bool saw_mixed_child_patch = false;
  for (const auto& patch : topology.curved_patches) {
    ASSERT_EQ(patch.embedded_kind, EmbeddedGeometryKind::BooleanComposite);
    EXPECT_EQ(patch.composition_operation, EmbeddedGeometryBooleanOperation::Intersection);
    EXPECT_TRUE(patch.exact_topology_available);
    EXPECT_FALSE(patch.linearized_surrogate);
    EXPECT_EQ(patch.construction_policy, "true-curved-subdivision-arrangement");
    ASSERT_EQ(patch.active_child_ordinals.size(), patch.ordered_vertices.size());
    ASSERT_EQ(patch.active_child_provenance.size(), patch.ordered_vertices.size());
    std::set<std::size_t> patch_children;
    for (std::size_t i = 0; i < patch.active_child_ordinals.size(); ++i) {
      patch_children.insert(patch.active_child_ordinals[i]);
      EXPECT_FALSE(patch.active_child_provenance[i].persistent_id.empty());
    }
    saw_child_zero = saw_child_zero || patch_children.count(0u) != 0u;
    saw_child_one = saw_child_one || patch_children.count(1u) != 0u;
    saw_mixed_child_patch = saw_mixed_child_patch || patch_children.size() > 1u;
  }
  EXPECT_TRUE(saw_child_zero);
  EXPECT_TRUE(saw_child_one);
  EXPECT_TRUE(saw_mixed_child_patch);

  ASSERT_EQ(topology.side_regions.size(), 2u);
  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  for (const auto& region : topology.side_regions) {
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_subcells.empty());
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-5, parent_measure * 1.0e-5));
    }
  }
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-4, parent_measure * 1.0e-4));

  expect_curved_topology_sensitivity_metadata(
      "true-curved-boolean-child-child-sensitivity",
      topology,
      mesh,
      CellFamily::Tetra,
      3,
      EmbeddedGeometryKind::BooleanComposite,
      "true-curved-subdivision-arrangement");

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
}

TEST(CutCell, TrueCurvedArrangementBooleanCompositeUsesPredicateFallbackForCoincidentChildren)
{
  const auto mesh = make_quadratic_pyramid_mesh();
  const auto coincident_boolean =
      boolean_composite(EmbeddedGeometryBooleanOperation::Intersection,
                        {sphere({{0.5, 0.5, 0.5}}, 0.55, 1720, "ambiguous-true-sphere-a"),
                         sphere({{0.5, 0.5, 0.5}}, 0.55, 1721, "ambiguous-true-sphere-b")},
                        1722,
                        "ambiguous-true-boolean-sphere");

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map =
      classify_embedded_geometry(mesh, coincident_boolean, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  EXPECT_TRUE(has_message_containing(topology.diagnostics, "near-degeneracy fallback"))
      << joined_messages(topology.diagnostics);
  ASSERT_FALSE(topology.curved_patches.empty());

  bool saw_fallback = false;
  for (const auto& patch : topology.curved_patches) {
    ASSERT_EQ(patch.embedded_kind, EmbeddedGeometryKind::BooleanComposite);
    EXPECT_EQ(patch.composition_operation, EmbeddedGeometryBooleanOperation::Intersection);
    EXPECT_EQ(patch.construction_policy, "true-curved-subdivision-arrangement");
    EXPECT_TRUE(patch.exact_topology_available);
    EXPECT_FALSE(patch.linearized_surrogate);
    ASSERT_EQ(patch.active_child_ordinals.size(), patch.ordered_vertices.size());
    ASSERT_EQ(patch.active_child_provenance.size(), patch.ordered_vertices.size());
    EXPECT_TRUE(patch.predicate_fallback_used);
    EXPECT_TRUE(patch.predicate_fallback_tolerance_resolved);
    EXPECT_EQ(patch.predicate_fallback_policy, "tolerance-resolved-active-child");
    EXPECT_TRUE(patch.predicate_fallback_reason.find("Boolean active-child distances") !=
                std::string::npos);
    EXPECT_FALSE(patch.predicate_fallback_child_ordinals.empty());
    EXPECT_NE(std::find(patch.predicate_fallback_child_ordinals.begin(),
                        patch.predicate_fallback_child_ordinals.end(),
                        0u),
              patch.predicate_fallback_child_ordinals.end());
    EXPECT_NE(std::find(patch.predicate_fallback_child_ordinals.begin(),
                        patch.predicate_fallback_child_ordinals.end(),
                        1u),
              patch.predicate_fallback_child_ordinals.end());
    saw_fallback = saw_fallback || patch.predicate_fallback_used;
  }
  EXPECT_TRUE(saw_fallback);

  const auto packet = make_distributed_cut_exchange_packet(map, topology);
  const auto fallback_entity =
      std::find_if(packet.entities.begin(), packet.entities.end(), [](const auto& entity) {
        return entity.kind == CutTopologyEntityKind::InterfacePolygon &&
               entity.embedded_kind == EmbeddedGeometryKind::BooleanComposite &&
               entity.predicate_fallback_used;
      });
  ASSERT_NE(fallback_entity, packet.entities.end());
  EXPECT_TRUE(fallback_entity->predicate_fallback_tolerance_resolved);
  EXPECT_EQ(fallback_entity->predicate_fallback_policy,
            "tolerance-resolved-active-child");
  EXPECT_FALSE(fallback_entity->predicate_fallback_child_ordinals.empty());
  EXPECT_FALSE(fallback_entity->active_child_provenance_ids.empty());

  const auto topology_b = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_EQ(topology.topology_revision, topology_b.topology_revision);
  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
}

TEST(CutCell, TrueCurvedArrangementRejectsMalformedLevelSetAndTriangulatedSurface)
{
  const auto mesh = make_cubic_curved_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;

  EmbeddedGeometryDescriptor bad_level_set;
  bad_level_set.kind = EmbeddedGeometryKind::LevelSetField;
  bad_level_set.geometry_epoch = 1705;
  bad_level_set.revisions.geometry_epoch = 1705;
  bad_level_set.revisions.field_layout_revision = 1;
  bad_level_set.revisions.field_value_revision = 1705;
  bad_level_set.provenance.persistent_id = "bad-level-set-no-gradient";
  bad_level_set.provenance.name = bad_level_set.provenance.persistent_id;
  bad_level_set.level_set_samples.push_back({{{0.0, 0.0, 0.0}}, -0.25, {{0.0, 0.0, 0.0}}});
  bad_level_set.level_set_samples.push_back({{{1.0, 0.0, 0.0}},  0.25, {{0.0, 0.0, 0.0}}});

  auto map = classify_embedded_geometry(mesh, bad_level_set, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = false;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(topology.supported);
  EXPECT_TRUE(has_message_containing(topology.diagnostics, "level-set descriptors require finite samples"));

  EmbeddedGeometryDescriptor bad_surface;
  bad_surface.kind = EmbeddedGeometryKind::TriangulatedSurface;
  bad_surface.geometry_epoch = 1706;
  bad_surface.revisions.geometry_epoch = 1706;
  bad_surface.revisions.source_surface_revision = 1706;
  bad_surface.provenance.persistent_id = "bad-triangulated-surface-degenerate";
  bad_surface.provenance.name = bad_surface.provenance.persistent_id;
  bad_surface.surface_triangles.push_back({{{{{0.25, 0.0, 0.0}},
                                             {{0.25, 0.0, 0.0}},
                                             {{0.25, 0.0, 0.0}}}},
                                          bad_surface.provenance});

  map = classify_embedded_geometry(mesh, bad_surface, options);
  ASSERT_EQ(map.cells.size(), 1u);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);
  topology = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(topology.supported);
  EXPECT_TRUE(has_message_containing(topology.diagnostics, "triangulated surfaces require finite nondegenerate triangles"));
}

TEST(CutCell, QuadraticCurvedStarterRowsHaveFullValidationEvidence)
{
  expect_quadratic_curved_starter_full_validation(
      "quadratic line",
      make_quadratic_line_mesh(),
      axis_plane(0, 0.4, 165),
      CellFamily::Line,
      1u,
      2,
      2);
  expect_quadratic_curved_starter_full_validation(
      "quadratic triangle",
      make_quadratic_triangle_mesh(),
      axis_plane(0, 0.25, 166),
      CellFamily::Triangle,
      2u,
      3,
      5);
  expect_quadratic_curved_starter_full_validation(
      "quadratic quad",
      make_quadratic_quad_mesh(),
      axis_plane(0, 0.0, 168),
      CellFamily::Quad,
      2u,
      4,
      7);
  expect_quadratic_curved_starter_full_validation(
      "quadratic tetra",
      make_quadratic_tetra_mesh(),
      axis_plane(0, 0.25, 167),
      CellFamily::Tetra,
      3u,
      4,
      9);
  expect_quadratic_curved_starter_full_validation(
      "quadratic hex",
      make_quadratic_hex_mesh(),
      axis_plane(0, 0.0, 169),
      CellFamily::Hex,
      3u,
      8,
      19);
}

TEST(CutCell, QuadraticCurvedStarterRowsHaveAnalyticADSensitivityEvidence)
{
  expect_quadratic_curved_starter_sensitivity_metadata(
      "quadratic line",
      make_quadratic_line_mesh(),
      axis_plane(0, 0.4, 265),
      CellFamily::Line);
  expect_quadratic_curved_starter_sensitivity_metadata(
      "quadratic triangle",
      make_quadratic_triangle_mesh(),
      axis_plane(0, 0.25, 266),
      CellFamily::Triangle);
  expect_quadratic_curved_starter_sensitivity_metadata(
      "quadratic quad",
      make_quadratic_quad_mesh(),
      axis_plane(0, 0.0, 268),
      CellFamily::Quad);
  expect_quadratic_curved_starter_sensitivity_metadata(
      "quadratic tetra",
      make_quadratic_tetra_mesh(),
      axis_plane(0, 0.25, 267),
      CellFamily::Tetra);
  expect_quadratic_curved_starter_sensitivity_metadata(
      "quadratic hex",
      make_quadratic_hex_mesh(),
      axis_plane(0, 0.0, 269),
      CellFamily::Hex);
}

TEST(CutCell, SecondOrderLinearizedRowsHaveFullValidationAndAnalyticADSensitivityEvidence)
{
  const std::array<CellFamily, 7> families{{
      CellFamily::Line,
      CellFamily::Triangle,
      CellFamily::Quad,
      CellFamily::Tetra,
      CellFamily::Hex,
      CellFamily::Wedge,
      CellFamily::Pyramid}};
  const std::array<EmbeddedGeometryKind, 5> kinds{{
      EmbeddedGeometryKind::Plane,
      EmbeddedGeometryKind::Sphere,
      EmbeddedGeometryKind::TriangulatedSurface,
      EmbeddedGeometryKind::LevelSetField,
      EmbeddedGeometryKind::BooleanComposite}};

  std::uint64_t epoch = 2800;
  for (const auto family : families) {
    for (const auto kind : kinds) {
      expect_second_order_linearized_validation_and_sensitivity(
          family, kind, epoch++);
    }
  }

  const auto records = qualify_cut_support_matrix();
  std::size_t full_validation_rows = 0u;
  std::size_t sensitivity_rows = 0u;
  for (const auto& record : records) {
    if (!is_second_order_linearized_validation_support_row(record.entry)) {
      continue;
    }
    ++full_validation_rows;
    EXPECT_FALSE(record.entry.distributed) << support_matrix_key(record.entry);
    EXPECT_TRUE(record.validation_evidence) << support_matrix_key(record.entry);
    EXPECT_TRUE(record.sensitivity_evidence) << support_matrix_key(record.entry);
    EXPECT_TRUE(record.qualified) << support_matrix_key(record.entry)
                                  << " :: " << joined_messages(record.missing);
    if (is_second_order_linearized_sensitivity_support_row(record.entry)) {
      ++sensitivity_rows;
    }
  }
  EXPECT_EQ(full_validation_rows, 6u * families.size() * kinds.size());
  EXPECT_EQ(sensitivity_rows, full_validation_rows);
}

TEST(CutCell, HighOrderTriangleUsesTessellatedClosedTopologyWithGeometryDofProvenance)
{
  const auto mesh = make_quadratic_triangle_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, axis_plane(0, 0.25, 166), options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported);
  EXPECT_TRUE(topology.linearized_cut_mode);
  expect_curved_patch_descriptors(topology, CellFamily::Triangle, 2u);
  ASSERT_EQ(topology.side_regions.size(), 2u);

  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  int nonzero_regions = 0;
  std::size_t triangle_subcells = 0u;
  bool saw_mid_edge_geometry_dof = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    ++nonzero_regions;
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(vertex.curved_isoparametric);
      EXPECT_TRUE(vertex.has_parent_parametric_coordinate);
      EXPECT_TRUE(vertex.parent_parametric_coordinate_valid);
      saw_mid_edge_geometry_dof =
          saw_mid_edge_geometry_dof ||
          (vertex.parent_geometry_dof >= 3 && vertex.parent_geometry_dof <= 5);
    }
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_EQ(subcell.family, CellFamily::Triangle);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
      ++triangle_subcells;
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }

  EXPECT_EQ(nonzero_regions, 2);
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-8, parent_measure * 1.0e-8));
  EXPECT_GT(triangle_subcells, 1u);
  EXPECT_TRUE(saw_mid_edge_geometry_dof);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_TRUE(validity.requires_curved_geometry_support);

  topology_options.allow_linearized_high_order_geometry = false;
  const auto rejected = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(rejected.supported);
}

TEST(CutCell, HighOrderTetraUsesTessellatedVolumeTopologyWithGeometryDofProvenance)
{
  const auto mesh = make_quadratic_tetra_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, axis_plane(0, 0.25, 167), options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported);
  EXPECT_TRUE(topology.linearized_cut_mode);
  expect_curved_patch_descriptors(topology, CellFamily::Tetra, 3u);
  ASSERT_EQ(topology.side_regions.size(), 2u);

  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  int nonzero_regions = 0;
  std::size_t volume_subcells = 0u;
  bool saw_mid_edge_geometry_dof = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    ++nonzero_regions;
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(vertex.curved_isoparametric);
      EXPECT_TRUE(vertex.has_parent_parametric_coordinate);
      EXPECT_TRUE(vertex.parent_parametric_coordinate_valid);
      saw_mid_edge_geometry_dof =
          saw_mid_edge_geometry_dof ||
          (vertex.parent_geometry_dof >= 4 && vertex.parent_geometry_dof <= 9);
    }
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Tetra ||
                  subcell.family == CellFamily::Polyhedron);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
      ++volume_subcells;
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }

  EXPECT_EQ(nonzero_regions, 2);
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-8, parent_measure * 1.0e-8));
  EXPECT_GT(volume_subcells, 1u);
  EXPECT_TRUE(saw_mid_edge_geometry_dof);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_TRUE(validity.requires_curved_geometry_support);

  topology_options.allow_linearized_high_order_geometry = false;
  const auto rejected = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(rejected.supported);
}

TEST(CutCell, HighOrderQuadUsesCurvedIsoparametricTopologyWithGeometryDofProvenance)
{
  const auto mesh = make_quadratic_quad_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, axis_plane(0, 0.0, 168), options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported);
  EXPECT_TRUE(topology.linearized_cut_mode);
  expect_curved_patch_descriptors(topology, CellFamily::Quad, 2u);
  ASSERT_EQ(topology.side_regions.size(), 2u);

  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  int nonzero_regions = 0;
  std::size_t triangle_subcells = 0u;
  bool saw_mid_edge_geometry_dof = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    ++nonzero_regions;
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(vertex.curved_isoparametric);
      EXPECT_TRUE(vertex.has_parent_parametric_coordinate);
      EXPECT_TRUE(vertex.parent_parametric_coordinate_valid);
      saw_mid_edge_geometry_dof =
          saw_mid_edge_geometry_dof ||
          (vertex.parent_geometry_dof >= 4 && vertex.parent_geometry_dof <= 7);
    }
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_EQ(subcell.family, CellFamily::Triangle);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
      ++triangle_subcells;
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }

  EXPECT_EQ(nonzero_regions, 2);
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-6, parent_measure * 1.0e-6));
  EXPECT_GT(triangle_subcells, 1u);
  EXPECT_TRUE(saw_mid_edge_geometry_dof);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_TRUE(validity.requires_curved_geometry_support);

  topology_options.allow_linearized_high_order_geometry = false;
  const auto rejected = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(rejected.supported);
}

TEST(CutCell, HighOrderHexUsesCurvedIsoparametricTopologyWithGeometryDofProvenance)
{
  const auto mesh = make_quadratic_hex_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, axis_plane(0, 0.0, 169), options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutTopologyOptions topology_options;
  topology_options.allow_linearized_high_order_geometry = true;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported);
  EXPECT_TRUE(topology.linearized_cut_mode);
  expect_curved_patch_descriptors(topology, CellFamily::Hex, 3u);
  ASSERT_EQ(topology.side_regions.size(), 2u);

  real_t measure_sum = 0.0;
  real_t parent_measure = -1.0;
  int nonzero_regions = 0;
  std::size_t volume_subcells = 0u;
  bool saw_mid_edge_geometry_dof = false;
  for (const auto& region : topology.side_regions) {
    if (region.measure_estimate <= 1.0e-12) {
      continue;
    }
    ++nonzero_regions;
    EXPECT_TRUE(region.measure_from_linear_topology);
    EXPECT_TRUE(region.closed_integration_topology);
    EXPECT_TRUE(region.curved_isoparametric_topology);
    EXPECT_FALSE(region.integration_vertices.empty());
    EXPECT_FALSE(region.integration_subcells.empty());
    for (const auto& vertex : region.integration_vertices) {
      EXPECT_TRUE(vertex.curved_isoparametric);
      EXPECT_TRUE(vertex.has_parent_parametric_coordinate);
      EXPECT_TRUE(vertex.parent_parametric_coordinate_valid);
      saw_mid_edge_geometry_dof =
          saw_mid_edge_geometry_dof ||
          (vertex.parent_geometry_dof >= 8 && vertex.parent_geometry_dof <= 19);
    }
    for (const auto& subcell : region.integration_subcells) {
      EXPECT_TRUE(subcell.family == CellFamily::Tetra ||
                  subcell.family == CellFamily::Polyhedron);
      EXPECT_TRUE(subcell.closed_topology);
      EXPECT_GT(subcell.measure, 0.0);
      EXPECT_TRUE(subcell.curved_isoparametric);
      EXPECT_TRUE(subcell.measure_from_isoparametric_quadrature);
      EXPECT_EQ(subcell.parent_parametric_vertices.size(), subcell.vertices.size());
      EXPECT_GT(subcell.parent_parametric_measure, 0.0);
      ++volume_subcells;
    }
    measure_sum += region.measure_estimate;
    if (parent_measure < 0.0) {
      parent_measure = region.parent_measure;
    } else {
      EXPECT_NEAR(region.parent_measure,
                  parent_measure,
                  std::max(1.0e-10, parent_measure * 1.0e-10));
    }
  }

  EXPECT_EQ(nonzero_regions, 2);
  ASSERT_GT(parent_measure, 0.0);
  EXPECT_NEAR(measure_sum, parent_measure, std::max(1.0e-6, parent_measure * 1.0e-6));
  EXPECT_GT(volume_subcells, 1u);
  EXPECT_TRUE(saw_mid_edge_geometry_dof);

  const auto validity = diagnose_cut_topology_validity(
      topology, /*high_order_parent_geometry=*/true);
  EXPECT_TRUE(validity.ok) << joined_messages(validity.messages);
  EXPECT_TRUE(validity.requires_curved_geometry_support);

  topology_options.allow_linearized_high_order_geometry = false;
  const auto rejected = reconstruct_cut_topology(mesh, map, topology_options);
  EXPECT_FALSE(rejected.supported);
}

TEST(CutCell, AsciiSTLSurfaceAndRegistryRestartRecordsRoundtripDeterministically)
{
  const auto path = unique_path(".stl");
  {
    std::ofstream out(path);
    out << "solid cut\n";
    out << "facet normal 1 0 0\nouter loop\n";
    out << "vertex 0.25 -1 -1\n";
    out << "vertex 0.25 2 -1\n";
    out << "vertex 0.25 -1 2\n";
    out << "endloop\nendfacet\nendsolid cut\n";
  }

  auto surface = read_ascii_stl_embedded_surface(path, "imported-surface", Configuration::Reference, 17);
  ASSERT_EQ(surface.surface_triangles.size(), 1u);
  EXPECT_EQ(surface.revisions.source_surface_revision, 17u);
  EXPECT_TRUE(surface.diagnose_query_support().ok);

  EmbeddedGeometryRegistry registry;
  registry.register_geometry(surface);
  const auto records = make_embedded_geometry_restart_records(registry);
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records[0].surface_triangles.size(), 1u);

  auto restored = restore_embedded_geometry_registry(records);
  ASSERT_TRUE(restored.contains("imported-surface"));
  auto mesh = make_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto maps = restored.classify_active(mesh, options);
  ASSERT_EQ(maps.size(), 1u);
  EXPECT_EQ(maps[0].cells[0].classification, CutClassification::Cut);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(CutCell, OperationDiagnosticsReportPredicateRevisionAndNearDegenerateFailures)
{
  auto mesh = make_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = true;
  options.classify_edges = true;
  options.fe_layout_revision = 23;
  options.predicate_policy.name = "phase26-robust";
  options.predicate_policy.robust.intersection_tolerance = 1.0e-10;

  EmbeddedGeometryDescriptor bad;
  bad.kind = EmbeddedGeometryKind::SignedDistanceCallback;
  bad.provenance.persistent_id = "bad-callback";
  bad.signed_distance_callback = [](const std::array<real_t, 3>&) {
    return std::nan("");
  };
  bad.normal_callback = [](const std::array<real_t, 3>&) {
    return std::array<real_t, 3>{{1.0, 0.0, 0.0}};
  };

  const auto map = classify_embedded_geometry(mesh, bad, options);
  const auto diagnostic = diagnose_cut_operation(map, "near-degenerate classification");
  EXPECT_FALSE(diagnostic.ok);
  EXPECT_EQ(diagnostic.fe_layout_revision, 23u);
  EXPECT_EQ(diagnostic.predicate_policy_key, options.predicate_policy.revision_key());
  EXPECT_EQ(diagnostic.mesh_and_embedded_revision.fe_layout_revision, 23u);
  EXPECT_FALSE(diagnostic.messages.empty());
}

TEST(CutCell, RobustPredicateCoverageIncludesNearCoplanarTangentVertexEdgeAndDuplicates)
{
  auto mesh = make_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = true;
  options.classify_edges = true;
  options.predicate_policy.robust.intersection_tolerance = 1.0e-8;

  const auto near_vertex = classify_embedded_geometry(mesh, plane(1.0e-10, 40), options);
  EXPECT_EQ(near_vertex.cells[0].classification, CutClassification::Cut);
  const auto near_vertex_diagnostic =
      diagnose_cut_operation(near_vertex, "adaptive predicate coverage");
  EXPECT_TRUE(has_message_containing(near_vertex_diagnostic.messages,
                                     "domain=signed-distance-side"))
      << joined_messages(near_vertex_diagnostic.messages);
  EXPECT_TRUE(has_message_containing(near_vertex_diagnostic.messages,
                                     "domain=edge-intersection"))
      << joined_messages(near_vertex_diagnostic.messages);
  EXPECT_TRUE(has_message_containing(near_vertex_diagnostic.messages,
                                     "adaptive/exact predicate coverage domains"))
      << joined_messages(near_vertex_diagnostic.messages);
  EXPECT_NE(std::find_if(near_vertex.edges.begin(), near_vertex.edges.end(), [](const auto& edge) {
              return edge.classification == CutClassification::Degenerate ||
                     !edge.intersections.empty();
            }),
            near_vertex.edges.end());

  EmbeddedGeometryDescriptor near_edge;
  near_edge.kind = EmbeddedGeometryKind::Plane;
  near_edge.origin = {{0.0, 0.0, 0.0}};
  near_edge.normal = {{0.0, 1.0, -1.0e-12}};
  near_edge.geometry_epoch = 41;
  near_edge.provenance.persistent_id = "near-edge-plane";
  const auto near_edge_map = classify_embedded_geometry(mesh, near_edge, options);
  EXPECT_EQ(near_edge_map.cells[0].classification, CutClassification::Cut);

  EmbeddedGeometryDescriptor tangent_sphere;
  tangent_sphere.kind = EmbeddedGeometryKind::Sphere;
  tangent_sphere.origin = {{2.0, 0.0, 0.0}};
  tangent_sphere.radius = 1.0;
  tangent_sphere.geometry_epoch = 42;
  tangent_sphere.provenance.persistent_id = "near-tangent-sphere";
  const auto tangent_map = classify_embedded_geometry(mesh, tangent_sphere, options);
  EXPECT_NE(tangent_map.cells[0].classification, CutClassification::Positive);

  auto topology = reconstruct_cut_topology(mesh, near_vertex);
  ASSERT_FALSE(topology.vertices.empty());
  topology.vertices.push_back(topology.vertices.front());
  const auto packet = make_distributed_cut_exchange_packet(near_vertex, topology);
  auto duplicated = packet;
  duplicated.entities.push_back(packet.entities.front());
  const auto deduped = deduplicate_cut_exchange_packet(std::move(duplicated));
  EXPECT_EQ(deduped.entities.size(), packet.entities.size());

  CutTopologyRecord degenerate_topology;
  degenerate_topology.interface_polygons.push_back({});
  const auto validity = diagnose_cut_topology_validity(degenerate_topology);
  EXPECT_FALSE(validity.ok);
  EXPECT_TRUE(validity.has_degenerate_polygon);
}

TEST(CutCell, CurvedTopologyValidityRejectsFoldedAndDegenerateInterfaceGeometry)
{
  CutTopologyRecord folded;
  folded.linearized_cut_mode = false;
  folded.vertices = {
      cut_topology_vertex(1, {{0.0, 0.0, 0.0}}),
      cut_topology_vertex(2, {{1.0, 1.0, 0.0}}),
      cut_topology_vertex(3, {{0.0, 1.0, 0.0}}),
      cut_topology_vertex(4, {{1.0, 0.0, 0.0}})};
  CutInterfacePolygon folded_polygon;
  folded_polygon.stable_id = 10;
  folded_polygon.parent_cell = 0;
  folded_polygon.parent_cell_gid = 100;
  folded_polygon.ordered_vertices = {1, 2, 3, 4};
  folded_polygon.normal = {{0.0, 0.0, 1.0}};
  folded.interface_polygons.push_back(folded_polygon);

  const auto folded_diagnostic = diagnose_cut_topology_validity(folded);
  EXPECT_FALSE(folded_diagnostic.ok);
  EXPECT_TRUE(folded_diagnostic.has_folded_interface);
  EXPECT_TRUE(has_message_containing(folded_diagnostic.messages,
                                     "domain=orientation-2d"))
      << joined_messages(folded_diagnostic.messages);

  CutTopologyRecord near_coplanar;
  near_coplanar.linearized_cut_mode = false;
  near_coplanar.vertices = {
      cut_topology_vertex(1, {{0.0, 0.0, 0.0}}),
      cut_topology_vertex(2, {{1.0, 0.0, 0.0}}),
      cut_topology_vertex(3, {{1.0, 1.0, 1.0e-13}}),
      cut_topology_vertex(4, {{0.0, 1.0, 0.0}})};
  CutInterfacePolygon near_coplanar_polygon;
  near_coplanar_polygon.stable_id = 14;
  near_coplanar_polygon.parent_cell = 0;
  near_coplanar_polygon.parent_cell_gid = 100;
  near_coplanar_polygon.ordered_vertices = {1, 2, 3, 4};
  near_coplanar_polygon.normal = {{0.0, 0.0, 1.0}};
  near_coplanar.interface_polygons.push_back(near_coplanar_polygon);

  const auto near_coplanar_diagnostic =
      diagnose_cut_topology_validity(near_coplanar);
  EXPECT_TRUE(has_message_containing(near_coplanar_diagnostic.messages,
                                     "domain=coplanarity-3d"))
      << joined_messages(near_coplanar_diagnostic.messages);

  CutTopologyRecord duplicated;
  duplicated.linearized_cut_mode = false;
  duplicated.vertices = {
      cut_topology_vertex(1, {{0.0, 0.0, 0.0}}),
      cut_topology_vertex(2, {{1.0, 0.0, 0.0}}),
      cut_topology_vertex(3, {{0.0, 1.0, 0.0}})};
  CutInterfacePolygon duplicated_polygon;
  duplicated_polygon.stable_id = 11;
  duplicated_polygon.parent_cell = 0;
  duplicated_polygon.parent_cell_gid = 100;
  duplicated_polygon.ordered_vertices = {1, 2, 99};
  duplicated_polygon.normal = {{0.0, 0.0, 1.0}};
  duplicated.interface_polygons.push_back(duplicated_polygon);

  const auto duplicated_diagnostic = diagnose_cut_topology_validity(duplicated);
  EXPECT_FALSE(duplicated_diagnostic.ok);
  EXPECT_TRUE(duplicated_diagnostic.has_degenerate_intersection);
  EXPECT_TRUE(duplicated_diagnostic.has_degenerate_polygon);

  CutTopologyRecord near_duplicate;
  near_duplicate.linearized_cut_mode = false;
  near_duplicate.vertices = {
      cut_topology_vertex(1, {{0.0, 0.0, 0.0}}),
      cut_topology_vertex(2, {{1.0, 0.0, 0.0}}),
      cut_topology_vertex(3, {{1.0, 1.0, 0.0}}),
      cut_topology_vertex(4, {{1.0 + 1.0e-13, 1.0, 0.0}}),
      cut_topology_vertex(5, {{0.0, 1.0, 0.0}})};
  CutInterfacePolygon near_duplicate_polygon;
  near_duplicate_polygon.stable_id = 13;
  near_duplicate_polygon.parent_cell = 0;
  near_duplicate_polygon.parent_cell_gid = 100;
  near_duplicate_polygon.ordered_vertices = {1, 2, 3, 4, 5};
  near_duplicate_polygon.normal = {{0.0, 0.0, 1.0}};
  near_duplicate.interface_polygons.push_back(near_duplicate_polygon);

  const auto near_duplicate_diagnostic =
      diagnose_cut_topology_validity(near_duplicate);
  EXPECT_FALSE(near_duplicate_diagnostic.ok);
  EXPECT_TRUE(near_duplicate_diagnostic.has_degenerate_intersection);
  EXPECT_TRUE(near_duplicate_diagnostic.has_degenerate_polygon);
  EXPECT_TRUE(has_message_containing(near_duplicate_diagnostic.messages,
                                     "domain=duplicate-detection"))
      << joined_messages(near_duplicate_diagnostic.messages);

  CutTopologyRecord nonfinite;
  nonfinite.linearized_cut_mode = false;
  nonfinite.vertices = {
      cut_topology_vertex(1, {{0.0, 0.0, 0.0}}),
      cut_topology_vertex(2, {{1.0, 0.0, 0.0}}),
      cut_topology_vertex(3, {{std::nan(""), 1.0, 0.0}})};
  CutInterfacePolygon nonfinite_polygon;
  nonfinite_polygon.stable_id = 12;
  nonfinite_polygon.parent_cell = 0;
  nonfinite_polygon.parent_cell_gid = 100;
  nonfinite_polygon.ordered_vertices = {1, 2, 3};
  nonfinite_polygon.normal = {{0.0, 0.0, 1.0}};
  nonfinite.interface_polygons.push_back(nonfinite_polygon);

  const auto nonfinite_diagnostic = diagnose_cut_topology_validity(nonfinite);
  EXPECT_FALSE(nonfinite_diagnostic.ok);
  EXPECT_TRUE(nonfinite_diagnostic.has_nonfinite_geometry);
}

TEST(CutCell, CurvedTopologyValidityRejectsSliversAndInconsistentSideRegions)
{
  CutTopologyRecord sliver;
  sliver.linearized_cut_mode = false;
  sliver.side_regions.push_back(
      closed_side_region(CutTopologySide::Negative, 1.0e-12, 1.0, 100));
  sliver.side_regions.push_back(
      closed_side_region(CutTopologySide::Positive, 1.0 - 1.0e-12, 1.0, 200));

  const auto rejected_sliver = diagnose_cut_topology_validity(sliver);
  EXPECT_FALSE(rejected_sliver.ok);
  EXPECT_TRUE(rejected_sliver.has_curved_sliver);

  CutCurvedValidityPolicy warning_policy;
  warning_policy.reject_slivers = false;
  const auto warning_sliver = diagnose_cut_topology_validity(sliver, false, warning_policy);
  EXPECT_TRUE(warning_sliver.ok);
  EXPECT_TRUE(warning_sliver.has_curved_sliver);

  CutTopologyRecord imbalanced;
  imbalanced.linearized_cut_mode = false;
  imbalanced.side_regions.push_back(
      closed_side_region(CutTopologySide::Negative, 0.4, 1.0, 300));
  imbalanced.side_regions.push_back(
      closed_side_region(CutTopologySide::Positive, 0.4, 1.0, 400));

  const auto imbalanced_diagnostic = diagnose_cut_topology_validity(imbalanced);
  EXPECT_FALSE(imbalanced_diagnostic.ok);
  EXPECT_TRUE(imbalanced_diagnostic.has_inconsistent_side_region);
}

TEST(CutCell, BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit)
{
  EmbeddedGeometryDescriptor a = plane(0.75, 1);
  a.provenance.persistent_id = "left-halfspace";
  EmbeddedGeometryDescriptor b = plane(0.25, 2);
  b.provenance.persistent_id = "right-halfspace";

  EmbeddedGeometryDescriptor region;
  region.kind = EmbeddedGeometryKind::BooleanComposite;
  region.boolean_operation = EmbeddedGeometryBooleanOperation::Union;
  region.provenance.persistent_id = "overlapping-union";
  region.children = {a, b};

  const auto diagnostic = diagnose_boolean_region_composition(
      region,
      {{{0.0, 0.0, 0.0}}, {{0.5, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}});
  EXPECT_TRUE(diagnostic.ok);
  EXPECT_FALSE(diagnostic.messages.empty());

  const auto supported = evaluate_cut_support(CellFamily::Tetra,
                                              1,
                                              EmbeddedGeometryKind::Plane,
                                              false,
                                              "linearized-cut",
                                              "topology-subdivision");
  EXPECT_EQ(supported.status, CutSupportStatus::ImplementedUnqualified);
  EXPECT_EQ(supported.conditioning_policy, "geometric-conditioning-hooks");
  EXPECT_FALSE(supported.fe_execution_path.empty());

  const std::array<std::string, 6> execution_paths{
      "standard-assembly",
      "matrix-free",
      "forms-interpreter",
      "ad",
      "symbolic-tangent",
      "jit"};
  for (const auto& execution_path : execution_paths) {
    const auto exact_path_support = evaluate_cut_support(
        CellFamily::Tetra,
        1,
        EmbeddedGeometryKind::Plane,
        false,
        "linearized-cut",
        "topology-subdivision",
        "geometric-conditioning-hooks",
        execution_path);
    EXPECT_EQ(exact_path_support.status, CutSupportStatus::ImplementedUnqualified);
    EXPECT_EQ(exact_path_support.fe_execution_path, execution_path);
  }

  const auto unsupported_execution_path = evaluate_cut_support(
      CellFamily::Tetra,
      1,
      EmbeddedGeometryKind::Plane,
      false,
      "linearized-cut",
      "topology-subdivision",
      "geometric-conditioning-hooks",
      "unregistered-backend");
  EXPECT_EQ(unsupported_execution_path.status, CutSupportStatus::Unsupported);

  for (const auto family : {CellFamily::Wedge,
                            CellFamily::Pyramid,
                            CellFamily::Polygon,
                            CellFamily::Polyhedron}) {
    const auto family_supported = evaluate_cut_support(family,
                                                       1,
                                                       EmbeddedGeometryKind::Plane,
                                                       false,
                                                       "linearized-cut",
                                                       "topology-subdivision");
    EXPECT_EQ(family_supported.status, CutSupportStatus::ImplementedUnqualified);
  }

  const auto unsupported = evaluate_cut_support(CellFamily::Pyramid,
                                                2,
                                                EmbeddedGeometryKind::Sphere,
                                                false,
                                                "curved-isoparametric-cut",
                                                "moment-fitted");
  EXPECT_EQ(unsupported.status, CutSupportStatus::Unsupported);

  const auto high_order_linearized = evaluate_cut_support(CellFamily::Tetra,
                                                          2,
                                                          EmbeddedGeometryKind::Plane,
                                                          false,
                                                          "linearized-cut",
                                                          "topology-subdivision");
  EXPECT_EQ(high_order_linearized.status, CutSupportStatus::ImplementedUnqualified);
  EXPECT_NE(high_order_linearized.qualification.find("controlled high-order linearized"),
            std::string::npos);

  for (const auto family : {CellFamily::Line, CellFamily::Triangle, CellFamily::Quad,
                            CellFamily::Tetra, CellFamily::Hex}) {
    const auto curved_quadratic = evaluate_cut_support(family,
                                                       2,
                                                       EmbeddedGeometryKind::Plane,
                                                       false,
                                                       "curved-isoparametric-cut",
                                                       "curved-topology-subdivision");
    EXPECT_EQ(curved_quadratic.status, CutSupportStatus::ImplementedUnqualified);
    EXPECT_NE(curved_quadratic.qualification.find("quadratic starter path"),
              std::string::npos);
  }

  const auto true_curved_line = evaluate_cut_support(CellFamily::Line,
                                                     3,
                                                     EmbeddedGeometryKind::Plane,
                                                     false,
                                                     "curved-isoparametric-cut",
                                                     "true-curved-arrangement");
  EXPECT_EQ(true_curved_line.status, CutSupportStatus::ImplementedUnqualified);
  EXPECT_NE(true_curved_line.qualification.find("arbitrary-order high-order line/plane cuts"),
            std::string::npos);
  const auto true_curved_line_higher_order = evaluate_cut_support(
      CellFamily::Line,
      7,
      EmbeddedGeometryKind::Plane,
      false,
      "curved-isoparametric-cut",
      "true-curved-arrangement");
  EXPECT_EQ(true_curved_line_higher_order.status, CutSupportStatus::ImplementedUnqualified);
  const auto true_curved_linear_line = evaluate_cut_support(CellFamily::Line,
                                                            1,
                                                            EmbeddedGeometryKind::Plane,
                                                            false,
                                                            "curved-isoparametric-cut",
                                                            "true-curved-arrangement");
  EXPECT_EQ(true_curved_linear_line.status, CutSupportStatus::Unsupported);
  for (const auto family : {CellFamily::Triangle, CellFamily::Quad}) {
    const auto true_curved_face = evaluate_cut_support(family,
                                                       5,
                                                       EmbeddedGeometryKind::Plane,
                                                       false,
                                                       "curved-isoparametric-cut",
                                                       "true-curved-arrangement");
    EXPECT_EQ(true_curved_face.status, CutSupportStatus::ImplementedUnqualified);
    EXPECT_NE(true_curved_face.qualification.find("graph-compatible arbitrary-order"),
              std::string::npos);
  }
  const auto true_curved_tetra = evaluate_cut_support(CellFamily::Tetra,
                                                      5,
                                                      EmbeddedGeometryKind::Plane,
                                                      false,
                                                      "curved-isoparametric-cut",
                                                      "true-curved-arrangement");
  EXPECT_EQ(true_curved_tetra.status, CutSupportStatus::ImplementedUnqualified);
  EXPECT_NE(true_curved_tetra.qualification.find("high-order tetra/plane cuts"),
            std::string::npos);
  const auto true_curved_hex = evaluate_cut_support(CellFamily::Hex,
                                                    3,
                                                    EmbeddedGeometryKind::Plane,
                                                    false,
                                                    "curved-isoparametric-cut",
                                                    "true-curved-arrangement");
  EXPECT_EQ(true_curved_hex.status, CutSupportStatus::ImplementedUnqualified);
  EXPECT_NE(true_curved_hex.qualification.find("high-order hex/plane cuts"),
            std::string::npos);
  const auto true_curved_wedge = evaluate_cut_support(CellFamily::Wedge,
                                                      3,
                                                      EmbeddedGeometryKind::Plane,
                                                      false,
                                                      "curved-isoparametric-cut",
                                                      "true-curved-arrangement");
  EXPECT_EQ(true_curved_wedge.status, CutSupportStatus::ImplementedUnqualified);
  EXPECT_NE(true_curved_wedge.qualification.find("high-order wedge/plane cuts"),
            std::string::npos);

  const auto true_curved_pyramid = evaluate_cut_support(CellFamily::Pyramid,
                                                        3,
                                                        EmbeddedGeometryKind::Plane,
                                                        false,
                                                        "curved-isoparametric-cut",
                                                        "true-curved-arrangement");
  EXPECT_EQ(true_curved_pyramid.status, CutSupportStatus::ImplementedUnqualified);
  EXPECT_NE(true_curved_pyramid.qualification.find("high-order pyramid/plane cuts"),
            std::string::npos);

  for (const auto family : {CellFamily::Triangle,
                            CellFamily::Quad,
                            CellFamily::Hex,
                            CellFamily::Wedge,
                            CellFamily::Pyramid}) {
    const auto true_curved_subdivision = evaluate_cut_support(
        family,
        4,
        EmbeddedGeometryKind::Plane,
        false,
        "curved-isoparametric-cut",
        "true-curved-subdivision-arrangement");
    EXPECT_EQ(true_curved_subdivision.status, CutSupportStatus::ImplementedUnqualified);
    EXPECT_NE(true_curved_subdivision.qualification.find("non-graph high-order"),
              std::string::npos);
  }

  for (const auto family : {CellFamily::Line,
                            CellFamily::Triangle,
                            CellFamily::Quad,
                            CellFamily::Tetra,
                            CellFamily::Hex,
                            CellFamily::Wedge,
                            CellFamily::Pyramid}) {
    for (const auto kind : {EmbeddedGeometryKind::Sphere,
                            EmbeddedGeometryKind::SignedDistanceCallback,
                            EmbeddedGeometryKind::LevelSetField,
                            EmbeddedGeometryKind::TriangulatedSurface,
                            EmbeddedGeometryKind::BooleanComposite}) {
      const auto non_plane_subdivision = evaluate_cut_support(
          family,
          4,
          kind,
          false,
          "curved-isoparametric-cut",
          "true-curved-subdivision-arrangement");
      EXPECT_EQ(non_plane_subdivision.status, CutSupportStatus::ImplementedUnqualified);
      EXPECT_NE(non_plane_subdivision.qualification.find(
                    "non-plane signed-distance, level-set, triangulated-surface, and Boolean composite"),
                std::string::npos);
    }
  }

  const auto high_order_polyhedron = evaluate_cut_support(CellFamily::Polyhedron,
                                                          2,
                                                          EmbeddedGeometryKind::Plane,
                                                          false,
                                                          "linearized-cut",
                                                          "topology-subdivision");
  EXPECT_EQ(high_order_polyhedron.status, CutSupportStatus::Unsupported);
}

TEST(CutCell, SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps)
{
  const auto entries = cut_support_matrix();
  const auto records = qualify_cut_support_matrix();
  ASSERT_FALSE(entries.empty());
  ASSERT_EQ(records.size(), entries.size());

  std::set<std::string> row_keys;
  std::size_t implemented_rows = 0u;
  std::size_t unsupported_rows = 0u;
  std::size_t distributed_rows = 0u;
  std::size_t curved_rows = 0u;
  std::size_t restart_gap_rows = 0u;
  std::size_t sensitivity_gap_rows = 0u;
  std::size_t validation_gap_rows = 0u;
  std::size_t validation_evidence_rows = 0u;
  std::size_t qualified_rows = 0u;
  std::size_t unsupported_qualified_rows = 0u;

  for (const auto& record : records) {
    const auto key = support_matrix_key(record.entry);
    EXPECT_TRUE(row_keys.insert(key).second) << key;
    EXPECT_TRUE(record.diagnostic_evidence) << key;
    EXPECT_FALSE(record.evidence.empty()) << key;

    const bool implemented = record.entry.status != CutSupportStatus::Unsupported;
    if (!implemented) {
      ++unsupported_rows;
      EXPECT_TRUE(record.qualified) << key << " :: " << joined_messages(record.missing);
      ++unsupported_qualified_rows;
      continue;
    }

    ++implemented_rows;
    EXPECT_TRUE(record.requires_topology_evidence) << key;
    EXPECT_TRUE(record.requires_quadrature_evidence) << key;
    EXPECT_TRUE(record.requires_fe_execution_evidence) << key;
    EXPECT_TRUE(record.requires_restart_rollback_evidence) << key;
    EXPECT_TRUE(record.requires_sensitivity_evidence) << key;
    EXPECT_TRUE(record.requires_diagnostic_evidence) << key;
    EXPECT_TRUE(record.requires_validation_evidence) << key;
    EXPECT_TRUE(record.topology_evidence) << key;
    EXPECT_TRUE(record.quadrature_evidence) << key;
    EXPECT_TRUE(record.fe_execution_evidence) << key;

    if (record.validation_evidence) {
      ++validation_evidence_rows;
    }

    if (record.entry.distributed) {
      ++distributed_rows;
      EXPECT_TRUE(record.requires_mpi_evidence) << key;
      EXPECT_TRUE(record.mpi_evidence) << key;
    }

    if (is_first_order_linearized_sensitivity_support_row(record.entry) ||
        is_second_order_linearized_sensitivity_support_row(record.entry)) {
      EXPECT_TRUE(record.sensitivity_evidence) << key;
    } else if (record.entry.cut_mode == "curved-isoparametric-cut") {
      ++curved_rows;
      EXPECT_TRUE(record.restart_rollback_evidence) << key;
      if (is_curved_topology_sensitivity_support_row(record.entry)) {
        EXPECT_TRUE(record.sensitivity_evidence) << key;
      } else {
        EXPECT_FALSE(record.sensitivity_evidence) << key;
      }
    }

    if (has_message_containing(record.missing, "restart/rollback")) {
      ++restart_gap_rows;
    }
    if (has_message_containing(record.missing, "cut-sensitivity")) {
      ++sensitivity_gap_rows;
    }
    if (has_message_containing(record.missing, "full support-matrix validation")) {
      ++validation_gap_rows;
    }

    if (record.qualified) {
      ++qualified_rows;
      EXPECT_TRUE(record.validation_evidence) << key;
      EXPECT_TRUE(record.missing.empty()) << key;
    } else {
      EXPECT_FALSE(record.missing.empty()) << key;
    }
  }

  EXPECT_GT(implemented_rows, 0u);
  EXPECT_GT(unsupported_rows, 0u);
  EXPECT_GT(distributed_rows, 0u);
  EXPECT_GT(curved_rows, 0u);
  EXPECT_EQ(unsupported_qualified_rows, unsupported_rows);
  EXPECT_EQ(restart_gap_rows, 0u);
  EXPECT_EQ(sensitivity_gap_rows, 0u);
  EXPECT_EQ(validation_gap_rows, 0u);
  EXPECT_GT(validation_evidence_rows, 0u);
  EXPECT_EQ(validation_evidence_rows, implemented_rows);
  EXPECT_EQ(qualified_rows, implemented_rows);
}

TEST(CutCell, SupportMatrixValidationLedgerIsRowExactAndDrivesQualification)
{
  const auto entries = cut_support_matrix();
  const auto ledger = cut_support_matrix_validation_ledger();
  const auto records = qualify_cut_support_matrix();
  ASSERT_FALSE(entries.empty());
  ASSERT_FALSE(ledger.empty());
  ASSERT_EQ(records.size(), entries.size());

  std::set<std::string> advertised_keys;
  for (const auto& entry : entries) {
    EXPECT_TRUE(advertised_keys.insert(support_matrix_key(entry)).second);
  }

  std::map<std::string, std::set<CutSupportEvidenceDomain>> ledger_domains;
  std::set<std::string> ledger_key_domain_pairs;
  std::set<std::string> linear_full_validation_paths;
  std::set<CellFamily> linear_full_validation_families;
  std::set<EmbeddedGeometryKind> linear_full_validation_kinds;
  std::set<std::string> second_order_linear_full_validation_paths;
  std::set<CellFamily> second_order_linear_full_validation_families;
  std::set<EmbeddedGeometryKind> second_order_linear_full_validation_kinds;
  std::set<std::string> curved_starter_full_validation_paths;
  std::set<CellFamily> curved_starter_full_validation_families;
  std::set<std::string> true_curved_full_validation_paths;
  std::set<CellFamily> true_curved_full_validation_families;
  std::set<EmbeddedGeometryKind> true_curved_full_validation_kinds;
  std::set<std::string> true_curved_full_validation_policies;
  std::set<std::string> linear_sensitivity_paths;
  std::set<CellFamily> linear_sensitivity_families;
  std::set<EmbeddedGeometryKind> linear_sensitivity_kinds;
  std::set<std::string> second_order_linear_sensitivity_paths;
  std::set<CellFamily> second_order_linear_sensitivity_families;
  std::set<EmbeddedGeometryKind> second_order_linear_sensitivity_kinds;
  std::set<std::string> curved_starter_sensitivity_paths;
  std::set<CellFamily> curved_starter_sensitivity_families;
  std::set<std::string> true_curved_sensitivity_paths;
  std::set<CellFamily> true_curved_sensitivity_families;
  std::set<EmbeddedGeometryKind> true_curved_sensitivity_kinds;
  std::set<std::string> true_curved_sensitivity_policies;
  std::size_t diagnostic_rows = 0u;
  std::size_t full_validation_rows = 0u;
  std::size_t distributed_full_validation_rows = 0u;
  std::size_t linear_sensitivity_rows = 0u;
  std::size_t second_order_linear_sensitivity_rows = 0u;
  std::size_t curved_starter_sensitivity_rows = 0u;
  std::size_t true_curved_sensitivity_rows = 0u;
  for (const auto& evidence : ledger) {
    const auto key = support_matrix_key(evidence.entry);
    EXPECT_NE(advertised_keys.find(key), advertised_keys.end()) << key;
    EXPECT_FALSE(evidence.evidence_id.empty()) << key;
    EXPECT_FALSE(evidence.description.empty()) << key;
    EXPECT_FALSE(evidence.verification.empty()) << key;
    const auto domain_key = support_evidence_domain_key(evidence.domain);
    EXPECT_TRUE(ledger_key_domain_pairs.insert(key + ":" + domain_key).second)
        << key << " " << domain_key;
    ledger_domains[key].insert(evidence.domain);
    if (evidence.domain == CutSupportEvidenceDomain::Diagnostic) {
      ++diagnostic_rows;
    }
    if (evidence.domain == CutSupportEvidenceDomain::FullValidation) {
      ++full_validation_rows;
      if (evidence.entry.distributed) {
        ++distributed_full_validation_rows;
      }
      if (evidence.entry.geometry_order == 1) {
        EXPECT_EQ(evidence.entry.cut_mode, "linearized-cut") << key;
        EXPECT_EQ(evidence.entry.quadrature_policy, "topology-subdivision") << key;
        linear_full_validation_paths.insert(evidence.entry.fe_execution_path);
        linear_full_validation_families.insert(evidence.entry.parent_family);
        linear_full_validation_kinds.insert(evidence.entry.embedded_kind);
      } else if (is_second_order_linearized_validation_support_row(evidence.entry)) {
        EXPECT_FALSE(evidence.entry.distributed) << key;
        EXPECT_EQ(evidence.entry.cut_mode, "linearized-cut") << key;
        EXPECT_EQ(evidence.entry.quadrature_policy, "topology-subdivision") << key;
        second_order_linear_full_validation_paths.insert(evidence.entry.fe_execution_path);
        second_order_linear_full_validation_families.insert(evidence.entry.parent_family);
        second_order_linear_full_validation_kinds.insert(evidence.entry.embedded_kind);
      } else if (evidence.entry.geometry_order == 2) {
        EXPECT_EQ(evidence.entry.embedded_kind, EmbeddedGeometryKind::Plane) << key;
        EXPECT_EQ(evidence.entry.cut_mode, "curved-isoparametric-cut") << key;
        EXPECT_EQ(evidence.entry.quadrature_policy, "curved-topology-subdivision") << key;
        curved_starter_full_validation_paths.insert(evidence.entry.fe_execution_path);
        curved_starter_full_validation_families.insert(evidence.entry.parent_family);
      } else if (evidence.entry.geometry_order < 0) {
        EXPECT_EQ(evidence.entry.cut_mode, "curved-isoparametric-cut") << key;
        EXPECT_TRUE(evidence.entry.quadrature_policy == "true-curved-arrangement" ||
                    evidence.entry.quadrature_policy == "true-curved-subdivision-arrangement")
            << key;
        true_curved_full_validation_paths.insert(evidence.entry.fe_execution_path);
        true_curved_full_validation_families.insert(evidence.entry.parent_family);
        true_curved_full_validation_kinds.insert(evidence.entry.embedded_kind);
        true_curved_full_validation_policies.insert(evidence.entry.quadrature_policy);
      } else {
        ADD_FAILURE() << key;
      }
    }
    if (evidence.domain == CutSupportEvidenceDomain::Sensitivity &&
        is_first_order_linearized_sensitivity_support_row(evidence.entry)) {
      ++linear_sensitivity_rows;
      linear_sensitivity_paths.insert(evidence.entry.fe_execution_path);
      linear_sensitivity_families.insert(evidence.entry.parent_family);
      linear_sensitivity_kinds.insert(evidence.entry.embedded_kind);
    }
    if (evidence.domain == CutSupportEvidenceDomain::Sensitivity &&
        is_second_order_linearized_sensitivity_support_row(evidence.entry)) {
      ++second_order_linear_sensitivity_rows;
      second_order_linear_sensitivity_paths.insert(evidence.entry.fe_execution_path);
      second_order_linear_sensitivity_families.insert(evidence.entry.parent_family);
      second_order_linear_sensitivity_kinds.insert(evidence.entry.embedded_kind);
    }
    if (evidence.domain == CutSupportEvidenceDomain::Sensitivity &&
        is_quadratic_curved_starter_support_row(evidence.entry)) {
      ++curved_starter_sensitivity_rows;
      curved_starter_sensitivity_paths.insert(evidence.entry.fe_execution_path);
      curved_starter_sensitivity_families.insert(evidence.entry.parent_family);
    }
    if (evidence.domain == CutSupportEvidenceDomain::Sensitivity &&
        is_true_curved_topology_sensitivity_support_row(evidence.entry)) {
      ++true_curved_sensitivity_rows;
      true_curved_sensitivity_paths.insert(evidence.entry.fe_execution_path);
      true_curved_sensitivity_families.insert(evidence.entry.parent_family);
      true_curved_sensitivity_kinds.insert(evidence.entry.embedded_kind);
      true_curved_sensitivity_policies.insert(evidence.entry.quadrature_policy);
    }
  }
  EXPECT_EQ(diagnostic_rows, entries.size());
  EXPECT_GT(full_validation_rows, 0u);
  const std::set<std::string> expected_paths{
      "standard-assembly",
      "matrix-free",
      "forms-interpreter",
      "ad",
      "symbolic-tangent",
      "jit"};
  EXPECT_EQ(linear_full_validation_paths, expected_paths);
  EXPECT_EQ(curved_starter_full_validation_paths, expected_paths);
  EXPECT_EQ(linear_full_validation_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex,
                                  CellFamily::Wedge,
                                  CellFamily::Pyramid,
                                  CellFamily::Polygon,
                                  CellFamily::Polyhedron}));
  EXPECT_EQ(linear_full_validation_kinds,
            (std::set<EmbeddedGeometryKind>{EmbeddedGeometryKind::Plane,
                                            EmbeddedGeometryKind::Sphere,
                                            EmbeddedGeometryKind::TriangulatedSurface,
                                            EmbeddedGeometryKind::LevelSetField,
                                            EmbeddedGeometryKind::BooleanComposite}));
  EXPECT_EQ(second_order_linear_full_validation_paths, expected_paths);
  EXPECT_EQ(second_order_linear_full_validation_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex,
                                  CellFamily::Wedge,
                                  CellFamily::Pyramid}));
  EXPECT_EQ(second_order_linear_full_validation_kinds,
            (std::set<EmbeddedGeometryKind>{EmbeddedGeometryKind::Plane,
                                            EmbeddedGeometryKind::Sphere,
                                            EmbeddedGeometryKind::TriangulatedSurface,
                                            EmbeddedGeometryKind::LevelSetField,
                                            EmbeddedGeometryKind::BooleanComposite}));
  EXPECT_EQ(curved_starter_full_validation_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex}));
  EXPECT_EQ(true_curved_full_validation_paths, expected_paths);
  EXPECT_EQ(true_curved_full_validation_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex,
                                  CellFamily::Wedge,
                                  CellFamily::Pyramid}));
  EXPECT_EQ(true_curved_full_validation_kinds,
            (std::set<EmbeddedGeometryKind>{EmbeddedGeometryKind::Plane,
                                            EmbeddedGeometryKind::Sphere,
                                            EmbeddedGeometryKind::SignedDistanceCallback,
                                            EmbeddedGeometryKind::LevelSetField,
                                            EmbeddedGeometryKind::TriangulatedSurface,
                                            EmbeddedGeometryKind::BooleanComposite}));
  EXPECT_EQ(true_curved_full_validation_policies,
            (std::set<std::string>{"true-curved-arrangement",
                                   "true-curved-subdivision-arrangement"}));
  EXPECT_EQ(linear_sensitivity_paths, expected_paths);
  EXPECT_EQ(linear_sensitivity_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex,
                                  CellFamily::Wedge,
                                  CellFamily::Pyramid,
                                  CellFamily::Polygon,
                                  CellFamily::Polyhedron}));
  EXPECT_EQ(linear_sensitivity_kinds,
            (std::set<EmbeddedGeometryKind>{EmbeddedGeometryKind::Plane,
                                            EmbeddedGeometryKind::Sphere,
                                            EmbeddedGeometryKind::TriangulatedSurface,
                                            EmbeddedGeometryKind::LevelSetField,
                                            EmbeddedGeometryKind::BooleanComposite}));
  EXPECT_EQ(linear_sensitivity_rows,
            2u * expected_paths.size() *
                linear_sensitivity_families.size() *
                linear_sensitivity_kinds.size());
  EXPECT_EQ(second_order_linear_sensitivity_paths, expected_paths);
  EXPECT_EQ(second_order_linear_sensitivity_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex,
                                  CellFamily::Wedge,
                                  CellFamily::Pyramid}));
  EXPECT_EQ(second_order_linear_sensitivity_kinds,
            (std::set<EmbeddedGeometryKind>{EmbeddedGeometryKind::Plane,
                                            EmbeddedGeometryKind::Sphere,
                                            EmbeddedGeometryKind::TriangulatedSurface,
                                            EmbeddedGeometryKind::LevelSetField,
                                            EmbeddedGeometryKind::BooleanComposite}));
  EXPECT_EQ(second_order_linear_sensitivity_rows,
            expected_paths.size() *
                second_order_linear_sensitivity_families.size() *
                second_order_linear_sensitivity_kinds.size());
  EXPECT_EQ(curved_starter_sensitivity_paths, expected_paths);
  EXPECT_EQ(curved_starter_sensitivity_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex}));
  EXPECT_EQ(curved_starter_sensitivity_rows,
            2u * expected_paths.size() * curved_starter_sensitivity_families.size());
  EXPECT_EQ(true_curved_sensitivity_paths, expected_paths);
  EXPECT_EQ(true_curved_sensitivity_families,
            (std::set<CellFamily>{CellFamily::Line,
                                  CellFamily::Triangle,
                                  CellFamily::Quad,
                                  CellFamily::Tetra,
                                  CellFamily::Hex,
                                  CellFamily::Wedge,
                                  CellFamily::Pyramid}));
  EXPECT_EQ(true_curved_sensitivity_kinds,
            (std::set<EmbeddedGeometryKind>{EmbeddedGeometryKind::Plane,
                                            EmbeddedGeometryKind::Sphere,
                                            EmbeddedGeometryKind::SignedDistanceCallback,
                                            EmbeddedGeometryKind::LevelSetField,
                                            EmbeddedGeometryKind::TriangulatedSurface,
                                            EmbeddedGeometryKind::BooleanComposite}));
  EXPECT_EQ(true_curved_sensitivity_policies,
            (std::set<std::string>{"true-curved-arrangement",
                                   "true-curved-subdivision-arrangement"}));
  EXPECT_EQ(true_curved_sensitivity_rows,
            2u * expected_paths.size() * (7u + 5u + 7u * 5u));
  const auto true_curved_full_validation_rows =
      2u * expected_paths.size() * (7u + 5u + 7u * 5u);
  const auto second_order_linear_full_validation_rows =
      expected_paths.size() *
      second_order_linear_full_validation_families.size() *
      second_order_linear_full_validation_kinds.size();
  EXPECT_EQ(full_validation_rows,
            2u * expected_paths.size() *
                (linear_full_validation_families.size() *
                     linear_full_validation_kinds.size() +
                 curved_starter_full_validation_families.size()) +
                second_order_linear_full_validation_rows +
                true_curved_full_validation_rows);
  EXPECT_GT(distributed_full_validation_rows, 0u);

  std::size_t ledger_complete_rows = 0u;
  std::size_t qualified_rows = 0u;
  for (const auto& record : records) {
    const auto key = support_matrix_key(record.entry);
    const auto domain_it = ledger_domains.find(key);
    ASSERT_NE(domain_it, ledger_domains.end()) << key;
    const auto& domains = domain_it->second;
    const auto has_domain = [&](CutSupportEvidenceDomain domain) {
      return domains.find(domain) != domains.end();
    };

    EXPECT_EQ(record.topology_evidence,
              has_domain(CutSupportEvidenceDomain::Topology)) << key;
    EXPECT_EQ(record.quadrature_evidence,
              has_domain(CutSupportEvidenceDomain::Quadrature)) << key;
    EXPECT_EQ(record.fe_execution_evidence,
              has_domain(CutSupportEvidenceDomain::FeExecution)) << key;
    EXPECT_EQ(record.restart_rollback_evidence,
              has_domain(CutSupportEvidenceDomain::RestartRollback)) << key;
    EXPECT_EQ(record.mpi_evidence,
              has_domain(CutSupportEvidenceDomain::Mpi)) << key;
    EXPECT_EQ(record.sensitivity_evidence,
              has_domain(CutSupportEvidenceDomain::Sensitivity)) << key;
    EXPECT_EQ(record.diagnostic_evidence,
              has_domain(CutSupportEvidenceDomain::Diagnostic)) << key;
    EXPECT_EQ(record.validation_evidence,
              has_domain(CutSupportEvidenceDomain::FullValidation)) << key;

    const bool ledger_complete =
        (!record.requires_topology_evidence || record.topology_evidence) &&
        (!record.requires_quadrature_evidence || record.quadrature_evidence) &&
        (!record.requires_fe_execution_evidence || record.fe_execution_evidence) &&
        (!record.requires_restart_rollback_evidence || record.restart_rollback_evidence) &&
        (!record.requires_mpi_evidence || record.mpi_evidence) &&
        (!record.requires_sensitivity_evidence || record.sensitivity_evidence) &&
        (!record.requires_diagnostic_evidence || record.diagnostic_evidence) &&
        (!record.requires_validation_evidence || record.validation_evidence);
    if (ledger_complete) {
      ++ledger_complete_rows;
    }
    if (record.qualified) {
      ++qualified_rows;
    }
    EXPECT_EQ(record.qualified, ledger_complete) << key;
  }

  EXPECT_GT(ledger_complete_rows, 0u);
  EXPECT_EQ(qualified_rows, ledger_complete_rows);
}

TEST(CutCell, SupportMatrixTopologyAndQuadratureAcceptanceEvidenceIsRowSpecific)
{
  const auto entries = cut_support_matrix();
  const auto ledger = cut_support_matrix_validation_ledger();
  const auto records = qualify_cut_support_matrix();
  ASSERT_FALSE(entries.empty());
  ASSERT_FALSE(ledger.empty());
  ASSERT_EQ(records.size(), entries.size());

  std::map<std::string, CutSupportMatrixQualificationRecord> records_by_key;
  for (const auto& record : records) {
    records_by_key.emplace(support_matrix_key(record.entry), record);
  }

  std::map<std::string, std::map<CutSupportEvidenceDomain, CutSupportMatrixEvidenceRecord>>
      evidence_by_key_and_domain;
  std::size_t topology_evidence_rows = 0u;
  std::size_t quadrature_evidence_rows = 0u;
  for (const auto& evidence : ledger) {
    if (evidence.domain == CutSupportEvidenceDomain::Topology) {
      ++topology_evidence_rows;
    }
    if (evidence.domain == CutSupportEvidenceDomain::Quadrature) {
      ++quadrature_evidence_rows;
    }
    evidence_by_key_and_domain[support_matrix_key(evidence.entry)][evidence.domain] =
        evidence;
  }

  const auto expected_topology_id = [](const CutSupportMatrixEntry& entry) {
    if (is_first_order_linearized_validation_support_row(entry)) {
      return std::string{"phase26.topology.linearized-first-order-all-families"};
    }
    if (is_second_order_linearized_validation_support_row(entry)) {
      return std::string{
          "phase26.topology.linearized-second-order-high-order-families"};
    }
    if (is_quadratic_curved_starter_support_row(entry)) {
      return std::string{"phase26.topology.quadratic-curved-plane-starter"};
    }
    if (is_true_curved_topology_sensitivity_support_row(entry)) {
      return std::string{"phase26.topology.true-curved-arrangement-subdivision"};
    }
    return std::string{};
  };

  const auto expected_quadrature_id = [](const CutSupportMatrixEntry& entry) {
    if (is_first_order_linearized_validation_support_row(entry)) {
      return std::string{"phase26.quadrature.linearized-first-order-topology-derived"};
    }
    if (is_second_order_linearized_validation_support_row(entry)) {
      return std::string{
          "phase26.quadrature.linearized-second-order-topology-derived"};
    }
    if (is_quadratic_curved_starter_support_row(entry)) {
      return std::string{"phase26.quadrature.quadratic-curved-plane-starter"};
    }
    if (is_true_curved_topology_sensitivity_support_row(entry)) {
      return std::string{"phase26.quadrature.true-curved-arrangement-subdivision"};
    }
    return std::string{};
  };

  std::size_t implemented_rows = 0u;
  std::set<std::string> observed_topology_ids;
  std::set<std::string> observed_quadrature_ids;
  for (const auto& entry : entries) {
    const auto key = support_matrix_key(entry);
    const auto record_it = records_by_key.find(key);
    ASSERT_NE(record_it, records_by_key.end()) << key;
    const auto& record = record_it->second;
    const auto evidence_it = evidence_by_key_and_domain.find(key);
    ASSERT_NE(evidence_it, evidence_by_key_and_domain.end()) << key;

    const bool implemented = entry.status != CutSupportStatus::Unsupported;
    if (!implemented) {
      EXPECT_EQ(evidence_it->second.find(CutSupportEvidenceDomain::Topology),
                evidence_it->second.end())
          << key;
      EXPECT_EQ(evidence_it->second.find(CutSupportEvidenceDomain::Quadrature),
                evidence_it->second.end())
          << key;
      EXPECT_TRUE(record.qualified) << key << " :: " << joined_messages(record.missing);
      continue;
    }

    ++implemented_rows;
    ASSERT_TRUE(record.qualified) << key << " :: " << joined_messages(record.missing);
    ASSERT_TRUE(record.topology_evidence) << key;
    ASSERT_TRUE(record.quadrature_evidence) << key;
    ASSERT_TRUE(record.restart_rollback_evidence) << key;
    if (entry.distributed) {
      ASSERT_TRUE(record.mpi_evidence) << key;
    }

    const auto topology_it =
        evidence_it->second.find(CutSupportEvidenceDomain::Topology);
    const auto quadrature_it =
        evidence_it->second.find(CutSupportEvidenceDomain::Quadrature);
    ASSERT_NE(topology_it, evidence_it->second.end()) << key;
    ASSERT_NE(quadrature_it, evidence_it->second.end()) << key;
    const auto& topology = topology_it->second;
    const auto& quadrature = quadrature_it->second;

    const auto expected_topology = expected_topology_id(entry);
    const auto expected_quadrature = expected_quadrature_id(entry);
    ASSERT_FALSE(expected_topology.empty()) << key;
    ASSERT_FALSE(expected_quadrature.empty()) << key;
    EXPECT_EQ(topology.evidence_id, expected_topology) << key;
    EXPECT_EQ(quadrature.evidence_id, expected_quadrature) << key;
    EXPECT_NE(topology.evidence_id, "phase26.topology.reconstructed-cut-records")
        << key;
    EXPECT_NE(quadrature.evidence_id, "phase26.quadrature.topology-derived-imports")
        << key;
    EXPECT_EQ(topology.evidence_id.find("unclassified"), std::string::npos)
        << key;
    EXPECT_EQ(quadrature.evidence_id.find("unclassified"), std::string::npos)
        << key;

    EXPECT_NE(topology.description.find("explicit ordered"), std::string::npos)
        << key;
    EXPECT_NE(topology.description.find("provenance"), std::string::npos) << key;
    EXPECT_NE(topology.description.find("stable"), std::string::npos) << key;
    EXPECT_NE(topology.verification.find("CutCell."), std::string::npos) << key;
    if (entry.distributed) {
      EXPECT_NE(topology.verification.find("MPI"), std::string::npos) << key;
    }

    EXPECT_NE(quadrature.description.find("topology"), std::string::npos)
        << key;
    EXPECT_NE(quadrature.description.find("conservation"), std::string::npos)
        << key;
    EXPECT_NE(quadrature.description.find("analytic exactness"), std::string::npos)
        << key;
    EXPECT_NE(quadrature.verification.find("conservation"), std::string::npos)
        << key;
    EXPECT_NE(quadrature.verification.find("analytic exactness"), std::string::npos)
        << key;
    EXPECT_NE(quadrature.verification.find("CutIntegrationInfrastructure"),
              std::string::npos)
        << key;

    observed_topology_ids.insert(topology.evidence_id);
    observed_quadrature_ids.insert(quadrature.evidence_id);
  }

  EXPECT_GT(implemented_rows, 0u);
  EXPECT_EQ(topology_evidence_rows, implemented_rows);
  EXPECT_EQ(quadrature_evidence_rows, implemented_rows);
  EXPECT_EQ(observed_topology_ids,
            (std::set<std::string>{
                "phase26.topology.linearized-first-order-all-families",
                "phase26.topology.linearized-second-order-high-order-families",
                "phase26.topology.quadratic-curved-plane-starter",
                "phase26.topology.true-curved-arrangement-subdivision"}));
  EXPECT_EQ(observed_quadrature_ids,
            (std::set<std::string>{
                "phase26.quadrature.linearized-first-order-topology-derived",
                "phase26.quadrature.linearized-second-order-topology-derived",
                "phase26.quadrature.quadratic-curved-plane-starter",
                "phase26.quadrature.true-curved-arrangement-subdivision"}));
}

TEST(CutCell, SupportMatrixValidationAuditCategorizesEveryAdvertisedRow)
{
  const auto entries = cut_support_matrix();
  const auto records = qualify_cut_support_matrix();
  const auto audit = audit_cut_support_matrix_validation();
  ASSERT_FALSE(entries.empty());
  ASSERT_EQ(audit.size(), entries.size());
  ASSERT_EQ(audit.size(), records.size());

  std::map<std::string, CutSupportMatrixQualificationRecord> records_by_key;
  for (const auto& record : records) {
    records_by_key.emplace(support_matrix_key(record.entry), record);
  }

  std::set<std::string> audit_keys;
  std::size_t unsupported_rows = 0u;
  std::size_t fully_validated_rows = 0u;
  std::size_t missing_validation_rows = 0u;
  std::size_t missing_sensitivity_rows = 0u;
  std::size_t missing_restart_or_mpi_rows = 0u;
  std::size_t missing_fe_rows = 0u;
  std::size_t advertised_too_broadly_rows = 0u;
  std::size_t rows_missing_full_validation_domain = 0u;
  std::size_t rows_missing_sensitivity_domain = 0u;

  for (const auto& row : audit) {
    const auto key = support_matrix_key(row.qualification.entry);
    EXPECT_TRUE(audit_keys.insert(key).second) << key;
    EXPECT_FALSE(row.summary.empty()) << key;
    const auto expected_record = records_by_key.find(key);
    ASSERT_NE(expected_record, records_by_key.end()) << key;
    EXPECT_EQ(row.qualification.qualified, expected_record->second.qualified) << key;
    EXPECT_EQ(row.qualification.missing, expected_record->second.missing) << key;

    const bool implemented =
        row.qualification.entry.status != CutSupportStatus::Unsupported;
    if (!implemented) {
      ++unsupported_rows;
      EXPECT_EQ(row.category, CutSupportAuditCategory::Unsupported) << key;
      EXPECT_TRUE(row.missing_domains.empty()) << key;
      EXPECT_TRUE(row.qualification.qualified) << key;
      continue;
    }

    if (has_missing_domain(row, CutSupportEvidenceDomain::FullValidation)) {
      ++rows_missing_full_validation_domain;
    }
    if (has_missing_domain(row, CutSupportEvidenceDomain::Sensitivity)) {
      ++rows_missing_sensitivity_domain;
    }

    if (row.qualification.qualified) {
      ++fully_validated_rows;
      EXPECT_EQ(row.category, CutSupportAuditCategory::FullyValidated) << key;
      EXPECT_TRUE(row.missing_domains.empty()) << key;
      EXPECT_TRUE(row.qualification.missing.empty()) << key;
    } else {
      EXPECT_FALSE(row.missing_domains.empty()) << key;
      EXPECT_FALSE(row.qualification.missing.empty()) << key;
      EXPECT_NE(row.category, CutSupportAuditCategory::FullyValidated) << key;
    }

    switch (row.category) {
      case CutSupportAuditCategory::Unsupported:
        ADD_FAILURE() << "implemented row categorized unsupported: " << key;
        break;
      case CutSupportAuditCategory::FullyValidated:
        break;
      case CutSupportAuditCategory::MissingAnalyticValidation:
        ++missing_validation_rows;
        EXPECT_TRUE(has_missing_domain(row, CutSupportEvidenceDomain::FullValidation)) << key;
        EXPECT_FALSE(has_missing_domain(row, CutSupportEvidenceDomain::Sensitivity)) << key;
        break;
      case CutSupportAuditCategory::MissingSensitivityEvidence:
        ++missing_sensitivity_rows;
        EXPECT_TRUE(has_missing_domain(row, CutSupportEvidenceDomain::Sensitivity)) << key;
        break;
      case CutSupportAuditCategory::MissingRestartOrMpiEvidence:
        ++missing_restart_or_mpi_rows;
        break;
      case CutSupportAuditCategory::MissingFeExecutionEvidence:
        ++missing_fe_rows;
        break;
      case CutSupportAuditCategory::AdvertisedTooBroadly:
        ++advertised_too_broadly_rows;
        break;
    }

    EXPECT_FALSE(has_missing_domain(row, CutSupportEvidenceDomain::Topology)) << key;
    EXPECT_FALSE(has_missing_domain(row, CutSupportEvidenceDomain::Quadrature)) << key;
    EXPECT_FALSE(has_missing_domain(row, CutSupportEvidenceDomain::Diagnostic)) << key;
    EXPECT_FALSE(has_missing_domain(row, CutSupportEvidenceDomain::FeExecution)) << key;
    EXPECT_FALSE(has_missing_domain(row, CutSupportEvidenceDomain::RestartRollback)) << key;
    EXPECT_FALSE(has_missing_domain(row, CutSupportEvidenceDomain::Mpi)) << key;
  }

  EXPECT_EQ(audit_keys.size(), entries.size());
  EXPECT_GT(unsupported_rows, 0u);
  EXPECT_GT(fully_validated_rows, 0u);
  EXPECT_EQ(missing_validation_rows, 0u);
  EXPECT_EQ(missing_sensitivity_rows, 0u);
  EXPECT_EQ(rows_missing_full_validation_domain, 0u);
  EXPECT_EQ(rows_missing_sensitivity_domain, 0u);
  EXPECT_EQ(missing_restart_or_mpi_rows, 0u);
  EXPECT_EQ(missing_fe_rows, 0u);
  EXPECT_EQ(advertised_too_broadly_rows, 0u);
}

TEST(CutCell, BooleanCompositionSemanticsCoverUnionIntersectionDifferenceAndNestedRegions)
{
  auto narrow = plane(0.25, 71);
  narrow.provenance.persistent_id = "narrow-halfspace";
  auto wide = plane(0.75, 72);
  wide.provenance.persistent_id = "wide-halfspace";

  EmbeddedGeometryDescriptor region_union;
  region_union.kind = EmbeddedGeometryKind::BooleanComposite;
  region_union.boolean_operation = EmbeddedGeometryBooleanOperation::Union;
  region_union.provenance.persistent_id = "halfspace-union";
  region_union.provenance.provenance_epoch = 73;
  region_union.children = {narrow, wide};

  EXPECT_LT(region_union.signed_distance({{0.50, 0.0, 0.0}}), 0.0);
  EXPECT_GT(region_union.signed_distance({{1.00, 0.0, 0.0}}), 0.0);
  const auto union_diagnostic = diagnose_boolean_region_composition(
      region_union,
      {{{0.10, 0.0, 0.0}}, {{0.50, 0.0, 0.0}}, {{1.00, 0.0, 0.0}}});
  EXPECT_TRUE(union_diagnostic.ok);
  EXPECT_TRUE(has_message_containing(union_diagnostic.messages, "overlapping child regions"));

  EmbeddedGeometryDescriptor region_intersection;
  region_intersection.kind = EmbeddedGeometryKind::BooleanComposite;
  region_intersection.boolean_operation = EmbeddedGeometryBooleanOperation::Intersection;
  region_intersection.provenance.persistent_id = "halfspace-intersection";
  region_intersection.provenance.provenance_epoch = 74;
  region_intersection.children = {narrow, wide};

  EXPECT_LT(region_intersection.signed_distance({{0.10, 0.0, 0.0}}), 0.0);
  EXPECT_GT(region_intersection.signed_distance({{0.50, 0.0, 0.0}}), 0.0);
  const auto intersection_diagnostic = diagnose_boolean_region_composition(
      region_intersection,
      {{{0.10, 0.0, 0.0}}, {{1.00, 0.0, 0.0}}});
  EXPECT_TRUE(intersection_diagnostic.ok);
  EXPECT_TRUE(has_message_containing(intersection_diagnostic.messages,
                                     "outside every child region"));

  EmbeddedGeometryDescriptor region_difference;
  region_difference.kind = EmbeddedGeometryKind::BooleanComposite;
  region_difference.boolean_operation = EmbeddedGeometryBooleanOperation::Difference;
  region_difference.provenance.persistent_id = "halfspace-difference";
  region_difference.provenance.provenance_epoch = 75;
  region_difference.children = {wide, narrow};

  EXPECT_GT(region_difference.signed_distance({{0.10, 0.0, 0.0}}), 0.0);
  EXPECT_LT(region_difference.signed_distance({{0.50, 0.0, 0.0}}), 0.0);
  EXPECT_GT(region_difference.signed_distance({{1.00, 0.0, 0.0}}), 0.0);
  const auto difference_diagnostic = diagnose_boolean_region_composition(
      region_difference,
      {{{0.10, 0.0, 0.0}}, {{0.50, 0.0, 0.0}}, {{1.00, 0.0, 0.0}}});
  EXPECT_TRUE(difference_diagnostic.ok);
  EXPECT_TRUE(has_message_containing(difference_diagnostic.messages, "base/subtracted overlap"));

  EmbeddedGeometryDescriptor nested;
  nested.kind = EmbeddedGeometryKind::BooleanComposite;
  nested.boolean_operation = EmbeddedGeometryBooleanOperation::Intersection;
  nested.provenance.persistent_id = "nested-boolean-region";
  nested.provenance.provenance_epoch = 76;
  nested.children = {region_union, sphere({{0.0, 0.0, 0.0}}, 10.0, 77, "enclosing-sphere")};

  EXPECT_LT(nested.signed_distance({{0.50, 0.0, 0.0}}), 0.0);
  EXPECT_GT(nested.signed_distance({{11.0, 0.0, 0.0}}), 0.0);
  const auto nested_diagnostic = diagnose_boolean_region_composition(
      nested,
      {{{0.10, 0.0, 0.0}}, {{0.50, 0.0, 0.0}}, {{11.0, 0.0, 0.0}}});
  EXPECT_TRUE(nested_diagnostic.ok);
  EXPECT_TRUE(has_message_containing(nested_diagnostic.messages, "nested Boolean child"));

  auto coincident_a = plane(0.0, 78);
  coincident_a.provenance.persistent_id = "coincident-a";
  auto coincident_b = plane(0.0, 79);
  coincident_b.provenance.persistent_id = "coincident-b";
  EmbeddedGeometryDescriptor coincident_intersection;
  coincident_intersection.kind = EmbeddedGeometryKind::BooleanComposite;
  coincident_intersection.boolean_operation = EmbeddedGeometryBooleanOperation::Intersection;
  coincident_intersection.provenance.persistent_id = "coincident-boolean-region";
  coincident_intersection.children = {coincident_a, coincident_b};
  CutPredicatePolicy predicate_policy;
  predicate_policy.name = "phase26-adaptive-exact";
  predicate_policy.robust.intersection_tolerance = 1.0e-8;
  const auto coincident_diagnostic = diagnose_boolean_region_composition(
      coincident_intersection,
      {{{0.0, 0.0, 0.0}}},
      predicate_policy);
  EXPECT_TRUE(coincident_diagnostic.ok);
  EXPECT_TRUE(has_message_containing(coincident_diagnostic.messages,
                                     "domain=boolean-arrangement"))
      << joined_messages(coincident_diagnostic.messages);

  const auto revisions = nested.effective_revisions();
  EXPECT_EQ(revisions.geometry_epoch, 77u);
  EXPECT_EQ(revisions.provenance_revision, 77u);
  EXPECT_TRUE(nested.diagnose_query_support().ok);
}

TEST(CutCell, NestedBooleanCompositionRestartRecordsPreserveFlattenedChildProvenance)
{
  auto mesh = make_tetra_mesh();
  auto inner_plane = plane(0.5, 81);
  inner_plane.provenance.persistent_id = "nested-plane";
  auto enclosing = sphere({{0.0, 0.0, 0.0}}, 10.0, 82, "nested-inner-sphere");

  EmbeddedGeometryDescriptor inner;
  inner.kind = EmbeddedGeometryKind::BooleanComposite;
  inner.boolean_operation = EmbeddedGeometryBooleanOperation::Intersection;
  inner.provenance.persistent_id = "inner-composite";
  inner.provenance.provenance_epoch = 83;
  inner.children = {inner_plane, enclosing};

  EmbeddedGeometryDescriptor outer;
  outer.kind = EmbeddedGeometryKind::BooleanComposite;
  outer.boolean_operation = EmbeddedGeometryBooleanOperation::Union;
  outer.provenance.persistent_id = "outer-composite";
  outer.provenance.provenance_epoch = 84;
  outer.children = {inner, sphere({{0.0, 0.0, 0.0}}, 0.05, 85, "nested-small-sphere")};

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, outer, options);
  ASSERT_EQ(map.cells.size(), 1u);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  const auto topology = reconstruct_cut_topology(mesh, map);
  ASSERT_FALSE(topology.side_regions.empty());
  const auto restart = make_cut_classification_restart_record(map, topology);
  EXPECT_TRUE(restart.is_composed_region);
  EXPECT_EQ(restart.composition_operation, EmbeddedGeometryBooleanOperation::Union);
  ASSERT_EQ(restart.composition_children.size(), 4u);
  EXPECT_EQ(restart.composition_children[0].depth, 1u);
  EXPECT_EQ(restart.composition_children[0].provenance.persistent_id, "inner-composite");
  EXPECT_EQ(restart.composition_children[1].depth, 2u);
  EXPECT_EQ(restart.composition_children[1].parent_persistent_id, "inner-composite");
  EXPECT_EQ(restart.composition_children[1].provenance.persistent_id, "nested-plane");
  EXPECT_EQ(restart.composition_children[2].provenance.persistent_id, "nested-inner-sphere");
  EXPECT_EQ(restart.composition_children[3].depth, 1u);
  EXPECT_EQ(restart.composition_children[3].provenance.persistent_id, "nested-small-sphere");
  EXPECT_EQ(restart.side_regions.size(), topology.side_regions.size());
}

TEST(CutCell, DistributedExchangePacketDeduplicatesAndProjectionPreservesProvenance)
{
  auto mesh = make_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, plane(0.5, 31), options);
  const auto topology = reconstruct_cut_topology(mesh, map);
  ASSERT_FALSE(topology.vertices.empty());

  auto packet = make_distributed_cut_exchange_packet(map, topology);
  const auto original_count = packet.entities.size();
  ASSERT_GT(original_count, 0u);
  const auto side_payload = std::find_if(packet.entities.begin(), packet.entities.end(), [](const auto& entity) {
    return entity.kind == CutTopologyEntityKind::SideRegion && entity.measure > 0.0;
  });
  ASSERT_NE(side_payload, packet.entities.end());
  EXPECT_FALSE(side_payload->vertex_ids.empty());
  EXPECT_FALSE(side_payload->face_ids.empty());
  EXPECT_TRUE(side_payload->closed_topology);
  EXPECT_GT(side_payload->parent_measure, 0.0);
  EXPECT_GT(side_payload->volume_fraction, 0.0);
  EXPECT_LE(side_payload->volume_fraction, 1.0);
  packet.entities.push_back(packet.entities.front());
  const auto deduped = deduplicate_cut_exchange_packet(std::move(packet));
  EXPECT_EQ(deduped.entities.size(), original_count);
  EXPECT_NE(deduped.revision_key, 0u);

  auto moved_plane = plane(0.55, 32);
  const auto projected = project_cut_topology_to_embedded_geometry(topology, moved_plane, 44);
  EXPECT_NE(projected.topology_revision, topology.topology_revision);
  ASSERT_EQ(projected.vertices.size(), topology.vertices.size());
  EXPECT_EQ(projected.vertices[0].provenance.persistent_id,
            topology.vertices[0].provenance.persistent_id);

  const auto validity = diagnose_cut_topology_validity(projected);
  EXPECT_TRUE(validity.ok);
}

TEST(CutCell, DistributedExchangePacketPreservesCurvedSensitivityMetadata)
{
  const auto mesh = make_cubic_curved_tetra_mesh();
  auto embedded = sphere({{0.30, 0.30, 0.30}}, 0.42, 3401, "packet-curved-sphere");

  CutClassificationOptions classification_options;
  classification_options.classify_faces = false;
  classification_options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, embedded, classification_options);

  CutTopologyOptions topology_options;
  topology_options.curved_arrangement_mode = CutCurvedArrangementMode::TrueArrangement;
  const auto topology = reconstruct_cut_topology(mesh, map, topology_options);
  ASSERT_TRUE(topology.supported) << joined_messages(topology.diagnostics);
  ASSERT_FALSE(topology.curved_patches.empty());
  ASSERT_FALSE(topology.sensitivity_records.empty());

  auto packet = make_distributed_cut_exchange_packet(map, topology);
  const auto curved_record = std::find_if(packet.entities.begin(),
                                         packet.entities.end(),
                                         [](const auto& entity) {
                                           return entity.kind == CutTopologyEntityKind::InterfacePolygon &&
                                                  entity.curved_isoparametric;
                                         });
  ASSERT_NE(curved_record, packet.entities.end());
  EXPECT_EQ(curved_record->geometry_order, 3);
  EXPECT_EQ(curved_record->embedded_kind, EmbeddedGeometryKind::Sphere);
  EXPECT_EQ(curved_record->construction_policy, "true-curved-subdivision-arrangement");
  EXPECT_TRUE(curved_record->parametric_coordinates_valid);
  EXPECT_TRUE(curved_record->exact_topology_available);
  EXPECT_FALSE(curved_record->linearized_surrogate);
  EXPECT_TRUE(curved_record->isoparametric_quadrature_available);
  EXPECT_FALSE(curved_record->parent_parametric_coordinates.empty());
  EXPECT_FALSE(curved_record->quadrature_parent_parametric_coordinates.empty());
  EXPECT_EQ(curved_record->quadrature_parent_parametric_coordinates.size(),
            curved_record->quadrature_weights.size());
  EXPECT_GT(curved_record->geometry_sensitivity_dof_count, 0u);
  EXPECT_GT(curved_record->geometry_sensitivity_sample_count, 0u);
  EXPECT_TRUE(curved_record->geometry_sensitivity_ad_compatible);
  EXPECT_TRUE(curved_record->location_sensitivity_available);
  EXPECT_TRUE(curved_record->jacobian_sensitivity_available);
  EXPECT_TRUE(curved_record->measure_sensitivity_available);
  EXPECT_TRUE(curved_record->normal_sensitivity_available);
  EXPECT_TRUE(curved_record->quadrature_weight_sensitivity_available);

  const auto duplicate_curved_record = *curved_record;
  packet.entities.push_back(duplicate_curved_record);
  const auto deduped = deduplicate_cut_exchange_packet(std::move(packet));
  const auto deduped_curved = std::find_if(deduped.entities.begin(),
                                           deduped.entities.end(),
                                           [](const auto& entity) {
                                             return entity.kind == CutTopologyEntityKind::InterfacePolygon &&
                                                    entity.curved_isoparametric;
                                           });
  ASSERT_NE(deduped_curved, deduped.entities.end());
  EXPECT_EQ(deduped_curved->construction_policy, "true-curved-subdivision-arrangement");
  EXPECT_TRUE(deduped_curved->geometry_sensitivity_ad_compatible);
}

TEST(CutCell, DistributedCutStateSerializesOwnedPayloadsAndInvalidatesOnOwnershipChange)
{
  DistributedMesh mesh(MeshComm::self());
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0},
      std::vector<offset_t>{0, 4},
      std::vector<index_t>{0, 1, 2, 3},
      std::vector<CellShape>{CellShape{CellFamily::Tetra, 4, 1}});

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  const auto map = classify_embedded_geometry(mesh, plane(0.5, 61), options);
  const auto topology = reconstruct_cut_topology(mesh.local_mesh(), map);
  const auto state = build_distributed_cut_state(mesh, map, topology);
  const auto diagnostic = diagnose_distributed_cut_state(state);
  EXPECT_TRUE(diagnostic.ok);
  EXPECT_TRUE(state.valid_for(mesh, map, topology));
  EXPECT_FALSE(state.neighbor_sparse_exchange);
  EXPECT_TRUE(state.communication_neighbors.empty());
  EXPECT_TRUE(state.received_neighbor_ranks.empty());
  EXPECT_FALSE(state.exchanged_packet.entities.empty());
  EXPECT_EQ(state.owned_records.size(), state.exchanged_packet.entities.size());
  EXPECT_TRUE(state.imported_records.empty());
  EXPECT_TRUE(state.ghost_records.empty());

  const auto side_payload = std::find_if(state.exchanged_packet.entities.begin(),
                                         state.exchanged_packet.entities.end(),
                                         [](const auto& entity) {
                                           return entity.kind == CutTopologyEntityKind::SideRegion &&
                                                  entity.measure > 0.0;
                                         });
  ASSERT_NE(side_payload, state.exchanged_packet.entities.end());
  EXPECT_TRUE(side_payload->closed_topology);
  EXPECT_FALSE(side_payload->vertex_ids.empty());
  EXPECT_FALSE(side_payload->face_ids.empty());

  mesh.set_cell_gids(std::vector<svmp::gid_t>{1001});
  EXPECT_FALSE(state.valid_for(mesh, map, topology));
}
