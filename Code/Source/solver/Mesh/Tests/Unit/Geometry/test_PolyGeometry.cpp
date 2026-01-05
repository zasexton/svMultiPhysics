/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved.
 */

#include "gtest/gtest.h"
#include "Geometry/PolyGeometry.h"
#include "Geometry/MeshGeometry.h"
#include "Core/MeshBase.h"

namespace svmp {
namespace test {

class PolyGeometryTest : public ::testing::Test {
protected:
  static constexpr real_t tol = 1e-12;

  static bool approx(real_t a, real_t b, real_t t = tol) { return std::abs(a - b) < t; }
  static bool approx3(const std::array<real_t,3>& a, const std::array<real_t,3>& b, real_t t = tol) {
    return approx(a[0], b[0], t) && approx(a[1], b[1], t) && approx(a[2], b[2], t);
  }
};

TEST_F(PolyGeometryTest, TriangleAreaAndCentroidRaw) {
  std::vector<std::array<real_t,3>> verts = {{ {0,0,0}, {1,0,0}, {0,1,0} }};
  real_t area = PolyGeometry::polygon_area(verts);
  EXPECT_TRUE(approx(area, 0.5));

  auto c = PolyGeometry::polygon_centroid(verts);
  EXPECT_TRUE(approx3(c, {{1.0/3.0, 1.0/3.0, 0.0}}));
}

TEST_F(PolyGeometryTest, QuadAreaAndCentroidRaw) {
  std::vector<std::array<real_t,3>> verts = {{ {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} }};
  real_t area = PolyGeometry::polygon_area(verts);
  EXPECT_TRUE(approx(area, 1.0));

  auto c = PolyGeometry::polygon_centroid(verts);
  EXPECT_TRUE(approx3(c, {{0.5, 0.5, 0.0}}));
}

TEST_F(PolyGeometryTest, ConcaveLPolygonCentroidRaw) {
  // "L" polygon, ordered so that triangle-fan centroid about verts[0] is not reliable.
  // Base polygon (z=0), CCW:
  //   (0,0) (2,0) (2,1) (1,1) (1,2) (0,2)
  // Rotated start index to (2,0): {1,2,3,4,5,0}.
  std::vector<std::array<real_t,3>> verts = {{
      {2, 0, 0},
      {2, 1, 0},
      {1, 1, 0},
      {1, 2, 0},
      {0, 2, 0},
      {0, 0, 0}
  }};

  // Analytic centroid of a 2x2 square minus 1x1 at top-right:
  //   (4*(1,1) - 1*(1.5,1.5))/3 = (5/6, 5/6).
  const auto c = PolyGeometry::polygon_centroid(verts);
  EXPECT_TRUE(approx3(c, {{5.0 / 6.0, 5.0 / 6.0, 0.0}}));
}

TEST_F(PolyGeometryTest, TriangulatePlanarPolygon_ConcaveL_AreaClosure) {
  std::vector<std::array<real_t,3>> verts = {{
      {2, 0, 0},
      {2, 1, 0},
      {1, 1, 0},
      {1, 2, 0},
      {0, 2, 0},
      {0, 0, 0}
  }};

  std::vector<std::array<index_t,3>> tris;
  ASSERT_TRUE(PolyGeometry::triangulate_planar_polygon(verts, tris));
  EXPECT_EQ(tris.size(), 4u);

  real_t sum = 0.0;
  for (const auto& t : tris) {
    const auto& a = verts[static_cast<size_t>(t[0])];
    const auto& b = verts[static_cast<size_t>(t[1])];
    const auto& c = verts[static_cast<size_t>(t[2])];
    sum += MeshGeometry::triangle_area(a, b, c);
  }
  EXPECT_TRUE(approx(sum, 3.0));
}

TEST_F(PolyGeometryTest, TriangulatePlanarPolygon_RejectsSelfIntersectingBowTie) {
  // Self-intersecting "bow tie" polygon in z=0.
  std::vector<std::array<real_t,3>> verts = {{
      {0, 0, 0},
      {1, 1, 0},
      {0, 1, 0},
      {1, 0, 0}
  }};
  std::vector<std::array<index_t,3>> tris;
  EXPECT_FALSE(PolyGeometry::triangulate_planar_polygon(verts, tris));
  EXPECT_TRUE(tris.empty());
}

TEST_F(PolyGeometryTest, TriangleAreaAndCentroidMesh) {
  // Build a mesh with 3 vertices in 3D (no cells)
  std::vector<real_t> X = {0,0,0, 1,0,0, 0,1,0};
  std::vector<offset_t> offs = {0};
  std::vector<index_t> conn; std::vector<CellShape> shapes;
  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);

  std::vector<index_t> tri = {0,1,2};
  real_t area = PolyGeometry::polygon_area(mesh, tri);
  EXPECT_TRUE(approx(area, 0.5));

  auto c = PolyGeometry::polygon_centroid(mesh, tri);
  EXPECT_TRUE(approx3(c, {{1.0/3.0, 1.0/3.0, 0.0}}));
}

TEST_F(PolyGeometryTest, FaceAreaNormalFromMeshGeometryUsesPoly) {
  // Square in z=0 plane
  std::vector<real_t> X = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
  std::vector<offset_t> offs = {0, 4};
  std::vector<index_t> conn = {0,1,2,3};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Polygon; // treat as generic polygon face when used as a face later
  shapes[0].order = 1; shapes[0].num_corners = 4;

  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);

  // No faces set in MeshBase; instead, use area from vertices and normal from vertices helpers
  std::vector<index_t> verts = {0,1,2,3};
  real_t area = MeshGeometry::compute_area_from_vertices(mesh, verts);
  EXPECT_TRUE(approx(area, 1.0));

  // Normal should be along +z or -z; check magnitude 1 after normalization
  auto n_unnorm = PolyGeometry::polygon_normal(mesh, verts);
  auto n = MeshGeometry::normalize(n_unnorm);
  EXPECT_TRUE(approx(std::abs(n[2]), 1.0));
}

TEST_F(PolyGeometryTest, DegenerateEmptyAndSmallRaw) {
  std::vector<std::array<real_t,3>> empty;
  EXPECT_TRUE(approx(PolyGeometry::polygon_area(empty), 0.0));
  auto c0 = PolyGeometry::polygon_centroid(empty);
  EXPECT_TRUE(approx3(c0, {{0.0, 0.0, 0.0}}));

  std::vector<std::array<real_t,3>> one = {{ {2.0, 3.0, 4.0} }};
  EXPECT_TRUE(approx(PolyGeometry::polygon_area(one), 0.0));
  auto c1 = PolyGeometry::polygon_centroid(one);
  EXPECT_TRUE(approx3(c1, one[0]));

  std::vector<std::array<real_t,3>> two = {{ {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0} }};
  EXPECT_TRUE(approx(PolyGeometry::polygon_area(two), 0.0));
  auto c2 = PolyGeometry::polygon_centroid(two);
  EXPECT_TRUE(approx3(c2, {{1.0, 0.0, 0.0}}));
}

TEST_F(PolyGeometryTest, DegenerateCollinearRaw) {
  // Collinear triangle -> zero area; centroid falls back to average
  std::vector<std::array<real_t,3>> verts = {{ {0,0,0}, {1,0,0}, {2,0,0} }};
  EXPECT_TRUE(approx(PolyGeometry::polygon_area(verts), 0.0));
  auto c = PolyGeometry::polygon_centroid(verts);
  EXPECT_TRUE(approx3(c, {{1.0, 0.0, 0.0}}));
}

TEST_F(PolyGeometryTest, NormalSignFlipsWithOrientation) {
  std::vector<std::array<real_t,3>> verts = {{ {0,0,0}, {1,0,0}, {0,1,0} }};
  auto n1 = PolyGeometry::newell_normal(verts);
  std::reverse(verts.begin(), verts.end());
  auto n2 = PolyGeometry::newell_normal(verts);
  EXPECT_TRUE(approx3(MeshGeometry::normalize(n1), MeshGeometry::normalize({{-n2[0], -n2[1], -n2[2]}})));
}

TEST_F(PolyGeometryTest, Mesh2DSquareAreaCentroidNormal) {
  // 2D mesh: square in xy plane
  std::vector<real_t> X = {0,0, 1,0, 1,1, 0,1};
  std::vector<offset_t> offs = {0};
  std::vector<index_t> conn; std::vector<CellShape> shapes;
  MeshBase mesh;
  mesh.build_from_arrays(2, X, offs, conn, shapes);

  std::vector<index_t> verts = {0,1,2,3};
  real_t a = PolyGeometry::polygon_area(mesh, verts);
  EXPECT_TRUE(approx(a, 1.0));
  auto c = PolyGeometry::polygon_centroid(mesh, verts);
  EXPECT_TRUE(approx3(c, {{0.5, 0.5, 0.0}}));

  auto nu = PolyGeometry::polygon_normal(mesh, verts);
  auto n = MeshGeometry::normalize(nu);
  EXPECT_TRUE(approx(std::abs(n[2]), 1.0));
  // |n_u| = 2*area for Newell
  EXPECT_TRUE(approx(0.5 * MeshGeometry::magnitude(nu), a));
}

TEST_F(PolyGeometryTest, Mesh3DRotatedTriangleAreaMatchesTriangleArea) {
  // Triangle not aligned with axes
  std::vector<std::array<real_t,3>> raw = {{ {0,0,0}, {1,0,1}, {0,1,1} }};
  real_t tri_area = MeshGeometry::triangle_area(raw[0], raw[1], raw[2]);

  std::vector<real_t> X = {0,0,0, 1,0,1, 0,1,1};
  std::vector<offset_t> offs = {0};
  std::vector<index_t> conn; std::vector<CellShape> shapes;
  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);
  std::vector<index_t> tri = {0,1,2};
  real_t area = PolyGeometry::polygon_area(mesh, tri);
  EXPECT_TRUE(approx(area, tri_area));
}

TEST_F(PolyGeometryTest, MeshPentagonAreaConsistency) {
  // Convex pentagon in xy plane
  std::vector<std::array<real_t,3>> raw = {{ {0,0,0}, {2,0,0}, {3,1,0}, {1.5,2,0}, {0,1,0} }};
  real_t area_raw = PolyGeometry::polygon_area(raw);

  std::vector<real_t> X;
  X.reserve(raw.size()*3);
  for (auto& p : raw) { X.push_back(p[0]); X.push_back(p[1]); X.push_back(p[2]); }
  std::vector<offset_t> offs = {0};
  std::vector<index_t> conn; std::vector<CellShape> shapes;
  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);
  std::vector<index_t> ids = {0,1,2,3,4};
  real_t area_mesh = PolyGeometry::polygon_area(mesh, ids);
  EXPECT_TRUE(approx(area_mesh, area_raw));
}

TEST_F(PolyGeometryTest, MeshGeometryAreaCentroidHelpersUsePoly) {
  // Use MeshGeometry helpers on polygon with 5 vertices, ensure they route to PolyGeometry
  std::vector<std::array<real_t,3>> raw = {{ {0,0,0}, {2,0,0}, {3,1,0}, {1.5,2,0}, {0,1,0} }};
  std::vector<real_t> X;
  for (auto& p : raw) { X.push_back(p[0]); X.push_back(p[1]); X.push_back(p[2]); }
  std::vector<offset_t> offs = {0};
  std::vector<index_t> conn; std::vector<CellShape> shapes;
  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);
  std::vector<index_t> ids = {0,1,2,3,4};
  real_t a1 = PolyGeometry::polygon_area(mesh, ids);
  real_t a2 = MeshGeometry::compute_area_from_vertices(mesh, ids);
  EXPECT_TRUE(approx(a1, a2));

  auto c1 = PolyGeometry::polygon_centroid(mesh, ids);
  auto c2 = MeshGeometry::compute_centroid_from_vertices(mesh, ids);
  EXPECT_TRUE(approx3(c1, c2));
}

static MeshBase make_concave_L_prism_polyhedron_mesh(bool include_top_face = true) {
  // An "L-shaped" concave polygon extruded in z: height=1.
  //
  // Base polygon (z=0), CCW:
  //   (0,0) (2,0) (2,1) (1,1) (1,2) (0,2)
  // Top polygon (z=1): same vertices shifted in z.
  //
  // Analytic volume = area*height = 3*1 = 3.
  // Analytic centroid_xy = centroid(2x2 square minus 1x1 square at top-right):
  //   (4*(1,1) - 1*(1.5,1.5)) / 3 = (5/6, 5/6).
  std::vector<real_t> X = {
      0, 0, 0,  // 0
      2, 0, 0,  // 1
      2, 1, 0,  // 2
      1, 1, 0,  // 3
      1, 2, 0,  // 4
      0, 2, 0,  // 5
      0, 0, 1,  // 6
      2, 0, 1,  // 7
      2, 1, 1,  // 8
      1, 1, 1,  // 9
      1, 2, 1,  // 10
      0, 2, 1   // 11
  };

  std::vector<offset_t> offs = {0, 12};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Polyhedron;
  shapes[0].order = 1;
  shapes[0].num_corners = 12;

  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);

  const int n_faces = include_top_face ? 8 : 7;
  std::vector<CellShape> face_shapes;
  face_shapes.reserve(static_cast<size_t>(n_faces));

  CellShape poly;
  poly.family = CellFamily::Polygon;
  poly.order = 1;
  poly.num_corners = 6;
  face_shapes.push_back(poly); // bottom
  if (include_top_face) {
    face_shapes.push_back(poly); // top
  }

  CellShape quad;
  quad.family = CellFamily::Quad;
  quad.order = 1;
  quad.num_corners = 4;
  for (int i = 0; i < 6; ++i) face_shapes.push_back(quad); // sides

  std::vector<offset_t> face_offs;
  face_offs.reserve(static_cast<size_t>(n_faces + 1));
  face_offs.push_back(0);

  std::vector<index_t> face_conn;

  // bottom (concave polygon); intentionally start from a non-kernel vertex to exercise triangulation.
  face_conn.insert(face_conn.end(), {1, 2, 3, 4, 5, 0});
  face_offs.push_back(static_cast<offset_t>(face_conn.size()));

  if (include_top_face) {
    // top polygon
    face_conn.insert(face_conn.end(), {7, 8, 9, 10, 11, 6});
    face_offs.push_back(static_cast<offset_t>(face_conn.size()));
  }

  // side quads following base boundary edges
  const std::array<index_t, 6> b = {0, 1, 2, 3, 4, 5};
  for (int i = 0; i < 6; ++i) {
    const int j = (i + 1) % 6;
    const index_t v0 = b[static_cast<size_t>(i)];
    const index_t v1 = b[static_cast<size_t>(j)];
    const index_t w0 = v0 + 6;
    const index_t w1 = v1 + 6;
    face_conn.insert(face_conn.end(), {v0, v1, w1, w0});
    face_offs.push_back(static_cast<offset_t>(face_conn.size()));
  }

  std::vector<std::array<index_t,2>> face2cell(static_cast<size_t>(n_faces));
  for (auto& fc : face2cell) fc = {{0, INVALID_INDEX}};

  mesh.set_faces_from_arrays(face_shapes, face_offs, face_conn, face2cell);
  return mesh;
}

TEST_F(PolyGeometryTest, PolyhedronMassProperties_ConcaveLPrism) {
  const MeshBase mesh = make_concave_L_prism_polyhedron_mesh(true);

  const auto props = PolyGeometry::polyhedron_mass_properties(mesh, 0);
  ASSERT_TRUE(props.is_valid);
  EXPECT_NEAR(props.volume, 3.0, 1e-12);
  EXPECT_NEAR(props.centroid[0], 5.0 / 6.0, 1e-12);
  EXPECT_NEAR(props.centroid[1], 5.0 / 6.0, 1e-12);
  EXPECT_NEAR(props.centroid[2], 0.5, 1e-12);

  // MeshGeometry should route to the same robust polyhedron method.
  EXPECT_NEAR(MeshGeometry::cell_measure(mesh, 0), 3.0, 1e-12);
  const auto c = MeshGeometry::cell_centroid(mesh, 0);
  EXPECT_NEAR(c[0], 5.0 / 6.0, 1e-12);
  EXPECT_NEAR(c[1], 5.0 / 6.0, 1e-12);
  EXPECT_NEAR(c[2], 0.5, 1e-12);
}

TEST_F(PolyGeometryTest, PolyhedronMassProperties_InvalidOpenSurface) {
  // Missing top face -> boundary edges appear once, not watertight.
  const MeshBase mesh = make_concave_L_prism_polyhedron_mesh(false);
  const auto props = PolyGeometry::polyhedron_mass_properties(mesh, 0);
  EXPECT_FALSE(props.is_valid);
  EXPECT_TRUE(approx(props.volume, 0.0));
}


} // namespace test
} // namespace svmp
