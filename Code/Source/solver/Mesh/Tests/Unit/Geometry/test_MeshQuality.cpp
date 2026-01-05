/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"
#include "Geometry/MeshQuality.h"
#include "Geometry/PolyhedronTessellation.h"
#include "Core/MeshBase.h"
#include "Topology/CellShape.h"

#include <cmath>

namespace svmp {
namespace test {

namespace {
constexpr real_t tol = 1e-9;

MeshBase make_triangle(const std::array<std::array<real_t, 2>, 3>& pts) {
  std::vector<real_t> X_ref = {
      pts[0][0], pts[0][1],
      pts[1][0], pts[1][1],
      pts[2][0], pts[2][1],
  };
  std::vector<offset_t> offs = {0, 3};
  std::vector<index_t> conn = {0, 1, 2};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Triangle;
  shapes[0].order = 1;
  shapes[0].num_corners = 3;

  MeshBase mesh;
  mesh.build_from_arrays(2, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_tet(const std::array<std::array<real_t, 3>, 4>& pts) {
  std::vector<real_t> X_ref = {
      pts[0][0], pts[0][1], pts[0][2],
      pts[1][0], pts[1][1], pts[1][2],
      pts[2][0], pts[2][1], pts[2][2],
      pts[3][0], pts[3][1], pts[3][2],
  };
  std::vector<offset_t> offs = {0, 4};
  std::vector<index_t> conn = {0, 1, 2, 3};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Tetra;
  shapes[0].order = 1;
  shapes[0].num_corners = 4;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_hex(const std::array<std::array<real_t, 3>, 8>& pts) {
  std::vector<real_t> X_ref;
  X_ref.reserve(8 * 3);
  for (const auto& p : pts) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Hex;
  shapes[0].order = 1;
  shapes[0].num_corners = 8;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_wedge(const std::array<std::array<real_t, 3>, 6>& pts) {
  std::vector<real_t> X_ref;
  X_ref.reserve(6 * 3);
  for (const auto& p : pts) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, 6};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Wedge;
  shapes[0].order = 1;
  shapes[0].num_corners = 6;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_pyramid(const std::array<std::array<real_t, 3>, 5>& pts) {
  std::vector<real_t> X_ref;
  X_ref.reserve(5 * 3);
  for (const auto& p : pts) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, 5};
  std::vector<index_t> conn = {0, 1, 2, 3, 4};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Pyramid;
  shapes[0].order = 1;
  shapes[0].num_corners = 5;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_quad(const std::array<std::array<real_t, 3>, 4>& pts) {
  std::vector<real_t> X_ref;
  X_ref.reserve(4 * 3);
  for (const auto& p : pts) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, 4};
  std::vector<index_t> conn = {0, 1, 2, 3};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Quad;
  shapes[0].order = 1;
  shapes[0].num_corners = 4;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_two_hexes(const std::array<std::array<real_t, 3>, 8>& a,
                        const std::array<std::array<real_t, 3>, 8>& b) {
  std::vector<real_t> X_ref;
  X_ref.reserve(16 * 3);
  for (const auto& p : a) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }
  for (const auto& p : b) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, 8, 16};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<CellShape> shapes(2);
  for (auto& s : shapes) {
    s.family = CellFamily::Hex;
    s.order = 1;
    s.num_corners = 8;
  }

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_triangle_and_quad(const std::array<std::array<real_t, 3>, 3>& tri,
                                const std::array<std::array<real_t, 3>, 4>& quad) {
  std::vector<real_t> X_ref;
  X_ref.reserve(7 * 3);
  for (const auto& p : tri) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }
  for (const auto& p : quad) {
    X_ref.push_back(p[0]);
    X_ref.push_back(p[1]);
    X_ref.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, 3, 7};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6};

  std::vector<CellShape> shapes(2);
  shapes[0].family = CellFamily::Triangle;
  shapes[0].order = 1;
  shapes[0].num_corners = 3;
  shapes[1].family = CellFamily::Quad;
  shapes[1].order = 1;
  shapes[1].num_corners = 4;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase make_cube_polyhedron() {
  // Unit cube as a polyhedron with explicit quad faces.
  std::vector<real_t> X_ref = {
      0.0, 0.0, 0.0,  // 0
      1.0, 0.0, 0.0,  // 1
      1.0, 1.0, 0.0,  // 2
      0.0, 1.0, 0.0,  // 3
      0.0, 0.0, 1.0,  // 4
      1.0, 0.0, 1.0,  // 5
      1.0, 1.0, 1.0,  // 6
      0.0, 1.0, 1.0   // 7
  };

  std::vector<offset_t> offs = {0, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Polyhedron;
  shapes[0].order = 1;
  shapes[0].num_corners = 8;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);

  std::vector<CellShape> face_shapes(6);
  for (auto& fs : face_shapes) {
    fs.family = CellFamily::Quad;
    fs.order = 1;
    fs.num_corners = 4;
  }

  std::vector<offset_t> face_offs = {0, 4, 8, 12, 16, 20, 24};
  std::vector<index_t> face_conn = {
      0, 1, 2, 3,
      4, 5, 6, 7,
      0, 1, 5, 4,
      1, 2, 6, 5,
      2, 3, 7, 6,
      3, 0, 4, 7
  };

  std::vector<std::array<index_t, 2>> face2cell(6);
  for (auto& fc : face2cell) fc = {{0, INVALID_INDEX}};
  mesh.set_faces_from_arrays(face_shapes, face_offs, face_conn, face2cell);

  return mesh;
}
} // namespace

TEST(MeshQualityTest, SkewnessTriangle) {
  const real_t s3 = std::sqrt(3.0);
  MeshBase equilateral = make_triangle({{{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.5 * s3}}});
  MeshBase right = make_triangle({{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}});

  real_t s_eq = MeshQuality::compute(equilateral, 0, "skewness");
  real_t s_rt = MeshQuality::compute(right, 0, "skewness");

  EXPECT_GE(s_eq, 0.0);
  EXPECT_LE(s_eq, 1.0);
  EXPECT_LT(s_eq, 1e-6);

  EXPECT_NEAR(s_rt, 0.5, tol);
}

TEST(MeshQualityTest, SkewnessTetra) {
  const real_t s3 = std::sqrt(3.0);
  const real_t s23 = std::sqrt(2.0 / 3.0);

  // Regular tetra with unit edges.
  MeshBase regular = make_tet({{{0.0, 0.0, 0.0},
                                {1.0, 0.0, 0.0},
                                {0.5, 0.5 * s3, 0.0},
                                {0.5, s3 / 6.0, s23}}});

  auto pts_skew = std::array<std::array<real_t, 3>, 4>{{{0.0, 0.0, 0.0},
                                                        {2.0, 0.0, 0.0},  // stretch one edge
                                                        {0.5, 0.5 * s3, 0.0},
                                                        {0.5, s3 / 6.0, s23}}};
  MeshBase skewed = make_tet(pts_skew);

  real_t s_reg = MeshQuality::compute(regular, 0, "skewness");
  real_t s_skw = MeshQuality::compute(skewed, 0, "skewness");

  EXPECT_GE(s_reg, 0.0);
  EXPECT_LE(s_reg, 1.0);
  EXPECT_LT(s_reg, 1e-6);

  EXPECT_GT(s_skw, 0.3);
  EXPECT_LE(s_skw, 1.0);
}

TEST(MeshQualityTest, AnglesTetraAndHex) {
  const real_t s3 = std::sqrt(3.0);
  const real_t s23 = std::sqrt(2.0 / 3.0);
  MeshBase regular_tet = make_tet({{{0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0},
                                    {0.5, 0.5 * s3, 0.0},
                                    {0.5, s3 / 6.0, s23}}});
  EXPECT_NEAR(MeshQuality::compute(regular_tet, 0, "min_angle"), 60.0, 1e-6);
  EXPECT_NEAR(MeshQuality::compute(regular_tet, 0, "max_angle"), 60.0, 1e-6);

  MeshBase unit_hex = make_hex({{{0.0, 0.0, 0.0},
                                 {1.0, 0.0, 0.0},
                                 {1.0, 1.0, 0.0},
                                 {0.0, 1.0, 0.0},
                                 {0.0, 0.0, 1.0},
                                 {1.0, 0.0, 1.0},
                                 {1.0, 1.0, 1.0},
                                 {0.0, 1.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(unit_hex, 0, "min_angle"), 90.0, 1e-6);
  EXPECT_NEAR(MeshQuality::compute(unit_hex, 0, "max_angle"), 90.0, 1e-6);
}

TEST(MeshQualityTest, JacobianAndScaledJacobian_IdentityHexAndInvertedTet) {
  MeshBase unit_hex = make_hex({{{-1.0, -1.0, -1.0},
                                 {1.0, -1.0, -1.0},
                                 {1.0, 1.0, -1.0},
                                 {-1.0, 1.0, -1.0},
                                 {-1.0, -1.0, 1.0},
                                 {1.0, -1.0, 1.0},
                                 {1.0, 1.0, 1.0},
                                 {-1.0, 1.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(unit_hex, 0, "jacobian"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(unit_hex, 0, "scaled_jacobian"), 1.0, 1e-12);

  // Identity reference tetra.
  MeshBase ref_tet = make_tet({{{0.0, 0.0, 0.0},
                                {1.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(ref_tet, 0, "jacobian"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(ref_tet, 0, "scaled_jacobian"), 1.0, 1e-12);

  // Inverted tetra (swap two vertices).
  MeshBase inv_tet = make_tet({{{0.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0},
                                {1.0, 0.0, 0.0},
                                {0.0, 0.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(inv_tet, 0, "jacobian"), 0.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(inv_tet, 0, "scaled_jacobian"), 0.0, 1e-12);
}

TEST(MeshQualityTest, JacobianScaledJacobianAndConditionNumber_IdentityWedgeAndPyramid) {
  MeshBase ref_wedge = make_wedge({{{0.0, 0.0, -1.0},
                                    {1.0, 0.0, -1.0},
                                    {0.0, 1.0, -1.0},
                                    {0.0, 0.0, 1.0},
                                    {1.0, 0.0, 1.0},
                                    {0.0, 1.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(ref_wedge, 0, "jacobian"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(ref_wedge, 0, "scaled_jacobian"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(ref_wedge, 0, "condition_number"), 1.0, 1e-12);

  MeshBase ref_pyr = make_pyramid({{{-1.0, -1.0, 0.0},
                                    {1.0, -1.0, 0.0},
                                    {1.0, 1.0, 0.0},
                                    {-1.0, 1.0, 0.0},
                                    {0.0, 0.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(ref_pyr, 0, "jacobian"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(ref_pyr, 0, "scaled_jacobian"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(ref_pyr, 0, "condition_number"), 1.0, 1e-12);
}

TEST(MeshQualityTest, EdgeRatio_WedgeAndPyramid) {
  MeshBase ref_wedge = make_wedge({{{0.0, 0.0, -1.0},
                                    {1.0, 0.0, -1.0},
                                    {0.0, 1.0, -1.0},
                                    {0.0, 0.0, 1.0},
                                    {1.0, 0.0, 1.0},
                                    {0.0, 1.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(ref_wedge, 0, "edge_ratio"), 2.0, 1e-12);

  MeshBase ref_pyr = make_pyramid({{{-1.0, -1.0, 0.0},
                                    {1.0, -1.0, 0.0},
                                    {1.0, 1.0, 0.0},
                                    {-1.0, 1.0, 0.0},
                                    {0.0, 0.0, 1.0}}});
  const real_t expected = 2.0 / std::sqrt(3.0);
  EXPECT_NEAR(MeshQuality::compute(ref_pyr, 0, "edge_ratio"), expected, 1e-12);
}

TEST(MeshQualityTest, DistortionReducesScaledJacobian_WedgeAndPyramid) {
  // Shear y by +0.5 * x: J columns become non-orthogonal.
  MeshBase shear_wedge = make_wedge({{{0.0, 0.0, -1.0},
                                      {1.0, 0.5, -1.0},
                                      {0.0, 1.0, -1.0},
                                      {0.0, 0.0, 1.0},
                                      {1.0, 0.5, 1.0},
                                      {0.0, 1.0, 1.0}}});
  const real_t sj_w = MeshQuality::compute(shear_wedge, 0, "scaled_jacobian");
  const real_t cn_w = MeshQuality::compute(shear_wedge, 0, "condition_number");
  EXPECT_GT(sj_w, 0.0);
  EXPECT_LT(sj_w, 1.0 - 1e-6);
  EXPECT_GT(cn_w, 0.0);
  EXPECT_LT(cn_w, 1.0 - 1e-6);

  MeshBase shear_pyr = make_pyramid({{{-1.0, -1.5, 0.0},  // y' = y + 0.5 x
                                      {1.0, -0.5, 0.0},
                                      {1.0, 1.5, 0.0},
                                      {-1.0, 0.5, 0.0},
                                      {0.0, 0.0, 1.0}}});
  const real_t sj_p = MeshQuality::compute(shear_pyr, 0, "scaled_jacobian");
  const real_t cn_p = MeshQuality::compute(shear_pyr, 0, "condition_number");
  EXPECT_GT(sj_p, 0.0);
  EXPECT_LT(sj_p, 1.0 - 1e-6);
  EXPECT_GT(cn_p, 0.0);
  EXPECT_LT(cn_p, 1.0 - 1e-6);
}

TEST(MeshQualityTest, PolyhedronMetricsMatchTetAggregation) {
  const MeshBase mesh = make_cube_polyhedron();

  // Polyhedron metrics are defined by tetrahedralization and worst-subtet reduction.
  const auto tets = PolyhedronTessellation::convex_star_tets(mesh, 0);
  ASSERT_FALSE(tets.empty());

  auto reduce_min = [&](MeshQuality::Metric metric) {
    real_t qmin = 1e300;
    for (const auto& tet : tets) {
      MeshBase tmp;
      std::vector<real_t> X_ref;
      X_ref.reserve(12);
      for (const auto& p : tet.vertices) {
        X_ref.push_back(p[0]);
        X_ref.push_back(p[1]);
        X_ref.push_back(p[2]);
      }
      std::vector<offset_t> offs = {0, 4};
      std::vector<index_t> conn = {0, 1, 2, 3};
      std::vector<CellShape> shapes(1);
      shapes[0].family = CellFamily::Tetra;
      shapes[0].order = 1;
      shapes[0].num_corners = 4;
      tmp.build_from_arrays(3, X_ref, offs, conn, shapes);
      qmin = std::min(qmin, MeshQuality::compute(tmp, 0, metric));
    }
    return qmin;
  };

  EXPECT_NEAR(MeshQuality::compute(mesh, 0, MeshQuality::Metric::Jacobian), reduce_min(MeshQuality::Metric::Jacobian), tol);
  EXPECT_NEAR(MeshQuality::compute(mesh, 0, MeshQuality::Metric::ScaledJacobian), reduce_min(MeshQuality::Metric::ScaledJacobian), tol);
  EXPECT_NEAR(MeshQuality::compute(mesh, 0, MeshQuality::Metric::ConditionNumber), reduce_min(MeshQuality::Metric::ConditionNumber), tol);
  EXPECT_NEAR(MeshQuality::compute(mesh, 0, MeshQuality::Metric::ShapeIndex), reduce_min(MeshQuality::Metric::ShapeIndex), tol);
}

TEST(MeshQualityTest, ConditionStretchAndShapeIndex_HexStretch) {
  // Axis-aligned boxes are affine; quality metrics are constant across samples.
  MeshBase unit_hex = make_hex({{{0.0, 0.0, 0.0},
                                 {1.0, 0.0, 0.0},
                                 {1.0, 1.0, 0.0},
                                 {0.0, 1.0, 0.0},
                                 {0.0, 0.0, 1.0},
                                 {1.0, 0.0, 1.0},
                                 {1.0, 1.0, 1.0},
                                 {0.0, 1.0, 1.0}}});
  EXPECT_NEAR(MeshQuality::compute(unit_hex, 0, "condition_number"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(unit_hex, 0, "stretch"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(unit_hex, 0, "shape_index"), 1.0, 1e-12);

  // Stretch by 2x in x.
  MeshBase stretch_hex = make_hex({{{0.0, 0.0, 0.0},
                                    {2.0, 0.0, 0.0},
                                    {2.0, 1.0, 0.0},
                                    {0.0, 1.0, 0.0},
                                    {0.0, 0.0, 1.0},
                                    {2.0, 0.0, 1.0},
                                    {2.0, 1.0, 1.0},
                                    {0.0, 1.0, 1.0}}});

  EXPECT_NEAR(MeshQuality::compute(stretch_hex, 0, "scaled_jacobian"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(stretch_hex, 0, "condition_number"), 0.5, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(stretch_hex, 0, "stretch"), std::pow(2.0, -2.0 / 3.0), 1e-9);
  EXPECT_NEAR(MeshQuality::compute(stretch_hex, 0, "shape_index"), 0.7937005259840998, 1e-9);
}

TEST(MeshQualityTest, QuadWarpageTaperAndDiagonalRatio) {
  MeshBase planar = make_quad({{{0.0, 0.0, 0.0},
                                {1.0, 0.0, 0.0},
                                {1.0, 1.0, 0.0},
                                {0.0, 1.0, 0.0}}});
  EXPECT_NEAR(MeshQuality::compute(planar, 0, "warpage"), 0.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(planar, 0, "taper"), 1.0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(planar, 0, "diagonal_ratio"), 1.0, 1e-12);

  MeshBase warped = make_quad({{{0.0, 0.0, 0.0},
                                {1.0, 0.0, 0.0},
                                {1.0, 1.0, 0.2},
                                {0.0, 1.0, 0.0}}});
  EXPECT_GT(MeshQuality::compute(warped, 0, "warpage"), 0.0);

  // Parallelogram -> taper quality 1.
  MeshBase parallelogram = make_quad({{{0.0, 0.0, 0.0},
                                       {2.0, 0.0, 0.0},
                                       {3.0, 1.0, 0.0},
                                       {1.0, 1.0, 0.0}}});
  EXPECT_NEAR(MeshQuality::compute(parallelogram, 0, "taper"), 1.0, 1e-12);

  // Trapezoid -> taper < 1.
  MeshBase trapezoid = make_quad({{{0.0, 0.0, 0.0},
                                   {2.0, 0.0, 0.0},
                                   {1.5, 1.0, 0.0},
                                   {0.0, 1.0, 0.0}}});
  EXPECT_LT(MeshQuality::compute(trapezoid, 0, "taper"), 0.99);

  // Unequal diagonals -> diagonal_ratio < 1.
  MeshBase diag = make_quad({{{0.0, 0.0, 0.0},
                              {2.0, 0.0, 0.0},
                              {2.0, 1.0, 0.0},
                              {0.0, 1.5, 0.0}}});
  EXPECT_LT(MeshQuality::compute(diag, 0, "diagonal_ratio"), 1.0 - 1e-6);
}

TEST(MeshQualityTest, RelativeSizeSquaredAndShapeAndSize) {
  // Two cubes of different size; shape quality is 1 for both, size quality differs.
  MeshBase mesh = make_two_hexes({{{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0},
                                   {1.0, 1.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 0.0, 1.0},
                                   {1.0, 0.0, 1.0},
                                   {1.0, 1.0, 1.0},
                                   {0.0, 1.0, 1.0}}},
                                 {{{10.0, 0.0, 0.0},
                                   {12.0, 0.0, 0.0},
                                   {12.0, 2.0, 0.0},
                                   {10.0, 2.0, 0.0},
                                   {10.0, 0.0, 2.0},
                                   {12.0, 0.0, 2.0},
                                   {12.0, 2.0, 2.0},
                                   {10.0, 2.0, 2.0}}});

  const real_t rs0 = MeshQuality::compute(mesh, 0, "relative_size_squared");
  const real_t rs1 = MeshQuality::compute(mesh, 1, "relative_size_squared");
  EXPECT_NEAR(rs0, (1.0 / 4.5) * (1.0 / 4.5), 1e-12);
  EXPECT_NEAR(rs1, 0.31640625, 1e-12); // (4.5/8)^2

  EXPECT_NEAR(MeshQuality::compute(mesh, 0, "shape_and_size"), rs0, 1e-12);
  EXPECT_NEAR(MeshQuality::compute(mesh, 1, "shape_and_size"), rs1, 1e-12);
}

TEST(MeshQualityTest, StatisticsRespectMetricDirectionAndFamily) {
  const real_t s3 = std::sqrt(3.0);
  MeshBase mixed = make_triangle_and_quad(
      {{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.5 * s3, 0.0}}}, // equilateral
      {{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.3, 0.0}, {0.0, 1.0, 0.0}}} // min angle ~31 deg
  );

  auto stats = MeshQuality::compute_statistics(mixed, MeshQuality::Metric::MinAngle);
  EXPECT_EQ(stats.count_poor, 1u);      // quad poor under quad thresholds
  EXPECT_EQ(stats.count_good, 1u);      // triangle good/excellent
  EXPECT_EQ(stats.count_excellent, 1u); // triangle excellent

  // Lower-is-better metric counts: equilateral is excellent; right triangle is acceptable (not good/excellent).
  MeshBase equilateral = make_triangle({{{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.5 * s3}}});
  MeshBase right = make_triangle({{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}});

  std::vector<real_t> X_ref;
  X_ref.reserve(6 * 2);
  const auto& Xa = equilateral.X_ref();
  const auto& Xb = right.X_ref();
  X_ref.insert(X_ref.end(), Xa.begin(), Xa.end());
  X_ref.insert(X_ref.end(), Xb.begin(), Xb.end());
  std::vector<offset_t> offs = {0, 3, 6};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5};
  std::vector<CellShape> shapes(2);
  for (auto& s : shapes) {
    s.family = CellFamily::Triangle;
    s.order = 1;
    s.num_corners = 3;
  }
  MeshBase two_tris;
  two_tris.build_from_arrays(2, X_ref, offs, conn, shapes);

  auto sk = MeshQuality::compute_statistics(two_tris, MeshQuality::Metric::Skewness);
  EXPECT_EQ(sk.count_poor, 0u);
  EXPECT_EQ(sk.count_excellent, 1u);
}

TEST(MeshQualityTest, IsAcceptableUsesCorrectThresholds) {
  MeshBase unit_hex = make_hex({{{0.0, 0.0, 0.0},
                                 {1.0, 0.0, 0.0},
                                 {1.0, 1.0, 0.0},
                                 {0.0, 1.0, 0.0},
                                 {0.0, 0.0, 1.0},
                                 {1.0, 0.0, 1.0},
                                 {1.0, 1.0, 1.0},
                                 {0.0, 1.0, 1.0}}});
  EXPECT_TRUE(MeshQuality::is_acceptable(unit_hex, 0, MeshQuality::Metric::MaxAngle));

  // Edge ratio is lower-is-better; this should be unacceptable under default thresholds.
  MeshBase stretch_hex = make_hex({{{0.0, 0.0, 0.0},
                                    {10.0, 0.0, 0.0},
                                    {10.0, 1.0, 0.0},
                                    {0.0, 1.0, 0.0},
                                    {0.0, 0.0, 1.0},
                                    {10.0, 0.0, 1.0},
                                    {10.0, 1.0, 1.0},
                                    {0.0, 1.0, 1.0}}});
  EXPECT_FALSE(MeshQuality::is_acceptable(stretch_hex, 0, MeshQuality::Metric::EdgeRatio));
}

} // namespace test
} // namespace svmp
