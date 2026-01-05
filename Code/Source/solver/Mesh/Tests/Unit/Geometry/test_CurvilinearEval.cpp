/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"

#include "Geometry/CurvilinearEval.h"
#include "Core/MeshBase.h"
#include "Topology/CellShape.h"
#include "Topology/CellTopology.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace svmp {
namespace test {

namespace {

constexpr real_t kTol = 1e-10;

CellShape make_shape(CellFamily family, int order) {
  CellShape s;
  s.family = family;
  s.order = order;
  switch (family) {
    case CellFamily::Line: s.num_corners = 2; break;
    case CellFamily::Triangle: s.num_corners = 3; break;
    case CellFamily::Quad: s.num_corners = 4; break;
    case CellFamily::Tetra: s.num_corners = 4; break;
    case CellFamily::Hex: s.num_corners = 8; break;
    case CellFamily::Wedge: s.num_corners = 6; break;
    case CellFamily::Pyramid: s.num_corners = 5; break;
    default: s.num_corners = 0; break;
  }
  return s;
}

ParametricPoint corner_param(CellFamily family, int corner_id) {
  switch (family) {
    case CellFamily::Line:
      return (corner_id == 0) ? ParametricPoint{-1, 0, 0} : ParametricPoint{1, 0, 0};
    case CellFamily::Triangle: {
      switch (corner_id) {
        case 0: return {0, 0, 0};
        case 1: return {1, 0, 0};
        case 2: return {0, 1, 0};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Quad: {
      switch (corner_id) {
        case 0: return {-1, -1, 0};
        case 1: return {1, -1, 0};
        case 2: return {1, 1, 0};
        case 3: return {-1, 1, 0};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Tetra: {
      switch (corner_id) {
        case 0: return {0, 0, 0};
        case 1: return {1, 0, 0};
        case 2: return {0, 1, 0};
        case 3: return {0, 0, 1};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Hex: {
      switch (corner_id) {
        case 0: return {-1, -1, -1};
        case 1: return {1, -1, -1};
        case 2: return {1, 1, -1};
        case 3: return {-1, 1, -1};
        case 4: return {-1, -1, 1};
        case 5: return {1, -1, 1};
        case 6: return {1, 1, 1};
        case 7: return {-1, 1, 1};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Wedge: {
      switch (corner_id) {
        case 0: return {0, 0, -1};
        case 1: return {1, 0, -1};
        case 2: return {0, 1, -1};
        case 3: return {0, 0, 1};
        case 4: return {1, 0, 1};
        case 5: return {0, 1, 1};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Pyramid: {
      switch (corner_id) {
        case 0: return {-1, -1, 0};
        case 1: return {1, -1, 0};
        case 2: return {1, 1, 0};
        case 3: return {-1, 1, 0};
        case 4: return {0, 0, 1};
        default: return {0, 0, 0};
      }
    }
    default:
      return {0, 0, 0};
  }
}

ParametricPoint lerp(const ParametricPoint& a, const ParametricPoint& b, real_t t) {
  return {(1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1], (1 - t) * a[2] + t * b[2]};
}

ParametricPoint bilerp(const ParametricPoint& a, const ParametricPoint& b,
                       const ParametricPoint& c, const ParametricPoint& d,
                       real_t u, real_t v) {
  const real_t w00 = (1 - u) * (1 - v);
  const real_t w10 = u * (1 - v);
  const real_t w11 = u * v;
  const real_t w01 = (1 - u) * v;
  return {w00 * a[0] + w10 * b[0] + w11 * c[0] + w01 * d[0],
          w00 * a[1] + w10 * b[1] + w11 * c[1] + w01 * d[1],
          w00 * a[2] + w10 * b[2] + w11 * c[2] + w01 * d[2]};
}

std::vector<ParametricPoint> vtk_lagrange_nodes(CellFamily family, int p) {
  const auto pat = CellTopology::high_order_pattern(family, p, CellTopology::HighOrderKind::Lagrange);
  const auto eview = CellTopology::get_edges_view(family);
  const auto fview = CellTopology::get_oriented_boundary_faces_view(family);

  std::vector<ParametricPoint> nodes;
  nodes.reserve(pat.sequence.size());

  for (const auto& role : pat.sequence) {
    switch (role.role) {
      case CellTopology::HONodeRole::Corner:
        nodes.push_back(corner_param(family, role.idx0));
        break;

      case CellTopology::HONodeRole::Edge: {
        const int ei = role.idx0;
        const int k = role.idx1;
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const real_t t = static_cast<real_t>(k) / static_cast<real_t>(p);
        nodes.push_back(lerp(corner_param(family, a), corner_param(family, b), t));
        break;
      }

      case CellTopology::HONodeRole::Face: {
        const int fi = role.idx0;
        const int i = role.idx1;
        const int j = role.idx2;

        if (family == CellFamily::Triangle) {
          nodes.push_back({static_cast<real_t>(i) / static_cast<real_t>(p),
                           static_cast<real_t>(j) / static_cast<real_t>(p), 0});
          break;
        }
        if (family == CellFamily::Quad) {
          nodes.push_back({-1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(p),
                           -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(p), 0});
          break;
        }

        const int b = fview.offsets[fi];
        const int e = fview.offsets[fi + 1];
        const int fv = e - b;
        if (fv == 3) {
          const int v0 = fview.indices[b + 0];
          const int v1 = fview.indices[b + 1];
          const int v2 = fview.indices[b + 2];
          const real_t w1 = static_cast<real_t>(i) / static_cast<real_t>(p);
          const real_t w2 = static_cast<real_t>(j) / static_cast<real_t>(p);
          const real_t w0 = 1.0 - w1 - w2;
          const auto A = corner_param(family, v0);
          const auto B = corner_param(family, v1);
          const auto C = corner_param(family, v2);
          nodes.push_back({w0 * A[0] + w1 * B[0] + w2 * C[0],
                           w0 * A[1] + w1 * B[1] + w2 * C[1],
                           w0 * A[2] + w1 * B[2] + w2 * C[2]});
        } else if (fv == 4) {
          const int v0 = fview.indices[b + 0];
          const int v1 = fview.indices[b + 1];
          const int v2 = fview.indices[b + 2];
          const int v3 = fview.indices[b + 3];
          const real_t u = static_cast<real_t>(i) / static_cast<real_t>(p);
          const real_t v = static_cast<real_t>(j) / static_cast<real_t>(p);
          nodes.push_back(bilerp(corner_param(family, v0),
                                 corner_param(family, v1),
                                 corner_param(family, v2),
                                 corner_param(family, v3),
                                 u, v));
        } else {
          nodes.push_back({0, 0, 0});
        }
        break;
      }

      case CellTopology::HONodeRole::Volume: {
        if (family == CellFamily::Hex) {
          nodes.push_back({-1.0 + 2.0 * static_cast<real_t>(role.idx0) / static_cast<real_t>(p),
                           -1.0 + 2.0 * static_cast<real_t>(role.idx1) / static_cast<real_t>(p),
                           -1.0 + 2.0 * static_cast<real_t>(role.idx2) / static_cast<real_t>(p)});
        } else if (family == CellFamily::Tetra) {
          nodes.push_back({static_cast<real_t>(role.idx0) / static_cast<real_t>(p),
                           static_cast<real_t>(role.idx1) / static_cast<real_t>(p),
                           static_cast<real_t>(role.idx2) / static_cast<real_t>(p)});
        } else if (family == CellFamily::Wedge) {
          nodes.push_back({static_cast<real_t>(role.idx0) / static_cast<real_t>(p),
                           static_cast<real_t>(role.idx1) / static_cast<real_t>(p),
                           -1.0 + 2.0 * static_cast<real_t>(role.idx2) / static_cast<real_t>(p)});
        } else if (family == CellFamily::Pyramid) {
          const int i = role.idx0;
          const int j = role.idx1;
          const int k = role.idx2;
          const real_t z = static_cast<real_t>(k) / static_cast<real_t>(p);
          const real_t scale = 1.0 - z;
          const int n = (p + 1) - k;
          const int m = n - 1;
          const real_t uu = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(m);
          const real_t vv = -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(m);
          nodes.push_back({scale * uu, scale * vv, z});
        } else {
          nodes.push_back({0, 0, 0});
        }
        break;
      }
    }
  }

  return nodes;
}

std::vector<ParametricPoint> vtk_serendipity_nodes_quadratic(CellFamily family) {
  std::vector<ParametricPoint> nodes;
  const auto eview = CellTopology::get_edges_view(family);

  int nc = 0;
  switch (family) {
    case CellFamily::Quad: nc = 4; break;
    case CellFamily::Hex: nc = 8; break;
    case CellFamily::Wedge: nc = 6; break;
    case CellFamily::Pyramid: nc = 5; break;
    default: nc = 0; break;
  }

  for (int c = 0; c < nc; ++c) nodes.push_back(corner_param(family, c));

  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = eview.pairs_flat[2 * ei + 0];
    const int b = eview.pairs_flat[2 * ei + 1];
    nodes.push_back(lerp(corner_param(family, a), corner_param(family, b), 0.5));
  }

  return nodes;
}

MeshBase make_identity_mesh(const CellShape& shape, const std::vector<ParametricPoint>& nodes) {
  std::vector<real_t> X;
  X.reserve(nodes.size() * 3);
  for (const auto& p : nodes) {
    X.push_back(p[0]);
    X.push_back(p[1]);
    X.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, static_cast<offset_t>(nodes.size())};
  std::vector<index_t> conn(nodes.size());
  std::iota(conn.begin(), conn.end(), 0);
  std::vector<CellShape> shapes = {shape};

  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);
  return mesh;
}

} // namespace

TEST(CurvilinearEvalTest, PartitionOfUnity_LagrangeP3) {
  const int p = 3;

  {
    const CellShape tri = make_shape(CellFamily::Triangle, p);
    const size_t n = static_cast<size_t>(tri.expected_vertices());
    const auto sf = CurvilinearEvaluator::evaluate_shape_functions(tri, n, {0.2, 0.3, 0.0});
    real_t sum = 0.0;
    for (real_t Ni : sf.N) sum += Ni;
    EXPECT_NEAR(sum, 1.0, 1e-12);
  }

  {
    const CellShape hex = make_shape(CellFamily::Hex, p);
    const size_t n = static_cast<size_t>(hex.expected_vertices());
    const auto sf = CurvilinearEvaluator::evaluate_shape_functions(hex, n, {0.1, -0.2, 0.3});
    real_t sum = 0.0;
    for (real_t Ni : sf.N) sum += Ni;
    EXPECT_NEAR(sum, 1.0, 1e-12);
  }
}

TEST(CurvilinearEvalTest, DerivativePartitionOfUnity_LagrangeP3) {
  const int p = 3;
  struct Case {
    CellFamily family;
    ParametricPoint xi;
    int pdim;
  };
  const std::vector<Case> cases = {
      {CellFamily::Triangle, {0.2, 0.3, 0.0}, 2},
      {CellFamily::Hex, {0.1, -0.2, 0.3}, 3},
      {CellFamily::Wedge, {0.2, 0.2, 0.0}, 3},
      {CellFamily::Pyramid, {0.0, 0.0, 0.3}, 3},
  };

  for (const auto& c : cases) {
    const CellShape shape = make_shape(c.family, p);
    const auto nodes = vtk_lagrange_nodes(c.family, p);
    const size_t n = nodes.size();
    ASSERT_EQ(n, static_cast<size_t>(shape.expected_vertices()));
    const auto sf = CurvilinearEvaluator::evaluate_shape_functions(shape, n, c.xi);

    for (int d = 0; d < c.pdim; ++d) {
      real_t sum = 0.0;
      for (const auto& g : sf.dN_dxi) sum += g[static_cast<size_t>(d)];
      EXPECT_NEAR(sum, 0.0, 1e-9) << "family=" << static_cast<int>(c.family) << " d=" << d;
    }
  }
}

TEST(CurvilinearEvalTest, FiniteDifference_dN_dxi_LagrangeP3) {
  const int p = 3;
  const real_t eps = 1e-6;
  struct Case {
    CellFamily family;
    ParametricPoint xi;
  };
  const std::vector<Case> cases = {
      {CellFamily::Hex, {0.1, -0.2, 0.3}},
      {CellFamily::Pyramid, {0.0, 0.0, 0.3}},
  };

  for (const auto& c : cases) {
    const CellShape shape = make_shape(c.family, p);
    const auto nodes = vtk_lagrange_nodes(c.family, p);
    const size_t n = nodes.size();
    ASSERT_EQ(n, static_cast<size_t>(shape.expected_vertices()));

    const auto sf0 = CurvilinearEvaluator::evaluate_shape_functions(shape, n, c.xi);
    ASSERT_EQ(sf0.N.size(), n);
    ASSERT_EQ(sf0.dN_dxi.size(), n);

    for (int d = 0; d < 3; ++d) {
      ParametricPoint xip = c.xi;
      ParametricPoint xim = c.xi;
      xip[static_cast<size_t>(d)] += eps;
      xim[static_cast<size_t>(d)] -= eps;
      const auto sfp = CurvilinearEvaluator::evaluate_shape_functions(shape, n, xip);
      const auto sfm = CurvilinearEvaluator::evaluate_shape_functions(shape, n, xim);
      ASSERT_EQ(sfp.N.size(), n);
      ASSERT_EQ(sfm.N.size(), n);

      real_t max_err = 0.0;
      for (size_t i = 0; i < n; ++i) {
        const real_t fd = (sfp.N[i] - sfm.N[i]) / (2.0 * eps);
        max_err = std::max(max_err, std::abs(fd - sf0.dN_dxi[i][static_cast<size_t>(d)]));
      }
      EXPECT_LT(max_err, 5e-6) << "family=" << static_cast<int>(c.family) << " d=" << d;
    }
  }
}

TEST(CurvilinearEvalTest, JacobianMatchesFiniteDifference_IdentityMapping_LagrangeP3) {
  const int p = 3;
  const real_t eps = 1e-6;

  struct Case { CellFamily family; ParametricPoint xi; };
  const std::vector<Case> cases = {
      {CellFamily::Tetra, {0.2, 0.2, 0.2}},
      {CellFamily::Hex, {0.1, -0.2, 0.3}},
      {CellFamily::Wedge, {0.2, 0.2, 0.0}},
      {CellFamily::Pyramid, {0.0, 0.0, 0.3}},
  };

  for (const auto& c : cases) {
    const CellShape shape = make_shape(c.family, p);
    const auto nodes = vtk_lagrange_nodes(c.family, p);
    ASSERT_EQ(nodes.size(), static_cast<size_t>(shape.expected_vertices()));
    const MeshBase mesh = make_identity_mesh(shape, nodes);

    const auto eval0 = CurvilinearEvaluator::evaluate_geometry(mesh, 0, c.xi);
    ASSERT_TRUE(eval0.is_valid);
    ASSERT_EQ(eval0.jacobian.parametric_dim, 3);

    for (int k = 0; k < 3; ++k) {
      for (int d = 0; d < 3; ++d) {
        const real_t expect = (k == d) ? 1.0 : 0.0;
        EXPECT_NEAR(eval0.jacobian.matrix[static_cast<size_t>(k)][static_cast<size_t>(d)], expect, 5e-9)
            << "family=" << static_cast<int>(c.family) << " k=" << k << " d=" << d;
      }
    }

    for (int d = 0; d < 3; ++d) {
      ParametricPoint xip = c.xi;
      ParametricPoint xim = c.xi;
      xip[static_cast<size_t>(d)] += eps;
      xim[static_cast<size_t>(d)] -= eps;
      const auto evalp = CurvilinearEvaluator::evaluate_geometry(mesh, 0, xip);
      const auto evalm = CurvilinearEvaluator::evaluate_geometry(mesh, 0, xim);
      ASSERT_TRUE(evalp.is_valid);
      ASSERT_TRUE(evalm.is_valid);

      for (int k = 0; k < 3; ++k) {
        const real_t fd = (evalp.coordinates[static_cast<size_t>(k)] - evalm.coordinates[static_cast<size_t>(k)]) / (2.0 * eps);
        EXPECT_NEAR(fd, eval0.jacobian.matrix[static_cast<size_t>(k)][static_cast<size_t>(d)], 1e-6)
            << "family=" << static_cast<int>(c.family) << " k=" << k << " d=" << d;
      }
    }
  }
}

TEST(CurvilinearEvalTest, NodalKronecker_TriangleAndHex_P3) {
  const int p = 3;

  for (CellFamily family : {CellFamily::Triangle, CellFamily::Hex}) {
    const CellShape shape = make_shape(family, p);
    const auto nodes = vtk_lagrange_nodes(family, p);
    const size_t n = nodes.size();
    ASSERT_EQ(n, static_cast<size_t>(shape.expected_vertices()));

    for (size_t j = 0; j < n; ++j) {
      const auto sf = CurvilinearEvaluator::evaluate_shape_functions(shape, n, nodes[j]);
      ASSERT_EQ(sf.N.size(), n);
      for (size_t i = 0; i < n; ++i) {
        const real_t expect = (i == j) ? 1.0 : 0.0;
        EXPECT_NEAR(sf.N[i], expect, kTol) << "family=" << static_cast<int>(family)
                                          << " i=" << i << " j=" << j;
      }
    }
  }
}

TEST(CurvilinearEvalTest, IdentityMapping_JacobianAndInverseMap_LagrangeP3) {
  const int p = 3;

  for (CellFamily family : {CellFamily::Tetra, CellFamily::Hex, CellFamily::Wedge, CellFamily::Pyramid}) {
    const CellShape shape = make_shape(family, p);
    const auto nodes = vtk_lagrange_nodes(family, p);
    ASSERT_EQ(nodes.size(), static_cast<size_t>(shape.expected_vertices()));

    const MeshBase mesh = make_identity_mesh(shape, nodes);

    ParametricPoint xi{0, 0, 0};
    if (family == CellFamily::Hex) xi = {0.2, -0.4, 0.6};
    if (family == CellFamily::Tetra) xi = {0.2, 0.3, 0.1};
    if (family == CellFamily::Wedge) xi = {0.2, 0.3, -0.2};
    if (family == CellFamily::Pyramid) xi = {0.2, -0.1, 0.3};

    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, 0, xi);
    EXPECT_NEAR(eval.coordinates[0], xi[0], kTol);
    EXPECT_NEAR(eval.coordinates[1], xi[1], kTol);
    EXPECT_NEAR(eval.coordinates[2], xi[2], kTol);
    EXPECT_NEAR(eval.det_jacobian, 1.0, 1e-10);
    EXPECT_TRUE(eval.is_valid);

    const auto inv = CurvilinearEvaluator::inverse_map(mesh, 0, eval.coordinates);
    EXPECT_TRUE(inv.second);
    EXPECT_NEAR(inv.first[0], xi[0], 1e-10);
    EXPECT_NEAR(inv.first[1], xi[1], 1e-10);
    EXPECT_NEAR(inv.first[2], xi[2], 1e-10);
  }
}

TEST(CurvilinearEvalTest, IdentityMapping_QuadraticSerendipity) {
  // Quad8, Hex20, Wedge15, Pyramid13
  struct Case { CellFamily family; size_t n_nodes; ParametricPoint xi; };
  const std::vector<Case> cases = {
      {CellFamily::Quad, 8, {0.25, -0.5, 0.0}},
      {CellFamily::Hex, 20, {0.1, -0.2, 0.3}},
      {CellFamily::Wedge, 15, {0.2, 0.3, -0.2}},
      {CellFamily::Pyramid, 13, {0.2, -0.1, 0.3}},
  };

  for (const auto& c : cases) {
    const CellShape shape = make_shape(c.family, 2);
    const auto nodes = vtk_serendipity_nodes_quadratic(c.family);
    ASSERT_EQ(nodes.size(), c.n_nodes);
    const MeshBase mesh = make_identity_mesh(shape, nodes);

    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, 0, c.xi);
    EXPECT_NEAR(eval.coordinates[0], c.xi[0], kTol);
    EXPECT_NEAR(eval.coordinates[1], c.xi[1], kTol);
    EXPECT_NEAR(eval.coordinates[2], c.xi[2], kTol);
    EXPECT_NEAR(eval.det_jacobian, 1.0, 1e-10);
    EXPECT_TRUE(eval.is_valid);

    const auto sf = CurvilinearEvaluator::evaluate_shape_functions(shape, c.n_nodes, c.xi);
    real_t sum = 0.0;
    for (real_t Ni : sf.N) sum += Ni;
    EXPECT_NEAR(sum, 1.0, 1e-12);
  }
}

TEST(CurvilinearEvalTest, PyramidLagrange_P5_PartitionOfUnityAndInverseMap) {
  const int p = 5;
  const CellShape shape = make_shape(CellFamily::Pyramid, p);
  const auto nodes = vtk_lagrange_nodes(CellFamily::Pyramid, p);
  ASSERT_EQ(nodes.size(), static_cast<size_t>(shape.expected_vertices()));

  // Distort reference node coordinates slightly to exercise conditioning + inverse_map.
  std::vector<ParametricPoint> X = nodes;
  for (auto& pt : X) {
    const real_t x = pt[0];
    const real_t y = pt[1];
    const real_t z = pt[2];
    pt[0] = x + 0.05 * x * (1.0 - z);
    pt[1] = y + 0.04 * y * (1.0 - z);
    pt[2] = z + 0.01 * x * y;
  }

  const MeshBase mesh = make_identity_mesh(shape, X);

  const ParametricPoint xi = {0.2, -0.1, 0.3};
  const auto sf = CurvilinearEvaluator::evaluate_shape_functions(shape, nodes.size(), xi);
  real_t sum = 0.0;
  for (real_t Ni : sf.N) sum += Ni;
  EXPECT_NEAR(sum, 1.0, 1e-12);

  const auto fwd = CurvilinearEvaluator::evaluate_geometry(mesh, 0, xi);
  ASSERT_TRUE(fwd.is_valid);
  const auto inv = CurvilinearEvaluator::inverse_map(mesh, 0, fwd.coordinates);
  EXPECT_TRUE(inv.second);
  EXPECT_NEAR(inv.first[0], xi[0], 1e-9);
  EXPECT_NEAR(inv.first[1], xi[1], 1e-9);
  EXPECT_NEAR(inv.first[2], xi[2], 1e-9);
}

TEST(CurvilinearEvalTest, PyramidLagrange_P5_NodalKronecker) {
  const int p = 5;
  const CellShape shape = make_shape(CellFamily::Pyramid, p);
  const auto nodes = vtk_lagrange_nodes(CellFamily::Pyramid, p);
  const size_t n = nodes.size();
  ASSERT_EQ(n, static_cast<size_t>(shape.expected_vertices()));

  for (size_t j = 0; j < n; ++j) {
    const auto sf = CurvilinearEvaluator::evaluate_shape_functions(shape, n, nodes[j]);
    ASSERT_EQ(sf.N.size(), n);
    for (size_t i = 0; i < n; ++i) {
      const real_t expect = (i == j) ? 1.0 : 0.0;
      EXPECT_NEAR(sf.N[i], expect, 5e-9) << "i=" << i << " j=" << j;
    }
  }
}

TEST(CurvilinearEvalTest, PyramidLagrange_P7_NodalKroneckerSubset) {
  const int p = 7;
  const CellShape shape = make_shape(CellFamily::Pyramid, p);
  const auto nodes = vtk_lagrange_nodes(CellFamily::Pyramid, p);
  const size_t n = nodes.size();
  ASSERT_EQ(n, static_cast<size_t>(shape.expected_vertices()));

  // Check a subset of nodes (corners + a few interior nodes) for Kronecker delta.
  const std::vector<size_t> sample = {0u, 1u, 2u, 3u, 4u, n / 3u, n / 2u, n - 2u};
  for (size_t j : sample) {
    const auto sf = CurvilinearEvaluator::evaluate_shape_functions(shape, n, nodes[j]);
    ASSERT_EQ(sf.N.size(), n);
    for (size_t i = 0; i < n; ++i) {
      const real_t expect = (i == j) ? 1.0 : 0.0;
      EXPECT_NEAR(sf.N[i], expect, 5e-9) << "i=" << i << " j=" << j;
    }
  }
}

} // namespace test
} // namespace svmp
