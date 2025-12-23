/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"

#include "Geometry/Tessellation.h"
#include "Geometry/MeshGeometry.h"
#include "Core/MeshBase.h"
#include "Topology/CellShape.h"
#include "Topology/CellTopology.h"

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

std::array<real_t, 3> corner_param(CellFamily family, int corner_id) {
  switch (family) {
    case CellFamily::Line:
      return (corner_id == 0) ? std::array<real_t,3>{-1, 0, 0} : std::array<real_t,3>{1, 0, 0};
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

std::array<real_t, 3> lerp(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b, real_t t) {
  return {(1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1], (1 - t) * a[2] + t * b[2]};
}

MeshBase make_linear_reference_cell(CellFamily family) {
  const CellShape shape = make_shape(family, 1);
  const int n = shape.num_corners;
  std::vector<real_t> X;
  X.reserve(static_cast<size_t>(n) * 3);
  for (int i = 0; i < n; ++i) {
    const auto p = corner_param(family, i);
    X.push_back(p[0]);
    X.push_back(p[1]);
    X.push_back(p[2]);
  }

  std::vector<offset_t> offs = {0, static_cast<offset_t>(n)};
  std::vector<index_t> conn(static_cast<size_t>(n));
  std::iota(conn.begin(), conn.end(), 0);
  std::vector<CellShape> shapes = {shape};

  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);
  return mesh;
}

MeshBase make_quadratic_serendipity_identity(CellFamily family) {
  const int order = 2;
  CellShape shape = make_shape(family, order);
  const auto eview = CellTopology::get_edges_view(family);

  int nc = 0;
  switch (family) {
    case CellFamily::Quad: nc = 4; break;
    case CellFamily::Hex: nc = 8; break;
    case CellFamily::Wedge: nc = 6; break;
    case CellFamily::Pyramid: nc = 5; break;
    default: nc = 0; break;
  }

  std::vector<std::array<real_t, 3>> nodes;
  for (int c = 0; c < nc; ++c) nodes.push_back(corner_param(family, c));
  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int a = eview.pairs_flat[2 * ei + 0];
    const int b = eview.pairs_flat[2 * ei + 1];
    nodes.push_back(lerp(corner_param(family, a), corner_param(family, b), 0.5));
  }

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

MeshBase make_curved_quadratic_line_mesh() {
  // Quadratic Lagrange line (3 nodes) with a bulged midpoint.
  // Node ordering matches CurvilinearEvaluator::eval_line_shape_functions for p=2: xi = {-1, 0, 1}.
  std::vector<real_t> X = {
      0.0, 0.0, 0.0,   // xi=-1
      0.5, 0.25, 0.0,  // xi=0 (bulge)
      1.0, 0.0, 0.0    // xi=+1
  };

  std::vector<offset_t> offs = {0, 3};
  std::vector<index_t> conn = {0, 1, 2};
  CellShape shape = make_shape(CellFamily::Line, 2);
  std::vector<CellShape> shapes = {shape};

  MeshBase mesh;
  mesh.build_from_arrays(3, X, offs, conn, shapes);
  return mesh;
}

MeshBase make_quadratic_serendipity_quad8_face_mesh() {
  // Single quadratic serendipity quad face (Q8) with lifted mid-edge nodes.
  std::vector<real_t> X = {
      0.0, 0.0, 0.0,  // 0 corner (-1,-1)
      1.0, 0.0, 0.0,  // 1 corner (+1,-1)
      1.0, 1.0, 0.0,  // 2 corner (+1,+1)
      0.0, 1.0, 0.0,  // 3 corner (-1,+1)
      0.5, 0.0, 0.2,  // 4 mid (0-1)
      1.0, 0.5, 0.2,  // 5 mid (1-2)
      0.5, 1.0, 0.2,  // 6 mid (2-3)
      0.0, 0.5, 0.2   // 7 mid (3-0)
  };

  // No cells needed for face tessellation tests.
  std::vector<offset_t> cell_offs = {0};
  std::vector<index_t> cell_conn;
  std::vector<CellShape> cell_shapes;

  MeshBase mesh;
  mesh.build_from_arrays(3, X, cell_offs, cell_conn, cell_shapes);

  CellShape fshape;
  fshape.family = CellFamily::Quad;
  fshape.order = 2;
  fshape.num_corners = 4;

  std::vector<CellShape> face_shapes = {fshape};
  std::vector<offset_t> face_offs = {0, 8};
  std::vector<index_t> face_conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::array<index_t,2>> face2cell = {{{INVALID_INDEX, INVALID_INDEX}}};
  mesh.set_faces_from_arrays(face_shapes, face_offs, face_conn, face2cell);
  return mesh;
}

real_t linear_subcell_measure(const TessellatedCell& tess, int sub_id) {
  const int b = tess.offsets[static_cast<size_t>(sub_id)];
  const int e = tess.offsets[static_cast<size_t>(sub_id + 1)];
  const int n = e - b;
  auto v = [&](int i) -> const std::array<real_t, 3>& {
    return tess.vertices[static_cast<size_t>(tess.connectivity[static_cast<size_t>(b + i)])];
  };

  switch (tess.sub_element_shape.family) {
    case CellFamily::Line:
      if (n == 2) return MeshGeometry::distance(v(0), v(1));
      break;
    case CellFamily::Triangle:
      if (n == 3) return MeshGeometry::triangle_area(v(0), v(1), v(2));
      break;
    case CellFamily::Quad:
      if (n == 4) {
        std::vector<std::array<real_t, 3>> q = {v(0), v(1), v(2), v(3)};
        return MeshGeometry::quad_area(q);
      }
      break;
    case CellFamily::Tetra:
      if (n == 4) return std::abs(MeshGeometry::tet_volume(v(0), v(1), v(2), v(3)));
      break;
    case CellFamily::Hex:
      if (n == 8) {
        std::vector<std::array<real_t, 3>> h = {v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7)};
        return MeshGeometry::hex_volume(h);
      }
      break;
    case CellFamily::Wedge:
      if (n == 6) {
        std::vector<std::array<real_t, 3>> w = {v(0), v(1), v(2), v(3), v(4), v(5)};
        return MeshGeometry::wedge_volume(w);
      }
      break;
    case CellFamily::Pyramid:
      if (n == 5) {
        std::vector<std::array<real_t, 3>> p = {v(0), v(1), v(2), v(3), v(4)};
        return MeshGeometry::pyramid_volume(p);
      }
      break;
    default:
      break;
  }
  return 0.0;
}

int expected_subcell_count(CellFamily family, int level) {
  const int n_div = (level <= 0) ? 1 : (1 << level);
  switch (family) {
    case CellFamily::Line: return n_div;
    case CellFamily::Triangle: return n_div * n_div;
    case CellFamily::Quad: return n_div * n_div;
    case CellFamily::Tetra: return static_cast<int>(std::pow(8.0, level));
    case CellFamily::Hex: return n_div * n_div * n_div;
    case CellFamily::Wedge: return n_div * n_div * n_div;
    case CellFamily::Pyramid:
      // Pyramid tessellates to tetrahedra: 2 * 8^level.
      return 2 * n_div * n_div * n_div;
    default: return 0;
  }
}

} // namespace

TEST(TessellationTest, SubElementCounts) {
  struct Case { CellFamily family; int level; };
  const std::vector<Case> cases = {
      {CellFamily::Line, 0},
      {CellFamily::Line, 2},
      {CellFamily::Triangle, 1},
      {CellFamily::Triangle, 2},
      {CellFamily::Quad, 2},
      {CellFamily::Tetra, 0},
      {CellFamily::Tetra, 2},
      {CellFamily::Hex, 2},
      {CellFamily::Wedge, 2},
      {CellFamily::Pyramid, 2},
  };

  for (const auto& c : cases) {
    const MeshBase mesh = make_linear_reference_cell(c.family);
    TessellationConfig cfg;
    cfg.refinement_level = c.level;
    const auto tess = Tessellator::tessellate_cell(mesh, 0, cfg);
    EXPECT_EQ(tess.n_sub_elements(), expected_subcell_count(c.family, c.level))
        << "family=" << static_cast<int>(c.family);
  }
}

TEST(TessellationTest, MeasureConservation_LinearReference) {
  const int level = 2;
  for (CellFamily family : {CellFamily::Triangle, CellFamily::Quad, CellFamily::Tetra, CellFamily::Hex,
                            CellFamily::Wedge, CellFamily::Pyramid}) {
    const MeshBase mesh = make_linear_reference_cell(family);
    const real_t ref = MeshGeometry::cell_measure(mesh, 0);

    TessellationConfig cfg;
    cfg.refinement_level = level;
    const auto tess = Tessellator::tessellate_cell(mesh, 0, cfg);

    real_t sum = 0.0;
    for (int si = 0; si < tess.n_sub_elements(); ++si) {
      const real_t m = linear_subcell_measure(tess, si);
      EXPECT_TRUE(std::isfinite(m));
      sum += m;
    }
    EXPECT_NEAR(sum, ref, kTol) << "family=" << static_cast<int>(family);
  }
}

TEST(TessellationTest, MeasureConservation_HighOrderIdentity_Hex20) {
  // High-order mapping is identity when nodal coordinates equal reference node coordinates.
  const MeshBase mesh = make_quadratic_serendipity_identity(CellFamily::Hex);
  const real_t ref = MeshGeometry::cell_measure(mesh, 0);

  TessellationConfig cfg;
  cfg.refinement_level = 2;
  const auto tess = Tessellator::tessellate_cell(mesh, 0, cfg);

  real_t sum = 0.0;
  for (int si = 0; si < tess.n_sub_elements(); ++si) sum += linear_subcell_measure(tess, si);
  EXPECT_NEAR(sum, ref, kTol);
}

TEST(TessellationTest, AdaptiveRefinement_CurvedQuadraticLine) {
  const MeshBase mesh = make_curved_quadratic_line_mesh();

  TessellationConfig cfg;
  cfg.refinement_level = 0;
  cfg.adaptive = true;
  cfg.curvature_threshold = 0.12;

  const auto tess = Tessellator::tessellate_cell(mesh, 0, cfg);

  // For p=2, adaptive starts at level>=1; with this threshold it should not need level 2.
  EXPECT_EQ(tess.n_sub_elements(), 2);
  ASSERT_EQ(tess.vertices.size(), 3u);
  EXPECT_NEAR(tess.vertices[1][1], 0.25, kTol);
}

TEST(TessellationTest, FaceTessellation_RefinementAndHighOrderMapping) {
  const MeshBase mesh = make_quadratic_serendipity_quad8_face_mesh();

  TessellationConfig cfg;
  cfg.refinement_level = 1;

  const auto tess = Tessellator::tessellate_face(mesh, 0, cfg);

  EXPECT_EQ(tess.sub_element_shape.family, CellFamily::Quad);
  EXPECT_EQ(tess.n_sub_elements(), 4);
  ASSERT_EQ(tess.vertices.size(), 9u);

  // Param point (xi=0, eta=-1) is the bottom edge midpoint -> should map to node 4 (z=0.2).
  EXPECT_NEAR(tess.vertices[1][2], 0.2, kTol);
}

} // namespace test
} // namespace svmp
