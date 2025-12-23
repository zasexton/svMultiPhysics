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

#include "MeshQuality.h"
#include "CurvilinearEval.h"
#include "MeshGeometry.h"
#include "PolyhedronTessellation.h"
#include "Tessellation.h"
#include "../Core/MeshBase.h"
#include "../Topology/CellTopology.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <unordered_map>

namespace svmp {

namespace {

constexpr real_t kHuge = static_cast<real_t>(1e300);

inline std::array<real_t, 3> sub(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline real_t dot3(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline std::array<real_t, 3> cross3(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b) {
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

inline real_t norm3(const std::array<real_t, 3>& a) {
  return std::sqrt(dot3(a, a));
}

inline real_t clamp(real_t x, real_t lo, real_t hi) {
  return std::max(lo, std::min(hi, x));
}

inline std::array<std::array<real_t, 3>, 3> jt_j(const Jacobian& J) {
  std::array<std::array<real_t, 3>, 3> G{};
  for (int i = 0; i < J.parametric_dim; ++i) {
    for (int j = 0; j < J.parametric_dim; ++j) {
      real_t acc = 0;
      for (int k = 0; k < 3; ++k) {
        acc += J.matrix[k][i] * J.matrix[k][j];
      }
      G[i][j] = acc;
    }
  }
  return G;
}

inline std::array<real_t, 3> eigenvalues_sym_2x2(real_t a00, real_t a01, real_t a11) {
  const real_t tr = a00 + a11;
  const real_t det = a00 * a11 - a01 * a01;
  const real_t disc = std::max(static_cast<real_t>(0), tr * tr - 4 * det);
  const real_t s = std::sqrt(disc);
  const real_t l1 = 0.5 * (tr - s);
  const real_t l2 = 0.5 * (tr + s);
  return {l1, l2, 0};
}

inline std::array<real_t, 3> eigenvalues_sym_3x3(const std::array<std::array<real_t, 3>, 3>& A) {
  const real_t a00 = A[0][0], a01 = A[0][1], a02 = A[0][2];
  const real_t a11 = A[1][1], a12 = A[1][2];
  const real_t a22 = A[2][2];

  const real_t p1 = a01 * a01 + a02 * a02 + a12 * a12;
  if (p1 <= 0) {
    // Diagonal
    std::array<real_t, 3> w = {a00, a11, a22};
    std::sort(w.begin(), w.end());
    return w;
  }

  const real_t q = (a00 + a11 + a22) / 3.0;
  const real_t b00 = a00 - q;
  const real_t b11 = a11 - q;
  const real_t b22 = a22 - q;
  const real_t p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
  const real_t p = std::sqrt(p2 / 6.0);
  if (p <= 0) {
    return {q, q, q};
  }

  // B = (1/p) * (A - q I)
  const real_t invp = 1.0 / p;
  const real_t c00 = b00 * invp;
  const real_t c01 = a01 * invp;
  const real_t c02 = a02 * invp;
  const real_t c11 = b11 * invp;
  const real_t c12 = a12 * invp;
  const real_t c22 = b22 * invp;

  const real_t detB =
      c00 * (c11 * c22 - c12 * c12) -
      c01 * (c01 * c22 - c12 * c02) +
      c02 * (c01 * c12 - c11 * c02);
  const real_t r = detB / 2.0;

  const real_t phi = std::acos(clamp(r, -1.0, 1.0)) / 3.0;
  const real_t two_p = 2.0 * p;

  const real_t eig1 = q + two_p * std::cos(phi);
  const real_t eig3 = q + two_p * std::cos(phi + 2.0 * M_PI / 3.0);
  const real_t eig2 = 3.0 * q - eig1 - eig3;

  std::array<real_t, 3> w = {eig1, eig2, eig3};
  std::sort(w.begin(), w.end());
  return w;
}

static std::vector<ParametricPoint> reference_sample_points(const CellShape& shape) {
  std::vector<ParametricPoint> pts;
  pts.reserve(32);
  pts.push_back(CurvilinearEvaluator::reference_element_center(shape));

  const auto add = [&](const ParametricPoint& p) {
    pts.push_back(p);
  };

  auto corners = [&](CellFamily fam) -> std::vector<ParametricPoint> {
    switch (fam) {
      case CellFamily::Line:
        return {{{-1, 0, 0}}, {{1, 0, 0}}};
      case CellFamily::Triangle:
        return {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}};
      case CellFamily::Quad:
        return {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}};
      case CellFamily::Tetra:
        return {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}};
      case CellFamily::Hex:
        return {{{-1, -1, -1}}, {{1, -1, -1}}, {{1, 1, -1}}, {{-1, 1, -1}},
                {{-1, -1, 1}}, {{1, -1, 1}}, {{1, 1, 1}}, {{-1, 1, 1}}};
      case CellFamily::Wedge:
        return {{{0, 0, -1}}, {{1, 0, -1}}, {{0, 1, -1}},
                {{0, 0, 1}},  {{1, 0, 1}},  {{0, 1, 1}}};
      case CellFamily::Pyramid:
        return {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}, {{0, 0, 1}}};
      default:
        return {};
    }
  };

  const auto c = corners(shape.family);
  for (const auto& p : c) add(p);

  // Edge midpoints
  if (shape.family == CellFamily::Triangle || shape.family == CellFamily::Quad ||
      shape.family == CellFamily::Tetra || shape.family == CellFamily::Hex ||
      shape.family == CellFamily::Wedge || shape.family == CellFamily::Pyramid) {
    const auto eview = CellTopology::get_edges_view(shape.family);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
      const int a = eview.pairs_flat[2 * ei + 0];
      const int b = eview.pairs_flat[2 * ei + 1];
      if (a < 0 || b < 0 || static_cast<size_t>(a) >= c.size() || static_cast<size_t>(b) >= c.size()) continue;
      add({0.5 * (c[static_cast<size_t>(a)][0] + c[static_cast<size_t>(b)][0]),
           0.5 * (c[static_cast<size_t>(a)][1] + c[static_cast<size_t>(b)][1]),
           0.5 * (c[static_cast<size_t>(a)][2] + c[static_cast<size_t>(b)][2])});
    }
  }

  // Face centers (3D)
  if (shape.family == CellFamily::Tetra || shape.family == CellFamily::Hex ||
      shape.family == CellFamily::Wedge || shape.family == CellFamily::Pyramid) {
    const auto fview = CellTopology::get_oriented_boundary_faces_view(shape.family);
    for (int fi = 0; fi < fview.face_count; ++fi) {
      const int b = fview.offsets[fi];
      const int e = fview.offsets[fi + 1];
      const int fv = e - b;
      ParametricPoint fc{0, 0, 0};
      for (int k = 0; k < fv; ++k) {
        const int vi = fview.indices[b + k];
        if (vi < 0 || static_cast<size_t>(vi) >= c.size()) continue;
        fc[0] += c[static_cast<size_t>(vi)][0];
        fc[1] += c[static_cast<size_t>(vi)][1];
        fc[2] += c[static_cast<size_t>(vi)][2];
      }
      if (fv > 0) {
        fc[0] /= fv;
        fc[1] /= fv;
        fc[2] /= fv;
        add(fc);
      }
    }
  }

  return pts;
}

static std::vector<ParametricPoint> quality_sample_points(const CellShape& shape, size_t n_nodes) {
  std::vector<ParametricPoint> pts = reference_sample_points(shape);
  const int p = std::max(1, CurvilinearEvaluator::deduce_order(shape, n_nodes));
  if (p <= 1) return pts;

  auto add = [&](real_t a, real_t b, real_t c) {
    pts.push_back({a, b, c});
  };

  switch (shape.family) {
    case CellFamily::Line: {
      for (int i = 0; i <= p; ++i) {
        const real_t xi = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(p);
        add(xi, 0, 0);
      }
      break;
    }
    case CellFamily::Triangle: {
      for (int j = 0; j <= p; ++j) {
        for (int i = 0; i <= p - j; ++i) {
          add(static_cast<real_t>(i) / static_cast<real_t>(p),
              static_cast<real_t>(j) / static_cast<real_t>(p),
              0);
        }
      }
      break;
    }
    case CellFamily::Quad: {
      for (int j = 0; j <= p; ++j) {
        for (int i = 0; i <= p; ++i) {
          const real_t xi = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(p);
          const real_t eta = -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(p);
          add(xi, eta, 0);
        }
      }
      break;
    }
    case CellFamily::Tetra: {
      for (int k = 0; k <= p; ++k) {
        for (int j = 0; j <= p - k; ++j) {
          for (int i = 0; i <= p - k - j; ++i) {
            add(static_cast<real_t>(i) / static_cast<real_t>(p),
                static_cast<real_t>(j) / static_cast<real_t>(p),
                static_cast<real_t>(k) / static_cast<real_t>(p));
          }
        }
      }
      break;
    }
    case CellFamily::Hex: {
      for (int k = 0; k <= p; ++k) {
        for (int j = 0; j <= p; ++j) {
          for (int i = 0; i <= p; ++i) {
            const real_t xi = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(p);
            const real_t eta = -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(p);
            const real_t zeta = -1.0 + 2.0 * static_cast<real_t>(k) / static_cast<real_t>(p);
            add(xi, eta, zeta);
          }
        }
      }
      break;
    }
    case CellFamily::Wedge: {
      for (int k = 0; k <= p; ++k) {
        const real_t zeta = -1.0 + 2.0 * static_cast<real_t>(k) / static_cast<real_t>(p);
        for (int j = 0; j <= p; ++j) {
          for (int i = 0; i <= p - j; ++i) {
            add(static_cast<real_t>(i) / static_cast<real_t>(p),
                static_cast<real_t>(j) / static_cast<real_t>(p),
                zeta);
          }
        }
      }
      break;
    }
    case CellFamily::Pyramid: {
      // Layered grid nodes on the collapsed domain.
      for (int k = 0; k <= p; ++k) {
        const real_t z = static_cast<real_t>(k) / static_cast<real_t>(p);
        const int m = p - k; // segments at this layer
        for (int j = 0; j <= m; ++j) {
          for (int i = 0; i <= m; ++i) {
            const real_t x = (m == 0) ? 0.0 : static_cast<real_t>(-m + 2 * i) / static_cast<real_t>(p);
            const real_t y = (m == 0) ? 0.0 : static_cast<real_t>(-m + 2 * j) / static_cast<real_t>(p);
            add(x, y, z);
          }
        }
      }
      break;
    }
    default:
      break;
  }

  return pts;
}

static bool metric_lower_is_better(MeshQuality::Metric metric) {
  switch (metric) {
    case MeshQuality::Metric::AspectRatio:
    case MeshQuality::Metric::Skewness:
    case MeshQuality::Metric::EdgeRatio:
    case MeshQuality::Metric::MaxAngle:
    case MeshQuality::Metric::Warpage:
      return true;
    default:
      return false;
  }
}

static MeshBase make_single_tet_mesh(const PolyhedronTet4& tet) {
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

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

static real_t polyhedron_metric_via_tets(
    const MeshBase& mesh,
    index_t cell,
    MeshQuality::Metric metric,
    Configuration cfg) {

  const auto tets = PolyhedronTessellation::convex_star_tets(mesh, cell, cfg);
  if (tets.empty()) return 0.0;

  const bool lower = metric_lower_is_better(metric);
  real_t agg = lower ? 0.0 : kHuge;

  for (const auto& tet : tets) {
    const MeshBase tet_mesh = make_single_tet_mesh(tet);
    const real_t q = MeshQuality::compute(tet_mesh, 0, metric, Configuration::Reference);
    if (lower) {
      agg = std::max(agg, q);
    } else {
      agg = std::min(agg, q);
    }
  }

  if (!lower && agg == kHuge) return 0.0;
  return agg;
}

static real_t quad_warpage_normalized(const std::array<real_t, 3>& p0,
                                      const std::array<real_t, 3>& p1,
                                      const std::array<real_t, 3>& p2,
                                      const std::array<real_t, 3>& p3) {
  const auto n0 = cross3(sub(p1, p0), sub(p2, p0));
  const auto n1 = cross3(sub(p2, p0), sub(p3, p0));
  const real_t a0 = norm3(n0);
  const real_t a1 = norm3(n1);
  if (a0 < 1e-14 || a1 < 1e-14) return 1.0;
  const real_t c = clamp(dot3(n0, n1) / (a0 * a1), static_cast<real_t>(-1.0), static_cast<real_t>(1.0));
  const real_t angle = std::acos(c); // [0, pi]
  return angle / static_cast<real_t>(M_PI);
}

static real_t quad_taper_quality(const std::array<real_t, 3>& p0,
                                 const std::array<real_t, 3>& p1,
                                 const std::array<real_t, 3>& p2,
                                 const std::array<real_t, 3>& p3) {
  const auto e01 = sub(p1, p0);
  const auto e23 = sub(p2, p3); // opposite of (p3->p2) to align with e01
  const auto e12 = sub(p2, p1);
  const auto e30 = sub(p3, p0);

  const real_t l01 = norm3(e01);
  const real_t l23 = norm3(e23);
  const real_t l12 = norm3(e12);
  const real_t l30 = norm3(e30);

  if (l01 < 1e-14 || l23 < 1e-14 || l12 < 1e-14 || l30 < 1e-14) return 0.0;

  const real_t c0 = std::abs(dot3(e01, e23) / (l01 * l23));
  const real_t c1 = std::abs(dot3(e12, e30) / (l12 * l30));
  return clamp(std::min(c0, c1), static_cast<real_t>(0.0), static_cast<real_t>(1.0));
}

static real_t linear_subcell_measure(
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& vertices,
    const std::vector<index_t>& connectivity,
    int begin,
    int end) {
  const int n = end - begin;
  auto v = [&](int i) -> const std::array<real_t, 3>& {
    return vertices[static_cast<size_t>(connectivity[static_cast<size_t>(begin + i)])];
  };

  switch (family) {
    case CellFamily::Line:
      if (n == 2) return norm3(sub(v(1), v(0)));
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

static real_t cell_measure_estimate(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const CellShape shape = mesh.cell_shape(cell);
  const auto [_, n_nodes] = mesh.cell_vertices_span(cell);
  const int order = CurvilinearEvaluator::deduce_order(shape, n_nodes);
  if (order <= 1) {
    return MeshGeometry::cell_measure(mesh, cell, cfg);
  }

  TessellationConfig tc;
  tc.refinement_level = std::min(3, Tessellator::suggest_refinement_level(order));
  tc.configuration = cfg;
  const auto tess = Tessellator::tessellate_cell(mesh, cell, tc);

  real_t sum = 0.0;
  const CellFamily fam = tess.sub_element_shape.family;
  for (int si = 0; si < tess.n_sub_elements(); ++si) {
    const int b = tess.offsets[static_cast<size_t>(si)];
    const int e = tess.offsets[static_cast<size_t>(si + 1)];
    sum += linear_subcell_measure(fam, tess.vertices, tess.connectivity, b, e);
  }
  return sum;
}

} // namespace

// Convert metric enum to string
std::string MeshQuality::metric_name(Metric m) {
  switch (m) {
    case Metric::AspectRatio: return "aspect_ratio";
    case Metric::Skewness: return "skewness";
    case Metric::Jacobian: return "jacobian";
    case Metric::EdgeRatio: return "edge_ratio";
    case Metric::MinAngle: return "min_angle";
    case Metric::MaxAngle: return "max_angle";
    case Metric::Warpage: return "warpage";
    case Metric::Taper: return "taper";
    case Metric::Stretch: return "stretch";
    case Metric::DiagonalRatio: return "diagonal_ratio";
    case Metric::ConditionNumber: return "condition_number";
    case Metric::ScaledJacobian: return "scaled_jacobian";
    case Metric::ShapeIndex: return "shape_index";
    case Metric::RelativeSizeSquared: return "relative_size_squared";
    case Metric::ShapeAndSize: return "shape_and_size";
    default: return "unknown";
  }
}

// Convert string to metric enum
MeshQuality::Metric MeshQuality::metric_from_name(const std::string& name) {
  if (name == "aspect_ratio") return Metric::AspectRatio;
  if (name == "skewness") return Metric::Skewness;
  if (name == "jacobian") return Metric::Jacobian;
  if (name == "edge_ratio") return Metric::EdgeRatio;
  if (name == "min_angle") return Metric::MinAngle;
  if (name == "max_angle") return Metric::MaxAngle;
  if (name == "warpage") return Metric::Warpage;
  if (name == "taper") return Metric::Taper;
  if (name == "stretch") return Metric::Stretch;
  if (name == "diagonal_ratio") return Metric::DiagonalRatio;
  if (name == "condition_number") return Metric::ConditionNumber;
  if (name == "scaled_jacobian") return Metric::ScaledJacobian;
  if (name == "shape_index") return Metric::ShapeIndex;
  if (name == "relative_size_squared") return Metric::RelativeSizeSquared;
  if (name == "shape_and_size") return Metric::ShapeAndSize;
  throw std::invalid_argument("Unknown quality metric: " + name);
}

// Main interface: compute quality for a single cell
real_t MeshQuality::compute(const MeshBase& mesh, index_t cell, Metric metric, Configuration cfg) {
  if (mesh.cell_shape(cell).family == CellFamily::Polyhedron) {
    if (metric == Metric::RelativeSizeSquared) {
      return compute_relative_size_squared(mesh, cell, cfg);
    }
    if (metric == Metric::ShapeAndSize) {
      const real_t shape_q = polyhedron_metric_via_tets(mesh, cell, Metric::ShapeIndex, cfg);
      const real_t size_q = compute_relative_size_squared(mesh, cell, cfg);
      return clamp(shape_q * size_q, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
    }
    return polyhedron_metric_via_tets(mesh, cell, metric, cfg);
  }

  switch (metric) {
    case Metric::AspectRatio:
      return compute_aspect_ratio(mesh, cell, cfg);
    case Metric::Skewness:
      return compute_skewness(mesh, cell, cfg);
    case Metric::Jacobian:
      return compute_jacobian_quality(mesh, cell, cfg);
    case Metric::EdgeRatio:
      return compute_edge_ratio(mesh, cell, cfg);
    case Metric::MinAngle:
      return compute_min_angle(mesh, cell, cfg);
    case Metric::MaxAngle:
      return compute_max_angle(mesh, cell, cfg);
    case Metric::Warpage:
      return compute_warpage(mesh, cell, cfg);
    case Metric::Taper:
      return compute_taper(mesh, cell, cfg);
    case Metric::Stretch:
      return compute_stretch(mesh, cell, cfg);
    case Metric::DiagonalRatio:
      return compute_diagonal_ratio(mesh, cell, cfg);
    case Metric::ConditionNumber:
      return compute_condition_number(mesh, cell, cfg);
    case Metric::ScaledJacobian:
      return compute_scaled_jacobian(mesh, cell, cfg);
    case Metric::ShapeIndex:
      return compute_shape_index(mesh, cell, cfg);
    case Metric::RelativeSizeSquared:
      return compute_relative_size_squared(mesh, cell, cfg);
    case Metric::ShapeAndSize:
      return compute_shape_and_size(mesh, cell, cfg);
    default:
      return 0.0;
  }
}

// Compute quality for a single cell by metric name
real_t MeshQuality::compute(const MeshBase& mesh, index_t cell, const std::string& metric_str,
                           Configuration cfg) {
  Metric metric = metric_from_name(metric_str);
  return compute(mesh, cell, metric, cfg);
}

// Compute quality for all cells
std::vector<real_t> MeshQuality::compute_all(const MeshBase& mesh, Metric metric, Configuration cfg) {
  std::vector<real_t> qualities;
  qualities.reserve(mesh.n_cells());

  if (metric == Metric::RelativeSizeSquared || metric == Metric::ShapeAndSize) {
    const size_t n_cells = mesh.n_cells();
    std::vector<real_t> measures(n_cells, 0.0);

    // Compute per-family average measure.
    struct Acc {
      real_t sum = 0.0;
      size_t count = 0;
    };
    std::unordered_map<CellFamily, Acc> acc;
    acc.reserve(8);

    for (size_t c = 0; c < n_cells; ++c) {
      const index_t cell = static_cast<index_t>(c);
      const CellFamily fam = mesh.cell_shape(cell).family;
      const real_t m = cell_measure_estimate(mesh, cell, cfg);
      measures[c] = m;
      auto& a = acc[fam];
      a.sum += m;
      a.count += 1;
    }

    std::unordered_map<CellFamily, real_t> avg;
    avg.reserve(acc.size());
    for (const auto& kv : acc) {
      const CellFamily fam = kv.first;
      const Acc& a = kv.second;
      avg.emplace(fam, (a.count > 0) ? (a.sum / static_cast<real_t>(a.count)) : 0.0);
    }

    for (size_t c = 0; c < n_cells; ++c) {
      const index_t cell = static_cast<index_t>(c);
      const CellFamily fam = mesh.cell_shape(cell).family;
      const real_t a = avg[fam];
      const real_t m = measures[c];
      real_t size_q = 0.0;
      if (a > 0 && m > 0) {
        const real_t r = m / a;
        const real_t s = std::min(r, static_cast<real_t>(1.0) / r);
        size_q = clamp(s * s, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
      }

      if (metric == Metric::RelativeSizeSquared) {
        qualities.push_back(size_q);
      } else {
        const real_t shape_q = compute_shape_index(mesh, cell, cfg);
        qualities.push_back(clamp(shape_q * size_q, static_cast<real_t>(0.0), static_cast<real_t>(1.0)));
      }
    }
    return qualities;
  }

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    qualities.push_back(compute(mesh, static_cast<index_t>(c), metric, cfg));
  }
  return qualities;
}

// Get global min/max quality
std::pair<real_t,real_t> MeshQuality::global_range(const MeshBase& mesh, Metric metric,
                                                  Configuration cfg) {
  real_t min_quality = 1e300;
  real_t max_quality = -1e300;
  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    real_t q = compute(mesh, static_cast<index_t>(c), metric, cfg);
    min_quality = std::min(min_quality, q);
    max_quality = std::max(max_quality, q);
  }
  return {min_quality, max_quality};
}

// Get statistics for quality metric
MeshQuality::QualityStats MeshQuality::compute_statistics(const MeshBase& mesh, Metric metric,
                                                         Configuration cfg) {
  QualityStats stats;
  std::vector<real_t> qualities = compute_all(mesh, metric, cfg);

  if (qualities.empty()) return stats;

  // Basic statistics
  stats.min = *std::min_element(qualities.begin(), qualities.end());
  stats.max = *std::max_element(qualities.begin(), qualities.end());
  stats.mean = std::accumulate(qualities.begin(), qualities.end(), 0.0) / qualities.size();

  // Standard deviation
  real_t sq_sum = 0;
  for (real_t q : qualities) {
    sq_sum += (q - stats.mean) * (q - stats.mean);
  }
  stats.std_dev = std::sqrt(sq_sum / qualities.size());

  // Count by thresholds
  const bool lower_is_better = metric_lower_is_better(metric);
  for (size_t c = 0; c < qualities.size(); ++c) {
    const real_t q = qualities[c];
    auto thresholds = get_thresholds(metric, mesh.cell_shape(static_cast<index_t>(c)).family);
    if (lower_is_better) {
      if (q > thresholds.acceptable) stats.count_poor++;
      if (q <= thresholds.good) stats.count_good++;
      if (q <= thresholds.excellent) stats.count_excellent++;
    } else {
      if (q < thresholds.acceptable) stats.count_poor++;
      if (q >= thresholds.good) stats.count_good++;
      if (q >= thresholds.excellent) stats.count_excellent++;
    }
  }

  return stats;
}

// Quality thresholds for different metrics
MeshQuality::QualityThresholds MeshQuality::get_thresholds(Metric metric, CellFamily family) {
  QualityThresholds thresh;

  // These are example thresholds - should be refined based on application
  switch (metric) {
    case Metric::AspectRatio:
      // Lower is better for aspect ratio
      thresh.poor = 10.0;
      thresh.acceptable = 5.0;
      thresh.good = 2.0;
      thresh.excellent = 1.5;
      break;

    case Metric::Skewness:
      // Lower is better for skewness
      thresh.poor = 0.9;
      thresh.acceptable = 0.6;
      thresh.good = 0.3;
      thresh.excellent = 0.1;
      break;

    case Metric::EdgeRatio:
      // Lower is better for edge ratio (max/min edge length).
      thresh.poor = 10.0;
      thresh.acceptable = 5.0;
      thresh.good = 2.0;
      thresh.excellent = 1.5;
      break;

    case Metric::MinAngle:
      // Higher is better for min angle
      if (family == CellFamily::Triangle) {
        thresh.poor = 10.0;
        thresh.acceptable = 20.0;
        thresh.good = 30.0;
        thresh.excellent = 40.0;
      } else if (family == CellFamily::Quad) {
        thresh.poor = 30.0;
        thresh.acceptable = 45.0;
        thresh.good = 60.0;
        thresh.excellent = 75.0;
      } else {
        // 3D elements
        thresh.poor = 15.0;
        thresh.acceptable = 25.0;
        thresh.good = 35.0;
        thresh.excellent = 45.0;
      }
      break;

    case Metric::MaxAngle:
      // Lower is better for max angle (degrees). These are example thresholds.
      if (family == CellFamily::Triangle) {
        thresh.poor = 170.0;
        thresh.acceptable = 120.0;
        thresh.good = 100.0;
        thresh.excellent = 80.0;
      } else if (family == CellFamily::Quad) {
        thresh.poor = 170.0;
        thresh.acceptable = 150.0;
        thresh.good = 120.0;
        thresh.excellent = 105.0;
      } else {
        thresh.poor = 170.0;
        thresh.acceptable = 150.0;
        thresh.good = 130.0;
        thresh.excellent = 110.0;
      }
      break;

    case Metric::Warpage:
      // Lower is better for warpage in [0, 1] (0 = planar). These are example thresholds.
      thresh.poor = 0.6;
      thresh.acceptable = 0.3;
      thresh.good = 0.1;
      thresh.excellent = 0.05;
      break;

    default:
      // Generic thresholds
      thresh.poor = 0.1;
      thresh.acceptable = 0.3;
      thresh.good = 0.6;
      thresh.excellent = 0.9;
  }

  return thresh;
}

// Check if cell quality is acceptable
bool MeshQuality::is_acceptable(const MeshBase& mesh, index_t cell, Metric metric, Configuration cfg) {
  real_t quality = compute(mesh, cell, metric, cfg);
  auto thresh = get_thresholds(metric, mesh.cell_shape(cell).family);

  // For metrics where lower is better (aspect ratio, skewness)
  if (metric_lower_is_better(metric)) {
    return quality <= thresh.acceptable;
  }
  // For metrics where higher is better
  return quality >= thresh.acceptable;
}

// Find cells with poor quality
std::vector<index_t> MeshQuality::find_poor_quality_cells(const MeshBase& mesh, Metric metric,
                                                         real_t threshold, Configuration cfg) {
  std::vector<index_t> poor_cells;

  // Determine if metric is "higher is better" or "lower is better"
  bool higher_is_better = !metric_lower_is_better(metric);

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    real_t quality = compute(mesh, static_cast<index_t>(c), metric, cfg);
    bool is_poor = higher_is_better ? (quality < threshold) : (quality > threshold);
    if (is_poor) {
      poor_cells.push_back(static_cast<index_t>(c));
    }
  }

  return poor_cells;
}

// Helper: get cell vertices
std::vector<std::array<real_t,3>> MeshQuality::get_cell_vertices(const MeshBase& mesh, index_t cell,
                                                                 Configuration cfg) {
  auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(cell);
  std::vector<std::array<real_t,3>> vertices;
  vertices.reserve(n_vertices);

  const std::vector<real_t>& coords = ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
  int spatial_dim = mesh.dim();

  for (size_t i = 0; i < n_vertices; ++i) {
    index_t vertex_id = vertices_ptr[i];
    std::array<real_t,3> pt = {{0, 0, 0}};
    for (int d = 0; d < spatial_dim; ++d) {
      pt[d] = coords[vertex_id * spatial_dim + d];
    }
    vertices.push_back(pt);
  }

  return vertices;
}

// Compute aspect ratio
real_t MeshQuality::compute_aspect_ratio(const MeshBase& mesh, index_t cell, Configuration cfg) {
  // Simple implementation using bounding box
  auto vertices = get_cell_vertices(mesh, cell, cfg);

  if (vertices.empty()) return 1e300;

  std::array<real_t,3> min_pt = {{1e300, 1e300, 1e300}};
  std::array<real_t,3> max_pt = {{-1e300, -1e300, -1e300}};

  for (const auto& v : vertices) {
    for (int d = 0; d < 3; ++d) {
      min_pt[d] = std::min(min_pt[d], v[d]);
      max_pt[d] = std::max(max_pt[d], v[d]);
    }
  }

  real_t min_len = 1e300;
  real_t max_len = -1e300;
  int spatial_dim = mesh.dim();

  for (int d = 0; d < spatial_dim; ++d) {
    real_t len = max_pt[d] - min_pt[d];
    if (len > 1e-12) {
      min_len = std::min(min_len, len);
      max_len = std::max(max_len, len);
    }
  }

  if (min_len < 1e-12) return 1e300;
  return max_len / min_len;
}

// Compute edge ratio
real_t MeshQuality::compute_edge_ratio(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto& shape = mesh.cell_shape(cell);

  auto edge_lengths = compute_edge_lengths(vertices, shape);
  if (edge_lengths.empty()) return 1.0;

  real_t min_len = *std::min_element(edge_lengths.begin(), edge_lengths.end());
  real_t max_len = *std::max_element(edge_lengths.begin(), edge_lengths.end());

  if (min_len < 1e-12) return 1e300;
  return max_len / min_len;
}

// Helper: compute edge lengths
std::vector<real_t> MeshQuality::compute_edge_lengths(const std::vector<std::array<real_t,3>>& vertices,
                                                     const CellShape& shape) {
  std::vector<real_t> lengths;

  // Define edges based on cell type
  std::vector<std::pair<int,int>> edges;

  switch (shape.family) {
    case CellFamily::Triangle:
      edges = {{0,1}, {1,2}, {2,0}};
      break;
    case CellFamily::Quad:
      edges = {{0,1}, {1,2}, {2,3}, {3,0}};
      break;
    case CellFamily::Tetra:
      edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
      break;
    case CellFamily::Hex:
      edges = {{0,1}, {1,2}, {2,3}, {3,0},  // bottom face
              {4,5}, {5,6}, {6,7}, {7,4},  // top face
              {0,4}, {1,5}, {2,6}, {3,7}}; // vertical edges
      break;
    case CellFamily::Wedge:
      edges = {{0,1}, {1,2}, {2,0},  // bottom triangle
              {3,4}, {4,5}, {5,3},  // top triangle
              {0,3}, {1,4}, {2,5}}; // vertical edges
      break;
    case CellFamily::Pyramid:
      edges = {{0,1}, {1,2}, {2,3}, {3,0},  // base
              {0,4}, {1,4}, {2,4}, {3,4}}; // to apex
      break;
    default:
      // For other shapes, compute all pairwise distances
      for (size_t i = 0; i < vertices.size(); ++i) {
        for (size_t j = i+1; j < vertices.size(); ++j) {
          edges.push_back({static_cast<int>(i), static_cast<int>(j)});
        }
      }
  }

  // Compute edge lengths
  for (const auto& [i, j] : edges) {
    real_t dx = vertices[j][0] - vertices[i][0];
    real_t dy = vertices[j][1] - vertices[i][1];
    real_t dz = vertices[j][2] - vertices[i][2];
    lengths.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
  }

  return lengths;
}

// Skewness
real_t MeshQuality::compute_skewness(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto& shape = mesh.cell_shape(cell);

  if (vertices.size() < 2) return 1.0;

  // 2D elements (including surfaces embedded in 3D): angle deviation from regular polygon.
  if (shape.is_2d()) {
    const auto angles = compute_angles_2d(vertices); // degrees
    if (angles.size() < 3) return 0.0;

    const real_t n = static_cast<real_t>(angles.size());
    const real_t ideal = static_cast<real_t>(180.0) * (n - 2.0) / n;
    if (ideal <= 1e-12) return 1.0;

    real_t max_dev = 0.0;
    for (real_t a : angles) {
      max_dev = std::max(max_dev, std::abs(a - ideal) / ideal);
    }
    return std::min(static_cast<real_t>(1.0), max_dev);
  }

  // 3D elements: spread of edge lengths as a simple skewness proxy.
  const auto edge_lengths = compute_edge_lengths(vertices, shape);
  if (edge_lengths.empty()) return 1.0;

  const real_t min_len = *std::min_element(edge_lengths.begin(), edge_lengths.end());
  const real_t max_len = *std::max_element(edge_lengths.begin(), edge_lengths.end());
  if (max_len < 1e-12 || min_len < 1e-12) return 1.0;

  // In [0, 1): 0 for equal edges, ->1 as min/max -> 0.
  return std::min(static_cast<real_t>(1.0), (max_len - min_len) / max_len);
}

real_t MeshQuality::compute_jacobian_quality(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const CellShape shape = mesh.cell_shape(cell);
  const auto [_, n_nodes] = mesh.cell_vertices_span(cell);
  const auto pts = quality_sample_points(shape, n_nodes);
  if (pts.empty()) return 0.0;

  real_t min_det = kHuge;
  real_t max_det = 0.0;

  for (const auto& xi : pts) {
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const real_t det = eval.det_jacobian;
    if (shape.is_3d()) {
      if (!(det > 0)) {
        return 0.0; // inverted or degenerate
      }
    } else {
      if (!(det > 0)) {
        continue;
      }
    }
    min_det = std::min(min_det, det);
    max_det = std::max(max_det, det);
  }

  if (!(max_det > 0) || min_det == kHuge) return 0.0;
  return clamp(min_det / max_det, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
}

real_t MeshQuality::compute_min_angle(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto& shape = mesh.cell_shape(cell);

  std::vector<real_t> angles;
  if (shape.is_2d()) {
    angles = compute_angles_2d(vertices);
  } else {
    angles = compute_angles_3d(vertices, shape);
  }

  if (angles.empty()) return 0.0;
  return *std::min_element(angles.begin(), angles.end());
}

real_t MeshQuality::compute_max_angle(const MeshBase& mesh, index_t cell, Configuration cfg) {
  auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto& shape = mesh.cell_shape(cell);

  std::vector<real_t> angles;
  if (shape.is_2d()) {
    angles = compute_angles_2d(vertices);
  } else {
    angles = compute_angles_3d(vertices, shape);
  }

  if (angles.empty()) return 180.0;
  return *std::max_element(angles.begin(), angles.end());
}

// Helper: compute angles for 2D elements
std::vector<real_t> MeshQuality::compute_angles_2d(const std::vector<std::array<real_t,3>>& vertices) {
  std::vector<real_t> angles;
  size_t n = vertices.size();

  for (size_t i = 0; i < n; ++i) {
    size_t prev = (i + n - 1) % n;
    size_t next = (i + 1) % n;

    const std::array<real_t,3> v1 = {{
      vertices[prev][0] - vertices[i][0],
      vertices[prev][1] - vertices[i][1],
      vertices[prev][2] - vertices[i][2]
    }};
    const std::array<real_t,3> v2 = {{
      vertices[next][0] - vertices[i][0],
      vertices[next][1] - vertices[i][1],
      vertices[next][2] - vertices[i][2]
    }};

    // Compute angle using 3D dot product (works for 2D and embedded surfaces).
    real_t dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    real_t len1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    real_t len2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);

    if (len1 > 1e-12 && len2 > 1e-12) {
      real_t cos_angle = dot / (len1 * len2);
      cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
      angles.push_back(std::acos(cos_angle) * 180.0 / M_PI);
    }
  }

  return angles;
}

std::vector<real_t> MeshQuality::compute_angles_3d(const std::vector<std::array<real_t,3>>& vertices,
                                                  const CellShape& shape) {
  std::vector<real_t> angles;
  if (vertices.size() < 4) return angles;

  // Build edge adjacency for standard 3D elements.
  std::vector<std::pair<int,int>> edges;
  switch (shape.family) {
    case CellFamily::Tetra:
      edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
      break;
    case CellFamily::Hex:
      edges = {{0,1}, {1,2}, {2,3}, {3,0},
              {4,5}, {5,6}, {6,7}, {7,4},
              {0,4}, {1,5}, {2,6}, {3,7}};
      break;
    case CellFamily::Wedge:
      edges = {{0,1}, {1,2}, {2,0},
              {3,4}, {4,5}, {5,3},
              {0,3}, {1,4}, {2,5}};
      break;
    case CellFamily::Pyramid:
      edges = {{0,1}, {1,2}, {2,3}, {3,0},
              {0,4}, {1,4}, {2,4}, {3,4}};
      break;
    default:
      return angles;
  }

  std::vector<std::vector<int>> neighbors(vertices.size());
  for (const auto& [a, b] : edges) {
    if (a < 0 || b < 0) continue;
    if (static_cast<size_t>(a) >= vertices.size() || static_cast<size_t>(b) >= vertices.size()) continue;
    neighbors[static_cast<size_t>(a)].push_back(b);
    neighbors[static_cast<size_t>(b)].push_back(a);
  }

  for (size_t i = 0; i < neighbors.size(); ++i) {
    auto& nbrs = neighbors[i];
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

    for (size_t a = 0; a + 1 < nbrs.size(); ++a) {
      for (size_t b = a + 1; b < nbrs.size(); ++b) {
        const auto& pa = vertices[static_cast<size_t>(nbrs[a])];
        const auto& pb = vertices[static_cast<size_t>(nbrs[b])];
        const auto& pi = vertices[i];

        const std::array<real_t,3> v1 = {{pa[0] - pi[0], pa[1] - pi[1], pa[2] - pi[2]}};
        const std::array<real_t,3> v2 = {{pb[0] - pi[0], pb[1] - pi[1], pb[2] - pi[2]}};

        real_t dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
        real_t len1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
        real_t len2 = std::sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);
        if (len1 < 1e-12 || len2 < 1e-12) continue;

        real_t cos_angle = dot / (len1 * len2);
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        angles.push_back(std::acos(cos_angle) * 180.0 / M_PI);
      }
    }
  }

  return angles;
}

// Additional metrics
real_t MeshQuality::compute_warpage(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto shape = mesh.cell_shape(cell);
  if (vertices.size() < static_cast<size_t>(shape.num_corners)) return 1.0;

  if (shape.family == CellFamily::Quad) {
    if (vertices.size() < 4) return 1.0;
    return quad_warpage_normalized(vertices[0], vertices[1], vertices[2], vertices[3]);
  }

  if (!shape.is_3d()) return 0.0;

  real_t max_warp = 0.0;
  bool has_quad = false;
  const auto fview = CellTopology::get_oriented_boundary_faces_view(shape.family);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    const int fv = e - b;
    if (fv != 4) continue;
    has_quad = true;
    const int i0 = fview.indices[b + 0];
    const int i1 = fview.indices[b + 1];
    const int i2 = fview.indices[b + 2];
    const int i3 = fview.indices[b + 3];
    if (i0 < 0 || i1 < 0 || i2 < 0 || i3 < 0) continue;
    if (static_cast<size_t>(i3) >= vertices.size()) continue;
    max_warp = std::max(max_warp, quad_warpage_normalized(vertices[static_cast<size_t>(i0)],
                                                          vertices[static_cast<size_t>(i1)],
                                                          vertices[static_cast<size_t>(i2)],
                                                          vertices[static_cast<size_t>(i3)]));
  }

  return has_quad ? max_warp : 0.0;
}

real_t MeshQuality::compute_taper(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const auto vertices = get_cell_vertices(mesh, cell, cfg);
  const auto shape = mesh.cell_shape(cell);
  if (vertices.size() < static_cast<size_t>(shape.num_corners)) return 0.0;

  if (shape.family == CellFamily::Quad) {
    if (vertices.size() < 4) return 0.0;
    return quad_taper_quality(vertices[0], vertices[1], vertices[2], vertices[3]);
  }

  if (!shape.is_3d()) return 1.0;

  real_t min_q = 1.0;
  bool has_quad = false;
  const auto fview = CellTopology::get_oriented_boundary_faces_view(shape.family);
  for (int fi = 0; fi < fview.face_count; ++fi) {
    const int b = fview.offsets[fi];
    const int e = fview.offsets[fi + 1];
    const int fv = e - b;
    if (fv != 4) continue;
    has_quad = true;
    const int i0 = fview.indices[b + 0];
    const int i1 = fview.indices[b + 1];
    const int i2 = fview.indices[b + 2];
    const int i3 = fview.indices[b + 3];
    if (i0 < 0 || i1 < 0 || i2 < 0 || i3 < 0) continue;
    if (static_cast<size_t>(i3) >= vertices.size()) continue;
    min_q = std::min(min_q, quad_taper_quality(vertices[static_cast<size_t>(i0)],
                                               vertices[static_cast<size_t>(i1)],
                                               vertices[static_cast<size_t>(i2)],
                                               vertices[static_cast<size_t>(i3)]));
  }
  return has_quad ? min_q : 1.0;
}

real_t MeshQuality::compute_stretch(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const CellShape shape = mesh.cell_shape(cell);
  const auto [_, n_nodes] = mesh.cell_vertices_span(cell);
  const auto pts = quality_sample_points(shape, n_nodes);
  if (pts.empty()) return 0.0;

  real_t min_q = 1.0;
  for (const auto& xi : pts) {
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const Jacobian& J = eval.jacobian;
    const int pdim = J.parametric_dim;
    if (pdim <= 1) continue;

    const auto G = jt_j(J);
    std::array<real_t, 3> w{};
    if (pdim == 2) {
      w = eigenvalues_sym_2x2(G[0][0], G[0][1], G[1][1]);
    } else {
      w = eigenvalues_sym_3x3(G);
    }
    const real_t lmin = std::max(static_cast<real_t>(0), w[0]);
    const real_t lmax = std::max(static_cast<real_t>(0), w[static_cast<size_t>(pdim - 1)]);
    if (lmin <= 1e-30 || lmax <= 1e-30) {
      min_q = 0.0;
      continue;
    }

    real_t detG = 1.0;
    for (int i = 0; i < pdim; ++i) {
      detG *= std::max(static_cast<real_t>(0), w[static_cast<size_t>(i)]);
    }
    if (detG <= 1e-300) {
      min_q = 0.0;
      continue;
    }

    const real_t sigma_max = std::sqrt(lmax);
    const real_t sigma_geo = std::pow(detG, static_cast<real_t>(1.0) / static_cast<real_t>(2.0 * pdim));
    const real_t q = sigma_geo / sigma_max;
    min_q = std::min(min_q, clamp(q, static_cast<real_t>(0.0), static_cast<real_t>(1.0)));
  }

  return min_q;
}

real_t MeshQuality::compute_diagonal_ratio(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const auto vertices = get_cell_vertices(mesh, cell, cfg);
  const CellShape shape = mesh.cell_shape(cell);

  auto dist = [&](int a, int b) -> real_t {
    if (a < 0 || b < 0) return 0.0;
    if (static_cast<size_t>(a) >= vertices.size() || static_cast<size_t>(b) >= vertices.size()) return 0.0;
    return norm3(sub(vertices[static_cast<size_t>(b)], vertices[static_cast<size_t>(a)]));
  };

  if (shape.family == CellFamily::Quad) {
    const real_t d0 = dist(0, 2);
    const real_t d1 = dist(1, 3);
    const real_t dmax = std::max(d0, d1);
    const real_t dmin = std::min(d0, d1);
    if (dmax < 1e-14) return 0.0;
    return clamp(dmin / dmax, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
  }

  if (shape.family == CellFamily::Hex) {
    const real_t d0 = dist(0, 6);
    const real_t d1 = dist(1, 7);
    const real_t d2 = dist(2, 4);
    const real_t d3 = dist(3, 5);
    const real_t dmax = std::max(std::max(d0, d1), std::max(d2, d3));
    const real_t dmin = std::min(std::min(d0, d1), std::min(d2, d3));
    if (dmax < 1e-14) return 0.0;
    return clamp(dmin / dmax, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
  }

  if (shape.family == CellFamily::Pyramid) {
    const real_t d0 = dist(0, 2);
    const real_t d1 = dist(1, 3);
    const real_t dmax = std::max(d0, d1);
    const real_t dmin = std::min(d0, d1);
    if (dmax < 1e-14) return 0.0;
    return clamp(dmin / dmax, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
  }

  return 1.0;
}

real_t MeshQuality::compute_condition_number(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const CellShape shape = mesh.cell_shape(cell);
  const auto [_, n_nodes] = mesh.cell_vertices_span(cell);
  const auto pts = quality_sample_points(shape, n_nodes);
  if (pts.empty()) return 0.0;

  real_t min_q = 1.0;
  for (const auto& xi : pts) {
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const Jacobian& J = eval.jacobian;
    const int pdim = J.parametric_dim;
    if (pdim <= 1) continue;

    const auto G = jt_j(J);
    std::array<real_t, 3> w{};
    if (pdim == 2) {
      w = eigenvalues_sym_2x2(G[0][0], G[0][1], G[1][1]);
    } else {
      w = eigenvalues_sym_3x3(G);
    }
    const real_t lmin = std::max(static_cast<real_t>(0), w[0]);
    const real_t lmax = std::max(static_cast<real_t>(0), w[static_cast<size_t>(pdim - 1)]);
    if (lmin <= 1e-30 || lmax <= 1e-30) {
      min_q = 0.0;
      continue;
    }
    // Quality form: 1/cond = sigma_min/sigma_max.
    const real_t q = std::sqrt(lmin / lmax);
    min_q = std::min(min_q, clamp(q, static_cast<real_t>(0.0), static_cast<real_t>(1.0)));
  }

  return min_q;
}

real_t MeshQuality::compute_scaled_jacobian(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const CellShape shape = mesh.cell_shape(cell);
  const auto [_, n_nodes] = mesh.cell_vertices_span(cell);
  const auto pts = quality_sample_points(shape, n_nodes);
  if (pts.empty()) return 0.0;

  real_t min_scaled = 1.0;
  for (const auto& xi : pts) {
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const Jacobian& J = eval.jacobian;
    const int pdim = J.parametric_dim;
    if (pdim <= 1) continue;

    // Column norms.
    real_t norms[3] = {0, 0, 0};
    for (int j = 0; j < pdim; ++j) {
      real_t s = 0.0;
      for (int k = 0; k < 3; ++k) {
        s += J.matrix[static_cast<size_t>(k)][static_cast<size_t>(j)] *
             J.matrix[static_cast<size_t>(k)][static_cast<size_t>(j)];
      }
      norms[j] = std::sqrt(s);
    }

    real_t scaled = 1.0;
    if (pdim == 3) {
      const real_t denom = norms[0] * norms[1] * norms[2];
      if (denom <= 1e-30) {
        scaled = -1.0;
      } else {
        scaled = eval.det_jacobian / denom;
      }
    } else if (pdim == 2) {
      const real_t denom = norms[0] * norms[1];
      if (denom <= 1e-30) {
        scaled = 0.0;
      } else {
        scaled = eval.det_jacobian / denom;
      }
    }

    min_scaled = std::min(min_scaled, clamp(scaled, static_cast<real_t>(-1.0), static_cast<real_t>(1.0)));
  }

  // Return a quality value in [0,1], clamping inverted cases to 0.
  return std::max(static_cast<real_t>(0.0), min_scaled);
}

real_t MeshQuality::compute_shape_index(const MeshBase& mesh, index_t cell, Configuration cfg) {
  if (mesh.cell_shape(cell).family == CellFamily::Polyhedron) {
    return polyhedron_metric_via_tets(mesh, cell, Metric::ShapeIndex, cfg);
  }

  const CellShape shape = mesh.cell_shape(cell);
  const auto [_, n_nodes] = mesh.cell_vertices_span(cell);
  const auto pts = quality_sample_points(shape, n_nodes);
  if (pts.empty()) return 0.0;

  real_t min_q = 1.0;
  for (const auto& xi : pts) {
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const Jacobian& J = eval.jacobian;
    const int pdim = J.parametric_dim;
    if (pdim <= 1) continue;

    const auto G = jt_j(J);
    std::array<real_t, 3> w{};
    if (pdim == 2) {
      w = eigenvalues_sym_2x2(G[0][0], G[0][1], G[1][1]);
    } else {
      w = eigenvalues_sym_3x3(G);
    }

    real_t trace = 0.0;
    real_t detG = 1.0;
    for (int i = 0; i < pdim; ++i) {
      const real_t li = std::max(static_cast<real_t>(0), w[static_cast<size_t>(i)]);
      trace += li;
      detG *= li;
    }
    if (trace <= 1e-30 || detG <= 1e-300) {
      min_q = 0.0;
      continue;
    }

    const real_t q = static_cast<real_t>(pdim) *
                     std::pow(detG, static_cast<real_t>(1.0) / static_cast<real_t>(pdim)) /
                     trace;
    min_q = std::min(min_q, clamp(q, static_cast<real_t>(0.0), static_cast<real_t>(1.0)));
  }

  return min_q;
}

real_t MeshQuality::compute_relative_size_squared(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const CellFamily fam = mesh.cell_shape(cell).family;

  real_t sum = 0.0;
  size_t cnt = 0;
  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    if (mesh.cell_shape(static_cast<index_t>(c)).family != fam) continue;
    sum += cell_measure_estimate(mesh, static_cast<index_t>(c), cfg);
    ++cnt;
  }
  if (cnt == 0) return 0.0;
  const real_t avg = sum / static_cast<real_t>(cnt);
  if (!(avg > 0)) return 0.0;

  const real_t m = cell_measure_estimate(mesh, cell, cfg);
  if (!(m > 0)) return 0.0;

  const real_t r = m / avg;
  const real_t s = std::min(r, static_cast<real_t>(1.0) / r);
  return clamp(s * s, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
}

real_t MeshQuality::compute_shape_and_size(const MeshBase& mesh, index_t cell, Configuration cfg) {
  const real_t shape_q = compute_shape_index(mesh, cell, cfg);
  const real_t size_q = compute_relative_size_squared(mesh, cell, cfg);
  return clamp(shape_q * size_q, static_cast<real_t>(0.0), static_cast<real_t>(1.0));
}

} // namespace svmp
