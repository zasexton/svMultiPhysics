/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Application/Core/LevelSetPdeVelocityExtension.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using application::core::PdeVelocityExtensionOperator;
using application::core::PdeVelocityExtensionOptions;
using application::core::WallVelocityExtensionConstraint;
using application::core::extendVelocityByPde;
using application::core::pdeVelocityExtensionOperatorFromToken;

constexpr svmp::label_t kSideWall = 11;
constexpr svmp::label_t kBottomWall = 12;
constexpr svmp::label_t kTopWall = 13;

// Unit square split into n x n cells, two triangles each, diagonal
// alternating with cell parity.  Side, bottom and top faces are labeled.
std::shared_ptr<svmp::Mesh> makeSquareTriangleMesh(int n)
{
  auto base = std::make_shared<svmp::MeshBase>();
  std::vector<svmp::real_t> x;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      x.push_back(static_cast<svmp::real_t>(i) / n);
      x.push_back(static_cast<svmp::real_t>(j) / n);
    }
  }
  const auto vid = [n](int i, int j) {
    return static_cast<svmp::index_t>(j * (n + 1) + i);
  };
  std::vector<svmp::offset_t> offsets{0};
  std::vector<svmp::index_t> connectivity;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const auto a = vid(i, j), b = vid(i + 1, j), c = vid(i + 1, j + 1),
                 d = vid(i, j + 1);
      const std::array<std::array<svmp::index_t, 3>, 2> tris =
          (i + j) % 2 == 0
              ? std::array<std::array<svmp::index_t, 3>, 2>{{{a, b, c}, {a, c, d}}}
              : std::array<std::array<svmp::index_t, 3>, 2>{{{a, b, d}, {b, c, d}}};
      for (const auto& tri : tris) {
        connectivity.insert(connectivity.end(), tri.begin(), tri.end());
        offsets.push_back(static_cast<svmp::offset_t>(connectivity.size()));
      }
    }
  }
  svmp::CellShape triangle{};
  triangle.family = svmp::CellFamily::Triangle;
  triangle.num_corners = 3;
  triangle.order = 1;
  base->build_from_arrays(2, x, offsets, connectivity,
                          std::vector<svmp::CellShape>(offsets.size() - 1u,
                                                       triangle));
  base->finalize();
  auto mesh = svmp::create_mesh(std::move(base));
  auto& local = mesh->local_mesh();
  for (const auto face : local.boundary_faces()) {
    const auto center = local.face_center(face);
    if (std::abs(center[0]) < 1e-12 || std::abs(center[0] - 1.0) < 1e-12) {
      mesh->set_boundary_label(face, kSideWall);
    } else if (std::abs(center[1]) < 1e-12) {
      mesh->set_boundary_label(face, kBottomWall);
    } else {
      mesh->set_boundary_label(face, kTopWall);
    }
  }
  return mesh;
}

std::array<double, 2> vertexPoint(const svmp::Mesh& mesh, std::size_t v)
{
  const auto& x = mesh.X_ref();
  return {static_cast<double>(x[2 * v]), static_cast<double>(x[2 * v + 1])};
}

struct Setup {
  std::vector<double> phi;
  std::vector<double> source;
  std::vector<std::uint8_t> known;
};

// Known set: wet vertices (phi < 0) plus every vertex of a cell the zero
// level set crosses, as in the production trace seed.
Setup makeSetup(const svmp::Mesh& mesh,
                const std::function<double(double, double)>& phi_fn,
                const std::function<std::array<double, 2>(double, double)>& u_fn)
{
  Setup s;
  const auto n = mesh.n_vertices();
  s.phi.resize(n);
  s.source.resize(2 * n);
  s.known.assign(n, 0u);
  for (std::size_t v = 0; v < n; ++v) {
    const auto p = vertexPoint(mesh, v);
    s.phi[v] = phi_fn(p[0], p[1]);
    const auto u = u_fn(p[0], p[1]);
    s.source[2 * v] = u[0];
    s.source[2 * v + 1] = u[1];
    s.known[v] = s.phi[v] < 0.0 ? 1u : 0u;
  }
  const auto& local = mesh.local_mesh();
  for (svmp::index_t c = 0; c < local.n_cells(); ++c) {
    auto [cv, count] = local.cell_vertices_span(c);
    bool neg = false, pos = false;
    for (std::size_t i = 0; i < count; ++i) {
      neg = neg || s.phi[static_cast<std::size_t>(cv[i])] < 0.0;
      pos = pos || s.phi[static_cast<std::size_t>(cv[i])] >= 0.0;
    }
    if (neg && pos) {
      for (std::size_t i = 0; i < count; ++i) {
        s.known[static_cast<std::size_t>(cv[i])] = 1u;
      }
    }
  }
  return s;
}

std::vector<double> run(const svmp::Mesh& mesh, const Setup& s,
                        PdeVelocityExtensionOperator op,
                        const std::vector<WallVelocityExtensionConstraint>& walls,
                        application::core::PdeVelocityExtensionReport* report = nullptr,
                        int band_layers = 0)
{
  std::vector<double> out;
  PdeVelocityExtensionOptions options;
  options.op = op;
  options.band_layers = band_layers;
  options.enforce_wall_impermeability = !walls.empty();
  const auto r = extendVelocityByPde(
      mesh, svmp::MeshComm::self(), s.phi, s.source, 2u, s.known, 2u,
      std::span<const WallVelocityExtensionConstraint>(walls), options, out);
  if (report != nullptr) {
    *report = r;
  }
  return out;
}

} // namespace

TEST(LevelSetPdeVelocityExtension, ParsesOnlyExplicitPdeMethodTokens)
{
  EXPECT_EQ(pdeVelocityExtensionOperatorFromToken("pde_harmonic"),
            PdeVelocityExtensionOperator::Harmonic);
  EXPECT_EQ(pdeVelocityExtensionOperatorFromToken("PDE-Harmonic"),
            PdeVelocityExtensionOperator::Harmonic);
  EXPECT_EQ(pdeVelocityExtensionOperatorFromToken("pde_normal"),
            PdeVelocityExtensionOperator::LeastSquaresNormal);
  EXPECT_EQ(pdeVelocityExtensionOperatorFromToken("pde_least_squares_normal"),
            PdeVelocityExtensionOperator::LeastSquaresNormal);
  EXPECT_FALSE(pdeVelocityExtensionOperatorFromToken("wall_compatible_normal"));
  EXPECT_FALSE(pdeVelocityExtensionOperatorFromToken("harmonic"));
  EXPECT_FALSE(pdeVelocityExtensionOperatorFromToken(""));
}

TEST(LevelSetPdeVelocityExtension, HarmonicReproducesLinearFieldWithDirichletBoundary)
{
  const auto mesh = makeSquareTriangleMesh(8);
  auto s = makeSetup(
      *mesh, [](double, double y) { return y - 0.43; },
      [](double x, double y) {
        return std::array<double, 2>{0.3 + 1.7 * x - 0.4 * y,
                                     -0.2 + 0.5 * x + 2.1 * y};
      });
  // Dirichlet data on the whole boundary: a P1 linear field is then the
  // exact discrete harmonic extension.
  for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
    const auto p = vertexPoint(*mesh, v);
    if (p[0] < 1e-12 || p[0] > 1 - 1e-12 || p[1] < 1e-12 || p[1] > 1 - 1e-12) {
      s.known[v] = 1u;
    }
  }
  application::core::PdeVelocityExtensionReport report;
  const auto w = run(*mesh, s, PdeVelocityExtensionOperator::Harmonic, {}, &report);
  EXPECT_GT(report.extension_vertices, 0u);
  for (std::size_t i = 0; i < w.size(); ++i) {
    EXPECT_NEAR(w[i], s.source[i], 1e-12) << "entry " << i;
  }
}

TEST(LevelSetPdeVelocityExtension, NormalExtensionIsConstantAlongNormals)
{
  const auto mesh = makeSquareTriangleMesh(10);
  // Tilted flat interface: n is proportional to (-0.2, 1).  A field that is
  // linear in the tangential coordinate t = x + 0.2 y is constant along n.
  const auto phi = [](double x, double y) { return (y - 0.4) - 0.2 * (x - 0.5); };
  const auto u = [](double x, double y) {
    const double t = x + 0.2 * y;
    return std::array<double, 2>{0.7 - 1.3 * t, 0.1 + 2.2 * t};
  };
  const auto s = makeSetup(*mesh, phi, u);
  application::core::PdeVelocityExtensionReport report;
  const auto w =
      run(*mesh, s, PdeVelocityExtensionOperator::LeastSquaresNormal, {}, &report);
  EXPECT_GT(report.extension_vertices, 20u);
  EXPECT_LT(report.max_relative_residual, 1e-10);
  for (std::size_t i = 0; i < w.size(); ++i) {
    EXPECT_NEAR(w[i], s.source[i], 1e-11) << "entry " << i;
  }
  // The harmonic extension of the same data (natural outer boundary) is not
  // constant along the normals.
  const auto h = run(*mesh, s, PdeVelocityExtensionOperator::Harmonic, {});
  double max_difference = 0.0;
  for (std::size_t i = 0; i < h.size(); ++i) {
    max_difference = std::max(max_difference, std::abs(h[i] - s.source[i]));
  }
  EXPECT_GT(max_difference, 1e-3);
}

TEST(LevelSetPdeVelocityExtension, KnownVerticesKeepTheSourceVelocity)
{
  const auto mesh = makeSquareTriangleMesh(8);
  const auto s = makeSetup(
      *mesh, [](double x, double y) { return y - 0.37 - 0.1 * std::sin(6 * x); },
      [](double x, double y) {
        return std::array<double, 2>{std::sin(3 * x) * std::cosh(y),
                                     std::cos(3 * x) * std::sinh(y)};
      });
  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    const auto w = run(*mesh, s, op, {});
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
      if (s.known[v] != 0u) {
        EXPECT_EQ(w[2 * v], s.source[2 * v]);
        EXPECT_EQ(w[2 * v + 1], s.source[2 * v + 1]);
      }
    }
  }
}

TEST(LevelSetPdeVelocityExtension, WallsZeroOnlyTheConstrainedComponent)
{
  const auto mesh = makeSquareTriangleMesh(8);
  const auto s = makeSetup(
      *mesh, [](double, double y) { return y - 0.43; },
      [](double x, double y) {
        return std::array<double, 2>{0.4 + x * (1 - x) + 0.2 * y, 1.0 + 0.3 * x};
      });
  // Free slip on the side walls (x component), no condition on the top.
  const std::vector<WallVelocityExtensionConstraint> walls{
      {.boundary_label = kSideWall, .constrained_components = {true, false, false}},
      {.boundary_label = kBottomWall, .constrained_components = {false, true, false}}};
  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    application::core::PdeVelocityExtensionReport report;
    const auto w = run(*mesh, s, op, walls, &report);
    EXPECT_EQ(report.max_wall_normal_velocity, 0.0);
    EXPECT_GT(report.wall_fixed[0], 0u);
    EXPECT_EQ(report.wall_fixed[1], 0u);  // the bottom wall is wet
    bool saw_tangential = false;
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
      const auto p = vertexPoint(*mesh, v);
      const bool side = p[0] < 1e-12 || p[0] > 1 - 1e-12;
      if (side && s.known[v] == 0u) {
        EXPECT_EQ(w[2 * v], 0.0);
        saw_tangential = saw_tangential || std::abs(w[2 * v + 1]) > 0.5;
      }
    }
    EXPECT_TRUE(saw_tangential);
    if (op == PdeVelocityExtensionOperator::LeastSquaresNormal) {
      // n = e_y: the tangential (y) component is carried up the wall unchanged.
      for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        const auto p = vertexPoint(*mesh, v);
        if (p[0] < 1e-12 && s.known[v] == 0u) {
          EXPECT_NEAR(w[2 * v + 1], 1.0, 1e-11);
        }
      }
    }
  }
}

TEST(LevelSetPdeVelocityExtension, BandTruncationLeavesOutsideVerticesAtZero)
{
  const auto mesh = makeSquareTriangleMesh(10);
  const auto s = makeSetup(
      *mesh, [](double, double y) { return y - 0.33; },
      [](double x, double) { return std::array<double, 2>{1.0 + x, 2.0}; });
  application::core::PdeVelocityExtensionReport full_report;
  application::core::PdeVelocityExtensionReport band_report;
  (void)run(*mesh, s, PdeVelocityExtensionOperator::Harmonic, {}, &full_report);
  const auto band = run(*mesh, s, PdeVelocityExtensionOperator::Harmonic, {},
                        &band_report, /*band_layers=*/2);
  EXPECT_EQ(full_report.outside_vertices, 0u);
  EXPECT_GT(band_report.outside_vertices, 0u);
  EXPECT_LT(band_report.extension_vertices, full_report.extension_vertices);
  for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
    if (vertexPoint(*mesh, v)[1] > 0.95) {
      EXPECT_EQ(band[2 * v], 0.0);
      EXPECT_EQ(band[2 * v + 1], 0.0);
    }
  }
}

TEST(LevelSetPdeVelocityExtension, RejectsInvalidInput)
{
  const auto mesh = makeSquareTriangleMesh(4);
  auto s = makeSetup(
      *mesh, [](double, double y) { return y - 0.43; },
      [](double, double) { return std::array<double, 2>{1.0, 0.0}; });
  std::vector<double> out;
  PdeVelocityExtensionOptions options;
  options.enforce_wall_impermeability = false;
  const std::vector<double> short_phi(3, 0.0);
  EXPECT_THROW((void)extendVelocityByPde(*mesh, svmp::MeshComm::self(), short_phi,
                                         s.source, 2u, s.known, 2u, {}, options, out),
               std::invalid_argument);
  options.enforce_wall_impermeability = true;
  EXPECT_THROW((void)extendVelocityByPde(*mesh, svmp::MeshComm::self(), s.phi,
                                         s.source, 2u, s.known, 2u, {}, options, out),
               std::invalid_argument);
  const std::vector<WallVelocityExtensionConstraint> projected{
      {.boundary_label = kSideWall,
       .constrained_components = {true, false, false},
       .project_boundary_normal = true}};
  EXPECT_THROW((void)extendVelocityByPde(
                   *mesh, svmp::MeshComm::self(), s.phi, s.source, 2u, s.known,
                   2u, std::span<const WallVelocityExtensionConstraint>(projected),
                   options, out),
               std::invalid_argument);
  // A mask that does not constrain the wall normal is not impermeable.
  const std::vector<WallVelocityExtensionConstraint> tangential{
      {.boundary_label = kSideWall, .constrained_components = {false, true, false}}};
  EXPECT_THROW((void)extendVelocityByPde(
                   *mesh, svmp::MeshComm::self(), s.phi, s.source, 2u, s.known,
                   2u, std::span<const WallVelocityExtensionConstraint>(tangential),
                   options, out),
               std::runtime_error);
  // The normal is undefined where the level set is flat.
  options.enforce_wall_impermeability = false;
  options.op = PdeVelocityExtensionOperator::LeastSquaresNormal;
  auto flat = s;
  for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
    if (flat.known[v] == 0u) {
      flat.phi[v] = 1.0;
    }
  }
  EXPECT_THROW((void)extendVelocityByPde(*mesh, svmp::MeshComm::self(), flat.phi,
                                         flat.source, 2u, flat.known, 2u, {},
                                         options, out),
               std::runtime_error);
}
