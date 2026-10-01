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
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
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
// interior_shift > 0 moves the interior vertices along x by
// interior_shift * sin(pi x) sin(pi y) (the boundary is unchanged).
std::shared_ptr<svmp::Mesh> makeSquareTriangleMesh(int n,
                                                   double interior_shift = 0.0)
{
  auto base = std::make_shared<svmp::MeshBase>();
  std::vector<svmp::real_t> x;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      const double xi = static_cast<double>(i) / n;
      const double yj = static_cast<double>(j) / n;
      x.push_back(static_cast<svmp::real_t>(
          xi + interior_shift * std::sin(std::numbers::pi * xi) *
                    std::sin(std::numbers::pi * yj)));
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

TEST(LevelSetPdeVelocityExtension, AlgebraicRowsAreSatisfiedByTheSolvedExtension)
{
  const auto mesh = makeSquareTriangleMesh(8);
  const auto s = makeSetup(
      *mesh, [](double x, double y) { return y - 0.41 - 0.05 * std::sin(5 * x); },
      [](double x, double y) {
        return std::array<double, 2>{std::sin(2 * x) * std::cosh(y),
                                     std::cos(2 * x) * std::sinh(y)};
      });
  const std::vector<WallVelocityExtensionConstraint> walls{
      {.boundary_label = kSideWall, .constrained_components = {true, false, false}},
      {.boundary_label = kBottomWall, .constrained_components = {false, true, false}}};
  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    std::vector<double> w;
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows;
    PdeVelocityExtensionOptions options;
    options.op = op;
    (void)extendVelocityByPde(*mesh, svmp::MeshComm::self(), s.phi, s.source, 2u,
                              s.known, 2u,
                              std::span<const WallVelocityExtensionConstraint>(walls),
                              options, w, &rows);
    ASSERT_EQ(rows.size(), 2u * mesh->n_vertices());
    double max_residual = 0.0;
    bool harmonic_rows_are_convex = true;
    for (const auto& row : rows) {
      double residual = w[2 * static_cast<std::size_t>(row.vertex) +
                          static_cast<std::size_t>(row.component)];
      double sum = 0.0;
      bool from_extension = false;
      for (const auto& dep : row.dependencies) {
        const auto index = 2 * static_cast<std::size_t>(dep.vertex) +
                           static_cast<std::size_t>(dep.component);
        const bool source = dep.field ==
            svmp::FE::level_set::VelocityExtensionDependencyField::SourceVelocity;
        residual -= dep.coefficient * (source ? s.source[index] : w[index]);
        sum += dep.coefficient;
        from_extension = from_extension || !source;
        harmonic_rows_are_convex = harmonic_rows_are_convex && dep.coefficient >= -1e-14;
      }
      if (from_extension) {
        EXPECT_NEAR(sum, 1.0, 1e-12);  // constants are reproduced
      }
      max_residual = std::max(max_residual, std::abs(residual));
    }
    EXPECT_LT(max_residual, 1e-11)
        << application::core::pdeVelocityExtensionOperatorName(op);
    if (op == PdeVelocityExtensionOperator::Harmonic) {
      // Right-triangle P1 stiffness has nonpositive off-diagonals: each dry
      // value is a convex combination of its neighbors.
      EXPECT_TRUE(harmonic_rows_are_convex);
    }
  }
}

namespace {

using application::core::PdeVelocityExtensionCache;
using application::core::PdeVelocityExtensionMeshRevisions;
using application::core::PdeVelocityExtensionReport;
using ExtensionRows =
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>;

struct CachedRun {
  std::vector<double> extended;
  ExtensionRows rows;
  PdeVelocityExtensionReport report;
};

CachedRun runCached(const svmp::Mesh& mesh, const Setup& s,
                    PdeVelocityExtensionOperator op,
                    const std::vector<WallVelocityExtensionConstraint>& walls,
                    PdeVelocityExtensionCache* cache,
                    const PdeVelocityExtensionMeshRevisions& revisions = {},
                    bool with_rows = true, int band_layers = 0)
{
  CachedRun out;
  PdeVelocityExtensionOptions options;
  options.op = op;
  options.band_layers = band_layers;
  options.enforce_wall_impermeability = !walls.empty();
  out.report = extendVelocityByPde(
      mesh, svmp::MeshComm::self(), s.phi, s.source, 2u, s.known, 2u,
      std::span<const WallVelocityExtensionConstraint>(walls), options,
      out.extended, with_rows ? &out.rows : nullptr, cache, revisions);
  return out;
}

bool sameBits(double a, double b)
{
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// Extension values, rows and solve metrics agree bit for bit.
void expectBitwiseEqual(const CachedRun& a, const CachedRun& b)
{
  ASSERT_EQ(a.extended.size(), b.extended.size());
  for (std::size_t i = 0; i < a.extended.size(); ++i) {
    EXPECT_TRUE(sameBits(a.extended[i], b.extended[i])) << "entry " << i;
  }
  ASSERT_EQ(a.rows.size(), b.rows.size());
  for (std::size_t r = 0; r < a.rows.size(); ++r) {
    const auto& ra = a.rows[r];
    const auto& rb = b.rows[r];
    EXPECT_EQ(ra.vertex, rb.vertex);
    EXPECT_EQ(ra.component, rb.component);
    ASSERT_EQ(ra.dependencies.size(), rb.dependencies.size()) << "row " << r;
    for (std::size_t d = 0; d < ra.dependencies.size(); ++d) {
      EXPECT_EQ(ra.dependencies[d].field, rb.dependencies[d].field);
      EXPECT_EQ(ra.dependencies[d].vertex, rb.dependencies[d].vertex);
      EXPECT_EQ(ra.dependencies[d].component, rb.dependencies[d].component);
      EXPECT_TRUE(sameBits(ra.dependencies[d].coefficient,
                           rb.dependencies[d].coefficient))
          << "row " << r << " dependency " << d;
    }
  }
  EXPECT_TRUE(sameBits(a.report.max_relative_residual,
                       b.report.max_relative_residual));
  EXPECT_TRUE(sameBits(a.report.max_extended_speed, b.report.max_extended_speed));
  EXPECT_EQ(a.report.unknowns, b.report.unknowns);
  EXPECT_EQ(a.report.wall_fixed, b.report.wall_fixed);
  EXPECT_EQ(a.report.extension_cells, b.report.extension_cells);
}

const std::vector<WallVelocityExtensionConstraint>& cacheTestWalls()
{
  static const std::vector<WallVelocityExtensionConstraint> walls{
      {.boundary_label = kSideWall, .constrained_components = {true, false, false}},
      {.boundary_label = kBottomWall, .constrained_components = {false, true, false}}};
  return walls;
}

double wavyPhi(double x, double y) { return y - 0.41 - 0.05 * std::sin(5 * x); }

std::array<double, 2> firstVelocity(double x, double y)
{
  return {std::sin(2 * x) * std::cosh(y), std::cos(2 * x) * std::sinh(y)};
}

std::array<double, 2> secondVelocity(double x, double y)
{
  return {0.3 - 0.8 * x * y, 1.1 + 0.25 * std::cos(3 * x)};
}

} // namespace

TEST(LevelSetPdeVelocityExtensionCache, ReusesTheFactorizationWhileTheKeyIsUnchanged)
{
  const auto mesh = makeSquareTriangleMesh(10);
  const auto first = makeSetup(*mesh, wavyPhi, firstVelocity);
  const auto second = makeSetup(*mesh, wavyPhi, secondVelocity);
  const auto& walls = cacheTestWalls();
  const auto op = PdeVelocityExtensionOperator::Harmonic;

  PdeVelocityExtensionCache cache;
  EXPECT_TRUE(cache.empty());
  const auto miss = runCached(*mesh, first, op, walls, &cache);
  EXPECT_FALSE(miss.report.reused_factorization);
  EXPECT_FALSE(cache.empty());
  EXPECT_EQ(cache.statistics().misses, 1u);
  EXPECT_EQ(cache.statistics().hits, 0u);
  EXPECT_GT(cache.statistics().bytes, 0u);

  // A new velocity with the same mesh, known set and walls reuses everything.
  const auto hit = runCached(*mesh, second, op, walls, &cache);
  EXPECT_TRUE(hit.report.reused_factorization);
  const auto hit_again = runCached(*mesh, first, op, walls, &cache);
  EXPECT_TRUE(hit_again.report.reused_factorization);
  // A call without rows (prescribed coupling) reuses the same entry.
  const auto hit_without_rows =
      runCached(*mesh, second, op, walls, &cache, {}, /*with_rows=*/false);
  EXPECT_TRUE(hit_without_rows.report.reused_factorization);
  EXPECT_TRUE(hit_without_rows.rows.empty());
  EXPECT_EQ(cache.statistics().misses, 1u);
  EXPECT_EQ(cache.statistics().hits, 3u);
  EXPECT_GE(cache.statistics().peak_bytes, cache.statistics().bytes);

  const auto first_reference = runCached(*mesh, first, op, walls, nullptr);
  const auto second_reference = runCached(*mesh, second, op, walls, nullptr);
  EXPECT_FALSE(first_reference.report.reused_factorization);
  expectBitwiseEqual(miss, first_reference);
  expectBitwiseEqual(hit, second_reference);
  expectBitwiseEqual(hit_again, first_reference);
  ASSERT_EQ(hit_without_rows.extended.size(), second_reference.extended.size());
  for (std::size_t i = 0; i < hit_without_rows.extended.size(); ++i) {
    EXPECT_TRUE(sameBits(hit_without_rows.extended[i],
                         second_reference.extended[i]));
  }
}

TEST(LevelSetPdeVelocityExtensionCache, RefactorsWhenTheKnownSetWallsOrMeshChange)
{
  const auto mesh = makeSquareTriangleMesh(10);
  const auto base = makeSetup(*mesh, wavyPhi, firstVelocity);
  const auto& walls = cacheTestWalls();
  const auto op = PdeVelocityExtensionOperator::Harmonic;
  PdeVelocityExtensionCache cache;

  const auto expect_miss_matching_reference =
      [&](const svmp::Mesh& m, const auto& s,
          const std::vector<WallVelocityExtensionConstraint>& w,
          const PdeVelocityExtensionMeshRevisions& revisions,
          const char* what) {
        SCOPED_TRACE(what);
        const auto cached = runCached(m, s, op, w, &cache, revisions);
        EXPECT_FALSE(cached.report.reused_factorization);
        expectBitwiseEqual(cached, runCached(m, s, op, w, nullptr));
        const auto again = runCached(m, s, op, w, &cache, revisions);
        EXPECT_TRUE(again.report.reused_factorization);
        expectBitwiseEqual(again, cached);
      };

  expect_miss_matching_reference(*mesh, base, walls, {}, "initial");

  // Known set: one more dry interior vertex becomes known.
  auto more_known = base;
  bool marked = false;
  for (std::size_t v = 0; v < mesh->n_vertices() && !marked; ++v) {
    const auto p = vertexPoint(*mesh, v);
    if (more_known.known[v] == 0u && p[0] > 0.25 && p[0] < 0.75 && p[1] > 0.7) {
      more_known.known[v] = 1u;
      marked = true;
    }
  }
  ASSERT_TRUE(marked);
  expect_miss_matching_reference(*mesh, more_known, walls, {}, "known set");
  expect_miss_matching_reference(*mesh, base, walls, {}, "known set restored");

  // A different mesh with the same topology and the same revisions: the
  // element matrices differ, so the content key differs.
  const auto moved = makeSquareTriangleMesh(10, /*interior_shift=*/0.02);
  const auto moved_setup = makeSetup(*moved, wavyPhi, firstVelocity);
  expect_miss_matching_reference(*moved, moved_setup, walls, {}, "moved mesh");
  expect_miss_matching_reference(*mesh, base, walls, {}, "original mesh");

  // Band truncation changes the dry region.
  {
    SCOPED_TRACE("band layers");
    const auto banded = runCached(*mesh, base, op, walls, &cache, {}, true, 2);
    EXPECT_FALSE(banded.report.reused_factorization);
    expectBitwiseEqual(banded,
                       runCached(*mesh, base, op, walls, nullptr, {}, true, 2));
  }

  // Walls: the side walls alone, then no wall condition.
  const std::vector<WallVelocityExtensionConstraint> side_only{walls.front()};
  expect_miss_matching_reference(*mesh, base, side_only, {}, "walls");
  expect_miss_matching_reference(*mesh, base, {}, {}, "no walls");
  expect_miss_matching_reference(*mesh, base, walls, {}, "walls restored");

  // Mesh revisions: any revision change refactors.
  expect_miss_matching_reference(*mesh, base, walls,
                                 PdeVelocityExtensionMeshRevisions{.geometry = 1u},
                                 "geometry revision");
  expect_miss_matching_reference(
      *mesh, base, walls,
      PdeVelocityExtensionMeshRevisions{.geometry = 1u, .topology = 2u},
      "topology revision");
  expect_miss_matching_reference(
      *mesh, base, walls,
      PdeVelocityExtensionMeshRevisions{
          .geometry = 1u, .topology = 2u, .ownership = 3u},
      "ownership revision");
  expect_miss_matching_reference(
      *mesh, base, walls,
      PdeVelocityExtensionMeshRevisions{
          .geometry = 1u, .topology = 2u, .ownership = 3u, .numbering = 4u},
      "numbering revision");
}

TEST(LevelSetPdeVelocityExtensionCache, NormalOperatorReusesOnlyIdenticalNormals)
{
  const auto mesh = makeSquareTriangleMesh(10);
  // Tilted flat interface; a field linear in t = x + 0.2 y is constant along
  // the normals and is therefore reproduced exactly.
  const auto phi = [](double x, double y) { return (y - 0.4) - 0.2 * (x - 0.5); };
  const auto linear_u = [](double x, double y) {
    const double t = x + 0.2 * y;
    return std::array<double, 2>{0.7 - 1.3 * t, 0.1 + 2.2 * t};
  };
  const auto op = PdeVelocityExtensionOperator::LeastSquaresNormal;
  const auto first = makeSetup(*mesh, phi, firstVelocity);
  const auto linear = makeSetup(*mesh, phi, linear_u);
  PdeVelocityExtensionCache cache;

  const auto miss = runCached(*mesh, first, op, {}, &cache);
  EXPECT_FALSE(miss.report.reused_factorization);
  expectBitwiseEqual(miss, runCached(*mesh, first, op, {}, nullptr));

  const auto hit = runCached(*mesh, linear, op, {}, &cache);
  EXPECT_TRUE(hit.report.reused_factorization);
  expectBitwiseEqual(hit, runCached(*mesh, linear, op, {}, nullptr));
  EXPECT_LT(hit.report.max_relative_residual, 1e-10);
  for (std::size_t i = 0; i < hit.extended.size(); ++i) {
    EXPECT_NEAR(hit.extended[i], linear.source[i], 1e-11) << "entry " << i;
  }

  // Doubling phi is exact in binary and leaves every normal bitwise
  // unchanged: the matrix is identical and is reused.
  auto doubled = linear;
  for (auto& value : doubled.phi) {
    value *= 2.0;
  }
  const auto doubled_hit = runCached(*mesh, doubled, op, {}, &cache);
  EXPECT_TRUE(doubled_hit.report.reused_factorization);
  expectBitwiseEqual(doubled_hit, runCached(*mesh, doubled, op, {}, nullptr));

  // Bending phi in the dry region keeps the known set but changes the
  // normals there: the operator is refactored.
  auto bent = linear;
  for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
    const auto p = vertexPoint(*mesh, v);
    if (p[1] > 0.7) {
      bent.phi[v] += 0.1 * (p[0] - 0.5) * (p[1] - 0.7);
    }
  }
  const auto bent_miss = runCached(*mesh, bent, op, {}, &cache);
  EXPECT_FALSE(bent_miss.report.reused_factorization);
  expectBitwiseEqual(bent_miss, runCached(*mesh, bent, op, {}, nullptr));
  EXPECT_EQ(cache.statistics().hits, 2u);
  EXPECT_EQ(cache.statistics().misses, 2u);
}

TEST(LevelSetPdeVelocityExtensionCache, CachedAndUncachedResultsAreBitwiseIdentical)
{
  const auto mesh = makeSquareTriangleMesh(12);
  const auto first = makeSetup(*mesh, wavyPhi, firstVelocity);
  const auto second = makeSetup(*mesh, wavyPhi, secondVelocity);
  const std::vector<WallVelocityExtensionConstraint> no_walls;
  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    for (const auto* walls : {&cacheTestWalls(), &no_walls}) {
      for (const bool with_rows : {true, false}) {
        for (const int band_layers : {0, 3}) {
          SCOPED_TRACE(std::string(application::core::pdeVelocityExtensionOperatorName(op)) +
                       (walls->empty() ? " no walls" : " walls") +
                       (with_rows ? " rows" : " no rows") + " band " +
                       std::to_string(band_layers));
          PdeVelocityExtensionCache cache;
          const auto miss = runCached(*mesh, first, op, *walls, &cache, {},
                                      with_rows, band_layers);
          const auto hit = runCached(*mesh, second, op, *walls, &cache, {},
                                     with_rows, band_layers);
          EXPECT_FALSE(miss.report.reused_factorization);
          EXPECT_TRUE(hit.report.reused_factorization);
          expectBitwiseEqual(miss, runCached(*mesh, first, op, *walls, nullptr,
                                             {}, with_rows, band_layers));
          expectBitwiseEqual(hit, runCached(*mesh, second, op, *walls, nullptr,
                                            {}, with_rows, band_layers));
        }
      }
    }
  }
}

TEST(LevelSetPdeVelocityExtensionCache, RowsAreBuiltOnFirstRequestAfterAReuse)
{
  const auto mesh = makeSquareTriangleMesh(10);
  const auto first = makeSetup(*mesh, wavyPhi, firstVelocity);
  const auto second = makeSetup(*mesh, wavyPhi, secondVelocity);
  const auto& walls = cacheTestWalls();
  const auto op = PdeVelocityExtensionOperator::Harmonic;
  PdeVelocityExtensionCache cache;
  const auto without_rows =
      runCached(*mesh, first, op, walls, &cache, {}, /*with_rows=*/false);
  EXPECT_FALSE(without_rows.report.reused_factorization);
  const auto with_rows = runCached(*mesh, second, op, walls, &cache);
  EXPECT_TRUE(with_rows.report.reused_factorization);
  EXPECT_FALSE(with_rows.rows.empty());
  expectBitwiseEqual(with_rows, runCached(*mesh, second, op, walls, nullptr));
  const auto rows_again = runCached(*mesh, first, op, walls, &cache);
  EXPECT_TRUE(rows_again.report.reused_factorization);
  expectBitwiseEqual(rows_again, runCached(*mesh, first, op, walls, nullptr));
}

TEST(LevelSetPdeVelocityExtensionCache, AFailedCallLeavesTheCacheEmpty)
{
  const auto mesh = makeSquareTriangleMesh(8);
  const auto good = makeSetup(*mesh, wavyPhi, firstVelocity);
  const auto& walls = cacheTestWalls();
  const auto op = PdeVelocityExtensionOperator::Harmonic;
  PdeVelocityExtensionCache cache;
  (void)runCached(*mesh, good, op, walls, &cache);
  ASSERT_FALSE(cache.empty());

  // Same key, but a non-finite known velocity: the call throws and drops the
  // cached entry.
  auto bad = good;
  std::size_t known_vertex = mesh->n_vertices();
  for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
    if (bad.known[v] != 0u) {
      known_vertex = v;
      break;
    }
  }
  ASSERT_LT(known_vertex, mesh->n_vertices());
  bad.source[2 * known_vertex] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)runCached(*mesh, bad, op, walls, &cache), std::runtime_error);
  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.statistics().bytes, 0u);

  const auto rebuilt = runCached(*mesh, good, op, walls, &cache);
  EXPECT_FALSE(rebuilt.report.reused_factorization);
  expectBitwiseEqual(rebuilt, runCached(*mesh, good, op, walls, nullptr));

  // A failure on the build path leaves no entry either.
  const std::vector<WallVelocityExtensionConstraint> tangential{
      {.boundary_label = kSideWall, .constrained_components = {false, true, false}}};
  EXPECT_THROW((void)runCached(*mesh, good, op, tangential, &cache),
               std::runtime_error);
  EXPECT_TRUE(cache.empty());
}
