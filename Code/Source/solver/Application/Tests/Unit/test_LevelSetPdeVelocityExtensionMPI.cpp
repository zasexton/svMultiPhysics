/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Two-rank parity of the PDE velocity extension: the distributed result on
// every local vertex (owned and ghost) must equal the serial result on the
// same global mesh.

#include <gtest/gtest.h>

#include "Application/Core/LevelSetPdeVelocityExtension.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

namespace {

using application::core::PdeVelocityExtensionOperator;
using application::core::PdeVelocityExtensionOptions;
using application::core::WallVelocityExtensionConstraint;

constexpr svmp::label_t kPdeSideWall = 21;
constexpr int kPdeCells = 12;

struct PdeMeshArrays {
  std::vector<svmp::real_t> x;
  std::vector<svmp::offset_t> offsets{0};
  std::vector<svmp::index_t> connectivity;
  std::vector<svmp::CellShape> shapes;
};

PdeMeshArrays makePdeArrays()
{
  PdeMeshArrays a;
  const int n = kPdeCells;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      a.x.push_back(static_cast<svmp::real_t>(i) / n);
      a.x.push_back(static_cast<svmp::real_t>(j) / n);
    }
  }
  const auto vid = [n](int i, int j) {
    return static_cast<svmp::index_t>(j * (n + 1) + i);
  };
  svmp::CellShape triangle{};
  triangle.family = svmp::CellFamily::Triangle;
  triangle.num_corners = 3;
  triangle.order = 1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const auto p = vid(i, j), q = vid(i + 1, j), r = vid(i + 1, j + 1),
                 s = vid(i, j + 1);
      const std::array<std::array<svmp::index_t, 3>, 2> tris =
          (i + j) % 2 == 0
              ? std::array<std::array<svmp::index_t, 3>, 2>{{{p, q, r}, {p, r, s}}}
              : std::array<std::array<svmp::index_t, 3>, 2>{{{p, q, s}, {q, r, s}}};
      for (const auto& tri : tris) {
        a.connectivity.insert(a.connectivity.end(), tri.begin(), tri.end());
        a.offsets.push_back(static_cast<svmp::offset_t>(a.connectivity.size()));
        a.shapes.push_back(triangle);
      }
    }
  }
  return a;
}

void labelSideWalls(svmp::Mesh& mesh)
{
  auto& local = mesh.local_mesh();
  for (const auto face : local.boundary_faces()) {
    const auto center = local.face_center(face);
    if (std::abs(center[0]) < 1e-12 || std::abs(center[0] - 1.0) < 1e-12) {
      mesh.set_boundary_label(face, kPdeSideWall);
    }
  }
}

double phiAt(double x, double y) { return y - 0.41 - 0.08 * std::sin(5.0 * x); }

std::array<double, 2> uAt(double x, double y)
{
  return {std::sin(2.0 * x) * std::cosh(y), std::cos(2.0 * x) * std::sinh(y)};
}

// Extension on `mesh`, returned per vertex coordinate (rounded key).
std::map<std::pair<long long, long long>, std::array<double, 2>> extend(
    const svmp::Mesh& mesh, const svmp::MeshComm& comm,
    PdeVelocityExtensionOperator op)
{
  const auto n = mesh.n_vertices();
  const auto& X = mesh.X_ref();
  std::vector<double> phi(n), source(2 * n);
  std::vector<std::uint8_t> known(n, 0u);
  for (std::size_t v = 0; v < n; ++v) {
    const double x = X[2 * v], y = X[2 * v + 1];
    phi[v] = phiAt(x, y);
    const auto u = uAt(x, y);
    source[2 * v] = u[0];
    source[2 * v + 1] = u[1];
    known[v] = phi[v] < 0.0 ? 1u : 0u;
  }
  const auto& local = mesh.local_mesh();
  for (svmp::index_t c = 0; c < local.n_cells(); ++c) {
    auto [cv, count] = local.cell_vertices_span(c);
    bool neg = false, pos = false;
    for (std::size_t i = 0; i < count; ++i) {
      neg = neg || phi[static_cast<std::size_t>(cv[i])] < 0.0;
      pos = pos || phi[static_cast<std::size_t>(cv[i])] >= 0.0;
    }
    if (neg && pos) {
      for (std::size_t i = 0; i < count; ++i) {
        known[static_cast<std::size_t>(cv[i])] = 1u;
      }
    }
  }
  // A ghost vertex may miss a cut cell of its star; the extension takes the
  // union of the known mask over the communicator.
  const std::vector<WallVelocityExtensionConstraint> walls{
      {.boundary_label = kPdeSideWall,
       .constrained_components = {true, false, false}}};
  PdeVelocityExtensionOptions options;
  options.op = op;
  std::vector<double> out;
  (void)application::core::extendVelocityByPde(
      mesh, comm, phi, source, 2u, known, 2u,
      std::span<const WallVelocityExtensionConstraint>(walls), options, out);
  std::map<std::pair<long long, long long>, std::array<double, 2>> result;
  for (std::size_t v = 0; v < n; ++v) {
    const auto key = std::make_pair(std::llround(X[2 * v] * kPdeCells),
                                    std::llround(X[2 * v + 1] * kPdeCells));
    result[key] = {out[2 * v], out[2 * v + 1]};
  }
  return result;
}

} // namespace

TEST(LevelSetPdeVelocityExtensionMPI, TwoRankResultMatchesSerialOnEveryLocalVertex)
{
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2) << "This parity test requires exactly two ranks.";

  const auto arrays = makePdeArrays();
  auto distributed = std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
  distributed->build_from_arrays_global_and_partition(
      2, arrays.x, arrays.offsets, arrays.connectivity, arrays.shapes,
      svmp::PartitionHint::Cells, /*ghost_layers=*/1,
      {{"partition_method", "block"}});
  labelSideWalls(*distributed);
  ASSERT_LT(distributed->n_vertices(),
            static_cast<std::size_t>((kPdeCells + 1) * (kPdeCells + 1)));

  auto base = std::make_shared<svmp::MeshBase>();
  base->build_from_arrays(2, arrays.x, arrays.offsets, arrays.connectivity,
                          arrays.shapes);
  base->finalize();
  auto serial = svmp::create_mesh(std::move(base));
  labelSideWalls(*serial);

  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    const auto reference = extend(*serial, svmp::MeshComm::self(), op);
    const auto parallel = extend(*distributed, svmp::MeshComm(MPI_COMM_WORLD), op);
    int local_failures = 0;
    for (const auto& [key, value] : parallel) {
      const auto found = reference.find(key);
      ASSERT_NE(found, reference.end());
      for (int c = 0; c < 2; ++c) {
        const double scale = std::max(1.0, std::abs(found->second[c]));
        if (std::abs(value[c] - found->second[c]) > 1e-13 * scale) {
          ++local_failures;
        }
      }
    }
    int failures = 0;
    MPI_Allreduce(&local_failures, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(failures, 0) << application::core::pdeVelocityExtensionOperatorName(op);
  }
}
