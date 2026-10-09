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
#include "FE/Backends/MUMPS/MumpsDistributedSolver.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
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
    PdeVelocityExtensionOperator op,
    application::core::PdeVelocityExtensionFactorization factorization =
        application::core::PdeVelocityExtensionFactorization::LuColamd)
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
  options.factorization = factorization;
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

namespace {

struct PdeCachedRun {
  std::vector<double> extended;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows;
  bool reused{false};
  bool distributed{false};
  bool self_checked{false};
};

// Known set as in extend(), plus the vertices at the given (i, j) lattice
// points; source velocity uAt scaled by `scale`.
PdeCachedRun extendCached(const svmp::Mesh& mesh, const svmp::MeshComm& comm,
                          PdeVelocityExtensionOperator op, double scale,
                          const std::vector<std::pair<long long, long long>>& extra,
                          application::core::PdeVelocityExtensionCache* cache,
                          application::core::PdeVelocityExtensionFactorization factorization =
                              application::core::PdeVelocityExtensionFactorization::LuColamd)
{
  const auto n = mesh.n_vertices();
  const auto& X = mesh.X_ref();
  std::vector<double> phi(n), source(2 * n);
  std::vector<std::uint8_t> known(n, 0u);
  for (std::size_t v = 0; v < n; ++v) {
    const double x = X[2 * v], y = X[2 * v + 1];
    phi[v] = phiAt(x, y);
    const auto u = uAt(x, y);
    source[2 * v] = scale * u[0];
    source[2 * v + 1] = scale * u[1] + (scale - 1.0);
    known[v] = phi[v] < 0.0 ? 1u : 0u;
    const auto key = std::make_pair(std::llround(x * kPdeCells),
                                    std::llround(y * kPdeCells));
    if (std::find(extra.begin(), extra.end(), key) != extra.end()) {
      known[v] = 1u;
    }
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
  const std::vector<WallVelocityExtensionConstraint> walls{
      {.boundary_label = kPdeSideWall,
       .constrained_components = {true, false, false}}};
  PdeVelocityExtensionOptions options;
  options.op = op;
  options.factorization = factorization;
  PdeCachedRun out;
  const auto report = application::core::extendVelocityByPde(
      mesh, comm, phi, source, 2u, known, 2u,
      std::span<const WallVelocityExtensionConstraint>(walls), options,
      out.extended, &out.rows, cache,
      application::core::PdeVelocityExtensionMeshRevisions{
          .geometry = 1u, .topology = 1u, .ownership = 1u, .numbering = 1u});
  out.reused = report.reused_factorization;
  out.distributed = report.distributed_solves;
  out.self_checked = report.self_checked;
  return out;
}

// Number of local mismatches between two runs (values and rows, bitwise).
int bitwiseMismatches(const PdeCachedRun& a, const PdeCachedRun& b)
{
  int mismatches = 0;
  if (a.extended.size() != b.extended.size() || a.rows.size() != b.rows.size()) {
    return 1;
  }
  if (!a.extended.empty() &&
      std::memcmp(a.extended.data(), b.extended.data(),
                  a.extended.size() * sizeof(double)) != 0) {
    ++mismatches;
  }
  for (std::size_t r = 0; r < a.rows.size(); ++r) {
    const auto& ra = a.rows[r];
    const auto& rb = b.rows[r];
    if (ra.vertex != rb.vertex || ra.component != rb.component ||
        ra.dependencies.size() != rb.dependencies.size()) {
      ++mismatches;
      continue;
    }
    for (std::size_t d = 0; d < ra.dependencies.size(); ++d) {
      const auto& da = ra.dependencies[d];
      const auto& db = rb.dependencies[d];
      if (da.field != db.field || da.vertex != db.vertex ||
          da.component != db.component ||
          std::memcmp(&da.coefficient, &db.coefficient, sizeof(double)) != 0) {
        ++mismatches;
      }
    }
  }
  return mismatches;
}

// Reuse flags over the ranks: {all reused, any reused}.
std::pair<bool, bool> reuseOverRanks(bool reused)
{
  int local = reused ? 1 : 0;
  int all = 0;
  int any = 0;
  MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return {all != 0, any != 0};
}

int globalSum(int value)
{
  int sum = 0;
  MPI_Allreduce(&value, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return sum;
}

} // namespace

TEST(LevelSetPdeVelocityExtensionMPI, CacheReuseDecisionIsCollective)
{
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";

  const auto arrays = makePdeArrays();
  auto distributed = std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
  distributed->build_from_arrays_global_and_partition(
      2, arrays.x, arrays.offsets, arrays.connectivity, arrays.shapes,
      svmp::PartitionHint::Cells, /*ghost_layers=*/3,
      {{"partition_method", "block"}});
  labelSideWalls(*distributed);
  const svmp::MeshComm comm(MPI_COMM_WORLD);

  // A dry vertex present on exactly one rank: marking it known changes only
  // that rank's contribution.
  const std::pair<long long, long long> lone{kPdeCells / 2, kPdeCells - 1};
  int present = 0;
  {
    const auto& X = distributed->X_ref();
    for (std::size_t v = 0; v < distributed->n_vertices(); ++v) {
      if (std::llround(X[2 * v] * kPdeCells) == lone.first &&
          std::llround(X[2 * v + 1] * kPdeCells) == lone.second) {
        present = 1;
      }
    }
  }
  ASSERT_EQ(globalSum(present), 1);

  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    SCOPED_TRACE(application::core::pdeVelocityExtensionOperatorName(op));
    application::core::PdeVelocityExtensionCache cache;
    const auto check = [&](double scale,
                           const std::vector<std::pair<long long, long long>>& extra,
                           bool expect_reuse, const char* what) {
      SCOPED_TRACE(what);
      const auto cached = extendCached(*distributed, comm, op, scale, extra, &cache);
      const auto reference =
          extendCached(*distributed, comm, op, scale, extra, nullptr);
      const auto [all, any] = reuseOverRanks(cached.reused);
      EXPECT_EQ(all, any) << "ranks disagree on reuse";
      EXPECT_EQ(all, expect_reuse);
      EXPECT_FALSE(reference.reused);
      EXPECT_EQ(globalSum(bitwiseMismatches(cached, reference)), 0);
    };

    check(1.0, {}, false, "first call");
    check(1.5, {}, true, "new velocity");
    // One rank drops its entry: every rank refactors.
    if (rank == 1) {
      cache.clear();
    }
    check(0.5, {}, false, "entry dropped on rank 1");
    check(2.0, {}, true, "reuse after the collective rebuild");
    // The known set changes on one rank only: every rank refactors.
    check(1.0, {lone}, false, "known set changed on one rank");
    check(1.25, {lone}, true, "reuse of the new known set");
    check(1.25, {}, false, "known set restored");
  }
}

namespace {

class ScopedEnvironment {
public:
  ScopedEnvironment(const char* name, const char* value) : name_(name)
  {
    if (const char* previous = std::getenv(name); previous != nullptr) {
      previous_ = std::string(previous);
    }
    ::setenv(name, value, 1);
  }
  ~ScopedEnvironment()
  {
    if (previous_.has_value()) {
      ::setenv(name_, previous_->c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }
  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
  const char* name_;
  std::optional<std::string> previous_;
};

std::pair<bool, bool> flagOverRanks(bool flag) { return reuseOverRanks(flag); }

} // namespace

// Four ranks: rank c factorizes and solves component c and broadcasts the
// solution; the result must equal the replicated uncached solve bit for bit,
// with several cache entries and collective reuse decisions.
TEST(LevelSetPdeVelocityExtensionMPI, FourRankDistributedSolvesMatchTheReplicatedSolve)
{
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size != 4) {
    GTEST_SKIP() << "This test requires exactly four ranks.";
  }

  const auto arrays = makePdeArrays();
  auto distributed = std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
  distributed->build_from_arrays_global_and_partition(
      2, arrays.x, arrays.offsets, arrays.connectivity, arrays.shapes,
      svmp::PartitionHint::Cells, /*ghost_layers=*/3,
      {{"partition_method", "block"}});
  labelSideWalls(*distributed);
  const svmp::MeshComm comm(MPI_COMM_WORLD);

  // A dry vertex owned by exactly one rank's local mesh (see the two-rank
  // test): marking it known changes one rank's contribution only.
  std::vector<std::pair<long long, long long>> lone_candidates;
  for (long long i = 1; i < kPdeCells; ++i) {
    lone_candidates.emplace_back(i, kPdeCells - 1);
  }
  std::pair<long long, long long> lone{-1, -1};
  for (const auto& candidate : lone_candidates) {
    int present = 0;
    const auto& X = distributed->X_ref();
    for (std::size_t v = 0; v < distributed->n_vertices(); ++v) {
      if (std::llround(X[2 * v] * kPdeCells) == candidate.first &&
          std::llround(X[2 * v + 1] * kPdeCells) == candidate.second) {
        present = 1;
      }
    }
    if (globalSum(present) == 1) {
      lone = candidate;
      break;
    }
  }
  const std::pair<long long, long long> other{kPdeCells / 2, kPdeCells - 2};

  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    SCOPED_TRACE(application::core::pdeVelocityExtensionOperatorName(op));
    application::core::PdeVelocityExtensionCache cache(3u);
    const auto check = [&](double scale,
                           const std::vector<std::pair<long long, long long>>& extra,
                           bool expect_reuse, const char* what) {
      SCOPED_TRACE(what);
      const auto cached = extendCached(*distributed, comm, op, scale, extra, &cache);
      const auto reference =
          extendCached(*distributed, comm, op, scale, extra, nullptr);
      const auto [all, any] = reuseOverRanks(cached.reused);
      EXPECT_EQ(all, any) << "ranks disagree on reuse";
      EXPECT_EQ(all, expect_reuse);
      const auto [all_distributed, any_distributed] =
          flagOverRanks(cached.distributed);
      EXPECT_TRUE(all_distributed);
      EXPECT_TRUE(any_distributed);
      EXPECT_FALSE(reference.distributed);
      EXPECT_EQ(globalSum(bitwiseMismatches(cached, reference)), 0);
    };

    check(1.0, {}, false, "first call");
    check(1.5, {}, true, "new velocity");
    check(1.0, {other}, false, "second known set");
    check(0.75, {}, true, "first known set kept");
    check(1.25, {other}, true, "second known set kept");
    if (lone.first >= 0) {
      // Entries, most recently used first: [B A] -> C: [C B A] -> A: [A C B]
      // -> B: [B A C] -> D evicts C: [D B A] -> C again is rebuilt.
      check(1.0, {lone}, false, "known set changed on one rank");
      check(1.0, {}, true, "first known set still kept");
      check(2.0, {other}, true, "second known set still kept");
      check(1.0, {other, lone}, false, "fourth known set evicts the oldest");
      check(1.5, {lone}, false, "evicted known set is rebuilt");
    }
    // One rank drops its entries: the next call refactors on every rank.
    if (rank == 2) {
      cache.clear();
    }
    check(0.5, {}, false, "entries dropped on rank 2");
    check(2.0, {}, true, "reuse after the collective rebuild");

    {
      ScopedEnvironment self_check("SVMP_PDE_EXTENSION_SELF_CHECK", "1");
      const auto checked = extendCached(*distributed, comm, op, 1.75, {}, &cache);
      const auto [all_checked, any_checked] = flagOverRanks(checked.self_checked);
      EXPECT_TRUE(all_checked);
      EXPECT_TRUE(any_checked);
    }
    {
      ScopedEnvironment replicated("SVMP_PDE_EXTENSION_REPLICATED_SOLVES", "1");
      application::core::PdeVelocityExtensionCache replicated_cache(2u);
      const auto miss =
          extendCached(*distributed, comm, op, 1.0, {}, &replicated_cache);
      const auto hit =
          extendCached(*distributed, comm, op, 1.5, {}, &replicated_cache);
      const auto [any_miss_distributed, unused] = flagOverRanks(miss.distributed);
      (void)unused;
      EXPECT_FALSE(any_miss_distributed);
      EXPECT_EQ(reuseOverRanks(hit.reused).first, true);
      EXPECT_EQ(globalSum(bitwiseMismatches(
                    miss, extendCached(*distributed, comm, op, 1.0, {}, nullptr))),
                0);
      EXPECT_EQ(globalSum(bitwiseMismatches(
                    hit, extendCached(*distributed, comm, op, 1.5, {}, nullptr))),
                0);
    }
    // A failure on the rank that solves component 1 is raised on every rank,
    // and every rank drops its entries.
    {
      ScopedEnvironment failure("SVMP_PDE_EXTENSION_FAIL_COMPONENT", "1");
      int threw = 0;
      std::string message;
      try {
        (void)extendCached(*distributed, comm, op, 3.0, {other}, &cache);
      } catch (const std::runtime_error& error) {
        threw = 1;
        message = error.what();
      }
      EXPECT_EQ(globalSum(threw), size);
      EXPECT_NE(message.find("injected failure of component 1"), std::string::npos)
          << message;
      EXPECT_TRUE(cache.empty());
    }
    check(1.0, {}, false, "after the failure");
  }
}

// Opt-in MUMPS factorization distributed over all ranks (any rank count):
// the result matches the serial LU solve to round-off, a reused
// factorization is applied collectively, and components are never solved on
// separate ranks.
TEST(LevelSetPdeVelocityExtensionMPI, MumpsFactorizationMatchesSerialLuOnAnyRankCount)
{
  if (!svmp::FE::backends::mumpsAvailable()) {
    GTEST_SKIP() << "Built without FE_ENABLE_MUMPS.";
  }
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  using application::core::PdeVelocityExtensionFactorization;

  const auto arrays = makePdeArrays();
  auto distributed = std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
  distributed->build_from_arrays_global_and_partition(
      2, arrays.x, arrays.offsets, arrays.connectivity, arrays.shapes,
      svmp::PartitionHint::Cells, /*ghost_layers=*/3,
      {{"partition_method", "block"}});
  labelSideWalls(*distributed);
  auto base = std::make_shared<svmp::MeshBase>();
  base->build_from_arrays(2, arrays.x, arrays.offsets, arrays.connectivity,
                          arrays.shapes);
  base->finalize();
  auto serial = svmp::create_mesh(std::move(base));
  labelSideWalls(*serial);
  const svmp::MeshComm comm(MPI_COMM_WORLD);

  for (const auto op : {PdeVelocityExtensionOperator::Harmonic,
                        PdeVelocityExtensionOperator::LeastSquaresNormal}) {
    const auto reference = extend(*serial, svmp::MeshComm::self(), op);
    const auto mumps =
        extend(*distributed, comm, op, PdeVelocityExtensionFactorization::Mumps);
    int local_failures = 0;
    for (const auto& [key, value] : mumps) {
      const auto found = reference.find(key);
      ASSERT_NE(found, reference.end());
      for (int c = 0; c < 2; ++c) {
        const double scale = std::max(1.0, std::abs(found->second[c]));
        if (std::abs(value[c] - found->second[c]) > 1e-11 * scale) {
          ++local_failures;
        }
      }
    }
    EXPECT_EQ(globalSum(local_failures), 0)
        << application::core::pdeVelocityExtensionOperatorName(op) << " ranks=" << size;
  }

  // Cached: the second call with new velocities reuses the factorization on
  // every rank and matches an uncached MUMPS solve.
  application::core::PdeVelocityExtensionCache cache;
  const auto first = extendCached(*distributed, comm, PdeVelocityExtensionOperator::Harmonic,
                                  1.0, {}, &cache, PdeVelocityExtensionFactorization::Mumps);
  const auto second = extendCached(*distributed, comm, PdeVelocityExtensionOperator::Harmonic,
                                   1.5, {}, &cache, PdeVelocityExtensionFactorization::Mumps);
  const auto fresh = extendCached(*distributed, comm, PdeVelocityExtensionOperator::Harmonic,
                                  1.5, {}, nullptr, PdeVelocityExtensionFactorization::Mumps);
  EXPECT_FALSE(reuseOverRanks(first.reused).second);
  EXPECT_TRUE(reuseOverRanks(second.reused).first);
  EXPECT_FALSE(reuseOverRanks(second.distributed).second);
  int local_far = 0;
  ASSERT_EQ(second.extended.size(), fresh.extended.size());
  for (std::size_t i = 0; i < second.extended.size(); ++i) {
    const double scale = std::max(1.0, std::abs(fresh.extended[i]));
    if (std::abs(second.extended[i] - fresh.extended[i]) > 1e-12 * scale) {
      ++local_far;
    }
  }
  EXPECT_EQ(globalSum(local_far), 0);
  const int bitwise = globalSum(bitwiseMismatches(second, fresh));
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::printf("MUMPS reuse vs fresh factorization: %s on %d ranks\n",
                bitwise == 0 ? "bitwise identical" : "round-off differences", size);
  }
}
