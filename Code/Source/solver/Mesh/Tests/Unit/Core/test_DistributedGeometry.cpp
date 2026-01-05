/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_DistributedGeometry.cpp
 * @brief MPI regression tests for MeshGeometry global reductions.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Geometry/MeshGeometry.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace svmp {
namespace test {

#define ASSERT(cond) \
  do { \
    if (!(cond)) { \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ \
                << " in " << __func__ << ": " #cond << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
  } while (0)

#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) < (tol))

static void build_line_of_unit_hexes(DistributedMesh& dmesh, int nx, int ghost_layers) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int ny = 1;
  const int nz = 1;

  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> connectivity;
  std::vector<CellShape> shapes;

  if (rank == 0) {
    const int n_verts = (nx + 1) * (ny + 1) * (nz + 1);
    coords.reserve(3 * n_verts);

    for (int k = 0; k <= nz; ++k) {
      for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
          coords.push_back(static_cast<real_t>(i));
          coords.push_back(static_cast<real_t>(j));
          coords.push_back(static_cast<real_t>(k));
        }
      }
    }

    offsets.push_back(0);
    shapes.reserve(static_cast<size_t>(nx) * ny * nz);

    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const int base = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
          connectivity.push_back(base);
          connectivity.push_back(base + 1);
          connectivity.push_back(base + nx + 2);
          connectivity.push_back(base + nx + 1);
          connectivity.push_back(base + (ny + 1) * (nx + 1));
          connectivity.push_back(base + (ny + 1) * (nx + 1) + 1);
          connectivity.push_back(base + (ny + 1) * (nx + 1) + nx + 2);
          connectivity.push_back(base + (ny + 1) * (nx + 1) + nx + 1);
          offsets.push_back(static_cast<offset_t>(connectivity.size()));

          shapes.push_back({CellFamily::Hex, 8, 1});
        }
      }
    }
  }

  dmesh.build_from_arrays_global_and_partition(
      3, coords, offsets, connectivity, shapes, PartitionHint::Cells, ghost_layers);
}

static void build_line_of_hexes_with_variable_dx(DistributedMesh& dmesh,
                                                 const std::vector<real_t>& cell_dx,
                                                 int ghost_layers) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int ny = 1;
  const int nz = 1;
  const int nx = static_cast<int>(cell_dx.size());

  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> connectivity;
  std::vector<CellShape> shapes;

  if (rank == 0) {
    // Build monotone x coordinates from dx per cell.
    std::vector<real_t> x(static_cast<size_t>(nx) + 1u, 0.0);
    for (int i = 0; i < nx; ++i) {
      x[static_cast<size_t>(i) + 1u] = x[static_cast<size_t>(i)] + cell_dx[static_cast<size_t>(i)];
    }

    const int n_verts = (nx + 1) * (ny + 1) * (nz + 1);
    coords.reserve(3 * n_verts);

    for (int k = 0; k <= nz; ++k) {
      for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
          coords.push_back(x[static_cast<size_t>(i)]);
          coords.push_back(static_cast<real_t>(j));
          coords.push_back(static_cast<real_t>(k));
        }
      }
    }

    offsets.push_back(0);
    shapes.reserve(static_cast<size_t>(nx) * ny * nz);

    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const int base = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
          connectivity.push_back(base);
          connectivity.push_back(base + 1);
          connectivity.push_back(base + nx + 2);
          connectivity.push_back(base + nx + 1);
          connectivity.push_back(base + (ny + 1) * (nx + 1));
          connectivity.push_back(base + (ny + 1) * (nx + 1) + 1);
          connectivity.push_back(base + (ny + 1) * (nx + 1) + nx + 2);
          connectivity.push_back(base + (ny + 1) * (nx + 1) + nx + 1);
          offsets.push_back(static_cast<offset_t>(connectivity.size()));

          shapes.push_back({CellFamily::Hex, 8, 1});
        }
      }
    }
  }

  dmesh.build_from_arrays_global_and_partition(
      3, coords, offsets, connectivity, shapes, PartitionHint::Cells, ghost_layers);
}

static void test_global_volume_and_boundary_area() {
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  DistributedMesh dmesh(MPI_COMM_WORLD);
  build_line_of_unit_hexes(dmesh, /*nx=*/4, /*ghost_layers=*/1);

  // Local-only computations include ghosts and are not globally meaningful.
  const real_t local_volume = MeshGeometry::total_volume(dmesh.local_mesh());
  const real_t local_area = MeshGeometry::boundary_area(dmesh.local_mesh());

  const real_t global_volume = MeshGeometry::total_volume_global(dmesh);
  const real_t global_area = MeshGeometry::boundary_area_global(dmesh);

  ASSERT_NEAR(global_volume, 4.0, 1e-12);
  ASSERT_NEAR(global_area, 18.0, 1e-12);

  // With >1 rank and a ghost layer, at least one rank should have local != global.
  int local_mismatch = ((std::abs(local_volume - global_volume) > 1e-12) ||
                        (std::abs(local_area - global_area) > 1e-12))
                           ? 1
                           : 0;
  int any_mismatch = 0;
  MPI_Allreduce(&local_mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (world > 1) {
    ASSERT(any_mismatch == 1);
  }

  if (rank == 0) {
    std::cout << "  ✓ Global volume/area reductions passed\n";
  }
}

static void test_global_quality_range() {
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  DistributedMesh dmesh(MPI_COMM_WORLD);

  // Make the first cell highly stretched to ensure the max aspect ratio lives on rank 0
  // and is not present on all ranks' local partitions.
  const std::vector<real_t> cell_dx = {
      9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
  };
  build_line_of_hexes_with_variable_dx(dmesh, cell_dx, /*ghost_layers=*/0);

  // Compute expected global min/max over owned cells with an explicit MPI reduction.
  real_t local_min = 1e300;
  real_t local_max = -1e300;
  bool any_owned = false;

  const MeshBase& local = dmesh.local_mesh();
  const index_t n_cells = static_cast<index_t>(local.n_cells());
  for (index_t c = 0; c < n_cells; ++c) {
    if (!dmesh.is_owned_cell(c)) {
      continue;
    }
    any_owned = true;
    const real_t q = local.compute_quality(c, "aspect_ratio");
    local_min = std::min(local_min, q);
    local_max = std::max(local_max, q);
  }
  if (!any_owned) {
    local_min = 1e300;
    local_max = -1e300;
  }

  real_t expected_min = 0.0;
  real_t expected_max = 0.0;
  MPI_Allreduce(&local_min, &expected_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &expected_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  const auto got = dmesh.global_quality_range("aspect_ratio");
  ASSERT_NEAR(got.first, expected_min, 1e-12);
  ASSERT_NEAR(got.second, expected_max, 1e-12);

  // In parallel runs, at least one rank should have a local-only range that differs from global.
  const auto local_range = local.global_quality_range("aspect_ratio");
  const int local_mismatch =
      ((std::abs(local_range.first - expected_min) > 1e-12) ||
       (std::abs(local_range.second - expected_max) > 1e-12))
          ? 1
          : 0;
  int any_mismatch = 0;
  MPI_Allreduce(&local_mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (world > 1) {
    ASSERT(any_mismatch == 1);
  }

  if (rank == 0) {
    std::cout << "  ✓ Global quality range reduction passed\n";
  }
}

} // namespace test
} // namespace svmp

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  try {
    svmp::test::test_global_volume_and_boundary_area();
    svmp::test::test_global_quality_range();
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
