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

/**
 * @file test_MeshSearchMPI.cpp
 * @brief MPI tests for MeshSearch global query APIs.
 */

#include "../../../Search/MeshSearch.h"
#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                               \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) <= (tol))

static void build_quad_grid_global(DistributedMesh& mesh,
                                  int rank,
                                  int nx,
                                  int ny,
                                  int ghost_layers) {
  std::vector<real_t> X_ref;
  std::vector<offset_t> cell2v_offsets;
  std::vector<index_t> cell2v;
  std::vector<CellShape> shapes;

  if (rank == 0) {
    X_ref.reserve(static_cast<size_t>((nx + 1) * (ny + 1) * 3));
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        X_ref.push_back(static_cast<real_t>(i));
        X_ref.push_back(static_cast<real_t>(j));
        X_ref.push_back(0.0);
      }
    }

    cell2v_offsets.reserve(static_cast<size_t>(nx * ny + 1));
    cell2v_offsets.push_back(0);
    cell2v.reserve(static_cast<size_t>(nx * ny * 4));
    shapes.reserve(static_cast<size_t>(nx * ny));

    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const int base = j * (nx + 1) + i;
        cell2v.push_back(static_cast<index_t>(base));
        cell2v.push_back(static_cast<index_t>(base + 1));
        cell2v.push_back(static_cast<index_t>(base + (nx + 1) + 1));
        cell2v.push_back(static_cast<index_t>(base + (nx + 1)));
        cell2v_offsets.push_back(static_cast<offset_t>(cell2v.size()));
        shapes.push_back({CellFamily::Quad, 4, 1});
      }
    }
  }

  mesh.build_from_arrays_global_and_partition(
      3, X_ref, cell2v_offsets, cell2v, shapes, PartitionHint::Cells, ghost_layers);
}

static void test_signed_distance_and_closest_point_global(const DistributedMesh& mesh) {
  const int rank = static_cast<int>(mesh.rank());
  const auto p = std::array<real_t,3>{{0.25, 0.7, 0.0}};

  const real_t sd = MeshSearch::signed_distance_global(mesh, p);
  ASSERT_NEAR(sd, -0.25, 1e-12);

  const auto [cp, id] = MeshSearch::closest_boundary_point_global(mesh, p);
  ASSERT_NEAR(cp[0], 0.0, 1e-12);
  ASSERT_NEAR(cp[1], 0.7, 1e-12);
  ASSERT_NEAR(cp[2], 0.0, 1e-12);

  const int has_id = (id != INVALID_INDEX) ? 1 : 0;
  int sum_has_id = 0;
  MPI_Allreduce(&has_id, &sum_has_id, 1, MPI_INT, MPI_SUM, mesh.mpi_comm());
  ASSERT(sum_has_id == 1);

  // Winner should be exactly one rank; everyone should get the same closest point.
  (void)rank;
}

static void test_ray_intersection_global(const DistributedMesh& mesh) {
  const int rank = static_cast<int>(mesh.rank());
  const auto origin = std::array<real_t,3>{{0.7, 0.8, -1.0}};
  const auto dir = std::array<real_t,3>{{0.0, 0.0, 1.0}};

  const auto hit = MeshSearch::intersect_ray_global(mesh, origin, dir);
  ASSERT(hit.found);
  ASSERT_NEAR(hit.t, 1.0, 1e-12);
  ASSERT_NEAR(hit.point[0], 0.7, 1e-12);
  ASSERT_NEAR(hit.point[1], 0.8, 1e-12);
  ASSERT_NEAR(hit.point[2], 0.0, 1e-12);

  const int has_id = (hit.face_id != INVALID_INDEX) ? 1 : 0;
  int sum_has_id = 0;
  MPI_Allreduce(&has_id, &sum_has_id, 1, MPI_INT, MPI_SUM, mesh.mpi_comm());
  ASSERT(sum_has_id == 1);

  // Global list is returned only on rank 0.
  const auto all_hits = MeshSearch::intersect_ray_all_global(mesh, origin, dir);
  if (rank == 0) {
    ASSERT(all_hits.size() == 1);
    ASSERT_NEAR(all_hits[0].t, 1.0, 1e-12);
  } else {
    ASSERT(all_hits.empty());
  }
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  svmp::DistributedMesh mesh(MPI_COMM_WORLD);
  svmp::test::build_quad_grid_global(mesh, rank, /*nx=*/2, /*ny=*/2, /*ghost_layers=*/1);

  svmp::test::test_signed_distance_and_closest_point_global(mesh);
  svmp::test::test_ray_intersection_global(mesh);

  MPI_Finalize();
  return 0;
}

