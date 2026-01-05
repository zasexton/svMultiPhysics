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
 * @file test_BoundaryDetectorMPI.cpp
 * @brief MPI tests for global boundary detection without ghost layers.
 */

#include "../../../Boundary/BoundaryDetector.h"
#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>
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

static void build_two_tets_global_no_ghosts(DistributedMesh& mesh, int rank) {
  std::vector<real_t> X_ref;
  std::vector<offset_t> cell2v_offsets;
  std::vector<index_t> cell2v;
  std::vector<CellShape> shapes;

  if (rank == 0) {
    X_ref = {
        0.0, 0.0, 0.0,  // v0
        1.0, 0.0, 0.0,  // v1
        0.0, 1.0, 0.0,  // v2
        0.0, 0.0, 1.0,  // v3
        0.0, 0.0, -1.0  // v4
    };

    cell2v_offsets = {0, 4, 8};
    cell2v = {
        0, 1, 2, 3,  // tet 0
        0, 1, 2, 4   // tet 1
    };

    shapes = {{CellFamily::Tetra, 4, 1}, {CellFamily::Tetra, 4, 1}};
  }

  mesh.build_from_arrays_global_and_partition(
      3, X_ref, cell2v_offsets, cell2v, shapes, PartitionHint::Cells, /*ghost_layers=*/0);
}

static void test_global_boundary_detector(const DistributedMesh& mesh) {
  const auto& local = mesh.local_mesh();
  BoundaryDetector bd(local);
  const auto local_info = bd.detect_boundary();
  const auto global_info = BoundaryDetector::detect_boundary_global(mesh);

  const size_t local_b = local_info.boundary_entities.size();
  const size_t global_b = global_info.boundary_entities.size();

  size_t sum_local = 0;
  size_t sum_global = 0;
  MPI_Allreduce(&local_b, &sum_local, 1, MPI_UNSIGNED_LONG, MPI_SUM, mesh.mpi_comm());
  MPI_Allreduce(&global_b, &sum_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, mesh.mpi_comm());

  if (mesh.world_size() >= 2) {
    // Without ghosts, each rank sees its partition-interface face as a boundary (4 faces per tet).
    // Global boundary detection excludes the shared interior face (3 boundary faces per tet).
    ASSERT(sum_local == 8u);
    ASSERT(sum_global == 6u);

    if (!mesh.owned_cells().empty()) {
      ASSERT(global_b == 3u);
      ASSERT(local_b == 4u);
    } else {
      ASSERT(global_b == 0u);
      ASSERT(local_b == 0u);
    }
  } else {
    ASSERT(sum_local == 6u);
    ASSERT(sum_global == 6u);
  }
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  svmp::DistributedMesh mesh(MPI_COMM_WORLD);
  svmp::test::build_two_tets_global_no_ghosts(mesh, rank);
  svmp::test::test_global_boundary_detector(mesh);

  MPI_Finalize();
  return 0;
}

