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
 * @file test_MeshValidationMPI.cpp
 * @brief MPI tests for MeshValidation distributed checks.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"
#include "../../../Topology/CellShape.h"
#include "../../../Validation/MeshValidation.h"

#include <mpi.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                               \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

static void build_two_tets_global(DistributedMesh& mesh, int rank) {
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
      3, X_ref, cell2v_offsets, cell2v, shapes, PartitionHint::Cells, /*ghost_layers=*/1);
}

static std::shared_ptr<MeshBase> build_single_quad_mesh_with_custom_gids(int rank) {
  auto mesh = std::make_shared<MeshBase>();

  const int dim = 3;
  std::vector<real_t> X_ref = {
      0.0 + rank, 0.0, 0.0,  // v0
      1.0 + rank, 0.0, 0.0,  // v1
      1.0 + rank, 1.0, 0.0,  // v2
      0.0 + rank, 1.0, 0.0   // v3
  };

  std::vector<offset_t> offs = {0, 4};
  std::vector<index_t> conn = {0, 1, 2, 3};
  std::vector<CellShape> shapes = {{CellFamily::Quad, 4, 1}};

  mesh->build_from_arrays(dim, X_ref, offs, conn, shapes);
  mesh->finalize();

  std::vector<gid_t> vertex_gids = {
      static_cast<gid_t>(1000 + rank * 10 + 0),
      static_cast<gid_t>(1000 + rank * 10 + 1),
      static_cast<gid_t>(1000 + rank * 10 + 2),
      static_cast<gid_t>(1000 + rank * 10 + 3),
  };
  mesh->set_vertex_gids(std::move(vertex_gids));

  std::vector<gid_t> cell_gids = {static_cast<gid_t>(12345)};  // duplicated across ranks
  mesh->set_cell_gids(std::move(cell_gids));

  return mesh;
}

static void test_checks_pass_on_partitioned_mesh(const DistributedMesh& mesh) {
  const auto gids = MeshValidation::check_global_ids(mesh);
  ASSERT(gids.passed);

  const auto ghosts = MeshValidation::check_ghost_cells(mesh);
  ASSERT(ghosts.passed);

  const auto par = MeshValidation::check_parallel_consistency(mesh);
  ASSERT(par.passed);
}

static void test_detect_duplicate_owned_cell_gids(MPI_Comm comm, int rank) {
  auto local_mesh = build_single_quad_mesh_with_custom_gids(rank);
  DistributedMesh mesh(local_mesh, comm);

  const auto gids = MeshValidation::check_global_ids(mesh);
  ASSERT(!gids.passed);
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm split_comm = MPI_COMM_NULL;
  if (world_size > 2) {
    const int color = (world_rank < 2) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &split_comm);
    if (color == MPI_UNDEFINED) {
      MPI_Finalize();
      return 0;
    }
    comm = split_comm;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  svmp::DistributedMesh dmesh(comm);
  svmp::test::build_two_tets_global(dmesh, rank);
  svmp::test::test_checks_pass_on_partitioned_mesh(dmesh);

  if (size >= 2) {
    svmp::test::test_detect_duplicate_owned_cell_gids(comm, rank);
  }

  if (split_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&split_comm);
  }

  MPI_Finalize();
  return 0;
}

