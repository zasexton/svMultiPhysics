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
 * @file test_DistributedSemantics.cpp
 * @brief MPI test validating owned/shared/ghost semantics and global counts.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"
#include <mpi.h>
#include <cassert>
#include <cmath>
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

#define ASSERT_EQ(a, b) ASSERT((a) == (b))

static gid_t strip_vertex_gid(int x_plane, int y, int z) {
  return static_cast<gid_t>(x_plane * 4 + (y + 2 * z));
}

static std::shared_ptr<MeshBase> create_hex_strip_partition(int rank) {
  // Global mesh: a strip of unit cubes along +x with consistent vertex GIDs.
  // Rank r owns cube [r,r+1]x[0,1]x[0,1].
  auto mesh = std::make_shared<MeshBase>();

  const real_t x0 = static_cast<real_t>(rank);
  const real_t x1 = static_cast<real_t>(rank + 1);

  std::vector<gid_t> vertex_gids = {
      strip_vertex_gid(rank, 0, 0),
      strip_vertex_gid(rank + 1, 0, 0),
      strip_vertex_gid(rank + 1, 1, 0),
      strip_vertex_gid(rank, 1, 0),
      strip_vertex_gid(rank, 0, 1),
      strip_vertex_gid(rank + 1, 0, 1),
      strip_vertex_gid(rank + 1, 1, 1),
      strip_vertex_gid(rank, 1, 1),
  };

  std::vector<real_t> coords = {
      x0, 0.0, 0.0,
      x1, 0.0, 0.0,
      x1, 1.0, 0.0,
      x0, 1.0, 0.0,
      x0, 0.0, 1.0,
      x1, 0.0, 1.0,
      x1, 1.0, 1.0,
      x0, 1.0, 1.0,
  };

  std::vector<offset_t> offsets = {0, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes = {{CellFamily::Hex, 8, 1}};
  std::vector<gid_t> cell_gids = {static_cast<gid_t>(rank)};

  mesh->build_from_arrays(3, coords, offsets, conn, shapes);
  mesh->set_vertex_gids(std::move(vertex_gids));
  mesh->set_cell_gids(std::move(cell_gids));
  mesh->finalize();

  return mesh;
}

static void assert_partition_invariants(const DistributedMesh& dmesh) {
  ASSERT_EQ(dmesh.n_owned_vertices() + dmesh.n_shared_vertices() + dmesh.n_ghost_vertices(),
            dmesh.n_vertices());
  ASSERT_EQ(dmesh.n_owned_cells() + dmesh.n_shared_cells() + dmesh.n_ghost_cells(),
            dmesh.n_cells());
  ASSERT_EQ(dmesh.n_owned_faces() + dmesh.n_shared_faces() + dmesh.n_ghost_faces(),
            dmesh.n_faces());

  ASSERT_EQ(dmesh.owned_vertices().size(), dmesh.n_owned_vertices());
  ASSERT_EQ(dmesh.shared_vertices().size(), dmesh.n_shared_vertices());
  ASSERT_EQ(dmesh.ghost_vertices().size(), dmesh.n_ghost_vertices());

  ASSERT_EQ(dmesh.owned_cells().size(), dmesh.n_owned_cells());
  ASSERT_EQ(dmesh.shared_cells().size(), dmesh.n_shared_cells());
  ASSERT_EQ(dmesh.ghost_cells().size(), dmesh.n_ghost_cells());

  ASSERT_EQ(dmesh.owned_faces().size(), dmesh.n_owned_faces());
  ASSERT_EQ(dmesh.shared_faces().size(), dmesh.n_shared_faces());
  ASSERT_EQ(dmesh.ghost_faces().size(), dmesh.n_ghost_faces());
}

static void test_base_partition(int rank, int world_size) {
  auto local_mesh = create_hex_strip_partition(rank);
  DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  ASSERT_EQ(dmesh.n_cells(), 1u);
  ASSERT_EQ(dmesh.n_vertices(), 8u);
  ASSERT_EQ(dmesh.n_faces(), 6u);

  // Establish owned/shared ownership via shared-entity gathering.
  dmesh.build_exchange_patterns();

  ASSERT_EQ(dmesh.n_owned_cells(), 1u);
  ASSERT_EQ(dmesh.n_shared_cells(), 0u);
  ASSERT_EQ(dmesh.n_ghost_cells(), 0u);

  const size_t expected_global_cells = static_cast<size_t>(world_size);
  const size_t expected_global_vertices = static_cast<size_t>(4 * (world_size + 1));
  const size_t expected_global_faces = static_cast<size_t>(5 * world_size + 1);

  ASSERT_EQ(dmesh.global_n_cells(), expected_global_cells);
  ASSERT_EQ(dmesh.global_n_vertices(), expected_global_vertices);
  ASSERT_EQ(dmesh.global_n_faces(), expected_global_faces);

  // Rank-local expected owned/shared counts for the strip partition (no ghosts).
  const size_t expected_shared_vertices = (rank == 0) ? 0u : 4u;
  const size_t expected_owned_vertices = 8u - expected_shared_vertices;
  ASSERT_EQ(dmesh.n_shared_vertices(), expected_shared_vertices);
  ASSERT_EQ(dmesh.n_owned_vertices(), expected_owned_vertices);
  ASSERT_EQ(dmesh.n_ghost_vertices(), 0u);

  const size_t expected_shared_faces = (rank == 0) ? 0u : 1u;
  const size_t expected_owned_faces = 6u - expected_shared_faces;
  ASSERT_EQ(dmesh.n_shared_faces(), expected_shared_faces);
  ASSERT_EQ(dmesh.n_owned_faces(), expected_owned_faces);
  ASSERT_EQ(dmesh.n_ghost_faces(), 0u);

  assert_partition_invariants(dmesh);
}

static void test_ghost_layer_semantics(int rank, int world_size) {
  auto local_mesh = create_hex_strip_partition(rank);
  DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  dmesh.build_exchange_patterns();
  const size_t expected_global_cells = static_cast<size_t>(world_size);
  const size_t expected_global_vertices = static_cast<size_t>(4 * (world_size + 1));
  const size_t expected_global_faces = static_cast<size_t>(5 * world_size + 1);

  dmesh.build_ghost_layer(1);

  const size_t expected_local_cells =
      1u + (rank > 0 ? 1u : 0u) + (rank + 1 < world_size ? 1u : 0u);
  const size_t expected_planes =
      2u + (rank > 0 ? 1u : 0u) + (rank + 1 < world_size ? 1u : 0u);
  const size_t expected_local_vertices = expected_planes * 4u;

  ASSERT_EQ(dmesh.n_cells(), expected_local_cells);
  ASSERT_EQ(dmesh.n_vertices(), expected_local_vertices);

  // Ghost layer should add ghost cells/vertices on ranks with neighbors.
  const size_t expected_ghost_cells =
      (rank > 0 ? 1u : 0u) + (rank + 1 < world_size ? 1u : 0u);
  ASSERT_EQ(dmesh.n_ghost_cells(), expected_ghost_cells);

  const size_t expected_ghost_vertices = expected_ghost_cells * 4u;
  ASSERT_EQ(dmesh.n_ghost_vertices(), expected_ghost_vertices);

  // Global counts remain invariant when ghosts are present.
  ASSERT_EQ(dmesh.global_n_cells(), expected_global_cells);
  ASSERT_EQ(dmesh.global_n_vertices(), expected_global_vertices);
  ASSERT_EQ(dmesh.global_n_faces(), expected_global_faces);

  assert_partition_invariants(dmesh);

  // Clearing ghosts returns to base partition sizes.
  dmesh.clear_ghosts();
  ASSERT_EQ(dmesh.n_cells(), 1u);
  ASSERT_EQ(dmesh.n_vertices(), 8u);
  ASSERT_EQ(dmesh.n_faces(), 6u);
  ASSERT_EQ(dmesh.n_ghost_cells(), 0u);
  ASSERT_EQ(dmesh.n_ghost_vertices(), 0u);
  ASSERT_EQ(dmesh.n_ghost_faces(), 0u);

  ASSERT_EQ(dmesh.global_n_cells(), expected_global_cells);
  ASSERT_EQ(dmesh.global_n_vertices(), expected_global_vertices);
  ASSERT_EQ(dmesh.global_n_faces(), expected_global_faces);
  assert_partition_invariants(dmesh);
}

static void test_migration_preserves_global_counts(int rank, int world_size) {
  auto local_mesh = create_hex_strip_partition(rank);
  DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  dmesh.build_exchange_patterns();

  const size_t expected_global_cells = static_cast<size_t>(world_size);
  const size_t expected_global_vertices = static_cast<size_t>(4 * (world_size + 1));
  const size_t expected_global_faces = static_cast<size_t>(5 * world_size + 1);

  std::vector<rank_t> new_owner(1, static_cast<rank_t>((rank + 1) % world_size));
  dmesh.migrate(new_owner);

  dmesh.build_exchange_patterns();

  ASSERT_EQ(dmesh.global_n_cells(), expected_global_cells);
  ASSERT_EQ(dmesh.global_n_vertices(), expected_global_vertices);
  ASSERT_EQ(dmesh.global_n_faces(), expected_global_faces);

  assert_partition_invariants(dmesh);
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (world < 2) {
    if (rank == 0) {
      std::cout << "Skipping distributed semantics test (requires >= 2 ranks)\n";
    }
    MPI_Finalize();
    return 0;
  }

  svmp::test::test_base_partition(rank, world);
  svmp::test::test_migration_preserves_global_counts(rank, world);
  svmp::test::test_ghost_layer_semantics(rank, world);

  if (rank == 0) {
    std::cout << "DistributedMesh semantics tests PASSED\n";
  }

  MPI_Finalize();
  return 0;
}
