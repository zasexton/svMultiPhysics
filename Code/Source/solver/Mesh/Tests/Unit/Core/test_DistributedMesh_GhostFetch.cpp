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
 * @file test_DistributedMesh_GhostFetch.cpp
 * @brief MPI test for DistributedMesh ghost-layer fetching (halo growth).
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"
#include <mpi.h>
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

#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) < (tol))

static std::shared_ptr<MeshBase> create_two_hex_partition(int rank) {
  // Global layout: two unit cubes [0,1]^3 and [1,2]x[0,1]^2.
  // Vertex GIDs are consistent across ranks, shared on the interface x=1.
  auto mesh = std::make_shared<MeshBase>();

  std::vector<gid_t> vertex_gids;
  std::vector<real_t> coords;
  vertex_gids.reserve(8);
  coords.reserve(24);

  auto add_vertex = [&](gid_t gid, real_t x, real_t y, real_t z) {
    vertex_gids.push_back(gid);
    coords.push_back(x);
    coords.push_back(y);
    coords.push_back(z);
  };

  if (rank == 0) {
    // Cube 0 vertices (GIDs 0..7)
    add_vertex(0, 0, 0, 0);
    add_vertex(1, 1, 0, 0);
    add_vertex(2, 1, 1, 0);
    add_vertex(3, 0, 1, 0);
    add_vertex(4, 0, 0, 1);
    add_vertex(5, 1, 0, 1);
    add_vertex(6, 1, 1, 1);
    add_vertex(7, 0, 1, 1);
  } else {
    // Cube 1 vertices (shared: 1,2,5,6; new: 8,9,10,11)
    add_vertex(1, 1, 0, 0);
    add_vertex(8, 2, 0, 0);
    add_vertex(9, 2, 1, 0);
    add_vertex(2, 1, 1, 0);
    add_vertex(5, 1, 0, 1);
    add_vertex(10, 2, 0, 1);
    add_vertex(11, 2, 1, 1);
    add_vertex(6, 1, 1, 1);
  }

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

static void test_halo_fetch_two_ranks(int rank, int world_size) {
  if (world_size != 2) {
    if (rank == 0) {
      std::cout << "Skipping ghost fetch test (requires exactly 2 ranks)\n";
    }
    return;
  }

  auto local_mesh = create_two_hex_partition(rank);
  DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  ASSERT(dmesh.local_mesh().n_cells() == 1);
  ASSERT(dmesh.local_mesh().n_vertices() == 8);

  dmesh.build_ghost_layer(1);

  ASSERT(dmesh.local_mesh().n_cells() == 2);
  ASSERT(dmesh.local_mesh().n_vertices() == 12);

  // Exactly one ghost cell, owned by the other rank.
  int ghost_count = 0;
  for (index_t c = 0; c < static_cast<index_t>(dmesh.local_mesh().n_cells()); ++c) {
    if (dmesh.is_ghost_cell(c)) {
      ghost_count++;
      ASSERT(dmesh.owner_rank_cell(c) == (rank == 0 ? 1 : 0));
    } else {
      ASSERT(dmesh.is_owned_cell(c));
      ASSERT(dmesh.owner_rank_cell(c) == rank);
    }
  }
  ASSERT(ghost_count == 1);

  // Spot-check a vertex that must have been fetched.
  const gid_t probe_gid = (rank == 0) ? 11 : 7;
  const index_t probe_v = dmesh.local_mesh().global_to_local_vertex(probe_gid);
  ASSERT(probe_v != INVALID_INDEX);
  const auto xyz = dmesh.local_mesh().get_vertex_coords(probe_v);

  if (rank == 0) {
    ASSERT_NEAR(xyz[0], 2.0, 1e-12);
    ASSERT_NEAR(xyz[1], 1.0, 1e-12);
    ASSERT_NEAR(xyz[2], 1.0, 1e-12);
  } else {
    ASSERT_NEAR(xyz[0], 0.0, 1e-12);
    ASSERT_NEAR(xyz[1], 1.0, 1e-12);
    ASSERT_NEAR(xyz[2], 1.0, 1e-12);
  }

  // Clearing ghosts removes imported cells/vertices.
  dmesh.clear_ghosts();
  ASSERT(dmesh.local_mesh().n_cells() == 1);
  ASSERT(dmesh.local_mesh().n_vertices() == 8);
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  svmp::test::test_halo_fetch_two_ranks(rank, world);

  if (rank == 0) {
    std::cout << "DistributedMesh ghost fetch tests PASSED\n";
  }

  MPI_Finalize();
  return 0;
}

