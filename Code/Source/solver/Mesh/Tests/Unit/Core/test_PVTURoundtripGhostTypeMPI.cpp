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
 * @file test_PVTURoundtripGhostTypeMPI.cpp
 * @brief MPI test validating PVTU save/load roundtrip preserves ghost ownership.
 *
 * The VTK parallel format typically stores duplicated entities and uses the
 * conventional "vtkGhostType" array to mark ghosts/duplicates. DistributedMesh
 * should interpret this metadata on load so ghost cells remain ghost cells.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"

#include <mpi.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

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

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

#if !defined(MESH_HAS_VTK)
  if (rank == 0) {
    std::cout << "Skipping PVTU roundtrip ghost test (VTK disabled)\n";
  }
  MPI_Finalize();
  return 0;
#else
  if (world < 2) {
    if (rank == 0) {
      std::cout << "Skipping PVTU roundtrip ghost test (requires >= 2 ranks)\n";
    }
    MPI_Finalize();
    return 0;
  }

  auto local_mesh = svmp::test::create_hex_strip_partition(rank);
  svmp::DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);
  dmesh.build_exchange_patterns();
  dmesh.build_ghost_layer(1);

  // Create a unique output directory shared by all ranks.
  long long pid = 0;
  long long stamp = 0;
  if (rank == 0) {
#if defined(_WIN32)
    pid = static_cast<long long>(_getpid());
#else
    pid = static_cast<long long>(getpid());
#endif
    stamp = static_cast<long long>(MPI_Wtime() * 1e9);
  }
  MPI_Bcast(&pid, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&stamp, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  const std::string out_dir =
      std::string("pvtu_roundtrip_") + std::to_string(pid) + "_" + std::to_string(stamp);
  if (rank == 0) {
    std::error_code ec;
    std::filesystem::remove_all(out_dir, ec);
    std::filesystem::create_directories(out_dir, ec);
    ASSERT(!ec);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  const std::string pvtu_path = out_dir + "/mesh.pvtu";

  svmp::MeshIOOptions save_opts;
  save_opts.format = "pvtu";
  save_opts.path = pvtu_path;
  dmesh.save_parallel(save_opts);

  MPI_Barrier(MPI_COMM_WORLD);

  svmp::MeshIOOptions load_opts;
  load_opts.format = "pvtu";
  load_opts.path = pvtu_path;
  auto loaded = svmp::DistributedMesh::load_parallel(load_opts, MPI_COMM_WORLD);

  const size_t expected_ghost_cells =
      static_cast<size_t>((rank > 0 ? 1 : 0) + (rank + 1 < world ? 1 : 0));

  ASSERT_EQ(loaded.global_n_cells(), static_cast<size_t>(world));
  ASSERT_EQ(loaded.n_ghost_cells(), expected_ghost_cells);

  // Sanity: ghost cells should have a different owner rank.
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(loaded.local_mesh().n_cells()); ++c) {
    if (!loaded.is_ghost_cell(c)) {
      continue;
    }
    const svmp::rank_t owner = loaded.owner_rank_cell(c);
    ASSERT(owner >= 0);
    ASSERT(owner < world);
    ASSERT(owner != static_cast<svmp::rank_t>(rank));
  }

  if (rank == 0) {
    std::cout << "PVTU roundtrip ghost test PASSED\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::error_code ec;
    std::filesystem::remove_all(out_dir, ec);
  }

  MPI_Finalize();
  return 0;
#endif
}
