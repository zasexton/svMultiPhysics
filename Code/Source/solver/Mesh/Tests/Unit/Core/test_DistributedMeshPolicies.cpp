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
 * @file test_DistributedMeshPolicies.cpp
 * @brief MPI test validating Phase 4 construction/I-O policy entry points.
 */

#include "Mesh.h"
#include "Observer/ObserverRegistry.h"

#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
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
  // 4 vertices per x-plane.
  return static_cast<gid_t>(x_plane * 4 + (y + 2 * z));
}

static index_t strip_vertex_lid(int x_plane, int y, int z) {
  return static_cast<index_t>(x_plane * 4 + (y + 2 * z));
}

static void build_hex_strip_global_arrays(int n_cells,
                                         std::vector<real_t>& coords,
                                         std::vector<offset_t>& offsets,
                                         std::vector<index_t>& conn,
                                         std::vector<CellShape>& shapes) {
  const int n_planes = n_cells + 1;
  const int n_vertices = 4 * n_planes;

  coords.clear();
  coords.reserve(static_cast<size_t>(n_vertices) * 3u);

  for (int x_plane = 0; x_plane < n_planes; ++x_plane) {
    // local order per plane: (y,z) = (0,0), (1,0), (0,1), (1,1) via (y + 2*z).
    for (int z = 0; z <= 1; ++z) {
      for (int y = 0; y <= 1; ++y) {
        coords.push_back(static_cast<real_t>(x_plane));
        coords.push_back(static_cast<real_t>(y));
        coords.push_back(static_cast<real_t>(z));
      }
    }
  }

  offsets.assign(static_cast<size_t>(n_cells) + 1u, 0);
  conn.clear();
  conn.reserve(static_cast<size_t>(n_cells) * 8u);
  shapes.assign(static_cast<size_t>(n_cells), CellShape{CellFamily::Hex, 8, 1});

  offsets[0] = 0;
  for (int c = 0; c < n_cells; ++c) {
    // Hex ordering consistent with other tests:
    // 0:(x0,0,0) 1:(x1,0,0) 2:(x1,1,0) 3:(x0,1,0)
    // 4:(x0,0,1) 5:(x1,0,1) 6:(x1,1,1) 7:(x0,1,1)
    const int x0 = c;
    const int x1 = c + 1;

    conn.push_back(strip_vertex_lid(x0, 0, 0));
    conn.push_back(strip_vertex_lid(x1, 0, 0));
    conn.push_back(strip_vertex_lid(x1, 1, 0));
    conn.push_back(strip_vertex_lid(x0, 1, 0));
    conn.push_back(strip_vertex_lid(x0, 0, 1));
    conn.push_back(strip_vertex_lid(x1, 0, 1));
    conn.push_back(strip_vertex_lid(x1, 1, 1));
    conn.push_back(strip_vertex_lid(x0, 1, 1));

    offsets[static_cast<size_t>(c) + 1u] = static_cast<offset_t>(conn.size());
  }
}

static std::shared_ptr<MeshBase> build_hex_strip_global_mesh(int n_cells) {
  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;
  build_hex_strip_global_arrays(n_cells, coords, offsets, conn, shapes);

  // Provide stable vertex/cell GIDs so shared detection after load/distribute is well-defined.
  std::vector<gid_t> vertex_gids(static_cast<size_t>(4 * (n_cells + 1)));
  for (int x_plane = 0; x_plane < n_cells + 1; ++x_plane) {
    for (int z = 0; z <= 1; ++z) {
      for (int y = 0; y <= 1; ++y) {
        vertex_gids[static_cast<size_t>(strip_vertex_lid(x_plane, y, z))] = strip_vertex_gid(x_plane, y, z);
      }
    }
  }

  std::vector<gid_t> cell_gids(static_cast<size_t>(n_cells));
  for (int c = 0; c < n_cells; ++c) {
    cell_gids[static_cast<size_t>(c)] = static_cast<gid_t>(c);
  }

  auto mesh = std::make_shared<MeshBase>();
  mesh->build_from_arrays(3, coords, offsets, conn, shapes);
  mesh->set_vertex_gids(std::move(vertex_gids));
  mesh->set_cell_gids(std::move(cell_gids));
  mesh->finalize();
  return mesh;
}

static std::string broadcast_string(const std::string& s, int root, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  int len = (rank == root) ? static_cast<int>(s.size()) : 0;
  MPI_Bcast(&len, 1, MPI_INT, root, comm);

  std::vector<char> buf(static_cast<size_t>(len) + 1u, '\0');
  if (rank == root && len > 0) {
    std::memcpy(buf.data(), s.data(), static_cast<size_t>(len));
  }
  if (len > 0) {
    MPI_Bcast(buf.data(), len, MPI_CHAR, root, comm);
  }
  return std::string(buf.data(), static_cast<size_t>(len));
}

static void test_global_build_and_partition(int rank, int world_size) {
  const int n_cells_global = world_size; // One cube per rank.

  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;

  if (rank == 0) {
    build_hex_strip_global_arrays(n_cells_global, coords, offsets, conn, shapes);
  }

  DistributedMesh dmesh(MPI_COMM_WORLD);
  dmesh.build_from_arrays_global_and_partition(3,
                                               coords,
                                               offsets,
                                               conn,
                                               shapes,
                                               PartitionHint::Cells,
                                               /*ghost_layers=*/0,
                                               {{"partition_method", "block"}});

  ASSERT_EQ(dmesh.world_size(), world_size);
  ASSERT_EQ(dmesh.rank(), rank);
  ASSERT_EQ(dmesh.global_n_cells(), static_cast<size_t>(n_cells_global));
  ASSERT_EQ(dmesh.n_cells(), 1u);
  ASSERT_EQ(dmesh.n_ghost_cells(), 0u);
  ASSERT_EQ(dmesh.n_ghost_vertices(), 0u);

  const size_t expected_shared_vertices = (rank == 0) ? 0u : 4u;
  ASSERT_EQ(dmesh.n_shared_vertices(), expected_shared_vertices);

  // PartitionChanged event should be emitted when exchange patterns are (re)built.
  auto counter = ObserverRegistry::attach_event_counter(dmesh.event_bus());
  counter->reset();
  dmesh.build_exchange_patterns();
  ASSERT_EQ(counter->count(MeshEvent::PartitionChanged), 1u);
}

static void test_load_mesh_distributes_when_vtk_available(int rank, int world_size) {
#if !defined(MESH_HAS_VTK)
  if (rank == 0) {
    std::cout << "Skipping load_mesh policy test (VTK disabled)\n";
  }
  (void)world_size;
  return;
#else
  const int n_cells_global = world_size;

  // Rank 0 writes a serial VTU that load_mesh will distribute.
  std::string filename;
  if (rank == 0) {
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    filename = "svmp_load_mesh_policy_" + std::to_string(world_size) + "_" +
               std::to_string(static_cast<long long>(stamp)) + ".vtu";

    auto global_mesh = build_hex_strip_global_mesh(n_cells_global);
    MeshIOOptions opts;
    opts.path = filename;
    opts.format = "vtu";
    global_mesh->save(opts);
  }

  filename = broadcast_string(filename, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  MeshIOOptions load_opts;
  load_opts.path = filename;
  load_opts.format = "vtu";
  load_opts.kv["partition_method"] = "block";

  auto mesh = svmp::load_mesh(load_opts, MeshComm::world());

  ASSERT(mesh != nullptr);
  ASSERT_EQ(mesh->world_size(), world_size);
  ASSERT_EQ(mesh->rank(), rank);

  ASSERT_EQ(mesh->global_n_cells(), static_cast<size_t>(n_cells_global));
  ASSERT_EQ(mesh->n_cells(), 1u);
  ASSERT_EQ(mesh->n_ghost_cells(), 0u);
  ASSERT_EQ(mesh->n_ghost_vertices(), 0u);

  const size_t expected_global_vertices = static_cast<size_t>(4 * (n_cells_global + 1));
  ASSERT_EQ(mesh->global_n_vertices(), expected_global_vertices);

  // Shared-vertex policy should match the strip partition ownership rule (lowest rank owns).
  const size_t expected_shared_vertices = (rank == 0) ? 0u : 4u;
  ASSERT_EQ(mesh->n_shared_vertices(), expected_shared_vertices);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::error_code ec;
    std::filesystem::remove(filename, ec);
  }
#endif
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  svmp::test::test_global_build_and_partition(rank, world_size);
  svmp::test::test_load_mesh_distributes_when_vtk_available(rank, world_size);

  MPI_Finalize();
  return 0;
}
