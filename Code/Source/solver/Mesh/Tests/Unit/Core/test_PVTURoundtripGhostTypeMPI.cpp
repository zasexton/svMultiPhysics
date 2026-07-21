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

#include <cmath>
#include <cstdint>
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

static void attach_point_verification_fields(DistributedMesh& dmesh) {
  auto& mesh = dmesh.local_mesh();
  const auto node_id_handle = mesh.attach_field(
      EntityKind::Vertex, "GlobalNodeID", FieldScalarType::Int32, 1);
  const auto value_handle = mesh.attach_field(
      EntityKind::Vertex, "PointVerificationValue", FieldScalarType::Float64, 1);
  auto* node_ids = mesh.field_data_as<std::int32_t>(node_id_handle);
  auto* values = mesh.field_data_as<real_t>(value_handle);
  ASSERT(node_ids != nullptr);
  ASSERT(values != nullptr);

  for (index_t vertex = 0;
       vertex < static_cast<index_t>(mesh.n_vertices()); ++vertex) {
    const auto gid = mesh.vertex_gids()[static_cast<size_t>(vertex)];
    node_ids[vertex] = static_cast<std::int32_t>(gid);
    values[vertex] = static_cast<real_t>(gid) + real_t{0.25};
  }
}

static void verify_written_point_duplicates(const DistributedMesh& dmesh,
                                            const std::string& piece_path,
                                            int rank,
                                            int world) {
  MeshIOOptions piece_opts;
  piece_opts.format = "vtu";
  piece_opts.path = piece_path;
  const auto piece = MeshBase::load(piece_opts);

  const auto ghost_handle =
      piece.field_handle(EntityKind::Vertex, "vtkGhostType");
  const auto node_id_handle =
      piece.field_handle(EntityKind::Vertex, "GlobalNodeID");
  const auto value_handle =
      piece.field_handle(EntityKind::Vertex, "PointVerificationValue");
  ASSERT(ghost_handle.id != 0);
  ASSERT(node_id_handle.id != 0);
  ASSERT(value_handle.id != 0);
  ASSERT_EQ(piece.field_type(ghost_handle), FieldScalarType::UInt8);
  ASSERT_EQ(piece.field_type(node_id_handle), FieldScalarType::Int32);
  ASSERT_EQ(piece.field_type(value_handle), FieldScalarType::Float64);

  const auto* ghosts =
      piece.field_data_as<const std::uint8_t>(ghost_handle);
  const auto* node_ids =
      piece.field_data_as<const std::int32_t>(node_id_handle);
  const auto* values = piece.field_data_as<const real_t>(value_handle);
  ASSERT(ghosts != nullptr);
  ASSERT(node_ids != nullptr);
  ASSERT(values != nullptr);

  const size_t global_vertex_count = static_cast<size_t>(4 * (world + 1));
  std::vector<int> local_copy_count(global_vertex_count, 0);
  std::vector<int> local_non_ghost_count(global_vertex_count, 0);
  const auto& source_mesh = dmesh.local_mesh();

  ASSERT_EQ(piece.n_vertices(), source_mesh.n_vertices());
  for (index_t vertex = 0;
       vertex < static_cast<index_t>(piece.n_vertices()); ++vertex) {
    const auto gid = piece.vertex_gids()[static_cast<size_t>(vertex)];
    ASSERT(gid >= 0);
    ASSERT(static_cast<size_t>(gid) < global_vertex_count);
    ASSERT_EQ(node_ids[vertex], static_cast<std::int32_t>(gid));
    ASSERT(std::abs(values[vertex] -
                    (static_cast<real_t>(gid) + real_t{0.25})) <= real_t{1e-12});
    ASSERT(ghosts[vertex] == std::uint8_t{0} ||
           ghosts[vertex] == std::uint8_t{1});

    const auto source_vertex = source_mesh.global_to_local_vertex(gid);
    ASSERT(source_vertex != INVALID_INDEX);
    const bool expected_duplicate =
        dmesh.is_ghost_vertex(source_vertex) ||
        (dmesh.is_shared_vertex(source_vertex) &&
         dmesh.owner_rank_vertex(source_vertex) != rank);
    ASSERT_EQ(ghosts[vertex],
              expected_duplicate ? std::uint8_t{1} : std::uint8_t{0});

    ++local_copy_count[static_cast<size_t>(gid)];
    if (ghosts[vertex] == std::uint8_t{0}) {
      ++local_non_ghost_count[static_cast<size_t>(gid)];
    }
  }

  std::vector<int> global_copy_count(global_vertex_count, 0);
  std::vector<int> global_non_ghost_count(global_vertex_count, 0);
  MPI_Allreduce(local_copy_count.data(),
                global_copy_count.data(),
                static_cast<int>(global_vertex_count),
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(local_non_ghost_count.data(),
                global_non_ghost_count.data(),
                static_cast<int>(global_vertex_count),
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);

  int duplicated_global_ids = 0;
  for (size_t gid = 0; gid < global_vertex_count; ++gid) {
    ASSERT(global_copy_count[gid] >= 1);
    ASSERT_EQ(global_non_ghost_count[gid], 1);
    if (global_copy_count[gid] > 1) {
      ++duplicated_global_ids;
    }
  }
  ASSERT(duplicated_global_ids > 0);
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
  svmp::test::attach_point_verification_fields(dmesh);

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
  save_opts.kv["binary"] = "true";
  save_opts.kv["streaming"] = "true";
  dmesh.save_parallel(save_opts);

  MPI_Barrier(MPI_COMM_WORLD);

  svmp::test::verify_written_point_duplicates(
      dmesh,
      out_dir + "/mesh_p" + std::to_string(rank) + ".vtu",
      rank,
      world);

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
