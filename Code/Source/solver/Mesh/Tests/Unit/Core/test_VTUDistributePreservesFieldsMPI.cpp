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
 * @file test_VTUDistributePreservesFieldsMPI.cpp
 * @brief MPI test ensuring root-load + distribute preserves labels and fields.
 *
 * This exercises the `DistributedMesh::load_parallel` serial-format path where
 * rank 0 loads a single-file VTU and distributes cell partitions to each rank.
 *
 * Production requirement: region labels and attached vertex/cell fields must not
 * be lost during distribution.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"

#include <mpi.h>

#include <cmath>
#include <filesystem>
#include <iostream>
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
#define ASSERT_NEAR(a, b, tol) ASSERT(std::fabs((a) - (b)) <= (tol))

static svmp::MeshBase make_global_tets(int world_size) {
  const int dim = 3;
  const int n_cells = world_size;
  const int verts_per_cell = 4;
  const int n_vertices = n_cells * verts_per_cell;

  std::vector<svmp::real_t> coords;
  coords.reserve(static_cast<size_t>(n_vertices) * dim);
  std::vector<svmp::gid_t> vertex_gids;
  vertex_gids.reserve(static_cast<size_t>(n_vertices));

  for (int c = 0; c < n_cells; ++c) {
    for (int j = 0; j < verts_per_cell; ++j) {
      const int v = c * verts_per_cell + j;
      vertex_gids.push_back(static_cast<svmp::gid_t>(v));
    }

    const svmp::real_t base = static_cast<svmp::real_t>(c) * 2.0;
    coords.insert(coords.end(),
                  {
                      base, 0.0, 0.0,
                      base + 1.0, 0.0, 0.0,
                      base, 1.0, 0.0,
                      base, 0.0, 1.0,
                  });
  }

  std::vector<svmp::offset_t> offsets;
  offsets.reserve(static_cast<size_t>(n_cells) + 1);
  offsets.push_back(0);

  std::vector<svmp::index_t> conn;
  conn.reserve(static_cast<size_t>(n_cells) * verts_per_cell);

  std::vector<svmp::CellShape> shapes;
  shapes.reserve(static_cast<size_t>(n_cells));

  std::vector<svmp::gid_t> cell_gids;
  cell_gids.reserve(static_cast<size_t>(n_cells));

  for (int c = 0; c < n_cells; ++c) {
    shapes.push_back({svmp::CellFamily::Tetra, 4, 1});
    cell_gids.push_back(static_cast<svmp::gid_t>(c));

    const int base_v = c * verts_per_cell;
    conn.push_back(base_v + 0);
    conn.push_back(base_v + 1);
    conn.push_back(base_v + 2);
    conn.push_back(base_v + 3);
    offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
  }

  svmp::MeshBase mesh;
  mesh.build_from_arrays(dim, coords, offsets, conn, shapes);
  mesh.set_vertex_gids(std::move(vertex_gids));
  mesh.set_cell_gids(std::move(cell_gids));

  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
    mesh.set_region_label(c, static_cast<svmp::label_t>(100 + c));
  }

  const auto v_field = mesh.attach_field(svmp::EntityKind::Vertex, "vval", svmp::FieldScalarType::Float64, 1);
  const auto c_field = mesh.attach_field(svmp::EntityKind::Volume, "cval", svmp::FieldScalarType::Float64, 1);

  auto* v_data = mesh.field_data_as<svmp::real_t>(v_field);
  auto* c_data = mesh.field_data_as<svmp::real_t>(c_field);

  const auto& vg = mesh.vertex_gids();
  for (size_t i = 0; i < vg.size(); ++i) {
    v_data[i] = 0.5 * static_cast<svmp::real_t>(vg[i]);
  }

  const auto& cg = mesh.cell_gids();
  for (size_t i = 0; i < cg.size(); ++i) {
    c_data[i] = 1000.0 + static_cast<svmp::real_t>(cg[i]);
  }

  mesh.finalize();
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
    std::cout << "Skipping VTU distribute field test (VTK disabled)\n";
  }
  MPI_Finalize();
  return 0;
#else
  if (world < 2) {
    if (rank == 0) {
      std::cout << "Skipping VTU distribute field test (requires >= 2 ranks)\n";
    }
    MPI_Finalize();
    return 0;
  }

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
      std::string("vtu_distribute_") + std::to_string(pid) + "_" + std::to_string(stamp);
  if (rank == 0) {
    std::error_code ec;
    std::filesystem::remove_all(out_dir, ec);
    std::filesystem::create_directories(out_dir, ec);
    ASSERT(!ec);

    auto mesh = svmp::test::make_global_tets(world);
    svmp::MeshIOOptions save_opts;
    save_opts.format = "vtu";
    save_opts.path = out_dir + "/mesh.vtu";
    mesh.save(save_opts);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  svmp::MeshIOOptions load_opts;
  load_opts.format = "vtu";
  load_opts.path = out_dir + "/mesh.vtu";
  auto dmesh = svmp::DistributedMesh::load_parallel(load_opts, MPI_COMM_WORLD);

  // Each rank should own exactly one cell (since n_cells == world).
  ASSERT_EQ(dmesh.local_mesh().n_cells(), 1u);

  const auto& mesh = dmesh.local_mesh();
  const svmp::gid_t cell_gid = mesh.cell_gids().empty() ? svmp::INVALID_GID : mesh.cell_gids()[0];
  ASSERT_EQ(cell_gid, static_cast<svmp::gid_t>(rank));

  ASSERT_EQ(mesh.region_label(0), static_cast<svmp::label_t>(100 + rank));

  // Cell field value should follow gid mapping.
  ASSERT(mesh.has_field(svmp::EntityKind::Volume, "cval"));
  const auto c_h = mesh.field_handle(svmp::EntityKind::Volume, "cval");
  const auto* c_data = mesh.field_data_as<const svmp::real_t>(c_h);
  ASSERT_NEAR(c_data[0], 1000.0 + static_cast<svmp::real_t>(rank), 1e-12);

  // Vertex field values should equal 0.5 * global vertex gid.
  ASSERT(mesh.has_field(svmp::EntityKind::Vertex, "vval"));
  const auto v_h = mesh.field_handle(svmp::EntityKind::Vertex, "vval");
  const auto* v_data = mesh.field_data_as<const svmp::real_t>(v_h);
  const auto& vg = mesh.vertex_gids();
  ASSERT_EQ(vg.size(), mesh.n_vertices());
  for (size_t i = 0; i < vg.size(); ++i) {
    ASSERT_NEAR(v_data[i], 0.5 * static_cast<svmp::real_t>(vg[i]), 1e-12);
  }

  if (rank == 0) {
    std::cout << "VTU distribute fields test PASSED\n";
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
