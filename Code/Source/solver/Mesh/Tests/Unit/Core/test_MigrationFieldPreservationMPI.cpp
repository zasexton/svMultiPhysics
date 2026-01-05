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
 * @file test_MigrationFieldPreservationMPI.cpp
 * @brief MPI test ensuring DistributedMesh::migrate preserves fields when entity GIDs overlap.
 *
 * This specifically targets the case where kept entities and received packets contain
 * duplicate GIDs (e.g., shared vertices). The migrated mesh de-duplicates by GID, so field
 * restoration must map by GID to avoid buffer overruns / incorrect associations.
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
#define ASSERT_NEAR(a, b, tol) ASSERT(std::fabs((a) - (b)) <= (tol))

static std::shared_ptr<MeshBase> make_rank_tet(int rank) {
  // Each rank owns one tetra. All ranks share vertex GID=0.
  // Rank r also has three unique vertex gids: 3*r+1..3*r+3.
  auto mesh = std::make_shared<MeshBase>();

  const gid_t shared_gid = 0;
  const gid_t g1 = static_cast<gid_t>(3 * rank + 1);
  const gid_t g2 = static_cast<gid_t>(3 * rank + 2);
  const gid_t g3 = static_cast<gid_t>(3 * rank + 3);

  std::vector<gid_t> vertex_gids = {shared_gid, g1, g2, g3};

  // Place the shared vertex at the origin and offset the other vertices by rank
  // so geometry is well-defined.
  const real_t s = static_cast<real_t>(rank + 1);
  std::vector<real_t> coords = {
      0.0, 0.0, 0.0,
      s,   0.0, 0.0,
      0.0, s,   0.0,
      0.0, 0.0, s,
  };

  std::vector<offset_t> offsets = {0, 4};
  std::vector<index_t> conn = {0, 1, 2, 3};
  std::vector<CellShape> shapes = {{CellFamily::Tetra, 4, 1}};
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

  if (world < 2) {
    if (rank == 0) {
      std::cout << "Skipping migration field preservation test (requires >= 2 ranks)\n";
    }
    MPI_Finalize();
    return 0;
  }

  auto local_mesh = svmp::test::make_rank_tet(rank);
  svmp::DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  // Attach fields whose values are derived from GIDs.
  auto v_field = dmesh.local_mesh().attach_field(
      svmp::EntityKind::Vertex, "v_gid",
      svmp::FieldScalarType::Float64, 1);
  auto c_field = dmesh.local_mesh().attach_field(
      svmp::EntityKind::Volume, "c_gid",
      svmp::FieldScalarType::Float64, 1);

  auto* v_data = dmesh.local_mesh().field_data_as<svmp::real_t>(v_field);
  const auto& v_gids = dmesh.local_mesh().vertex_gids();
  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(v_gids.size()); ++v) {
    v_data[static_cast<size_t>(v)] = static_cast<svmp::real_t>(v_gids[static_cast<size_t>(v)]);
  }

  auto* c_data = dmesh.local_mesh().field_data_as<svmp::real_t>(c_field);
  const auto& c_gids = dmesh.local_mesh().cell_gids();
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(c_gids.size()); ++c) {
    c_data[static_cast<size_t>(c)] = static_cast<svmp::real_t>(c_gids[static_cast<size_t>(c)]);
  }

  // Migrate all cells to rank 0. This forces rank 0 to receive multiple packets
  // containing the shared vertex GID=0, exercising the duplicate-GID restoration path.
  std::vector<svmp::rank_t> new_owner(dmesh.local_mesh().n_cells(), 0);
  dmesh.migrate(new_owner);

  if (rank == 0) {
    const auto& mesh = dmesh.local_mesh();

    ASSERT_EQ(mesh.n_cells(), static_cast<size_t>(world));
    ASSERT_EQ(mesh.n_vertices(), static_cast<size_t>(1 + 3 * world));

    ASSERT(mesh.has_field(svmp::EntityKind::Vertex, "v_gid"));
    ASSERT(mesh.has_field(svmp::EntityKind::Volume, "c_gid"));

    const auto v_h = mesh.field_handle(svmp::EntityKind::Vertex, "v_gid");
    const auto c_h = mesh.field_handle(svmp::EntityKind::Volume, "c_gid");
    ASSERT(v_h.id != 0);
    ASSERT(c_h.id != 0);

    const auto* v_out = mesh.field_data_as<svmp::real_t>(v_h);
    const auto* c_out = mesh.field_data_as<svmp::real_t>(c_h);

    const auto& new_v_gids = mesh.vertex_gids();
    for (size_t i = 0; i < new_v_gids.size(); ++i) {
      ASSERT_NEAR(v_out[i], static_cast<svmp::real_t>(new_v_gids[i]), 1e-12);
    }

    const auto& new_c_gids = mesh.cell_gids();
    for (size_t i = 0; i < new_c_gids.size(); ++i) {
      ASSERT_NEAR(c_out[i], static_cast<svmp::real_t>(new_c_gids[i]), 1e-12);
    }

    std::cout << "Migration field preservation test PASSED\n";
  } else {
    ASSERT_EQ(dmesh.local_mesh().n_cells(), 0u);
  }

  MPI_Finalize();
  return 0;
}

