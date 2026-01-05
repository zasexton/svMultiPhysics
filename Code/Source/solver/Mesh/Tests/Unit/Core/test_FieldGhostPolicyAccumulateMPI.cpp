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
 * @file test_FieldGhostPolicyAccumulateMPI.cpp
 * @brief MPI test validating FieldGhostPolicy::Accumulate (sum + broadcast) behavior.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"

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

#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) < (tol))

static gid_t strip_vertex_gid(int x_plane, int y, int z) {
  // 4 vertices per x-plane with a fixed (y,z) ordering.
  return static_cast<gid_t>(x_plane * 4 + (y + 2 * z));
}

static std::shared_ptr<MeshBase> create_hex_strip_partition(int rank, int /*world_size*/) {
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

static real_t expected_strip_accum_value(int world_size, gid_t vertex_gid) {
  // vertex_gid = x_plane*4 + local_id.
  const int x_plane = static_cast<int>(vertex_gid / 4);

  // The strip has (world_size) cells and (world_size+1) x-planes.
  // Each internal plane is shared by exactly two ranks: x_plane-1 and x_plane.
  if (x_plane <= 0) {
    return static_cast<real_t>(1.0);  // rank 0 contributes 1.0
  }
  if (x_plane >= world_size) {
    return static_cast<real_t>(world_size);  // last rank contributes world_size
  }

  // sum = (x_plane-1 + 1) + (x_plane + 1) = 2*x_plane + 1
  return static_cast<real_t>(2 * x_plane + 1);
}

static void test_accumulate_policy_sums_and_broadcasts(int rank, int world_size) {
  if (world_size < 2) {
    if (rank == 0) {
      std::cout << "Skipping accumulate test (requires >= 2 ranks)\n";
    }
    return;
  }

  auto local_mesh = create_hex_strip_partition(rank, world_size);
  DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  FieldDescriptor desc = FieldDescriptor::scalar(EntityKind::Vertex);
  desc.ghost_policy = FieldGhostPolicy::Accumulate;
  MeshFields::attach_field_with_descriptor(dmesh.local_mesh(),
                                           EntityKind::Vertex,
                                           "v_accumulate",
                                           FieldScalarType::Float64,
                                           desc);

  dmesh.build_exchange_patterns();

  const auto h = dmesh.local_mesh().field_handle(EntityKind::Vertex, "v_accumulate");
  ASSERT(h.id != 0);
  auto* data = dmesh.local_mesh().field_data_as<real_t>(h);
  ASSERT(data != nullptr);

  // Each rank contributes a constant value (rank+1) to every local vertex.
  const real_t contrib = static_cast<real_t>(rank + 1);
  for (index_t v = 0; v < static_cast<index_t>(dmesh.local_mesh().n_vertices()); ++v) {
    data[v] = contrib;
  }

  dmesh.update_exchange_ghost_fields();

  // After Accumulate: each shared vertex value should equal the sum of rank contributions,
  // and the result should be broadcast so all copies match.
  const auto& vgids = dmesh.local_mesh().vertex_gids();
  ASSERT(vgids.size() == dmesh.local_mesh().n_vertices());

  for (index_t v = 0; v < static_cast<index_t>(dmesh.local_mesh().n_vertices()); ++v) {
    const gid_t gid = vgids[static_cast<size_t>(v)];
    ASSERT(gid >= 0);
    const real_t expected = expected_strip_accum_value(world_size, gid);
    ASSERT_NEAR(data[v], expected, 1e-12);
  }
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  svmp::test::test_accumulate_policy_sums_and_broadcasts(rank, world);

  if (rank == 0) {
    std::cout << "FieldGhostPolicy::Accumulate tests PASSED\n";
  }

  MPI_Finalize();
  return 0;
}

