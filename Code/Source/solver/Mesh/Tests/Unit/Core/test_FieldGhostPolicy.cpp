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
 * @file test_FieldGhostPolicy.cpp
 * @brief MPI test that validates FieldGhostPolicy::Exchange integrates with DistributedMesh ghost updates.
 *
 * This test checks:
 * - FieldDescriptor::ghost_policy survives ghost-layer rebuilds
 * - DistributedMesh::update_exchange_ghost_fields() updates ghost/shared entity values
 * - Fields without Exchange policy are not updated
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

static void test_exchange_policy_updates_ghosts(int rank, int world_size) {
  if (world_size < 2) {
    if (rank == 0) {
      std::cout << "Skipping FieldGhostPolicy exchange test (requires >= 2 ranks)\n";
    }
    return;
  }

  auto local_mesh = create_hex_strip_partition(rank, world_size);
  DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  // Attach fields WITH descriptors before ghost-layer rebuild.
  FieldDescriptor vdesc = FieldDescriptor::scalar(EntityKind::Vertex);
  vdesc.ghost_policy = FieldGhostPolicy::Exchange;
  MeshFields::attach_field_with_descriptor(dmesh.local_mesh(), EntityKind::Vertex, "v_exchange",
                                           FieldScalarType::Float64, vdesc);

  FieldDescriptor cdesc = FieldDescriptor::scalar(EntityKind::Volume);
  cdesc.ghost_policy = FieldGhostPolicy::Exchange;
  MeshFields::attach_field_with_descriptor(dmesh.local_mesh(), EntityKind::Volume, "c_exchange",
                                           FieldScalarType::Float64, cdesc);

  FieldDescriptor edesc = FieldDescriptor::scalar(EntityKind::Edge);
  edesc.ghost_policy = FieldGhostPolicy::Exchange;
  MeshFields::attach_field_with_descriptor(dmesh.local_mesh(), EntityKind::Edge, "e_exchange",
                                           FieldScalarType::Float64, edesc);

  FieldDescriptor vlocal = FieldDescriptor::scalar(EntityKind::Vertex);
  vlocal.ghost_policy = FieldGhostPolicy::None;
  MeshFields::attach_field_with_descriptor(dmesh.local_mesh(), EntityKind::Vertex, "v_local",
                                           FieldScalarType::Float64, vlocal);

  dmesh.build_ghost_layer(1);

  // Verify descriptors survive ghost-layer rebuild.
  const auto vh = dmesh.local_mesh().field_handle(EntityKind::Vertex, "v_exchange");
  ASSERT(vh.id != 0);
  const auto* vdesc_after = dmesh.local_mesh().field_descriptor(vh);
  ASSERT(vdesc_after != nullptr);
  ASSERT(vdesc_after->ghost_policy == FieldGhostPolicy::Exchange);

  const auto ch = dmesh.local_mesh().field_handle(EntityKind::Volume, "c_exchange");
  ASSERT(ch.id != 0);
  const auto* cdesc_after = dmesh.local_mesh().field_descriptor(ch);
  ASSERT(cdesc_after != nullptr);
  ASSERT(cdesc_after->ghost_policy == FieldGhostPolicy::Exchange);

  const auto eh = dmesh.local_mesh().field_handle(EntityKind::Edge, "e_exchange");
  ASSERT(eh.id != 0);
  const auto* edesc_after = dmesh.local_mesh().field_descriptor(eh);
  ASSERT(edesc_after != nullptr);
  ASSERT(edesc_after->ghost_policy == FieldGhostPolicy::Exchange);

  // Initialize values: owned entities = rank, non-owned = sentinel.
  constexpr real_t sentinel = static_cast<real_t>(-12345.0);

  auto* v_exchange = dmesh.local_mesh().field_data_as<real_t>(vh);
  ASSERT(v_exchange != nullptr);
  for (index_t v = 0; v < static_cast<index_t>(dmesh.local_mesh().n_vertices()); ++v) {
    const rank_t owner = dmesh.owner_rank_vertex(v);
    v_exchange[v] = (owner == rank) ? static_cast<real_t>(rank) : sentinel;
  }

  auto* c_exchange = dmesh.local_mesh().field_data_as<real_t>(ch);
  ASSERT(c_exchange != nullptr);
  for (index_t c = 0; c < static_cast<index_t>(dmesh.local_mesh().n_cells()); ++c) {
    const rank_t owner = dmesh.owner_rank_cell(c);
    c_exchange[c] = (owner == rank) ? static_cast<real_t>(rank) : sentinel;
  }

  auto* e_exchange = dmesh.local_mesh().field_data_as<real_t>(eh);
  ASSERT(e_exchange != nullptr);
  for (index_t e = 0; e < static_cast<index_t>(dmesh.local_mesh().n_edges()); ++e) {
    const rank_t owner = dmesh.owner_rank_edge(e);
    e_exchange[e] = (owner == rank) ? static_cast<real_t>(rank) : sentinel;
  }

  const auto vlocal_h = dmesh.local_mesh().field_handle(EntityKind::Vertex, "v_local");
  ASSERT(vlocal_h.id != 0);
  auto* v_local_data = dmesh.local_mesh().field_data_as<real_t>(vlocal_h);
  ASSERT(v_local_data != nullptr);
  for (index_t v = 0; v < static_cast<index_t>(dmesh.local_mesh().n_vertices()); ++v) {
    const rank_t owner = dmesh.owner_rank_vertex(v);
    v_local_data[v] = (owner == rank) ? static_cast<real_t>(rank) : sentinel;
  }

  // Ensure we actually have ghosts to validate.
  int ghost_cells = 0;
  for (index_t c = 0; c < static_cast<index_t>(dmesh.local_mesh().n_cells()); ++c) {
    if (dmesh.is_ghost_cell(c)) ghost_cells++;
  }
  int ghost_verts = 0;
  for (index_t v = 0; v < static_cast<index_t>(dmesh.local_mesh().n_vertices()); ++v) {
    if (dmesh.is_ghost_vertex(v)) ghost_verts++;
  }
  ASSERT(ghost_cells > 0);
  ASSERT(ghost_verts > 0);
  int ghost_edges = 0;
  for (index_t e = 0; e < static_cast<index_t>(dmesh.local_mesh().n_edges()); ++e) {
    if (dmesh.is_ghost_edge(e)) ghost_edges++;
  }
  ASSERT(ghost_edges > 0);

  dmesh.update_exchange_ghost_fields();

  // Exchange field values should match the owner rank everywhere.
  for (index_t v = 0; v < static_cast<index_t>(dmesh.local_mesh().n_vertices()); ++v) {
    const rank_t owner = dmesh.owner_rank_vertex(v);
    ASSERT(owner >= 0);
    ASSERT_NEAR(v_exchange[v], static_cast<real_t>(owner), 1e-12);
  }

  for (index_t c = 0; c < static_cast<index_t>(dmesh.local_mesh().n_cells()); ++c) {
    const rank_t owner = dmesh.owner_rank_cell(c);
    ASSERT(owner >= 0);
    ASSERT_NEAR(c_exchange[c], static_cast<real_t>(owner), 1e-12);
  }

  for (index_t e = 0; e < static_cast<index_t>(dmesh.local_mesh().n_edges()); ++e) {
    const rank_t owner = dmesh.owner_rank_edge(e);
    ASSERT(owner >= 0);
    ASSERT_NEAR(e_exchange[e], static_cast<real_t>(owner), 1e-12);
  }

  // Non-exchange fields should remain unchanged on non-owned entities.
  for (index_t v = 0; v < static_cast<index_t>(dmesh.local_mesh().n_vertices()); ++v) {
    const rank_t owner = dmesh.owner_rank_vertex(v);
    if (owner != rank) {
      ASSERT_NEAR(v_local_data[v], sentinel, 1e-12);
    }
  }
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  svmp::test::test_exchange_policy_updates_ghosts(rank, world);

  if (rank == 0) {
    std::cout << "Field ghost-policy exchange tests PASSED\n";
  }

  MPI_Finalize();
  return 0;
}
