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
 * @file test_HangingVertexConstraintsMPI.cpp
 * @brief MPI tests for HangingVertexConstraints synchronization.
 */

#include "../../../Constraints/HangingVertexConstraints.h"
#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>
#include <cmath>
#include <iostream>
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

static void test_constraint_sync(DistributedMesh& mesh) {
  const gid_t constrained_gid = 0;
  const gid_t parent_gid_a = 1;
  const gid_t parent_gid_b = 2;

  const index_t constrained_local = mesh.global_to_local_vertex(constrained_gid);
  ASSERT(constrained_local != INVALID_INDEX);

  const rank_t owner = mesh.owner_rank_vertex(constrained_local);
  ASSERT(owner >= 0);

  HangingVertexConstraints constraints;

  if (mesh.rank() == owner) {
    const index_t pa = mesh.global_to_local_vertex(parent_gid_a);
    const index_t pb = mesh.global_to_local_vertex(parent_gid_b);
    ASSERT(pa != INVALID_INDEX);
    ASSERT(pb != INVALID_INDEX);

    HangingVertexConstraint c;
    c.constrained_vertex = constrained_local;
    c.parent_type = ConstraintParentType::Edge;
    // Intentionally reverse parent order; sync canonicalizes by parent GID.
    c.parent_vertices = {pb, pa};
    c.weights = {0.2, 0.8};  // weight(pb)=0.2, weight(pa)=0.8
    c.refinement_level = 7;
    ASSERT(constraints.add_constraint(c));
  }

  const bool ok = constraints.synchronize(mesh, /*weight_tolerance=*/1e-12);
  ASSERT(ok);

  ASSERT(constraints.is_hanging(constrained_local));
  const auto synced = constraints.get_constraint(constrained_local);
  ASSERT(synced.is_valid());

  // Check by GID to avoid dependence on local index ordering.
  const auto& vg = mesh.local_mesh().vertex_gids();
  gid_t seen_parent_a = INVALID_GID;
  gid_t seen_parent_b = INVALID_GID;
  real_t w_a = 0.0;
  real_t w_b = 0.0;
  for (size_t i = 0; i < synced.parent_vertices.size(); ++i) {
    const index_t pv = synced.parent_vertices[i];
    ASSERT(pv >= 0);
    const gid_t pg = vg[static_cast<size_t>(pv)];
    if (pg == parent_gid_a) {
      seen_parent_a = pg;
      w_a = synced.weights[i];
    } else if (pg == parent_gid_b) {
      seen_parent_b = pg;
      w_b = synced.weights[i];
    }
  }

  ASSERT(seen_parent_a == parent_gid_a);
  ASSERT(seen_parent_b == parent_gid_b);
  ASSERT(std::abs(w_a - 0.8) < 1e-12);
  ASSERT(std::abs(w_b - 0.2) < 1e-12);
  ASSERT(synced.refinement_level == 7);
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
  MPI_Comm_rank(comm, &rank);

  svmp::DistributedMesh mesh(comm);
  svmp::test::build_two_tets_global(mesh, rank);
  svmp::test::test_constraint_sync(mesh);

  if (split_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&split_comm);
  }

  MPI_Finalize();
  return 0;
}

