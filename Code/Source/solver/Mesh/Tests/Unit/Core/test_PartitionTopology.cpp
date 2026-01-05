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
 * @file test_PartitionTopology.cpp
 * @brief MPI test validating partition-boundary classification utilities.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/PartitionTopology.h"
#include "../../../Topology/DistributedTopology.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>
#include <iostream>
#include <sstream>
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

static void build_two_tets_global(DistributedMesh& mesh, int rank) {
  // Two tetrahedra sharing the face (0,1,2). With 2+ ranks, the default
  // partitioner assigns cell 0 to rank 0 and cell 1 to rank 1 (others empty).
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

static void test_partition_boundary_faces(const DistributedMesh& mesh, int rank, int world) {
  (void)world;

  const auto owned_iface = PartitionTopology::partition_boundary_faces(mesh, /*owned_only=*/true);
  const auto all_iface = PartitionTopology::partition_boundary_faces(mesh, /*owned_only=*/false);

  const size_t local_owned = owned_iface.size();
  const size_t local_all = all_iface.size();

  size_t global_owned = 0;
  size_t global_all = 0;
  MPI_Allreduce(&local_owned, &global_owned, 1, MPI_UNSIGNED_LONG, MPI_SUM, mesh.mpi_comm());
  MPI_Allreduce(&local_all, &global_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mesh.mpi_comm());

  // There is exactly one partition-interface face between the two tetrahedra.
  // With owned_only=true, it should appear exactly once globally.
  if (world >= 2) {
    if (global_owned != 1u) {
      std::ostringstream oss;
      oss << "rank " << rank << ": local_owned=" << local_owned << " local_all=" << local_all
          << " global_owned=" << global_owned << " global_all=" << global_all << "\n";

      const auto dump_faces = [&](const char* label, const std::vector<index_t>& faces) {
        oss << "  " << label << " (count=" << faces.size() << "):\n";
        for (const auto f : faces) {
          const auto fc = mesh.face_cells(f);
          const index_t c0 = fc[0];
          const index_t c1 = fc[1];
          oss << "    face " << f << " cells=(" << c0 << "," << c1 << ")"
              << " cell_owner_ranks=(" << mesh.owner_rank_cell(c0) << "," << mesh.owner_rank_cell(c1) << ")"
              << " face_owner_rank=" << mesh.owner_rank_face(f)
              << " face_owner=" << static_cast<int>(mesh.is_owned_face(f))
              << " face_shared=" << static_cast<int>(mesh.is_shared_face(f))
              << " face_ghost=" << static_cast<int>(mesh.is_ghost_face(f)) << "\n";
        }
      };

      dump_faces("owned_iface", owned_iface);
      dump_faces("all_iface", all_iface);

      std::cerr << oss.str() << std::flush;
    }
    ASSERT_EQ(global_owned, 1u);
    ASSERT_EQ(global_all, 2u);  // present on both ranks touching the interface
  } else {
    ASSERT_EQ(global_owned, 0u);
    ASSERT_EQ(global_all, 0u);
  }

  // If this rank owns a cell, it should have a partition-boundary neighbor.
  const auto owned_cells = mesh.owned_cells();
  if (!owned_cells.empty()) {
    ASSERT_EQ(owned_cells.size(), 1u);
    const index_t c = owned_cells.front();
    ASSERT(PartitionTopology::is_partition_boundary_cell(mesh, c));

    const auto adj = PartitionTopology::classify_cell_neighbors(mesh, c);
    ASSERT_EQ(adj.boundary_neighbors.size(), 1u);
    ASSERT_EQ(adj.internal_neighbors.size(), 0u);

    // Dual graph: the owned cell has exactly one neighbor (the other tet).
    const auto graph = DistributedTopology::build_global_dual_graph(mesh, /*owned_only=*/true);
    ASSERT_EQ(graph.local_cells.size(), 1u);
    ASSERT_EQ(graph.cell_gids.size(), 1u);
    ASSERT_EQ(graph.offsets.size(), 2u);
    ASSERT_EQ(graph.offsets[0], 0);
    ASSERT_EQ(graph.offsets[1], 1);
    ASSERT_EQ(graph.neighbors.size(), 1u);
    const gid_t my_gid = graph.cell_gids.front();
    const gid_t nbr_gid = graph.neighbors.front();
    ASSERT(my_gid != nbr_gid);
  } else {
    // Ranks with no owned cells should have no owned interface faces.
    ASSERT_EQ(local_owned, 0u);

    const auto graph = DistributedTopology::build_global_dual_graph(mesh, /*owned_only=*/true);
    ASSERT_EQ(graph.local_cells.size(), 0u);
    ASSERT_EQ(graph.cell_gids.size(), 0u);
    ASSERT_EQ(graph.offsets.size(), 1u);
    ASSERT_EQ(graph.offsets[0], 0);
    ASSERT_EQ(graph.neighbors.size(), 0u);
  }

  // Deterministic: the interface face owner is the lowest-rank participant.
  if (world >= 2) {
    if (rank == 0) {
      ASSERT_EQ(local_owned, 1u);
    } else if (rank == 1) {
      ASSERT_EQ(local_owned, 0u);
    }
  }

  // Global boundary faces: the shared interior face is excluded.
  const auto owned_global_bdry = DistributedTopology::global_boundary_faces(mesh, /*owned_only=*/true);
  const size_t local_global_bdry = owned_global_bdry.size();
  size_t global_global_bdry = 0;
  MPI_Allreduce(&local_global_bdry, &global_global_bdry, 1, MPI_UNSIGNED_LONG, MPI_SUM, mesh.mpi_comm());
  ASSERT_EQ(global_global_bdry, 6u);

  // Connected components: the two tets are connected through the shared face,
  // so the global component label is the minimum cell GID (0) everywhere.
  const auto comps = DistributedTopology::connected_components_global(mesh);
  ASSERT_EQ(comps.size(), mesh.n_cells());
  for (const auto c : mesh.owned_cells()) {
    ASSERT_EQ(comps[static_cast<size_t>(c)], 0);
  }

  // Parallel graph coloring: adjacent cells must have different colors.
  const auto colors = DistributedTopology::parallel_graph_coloring(mesh);
  ASSERT_EQ(colors.size(), mesh.n_cells());
  for (const auto c : mesh.owned_cells()) {
    ASSERT(colors[static_cast<size_t>(c)] >= 0);
    const auto neigh = mesh.cell_neighbors(c);
    for (const auto n : neigh) {
      if (n < 0 || static_cast<size_t>(n) >= colors.size()) continue;
      if (n == c) continue;
      ASSERT(colors[static_cast<size_t>(c)] != colors[static_cast<size_t>(n)]);
    }
  }

  int local_max_color = -1;
  for (const auto c : mesh.owned_cells()) {
    local_max_color = std::max(local_max_color, colors[static_cast<size_t>(c)]);
  }
  int global_max_color = -1;
  MPI_Allreduce(&local_max_color, &global_max_color, 1, MPI_INT, MPI_MAX, mesh.mpi_comm());
  ASSERT(global_max_color <= 1);
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // This test is designed for a 2-rank partition. When launched with more
  // ranks (ctest registers a 4-rank variant), restrict to ranks {0,1} to keep
  // expectations deterministic and avoid exercising empty-rank edge cases.
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
  int world = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world);

  svmp::DistributedMesh mesh(comm);
  svmp::test::build_two_tets_global(mesh, rank);
  svmp::test::test_partition_boundary_faces(mesh, rank, world);

  if (split_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&split_comm);
  }

  MPI_Finalize();
  return 0;
}
