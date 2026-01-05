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
 * @file test_StartupParMetisCoordinatesMPI.cpp
 * @brief MPI test validating startup ParMETIS coordinate distribution correctness.
 *
 * This specifically exercises the two-phase startup path:
 * - scatter connectivity (no coords), ParMETIS partition, migrate cells
 * - distribute only needed vertex coordinates and build local MeshBase
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                               \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) <= (tol))

static void build_hex_chain_global_with_gid_coords(
    int n_cells,
    int rank,
    std::vector<real_t>& X_ref,
    std::vector<offset_t>& cell2v_offsets,
    std::vector<index_t>& cell2v,
    std::vector<CellShape>& shapes) {
  X_ref.clear();
  cell2v_offsets.clear();
  cell2v.clear();
  shapes.clear();

  if (rank != 0) {
    return;
  }

  const int nx = n_cells + 1;
  const int ny = 2;
  const int nz = 2;
  const int n_vertices = nx * ny * nz;

  X_ref.resize(static_cast<size_t>(n_vertices) * 3u);
  for (int v = 0; v < n_vertices; ++v) {
    X_ref[static_cast<size_t>(v) * 3u + 0u] = static_cast<real_t>(v);
    X_ref[static_cast<size_t>(v) * 3u + 1u] = static_cast<real_t>(v) + static_cast<real_t>(0.25);
    X_ref[static_cast<size_t>(v) * 3u + 2u] = static_cast<real_t>(-v);
  }

  auto vid = [&](int i, int j, int k) -> index_t {
    return static_cast<index_t>((i * ny + j) * nz + k);
  };

  cell2v_offsets.push_back(0);
  cell2v.reserve(static_cast<size_t>(n_cells) * 8u);
  shapes.reserve(static_cast<size_t>(n_cells));

  for (int c = 0; c < n_cells; ++c) {
    const index_t v0 = vid(c, 0, 0);
    const index_t v1 = vid(c + 1, 0, 0);
    const index_t v2 = vid(c + 1, 1, 0);
    const index_t v3 = vid(c, 1, 0);
    const index_t v4 = vid(c, 0, 1);
    const index_t v5 = vid(c + 1, 0, 1);
    const index_t v6 = vid(c + 1, 1, 1);
    const index_t v7 = vid(c, 1, 1);
    cell2v.insert(cell2v.end(), {v0, v1, v2, v3, v4, v5, v6, v7});
    cell2v_offsets.push_back(static_cast<offset_t>(cell2v.size()));
    shapes.push_back({CellFamily::Hex, 8, 1});
  }
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (world <= 1) {
    MPI_Finalize();
    return 0;
  }

#if !defined(SVMP_HAS_PARMETIS)
  if (rank == 0) {
    std::cerr << "Skipping: SVMP_HAS_PARMETIS not enabled.\n";
  }
  MPI_Finalize();
  return 0;
#endif

  const int n_cells = world * 8;
  std::vector<svmp::real_t> X_ref;
  std::vector<svmp::offset_t> cell2v_offsets;
  std::vector<svmp::index_t> cell2v;
  std::vector<svmp::CellShape> shapes;
  svmp::test::build_hex_chain_global_with_gid_coords(n_cells, rank, X_ref, cell2v_offsets, cell2v, shapes);

  svmp::DistributedMesh mesh(MPI_COMM_WORLD);
  std::unordered_map<std::string, std::string> opts;
  opts["partition_method"] = "parmetis";
  mesh.build_from_arrays_global_and_partition(3,
                                              X_ref,
                                              cell2v_offsets,
                                              cell2v,
                                              shapes,
                                              svmp::PartitionHint::Cells,
                                              /*ghost_layers=*/0,
                                              opts);

  // Global cell ownership is complete.
  std::uint64_t local_cells = static_cast<std::uint64_t>(mesh.n_cells());
  std::uint64_t global_cells = 0;
  MPI_Allreduce(&local_cells, &global_cells, 1, MPI_UINT64_T, MPI_SUM, mesh.mpi_comm());
  ASSERT(global_cells == static_cast<std::uint64_t>(n_cells));

  // Vertex coordinates correspond to global vertex IDs (exact equality expected).
  const auto& vg = mesh.vertex_gids();
  const auto& coords = mesh.X_ref();
  ASSERT(coords.size() == mesh.n_vertices() * 3u);

  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
    const svmp::gid_t gid = vg[static_cast<size_t>(v)];
    ASSERT(gid >= 0);
    const svmp::real_t expected0 = static_cast<svmp::real_t>(gid);
    const svmp::real_t expected1 = static_cast<svmp::real_t>(gid) + static_cast<svmp::real_t>(0.25);
    const svmp::real_t expected2 = static_cast<svmp::real_t>(-gid);

    const size_t off = static_cast<size_t>(v) * 3u;
    ASSERT_NEAR(coords[off + 0u], expected0, 0.0);
    ASSERT_NEAR(coords[off + 1u], expected1, 0.0);
    ASSERT_NEAR(coords[off + 2u], expected2, 0.0);
  }

  MPI_Finalize();
  return 0;
}
