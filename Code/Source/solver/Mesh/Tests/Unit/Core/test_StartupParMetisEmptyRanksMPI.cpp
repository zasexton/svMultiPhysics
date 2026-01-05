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
 * @file test_StartupParMetisEmptyRanksMPI.cpp
 * @brief MPI test ensuring ParMETIS startup partitioning handles empty ranks safely.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
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

static void build_two_tets_global(int rank,
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

  // Two tetrahedra sharing face (0,1,2).
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

static std::vector<gid_t> gather_owned_cell_gids(const DistributedMesh& mesh) {
  const int world = mesh.world_size();
  std::vector<gid_t> owned;
  owned.reserve(mesh.n_cells());

  const auto& gids = mesh.cell_gids();
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    if (!mesh.is_owned_cell(c)) {
      continue;
    }
    owned.push_back(gids[static_cast<size_t>(c)]);
  }

  int local_n = static_cast<int>(owned.size());
  std::vector<int> counts(static_cast<size_t>(world), 0);
  MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, mesh.mpi_comm());

  std::vector<int> displs(static_cast<size_t>(world), 0);
  int total = 0;
  for (int r = 0; r < world; ++r) {
    displs[static_cast<size_t>(r)] = total;
    total += counts[static_cast<size_t>(r)];
  }

  std::vector<gid_t> all(static_cast<size_t>(total), INVALID_GID);
  MPI_Allgatherv(owned.data(),
                 local_n,
                 MPI_INT64_T,
                 all.data(),
                 counts.data(),
                 displs.data(),
                 MPI_INT64_T,
                 mesh.mpi_comm());
  return all;
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  const bool debug = (std::getenv("SVMP_TEST_DEBUG") != nullptr);
  if (debug && rank == 0) {
    std::cerr << "[StartupParMetisEmptyRanksMPI] world_size=" << world << "\n" << std::flush;
  }

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

  // Pick a tiny mesh so multi-rank runs necessarily have empty ranks.
  constexpr int n_cells = 2;
  std::vector<svmp::real_t> X_ref;
  std::vector<svmp::offset_t> cell2v_offsets;
  std::vector<svmp::index_t> cell2v;
  std::vector<svmp::CellShape> shapes;
  svmp::test::build_two_tets_global(rank, X_ref, cell2v_offsets, cell2v, shapes);

  svmp::DistributedMesh mesh(MPI_COMM_WORLD);
  std::unordered_map<std::string, std::string> opts;
  std::string partition_method = "parmetis";
  if (const char* env = std::getenv("SVMP_TEST_PARTITION_METHOD")) {
    partition_method = env;
  }
  opts["partition_method"] = partition_method;
  opts["parmetis_algorithm"] = "mesh";
  if (debug && rank == 0) {
    std::cerr << "[StartupParMetisEmptyRanksMPI] building mesh (partition_method=" << partition_method << ")\n"
              << std::flush;
  }
  int ghost_layers = 1;
  if (const char* env = std::getenv("SVMP_TEST_GHOST_LAYERS")) {
    ghost_layers = std::max(0, std::atoi(env));
  }
  mesh.build_from_arrays_global_and_partition(3,
                                              X_ref,
                                              cell2v_offsets,
                                              cell2v,
                                              shapes,
                                              svmp::PartitionHint::Cells,
                                              ghost_layers,
                                              opts);
  if (debug && rank == 0) {
    std::cerr << "[StartupParMetisEmptyRanksMPI] build complete\n" << std::flush;
  }

  std::uint64_t local_owned = 0;
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
    if (mesh.is_owned_cell(c)) {
      ++local_owned;
    }
  }

  std::uint64_t global_owned = 0;
  MPI_Allreduce(&local_owned, &global_owned, 1, MPI_UINT64_T, MPI_SUM, mesh.mpi_comm());
  ASSERT(global_owned == static_cast<std::uint64_t>(n_cells));

  if (debug && rank == 0) {
    std::cerr << "[StartupParMetisEmptyRanksMPI] gathering owned GIDs\n" << std::flush;
  }
  auto all_owned_gids = svmp::test::gather_owned_cell_gids(mesh);
  std::sort(all_owned_gids.begin(), all_owned_gids.end());

  ASSERT(all_owned_gids.size() == static_cast<size_t>(n_cells));
  ASSERT(std::unique(all_owned_gids.begin(), all_owned_gids.end()) == all_owned_gids.end());
  for (int i = 0; i < n_cells; ++i) {
    ASSERT(all_owned_gids[static_cast<size_t>(i)] == static_cast<svmp::gid_t>(i));
  }
  if (debug && rank == 0) {
    std::cerr << "[StartupParMetisEmptyRanksMPI] done\n" << std::flush;
  }

  MPI_Finalize();
  return 0;
}
