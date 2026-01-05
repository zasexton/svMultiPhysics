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
 * @file test_PartitionQualityMetisMPI.cpp
 * @brief MPI test validating that METIS partitioning reduces edge cuts vs. block.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>
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

static void build_permuted_hex_chain_global(
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

  X_ref.reserve(static_cast<size_t>(nx * ny * nz) * 3u);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        X_ref.push_back(static_cast<real_t>(i));
        X_ref.push_back(static_cast<real_t>(j));
        X_ref.push_back(static_cast<real_t>(k));
      }
    }
  }

  auto vid = [&](int i, int j, int k) -> index_t {
    return static_cast<index_t>((i * ny + j) * nz + k);
  };

  std::vector<int> perm;
  perm.reserve(static_cast<size_t>(n_cells));
  int lo = 0;
  int hi = n_cells - 1;
  while (lo <= hi) {
    perm.push_back(lo++);
    if (lo <= hi) {
      perm.push_back(hi--);
    }
  }

  cell2v_offsets.push_back(0);
  cell2v.reserve(static_cast<size_t>(n_cells) * 8u);
  shapes.reserve(static_cast<size_t>(n_cells));

  for (const int c : perm) {
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

static size_t total_dual_edge_cuts_unique(const DistributedMesh& mesh) {
  const auto& gids = mesh.cell_gids();
  size_t local_unique = 0;

  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    if (!mesh.is_owned_cell(c)) {
      continue;
    }
    const gid_t cg = gids[static_cast<size_t>(c)];
    const rank_t cr = mesh.owner_rank_cell(c);

    const auto neigh = mesh.cell_neighbors(c);
    for (const auto n : neigh) {
      if (n < 0 || static_cast<size_t>(n) >= mesh.n_cells()) {
        continue;
      }
      if (mesh.owner_rank_cell(n) == cr) {
        continue;
      }
      const gid_t ng = gids[static_cast<size_t>(n)];
      if (cg < ng) {
        ++local_unique;
      }
    }
  }

  size_t global_unique = 0;
  MPI_Allreduce(&local_unique, &global_unique, 1, MPI_UNSIGNED_LONG, MPI_SUM, mesh.mpi_comm());
  return global_unique;
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

  const int n_cells = 32;
  std::vector<svmp::real_t> X_ref;
  std::vector<svmp::offset_t> cell2v_offsets;
  std::vector<svmp::index_t> cell2v;
  std::vector<svmp::CellShape> shapes;
  svmp::test::build_permuted_hex_chain_global(n_cells, rank, X_ref, cell2v_offsets, cell2v, shapes);

  svmp::DistributedMesh block_mesh(MPI_COMM_WORLD);
  std::unordered_map<std::string, std::string> block_opts;
  block_opts["partition_method"] = "block";
  block_mesh.build_from_arrays_global_and_partition(3,
                                                    X_ref,
                                                    cell2v_offsets,
                                                    cell2v,
                                                    shapes,
                                                    svmp::PartitionHint::Cells,
                                                    /*ghost_layers=*/1,
                                                    block_opts);

  svmp::DistributedMesh metis_mesh(MPI_COMM_WORLD);
  std::unordered_map<std::string, std::string> metis_opts;
  metis_opts["partition_method"] = "metis";
  metis_mesh.build_from_arrays_global_and_partition(3,
                                                    X_ref,
                                                    cell2v_offsets,
                                                    cell2v,
                                                    shapes,
                                                    svmp::PartitionHint::Cells,
                                                    /*ghost_layers=*/1,
                                                    metis_opts);

  const size_t cuts_block = svmp::test::total_dual_edge_cuts_unique(block_mesh);
  const size_t cuts_metis = svmp::test::total_dual_edge_cuts_unique(metis_mesh);

  if (rank == 0) {
    std::cout << "Partition edge cuts: block=" << cuts_block << " metis=" << cuts_metis << "\n";
  }

  ASSERT(cuts_metis < cuts_block);

  MPI_Finalize();
  return 0;
}
