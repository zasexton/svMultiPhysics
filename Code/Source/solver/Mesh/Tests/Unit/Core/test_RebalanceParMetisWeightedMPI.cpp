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
 * @file test_RebalanceParMetisWeightedMPI.cpp
 * @brief MPI test validating that ParMETIS respects user-provided cell weights.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>
#include <algorithm>
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

static void build_hex_chain_global(
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

static std::uint64_t owned_cell_weight_sum(const DistributedMesh& mesh, const FieldHandle& h) {
  const auto* data = mesh.local_mesh().field_data_as<std::int32_t>(h);
  ASSERT(data != nullptr);

  std::uint64_t sum = 0;
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    if (!mesh.is_owned_cell(c)) {
      continue;
    }
    sum += static_cast<std::uint64_t>(std::max<std::int32_t>(1, data[static_cast<size_t>(c)]));
  }
  return sum;
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
  svmp::test::build_hex_chain_global(n_cells, rank, X_ref, cell2v_offsets, cell2v, shapes);

  svmp::DistributedMesh mesh(MPI_COMM_WORLD);
  std::unordered_map<std::string, std::string> opts;
  opts["partition_method"] = "block";
  mesh.build_from_arrays_global_and_partition(3,
                                              X_ref,
                                              cell2v_offsets,
                                              cell2v,
                                              shapes,
                                              svmp::PartitionHint::Cells,
                                              /*ghost_layers=*/1,
                                              opts);

  // Attach a scalar cell weight field: first half of cells are heavy.
  auto h = mesh.local_mesh().attach_field(svmp::EntityKind::Volume, "cell_w", svmp::FieldScalarType::Int32, 1);
  ASSERT(h.id != 0);
  auto* w = mesh.local_mesh().field_data_as<std::int32_t>(h);
  ASSERT(w != nullptr);

  const auto& gids = mesh.local_mesh().cell_gids();
  ASSERT(gids.size() == mesh.n_cells());
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
    const gid_t gid = gids[static_cast<size_t>(c)];
    w[static_cast<size_t>(c)] = (gid >= 0 && gid < n_cells / 2) ? 10 : 1;
  }

  const std::uint64_t local_before = svmp::test::owned_cell_weight_sum(mesh, h);
  std::uint64_t max_before = 0;
  MPI_Allreduce(&local_before, &max_before, 1, MPI_UINT64_T, MPI_MAX, mesh.mpi_comm());

  // Rebalance using ParMETIS with explicit weights.
  mesh.rebalance(svmp::PartitionHint::Cells,
                 {{"partition_method", "parmetis"},
                  {"cell_weight_field", "cell_w"}});

  const auto h_after = mesh.local_mesh().field_handle(svmp::EntityKind::Volume, "cell_w");
  ASSERT(h_after.id != 0);
  const std::uint64_t local_after = svmp::test::owned_cell_weight_sum(mesh, h_after);
  std::uint64_t max_after = 0;
  MPI_Allreduce(&local_after, &max_after, 1, MPI_UINT64_T, MPI_MAX, mesh.mpi_comm());

  if (rank == 0) {
    std::cout << "Weighted ParMETIS max owned weight: before=" << max_before << " after=" << max_after << "\n";
  }

  ASSERT(max_after < max_before);

  MPI_Finalize();
  return 0;
}
