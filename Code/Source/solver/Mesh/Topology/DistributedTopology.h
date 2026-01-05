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

#ifndef SVMP_DISTRIBUTED_TOPOLOGY_H
#define SVMP_DISTRIBUTED_TOPOLOGY_H

#include "../Core/MeshTypes.h"

#include <vector>

namespace svmp {

class DistributedMesh;

/**
 * @brief Distributed dual graph in CSR form (local rows, global cell IDs).
 *
 * Row i corresponds to local cell `local_cells[i]` whose global ID is
 * `cell_gids[i]`. Neighbor entries in `neighbors` are global cell IDs.
 */
struct DistributedDualGraph {
  std::vector<index_t> local_cells;
  std::vector<gid_t> cell_gids;
  std::vector<offset_t> offsets;
  std::vector<gid_t> neighbors;
};

/**
 * @brief Distributed topology algorithms.
 *
 * These algorithms operate on `DistributedMesh` and may rely on distributed
 * metadata (ownership, exchange patterns, ghost layers). Implementations are
 * designed as bulk operations (not per-query communication).
 */
class DistributedTopology {
public:
  /**
   * @brief Build the cell dual graph using global cell IDs.
   *
   * For correctness across partitions, this routine assumes that partition
   * interfaces are represented locally (e.g., a 1-deep ghost layer so that
   * cross-rank cell neighbors appear as ghost cells). If shared faces exist
   * but no ghost layer is present, the returned graph will be incomplete.
   */
  static DistributedDualGraph build_global_dual_graph(const DistributedMesh& mesh, bool owned_only = true);

  /**
   * @brief Parallel connected components on the cell dual graph.
   *
   * Returns a per-local-cell component label equal to the minimum global cell
   * ID in that component (stable across ranks). Requires a 1-deep ghost layer
   * for cross-rank connectivity.
   */
  static std::vector<gid_t> connected_components_global(const DistributedMesh& mesh);

  /**
   * @brief Parallel graph coloring for the cell dual graph.
   *
   * Returns per-local-cell colors (owned cells are colored, ghost colors are
   * synchronized from owners). Requires a 1-deep ghost layer for correctness
   * across partitions.
   */
  static std::vector<int> parallel_graph_coloring(const DistributedMesh& mesh);

  /**
   * @brief True domain-boundary faces (excluding partition interfaces).
   *
   * This filters rank-local boundary faces and removes those that are shared
   * across ranks (partition boundaries). When owned_only=true, applies a
   * deterministic tie-break by incident-cell owner rank to avoid duplicates.
   */
  static std::vector<index_t> global_boundary_faces(const DistributedMesh& mesh, bool owned_only = true);
};

} // namespace svmp

#endif // SVMP_DISTRIBUTED_TOPOLOGY_H
