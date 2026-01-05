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

#ifndef SVMP_PARTITION_TOPOLOGY_H
#define SVMP_PARTITION_TOPOLOGY_H

#include "../Core/MeshTypes.h"

#include <vector>

namespace svmp {

class DistributedMesh;

/**
 * @brief Neighbor classification for partition-aware adjacency.
 *
 * - internal_neighbors: neighbors owned by this rank
 * - boundary_neighbors: neighbors owned by other ranks (typically present as ghosts)
 */
struct PartitionedAdjacency {
  std::vector<index_t> internal_neighbors;
  std::vector<index_t> boundary_neighbors;
};

/**
 * @brief Partition-aware topology helpers (no MPI communication).
 *
 * These routines use ownership/owner-rank metadata already stored in
 * `DistributedMesh` to classify local topology with respect to partition
 * boundaries.
 *
 * Notes:
 * - These are rank-local computations; they do not communicate.
 * - For robust partition-interface detection without a ghost layer, callers
 *   should ensure `DistributedMesh::build_exchange_patterns()` was called so
 *   shared-face metadata is available.
 */
class PartitionTopology {
public:
  /**
   * @brief Classify cell neighbors as internal vs partition-boundary.
   *
   * Boundary neighbors are those whose owning rank differs from `mesh.rank()`.
   */
  static PartitionedAdjacency classify_cell_neighbors(const DistributedMesh& mesh, index_t cell);

  /**
   * @brief Check whether a face lies on a partition boundary.
   *
   * A face is considered a partition-boundary face if it separates cells owned
   * by different ranks. When a ghost layer is not present, shared-face metadata
   * from exchange patterns may be used to identify partition interfaces even if
   * only one incident cell is present locally.
   */
  static bool is_partition_boundary_face(const DistributedMesh& mesh, index_t face);

  /**
   * @brief Collect partition-boundary faces on this rank.
   *
   * When owned_only=true, applies a deterministic tie-break (by owner rank)
   * to return each partition-interface face exactly once across ranks.
   */
  static std::vector<index_t> partition_boundary_faces(const DistributedMesh& mesh, bool owned_only = true);

  /**
   * @brief Check whether a cell lies on a partition boundary.
   *
   * A cell is considered a partition-boundary cell if it has at least one
   * partition-boundary neighbor.
   */
  static bool is_partition_boundary_cell(const DistributedMesh& mesh, index_t cell);
};

} // namespace svmp

#endif // SVMP_PARTITION_TOPOLOGY_H
