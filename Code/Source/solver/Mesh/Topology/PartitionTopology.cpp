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

#include "PartitionTopology.h"

#include "../Core/DistributedMesh.h"

#include <algorithm>
#include <cstddef>

namespace svmp {
namespace {

bool list_contains(const std::vector<index_t>& list, index_t value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

bool face_in_exchange_pattern(const DistributedMesh& mesh, index_t face) {
  const auto& pat = mesh.face_exchange_pattern();
  for (const auto& list : pat.send_lists) {
    if (list_contains(list, face)) {
      return true;
    }
  }
  for (const auto& list : pat.recv_lists) {
    if (list_contains(list, face)) {
      return true;
    }
  }
  return false;
}

std::vector<uint8_t> build_face_exchange_mask(const DistributedMesh& mesh) {
  const size_t n_faces = mesh.n_faces();
  std::vector<uint8_t> mask(n_faces, 0);
  const auto& pat = mesh.face_exchange_pattern();
  for (const auto& list : pat.send_lists) {
    for (index_t f : list) {
      if (f >= 0 && static_cast<size_t>(f) < n_faces) {
        mask[static_cast<size_t>(f)] = 1;
      }
    }
  }
  for (const auto& list : pat.recv_lists) {
    for (index_t f : list) {
      if (f >= 0 && static_cast<size_t>(f) < n_faces) {
        mask[static_cast<size_t>(f)] = 1;
      }
    }
  }
  return mask;
}

} // namespace

PartitionedAdjacency PartitionTopology::classify_cell_neighbors(const DistributedMesh& mesh, index_t cell) {
  PartitionedAdjacency out;
  const auto neighbors = mesh.cell_neighbors(cell);
  if (mesh.world_size() <= 1) {
    out.internal_neighbors = neighbors;
    return out;
  }

  const rank_t my_rank = mesh.rank();
  out.internal_neighbors.reserve(neighbors.size());
  out.boundary_neighbors.reserve(neighbors.size());

  for (const auto nbr : neighbors) {
    if (mesh.owner_rank_cell(nbr) == my_rank) {
      out.internal_neighbors.push_back(nbr);
    } else {
      out.boundary_neighbors.push_back(nbr);
    }
  }

  return out;
}

bool PartitionTopology::is_partition_boundary_face(const DistributedMesh& mesh, index_t face) {
  if (mesh.world_size() <= 1) {
    return false;
  }
  if (mesh.is_ghost_face(face)) {
    return false;
  }

  const auto cells = mesh.face_cells(face);
  const index_t c0 = cells[0];
  const index_t c1 = cells[1];

  if (c0 >= 0 && c1 >= 0) {
    return mesh.owner_rank_cell(c0) != mesh.owner_rank_cell(c1);
  }

  // Without a ghost layer, a shared interface face may have only one incident
  // cell locally. When ghosts are present, boundary faces may still be exchanged
  // (to populate ghost-face data) and should not be classified as partition interfaces.
  if (mesh.n_ghost_cells() != 0) {
    return false;
  }

  // No ghosts: use shared/owner metadata and (if available) exchange-pattern
  // involvement to detect partition interfaces even if only one incident cell is present.
  return mesh.is_shared_face(face) ||
         (mesh.owner_rank_face(face) != mesh.rank()) ||
         face_in_exchange_pattern(mesh, face);
}

std::vector<index_t> PartitionTopology::partition_boundary_faces(const DistributedMesh& mesh, bool owned_only) {
  std::vector<index_t> out;
  if (mesh.world_size() <= 1) {
    return out;
  }

  const size_t n_faces = mesh.n_faces();
  out.reserve(n_faces / 8);

  const bool allow_boundary_face_heuristic = (mesh.n_ghost_cells() == 0);
  const auto exchange_mask = allow_boundary_face_heuristic ? build_face_exchange_mask(mesh)
                                                           : std::vector<uint8_t>{};

  for (index_t f = 0; f < static_cast<index_t>(n_faces); ++f) {
    if (mesh.is_ghost_face(f)) {
      continue;
    }

    const auto cells = mesh.face_cells(f);
    const index_t c0 = cells[0];
    const index_t c1 = cells[1];

    bool is_partition_boundary = false;
    rank_t face_owner = mesh.rank();
    if (c0 >= 0 && c1 >= 0) {
      const rank_t r0 = mesh.owner_rank_cell(c0);
      const rank_t r1 = mesh.owner_rank_cell(c1);
      is_partition_boundary = (r0 != r1);
      face_owner = std::min(r0, r1);
    } else if (allow_boundary_face_heuristic) {
      const bool exchanged = exchange_mask.empty() ? false : (exchange_mask[static_cast<size_t>(f)] != 0);
      is_partition_boundary = exchanged || mesh.is_shared_face(f) || (mesh.owner_rank_face(f) != mesh.rank());
      face_owner = mesh.owner_rank_face(f);
    }

    if (is_partition_boundary) {
      if (owned_only && mesh.rank() != face_owner) {
        continue;
      }
      out.push_back(f);
    }
  }

  return out;
}

bool PartitionTopology::is_partition_boundary_cell(const DistributedMesh& mesh, index_t cell) {
  if (mesh.world_size() <= 1) {
    return false;
  }
  const auto adj = classify_cell_neighbors(mesh, cell);
  return !adj.boundary_neighbors.empty();
}

} // namespace svmp
