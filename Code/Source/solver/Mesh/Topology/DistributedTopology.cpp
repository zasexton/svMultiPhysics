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

#include "DistributedTopology.h"

#include "../Core/DistributedMesh.h"
#include "PartitionTopology.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>

namespace svmp {
namespace {

std::vector<index_t> all_cells_local(const DistributedMesh& mesh) {
  const size_t n = mesh.n_cells();
  std::vector<index_t> out;
  out.reserve(n);
  for (index_t c = 0; c < static_cast<index_t>(n); ++c) {
    out.push_back(c);
  }
  return out;
}

void require_partition_graph_visibility(const DistributedMesh& mesh, const char* caller) {
  if (mesh.world_size() <= 1) {
    return;
  }
  if (mesh.n_ghost_cells() != 0) {
    return;
  }

  const auto iface = PartitionTopology::partition_boundary_faces(mesh, /*owned_only=*/false);
  if (!iface.empty()) {
    throw std::runtime_error(
        std::string("DistributedTopology::") + caller +
        ": partition interfaces detected but no ghost cells present. "
        "Build a ghost layer (e.g. mesh.build_ghost_layer(1)) before running distributed graph algorithms.");
  }
}

uint64_t hash_u64(uint64_t x) {
  // splitmix64
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

#ifdef MESH_HAS_MPI
template <typename T>
MPI_Datatype mpi_datatype();

template <>
MPI_Datatype mpi_datatype<gid_t>() {
#ifdef MPI_INT64_T
  return MPI_INT64_T;
#else
  return MPI_LONG_LONG;
#endif
}

template <>
MPI_Datatype mpi_datatype<int>() {
  return MPI_INT;
}

template <typename T>
void exchange_cell_values(const DistributedMesh& mesh, std::vector<T>& values, int tag) {
  if (mesh.world_size() <= 1) {
    return;
  }
  const auto& pat = mesh.cell_exchange_pattern();
  if (pat.send_ranks.empty() && pat.recv_ranks.empty()) {
    return;
  }

  const auto n_cells = mesh.n_cells();
  if (values.size() != n_cells) {
    throw std::runtime_error("exchange_cell_values: input size mismatch with mesh.n_cells()");
  }

  MPI_Comm comm = mesh.mpi_comm();
  if (comm == MPI_COMM_NULL) {
    return;
  }

  const MPI_Datatype dtype = mpi_datatype<T>();

  std::vector<std::vector<T>> send_buffers(pat.send_ranks.size());
  std::vector<std::vector<T>> recv_buffers(pat.recv_ranks.size());

  for (size_t i = 0; i < pat.send_ranks.size(); ++i) {
    const auto& send_list = pat.send_lists[i];
    send_buffers[i].resize(send_list.size());
    for (size_t j = 0; j < send_list.size(); ++j) {
      const index_t c = send_list[j];
      if (c < 0 || static_cast<size_t>(c) >= n_cells) {
        send_buffers[i][j] = T{};
      } else {
        send_buffers[i][j] = values[static_cast<size_t>(c)];
      }
    }
  }

  for (size_t i = 0; i < pat.recv_ranks.size(); ++i) {
    recv_buffers[i].resize(pat.recv_lists[i].size());
  }

  std::vector<MPI_Request> reqs;
  reqs.reserve(pat.send_ranks.size() + pat.recv_ranks.size());

  for (size_t i = 0; i < pat.send_ranks.size(); ++i) {
    MPI_Request req;
    const auto count = send_buffers[i].size();
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("exchange_cell_values: send buffer too large for MPI");
    }
    MPI_Isend(send_buffers[i].data(),
              static_cast<int>(count),
              dtype,
              pat.send_ranks[i],
              tag,
              comm,
              &req);
    reqs.push_back(req);
  }

  for (size_t i = 0; i < pat.recv_ranks.size(); ++i) {
    MPI_Request req;
    const auto count = recv_buffers[i].size();
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("exchange_cell_values: recv buffer too large for MPI");
    }
    MPI_Irecv(recv_buffers[i].data(),
              static_cast<int>(count),
              dtype,
              pat.recv_ranks[i],
              tag,
              comm,
              &req);
    reqs.push_back(req);
  }

  if (!reqs.empty()) {
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  }

  for (size_t i = 0; i < pat.recv_ranks.size(); ++i) {
    const auto& recv_list = pat.recv_lists[i];
    for (size_t j = 0; j < recv_list.size(); ++j) {
      const index_t c = recv_list[j];
      if (c < 0 || static_cast<size_t>(c) >= n_cells) {
        continue;
      }
      values[static_cast<size_t>(c)] = recv_buffers[i][j];
    }
  }
}
#endif

} // namespace

DistributedDualGraph DistributedTopology::build_global_dual_graph(const DistributedMesh& mesh, bool owned_only) {
  DistributedDualGraph g;

  const auto& cell_gids = mesh.cell_gids();
  g.local_cells = owned_only ? mesh.owned_cells() : all_cells_local(mesh);
  g.cell_gids.reserve(g.local_cells.size());
  g.offsets.resize(g.local_cells.size() + 1, 0);

  std::vector<gid_t> adj;
  adj.reserve(g.local_cells.size() * 8);

  require_partition_graph_visibility(mesh, "build_global_dual_graph");

  for (size_t i = 0; i < g.local_cells.size(); ++i) {
    const index_t c = g.local_cells[i];
    if (c < 0 || static_cast<size_t>(c) >= cell_gids.size()) {
      throw std::runtime_error("DistributedTopology::build_global_dual_graph: invalid local cell index");
    }

    g.cell_gids.push_back(cell_gids[static_cast<size_t>(c)]);

    auto neighbors_local = mesh.cell_neighbors(c);
    std::vector<gid_t> row;
    row.reserve(neighbors_local.size());

    for (const auto n : neighbors_local) {
      if (n == c) {
        continue;
      }
      if (n < 0 || static_cast<size_t>(n) >= cell_gids.size()) {
        continue;
      }
      row.push_back(cell_gids[static_cast<size_t>(n)]);
    }

    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());

    adj.insert(adj.end(), row.begin(), row.end());
    g.offsets[i + 1] = static_cast<offset_t>(adj.size());
  }

  g.neighbors = std::move(adj);
  return g;
}

std::vector<gid_t> DistributedTopology::connected_components_global(const DistributedMesh& mesh) {
  const auto& cell_gids = mesh.cell_gids();
  const size_t n_cells = mesh.n_cells();
  std::vector<gid_t> labels(cell_gids.begin(), cell_gids.end());

  if (n_cells == 0) {
    return labels;
  }

  require_partition_graph_visibility(mesh, "connected_components_global");

  if (mesh.world_size() <= 1) {
    std::vector<uint8_t> visited(n_cells, 0);
    std::queue<index_t> q;
    std::vector<index_t> component;

    for (index_t seed = 0; seed < static_cast<index_t>(n_cells); ++seed) {
      if (visited[static_cast<size_t>(seed)] != 0) {
        continue;
      }
      component.clear();
      gid_t min_gid = cell_gids[static_cast<size_t>(seed)];

      visited[static_cast<size_t>(seed)] = 1;
      q.push(seed);

      while (!q.empty()) {
        const index_t c = q.front();
        q.pop();
        component.push_back(c);

        min_gid = std::min(min_gid, cell_gids[static_cast<size_t>(c)]);

        const auto neigh = mesh.cell_neighbors(c);
        for (const auto n : neigh) {
          if (n < 0 || static_cast<size_t>(n) >= n_cells) {
            continue;
          }
          if (visited[static_cast<size_t>(n)] != 0) {
            continue;
          }
          visited[static_cast<size_t>(n)] = 1;
          q.push(n);
        }
      }

      for (const auto c : component) {
        labels[static_cast<size_t>(c)] = min_gid;
      }
    }

    return labels;
  }

#ifdef MESH_HAS_MPI
  const auto owned = mesh.owned_cells();
  MPI_Comm comm = mesh.mpi_comm();

  const size_t max_iters = std::max<size_t>(1, mesh.global_n_cells());
  for (size_t iter = 0; iter < max_iters; ++iter) {
    exchange_cell_values(mesh, labels, /*tag=*/1600);

    bool local_changed = false;
    for (const auto c : owned) {
      if (c < 0 || static_cast<size_t>(c) >= n_cells) {
        continue;
      }
      gid_t min_label = labels[static_cast<size_t>(c)];
      const auto neigh = mesh.cell_neighbors(c);
      for (const auto n : neigh) {
        if (n < 0 || static_cast<size_t>(n) >= n_cells) {
          continue;
        }
        min_label = std::min(min_label, labels[static_cast<size_t>(n)]);
      }
      if (min_label < labels[static_cast<size_t>(c)]) {
        labels[static_cast<size_t>(c)] = min_label;
        local_changed = true;
      }
    }

    int local_flag = local_changed ? 1 : 0;
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, comm);
    if (global_flag == 0) {
      break;
    }
    if (iter + 1 == max_iters) {
      throw std::runtime_error(
          "DistributedTopology::connected_components_global: failed to converge within global_n_cells() iterations");
    }
  }

  exchange_cell_values(mesh, labels, /*tag=*/1600);
  return labels;
#else
  return labels;
#endif
}

std::vector<int> DistributedTopology::parallel_graph_coloring(const DistributedMesh& mesh) {
  const auto& cell_gids = mesh.cell_gids();
  const size_t n_cells = mesh.n_cells();
  std::vector<int> color(n_cells, -1);

  if (n_cells == 0) {
    return color;
  }

  require_partition_graph_visibility(mesh, "parallel_graph_coloring");

  if (mesh.world_size() <= 1) {
    // Serial greedy coloring in deterministic GID order.
    std::vector<index_t> order;
    order.reserve(n_cells);
    for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
      order.push_back(c);
    }
    std::sort(order.begin(), order.end(),
              [&](index_t a, index_t b) { return cell_gids[static_cast<size_t>(a)] < cell_gids[static_cast<size_t>(b)]; });

    for (const auto c : order) {
      const auto neigh = mesh.cell_neighbors(c);
      std::vector<uint8_t> used(neigh.size() + 2, 0);
      for (const auto n : neigh) {
        if (n < 0 || static_cast<size_t>(n) >= n_cells) continue;
        const int nc = color[static_cast<size_t>(n)];
        if (nc >= 0) {
          if (static_cast<size_t>(nc) >= used.size()) used.resize(static_cast<size_t>(nc) + 1, 0);
          used[static_cast<size_t>(nc)] = 1;
        }
      }
      int chosen = 0;
      while (static_cast<size_t>(chosen) < used.size() && used[static_cast<size_t>(chosen)] != 0) {
        ++chosen;
      }
      color[static_cast<size_t>(c)] = chosen;
    }

    return color;
  }

#ifdef MESH_HAS_MPI
  MPI_Comm comm = mesh.mpi_comm();
  const auto owned = mesh.owned_cells();

  std::vector<uint64_t> weight(n_cells, 0);
  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    weight[static_cast<size_t>(c)] = hash_u64(static_cast<uint64_t>(cell_gids[static_cast<size_t>(c)]));
  }

  const size_t max_iters = std::max<size_t>(1, mesh.global_n_cells());
  for (size_t iter = 0; iter < max_iters; ++iter) {
    exchange_cell_values(mesh, color, /*tag=*/1601);

    bool local_progress = false;

    for (const auto c : owned) {
      if (c < 0 || static_cast<size_t>(c) >= n_cells) {
        continue;
      }
      if (color[static_cast<size_t>(c)] >= 0) {
        continue;
      }

      const auto neigh = mesh.cell_neighbors(c);
      const auto my_key = std::pair<uint64_t, gid_t>{weight[static_cast<size_t>(c)], cell_gids[static_cast<size_t>(c)]};

      bool is_max = true;
      for (const auto n : neigh) {
        if (n < 0 || static_cast<size_t>(n) >= n_cells) continue;
        if (color[static_cast<size_t>(n)] >= 0) continue;

        const auto nb_key = std::pair<uint64_t, gid_t>{weight[static_cast<size_t>(n)], cell_gids[static_cast<size_t>(n)]};
        if (nb_key > my_key) {
          is_max = false;
          break;
        }
      }
      if (!is_max) {
        continue;
      }

      std::vector<uint8_t> used(neigh.size() + 2, 0);
      for (const auto n : neigh) {
        if (n < 0 || static_cast<size_t>(n) >= n_cells) continue;
        const int nc = color[static_cast<size_t>(n)];
        if (nc >= 0) {
          if (static_cast<size_t>(nc) >= used.size()) used.resize(static_cast<size_t>(nc) + 1, 0);
          used[static_cast<size_t>(nc)] = 1;
        }
      }
      int chosen = 0;
      while (static_cast<size_t>(chosen) < used.size() && used[static_cast<size_t>(chosen)] != 0) {
        ++chosen;
      }
      color[static_cast<size_t>(c)] = chosen;
      local_progress = true;
    }

    int local_uncolored = 0;
    for (const auto c : owned) {
      if (c >= 0 && static_cast<size_t>(c) < n_cells && color[static_cast<size_t>(c)] < 0) {
        local_uncolored = 1;
        break;
      }
    }

    int any_uncolored = 0;
    MPI_Allreduce(&local_uncolored, &any_uncolored, 1, MPI_INT, MPI_LOR, comm);
    if (any_uncolored == 0) {
      break;
    }

    int local_prog = local_progress ? 1 : 0;
    int any_prog = 0;
    MPI_Allreduce(&local_prog, &any_prog, 1, MPI_INT, MPI_LOR, comm);
    if (any_prog == 0) {
      throw std::runtime_error("DistributedTopology::parallel_graph_coloring: stalled; no progress");
    }

    if (iter + 1 == max_iters) {
      throw std::runtime_error(
          "DistributedTopology::parallel_graph_coloring: failed to color within global_n_cells() iterations");
    }
  }

  exchange_cell_values(mesh, color, /*tag=*/1601);
  return color;
#else
  return color;
#endif
}

std::vector<index_t> DistributedTopology::global_boundary_faces(const DistributedMesh& mesh, bool owned_only) {
  std::vector<index_t> out;
  const size_t n_faces = mesh.n_faces();
  if (n_faces == 0) {
    return out;
  }

  out.reserve(n_faces / 4);

  for (index_t f = 0; f < static_cast<index_t>(n_faces); ++f) {
    if (mesh.is_ghost_face(f)) {
      continue;
    }
    const auto cells = mesh.face_cells(f);
    const bool is_boundary = (cells[0] == INVALID_INDEX) || (cells[1] == INVALID_INDEX);
    if (!is_boundary) {
      continue;
    }
    const index_t incident = (cells[0] != INVALID_INDEX) ? cells[0] : cells[1];
    if (incident == INVALID_INDEX) {
      continue;
    }
    const rank_t face_owner = mesh.owner_rank_cell(incident);
    if (owned_only && mesh.rank() != face_owner) {
      continue;
    }
    if (PartitionTopology::is_partition_boundary_face(mesh, f)) {
      continue;
    }
    out.push_back(f);
  }

  return out;
}

} // namespace svmp
