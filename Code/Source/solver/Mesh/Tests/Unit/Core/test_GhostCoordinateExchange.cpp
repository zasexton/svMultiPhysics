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
 * @file test_GhostCoordinateExchange.cpp
 * @brief MPI test validating ghost coordinate exchange + GeometryChanged events.
 *
 * This covers Phase 4 partition/ghost lifecycle behavior:
 * - build_from_arrays_global_and_partition(..., ghost_layers=1) produces ghosts
 * - update_exchange_ghost_coordinates() updates shared/ghost vertex coordinates
 * - GeometryChanged event is emitted
 */

#include "Mesh.h"
#include "Observer/ObserverRegistry.h"
#include "Topology/CellTopology.h"

#include <mpi.h>

#include <array>
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

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) <= (tol))

static index_t strip_vertex_lid(int x_plane, int y, int z) {
  return static_cast<index_t>(x_plane * 4 + (y + 2 * z));
}

static index_t hex27_vertex_lid(int x_half, int y_half, int z_half) {
  return static_cast<index_t>(x_half * 9 + z_half * 3 + y_half);
}

static int all_true(MPI_Comm comm, bool local) {
  const int l = local ? 1 : 0;
  int g = 0;
  MPI_Allreduce(&l, &g, 1, MPI_INT, MPI_MIN, comm);
  return g;
}

static void build_hex_strip_global_arrays(int n_cells,
                                         std::vector<real_t>& coords,
                                         std::vector<offset_t>& offsets,
                                         std::vector<index_t>& conn,
                                         std::vector<CellShape>& shapes) {
  const int n_planes = n_cells + 1;
  const int n_vertices = 4 * n_planes;

  coords.clear();
  coords.reserve(static_cast<size_t>(n_vertices) * 3u);

  for (int x_plane = 0; x_plane < n_planes; ++x_plane) {
    for (int z = 0; z <= 1; ++z) {
      for (int y = 0; y <= 1; ++y) {
        coords.push_back(static_cast<real_t>(x_plane));
        coords.push_back(static_cast<real_t>(y));
        coords.push_back(static_cast<real_t>(z));
      }
    }
  }

  offsets.assign(static_cast<size_t>(n_cells) + 1u, 0);
  conn.clear();
  conn.reserve(static_cast<size_t>(n_cells) * 8u);
  shapes.assign(static_cast<size_t>(n_cells), CellShape{CellFamily::Hex, 8, 1});

  offsets[0] = 0;
  for (int c = 0; c < n_cells; ++c) {
    const int x0 = c;
    const int x1 = c + 1;

    conn.push_back(strip_vertex_lid(x0, 0, 0));
    conn.push_back(strip_vertex_lid(x1, 0, 0));
    conn.push_back(strip_vertex_lid(x1, 1, 0));
    conn.push_back(strip_vertex_lid(x0, 1, 0));
    conn.push_back(strip_vertex_lid(x0, 0, 1));
    conn.push_back(strip_vertex_lid(x1, 0, 1));
    conn.push_back(strip_vertex_lid(x1, 1, 1));
    conn.push_back(strip_vertex_lid(x0, 1, 1));

    offsets[static_cast<size_t>(c) + 1u] = static_cast<offset_t>(conn.size());
  }
}

static void build_hex27_strip_global_arrays(int n_cells,
                                           std::vector<real_t>& coords,
                                           std::vector<offset_t>& offsets,
                                           std::vector<index_t>& conn,
                                           std::vector<CellShape>& shapes) {
  const int n_x = 2 * n_cells + 1;
  const int n_vertices = n_x * 9;

  coords.clear();
  coords.reserve(static_cast<size_t>(n_vertices) * 3u);
  for (int x = 0; x < n_x; ++x) {
    for (int z = 0; z <= 2; ++z) {
      for (int y = 0; y <= 2; ++y) {
        coords.push_back(static_cast<real_t>(x) * static_cast<real_t>(0.5));
        coords.push_back(static_cast<real_t>(y) * static_cast<real_t>(0.5));
        coords.push_back(static_cast<real_t>(z) * static_cast<real_t>(0.5));
      }
    }
  }

  offsets.assign(static_cast<size_t>(n_cells) + 1u, 0);
  conn.clear();
  conn.reserve(static_cast<size_t>(n_cells) * 27u);
  shapes.assign(static_cast<size_t>(n_cells), CellShape{CellFamily::Hex, 8, 2});

  const auto eview = CellTopology::get_edges_view(CellFamily::Hex);
  const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);

  offsets[0] = 0;
  for (int c = 0; c < n_cells; ++c) {
    const int x0 = 2 * c;
    const int x1 = 2 * c + 2;
    const std::array<std::array<int, 3>, 8> corners = {{
        {{x0, 0, 0}},
        {{x1, 0, 0}},
        {{x1, 2, 0}},
        {{x0, 2, 0}},
        {{x0, 0, 2}},
        {{x1, 0, 2}},
        {{x1, 2, 2}},
        {{x0, 2, 2}},
    }};

    auto add_ijk = [&conn](const std::array<int, 3>& ijk) {
      conn.push_back(hex27_vertex_lid(ijk[0], ijk[1], ijk[2]));
    };

    for (const auto& corner : corners) {
      add_ijk(corner);
    }

    for (int e = 0; e < eview.edge_count; ++e) {
      const int a = eview.pairs_flat[2 * e + 0];
      const int b = eview.pairs_flat[2 * e + 1];
      add_ijk({{(corners[static_cast<size_t>(a)][0] + corners[static_cast<size_t>(b)][0]) / 2,
                (corners[static_cast<size_t>(a)][1] + corners[static_cast<size_t>(b)][1]) / 2,
                (corners[static_cast<size_t>(a)][2] + corners[static_cast<size_t>(b)][2]) / 2}});
    }

    for (int f = 0; f < fview.face_count; ++f) {
      std::array<int, 3> sum{{0, 0, 0}};
      const int begin = fview.offsets[f];
      const int end = fview.offsets[f + 1];
      for (int i = begin; i < end; ++i) {
        const auto& corner = corners[static_cast<size_t>(fview.indices[i])];
        sum[0] += corner[0];
        sum[1] += corner[1];
        sum[2] += corner[2];
      }
      const int n = end - begin;
      add_ijk({{sum[0] / n, sum[1] / n, sum[2] / n}});
    }

    add_ijk({{2 * c + 1, 1, 1}});
    offsets[static_cast<size_t>(c) + 1u] = static_cast<offset_t>(conn.size());
  }
}

static void test_exchange_ghost_coordinates_after_global_partition(int rank, int world_size) {
  if (world_size < 2) {
    if (rank == 0) {
      std::cout << "Skipping ghost coordinate exchange test (requires >= 2 ranks)\n";
    }
    return;
  }

  const int n_cells_global = world_size;
  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;
  build_hex_strip_global_arrays(n_cells_global, coords, offsets, conn, shapes);

  Mesh mesh(MeshComm::world());
  mesh.build_from_arrays_global_and_partition(3,
                                             coords,
                                             offsets,
                                             conn,
                                             shapes,
                                             PartitionHint::Cells,
                                             /*ghost_layers=*/1,
                                             {{"partition_method", "block"}});

  ASSERT_EQ(mesh.rank(), rank);
  ASSERT_EQ(mesh.world_size(), world_size);

  const size_t expected_ghost_cells =
      static_cast<size_t>((rank > 0 ? 1 : 0) + (rank + 1 < world_size ? 1 : 0));
  const size_t expected_local_cells = 1u + expected_ghost_cells;
  const size_t expected_local_vertices = 8u + expected_ghost_cells * 4u;

  ASSERT_EQ(mesh.n_cells(), expected_local_cells);
  ASSERT_EQ(mesh.n_vertices(), expected_local_vertices);
  ASSERT_EQ(mesh.n_ghost_cells(), expected_ghost_cells);
  ASSERT_EQ(mesh.n_ghost_vertices(), expected_ghost_cells * 4u);

  ASSERT_EQ(mesh.dim(), 3);
  mesh.set_current_coords(mesh.X_ref());
  real_t* cur = mesh.X_cur_data_mutable();
  ASSERT(cur != nullptr);

  constexpr real_t sentinel = static_cast<real_t>(-12345.0);
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const rank_t owner = mesh.owner_rank_vertex(v);
    for (int d = 0; d < 3; ++d) {
      cur[static_cast<size_t>(v) * 3u + static_cast<size_t>(d)] =
          (owner == rank) ? static_cast<real_t>(owner * 10 + d) : sentinel;
    }
  }

  auto counter = ObserverRegistry::attach_event_counter(mesh.event_bus());
  counter->reset();
  mesh.update_exchange_ghost_coordinates(Configuration::Current);
  ASSERT_EQ(counter->count(MeshEvent::GeometryChanged), 1u);

  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const rank_t owner = mesh.owner_rank_vertex(v);
    for (int d = 0; d < 3; ++d) {
      const real_t expected = static_cast<real_t>(owner * 10 + d);
      const real_t got = cur[static_cast<size_t>(v) * 3u + static_cast<size_t>(d)];
      ASSERT_NEAR(got, expected, 1e-12);
    }
  }
}

static void test_high_order_geometry_dofs_exchange_as_vertices(int rank, int world_size) {
  if (world_size < 2) {
    if (rank == 0) {
      std::cout << "Skipping high-order geometry DOF ghost exchange test (requires >= 2 ranks)\n";
    }
    return;
  }

  const int n_cells_global = world_size;
  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;
  build_hex27_strip_global_arrays(n_cells_global, coords, offsets, conn, shapes);

  Mesh mesh(MeshComm::world());
  mesh.build_from_arrays_global_and_partition(3,
                                             coords,
                                             offsets,
                                             conn,
                                             shapes,
                                             PartitionHint::Cells,
                                             /*ghost_layers=*/1,
                                             {{"partition_method", "block"}});

  const auto descriptor = mesh.geometry_order_descriptor();
  ASSERT_EQ(descriptor.storage, GeometryDofStorage::VertexCoordinates);
  ASSERT_EQ(descriptor.max_order, 2);
  ASSERT(descriptor.has_high_order);
  ASSERT_EQ(mesh.geometry_dof_count(), mesh.n_vertices());

  bool local_saw_high_order_cell = false;
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    if (mesh.geometry_order(c) == 2 && mesh.cell_geometry_dofs(c).size() == 27u) {
      local_saw_high_order_cell = true;
      ASSERT_EQ(mesh.cell_edge_geometry_dofs(c, 0).size(), 3u);
      ASSERT_EQ(mesh.cell_face_geometry_dofs(c, 0).size(), 9u);
      ASSERT_EQ(mesh.cell_interior_geometry_dofs(c).size(), 1u);
    }
  }
  ASSERT(all_true(MPI_COMM_WORLD, local_saw_high_order_cell) == 1);

  mesh.set_current_coords(mesh.X_ref());
  real_t* cur = mesh.X_cur_data_mutable();
  ASSERT(cur != nullptr);

  constexpr real_t sentinel = static_cast<real_t>(-54321.0);
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const rank_t owner = mesh.owner_rank_vertex(v);
    for (int d = 0; d < 3; ++d) {
      cur[static_cast<size_t>(v) * 3u + static_cast<size_t>(d)] =
          (owner == rank) ? static_cast<real_t>(owner * 100 + d) : sentinel;
    }
  }

  auto counter = ObserverRegistry::attach_event_counter(mesh.event_bus());
  counter->reset();
  mesh.update_exchange_ghost_coordinates(Configuration::Current);
  ASSERT_EQ(counter->count(MeshEvent::GeometryChanged), 1u);

  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const rank_t owner = mesh.owner_rank_vertex(v);
    for (int d = 0; d < 3; ++d) {
      const real_t expected = static_cast<real_t>(owner * 100 + d);
      const real_t got = cur[static_cast<size_t>(v) * 3u + static_cast<size_t>(d)];
      ASSERT_NEAR(got, expected, 1e-12);
    }
  }
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  svmp::test::test_exchange_ghost_coordinates_after_global_partition(rank, world_size);
  svmp::test::test_high_order_geometry_dofs_exchange_as_vertices(rank, world_size);

  if (rank == 0) {
    std::cout << "Ghost coordinate exchange tests PASSED\n";
  }

  MPI_Finalize();
  return 0;
}
