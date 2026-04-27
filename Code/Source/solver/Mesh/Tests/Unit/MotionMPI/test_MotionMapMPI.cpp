/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_MotionMapMPI.cpp
 * @brief MPI tests for physics-agnostic prescribed motion maps.
 */

#include "Mesh.h"
#include "Fields/MeshFields.h"
#include "Motion/MotionFields.h"
#include "Motion/MotionMap.h"

#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace svmp::test {

#define ASSERT_MPI(cond)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

#define ASSERT_NEAR_MPI(a, b, tol) ASSERT_MPI(std::abs((a) - (b)) <= (tol))

static int all_true(MPI_Comm comm, bool local)
{
  const int l = local ? 1 : 0;
  int g = 0;
  MPI_Allreduce(&l, &g, 1, MPI_INT, MPI_MIN, comm);
  return g;
}

static index_t strip_vertex_lid(int x_plane, int y, int z)
{
  return static_cast<index_t>(x_plane * 4 + (y + 2 * z));
}

static void build_hex_strip_global_arrays(int n_cells,
                                         std::vector<real_t>& coords,
                                         std::vector<offset_t>& offsets,
                                         std::vector<index_t>& conn,
                                         std::vector<CellShape>& shapes)
{
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

static Mesh build_partitioned_hex_strip(int world_size)
{
  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;
  build_hex_strip_global_arrays(world_size, coords, offsets, conn, shapes);

  Mesh mesh(MeshComm::world());
  mesh.build_from_arrays_global_and_partition(3,
                                             coords,
                                             offsets,
                                             conn,
                                             shapes,
                                             PartitionHint::Cells,
                                             /*ghost_layers=*/1,
                                             {{"partition_method", "block"}});
  return mesh;
}

static std::array<real_t, 3> rotated_z(const std::array<real_t, 3>& p, real_t angle)
{
  const real_t c = std::cos(angle);
  const real_t s = std::sin(angle);
  return {{c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]}};
}

static void test_owner_target_motion_map_updates_ghosts(MPI_Comm comm, int rank, int size)
{
  auto mesh = build_partitioned_hex_strip(size);
  std::vector<index_t> owner_dofs;
  owner_dofs.reserve(mesh.n_vertices());
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    if (mesh.owner_rank_vertex(v) == rank) {
      owner_dofs.push_back(v);
    }
  }
  ASSERT_MPI(!owner_dofs.empty());

  const bool has_local_ghost = !mesh.ghost_vertices().empty();
  int ghost_count = has_local_ghost ? 1 : 0;
  int global_ghost_count = 0;
  MPI_Allreduce(&ghost_count, &global_ghost_count, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_ghost_count > 0);

  motion::RigidBodyMotionParameters params;
  params.origin = {{0.0, 0.0, 0.0}};
  params.rotation_axis = {{0.0, 0.0, 1.0}};
  params.angular_speed = 2.0;
  motion::RigidBodyMotionMap map(params, "mpi_rigid_rotation");

  motion::MotionMapTimeState time_state;
  time_state.time = 0.25;
  time_state.reference_time = 0.0;
  time_state.dt = 0.25;
  time_state.time_level = motion::MotionMapTimeLevel::AcceptedTimeStep;

  const auto before_rev = mesh.local_mesh().geometry_revision();
  const auto result = motion::apply_motion_map(
      mesh, map, motion::MotionMapTarget::explicit_dofs(owner_dofs, "rotating-strip"), time_state);
  ASSERT_MPI(result.geometry_revision_after > before_rev);

  const auto& local = mesh.local_mesh();
  const int dim = local.dim();
  const auto v_handle = MeshFields::get_field_handle(
      local, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Velocity));
  const auto a_handle = MeshFields::get_field_handle(
      local, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Acceleration));
  ASSERT_MPI(v_handle.id != 0);
  ASSERT_MPI(a_handle.id != 0);
  const real_t* velocity = MeshFields::field_data_as<real_t>(local, v_handle);
  const real_t* acceleration = MeshFields::field_data_as<real_t>(local, a_handle);

  bool local_ok = true;
  const real_t angle = params.angular_speed * time_state.time;
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    if (mesh.owner_rank_vertex(v) != rank && !mesh.is_ghost_vertex(v)) {
      continue;
    }

    const auto ref = local.geometry_dof_coords(v, Configuration::Reference);
    const auto expected_x = rotated_z(ref, angle);
    const auto cur = local.geometry_dof_coords(v, Configuration::Current);
    const size_t base = static_cast<size_t>(v) * static_cast<size_t>(dim);

    const std::array<real_t, 3> expected_v{{-params.angular_speed * expected_x[1],
                                             params.angular_speed * expected_x[0],
                                             0.0}};
    const real_t omega2 = params.angular_speed * params.angular_speed;
    const std::array<real_t, 3> expected_a{{-omega2 * expected_x[0],
                                             -omega2 * expected_x[1],
                                             0.0}};

    for (int d = 0; d < dim; ++d) {
      const size_t k = static_cast<size_t>(d);
      local_ok = local_ok && std::abs(cur[k] - expected_x[k]) <= 1.0e-12;
      local_ok = local_ok && std::abs(velocity[base + k] - expected_v[k]) <= 1.0e-12;
      local_ok = local_ok && std::abs(acceleration[base + k] - expected_a[k]) <= 1.0e-12;
    }
  }

  ASSERT_MPI(all_true(comm, local_ok));
}

} // namespace svmp::test

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size >= 2) {
    svmp::test::test_owner_target_motion_map_updates_ghosts(MPI_COMM_WORLD, rank, size);
  }

  if (rank == 0) {
    std::cout << "MotionMap MPI tests PASSED\n";
  }
  MPI_Finalize();
  return 0;
}
