/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/MeshBase.h"
#include "Fields/MeshFields.h"
#include "IO/MovingMeshRestart.h"
#include "Motion/MotionFields.h"

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace svmp::test {
namespace {

#define ASSERT_MPI(cond)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

int all_true(MPI_Comm comm, bool local)
{
  const int l = local ? 1 : 0;
  int g = 0;
  MPI_Allreduce(&l, &g, 1, MPI_INT, MPI_MIN, comm);
  return g;
}

MeshBase make_rank_local_moved_mesh(int rank)
{
  MeshBase mesh(2);
  const real_t x0 = static_cast<real_t>(rank);
  const std::vector<real_t> x_ref = {
      x0, 0.0,
      x0 + 1.0, 0.0,
  };
  const std::vector<offset_t> offsets = {0, 2};
  const std::vector<index_t> conn = {0, 1};
  const std::vector<CellShape> shapes = {{CellFamily::Line, 2, 1}};
  mesh.build_from_arrays(2, x_ref, offsets, conn, shapes);
  mesh.set_vertex_gids({1000 + 2 * rank, 1000 + 2 * rank + 1});
  mesh.finalize();

  const auto handles = motion::attach_motion_fields(mesh, 2);
  auto* displacement = MeshFields::field_data_as<real_t>(mesh, handles.displacement);
  auto* velocity = MeshFields::field_data_as<real_t>(mesh, handles.velocity);
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    displacement[2 * v + 0] = 0.1 * static_cast<real_t>(rank + 1 + static_cast<int>(v));
    displacement[2 * v + 1] = 0.2 * static_cast<real_t>(rank + 1);
    velocity[2 * v + 0] = 10.0 + static_cast<real_t>(rank);
    velocity[2 * v + 1] = 20.0 + static_cast<real_t>(v);
  }

  std::vector<real_t> x_cur = mesh.X_ref();
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    x_cur[2 * v + 0] += displacement[2 * v + 0];
    x_cur[2 * v + 1] += displacement[2 * v + 1];
  }
  mesh.set_current_coords(x_cur);
  mesh.use_current_configuration();
  return mesh;
}

bool same_real_vector(const std::vector<real_t>& a, const std::vector<real_t>& b)
{
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > 1.0e-12) return false;
  }
  return true;
}

} // namespace

int run_moving_mesh_restart_mpi(MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  auto mesh = make_rank_local_moved_mesh(rank);
  const auto stamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const auto path =
      (std::filesystem::temp_directory_path() /
       ("svmp_moving_mesh_restart_mpi_" + std::to_string(static_cast<long long>(stamp)) +
        "_r" + std::to_string(rank) + ".mmrst"))
          .string();

  moving_mesh_restart::WriteOptions options;
  options.restart_epoch = static_cast<std::uint64_t>(rank + 1);
  options.motion_backend_state.emplace("rank_local_backend", std::to_string(rank));
  moving_mesh_restart::write(mesh, path, options);

  const auto metadata = moving_mesh_restart::inspect(path);
  const auto loaded = moving_mesh_restart::read(path);

  bool ok = true;
  ok = ok && metadata.restart_epoch == static_cast<std::uint64_t>(rank + 1);
  ok = ok && metadata.active_configuration == Configuration::Current;
  ok = ok && metadata.has_current_coordinates;
  ok = ok && loaded.active_configuration() == Configuration::Current;
  ok = ok && loaded.has_current_coords();
  ok = ok && same_real_vector(loaded.X_ref(), mesh.X_ref());
  ok = ok && same_real_vector(loaded.X_cur(), mesh.X_cur());
  ok = ok && loaded.vertex_gids() == mesh.vertex_gids();

  const std::string disp_name =
      motion::standard_motion_field_name(motion::MotionFieldRole::Displacement);
  const auto disp = loaded.field_handle(EntityKind::Vertex, disp_name);
  ok = ok && disp.id != 0;

  std::error_code ec;
  std::filesystem::remove(path, ec);

  ASSERT_MPI(all_true(comm, ok) == 1);
  return 0;
}

} // namespace svmp::test

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rc = 0;
  try {
    rc = svmp::test::run_moving_mesh_restart_mpi(MPI_COMM_WORLD);
  } catch (const std::exception& e) {
    std::cerr << "test_MovingMeshRestartMPI failed: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  if (rc == 0) {
    std::cout << "MovingMeshRestart MPI tests PASSED\n";
  }
  return rc;
}
