/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_MovingAdaptivityMPI.cpp
 * @brief Representative MPI smoke test for moving-mesh adaptivity transfer.
 */

#include "Mesh.h"
#include "Fields/MeshFields.h"
#include "Motion/MotionFields.h"

#ifdef MESH_HAS_ADAPTIVITY
#include "Adaptivity/AdaptivityManager.h"
#endif

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace svmp::test {

#define ASSERT_TRUE_MPI(cond)                                                                      \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

static int all_true(MPI_Comm comm, bool local)
{
  const int l = local ? 1 : 0;
  int g = 0;
  MPI_Allreduce(&l, &g, 1, MPI_INT, MPI_MIN, comm);
  return g;
}

#ifdef MESH_HAS_ADAPTIVITY
static MeshBase make_rank_local_line_mesh()
{
  MeshBase mesh(2);
  const std::vector<real_t> coords = {
      0.0, 0.0,
      1.0, 0.0,
  };
  const std::vector<offset_t> offsets = {0, 2};
  const std::vector<index_t> conn = {0, 1};
  const std::vector<CellShape> shapes = {{CellFamily::Line, 2, 1}};
  mesh.build_from_arrays(2, coords, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

static AdaptivityOptions moving_refine_options()
{
  AdaptivityOptions options;
  options.max_refinement_level = 1;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES;
  options.check_quality = false;
  options.enforce_quality_after_refinement = false;
  options.verbosity = 0;
  return options;
}

static bool run_rank_local_moving_adaptivity(int rank)
{
  auto mesh = make_rank_local_line_mesh();
  const auto handles = motion::attach_motion_fields(mesh, 2);

  auto* displacement = MeshFields::field_data_as<real_t>(mesh, handles.displacement);
  auto* velocity = MeshFields::field_data_as<real_t>(mesh, handles.velocity);
  if (displacement == nullptr || velocity == nullptr) {
    return false;
  }

  displacement[0] = static_cast<real_t>(rank);
  displacement[1] = 0.0;
  displacement[2] = static_cast<real_t>(rank + 2);
  displacement[3] = 0.0;
  velocity[0] = static_cast<real_t>(10 + rank);
  velocity[1] = 0.0;
  velocity[2] = static_cast<real_t>(20 + rank);
  velocity[3] = 0.0;

  mesh.set_current_coords({0.0, 0.0, 2.0, 0.0});
  mesh.use_current_configuration();

  AdaptivityManager manager(moving_refine_options());
  auto result = manager.refine(mesh, {true}, nullptr);
  if (!result.success || mesh.n_vertices() != 3u || !mesh.has_current_coords()) {
    return false;
  }

  std::size_t midpoint = mesh.n_vertices();
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    if (std::abs(mesh.X_ref()[2 * v] - 0.5) < 1.0e-12) {
      midpoint = v;
      break;
    }
  }
  if (midpoint == mesh.n_vertices()) {
    return false;
  }

  const auto disp_handle =
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Displacement));
  const auto vel_handle =
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Velocity));
  const auto* new_displacement = MeshFields::field_data_as<const real_t>(mesh, disp_handle);
  const auto* new_velocity = MeshFields::field_data_as<const real_t>(mesh, vel_handle);
  if (new_displacement == nullptr || new_velocity == nullptr) {
    return false;
  }

  return std::abs(mesh.X_cur()[2 * midpoint] - 1.0) < 1.0e-12 &&
         std::abs(new_displacement[2 * midpoint] - static_cast<real_t>(rank + 1)) < 1.0e-12 &&
         std::abs(new_velocity[2 * midpoint] - static_cast<real_t>(15 + rank)) < 1.0e-12 &&
         result.transfer_stats.motion_values_transferred.count("mesh_displacement") > 0u &&
         result.transfer_stats.motion_values_transferred.count("mesh_velocity") > 0u;
}
#endif

} // namespace svmp::test

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef MESH_HAS_ADAPTIVITY
  const bool local_ok = svmp::test::run_rank_local_moving_adaptivity(rank);
  ASSERT_TRUE_MPI(svmp::test::all_true(MPI_COMM_WORLD, local_ok) == 1);
  if (rank == 0) {
    std::cout << "Moving adaptivity MPI tests PASSED\n";
  }
#else
  if (rank == 0) {
    std::cout << "Moving adaptivity MPI tests SKIPPED (MESH_HAS_ADAPTIVITY not enabled)\n";
  }
#endif

  MPI_Finalize();
  return 0;
}
