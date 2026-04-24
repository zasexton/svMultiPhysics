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
 * @file test_MotionMPI.cpp
 * @brief MPI unit tests for Mesh/Motion (MeshMotion + ghost sync + failure propagation).
 *
 * These tests are intentionally standalone (no gtest) to match the existing
 * Mesh MPI test style under Unit/Core.
 */

#include "Mesh.h"
#include "Fields/MeshFields.h"
#include "Motion/IMotionBackend.h"
#include "Motion/MeshMotion.h"
#include "Motion/MotionFields.h"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
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

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) <= (tol))

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
  return mesh;
}

class FailsOnRankBackend final : public motion::IMotionBackend {
public:
  explicit FailsOnRankBackend(rank_t fail_rank) : fail_rank_(fail_rank) {}
  const char* name() const noexcept override { return "FailsOnRankBackend"; }

  motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override
  {
    motion::MotionSolveResult result{};

    if (request.mesh.rank() == fail_rank_) {
      result.success = false;
      result.message = "intentional MPI failure";
      return result;
    }

    if (request.displacement.valid()) {
      const size_t n = request.displacement.n_entities;
      const size_t c = request.displacement.components;
      for (size_t v = 0; v < n; ++v) {
        request.displacement.data[v * c + 0] = 0.0;
        if (c > 1) request.displacement.data[v * c + 1] = 0.0;
        if (c > 2) request.displacement.data[v * c + 2] = 0.0;
      }
    }

    result.success = true;
    result.wrote_velocity = false;
    return result;
  }

private:
  rank_t fail_rank_{0};
};

class OwnedOnlyDisplacementBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "OwnedOnlyDisplacementBackend"; }

  motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override
  {
    motion::MotionSolveResult result{};
    if (!request.displacement.valid()) {
      result.success = false;
      result.message = "displacement view invalid";
      return result;
    }

    const size_t n = request.displacement.n_entities;
    const size_t c = request.displacement.components;

    const real_t a = static_cast<real_t>(request.mesh.rank() + 1);

    for (size_t v = 0; v < n; ++v) {
      if (!request.mesh.is_owned_vertex(static_cast<index_t>(v))) {
        continue; // leave as-is (MeshMotion zeroes before solve)
      }
      request.displacement.data[v * c + 0] = a;
      if (c > 1) request.displacement.data[v * c + 1] = 0.0;
      if (c > 2) request.displacement.data[v * c + 2] = 0.0;
    }

    result.success = true;
    result.wrote_velocity = false;
    return result;
  }
};

static void test_failure_propagates_across_ranks(int rank, int world_size)
{
  if (world_size < 2) {
    if (rank == 0) {
      std::cout << "Skipping MeshMotion MPI failure test (requires >= 2 ranks)\n";
    }
    return;
  }

  auto mesh = build_partitioned_hex_strip(world_size);
  ASSERT_EQ(mesh.rank(), rank);
  ASSERT_EQ(mesh.world_size(), world_size);

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<FailsOnRankBackend>(/*fail_rank=*/0));

  const bool local_ok = mm.advance(/*dt=*/0.25);

  ASSERT_EQ(all_true(mesh.mpi_comm(), local_ok), 0);
  ASSERT(!local_ok);

  // Entry state had no current coords; failure must restore that state.
  ASSERT(!mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), Configuration::Reference);

  // Motion fields should exist and be deterministically zeroed.
  const auto hnd = motion::attach_motion_fields(mesh, mesh.dim());
  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT(disp != nullptr);
  ASSERT(vel != nullptr);

  const size_t n = mesh.n_vertices();
  const size_t c = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(c, 3u);
  for (size_t i = 0; i < n * c; ++i) {
    ASSERT_NEAR(disp[i], 0.0, 1e-12);
    ASSERT_NEAR(vel[i], 0.0, 1e-12);
  }
}

static void test_owned_displacement_is_exchanged_and_updates_ghosts(int rank, int world_size)
{
  if (world_size < 2) {
    if (rank == 0) {
      std::cout << "Skipping MeshMotion ghost exchange test (requires >= 2 ranks)\n";
    }
    return;
  }

  auto mesh = build_partitioned_hex_strip(world_size);
  ASSERT_EQ(mesh.rank(), rank);
  ASSERT_EQ(mesh.world_size(), world_size);
  ASSERT(mesh.n_ghost_vertices() > 0);

  const auto initial_hnd = motion::attach_motion_fields(mesh, mesh.dim());
  auto* initial_disp =
      MeshFields::field_data_as<real_t>(mesh.local_mesh(), initial_hnd.displacement);
  auto* initial_vel =
      MeshFields::field_data_as<real_t>(mesh.local_mesh(), initial_hnd.velocity);
  ASSERT(initial_disp != nullptr);
  ASSERT(initial_vel != nullptr);

  const size_t initial_c =
      MeshFields::field_components(mesh.local_mesh(), initial_hnd.displacement);
  ASSERT_EQ(initial_c, 3u);
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const size_t base = static_cast<size_t>(v) * initial_c;
    if (!mesh.is_owned_vertex(v)) {
      continue;
    }
    const real_t owner_value = static_cast<real_t>(rank + 1);
    initial_disp[base + 0] = owner_value + real_t(10);
    initial_disp[base + 1] = owner_value + real_t(20);
    initial_disp[base + 2] = owner_value + real_t(30);
    initial_vel[base + 0] = owner_value + real_t(100);
    initial_vel[base + 1] = owner_value + real_t(200);
    initial_vel[base + 2] = owner_value + real_t(300);
  }

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<OwnedOnlyDisplacementBackend>());

  const double dt = 0.5;
  const bool local_ok = mm.advance(dt);
  ASSERT_EQ(all_true(mesh.mpi_comm(), local_ok), 1);
  ASSERT(local_ok);

  ASSERT(mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), Configuration::Current);

  const auto hnd = motion::attach_motion_fields(mesh, mesh.dim());
  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  const auto* acc  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.acceleration);
  const auto* prev_x =
      MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.previous_coordinates);
  const auto* prev_disp =
      MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.previous_displacement);
  const auto* prev_vel =
      MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.previous_velocity);
  ASSERT(disp != nullptr);
  ASSERT(vel != nullptr);
  ASSERT(acc != nullptr);
  ASSERT(prev_x != nullptr);
  ASSERT(prev_disp != nullptr);
  ASSERT(prev_vel != nullptr);

  const size_t c = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(c, 3u);

  const auto& X_ref = mesh.X_ref();
  const auto& X_cur = mesh.X_cur();
  ASSERT_EQ(X_ref.size(), X_cur.size());

  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const rank_t owner = mesh.owner_rank_vertex(v);
    const real_t owner_value = static_cast<real_t>(owner + 1);
    const real_t expected_disp_x = owner_value;
    const real_t expected_prev_disp[3] = {
        owner_value + real_t(10),
        owner_value + real_t(20),
        owner_value + real_t(30)};
    const real_t expected_prev_vel[3] = {
        owner_value + real_t(100),
        owner_value + real_t(200),
        owner_value + real_t(300)};

    const size_t base = static_cast<size_t>(v) * c;
    ASSERT_NEAR(disp[base + 0], expected_disp_x, 1e-12);
    ASSERT_NEAR(disp[base + 1], 0.0, 1e-12);
    ASSERT_NEAR(disp[base + 2], 0.0, 1e-12);

    ASSERT_NEAR(vel[base + 0], expected_disp_x / static_cast<real_t>(dt), 1e-12);
    ASSERT_NEAR(vel[base + 1], 0.0, 1e-12);
    ASSERT_NEAR(vel[base + 2], 0.0, 1e-12);

    ASSERT_NEAR(acc[base + 0],
                (expected_disp_x / static_cast<real_t>(dt) - expected_prev_vel[0]) /
                    static_cast<real_t>(dt),
                1e-12);
    ASSERT_NEAR(acc[base + 1],
                (real_t(0) - expected_prev_vel[1]) / static_cast<real_t>(dt),
                1e-12);
    ASSERT_NEAR(acc[base + 2],
                (real_t(0) - expected_prev_vel[2]) / static_cast<real_t>(dt),
                1e-12);

    ASSERT_NEAR(prev_x[base + 0], X_ref[base + 0], 1e-12);
    ASSERT_NEAR(prev_x[base + 1], X_ref[base + 1], 1e-12);
    ASSERT_NEAR(prev_x[base + 2], X_ref[base + 2], 1e-12);

    ASSERT_NEAR(prev_disp[base + 0], expected_prev_disp[0], 1e-12);
    ASSERT_NEAR(prev_disp[base + 1], expected_prev_disp[1], 1e-12);
    ASSERT_NEAR(prev_disp[base + 2], expected_prev_disp[2], 1e-12);

    ASSERT_NEAR(prev_vel[base + 0], expected_prev_vel[0], 1e-12);
    ASSERT_NEAR(prev_vel[base + 1], expected_prev_vel[1], 1e-12);
    ASSERT_NEAR(prev_vel[base + 2], expected_prev_vel[2], 1e-12);

    ASSERT_NEAR(X_cur[base + 0], X_ref[base + 0] + expected_disp_x, 1e-12);
    ASSERT_NEAR(X_cur[base + 1], X_ref[base + 1], 1e-12);
    ASSERT_NEAR(X_cur[base + 2], X_ref[base + 2], 1e-12);
  }
}

} // namespace svmp::test

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  svmp::test::test_failure_propagates_across_ranks(rank, world_size);
  svmp::test::test_owned_displacement_is_exchanged_and_updates_ghosts(rank, world_size);

  if (rank == 0) {
    std::cout << "MeshMotion MPI tests PASSED\n";
  }

  MPI_Finalize();
  return 0;
}
