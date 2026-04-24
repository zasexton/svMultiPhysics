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

#include <gtest/gtest.h>

#include "../../../Core/MeshBase.h"
#include "../../../Core/DistributedMesh.h"
#include "../../../Fields/MeshFields.h"

#include "../../../Motion/MeshMotion.h"
#include "../../../Motion/MotionFields.h"
#include "../../../Motion/MotionQuality.h"
#include "../../../Motion/MotionState.h"

#include <memory>
#include <vector>

using namespace svmp;

namespace {

Mesh make_unit_tet_mesh()
{
  Mesh mesh;

  const std::vector<real_t> coords = {
      0.0, 0.0, 0.0,  // v0
      1.0, 0.0, 0.0,  // v1
      0.0, 1.0, 0.0,  // v2
      0.0, 0.0, 1.0   // v3
  };

  const std::vector<offset_t> offsets = {0, 4};
  const std::vector<index_t> conn = {0, 1, 2, 3};
  const std::vector<CellShape> shapes = {{CellFamily::Tetra, 4, 1}};

  mesh.build_from_arrays(3, coords, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

Mesh make_two_tet_mesh_with_inverted_second()
{
  Mesh mesh;

  const std::vector<real_t> coords = {
      // Tet 0 (well-oriented)
      0.0, 0.0, 0.0,  // v0
      1.0, 0.0, 0.0,  // v1
      0.0, 1.0, 0.0,  // v2
      0.0, 0.0, 1.0,  // v3

      // Tet 1 (same shape, translated)
      2.0, 0.0, 0.0,  // v4
      3.0, 0.0, 0.0,  // v5
      2.0, 1.0, 0.0,  // v6
      2.0, 0.0, 1.0   // v7
  };

  const std::vector<offset_t> offsets = {0, 4, 8};
  const std::vector<index_t> conn = {
      0, 1, 2, 3,
      // Swap two vertices to invert the Jacobian sign.
      4, 6, 5, 7
  };
  const std::vector<CellShape> shapes = {
      {CellFamily::Tetra, 4, 1},
      {CellFamily::Tetra, 4, 1},
  };

  mesh.build_from_arrays(3, coords, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

class ConstantDisplacementBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "ConstantDisplacementBackend"; }

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
    for (size_t v = 0; v < n; ++v) {
      request.displacement.data[v * c + 0] = 1.0;
      if (c > 1) request.displacement.data[v * c + 1] = 0.0;
      if (c > 2) request.displacement.data[v * c + 2] = 0.0;
    }
    result.success = true;
    result.wrote_velocity = false;
    return result;
  }
};

class FailingBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "FailingBackend"; }

  motion::MotionSolveResult solve(const motion::MotionSolveRequest& /*request*/) override
  {
    motion::MotionSolveResult result{};
    result.success = false;
    result.message = "intentional failure";
    return result;
  }
};

class DisplacementAndVelocityBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "DisplacementAndVelocityBackend"; }

  motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override
  {
    motion::MotionSolveResult result{};
    if (!request.displacement.valid() || !request.velocity.valid()) {
      result.success = false;
      result.message = "field views invalid";
      return result;
    }

    const size_t n = request.displacement.n_entities;
    const size_t c = request.displacement.components;
    const size_t vc = request.velocity.components;

    for (size_t v = 0; v < n; ++v) {
      request.displacement.data[v * c + 0] = 1.0;
      if (c > 1) request.displacement.data[v * c + 1] = 0.0;
      if (c > 2) request.displacement.data[v * c + 2] = 0.0;

      request.velocity.data[v * vc + 0] = -7.0;
      if (vc > 1) request.velocity.data[v * vc + 1] = 0.0;
      if (vc > 2) request.velocity.data[v * vc + 2] = 0.0;
    }

    result.success = true;
    result.wrote_velocity = true;
    return result;
  }
};

class ScaledConstantDisplacementBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "ScaledConstantDisplacementBackend"; }

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
    const real_t a = static_cast<real_t>(request.step_scale);

    for (size_t v = 0; v < n; ++v) {
      request.displacement.data[v * c + 0] = a;
      if (c > 1) request.displacement.data[v * c + 1] = 0.0;
      if (c > 2) request.displacement.data[v * c + 2] = 0.0;
    }

    result.success = true;
    result.wrote_velocity = false;
    return result;
  }
};

class ExponentialZDecayBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "ExponentialZDecayBackend"; }

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
    for (size_t v = 0; v < n; ++v) {
      request.displacement.data[v * c + 0] = 0.0;
      if (c > 1) request.displacement.data[v * c + 1] = 0.0;
      if (c > 2) request.displacement.data[v * c + 2] = 0.0;
    }

    // Nonlinear "motion model": z_{n+1} = z_n - 2*z_n*step_scale.
    // A full step (step_scale=1) inverts the unit tet (z becomes negative),
    // but smaller substeps can complete the full advance via backtracking.
    const auto xyz = request.mesh.get_vertex_coords(3);
    const real_t z = xyz[2];
    const real_t duz = static_cast<real_t>(-2.0) * z * static_cast<real_t>(request.step_scale);

    if (c > 2) {
      request.displacement.data[3 * c + 2] = duz;
    }

    result.success = true;
    result.wrote_velocity = false;
    return result;
  }
};

class OneThenFailBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "OneThenFailBackend"; }

  motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override
  {
    ++calls;

    if (calls == 1) {
      motion::MotionSolveResult result{};
      if (!request.displacement.valid()) {
        result.success = false;
        result.message = "displacement view invalid";
        return result;
      }

      const size_t n = request.displacement.n_entities;
      const size_t c = request.displacement.components;
      const real_t a = static_cast<real_t>(request.step_scale);

      for (size_t v = 0; v < n; ++v) {
        request.displacement.data[v * c + 0] = a;
        if (c > 1) request.displacement.data[v * c + 1] = 0.0;
        if (c > 2) request.displacement.data[v * c + 2] = 0.0;
      }

      result.success = true;
      result.wrote_velocity = false;
      return result;
    }

    motion::MotionSolveResult fail{};
    fail.success = false;
    fail.message = "intentional failure after first substep";
    return fail;
  }

  int calls{0};
};

class AngleThresholdBacktrackingBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "AngleThresholdBacktrackingBackend"; }

  motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override
  {
    scales.push_back(request.step_scale);

    motion::MotionSolveResult result{};
    if (!request.displacement.valid()) {
      result.success = false;
      result.message = "displacement view invalid";
      return result;
    }

    // Nonlinear "motion model" applied to vertex 2 only:
    // - x, z relax toward 0.5 with an explicit Euler step.
    // - y decays multiplicatively (with alpha close to 1), so a single full step
    //   collapses the element angle, while multiple substeps remain acceptable.
    const auto xyz = request.mesh.get_vertex_coords(2);
    const real_t x = xyz[0];
    const real_t y = xyz[1];
    const real_t z = xyz[2];

    const real_t s = static_cast<real_t>(request.step_scale);
    const real_t x_new = x + s * (static_cast<real_t>(0.5) - x);
    const real_t z_new = z + s * (static_cast<real_t>(0.5) - z);
    constexpr real_t alpha = static_cast<real_t>(0.99);
    const real_t y_new = y * (static_cast<real_t>(1.0) - alpha * s);

    const real_t dux = x_new - x;
    const real_t duy = y_new - y;
    const real_t duz = z_new - z;

    const size_t c = request.displacement.components;
    request.displacement.data[2 * c + 0] = dux;
    if (c > 1) request.displacement.data[2 * c + 1] = duy;
    if (c > 2) request.displacement.data[2 * c + 2] = duz;

    result.success = true;
    result.wrote_velocity = false;
    return result;
  }

  std::vector<double> scales;
};

} // namespace

TEST(MotionFieldsTest, AttachMotionFieldsCreatesExpectedFields)
{
  auto mesh = make_unit_tet_mesh();

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  ASSERT_NE(hnd.displacement.id, 0u);
  ASSERT_NE(hnd.velocity.id, 0u);
  ASSERT_NE(hnd.acceleration.id, 0u);
  ASSERT_NE(hnd.previous_coordinates.id, 0u);
  ASSERT_NE(hnd.previous_displacement.id, 0u);
  ASSERT_NE(hnd.previous_velocity.id, 0u);

  EXPECT_TRUE(MeshFields::has_field(mesh.local_mesh(), EntityKind::Vertex, "mesh_displacement"));
  EXPECT_TRUE(MeshFields::has_field(mesh.local_mesh(), EntityKind::Vertex, "mesh_velocity"));
  EXPECT_TRUE(MeshFields::has_field(mesh.local_mesh(), EntityKind::Vertex, "mesh_acceleration"));
  EXPECT_TRUE(MeshFields::has_field(mesh.local_mesh(), EntityKind::Vertex, "previous_coordinates"));
  EXPECT_TRUE(MeshFields::has_field(mesh.local_mesh(), EntityKind::Vertex, "previous_mesh_displacement"));
  EXPECT_TRUE(MeshFields::has_field(mesh.local_mesh(), EntityKind::Vertex, "previous_mesh_velocity"));

  EXPECT_EQ(MeshFields::field_type(mesh.local_mesh(), hnd.displacement), FieldScalarType::Float64);
  EXPECT_EQ(MeshFields::field_type(mesh.local_mesh(), hnd.velocity), FieldScalarType::Float64);
  EXPECT_EQ(MeshFields::field_type(mesh.local_mesh(), hnd.acceleration), FieldScalarType::Float64);
  EXPECT_EQ(MeshFields::field_components(mesh.local_mesh(), hnd.displacement), 3u);
  EXPECT_EQ(MeshFields::field_components(mesh.local_mesh(), hnd.velocity), 3u);
  EXPECT_EQ(MeshFields::field_components(mesh.local_mesh(), hnd.acceleration), 3u);

  const auto* disp_desc = MeshFields::field_descriptor(mesh.local_mesh(), hnd.displacement);
  const auto* vel_desc  = MeshFields::field_descriptor(mesh.local_mesh(), hnd.velocity);
  const auto* acc_desc  = MeshFields::field_descriptor(mesh.local_mesh(), hnd.acceleration);
  const auto* prev_x_desc = MeshFields::field_descriptor(mesh.local_mesh(), hnd.previous_coordinates);
  ASSERT_NE(disp_desc, nullptr);
  ASSERT_NE(vel_desc, nullptr);
  ASSERT_NE(acc_desc, nullptr);
  ASSERT_NE(prev_x_desc, nullptr);

  EXPECT_EQ(disp_desc->units, "m");
  EXPECT_EQ(disp_desc->ghost_policy, FieldGhostPolicy::Exchange);
  EXPECT_TRUE(disp_desc->time_dependent);

  EXPECT_EQ(vel_desc->units, "m/s");
  EXPECT_EQ(vel_desc->ghost_policy, FieldGhostPolicy::Exchange);
  EXPECT_TRUE(vel_desc->time_dependent);

  EXPECT_EQ(acc_desc->units, "m/s^2");
  EXPECT_EQ(acc_desc->ghost_policy, FieldGhostPolicy::Exchange);
  EXPECT_TRUE(acc_desc->time_dependent);

  EXPECT_EQ(prev_x_desc->units, "m");
  EXPECT_EQ(prev_x_desc->ghost_policy, FieldGhostPolicy::Exchange);
  EXPECT_TRUE(prev_x_desc->time_dependent);
  EXPECT_EQ(motion::parse_motion_field_role("mesh_velocity"), motion::MotionFieldRole::Velocity);
  EXPECT_TRUE(motion::is_standard_motion_field_name("previous_mesh_velocity"));
  EXPECT_STREQ(motion::standard_motion_field_name(motion::MotionFieldRole::Acceleration),
               "mesh_acceleration");

  const auto hnd2 = motion::attach_motion_fields(mesh, 3);
  EXPECT_EQ(hnd2.displacement.id, hnd.displacement.id);
  EXPECT_EQ(hnd2.velocity.id, hnd.velocity.id);
  EXPECT_EQ(hnd2.acceleration.id, hnd.acceleration.id);
  EXPECT_EQ(hnd2.previous_coordinates.id, hnd.previous_coordinates.id);

  EXPECT_THROW(motion::attach_motion_fields(mesh, 2), std::runtime_error);
  EXPECT_THROW(motion::attach_motion_fields(mesh, 0), std::invalid_argument);
}

TEST(MotionFieldsTest, AttachMotionFieldsRejectsIncompatibleExistingStandardField)
{
  auto mesh = make_unit_tet_mesh();
  MeshFields::attach_field(mesh.local_mesh(),
                           EntityKind::Vertex,
                           "mesh_displacement",
                           FieldScalarType::Float64,
                           2);

  EXPECT_THROW(motion::attach_motion_fields(mesh, 3), std::runtime_error);
}

TEST(MotionFieldsTest, UpdateCoordinatesFromDisplacementAbsoluteAndIncremental)
{
  auto mesh = make_unit_tet_mesh();
  const auto X_ref = mesh.X_ref();

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  ASSERT_NE(disp, nullptr);

  const size_t n_vertices = mesh.n_vertices();
  const size_t ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(ncomp, 3u);

  // Absolute update: X_cur = X_ref + u
  for (size_t v = 0; v < n_vertices; ++v) {
    disp[v * ncomp + 0] = 0.1 * static_cast<real_t>(v + 1);
    disp[v * ncomp + 1] = -0.2 * static_cast<real_t>(v + 1);
    disp[v * ncomp + 2] = 0.3 * static_cast<real_t>(v + 1);
  }

  motion::update_coordinates_from_displacement(
      mesh, hnd, motion::DisplacementUpdateMode::AbsoluteFromReference);

  ASSERT_TRUE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Current);

  const auto& X_cur_1 = mesh.X_cur();
  ASSERT_EQ(X_cur_1.size(), X_ref.size());

  for (size_t v = 0; v < n_vertices; ++v) {
    const size_t base = v * ncomp;
    EXPECT_NEAR(X_cur_1[base + 0], X_ref[base + 0] + 0.1 * static_cast<real_t>(v + 1), 1e-12);
    EXPECT_NEAR(X_cur_1[base + 1], X_ref[base + 1] - 0.2 * static_cast<real_t>(v + 1), 1e-12);
    EXPECT_NEAR(X_cur_1[base + 2], X_ref[base + 2] + 0.3 * static_cast<real_t>(v + 1), 1e-12);
  }

  // Incremental update: X_cur_new = X_cur_old + u
  const std::vector<real_t> X_prev = mesh.X_cur();

  for (size_t v = 0; v < n_vertices; ++v) {
    disp[v * ncomp + 0] = -0.05;
    disp[v * ncomp + 1] = 0.10;
    disp[v * ncomp + 2] = 0.00;
  }

  motion::update_coordinates_from_displacement(
      mesh, hnd, motion::DisplacementUpdateMode::IncrementalFromCurrent);

  const auto& X_cur_2 = mesh.X_cur();
  ASSERT_EQ(X_cur_2.size(), X_prev.size());

  for (size_t i = 0; i < X_cur_2.size(); ++i) {
    const size_t comp = i % ncomp;
    real_t du = 0.0;
    if (comp == 0) du = -0.05;
    if (comp == 1) du = 0.10;
    EXPECT_NEAR(X_cur_2[i], X_prev[i] + du, 1e-12);
  }
}

TEST(MotionStateTest, RestoreWithoutCurrentForcesReferenceConfiguration)
{
  auto mesh = make_unit_tet_mesh();

  // Create an intentionally inconsistent state to ensure restore() is robust.
  // (Mesh local view allows switching active configuration independent of X_cur.)
  mesh.use_current_configuration();
  ASSERT_EQ(mesh.active_configuration(), Configuration::Current);
  ASSERT_FALSE(mesh.has_current_coords());

  motion::MotionCoordinateBackup backup;
  motion::save_coordinates(mesh, backup);
  ASSERT_TRUE(backup.valid());
  EXPECT_FALSE(backup.has_current);
  EXPECT_EQ(backup.active_config, Configuration::Current);

  // Mutate mesh state.
  std::vector<real_t> X_cur = mesh.X_ref();
  X_cur[0] += 1.0;
  mesh.set_current_coords(X_cur);
  mesh.use_current_configuration();
  ASSERT_TRUE(mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), Configuration::Current);

  // Restore.
  motion::restore_coordinates(mesh, backup);
  EXPECT_FALSE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Reference);
}

TEST(MotionStateTest, UpdateVelocityFromDisplacement)
{
  auto mesh = make_unit_tet_mesh();
  const auto hnd = motion::attach_motion_fields(mesh, 3);

  auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(disp, nullptr);
  ASSERT_NE(vel, nullptr);

  const size_t n_vertices = mesh.n_vertices();
  const size_t ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(ncomp, 3u);

  for (size_t v = 0; v < n_vertices; ++v) {
    disp[v * ncomp + 0] = static_cast<real_t>(v + 1);
    disp[v * ncomp + 1] = 2.0 * static_cast<real_t>(v + 1);
    disp[v * ncomp + 2] = -3.0 * static_cast<real_t>(v + 1);
  }

  const real_t dt = 0.5;
  motion::update_velocity_from_displacement(mesh, hnd, dt);

  for (size_t v = 0; v < n_vertices; ++v) {
    EXPECT_NEAR(vel[v * ncomp + 0], disp[v * ncomp + 0] / dt, 1e-12);
    EXPECT_NEAR(vel[v * ncomp + 1], disp[v * ncomp + 1] / dt, 1e-12);
    EXPECT_NEAR(vel[v * ncomp + 2], disp[v * ncomp + 2] / dt, 1e-12);
  }

  EXPECT_THROW(motion::update_velocity_from_displacement(mesh, hnd, 0.0), std::invalid_argument);
}

TEST(MotionQualityTest, SuggestedStepScaleRejectsAndReduces)
{
  motion::MotionQualityReport report{};

  report.min_jacobian = 1.0;
  report.min_angle_deg = 60.0;
  report.max_skewness = 0.1;
  report.has_inverted_cells = false;

  EXPECT_DOUBLE_EQ(motion::suggested_step_scale(report, 1.0, 0.5, 10.0, 0.9), 1.0);

  report.min_jacobian = 0.4; // below threshold => reduce
  EXPECT_DOUBLE_EQ(motion::suggested_step_scale(report, 1.0, 0.5, 10.0, 0.9), 0.5);

  report.min_jacobian = -0.1;
  report.has_inverted_cells = true;
  EXPECT_DOUBLE_EQ(motion::suggested_step_scale(report, 1.0, 0.5, 10.0, 0.9), 0.0);
}

TEST(MeshMotionTest, AdvanceInitializesAndResetClearsCurrentCoordinates)
{
  auto mesh = make_unit_tet_mesh();
  ASSERT_FALSE(mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), Configuration::Reference);

  motion::MeshMotion mm(mesh);
  EXPECT_TRUE(mm.advance(0.1));
  EXPECT_TRUE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Current);
  EXPECT_EQ(mesh.X_cur(), mesh.X_ref());

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  const auto* acc = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.acceleration);
  const auto* prev_x =
      MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.previous_coordinates);
  ASSERT_NE(disp, nullptr);
  ASSERT_NE(vel, nullptr);
  ASSERT_NE(acc, nullptr);
  ASSERT_NE(prev_x, nullptr);

  const auto& X_ref = mesh.X_ref();
  for (size_t i = 0; i < mesh.n_vertices() * 3u; ++i) {
    EXPECT_NEAR(disp[i], 0.0, 1e-12);
    EXPECT_NEAR(vel[i], 0.0, 1e-12);
    EXPECT_NEAR(acc[i], 0.0, 1e-12);
    EXPECT_NEAR(prev_x[i], X_ref[i], 1e-12);
  }

  mm.reset_to_reference();
  EXPECT_FALSE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Reference);
}

TEST(MeshMotionTest, BackendFailureRestoresCoordinateState)
{
  auto mesh = make_unit_tet_mesh();

  ASSERT_FALSE(mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), Configuration::Reference);

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<FailingBackend>());

  EXPECT_FALSE(mm.advance(0.1));
  EXPECT_FALSE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Reference);
}

TEST(MeshMotionTest, InjectedBackendWritesFieldsAndUpdatesCoordinatesAndVelocity)
{
  auto mesh = make_unit_tet_mesh();

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  auto* initial_vel = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(initial_vel, nullptr);
  const size_t initial_ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.velocity);
  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    initial_vel[v * initial_ncomp + 0] = 0.5;
    initial_vel[v * initial_ncomp + 1] = 0.25;
    initial_vel[v * initial_ncomp + 2] = -0.5;
  }

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<ConstantDisplacementBackend>());

  const double dt = 0.5;
  ASSERT_TRUE(mm.advance(dt));
  ASSERT_TRUE(mesh.has_current_coords());

  // X_cur should equal X_ref + [1,0,0] per vertex.
  const auto& X_ref = mesh.X_ref();
  const auto& X_cur = mesh.X_cur();
  ASSERT_EQ(X_cur.size(), X_ref.size());

  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    const size_t base = v * 3;
    EXPECT_NEAR(X_cur[base + 0], X_ref[base + 0] + 1.0, 1e-12);
    EXPECT_NEAR(X_cur[base + 1], X_ref[base + 1], 1e-12);
    EXPECT_NEAR(X_cur[base + 2], X_ref[base + 2], 1e-12);
  }

  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  const auto* acc  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.acceleration);
  const auto* prev_x = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.previous_coordinates);
  const auto* prev_vel = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.previous_velocity);
  ASSERT_NE(disp, nullptr);
  ASSERT_NE(vel, nullptr);
  ASSERT_NE(acc, nullptr);
  ASSERT_NE(prev_x, nullptr);
  ASSERT_NE(prev_vel, nullptr);

  const size_t ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(ncomp, 3u);

  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    const size_t base = v * ncomp;
    EXPECT_NEAR(disp[base + 0], 1.0, 1e-12);
    EXPECT_NEAR(disp[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(disp[base + 2], 0.0, 1e-12);

    EXPECT_NEAR(vel[base + 0], 1.0 / dt, 1e-12);
    EXPECT_NEAR(vel[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(vel[base + 2], 0.0, 1e-12);

    EXPECT_NEAR(acc[base + 0], (1.0 / dt - 0.5) / dt, 1e-12);
    EXPECT_NEAR(acc[base + 1], (0.0 - 0.25) / dt, 1e-12);
    EXPECT_NEAR(acc[base + 2], (0.0 + 0.5) / dt, 1e-12);

    EXPECT_NEAR(prev_x[base + 0], X_ref[base + 0], 1e-12);
    EXPECT_NEAR(prev_x[base + 1], X_ref[base + 1], 1e-12);
    EXPECT_NEAR(prev_x[base + 2], X_ref[base + 2], 1e-12);

    EXPECT_NEAR(prev_vel[base + 0], 0.5, 1e-12);
    EXPECT_NEAR(prev_vel[base + 1], 0.25, 1e-12);
    EXPECT_NEAR(prev_vel[base + 2], -0.5, 1e-12);
  }
}

TEST(MeshMotionTest, BackendProvidedVelocityIsNotOverwritten)
{
  auto mesh = make_unit_tet_mesh();

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<DisplacementAndVelocityBackend>());

  const double dt = 0.5;
  ASSERT_TRUE(mm.advance(dt));

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(vel, nullptr);

  const size_t ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.velocity);
  ASSERT_EQ(ncomp, 3u);

  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    const size_t base = v * ncomp;
    EXPECT_NEAR(vel[base + 0], -7.0, 1e-12);
    EXPECT_NEAR(vel[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(vel[base + 2], 0.0, 1e-12);
  }
}

TEST(MeshMotionTest, DtZeroDoesNotComputeVelocityFallbackAndIsDeterministic)
{
  auto mesh = make_unit_tet_mesh();

  // Backend provides displacement only.
  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<ConstantDisplacementBackend>());

  ASSERT_TRUE(mm.advance(/*dt=*/0.0));

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  const auto* vel = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(vel, nullptr);

  for (size_t i = 0; i < mesh.n_vertices() * 3u; ++i) {
    EXPECT_NEAR(vel[i], 0.0, 1e-12);
  }

  // Backend provides both displacement and velocity; dt==0 must preserve backend velocity.
  motion::MeshMotion mm2(mesh);
  mm2.set_backend(std::make_shared<DisplacementAndVelocityBackend>());

  ASSERT_TRUE(mm2.advance(/*dt=*/0.0));

  const auto hnd2 = motion::attach_motion_fields(mesh, 3);
  const auto* vel2 = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd2.velocity);
  ASSERT_NE(vel2, nullptr);
  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    const size_t base = v * 3u;
    EXPECT_NEAR(vel2[base + 0], -7.0, 1e-12);
    EXPECT_NEAR(vel2[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(vel2[base + 2], 0.0, 1e-12);
  }
}

TEST(MeshMotionTest, BackendFailureAfterAcceptedSubstepRestoresEntryCurrentCoordsAndConfig)
{
  auto mesh = make_unit_tet_mesh();

  // Create an initial current configuration that differs from reference.
  std::vector<real_t> X_cur = mesh.X_ref();
  for (auto& x : X_cur) {
    x += static_cast<real_t>(0.123);
  }
  mesh.set_current_coords(X_cur);
  mesh.use_reference_configuration();
  ASSERT_TRUE(mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), Configuration::Reference);

  motion::MeshMotion mm(mesh);
  auto backend = std::make_shared<OneThenFailBackend>();
  mm.set_backend(backend);

  motion::MotionConfig cfg;
  cfg.max_step_scale = 0.5; // forces two solve calls to cover partial acceptance + failure path
  cfg.max_substeps = 10;
  cfg.enable_quality_guard = false;
  mm.set_config(cfg);

  EXPECT_FALSE(mm.advance(1.0));
  EXPECT_EQ(backend->calls, 2);

  // Coordinates are restored to entry current coordinates and configuration.
  ASSERT_TRUE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Reference);
  EXPECT_EQ(mesh.X_cur().size(), X_cur.size());
  for (size_t i = 0; i < X_cur.size(); ++i) {
    EXPECT_NEAR(mesh.X_cur()[i], X_cur[i], 1e-12);
  }

  // Motion fields remain attached but are zeroed on failure.
  const auto hnd = motion::attach_motion_fields(mesh, 3);
  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(disp, nullptr);
  ASSERT_NE(vel, nullptr);

  for (size_t i = 0; i < mesh.n_vertices() * 3u; ++i) {
    EXPECT_NEAR(disp[i], 0.0, 1e-12);
    EXPECT_NEAR(vel[i], 0.0, 1e-12);
  }
}

TEST(MeshMotionTest, EnforcedAngleThresholdBacktracksToSmallerSubstepsAndSucceeds)
{
  auto mesh = make_unit_tet_mesh();

  motion::MeshMotion mm(mesh);
  auto backend = std::make_shared<AngleThresholdBacktrackingBackend>();
  mm.set_backend(backend);

  motion::MotionConfig cfg;
  cfg.max_step_scale = 1.0;
  cfg.max_substeps = 10;
  cfg.enable_quality_guard = true;
  cfg.enforce_quality_thresholds = true;
  cfg.quality_min_angle_deg = 20.0;
  mm.set_config(cfg);

  ASSERT_TRUE(mm.advance(1.0));

  ASSERT_EQ(backend->scales.size(), 3u);
  EXPECT_NEAR(backend->scales[0], 1.0, 1e-12);
  EXPECT_NEAR(backend->scales[1], 0.5, 1e-12);
  EXPECT_NEAR(backend->scales[2], 0.5, 1e-12);

  const auto& X_cur = mesh.X_cur();
  ASSERT_EQ(X_cur.size(), 12u);
  // Final vertex 2 coordinates after two half substeps of the nonlinear motion model.
  EXPECT_NEAR(X_cur[6], 0.375, 1e-12);
  EXPECT_NEAR(X_cur[7], 0.255025, 1e-12);
  EXPECT_NEAR(X_cur[8], 0.375, 1e-12);

  const auto report = motion::evaluate_motion_quality(mesh, Configuration::Current);
  EXPECT_GE(report.min_angle_deg, 20.0);
}

TEST(MeshMotionTest, SubsteppingAccumulatesTotalDisplacementAndVelocity)
{
  auto mesh = make_unit_tet_mesh();

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<ScaledConstantDisplacementBackend>());

  motion::MotionConfig cfg;
  cfg.max_step_scale = 0.5;
  cfg.max_substeps = 4;
  mm.set_config(cfg);

  const double dt = 2.0;
  ASSERT_TRUE(mm.advance(dt));
  ASSERT_TRUE(mesh.has_current_coords());

  // With max_step_scale=0.5, advance() performs two substeps with displacement=a=0.5 each.
  // Total displacement over the call is 1.0 in x for every vertex.
  const auto& X_ref = mesh.X_ref();
  const auto& X_cur = mesh.X_cur();
  ASSERT_EQ(X_cur.size(), X_ref.size());

  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    const size_t base = v * 3;
    EXPECT_NEAR(X_cur[base + 0], X_ref[base + 0] + 1.0, 1e-12);
    EXPECT_NEAR(X_cur[base + 1], X_ref[base + 1], 1e-12);
    EXPECT_NEAR(X_cur[base + 2], X_ref[base + 2], 1e-12);
  }

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(disp, nullptr);
  ASSERT_NE(vel, nullptr);

  const size_t ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(ncomp, 3u);

  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    const size_t base = v * ncomp;
    EXPECT_NEAR(disp[base + 0], 1.0, 1e-12);
    EXPECT_NEAR(disp[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(disp[base + 2], 0.0, 1e-12);

    EXPECT_NEAR(vel[base + 0], 1.0 / dt, 1e-12);
    EXPECT_NEAR(vel[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(vel[base + 2], 0.0, 1e-12);
  }
}

TEST(MeshMotionTest, BacktrackingAvoidsInversionViaSmallerSubsteps)
{
  auto mesh = make_unit_tet_mesh();

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<ExponentialZDecayBackend>());

  motion::MotionConfig cfg;
  cfg.max_step_scale = 1.0;
  cfg.max_substeps = 10;
  cfg.enable_quality_guard = true;
  mm.set_config(cfg);

  const double dt = 1.0;
  ASSERT_TRUE(mm.advance(dt));

  const auto& X_cur = mesh.X_cur();
  ASSERT_EQ(X_cur.size(), 12u);
  EXPECT_NEAR(X_cur[11], 0.0625, 1e-12); // z-coordinate of vertex 3

  const auto hnd = motion::attach_motion_fields(mesh, 3);
  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(disp, nullptr);
  ASSERT_NE(vel, nullptr);

  const size_t ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(ncomp, 3u);

  // Total z displacement on vertex 3: 0.0625 - 1.0 = -0.9375
  EXPECT_NEAR(disp[3 * ncomp + 2], -0.9375, 1e-12);
  EXPECT_NEAR(vel[3 * ncomp + 2], -0.9375 / dt, 1e-12);
}

TEST(MeshMotionTest, MaxSubstepsLimitRejectsAndRestores)
{
  auto mesh = make_unit_tet_mesh();
  ASSERT_FALSE(mesh.has_current_coords());

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<ScaledConstantDisplacementBackend>());

  motion::MotionConfig cfg;
  cfg.max_step_scale = 0.25;
  cfg.max_substeps = 3; // Would require 4 substeps to complete.
  mm.set_config(cfg);

  EXPECT_FALSE(mm.advance(1.0));
  EXPECT_FALSE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Reference);

  // Motion fields should remain attached but zeroed on failure.
  const auto hnd = motion::attach_motion_fields(mesh, 3);
  const auto* disp = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.displacement);
  const auto* vel  = MeshFields::field_data_as<real_t>(mesh.local_mesh(), hnd.velocity);
  ASSERT_NE(disp, nullptr);
  ASSERT_NE(vel, nullptr);

  const size_t ncomp = MeshFields::field_components(mesh.local_mesh(), hnd.displacement);
  ASSERT_EQ(ncomp, 3u);

  for (size_t v = 0; v < mesh.n_vertices(); ++v) {
    const size_t base = v * ncomp;
    EXPECT_NEAR(disp[base + 0], 0.0, 1e-12);
    EXPECT_NEAR(disp[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(disp[base + 2], 0.0, 1e-12);

    EXPECT_NEAR(vel[base + 0], 0.0, 1e-12);
    EXPECT_NEAR(vel[base + 1], 0.0, 1e-12);
    EXPECT_NEAR(vel[base + 2], 0.0, 1e-12);
  }
}

TEST(MotionQualityTest, DistributedQualityIgnoresGhostCells)
{
  auto mesh = make_two_tet_mesh_with_inverted_second();

  // Mark the inverted tet as a ghost cell.
  mesh.set_ownership(1, EntityKind::Volume, Ownership::Ghost, 0);

  const auto report = motion::evaluate_motion_quality(mesh, Configuration::Reference);
  EXPECT_FALSE(report.has_inverted_cells);
}

TEST(MeshMotionTest, DistributedQualityGuardUsesOwnedCells)
{
  auto mesh = make_two_tet_mesh_with_inverted_second();

  // Mark the inverted tet as a ghost cell. If MeshMotion evaluated all cells
  // in the local mesh (including ghosts), the step would be rejected.
  mesh.set_ownership(1, EntityKind::Volume, Ownership::Ghost, 0);

  motion::MeshMotion mm(mesh);
  mm.set_backend(std::make_shared<ConstantDisplacementBackend>());

  motion::MotionConfig cfg;
  cfg.enable_quality_guard = true;
  cfg.enforce_quality_thresholds = true;
  mm.set_config(cfg);

  EXPECT_TRUE(mm.advance(1.0));
}
