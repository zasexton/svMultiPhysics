#include "Motion/MotionMap.h"
#include "Fields/MeshFields.h"
#include "Core/MeshBase.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace svmp;
using namespace svmp::motion;

namespace {

MeshBase make_tetra_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0},
      std::vector<offset_t>{0, 4},
      std::vector<index_t>{0, 1, 2, 3},
      std::vector<CellShape>{CellShape{CellFamily::Tetra, 4}});
  mesh.finalize();
  mesh.set_region_label(0, 12);
  return mesh;
}

} // namespace

TEST(MotionMap, RigidRotationPublishesAnalyticDisplacementVelocityAndAcceleration)
{
  auto mesh = make_tetra_mesh();

  RigidBodyMotionParameters parameters;
  parameters.rotation_axis = {{0.0, 0.0, 1.0}};
  parameters.angular_speed = 2.0;
  RigidBodyMotionMap map(parameters, "spin");

  const auto result = apply_motion_map(
      mesh,
      map,
      MotionMapTarget::all("rotating_region"),
      MotionMapTimeState{0.0, 0.0, 0.0, MotionMapTimeLevel::AcceptedTimeStep});

  ASSERT_EQ(result.geometry_dofs.size(), 4u);
  const auto x1 = mesh.geometry_dof_coords(1, Configuration::Current);
  EXPECT_DOUBLE_EQ(x1[0], 1.0);
  EXPECT_DOUBLE_EQ(x1[1], 0.0);

  const auto velocity = MeshFields::field_data_as<real_t>(
      mesh,
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, "mesh_velocity"));
  ASSERT_NE(velocity, nullptr);
  EXPECT_DOUBLE_EQ(velocity[3], 0.0);
  EXPECT_DOUBLE_EQ(velocity[4], 2.0);
  EXPECT_DOUBLE_EQ(velocity[5], 0.0);

  const auto acceleration = MeshFields::field_data_as<real_t>(
      mesh,
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, "mesh_acceleration"));
  ASSERT_NE(acceleration, nullptr);
  EXPECT_DOUBLE_EQ(acceleration[3], -4.0);
  EXPECT_DOUBLE_EQ(acceleration[4], 0.0);
  EXPECT_DOUBLE_EQ(acceleration[5], 0.0);
}

TEST(MotionMap, RotationAtTimeMovesGeometryDofsAndCanRollback)
{
  auto mesh = make_tetra_mesh();

  RigidBodyMotionParameters parameters;
  parameters.rotation_axis = {{0.0, 0.0, 1.0}};
  parameters.angular_speed = std::acos(-1.0) / 2.0;
  RigidBodyMotionMap map(parameters);

  MotionMapTransaction tx(mesh);
  const auto result = tx.apply(
      map,
      MotionMapTarget::region(12, "rotating_region"),
      MotionMapTimeState{1.0, 0.0, 1.0, MotionMapTimeLevel::TrialIterate});

  ASSERT_EQ(result.geometry_dofs.size(), 4u);
  const auto moved = mesh.geometry_dof_coords(1, Configuration::Current);
  EXPECT_NEAR(moved[0], 0.0, 1e-12);
  EXPECT_NEAR(moved[1], 1.0, 1e-12);

  tx.rollback();
  EXPECT_EQ(tx.state(), MotionMapTransactionState::RolledBack);
  EXPECT_FALSE(mesh.has_current_coords());
}

TEST(MotionMap, AffineMotionSupportsExplicitGeometryDofTargets)
{
  auto mesh = make_tetra_mesh();

  AffineMotionParameters parameters;
  parameters.transform[0][0] = 2.0;
  parameters.transform[1][1] = 3.0;
  parameters.translation = {{1.0, 0.0, 0.0}};
  parameters.velocity_gradient[0][0] = 5.0;
  AffineMotionMap map(parameters);

  const auto result = apply_motion_map(
      mesh,
      map,
      MotionMapTarget::explicit_dofs({1}, "scaled_tip"),
      MotionMapTimeState{0.0, 0.0, 0.0, MotionMapTimeLevel::AcceptedNonlinearState});

  ASSERT_EQ(result.geometry_dofs.size(), 1u);
  const auto moved = mesh.geometry_dof_coords(1, Configuration::Current);
  EXPECT_DOUBLE_EQ(moved[0], 3.0);
  EXPECT_DOUBLE_EQ(moved[1], 0.0);
  EXPECT_DOUBLE_EQ(moved[2], 0.0);

  const auto untouched = mesh.geometry_dof_coords(2, Configuration::Current);
  EXPECT_DOUBLE_EQ(untouched[0], 0.0);
  EXPECT_DOUBLE_EQ(untouched[1], 1.0);
  EXPECT_DOUBLE_EQ(untouched[2], 0.0);
  EXPECT_DOUBLE_EQ(result.dof_states.front().velocity[0], 5.0);
}

TEST(MotionMap, RestartRecordCapturesMapTargetTimeAndRevisionState)
{
  auto mesh = make_tetra_mesh();

  RigidBodyMotionParameters parameters;
  parameters.rotation_axis = {{0.0, 0.0, 1.0}};
  parameters.angular_speed = 1.5;
  RigidBodyMotionMap map(parameters, "restartable_spin");

  const auto time_state = MotionMapTimeState{
      0.25, 0.0, 0.25, MotionMapTimeLevel::AcceptedTimeStep};
  const auto target = MotionMapTarget::region(12, "rotating_region");
  (void)apply_motion_map(mesh, map, target, time_state);

  const auto record = make_motion_map_restart_record(mesh, map, target, time_state);
  EXPECT_EQ(record.map_name, "restartable_spin");
  EXPECT_EQ(record.map_kind, MotionMapKind::RigidBody);
  EXPECT_EQ(record.target.kind, MotionMapTargetKind::RegionLabel);
  EXPECT_EQ(record.target.label, 12);
  EXPECT_EQ(record.target.logical_region_id, "rotating_region");
  EXPECT_EQ(record.time_state.time_level, MotionMapTimeLevel::AcceptedTimeStep);
  EXPECT_DOUBLE_EQ(record.time_state.time, 0.25);
  EXPECT_EQ(record.geometry_revision, mesh.geometry_revision());
  EXPECT_EQ(record.field_layout_revision, mesh.field_layout_revision());
  EXPECT_EQ(record.active_configuration_epoch, mesh.active_configuration_epoch());
}
