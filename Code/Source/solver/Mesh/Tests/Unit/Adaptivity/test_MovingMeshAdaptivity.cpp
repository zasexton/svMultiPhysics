/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"
#include "../../../Motion/MotionFields.h"

#include <cmath>
#include <vector>

namespace svmp {
namespace test {
namespace {

MeshBase make_single_line_mesh()
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

AdaptivityOptions moving_refine_options()
{
  AdaptivityOptions options;
  options.enable_refinement = true;
  options.max_refinement_level = 1;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES;
  options.check_quality = false;
  options.enforce_quality_after_refinement = false;
  options.verbosity = 0;
  return options;
}

std::size_t vertex_at_x(const MeshBase& mesh, real_t x)
{
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    if (std::abs(mesh.X_ref()[2 * v] - x) < 1.0e-12) {
      return v;
    }
  }
  return mesh.n_vertices();
}

} // namespace

TEST(MovingMeshAdaptivity, RefinePreservesCurrentCoordinatesMotionFieldsAndProvenance)
{
  auto mesh = make_single_line_mesh();
  const auto motion_handles = motion::attach_motion_fields(mesh, 2);

  auto* displacement = MeshFields::field_data_as<real_t>(mesh, motion_handles.displacement);
  auto* velocity = MeshFields::field_data_as<real_t>(mesh, motion_handles.velocity);
  ASSERT_NE(displacement, nullptr);
  ASSERT_NE(velocity, nullptr);

  displacement[0] = 0.0;
  displacement[1] = 0.0;
  displacement[2] = 1.0;
  displacement[3] = 0.0;
  velocity[0] = 10.0;
  velocity[1] = 0.0;
  velocity[2] = 20.0;
  velocity[3] = 0.0;

  mesh.set_current_coords({0.0, 0.0, 2.0, 0.0});
  mesh.use_current_configuration();

  AdaptivityManager manager(moving_refine_options());
  auto result = manager.refine(mesh, {true}, nullptr);

  ASSERT_TRUE(result.success) << result.summary();
  ASSERT_EQ(mesh.n_vertices(), 3u);
  ASSERT_TRUE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Current);

  const auto midpoint = vertex_at_x(mesh, 0.5);
  ASSERT_LT(midpoint, mesh.n_vertices());
  EXPECT_NEAR(mesh.X_ref()[2 * midpoint + 0], 0.5, 1.0e-12);
  EXPECT_NEAR(mesh.X_cur()[2 * midpoint + 0], 1.0, 1.0e-12);

  const auto disp_handle =
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Displacement));
  const auto vel_handle =
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Velocity));
  const auto* new_displacement = MeshFields::field_data_as<const real_t>(mesh, disp_handle);
  const auto* new_velocity = MeshFields::field_data_as<const real_t>(mesh, vel_handle);
  ASSERT_NE(new_displacement, nullptr);
  ASSERT_NE(new_velocity, nullptr);
  EXPECT_NEAR(new_displacement[2 * midpoint + 0], 0.5, 1.0e-12);
  EXPECT_NEAR(new_velocity[2 * midpoint + 0], 15.0, 1.0e-12);

  ASSERT_NE(result.refinement_delta, nullptr);
  ASSERT_FALSE(result.refinement_delta->new_vertices.empty());
  EXPECT_FALSE(result.refinement_delta->new_vertices.front().reference_coordinate_weights.empty());
  EXPECT_FALSE(result.refinement_delta->new_vertices.front().current_coordinate_weights.empty());

  EXPECT_GT(result.transfer_stats.motion_values_transferred.count("mesh_displacement"), 0u);
  EXPECT_GT(result.transfer_stats.motion_values_transferred.count("mesh_velocity"), 0u);
}

TEST(MovingMeshAdaptivity, CoarsenPreservesAcceptedCurrentCoordinatesAndMotionFields)
{
  auto mesh = make_single_line_mesh();
  const auto motion_handles = motion::attach_motion_fields(mesh, 2);

  auto* displacement = MeshFields::field_data_as<real_t>(mesh, motion_handles.displacement);
  auto* velocity = MeshFields::field_data_as<real_t>(mesh, motion_handles.velocity);
  ASSERT_NE(displacement, nullptr);
  ASSERT_NE(velocity, nullptr);

  displacement[0] = 0.0;
  displacement[1] = 0.0;
  displacement[2] = 1.0;
  displacement[3] = 0.0;
  velocity[0] = 10.0;
  velocity[1] = 0.0;
  velocity[2] = 20.0;
  velocity[3] = 0.0;

  mesh.set_current_coords({0.0, 0.0, 2.0, 0.0});
  mesh.use_current_configuration();

  auto options = moving_refine_options();
  options.enable_coarsening = true;
  AdaptivityManager manager(options);

  auto refined = manager.refine(mesh, {true}, nullptr);
  ASSERT_TRUE(refined.success) << refined.summary();
  ASSERT_EQ(mesh.n_cells(), 2u);
  ASSERT_EQ(mesh.n_vertices(), 3u);
  ASSERT_TRUE(mesh.has_current_coords());

  auto coarsened = manager.coarsen(mesh, std::vector<bool>(mesh.n_cells(), true), nullptr);
  ASSERT_TRUE(coarsened.success) << coarsened.summary();
  ASSERT_EQ(mesh.n_cells(), 1u);
  ASSERT_EQ(mesh.n_vertices(), 2u);
  ASSERT_TRUE(mesh.has_current_coords());
  EXPECT_EQ(mesh.active_configuration(), Configuration::Current);

  const auto left = vertex_at_x(mesh, 0.0);
  const auto right = vertex_at_x(mesh, 1.0);
  ASSERT_LT(left, mesh.n_vertices());
  ASSERT_LT(right, mesh.n_vertices());
  EXPECT_NEAR(mesh.X_cur()[2 * left + 0], 0.0, 1.0e-12);
  EXPECT_NEAR(mesh.X_cur()[2 * right + 0], 2.0, 1.0e-12);

  const auto disp_handle =
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Displacement));
  const auto vel_handle =
      MeshFields::get_field_handle(mesh, EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Velocity));
  const auto* new_displacement = MeshFields::field_data_as<const real_t>(mesh, disp_handle);
  const auto* new_velocity = MeshFields::field_data_as<const real_t>(mesh, vel_handle);
  ASSERT_NE(new_displacement, nullptr);
  ASSERT_NE(new_velocity, nullptr);
  EXPECT_NEAR(new_displacement[2 * left + 0], 0.0, 1.0e-12);
  EXPECT_NEAR(new_displacement[2 * right + 0], 1.0, 1.0e-12);
  EXPECT_NEAR(new_velocity[2 * left + 0], 10.0, 1.0e-12);
  EXPECT_NEAR(new_velocity[2 * right + 0], 20.0, 1.0e-12);
}

TEST(MovingMeshAdaptivity, ConservativeTransferPreservesScalarComponentSum)
{
  auto mesh = make_single_line_mesh();
  auto scalar = MeshFields::attach_field(mesh, EntityKind::Vertex, "mass_like", FieldScalarType::Float64, 1);
  auto* values = MeshFields::field_data_as<real_t>(mesh, scalar);
  ASSERT_NE(values, nullptr);
  values[0] = 2.0;
  values[1] = 4.0;

  auto options = moving_refine_options();
  options.field_transfer = FieldTransferType::CONSERVATIVE;
  options.preserve_integrals = true;
  options.transfer_fields = {"mass_like"};

  AdaptivityManager manager(options);
  auto result = manager.refine(mesh, {true}, nullptr);

  ASSERT_TRUE(result.success) << result.summary();
  auto transferred = MeshFields::get_field_handle(mesh, EntityKind::Vertex, "mass_like");
  const auto* new_values = MeshFields::field_data_as<const real_t>(mesh, transferred);
  ASSERT_NE(new_values, nullptr);

  long double sum = 0.0L;
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    sum += new_values[v];
  }
  EXPECT_NEAR(static_cast<double>(sum), 6.0, 1.0e-12);
  ASSERT_GT(result.transfer_stats.conservation_errors.count("mass_like"), 0u);
  EXPECT_LE(result.transfer_stats.conservation_errors["mass_like"], 1.0e-12);
}

} // namespace test
} // namespace svmp
