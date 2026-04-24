/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/MeshBase.h"
#include "Fields/MeshFields.h"
#include "IO/MovingMeshRestart.h"
#include "Motion/MotionFields.h"

#ifdef MESH_HAS_ADAPTIVITY
#include "Adaptivity/AdaptivityManager.h"
#endif

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace svmp::test {
namespace {

std::string unique_path(const std::string& suffix)
{
  const auto stamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return (std::filesystem::temp_directory_path() /
          ("svmp_moving_mesh_restart_" + std::to_string(static_cast<long long>(stamp)) + suffix))
      .string();
}

MeshBase make_moved_quad_mesh()
{
  MeshBase mesh(2);
  const std::vector<real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
  };
  const std::vector<offset_t> offsets = {0, 4};
  const std::vector<index_t> conn = {0, 1, 2, 3};
  const std::vector<CellShape> shapes = {{CellFamily::Quad, 4, 1}};
  mesh.build_from_arrays(2, x_ref, offsets, conn, shapes);
  mesh.set_vertex_gids({10, 11, 12, 13});
  mesh.set_cell_gids({100});
  mesh.set_region_label(0, 7);
  mesh.register_label("fluid", 7);
  mesh.add_to_set(EntityKind::Volume, "mixing_region", 0);
  mesh.finalize();

  auto handles = motion::attach_motion_fields(mesh, 2);
  auto* displacement = MeshFields::field_data_as<real_t>(mesh, handles.displacement);
  auto* velocity = MeshFields::field_data_as<real_t>(mesh, handles.velocity);
  auto* previous_coordinates = MeshFields::field_data_as<real_t>(mesh, handles.previous_coordinates);
  auto* previous_displacement = MeshFields::field_data_as<real_t>(mesh, handles.previous_displacement);
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    displacement[2 * v + 0] = 0.25 * static_cast<real_t>(v + 1);
    displacement[2 * v + 1] = -0.05 * static_cast<real_t>(v);
    velocity[2 * v + 0] = 10.0 + static_cast<real_t>(v);
    velocity[2 * v + 1] = -20.0 - static_cast<real_t>(v);
    previous_coordinates[2 * v + 0] = mesh.X_ref()[2 * v + 0] - 0.1;
    previous_coordinates[2 * v + 1] = mesh.X_ref()[2 * v + 1] + 0.2;
    previous_displacement[2 * v + 0] = displacement[2 * v + 0] - 0.03;
    previous_displacement[2 * v + 1] = displacement[2 * v + 1] + 0.04;
  }

  std::vector<real_t> x_cur = mesh.X_ref();
  for (std::size_t i = 0; i < mesh.n_vertices(); ++i) {
    x_cur[2 * i + 0] += displacement[2 * i + 0];
    x_cur[2 * i + 1] += displacement[2 * i + 1];
  }
  mesh.set_current_coords(x_cur);
  mesh.use_current_configuration();
  return mesh;
}

void expect_same_vector(const std::vector<real_t>& a, const std::vector<real_t>& b)
{
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i], b[i], 1.0e-12);
  }
}

#ifdef MESH_HAS_ADAPTIVITY
AdaptivityOptions restart_adaptivity_options()
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
#endif

} // namespace

TEST(MovingMeshRestart, WriteReadMovedMeshRestoresGeometryMotionAndMetadata)
{
  auto mesh = make_moved_quad_mesh();
  const auto path = unique_path(".mmrst");

  moving_mesh_restart::WriteOptions options;
  options.restart_epoch = 42;
  options.motion_backend_state.emplace("backend", "prescribed");
  options.motion_backend_state.emplace("time_level", "accepted_step");
  options.adaptivity_provenance.push_back("none");

  moving_mesh_restart::write(mesh, path, options);
  auto metadata = moving_mesh_restart::inspect(path);
  auto loaded = moving_mesh_restart::read(path);
  const auto registry_path = unique_path(".registry.mmrst");
  MeshIOOptions registry_options;
  registry_options.format = "svmp_restart";
  registry_options.path = registry_path;
  registry_options.kv["restart_epoch"] = "43";
  mesh.save(registry_options);
  auto registry_loaded = MeshBase::load(registry_options);

  EXPECT_EQ(metadata.version, moving_mesh_restart::kSupportedVersion);
  EXPECT_EQ(metadata.restart_epoch, 42u);
  EXPECT_TRUE(metadata.has_current_coordinates);
  EXPECT_EQ(metadata.active_configuration, Configuration::Current);
  ASSERT_EQ(metadata.motion_backend_state.count("backend"), 1u);
  EXPECT_EQ(metadata.motion_backend_state["backend"], "prescribed");
  ASSERT_EQ(metadata.adaptivity_provenance.size(), 1u);
  EXPECT_EQ(metadata.adaptivity_provenance.front(), "none");

  EXPECT_EQ(loaded.active_configuration(), Configuration::Current);
  EXPECT_TRUE(loaded.has_current_coords());
  EXPECT_EQ(loaded.n_vertices(), mesh.n_vertices());
  EXPECT_EQ(loaded.n_cells(), mesh.n_cells());
  EXPECT_EQ(loaded.vertex_gids(), mesh.vertex_gids());
  EXPECT_EQ(loaded.cell_gids(), mesh.cell_gids());
  EXPECT_EQ(loaded.region_label(0), 7);
  EXPECT_TRUE(loaded.has_set(EntityKind::Volume, "mixing_region"));
  expect_same_vector(loaded.X_ref(), mesh.X_ref());
  expect_same_vector(loaded.X_cur(), mesh.X_cur());
  EXPECT_EQ(registry_loaded.active_configuration(), Configuration::Current);
  ASSERT_TRUE(registry_loaded.has_current_coords());
  expect_same_vector(registry_loaded.X_cur(), mesh.X_cur());

  for (const auto role : motion::standard_motion_field_roles()) {
    const std::string name = motion::standard_motion_field_name(role);
    ASSERT_TRUE(loaded.has_field(EntityKind::Vertex, name)) << name;
    const auto src = mesh.field_handle(EntityKind::Vertex, name);
    const auto dst = loaded.field_handle(EntityKind::Vertex, name);
    ASSERT_EQ(loaded.field_components(dst), mesh.field_components(src));
    ASSERT_NE(loaded.field_descriptor(dst), nullptr);
    const auto* a = mesh.field_data_as<const real_t>(src);
    const auto* b = loaded.field_data_as<const real_t>(dst);
    const auto n = loaded.field_entity_count(dst) * loaded.field_components(dst);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(b[i], a[i], 1.0e-12);
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::filesystem::remove(registry_path, ec);
}

TEST(MovingMeshRestart, VersionMismatchFailsClearly)
{
  const auto path = unique_path(".bad.mmrst");
  {
    std::ofstream out(path);
    out << "SVMP_MOVING_MESH_RESTART\n";
    out << "version 999\n";
  }

  EXPECT_THROW((void)moving_mesh_restart::read(path), std::runtime_error);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

#ifdef MESH_HAS_VTK
TEST(MovingMeshRestart, VtkOutputRoundtripRestoresCurrentCoordinatesBeforeConsumers)
{
  auto mesh = make_moved_quad_mesh();
  const auto path = unique_path(".vtu");

  MeshIOOptions options;
  options.format = "vtu";
  options.path = path;
  mesh.save(options);

  auto loaded = MeshBase::load(options);
  ASSERT_TRUE(loaded.has_current_coords());
  EXPECT_EQ(loaded.active_configuration(), Configuration::Current);
  expect_same_vector(loaded.X_cur(), mesh.X_cur());
  EXPECT_FALSE(loaded.has_field(EntityKind::Vertex, "CurrentCoordinates"));

  std::error_code ec;
  std::filesystem::remove(path, ec);
}
#endif

#ifdef MESH_HAS_ADAPTIVITY
TEST(MovingMeshRestart, AdaptedMovingMeshRestartPreservesTransferredMotionState)
{
  MeshBase mesh(2);
  const std::vector<real_t> x_ref = {0.0, 0.0, 1.0, 0.0};
  const std::vector<offset_t> offsets = {0, 2};
  const std::vector<index_t> conn = {0, 1};
  const std::vector<CellShape> shapes = {{CellFamily::Line, 2, 1}};
  mesh.build_from_arrays(2, x_ref, offsets, conn, shapes);
  mesh.finalize();

  const auto handles = motion::attach_motion_fields(mesh, 2);
  auto* displacement = MeshFields::field_data_as<real_t>(mesh, handles.displacement);
  auto* velocity = MeshFields::field_data_as<real_t>(mesh, handles.velocity);
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

  AdaptivityManager manager(restart_adaptivity_options());
  auto result = manager.refine(mesh, {true}, nullptr);
  ASSERT_TRUE(result.success) << result.summary();
  ASSERT_EQ(mesh.n_vertices(), 3u);

  const auto path = unique_path(".adapted.mmrst");
  moving_mesh_restart::WriteOptions options;
  options.restart_epoch = 2;
  options.adaptivity_provenance.push_back(result.summary());
  moving_mesh_restart::write(mesh, path, options);
  auto loaded = moving_mesh_restart::read(path);
  auto metadata = moving_mesh_restart::inspect(path);

  ASSERT_EQ(loaded.n_vertices(), 3u);
  ASSERT_TRUE(loaded.has_current_coords());
  EXPECT_EQ(loaded.active_configuration(), Configuration::Current);
  ASSERT_EQ(metadata.adaptivity_provenance.size(), 1u);
  EXPECT_FALSE(metadata.adaptivity_provenance.front().empty());

  const auto disp =
      loaded.field_handle(EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Displacement));
  const auto vel =
      loaded.field_handle(EntityKind::Vertex, motion::standard_motion_field_name(motion::MotionFieldRole::Velocity));
  ASSERT_NE(disp.id, 0u);
  ASSERT_NE(vel.id, 0u);
  const auto* new_displacement = loaded.field_data_as<const real_t>(disp);
  const auto* new_velocity = loaded.field_data_as<const real_t>(vel);
  ASSERT_NE(new_displacement, nullptr);
  ASSERT_NE(new_velocity, nullptr);

  std::size_t midpoint = loaded.n_vertices();
  for (std::size_t v = 0; v < loaded.n_vertices(); ++v) {
    if (std::abs(loaded.X_ref()[2 * v] - 0.5) < 1.0e-12) {
      midpoint = v;
      break;
    }
  }
  ASSERT_LT(midpoint, loaded.n_vertices());
  EXPECT_NEAR(loaded.X_cur()[2 * midpoint], 1.0, 1.0e-12);
  EXPECT_NEAR(new_displacement[2 * midpoint], 0.5, 1.0e-12);
  EXPECT_NEAR(new_velocity[2 * midpoint], 15.0, 1.0e-12);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}
#endif

} // namespace svmp::test
