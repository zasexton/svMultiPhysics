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
#include "Search/CutCell.h"

#ifdef MESH_HAS_ADAPTIVITY
#include "Adaptivity/AdaptivityManager.h"
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

MeshBase make_curved_quad_p3_mesh()
{
  constexpr int p = 3;
  std::vector<std::array<real_t, 3>> pts;
  pts.reserve(16);

  const std::array<std::array<real_t, 3>, 4> corners{{
      {{-1.0, -1.0, 0.0}},
      {{ 1.0, -1.0, 0.0}},
      {{ 1.0,  1.0, 0.0}},
      {{-1.0,  1.0, 0.0}},
  }};
  for (const auto& pt : corners) {
    pts.push_back(pt);
  }

  const std::array<std::array<int, 2>, 4> edges = {{{0, 1}, {1, 2}, {2, 3}, {3, 0}}};
  for (const auto& edge : edges) {
    for (int k = 1; k < p; ++k) {
      const real_t t = static_cast<real_t>(k) / static_cast<real_t>(p);
      const auto& a = corners[static_cast<std::size_t>(edge[0])];
      const auto& b = corners[static_cast<std::size_t>(edge[1])];
      pts.push_back({(1.0 - t) * a[0] + t * b[0],
                     (1.0 - t) * a[1] + t * b[1],
                     0.10 + 0.02 * static_cast<real_t>(k)});
    }
  }

  for (int i = 1; i < p; ++i) {
    for (int j = 1; j < p; ++j) {
      pts.push_back({-1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(p),
                     -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(p),
                     0.20 + 0.03 * static_cast<real_t>(i * j)});
    }
  }

  std::vector<real_t> x_ref;
  x_ref.reserve(pts.size() * 3u);
  for (const auto& pt : pts) {
    x_ref.push_back(pt[0]);
    x_ref.push_back(pt[1]);
    x_ref.push_back(pt[2]);
  }

  std::vector<offset_t> offsets = {0, static_cast<offset_t>(pts.size())};
  std::vector<index_t> conn(pts.size());
  for (std::size_t i = 0; i < conn.size(); ++i) {
    conn[i] = static_cast<index_t>(i);
  }
  std::vector<CellShape> shapes = {{CellFamily::Quad, 4, p}};

  MeshBase mesh;
  mesh.build_from_arrays(3, x_ref, offsets, conn, shapes);
  mesh.finalize();

  auto x_cur = mesh.X_ref();
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    x_cur[3 * v + 0] += 0.01 * static_cast<real_t>(v);
    x_cur[3 * v + 2] += 0.15 + 0.005 * static_cast<real_t>(v);
  }
  mesh.set_current_coords(x_cur);
  mesh.use_current_configuration();

  auto handles = motion::attach_motion_fields(mesh, 3);
  auto* displacement = MeshFields::field_data_as<real_t>(mesh, handles.displacement);
  if (!displacement) {
    throw std::runtime_error("failed to attach high-order displacement field");
  }
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    for (int d = 0; d < 3; ++d) {
      displacement[3 * v + static_cast<std::size_t>(d)] =
          mesh.X_cur()[3 * v + static_cast<std::size_t>(d)] -
          mesh.X_ref()[3 * v + static_cast<std::size_t>(d)];
    }
  }

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

TEST(MovingMeshRestart, CutRegistryAndClassificationMetadataRoundtrip)
{
  auto mesh = make_moved_quad_mesh();

  search::EmbeddedGeometryDescriptor plane;
  plane.kind = search::EmbeddedGeometryKind::Plane;
  plane.origin = {{0.5, 0.0, 0.0}};
  plane.normal = {{1.0, 0.0, 0.0}};
  plane.geometry_epoch = 6;
  plane.revisions.geometry_epoch = 6;
  plane.provenance.persistent_id = "restart-cut-plane";
  plane.provenance.name = "restart cut plane";
  plane.provenance.provenance_epoch = 6;

  search::EmbeddedGeometryRegistry registry;
  registry.register_geometry(plane);

  search::CutClassificationOptions cut_options;
  cut_options.classify_faces = false;
  cut_options.classify_edges = false;
  cut_options.fe_layout_revision = 12;
  const auto cut_map = search::classify_embedded_geometry(mesh, plane, cut_options);

  const auto path = unique_path(".cut.mmrst");
  moving_mesh_restart::WriteOptions options;
  options.restart_epoch = 77;
  options.embedded_geometry_registry = search::make_embedded_geometry_restart_records(registry);
  options.cut_classification_state.push_back(
      search::make_cut_classification_restart_record(cut_map));

  moving_mesh_restart::write(mesh, path, options);
  const auto metadata = moving_mesh_restart::inspect(path);

  EXPECT_EQ(metadata.version, moving_mesh_restart::kSupportedVersion);
  ASSERT_EQ(metadata.embedded_geometry_registry.size(), 1u);
  EXPECT_EQ(metadata.embedded_geometry_registry[0].persistent_id, "restart-cut-plane");
  EXPECT_EQ(metadata.embedded_geometry_registry[0].revisions.geometry_epoch, 6u);
  ASSERT_EQ(metadata.cut_classification_state.size(), 1u);
  EXPECT_EQ(metadata.cut_classification_state[0].provenance.persistent_id, "restart-cut-plane");
  EXPECT_EQ(metadata.cut_classification_state[0].fe_layout_revision, 12u);
  EXPECT_GT(metadata.cut_classification_state[0].cut_cell_count, 0u);

  const auto restored_registry =
      search::restore_embedded_geometry_registry(metadata.embedded_geometry_registry);
  ASSERT_TRUE(restored_registry.contains("restart-cut-plane"));
  const auto rebuilt = restored_registry.classify_active(mesh, cut_options);
  ASSERT_EQ(rebuilt.size(), 1u);
  EXPECT_EQ(search::make_cut_classification_restart_record(rebuilt[0]).cut_topology_revision,
            metadata.cut_classification_state[0].cut_topology_revision);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(MovingMeshRestart, BooleanCutCompositionProvenanceRoundtripsThroughRestart)
{
  auto mesh = make_moved_quad_mesh();

  search::EmbeddedGeometryDescriptor left;
  left.kind = search::EmbeddedGeometryKind::Plane;
  left.origin = {{0.5, 0.0, 0.0}};
  left.normal = {{1.0, 0.0, 0.0}};
  left.geometry_epoch = 21;
  left.revisions.geometry_epoch = 21;
  left.provenance.persistent_id = "restart-boolean-plane";
  left.provenance.name = "restart boolean plane";
  left.provenance.provenance_epoch = 21;

  search::EmbeddedGeometryDescriptor right;
  right.kind = search::EmbeddedGeometryKind::Sphere;
  right.origin = {{0.0, 0.0, 0.0}};
  right.normal = {{1.0, 0.0, 0.0}};
  right.radius = 10.0;
  right.geometry_epoch = 22;
  right.revisions.geometry_epoch = 22;
  right.provenance.persistent_id = "restart-enclosing-sphere";
  right.provenance.name = "restart enclosing sphere";
  right.provenance.provenance_epoch = 22;

  search::EmbeddedGeometryDescriptor band;
  band.kind = search::EmbeddedGeometryKind::BooleanComposite;
  band.boolean_operation = search::EmbeddedGeometryBooleanOperation::Intersection;
  band.geometry_epoch = 23;
  band.revisions.geometry_epoch = 23;
  band.provenance.persistent_id = "restart-inner-composite";
  band.provenance.name = "restart inner composite";
  band.provenance.provenance_epoch = 23;
  band.children = {left, right};

  search::EmbeddedGeometryDescriptor small_sphere;
  small_sphere.kind = search::EmbeddedGeometryKind::Sphere;
  small_sphere.origin = {{0.0, 0.0, 0.0}};
  small_sphere.radius = 0.05;
  small_sphere.geometry_epoch = 24;
  small_sphere.revisions.geometry_epoch = 24;
  small_sphere.provenance.persistent_id = "restart-small-nested-sphere";
  small_sphere.provenance.name = "restart small nested sphere";
  small_sphere.provenance.provenance_epoch = 24;

  search::EmbeddedGeometryDescriptor composed;
  composed.kind = search::EmbeddedGeometryKind::BooleanComposite;
  composed.boolean_operation = search::EmbeddedGeometryBooleanOperation::Union;
  composed.geometry_epoch = 25;
  composed.revisions.geometry_epoch = 25;
  composed.provenance.persistent_id = "restart-nested-composition";
  composed.provenance.name = "restart nested composition";
  composed.provenance.provenance_epoch = 25;
  composed.children = {band, small_sphere};

  search::EmbeddedGeometryRegistry registry;
  registry.register_geometry(composed);

  search::CutClassificationOptions cut_options;
  cut_options.classify_faces = false;
  cut_options.classify_edges = false;
  cut_options.fe_layout_revision = 34;
  const auto cut_map = search::classify_embedded_geometry(mesh, composed, cut_options);
  const auto topology = search::reconstruct_cut_topology(mesh, cut_map);
  ASSERT_FALSE(topology.side_regions.empty());

  const auto path = unique_path(".cut.boolean.mmrst");
  moving_mesh_restart::WriteOptions options;
  options.restart_epoch = 88;
  options.embedded_geometry_registry = search::make_embedded_geometry_restart_records(registry);
  options.cut_classification_state.push_back(
      search::make_cut_classification_restart_record(cut_map, topology));

  moving_mesh_restart::write(mesh, path, options);
  const auto metadata = moving_mesh_restart::inspect(path);

  ASSERT_EQ(metadata.embedded_geometry_registry.size(), 1u);
  EXPECT_EQ(metadata.embedded_geometry_registry[0].persistent_id,
            "restart-nested-composition");
  EXPECT_EQ(metadata.embedded_geometry_registry[0].boolean_operation,
            search::EmbeddedGeometryBooleanOperation::Union);
  ASSERT_EQ(metadata.embedded_geometry_registry[0].children.size(), 2u);
  EXPECT_EQ(metadata.embedded_geometry_registry[0].children[0].persistent_id,
            "restart-inner-composite");
  EXPECT_EQ(metadata.embedded_geometry_registry[0].children[1].persistent_id,
            "restart-small-nested-sphere");
  ASSERT_EQ(metadata.embedded_geometry_registry[0].children[0].children.size(), 2u);
  EXPECT_EQ(metadata.embedded_geometry_registry[0].children[0].children[0].persistent_id,
            "restart-boolean-plane");
  EXPECT_EQ(metadata.embedded_geometry_registry[0].children[0].children[1].persistent_id,
            "restart-enclosing-sphere");

  ASSERT_EQ(metadata.cut_classification_state.size(), 1u);
  const auto& cut_restart = metadata.cut_classification_state[0];
  EXPECT_TRUE(cut_restart.is_composed_region);
  EXPECT_EQ(cut_restart.composition_operation,
            search::EmbeddedGeometryBooleanOperation::Union);
  EXPECT_EQ(cut_restart.fe_layout_revision, 34u);
  ASSERT_EQ(cut_restart.composition_children.size(), 4u);
  EXPECT_EQ(cut_restart.composition_children[0].parent_persistent_id,
            "restart-nested-composition");
  EXPECT_EQ(cut_restart.composition_children[0].provenance.persistent_id,
            "restart-inner-composite");
  EXPECT_EQ(cut_restart.composition_children[1].parent_persistent_id,
            "restart-inner-composite");
  EXPECT_EQ(cut_restart.composition_children[1].provenance.persistent_id,
            "restart-boolean-plane");
  EXPECT_EQ(cut_restart.composition_children[2].provenance.persistent_id,
            "restart-enclosing-sphere");
  EXPECT_EQ(cut_restart.composition_children[3].provenance.persistent_id,
            "restart-small-nested-sphere");
  EXPECT_EQ(cut_restart.side_regions.size(), topology.side_regions.size());
  ASSERT_FALSE(cut_restart.side_regions.empty());
  EXPECT_EQ(cut_restart.side_regions[0].provenance.persistent_id,
            "restart-nested-composition");

  const auto restored_registry =
      search::restore_embedded_geometry_registry(metadata.embedded_geometry_registry);
  ASSERT_TRUE(restored_registry.contains("restart-nested-composition"));
  const auto rebuilt = restored_registry.classify_active(mesh, cut_options);
  ASSERT_EQ(rebuilt.size(), 1u);
  const auto rebuilt_topology = search::reconstruct_cut_topology(mesh, rebuilt[0]);
  const auto rebuilt_restart =
      search::make_cut_classification_restart_record(rebuilt[0], rebuilt_topology);
  EXPECT_EQ(rebuilt_restart.composition_children.size(),
            cut_restart.composition_children.size());
  EXPECT_EQ(rebuilt_restart.side_regions.size(), cut_restart.side_regions.size());
  EXPECT_EQ(rebuilt_restart.cut_topology_revision, cut_restart.cut_topology_revision);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(MovingMeshRestart, RejectedTrialCutStateIsNotWrittenAsAcceptedRestartState)
{
  auto mesh = make_moved_quad_mesh();

  search::EmbeddedGeometryDescriptor accepted_plane;
  accepted_plane.kind = search::EmbeddedGeometryKind::Plane;
  accepted_plane.origin = {{0.5, 0.0, 0.0}};
  accepted_plane.normal = {{1.0, 0.0, 0.0}};
  accepted_plane.geometry_epoch = 10;
  accepted_plane.revisions.geometry_epoch = 10;
  accepted_plane.provenance.persistent_id = "accepted-cut-plane";
  accepted_plane.provenance.provenance_epoch = 10;

  search::CutClassificationOptions cut_options;
  cut_options.classify_faces = false;
  cut_options.classify_edges = false;
  auto accepted_map = search::classify_embedded_geometry(mesh, accepted_plane, cut_options);
  accepted_map.accept_trial();
  const auto accepted_restart = search::make_cut_classification_restart_record(accepted_map);

  search::CutClassificationTransaction tx(accepted_map);
  auto rejected_plane = accepted_plane;
  rejected_plane.origin = {{5.0, 0.0, 0.0}};
  rejected_plane.geometry_epoch = 99;
  rejected_plane.revisions.geometry_epoch = 99;
  tx.stage(search::classify_embedded_geometry(mesh, rejected_plane, cut_options));
  tx.rollback();

  search::EmbeddedGeometryRegistry registry;
  registry.register_geometry(accepted_plane);

  const auto path = unique_path(".cut.rollback.mmrst");
  moving_mesh_restart::WriteOptions options;
  options.embedded_geometry_registry = search::make_embedded_geometry_restart_records(registry);
  options.cut_classification_state.push_back(accepted_restart);
  moving_mesh_restart::write(mesh, path, options);
  const auto metadata = moving_mesh_restart::inspect(path);

  ASSERT_EQ(metadata.embedded_geometry_registry.size(), 1u);
  EXPECT_EQ(metadata.embedded_geometry_registry[0].revisions.geometry_epoch, 10u);
  ASSERT_EQ(metadata.cut_classification_state.size(), 1u);
  EXPECT_EQ(metadata.cut_classification_state[0].embedded_geometry_epoch, 10u);
  EXPECT_EQ(metadata.cut_classification_state[0].cut_topology_revision,
            accepted_restart.cut_topology_revision);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(MovingMeshRestart, ReferenceRebaseMetadataRoundtrips)
{
  auto mesh = make_moved_quad_mesh();
  const auto expected_reference = mesh.X_cur();
  ASSERT_FALSE(expected_reference.empty());

  ReferenceRebaseOptions rebase_options;
  rebase_options.current_policy = ReferenceRebaseCurrentPolicy::ClearCurrent;
  rebase_options.motion_policy = ReferenceRebaseMotionPolicy::ResetDisplacementLikeFields;
  rebase_options.active_configuration_after = Configuration::Reference;
  mesh.rebase_reference_to_current(rebase_options);

  const auto path = unique_path(".rebase.mmrst");
  moving_mesh_restart::write(mesh, path);
  const auto metadata = moving_mesh_restart::inspect(path);
  auto loaded = moving_mesh_restart::read(path);

  EXPECT_EQ(metadata.version, moving_mesh_restart::kSupportedVersion);
  EXPECT_FALSE(metadata.has_current_coordinates);
  EXPECT_EQ(metadata.active_configuration, Configuration::Reference);
  EXPECT_EQ(metadata.mesh_revisions.reference_rebase, mesh.reference_rebase_epoch());
  EXPECT_EQ(metadata.reference_rebase.mode, ReferenceConfigurationMode::UpdatedLagrangianRebased);
  EXPECT_EQ(metadata.reference_rebase.last_source, ReferenceRebaseSource::CurrentConfiguration);
  EXPECT_EQ(metadata.reference_rebase.epoch, mesh.reference_rebase_info().epoch);

  EXPECT_EQ(loaded.reference_configuration_mode(),
            ReferenceConfigurationMode::UpdatedLagrangianRebased);
  EXPECT_EQ(loaded.reference_rebase_info().last_source,
            ReferenceRebaseSource::CurrentConfiguration);
  EXPECT_EQ(loaded.reference_rebase_info().epoch, mesh.reference_rebase_info().epoch);
  EXPECT_EQ(loaded.reference_rebase_epoch(), mesh.reference_rebase_epoch());
  EXPECT_EQ(loaded.active_configuration(), Configuration::Reference);
  EXPECT_FALSE(loaded.has_current_coords());
  expect_same_vector(loaded.X_ref(), expected_reference);

  std::error_code ec;
  std::filesystem::remove(path, ec);
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

TEST(MovingMeshRestart, HighOrderCurvedGeometryDescriptorAndMotionFieldsRoundtrip)
{
  auto mesh = make_curved_quad_p3_mesh();
  const auto path = unique_path(".curved-p3.mmrst");

  moving_mesh_restart::WriteOptions options;
  options.restart_epoch = 15;
  options.motion_backend_state.emplace("geometry_storage", "vertex_coordinates");
  moving_mesh_restart::write(mesh, path, options);

  const auto metadata = moving_mesh_restart::inspect(path);
  auto loaded = moving_mesh_restart::read(path);

  EXPECT_EQ(metadata.version, moving_mesh_restart::kSupportedVersion);
  EXPECT_EQ(metadata.geometry_order.storage, GeometryDofStorage::VertexCoordinates);
  EXPECT_EQ(metadata.geometry_order.max_order, 3);
  EXPECT_TRUE(metadata.geometry_order.has_high_order);
  EXPECT_EQ(metadata.geometry_order.reference_dofs, mesh.n_vertices());
  EXPECT_EQ(metadata.geometry_order.current_dofs, mesh.n_vertices());

  EXPECT_EQ(loaded.geometry_order_descriptor().max_order, 3);
  EXPECT_TRUE(loaded.has_high_order_geometry());
  EXPECT_EQ(loaded.cell_geometry_dofs(0), mesh.cell_geometry_dofs(0));
  EXPECT_EQ(loaded.cell_edge_geometry_dofs(0, 0).size(), 4u);
  EXPECT_EQ(loaded.cell_face_geometry_dofs(0, 0).size(), 4u);
  EXPECT_EQ(loaded.cell_interior_geometry_dofs(0).size(), 4u);
  expect_same_vector(loaded.X_ref(), mesh.X_ref());
  expect_same_vector(loaded.X_cur(), mesh.X_cur());

  const auto disp_name = motion::standard_motion_field_name(motion::MotionFieldRole::Displacement);
  ASSERT_TRUE(loaded.has_field(EntityKind::Vertex, disp_name));
  const auto src = mesh.field_handle(EntityKind::Vertex, disp_name);
  const auto dst = loaded.field_handle(EntityKind::Vertex, disp_name);
  const auto* a = mesh.field_data_as<const real_t>(src);
  const auto* b = loaded.field_data_as<const real_t>(dst);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  for (std::size_t i = 0; i < loaded.n_vertices() * 3u; ++i) {
    EXPECT_NEAR(b[i], a[i], 1.0e-12);
  }

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
