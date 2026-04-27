#include "Search/MultiMeshInterface.h"

#include "Core/MeshBase.h"
#include "Topology/CellShape.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

constexpr svmp::label_t kSourceLabel = 41;
constexpr svmp::label_t kTargetLabel = 42;

svmp::CellShape line_shape() {
  return {svmp::CellFamily::Line, 2, 1};
}

svmp::CellShape quad_shape() {
  return {svmp::CellFamily::Quad, 4, 1};
}

svmp::CellShape triangle_shape() {
  return {svmp::CellFamily::Triangle, 3, 1};
}

svmp::MeshBase make_single_quad_mesh(double x0, double x1, double y0, double y1,
                                     svmp::index_t labelled_face, svmp::label_t label) {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {
      x0, y0, 0.0,
      x1, y0, 0.0,
      x1, y1, 0.0,
      x0, y1, 0.0};
  const std::vector<svmp::offset_t> cell_offsets = {0, 4};
  const std::vector<svmp::index_t> cells = {0, 1, 2, 3};
  mesh.build_from_arrays(3, x_ref, cell_offsets, cells, {quad_shape()});

  const std::vector<svmp::CellShape> face_shapes(4, line_shape());
  const std::vector<svmp::offset_t> face_offsets = {0, 2, 4, 6, 8};
  const std::vector<svmp::index_t> faces = {
      0, 1,
      1, 2,
      2, 3,
      3, 0};
  const std::vector<std::array<svmp::index_t, 2>> face2cell = {
      {{0, svmp::INVALID_INDEX}},
      {{0, svmp::INVALID_INDEX}},
      {{0, svmp::INVALID_INDEX}},
      {{0, svmp::INVALID_INDEX}}};
  mesh.set_faces_from_arrays(face_shapes, face_offsets, faces, face2cell);
  mesh.set_face_gids({10, 11, 12, 13});
  mesh.set_boundary_label(labelled_face, label);
  return mesh;
}

svmp::MeshBase make_split_target_mesh() {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {
      1.1, 0.0, 0.0,
      1.4, 0.0, 0.0,
      1.4, 0.5, 0.0,
      1.1, 0.5, 0.0,
      1.4, 1.0, 0.0,
      1.1, 1.0, 0.0};
  const std::vector<svmp::offset_t> cell_offsets = {0, 4, 8};
  const std::vector<svmp::index_t> cells = {
      0, 1, 2, 3,
      3, 2, 4, 5};
  mesh.build_from_arrays(3, x_ref, cell_offsets, cells, {quad_shape(), quad_shape()});

  const std::vector<svmp::CellShape> face_shapes(2, line_shape());
  const std::vector<svmp::offset_t> face_offsets = {0, 2, 4};
  const std::vector<svmp::index_t> faces = {3, 0, 5, 3};
  const std::vector<std::array<svmp::index_t, 2>> face2cell = {
      {{0, svmp::INVALID_INDEX}},
      {{1, svmp::INVALID_INDEX}}};
  mesh.set_faces_from_arrays(face_shapes, face_offsets, faces, face2cell);
  mesh.set_face_gids({20, 21});
  mesh.set_boundary_label(0, kTargetLabel);
  mesh.set_boundary_label(1, kTargetLabel);
  return mesh;
}

svmp::MeshBase make_quad_surface(double x) {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {
      x, 0.0, 0.0,
      x, 1.0, 0.0,
      x, 1.0, 1.0,
      x, 0.0, 1.0};
  const std::vector<svmp::offset_t> cell_offsets = {0};
  const std::vector<svmp::index_t> cells;
  const std::vector<svmp::CellShape> cell_shapes;
  mesh.build_from_arrays(3, x_ref, cell_offsets, cells, cell_shapes);

  const std::vector<svmp::offset_t> face_offsets = {0, 4};
  const std::vector<svmp::index_t> faces = {0, 1, 2, 3};
  mesh.set_faces_from_arrays({quad_shape()}, face_offsets, faces, {{{svmp::INVALID_INDEX, svmp::INVALID_INDEX}}});
  mesh.set_face_gids({30});
  mesh.set_boundary_label(0, kSourceLabel);
  return mesh;
}

svmp::MeshBase make_split_triangle_surface(double x) {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {
      x, 0.0, 0.0,
      x, 1.0, 0.0,
      x, 1.0, 1.0,
      x, 0.0, 1.0};
  const std::vector<svmp::offset_t> cell_offsets = {0};
  const std::vector<svmp::index_t> cells;
  const std::vector<svmp::CellShape> cell_shapes;
  mesh.build_from_arrays(3, x_ref, cell_offsets, cells, cell_shapes);

  const std::vector<svmp::offset_t> face_offsets = {0, 3, 6};
  const std::vector<svmp::index_t> faces = {0, 1, 2, 0, 2, 3};
  const std::vector<std::array<svmp::index_t, 2>> face2cell = {
      {{svmp::INVALID_INDEX, svmp::INVALID_INDEX}},
      {{svmp::INVALID_INDEX, svmp::INVALID_INDEX}}};
  mesh.set_faces_from_arrays({triangle_shape(), triangle_shape()}, face_offsets, faces, face2cell);
  mesh.set_face_gids({40, 41});
  mesh.set_boundary_label(0, kTargetLabel);
  mesh.set_boundary_label(1, kTargetLabel);
  return mesh;
}

svmp::search::InterfaceMap build_quad_to_split_map(
    const svmp::MeshBase& source,
    const svmp::MeshBase& target,
    svmp::Configuration cfg = svmp::Configuration::Reference) {
  svmp::search::InterfaceSearchRegistry registry;
  registry.register_interface(
      "wall",
      svmp::search::InterfaceSideSpec::from_mesh(source, kSourceLabel, cfg, "source"),
      svmp::search::InterfaceSideSpec::from_mesh(target, kTargetLabel, cfg, "target"),
      1.0);
  return registry.build_trial_map("wall");
}

} // namespace

TEST(MultiMeshInterfaceTest, BuildsNonmatching2DBoundaryPairWithFaceAndCellCoordinates) {
  auto source = make_single_quad_mesh(0.0, 1.0, 0.0, 1.0, 1, kSourceLabel);
  auto target = make_split_target_mesh();

  const auto map = build_quad_to_split_map(source, target);

  ASSERT_EQ(map.state, svmp::search::InterfaceMapState::Trial);
  ASSERT_EQ(map.pairs.size(), 1u);
  const auto& pair = map.pairs.front();
  EXPECT_EQ(pair.source_face, 1);
  EXPECT_TRUE(pair.target_face == 0 || pair.target_face == 1);
  EXPECT_NEAR(pair.distance, 0.1, 1e-12);
  EXPECT_TRUE(std::isfinite(pair.source_face_xi[0]));
  EXPECT_TRUE(std::isfinite(pair.target_face_xi[0]));
  EXPECT_NE(pair.source_cell, svmp::INVALID_INDEX);
  EXPECT_NEAR(pair.source_point[0], 1.0, 1e-12);
  EXPECT_NEAR(pair.target_point[0], 1.1, 1e-12);
  EXPECT_TRUE(map.valid_for_current_revisions());
}

TEST(MultiMeshInterfaceTest, BuildsNonmatching3DSurfacePair) {
  auto source = make_quad_surface(0.0);
  auto target = make_split_triangle_surface(0.2);

  const auto map = build_quad_to_split_map(source, target);

  ASSERT_EQ(map.pairs.size(), 1u);
  const auto& pair = map.pairs.front();
  EXPECT_EQ(pair.source_face, 0);
  EXPECT_TRUE(pair.target_face == 0 || pair.target_face == 1);
  EXPECT_NEAR(pair.distance, 0.2, 1e-12);
  EXPECT_GT(pair.source_measure, 0.0);
  EXPECT_GT(pair.target_measure, 0.0);
  EXPECT_TRUE(std::isfinite(pair.target_face_xi[0]));
  EXPECT_TRUE(std::isfinite(pair.target_face_xi[1]));
}

TEST(MultiMeshInterfaceTest, InvalidatesWhenEitherSideMoves) {
  auto source = make_single_quad_mesh(0.0, 1.0, 0.0, 1.0, 1, kSourceLabel);
  auto target = make_split_target_mesh();
  source.set_current_coords(source.X_ref());
  target.set_current_coords(target.X_ref());

  auto map = build_quad_to_split_map(source, target, svmp::Configuration::Current);
  ASSERT_TRUE(map.valid_for_current_revisions());

  auto moved = target.X_ref();
  for (std::size_t i = 0; i < moved.size(); i += 3) {
    moved[i] += 0.05;
  }
  target.set_current_coords(moved);
  EXPECT_FALSE(map.valid_for_current_revisions());
}

TEST(MultiMeshInterfaceTest, InvalidatesWhenEitherSideChangesTopology) {
  auto source = make_single_quad_mesh(0.0, 1.0, 0.0, 1.0, 1, kSourceLabel);
  auto target = make_split_target_mesh();
  auto map = build_quad_to_split_map(source, target);
  ASSERT_TRUE(map.valid_for_current_revisions());

  const std::vector<svmp::CellShape> face_shapes(1, line_shape());
  const std::vector<svmp::offset_t> face_offsets = {0, 2};
  const std::vector<svmp::index_t> faces = {3, 0};
  const std::vector<std::array<svmp::index_t, 2>> face2cell = {{{0, svmp::INVALID_INDEX}}};
  target.set_faces_from_arrays(face_shapes, face_offsets, faces, face2cell);
  EXPECT_FALSE(map.valid_for_current_revisions());
}

TEST(MultiMeshInterfaceTest, InvalidatesWhenEitherSideRenumbersOrChangesLabels) {
  auto source = make_single_quad_mesh(0.0, 1.0, 0.0, 1.0, 1, kSourceLabel);
  auto target = make_split_target_mesh();

  auto numbering_map = build_quad_to_split_map(source, target);
  ASSERT_TRUE(numbering_map.valid_for_current_revisions());
  target.set_face_gids({200, 201});
  EXPECT_FALSE(numbering_map.valid_for_current_revisions());

  target = make_split_target_mesh();
  auto label_map = build_quad_to_split_map(source, target);
  ASSERT_TRUE(label_map.valid_for_current_revisions());
  target.set_boundary_label(0, kTargetLabel + 1);
  EXPECT_FALSE(label_map.valid_for_current_revisions());
}

TEST(MultiMeshInterfaceTest, RegistryCommitAndRollbackAreExplicit) {
  auto source = make_single_quad_mesh(0.0, 1.0, 0.0, 1.0, 1, kSourceLabel);
  auto target = make_split_target_mesh();
  svmp::search::InterfaceSearchRegistry registry;
  registry.register_interface(
      "wall",
      svmp::search::InterfaceSideSpec::from_mesh(source, kSourceLabel),
      svmp::search::InterfaceSideSpec::from_mesh(target, kTargetLabel),
      1.0);

  auto map = registry.build_trial_map("wall");
  ASSERT_EQ(map.state, svmp::search::InterfaceMapState::Trial);
  registry.commit_map(map);
  ASSERT_NE(registry.committed_map("wall"), nullptr);
  EXPECT_EQ(registry.committed_map("wall")->state, svmp::search::InterfaceMapState::Committed);
  EXPECT_TRUE(registry.committed_map_valid("wall"));

  registry.rollback_committed_map("wall");
  EXPECT_EQ(registry.committed_map("wall"), nullptr);

  map.rollback_trial();
  EXPECT_EQ(map.state, svmp::search::InterfaceMapState::RolledBack);
  EXPECT_TRUE(map.pairs.empty());
}
