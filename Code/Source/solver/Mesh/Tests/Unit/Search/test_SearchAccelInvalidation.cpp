/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"

#include "Core/MeshBase.h"
#include "Search/MeshSearch.h"
#include "Topology/CellShape.h"

namespace svmp {
namespace test {

static MeshBase create_unit_tet_mesh() {
  std::vector<real_t> X_ref = {
      0.0, 0.0, 0.0,  // Vertex 0
      1.0, 0.0, 0.0,  // Vertex 1
      0.0, 1.0, 0.0,  // Vertex 2
      0.0, 0.0, 1.0   // Vertex 3
  };

  std::vector<offset_t> offs = {0, 4};
  std::vector<index_t> conn = {0, 1, 2, 3};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Tetra;
  shapes[0].order = 1;
  shapes[0].num_corners = 4;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

TEST(SearchAccelInvalidationTest, InvalidatesOnGeometryChanged) {
  MeshBase mesh = create_unit_tet_mesh();

  MeshSearch::build_search_structure(mesh);
  ASSERT_TRUE(MeshSearch::has_search_structure(mesh));

  mesh.set_vertex_coords(0, {2.0, 2.0, 2.0});
  EXPECT_FALSE(MeshSearch::has_search_structure(mesh));
}

TEST(SearchAccelInvalidationTest, InvalidatesOnTopologyChanged) {
  MeshBase mesh = create_unit_tet_mesh();

  MeshSearch::build_search_structure(mesh);
  ASSERT_TRUE(MeshSearch::has_search_structure(mesh));

  mesh.clear();
  EXPECT_FALSE(MeshSearch::has_search_structure(mesh));
}

TEST(SearchAccelInvalidationTest, InvalidatesOnLabelsChanged) {
  MeshBase mesh = create_unit_tet_mesh();

  MeshSearch::build_search_structure(mesh);
  ASSERT_TRUE(MeshSearch::has_search_structure(mesh));

  mesh.set_region_label(0, 17);
  EXPECT_FALSE(MeshSearch::has_search_structure(mesh));
}

TEST(SearchAccelInvalidationTest, InvalidatesOnNumberingChanged) {
  MeshBase mesh = create_unit_tet_mesh();

  MeshSearch::build_search_structure(mesh);
  ASSERT_TRUE(MeshSearch::has_search_structure(mesh));

  mesh.set_cell_gids({101});
  EXPECT_FALSE(MeshSearch::has_search_structure(mesh));
}

TEST(SearchAccelInvalidationTest, ActiveConfigurationSwitchIsEpochVersioned) {
  MeshBase mesh = create_unit_tet_mesh();
  const auto initial_epoch = mesh.active_configuration_epoch();
  const auto initial_geometry_revision = mesh.geometry_revision();

  std::vector<real_t> current = mesh.X_ref();
  current[0] += 0.25;
  mesh.set_current_coords(current);
  EXPECT_GT(mesh.geometry_revision(), initial_geometry_revision);
  EXPECT_EQ(mesh.active_configuration_epoch(), initial_epoch);

  const auto after_current_coords_epoch = mesh.active_configuration_epoch();
  mesh.use_current_configuration();
  EXPECT_GT(mesh.active_configuration_epoch(), after_current_coords_epoch);

  const auto after_current_epoch = mesh.active_configuration_epoch();
  mesh.use_reference_configuration();
  EXPECT_GT(mesh.active_configuration_epoch(), after_current_epoch);
}

} // namespace test
} // namespace svmp
