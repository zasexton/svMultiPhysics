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

} // namespace test
} // namespace svmp

