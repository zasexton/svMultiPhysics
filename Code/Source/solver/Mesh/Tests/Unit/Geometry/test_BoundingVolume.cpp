/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"
#include "Geometry/BoundingVolume.h"
#include "Core/MeshBase.h"
#include "Topology/CellShape.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace svmp {
namespace test {

namespace {
constexpr real_t tol = 1e-12;

inline bool approx(real_t a, real_t b, real_t t = tol) { return std::abs(a - b) < t; }

MeshBase create_two_tet_mesh() {
  // Two tetrahedra separated in +x for clean AABB splits.
  std::vector<real_t> X_ref = {
      // Cell 0
      0.0, 0.0, 0.0,  // 0
      1.0, 0.0, 0.0,  // 1
      0.0, 1.0, 0.0,  // 2
      0.0, 0.0, 1.0,  // 3
      // Cell 1 (shifted)
      10.0, 0.0, 0.0,  // 4
      11.0, 0.0, 0.0,  // 5
      10.0, 1.0, 0.0,  // 6
      10.0, 0.0, 1.0   // 7
  };

  std::vector<offset_t> offs = {0, 4, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes(2);
  shapes[0].family = CellFamily::Tetra;
  shapes[0].order = 1;
  shapes[0].num_corners = 4;
  shapes[1].family = CellFamily::Tetra;
  shapes[1].order = 1;
  shapes[1].num_corners = 4;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}

MeshBase create_unit_hex_mesh(bool finalize_topology = true) {
  std::vector<real_t> X_ref = {
      0.0, 0.0, 0.0,  // 0
      1.0, 0.0, 0.0,  // 1
      1.0, 1.0, 0.0,  // 2
      0.0, 1.0, 0.0,  // 3
      0.0, 0.0, 1.0,  // 4
      1.0, 0.0, 1.0,  // 5
      1.0, 1.0, 1.0,  // 6
      0.0, 1.0, 1.0   // 7
  };

  std::vector<offset_t> offs = {0, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Hex;
  shapes[0].order = 1;
  shapes[0].num_corners = 8;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  if (finalize_topology) {
    mesh.finalize();
  }
  return mesh;
}

#ifdef MESH_HAS_EIGEN
MeshBase create_box_hex_mesh() {
  // Rectangular box: 2 x 1 x 0.5 (distinct PCA eigenvalues).
  std::vector<real_t> X_ref = {
      0.0, 0.0, 0.0,  // 0
      2.0, 0.0, 0.0,  // 1
      2.0, 1.0, 0.0,  // 2
      0.0, 1.0, 0.0,  // 3
      0.0, 0.0, 0.5,  // 4
      2.0, 0.0, 0.5,  // 5
      2.0, 1.0, 0.5,  // 6
      0.0, 1.0, 0.5   // 7
  };

  std::vector<offset_t> offs = {0, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Hex;
  shapes[0].order = 1;
  shapes[0].num_corners = 8;

  MeshBase mesh;
  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  return mesh;
}
#endif
} // namespace

TEST(BoundingVolumeTest, AABBIntersectsAndIntersection) {
  AABB a({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  AABB b({0.5, 0.5, 0.5}, {2.0, 2.0, 2.0});
  EXPECT_TRUE(a.intersects(b));

  const auto i = a.intersection(b);
  EXPECT_TRUE(i.is_valid());
  EXPECT_TRUE(approx(i.min[0], 0.5));
  EXPECT_TRUE(approx(i.min[1], 0.5));
  EXPECT_TRUE(approx(i.min[2], 0.5));
  EXPECT_TRUE(approx(i.max[0], 1.0));
  EXPECT_TRUE(approx(i.max[1], 1.0));
  EXPECT_TRUE(approx(i.max[2], 1.0));
  EXPECT_TRUE(approx(i.volume(), 0.125));

  AABB c({2.0, 2.0, 2.0}, {3.0, 3.0, 3.0});
  EXPECT_FALSE(a.intersects(c));
  EXPECT_FALSE(a.intersection(c).is_valid());
}

TEST(BoundingVolumeTest, BoundingSphereFromAABBContainsCorners) {
  AABB box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  BoundingSphere s(box);
  EXPECT_TRUE(approx(s.center[0], 0.5));
  EXPECT_TRUE(approx(s.center[1], 0.5));
  EXPECT_TRUE(approx(s.center[2], 0.5));
  EXPECT_TRUE(approx(s.radius, std::sqrt(3.0) * 0.5));

  const std::array<std::array<real_t, 3>, 8> corners = {{
      {{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{1.0, 1.0, 0.0}}, {{0.0, 1.0, 0.0}},
      {{0.0, 0.0, 1.0}}, {{1.0, 0.0, 1.0}}, {{1.0, 1.0, 1.0}}, {{0.0, 1.0, 1.0}},
  }};
  for (const auto& c : corners) {
    EXPECT_TRUE(s.contains(c));
  }
}

TEST(BoundingVolumeTest, BuildCellAABBs) {
  MeshBase mesh = create_two_tet_mesh();
  auto boxes = BoundingVolumeBuilder::build_cell_aabbs(mesh, Configuration::Reference);
  ASSERT_EQ(boxes.size(), 2u);

  EXPECT_TRUE(approx(boxes[0].min[0], 0.0));
  EXPECT_TRUE(approx(boxes[0].min[1], 0.0));
  EXPECT_TRUE(approx(boxes[0].min[2], 0.0));
  EXPECT_TRUE(approx(boxes[0].max[0], 1.0));
  EXPECT_TRUE(approx(boxes[0].max[1], 1.0));
  EXPECT_TRUE(approx(boxes[0].max[2], 1.0));

  EXPECT_TRUE(approx(boxes[1].min[0], 10.0));
  EXPECT_TRUE(approx(boxes[1].min[1], 0.0));
  EXPECT_TRUE(approx(boxes[1].min[2], 0.0));
  EXPECT_TRUE(approx(boxes[1].max[0], 11.0));
  EXPECT_TRUE(approx(boxes[1].max[1], 1.0));
  EXPECT_TRUE(approx(boxes[1].max[2], 1.0));
}

TEST(BoundingVolumeTest, BuildFaceAndEdgeAABBs_UnitHexDimensions) {
  MeshBase mesh = create_unit_hex_mesh(/*finalize_topology=*/true);
  ASSERT_EQ(mesh.n_faces(), 6u);
  ASSERT_EQ(mesh.n_edges(), 12u);

  const auto face_boxes = BoundingVolumeBuilder::build_face_aabbs(mesh, Configuration::Reference);
  ASSERT_EQ(face_boxes.size(), mesh.n_faces());

  for (const auto& b : face_boxes) {
    ASSERT_TRUE(b.is_valid());
    const auto dims = b.dimensions();
    int zeros = 0;
    int ones = 0;
    for (int d = 0; d < 3; ++d) {
      if (approx(dims[static_cast<size_t>(d)], 0.0)) zeros++;
      if (approx(dims[static_cast<size_t>(d)], 1.0)) ones++;
    }
    EXPECT_EQ(zeros, 1);
    EXPECT_EQ(ones, 2);
  }

  const auto edge_boxes = BoundingVolumeBuilder::build_edge_aabbs(mesh, Configuration::Reference);
  ASSERT_EQ(edge_boxes.size(), mesh.n_edges());

  for (const auto& b : edge_boxes) {
    ASSERT_TRUE(b.is_valid());
    const auto dims = b.dimensions();
    int zeros = 0;
    int ones = 0;
    for (int d = 0; d < 3; ++d) {
      if (approx(dims[static_cast<size_t>(d)], 0.0)) zeros++;
      if (approx(dims[static_cast<size_t>(d)], 1.0)) ones++;
    }
    EXPECT_EQ(zeros, 2);
    EXPECT_EQ(ones, 1);
  }
}

TEST(BoundingVolumeTest, BuildMeshAABBAndCellSpheres_UnitHex) {
  MeshBase mesh = create_unit_hex_mesh(/*finalize_topology=*/false);
  const auto mesh_aabb = BoundingVolumeBuilder::build_mesh_aabb(mesh, Configuration::Reference);
  EXPECT_TRUE(approx(mesh_aabb.min[0], 0.0));
  EXPECT_TRUE(approx(mesh_aabb.min[1], 0.0));
  EXPECT_TRUE(approx(mesh_aabb.min[2], 0.0));
  EXPECT_TRUE(approx(mesh_aabb.max[0], 1.0));
  EXPECT_TRUE(approx(mesh_aabb.max[1], 1.0));
  EXPECT_TRUE(approx(mesh_aabb.max[2], 1.0));

  const auto spheres = BoundingVolumeBuilder::build_cell_spheres(mesh, Configuration::Reference);
  ASSERT_EQ(spheres.size(), 1u);
  EXPECT_TRUE(approx(spheres[0].center[0], 0.5));
  EXPECT_TRUE(approx(spheres[0].center[1], 0.5));
  EXPECT_TRUE(approx(spheres[0].center[2], 0.5));
  EXPECT_TRUE(approx(spheres[0].radius, std::sqrt(3.0) * 0.5));

  // Sphere should contain all mesh vertices.
  const auto& X = mesh.X_ref();
  const int dim = mesh.dim();
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    std::array<real_t, 3> pt = {0.0, 0.0, 0.0};
    pt[0] = X[static_cast<size_t>(v) * dim + 0];
    pt[1] = X[static_cast<size_t>(v) * dim + 1];
    pt[2] = X[static_cast<size_t>(v) * dim + 2];
    EXPECT_TRUE(spheres[0].contains(pt));
  }
}

TEST(BoundingVolumeTest, BuildAABBTreePartitionsCells) {
  MeshBase mesh = create_two_tet_mesh();
  auto nodes = BoundingVolumeBuilder::build_aabb_tree(mesh, Configuration::Reference, /*max_depth=*/8, /*max_cells_per_leaf=*/1);
  ASSERT_FALSE(nodes.empty());

  const auto& root = nodes[0];
  EXPECT_NE(root.left_child, -1);
  EXPECT_NE(root.right_child, -1);
  EXPECT_TRUE(root.box.is_valid());

  // Root should enclose all vertices from both cells.
  EXPECT_TRUE(approx(root.box.min[0], 0.0));
  EXPECT_TRUE(approx(root.box.min[1], 0.0));
  EXPECT_TRUE(approx(root.box.min[2], 0.0));
  EXPECT_TRUE(approx(root.box.max[0], 11.0));
  EXPECT_TRUE(approx(root.box.max[1], 1.0));
  EXPECT_TRUE(approx(root.box.max[2], 1.0));

  std::unordered_set<index_t> leaf_cells;
  for (const auto& n : nodes) {
    const bool is_leaf = (n.left_child < 0 && n.right_child < 0);
    if (!is_leaf) continue;
    EXPECT_LE(n.cell_indices.size(), 1u);
    for (auto cid : n.cell_indices) leaf_cells.insert(cid);
  }

  EXPECT_EQ(leaf_cells.size(), 2u);
  EXPECT_TRUE(leaf_cells.count(0) == 1u);
  EXPECT_TRUE(leaf_cells.count(1) == 1u);
}

#ifdef MESH_HAS_EIGEN
TEST(BoundingVolumeTest, BuildCellOBBContainsVertices) {
  MeshBase mesh = create_box_hex_mesh();
  auto obb = BoundingVolumeBuilder::build_cell_obb_pca(mesh, 0, Configuration::Reference);
  auto aabb = obb.to_aabb();

  const auto& X = mesh.X_ref();
  const int dim = mesh.dim();
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    std::array<real_t, 3> pt = {0.0, 0.0, 0.0};
    pt[0] = X[static_cast<size_t>(v) * dim + 0];
    pt[1] = X[static_cast<size_t>(v) * dim + 1];
    pt[2] = X[static_cast<size_t>(v) * dim + 2];
    EXPECT_TRUE(obb.contains(pt));
    EXPECT_TRUE(aabb.contains(pt));
  }
}
#endif

} // namespace test
} // namespace svmp
