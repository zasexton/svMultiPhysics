/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"

#include "Topology/NodeOrdering.h"

namespace svmp {
namespace test {

TEST(NodeOrderingTest, LineP2_GmshToVTKPermutation) {
  const auto perm = NodeOrdering::permutation_to_vtk(NodeOrderingFormat::Gmsh, CellFamily::Line, /*order=*/2, /*n=*/3);
  ASSERT_EQ(perm.size(), 3u);
  EXPECT_EQ(perm[0], 0);
  EXPECT_EQ(perm[1], 2);
  EXPECT_EQ(perm[2], 1);
}

TEST(NodeOrderingTest, LineP3_GmshToVTKPermutation) {
  const auto perm = NodeOrdering::permutation_to_vtk(NodeOrderingFormat::Gmsh, CellFamily::Line, /*order=*/3, /*n=*/4);
  ASSERT_EQ(perm.size(), 4u);
  EXPECT_EQ(perm[0], 0);
  EXPECT_EQ(perm[1], 2);
  EXPECT_EQ(perm[2], 3);
  EXPECT_EQ(perm[3], 1);
}

TEST(NodeOrderingTest, InvertPermutationRoundTrip) {
  // Use a nontrivial mapping (LineP2) and verify reorder_to_vtk + reorder_from_vtk is identity.
  std::vector<size_t> nodes = {10, 11, 12};  // gmsh ordering: [v0,v1,mid]
  const auto orig = nodes;

  NodeOrdering::reorder_to_vtk(NodeOrderingFormat::Gmsh, CellFamily::Line, /*order=*/2, nodes);
  NodeOrdering::reorder_from_vtk(NodeOrderingFormat::Gmsh, CellFamily::Line, /*order=*/2, nodes);

  EXPECT_EQ(nodes, orig);
}

} // namespace test
} // namespace svmp

