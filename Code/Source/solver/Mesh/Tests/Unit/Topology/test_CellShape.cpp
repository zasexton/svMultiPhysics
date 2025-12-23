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

#include "gtest/gtest.h"

#include "Topology/CellShape.h"

namespace svmp {
namespace test {

TEST(CellShapeTest, DimensionHelpersAreConsistent) {
  CellShape p;
  p.family = CellFamily::Point;
  EXPECT_FALSE(p.is_1d());
  EXPECT_FALSE(p.is_2d());
  EXPECT_FALSE(p.is_3d());
  EXPECT_EQ(p.topo_kind(), EntityKind::Vertex);

  CellShape l;
  l.family = CellFamily::Line;
  EXPECT_TRUE(l.is_1d());
  EXPECT_FALSE(l.is_2d());
  EXPECT_FALSE(l.is_3d());
  EXPECT_EQ(l.topo_kind(), EntityKind::Edge);

  CellShape tri;
  tri.family = CellFamily::Triangle;
  EXPECT_FALSE(tri.is_1d());
  EXPECT_TRUE(tri.is_2d());
  EXPECT_FALSE(tri.is_3d());
  EXPECT_EQ(tri.topo_kind(), EntityKind::Face);

  CellShape tet;
  tet.family = CellFamily::Tetra;
  EXPECT_FALSE(tet.is_1d());
  EXPECT_FALSE(tet.is_2d());
  EXPECT_TRUE(tet.is_3d());
  EXPECT_EQ(tet.topo_kind(), EntityKind::Volume);
}

TEST(CellShapeTest, ExpectedVerticesMatchesKnownCounts) {
  CellShape line;
  line.family = CellFamily::Line;
  line.order = 1;
  EXPECT_EQ(line.expected_vertices(), 2);
  line.order = 3;
  EXPECT_EQ(line.expected_vertices(), 4);

  CellShape tri;
  tri.family = CellFamily::Triangle;
  tri.order = 1;
  EXPECT_EQ(tri.expected_vertices(), 3);
  tri.order = 2;
  EXPECT_EQ(tri.expected_vertices(), 6);

  CellShape hex;
  hex.family = CellFamily::Hex;
  hex.order = 2;
  EXPECT_EQ(hex.expected_vertices(), 27);

  CellShape wedge;
  wedge.family = CellFamily::Wedge;
  wedge.order = 2;
  EXPECT_EQ(wedge.expected_vertices(), 18);

  CellShape pyr;
  pyr.family = CellFamily::Pyramid;
  pyr.order = 1;
  EXPECT_EQ(pyr.expected_vertices(), 5);
  pyr.order = 2;
  EXPECT_EQ(pyr.expected_vertices(), 14);
}

} // namespace test
} // namespace svmp

