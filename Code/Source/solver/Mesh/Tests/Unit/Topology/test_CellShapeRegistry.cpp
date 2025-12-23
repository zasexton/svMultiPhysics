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

#include <stdexcept>
#include <string>
#include <vector>

namespace svmp {
namespace test {

class CellShapeRegistryTest : public ::testing::Test {
protected:
  void SetUp() override { CellShapeRegistry::clear_all(); }
  void TearDown() override { CellShapeRegistry::clear_all(); }
};

TEST_F(CellShapeRegistryTest, RegisterHasGetRoundtrip) {
  CellShape s;
  s.family = CellFamily::Tetra;
  s.num_corners = 4;
  s.order = 2;
  s.is_mixed_order = false;

  CellShapeRegistry::register_shape("vtk", 10, s);

  EXPECT_TRUE(CellShapeRegistry::has("vtk", 10));

  const auto got = CellShapeRegistry::get("vtk", 10);
  EXPECT_EQ(got.family, s.family);
  EXPECT_EQ(got.num_corners, s.num_corners);
  EXPECT_EQ(got.order, s.order);
  EXPECT_EQ(got.is_mixed_order, s.is_mixed_order);
}

TEST_F(CellShapeRegistryTest, GetThrowsForMissingFormat) {
  EXPECT_THROW((void)CellShapeRegistry::get("missing", 1), std::runtime_error);
}

TEST_F(CellShapeRegistryTest, GetThrowsForMissingTypeId) {
  CellShape s;
  s.family = CellFamily::Hex;
  s.num_corners = 8;
  s.order = 1;
  CellShapeRegistry::register_shape("vtk", 12, s);

  EXPECT_THROW((void)CellShapeRegistry::get("vtk", 99), std::runtime_error);
}

TEST_F(CellShapeRegistryTest, FormatsAreSortedAndClearFormatWorks) {
  CellShape a;
  a.family = CellFamily::Triangle;
  a.num_corners = 3;
  a.order = 1;
  CellShapeRegistry::register_shape("vtk", 5, a);

  CellShape b;
  b.family = CellFamily::Quad;
  b.num_corners = 4;
  b.order = 1;
  CellShapeRegistry::register_shape("gmsh", 3, b);

  const std::vector<std::string> expected_formats = {"gmsh", "vtk"};
  EXPECT_EQ(CellShapeRegistry::formats(), expected_formats);

  CellShapeRegistry::clear_format("vtk");
  EXPECT_FALSE(CellShapeRegistry::has("vtk", 5));
  EXPECT_TRUE(CellShapeRegistry::has("gmsh", 3));

  const std::vector<std::string> expected_after = {"gmsh"};
  EXPECT_EQ(CellShapeRegistry::formats(), expected_after);
}

} // namespace test
} // namespace svmp

