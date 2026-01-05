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

#include <gtest/gtest.h>

#include "../../../Core/MeshBase.h"
#include "../../../Topology/CellShape.h"
#include "../../../Validation/MeshValidation.h"

namespace svmp::test {
namespace {

real_t signed_triangle_area_xy(const MeshBase& mesh, index_t cell) {
  auto [vptr, nv] = mesh.cell_vertices_span(cell);
  EXPECT_GE(nv, 3u);
  const auto p0 = mesh.get_vertex_coords(vptr[0]);
  const auto p1 = mesh.get_vertex_coords(vptr[1]);
  const auto p2 = mesh.get_vertex_coords(vptr[2]);
  return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
}

} // namespace

TEST(MeshValidationRepair, MergeDuplicateVertices) {
  MeshBase mesh;

  // Two triangles forming a quad, with a duplicate of vertex 0 used in the second triangle.
  const std::vector<real_t> X_ref = {
      0.0, 0.0,  // v0
      1.0, 0.0,  // v1
      1.0, 1.0,  // v2
      0.0, 1.0,  // v3
      0.0, 0.0   // v4 (duplicate of v0)
  };
  const std::vector<offset_t> offsets = {0, 3, 6};
  const std::vector<index_t> conn = {0, 1, 2, 4, 2, 3};
  const std::vector<CellShape> shapes = {{CellFamily::Triangle, 3, 1}, {CellFamily::Triangle, 3, 1}};

  mesh.build_from_arrays(2, X_ref, offsets, conn, shapes);
  mesh.finalize();
  ASSERT_EQ(mesh.n_vertices(), 5u);

  const index_t merged = MeshValidation::merge_duplicate_vertices(mesh, 1e-12);
  EXPECT_EQ(merged, 1);
  EXPECT_EQ(mesh.n_vertices(), 4u);
  EXPECT_EQ(mesh.n_cells(), 2u);

  // Ensure the second triangle no longer references a now-removed vertex id.
  const auto c1 = mesh.cell_vertices(1);
  for (const auto v : c1) {
    EXPECT_LT(v, static_cast<index_t>(mesh.n_vertices()));
  }
}

TEST(MeshValidationRepair, RemoveIsolatedVertices) {
  MeshBase mesh;

  // One triangle plus an unused vertex.
  const std::vector<real_t> X_ref = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0};
  const std::vector<offset_t> offsets = {0, 3};
  const std::vector<index_t> conn = {0, 1, 2};
  const std::vector<CellShape> shapes = {{CellFamily::Triangle, 3, 1}};

  mesh.build_from_arrays(2, X_ref, offsets, conn, shapes);
  mesh.finalize();
  ASSERT_EQ(mesh.n_vertices(), 4u);

  const index_t removed = MeshValidation::remove_isolated_vertices(mesh);
  EXPECT_EQ(removed, 1);
  EXPECT_EQ(mesh.n_vertices(), 3u);
  EXPECT_EQ(mesh.n_cells(), 1u);
}

TEST(MeshValidationRepair, RemoveDegenerateCells) {
  MeshBase mesh;

  // Two triangles: the second is degenerate (collinear).
  const std::vector<real_t> X_ref = {
      0.0, 0.0,  // v0
      1.0, 0.0,  // v1
      0.0, 1.0,  // v2
      2.0, 0.0,  // v3
      3.0, 0.0   // v4 (collinear with v3 and v1)
  };
  const std::vector<offset_t> offsets = {0, 3, 6};
  const std::vector<index_t> conn = {0, 1, 2, 1, 3, 4};
  const std::vector<CellShape> shapes = {{CellFamily::Triangle, 3, 1}, {CellFamily::Triangle, 3, 1}};

  mesh.build_from_arrays(2, X_ref, offsets, conn, shapes);
  mesh.finalize();
  ASSERT_EQ(mesh.n_cells(), 2u);

  const index_t removed = MeshValidation::remove_degenerate_cells(mesh, 1e-14);
  EXPECT_EQ(removed, 1);
  EXPECT_EQ(mesh.n_cells(), 1u);
}

TEST(MeshValidationRepair, FixInvertedTetra) {
  MeshBase mesh;

  const std::vector<real_t> X_ref = {
      0.0, 0.0, 0.0,  // v0
      1.0, 0.0, 0.0,  // v1
      0.0, 1.0, 0.0,  // v2
      0.0, 0.0, 1.0   // v3
  };
  const std::vector<offset_t> offsets = {0, 4};
  // Swapping v1 and v2 in a tet flips orientation.
  const std::vector<index_t> conn = {0, 2, 1, 3};
  const std::vector<CellShape> shapes = {{CellFamily::Tetra, 4, 1}};

  mesh.build_from_arrays(3, X_ref, offsets, conn, shapes);
  mesh.finalize();

  EXPECT_FALSE(MeshValidation::find_inverted_cells(mesh).passed);
  const index_t fixed = MeshValidation::fix_inverted_cells(mesh);
  EXPECT_EQ(fixed, 1);
  EXPECT_TRUE(MeshValidation::find_inverted_cells(mesh).passed);
}

TEST(MeshValidationRepair, OrientFacesConsistently_Triangles) {
  MeshBase mesh;

  const std::vector<real_t> X_ref = {
      0.0, 0.0,  // v0
      1.0, 0.0,  // v1
      1.0, 1.0,  // v2
      0.0, 1.0   // v3
  };
  const std::vector<offset_t> offsets = {0, 3, 6};
  // Second triangle is intentionally flipped (CW) relative to the first.
  const std::vector<index_t> conn = {0, 1, 2, 0, 3, 2};
  const std::vector<CellShape> shapes = {{CellFamily::Triangle, 3, 1}, {CellFamily::Triangle, 3, 1}};

  mesh.build_from_arrays(2, X_ref, offsets, conn, shapes);
  mesh.finalize();

  EXPECT_GT(signed_triangle_area_xy(mesh, 0), 0.0);
  EXPECT_LT(signed_triangle_area_xy(mesh, 1), 0.0);

  const index_t flipped = MeshValidation::orient_faces_consistently(mesh);
  EXPECT_EQ(flipped, 1);

  EXPECT_GT(signed_triangle_area_xy(mesh, 0), 0.0);
  EXPECT_GT(signed_triangle_area_xy(mesh, 1), 0.0);
}

} // namespace svmp::test

