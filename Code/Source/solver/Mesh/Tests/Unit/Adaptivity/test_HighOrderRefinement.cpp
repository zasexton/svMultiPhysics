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

/**
 * @file test_HighOrderRefinement.cpp
 * @brief High-order (quadratic) refinement regression tests.
 *
 * These tests are analogous to MFEM/deal.II-style checks:
 * - refine a single high-order element and verify the refined mesh has the expected
 *   number of unique vertices and the expected per-cell node counts
 * - verify deterministic vertex-field prolongation using the stored interpolation weights
 *   (including negative weights for serendipity mappings)
 * - verify existing vertex GIDs are preserved
 */

#include <gtest/gtest.h>

#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace svmp {
namespace test {

namespace {

AdaptivityOptions base_refine_options() {
  AdaptivityOptions options;
  options.enable_refinement = true;
  options.enable_coarsening = false;
  options.check_quality = false;
  options.enforce_quality_after_refinement = false;
  options.rollback_on_poor_quality = false;
  options.verbosity = 0;
  options.max_refinement_level = 8;
  options.max_level_difference = 1;
  options.use_green_closure = false;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  return options;
}

std::vector<gid_t> make_nontrivial_gids(size_t n, gid_t base) {
  std::vector<gid_t> gids(n, INVALID_GID);
  for (size_t i = 0; i < n; ++i) gids[i] = base + static_cast<gid_t>(17 * i + 3);
  return gids;
}

void attach_vertex_x_field(MeshBase& mesh, const std::string& name) {
  auto h = MeshFields::attach_field(mesh, EntityKind::Vertex, name, FieldScalarType::Float64, 1);
  auto* data = MeshFields::field_data_as<double>(mesh, h);
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    data[static_cast<size_t>(v)] = static_cast<double>(mesh.get_vertex_coords(v)[0]);
  }
}

void expect_vertex_field_matches_x(const MeshBase& mesh, const std::string& name, double tol = 1e-12) {
  ASSERT_TRUE(MeshFields::has_field(mesh, EntityKind::Vertex, name));
  auto h = MeshFields::get_field_handle(mesh, EntityKind::Vertex, name);
  const auto* data = MeshFields::field_data_as<const double>(mesh, h);
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    const double x = static_cast<double>(mesh.get_vertex_coords(v)[0]);
    EXPECT_NEAR(data[static_cast<size_t>(v)], x, tol) << "vertex " << v;
  }
}

MeshBase build_single_quad_quadratic(bool with_center) {
  // Reference-space coordinates (z=0) with internal ordering:
  // corners (4) -> edge mids (4) -> optional center (1).
  std::vector<std::array<real_t, 3>> pts;
  pts.reserve(with_center ? 9 : 8);

  // Corners: (-1,-1), (1,-1), (1,1), (-1,1)
  pts.push_back({-1.0, -1.0, 0.0});
  pts.push_back({+1.0, -1.0, 0.0});
  pts.push_back({+1.0, +1.0, 0.0});
  pts.push_back({-1.0, +1.0, 0.0});

  // Edge mids in quad edge order: (0-1),(1-2),(2-3),(3-0)
  pts.push_back({0.0, -1.0, 0.0});
  pts.push_back({+1.0, 0.0, 0.0});
  pts.push_back({0.0, +1.0, 0.0});
  pts.push_back({-1.0, 0.0, 0.0});

  if (with_center) {
    pts.push_back({0.0, 0.0, 0.0});
  }

  std::vector<real_t> X;
  X.reserve(pts.size() * 3);
  for (const auto& p : pts) {
    X.push_back(p[0]);
    X.push_back(p[1]);
    X.push_back(p[2]);
  }

  std::vector<index_t> conn;
  conn.reserve(pts.size());
  for (index_t i = 0; i < static_cast<index_t>(pts.size()); ++i) conn.push_back(i);

  std::vector<offset_t> off = {0, static_cast<offset_t>(conn.size())};

  CellShape cs;
  cs.family = CellFamily::Quad;
  cs.order = 2;
  cs.num_corners = 4;

  MeshBase mesh(3);
  mesh.build_from_arrays(3, X, off, conn, std::vector<CellShape>{cs});
  mesh.finalize();
  mesh.set_vertex_gids(make_nontrivial_gids(mesh.n_vertices(), 1000));
  mesh.set_cell_gids({5000});
  return mesh;
}

MeshBase build_single_tri6() {
  // Reference triangle corners: (0,0), (1,0), (0,1).
  // Ordering: corners (3) -> edge mids (3) with edges (0-1),(1-2),(2-0).
  std::vector<std::array<real_t, 3>> pts;
  pts.reserve(6);
  pts.push_back({0.0, 0.0, 0.0});   // 0
  pts.push_back({1.0, 0.0, 0.0});   // 1
  pts.push_back({0.0, 1.0, 0.0});   // 2
  pts.push_back({0.5, 0.0, 0.0});   // (0-1)
  pts.push_back({0.5, 0.5, 0.0});   // (1-2)
  pts.push_back({0.0, 0.5, 0.0});   // (2-0)

  std::vector<real_t> X;
  X.reserve(pts.size() * 3);
  for (const auto& p : pts) {
    X.push_back(p[0]);
    X.push_back(p[1]);
    X.push_back(p[2]);
  }

  std::vector<index_t> conn;
  conn.reserve(pts.size());
  for (index_t i = 0; i < static_cast<index_t>(pts.size()); ++i) conn.push_back(i);

  std::vector<offset_t> off = {0, static_cast<offset_t>(conn.size())};

  CellShape cs;
  cs.family = CellFamily::Triangle;
  cs.order = 2;
  cs.num_corners = 3;

  MeshBase mesh(3);
  mesh.build_from_arrays(3, X, off, conn, std::vector<CellShape>{cs});
  mesh.finalize();
  mesh.set_vertex_gids(make_nontrivial_gids(mesh.n_vertices(), 3000));
  mesh.set_cell_gids({7000});
  return mesh;
}

MeshBase build_single_tet10() {
  // Reference tetra corners: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
  // Ordering: corners (4) -> edge mids (6) with edges (0-1),(0-2),(0-3),(1-2),(1-3),(2-3).
  std::vector<std::array<real_t, 3>> pts;
  pts.reserve(10);
  pts.push_back({0.0, 0.0, 0.0});   // 0
  pts.push_back({1.0, 0.0, 0.0});   // 1
  pts.push_back({0.0, 1.0, 0.0});   // 2
  pts.push_back({0.0, 0.0, 1.0});   // 3
  pts.push_back({0.5, 0.0, 0.0});   // (0-1)
  pts.push_back({0.0, 0.5, 0.0});   // (0-2)
  pts.push_back({0.0, 0.0, 0.5});   // (0-3)
  pts.push_back({0.5, 0.5, 0.0});   // (1-2)
  pts.push_back({0.5, 0.0, 0.5});   // (1-3)
  pts.push_back({0.0, 0.5, 0.5});   // (2-3)

  std::vector<real_t> X;
  X.reserve(pts.size() * 3);
  for (const auto& p : pts) {
    X.push_back(p[0]);
    X.push_back(p[1]);
    X.push_back(p[2]);
  }

  std::vector<index_t> conn;
  conn.reserve(pts.size());
  for (index_t i = 0; i < static_cast<index_t>(pts.size()); ++i) conn.push_back(i);

  std::vector<offset_t> off = {0, static_cast<offset_t>(conn.size())};

  CellShape cs;
  cs.family = CellFamily::Tetra;
  cs.order = 2;
  cs.num_corners = 4;

  MeshBase mesh(3);
  mesh.build_from_arrays(3, X, off, conn, std::vector<CellShape>{cs});
  mesh.finalize();
  mesh.set_vertex_gids(make_nontrivial_gids(mesh.n_vertices(), 4000));
  mesh.set_cell_gids({8000});
  return mesh;
}

MeshBase build_single_hex_quadratic(bool with_faces_and_center) {
  // Reference-space coordinates with internal ordering:
  // corners (8) -> edge mids (12) -> optional face centers (6) -> optional center (1).
  std::vector<std::array<real_t, 3>> pts;
  pts.reserve(with_faces_and_center ? 27 : 20);

  // Corners (VTK / CellTopology ordering)
  pts.push_back({-1.0, -1.0, -1.0}); // 0
  pts.push_back({+1.0, -1.0, -1.0}); // 1
  pts.push_back({+1.0, +1.0, -1.0}); // 2
  pts.push_back({-1.0, +1.0, -1.0}); // 3
  pts.push_back({-1.0, -1.0, +1.0}); // 4
  pts.push_back({+1.0, -1.0, +1.0}); // 5
  pts.push_back({+1.0, +1.0, +1.0}); // 6
  pts.push_back({-1.0, +1.0, +1.0}); // 7

  // Edge mids (12) in hex edge order
  pts.push_back({0.0, -1.0, -1.0});  // (0-1)
  pts.push_back({+1.0, 0.0, -1.0});  // (1-2)
  pts.push_back({0.0, +1.0, -1.0});  // (2-3)
  pts.push_back({-1.0, 0.0, -1.0});  // (3-0)
  pts.push_back({0.0, -1.0, +1.0});  // (4-5)
  pts.push_back({+1.0, 0.0, +1.0});  // (5-6)
  pts.push_back({0.0, +1.0, +1.0});  // (6-7)
  pts.push_back({-1.0, 0.0, +1.0});  // (7-4)
  pts.push_back({-1.0, -1.0, 0.0});  // (0-4)
  pts.push_back({+1.0, -1.0, 0.0});  // (1-5)
  pts.push_back({+1.0, +1.0, 0.0});  // (2-6)
  pts.push_back({-1.0, +1.0, 0.0});  // (3-7)

  if (with_faces_and_center) {
    // Face centers (6) in oriented face order (see VTKWriter reorderer)
    pts.push_back({0.0, 0.0, -1.0}); // bottom
    pts.push_back({0.0, 0.0, +1.0}); // top
    pts.push_back({0.0, -1.0, 0.0}); // y=-1
    pts.push_back({+1.0, 0.0, 0.0}); // x=+1
    pts.push_back({0.0, +1.0, 0.0}); // y=+1
    pts.push_back({-1.0, 0.0, 0.0}); // x=-1

    // Center
    pts.push_back({0.0, 0.0, 0.0});
  }

  std::vector<real_t> X;
  X.reserve(pts.size() * 3);
  for (const auto& p : pts) {
    X.push_back(p[0]);
    X.push_back(p[1]);
    X.push_back(p[2]);
  }

  std::vector<index_t> conn;
  conn.reserve(pts.size());
  for (index_t i = 0; i < static_cast<index_t>(pts.size()); ++i) conn.push_back(i);

  std::vector<offset_t> off = {0, static_cast<offset_t>(conn.size())};

  CellShape cs;
  cs.family = CellFamily::Hex;
  cs.order = 2;
  cs.num_corners = 8;

  MeshBase mesh(3);
  mesh.build_from_arrays(3, X, off, conn, std::vector<CellShape>{cs});
  mesh.finalize();
  mesh.set_vertex_gids(make_nontrivial_gids(mesh.n_vertices(), 2000));
  mesh.set_cell_gids({6000});
  return mesh;
}

void expect_all_cells_have_vertex_count(const MeshBase& mesh, size_t expected) {
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    EXPECT_EQ(mesh.cell_vertices(c).size(), expected) << "cell " << c;
  }
}

void expect_first_vertex_gids_preserved(const std::vector<gid_t>& old_vgids,
                                       const MeshBase& new_mesh) {
  const auto& new_vgids = new_mesh.vertex_gids();
  ASSERT_GE(new_vgids.size(), old_vgids.size());
  for (size_t i = 0; i < old_vgids.size(); ++i) {
    EXPECT_EQ(new_vgids[i], old_vgids[i]) << "vertex " << i;
  }
}

} // namespace

TEST(HighOrderRefinement, Quad9_RefinesToQ9Children_With5x5UniqueNodes_AndTransfersVertexField) {
  MeshBase mesh = build_single_quad_quadratic(/*with_center=*/true);
  const auto old_vgids = mesh.vertex_gids();

  MeshFields fields;
  attach_vertex_x_field(mesh, "phi");

  AdaptivityManager manager(base_refine_options());
  std::vector<bool> marks(mesh.n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(mesh, marks, &fields);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh.n_cells(), 4u);
  EXPECT_EQ(mesh.n_vertices(), 25u);
  expect_all_cells_have_vertex_count(mesh, 9u);
  expect_first_vertex_gids_preserved(old_vgids, mesh);
  expect_vertex_field_matches_x(mesh, "phi");
}

TEST(HighOrderRefinement, Quad8_RefinesToQ8Children_With21UniqueNodes_AndTransfersVertexField) {
  MeshBase mesh = build_single_quad_quadratic(/*with_center=*/false);
  const auto old_vgids = mesh.vertex_gids();

  MeshFields fields;
  attach_vertex_x_field(mesh, "phi");

  AdaptivityManager manager(base_refine_options());
  std::vector<bool> marks(mesh.n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(mesh, marks, &fields);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh.n_cells(), 4u);
  EXPECT_EQ(mesh.n_vertices(), 21u);
  expect_all_cells_have_vertex_count(mesh, 8u);
  expect_first_vertex_gids_preserved(old_vgids, mesh);
  expect_vertex_field_matches_x(mesh, "phi");
}

TEST(HighOrderRefinement, Hex27_RefinesToHex27Children_With5x5x5UniqueNodes_AndTransfersVertexField) {
  MeshBase mesh = build_single_hex_quadratic(/*with_faces_and_center=*/true);
  const auto old_vgids = mesh.vertex_gids();

  MeshFields fields;
  attach_vertex_x_field(mesh, "phi");

  AdaptivityManager manager(base_refine_options());
  std::vector<bool> marks(mesh.n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(mesh, marks, &fields);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh.n_cells(), 8u);
  EXPECT_EQ(mesh.n_vertices(), 125u);
  expect_all_cells_have_vertex_count(mesh, 27u);
  expect_first_vertex_gids_preserved(old_vgids, mesh);
  expect_vertex_field_matches_x(mesh, "phi");
}

TEST(HighOrderRefinement, Hex20_RefinesToHex20Children_With81UniqueNodes_AndTransfersVertexField) {
  MeshBase mesh = build_single_hex_quadratic(/*with_faces_and_center=*/false);
  const auto old_vgids = mesh.vertex_gids();

  MeshFields fields;
  attach_vertex_x_field(mesh, "phi");

  AdaptivityManager manager(base_refine_options());
  std::vector<bool> marks(mesh.n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(mesh, marks, &fields);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh.n_cells(), 8u);
  EXPECT_EQ(mesh.n_vertices(), 81u);
  expect_all_cells_have_vertex_count(mesh, 20u);
  expect_first_vertex_gids_preserved(old_vgids, mesh);
  expect_vertex_field_matches_x(mesh, "phi");
}

TEST(HighOrderRefinement, Tri6_RefinesToTri6Children_With15UniqueNodes_AndTransfersVertexField) {
  MeshBase mesh = build_single_tri6();
  const auto old_vgids = mesh.vertex_gids();

  MeshFields fields;
  attach_vertex_x_field(mesh, "phi");

  AdaptivityManager manager(base_refine_options());
  std::vector<bool> marks(mesh.n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(mesh, marks, &fields);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh.n_cells(), 4u);
  EXPECT_EQ(mesh.n_vertices(), 15u);
  expect_all_cells_have_vertex_count(mesh, 6u);
  expect_first_vertex_gids_preserved(old_vgids, mesh);
  expect_vertex_field_matches_x(mesh, "phi");
}

TEST(HighOrderRefinement, Tet10_RefinesToTet10Children_With35UniqueNodes_AndTransfersVertexField) {
  MeshBase mesh = build_single_tet10();
  const auto old_vgids = mesh.vertex_gids();

  MeshFields fields;
  attach_vertex_x_field(mesh, "phi");

  AdaptivityManager manager(base_refine_options());
  std::vector<bool> marks(mesh.n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(mesh, marks, &fields);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh.n_cells(), 8u);
  EXPECT_EQ(mesh.n_vertices(), 35u);
  expect_all_cells_have_vertex_count(mesh, 10u);
  expect_first_vertex_gids_preserved(old_vgids, mesh);
  expect_vertex_field_matches_x(mesh, "phi");
}

} // namespace test
} // namespace svmp
