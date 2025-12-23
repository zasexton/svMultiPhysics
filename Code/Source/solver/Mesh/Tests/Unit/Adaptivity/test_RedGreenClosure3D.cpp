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
 * @file test_RedGreenClosure3D.cpp
 * @brief Tests for 3D face-based RED–GREEN closure variants (tets/hexes)
 *
 * This suite validates selector-aware closure specs for 3D:
 * - Tetrahedra: one refined neighbor across a face triggers face-based GREEN refinement (4 children).
 * - Hexahedra: one refined neighbor across a face triggers a 2D anisotropic (2-direction) split (4 children).
 *
 * The intent is analogous to deal.II/MFEM closure tests: minimal refinement to preserve conformity.
 */

#include <gtest/gtest.h>

#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Adaptivity/Conformity.h"
#include "../../../Adaptivity/FieldTransfer.h"
#include "../../../Adaptivity/Options.h"
#include "../../../Adaptivity/QualityGuards.h"
#include "../../../Core/MeshBase.h"

#include <cmath>
#include <memory>
#include <vector>

namespace svmp {
namespace test {

namespace {

size_t count_vertices_near(const MeshBase& mesh, const std::array<double, 3>& p, double tol = 1e-12) {
  size_t count = 0;
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    auto x = mesh.get_vertex_coords(v);
    const double dx = static_cast<double>(x[0]) - p[0];
    const double dy = static_cast<double>(x[1]) - p[1];
    const double dz = static_cast<double>(x[2]) - p[2];
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < tol * tol) count++;
  }
  return count;
}

AdaptivityOptions base_options() {
  AdaptivityOptions options;
  options.enable_refinement = true;
  options.enable_coarsening = false;
  options.check_quality = false;
  options.verbosity = 0;
  options.max_refinement_level = 5;
  options.max_level_difference = 1;
  options.use_green_closure = true;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED_GREEN;
  return options;
}

std::unique_ptr<MeshBase> make_two_tetra_share_face() {
  auto mesh = std::make_unique<MeshBase>();

  // Two tets share the face (0,1,2) in plane z=0.
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {0.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 0.0, 1.0});
  mesh->add_vertex(4, {0.0, 0.0, -1.0});

  // Positive orientation tets.
  mesh->add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh->add_cell(1, CellFamily::Tetra, {0, 2, 1, 4});

  return mesh;
}

std::unique_ptr<MeshBase> make_two_hex_stacked() {
  auto mesh = std::make_unique<MeshBase>();

  // Two unit cubes stacked in z, sharing the face z=1.
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {1.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 1.0, 0.0});
  mesh->add_vertex(4, {0.0, 0.0, 1.0});
  mesh->add_vertex(5, {1.0, 0.0, 1.0});
  mesh->add_vertex(6, {1.0, 1.0, 1.0});
  mesh->add_vertex(7, {0.0, 1.0, 1.0});
  mesh->add_vertex(8, {0.0, 0.0, 2.0});
  mesh->add_vertex(9, {1.0, 0.0, 2.0});
  mesh->add_vertex(10, {1.0, 1.0, 2.0});
  mesh->add_vertex(11, {0.0, 1.0, 2.0});

  mesh->add_cell(0, CellFamily::Hex, {0, 1, 2, 3, 4, 5, 6, 7});
  mesh->add_cell(1, CellFamily::Hex, {4, 5, 6, 7, 8, 9, 10, 11});

  return mesh;
}

std::unique_ptr<MeshBase> make_two_wedges_share_quad_face() {
  auto mesh = std::make_unique<MeshBase>();

  // Two wedges share the quad face (0,1,4,3) in plane y=0.
  // Wedge A: triangle (0,1,2) extruded to (3,4,5).
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {0.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 0.0, 1.0});
  mesh->add_vertex(4, {1.0, 0.0, 1.0});
  mesh->add_vertex(5, {0.0, 1.0, 1.0});

  // Wedge B shares the quad face (0,1,4,3) but lies on the other side of edge (0,1).
  mesh->add_vertex(6, {0.0, -1.0, 0.0});
  mesh->add_vertex(7, {0.0, -1.0, 1.0});

  mesh->add_cell(0, CellFamily::Wedge, {0, 1, 2, 3, 4, 5});
  mesh->add_cell(1, CellFamily::Wedge, {0, 1, 6, 3, 4, 7});
  return mesh;
}

std::unique_ptr<MeshBase> make_hex_wedge_share_quad_face() {
  auto mesh = std::make_unique<MeshBase>();

  // Wedge shares quad face (0,1,4,3) with the hex face (0,1,5,4).
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {0.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 0.0, 1.0});
  mesh->add_vertex(4, {1.0, 0.0, 1.0});
  mesh->add_vertex(5, {0.0, 1.0, 1.0});

  // Hex extends in negative y direction so its face at y=0 matches the wedge quad face.
  // NOTE: MeshBase::add_vertex() appends by insertion order; use contiguous indices.
  mesh->add_vertex(6, {1.0, -1.0, 0.0});
  mesh->add_vertex(7, {0.0, -1.0, 0.0});
  mesh->add_vertex(8, {1.0, -1.0, 1.0});
  mesh->add_vertex(9, {0.0, -1.0, 1.0});

  mesh->add_cell(0, CellFamily::Wedge, {0, 1, 2, 3, 4, 5});
  mesh->add_cell(1, CellFamily::Hex, {0, 1, 6, 7, 3, 4, 8, 9});
  return mesh;
}

std::unique_ptr<MeshBase> make_hex_pyramid_share_face() {
  auto mesh = std::make_unique<MeshBase>();

  // Unit cube (hex) with a pyramid on top sharing the quad face z=1.
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {1.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 1.0, 0.0});
  mesh->add_vertex(4, {0.0, 0.0, 1.0});
  mesh->add_vertex(5, {1.0, 0.0, 1.0});
  mesh->add_vertex(6, {1.0, 1.0, 1.0});
  mesh->add_vertex(7, {0.0, 1.0, 1.0});
  mesh->add_vertex(8, {0.5, 0.5, 2.0}); // apex

  mesh->add_cell(0, CellFamily::Hex, {0, 1, 2, 3, 4, 5, 6, 7});
  mesh->add_cell(1, CellFamily::Pyramid, {4, 5, 6, 7, 8});
  return mesh;
}

} // namespace

TEST(RedGreenClosure3D, ClosureComputesTetraFaceGreenSpecForNeighbor) {
  auto mesh = make_two_tetra_share_face();

  std::vector<MarkType> marks(mesh->n_cells(), MarkType::NONE);
  marks[0] = MarkType::REFINE;

  AdaptivityOptions options = base_options();
  ClosureConformityEnforcer enforcer;
  (void)enforcer.enforce_conformity(*mesh, marks, options);

  const auto specs = enforcer.get_cell_refinement_specs();
  ASSERT_TRUE(specs.count(0u) > 0);
  ASSERT_TRUE(specs.count(1u) > 0);

  EXPECT_EQ(specs.at(0).pattern, RefinementPattern::RED);
  EXPECT_EQ(specs.at(1).pattern, RefinementPattern::GREEN);
  EXPECT_EQ(specs.at(1).selector, 3u) << "Face (0,1,2) is opposite local vertex 3";
  EXPECT_EQ(marks[1], MarkType::REFINE);
}

TEST(RedGreenClosure3D, AdaptivityManagerRefinesTetraNeighborWithFaceGreenAndProducesConformingMesh) {
  auto mesh = make_two_tetra_share_face();

  AdaptivityOptions options = base_options();
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(manager.get_last_marks().size(), 2u);
  EXPECT_EQ(manager.get_last_marks()[1], MarkType::REFINE);

  // 1 RED tetra -> 8 children, 1 face-GREEN tetra -> 4 children => 12 total.
  EXPECT_EQ(mesh->n_cells(), 12u);
  // Original vertices (5) + 6 unique edge midpoints from the RED parent => 11.
  EXPECT_EQ(mesh->n_vertices(), 11u);
  EXPECT_TRUE(ConformityUtils::is_mesh_conforming(*mesh));

  // Shared-face edge midpoints must exist exactly once (no duplicate vertices).
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.0, 0.0}), 1u);  // edge (0,1)
  EXPECT_EQ(count_vertices_near(*mesh, {0.0, 0.5, 0.0}), 1u);  // edge (0,2)
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.5, 0.0}), 1u);  // edge (1,2)
}

TEST(RedGreenClosure3D, ClosureComputesHex2DSplitSpecForFaceNeighbor) {
  auto mesh = make_two_hex_stacked();

  std::vector<MarkType> marks(mesh->n_cells(), MarkType::NONE);
  marks[0] = MarkType::REFINE;

  AdaptivityOptions options = base_options();
  ClosureConformityEnforcer enforcer;
  (void)enforcer.enforce_conformity(*mesh, marks, options);

  const auto specs = enforcer.get_cell_refinement_specs();
  ASSERT_TRUE(specs.count(1u) > 0);
  EXPECT_EQ(marks[1], MarkType::REFINE);

  // Shared face is normal to Z, so closure should choose a 2-direction split (X+Y).
  EXPECT_EQ(specs.at(1).pattern, RefinementPattern::ANISOTROPIC);
  EXPECT_EQ(specs.at(1).selector, 3u) << "Hex 2D split selector uses bitmask X=1,Y=2,Z=4; X+Y => 3";
}

TEST(RedGreenClosure3D, AdaptivityManagerRefinesHexNeighborWith2DSplitAndProducesConformingMesh) {
  auto mesh = make_two_hex_stacked();

  AdaptivityOptions options = base_options();
  // Keep inversion/Jacobian checks but relax the aggregate quality threshold so the
  // minimal 2D split is committed (test focuses on topology/conformity).
  options.min_refined_quality = 0.0;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(manager.get_last_marks().size(), 2u);
  EXPECT_EQ(manager.get_last_marks()[1], MarkType::REFINE);

  // 1 regular hex refine -> 8 children, 1 2D split -> 4 children => 12 total.
  EXPECT_EQ(mesh->n_cells(), 12u);
  // Original vertices (12) + 19 (regular) + 5 (top-only) = 36 unique vertices.
  EXPECT_EQ(mesh->n_vertices(), 36u);
  EXPECT_TRUE(ConformityUtils::is_mesh_conforming(*mesh));

  // Shared-face vertices must not be duplicated.
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.0, 1.0}), 1u);  // edge midpoint (4,5)
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.5, 1.0}), 1u);  // face center (4,5,6,7)
}

TEST(RedGreenClosure3D, ClosureComputesWedgeGreenSpecForQuadFaceNeighbor) {
  auto mesh = make_two_wedges_share_quad_face();

  std::vector<MarkType> marks(mesh->n_cells(), MarkType::NONE);
  marks[0] = MarkType::REFINE;

  AdaptivityOptions options = base_options();
  ClosureConformityEnforcer enforcer;
  (void)enforcer.enforce_conformity(*mesh, marks, options);

  const auto specs = enforcer.get_cell_refinement_specs();
  ASSERT_TRUE(specs.count(0u) > 0);
  ASSERT_TRUE(specs.count(1u) > 0);
  EXPECT_EQ(specs.at(0).pattern, RefinementPattern::RED);
  EXPECT_EQ(marks[1], MarkType::REFINE);

  // Shared quad face corresponds to base edge (0,1) => selector 0.
  EXPECT_EQ(specs.at(1).pattern, RefinementPattern::GREEN);
  EXPECT_EQ(specs.at(1).selector, 0u);
}

TEST(RedGreenClosure3D, AdaptivityManagerRefinesWedgeNeighborWithGreenAndProducesConformingMesh) {
  auto mesh = make_two_wedges_share_quad_face();

  AdaptivityOptions options = base_options();
  options.min_refined_quality = 0.0;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(manager.get_last_marks().size(), 2u);
  EXPECT_EQ(manager.get_last_marks()[1], MarkType::REFINE);

  // 1 regular wedge refine -> 8 children, 1 GREEN wedge -> 4 children => 12 total.
  EXPECT_EQ(mesh->n_cells(), 12u);
  EXPECT_TRUE(ConformityUtils::is_mesh_conforming(*mesh));

  // Shared face vertices must not be duplicated.
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.0, 0.0}), 1u);  // edge midpoint (0,1)
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.0, 1.0}), 1u);  // edge midpoint (3,4)
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.0, 0.5}), 1u);  // quad face center
}

TEST(RedGreenClosure3D, ClosureComputesWedgeGreenSpecForHexNeighbor) {
  auto mesh = make_hex_wedge_share_quad_face();

  std::vector<MarkType> marks(mesh->n_cells(), MarkType::NONE);
  marks[1] = MarkType::REFINE; // refine the hex

  AdaptivityOptions options = base_options();
  ClosureConformityEnforcer enforcer;
  (void)enforcer.enforce_conformity(*mesh, marks, options);

  const auto specs = enforcer.get_cell_refinement_specs();
  ASSERT_TRUE(specs.count(0u) > 0);
  ASSERT_TRUE(specs.count(1u) > 0);

  // Wedge closure should choose the quad-face-compatible GREEN variant.
  EXPECT_EQ(marks[0], MarkType::REFINE);
  EXPECT_EQ(specs.at(0).pattern, RefinementPattern::GREEN);
  EXPECT_EQ(specs.at(0).selector, 0u);
}

TEST(RedGreenClosure3D, ClosureComputesPyramidBaseSplitSpecForHexNeighbor) {
  auto mesh = make_hex_pyramid_share_face();

  std::vector<MarkType> marks(mesh->n_cells(), MarkType::NONE);
  marks[0] = MarkType::REFINE; // refine the hex

  AdaptivityOptions options = base_options();
  ClosureConformityEnforcer enforcer;
  (void)enforcer.enforce_conformity(*mesh, marks, options);

  const auto specs = enforcer.get_cell_refinement_specs();
  ASSERT_TRUE(specs.count(1u) > 0);
  EXPECT_EQ(marks[1], MarkType::REFINE);
  EXPECT_EQ(specs.at(1).pattern, RefinementPattern::ANISOTROPIC);
}

TEST(RedGreenClosure3D, AdaptivityManagerRefinesPyramidNeighborWithBaseSplitAndProducesConformingMesh) {
  auto mesh = make_hex_pyramid_share_face();

  AdaptivityOptions options = base_options();
  options.min_refined_quality = 0.0;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(manager.get_last_marks().size(), 2u);
  EXPECT_EQ(manager.get_last_marks()[1], MarkType::REFINE);

  // 1 regular hex refine -> 8 children, 1 pyramid base split -> 4 children => 12 total.
  EXPECT_EQ(mesh->n_cells(), 12u);
  EXPECT_TRUE(ConformityUtils::is_mesh_conforming(*mesh));

  // Shared base-face center must not be duplicated.
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.5, 1.0}), 1u);
}

} // namespace test
} // namespace svmp
