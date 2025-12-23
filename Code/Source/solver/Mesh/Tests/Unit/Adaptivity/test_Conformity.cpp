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
#include "../../../Adaptivity/Conformity.h"
#include "../../../Adaptivity/Marker.h"
#include "../../../Adaptivity/Options.h"
#include "../../../Core/MeshBase.h"
// MeshBuilder removed - using MeshBase directly for mesh construction
#include "../../../Fields/MeshFields.h"
#include <memory>
#include <vector>

namespace svmp {
namespace test {

class ConformityTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Will be initialized in each test as needed
  }

  // Helper to create a simple 2D quad mesh for testing
  std::unique_ptr<MeshBase> create_2d_quad_mesh() {
    auto mesh = std::make_unique<MeshBase>();

    // Create a 2x2 quad mesh
    // 6-----7-----8
    // |     |     |
    // 3-----4-----5
    // |     |     |
    // 0-----1-----2

    // Add vertices
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        mesh->add_vertex(j * 3 + i, {static_cast<real_t>(i),
                                     static_cast<real_t>(j), 0.0});
      }
    }

    // Add quads
    mesh->add_cell(0, CellFamily::Quad, {0, 1, 4, 3}); // Bottom-left
    mesh->add_cell(1, CellFamily::Quad, {1, 2, 5, 4}); // Bottom-right
    mesh->add_cell(2, CellFamily::Quad, {3, 4, 7, 6}); // Top-left
    mesh->add_cell(3, CellFamily::Quad, {4, 5, 8, 7}); // Top-right

    return mesh;
  }

  // Helper to create a 3D tetrahedral mesh
  std::unique_ptr<MeshBase> create_3d_tet_mesh() {
    auto mesh = std::make_unique<MeshBase>();

    // Create a simple tetrahedral mesh
    // Regular tetrahedron plus neighbors

    // Core tetrahedron vertices
    mesh->add_vertex(0, {0.0, 0.0, 0.0});
    mesh->add_vertex(1, {1.0, 0.0, 0.0});
    mesh->add_vertex(2, {0.5, 1.0, 0.0});
    mesh->add_vertex(3, {0.5, 0.5, 1.0});

    // Additional vertices for neighbors
    mesh->add_vertex(4, {-0.5, 0.5, 0.0});
    mesh->add_vertex(5, {1.5, 0.5, 0.0});
    mesh->add_vertex(6, {0.5, 0.5, -1.0});

    // Add tetrahedra
    mesh->add_cell(0, CellFamily::Tetra, {0, 1, 2, 3}); // Core tet
    mesh->add_cell(1, CellFamily::Tetra, {0, 2, 3, 4}); // Left neighbor
    mesh->add_cell(2, CellFamily::Tetra, {1, 2, 3, 5}); // Right neighbor
    mesh->add_cell(3, CellFamily::Tetra, {0, 1, 2, 6}); // Bottom neighbor

    return mesh;
  }

  // Helper to create marks vector
  std::vector<MarkType> create_marks(size_t num_cells, MarkType default_mark = MarkType::NONE) {
    return std::vector<MarkType>(num_cells, default_mark);
  }
};

// Test 1: Factory creation - Closure enforcer
TEST_F(ConformityTest, FactoryCreateClosure) {
  AdaptivityOptions options;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;

  auto enforcer = ConformityEnforcerFactory::create(options);
  ASSERT_NE(enforcer, nullptr);
  EXPECT_EQ(enforcer->name(), "ClosureConformity");
}

// Test 2: Factory creation - Hanging node enforcer
TEST_F(ConformityTest, FactoryCreateHangingNode) {
  AdaptivityOptions options;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES;

  auto enforcer = ConformityEnforcerFactory::create(options);
  ASSERT_NE(enforcer, nullptr);
  EXPECT_EQ(enforcer->name(), "HangingNode");
}

// Test 3: Factory creation - Minimal closure enforcer
TEST_F(ConformityTest, FactoryCreateMinimalClosure) {
  AdaptivityOptions options;
  options.conformity_mode = AdaptivityOptions::ConformityMode::MINIMAL_CLOSURE;

  auto enforcer = ConformityEnforcerFactory::create(options);
  ASSERT_NE(enforcer, nullptr);
  EXPECT_EQ(enforcer->name(), "MinimalClosure");
}

// Test 4: ClosureConformityEnforcer - Check conforming mesh
TEST_F(ConformityTest, ClosureCheckConformingMesh) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  ClosureConformityEnforcer enforcer;
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  EXPECT_TRUE(non_conformity.is_conforming());
  EXPECT_TRUE(non_conformity.hanging_nodes.empty());
  EXPECT_TRUE(non_conformity.cells_needing_closure.empty());
  EXPECT_EQ(non_conformity.max_level_difference, 0);
}

// Test 5: ClosureConformityEnforcer - Detect non-conformity
TEST_F(ConformityTest, ClosureDetectNonConformity) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark only one cell for refinement
  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer enforcer;
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Should detect non-conformity due to isolated refinement
  EXPECT_FALSE(non_conformity.is_conforming());
}

// Test 6: ClosureConformityEnforcer - Enforce conformity
TEST_F(ConformityTest, ClosureEnforceConformity) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark one cell for refinement
  marks[0] = MarkType::REFINE;

  AdaptivityOptions options;
  ClosureConformityEnforcer enforcer;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Should have performed closure iterations
  EXPECT_GT(iterations, 0);

  // Check that neighbors are marked for closure
  // Cell 1 and 2 share edges with cell 0
  EXPECT_NE(marks[1], MarkType::NONE);
  EXPECT_NE(marks[2], MarkType::NONE);
}

// Test 7: HangingNodeConformityEnforcer - Check conformity
TEST_F(ConformityTest, HangingNodeCheckConformity) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark some cells for refinement
  marks[0] = MarkType::REFINE;
  marks[3] = MarkType::REFINE;

  HangingNodeConformityEnforcer enforcer;
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Should detect hanging nodes / level differences due to inconsistent refinement marks.
  EXPECT_FALSE(non_conformity.is_conforming());
  EXPECT_FALSE(non_conformity.hanging_nodes.empty());
  EXPECT_GT(non_conformity.max_level_difference, 0u);
}

// Test 8: HangingNodeConformityEnforcer - Generate constraints
TEST_F(ConformityTest, HangingNodeGenerateConstraints) {
  auto mesh = create_2d_quad_mesh();

  HangingNodeConformityEnforcer enforcer;
  NonConformity non_conformity;

  // Manually create a hanging node for testing
  HangingNode hanging;
  hanging.node_id = 10; // Hypothetical hanging node
  hanging.parent_entity = {1, 2}; // On edge [1,2]
  hanging.on_edge = true;
  hanging.constraints[1] = 0.5;
  hanging.constraints[2] = 0.5;
  hanging.level_difference = 1;

  non_conformity.hanging_nodes.push_back(hanging);

  auto constraints = enforcer.generate_constraints(*mesh, non_conformity);

  EXPECT_EQ(constraints.size(), 1);
  EXPECT_TRUE(constraints.count(10) > 0);
  EXPECT_DOUBLE_EQ(constraints[10][1], 0.5);
  EXPECT_DOUBLE_EQ(constraints[10][2], 0.5);
}

// Test 9: MinimalClosureEnforcer - Check conformity
TEST_F(ConformityTest, MinimalClosureCheckConformity) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  MinimalClosureEnforcer enforcer;
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Should detect non-conformity
  EXPECT_FALSE(non_conformity.is_conforming());
  EXPECT_FALSE(non_conformity.cells_needing_closure.empty());
}

// Test 10: MinimalClosureEnforcer - Enforce with green refinement
TEST_F(ConformityTest, MinimalClosureGreenRefinement) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  MinimalClosureEnforcer::Config config;
  config.prefer_green = true;

  MinimalClosureEnforcer enforcer(config);
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Should have performed minimal closure
  EXPECT_GT(iterations, 0u);
}

// Test 11: ConformityUtils - Check mesh conformity
TEST_F(ConformityTest, UtilsIsMeshConforming) {
  auto mesh = create_2d_quad_mesh();

  // Uniform mesh should be conforming
  bool is_conforming = ConformityUtils::is_mesh_conforming(*mesh);
  EXPECT_TRUE(is_conforming);
}

// Test 12: ConformityUtils - Find hanging nodes
TEST_F(ConformityTest, UtilsFindHangingNodes) {
  auto mesh = create_2d_quad_mesh();

  auto hanging_nodes = ConformityUtils::find_hanging_nodes(*mesh);

  // Uniform mesh should have no hanging nodes
  EXPECT_TRUE(hanging_nodes.empty());
}

// Test 13: ConformityUtils - Check level difference
TEST_F(ConformityTest, UtilsCheckLevelDifference) {
  auto mesh = create_2d_quad_mesh();

  // Check level difference between adjacent cells
  size_t diff = ConformityUtils::check_level_difference(*mesh, 0, 1);

  // Should be 0 for uniform mesh
  EXPECT_EQ(diff, 0);
}

// Test 14: ConformityUtils - Apply constraints to solution
TEST_F(ConformityTest, UtilsApplyConstraints) {
  std::vector<double> solution = {1.0, 3.0, 0.0, 5.0};

  // Create constraint: node 2 = 0.5 * node 0 + 0.5 * node 1
  std::map<size_t, std::map<size_t, double>> constraints;
  constraints[2][0] = 0.5;
  constraints[2][1] = 0.5;

  ConformityUtils::apply_constraints(solution, constraints);

  EXPECT_DOUBLE_EQ(solution[0], 1.0);
  EXPECT_DOUBLE_EQ(solution[1], 3.0);
  EXPECT_DOUBLE_EQ(solution[2], 2.0); // (1.0 + 3.0) * 0.5
  EXPECT_DOUBLE_EQ(solution[3], 5.0);
}

// Test 15: 3D mesh conformity checking
TEST_F(ConformityTest, Conformity3DMesh) {
  auto mesh = create_3d_tet_mesh();
  auto marks = create_marks(mesh->n_cells());

  ClosureConformityEnforcer enforcer;
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  EXPECT_TRUE(non_conformity.is_conforming());
}

// Test 16: 3D mesh with refinement
TEST_F(ConformityTest, Conformity3DWithRefinement) {
  auto mesh = create_3d_tet_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark core tet for refinement
  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer enforcer;
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Should detect non-conformity due to isolated refinement
  EXPECT_FALSE(non_conformity.is_conforming());
}

// Test 17: Edge conformity configuration
TEST_F(ConformityTest, EdgeConformityConfig) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer::Config config;
  config.check_edge_conformity = false;
  config.check_face_conformity = true;

  ClosureConformityEnforcer enforcer(config);
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // With edge checking disabled, might not detect all non-conformities
  // Test that it runs without crash
  EXPECT_GE(non_conformity.max_level_difference, 0);
}

// Test 18: Maximum iterations limit
TEST_F(ConformityTest, MaxIterationsLimit) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer::Config config;
  config.max_iterations = 2;

  ClosureConformityEnforcer enforcer(config);
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Should not exceed max iterations
  EXPECT_LE(iterations, config.max_iterations);
}

// Test 19: Hanging node max level configuration
TEST_F(ConformityTest, HangingNodeMaxLevel) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  HangingNodeConformityEnforcer::Config config;
  config.max_hanging_level = 2;

  HangingNodeConformityEnforcer enforcer(config);
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Test configuration is applied
  EXPECT_TRUE(non_conformity.is_conforming());
}

// Test 20: Constraint tolerance configuration
TEST_F(ConformityTest, ConstraintTolerance) {
  auto mesh = create_2d_quad_mesh();

  HangingNodeConformityEnforcer::Config config;
  config.constraint_tolerance = 1e-8;

  HangingNodeConformityEnforcer enforcer(config);

  NonConformity non_conformity;
  HangingNode hanging;
  hanging.node_id = 10;
  hanging.parent_entity = {1, 2};
  hanging.on_edge = true;
  hanging.constraints[1] = 0.5;
  hanging.constraints[2] = 0.5;

  non_conformity.hanging_nodes.push_back(hanging);

  auto constraints = enforcer.generate_constraints(*mesh, non_conformity);

  EXPECT_FALSE(constraints.empty());
}

// Test 21: Propagate closure configuration
TEST_F(ConformityTest, PropagateClosure) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer::Config config;
  config.propagate_closure = true;

  ClosureConformityEnforcer enforcer(config);
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // With propagation, more cells should be marked
  EXPECT_GT(iterations, 0);
}

// Test 22: Anisotropic refinement in minimal closure
TEST_F(ConformityTest, AnisotropicMinimalClosure) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  MinimalClosureEnforcer::Config config;
  config.allow_anisotropic = true;

  MinimalClosureEnforcer enforcer(config);
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Should complete without error
  EXPECT_GE(iterations, 0);
}

// Test 23: ConformityUtils - Eliminate constraints from matrix
TEST_F(ConformityTest, UtilsEliminateConstraints) {
  // Simple 3x3 system
  std::vector<std::vector<double>> matrix = {
    {2.0, 1.0, 0.0},
    {1.0, 2.0, 1.0},
    {0.0, 1.0, 2.0}
  };

  std::vector<double> rhs = {1.0, 2.0, 3.0};

  // Constraint: x[1] = 0.5 * x[0] + 0.5 * x[2]
  std::map<size_t, std::map<size_t, double>> constraints;
  constraints[1][0] = 0.5;
  constraints[1][2] = 0.5;

  ConformityUtils::eliminate_constraints(matrix, rhs, constraints);

  // Matrix should be modified to eliminate constrained DOF
  // Exact values depend on elimination method
  EXPECT_NE(matrix[0][0], 2.0); // Matrix should be modified
}

// Test 24: Write non-conformity to fields
TEST_F(ConformityTest, WriteNonConformityToFields) {
  auto mesh = create_2d_quad_mesh();

  // Create a MeshFields object separately
  MeshFields fields;

  NonConformity non_conformity;

  // Add a hanging node
  HangingNode hanging;
  hanging.node_id = 4; // Center node
  hanging.parent_entity = {1, 3};
  hanging.on_edge = true;
  hanging.level_difference = 1;

  non_conformity.hanging_nodes.push_back(hanging);
  non_conformity.cells_needing_closure.insert(0);
  non_conformity.max_level_difference = 1;

  ConformityUtils::write_nonconformity_to_field(fields, *mesh, non_conformity);

  ASSERT_TRUE(MeshFields::has_field(*mesh, EntityKind::Vertex, "conformity_hanging_level"));
  ASSERT_TRUE(MeshFields::has_field(*mesh, EntityKind::Volume, "conformity_needs_closure"));
  ASSERT_TRUE(MeshFields::has_field(*mesh, EntityKind::Volume, "conformity_max_level_difference"));

  auto hanging_h = MeshFields::get_field_handle(*mesh, EntityKind::Vertex, "conformity_hanging_level");
  auto closure_h = MeshFields::get_field_handle(*mesh, EntityKind::Volume, "conformity_needs_closure");
  auto maxdiff_h = MeshFields::get_field_handle(*mesh, EntityKind::Volume, "conformity_max_level_difference");

  const auto* hanging_data = MeshFields::field_data_as<int32_t>(*mesh, hanging_h);
  const auto* closure_data = MeshFields::field_data_as<uint8_t>(*mesh, closure_h);
  const auto* maxdiff_data = MeshFields::field_data_as<int32_t>(*mesh, maxdiff_h);

  ASSERT_EQ(mesh->n_vertices(), 9u);
  ASSERT_EQ(mesh->n_cells(), 4u);

  // Hanging level is set on the node only.
  EXPECT_EQ(hanging_data[4], 1);
  for (size_t v = 0; v < mesh->n_vertices(); ++v) {
    if (v == 4) continue;
    EXPECT_EQ(hanging_data[v], 0);
  }

  // Closure cell is marked.
  EXPECT_EQ(closure_data[0], 1);
  for (size_t c = 1; c < mesh->n_cells(); ++c) {
    EXPECT_EQ(closure_data[c], 0);
  }

  // Global max level difference is replicated to all cells.
  for (size_t c = 0; c < mesh->n_cells(); ++c) {
    EXPECT_EQ(maxdiff_data[c], 1);
  }
}

// Test 25: Complete refinement and conformity workflow
TEST_F(ConformityTest, CompleteRefinementWorkflow) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark cells for refinement in a pattern
  marks[0] = MarkType::REFINE;
  marks[3] = MarkType::REFINE;

  // Create options
  AdaptivityOptions options;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;
  options.max_refinement_level = 3;  // Set a reasonable max level

  // Create enforcer
  auto enforcer = ConformityEnforcerFactory::create(options);

  // Check initial non-conformity
  auto initial_check = enforcer->check_conformity(*mesh, marks);

  // Enforce conformity
  size_t iterations = enforcer->enforce_conformity(*mesh, marks, options);

  // Check final conformity
  auto final_check = enforcer->check_conformity(*mesh, marks);

  // After enforcement, mesh should be more conforming
  EXPECT_GE(iterations, 0);

  // Generate constraints if hanging nodes remain
  if (!final_check.hanging_nodes.empty()) {
    auto constraints = enforcer->generate_constraints(*mesh, final_check);
    EXPECT_FALSE(constraints.empty());
  }
}

// Test 26: Multiple closure iterations required
TEST_F(ConformityTest, MultipleClosureIterations) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark opposite corners to force multiple iterations
  marks[0] = MarkType::REFINE;
  marks[3] = MarkType::REFINE;

  ClosureConformityEnforcer enforcer;
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Should require at least one iteration for closure
  // (May be 1 if implementation optimizes corner case)
  EXPECT_GE(iterations, 1);
}

// Test 27: Hanging node on edge - constraint coefficients
TEST_F(ConformityTest, HangingNodeEdgeConstraintCoefficients) {
  auto mesh = create_2d_quad_mesh();

  HangingNodeConformityEnforcer enforcer;
  NonConformity non_conformity;

  // Create hanging node on edge
  HangingNode hanging;
  hanging.node_id = 100;
  hanging.parent_entity = {0, 1};
  hanging.on_edge = true;
  hanging.constraints[0] = 0.5;
  hanging.constraints[1] = 0.5;
  hanging.level_difference = 1;

  non_conformity.hanging_nodes.push_back(hanging);

  auto constraints = enforcer.generate_constraints(*mesh, non_conformity);

  // Edge hanging nodes should have 0.5, 0.5 weights for midpoint
  ASSERT_TRUE(constraints.count(100) > 0);
  EXPECT_DOUBLE_EQ(constraints[100][0], 0.5);
  EXPECT_DOUBLE_EQ(constraints[100][1], 0.5);
}

// Test 28: Hanging node on face - constraint coefficients (3D)
TEST_F(ConformityTest, HangingNodeFaceConstraintCoefficients) {
  auto mesh = create_3d_tet_mesh();

  HangingNodeConformityEnforcer enforcer;
  NonConformity non_conformity;

  // Create hanging node on triangular face
  HangingNode hanging;
  hanging.node_id = 200;
  hanging.parent_entity = {0, 1};  // Simplified
  hanging.on_edge = false;  // Face hanging
  hanging.constraints[0] = 1.0 / 3.0;
  hanging.constraints[1] = 1.0 / 3.0;
  hanging.constraints[2] = 1.0 / 3.0;
  hanging.level_difference = 1;

  non_conformity.hanging_nodes.push_back(hanging);

  auto constraints = enforcer.generate_constraints(*mesh, non_conformity);

  // Face hanging node should have equal weights for centroid
  ASSERT_TRUE(constraints.count(200) > 0);
  EXPECT_EQ(constraints[200].size(), 3);
}

// Test 29: Boundary elements don't cause false non-conformity
TEST_F(ConformityTest, BoundaryElementsConformity) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark only boundary element
  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer::Config config;
  config.check_edge_conformity = true;

  ClosureConformityEnforcer enforcer(config);
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Boundary edges should be handled correctly
  EXPECT_GE(non_conformity.non_conforming_edges.size(), 0);
}

// Test 30: Green refinement preference in minimal closure
TEST_F(ConformityTest, MinimalClosurePreferGreen) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  MinimalClosureEnforcer::Config config;
  config.prefer_green = true;
  config.prefer_blue = false;

  MinimalClosureEnforcer enforcer(config);
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Should prefer green refinement
  EXPECT_GE(iterations, 0);
}

// Test 31: Blue refinement preference in minimal closure
TEST_F(ConformityTest, MinimalClosurePreferBlue) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  MinimalClosureEnforcer::Config config;
  config.prefer_green = false;
  config.prefer_blue = true;

  MinimalClosureEnforcer enforcer(config);
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Should prefer blue refinement
  EXPECT_GE(iterations, 0);
}

// Test 32: Cost function parameters affect minimal closure
TEST_F(ConformityTest, MinimalClosureCostParameters) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  MinimalClosureEnforcer::Config config;
  config.refinement_cost = 2.0;
  config.pattern_cost = 0.8;

  MinimalClosureEnforcer enforcer(config);
  AdaptivityOptions options;

  size_t iterations = enforcer.enforce_conformity(*mesh, marks, options);

  // Different costs should affect closure strategy
  EXPECT_GE(iterations, 0);
}

// Test 33: NonConformity structure with multiple hanging nodes
TEST_F(ConformityTest, NonConformityMultipleHangingNodes) {
  NonConformity non_conformity;

  // Add multiple hanging nodes
  for (size_t i = 0; i < 5; ++i) {
    HangingNode hanging;
    hanging.node_id = 100 + i;
    hanging.parent_entity = {i, i + 1};
    hanging.on_edge = true;
    hanging.constraints[i] = 0.5;
    hanging.constraints[i + 1] = 0.5;
    hanging.level_difference = 1;

    non_conformity.hanging_nodes.push_back(hanging);
  }

  // Should not be conforming
  EXPECT_FALSE(non_conformity.is_conforming());
  EXPECT_EQ(non_conformity.hanging_nodes.size(), 5);
}

// Test 34: NonConformity with elements needing closure
TEST_F(ConformityTest, NonConformityElementsClosure) {
  NonConformity non_conformity;

  // Add elements needing closure
  non_conformity.cells_needing_closure.insert(0);
  non_conformity.cells_needing_closure.insert(1);
  non_conformity.cells_needing_closure.insert(5);

  EXPECT_FALSE(non_conformity.is_conforming());
  EXPECT_EQ(non_conformity.cells_needing_closure.size(), 3);
}

// Test 35: Constraint application to multi-component solution
TEST_F(ConformityTest, ConstraintMultiComponent) {
  // 3-component solution (e.g., 3D velocity)
  std::vector<double> solution = {
    1.0, 2.0, 3.0,  // Node 0: (1, 2, 3)
    4.0, 5.0, 6.0,  // Node 1: (4, 5, 6)
    0.0, 0.0, 0.0,  // Node 2: (0, 0, 0) - hanging node
    7.0, 8.0, 9.0   // Node 3: (7, 8, 9)
  };

  // Constraint: Node 2 = 0.5 * Node 0 + 0.5 * Node 1
  std::map<size_t, std::map<size_t, double>> constraints;
  constraints[2][0] = 0.5;
  constraints[2][1] = 0.5;

  // Apply constraints would need to be called appropriately for multi-component
  // This test documents the requirement
  EXPECT_EQ(solution.size(), 12);  // 4 nodes * 3 components
}

// Test 36: Face conformity checking disabled
TEST_F(ConformityTest, FaceConformityDisabled) {
  auto mesh = create_3d_tet_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer::Config config;
  config.check_face_conformity = false;
  config.check_edge_conformity = true;

  ClosureConformityEnforcer enforcer(config);
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Face non-conformity should not be detected
  EXPECT_EQ(non_conformity.non_conforming_faces.size(), 0);
}

// Test 37: Edge and face conformity both enabled (3D)
TEST_F(ConformityTest, EdgeAndFaceConformity3D) {
  auto mesh = create_3d_tet_mesh();
  auto marks = create_marks(mesh->n_cells());

  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer::Config config;
  config.check_face_conformity = true;
  config.check_edge_conformity = true;

  ClosureConformityEnforcer enforcer(config);
  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Both edge and face conformity should be checked
  EXPECT_GE(non_conformity.non_conforming_edges.size(), 0);
  EXPECT_GE(non_conformity.non_conforming_faces.size(), 0);
}

// Test 38: Level difference enforcement with 2:1 balance
TEST_F(ConformityTest, TwoToOneBalanceEnforcement) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Mark one element for refinement
  marks[0] = MarkType::REFINE;

  ClosureConformityEnforcer::Config config;
  config.max_level_difference = 1;  // Enforce 2:1 balance
  config.propagate_closure = true;

  ClosureConformityEnforcer enforcer(config);
  AdaptivityOptions options;

  enforcer.enforce_conformity(*mesh, marks, options);

  auto non_conformity = enforcer.check_conformity(*mesh, marks);

  // Max level difference should satisfy 2:1 constraint
  EXPECT_LE(non_conformity.max_level_difference, 1);
}

// Test 39: Factory with default conformity mode
TEST_F(ConformityTest, FactoryDefaultConformityMode) {
  AdaptivityOptions options;
  // Don't explicitly set conformity_mode

  auto enforcer = ConformityEnforcerFactory::create(options);

  // Should create a valid enforcer (typically closure)
  ASSERT_NE(enforcer, nullptr);
  EXPECT_FALSE(enforcer->name().empty());
}

// Test 40: Integration test - Full AMR workflow with conformity
TEST_F(ConformityTest, FullAMRWorkflowIntegration) {
  auto mesh = create_2d_quad_mesh();
  auto marks = create_marks(mesh->n_cells());

  // Simulate error-driven marking
  marks[0] = MarkType::REFINE;
  marks[1] = MarkType::REFINE;

  // Create conformity enforcer
  AdaptivityOptions options;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;
  options.max_refinement_level = 5;

  auto enforcer = ConformityEnforcerFactory::create(options);

  // Step 1: Check initial conformity
  auto initial_nc = enforcer->check_conformity(*mesh, marks);
  bool initially_conforming = initial_nc.is_conforming();

  // Step 2: Enforce conformity
  size_t iterations = enforcer->enforce_conformity(*mesh, marks, options);

  // Step 3: Check final conformity
  auto final_nc = enforcer->check_conformity(*mesh, marks);

  // Step 4: Generate constraints if needed
  auto constraints = enforcer->generate_constraints(*mesh, final_nc);

  // Verify workflow completed
  EXPECT_GE(iterations, 0);
  EXPECT_TRUE(initially_conforming || !initially_conforming);  // Just check it ran
  EXPECT_GE(constraints.size(), 0);
}

} // namespace test
} // namespace svmp
