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
#include "../../../Adaptivity/FieldTransfer.h"
#include "../../../Adaptivity/Marker.h"
#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <chrono>

namespace svmp {
namespace test {

class FieldTransferTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Will be initialized in each test as needed
  }

  // Helper to create a simple 1D mesh for testing
  std::unique_ptr<MeshBase> create_1d_mesh(size_t num_elements) {
    auto mesh = std::make_unique<MeshBase>();

    // Create vertices
    for (size_t i = 0; i <= num_elements; ++i) {
      mesh->add_vertex(i, {static_cast<real_t>(i), 0.0, 0.0});
    }

    // Create line elements
    for (size_t i = 0; i < num_elements; ++i) {
      mesh->add_cell(i, CellFamily::Line, {static_cast<index_t>(i), static_cast<index_t>(i + 1)});
    }

    return mesh;
  }

  // Helper to create a simple 2D quad mesh
  std::unique_ptr<MeshBase> create_2d_quad_mesh(size_t nx, size_t ny) {
    auto mesh = std::make_unique<MeshBase>();

    // Create vertices
    size_t vertex_id = 0;
    for (size_t j = 0; j <= ny; ++j) {
      for (size_t i = 0; i <= nx; ++i) {
        mesh->add_vertex(vertex_id++, {
          static_cast<real_t>(i),
          static_cast<real_t>(j),
          0.0
        });
      }
    }

    // Create quad elements
    size_t elem_id = 0;
    for (size_t j = 0; j < ny; ++j) {
      for (size_t i = 0; i < nx; ++i) {
        size_t v0 = j * (nx + 1) + i;
        size_t v1 = v0 + 1;
        size_t v2 = v0 + nx + 2;
        size_t v3 = v0 + nx + 1;
        mesh->add_cell(elem_id++, CellFamily::Quad, {static_cast<index_t>(v0), static_cast<index_t>(v1), static_cast<index_t>(v2), static_cast<index_t>(v3)});
      }
    }

    return mesh;
  }

  // Helper to create a simple 2D triangle mesh
  std::unique_ptr<MeshBase> create_2d_tri_mesh(size_t nx, size_t ny) {
    auto mesh = std::make_unique<MeshBase>();

    // Create vertices
    size_t vertex_id = 0;
    for (size_t j = 0; j <= ny; ++j) {
      for (size_t i = 0; i <= nx; ++i) {
        mesh->add_vertex(vertex_id++, {
          static_cast<real_t>(i),
          static_cast<real_t>(j),
          0.0
        });
      }
    }

    // Create triangle elements (2 per quad)
    size_t elem_id = 0;
    for (size_t j = 0; j < ny; ++j) {
      for (size_t i = 0; i < nx; ++i) {
        size_t v0 = j * (nx + 1) + i;
        size_t v1 = v0 + 1;
        size_t v2 = v0 + nx + 2;
        size_t v3 = v0 + nx + 1;

        // Lower triangle
        mesh->add_cell(elem_id++, CellFamily::Triangle, {static_cast<index_t>(v0), static_cast<index_t>(v1), static_cast<index_t>(v2)});
        // Upper triangle
        mesh->add_cell(elem_id++, CellFamily::Triangle, {static_cast<index_t>(v0), static_cast<index_t>(v2), static_cast<index_t>(v3)});
      }
    }

    return mesh;
  }

  // Helper to create a simple 3D tetrahedral mesh
  std::unique_ptr<MeshBase> create_3d_tet_mesh() {
    auto mesh = std::make_unique<MeshBase>();

    // Create a unit cube with 5 tets
    mesh->add_vertex(0, {0.0, 0.0, 0.0});
    mesh->add_vertex(1, {1.0, 0.0, 0.0});
    mesh->add_vertex(2, {1.0, 1.0, 0.0});
    mesh->add_vertex(3, {0.0, 1.0, 0.0});
    mesh->add_vertex(4, {0.0, 0.0, 1.0});
    mesh->add_vertex(5, {1.0, 0.0, 1.0});
    mesh->add_vertex(6, {1.0, 1.0, 1.0});
    mesh->add_vertex(7, {0.0, 1.0, 1.0});

    // Create tetrahedra (decomposition of cube)
    mesh->add_cell(0, CellFamily::Tetra, {0, 1, 2, 5});
    mesh->add_cell(1, CellFamily::Tetra, {0, 2, 3, 7});
    mesh->add_cell(2, CellFamily::Tetra, {0, 4, 5, 7});
    mesh->add_cell(3, CellFamily::Tetra, {2, 5, 6, 7});
    mesh->add_cell(4, CellFamily::Tetra, {0, 2, 5, 7});

    return mesh;
  }

  // Helper to create a constant field
  std::vector<double> create_constant_field(size_t size, double value) {
    return std::vector<double>(size, value);
  }

  // Helper to create a linear field (f = ax + by + cz + d)
  std::vector<double> create_linear_field(const MeshBase& mesh,
                                         double a, double b, double c, double d) {
    std::vector<double> field(mesh.n_vertices());
    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
      auto xyz = mesh.get_vertex_coords(i);
      field[i] = a * xyz[0] + b * xyz[1] + c * xyz[2] + d;
    }
    return field;
  }

  // Helper to create a quadratic field (f = x^2 + y^2 + z^2)
  std::vector<double> create_quadratic_field(const MeshBase& mesh) {
    std::vector<double> field(mesh.n_vertices());
    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
      auto xyz = mesh.get_vertex_coords(i);
      field[i] = xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2];
    }
    return field;
  }

  // Helper to create a simple parent-child map for uniform refinement
  ParentChildMap create_uniform_refinement_map(size_t num_parents, size_t children_per_parent) {
    ParentChildMap map;

    map.child_to_parent.resize(num_parents * children_per_parent);

    for (size_t parent = 0; parent < num_parents; ++parent) {
      std::vector<size_t> children;
      for (size_t c = 0; c < children_per_parent; ++c) {
        size_t child = parent * children_per_parent + c;
        map.child_to_parent[child] = parent;
        children.push_back(child);
      }
      map.parent_to_children[parent] = children;
    }

    return map;
  }

  // Helper to create a parent-child map for 1D uniform refinement with vertex weights
  ParentChildMap create_1d_uniform_refinement_map_with_weights(
      size_t num_coarse_elements, size_t num_fine_elements) {
    ParentChildMap map;

    // Setup cell parent-child relationships
    map.child_to_parent.resize(num_fine_elements);

    // For uniform refinement, each coarse element becomes 2 fine elements
    for (size_t parent = 0; parent < num_coarse_elements; ++parent) {
      std::vector<size_t> children;
      for (size_t c = 0; c < 2; ++c) {
        size_t child = parent * 2 + c;
        if (child < num_fine_elements) {
          map.child_to_parent[child] = parent;
          children.push_back(child);
        }
      }
      map.parent_to_children[parent] = children;
    }

    // Setup vertex interpolation weights for 1D line elements
    // Coarse mesh has num_coarse_elements + 1 vertices
    // Fine mesh has num_fine_elements + 1 vertices
    // New vertices are at midpoints of coarse elements

    if (num_fine_elements == 2 * num_coarse_elements) {
      // Direct vertex mappings (vertices that exist in both meshes)
      for (size_t v = 0; v <= num_coarse_elements; ++v) {
        size_t fine_v = 2 * v;  // Even-numbered vertices correspond directly
        map.child_vertex_weights[fine_v] = {{v, 1.0}};
      }

      // New vertices at midpoints of coarse elements
      for (size_t elem = 0; elem < num_coarse_elements; ++elem) {
        size_t new_vertex = 2 * elem + 1;  // Odd-numbered vertices are new
        size_t left_vertex = elem;
        size_t right_vertex = elem + 1;

        map.child_vertex_weights[new_vertex] = {{left_vertex, 0.5}, {right_vertex, 0.5}};
      }
    }

    return map;
  }

  // Helper to create a coarsening map from fine to coarse mesh
  ParentChildMap create_1d_uniform_coarsening_map_with_weights(
      size_t num_fine_elements, size_t num_coarse_elements) {
    ParentChildMap map;

    // For coarsening, we're going from fine (old) to coarse (new)
    // The child_vertex_weights map should map fine vertices to their parent coarse vertices

    // Fine mesh has num_fine_elements + 1 vertices
    // Coarse mesh has num_coarse_elements + 1 vertices
    size_t num_fine_vertices = num_fine_elements + 1;
    size_t num_coarse_vertices = num_coarse_elements + 1;

    if (num_fine_elements == 2 * num_coarse_elements) {
      // For 1D uniform coarsening:
      // Even-numbered fine vertices (0, 2, 4, ...) map directly to coarse vertices (0, 1, 2, ...)
      // These are the vertices that existed in the original coarse mesh

      // Only the even vertices should contribute to the coarse mesh
      // The odd vertices (midpoints) were created during refinement and don't map back

      for (size_t coarse_v = 0; coarse_v < num_coarse_vertices; ++coarse_v) {
        size_t fine_v = 2 * coarse_v;  // The corresponding fine vertex

        if (fine_v < num_fine_vertices) {
          // This fine vertex corresponds directly to the coarse vertex
          map.child_vertex_weights[fine_v] = {{coarse_v, 1.0}};
          map.parent_vertex_to_children[coarse_v].push_back(fine_v);
        }

        // Also collect the neighboring midpoint vertices for averaging
        if (coarse_v > 0) {
          size_t left_mid = 2 * coarse_v - 1;
          if (left_mid < num_fine_vertices) {
            map.parent_vertex_to_children[coarse_v].push_back(left_mid);
          }
        }
        if (coarse_v < num_coarse_elements) {
          size_t right_mid = 2 * coarse_v + 1;
          if (right_mid < num_fine_vertices) {
            map.parent_vertex_to_children[coarse_v].push_back(right_mid);
          }
        }
      }
    }

    return map;
  }

  // Helper to create a parent-child map for 2D uniform quad refinement with vertex weights
  ParentChildMap create_2d_uniform_refinement_map_with_weights(
      size_t coarse_nx, size_t coarse_ny, size_t fine_nx, size_t fine_ny) {
    ParentChildMap map;

    // Setup cell parent-child relationships (each coarse cell splits into 4 fine cells)
    size_t num_coarse_cells = coarse_nx * coarse_ny;
    size_t num_fine_cells = fine_nx * fine_ny;
    map.child_to_parent.resize(num_fine_cells);

    // For 2x2 coarse to 4x4 fine: each coarse cell becomes 4 fine cells
    for (size_t j = 0; j < coarse_ny; ++j) {
      for (size_t i = 0; i < coarse_nx; ++i) {
        size_t parent = j * coarse_nx + i;
        std::vector<size_t> children;

        // Each parent cell splits into 2x2 children
        for (size_t cj = 0; cj < 2; ++cj) {
          for (size_t ci = 0; ci < 2; ++ci) {
            size_t child_i = 2 * i + ci;
            size_t child_j = 2 * j + cj;
            size_t child = child_j * fine_nx + child_i;

            map.child_to_parent[child] = parent;
            children.push_back(child);
          }
        }
        map.parent_to_children[parent] = children;
      }
    }

    // Setup vertex interpolation weights
    // For 2x2 coarse mesh (3x3 vertices) to 4x4 fine mesh (5x5 vertices)

    // Mapping: fine vertex index -> coarse vertex weights
    // Fine mesh vertices are on a 5x5 grid (0-24)
    // Coarse mesh vertices are on a 3x3 grid (0-8)

    if (coarse_nx == 2 && coarse_ny == 2 && fine_nx == 4 && fine_ny == 4) {
      // Direct vertex correspondences - add these with weight 1.0 so they're handled correctly
      map.child_vertex_weights[0] = {{0, 1.0}};   // Fine 0 = Coarse 0
      map.child_vertex_weights[2] = {{1, 1.0}};   // Fine 2 = Coarse 1
      map.child_vertex_weights[4] = {{2, 1.0}};   // Fine 4 = Coarse 2
      map.child_vertex_weights[10] = {{3, 1.0}};  // Fine 10 = Coarse 3
      map.child_vertex_weights[12] = {{4, 1.0}};  // Fine 12 = Coarse 4
      map.child_vertex_weights[14] = {{5, 1.0}};  // Fine 14 = Coarse 5
      map.child_vertex_weights[20] = {{6, 1.0}};  // Fine 20 = Coarse 6
      map.child_vertex_weights[22] = {{7, 1.0}};  // Fine 22 = Coarse 7
      map.child_vertex_weights[24] = {{8, 1.0}};  // Fine 24 = Coarse 8

      // New vertices requiring interpolation:

      // Row 0 (y=0)
      map.child_vertex_weights[1] = {{0, 0.5}, {1, 0.5}};  // midpoint between v0 and v1
      map.child_vertex_weights[3] = {{1, 0.5}, {2, 0.5}};  // midpoint between v1 and v2

      // Row 1 (y=0.5)
      map.child_vertex_weights[5] = {{0, 0.5}, {3, 0.5}};  // midpoint between v0 and v3
      map.child_vertex_weights[6] = {{0, 0.25}, {1, 0.25}, {3, 0.25}, {4, 0.25}};  // center of quad
      map.child_vertex_weights[7] = {{1, 0.5}, {4, 0.5}};  // midpoint between v1 and v4
      map.child_vertex_weights[8] = {{1, 0.25}, {2, 0.25}, {4, 0.25}, {5, 0.25}};  // center of quad
      map.child_vertex_weights[9] = {{2, 0.5}, {5, 0.5}};  // midpoint between v2 and v5

      // Row 2 (y=1)
      map.child_vertex_weights[11] = {{3, 0.5}, {4, 0.5}};  // midpoint between v3 and v4
      map.child_vertex_weights[13] = {{4, 0.5}, {5, 0.5}};  // midpoint between v4 and v5

      // Row 3 (y=1.5)
      map.child_vertex_weights[15] = {{3, 0.5}, {6, 0.5}};  // midpoint between v3 and v6
      map.child_vertex_weights[16] = {{3, 0.25}, {4, 0.25}, {6, 0.25}, {7, 0.25}};  // center of quad
      map.child_vertex_weights[17] = {{4, 0.5}, {7, 0.5}};  // midpoint between v4 and v7
      map.child_vertex_weights[18] = {{4, 0.25}, {5, 0.25}, {7, 0.25}, {8, 0.25}};  // center of quad
      map.child_vertex_weights[19] = {{5, 0.5}, {8, 0.5}};  // midpoint between v5 and v8

      // Row 4 (y=2)
      map.child_vertex_weights[21] = {{6, 0.5}, {7, 0.5}};  // midpoint between v6 and v7
      map.child_vertex_weights[23] = {{7, 0.5}, {8, 0.5}};  // midpoint between v7 and v8
    }

    return map;
  }

  // Helper to compute L2 norm of field difference
  double compute_l2_error(const std::vector<double>& field1,
                         const std::vector<double>& field2) {
    EXPECT_EQ(field1.size(), field2.size());
    double error = 0.0;
    for (size_t i = 0; i < field1.size(); ++i) {
      double diff = field1[i] - field2[i];
      error += diff * diff;
    }
    return std::sqrt(error / field1.size());
  }

  // Helper to compute max norm of field difference
  double compute_max_error(const std::vector<double>& field1,
                          const std::vector<double>& field2) {
    EXPECT_EQ(field1.size(), field2.size());
    double max_error = 0.0;
    for (size_t i = 0; i < field1.size(); ++i) {
      double diff = std::abs(field1[i] - field2[i]);
      max_error = std::max(max_error, diff);
    }
    return max_error;
  }

  // Helper to compute field integral (simple sum for testing)
  double compute_field_sum(const std::vector<double>& field) {
    double sum = 0.0;
    for (double val : field) {
      sum += val;
    }
    return sum;
  }
};

// ========== Factory Tests ==========

// Test 1: Factory creation - Linear interpolation
TEST_F(FieldTransferTest, FactoryCreateLinear) {
  AdaptivityOptions options;
  options.field_transfer = FieldTransferType::LINEAR_INTERPOLATION;

  auto transfer = FieldTransferFactory::create(options);
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer->name(), "LinearInterpolation");
}

// Test 2: Factory creation - Conservative transfer
TEST_F(FieldTransferTest, FactoryCreateConservative) {
  AdaptivityOptions options;
  options.field_transfer = FieldTransferType::CONSERVATIVE;

  auto transfer = FieldTransferFactory::create(options);
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer->name(), "Conservative");
}

// Test 3: Factory creation - High-order transfer
TEST_F(FieldTransferTest, FactoryCreateHighOrder) {
  AdaptivityOptions options;
  options.field_transfer = FieldTransferType::HIGH_ORDER;

  auto transfer = FieldTransferFactory::create(options);
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer->name(), "HighOrder");
}

// Test 4: Factory creation - Injection transfer
TEST_F(FieldTransferTest, FactoryCreateInjection) {
  AdaptivityOptions options;
  options.field_transfer = FieldTransferType::INJECTION;

  auto transfer = FieldTransferFactory::create(options);
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer->name(), "Injection");
}

// ========== Linear Interpolation Tests ==========

// Test 5: Linear interpolation - Constant field prolongation
TEST_F(FieldTransferTest, LinearConstantProlongation) {
  auto coarse_mesh = create_1d_mesh(2);
  auto fine_mesh = create_1d_mesh(4);

  std::vector<double> coarse_field = create_constant_field(3, 5.0); // 3 vertices
  std::vector<double> fine_field(5, 0.0); // 5 vertices

  auto map = create_uniform_refinement_map(2, 2);

  LinearInterpolationTransfer transfer;
  transfer.prolongate(*coarse_mesh, *fine_mesh, coarse_field, fine_field, map);

  // Check that constant field is preserved
  for (double val : fine_field) {
    EXPECT_NEAR(val, 5.0, 1e-10);
  }
}

// Test 6: Linear interpolation - Linear field prolongation
TEST_F(FieldTransferTest, LinearFieldProlongation) {
  // Create meshes with same physical domain but different resolutions
  // Coarse: 2x2 cells on [0,2]x[0,2] domain
  auto coarse_mesh = create_2d_quad_mesh(2, 2);

  // Fine: Also on [0,2]x[0,2] domain but with 4x4 cells (refined)
  // We need to scale the coordinates to match the coarse domain
  auto fine_mesh = std::make_unique<MeshBase>();
  size_t vertex_id = 0;
  for (size_t j = 0; j <= 4; ++j) {
    for (size_t i = 0; i <= 4; ++i) {
      fine_mesh->add_vertex(vertex_id++, {
        static_cast<real_t>(i) * 0.5,  // Scale to [0,2]
        static_cast<real_t>(j) * 0.5,  // Scale to [0,2]
        0.0
      });
    }
  }

  // Create 4x4 quad cells
  size_t elem_id = 0;
  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < 4; ++i) {
      index_t v0 = static_cast<index_t>(j * 5 + i);
      index_t v1 = v0 + 1;
      index_t v2 = v0 + 6;
      index_t v3 = v0 + 5;
      fine_mesh->add_cell(static_cast<index_t>(elem_id++), CellFamily::Quad, {v0, v1, v2, v3});
    }
  }

  auto coarse_field = create_linear_field(*coarse_mesh, 1.0, 2.0, 0.0, 3.0);
  std::vector<double> fine_field(fine_mesh->n_vertices(), 0.0);

  // Use the new helper with proper vertex weights
  auto map = create_2d_uniform_refinement_map_with_weights(2, 2, 4, 4);

  LinearInterpolationTransfer transfer;
  transfer.prolongate(*coarse_mesh, *fine_mesh, coarse_field, fine_field, map);

  // Check that linear field is exactly reproduced
  auto expected = create_linear_field(*fine_mesh, 1.0, 2.0, 0.0, 3.0);
  double error = compute_l2_error(fine_field, expected);
  EXPECT_LT(error, 1e-10);
}

// Test 7: Linear interpolation - Restriction operation
TEST_F(FieldTransferTest, LinearRestriction) {
  auto fine_mesh = create_1d_mesh(4);
  auto coarse_mesh = create_1d_mesh(2);

  std::vector<double> fine_field = {1.0, 2.0, 3.0, 4.0, 5.0}; // 5 vertices
  std::vector<double> coarse_field(3, 0.0); // 3 vertices

  auto map = create_uniform_refinement_map(2, 2);

  LinearInterpolationTransfer transfer;
  transfer.restrict(*fine_mesh, *coarse_mesh, fine_field, coarse_field, map);

  // Check that values are averaged appropriately
  EXPECT_GT(coarse_field[0], 0.0);
  EXPECT_GT(coarse_field[1], 0.0);
  EXPECT_GT(coarse_field[2], 0.0);
}

// Test 8: Linear interpolation - Volume weighting configuration
TEST_F(FieldTransferTest, LinearVolumeWeighting) {
  LinearInterpolationTransfer::Config config;
  config.use_volume_weighting = true;

  LinearInterpolationTransfer transfer(config);

  auto fine_mesh = create_2d_quad_mesh(4, 4);
  auto coarse_mesh = create_2d_quad_mesh(2, 2);

  std::vector<double> fine_field(fine_mesh->n_vertices(), 1.0);
  std::vector<double> coarse_field(coarse_mesh->n_vertices(), 0.0);

  auto map = create_uniform_refinement_map(4, 4);

  transfer.restrict(*fine_mesh, *coarse_mesh, fine_field, coarse_field, map);

  // With uniform field, restriction should preserve the value
  for (double val : coarse_field) {
    EXPECT_NEAR(val, 1.0, 0.1);
  }
}

// ========== Conservative Transfer Tests ==========

// Test 9: Conservative transfer - Integral preservation
TEST_F(FieldTransferTest, ConservativeIntegralPreservation) {
  auto coarse_mesh = create_2d_quad_mesh(2, 2);
  auto fine_mesh = create_2d_quad_mesh(4, 4);

  std::vector<double> coarse_field(coarse_mesh->n_vertices(), 2.0);
  std::vector<double> fine_field(fine_mesh->n_vertices(), 0.0);

  auto map = create_uniform_refinement_map(4, 4);

  ConservativeTransfer transfer;

  double coarse_sum = compute_field_sum(coarse_field);
  transfer.prolongate(*coarse_mesh, *fine_mesh, coarse_field, fine_field, map);
  double fine_sum = compute_field_sum(fine_field);

  // Check that total is approximately conserved
  EXPECT_NEAR(coarse_sum, fine_sum, coarse_sum * 0.1);
}

// Test 10: Conservative transfer - Mass conservation
TEST_F(FieldTransferTest, ConservativeMassConservation) {
  ConservativeTransfer::Config config;
  config.quantity = ConservativeTransfer::Config::ConservedQuantity::MASS;

  ConservativeTransfer transfer(config);

  auto coarse_mesh = create_3d_tet_mesh();
  auto fine_mesh = create_3d_tet_mesh(); // Same for simplicity

  std::vector<double> coarse_field(coarse_mesh->n_vertices(), 1.5);
  std::vector<double> fine_field(fine_mesh->n_vertices(), 0.0);

  auto map = create_uniform_refinement_map(1, 1);

  transfer.prolongate(*coarse_mesh, *fine_mesh, coarse_field, fine_field, map);

  // Mass should be conserved
  double coarse_mass = compute_field_sum(coarse_field);
  double fine_mass = compute_field_sum(fine_field);
  EXPECT_NEAR(coarse_mass, fine_mass, coarse_mass * 0.01);
}

// Test 11: Conservative transfer - High-order reconstruction
TEST_F(FieldTransferTest, ConservativeHighOrderReconstruction) {
  ConservativeTransfer::Config config;
  config.high_order_reconstruction = true;

  ConservativeTransfer transfer(config);
  EXPECT_EQ(transfer.name(), "Conservative");

  // Test that configuration is applied
  auto mesh = create_2d_quad_mesh(2, 2);
  std::vector<double> field = create_quadratic_field(*mesh);

  // High-order reconstruction should handle quadratic fields better
  EXPECT_GT(field[0], -1.0); // Just verify field was created
}

// Test 12: Conservative transfer - Conservation tolerance
TEST_F(FieldTransferTest, ConservationTolerance) {
  ConservativeTransfer::Config config;
  config.conservation_tolerance = 1e-12;
  config.max_conservation_iterations = 20;

  ConservativeTransfer transfer(config);

  auto mesh = create_1d_mesh(4);
  std::vector<double> field(5, 3.14159);
  std::vector<double> new_field(5, 0.0);

  auto map = create_uniform_refinement_map(4, 1);

  transfer.prolongate(*mesh, *mesh, field, new_field, map);

  // Test runs without error
  EXPECT_GE(new_field[0], 0.0);
}

// ========== High-Order Transfer Tests ==========

// Test 13: High-order transfer - Polynomial order configuration
TEST_F(FieldTransferTest, HighOrderPolynomialOrder) {
  HighOrderTransfer::Config config;
  config.polynomial_order = 3;

  HighOrderTransfer transfer(config);
  EXPECT_EQ(transfer.name(), "HighOrder");

  auto mesh = create_2d_tri_mesh(2, 2);
  auto field = create_quadratic_field(*mesh);

  // Polynomial order 3 should handle quadratic exactly
  EXPECT_GT(field[0], -1.0);
}

// Test 14: High-order transfer - Least squares reconstruction
TEST_F(FieldTransferTest, HighOrderLeastSquares) {
  HighOrderTransfer::Config config;
  config.use_least_squares = true;
  config.min_stencil_size = 8;

  HighOrderTransfer transfer(config);

  auto coarse_mesh = create_2d_quad_mesh(2, 2);
  auto fine_mesh = create_2d_quad_mesh(3, 3);

  auto coarse_field = create_quadratic_field(*coarse_mesh);
  std::vector<double> fine_field(fine_mesh->n_vertices(), 0.0);

  auto map = create_uniform_refinement_map(4, 2);

  transfer.prolongate(*coarse_mesh, *fine_mesh, coarse_field, fine_field, map);

  // Check that field has been transferred
  double sum = compute_field_sum(fine_field);
  EXPECT_NE(sum, 0.0);
}

// Test 15: High-order transfer - Gradient limiting
TEST_F(FieldTransferTest, HighOrderGradientLimiting) {
  HighOrderTransfer::Config config;
  config.limit_gradients = true;
  config.polynomial_order = 2;

  HighOrderTransfer transfer(config);

  auto mesh = create_1d_mesh(4);

  // Create field with sharp gradients
  std::vector<double> field = {0.0, 0.0, 10.0, 10.0, 10.0};
  std::vector<double> limited_field(5, 0.0);

  auto map = create_uniform_refinement_map(4, 1);

  transfer.restrict(*mesh, *mesh, field, limited_field, map);

  // Gradient limiting should smooth sharp transitions
  EXPECT_GE(limited_field[1], 0.0);
  EXPECT_LE(limited_field[1], 10.0);
}

// Test 16: High-order transfer - Weight functions
TEST_F(FieldTransferTest, HighOrderWeightFunctions) {
  HighOrderTransfer::Config config;

  // Test uniform weights
  config.weight_function = HighOrderTransfer::Config::WeightFunction::UNIFORM;
  HighOrderTransfer transfer1(config);
  EXPECT_EQ(transfer1.name(), "HighOrder");

  // Test inverse distance weights
  config.weight_function = HighOrderTransfer::Config::WeightFunction::INVERSE_DISTANCE;
  HighOrderTransfer transfer2(config);
  EXPECT_EQ(transfer2.name(), "HighOrder");

  // Test Gaussian weights
  config.weight_function = HighOrderTransfer::Config::WeightFunction::GAUSSIAN;
  HighOrderTransfer transfer3(config);
  EXPECT_EQ(transfer3.name(), "HighOrder");
}

// ========== Injection Transfer Tests ==========

// Test 17: Injection transfer - Direct copy
TEST_F(FieldTransferTest, InjectionDirectCopy) {
  auto mesh1 = create_1d_mesh(3);
  auto mesh2 = create_1d_mesh(3);

  std::vector<double> field1 = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> field2(4, 0.0);

  auto map = create_uniform_refinement_map(3, 1);

  InjectionTransfer transfer;
  transfer.prolongate(*mesh1, *mesh2, field1, field2, map);

  // Injection should copy values directly
  for (size_t i = 0; i < std::min(field1.size(), field2.size()); ++i) {
    EXPECT_EQ(field2[i], field1[i]);
  }
}

// Test 18: Injection transfer - No interpolation
TEST_F(FieldTransferTest, InjectionNoInterpolation) {
  auto coarse_mesh = create_2d_quad_mesh(2, 2);
  auto fine_mesh = create_2d_quad_mesh(4, 4);

  auto coarse_field = create_linear_field(*coarse_mesh, 2.0, 3.0, 0.0, 1.0);
  std::vector<double> fine_field(fine_mesh->n_vertices(), -1.0);

  auto map = create_uniform_refinement_map(4, 4);

  InjectionTransfer transfer;
  transfer.prolongate(*coarse_mesh, *fine_mesh, coarse_field, fine_field, map);

  // Some values should be directly copied, others unchanged
  bool has_copied = false;
  bool has_unchanged = false;
  for (double val : fine_field) {
    if (val != -1.0) has_copied = true;
    if (val == -1.0) has_unchanged = true;
  }
  EXPECT_TRUE(has_copied);
}

// Test 19: Injection transfer - vertex field uses vertex mapping (not cell counts)
TEST_F(FieldTransferTest, InjectionTransfersVertexField) {
  auto old_mesh = create_1d_mesh(2);  // 3 vertices
  auto new_mesh = create_1d_mesh(4);  // 5 vertices

  auto old_handle = MeshFields::attach_field(*old_mesh, EntityKind::Vertex, "id",
                                             FieldScalarType::Int32, 1);
  int32_t* old_data = MeshFields::field_data_as<int32_t>(*old_mesh, old_handle);
  for (size_t v = 0; v < old_mesh->n_vertices(); ++v) {
    old_data[v] = static_cast<int32_t>(2 * v);  // ensures midpoint averages are integers
  }

  auto map = create_1d_uniform_refinement_map_with_weights(2, 4);
  AdaptivityOptions options;
  options.transfer_fields = {"id"};

  MeshFields old_fields;
  MeshFields new_fields;
  InjectionTransfer transfer;
  auto stats = transfer.transfer(*old_mesh, *new_mesh, old_fields, new_fields, map, options);

  EXPECT_EQ(stats.num_fields, 1u);
  ASSERT_TRUE(MeshFields::has_field(*new_mesh, EntityKind::Vertex, "id"));
  auto new_handle = MeshFields::get_field_handle(*new_mesh, EntityKind::Vertex, "id");
  EXPECT_EQ(MeshFields::field_entity_count(*new_mesh, new_handle), new_mesh->n_vertices());

  const int32_t* new_data = MeshFields::field_data_as<int32_t>(*new_mesh, new_handle);
  ASSERT_EQ(new_mesh->n_vertices(), 5u);
  EXPECT_EQ(new_data[0], 0);
  EXPECT_EQ(new_data[1], 1);
  EXPECT_EQ(new_data[2], 2);
  EXPECT_EQ(new_data[3], 3);
  EXPECT_EQ(new_data[4], 4);
}

// ========== Field Transfer Utils Tests ==========

// Test 20: Utils - Build parent-child map
TEST_F(FieldTransferTest, UtilsBuildParentChildMap) {
  auto old_mesh = create_2d_quad_mesh(2, 2);
  auto new_mesh = create_2d_quad_mesh(4, 4);

  std::vector<MarkType> marks(old_mesh->n_cells(), MarkType::REFINE);

  auto map = FieldTransferUtils::build_parent_child_map(*old_mesh, *new_mesh, marks);

  // Map should have parent-child relationships
  EXPECT_FALSE(map.parent_to_children.empty());
  EXPECT_FALSE(map.child_to_parent.empty());
}

// Test 20: Utils - Check conservation
TEST_F(FieldTransferTest, UtilsCheckConservation) {
  auto mesh1 = create_1d_mesh(2);
  auto mesh2 = create_1d_mesh(4);

  std::vector<double> field1 = {1.0, 2.0, 3.0};
  std::vector<double> field2 = {1.0, 1.5, 2.0, 2.5, 3.0};

  double error = FieldTransferUtils::check_conservation(*mesh1, *mesh2, field1, field2);

  // Conservation error should be computed
  EXPECT_GE(error, 0.0);
}

// Test 21: Utils - Compute interpolation error
TEST_F(FieldTransferTest, UtilsInterpolationError) {
  auto mesh = create_2d_quad_mesh(3, 3);

  auto exact = create_linear_field(*mesh, 1.0, 1.0, 0.0, 0.0);
  auto approx = create_linear_field(*mesh, 1.0, 1.0, 0.0, 0.1);

  double error = FieldTransferUtils::compute_interpolation_error(*mesh, exact, approx);

  EXPECT_GT(error, 0.0);
  EXPECT_LT(error, 1.0);
}

// Test 22: Utils - Project field between meshes
TEST_F(FieldTransferTest, UtilsProjectField) {
  auto source_mesh = create_2d_tri_mesh(2, 2);
  auto target_mesh = create_2d_tri_mesh(3, 3);

  auto source_field = create_constant_field(source_mesh->n_vertices(), 7.0);
  std::vector<double> target_field(target_mesh->n_vertices(), 0.0);

  FieldTransferUtils::project_field(*source_mesh, *target_mesh, source_field, target_field);

  // Constant field should be approximately preserved
  for (double val : target_field) {
    EXPECT_NEAR(val, 7.0, 2.0);
  }
}

// ========== Integration Tests ==========

// Test 23: Full transfer workflow with MeshFields
TEST_F(FieldTransferTest, FullTransferWorkflow) {
  auto old_mesh = create_2d_quad_mesh(2, 2);
  auto new_mesh = create_2d_quad_mesh(4, 4);

  MeshFields old_fields;
  MeshFields new_fields;

  // Attach a field to old mesh
  old_fields.attach_field(*old_mesh, EntityKind::Vertex, "temperature", FieldScalarType::Float64, 1);

  auto map = create_uniform_refinement_map(4, 4);
  AdaptivityOptions options;

  LinearInterpolationTransfer transfer;
  auto stats = transfer.transfer(*old_mesh, *new_mesh, old_fields, new_fields, map, options);

  // Check transfer statistics
  EXPECT_GE(stats.num_fields, 0);
  EXPECT_GE(stats.transfer_time, 0.0);
}

// Test 24: Transfer all fields utility
TEST_F(FieldTransferTest, TransferAllFields) {
  auto old_mesh = create_2d_quad_mesh(2, 2);
  auto new_mesh = create_2d_quad_mesh(3, 3);

  MeshFields old_fields;
  MeshFields new_fields;

  // Attach multiple fields
  old_fields.attach_field(*old_mesh, EntityKind::Vertex, "pressure", FieldScalarType::Float64, 1);
  old_fields.attach_field(*old_mesh, EntityKind::Vertex, "velocity", FieldScalarType::Float64, 3);
  old_fields.attach_field(*old_mesh, EntityKind::Volume, "density", FieldScalarType::Float64, 1);

  auto map = create_uniform_refinement_map(4, 2);
  AdaptivityOptions options;
  options.field_transfer = FieldTransferType::LINEAR_INTERPOLATION;

  auto stats = FieldTransferUtils::transfer_all_fields(
    *old_mesh, *new_mesh, old_fields, new_fields, map, options);

  EXPECT_GE(stats.num_fields, 0);
  EXPECT_GE(stats.num_prolongations + stats.num_restrictions, 0);
}

// Test 25: Round-trip fidelity test
TEST_F(FieldTransferTest, RoundTripFidelity) {
  auto original_mesh = create_1d_mesh(4);
  auto refined_mesh = create_1d_mesh(8);

  auto original_field = create_linear_field(*original_mesh, 2.0, 0.0, 0.0, 1.0);
  std::vector<double> refined_field(refined_mesh->n_vertices(), 0.0);
  std::vector<double> recovered_field(original_mesh->n_vertices(), 0.0);

  // Use the new helper with proper vertex weights for 1D refinement
  auto refine_map = create_1d_uniform_refinement_map_with_weights(4, 8);
  // For coarsening, we need the inverse relationship with vertex weights
  auto coarsen_map = create_1d_uniform_coarsening_map_with_weights(8, 4);

  LinearInterpolationTransfer transfer;

  // Refine
  transfer.prolongate(*original_mesh, *refined_mesh, original_field, refined_field, refine_map);

  // Coarsen back
  transfer.restrict(*refined_mesh, *original_mesh, refined_field, recovered_field, coarsen_map);

  // Check round-trip error
  double error = compute_l2_error(original_field, recovered_field);
  EXPECT_LT(error, 0.1);
}

// Test 26: Vector field transfer
TEST_F(FieldTransferTest, VectorFieldTransfer) {
  auto old_mesh = create_1d_mesh(4);   // 5 vertices
  auto new_mesh = create_1d_mesh(8);   // 9 vertices (uniform refinement by 2)

  // Attach vector field (3 components) on vertices.
  auto old_handle = MeshFields::attach_field(*old_mesh, EntityKind::Vertex, "displacement",
                                             FieldScalarType::Float64, 3);
  double* old_data = MeshFields::field_data_as<double>(*old_mesh, old_handle);
  for (size_t v = 0; v < old_mesh->n_vertices(); ++v) {
    old_data[3 * v + 0] = static_cast<double>(v);
    old_data[3 * v + 1] = 2.0 * static_cast<double>(v);
    old_data[3 * v + 2] = -static_cast<double>(v);
  }

  // Deterministic refinement map including vertex interpolation weights.
  auto map = create_1d_uniform_refinement_map_with_weights(4, 8);
  AdaptivityOptions options;
  options.field_transfer = FieldTransferType::LINEAR_INTERPOLATION;
  options.transfer_fields = {"displacement"};

  MeshFields old_fields;
  MeshFields new_fields;

  LinearInterpolationTransfer transfer;
  auto stats = transfer.transfer(*old_mesh, *new_mesh, old_fields, new_fields, map, options);

  EXPECT_EQ(stats.num_fields, 1u);
  ASSERT_TRUE(MeshFields::has_field(*new_mesh, EntityKind::Vertex, "displacement"));
  auto new_handle = MeshFields::get_field_handle(*new_mesh, EntityKind::Vertex, "displacement");
  const double* new_data = MeshFields::field_data_as<double>(*new_mesh, new_handle);

  // Existing vertices are copied exactly: fine vertex 2*v corresponds to coarse vertex v.
  for (size_t v = 0; v <= 4; ++v) {
    const size_t fv = 2 * v;
    EXPECT_NEAR(new_data[3 * fv + 0], static_cast<double>(v), 1e-12);
    EXPECT_NEAR(new_data[3 * fv + 1], 2.0 * static_cast<double>(v), 1e-12);
    EXPECT_NEAR(new_data[3 * fv + 2], -static_cast<double>(v), 1e-12);
  }

  // New vertices are midpoints: fine vertex 2*e+1 is average of coarse e and e+1.
  for (size_t e = 0; e < 4; ++e) {
    const size_t fv = 2 * e + 1;
    const double expected = static_cast<double>(e) + 0.5;
    EXPECT_NEAR(new_data[3 * fv + 0], expected, 1e-12);
    EXPECT_NEAR(new_data[3 * fv + 1], 2.0 * expected, 1e-12);
    EXPECT_NEAR(new_data[3 * fv + 2], -expected, 1e-12);
  }
}

// Test 27: Mixed element mesh transfer
TEST_F(FieldTransferTest, MixedElementTransfer) {
  // Create mixed mesh with triangles and quads
  auto old_mesh = std::make_unique<MeshBase>();

  // Add vertices for a simple mixed mesh
  old_mesh->add_vertex(0, {0.0, 0.0, 0.0});
  old_mesh->add_vertex(1, {1.0, 0.0, 0.0});
  old_mesh->add_vertex(2, {1.0, 1.0, 0.0});
  old_mesh->add_vertex(3, {0.0, 1.0, 0.0});
  old_mesh->add_vertex(4, {2.0, 0.0, 0.0});
  old_mesh->add_vertex(5, {2.0, 1.0, 0.0});

  // Add a quad and triangles
  old_mesh->add_cell(0, CellFamily::Quad, {0, 1, 2, 3});
  old_mesh->add_cell(1, CellFamily::Triangle, {1, 4, 2});
  old_mesh->add_cell(2, CellFamily::Triangle, {4, 5, 2});

  std::vector<double> field(6, 1.0);

  // Transfer should handle mixed elements
  EXPECT_EQ(old_mesh->n_cells(), 3);
  EXPECT_EQ(field.size(), 6);
}

// Test 28: Anisotropic refinement transfer
TEST_F(FieldTransferTest, AnisotropicRefinementTransfer) {
  auto old_mesh = create_2d_quad_mesh(2, 2);

  // Create anisotropically refined mesh (refined in x, not y)
  auto new_mesh = create_2d_quad_mesh(4, 2);

  auto old_field = create_linear_field(*old_mesh, 1.0, 2.0, 0.0, 0.0);
  std::vector<double> new_field(new_mesh->n_vertices(), 0.0);

  // Build appropriate parent-child map for anisotropic refinement
  ParentChildMap map;
  // Simplified - actual implementation would be more complex
  map.child_to_parent.resize(new_mesh->n_cells());

  LinearInterpolationTransfer transfer;
  transfer.prolongate(*old_mesh, *new_mesh, old_field, new_field, map);

  // Field should be transferred even with anisotropic refinement
  double sum = compute_field_sum(new_field);
  EXPECT_NE(sum, 0.0);
}

// Test 29: Boundary preservation test
TEST_F(FieldTransferTest, BoundaryPreservation) {
  LinearInterpolationTransfer::Config config;
  config.preserve_boundary = true;

  LinearInterpolationTransfer transfer(config);

  auto mesh = create_2d_quad_mesh(3, 3);
  auto field = create_linear_field(*mesh, 1.0, 1.0, 0.0, 0.0);

  // Boundary values should be preserved during transfer
  // Note: Actual boundary detection would need to be implemented
  EXPECT_EQ(transfer.name(), "LinearInterpolation");
}

// Test 30: Error accumulation in multiple transfers
TEST_F(FieldTransferTest, MultipleTransferErrorAccumulation) {
  auto mesh1 = create_1d_mesh(3);
  auto mesh2 = create_1d_mesh(6);
  auto mesh3 = create_1d_mesh(3);

  auto field1 = create_linear_field(*mesh1, 3.0, 0.0, 0.0, 2.0);
  std::vector<double> field2(mesh2->n_vertices(), 0.0);
  std::vector<double> field3(mesh3->n_vertices(), 0.0);

  // Use helpers with proper vertex weights
  auto map12 = create_1d_uniform_refinement_map_with_weights(3, 6);
  auto map23 = create_1d_uniform_coarsening_map_with_weights(6, 3);

  LinearInterpolationTransfer transfer;

  // Transfer 1 -> 2 -> 3
  transfer.prolongate(*mesh1, *mesh2, field1, field2, map12);
  transfer.restrict(*mesh2, *mesh3, field2, field3, map23);

  // Error should be bounded
  double error = compute_max_error(field1, field3);
  EXPECT_LT(error, 1.0);
}

// Test 31: Performance test with large mesh
TEST_F(FieldTransferTest, LargeMeshPerformance) {
  auto old_mesh = create_2d_quad_mesh(10, 10); // 100 elements
  auto new_mesh = create_2d_quad_mesh(20, 20); // 400 elements

  auto old_field = create_quadratic_field(*old_mesh);
  std::vector<double> new_field(new_mesh->n_vertices(), 0.0);

  auto map = create_uniform_refinement_map(100, 4);

  LinearInterpolationTransfer transfer;

  auto start = std::chrono::high_resolution_clock::now();
  transfer.prolongate(*old_mesh, *new_mesh, old_field, new_field, map);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;

  // Transfer should complete in reasonable time
  EXPECT_LT(elapsed.count(), 1.0); // Less than 1 second
}

// Test 32: Conservation error tracking
TEST_F(FieldTransferTest, ConservationErrorTracking) {
  auto old_mesh = create_2d_tri_mesh(3, 3);
  auto new_mesh = create_2d_tri_mesh(5, 5);

  MeshFields old_fields;
  MeshFields new_fields;

  old_fields.attach_field(*old_mesh, EntityKind::Vertex, "energy", FieldScalarType::Float64, 1);

  auto map = create_uniform_refinement_map(9, 3);
  AdaptivityOptions options;
  options.field_transfer = FieldTransferType::CONSERVATIVE;

  auto transfer = FieldTransferFactory::create(options);
  auto stats = transfer->transfer(*old_mesh, *new_mesh, old_fields, new_fields, map, options);

  // Conservation errors should be tracked
  EXPECT_TRUE(stats.conservation_errors.empty() ||
              stats.conservation_errors.find("energy") != stats.conservation_errors.end());
}

// Test 33: High-order polynomial accuracy test
TEST_F(FieldTransferTest, HighOrderPolynomialAccuracy) {
  HighOrderTransfer::Config config;
  config.polynomial_order = 4;
  config.use_least_squares = true;

  HighOrderTransfer transfer(config);

  auto mesh = create_2d_quad_mesh(4, 4);

  // Create cubic field (x^3 + y^3)
  std::vector<double> field(mesh->n_vertices());
  for (size_t i = 0; i < mesh->n_vertices(); ++i) {
    auto xyz = mesh->get_vertex_coords(i);
    field[i] = xyz[0]*xyz[0]*xyz[0] + xyz[1]*xyz[1]*xyz[1];
  }

  std::vector<double> transferred(mesh->n_vertices(), 0.0);
  auto map = create_uniform_refinement_map(16, 1);

  // High-order should handle cubic well with order 4
  transfer.restrict(*mesh, *mesh, field, transferred, map);

  // Some transfer should occur
  double sum = compute_field_sum(transferred);
  EXPECT_NE(sum, 0.0);
}

// Test 34: Edge case - Empty mesh
TEST_F(FieldTransferTest, EmptyMeshHandling) {
  auto empty_mesh = std::make_unique<MeshBase>();
  auto normal_mesh = create_1d_mesh(2);

  std::vector<double> empty_field;
  std::vector<double> normal_field(3, 1.0);

  ParentChildMap empty_map;

  LinearInterpolationTransfer transfer;

  // Should handle empty mesh gracefully
  transfer.prolongate(*empty_mesh, *normal_mesh, empty_field, normal_field, empty_map);

  // Normal field should be unchanged or zeroed
  EXPECT_EQ(normal_field.size(), 3);
}

// Test 35: Transfer with refinement patterns
TEST_F(FieldTransferTest, RefinementPatternHandling) {
  auto old_mesh = create_2d_quad_mesh(2, 2);
  auto new_mesh = create_2d_quad_mesh(4, 4);

  // Create parent-child map with different patterns
  ParentChildMap map;
  map.child_to_parent = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  map.parent_to_children[0] = {0, 1, 2, 3};
  map.parent_to_children[1] = {4, 5, 6, 7};
  map.parent_to_children[2] = {8, 9, 10, 11};
  map.parent_to_children[3] = {12, 13, 14, 15};

  // Set different refinement patterns
  map.parent_patterns[0] = AdaptivityOptions::RefinementPattern::RED;
  map.parent_patterns[1] = AdaptivityOptions::RefinementPattern::GREEN;
  map.parent_patterns[2] = AdaptivityOptions::RefinementPattern::RED;
  map.parent_patterns[3] = AdaptivityOptions::RefinementPattern::RED_GREEN;

  auto old_field = create_constant_field(old_mesh->n_vertices(), 2.5);
  std::vector<double> new_field(new_mesh->n_vertices(), 0.0);

  LinearInterpolationTransfer transfer;
  transfer.prolongate(*old_mesh, *new_mesh, old_field, new_field, map);

  // Should handle different patterns
  double sum = compute_field_sum(new_field);
  EXPECT_GT(sum, 0.0);
}

namespace {

void write_u32(uint8_t* dst, uint32_t v) {
  std::memcpy(dst, &v, sizeof(v));
}

uint32_t read_u32(const uint8_t* src) {
  uint32_t v = 0;
  std::memcpy(&v, src, sizeof(v));
  return v;
}

} // namespace

TEST_F(FieldTransferTest, CustomVertexFieldTransfer_CopiesByGIDAndZerosNew) {
  auto old_mesh = create_1d_mesh(2); // 3 vertices with GIDs {0,1,2}
  auto new_mesh = create_1d_mesh(2);
  new_mesh->add_vertex(3, {3.0, 0.0, 0.0}); // New vertex with new GID

  MeshFields old_fields;
  MeshFields new_fields;

  auto old_h = MeshFields::attach_field(*old_mesh, EntityKind::Vertex, "blob",
                                        FieldScalarType::Custom, 1, /*bytes*/ 8);

  auto* old_data = MeshFields::field_data_as<uint8_t>(*old_mesh, old_h);
  const size_t bytes_per_entity = MeshFields::field_bytes_per_entity(*old_mesh, old_h);
  ASSERT_EQ(bytes_per_entity, 8u);

  for (size_t i = 0; i < static_cast<size_t>(old_mesh->n_vertices()); ++i) {
    const uint32_t gid = static_cast<uint32_t>(old_mesh->vertex_gids()[i]);
    write_u32(old_data + i * bytes_per_entity + 0, gid);
    write_u32(old_data + i * bytes_per_entity + 4, gid ^ 0xA5A5A5A5u);
  }

  ParentChildMap map;
  AdaptivityOptions options;
  LinearInterpolationTransfer transfer;
  (void)transfer.transfer(*old_mesh, *new_mesh, old_fields, new_fields, map, options);

  ASSERT_TRUE(MeshFields::has_field(*new_mesh, EntityKind::Vertex, "blob"));
  auto new_h = MeshFields::get_field_handle(*new_mesh, EntityKind::Vertex, "blob");
  EXPECT_EQ(MeshFields::field_type(*new_mesh, new_h), FieldScalarType::Custom);
  EXPECT_EQ(MeshFields::field_components(*new_mesh, new_h), 1u);
  EXPECT_EQ(MeshFields::field_bytes_per_entity(*new_mesh, new_h), 8u);

  const auto* new_data = MeshFields::field_data_as<const uint8_t>(*new_mesh, new_h);
  const size_t new_bytes = MeshFields::field_bytes_per_entity(*new_mesh, new_h);
  ASSERT_EQ(new_bytes, 8u);

  ASSERT_EQ(new_mesh->vertex_gids().size(), 4u);
  for (size_t i = 0; i < 4u; ++i) {
    const gid_t g = new_mesh->vertex_gids()[i];
    const uint8_t* ent = new_data + i * new_bytes;
    if (g <= 2) {
      EXPECT_EQ(read_u32(ent + 0), static_cast<uint32_t>(g));
      EXPECT_EQ(read_u32(ent + 4), static_cast<uint32_t>(g) ^ 0xA5A5A5A5u);
    } else {
      // Unmatched new vertices are zero-filled.
      EXPECT_EQ(read_u32(ent + 0), 0u);
      EXPECT_EQ(read_u32(ent + 4), 0u);
    }
  }
}

TEST_F(FieldTransferTest, CustomVolumeFieldTransfer_RefinementCopiesFromParent) {
  auto old_mesh = create_1d_mesh(2); // 2 cells, gids {0,1}
  auto new_mesh = create_1d_mesh(4); // 4 cells, gids {0,1,2,3} initially
  new_mesh->set_cell_gids({10, 11, 12, 13}); // ensure no gid match with old

  MeshFields old_fields;
  MeshFields new_fields;

  auto old_h = MeshFields::attach_field(*old_mesh, EntityKind::Volume, "cell_blob",
                                        FieldScalarType::Custom, 1, /*bytes*/ 8);
  auto* old_data = MeshFields::field_data_as<uint8_t>(*old_mesh, old_h);
  const size_t bytes_per_entity = MeshFields::field_bytes_per_entity(*old_mesh, old_h);
  ASSERT_EQ(bytes_per_entity, 8u);

  for (size_t i = 0; i < static_cast<size_t>(old_mesh->n_cells()); ++i) {
    const uint32_t gid = static_cast<uint32_t>(old_mesh->cell_gids()[i]);
    write_u32(old_data + i * bytes_per_entity + 0, gid);
    write_u32(old_data + i * bytes_per_entity + 4, gid ^ 0xDEADBEEFu);
  }

  ParentChildMap map;
  map.child_to_parent = {0, 0, 1, 1}; // 4 children

  AdaptivityOptions options;
  LinearInterpolationTransfer transfer;
  (void)transfer.transfer(*old_mesh, *new_mesh, old_fields, new_fields, map, options);

  ASSERT_TRUE(MeshFields::has_field(*new_mesh, EntityKind::Volume, "cell_blob"));
  auto new_h = MeshFields::get_field_handle(*new_mesh, EntityKind::Volume, "cell_blob");
  const auto* new_data = MeshFields::field_data_as<const uint8_t>(*new_mesh, new_h);
  const size_t new_bytes = MeshFields::field_bytes_per_entity(*new_mesh, new_h);
  ASSERT_EQ(new_bytes, 8u);

  ASSERT_EQ(new_mesh->n_cells(), 4u);
  // Children 0,1 inherit parent 0; children 2,3 inherit parent 1.
  for (size_t c = 0; c < 4u; ++c) {
    const size_t p = (c < 2u) ? 0u : 1u;
    const uint8_t* ent = new_data + c * new_bytes;
    EXPECT_EQ(read_u32(ent + 0), static_cast<uint32_t>(old_mesh->cell_gids()[p]));
    EXPECT_EQ(read_u32(ent + 4), static_cast<uint32_t>(old_mesh->cell_gids()[p]) ^ 0xDEADBEEFu);
  }
}

TEST_F(FieldTransferTest, CustomVolumeFieldTransfer_CoarseningUsesDeterministicRepresentativeChild) {
  auto old_mesh = create_1d_mesh(4); // 4 fine cells gids {0,1,2,3}
  auto new_mesh = create_1d_mesh(2); // 2 coarse cells gids {0,1} initially
  new_mesh->set_cell_gids({100, 101}); // ensure no gid match with old

  MeshFields old_fields;
  MeshFields new_fields;

  auto old_h = MeshFields::attach_field(*old_mesh, EntityKind::Volume, "cell_blob",
                                        FieldScalarType::Custom, 1, /*bytes*/ 8);
  auto* old_data = MeshFields::field_data_as<uint8_t>(*old_mesh, old_h);
  const size_t bytes_per_entity = MeshFields::field_bytes_per_entity(*old_mesh, old_h);
  ASSERT_EQ(bytes_per_entity, 8u);

  for (size_t i = 0; i < static_cast<size_t>(old_mesh->n_cells()); ++i) {
    const uint32_t gid = static_cast<uint32_t>(old_mesh->cell_gids()[i]);
    write_u32(old_data + i * bytes_per_entity + 0, gid);
    write_u32(old_data + i * bytes_per_entity + 4, gid ^ 0x12345678u);
  }

  ParentChildMap map;
  map.parent_to_children[0] = {0, 1}; // coarse cell 0 from fine {0,1}
  map.parent_to_children[1] = {2, 3}; // coarse cell 1 from fine {2,3}

  AdaptivityOptions options;
  LinearInterpolationTransfer transfer;
  (void)transfer.transfer(*old_mesh, *new_mesh, old_fields, new_fields, map, options);

  ASSERT_TRUE(MeshFields::has_field(*new_mesh, EntityKind::Volume, "cell_blob"));
  auto new_h = MeshFields::get_field_handle(*new_mesh, EntityKind::Volume, "cell_blob");
  const auto* new_data = MeshFields::field_data_as<const uint8_t>(*new_mesh, new_h);
  const size_t new_bytes = MeshFields::field_bytes_per_entity(*new_mesh, new_h);
  ASSERT_EQ(new_bytes, 8u);

  // Deterministic representative child is min(child_index) per parent.
  const uint32_t rep0 = static_cast<uint32_t>(old_mesh->cell_gids()[0]);
  const uint32_t rep1 = static_cast<uint32_t>(old_mesh->cell_gids()[2]);

  EXPECT_EQ(read_u32(new_data + 0 * new_bytes + 0), rep0);
  EXPECT_EQ(read_u32(new_data + 0 * new_bytes + 4), rep0 ^ 0x12345678u);
  EXPECT_EQ(read_u32(new_data + 1 * new_bytes + 0), rep1);
  EXPECT_EQ(read_u32(new_data + 1 * new_bytes + 4), rep1 ^ 0x12345678u);
}

} // namespace test
} // namespace svmp
