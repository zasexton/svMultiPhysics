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
#include "Search/BVHAccel.h"
#include "Core/MeshBase.h"
#include "Core/DistributedMesh.h"
#include "Geometry/MeshGeometry.h"
#include <random>
#include <chrono>

namespace svmp {
namespace test {

// Test fixture for BVHAccel tests
class BVHAccelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a simple test mesh
    create_test_mesh();
  }

  void create_test_mesh() {
    // Create a mesh with boundary faces for ray tracing tests
    mesh_ = std::make_unique<DistributedMesh>();

    // Create a cube mesh with explicit boundary
    std::vector<std::array<real_t,3>> vertices = {
      // Bottom face (z=0)
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
      // Top face (z=1)
      {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0},
      {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}
    };

    for (size_t i = 0; i < vertices.size(); ++i) {
      mesh_->add_vertex(i, vertices[i]);
    }

    // Add hexahedron
    mesh_->add_cell(0, CellShape::Hexahedron, {0, 1, 2, 3, 4, 5, 6, 7});

    // Add boundary faces (quads)
    mesh_->add_boundary_face(0, {0, 1, 2, 3});  // Bottom
    mesh_->add_boundary_face(1, {4, 7, 6, 5});  // Top
    mesh_->add_boundary_face(2, {0, 4, 5, 1});  // Front
    mesh_->add_boundary_face(3, {2, 6, 7, 3});  // Back
    mesh_->add_boundary_face(4, {0, 3, 7, 4});  // Left
    mesh_->add_boundary_face(5, {1, 5, 6, 2});  // Right

    mesh_->finalize();
  }

  void create_complex_mesh(int n_cells) {
    mesh_ = std::make_unique<DistributedMesh>();

    // Create a mesh with many cells for performance testing
    std::mt19937 gen(42);
    std::uniform_real_distribution<real_t> dist(0.0, 10.0);

    // Create grid of vertices
    int n_per_dim = std::cbrt(n_cells) + 1;
    int vid = 0;

    std::vector<std::vector<std::vector<int>>> vertex_grid(
      n_per_dim, std::vector<std::vector<int>>(
        n_per_dim, std::vector<int>(n_per_dim, -1)));

    for (int i = 0; i < n_per_dim; ++i) {
      for (int j = 0; j < n_per_dim; ++j) {
        for (int k = 0; k < n_per_dim; ++k) {
          real_t x = i * 10.0 / (n_per_dim - 1);
          real_t y = j * 10.0 / (n_per_dim - 1);
          real_t z = k * 10.0 / (n_per_dim - 1);

          // Add some randomness
          x += dist(gen) * 0.1;
          y += dist(gen) * 0.1;
          z += dist(gen) * 0.1;

          mesh_->add_vertex(vid, {x, y, z});
          vertex_grid[i][j][k] = vid++;
        }
      }
    }

    // Create hexahedral cells
    int cell_id = 0;
    for (int i = 0; i < n_per_dim - 1; ++i) {
      for (int j = 0; j < n_per_dim - 1; ++j) {
        for (int k = 0; k < n_per_dim - 1; ++k) {
          std::vector<index_t> hex = {
            vertex_grid[i][j][k],
            vertex_grid[i+1][j][k],
            vertex_grid[i+1][j+1][k],
            vertex_grid[i][j+1][k],
            vertex_grid[i][j][k+1],
            vertex_grid[i+1][j][k+1],
            vertex_grid[i+1][j+1][k+1],
            vertex_grid[i][j+1][k+1]
          };
          mesh_->add_cell(cell_id++, CellShape::Hexahedron, hex);

          // Add boundary faces for cells on the boundary
          if (i == 0) {
            mesh_->add_boundary_face(mesh_->n_boundary_faces(),
              {hex[0], hex[3], hex[7], hex[4]});
          }
          if (i == n_per_dim - 2) {
            mesh_->add_boundary_face(mesh_->n_boundary_faces(),
              {hex[1], hex[5], hex[6], hex[2]});
          }
          if (j == 0) {
            mesh_->add_boundary_face(mesh_->n_boundary_faces(),
              {hex[0], hex[4], hex[5], hex[1]});
          }
          if (j == n_per_dim - 2) {
            mesh_->add_boundary_face(mesh_->n_boundary_faces(),
              {hex[2], hex[6], hex[7], hex[3]});
          }
          if (k == 0) {
            mesh_->add_boundary_face(mesh_->n_boundary_faces(),
              {hex[0], hex[1], hex[2], hex[3]});
          }
          if (k == n_per_dim - 2) {
            mesh_->add_boundary_face(mesh_->n_boundary_faces(),
              {hex[4], hex[7], hex[6], hex[5]});
          }
        }
      }
    }

    mesh_->finalize();
  }

  std::unique_ptr<MeshBase> mesh_;
  std::unique_ptr<BVHAccel> bvh_;
};

// Test building BVH
TEST_F(BVHAccelTest, Build) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;

  ASSERT_NO_THROW(bvh_->build(*mesh_, IAccel::Configuration::Reference, config));
  EXPECT_TRUE(bvh_->is_built());
  EXPECT_EQ(bvh_->built_config(), IAccel::Configuration::Reference);

  auto stats = bvh_->get_stats();
  EXPECT_GT(stats.n_nodes, 0);
  EXPECT_GT(stats.tree_depth, 0);
  EXPECT_GT(stats.memory_bytes, 0);

  std::cout << "BVH stats for test mesh:\n";
  std::cout << "  Nodes: " << stats.n_nodes << "\n";
  std::cout << "  Depth: " << stats.tree_depth << "\n";
  std::cout << "  Memory: " << stats.memory_bytes << " bytes\n";
}

// Test clearing BVH
TEST_F(BVHAccelTest, Clear) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);
  EXPECT_TRUE(bvh_->is_built());

  bvh_->clear();
  EXPECT_FALSE(bvh_->is_built());

  auto stats = bvh_->get_stats();
  EXPECT_EQ(stats.n_nodes, 0);
}

// Test point location
TEST_F(BVHAccelTest, PointLocation) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::PointLocation;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test point inside mesh
  std::array<real_t,3> point1 = {0.5, 0.5, 0.5};
  auto result1 = bvh_->locate_point(*mesh_, point1);
  EXPECT_TRUE(result1.found);
  EXPECT_EQ(result1.cell_id, 0);

  // Test point outside mesh
  std::array<real_t,3> point2 = {2.0, 2.0, 2.0};
  auto result2 = bvh_->locate_point(*mesh_, point2);
  EXPECT_FALSE(result2.found);
  EXPECT_EQ(result2.cell_id, -1);

  // Test point on boundary
  std::array<real_t,3> point3 = {0.5, 0.5, 0.0};
  auto result3 = bvh_->locate_point(*mesh_, point3);
  EXPECT_TRUE(result3.found);
  EXPECT_EQ(result3.cell_id, 0);

  // Test with hint
  std::array<real_t,3> point4 = {0.6, 0.6, 0.6};
  auto result4 = bvh_->locate_point(*mesh_, point4, 0);
  EXPECT_TRUE(result4.found);
  EXPECT_EQ(result4.cell_id, 0);
}

// Test batch point location
TEST_F(BVHAccelTest, BatchPointLocation) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::vector<std::array<real_t,3>> points = {
    {0.5, 0.5, 0.5},
    {0.1, 0.1, 0.1},
    {0.9, 0.9, 0.9},
    {2.0, 2.0, 2.0},
    {-1.0, -1.0, -1.0}
  };

  auto results = bvh_->locate_points(*mesh_, points);
  EXPECT_EQ(results.size(), points.size());

  EXPECT_TRUE(results[0].found);
  EXPECT_TRUE(results[1].found);
  EXPECT_TRUE(results[2].found);
  EXPECT_FALSE(results[3].found);
  EXPECT_FALSE(results[4].found);
}

// Test nearest vertex
TEST_F(BVHAccelTest, NearestVertex) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test nearest to origin
  std::array<real_t,3> point1 = {-0.1, -0.1, -0.1};
  auto [vid1, dist1] = bvh_->nearest_vertex(*mesh_, point1);
  EXPECT_EQ(vid1, 0);  // Vertex at origin
  EXPECT_NEAR(dist1, std::sqrt(3 * 0.01), 1e-6);

  // Test nearest to center
  std::array<real_t,3> point2 = {0.5, 0.5, 0.5};
  auto [vid2, dist2] = bvh_->nearest_vertex(*mesh_, point2);
  EXPECT_GE(vid2, 0);
  EXPECT_LT(vid2, mesh_->n_vertices());
  EXPECT_GT(dist2, 0.0);
}

// Test k-nearest neighbors
TEST_F(BVHAccelTest, KNearestVertices) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  size_t k = 4;

  auto results = bvh_->k_nearest_vertices(*mesh_, point, k);
  EXPECT_EQ(results.size(), k);

  // Check ordering (should be sorted by distance)
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i-1].second, results[i].second);
  }

  // All vertices should be valid
  for (const auto& [vid, dist] : results) {
    EXPECT_GE(vid, 0);
    EXPECT_LT(vid, mesh_->n_vertices());
    EXPECT_GE(dist, 0.0);
  }
}

// Test vertices in radius
TEST_F(BVHAccelTest, VerticesInRadius) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  real_t radius = 1.0;

  auto results = bvh_->vertices_in_radius(*mesh_, point, radius);
  EXPECT_GT(results.size(), 0);
  EXPECT_LE(results.size(), mesh_->n_vertices());

  // Verify all results are within radius
  for (index_t vid : results) {
    auto vertex = mesh_->get_vertex_coords(vid);
    real_t dist = std::sqrt(
      std::pow(vertex[0] - point[0], 2) +
      std::pow(vertex[1] - point[1], 2) +
      std::pow(vertex[2] - point[2], 2)
    );
    EXPECT_LE(dist, radius);
  }
}

// Test ray intersection
TEST_F(BVHAccelTest, RayIntersection) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through mesh
  std::array<real_t,3> origin = {-0.5, 0.5, 0.5};
  std::array<real_t,3> direction = {1.0, 0.0, 0.0};

  auto result = bvh_->intersect_ray(*mesh_, origin, direction);
  EXPECT_TRUE(result.hit);
  EXPECT_NEAR(result.distance, 0.5, 1e-6);  // Should hit at x=0
  EXPECT_NEAR(result.hit_point[0], 0.0, 1e-6);
  EXPECT_NEAR(result.hit_point[1], 0.5, 1e-6);
  EXPECT_NEAR(result.hit_point[2], 0.5, 1e-6);

  // Ray missing mesh
  std::array<real_t,3> origin2 = {-0.5, -0.5, -0.5};
  std::array<real_t,3> direction2 = {0.0, -1.0, 0.0};

  auto result2 = bvh_->intersect_ray(*mesh_, origin2, direction2);
  EXPECT_FALSE(result2.hit);

  // Ray with max distance
  auto result3 = bvh_->intersect_ray(*mesh_, origin, direction, 0.25);
  EXPECT_FALSE(result3.hit);  // Max distance too short
}

// Test all ray intersections
TEST_F(BVHAccelTest, AllRayIntersections) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through entire mesh
  std::array<real_t,3> origin = {-0.5, 0.5, 0.5};
  std::array<real_t,3> direction = {1.0, 0.0, 0.0};

  auto results = bvh_->intersect_ray_all(*mesh_, origin, direction);
  EXPECT_GE(results.size(), 2);  // Should hit entry and exit

  // Check ordering
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i-1].distance, results[i].distance);
  }

  // First hit should be at x=0, last at x=1
  if (results.size() >= 2) {
    EXPECT_NEAR(results[0].distance, 0.5, 1e-6);
    EXPECT_NEAR(results.back().distance, 1.5, 1e-6);
  }
}

// Test cells in box
TEST_F(BVHAccelTest, CellsInBox) {
  create_complex_mesh(125);  // 5x5x5 grid
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Box containing part of mesh
  std::array<real_t,3> box_min = {2.0, 2.0, 2.0};
  std::array<real_t,3> box_max = {8.0, 8.0, 8.0};

  auto results = bvh_->cells_in_box(*mesh_, box_min, box_max);
  EXPECT_GT(results.size(), 0);
  EXPECT_LT(results.size(), mesh_->n_cells());

  // All cells should be valid
  for (index_t cell_id : results) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }
}

// Test cells in sphere
TEST_F(BVHAccelTest, CellsInSphere) {
  create_complex_mesh(125);  // 5x5x5 grid
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> center = {5.0, 5.0, 5.0};
  real_t radius = 3.0;

  auto results = bvh_->cells_in_sphere(*mesh_, center, radius);
  EXPECT_GT(results.size(), 0);
  EXPECT_LT(results.size(), mesh_->n_cells());

  // All cells should be valid
  for (index_t cell_id : results) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }
}

// Test BVH refitting
TEST_F(BVHAccelTest, Refit) {
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  auto initial_sah = bvh_->compute_sah_cost();
  std::cout << "Initial SAH cost: " << initial_sah << "\n";

  // Deform mesh
  for (index_t i = 0; i < mesh_->n_vertices(); ++i) {
    auto coords = mesh_->get_vertex_coords(i);
    coords[0] *= 1.1;  // Slight deformation
    coords[1] *= 0.9;
    mesh_->set_vertex_coords(i, coords);
  }

  // Refit BVH
  auto refit_start = std::chrono::steady_clock::now();
  bvh_->refit(*mesh_);
  auto refit_end = std::chrono::steady_clock::now();

  auto refit_time = std::chrono::duration<double, std::milli>(
    refit_end - refit_start).count();

  auto refit_sah = bvh_->compute_sah_cost();
  std::cout << "SAH cost after refit: " << refit_sah << "\n";
  std::cout << "Refit time: " << refit_time << " ms\n";

  // Test that refitted BVH still works
  std::array<real_t,3> point = {0.55, 0.45, 0.5};  // Adjusted for deformation
  auto result = bvh_->locate_point(*mesh_, point);
  EXPECT_TRUE(result.found);

  // Rebuild from scratch for comparison
  auto rebuild_start = std::chrono::steady_clock::now();
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);
  auto rebuild_end = std::chrono::steady_clock::now();

  auto rebuild_time = std::chrono::duration<double, std::milli>(
    rebuild_end - rebuild_start).count();

  auto rebuild_sah = bvh_->compute_sah_cost();
  std::cout << "SAH cost after rebuild: " << rebuild_sah << "\n";
  std::cout << "Rebuild time: " << rebuild_time << " ms\n";

  // Refit should be much faster than rebuild
  EXPECT_LT(refit_time, rebuild_time);

  // SAH costs should be similar (refit slightly worse)
  EXPECT_LE(refit_sah, rebuild_sah * 1.5);
}

// Test SAH cost computation
TEST_F(BVHAccelTest, SAHCost) {
  create_complex_mesh(27);  // 3x3x3 grid
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  auto sah_cost = bvh_->compute_sah_cost();
  EXPECT_GT(sah_cost, 0.0);

  std::cout << "SAH cost for 3x3x3 mesh: " << sah_cost << "\n";

  // Create unbalanced mesh and compare
  mesh_ = std::make_unique<DistributedMesh>();

  // Add vertices in a line (worst case for BVH)
  for (int i = 0; i < 10; ++i) {
    mesh_->add_vertex(i, {real_t(i), 0.0, 0.0});
  }

  // Add cells
  for (int i = 0; i < 9; ++i) {
    mesh_->add_cell(i, CellShape::Line, {i, i+1});
  }

  mesh_->finalize();

  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);
  auto line_sah = bvh_->compute_sah_cost();

  std::cout << "SAH cost for line mesh: " << line_sah << "\n";

  // Line mesh should have worse SAH cost per primitive
  EXPECT_GT(line_sah / 9, sah_cost / 27);
}

// Performance test with large mesh
TEST_F(BVHAccelTest, PerformanceLargeMesh) {
  create_complex_mesh(1000);  // 10x10x10 grid
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;

  auto build_start = std::chrono::steady_clock::now();
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);
  auto build_end = std::chrono::steady_clock::now();

  auto build_time = std::chrono::duration<double, std::milli>(
    build_end - build_start).count();

  std::cout << "BVH build time for 1000 cells: " << build_time << " ms\n";

  auto stats = bvh_->get_stats();
  std::cout << "Tree depth: " << stats.tree_depth << "\n";
  std::cout << "Number of nodes: " << stats.n_nodes << "\n";
  std::cout << "Memory usage: " << stats.memory_bytes / 1024.0 << " KB\n";
  std::cout << "SAH cost: " << bvh_->compute_sah_cost() << "\n";

  // Test ray tracing performance
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_t> dist(-1.0, 11.0);
  std::uniform_real_distribution<real_t> dir_dist(-1.0, 1.0);

  int hit_count = 0;
  auto query_start = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    std::array<real_t,3> origin = {dist(gen), dist(gen), dist(gen)};
    std::array<real_t,3> direction = {dir_dist(gen), dir_dist(gen), dir_dist(gen)};

    // Normalize direction
    real_t len = std::sqrt(direction[0]*direction[0] +
                          direction[1]*direction[1] +
                          direction[2]*direction[2]);
    if (len > 0) {
      for (int j = 0; j < 3; ++j) direction[j] /= len;

      auto result = bvh_->intersect_ray(*mesh_, origin, direction);
      if (result.hit) hit_count++;
    }
  }
  auto query_end = std::chrono::steady_clock::now();

  auto query_time = std::chrono::duration<double, std::milli>(
    query_end - query_start).count();

  std::cout << "Average ray intersection time: "
            << query_time / 1000.0 << " ms\n";
  std::cout << "Rays hit: " << hit_count << " / 1000\n";

  EXPECT_LT(build_time, 5000.0);  // Should build in under 5 seconds
  EXPECT_LT(query_time / 1000.0, 1.0);  // Queries should be under 1ms each
}

// Test deformed configuration
TEST_F(BVHAccelTest, DeformedConfiguration) {
  bvh_ = std::make_unique<BVHAccel>();

  // Add deformed coordinates
  for (index_t i = 0; i < mesh_->n_vertices(); ++i) {
    auto coords = mesh_->get_vertex_coords(i);
    // Apply twist deformation
    real_t angle = coords[2] * M_PI / 2;
    real_t x_new = coords[0] * cos(angle) - coords[1] * sin(angle);
    real_t y_new = coords[0] * sin(angle) + coords[1] * cos(angle);
    mesh_->set_vertex_deformed_coords(i, {x_new, y_new, coords[2]});
  }

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Deformed, config);

  EXPECT_TRUE(bvh_->is_built());
  EXPECT_EQ(bvh_->built_config(), IAccel::Configuration::Deformed);

  // Test point in deformed mesh
  std::array<real_t,3> point = {0.0, 0.5, 0.5};  // Center after twist
  auto result = bvh_->locate_point(*mesh_, point);
  EXPECT_TRUE(result.found);
}

// Test empty mesh
TEST_F(BVHAccelTest, EmptyMesh) {
  mesh_ = std::make_unique<DistributedMesh>();
  mesh_->finalize();

  bvh_ = std::make_unique<BVHAccel>();
  MeshSearch::SearchConfig config;

  ASSERT_NO_THROW(bvh_->build(*mesh_, IAccel::Configuration::Reference, config));

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  auto result = bvh_->locate_point(*mesh_, point);
  EXPECT_FALSE(result.found);

  auto [vid, dist] = bvh_->nearest_vertex(*mesh_, point);
  EXPECT_EQ(vid, -1);

  auto sah = bvh_->compute_sah_cost();
  EXPECT_EQ(sah, 0.0);
}

// Test tree quality metrics
TEST_F(BVHAccelTest, TreeQuality) {
  create_complex_mesh(125);  // 5x5x5 grid
  bvh_ = std::make_unique<BVHAccel>();

  MeshSearch::SearchConfig config;
  bvh_->build(*mesh_, IAccel::Configuration::Reference, config);

  auto stats = bvh_->get_stats();

  // For a binary tree with n leaves, internal nodes = n-1
  // Total nodes should be approximately 2n-1
  int expected_nodes = 2 * mesh_->n_cells() - 1;
  real_t node_ratio = static_cast<real_t>(stats.n_nodes) / expected_nodes;

  std::cout << "Tree quality metrics:\n";
  std::cout << "  Expected nodes: " << expected_nodes << "\n";
  std::cout << "  Actual nodes: " << stats.n_nodes << "\n";
  std::cout << "  Node ratio: " << node_ratio << "\n";

  // Good BVH should have close to theoretical node count
  EXPECT_GT(node_ratio, 0.8);
  EXPECT_LT(node_ratio, 1.2);

  // Tree depth should be logarithmic
  int expected_depth = std::ceil(std::log2(mesh_->n_cells()));
  std::cout << "  Expected depth: ~" << expected_depth << "\n";
  std::cout << "  Actual depth: " << stats.tree_depth << "\n";

  // Allow some deviation from perfect balance
  EXPECT_LT(stats.tree_depth, expected_depth * 2);
}

} // namespace test
} // namespace svmp