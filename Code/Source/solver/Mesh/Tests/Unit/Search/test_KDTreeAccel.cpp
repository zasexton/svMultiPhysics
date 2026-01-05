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
#include "Search/KDTreeAccel.h"
#include "Core/MeshBase.h"
#include "Geometry/MeshGeometry.h"
#include <random>
#include <chrono>

namespace svmp {
namespace test {

// Test fixture for KDTreeAccel tests
class KDTreeAccelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a simple test mesh
    create_test_mesh();
  }

  void create_test_mesh() {
    // Create a simple 2x2x2 hex mesh
    mesh_ = std::make_unique<MeshBase>();

    // Add vertices for a unit cube subdivided into 8 hexahedra
    std::vector<std::array<real_t,3>> vertices = {
      // Bottom layer (z=0)
      {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {0.0, 0.5, 0.0}, {0.5, 0.5, 0.0}, {1.0, 0.5, 0.0},
      {0.0, 1.0, 0.0}, {0.5, 1.0, 0.0}, {1.0, 1.0, 0.0},

      // Middle layer (z=0.5)
      {0.0, 0.0, 0.5}, {0.5, 0.0, 0.5}, {1.0, 0.0, 0.5},
      {0.0, 0.5, 0.5}, {0.5, 0.5, 0.5}, {1.0, 0.5, 0.5},
      {0.0, 1.0, 0.5}, {0.5, 1.0, 0.5}, {1.0, 1.0, 0.5},

      // Top layer (z=1.0)
      {0.0, 0.0, 1.0}, {0.5, 0.0, 1.0}, {1.0, 0.0, 1.0},
      {0.0, 0.5, 1.0}, {0.5, 0.5, 1.0}, {1.0, 0.5, 1.0},
      {0.0, 1.0, 1.0}, {0.5, 1.0, 1.0}, {1.0, 1.0, 1.0}
    };

    for (size_t i = 0; i < vertices.size(); ++i) {
      mesh_->add_vertex(i, vertices[i]);
    }

    // Add 8 hexahedra
    std::vector<std::vector<index_t>> hexes = {
      // Bottom layer
      {0, 1, 4, 3, 9, 10, 13, 12},
      {1, 2, 5, 4, 10, 11, 14, 13},
      {3, 4, 7, 6, 12, 13, 16, 15},
      {4, 5, 8, 7, 13, 14, 17, 16},

      // Top layer
      {9, 10, 13, 12, 18, 19, 22, 21},
      {10, 11, 14, 13, 19, 20, 23, 22},
      {12, 13, 16, 15, 21, 22, 25, 24},
      {13, 14, 17, 16, 22, 23, 26, 25}
    };

    for (size_t i = 0; i < hexes.size(); ++i) {
      mesh_->add_cell(i, CellShape::Hexahedron, hexes[i]);
    }

    mesh_->finalize();
  }

  void create_large_mesh(int n_points) {
    mesh_ = std::make_unique<MeshBase>();

    // Create random points
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<real_t> dist(0.0, 10.0);

    for (int i = 0; i < n_points; ++i) {
      std::array<real_t,3> vertex = {dist(gen), dist(gen), dist(gen)};
      mesh_->add_vertex(i, vertex);
    }

    // Create tetrahedra from nearby points
    for (int i = 0; i < n_points - 3; i += 4) {
      mesh_->add_cell(i/4, CellShape::Tetrahedron, {i, i+1, i+2, i+3});
    }

    mesh_->finalize();
  }

  std::unique_ptr<MeshBase> mesh_;
  std::unique_ptr<KDTreeAccel> kdtree_;
};

// Test building KD-tree
TEST_F(KDTreeAccelTest, Build) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;

  ASSERT_NO_THROW(kdtree_->build(*mesh_, IAccel::Configuration::Reference, config));
  EXPECT_TRUE(kdtree_->is_built());
  EXPECT_EQ(kdtree_->built_config(), IAccel::Configuration::Reference);

  auto stats = kdtree_->get_stats();
  EXPECT_GT(stats.n_nodes, 0);
  EXPECT_GT(stats.tree_depth, 0);
  EXPECT_GT(stats.memory_bytes, 0);
}

// Test clearing KD-tree
TEST_F(KDTreeAccelTest, Clear) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);
  EXPECT_TRUE(kdtree_->is_built());

  kdtree_->clear();
  EXPECT_FALSE(kdtree_->is_built());

  auto stats = kdtree_->get_stats();
  EXPECT_EQ(stats.n_nodes, 0);
}

// Test point location
TEST_F(KDTreeAccelTest, PointLocation) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test point inside mesh
  std::array<real_t,3> point1 = {0.25, 0.25, 0.25};
  auto result1 = kdtree_->locate_point(*mesh_, point1);
  EXPECT_TRUE(result1.found);
  EXPECT_GE(result1.cell_id, 0);
  EXPECT_LT(result1.cell_id, mesh_->n_cells());

  // Test point outside mesh
  std::array<real_t,3> point2 = {2.0, 2.0, 2.0};
  auto result2 = kdtree_->locate_point(*mesh_, point2);
  EXPECT_FALSE(result2.found);
  EXPECT_EQ(result2.cell_id, -1);

  // Test point on boundary
  std::array<real_t,3> point3 = {0.5, 0.5, 0.0};
  auto result3 = kdtree_->locate_point(*mesh_, point3);
  EXPECT_TRUE(result3.found);
  EXPECT_GE(result3.cell_id, 0);
}

// Test batch point location
TEST_F(KDTreeAccelTest, BatchPointLocation) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::vector<std::array<real_t,3>> points = {
    {0.25, 0.25, 0.25},
    {0.75, 0.75, 0.75},
    {2.0, 2.0, 2.0},
    {0.5, 0.5, 0.5}
  };

  auto results = kdtree_->locate_points(*mesh_, points);
  EXPECT_EQ(results.size(), points.size());

  EXPECT_TRUE(results[0].found);
  EXPECT_TRUE(results[1].found);
  EXPECT_FALSE(results[2].found);  // Outside mesh
  EXPECT_TRUE(results[3].found);
}

// Test nearest vertex
TEST_F(KDTreeAccelTest, NearestVertex) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test nearest to origin
  std::array<real_t,3> point1 = {0.1, 0.1, 0.1};
  auto [vid1, dist1] = kdtree_->nearest_vertex(*mesh_, point1);
  EXPECT_GE(vid1, 0);
  EXPECT_LT(vid1, mesh_->n_vertices());
  EXPECT_GT(dist1, 0.0);
  EXPECT_LT(dist1, 0.5);  // Should be close

  // Test nearest to center
  std::array<real_t,3> point2 = {0.5, 0.5, 0.5};
  auto [vid2, dist2] = kdtree_->nearest_vertex(*mesh_, point2);
  EXPECT_EQ(vid2, 13);  // Center vertex
  EXPECT_NEAR(dist2, 0.0, 1e-10);
}

// Test k-nearest neighbors
TEST_F(KDTreeAccelTest, KNearestVertices) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> point = {0.25, 0.25, 0.25};
  size_t k = 4;

  auto results = kdtree_->k_nearest_vertices(*mesh_, point, k);
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
TEST_F(KDTreeAccelTest, VerticesInRadius) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  real_t radius = 0.6;

  auto results = kdtree_->vertices_in_radius(*mesh_, point, radius);
  EXPECT_GT(results.size(), 0);

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
TEST_F(KDTreeAccelTest, RayIntersection) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through mesh
  std::array<real_t,3> origin = {-0.5, 0.5, 0.5};
  std::array<real_t,3> direction = {1.0, 0.0, 0.0};

  auto result = kdtree_->intersect_ray(*mesh_, origin, direction);
  EXPECT_TRUE(result.hit);
  EXPECT_GT(result.distance, 0.0);
  EXPECT_LT(result.distance, 1.0);

  // Ray missing mesh
  std::array<real_t,3> origin2 = {-0.5, -0.5, -0.5};
  std::array<real_t,3> direction2 = {0.0, -1.0, 0.0};

  auto result2 = kdtree_->intersect_ray(*mesh_, origin2, direction2);
  EXPECT_FALSE(result2.hit);
}

// Test all ray intersections
TEST_F(KDTreeAccelTest, AllRayIntersections) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through entire mesh
  std::array<real_t,3> origin = {-0.5, 0.5, 0.5};
  std::array<real_t,3> direction = {1.0, 0.0, 0.0};

  auto results = kdtree_->intersect_ray_all(*mesh_, origin, direction);
  EXPECT_GE(results.size(), 2);  // Should hit at least entry and exit

  // Check ordering
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i-1].distance, results[i].distance);
  }
}

// Test cells in box
TEST_F(KDTreeAccelTest, CellsInBox) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Box containing part of mesh
  std::array<real_t,3> box_min = {0.25, 0.25, 0.25};
  std::array<real_t,3> box_max = {0.75, 0.75, 0.75};

  auto results = kdtree_->cells_in_box(*mesh_, box_min, box_max);
  EXPECT_GT(results.size(), 0);
  EXPECT_LE(results.size(), mesh_->n_cells());

  // All cells should be valid
  for (index_t cell_id : results) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }
}

// Test cells in sphere
TEST_F(KDTreeAccelTest, CellsInSphere) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> center = {0.5, 0.5, 0.5};
  real_t radius = 0.4;

  auto results = kdtree_->cells_in_sphere(*mesh_, center, radius);
  EXPECT_GT(results.size(), 0);
  EXPECT_LE(results.size(), mesh_->n_cells());

  // All cells should be valid
  for (index_t cell_id : results) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }
}

// Test with hint
TEST_F(KDTreeAccelTest, PointLocationWithHint) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // First find without hint
  std::array<real_t,3> point1 = {0.25, 0.25, 0.25};
  auto result1 = kdtree_->locate_point(*mesh_, point1);
  EXPECT_TRUE(result1.found);

  // Now use as hint for nearby point
  std::array<real_t,3> point2 = {0.26, 0.26, 0.26};
  auto result2 = kdtree_->locate_point(*mesh_, point2, result1.cell_id);
  EXPECT_TRUE(result2.found);
  EXPECT_EQ(result2.cell_id, result1.cell_id);  // Should be same cell
}

// Performance test with larger mesh
TEST_F(KDTreeAccelTest, PerformanceLargeMesh) {
  create_large_mesh(10000);
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;

  auto build_start = std::chrono::steady_clock::now();
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);
  auto build_end = std::chrono::steady_clock::now();

  auto build_time = std::chrono::duration<double, std::milli>(
    build_end - build_start).count();

  std::cout << "KD-Tree build time for 10000 vertices: " << build_time << " ms\n";

  auto stats = kdtree_->get_stats();
  std::cout << "Tree depth: " << stats.tree_depth << "\n";
  std::cout << "Number of nodes: " << stats.n_nodes << "\n";
  std::cout << "Memory usage: " << stats.memory_bytes / 1024.0 << " KB\n";

  // Test query performance
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_t> dist(0.0, 10.0);

  auto query_start = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    std::array<real_t,3> point = {dist(gen), dist(gen), dist(gen)};
    kdtree_->nearest_vertex(*mesh_, point);
  }
  auto query_end = std::chrono::steady_clock::now();

  auto query_time = std::chrono::duration<double, std::milli>(
    query_end - query_start).count();

  std::cout << "Average nearest neighbor query time: "
            << query_time / 1000.0 << " ms\n";

  EXPECT_LT(build_time, 1000.0);  // Should build in under 1 second
  EXPECT_LT(query_time / 1000.0, 1.0);  // Queries should be under 1ms each
}

// Test deformed configuration
TEST_F(KDTreeAccelTest, DeformedConfiguration) {
  kdtree_ = std::make_unique<KDTreeAccel>();

  // Add deformed coordinates
  for (index_t i = 0; i < mesh_->n_vertices(); ++i) {
    auto coords = mesh_->get_vertex_coords(i);
    // Apply simple deformation (stretch in x)
    coords[0] *= 1.5;
    mesh_->set_vertex_deformed_coords(i, coords);
  }

  MeshSearch::SearchConfig config;
  kdtree_->build(*mesh_, IAccel::Configuration::Deformed, config);

  EXPECT_TRUE(kdtree_->is_built());
  EXPECT_EQ(kdtree_->built_config(), IAccel::Configuration::Deformed);

  // Test point in deformed mesh
  std::array<real_t,3> point = {0.75, 0.5, 0.5};  // Stretched point
  auto result = kdtree_->locate_point(*mesh_, point);
  EXPECT_TRUE(result.found);
}

// Test empty mesh
TEST_F(KDTreeAccelTest, EmptyMesh) {
  mesh_ = std::make_unique<MeshBase>();
  mesh_->finalize();

  kdtree_ = std::make_unique<KDTreeAccel>();
  MeshSearch::SearchConfig config;

  ASSERT_NO_THROW(kdtree_->build(*mesh_, IAccel::Configuration::Reference, config));

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  auto result = kdtree_->locate_point(*mesh_, point);
  EXPECT_FALSE(result.found);

  auto [vid, dist] = kdtree_->nearest_vertex(*mesh_, point);
  EXPECT_EQ(vid, -1);
}

// Test tree balance
TEST_F(KDTreeAccelTest, TreeBalance) {
  create_large_mesh(1000);
  kdtree_ = std::make_unique<KDTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  kdtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  auto stats = kdtree_->get_stats();

  // For a balanced tree with n nodes, depth should be approximately log2(n)
  real_t expected_depth = std::log2(mesh_->n_vertices());
  real_t actual_depth = static_cast<real_t>(stats.tree_depth);

  // Allow some deviation from perfect balance
  EXPECT_LT(actual_depth, expected_depth * 1.5);

  std::cout << "Tree balance: expected depth ~" << expected_depth
            << ", actual depth = " << actual_depth << "\n";
}

} // namespace test
} // namespace svmp
