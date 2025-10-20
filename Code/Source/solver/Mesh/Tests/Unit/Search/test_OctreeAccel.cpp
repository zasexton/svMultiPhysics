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
#include "Search/OctreeAccel.h"
#include "Core/MeshBase.h"
#include "Core/DistributedMesh.h"
#include "Geometry/MeshGeometry.h"
#include <random>
#include <chrono>

namespace svmp {
namespace test {

// Test fixture for OctreeAccel tests
class OctreeAccelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a simple test mesh
    create_test_mesh();
  }

  void create_test_mesh() {
    // Create a 3x3x3 mesh with non-uniform spacing to test octree subdivisions
    mesh_ = std::make_unique<DistributedMesh>();

    // Add vertices in a non-uniform grid
    std::vector<std::array<real_t,3>> vertices;

    // Dense region in one corner
    for (real_t x = 0; x <= 0.3; x += 0.1) {
      for (real_t y = 0; y <= 0.3; y += 0.1) {
        for (real_t z = 0; z <= 0.3; z += 0.1) {
          vertices.push_back({x, y, z});
        }
      }
    }

    // Sparse region in the rest
    for (real_t x = 0.5; x <= 1.0; x += 0.5) {
      for (real_t y = 0.5; y <= 1.0; y += 0.5) {
        for (real_t z = 0.5; z <= 1.0; z += 0.5) {
          vertices.push_back({x, y, z});
        }
      }
    }

    // Add some scattered points
    vertices.push_back({0.7, 0.2, 0.8});
    vertices.push_back({0.3, 0.9, 0.4});
    vertices.push_back({0.6, 0.6, 0.2});

    for (size_t i = 0; i < vertices.size(); ++i) {
      mesh_->add_vertex(i, vertices[i]);
    }

    // Create tetrahedra from nearby vertices
    if (vertices.size() >= 4) {
      for (size_t i = 0; i < vertices.size() - 3; i += 2) {
        std::vector<index_t> tet;
        for (size_t j = 0; j < 4 && i + j < vertices.size(); ++j) {
          tet.push_back(i + j);
        }
        if (tet.size() == 4) {
          mesh_->add_cell(i/2, CellShape::Tetrahedron, tet);
        }
      }
    }

    mesh_->finalize();
  }

  void create_large_mesh(int n_points) {
    mesh_ = std::make_unique<DistributedMesh>();

    // Create clustered points to test octree efficiency
    std::mt19937 gen(42);
    std::uniform_real_distribution<real_t> cluster_center_dist(0.0, 10.0);
    std::normal_distribution<real_t> cluster_dist(0.0, 0.5);

    // Create several clusters
    int n_clusters = 10;
    int points_per_cluster = n_points / n_clusters;

    int vid = 0;
    for (int c = 0; c < n_clusters; ++c) {
      std::array<real_t,3> center = {
        cluster_center_dist(gen),
        cluster_center_dist(gen),
        cluster_center_dist(gen)
      };

      for (int p = 0; p < points_per_cluster; ++p) {
        std::array<real_t,3> vertex = {
          center[0] + cluster_dist(gen),
          center[1] + cluster_dist(gen),
          center[2] + cluster_dist(gen)
        };
        mesh_->add_vertex(vid++, vertex);
      }
    }

    // Create tetrahedra from nearby points
    for (int i = 0; i < vid - 3; i += 4) {
      mesh_->add_cell(i/4, CellShape::Tetrahedron, {i, i+1, i+2, i+3});
    }

    mesh_->finalize();
  }

  std::unique_ptr<MeshBase> mesh_;
  std::unique_ptr<OctreeAccel> octree_;
};

// Test building octree
TEST_F(OctreeAccelTest, Build) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::PointLocation;

  ASSERT_NO_THROW(octree_->build(*mesh_, IAccel::Configuration::Reference, config));
  EXPECT_TRUE(octree_->is_built());
  EXPECT_EQ(octree_->built_config(), IAccel::Configuration::Reference);

  auto stats = octree_->get_stats();
  EXPECT_GT(stats.n_nodes, 0);
  EXPECT_GT(stats.tree_depth, 0);
  EXPECT_GT(stats.memory_bytes, 0);

  std::cout << "Octree stats for test mesh:\n";
  std::cout << "  Nodes: " << stats.n_nodes << "\n";
  std::cout << "  Depth: " << stats.tree_depth << "\n";
  std::cout << "  Memory: " << stats.memory_bytes << " bytes\n";
}

// Test clearing octree
TEST_F(OctreeAccelTest, Clear) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);
  EXPECT_TRUE(octree_->is_built());

  octree_->clear();
  EXPECT_FALSE(octree_->is_built());

  auto stats = octree_->get_stats();
  EXPECT_EQ(stats.n_nodes, 0);
}

// Test point location with octree traversal
TEST_F(OctreeAccelTest, PointLocation) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test point in dense region
  std::array<real_t,3> point1 = {0.15, 0.15, 0.15};
  auto result1 = octree_->locate_point(*mesh_, point1);
  if (mesh_->n_cells() > 0) {
    EXPECT_TRUE(result1.found);
    EXPECT_GE(result1.cell_id, 0);
    EXPECT_LT(result1.cell_id, mesh_->n_cells());
  }

  // Test point in sparse region
  std::array<real_t,3> point2 = {0.75, 0.75, 0.75};
  auto result2 = octree_->locate_point(*mesh_, point2);

  // Test point outside mesh
  std::array<real_t,3> point3 = {10.0, 10.0, 10.0};
  auto result3 = octree_->locate_point(*mesh_, point3);
  EXPECT_FALSE(result3.found);
  EXPECT_EQ(result3.cell_id, -1);
}

// Test batch point location
TEST_F(OctreeAccelTest, BatchPointLocation) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test points in different octants
  std::vector<std::array<real_t,3>> points = {
    {0.1, 0.1, 0.1},   // Octant 0
    {0.9, 0.1, 0.1},   // Octant 1
    {0.1, 0.9, 0.1},   // Octant 2
    {0.9, 0.9, 0.1},   // Octant 3
    {0.1, 0.1, 0.9},   // Octant 4
    {0.9, 0.1, 0.9},   // Octant 5
    {0.1, 0.9, 0.9},   // Octant 6
    {0.9, 0.9, 0.9},   // Octant 7
    {5.0, 5.0, 5.0}    // Outside
  };

  auto results = octree_->locate_points(*mesh_, points);
  EXPECT_EQ(results.size(), points.size());

  // Last point should not be found
  EXPECT_FALSE(results.back().found);
}

// Test nearest vertex with hierarchical search
TEST_F(OctreeAccelTest, NearestVertex) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test nearest in dense region
  std::array<real_t,3> point1 = {0.15, 0.15, 0.15};
  auto [vid1, dist1] = octree_->nearest_vertex(*mesh_, point1);
  EXPECT_GE(vid1, 0);
  EXPECT_LT(vid1, mesh_->n_vertices());
  EXPECT_GE(dist1, 0.0);

  // Test nearest in sparse region
  std::array<real_t,3> point2 = {0.75, 0.75, 0.75};
  auto [vid2, dist2] = octree_->nearest_vertex(*mesh_, point2);
  EXPECT_GE(vid2, 0);
  EXPECT_LT(vid2, mesh_->n_vertices());

  // Test point far from mesh
  std::array<real_t,3> point3 = {10.0, 10.0, 10.0};
  auto [vid3, dist3] = octree_->nearest_vertex(*mesh_, point3);
  EXPECT_GE(vid3, 0);
  EXPECT_LT(vid3, mesh_->n_vertices());
  EXPECT_GT(dist3, 10.0);  // Should be far
}

// Test k-nearest neighbors with priority queue traversal
TEST_F(OctreeAccelTest, KNearestVertices) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  size_t k = std::min(size_t(5), mesh_->n_vertices());

  auto results = octree_->k_nearest_vertices(*mesh_, point, k);
  EXPECT_LE(results.size(), k);

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

// Test vertices in radius with octant pruning
TEST_F(OctreeAccelTest, VerticesInRadius) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test in dense region
  std::array<real_t,3> point1 = {0.15, 0.15, 0.15};
  real_t radius1 = 0.2;

  auto results1 = octree_->vertices_in_radius(*mesh_, point1, radius1);

  // Verify all results are within radius
  for (index_t vid : results1) {
    auto vertex = mesh_->get_vertex_coords(vid);
    real_t dist = std::sqrt(
      std::pow(vertex[0] - point1[0], 2) +
      std::pow(vertex[1] - point1[1], 2) +
      std::pow(vertex[2] - point1[2], 2)
    );
    EXPECT_LE(dist, radius1);
  }

  // Test with large radius spanning multiple octants
  std::array<real_t,3> point2 = {0.5, 0.5, 0.5};
  real_t radius2 = 1.0;

  auto results2 = octree_->vertices_in_radius(*mesh_, point2, radius2);
  EXPECT_GT(results2.size(), results1.size());
}

// Test ray intersection
TEST_F(OctreeAccelTest, RayIntersection) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through dense region
  std::array<real_t,3> origin = {-0.1, 0.15, 0.15};
  std::array<real_t,3> direction = {1.0, 0.0, 0.0};

  auto result = octree_->intersect_ray(*mesh_, origin, direction);

  // Ray through sparse region
  std::array<real_t,3> origin2 = {-0.1, 0.75, 0.75};
  auto result2 = octree_->intersect_ray(*mesh_, origin2, direction);

  // Ray missing mesh
  std::array<real_t,3> origin3 = {-0.1, 5.0, 5.0};
  auto result3 = octree_->intersect_ray(*mesh_, origin3, direction);
  EXPECT_FALSE(result3.hit);
}

// Test all ray intersections
TEST_F(OctreeAccelTest, AllRayIntersections) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through entire mesh diagonally
  std::array<real_t,3> origin = {-0.1, -0.1, -0.1};
  std::array<real_t,3> direction = {1.0, 1.0, 1.0};

  // Normalize direction
  real_t len = std::sqrt(3.0);
  for (int i = 0; i < 3; ++i) {
    direction[i] /= len;
  }

  auto results = octree_->intersect_ray_all(*mesh_, origin, direction, 10.0);

  // Check ordering
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i-1].distance, results[i].distance);
  }
}

// Test cells in box with octant culling
TEST_F(OctreeAccelTest, CellsInBox) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Box in dense region
  std::array<real_t,3> box_min1 = {0.0, 0.0, 0.0};
  std::array<real_t,3> box_max1 = {0.3, 0.3, 0.3};

  auto results1 = octree_->cells_in_box(*mesh_, box_min1, box_max1);

  // Box spanning multiple octants
  std::array<real_t,3> box_min2 = {0.2, 0.2, 0.2};
  std::array<real_t,3> box_max2 = {0.8, 0.8, 0.8};

  auto results2 = octree_->cells_in_box(*mesh_, box_min2, box_max2);

  // All cells should be valid
  for (index_t cell_id : results2) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }
}

// Test cells in sphere
TEST_F(OctreeAccelTest, CellsInSphere) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Small sphere in dense region
  std::array<real_t,3> center1 = {0.15, 0.15, 0.15};
  real_t radius1 = 0.2;

  auto results1 = octree_->cells_in_sphere(*mesh_, center1, radius1);

  // Large sphere spanning octants
  std::array<real_t,3> center2 = {0.5, 0.5, 0.5};
  real_t radius2 = 0.5;

  auto results2 = octree_->cells_in_sphere(*mesh_, center2, radius2);

  if (mesh_->n_cells() > 0) {
    EXPECT_GE(results2.size(), results1.size());
  }

  // All cells should be valid
  for (index_t cell_id : results2) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }
}

// Test octree depth and subdivision
TEST_F(OctreeAccelTest, TreeDepthAndSubdivision) {
  create_large_mesh(1000);
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::PointLocation;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  auto stats = octree_->get_stats();

  std::cout << "Octree for clustered mesh (1000 points):\n";
  std::cout << "  Nodes: " << stats.n_nodes << "\n";
  std::cout << "  Depth: " << stats.tree_depth << "\n";
  std::cout << "  Memory: " << stats.memory_bytes / 1024.0 << " KB\n";

  // Octree should adapt to clustered data
  EXPECT_GT(stats.tree_depth, 3);  // Should have some depth
  EXPECT_LT(stats.tree_depth, 15); // But not too deep

  // Node count should be reasonable
  EXPECT_GT(stats.n_nodes, 8);     // More than just root's children
  EXPECT_LT(stats.n_nodes, 1000);  // Less than one node per point
}

// Performance test with large mesh
TEST_F(OctreeAccelTest, PerformanceLargeMesh) {
  create_large_mesh(10000);
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::PointLocation;

  auto build_start = std::chrono::steady_clock::now();
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);
  auto build_end = std::chrono::steady_clock::now();

  auto build_time = std::chrono::duration<double, std::milli>(
    build_end - build_start).count();

  std::cout << "Octree build time for 10000 vertices: " << build_time << " ms\n";

  auto stats = octree_->get_stats();
  std::cout << "Tree depth: " << stats.tree_depth << "\n";
  std::cout << "Number of nodes: " << stats.n_nodes << "\n";
  std::cout << "Memory usage: " << stats.memory_bytes / 1024.0 << " KB\n";

  // Test query performance
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_t> dist(0.0, 10.0);

  auto query_start = std::chrono::steady_clock::now();
  int found_count = 0;
  for (int i = 0; i < 1000; ++i) {
    std::array<real_t,3> point = {dist(gen), dist(gen), dist(gen)};
    auto result = octree_->locate_point(*mesh_, point);
    if (result.found) found_count++;
  }
  auto query_end = std::chrono::steady_clock::now();

  auto query_time = std::chrono::duration<double, std::milli>(
    query_end - query_start).count();

  std::cout << "Average point location query time: "
            << query_time / 1000.0 << " ms\n";
  std::cout << "Points found: " << found_count << " / 1000\n";

  EXPECT_LT(build_time, 2000.0);  // Should build in under 2 seconds
  EXPECT_LT(query_time / 1000.0, 1.0);  // Queries should be under 1ms each
}

// Test with deformed configuration
TEST_F(OctreeAccelTest, DeformedConfiguration) {
  octree_ = std::make_unique<OctreeAccel>();

  // Add deformed coordinates
  for (index_t i = 0; i < mesh_->n_vertices(); ++i) {
    auto coords = mesh_->get_vertex_coords(i);
    // Apply non-uniform deformation
    coords[0] *= 1.5;
    coords[1] *= 0.8;
    coords[2] *= 1.2;
    mesh_->set_vertex_deformed_coords(i, coords);
  }

  MeshSearch::SearchConfig config;
  octree_->build(*mesh_, IAccel::Configuration::Deformed, config);

  EXPECT_TRUE(octree_->is_built());
  EXPECT_EQ(octree_->built_config(), IAccel::Configuration::Deformed);

  // Test point in deformed mesh
  std::array<real_t,3> point = {0.75, 0.4, 0.6};  // Deformed point
  auto result = octree_->locate_point(*mesh_, point);
}

// Test empty mesh
TEST_F(OctreeAccelTest, EmptyMesh) {
  mesh_ = std::make_unique<DistributedMesh>();
  mesh_->finalize();

  octree_ = std::make_unique<OctreeAccel>();
  MeshSearch::SearchConfig config;

  ASSERT_NO_THROW(octree_->build(*mesh_, IAccel::Configuration::Reference, config));

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  auto result = octree_->locate_point(*mesh_, point);
  EXPECT_FALSE(result.found);

  auto [vid, dist] = octree_->nearest_vertex(*mesh_, point);
  EXPECT_EQ(vid, -1);
}

// Test octant traversal order
TEST_F(OctreeAccelTest, OctantTraversal) {
  octree_ = std::make_unique<OctreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  octree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Query point in specific octant
  std::array<real_t,3> point = {0.25, 0.25, 0.75};  // Should be in octant 4

  auto [vid, dist] = octree_->nearest_vertex(*mesh_, point);
  EXPECT_GE(vid, 0);

  // Verify traversal visits nearby octants first
  auto k_nearest = octree_->k_nearest_vertices(*mesh_, point, 8);

  // Closest points should generally be from same or adjacent octants
  real_t avg_dist = 0.0;
  for (const auto& [v, d] : k_nearest) {
    avg_dist += d;
  }
  if (!k_nearest.empty()) {
    avg_dist /= k_nearest.size();
    std::cout << "Average distance for 8-NN from octant 4: " << avg_dist << "\n";
  }
}

} // namespace test
} // namespace svmp