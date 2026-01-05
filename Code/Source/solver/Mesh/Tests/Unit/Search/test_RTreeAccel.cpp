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
#include "Search/RTreeAccel.h"
#include "Search/SearchPrimitives.h"
#include "Core/MeshBase.h"
#include "Geometry/MeshGeometry.h"
#include <random>
#include <chrono>

namespace svmp {
namespace test {

// Test fixture for RTreeAccel tests
class RTreeAccelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a simple test mesh
    create_test_mesh();
  }

  void create_test_mesh() {
    // Create a mesh suitable for testing R-Tree dynamic operations
    mesh_ = std::make_unique<MeshBase>();

    // Create a 3x3 grid of vertices
    int idx = 0;
    for (real_t x = 0; x <= 2.0; x += 1.0) {
      for (real_t y = 0; y <= 2.0; y += 1.0) {
        mesh_->add_vertex(idx++, {x, y, 0.0});
      }
    }

    // Add more vertices for 3D
    for (real_t x = 0; x <= 2.0; x += 1.0) {
      for (real_t y = 0; y <= 2.0; y += 1.0) {
        mesh_->add_vertex(idx++, {x, y, 1.0});
      }
    }

    // Create quad cells in the bottom layer
    int cell_id = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        int v0 = i * 3 + j;
        int v1 = v0 + 1;
        int v2 = v0 + 4;
        int v3 = v0 + 3;
        mesh_->add_cell(cell_id++, CellShape::Quadrilateral, {v0, v1, v2, v3});
      }
    }

    // Create prism cells between layers
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        int v0 = i * 3 + j;
        int v1 = v0 + 1;
        int v2 = v0 + 3;
        int v3 = v0 + 9;
        int v4 = v0 + 10;
        int v5 = v0 + 12;
        mesh_->add_cell(cell_id++, CellShape::Prism, {v0, v1, v2, v3, v4, v5});
      }
    }

    mesh_->finalize();
  }

  void create_dynamic_mesh(int initial_cells) {
    mesh_ = std::make_unique<MeshBase>();

    // Create initial vertices
    std::mt19937 gen(42);
    std::uniform_real_distribution<real_t> dist(0.0, 10.0);

    int n_vertices = initial_cells * 4;
    for (int i = 0; i < n_vertices; ++i) {
      mesh_->add_vertex(i, {dist(gen), dist(gen), dist(gen)});
    }

    // Create initial cells (tetrahedra)
    for (int i = 0; i < initial_cells; ++i) {
      std::vector<index_t> tet;
      for (int j = 0; j < 4; ++j) {
        tet.push_back((i * 4 + j) % n_vertices);
      }
      mesh_->add_cell(i, CellShape::Tetrahedron, tet);
    }

    mesh_->finalize();
  }

  std::unique_ptr<MeshBase> mesh_;
  std::unique_ptr<RTreeAccel> rtree_;
};

// Test building R-Tree
TEST_F(RTreeAccelTest, Build) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::PointLocation;

  ASSERT_NO_THROW(rtree_->build(*mesh_, IAccel::Configuration::Reference, config));
  EXPECT_TRUE(rtree_->is_built());
  EXPECT_EQ(rtree_->built_config(), IAccel::Configuration::Reference);

  auto stats = rtree_->get_stats();
  EXPECT_GT(stats.n_nodes, 0);
  EXPECT_GT(stats.tree_depth, 0);
  EXPECT_GT(stats.memory_bytes, 0);

  std::cout << "R-Tree stats for test mesh:\n";
  std::cout << "  Nodes: " << stats.n_nodes << "\n";
  std::cout << "  Depth: " << stats.tree_depth << "\n";
  std::cout << "  Memory: " << stats.memory_bytes << " bytes\n";
}

// Test clearing R-Tree
TEST_F(RTreeAccelTest, Clear) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);
  EXPECT_TRUE(rtree_->is_built());

  rtree_->clear();
  EXPECT_FALSE(rtree_->is_built());

  auto stats = rtree_->get_stats();
  EXPECT_EQ(stats.n_nodes, 0);
}

// Test dynamic insertion
TEST_F(RTreeAccelTest, DynamicInsertion) {
  rtree_ = std::make_unique<RTreeAccel>();

  // Start with empty tree
  rtree_->clear();

  // Insert entities one by one
  std::vector<search::AABB> bounds;
  for (int i = 0; i < 10; ++i) {
    search::AABB aabb;
    aabb.min = {real_t(i), 0.0, 0.0};
    aabb.max = {real_t(i + 1), 1.0, 1.0};
    bounds.push_back(aabb);
    rtree_->insert(i, aabb);
  }

  // Verify insertions with searches
  for (int i = 0; i < 10; ++i) {
    std::vector<index_t> results;
    std::array<real_t,3> box_min = {real_t(i) + 0.1, 0.1, 0.1};
    std::array<real_t,3> box_max = {real_t(i) + 0.9, 0.9, 0.9};

    // Create a dummy mesh for testing
    MeshBase dummy_mesh;
    dummy_mesh.finalize();

    auto found = rtree_->cells_in_box(dummy_mesh, box_min, box_max);
    EXPECT_GT(found.size(), 0);

    bool found_i = false;
    for (index_t id : found) {
      if (id == i) {
        found_i = true;
        break;
      }
    }
    EXPECT_TRUE(found_i);
  }

  auto stats = rtree_->get_stats();
  std::cout << "R-Tree after 10 insertions:\n";
  std::cout << "  Nodes: " << stats.n_nodes << "\n";
  std::cout << "  Depth: " << stats.tree_depth << "\n";
}

// Test dynamic deletion
TEST_F(RTreeAccelTest, DynamicDeletion) {
  rtree_ = std::make_unique<RTreeAccel>();

  // Build initial tree with bulk load
  std::vector<std::pair<index_t, search::AABB>> entries;
  for (int i = 0; i < 20; ++i) {
    search::AABB aabb;
    aabb.min = {real_t(i % 5), real_t(i / 5), 0.0};
    aabb.max = {real_t(i % 5 + 1), real_t(i / 5 + 1), 1.0};
    entries.emplace_back(i, aabb);
  }

  rtree_->bulk_load(entries);

  auto initial_stats = rtree_->get_stats();
  std::cout << "Initial R-Tree with 20 entries:\n";
  std::cout << "  Nodes: " << initial_stats.n_nodes << "\n";

  // Remove some entries
  for (int i = 0; i < 10; i += 2) {
    rtree_->remove(i, entries[i].second);
  }

  auto after_delete_stats = rtree_->get_stats();
  std::cout << "R-Tree after removing 5 entries:\n";
  std::cout << "  Nodes: " << after_delete_stats.n_nodes << "\n";

  EXPECT_LE(after_delete_stats.n_nodes, initial_stats.n_nodes);

  // Verify removed entries are gone
  MeshBase dummy_mesh;
  dummy_mesh.finalize();

  for (int i = 0; i < 10; i += 2) {
    auto& aabb = entries[i].second;
    std::array<real_t,3> center = aabb.center();

    auto found = rtree_->cells_in_sphere(dummy_mesh, center, 0.1);

    bool found_removed = false;
    for (index_t id : found) {
      if (id == i) {
        found_removed = true;
        break;
      }
    }
    EXPECT_FALSE(found_removed);
  }
}

// Test bulk loading (STR algorithm)
TEST_F(RTreeAccelTest, BulkLoading) {
  rtree_ = std::make_unique<RTreeAccel>();

  // Create many entries for bulk loading
  std::vector<std::pair<index_t, search::AABB>> entries;

  std::mt19937 gen(42);
  std::uniform_real_distribution<real_t> dist(0.0, 100.0);

  for (int i = 0; i < 1000; ++i) {
    search::AABB aabb;
    aabb.min = {dist(gen), dist(gen), dist(gen)};
    for (int j = 0; j < 3; ++j) {
      aabb.max[j] = aabb.min[j] + dist(gen) * 0.1;  // Small boxes
    }
    entries.emplace_back(i, aabb);
  }

  auto bulk_start = std::chrono::steady_clock::now();
  rtree_->bulk_load(entries);
  auto bulk_end = std::chrono::steady_clock::now();

  auto bulk_time = std::chrono::duration<double, std::milli>(
    bulk_end - bulk_start).count();

  auto bulk_stats = rtree_->get_stats();

  std::cout << "Bulk load 1000 entries:\n";
  std::cout << "  Time: " << bulk_time << " ms\n";
  std::cout << "  Nodes: " << bulk_stats.n_nodes << "\n";
  std::cout << "  Depth: " << bulk_stats.tree_depth << "\n";

  // Compare with incremental insertion
  rtree_->clear();

  auto inc_start = std::chrono::steady_clock::now();
  for (const auto& [id, aabb] : entries) {
    rtree_->insert(id, aabb);
  }
  auto inc_end = std::chrono::steady_clock::now();

  auto inc_time = std::chrono::duration<double, std::milli>(
    inc_end - inc_start).count();

  auto inc_stats = rtree_->get_stats();

  std::cout << "Incremental insert 1000 entries:\n";
  std::cout << "  Time: " << inc_time << " ms\n";
  std::cout << "  Nodes: " << inc_stats.n_nodes << "\n";
  std::cout << "  Depth: " << inc_stats.tree_depth << "\n";

  // Bulk load should be significantly faster
  EXPECT_LT(bulk_time, inc_time * 0.5);

  // Bulk load should produce better tree
  EXPECT_LE(bulk_stats.tree_depth, inc_stats.tree_depth);
}

// Test point location
TEST_F(RTreeAccelTest, PointLocation) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test point inside mesh
  std::array<real_t,3> point1 = {0.5, 0.5, 0.0};
  auto result1 = rtree_->locate_point(*mesh_, point1);
  EXPECT_TRUE(result1.found);
  EXPECT_GE(result1.cell_id, 0);
  EXPECT_LT(result1.cell_id, mesh_->n_cells());

  // Test point outside mesh
  std::array<real_t,3> point2 = {10.0, 10.0, 10.0};
  auto result2 = rtree_->locate_point(*mesh_, point2);
  EXPECT_FALSE(result2.found);
  EXPECT_EQ(result2.cell_id, -1);

  // Test point on boundary
  std::array<real_t,3> point3 = {1.0, 1.0, 0.0};
  auto result3 = rtree_->locate_point(*mesh_, point3);
  EXPECT_TRUE(result3.found);
}

// Test batch point location
TEST_F(RTreeAccelTest, BatchPointLocation) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::vector<std::array<real_t,3>> points = {
    {0.5, 0.5, 0.0},
    {1.5, 0.5, 0.0},
    {0.5, 1.5, 0.0},
    {1.5, 1.5, 0.0},
    {5.0, 5.0, 5.0}
  };

  auto results = rtree_->locate_points(*mesh_, points);
  EXPECT_EQ(results.size(), points.size());

  // First 4 should be found
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(results[i].found);
  }

  // Last one outside mesh
  EXPECT_FALSE(results[4].found);
}

// Test nearest vertex
TEST_F(RTreeAccelTest, NearestVertex) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Test nearest to a point
  std::array<real_t,3> point = {0.1, 0.1, 0.1};
  auto [vid, dist] = rtree_->nearest_vertex(*mesh_, point);
  EXPECT_GE(vid, 0);
  EXPECT_LT(vid, mesh_->n_vertices());

  // Should find vertex at origin or nearby
  EXPECT_LT(dist, 0.5);

  // Test far point
  std::array<real_t,3> far_point = {100.0, 100.0, 100.0};
  auto [vid2, dist2] = rtree_->nearest_vertex(*mesh_, far_point);
  EXPECT_GE(vid2, 0);
  EXPECT_LT(vid2, mesh_->n_vertices());
  EXPECT_GT(dist2, 100.0);
}

// Test k-nearest neighbors
TEST_F(RTreeAccelTest, KNearestVertices) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> point = {1.0, 1.0, 0.5};
  size_t k = std::min(size_t(5), mesh_->n_vertices());

  auto results = rtree_->k_nearest_vertices(*mesh_, point, k);
  EXPECT_EQ(results.size(), k);

  // Check ordering
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
TEST_F(RTreeAccelTest, VerticesInRadius) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::NearestNeighbor;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> point = {1.0, 1.0, 0.5};
  real_t radius = 1.5;

  auto results = rtree_->vertices_in_radius(*mesh_, point, radius);
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
TEST_F(RTreeAccelTest, RayIntersection) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through mesh
  std::array<real_t,3> origin = {-0.5, 1.0, 0.5};
  std::array<real_t,3> direction = {1.0, 0.0, 0.0};

  auto result = rtree_->intersect_ray(*mesh_, origin, direction);

  // Ray missing mesh
  std::array<real_t,3> origin2 = {-0.5, -0.5, -0.5};
  std::array<real_t,3> direction2 = {0.0, -1.0, 0.0};

  auto result2 = rtree_->intersect_ray(*mesh_, origin2, direction2);
  EXPECT_FALSE(result2.hit);
}

// Test all ray intersections
TEST_F(RTreeAccelTest, AllRayIntersections) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::RayIntersection;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Ray through mesh
  std::array<real_t,3> origin = {-0.5, 1.0, 0.5};
  std::array<real_t,3> direction = {1.0, 0.0, 0.0};

  auto results = rtree_->intersect_ray_all(*mesh_, origin, direction, 10.0);

  // Check ordering
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i-1].distance, results[i].distance);
  }
}

// Test cells in box
TEST_F(RTreeAccelTest, CellsInBox) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  // Box containing part of mesh
  std::array<real_t,3> box_min = {0.5, 0.5, -0.5};
  std::array<real_t,3> box_max = {1.5, 1.5, 1.5};

  auto results = rtree_->cells_in_box(*mesh_, box_min, box_max);
  EXPECT_GT(results.size(), 0);
  EXPECT_LE(results.size(), mesh_->n_cells());

  // All cells should be valid
  for (index_t cell_id : results) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }

  // Empty box
  std::array<real_t,3> empty_min = {10.0, 10.0, 10.0};
  std::array<real_t,3> empty_max = {11.0, 11.0, 11.0};

  auto empty_results = rtree_->cells_in_box(*mesh_, empty_min, empty_max);
  EXPECT_EQ(empty_results.size(), 0);
}

// Test cells in sphere
TEST_F(RTreeAccelTest, CellsInSphere) {
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);

  std::array<real_t,3> center = {1.0, 1.0, 0.5};
  real_t radius = 1.0;

  auto results = rtree_->cells_in_sphere(*mesh_, center, radius);
  EXPECT_GT(results.size(), 0);
  EXPECT_LE(results.size(), mesh_->n_cells());

  // All cells should be valid
  for (index_t cell_id : results) {
    EXPECT_GE(cell_id, 0);
    EXPECT_LT(cell_id, mesh_->n_cells());
  }
}

// Test mixed operations
TEST_F(RTreeAccelTest, MixedOperations) {
  rtree_ = std::make_unique<RTreeAccel>();

  // Start with bulk load
  std::vector<std::pair<index_t, search::AABB>> initial_entries;
  for (int i = 0; i < 50; ++i) {
    search::AABB aabb;
    aabb.min = {real_t(i % 10), real_t(i / 10), 0.0};
    aabb.max = {real_t(i % 10 + 1), real_t(i / 10 + 1), 1.0};
    initial_entries.emplace_back(i, aabb);
  }

  rtree_->bulk_load(initial_entries);

  // Add more entries dynamically
  for (int i = 50; i < 60; ++i) {
    search::AABB aabb;
    aabb.min = {real_t(i % 10), real_t(i / 10), 0.0};
    aabb.max = {real_t(i % 10 + 1), real_t(i / 10 + 1), 1.0};
    rtree_->insert(i, aabb);
  }

  // Remove some entries
  for (int i = 10; i < 20; ++i) {
    rtree_->remove(i, initial_entries[i].second);
  }

  auto final_stats = rtree_->get_stats();
  std::cout << "R-Tree after mixed operations (50 bulk + 10 insert - 10 remove):\n";
  std::cout << "  Nodes: " << final_stats.n_nodes << "\n";
  std::cout << "  Depth: " << final_stats.tree_depth << "\n";

  // Verify tree still works
  MeshBase dummy_mesh;
  dummy_mesh.finalize();

  std::array<real_t,3> query_min = {2.0, 2.0, 0.0};
  std::array<real_t,3> query_max = {4.0, 4.0, 1.0};

  auto results = rtree_->cells_in_box(dummy_mesh, query_min, query_max);

  // Should find entries in this range (excluding removed ones)
  for (index_t id : results) {
    EXPECT_TRUE(id < 10 || id >= 20);  // Removed entries should not be found
  }
}

// Performance test with large mesh
TEST_F(RTreeAccelTest, PerformanceLargeMesh) {
  create_dynamic_mesh(1000);
  rtree_ = std::make_unique<RTreeAccel>();

  MeshSearch::SearchConfig config;
  config.primary_use = MeshSearch::QueryType::PointLocation;

  auto build_start = std::chrono::steady_clock::now();
  rtree_->build(*mesh_, IAccel::Configuration::Reference, config);
  auto build_end = std::chrono::steady_clock::now();

  auto build_time = std::chrono::duration<double, std::milli>(
    build_end - build_start).count();

  std::cout << "R-Tree build time for 1000 cells: " << build_time << " ms\n";

  auto stats = rtree_->get_stats();
  std::cout << "Tree height: " << stats.tree_depth << "\n";
  std::cout << "Number of nodes: " << stats.n_nodes << "\n";
  std::cout << "Memory usage: " << stats.memory_bytes / 1024.0 << " KB\n";

  // Test query performance
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_t> dist(0.0, 10.0);

  auto query_start = std::chrono::steady_clock::now();
  int found_count = 0;
  for (int i = 0; i < 1000; ++i) {
    std::array<real_t,3> point = {dist(gen), dist(gen), dist(gen)};
    auto result = rtree_->locate_point(*mesh_, point);
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

// Test deformed configuration
TEST_F(RTreeAccelTest, DeformedConfiguration) {
  rtree_ = std::make_unique<RTreeAccel>();

  // Add deformed coordinates
  for (index_t i = 0; i < mesh_->n_vertices(); ++i) {
    auto coords = mesh_->get_vertex_coords(i);
    // Apply shear deformation
    coords[0] += coords[1] * 0.3;
    mesh_->set_vertex_deformed_coords(i, coords);
  }

  MeshSearch::SearchConfig config;
  rtree_->build(*mesh_, IAccel::Configuration::Deformed, config);

  EXPECT_TRUE(rtree_->is_built());
  EXPECT_EQ(rtree_->built_config(), IAccel::Configuration::Deformed);

  // Test point in deformed mesh
  std::array<real_t,3> point = {0.65, 0.5, 0.0};  // Adjusted for shear
  auto result = rtree_->locate_point(*mesh_, point);
  EXPECT_TRUE(result.found);
}

// Test empty mesh
TEST_F(RTreeAccelTest, EmptyMesh) {
  mesh_ = std::make_unique<MeshBase>();
  mesh_->finalize();

  rtree_ = std::make_unique<RTreeAccel>();
  MeshSearch::SearchConfig config;

  ASSERT_NO_THROW(rtree_->build(*mesh_, IAccel::Configuration::Reference, config));

  std::array<real_t,3> point = {0.5, 0.5, 0.5};
  auto result = rtree_->locate_point(*mesh_, point);
  EXPECT_FALSE(result.found);

  auto [vid, dist] = rtree_->nearest_vertex(*mesh_, point);
  EXPECT_EQ(vid, -1);
}

// Test tree balance after operations
TEST_F(RTreeAccelTest, TreeBalance) {
  rtree_ = std::make_unique<RTreeAccel>();

  // Create entries with varying distributions
  std::vector<std::pair<index_t, search::AABB>> entries;

  // Clustered entries
  for (int i = 0; i < 100; ++i) {
    search::AABB aabb;
    aabb.min = {0.0, 0.0, real_t(i * 0.01)};
    aabb.max = {1.0, 1.0, real_t(i * 0.01 + 0.01)};
    entries.emplace_back(i, aabb);
  }

  // Scattered entries
  for (int i = 100; i < 200; ++i) {
    search::AABB aabb;
    aabb.min = {real_t(i), real_t(i), 0.0};
    aabb.max = {real_t(i + 1), real_t(i + 1), 1.0};
    entries.emplace_back(i, aabb);
  }

  rtree_->bulk_load(entries);

  auto stats = rtree_->get_stats();

  // Check tree balance
  // For 200 entries with min=2 and max=6, expect depth around log_3(200/4) ≈ 3-4
  std::cout << "Tree balance test (200 mixed entries):\n";
  std::cout << "  Tree height: " << stats.tree_depth << "\n";
  std::cout << "  Nodes: " << stats.n_nodes << "\n";

  EXPECT_GE(stats.tree_depth, 2);
  EXPECT_LE(stats.tree_depth, 10);  // Should not be too deep

  // Node count should be reasonable
  // With max 6 entries per node, minimum nodes = 200/6 ≈ 34
  // With min 2 entries per node, maximum nodes = 200/2 = 100
  EXPECT_GE(stats.n_nodes, 30);
  EXPECT_LE(stats.n_nodes, 200);
}

} // namespace test
} // namespace svmp
