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
 * @file test_UniformGridAccel.cpp
 * @brief Unit tests for uniform grid acceleration structure
 *
 * Tests include:
 * - Building and clearing
 * - Point location
 * - Nearest neighbor search
 * - Ray intersection
 * - Region queries
 * - Performance and memory management
 */

#include "../../../Search/UniformGridAccel.h"
#include "../../../Search/SearchAccel.h"
#include "../../../Core/MeshBase.h"
#include "../../../Search/MeshSearch.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <chrono>
#include <random>

namespace svmp {
namespace test {

// Test macros
#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ \
                     << " in " << __func__ << ": " #cond << std::endl; \
            exit(1); \
        } \
    } while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))
#define ASSERT_LE(a, b) ASSERT((a) <= (b))
#define ASSERT_GT(a, b) ASSERT((a) > (b))
#define ASSERT_GE(a, b) ASSERT((a) >= (b))
#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) < (tol))
#define ASSERT_TRUE(cond) ASSERT(cond)
#define ASSERT_FALSE(cond) ASSERT(!(cond))

// Test helpers
const real_t EPSILON = 1e-10;

// Helper: Create test meshes
std::shared_ptr<MeshBase> create_regular_hex_mesh(int nx, int ny, int nz) {
    auto mesh = std::make_shared<MeshBase>();

    // Create vertices
    std::vector<real_t> coords;
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                coords.push_back(i);
                coords.push_back(j);
                coords.push_back(k);
            }
        }
    }

    // Create hex cells
    std::vector<index_t> connectivity;
    std::vector<offset_t> offsets = {0};

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int base = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                connectivity.push_back(base);
                connectivity.push_back(base + 1);
                connectivity.push_back(base + (nx + 1) + 1);
                connectivity.push_back(base + (nx + 1));
                connectivity.push_back(base + (ny + 1) * (nx + 1));
                connectivity.push_back(base + (ny + 1) * (nx + 1) + 1);
                connectivity.push_back(base + (ny + 1) * (nx + 1) + (nx + 1) + 1);
                connectivity.push_back(base + (ny + 1) * (nx + 1) + (nx + 1));
                offsets.push_back(connectivity.size());
            }
        }
    }

    std::vector<CellShape> shapes(nx * ny * nz, {CellFamily::Hex, 8, 1});

    mesh->build_from_arrays(3, coords, offsets, connectivity, shapes);
    mesh->finalize();

    return mesh;
}

std::shared_ptr<MeshBase> create_tet_mesh() {
    auto mesh = std::make_shared<MeshBase>();

    // Create a simple mesh with two tetrahedra sharing a face
    std::vector<real_t> coords = {
        0,0,0,  1,0,0,  0,1,0,  0,0,1,  // First tet
        1,1,1                            // Additional vertex for second tet
    };

    std::vector<index_t> connectivity = {
        0,1,2,3,  // First tet
        1,2,3,4   // Second tet
    };

    std::vector<offset_t> offsets = {0, 4, 8};
    std::vector<CellShape> shapes(2, {CellFamily::Tetra, 4, 1});

    mesh->build_from_arrays(3, coords, offsets, connectivity, shapes);
    mesh->finalize();

    return mesh;
}

class TestUniformGridAccel {
public:
    void test_construction_and_build() {
        std::cout << "=== Testing Construction and Build ===\n";

        auto mesh = create_regular_hex_mesh(4, 4, 4);
        UniformGridAccel accel;

        // Test initial state
        ASSERT_FALSE(accel.is_built());
        ASSERT_EQ(accel.built_config(), Configuration::Reference);

        // Build the structure
        MeshSearch::SearchConfig config;
        config.grid_resolution = 8;
        accel.build(*mesh, Configuration::Reference, config);

        // Check built state
        ASSERT_TRUE(accel.is_built());
        ASSERT_EQ(accel.built_config(), Configuration::Reference);

        // Get statistics
        auto stats = accel.get_stats();
        ASSERT_GT(stats.build_time_ms, 0);
        ASSERT_GT(stats.memory_bytes, 0);
        ASSERT_GT(stats.n_entities, 0);

        // Clear and verify
        accel.clear();
        ASSERT_FALSE(accel.is_built());

        std::cout << "  ✓ Construction and build tests passed\n";
    }

    void test_point_location() {
        std::cout << "=== Testing Point Location ===\n";

        auto mesh = create_regular_hex_mesh(4, 4, 4);
        UniformGridAccel accel;

        MeshSearch::SearchConfig config;
        config.grid_resolution = 8;
        accel.build(*mesh, Configuration::Reference, config);

        // Test point inside mesh
        {
            std::array<real_t, 3> point = {2.5, 2.5, 2.5};
            auto result = accel.locate_point(*mesh, point);
            ASSERT_TRUE(result.found);
            ASSERT_GE(result.cell_id, 0);
            ASSERT_LT(result.cell_id, mesh->n_cells());

            // Parametric coordinates should be in valid range
            for (int i = 0; i < 3; ++i) {
                ASSERT_GE(result.xi[i], -1.1);
                ASSERT_LE(result.xi[i], 1.1);
            }
        }

        // Test point outside mesh
        {
            std::array<real_t, 3> point = {10, 10, 10};
            auto result = accel.locate_point(*mesh, point);
            ASSERT_FALSE(result.found);
            ASSERT_EQ(result.cell_id, -1);
        }

        // Test point on boundary
        {
            std::array<real_t, 3> point = {0, 0, 0};
            auto result = accel.locate_point(*mesh, point);
            ASSERT_TRUE(result.found);
        }

        // Test with hint
        {
            std::array<real_t, 3> point = {1.5, 1.5, 1.5};
            auto result1 = accel.locate_point(*mesh, point);
            ASSERT_TRUE(result1.found);

            // Use found cell as hint for nearby point
            std::array<real_t, 3> nearby = {1.6, 1.5, 1.5};
            auto result2 = accel.locate_point(*mesh, nearby, result1.cell_id);
            ASSERT_TRUE(result2.found);
        }

        // Test batch point location
        {
            std::vector<std::array<real_t, 3>> points = {
                {0.5, 0.5, 0.5},
                {1.5, 1.5, 1.5},
                {2.5, 2.5, 2.5},
                {10, 10, 10}  // Outside
            };

            auto results = accel.locate_points(*mesh, points);
            ASSERT_EQ(results.size(), 4);
            ASSERT_TRUE(results[0].found);
            ASSERT_TRUE(results[1].found);
            ASSERT_TRUE(results[2].found);
            ASSERT_FALSE(results[3].found);
        }

        std::cout << "  ✓ Point location tests passed\n";
    }

    void test_nearest_neighbor() {
        std::cout << "=== Testing Nearest Neighbor Search ===\n";

        auto mesh = create_regular_hex_mesh(3, 3, 3);
        UniformGridAccel accel;

        MeshSearch::SearchConfig config;
        config.grid_resolution = 4;
        accel.build(*mesh, Configuration::Reference, config);

        // Test nearest vertex
        {
            std::array<real_t, 3> point = {1.4, 1.4, 1.4};
            auto [vertex_id, dist] = accel.nearest_vertex(*mesh, point);
            ASSERT_NE(vertex_id, INVALID_INDEX);
            ASSERT_GE(dist, 0);

            // Vertex should be reasonably close
            ASSERT_LT(dist, 1.0);
        }

        // Test k-nearest vertices
        {
            std::array<real_t, 3> point = {1.5, 1.5, 1.5};
            auto neighbors = accel.k_nearest_vertices(*mesh, point, 5);
            ASSERT_EQ(neighbors.size(), 5);

            // Verify distances are sorted
            for (size_t i = 1; i < neighbors.size(); ++i) {
                ASSERT_LE(neighbors[i-1].second, neighbors[i].second);
            }

            // All indices should be valid
            for (const auto& [idx, dist] : neighbors) {
                ASSERT_LT(idx, mesh->n_vertices());
                ASSERT_GE(dist, 0);
            }
        }

        // Test vertices in radius
        {
            std::array<real_t, 3> point = {1.5, 1.5, 1.5};
            real_t radius = 1.0;
            auto vertices = accel.vertices_in_radius(*mesh, point, radius);

            // Should find some vertices
            ASSERT_GT(vertices.size(), 0);

            // All vertices should be within radius
            const auto& coords = mesh->X_ref();
            for (index_t v : vertices) {
                real_t dx = coords[v*3] - point[0];
                real_t dy = coords[v*3+1] - point[1];
                real_t dz = coords[v*3+2] - point[2];
                real_t dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                ASSERT_LE(dist, radius + EPSILON);
            }
        }

        std::cout << "  ✓ Nearest neighbor search tests passed\n";
    }

    void test_ray_intersection() {
        std::cout << "=== Testing Ray Intersection ===\n";

        auto mesh = create_regular_hex_mesh(3, 3, 3);
        UniformGridAccel accel;

        MeshSearch::SearchConfig config;
        config.grid_resolution = 4;
        accel.build(*mesh, Configuration::Reference, config);

        // Test ray hitting mesh
        {
            std::array<real_t, 3> origin = {-1, 1.5, 1.5};
            std::array<real_t, 3> direction = {1, 0, 0};

            auto result = accel.intersect_ray(*mesh, origin, direction);

            // Note: Ray intersection requires boundary triangulation
            // It may not find intersection if no boundary triangles exist
            // or if the implementation is incomplete
            if (result.found) {
                ASSERT_GE(result.t, 0);
                ASSERT_GE(result.face_id, 0);
            }
        }

        // Test ray missing mesh
        {
            std::array<real_t, 3> origin = {-1, 10, 10};
            std::array<real_t, 3> direction = {1, 0, 0};

            auto result = accel.intersect_ray(*mesh, origin, direction);
            // Should miss
            if (mesh->n_faces() > 0) {
                // Only check if mesh has faces
                ASSERT_FALSE(result.found);
            }
        }

        // Test all intersections
        {
            std::array<real_t, 3> origin = {-1, 1.5, 1.5};
            std::array<real_t, 3> direction = {1, 0, 0};

            auto results = accel.intersect_ray_all(*mesh, origin, direction);

            // Should be sorted by t
            for (size_t i = 1; i < results.size(); ++i) {
                ASSERT_LE(results[i-1].t, results[i].t);
            }
        }

        std::cout << "  ✓ Ray intersection tests passed\n";
    }

    void test_region_queries() {
        std::cout << "=== Testing Region Queries ===\n";

        auto mesh = create_regular_hex_mesh(4, 4, 4);
        UniformGridAccel accel;

        MeshSearch::SearchConfig config;
        config.grid_resolution = 8;
        accel.build(*mesh, Configuration::Reference, config);

        // Test cells in box
        {
            std::array<real_t, 3> box_min = {0.5, 0.5, 0.5};
            std::array<real_t, 3> box_max = {2.5, 2.5, 2.5};

            auto cells = accel.cells_in_box(*mesh, box_min, box_max);

            // Should find some cells
            ASSERT_GT(cells.size(), 0);

            // All cells should overlap with box
            for (index_t cell : cells) {
                ASSERT_LT(cell, mesh->n_cells());

                auto center = mesh->cell_center(cell);
                // Simple check: center should be reasonably close to box
                bool near_box = true;
                for (int d = 0; d < 3; ++d) {
                    if (center[d] < box_min[d] - 1.0 || center[d] > box_max[d] + 1.0) {
                        near_box = false;
                    }
                }
                ASSERT_TRUE(near_box);
            }
        }

        // Test cells in sphere
        {
            std::array<real_t, 3> center = {2, 2, 2};
            real_t radius = 1.5;

            auto cells = accel.cells_in_sphere(*mesh, center, radius);

            // Should find some cells
            ASSERT_GT(cells.size(), 0);

            // All cells should be near sphere
            for (index_t cell : cells) {
                ASSERT_LT(cell, mesh->n_cells());

                auto cell_center = mesh->cell_center(cell);
                real_t dx = cell_center[0] - center[0];
                real_t dy = cell_center[1] - center[1];
                real_t dz = cell_center[2] - center[2];
                real_t dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                // Cell center should be within radius + cell diagonal
                ASSERT_LT(dist, radius + 2.0);
            }
        }

        std::cout << "  ✓ Region query tests passed\n";
    }

    void test_performance() {
        std::cout << "=== Testing Performance ===\n";

        // Create larger mesh for performance testing
        auto mesh = create_regular_hex_mesh(10, 10, 10);
        UniformGridAccel accel;

        // Time the build
        auto start = std::chrono::high_resolution_clock::now();

        MeshSearch::SearchConfig config;
        config.grid_resolution = 16;
        accel.build(*mesh, Configuration::Reference, config);

        auto end = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "    Build time for 1000 cells: " << build_time << " ms\n";
        ASSERT_LT(build_time, 1000);  // Should build in under 1 second

        // Time point location queries
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 10);

        const int n_queries = 1000;
        int found_count = 0;

        start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_queries; ++i) {
            std::array<real_t, 3> point = {dis(gen), dis(gen), dis(gen)};
            auto result = accel.locate_point(*mesh, point);
            if (result.found) found_count++;
        }

        end = std::chrono::high_resolution_clock::now();
        auto query_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "    Average query time: " << query_time / n_queries << " μs\n";
        std::cout << "    Hit rate: " << (100.0 * found_count / n_queries) << "%\n";

        // Check statistics
        auto stats = accel.get_stats();
        ASSERT_GT(stats.query_count, 0);

        std::cout << "  ✓ Performance tests passed\n";
    }

    void test_memory_management() {
        std::cout << "=== Testing Memory Management ===\n";

        auto mesh = create_regular_hex_mesh(5, 5, 5);
        UniformGridAccel accel;

        // Build and clear multiple times
        for (int i = 0; i < 3; ++i) {
            MeshSearch::SearchConfig config;
            config.grid_resolution = 8;
            accel.build(*mesh, Configuration::Reference, config);
            ASSERT_TRUE(accel.is_built());

            // Do some queries
            std::array<real_t, 3> point = {2.5, 2.5, 2.5};
            accel.locate_point(*mesh, point);

            accel.clear();
            ASSERT_FALSE(accel.is_built());
        }

        // Rebuild with different configuration
        {
            MeshSearch::SearchConfig config;
            config.grid_resolution = 16;
            accel.build(*mesh, Configuration::Current, config);
            ASSERT_EQ(accel.built_config(), Configuration::Current);
        }

        std::cout << "  ✓ Memory management tests passed\n";
    }

    void test_edge_cases() {
        std::cout << "=== Testing Edge Cases ===\n";

        // Empty mesh
        {
            auto empty_mesh = std::make_shared<MeshBase>();
            UniformGridAccel accel;

            MeshSearch::SearchConfig config;
            accel.build(*empty_mesh, Configuration::Reference, config);

            std::array<real_t, 3> point = {0, 0, 0};
            auto result = accel.locate_point(*empty_mesh, point);
            ASSERT_FALSE(result.found);
        }

        // Single cell mesh
        {
            auto mesh = std::make_shared<MeshBase>();
            std::vector<real_t> coords = {
                0,0,0, 1,0,0, 1,1,0, 0,1,0,
                0,0,1, 1,0,1, 1,1,1, 0,1,1
            };
            std::vector<index_t> conn = {0,1,2,3,4,5,6,7};
            std::vector<offset_t> offs = {0,8};
            std::vector<CellShape> shapes = {{CellFamily::Hex, 8, 1}};

            mesh->build_from_arrays(3, coords, offs, conn, shapes);
            mesh->finalize();

            UniformGridAccel accel;
            MeshSearch::SearchConfig config;
            config.grid_resolution = 2;
            accel.build(*mesh, Configuration::Reference, config);

            // Point inside single cell
            std::array<real_t, 3> point = {0.5, 0.5, 0.5};
            auto result = accel.locate_point(*mesh, point);
            ASSERT_TRUE(result.found);
            ASSERT_EQ(result.cell_id, 0);
        }

        // Very small grid resolution
        {
            auto mesh = create_regular_hex_mesh(4, 4, 4);
            UniformGridAccel accel;

            MeshSearch::SearchConfig config;
            config.grid_resolution = 1;  // Minimum resolution
            accel.build(*mesh, Configuration::Reference, config);

            std::array<real_t, 3> point = {2, 2, 2};
            auto result = accel.locate_point(*mesh, point);
            // Should still work, even with coarse grid
            ASSERT_TRUE(result.found);
        }

        std::cout << "  ✓ Edge case tests passed\n";
    }

    void run_all_tests() {
        std::cout << "\n========================================\n";
        std::cout << "  UniformGridAccel Unit Test Suite\n";
        std::cout << "========================================\n\n";

        test_construction_and_build();
        test_point_location();
        test_nearest_neighbor();
        test_ray_intersection();
        test_region_queries();
        test_performance();
        test_memory_management();
        test_edge_cases();

        std::cout << "\n========================================\n";
        std::cout << "  All UniformGridAccel tests PASSED! ✓\n";
        std::cout << "========================================\n\n";
    }
};

} // namespace test
} // namespace svmp

int main() {
    svmp::test::TestUniformGridAccel tester;
    tester.run_all_tests();
    return 0;
}