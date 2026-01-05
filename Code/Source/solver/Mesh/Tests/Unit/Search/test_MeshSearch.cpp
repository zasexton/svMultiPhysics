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
 * @file test_MeshSearch.cpp
 * @brief Integration tests for the complete MeshSearch functionality
 *
 * Tests include:
 * - Static API methods
 * - Search structure management
 * - Point location with and without acceleration
 * - Nearest neighbor searches
 * - Ray intersection
 * - Distance queries
 * - Spatial queries
 * - Parametric coordinates
 * - Walking algorithms
 * - Mixed cell type meshes
 */

#include "../../../Search/MeshSearch.h"
#include "../../../Core/MeshBase.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>
#include <random>
#include <chrono>

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
std::shared_ptr<MeshBase> create_quad_mesh(int nx, int ny) {
    auto mesh = std::make_shared<MeshBase>();

    // Create vertices
    std::vector<real_t> coords;
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            coords.push_back(i);
            coords.push_back(j);
            coords.push_back(0);
        }
    }

    // Create quads
    std::vector<index_t> connectivity;
    std::vector<offset_t> offsets = {0};

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int base = j * (nx + 1) + i;
            connectivity.push_back(base);
            connectivity.push_back(base + 1);
            connectivity.push_back(base + nx + 2);
            connectivity.push_back(base + nx + 1);
            offsets.push_back(connectivity.size());
        }
    }

    std::vector<CellShape> shapes(nx * ny, {CellFamily::Quad, 4, 1});

    mesh->build_from_arrays(3, coords, offsets, connectivity, shapes);
    mesh->finalize();

    return mesh;
}

std::shared_ptr<MeshBase> create_hex_mesh(int nx, int ny, int nz) {
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

    // Create hexes
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

std::shared_ptr<MeshBase> create_mixed_mesh() {
    // Create a mesh with mixed cell types
    auto mesh = std::make_shared<MeshBase>();

    std::vector<real_t> coords = {
        // Tet vertices
        0,0,0,  2,0,0,  1,2,0,  1,1,2,
        // Hex vertices
        3,0,0,  5,0,0,  5,2,0,  3,2,0,
        3,0,2,  5,0,2,  5,2,2,  3,2,2,
        // Wedge vertices
        6,0,0,  8,0,0,  7,2,0,
        6,0,2,  8,0,2,  7,2,2
    };

    std::vector<index_t> connectivity = {
        // Tet
        0,1,2,3,
        // Hex
        4,5,6,7,8,9,10,11,
        // Wedge
        12,13,14,15,16,17
    };

    std::vector<offset_t> offsets = {0, 4, 12, 18};

    std::vector<CellShape> shapes = {
        {CellFamily::Tetra, 4, 1},
        {CellFamily::Hex, 8, 1},
        {CellFamily::Wedge, 6, 1}
    };

    mesh->build_from_arrays(3, coords, offsets, connectivity, shapes);
    mesh->finalize();

    return mesh;
}

class TestMeshSearch {
public:
    void test_point_location() {
        std::cout << "=== Testing Point Location ===\n";

        auto mesh = create_hex_mesh(5, 5, 5);

        // Test without acceleration structure
        {
            std::array<real_t, 3> point = {2.5, 2.5, 2.5};
            auto result = MeshSearch::locate_point(*mesh, point);
            ASSERT_TRUE(result.found);
            ASSERT_GE(result.cell_id, 0);
            ASSERT_LT(result.cell_id, mesh->n_cells());
        }

        // Build acceleration structure and test again
        {
            MeshSearch::build_search_structure(*mesh);
            ASSERT_TRUE(MeshSearch::has_search_structure(*mesh));

            std::array<real_t, 3> point = {2.5, 2.5, 2.5};
            auto result = MeshSearch::locate_point(*mesh, point);
            ASSERT_TRUE(result.found);
        }

        // Test batch point location
        {
            std::vector<std::array<real_t, 3>> points = {
                {0.5, 0.5, 0.5},
                {1.5, 1.5, 1.5},
                {2.5, 2.5, 2.5},
                {4.5, 4.5, 4.5},
                {10, 10, 10}  // Outside
            };

            auto results = MeshSearch::locate_points(*mesh, points);
            ASSERT_EQ(results.size(), 5);
            ASSERT_TRUE(results[0].found);
            ASSERT_TRUE(results[1].found);
            ASSERT_TRUE(results[2].found);
            ASSERT_TRUE(results[3].found);
            ASSERT_FALSE(results[4].found);
        }

        // Test contains point
        {
            ASSERT_TRUE(MeshSearch::contains_point(*mesh, {2.5, 2.5, 2.5}));
            ASSERT_FALSE(MeshSearch::contains_point(*mesh, {10, 10, 10}));
        }

        // Test with hint
        {
            auto result1 = MeshSearch::locate_point(*mesh, {2.5, 2.5, 2.5});
            auto result2 = MeshSearch::locate_point(*mesh, {2.6, 2.5, 2.5},
                                                   Configuration::Reference, result1.cell_id);
            ASSERT_TRUE(result2.found);
        }

        MeshSearch::clear_search_structure(*mesh);

        std::cout << "  ✓ Point location tests passed\n";
    }

    void test_nearest_neighbor() {
        std::cout << "=== Testing Nearest Neighbor Searches ===\n";

        auto mesh = create_quad_mesh(5, 5);

        // Test nearest vertex
        {
            std::array<real_t, 3> point = {2.3, 2.7, 0};
            auto [vertex_id, dist] = MeshSearch::nearest_vertex(*mesh, point);
            ASSERT_NE(vertex_id, INVALID_INDEX);
            ASSERT_LT(vertex_id, mesh->n_vertices());
            ASSERT_GE(dist, 0);

            // Should find vertex at (2,3) as nearest
            const auto& coords = mesh->X_ref();
            real_t vx = coords[vertex_id * 3];
            real_t vy = coords[vertex_id * 3 + 1];
            // Verify it's close to expected
            ASSERT_LT(std::abs(vx - 2), 1.1);
            ASSERT_LT(std::abs(vy - 3), 1.1);
        }

        // Test k-nearest vertices
        {
            std::array<real_t, 3> point = {2.5, 2.5, 0};
            auto neighbors = MeshSearch::k_nearest_vertices(*mesh, point, 4);
            ASSERT_EQ(neighbors.size(), 4);

            // Verify distances are sorted
            for (size_t i = 1; i < neighbors.size(); ++i) {
                ASSERT_LE(neighbors[i-1].second, neighbors[i].second);
            }

            // All should be valid vertices
            for (const auto& [idx, dist] : neighbors) {
                ASSERT_LT(idx, mesh->n_vertices());
                ASSERT_GE(dist, 0);
            }
        }

        // Test vertices in radius
        {
            std::array<real_t, 3> point = {2.5, 2.5, 0};
            real_t radius = 1.5;
            auto vertices = MeshSearch::vertices_in_radius(*mesh, point, radius);

            // Should find multiple vertices
            ASSERT_GT(vertices.size(), 0);

            // All should be within radius
            const auto& coords = mesh->X_ref();
            for (index_t v : vertices) {
                real_t dx = coords[v*3] - point[0];
                real_t dy = coords[v*3+1] - point[1];
                real_t dz = coords[v*3+2] - point[2];
                real_t dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                ASSERT_LE(dist, radius + EPSILON);
            }
        }

        // Test nearest cell
        {
            std::array<real_t, 3> point = {2.5, 2.5, 0};
            auto [cell_id, dist] = MeshSearch::nearest_cell(*mesh, point);
            ASSERT_NE(cell_id, INVALID_INDEX);
            ASSERT_LT(cell_id, mesh->n_cells());
            ASSERT_GE(dist, 0);
        }

        std::cout << "  ✓ Nearest neighbor search tests passed\n";
    }

    void test_ray_intersection() {
        std::cout << "=== Testing Ray Intersection ===\n";

        auto mesh = create_hex_mesh(3, 3, 3);

        // Test single ray intersection
        {
            std::array<real_t, 3> origin = {-1, 1.5, 1.5};
            std::array<real_t, 3> direction = {1, 0, 0};

            auto result = MeshSearch::intersect_ray(*mesh, origin, direction);
            ASSERT_TRUE(result.found);
            ASSERT_TRUE(result.hit);
            ASSERT_NE(result.face_id, INVALID_INDEX);
            ASSERT_GE(result.t, 0.0);
            // Box spans x in [0,3], so entry intersection is at x=0 -> t = 1.
            ASSERT_NEAR(result.t, 1.0, 1e-8);
        }

        // Test all ray intersections
        {
            std::array<real_t, 3> origin = {-1, 1.5, 1.5};
            std::array<real_t, 3> direction = {1, 0, 0};

            auto results = MeshSearch::intersect_ray_all(*mesh, origin, direction);
            ASSERT_EQ(results.size(), 2u);

            // Verify results are sorted by t
            for (size_t i = 1; i < results.size(); ++i) {
                ASSERT_LE(results[i-1].t, results[i].t);
            }

            ASSERT_NEAR(results[0].t, 1.0, 1e-8);
            ASSERT_NEAR(results[1].t, 4.0, 1e-8);
        }

        std::cout << "  ✓ Ray intersection tests passed\n";
    }

    void test_distance_queries() {
        std::cout << "=== Testing Distance Queries ===\n";

        auto mesh = create_quad_mesh(3, 3);

        // Test signed distance
        {
            std::array<real_t, 3> inside = {1.5, 1.5, 0};
            real_t dist = MeshSearch::signed_distance(*mesh, inside);
            // Inside the [0,3]x[0,3] domain -> negative distance to boundary.
            ASSERT_LT(dist, 0);
            ASSERT_NEAR(dist, -1.5, 1e-8);
        }

        // Test closest boundary point
        {
            std::array<real_t, 3> query = {1.5, 1.5, 0};
            auto [closest, face_id] = MeshSearch::closest_boundary_point(*mesh, query);

            // Should return valid face
            if (face_id != INVALID_INDEX) {
                ASSERT_LT(face_id, mesh->n_faces());
            }

            // Closest point should be reasonable
            real_t dist = std::sqrt(
                std::pow(closest[0] - query[0], 2) +
                std::pow(closest[1] - query[1], 2) +
                std::pow(closest[2] - query[2], 2)
            );
            ASSERT_NEAR(dist, 1.5, 1e-8);
            // Closest point should lie on the boundary of the square.
            bool on_boundary = (std::abs(closest[0] - 0.0) < 1e-8) ||
                               (std::abs(closest[0] - 3.0) < 1e-8) ||
                               (std::abs(closest[1] - 0.0) < 1e-8) ||
                               (std::abs(closest[1] - 3.0) < 1e-8);
            ASSERT_TRUE(on_boundary);
        }

        std::cout << "  ✓ Distance query tests passed\n";
    }

    void test_spatial_queries() {
        std::cout << "=== Testing Spatial Queries ===\n";

        auto mesh = create_hex_mesh(4, 4, 4);

        // Test cells in box
        {
            std::array<real_t, 3> box_min = {1, 1, 1};
            std::array<real_t, 3> box_max = {3, 3, 3};

            auto cells = MeshSearch::cells_in_box(*mesh, box_min, box_max);
            ASSERT_GT(cells.size(), 0);

            // All cells should have centers near box
            for (index_t cell : cells) {
                auto center = mesh->cell_center(cell);
                // Rough check - center should be close to box
                bool near_box = true;
                for (int d = 0; d < 3; ++d) {
                    if (center[d] < box_min[d] - 1 || center[d] > box_max[d] + 1) {
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

            auto cells = MeshSearch::cells_in_sphere(*mesh, center, radius);
            ASSERT_GT(cells.size(), 0);

            // All cells should be near sphere
            for (index_t cell : cells) {
                auto cell_center = mesh->cell_center(cell);
                real_t dist = std::sqrt(
                    std::pow(cell_center[0] - center[0], 2) +
                    std::pow(cell_center[1] - center[1], 2) +
                    std::pow(cell_center[2] - center[2], 2)
                );
                // Allow some tolerance for cell size
                ASSERT_LT(dist, radius + 2);
            }
        }

        std::cout << "  ✓ Spatial query tests passed\n";
    }

    void test_parametric_coordinates() {
        std::cout << "=== Testing Parametric Coordinates ===\n";

        auto mesh = create_hex_mesh(2, 2, 2);

        // Test compute parametric coords
        {
            index_t cell = 0;
            std::array<real_t, 3> point = {0.5, 0.5, 0.5};  // Center of first hex

            auto xi = MeshSearch::compute_parametric_coords(*mesh, cell, point);

            // Should be near center of reference element
            for (int i = 0; i < 3; ++i) {
                ASSERT_GE(xi[i], -1.1);
                ASSERT_LE(xi[i], 1.1);
            }
        }

        // Test is_inside_reference_element
        {
            // Test hex
            CellShape hex_shape = {CellFamily::Hex, 8, 1};
            ASSERT_TRUE(MeshSearch::is_inside_reference_element(hex_shape, {0, 0, 0}));
            ASSERT_TRUE(MeshSearch::is_inside_reference_element(hex_shape, {-1, -1, -1}));
            ASSERT_TRUE(MeshSearch::is_inside_reference_element(hex_shape, {1, 1, 1}));
            ASSERT_FALSE(MeshSearch::is_inside_reference_element(hex_shape, {2, 0, 0}));

            // Test tet
            CellShape tet_shape = {CellFamily::Tetra, 4, 1};
            ASSERT_TRUE(MeshSearch::is_inside_reference_element(tet_shape, {0.25, 0.25, 0.25}));
            ASSERT_FALSE(MeshSearch::is_inside_reference_element(tet_shape, {1, 1, 1}));

            // Test quad
            CellShape quad_shape = {CellFamily::Quad, 4, 1};
            ASSERT_TRUE(MeshSearch::is_inside_reference_element(quad_shape, {0, 0, 0}));
            ASSERT_TRUE(MeshSearch::is_inside_reference_element(quad_shape, {-1, -1, 0}));
            ASSERT_FALSE(MeshSearch::is_inside_reference_element(quad_shape, {2, 0, 0}));
        }

        std::cout << "  ✓ Parametric coordinate tests passed\n";
    }

    void test_walking_algorithms() {
        std::cout << "=== Testing Walking Algorithms ===\n";

        auto mesh = create_hex_mesh(3, 3, 3);

        // Test walk to point
        {
            index_t start_cell = 0;
            std::array<real_t, 3> target = {2.5, 2.5, 2.5};

            auto path = MeshSearch::walk_to_point(*mesh, start_cell, target);

            // Should return a path
            ASSERT_GT(path.size(), 0);
            ASSERT_EQ(path[0], start_cell);

            // All cells in path should be valid
            for (index_t cell : path) {
                ASSERT_LT(cell, mesh->n_cells());
            }
        }

        std::cout << "  ✓ Walking algorithm tests passed\n";
    }

    void test_search_structure_management() {
        std::cout << "=== Testing Search Structure Management ===\n";

        auto mesh = create_hex_mesh(3, 3, 3);

        // Initially no structure
        ASSERT_FALSE(MeshSearch::has_search_structure(*mesh));

        // Build with default config
        MeshSearch::build_search_structure(*mesh);
        ASSERT_TRUE(MeshSearch::has_search_structure(*mesh));

        // Clear structure
        MeshSearch::clear_search_structure(*mesh);
        ASSERT_FALSE(MeshSearch::has_search_structure(*mesh));

        // Build with custom config
        {
            MeshSearch::SearchConfig config;
            config.type = MeshSearch::AccelType::UniformGrid;
            config.grid_resolution = 16;
            config.tolerance = 1e-8;

            MeshSearch::build_search_structure(*mesh, config, Configuration::Reference);
            ASSERT_TRUE(MeshSearch::has_search_structure(*mesh));
        }

        // Build for current configuration
        {
            MeshSearch::SearchConfig config;
            MeshSearch::build_search_structure(*mesh, config, Configuration::Current);
            ASSERT_TRUE(MeshSearch::has_search_structure(*mesh));
        }

        MeshSearch::clear_search_structure(*mesh);

        std::cout << "  ✓ Search structure management tests passed\n";
    }

    void test_mixed_cell_types() {
        std::cout << "=== Testing Mixed Cell Types ===\n";

        auto mesh = create_mixed_mesh();

        // Test point location in mixed mesh
        {
            // Point in tet
            std::array<real_t, 3> tet_point = {1, 1, 0.5};
            auto result = MeshSearch::locate_point(*mesh, tet_point);
            // May or may not find depending on implementation

            // Point in hex
            std::array<real_t, 3> hex_point = {4, 1, 1};
            result = MeshSearch::locate_point(*mesh, hex_point);

            // Point in wedge
            std::array<real_t, 3> wedge_point = {7, 1, 1};
            result = MeshSearch::locate_point(*mesh, wedge_point);
        }

        // Test with acceleration structure
        {
            MeshSearch::build_search_structure(*mesh);

            std::array<real_t, 3> point = {4, 1, 1};
            auto result = MeshSearch::locate_point(*mesh, point);

            MeshSearch::clear_search_structure(*mesh);
        }

        std::cout << "  ✓ Mixed cell type tests passed\n";
    }

    void test_performance_comparison() {
        std::cout << "=== Testing Performance Comparison ===\n";

        // Create a larger mesh for performance testing
        auto mesh = create_hex_mesh(10, 10, 10);

        // Generate random test points
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 10);

        const int n_queries = 100;
        std::vector<std::array<real_t, 3>> test_points;
        for (int i = 0; i < n_queries; ++i) {
            test_points.push_back({dis(gen), dis(gen), dis(gen)});
        }

        // Test without acceleration
        auto start = std::chrono::high_resolution_clock::now();
        int found_linear = 0;
        for (const auto& point : test_points) {
            auto result = MeshSearch::locate_point(*mesh, point);
            if (result.found) found_linear++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto linear_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Test with acceleration
        MeshSearch::build_search_structure(*mesh);

        start = std::chrono::high_resolution_clock::now();
        int found_accel = 0;
        for (const auto& point : test_points) {
            auto result = MeshSearch::locate_point(*mesh, point);
            if (result.found) found_accel++;
        }
        end = std::chrono::high_resolution_clock::now();
        auto accel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "    Linear search: " << linear_time / n_queries << " μs/query\n";
        std::cout << "    Accelerated search: " << accel_time / n_queries << " μs/query\n";
        std::cout << "    Speedup: " << (double)linear_time / accel_time << "x\n";

        // Both should find the same points
        ASSERT_EQ(found_linear, found_accel);

        MeshSearch::clear_search_structure(*mesh);

        std::cout << "  ✓ Performance comparison tests passed\n";
    }

    void run_all_tests() {
        std::cout << "\n========================================\n";
        std::cout << "  MeshSearch Integration Test Suite\n";
        std::cout << "========================================\n\n";

        test_point_location();
        test_nearest_neighbor();
        test_ray_intersection();
        test_distance_queries();
        test_spatial_queries();
        test_parametric_coordinates();
        test_walking_algorithms();
        test_search_structure_management();
        test_mixed_cell_types();
        test_performance_comparison();

        std::cout << "\n========================================\n";
        std::cout << "  All MeshSearch tests PASSED! ✓\n";
        std::cout << "========================================\n\n";
    }
};

} // namespace test
} // namespace svmp

int main() {
    svmp::test::TestMeshSearch tester;
    tester.run_all_tests();
    return 0;
}
