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
 * @file test_SearchBuilders.cpp
 * @brief Unit tests for mesh data extraction utilities
 *
 * Tests include:
 * - Coordinate extraction
 * - AABB computation
 * - Boundary extraction
 * - Face triangulation
 * - Neighbor information
 * - Parametric coordinates
 * - Mesh statistics
 */

#include "../../../Search/SearchBuilders.h"
#include "../../../Core/MeshBase.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <algorithm>

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

// Helper: Create simple test meshes
std::shared_ptr<MeshBase> create_test_quad_mesh() {
    // Create a 2x2 quad mesh
    auto mesh = std::make_shared<MeshBase>();

    // Create 9 vertices (3x3 grid)
    std::vector<real_t> coords = {
        0,0,0,  1,0,0,  2,0,0,
        0,1,0,  1,1,0,  2,1,0,
        0,2,0,  1,2,0,  2,2,0
    };

    // Create 4 quads
    std::vector<index_t> connectivity = {
        0,1,4,3,  // quad 0
        1,2,5,4,  // quad 1
        3,4,7,6,  // quad 2
        4,5,8,7   // quad 3
    };

    std::vector<offset_t> offsets = {0, 4, 8, 12, 16};

    std::vector<CellShape> shapes(4, {CellFamily::Quad, 4, 1});

    mesh->build_from_arrays(3, coords, offsets, connectivity, shapes);
    mesh->finalize();

    return mesh;
}

std::shared_ptr<MeshBase> create_test_tet_mesh() {
    // Create a simple tetrahedron
    auto mesh = std::make_shared<MeshBase>();

    std::vector<real_t> coords = {
        0,0,0,  1,0,0,  0,1,0,  0,0,1
    };

    std::vector<index_t> connectivity = {0,1,2,3};
    std::vector<offset_t> offsets = {0, 4};
    std::vector<CellShape> shapes = {{CellFamily::Tetra, 4, 1}};

    mesh->build_from_arrays(3, coords, offsets, connectivity, shapes);
    mesh->finalize();

    return mesh;
}

std::shared_ptr<MeshBase> create_test_hex_mesh() {
    // Create a 2x2x2 hex mesh
    auto mesh = std::make_shared<MeshBase>();

    // Create 27 vertices (3x3x3 grid)
    std::vector<real_t> coords;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                coords.push_back(i);
                coords.push_back(j);
                coords.push_back(k);
            }
        }
    }

    // Create 8 hexes
    std::vector<index_t> connectivity;
    std::vector<offset_t> offsets = {0};

    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                int base = k * 9 + j * 3 + i;
                connectivity.push_back(base);
                connectivity.push_back(base + 1);
                connectivity.push_back(base + 4);
                connectivity.push_back(base + 3);
                connectivity.push_back(base + 9);
                connectivity.push_back(base + 10);
                connectivity.push_back(base + 13);
                connectivity.push_back(base + 12);
                offsets.push_back(connectivity.size());
            }
        }
    }

    std::vector<CellShape> shapes(8, {CellFamily::Hex, 8, 1});

    mesh->build_from_arrays(3, coords, offsets, connectivity, shapes);
    mesh->finalize();

    return mesh;
}

class TestSearchBuilders {
public:
    void test_coordinate_extraction() {
        std::cout << "=== Testing Coordinate Extraction ===\n";

        auto mesh = create_test_quad_mesh();

        // Test extract all vertex coords
        {
            auto coords = search::SearchBuilders::extract_vertex_coords(*mesh, Configuration::Reference);
            ASSERT_EQ(coords.size(), 9);

            // Check first vertex
            ASSERT_EQ(coords[0][0], 0);
            ASSERT_EQ(coords[0][1], 0);
            ASSERT_EQ(coords[0][2], 0);

            // Check last vertex
            ASSERT_EQ(coords[8][0], 2);
            ASSERT_EQ(coords[8][1], 2);
            ASSERT_EQ(coords[8][2], 0);
        }

        // Test get single vertex coord
        {
            auto coord = search::SearchBuilders::get_vertex_coord(*mesh, 4, Configuration::Reference);
            ASSERT_EQ(coord[0], 1);
            ASSERT_EQ(coord[1], 1);
            ASSERT_EQ(coord[2], 0);
        }

        // Test get cell vertex coords
        {
            auto coords = search::SearchBuilders::get_cell_vertex_coords(*mesh, 0, Configuration::Reference);
            ASSERT_EQ(coords.size(), 4);

            // First vertex of first quad
            ASSERT_EQ(coords[0][0], 0);
            ASSERT_EQ(coords[0][1], 0);

            // Last vertex of first quad
            ASSERT_EQ(coords[3][0], 0);
            ASSERT_EQ(coords[3][1], 1);
        }

        // Test get face vertex coords
        {
            if (mesh->n_faces() > 0) {
                auto coords = search::SearchBuilders::get_face_vertex_coords(*mesh, 0, Configuration::Reference);
                ASSERT_GE(coords.size(), 2);  // At least an edge
            }
        }

        std::cout << "  ✓ Coordinate extraction tests passed\n";
    }

    void test_aabb_computation() {
        std::cout << "=== Testing AABB Computation ===\n";

        auto mesh = create_test_quad_mesh();

        // Test mesh AABB
        {
            auto aabb = search::SearchBuilders::compute_mesh_aabb(*mesh, Configuration::Reference);
            ASSERT_EQ(aabb.min[0], 0);
            ASSERT_EQ(aabb.min[1], 0);
            ASSERT_EQ(aabb.min[2], 0);
            ASSERT_EQ(aabb.max[0], 2);
            ASSERT_EQ(aabb.max[1], 2);
            ASSERT_EQ(aabb.max[2], 0);
        }

        // Test cell AABB
        {
            auto aabb = search::SearchBuilders::compute_cell_aabb(*mesh, 0, Configuration::Reference);
            ASSERT_EQ(aabb.min[0], 0);
            ASSERT_EQ(aabb.min[1], 0);
            ASSERT_EQ(aabb.max[0], 1);
            ASSERT_EQ(aabb.max[1], 1);
        }

        // Test all cell AABBs
        {
            auto aabbs = search::SearchBuilders::compute_all_cell_aabbs(*mesh, Configuration::Reference);
            ASSERT_EQ(aabbs.size(), 4);

            // First cell
            ASSERT_EQ(aabbs[0].min[0], 0);
            ASSERT_EQ(aabbs[0].max[0], 1);

            // Last cell
            ASSERT_EQ(aabbs[3].min[0], 1);
            ASSERT_EQ(aabbs[3].max[0], 2);
        }

        // Test with hex mesh
        {
            auto hex_mesh = create_test_hex_mesh();
            auto aabb = search::SearchBuilders::compute_mesh_aabb(*hex_mesh, Configuration::Reference);
            ASSERT_EQ(aabb.min[0], 0);
            ASSERT_EQ(aabb.min[1], 0);
            ASSERT_EQ(aabb.min[2], 0);
            ASSERT_EQ(aabb.max[0], 2);
            ASSERT_EQ(aabb.max[1], 2);
            ASSERT_EQ(aabb.max[2], 2);
        }

        std::cout << "  ✓ AABB computation tests passed\n";
    }

    void test_boundary_extraction() {
        std::cout << "=== Testing Boundary Extraction ===\n";

        auto mesh = create_test_quad_mesh();

        // Test boundary face detection
        {
            auto boundary_faces = search::SearchBuilders::get_boundary_faces(*mesh);

            // Count actual boundary faces
            size_t boundary_count = 0;
            for (size_t f = 0; f < mesh->n_faces(); ++f) {
                if (search::SearchBuilders::is_boundary_face(*mesh, f)) {
                    boundary_count++;
                }
            }
            ASSERT_EQ(boundary_faces.size(), boundary_count);
        }

        // Test is_boundary_face
        {
            if (mesh->n_faces() > 0) {
                // Find a boundary face
                bool found_boundary = false;
                for (size_t f = 0; f < mesh->n_faces(); ++f) {
                    auto face_cells = mesh->face_cells(f);
                    if (face_cells[1] == INVALID_INDEX) {
                        ASSERT_TRUE(search::SearchBuilders::is_boundary_face(*mesh, f));
                        found_boundary = true;
                        break;
                    }
                }
                // For 2D quad mesh, we should have boundary faces
                if (mesh->dim() == 3 || mesh->n_cells() > 1) {
                    ASSERT_TRUE(found_boundary);
                }
            }
        }

        std::cout << "  ✓ Boundary extraction tests passed\n";
    }

    void test_face_triangulation() {
        std::cout << "=== Testing Face Triangulation ===\n";

        auto mesh = create_test_quad_mesh();

        // Test triangulating a quad face
        if (mesh->n_faces() > 0) {
            // Find a quad face (4 vertices)
            for (size_t f = 0; f < mesh->n_faces(); ++f) {
                auto vertices = mesh->face_vertices(f);
                if (vertices.size() == 4) {
                    auto triangles = search::SearchBuilders::triangulate_face(*mesh, f, Configuration::Reference);
                    ASSERT_EQ(triangles.size(), 2);  // Quad splits into 2 triangles

                    // Each triangle should have 3 vertices
                    for (const auto& tri : triangles) {
                        // Verify triangle has valid coordinates
                        for (int v = 0; v < 3; ++v) {
                            ASSERT_FALSE(std::isnan(tri[v][0]));
                            ASSERT_FALSE(std::isnan(tri[v][1]));
                            ASSERT_FALSE(std::isnan(tri[v][2]));
                        }
                    }
                    break;
                }
            }
        }

        // Test boundary triangulation
        {
            auto triangles = search::SearchBuilders::triangulate_boundary(*mesh, Configuration::Reference);

            // Should have triangles for boundary faces
            // Each quad boundary face becomes 2 triangles
            ASSERT_GT(triangles.size(), 0);

            for (const auto& tri : triangles) {
                // Each triangle should have valid face ID
                ASSERT_LT(tri.face_id, mesh->n_faces());

                // Vertices should be valid
                for (int v = 0; v < 3; ++v) {
                    ASSERT_FALSE(std::isnan(tri.vertices[v][0]));
                    ASSERT_FALSE(std::isnan(tri.vertices[v][1]));
                    ASSERT_FALSE(std::isnan(tri.vertices[v][2]));
                }
            }
        }

        std::cout << "  ✓ Face triangulation tests passed\n";
    }

    void test_cell_centers() {
        std::cout << "=== Testing Cell Center Computation ===\n";

        // Test quad mesh
        {
            auto mesh = create_test_quad_mesh();

            // Test single cell center
            auto center = search::SearchBuilders::compute_cell_center(*mesh, 0, Configuration::Reference);
            ASSERT_NEAR(center[0], 0.5, EPSILON);
            ASSERT_NEAR(center[1], 0.5, EPSILON);
            ASSERT_NEAR(center[2], 0, EPSILON);

            // Test all cell centers
            auto centers = search::SearchBuilders::compute_all_cell_centers(*mesh, Configuration::Reference);
            ASSERT_EQ(centers.size(), 4);

            // Check centers are in expected positions
            ASSERT_NEAR(centers[0][0], 0.5, EPSILON);  // First quad center
            ASSERT_NEAR(centers[0][1], 0.5, EPSILON);

            ASSERT_NEAR(centers[3][0], 1.5, EPSILON);  // Last quad center
            ASSERT_NEAR(centers[3][1], 1.5, EPSILON);
        }

        // Test tet mesh
        {
            auto mesh = create_test_tet_mesh();
            auto center = search::SearchBuilders::compute_cell_center(*mesh, 0, Configuration::Reference);

            // Center of tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1)
            ASSERT_NEAR(center[0], 0.25, EPSILON);
            ASSERT_NEAR(center[1], 0.25, EPSILON);
            ASSERT_NEAR(center[2], 0.25, EPSILON);
        }

        std::cout << "  ✓ Cell center computation tests passed\n";
    }

    void test_neighbor_information() {
        std::cout << "=== Testing Neighbor Information ===\n";

        auto mesh = create_test_quad_mesh();

        // Test cell neighbors
        {
            auto neighbors = search::SearchBuilders::get_cell_neighbors(*mesh, 0);

            // First quad should have neighbors (at least quad 1 and quad 2)
            ASSERT_GT(neighbors.size(), 0);

            // Check that neighbors are valid cells
            for (index_t neighbor : neighbors) {
                ASSERT_LT(neighbor, mesh->n_cells());
            }
        }

        // Test vertex cells
        {
            // Central vertex (index 4) should be shared by all 4 quads
            auto cells = search::SearchBuilders::get_vertex_cells(*mesh, 4);
            ASSERT_EQ(cells.size(), 4);

            // All cells should be valid
            for (index_t cell : cells) {
                ASSERT_LT(cell, mesh->n_cells());
            }
        }

        // Test edge cells
        {
            // Edge between vertices 1 and 4 should be shared by quads 0 and 1
            auto cells = search::SearchBuilders::get_edge_cells(*mesh, 1, 4);
            ASSERT_GE(cells.size(), 2);

            for (index_t cell : cells) {
                ASSERT_LT(cell, mesh->n_cells());
            }
        }

        std::cout << "  ✓ Neighbor information tests passed\n";
    }

    void test_parametric_coords() {
        std::cout << "=== Testing Parametric Coordinates ===\n";

        // Test tetrahedron parametric coords
        {
            std::vector<std::array<real_t,3>> tet_verts = {
                {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
            };

            // Test at vertices
            auto xi = search::SearchBuilders::tetra_parametric_coords({0,0,0}, tet_verts);
            ASSERT_NEAR(xi[0], 0, EPSILON);
            ASSERT_NEAR(xi[1], 0, EPSILON);
            ASSERT_NEAR(xi[2], 0, EPSILON);

            xi = search::SearchBuilders::tetra_parametric_coords({1,0,0}, tet_verts);
            ASSERT_NEAR(xi[0], 1, EPSILON);
            ASSERT_NEAR(xi[1], 0, EPSILON);
            ASSERT_NEAR(xi[2], 0, EPSILON);

            // Test at center
            xi = search::SearchBuilders::tetra_parametric_coords({0.25,0.25,0.25}, tet_verts);
            ASSERT_NEAR(xi[0], 0.25, EPSILON);
            ASSERT_NEAR(xi[1], 0.25, EPSILON);
            ASSERT_NEAR(xi[2], 0.25, EPSILON);
        }

        // Test hexahedron parametric coords
        {
            std::vector<std::array<real_t,3>> hex_verts = {
                {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},  // bottom
                {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}   // top
            };

            // Test at center
            auto xi = search::SearchBuilders::hex_parametric_coords({0.5,0.5,0.5}, hex_verts);
            ASSERT_NEAR(xi[0], 0, 0.1);  // Should be near center in reference coords
            ASSERT_NEAR(xi[1], 0, 0.1);
            ASSERT_NEAR(xi[2], 0, 0.1);

            // Test at a vertex
            xi = search::SearchBuilders::hex_parametric_coords({0,0,0}, hex_verts);
            ASSERT_NEAR(xi[0], -1, 0.1);  // (-1,-1,-1) in reference hex
            ASSERT_NEAR(xi[1], -1, 0.1);
            ASSERT_NEAR(xi[2], -1, 0.1);
        }

        // Test with actual mesh
        {
            auto mesh = create_test_tet_mesh();
            auto xi = search::SearchBuilders::compute_parametric_coords(*mesh, 0, {0.25,0.25,0.25}, Configuration::Reference);

            // Should return reasonable parametric coords
            for (int i = 0; i < 3; ++i) {
                ASSERT_GE(xi[i], -0.1);
                ASSERT_LE(xi[i], 1.1);
            }
        }

        std::cout << "  ✓ Parametric coordinate tests passed\n";
    }

    void test_mesh_statistics() {
        std::cout << "=== Testing Mesh Statistics ===\n";

        auto mesh = create_test_quad_mesh();

        // Test characteristic length
        {
            real_t char_len = search::SearchBuilders::compute_mesh_characteristic_length(*mesh, Configuration::Reference);
            // For a regular grid, characteristic length should be around 1
            ASSERT_GT(char_len, 0);
            ASSERT_LT(char_len, 10);
        }

        // Test grid resolution estimation
        {
            int resolution = search::SearchBuilders::estimate_grid_resolution(*mesh, Configuration::Reference, 1);
            // For 4 cells with target of 1 cell per bucket, we expect small resolution
            ASSERT_GE(resolution, 2);
            ASSERT_LE(resolution, 128);
        }

        // Test linear search threshold
        {
            // Small mesh should use linear search
            ASSERT_TRUE(search::SearchBuilders::use_linear_search(*mesh, 100));

            // Large threshold
            ASSERT_TRUE(search::SearchBuilders::use_linear_search(*mesh, 10));

            // Very small threshold
            ASSERT_FALSE(search::SearchBuilders::use_linear_search(*mesh, 1));
        }

        // Test with larger mesh
        {
            auto hex_mesh = create_test_hex_mesh();
            real_t char_len = search::SearchBuilders::compute_mesh_characteristic_length(*hex_mesh, Configuration::Reference);
            ASSERT_GT(char_len, 0);

            int resolution = search::SearchBuilders::estimate_grid_resolution(*hex_mesh, Configuration::Reference);
            ASSERT_GE(resolution, 2);
        }

        std::cout << "  ✓ Mesh statistics tests passed\n";
    }

    void run_all_tests() {
        std::cout << "\n========================================\n";
        std::cout << "  SearchBuilders Unit Test Suite\n";
        std::cout << "========================================\n\n";

        test_coordinate_extraction();
        test_aabb_computation();
        test_boundary_extraction();
        test_face_triangulation();
        test_cell_centers();
        test_neighbor_information();
        test_parametric_coords();
        test_mesh_statistics();

        std::cout << "\n========================================\n";
        std::cout << "  All SearchBuilders tests PASSED! ✓\n";
        std::cout << "========================================\n\n";
    }
};

} // namespace test
} // namespace svmp

int main() {
    svmp::test::TestSearchBuilders tester;
    tester.run_all_tests();
    return 0;
}