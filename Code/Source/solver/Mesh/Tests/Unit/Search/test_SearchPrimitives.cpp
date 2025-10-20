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
 * @file test_SearchPrimitives.cpp
 * @brief Unit tests for geometric primitives used in spatial search
 *
 * Tests include:
 * - AABB operations
 * - Ray operations
 * - Point-in-shape tests
 * - Ray-shape intersection
 * - Distance computations
 * - Barycentric coordinates
 * - Vector operations
 */

#include "../../../Search/SearchPrimitives.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <array>

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

class TestSearchPrimitives {
public:
    void test_aabb() {
        std::cout << "=== Testing AABB ===\n";

        // Test construction
        {
            search::AABB aabb;
            ASSERT_GT(aabb.min[0], 0);  // Should be initialized to large value
            ASSERT_LT(aabb.max[0], 0);  // Should be initialized to small value

            search::AABB aabb2({1, 2, 3});
            ASSERT_EQ(aabb2.min[0], 1);
            ASSERT_EQ(aabb2.min[1], 2);
            ASSERT_EQ(aabb2.min[2], 3);
            ASSERT_EQ(aabb2.max[0], 1);
            ASSERT_EQ(aabb2.max[1], 2);
            ASSERT_EQ(aabb2.max[2], 3);

            search::AABB aabb3({0, 0, 0}, {1, 1, 1});
            ASSERT_EQ(aabb3.min[0], 0);
            ASSERT_EQ(aabb3.max[0], 1);
        }

        // Test include point
        {
            search::AABB aabb;
            aabb.include({0, 0, 0});
            aabb.include({1, 2, 3});

            ASSERT_EQ(aabb.min[0], 0);
            ASSERT_EQ(aabb.min[1], 0);
            ASSERT_EQ(aabb.min[2], 0);
            ASSERT_EQ(aabb.max[0], 1);
            ASSERT_EQ(aabb.max[1], 2);
            ASSERT_EQ(aabb.max[2], 3);
        }

        // Test include AABB
        {
            search::AABB aabb1({0, 0, 0}, {1, 1, 1});
            search::AABB aabb2({0.5, 0.5, 0.5}, {2, 2, 2});
            aabb1.include(aabb2);

            ASSERT_EQ(aabb1.min[0], 0);
            ASSERT_EQ(aabb1.max[0], 2);
        }

        // Test overlaps
        {
            search::AABB aabb1({0, 0, 0}, {1, 1, 1});
            search::AABB aabb2({0.5, 0.5, 0.5}, {2, 2, 2});
            search::AABB aabb3({2, 2, 2}, {3, 3, 3});

            ASSERT_TRUE(aabb1.overlaps(aabb2));
            ASSERT_TRUE(aabb2.overlaps(aabb1));
            ASSERT_FALSE(aabb1.overlaps(aabb3));
        }

        // Test contains
        {
            search::AABB aabb({0, 0, 0}, {2, 2, 2});
            ASSERT_TRUE(aabb.contains({1, 1, 1}));
            ASSERT_TRUE(aabb.contains({0, 0, 0}));
            ASSERT_TRUE(aabb.contains({2, 2, 2}));
            ASSERT_FALSE(aabb.contains({3, 1, 1}));
        }

        // Test center and extents
        {
            search::AABB aabb({0, 0, 0}, {2, 4, 6});
            auto center = aabb.center();
            ASSERT_EQ(center[0], 1);
            ASSERT_EQ(center[1], 2);
            ASSERT_EQ(center[2], 3);

            auto extents = aabb.extents();
            ASSERT_EQ(extents[0], 2);
            ASSERT_EQ(extents[1], 4);
            ASSERT_EQ(extents[2], 6);
        }

        // Test volume and surface area
        {
            search::AABB aabb({0, 0, 0}, {2, 3, 4});
            ASSERT_EQ(aabb.volume(), 24);  // 2*3*4
            ASSERT_EQ(aabb.surface_area(), 52);  // 2*(2*3 + 3*4 + 4*2)
        }

        std::cout << "  ✓ AABB tests passed\n";
    }

    void test_ray() {
        std::cout << "=== Testing Ray ===\n";

        // Test construction and point_at
        {
            search::Ray ray({0, 0, 0}, {1, 0, 0});
            auto p = ray.point_at(2.5);
            ASSERT_EQ(p[0], 2.5);
            ASSERT_EQ(p[1], 0);
            ASSERT_EQ(p[2], 0);
        }

        std::cout << "  ✓ Ray tests passed\n";
    }

    void test_point_in_triangle() {
        std::cout << "=== Testing Point in Triangle ===\n";

        std::array<real_t, 3> a = {0, 0, 0};
        std::array<real_t, 3> b = {1, 0, 0};
        std::array<real_t, 3> c = {0, 1, 0};

        // Test inside
        ASSERT_TRUE(search::point_in_triangle({0.25, 0.25, 0}, a, b, c));

        // Test vertices
        ASSERT_TRUE(search::point_in_triangle(a, a, b, c));
        ASSERT_TRUE(search::point_in_triangle(b, a, b, c));
        ASSERT_TRUE(search::point_in_triangle(c, a, b, c));

        // Test edges
        ASSERT_TRUE(search::point_in_triangle({0.5, 0, 0}, a, b, c));
        ASSERT_TRUE(search::point_in_triangle({0, 0.5, 0}, a, b, c));
        ASSERT_TRUE(search::point_in_triangle({0.5, 0.5, 0}, a, b, c));

        // Test outside
        ASSERT_FALSE(search::point_in_triangle({-0.1, 0.5, 0}, a, b, c));
        ASSERT_FALSE(search::point_in_triangle({0.5, -0.1, 0}, a, b, c));
        ASSERT_FALSE(search::point_in_triangle({1, 1, 0}, a, b, c));

        std::cout << "  ✓ Point in triangle tests passed\n";
    }

    void test_point_in_tetrahedron() {
        std::cout << "=== Testing Point in Tetrahedron ===\n";

        std::array<real_t, 3> a = {0, 0, 0};
        std::array<real_t, 3> b = {1, 0, 0};
        std::array<real_t, 3> c = {0, 1, 0};
        std::array<real_t, 3> d = {0, 0, 1};

        // Test inside
        ASSERT_TRUE(search::point_in_tetrahedron({0.25, 0.25, 0.25}, a, b, c, d));

        // Test vertices
        ASSERT_TRUE(search::point_in_tetrahedron(a, a, b, c, d));
        ASSERT_TRUE(search::point_in_tetrahedron(b, a, b, c, d));
        ASSERT_TRUE(search::point_in_tetrahedron(c, a, b, c, d));
        ASSERT_TRUE(search::point_in_tetrahedron(d, a, b, c, d));

        // Test outside
        ASSERT_FALSE(search::point_in_tetrahedron({1, 1, 1}, a, b, c, d));
        ASSERT_FALSE(search::point_in_tetrahedron({-0.1, 0.25, 0.25}, a, b, c, d));

        std::cout << "  ✓ Point in tetrahedron tests passed\n";
    }

    void test_point_in_hex() {
        std::cout << "=== Testing Point in Hexahedron ===\n";

        // Unit cube vertices
        std::vector<std::array<real_t, 3>> vertices = {
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},  // bottom
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}   // top
        };

        // Test inside
        ASSERT_TRUE(search::point_in_hexahedron({0.5, 0.5, 0.5}, vertices));

        // Test vertices
        for (const auto& v : vertices) {
            ASSERT_TRUE(search::point_in_hexahedron(v, vertices));
        }

        // Test outside
        ASSERT_FALSE(search::point_in_hexahedron({1.1, 0.5, 0.5}, vertices));
        ASSERT_FALSE(search::point_in_hexahedron({0.5, 1.1, 0.5}, vertices));
        ASSERT_FALSE(search::point_in_hexahedron({0.5, 0.5, 1.1}, vertices));
        ASSERT_FALSE(search::point_in_hexahedron({-0.1, 0.5, 0.5}, vertices));

        std::cout << "  ✓ Point in hexahedron tests passed\n";
    }

    void test_ray_triangle_intersection() {
        std::cout << "=== Testing Ray-Triangle Intersection ===\n";

        std::array<real_t, 3> a = {0, 0, 0};
        std::array<real_t, 3> b = {1, 0, 0};
        std::array<real_t, 3> c = {0, 1, 0};

        // Test hit from above
        {
            search::Ray ray({0.25, 0.25, 1}, {0, 0, -1});
            real_t t;
            ASSERT_TRUE(search::ray_triangle_intersect(ray, a, b, c, t));
            ASSERT_NEAR(t, 1.0, EPSILON);
        }

        // Test hit with UV coords
        {
            search::Ray ray({0.25, 0.25, 1}, {0, 0, -1});
            real_t t;
            std::array<real_t, 2> uv;
            ASSERT_TRUE(search::ray_triangle_intersect(ray, a, b, c, t, &uv));
            ASSERT_NEAR(uv[0], 0.25, EPSILON);
            ASSERT_NEAR(uv[1], 0.25, EPSILON);
        }

        // Test miss
        {
            search::Ray ray({2, 2, 1}, {0, 0, -1});
            real_t t;
            ASSERT_FALSE(search::ray_triangle_intersect(ray, a, b, c, t));
        }

        // Test parallel ray
        {
            search::Ray ray({0.5, 0.5, 0}, {1, 0, 0});
            real_t t;
            ASSERT_FALSE(search::ray_triangle_intersect(ray, a, b, c, t));
        }

        std::cout << "  ✓ Ray-triangle intersection tests passed\n";
    }

    void test_ray_aabb_intersection() {
        std::cout << "=== Testing Ray-AABB Intersection ===\n";

        search::AABB aabb({0, 0, 0}, {1, 1, 1});

        // Test hit through center
        {
            search::Ray ray({-1, 0.5, 0.5}, {1, 0, 0});
            real_t t_near, t_far;
            ASSERT_TRUE(search::ray_aabb_intersect(ray, aabb, t_near, t_far));
            ASSERT_NEAR(t_near, 1.0, EPSILON);
            ASSERT_NEAR(t_far, 2.0, EPSILON);
        }

        // Test ray starting inside
        {
            search::Ray ray({0.5, 0.5, 0.5}, {1, 0, 0});
            real_t t_near, t_far;
            ASSERT_TRUE(search::ray_aabb_intersect(ray, aabb, t_near, t_far));
            ASSERT_LE(t_near, 0);
            ASSERT_NEAR(t_far, 0.5, EPSILON);
        }

        // Test miss
        {
            search::Ray ray({-1, 2, 0.5}, {1, 0, 0});
            real_t t_near, t_far;
            ASSERT_FALSE(search::ray_aabb_intersect(ray, aabb, t_near, t_far));
        }

        std::cout << "  ✓ Ray-AABB intersection tests passed\n";
    }

    void test_distance_functions() {
        std::cout << "=== Testing Distance Functions ===\n";

        // Test point to segment distance
        {
            std::array<real_t, 3> a = {0, 0, 0};
            std::array<real_t, 3> b = {1, 0, 0};

            // Point on segment
            ASSERT_NEAR(search::point_segment_distance({0.5, 0, 0}, a, b), 0, EPSILON);

            // Point perpendicular to segment
            ASSERT_NEAR(search::point_segment_distance({0.5, 1, 0}, a, b), 1, EPSILON);

            // Point beyond endpoints
            ASSERT_NEAR(search::point_segment_distance({2, 0, 0}, a, b), 1, EPSILON);
            ASSERT_NEAR(search::point_segment_distance({-1, 0, 0}, a, b), 1, EPSILON);

            // Test closest point output
            std::array<real_t, 3> closest;
            search::point_segment_distance({0.5, 1, 0}, a, b, &closest);
            ASSERT_NEAR(closest[0], 0.5, EPSILON);
            ASSERT_NEAR(closest[1], 0, EPSILON);
        }

        // Test point to triangle distance
        {
            std::array<real_t, 3> a = {0, 0, 0};
            std::array<real_t, 3> b = {1, 0, 0};
            std::array<real_t, 3> c = {0, 1, 0};

            // Point on triangle
            ASSERT_NEAR(search::point_triangle_distance({0.25, 0.25, 0}, a, b, c), 0, EPSILON);

            // Point above triangle
            ASSERT_NEAR(search::point_triangle_distance({0.25, 0.25, 1}, a, b, c), 1, EPSILON);

            // Point closest to edge
            std::array<real_t, 3> closest;
            real_t dist = search::point_triangle_distance({0.5, -0.5, 0}, a, b, c, &closest);
            ASSERT_NEAR(dist, 0.5, EPSILON);
            ASSERT_NEAR(closest[1], 0, EPSILON);
        }

        // Test point to AABB distance
        {
            search::AABB aabb({0, 0, 0}, {1, 1, 1});

            // Point inside
            ASSERT_EQ(search::point_aabb_distance({0.5, 0.5, 0.5}, aabb), 0);

            // Point outside
            ASSERT_NEAR(search::point_aabb_distance({2, 0.5, 0.5}, aabb), 1, EPSILON);
            ASSERT_NEAR(search::point_aabb_distance({2, 2, 0.5}, aabb), std::sqrt(2), EPSILON);
            ASSERT_NEAR(search::point_aabb_distance({2, 2, 2}, aabb), std::sqrt(3), EPSILON);
        }

        std::cout << "  ✓ Distance function tests passed\n";
    }

    void test_barycentric_coords() {
        std::cout << "=== Testing Barycentric Coordinates ===\n";

        // Test triangle barycentric
        {
            std::array<real_t, 3> a = {0, 0, 0};
            std::array<real_t, 3> b = {1, 0, 0};
            std::array<real_t, 3> c = {0, 1, 0};

            // Test vertices
            auto bary = search::triangle_barycentric(a, a, b, c);
            ASSERT_NEAR(bary[0], 1, EPSILON);
            ASSERT_NEAR(bary[1], 0, EPSILON);
            ASSERT_NEAR(bary[2], 0, EPSILON);

            bary = search::triangle_barycentric(b, a, b, c);
            ASSERT_NEAR(bary[0], 0, EPSILON);
            ASSERT_NEAR(bary[1], 1, EPSILON);
            ASSERT_NEAR(bary[2], 0, EPSILON);

            // Test center
            bary = search::triangle_barycentric({1.0/3, 1.0/3, 0}, a, b, c);
            ASSERT_NEAR(bary[0], 1.0/3, EPSILON);
            ASSERT_NEAR(bary[1], 1.0/3, EPSILON);
            ASSERT_NEAR(bary[2], 1.0/3, EPSILON);
        }

        // Test tetrahedron barycentric
        {
            std::array<real_t, 3> a = {0, 0, 0};
            std::array<real_t, 3> b = {1, 0, 0};
            std::array<real_t, 3> c = {0, 1, 0};
            std::array<real_t, 3> d = {0, 0, 1};

            // Test vertices
            auto bary = search::tetrahedron_barycentric(a, a, b, c, d);
            ASSERT_NEAR(bary[0], 1, EPSILON);
            ASSERT_NEAR(bary[1], 0, EPSILON);
            ASSERT_NEAR(bary[2], 0, EPSILON);
            ASSERT_NEAR(bary[3], 0, EPSILON);

            // Test center
            bary = search::tetrahedron_barycentric({0.25, 0.25, 0.25}, a, b, c, d);
            ASSERT_NEAR(bary[0], 0.25, EPSILON);
            ASSERT_NEAR(bary[1], 0.25, EPSILON);
            ASSERT_NEAR(bary[2], 0.25, EPSILON);
            ASSERT_NEAR(bary[3], 0.25, EPSILON);
        }

        std::cout << "  ✓ Barycentric coordinate tests passed\n";
    }

    void test_geometric_utilities() {
        std::cout << "=== Testing Geometric Utilities ===\n";

        // Test triangle area
        {
            std::array<real_t, 3> a = {0, 0, 0};
            std::array<real_t, 3> b = {1, 0, 0};
            std::array<real_t, 3> c = {0, 1, 0};
            ASSERT_NEAR(search::triangle_area(a, b, c), 0.5, EPSILON);
        }

        // Test triangle normal
        {
            std::array<real_t, 3> a = {0, 0, 0};
            std::array<real_t, 3> b = {1, 0, 0};
            std::array<real_t, 3> c = {0, 1, 0};
            auto normal = search::triangle_normal(a, b, c);
            // Normal should point in +z direction with magnitude 1
            ASSERT_NEAR(normal[0], 0, EPSILON);
            ASSERT_NEAR(normal[1], 0, EPSILON);
            ASSERT_NEAR(normal[2], 1, EPSILON);
        }

        // Test tetrahedron volume
        {
            std::array<real_t, 3> a = {0, 0, 0};
            std::array<real_t, 3> b = {1, 0, 0};
            std::array<real_t, 3> c = {0, 1, 0};
            std::array<real_t, 3> d = {0, 0, 1};
            ASSERT_NEAR(std::abs(search::tetrahedron_volume(a, b, c, d)), 1.0/6, EPSILON);
        }

        // Test AABB-sphere overlap
        {
            search::AABB aabb({0, 0, 0}, {1, 1, 1});

            // Sphere inside
            ASSERT_TRUE(search::aabb_sphere_overlap(aabb, {0.5, 0.5, 0.5}, 0.25));

            // Sphere overlapping
            ASSERT_TRUE(search::aabb_sphere_overlap(aabb, {1.5, 0.5, 0.5}, 0.75));

            // Sphere outside
            ASSERT_FALSE(search::aabb_sphere_overlap(aabb, {2, 0.5, 0.5}, 0.5));
        }

        std::cout << "  ✓ Geometric utility tests passed\n";
    }

    void test_vector_operations() {
        std::cout << "=== Testing Vector Operations ===\n";

        std::array<real_t, 3> v1 = {1, 2, 3};
        std::array<real_t, 3> v2 = {4, 5, 6};

        // Test dot product
        ASSERT_EQ(search::dot3(v1, v2), 32);  // 1*4 + 2*5 + 3*6

        // Test cross product
        auto cross = search::cross3(v1, v2);
        ASSERT_EQ(cross[0], -3);  // 2*6 - 3*5
        ASSERT_EQ(cross[1], 6);   // 3*4 - 1*6
        ASSERT_EQ(cross[2], -3);  // 1*5 - 2*4

        // Test normalize
        auto norm = search::normalize3({3, 4, 0});
        ASSERT_NEAR(norm[0], 0.6, EPSILON);
        ASSERT_NEAR(norm[1], 0.8, EPSILON);
        ASSERT_NEAR(norm[2], 0, EPSILON);

        // Test vector arithmetic
        auto sum = search::add3(v1, v2);
        ASSERT_EQ(sum[0], 5);
        ASSERT_EQ(sum[1], 7);
        ASSERT_EQ(sum[2], 9);

        auto diff = search::sub3(v2, v1);
        ASSERT_EQ(diff[0], 3);
        ASSERT_EQ(diff[1], 3);
        ASSERT_EQ(diff[2], 3);

        auto scaled = search::scale3(v1, 2);
        ASSERT_EQ(scaled[0], 2);
        ASSERT_EQ(scaled[1], 4);
        ASSERT_EQ(scaled[2], 6);

        std::cout << "  ✓ Vector operation tests passed\n";
    }

    void run_all_tests() {
        std::cout << "\n========================================\n";
        std::cout << "  SearchPrimitives Unit Test Suite\n";
        std::cout << "========================================\n\n";

        test_aabb();
        test_ray();
        test_point_in_triangle();
        test_point_in_tetrahedron();
        test_point_in_hex();
        test_ray_triangle_intersection();
        test_ray_aabb_intersection();
        test_distance_functions();
        test_barycentric_coords();
        test_geometric_utilities();
        test_vector_operations();

        std::cout << "\n========================================\n";
        std::cout << "  All SearchPrimitives tests PASSED! ✓\n";
        std::cout << "========================================\n\n";
    }
};

} // namespace test
} // namespace svmp

int main() {
    svmp::test::TestSearchPrimitives tester;
    tester.run_all_tests();
    return 0;
}