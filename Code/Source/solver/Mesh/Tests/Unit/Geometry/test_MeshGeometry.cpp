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

#include "gtest/gtest.h"
#include "Geometry/MeshGeometry.h"
#include "Core/MeshBase.h"
#include "Topology/CellShape.h"
#include <cmath>

namespace svmp {
namespace test {

/**
 * @brief Test fixture for MeshGeometry tests
 */
class MeshGeometryTest : public ::testing::Test {
protected:
    static constexpr real_t tolerance = 1e-10;

    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        // Cleanup if needed
    }

    /**
     * @brief Create a unit tetrahedron mesh
     */
    MeshBase create_unit_tet_mesh() {
        MeshBase mesh(3, 4, 1);

        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // Node 0
            1.0, 0.0, 0.0,  // Node 1
            0.0, 1.0, 0.0,  // Node 2
            0.0, 0.0, 1.0   // Node 3
        };
        mesh.set_X_ref(X_ref);

        CellShape shape(CellFamily::Tetra, 1);
        mesh.add_cell({0, 1, 2, 3}, shape);

        return mesh;
    }

    /**
     * @brief Create a unit triangle mesh (2D)
     */
    MeshBase create_unit_triangle_mesh() {
        MeshBase mesh(2, 3, 1);

        std::vector<real_t> X_ref = {
            0.0, 0.0,  // Node 0
            1.0, 0.0,  // Node 1
            0.0, 1.0   // Node 2
        };
        mesh.set_X_ref(X_ref);

        CellShape shape(CellFamily::Triangle, 1);
        mesh.add_cell({0, 1, 2}, shape);

        return mesh;
    }

    /**
     * @brief Helper to compare real_t values within tolerance
     */
    bool approx_equal(real_t a, real_t b, real_t tol = tolerance) {
        return std::abs(a - b) < tol;
    }

    /**
     * @brief Helper to compare arrays within tolerance
     */
    bool approx_equal(const std::array<real_t,3>& a, const std::array<real_t,3>& b, real_t tol = tolerance) {
        return approx_equal(a[0], b[0], tol) &&
               approx_equal(a[1], b[1], tol) &&
               approx_equal(a[2], b[2], tol);
    }
};

// ==========================================
// Tests: Vector Operations
// ==========================================

TEST_F(MeshGeometryTest, CrossProduct) {
    std::array<real_t,3> a = {{1.0, 0.0, 0.0}};
    std::array<real_t,3> b = {{0.0, 1.0, 0.0}};

    auto result = MeshGeometry::cross(a, b);

    EXPECT_TRUE(approx_equal(result, {{0.0, 0.0, 1.0}}));
}

TEST_F(MeshGeometryTest, DotProduct) {
    std::array<real_t,3> a = {{1.0, 2.0, 3.0}};
    std::array<real_t,3> b = {{4.0, 5.0, 6.0}};

    real_t result = MeshGeometry::dot(a, b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_TRUE(approx_equal(result, 32.0));
}

TEST_F(MeshGeometryTest, Magnitude) {
    std::array<real_t,3> v = {{3.0, 4.0, 0.0}};

    real_t result = MeshGeometry::magnitude(v);

    // sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5
    EXPECT_TRUE(approx_equal(result, 5.0));
}

TEST_F(MeshGeometryTest, Normalize) {
    std::array<real_t,3> v = {{3.0, 4.0, 0.0}};

    auto result = MeshGeometry::normalize(v);

    // Should be (3/5, 4/5, 0)
    EXPECT_TRUE(approx_equal(result, {{0.6, 0.8, 0.0}}));
    EXPECT_TRUE(approx_equal(MeshGeometry::magnitude(result), 1.0));
}

TEST_F(MeshGeometryTest, NormalizeZeroVector) {
    std::array<real_t,3> v = {{0.0, 0.0, 0.0}};

    auto result = MeshGeometry::normalize(v);

    // Should return zero vector
    EXPECT_TRUE(approx_equal(result, {{0.0, 0.0, 0.0}}));
}

// ==========================================
// Tests: Distance and Angles
// ==========================================

TEST_F(MeshGeometryTest, Distance) {
    std::array<real_t,3> p1 = {{0.0, 0.0, 0.0}};
    std::array<real_t,3> p2 = {{3.0, 4.0, 0.0}};

    real_t result = MeshGeometry::distance(p1, p2);

    EXPECT_TRUE(approx_equal(result, 5.0));
}

// ==========================================
// Tests: Triangle Area
// ==========================================

TEST_F(MeshGeometryTest, TriangleAreaRightAngle) {
    std::array<real_t,3> p0 = {{0.0, 0.0, 0.0}};
    std::array<real_t,3> p1 = {{1.0, 0.0, 0.0}};
    std::array<real_t,3> p2 = {{0.0, 1.0, 0.0}};

    real_t area = MeshGeometry::triangle_area(p0, p1, p2);

    // Right triangle with legs 1 and 1: area = 0.5
    EXPECT_TRUE(approx_equal(area, 0.5));
}

TEST_F(MeshGeometryTest, TriangleAreaDegenerate) {
    std::array<real_t,3> p0 = {{0.0, 0.0, 0.0}};
    std::array<real_t,3> p1 = {{1.0, 0.0, 0.0}};
    std::array<real_t,3> p2 = {{2.0, 0.0, 0.0}};  // Collinear

    real_t area = MeshGeometry::triangle_area(p0, p1, p2);

    // Degenerate triangle (collinear points) should have zero area
    EXPECT_TRUE(approx_equal(area, 0.0));
}

// ==========================================
// Tests: Tetrahedron Volume
// ==========================================

TEST_F(MeshGeometryTest, TetVolumeUnitTet) {
    std::array<real_t,3> p0 = {{0.0, 0.0, 0.0}};
    std::array<real_t,3> p1 = {{1.0, 0.0, 0.0}};
    std::array<real_t,3> p2 = {{0.0, 1.0, 0.0}};
    std::array<real_t,3> p3 = {{0.0, 0.0, 1.0}};

    real_t volume = MeshGeometry::tet_volume(p0, p1, p2, p3);

    // Volume of unit tetrahedron = 1/6
    EXPECT_TRUE(approx_equal(std::abs(volume), 1.0/6.0));
}

// ==========================================
// Tests: Cell Centers
// ==========================================

TEST_F(MeshGeometryTest, TetCellCenter) {
    MeshBase mesh = create_unit_tet_mesh();

    auto center = MeshGeometry::cell_center(mesh, 0);

    // Center of unit tet should be at (0.25, 0.25, 0.25)
    EXPECT_TRUE(approx_equal(center, {{0.25, 0.25, 0.25}}));
}

TEST_F(MeshGeometryTest, TriangleCellCenter) {
    MeshBase mesh = create_unit_triangle_mesh();

    auto center = MeshGeometry::cell_center(mesh, 0);

    // Center of unit triangle should be at (1/3, 1/3, 0)
    EXPECT_TRUE(approx_equal(center, {{1.0/3.0, 1.0/3.0, 0.0}}));
}

// ==========================================
// Tests: Cell Measures
// ==========================================

TEST_F(MeshGeometryTest, TetCellVolume) {
    MeshBase mesh = create_unit_tet_mesh();

    real_t volume = MeshGeometry::cell_measure(mesh, 0);

    // Volume of unit tet = 1/6
    EXPECT_TRUE(approx_equal(volume, 1.0/6.0));
}

TEST_F(MeshGeometryTest, TriangleCellArea) {
    MeshBase mesh = create_unit_triangle_mesh();

    real_t area = MeshGeometry::cell_measure(mesh, 0);

    // Area of unit right triangle = 0.5
    EXPECT_TRUE(approx_equal(area, 0.5));
}

// ==========================================
// Tests: Bounding Boxes
// ==========================================

TEST_F(MeshGeometryTest, TetBoundingBox) {
    MeshBase mesh = create_unit_tet_mesh();

    auto bbox = MeshGeometry::bounding_box(mesh);

    EXPECT_TRUE(approx_equal(bbox.min[0], 0.0));
    EXPECT_TRUE(approx_equal(bbox.min[1], 0.0));
    EXPECT_TRUE(approx_equal(bbox.min[2], 0.0));

    EXPECT_TRUE(approx_equal(bbox.max[0], 1.0));
    EXPECT_TRUE(approx_equal(bbox.max[1], 1.0));
    EXPECT_TRUE(approx_equal(bbox.max[2], 1.0));
}

// ==========================================
// Tests: Boundary Geometry (from vertices)
// ==========================================

TEST_F(MeshGeometryTest, ComputeNormalFromVerticesTriangle) {
    MeshBase mesh(3, 3, 0);
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        0.0, 1.0, 0.0   // Node 2
    };
    mesh.set_X_ref(X_ref);

    std::vector<index_t> oriented_verts = {0, 1, 2};

    auto normal = MeshGeometry::compute_normal_from_vertices(mesh, oriented_verts);

    // Normal should point in +z direction
    EXPECT_TRUE(approx_equal(normal[0], 0.0));
    EXPECT_TRUE(approx_equal(normal[1], 0.0));
    EXPECT_TRUE(approx_equal(std::abs(normal[2]), 1.0));
}

TEST_F(MeshGeometryTest, ComputeAreaFromVerticesTriangle) {
    MeshBase mesh(3, 3, 0);
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        0.0, 1.0, 0.0   // Node 2
    };
    mesh.set_X_ref(X_ref);

    std::vector<index_t> oriented_verts = {0, 1, 2};

    real_t area = MeshGeometry::compute_area_from_vertices(mesh, oriented_verts);

    // Area of unit right triangle = 0.5
    EXPECT_TRUE(approx_equal(area, 0.5));
}

TEST_F(MeshGeometryTest, ComputeCentroidFromVertices) {
    MeshBase mesh(3, 3, 0);
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // Node 0
        3.0, 0.0, 0.0,  // Node 1
        0.0, 3.0, 0.0   // Node 2
    };
    mesh.set_X_ref(X_ref);

    std::vector<index_t> verts = {0, 1, 2};

    auto centroid = MeshGeometry::compute_centroid_from_vertices(mesh, verts);

    // Centroid should be at (1, 1, 0)
    EXPECT_TRUE(approx_equal(centroid, {{1.0, 1.0, 0.0}}));
}

TEST_F(MeshGeometryTest, ComputeEdgeNormalFromVertices2D) {
    MeshBase mesh(2, 2, 0);
    std::vector<real_t> X_ref = {
        0.0, 0.0,  // Node 0
        1.0, 0.0   // Node 1
    };
    mesh.set_X_ref(X_ref);

    std::vector<index_t> oriented_verts = {0, 1};

    auto normal = MeshGeometry::compute_edge_normal_from_vertices(mesh, oriented_verts);

    // Edge goes in +x direction, normal should be in +y direction (90° CCW)
    EXPECT_TRUE(approx_equal(normal, {{0.0, 1.0, 0.0}}));
}

} // namespace test
} // namespace svmp
