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
        // Coordinates for 4 vertices in 3D
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // Vertex 0
            1.0, 0.0, 0.0,  // Vertex 1
            0.0, 1.0, 0.0,  // Vertex 2
            0.0, 0.0, 1.0   // Vertex 3
        };

        // Single tetrahedron connectivity
        std::vector<offset_t> offs = {0, 4};
        std::vector<index_t> conn = {0, 1, 2, 3};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Tetra;
        shapes[0].order = 1;
        shapes[0].num_corners = 4;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
        return mesh;
    }

    /**
     * @brief Create a unit triangle mesh (2D)
     */
    MeshBase create_unit_triangle_mesh() {
        // Coordinates for 3 vertices in 2D (z is omitted for 2D meshes)
        std::vector<real_t> X_ref = {
            0.0, 0.0,  // Vertex 0
            1.0, 0.0,  // Vertex 1
            0.0, 1.0   // Vertex 2
        };

        // Single triangle connectivity
        std::vector<offset_t> offs = {0, 3};
        std::vector<index_t> conn = {0, 1, 2};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Triangle;
        shapes[0].order = 1;
        shapes[0].num_corners = 3;

        MeshBase mesh;
        mesh.build_from_arrays(2, X_ref, offs, conn, shapes);
        return mesh;
    }

    MeshBase create_unit_quad_mesh() {
        // Unit square in 2D.
        std::vector<real_t> X_ref = {
            0.0, 0.0,  // 0
            1.0, 0.0,  // 1
            1.0, 1.0,  // 2
            0.0, 1.0   // 3
        };

        std::vector<offset_t> offs = {0, 4};
        std::vector<index_t> conn = {0, 1, 2, 3};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Quad;
        shapes[0].order = 1;
        shapes[0].num_corners = 4;

        MeshBase mesh;
        mesh.build_from_arrays(2, X_ref, offs, conn, shapes);
        return mesh;
    }

    MeshBase create_unit_hex_mesh(bool finalize_topology = false) {
        // Unit cube in 3D.
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            1.0, 1.0, 0.0,  // 2
            0.0, 1.0, 0.0,  // 3
            0.0, 0.0, 1.0,  // 4
            1.0, 0.0, 1.0,  // 5
            1.0, 1.0, 1.0,  // 6
            0.0, 1.0, 1.0   // 7
        };

        std::vector<offset_t> offs = {0, 8};
        std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Hex;
        shapes[0].order = 1;
        shapes[0].num_corners = 8;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
        if (finalize_topology) {
            mesh.finalize();
        }
        return mesh;
    }

    MeshBase create_unit_wedge_mesh() {
        // Right triangular prism:
        // base triangle (z=0): (0,0), (1,0), (0,1)
        // top (z=1): base + (0,0,1)
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0,  // 2
            0.0, 0.0, 1.0,  // 3
            1.0, 0.0, 1.0,  // 4
            0.0, 1.0, 1.0   // 5
        };

        std::vector<offset_t> offs = {0, 6};
        std::vector<index_t> conn = {0, 1, 2, 3, 4, 5};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Wedge;
        shapes[0].order = 1;
        shapes[0].num_corners = 6;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
        return mesh;
    }

    MeshBase create_unit_pyramid_mesh() {
        // Square base on z=0 with apex at z=1 above center.
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            1.0, 1.0, 0.0,  // 2
            0.0, 1.0, 0.0,  // 3
            0.5, 0.5, 1.0   // 4
        };

        std::vector<offset_t> offs = {0, 5};
        std::vector<index_t> conn = {0, 1, 2, 3, 4};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Pyramid;
        shapes[0].order = 1;
        shapes[0].num_corners = 5;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
        return mesh;
    }

    MeshBase create_unit_cube_polyhedron_mesh() {
        // Unit cube as a polyhedron with explicit quad faces.
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            1.0, 1.0, 0.0,  // 2
            0.0, 1.0, 0.0,  // 3
            0.0, 0.0, 1.0,  // 4
            1.0, 0.0, 1.0,  // 5
            1.0, 1.0, 1.0,  // 6
            0.0, 1.0, 1.0   // 7
        };

        std::vector<offset_t> offs = {0, 8};
        std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Polyhedron;
        shapes[0].order = 1;
        shapes[0].num_corners = 8;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, conn, shapes);

        // Face list: 6 quads (vertex ordering does not matter for volume/centroid).
        std::vector<CellShape> face_shapes(6);
        for (auto& fs : face_shapes) {
            fs.family = CellFamily::Quad;
            fs.order = 1;
            fs.num_corners = 4;
        }

        std::vector<offset_t> face_offs = {0, 4, 8, 12, 16, 20, 24};
        std::vector<index_t> face_conn = {
            0, 1, 2, 3,  // z=0
            4, 5, 6, 7,  // z=1
            0, 1, 5, 4,  // y=0
            1, 2, 6, 5,  // x=1
            2, 3, 7, 6,  // y=1
            3, 0, 4, 7   // x=0
        };

        std::vector<std::array<index_t,2>> face2cell(6);
        for (auto& fc : face2cell) {
            fc = {{0, INVALID_INDEX}};
        }
        mesh.set_faces_from_arrays(face_shapes, face_offs, face_conn, face2cell);
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

TEST_F(MeshGeometryTest, AngleAtVertex) {
    std::array<real_t,3> p1 = {{1.0, 0.0, 0.0}};
    std::array<real_t,3> p2 = {{0.0, 0.0, 0.0}};
    std::array<real_t,3> p3 = {{0.0, 1.0, 0.0}};

    // Right angle at p2.
    EXPECT_TRUE(approx_equal(MeshGeometry::angle(p1, p2, p3), static_cast<real_t>(M_PI) * 0.5));
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

TEST_F(MeshGeometryTest, HexWedgePyramidCellCenters) {
    auto hex = create_unit_hex_mesh();
    auto wedge = create_unit_wedge_mesh();
    auto pyramid = create_unit_pyramid_mesh();

    EXPECT_TRUE(approx_equal(MeshGeometry::cell_center(hex, 0), {{0.5, 0.5, 0.5}}));
    EXPECT_TRUE(approx_equal(MeshGeometry::cell_center(wedge, 0), {{1.0/3.0, 1.0/3.0, 0.5}}));
    // Note: current MeshGeometry::cell_center is the vertex-average ("barycenter of corners").
    EXPECT_TRUE(approx_equal(MeshGeometry::cell_center(pyramid, 0), {{0.5, 0.5, 0.2}}));
}

TEST_F(MeshGeometryTest, CellCentroidPyramidDiffersFromVertexAverage) {
    auto pyramid = create_unit_pyramid_mesh();

    const auto bary = MeshGeometry::cell_center(pyramid, 0);
    const auto cent = MeshGeometry::cell_centroid(pyramid, 0);

    EXPECT_TRUE(approx_equal(bary, {{0.5, 0.5, 0.2}}));
    // Volume centroid of a right pyramid is at z = h/4.
    EXPECT_TRUE(approx_equal(cent, {{0.5, 0.5, 0.25}}));
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

TEST_F(MeshGeometryTest, QuadCellArea) {
    MeshBase mesh = create_unit_quad_mesh();
    EXPECT_TRUE(approx_equal(MeshGeometry::cell_measure(mesh, 0), 1.0));
}

TEST_F(MeshGeometryTest, PolygonCellArea) {
    // Unit square as a generic polygon cell in a 3D-embedded mesh.
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // 0
        1.0, 0.0, 0.0,  // 1
        1.0, 1.0, 0.0,  // 2
        0.0, 1.0, 0.0   // 3
    };
    std::vector<offset_t> offs = {0, 4};
    std::vector<index_t> conn = {0, 1, 2, 3};
    std::vector<CellShape> shapes(1);
    shapes[0].family = CellFamily::Polygon;
    shapes[0].order = 1;
    shapes[0].num_corners = 4;

    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs, conn, shapes);

    EXPECT_TRUE(approx_equal(MeshGeometry::cell_measure(mesh, 0), 1.0));
}

TEST_F(MeshGeometryTest, HexWedgePyramidCellMeasures) {
    auto hex = create_unit_hex_mesh();
    auto wedge = create_unit_wedge_mesh();
    auto pyramid = create_unit_pyramid_mesh();

    EXPECT_TRUE(approx_equal(MeshGeometry::cell_measure(hex, 0), 1.0));
    EXPECT_TRUE(approx_equal(MeshGeometry::cell_measure(wedge, 0), 0.5));
    EXPECT_TRUE(approx_equal(MeshGeometry::cell_measure(pyramid, 0), 1.0/3.0));
}

TEST_F(MeshGeometryTest, PolyhedronCellVolumeAndCentroid) {
    auto poly = create_unit_cube_polyhedron_mesh();
    EXPECT_TRUE(approx_equal(MeshGeometry::cell_measure(poly, 0), 1.0));
    EXPECT_TRUE(approx_equal(MeshGeometry::cell_centroid(poly, 0), {{0.5, 0.5, 0.5}}));
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

TEST_F(MeshGeometryTest, HexBoundingBox) {
    MeshBase mesh = create_unit_hex_mesh();
    auto bbox = MeshGeometry::cell_bounding_box(mesh, 0);

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
    // 3D coordinates for 3 vertices (triangle)
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // Vertex 0
        1.0, 0.0, 0.0,  // Vertex 1
        0.0, 1.0, 0.0   // Vertex 2
    };
    // Build a mesh with coordinates only (no cells)
    std::vector<offset_t> offs0 = {0};
    std::vector<index_t> empty_conn;
    std::vector<CellShape> empty_shapes;
    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs0, empty_conn, empty_shapes);

    std::vector<index_t> oriented_verts = {0, 1, 2};

    auto normal = MeshGeometry::compute_normal_from_vertices(mesh, oriented_verts);

    // Normal should point in +z direction
    EXPECT_TRUE(approx_equal(normal[0], 0.0));
    EXPECT_TRUE(approx_equal(normal[1], 0.0));
    EXPECT_TRUE(approx_equal(std::abs(normal[2]), 1.0));
}

TEST_F(MeshGeometryTest, ComputeNormalFromVerticesCollinearFirstThree) {
    // Polygon where the first three vertices are collinear (cross-product would fail).
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // 0
        1.0, 0.0, 0.0,  // 1
        2.0, 0.0, 0.0,  // 2 (collinear with 0,1)
        2.0, 1.0, 0.0,  // 3
        0.0, 1.0, 0.0   // 4
    };
    std::vector<offset_t> offs0 = {0};
    std::vector<index_t> empty_conn;
    std::vector<CellShape> empty_shapes;
    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs0, empty_conn, empty_shapes);

    std::vector<index_t> oriented_verts = {0, 1, 2, 3, 4};
    auto normal = MeshGeometry::compute_normal_from_vertices(mesh, oriented_verts);

    EXPECT_TRUE(approx_equal(normal[0], 0.0));
    EXPECT_TRUE(approx_equal(normal[1], 0.0));
    EXPECT_GT(normal[2], 0.0);
    EXPECT_TRUE(approx_equal(normal[2], 4.0));  // 2*area for 2x1 rectangle = 4
}

TEST_F(MeshGeometryTest, ComputeAreaFromVerticesTriangle) {
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // Vertex 0
        1.0, 0.0, 0.0,  // Vertex 1
        0.0, 1.0, 0.0   // Vertex 2
    };
    std::vector<offset_t> offs0 = {0};
    std::vector<index_t> empty_conn;
    std::vector<CellShape> empty_shapes;
    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs0, empty_conn, empty_shapes);

    std::vector<index_t> oriented_verts = {0, 1, 2};

    real_t area = MeshGeometry::compute_area_from_vertices(mesh, oriented_verts);

    // Area of unit right triangle = 0.5
    EXPECT_TRUE(approx_equal(area, 0.5));
}

TEST_F(MeshGeometryTest, ComputeCentroidFromVertices) {
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // Vertex 0
        3.0, 0.0, 0.0,  // Vertex 1
        0.0, 3.0, 0.0   // Vertex 2
    };
    std::vector<offset_t> offs0 = {0};
    std::vector<index_t> empty_conn;
    std::vector<CellShape> empty_shapes;
    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs0, empty_conn, empty_shapes);

    std::vector<index_t> verts = {0, 1, 2};

    auto centroid = MeshGeometry::compute_centroid_from_vertices(mesh, verts);

    // Centroid should be at (1, 1, 0)
    EXPECT_TRUE(approx_equal(centroid, {{1.0, 1.0, 0.0}}));
}

TEST_F(MeshGeometryTest, ComputeEdgeNormalFromVertices2D) {
    std::vector<real_t> X_ref = {
        0.0, 0.0,  // Vertex 0
        1.0, 0.0   // Vertex 1
    };
    std::vector<offset_t> offs0 = {0};
    std::vector<index_t> empty_conn;
    std::vector<CellShape> empty_shapes;
    MeshBase mesh;
    mesh.build_from_arrays(2, X_ref, offs0, empty_conn, empty_shapes);

    std::vector<index_t> oriented_verts = {0, 1};

    auto normal = MeshGeometry::compute_edge_normal_from_vertices(mesh, oriented_verts);

    // Edge goes in +x direction, normal should be in +y direction (90° CCW)
    EXPECT_TRUE(approx_equal(normal, {{0.0, 1.0, 0.0}}));
}

TEST_F(MeshGeometryTest, TotalVolumeAndBoundaryAreaHex) {
    MeshBase mesh = create_unit_hex_mesh(/*finalize_topology=*/true);

    EXPECT_TRUE(approx_equal(MeshGeometry::total_volume(mesh), 1.0));
    EXPECT_TRUE(approx_equal(MeshGeometry::boundary_area(mesh), 6.0));

    // Edge lengths should all be 1 for a unit cube.
    ASSERT_GT(mesh.n_edges(), 0u);
    for (index_t e = 0; e < static_cast<index_t>(mesh.n_edges()); ++e) {
        EXPECT_TRUE(approx_equal(MeshGeometry::edge_length(mesh, e), 1.0));
    }
}

TEST_F(MeshGeometryTest, DeformedUsesCurrentCoordsWhenAvailable) {
    MeshBase mesh = create_unit_tet_mesh();

    // Shift current coordinates by +1 in x.
    auto X_cur = mesh.X_ref();
    for (size_t i = 0; i < X_cur.size(); i += 3) {
        X_cur[i] += 1.0;
    }
    mesh.set_current_coords(X_cur);

    auto cref = MeshGeometry::cell_center(mesh, 0, Configuration::Reference);
    auto cdef = MeshGeometry::cell_center(mesh, 0, Configuration::Deformed);
    EXPECT_TRUE(approx_equal(cref, {{0.25, 0.25, 0.25}}));
    EXPECT_TRUE(approx_equal(cdef, {{1.25, 0.25, 0.25}}));
}

} // namespace test
} // namespace svmp
