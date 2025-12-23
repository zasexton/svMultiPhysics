/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>
#include "../../../IO/GmshReader.h"
#include "../../../Core/MeshBase.h"
#include "../../../Core/MeshTypes.h"
#include "Geometry/CurvilinearEval.h"

#include <fstream>
#include <sstream>
#include <cstdio>

namespace svmp {
namespace test {

// ==========================================
// Test Fixture
// ==========================================

class GmshReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory path
        temp_dir_ = "/tmp/gmsh_test_";
        temp_dir_ += std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    }

    void TearDown() override {
        // Clean up temp files
        for (const auto& file : temp_files_) {
            std::remove(file.c_str());
        }
    }

    std::string write_temp_file(const std::string& content, const std::string& suffix = ".msh") {
        std::string filename = temp_dir_ + "_" + std::to_string(temp_files_.size()) + suffix;
        std::ofstream file(filename);
        file << content;
        file.close();
        temp_files_.push_back(filename);
        return filename;
    }

    std::string temp_dir_;
    std::vector<std::string> temp_files_;
};

// ==========================================
// Sample Gmsh Files (MSH 2.2 Format)
// ==========================================

// Simple 2D quad mesh (2x2 quads)
const char* QUAD_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
9
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 0.0 1.0 0.0
5 1.0 1.0 0.0
6 2.0 1.0 0.0
7 0.0 2.0 0.0
8 1.0 2.0 0.0
9 2.0 2.0 0.0
$EndNodes
$Elements
4
1 3 2 1 1 1 2 5 4
2 3 2 1 1 2 3 6 5
3 3 2 1 1 4 5 8 7
4 3 2 1 1 5 6 9 8
$EndElements
)";

// Simple 2D triangle mesh (4 triangles forming a square)
const char* TRIANGLE_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.5 0.5 0.0
$EndNodes
$Elements
4
1 2 2 1 1 1 2 5
2 2 2 1 1 2 3 5
3 2 2 1 1 3 4 5
4 2 2 1 1 4 1 5
$EndElements
)";

// 3D tetrahedron mesh (single tet)
const char* TET_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.5 0.866025 0.0
4 0.5 0.288675 0.816497
$EndNodes
$Elements
1
1 4 2 1 1 1 2 3 4
$EndElements
)";

// 3D hexahedron mesh (single hex)
const char* HEX_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
8
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
$EndNodes
$Elements
1
1 5 2 1 1 1 2 3 4 5 6 7 8
$EndElements
)";

// Mixed element mesh (hex + tet + pyramid + wedge)
const char* MIXED_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
3 1 "domain"
2 2 "boundary"
$EndPhysicalNames
$Nodes
14
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
9 2.0 0.0 0.0
10 2.0 1.0 0.0
11 2.0 0.0 1.0
12 2.0 1.0 1.0
13 1.5 0.5 0.5
14 2.5 0.5 0.5
$EndNodes
$Elements
4
1 5 2 1 1 1 2 3 4 5 6 7 8
2 4 2 1 1 9 10 12 14
3 7 2 1 1 2 9 10 3 13
4 6 2 1 1 6 11 12 7 13 13
$EndElements
)";

// L-shaped domain (2D, classic benchmark)
const char* L_SHAPE_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "domain"
$EndPhysicalNames
$Nodes
8
1 -1.0 -1.0 0.0
2  0.0 -1.0 0.0
3  1.0 -1.0 0.0
4 -1.0  0.0 0.0
5  0.0  0.0 0.0
6  1.0  0.0 0.0
7 -1.0  1.0 0.0
8  0.0  1.0 0.0
$EndNodes
$Elements
3
1 3 2 1 1 1 2 5 4
2 3 2 1 1 2 3 6 5
3 3 2 1 1 4 5 8 7
$EndElements
)";

// Quadratic triangle mesh (6-node triangles)
const char* TRIANGLE6_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.5 0.866 0.0
4 0.5 0.0 0.0
5 0.75 0.433 0.0
6 0.25 0.433 0.0
$EndNodes
$Elements
1
1 9 2 1 1 1 2 3 4 5 6
$EndElements
)";

// Quadratic line mesh (3-node line / LINE3). Connectivity in Gmsh ordering is [v0, v1, mid].
const char* LINE3_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 -1.0 0.0 0.0
2  1.0 0.0 0.0
3  0.0 0.0 0.0
$EndNodes
$Elements
1
1 8 2 1 1 1 2 3
$EndElements
)";

// Cubic line mesh (4-node edge / EDGE4). Connectivity in Gmsh ordering is [v0, v1, interior...].
const char* EDGE4_MESH_V2 = R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 -1.0 0.0 0.0
2  1.0 0.0 0.0
3 -0.3333333333333333 0.0 0.0
4  0.3333333333333333 0.0 0.0
$EndNodes
$Elements
1
1 26 2 1 1 1 2 3 4
$EndElements
)";

// ==========================================
// Sample Gmsh Files (MSH 4.1 Format)
// ==========================================

const char* QUAD_MESH_V4 = R"(
$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 9 1 9
2 0 0 9
1
2
3
4
5
6
7
8
9
0.0 0.0 0.0
1.0 0.0 0.0
2.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
2.0 1.0 0.0
0.0 2.0 0.0
1.0 2.0 0.0
2.0 2.0 0.0
$EndNodes
$Elements
1 4 1 4
2 1 3 4
1 1 2 5 4
2 2 3 6 5
3 4 5 8 7
4 5 6 9 8
$EndElements
)";

// ==========================================
// Tests
// ==========================================

TEST_F(GmshReaderTest, ReadQuadMeshV2) {
    std::string filename = write_temp_file(QUAD_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 9);
    EXPECT_EQ(mesh.n_cells(), 4);
    EXPECT_EQ(mesh.dim(), 3);  // Always 3D coordinates

    // Check cell shapes
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        auto shape = mesh.cell_shape(c);
        EXPECT_EQ(shape.family, CellFamily::Quad);
        EXPECT_EQ(shape.num_corners, 4);
    }
}

TEST_F(GmshReaderTest, ReadTriangleMeshV2) {
    std::string filename = write_temp_file(TRIANGLE_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 5);
    EXPECT_EQ(mesh.n_cells(), 4);

    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        auto shape = mesh.cell_shape(c);
        EXPECT_EQ(shape.family, CellFamily::Triangle);
        EXPECT_EQ(shape.num_corners, 3);
    }
}

TEST_F(GmshReaderTest, ReadTetMeshV2) {
    std::string filename = write_temp_file(TET_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 4);
    EXPECT_EQ(mesh.n_cells(), 1);

    auto shape = mesh.cell_shape(0);
    EXPECT_EQ(shape.family, CellFamily::Tetra);
    EXPECT_EQ(shape.num_corners, 4);
}

TEST_F(GmshReaderTest, ReadHexMeshV2) {
    std::string filename = write_temp_file(HEX_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 8);
    EXPECT_EQ(mesh.n_cells(), 1);

    auto shape = mesh.cell_shape(0);
    EXPECT_EQ(shape.family, CellFamily::Hex);
    EXPECT_EQ(shape.num_corners, 8);
}

TEST_F(GmshReaderTest, ReadMixedMeshV2) {
    std::string filename = write_temp_file(MIXED_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 14);
    EXPECT_EQ(mesh.n_cells(), 4);

    // Count element types
    int hex_count = 0, tet_count = 0, pyramid_count = 0, wedge_count = 0;
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        auto shape = mesh.cell_shape(c);
        switch (shape.family) {
            case CellFamily::Hex: hex_count++; break;
            case CellFamily::Tetra: tet_count++; break;
            case CellFamily::Pyramid: pyramid_count++; break;
            case CellFamily::Wedge: wedge_count++; break;
            default: break;
        }
    }

    EXPECT_EQ(hex_count, 1);
    EXPECT_EQ(tet_count, 1);
    EXPECT_EQ(pyramid_count, 1);
    EXPECT_EQ(wedge_count, 1);
}

TEST_F(GmshReaderTest, ReadLShapeMeshV2) {
    std::string filename = write_temp_file(L_SHAPE_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 8);
    EXPECT_EQ(mesh.n_cells(), 3);

    // All cells should be quads
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        auto shape = mesh.cell_shape(c);
        EXPECT_EQ(shape.family, CellFamily::Quad);
    }

    // Check coordinates cover L-shape region [-1,1] x [-1,1] minus [0,1] x [0,1]
    const auto& coords = mesh.X_ref();
    double x_min = 1e10, x_max = -1e10;
    double y_min = 1e10, y_max = -1e10;

    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
        x_min = std::min(x_min, coords[i * 3]);
        x_max = std::max(x_max, coords[i * 3]);
        y_min = std::min(y_min, coords[i * 3 + 1]);
        y_max = std::max(y_max, coords[i * 3 + 1]);
    }

    EXPECT_NEAR(x_min, -1.0, 1e-10);
    EXPECT_NEAR(x_max, 1.0, 1e-10);
    EXPECT_NEAR(y_min, -1.0, 1e-10);
    EXPECT_NEAR(y_max, 1.0, 1e-10);
}

TEST_F(GmshReaderTest, ReadQuadraticTriangleV2) {
    std::string filename = write_temp_file(TRIANGLE6_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 6);
    EXPECT_EQ(mesh.n_cells(), 1);

    auto shape = mesh.cell_shape(0);
    EXPECT_EQ(shape.family, CellFamily::Triangle);
    EXPECT_EQ(shape.order, 2);  // Quadratic
}

TEST_F(GmshReaderTest, ReadQuadraticLine3V2_ReordersToVTK) {
    std::string filename = write_temp_file(LINE3_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);
    ASSERT_EQ(mesh.n_cells(), 1u);

    auto shape = mesh.cell_shape(0);
    EXPECT_EQ(shape.family, CellFamily::Line);
    EXPECT_EQ(shape.order, 2);

    // Identity mapping check: nodes are located at the reference parametric positions.
    const auto eval0 = CurvilinearEvaluator::evaluate_geometry(mesh, 0, {0.0, 0.0, 0.0});
    EXPECT_NEAR(eval0.coordinates[0], 0.0, 1e-12);
    EXPECT_NEAR(eval0.coordinates[1], 0.0, 1e-12);
    EXPECT_NEAR(eval0.coordinates[2], 0.0, 1e-12);
}

TEST_F(GmshReaderTest, ReadCubicEdge4V2_ReordersToVTK) {
    std::string filename = write_temp_file(EDGE4_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);
    ASSERT_EQ(mesh.n_cells(), 1u);

    auto shape = mesh.cell_shape(0);
    EXPECT_EQ(shape.family, CellFamily::Line);
    EXPECT_EQ(shape.order, 3);

    const auto eval0 = CurvilinearEvaluator::evaluate_geometry(mesh, 0, {0.0, 0.0, 0.0});
    EXPECT_NEAR(eval0.coordinates[0], 0.0, 1e-12);
    EXPECT_NEAR(eval0.coordinates[1], 0.0, 1e-12);
    EXPECT_NEAR(eval0.coordinates[2], 0.0, 1e-12);
}

TEST_F(GmshReaderTest, ReadQuadMeshV4) {
    std::string filename = write_temp_file(QUAD_MESH_V4);

    MeshBase mesh = GmshReader::read(filename);

    EXPECT_EQ(mesh.n_vertices(), 9);
    EXPECT_EQ(mesh.n_cells(), 4);

    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        auto shape = mesh.cell_shape(c);
        EXPECT_EQ(shape.family, CellFamily::Quad);
    }
}

TEST_F(GmshReaderTest, IsGmshFile) {
    std::string gmsh_file = write_temp_file(QUAD_MESH_V2);
    std::string non_gmsh_file = write_temp_file("This is not a gmsh file", ".txt");

    EXPECT_TRUE(GmshReader::is_gmsh_file(gmsh_file));
    EXPECT_FALSE(GmshReader::is_gmsh_file(non_gmsh_file));
    EXPECT_FALSE(GmshReader::is_gmsh_file("/nonexistent/file.msh"));
}

TEST_F(GmshReaderTest, GetFormatVersion) {
    std::string v2_file = write_temp_file(QUAD_MESH_V2);
    std::string v4_file = write_temp_file(QUAD_MESH_V4);

    EXPECT_NEAR(GmshReader::get_format_version(v2_file), 2.2, 0.01);
    EXPECT_NEAR(GmshReader::get_format_version(v4_file), 4.1, 0.01);
}

TEST_F(GmshReaderTest, InvalidFileThrows) {
    EXPECT_THROW(GmshReader::read("/nonexistent/file.msh"), std::runtime_error);

    std::string invalid_file = write_temp_file("Not a valid mesh format");
    EXPECT_THROW(GmshReader::read(invalid_file), std::runtime_error);
}

TEST_F(GmshReaderTest, VertexGIDsAssigned) {
    std::string filename = write_temp_file(QUAD_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    const auto& gids = mesh.vertex_gids();
    EXPECT_EQ(gids.size(), mesh.n_vertices());

    // GIDs should be sequential 0, 1, 2, ...
    for (size_t i = 0; i < gids.size(); ++i) {
        EXPECT_EQ(gids[i], static_cast<gid_t>(i));
    }
}

TEST_F(GmshReaderTest, CellGIDsAssigned) {
    std::string filename = write_temp_file(QUAD_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    const auto& gids = mesh.cell_gids();
    EXPECT_EQ(gids.size(), mesh.n_cells());

    for (size_t i = 0; i < gids.size(); ++i) {
        EXPECT_EQ(gids[i], static_cast<gid_t>(i));
    }
}

TEST_F(GmshReaderTest, MeshIsFinalized) {
    std::string filename = write_temp_file(QUAD_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    // Verify mesh was properly finalized by checking it has valid data
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_GT(mesh.n_cells(), 0);
}

TEST_F(GmshReaderTest, ReadViaOptions) {
    std::string filename = write_temp_file(TRIANGLE_MESH_V2);

    MeshIOOptions opts;
    opts.path = filename;

    MeshBase mesh = GmshReader::read(opts);

    EXPECT_EQ(mesh.n_cells(), 4);
}

TEST_F(GmshReaderTest, CoordinatesAreCorrect) {
    std::string filename = write_temp_file(TET_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    const auto& coords = mesh.X_ref();

    // First vertex should be at origin
    EXPECT_NEAR(coords[0], 0.0, 1e-10);
    EXPECT_NEAR(coords[1], 0.0, 1e-10);
    EXPECT_NEAR(coords[2], 0.0, 1e-10);

    // Second vertex at (1, 0, 0)
    EXPECT_NEAR(coords[3], 1.0, 1e-10);
    EXPECT_NEAR(coords[4], 0.0, 1e-10);
    EXPECT_NEAR(coords[5], 0.0, 1e-10);
}

TEST_F(GmshReaderTest, ConnectivityIsValid) {
    std::string filename = write_temp_file(QUAD_MESH_V2);

    MeshBase mesh = GmshReader::read(filename);

    // Check that all connectivity indices are valid
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        auto [verts, n_verts] = mesh.cell_vertices_span(c);
        EXPECT_EQ(n_verts, 4);  // All quads

        for (size_t i = 0; i < n_verts; ++i) {
            EXPECT_GE(verts[i], 0);
            EXPECT_LT(verts[i], static_cast<index_t>(mesh.n_vertices()));
        }
    }
}

} // namespace test
} // namespace svmp
