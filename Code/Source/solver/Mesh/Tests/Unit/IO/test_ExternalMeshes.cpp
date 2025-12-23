/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Tests using external benchmark meshes from MFEM and deal.II repositories.
 */

#include <gtest/gtest.h>
#include "../../../IO/GmshReader.h"
#include "../../../IO/MFEMReader.h"
#include "../../../Core/MeshBase.h"
#include "../../../Core/MeshTypes.h"
#include "../../../Topology/CellShape.h"

#include <filesystem>
#include <fstream>
#include <set>

namespace svmp {
namespace test {

// ==========================================
// Test Fixture
// ==========================================

class ExternalMeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Path to test data directory (relative to build dir)
        data_dir_ = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "Data";
        mfem_dir_ = data_dir_ / "MFEM";
        dealii_dir_ = data_dir_ / "dealII";
        gmsh_dir_ = mfem_dir_ / "gmsh";
    }

    bool file_exists(const std::filesystem::path& path) {
        return std::filesystem::exists(path);
    }

    std::filesystem::path data_dir_;
    std::filesystem::path mfem_dir_;
    std::filesystem::path dealii_dir_;
    std::filesystem::path gmsh_dir_;
};

// ==========================================
// MFEM Native Format Tests
// ==========================================

TEST_F(ExternalMeshTest, MFEM_StarMesh) {
    auto path = mfem_dir_ / "star.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Star mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    // Star mesh: 20 quads, 31 vertices, topologically 2D (stored in 3D coords)
    EXPECT_EQ(mesh.n_cells(), 20);
    EXPECT_EQ(mesh.n_vertices(), 31);
    // All cells should be quads (2D elements)
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_EQ(mesh.cell_shape(i).family, CellFamily::Quad);
    }
}

TEST_F(ExternalMeshTest, MFEM_LShapeMesh) {
    auto path = mfem_dir_ / "l-shape.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "L-shape mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    // Topologically 2D mesh - verify cells are 2D
    EXPECT_TRUE(mesh.cell_shape(0).is_2d());
}

TEST_F(ExternalMeshTest, MFEM_BeamHex) {
    auto path = mfem_dir_ / "beam-hex.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Beam hex mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_EQ(mesh.dim(), 3);

    // Verify all cells are hexahedra
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_EQ(mesh.cell_shape(i).family, CellFamily::Hex);
    }
}

TEST_F(ExternalMeshTest, MFEM_BeamTet) {
    auto path = mfem_dir_ / "beam-tet.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Beam tet mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_EQ(mesh.dim(), 3);

    // Verify all cells are tetrahedra
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_EQ(mesh.cell_shape(i).family, CellFamily::Tetra);
    }
}

TEST_F(ExternalMeshTest, MFEM_BeamWedge) {
    auto path = mfem_dir_ / "beam-wedge.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Beam wedge mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_EQ(mesh.dim(), 3);

    // Verify all cells are wedges/prisms
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_EQ(mesh.cell_shape(i).family, CellFamily::Wedge);
    }
}

TEST_F(ExternalMeshTest, MFEM_FicheraCorner) {
    auto path = mfem_dir_ / "fichera.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Fichera mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    // Fichera corner is an L-shaped 3D domain
    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_EQ(mesh.dim(), 3);
}

TEST_F(ExternalMeshTest, MFEM_FicheraMixed) {
    auto path = mfem_dir_ / "fichera-mixed.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Fichera mixed mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_EQ(mesh.dim(), 3);

    // Should have multiple element types - collect unique families
    std::set<CellFamily> families;
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        families.insert(mesh.cell_shape(i).family);
    }
    EXPECT_GT(families.size(), 1) << "Mixed mesh should have multiple element types";
}

TEST_F(ExternalMeshTest, MFEM_InlinePyramid) {
    // Skip - MFEM INLINE format is a parametric generator, not full mesh data
    // Would require a separate parser to expand the inline mesh description
    GTEST_SKIP() << "MFEM INLINE format not yet supported";
}

TEST_F(ExternalMeshTest, MFEM_Escher) {
    auto path = mfem_dir_ / "escher.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Escher mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    // Escher is a 3D tetrahedral mesh (42 tets)
    EXPECT_EQ(mesh.n_cells(), 42);
    EXPECT_GT(mesh.n_vertices(), 0);
    // Verify all cells are tetrahedra
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_EQ(mesh.cell_shape(i).family, CellFamily::Tetra);
    }
}

TEST_F(ExternalMeshTest, MFEM_ToroidHex) {
    auto path = mfem_dir_ / "toroid-hex.mesh";
    if (!file_exists(path)) GTEST_SKIP() << "Toroid hex mesh not found: " << path;

    MeshBase mesh = MFEMReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_EQ(mesh.dim(), 3);

    // All hexes
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_EQ(mesh.cell_shape(i).family, CellFamily::Hex);
    }
}

TEST_F(ExternalMeshTest, MFEM_EquilateralElements) {
    // Test equilateral reference elements
    std::vector<std::pair<std::string, CellFamily>> elements = {
        {"equilateral-triangle.mesh", CellFamily::Triangle},
        {"equilateral-tetrahedron.mesh", CellFamily::Tetra},
        {"equilateral-prism.mesh", CellFamily::Wedge},
        {"equilateral-pyramid.mesh", CellFamily::Pyramid}
    };

    for (const auto& [filename, expected_family] : elements) {
        auto path = mfem_dir_ / filename;
        if (!file_exists(path)) continue;

        MeshBase mesh = MFEMReader::read(path.string());

        EXPECT_EQ(mesh.n_cells(), 1) << "Expected single element in " << filename;
        EXPECT_EQ(mesh.cell_shape(0).family, expected_family) << "Shape mismatch in " << filename;
    }
}

// ==========================================
// Gmsh Format Tests (from MFEM data)
// ==========================================

TEST_F(ExternalMeshTest, Gmsh_UnstructuredQuad) {
    auto path = gmsh_dir_ / "unstructured_quad.v22.msh";
    if (!file_exists(path)) GTEST_SKIP() << "Gmsh unstructured quad not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
}

TEST_F(ExternalMeshTest, Gmsh_UnstructuredHex) {
    auto path = gmsh_dir_ / "unstructured_hex.v22.msh";
    if (!file_exists(path)) GTEST_SKIP() << "Gmsh unstructured hex not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    EXPECT_EQ(mesh.dim(), 3);
}

TEST_F(ExternalMeshTest, Gmsh_Surfaces3D) {
    auto path = gmsh_dir_ / "surfaces_in_3d.asc.v22.msh";
    if (!file_exists(path)) GTEST_SKIP() << "Gmsh surfaces in 3D not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
}

TEST_F(ExternalMeshTest, Gmsh_Piece) {
    auto path = gmsh_dir_ / "piece.asc.v22.msh";
    if (!file_exists(path)) GTEST_SKIP() << "Gmsh piece mesh not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    // This is a larger mesh
    EXPECT_GT(mesh.n_cells(), 100);
    EXPECT_GT(mesh.n_vertices(), 100);
}

TEST_F(ExternalMeshTest, Gmsh_Indheat) {
    auto path = gmsh_dir_ / "indheat.asc.v22.msh";
    if (!file_exists(path)) GTEST_SKIP() << "Gmsh indheat mesh not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    // Large industrial mesh
    EXPECT_GT(mesh.n_cells(), 1000);
    EXPECT_GT(mesh.n_vertices(), 1000);
}

// ==========================================
// deal.II Gmsh Format Tests
// ==========================================

TEST_F(ExternalMeshTest, DealII_UnitSquare) {
    auto path = dealii_dir_ / "unit-square.msh";
    if (!file_exists(path)) GTEST_SKIP() << "deal.II unit square not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    // Topologically 2D mesh
    EXPECT_TRUE(mesh.cell_shape(0).is_2d());
}

TEST_F(ExternalMeshTest, DealII_LShape) {
    auto path = dealii_dir_ / "lshaped.msh";
    if (!file_exists(path)) GTEST_SKIP() << "deal.II L-shape not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
    // Topologically 2D mesh
    EXPECT_TRUE(mesh.cell_shape(0).is_2d());
}

TEST_F(ExternalMeshTest, DealII_SquareQuad) {
    auto path = dealii_dir_ / "square-quad.msh";
    if (!file_exists(path)) GTEST_SKIP() << "deal.II square quad not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);

    // All quads (2D elements)
    for (size_t i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_EQ(mesh.cell_shape(i).family, CellFamily::Quad);
    }
}

TEST_F(ExternalMeshTest, DealII_MeshFromStep49) {
    auto path = dealii_dir_ / "mesh-from-step49.msh";
    if (!file_exists(path)) GTEST_SKIP() << "deal.II step49 mesh not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    EXPECT_GT(mesh.n_cells(), 0);
    EXPECT_GT(mesh.n_vertices(), 0);
}

TEST_F(ExternalMeshTest, DealII_UnstructuredRefinedBall) {
    auto path = dealii_dir_ / "unstructured_refined_ball.msh";
    if (!file_exists(path)) GTEST_SKIP() << "deal.II refined ball not found: " << path;

    MeshBase mesh = GmshReader::read(path.string());

    // 3D unstructured mesh
    EXPECT_GT(mesh.n_cells(), 100);
    EXPECT_GT(mesh.n_vertices(), 100);
    EXPECT_EQ(mesh.dim(), 3);
}

// ==========================================
// Mesh Quality Validation Tests
// ==========================================

TEST_F(ExternalMeshTest, ValidateMeshTopology) {
    // Test a selection of meshes for valid topology
    std::vector<std::pair<std::filesystem::path, std::string>> meshes = {
        {mfem_dir_ / "star.mesh", "MFEM"},
        {mfem_dir_ / "beam-hex.mesh", "MFEM"},
        {dealii_dir_ / "unit-square.msh", "Gmsh"},
    };

    for (const auto& [path, format] : meshes) {
        if (!file_exists(path)) continue;

        MeshBase mesh = (format == "MFEM") ?
            MFEMReader::read(path.string()) :
            GmshReader::read(path.string());

        // Validate all vertices are referenced
        std::vector<bool> vertex_used(mesh.n_vertices(), false);
        for (size_t i = 0; i < mesh.n_cells(); ++i) {
            auto [ptr, len] = mesh.cell_vertices_span(i);
            for (size_t j = 0; j < len; ++j) {
                auto v = ptr[j];
                ASSERT_LT(static_cast<size_t>(v), mesh.n_vertices()) << "Invalid vertex index in " << path;
                vertex_used[v] = true;
            }
        }

        // Count used vertices
        size_t used_count = std::count(vertex_used.begin(), vertex_used.end(), true);
        EXPECT_EQ(used_count, mesh.n_vertices())
            << "Not all vertices used in " << path;
    }
}

TEST_F(ExternalMeshTest, ValidateGIDs) {
    auto path = mfem_dir_ / "star.mesh";
    if (!file_exists(path)) GTEST_SKIP();

    MeshBase mesh = MFEMReader::read(path.string());

    // Check vertex GIDs are sequential
    auto vertex_gids = mesh.vertex_gids();
    for (size_t i = 0; i < vertex_gids.size(); ++i) {
        EXPECT_EQ(vertex_gids[i], static_cast<gid_t>(i));
    }

    // Check cell GIDs are sequential
    auto cell_gids = mesh.cell_gids();
    for (size_t i = 0; i < cell_gids.size(); ++i) {
        EXPECT_EQ(cell_gids[i], static_cast<gid_t>(i));
    }
}

} // namespace test
} // namespace svmp
