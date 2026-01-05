/**
 * @file test_MeshAccess_HighOrder.cpp
 * @brief Unit tests for Assembly MeshAccess adapter with high-order elements
 */

#include <gtest/gtest.h>

#include "Assembly/MeshAccess.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::MeshAccess;

namespace {

Mesh build_single_tetra10_mesh() {
    auto base = std::make_shared<MeshBase>();
    auto& mesh = *base;

    // Single Tetra10 cell
    // Nodes 0-3: Corners
    // Nodes 4-9: Mid-edge nodes
    // Standard VTK/SimVascular ordering for Tetra10:
    // 0, 1, 2, 3 (corners)
    // 4: 0-1
    // 5: 1-2
    // 6: 2-0
    // 7: 0-3
    // 8: 1-3
    // 9: 2-3
    
    // Coordinates (Reference Tetrahedron)
    // 0: (0,0,0)
    // 1: (1,0,0)
    // 2: (0,1,0)
    // 3: (0,0,1)
    // 4: (0.5, 0, 0)
    // 5: (0.5, 0.5, 0)
    // 6: (0, 0.5, 0)
    // 7: (0, 0, 0.5)
    // 8: (0.5, 0, 0.5)
    // 9: (0, 0.5, 0.5)
    
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, // 2
        0.0, 0.0, 1.0, // 3
        0.5, 0.0, 0.0, // 4
        0.5, 0.5, 0.0, // 5
        0.0, 0.5, 0.0, // 6
        0.0, 0.0, 0.5, // 7
        0.5, 0.0, 0.5, // 8
        0.0, 0.5, 0.5  // 9
    };

    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 10};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    std::vector<CellShape> cell_shapes(1);
    cell_shapes[0].family = CellFamily::Tetra;
    cell_shapes[0].order = 2; // Quadratic
    cell_shapes[0].num_corners = 10;

    mesh.build_from_arrays(/*spatial_dim=*/3, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);
    
    // Finalize to build faces/connectivity
    // Note: Tetra10 faces are technically Triangle6 (6 nodes).
    // svmp::Mesh might auto-generate faces or require explicit input.
    // For this test, we rely on build_from_arrays and finalize() default behavior if possible.
    // However, if default finalize doesn't create high-order faces, we might check just the cell info.
    mesh.finalize();
    return Mesh(std::move(base), svmp::MeshComm::self());
}

} // namespace

TEST(MeshAccess, SingleTetra10CellNodes) {
    auto mesh = build_single_tetra10_mesh();
    MeshAccess access(mesh);

    EXPECT_EQ(access.dimension(), 3);
    EXPECT_EQ(access.numCells(), 1);
    
    // Check element type
    EXPECT_EQ(access.getCellType(0), ElementType::Tetra10);

    // Check nodes
    std::vector<GlobalIndex> nodes;
    access.getCellNodes(0, nodes);
    
    ASSERT_EQ(nodes.size(), 10u);
    std::vector<GlobalIndex> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(nodes, expected);
    
    // Check coordinates of a mid-side node (e.g., node 4 at 0.5, 0, 0)
    auto x4 = access.getNodeCoordinates(4);
    EXPECT_NEAR(x4[0], 0.5, 1e-12);
    EXPECT_NEAR(x4[1], 0.0, 1e-12);
    EXPECT_NEAR(x4[2], 0.0, 1e-12);
}
