/**
 * @file test_MeshTopologyBuilder.cpp
 * @brief Unit tests for MeshTopologyBuilder utilities
 */

#include <gtest/gtest.h>

#include "FE/Dofs/MeshTopologyBuilder.h"

using svmp::FE::MeshIndex;
using svmp::FE::MeshOffset;
using svmp::FE::dofs::buildCellToEdgesRefOrder;
using svmp::FE::dofs::buildCellToFacesRefOrder;

TEST(MeshTopologyBuilder, CellToEdgesQuadMapsToReferenceOrder) {
    const std::vector<MeshOffset> cell2vertex_offsets = {0, 4};
    const std::vector<MeshIndex> cell2vertex = {0, 1, 2, 3};

    // Intentionally shuffled edge IDs.
    const std::vector<std::array<MeshIndex, 2>> edge2vertex = {
        {1, 2}, // id 0
        {0, 1}, // id 1
        {3, 0}, // id 2
        {2, 3}  // id 3
    };

    const auto csr = buildCellToEdgesRefOrder(/*dim=*/2,
                                              cell2vertex_offsets,
                                              cell2vertex,
                                              edge2vertex);

    ASSERT_EQ(csr.offsets.size(), 2u);
    ASSERT_EQ(csr.data.size(), 4u);
    EXPECT_EQ(csr.offsets[0], 0);
    EXPECT_EQ(csr.offsets[1], 4);

    // Quad reference edge order: (0-1), (1-2), (2-3), (3-0).
    EXPECT_EQ(csr.data, (std::vector<MeshIndex>{1, 0, 3, 2}));
}

TEST(MeshTopologyBuilder, CellToFacesHexMapsToReferenceOrder) {
    const std::vector<MeshOffset> cell2vertex_offsets = {0, 8};
    const std::vector<MeshIndex> cell2vertex = {0, 1, 2, 3, 4, 5, 6, 7};

    // Intentionally shuffled face IDs and rotated vertex orderings.
    // Expected Hex8 reference face order:
    //  0: (0,1,2,3)
    //  1: (4,5,6,7)
    //  2: (0,1,5,4)
    //  3: (1,2,6,5)
    //  4: (2,3,7,6)
    //  5: (3,0,4,7)
    const std::vector<MeshOffset> face2vertex_offsets = {0, 4, 8, 12, 16, 20, 24};
    const std::vector<MeshIndex> face2vertex = {
        7, 6, 2, 3, // id 0: face 4 (rotated/reversed)
        5, 4, 0, 1, // id 1: face 2 (rotated)
        0, 3, 7, 4, // id 2: face 5 (rotated)
        6, 7, 4, 5, // id 3: face 1 (rotated)
        2, 1, 5, 6, // id 4: face 3 (rotated/reversed)
        1, 0, 3, 2  // id 5: face 0 (rotated)
    };

    const auto csr = buildCellToFacesRefOrder(/*dim=*/3,
                                              cell2vertex_offsets,
                                              cell2vertex,
                                              face2vertex_offsets,
                                              face2vertex);

    ASSERT_EQ(csr.offsets.size(), 2u);
    ASSERT_EQ(csr.data.size(), 6u);
    EXPECT_EQ(csr.offsets[0], 0);
    EXPECT_EQ(csr.offsets[1], 6);

    EXPECT_EQ(csr.data, (std::vector<MeshIndex>{5, 3, 1, 4, 0, 2}));
}

TEST(MeshTopologyBuilder, CellToFacesTriangle2DMapsToReferenceOrder) {
    const std::vector<MeshOffset> cell2vertex_offsets = {0, 3};
    const std::vector<MeshIndex> cell2vertex = {0, 1, 2};

    // Intentionally shuffled face IDs and reversed edge orderings.
    // Triangle3 reference face/edge order: (0-1), (1-2), (2-0).
    const std::vector<MeshOffset> face2vertex_offsets = {0, 2, 4, 6};
    const std::vector<MeshIndex> face2vertex = {
        2, 1, // id 0: edge (1,2)
        0, 2, // id 1: edge (2,0)
        1, 0  // id 2: edge (0,1) reversed
    };

    const auto csr = buildCellToFacesRefOrder(/*dim=*/2,
                                              cell2vertex_offsets,
                                              cell2vertex,
                                              face2vertex_offsets,
                                              face2vertex);

    ASSERT_EQ(csr.offsets.size(), 2u);
    ASSERT_EQ(csr.data.size(), 3u);
    EXPECT_EQ(csr.offsets[0], 0);
    EXPECT_EQ(csr.offsets[1], 3);

    EXPECT_EQ(csr.data, (std::vector<MeshIndex>{2, 0, 1}));
}
