/**
 * @file test_AuxiliaryStateIndexing.cpp
 * @brief Unit tests for AuxiliaryBlockIndexing — scope-specific entity indexing
 */

#include <gtest/gtest.h>

#include "Systems/AuxiliaryStateIndexing.h"

#include <vector>

using namespace svmp::FE::systems;

// ---------------------------------------------------------------------------
//  Global scope
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockIndexing, GlobalScope)
{
    auto idx = AuxiliaryBlockIndexing::createGlobal(3);

    EXPECT_EQ(idx.scope(), AuxiliaryStateScope::Global);
    EXPECT_EQ(idx.totalEntityCount(), 1u);
    EXPECT_EQ(idx.ownedEntityCount(), 1u);
    EXPECT_EQ(idx.ghostEntityCount(), 0u);
    EXPECT_EQ(idx.componentStride(), 3);
    EXPECT_EQ(idx.totalStorageSize(), 3u);
    EXPECT_EQ(idx.ownedStorageSize(), 3u);

    // Flat index: entity 0, component 2
    EXPECT_EQ(idx.flatIndex(0, 0), 0u);
    EXPECT_EQ(idx.flatIndex(0, 1), 1u);
    EXPECT_EQ(idx.flatIndex(0, 2), 2u);
}

// ---------------------------------------------------------------------------
//  Node scope
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockIndexing, NodeScope_Serial)
{
    auto idx = AuxiliaryBlockIndexing::createNode(
        /*n_owned=*/100, /*n_ghost=*/0, /*stride=*/4);

    EXPECT_EQ(idx.scope(), AuxiliaryStateScope::Node);
    EXPECT_EQ(idx.totalEntityCount(), 100u);
    EXPECT_EQ(idx.ownedEntityCount(), 100u);
    EXPECT_EQ(idx.ghostEntityCount(), 0u);
    EXPECT_EQ(idx.totalStorageSize(), 400u);
    EXPECT_EQ(idx.ownedStorageSize(), 400u);

    // Entity 5, component 2
    EXPECT_EQ(idx.flatIndex(5, 2), 22u); // 5*4 + 2
}

TEST(AuxiliaryBlockIndexing, NodeScope_WithGhosts)
{
    auto idx = AuxiliaryBlockIndexing::createNode(
        /*n_owned=*/80, /*n_ghost=*/20, /*stride=*/2);

    EXPECT_EQ(idx.totalEntityCount(), 100u);
    EXPECT_EQ(idx.ownedEntityCount(), 80u);
    EXPECT_EQ(idx.ghostEntityCount(), 20u);
    EXPECT_EQ(idx.totalStorageSize(), 200u);
    EXPECT_EQ(idx.ownedStorageSize(), 160u);

    // Ghost node 85 (within ghost range), component 1
    EXPECT_EQ(idx.flatIndex(85, 1), 171u); // 85*2 + 1
}

// ---------------------------------------------------------------------------
//  Cell scope
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockIndexing, CellScope)
{
    auto idx = AuxiliaryBlockIndexing::createCell(
        /*n_owned=*/500, /*stride=*/1);

    EXPECT_EQ(idx.scope(), AuxiliaryStateScope::Cell);
    EXPECT_EQ(idx.totalEntityCount(), 500u);
    EXPECT_EQ(idx.ownedEntityCount(), 500u);
    EXPECT_EQ(idx.ghostEntityCount(), 0u);
    EXPECT_EQ(idx.totalStorageSize(), 500u);

    EXPECT_EQ(idx.flatIndex(123, 0), 123u);
}

TEST(AuxiliaryBlockIndexing, CellScope_MultiComponent)
{
    auto idx = AuxiliaryBlockIndexing::createCell(200, 6);

    EXPECT_EQ(idx.totalStorageSize(), 1200u);

    // Cell 10, component 3
    EXPECT_EQ(idx.flatIndex(10, 3), 63u); // 10*6 + 3
}

// ---------------------------------------------------------------------------
//  QuadraturePoint scope
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockIndexing, QuadraturePointScope)
{
    // 3 cells: 4 QPs, 9 QPs, 4 QPs = 17 total
    const std::vector<std::size_t> qp_offsets = {0, 4, 13, 17};

    auto idx = AuxiliaryBlockIndexing::createQuadraturePoint(qp_offsets, 2);

    EXPECT_EQ(idx.scope(), AuxiliaryStateScope::QuadraturePoint);
    EXPECT_EQ(idx.totalEntityCount(), 17u);
    EXPECT_EQ(idx.componentStride(), 2);
    EXPECT_EQ(idx.totalStorageSize(), 34u);

    // Cell 0 has 4 QPs
    EXPECT_EQ(idx.qpsForCell(0), 4u);
    EXPECT_EQ(idx.qpsForCell(1), 9u);
    EXPECT_EQ(idx.qpsForCell(2), 4u);

    // Cell 0, local QP 2, component 1 → global QP = 0+2=2, flat = 2*2+1=5
    EXPECT_EQ(idx.qpFlatIndex(0, 2, 1), 5u);

    // Cell 1, local QP 0, component 0 → global QP = 4+0=4, flat = 4*2+0=8
    EXPECT_EQ(idx.qpFlatIndex(1, 0, 0), 8u);

    // Cell 2, local QP 3, component 1 → global QP = 13+3=16, flat = 16*2+1=33
    EXPECT_EQ(idx.qpFlatIndex(2, 3, 1), 33u);
}

TEST(AuxiliaryBlockIndexing, QuadraturePointScope_UniformQPs)
{
    // 5 cells × 4 QPs each = 20 total
    const std::vector<std::size_t> qp_offsets = {0, 4, 8, 12, 16, 20};

    auto idx = AuxiliaryBlockIndexing::createQuadraturePoint(qp_offsets, 1);

    EXPECT_EQ(idx.totalEntityCount(), 20u);
    EXPECT_EQ(idx.totalStorageSize(), 20u);

    for (std::size_t c = 0; c < 5; ++c) {
        EXPECT_EQ(idx.qpsForCell(c), 4u);
    }
}

// ---------------------------------------------------------------------------
//  Facet scope
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockIndexing, FacetScope)
{
    auto idx = AuxiliaryBlockIndexing::createFacet(
        /*n_boundary_entities=*/30, /*stride=*/2);

    EXPECT_EQ(idx.scope(), AuxiliaryStateScope::Facet);
    EXPECT_EQ(idx.totalEntityCount(), 30u);
    EXPECT_EQ(idx.ownedEntityCount(), 30u);
    EXPECT_EQ(idx.totalStorageSize(), 60u);

    EXPECT_EQ(idx.flatIndex(15, 1), 31u); // 15*2 + 1
}

// ---------------------------------------------------------------------------
//  Validation
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockIndexing, ZeroStrideThrows)
{
    EXPECT_THROW(AuxiliaryBlockIndexing::createGlobal(0),
                 svmp::FE::InvalidArgumentException);
    EXPECT_THROW(AuxiliaryBlockIndexing::createNode(10, 0, 0),
                 svmp::FE::InvalidArgumentException);
    EXPECT_THROW(AuxiliaryBlockIndexing::createCell(10, -1),
                 svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryBlockIndexing, QPOffsetsValidation)
{
    // Too few entries
    EXPECT_THROW(
        AuxiliaryBlockIndexing::createQuadraturePoint(
            std::vector<std::size_t>{0}, 1),
        svmp::FE::InvalidArgumentException);

    // offsets[0] != 0
    EXPECT_THROW(
        AuxiliaryBlockIndexing::createQuadraturePoint(
            std::vector<std::size_t>{1, 5}, 1),
        svmp::FE::InvalidArgumentException);
}
