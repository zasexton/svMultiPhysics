/**
 * @file test_EntityDofMap.cpp
 * @brief Unit tests for EntityDofMap
 */

#include <gtest/gtest.h>

#include "FE/Dofs/EntityDofMap.h"
#include "FE/Core/FEException.h"

#include <vector>
#include <algorithm>

using svmp::FE::FEException;
using svmp::FE::GlobalIndex;
using svmp::FE::dofs::EntityDofMap;
using svmp::FE::dofs::EntityKind;

TEST(EntityDofMap, DefaultConstruction) {
    EntityDofMap map;
    EXPECT_FALSE(map.isFinalized());
    EXPECT_FALSE(map.hasReverseMapping());
    EXPECT_EQ(map.numVertices(), 0);
}

TEST(EntityDofMap, SetAndQueryEntityDofs) {
    EntityDofMap map;
    map.reserve(/*n_vertices=*/3, /*n_edges=*/2, /*n_faces=*/1, /*n_cells=*/1);

    map.setVertexDofs(0, std::vector<GlobalIndex>{0});
    map.setVertexDofs(1, std::vector<GlobalIndex>{1});
    map.setVertexDofs(2, std::vector<GlobalIndex>{2});

    map.setEdgeDofs(0, std::vector<GlobalIndex>{10, 11});
    map.setEdgeDofs(1, std::vector<GlobalIndex>{12, 13});

    map.setFaceDofs(0, std::vector<GlobalIndex>{20});
    map.setCellInteriorDofs(0, std::vector<GlobalIndex>{30});

    map.buildReverseMapping();
    map.finalize();

    EXPECT_TRUE(map.isFinalized());
    EXPECT_TRUE(map.hasReverseMapping());

    auto v0 = map.getVertexDofs(0);
    ASSERT_EQ(v0.size(), 1u);
    EXPECT_EQ(v0[0], 0);

    auto e0 = map.getEdgeDofs(0);
    ASSERT_EQ(e0.size(), 2u);
    EXPECT_EQ(e0[0], 10);
    EXPECT_EQ(e0[1], 11);

    auto f0 = map.getFaceDofs(0);
    ASSERT_EQ(f0.size(), 1u);
    EXPECT_EQ(f0[0], 20);

    auto c0 = map.getCellInteriorDofs(0);
    ASSERT_EQ(c0.size(), 1u);
    EXPECT_EQ(c0[0], 30);

    // Generic entity accessor
    auto e0_generic = map.getEntityDofs(EntityKind::Edge, 0);
    EXPECT_EQ(std::vector<GlobalIndex>(e0_generic.begin(), e0_generic.end()),
              std::vector<GlobalIndex>({10, 11}));

    // Reverse lookup
    auto ent = map.getDofEntity(12);
    ASSERT_TRUE(ent.has_value());
    EXPECT_EQ(ent->kind, EntityKind::Edge);
    EXPECT_EQ(ent->id, 1);
}

TEST(EntityDofMap, ReverseLookupRequiresBuildReverseMapping) {
    EntityDofMap map;
    map.reserve(1, 0, 0, 0);
    map.setVertexDofs(0, std::vector<GlobalIndex>{0});
    map.finalize();

    EXPECT_THROW(map.getDofEntity(0), FEException);
}

TEST(EntityDofMap, InterfaceDofsExcludeCellInterior) {
    EntityDofMap map;
    map.reserve(2, 1, 1, 1);

    map.setVertexDofs(0, std::vector<GlobalIndex>{0});
    map.setVertexDofs(1, std::vector<GlobalIndex>{1});
    map.setEdgeDofs(0, std::vector<GlobalIndex>{10});
    map.setFaceDofs(0, std::vector<GlobalIndex>{20});
    map.setCellInteriorDofs(0, std::vector<GlobalIndex>{30});

    map.buildReverseMapping();
    map.finalize();

    auto iface = map.getInterfaceDofs();
    EXPECT_TRUE(std::find(iface.begin(), iface.end(), GlobalIndex{0}) != iface.end());
    EXPECT_TRUE(std::find(iface.begin(), iface.end(), GlobalIndex{10}) != iface.end());
    EXPECT_TRUE(std::find(iface.begin(), iface.end(), GlobalIndex{20}) != iface.end());
    EXPECT_TRUE(std::find(iface.begin(), iface.end(), GlobalIndex{30}) == iface.end());
}
