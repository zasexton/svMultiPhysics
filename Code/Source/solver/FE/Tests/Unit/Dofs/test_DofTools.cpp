/**
 * @file test_DofTools.cpp
 * @brief Unit tests for DofTools utilities
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofTools.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Dofs/EntityDofMap.h"

#include <algorithm>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::dofs::ComponentMask;
using svmp::FE::dofs::DofExtractionOptions;
using svmp::FE::dofs::EntityDofMap;
using svmp::FE::dofs::EntityKind;
using svmp::FE::dofs::FieldMask;
using svmp::FE::dofs::IndexSet;
using svmp::FE::dofs::DofHandler;
using svmp::FE::dofs::DofLayoutInfo;
using svmp::FE::dofs::MeshTopologyInfo;
using namespace svmp::FE::dofs::DofTools;

TEST(ComponentMask, BasicSelection) {
    ComponentMask mask = ComponentMask::none();
    mask.setSize(3);
    EXPECT_TRUE(mask.selectsNone());

    mask.select(1);
    EXPECT_TRUE(mask.isSelected(1));
    EXPECT_FALSE(mask.isSelected(0));
    EXPECT_FALSE(mask.selectsAll());
}

TEST(FieldMask, BasicSelection) {
    FieldMask mask = FieldMask::none();
    EXPECT_EQ(mask.numSelected(), 0u);

    mask.select(0);
    EXPECT_TRUE(mask.isSelected(0));
    EXPECT_FALSE(mask.isSelected(1));
}

TEST(DofTools, ExtractBoundaryFaceDofs) {
    EntityDofMap entity;
    entity.reserve(/*v=*/4, /*e=*/0, /*f=*/3, /*c=*/0);
    entity.setVertexDofs(0, std::vector<GlobalIndex>{0});
    entity.setVertexDofs(1, std::vector<GlobalIndex>{1});
    entity.setVertexDofs(2, std::vector<GlobalIndex>{2});
    entity.setVertexDofs(3, std::vector<GlobalIndex>{3});
    entity.setFaceDofs(0, std::vector<GlobalIndex>{10});
    entity.setFaceDofs(1, std::vector<GlobalIndex>{11});
    entity.setFaceDofs(2, std::vector<GlobalIndex>{12});
    entity.finalize();

    const std::vector<int> face_labels = {1, 2, 1};
    const std::vector<GlobalIndex> face2vertex_offsets = {0, 3, 6, 9};
    const std::vector<GlobalIndex> face2vertex_data = {
        0, 1, 2,  // face 0
        0, 2, 3,  // face 1
        0, 3, 1   // face 2
    };
    const std::vector<GlobalIndex> edge2vertex_data; // no edge DOFs in this setup

    // Boundary id 1 should include vertex DOFs plus face-interior DOFs from faces 0 and 2.
    auto dofs = extractBoundaryDofs(entity,
                                   /*boundary_id=*/1,
                                   face_labels,
                                   face2vertex_offsets,
                                   face2vertex_data,
                                   edge2vertex_data,
                                   DofExtractionOptions{});
    EXPECT_EQ(dofs, (std::vector<GlobalIndex>{0, 1, 2, 3, 10, 12}));
}

TEST(DofTools, ExtractBoundaryEdgeDofsIncludesVerticesAndEdges) {
    EntityDofMap entity;
    entity.reserve(/*v=*/3, /*e=*/3, /*f=*/0, /*c=*/0);
    entity.setVertexDofs(0, std::vector<GlobalIndex>{0});
    entity.setVertexDofs(1, std::vector<GlobalIndex>{1});
    entity.setVertexDofs(2, std::vector<GlobalIndex>{2});
    entity.setEdgeDofs(0, std::vector<GlobalIndex>{10});
    entity.setEdgeDofs(1, std::vector<GlobalIndex>{11});
    entity.setEdgeDofs(2, std::vector<GlobalIndex>{12});
    entity.finalize();

    // 2D facets are edges (2 vertices each).
    const std::vector<int> edge_labels = {1, 0, 1};
    const std::vector<GlobalIndex> edge2vertex_offsets = {0, 2, 4, 6};
    const std::vector<GlobalIndex> edge2vertex_data = {
        0, 1, // edge 0
        1, 2, // edge 1
        2, 0  // edge 2
    };

    auto dofs = extractBoundaryDofs(entity,
                                   /*boundary_id=*/1,
                                   edge_labels,
                                   edge2vertex_offsets,
                                   edge2vertex_data,
                                   edge2vertex_data,
                                   DofExtractionOptions{});

    EXPECT_EQ(dofs, (std::vector<GlobalIndex>{0, 1, 2, 10, 12}));
}

TEST(DofTools, ExtractBoundaryDofsComponentMask_BlockLayout) {
    EntityDofMap entity;
    entity.reserve(/*v=*/2, /*e=*/0, /*f=*/0, /*c=*/0);
    // Block-by-component, stride=2:
    //  vertex0: c0=0, c1=2
    //  vertex1: c0=1, c1=3
    entity.setVertexDofs(0, std::vector<GlobalIndex>{0, 2});
    entity.setVertexDofs(1, std::vector<GlobalIndex>{1, 3});
    entity.finalize();

    const std::vector<int> edge_labels = {1};
    const std::vector<GlobalIndex> edge2vertex_offsets = {0, 2};
    const std::vector<GlobalIndex> edge2vertex_data = {0, 1};

    const auto dofs = extractBoundaryDofs(entity,
                                         /*boundary_id=*/1,
                                         edge_labels,
                                         edge2vertex_offsets,
                                         edge2vertex_data,
                                         /*edge2vertex_data=*/edge2vertex_data,
                                         ComponentMask::component(0),
                                         /*n_components=*/2,
                                         DofExtractionOptions{});

    EXPECT_EQ(dofs, (std::vector<GlobalIndex>{0, 1}));
}

TEST(DofTools, ExtractBoundaryDofsComponentMask_InterleavedLayout) {
    EntityDofMap entity;
    entity.reserve(/*v=*/2, /*e=*/0, /*f=*/0, /*c=*/0);
    // Interleaved-by-node for 2 components:
    //  vertex0: c0=0, c1=1
    //  vertex1: c0=2, c1=3
    entity.setVertexDofs(0, std::vector<GlobalIndex>{0, 1});
    entity.setVertexDofs(1, std::vector<GlobalIndex>{2, 3});
    entity.finalize();

    const std::vector<int> edge_labels = {1};
    const std::vector<GlobalIndex> edge2vertex_offsets = {0, 2};
    const std::vector<GlobalIndex> edge2vertex_data = {0, 1};

    const auto dofs = extractBoundaryDofs(entity,
                                         /*boundary_id=*/1,
                                         edge_labels,
                                         edge2vertex_offsets,
                                         edge2vertex_data,
                                         /*edge2vertex_data=*/edge2vertex_data,
                                         ComponentMask::component(0),
                                         /*n_components=*/2,
                                         DofExtractionOptions{});

    EXPECT_EQ(dofs, (std::vector<GlobalIndex>{0, 2}));
}

TEST(DofTools, ExtractEntityDofs) {
    EntityDofMap entity;
    entity.reserve(/*v=*/2, /*e=*/1, /*f=*/0, /*c=*/0);
    entity.setVertexDofs(0, std::vector<GlobalIndex>{0});
    entity.setVertexDofs(1, std::vector<GlobalIndex>{1});
    entity.setEdgeDofs(0, std::vector<GlobalIndex>{10});
    entity.finalize();

    const std::vector<GlobalIndex> vids = {0, 1};
    auto vdofs = extractEntityDofs(entity, EntityKind::Vertex, vids);
    EXPECT_EQ(vdofs, (std::vector<GlobalIndex>{0, 1}));
}

TEST(DofTools, ExtractInterfaceDofsExcludesCellInterior) {
    EntityDofMap entity;
    entity.reserve(/*v=*/2, /*e=*/1, /*f=*/1, /*c=*/1);
    entity.setVertexDofs(0, std::vector<GlobalIndex>{0});
    entity.setVertexDofs(1, std::vector<GlobalIndex>{1});
    entity.setEdgeDofs(0, std::vector<GlobalIndex>{10});
    entity.setFaceDofs(0, std::vector<GlobalIndex>{20});
    entity.setCellInteriorDofs(0, std::vector<GlobalIndex>{30});
    entity.finalize();

    IndexSet iface = extractInterfaceDofs(entity);
    EXPECT_TRUE(iface.contains(0));
    EXPECT_TRUE(iface.contains(10));
    EXPECT_TRUE(iface.contains(20));
    EXPECT_FALSE(iface.contains(30));
}

TEST(DofTools, GetDofSupportEntitiesVertexIncludesEdgesAndCell) {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 4;
    topo.dim = 2;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell2edge_offsets = {0, 4};
    topo.cell2edge_data = {0, 1, 2, 3};
    topo.edge2vertex_data = {0, 1, 1, 2, 2, 3, 3, 0};

    DofHandler handler;
    auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/4);
    handler.distributeDofs(topo, layout);
    handler.finalize();

    const auto* edm = handler.getEntityDofMap();
    ASSERT_NE(edm, nullptr);

    const auto support = getDofSupportEntities(/*dof_id=*/0, *edm, topo);
    auto has = [&](EntityKind kind, GlobalIndex id) {
        return std::find(support.begin(), support.end(),
                         svmp::FE::dofs::EntityRef{kind, id}) != support.end();
    };

    EXPECT_TRUE(has(EntityKind::Vertex, 0));
    EXPECT_TRUE(has(EntityKind::Edge, 0));
    EXPECT_TRUE(has(EntityKind::Edge, 3));
    EXPECT_TRUE(has(EntityKind::Cell, 0));
}
