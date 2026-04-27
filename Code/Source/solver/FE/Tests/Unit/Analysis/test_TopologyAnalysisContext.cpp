/**
 * @file test_TopologyAnalysisContext.cpp
 * @brief Unit tests for TopologyAnalysisContext — mesh topology analysis
 */

#include <gtest/gtest.h>

#include "Analysis/TopologyAnalysisContext.h"
#include "Analysis/InterfaceTopologyContext.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Assembly/Assembler.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::assembly;

// ============================================================================
// Mock mesh: single tetrahedron
// ============================================================================

class SingleTetraMesh final : public IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex, std::vector<GlobalIndex>& nodes) const override {
        nodes = {0, 1, 2, 3};
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& c) const override {
        c = {{{{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0,0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}
};

// ============================================================================
// Mock mesh: two disconnected tetrahedra (no shared nodes)
// ============================================================================

class TwoDisconnectedTetraMesh final : public IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& c) const override {
        c = {{{{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0,0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}
};

// ============================================================================
// Mock mesh: two connected tetrahedra sharing a face (nodes 1,2,3 shared)
// ============================================================================

class TwoConnectedTetraMesh final : public IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {1, 2, 3, 4};  // shares nodes 1,2,3 with cell 0
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& c) const override {
        c = {{{{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0,1}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}
};

// ============================================================================
// Mock mesh: two disconnected tetra with boundary markers
//   - Face 0 on cell 0, marker 1
//   - Face 1 on cell 1, marker 2
// ============================================================================

class TwoTetraWithBoundaryMarkersMesh final : public IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& c) const override {
        c = {{{{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override {
        if (face_id == 0) return 1;
        if (face_id == 1) return 2;
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0,0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }

    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> cb) const override {
        if (marker < 0) {
            cb(0, 0);
            cb(1, 1);
            return;
        }
        if (marker == 1) cb(0, 0);  // face 0 on cell 0
        if (marker == 2) cb(1, 1);  // face 1 on cell 1
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}
};

class SparseBoundaryFaceIdMesh final : public IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& c) const override {
        c = {{{{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override {
        if (face_id == 10) return 7;
        if (face_id == 42) return 9;
        return -1;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0,0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> cb) const override {
        if (marker < 0 || marker == 7) cb(10, 0);
        if (marker < 0 || marker == 9) cb(42, 1);
    }
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}
};

// ============================================================================
// Tests
// ============================================================================

TEST(TopologyAnalysisContext, SingleTetra_OneComponent) {
    SingleTetraMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    EXPECT_EQ(ctx.numRegions(), 1);
    EXPECT_FALSE(ctx.isDisconnected());
    ASSERT_EQ(ctx.components.size(), 1u);
    EXPECT_EQ(ctx.components[0].num_cells, 1);
    EXPECT_EQ(ctx.components[0].num_vertices, 4);
    EXPECT_EQ(ctx.regionForCell(0), 0);
    EXPECT_EQ(ctx.regionForCell(1), -1);  // out of range
}

TEST(TopologyAnalysisContext, TwoDisconnected_TwoComponents) {
    TwoDisconnectedTetraMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    EXPECT_EQ(ctx.numRegions(), 2);
    EXPECT_TRUE(ctx.isDisconnected());
    ASSERT_EQ(ctx.components.size(), 2u);

    // Each component has 1 cell and 4 vertices
    EXPECT_EQ(ctx.components[0].num_cells, 1);
    EXPECT_EQ(ctx.components[0].num_vertices, 4);
    EXPECT_EQ(ctx.components[1].num_cells, 1);
    EXPECT_EQ(ctx.components[1].num_vertices, 4);

    // Cells in different regions
    int r0 = ctx.regionForCell(0);
    int r1 = ctx.regionForCell(1);
    EXPECT_NE(r0, r1);
    EXPECT_GE(r0, 0);
    EXPECT_GE(r1, 0);
}

TEST(TopologyAnalysisContext, TwoConnected_OneComponent) {
    TwoConnectedTetraMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    EXPECT_EQ(ctx.numRegions(), 1);
    EXPECT_FALSE(ctx.isDisconnected());

    // Both cells in same region
    EXPECT_EQ(ctx.regionForCell(0), ctx.regionForCell(1));

    // 5 unique vertices (0,1,2,3,4)
    EXPECT_EQ(ctx.components[0].num_vertices, 5);
    EXPECT_EQ(ctx.components[0].num_cells, 2);
}

TEST(TopologyAnalysisContext, BoundaryMarkers_MappedToRegions) {
    TwoTetraWithBoundaryMarkersMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    ASSERT_EQ(ctx.numRegions(), 2);

    // Marker 1 is on cell 0 (region A), marker 2 is on cell 1 (region B)
    auto regions_m1 = ctx.regionsForBoundaryMarker(1);
    ASSERT_EQ(regions_m1.size(), 1u);
    EXPECT_EQ(regions_m1[0], ctx.regionForCell(0));

    auto regions_m2 = ctx.regionsForBoundaryMarker(2);
    ASSERT_EQ(regions_m2.size(), 1u);
    EXPECT_EQ(regions_m2[0], ctx.regionForCell(1));

    // Each component should know its boundary markers
    int r0 = ctx.regionForCell(0);
    int r1 = ctx.regionForCell(1);
    EXPECT_TRUE(ctx.components[static_cast<std::size_t>(r0)].boundary_markers.count(1));
    EXPECT_TRUE(ctx.components[static_cast<std::size_t>(r1)].boundary_markers.count(2));

    // region_to_markers
    EXPECT_TRUE(ctx.boundary_mapping.region_to_markers.at(r0).count(1));
    EXPECT_TRUE(ctx.boundary_mapping.region_to_markers.at(r1).count(2));

    // Nonexistent marker
    auto regions_bad = ctx.regionsForBoundaryMarker(99);
    EXPECT_TRUE(regions_bad.empty());
}

TEST(TopologyAnalysisContext, BoundaryMarkerDiscoveryUsesFaceIdsNotOrdinals) {
    SparseBoundaryFaceIdMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    ASSERT_EQ(ctx.numRegions(), 2);
    auto regions_m7 = ctx.regionsForBoundaryMarker(7);
    ASSERT_EQ(regions_m7.size(), 1u);
    EXPECT_EQ(regions_m7[0], ctx.regionForCell(0));

    auto regions_m9 = ctx.regionsForBoundaryMarker(9);
    ASSERT_EQ(regions_m9.size(), 1u);
    EXPECT_EQ(regions_m9[0], ctx.regionForCell(1));
}

TEST(TopologyAnalysisContext, EmptyMesh) {
    class EmptyMesh final : public IMeshAccess {
    public:
        [[nodiscard]] GlobalIndex numCells() const override { return 0; }
        [[nodiscard]] GlobalIndex numOwnedCells() const override { return 0; }
        [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
        [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
        [[nodiscard]] int dimension() const override { return 3; }
        [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
        [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }
        void getCellNodes(GlobalIndex, std::vector<GlobalIndex>&) const override {}
        [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
        void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>&) const override {}
        [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
        [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
        [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0,0}; }
        void forEachCell(std::function<void(GlobalIndex)>) const override {}
        void forEachOwnedCell(std::function<void(GlobalIndex)>) const override {}
        void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
        void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}
    };

    EmptyMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    EXPECT_EQ(ctx.numRegions(), 0);
    EXPECT_FALSE(ctx.isDisconnected());
    EXPECT_EQ(ctx.regionForCell(0), -1);
}

TEST(TopologyAnalysisContext, RegionForCell_OutOfRange) {
    SingleTetraMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    EXPECT_EQ(ctx.regionForCell(-1), -1);
    EXPECT_EQ(ctx.regionForCell(999), -1);
}

TEST(TopologyAnalysisContext, DefaultConstruction) {
    TopologyAnalysisContext ctx;
    EXPECT_EQ(ctx.numRegions(), 0);
    EXPECT_FALSE(ctx.isDisconnected());
    EXPECT_TRUE(ctx.components.empty());
}

// ============================================================================
// Higher-order (P2) mesh: 10-node tetrahedra share edge/face nodes
// ============================================================================

class TwoConnectedTet10Mesh final : public IMeshAccess {
public:
    // Two Tet10 elements sharing a face (6 nodes on shared face).
    // Tet A: vertices 0-3 + mid-edge nodes 4-9
    // Tet B: vertices 1,2,3,10 + mid-edge nodes 5,6,7,11,12,13
    // Shared nodes: 1,2,3 (vertices) + 5,6,7 (mid-edge on shared face)
    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra10; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        else              nodes = {1, 2, 3, 10, 5, 6, 7, 11, 12, 13};
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& c) const override {
        c.resize(10, {{0,0,0}});
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0,1}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}
};

TEST(TopologyAnalysisContext, HigherOrder_Tet10_SharedFaceNodes) {
    TwoConnectedTet10Mesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    // Two Tet10 elements sharing vertices 1,2,3 and mid-edge nodes 5,6,7
    // → they are in the same connected component
    EXPECT_EQ(ctx.numRegions(), 1);
    EXPECT_FALSE(ctx.isDisconnected());
    EXPECT_EQ(ctx.regionForCell(0), ctx.regionForCell(1));

    // Component has 14 unique nodes (0-13)
    EXPECT_EQ(ctx.components[0].num_vertices, 14);
    EXPECT_EQ(ctx.components[0].num_cells, 2);
}

// ============================================================================
// Interface faces: interior faces connecting different regions
// ============================================================================

class TwoTetraWithInteriorFaceMesh final : public IMeshAccess {
public:
    // Two tets in different connected components (no shared nodes),
    // but with an interior face connecting them. This models an
    // interface coupling between subdomain A (cell 0) and B (cell 1).
    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};  // No shared nodes → two components
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex) const override { return {0,0,0}; }
    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& c) const override {
        c = {{{{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override {
        return {0, 1};  // Interior face connects cell 0 and cell 1
    }

    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> cb) const override {
        cb(0, 0, 1);  // face 0 connects cell 0 and cell 1
    }
};

TEST(TopologyAnalysisContext, InteriorFaces_NoSyntheticInterfaceMapping) {
    // Since Phase 14, TopologyAnalysisContext no longer does synthetic
    // interface detection. Interface topology is in InterfaceTopologyContext.
    TwoTetraWithInteriorFaceMesh mesh;
    auto ctx = TopologyAnalysisContext::build(mesh);

    EXPECT_EQ(ctx.numRegions(), 2);
    EXPECT_TRUE(ctx.isDisconnected());

    // interface_mapping should be empty (no synthetic detection)
    EXPECT_TRUE(ctx.interface_mapping.interface_to_region_pairs.empty());
}

// ============================================================================
// InterfaceTopologyContext
// ============================================================================

TEST(InterfaceTopologyContext, DefaultEmpty) {
    InterfaceTopologyContext ctx;
    EXPECT_TRUE(ctx.empty());
    EXPECT_EQ(ctx.numFaces(), 0u);
    EXPECT_TRUE(ctx.markers().empty());
    EXPECT_FALSE(ctx.hasMarker(0));
}

TEST(InterfaceTopologyContext, ManualPopulation) {
    InterfaceTopologyContext ctx;

    InterfaceFaceRecord rec;
    rec.interface_marker = 5;
    rec.minus_cell = 0;
    rec.plus_cell = 1;
    rec.minus_local_face = 2;
    rec.plus_local_face = 3;
    rec.is_two_sided = true;
    rec.has_orientation = true;
    rec.minus_region = 0;
    rec.plus_region = 1;

    ctx.faces.push_back(rec);
    ctx.marker_to_faces[5].push_back(0);

    EXPECT_FALSE(ctx.empty());
    EXPECT_EQ(ctx.numFaces(), 1u);
    EXPECT_TRUE(ctx.hasMarker(5));
    EXPECT_FALSE(ctx.hasMarker(3));
    EXPECT_EQ(ctx.numFacesForMarker(5), 1u);
    EXPECT_EQ(ctx.numFacesForMarker(99), 0u);

    auto m = ctx.markers();
    ASSERT_EQ(m.size(), 1u);
    EXPECT_TRUE(m.count(5));

    EXPECT_EQ(ctx.faces[0].minus_cell, GlobalIndex{0});
    EXPECT_EQ(ctx.faces[0].plus_cell, GlobalIndex{1});
    EXPECT_TRUE(ctx.faces[0].is_two_sided);
}

TEST(InterfaceTopologyContext, MultipleMarkers) {
    InterfaceTopologyContext ctx;

    InterfaceFaceRecord r1;
    r1.interface_marker = 1;
    r1.minus_cell = 0;
    r1.plus_cell = 1;
    r1.is_two_sided = true;
    ctx.faces.push_back(r1);
    ctx.marker_to_faces[1].push_back(0);

    InterfaceFaceRecord r2;
    r2.interface_marker = 2;
    r2.minus_cell = 2;
    r2.plus_cell = 3;
    r2.is_two_sided = true;
    ctx.faces.push_back(r2);
    ctx.marker_to_faces[2].push_back(1);

    EXPECT_EQ(ctx.numFaces(), 2u);
    EXPECT_TRUE(ctx.hasMarker(1));
    EXPECT_TRUE(ctx.hasMarker(2));
    EXPECT_EQ(ctx.markers().size(), 2u);
}

TEST(InterfaceTopologyContext, StoredInProblemAnalysisContext) {
    ProblemAnalysisContext pctx;
    EXPECT_EQ(pctx.interfaceTopologyContext(), nullptr);

    InterfaceTopologyContext ictx;
    InterfaceFaceRecord rec;
    rec.interface_marker = 1;
    rec.minus_cell = 0;
    rec.is_two_sided = false;
    ictx.faces.push_back(rec);
    ictx.marker_to_faces[1].push_back(0);

    pctx.setInterfaceTopologyContext(std::move(ictx));
    EXPECT_NE(pctx.interfaceTopologyContext(), nullptr);
    EXPECT_EQ(pctx.interfaceTopologyContext()->numFaces(), 1u);
}
