/**
 * @file test_CompositeMeshAccess.cpp
 * @brief Unit tests for multi-participant Assembly mesh access.
 */

#include <gtest/gtest.h>

#include "Assembly/CompositeMeshAccess.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <memory>
#include <tuple>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::CompositeMeshAccess;
using svmp::FE::assembly::CompositeMeshParticipant;

namespace {

Mesh build_two_tetrahedra_mesh(double x_shift,
                               int first_cell_marker,
                               int second_cell_marker,
                               int region_label)
{
    auto base = std::make_shared<MeshBase>();
    auto& mesh = *base;

    const std::vector<svmp::real_t> x_ref = {
        x_shift + 0.0, 0.0, 0.0,
        x_shift + 1.0, 0.0, 0.0,
        x_shift + 0.0, 1.0, 0.0,
        x_shift + 0.0, 0.0, 1.0,
        x_shift + 1.0, 1.0, 1.0
    };

    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 2, 3,
        1, 2, 3, 4
    };

    std::vector<CellShape> cell_shapes(2);
    for (auto& shape : cell_shapes) {
        shape.family = CellFamily::Tetra;
        shape.order = 1;
        shape.num_corners = 4;
    }

    mesh.build_from_arrays(3, x_ref, cell2vertex_offsets, cell2vertex, cell_shapes);
    mesh.set_region_label(0, static_cast<svmp::label_t>(region_label));
    mesh.set_region_label(1, static_cast<svmp::label_t>(region_label));

    std::vector<CellShape> face_shapes(7);
    for (auto& shape : face_shapes) {
        shape.family = CellFamily::Triangle;
        shape.order = 1;
        shape.num_corners = 3;
    }

    const std::vector<svmp::offset_t> face2vertex_offsets = {
        0, 3, 6, 9, 12, 15, 18, 21
    };
    const std::vector<svmp::index_t> face2vertex = {
        0, 1, 2,
        0, 1, 3,
        1, 2, 3,
        0, 2, 3,
        1, 2, 4,
        2, 3, 4,
        1, 3, 4
    };
    const std::vector<std::array<svmp::index_t, 2>> face2cell = {
        {0, svmp::INVALID_INDEX},
        {0, svmp::INVALID_INDEX},
        {0, 1},
        {0, svmp::INVALID_INDEX},
        {1, svmp::INVALID_INDEX},
        {1, svmp::INVALID_INDEX},
        {1, svmp::INVALID_INDEX}
    };

    mesh.set_faces_from_arrays(face_shapes, face2vertex_offsets, face2vertex, face2cell);
    mesh.set_boundary_label(0, static_cast<svmp::label_t>(first_cell_marker));
    mesh.set_boundary_label(1, static_cast<svmp::label_t>(first_cell_marker));
    mesh.set_boundary_label(3, static_cast<svmp::label_t>(first_cell_marker));
    mesh.set_boundary_label(4, static_cast<svmp::label_t>(second_cell_marker));
    mesh.set_boundary_label(5, static_cast<svmp::label_t>(second_cell_marker));
    mesh.set_boundary_label(6, static_cast<svmp::label_t>(second_cell_marker));

    mesh.finalize();
    return Mesh(std::move(base), svmp::MeshComm::self());
}

CompositeMeshAccess make_composite(Mesh& fluid, Mesh& solid)
{
    return CompositeMeshAccess({
        CompositeMeshParticipant{"fluid", &fluid, 10},
        CompositeMeshParticipant{"solid", &solid, 20}
    });
}

} // namespace

TEST(CompositeMeshAccess, OffsetsDisconnectedTetrahedralParticipants)
{
    auto fluid = build_two_tetrahedra_mesh(0.0, 5, 6, 101);
    auto solid = build_two_tetrahedra_mesh(10.0, 5, 6, 202);
    auto access = make_composite(fluid, solid);

    EXPECT_EQ(access.numParticipants(), 2u);
    EXPECT_EQ(access.dimension(), 3);
    EXPECT_EQ(access.numCells(), 4);
    EXPECT_EQ(access.numOwnedCells(), 4);
    EXPECT_EQ(access.numVertices(), 10);
    EXPECT_EQ(access.numOwnedVertices(), 10);
    EXPECT_EQ(access.numBoundaryFaces(), 12);
    EXPECT_EQ(access.numInteriorFaces(), 2);
    EXPECT_TRUE(access.cellIdsAreDense());

    EXPECT_EQ(access.participantIndex("fluid"), 0u);
    EXPECT_EQ(access.participantIndex("solid"), 1u);
    EXPECT_EQ(access.participantName(1), "solid");

    EXPECT_EQ(access.cellLocation(0).participant_name, "fluid");
    EXPECT_EQ(access.cellLocation(2).participant_name, "solid");
    EXPECT_EQ(access.cellLocation(2).local_id, 0);
    EXPECT_EQ(access.vertexLocation(5).participant_name, "solid");
    EXPECT_EQ(access.vertexLocation(5).local_id, 0);

    EXPECT_EQ(access.getCellType(0), ElementType::Tetra4);
    EXPECT_EQ(access.getCellType(2), ElementType::Tetra4);
    EXPECT_EQ(access.getCellDomainId(0), 10);
    EXPECT_EQ(access.getCellDomainId(2), 20);

    std::vector<GlobalIndex> nodes;
    access.getCellNodes(2, nodes);
    EXPECT_EQ(nodes, (std::vector<GlobalIndex>{5, 6, 7, 8}));

    const auto x = access.getNodeCoordinates(5);
    EXPECT_NEAR(x[0], Real(10.0), 1e-12);
    EXPECT_NEAR(x[1], Real(0.0), 1e-12);
    EXPECT_NEAR(x[2], Real(0.0), 1e-12);
}

TEST(CompositeMeshAccess, RemapsBoundaryMarkersByParticipant)
{
    auto fluid = build_two_tetrahedra_mesh(0.0, 5, 6, 101);
    auto solid = build_two_tetrahedra_mesh(10.0, 5, 6, 202);
    auto access = make_composite(fluid, solid);

    const int fluid_marker = access.globalBoundaryMarker("fluid", 5);
    const int solid_marker = access.globalBoundaryMarker("solid", 5);
    ASSERT_NE(fluid_marker, solid_marker);

    const auto fluid_info = access.boundaryMarkerInfo(fluid_marker);
    ASSERT_TRUE(fluid_info.has_value());
    EXPECT_EQ(fluid_info->participant_name, "fluid");
    EXPECT_EQ(fluid_info->local_marker, 5);

    std::vector<std::pair<GlobalIndex, GlobalIndex>> fluid_faces;
    access.forEachBoundaryFace(
        fluid_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            fluid_faces.emplace_back(face_id, cell_id);
            EXPECT_EQ(access.getBoundaryFaceMarker(face_id), fluid_marker);
        });
    std::sort(fluid_faces.begin(), fluid_faces.end());
    EXPECT_EQ(fluid_faces,
              (std::vector<std::pair<GlobalIndex, GlobalIndex>>{
                  {0, 0}, {1, 0}, {3, 0}}));

    std::vector<std::pair<GlobalIndex, GlobalIndex>> solid_faces;
    access.forEachBoundaryFace(
        solid_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            solid_faces.emplace_back(face_id, cell_id);
            EXPECT_EQ(access.getBoundaryFaceMarker(face_id), solid_marker);
        });
    std::sort(solid_faces.begin(), solid_faces.end());
    EXPECT_EQ(solid_faces,
              (std::vector<std::pair<GlobalIndex, GlobalIndex>>{
                  {7, 2}, {8, 2}, {10, 2}}));

    std::vector<std::pair<GlobalIndex, GlobalIndex>> all_faces;
    access.forEachBoundaryFace(
        -1,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            all_faces.emplace_back(face_id, cell_id);
        });
    EXPECT_EQ(all_faces.size(), 12u);
}

TEST(CompositeMeshAccess, TraversesInteriorFacesPerParticipant)
{
    auto fluid = build_two_tetrahedra_mesh(0.0, 5, 6, 101);
    auto solid = build_two_tetrahedra_mesh(10.0, 5, 6, 202);
    auto access = make_composite(fluid, solid);

    std::vector<std::tuple<GlobalIndex, GlobalIndex, GlobalIndex>> interior;
    access.forEachInteriorFace(
        [&](GlobalIndex face_id, GlobalIndex cell_0, GlobalIndex cell_1) {
            interior.emplace_back(face_id, cell_0, cell_1);
        });
    std::sort(interior.begin(), interior.end());
    EXPECT_EQ(interior,
              (std::vector<std::tuple<GlobalIndex, GlobalIndex, GlobalIndex>>{
                  {2, 0, 1}, {9, 2, 3}}));

    const auto fluid_cells = access.getInteriorFaceCells(2);
    EXPECT_EQ(fluid_cells.first, 0);
    EXPECT_EQ(fluid_cells.second, 1);

    const auto solid_cells = access.getInteriorFaceCells(9);
    EXPECT_EQ(solid_cells.first, 2);
    EXPECT_EQ(solid_cells.second, 3);

    EXPECT_EQ(access.getLocalFaceIndex(2, 0), LocalIndex{2});
    EXPECT_EQ(access.getLocalFaceIndex(2, 1), LocalIndex{0});
    EXPECT_EQ(access.getLocalFaceIndex(9, 2), LocalIndex{2});
    EXPECT_EQ(access.getLocalFaceIndex(9, 3), LocalIndex{0});
}

TEST(CompositeMeshAccess, ProvidesCompressedBoundaryAndInteriorReverseMaps)
{
    auto fluid = build_two_tetrahedra_mesh(0.0, 5, 6, 101);
    auto solid = build_two_tetrahedra_mesh(10.0, 5, 6, 202);
    auto access = make_composite(fluid, solid);

    const auto first_boundary = access.boundaryFaceLocation(0);
    EXPECT_EQ(first_boundary.participant_name, "fluid");
    EXPECT_EQ(first_boundary.local_id, 0);

    const auto first_solid_boundary = access.boundaryFaceLocation(6);
    EXPECT_EQ(first_solid_boundary.participant_name, "solid");
    EXPECT_EQ(first_solid_boundary.local_id, 0);

    const auto first_interior = access.interiorFaceLocation(0);
    EXPECT_EQ(first_interior.participant_name, "fluid");
    EXPECT_EQ(first_interior.local_id, 2);

    const auto first_solid_interior = access.interiorFaceLocation(1);
    EXPECT_EQ(first_solid_interior.participant_name, "solid");
    EXPECT_EQ(first_solid_interior.local_id, 2);

    const auto stored_face = access.storedFaceLocation(9);
    EXPECT_EQ(stored_face.participant_name, "solid");
    EXPECT_EQ(stored_face.local_id, 2);
}
