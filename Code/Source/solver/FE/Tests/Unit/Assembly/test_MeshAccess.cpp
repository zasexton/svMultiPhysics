/**
 * @file test_MeshAccess.cpp
 * @brief Unit tests for Assembly MeshAccess adapter (requires Mesh library)
 */

#include <gtest/gtest.h>

#include "Assembly/MeshAccess.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::EntityKind;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::Ownership;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::MeshAccess;

namespace {

Mesh build_two_triangles_mesh() {
    auto base = std::make_shared<MeshBase>();
    auto& mesh = *base;

    // 2D mesh with two triangles sharing edge (1,2):
    //  cell0: (0,1,2), cell1: (1,3,2)
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0, // v0
        1.0, 0.0, // v1
        0.0, 1.0, // v2
        1.0, 1.0  // v3
    };

    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 3, 6};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 2,
        1, 3, 2
    };

    std::vector<CellShape> cell_shapes(2);
    for (auto& cs : cell_shapes) {
        cs.family = CellFamily::Triangle;
        cs.order = 1;
        cs.num_corners = 3;
    }

    mesh.build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);

    // Faces are codim-1 entities: in 2D, faces == edges (Line segments).
    // Face IDs are deterministic in this test by explicit construction:
    //  f0: (0,1) boundary on cell0, marker 1
    //  f1: (1,2) interior between cell0 and cell1
    //  f2: (2,0) boundary on cell0, marker 2
    //  f3: (1,3) boundary on cell1, marker 1
    //  f4: (3,2) boundary on cell1, marker 2
    std::vector<CellShape> face_shapes(5);
    for (auto& fs : face_shapes) {
        fs.family = CellFamily::Line;
        fs.order = 1;
        fs.num_corners = 2;
    }

    const std::vector<svmp::offset_t> face2vertex_offsets = {0, 2, 4, 6, 8, 10};
    const std::vector<svmp::index_t> face2vertex = {
        0, 1,
        1, 2,
        2, 0,
        1, 3,
        3, 2
    };
    const std::vector<std::array<svmp::index_t, 2>> face2cell = {
        {0, svmp::INVALID_INDEX},
        {0, 1},
        {0, svmp::INVALID_INDEX},
        {1, svmp::INVALID_INDEX},
        {1, svmp::INVALID_INDEX}
    };

    mesh.set_faces_from_arrays(face_shapes, face2vertex_offsets, face2vertex, face2cell);

    mesh.set_boundary_label(/*face=*/0, /*label=*/1);
    mesh.set_boundary_label(/*face=*/2, /*label=*/2);
    mesh.set_boundary_label(/*face=*/3, /*label=*/1);
    mesh.set_boundary_label(/*face=*/4, /*label=*/2);

    mesh.finalize();
    return Mesh(std::move(base), svmp::MeshComm::self());
}

Mesh build_two_triangles_boundary_only_mesh() {
    auto base = std::make_shared<MeshBase>();
    auto& mesh = *base;

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0
    };

    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 3, 6};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 2,
        1, 3, 2
    };

    std::vector<CellShape> cell_shapes(2);
    for (auto& cs : cell_shapes) {
        cs.family = CellFamily::Triangle;
        cs.order = 1;
        cs.num_corners = 3;
    }

    mesh.build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);
    mesh.finalize(svmp::MeshFinalizeOptions{
        .codim1_storage = svmp::MeshCodim1StorageMode::BoundaryOnly,
        .edge_storage = false,
    });
    return Mesh(std::move(base), svmp::MeshComm::self());
}

Mesh build_two_tetrahedra_mesh() {
    auto base = std::make_shared<MeshBase>();
    auto& mesh = *base;

    // 3D mesh with two tetrahedra sharing face (1,2,3):
    //  cell0: (0,1,2,3), cell1: (1,2,3,4)
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0, 0.0, // v0
        1.0, 0.0, 0.0, // v1
        0.0, 1.0, 0.0, // v2
        0.0, 0.0, 1.0, // v3
        1.0, 1.0, 1.0  // v4
    };

    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 2, 3,
        1, 2, 3, 4
    };

    std::vector<CellShape> cell_shapes(2);
    for (auto& cs : cell_shapes) {
        cs.family = CellFamily::Tetra;
        cs.order = 1;
        cs.num_corners = 4;
    }

    mesh.build_from_arrays(/*spatial_dim=*/3, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);

    // Deterministic face IDs:
    //  f0: (0,1,2) boundary on cell0, marker 10
    //  f1: (0,1,3) boundary on cell0, marker 10
    //  f2: (1,2,3) interior between cell0 and cell1
    //  f3: (0,2,3) boundary on cell0, marker 10
    //  f4: (1,2,4) boundary on cell1, marker 20
    //  f5: (2,3,4) boundary on cell1, marker 20
    //  f6: (1,3,4) boundary on cell1, marker 20
    std::vector<CellShape> face_shapes(7);
    for (auto& fs : face_shapes) {
        fs.family = CellFamily::Triangle;
        fs.order = 1;
        fs.num_corners = 3;
    }

    const std::vector<svmp::offset_t> face2vertex_offsets = {0, 3, 6, 9, 12, 15, 18, 21};
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

    mesh.set_boundary_label(/*face=*/0, /*label=*/10);
    mesh.set_boundary_label(/*face=*/1, /*label=*/10);
    mesh.set_boundary_label(/*face=*/3, /*label=*/10);
    mesh.set_boundary_label(/*face=*/4, /*label=*/20);
    mesh.set_boundary_label(/*face=*/5, /*label=*/20);
    mesh.set_boundary_label(/*face=*/6, /*label=*/20);

    mesh.finalize();
    return Mesh(std::move(base), svmp::MeshComm::self());
}

} // namespace

TEST(MeshAccess, TwoTrianglesBoundaryAndInteriorFaces) {
    auto mesh = build_two_triangles_mesh();
    MeshAccess access(mesh);

    EXPECT_EQ(access.dimension(), 2);
    EXPECT_EQ(access.numCells(), 2);
    EXPECT_EQ(access.numOwnedCells(), 2);
    EXPECT_EQ(access.numVertices(), 4);
    EXPECT_EQ(access.numOwnedVertices(), 4);
    EXPECT_EQ(access.numBoundaryFaces(), 4);
    EXPECT_EQ(access.numInteriorFaces(), 1);

    EXPECT_EQ(access.getCellType(0), ElementType::Triangle3);
    EXPECT_EQ(access.getCellType(1), ElementType::Triangle3);

    std::vector<GlobalIndex> nodes;
    access.getCellNodes(0, nodes);
    EXPECT_EQ(nodes, (std::vector<GlobalIndex>{0, 1, 2}));
    access.getCellNodes(1, nodes);
    EXPECT_EQ(nodes, (std::vector<GlobalIndex>{1, 3, 2}));

    const auto x3 = access.getNodeCoordinates(3);
    EXPECT_NEAR(x3[0], Real(1), 1e-12);
    EXPECT_NEAR(x3[1], Real(1), 1e-12);
    EXPECT_NEAR(x3[2], Real(0), 1e-12);

    // Local face indices (Triangle3 reference edge order: (0-1), (1-2), (2-0))
    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/0, /*cell_id=*/0), LocalIndex{0});
    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/1, /*cell_id=*/0), LocalIndex{1});
    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/2, /*cell_id=*/0), LocalIndex{2});

    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/3, /*cell_id=*/1), LocalIndex{0});
    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/4, /*cell_id=*/1), LocalIndex{1});
    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/1, /*cell_id=*/1), LocalIndex{2});

    EXPECT_EQ(access.getBoundaryFaceMarker(0), 1);
    EXPECT_EQ(access.getBoundaryFaceMarker(2), 2);
    EXPECT_EQ(access.getBoundaryFaceMarker(3), 1);
    EXPECT_EQ(access.getBoundaryFaceMarker(4), 2);

    std::vector<std::pair<GlobalIndex, GlobalIndex>> b1;
    access.forEachBoundaryFace(1, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        b1.emplace_back(face_id, cell_id);
    });
    std::sort(b1.begin(), b1.end());
    EXPECT_EQ(b1, (std::vector<std::pair<GlobalIndex, GlobalIndex>>{{0, 0}, {3, 1}}));

    std::vector<std::pair<GlobalIndex, GlobalIndex>> b2;
    access.forEachBoundaryFace(2, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        b2.emplace_back(face_id, cell_id);
    });
    std::sort(b2.begin(), b2.end());
    EXPECT_EQ(b2, (std::vector<std::pair<GlobalIndex, GlobalIndex>>{{2, 0}, {4, 1}}));

    std::vector<std::tuple<GlobalIndex, GlobalIndex, GlobalIndex>> interior;
    access.forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex c0, GlobalIndex c1) {
        interior.emplace_back(face_id, c0, c1);
    });
    ASSERT_EQ(interior.size(), 1u);
    EXPECT_EQ(interior[0], std::make_tuple(GlobalIndex{1}, GlobalIndex{0}, GlobalIndex{1}));

    const auto adj = access.getInteriorFaceCells(1);
    EXPECT_EQ(adj.first, 0);
    EXPECT_EQ(adj.second, 1);
}

TEST(MeshAccess, BoundaryOnlyFacesResolveLocalReferenceFace)
{
    auto mesh = build_two_triangles_boundary_only_mesh();
    MeshAccess access(mesh);

    ASSERT_EQ(mesh.base().codim1_storage_mode(), svmp::MeshCodim1StorageMode::BoundaryOnly);
    ASSERT_EQ(access.numCells(), 2);
    ASSERT_EQ(mesh.base().n_faces(), 4u);

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.base().n_faces()); ++f) {
        const auto cells = mesh.base().face_cells(f);
        ASSERT_NE(cells[0], svmp::INVALID_INDEX);
        ASSERT_EQ(cells[1], svmp::INVALID_INDEX);

        const auto local_face =
            access.getLocalFaceIndex(static_cast<GlobalIndex>(f),
                                     static_cast<GlobalIndex>(cells[0]));
        EXPECT_GE(local_face, LocalIndex{0});
        EXPECT_LT(local_face, LocalIndex{3});
    }
}

TEST(MeshAccess, RevisionQueriesFollowMeshState)
{
    auto mesh = build_two_triangles_mesh();
    MeshAccess access(mesh);

    EXPECT_TRUE(access.revisionTrackingAvailable());
    const auto initial_geometry = access.geometryRevision();
    const auto initial_current_geometry = mesh.current_geometry_revision();
    const auto initial_topology = access.topologyRevision();
    const auto initial_numbering = access.numberingRevision();
    const auto initial_labels = access.labelRevision();
    const auto initial_epoch = access.activeConfigurationEpoch();
    const auto initial_coordinate_key = access.coordinateConfigurationKey();

    auto x_cur = mesh.local_mesh().X_ref();
    ASSERT_FALSE(x_cur.empty());
    x_cur[0] += 0.25;
    mesh.set_current_coords(x_cur);
    EXPECT_EQ(access.geometryRevision(), initial_geometry);
    EXPECT_EQ(mesh.current_geometry_revision(), initial_current_geometry + 1u);
    EXPECT_EQ(access.topologyRevision(), initial_topology);

    auto* x_cur_mut = mesh.local_mesh().X_cur_data_mutable();
    ASSERT_NE(x_cur_mut, nullptr);
    x_cur_mut[0] += 0.125;
    mesh.local_mesh().mark_geometry_changed();
    EXPECT_EQ(access.geometryRevision(), initial_geometry);
    EXPECT_EQ(mesh.current_geometry_revision(), initial_current_geometry + 2u);

    mesh.use_current_configuration();
    EXPECT_EQ(access.geometryRevision(), initial_current_geometry + 2u);
    EXPECT_EQ(access.activeConfigurationEpoch(), initial_epoch + 1u);
    EXPECT_NE(access.coordinateConfigurationKey(), initial_coordinate_key);

    MeshAccess current_access(mesh, svmp::Configuration::Current);
    EXPECT_EQ(current_access.geometryRevision(), initial_current_geometry + 2u);
    EXPECT_EQ(current_access.activeConfigurationEpoch(), 0u);
    EXPECT_EQ(current_access.coordinateConfigurationKey(), access.coordinateConfigurationKey());

    mesh.local_mesh().set_boundary_label(0, 3);
    EXPECT_EQ(access.labelRevision(), initial_labels + 1u);

    auto cell_gids = mesh.local_mesh().cell_gids();
    mesh.local_mesh().set_cell_gids(std::move(cell_gids));
    EXPECT_EQ(access.numberingRevision(), initial_numbering + 1u);
}

TEST(MeshAccess, TwoTetrahedraLocalFaceIndexAndMarkers) {
    auto mesh = build_two_tetrahedra_mesh();
    MeshAccess access(mesh);

    EXPECT_EQ(access.dimension(), 3);
    EXPECT_EQ(access.numCells(), 2);
    EXPECT_EQ(access.numBoundaryFaces(), 6);
    EXPECT_EQ(access.numInteriorFaces(), 1);

    EXPECT_EQ(access.getCellType(0), ElementType::Tetra4);
    EXPECT_EQ(access.getCellType(1), ElementType::Tetra4);

    // Shared face f2 is local face 2 on cell0 and local face 0 on cell1.
    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/2, /*cell_id=*/0), LocalIndex{2});
    EXPECT_EQ(access.getLocalFaceIndex(/*face_id=*/2, /*cell_id=*/1), LocalIndex{0});

    std::vector<std::pair<GlobalIndex, GlobalIndex>> b10;
    access.forEachBoundaryFace(10, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        b10.emplace_back(face_id, cell_id);
    });
    std::sort(b10.begin(), b10.end());
    EXPECT_EQ(b10, (std::vector<std::pair<GlobalIndex, GlobalIndex>>{{0, 0}, {1, 0}, {3, 0}}));

    std::vector<std::pair<GlobalIndex, GlobalIndex>> b20;
    access.forEachBoundaryFace(20, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        b20.emplace_back(face_id, cell_id);
    });
    std::sort(b20.begin(), b20.end());
    EXPECT_EQ(b20, (std::vector<std::pair<GlobalIndex, GlobalIndex>>{{4, 1}, {5, 1}, {6, 1}}));

    std::vector<std::tuple<GlobalIndex, GlobalIndex, GlobalIndex>> interior;
    access.forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex c0, GlobalIndex c1) {
        interior.emplace_back(face_id, c0, c1);
    });
    ASSERT_EQ(interior.size(), 1u);
    EXPECT_EQ(interior[0], std::make_tuple(GlobalIndex{2}, GlobalIndex{0}, GlobalIndex{1}));
}

TEST(MeshAccess, CoordinateConfigurationOverrideIsDeterministic) {
    auto mesh = build_two_triangles_mesh();

    // Make current coordinates different from reference.
    auto X_cur = mesh.base().X_ref();
    for (std::size_t i = 0; i < X_cur.size(); i += 2) {
        X_cur[i] += 10.0; // shift x only
    }
    mesh.set_current_coords(X_cur);

    // Even if the mesh "active configuration" is reference, an explicit override
    // must deterministically select the requested coordinate set.
    mesh.use_reference_configuration();
    MeshAccess use_current(mesh, svmp::Configuration::Current);
    MeshAccess use_reference(mesh, svmp::Configuration::Reference);

    const auto x0_cur = use_current.getNodeCoordinates(0);
    const auto x0_ref = use_reference.getNodeCoordinates(0);

    EXPECT_NEAR(x0_cur[0], Real(10.0), 1e-12);
    EXPECT_NEAR(x0_ref[0], Real(0.0), 1e-12);

    // And the override should still win if the mesh is in current configuration.
    mesh.use_current_configuration();
    MeshAccess use_reference2(mesh, svmp::Configuration::Reference);
    const auto x0_ref2 = use_reference2.getNodeCoordinates(0);
    EXPECT_NEAR(x0_ref2[0], Real(0.0), 1e-12);
}

TEST(MeshAccess, DeformedConfigurationOverrideMatchesCurrentWhenPresent) {
    auto mesh = build_two_triangles_mesh();

    // Make current coordinates different from reference.
    auto X_cur = mesh.base().X_ref();
    for (std::size_t i = 0; i < X_cur.size(); i += 2) {
        X_cur[i] += 10.0; // shift x only
    }
    mesh.set_current_coords(X_cur);

    // Deformed is a compatibility alias for Current.
    MeshAccess use_deformed(mesh, svmp::Configuration::Deformed);
    MeshAccess use_current(mesh, svmp::Configuration::Current);

    const auto x0_def = use_deformed.getNodeCoordinates(0);
    const auto x0_cur = use_current.getNodeCoordinates(0);
    EXPECT_NEAR(x0_def[0], x0_cur[0], 1e-12);
    EXPECT_NEAR(x0_def[1], x0_cur[1], 1e-12);

    std::vector<std::array<Real, 3>> c0_def;
    std::vector<std::array<Real, 3>> c0_cur;
    use_deformed.getCellCoordinates(0, c0_def);
    use_current.getCellCoordinates(0, c0_cur);
    ASSERT_EQ(c0_def.size(), c0_cur.size());
    for (std::size_t i = 0; i < c0_def.size(); ++i) {
        EXPECT_NEAR(c0_def[i][0], c0_cur[i][0], 1e-12);
        EXPECT_NEAR(c0_def[i][1], c0_cur[i][1], 1e-12);
        EXPECT_NEAR(c0_def[i][2], c0_cur[i][2], 1e-12);
    }
}

TEST(MeshAccess, WithoutOverrideUsesMeshActiveConfiguration) {
    auto mesh = build_two_triangles_mesh();

    // Make current coordinates different from reference.
    auto X_cur = mesh.base().X_ref();
    for (std::size_t i = 0; i < X_cur.size(); i += 2) {
        X_cur[i] += 10.0; // shift x only
    }
    mesh.set_current_coords(X_cur);

    MeshAccess access(mesh); // no override: follows mesh.active_configuration()

    mesh.use_reference_configuration();
    const auto x0_ref = access.getNodeCoordinates(0);
    EXPECT_NEAR(x0_ref[0], Real(0.0), 1e-12);
    EXPECT_NEAR(x0_ref[1], Real(0.0), 1e-12);

    mesh.use_current_configuration();
    const auto x0_cur = access.getNodeCoordinates(0);
    EXPECT_NEAR(x0_cur[0], Real(10.0), 1e-12);
    EXPECT_NEAR(x0_cur[1], Real(0.0), 1e-12);
}

TEST(MeshAccess, CurrentActiveConfigurationWithoutCurrentCoordsFallsBackToReference) {
    auto mesh = build_two_triangles_mesh();
    ASSERT_FALSE(mesh.base().has_current_coords());

    MeshAccess access(mesh); // no override
    mesh.use_current_configuration(); // but X_cur is not set

    const auto x0 = access.getNodeCoordinates(0);
    EXPECT_NEAR(x0[0], Real(0.0), 1e-12);
    EXPECT_NEAR(x0[1], Real(0.0), 1e-12);
}

TEST(MeshAccess, OwnedCellIterationRespectsGhosts) {
    auto mesh = build_two_triangles_mesh();
    mesh.set_ownership(/*id=*/1, EntityKind::Volume, Ownership::Ghost, /*owner_rank=*/1);
    mesh.set_ownership(/*id=*/3, EntityKind::Vertex, Ownership::Ghost, /*owner_rank=*/1);

    MeshAccess access(mesh);
    EXPECT_EQ(access.numCells(), 2);
    EXPECT_EQ(access.numOwnedCells(), 1);
    EXPECT_EQ(access.numVertices(), 4);
    EXPECT_EQ(access.numOwnedVertices(), 3);
    EXPECT_TRUE(access.isOwnedCell(0));
    EXPECT_FALSE(access.isOwnedCell(1));

    std::vector<GlobalIndex> owned;
    access.forEachOwnedCell([&](GlobalIndex c) { owned.push_back(c); });
    EXPECT_EQ(owned, (std::vector<GlobalIndex>{0}));

    // Boundary faces adjacent to ghost cells should not be iterated for assembly.
    std::vector<std::pair<GlobalIndex, GlobalIndex>> b_all;
    access.forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex f, GlobalIndex c) {
        b_all.emplace_back(f, c);
    });
    std::sort(b_all.begin(), b_all.end());
    EXPECT_EQ(b_all, (std::vector<std::pair<GlobalIndex, GlobalIndex>>{{0, 0}, {2, 0}}));
}
