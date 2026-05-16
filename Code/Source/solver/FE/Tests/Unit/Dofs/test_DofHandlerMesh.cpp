/**
 * @file test_DofHandlerMesh.cpp
 * @brief Unit tests for DofHandler mesh convenience overloads (requires Mesh library)
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofHandler.h"
#include "FE/Spaces/H1Space.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Observer/MeshObserver.h"
#include "Mesh/Observer/ScopedSubscription.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::MeshBase;
using svmp::MeshCodim1StorageMode;
using svmp::MeshEvent;
using svmp::MeshFinalizeOptions;
using svmp::EventCounter;
using svmp::ScopedSubscription;

using svmp::FE::ElementType;
using svmp::FE::dofs::DofDistributionOptions;
using svmp::FE::dofs::DofHandler;
using svmp::FE::dofs::TopologyCompletion;
using svmp::FE::spaces::H1Space;

namespace {

MeshBase build_single_quad_mesh(bool finalize_mesh) {
    MeshBase mesh;

    // 2D quad with vertices (0,0), (1,0), (1,1), (0,1)
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    const std::vector<CellShape> cell_shapes = {shape};

    mesh.build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);
    if (finalize_mesh) {
        mesh.finalize();
    }
    return mesh;
}

MeshBase build_two_quad_boundary_only_mesh() {
    MeshBase mesh;

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    const std::vector<CellShape> cell_shapes = {shape, shape};

    mesh.build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);

    MeshFinalizeOptions options;
    options.codim1_storage = MeshCodim1StorageMode::BoundaryOnly;
    options.edge_storage = true;
    mesh.finalize(options);

    return mesh;
}

MeshBase build_two_biquadratic_quad_mesh() {
    MeshBase mesh;

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        0.5, 0.0,
        1.0, 0.5,
        0.5, 1.0,
        0.0, 0.5,
        0.5, 0.5,
        1.5, 0.0,
        2.0, 0.5,
        1.5, 1.0,
        1.5, 0.5
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 9, 18};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 4, 3, 6, 7, 8, 9, 10,
        1, 2, 5, 4, 11, 12, 13, 7, 14
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    const std::vector<CellShape> cell_shapes = {shape, shape};

    mesh.build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);

    MeshFinalizeOptions options;
    options.codim1_storage = MeshCodim1StorageMode::Full;
    options.edge_storage = false;
    mesh.finalize(options);

    return mesh;
}

} // namespace

TEST(DofHandlerMesh, CacheSkipsRedundantDistributionWhenMeshUnchanged) {
    auto mesh = build_single_quad_mesh(/*finalize_mesh=*/true);
    H1Space space(ElementType::Quad4, /*order=*/1);

    DofHandler handler;
    handler.distributeDofs(mesh, space);
    const auto rev1 = handler.getDofStateRevision();
    ASSERT_GT(rev1, 0u);

    handler.distributeDofs(mesh, space);
    const auto rev2 = handler.getDofStateRevision();

    EXPECT_EQ(rev2, rev1);
}

TEST(DofHandlerMesh, CacheInvalidatesWhenVertexGidsChange) {
    auto mesh = build_single_quad_mesh(/*finalize_mesh=*/true);
    H1Space space(ElementType::Quad4, /*order=*/1);

    DofHandler handler;
    handler.distributeDofs(mesh, space);
    const auto rev1 = handler.getDofStateRevision();

    EventCounter counter;
    ScopedSubscription subscription(&mesh.event_bus(), &counter);

    auto gids = mesh.vertex_gids();
    std::reverse(gids.begin(), gids.end());
    mesh.set_vertex_gids(std::move(gids));

    EXPECT_EQ(counter.count(MeshEvent::NumberingChanged), 1u);
    EXPECT_EQ(counter.count(MeshEvent::TopologyChanged), 0u);

    handler.distributeDofs(mesh, space);
    const auto rev2 = handler.getDofStateRevision();

    EXPECT_GT(rev2, rev1);
}

TEST(DofHandlerMesh, RequireCompleteIn2DUsesMeshFacesAsEdges) {
    auto mesh = build_single_quad_mesh(/*finalize_mesh=*/false);
    H1Space space(ElementType::Quad4, /*order=*/2);

    DofDistributionOptions opts;
    opts.topology_completion = TopologyCompletion::RequireComplete;

    DofHandler handler;
    EXPECT_THROW(handler.distributeDofs(mesh, space, opts), svmp::FE::FEException);

    mesh.finalize();

    handler.distributeDofs(mesh, space, opts);
    handler.finalize();
    EXPECT_EQ(handler.getNumDofs(), 9);
}

TEST(DofHandlerMesh, BoundaryOnly2DFacesUseFullEdgeStorageForQ2Distribution) {
    auto mesh = build_two_quad_boundary_only_mesh();
    ASSERT_EQ(mesh.codim1_storage_mode(), MeshCodim1StorageMode::BoundaryOnly);
    ASSERT_EQ(mesh.n_faces(), 6u);
    ASSERT_EQ(mesh.n_edges(), 7u);

    H1Space space(ElementType::Quad4, /*order=*/2);

    DofHandler handler;
    handler.distributeDofs(mesh, space);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 15);
}

TEST(DofHandlerMesh, FullHighOrder2DFacesUseEndpointCornersForQ2Distribution) {
    auto mesh = build_two_biquadratic_quad_mesh();
    ASSERT_EQ(mesh.codim1_storage_mode(), MeshCodim1StorageMode::Full);
    ASSERT_EQ(mesh.n_faces(), 7u);
    ASSERT_GT(mesh.face_vertices_span(0).second, 2u);

    H1Space space(ElementType::Quad4, /*order=*/2);

    DofDistributionOptions opts;
    opts.topology_completion = TopologyCompletion::RequireComplete;

    DofHandler handler;
    handler.distributeDofs(mesh, space, opts);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 15);
}
