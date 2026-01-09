/**
 * @file test_SearchAccess.cpp
 * @brief Unit tests for Systems search access interfaces
 */

#include <gtest/gtest.h>

#include "Systems/SearchAccess.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Systems/MeshSearchAccess.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::INVALID_GLOBAL_INDEX;
using svmp::FE::Real;
using svmp::FE::systems::ISearchAccess;

namespace {

class StubSearchAccess final : public ISearchAccess {
public:
    [[nodiscard]] int dimension() const noexcept override { return 3; }
    void build() const override {}
    [[nodiscard]] std::vector<GlobalIndex> verticesInRadius(const std::array<Real, 3>&, Real) const override { return {}; }
};

} // namespace

TEST(SearchAccess, ISearchAccess_DefaultLocatePoint_ReturnsNotFound)
{
    StubSearchAccess access;
    const auto loc = access.locatePoint({0.0, 0.0, 0.0});
    EXPECT_FALSE(loc.found);
    EXPECT_EQ(loc.cell_id, INVALID_GLOBAL_INDEX);
}

TEST(SearchAccess, ISearchAccess_DefaultNearestVertex_ReturnsNotFound)
{
    StubSearchAccess access;
    const auto nn = access.nearestVertex({0.0, 0.0, 0.0});
    EXPECT_FALSE(nn.found);
    EXPECT_EQ(nn.vertex_id, INVALID_GLOBAL_INDEX);
}

TEST(SearchAccess, ISearchAccess_DefaultKNearestVertices_ReturnsEmpty)
{
    StubSearchAccess access;
    const auto knn = access.kNearestVertices({0.0, 0.0, 0.0}, 3u);
    EXPECT_TRUE(knn.empty());
}

TEST(SearchAccess, ISearchAccess_DefaultNearestCell_ReturnsNotFound)
{
    StubSearchAccess access;
    const auto nc = access.nearestCell({0.0, 0.0, 0.0});
    EXPECT_FALSE(nc.found);
    EXPECT_EQ(nc.cell_id, INVALID_GLOBAL_INDEX);
}

TEST(SearchAccess, ISearchAccess_DefaultClosestBoundaryPoint_ReturnsNotFound)
{
    StubSearchAccess access;
    const auto cp = access.closestBoundaryPoint({0.0, 0.0, 0.0});
    EXPECT_FALSE(cp.found);
    EXPECT_EQ(cp.face_id, INVALID_GLOBAL_INDEX);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::FE::systems::MeshSearchAccess;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();

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
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_two_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();

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
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_two_segment_line_mesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {0.0, 1.0, 2.0};
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 2, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 1, 2};

    CellShape shape{};
    shape.family = CellFamily::Line;
    shape.num_corners = 2;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/1, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_single_tetra_mesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/3, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

svmp::index_t mark_left_edge(MeshBase& base, int marker)
{
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base.n_faces()); ++f) {
        const auto verts = base.face_vertices(f);
        if (verts.size() != 2u) continue;
        const bool has0 = (verts[0] == 0 || verts[1] == 0);
        const bool has3 = (verts[0] == 3 || verts[1] == 3);
        if (has0 && has3) {
            base.set_boundary_label(f, marker);
            return f;
        }
    }
    return svmp::INVALID_INDEX;
}

template <typename T>
std::vector<T> sorted(std::vector<T> v)
{
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

TEST(SearchAccess, MeshSearchAccess_Dimension_MatchesMesh)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);
    EXPECT_EQ(access.dimension(), 2);
}

TEST(SearchAccess, MeshSearchAccess_Build_EnablesQueries)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const std::array<Real, 3> x{0.3, 0.7, 0.0};
    const auto before = access.locatePoint(x);
    EXPECT_TRUE(before.found);

    access.build();
    const auto after = access.locatePoint(x);
    EXPECT_TRUE(after.found);
    EXPECT_EQ(after.cell_id, before.cell_id);
}

TEST(SearchAccess, MeshSearchAccess_VerticesInRadius_ReturnsCorrectSet)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto ids = sorted(access.verticesInRadius({0.05, 0.05, 0.0}, 0.2));
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 0);
}

TEST(SearchAccess, MeshSearchAccess_VerticesInRadius_EmptyForZeroRadius)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto ids = access.verticesInRadius({0.1, 0.1, 0.0}, 0.0);
    EXPECT_TRUE(ids.empty());
}

TEST(SearchAccess, MeshSearchAccess_VerticesInRadius_AllVerticesForLargeRadius)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto ids = sorted(access.verticesInRadius({0.5, 0.5, 0.0}, 100.0));
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(ids[1], 1);
    EXPECT_EQ(ids[2], 2);
    EXPECT_EQ(ids[3], 3);
}

TEST(SearchAccess, MeshSearchAccess_LocatePoint_FindsContainingCell)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto loc = access.locatePoint({0.3, 0.7, 0.0});
    ASSERT_TRUE(loc.found);
    EXPECT_EQ(loc.cell_id, 0);
}

TEST(SearchAccess, MeshSearchAccess_LocatePoint_ReturnsNotFoundOutsideMesh)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto loc = access.locatePoint({3.0, 0.5, 0.0});
    EXPECT_FALSE(loc.found);
}

TEST(SearchAccess, MeshSearchAccess_LocatePoint_HintCellAcceleratesSearch)
{
    auto mesh = build_two_quad_mesh();
    MeshSearchAccess access(*mesh);

    const std::array<Real, 3> x{1.5, 0.5, 0.0};

    const auto no_hint = access.locatePoint(x);
    ASSERT_TRUE(no_hint.found);
    EXPECT_EQ(no_hint.cell_id, 1);

    const auto good_hint = access.locatePoint(x, /*hint_cell=*/1);
    ASSERT_TRUE(good_hint.found);
    EXPECT_EQ(good_hint.cell_id, 1);

    const auto bad_hint = access.locatePoint(x, /*hint_cell=*/0);
    ASSERT_TRUE(bad_hint.found);
    EXPECT_EQ(bad_hint.cell_id, 1);
}

TEST(SearchAccess, MeshSearchAccess_NearestVertex_ReturnsClosest)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto nn = access.nearestVertex({0.05, 0.95, 0.0});
    ASSERT_TRUE(nn.found);
    EXPECT_EQ(nn.vertex_id, 3);
}

TEST(SearchAccess, MeshSearchAccess_KNearestVertices_ReturnsKClosest)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto knn = access.kNearestVertices({0.1, 0.1, 0.0}, 3u);
    ASSERT_EQ(knn.size(), 3u);
    EXPECT_EQ(knn[0].vertex_id, 0);
    EXPECT_LE(knn[0].distance, knn[1].distance);
    EXPECT_LE(knn[1].distance, knn[2].distance);
}

TEST(SearchAccess, MeshSearchAccess_KNearestVertices_ReturnsAllIfKExceedsCount)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto knn = access.kNearestVertices({0.1, 0.1, 0.0}, 100u);
    EXPECT_EQ(knn.size(), 4u);
}

TEST(SearchAccess, MeshSearchAccess_NearestCell_ReturnsClosestCellCentroid)
{
    auto mesh = build_two_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto nc = access.nearestCell({3.0, 0.5, 0.0});
    ASSERT_TRUE(nc.found);
    EXPECT_EQ(nc.cell_id, 1);
    EXPECT_TRUE(std::isfinite(nc.distance));
}

TEST(SearchAccess, MeshSearchAccess_ClosestBoundaryPoint_ReturnsProjection)
{
    auto mesh = build_single_quad_mesh();
    MeshSearchAccess access(*mesh);

    const std::array<Real, 3> p{-0.1, 0.2, 0.0};
    const auto cp = access.closestBoundaryPoint(p);
    ASSERT_TRUE(cp.found);
    EXPECT_NE(cp.face_id, INVALID_GLOBAL_INDEX);
    EXPECT_NEAR(cp.closest_point[0], 0.0, 1e-12);
    EXPECT_NEAR(cp.closest_point[1], 0.2, 1e-12);
    EXPECT_NEAR(cp.distance, 0.1, 1e-12);
}

TEST(SearchAccess, MeshSearchAccess_ClosestBoundaryPointOnMarker_FiltersCorrectly)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh();
    auto& base = mesh->local_mesh();
    const auto left = mark_left_edge(base, marker);
    ASSERT_NE(left, svmp::INVALID_INDEX);

    MeshSearchAccess access(*mesh);

    const std::array<Real, 3> p{-0.1, 0.2, 0.0};
    const auto cp = access.closestBoundaryPointOnMarker(marker, p);
    ASSERT_TRUE(cp.found);
    EXPECT_EQ(cp.face_id, static_cast<GlobalIndex>(left));

    const auto missing = access.closestBoundaryPointOnMarker(999, p);
    EXPECT_FALSE(missing.found);
}

TEST(SearchAccess, MeshSearchAccess_QueryBeforeBuild_Behavior)
{
    auto mesh = build_two_quad_mesh();
    MeshSearchAccess access(*mesh);

    const auto loc = access.locatePoint({1.5, 0.5, 0.0});
    ASSERT_TRUE(loc.found);
    EXPECT_EQ(loc.cell_id, 1);

    const auto nn = access.nearestVertex({0.05, 0.05, 0.0});
    EXPECT_TRUE(nn.found);
    EXPECT_EQ(nn.vertex_id, 0);
}

TEST(SearchAccess, MeshSearchAccess_1DMesh_QueriesWork)
{
    auto mesh = build_two_segment_line_mesh();
    MeshSearchAccess access(*mesh);

    EXPECT_EQ(access.dimension(), 1);

    const auto loc = access.locatePoint({0.25, 0.0, 0.0});
    ASSERT_TRUE(loc.found);
    EXPECT_EQ(loc.cell_id, 0);

    const auto nn = access.nearestVertex({0.2, 0.0, 0.0});
    ASSERT_TRUE(nn.found);
    EXPECT_EQ(nn.vertex_id, 0);

    const auto ids = sorted(access.verticesInRadius({0.2, 0.0, 0.0}, 0.3));
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 0);

    const auto cp = access.closestBoundaryPoint({-0.1, 0.0, 0.0});
    ASSERT_TRUE(cp.found);
    EXPECT_TRUE(std::isfinite(cp.distance));

    const auto nc = access.nearestCell({10.0, 0.0, 0.0});
    ASSERT_TRUE(nc.found);
    EXPECT_EQ(nc.cell_id, 1);
}

TEST(SearchAccess, MeshSearchAccess_3DMesh_QueriesWork)
{
    auto mesh = build_single_tetra_mesh();
    MeshSearchAccess access(*mesh);

    EXPECT_EQ(access.dimension(), 3);

    const auto loc = access.locatePoint({0.1, 0.1, 0.1});
    ASSERT_TRUE(loc.found);
    EXPECT_EQ(loc.cell_id, 0);

    const auto nn = access.nearestVertex({0.01, 0.01, 0.01});
    ASSERT_TRUE(nn.found);
    EXPECT_EQ(nn.vertex_id, 0);

    const auto knn = access.kNearestVertices({0.2, 0.2, 0.2}, 4u);
    EXPECT_EQ(knn.size(), 4u);

    const auto cp = access.closestBoundaryPoint({-0.2, 0.2, 0.2});
    ASSERT_TRUE(cp.found);
    EXPECT_NE(cp.face_id, INVALID_GLOBAL_INDEX);
    EXPECT_TRUE(std::isfinite(cp.distance));
}

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
