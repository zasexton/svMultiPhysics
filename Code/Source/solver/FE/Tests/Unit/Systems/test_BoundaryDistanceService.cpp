/**
 * @file test_BoundaryDistanceService.cpp
 * @brief Unit tests for generic boundary-distance queries.
 */

#include <gtest/gtest.h>

#include "Systems/BoundaryDistanceService.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::INVALID_GLOBAL_INDEX;
using svmp::FE::Real;
using svmp::FE::systems::BoundaryDistanceQueryLocation;
using svmp::FE::systems::BoundaryDistanceQueryPoint;
using svmp::FE::systems::BoundaryDistanceService;
using svmp::FE::systems::BoundaryDistanceTarget;
using svmp::FE::systems::BoundaryDistanceTargetKind;
using svmp::FE::systems::BoundaryMarkerSet;
using svmp::FE::systems::ISearchAccess;

namespace {

struct BoundaryPoint {
    int marker{0};
    GlobalIndex face_id{INVALID_GLOBAL_INDEX};
    std::array<Real, 3> point{0.0, 0.0, 0.0};
};

class PointBoundarySearch final : public ISearchAccess {
public:
    explicit PointBoundarySearch(int dim) : dim_(dim) {}

    [[nodiscard]] int dimension() const noexcept override { return dim_; }
    void build() const override { ++build_count_; }

    [[nodiscard]] std::vector<GlobalIndex> verticesInRadius(
        const std::array<Real, 3>&,
        Real) const override
    {
        return {};
    }

    [[nodiscard]] ClosestBoundaryPoint closestBoundaryPoint(
        const std::array<Real, 3>& point,
        Real max_distance = std::numeric_limits<Real>::infinity()) const override
    {
        return closest(point, std::nullopt, max_distance);
    }

    [[nodiscard]] ClosestBoundaryPoint closestBoundaryPointOnMarker(
        int boundary_marker,
        const std::array<Real, 3>& point,
        Real max_distance = std::numeric_limits<Real>::infinity()) const override
    {
        return closest(point, boundary_marker, max_distance);
    }

    void setBoundaries(std::vector<BoundaryPoint> boundaries)
    {
        boundaries_ = std::move(boundaries);
    }

    [[nodiscard]] int buildCount() const noexcept { return build_count_; }

private:
    [[nodiscard]] ClosestBoundaryPoint closest(
        const std::array<Real, 3>& point,
        std::optional<int> marker,
        Real max_distance) const
    {
        ClosestBoundaryPoint out;
        Real best = max_distance;
        for (const auto& boundary : boundaries_) {
            if (marker.has_value() && boundary.marker != *marker) {
                continue;
            }
            const Real d = distance(point, boundary.point);
            if (d < best || (d == best && boundary.face_id < out.face_id)) {
                best = d;
                out.found = true;
                out.face_id = boundary.face_id;
                out.closest_point = boundary.point;
                out.distance = d;
            }
        }
        return out;
    }

    [[nodiscard]] Real distance(const std::array<Real, 3>& a,
                                const std::array<Real, 3>& b) const noexcept
    {
        Real sum = 0.0;
        for (int d = 0; d < dim_; ++d) {
            const Real diff = a[static_cast<std::size_t>(d)] -
                              b[static_cast<std::size_t>(d)];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    int dim_{3};
    mutable int build_count_{0};
    std::vector<BoundaryPoint> boundaries_{};
};

BoundaryDistanceQueryPoint query(BoundaryDistanceQueryLocation location,
                                 GlobalIndex entity_id,
                                 std::array<Real, 3> point,
                                 std::size_t ordinal = 0)
{
    return BoundaryDistanceQueryPoint{.location = location,
                                      .entity_id = entity_id,
                                      .point = point,
                                      .local_ordinal = ordinal};
}

} // namespace

TEST(BoundaryDistanceService, SupportsSimpleOneTwoThreeDimensionalQueries)
{
    for (int dim = 1; dim <= 3; ++dim) {
        PointBoundarySearch search(dim);
        search.setBoundaries({BoundaryPoint{.marker = 1,
                                            .face_id = 10,
                                            .point = {0.0, 0.0, 0.0}}});

        BoundaryDistanceService service;
        BoundaryDistanceTarget target;
        target.kind = BoundaryDistanceTargetKind::AllBoundaries;
        const auto result = service.evaluateOne(
            search, target,
            query(BoundaryDistanceQueryLocation::Node, 5, {1.0, 2.0, 2.0}));

        ASSERT_TRUE(result.found);
        EXPECT_EQ(result.nearest_boundary_id, 10);
        if (dim == 1) {
            EXPECT_DOUBLE_EQ(result.distance, 1.0);
        } else if (dim == 2) {
            EXPECT_DOUBLE_EQ(result.distance, std::sqrt(5.0));
        } else {
            EXPECT_DOUBLE_EQ(result.distance, 3.0);
        }
    }
}

TEST(BoundaryDistanceService, PreservesQueryLocationMetadata)
{
    PointBoundarySearch search(3);
    search.setBoundaries({BoundaryPoint{.marker = 1,
                                        .face_id = 20,
                                        .point = {0.0, 0.0, 0.0}}});
    BoundaryDistanceService service;
    BoundaryDistanceTarget target;

    const std::vector<BoundaryDistanceQueryPoint> queries = {
        query(BoundaryDistanceQueryLocation::Node, 1, {1.0, 0.0, 0.0}),
        query(BoundaryDistanceQueryLocation::Cell, 2, {0.0, 2.0, 0.0}),
        query(BoundaryDistanceQueryLocation::QuadraturePoint, 3, {0.0, 0.0, 3.0}, 7)};

    service.rebuild(search, target, queries);
    ASSERT_TRUE(service.cacheValid());
    ASSERT_EQ(search.buildCount(), 1);
    const auto results = service.cachedResults();
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].location, BoundaryDistanceQueryLocation::Node);
    EXPECT_EQ(results[1].location, BoundaryDistanceQueryLocation::Cell);
    EXPECT_EQ(results[2].location, BoundaryDistanceQueryLocation::QuadraturePoint);
    EXPECT_EQ(results[2].local_ordinal, 7u);
    EXPECT_DOUBLE_EQ(results[2].distance, 3.0);
}

TEST(BoundaryDistanceService, FiltersSelectedMarkers)
{
    PointBoundarySearch search(1);
    search.setBoundaries({
        BoundaryPoint{.marker = 1, .face_id = 1, .point = {0.0, 0.0, 0.0}},
        BoundaryPoint{.marker = 2, .face_id = 2, .point = {10.0, 0.0, 0.0}}});

    BoundaryDistanceService service;
    BoundaryDistanceTarget target;
    target.kind = BoundaryDistanceTargetKind::SelectedMarkers;
    target.markers = {1};

    const auto result = service.evaluateOne(
        search, target,
        query(BoundaryDistanceQueryLocation::Cell, 8, {9.0, 0.0, 0.0}));

    ASSERT_TRUE(result.found);
    ASSERT_TRUE(result.nearest_marker.has_value());
    EXPECT_EQ(*result.nearest_marker, 1);
    EXPECT_EQ(result.nearest_boundary_id, 1);
    EXPECT_DOUBLE_EQ(result.distance, 9.0);
}

TEST(BoundaryDistanceService, UsesWallTaggedBoundarySets)
{
    PointBoundarySearch search(2);
    search.setBoundaries({
        BoundaryPoint{.marker = 3, .face_id = 3, .point = {0.0, 0.0, 0.0}},
        BoundaryPoint{.marker = 5, .face_id = 5, .point = {10.0, 0.0, 0.0}}});

    BoundaryDistanceService service;
    service.defineBoundarySet(BoundaryMarkerSet{.name = "walls",
                                                .markers = {3},
                                                .wall_tagged = true});
    service.defineBoundarySet(BoundaryMarkerSet{.name = "open",
                                                .markers = {5},
                                                .wall_tagged = false});

    BoundaryDistanceTarget target;
    target.kind = BoundaryDistanceTargetKind::WallTaggedBoundarySets;
    const auto result = service.evaluateOne(
        search, target,
        query(BoundaryDistanceQueryLocation::Node, 4, {9.0, 0.0, 0.0}));

    ASSERT_TRUE(result.found);
    ASSERT_TRUE(result.nearest_marker.has_value());
    EXPECT_EQ(*result.nearest_marker, 3);
    EXPECT_EQ(result.nearest_boundary_id, 3);
    EXPECT_DOUBLE_EQ(result.distance, 9.0);
}

TEST(BoundaryDistanceService, InvalidatesAndRebuildsAfterMovingGeometry)
{
    PointBoundarySearch search(1);
    search.setBoundaries({BoundaryPoint{.marker = 1,
                                        .face_id = 11,
                                        .point = {0.0, 0.0, 0.0}}});
    BoundaryDistanceService service;
    BoundaryDistanceTarget target;
    const std::vector<BoundaryDistanceQueryPoint> queries = {
        query(BoundaryDistanceQueryLocation::Node, 1, {1.0, 0.0, 0.0})};

    service.rebuild(search, target, queries);
    ASSERT_TRUE(service.cacheValid());
    ASSERT_EQ(service.cachedResults().size(), 1u);
    EXPECT_DOUBLE_EQ(service.cachedResults()[0].distance, 1.0);

    search.setBoundaries({BoundaryPoint{.marker = 1,
                                        .face_id = 11,
                                        .point = {2.0, 0.0, 0.0}}});
    EXPECT_DOUBLE_EQ(service.cachedResults()[0].distance, 1.0);

    service.invalidate("moving mesh coordinates changed");
    EXPECT_FALSE(service.cacheValid());
    EXPECT_EQ(service.lastInvalidationReason(), "moving mesh coordinates changed");
    service.rebuild(search, target, queries);

    ASSERT_TRUE(service.cacheValid());
    EXPECT_DOUBLE_EQ(service.cachedResults()[0].distance, 1.0);
    EXPECT_DOUBLE_EQ(service.cachedResults()[0].nearest_point[0], 2.0);
}

TEST(BoundaryDistanceService, ChoosesDeterministicNearestBoundaryForTies)
{
    PointBoundarySearch search(1);
    search.setBoundaries({
        BoundaryPoint{.marker = 7, .face_id = 70, .point = {0.0, 0.0, 0.0}},
        BoundaryPoint{.marker = 3, .face_id = 30, .point = {2.0, 0.0, 0.0}}});

    BoundaryDistanceService service;
    BoundaryDistanceTarget target;
    target.kind = BoundaryDistanceTargetKind::SelectedMarkers;
    target.markers = {7, 3};

    const auto result = service.evaluateOne(
        search, target,
        query(BoundaryDistanceQueryLocation::QuadraturePoint, 2, {1.0, 0.0, 0.0}));

    ASSERT_TRUE(result.found);
    ASSERT_TRUE(result.nearest_marker.has_value());
    EXPECT_EQ(*result.nearest_marker, 3);
    EXPECT_EQ(result.nearest_boundary_id, 30);
}
