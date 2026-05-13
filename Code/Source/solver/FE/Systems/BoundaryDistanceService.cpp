/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/BoundaryDistanceService.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] std::vector<int> sortedUnique(std::vector<int> markers)
{
    std::sort(markers.begin(), markers.end());
    markers.erase(std::unique(markers.begin(), markers.end()), markers.end());
    return markers;
}

[[nodiscard]] bool betterHit(const ISearchAccess::ClosestBoundaryPoint& candidate,
                             std::optional<int> candidate_marker,
                             const BoundaryDistanceResult& current)
{
    if (!candidate.found) {
        return false;
    }
    if (!current.found) {
        return true;
    }

    const Real scale = std::max<Real>({1.0, std::abs(candidate.distance),
                                       std::abs(current.distance)});
    const Real tol = static_cast<Real>(1.0e-14) * scale;
    if (candidate.distance + tol < current.distance) {
        return true;
    }
    if (current.distance + tol < candidate.distance) {
        return false;
    }

    const int current_marker = current.nearest_marker.value_or(
        std::numeric_limits<int>::max());
    const int marker = candidate_marker.value_or(std::numeric_limits<int>::max());
    if (marker != current_marker) {
        return marker < current_marker;
    }
    return candidate.face_id < current.nearest_boundary_id;
}

void copyHit(const ISearchAccess::ClosestBoundaryPoint& hit,
             std::optional<int> marker,
             BoundaryDistanceResult& result)
{
    result.found = true;
    result.nearest_boundary_id = hit.face_id;
    result.nearest_marker = marker;
    result.nearest_point = hit.closest_point;
    result.distance = hit.distance;
}

} // namespace

void BoundaryDistanceService::defineBoundarySet(BoundaryMarkerSet marker_set)
{
    FE_THROW_IF(marker_set.name.empty(), InvalidArgumentException,
                "BoundaryDistanceService::defineBoundarySet: set name is empty");
    marker_set.markers = sortedUnique(std::move(marker_set.markers));
    FE_THROW_IF(marker_set.markers.empty(), InvalidArgumentException,
                "BoundaryDistanceService::defineBoundarySet: marker set is empty");

    for (auto& existing : boundary_sets_) {
        if (existing.name == marker_set.name) {
            existing = std::move(marker_set);
            invalidate("boundary marker set changed");
            return;
        }
    }
    boundary_sets_.push_back(std::move(marker_set));
    invalidate("boundary marker set added");
}

void BoundaryDistanceService::clearBoundarySets()
{
    boundary_sets_.clear();
    invalidate("boundary marker sets cleared");
}

bool BoundaryDistanceService::hasBoundarySet(std::string_view name) const noexcept
{
    return boundarySet(name) != nullptr;
}

const BoundaryMarkerSet* BoundaryDistanceService::boundarySet(std::string_view name) const noexcept
{
    for (const auto& marker_set : boundary_sets_) {
        if (marker_set.name == name) {
            return &marker_set;
        }
    }
    return nullptr;
}

void BoundaryDistanceService::invalidate(std::string reason)
{
    cache_valid_ = false;
    ++cache_revision_;
    last_invalidation_reason_ = std::move(reason);
}

void BoundaryDistanceService::rebuild(
    const ISearchAccess& search,
    const BoundaryDistanceTarget& target,
    std::span<const BoundaryDistanceQueryPoint> queries)
{
    search.build();
    cached_results_ = evaluate(search, target, queries);
    cache_valid_ = true;
    last_rebuild_revision_ = cache_revision_;
}

std::vector<BoundaryDistanceResult> BoundaryDistanceService::evaluate(
    const ISearchAccess& search,
    const BoundaryDistanceTarget& target,
    std::span<const BoundaryDistanceQueryPoint> queries) const
{
    std::vector<BoundaryDistanceResult> results;
    results.reserve(queries.size());
    for (const auto& query : queries) {
        results.push_back(evaluateOne(search, target, query));
    }
    return results;
}

BoundaryDistanceResult BoundaryDistanceService::evaluateOne(
    const ISearchAccess& search,
    const BoundaryDistanceTarget& target,
    const BoundaryDistanceQueryPoint& query) const
{
    FE_THROW_IF(std::isnan(target.max_distance) || target.max_distance < 0.0,
                InvalidArgumentException,
                "BoundaryDistanceService::evaluateOne: max_distance must be nonnegative");

    BoundaryDistanceResult result;
    result.location = query.location;
    result.entity_id = query.entity_id;
    result.local_ordinal = query.local_ordinal;
    result.query_point = query.point;

    if (target.kind == BoundaryDistanceTargetKind::AllBoundaries) {
        const auto hit = search.closestBoundaryPoint(query.point, target.max_distance);
        if (hit.found) {
            copyHit(hit, std::nullopt, result);
        }
        return result;
    }

    const auto markers = targetMarkers(target);
    for (const int marker : markers) {
        const auto hit = search.closestBoundaryPointOnMarker(marker,
                                                            query.point,
                                                            target.max_distance);
        if (betterHit(hit, marker, result)) {
            copyHit(hit, marker, result);
        }
    }
    return result;
}

std::vector<int> BoundaryDistanceService::targetMarkers(
    const BoundaryDistanceTarget& target) const
{
    switch (target.kind) {
        case BoundaryDistanceTargetKind::AllBoundaries:
            return {};
        case BoundaryDistanceTargetKind::SelectedMarkers:
            FE_THROW_IF(target.markers.empty(), InvalidArgumentException,
                        "BoundaryDistanceService: selected marker target is empty");
            return sortedUnique(target.markers);
        case BoundaryDistanceTargetKind::BoundarySet: {
            const auto* marker_set = boundarySet(target.marker_set_name);
            FE_THROW_IF(marker_set == nullptr, InvalidArgumentException,
                        "BoundaryDistanceService: unknown boundary marker set '" +
                            target.marker_set_name + "'");
            return marker_set->markers;
        }
        case BoundaryDistanceTargetKind::WallTaggedBoundarySets: {
            std::vector<int> markers;
            for (const auto& marker_set : boundary_sets_) {
                if (marker_set.wall_tagged) {
                    markers.insert(markers.end(),
                                   marker_set.markers.begin(),
                                   marker_set.markers.end());
                }
            }
            FE_THROW_IF(markers.empty(), InvalidArgumentException,
                        "BoundaryDistanceService: no wall-tagged boundary marker sets");
            return sortedUnique(std::move(markers));
        }
    }
    return {};
}

} // namespace systems
} // namespace FE
} // namespace svmp
