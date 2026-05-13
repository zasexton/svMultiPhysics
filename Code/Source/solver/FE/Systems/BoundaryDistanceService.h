/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_BOUNDARYDISTANCESERVICE_H
#define SVMP_FE_SYSTEMS_BOUNDARYDISTANCESERVICE_H

#include "Core/Types.h"
#include "Systems/SearchAccess.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

enum class BoundaryDistanceQueryLocation : std::uint8_t {
    Node,
    Cell,
    QuadraturePoint
};

enum class BoundaryDistanceTargetKind : std::uint8_t {
    AllBoundaries,
    SelectedMarkers,
    BoundarySet,
    WallTaggedBoundarySets
};

struct BoundaryMarkerSet {
    std::string name;
    std::vector<int> markers;
    bool wall_tagged{false};
};

struct BoundaryDistanceTarget {
    BoundaryDistanceTargetKind kind{BoundaryDistanceTargetKind::AllBoundaries};
    std::vector<int> markers;
    std::string marker_set_name;
    Real max_distance{std::numeric_limits<Real>::infinity()};
};

struct BoundaryDistanceQueryPoint {
    BoundaryDistanceQueryLocation location{BoundaryDistanceQueryLocation::Node};
    GlobalIndex entity_id{INVALID_GLOBAL_INDEX};
    std::array<Real, 3> point{0.0, 0.0, 0.0};
    std::size_t local_ordinal{0};
};

struct BoundaryDistanceResult {
    bool found{false};
    BoundaryDistanceQueryLocation location{BoundaryDistanceQueryLocation::Node};
    GlobalIndex entity_id{INVALID_GLOBAL_INDEX};
    std::size_t local_ordinal{0};
    std::array<Real, 3> query_point{0.0, 0.0, 0.0};
    GlobalIndex nearest_boundary_id{INVALID_GLOBAL_INDEX};
    std::optional<int> nearest_marker{};
    std::array<Real, 3> nearest_point{0.0, 0.0, 0.0};
    Real distance{std::numeric_limits<Real>::infinity()};
};

class BoundaryDistanceService {
public:
    void defineBoundarySet(BoundaryMarkerSet marker_set);
    void clearBoundarySets();

    [[nodiscard]] bool hasBoundarySet(std::string_view name) const noexcept;
    [[nodiscard]] const BoundaryMarkerSet* boundarySet(std::string_view name) const noexcept;
    [[nodiscard]] std::span<const BoundaryMarkerSet> boundarySets() const noexcept
    {
        return boundary_sets_;
    }

    void invalidate(std::string reason = {});
    [[nodiscard]] bool cacheValid() const noexcept { return cache_valid_; }
    [[nodiscard]] std::uint64_t cacheRevision() const noexcept { return cache_revision_; }
    [[nodiscard]] std::uint64_t lastRebuildRevision() const noexcept
    {
        return last_rebuild_revision_;
    }
    [[nodiscard]] const std::string& lastInvalidationReason() const noexcept
    {
        return last_invalidation_reason_;
    }

    void rebuild(const ISearchAccess& search,
                 const BoundaryDistanceTarget& target,
                 std::span<const BoundaryDistanceQueryPoint> queries);

    [[nodiscard]] std::vector<BoundaryDistanceResult> evaluate(
        const ISearchAccess& search,
        const BoundaryDistanceTarget& target,
        std::span<const BoundaryDistanceQueryPoint> queries) const;

    [[nodiscard]] BoundaryDistanceResult evaluateOne(
        const ISearchAccess& search,
        const BoundaryDistanceTarget& target,
        const BoundaryDistanceQueryPoint& query) const;

    [[nodiscard]] std::span<const BoundaryDistanceResult> cachedResults() const noexcept
    {
        return cached_results_;
    }

private:
    std::vector<BoundaryMarkerSet> boundary_sets_{};
    std::vector<BoundaryDistanceResult> cached_results_{};
    bool cache_valid_{false};
    std::uint64_t cache_revision_{0};
    std::uint64_t last_rebuild_revision_{0};
    std::string last_invalidation_reason_{};

    [[nodiscard]] std::vector<int> targetMarkers(
        const BoundaryDistanceTarget& target) const;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_BOUNDARYDISTANCESERVICE_H
