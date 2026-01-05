/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SEARCHACCESS_H
#define SVMP_FE_SYSTEMS_SEARCHACCESS_H

#include "Core/Types.h"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Minimal spatial search interface for FE global evaluations
 *
 * This interface allows Systems-level kernels (e.g., contact, global constraints,
 * nonlocal closures) to query neighborhood information without directly
 * depending on the Mesh library. A Mesh-backed implementation can be provided
 * when `SVMP_FE_WITH_MESH` is enabled.
 */
class ISearchAccess {
public:
    struct PointLocation {
        bool found{false};
        GlobalIndex cell_id{INVALID_GLOBAL_INDEX};
        std::array<Real, 3> xi{0.0, 0.0, 0.0};
    };

    struct NearestVertex {
        bool found{false};
        GlobalIndex vertex_id{INVALID_GLOBAL_INDEX};
        Real distance{std::numeric_limits<Real>::infinity()};
    };

    struct VertexNeighbor {
        GlobalIndex vertex_id{INVALID_GLOBAL_INDEX};
        Real distance{std::numeric_limits<Real>::infinity()};
    };

    struct NearestCell {
        bool found{false};
        GlobalIndex cell_id{INVALID_GLOBAL_INDEX};
        Real distance{std::numeric_limits<Real>::infinity()};
    };

    struct ClosestBoundaryPoint {
        bool found{false};
        GlobalIndex face_id{INVALID_GLOBAL_INDEX};
        std::array<Real, 3> closest_point{0.0, 0.0, 0.0};
        Real distance{std::numeric_limits<Real>::infinity()};
    };

    virtual ~ISearchAccess() = default;

    [[nodiscard]] virtual int dimension() const noexcept = 0;

    /**
     * @brief (Re)build any internal search structures for the active geometry.
     *
     * Implementations may be a no-op if they do not use acceleration structures.
     */
    virtual void build() const = 0;

    /**
     * @brief Return mesh vertices within a given radius of a query point.
     */
    [[nodiscard]] virtual std::vector<GlobalIndex> verticesInRadius(
        const std::array<Real, 3>& point,
        Real radius) const = 0;

    /**
     * @brief Locate a point in the mesh (find containing cell + reference coords)
     *
     * Default implementation returns found=false.
     */
    [[nodiscard]] virtual PointLocation locatePoint(
        const std::array<Real, 3>& /*point*/,
        GlobalIndex /*hint_cell*/ = INVALID_GLOBAL_INDEX) const
    {
        return {};
    }

    /**
     * @brief Find the nearest vertex to a point.
     *
     * Default implementation returns found=false.
     */
    [[nodiscard]] virtual NearestVertex nearestVertex(
        const std::array<Real, 3>& /*point*/) const
    {
        return {};
    }

    /**
     * @brief Find k nearest vertices to a point.
     *
     * Default implementation returns an empty list.
     */
    [[nodiscard]] virtual std::vector<VertexNeighbor> kNearestVertices(
        const std::array<Real, 3>& /*point*/,
        std::size_t /*k*/) const
    {
        return {};
    }

    /**
     * @brief Find the nearest cell to a point (useful for out-of-domain queries).
     *
     * Default implementation returns found=false.
     */
    [[nodiscard]] virtual NearestCell nearestCell(
        const std::array<Real, 3>& /*point*/) const
    {
        return {};
    }

    /**
     * @brief Closest point query on the mesh boundary
     *
     * Default implementation returns found=false; Mesh-backed implementations
     * may support this via boundary triangle acceleration.
     */
    [[nodiscard]] virtual ClosestBoundaryPoint closestBoundaryPoint(
        const std::array<Real, 3>& /*point*/,
        Real /*max_distance*/ = std::numeric_limits<Real>::infinity()) const
    {
        return {};
    }

    /**
     * @brief Closest point query restricted to a boundary marker
     *
     * Default implementation returns found=false.
     */
    [[nodiscard]] virtual ClosestBoundaryPoint closestBoundaryPointOnMarker(
        int /*boundary_marker*/,
        const std::array<Real, 3>& /*point*/,
        Real /*max_distance*/ = std::numeric_limits<Real>::infinity()) const
    {
        return {};
    }
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SEARCHACCESS_H
