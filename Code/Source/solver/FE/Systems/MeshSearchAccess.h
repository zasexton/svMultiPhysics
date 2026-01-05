/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_MESHSEARCHACCESS_H
#define SVMP_FE_SYSTEMS_MESHSEARCHACCESS_H

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Systems/SearchAccess.h"

#include "Mesh/Mesh.h"

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief ISearchAccess adapter backed by svmp::MeshSearch
 *
 * Mirrors the role of `assembly::MeshAccess` for search queries: Systems-level
 * code depends only on `ISearchAccess`, while this adapter provides the Mesh
 * library implementation.
 */
class MeshSearchAccess final : public ISearchAccess {
public:
    explicit MeshSearchAccess(const svmp::Mesh& mesh);
    MeshSearchAccess(const svmp::Mesh& mesh, svmp::Configuration cfg_override);

    [[nodiscard]] int dimension() const noexcept override;
    void build() const override;

    [[nodiscard]] std::vector<GlobalIndex> verticesInRadius(
        const std::array<Real, 3>& point,
        Real radius) const override;

    [[nodiscard]] PointLocation locatePoint(
        const std::array<Real, 3>& point,
        GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const override;

    [[nodiscard]] NearestVertex nearestVertex(
        const std::array<Real, 3>& point) const override;

    [[nodiscard]] std::vector<VertexNeighbor> kNearestVertices(
        const std::array<Real, 3>& point,
        std::size_t k) const override;

    [[nodiscard]] NearestCell nearestCell(
        const std::array<Real, 3>& point) const override;

    [[nodiscard]] ClosestBoundaryPoint closestBoundaryPoint(
        const std::array<Real, 3>& point,
        Real max_distance = std::numeric_limits<Real>::infinity()) const override;

    [[nodiscard]] ClosestBoundaryPoint closestBoundaryPointOnMarker(
        int boundary_marker,
        const std::array<Real, 3>& point,
        Real max_distance = std::numeric_limits<Real>::infinity()) const override;

private:
    const svmp::Mesh& mesh_;
    bool coord_cfg_override_enabled_{false};
    svmp::Configuration coord_cfg_override_{svmp::Configuration::Reference};

    [[nodiscard]] svmp::Configuration queryConfig() const noexcept;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_SYSTEMS_MESHSEARCHACCESS_H
