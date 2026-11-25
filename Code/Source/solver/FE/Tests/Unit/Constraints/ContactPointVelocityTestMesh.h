/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_CONTACTPOINTVELOCITYTESTMESH_H
#define SVMP_FE_TESTS_CONTACTPOINTVELOCITYTESTMESH_H

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"

#include <array>
#include <functional>
#include <vector>

namespace svmp::FE::constraints::test {

inline constexpr int kWallMarker = 11;

// Strip [0,2]x[0,1] in four triangles; global vertices
//   0 (0,0)  1 (1,0)  2 (2,0)  3 (0,1)  4 (1,1)  5 (2,1),
// cells T0 (0,1,4), T1 (0,4,3) on rank 0 and T2 (1,2,5), T3 (1,5,4) on rank 1.
// The bottom faces (0,1) of T0 and (1,2) of T2 carry the wall marker.
// With owned_only each rank sees only its own cells, with rank-local vertex numbers.
class StripMeshAccess final : public assembly::IMeshAccess {
public:
    explicit StripMeshAccess(int rank = 0, int size = 1, bool owned_only = false)
        : rank_(rank), size_(size), owned_only_(owned_only)
    {
        const std::array<std::array<Real, 3>, 6> all{{
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {2.0, 1.0, 0.0},
        }};
        if (owned_only_ && rank_ == 0) {
            gids_ = {0, 1, 3, 4};
            cells_ = {{0, 1, 3}, {0, 3, 2}};
            cell_gids_ = {0, 1};
        } else if (owned_only_) {
            gids_ = {1, 2, 4, 5};
            cells_ = {{0, 1, 3}, {0, 3, 2}};
            cell_gids_ = {2, 3};
        } else {
            gids_ = {0, 1, 2, 3, 4, 5};
            cells_ = {{0, 1, 4}, {0, 4, 3}, {1, 2, 5}, {1, 5, 4}};
            cell_gids_ = {0, 1, 2, 3};
        }
        for (const auto g : gids_) {
            nodes_.push_back(all[static_cast<std::size_t>(g)]);
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        GlobalIndex n = 0;
        forEachOwnedCell([&](GlobalIndex) { ++n; });
        return n;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return owned_only_ ? 1 : 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override { return true; }
    [[nodiscard]] GlobalIndex getCellGlobalId(GlobalIndex cell) const override
    {
        return cell_gids_.at(static_cast<std::size_t>(cell));
    }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }
    [[nodiscard]] int getCellOwnerRank(GlobalIndex cell) const override
    {
        return size_ == 1 || getCellGlobalId(cell) < 2 ? 0 : 1;
    }
    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(GlobalIndex face) const override
    {
        return owned_only_ ? rank_ : face;
    }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return getCellOwnerRank(cell) == rank_;
    }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Triangle3;
    }
    void getCellNodes(GlobalIndex cell, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(c.begin(), c.end());
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node) const override
    {
        return nodes_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(GlobalIndex cell,
                            std::vector<std::array<Real, 3>>& coordinates) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell));
        coordinates.clear();
        for (const auto n : c) {
            coordinates.push_back(nodes_.at(static_cast<std::size_t>(n)));
        }
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override
    {
        return LocalIndex{0};  // the bottom edge is local edge (0,1)
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return kWallMarker; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {-1, -1};
    }
    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                callback(c);
            }
        }
    }
    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker >= 0 && marker != kWallMarker) {
            return;
        }
        callback(0, 0);  // first visible wall triangle
        if (!owned_only_) {
            callback(1, 2);  // T2
        }
    }
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override
    {
    }

    [[nodiscard]] const std::vector<GlobalIndex>& gids() const noexcept { return gids_; }

private:
    int rank_{0};
    int size_{1};
    bool owned_only_{false};
    std::vector<GlobalIndex> gids_{};
    std::vector<std::array<GlobalIndex, 3>> cells_{};
    std::vector<GlobalIndex> cell_gids_{};
    std::vector<std::array<Real, 3>> nodes_{};
};

inline dofs::MeshTopologyInfo topologyFor(const StripMeshAccess& mesh)
{
    dofs::MeshTopologyInfo t;
    t.dim = 2;
    t.n_cells = static_cast<GlobalIndex>(mesh.numCells());
    t.n_vertices = static_cast<GlobalIndex>(mesh.gids().size());
    t.cell2vertex_offsets.push_back(0);
    std::vector<GlobalIndex> nodes;
    for (GlobalIndex c = 0; c < mesh.numCells(); ++c) {
        mesh.getCellNodes(c, nodes);
        for (const auto n : nodes) {
            t.cell2vertex_data.push_back(static_cast<MeshIndex>(n));
        }
        t.cell2vertex_offsets.push_back(static_cast<MeshOffset>(t.cell2vertex_data.size()));
        t.cell_gids.push_back(mesh.getCellGlobalId(c));
        t.cell_owner_ranks.push_back(mesh.getCellOwnerRank(c));
    }
    for (std::size_t v = 0; v < mesh.gids().size(); ++v) {
        t.vertex_gids.push_back(mesh.gids()[v]);
        const auto x = mesh.getNodeCoordinates(static_cast<GlobalIndex>(v));
        t.vertex_coords.insert(t.vertex_coords.end(), {x[0], x[1]});
    }
    if (mesh.parallelSize() == 2) {
        t.neighbor_ranks = {1 - mesh.parallelRank()};
    }
    return t;
}

// phi = (x - 1.3) - 0.2 y: the contact point is at x = 1.3 on edge (1,2),
// s = 0.3 from the wet vertex 1; cos(theta) = -n.n_wall = n_y.
inline Real phiAt(const std::array<Real, 3>& x) { return (x[0] - 1.3) - 0.2 * x[1]; }
inline Real law(Real cos_theta) { return 0.5 - cos_theta; }

} // namespace svmp::FE::constraints::test

#endif // SVMP_FE_TESTS_CONTACTPOINTVELOCITYTESTMESH_H
