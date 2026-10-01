/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_MESHACCESS_H
#define SVMP_FE_ASSEMBLY_MESHACCESS_H

#include "Assembly/Assembler.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Mesh.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief IMeshAccess adapter for the unified svmp::Mesh (DistributedMesh)
 *
 * This adapter provides the Assembly module with mesh iteration and connectivity
 * access without baking Mesh dependencies into assembler implementations.
 *
 * Notes:
 * - This class assumes the mesh topology is finalized (faces present) when face
 *   iteration or local-face lookup is used.
 * - Cells/faces are interpreted as the *local* partition (owned + ghosts).
 * - Ownership queries and "owned-only" iteration are driven by DistributedMesh
 *   ownership metadata (serial-safe; defaults to all-owned in non-MPI builds).
 */
class MeshAccess final : public IMeshAccess {
public:
    explicit MeshAccess(const svmp::Mesh& mesh);
    MeshAccess(const svmp::Mesh& mesh, svmp::Configuration cfg_override);

    [[nodiscard]] GlobalIndex numCells() const override;
    [[nodiscard]] GlobalIndex numOwnedCells() const override;
    [[nodiscard]] GlobalIndex numVertices() const override;
    [[nodiscard]] GlobalIndex numOwnedVertices() const override;
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override;
    [[nodiscard]] GlobalIndex numInteriorFaces() const override;
    [[nodiscard]] int dimension() const override;
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override;
    [[nodiscard]] std::uint64_t topologyRevision() const override;
    [[nodiscard]] std::uint64_t ownershipRevision() const override;
    [[nodiscard]] std::uint64_t numberingRevision() const override;
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override;
    [[nodiscard]] std::uint64_t labelRevision() const override;
    [[nodiscard]] std::uint64_t activeConfigurationEpoch() const override;
    [[nodiscard]] std::uint64_t coordinateConfigurationKey() const override;
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override;
    [[nodiscard]] GlobalIndex getCellGlobalId(GlobalIndex cell_id) const override;
    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(GlobalIndex face_id) const override;
    [[nodiscard]] int parallelRank() const override;
    [[nodiscard]] int parallelSize() const override;
    [[nodiscard]] int getCellOwnerRank(GlobalIndex cell_id) const override;
    [[nodiscard]] int getBoundaryFaceOwnerRank(
        GlobalIndex face_id, GlobalIndex parent_cell) const override;

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override;
    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override;
    [[nodiscard]] int getCellGeometryOrder(GlobalIndex cell_id) const override;
    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override;

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override;

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override;

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override;

    [[nodiscard]] bool supportsCoordinateFrame(CoordinateFrame frame) const override;
    void getCellCoordinates(GlobalIndex cell_id,
                            CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override;

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override;

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override;

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override;

    void forEachCell(std::function<void(GlobalIndex)> callback) const override;
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override;

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override;

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override;

private:
    const svmp::Mesh& mesh_;
    bool coord_cfg_override_enabled_{false};
    svmp::Configuration coord_cfg_override_{};

    mutable bool cell2face_ready_{false};
    mutable std::vector<MeshOffset> cell2face_offsets_;
    mutable std::vector<MeshIndex> cell2face_data_;

    // Memoized globalEntityIdsAvailable().  The answer is reused only while
    // the mesh topology, numbering and ownership revisions and the cell/face
    // id storage (size and address) are unchanged; otherwise it is recomputed.
    // A copied MeshAccess starts with an empty cache.
    struct GlobalEntityIdCache {
        std::mutex mutex{};
        bool valid{false};
        bool available{false};
        std::uint64_t topology_revision{0};
        std::uint64_t numbering_revision{0};
        std::uint64_t ownership_revision{0};
        std::size_t n_cells{0};
        std::size_t n_faces{0};
        const void* cell_ids{nullptr};
        const void* face_ids{nullptr};
        std::size_t cell_id_count{0};
        std::size_t face_id_count{0};

        GlobalEntityIdCache() = default;
        GlobalEntityIdCache(const GlobalEntityIdCache&) {}
        GlobalEntityIdCache& operator=(const GlobalEntityIdCache&) = delete;
    };
    mutable GlobalEntityIdCache global_entity_id_cache_{};

    void ensureCellToFace() const;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_ASSEMBLY_MESHACCESS_H
