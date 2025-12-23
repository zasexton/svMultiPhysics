/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_MESHBASEACCESS_H
#define SVMP_FE_ASSEMBLY_MESHBASEACCESS_H

#include "Assembly/Assembler.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Core/MeshTypes.h"

#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief IMeshAccess adapter for svmp::MeshBase
 *
 * This adapter provides the Assembly module with mesh iteration and connectivity
 * access without baking Mesh dependencies into assembler implementations.
 *
 * Notes:
 * - This class assumes the mesh topology is finalized (faces present) when face
 *   iteration or local-face lookup is used.
 * - Faces correspond to codimension-1 entities:
 *   - in 3D: polygonal faces
 *   - in 2D: edges (stored as Mesh "faces")
 */
class MeshBaseAccess final : public IMeshAccess {
public:
    explicit MeshBaseAccess(const svmp::MeshBase& mesh);
    MeshBaseAccess(const svmp::MeshBase& mesh, svmp::Configuration cfg_override);

    [[nodiscard]] GlobalIndex numCells() const override;
    [[nodiscard]] GlobalIndex numOwnedCells() const override;
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override;
    [[nodiscard]] GlobalIndex numInteriorFaces() const override;
    [[nodiscard]] int dimension() const override;

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override;
    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override;

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override;

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override;

    void getCellCoordinates(GlobalIndex cell_id,
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
    const svmp::MeshBase& mesh_;
    bool coord_cfg_override_enabled_{false};
    svmp::Configuration coord_cfg_override_{};

    mutable bool cell2face_ready_{false};
    mutable std::vector<MeshOffset> cell2face_offsets_;
    mutable std::vector<MeshIndex> cell2face_data_;

    void ensureCellToFace() const;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_ASSEMBLY_MESHBASEACCESS_H
