/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_COMPOSITEMESHACCESS_H
#define SVMP_FE_ASSEMBLY_COMPOSITEMESHACCESS_H

#include "Assembly/MeshAccess.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

struct CompositeMeshParticipant {
    std::string name{};
    const svmp::Mesh* mesh{nullptr};
    std::optional<int> domain_id{};
};

struct CompositeMeshEntityLocation {
    std::size_t participant_index{0};
    std::string participant_name{};
    GlobalIndex local_id{INVALID_GLOBAL_INDEX};
};

struct CompositeBoundaryMarker {
    std::size_t participant_index{0};
    std::string participant_name{};
    int local_marker{0};
    int global_marker{0};
};

class CompositeMeshAccess final : public IMeshAccess {
public:
    explicit CompositeMeshAccess(std::vector<CompositeMeshParticipant> participants);
    CompositeMeshAccess(std::vector<CompositeMeshParticipant> participants,
                        svmp::Configuration cfg_override);

    [[nodiscard]] std::size_t numParticipants() const noexcept;
    [[nodiscard]] const std::string& participantName(std::size_t participant_index) const;
    [[nodiscard]] const svmp::Mesh& participantMesh(std::size_t participant_index) const;
    [[nodiscard]] const std::optional<int>& participantDomainId(std::size_t participant_index) const;
    [[nodiscard]] std::optional<std::size_t> participantIndex(std::string_view name) const noexcept;

    [[nodiscard]] CompositeMeshEntityLocation cellLocation(GlobalIndex cell_id) const;
    [[nodiscard]] CompositeMeshEntityLocation vertexLocation(GlobalIndex vertex_id) const;
    [[nodiscard]] CompositeMeshEntityLocation storedFaceLocation(GlobalIndex face_id) const;
    [[nodiscard]] CompositeMeshEntityLocation boundaryFaceLocation(GlobalIndex boundary_face_id) const;
    [[nodiscard]] CompositeMeshEntityLocation interiorFaceLocation(GlobalIndex interior_face_id) const;

    [[nodiscard]] GlobalIndex globalCellId(std::size_t participant_index,
                                           GlobalIndex local_cell_id) const;
    [[nodiscard]] GlobalIndex globalVertexId(std::size_t participant_index,
                                             GlobalIndex local_vertex_id) const;
    [[nodiscard]] GlobalIndex globalStoredFaceId(std::size_t participant_index,
                                                 GlobalIndex local_face_id) const;

    [[nodiscard]] int globalBoundaryMarker(std::string_view participant_name,
                                           int local_marker) const;
    [[nodiscard]] std::optional<CompositeBoundaryMarker>
    boundaryMarkerInfo(int global_marker) const;

    [[nodiscard]] GlobalIndex numCells() const override;
    [[nodiscard]] GlobalIndex numOwnedCells() const override;
    [[nodiscard]] GlobalIndex numVertices() const override;
    [[nodiscard]] GlobalIndex numOwnedVertices() const override;
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override;
    [[nodiscard]] GlobalIndex numInteriorFaces() const override;
    [[nodiscard]] int dimension() const override;
    [[nodiscard]] bool revisionTrackingAvailable() const override;
    [[nodiscard]] std::uint64_t geometryRevision() const override;
    [[nodiscard]] std::uint64_t topologyRevision() const override;
    [[nodiscard]] std::uint64_t ownershipRevision() const override;
    [[nodiscard]] std::uint64_t numberingRevision() const override;
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override;
    [[nodiscard]] std::uint64_t labelRevision() const override;
    [[nodiscard]] std::uint64_t activeConfigurationEpoch() const override;
    [[nodiscard]] std::uint64_t coordinateConfigurationKey() const override;
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }

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
    struct ParticipantState {
        std::string name{};
        const svmp::Mesh* mesh{nullptr};
        std::optional<int> domain_id{};
        std::unique_ptr<MeshAccess> access{};
        GlobalIndex cell_offset{0};
        GlobalIndex vertex_offset{0};
        GlobalIndex stored_face_offset{0};
        GlobalIndex boundary_face_offset{0};
        GlobalIndex interior_face_offset{0};
        std::vector<GlobalIndex> boundary_faces{};
        std::vector<GlobalIndex> interior_faces{};
        std::unordered_map<int, int> global_marker_by_local_marker{};
    };

    void build(std::vector<CompositeMeshParticipant> participants,
               std::optional<svmp::Configuration> cfg_override);
    [[nodiscard]] const ParticipantState& participant(std::size_t participant_index) const;
    [[nodiscard]] std::size_t locateCell(GlobalIndex cell_id) const;
    [[nodiscard]] std::size_t locateVertex(GlobalIndex vertex_id) const;
    [[nodiscard]] std::size_t locateStoredFace(GlobalIndex face_id) const;
    [[nodiscard]] std::size_t locateBoundaryFace(GlobalIndex boundary_face_id) const;
    [[nodiscard]] std::size_t locateInteriorFace(GlobalIndex interior_face_id) const;
    [[nodiscard]] std::uint64_t combinedRevision(
        std::uint64_t (IMeshAccess::*revision)() const) const;

    std::vector<ParticipantState> participants_{};
    std::unordered_map<std::string, std::size_t> name_to_participant_{};
    std::unordered_map<int, CompositeBoundaryMarker> marker_info_by_global_marker_{};
    GlobalIndex total_cells_{0};
    GlobalIndex total_owned_cells_{0};
    GlobalIndex total_vertices_{0};
    GlobalIndex total_owned_vertices_{0};
    GlobalIndex total_stored_faces_{0};
    GlobalIndex total_boundary_faces_{0};
    GlobalIndex total_interior_faces_{0};
    int dimension_{0};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_ASSEMBLY_COMPOSITEMESHACCESS_H
