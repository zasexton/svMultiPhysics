/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "CompositeMeshAccess.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Core/FEException.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace svmp {
namespace FE {
namespace assembly {

namespace {

FEException composite_error(const std::string& message,
                            FEStatus status = FEStatus::InvalidArgument)
{
    return FEException("CompositeMeshAccess: " + message,
                       __FILE__, __LINE__, __func__, status);
}

void mix_revision(std::uint64_t& key, std::uint64_t value)
{
    key ^= value + 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u);
}

std::string participant_name_for_message(const std::string& name)
{
    return name.empty() ? std::string{"<unnamed>"} : name;
}

bool contains_id(const std::vector<GlobalIndex>& ids, GlobalIndex id)
{
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

} // namespace

CompositeMeshAccess::CompositeMeshAccess(
    std::vector<CompositeMeshParticipant> participants)
{
    build(std::move(participants), std::nullopt);
}

CompositeMeshAccess::CompositeMeshAccess(
    std::vector<CompositeMeshParticipant> participants,
    svmp::Configuration cfg_override)
{
    build(std::move(participants), cfg_override);
}

void CompositeMeshAccess::build(
    std::vector<CompositeMeshParticipant> participants,
    std::optional<svmp::Configuration> cfg_override)
{
    if (participants.empty()) {
        throw composite_error("at least one participant is required");
    }

    participants_.clear();
    name_to_participant_.clear();
    marker_info_by_global_marker_.clear();
    total_cells_ = 0;
    total_owned_cells_ = 0;
    total_vertices_ = 0;
    total_owned_vertices_ = 0;
    total_stored_faces_ = 0;
    total_boundary_faces_ = 0;
    total_interior_faces_ = 0;
    dimension_ = 0;

    int next_global_marker = 1;

    for (auto& participant_request : participants) {
        if (participant_request.name.empty()) {
            throw composite_error("participant name cannot be empty");
        }
        if (participant_request.mesh == nullptr) {
            throw composite_error("participant '" +
                                  participant_name_for_message(participant_request.name) +
                                  "' has no mesh");
        }
        if (name_to_participant_.count(participant_request.name) != 0) {
            throw composite_error("duplicate participant name '" +
                                  participant_request.name + "'");
        }

        ParticipantState state;
        state.name = std::move(participant_request.name);
        state.mesh = participant_request.mesh;
        state.domain_id = participant_request.domain_id;
        if (cfg_override.has_value()) {
            state.access = std::make_unique<MeshAccess>(*state.mesh, *cfg_override);
        } else {
            state.access = std::make_unique<MeshAccess>(*state.mesh);
        }

        if (dimension_ == 0) {
            dimension_ = state.access->dimension();
        } else if (dimension_ != state.access->dimension()) {
            throw composite_error("participant '" + state.name +
                                  "' has dimension " +
                                  std::to_string(state.access->dimension()) +
                                  ", expected " + std::to_string(dimension_));
        }

        state.cell_offset = total_cells_;
        state.vertex_offset = total_vertices_;
        state.stored_face_offset = total_stored_faces_;
        state.boundary_face_offset = total_boundary_faces_;
        state.interior_face_offset = total_interior_faces_;

        state.access->forEachBoundaryFace(
            -1,
            [&](GlobalIndex local_face_id, GlobalIndex) {
                state.boundary_faces.push_back(local_face_id);
                const int local_marker =
                    state.access->getBoundaryFaceMarker(local_face_id);
                if (state.global_marker_by_local_marker.count(local_marker) == 0) {
                    const int global_marker = next_global_marker++;
                    state.global_marker_by_local_marker.emplace(local_marker,
                                                                global_marker);
                    marker_info_by_global_marker_.emplace(
                        global_marker,
                        CompositeBoundaryMarker{
                            participants_.size(),
                            state.name,
                            local_marker,
                            global_marker});
                }
            });

        state.access->forEachInteriorFace(
            [&](GlobalIndex local_face_id, GlobalIndex, GlobalIndex) {
                state.interior_faces.push_back(local_face_id);
            });

        total_cells_ += state.access->numCells();
        total_owned_cells_ += state.access->numOwnedCells();
        total_vertices_ += state.access->numVertices();
        total_owned_vertices_ += state.access->numOwnedVertices();
        total_stored_faces_ += static_cast<GlobalIndex>(state.mesh->n_faces());
        total_boundary_faces_ += static_cast<GlobalIndex>(state.boundary_faces.size());
        total_interior_faces_ += static_cast<GlobalIndex>(state.interior_faces.size());

        const auto participant_index = participants_.size();
        name_to_participant_.emplace(state.name, participant_index);
        participants_.push_back(std::move(state));
    }
}

std::size_t CompositeMeshAccess::numParticipants() const noexcept
{
    return participants_.size();
}

const std::string& CompositeMeshAccess::participantName(
    std::size_t participant_index) const
{
    return participant(participant_index).name;
}

const svmp::Mesh& CompositeMeshAccess::participantMesh(
    std::size_t participant_index) const
{
    return *participant(participant_index).mesh;
}

const std::optional<int>& CompositeMeshAccess::participantDomainId(
    std::size_t participant_index) const
{
    return participant(participant_index).domain_id;
}

std::optional<std::size_t> CompositeMeshAccess::participantIndex(
    std::string_view name) const noexcept
{
    const auto it = name_to_participant_.find(std::string(name));
    if (it == name_to_participant_.end()) {
        return std::nullopt;
    }
    return it->second;
}

CompositeMeshEntityLocation CompositeMeshAccess::cellLocation(
    GlobalIndex cell_id) const
{
    const auto index = locateCell(cell_id);
    const auto& state = participants_[index];
    return {index, state.name, cell_id - state.cell_offset};
}

CompositeMeshEntityLocation CompositeMeshAccess::vertexLocation(
    GlobalIndex vertex_id) const
{
    const auto index = locateVertex(vertex_id);
    const auto& state = participants_[index];
    return {index, state.name, vertex_id - state.vertex_offset};
}

CompositeMeshEntityLocation CompositeMeshAccess::storedFaceLocation(
    GlobalIndex face_id) const
{
    const auto index = locateStoredFace(face_id);
    const auto& state = participants_[index];
    return {index, state.name, face_id - state.stored_face_offset};
}

CompositeMeshEntityLocation CompositeMeshAccess::boundaryFaceLocation(
    GlobalIndex boundary_face_id) const
{
    const auto index = locateBoundaryFace(boundary_face_id);
    const auto& state = participants_[index];
    const auto local_ordinal =
        static_cast<std::size_t>(boundary_face_id - state.boundary_face_offset);
    return {index, state.name, state.boundary_faces.at(local_ordinal)};
}

CompositeMeshEntityLocation CompositeMeshAccess::interiorFaceLocation(
    GlobalIndex interior_face_id) const
{
    const auto index = locateInteriorFace(interior_face_id);
    const auto& state = participants_[index];
    const auto local_ordinal =
        static_cast<std::size_t>(interior_face_id - state.interior_face_offset);
    return {index, state.name, state.interior_faces.at(local_ordinal)};
}

GlobalIndex CompositeMeshAccess::globalCellId(
    std::size_t participant_index,
    GlobalIndex local_cell_id) const
{
    const auto& state = participant(participant_index);
    if (local_cell_id < 0 || local_cell_id >= state.access->numCells()) {
        throw composite_error("local cell id out of range for participant '" +
                              state.name + "'");
    }
    return state.cell_offset + local_cell_id;
}

GlobalIndex CompositeMeshAccess::globalVertexId(
    std::size_t participant_index,
    GlobalIndex local_vertex_id) const
{
    const auto& state = participant(participant_index);
    if (local_vertex_id < 0 || local_vertex_id >= state.access->numVertices()) {
        throw composite_error("local vertex id out of range for participant '" +
                              state.name + "'");
    }
    return state.vertex_offset + local_vertex_id;
}

GlobalIndex CompositeMeshAccess::globalStoredFaceId(
    std::size_t participant_index,
    GlobalIndex local_face_id) const
{
    const auto& state = participant(participant_index);
    if (local_face_id < 0 ||
        local_face_id >= static_cast<GlobalIndex>(state.mesh->n_faces())) {
        throw composite_error("local face id out of range for participant '" +
                              state.name + "'");
    }
    return state.stored_face_offset + local_face_id;
}

int CompositeMeshAccess::globalBoundaryMarker(
    std::string_view participant_name,
    int local_marker) const
{
    const auto participant_index_value = participantIndex(participant_name);
    if (!participant_index_value.has_value()) {
        throw composite_error("unknown participant '" +
                              std::string(participant_name) + "'");
    }
    const auto& state = participants_[*participant_index_value];
    const auto marker_it = state.global_marker_by_local_marker.find(local_marker);
    if (marker_it == state.global_marker_by_local_marker.end()) {
        throw composite_error("participant '" + state.name +
                              "' has no boundary marker " +
                              std::to_string(local_marker));
    }
    return marker_it->second;
}

std::optional<CompositeBoundaryMarker> CompositeMeshAccess::boundaryMarkerInfo(
    int global_marker) const
{
    const auto it = marker_info_by_global_marker_.find(global_marker);
    if (it == marker_info_by_global_marker_.end()) {
        return std::nullopt;
    }
    return it->second;
}

GlobalIndex CompositeMeshAccess::numCells() const
{
    return total_cells_;
}

GlobalIndex CompositeMeshAccess::numOwnedCells() const
{
    return total_owned_cells_;
}

GlobalIndex CompositeMeshAccess::numVertices() const
{
    return total_vertices_;
}

GlobalIndex CompositeMeshAccess::numOwnedVertices() const
{
    return total_owned_vertices_;
}

GlobalIndex CompositeMeshAccess::numBoundaryFaces() const
{
    return total_boundary_faces_;
}

GlobalIndex CompositeMeshAccess::numInteriorFaces() const
{
    return total_interior_faces_;
}

int CompositeMeshAccess::dimension() const
{
    return dimension_;
}

bool CompositeMeshAccess::revisionTrackingAvailable() const
{
    return std::all_of(participants_.begin(), participants_.end(),
                       [](const auto& state) {
                           return state.access->revisionTrackingAvailable();
                       });
}

std::uint64_t CompositeMeshAccess::combinedRevision(
    std::uint64_t (IMeshAccess::*revision)() const) const
{
    std::uint64_t key = 1469598103934665603ull;
    for (const auto& state : participants_) {
        mix_revision(key, static_cast<std::uint64_t>(state.cell_offset));
        mix_revision(key, static_cast<std::uint64_t>(state.vertex_offset));
        mix_revision(key, static_cast<std::uint64_t>(state.stored_face_offset));
        mix_revision(key, (state.access.get()->*revision)());
    }
    return key;
}

std::uint64_t CompositeMeshAccess::geometryRevision() const
{
    return combinedRevision(&IMeshAccess::geometryRevision);
}

std::uint64_t CompositeMeshAccess::topologyRevision() const
{
    return combinedRevision(&IMeshAccess::topologyRevision);
}

std::uint64_t CompositeMeshAccess::ownershipRevision() const
{
    return combinedRevision(&IMeshAccess::ownershipRevision);
}

std::uint64_t CompositeMeshAccess::numberingRevision() const
{
    return combinedRevision(&IMeshAccess::numberingRevision);
}

std::uint64_t CompositeMeshAccess::fieldLayoutRevision() const
{
    return combinedRevision(&IMeshAccess::fieldLayoutRevision);
}

std::uint64_t CompositeMeshAccess::labelRevision() const
{
    return combinedRevision(&IMeshAccess::labelRevision);
}

std::uint64_t CompositeMeshAccess::activeConfigurationEpoch() const
{
    return combinedRevision(&IMeshAccess::activeConfigurationEpoch);
}

std::uint64_t CompositeMeshAccess::coordinateConfigurationKey() const
{
    return combinedRevision(&IMeshAccess::coordinateConfigurationKey);
}

bool CompositeMeshAccess::isOwnedCell(GlobalIndex cell_id) const
{
    const auto location = cellLocation(cell_id);
    const auto& state = participants_[location.participant_index];
    return state.access->isOwnedCell(location.local_id);
}

ElementType CompositeMeshAccess::getCellType(GlobalIndex cell_id) const
{
    const auto location = cellLocation(cell_id);
    const auto& state = participants_[location.participant_index];
    return state.access->getCellType(location.local_id);
}

int CompositeMeshAccess::getCellGeometryOrder(GlobalIndex cell_id) const
{
    const auto location = cellLocation(cell_id);
    const auto& state = participants_[location.participant_index];
    return state.access->getCellGeometryOrder(location.local_id);
}

int CompositeMeshAccess::getCellDomainId(GlobalIndex cell_id) const
{
    const auto location = cellLocation(cell_id);
    const auto& state = participants_[location.participant_index];
    if (state.domain_id.has_value()) {
        return *state.domain_id;
    }
    return state.access->getCellDomainId(location.local_id);
}

void CompositeMeshAccess::getCellNodes(GlobalIndex cell_id,
                                       std::vector<GlobalIndex>& nodes) const
{
    const auto location = cellLocation(cell_id);
    const auto& state = participants_[location.participant_index];
    state.access->getCellNodes(location.local_id, nodes);
    for (auto& node : nodes) {
        node += state.vertex_offset;
    }
}

std::array<Real, 3> CompositeMeshAccess::getNodeCoordinates(
    GlobalIndex node_id) const
{
    const auto location = vertexLocation(node_id);
    const auto& state = participants_[location.participant_index];
    return state.access->getNodeCoordinates(location.local_id);
}

void CompositeMeshAccess::getCellCoordinates(
    GlobalIndex cell_id,
    std::vector<std::array<Real, 3>>& coords) const
{
    const auto location = cellLocation(cell_id);
    const auto& state = participants_[location.participant_index];
    state.access->getCellCoordinates(location.local_id, coords);
}

bool CompositeMeshAccess::supportsCoordinateFrame(CoordinateFrame frame) const
{
    return std::all_of(participants_.begin(), participants_.end(),
                       [frame](const auto& state) {
                           return state.access->supportsCoordinateFrame(frame);
                       });
}

void CompositeMeshAccess::getCellCoordinates(
    GlobalIndex cell_id,
    CoordinateFrame frame,
    std::vector<std::array<Real, 3>>& coords) const
{
    const auto location = cellLocation(cell_id);
    const auto& state = participants_[location.participant_index];
    state.access->getCellCoordinates(location.local_id, frame, coords);
}

LocalIndex CompositeMeshAccess::getLocalFaceIndex(
    GlobalIndex face_id,
    GlobalIndex cell_id) const
{
    const auto face_location = storedFaceLocation(face_id);
    const auto cell_location = cellLocation(cell_id);
    if (face_location.participant_index != cell_location.participant_index) {
        throw composite_error("face and cell belong to different participants");
    }

    const auto& state = participants_[cell_location.participant_index];
    return state.access->getLocalFaceIndex(face_location.local_id,
                                           cell_location.local_id);
}

int CompositeMeshAccess::getBoundaryFaceMarker(GlobalIndex face_id) const
{
    const auto face_location = storedFaceLocation(face_id);
    const auto& state = participants_[face_location.participant_index];
    if (!contains_id(state.boundary_faces, face_location.local_id)) {
        throw composite_error("face is not a boundary face");
    }

    const int local_marker =
        state.access->getBoundaryFaceMarker(face_location.local_id);
    const auto marker_it =
        state.global_marker_by_local_marker.find(local_marker);
    if (marker_it == state.global_marker_by_local_marker.end()) {
        throw composite_error("boundary marker remap is missing");
    }
    return marker_it->second;
}

std::pair<GlobalIndex, GlobalIndex> CompositeMeshAccess::getInteriorFaceCells(
    GlobalIndex face_id) const
{
    const auto face_location = storedFaceLocation(face_id);
    const auto& state = participants_[face_location.participant_index];
    if (!contains_id(state.interior_faces, face_location.local_id)) {
        throw composite_error("face is not an interior face");
    }

    auto cells = state.access->getInteriorFaceCells(face_location.local_id);
    cells.first += state.cell_offset;
    cells.second += state.cell_offset;
    return cells;
}

void CompositeMeshAccess::forEachCell(
    std::function<void(GlobalIndex)> callback) const
{
    for (const auto& state : participants_) {
        state.access->forEachCell(
            [&](GlobalIndex local_cell_id) {
                callback(state.cell_offset + local_cell_id);
            });
    }
}

void CompositeMeshAccess::forEachOwnedCell(
    std::function<void(GlobalIndex)> callback) const
{
    for (const auto& state : participants_) {
        state.access->forEachOwnedCell(
            [&](GlobalIndex local_cell_id) {
                callback(state.cell_offset + local_cell_id);
            });
    }
}

void CompositeMeshAccess::forEachBoundaryFace(
    int marker,
    std::function<void(GlobalIndex, GlobalIndex)> callback) const
{
    const bool match_all = marker < 0;
    for (const auto& state : participants_) {
        state.access->forEachBoundaryFace(
            -1,
            [&](GlobalIndex local_face_id, GlobalIndex local_cell_id) {
                const auto global_face_id =
                    state.stored_face_offset + local_face_id;
                const int global_marker = getBoundaryFaceMarker(global_face_id);
                if (!match_all && global_marker != marker) {
                    return;
                }
                callback(global_face_id, state.cell_offset + local_cell_id);
            });
    }
}

void CompositeMeshAccess::forEachInteriorFace(
    std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const
{
    for (const auto& state : participants_) {
        state.access->forEachInteriorFace(
            [&](GlobalIndex local_face_id,
                GlobalIndex local_cell_0,
                GlobalIndex local_cell_1) {
                callback(state.stored_face_offset + local_face_id,
                         state.cell_offset + local_cell_0,
                         state.cell_offset + local_cell_1);
            });
    }
}

const CompositeMeshAccess::ParticipantState& CompositeMeshAccess::participant(
    std::size_t participant_index) const
{
    if (participant_index >= participants_.size()) {
        throw composite_error("participant index out of range");
    }
    return participants_[participant_index];
}

std::size_t CompositeMeshAccess::locateCell(GlobalIndex cell_id) const
{
    if (cell_id < 0 || cell_id >= total_cells_) {
        throw composite_error("cell id out of range");
    }
    for (std::size_t i = 0; i < participants_.size(); ++i) {
        const auto& state = participants_[i];
        const auto begin = state.cell_offset;
        const auto end = begin + state.access->numCells();
        if (cell_id >= begin && cell_id < end) {
            return i;
        }
    }
    throw composite_error("cell id has no participant",
                          FEStatus::AssemblyError);
}

std::size_t CompositeMeshAccess::locateVertex(GlobalIndex vertex_id) const
{
    if (vertex_id < 0 || vertex_id >= total_vertices_) {
        throw composite_error("vertex id out of range");
    }
    for (std::size_t i = 0; i < participants_.size(); ++i) {
        const auto& state = participants_[i];
        const auto begin = state.vertex_offset;
        const auto end = begin + state.access->numVertices();
        if (vertex_id >= begin && vertex_id < end) {
            return i;
        }
    }
    throw composite_error("vertex id has no participant",
                          FEStatus::AssemblyError);
}

std::size_t CompositeMeshAccess::locateStoredFace(GlobalIndex face_id) const
{
    if (face_id < 0 || face_id >= total_stored_faces_) {
        throw composite_error("face id out of range");
    }
    for (std::size_t i = 0; i < participants_.size(); ++i) {
        const auto& state = participants_[i];
        const auto begin = state.stored_face_offset;
        const auto end =
            begin + static_cast<GlobalIndex>(state.mesh->n_faces());
        if (face_id >= begin && face_id < end) {
            return i;
        }
    }
    throw composite_error("face id has no participant",
                          FEStatus::AssemblyError);
}

std::size_t CompositeMeshAccess::locateBoundaryFace(
    GlobalIndex boundary_face_id) const
{
    if (boundary_face_id < 0 || boundary_face_id >= total_boundary_faces_) {
        throw composite_error("boundary face id out of range");
    }
    for (std::size_t i = 0; i < participants_.size(); ++i) {
        const auto& state = participants_[i];
        const auto begin = state.boundary_face_offset;
        const auto end =
            begin + static_cast<GlobalIndex>(state.boundary_faces.size());
        if (boundary_face_id >= begin && boundary_face_id < end) {
            return i;
        }
    }
    throw composite_error("boundary face id has no participant",
                          FEStatus::AssemblyError);
}

std::size_t CompositeMeshAccess::locateInteriorFace(
    GlobalIndex interior_face_id) const
{
    if (interior_face_id < 0 || interior_face_id >= total_interior_faces_) {
        throw composite_error("interior face id out of range");
    }
    for (std::size_t i = 0; i < participants_.size(); ++i) {
        const auto& state = participants_[i];
        const auto begin = state.interior_face_offset;
        const auto end =
            begin + static_cast<GlobalIndex>(state.interior_faces.size());
        if (interior_face_id >= begin && interior_face_id < end) {
            return i;
        }
    }
    throw composite_error("interior face id has no participant",
                          FEStatus::AssemblyError);
}

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
