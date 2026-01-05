/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MeshAccess.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Core/FEException.h"
#include "Dofs/MeshTopologyBuilder.h"

#include "Mesh/Core/MeshBase.h"

#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {

namespace {

ElementType element_type_from_mesh_cell(const svmp::MeshBase& mesh, GlobalIndex cell_id) {
    const auto c = static_cast<svmp::index_t>(cell_id);
    const auto& shape = mesh.cell_shape(c);
    const auto [ptr, count] = mesh.cell_vertices_span(c);
    (void)ptr;
    const auto n = count;

    using svmp::CellFamily;

    switch (shape.family) {
        case CellFamily::Point:
            if (n == 1u) return ElementType::Point1;
            break;
        case CellFamily::Line:
            if (n == 2u) return ElementType::Line2;
            if (n == 3u) return ElementType::Line3;
            break;
        case CellFamily::Triangle:
            if (n == 3u) return ElementType::Triangle3;
            if (n == 6u) return ElementType::Triangle6;
            break;
        case CellFamily::Quad:
            if (n == 4u) return ElementType::Quad4;
            if (n == 8u) return ElementType::Quad8;
            if (n == 9u) return ElementType::Quad9;
            break;
        case CellFamily::Tetra:
            if (n == 4u) return ElementType::Tetra4;
            if (n == 10u) return ElementType::Tetra10;
            break;
        case CellFamily::Hex:
            if (n == 8u) return ElementType::Hex8;
            if (n == 20u) return ElementType::Hex20;
            if (n == 27u) return ElementType::Hex27;
            break;
        case CellFamily::Wedge:
            if (n == 6u) return ElementType::Wedge6;
            if (n == 15u) return ElementType::Wedge15;
            if (n == 18u) return ElementType::Wedge18;
            break;
        case CellFamily::Pyramid:
            if (n == 5u) return ElementType::Pyramid5;
            if (n == 13u) return ElementType::Pyramid13;
            if (n == 14u) return ElementType::Pyramid14;
            break;
        default:
            break;
    }

    throw FEException("MeshAccess: unsupported cell type for assembly",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
}

std::array<Real, 3> vertex_coords_from_mesh(const svmp::MeshBase& mesh,
                                            GlobalIndex node_id,
                                            bool coord_cfg_override_enabled,
                                            svmp::Configuration coord_cfg_override) {
    if (node_id < 0) {
        throw FEException("MeshAccess: node_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto v = static_cast<std::size_t>(node_id);
    if (v >= mesh.n_vertices()) {
        throw FEException("MeshAccess: node_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto cfg = coord_cfg_override_enabled ? coord_cfg_override : mesh.active_configuration();
    const bool use_current = (cfg == svmp::Configuration::Current ||
                              cfg == svmp::Configuration::Deformed) &&
                             mesh.has_current_coords();
    const auto& coords = use_current ? mesh.X_cur() : mesh.X_ref();

    const int dim = mesh.dim();
    const std::size_t stride = static_cast<std::size_t>(std::max(1, dim));
    const std::size_t base = v * stride;
    if (base + stride > coords.size()) {
        throw FEException("MeshAccess: coordinate buffer out of range",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    std::array<Real, 3> x{Real(0), Real(0), Real(0)};
    for (int d = 0; d < dim && d < 3; ++d) {
        x[static_cast<std::size_t>(d)] = coords[base + static_cast<std::size_t>(d)];
    }
    return x;
}

} // namespace

MeshAccess::MeshAccess(const svmp::Mesh& mesh)
    : mesh_(mesh), coord_cfg_override_enabled_(false), coord_cfg_override_(svmp::Configuration::Reference) {}

MeshAccess::MeshAccess(const svmp::Mesh& mesh, svmp::Configuration cfg_override)
    : mesh_(mesh), coord_cfg_override_enabled_(true), coord_cfg_override_(cfg_override) {}

GlobalIndex MeshAccess::numCells() const {
    return static_cast<GlobalIndex>(mesh_.n_cells());
}

GlobalIndex MeshAccess::numOwnedCells() const {
    return static_cast<GlobalIndex>(mesh_.n_owned_cells());
}

GlobalIndex MeshAccess::numBoundaryFaces() const {
    const auto& f2c = mesh_.base().face2cell();
    GlobalIndex count = 0;
    for (std::size_t f = 0; f < f2c.size(); ++f) {
        const auto& fc = f2c[f];
        const bool c0_valid = (fc[0] != svmp::INVALID_INDEX);
        const bool c1_valid = (fc[1] != svmp::INVALID_INDEX);
        if (c0_valid == c1_valid) continue; // interior or invalid
        const auto adj = static_cast<GlobalIndex>(c0_valid ? fc[0] : fc[1]);
        if (!isOwnedCell(adj)) continue;
        ++count;
    }
    return count;
}

GlobalIndex MeshAccess::numInteriorFaces() const {
    const auto& f2c = mesh_.base().face2cell();
    GlobalIndex count = 0;
    for (std::size_t f = 0; f < f2c.size(); ++f) {
        const auto& fc = f2c[f];
        if (fc[0] == svmp::INVALID_INDEX || fc[1] == svmp::INVALID_INDEX) continue;
        if (!mesh_.is_owned_face(static_cast<svmp::index_t>(f))) continue;
        ++count;
    }
    return count;
}

int MeshAccess::dimension() const {
    return mesh_.dim();
}

bool MeshAccess::isOwnedCell(GlobalIndex cell_id) const {
    if (cell_id < 0 || cell_id >= numCells()) return false;
    return mesh_.is_owned_cell(static_cast<svmp::index_t>(cell_id));
}

ElementType MeshAccess::getCellType(GlobalIndex cell_id) const {
    if (cell_id < 0 || cell_id >= numCells()) {
        throw FEException("MeshAccess: cell_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return element_type_from_mesh_cell(mesh_.base(), cell_id);
}

int MeshAccess::getCellDomainId(GlobalIndex cell_id) const {
    if (cell_id < 0 || cell_id >= numCells()) {
        throw FEException("MeshAccess: cell_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    const auto c = static_cast<svmp::index_t>(cell_id);
    return static_cast<int>(mesh_.region_label(c));
}

void MeshAccess::getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const {
    if (cell_id < 0 || cell_id >= numCells()) {
        throw FEException("MeshAccess: cell_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto c = static_cast<svmp::index_t>(cell_id);
    const auto [ptr, count] = mesh_.base().cell_vertices_span(c);
    nodes.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        nodes[i] = static_cast<GlobalIndex>(ptr[i]);
    }
}

std::array<Real, 3> MeshAccess::getNodeCoordinates(GlobalIndex node_id) const {
    return vertex_coords_from_mesh(mesh_.base(), node_id,
                                   coord_cfg_override_enabled_,
                                   coord_cfg_override_);
}

void MeshAccess::getCellCoordinates(GlobalIndex cell_id,
                                    std::vector<std::array<Real, 3>>& coords) const {
    if (cell_id < 0 || cell_id >= numCells()) {
        throw FEException("MeshAccess: cell_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto c = static_cast<svmp::index_t>(cell_id);
    const auto [ptr, count] = mesh_.base().cell_vertices_span(c);
    coords.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        coords[i] = vertex_coords_from_mesh(mesh_.base(), static_cast<GlobalIndex>(ptr[i]),
                                            coord_cfg_override_enabled_,
                                            coord_cfg_override_);
    }
}

void MeshAccess::ensureCellToFace() const {
    if (cell2face_ready_) return;

    if (mesh_.dim() < 1) {
        cell2face_ready_ = true;
        return;
    }
    if (mesh_.n_faces() == 0) {
        throw FEException("MeshAccess: mesh has no faces; call MeshBase::finalize() first",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    const auto& base = mesh_.base();
    const auto csr = dofs::buildCellToFacesRefOrder(
        base.dim(),
        std::span<const MeshOffset>(base.cell2vertex_offsets().data(),
                                    base.cell2vertex_offsets().size()),
        std::span<const MeshIndex>(base.cell2vertex().data(),
                                   base.cell2vertex().size()),
        std::span<const MeshOffset>(base.face2vertex_offsets().data(),
                                    base.face2vertex_offsets().size()),
        std::span<const MeshIndex>(base.face2vertex().data(),
                                   base.face2vertex().size()));

    cell2face_offsets_ = csr.offsets;
    cell2face_data_ = csr.data;
    cell2face_ready_ = true;
}

LocalIndex MeshAccess::getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const {
    if (face_id < 0 || face_id >= static_cast<GlobalIndex>(mesh_.n_faces())) {
        throw FEException("MeshAccess: face_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (cell_id < 0 || cell_id >= numCells()) {
        throw FEException("MeshAccess: cell_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    ensureCellToFace();

    const auto cid = static_cast<std::size_t>(cell_id);
    if (cid + 1 >= cell2face_offsets_.size()) {
        throw FEException("MeshAccess: cell2face offsets out of range",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    const auto begin = static_cast<std::size_t>(cell2face_offsets_[cid]);
    const auto end = static_cast<std::size_t>(cell2face_offsets_[cid + 1]);
    if (begin > end || end > cell2face_data_.size()) {
        throw FEException("MeshAccess: cell2face CSR out of range",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    const auto fid = static_cast<MeshIndex>(face_id);
    for (std::size_t i = begin; i < end; ++i) {
        if (cell2face_data_[i] == fid) {
            return static_cast<LocalIndex>(i - begin);
        }
    }

    throw FEException("MeshAccess: face is not incident to cell",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

int MeshAccess::getBoundaryFaceMarker(GlobalIndex face_id) const {
    if (face_id < 0 || face_id >= static_cast<GlobalIndex>(mesh_.n_faces())) {
        throw FEException("MeshAccess: face_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    const auto f = static_cast<svmp::index_t>(face_id);
    return static_cast<int>(mesh_.base().boundary_label(f));
}

std::pair<GlobalIndex, GlobalIndex> MeshAccess::getInteriorFaceCells(GlobalIndex face_id) const {
    if (face_id < 0 || face_id >= static_cast<GlobalIndex>(mesh_.n_faces())) {
        throw FEException("MeshAccess: face_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto f = static_cast<std::size_t>(face_id);
    const auto& fc = mesh_.base().face2cell().at(f);
    if (fc[0] == svmp::INVALID_INDEX || fc[1] == svmp::INVALID_INDEX) {
        throw FEException("MeshAccess: face is not an interior face",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return {static_cast<GlobalIndex>(fc[0]), static_cast<GlobalIndex>(fc[1])};
}

void MeshAccess::forEachCell(std::function<void(GlobalIndex)> callback) const {
    const auto n_cells = numCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        callback(c);
    }
}

void MeshAccess::forEachOwnedCell(std::function<void(GlobalIndex)> callback) const {
    const auto n_cells = numCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        if (!isOwnedCell(c)) continue;
        callback(c);
    }
}

void MeshAccess::forEachBoundaryFace(
    int marker,
    std::function<void(GlobalIndex, GlobalIndex)> callback) const {
    const auto& f2c = mesh_.base().face2cell();
    const bool match_all = (marker < 0);

    for (std::size_t f = 0; f < f2c.size(); ++f) {
        const auto& fc = f2c[f];
        const bool c0_valid = (fc[0] != svmp::INVALID_INDEX);
        const bool c1_valid = (fc[1] != svmp::INVALID_INDEX);
        if (c0_valid == c1_valid) continue;

        const auto adj_cell = static_cast<GlobalIndex>(c0_valid ? fc[0] : fc[1]);
        if (!isOwnedCell(adj_cell)) continue;

        if (!match_all) {
            const auto lbl = mesh_.base().boundary_label(static_cast<svmp::index_t>(f));
            if (static_cast<int>(lbl) != marker) continue;
        }

        callback(static_cast<GlobalIndex>(f), adj_cell);
    }
}

void MeshAccess::forEachInteriorFace(
    std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const {
    const auto& f2c = mesh_.base().face2cell();
    for (std::size_t f = 0; f < f2c.size(); ++f) {
        if (!mesh_.is_owned_face(static_cast<svmp::index_t>(f))) continue;
        const auto& fc = f2c[f];
        if (fc[0] == svmp::INVALID_INDEX || fc[1] == svmp::INVALID_INDEX) continue;
        callback(static_cast<GlobalIndex>(f),
                 static_cast<GlobalIndex>(fc[0]),
                 static_cast<GlobalIndex>(fc[1]));
    }
}

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
