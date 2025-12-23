/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MeshTopologyBuilder.h"

#include "Core/FEException.h"
#include "Elements/ReferenceElement.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace dofs {

namespace {

ElementType infer_element_type_from_cell(int dim, std::size_t n_verts) {
    if (dim == 0 && n_verts == 1) return ElementType::Point1;
    if (dim == 1) {
        if (n_verts == 2) return ElementType::Line2;
        if (n_verts == 3) return ElementType::Line3;
    }
    if (dim == 2) {
        if (n_verts == 3) return ElementType::Triangle3;
        if (n_verts == 6) return ElementType::Triangle6;
        if (n_verts == 4) return ElementType::Quad4;
        if (n_verts == 8) return ElementType::Quad8;
        if (n_verts == 9) return ElementType::Quad9;
    }
    if (dim == 3) {
        if (n_verts == 4) return ElementType::Tetra4;
        if (n_verts == 10) return ElementType::Tetra10;
        if (n_verts == 8) return ElementType::Hex8;
        if (n_verts == 20) return ElementType::Hex20;
        if (n_verts == 27) return ElementType::Hex27;
        if (n_verts == 6) return ElementType::Wedge6;
        if (n_verts == 15) return ElementType::Wedge15;
        if (n_verts == 18) return ElementType::Wedge18;
        if (n_verts == 5) return ElementType::Pyramid5;
        if (n_verts == 13) return ElementType::Pyramid13;
        if (n_verts == 14) return ElementType::Pyramid14;
    }
    return ElementType::Unknown;
}

struct EnumHash {
    std::size_t operator()(ElementType t) const noexcept {
        return static_cast<std::size_t>(t);
    }
};

struct EdgeRecord {
    MeshIndex a{0};
    MeshIndex b{0};
    MeshIndex id{0};
};

bool operator<(const EdgeRecord& lhs, const EdgeRecord& rhs) {
    if (lhs.a != rhs.a) return lhs.a < rhs.a;
    return lhs.b < rhs.b;
}

struct FaceRecord {
    std::uint8_t n{0};
    std::array<MeshIndex, 4> verts{};
    MeshIndex id{0};
};

bool operator<(const FaceRecord& lhs, const FaceRecord& rhs) {
    if (lhs.n != rhs.n) return lhs.n < rhs.n;
    return lhs.verts < rhs.verts;
}

std::span<const MeshIndex> cell_vertices_span(std::span<const MeshOffset> offsets,
                                              std::span<const MeshIndex> data,
                                              GlobalIndex cell_id) {
    if (cell_id < 0) return {};
    const auto c = static_cast<std::size_t>(cell_id);
    if (c + 1 >= offsets.size()) return {};
    const auto begin = static_cast<std::size_t>(offsets[c]);
    const auto end = static_cast<std::size_t>(offsets[c + 1]);
    if (begin > end || end > data.size()) return {};
    return {data.data() + begin, end - begin};
}

} // namespace

CellToEntityCSR buildCellToEdgesRefOrder(
    int dim,
    std::span<const MeshOffset> cell2vertex_offsets,
    std::span<const MeshIndex> cell2vertex,
    std::span<const std::array<MeshIndex, 2>> edge2vertex) {

    const auto n_cells = static_cast<GlobalIndex>(
        cell2vertex_offsets.empty() ? 0 : static_cast<GlobalIndex>(cell2vertex_offsets.size() - 1));

    CellToEntityCSR out;
    out.offsets.resize(static_cast<std::size_t>(n_cells) + 1u, MeshOffset{0});
    if (n_cells <= 0) {
        return out;
    }

    std::unordered_map<ElementType, elements::ReferenceElement, EnumHash> ref_cache;
    auto get_ref = [&](ElementType type) -> const elements::ReferenceElement& {
        auto it = ref_cache.find(type);
        if (it != ref_cache.end()) return it->second;
        auto [inserted_it, did_insert] =
            ref_cache.emplace(type, elements::ReferenceElement::create(type));
        (void)did_insert;
        return inserted_it->second;
    };

    const auto first_cell = cell_vertices_span(cell2vertex_offsets, cell2vertex, 0);
    if (first_cell.empty()) {
        throw FEException("buildCellToEdgesRefOrder: missing cell2vertex connectivity for cell 0");
    }

    const auto base_type = infer_element_type_from_cell(dim, first_cell.size());
    if (base_type == ElementType::Unknown) {
        throw FEException("buildCellToEdgesRefOrder: unsupported cell type");
    }
    const auto& ref0 = get_ref(base_type);
    out.data.reserve(static_cast<std::size_t>(n_cells) * ref0.num_edges());

    // Build a deterministic lookup table for (min(v0,v1),max(v0,v1)) -> edge_id.
    std::vector<EdgeRecord> edges;
    edges.reserve(edge2vertex.size());
    for (MeshIndex e = 0; e < static_cast<MeshIndex>(edge2vertex.size()); ++e) {
        const auto v0 = edge2vertex[static_cast<std::size_t>(e)][0];
        const auto v1 = edge2vertex[static_cast<std::size_t>(e)][1];
        edges.push_back(EdgeRecord{std::min(v0, v1), std::max(v0, v1), e});
    }
    std::sort(edges.begin(), edges.end());

    auto find_edge = [&](MeshIndex a, MeshIndex b) -> MeshIndex {
        EdgeRecord key{a, b, 0};
        const auto it = std::lower_bound(edges.begin(), edges.end(), key);
        if (it == edges.end() || it->a != a || it->b != b) {
            return MeshIndex{-1};
        }
        return it->id;
    };

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        const auto cell_verts = cell_vertices_span(cell2vertex_offsets, cell2vertex, c);
        if (cell_verts.empty()) {
            throw FEException("buildCellToEdgesRefOrder: missing cell2vertex connectivity for cell");
        }
        const auto cell_type = infer_element_type_from_cell(dim, cell_verts.size());
        if (cell_type == ElementType::Unknown) {
            throw FEException("buildCellToEdgesRefOrder: unsupported cell type");
        }
        const auto& ref = get_ref(cell_type);

        out.offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(out.data.size());
        for (std::size_t le = 0; le < ref.num_edges(); ++le) {
            const auto& en = ref.edge_nodes(le);
            if (en.size() != 2u) {
                throw FEException("buildCellToEdgesRefOrder: unexpected edge node count");
            }
            const auto lv0 = static_cast<std::size_t>(en[0]);
            const auto lv1 = static_cast<std::size_t>(en[1]);
            if (lv0 >= cell_verts.size() || lv1 >= cell_verts.size()) {
                throw FEException("buildCellToEdgesRefOrder: reference edge node out of range");
            }
            const MeshIndex gv0 = cell_verts[lv0];
            const MeshIndex gv1 = cell_verts[lv1];
            const MeshIndex a = std::min(gv0, gv1);
            const MeshIndex b = std::max(gv0, gv1);
            const MeshIndex eid = find_edge(a, b);
            if (eid < 0) {
                throw FEException("buildCellToEdgesRefOrder: edge not found for cell");
            }
            out.data.push_back(eid);
        }
    }
    out.offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(out.data.size());

    return out;
}

CellToEntityCSR buildCellToFacesRefOrder(
    int dim,
    std::span<const MeshOffset> cell2vertex_offsets,
    std::span<const MeshIndex> cell2vertex,
    std::span<const MeshOffset> face2vertex_offsets,
    std::span<const MeshIndex> face2vertex) {

    const auto n_cells = static_cast<GlobalIndex>(
        cell2vertex_offsets.empty() ? 0 : static_cast<GlobalIndex>(cell2vertex_offsets.size() - 1));
    const auto n_faces = static_cast<GlobalIndex>(
        face2vertex_offsets.empty() ? 0 : static_cast<GlobalIndex>(face2vertex_offsets.size() - 1));

    CellToEntityCSR out;
    out.offsets.resize(static_cast<std::size_t>(n_cells) + 1u, MeshOffset{0});
    if (n_cells <= 0) {
        return out;
    }

    if (n_faces <= 0 || face2vertex_offsets.size() < 2u) {
        throw FEException("buildCellToFacesRefOrder: missing face2vertex connectivity");
    }

    std::unordered_map<ElementType, elements::ReferenceElement, EnumHash> ref_cache;
    auto get_ref = [&](ElementType type) -> const elements::ReferenceElement& {
        auto it = ref_cache.find(type);
        if (it != ref_cache.end()) return it->second;
        auto [inserted_it, did_insert] =
            ref_cache.emplace(type, elements::ReferenceElement::create(type));
        (void)did_insert;
        return inserted_it->second;
    };

    const auto first_cell = cell_vertices_span(cell2vertex_offsets, cell2vertex, 0);
    if (first_cell.empty()) {
        throw FEException("buildCellToFacesRefOrder: missing cell2vertex connectivity for cell 0");
    }

    const auto base_type = infer_element_type_from_cell(dim, first_cell.size());
    if (base_type == ElementType::Unknown) {
        throw FEException("buildCellToFacesRefOrder: unsupported cell type");
    }
    const auto& ref0 = get_ref(base_type);
    out.data.reserve(static_cast<std::size_t>(n_cells) * ref0.num_faces());

    std::vector<FaceRecord> faces;
    faces.reserve(static_cast<std::size_t>(n_faces));

    for (GlobalIndex f = 0; f < n_faces; ++f) {
        const auto fid = static_cast<std::size_t>(f);
        const auto begin = static_cast<std::size_t>(face2vertex_offsets[fid]);
        const auto end = static_cast<std::size_t>(face2vertex_offsets[fid + 1]);
        if (begin > end || end > face2vertex.size()) {
            throw FEException("buildCellToFacesRefOrder: invalid face2vertex offsets");
        }
        const std::size_t n = end - begin;
        if (n < 1u || n > 4u) {
            throw FEException("buildCellToFacesRefOrder: unsupported face vertex count");
        }

        FaceRecord rec{};
        rec.n = static_cast<std::uint8_t>(n);
        if (n == 1u) {
            rec.verts[0] = face2vertex[begin + 0];
            rec.verts[1] = MeshIndex{0};
            rec.verts[2] = MeshIndex{0};
            rec.verts[3] = MeshIndex{0};
        } else if (n == 2u) {
            rec.verts[0] = face2vertex[begin + 0];
            rec.verts[1] = face2vertex[begin + 1];
            if (rec.verts[0] > rec.verts[1]) {
                std::swap(rec.verts[0], rec.verts[1]);
            }
            rec.verts[2] = MeshIndex{0};
            rec.verts[3] = MeshIndex{0};
        } else if (n == 3u) {
            rec.verts[0] = face2vertex[begin + 0];
            rec.verts[1] = face2vertex[begin + 1];
            rec.verts[2] = face2vertex[begin + 2];
            std::sort(rec.verts.begin(), rec.verts.begin() + 3);
            rec.verts[3] = MeshIndex{0};
        } else {
            rec.verts[0] = face2vertex[begin + 0];
            rec.verts[1] = face2vertex[begin + 1];
            rec.verts[2] = face2vertex[begin + 2];
            rec.verts[3] = face2vertex[begin + 3];
            std::sort(rec.verts.begin(), rec.verts.begin() + 4);
        }
        rec.id = static_cast<MeshIndex>(f);
        faces.push_back(rec);
    }

    std::sort(faces.begin(), faces.end());

    auto find_face = [&](const FaceRecord& key) -> MeshIndex {
        const auto it = std::lower_bound(faces.begin(), faces.end(), key);
        if (it == faces.end() || it->n != key.n || it->verts != key.verts) {
            return MeshIndex{-1};
        }
        return it->id;
    };

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        const auto cell_verts = cell_vertices_span(cell2vertex_offsets, cell2vertex, c);
        if (cell_verts.empty()) {
            throw FEException("buildCellToFacesRefOrder: missing cell2vertex connectivity for cell");
        }
        const auto cell_type = infer_element_type_from_cell(dim, cell_verts.size());
        if (cell_type == ElementType::Unknown) {
            throw FEException("buildCellToFacesRefOrder: unsupported cell type");
        }
        const auto& ref = get_ref(cell_type);

        out.offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(out.data.size());
        for (std::size_t lf = 0; lf < ref.num_faces(); ++lf) {
            const auto& fn = ref.face_nodes(lf);
            if (fn.empty() || fn.size() > 4u) {
                throw FEException("buildCellToFacesRefOrder: unexpected face node count");
            }

            FaceRecord key{};
            const std::size_t fn_size = fn.size();
            key.n = static_cast<std::uint8_t>(fn_size);
            for (std::size_t i = 0; i < fn_size; ++i) {
                const auto lv = static_cast<std::size_t>(fn[i]);
                if (lv >= cell_verts.size()) {
                    throw FEException("buildCellToFacesRefOrder: reference face node out of range");
                }
                key.verts[i] = cell_verts[lv];
            }
            if (fn_size == 1u) {
                key.verts[1] = MeshIndex{0};
                key.verts[2] = MeshIndex{0};
                key.verts[3] = MeshIndex{0};
            } else if (fn_size == 2u) {
                if (key.verts[0] > key.verts[1]) {
                    std::swap(key.verts[0], key.verts[1]);
                }
                key.verts[2] = MeshIndex{0};
                key.verts[3] = MeshIndex{0};
            } else if (fn_size == 3u) {
                std::sort(key.verts.begin(), key.verts.begin() + 3);
                key.verts[3] = MeshIndex{0};
            } else {
                std::sort(key.verts.begin(), key.verts.begin() + 4);
            }

            const MeshIndex fid = find_face(key);
            if (fid < 0) {
                throw FEException("buildCellToFacesRefOrder: face not found for cell");
            }
            out.data.push_back(fid);
        }
    }
    out.offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(out.data.size());

    return out;
}

} // namespace dofs
} // namespace FE
} // namespace svmp
