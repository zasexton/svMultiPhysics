/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemConstraints.h"

#include "Core/FEException.h"
#include "Dofs/DofTools.h"
#include "Systems/SystemsExceptions.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {

namespace {

struct EdgeKey {
    GlobalIndex a;
    GlobalIndex b;
    bool operator==(const EdgeKey& other) const noexcept { return a == other.a && b == other.b; }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& k) const noexcept
    {
        const std::size_t h1 = std::hash<GlobalIndex>{}(k.a);
        const std::size_t h2 = std::hash<GlobalIndex>{}(k.b);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

std::vector<std::array<svmp::index_t, 2>> deriveEdgeTable2D(const svmp::MeshBase& base)
{
    const auto& offsets = base.face2vertex_offsets();
    const auto& data = base.face2vertex();
    std::vector<std::array<svmp::index_t, 2>> edges;
    edges.resize(base.n_faces());

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base.n_faces()); ++f) {
        const auto begin = static_cast<std::size_t>(offsets[static_cast<std::size_t>(f)]);
        const auto end = static_cast<std::size_t>(offsets[static_cast<std::size_t>(f + 1)]);
        FE_THROW_IF(end - begin != 2u, InvalidStateException,
                    "deriveEdgeTable2D: expected 2 vertices per face in 2D");
        edges[static_cast<std::size_t>(f)] = {data[begin], data[begin + 1]};
    }
    return edges;
}

std::vector<GlobalIndex> extractBoundaryDofsFromFaces(
    const dofs::EntityDofMap& entity_map,
    const svmp::MeshBase& base,
    std::span<const svmp::index_t> face_ids)
{
    std::unordered_set<GlobalIndex> seen;
    std::vector<GlobalIndex> result;

    auto push_span = [&](std::span<const GlobalIndex> dofs) {
        for (auto d : dofs) {
            if (seen.insert(d).second) {
                result.push_back(d);
            }
        }
    };

    const std::vector<std::array<svmp::index_t, 2>>* edge2vertex = &base.edge2vertex();
    std::vector<std::array<svmp::index_t, 2>> derived_edges;

    if (entity_map.numEdges() > 0 && edge2vertex->empty()) {
        FE_THROW_IF(base.dim() != 2, InvalidStateException,
                    "extractBoundaryDofsFromFaces: missing edge table for edge DOFs");
        derived_edges = deriveEdgeTable2D(base);
        edge2vertex = &derived_edges;
    }

    std::unordered_map<EdgeKey, GlobalIndex, EdgeKeyHash> edge_ids;
    if (entity_map.numEdges() > 0) {
        FE_THROW_IF(edge2vertex->size() < static_cast<std::size_t>(entity_map.numEdges()), InvalidStateException,
                    "extractBoundaryDofsFromFaces: edge table too small for EntityDofMap edge count");
        edge_ids.reserve(static_cast<std::size_t>(entity_map.numEdges()));
        for (GlobalIndex e = 0; e < entity_map.numEdges(); ++e) {
            const auto& ev = (*edge2vertex)[static_cast<std::size_t>(e)];
            EdgeKey key{std::min<GlobalIndex>(ev[0], ev[1]), std::max<GlobalIndex>(ev[0], ev[1])};
            edge_ids.emplace(key, e);
        }
    }

    for (auto fid_local : face_ids) {
        const auto fid = static_cast<GlobalIndex>(fid_local);
        auto verts_span = base.face_vertices_span(fid_local);
        const auto* verts = verts_span.first;
        const auto n_verts = verts_span.second;
        if (!verts || n_verts == 0u) {
            continue;
        }

        if (n_verts >= 3u) {
            FE_THROW_IF(entity_map.numFaces() <= 0 || fid >= entity_map.numFaces(), InvalidStateException,
                        "extractBoundaryDofsFromFaces: face DOFs requested but EntityDofMap has no matching faces");
            push_span(entity_map.getFaceDofs(fid));
        }

        for (std::size_t i = 0; i < n_verts; ++i) {
            push_span(entity_map.getVertexDofs(static_cast<GlobalIndex>(verts[i])));
        }

        if (!edge_ids.empty()) {
            auto add_edge = [&](GlobalIndex v0, GlobalIndex v1) {
                EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
                auto it = edge_ids.find(key);
                FE_THROW_IF(it == edge_ids.end(), InvalidStateException,
                            "extractBoundaryDofsFromFaces: facet edge not found in edge table");
                push_span(entity_map.getEdgeDofs(it->second));
            };

            if (n_verts == 2u) {
                add_edge(verts[0], verts[1]);
            } else {
                for (std::size_t i = 0; i < n_verts; ++i) {
                    const auto v0 = static_cast<GlobalIndex>(verts[i]);
                    const auto v1 = static_cast<GlobalIndex>(verts[(i + 1u) % n_verts]);
                    add_edge(v0, v1);
                }
            }
        }
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace

std::vector<GlobalIndex> boundaryDofsByMarker(const svmp::Mesh& mesh,
                                              const dofs::DofHandler& dof_handler,
                                              int boundary_marker)
{
    FE_THROW_IF(!dof_handler.isFinalized(), InvalidStateException,
                "boundaryDofsByMarker: DofHandler must be finalized");
    const auto* entity_map = dof_handler.getEntityDofMap();
    FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                "boundaryDofsByMarker: EntityDofMap is not available");

    const auto& base = mesh.local_mesh();
    const auto& face_labels = base.face_boundary_ids();

    std::vector<svmp::index_t> faces;
    faces.reserve(face_labels.size());
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(face_labels.size()); ++f) {
        if (static_cast<int>(face_labels[static_cast<std::size_t>(f)]) == boundary_marker) {
            faces.push_back(f);
        }
    }

    return extractBoundaryDofsFromFaces(
        *entity_map, base,
        std::span<const svmp::index_t>(faces.data(), faces.size()));
}

std::vector<GlobalIndex> boundaryDofsByFaceSet(const svmp::Mesh& mesh,
                                               const dofs::DofHandler& dof_handler,
                                               std::string_view face_set_name)
{
    FE_THROW_IF(!dof_handler.isFinalized(), InvalidStateException,
                "boundaryDofsByFaceSet: DofHandler must be finalized");
    const auto* entity_map = dof_handler.getEntityDofMap();
    FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                "boundaryDofsByFaceSet: EntityDofMap is not available");

    const auto& faces = mesh.get_set(svmp::EntityKind::Face, std::string(face_set_name));
    return extractBoundaryDofsFromFaces(
        *entity_map, mesh.local_mesh(),
        std::span<const svmp::index_t>(faces.data(), faces.size()));
}

std::unique_ptr<constraints::Constraint>
makeDirichletConstantByMarker(const svmp::Mesh& mesh,
                              const dofs::DofHandler& dof_handler,
                              int boundary_marker,
                              double value,
                              const constraints::DirichletBCOptions& opts)
{
    auto dofs = boundaryDofsByMarker(mesh, dof_handler, boundary_marker);
    return std::make_unique<constraints::DirichletBC>(std::move(dofs), value, opts);
}

std::unique_ptr<constraints::Constraint>
makeDirichletConstantByFaceSet(const svmp::Mesh& mesh,
                               const dofs::DofHandler& dof_handler,
                               std::string_view face_set_name,
                               double value,
                               const constraints::DirichletBCOptions& opts)
{
    auto dofs = boundaryDofsByFaceSet(mesh, dof_handler, face_set_name);
    return std::make_unique<constraints::DirichletBC>(std::move(dofs), value, opts);
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_WITH_MESH
